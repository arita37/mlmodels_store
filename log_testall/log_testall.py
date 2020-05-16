
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
From github.com:arita37/mlmodels_store
   ffd7251..0508032  master     -> origin/master
Updating ffd7251..0508032
Fast-forward
 ...-09_76b7a81be9b27c2e92c4951280c0a8da664b997c.py | 632 +++++++++++++++++++++
 1 file changed, 632 insertions(+)
 create mode 100644 log_pullrequest/log_pr_2020-05-16-16-09_76b7a81be9b27c2e92c4951280c0a8da664b997c.py
[master dbd28b0] ml_store
 2 files changed, 67 insertions(+), 5 deletions(-)
 create mode 100644 log_testall/log_testall.py
To github.com:arita37/mlmodels_store.git
   0508032..dbd28b0  master -> master





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
[master d6e674e] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   dbd28b0..d6e674e  master -> master





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
[master d11d9a4] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   d6e674e..d11d9a4  master -> master





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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-16 16:12:12.709023: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-16 16:12:12.714670: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-16 16:12:12.714903: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5621ee797d00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 16:12:12.714925: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 2ms/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2499 - val_binary_crossentropy: 0.6929

  #### metrics   #################################################### 
{'MSE': 0.24981983416445322}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_max[0][0]               
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
Total params: 408
Trainable params: 408
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2606 - binary_crossentropy: 0.7156500/500 [==============================] - 1s 2ms/sample - loss: 0.2594 - binary_crossentropy: 0.7129 - val_loss: 0.2558 - val_binary_crossentropy: 0.7055

  #### metrics   #################################################### 
{'MSE': 0.25731923361583947}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_max[0][0]               
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
Total params: 408
Trainable params: 408
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
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
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2526 - binary_crossentropy: 0.7512 - val_loss: 0.2489 - val_binary_crossentropy: 0.6905

  #### metrics   #################################################### 
{'MSE': 0.2505214087725283}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
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
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         20          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2809 - binary_crossentropy: 1.0179500/500 [==============================] - 1s 3ms/sample - loss: 0.2825 - binary_crossentropy: 1.2327 - val_loss: 0.2936 - val_binary_crossentropy: 1.2819

  #### metrics   #################################################### 
{'MSE': 0.2857531920715601}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         20          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 138
Trainable params: 138
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 3s - loss: 0.2990 - binary_crossentropy: 0.9422500/500 [==============================] - 2s 4ms/sample - loss: 0.3062 - binary_crossentropy: 0.8525 - val_loss: 0.2941 - val_binary_crossentropy: 0.7960

  #### metrics   #################################################### 
{'MSE': 0.29888570825120414}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 138
Trainable params: 138
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-16 16:13:42.423283: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:13:42.425636: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:13:42.431467: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-16 16:13:42.441935: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-16 16:13:42.443720: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 16:13:42.445347: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:13:42.446869: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2557 - val_binary_crossentropy: 0.7045
2020-05-16 16:13:43.779436: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:13:43.781212: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:13:43.785561: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-16 16:13:43.794583: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-16 16:13:43.796133: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 16:13:43.797588: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:13:43.798878: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25707608151685335}

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
2020-05-16 16:14:08.811291: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:14:08.812691: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:14:08.816361: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-16 16:14:08.823077: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-16 16:14:08.824316: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 16:14:08.825356: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:14:08.826283: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2516 - val_binary_crossentropy: 0.6964
2020-05-16 16:14:10.591793: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:14:10.592989: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:14:10.595680: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-16 16:14:10.601052: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-16 16:14:10.601988: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 16:14:10.602834: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:14:10.603597: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25203920357485227}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-16 16:14:47.149403: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:14:47.154729: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:14:47.170265: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-16 16:14:47.197674: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-16 16:14:47.202391: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 16:14:47.206805: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:14:47.211188: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.7227 - binary_crossentropy: 1.8981 - val_loss: 0.2532 - val_binary_crossentropy: 0.6997
2020-05-16 16:14:49.567310: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:14:49.572391: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:14:49.585250: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-16 16:14:49.622918: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-16 16:14:49.627488: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 16:14:49.631708: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 16:14:49.635980: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2717021653395433}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 730
Trainable params: 730
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2527 - binary_crossentropy: 0.6994500/500 [==============================] - 5s 9ms/sample - loss: 0.2760 - binary_crossentropy: 0.7485 - val_loss: 0.2856 - val_binary_crossentropy: 0.7685

  #### metrics   #################################################### 
{'MSE': 0.2798663785842203}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         14          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         10          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
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
Total params: 215
Trainable params: 215
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2593 - binary_crossentropy: 0.7141500/500 [==============================] - 5s 10ms/sample - loss: 0.2760 - binary_crossentropy: 0.7510 - val_loss: 0.2717 - val_binary_crossentropy: 0.7396

  #### metrics   #################################################### 
{'MSE': 0.26979702926235394}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         14          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         10          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
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
Total params: 215
Trainable params: 215
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 1,924
Trainable params: 1,924
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2595 - binary_crossentropy: 0.8446500/500 [==============================] - 5s 9ms/sample - loss: 0.2551 - binary_crossentropy: 0.7300 - val_loss: 0.2502 - val_binary_crossentropy: 0.6936

  #### metrics   #################################################### 
{'MSE': 0.25083794171514867}

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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 1,924
Trainable params: 1,924
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
regionsequence_mean (InputLayer [(None, 7)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 3)]          0                                            
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
region_10sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
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
Total params: 176
Trainable params: 176
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2564 - binary_crossentropy: 0.8361500/500 [==============================] - 6s 12ms/sample - loss: 0.2496 - binary_crossentropy: 0.7427 - val_loss: 0.2533 - val_binary_crossentropy: 0.7774

  #### metrics   #################################################### 
{'MSE': 0.2511578122787996}

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
regionsequence_mean (InputLayer [(None, 7)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 3)]          0                                            
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
region_10sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
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
Total params: 176
Trainable params: 176
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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_max[0][0]               
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
Total params: 1,417
Trainable params: 1,417
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2569 - binary_crossentropy: 0.8367500/500 [==============================] - 6s 12ms/sample - loss: 0.2613 - binary_crossentropy: 0.8734 - val_loss: 0.2535 - val_binary_crossentropy: 0.8298

  #### metrics   #################################################### 
{'MSE': 0.25611627052724956}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         24          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         24          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         24          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         24          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_11[0][0]                    
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
Total params: 3,307
Trainable params: 3,227
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2531 - binary_crossentropy: 0.7000500/500 [==============================] - 7s 14ms/sample - loss: 0.2538 - binary_crossentropy: 0.7007 - val_loss: 0.2472 - val_binary_crossentropy: 0.6869

  #### metrics   #################################################### 
{'MSE': 0.24962821328699142}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         24          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         24          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         24          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         24          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_11[0][0]                    
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
Total params: 3,307
Trainable params: 3,227
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
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         24          sequence_max[0][0]               
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
Total params: 461
Trainable params: 461
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2557 - binary_crossentropy: 0.7054500/500 [==============================] - 7s 13ms/sample - loss: 0.2513 - binary_crossentropy: 0.6960 - val_loss: 0.2506 - val_binary_crossentropy: 0.6944

  #### metrics   #################################################### 
{'MSE': 0.2504752438021332}

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
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         24          sequence_max[0][0]               
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
Total params: 461
Trainable params: 461
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 2,074
Trainable params: 2,074
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2975 - binary_crossentropy: 1.7088500/500 [==============================] - 8s 15ms/sample - loss: 0.2878 - binary_crossentropy: 1.6830 - val_loss: 0.2714 - val_binary_crossentropy: 1.3343

  #### metrics   #################################################### 
{'MSE': 0.27825280279722847}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 2,074
Trainable params: 2,074
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
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
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 9s - loss: 0.3164 - binary_crossentropy: 0.8654500/500 [==============================] - 7s 14ms/sample - loss: 0.3038 - binary_crossentropy: 0.8612 - val_loss: 0.3176 - val_binary_crossentropy: 0.9421

  #### metrics   #################################################### 
{'MSE': 0.3089257412835832}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
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
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
From github.com:arita37/mlmodels_store
   d11d9a4..3bd61b4  master     -> origin/master
Updating d11d9a4..3bd61b4
Fast-forward
 error_list/20200516/list_log_import_20200516.md      | 2 +-
 error_list/20200516/list_log_pullrequest_20200516.md | 2 +-
 2 files changed, 2 insertions(+), 2 deletions(-)
[master 77a8742] ml_store
 1 file changed, 5669 insertions(+)
To github.com:arita37/mlmodels_store.git
   3bd61b4..77a8742  master -> master





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
[master 1415a07] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   77a8742..1415a07  master -> master





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
[master ee460f1] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   1415a07..ee460f1  master -> master





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
[master 6465be9] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   ee460f1..6465be9  master -> master





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

2020-05-16 16:28:50.508008: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-16 16:28:50.513033: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-16 16:28:50.513189: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56484da9e340 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 16:28:50.513203: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 5s - loss: 1.3877
256/354 [====================>.........] - ETA: 2s - loss: 1.2943
354/354 [==============================] - 10s 27ms/step - loss: 1.2760 - val_loss: 1.9683

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
Already up to date.
[master 90f5d4b] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   6465be9..90f5d4b  master -> master





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
[master 359d3cb] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   90f5d4b..359d3cb  master -> master





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
[master 579401b] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   359d3cb..579401b  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2064384/17464789 [==>...........................] - ETA: 0s
 7127040/17464789 [===========>..................] - ETA: 0s
15261696/17464789 [=========================>....] - ETA: 0s
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
2020-05-16 16:29:47.460231: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-16 16:29:47.464689: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-16 16:29:47.464823: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d937209880 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 16:29:47.464837: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9886 - accuracy: 0.4790
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6283 - accuracy: 0.5025 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7586 - accuracy: 0.4940
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7701 - accuracy: 0.4933
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7832 - accuracy: 0.4924
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.8532 - accuracy: 0.4878
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7915 - accuracy: 0.4919
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.8008 - accuracy: 0.4913
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7348 - accuracy: 0.4956
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7448 - accuracy: 0.4949
11000/25000 [============>.................] - ETA: 3s - loss: 7.7614 - accuracy: 0.4938
12000/25000 [=============>................] - ETA: 3s - loss: 7.7433 - accuracy: 0.4950
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7079 - accuracy: 0.4973
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7006 - accuracy: 0.4978
15000/25000 [=================>............] - ETA: 2s - loss: 7.7024 - accuracy: 0.4977
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6896 - accuracy: 0.4985
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6838 - accuracy: 0.4989
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6888 - accuracy: 0.4986
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6981 - accuracy: 0.4979
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6958 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6900 - accuracy: 0.4985
22000/25000 [=========================>....] - ETA: 0s - loss: 7.7029 - accuracy: 0.4976
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6900 - accuracy: 0.4985
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6737 - accuracy: 0.4995
25000/25000 [==============================] - 7s 282us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7fbaf0fb6978>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7fbaf0fb6978> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.2680 - accuracy: 0.5260
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4366 - accuracy: 0.5150 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.4724 - accuracy: 0.5127
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.4980 - accuracy: 0.5110
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.4918 - accuracy: 0.5114
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5235 - accuracy: 0.5093
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5834 - accuracy: 0.5054
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6494 - accuracy: 0.5011
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6683 - accuracy: 0.4999
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6758 - accuracy: 0.4994
11000/25000 [============>.................] - ETA: 3s - loss: 7.6555 - accuracy: 0.5007
12000/25000 [=============>................] - ETA: 3s - loss: 7.7037 - accuracy: 0.4976
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6879 - accuracy: 0.4986
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6841 - accuracy: 0.4989
15000/25000 [=================>............] - ETA: 2s - loss: 7.6533 - accuracy: 0.5009
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6542 - accuracy: 0.5008
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6621 - accuracy: 0.5003
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6487 - accuracy: 0.5012
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6440 - accuracy: 0.5015
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6306 - accuracy: 0.5023
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6447 - accuracy: 0.5014
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6415 - accuracy: 0.5016
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6698 - accuracy: 0.4998
25000/25000 [==============================] - 8s 321us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8506 - accuracy: 0.4880
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7970 - accuracy: 0.4915 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7842 - accuracy: 0.4923
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7203 - accuracy: 0.4965
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6084 - accuracy: 0.5038
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5900 - accuracy: 0.5050
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6535 - accuracy: 0.5009
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6762 - accuracy: 0.4994
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6291 - accuracy: 0.5024
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6406 - accuracy: 0.5017
11000/25000 [============>.................] - ETA: 3s - loss: 7.6443 - accuracy: 0.5015
12000/25000 [=============>................] - ETA: 3s - loss: 7.6807 - accuracy: 0.4991
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6678 - accuracy: 0.4999
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6513 - accuracy: 0.5010
15000/25000 [=================>............] - ETA: 2s - loss: 7.6411 - accuracy: 0.5017
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6427 - accuracy: 0.5016
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6224 - accuracy: 0.5029
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6249 - accuracy: 0.5027
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6279 - accuracy: 0.5025
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6291 - accuracy: 0.5024
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6403 - accuracy: 0.5017
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6485 - accuracy: 0.5012
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6615 - accuracy: 0.5003
25000/25000 [==============================] - 7s 297us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   579401b..ea18b90  master     -> origin/master
Updating 579401b..ea18b90
Fast-forward
 error_list/20200516/list_log_benchmark_20200516.md |  182 +-
 error_list/20200516/list_log_import_20200516.md    |    2 +-
 error_list/20200516/list_log_jupyter_20200516.md   | 1749 ++++++++++----------
 .../20200516/list_log_pullrequest_20200516.md      |    2 +-
 error_list/20200516/list_log_test_cli_20200516.md  |  364 ++--
 error_list/20200516/list_log_testall_20200516.md   |  386 ++---
 6 files changed, 1317 insertions(+), 1368 deletions(-)
[master 8c85217] ml_store
 1 file changed, 329 insertions(+)
To github.com:arita37/mlmodels_store.git
   ea18b90..8c85217  master -> master





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

13/13 [==============================] - 2s 127ms/step - loss: nan
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
[master 699882b] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   8c85217..699882b  master -> master





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
 1556480/11490434 [===>..........................] - ETA: 0s
 4251648/11490434 [==========>...................] - ETA: 0s
10395648/11490434 [==========================>...] - ETA: 0s
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

   32/60000 [..............................] - ETA: 8:29 - loss: 2.3297 - categorical_accuracy: 0.0938
   64/60000 [..............................] - ETA: 5:18 - loss: 2.2912 - categorical_accuracy: 0.1250
   96/60000 [..............................] - ETA: 4:03 - loss: 2.2497 - categorical_accuracy: 0.1875
  128/60000 [..............................] - ETA: 3:28 - loss: 2.2090 - categorical_accuracy: 0.1953
  160/60000 [..............................] - ETA: 3:06 - loss: 2.2245 - categorical_accuracy: 0.1937
  192/60000 [..............................] - ETA: 2:51 - loss: 2.2269 - categorical_accuracy: 0.1875
  224/60000 [..............................] - ETA: 2:40 - loss: 2.2119 - categorical_accuracy: 0.1920
  256/60000 [..............................] - ETA: 2:35 - loss: 2.1783 - categorical_accuracy: 0.2109
  288/60000 [..............................] - ETA: 2:28 - loss: 2.1528 - categorical_accuracy: 0.2361
  352/60000 [..............................] - ETA: 2:19 - loss: 2.0894 - categorical_accuracy: 0.2557
  384/60000 [..............................] - ETA: 2:15 - loss: 2.0462 - categorical_accuracy: 0.2760
  416/60000 [..............................] - ETA: 2:12 - loss: 2.0090 - categorical_accuracy: 0.2909
  448/60000 [..............................] - ETA: 2:10 - loss: 1.9802 - categorical_accuracy: 0.3036
  480/60000 [..............................] - ETA: 2:08 - loss: 1.9500 - categorical_accuracy: 0.3146
  512/60000 [..............................] - ETA: 2:06 - loss: 1.9328 - categorical_accuracy: 0.3184
  544/60000 [..............................] - ETA: 2:04 - loss: 1.9095 - categorical_accuracy: 0.3346
  576/60000 [..............................] - ETA: 2:03 - loss: 1.8756 - categorical_accuracy: 0.3472
  608/60000 [..............................] - ETA: 2:02 - loss: 1.8359 - categorical_accuracy: 0.3618
  640/60000 [..............................] - ETA: 2:01 - loss: 1.8143 - categorical_accuracy: 0.3766
  672/60000 [..............................] - ETA: 2:00 - loss: 1.8002 - categorical_accuracy: 0.3810
  704/60000 [..............................] - ETA: 1:59 - loss: 1.7823 - categorical_accuracy: 0.3878
  736/60000 [..............................] - ETA: 1:58 - loss: 1.7628 - categorical_accuracy: 0.3940
  768/60000 [..............................] - ETA: 1:58 - loss: 1.7423 - categorical_accuracy: 0.4036
  800/60000 [..............................] - ETA: 1:57 - loss: 1.7174 - categorical_accuracy: 0.4162
  832/60000 [..............................] - ETA: 1:56 - loss: 1.6858 - categorical_accuracy: 0.4303
  864/60000 [..............................] - ETA: 1:56 - loss: 1.6728 - categorical_accuracy: 0.4363
  896/60000 [..............................] - ETA: 1:55 - loss: 1.6402 - categorical_accuracy: 0.4464
  928/60000 [..............................] - ETA: 1:54 - loss: 1.6286 - categorical_accuracy: 0.4504
  992/60000 [..............................] - ETA: 1:52 - loss: 1.5866 - categorical_accuracy: 0.4637
 1024/60000 [..............................] - ETA: 1:52 - loss: 1.5536 - categorical_accuracy: 0.4766
 1056/60000 [..............................] - ETA: 1:52 - loss: 1.5312 - categorical_accuracy: 0.4839
 1088/60000 [..............................] - ETA: 1:51 - loss: 1.5120 - categorical_accuracy: 0.4871
 1120/60000 [..............................] - ETA: 1:51 - loss: 1.4964 - categorical_accuracy: 0.4946
 1152/60000 [..............................] - ETA: 1:50 - loss: 1.4735 - categorical_accuracy: 0.5035
 1184/60000 [..............................] - ETA: 1:50 - loss: 1.4585 - categorical_accuracy: 0.5084
 1216/60000 [..............................] - ETA: 1:50 - loss: 1.4381 - categorical_accuracy: 0.5181
 1248/60000 [..............................] - ETA: 1:50 - loss: 1.4200 - categorical_accuracy: 0.5248
 1280/60000 [..............................] - ETA: 1:50 - loss: 1.4108 - categorical_accuracy: 0.5281
 1312/60000 [..............................] - ETA: 1:50 - loss: 1.3950 - categorical_accuracy: 0.5351
 1376/60000 [..............................] - ETA: 1:49 - loss: 1.3567 - categorical_accuracy: 0.5480
 1408/60000 [..............................] - ETA: 1:49 - loss: 1.3361 - categorical_accuracy: 0.5533
 1472/60000 [..............................] - ETA: 1:48 - loss: 1.3042 - categorical_accuracy: 0.5666
 1504/60000 [..............................] - ETA: 1:47 - loss: 1.2963 - categorical_accuracy: 0.5691
 1536/60000 [..............................] - ETA: 1:48 - loss: 1.2878 - categorical_accuracy: 0.5716
 1568/60000 [..............................] - ETA: 1:47 - loss: 1.2759 - categorical_accuracy: 0.5746
 1600/60000 [..............................] - ETA: 1:47 - loss: 1.2620 - categorical_accuracy: 0.5794
 1632/60000 [..............................] - ETA: 1:47 - loss: 1.2442 - categorical_accuracy: 0.5858
 1696/60000 [..............................] - ETA: 1:46 - loss: 1.2256 - categorical_accuracy: 0.5938
 1728/60000 [..............................] - ETA: 1:46 - loss: 1.2144 - categorical_accuracy: 0.5972
 1760/60000 [..............................] - ETA: 1:46 - loss: 1.2014 - categorical_accuracy: 0.6011
 1824/60000 [..............................] - ETA: 1:45 - loss: 1.1815 - categorical_accuracy: 0.6080
 1856/60000 [..............................] - ETA: 1:45 - loss: 1.1659 - categorical_accuracy: 0.6142
 1888/60000 [..............................] - ETA: 1:45 - loss: 1.1509 - categorical_accuracy: 0.6192
 1952/60000 [..............................] - ETA: 1:44 - loss: 1.1277 - categorical_accuracy: 0.6270
 2016/60000 [>.............................] - ETA: 1:44 - loss: 1.1125 - categorical_accuracy: 0.6324
 2080/60000 [>.............................] - ETA: 1:43 - loss: 1.0967 - categorical_accuracy: 0.6385
 2144/60000 [>.............................] - ETA: 1:43 - loss: 1.0833 - categorical_accuracy: 0.6418
 2176/60000 [>.............................] - ETA: 1:43 - loss: 1.0769 - categorical_accuracy: 0.6448
 2240/60000 [>.............................] - ETA: 1:42 - loss: 1.0624 - categorical_accuracy: 0.6496
 2272/60000 [>.............................] - ETA: 1:42 - loss: 1.0549 - categorical_accuracy: 0.6523
 2336/60000 [>.............................] - ETA: 1:41 - loss: 1.0355 - categorical_accuracy: 0.6580
 2400/60000 [>.............................] - ETA: 1:41 - loss: 1.0174 - categorical_accuracy: 0.6637
 2432/60000 [>.............................] - ETA: 1:41 - loss: 1.0218 - categorical_accuracy: 0.6645
 2464/60000 [>.............................] - ETA: 1:41 - loss: 1.0163 - categorical_accuracy: 0.6668
 2496/60000 [>.............................] - ETA: 1:41 - loss: 1.0130 - categorical_accuracy: 0.6679
 2528/60000 [>.............................] - ETA: 1:41 - loss: 1.0065 - categorical_accuracy: 0.6705
 2560/60000 [>.............................] - ETA: 1:41 - loss: 1.0009 - categorical_accuracy: 0.6730
 2624/60000 [>.............................] - ETA: 1:40 - loss: 0.9888 - categorical_accuracy: 0.6772
 2688/60000 [>.............................] - ETA: 1:40 - loss: 0.9769 - categorical_accuracy: 0.6812
 2752/60000 [>.............................] - ETA: 1:39 - loss: 0.9654 - categorical_accuracy: 0.6857
 2784/60000 [>.............................] - ETA: 1:39 - loss: 0.9588 - categorical_accuracy: 0.6875
 2816/60000 [>.............................] - ETA: 1:39 - loss: 0.9502 - categorical_accuracy: 0.6903
 2848/60000 [>.............................] - ETA: 1:39 - loss: 0.9422 - categorical_accuracy: 0.6928
 2912/60000 [>.............................] - ETA: 1:39 - loss: 0.9304 - categorical_accuracy: 0.6975
 2944/60000 [>.............................] - ETA: 1:39 - loss: 0.9239 - categorical_accuracy: 0.6994
 2976/60000 [>.............................] - ETA: 1:38 - loss: 0.9168 - categorical_accuracy: 0.7016
 3040/60000 [>.............................] - ETA: 1:38 - loss: 0.9023 - categorical_accuracy: 0.7069
 3072/60000 [>.............................] - ETA: 1:38 - loss: 0.8983 - categorical_accuracy: 0.7083
 3136/60000 [>.............................] - ETA: 1:38 - loss: 0.8893 - categorical_accuracy: 0.7121
 3168/60000 [>.............................] - ETA: 1:38 - loss: 0.8835 - categorical_accuracy: 0.7140
 3200/60000 [>.............................] - ETA: 1:38 - loss: 0.8783 - categorical_accuracy: 0.7159
 3232/60000 [>.............................] - ETA: 1:38 - loss: 0.8735 - categorical_accuracy: 0.7178
 3296/60000 [>.............................] - ETA: 1:38 - loss: 0.8690 - categorical_accuracy: 0.7191
 3328/60000 [>.............................] - ETA: 1:38 - loss: 0.8642 - categorical_accuracy: 0.7206
 3360/60000 [>.............................] - ETA: 1:38 - loss: 0.8598 - categorical_accuracy: 0.7220
 3424/60000 [>.............................] - ETA: 1:37 - loss: 0.8509 - categorical_accuracy: 0.7246
 3456/60000 [>.............................] - ETA: 1:37 - loss: 0.8459 - categorical_accuracy: 0.7263
 3488/60000 [>.............................] - ETA: 1:37 - loss: 0.8402 - categorical_accuracy: 0.7279
 3520/60000 [>.............................] - ETA: 1:37 - loss: 0.8351 - categorical_accuracy: 0.7298
 3552/60000 [>.............................] - ETA: 1:37 - loss: 0.8313 - categorical_accuracy: 0.7303
 3584/60000 [>.............................] - ETA: 1:37 - loss: 0.8255 - categorical_accuracy: 0.7321
 3616/60000 [>.............................] - ETA: 1:37 - loss: 0.8221 - categorical_accuracy: 0.7334
 3648/60000 [>.............................] - ETA: 1:37 - loss: 0.8160 - categorical_accuracy: 0.7355
 3680/60000 [>.............................] - ETA: 1:37 - loss: 0.8124 - categorical_accuracy: 0.7370
 3712/60000 [>.............................] - ETA: 1:37 - loss: 0.8083 - categorical_accuracy: 0.7381
 3744/60000 [>.............................] - ETA: 1:37 - loss: 0.8059 - categorical_accuracy: 0.7396
 3776/60000 [>.............................] - ETA: 1:37 - loss: 0.8002 - categorical_accuracy: 0.7415
 3808/60000 [>.............................] - ETA: 1:37 - loss: 0.7955 - categorical_accuracy: 0.7434
 3840/60000 [>.............................] - ETA: 1:36 - loss: 0.7930 - categorical_accuracy: 0.7437
 3872/60000 [>.............................] - ETA: 1:36 - loss: 0.7885 - categorical_accuracy: 0.7448
 3904/60000 [>.............................] - ETA: 1:36 - loss: 0.7847 - categorical_accuracy: 0.7456
 3936/60000 [>.............................] - ETA: 1:36 - loss: 0.7792 - categorical_accuracy: 0.7475
 3968/60000 [>.............................] - ETA: 1:36 - loss: 0.7761 - categorical_accuracy: 0.7485
 4000/60000 [=>............................] - ETA: 1:36 - loss: 0.7751 - categorical_accuracy: 0.7495
 4032/60000 [=>............................] - ETA: 1:36 - loss: 0.7706 - categorical_accuracy: 0.7510
 4064/60000 [=>............................] - ETA: 1:36 - loss: 0.7684 - categorical_accuracy: 0.7512
 4096/60000 [=>............................] - ETA: 1:36 - loss: 0.7658 - categorical_accuracy: 0.7522
 4128/60000 [=>............................] - ETA: 1:36 - loss: 0.7618 - categorical_accuracy: 0.7534
 4192/60000 [=>............................] - ETA: 1:36 - loss: 0.7560 - categorical_accuracy: 0.7550
 4224/60000 [=>............................] - ETA: 1:36 - loss: 0.7516 - categorical_accuracy: 0.7564
 4256/60000 [=>............................] - ETA: 1:35 - loss: 0.7469 - categorical_accuracy: 0.7582
 4288/60000 [=>............................] - ETA: 1:35 - loss: 0.7433 - categorical_accuracy: 0.7598
 4320/60000 [=>............................] - ETA: 1:35 - loss: 0.7403 - categorical_accuracy: 0.7606
 4352/60000 [=>............................] - ETA: 1:35 - loss: 0.7367 - categorical_accuracy: 0.7617
 4384/60000 [=>............................] - ETA: 1:35 - loss: 0.7334 - categorical_accuracy: 0.7625
 4416/60000 [=>............................] - ETA: 1:35 - loss: 0.7294 - categorical_accuracy: 0.7638
 4448/60000 [=>............................] - ETA: 1:35 - loss: 0.7255 - categorical_accuracy: 0.7648
 4480/60000 [=>............................] - ETA: 1:35 - loss: 0.7220 - categorical_accuracy: 0.7658
 4512/60000 [=>............................] - ETA: 1:35 - loss: 0.7195 - categorical_accuracy: 0.7668
 4576/60000 [=>............................] - ETA: 1:35 - loss: 0.7110 - categorical_accuracy: 0.7697
 4608/60000 [=>............................] - ETA: 1:35 - loss: 0.7097 - categorical_accuracy: 0.7700
 4640/60000 [=>............................] - ETA: 1:35 - loss: 0.7073 - categorical_accuracy: 0.7711
 4672/60000 [=>............................] - ETA: 1:35 - loss: 0.7045 - categorical_accuracy: 0.7723
 4704/60000 [=>............................] - ETA: 1:35 - loss: 0.7017 - categorical_accuracy: 0.7736
 4768/60000 [=>............................] - ETA: 1:35 - loss: 0.6942 - categorical_accuracy: 0.7762
 4800/60000 [=>............................] - ETA: 1:35 - loss: 0.6919 - categorical_accuracy: 0.7771
 4832/60000 [=>............................] - ETA: 1:35 - loss: 0.6903 - categorical_accuracy: 0.7781
 4864/60000 [=>............................] - ETA: 1:35 - loss: 0.6879 - categorical_accuracy: 0.7786
 4896/60000 [=>............................] - ETA: 1:35 - loss: 0.6852 - categorical_accuracy: 0.7796
 4928/60000 [=>............................] - ETA: 1:35 - loss: 0.6834 - categorical_accuracy: 0.7806
 4960/60000 [=>............................] - ETA: 1:34 - loss: 0.6802 - categorical_accuracy: 0.7817
 4992/60000 [=>............................] - ETA: 1:34 - loss: 0.6769 - categorical_accuracy: 0.7831
 5024/60000 [=>............................] - ETA: 1:34 - loss: 0.6748 - categorical_accuracy: 0.7832
 5056/60000 [=>............................] - ETA: 1:34 - loss: 0.6715 - categorical_accuracy: 0.7840
 5088/60000 [=>............................] - ETA: 1:34 - loss: 0.6693 - categorical_accuracy: 0.7846
 5152/60000 [=>............................] - ETA: 1:34 - loss: 0.6681 - categorical_accuracy: 0.7847
 5184/60000 [=>............................] - ETA: 1:34 - loss: 0.6655 - categorical_accuracy: 0.7857
 5248/60000 [=>............................] - ETA: 1:34 - loss: 0.6625 - categorical_accuracy: 0.7868
 5312/60000 [=>............................] - ETA: 1:34 - loss: 0.6583 - categorical_accuracy: 0.7880
 5344/60000 [=>............................] - ETA: 1:34 - loss: 0.6563 - categorical_accuracy: 0.7887
 5376/60000 [=>............................] - ETA: 1:34 - loss: 0.6540 - categorical_accuracy: 0.7896
 5408/60000 [=>............................] - ETA: 1:33 - loss: 0.6537 - categorical_accuracy: 0.7901
 5472/60000 [=>............................] - ETA: 1:33 - loss: 0.6499 - categorical_accuracy: 0.7915
 5504/60000 [=>............................] - ETA: 1:33 - loss: 0.6484 - categorical_accuracy: 0.7918
 5536/60000 [=>............................] - ETA: 1:33 - loss: 0.6477 - categorical_accuracy: 0.7921
 5568/60000 [=>............................] - ETA: 1:33 - loss: 0.6467 - categorical_accuracy: 0.7926
 5600/60000 [=>............................] - ETA: 1:33 - loss: 0.6451 - categorical_accuracy: 0.7934
 5632/60000 [=>............................] - ETA: 1:33 - loss: 0.6427 - categorical_accuracy: 0.7939
 5664/60000 [=>............................] - ETA: 1:33 - loss: 0.6409 - categorical_accuracy: 0.7941
 5696/60000 [=>............................] - ETA: 1:33 - loss: 0.6394 - categorical_accuracy: 0.7948
 5728/60000 [=>............................] - ETA: 1:33 - loss: 0.6363 - categorical_accuracy: 0.7959
 5760/60000 [=>............................] - ETA: 1:33 - loss: 0.6336 - categorical_accuracy: 0.7967
 5824/60000 [=>............................] - ETA: 1:33 - loss: 0.6301 - categorical_accuracy: 0.7979
 5856/60000 [=>............................] - ETA: 1:33 - loss: 0.6275 - categorical_accuracy: 0.7988
 5888/60000 [=>............................] - ETA: 1:32 - loss: 0.6252 - categorical_accuracy: 0.7998
 5920/60000 [=>............................] - ETA: 1:32 - loss: 0.6228 - categorical_accuracy: 0.8005
 5952/60000 [=>............................] - ETA: 1:32 - loss: 0.6212 - categorical_accuracy: 0.8011
 5984/60000 [=>............................] - ETA: 1:32 - loss: 0.6201 - categorical_accuracy: 0.8015
 6016/60000 [==>...........................] - ETA: 1:32 - loss: 0.6175 - categorical_accuracy: 0.8024
 6048/60000 [==>...........................] - ETA: 1:32 - loss: 0.6180 - categorical_accuracy: 0.8021
 6080/60000 [==>...........................] - ETA: 1:32 - loss: 0.6165 - categorical_accuracy: 0.8025
 6144/60000 [==>...........................] - ETA: 1:32 - loss: 0.6133 - categorical_accuracy: 0.8035
 6176/60000 [==>...........................] - ETA: 1:32 - loss: 0.6110 - categorical_accuracy: 0.8042
 6208/60000 [==>...........................] - ETA: 1:32 - loss: 0.6103 - categorical_accuracy: 0.8046
 6240/60000 [==>...........................] - ETA: 1:32 - loss: 0.6092 - categorical_accuracy: 0.8053
 6272/60000 [==>...........................] - ETA: 1:32 - loss: 0.6072 - categorical_accuracy: 0.8060
 6304/60000 [==>...........................] - ETA: 1:32 - loss: 0.6051 - categorical_accuracy: 0.8066
 6336/60000 [==>...........................] - ETA: 1:32 - loss: 0.6026 - categorical_accuracy: 0.8074
 6368/60000 [==>...........................] - ETA: 1:32 - loss: 0.6013 - categorical_accuracy: 0.8078
 6400/60000 [==>...........................] - ETA: 1:32 - loss: 0.5992 - categorical_accuracy: 0.8083
 6432/60000 [==>...........................] - ETA: 1:32 - loss: 0.5967 - categorical_accuracy: 0.8092
 6464/60000 [==>...........................] - ETA: 1:32 - loss: 0.5945 - categorical_accuracy: 0.8099
 6496/60000 [==>...........................] - ETA: 1:32 - loss: 0.5928 - categorical_accuracy: 0.8103
 6528/60000 [==>...........................] - ETA: 1:32 - loss: 0.5911 - categorical_accuracy: 0.8110
 6560/60000 [==>...........................] - ETA: 1:31 - loss: 0.5893 - categorical_accuracy: 0.8113
 6624/60000 [==>...........................] - ETA: 1:31 - loss: 0.5876 - categorical_accuracy: 0.8120
 6656/60000 [==>...........................] - ETA: 1:31 - loss: 0.5862 - categorical_accuracy: 0.8122
 6720/60000 [==>...........................] - ETA: 1:31 - loss: 0.5836 - categorical_accuracy: 0.8128
 6752/60000 [==>...........................] - ETA: 1:31 - loss: 0.5815 - categorical_accuracy: 0.8134
 6784/60000 [==>...........................] - ETA: 1:31 - loss: 0.5816 - categorical_accuracy: 0.8137
 6816/60000 [==>...........................] - ETA: 1:31 - loss: 0.5803 - categorical_accuracy: 0.8143
 6848/60000 [==>...........................] - ETA: 1:31 - loss: 0.5791 - categorical_accuracy: 0.8148
 6912/60000 [==>...........................] - ETA: 1:31 - loss: 0.5761 - categorical_accuracy: 0.8154
 6944/60000 [==>...........................] - ETA: 1:31 - loss: 0.5759 - categorical_accuracy: 0.8157
 6976/60000 [==>...........................] - ETA: 1:31 - loss: 0.5747 - categorical_accuracy: 0.8159
 7008/60000 [==>...........................] - ETA: 1:31 - loss: 0.5733 - categorical_accuracy: 0.8165
 7040/60000 [==>...........................] - ETA: 1:30 - loss: 0.5722 - categorical_accuracy: 0.8166
 7104/60000 [==>...........................] - ETA: 1:30 - loss: 0.5695 - categorical_accuracy: 0.8177
 7136/60000 [==>...........................] - ETA: 1:30 - loss: 0.5684 - categorical_accuracy: 0.8181
 7168/60000 [==>...........................] - ETA: 1:30 - loss: 0.5674 - categorical_accuracy: 0.8186
 7200/60000 [==>...........................] - ETA: 1:30 - loss: 0.5658 - categorical_accuracy: 0.8192
 7264/60000 [==>...........................] - ETA: 1:30 - loss: 0.5627 - categorical_accuracy: 0.8203
 7296/60000 [==>...........................] - ETA: 1:30 - loss: 0.5608 - categorical_accuracy: 0.8209
 7360/60000 [==>...........................] - ETA: 1:30 - loss: 0.5575 - categorical_accuracy: 0.8219
 7392/60000 [==>...........................] - ETA: 1:30 - loss: 0.5561 - categorical_accuracy: 0.8222
 7456/60000 [==>...........................] - ETA: 1:30 - loss: 0.5545 - categorical_accuracy: 0.8230
 7488/60000 [==>...........................] - ETA: 1:30 - loss: 0.5538 - categorical_accuracy: 0.8231
 7520/60000 [==>...........................] - ETA: 1:30 - loss: 0.5526 - categorical_accuracy: 0.8233
 7552/60000 [==>...........................] - ETA: 1:29 - loss: 0.5518 - categorical_accuracy: 0.8236
 7616/60000 [==>...........................] - ETA: 1:29 - loss: 0.5518 - categorical_accuracy: 0.8238
 7680/60000 [==>...........................] - ETA: 1:29 - loss: 0.5488 - categorical_accuracy: 0.8249
 7744/60000 [==>...........................] - ETA: 1:29 - loss: 0.5453 - categorical_accuracy: 0.8262
 7776/60000 [==>...........................] - ETA: 1:29 - loss: 0.5443 - categorical_accuracy: 0.8264
 7840/60000 [==>...........................] - ETA: 1:29 - loss: 0.5415 - categorical_accuracy: 0.8272
 7872/60000 [==>...........................] - ETA: 1:29 - loss: 0.5396 - categorical_accuracy: 0.8279
 7904/60000 [==>...........................] - ETA: 1:29 - loss: 0.5394 - categorical_accuracy: 0.8279
 7968/60000 [==>...........................] - ETA: 1:28 - loss: 0.5377 - categorical_accuracy: 0.8287
 8000/60000 [===>..........................] - ETA: 1:28 - loss: 0.5361 - categorical_accuracy: 0.8292
 8064/60000 [===>..........................] - ETA: 1:28 - loss: 0.5335 - categorical_accuracy: 0.8301
 8096/60000 [===>..........................] - ETA: 1:28 - loss: 0.5320 - categorical_accuracy: 0.8304
 8160/60000 [===>..........................] - ETA: 1:28 - loss: 0.5296 - categorical_accuracy: 0.8309
 8192/60000 [===>..........................] - ETA: 1:28 - loss: 0.5279 - categorical_accuracy: 0.8314
 8224/60000 [===>..........................] - ETA: 1:28 - loss: 0.5269 - categorical_accuracy: 0.8317
 8288/60000 [===>..........................] - ETA: 1:28 - loss: 0.5247 - categorical_accuracy: 0.8322
 8352/60000 [===>..........................] - ETA: 1:27 - loss: 0.5221 - categorical_accuracy: 0.8327
 8384/60000 [===>..........................] - ETA: 1:27 - loss: 0.5215 - categorical_accuracy: 0.8329
 8416/60000 [===>..........................] - ETA: 1:27 - loss: 0.5198 - categorical_accuracy: 0.8335
 8448/60000 [===>..........................] - ETA: 1:27 - loss: 0.5196 - categorical_accuracy: 0.8337
 8512/60000 [===>..........................] - ETA: 1:27 - loss: 0.5176 - categorical_accuracy: 0.8342
 8576/60000 [===>..........................] - ETA: 1:27 - loss: 0.5143 - categorical_accuracy: 0.8354
 8608/60000 [===>..........................] - ETA: 1:27 - loss: 0.5142 - categorical_accuracy: 0.8355
 8640/60000 [===>..........................] - ETA: 1:27 - loss: 0.5130 - categorical_accuracy: 0.8360
 8704/60000 [===>..........................] - ETA: 1:27 - loss: 0.5117 - categorical_accuracy: 0.8364
 8736/60000 [===>..........................] - ETA: 1:27 - loss: 0.5102 - categorical_accuracy: 0.8369
 8768/60000 [===>..........................] - ETA: 1:27 - loss: 0.5094 - categorical_accuracy: 0.8371
 8800/60000 [===>..........................] - ETA: 1:27 - loss: 0.5079 - categorical_accuracy: 0.8376
 8832/60000 [===>..........................] - ETA: 1:26 - loss: 0.5080 - categorical_accuracy: 0.8377
 8896/60000 [===>..........................] - ETA: 1:26 - loss: 0.5061 - categorical_accuracy: 0.8385
 8928/60000 [===>..........................] - ETA: 1:26 - loss: 0.5052 - categorical_accuracy: 0.8385
 8960/60000 [===>..........................] - ETA: 1:26 - loss: 0.5052 - categorical_accuracy: 0.8386
 8992/60000 [===>..........................] - ETA: 1:26 - loss: 0.5052 - categorical_accuracy: 0.8387
 9024/60000 [===>..........................] - ETA: 1:26 - loss: 0.5036 - categorical_accuracy: 0.8392
 9056/60000 [===>..........................] - ETA: 1:26 - loss: 0.5023 - categorical_accuracy: 0.8396
 9088/60000 [===>..........................] - ETA: 1:26 - loss: 0.5025 - categorical_accuracy: 0.8397
 9120/60000 [===>..........................] - ETA: 1:26 - loss: 0.5015 - categorical_accuracy: 0.8399
 9152/60000 [===>..........................] - ETA: 1:26 - loss: 0.5001 - categorical_accuracy: 0.8404
 9216/60000 [===>..........................] - ETA: 1:26 - loss: 0.4981 - categorical_accuracy: 0.8410
 9248/60000 [===>..........................] - ETA: 1:26 - loss: 0.4967 - categorical_accuracy: 0.8415
 9280/60000 [===>..........................] - ETA: 1:25 - loss: 0.4958 - categorical_accuracy: 0.8419
 9344/60000 [===>..........................] - ETA: 1:25 - loss: 0.4938 - categorical_accuracy: 0.8424
 9408/60000 [===>..........................] - ETA: 1:25 - loss: 0.4925 - categorical_accuracy: 0.8427
 9472/60000 [===>..........................] - ETA: 1:25 - loss: 0.4908 - categorical_accuracy: 0.8433
 9536/60000 [===>..........................] - ETA: 1:25 - loss: 0.4887 - categorical_accuracy: 0.8440
 9568/60000 [===>..........................] - ETA: 1:25 - loss: 0.4880 - categorical_accuracy: 0.8441
 9600/60000 [===>..........................] - ETA: 1:25 - loss: 0.4868 - categorical_accuracy: 0.8444
 9632/60000 [===>..........................] - ETA: 1:25 - loss: 0.4863 - categorical_accuracy: 0.8445
 9664/60000 [===>..........................] - ETA: 1:25 - loss: 0.4855 - categorical_accuracy: 0.8447
 9728/60000 [===>..........................] - ETA: 1:24 - loss: 0.4847 - categorical_accuracy: 0.8448
 9760/60000 [===>..........................] - ETA: 1:24 - loss: 0.4842 - categorical_accuracy: 0.8449
 9792/60000 [===>..........................] - ETA: 1:24 - loss: 0.4827 - categorical_accuracy: 0.8454
 9824/60000 [===>..........................] - ETA: 1:24 - loss: 0.4820 - categorical_accuracy: 0.8458
 9888/60000 [===>..........................] - ETA: 1:24 - loss: 0.4801 - categorical_accuracy: 0.8463
 9920/60000 [===>..........................] - ETA: 1:24 - loss: 0.4789 - categorical_accuracy: 0.8467
 9952/60000 [===>..........................] - ETA: 1:24 - loss: 0.4792 - categorical_accuracy: 0.8466
10016/60000 [====>.........................] - ETA: 1:24 - loss: 0.4786 - categorical_accuracy: 0.8467
10080/60000 [====>.........................] - ETA: 1:24 - loss: 0.4765 - categorical_accuracy: 0.8475
10144/60000 [====>.........................] - ETA: 1:23 - loss: 0.4746 - categorical_accuracy: 0.8482
10208/60000 [====>.........................] - ETA: 1:23 - loss: 0.4725 - categorical_accuracy: 0.8489
10272/60000 [====>.........................] - ETA: 1:23 - loss: 0.4708 - categorical_accuracy: 0.8496
10336/60000 [====>.........................] - ETA: 1:23 - loss: 0.4687 - categorical_accuracy: 0.8502
10400/60000 [====>.........................] - ETA: 1:23 - loss: 0.4676 - categorical_accuracy: 0.8505
10464/60000 [====>.........................] - ETA: 1:23 - loss: 0.4654 - categorical_accuracy: 0.8513
10496/60000 [====>.........................] - ETA: 1:23 - loss: 0.4646 - categorical_accuracy: 0.8517
10528/60000 [====>.........................] - ETA: 1:23 - loss: 0.4646 - categorical_accuracy: 0.8516
10592/60000 [====>.........................] - ETA: 1:22 - loss: 0.4628 - categorical_accuracy: 0.8522
10656/60000 [====>.........................] - ETA: 1:22 - loss: 0.4613 - categorical_accuracy: 0.8526
10720/60000 [====>.........................] - ETA: 1:22 - loss: 0.4590 - categorical_accuracy: 0.8533
10784/60000 [====>.........................] - ETA: 1:22 - loss: 0.4573 - categorical_accuracy: 0.8537
10848/60000 [====>.........................] - ETA: 1:22 - loss: 0.4554 - categorical_accuracy: 0.8544
10912/60000 [====>.........................] - ETA: 1:22 - loss: 0.4542 - categorical_accuracy: 0.8548
10944/60000 [====>.........................] - ETA: 1:22 - loss: 0.4535 - categorical_accuracy: 0.8549
10976/60000 [====>.........................] - ETA: 1:22 - loss: 0.4523 - categorical_accuracy: 0.8553
11008/60000 [====>.........................] - ETA: 1:22 - loss: 0.4520 - categorical_accuracy: 0.8555
11072/60000 [====>.........................] - ETA: 1:21 - loss: 0.4515 - categorical_accuracy: 0.8558
11104/60000 [====>.........................] - ETA: 1:21 - loss: 0.4507 - categorical_accuracy: 0.8561
11136/60000 [====>.........................] - ETA: 1:21 - loss: 0.4508 - categorical_accuracy: 0.8561
11168/60000 [====>.........................] - ETA: 1:21 - loss: 0.4506 - categorical_accuracy: 0.8561
11232/60000 [====>.........................] - ETA: 1:21 - loss: 0.4490 - categorical_accuracy: 0.8567
11296/60000 [====>.........................] - ETA: 1:21 - loss: 0.4471 - categorical_accuracy: 0.8573
11360/60000 [====>.........................] - ETA: 1:21 - loss: 0.4464 - categorical_accuracy: 0.8574
11392/60000 [====>.........................] - ETA: 1:21 - loss: 0.4453 - categorical_accuracy: 0.8578
11424/60000 [====>.........................] - ETA: 1:21 - loss: 0.4443 - categorical_accuracy: 0.8581
11456/60000 [====>.........................] - ETA: 1:21 - loss: 0.4442 - categorical_accuracy: 0.8582
11520/60000 [====>.........................] - ETA: 1:20 - loss: 0.4433 - categorical_accuracy: 0.8586
11584/60000 [====>.........................] - ETA: 1:20 - loss: 0.4417 - categorical_accuracy: 0.8590
11648/60000 [====>.........................] - ETA: 1:20 - loss: 0.4408 - categorical_accuracy: 0.8595
11712/60000 [====>.........................] - ETA: 1:20 - loss: 0.4395 - categorical_accuracy: 0.8601
11776/60000 [====>.........................] - ETA: 1:20 - loss: 0.4378 - categorical_accuracy: 0.8607
11840/60000 [====>.........................] - ETA: 1:20 - loss: 0.4369 - categorical_accuracy: 0.8612
11904/60000 [====>.........................] - ETA: 1:20 - loss: 0.4349 - categorical_accuracy: 0.8619
11936/60000 [====>.........................] - ETA: 1:20 - loss: 0.4355 - categorical_accuracy: 0.8618
12000/60000 [=====>........................] - ETA: 1:19 - loss: 0.4348 - categorical_accuracy: 0.8622
12064/60000 [=====>........................] - ETA: 1:19 - loss: 0.4338 - categorical_accuracy: 0.8626
12128/60000 [=====>........................] - ETA: 1:19 - loss: 0.4330 - categorical_accuracy: 0.8629
12192/60000 [=====>........................] - ETA: 1:19 - loss: 0.4314 - categorical_accuracy: 0.8635
12256/60000 [=====>........................] - ETA: 1:19 - loss: 0.4309 - categorical_accuracy: 0.8638
12320/60000 [=====>........................] - ETA: 1:19 - loss: 0.4306 - categorical_accuracy: 0.8639
12384/60000 [=====>........................] - ETA: 1:18 - loss: 0.4296 - categorical_accuracy: 0.8640
12416/60000 [=====>........................] - ETA: 1:18 - loss: 0.4289 - categorical_accuracy: 0.8642
12448/60000 [=====>........................] - ETA: 1:18 - loss: 0.4280 - categorical_accuracy: 0.8646
12512/60000 [=====>........................] - ETA: 1:18 - loss: 0.4266 - categorical_accuracy: 0.8650
12576/60000 [=====>........................] - ETA: 1:18 - loss: 0.4257 - categorical_accuracy: 0.8654
12640/60000 [=====>........................] - ETA: 1:18 - loss: 0.4239 - categorical_accuracy: 0.8660
12672/60000 [=====>........................] - ETA: 1:18 - loss: 0.4231 - categorical_accuracy: 0.8662
12736/60000 [=====>........................] - ETA: 1:18 - loss: 0.4224 - categorical_accuracy: 0.8664
12800/60000 [=====>........................] - ETA: 1:17 - loss: 0.4209 - categorical_accuracy: 0.8668
12864/60000 [=====>........................] - ETA: 1:17 - loss: 0.4196 - categorical_accuracy: 0.8670
12928/60000 [=====>........................] - ETA: 1:17 - loss: 0.4182 - categorical_accuracy: 0.8674
12992/60000 [=====>........................] - ETA: 1:17 - loss: 0.4173 - categorical_accuracy: 0.8678
13056/60000 [=====>........................] - ETA: 1:17 - loss: 0.4176 - categorical_accuracy: 0.8680
13120/60000 [=====>........................] - ETA: 1:17 - loss: 0.4175 - categorical_accuracy: 0.8682
13152/60000 [=====>........................] - ETA: 1:17 - loss: 0.4168 - categorical_accuracy: 0.8684
13184/60000 [=====>........................] - ETA: 1:17 - loss: 0.4161 - categorical_accuracy: 0.8686
13248/60000 [=====>........................] - ETA: 1:16 - loss: 0.4148 - categorical_accuracy: 0.8690
13312/60000 [=====>........................] - ETA: 1:16 - loss: 0.4137 - categorical_accuracy: 0.8694
13376/60000 [=====>........................] - ETA: 1:16 - loss: 0.4127 - categorical_accuracy: 0.8698
13440/60000 [=====>........................] - ETA: 1:16 - loss: 0.4116 - categorical_accuracy: 0.8700
13504/60000 [=====>........................] - ETA: 1:16 - loss: 0.4108 - categorical_accuracy: 0.8703
13568/60000 [=====>........................] - ETA: 1:16 - loss: 0.4104 - categorical_accuracy: 0.8702
13632/60000 [=====>........................] - ETA: 1:16 - loss: 0.4091 - categorical_accuracy: 0.8707
13696/60000 [=====>........................] - ETA: 1:15 - loss: 0.4093 - categorical_accuracy: 0.8708
13760/60000 [=====>........................] - ETA: 1:15 - loss: 0.4088 - categorical_accuracy: 0.8709
13824/60000 [=====>........................] - ETA: 1:15 - loss: 0.4078 - categorical_accuracy: 0.8712
13888/60000 [=====>........................] - ETA: 1:15 - loss: 0.4068 - categorical_accuracy: 0.8714
13952/60000 [=====>........................] - ETA: 1:15 - loss: 0.4058 - categorical_accuracy: 0.8718
14016/60000 [======>.......................] - ETA: 1:15 - loss: 0.4047 - categorical_accuracy: 0.8721
14080/60000 [======>.......................] - ETA: 1:14 - loss: 0.4034 - categorical_accuracy: 0.8724
14144/60000 [======>.......................] - ETA: 1:14 - loss: 0.4025 - categorical_accuracy: 0.8727
14208/60000 [======>.......................] - ETA: 1:14 - loss: 0.4009 - categorical_accuracy: 0.8732
14272/60000 [======>.......................] - ETA: 1:14 - loss: 0.4000 - categorical_accuracy: 0.8736
14336/60000 [======>.......................] - ETA: 1:14 - loss: 0.3986 - categorical_accuracy: 0.8741
14368/60000 [======>.......................] - ETA: 1:14 - loss: 0.3981 - categorical_accuracy: 0.8742
14432/60000 [======>.......................] - ETA: 1:14 - loss: 0.3973 - categorical_accuracy: 0.8746
14496/60000 [======>.......................] - ETA: 1:14 - loss: 0.3962 - categorical_accuracy: 0.8749
14560/60000 [======>.......................] - ETA: 1:13 - loss: 0.3955 - categorical_accuracy: 0.8752
14624/60000 [======>.......................] - ETA: 1:13 - loss: 0.3939 - categorical_accuracy: 0.8758
14656/60000 [======>.......................] - ETA: 1:13 - loss: 0.3933 - categorical_accuracy: 0.8760
14720/60000 [======>.......................] - ETA: 1:13 - loss: 0.3920 - categorical_accuracy: 0.8764
14784/60000 [======>.......................] - ETA: 1:13 - loss: 0.3910 - categorical_accuracy: 0.8766
14848/60000 [======>.......................] - ETA: 1:13 - loss: 0.3903 - categorical_accuracy: 0.8769
14880/60000 [======>.......................] - ETA: 1:13 - loss: 0.3899 - categorical_accuracy: 0.8770
14912/60000 [======>.......................] - ETA: 1:13 - loss: 0.3892 - categorical_accuracy: 0.8772
14944/60000 [======>.......................] - ETA: 1:13 - loss: 0.3886 - categorical_accuracy: 0.8774
15008/60000 [======>.......................] - ETA: 1:13 - loss: 0.3877 - categorical_accuracy: 0.8777
15072/60000 [======>.......................] - ETA: 1:12 - loss: 0.3870 - categorical_accuracy: 0.8780
15104/60000 [======>.......................] - ETA: 1:12 - loss: 0.3865 - categorical_accuracy: 0.8781
15168/60000 [======>.......................] - ETA: 1:12 - loss: 0.3851 - categorical_accuracy: 0.8786
15232/60000 [======>.......................] - ETA: 1:12 - loss: 0.3841 - categorical_accuracy: 0.8787
15296/60000 [======>.......................] - ETA: 1:12 - loss: 0.3830 - categorical_accuracy: 0.8791
15360/60000 [======>.......................] - ETA: 1:12 - loss: 0.3819 - categorical_accuracy: 0.8794
15424/60000 [======>.......................] - ETA: 1:12 - loss: 0.3808 - categorical_accuracy: 0.8799
15456/60000 [======>.......................] - ETA: 1:12 - loss: 0.3802 - categorical_accuracy: 0.8800
15520/60000 [======>.......................] - ETA: 1:12 - loss: 0.3789 - categorical_accuracy: 0.8803
15584/60000 [======>.......................] - ETA: 1:11 - loss: 0.3780 - categorical_accuracy: 0.8807
15648/60000 [======>.......................] - ETA: 1:11 - loss: 0.3768 - categorical_accuracy: 0.8810
15712/60000 [======>.......................] - ETA: 1:11 - loss: 0.3764 - categorical_accuracy: 0.8812
15744/60000 [======>.......................] - ETA: 1:11 - loss: 0.3759 - categorical_accuracy: 0.8814
15808/60000 [======>.......................] - ETA: 1:11 - loss: 0.3751 - categorical_accuracy: 0.8816
15872/60000 [======>.......................] - ETA: 1:11 - loss: 0.3746 - categorical_accuracy: 0.8817
15936/60000 [======>.......................] - ETA: 1:11 - loss: 0.3738 - categorical_accuracy: 0.8818
16000/60000 [=======>......................] - ETA: 1:11 - loss: 0.3726 - categorical_accuracy: 0.8823
16064/60000 [=======>......................] - ETA: 1:10 - loss: 0.3717 - categorical_accuracy: 0.8826
16128/60000 [=======>......................] - ETA: 1:10 - loss: 0.3709 - categorical_accuracy: 0.8828
16192/60000 [=======>......................] - ETA: 1:10 - loss: 0.3705 - categorical_accuracy: 0.8829
16256/60000 [=======>......................] - ETA: 1:10 - loss: 0.3696 - categorical_accuracy: 0.8832
16320/60000 [=======>......................] - ETA: 1:10 - loss: 0.3687 - categorical_accuracy: 0.8835
16352/60000 [=======>......................] - ETA: 1:10 - loss: 0.3681 - categorical_accuracy: 0.8837
16416/60000 [=======>......................] - ETA: 1:10 - loss: 0.3671 - categorical_accuracy: 0.8841
16480/60000 [=======>......................] - ETA: 1:10 - loss: 0.3665 - categorical_accuracy: 0.8844
16512/60000 [=======>......................] - ETA: 1:10 - loss: 0.3662 - categorical_accuracy: 0.8845
16544/60000 [=======>......................] - ETA: 1:09 - loss: 0.3656 - categorical_accuracy: 0.8847
16576/60000 [=======>......................] - ETA: 1:09 - loss: 0.3650 - categorical_accuracy: 0.8849
16608/60000 [=======>......................] - ETA: 1:09 - loss: 0.3644 - categorical_accuracy: 0.8851
16672/60000 [=======>......................] - ETA: 1:09 - loss: 0.3633 - categorical_accuracy: 0.8855
16736/60000 [=======>......................] - ETA: 1:09 - loss: 0.3623 - categorical_accuracy: 0.8858
16800/60000 [=======>......................] - ETA: 1:09 - loss: 0.3613 - categorical_accuracy: 0.8862
16864/60000 [=======>......................] - ETA: 1:09 - loss: 0.3619 - categorical_accuracy: 0.8863
16928/60000 [=======>......................] - ETA: 1:09 - loss: 0.3610 - categorical_accuracy: 0.8866
16992/60000 [=======>......................] - ETA: 1:09 - loss: 0.3612 - categorical_accuracy: 0.8865
17056/60000 [=======>......................] - ETA: 1:08 - loss: 0.3609 - categorical_accuracy: 0.8867
17120/60000 [=======>......................] - ETA: 1:08 - loss: 0.3606 - categorical_accuracy: 0.8867
17152/60000 [=======>......................] - ETA: 1:08 - loss: 0.3601 - categorical_accuracy: 0.8870
17216/60000 [=======>......................] - ETA: 1:08 - loss: 0.3591 - categorical_accuracy: 0.8873
17248/60000 [=======>......................] - ETA: 1:08 - loss: 0.3588 - categorical_accuracy: 0.8874
17280/60000 [=======>......................] - ETA: 1:08 - loss: 0.3583 - categorical_accuracy: 0.8876
17344/60000 [=======>......................] - ETA: 1:08 - loss: 0.3575 - categorical_accuracy: 0.8878
17408/60000 [=======>......................] - ETA: 1:08 - loss: 0.3577 - categorical_accuracy: 0.8879
17472/60000 [=======>......................] - ETA: 1:08 - loss: 0.3573 - categorical_accuracy: 0.8879
17536/60000 [=======>......................] - ETA: 1:08 - loss: 0.3565 - categorical_accuracy: 0.8881
17600/60000 [=======>......................] - ETA: 1:07 - loss: 0.3558 - categorical_accuracy: 0.8884
17664/60000 [=======>......................] - ETA: 1:07 - loss: 0.3552 - categorical_accuracy: 0.8886
17728/60000 [=======>......................] - ETA: 1:07 - loss: 0.3553 - categorical_accuracy: 0.8886
17792/60000 [=======>......................] - ETA: 1:07 - loss: 0.3544 - categorical_accuracy: 0.8889
17856/60000 [=======>......................] - ETA: 1:07 - loss: 0.3539 - categorical_accuracy: 0.8890
17920/60000 [=======>......................] - ETA: 1:07 - loss: 0.3529 - categorical_accuracy: 0.8893
17984/60000 [=======>......................] - ETA: 1:07 - loss: 0.3523 - categorical_accuracy: 0.8895
18016/60000 [========>.....................] - ETA: 1:07 - loss: 0.3518 - categorical_accuracy: 0.8897
18080/60000 [========>.....................] - ETA: 1:07 - loss: 0.3515 - categorical_accuracy: 0.8897
18144/60000 [========>.....................] - ETA: 1:06 - loss: 0.3504 - categorical_accuracy: 0.8901
18208/60000 [========>.....................] - ETA: 1:06 - loss: 0.3496 - categorical_accuracy: 0.8904
18272/60000 [========>.....................] - ETA: 1:06 - loss: 0.3489 - categorical_accuracy: 0.8905
18336/60000 [========>.....................] - ETA: 1:06 - loss: 0.3489 - categorical_accuracy: 0.8904
18368/60000 [========>.....................] - ETA: 1:06 - loss: 0.3485 - categorical_accuracy: 0.8906
18400/60000 [========>.....................] - ETA: 1:06 - loss: 0.3480 - categorical_accuracy: 0.8908
18464/60000 [========>.....................] - ETA: 1:06 - loss: 0.3469 - categorical_accuracy: 0.8911
18528/60000 [========>.....................] - ETA: 1:06 - loss: 0.3462 - categorical_accuracy: 0.8913
18592/60000 [========>.....................] - ETA: 1:06 - loss: 0.3455 - categorical_accuracy: 0.8915
18656/60000 [========>.....................] - ETA: 1:05 - loss: 0.3447 - categorical_accuracy: 0.8917
18688/60000 [========>.....................] - ETA: 1:05 - loss: 0.3442 - categorical_accuracy: 0.8919
18752/60000 [========>.....................] - ETA: 1:05 - loss: 0.3436 - categorical_accuracy: 0.8920
18816/60000 [========>.....................] - ETA: 1:05 - loss: 0.3430 - categorical_accuracy: 0.8922
18880/60000 [========>.....................] - ETA: 1:05 - loss: 0.3426 - categorical_accuracy: 0.8924
18944/60000 [========>.....................] - ETA: 1:05 - loss: 0.3424 - categorical_accuracy: 0.8925
19008/60000 [========>.....................] - ETA: 1:05 - loss: 0.3418 - categorical_accuracy: 0.8927
19072/60000 [========>.....................] - ETA: 1:05 - loss: 0.3410 - categorical_accuracy: 0.8929
19136/60000 [========>.....................] - ETA: 1:05 - loss: 0.3400 - categorical_accuracy: 0.8931
19200/60000 [========>.....................] - ETA: 1:04 - loss: 0.3392 - categorical_accuracy: 0.8934
19232/60000 [========>.....................] - ETA: 1:04 - loss: 0.3387 - categorical_accuracy: 0.8936
19296/60000 [========>.....................] - ETA: 1:04 - loss: 0.3385 - categorical_accuracy: 0.8937
19360/60000 [========>.....................] - ETA: 1:04 - loss: 0.3380 - categorical_accuracy: 0.8940
19424/60000 [========>.....................] - ETA: 1:04 - loss: 0.3375 - categorical_accuracy: 0.8941
19456/60000 [========>.....................] - ETA: 1:04 - loss: 0.3372 - categorical_accuracy: 0.8942
19520/60000 [========>.....................] - ETA: 1:04 - loss: 0.3367 - categorical_accuracy: 0.8945
19584/60000 [========>.....................] - ETA: 1:04 - loss: 0.3360 - categorical_accuracy: 0.8947
19648/60000 [========>.....................] - ETA: 1:04 - loss: 0.3357 - categorical_accuracy: 0.8947
19712/60000 [========>.....................] - ETA: 1:03 - loss: 0.3349 - categorical_accuracy: 0.8950
19776/60000 [========>.....................] - ETA: 1:03 - loss: 0.3348 - categorical_accuracy: 0.8951
19840/60000 [========>.....................] - ETA: 1:03 - loss: 0.3341 - categorical_accuracy: 0.8953
19904/60000 [========>.....................] - ETA: 1:03 - loss: 0.3335 - categorical_accuracy: 0.8954
19968/60000 [========>.....................] - ETA: 1:03 - loss: 0.3332 - categorical_accuracy: 0.8955
20032/60000 [=========>....................] - ETA: 1:03 - loss: 0.3323 - categorical_accuracy: 0.8959
20096/60000 [=========>....................] - ETA: 1:03 - loss: 0.3316 - categorical_accuracy: 0.8960
20160/60000 [=========>....................] - ETA: 1:03 - loss: 0.3310 - categorical_accuracy: 0.8962
20224/60000 [=========>....................] - ETA: 1:03 - loss: 0.3301 - categorical_accuracy: 0.8965
20288/60000 [=========>....................] - ETA: 1:02 - loss: 0.3292 - categorical_accuracy: 0.8968
20320/60000 [=========>....................] - ETA: 1:02 - loss: 0.3289 - categorical_accuracy: 0.8969
20384/60000 [=========>....................] - ETA: 1:02 - loss: 0.3283 - categorical_accuracy: 0.8970
20448/60000 [=========>....................] - ETA: 1:02 - loss: 0.3281 - categorical_accuracy: 0.8971
20512/60000 [=========>....................] - ETA: 1:02 - loss: 0.3274 - categorical_accuracy: 0.8972
20576/60000 [=========>....................] - ETA: 1:02 - loss: 0.3266 - categorical_accuracy: 0.8975
20640/60000 [=========>....................] - ETA: 1:02 - loss: 0.3261 - categorical_accuracy: 0.8977
20704/60000 [=========>....................] - ETA: 1:02 - loss: 0.3259 - categorical_accuracy: 0.8978
20768/60000 [=========>....................] - ETA: 1:02 - loss: 0.3254 - categorical_accuracy: 0.8980
20832/60000 [=========>....................] - ETA: 1:01 - loss: 0.3251 - categorical_accuracy: 0.8980
20896/60000 [=========>....................] - ETA: 1:01 - loss: 0.3244 - categorical_accuracy: 0.8982
20960/60000 [=========>....................] - ETA: 1:01 - loss: 0.3238 - categorical_accuracy: 0.8984
21024/60000 [=========>....................] - ETA: 1:01 - loss: 0.3235 - categorical_accuracy: 0.8985
21088/60000 [=========>....................] - ETA: 1:01 - loss: 0.3229 - categorical_accuracy: 0.8987
21152/60000 [=========>....................] - ETA: 1:01 - loss: 0.3224 - categorical_accuracy: 0.8990
21216/60000 [=========>....................] - ETA: 1:01 - loss: 0.3217 - categorical_accuracy: 0.8992
21280/60000 [=========>....................] - ETA: 1:01 - loss: 0.3212 - categorical_accuracy: 0.8994
21344/60000 [=========>....................] - ETA: 1:00 - loss: 0.3206 - categorical_accuracy: 0.8996
21408/60000 [=========>....................] - ETA: 1:00 - loss: 0.3201 - categorical_accuracy: 0.8999
21472/60000 [=========>....................] - ETA: 1:00 - loss: 0.3194 - categorical_accuracy: 0.9001
21536/60000 [=========>....................] - ETA: 1:00 - loss: 0.3188 - categorical_accuracy: 0.9003
21600/60000 [=========>....................] - ETA: 1:00 - loss: 0.3181 - categorical_accuracy: 0.9005
21664/60000 [=========>....................] - ETA: 1:00 - loss: 0.3175 - categorical_accuracy: 0.9006
21728/60000 [=========>....................] - ETA: 1:00 - loss: 0.3168 - categorical_accuracy: 0.9009
21792/60000 [=========>....................] - ETA: 1:00 - loss: 0.3162 - categorical_accuracy: 0.9010
21856/60000 [=========>....................] - ETA: 1:00 - loss: 0.3155 - categorical_accuracy: 0.9012
21920/60000 [=========>....................] - ETA: 59s - loss: 0.3151 - categorical_accuracy: 0.9013 
21984/60000 [=========>....................] - ETA: 59s - loss: 0.3145 - categorical_accuracy: 0.9015
22048/60000 [==========>...................] - ETA: 59s - loss: 0.3138 - categorical_accuracy: 0.9017
22112/60000 [==========>...................] - ETA: 59s - loss: 0.3133 - categorical_accuracy: 0.9018
22176/60000 [==========>...................] - ETA: 59s - loss: 0.3130 - categorical_accuracy: 0.9020
22240/60000 [==========>...................] - ETA: 59s - loss: 0.3132 - categorical_accuracy: 0.9019
22304/60000 [==========>...................] - ETA: 59s - loss: 0.3127 - categorical_accuracy: 0.9021
22368/60000 [==========>...................] - ETA: 59s - loss: 0.3121 - categorical_accuracy: 0.9022
22432/60000 [==========>...................] - ETA: 59s - loss: 0.3120 - categorical_accuracy: 0.9022
22496/60000 [==========>...................] - ETA: 58s - loss: 0.3118 - categorical_accuracy: 0.9023
22560/60000 [==========>...................] - ETA: 58s - loss: 0.3115 - categorical_accuracy: 0.9024
22624/60000 [==========>...................] - ETA: 58s - loss: 0.3109 - categorical_accuracy: 0.9026
22688/60000 [==========>...................] - ETA: 58s - loss: 0.3104 - categorical_accuracy: 0.9028
22752/60000 [==========>...................] - ETA: 58s - loss: 0.3099 - categorical_accuracy: 0.9029
22816/60000 [==========>...................] - ETA: 58s - loss: 0.3094 - categorical_accuracy: 0.9031
22880/60000 [==========>...................] - ETA: 58s - loss: 0.3091 - categorical_accuracy: 0.9032
22944/60000 [==========>...................] - ETA: 58s - loss: 0.3085 - categorical_accuracy: 0.9034
23008/60000 [==========>...................] - ETA: 57s - loss: 0.3081 - categorical_accuracy: 0.9035
23072/60000 [==========>...................] - ETA: 57s - loss: 0.3073 - categorical_accuracy: 0.9037
23136/60000 [==========>...................] - ETA: 57s - loss: 0.3069 - categorical_accuracy: 0.9038
23200/60000 [==========>...................] - ETA: 57s - loss: 0.3065 - categorical_accuracy: 0.9040
23264/60000 [==========>...................] - ETA: 57s - loss: 0.3060 - categorical_accuracy: 0.9040
23328/60000 [==========>...................] - ETA: 57s - loss: 0.3056 - categorical_accuracy: 0.9042
23392/60000 [==========>...................] - ETA: 57s - loss: 0.3052 - categorical_accuracy: 0.9044
23424/60000 [==========>...................] - ETA: 57s - loss: 0.3048 - categorical_accuracy: 0.9045
23456/60000 [==========>...................] - ETA: 57s - loss: 0.3046 - categorical_accuracy: 0.9046
23520/60000 [==========>...................] - ETA: 57s - loss: 0.3038 - categorical_accuracy: 0.9048
23584/60000 [==========>...................] - ETA: 57s - loss: 0.3035 - categorical_accuracy: 0.9049
23648/60000 [==========>...................] - ETA: 56s - loss: 0.3028 - categorical_accuracy: 0.9052
23712/60000 [==========>...................] - ETA: 56s - loss: 0.3023 - categorical_accuracy: 0.9053
23776/60000 [==========>...................] - ETA: 56s - loss: 0.3017 - categorical_accuracy: 0.9055
23840/60000 [==========>...................] - ETA: 56s - loss: 0.3016 - categorical_accuracy: 0.9056
23904/60000 [==========>...................] - ETA: 56s - loss: 0.3010 - categorical_accuracy: 0.9058
23968/60000 [==========>...................] - ETA: 56s - loss: 0.3005 - categorical_accuracy: 0.9059
24000/60000 [===========>..................] - ETA: 56s - loss: 0.3006 - categorical_accuracy: 0.9060
24064/60000 [===========>..................] - ETA: 56s - loss: 0.3002 - categorical_accuracy: 0.9061
24128/60000 [===========>..................] - ETA: 56s - loss: 0.2998 - categorical_accuracy: 0.9062
24192/60000 [===========>..................] - ETA: 55s - loss: 0.2994 - categorical_accuracy: 0.9064
24256/60000 [===========>..................] - ETA: 55s - loss: 0.2989 - categorical_accuracy: 0.9065
24320/60000 [===========>..................] - ETA: 55s - loss: 0.2982 - categorical_accuracy: 0.9067
24384/60000 [===========>..................] - ETA: 55s - loss: 0.2981 - categorical_accuracy: 0.9069
24448/60000 [===========>..................] - ETA: 55s - loss: 0.2976 - categorical_accuracy: 0.9070
24512/60000 [===========>..................] - ETA: 55s - loss: 0.2969 - categorical_accuracy: 0.9073
24544/60000 [===========>..................] - ETA: 55s - loss: 0.2965 - categorical_accuracy: 0.9074
24608/60000 [===========>..................] - ETA: 55s - loss: 0.2960 - categorical_accuracy: 0.9076
24672/60000 [===========>..................] - ETA: 55s - loss: 0.2956 - categorical_accuracy: 0.9076
24736/60000 [===========>..................] - ETA: 54s - loss: 0.2953 - categorical_accuracy: 0.9078
24800/60000 [===========>..................] - ETA: 54s - loss: 0.2953 - categorical_accuracy: 0.9078
24864/60000 [===========>..................] - ETA: 54s - loss: 0.2947 - categorical_accuracy: 0.9080
24928/60000 [===========>..................] - ETA: 54s - loss: 0.2945 - categorical_accuracy: 0.9082
24992/60000 [===========>..................] - ETA: 54s - loss: 0.2942 - categorical_accuracy: 0.9083
25056/60000 [===========>..................] - ETA: 54s - loss: 0.2937 - categorical_accuracy: 0.9084
25120/60000 [===========>..................] - ETA: 54s - loss: 0.2931 - categorical_accuracy: 0.9086
25184/60000 [===========>..................] - ETA: 54s - loss: 0.2928 - categorical_accuracy: 0.9088
25248/60000 [===========>..................] - ETA: 54s - loss: 0.2926 - categorical_accuracy: 0.9089
25312/60000 [===========>..................] - ETA: 53s - loss: 0.2920 - categorical_accuracy: 0.9091
25376/60000 [===========>..................] - ETA: 53s - loss: 0.2916 - categorical_accuracy: 0.9092
25440/60000 [===========>..................] - ETA: 53s - loss: 0.2913 - categorical_accuracy: 0.9094
25504/60000 [===========>..................] - ETA: 53s - loss: 0.2908 - categorical_accuracy: 0.9095
25568/60000 [===========>..................] - ETA: 53s - loss: 0.2902 - categorical_accuracy: 0.9096
25632/60000 [===========>..................] - ETA: 53s - loss: 0.2898 - categorical_accuracy: 0.9098
25696/60000 [===========>..................] - ETA: 53s - loss: 0.2895 - categorical_accuracy: 0.9099
25760/60000 [===========>..................] - ETA: 53s - loss: 0.2891 - categorical_accuracy: 0.9101
25824/60000 [===========>..................] - ETA: 53s - loss: 0.2891 - categorical_accuracy: 0.9101
25888/60000 [===========>..................] - ETA: 53s - loss: 0.2888 - categorical_accuracy: 0.9101
25952/60000 [===========>..................] - ETA: 52s - loss: 0.2883 - categorical_accuracy: 0.9103
26016/60000 [============>.................] - ETA: 52s - loss: 0.2877 - categorical_accuracy: 0.9104
26080/60000 [============>.................] - ETA: 52s - loss: 0.2872 - categorical_accuracy: 0.9106
26144/60000 [============>.................] - ETA: 52s - loss: 0.2867 - categorical_accuracy: 0.9107
26208/60000 [============>.................] - ETA: 52s - loss: 0.2864 - categorical_accuracy: 0.9108
26272/60000 [============>.................] - ETA: 52s - loss: 0.2859 - categorical_accuracy: 0.9109
26336/60000 [============>.................] - ETA: 52s - loss: 0.2858 - categorical_accuracy: 0.9110
26400/60000 [============>.................] - ETA: 52s - loss: 0.2852 - categorical_accuracy: 0.9112
26464/60000 [============>.................] - ETA: 52s - loss: 0.2850 - categorical_accuracy: 0.9114
26528/60000 [============>.................] - ETA: 51s - loss: 0.2849 - categorical_accuracy: 0.9113
26592/60000 [============>.................] - ETA: 51s - loss: 0.2847 - categorical_accuracy: 0.9114
26624/60000 [============>.................] - ETA: 51s - loss: 0.2844 - categorical_accuracy: 0.9115
26688/60000 [============>.................] - ETA: 51s - loss: 0.2841 - categorical_accuracy: 0.9116
26752/60000 [============>.................] - ETA: 51s - loss: 0.2837 - categorical_accuracy: 0.9117
26816/60000 [============>.................] - ETA: 51s - loss: 0.2833 - categorical_accuracy: 0.9118
26880/60000 [============>.................] - ETA: 51s - loss: 0.2829 - categorical_accuracy: 0.9119
26944/60000 [============>.................] - ETA: 51s - loss: 0.2826 - categorical_accuracy: 0.9120
27008/60000 [============>.................] - ETA: 51s - loss: 0.2822 - categorical_accuracy: 0.9122
27072/60000 [============>.................] - ETA: 51s - loss: 0.2816 - categorical_accuracy: 0.9124
27136/60000 [============>.................] - ETA: 50s - loss: 0.2814 - categorical_accuracy: 0.9125
27200/60000 [============>.................] - ETA: 50s - loss: 0.2809 - categorical_accuracy: 0.9127
27264/60000 [============>.................] - ETA: 50s - loss: 0.2806 - categorical_accuracy: 0.9127
27328/60000 [============>.................] - ETA: 50s - loss: 0.2803 - categorical_accuracy: 0.9129
27392/60000 [============>.................] - ETA: 50s - loss: 0.2801 - categorical_accuracy: 0.9129
27456/60000 [============>.................] - ETA: 50s - loss: 0.2798 - categorical_accuracy: 0.9130
27520/60000 [============>.................] - ETA: 50s - loss: 0.2796 - categorical_accuracy: 0.9130
27584/60000 [============>.................] - ETA: 50s - loss: 0.2792 - categorical_accuracy: 0.9132
27648/60000 [============>.................] - ETA: 50s - loss: 0.2788 - categorical_accuracy: 0.9133
27680/60000 [============>.................] - ETA: 49s - loss: 0.2786 - categorical_accuracy: 0.9134
27744/60000 [============>.................] - ETA: 49s - loss: 0.2782 - categorical_accuracy: 0.9135
27808/60000 [============>.................] - ETA: 49s - loss: 0.2780 - categorical_accuracy: 0.9136
27872/60000 [============>.................] - ETA: 49s - loss: 0.2776 - categorical_accuracy: 0.9137
27904/60000 [============>.................] - ETA: 49s - loss: 0.2774 - categorical_accuracy: 0.9138
27936/60000 [============>.................] - ETA: 49s - loss: 0.2772 - categorical_accuracy: 0.9139
28000/60000 [=============>................] - ETA: 49s - loss: 0.2767 - categorical_accuracy: 0.9140
28064/60000 [=============>................] - ETA: 49s - loss: 0.2767 - categorical_accuracy: 0.9140
28128/60000 [=============>................] - ETA: 49s - loss: 0.2764 - categorical_accuracy: 0.9141
28192/60000 [=============>................] - ETA: 49s - loss: 0.2761 - categorical_accuracy: 0.9142
28256/60000 [=============>................] - ETA: 49s - loss: 0.2756 - categorical_accuracy: 0.9144
28320/60000 [=============>................] - ETA: 48s - loss: 0.2751 - categorical_accuracy: 0.9145
28384/60000 [=============>................] - ETA: 48s - loss: 0.2751 - categorical_accuracy: 0.9146
28448/60000 [=============>................] - ETA: 48s - loss: 0.2746 - categorical_accuracy: 0.9148
28512/60000 [=============>................] - ETA: 48s - loss: 0.2742 - categorical_accuracy: 0.9149
28576/60000 [=============>................] - ETA: 48s - loss: 0.2737 - categorical_accuracy: 0.9151
28640/60000 [=============>................] - ETA: 48s - loss: 0.2736 - categorical_accuracy: 0.9151
28704/60000 [=============>................] - ETA: 48s - loss: 0.2732 - categorical_accuracy: 0.9152
28768/60000 [=============>................] - ETA: 48s - loss: 0.2728 - categorical_accuracy: 0.9154
28832/60000 [=============>................] - ETA: 48s - loss: 0.2723 - categorical_accuracy: 0.9155
28896/60000 [=============>................] - ETA: 47s - loss: 0.2720 - categorical_accuracy: 0.9156
28960/60000 [=============>................] - ETA: 47s - loss: 0.2715 - categorical_accuracy: 0.9158
29024/60000 [=============>................] - ETA: 47s - loss: 0.2712 - categorical_accuracy: 0.9158
29088/60000 [=============>................] - ETA: 47s - loss: 0.2709 - categorical_accuracy: 0.9158
29152/60000 [=============>................] - ETA: 47s - loss: 0.2706 - categorical_accuracy: 0.9160
29216/60000 [=============>................] - ETA: 47s - loss: 0.2703 - categorical_accuracy: 0.9160
29280/60000 [=============>................] - ETA: 47s - loss: 0.2698 - categorical_accuracy: 0.9162
29344/60000 [=============>................] - ETA: 47s - loss: 0.2697 - categorical_accuracy: 0.9162
29408/60000 [=============>................] - ETA: 47s - loss: 0.2697 - categorical_accuracy: 0.9162
29472/60000 [=============>................] - ETA: 46s - loss: 0.2693 - categorical_accuracy: 0.9164
29536/60000 [=============>................] - ETA: 46s - loss: 0.2689 - categorical_accuracy: 0.9165
29568/60000 [=============>................] - ETA: 46s - loss: 0.2688 - categorical_accuracy: 0.9165
29632/60000 [=============>................] - ETA: 46s - loss: 0.2684 - categorical_accuracy: 0.9167
29696/60000 [=============>................] - ETA: 46s - loss: 0.2680 - categorical_accuracy: 0.9167
29728/60000 [=============>................] - ETA: 46s - loss: 0.2679 - categorical_accuracy: 0.9167
29792/60000 [=============>................] - ETA: 46s - loss: 0.2674 - categorical_accuracy: 0.9169
29824/60000 [=============>................] - ETA: 46s - loss: 0.2671 - categorical_accuracy: 0.9170
29888/60000 [=============>................] - ETA: 46s - loss: 0.2668 - categorical_accuracy: 0.9171
29952/60000 [=============>................] - ETA: 46s - loss: 0.2663 - categorical_accuracy: 0.9173
30016/60000 [==============>...............] - ETA: 46s - loss: 0.2661 - categorical_accuracy: 0.9174
30080/60000 [==============>...............] - ETA: 46s - loss: 0.2660 - categorical_accuracy: 0.9175
30144/60000 [==============>...............] - ETA: 45s - loss: 0.2659 - categorical_accuracy: 0.9175
30208/60000 [==============>...............] - ETA: 45s - loss: 0.2656 - categorical_accuracy: 0.9176
30272/60000 [==============>...............] - ETA: 45s - loss: 0.2653 - categorical_accuracy: 0.9177
30336/60000 [==============>...............] - ETA: 45s - loss: 0.2649 - categorical_accuracy: 0.9178
30400/60000 [==============>...............] - ETA: 45s - loss: 0.2645 - categorical_accuracy: 0.9179
30464/60000 [==============>...............] - ETA: 45s - loss: 0.2646 - categorical_accuracy: 0.9179
30528/60000 [==============>...............] - ETA: 45s - loss: 0.2642 - categorical_accuracy: 0.9179
30592/60000 [==============>...............] - ETA: 45s - loss: 0.2638 - categorical_accuracy: 0.9181
30656/60000 [==============>...............] - ETA: 45s - loss: 0.2633 - categorical_accuracy: 0.9182
30720/60000 [==============>...............] - ETA: 44s - loss: 0.2630 - categorical_accuracy: 0.9183
30784/60000 [==============>...............] - ETA: 44s - loss: 0.2625 - categorical_accuracy: 0.9184
30848/60000 [==============>...............] - ETA: 44s - loss: 0.2622 - categorical_accuracy: 0.9185
30912/60000 [==============>...............] - ETA: 44s - loss: 0.2617 - categorical_accuracy: 0.9187
30976/60000 [==============>...............] - ETA: 44s - loss: 0.2612 - categorical_accuracy: 0.9188
31040/60000 [==============>...............] - ETA: 44s - loss: 0.2609 - categorical_accuracy: 0.9189
31104/60000 [==============>...............] - ETA: 44s - loss: 0.2605 - categorical_accuracy: 0.9190
31168/60000 [==============>...............] - ETA: 44s - loss: 0.2603 - categorical_accuracy: 0.9191
31232/60000 [==============>...............] - ETA: 44s - loss: 0.2599 - categorical_accuracy: 0.9192
31296/60000 [==============>...............] - ETA: 44s - loss: 0.2594 - categorical_accuracy: 0.9194
31360/60000 [==============>...............] - ETA: 43s - loss: 0.2591 - categorical_accuracy: 0.9195
31424/60000 [==============>...............] - ETA: 43s - loss: 0.2586 - categorical_accuracy: 0.9197
31488/60000 [==============>...............] - ETA: 43s - loss: 0.2585 - categorical_accuracy: 0.9197
31552/60000 [==============>...............] - ETA: 43s - loss: 0.2579 - categorical_accuracy: 0.9198
31616/60000 [==============>...............] - ETA: 43s - loss: 0.2577 - categorical_accuracy: 0.9199
31680/60000 [==============>...............] - ETA: 43s - loss: 0.2574 - categorical_accuracy: 0.9199
31712/60000 [==============>...............] - ETA: 43s - loss: 0.2572 - categorical_accuracy: 0.9200
31744/60000 [==============>...............] - ETA: 43s - loss: 0.2570 - categorical_accuracy: 0.9201
31776/60000 [==============>...............] - ETA: 43s - loss: 0.2569 - categorical_accuracy: 0.9200
31808/60000 [==============>...............] - ETA: 43s - loss: 0.2568 - categorical_accuracy: 0.9201
31840/60000 [==============>...............] - ETA: 43s - loss: 0.2569 - categorical_accuracy: 0.9201
31904/60000 [==============>...............] - ETA: 43s - loss: 0.2566 - categorical_accuracy: 0.9202
31936/60000 [==============>...............] - ETA: 43s - loss: 0.2565 - categorical_accuracy: 0.9202
32000/60000 [===============>..............] - ETA: 42s - loss: 0.2562 - categorical_accuracy: 0.9202
32064/60000 [===============>..............] - ETA: 42s - loss: 0.2560 - categorical_accuracy: 0.9203
32096/60000 [===============>..............] - ETA: 42s - loss: 0.2558 - categorical_accuracy: 0.9204
32160/60000 [===============>..............] - ETA: 42s - loss: 0.2555 - categorical_accuracy: 0.9204
32224/60000 [===============>..............] - ETA: 42s - loss: 0.2553 - categorical_accuracy: 0.9204
32256/60000 [===============>..............] - ETA: 42s - loss: 0.2551 - categorical_accuracy: 0.9205
32320/60000 [===============>..............] - ETA: 42s - loss: 0.2549 - categorical_accuracy: 0.9205
32384/60000 [===============>..............] - ETA: 42s - loss: 0.2547 - categorical_accuracy: 0.9206
32416/60000 [===============>..............] - ETA: 42s - loss: 0.2545 - categorical_accuracy: 0.9207
32448/60000 [===============>..............] - ETA: 42s - loss: 0.2545 - categorical_accuracy: 0.9207
32480/60000 [===============>..............] - ETA: 42s - loss: 0.2545 - categorical_accuracy: 0.9207
32512/60000 [===============>..............] - ETA: 42s - loss: 0.2543 - categorical_accuracy: 0.9208
32544/60000 [===============>..............] - ETA: 42s - loss: 0.2541 - categorical_accuracy: 0.9208
32576/60000 [===============>..............] - ETA: 42s - loss: 0.2540 - categorical_accuracy: 0.9209
32608/60000 [===============>..............] - ETA: 42s - loss: 0.2539 - categorical_accuracy: 0.9209
32640/60000 [===============>..............] - ETA: 42s - loss: 0.2537 - categorical_accuracy: 0.9210
32672/60000 [===============>..............] - ETA: 41s - loss: 0.2536 - categorical_accuracy: 0.9210
32704/60000 [===============>..............] - ETA: 41s - loss: 0.2534 - categorical_accuracy: 0.9211
32736/60000 [===============>..............] - ETA: 41s - loss: 0.2532 - categorical_accuracy: 0.9212
32800/60000 [===============>..............] - ETA: 41s - loss: 0.2530 - categorical_accuracy: 0.9212
32832/60000 [===============>..............] - ETA: 41s - loss: 0.2528 - categorical_accuracy: 0.9213
32864/60000 [===============>..............] - ETA: 41s - loss: 0.2526 - categorical_accuracy: 0.9214
32896/60000 [===============>..............] - ETA: 41s - loss: 0.2524 - categorical_accuracy: 0.9214
32928/60000 [===============>..............] - ETA: 41s - loss: 0.2522 - categorical_accuracy: 0.9215
32960/60000 [===============>..............] - ETA: 41s - loss: 0.2519 - categorical_accuracy: 0.9216
33024/60000 [===============>..............] - ETA: 41s - loss: 0.2515 - categorical_accuracy: 0.9218
33056/60000 [===============>..............] - ETA: 41s - loss: 0.2513 - categorical_accuracy: 0.9218
33088/60000 [===============>..............] - ETA: 41s - loss: 0.2512 - categorical_accuracy: 0.9219
33120/60000 [===============>..............] - ETA: 41s - loss: 0.2510 - categorical_accuracy: 0.9220
33184/60000 [===============>..............] - ETA: 41s - loss: 0.2507 - categorical_accuracy: 0.9220
33216/60000 [===============>..............] - ETA: 41s - loss: 0.2504 - categorical_accuracy: 0.9221
33248/60000 [===============>..............] - ETA: 41s - loss: 0.2504 - categorical_accuracy: 0.9221
33280/60000 [===============>..............] - ETA: 41s - loss: 0.2502 - categorical_accuracy: 0.9222
33312/60000 [===============>..............] - ETA: 41s - loss: 0.2499 - categorical_accuracy: 0.9223
33376/60000 [===============>..............] - ETA: 40s - loss: 0.2496 - categorical_accuracy: 0.9223
33408/60000 [===============>..............] - ETA: 40s - loss: 0.2495 - categorical_accuracy: 0.9223
33440/60000 [===============>..............] - ETA: 40s - loss: 0.2493 - categorical_accuracy: 0.9223
33472/60000 [===============>..............] - ETA: 40s - loss: 0.2492 - categorical_accuracy: 0.9224
33504/60000 [===============>..............] - ETA: 40s - loss: 0.2491 - categorical_accuracy: 0.9224
33536/60000 [===============>..............] - ETA: 40s - loss: 0.2489 - categorical_accuracy: 0.9224
33600/60000 [===============>..............] - ETA: 40s - loss: 0.2487 - categorical_accuracy: 0.9225
33632/60000 [===============>..............] - ETA: 40s - loss: 0.2485 - categorical_accuracy: 0.9226
33696/60000 [===============>..............] - ETA: 40s - loss: 0.2482 - categorical_accuracy: 0.9227
33760/60000 [===============>..............] - ETA: 40s - loss: 0.2478 - categorical_accuracy: 0.9228
33792/60000 [===============>..............] - ETA: 40s - loss: 0.2477 - categorical_accuracy: 0.9229
33824/60000 [===============>..............] - ETA: 40s - loss: 0.2475 - categorical_accuracy: 0.9229
33856/60000 [===============>..............] - ETA: 40s - loss: 0.2473 - categorical_accuracy: 0.9230
33888/60000 [===============>..............] - ETA: 40s - loss: 0.2471 - categorical_accuracy: 0.9230
33920/60000 [===============>..............] - ETA: 40s - loss: 0.2469 - categorical_accuracy: 0.9231
33952/60000 [===============>..............] - ETA: 40s - loss: 0.2467 - categorical_accuracy: 0.9232
34016/60000 [================>.............] - ETA: 40s - loss: 0.2465 - categorical_accuracy: 0.9232
34080/60000 [================>.............] - ETA: 39s - loss: 0.2462 - categorical_accuracy: 0.9233
34112/60000 [================>.............] - ETA: 39s - loss: 0.2461 - categorical_accuracy: 0.9233
34144/60000 [================>.............] - ETA: 39s - loss: 0.2461 - categorical_accuracy: 0.9234
34208/60000 [================>.............] - ETA: 39s - loss: 0.2458 - categorical_accuracy: 0.9235
34240/60000 [================>.............] - ETA: 39s - loss: 0.2456 - categorical_accuracy: 0.9236
34272/60000 [================>.............] - ETA: 39s - loss: 0.2453 - categorical_accuracy: 0.9236
34336/60000 [================>.............] - ETA: 39s - loss: 0.2450 - categorical_accuracy: 0.9238
34368/60000 [================>.............] - ETA: 39s - loss: 0.2449 - categorical_accuracy: 0.9238
34400/60000 [================>.............] - ETA: 39s - loss: 0.2453 - categorical_accuracy: 0.9238
34432/60000 [================>.............] - ETA: 39s - loss: 0.2452 - categorical_accuracy: 0.9238
34496/60000 [================>.............] - ETA: 39s - loss: 0.2450 - categorical_accuracy: 0.9238
34528/60000 [================>.............] - ETA: 39s - loss: 0.2450 - categorical_accuracy: 0.9238
34560/60000 [================>.............] - ETA: 39s - loss: 0.2448 - categorical_accuracy: 0.9239
34592/60000 [================>.............] - ETA: 39s - loss: 0.2446 - categorical_accuracy: 0.9239
34624/60000 [================>.............] - ETA: 39s - loss: 0.2444 - categorical_accuracy: 0.9240
34656/60000 [================>.............] - ETA: 39s - loss: 0.2442 - categorical_accuracy: 0.9241
34688/60000 [================>.............] - ETA: 39s - loss: 0.2441 - categorical_accuracy: 0.9242
34720/60000 [================>.............] - ETA: 38s - loss: 0.2438 - categorical_accuracy: 0.9242
34752/60000 [================>.............] - ETA: 38s - loss: 0.2441 - categorical_accuracy: 0.9241
34784/60000 [================>.............] - ETA: 38s - loss: 0.2439 - categorical_accuracy: 0.9242
34816/60000 [================>.............] - ETA: 38s - loss: 0.2437 - categorical_accuracy: 0.9243
34848/60000 [================>.............] - ETA: 38s - loss: 0.2436 - categorical_accuracy: 0.9244
34880/60000 [================>.............] - ETA: 38s - loss: 0.2436 - categorical_accuracy: 0.9244
34912/60000 [================>.............] - ETA: 38s - loss: 0.2434 - categorical_accuracy: 0.9244
34944/60000 [================>.............] - ETA: 38s - loss: 0.2433 - categorical_accuracy: 0.9245
34976/60000 [================>.............] - ETA: 38s - loss: 0.2432 - categorical_accuracy: 0.9245
35008/60000 [================>.............] - ETA: 38s - loss: 0.2430 - categorical_accuracy: 0.9245
35040/60000 [================>.............] - ETA: 38s - loss: 0.2432 - categorical_accuracy: 0.9245
35072/60000 [================>.............] - ETA: 38s - loss: 0.2431 - categorical_accuracy: 0.9246
35104/60000 [================>.............] - ETA: 38s - loss: 0.2429 - categorical_accuracy: 0.9246
35136/60000 [================>.............] - ETA: 38s - loss: 0.2427 - categorical_accuracy: 0.9247
35168/60000 [================>.............] - ETA: 38s - loss: 0.2426 - categorical_accuracy: 0.9247
35200/60000 [================>.............] - ETA: 38s - loss: 0.2424 - categorical_accuracy: 0.9248
35232/60000 [================>.............] - ETA: 38s - loss: 0.2423 - categorical_accuracy: 0.9248
35264/60000 [================>.............] - ETA: 38s - loss: 0.2423 - categorical_accuracy: 0.9248
35296/60000 [================>.............] - ETA: 38s - loss: 0.2422 - categorical_accuracy: 0.9248
35328/60000 [================>.............] - ETA: 38s - loss: 0.2421 - categorical_accuracy: 0.9248
35360/60000 [================>.............] - ETA: 38s - loss: 0.2419 - categorical_accuracy: 0.9249
35392/60000 [================>.............] - ETA: 38s - loss: 0.2418 - categorical_accuracy: 0.9249
35424/60000 [================>.............] - ETA: 38s - loss: 0.2417 - categorical_accuracy: 0.9249
35456/60000 [================>.............] - ETA: 38s - loss: 0.2415 - categorical_accuracy: 0.9250
35488/60000 [================>.............] - ETA: 37s - loss: 0.2415 - categorical_accuracy: 0.9250
35520/60000 [================>.............] - ETA: 37s - loss: 0.2414 - categorical_accuracy: 0.9250
35552/60000 [================>.............] - ETA: 37s - loss: 0.2412 - categorical_accuracy: 0.9251
35584/60000 [================>.............] - ETA: 37s - loss: 0.2411 - categorical_accuracy: 0.9251
35616/60000 [================>.............] - ETA: 37s - loss: 0.2411 - categorical_accuracy: 0.9251
35648/60000 [================>.............] - ETA: 37s - loss: 0.2412 - categorical_accuracy: 0.9251
35680/60000 [================>.............] - ETA: 37s - loss: 0.2410 - categorical_accuracy: 0.9252
35712/60000 [================>.............] - ETA: 37s - loss: 0.2410 - categorical_accuracy: 0.9252
35744/60000 [================>.............] - ETA: 37s - loss: 0.2408 - categorical_accuracy: 0.9253
35776/60000 [================>.............] - ETA: 37s - loss: 0.2406 - categorical_accuracy: 0.9253
35808/60000 [================>.............] - ETA: 37s - loss: 0.2406 - categorical_accuracy: 0.9254
35840/60000 [================>.............] - ETA: 37s - loss: 0.2405 - categorical_accuracy: 0.9254
35904/60000 [================>.............] - ETA: 37s - loss: 0.2403 - categorical_accuracy: 0.9255
35936/60000 [================>.............] - ETA: 37s - loss: 0.2402 - categorical_accuracy: 0.9255
35968/60000 [================>.............] - ETA: 37s - loss: 0.2402 - categorical_accuracy: 0.9255
36032/60000 [=================>............] - ETA: 37s - loss: 0.2399 - categorical_accuracy: 0.9256
36064/60000 [=================>............] - ETA: 37s - loss: 0.2398 - categorical_accuracy: 0.9256
36128/60000 [=================>............] - ETA: 37s - loss: 0.2395 - categorical_accuracy: 0.9257
36160/60000 [=================>............] - ETA: 37s - loss: 0.2394 - categorical_accuracy: 0.9257
36192/60000 [=================>............] - ETA: 36s - loss: 0.2392 - categorical_accuracy: 0.9258
36224/60000 [=================>............] - ETA: 36s - loss: 0.2391 - categorical_accuracy: 0.9258
36256/60000 [=================>............] - ETA: 36s - loss: 0.2389 - categorical_accuracy: 0.9259
36288/60000 [=================>............] - ETA: 36s - loss: 0.2388 - categorical_accuracy: 0.9259
36320/60000 [=================>............] - ETA: 36s - loss: 0.2388 - categorical_accuracy: 0.9259
36352/60000 [=================>............] - ETA: 36s - loss: 0.2387 - categorical_accuracy: 0.9260
36384/60000 [=================>............] - ETA: 36s - loss: 0.2386 - categorical_accuracy: 0.9260
36448/60000 [=================>............] - ETA: 36s - loss: 0.2384 - categorical_accuracy: 0.9260
36480/60000 [=================>............] - ETA: 36s - loss: 0.2384 - categorical_accuracy: 0.9260
36544/60000 [=================>............] - ETA: 36s - loss: 0.2381 - categorical_accuracy: 0.9261
36576/60000 [=================>............] - ETA: 36s - loss: 0.2379 - categorical_accuracy: 0.9262
36608/60000 [=================>............] - ETA: 36s - loss: 0.2378 - categorical_accuracy: 0.9263
36640/60000 [=================>............] - ETA: 36s - loss: 0.2377 - categorical_accuracy: 0.9263
36672/60000 [=================>............] - ETA: 36s - loss: 0.2377 - categorical_accuracy: 0.9263
36704/60000 [=================>............] - ETA: 36s - loss: 0.2377 - categorical_accuracy: 0.9264
36736/60000 [=================>............] - ETA: 36s - loss: 0.2377 - categorical_accuracy: 0.9264
36768/60000 [=================>............] - ETA: 36s - loss: 0.2376 - categorical_accuracy: 0.9264
36800/60000 [=================>............] - ETA: 36s - loss: 0.2374 - categorical_accuracy: 0.9265
36832/60000 [=================>............] - ETA: 36s - loss: 0.2375 - categorical_accuracy: 0.9264
36864/60000 [=================>............] - ETA: 35s - loss: 0.2374 - categorical_accuracy: 0.9264
36896/60000 [=================>............] - ETA: 35s - loss: 0.2372 - categorical_accuracy: 0.9265
36928/60000 [=================>............] - ETA: 35s - loss: 0.2371 - categorical_accuracy: 0.9265
36960/60000 [=================>............] - ETA: 35s - loss: 0.2371 - categorical_accuracy: 0.9265
37024/60000 [=================>............] - ETA: 35s - loss: 0.2368 - categorical_accuracy: 0.9266
37088/60000 [=================>............] - ETA: 35s - loss: 0.2367 - categorical_accuracy: 0.9267
37120/60000 [=================>............] - ETA: 35s - loss: 0.2367 - categorical_accuracy: 0.9267
37152/60000 [=================>............] - ETA: 35s - loss: 0.2365 - categorical_accuracy: 0.9267
37184/60000 [=================>............] - ETA: 35s - loss: 0.2364 - categorical_accuracy: 0.9267
37216/60000 [=================>............] - ETA: 35s - loss: 0.2364 - categorical_accuracy: 0.9268
37248/60000 [=================>............] - ETA: 35s - loss: 0.2363 - categorical_accuracy: 0.9267
37280/60000 [=================>............] - ETA: 35s - loss: 0.2362 - categorical_accuracy: 0.9268
37312/60000 [=================>............] - ETA: 35s - loss: 0.2360 - categorical_accuracy: 0.9268
37344/60000 [=================>............] - ETA: 35s - loss: 0.2360 - categorical_accuracy: 0.9268
37376/60000 [=================>............] - ETA: 35s - loss: 0.2358 - categorical_accuracy: 0.9269
37408/60000 [=================>............] - ETA: 35s - loss: 0.2357 - categorical_accuracy: 0.9270
37440/60000 [=================>............] - ETA: 35s - loss: 0.2356 - categorical_accuracy: 0.9270
37504/60000 [=================>............] - ETA: 34s - loss: 0.2355 - categorical_accuracy: 0.9270
37568/60000 [=================>............] - ETA: 34s - loss: 0.2352 - categorical_accuracy: 0.9271
37600/60000 [=================>............] - ETA: 34s - loss: 0.2350 - categorical_accuracy: 0.9272
37632/60000 [=================>............] - ETA: 34s - loss: 0.2349 - categorical_accuracy: 0.9272
37664/60000 [=================>............] - ETA: 34s - loss: 0.2347 - categorical_accuracy: 0.9273
37696/60000 [=================>............] - ETA: 34s - loss: 0.2346 - categorical_accuracy: 0.9273
37760/60000 [=================>............] - ETA: 34s - loss: 0.2344 - categorical_accuracy: 0.9274
37792/60000 [=================>............] - ETA: 34s - loss: 0.2343 - categorical_accuracy: 0.9274
37824/60000 [=================>............] - ETA: 34s - loss: 0.2341 - categorical_accuracy: 0.9275
37856/60000 [=================>............] - ETA: 34s - loss: 0.2339 - categorical_accuracy: 0.9275
37920/60000 [=================>............] - ETA: 34s - loss: 0.2336 - categorical_accuracy: 0.9276
37952/60000 [=================>............] - ETA: 34s - loss: 0.2335 - categorical_accuracy: 0.9277
37984/60000 [=================>............] - ETA: 34s - loss: 0.2334 - categorical_accuracy: 0.9277
38016/60000 [==================>...........] - ETA: 34s - loss: 0.2333 - categorical_accuracy: 0.9278
38080/60000 [==================>...........] - ETA: 34s - loss: 0.2331 - categorical_accuracy: 0.9278
38112/60000 [==================>...........] - ETA: 34s - loss: 0.2330 - categorical_accuracy: 0.9279
38144/60000 [==================>...........] - ETA: 34s - loss: 0.2329 - categorical_accuracy: 0.9279
38176/60000 [==================>...........] - ETA: 33s - loss: 0.2327 - categorical_accuracy: 0.9279
38208/60000 [==================>...........] - ETA: 33s - loss: 0.2326 - categorical_accuracy: 0.9279
38272/60000 [==================>...........] - ETA: 33s - loss: 0.2324 - categorical_accuracy: 0.9280
38304/60000 [==================>...........] - ETA: 33s - loss: 0.2322 - categorical_accuracy: 0.9280
38368/60000 [==================>...........] - ETA: 33s - loss: 0.2319 - categorical_accuracy: 0.9281
38400/60000 [==================>...........] - ETA: 33s - loss: 0.2319 - categorical_accuracy: 0.9281
38432/60000 [==================>...........] - ETA: 33s - loss: 0.2321 - categorical_accuracy: 0.9281
38464/60000 [==================>...........] - ETA: 33s - loss: 0.2319 - categorical_accuracy: 0.9281
38496/60000 [==================>...........] - ETA: 33s - loss: 0.2317 - categorical_accuracy: 0.9282
38528/60000 [==================>...........] - ETA: 33s - loss: 0.2317 - categorical_accuracy: 0.9282
38592/60000 [==================>...........] - ETA: 33s - loss: 0.2314 - categorical_accuracy: 0.9282
38624/60000 [==================>...........] - ETA: 33s - loss: 0.2313 - categorical_accuracy: 0.9283
38656/60000 [==================>...........] - ETA: 33s - loss: 0.2312 - categorical_accuracy: 0.9283
38688/60000 [==================>...........] - ETA: 33s - loss: 0.2311 - categorical_accuracy: 0.9283
38720/60000 [==================>...........] - ETA: 33s - loss: 0.2312 - categorical_accuracy: 0.9284
38752/60000 [==================>...........] - ETA: 33s - loss: 0.2310 - categorical_accuracy: 0.9284
38784/60000 [==================>...........] - ETA: 33s - loss: 0.2310 - categorical_accuracy: 0.9284
38816/60000 [==================>...........] - ETA: 33s - loss: 0.2308 - categorical_accuracy: 0.9285
38848/60000 [==================>...........] - ETA: 32s - loss: 0.2306 - categorical_accuracy: 0.9285
38880/60000 [==================>...........] - ETA: 32s - loss: 0.2305 - categorical_accuracy: 0.9286
38912/60000 [==================>...........] - ETA: 32s - loss: 0.2303 - categorical_accuracy: 0.9286
38976/60000 [==================>...........] - ETA: 32s - loss: 0.2301 - categorical_accuracy: 0.9287
39008/60000 [==================>...........] - ETA: 32s - loss: 0.2300 - categorical_accuracy: 0.9288
39072/60000 [==================>...........] - ETA: 32s - loss: 0.2297 - categorical_accuracy: 0.9288
39136/60000 [==================>...........] - ETA: 32s - loss: 0.2295 - categorical_accuracy: 0.9289
39200/60000 [==================>...........] - ETA: 32s - loss: 0.2292 - categorical_accuracy: 0.9290
39232/60000 [==================>...........] - ETA: 32s - loss: 0.2291 - categorical_accuracy: 0.9290
39264/60000 [==================>...........] - ETA: 32s - loss: 0.2289 - categorical_accuracy: 0.9291
39296/60000 [==================>...........] - ETA: 32s - loss: 0.2288 - categorical_accuracy: 0.9291
39328/60000 [==================>...........] - ETA: 32s - loss: 0.2287 - categorical_accuracy: 0.9291
39392/60000 [==================>...........] - ETA: 32s - loss: 0.2284 - categorical_accuracy: 0.9292
39424/60000 [==================>...........] - ETA: 32s - loss: 0.2283 - categorical_accuracy: 0.9292
39456/60000 [==================>...........] - ETA: 32s - loss: 0.2282 - categorical_accuracy: 0.9293
39488/60000 [==================>...........] - ETA: 32s - loss: 0.2282 - categorical_accuracy: 0.9293
39520/60000 [==================>...........] - ETA: 31s - loss: 0.2282 - categorical_accuracy: 0.9293
39552/60000 [==================>...........] - ETA: 31s - loss: 0.2282 - categorical_accuracy: 0.9293
39584/60000 [==================>...........] - ETA: 31s - loss: 0.2281 - categorical_accuracy: 0.9293
39616/60000 [==================>...........] - ETA: 31s - loss: 0.2281 - categorical_accuracy: 0.9293
39680/60000 [==================>...........] - ETA: 31s - loss: 0.2279 - categorical_accuracy: 0.9294
39712/60000 [==================>...........] - ETA: 31s - loss: 0.2277 - categorical_accuracy: 0.9294
39744/60000 [==================>...........] - ETA: 31s - loss: 0.2276 - categorical_accuracy: 0.9295
39776/60000 [==================>...........] - ETA: 31s - loss: 0.2275 - categorical_accuracy: 0.9295
39808/60000 [==================>...........] - ETA: 31s - loss: 0.2273 - categorical_accuracy: 0.9296
39840/60000 [==================>...........] - ETA: 31s - loss: 0.2272 - categorical_accuracy: 0.9296
39872/60000 [==================>...........] - ETA: 31s - loss: 0.2270 - categorical_accuracy: 0.9297
39904/60000 [==================>...........] - ETA: 31s - loss: 0.2270 - categorical_accuracy: 0.9297
39936/60000 [==================>...........] - ETA: 31s - loss: 0.2268 - categorical_accuracy: 0.9298
39968/60000 [==================>...........] - ETA: 31s - loss: 0.2267 - categorical_accuracy: 0.9297
40000/60000 [===================>..........] - ETA: 31s - loss: 0.2266 - categorical_accuracy: 0.9298
40032/60000 [===================>..........] - ETA: 31s - loss: 0.2265 - categorical_accuracy: 0.9298
40064/60000 [===================>..........] - ETA: 31s - loss: 0.2263 - categorical_accuracy: 0.9299
40096/60000 [===================>..........] - ETA: 31s - loss: 0.2263 - categorical_accuracy: 0.9298
40128/60000 [===================>..........] - ETA: 31s - loss: 0.2263 - categorical_accuracy: 0.9298
40160/60000 [===================>..........] - ETA: 31s - loss: 0.2262 - categorical_accuracy: 0.9299
40192/60000 [===================>..........] - ETA: 31s - loss: 0.2261 - categorical_accuracy: 0.9299
40224/60000 [===================>..........] - ETA: 30s - loss: 0.2259 - categorical_accuracy: 0.9299
40256/60000 [===================>..........] - ETA: 30s - loss: 0.2258 - categorical_accuracy: 0.9300
40288/60000 [===================>..........] - ETA: 30s - loss: 0.2257 - categorical_accuracy: 0.9300
40352/60000 [===================>..........] - ETA: 30s - loss: 0.2254 - categorical_accuracy: 0.9301
40384/60000 [===================>..........] - ETA: 30s - loss: 0.2253 - categorical_accuracy: 0.9301
40416/60000 [===================>..........] - ETA: 30s - loss: 0.2252 - categorical_accuracy: 0.9302
40448/60000 [===================>..........] - ETA: 30s - loss: 0.2252 - categorical_accuracy: 0.9302
40480/60000 [===================>..........] - ETA: 30s - loss: 0.2250 - categorical_accuracy: 0.9302
40512/60000 [===================>..........] - ETA: 30s - loss: 0.2250 - categorical_accuracy: 0.9302
40544/60000 [===================>..........] - ETA: 30s - loss: 0.2249 - categorical_accuracy: 0.9302
40576/60000 [===================>..........] - ETA: 30s - loss: 0.2248 - categorical_accuracy: 0.9303
40608/60000 [===================>..........] - ETA: 30s - loss: 0.2247 - categorical_accuracy: 0.9303
40640/60000 [===================>..........] - ETA: 30s - loss: 0.2246 - categorical_accuracy: 0.9303
40672/60000 [===================>..........] - ETA: 30s - loss: 0.2244 - categorical_accuracy: 0.9304
40704/60000 [===================>..........] - ETA: 30s - loss: 0.2242 - categorical_accuracy: 0.9304
40736/60000 [===================>..........] - ETA: 30s - loss: 0.2241 - categorical_accuracy: 0.9305
40768/60000 [===================>..........] - ETA: 30s - loss: 0.2239 - categorical_accuracy: 0.9305
40800/60000 [===================>..........] - ETA: 30s - loss: 0.2238 - categorical_accuracy: 0.9306
40864/60000 [===================>..........] - ETA: 29s - loss: 0.2237 - categorical_accuracy: 0.9306
40928/60000 [===================>..........] - ETA: 29s - loss: 0.2238 - categorical_accuracy: 0.9306
40960/60000 [===================>..........] - ETA: 29s - loss: 0.2238 - categorical_accuracy: 0.9306
41024/60000 [===================>..........] - ETA: 29s - loss: 0.2235 - categorical_accuracy: 0.9307
41056/60000 [===================>..........] - ETA: 29s - loss: 0.2236 - categorical_accuracy: 0.9307
41088/60000 [===================>..........] - ETA: 29s - loss: 0.2235 - categorical_accuracy: 0.9307
41120/60000 [===================>..........] - ETA: 29s - loss: 0.2233 - categorical_accuracy: 0.9308
41152/60000 [===================>..........] - ETA: 29s - loss: 0.2232 - categorical_accuracy: 0.9308
41184/60000 [===================>..........] - ETA: 29s - loss: 0.2233 - categorical_accuracy: 0.9308
41216/60000 [===================>..........] - ETA: 29s - loss: 0.2232 - categorical_accuracy: 0.9309
41248/60000 [===================>..........] - ETA: 29s - loss: 0.2231 - categorical_accuracy: 0.9309
41280/60000 [===================>..........] - ETA: 29s - loss: 0.2231 - categorical_accuracy: 0.9309
41312/60000 [===================>..........] - ETA: 29s - loss: 0.2230 - categorical_accuracy: 0.9309
41344/60000 [===================>..........] - ETA: 29s - loss: 0.2231 - categorical_accuracy: 0.9308
41376/60000 [===================>..........] - ETA: 29s - loss: 0.2230 - categorical_accuracy: 0.9309
41408/60000 [===================>..........] - ETA: 29s - loss: 0.2228 - categorical_accuracy: 0.9309
41440/60000 [===================>..........] - ETA: 29s - loss: 0.2227 - categorical_accuracy: 0.9309
41472/60000 [===================>..........] - ETA: 29s - loss: 0.2226 - categorical_accuracy: 0.9310
41504/60000 [===================>..........] - ETA: 29s - loss: 0.2225 - categorical_accuracy: 0.9310
41536/60000 [===================>..........] - ETA: 28s - loss: 0.2225 - categorical_accuracy: 0.9310
41568/60000 [===================>..........] - ETA: 28s - loss: 0.2224 - categorical_accuracy: 0.9311
41600/60000 [===================>..........] - ETA: 28s - loss: 0.2222 - categorical_accuracy: 0.9311
41632/60000 [===================>..........] - ETA: 28s - loss: 0.2222 - categorical_accuracy: 0.9311
41664/60000 [===================>..........] - ETA: 28s - loss: 0.2220 - categorical_accuracy: 0.9311
41696/60000 [===================>..........] - ETA: 28s - loss: 0.2219 - categorical_accuracy: 0.9312
41728/60000 [===================>..........] - ETA: 28s - loss: 0.2219 - categorical_accuracy: 0.9312
41760/60000 [===================>..........] - ETA: 28s - loss: 0.2217 - categorical_accuracy: 0.9313
41792/60000 [===================>..........] - ETA: 28s - loss: 0.2216 - categorical_accuracy: 0.9313
41824/60000 [===================>..........] - ETA: 28s - loss: 0.2215 - categorical_accuracy: 0.9314
41856/60000 [===================>..........] - ETA: 28s - loss: 0.2214 - categorical_accuracy: 0.9314
41888/60000 [===================>..........] - ETA: 28s - loss: 0.2212 - categorical_accuracy: 0.9314
41952/60000 [===================>..........] - ETA: 28s - loss: 0.2212 - categorical_accuracy: 0.9314
41984/60000 [===================>..........] - ETA: 28s - loss: 0.2213 - categorical_accuracy: 0.9315
42016/60000 [====================>.........] - ETA: 28s - loss: 0.2212 - categorical_accuracy: 0.9315
42048/60000 [====================>.........] - ETA: 28s - loss: 0.2212 - categorical_accuracy: 0.9315
42080/60000 [====================>.........] - ETA: 28s - loss: 0.2210 - categorical_accuracy: 0.9315
42112/60000 [====================>.........] - ETA: 28s - loss: 0.2209 - categorical_accuracy: 0.9315
42144/60000 [====================>.........] - ETA: 28s - loss: 0.2207 - categorical_accuracy: 0.9316
42176/60000 [====================>.........] - ETA: 28s - loss: 0.2206 - categorical_accuracy: 0.9316
42208/60000 [====================>.........] - ETA: 27s - loss: 0.2205 - categorical_accuracy: 0.9317
42240/60000 [====================>.........] - ETA: 27s - loss: 0.2203 - categorical_accuracy: 0.9317
42272/60000 [====================>.........] - ETA: 27s - loss: 0.2202 - categorical_accuracy: 0.9318
42336/60000 [====================>.........] - ETA: 27s - loss: 0.2199 - categorical_accuracy: 0.9319
42368/60000 [====================>.........] - ETA: 27s - loss: 0.2198 - categorical_accuracy: 0.9319
42400/60000 [====================>.........] - ETA: 27s - loss: 0.2197 - categorical_accuracy: 0.9319
42432/60000 [====================>.........] - ETA: 27s - loss: 0.2197 - categorical_accuracy: 0.9319
42496/60000 [====================>.........] - ETA: 27s - loss: 0.2196 - categorical_accuracy: 0.9320
42528/60000 [====================>.........] - ETA: 27s - loss: 0.2195 - categorical_accuracy: 0.9320
42560/60000 [====================>.........] - ETA: 27s - loss: 0.2194 - categorical_accuracy: 0.9320
42592/60000 [====================>.........] - ETA: 27s - loss: 0.2194 - categorical_accuracy: 0.9320
42624/60000 [====================>.........] - ETA: 27s - loss: 0.2194 - categorical_accuracy: 0.9320
42656/60000 [====================>.........] - ETA: 27s - loss: 0.2192 - categorical_accuracy: 0.9321
42688/60000 [====================>.........] - ETA: 27s - loss: 0.2191 - categorical_accuracy: 0.9321
42752/60000 [====================>.........] - ETA: 27s - loss: 0.2188 - categorical_accuracy: 0.9322
42784/60000 [====================>.........] - ETA: 27s - loss: 0.2187 - categorical_accuracy: 0.9322
42816/60000 [====================>.........] - ETA: 27s - loss: 0.2188 - categorical_accuracy: 0.9322
42848/60000 [====================>.........] - ETA: 27s - loss: 0.2188 - categorical_accuracy: 0.9322
42880/60000 [====================>.........] - ETA: 26s - loss: 0.2188 - categorical_accuracy: 0.9322
42944/60000 [====================>.........] - ETA: 26s - loss: 0.2188 - categorical_accuracy: 0.9322
42976/60000 [====================>.........] - ETA: 26s - loss: 0.2186 - categorical_accuracy: 0.9323
43008/60000 [====================>.........] - ETA: 26s - loss: 0.2185 - categorical_accuracy: 0.9323
43040/60000 [====================>.........] - ETA: 26s - loss: 0.2184 - categorical_accuracy: 0.9323
43072/60000 [====================>.........] - ETA: 26s - loss: 0.2185 - categorical_accuracy: 0.9323
43104/60000 [====================>.........] - ETA: 26s - loss: 0.2184 - categorical_accuracy: 0.9324
43136/60000 [====================>.........] - ETA: 26s - loss: 0.2182 - categorical_accuracy: 0.9324
43168/60000 [====================>.........] - ETA: 26s - loss: 0.2181 - categorical_accuracy: 0.9325
43200/60000 [====================>.........] - ETA: 26s - loss: 0.2181 - categorical_accuracy: 0.9324
43264/60000 [====================>.........] - ETA: 26s - loss: 0.2179 - categorical_accuracy: 0.9325
43296/60000 [====================>.........] - ETA: 26s - loss: 0.2177 - categorical_accuracy: 0.9326
43328/60000 [====================>.........] - ETA: 26s - loss: 0.2176 - categorical_accuracy: 0.9326
43360/60000 [====================>.........] - ETA: 26s - loss: 0.2176 - categorical_accuracy: 0.9326
43424/60000 [====================>.........] - ETA: 26s - loss: 0.2174 - categorical_accuracy: 0.9327
43456/60000 [====================>.........] - ETA: 26s - loss: 0.2174 - categorical_accuracy: 0.9327
43520/60000 [====================>.........] - ETA: 25s - loss: 0.2173 - categorical_accuracy: 0.9327
43552/60000 [====================>.........] - ETA: 25s - loss: 0.2172 - categorical_accuracy: 0.9327
43584/60000 [====================>.........] - ETA: 25s - loss: 0.2171 - categorical_accuracy: 0.9328
43616/60000 [====================>.........] - ETA: 25s - loss: 0.2170 - categorical_accuracy: 0.9328
43648/60000 [====================>.........] - ETA: 25s - loss: 0.2168 - categorical_accuracy: 0.9329
43680/60000 [====================>.........] - ETA: 25s - loss: 0.2167 - categorical_accuracy: 0.9329
43712/60000 [====================>.........] - ETA: 25s - loss: 0.2166 - categorical_accuracy: 0.9329
43744/60000 [====================>.........] - ETA: 25s - loss: 0.2164 - categorical_accuracy: 0.9330
43776/60000 [====================>.........] - ETA: 25s - loss: 0.2163 - categorical_accuracy: 0.9330
43808/60000 [====================>.........] - ETA: 25s - loss: 0.2162 - categorical_accuracy: 0.9330
43840/60000 [====================>.........] - ETA: 25s - loss: 0.2161 - categorical_accuracy: 0.9331
43872/60000 [====================>.........] - ETA: 25s - loss: 0.2160 - categorical_accuracy: 0.9331
43904/60000 [====================>.........] - ETA: 25s - loss: 0.2159 - categorical_accuracy: 0.9332
43936/60000 [====================>.........] - ETA: 25s - loss: 0.2157 - categorical_accuracy: 0.9332
43968/60000 [====================>.........] - ETA: 25s - loss: 0.2156 - categorical_accuracy: 0.9333
44000/60000 [=====================>........] - ETA: 25s - loss: 0.2154 - categorical_accuracy: 0.9333
44032/60000 [=====================>........] - ETA: 25s - loss: 0.2153 - categorical_accuracy: 0.9333
44096/60000 [=====================>........] - ETA: 25s - loss: 0.2152 - categorical_accuracy: 0.9334
44128/60000 [=====================>........] - ETA: 25s - loss: 0.2151 - categorical_accuracy: 0.9334
44160/60000 [=====================>........] - ETA: 25s - loss: 0.2151 - categorical_accuracy: 0.9334
44224/60000 [=====================>........] - ETA: 24s - loss: 0.2150 - categorical_accuracy: 0.9335
44256/60000 [=====================>........] - ETA: 24s - loss: 0.2150 - categorical_accuracy: 0.9335
44288/60000 [=====================>........] - ETA: 24s - loss: 0.2148 - categorical_accuracy: 0.9335
44320/60000 [=====================>........] - ETA: 24s - loss: 0.2149 - categorical_accuracy: 0.9335
44352/60000 [=====================>........] - ETA: 24s - loss: 0.2148 - categorical_accuracy: 0.9336
44384/60000 [=====================>........] - ETA: 24s - loss: 0.2146 - categorical_accuracy: 0.9336
44448/60000 [=====================>........] - ETA: 24s - loss: 0.2145 - categorical_accuracy: 0.9336
44480/60000 [=====================>........] - ETA: 24s - loss: 0.2144 - categorical_accuracy: 0.9337
44512/60000 [=====================>........] - ETA: 24s - loss: 0.2145 - categorical_accuracy: 0.9337
44544/60000 [=====================>........] - ETA: 24s - loss: 0.2143 - categorical_accuracy: 0.9337
44576/60000 [=====================>........] - ETA: 24s - loss: 0.2143 - categorical_accuracy: 0.9337
44608/60000 [=====================>........] - ETA: 24s - loss: 0.2142 - categorical_accuracy: 0.9338
44672/60000 [=====================>........] - ETA: 24s - loss: 0.2140 - categorical_accuracy: 0.9339
44704/60000 [=====================>........] - ETA: 24s - loss: 0.2139 - categorical_accuracy: 0.9339
44768/60000 [=====================>........] - ETA: 24s - loss: 0.2137 - categorical_accuracy: 0.9339
44800/60000 [=====================>........] - ETA: 24s - loss: 0.2136 - categorical_accuracy: 0.9340
44864/60000 [=====================>........] - ETA: 23s - loss: 0.2136 - categorical_accuracy: 0.9340
44928/60000 [=====================>........] - ETA: 23s - loss: 0.2134 - categorical_accuracy: 0.9340
44960/60000 [=====================>........] - ETA: 23s - loss: 0.2133 - categorical_accuracy: 0.9341
44992/60000 [=====================>........] - ETA: 23s - loss: 0.2133 - categorical_accuracy: 0.9341
45024/60000 [=====================>........] - ETA: 23s - loss: 0.2132 - categorical_accuracy: 0.9341
45056/60000 [=====================>........] - ETA: 23s - loss: 0.2131 - categorical_accuracy: 0.9341
45088/60000 [=====================>........] - ETA: 23s - loss: 0.2130 - categorical_accuracy: 0.9341
45152/60000 [=====================>........] - ETA: 23s - loss: 0.2131 - categorical_accuracy: 0.9341
45184/60000 [=====================>........] - ETA: 23s - loss: 0.2129 - categorical_accuracy: 0.9341
45248/60000 [=====================>........] - ETA: 23s - loss: 0.2128 - categorical_accuracy: 0.9342
45280/60000 [=====================>........] - ETA: 23s - loss: 0.2127 - categorical_accuracy: 0.9342
45312/60000 [=====================>........] - ETA: 23s - loss: 0.2126 - categorical_accuracy: 0.9343
45344/60000 [=====================>........] - ETA: 23s - loss: 0.2124 - categorical_accuracy: 0.9343
45376/60000 [=====================>........] - ETA: 23s - loss: 0.2123 - categorical_accuracy: 0.9344
45408/60000 [=====================>........] - ETA: 23s - loss: 0.2122 - categorical_accuracy: 0.9344
45440/60000 [=====================>........] - ETA: 23s - loss: 0.2121 - categorical_accuracy: 0.9344
45472/60000 [=====================>........] - ETA: 22s - loss: 0.2120 - categorical_accuracy: 0.9345
45504/60000 [=====================>........] - ETA: 22s - loss: 0.2119 - categorical_accuracy: 0.9345
45536/60000 [=====================>........] - ETA: 22s - loss: 0.2118 - categorical_accuracy: 0.9346
45600/60000 [=====================>........] - ETA: 22s - loss: 0.2116 - categorical_accuracy: 0.9346
45664/60000 [=====================>........] - ETA: 22s - loss: 0.2115 - categorical_accuracy: 0.9346
45696/60000 [=====================>........] - ETA: 22s - loss: 0.2114 - categorical_accuracy: 0.9346
45728/60000 [=====================>........] - ETA: 22s - loss: 0.2113 - categorical_accuracy: 0.9346
45792/60000 [=====================>........] - ETA: 22s - loss: 0.2114 - categorical_accuracy: 0.9346
45824/60000 [=====================>........] - ETA: 22s - loss: 0.2113 - categorical_accuracy: 0.9347
45856/60000 [=====================>........] - ETA: 22s - loss: 0.2112 - categorical_accuracy: 0.9347
45888/60000 [=====================>........] - ETA: 22s - loss: 0.2111 - categorical_accuracy: 0.9348
45952/60000 [=====================>........] - ETA: 22s - loss: 0.2110 - categorical_accuracy: 0.9348
46016/60000 [======================>.......] - ETA: 22s - loss: 0.2109 - categorical_accuracy: 0.9348
46048/60000 [======================>.......] - ETA: 22s - loss: 0.2108 - categorical_accuracy: 0.9349
46080/60000 [======================>.......] - ETA: 22s - loss: 0.2107 - categorical_accuracy: 0.9349
46112/60000 [======================>.......] - ETA: 21s - loss: 0.2106 - categorical_accuracy: 0.9349
46144/60000 [======================>.......] - ETA: 21s - loss: 0.2105 - categorical_accuracy: 0.9350
46176/60000 [======================>.......] - ETA: 21s - loss: 0.2104 - categorical_accuracy: 0.9350
46208/60000 [======================>.......] - ETA: 21s - loss: 0.2103 - categorical_accuracy: 0.9351
46272/60000 [======================>.......] - ETA: 21s - loss: 0.2103 - categorical_accuracy: 0.9351
46336/60000 [======================>.......] - ETA: 21s - loss: 0.2101 - categorical_accuracy: 0.9351
46368/60000 [======================>.......] - ETA: 21s - loss: 0.2100 - categorical_accuracy: 0.9351
46400/60000 [======================>.......] - ETA: 21s - loss: 0.2099 - categorical_accuracy: 0.9352
46432/60000 [======================>.......] - ETA: 21s - loss: 0.2100 - categorical_accuracy: 0.9352
46464/60000 [======================>.......] - ETA: 21s - loss: 0.2099 - categorical_accuracy: 0.9352
46496/60000 [======================>.......] - ETA: 21s - loss: 0.2099 - categorical_accuracy: 0.9352
46528/60000 [======================>.......] - ETA: 21s - loss: 0.2099 - categorical_accuracy: 0.9352
46560/60000 [======================>.......] - ETA: 21s - loss: 0.2098 - categorical_accuracy: 0.9352
46624/60000 [======================>.......] - ETA: 21s - loss: 0.2096 - categorical_accuracy: 0.9353
46656/60000 [======================>.......] - ETA: 21s - loss: 0.2095 - categorical_accuracy: 0.9354
46688/60000 [======================>.......] - ETA: 21s - loss: 0.2095 - categorical_accuracy: 0.9354
46720/60000 [======================>.......] - ETA: 21s - loss: 0.2094 - categorical_accuracy: 0.9354
46752/60000 [======================>.......] - ETA: 20s - loss: 0.2093 - categorical_accuracy: 0.9354
46784/60000 [======================>.......] - ETA: 20s - loss: 0.2093 - categorical_accuracy: 0.9354
46816/60000 [======================>.......] - ETA: 20s - loss: 0.2091 - categorical_accuracy: 0.9355
46848/60000 [======================>.......] - ETA: 20s - loss: 0.2090 - categorical_accuracy: 0.9355
46880/60000 [======================>.......] - ETA: 20s - loss: 0.2090 - categorical_accuracy: 0.9355
46944/60000 [======================>.......] - ETA: 20s - loss: 0.2089 - categorical_accuracy: 0.9355
46976/60000 [======================>.......] - ETA: 20s - loss: 0.2089 - categorical_accuracy: 0.9355
47008/60000 [======================>.......] - ETA: 20s - loss: 0.2088 - categorical_accuracy: 0.9356
47040/60000 [======================>.......] - ETA: 20s - loss: 0.2087 - categorical_accuracy: 0.9356
47072/60000 [======================>.......] - ETA: 20s - loss: 0.2086 - categorical_accuracy: 0.9356
47136/60000 [======================>.......] - ETA: 20s - loss: 0.2085 - categorical_accuracy: 0.9357
47200/60000 [======================>.......] - ETA: 20s - loss: 0.2085 - categorical_accuracy: 0.9357
47232/60000 [======================>.......] - ETA: 20s - loss: 0.2084 - categorical_accuracy: 0.9357
47296/60000 [======================>.......] - ETA: 20s - loss: 0.2083 - categorical_accuracy: 0.9357
47360/60000 [======================>.......] - ETA: 20s - loss: 0.2081 - categorical_accuracy: 0.9358
47392/60000 [======================>.......] - ETA: 19s - loss: 0.2080 - categorical_accuracy: 0.9358
47456/60000 [======================>.......] - ETA: 19s - loss: 0.2078 - categorical_accuracy: 0.9359
47488/60000 [======================>.......] - ETA: 19s - loss: 0.2076 - categorical_accuracy: 0.9359
47520/60000 [======================>.......] - ETA: 19s - loss: 0.2075 - categorical_accuracy: 0.9360
47584/60000 [======================>.......] - ETA: 19s - loss: 0.2075 - categorical_accuracy: 0.9360
47616/60000 [======================>.......] - ETA: 19s - loss: 0.2075 - categorical_accuracy: 0.9359
47648/60000 [======================>.......] - ETA: 19s - loss: 0.2075 - categorical_accuracy: 0.9359
47680/60000 [======================>.......] - ETA: 19s - loss: 0.2074 - categorical_accuracy: 0.9360
47712/60000 [======================>.......] - ETA: 19s - loss: 0.2073 - categorical_accuracy: 0.9360
47744/60000 [======================>.......] - ETA: 19s - loss: 0.2071 - categorical_accuracy: 0.9361
47776/60000 [======================>.......] - ETA: 19s - loss: 0.2070 - categorical_accuracy: 0.9361
47840/60000 [======================>.......] - ETA: 19s - loss: 0.2068 - categorical_accuracy: 0.9362
47904/60000 [======================>.......] - ETA: 19s - loss: 0.2066 - categorical_accuracy: 0.9362
47968/60000 [======================>.......] - ETA: 19s - loss: 0.2065 - categorical_accuracy: 0.9362
48000/60000 [=======================>......] - ETA: 19s - loss: 0.2065 - categorical_accuracy: 0.9362
48032/60000 [=======================>......] - ETA: 18s - loss: 0.2064 - categorical_accuracy: 0.9363
48096/60000 [=======================>......] - ETA: 18s - loss: 0.2062 - categorical_accuracy: 0.9363
48128/60000 [=======================>......] - ETA: 18s - loss: 0.2062 - categorical_accuracy: 0.9363
48160/60000 [=======================>......] - ETA: 18s - loss: 0.2062 - categorical_accuracy: 0.9363
48192/60000 [=======================>......] - ETA: 18s - loss: 0.2061 - categorical_accuracy: 0.9364
48224/60000 [=======================>......] - ETA: 18s - loss: 0.2059 - categorical_accuracy: 0.9364
48288/60000 [=======================>......] - ETA: 18s - loss: 0.2058 - categorical_accuracy: 0.9365
48320/60000 [=======================>......] - ETA: 18s - loss: 0.2057 - categorical_accuracy: 0.9365
48352/60000 [=======================>......] - ETA: 18s - loss: 0.2058 - categorical_accuracy: 0.9365
48416/60000 [=======================>......] - ETA: 18s - loss: 0.2056 - categorical_accuracy: 0.9365
48480/60000 [=======================>......] - ETA: 18s - loss: 0.2056 - categorical_accuracy: 0.9365
48544/60000 [=======================>......] - ETA: 18s - loss: 0.2055 - categorical_accuracy: 0.9365
48576/60000 [=======================>......] - ETA: 18s - loss: 0.2054 - categorical_accuracy: 0.9366
48640/60000 [=======================>......] - ETA: 18s - loss: 0.2052 - categorical_accuracy: 0.9366
48704/60000 [=======================>......] - ETA: 17s - loss: 0.2052 - categorical_accuracy: 0.9366
48768/60000 [=======================>......] - ETA: 17s - loss: 0.2051 - categorical_accuracy: 0.9366
48832/60000 [=======================>......] - ETA: 17s - loss: 0.2050 - categorical_accuracy: 0.9367
48864/60000 [=======================>......] - ETA: 17s - loss: 0.2049 - categorical_accuracy: 0.9367
48896/60000 [=======================>......] - ETA: 17s - loss: 0.2048 - categorical_accuracy: 0.9367
48928/60000 [=======================>......] - ETA: 17s - loss: 0.2047 - categorical_accuracy: 0.9367
48992/60000 [=======================>......] - ETA: 17s - loss: 0.2047 - categorical_accuracy: 0.9367
49056/60000 [=======================>......] - ETA: 17s - loss: 0.2045 - categorical_accuracy: 0.9368
49120/60000 [=======================>......] - ETA: 17s - loss: 0.2044 - categorical_accuracy: 0.9368
49184/60000 [=======================>......] - ETA: 17s - loss: 0.2044 - categorical_accuracy: 0.9368
49248/60000 [=======================>......] - ETA: 17s - loss: 0.2042 - categorical_accuracy: 0.9369
49312/60000 [=======================>......] - ETA: 16s - loss: 0.2042 - categorical_accuracy: 0.9369
49376/60000 [=======================>......] - ETA: 16s - loss: 0.2040 - categorical_accuracy: 0.9370
49440/60000 [=======================>......] - ETA: 16s - loss: 0.2039 - categorical_accuracy: 0.9370
49472/60000 [=======================>......] - ETA: 16s - loss: 0.2038 - categorical_accuracy: 0.9370
49536/60000 [=======================>......] - ETA: 16s - loss: 0.2036 - categorical_accuracy: 0.9371
49568/60000 [=======================>......] - ETA: 16s - loss: 0.2035 - categorical_accuracy: 0.9371
49632/60000 [=======================>......] - ETA: 16s - loss: 0.2033 - categorical_accuracy: 0.9371
49696/60000 [=======================>......] - ETA: 16s - loss: 0.2031 - categorical_accuracy: 0.9372
49760/60000 [=======================>......] - ETA: 16s - loss: 0.2029 - categorical_accuracy: 0.9372
49824/60000 [=======================>......] - ETA: 16s - loss: 0.2027 - categorical_accuracy: 0.9373
49888/60000 [=======================>......] - ETA: 16s - loss: 0.2025 - categorical_accuracy: 0.9374
49952/60000 [=======================>......] - ETA: 15s - loss: 0.2024 - categorical_accuracy: 0.9374
50016/60000 [========================>.....] - ETA: 15s - loss: 0.2023 - categorical_accuracy: 0.9375
50080/60000 [========================>.....] - ETA: 15s - loss: 0.2021 - categorical_accuracy: 0.9375
50144/60000 [========================>.....] - ETA: 15s - loss: 0.2021 - categorical_accuracy: 0.9375
50208/60000 [========================>.....] - ETA: 15s - loss: 0.2019 - categorical_accuracy: 0.9376
50272/60000 [========================>.....] - ETA: 15s - loss: 0.2018 - categorical_accuracy: 0.9376
50304/60000 [========================>.....] - ETA: 15s - loss: 0.2018 - categorical_accuracy: 0.9376
50368/60000 [========================>.....] - ETA: 15s - loss: 0.2018 - categorical_accuracy: 0.9376
50432/60000 [========================>.....] - ETA: 15s - loss: 0.2017 - categorical_accuracy: 0.9376
50496/60000 [========================>.....] - ETA: 15s - loss: 0.2015 - categorical_accuracy: 0.9376
50560/60000 [========================>.....] - ETA: 14s - loss: 0.2014 - categorical_accuracy: 0.9377
50624/60000 [========================>.....] - ETA: 14s - loss: 0.2012 - categorical_accuracy: 0.9378
50656/60000 [========================>.....] - ETA: 14s - loss: 0.2012 - categorical_accuracy: 0.9378
50720/60000 [========================>.....] - ETA: 14s - loss: 0.2010 - categorical_accuracy: 0.9378
50784/60000 [========================>.....] - ETA: 14s - loss: 0.2010 - categorical_accuracy: 0.9378
50848/60000 [========================>.....] - ETA: 14s - loss: 0.2011 - categorical_accuracy: 0.9378
50912/60000 [========================>.....] - ETA: 14s - loss: 0.2011 - categorical_accuracy: 0.9378
50976/60000 [========================>.....] - ETA: 14s - loss: 0.2010 - categorical_accuracy: 0.9379
51040/60000 [========================>.....] - ETA: 14s - loss: 0.2009 - categorical_accuracy: 0.9380
51104/60000 [========================>.....] - ETA: 14s - loss: 0.2006 - categorical_accuracy: 0.9380
51168/60000 [========================>.....] - ETA: 13s - loss: 0.2005 - categorical_accuracy: 0.9381
51232/60000 [========================>.....] - ETA: 13s - loss: 0.2004 - categorical_accuracy: 0.9381
51296/60000 [========================>.....] - ETA: 13s - loss: 0.2004 - categorical_accuracy: 0.9381
51360/60000 [========================>.....] - ETA: 13s - loss: 0.2003 - categorical_accuracy: 0.9382
51424/60000 [========================>.....] - ETA: 13s - loss: 0.2001 - categorical_accuracy: 0.9382
51488/60000 [========================>.....] - ETA: 13s - loss: 0.1999 - categorical_accuracy: 0.9383
51552/60000 [========================>.....] - ETA: 13s - loss: 0.1998 - categorical_accuracy: 0.9383
51616/60000 [========================>.....] - ETA: 13s - loss: 0.1997 - categorical_accuracy: 0.9383
51680/60000 [========================>.....] - ETA: 13s - loss: 0.1996 - categorical_accuracy: 0.9384
51744/60000 [========================>.....] - ETA: 13s - loss: 0.1994 - categorical_accuracy: 0.9384
51808/60000 [========================>.....] - ETA: 12s - loss: 0.1992 - categorical_accuracy: 0.9385
51872/60000 [========================>.....] - ETA: 12s - loss: 0.1990 - categorical_accuracy: 0.9386
51936/60000 [========================>.....] - ETA: 12s - loss: 0.1988 - categorical_accuracy: 0.9386
52000/60000 [=========================>....] - ETA: 12s - loss: 0.1988 - categorical_accuracy: 0.9387
52064/60000 [=========================>....] - ETA: 12s - loss: 0.1987 - categorical_accuracy: 0.9387
52128/60000 [=========================>....] - ETA: 12s - loss: 0.1986 - categorical_accuracy: 0.9387
52192/60000 [=========================>....] - ETA: 12s - loss: 0.1984 - categorical_accuracy: 0.9387
52256/60000 [=========================>....] - ETA: 12s - loss: 0.1983 - categorical_accuracy: 0.9388
52320/60000 [=========================>....] - ETA: 12s - loss: 0.1982 - categorical_accuracy: 0.9388
52384/60000 [=========================>....] - ETA: 12s - loss: 0.1981 - categorical_accuracy: 0.9388
52448/60000 [=========================>....] - ETA: 11s - loss: 0.1980 - categorical_accuracy: 0.9389
52480/60000 [=========================>....] - ETA: 11s - loss: 0.1979 - categorical_accuracy: 0.9389
52544/60000 [=========================>....] - ETA: 11s - loss: 0.1977 - categorical_accuracy: 0.9390
52608/60000 [=========================>....] - ETA: 11s - loss: 0.1975 - categorical_accuracy: 0.9391
52672/60000 [=========================>....] - ETA: 11s - loss: 0.1973 - categorical_accuracy: 0.9391
52736/60000 [=========================>....] - ETA: 11s - loss: 0.1971 - categorical_accuracy: 0.9391
52800/60000 [=========================>....] - ETA: 11s - loss: 0.1969 - categorical_accuracy: 0.9392
52864/60000 [=========================>....] - ETA: 11s - loss: 0.1967 - categorical_accuracy: 0.9393
52928/60000 [=========================>....] - ETA: 11s - loss: 0.1964 - categorical_accuracy: 0.9394
52992/60000 [=========================>....] - ETA: 11s - loss: 0.1963 - categorical_accuracy: 0.9394
53056/60000 [=========================>....] - ETA: 10s - loss: 0.1962 - categorical_accuracy: 0.9394
53120/60000 [=========================>....] - ETA: 10s - loss: 0.1961 - categorical_accuracy: 0.9394
53184/60000 [=========================>....] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9395
53248/60000 [=========================>....] - ETA: 10s - loss: 0.1958 - categorical_accuracy: 0.9395
53312/60000 [=========================>....] - ETA: 10s - loss: 0.1957 - categorical_accuracy: 0.9395
53344/60000 [=========================>....] - ETA: 10s - loss: 0.1956 - categorical_accuracy: 0.9395
53408/60000 [=========================>....] - ETA: 10s - loss: 0.1954 - categorical_accuracy: 0.9396
53472/60000 [=========================>....] - ETA: 10s - loss: 0.1952 - categorical_accuracy: 0.9396
53536/60000 [=========================>....] - ETA: 10s - loss: 0.1950 - categorical_accuracy: 0.9397
53600/60000 [=========================>....] - ETA: 10s - loss: 0.1951 - categorical_accuracy: 0.9397
53664/60000 [=========================>....] - ETA: 9s - loss: 0.1950 - categorical_accuracy: 0.9397 
53696/60000 [=========================>....] - ETA: 9s - loss: 0.1950 - categorical_accuracy: 0.9397
53760/60000 [=========================>....] - ETA: 9s - loss: 0.1949 - categorical_accuracy: 0.9398
53824/60000 [=========================>....] - ETA: 9s - loss: 0.1948 - categorical_accuracy: 0.9398
53888/60000 [=========================>....] - ETA: 9s - loss: 0.1947 - categorical_accuracy: 0.9398
53952/60000 [=========================>....] - ETA: 9s - loss: 0.1945 - categorical_accuracy: 0.9399
54016/60000 [==========================>...] - ETA: 9s - loss: 0.1944 - categorical_accuracy: 0.9399
54080/60000 [==========================>...] - ETA: 9s - loss: 0.1944 - categorical_accuracy: 0.9399
54144/60000 [==========================>...] - ETA: 9s - loss: 0.1942 - categorical_accuracy: 0.9400
54208/60000 [==========================>...] - ETA: 9s - loss: 0.1942 - categorical_accuracy: 0.9400
54240/60000 [==========================>...] - ETA: 9s - loss: 0.1942 - categorical_accuracy: 0.9400
54304/60000 [==========================>...] - ETA: 8s - loss: 0.1940 - categorical_accuracy: 0.9400
54336/60000 [==========================>...] - ETA: 8s - loss: 0.1939 - categorical_accuracy: 0.9400
54400/60000 [==========================>...] - ETA: 8s - loss: 0.1937 - categorical_accuracy: 0.9401
54464/60000 [==========================>...] - ETA: 8s - loss: 0.1936 - categorical_accuracy: 0.9401
54528/60000 [==========================>...] - ETA: 8s - loss: 0.1936 - categorical_accuracy: 0.9402
54560/60000 [==========================>...] - ETA: 8s - loss: 0.1935 - categorical_accuracy: 0.9402
54624/60000 [==========================>...] - ETA: 8s - loss: 0.1933 - categorical_accuracy: 0.9403
54688/60000 [==========================>...] - ETA: 8s - loss: 0.1932 - categorical_accuracy: 0.9403
54752/60000 [==========================>...] - ETA: 8s - loss: 0.1932 - categorical_accuracy: 0.9403
54816/60000 [==========================>...] - ETA: 8s - loss: 0.1930 - categorical_accuracy: 0.9404
54880/60000 [==========================>...] - ETA: 8s - loss: 0.1929 - categorical_accuracy: 0.9404
54912/60000 [==========================>...] - ETA: 7s - loss: 0.1929 - categorical_accuracy: 0.9405
54976/60000 [==========================>...] - ETA: 7s - loss: 0.1927 - categorical_accuracy: 0.9405
55040/60000 [==========================>...] - ETA: 7s - loss: 0.1926 - categorical_accuracy: 0.9405
55104/60000 [==========================>...] - ETA: 7s - loss: 0.1924 - categorical_accuracy: 0.9406
55136/60000 [==========================>...] - ETA: 7s - loss: 0.1925 - categorical_accuracy: 0.9406
55200/60000 [==========================>...] - ETA: 7s - loss: 0.1923 - categorical_accuracy: 0.9407
55264/60000 [==========================>...] - ETA: 7s - loss: 0.1922 - categorical_accuracy: 0.9407
55296/60000 [==========================>...] - ETA: 7s - loss: 0.1921 - categorical_accuracy: 0.9407
55360/60000 [==========================>...] - ETA: 7s - loss: 0.1919 - categorical_accuracy: 0.9408
55424/60000 [==========================>...] - ETA: 7s - loss: 0.1919 - categorical_accuracy: 0.9408
55488/60000 [==========================>...] - ETA: 7s - loss: 0.1919 - categorical_accuracy: 0.9408
55552/60000 [==========================>...] - ETA: 6s - loss: 0.1918 - categorical_accuracy: 0.9409
55616/60000 [==========================>...] - ETA: 6s - loss: 0.1918 - categorical_accuracy: 0.9408
55680/60000 [==========================>...] - ETA: 6s - loss: 0.1916 - categorical_accuracy: 0.9409
55744/60000 [==========================>...] - ETA: 6s - loss: 0.1915 - categorical_accuracy: 0.9409
55808/60000 [==========================>...] - ETA: 6s - loss: 0.1913 - categorical_accuracy: 0.9410
55872/60000 [==========================>...] - ETA: 6s - loss: 0.1911 - categorical_accuracy: 0.9410
55936/60000 [==========================>...] - ETA: 6s - loss: 0.1912 - categorical_accuracy: 0.9411
56000/60000 [===========================>..] - ETA: 6s - loss: 0.1912 - categorical_accuracy: 0.9411
56064/60000 [===========================>..] - ETA: 6s - loss: 0.1911 - categorical_accuracy: 0.9411
56128/60000 [===========================>..] - ETA: 6s - loss: 0.1911 - categorical_accuracy: 0.9411
56192/60000 [===========================>..] - ETA: 5s - loss: 0.1910 - categorical_accuracy: 0.9412
56256/60000 [===========================>..] - ETA: 5s - loss: 0.1909 - categorical_accuracy: 0.9412
56320/60000 [===========================>..] - ETA: 5s - loss: 0.1908 - categorical_accuracy: 0.9412
56384/60000 [===========================>..] - ETA: 5s - loss: 0.1909 - categorical_accuracy: 0.9413
56448/60000 [===========================>..] - ETA: 5s - loss: 0.1907 - categorical_accuracy: 0.9413
56512/60000 [===========================>..] - ETA: 5s - loss: 0.1907 - categorical_accuracy: 0.9413
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1906 - categorical_accuracy: 0.9413
56640/60000 [===========================>..] - ETA: 5s - loss: 0.1905 - categorical_accuracy: 0.9413
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1904 - categorical_accuracy: 0.9414
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1903 - categorical_accuracy: 0.9414
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1901 - categorical_accuracy: 0.9415
56864/60000 [===========================>..] - ETA: 4s - loss: 0.1901 - categorical_accuracy: 0.9415
56896/60000 [===========================>..] - ETA: 4s - loss: 0.1900 - categorical_accuracy: 0.9415
56960/60000 [===========================>..] - ETA: 4s - loss: 0.1899 - categorical_accuracy: 0.9416
56992/60000 [===========================>..] - ETA: 4s - loss: 0.1899 - categorical_accuracy: 0.9416
57056/60000 [===========================>..] - ETA: 4s - loss: 0.1897 - categorical_accuracy: 0.9416
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1896 - categorical_accuracy: 0.9416
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1896 - categorical_accuracy: 0.9416
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1894 - categorical_accuracy: 0.9417
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1893 - categorical_accuracy: 0.9417
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1892 - categorical_accuracy: 0.9417
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1890 - categorical_accuracy: 0.9418
57472/60000 [===========================>..] - ETA: 3s - loss: 0.1889 - categorical_accuracy: 0.9418
57536/60000 [===========================>..] - ETA: 3s - loss: 0.1887 - categorical_accuracy: 0.9418
57600/60000 [===========================>..] - ETA: 3s - loss: 0.1887 - categorical_accuracy: 0.9418
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1885 - categorical_accuracy: 0.9419
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1883 - categorical_accuracy: 0.9420
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1882 - categorical_accuracy: 0.9420
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1881 - categorical_accuracy: 0.9420
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1880 - categorical_accuracy: 0.9421
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1878 - categorical_accuracy: 0.9421
58048/60000 [============================>.] - ETA: 3s - loss: 0.1877 - categorical_accuracy: 0.9422
58112/60000 [============================>.] - ETA: 2s - loss: 0.1876 - categorical_accuracy: 0.9422
58176/60000 [============================>.] - ETA: 2s - loss: 0.1874 - categorical_accuracy: 0.9422
58208/60000 [============================>.] - ETA: 2s - loss: 0.1873 - categorical_accuracy: 0.9423
58272/60000 [============================>.] - ETA: 2s - loss: 0.1872 - categorical_accuracy: 0.9423
58304/60000 [============================>.] - ETA: 2s - loss: 0.1871 - categorical_accuracy: 0.9423
58368/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9424
58432/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9424
58464/60000 [============================>.] - ETA: 2s - loss: 0.1867 - categorical_accuracy: 0.9424
58528/60000 [============================>.] - ETA: 2s - loss: 0.1866 - categorical_accuracy: 0.9425
58560/60000 [============================>.] - ETA: 2s - loss: 0.1865 - categorical_accuracy: 0.9425
58624/60000 [============================>.] - ETA: 2s - loss: 0.1865 - categorical_accuracy: 0.9425
58688/60000 [============================>.] - ETA: 2s - loss: 0.1866 - categorical_accuracy: 0.9425
58720/60000 [============================>.] - ETA: 2s - loss: 0.1865 - categorical_accuracy: 0.9425
58784/60000 [============================>.] - ETA: 1s - loss: 0.1865 - categorical_accuracy: 0.9425
58848/60000 [============================>.] - ETA: 1s - loss: 0.1864 - categorical_accuracy: 0.9425
58912/60000 [============================>.] - ETA: 1s - loss: 0.1863 - categorical_accuracy: 0.9426
58976/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9426
59040/60000 [============================>.] - ETA: 1s - loss: 0.1859 - categorical_accuracy: 0.9427
59104/60000 [============================>.] - ETA: 1s - loss: 0.1858 - categorical_accuracy: 0.9427
59168/60000 [============================>.] - ETA: 1s - loss: 0.1857 - categorical_accuracy: 0.9427
59232/60000 [============================>.] - ETA: 1s - loss: 0.1856 - categorical_accuracy: 0.9428
59296/60000 [============================>.] - ETA: 1s - loss: 0.1854 - categorical_accuracy: 0.9428
59360/60000 [============================>.] - ETA: 1s - loss: 0.1853 - categorical_accuracy: 0.9429
59424/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9429
59488/60000 [============================>.] - ETA: 0s - loss: 0.1851 - categorical_accuracy: 0.9429
59552/60000 [============================>.] - ETA: 0s - loss: 0.1851 - categorical_accuracy: 0.9429
59616/60000 [============================>.] - ETA: 0s - loss: 0.1849 - categorical_accuracy: 0.9430
59680/60000 [============================>.] - ETA: 0s - loss: 0.1847 - categorical_accuracy: 0.9430
59744/60000 [============================>.] - ETA: 0s - loss: 0.1846 - categorical_accuracy: 0.9431
59808/60000 [============================>.] - ETA: 0s - loss: 0.1845 - categorical_accuracy: 0.9431
59840/60000 [============================>.] - ETA: 0s - loss: 0.1844 - categorical_accuracy: 0.9431
59904/60000 [============================>.] - ETA: 0s - loss: 0.1843 - categorical_accuracy: 0.9431
59968/60000 [============================>.] - ETA: 0s - loss: 0.1842 - categorical_accuracy: 0.9432
60000/60000 [==============================] - 97s 2ms/step - loss: 0.1842 - categorical_accuracy: 0.9432 - val_loss: 0.0529 - val_categorical_accuracy: 0.9819

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 16s
  256/10000 [..............................] - ETA: 4s 
  448/10000 [>.............................] - ETA: 3s
  640/10000 [>.............................] - ETA: 3s
  832/10000 [=>............................] - ETA: 2s
 1056/10000 [==>...........................] - ETA: 2s
 1248/10000 [==>...........................] - ETA: 2s
 1440/10000 [===>..........................] - ETA: 2s
 1632/10000 [===>..........................] - ETA: 2s
 1824/10000 [====>.........................] - ETA: 2s
 2048/10000 [=====>........................] - ETA: 2s
 2240/10000 [=====>........................] - ETA: 2s
 2432/10000 [======>.......................] - ETA: 2s
 2656/10000 [======>.......................] - ETA: 2s
 2880/10000 [=======>......................] - ETA: 1s
 3104/10000 [========>.....................] - ETA: 1s
 3296/10000 [========>.....................] - ETA: 1s
 3456/10000 [=========>....................] - ETA: 1s
 3648/10000 [=========>....................] - ETA: 1s
 3840/10000 [==========>...................] - ETA: 1s
 4032/10000 [===========>..................] - ETA: 1s
 4256/10000 [===========>..................] - ETA: 1s
 4448/10000 [============>.................] - ETA: 1s
 4640/10000 [============>.................] - ETA: 1s
 4864/10000 [=============>................] - ETA: 1s
 5056/10000 [==============>...............] - ETA: 1s
 5248/10000 [==============>...............] - ETA: 1s
 5472/10000 [===============>..............] - ETA: 1s
 5696/10000 [================>.............] - ETA: 1s
 5888/10000 [================>.............] - ETA: 1s
 6080/10000 [=================>............] - ETA: 1s
 6272/10000 [=================>............] - ETA: 1s
 6464/10000 [==================>...........] - ETA: 0s
 6656/10000 [==================>...........] - ETA: 0s
 6880/10000 [===================>..........] - ETA: 0s
 7072/10000 [====================>.........] - ETA: 0s
 7264/10000 [====================>.........] - ETA: 0s
 7456/10000 [=====================>........] - ETA: 0s
 7648/10000 [=====================>........] - ETA: 0s
 7840/10000 [======================>.......] - ETA: 0s
 8032/10000 [=======================>......] - ETA: 0s
 8224/10000 [=======================>......] - ETA: 0s
 8416/10000 [========================>.....] - ETA: 0s
 8608/10000 [========================>.....] - ETA: 0s
 8800/10000 [=========================>....] - ETA: 0s
 8992/10000 [=========================>....] - ETA: 0s
 9152/10000 [==========================>...] - ETA: 0s
 9344/10000 [===========================>..] - ETA: 0s
 9536/10000 [===========================>..] - ETA: 0s
 9728/10000 [============================>.] - ETA: 0s
 9952/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 272us/step
[[5.5796958e-09 1.4484580e-09 1.2264626e-07 ... 9.9999976e-01
  1.0534370e-09 1.2436136e-07]
 [5.6004035e-07 1.0691504e-05 9.9998760e-01 ... 5.8054486e-09
  7.6055912e-08 4.5325917e-12]
 [3.1461079e-07 9.9994659e-01 5.3512381e-06 ... 4.9588753e-06
  5.1851189e-06 1.2315519e-07]
 ...
 [1.9999676e-10 1.0496836e-07 1.0316559e-09 ... 2.9856454e-07
  7.4767378e-08 3.3626588e-06]
 [4.0388777e-08 8.8160429e-10 1.5699929e-10 ... 2.7730183e-09
  1.6370956e-05 4.2520367e-07]
 [1.9698657e-06 2.3962909e-07 7.4139189e-06 ... 4.4028006e-10
  8.4298740e-08 2.5101108e-09]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.052882509162032514, 'accuracy_test:': 0.9818999767303467}

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
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   699882b..61c2162  master     -> origin/master
Updating 699882b..61c2162
Fast-forward
 error_list/20200516/list_log_benchmark_20200516.md |  182 +-
 error_list/20200516/list_log_jupyter_20200516.md   | 1749 ++++++++++----------
 error_list/20200516/list_log_testall_20200516.md   |  386 +++--
 3 files changed, 1184 insertions(+), 1133 deletions(-)
[master c9545f5] ml_store
 1 file changed, 1415 insertions(+)
To github.com:arita37/mlmodels_store.git
   61c2162..c9545f5  master -> master





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
{'loss': 0.3897261992096901, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-16 16:33:12.598399: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
Already up to date.
[master fcc8496] ml_store
 1 file changed, 235 insertions(+)
To github.com:arita37/mlmodels_store.git
   c9545f5..fcc8496  master -> master





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
[master 40eecd3] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   fcc8496..40eecd3  master -> master





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
 40%|████      | 2/5 [00:21<00:31, 10.63s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.8166110884987331, 'learning_rate': 0.06850638074628712, 'min_data_in_leaf': 25, 'num_leaves': 36} and reward: 0.3958
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea!\xad\x93\xd4\xe5\x8bX\r\x00\x00\x00learning_rateq\x02G?\xb1\x89\xa2X\xdfd\x80X\x10\x00\x00\x00min_data_in_leafq\x03K\x19X\n\x00\x00\x00num_leavesq\x04K$u.' and reward: 0.3958
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea!\xad\x93\xd4\xe5\x8bX\r\x00\x00\x00learning_rateq\x02G?\xb1\x89\xa2X\xdfd\x80X\x10\x00\x00\x00min_data_in_leafq\x03K\x19X\n\x00\x00\x00num_leavesq\x04K$u.' and reward: 0.3958
 60%|██████    | 3/5 [00:49<00:31, 15.77s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9349521958751097, 'learning_rate': 0.028443257852805163, 'min_data_in_leaf': 27, 'num_leaves': 38} and reward: 0.3922
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xeb \xde\x13lfX\r\x00\x00\x00learning_rateq\x02G?\x9d :\xb9\x13\xf6\x11X\x10\x00\x00\x00min_data_in_leafq\x03K\x1bX\n\x00\x00\x00num_leavesq\x04K&u.' and reward: 0.3922
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xeb \xde\x13lfX\r\x00\x00\x00learning_rateq\x02G?\x9d :\xb9\x13\xf6\x11X\x10\x00\x00\x00min_data_in_leafq\x03K\x1bX\n\x00\x00\x00num_leavesq\x04K&u.' and reward: 0.3922
 80%|████████  | 4/5 [01:12<00:18, 18.08s/it] 80%|████████  | 4/5 [01:12<00:18, 18.12s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8229482962138884, 'learning_rate': 0.008784130666063221, 'min_data_in_leaf': 3, 'num_leaves': 42} and reward: 0.3872
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeaU\x97\xaaQ3\xc9X\r\x00\x00\x00learning_rateq\x02G?\x81\xfdj\x0fy\xec\tX\x10\x00\x00\x00min_data_in_leafq\x03K\x03X\n\x00\x00\x00num_leavesq\x04K*u.' and reward: 0.3872
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeaU\x97\xaaQ3\xc9X\r\x00\x00\x00learning_rateq\x02G?\x81\xfdj\x0fy\xec\tX\x10\x00\x00\x00min_data_in_leafq\x03K\x03X\n\x00\x00\x00num_leavesq\x04K*u.' and reward: 0.3872
Time for Gradient Boosting hyperparameter optimization: 100.42218446731567
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.8166110884987331, 'learning_rate': 0.06850638074628712, 'min_data_in_leaf': 25, 'num_leaves': 36}
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
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
 40%|████      | 2/5 [00:52<01:19, 26.50s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.42651126513473625, 'embedding_size_factor': 1.0454813723997032, 'layers.choice': 2, 'learning_rate': 0.008773865854121416, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1.604321469824157e-12} and reward: 0.388
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdbK\xf5\xe7\xc8F\x14X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\xbaJ\xac\xf0\x8a\xf2X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x81\xf8\x08W99\xf7X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=|98\xb9\r\x88@u.' and reward: 0.388
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdbK\xf5\xe7\xc8F\x14X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\xbaJ\xac\xf0\x8a\xf2X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x81\xf8\x08W99\xf7X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=|98\xb9\r\x88@u.' and reward: 0.388
 60%|██████    | 3/5 [01:46<01:09, 34.60s/it] 60%|██████    | 3/5 [01:46<01:11, 35.50s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.313065906041753, 'embedding_size_factor': 0.8678335831648469, 'layers.choice': 2, 'learning_rate': 0.00041901109664739143, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 4.053431888449042e-06} and reward: 0.3556
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\tE\x94\xfcH\xb6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeb\xc5J\xefB\x08NX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?;u\xd6\xf4\xeeT\xb8X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xd1\x00V\xdf\x99\x90,u.' and reward: 0.3556
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\tE\x94\xfcH\xb6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeb\xc5J\xefB\x08NX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?;u\xd6\xf4\xeeT\xb8X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xd1\x00V\xdf\x99\x90,u.' and reward: 0.3556
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 161.18317675590515
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -146.01s of remaining time.
Ensemble size: 35
Ensemble weights: 
[0.6        0.         0.02857143 0.25714286 0.         0.05714286
 0.05714286]
	0.4004	 = Validation accuracy score
	1.69s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 267.75s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7ff5953a44e0>

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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   40eecd3..0511578  master     -> origin/master
Updating 40eecd3..0511578
Fast-forward
 error_list/20200516/list_log_import_20200516.md  |    2 +-
 error_list/20200516/list_log_json_20200516.md    | 1146 +++++++++++-----------
 error_list/20200516/list_log_testall_20200516.md |  386 ++++----
 3 files changed, 747 insertions(+), 787 deletions(-)
[master fbd2041] ml_store
 1 file changed, 252 insertions(+)
To github.com:arita37/mlmodels_store.git
   0511578..fbd2041  master -> master





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
[master 4ee765c] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   fbd2041..4ee765c  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.57it/s, avg_epoch_loss=5.23]
INFO:root:Epoch[0] Elapsed time 2.805 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.229468
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.229468250274659 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f5a0ad5b518>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f5a0ad5b518>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 105.88it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1078.7379557291667,
    "abs_error": 373.22406005859375,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.47296389425763,
    "sMAPE": 0.515680626106712,
    "MSIS": 98.91855253497815,
    "QuantileLoss[0.5]": 373.22403717041016,
    "Coverage[0.5]": 1.0,
    "RMSE": 32.84414644543479,
    "NRMSE": 0.6914557146407324,
    "ND": 0.654779052734375,
    "wQuantileLoss[0.5]": 0.654779012579667,
    "mean_wQuantileLoss": 0.654779012579667,
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
100%|██████████| 10/10 [00:01<00:00,  8.00it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.251 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f59c77d37f0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f59c77d37f0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 153.69it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.11it/s, avg_epoch_loss=5.23]
INFO:root:Epoch[0] Elapsed time 1.956 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.230885
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.230885362625122 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f59c77d37f0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f59c77d37f0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 123.87it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 251.4499715169271,
    "abs_error": 173.6564178466797,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.1506387108953668,
    "sMAPE": 0.28853837477633465,
    "MSIS": 46.02555247997349,
    "QuantileLoss[0.5]": 173.65642929077148,
    "Coverage[0.5]": 0.75,
    "RMSE": 15.857174134029275,
    "NRMSE": 0.3338352449269321,
    "ND": 0.30466038218715735,
    "wQuantileLoss[0.5]": 0.30466040226451135,
    "mean_wQuantileLoss": 0.30466040226451135,
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
 30%|███       | 3/10 [00:13<00:31,  4.48s/it, avg_epoch_loss=6.92] 60%|██████    | 6/10 [00:25<00:17,  4.32s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:37<00:04,  4.21s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:41<00:00,  4.12s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 41.152 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.864166
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.864166021347046 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f59c76daf60>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f59c76daf60>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 150.16it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 53670.182291666664,
    "abs_error": 2735.45263671875,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 18.12497191096226,
    "sMAPE": 1.4148195345383812,
    "MSIS": 724.9988505558739,
    "QuantileLoss[0.5]": 2735.4524841308594,
    "Coverage[0.5]": 1.0,
    "RMSE": 231.66825913721254,
    "NRMSE": 4.877226508151843,
    "ND": 4.799039713541666,
    "wQuantileLoss[0.5]": 4.799039445843613,
    "mean_wQuantileLoss": 4.799039445843613,
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
100%|██████████| 10/10 [00:00<00:00, 55.49it/s, avg_epoch_loss=5.19]
INFO:root:Epoch[0] Elapsed time 0.181 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.192956
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.1929563045501705 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f59c7569f60>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f59c7569f60>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 135.13it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 495.0712483723958,
    "abs_error": 184.66519165039062,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.2235822937189187,
    "sMAPE": 0.3124451142204368,
    "MSIS": 48.94328366043909,
    "QuantileLoss[0.5]": 184.66517639160156,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 22.25019659176961,
    "NRMSE": 0.46842519140567596,
    "ND": 0.3239740204392818,
    "wQuantileLoss[0.5]": 0.32397399366947643,
    "mean_wQuantileLoss": 0.32397399366947643,
    "MAE_Coverage": 0.16666666666666663
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
100%|██████████| 10/10 [00:01<00:00,  7.57it/s, avg_epoch_loss=160]
INFO:root:Epoch[0] Elapsed time 1.322 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=160.137221
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 160.13722096982082 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f59c7764f98>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f59c7764f98>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 128.89it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 721.6386473701223,
    "abs_error": 263.04314901123786,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.7429107063314655,
    "sMAPE": 0.5744170818242088,
    "MSIS": 69.71642825325863,
    "QuantileLoss[0.5]": 263.04314901123786,
    "Coverage[0.5]": 0.08333333333333333,
    "RMSE": 26.86333276736381,
    "NRMSE": 0.5655438477339749,
    "ND": 0.4614792087916454,
    "wQuantileLoss[0.5]": 0.4614792087916454,
    "mean_wQuantileLoss": 0.4614792087916454,
    "MAE_Coverage": 0.4166666666666667
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
 10%|█         | 1/10 [02:02<18:21, 122.43s/it, avg_epoch_loss=0.653] 20%|██        | 2/10 [05:10<18:56, 142.11s/it, avg_epoch_loss=0.635] 30%|███       | 3/10 [08:15<18:05, 155.11s/it, avg_epoch_loss=0.618] 40%|████      | 4/10 [11:27<16:36, 166.05s/it, avg_epoch_loss=0.6]   50%|█████     | 5/10 [15:06<15:09, 181.91s/it, avg_epoch_loss=0.582] 60%|██████    | 6/10 [18:10<12:10, 182.65s/it, avg_epoch_loss=0.565] 70%|███████   | 7/10 [21:16<09:10, 183.63s/it, avg_epoch_loss=0.548] 80%|████████  | 8/10 [24:37<06:17, 188.81s/it, avg_epoch_loss=0.531] 90%|█████████ | 9/10 [28:19<03:18, 198.84s/it, avg_epoch_loss=0.516]100%|██████████| 10/10 [31:46<00:00, 201.33s/it, avg_epoch_loss=0.502]100%|██████████| 10/10 [31:47<00:00, 190.70s/it, avg_epoch_loss=0.502]
INFO:root:Epoch[0] Elapsed time 1907.032 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.501657
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.5016570329666138 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f59c7552080>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f59c7552080>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00,  4.92it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 143.64437866210938,
    "abs_error": 113.09156799316406,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 0.7493390547977266,
    "sMAPE": 0.20097564209453803,
    "MSIS": 29.97356785373142,
    "QuantileLoss[0.5]": 113.0915756225586,
    "Coverage[0.5]": 0.4166666666666667,
    "RMSE": 11.985173284609171,
    "NRMSE": 0.2523194375707194,
    "ND": 0.19840625963712993,
    "wQuantileLoss[0.5]": 0.19840627302203262,
    "mean_wQuantileLoss": 0.19840627302203262,
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
From github.com:arita37/mlmodels_store
   4ee765c..9188765  master     -> origin/master
Updating 4ee765c..9188765
Fast-forward
 error_list/20200516/list_log_benchmark_20200516.md |  182 ++--
 error_list/20200516/list_log_json_20200516.md      | 1146 ++++++++++----------
 2 files changed, 659 insertions(+), 669 deletions(-)
[master 0bf6ec1] ml_store
 1 file changed, 506 insertions(+)
To github.com:arita37/mlmodels_store.git
   9188765..0bf6ec1  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7fbd4fdc75c0> 

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
[master d2983fd] ml_store
 1 file changed, 109 insertions(+)
To github.com:arita37/mlmodels_store.git
   0bf6ec1..d2983fd  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 1.44682180e+00  8.07455917e-01  1.49810818e+00  3.12238689e-01
  -6.82430193e-01 -1.93321640e-01  2.88078167e-01 -2.07680202e+00
   9.47501167e-01 -3.00976154e-01]
 [ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
   1.10417433e+00 -4.22856961e-01]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 1.02242019e+00  1.85300949e+00  6.44353666e-01  1.42251373e-01
   1.15080755e+00  5.13505480e-01 -4.59942831e-01  3.72456852e-01
  -1.48489803e-01  3.71670291e-01]
 [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 7.88018455e-01  3.01960045e-01  7.00982122e-01 -3.94689681e-01
  -1.20376927e+00 -1.17181338e+00  7.55392029e-01  9.84012237e-01
  -5.59681422e-01 -1.98937450e-01]
 [ 8.98917161e-01  5.57439453e-01 -7.58067329e-01  1.81038744e-01
   8.41467206e-01  1.10717545e+00  6.93366226e-01  1.44287693e+00
  -5.39681562e-01 -8.08847196e-01]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 5.69983848e-01 -5.33020326e-01 -1.75458969e-01 -1.42655542e+00
   6.06604307e-01  1.76795995e+00 -1.15985185e-01 -4.75372875e-01
   4.77610182e-01 -9.33914656e-01]
 [ 7.83440538e-01 -5.11884476e-02  8.24584625e-01 -7.25597119e-01
   9.31717197e-01 -8.67768678e-01  3.03085711e+00 -1.35977326e-01
  -7.97269785e-01  6.54580153e-01]
 [ 1.12062155e+00 -7.02920403e-01 -1.22957425e+00  7.25550518e-01
  -1.18013412e+00 -3.24204219e-01  1.10223673e+00  8.14343129e-01
   7.80469930e-01  1.10861676e+00]
 [ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
  -4.26428618e-01 -8.39988915e-01]
 [ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 8.58774962e-01  2.29371761e+00 -1.47023709e+00 -8.30010986e-01
  -6.72049816e-01 -1.01951985e+00  5.99213235e-01 -2.14653842e-01
   1.02124813e+00  6.06403944e-01]
 [ 1.18468624e+00 -1.00016919e+00 -5.93843067e-01  1.04499441e+00
   9.65482331e-01  6.08514698e-01 -6.25342001e-01 -6.93286967e-02
  -1.08392067e-01 -3.43900709e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 1.64661853e+00 -1.52568032e+00 -6.06998398e-01  7.95026094e-01
   1.08480038e+00 -3.74438319e-01  4.29526140e-01  1.34048197e-01
   1.20205486e+00  1.06222724e-01]
 [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
   1.40925339e+00  9.66539250e-01]
 [ 8.95510508e-01  9.20615118e-01  7.94528240e-01 -3.53679249e-02
   8.78099103e-01  2.11060505e+00 -1.02188594e+00 -1.30653407e+00
   7.63804802e-02 -1.87316098e+00]
 [ 3.54133613e-01  2.11124755e-01  9.21450069e-01  1.65275673e-02
   9.03945451e-01  1.77187720e-01  9.54250872e-02 -1.11647002e+00
   8.09271010e-02  6.07501958e-02]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f5a51cabeb8>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f5a6c01d710> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 1.17867274e+00 -5.99804531e-01 -6.94693595e-01  1.12341216e+00
   1.17899425e+00  3.05267040e-01  1.33526763e-02  1.38877940e+00
  -6.61344243e-01  6.21803504e-01]
 [ 1.22867367e+00  1.34373116e-01 -1.82420406e-01 -2.68371304e-01
  -1.73963799e+00 -1.31675626e-01 -9.26871939e-01  1.01855247e+00
   1.23055820e+00 -4.91125138e-01]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]
 [ 6.23688521e-01  1.20660790e+00  9.03999174e-01 -2.82863552e-01
  -1.18913787e+00 -2.66326884e-01  1.42361443e+00  1.06897162e+00
   4.03714310e-02  1.57546791e+00]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 8.57296491e-01  9.56121704e-01 -8.26097432e-01 -7.05840507e-01
   1.13872896e+00  1.19268607e+00  2.82675712e-01 -2.37941936e-01
   1.15528789e+00  6.21082701e-01]
 [ 8.58774962e-01  2.29371761e+00 -1.47023709e+00 -8.30010986e-01
  -6.72049816e-01 -1.01951985e+00  5.99213235e-01 -2.14653842e-01
   1.02124813e+00  6.06403944e-01]
 [ 1.44682180e+00  8.07455917e-01  1.49810818e+00  3.12238689e-01
  -6.82430193e-01 -1.93321640e-01  2.88078167e-01 -2.07680202e+00
   9.47501167e-01 -3.00976154e-01]
 [ 1.27991386e+00 -8.71422066e-01 -3.24032329e-01 -8.64829941e-01
  -9.68539694e-01  6.08749082e-01  5.07984337e-01  5.61638097e-01
   1.51475038e+00 -1.51107661e+00]
 [ 8.88838813e-01  1.03368687e+00 -4.97025792e-02  8.08844360e-01
   8.14051347e-01  1.78975468e+00  1.14690038e+00  4.51284016e-01
  -1.68405999e+00  4.66643267e-01]
 [ 1.32720112e+00 -1.61198320e-01  6.02450901e-01 -2.86384915e-01
  -5.78962302e-01 -8.70887650e-01  1.37975819e+00  5.01429590e-01
  -4.78614074e-01 -8.92646674e-01]
 [ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 1.46893146e+00 -1.47115693e+00  5.85910431e-01 -8.30171895e-01
   1.03345052e+00 -8.80577600e-01 -9.55425262e-01 -2.79097722e-01
   1.62284909e+00  2.06578332e+00]
 [ 1.21619061e+00 -1.90005215e-02  8.60891241e-01 -2.26760192e-01
  -1.36419132e+00 -1.56450785e+00  1.63169151e+00  9.31255679e-01
   9.49808815e-01 -8.80189065e-01]
 [ 1.34728643e+00 -3.64538050e-01  8.07509886e-02 -4.59717681e-01
  -8.89487596e-01  1.70548352e+00  9.49961101e-02  2.40505552e-01
  -9.99426501e-01 -7.67803746e-01]
 [ 5.69983848e-01 -5.33020326e-01 -1.75458969e-01 -1.42655542e+00
   6.06604307e-01  1.76795995e+00 -1.15985185e-01 -4.75372875e-01
   4.77610182e-01 -9.33914656e-01]
 [ 1.16777676e+00 -6.65754518e-01 -1.23312074e+00 -1.67419581e+00
   1.01313574e+00  8.25029824e-01 -1.20464572e-01 -4.98213564e-01
  -3.10984978e-01 -1.18231813e+00]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 7.83440538e-01 -5.11884476e-02  8.24584625e-01 -7.25597119e-01
   9.31717197e-01 -8.67768678e-01  3.03085711e+00 -1.35977326e-01
  -7.97269785e-01  6.54580153e-01]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
   7.18290736e-01 -5.30119800e-01]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]]
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
[[ 1.06702918e+00 -4.29142278e-01  3.50167159e-01  1.20845633e+00
   7.51480619e-01  1.11570180e+00 -4.79157099e-01  8.40861558e-01
  -1.02887218e-01  1.71647264e-02]
 [ 3.54133613e-01  2.11124755e-01  9.21450069e-01  1.65275673e-02
   9.03945451e-01  1.77187720e-01  9.54250872e-02 -1.11647002e+00
   8.09271010e-02  6.07501958e-02]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 7.90323893e-01  1.61336137e+00 -2.09424782e+00 -3.74804687e-01
   9.15884042e-01 -7.49969617e-01  3.10272288e-01  2.05462410e+00
   5.34095368e-02 -2.28765829e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 1.66752297e+00  1.22372221e+00 -4.59930104e-01 -5.93679025e-02
  -4.93856997e-01  1.44898940e+00 -1.18110317e+00 -4.77580855e-01
   2.59999942e-02 -7.90799954e-01]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]
 [ 9.47814113e-01 -1.13379204e+00  6.40985866e-01 -1.90548298e-01
  -1.23912256e+00  2.33339126e-01 -3.16901197e-01  4.34998324e-01
   9.10423603e-01  1.21987438e+00]
 [ 7.88018455e-01  3.01960045e-01  7.00982122e-01 -3.94689681e-01
  -1.20376927e+00 -1.17181338e+00  7.55392029e-01  9.84012237e-01
  -5.59681422e-01 -1.98937450e-01]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]
 [ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
   7.18290736e-01 -5.30119800e-01]
 [ 1.27991386e+00 -8.71422066e-01 -3.24032329e-01 -8.64829941e-01
  -9.68539694e-01  6.08749082e-01  5.07984337e-01  5.61638097e-01
   1.51475038e+00 -1.51107661e+00]
 [ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
  -4.26428618e-01 -8.39988915e-01]
 [ 9.26869810e-01  3.92334911e-01 -4.23478297e-01  4.48380651e-01
  -1.09230828e+00  1.12532350e+00 -9.48439656e-01  1.04053390e-01
   5.28003422e-01  1.00796648e+00]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 6.92114488e-01 -6.06524918e-02  2.05635552e+00 -2.41350300e+00
   1.17456965e+00 -1.77756638e+00 -2.81736269e-01 -7.77858827e-01
   1.11584111e+00  1.76024923e+00]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 6.21530991e-01 -1.50957268e+00 -1.01932039e-01 -1.08071069e+00
  -1.13742855e+00  7.25474004e-01  7.98063795e-01 -3.91782562e-02
  -2.28754171e-01  7.43356544e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 1.34740825e+00  7.33023232e-01  8.38634747e-01 -1.89881206e+00
  -5.42459922e-01 -1.11711069e+00 -1.09715436e+00 -5.08972278e-01
  -1.66485955e-01 -1.03918232e+00]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]]
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
[master 4958fd9] ml_store
 1 file changed, 322 insertions(+)
To github.com:arita37/mlmodels_store.git
   d2983fd..4958fd9  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139778956251088
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139778956250864
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139778956249632
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139778956249184
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139778956248680
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139778956248344

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
grad_step = 000000, loss = 1.150956
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.967846
grad_step = 000002, loss = 0.853825
grad_step = 000003, loss = 0.712181
grad_step = 000004, loss = 0.539903
grad_step = 000005, loss = 0.358151
grad_step = 000006, loss = 0.217568
grad_step = 000007, loss = 0.202084
grad_step = 000008, loss = 0.237995
grad_step = 000009, loss = 0.211636
grad_step = 000010, loss = 0.140054
grad_step = 000011, loss = 0.077714
grad_step = 000012, loss = 0.052540
grad_step = 000013, loss = 0.053797
grad_step = 000014, loss = 0.061622
grad_step = 000015, loss = 0.062515
grad_step = 000016, loss = 0.053013
grad_step = 000017, loss = 0.035761
grad_step = 000018, loss = 0.018369
grad_step = 000019, loss = 0.008504
grad_step = 000020, loss = 0.009856
grad_step = 000021, loss = 0.018382
grad_step = 000022, loss = 0.024527
grad_step = 000023, loss = 0.023048
grad_step = 000024, loss = 0.016745
grad_step = 000025, loss = 0.011775
grad_step = 000026, loss = 0.010775
grad_step = 000027, loss = 0.012364
grad_step = 000028, loss = 0.013835
grad_step = 000029, loss = 0.013533
grad_step = 000030, loss = 0.011634
grad_step = 000031, loss = 0.009579
grad_step = 000032, loss = 0.008786
grad_step = 000033, loss = 0.009342
grad_step = 000034, loss = 0.009923
grad_step = 000035, loss = 0.009246
grad_step = 000036, loss = 0.007711
grad_step = 000037, loss = 0.006545
grad_step = 000038, loss = 0.006375
grad_step = 000039, loss = 0.006753
grad_step = 000040, loss = 0.006793
grad_step = 000041, loss = 0.006188
grad_step = 000042, loss = 0.005270
grad_step = 000043, loss = 0.004631
grad_step = 000044, loss = 0.004654
grad_step = 000045, loss = 0.005194
grad_step = 000046, loss = 0.005683
grad_step = 000047, loss = 0.005635
grad_step = 000048, loss = 0.005109
grad_step = 000049, loss = 0.004575
grad_step = 000050, loss = 0.004409
grad_step = 000051, loss = 0.004597
grad_step = 000052, loss = 0.004837
grad_step = 000053, loss = 0.004855
grad_step = 000054, loss = 0.004634
grad_step = 000055, loss = 0.004365
grad_step = 000056, loss = 0.004242
grad_step = 000057, loss = 0.004272
grad_step = 000058, loss = 0.004314
grad_step = 000059, loss = 0.004262
grad_step = 000060, loss = 0.004150
grad_step = 000061, loss = 0.004071
grad_step = 000062, loss = 0.004050
grad_step = 000063, loss = 0.004033
grad_step = 000064, loss = 0.003969
grad_step = 000065, loss = 0.003874
grad_step = 000066, loss = 0.003818
grad_step = 000067, loss = 0.003829
grad_step = 000068, loss = 0.003860
grad_step = 000069, loss = 0.003841
grad_step = 000070, loss = 0.003765
grad_step = 000071, loss = 0.003687
grad_step = 000072, loss = 0.003658
grad_step = 000073, loss = 0.003667
grad_step = 000074, loss = 0.003663
grad_step = 000075, loss = 0.003619
grad_step = 000076, loss = 0.003555
grad_step = 000077, loss = 0.003509
grad_step = 000078, loss = 0.003491
grad_step = 000079, loss = 0.003476
grad_step = 000080, loss = 0.003446
grad_step = 000081, loss = 0.003406
grad_step = 000082, loss = 0.003374
grad_step = 000083, loss = 0.003351
grad_step = 000084, loss = 0.003325
grad_step = 000085, loss = 0.003290
grad_step = 000086, loss = 0.003256
grad_step = 000087, loss = 0.003230
grad_step = 000088, loss = 0.003210
grad_step = 000089, loss = 0.003181
grad_step = 000090, loss = 0.003144
grad_step = 000091, loss = 0.003110
grad_step = 000092, loss = 0.003083
grad_step = 000093, loss = 0.003059
grad_step = 000094, loss = 0.003028
grad_step = 000095, loss = 0.002990
grad_step = 000096, loss = 0.002953
grad_step = 000097, loss = 0.002923
grad_step = 000098, loss = 0.002893
grad_step = 000099, loss = 0.002861
grad_step = 000100, loss = 0.002828
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002798
grad_step = 000102, loss = 0.002770
grad_step = 000103, loss = 0.002740
grad_step = 000104, loss = 0.002710
grad_step = 000105, loss = 0.002681
grad_step = 000106, loss = 0.002654
grad_step = 000107, loss = 0.002624
grad_step = 000108, loss = 0.002595
grad_step = 000109, loss = 0.002567
grad_step = 000110, loss = 0.002541
grad_step = 000111, loss = 0.002514
grad_step = 000112, loss = 0.002486
grad_step = 000113, loss = 0.002459
grad_step = 000114, loss = 0.002433
grad_step = 000115, loss = 0.002407
grad_step = 000116, loss = 0.002381
grad_step = 000117, loss = 0.002355
grad_step = 000118, loss = 0.002330
grad_step = 000119, loss = 0.002306
grad_step = 000120, loss = 0.002280
grad_step = 000121, loss = 0.002256
grad_step = 000122, loss = 0.002231
grad_step = 000123, loss = 0.002206
grad_step = 000124, loss = 0.002182
grad_step = 000125, loss = 0.002158
grad_step = 000126, loss = 0.002133
grad_step = 000127, loss = 0.002109
grad_step = 000128, loss = 0.002085
grad_step = 000129, loss = 0.002062
grad_step = 000130, loss = 0.002038
grad_step = 000131, loss = 0.002015
grad_step = 000132, loss = 0.001993
grad_step = 000133, loss = 0.001970
grad_step = 000134, loss = 0.001949
grad_step = 000135, loss = 0.001927
grad_step = 000136, loss = 0.001906
grad_step = 000137, loss = 0.001885
grad_step = 000138, loss = 0.001865
grad_step = 000139, loss = 0.001846
grad_step = 000140, loss = 0.001827
grad_step = 000141, loss = 0.001808
grad_step = 000142, loss = 0.001790
grad_step = 000143, loss = 0.001771
grad_step = 000144, loss = 0.001753
grad_step = 000145, loss = 0.001736
grad_step = 000146, loss = 0.001718
grad_step = 000147, loss = 0.001700
grad_step = 000148, loss = 0.001683
grad_step = 000149, loss = 0.001665
grad_step = 000150, loss = 0.001649
grad_step = 000151, loss = 0.001633
grad_step = 000152, loss = 0.001617
grad_step = 000153, loss = 0.001602
grad_step = 000154, loss = 0.001586
grad_step = 000155, loss = 0.001571
grad_step = 000156, loss = 0.001556
grad_step = 000157, loss = 0.001540
grad_step = 000158, loss = 0.001525
grad_step = 000159, loss = 0.001510
grad_step = 000160, loss = 0.001496
grad_step = 000161, loss = 0.001481
grad_step = 000162, loss = 0.001466
grad_step = 000163, loss = 0.001450
grad_step = 000164, loss = 0.001429
grad_step = 000165, loss = 0.001401
grad_step = 000166, loss = 0.001388
grad_step = 000167, loss = 0.001378
grad_step = 000168, loss = 0.001342
grad_step = 000169, loss = 0.001313
grad_step = 000170, loss = 0.001304
grad_step = 000171, loss = 0.001268
grad_step = 000172, loss = 0.001254
grad_step = 000173, loss = 0.001224
grad_step = 000174, loss = 0.001207
grad_step = 000175, loss = 0.001174
grad_step = 000176, loss = 0.001161
grad_step = 000177, loss = 0.001134
grad_step = 000178, loss = 0.001114
grad_step = 000179, loss = 0.001088
grad_step = 000180, loss = 0.001066
grad_step = 000181, loss = 0.001041
grad_step = 000182, loss = 0.001023
grad_step = 000183, loss = 0.001016
grad_step = 000184, loss = 0.001063
grad_step = 000185, loss = 0.001167
grad_step = 000186, loss = 0.001114
grad_step = 000187, loss = 0.000919
grad_step = 000188, loss = 0.001016
grad_step = 000189, loss = 0.001046
grad_step = 000190, loss = 0.000876
grad_step = 000191, loss = 0.000952
grad_step = 000192, loss = 0.000947
grad_step = 000193, loss = 0.000830
grad_step = 000194, loss = 0.000908
grad_step = 000195, loss = 0.000851
grad_step = 000196, loss = 0.000800
grad_step = 000197, loss = 0.000851
grad_step = 000198, loss = 0.000777
grad_step = 000199, loss = 0.000773
grad_step = 000200, loss = 0.000791
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000729
grad_step = 000202, loss = 0.000743
grad_step = 000203, loss = 0.000738
grad_step = 000204, loss = 0.000692
grad_step = 000205, loss = 0.000704
grad_step = 000206, loss = 0.000692
grad_step = 000207, loss = 0.000662
grad_step = 000208, loss = 0.000667
grad_step = 000209, loss = 0.000658
grad_step = 000210, loss = 0.000637
grad_step = 000211, loss = 0.000634
grad_step = 000212, loss = 0.000634
grad_step = 000213, loss = 0.000617
grad_step = 000214, loss = 0.000610
grad_step = 000215, loss = 0.000610
grad_step = 000216, loss = 0.000602
grad_step = 000217, loss = 0.000590
grad_step = 000218, loss = 0.000585
grad_step = 000219, loss = 0.000583
grad_step = 000220, loss = 0.000577
grad_step = 000221, loss = 0.000570
grad_step = 000222, loss = 0.000564
grad_step = 000223, loss = 0.000561
grad_step = 000224, loss = 0.000557
grad_step = 000225, loss = 0.000552
grad_step = 000226, loss = 0.000545
grad_step = 000227, loss = 0.000542
grad_step = 000228, loss = 0.000539
grad_step = 000229, loss = 0.000535
grad_step = 000230, loss = 0.000529
grad_step = 000231, loss = 0.000525
grad_step = 000232, loss = 0.000522
grad_step = 000233, loss = 0.000518
grad_step = 000234, loss = 0.000514
grad_step = 000235, loss = 0.000511
grad_step = 000236, loss = 0.000506
grad_step = 000237, loss = 0.000502
grad_step = 000238, loss = 0.000499
grad_step = 000239, loss = 0.000496
grad_step = 000240, loss = 0.000492
grad_step = 000241, loss = 0.000487
grad_step = 000242, loss = 0.000484
grad_step = 000243, loss = 0.000480
grad_step = 000244, loss = 0.000478
grad_step = 000245, loss = 0.000475
grad_step = 000246, loss = 0.000471
grad_step = 000247, loss = 0.000467
grad_step = 000248, loss = 0.000462
grad_step = 000249, loss = 0.000459
grad_step = 000250, loss = 0.000457
grad_step = 000251, loss = 0.000455
grad_step = 000252, loss = 0.000452
grad_step = 000253, loss = 0.000447
grad_step = 000254, loss = 0.000444
grad_step = 000255, loss = 0.000444
grad_step = 000256, loss = 0.000448
grad_step = 000257, loss = 0.000460
grad_step = 000258, loss = 0.000491
grad_step = 000259, loss = 0.000526
grad_step = 000260, loss = 0.000541
grad_step = 000261, loss = 0.000480
grad_step = 000262, loss = 0.000426
grad_step = 000263, loss = 0.000435
grad_step = 000264, loss = 0.000472
grad_step = 000265, loss = 0.000468
grad_step = 000266, loss = 0.000423
grad_step = 000267, loss = 0.000416
grad_step = 000268, loss = 0.000444
grad_step = 000269, loss = 0.000435
grad_step = 000270, loss = 0.000405
grad_step = 000271, loss = 0.000408
grad_step = 000272, loss = 0.000423
grad_step = 000273, loss = 0.000412
grad_step = 000274, loss = 0.000396
grad_step = 000275, loss = 0.000400
grad_step = 000276, loss = 0.000407
grad_step = 000277, loss = 0.000395
grad_step = 000278, loss = 0.000384
grad_step = 000279, loss = 0.000390
grad_step = 000280, loss = 0.000394
grad_step = 000281, loss = 0.000388
grad_step = 000282, loss = 0.000389
grad_step = 000283, loss = 0.000393
grad_step = 000284, loss = 0.000389
grad_step = 000285, loss = 0.000373
grad_step = 000286, loss = 0.000367
grad_step = 000287, loss = 0.000375
grad_step = 000288, loss = 0.000376
grad_step = 000289, loss = 0.000367
grad_step = 000290, loss = 0.000356
grad_step = 000291, loss = 0.000356
grad_step = 000292, loss = 0.000360
grad_step = 000293, loss = 0.000360
grad_step = 000294, loss = 0.000357
grad_step = 000295, loss = 0.000351
grad_step = 000296, loss = 0.000346
grad_step = 000297, loss = 0.000342
grad_step = 000298, loss = 0.000340
grad_step = 000299, loss = 0.000340
grad_step = 000300, loss = 0.000340
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000339
grad_step = 000302, loss = 0.000333
grad_step = 000303, loss = 0.000328
grad_step = 000304, loss = 0.000323
grad_step = 000305, loss = 0.000323
grad_step = 000306, loss = 0.000326
grad_step = 000307, loss = 0.000325
grad_step = 000308, loss = 0.000324
grad_step = 000309, loss = 0.000316
grad_step = 000310, loss = 0.000309
grad_step = 000311, loss = 0.000306
grad_step = 000312, loss = 0.000306
grad_step = 000313, loss = 0.000309
grad_step = 000314, loss = 0.000308
grad_step = 000315, loss = 0.000308
grad_step = 000316, loss = 0.000302
grad_step = 000317, loss = 0.000295
grad_step = 000318, loss = 0.000289
grad_step = 000319, loss = 0.000286
grad_step = 000320, loss = 0.000286
grad_step = 000321, loss = 0.000286
grad_step = 000322, loss = 0.000288
grad_step = 000323, loss = 0.000286
grad_step = 000324, loss = 0.000284
grad_step = 000325, loss = 0.000279
grad_step = 000326, loss = 0.000275
grad_step = 000327, loss = 0.000275
grad_step = 000328, loss = 0.000285
grad_step = 000329, loss = 0.000313
grad_step = 000330, loss = 0.000352
grad_step = 000331, loss = 0.000382
grad_step = 000332, loss = 0.000356
grad_step = 000333, loss = 0.000297
grad_step = 000334, loss = 0.000268
grad_step = 000335, loss = 0.000283
grad_step = 000336, loss = 0.000306
grad_step = 000337, loss = 0.000291
grad_step = 000338, loss = 0.000265
grad_step = 000339, loss = 0.000264
grad_step = 000340, loss = 0.000268
grad_step = 000341, loss = 0.000264
grad_step = 000342, loss = 0.000257
grad_step = 000343, loss = 0.000259
grad_step = 000344, loss = 0.000262
grad_step = 000345, loss = 0.000246
grad_step = 000346, loss = 0.000244
grad_step = 000347, loss = 0.000255
grad_step = 000348, loss = 0.000251
grad_step = 000349, loss = 0.000237
grad_step = 000350, loss = 0.000235
grad_step = 000351, loss = 0.000241
grad_step = 000352, loss = 0.000241
grad_step = 000353, loss = 0.000233
grad_step = 000354, loss = 0.000232
grad_step = 000355, loss = 0.000235
grad_step = 000356, loss = 0.000229
grad_step = 000357, loss = 0.000224
grad_step = 000358, loss = 0.000225
grad_step = 000359, loss = 0.000226
grad_step = 000360, loss = 0.000225
grad_step = 000361, loss = 0.000222
grad_step = 000362, loss = 0.000220
grad_step = 000363, loss = 0.000220
grad_step = 000364, loss = 0.000221
grad_step = 000365, loss = 0.000221
grad_step = 000366, loss = 0.000219
grad_step = 000367, loss = 0.000218
grad_step = 000368, loss = 0.000220
grad_step = 000369, loss = 0.000225
grad_step = 000370, loss = 0.000232
grad_step = 000371, loss = 0.000243
grad_step = 000372, loss = 0.000260
grad_step = 000373, loss = 0.000286
grad_step = 000374, loss = 0.000298
grad_step = 000375, loss = 0.000287
grad_step = 000376, loss = 0.000245
grad_step = 000377, loss = 0.000214
grad_step = 000378, loss = 0.000215
grad_step = 000379, loss = 0.000231
grad_step = 000380, loss = 0.000240
grad_step = 000381, loss = 0.000229
grad_step = 000382, loss = 0.000215
grad_step = 000383, loss = 0.000209
grad_step = 000384, loss = 0.000213
grad_step = 000385, loss = 0.000222
grad_step = 000386, loss = 0.000222
grad_step = 000387, loss = 0.000212
grad_step = 000388, loss = 0.000201
grad_step = 000389, loss = 0.000204
grad_step = 000390, loss = 0.000213
grad_step = 000391, loss = 0.000213
grad_step = 000392, loss = 0.000206
grad_step = 000393, loss = 0.000199
grad_step = 000394, loss = 0.000200
grad_step = 000395, loss = 0.000203
grad_step = 000396, loss = 0.000203
grad_step = 000397, loss = 0.000201
grad_step = 000398, loss = 0.000199
grad_step = 000399, loss = 0.000200
grad_step = 000400, loss = 0.000197
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000196
grad_step = 000402, loss = 0.000195
grad_step = 000403, loss = 0.000196
grad_step = 000404, loss = 0.000198
grad_step = 000405, loss = 0.000196
grad_step = 000406, loss = 0.000194
grad_step = 000407, loss = 0.000192
grad_step = 000408, loss = 0.000192
grad_step = 000409, loss = 0.000193
grad_step = 000410, loss = 0.000192
grad_step = 000411, loss = 0.000191
grad_step = 000412, loss = 0.000191
grad_step = 000413, loss = 0.000193
grad_step = 000414, loss = 0.000195
grad_step = 000415, loss = 0.000196
grad_step = 000416, loss = 0.000195
grad_step = 000417, loss = 0.000195
grad_step = 000418, loss = 0.000195
grad_step = 000419, loss = 0.000198
grad_step = 000420, loss = 0.000201
grad_step = 000421, loss = 0.000207
grad_step = 000422, loss = 0.000208
grad_step = 000423, loss = 0.000209
grad_step = 000424, loss = 0.000207
grad_step = 000425, loss = 0.000208
grad_step = 000426, loss = 0.000206
grad_step = 000427, loss = 0.000200
grad_step = 000428, loss = 0.000191
grad_step = 000429, loss = 0.000187
grad_step = 000430, loss = 0.000186
grad_step = 000431, loss = 0.000187
grad_step = 000432, loss = 0.000188
grad_step = 000433, loss = 0.000188
grad_step = 000434, loss = 0.000191
grad_step = 000435, loss = 0.000194
grad_step = 000436, loss = 0.000199
grad_step = 000437, loss = 0.000200
grad_step = 000438, loss = 0.000201
grad_step = 000439, loss = 0.000197
grad_step = 000440, loss = 0.000195
grad_step = 000441, loss = 0.000191
grad_step = 000442, loss = 0.000188
grad_step = 000443, loss = 0.000184
grad_step = 000444, loss = 0.000181
grad_step = 000445, loss = 0.000180
grad_step = 000446, loss = 0.000180
grad_step = 000447, loss = 0.000181
grad_step = 000448, loss = 0.000181
grad_step = 000449, loss = 0.000182
grad_step = 000450, loss = 0.000182
grad_step = 000451, loss = 0.000184
grad_step = 000452, loss = 0.000186
grad_step = 000453, loss = 0.000189
grad_step = 000454, loss = 0.000194
grad_step = 000455, loss = 0.000202
grad_step = 000456, loss = 0.000206
grad_step = 000457, loss = 0.000212
grad_step = 000458, loss = 0.000209
grad_step = 000459, loss = 0.000206
grad_step = 000460, loss = 0.000198
grad_step = 000461, loss = 0.000192
grad_step = 000462, loss = 0.000183
grad_step = 000463, loss = 0.000177
grad_step = 000464, loss = 0.000177
grad_step = 000465, loss = 0.000181
grad_step = 000466, loss = 0.000187
grad_step = 000467, loss = 0.000188
grad_step = 000468, loss = 0.000186
grad_step = 000469, loss = 0.000181
grad_step = 000470, loss = 0.000178
grad_step = 000471, loss = 0.000177
grad_step = 000472, loss = 0.000180
grad_step = 000473, loss = 0.000180
grad_step = 000474, loss = 0.000179
grad_step = 000475, loss = 0.000175
grad_step = 000476, loss = 0.000172
grad_step = 000477, loss = 0.000172
grad_step = 000478, loss = 0.000173
grad_step = 000479, loss = 0.000175
grad_step = 000480, loss = 0.000174
grad_step = 000481, loss = 0.000173
grad_step = 000482, loss = 0.000172
grad_step = 000483, loss = 0.000172
grad_step = 000484, loss = 0.000173
grad_step = 000485, loss = 0.000175
grad_step = 000486, loss = 0.000177
grad_step = 000487, loss = 0.000179
grad_step = 000488, loss = 0.000182
grad_step = 000489, loss = 0.000188
grad_step = 000490, loss = 0.000199
grad_step = 000491, loss = 0.000220
grad_step = 000492, loss = 0.000235
grad_step = 000493, loss = 0.000246
grad_step = 000494, loss = 0.000239
grad_step = 000495, loss = 0.000225
grad_step = 000496, loss = 0.000201
grad_step = 000497, loss = 0.000182
grad_step = 000498, loss = 0.000175
grad_step = 000499, loss = 0.000179
grad_step = 000500, loss = 0.000190
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000192
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
[[0.8416163  0.81919795 0.9264041  0.92805636 1.0114694 ]
 [0.81001997 0.9017228  0.94384897 0.9942239  1.0061698 ]
 [0.87874246 0.90882605 1.0011234  0.96170306 0.9517765 ]
 [0.9058561  0.9723824  1.0026755  0.94142246 0.9032857 ]
 [0.96533674 1.0039848  0.9573188  0.927539   0.86739904]
 [0.99962866 0.95014    0.9128718  0.8673313  0.85319805]
 [0.94995093 0.90082407 0.8568929  0.8440399  0.82554406]
 [0.8988174  0.8503872  0.8561262  0.8100116  0.84678805]
 [0.8471986  0.8396171  0.8149229  0.8373301  0.854414  ]
 [0.83378583 0.7990328  0.8544803  0.84887516 0.83610535]
 [0.7820488  0.82164395 0.8558423  0.8119186  0.92592525]
 [0.82672703 0.83912075 0.81583315 0.9176674  0.96155095]
 [0.82232594 0.81486607 0.92119205 0.93048906 1.0094287 ]
 [0.81041265 0.91199803 0.9500139  0.9966365  0.99461865]
 [0.8911245  0.92538226 1.0042259  0.959818   0.9332454 ]
 [0.9213956  0.9931773  0.9905089  0.9300772  0.8862238 ]
 [0.98811424 0.9827553  0.93756014 0.8965135  0.8359532 ]
 [0.9799292  0.9278229  0.8919313  0.848571   0.84084964]
 [0.9411408  0.8877252  0.8473482  0.83403313 0.81612945]
 [0.8997475  0.8518612  0.8552346  0.8047597  0.8447989 ]
 [0.84924936 0.85982937 0.8127903  0.8312753  0.86576533]
 [0.8569618  0.8164848  0.85905355 0.85772884 0.83998805]
 [0.79736656 0.8349121  0.871171   0.8258786  0.92951745]
 [0.8340739  0.8526798  0.8272299  0.9181028  0.9661838 ]
 [0.84780574 0.8213563  0.9276173  0.9325444  1.0187576 ]
 [0.8190677  0.9082218  0.95025945 1.0060383  1.017777  ]
 [0.8848697  0.92141885 1.0097101  0.9756826  0.96639824]
 [0.9172319  0.98186517 1.0167345  0.9531169  0.9146235 ]
 [0.97191054 1.010365   0.96647584 0.9374492  0.8772073 ]
 [1.0070673  0.955902   0.9224359  0.87501764 0.85890573]
 [0.95663905 0.90807486 0.8632084  0.8505126  0.8332046 ]]

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
Warning: Permanently added the RSA host key for IP address '140.82.118.4' to the list of known hosts.
Already up to date.
[master 85ff7b4] ml_store
 1 file changed, 1124 insertions(+)
To github.com:arita37/mlmodels_store.git
   4958fd9..85ff7b4  master -> master





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
[master 50a9e00] ml_store
 1 file changed, 38 insertions(+)
To github.com:arita37/mlmodels_store.git
   85ff7b4..50a9e00  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 22888339.86B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 405386.97B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 5192704/440473133 [00:00<00:08, 51924861.15B/s]  3%|▎         | 11505664/440473133 [00:00<00:07, 54843987.70B/s]  4%|▍         | 17917952/440473133 [00:00<00:07, 57332796.79B/s]  6%|▌         | 24374272/440473133 [00:00<00:07, 59324185.08B/s]  7%|▋         | 30846976/440473133 [00:00<00:06, 60846936.93B/s]  8%|▊         | 37073920/440473133 [00:00<00:06, 61266661.36B/s] 10%|▉         | 43484160/440473133 [00:00<00:06, 62087475.99B/s] 11%|█▏        | 50022400/440473133 [00:00<00:06, 63039435.20B/s] 13%|█▎        | 56592384/440473133 [00:00<00:06, 63809796.78B/s] 14%|█▍        | 63167488/440473133 [00:01<00:05, 64379413.15B/s] 16%|█▌        | 69736448/440473133 [00:01<00:05, 64764820.14B/s] 17%|█▋        | 76263424/440473133 [00:01<00:05, 64913928.28B/s] 19%|█▉        | 82838528/440473133 [00:01<00:05, 65160616.47B/s] 20%|██        | 89485312/440473133 [00:01<00:05, 65544828.29B/s] 22%|██▏       | 96015360/440473133 [00:01<00:05, 65470834.33B/s] 23%|██▎       | 102624256/440473133 [00:01<00:05, 65653906.93B/s] 25%|██▍       | 109174784/440473133 [00:01<00:05, 64835313.09B/s] 26%|██▋       | 115650560/440473133 [00:01<00:05, 64611357.12B/s] 28%|██▊       | 122241024/440473133 [00:01<00:04, 64993403.24B/s] 29%|██▉       | 128819200/440473133 [00:02<00:04, 65223635.46B/s] 31%|███       | 135469056/440473133 [00:02<00:04, 65596664.07B/s] 32%|███▏      | 142027776/440473133 [00:02<00:04, 65562139.98B/s] 34%|███▎      | 148583424/440473133 [00:02<00:04, 65447216.76B/s] 35%|███▌      | 155173888/440473133 [00:02<00:04, 65582580.62B/s] 37%|███▋      | 161810432/440473133 [00:02<00:04, 65814997.39B/s] 38%|███▊      | 168429568/440473133 [00:02<00:04, 65925643.13B/s] 40%|███▉      | 175023104/440473133 [00:02<00:04, 65412758.01B/s] 41%|████      | 181582848/440473133 [00:02<00:03, 65463632.58B/s] 43%|████▎     | 188130304/440473133 [00:02<00:03, 64343025.71B/s] 44%|████▍     | 194570240/440473133 [00:03<00:03, 64153663.93B/s] 46%|████▌     | 201146368/440473133 [00:03<00:03, 64625699.55B/s] 47%|████▋     | 207780864/440473133 [00:03<00:03, 65128682.41B/s] 49%|████▊     | 214297600/440473133 [00:03<00:03, 64991097.32B/s] 50%|█████     | 220901376/440473133 [00:03<00:03, 65299535.92B/s] 52%|█████▏    | 227487744/440473133 [00:03<00:03, 65465127.47B/s] 53%|█████▎    | 234100736/440473133 [00:03<00:03, 65661544.03B/s] 55%|█████▍    | 240669696/440473133 [00:03<00:03, 65668098.87B/s] 56%|█████▌    | 247237632/440473133 [00:03<00:02, 65002213.06B/s] 58%|█████▊    | 253774848/440473133 [00:03<00:02, 65111569.25B/s] 59%|█████▉    | 260355072/440473133 [00:04<00:02, 65315824.27B/s] 61%|██████    | 266925056/440473133 [00:04<00:02, 65428664.58B/s] 62%|██████▏   | 273476608/440473133 [00:04<00:02, 65453780.27B/s] 64%|██████▎   | 280023040/440473133 [00:04<00:02, 65220335.67B/s] 65%|██████▌   | 286545920/440473133 [00:04<00:02, 65217436.18B/s] 67%|██████▋   | 293068800/440473133 [00:04<00:02, 64965836.21B/s] 68%|██████▊   | 299566080/440473133 [00:04<00:02, 64890068.59B/s] 69%|██████▉   | 306056192/440473133 [00:04<00:02, 64813212.34B/s] 71%|███████   | 312538112/440473133 [00:04<00:01, 64048429.23B/s] 72%|███████▏  | 319052800/440473133 [00:04<00:01, 64371707.65B/s] 74%|███████▍  | 325588992/440473133 [00:05<00:01, 64662788.00B/s] 75%|███████▌  | 332122112/440473133 [00:05<00:01, 64860553.80B/s] 77%|███████▋  | 338610176/440473133 [00:05<00:01, 64796953.36B/s] 78%|███████▊  | 345091072/440473133 [00:05<00:01, 64515617.75B/s] 80%|███████▉  | 351583232/440473133 [00:05<00:01, 64636344.35B/s] 81%|████████▏ | 358110208/440473133 [00:05<00:01, 64822974.56B/s] 83%|████████▎ | 364610560/440473133 [00:05<00:01, 64872096.17B/s] 84%|████████▍ | 371170304/440473133 [00:05<00:01, 65087889.31B/s] 86%|████████▌ | 377679872/440473133 [00:05<00:00, 64706555.11B/s] 87%|████████▋ | 384180224/440473133 [00:05<00:00, 64793945.70B/s] 89%|████████▊ | 390732800/440473133 [00:06<00:00, 65010310.01B/s] 90%|█████████ | 397264896/440473133 [00:06<00:00, 65101820.60B/s] 92%|█████████▏| 403776512/440473133 [00:06<00:00, 64650545.05B/s] 93%|█████████▎| 410243072/440473133 [00:06<00:00, 64544530.08B/s] 95%|█████████▍| 416698368/440473133 [00:06<00:00, 64303976.28B/s] 96%|█████████▌| 423130112/440473133 [00:06<00:00, 64271491.96B/s] 98%|█████████▊| 429625344/440473133 [00:06<00:00, 64472561.63B/s] 99%|█████████▉| 436132864/440473133 [00:06<00:00, 64651139.63B/s]100%|██████████| 440473133/440473133 [00:06<00:00, 64742542.02B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
  16384/7094233 [..............................] - ETA: 27s
  65536/7094233 [..............................] - ETA: 14s
 147456/7094233 [..............................] - ETA: 9s 
 327680/7094233 [>.............................] - ETA: 5s
 671744/7094233 [=>............................] - ETA: 3s
1048576/7094233 [===>..........................] - ETA: 2s
1433600/7094233 [=====>........................] - ETA: 1s
2170880/7094233 [========>.....................] - ETA: 1s
2924544/7094233 [===========>..................] - ETA: 0s
3629056/7094233 [==============>...............] - ETA: 0s
4538368/7094233 [==================>...........] - ETA: 0s
5300224/7094233 [=====================>........] - ETA: 0s
6021120/7094233 [========================>.....] - ETA: 0s
6717440/7094233 [===========================>..] - ETA: 0s
7094272/7094233 [==============================] - 1s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   0%|          | 2/2118 [00:00<01:59, 17.72it/s]Processing text_left with encode:  20%|██        | 427/2118 [00:00<01:06, 25.28it/s]Processing text_left with encode:  41%|████      | 867/2118 [00:00<00:34, 36.02it/s]Processing text_left with encode:  57%|█████▋    | 1213/2118 [00:00<00:17, 51.23it/s]Processing text_left with encode:  78%|███████▊  | 1660/2118 [00:00<00:06, 72.82it/s]Processing text_left with encode:  99%|█████████▉| 2100/2118 [00:00<00:00, 103.30it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3423.85it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 154/18841 [00:00<00:12, 1535.71it/s]Processing text_right with encode:   2%|▏         | 320/18841 [00:00<00:11, 1569.47it/s]Processing text_right with encode:   3%|▎         | 483/18841 [00:00<00:11, 1585.92it/s]Processing text_right with encode:   3%|▎         | 654/18841 [00:00<00:11, 1620.86it/s]Processing text_right with encode:   4%|▍         | 811/18841 [00:00<00:11, 1601.97it/s]Processing text_right with encode:   5%|▌         | 966/18841 [00:00<00:11, 1584.80it/s]Processing text_right with encode:   6%|▌         | 1122/18841 [00:00<00:11, 1573.64it/s]Processing text_right with encode:   7%|▋         | 1282/18841 [00:00<00:11, 1580.20it/s]Processing text_right with encode:   8%|▊         | 1432/18841 [00:00<00:11, 1531.41it/s]Processing text_right with encode:   8%|▊         | 1581/18841 [00:01<00:11, 1515.76it/s]Processing text_right with encode:   9%|▉         | 1744/18841 [00:01<00:11, 1545.85it/s]Processing text_right with encode:  10%|█         | 1901/18841 [00:01<00:10, 1552.48it/s]Processing text_right with encode:  11%|█         | 2056/18841 [00:01<00:10, 1551.72it/s]Processing text_right with encode:  12%|█▏        | 2225/18841 [00:01<00:10, 1590.35it/s]Processing text_right with encode:  13%|█▎        | 2391/18841 [00:01<00:10, 1607.54it/s]Processing text_right with encode:  14%|█▎        | 2556/18841 [00:01<00:10, 1617.66it/s]Processing text_right with encode:  14%|█▍        | 2719/18841 [00:01<00:09, 1620.84it/s]Processing text_right with encode:  15%|█▌        | 2898/18841 [00:01<00:09, 1664.91it/s]Processing text_right with encode:  16%|█▋        | 3065/18841 [00:01<00:09, 1615.99it/s]Processing text_right with encode:  17%|█▋        | 3228/18841 [00:02<00:09, 1617.28it/s]Processing text_right with encode:  18%|█▊        | 3391/18841 [00:02<00:09, 1589.02it/s]Processing text_right with encode:  19%|█▉        | 3551/18841 [00:02<00:09, 1566.68it/s]Processing text_right with encode:  20%|█▉        | 3710/18841 [00:02<00:09, 1571.47it/s]Processing text_right with encode:  21%|██        | 3878/18841 [00:02<00:09, 1602.26it/s]Processing text_right with encode:  21%|██▏       | 4046/18841 [00:02<00:09, 1624.13it/s]Processing text_right with encode:  22%|██▏       | 4209/18841 [00:02<00:09, 1578.58it/s]Processing text_right with encode:  23%|██▎       | 4368/18841 [00:02<00:09, 1578.62it/s]Processing text_right with encode:  24%|██▍       | 4527/18841 [00:02<00:09, 1568.36it/s]Processing text_right with encode:  25%|██▍       | 4693/18841 [00:02<00:08, 1593.72it/s]Processing text_right with encode:  26%|██▌       | 4862/18841 [00:03<00:08, 1620.71it/s]Processing text_right with encode:  27%|██▋       | 5025/18841 [00:03<00:08, 1622.10it/s]Processing text_right with encode:  28%|██▊       | 5198/18841 [00:03<00:08, 1650.88it/s]Processing text_right with encode:  29%|██▊       | 5371/18841 [00:03<00:08, 1671.71it/s]Processing text_right with encode:  29%|██▉       | 5539/18841 [00:03<00:07, 1672.65it/s]Processing text_right with encode:  30%|███       | 5707/18841 [00:03<00:07, 1651.55it/s]Processing text_right with encode:  31%|███       | 5873/18841 [00:03<00:07, 1632.75it/s]Processing text_right with encode:  32%|███▏      | 6037/18841 [00:03<00:07, 1615.14it/s]Processing text_right with encode:  33%|███▎      | 6199/18841 [00:03<00:07, 1607.86it/s]Processing text_right with encode:  34%|███▍      | 6362/18841 [00:03<00:07, 1612.60it/s]Processing text_right with encode:  35%|███▍      | 6534/18841 [00:04<00:07, 1642.58it/s]Processing text_right with encode:  36%|███▌      | 6699/18841 [00:04<00:07, 1639.24it/s]Processing text_right with encode:  36%|███▋      | 6864/18841 [00:04<00:07, 1633.67it/s]Processing text_right with encode:  37%|███▋      | 7028/18841 [00:04<00:07, 1614.75it/s]Processing text_right with encode:  38%|███▊      | 7190/18841 [00:04<00:07, 1593.67it/s]Processing text_right with encode:  39%|███▉      | 7368/18841 [00:04<00:06, 1645.28it/s]Processing text_right with encode:  40%|███▉      | 7534/18841 [00:04<00:06, 1640.37it/s]Processing text_right with encode:  41%|████      | 7699/18841 [00:04<00:06, 1633.35it/s]Processing text_right with encode:  42%|████▏     | 7867/18841 [00:04<00:06, 1645.21it/s]Processing text_right with encode:  43%|████▎     | 8032/18841 [00:04<00:06, 1615.91it/s]Processing text_right with encode:  43%|████▎     | 8194/18841 [00:05<00:06, 1616.87it/s]Processing text_right with encode:  44%|████▍     | 8367/18841 [00:05<00:06, 1648.64it/s]Processing text_right with encode:  45%|████▌     | 8533/18841 [00:05<00:06, 1610.18it/s]Processing text_right with encode:  46%|████▌     | 8700/18841 [00:05<00:06, 1625.89it/s]Processing text_right with encode:  47%|████▋     | 8877/18841 [00:05<00:05, 1664.74it/s]Processing text_right with encode:  48%|████▊     | 9044/18841 [00:05<00:06, 1599.54it/s]Processing text_right with encode:  49%|████▉     | 9210/18841 [00:05<00:05, 1617.13it/s]Processing text_right with encode:  50%|████▉     | 9373/18841 [00:05<00:05, 1598.66it/s]Processing text_right with encode:  51%|█████     | 9542/18841 [00:05<00:05, 1623.12it/s]Processing text_right with encode:  52%|█████▏    | 9708/18841 [00:06<00:05, 1633.47it/s]Processing text_right with encode:  52%|█████▏    | 9872/18841 [00:06<00:05, 1605.46it/s]Processing text_right with encode:  53%|█████▎    | 10037/18841 [00:06<00:05, 1617.89it/s]Processing text_right with encode:  54%|█████▍    | 10200/18841 [00:06<00:05, 1577.36it/s]Processing text_right with encode:  55%|█████▌    | 10364/18841 [00:06<00:05, 1595.14it/s]Processing text_right with encode:  56%|█████▌    | 10553/18841 [00:06<00:04, 1673.01it/s]Processing text_right with encode:  57%|█████▋    | 10722/18841 [00:06<00:04, 1664.53it/s]Processing text_right with encode:  58%|█████▊    | 10890/18841 [00:06<00:04, 1640.89it/s]Processing text_right with encode:  59%|█████▊    | 11055/18841 [00:06<00:04, 1634.36it/s]Processing text_right with encode:  60%|█████▉    | 11219/18841 [00:06<00:04, 1611.26it/s]Processing text_right with encode:  60%|██████    | 11381/18841 [00:07<00:04, 1600.25it/s]Processing text_right with encode:  61%|██████▏   | 11542/18841 [00:07<00:04, 1600.63it/s]Processing text_right with encode:  62%|██████▏   | 11703/18841 [00:07<00:04, 1596.24it/s]Processing text_right with encode:  63%|██████▎   | 11873/18841 [00:07<00:04, 1623.65it/s]Processing text_right with encode:  64%|██████▍   | 12039/18841 [00:07<00:04, 1634.32it/s]Processing text_right with encode:  65%|██████▍   | 12203/18841 [00:07<00:04, 1611.02it/s]Processing text_right with encode:  66%|██████▌   | 12371/18841 [00:07<00:03, 1627.93it/s]Processing text_right with encode:  67%|██████▋   | 12534/18841 [00:07<00:03, 1613.92it/s]Processing text_right with encode:  67%|██████▋   | 12696/18841 [00:07<00:03, 1603.64it/s]Processing text_right with encode:  68%|██████▊   | 12857/18841 [00:07<00:03, 1588.80it/s]Processing text_right with encode:  69%|██████▉   | 13018/18841 [00:08<00:03, 1593.20it/s]Processing text_right with encode:  70%|██████▉   | 13178/18841 [00:08<00:03, 1592.78it/s]Processing text_right with encode:  71%|███████   | 13350/18841 [00:08<00:03, 1624.34it/s]Processing text_right with encode:  72%|███████▏  | 13513/18841 [00:08<00:03, 1622.96it/s]Processing text_right with encode:  73%|███████▎  | 13683/18841 [00:08<00:03, 1644.83it/s]Processing text_right with encode:  73%|███████▎  | 13848/18841 [00:08<00:03, 1612.14it/s]Processing text_right with encode:  74%|███████▍  | 14018/18841 [00:08<00:02, 1636.71it/s]Processing text_right with encode:  75%|███████▌  | 14182/18841 [00:08<00:02, 1601.13it/s]Processing text_right with encode:  76%|███████▌  | 14343/18841 [00:08<00:02, 1597.95it/s]Processing text_right with encode:  77%|███████▋  | 14506/18841 [00:08<00:02, 1604.55it/s]Processing text_right with encode:  78%|███████▊  | 14684/18841 [00:09<00:02, 1651.19it/s]Processing text_right with encode:  79%|███████▉  | 14852/18841 [00:09<00:02, 1659.38it/s]Processing text_right with encode:  80%|███████▉  | 15019/18841 [00:09<00:02, 1634.50it/s]Processing text_right with encode:  81%|████████  | 15183/18841 [00:09<00:02, 1635.03it/s]Processing text_right with encode:  81%|████████▏ | 15347/18841 [00:09<00:02, 1634.21it/s]Processing text_right with encode:  82%|████████▏ | 15511/18841 [00:09<00:02, 1629.29it/s]Processing text_right with encode:  83%|████████▎ | 15680/18841 [00:09<00:01, 1644.67it/s]Processing text_right with encode:  84%|████████▍ | 15848/18841 [00:09<00:01, 1655.11it/s]Processing text_right with encode:  85%|████████▍ | 16014/18841 [00:09<00:01, 1634.51it/s]Processing text_right with encode:  86%|████████▌ | 16178/18841 [00:10<00:01, 1619.38it/s]Processing text_right with encode:  87%|████████▋ | 16344/18841 [00:10<00:01, 1627.98it/s]Processing text_right with encode:  88%|████████▊ | 16507/18841 [00:10<00:01, 1627.86it/s]Processing text_right with encode:  88%|████████▊ | 16670/18841 [00:10<00:01, 1600.20it/s]Processing text_right with encode:  89%|████████▉ | 16833/18841 [00:10<00:01, 1608.99it/s]Processing text_right with encode:  90%|█████████ | 16995/18841 [00:10<00:01, 1599.76it/s]Processing text_right with encode:  91%|█████████ | 17156/18841 [00:10<00:01, 1593.76it/s]Processing text_right with encode:  92%|█████████▏| 17316/18841 [00:10<00:00, 1593.49it/s]Processing text_right with encode:  93%|█████████▎| 17476/18841 [00:10<00:00, 1589.48it/s]Processing text_right with encode:  94%|█████████▎| 17641/18841 [00:10<00:00, 1605.15it/s]Processing text_right with encode:  95%|█████████▍| 17810/18841 [00:11<00:00, 1626.54it/s]Processing text_right with encode:  95%|█████████▌| 17981/18841 [00:11<00:00, 1649.71it/s]Processing text_right with encode:  96%|█████████▋| 18161/18841 [00:11<00:00, 1687.39it/s]Processing text_right with encode:  97%|█████████▋| 18331/18841 [00:11<00:00, 1639.09it/s]Processing text_right with encode:  98%|█████████▊| 18510/18841 [00:11<00:00, 1680.59it/s]Processing text_right with encode:  99%|█████████▉| 18679/18841 [00:11<00:00, 1660.92it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:11<00:00, 1619.26it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 626748.69it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 740717.07it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  70%|██████▉   | 442/633 [00:00<00:00, 4418.79it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4345.61it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 171/5961 [00:00<00:03, 1708.63it/s]Processing text_right with encode:   6%|▌         | 341/5961 [00:00<00:03, 1699.77it/s]Processing text_right with encode:   8%|▊         | 493/5961 [00:00<00:03, 1640.79it/s]Processing text_right with encode:  11%|█         | 655/5961 [00:00<00:03, 1634.46it/s]Processing text_right with encode:  14%|█▍        | 821/5961 [00:00<00:03, 1640.79it/s]Processing text_right with encode:  17%|█▋        | 994/5961 [00:00<00:02, 1662.34it/s]Processing text_right with encode:  19%|█▉        | 1162/5961 [00:00<00:02, 1666.50it/s]Processing text_right with encode:  22%|██▏       | 1333/5961 [00:00<00:02, 1679.01it/s]Processing text_right with encode:  25%|██▌       | 1502/5961 [00:00<00:02, 1682.22it/s]Processing text_right with encode:  28%|██▊       | 1666/5961 [00:01<00:02, 1667.79it/s]Processing text_right with encode:  31%|███       | 1829/5961 [00:01<00:02, 1655.67it/s]Processing text_right with encode:  33%|███▎      | 1992/5961 [00:01<00:02, 1639.71it/s]Processing text_right with encode:  36%|███▋      | 2167/5961 [00:01<00:02, 1669.61it/s]Processing text_right with encode:  39%|███▉      | 2333/5961 [00:01<00:02, 1650.99it/s]Processing text_right with encode:  42%|████▏     | 2498/5961 [00:01<00:02, 1627.51it/s]Processing text_right with encode:  45%|████▍     | 2666/5961 [00:01<00:02, 1639.19it/s]Processing text_right with encode:  48%|████▊     | 2856/5961 [00:01<00:01, 1705.50it/s]Processing text_right with encode:  51%|█████     | 3027/5961 [00:01<00:01, 1696.14it/s]Processing text_right with encode:  54%|█████▍    | 3207/5961 [00:01<00:01, 1725.28it/s]Processing text_right with encode:  57%|█████▋    | 3380/5961 [00:02<00:01, 1649.74it/s]Processing text_right with encode:  59%|█████▉    | 3546/5961 [00:02<00:01, 1648.11it/s]Processing text_right with encode:  62%|██████▏   | 3712/5961 [00:02<00:01, 1647.10it/s]Processing text_right with encode:  65%|██████▌   | 3880/5961 [00:02<00:01, 1656.62it/s]Processing text_right with encode:  68%|██████▊   | 4059/5961 [00:02<00:01, 1693.33it/s]Processing text_right with encode:  71%|███████   | 4229/5961 [00:02<00:01, 1668.74it/s]Processing text_right with encode:  74%|███████▍  | 4398/5961 [00:02<00:00, 1674.96it/s]Processing text_right with encode:  77%|███████▋  | 4568/5961 [00:02<00:00, 1681.84it/s]Processing text_right with encode:  79%|███████▉  | 4737/5961 [00:02<00:00, 1656.56it/s]Processing text_right with encode:  82%|████████▏ | 4903/5961 [00:02<00:00, 1623.87it/s]Processing text_right with encode:  85%|████████▍ | 5066/5961 [00:03<00:00, 1615.08it/s]Processing text_right with encode:  88%|████████▊ | 5239/5961 [00:03<00:00, 1646.93it/s]Processing text_right with encode:  91%|█████████ | 5405/5961 [00:03<00:00, 1616.59it/s]Processing text_right with encode:  93%|█████████▎| 5568/5961 [00:03<00:00, 1590.48it/s]Processing text_right with encode:  96%|█████████▌| 5728/5961 [00:03<00:00, 1554.82it/s]Processing text_right with encode:  99%|█████████▉| 5904/5961 [00:03<00:00, 1609.19it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1653.46it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 454855.99it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 714370.30it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [01:44<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [01:44<?, ?it/s, loss=0.921]Epoch 1/1:   1%|          | 1/102 [01:44<2:56:18, 104.74s/it, loss=0.921]Epoch 1/1:   1%|          | 1/102 [02:50<2:56:18, 104.74s/it, loss=0.921]Epoch 1/1:   1%|          | 1/102 [02:50<2:56:18, 104.74s/it, loss=0.842]Epoch 1/1:   2%|▏         | 2/102 [02:50<2:35:08, 93.09s/it, loss=0.842] Epoch 1/1:   2%|▏         | 2/102 [04:00<2:35:08, 93.09s/it, loss=0.842]Epoch 1/1:   2%|▏         | 2/102 [04:00<2:35:08, 93.09s/it, loss=1.000]Epoch 1/1:   3%|▎         | 3/102 [04:00<2:21:52, 85.98s/it, loss=1.000]Epoch 1/1:   3%|▎         | 3/102 [04:37<2:21:52, 85.98s/it, loss=1.000]Epoch 1/1:   3%|▎         | 3/102 [04:37<2:21:52, 85.98s/it, loss=1.020]Epoch 1/1:   4%|▍         | 4/102 [04:37<1:56:42, 71.46s/it, loss=1.020]Epoch 1/1:   4%|▍         | 4/102 [06:18<1:56:42, 71.46s/it, loss=1.020]Epoch 1/1:   4%|▍         | 4/102 [06:18<1:56:42, 71.46s/it, loss=1.064]Epoch 1/1:   5%|▍         | 5/102 [06:18<2:09:33, 80.14s/it, loss=1.064]Epoch 1/1:   5%|▍         | 5/102 [07:10<2:09:33, 80.14s/it, loss=1.064]Epoch 1/1:   5%|▍         | 5/102 [07:10<2:09:33, 80.14s/it, loss=0.975]Epoch 1/1:   6%|▌         | 6/102 [07:10<1:54:53, 71.81s/it, loss=0.975]Epoch 1/1:   6%|▌         | 6/102 [08:30<1:54:53, 71.81s/it, loss=0.975]Epoch 1/1:   6%|▌         | 6/102 [08:30<1:54:53, 71.81s/it, loss=1.053]Epoch 1/1:   7%|▋         | 7/102 [08:30<1:57:48, 74.41s/it, loss=1.053]Epoch 1/1:   7%|▋         | 7/102 [09:05<1:57:48, 74.41s/it, loss=1.053]Epoch 1/1:   7%|▋         | 7/102 [09:05<1:57:48, 74.41s/it, loss=0.943]Epoch 1/1:   8%|▊         | 8/102 [09:05<1:37:59, 62.55s/it, loss=0.943]Epoch 1/1:   8%|▊         | 8/102 [09:42<1:37:59, 62.55s/it, loss=0.943]Epoch 1/1:   8%|▊         | 8/102 [09:42<1:37:59, 62.55s/it, loss=0.770]Epoch 1/1:   9%|▉         | 9/102 [09:42<1:24:57, 54.82s/it, loss=0.770]Epoch 1/1:   9%|▉         | 9/102 [11:22<1:24:57, 54.82s/it, loss=0.770]Epoch 1/1:   9%|▉         | 9/102 [11:22<1:24:57, 54.82s/it, loss=0.913]Epoch 1/1:  10%|▉         | 10/102 [11:22<1:44:59, 68.47s/it, loss=0.913]Epoch 1/1:  10%|▉         | 10/102 [13:18<1:44:59, 68.47s/it, loss=0.913]Epoch 1/1:  10%|▉         | 10/102 [13:18<1:44:59, 68.47s/it, loss=0.989]Epoch 1/1:  11%|█         | 11/102 [13:18<2:05:24, 82.68s/it, loss=0.989]Epoch 1/1:  11%|█         | 11/102 [15:35<2:05:24, 82.68s/it, loss=0.989]Epoch 1/1:  11%|█         | 11/102 [15:35<2:05:24, 82.68s/it, loss=0.857]Epoch 1/1:  12%|█▏        | 12/102 [15:35<2:28:17, 98.86s/it, loss=0.857]Epoch 1/1:  12%|█▏        | 12/102 [17:28<2:28:17, 98.86s/it, loss=0.857]Epoch 1/1:  12%|█▏        | 12/102 [17:28<2:28:17, 98.86s/it, loss=0.782]Epoch 1/1:  13%|█▎        | 13/102 [17:28<2:32:58, 103.13s/it, loss=0.782]Epoch 1/1:  13%|█▎        | 13/102 [20:03<2:32:58, 103.13s/it, loss=0.782]Epoch 1/1:  13%|█▎        | 13/102 [20:03<2:32:58, 103.13s/it, loss=0.824]Epoch 1/1:  14%|█▎        | 14/102 [20:03<2:54:05, 118.70s/it, loss=0.824]Epoch 1/1:  14%|█▎        | 14/102 [22:13<2:54:05, 118.70s/it, loss=0.824]Epoch 1/1:  14%|█▎        | 14/102 [22:13<2:54:05, 118.70s/it, loss=0.692]Epoch 1/1:  15%|█▍        | 15/102 [22:13<2:57:04, 122.12s/it, loss=0.692]Epoch 1/1:  15%|█▍        | 15/102 [24:01<2:57:04, 122.12s/it, loss=0.692]Epoch 1/1:  15%|█▍        | 15/102 [24:01<2:57:04, 122.12s/it, loss=0.481]Epoch 1/1:  16%|█▌        | 16/102 [24:01<2:48:51, 117.80s/it, loss=0.481]Epoch 1/1:  16%|█▌        | 16/102 [25:24<2:48:51, 117.80s/it, loss=0.481]Epoch 1/1:  16%|█▌        | 16/102 [25:24<2:48:51, 117.80s/it, loss=0.597]Epoch 1/1:  17%|█▋        | 17/102 [25:24<2:32:19, 107.53s/it, loss=0.597]Killed

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
   50a9e00..edcb75b  master     -> origin/master
Updating 50a9e00..edcb75b
Fast-forward
 deps.txt                                           |    6 +-
 error_list/20200516/list_log_benchmark_20200516.md |  182 +-
 error_list/20200516/list_log_import_20200516.md    |    2 +-
 error_list/20200516/list_log_jupyter_20200516.md   | 1749 ++++++++++----------
 error_list/20200516/list_log_testall_20200516.md   |  386 +++--
 ...-10_76b7a81be9b27c2e92c4951280c0a8da664b997c.py |  622 +++++++
 6 files changed, 1811 insertions(+), 1136 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-16-17-10_76b7a81be9b27c2e92c4951280c0a8da664b997c.py
[master e8a3400] ml_store
 1 file changed, 83 insertions(+)
To github.com:arita37/mlmodels_store.git
   edcb75b..e8a3400  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd0a8e22d90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd0a8e22d90>

  function with postional parmater data_info <function get_dataset_torch at 0x7fd0a8e22d90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 457325.97it/s] 93%|█████████▎| 9248768/9912422 [00:00<00:01, 651928.40it/s]9920512it [00:00, 45896667.27it/s]                           
0it [00:00, ?it/s]32768it [00:00, 686642.02it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 487579.66it/s]1654784it [00:00, 12393650.12it/s]                         
0it [00:00, ?it/s]8192it [00:00, 212048.72it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd0a8156ae8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd0a8156ae8>

  function with postional parmater data_info <function get_dataset_torch at 0x7fd0a8156ae8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0018893660505612691 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.011006573915481568 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0012023999094963073 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.011323766946792603 	 Accuracy: 1
model saves at 1 accuracy

  #### Predict   ##################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd0a81568c8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd0a81568c8>

  function with postional parmater data_info <function get_dataset_torch at 0x7fd0a81568c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  #### metrics   ##################################################### 
None

  #### Plot   ######################################################## 

  #### Save  ######################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 

  #### Load   ######################################################## 
<__main__.Model object at 0x7fd0a87cc908>

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
<__main__.Model object at 0x7fd0a0df2b00>

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
