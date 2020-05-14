
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
[master 99ee0b7] ml_store
 1 file changed, 59 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-14-04-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
To github.com:arita37/mlmodels_store.git
   a3030c7..99ee0b7  master -> master





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
[master 7dad9f6] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   99ee0b7..7dad9f6  master -> master





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
[master 47a710e] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   7dad9f6..47a710e  master -> master





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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-14 04:12:05.249584: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 04:12:05.255317: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 04:12:05.255479: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562b2e4e7b40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 04:12:05.255493: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 173
Trainable params: 173
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 1ms/sample - loss: 0.2501 - binary_crossentropy: 0.6932 - val_loss: 0.2500 - val_binary_crossentropy: 0.6930

  #### metrics   #################################################### 
{'MSE': 0.2497989532973219}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
Total params: 173
Trainable params: 173
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 483
Trainable params: 483
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2482 - binary_crossentropy: 0.6939500/500 [==============================] - 1s 2ms/sample - loss: 0.2753 - binary_crossentropy: 0.7753 - val_loss: 0.2775 - val_binary_crossentropy: 0.7533

  #### metrics   #################################################### 
{'MSE': 0.27601783806121444}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 483
Trainable params: 483
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 597
Trainable params: 597
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 2ms/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2497 - val_binary_crossentropy: 0.6925

  #### metrics   #################################################### 
{'MSE': 0.24967416997660807}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 597
Trainable params: 597
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 4, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.3281 - binary_crossentropy: 1.6439500/500 [==============================] - 1s 3ms/sample - loss: 0.3303 - binary_crossentropy: 2.1108 - val_loss: 0.2996 - val_binary_crossentropy: 1.7611

  #### metrics   #################################################### 
{'MSE': 0.30720286422161086}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 4, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
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
Total params: 168
Trainable params: 168
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.5300 - binary_crossentropy: 8.1225500/500 [==============================] - 2s 3ms/sample - loss: 0.4680 - binary_crossentropy: 7.1757 - val_loss: 0.4940 - val_binary_crossentropy: 7.6199

  #### metrics   #################################################### 
{'MSE': 0.482}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
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
Total params: 168
Trainable params: 168
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-14 04:13:26.807294: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:13:26.809308: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:13:26.814994: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-14 04:13:26.825099: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-14 04:13:26.826893: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 04:13:26.828445: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:13:26.829931: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2447 - val_binary_crossentropy: 0.6825
2020-05-14 04:13:28.246441: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:13:28.248228: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:13:28.253067: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-14 04:13:28.262799: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-14 04:13:28.264444: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 04:13:28.265935: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:13:28.267289: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2421535363941949}

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
2020-05-14 04:13:51.878265: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:13:51.879587: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:13:51.883052: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-14 04:13:51.889101: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-14 04:13:51.890121: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 04:13:51.891057: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:13:51.891926: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2508 - val_binary_crossentropy: 0.6947
2020-05-14 04:13:53.428802: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:13:53.429940: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:13:53.432462: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-14 04:13:53.437416: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-14 04:13:53.438266: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 04:13:53.439061: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:13:53.439880: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25092009736044335}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-14 04:14:27.149186: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:14:27.154634: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:14:27.170835: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-14 04:14:27.200708: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-14 04:14:27.205884: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 04:14:27.210646: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:14:27.215235: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.5861 - binary_crossentropy: 1.4505 - val_loss: 0.2507 - val_binary_crossentropy: 0.6945
2020-05-14 04:14:29.525056: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:14:29.530216: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:14:29.543006: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-14 04:14:29.568654: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-14 04:14:29.573118: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-14 04:14:29.577317: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-14 04:14:29.581599: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2588505192440369}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 645
Trainable params: 645
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2742 - binary_crossentropy: 0.7432500/500 [==============================] - 4s 9ms/sample - loss: 0.2722 - binary_crossentropy: 0.7391 - val_loss: 0.2722 - val_binary_crossentropy: 0.7390

  #### metrics   #################################################### 
{'MSE': 0.2715422853088129}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 645
Trainable params: 645
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 2)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         4           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         14          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         18          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         12          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
Total params: 254
Trainable params: 254
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2512 - binary_crossentropy: 0.6983500/500 [==============================] - 5s 9ms/sample - loss: 0.2624 - binary_crossentropy: 0.7469 - val_loss: 0.2656 - val_binary_crossentropy: 0.7526

  #### metrics   #################################################### 
{'MSE': 0.26089795081264405}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 2)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         4           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         14          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         18          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         12          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 6s - loss: 0.2541 - binary_crossentropy: 0.6996500/500 [==============================] - 5s 9ms/sample - loss: 0.2519 - binary_crossentropy: 0.6967 - val_loss: 0.2524 - val_binary_crossentropy: 0.7505

  #### metrics   #################################################### 
{'MSE': 0.25157050945615306}

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
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
regionsequence_sum (InputLayer) [(None, 2)]          0                                            
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
region_10sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 9s - loss: 0.3257 - binary_crossentropy: 2.2832500/500 [==============================] - 6s 12ms/sample - loss: 0.2945 - binary_crossentropy: 1.8768 - val_loss: 0.2892 - val_binary_crossentropy: 1.7875

  #### metrics   #################################################### 
{'MSE': 0.2917312491512539}

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
regionsequence_sum (InputLayer) [(None, 2)]          0                                            
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
region_10sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_max[0][0]               
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
100/500 [=====>........................] - ETA: 7s - loss: 0.2809 - binary_crossentropy: 1.9173500/500 [==============================] - 6s 11ms/sample - loss: 0.3331 - binary_crossentropy: 2.6153 - val_loss: 0.3073 - val_binary_crossentropy: 2.3467

  #### metrics   #################################################### 
{'MSE': 0.31849699421965666}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         32          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         32          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         32          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         32          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_11[0][0]                    
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
Total params: 3,205
Trainable params: 3,125
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2871 - binary_crossentropy: 0.7780500/500 [==============================] - 6s 13ms/sample - loss: 0.2668 - binary_crossentropy: 0.7310 - val_loss: 0.2681 - val_binary_crossentropy: 0.7319

  #### metrics   #################################################### 
{'MSE': 0.2643489509929266}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         32          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         32          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         32          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         32          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_11[0][0]                    
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
Total params: 3,205
Trainable params: 3,125
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         4           sequence_max[0][0]               
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
Total params: 433
Trainable params: 433
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 6s 12ms/sample - loss: 0.2500 - binary_crossentropy: 0.6932 - val_loss: 0.2500 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.24999962175571105}

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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         4           sequence_max[0][0]               
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
Total params: 433
Trainable params: 433
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
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
Total params: 2,034
Trainable params: 2,034
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2925 - binary_crossentropy: 1.5701500/500 [==============================] - 7s 13ms/sample - loss: 0.2659 - binary_crossentropy: 1.0909 - val_loss: 0.2586 - val_binary_crossentropy: 0.9194

  #### metrics   #################################################### 
{'MSE': 0.26109934371742893}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
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
Total params: 2,034
Trainable params: 2,034
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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 276
Trainable params: 276
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.5800 - binary_crossentropy: 8.9465500/500 [==============================] - 7s 14ms/sample - loss: 0.5100 - binary_crossentropy: 7.8667 - val_loss: 0.5040 - val_binary_crossentropy: 7.7742

  #### metrics   #################################################### 
{'MSE': 0.507}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 276
Trainable params: 276
Non-trainable params: 0
__________________________________________________________________________________________________

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   47a710e..56f4086  master     -> origin/master
Updating 47a710e..56f4086
Fast-forward
 error_list/20200514/list_log_import_20200514.md    |   2 +-
 .../20200514/list_log_pullrequest_20200514.md      |   2 +-
 ...-10_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 620 +++++++++++++++++++++
 3 files changed, 622 insertions(+), 2 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-14-04-10_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
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
[master 4162959] ml_store
 1 file changed, 5670 insertions(+)
To github.com:arita37/mlmodels_store.git
   56f4086..4162959  master -> master





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
[master 291a614] ml_store
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   4162959..291a614  master -> master





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
[master f6be686] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   291a614..f6be686  master -> master





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
[master 429e20d] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   f6be686..429e20d  master -> master





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

2020-05-14 04:27:23.263561: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 04:27:23.269226: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 04:27:23.269469: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fb85a5ce90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 04:27:23.269486: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 6s - loss: 1.3862
256/354 [====================>.........] - ETA: 2s - loss: 1.2092
354/354 [==============================] - 11s 30ms/step - loss: 1.4049 - val_loss: 2.0604

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
[master b7c78b6] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   429e20d..b7c78b6  master -> master





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
[master 632ee12] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   b7c78b6..632ee12  master -> master





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
[master 391570f] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   632ee12..391570f  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3203072/17464789 [====>.........................] - ETA: 0s
11354112/17464789 [==================>...........] - ETA: 0s
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
2020-05-14 04:28:20.174009: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 04:28:20.178381: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-14 04:28:20.178539: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561edeb089c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 04:28:20.178553: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7740 - accuracy: 0.4930 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7944 - accuracy: 0.4917
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8123 - accuracy: 0.4905
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7341 - accuracy: 0.4956
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7791 - accuracy: 0.4927
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7674 - accuracy: 0.4934
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7069 - accuracy: 0.4974
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6734 - accuracy: 0.4996
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6620 - accuracy: 0.5003
11000/25000 [============>.................] - ETA: 3s - loss: 7.6959 - accuracy: 0.4981
12000/25000 [=============>................] - ETA: 3s - loss: 7.6832 - accuracy: 0.4989
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7280 - accuracy: 0.4960
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7214 - accuracy: 0.4964
15000/25000 [=================>............] - ETA: 2s - loss: 7.7545 - accuracy: 0.4943
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7605 - accuracy: 0.4939
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7514 - accuracy: 0.4945
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7245 - accuracy: 0.4962
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7118 - accuracy: 0.4971
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7088 - accuracy: 0.4972
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6936 - accuracy: 0.4982
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6889 - accuracy: 0.4985
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6800 - accuracy: 0.4991
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6730 - accuracy: 0.4996
25000/25000 [==============================] - 8s 303us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f0c3079cac8>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f0c3aa47978> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.7433 - accuracy: 0.4950
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5823 - accuracy: 0.5055 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6411 - accuracy: 0.5017
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6091 - accuracy: 0.5038
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5532 - accuracy: 0.5074
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5618 - accuracy: 0.5068
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5768 - accuracy: 0.5059
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5440 - accuracy: 0.5080
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5337 - accuracy: 0.5087
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5792 - accuracy: 0.5057
11000/25000 [============>.................] - ETA: 3s - loss: 7.6095 - accuracy: 0.5037
12000/25000 [=============>................] - ETA: 3s - loss: 7.6551 - accuracy: 0.5008
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6324 - accuracy: 0.5022
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6425 - accuracy: 0.5016
15000/25000 [=================>............] - ETA: 2s - loss: 7.6165 - accuracy: 0.5033
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6388 - accuracy: 0.5018
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6314 - accuracy: 0.5023
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6343 - accuracy: 0.5021
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6279 - accuracy: 0.5025
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6582 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6681 - accuracy: 0.4999
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6457 - accuracy: 0.5014
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6633 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6705 - accuracy: 0.4997
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

 1000/25000 [>.............................] - ETA: 11s - loss: 7.0226 - accuracy: 0.5420
 2000/25000 [=>............................] - ETA: 8s - loss: 7.2373 - accuracy: 0.5280 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.4417 - accuracy: 0.5147
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6206 - accuracy: 0.5030
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5900 - accuracy: 0.5050
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6155 - accuracy: 0.5033
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6417 - accuracy: 0.5016
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6530 - accuracy: 0.5009
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6452 - accuracy: 0.5014
11000/25000 [============>.................] - ETA: 3s - loss: 7.6569 - accuracy: 0.5006
12000/25000 [=============>................] - ETA: 3s - loss: 7.7113 - accuracy: 0.4971
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7079 - accuracy: 0.4973
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7170 - accuracy: 0.4967
15000/25000 [=================>............] - ETA: 2s - loss: 7.6850 - accuracy: 0.4988
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6848 - accuracy: 0.4988
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6892 - accuracy: 0.4985
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6734 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6889 - accuracy: 0.4985
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6790 - accuracy: 0.4992
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6749 - accuracy: 0.4995
25000/25000 [==============================] - 8s 307us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   391570f..f2a37ee  master     -> origin/master
Updating 391570f..f2a37ee
Fast-forward
 error_list/20200514/list_log_import_20200514.md   |    2 +-
 error_list/20200514/list_log_jupyter_20200514.md  | 1788 ++++++++++-----------
 error_list/20200514/list_log_test_cli_20200514.md |  138 +-
 3 files changed, 958 insertions(+), 970 deletions(-)
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
[master f02b8d7] ml_store
 1 file changed, 324 insertions(+)
To github.com:arita37/mlmodels_store.git
   f2a37ee..f02b8d7  master -> master





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

13/13 [==============================] - 2s 129ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 8/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 9/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master 3a05676] ml_store
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   f02b8d7..3a05676  master -> master





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
 3497984/11490434 [========>.....................] - ETA: 0s
11165696/11490434 [============================>.] - ETA: 0s
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

   32/60000 [..............................] - ETA: 8:05 - loss: 2.3728 - categorical_accuracy: 0.0312
   64/60000 [..............................] - ETA: 4:52 - loss: 2.3183 - categorical_accuracy: 0.1094
   96/60000 [..............................] - ETA: 3:47 - loss: 2.2765 - categorical_accuracy: 0.1771
  128/60000 [..............................] - ETA: 3:15 - loss: 2.2615 - categorical_accuracy: 0.1875
  192/60000 [..............................] - ETA: 2:39 - loss: 2.2572 - categorical_accuracy: 0.1771
  224/60000 [..............................] - ETA: 2:30 - loss: 2.2339 - categorical_accuracy: 0.1875
  288/60000 [..............................] - ETA: 2:17 - loss: 2.1754 - categorical_accuracy: 0.2292
  320/60000 [..............................] - ETA: 2:13 - loss: 2.1441 - categorical_accuracy: 0.2469
  384/60000 [..............................] - ETA: 2:06 - loss: 2.0415 - categorical_accuracy: 0.2995
  448/60000 [..............................] - ETA: 2:01 - loss: 1.9883 - categorical_accuracy: 0.3192
  512/60000 [..............................] - ETA: 1:57 - loss: 1.9289 - categorical_accuracy: 0.3438
  576/60000 [..............................] - ETA: 1:54 - loss: 1.8799 - categorical_accuracy: 0.3628
  640/60000 [..............................] - ETA: 1:51 - loss: 1.7997 - categorical_accuracy: 0.3922
  704/60000 [..............................] - ETA: 1:49 - loss: 1.7367 - categorical_accuracy: 0.4134
  768/60000 [..............................] - ETA: 1:47 - loss: 1.6764 - categorical_accuracy: 0.4375
  832/60000 [..............................] - ETA: 1:46 - loss: 1.6527 - categorical_accuracy: 0.4411
  864/60000 [..............................] - ETA: 1:46 - loss: 1.6173 - categorical_accuracy: 0.4549
  928/60000 [..............................] - ETA: 1:44 - loss: 1.5832 - categorical_accuracy: 0.4666
  960/60000 [..............................] - ETA: 1:44 - loss: 1.5560 - categorical_accuracy: 0.4719
 1024/60000 [..............................] - ETA: 1:43 - loss: 1.5124 - categorical_accuracy: 0.4902
 1088/60000 [..............................] - ETA: 1:42 - loss: 1.4674 - categorical_accuracy: 0.5037
 1152/60000 [..............................] - ETA: 1:41 - loss: 1.4339 - categorical_accuracy: 0.5122
 1216/60000 [..............................] - ETA: 1:40 - loss: 1.3925 - categorical_accuracy: 0.5271
 1280/60000 [..............................] - ETA: 1:40 - loss: 1.3544 - categorical_accuracy: 0.5398
 1344/60000 [..............................] - ETA: 1:39 - loss: 1.3242 - categorical_accuracy: 0.5484
 1408/60000 [..............................] - ETA: 1:38 - loss: 1.2951 - categorical_accuracy: 0.5611
 1440/60000 [..............................] - ETA: 1:38 - loss: 1.2819 - categorical_accuracy: 0.5660
 1504/60000 [..............................] - ETA: 1:38 - loss: 1.2570 - categorical_accuracy: 0.5758
 1536/60000 [..............................] - ETA: 1:38 - loss: 1.2412 - categorical_accuracy: 0.5814
 1568/60000 [..............................] - ETA: 1:38 - loss: 1.2283 - categorical_accuracy: 0.5855
 1632/60000 [..............................] - ETA: 1:37 - loss: 1.1988 - categorical_accuracy: 0.5968
 1664/60000 [..............................] - ETA: 1:37 - loss: 1.1846 - categorical_accuracy: 0.6010
 1696/60000 [..............................] - ETA: 1:37 - loss: 1.1731 - categorical_accuracy: 0.6044
 1760/60000 [..............................] - ETA: 1:37 - loss: 1.1528 - categorical_accuracy: 0.6119
 1824/60000 [..............................] - ETA: 1:36 - loss: 1.1327 - categorical_accuracy: 0.6195
 1856/60000 [..............................] - ETA: 1:36 - loss: 1.1245 - categorical_accuracy: 0.6228
 1920/60000 [..............................] - ETA: 1:36 - loss: 1.1083 - categorical_accuracy: 0.6286
 1984/60000 [..............................] - ETA: 1:35 - loss: 1.0968 - categorical_accuracy: 0.6356
 2048/60000 [>.............................] - ETA: 1:35 - loss: 1.0802 - categorical_accuracy: 0.6411
 2080/60000 [>.............................] - ETA: 1:35 - loss: 1.0703 - categorical_accuracy: 0.6447
 2144/60000 [>.............................] - ETA: 1:35 - loss: 1.0551 - categorical_accuracy: 0.6502
 2208/60000 [>.............................] - ETA: 1:34 - loss: 1.0364 - categorical_accuracy: 0.6581
 2272/60000 [>.............................] - ETA: 1:34 - loss: 1.0221 - categorical_accuracy: 0.6633
 2336/60000 [>.............................] - ETA: 1:34 - loss: 1.0115 - categorical_accuracy: 0.6665
 2400/60000 [>.............................] - ETA: 1:33 - loss: 0.9980 - categorical_accuracy: 0.6700
 2464/60000 [>.............................] - ETA: 1:33 - loss: 0.9869 - categorical_accuracy: 0.6733
 2528/60000 [>.............................] - ETA: 1:33 - loss: 0.9732 - categorical_accuracy: 0.6788
 2592/60000 [>.............................] - ETA: 1:33 - loss: 0.9594 - categorical_accuracy: 0.6821
 2624/60000 [>.............................] - ETA: 1:33 - loss: 0.9568 - categorical_accuracy: 0.6825
 2656/60000 [>.............................] - ETA: 1:33 - loss: 0.9503 - categorical_accuracy: 0.6845
 2688/60000 [>.............................] - ETA: 1:33 - loss: 0.9455 - categorical_accuracy: 0.6868
 2720/60000 [>.............................] - ETA: 1:33 - loss: 0.9371 - categorical_accuracy: 0.6901
 2784/60000 [>.............................] - ETA: 1:32 - loss: 0.9271 - categorical_accuracy: 0.6947
 2848/60000 [>.............................] - ETA: 1:32 - loss: 0.9199 - categorical_accuracy: 0.6984
 2912/60000 [>.............................] - ETA: 1:32 - loss: 0.9121 - categorical_accuracy: 0.7019
 2976/60000 [>.............................] - ETA: 1:31 - loss: 0.9002 - categorical_accuracy: 0.7067
 3040/60000 [>.............................] - ETA: 1:31 - loss: 0.8914 - categorical_accuracy: 0.7089
 3072/60000 [>.............................] - ETA: 1:31 - loss: 0.8850 - categorical_accuracy: 0.7113
 3104/60000 [>.............................] - ETA: 1:31 - loss: 0.8813 - categorical_accuracy: 0.7126
 3168/60000 [>.............................] - ETA: 1:31 - loss: 0.8694 - categorical_accuracy: 0.7159
 3232/60000 [>.............................] - ETA: 1:31 - loss: 0.8561 - categorical_accuracy: 0.7203
 3296/60000 [>.............................] - ETA: 1:31 - loss: 0.8499 - categorical_accuracy: 0.7227
 3360/60000 [>.............................] - ETA: 1:30 - loss: 0.8381 - categorical_accuracy: 0.7262
 3424/60000 [>.............................] - ETA: 1:30 - loss: 0.8277 - categorical_accuracy: 0.7296
 3488/60000 [>.............................] - ETA: 1:30 - loss: 0.8206 - categorical_accuracy: 0.7322
 3552/60000 [>.............................] - ETA: 1:30 - loss: 0.8109 - categorical_accuracy: 0.7356
 3616/60000 [>.............................] - ETA: 1:29 - loss: 0.8064 - categorical_accuracy: 0.7370
 3648/60000 [>.............................] - ETA: 1:29 - loss: 0.8025 - categorical_accuracy: 0.7379
 3712/60000 [>.............................] - ETA: 1:29 - loss: 0.7951 - categorical_accuracy: 0.7408
 3776/60000 [>.............................] - ETA: 1:29 - loss: 0.7881 - categorical_accuracy: 0.7434
 3840/60000 [>.............................] - ETA: 1:29 - loss: 0.7812 - categorical_accuracy: 0.7466
 3904/60000 [>.............................] - ETA: 1:29 - loss: 0.7751 - categorical_accuracy: 0.7485
 3968/60000 [>.............................] - ETA: 1:28 - loss: 0.7659 - categorical_accuracy: 0.7515
 4032/60000 [=>............................] - ETA: 1:28 - loss: 0.7576 - categorical_accuracy: 0.7547
 4096/60000 [=>............................] - ETA: 1:28 - loss: 0.7530 - categorical_accuracy: 0.7566
 4160/60000 [=>............................] - ETA: 1:28 - loss: 0.7443 - categorical_accuracy: 0.7594
 4192/60000 [=>............................] - ETA: 1:28 - loss: 0.7422 - categorical_accuracy: 0.7600
 4256/60000 [=>............................] - ETA: 1:28 - loss: 0.7365 - categorical_accuracy: 0.7617
 4288/60000 [=>............................] - ETA: 1:28 - loss: 0.7326 - categorical_accuracy: 0.7631
 4352/60000 [=>............................] - ETA: 1:28 - loss: 0.7238 - categorical_accuracy: 0.7659
 4416/60000 [=>............................] - ETA: 1:27 - loss: 0.7199 - categorical_accuracy: 0.7672
 4480/60000 [=>............................] - ETA: 1:27 - loss: 0.7127 - categorical_accuracy: 0.7692
 4544/60000 [=>............................] - ETA: 1:27 - loss: 0.7093 - categorical_accuracy: 0.7713
 4608/60000 [=>............................] - ETA: 1:27 - loss: 0.7035 - categorical_accuracy: 0.7732
 4672/60000 [=>............................] - ETA: 1:27 - loss: 0.6965 - categorical_accuracy: 0.7759
 4736/60000 [=>............................] - ETA: 1:27 - loss: 0.6911 - categorical_accuracy: 0.7774
 4800/60000 [=>............................] - ETA: 1:26 - loss: 0.6869 - categorical_accuracy: 0.7792
 4864/60000 [=>............................] - ETA: 1:26 - loss: 0.6813 - categorical_accuracy: 0.7810
 4928/60000 [=>............................] - ETA: 1:26 - loss: 0.6778 - categorical_accuracy: 0.7827
 4992/60000 [=>............................] - ETA: 1:26 - loss: 0.6729 - categorical_accuracy: 0.7839
 5024/60000 [=>............................] - ETA: 1:26 - loss: 0.6702 - categorical_accuracy: 0.7848
 5088/60000 [=>............................] - ETA: 1:26 - loss: 0.6684 - categorical_accuracy: 0.7852
 5120/60000 [=>............................] - ETA: 1:26 - loss: 0.6662 - categorical_accuracy: 0.7855
 5184/60000 [=>............................] - ETA: 1:26 - loss: 0.6616 - categorical_accuracy: 0.7868
 5248/60000 [=>............................] - ETA: 1:26 - loss: 0.6560 - categorical_accuracy: 0.7889
 5312/60000 [=>............................] - ETA: 1:25 - loss: 0.6502 - categorical_accuracy: 0.7907
 5376/60000 [=>............................] - ETA: 1:25 - loss: 0.6456 - categorical_accuracy: 0.7919
 5440/60000 [=>............................] - ETA: 1:25 - loss: 0.6413 - categorical_accuracy: 0.7930
 5472/60000 [=>............................] - ETA: 1:25 - loss: 0.6404 - categorical_accuracy: 0.7937
 5504/60000 [=>............................] - ETA: 1:25 - loss: 0.6375 - categorical_accuracy: 0.7945
 5536/60000 [=>............................] - ETA: 1:25 - loss: 0.6352 - categorical_accuracy: 0.7952
 5600/60000 [=>............................] - ETA: 1:25 - loss: 0.6325 - categorical_accuracy: 0.7955
 5664/60000 [=>............................] - ETA: 1:25 - loss: 0.6279 - categorical_accuracy: 0.7971
 5728/60000 [=>............................] - ETA: 1:24 - loss: 0.6235 - categorical_accuracy: 0.7984
 5792/60000 [=>............................] - ETA: 1:24 - loss: 0.6218 - categorical_accuracy: 0.7990
 5856/60000 [=>............................] - ETA: 1:24 - loss: 0.6207 - categorical_accuracy: 0.7997
 5920/60000 [=>............................] - ETA: 1:24 - loss: 0.6182 - categorical_accuracy: 0.8005
 5984/60000 [=>............................] - ETA: 1:24 - loss: 0.6133 - categorical_accuracy: 0.8023
 6048/60000 [==>...........................] - ETA: 1:24 - loss: 0.6092 - categorical_accuracy: 0.8036
 6112/60000 [==>...........................] - ETA: 1:24 - loss: 0.6051 - categorical_accuracy: 0.8050
 6176/60000 [==>...........................] - ETA: 1:23 - loss: 0.6036 - categorical_accuracy: 0.8055
 6208/60000 [==>...........................] - ETA: 1:23 - loss: 0.6012 - categorical_accuracy: 0.8064
 6272/60000 [==>...........................] - ETA: 1:23 - loss: 0.5978 - categorical_accuracy: 0.8074
 6336/60000 [==>...........................] - ETA: 1:23 - loss: 0.5934 - categorical_accuracy: 0.8087
 6400/60000 [==>...........................] - ETA: 1:23 - loss: 0.5900 - categorical_accuracy: 0.8100
 6464/60000 [==>...........................] - ETA: 1:23 - loss: 0.5887 - categorical_accuracy: 0.8103
 6528/60000 [==>...........................] - ETA: 1:23 - loss: 0.5857 - categorical_accuracy: 0.8111
 6560/60000 [==>...........................] - ETA: 1:23 - loss: 0.5834 - categorical_accuracy: 0.8119
 6624/60000 [==>...........................] - ETA: 1:23 - loss: 0.5807 - categorical_accuracy: 0.8127
 6688/60000 [==>...........................] - ETA: 1:22 - loss: 0.5776 - categorical_accuracy: 0.8135
 6720/60000 [==>...........................] - ETA: 1:22 - loss: 0.5759 - categorical_accuracy: 0.8143
 6784/60000 [==>...........................] - ETA: 1:22 - loss: 0.5738 - categorical_accuracy: 0.8153
 6848/60000 [==>...........................] - ETA: 1:22 - loss: 0.5716 - categorical_accuracy: 0.8160
 6912/60000 [==>...........................] - ETA: 1:22 - loss: 0.5683 - categorical_accuracy: 0.8171
 6944/60000 [==>...........................] - ETA: 1:22 - loss: 0.5662 - categorical_accuracy: 0.8178
 6976/60000 [==>...........................] - ETA: 1:22 - loss: 0.5645 - categorical_accuracy: 0.8184
 7040/60000 [==>...........................] - ETA: 1:22 - loss: 0.5621 - categorical_accuracy: 0.8195
 7072/60000 [==>...........................] - ETA: 1:22 - loss: 0.5606 - categorical_accuracy: 0.8200
 7104/60000 [==>...........................] - ETA: 1:22 - loss: 0.5586 - categorical_accuracy: 0.8207
 7136/60000 [==>...........................] - ETA: 1:22 - loss: 0.5565 - categorical_accuracy: 0.8215
 7168/60000 [==>...........................] - ETA: 1:22 - loss: 0.5565 - categorical_accuracy: 0.8213
 7232/60000 [==>...........................] - ETA: 1:22 - loss: 0.5541 - categorical_accuracy: 0.8219
 7296/60000 [==>...........................] - ETA: 1:22 - loss: 0.5525 - categorical_accuracy: 0.8228
 7360/60000 [==>...........................] - ETA: 1:22 - loss: 0.5494 - categorical_accuracy: 0.8238
 7424/60000 [==>...........................] - ETA: 1:21 - loss: 0.5459 - categorical_accuracy: 0.8252
 7488/60000 [==>...........................] - ETA: 1:21 - loss: 0.5427 - categorical_accuracy: 0.8263
 7552/60000 [==>...........................] - ETA: 1:21 - loss: 0.5401 - categorical_accuracy: 0.8272
 7616/60000 [==>...........................] - ETA: 1:21 - loss: 0.5381 - categorical_accuracy: 0.8280
 7680/60000 [==>...........................] - ETA: 1:21 - loss: 0.5359 - categorical_accuracy: 0.8289
 7744/60000 [==>...........................] - ETA: 1:21 - loss: 0.5353 - categorical_accuracy: 0.8295
 7808/60000 [==>...........................] - ETA: 1:21 - loss: 0.5326 - categorical_accuracy: 0.8303
 7872/60000 [==>...........................] - ETA: 1:20 - loss: 0.5312 - categorical_accuracy: 0.8309
 7936/60000 [==>...........................] - ETA: 1:20 - loss: 0.5288 - categorical_accuracy: 0.8313
 8000/60000 [===>..........................] - ETA: 1:20 - loss: 0.5264 - categorical_accuracy: 0.8319
 8032/60000 [===>..........................] - ETA: 1:20 - loss: 0.5249 - categorical_accuracy: 0.8324
 8096/60000 [===>..........................] - ETA: 1:20 - loss: 0.5222 - categorical_accuracy: 0.8331
 8128/60000 [===>..........................] - ETA: 1:20 - loss: 0.5207 - categorical_accuracy: 0.8337
 8160/60000 [===>..........................] - ETA: 1:20 - loss: 0.5204 - categorical_accuracy: 0.8338
 8224/60000 [===>..........................] - ETA: 1:20 - loss: 0.5179 - categorical_accuracy: 0.8346
 8288/60000 [===>..........................] - ETA: 1:20 - loss: 0.5174 - categorical_accuracy: 0.8349
 8320/60000 [===>..........................] - ETA: 1:20 - loss: 0.5158 - categorical_accuracy: 0.8356
 8384/60000 [===>..........................] - ETA: 1:20 - loss: 0.5131 - categorical_accuracy: 0.8364
 8416/60000 [===>..........................] - ETA: 1:20 - loss: 0.5117 - categorical_accuracy: 0.8367
 8448/60000 [===>..........................] - ETA: 1:20 - loss: 0.5117 - categorical_accuracy: 0.8369
 8512/60000 [===>..........................] - ETA: 1:19 - loss: 0.5110 - categorical_accuracy: 0.8372
 8544/60000 [===>..........................] - ETA: 1:19 - loss: 0.5093 - categorical_accuracy: 0.8377
 8608/60000 [===>..........................] - ETA: 1:19 - loss: 0.5072 - categorical_accuracy: 0.8384
 8672/60000 [===>..........................] - ETA: 1:19 - loss: 0.5041 - categorical_accuracy: 0.8394
 8736/60000 [===>..........................] - ETA: 1:19 - loss: 0.5018 - categorical_accuracy: 0.8403
 8800/60000 [===>..........................] - ETA: 1:19 - loss: 0.5011 - categorical_accuracy: 0.8407
 8832/60000 [===>..........................] - ETA: 1:19 - loss: 0.4999 - categorical_accuracy: 0.8410
 8864/60000 [===>..........................] - ETA: 1:19 - loss: 0.4989 - categorical_accuracy: 0.8414
 8896/60000 [===>..........................] - ETA: 1:19 - loss: 0.4974 - categorical_accuracy: 0.8420
 8928/60000 [===>..........................] - ETA: 1:19 - loss: 0.4962 - categorical_accuracy: 0.8422
 8960/60000 [===>..........................] - ETA: 1:19 - loss: 0.4957 - categorical_accuracy: 0.8424
 9024/60000 [===>..........................] - ETA: 1:19 - loss: 0.4933 - categorical_accuracy: 0.8433
 9088/60000 [===>..........................] - ETA: 1:19 - loss: 0.4910 - categorical_accuracy: 0.8440
 9120/60000 [===>..........................] - ETA: 1:19 - loss: 0.4898 - categorical_accuracy: 0.8443
 9152/60000 [===>..........................] - ETA: 1:19 - loss: 0.4883 - categorical_accuracy: 0.8448
 9216/60000 [===>..........................] - ETA: 1:18 - loss: 0.4869 - categorical_accuracy: 0.8454
 9280/60000 [===>..........................] - ETA: 1:18 - loss: 0.4842 - categorical_accuracy: 0.8463
 9344/60000 [===>..........................] - ETA: 1:18 - loss: 0.4814 - categorical_accuracy: 0.8473
 9408/60000 [===>..........................] - ETA: 1:18 - loss: 0.4799 - categorical_accuracy: 0.8479
 9440/60000 [===>..........................] - ETA: 1:18 - loss: 0.4787 - categorical_accuracy: 0.8482
 9504/60000 [===>..........................] - ETA: 1:18 - loss: 0.4774 - categorical_accuracy: 0.8486
 9568/60000 [===>..........................] - ETA: 1:18 - loss: 0.4754 - categorical_accuracy: 0.8491
 9632/60000 [===>..........................] - ETA: 1:18 - loss: 0.4731 - categorical_accuracy: 0.8499
 9696/60000 [===>..........................] - ETA: 1:18 - loss: 0.4715 - categorical_accuracy: 0.8502
 9728/60000 [===>..........................] - ETA: 1:17 - loss: 0.4703 - categorical_accuracy: 0.8507
 9792/60000 [===>..........................] - ETA: 1:17 - loss: 0.4703 - categorical_accuracy: 0.8511
 9856/60000 [===>..........................] - ETA: 1:17 - loss: 0.4689 - categorical_accuracy: 0.8517
 9920/60000 [===>..........................] - ETA: 1:17 - loss: 0.4672 - categorical_accuracy: 0.8522
 9984/60000 [===>..........................] - ETA: 1:17 - loss: 0.4652 - categorical_accuracy: 0.8528
10048/60000 [====>.........................] - ETA: 1:17 - loss: 0.4632 - categorical_accuracy: 0.8534
10080/60000 [====>.........................] - ETA: 1:17 - loss: 0.4628 - categorical_accuracy: 0.8535
10144/60000 [====>.........................] - ETA: 1:17 - loss: 0.4609 - categorical_accuracy: 0.8541
10208/60000 [====>.........................] - ETA: 1:17 - loss: 0.4591 - categorical_accuracy: 0.8545
10272/60000 [====>.........................] - ETA: 1:16 - loss: 0.4577 - categorical_accuracy: 0.8548
10336/60000 [====>.........................] - ETA: 1:16 - loss: 0.4571 - categorical_accuracy: 0.8551
10400/60000 [====>.........................] - ETA: 1:16 - loss: 0.4553 - categorical_accuracy: 0.8556
10464/60000 [====>.........................] - ETA: 1:16 - loss: 0.4538 - categorical_accuracy: 0.8562
10528/60000 [====>.........................] - ETA: 1:16 - loss: 0.4515 - categorical_accuracy: 0.8569
10592/60000 [====>.........................] - ETA: 1:16 - loss: 0.4499 - categorical_accuracy: 0.8573
10656/60000 [====>.........................] - ETA: 1:16 - loss: 0.4481 - categorical_accuracy: 0.8579
10688/60000 [====>.........................] - ETA: 1:16 - loss: 0.4472 - categorical_accuracy: 0.8582
10752/60000 [====>.........................] - ETA: 1:16 - loss: 0.4451 - categorical_accuracy: 0.8588
10816/60000 [====>.........................] - ETA: 1:15 - loss: 0.4434 - categorical_accuracy: 0.8595
10848/60000 [====>.........................] - ETA: 1:15 - loss: 0.4424 - categorical_accuracy: 0.8598
10880/60000 [====>.........................] - ETA: 1:15 - loss: 0.4413 - categorical_accuracy: 0.8602
10944/60000 [====>.........................] - ETA: 1:15 - loss: 0.4408 - categorical_accuracy: 0.8605
11008/60000 [====>.........................] - ETA: 1:15 - loss: 0.4388 - categorical_accuracy: 0.8612
11072/60000 [====>.........................] - ETA: 1:15 - loss: 0.4372 - categorical_accuracy: 0.8616
11136/60000 [====>.........................] - ETA: 1:15 - loss: 0.4351 - categorical_accuracy: 0.8623
11168/60000 [====>.........................] - ETA: 1:15 - loss: 0.4345 - categorical_accuracy: 0.8625
11232/60000 [====>.........................] - ETA: 1:15 - loss: 0.4334 - categorical_accuracy: 0.8627
11296/60000 [====>.........................] - ETA: 1:15 - loss: 0.4323 - categorical_accuracy: 0.8633
11328/60000 [====>.........................] - ETA: 1:15 - loss: 0.4315 - categorical_accuracy: 0.8635
11360/60000 [====>.........................] - ETA: 1:15 - loss: 0.4313 - categorical_accuracy: 0.8636
11424/60000 [====>.........................] - ETA: 1:15 - loss: 0.4296 - categorical_accuracy: 0.8641
11488/60000 [====>.........................] - ETA: 1:15 - loss: 0.4279 - categorical_accuracy: 0.8647
11552/60000 [====>.........................] - ETA: 1:14 - loss: 0.4261 - categorical_accuracy: 0.8654
11616/60000 [====>.........................] - ETA: 1:14 - loss: 0.4252 - categorical_accuracy: 0.8657
11680/60000 [====>.........................] - ETA: 1:14 - loss: 0.4249 - categorical_accuracy: 0.8659
11712/60000 [====>.........................] - ETA: 1:14 - loss: 0.4241 - categorical_accuracy: 0.8661
11776/60000 [====>.........................] - ETA: 1:14 - loss: 0.4240 - categorical_accuracy: 0.8663
11808/60000 [====>.........................] - ETA: 1:14 - loss: 0.4243 - categorical_accuracy: 0.8659
11872/60000 [====>.........................] - ETA: 1:14 - loss: 0.4241 - categorical_accuracy: 0.8662
11904/60000 [====>.........................] - ETA: 1:14 - loss: 0.4237 - categorical_accuracy: 0.8663
11936/60000 [====>.........................] - ETA: 1:14 - loss: 0.4230 - categorical_accuracy: 0.8665
11968/60000 [====>.........................] - ETA: 1:14 - loss: 0.4223 - categorical_accuracy: 0.8667
12000/60000 [=====>........................] - ETA: 1:14 - loss: 0.4217 - categorical_accuracy: 0.8669
12064/60000 [=====>........................] - ETA: 1:14 - loss: 0.4207 - categorical_accuracy: 0.8670
12128/60000 [=====>........................] - ETA: 1:14 - loss: 0.4191 - categorical_accuracy: 0.8676
12192/60000 [=====>........................] - ETA: 1:13 - loss: 0.4175 - categorical_accuracy: 0.8680
12256/60000 [=====>........................] - ETA: 1:13 - loss: 0.4165 - categorical_accuracy: 0.8684
12320/60000 [=====>........................] - ETA: 1:13 - loss: 0.4159 - categorical_accuracy: 0.8687
12384/60000 [=====>........................] - ETA: 1:13 - loss: 0.4146 - categorical_accuracy: 0.8690
12448/60000 [=====>........................] - ETA: 1:13 - loss: 0.4128 - categorical_accuracy: 0.8696
12512/60000 [=====>........................] - ETA: 1:13 - loss: 0.4118 - categorical_accuracy: 0.8700
12576/60000 [=====>........................] - ETA: 1:13 - loss: 0.4105 - categorical_accuracy: 0.8704
12640/60000 [=====>........................] - ETA: 1:13 - loss: 0.4091 - categorical_accuracy: 0.8708
12704/60000 [=====>........................] - ETA: 1:13 - loss: 0.4083 - categorical_accuracy: 0.8712
12768/60000 [=====>........................] - ETA: 1:12 - loss: 0.4070 - categorical_accuracy: 0.8716
12800/60000 [=====>........................] - ETA: 1:12 - loss: 0.4067 - categorical_accuracy: 0.8718
12832/60000 [=====>........................] - ETA: 1:12 - loss: 0.4064 - categorical_accuracy: 0.8720
12896/60000 [=====>........................] - ETA: 1:12 - loss: 0.4068 - categorical_accuracy: 0.8721
12960/60000 [=====>........................] - ETA: 1:12 - loss: 0.4060 - categorical_accuracy: 0.8724
12992/60000 [=====>........................] - ETA: 1:12 - loss: 0.4061 - categorical_accuracy: 0.8722
13056/60000 [=====>........................] - ETA: 1:12 - loss: 0.4048 - categorical_accuracy: 0.8726
13088/60000 [=====>........................] - ETA: 1:12 - loss: 0.4041 - categorical_accuracy: 0.8729
13120/60000 [=====>........................] - ETA: 1:12 - loss: 0.4036 - categorical_accuracy: 0.8731
13152/60000 [=====>........................] - ETA: 1:12 - loss: 0.4032 - categorical_accuracy: 0.8733
13216/60000 [=====>........................] - ETA: 1:12 - loss: 0.4031 - categorical_accuracy: 0.8733
13248/60000 [=====>........................] - ETA: 1:12 - loss: 0.4024 - categorical_accuracy: 0.8734
13280/60000 [=====>........................] - ETA: 1:12 - loss: 0.4017 - categorical_accuracy: 0.8736
13312/60000 [=====>........................] - ETA: 1:12 - loss: 0.4011 - categorical_accuracy: 0.8738
13376/60000 [=====>........................] - ETA: 1:12 - loss: 0.3996 - categorical_accuracy: 0.8743
13440/60000 [=====>........................] - ETA: 1:12 - loss: 0.3980 - categorical_accuracy: 0.8747
13504/60000 [=====>........................] - ETA: 1:11 - loss: 0.3968 - categorical_accuracy: 0.8751
13568/60000 [=====>........................] - ETA: 1:11 - loss: 0.3966 - categorical_accuracy: 0.8751
13632/60000 [=====>........................] - ETA: 1:11 - loss: 0.3954 - categorical_accuracy: 0.8754
13696/60000 [=====>........................] - ETA: 1:11 - loss: 0.3951 - categorical_accuracy: 0.8756
13728/60000 [=====>........................] - ETA: 1:11 - loss: 0.3945 - categorical_accuracy: 0.8758
13792/60000 [=====>........................] - ETA: 1:11 - loss: 0.3938 - categorical_accuracy: 0.8761
13856/60000 [=====>........................] - ETA: 1:11 - loss: 0.3931 - categorical_accuracy: 0.8764
13920/60000 [=====>........................] - ETA: 1:11 - loss: 0.3928 - categorical_accuracy: 0.8764
13984/60000 [=====>........................] - ETA: 1:11 - loss: 0.3916 - categorical_accuracy: 0.8767
14016/60000 [======>.......................] - ETA: 1:11 - loss: 0.3909 - categorical_accuracy: 0.8770
14048/60000 [======>.......................] - ETA: 1:11 - loss: 0.3905 - categorical_accuracy: 0.8771
14080/60000 [======>.......................] - ETA: 1:10 - loss: 0.3904 - categorical_accuracy: 0.8771
14112/60000 [======>.......................] - ETA: 1:10 - loss: 0.3898 - categorical_accuracy: 0.8773
14176/60000 [======>.......................] - ETA: 1:10 - loss: 0.3887 - categorical_accuracy: 0.8777
14240/60000 [======>.......................] - ETA: 1:10 - loss: 0.3879 - categorical_accuracy: 0.8780
14304/60000 [======>.......................] - ETA: 1:10 - loss: 0.3869 - categorical_accuracy: 0.8783
14368/60000 [======>.......................] - ETA: 1:10 - loss: 0.3866 - categorical_accuracy: 0.8784
14432/60000 [======>.......................] - ETA: 1:10 - loss: 0.3856 - categorical_accuracy: 0.8787
14496/60000 [======>.......................] - ETA: 1:10 - loss: 0.3846 - categorical_accuracy: 0.8791
14560/60000 [======>.......................] - ETA: 1:10 - loss: 0.3834 - categorical_accuracy: 0.8795
14624/60000 [======>.......................] - ETA: 1:10 - loss: 0.3823 - categorical_accuracy: 0.8798
14656/60000 [======>.......................] - ETA: 1:10 - loss: 0.3819 - categorical_accuracy: 0.8798
14688/60000 [======>.......................] - ETA: 1:10 - loss: 0.3813 - categorical_accuracy: 0.8800
14720/60000 [======>.......................] - ETA: 1:09 - loss: 0.3811 - categorical_accuracy: 0.8802
14752/60000 [======>.......................] - ETA: 1:09 - loss: 0.3806 - categorical_accuracy: 0.8803
14784/60000 [======>.......................] - ETA: 1:09 - loss: 0.3799 - categorical_accuracy: 0.8805
14816/60000 [======>.......................] - ETA: 1:09 - loss: 0.3792 - categorical_accuracy: 0.8807
14880/60000 [======>.......................] - ETA: 1:09 - loss: 0.3790 - categorical_accuracy: 0.8810
14912/60000 [======>.......................] - ETA: 1:09 - loss: 0.3785 - categorical_accuracy: 0.8810
14944/60000 [======>.......................] - ETA: 1:09 - loss: 0.3780 - categorical_accuracy: 0.8812
15008/60000 [======>.......................] - ETA: 1:09 - loss: 0.3772 - categorical_accuracy: 0.8815
15072/60000 [======>.......................] - ETA: 1:09 - loss: 0.3768 - categorical_accuracy: 0.8818
15104/60000 [======>.......................] - ETA: 1:09 - loss: 0.3767 - categorical_accuracy: 0.8818
15168/60000 [======>.......................] - ETA: 1:09 - loss: 0.3758 - categorical_accuracy: 0.8821
15232/60000 [======>.......................] - ETA: 1:09 - loss: 0.3750 - categorical_accuracy: 0.8823
15296/60000 [======>.......................] - ETA: 1:09 - loss: 0.3737 - categorical_accuracy: 0.8827
15328/60000 [======>.......................] - ETA: 1:09 - loss: 0.3733 - categorical_accuracy: 0.8828
15392/60000 [======>.......................] - ETA: 1:09 - loss: 0.3724 - categorical_accuracy: 0.8831
15456/60000 [======>.......................] - ETA: 1:08 - loss: 0.3717 - categorical_accuracy: 0.8832
15520/60000 [======>.......................] - ETA: 1:08 - loss: 0.3712 - categorical_accuracy: 0.8834
15584/60000 [======>.......................] - ETA: 1:08 - loss: 0.3706 - categorical_accuracy: 0.8836
15648/60000 [======>.......................] - ETA: 1:08 - loss: 0.3694 - categorical_accuracy: 0.8840
15712/60000 [======>.......................] - ETA: 1:08 - loss: 0.3692 - categorical_accuracy: 0.8841
15744/60000 [======>.......................] - ETA: 1:08 - loss: 0.3687 - categorical_accuracy: 0.8842
15776/60000 [======>.......................] - ETA: 1:08 - loss: 0.3682 - categorical_accuracy: 0.8844
15840/60000 [======>.......................] - ETA: 1:08 - loss: 0.3677 - categorical_accuracy: 0.8845
15904/60000 [======>.......................] - ETA: 1:08 - loss: 0.3666 - categorical_accuracy: 0.8849
15968/60000 [======>.......................] - ETA: 1:08 - loss: 0.3658 - categorical_accuracy: 0.8851
16032/60000 [=======>......................] - ETA: 1:08 - loss: 0.3657 - categorical_accuracy: 0.8852
16096/60000 [=======>......................] - ETA: 1:07 - loss: 0.3646 - categorical_accuracy: 0.8855
16160/60000 [=======>......................] - ETA: 1:07 - loss: 0.3645 - categorical_accuracy: 0.8856
16224/60000 [=======>......................] - ETA: 1:07 - loss: 0.3637 - categorical_accuracy: 0.8860
16288/60000 [=======>......................] - ETA: 1:07 - loss: 0.3626 - categorical_accuracy: 0.8863
16352/60000 [=======>......................] - ETA: 1:07 - loss: 0.3617 - categorical_accuracy: 0.8866
16384/60000 [=======>......................] - ETA: 1:07 - loss: 0.3612 - categorical_accuracy: 0.8868
16448/60000 [=======>......................] - ETA: 1:07 - loss: 0.3606 - categorical_accuracy: 0.8869
16512/60000 [=======>......................] - ETA: 1:07 - loss: 0.3593 - categorical_accuracy: 0.8874
16576/60000 [=======>......................] - ETA: 1:07 - loss: 0.3583 - categorical_accuracy: 0.8876
16608/60000 [=======>......................] - ETA: 1:07 - loss: 0.3579 - categorical_accuracy: 0.8878
16640/60000 [=======>......................] - ETA: 1:07 - loss: 0.3573 - categorical_accuracy: 0.8880
16704/60000 [=======>......................] - ETA: 1:06 - loss: 0.3573 - categorical_accuracy: 0.8881
16768/60000 [=======>......................] - ETA: 1:06 - loss: 0.3565 - categorical_accuracy: 0.8884
16832/60000 [=======>......................] - ETA: 1:06 - loss: 0.3553 - categorical_accuracy: 0.8888
16864/60000 [=======>......................] - ETA: 1:06 - loss: 0.3548 - categorical_accuracy: 0.8890
16928/60000 [=======>......................] - ETA: 1:06 - loss: 0.3550 - categorical_accuracy: 0.8890
16992/60000 [=======>......................] - ETA: 1:06 - loss: 0.3541 - categorical_accuracy: 0.8893
17056/60000 [=======>......................] - ETA: 1:06 - loss: 0.3534 - categorical_accuracy: 0.8895
17088/60000 [=======>......................] - ETA: 1:06 - loss: 0.3530 - categorical_accuracy: 0.8897
17120/60000 [=======>......................] - ETA: 1:06 - loss: 0.3526 - categorical_accuracy: 0.8898
17152/60000 [=======>......................] - ETA: 1:06 - loss: 0.3520 - categorical_accuracy: 0.8900
17216/60000 [=======>......................] - ETA: 1:06 - loss: 0.3514 - categorical_accuracy: 0.8903
17280/60000 [=======>......................] - ETA: 1:05 - loss: 0.3503 - categorical_accuracy: 0.8907
17344/60000 [=======>......................] - ETA: 1:05 - loss: 0.3500 - categorical_accuracy: 0.8909
17408/60000 [=======>......................] - ETA: 1:05 - loss: 0.3494 - categorical_accuracy: 0.8911
17440/60000 [=======>......................] - ETA: 1:05 - loss: 0.3490 - categorical_accuracy: 0.8912
17472/60000 [=======>......................] - ETA: 1:05 - loss: 0.3490 - categorical_accuracy: 0.8913
17536/60000 [=======>......................] - ETA: 1:05 - loss: 0.3485 - categorical_accuracy: 0.8915
17568/60000 [=======>......................] - ETA: 1:05 - loss: 0.3480 - categorical_accuracy: 0.8917
17600/60000 [=======>......................] - ETA: 1:05 - loss: 0.3474 - categorical_accuracy: 0.8919
17664/60000 [=======>......................] - ETA: 1:05 - loss: 0.3469 - categorical_accuracy: 0.8921
17728/60000 [=======>......................] - ETA: 1:05 - loss: 0.3458 - categorical_accuracy: 0.8924
17792/60000 [=======>......................] - ETA: 1:05 - loss: 0.3449 - categorical_accuracy: 0.8927
17824/60000 [=======>......................] - ETA: 1:05 - loss: 0.3445 - categorical_accuracy: 0.8928
17856/60000 [=======>......................] - ETA: 1:05 - loss: 0.3442 - categorical_accuracy: 0.8929
17888/60000 [=======>......................] - ETA: 1:05 - loss: 0.3441 - categorical_accuracy: 0.8929
17920/60000 [=======>......................] - ETA: 1:05 - loss: 0.3438 - categorical_accuracy: 0.8930
17984/60000 [=======>......................] - ETA: 1:04 - loss: 0.3429 - categorical_accuracy: 0.8933
18016/60000 [========>.....................] - ETA: 1:04 - loss: 0.3424 - categorical_accuracy: 0.8935
18048/60000 [========>.....................] - ETA: 1:04 - loss: 0.3420 - categorical_accuracy: 0.8937
18112/60000 [========>.....................] - ETA: 1:04 - loss: 0.3412 - categorical_accuracy: 0.8939
18176/60000 [========>.....................] - ETA: 1:04 - loss: 0.3404 - categorical_accuracy: 0.8941
18240/60000 [========>.....................] - ETA: 1:04 - loss: 0.3399 - categorical_accuracy: 0.8942
18304/60000 [========>.....................] - ETA: 1:04 - loss: 0.3392 - categorical_accuracy: 0.8944
18368/60000 [========>.....................] - ETA: 1:04 - loss: 0.3388 - categorical_accuracy: 0.8945
18432/60000 [========>.....................] - ETA: 1:04 - loss: 0.3380 - categorical_accuracy: 0.8948
18496/60000 [========>.....................] - ETA: 1:04 - loss: 0.3372 - categorical_accuracy: 0.8950
18560/60000 [========>.....................] - ETA: 1:04 - loss: 0.3364 - categorical_accuracy: 0.8952
18592/60000 [========>.....................] - ETA: 1:03 - loss: 0.3361 - categorical_accuracy: 0.8953
18624/60000 [========>.....................] - ETA: 1:03 - loss: 0.3359 - categorical_accuracy: 0.8954
18656/60000 [========>.....................] - ETA: 1:03 - loss: 0.3355 - categorical_accuracy: 0.8955
18688/60000 [========>.....................] - ETA: 1:03 - loss: 0.3351 - categorical_accuracy: 0.8956
18720/60000 [========>.....................] - ETA: 1:03 - loss: 0.3346 - categorical_accuracy: 0.8957
18752/60000 [========>.....................] - ETA: 1:03 - loss: 0.3341 - categorical_accuracy: 0.8959
18784/60000 [========>.....................] - ETA: 1:03 - loss: 0.3336 - categorical_accuracy: 0.8961
18816/60000 [========>.....................] - ETA: 1:03 - loss: 0.3331 - categorical_accuracy: 0.8963
18880/60000 [========>.....................] - ETA: 1:03 - loss: 0.3324 - categorical_accuracy: 0.8965
18944/60000 [========>.....................] - ETA: 1:03 - loss: 0.3314 - categorical_accuracy: 0.8969
18976/60000 [========>.....................] - ETA: 1:03 - loss: 0.3311 - categorical_accuracy: 0.8970
19040/60000 [========>.....................] - ETA: 1:03 - loss: 0.3301 - categorical_accuracy: 0.8973
19072/60000 [========>.....................] - ETA: 1:03 - loss: 0.3295 - categorical_accuracy: 0.8974
19136/60000 [========>.....................] - ETA: 1:03 - loss: 0.3286 - categorical_accuracy: 0.8977
19200/60000 [========>.....................] - ETA: 1:03 - loss: 0.3280 - categorical_accuracy: 0.8979
19232/60000 [========>.....................] - ETA: 1:03 - loss: 0.3275 - categorical_accuracy: 0.8980
19264/60000 [========>.....................] - ETA: 1:02 - loss: 0.3272 - categorical_accuracy: 0.8981
19296/60000 [========>.....................] - ETA: 1:02 - loss: 0.3269 - categorical_accuracy: 0.8982
19360/60000 [========>.....................] - ETA: 1:02 - loss: 0.3267 - categorical_accuracy: 0.8985
19392/60000 [========>.....................] - ETA: 1:02 - loss: 0.3263 - categorical_accuracy: 0.8987
19424/60000 [========>.....................] - ETA: 1:02 - loss: 0.3259 - categorical_accuracy: 0.8988
19456/60000 [========>.....................] - ETA: 1:02 - loss: 0.3254 - categorical_accuracy: 0.8990
19520/60000 [========>.....................] - ETA: 1:02 - loss: 0.3245 - categorical_accuracy: 0.8992
19584/60000 [========>.....................] - ETA: 1:02 - loss: 0.3237 - categorical_accuracy: 0.8995
19648/60000 [========>.....................] - ETA: 1:02 - loss: 0.3229 - categorical_accuracy: 0.8996
19712/60000 [========>.....................] - ETA: 1:02 - loss: 0.3220 - categorical_accuracy: 0.9000
19744/60000 [========>.....................] - ETA: 1:02 - loss: 0.3218 - categorical_accuracy: 0.9001
19808/60000 [========>.....................] - ETA: 1:02 - loss: 0.3213 - categorical_accuracy: 0.9002
19840/60000 [========>.....................] - ETA: 1:02 - loss: 0.3207 - categorical_accuracy: 0.9004
19872/60000 [========>.....................] - ETA: 1:02 - loss: 0.3208 - categorical_accuracy: 0.9003
19936/60000 [========>.....................] - ETA: 1:01 - loss: 0.3209 - categorical_accuracy: 0.9005
20000/60000 [=========>....................] - ETA: 1:01 - loss: 0.3200 - categorical_accuracy: 0.9008
20064/60000 [=========>....................] - ETA: 1:01 - loss: 0.3193 - categorical_accuracy: 0.9011
20096/60000 [=========>....................] - ETA: 1:01 - loss: 0.3189 - categorical_accuracy: 0.9012
20160/60000 [=========>....................] - ETA: 1:01 - loss: 0.3188 - categorical_accuracy: 0.9013
20224/60000 [=========>....................] - ETA: 1:01 - loss: 0.3179 - categorical_accuracy: 0.9017
20288/60000 [=========>....................] - ETA: 1:01 - loss: 0.3172 - categorical_accuracy: 0.9019
20352/60000 [=========>....................] - ETA: 1:01 - loss: 0.3167 - categorical_accuracy: 0.9020
20416/60000 [=========>....................] - ETA: 1:01 - loss: 0.3161 - categorical_accuracy: 0.9023
20480/60000 [=========>....................] - ETA: 1:01 - loss: 0.3156 - categorical_accuracy: 0.9025
20512/60000 [=========>....................] - ETA: 1:01 - loss: 0.3153 - categorical_accuracy: 0.9025
20576/60000 [=========>....................] - ETA: 1:00 - loss: 0.3145 - categorical_accuracy: 0.9028
20640/60000 [=========>....................] - ETA: 1:00 - loss: 0.3140 - categorical_accuracy: 0.9030
20704/60000 [=========>....................] - ETA: 1:00 - loss: 0.3134 - categorical_accuracy: 0.9030
20768/60000 [=========>....................] - ETA: 1:00 - loss: 0.3132 - categorical_accuracy: 0.9032
20832/60000 [=========>....................] - ETA: 1:00 - loss: 0.3128 - categorical_accuracy: 0.9033
20896/60000 [=========>....................] - ETA: 1:00 - loss: 0.3124 - categorical_accuracy: 0.9035
20960/60000 [=========>....................] - ETA: 1:00 - loss: 0.3119 - categorical_accuracy: 0.9037
20992/60000 [=========>....................] - ETA: 1:00 - loss: 0.3115 - categorical_accuracy: 0.9038
21056/60000 [=========>....................] - ETA: 1:00 - loss: 0.3112 - categorical_accuracy: 0.9040
21088/60000 [=========>....................] - ETA: 1:00 - loss: 0.3108 - categorical_accuracy: 0.9041
21120/60000 [=========>....................] - ETA: 1:00 - loss: 0.3105 - categorical_accuracy: 0.9042
21152/60000 [=========>....................] - ETA: 1:00 - loss: 0.3101 - categorical_accuracy: 0.9043
21216/60000 [=========>....................] - ETA: 59s - loss: 0.3097 - categorical_accuracy: 0.9044 
21280/60000 [=========>....................] - ETA: 59s - loss: 0.3097 - categorical_accuracy: 0.9045
21344/60000 [=========>....................] - ETA: 59s - loss: 0.3091 - categorical_accuracy: 0.9048
21408/60000 [=========>....................] - ETA: 59s - loss: 0.3084 - categorical_accuracy: 0.9050
21440/60000 [=========>....................] - ETA: 59s - loss: 0.3082 - categorical_accuracy: 0.9051
21472/60000 [=========>....................] - ETA: 59s - loss: 0.3079 - categorical_accuracy: 0.9051
21504/60000 [=========>....................] - ETA: 59s - loss: 0.3075 - categorical_accuracy: 0.9053
21536/60000 [=========>....................] - ETA: 59s - loss: 0.3072 - categorical_accuracy: 0.9053
21600/60000 [=========>....................] - ETA: 59s - loss: 0.3070 - categorical_accuracy: 0.9054
21632/60000 [=========>....................] - ETA: 59s - loss: 0.3068 - categorical_accuracy: 0.9055
21696/60000 [=========>....................] - ETA: 59s - loss: 0.3061 - categorical_accuracy: 0.9057
21760/60000 [=========>....................] - ETA: 59s - loss: 0.3053 - categorical_accuracy: 0.9059
21824/60000 [=========>....................] - ETA: 58s - loss: 0.3049 - categorical_accuracy: 0.9060
21888/60000 [=========>....................] - ETA: 58s - loss: 0.3041 - categorical_accuracy: 0.9063
21952/60000 [=========>....................] - ETA: 58s - loss: 0.3035 - categorical_accuracy: 0.9064
22016/60000 [==========>...................] - ETA: 58s - loss: 0.3030 - categorical_accuracy: 0.9066
22048/60000 [==========>...................] - ETA: 58s - loss: 0.3027 - categorical_accuracy: 0.9067
22080/60000 [==========>...................] - ETA: 58s - loss: 0.3023 - categorical_accuracy: 0.9068
22112/60000 [==========>...................] - ETA: 58s - loss: 0.3030 - categorical_accuracy: 0.9067
22176/60000 [==========>...................] - ETA: 58s - loss: 0.3030 - categorical_accuracy: 0.9067
22240/60000 [==========>...................] - ETA: 58s - loss: 0.3026 - categorical_accuracy: 0.9068
22304/60000 [==========>...................] - ETA: 58s - loss: 0.3022 - categorical_accuracy: 0.9069
22336/60000 [==========>...................] - ETA: 58s - loss: 0.3019 - categorical_accuracy: 0.9070
22400/60000 [==========>...................] - ETA: 58s - loss: 0.3013 - categorical_accuracy: 0.9072
22464/60000 [==========>...................] - ETA: 58s - loss: 0.3006 - categorical_accuracy: 0.9074
22496/60000 [==========>...................] - ETA: 57s - loss: 0.3008 - categorical_accuracy: 0.9073
22528/60000 [==========>...................] - ETA: 57s - loss: 0.3004 - categorical_accuracy: 0.9074
22592/60000 [==========>...................] - ETA: 57s - loss: 0.2999 - categorical_accuracy: 0.9076
22656/60000 [==========>...................] - ETA: 57s - loss: 0.2994 - categorical_accuracy: 0.9078
22720/60000 [==========>...................] - ETA: 57s - loss: 0.2992 - categorical_accuracy: 0.9077
22784/60000 [==========>...................] - ETA: 57s - loss: 0.2986 - categorical_accuracy: 0.9079
22848/60000 [==========>...................] - ETA: 57s - loss: 0.2989 - categorical_accuracy: 0.9078
22912/60000 [==========>...................] - ETA: 57s - loss: 0.2985 - categorical_accuracy: 0.9079
22976/60000 [==========>...................] - ETA: 57s - loss: 0.2982 - categorical_accuracy: 0.9079
23040/60000 [==========>...................] - ETA: 57s - loss: 0.2980 - categorical_accuracy: 0.9080
23104/60000 [==========>...................] - ETA: 56s - loss: 0.2974 - categorical_accuracy: 0.9082
23168/60000 [==========>...................] - ETA: 56s - loss: 0.2969 - categorical_accuracy: 0.9083
23200/60000 [==========>...................] - ETA: 56s - loss: 0.2965 - categorical_accuracy: 0.9084
23232/60000 [==========>...................] - ETA: 56s - loss: 0.2962 - categorical_accuracy: 0.9085
23296/60000 [==========>...................] - ETA: 56s - loss: 0.2955 - categorical_accuracy: 0.9088
23360/60000 [==========>...................] - ETA: 56s - loss: 0.2951 - categorical_accuracy: 0.9088
23424/60000 [==========>...................] - ETA: 56s - loss: 0.2947 - categorical_accuracy: 0.9089
23488/60000 [==========>...................] - ETA: 56s - loss: 0.2947 - categorical_accuracy: 0.9090
23520/60000 [==========>...................] - ETA: 56s - loss: 0.2944 - categorical_accuracy: 0.9091
23584/60000 [==========>...................] - ETA: 56s - loss: 0.2938 - categorical_accuracy: 0.9093
23616/60000 [==========>...................] - ETA: 56s - loss: 0.2940 - categorical_accuracy: 0.9092
23680/60000 [==========>...................] - ETA: 56s - loss: 0.2934 - categorical_accuracy: 0.9094
23744/60000 [==========>...................] - ETA: 55s - loss: 0.2935 - categorical_accuracy: 0.9093
23808/60000 [==========>...................] - ETA: 55s - loss: 0.2930 - categorical_accuracy: 0.9094
23872/60000 [==========>...................] - ETA: 55s - loss: 0.2928 - categorical_accuracy: 0.9094
23904/60000 [==========>...................] - ETA: 55s - loss: 0.2926 - categorical_accuracy: 0.9095
23936/60000 [==========>...................] - ETA: 55s - loss: 0.2922 - categorical_accuracy: 0.9096
24000/60000 [===========>..................] - ETA: 55s - loss: 0.2919 - categorical_accuracy: 0.9097
24064/60000 [===========>..................] - ETA: 55s - loss: 0.2913 - categorical_accuracy: 0.9099
24128/60000 [===========>..................] - ETA: 55s - loss: 0.2909 - categorical_accuracy: 0.9100
24192/60000 [===========>..................] - ETA: 55s - loss: 0.2905 - categorical_accuracy: 0.9102
24256/60000 [===========>..................] - ETA: 55s - loss: 0.2902 - categorical_accuracy: 0.9102
24320/60000 [===========>..................] - ETA: 55s - loss: 0.2898 - categorical_accuracy: 0.9104
24384/60000 [===========>..................] - ETA: 54s - loss: 0.2895 - categorical_accuracy: 0.9105
24448/60000 [===========>..................] - ETA: 54s - loss: 0.2893 - categorical_accuracy: 0.9105
24512/60000 [===========>..................] - ETA: 54s - loss: 0.2890 - categorical_accuracy: 0.9107
24576/60000 [===========>..................] - ETA: 54s - loss: 0.2889 - categorical_accuracy: 0.9107
24640/60000 [===========>..................] - ETA: 54s - loss: 0.2886 - categorical_accuracy: 0.9108
24704/60000 [===========>..................] - ETA: 54s - loss: 0.2883 - categorical_accuracy: 0.9108
24736/60000 [===========>..................] - ETA: 54s - loss: 0.2881 - categorical_accuracy: 0.9109
24768/60000 [===========>..................] - ETA: 54s - loss: 0.2882 - categorical_accuracy: 0.9110
24832/60000 [===========>..................] - ETA: 54s - loss: 0.2877 - categorical_accuracy: 0.9111
24896/60000 [===========>..................] - ETA: 54s - loss: 0.2872 - categorical_accuracy: 0.9113
24960/60000 [===========>..................] - ETA: 54s - loss: 0.2870 - categorical_accuracy: 0.9114
25024/60000 [===========>..................] - ETA: 53s - loss: 0.2867 - categorical_accuracy: 0.9114
25088/60000 [===========>..................] - ETA: 53s - loss: 0.2865 - categorical_accuracy: 0.9115
25120/60000 [===========>..................] - ETA: 53s - loss: 0.2862 - categorical_accuracy: 0.9115
25184/60000 [===========>..................] - ETA: 53s - loss: 0.2857 - categorical_accuracy: 0.9117
25248/60000 [===========>..................] - ETA: 53s - loss: 0.2852 - categorical_accuracy: 0.9118
25312/60000 [===========>..................] - ETA: 53s - loss: 0.2845 - categorical_accuracy: 0.9121
25376/60000 [===========>..................] - ETA: 53s - loss: 0.2842 - categorical_accuracy: 0.9121
25440/60000 [===========>..................] - ETA: 53s - loss: 0.2837 - categorical_accuracy: 0.9123
25472/60000 [===========>..................] - ETA: 53s - loss: 0.2834 - categorical_accuracy: 0.9124
25504/60000 [===========>..................] - ETA: 53s - loss: 0.2832 - categorical_accuracy: 0.9125
25568/60000 [===========>..................] - ETA: 53s - loss: 0.2828 - categorical_accuracy: 0.9125
25632/60000 [===========>..................] - ETA: 52s - loss: 0.2825 - categorical_accuracy: 0.9126
25696/60000 [===========>..................] - ETA: 52s - loss: 0.2827 - categorical_accuracy: 0.9126
25760/60000 [===========>..................] - ETA: 52s - loss: 0.2822 - categorical_accuracy: 0.9128
25824/60000 [===========>..................] - ETA: 52s - loss: 0.2820 - categorical_accuracy: 0.9128
25856/60000 [===========>..................] - ETA: 52s - loss: 0.2818 - categorical_accuracy: 0.9129
25888/60000 [===========>..................] - ETA: 52s - loss: 0.2815 - categorical_accuracy: 0.9130
25952/60000 [===========>..................] - ETA: 52s - loss: 0.2812 - categorical_accuracy: 0.9131
26016/60000 [============>.................] - ETA: 52s - loss: 0.2807 - categorical_accuracy: 0.9132
26080/60000 [============>.................] - ETA: 52s - loss: 0.2801 - categorical_accuracy: 0.9133
26112/60000 [============>.................] - ETA: 52s - loss: 0.2799 - categorical_accuracy: 0.9134
26176/60000 [============>.................] - ETA: 52s - loss: 0.2793 - categorical_accuracy: 0.9136
26240/60000 [============>.................] - ETA: 52s - loss: 0.2789 - categorical_accuracy: 0.9137
26304/60000 [============>.................] - ETA: 51s - loss: 0.2784 - categorical_accuracy: 0.9139
26368/60000 [============>.................] - ETA: 51s - loss: 0.2786 - categorical_accuracy: 0.9139
26432/60000 [============>.................] - ETA: 51s - loss: 0.2781 - categorical_accuracy: 0.9140
26496/60000 [============>.................] - ETA: 51s - loss: 0.2785 - categorical_accuracy: 0.9140
26560/60000 [============>.................] - ETA: 51s - loss: 0.2784 - categorical_accuracy: 0.9140
26624/60000 [============>.................] - ETA: 51s - loss: 0.2780 - categorical_accuracy: 0.9141
26688/60000 [============>.................] - ETA: 51s - loss: 0.2775 - categorical_accuracy: 0.9142
26752/60000 [============>.................] - ETA: 51s - loss: 0.2772 - categorical_accuracy: 0.9143
26816/60000 [============>.................] - ETA: 51s - loss: 0.2768 - categorical_accuracy: 0.9144
26880/60000 [============>.................] - ETA: 50s - loss: 0.2763 - categorical_accuracy: 0.9145
26944/60000 [============>.................] - ETA: 50s - loss: 0.2760 - categorical_accuracy: 0.9146
27008/60000 [============>.................] - ETA: 50s - loss: 0.2759 - categorical_accuracy: 0.9147
27072/60000 [============>.................] - ETA: 50s - loss: 0.2760 - categorical_accuracy: 0.9147
27104/60000 [============>.................] - ETA: 50s - loss: 0.2760 - categorical_accuracy: 0.9147
27136/60000 [============>.................] - ETA: 50s - loss: 0.2760 - categorical_accuracy: 0.9148
27200/60000 [============>.................] - ETA: 50s - loss: 0.2757 - categorical_accuracy: 0.9149
27264/60000 [============>.................] - ETA: 50s - loss: 0.2754 - categorical_accuracy: 0.9150
27328/60000 [============>.................] - ETA: 50s - loss: 0.2749 - categorical_accuracy: 0.9151
27392/60000 [============>.................] - ETA: 50s - loss: 0.2749 - categorical_accuracy: 0.9152
27424/60000 [============>.................] - ETA: 50s - loss: 0.2747 - categorical_accuracy: 0.9152
27456/60000 [============>.................] - ETA: 50s - loss: 0.2747 - categorical_accuracy: 0.9152
27520/60000 [============>.................] - ETA: 50s - loss: 0.2745 - categorical_accuracy: 0.9154
27552/60000 [============>.................] - ETA: 49s - loss: 0.2743 - categorical_accuracy: 0.9154
27616/60000 [============>.................] - ETA: 49s - loss: 0.2740 - categorical_accuracy: 0.9155
27680/60000 [============>.................] - ETA: 49s - loss: 0.2740 - categorical_accuracy: 0.9155
27712/60000 [============>.................] - ETA: 49s - loss: 0.2737 - categorical_accuracy: 0.9156
27744/60000 [============>.................] - ETA: 49s - loss: 0.2736 - categorical_accuracy: 0.9156
27808/60000 [============>.................] - ETA: 49s - loss: 0.2734 - categorical_accuracy: 0.9157
27872/60000 [============>.................] - ETA: 49s - loss: 0.2729 - categorical_accuracy: 0.9159
27904/60000 [============>.................] - ETA: 49s - loss: 0.2726 - categorical_accuracy: 0.9160
27936/60000 [============>.................] - ETA: 49s - loss: 0.2725 - categorical_accuracy: 0.9160
27968/60000 [============>.................] - ETA: 49s - loss: 0.2724 - categorical_accuracy: 0.9160
28000/60000 [=============>................] - ETA: 49s - loss: 0.2722 - categorical_accuracy: 0.9161
28032/60000 [=============>................] - ETA: 49s - loss: 0.2719 - categorical_accuracy: 0.9162
28096/60000 [=============>................] - ETA: 49s - loss: 0.2714 - categorical_accuracy: 0.9164
28160/60000 [=============>................] - ETA: 49s - loss: 0.2709 - categorical_accuracy: 0.9165
28224/60000 [=============>................] - ETA: 48s - loss: 0.2705 - categorical_accuracy: 0.9166
28256/60000 [=============>................] - ETA: 48s - loss: 0.2702 - categorical_accuracy: 0.9167
28288/60000 [=============>................] - ETA: 48s - loss: 0.2700 - categorical_accuracy: 0.9168
28320/60000 [=============>................] - ETA: 48s - loss: 0.2702 - categorical_accuracy: 0.9167
28352/60000 [=============>................] - ETA: 48s - loss: 0.2700 - categorical_accuracy: 0.9168
28384/60000 [=============>................] - ETA: 48s - loss: 0.2699 - categorical_accuracy: 0.9168
28416/60000 [=============>................] - ETA: 48s - loss: 0.2697 - categorical_accuracy: 0.9169
28448/60000 [=============>................] - ETA: 48s - loss: 0.2696 - categorical_accuracy: 0.9169
28480/60000 [=============>................] - ETA: 48s - loss: 0.2696 - categorical_accuracy: 0.9168
28512/60000 [=============>................] - ETA: 48s - loss: 0.2695 - categorical_accuracy: 0.9168
28576/60000 [=============>................] - ETA: 48s - loss: 0.2693 - categorical_accuracy: 0.9169
28640/60000 [=============>................] - ETA: 48s - loss: 0.2689 - categorical_accuracy: 0.9170
28672/60000 [=============>................] - ETA: 48s - loss: 0.2686 - categorical_accuracy: 0.9171
28704/60000 [=============>................] - ETA: 48s - loss: 0.2685 - categorical_accuracy: 0.9171
28736/60000 [=============>................] - ETA: 48s - loss: 0.2684 - categorical_accuracy: 0.9171
28800/60000 [=============>................] - ETA: 48s - loss: 0.2679 - categorical_accuracy: 0.9173
28832/60000 [=============>................] - ETA: 48s - loss: 0.2678 - categorical_accuracy: 0.9173
28896/60000 [=============>................] - ETA: 48s - loss: 0.2675 - categorical_accuracy: 0.9174
28960/60000 [=============>................] - ETA: 47s - loss: 0.2672 - categorical_accuracy: 0.9175
29024/60000 [=============>................] - ETA: 47s - loss: 0.2669 - categorical_accuracy: 0.9176
29056/60000 [=============>................] - ETA: 47s - loss: 0.2667 - categorical_accuracy: 0.9176
29088/60000 [=============>................] - ETA: 47s - loss: 0.2665 - categorical_accuracy: 0.9177
29120/60000 [=============>................] - ETA: 47s - loss: 0.2662 - categorical_accuracy: 0.9178
29184/60000 [=============>................] - ETA: 47s - loss: 0.2657 - categorical_accuracy: 0.9179
29248/60000 [=============>................] - ETA: 47s - loss: 0.2652 - categorical_accuracy: 0.9181
29312/60000 [=============>................] - ETA: 47s - loss: 0.2649 - categorical_accuracy: 0.9182
29376/60000 [=============>................] - ETA: 47s - loss: 0.2645 - categorical_accuracy: 0.9183
29440/60000 [=============>................] - ETA: 47s - loss: 0.2643 - categorical_accuracy: 0.9184
29504/60000 [=============>................] - ETA: 47s - loss: 0.2639 - categorical_accuracy: 0.9185
29568/60000 [=============>................] - ETA: 46s - loss: 0.2635 - categorical_accuracy: 0.9186
29632/60000 [=============>................] - ETA: 46s - loss: 0.2630 - categorical_accuracy: 0.9188
29664/60000 [=============>................] - ETA: 46s - loss: 0.2628 - categorical_accuracy: 0.9189
29728/60000 [=============>................] - ETA: 46s - loss: 0.2627 - categorical_accuracy: 0.9189
29792/60000 [=============>................] - ETA: 46s - loss: 0.2624 - categorical_accuracy: 0.9190
29856/60000 [=============>................] - ETA: 46s - loss: 0.2621 - categorical_accuracy: 0.9191
29920/60000 [=============>................] - ETA: 46s - loss: 0.2616 - categorical_accuracy: 0.9193
29984/60000 [=============>................] - ETA: 46s - loss: 0.2614 - categorical_accuracy: 0.9194
30048/60000 [==============>...............] - ETA: 46s - loss: 0.2613 - categorical_accuracy: 0.9194
30112/60000 [==============>...............] - ETA: 46s - loss: 0.2609 - categorical_accuracy: 0.9196
30176/60000 [==============>...............] - ETA: 46s - loss: 0.2605 - categorical_accuracy: 0.9196
30208/60000 [==============>...............] - ETA: 45s - loss: 0.2603 - categorical_accuracy: 0.9197
30240/60000 [==============>...............] - ETA: 45s - loss: 0.2602 - categorical_accuracy: 0.9197
30304/60000 [==============>...............] - ETA: 45s - loss: 0.2598 - categorical_accuracy: 0.9198
30368/60000 [==============>...............] - ETA: 45s - loss: 0.2597 - categorical_accuracy: 0.9198
30432/60000 [==============>...............] - ETA: 45s - loss: 0.2594 - categorical_accuracy: 0.9200
30496/60000 [==============>...............] - ETA: 45s - loss: 0.2591 - categorical_accuracy: 0.9201
30528/60000 [==============>...............] - ETA: 45s - loss: 0.2589 - categorical_accuracy: 0.9201
30560/60000 [==============>...............] - ETA: 45s - loss: 0.2588 - categorical_accuracy: 0.9202
30624/60000 [==============>...............] - ETA: 45s - loss: 0.2586 - categorical_accuracy: 0.9202
30656/60000 [==============>...............] - ETA: 45s - loss: 0.2584 - categorical_accuracy: 0.9202
30720/60000 [==============>...............] - ETA: 45s - loss: 0.2581 - categorical_accuracy: 0.9203
30784/60000 [==============>...............] - ETA: 45s - loss: 0.2576 - categorical_accuracy: 0.9204
30848/60000 [==============>...............] - ETA: 45s - loss: 0.2574 - categorical_accuracy: 0.9205
30912/60000 [==============>...............] - ETA: 44s - loss: 0.2571 - categorical_accuracy: 0.9206
30976/60000 [==============>...............] - ETA: 44s - loss: 0.2568 - categorical_accuracy: 0.9207
31008/60000 [==============>...............] - ETA: 44s - loss: 0.2566 - categorical_accuracy: 0.9207
31040/60000 [==============>...............] - ETA: 44s - loss: 0.2564 - categorical_accuracy: 0.9208
31104/60000 [==============>...............] - ETA: 44s - loss: 0.2563 - categorical_accuracy: 0.9209
31168/60000 [==============>...............] - ETA: 44s - loss: 0.2560 - categorical_accuracy: 0.9210
31200/60000 [==============>...............] - ETA: 44s - loss: 0.2558 - categorical_accuracy: 0.9211
31232/60000 [==============>...............] - ETA: 44s - loss: 0.2556 - categorical_accuracy: 0.9211
31296/60000 [==============>...............] - ETA: 44s - loss: 0.2554 - categorical_accuracy: 0.9212
31328/60000 [==============>...............] - ETA: 44s - loss: 0.2552 - categorical_accuracy: 0.9212
31392/60000 [==============>...............] - ETA: 44s - loss: 0.2550 - categorical_accuracy: 0.9213
31456/60000 [==============>...............] - ETA: 44s - loss: 0.2547 - categorical_accuracy: 0.9213
31520/60000 [==============>...............] - ETA: 43s - loss: 0.2545 - categorical_accuracy: 0.9214
31552/60000 [==============>...............] - ETA: 43s - loss: 0.2543 - categorical_accuracy: 0.9215
31616/60000 [==============>...............] - ETA: 43s - loss: 0.2540 - categorical_accuracy: 0.9216
31648/60000 [==============>...............] - ETA: 43s - loss: 0.2539 - categorical_accuracy: 0.9216
31680/60000 [==============>...............] - ETA: 43s - loss: 0.2542 - categorical_accuracy: 0.9216
31712/60000 [==============>...............] - ETA: 43s - loss: 0.2541 - categorical_accuracy: 0.9216
31776/60000 [==============>...............] - ETA: 43s - loss: 0.2539 - categorical_accuracy: 0.9217
31840/60000 [==============>...............] - ETA: 43s - loss: 0.2535 - categorical_accuracy: 0.9218
31904/60000 [==============>...............] - ETA: 43s - loss: 0.2532 - categorical_accuracy: 0.9219
31936/60000 [==============>...............] - ETA: 43s - loss: 0.2534 - categorical_accuracy: 0.9218
32000/60000 [===============>..............] - ETA: 43s - loss: 0.2529 - categorical_accuracy: 0.9220
32064/60000 [===============>..............] - ETA: 43s - loss: 0.2525 - categorical_accuracy: 0.9221
32128/60000 [===============>..............] - ETA: 43s - loss: 0.2526 - categorical_accuracy: 0.9222
32160/60000 [===============>..............] - ETA: 42s - loss: 0.2525 - categorical_accuracy: 0.9222
32192/60000 [===============>..............] - ETA: 42s - loss: 0.2523 - categorical_accuracy: 0.9223
32224/60000 [===============>..............] - ETA: 42s - loss: 0.2520 - categorical_accuracy: 0.9224
32256/60000 [===============>..............] - ETA: 42s - loss: 0.2520 - categorical_accuracy: 0.9224
32288/60000 [===============>..............] - ETA: 42s - loss: 0.2518 - categorical_accuracy: 0.9224
32352/60000 [===============>..............] - ETA: 42s - loss: 0.2514 - categorical_accuracy: 0.9225
32384/60000 [===============>..............] - ETA: 42s - loss: 0.2512 - categorical_accuracy: 0.9226
32416/60000 [===============>..............] - ETA: 42s - loss: 0.2510 - categorical_accuracy: 0.9227
32480/60000 [===============>..............] - ETA: 42s - loss: 0.2506 - categorical_accuracy: 0.9228
32512/60000 [===============>..............] - ETA: 42s - loss: 0.2504 - categorical_accuracy: 0.9229
32544/60000 [===============>..............] - ETA: 42s - loss: 0.2503 - categorical_accuracy: 0.9228
32576/60000 [===============>..............] - ETA: 42s - loss: 0.2502 - categorical_accuracy: 0.9229
32608/60000 [===============>..............] - ETA: 42s - loss: 0.2501 - categorical_accuracy: 0.9229
32640/60000 [===============>..............] - ETA: 42s - loss: 0.2499 - categorical_accuracy: 0.9229
32672/60000 [===============>..............] - ETA: 42s - loss: 0.2497 - categorical_accuracy: 0.9230
32736/60000 [===============>..............] - ETA: 42s - loss: 0.2495 - categorical_accuracy: 0.9231
32768/60000 [===============>..............] - ETA: 42s - loss: 0.2497 - categorical_accuracy: 0.9231
32800/60000 [===============>..............] - ETA: 42s - loss: 0.2495 - categorical_accuracy: 0.9232
32864/60000 [===============>..............] - ETA: 41s - loss: 0.2494 - categorical_accuracy: 0.9232
32928/60000 [===============>..............] - ETA: 41s - loss: 0.2492 - categorical_accuracy: 0.9233
32960/60000 [===============>..............] - ETA: 41s - loss: 0.2491 - categorical_accuracy: 0.9234
32992/60000 [===============>..............] - ETA: 41s - loss: 0.2490 - categorical_accuracy: 0.9234
33056/60000 [===============>..............] - ETA: 41s - loss: 0.2489 - categorical_accuracy: 0.9235
33120/60000 [===============>..............] - ETA: 41s - loss: 0.2486 - categorical_accuracy: 0.9236
33184/60000 [===============>..............] - ETA: 41s - loss: 0.2485 - categorical_accuracy: 0.9236
33248/60000 [===============>..............] - ETA: 41s - loss: 0.2481 - categorical_accuracy: 0.9237
33280/60000 [===============>..............] - ETA: 41s - loss: 0.2479 - categorical_accuracy: 0.9237
33344/60000 [===============>..............] - ETA: 41s - loss: 0.2477 - categorical_accuracy: 0.9238
33376/60000 [===============>..............] - ETA: 41s - loss: 0.2475 - categorical_accuracy: 0.9238
33408/60000 [===============>..............] - ETA: 41s - loss: 0.2474 - categorical_accuracy: 0.9239
33472/60000 [===============>..............] - ETA: 41s - loss: 0.2471 - categorical_accuracy: 0.9239
33504/60000 [===============>..............] - ETA: 40s - loss: 0.2471 - categorical_accuracy: 0.9240
33536/60000 [===============>..............] - ETA: 40s - loss: 0.2470 - categorical_accuracy: 0.9240
33600/60000 [===============>..............] - ETA: 40s - loss: 0.2470 - categorical_accuracy: 0.9240
33664/60000 [===============>..............] - ETA: 40s - loss: 0.2468 - categorical_accuracy: 0.9240
33728/60000 [===============>..............] - ETA: 40s - loss: 0.2466 - categorical_accuracy: 0.9241
33792/60000 [===============>..............] - ETA: 40s - loss: 0.2465 - categorical_accuracy: 0.9241
33856/60000 [===============>..............] - ETA: 40s - loss: 0.2462 - categorical_accuracy: 0.9242
33888/60000 [===============>..............] - ETA: 40s - loss: 0.2461 - categorical_accuracy: 0.9242
33920/60000 [===============>..............] - ETA: 40s - loss: 0.2459 - categorical_accuracy: 0.9242
33952/60000 [===============>..............] - ETA: 40s - loss: 0.2457 - categorical_accuracy: 0.9243
34016/60000 [================>.............] - ETA: 40s - loss: 0.2455 - categorical_accuracy: 0.9244
34048/60000 [================>.............] - ETA: 40s - loss: 0.2454 - categorical_accuracy: 0.9244
34112/60000 [================>.............] - ETA: 40s - loss: 0.2451 - categorical_accuracy: 0.9245
34176/60000 [================>.............] - ETA: 39s - loss: 0.2449 - categorical_accuracy: 0.9245
34240/60000 [================>.............] - ETA: 39s - loss: 0.2445 - categorical_accuracy: 0.9246
34304/60000 [================>.............] - ETA: 39s - loss: 0.2443 - categorical_accuracy: 0.9247
34368/60000 [================>.............] - ETA: 39s - loss: 0.2441 - categorical_accuracy: 0.9248
34400/60000 [================>.............] - ETA: 39s - loss: 0.2439 - categorical_accuracy: 0.9248
34464/60000 [================>.............] - ETA: 39s - loss: 0.2436 - categorical_accuracy: 0.9248
34528/60000 [================>.............] - ETA: 39s - loss: 0.2434 - categorical_accuracy: 0.9249
34592/60000 [================>.............] - ETA: 39s - loss: 0.2430 - categorical_accuracy: 0.9250
34624/60000 [================>.............] - ETA: 39s - loss: 0.2429 - categorical_accuracy: 0.9251
34688/60000 [================>.............] - ETA: 39s - loss: 0.2426 - categorical_accuracy: 0.9251
34720/60000 [================>.............] - ETA: 39s - loss: 0.2425 - categorical_accuracy: 0.9252
34784/60000 [================>.............] - ETA: 38s - loss: 0.2421 - categorical_accuracy: 0.9253
34848/60000 [================>.............] - ETA: 38s - loss: 0.2417 - categorical_accuracy: 0.9254
34912/60000 [================>.............] - ETA: 38s - loss: 0.2414 - categorical_accuracy: 0.9254
34944/60000 [================>.............] - ETA: 38s - loss: 0.2415 - categorical_accuracy: 0.9254
34976/60000 [================>.............] - ETA: 38s - loss: 0.2414 - categorical_accuracy: 0.9255
35008/60000 [================>.............] - ETA: 38s - loss: 0.2412 - categorical_accuracy: 0.9255
35040/60000 [================>.............] - ETA: 38s - loss: 0.2410 - categorical_accuracy: 0.9255
35104/60000 [================>.............] - ETA: 38s - loss: 0.2408 - categorical_accuracy: 0.9256
35168/60000 [================>.............] - ETA: 38s - loss: 0.2406 - categorical_accuracy: 0.9257
35232/60000 [================>.............] - ETA: 38s - loss: 0.2404 - categorical_accuracy: 0.9257
35296/60000 [================>.............] - ETA: 38s - loss: 0.2402 - categorical_accuracy: 0.9258
35360/60000 [================>.............] - ETA: 38s - loss: 0.2399 - categorical_accuracy: 0.9259
35392/60000 [================>.............] - ETA: 38s - loss: 0.2399 - categorical_accuracy: 0.9259
35424/60000 [================>.............] - ETA: 37s - loss: 0.2399 - categorical_accuracy: 0.9259
35456/60000 [================>.............] - ETA: 37s - loss: 0.2397 - categorical_accuracy: 0.9260
35488/60000 [================>.............] - ETA: 37s - loss: 0.2396 - categorical_accuracy: 0.9260
35520/60000 [================>.............] - ETA: 37s - loss: 0.2394 - categorical_accuracy: 0.9261
35584/60000 [================>.............] - ETA: 37s - loss: 0.2392 - categorical_accuracy: 0.9261
35648/60000 [================>.............] - ETA: 37s - loss: 0.2391 - categorical_accuracy: 0.9261
35712/60000 [================>.............] - ETA: 37s - loss: 0.2388 - categorical_accuracy: 0.9262
35776/60000 [================>.............] - ETA: 37s - loss: 0.2385 - categorical_accuracy: 0.9263
35840/60000 [================>.............] - ETA: 37s - loss: 0.2383 - categorical_accuracy: 0.9264
35904/60000 [================>.............] - ETA: 37s - loss: 0.2380 - categorical_accuracy: 0.9264
35968/60000 [================>.............] - ETA: 37s - loss: 0.2380 - categorical_accuracy: 0.9264
36032/60000 [=================>............] - ETA: 37s - loss: 0.2378 - categorical_accuracy: 0.9265
36096/60000 [=================>............] - ETA: 36s - loss: 0.2375 - categorical_accuracy: 0.9265
36128/60000 [=================>............] - ETA: 36s - loss: 0.2374 - categorical_accuracy: 0.9265
36160/60000 [=================>............] - ETA: 36s - loss: 0.2372 - categorical_accuracy: 0.9266
36224/60000 [=================>............] - ETA: 36s - loss: 0.2370 - categorical_accuracy: 0.9267
36256/60000 [=================>............] - ETA: 36s - loss: 0.2368 - categorical_accuracy: 0.9267
36320/60000 [=================>............] - ETA: 36s - loss: 0.2366 - categorical_accuracy: 0.9268
36352/60000 [=================>............] - ETA: 36s - loss: 0.2365 - categorical_accuracy: 0.9269
36384/60000 [=================>............] - ETA: 36s - loss: 0.2365 - categorical_accuracy: 0.9269
36448/60000 [=================>............] - ETA: 36s - loss: 0.2362 - categorical_accuracy: 0.9270
36480/60000 [=================>............] - ETA: 36s - loss: 0.2361 - categorical_accuracy: 0.9270
36544/60000 [=================>............] - ETA: 36s - loss: 0.2360 - categorical_accuracy: 0.9271
36608/60000 [=================>............] - ETA: 36s - loss: 0.2356 - categorical_accuracy: 0.9272
36672/60000 [=================>............] - ETA: 36s - loss: 0.2353 - categorical_accuracy: 0.9273
36736/60000 [=================>............] - ETA: 35s - loss: 0.2352 - categorical_accuracy: 0.9273
36800/60000 [=================>............] - ETA: 35s - loss: 0.2349 - categorical_accuracy: 0.9274
36864/60000 [=================>............] - ETA: 35s - loss: 0.2349 - categorical_accuracy: 0.9274
36928/60000 [=================>............] - ETA: 35s - loss: 0.2349 - categorical_accuracy: 0.9275
36992/60000 [=================>............] - ETA: 35s - loss: 0.2346 - categorical_accuracy: 0.9276
37056/60000 [=================>............] - ETA: 35s - loss: 0.2343 - categorical_accuracy: 0.9276
37120/60000 [=================>............] - ETA: 35s - loss: 0.2344 - categorical_accuracy: 0.9277
37184/60000 [=================>............] - ETA: 35s - loss: 0.2342 - categorical_accuracy: 0.9277
37216/60000 [=================>............] - ETA: 35s - loss: 0.2341 - categorical_accuracy: 0.9277
37248/60000 [=================>............] - ETA: 35s - loss: 0.2340 - categorical_accuracy: 0.9277
37280/60000 [=================>............] - ETA: 35s - loss: 0.2340 - categorical_accuracy: 0.9277
37344/60000 [=================>............] - ETA: 35s - loss: 0.2340 - categorical_accuracy: 0.9277
37408/60000 [=================>............] - ETA: 34s - loss: 0.2337 - categorical_accuracy: 0.9278
37472/60000 [=================>............] - ETA: 34s - loss: 0.2334 - categorical_accuracy: 0.9279
37536/60000 [=================>............] - ETA: 34s - loss: 0.2332 - categorical_accuracy: 0.9279
37600/60000 [=================>............] - ETA: 34s - loss: 0.2329 - categorical_accuracy: 0.9280
37664/60000 [=================>............] - ETA: 34s - loss: 0.2327 - categorical_accuracy: 0.9281
37728/60000 [=================>............] - ETA: 34s - loss: 0.2325 - categorical_accuracy: 0.9281
37792/60000 [=================>............] - ETA: 34s - loss: 0.2323 - categorical_accuracy: 0.9282
37856/60000 [=================>............] - ETA: 34s - loss: 0.2320 - categorical_accuracy: 0.9283
37888/60000 [=================>............] - ETA: 34s - loss: 0.2318 - categorical_accuracy: 0.9284
37952/60000 [=================>............] - ETA: 34s - loss: 0.2316 - categorical_accuracy: 0.9284
38016/60000 [==================>...........] - ETA: 33s - loss: 0.2316 - categorical_accuracy: 0.9284
38080/60000 [==================>...........] - ETA: 33s - loss: 0.2315 - categorical_accuracy: 0.9285
38144/60000 [==================>...........] - ETA: 33s - loss: 0.2312 - categorical_accuracy: 0.9286
38176/60000 [==================>...........] - ETA: 33s - loss: 0.2311 - categorical_accuracy: 0.9286
38240/60000 [==================>...........] - ETA: 33s - loss: 0.2308 - categorical_accuracy: 0.9286
38304/60000 [==================>...........] - ETA: 33s - loss: 0.2305 - categorical_accuracy: 0.9287
38336/60000 [==================>...........] - ETA: 33s - loss: 0.2304 - categorical_accuracy: 0.9288
38400/60000 [==================>...........] - ETA: 33s - loss: 0.2303 - categorical_accuracy: 0.9288
38464/60000 [==================>...........] - ETA: 33s - loss: 0.2302 - categorical_accuracy: 0.9288
38528/60000 [==================>...........] - ETA: 33s - loss: 0.2300 - categorical_accuracy: 0.9288
38560/60000 [==================>...........] - ETA: 33s - loss: 0.2301 - categorical_accuracy: 0.9288
38592/60000 [==================>...........] - ETA: 33s - loss: 0.2300 - categorical_accuracy: 0.9289
38656/60000 [==================>...........] - ETA: 32s - loss: 0.2298 - categorical_accuracy: 0.9289
38720/60000 [==================>...........] - ETA: 32s - loss: 0.2296 - categorical_accuracy: 0.9290
38784/60000 [==================>...........] - ETA: 32s - loss: 0.2293 - categorical_accuracy: 0.9291
38816/60000 [==================>...........] - ETA: 32s - loss: 0.2292 - categorical_accuracy: 0.9291
38880/60000 [==================>...........] - ETA: 32s - loss: 0.2291 - categorical_accuracy: 0.9292
38912/60000 [==================>...........] - ETA: 32s - loss: 0.2290 - categorical_accuracy: 0.9293
38944/60000 [==================>...........] - ETA: 32s - loss: 0.2289 - categorical_accuracy: 0.9293
39008/60000 [==================>...........] - ETA: 32s - loss: 0.2286 - categorical_accuracy: 0.9293
39072/60000 [==================>...........] - ETA: 32s - loss: 0.2284 - categorical_accuracy: 0.9294
39136/60000 [==================>...........] - ETA: 32s - loss: 0.2281 - categorical_accuracy: 0.9295
39200/60000 [==================>...........] - ETA: 32s - loss: 0.2277 - categorical_accuracy: 0.9296
39264/60000 [==================>...........] - ETA: 32s - loss: 0.2277 - categorical_accuracy: 0.9296
39328/60000 [==================>...........] - ETA: 31s - loss: 0.2276 - categorical_accuracy: 0.9297
39392/60000 [==================>...........] - ETA: 31s - loss: 0.2276 - categorical_accuracy: 0.9297
39456/60000 [==================>...........] - ETA: 31s - loss: 0.2273 - categorical_accuracy: 0.9298
39488/60000 [==================>...........] - ETA: 31s - loss: 0.2272 - categorical_accuracy: 0.9298
39520/60000 [==================>...........] - ETA: 31s - loss: 0.2271 - categorical_accuracy: 0.9299
39584/60000 [==================>...........] - ETA: 31s - loss: 0.2270 - categorical_accuracy: 0.9299
39648/60000 [==================>...........] - ETA: 31s - loss: 0.2267 - categorical_accuracy: 0.9300
39712/60000 [==================>...........] - ETA: 31s - loss: 0.2265 - categorical_accuracy: 0.9301
39776/60000 [==================>...........] - ETA: 31s - loss: 0.2262 - categorical_accuracy: 0.9302
39840/60000 [==================>...........] - ETA: 31s - loss: 0.2260 - categorical_accuracy: 0.9302
39872/60000 [==================>...........] - ETA: 31s - loss: 0.2259 - categorical_accuracy: 0.9303
39936/60000 [==================>...........] - ETA: 30s - loss: 0.2257 - categorical_accuracy: 0.9303
40000/60000 [===================>..........] - ETA: 30s - loss: 0.2254 - categorical_accuracy: 0.9304
40064/60000 [===================>..........] - ETA: 30s - loss: 0.2252 - categorical_accuracy: 0.9305
40128/60000 [===================>..........] - ETA: 30s - loss: 0.2250 - categorical_accuracy: 0.9305
40192/60000 [===================>..........] - ETA: 30s - loss: 0.2249 - categorical_accuracy: 0.9306
40256/60000 [===================>..........] - ETA: 30s - loss: 0.2246 - categorical_accuracy: 0.9307
40320/60000 [===================>..........] - ETA: 30s - loss: 0.2244 - categorical_accuracy: 0.9307
40384/60000 [===================>..........] - ETA: 30s - loss: 0.2242 - categorical_accuracy: 0.9308
40448/60000 [===================>..........] - ETA: 30s - loss: 0.2240 - categorical_accuracy: 0.9308
40512/60000 [===================>..........] - ETA: 30s - loss: 0.2241 - categorical_accuracy: 0.9308
40544/60000 [===================>..........] - ETA: 30s - loss: 0.2240 - categorical_accuracy: 0.9309
40576/60000 [===================>..........] - ETA: 29s - loss: 0.2238 - categorical_accuracy: 0.9309
40640/60000 [===================>..........] - ETA: 29s - loss: 0.2236 - categorical_accuracy: 0.9310
40704/60000 [===================>..........] - ETA: 29s - loss: 0.2235 - categorical_accuracy: 0.9311
40768/60000 [===================>..........] - ETA: 29s - loss: 0.2233 - categorical_accuracy: 0.9311
40800/60000 [===================>..........] - ETA: 29s - loss: 0.2233 - categorical_accuracy: 0.9312
40832/60000 [===================>..........] - ETA: 29s - loss: 0.2232 - categorical_accuracy: 0.9312
40864/60000 [===================>..........] - ETA: 29s - loss: 0.2230 - categorical_accuracy: 0.9312
40928/60000 [===================>..........] - ETA: 29s - loss: 0.2230 - categorical_accuracy: 0.9312
40960/60000 [===================>..........] - ETA: 29s - loss: 0.2229 - categorical_accuracy: 0.9313
41024/60000 [===================>..........] - ETA: 29s - loss: 0.2227 - categorical_accuracy: 0.9314
41088/60000 [===================>..........] - ETA: 29s - loss: 0.2226 - categorical_accuracy: 0.9314
41152/60000 [===================>..........] - ETA: 29s - loss: 0.2224 - categorical_accuracy: 0.9314
41184/60000 [===================>..........] - ETA: 29s - loss: 0.2224 - categorical_accuracy: 0.9314
41248/60000 [===================>..........] - ETA: 28s - loss: 0.2221 - categorical_accuracy: 0.9315
41280/60000 [===================>..........] - ETA: 28s - loss: 0.2220 - categorical_accuracy: 0.9316
41344/60000 [===================>..........] - ETA: 28s - loss: 0.2218 - categorical_accuracy: 0.9316
41408/60000 [===================>..........] - ETA: 28s - loss: 0.2216 - categorical_accuracy: 0.9317
41472/60000 [===================>..........] - ETA: 28s - loss: 0.2216 - categorical_accuracy: 0.9318
41536/60000 [===================>..........] - ETA: 28s - loss: 0.2213 - categorical_accuracy: 0.9318
41600/60000 [===================>..........] - ETA: 28s - loss: 0.2213 - categorical_accuracy: 0.9319
41664/60000 [===================>..........] - ETA: 28s - loss: 0.2210 - categorical_accuracy: 0.9319
41728/60000 [===================>..........] - ETA: 28s - loss: 0.2208 - categorical_accuracy: 0.9320
41792/60000 [===================>..........] - ETA: 28s - loss: 0.2205 - categorical_accuracy: 0.9321
41856/60000 [===================>..........] - ETA: 27s - loss: 0.2204 - categorical_accuracy: 0.9321
41920/60000 [===================>..........] - ETA: 27s - loss: 0.2201 - categorical_accuracy: 0.9322
41952/60000 [===================>..........] - ETA: 27s - loss: 0.2201 - categorical_accuracy: 0.9322
41984/60000 [===================>..........] - ETA: 27s - loss: 0.2203 - categorical_accuracy: 0.9321
42048/60000 [====================>.........] - ETA: 27s - loss: 0.2201 - categorical_accuracy: 0.9322
42112/60000 [====================>.........] - ETA: 27s - loss: 0.2198 - categorical_accuracy: 0.9322
42176/60000 [====================>.........] - ETA: 27s - loss: 0.2198 - categorical_accuracy: 0.9322
42240/60000 [====================>.........] - ETA: 27s - loss: 0.2196 - categorical_accuracy: 0.9323
42304/60000 [====================>.........] - ETA: 27s - loss: 0.2194 - categorical_accuracy: 0.9323
42336/60000 [====================>.........] - ETA: 27s - loss: 0.2192 - categorical_accuracy: 0.9324
42368/60000 [====================>.........] - ETA: 27s - loss: 0.2191 - categorical_accuracy: 0.9324
42432/60000 [====================>.........] - ETA: 27s - loss: 0.2189 - categorical_accuracy: 0.9325
42464/60000 [====================>.........] - ETA: 27s - loss: 0.2187 - categorical_accuracy: 0.9325
42496/60000 [====================>.........] - ETA: 27s - loss: 0.2187 - categorical_accuracy: 0.9325
42560/60000 [====================>.........] - ETA: 26s - loss: 0.2184 - categorical_accuracy: 0.9326
42624/60000 [====================>.........] - ETA: 26s - loss: 0.2182 - categorical_accuracy: 0.9327
42688/60000 [====================>.........] - ETA: 26s - loss: 0.2179 - categorical_accuracy: 0.9328
42752/60000 [====================>.........] - ETA: 26s - loss: 0.2179 - categorical_accuracy: 0.9328
42784/60000 [====================>.........] - ETA: 26s - loss: 0.2178 - categorical_accuracy: 0.9329
42816/60000 [====================>.........] - ETA: 26s - loss: 0.2178 - categorical_accuracy: 0.9329
42880/60000 [====================>.........] - ETA: 26s - loss: 0.2177 - categorical_accuracy: 0.9329
42912/60000 [====================>.........] - ETA: 26s - loss: 0.2176 - categorical_accuracy: 0.9329
42944/60000 [====================>.........] - ETA: 26s - loss: 0.2176 - categorical_accuracy: 0.9330
42976/60000 [====================>.........] - ETA: 26s - loss: 0.2175 - categorical_accuracy: 0.9330
43008/60000 [====================>.........] - ETA: 26s - loss: 0.2174 - categorical_accuracy: 0.9330
43072/60000 [====================>.........] - ETA: 26s - loss: 0.2172 - categorical_accuracy: 0.9330
43136/60000 [====================>.........] - ETA: 26s - loss: 0.2169 - categorical_accuracy: 0.9331
43200/60000 [====================>.........] - ETA: 25s - loss: 0.2168 - categorical_accuracy: 0.9332
43232/60000 [====================>.........] - ETA: 25s - loss: 0.2167 - categorical_accuracy: 0.9332
43296/60000 [====================>.........] - ETA: 25s - loss: 0.2165 - categorical_accuracy: 0.9333
43360/60000 [====================>.........] - ETA: 25s - loss: 0.2162 - categorical_accuracy: 0.9334
43392/60000 [====================>.........] - ETA: 25s - loss: 0.2161 - categorical_accuracy: 0.9334
43456/60000 [====================>.........] - ETA: 25s - loss: 0.2160 - categorical_accuracy: 0.9334
43488/60000 [====================>.........] - ETA: 25s - loss: 0.2159 - categorical_accuracy: 0.9335
43552/60000 [====================>.........] - ETA: 25s - loss: 0.2156 - categorical_accuracy: 0.9336
43616/60000 [====================>.........] - ETA: 25s - loss: 0.2154 - categorical_accuracy: 0.9336
43648/60000 [====================>.........] - ETA: 25s - loss: 0.2153 - categorical_accuracy: 0.9336
43712/60000 [====================>.........] - ETA: 25s - loss: 0.2151 - categorical_accuracy: 0.9337
43776/60000 [====================>.........] - ETA: 25s - loss: 0.2150 - categorical_accuracy: 0.9337
43808/60000 [====================>.........] - ETA: 24s - loss: 0.2149 - categorical_accuracy: 0.9338
43872/60000 [====================>.........] - ETA: 24s - loss: 0.2149 - categorical_accuracy: 0.9338
43936/60000 [====================>.........] - ETA: 24s - loss: 0.2151 - categorical_accuracy: 0.9338
44000/60000 [=====================>........] - ETA: 24s - loss: 0.2149 - categorical_accuracy: 0.9339
44032/60000 [=====================>........] - ETA: 24s - loss: 0.2148 - categorical_accuracy: 0.9339
44096/60000 [=====================>........] - ETA: 24s - loss: 0.2147 - categorical_accuracy: 0.9339
44160/60000 [=====================>........] - ETA: 24s - loss: 0.2146 - categorical_accuracy: 0.9339
44192/60000 [=====================>........] - ETA: 24s - loss: 0.2145 - categorical_accuracy: 0.9339
44256/60000 [=====================>........] - ETA: 24s - loss: 0.2143 - categorical_accuracy: 0.9340
44320/60000 [=====================>........] - ETA: 24s - loss: 0.2142 - categorical_accuracy: 0.9340
44384/60000 [=====================>........] - ETA: 24s - loss: 0.2140 - categorical_accuracy: 0.9341
44448/60000 [=====================>........] - ETA: 23s - loss: 0.2139 - categorical_accuracy: 0.9341
44512/60000 [=====================>........] - ETA: 23s - loss: 0.2140 - categorical_accuracy: 0.9341
44576/60000 [=====================>........] - ETA: 23s - loss: 0.2137 - categorical_accuracy: 0.9341
44608/60000 [=====================>........] - ETA: 23s - loss: 0.2136 - categorical_accuracy: 0.9342
44640/60000 [=====================>........] - ETA: 23s - loss: 0.2137 - categorical_accuracy: 0.9342
44672/60000 [=====================>........] - ETA: 23s - loss: 0.2136 - categorical_accuracy: 0.9342
44704/60000 [=====================>........] - ETA: 23s - loss: 0.2135 - categorical_accuracy: 0.9343
44768/60000 [=====================>........] - ETA: 23s - loss: 0.2133 - categorical_accuracy: 0.9343
44832/60000 [=====================>........] - ETA: 23s - loss: 0.2132 - categorical_accuracy: 0.9343
44864/60000 [=====================>........] - ETA: 23s - loss: 0.2131 - categorical_accuracy: 0.9344
44928/60000 [=====================>........] - ETA: 23s - loss: 0.2129 - categorical_accuracy: 0.9344
44960/60000 [=====================>........] - ETA: 23s - loss: 0.2129 - categorical_accuracy: 0.9344
45024/60000 [=====================>........] - ETA: 23s - loss: 0.2127 - categorical_accuracy: 0.9345
45088/60000 [=====================>........] - ETA: 23s - loss: 0.2125 - categorical_accuracy: 0.9346
45152/60000 [=====================>........] - ETA: 22s - loss: 0.2123 - categorical_accuracy: 0.9346
45216/60000 [=====================>........] - ETA: 22s - loss: 0.2122 - categorical_accuracy: 0.9346
45248/60000 [=====================>........] - ETA: 22s - loss: 0.2122 - categorical_accuracy: 0.9346
45312/60000 [=====================>........] - ETA: 22s - loss: 0.2119 - categorical_accuracy: 0.9347
45344/60000 [=====================>........] - ETA: 22s - loss: 0.2118 - categorical_accuracy: 0.9347
45376/60000 [=====================>........] - ETA: 22s - loss: 0.2117 - categorical_accuracy: 0.9347
45440/60000 [=====================>........] - ETA: 22s - loss: 0.2118 - categorical_accuracy: 0.9347
45504/60000 [=====================>........] - ETA: 22s - loss: 0.2117 - categorical_accuracy: 0.9348
45568/60000 [=====================>........] - ETA: 22s - loss: 0.2116 - categorical_accuracy: 0.9348
45600/60000 [=====================>........] - ETA: 22s - loss: 0.2114 - categorical_accuracy: 0.9348
45664/60000 [=====================>........] - ETA: 22s - loss: 0.2114 - categorical_accuracy: 0.9349
45696/60000 [=====================>........] - ETA: 22s - loss: 0.2112 - categorical_accuracy: 0.9349
45728/60000 [=====================>........] - ETA: 22s - loss: 0.2111 - categorical_accuracy: 0.9349
45760/60000 [=====================>........] - ETA: 21s - loss: 0.2110 - categorical_accuracy: 0.9350
45824/60000 [=====================>........] - ETA: 21s - loss: 0.2108 - categorical_accuracy: 0.9350
45856/60000 [=====================>........] - ETA: 21s - loss: 0.2107 - categorical_accuracy: 0.9351
45888/60000 [=====================>........] - ETA: 21s - loss: 0.2106 - categorical_accuracy: 0.9351
45920/60000 [=====================>........] - ETA: 21s - loss: 0.2105 - categorical_accuracy: 0.9351
45984/60000 [=====================>........] - ETA: 21s - loss: 0.2103 - categorical_accuracy: 0.9352
46048/60000 [======================>.......] - ETA: 21s - loss: 0.2102 - categorical_accuracy: 0.9352
46080/60000 [======================>.......] - ETA: 21s - loss: 0.2101 - categorical_accuracy: 0.9352
46144/60000 [======================>.......] - ETA: 21s - loss: 0.2100 - categorical_accuracy: 0.9353
46176/60000 [======================>.......] - ETA: 21s - loss: 0.2101 - categorical_accuracy: 0.9352
46240/60000 [======================>.......] - ETA: 21s - loss: 0.2098 - categorical_accuracy: 0.9353
46304/60000 [======================>.......] - ETA: 21s - loss: 0.2098 - categorical_accuracy: 0.9353
46368/60000 [======================>.......] - ETA: 21s - loss: 0.2097 - categorical_accuracy: 0.9354
46432/60000 [======================>.......] - ETA: 20s - loss: 0.2094 - categorical_accuracy: 0.9354
46496/60000 [======================>.......] - ETA: 20s - loss: 0.2093 - categorical_accuracy: 0.9355
46528/60000 [======================>.......] - ETA: 20s - loss: 0.2092 - categorical_accuracy: 0.9355
46592/60000 [======================>.......] - ETA: 20s - loss: 0.2090 - categorical_accuracy: 0.9356
46656/60000 [======================>.......] - ETA: 20s - loss: 0.2089 - categorical_accuracy: 0.9356
46720/60000 [======================>.......] - ETA: 20s - loss: 0.2088 - categorical_accuracy: 0.9356
46784/60000 [======================>.......] - ETA: 20s - loss: 0.2089 - categorical_accuracy: 0.9357
46848/60000 [======================>.......] - ETA: 20s - loss: 0.2087 - categorical_accuracy: 0.9357
46880/60000 [======================>.......] - ETA: 20s - loss: 0.2086 - categorical_accuracy: 0.9358
46912/60000 [======================>.......] - ETA: 20s - loss: 0.2084 - categorical_accuracy: 0.9358
46976/60000 [======================>.......] - ETA: 20s - loss: 0.2083 - categorical_accuracy: 0.9358
47040/60000 [======================>.......] - ETA: 20s - loss: 0.2082 - categorical_accuracy: 0.9358
47072/60000 [======================>.......] - ETA: 19s - loss: 0.2081 - categorical_accuracy: 0.9359
47136/60000 [======================>.......] - ETA: 19s - loss: 0.2084 - categorical_accuracy: 0.9358
47200/60000 [======================>.......] - ETA: 19s - loss: 0.2084 - categorical_accuracy: 0.9358
47232/60000 [======================>.......] - ETA: 19s - loss: 0.2084 - categorical_accuracy: 0.9358
47264/60000 [======================>.......] - ETA: 19s - loss: 0.2083 - categorical_accuracy: 0.9359
47296/60000 [======================>.......] - ETA: 19s - loss: 0.2082 - categorical_accuracy: 0.9359
47328/60000 [======================>.......] - ETA: 19s - loss: 0.2082 - categorical_accuracy: 0.9359
47360/60000 [======================>.......] - ETA: 19s - loss: 0.2082 - categorical_accuracy: 0.9359
47392/60000 [======================>.......] - ETA: 19s - loss: 0.2081 - categorical_accuracy: 0.9360
47424/60000 [======================>.......] - ETA: 19s - loss: 0.2080 - categorical_accuracy: 0.9360
47488/60000 [======================>.......] - ETA: 19s - loss: 0.2079 - categorical_accuracy: 0.9360
47552/60000 [======================>.......] - ETA: 19s - loss: 0.2077 - categorical_accuracy: 0.9361
47616/60000 [======================>.......] - ETA: 19s - loss: 0.2075 - categorical_accuracy: 0.9361
47680/60000 [======================>.......] - ETA: 19s - loss: 0.2074 - categorical_accuracy: 0.9362
47712/60000 [======================>.......] - ETA: 18s - loss: 0.2073 - categorical_accuracy: 0.9362
47744/60000 [======================>.......] - ETA: 18s - loss: 0.2072 - categorical_accuracy: 0.9362
47776/60000 [======================>.......] - ETA: 18s - loss: 0.2071 - categorical_accuracy: 0.9363
47840/60000 [======================>.......] - ETA: 18s - loss: 0.2069 - categorical_accuracy: 0.9363
47872/60000 [======================>.......] - ETA: 18s - loss: 0.2069 - categorical_accuracy: 0.9363
47904/60000 [======================>.......] - ETA: 18s - loss: 0.2067 - categorical_accuracy: 0.9364
47968/60000 [======================>.......] - ETA: 18s - loss: 0.2065 - categorical_accuracy: 0.9364
48032/60000 [=======================>......] - ETA: 18s - loss: 0.2064 - categorical_accuracy: 0.9365
48096/60000 [=======================>......] - ETA: 18s - loss: 0.2065 - categorical_accuracy: 0.9364
48160/60000 [=======================>......] - ETA: 18s - loss: 0.2064 - categorical_accuracy: 0.9365
48224/60000 [=======================>......] - ETA: 18s - loss: 0.2063 - categorical_accuracy: 0.9365
48288/60000 [=======================>......] - ETA: 18s - loss: 0.2063 - categorical_accuracy: 0.9365
48352/60000 [=======================>......] - ETA: 17s - loss: 0.2061 - categorical_accuracy: 0.9366
48416/60000 [=======================>......] - ETA: 17s - loss: 0.2059 - categorical_accuracy: 0.9367
48480/60000 [=======================>......] - ETA: 17s - loss: 0.2058 - categorical_accuracy: 0.9367
48512/60000 [=======================>......] - ETA: 17s - loss: 0.2057 - categorical_accuracy: 0.9367
48544/60000 [=======================>......] - ETA: 17s - loss: 0.2056 - categorical_accuracy: 0.9367
48576/60000 [=======================>......] - ETA: 17s - loss: 0.2055 - categorical_accuracy: 0.9367
48640/60000 [=======================>......] - ETA: 17s - loss: 0.2054 - categorical_accuracy: 0.9368
48704/60000 [=======================>......] - ETA: 17s - loss: 0.2052 - categorical_accuracy: 0.9368
48768/60000 [=======================>......] - ETA: 17s - loss: 0.2050 - categorical_accuracy: 0.9369
48832/60000 [=======================>......] - ETA: 17s - loss: 0.2049 - categorical_accuracy: 0.9369
48864/60000 [=======================>......] - ETA: 17s - loss: 0.2048 - categorical_accuracy: 0.9370
48928/60000 [=======================>......] - ETA: 17s - loss: 0.2046 - categorical_accuracy: 0.9371
48992/60000 [=======================>......] - ETA: 16s - loss: 0.2046 - categorical_accuracy: 0.9371
49024/60000 [=======================>......] - ETA: 16s - loss: 0.2046 - categorical_accuracy: 0.9371
49056/60000 [=======================>......] - ETA: 16s - loss: 0.2045 - categorical_accuracy: 0.9371
49088/60000 [=======================>......] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9371
49120/60000 [=======================>......] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9371
49152/60000 [=======================>......] - ETA: 16s - loss: 0.2043 - categorical_accuracy: 0.9372
49184/60000 [=======================>......] - ETA: 16s - loss: 0.2042 - categorical_accuracy: 0.9372
49216/60000 [=======================>......] - ETA: 16s - loss: 0.2041 - categorical_accuracy: 0.9372
49248/60000 [=======================>......] - ETA: 16s - loss: 0.2041 - categorical_accuracy: 0.9372
49280/60000 [=======================>......] - ETA: 16s - loss: 0.2041 - categorical_accuracy: 0.9372
49344/60000 [=======================>......] - ETA: 16s - loss: 0.2039 - categorical_accuracy: 0.9373
49376/60000 [=======================>......] - ETA: 16s - loss: 0.2038 - categorical_accuracy: 0.9373
49440/60000 [=======================>......] - ETA: 16s - loss: 0.2036 - categorical_accuracy: 0.9373
49504/60000 [=======================>......] - ETA: 16s - loss: 0.2036 - categorical_accuracy: 0.9373
49568/60000 [=======================>......] - ETA: 16s - loss: 0.2036 - categorical_accuracy: 0.9373
49632/60000 [=======================>......] - ETA: 16s - loss: 0.2034 - categorical_accuracy: 0.9374
49696/60000 [=======================>......] - ETA: 15s - loss: 0.2032 - categorical_accuracy: 0.9375
49760/60000 [=======================>......] - ETA: 15s - loss: 0.2031 - categorical_accuracy: 0.9375
49792/60000 [=======================>......] - ETA: 15s - loss: 0.2030 - categorical_accuracy: 0.9376
49824/60000 [=======================>......] - ETA: 15s - loss: 0.2029 - categorical_accuracy: 0.9376
49856/60000 [=======================>......] - ETA: 15s - loss: 0.2028 - categorical_accuracy: 0.9377
49888/60000 [=======================>......] - ETA: 15s - loss: 0.2027 - categorical_accuracy: 0.9377
49952/60000 [=======================>......] - ETA: 15s - loss: 0.2026 - categorical_accuracy: 0.9377
49984/60000 [=======================>......] - ETA: 15s - loss: 0.2025 - categorical_accuracy: 0.9377
50016/60000 [========================>.....] - ETA: 15s - loss: 0.2024 - categorical_accuracy: 0.9378
50080/60000 [========================>.....] - ETA: 15s - loss: 0.2022 - categorical_accuracy: 0.9379
50112/60000 [========================>.....] - ETA: 15s - loss: 0.2022 - categorical_accuracy: 0.9379
50176/60000 [========================>.....] - ETA: 15s - loss: 0.2020 - categorical_accuracy: 0.9379
50240/60000 [========================>.....] - ETA: 15s - loss: 0.2019 - categorical_accuracy: 0.9380
50304/60000 [========================>.....] - ETA: 14s - loss: 0.2018 - categorical_accuracy: 0.9380
50368/60000 [========================>.....] - ETA: 14s - loss: 0.2016 - categorical_accuracy: 0.9381
50432/60000 [========================>.....] - ETA: 14s - loss: 0.2014 - categorical_accuracy: 0.9381
50496/60000 [========================>.....] - ETA: 14s - loss: 0.2012 - categorical_accuracy: 0.9382
50560/60000 [========================>.....] - ETA: 14s - loss: 0.2011 - categorical_accuracy: 0.9382
50624/60000 [========================>.....] - ETA: 14s - loss: 0.2009 - categorical_accuracy: 0.9383
50656/60000 [========================>.....] - ETA: 14s - loss: 0.2008 - categorical_accuracy: 0.9383
50688/60000 [========================>.....] - ETA: 14s - loss: 0.2007 - categorical_accuracy: 0.9383
50752/60000 [========================>.....] - ETA: 14s - loss: 0.2006 - categorical_accuracy: 0.9383
50816/60000 [========================>.....] - ETA: 14s - loss: 0.2004 - categorical_accuracy: 0.9384
50848/60000 [========================>.....] - ETA: 14s - loss: 0.2003 - categorical_accuracy: 0.9384
50912/60000 [========================>.....] - ETA: 14s - loss: 0.2002 - categorical_accuracy: 0.9384
50944/60000 [========================>.....] - ETA: 13s - loss: 0.2001 - categorical_accuracy: 0.9385
51008/60000 [========================>.....] - ETA: 13s - loss: 0.1998 - categorical_accuracy: 0.9386
51072/60000 [========================>.....] - ETA: 13s - loss: 0.1998 - categorical_accuracy: 0.9386
51136/60000 [========================>.....] - ETA: 13s - loss: 0.1997 - categorical_accuracy: 0.9386
51200/60000 [========================>.....] - ETA: 13s - loss: 0.1995 - categorical_accuracy: 0.9387
51232/60000 [========================>.....] - ETA: 13s - loss: 0.1995 - categorical_accuracy: 0.9387
51296/60000 [========================>.....] - ETA: 13s - loss: 0.1994 - categorical_accuracy: 0.9387
51328/60000 [========================>.....] - ETA: 13s - loss: 0.1993 - categorical_accuracy: 0.9387
51360/60000 [========================>.....] - ETA: 13s - loss: 0.1992 - categorical_accuracy: 0.9387
51424/60000 [========================>.....] - ETA: 13s - loss: 0.1991 - categorical_accuracy: 0.9388
51456/60000 [========================>.....] - ETA: 13s - loss: 0.1991 - categorical_accuracy: 0.9388
51520/60000 [========================>.....] - ETA: 13s - loss: 0.1990 - categorical_accuracy: 0.9388
51584/60000 [========================>.....] - ETA: 12s - loss: 0.1988 - categorical_accuracy: 0.9389
51616/60000 [========================>.....] - ETA: 12s - loss: 0.1988 - categorical_accuracy: 0.9389
51680/60000 [========================>.....] - ETA: 12s - loss: 0.1986 - categorical_accuracy: 0.9390
51712/60000 [========================>.....] - ETA: 12s - loss: 0.1985 - categorical_accuracy: 0.9390
51776/60000 [========================>.....] - ETA: 12s - loss: 0.1983 - categorical_accuracy: 0.9390
51808/60000 [========================>.....] - ETA: 12s - loss: 0.1984 - categorical_accuracy: 0.9390
51872/60000 [========================>.....] - ETA: 12s - loss: 0.1982 - categorical_accuracy: 0.9391
51936/60000 [========================>.....] - ETA: 12s - loss: 0.1980 - categorical_accuracy: 0.9391
52000/60000 [=========================>....] - ETA: 12s - loss: 0.1980 - categorical_accuracy: 0.9392
52064/60000 [=========================>....] - ETA: 12s - loss: 0.1978 - categorical_accuracy: 0.9392
52128/60000 [=========================>....] - ETA: 12s - loss: 0.1978 - categorical_accuracy: 0.9392
52192/60000 [=========================>....] - ETA: 12s - loss: 0.1975 - categorical_accuracy: 0.9393
52256/60000 [=========================>....] - ETA: 11s - loss: 0.1975 - categorical_accuracy: 0.9394
52320/60000 [=========================>....] - ETA: 11s - loss: 0.1975 - categorical_accuracy: 0.9394
52384/60000 [=========================>....] - ETA: 11s - loss: 0.1972 - categorical_accuracy: 0.9395
52416/60000 [=========================>....] - ETA: 11s - loss: 0.1971 - categorical_accuracy: 0.9395
52448/60000 [=========================>....] - ETA: 11s - loss: 0.1972 - categorical_accuracy: 0.9395
52480/60000 [=========================>....] - ETA: 11s - loss: 0.1972 - categorical_accuracy: 0.9395
52512/60000 [=========================>....] - ETA: 11s - loss: 0.1971 - categorical_accuracy: 0.9395
52576/60000 [=========================>....] - ETA: 11s - loss: 0.1971 - categorical_accuracy: 0.9395
52640/60000 [=========================>....] - ETA: 11s - loss: 0.1970 - categorical_accuracy: 0.9396
52672/60000 [=========================>....] - ETA: 11s - loss: 0.1969 - categorical_accuracy: 0.9396
52736/60000 [=========================>....] - ETA: 11s - loss: 0.1968 - categorical_accuracy: 0.9396
52800/60000 [=========================>....] - ETA: 11s - loss: 0.1968 - categorical_accuracy: 0.9396
52832/60000 [=========================>....] - ETA: 11s - loss: 0.1967 - categorical_accuracy: 0.9397
52896/60000 [=========================>....] - ETA: 10s - loss: 0.1966 - categorical_accuracy: 0.9397
52928/60000 [=========================>....] - ETA: 10s - loss: 0.1966 - categorical_accuracy: 0.9397
52960/60000 [=========================>....] - ETA: 10s - loss: 0.1965 - categorical_accuracy: 0.9397
52992/60000 [=========================>....] - ETA: 10s - loss: 0.1965 - categorical_accuracy: 0.9397
53024/60000 [=========================>....] - ETA: 10s - loss: 0.1964 - categorical_accuracy: 0.9398
53088/60000 [=========================>....] - ETA: 10s - loss: 0.1962 - categorical_accuracy: 0.9398
53152/60000 [=========================>....] - ETA: 10s - loss: 0.1961 - categorical_accuracy: 0.9399
53184/60000 [=========================>....] - ETA: 10s - loss: 0.1960 - categorical_accuracy: 0.9399
53248/60000 [=========================>....] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9399
53312/60000 [=========================>....] - ETA: 10s - loss: 0.1958 - categorical_accuracy: 0.9400
53376/60000 [=========================>....] - ETA: 10s - loss: 0.1956 - categorical_accuracy: 0.9400
53440/60000 [=========================>....] - ETA: 10s - loss: 0.1956 - categorical_accuracy: 0.9400
53504/60000 [=========================>....] - ETA: 10s - loss: 0.1955 - categorical_accuracy: 0.9400
53568/60000 [=========================>....] - ETA: 9s - loss: 0.1953 - categorical_accuracy: 0.9401 
53632/60000 [=========================>....] - ETA: 9s - loss: 0.1952 - categorical_accuracy: 0.9401
53696/60000 [=========================>....] - ETA: 9s - loss: 0.1951 - categorical_accuracy: 0.9401
53760/60000 [=========================>....] - ETA: 9s - loss: 0.1951 - categorical_accuracy: 0.9402
53824/60000 [=========================>....] - ETA: 9s - loss: 0.1949 - categorical_accuracy: 0.9402
53856/60000 [=========================>....] - ETA: 9s - loss: 0.1948 - categorical_accuracy: 0.9403
53920/60000 [=========================>....] - ETA: 9s - loss: 0.1947 - categorical_accuracy: 0.9403
53984/60000 [=========================>....] - ETA: 9s - loss: 0.1946 - categorical_accuracy: 0.9404
54048/60000 [==========================>...] - ETA: 9s - loss: 0.1944 - categorical_accuracy: 0.9404
54112/60000 [==========================>...] - ETA: 9s - loss: 0.1942 - categorical_accuracy: 0.9405
54144/60000 [==========================>...] - ETA: 9s - loss: 0.1941 - categorical_accuracy: 0.9405
54208/60000 [==========================>...] - ETA: 8s - loss: 0.1940 - categorical_accuracy: 0.9405
54272/60000 [==========================>...] - ETA: 8s - loss: 0.1938 - categorical_accuracy: 0.9406
54336/60000 [==========================>...] - ETA: 8s - loss: 0.1938 - categorical_accuracy: 0.9406
54368/60000 [==========================>...] - ETA: 8s - loss: 0.1937 - categorical_accuracy: 0.9406
54400/60000 [==========================>...] - ETA: 8s - loss: 0.1937 - categorical_accuracy: 0.9406
54464/60000 [==========================>...] - ETA: 8s - loss: 0.1936 - categorical_accuracy: 0.9407
54528/60000 [==========================>...] - ETA: 8s - loss: 0.1935 - categorical_accuracy: 0.9407
54560/60000 [==========================>...] - ETA: 8s - loss: 0.1934 - categorical_accuracy: 0.9407
54624/60000 [==========================>...] - ETA: 8s - loss: 0.1933 - categorical_accuracy: 0.9407
54688/60000 [==========================>...] - ETA: 8s - loss: 0.1931 - categorical_accuracy: 0.9408
54752/60000 [==========================>...] - ETA: 8s - loss: 0.1930 - categorical_accuracy: 0.9408
54816/60000 [==========================>...] - ETA: 8s - loss: 0.1930 - categorical_accuracy: 0.9408
54880/60000 [==========================>...] - ETA: 7s - loss: 0.1928 - categorical_accuracy: 0.9409
54944/60000 [==========================>...] - ETA: 7s - loss: 0.1927 - categorical_accuracy: 0.9409
55008/60000 [==========================>...] - ETA: 7s - loss: 0.1926 - categorical_accuracy: 0.9410
55072/60000 [==========================>...] - ETA: 7s - loss: 0.1926 - categorical_accuracy: 0.9410
55136/60000 [==========================>...] - ETA: 7s - loss: 0.1925 - categorical_accuracy: 0.9410
55200/60000 [==========================>...] - ETA: 7s - loss: 0.1925 - categorical_accuracy: 0.9410
55264/60000 [==========================>...] - ETA: 7s - loss: 0.1923 - categorical_accuracy: 0.9410
55328/60000 [==========================>...] - ETA: 7s - loss: 0.1923 - categorical_accuracy: 0.9410
55360/60000 [==========================>...] - ETA: 7s - loss: 0.1923 - categorical_accuracy: 0.9410
55424/60000 [==========================>...] - ETA: 7s - loss: 0.1923 - categorical_accuracy: 0.9411
55488/60000 [==========================>...] - ETA: 6s - loss: 0.1922 - categorical_accuracy: 0.9411
55552/60000 [==========================>...] - ETA: 6s - loss: 0.1921 - categorical_accuracy: 0.9411
55616/60000 [==========================>...] - ETA: 6s - loss: 0.1920 - categorical_accuracy: 0.9412
55648/60000 [==========================>...] - ETA: 6s - loss: 0.1920 - categorical_accuracy: 0.9412
55680/60000 [==========================>...] - ETA: 6s - loss: 0.1919 - categorical_accuracy: 0.9412
55744/60000 [==========================>...] - ETA: 6s - loss: 0.1918 - categorical_accuracy: 0.9413
55808/60000 [==========================>...] - ETA: 6s - loss: 0.1917 - categorical_accuracy: 0.9413
55840/60000 [==========================>...] - ETA: 6s - loss: 0.1916 - categorical_accuracy: 0.9414
55872/60000 [==========================>...] - ETA: 6s - loss: 0.1915 - categorical_accuracy: 0.9414
55904/60000 [==========================>...] - ETA: 6s - loss: 0.1914 - categorical_accuracy: 0.9414
55968/60000 [==========================>...] - ETA: 6s - loss: 0.1914 - categorical_accuracy: 0.9414
56032/60000 [===========================>..] - ETA: 6s - loss: 0.1913 - categorical_accuracy: 0.9415
56096/60000 [===========================>..] - ETA: 6s - loss: 0.1911 - categorical_accuracy: 0.9415
56160/60000 [===========================>..] - ETA: 5s - loss: 0.1910 - categorical_accuracy: 0.9415
56224/60000 [===========================>..] - ETA: 5s - loss: 0.1909 - categorical_accuracy: 0.9416
56288/60000 [===========================>..] - ETA: 5s - loss: 0.1907 - categorical_accuracy: 0.9416
56352/60000 [===========================>..] - ETA: 5s - loss: 0.1906 - categorical_accuracy: 0.9417
56416/60000 [===========================>..] - ETA: 5s - loss: 0.1904 - categorical_accuracy: 0.9417
56480/60000 [===========================>..] - ETA: 5s - loss: 0.1903 - categorical_accuracy: 0.9417
56544/60000 [===========================>..] - ETA: 5s - loss: 0.1902 - categorical_accuracy: 0.9417
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1902 - categorical_accuracy: 0.9417
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1900 - categorical_accuracy: 0.9418
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1899 - categorical_accuracy: 0.9418
56800/60000 [===========================>..] - ETA: 4s - loss: 0.1898 - categorical_accuracy: 0.9418
56832/60000 [===========================>..] - ETA: 4s - loss: 0.1898 - categorical_accuracy: 0.9418
56864/60000 [===========================>..] - ETA: 4s - loss: 0.1897 - categorical_accuracy: 0.9418
56896/60000 [===========================>..] - ETA: 4s - loss: 0.1897 - categorical_accuracy: 0.9418
56960/60000 [===========================>..] - ETA: 4s - loss: 0.1895 - categorical_accuracy: 0.9419
56992/60000 [===========================>..] - ETA: 4s - loss: 0.1894 - categorical_accuracy: 0.9419
57024/60000 [===========================>..] - ETA: 4s - loss: 0.1894 - categorical_accuracy: 0.9419
57088/60000 [===========================>..] - ETA: 4s - loss: 0.1893 - categorical_accuracy: 0.9419
57152/60000 [===========================>..] - ETA: 4s - loss: 0.1891 - categorical_accuracy: 0.9420
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1889 - categorical_accuracy: 0.9421
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1889 - categorical_accuracy: 0.9421
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1888 - categorical_accuracy: 0.9421
57408/60000 [===========================>..] - ETA: 3s - loss: 0.1886 - categorical_accuracy: 0.9422
57440/60000 [===========================>..] - ETA: 3s - loss: 0.1885 - categorical_accuracy: 0.9422
57504/60000 [===========================>..] - ETA: 3s - loss: 0.1884 - categorical_accuracy: 0.9423
57568/60000 [===========================>..] - ETA: 3s - loss: 0.1882 - categorical_accuracy: 0.9423
57632/60000 [===========================>..] - ETA: 3s - loss: 0.1881 - categorical_accuracy: 0.9424
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1881 - categorical_accuracy: 0.9424
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1880 - categorical_accuracy: 0.9424
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1879 - categorical_accuracy: 0.9424
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1878 - categorical_accuracy: 0.9424
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1876 - categorical_accuracy: 0.9425
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1875 - categorical_accuracy: 0.9425
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1875 - categorical_accuracy: 0.9425
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1875 - categorical_accuracy: 0.9426
58016/60000 [============================>.] - ETA: 3s - loss: 0.1873 - categorical_accuracy: 0.9426
58080/60000 [============================>.] - ETA: 2s - loss: 0.1872 - categorical_accuracy: 0.9426
58144/60000 [============================>.] - ETA: 2s - loss: 0.1871 - categorical_accuracy: 0.9426
58208/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9427
58272/60000 [============================>.] - ETA: 2s - loss: 0.1869 - categorical_accuracy: 0.9427
58336/60000 [============================>.] - ETA: 2s - loss: 0.1867 - categorical_accuracy: 0.9428
58368/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9428
58432/60000 [============================>.] - ETA: 2s - loss: 0.1867 - categorical_accuracy: 0.9428
58496/60000 [============================>.] - ETA: 2s - loss: 0.1866 - categorical_accuracy: 0.9427
58560/60000 [============================>.] - ETA: 2s - loss: 0.1865 - categorical_accuracy: 0.9428
58624/60000 [============================>.] - ETA: 2s - loss: 0.1864 - categorical_accuracy: 0.9428
58656/60000 [============================>.] - ETA: 2s - loss: 0.1864 - categorical_accuracy: 0.9428
58720/60000 [============================>.] - ETA: 1s - loss: 0.1863 - categorical_accuracy: 0.9428
58784/60000 [============================>.] - ETA: 1s - loss: 0.1862 - categorical_accuracy: 0.9429
58848/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9429
58880/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9429
58912/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9429
58976/60000 [============================>.] - ETA: 1s - loss: 0.1859 - categorical_accuracy: 0.9429
59008/60000 [============================>.] - ETA: 1s - loss: 0.1858 - categorical_accuracy: 0.9430
59040/60000 [============================>.] - ETA: 1s - loss: 0.1858 - categorical_accuracy: 0.9430
59104/60000 [============================>.] - ETA: 1s - loss: 0.1858 - categorical_accuracy: 0.9430
59168/60000 [============================>.] - ETA: 1s - loss: 0.1856 - categorical_accuracy: 0.9431
59232/60000 [============================>.] - ETA: 1s - loss: 0.1855 - categorical_accuracy: 0.9431
59296/60000 [============================>.] - ETA: 1s - loss: 0.1853 - categorical_accuracy: 0.9432
59328/60000 [============================>.] - ETA: 1s - loss: 0.1853 - categorical_accuracy: 0.9431
59392/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9432
59456/60000 [============================>.] - ETA: 0s - loss: 0.1850 - categorical_accuracy: 0.9432
59520/60000 [============================>.] - ETA: 0s - loss: 0.1849 - categorical_accuracy: 0.9433
59584/60000 [============================>.] - ETA: 0s - loss: 0.1847 - categorical_accuracy: 0.9433
59648/60000 [============================>.] - ETA: 0s - loss: 0.1846 - categorical_accuracy: 0.9434
59712/60000 [============================>.] - ETA: 0s - loss: 0.1845 - categorical_accuracy: 0.9434
59776/60000 [============================>.] - ETA: 0s - loss: 0.1843 - categorical_accuracy: 0.9434
59840/60000 [============================>.] - ETA: 0s - loss: 0.1842 - categorical_accuracy: 0.9435
59904/60000 [============================>.] - ETA: 0s - loss: 0.1842 - categorical_accuracy: 0.9435
59968/60000 [============================>.] - ETA: 0s - loss: 0.1842 - categorical_accuracy: 0.9435
60000/60000 [==============================] - 96s 2ms/step - loss: 0.1841 - categorical_accuracy: 0.9435 - val_loss: 0.0524 - val_categorical_accuracy: 0.9836

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 15s
  224/10000 [..............................] - ETA: 4s 
  384/10000 [>.............................] - ETA: 3s
  576/10000 [>.............................] - ETA: 3s
  768/10000 [=>............................] - ETA: 3s
  960/10000 [=>............................] - ETA: 2s
 1152/10000 [==>...........................] - ETA: 2s
 1344/10000 [===>..........................] - ETA: 2s
 1536/10000 [===>..........................] - ETA: 2s
 1728/10000 [====>.........................] - ETA: 2s
 1888/10000 [====>.........................] - ETA: 2s
 2048/10000 [=====>........................] - ETA: 2s
 2240/10000 [=====>........................] - ETA: 2s
 2432/10000 [======>.......................] - ETA: 2s
 2624/10000 [======>.......................] - ETA: 2s
 2816/10000 [=======>......................] - ETA: 2s
 3008/10000 [========>.....................] - ETA: 2s
 3168/10000 [========>.....................] - ETA: 2s
 3360/10000 [=========>....................] - ETA: 2s
 3552/10000 [=========>....................] - ETA: 1s
 3744/10000 [==========>...................] - ETA: 1s
 3936/10000 [==========>...................] - ETA: 1s
 4128/10000 [===========>..................] - ETA: 1s
 4320/10000 [===========>..................] - ETA: 1s
 4512/10000 [============>.................] - ETA: 1s
 4704/10000 [=============>................] - ETA: 1s
 4896/10000 [=============>................] - ETA: 1s
 5088/10000 [==============>...............] - ETA: 1s
 5280/10000 [==============>...............] - ETA: 1s
 5472/10000 [===============>..............] - ETA: 1s
 5664/10000 [===============>..............] - ETA: 1s
 5856/10000 [================>.............] - ETA: 1s
 6048/10000 [=================>............] - ETA: 1s
 6240/10000 [=================>............] - ETA: 1s
 6432/10000 [==================>...........] - ETA: 1s
 6592/10000 [==================>...........] - ETA: 1s
 6784/10000 [===================>..........] - ETA: 0s
 6976/10000 [===================>..........] - ETA: 0s
 7168/10000 [====================>.........] - ETA: 0s
 7360/10000 [=====================>........] - ETA: 0s
 7552/10000 [=====================>........] - ETA: 0s
 7744/10000 [======================>.......] - ETA: 0s
 7904/10000 [======================>.......] - ETA: 0s
 8096/10000 [=======================>......] - ETA: 0s
 8288/10000 [=======================>......] - ETA: 0s
 8480/10000 [========================>.....] - ETA: 0s
 8672/10000 [=========================>....] - ETA: 0s
 8864/10000 [=========================>....] - ETA: 0s
 9056/10000 [==========================>...] - ETA: 0s
 9248/10000 [==========================>...] - ETA: 0s
 9440/10000 [===========================>..] - ETA: 0s
 9632/10000 [===========================>..] - ETA: 0s
 9824/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 292us/step
[[3.6602746e-08 7.9060607e-09 1.6493792e-06 ... 9.9999583e-01
  7.3880053e-09 1.5965611e-06]
 [3.7593775e-06 2.4389556e-05 9.9996901e-01 ... 1.3863258e-09
  1.1722750e-06 2.1909365e-09]
 [6.2077760e-08 9.9996686e-01 8.6125729e-06 ... 8.5130287e-06
  1.7069954e-06 1.5666325e-07]
 ...
 [4.7629567e-10 5.0481159e-08 3.2064987e-10 ... 2.5476018e-06
  1.6580138e-06 8.2405986e-06]
 [2.8845213e-07 9.6727923e-09 1.1002412e-08 ... 1.5263220e-08
  5.9265334e-05 1.6687548e-07]
 [7.4323498e-07 3.7446426e-08 1.5622311e-06 ... 2.8356789e-10
  1.6525858e-07 3.1390635e-09]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.05241419630543096, 'accuracy_test:': 0.9836000204086304}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   3a05676..392725e  master     -> origin/master
Updating 3a05676..392725e
Fast-forward
 error_list/20200514/list_log_benchmark_20200514.md |  162 +-
 error_list/20200514/list_log_import_20200514.md    |    2 +-
 error_list/20200514/list_log_jupyter_20200514.md   | 1788 ++++++++++----------
 .../20200514/list_log_pullrequest_20200514.md      |    2 +-
 error_list/20200514/list_log_test_cli_20200514.md  |  138 +-
 error_list/20200514/list_log_testall_20200514.md   |  491 ++----
 6 files changed, 1163 insertions(+), 1420 deletions(-)
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
[master a5d9615] ml_store
 1 file changed, 1298 insertions(+)
To github.com:arita37/mlmodels_store.git
   392725e..a5d9615  master -> master





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
{'loss': 0.5251987650990486, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-14 04:31:41.506687: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master 58071e6] ml_store
 1 file changed, 233 insertions(+)
To github.com:arita37/mlmodels_store.git
   a5d9615..58071e6  master -> master





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
[master bc4169f] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   58071e6..bc4169f  master -> master





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
	Data preprocessing and feature engineering runtime = 0.24s ...
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
 40%|████      | 2/5 [00:22<00:33, 11.03s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.8334155809699363, 'learning_rate': 0.18259780110368454, 'min_data_in_leaf': 29, 'num_leaves': 39} and reward: 0.3878
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xabW'\x07\xc4UX\r\x00\x00\x00learning_rateq\x02G?\xc7_]`\x07\xeaMX\x10\x00\x00\x00min_data_in_leafq\x03K\x1dX\n\x00\x00\x00num_leavesq\x04K'u." and reward: 0.3878
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xabW'\x07\xc4UX\r\x00\x00\x00learning_rateq\x02G?\xc7_]`\x07\xeaMX\x10\x00\x00\x00min_data_in_leafq\x03K\x1dX\n\x00\x00\x00num_leavesq\x04K'u." and reward: 0.3878
 60%|██████    | 3/5 [00:44<00:28, 14.38s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9418923019744654, 'learning_rate': 0.01693612578997151, 'min_data_in_leaf': 15, 'num_leaves': 44} and reward: 0.3906
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee#\xfbS*\xb4\x1aX\r\x00\x00\x00learning_rateq\x02G?\x91W\xb4)\x8e3\xe3X\x10\x00\x00\x00min_data_in_leafq\x03K\x0fX\n\x00\x00\x00num_leavesq\x04K,u.' and reward: 0.3906
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee#\xfbS*\xb4\x1aX\r\x00\x00\x00learning_rateq\x02G?\x91W\xb4)\x8e3\xe3X\x10\x00\x00\x00min_data_in_leafq\x03K\x0fX\n\x00\x00\x00num_leavesq\x04K,u.' and reward: 0.3906
 80%|████████  | 4/5 [01:10<00:18, 18.01s/it] 80%|████████  | 4/5 [01:10<00:17, 17.69s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.9760067194685343, 'learning_rate': 0.04093894702507959, 'min_data_in_leaf': 17, 'num_leaves': 36} and reward: 0.3926
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef;rq\x99e/X\r\x00\x00\x00learning_rateq\x02G?\xa4\xf5\xf3\x1d5\xf6\x1dX\x10\x00\x00\x00min_data_in_leafq\x03K\x11X\n\x00\x00\x00num_leavesq\x04K$u.' and reward: 0.3926
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef;rq\x99e/X\r\x00\x00\x00learning_rateq\x02G?\xa4\xf5\xf3\x1d5\xf6\x1dX\x10\x00\x00\x00min_data_in_leafq\x03K\x11X\n\x00\x00\x00num_leavesq\x04K$u.' and reward: 0.3926
Time for Gradient Boosting hyperparameter optimization: 94.25974106788635
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.9760067194685343, 'learning_rate': 0.04093894702507959, 'min_data_in_leaf': 17, 'num_leaves': 36}
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
 40%|████      | 2/5 [00:52<01:18, 26.16s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.10341832784290347, 'embedding_size_factor': 0.5887169586006009, 'layers.choice': 3, 'learning_rate': 0.0008852903423626213, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1.9240357862535714e-10} and reward: 0.3882
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xbay\x9f\x9f\xe4l>X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe2\xd6\xc4\xf2yI\x0cX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?M\x02Z\x88\xb1X\x00X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xeaq\x99^\xb8\x91\xf4u.' and reward: 0.3882
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xbay\x9f\x9f\xe4l>X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe2\xd6\xc4\xf2yI\x0cX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?M\x02Z\x88\xb1X\x00X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xeaq\x99^\xb8\x91\xf4u.' and reward: 0.3882
 60%|██████    | 3/5 [01:47<01:09, 34.83s/it] 60%|██████    | 3/5 [01:47<01:11, 35.80s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.05591704257339859, 'embedding_size_factor': 0.9016038727359862, 'layers.choice': 0, 'learning_rate': 0.0007599747985778283, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 1.9759491727685842e-08} and reward: 0.3702
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xac\xa1(\x9aH\x98BX\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\xd9\xf0]k"\xdfX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?H\xe7!s\xed0\x9aX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>U7r\x9eS\xcc\xe9u.' and reward: 0.3702
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xac\xa1(\x9aH\x98BX\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\xd9\xf0]k"\xdfX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?H\xe7!s\xed0\x9aX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>U7r\x9eS\xcc\xe9u.' and reward: 0.3702
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 160.25344491004944
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -138.5s of remaining time.
Ensemble size: 22
Ensemble weights: 
[0.13636364 0.18181818 0.09090909 0.18181818 0.31818182 0.
 0.09090909]
	0.4	 = Validation accuracy score
	1.6s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 260.15s ...
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
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7fd4e7f95eb8>

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   bc4169f..ea4e7e8  master     -> origin/master
Updating bc4169f..ea4e7e8
Fast-forward
 .../20200514/list_log_pullrequest_20200514.md      |   2 +-
 error_list/20200514/list_log_testall_20200514.md   | 501 ++++++++++++++++-----
 2 files changed, 391 insertions(+), 112 deletions(-)
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
[master f886790] ml_store
 1 file changed, 214 insertions(+)
To github.com:arita37/mlmodels_store.git
   ea4e7e8..f886790  master -> master





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
[master 4c245c9] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   f886790..4c245c9  master -> master





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
[master 2521552] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   4c245c9..2521552  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7fe0593f9550> 

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
[master 78bfcb5] ml_store
 1 file changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   2521552..78bfcb5  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 1.16777676 -0.66575452 -1.23312074 -1.67419581  1.01313574  0.82502982
  -0.12046457 -0.49821356 -0.31098498 -1.18231813]
 [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]
 [ 0.84806927  0.45194604  0.63019567 -1.57915629  0.82798737 -0.82862798
  -0.10534471  0.52887975 -2.23708651 -0.4148469 ]
 [ 0.78801845  0.30196005  0.70098212 -0.39468968 -1.20376927 -1.17181338
   0.75539203  0.98401224 -0.55968142 -0.19893745]
 [ 1.09488485 -0.06962454 -0.11644415  0.35387043 -1.44189096 -0.18695502
   1.2911889  -0.15323616 -2.43250851 -2.277298  ]
 [ 0.85982375  0.17195713 -0.34898419  0.49056104 -1.15649503 -1.39528303
   0.61472628 -0.52235647 -0.3692559  -0.977773  ]
 [ 1.37661405 -0.60022533  0.72591685 -0.37951752 -0.62754626 -1.01480369
   0.96622086  0.4359862  -0.68748739  3.32107876]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 1.77547698 -0.20339445 -0.19883786  0.24266944  0.96435056  0.20183018
  -0.54577417  0.66102029  1.79215821 -0.7003985 ]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]
 [ 0.46739791 -0.23787527 -0.15449119 -0.75566277 -0.54706224  1.85143789
  -1.46405357  0.20909668  1.55501599 -0.09243232]
 [ 0.93621125  0.20437739 -1.49419377  0.61223252 -0.98437725  0.74488454
   0.49434165 -0.03628129 -0.83239535 -0.4466992 ]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 1.06523311 -0.66486777  1.00806543 -1.94504696 -1.23017555 -0.91542437
   0.33722094  1.22515585 -1.05354607  0.78522692]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 0.89551051  0.92061512  0.79452824 -0.03536792  0.8780991   2.11060505
  -1.02188594 -1.30653407  0.07638048 -1.87316098]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
  -0.87728152 -0.78191168 -0.43750898 -1.44087602]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f522e995ef0>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f5248d1c6a0> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]
 [ 0.99785516 -0.6001388   0.45794708  0.14676526 -0.93355729  0.57180488
   0.57296273 -0.03681766  0.11236849 -0.01781755]
 [ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 0.89551051  0.92061512  0.79452824 -0.03536792  0.8780991   2.11060505
  -1.02188594 -1.30653407  0.07638048 -1.87316098]
 [ 0.8786438   1.03703898 -0.47712421  0.67261975 -1.04948638  2.42887697
   0.52475049  1.00568668  0.35356722 -0.03599018]
 [ 0.44689516  0.38653915  1.35010682 -0.85145566  0.85063796  1.00088142
  -1.1601701  -0.38483225  1.45810824 -0.33128317]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 1.09488485 -0.06962454 -0.11644415  0.35387043 -1.44189096 -0.18695502
   1.2911889  -0.15323616 -2.43250851 -2.277298  ]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 0.89562312 -2.29820588 -0.01952256  1.45652739 -1.85064099  0.31663724
   0.11133727 -2.66412594 -0.42642862 -0.83998891]
 [ 1.37661405 -0.60022533  0.72591685 -0.37951752 -0.62754626 -1.01480369
   0.96622086  0.4359862  -0.68748739  3.32107876]
 [ 1.24549398 -0.72239191  1.1181334   1.09899633  1.00277655 -0.90163449
  -0.53223402 -0.82246719  0.72171129  0.6743961 ]
 [ 0.9292506  -1.10657307 -1.95816909 -0.3592241  -1.21258781  0.5053819
   0.54264529  1.2179409  -1.94068096  0.67780757]
 [ 1.18559003  0.08646441  1.23289919 -2.14246673  1.033341   -0.83016886
   0.36723181  0.45161595  1.10417433 -0.42285696]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 0.46739791 -0.23787527 -0.15449119 -0.75566277 -0.54706224  1.85143789
  -1.46405357  0.20909668  1.55501599 -0.09243232]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 0.56998385 -0.53302033 -0.17545897 -1.42655542  0.60660431  1.76795995
  -0.11598519 -0.47537288  0.47761018 -0.93391466]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]
 [ 1.21619061 -0.01900052  0.86089124 -0.22676019 -1.36419132 -1.56450785
   1.63169151  0.93125568  0.94980882 -0.88018906]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 1.77547698 -0.20339445 -0.19883786  0.24266944  0.96435056  0.20183018
  -0.54577417  0.66102029  1.79215821 -0.7003985 ]]
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
[[ 1.06523311 -0.66486777  1.00806543 -1.94504696 -1.23017555 -0.91542437
   0.33722094  1.22515585 -1.05354607  0.78522692]
 [ 1.1437713   0.7278135   0.35249436  0.51507361  1.17718111 -2.78253447
  -1.94332341  0.58464661  0.32427424 -0.23643695]
 [ 0.8786438   1.03703898 -0.47712421  0.67261975 -1.04948638  2.42887697
   0.52475049  1.00568668  0.35356722 -0.03599018]
 [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]
 [ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
  -0.17961229  0.98495167  1.07125243 -0.59334375]
 [ 0.87122579 -0.20975294 -0.45698786  0.93514778 -0.87353582  1.81252782
   0.92550121  0.14010988 -1.41914878  1.06898597]
 [ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]
 [ 1.09488485 -0.06962454 -0.11644415  0.35387043 -1.44189096 -0.18695502
   1.2911889  -0.15323616 -2.43250851 -2.277298  ]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
  -0.87728152 -0.78191168 -0.43750898 -1.44087602]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 0.88861146  0.84958685 -0.03091142 -0.12215402 -1.14722826 -0.68085157
  -0.32606131 -1.06787658 -0.07667936  0.35571726]
 [ 0.93621125  0.20437739 -1.49419377  0.61223252 -0.98437725  0.74488454
   0.49434165 -0.03628129 -0.83239535 -0.4466992 ]
 [ 0.46739791 -0.23787527 -0.15449119 -0.75566277 -0.54706224  1.85143789
  -1.46405357  0.20909668  1.55501599 -0.09243232]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 0.84806927  0.45194604  0.63019567 -1.57915629  0.82798737 -0.82862798
  -0.10534471  0.52887975 -2.23708651 -0.4148469 ]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 0.88838944  0.28299553  0.01795589  0.10803082 -0.84967187  0.02941762
  -0.50397395 -0.13479313  1.04921829 -1.27046078]]
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
[master 40041e8] ml_store
 1 file changed, 246 insertions(+)
To github.com:arita37/mlmodels_store.git
   78bfcb5..40041e8  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140428795301840
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140428795301616
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140428795300384
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140428795299936
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140428795299432
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140428795299096

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
grad_step = 000000, loss = 1.621858
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 1.419645
grad_step = 000002, loss = 1.225389
grad_step = 000003, loss = 0.995346
grad_step = 000004, loss = 0.733744
grad_step = 000005, loss = 0.483095
grad_step = 000006, loss = 0.320752
grad_step = 000007, loss = 0.270369
grad_step = 000008, loss = 0.234063
grad_step = 000009, loss = 0.192816
grad_step = 000010, loss = 0.141353
grad_step = 000011, loss = 0.083000
grad_step = 000012, loss = 0.045782
grad_step = 000013, loss = 0.038443
grad_step = 000014, loss = 0.047206
grad_step = 000015, loss = 0.057901
grad_step = 000016, loss = 0.063659
grad_step = 000017, loss = 0.061913
grad_step = 000018, loss = 0.051440
grad_step = 000019, loss = 0.035015
grad_step = 000020, loss = 0.020387
grad_step = 000021, loss = 0.014902
grad_step = 000022, loss = 0.019345
grad_step = 000023, loss = 0.027002
grad_step = 000024, loss = 0.030188
grad_step = 000025, loss = 0.027019
grad_step = 000026, loss = 0.020710
grad_step = 000027, loss = 0.015331
grad_step = 000028, loss = 0.012667
grad_step = 000029, loss = 0.012209
grad_step = 000030, loss = 0.012645
grad_step = 000031, loss = 0.012833
grad_step = 000032, loss = 0.012198
grad_step = 000033, loss = 0.010692
grad_step = 000034, loss = 0.008770
grad_step = 000035, loss = 0.007205
grad_step = 000036, loss = 0.006705
grad_step = 000037, loss = 0.007371
grad_step = 000038, loss = 0.008608
grad_step = 000039, loss = 0.009517
grad_step = 000040, loss = 0.009487
grad_step = 000041, loss = 0.008508
grad_step = 000042, loss = 0.007060
grad_step = 000043, loss = 0.005775
grad_step = 000044, loss = 0.005107
grad_step = 000045, loss = 0.005134
grad_step = 000046, loss = 0.005550
grad_step = 000047, loss = 0.005866
grad_step = 000048, loss = 0.005774
grad_step = 000049, loss = 0.005348
grad_step = 000050, loss = 0.004925
grad_step = 000051, loss = 0.004799
grad_step = 000052, loss = 0.004973
grad_step = 000053, loss = 0.005215
grad_step = 000054, loss = 0.005261
grad_step = 000055, loss = 0.005016
grad_step = 000056, loss = 0.004607
grad_step = 000057, loss = 0.004268
grad_step = 000058, loss = 0.004166
grad_step = 000059, loss = 0.004283
grad_step = 000060, loss = 0.004453
grad_step = 000061, loss = 0.004508
grad_step = 000062, loss = 0.004405
grad_step = 000063, loss = 0.004223
grad_step = 000064, loss = 0.004075
grad_step = 000065, loss = 0.004019
grad_step = 000066, loss = 0.004036
grad_step = 000067, loss = 0.004065
grad_step = 000068, loss = 0.004054
grad_step = 000069, loss = 0.003987
grad_step = 000070, loss = 0.003896
grad_step = 000071, loss = 0.003830
grad_step = 000072, loss = 0.003814
grad_step = 000073, loss = 0.003829
grad_step = 000074, loss = 0.003822
grad_step = 000075, loss = 0.003772
grad_step = 000076, loss = 0.003699
grad_step = 000077, loss = 0.003642
grad_step = 000078, loss = 0.003619
grad_step = 000079, loss = 0.003617
grad_step = 000080, loss = 0.003608
grad_step = 000081, loss = 0.003578
grad_step = 000082, loss = 0.003534
grad_step = 000083, loss = 0.003497
grad_step = 000084, loss = 0.003474
grad_step = 000085, loss = 0.003459
grad_step = 000086, loss = 0.003436
grad_step = 000087, loss = 0.003399
grad_step = 000088, loss = 0.003355
grad_step = 000089, loss = 0.003317
grad_step = 000090, loss = 0.003291
grad_step = 000091, loss = 0.003271
grad_step = 000092, loss = 0.003248
grad_step = 000093, loss = 0.003216
grad_step = 000094, loss = 0.003180
grad_step = 000095, loss = 0.003147
grad_step = 000096, loss = 0.003116
grad_step = 000097, loss = 0.003085
grad_step = 000098, loss = 0.003052
grad_step = 000099, loss = 0.003017
grad_step = 000100, loss = 0.002982
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002947
grad_step = 000102, loss = 0.002913
grad_step = 000103, loss = 0.002881
grad_step = 000104, loss = 0.002846
grad_step = 000105, loss = 0.002808
grad_step = 000106, loss = 0.002771
grad_step = 000107, loss = 0.002736
grad_step = 000108, loss = 0.002701
grad_step = 000109, loss = 0.002664
grad_step = 000110, loss = 0.002627
grad_step = 000111, loss = 0.002590
grad_step = 000112, loss = 0.002554
grad_step = 000113, loss = 0.002519
grad_step = 000114, loss = 0.002481
grad_step = 000115, loss = 0.002444
grad_step = 000116, loss = 0.002407
grad_step = 000117, loss = 0.002371
grad_step = 000118, loss = 0.002335
grad_step = 000119, loss = 0.002299
grad_step = 000120, loss = 0.002264
grad_step = 000121, loss = 0.002229
grad_step = 000122, loss = 0.002194
grad_step = 000123, loss = 0.002159
grad_step = 000124, loss = 0.002126
grad_step = 000125, loss = 0.002093
grad_step = 000126, loss = 0.002061
grad_step = 000127, loss = 0.002029
grad_step = 000128, loss = 0.001998
grad_step = 000129, loss = 0.001966
grad_step = 000130, loss = 0.001934
grad_step = 000131, loss = 0.001903
grad_step = 000132, loss = 0.001873
grad_step = 000133, loss = 0.001843
grad_step = 000134, loss = 0.001814
grad_step = 000135, loss = 0.001785
grad_step = 000136, loss = 0.001757
grad_step = 000137, loss = 0.001730
grad_step = 000138, loss = 0.001704
grad_step = 000139, loss = 0.001679
grad_step = 000140, loss = 0.001656
grad_step = 000141, loss = 0.001634
grad_step = 000142, loss = 0.001614
grad_step = 000143, loss = 0.001595
grad_step = 000144, loss = 0.001577
grad_step = 000145, loss = 0.001561
grad_step = 000146, loss = 0.001547
grad_step = 000147, loss = 0.001533
grad_step = 000148, loss = 0.001521
grad_step = 000149, loss = 0.001511
grad_step = 000150, loss = 0.001501
grad_step = 000151, loss = 0.001489
grad_step = 000152, loss = 0.001477
grad_step = 000153, loss = 0.001492
grad_step = 000154, loss = 0.001454
grad_step = 000155, loss = 0.001442
grad_step = 000156, loss = 0.001433
grad_step = 000157, loss = 0.001414
grad_step = 000158, loss = 0.001406
grad_step = 000159, loss = 0.001398
grad_step = 000160, loss = 0.001390
grad_step = 000161, loss = 0.001380
grad_step = 000162, loss = 0.001366
grad_step = 000163, loss = 0.001357
grad_step = 000164, loss = 0.001344
grad_step = 000165, loss = 0.001335
grad_step = 000166, loss = 0.001324
grad_step = 000167, loss = 0.001318
grad_step = 000168, loss = 0.001308
grad_step = 000169, loss = 0.001299
grad_step = 000170, loss = 0.001287
grad_step = 000171, loss = 0.001279
grad_step = 000172, loss = 0.001270
grad_step = 000173, loss = 0.001261
grad_step = 000174, loss = 0.001250
grad_step = 000175, loss = 0.001241
grad_step = 000176, loss = 0.001232
grad_step = 000177, loss = 0.001223
grad_step = 000178, loss = 0.001214
grad_step = 000179, loss = 0.001205
grad_step = 000180, loss = 0.001196
grad_step = 000181, loss = 0.001187
grad_step = 000182, loss = 0.001178
grad_step = 000183, loss = 0.001169
grad_step = 000184, loss = 0.001161
grad_step = 000185, loss = 0.001152
grad_step = 000186, loss = 0.001143
grad_step = 000187, loss = 0.001134
grad_step = 000188, loss = 0.001125
grad_step = 000189, loss = 0.001116
grad_step = 000190, loss = 0.001107
grad_step = 000191, loss = 0.001098
grad_step = 000192, loss = 0.001088
grad_step = 000193, loss = 0.001079
grad_step = 000194, loss = 0.001070
grad_step = 000195, loss = 0.001065
grad_step = 000196, loss = 0.001058
grad_step = 000197, loss = 0.001049
grad_step = 000198, loss = 0.001037
grad_step = 000199, loss = 0.001031
grad_step = 000200, loss = 0.001024
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001013
grad_step = 000202, loss = 0.001003
grad_step = 000203, loss = 0.000997
grad_step = 000204, loss = 0.000990
grad_step = 000205, loss = 0.000980
grad_step = 000206, loss = 0.000974
grad_step = 000207, loss = 0.000967
grad_step = 000208, loss = 0.000959
grad_step = 000209, loss = 0.000951
grad_step = 000210, loss = 0.000947
grad_step = 000211, loss = 0.000939
grad_step = 000212, loss = 0.000931
grad_step = 000213, loss = 0.000927
grad_step = 000214, loss = 0.000920
grad_step = 000215, loss = 0.000914
grad_step = 000216, loss = 0.000907
grad_step = 000217, loss = 0.000901
grad_step = 000218, loss = 0.000896
grad_step = 000219, loss = 0.000890
grad_step = 000220, loss = 0.000883
grad_step = 000221, loss = 0.000878
grad_step = 000222, loss = 0.000872
grad_step = 000223, loss = 0.000866
grad_step = 000224, loss = 0.000861
grad_step = 000225, loss = 0.000854
grad_step = 000226, loss = 0.000849
grad_step = 000227, loss = 0.000844
grad_step = 000228, loss = 0.000838
grad_step = 000229, loss = 0.000832
grad_step = 000230, loss = 0.000827
grad_step = 000231, loss = 0.000822
grad_step = 000232, loss = 0.000816
grad_step = 000233, loss = 0.000812
grad_step = 000234, loss = 0.000807
grad_step = 000235, loss = 0.000800
grad_step = 000236, loss = 0.000795
grad_step = 000237, loss = 0.000791
grad_step = 000238, loss = 0.000784
grad_step = 000239, loss = 0.000780
grad_step = 000240, loss = 0.000776
grad_step = 000241, loss = 0.000769
grad_step = 000242, loss = 0.000766
grad_step = 000243, loss = 0.000762
grad_step = 000244, loss = 0.000755
grad_step = 000245, loss = 0.000752
grad_step = 000246, loss = 0.000747
grad_step = 000247, loss = 0.000740
grad_step = 000248, loss = 0.000737
grad_step = 000249, loss = 0.000733
grad_step = 000250, loss = 0.000726
grad_step = 000251, loss = 0.000723
grad_step = 000252, loss = 0.000720
grad_step = 000253, loss = 0.000715
grad_step = 000254, loss = 0.000714
grad_step = 000255, loss = 0.000712
grad_step = 000256, loss = 0.000709
grad_step = 000257, loss = 0.000707
grad_step = 000258, loss = 0.000698
grad_step = 000259, loss = 0.000686
grad_step = 000260, loss = 0.000681
grad_step = 000261, loss = 0.000681
grad_step = 000262, loss = 0.000680
grad_step = 000263, loss = 0.000676
grad_step = 000264, loss = 0.000669
grad_step = 000265, loss = 0.000660
grad_step = 000266, loss = 0.000656
grad_step = 000267, loss = 0.000656
grad_step = 000268, loss = 0.000653
grad_step = 000269, loss = 0.000647
grad_step = 000270, loss = 0.000642
grad_step = 000271, loss = 0.000639
grad_step = 000272, loss = 0.000634
grad_step = 000273, loss = 0.000628
grad_step = 000274, loss = 0.000625
grad_step = 000275, loss = 0.000623
grad_step = 000276, loss = 0.000620
grad_step = 000277, loss = 0.000613
grad_step = 000278, loss = 0.000609
grad_step = 000279, loss = 0.000605
grad_step = 000280, loss = 0.000602
grad_step = 000281, loss = 0.000597
grad_step = 000282, loss = 0.000593
grad_step = 000283, loss = 0.000589
grad_step = 000284, loss = 0.000586
grad_step = 000285, loss = 0.000583
grad_step = 000286, loss = 0.000580
grad_step = 000287, loss = 0.000577
grad_step = 000288, loss = 0.000577
grad_step = 000289, loss = 0.000580
grad_step = 000290, loss = 0.000584
grad_step = 000291, loss = 0.000585
grad_step = 000292, loss = 0.000578
grad_step = 000293, loss = 0.000564
grad_step = 000294, loss = 0.000549
grad_step = 000295, loss = 0.000545
grad_step = 000296, loss = 0.000550
grad_step = 000297, loss = 0.000552
grad_step = 000298, loss = 0.000543
grad_step = 000299, loss = 0.000533
grad_step = 000300, loss = 0.000529
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000526
grad_step = 000302, loss = 0.000523
grad_step = 000303, loss = 0.000522
grad_step = 000304, loss = 0.000517
grad_step = 000305, loss = 0.000509
grad_step = 000306, loss = 0.000505
grad_step = 000307, loss = 0.000505
grad_step = 000308, loss = 0.000503
grad_step = 000309, loss = 0.000498
grad_step = 000310, loss = 0.000495
grad_step = 000311, loss = 0.000491
grad_step = 000312, loss = 0.000486
grad_step = 000313, loss = 0.000482
grad_step = 000314, loss = 0.000480
grad_step = 000315, loss = 0.000479
grad_step = 000316, loss = 0.000477
grad_step = 000317, loss = 0.000473
grad_step = 000318, loss = 0.000471
grad_step = 000319, loss = 0.000470
grad_step = 000320, loss = 0.000467
grad_step = 000321, loss = 0.000463
grad_step = 000322, loss = 0.000460
grad_step = 000323, loss = 0.000459
grad_step = 000324, loss = 0.000458
grad_step = 000325, loss = 0.000457
grad_step = 000326, loss = 0.000458
grad_step = 000327, loss = 0.000462
grad_step = 000328, loss = 0.000462
grad_step = 000329, loss = 0.000461
grad_step = 000330, loss = 0.000459
grad_step = 000331, loss = 0.000456
grad_step = 000332, loss = 0.000447
grad_step = 000333, loss = 0.000439
grad_step = 000334, loss = 0.000437
grad_step = 000335, loss = 0.000438
grad_step = 000336, loss = 0.000437
grad_step = 000337, loss = 0.000437
grad_step = 000338, loss = 0.000439
grad_step = 000339, loss = 0.000440
grad_step = 000340, loss = 0.000437
grad_step = 000341, loss = 0.000434
grad_step = 000342, loss = 0.000432
grad_step = 000343, loss = 0.000428
grad_step = 000344, loss = 0.000424
grad_step = 000345, loss = 0.000422
grad_step = 000346, loss = 0.000422
grad_step = 000347, loss = 0.000420
grad_step = 000348, loss = 0.000418
grad_step = 000349, loss = 0.000417
grad_step = 000350, loss = 0.000417
grad_step = 000351, loss = 0.000417
grad_step = 000352, loss = 0.000416
grad_step = 000353, loss = 0.000416
grad_step = 000354, loss = 0.000419
grad_step = 000355, loss = 0.000423
grad_step = 000356, loss = 0.000428
grad_step = 000357, loss = 0.000438
grad_step = 000358, loss = 0.000445
grad_step = 000359, loss = 0.000442
grad_step = 000360, loss = 0.000424
grad_step = 000361, loss = 0.000411
grad_step = 000362, loss = 0.000405
grad_step = 000363, loss = 0.000409
grad_step = 000364, loss = 0.000419
grad_step = 000365, loss = 0.000416
grad_step = 000366, loss = 0.000404
grad_step = 000367, loss = 0.000398
grad_step = 000368, loss = 0.000402
grad_step = 000369, loss = 0.000405
grad_step = 000370, loss = 0.000401
grad_step = 000371, loss = 0.000395
grad_step = 000372, loss = 0.000394
grad_step = 000373, loss = 0.000392
grad_step = 000374, loss = 0.000391
grad_step = 000375, loss = 0.000393
grad_step = 000376, loss = 0.000393
grad_step = 000377, loss = 0.000389
grad_step = 000378, loss = 0.000386
grad_step = 000379, loss = 0.000384
grad_step = 000380, loss = 0.000384
grad_step = 000381, loss = 0.000385
grad_step = 000382, loss = 0.000385
grad_step = 000383, loss = 0.000387
grad_step = 000384, loss = 0.000389
grad_step = 000385, loss = 0.000390
grad_step = 000386, loss = 0.000388
grad_step = 000387, loss = 0.000384
grad_step = 000388, loss = 0.000380
grad_step = 000389, loss = 0.000374
grad_step = 000390, loss = 0.000373
grad_step = 000391, loss = 0.000374
grad_step = 000392, loss = 0.000376
grad_step = 000393, loss = 0.000377
grad_step = 000394, loss = 0.000376
grad_step = 000395, loss = 0.000374
grad_step = 000396, loss = 0.000374
grad_step = 000397, loss = 0.000373
grad_step = 000398, loss = 0.000370
grad_step = 000399, loss = 0.000366
grad_step = 000400, loss = 0.000366
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000365
grad_step = 000402, loss = 0.000362
grad_step = 000403, loss = 0.000361
grad_step = 000404, loss = 0.000363
grad_step = 000405, loss = 0.000365
grad_step = 000406, loss = 0.000367
grad_step = 000407, loss = 0.000364
grad_step = 000408, loss = 0.000362
grad_step = 000409, loss = 0.000361
grad_step = 000410, loss = 0.000363
grad_step = 000411, loss = 0.000366
grad_step = 000412, loss = 0.000368
grad_step = 000413, loss = 0.000370
grad_step = 000414, loss = 0.000371
grad_step = 000415, loss = 0.000368
grad_step = 000416, loss = 0.000363
grad_step = 000417, loss = 0.000355
grad_step = 000418, loss = 0.000350
grad_step = 000419, loss = 0.000349
grad_step = 000420, loss = 0.000352
grad_step = 000421, loss = 0.000356
grad_step = 000422, loss = 0.000358
grad_step = 000423, loss = 0.000356
grad_step = 000424, loss = 0.000352
grad_step = 000425, loss = 0.000346
grad_step = 000426, loss = 0.000342
grad_step = 000427, loss = 0.000341
grad_step = 000428, loss = 0.000342
grad_step = 000429, loss = 0.000344
grad_step = 000430, loss = 0.000346
grad_step = 000431, loss = 0.000349
grad_step = 000432, loss = 0.000351
grad_step = 000433, loss = 0.000355
grad_step = 000434, loss = 0.000349
grad_step = 000435, loss = 0.000341
grad_step = 000436, loss = 0.000336
grad_step = 000437, loss = 0.000336
grad_step = 000438, loss = 0.000340
grad_step = 000439, loss = 0.000339
grad_step = 000440, loss = 0.000341
grad_step = 000441, loss = 0.000336
grad_step = 000442, loss = 0.000334
grad_step = 000443, loss = 0.000338
grad_step = 000444, loss = 0.000341
grad_step = 000445, loss = 0.000341
grad_step = 000446, loss = 0.000333
grad_step = 000447, loss = 0.000329
grad_step = 000448, loss = 0.000329
grad_step = 000449, loss = 0.000329
grad_step = 000450, loss = 0.000330
grad_step = 000451, loss = 0.000325
grad_step = 000452, loss = 0.000322
grad_step = 000453, loss = 0.000323
grad_step = 000454, loss = 0.000326
grad_step = 000455, loss = 0.000327
grad_step = 000456, loss = 0.000324
grad_step = 000457, loss = 0.000324
grad_step = 000458, loss = 0.000327
grad_step = 000459, loss = 0.000331
grad_step = 000460, loss = 0.000338
grad_step = 000461, loss = 0.000337
grad_step = 000462, loss = 0.000331
grad_step = 000463, loss = 0.000320
grad_step = 000464, loss = 0.000316
grad_step = 000465, loss = 0.000318
grad_step = 000466, loss = 0.000316
grad_step = 000467, loss = 0.000313
grad_step = 000468, loss = 0.000313
grad_step = 000469, loss = 0.000314
grad_step = 000470, loss = 0.000316
grad_step = 000471, loss = 0.000316
grad_step = 000472, loss = 0.000316
grad_step = 000473, loss = 0.000310
grad_step = 000474, loss = 0.000304
grad_step = 000475, loss = 0.000305
grad_step = 000476, loss = 0.000307
grad_step = 000477, loss = 0.000306
grad_step = 000478, loss = 0.000302
grad_step = 000479, loss = 0.000302
grad_step = 000480, loss = 0.000305
grad_step = 000481, loss = 0.000310
grad_step = 000482, loss = 0.000317
grad_step = 000483, loss = 0.000317
grad_step = 000484, loss = 0.000317
grad_step = 000485, loss = 0.000316
grad_step = 000486, loss = 0.000317
grad_step = 000487, loss = 0.000311
grad_step = 000488, loss = 0.000301
grad_step = 000489, loss = 0.000294
grad_step = 000490, loss = 0.000293
grad_step = 000491, loss = 0.000298
grad_step = 000492, loss = 0.000301
grad_step = 000493, loss = 0.000300
grad_step = 000494, loss = 0.000296
grad_step = 000495, loss = 0.000292
grad_step = 000496, loss = 0.000291
grad_step = 000497, loss = 0.000291
grad_step = 000498, loss = 0.000293
grad_step = 000499, loss = 0.000293
grad_step = 000500, loss = 0.000291
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000287
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
[[0.83562857 0.86221135 0.92521834 0.95195895 1.0294505 ]
 [0.8759101  0.9177356  0.9536493  0.9897206  1.0162365 ]
 [0.89117455 0.9224646  0.9887581  0.9892066  0.9584077 ]
 [0.9424716  0.982829   0.9787226  0.9370553  0.919752  ]
 [0.98994446 1.0058959  0.94715923 0.91055214 0.8533291 ]
 [0.9732537  0.9558608  0.9154321  0.85099864 0.85348797]
 [0.9353581  0.9014401  0.86150867 0.8645797  0.81039643]
 [0.8850898  0.85249925 0.85007113 0.8132198  0.84813726]
 [0.83156985 0.8293618  0.81878585 0.8455888  0.85971963]
 [0.82937795 0.8160223  0.8331713  0.8659463  0.8349333 ]
 [0.7975756  0.8098159  0.84753096 0.83158976 0.9262121 ]
 [0.8242843  0.8500852  0.82429755 0.9335022  0.94401985]
 [0.83072996 0.85903835 0.92109984 0.94856197 1.0280796 ]
 [0.8813633  0.9227448  0.9561857  0.9929068  0.99808794]
 [0.90214205 0.9343207  0.9855098  0.98253345 0.9318647 ]
 [0.95352733 0.9926666  0.9710413  0.9224933  0.89437795]
 [0.9928971  0.99998194 0.9343827  0.8925555  0.8281386 ]
 [0.96987146 0.9433557  0.89485043 0.8292372  0.8415955 ]
 [0.9334085  0.8938176  0.8424558  0.84760416 0.8107115 ]
 [0.8945101  0.85560656 0.83763945 0.8110696  0.84632874]
 [0.84674776 0.8376821  0.81628203 0.8477548  0.8598987 ]
 [0.8528799  0.8331872  0.83085614 0.8712888  0.8403925 ]
 [0.8154669  0.8221607  0.8519624  0.8365073  0.92993695]
 [0.83992183 0.8620685  0.8279368  0.93699735 0.94842446]
 [0.8420296  0.867787   0.9245505  0.9560343  1.0353024 ]
 [0.8848299  0.9253478  0.95611835 0.9961939  1.0304666 ]
 [0.90353155 0.9358727  0.99570584 1.001935   0.9709553 ]
 [0.9545834  0.99743414 0.9877261  0.9497153  0.9286463 ]
 [0.9991032  1.0194464  0.9567393  0.92254996 0.85914475]
 [0.98555267 0.967835   0.9241803  0.8572217  0.85937464]
 [0.9442762  0.90985274 0.8672365  0.8683753  0.81897736]]

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
[master 635cefb] ml_store
 1 file changed, 1122 insertions(+)
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (fetch first)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally. This is usually caused by another repository pushing
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
From github.com:arita37/mlmodels_store
   40041e8..9e05fd7  master     -> origin/master
Merge made by the 'recursive' strategy.
 ...-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 2506 ++++++++++++++++++++
 1 file changed, 2506 insertions(+)
 create mode 100644 log_benchmark/log_benchmark_2020-05-14-04-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
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
[master b9c62f5] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   9e05fd7..b9c62f5  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 26513800.90B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 451886.95B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 5182464/440473133 [00:00<00:08, 51808508.14B/s]  2%|▏         | 10855424/440473133 [00:00<00:08, 53192701.33B/s]  4%|▍         | 16744448/440473133 [00:00<00:07, 54781884.59B/s]  5%|▌         | 22591488/440473133 [00:00<00:07, 55837732.80B/s]  6%|▋         | 28583936/440473133 [00:00<00:07, 57000569.37B/s]  8%|▊         | 34070528/440473133 [00:00<00:07, 56341751.28B/s]  9%|▉         | 40022016/440473133 [00:00<00:06, 57256300.10B/s] 10%|█         | 45917184/440473133 [00:00<00:06, 57753630.88B/s] 12%|█▏        | 51959808/440473133 [00:00<00:06, 58527555.68B/s] 13%|█▎        | 57798656/440473133 [00:01<00:06, 58484606.99B/s] 14%|█▍        | 63698944/440473133 [00:01<00:06, 58636937.89B/s] 16%|█▌        | 69662720/440473133 [00:01<00:06, 58931615.26B/s] 17%|█▋        | 75488256/440473133 [00:01<00:06, 58066549.31B/s] 18%|█▊        | 81434624/440473133 [00:01<00:06, 58475595.33B/s] 20%|█▉        | 87336960/440473133 [00:01<00:06, 58637995.85B/s] 21%|██        | 93381632/440473133 [00:01<00:05, 59167752.67B/s] 23%|██▎       | 99482624/440473133 [00:01<00:05, 59704912.28B/s] 24%|██▍       | 105456640/440473133 [00:01<00:05, 59714944.28B/s] 25%|██▌       | 111534080/440473133 [00:01<00:05, 60026693.86B/s] 27%|██▋       | 117533696/440473133 [00:02<00:05, 59653676.37B/s] 28%|██▊       | 123497472/440473133 [00:02<00:05, 58576361.28B/s] 29%|██▉       | 129358848/440473133 [00:02<00:05, 56689049.08B/s] 31%|███       | 135043072/440473133 [00:02<00:05, 56084382.93B/s] 32%|███▏      | 140663808/440473133 [00:02<00:05, 55951891.04B/s] 33%|███▎      | 146629632/440473133 [00:02<00:05, 57013602.62B/s] 35%|███▍      | 152553472/440473133 [00:02<00:04, 57659216.44B/s] 36%|███▌      | 158477312/440473133 [00:02<00:04, 58121628.57B/s] 37%|███▋      | 164500480/440473133 [00:02<00:04, 58734588.65B/s] 39%|███▊      | 170482688/440473133 [00:02<00:04, 59055552.54B/s] 40%|████      | 176504832/440473133 [00:03<00:04, 59399855.40B/s] 41%|████▏     | 182502400/440473133 [00:03<00:04, 59568577.64B/s] 43%|████▎     | 188463104/440473133 [00:03<00:04, 59382163.97B/s] 44%|████▍     | 194404352/440473133 [00:03<00:04, 59202903.57B/s] 45%|████▌     | 200343552/440473133 [00:03<00:04, 59170506.99B/s] 47%|████▋     | 206393344/440473133 [00:03<00:03, 59558019.29B/s] 48%|████▊     | 212390912/440473133 [00:03<00:03, 59678347.32B/s] 50%|████▉     | 218359808/440473133 [00:03<00:03, 59606308.57B/s] 51%|█████     | 224321536/440473133 [00:03<00:03, 59588308.92B/s] 52%|█████▏    | 230354944/440473133 [00:03<00:03, 59805110.45B/s] 54%|█████▎    | 236336128/440473133 [00:04<00:03, 59504392.08B/s] 55%|█████▌    | 242309120/440473133 [00:04<00:03, 59569083.80B/s] 56%|█████▋    | 248266752/440473133 [00:04<00:03, 58810431.28B/s] 58%|█████▊    | 254150656/440473133 [00:04<00:03, 58361964.24B/s] 59%|█████▉    | 260233216/440473133 [00:04<00:03, 59078709.49B/s] 61%|██████    | 266506240/440473133 [00:04<00:02, 60125416.13B/s] 62%|██████▏   | 272534528/440473133 [00:04<00:02, 60170008.54B/s] 63%|██████▎   | 278556672/440473133 [00:04<00:02, 59780679.76B/s] 65%|██████▍   | 284539904/440473133 [00:04<00:02, 58890353.10B/s] 66%|██████▌   | 290435072/440473133 [00:04<00:02, 57695197.62B/s] 67%|██████▋   | 296349696/440473133 [00:05<00:02, 58119849.98B/s] 69%|██████▊   | 302320640/440473133 [00:05<00:02, 58584906.94B/s] 70%|██████▉   | 308293632/440473133 [00:05<00:02, 58922627.26B/s] 71%|███████▏  | 314223616/440473133 [00:05<00:02, 59033453.47B/s] 73%|███████▎  | 320224256/440473133 [00:05<00:02, 59320347.37B/s] 74%|███████▍  | 326159360/440473133 [00:05<00:01, 58579093.52B/s] 75%|███████▌  | 332021760/440473133 [00:05<00:01, 57361328.05B/s] 77%|███████▋  | 337766400/440473133 [00:05<00:01, 56860508.51B/s] 78%|███████▊  | 343459840/440473133 [00:05<00:01, 56503204.12B/s] 79%|███████▉  | 349116416/440473133 [00:05<00:01, 56125973.53B/s] 81%|████████  | 354734080/440473133 [00:06<00:01, 55735805.98B/s] 82%|████████▏ | 360531968/440473133 [00:06<00:01, 56387952.57B/s] 83%|████████▎ | 366503936/440473133 [00:06<00:01, 57339399.67B/s] 85%|████████▍ | 372245504/440473133 [00:06<00:01, 57339742.93B/s] 86%|████████▌ | 377985024/440473133 [00:06<00:01, 56101424.09B/s] 87%|████████▋ | 383960064/440473133 [00:06<00:00, 57147484.48B/s] 89%|████████▊ | 389965824/440473133 [00:06<00:00, 57990165.67B/s] 90%|████████▉ | 395975680/440473133 [00:06<00:00, 58603404.87B/s] 91%|█████████▏| 401940480/440473133 [00:06<00:00, 58911190.18B/s] 93%|█████████▎| 407882752/440473133 [00:06<00:00, 59060001.89B/s] 94%|█████████▍| 413830144/440473133 [00:07<00:00, 59181987.26B/s] 95%|█████████▌| 419803136/440473133 [00:07<00:00, 59344566.36B/s] 97%|█████████▋| 425789440/440473133 [00:07<00:00, 59497784.10B/s] 98%|█████████▊| 431892480/440473133 [00:07<00:00, 59948557.47B/s] 99%|█████████▉| 437933056/440473133 [00:07<00:00, 60083254.36B/s]100%|██████████| 440473133/440473133 [00:07<00:00, 58534505.05B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
5840896/7094233 [=======================>......] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   2%|▏         | 37/2118 [00:00<00:05, 369.85it/s]Processing text_left with encode:  23%|██▎       | 493/2118 [00:00<00:03, 510.60it/s]Processing text_left with encode:  45%|████▍     | 947/2118 [00:00<00:01, 694.06it/s]Processing text_left with encode:  67%|██████▋   | 1419/2118 [00:00<00:00, 932.72it/s]Processing text_left with encode:  89%|████████▉ | 1894/2118 [00:00<00:00, 1228.97it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3822.54it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 150/18841 [00:00<00:13, 1431.90it/s]Processing text_right with encode:   2%|▏         | 316/18841 [00:00<00:12, 1492.15it/s]Processing text_right with encode:   3%|▎         | 475/18841 [00:00<00:12, 1520.04it/s]Processing text_right with encode:   3%|▎         | 653/18841 [00:00<00:11, 1588.38it/s]Processing text_right with encode:   4%|▍         | 808/18841 [00:00<00:11, 1575.69it/s]Processing text_right with encode:   5%|▌         | 970/18841 [00:00<00:11, 1588.21it/s]Processing text_right with encode:   6%|▌         | 1130/18841 [00:00<00:11, 1588.68it/s]Processing text_right with encode:   7%|▋         | 1292/18841 [00:00<00:10, 1597.63it/s]Processing text_right with encode:   8%|▊         | 1444/18841 [00:00<00:11, 1570.94it/s]Processing text_right with encode:   9%|▊         | 1608/18841 [00:01<00:10, 1585.67it/s]Processing text_right with encode:   9%|▉         | 1775/18841 [00:01<00:10, 1608.65it/s]Processing text_right with encode:  10%|█         | 1934/18841 [00:01<00:10, 1580.09it/s]Processing text_right with encode:  11%|█         | 2101/18841 [00:01<00:10, 1603.80it/s]Processing text_right with encode:  12%|█▏        | 2280/18841 [00:01<00:10, 1655.04it/s]Processing text_right with encode:  13%|█▎        | 2446/18841 [00:01<00:09, 1649.62it/s]Processing text_right with encode:  14%|█▍        | 2617/18841 [00:01<00:09, 1665.82it/s]Processing text_right with encode:  15%|█▍        | 2807/18841 [00:01<00:09, 1729.36it/s]Processing text_right with encode:  16%|█▌        | 2981/18841 [00:01<00:09, 1702.19it/s]Processing text_right with encode:  17%|█▋        | 3152/18841 [00:01<00:09, 1685.09it/s]Processing text_right with encode:  18%|█▊        | 3321/18841 [00:02<00:09, 1674.64it/s]Processing text_right with encode:  19%|█▊        | 3489/18841 [00:02<00:09, 1654.17it/s]Processing text_right with encode:  19%|█▉        | 3655/18841 [00:02<00:09, 1653.23it/s]Processing text_right with encode:  20%|██        | 3821/18841 [00:02<00:09, 1618.63it/s]Processing text_right with encode:  21%|██        | 3999/18841 [00:02<00:08, 1661.27it/s]Processing text_right with encode:  22%|██▏       | 4167/18841 [00:02<00:08, 1662.84it/s]Processing text_right with encode:  23%|██▎       | 4336/18841 [00:02<00:08, 1670.78it/s]Processing text_right with encode:  24%|██▍       | 4504/18841 [00:02<00:08, 1666.80it/s]Processing text_right with encode:  25%|██▍       | 4675/18841 [00:02<00:08, 1679.17it/s]Processing text_right with encode:  26%|██▌       | 4855/18841 [00:02<00:08, 1711.62it/s]Processing text_right with encode:  27%|██▋       | 5031/18841 [00:03<00:08, 1725.72it/s]Processing text_right with encode:  28%|██▊       | 5215/18841 [00:03<00:07, 1758.12it/s]Processing text_right with encode:  29%|██▊       | 5398/18841 [00:03<00:07, 1777.47it/s]Processing text_right with encode:  30%|██▉       | 5577/18841 [00:03<00:07, 1759.20it/s]Processing text_right with encode:  31%|███       | 5754/18841 [00:03<00:07, 1743.39it/s]Processing text_right with encode:  31%|███▏      | 5929/18841 [00:03<00:07, 1723.69it/s]Processing text_right with encode:  32%|███▏      | 6102/18841 [00:03<00:07, 1684.81it/s]Processing text_right with encode:  33%|███▎      | 6271/18841 [00:03<00:07, 1645.19it/s]Processing text_right with encode:  34%|███▍      | 6450/18841 [00:03<00:07, 1683.59it/s]Processing text_right with encode:  35%|███▌      | 6637/18841 [00:03<00:07, 1733.51it/s]Processing text_right with encode:  36%|███▌      | 6812/18841 [00:04<00:07, 1709.46it/s]Processing text_right with encode:  37%|███▋      | 6984/18841 [00:04<00:07, 1623.25it/s]Processing text_right with encode:  38%|███▊      | 7148/18841 [00:04<00:07, 1622.79it/s]Processing text_right with encode:  39%|███▉      | 7322/18841 [00:04<00:06, 1654.04it/s]Processing text_right with encode:  40%|███▉      | 7495/18841 [00:04<00:06, 1674.54it/s]Processing text_right with encode:  41%|████      | 7670/18841 [00:04<00:06, 1695.21it/s]Processing text_right with encode:  42%|████▏     | 7848/18841 [00:04<00:06, 1716.96it/s]Processing text_right with encode:  43%|████▎     | 8021/18841 [00:04<00:06, 1701.15it/s]Processing text_right with encode:  43%|████▎     | 8192/18841 [00:04<00:06, 1702.86it/s]Processing text_right with encode:  44%|████▍     | 8375/18841 [00:05<00:06, 1737.89it/s]Processing text_right with encode:  45%|████▌     | 8550/18841 [00:05<00:06, 1697.80it/s]Processing text_right with encode:  46%|████▋     | 8730/18841 [00:05<00:05, 1727.17it/s]Processing text_right with encode:  47%|████▋     | 8905/18841 [00:05<00:05, 1732.07it/s]Processing text_right with encode:  48%|████▊     | 9079/18841 [00:05<00:05, 1665.33it/s]Processing text_right with encode:  49%|████▉     | 9259/18841 [00:05<00:05, 1701.51it/s]Processing text_right with encode:  50%|█████     | 9430/18841 [00:05<00:05, 1688.02it/s]Processing text_right with encode:  51%|█████     | 9615/18841 [00:05<00:05, 1729.39it/s]Processing text_right with encode:  52%|█████▏    | 9789/18841 [00:05<00:05, 1678.80it/s]Processing text_right with encode:  53%|█████▎    | 9970/18841 [00:05<00:05, 1715.52it/s]Processing text_right with encode:  54%|█████▍    | 10147/18841 [00:06<00:05, 1730.20it/s]Processing text_right with encode:  55%|█████▍    | 10321/18841 [00:06<00:04, 1706.62it/s]Processing text_right with encode:  56%|█████▌    | 10525/18841 [00:06<00:04, 1792.07it/s]Processing text_right with encode:  57%|█████▋    | 10706/18841 [00:06<00:04, 1747.32it/s]Processing text_right with encode:  58%|█████▊    | 10882/18841 [00:06<00:04, 1703.57it/s]Processing text_right with encode:  59%|█████▊    | 11054/18841 [00:06<00:04, 1703.93it/s]Processing text_right with encode:  60%|█████▉    | 11226/18841 [00:06<00:04, 1689.67it/s]Processing text_right with encode:  60%|██████    | 11396/18841 [00:06<00:04, 1661.94it/s]Processing text_right with encode:  61%|██████▏   | 11564/18841 [00:06<00:04, 1666.24it/s]Processing text_right with encode:  62%|██████▏   | 11733/18841 [00:06<00:04, 1672.44it/s]Processing text_right with encode:  63%|██████▎   | 11906/18841 [00:07<00:04, 1688.06it/s]Processing text_right with encode:  64%|██████▍   | 12082/18841 [00:07<00:03, 1705.98it/s]Processing text_right with encode:  65%|██████▌   | 12253/18841 [00:07<00:03, 1697.08it/s]Processing text_right with encode:  66%|██████▌   | 12427/18841 [00:07<00:03, 1708.12it/s]Processing text_right with encode:  67%|██████▋   | 12598/18841 [00:07<00:03, 1672.97it/s]Processing text_right with encode:  68%|██████▊   | 12776/18841 [00:07<00:03, 1703.35it/s]Processing text_right with encode:  69%|██████▊   | 12948/18841 [00:07<00:03, 1706.23it/s]Processing text_right with encode:  70%|██████▉   | 13119/18841 [00:07<00:03, 1701.46it/s]Processing text_right with encode:  71%|███████   | 13292/18841 [00:07<00:03, 1708.24it/s]Processing text_right with encode:  71%|███████▏  | 13463/18841 [00:07<00:03, 1691.48it/s]Processing text_right with encode:  72%|███████▏  | 13645/18841 [00:08<00:03, 1724.42it/s]Processing text_right with encode:  73%|███████▎  | 13822/18841 [00:08<00:02, 1736.91it/s]Processing text_right with encode:  74%|███████▍  | 13996/18841 [00:08<00:02, 1706.41it/s]Processing text_right with encode:  75%|███████▌  | 14167/18841 [00:08<00:02, 1658.15it/s]Processing text_right with encode:  76%|███████▌  | 14334/18841 [00:08<00:02, 1622.34it/s]Processing text_right with encode:  77%|███████▋  | 14507/18841 [00:08<00:02, 1650.33it/s]Processing text_right with encode:  78%|███████▊  | 14693/18841 [00:08<00:02, 1707.18it/s]Processing text_right with encode:  79%|███████▉  | 14868/18841 [00:08<00:02, 1716.25it/s]Processing text_right with encode:  80%|███████▉  | 15041/18841 [00:08<00:02, 1716.47it/s]Processing text_right with encode:  81%|████████  | 15215/18841 [00:09<00:02, 1720.92it/s]Processing text_right with encode:  82%|████████▏ | 15388/18841 [00:09<00:02, 1714.89it/s]Processing text_right with encode:  83%|████████▎ | 15565/18841 [00:09<00:01, 1731.01it/s]Processing text_right with encode:  84%|████████▎ | 15748/18841 [00:09<00:01, 1757.79it/s]Processing text_right with encode:  85%|████████▍ | 15925/18841 [00:09<00:01, 1735.86it/s]Processing text_right with encode:  85%|████████▌ | 16099/18841 [00:09<00:01, 1687.57it/s]Processing text_right with encode:  86%|████████▋ | 16269/18841 [00:09<00:01, 1688.56it/s]Processing text_right with encode:  87%|████████▋ | 16440/18841 [00:09<00:01, 1693.30it/s]Processing text_right with encode:  88%|████████▊ | 16610/18841 [00:09<00:01, 1675.20it/s]Processing text_right with encode:  89%|████████▉ | 16778/18841 [00:09<00:01, 1669.73it/s]Processing text_right with encode:  90%|████████▉ | 16946/18841 [00:10<00:01, 1647.39it/s]Processing text_right with encode:  91%|█████████ | 17119/18841 [00:10<00:01, 1670.87it/s]Processing text_right with encode:  92%|█████████▏| 17287/18841 [00:10<00:00, 1659.84it/s]Processing text_right with encode:  93%|█████████▎| 17456/18841 [00:10<00:00, 1667.33it/s]Processing text_right with encode:  94%|█████████▎| 17629/18841 [00:10<00:00, 1682.49it/s]Processing text_right with encode:  95%|█████████▍| 17806/18841 [00:10<00:00, 1705.52it/s]Processing text_right with encode:  95%|█████████▌| 17987/18841 [00:10<00:00, 1731.73it/s]Processing text_right with encode:  96%|█████████▋| 18174/18841 [00:10<00:00, 1769.44it/s]Processing text_right with encode:  97%|█████████▋| 18352/18841 [00:10<00:00, 1729.02it/s]Processing text_right with encode:  98%|█████████▊| 18539/18841 [00:10<00:00, 1768.92it/s]Processing text_right with encode:  99%|█████████▉| 18717/18841 [00:11<00:00, 1752.05it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:11<00:00, 1690.93it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 550030.08it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 718617.07it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  75%|███████▍  | 473/633 [00:00<00:00, 4728.58it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4659.67it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 179/5961 [00:00<00:03, 1786.68it/s]Processing text_right with encode:   6%|▌         | 351/5961 [00:00<00:03, 1765.84it/s]Processing text_right with encode:   8%|▊         | 501/5961 [00:00<00:03, 1675.95it/s]Processing text_right with encode:  11%|█▏        | 680/5961 [00:00<00:03, 1707.43it/s]Processing text_right with encode:  14%|█▍        | 861/5961 [00:00<00:02, 1736.66it/s]Processing text_right with encode:  17%|█▋        | 1042/5961 [00:00<00:02, 1755.03it/s]Processing text_right with encode:  20%|██        | 1219/5961 [00:00<00:02, 1756.58it/s]Processing text_right with encode:  23%|██▎       | 1399/5961 [00:00<00:02, 1767.83it/s]Processing text_right with encode:  26%|██▋       | 1576/5961 [00:00<00:02, 1766.78it/s]Processing text_right with encode:  29%|██▉       | 1746/5961 [00:01<00:02, 1694.37it/s]Processing text_right with encode:  32%|███▏      | 1926/5961 [00:01<00:02, 1721.92it/s]Processing text_right with encode:  35%|███▌      | 2103/5961 [00:01<00:02, 1735.45it/s]Processing text_right with encode:  38%|███▊      | 2281/5961 [00:01<00:02, 1746.41it/s]Processing text_right with encode:  41%|████      | 2455/5961 [00:01<00:02, 1686.21it/s]Processing text_right with encode:  44%|████▍     | 2638/5961 [00:01<00:01, 1726.45it/s]Processing text_right with encode:  48%|████▊     | 2832/5961 [00:01<00:01, 1784.58it/s]Processing text_right with encode:  51%|█████     | 3011/5961 [00:01<00:01, 1775.87it/s]Processing text_right with encode:  54%|█████▎    | 3197/5961 [00:01<00:01, 1800.10it/s]Processing text_right with encode:  57%|█████▋    | 3378/5961 [00:01<00:01, 1703.32it/s]Processing text_right with encode:  60%|█████▉    | 3553/5961 [00:02<00:01, 1716.08it/s]Processing text_right with encode:  63%|██████▎   | 3726/5961 [00:02<00:01, 1707.30it/s]Processing text_right with encode:  66%|██████▌   | 3908/5961 [00:02<00:01, 1735.51it/s]Processing text_right with encode:  69%|██████▊   | 4093/5961 [00:02<00:01, 1766.20it/s]Processing text_right with encode:  72%|███████▏  | 4271/5961 [00:02<00:00, 1752.25it/s]Processing text_right with encode:  75%|███████▍  | 4447/5961 [00:02<00:00, 1750.48it/s]Processing text_right with encode:  78%|███████▊  | 4627/5961 [00:02<00:00, 1763.86it/s]Processing text_right with encode:  81%|████████  | 4804/5961 [00:02<00:00, 1748.56it/s]Processing text_right with encode:  84%|████████▎ | 4980/5961 [00:02<00:00, 1707.27it/s]Processing text_right with encode:  86%|████████▋ | 5152/5961 [00:02<00:00, 1658.74it/s]Processing text_right with encode:  89%|████████▉ | 5319/5961 [00:03<00:00, 1630.86it/s]Processing text_right with encode:  92%|█████████▏| 5483/5961 [00:03<00:00, 1614.80it/s]Processing text_right with encode:  95%|█████████▍| 5645/5961 [00:03<00:00, 1604.24it/s]Processing text_right with encode:  97%|█████████▋| 5806/5961 [00:03<00:00, 1604.94it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1718.86it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 429402.30it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 743671.81it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:19<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:19<?, ?it/s, loss=0.989]Epoch 1/1:   1%|          | 1/102 [00:19<33:06, 19.67s/it, loss=0.989]Epoch 1/1:   1%|          | 1/102 [00:31<33:06, 19.67s/it, loss=0.989]Epoch 1/1:   1%|          | 1/102 [00:31<33:06, 19.67s/it, loss=0.900]Epoch 1/1:   2%|▏         | 2/102 [00:31<28:37, 17.18s/it, loss=0.900]Epoch 1/1:   2%|▏         | 2/102 [02:16<28:37, 17.18s/it, loss=0.900]Epoch 1/1:   2%|▏         | 2/102 [02:16<28:37, 17.18s/it, loss=0.979]Epoch 1/1:   3%|▎         | 3/102 [02:16<1:12:00, 43.64s/it, loss=0.979]Killed

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   b9c62f5..0cf464a  master     -> origin/master
Updating b9c62f5..0cf464a
Fast-forward
 error_list/20200514/list_log_benchmark_20200514.md |  162 +-
 error_list/20200514/list_log_import_20200514.md    |    2 +-
 error_list/20200514/list_log_jupyter_20200514.md   | 1788 ++++++++++----------
 error_list/20200514/list_log_testall_20200514.md   |  311 ++--
 4 files changed, 1091 insertions(+), 1172 deletions(-)
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
[master 7e99a49] ml_store
 1 file changed, 67 insertions(+)
To github.com:arita37/mlmodels_store.git
   0cf464a..7e99a49  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'dataset': 'torchvision.datasets:MNIST', 'transform_uri': 'mlmodels.preprocess.image.py:torch_transform_mnist', '2nd___transform_uri': '/mnt/hgfs/d/gitdev/mlmodels/mlmodels/preprocess/image.py:torch_transform_mnist', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:01, 159751.44it/s] 74%|███████▍  | 7356416/9912422 [00:00<00:11, 228002.88it/s]9920512it [00:00, 44659052.59it/s]                           
0it [00:00, ?it/s]32768it [00:00, 723416.22it/s]
0it [00:00, ?it/s]  6%|▌         | 98304/1648877 [00:00<00:01, 941358.40it/s]1654784it [00:00, 11847841.82it/s]                         
0it [00:00, ?it/s]8192it [00:00, 250059.96it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
[master dda0471] ml_store
 1 file changed, 84 insertions(+)
To github.com:arita37/mlmodels_store.git
   7e99a49..dda0471  master -> master





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
