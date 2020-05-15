
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
[master 2684e18] ml_store
 1 file changed, 59 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-15-08-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
To github.com:arita37/mlmodels_store.git
   c2a3ade..2684e18  master -> master





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
[master 5072c69] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (fetch first)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally. This is usually caused by another repository pushing
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
From github.com:arita37/mlmodels_store
   2684e18..74d1c56  master     -> origin/master
Merge made by the 'recursive' strategy.
 ...-10_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 628 +++++++++++++++++++++
 1 file changed, 628 insertions(+)
 create mode 100644 log_pullrequest/log_pr_2020-05-15-08-10_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
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
[master d7fbf4a] ml_store
 1 file changed, 58 insertions(+)
To github.com:arita37/mlmodels_store.git
   74d1c56..d7fbf4a  master -> master





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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         4           sequence_max[0][0]               
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
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-15 08:12:37.041203: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 08:12:37.061322: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-05-15 08:12:37.061665: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561b7ae67fb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 08:12:37.061732: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 158
Trainable params: 158
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.4600 - binary_crossentropy: 7.0955500/500 [==============================] - 1s 2ms/sample - loss: 0.5040 - binary_crossentropy: 7.7742 - val_loss: 0.4880 - val_binary_crossentropy: 7.5274

  #### metrics   #################################################### 
{'MSE': 0.496}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         4           sequence_max[0][0]               
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
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
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
Total params: 158
Trainable params: 158
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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 458
Trainable params: 458
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3175 - binary_crossentropy: 0.8519500/500 [==============================] - 1s 2ms/sample - loss: 0.2912 - binary_crossentropy: 0.7898 - val_loss: 0.2873 - val_binary_crossentropy: 0.7784

  #### metrics   #################################################### 
{'MSE': 0.2882746266250316}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 458
Trainable params: 458
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
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
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
Total params: 617
Trainable params: 617
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 3ms/sample - loss: 0.2498 - binary_crossentropy: 0.6928 - val_loss: 0.2500 - val_binary_crossentropy: 0.6931

  #### metrics   #################################################### 
{'MSE': 0.24959739263181993}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
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
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
Total params: 617
Trainable params: 617
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 418
Trainable params: 418
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3625 - binary_crossentropy: 1.0978500/500 [==============================] - 1s 3ms/sample - loss: 0.3306 - binary_crossentropy: 0.9172 - val_loss: 0.2844 - val_binary_crossentropy: 0.8058

  #### metrics   #################################################### 
{'MSE': 0.3044187432146513}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 418
Trainable params: 418
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
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
Total params: 148
Trainable params: 148
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 3s - loss: 0.2542 - binary_crossentropy: 0.7022500/500 [==============================] - 2s 4ms/sample - loss: 0.2647 - binary_crossentropy: 0.7241 - val_loss: 0.2656 - val_binary_crossentropy: 0.7255

  #### metrics   #################################################### 
{'MSE': 0.26468665258403873}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-15 08:14:07.807917: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:14:07.810140: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:14:07.816539: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 08:14:07.827571: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 08:14:07.829425: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 08:14:07.831190: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:14:07.832807: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2473 - val_binary_crossentropy: 0.6876
2020-05-15 08:14:09.261757: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:14:09.263899: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:14:09.269314: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 08:14:09.279810: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-15 08:14:09.281741: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 08:14:09.283443: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:14:09.285067: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24609298247160064}

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
2020-05-15 08:14:35.156579: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:14:35.158231: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:14:35.162684: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 08:14:35.170237: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 08:14:35.171703: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 08:14:35.172908: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:14:35.174055: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2494 - val_binary_crossentropy: 0.6920
2020-05-15 08:14:36.788910: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:14:36.790030: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:14:36.792588: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 08:14:36.798344: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-15 08:14:36.799211: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 08:14:36.799976: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:14:36.800684: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24912819869001743}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-15 08:15:12.777017: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:15:12.781438: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:15:12.794584: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 08:15:12.818276: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 08:15:12.822076: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 08:15:12.825806: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:15:12.830166: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.2752 - binary_crossentropy: 0.7437 - val_loss: 0.2765 - val_binary_crossentropy: 0.7491
2020-05-15 08:15:15.212952: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:15:15.217145: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:15:15.228569: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 08:15:15.251935: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-15 08:15:15.255783: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-15 08:15:15.259415: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-15 08:15:15.262913: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.3311525983764086}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 730
Trainable params: 730
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2926 - binary_crossentropy: 0.7963500/500 [==============================] - 4s 9ms/sample - loss: 0.2977 - binary_crossentropy: 0.9419 - val_loss: 0.3105 - val_binary_crossentropy: 0.9437

  #### metrics   #################################################### 
{'MSE': 0.30329412725331245}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         16          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         18          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
Total params: 269
Trainable params: 269
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2671 - binary_crossentropy: 0.7284500/500 [==============================] - 5s 10ms/sample - loss: 0.2575 - binary_crossentropy: 0.7354 - val_loss: 0.2559 - val_binary_crossentropy: 0.7051

  #### metrics   #################################################### 
{'MSE': 0.25641046473540824}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         16          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         18          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
Total params: 269
Trainable params: 269
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
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
Total params: 1,974
Trainable params: 1,974
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2853 - binary_crossentropy: 1.0273500/500 [==============================] - 5s 9ms/sample - loss: 0.2573 - binary_crossentropy: 0.8113 - val_loss: 0.2564 - val_binary_crossentropy: 0.7586

  #### metrics   #################################################### 
{'MSE': 0.25587245329728386}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
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
Total params: 1,974
Trainable params: 1,974
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
regionsequence_sum (InputLayer) [(None, 5)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 2)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 2)]          0                                            
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
region_10sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         6           regionsequence_max[0][0]         
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
Total params: 168
Trainable params: 168
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2576 - binary_crossentropy: 0.8386500/500 [==============================] - 6s 12ms/sample - loss: 0.2574 - binary_crossentropy: 0.8651 - val_loss: 0.2578 - val_binary_crossentropy: 0.9176

  #### metrics   #################################################### 
{'MSE': 0.2573121617322976}

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
regionsequence_sum (InputLayer) [(None, 5)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 2)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 2)]          0                                            
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
region_10sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         6           regionsequence_max[0][0]         
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
Total params: 168
Trainable params: 168
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 1,352
Trainable params: 1,352
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2622 - binary_crossentropy: 0.7228500/500 [==============================] - 6s 12ms/sample - loss: 0.2861 - binary_crossentropy: 0.8282 - val_loss: 0.2735 - val_binary_crossentropy: 0.7715

  #### metrics   #################################################### 
{'MSE': 0.277483537739547}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 1,352
Trainable params: 1,352
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         20          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         16          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_11[0][0]                    
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
100/500 [=====>........................] - ETA: 8s - loss: 0.2748 - binary_crossentropy: 0.7493500/500 [==============================] - 7s 14ms/sample - loss: 0.2673 - binary_crossentropy: 0.7311 - val_loss: 0.2648 - val_binary_crossentropy: 0.7246

  #### metrics   #################################################### 
{'MSE': 0.26504518683150496}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         20          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         16          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_11[0][0]                    
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         20          sequence_max[0][0]               
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
Total params: 417
Trainable params: 417
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2492 - binary_crossentropy: 0.6916500/500 [==============================] - 6s 13ms/sample - loss: 0.2525 - binary_crossentropy: 0.6984 - val_loss: 0.2501 - val_binary_crossentropy: 0.6933

  #### metrics   #################################################### 
{'MSE': 0.2500732061297755}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         20          sequence_max[0][0]               
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
Total params: 417
Trainable params: 417
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
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
Total params: 2,054
Trainable params: 2,054
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 7s 14ms/sample - loss: 0.2499 - binary_crossentropy: 0.6930 - val_loss: 0.2494 - val_binary_crossentropy: 0.6916

  #### metrics   #################################################### 
{'MSE': 0.24937276585259213}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
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
Total params: 2,054
Trainable params: 2,054
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
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
Total params: 286
Trainable params: 286
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.3225 - binary_crossentropy: 0.8873500/500 [==============================] - 7s 14ms/sample - loss: 0.3002 - binary_crossentropy: 0.8479 - val_loss: 0.2992 - val_binary_crossentropy: 0.8120

  #### metrics   #################################################### 
{'MSE': 0.29864384585613324}

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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
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
Total params: 286
Trainable params: 286
Non-trainable params: 0
__________________________________________________________________________________________________

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '192.30.255.113' to the list of known hosts.
From github.com:arita37/mlmodels_store
   d7fbf4a..537fd3b  master     -> origin/master
Updating d7fbf4a..537fd3b
Fast-forward
 error_list/20200515/list_log_benchmark_20200515.md | 180 +++++++-------
 .../20200515/list_log_dataloader_20200515.md       |   2 +-
 error_list/20200515/list_log_json_20200515.md      | 276 ++++++++++-----------
 .../20200515/list_log_pullrequest_20200515.md      |   2 +-
 4 files changed, 225 insertions(+), 235 deletions(-)
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
[master 9741c6a] ml_store
 1 file changed, 5670 insertions(+)
To github.com:arita37/mlmodels_store.git
   537fd3b..9741c6a  master -> master





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
[master dbe4544] ml_store
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   9741c6a..dbe4544  master -> master





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
[master ec7e2b2] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   dbe4544..ec7e2b2  master -> master





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
[master dfb5dac] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   ec7e2b2..dfb5dac  master -> master





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

2020-05-15 08:29:00.928045: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 08:29:00.932858: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-05-15 08:29:00.933593: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55db35def570 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 08:29:00.933660: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3860
256/354 [====================>.........] - ETA: 3s - loss: 1.2752
354/354 [==============================] - 14s 40ms/step - loss: 1.1983 - val_loss: 1.9306

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
[master cbb8e9e] ml_store
 1 file changed, 149 insertions(+)
To github.com:arita37/mlmodels_store.git
   dfb5dac..cbb8e9e  master -> master





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
[master 988b6ac] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   cbb8e9e..988b6ac  master -> master





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
[master 8b89fae] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   988b6ac..8b89fae  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 47s
   57344/17464789 [..............................] - ETA: 40s
  106496/17464789 [..............................] - ETA: 33s
  212992/17464789 [..............................] - ETA: 21s
  417792/17464789 [..............................] - ETA: 13s
  860160/17464789 [>.............................] - ETA: 7s 
 1728512/17464789 [=>............................] - ETA: 4s
 3448832/17464789 [====>.........................] - ETA: 2s
 5775360/17464789 [========>.....................] - ETA: 1s
 7872512/17464789 [============>.................] - ETA: 0s
10035200/17464789 [================>.............] - ETA: 0s
12337152/17464789 [====================>.........] - ETA: 0s
14680064/17464789 [========================>.....] - ETA: 0s
16850944/17464789 [===========================>..] - ETA: 0s
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
2020-05-15 08:30:06.101348: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 08:30:06.106947: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-05-15 08:30:06.107132: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fe81c501d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 08:30:06.107153: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5900 - accuracy: 0.5050
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5823 - accuracy: 0.5055 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6206 - accuracy: 0.5030
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6283 - accuracy: 0.5025
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6697 - accuracy: 0.4998
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6308 - accuracy: 0.5023
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6403 - accuracy: 0.5017
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5880 - accuracy: 0.5051
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5831 - accuracy: 0.5054
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5976 - accuracy: 0.5045
11000/25000 [============>.................] - ETA: 4s - loss: 7.6387 - accuracy: 0.5018
12000/25000 [=============>................] - ETA: 4s - loss: 7.6590 - accuracy: 0.5005
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6749 - accuracy: 0.4995
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6831 - accuracy: 0.4989
15000/25000 [=================>............] - ETA: 3s - loss: 7.6738 - accuracy: 0.4995
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6887 - accuracy: 0.4986
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6919 - accuracy: 0.4984
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6930 - accuracy: 0.4983
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6787 - accuracy: 0.4992
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6536 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6834 - accuracy: 0.4989
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6733 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 9s 366us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f6670a942b0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f6670a942b0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.7433 - accuracy: 0.4950
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7126 - accuracy: 0.4970 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7280 - accuracy: 0.4960
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6398 - accuracy: 0.5017
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6942 - accuracy: 0.4982
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6622 - accuracy: 0.5003
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6475 - accuracy: 0.5013
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6223 - accuracy: 0.5029
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6467 - accuracy: 0.5013
11000/25000 [============>.................] - ETA: 4s - loss: 7.6527 - accuracy: 0.5009
12000/25000 [=============>................] - ETA: 4s - loss: 7.6756 - accuracy: 0.4994
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6579 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 3s - loss: 7.6370 - accuracy: 0.5019
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6465 - accuracy: 0.5013
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6621 - accuracy: 0.5003
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6602 - accuracy: 0.5004
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6536 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6506 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
25000/25000 [==============================] - 9s 368us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 13s - loss: 8.0960 - accuracy: 0.4720
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7816 - accuracy: 0.4925
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7484 - accuracy: 0.4947 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7126 - accuracy: 0.4970
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6605 - accuracy: 0.5004
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6922 - accuracy: 0.4983
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7126 - accuracy: 0.4970
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7433 - accuracy: 0.4950
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7007 - accuracy: 0.4978
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7111 - accuracy: 0.4971
11000/25000 [============>.................] - ETA: 4s - loss: 7.7112 - accuracy: 0.4971
12000/25000 [=============>................] - ETA: 3s - loss: 7.6973 - accuracy: 0.4980
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6985 - accuracy: 0.4979
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7170 - accuracy: 0.4967
15000/25000 [=================>............] - ETA: 3s - loss: 7.7280 - accuracy: 0.4960
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6992 - accuracy: 0.4979
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6838 - accuracy: 0.4989
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6947 - accuracy: 0.4982
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6755 - accuracy: 0.4994
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6643 - accuracy: 0.5002
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6608 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6708 - accuracy: 0.4997
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6737 - accuracy: 0.4995
25000/25000 [==============================] - 9s 363us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   8b89fae..bb24dc9  master     -> origin/master
Updating 8b89fae..bb24dc9
Fast-forward
 error_list/20200515/list_log_benchmark_20200515.md | 180 +++++++++++----------
 .../20200515/list_log_dataloader_20200515.md       |   2 +-
 2 files changed, 96 insertions(+), 86 deletions(-)
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
[master b6d31c3] ml_store
 1 file changed, 334 insertions(+)
To github.com:arita37/mlmodels_store.git
   bb24dc9..b6d31c3  master -> master





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

13/13 [==============================] - 2s 122ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master a8c5ebe] ml_store
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   b6d31c3..a8c5ebe  master -> master





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
   57344/11490434 [..............................] - ETA: 26s
   90112/11490434 [..............................] - ETA: 25s
  180224/11490434 [..............................] - ETA: 16s
  335872/11490434 [..............................] - ETA: 11s
  663552/11490434 [>.............................] - ETA: 6s 
 1327104/11490434 [==>...........................] - ETA: 3s
 2662400/11490434 [=====>........................] - ETA: 1s
 5316608/11490434 [============>.................] - ETA: 0s
 8298496/11490434 [====================>.........] - ETA: 0s
11345920/11490434 [============================>.] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:44 - loss: 2.3024 - categorical_accuracy: 0.0938
   64/60000 [..............................] - ETA: 4:48 - loss: 2.2768 - categorical_accuracy: 0.0938
   96/60000 [..............................] - ETA: 3:48 - loss: 2.2593 - categorical_accuracy: 0.1042
  128/60000 [..............................] - ETA: 3:15 - loss: 2.2615 - categorical_accuracy: 0.1406
  160/60000 [..............................] - ETA: 2:58 - loss: 2.2186 - categorical_accuracy: 0.1937
  192/60000 [..............................] - ETA: 2:47 - loss: 2.1559 - categorical_accuracy: 0.2500
  224/60000 [..............................] - ETA: 2:39 - loss: 2.1338 - categorical_accuracy: 0.2589
  256/60000 [..............................] - ETA: 2:32 - loss: 2.0648 - categorical_accuracy: 0.2891
  288/60000 [..............................] - ETA: 2:27 - loss: 2.0408 - categorical_accuracy: 0.2882
  320/60000 [..............................] - ETA: 2:22 - loss: 2.0304 - categorical_accuracy: 0.2812
  352/60000 [..............................] - ETA: 2:18 - loss: 1.9866 - categorical_accuracy: 0.3040
  384/60000 [..............................] - ETA: 2:15 - loss: 1.9543 - categorical_accuracy: 0.3255
  416/60000 [..............................] - ETA: 2:12 - loss: 1.9130 - categorical_accuracy: 0.3486
  448/60000 [..............................] - ETA: 2:10 - loss: 1.8928 - categorical_accuracy: 0.3594
  480/60000 [..............................] - ETA: 2:09 - loss: 1.8772 - categorical_accuracy: 0.3583
  512/60000 [..............................] - ETA: 2:07 - loss: 1.8609 - categorical_accuracy: 0.3633
  544/60000 [..............................] - ETA: 2:06 - loss: 1.8325 - categorical_accuracy: 0.3750
  576/60000 [..............................] - ETA: 2:04 - loss: 1.8002 - categorical_accuracy: 0.3872
  608/60000 [..............................] - ETA: 2:04 - loss: 1.7559 - categorical_accuracy: 0.3997
  640/60000 [..............................] - ETA: 2:03 - loss: 1.7282 - categorical_accuracy: 0.4047
  672/60000 [..............................] - ETA: 2:02 - loss: 1.7007 - categorical_accuracy: 0.4122
  704/60000 [..............................] - ETA: 2:01 - loss: 1.6659 - categorical_accuracy: 0.4233
  736/60000 [..............................] - ETA: 2:01 - loss: 1.6255 - categorical_accuracy: 0.4375
  768/60000 [..............................] - ETA: 2:00 - loss: 1.6131 - categorical_accuracy: 0.4427
  800/60000 [..............................] - ETA: 1:59 - loss: 1.5996 - categorical_accuracy: 0.4487
  832/60000 [..............................] - ETA: 1:59 - loss: 1.5821 - categorical_accuracy: 0.4567
  864/60000 [..............................] - ETA: 1:58 - loss: 1.5524 - categorical_accuracy: 0.4664
  896/60000 [..............................] - ETA: 1:58 - loss: 1.5306 - categorical_accuracy: 0.4732
  928/60000 [..............................] - ETA: 1:57 - loss: 1.5126 - categorical_accuracy: 0.4817
  960/60000 [..............................] - ETA: 1:56 - loss: 1.4950 - categorical_accuracy: 0.4896
  992/60000 [..............................] - ETA: 1:56 - loss: 1.4777 - categorical_accuracy: 0.4980
 1024/60000 [..............................] - ETA: 1:56 - loss: 1.4532 - categorical_accuracy: 0.5078
 1056/60000 [..............................] - ETA: 1:55 - loss: 1.4413 - categorical_accuracy: 0.5123
 1088/60000 [..............................] - ETA: 1:55 - loss: 1.4219 - categorical_accuracy: 0.5165
 1120/60000 [..............................] - ETA: 1:54 - loss: 1.4066 - categorical_accuracy: 0.5214
 1152/60000 [..............................] - ETA: 1:54 - loss: 1.3915 - categorical_accuracy: 0.5269
 1184/60000 [..............................] - ETA: 1:53 - loss: 1.3808 - categorical_accuracy: 0.5321
 1216/60000 [..............................] - ETA: 1:53 - loss: 1.3655 - categorical_accuracy: 0.5378
 1248/60000 [..............................] - ETA: 1:52 - loss: 1.3412 - categorical_accuracy: 0.5465
 1280/60000 [..............................] - ETA: 1:52 - loss: 1.3248 - categorical_accuracy: 0.5516
 1312/60000 [..............................] - ETA: 1:51 - loss: 1.3166 - categorical_accuracy: 0.5541
 1344/60000 [..............................] - ETA: 1:51 - loss: 1.3033 - categorical_accuracy: 0.5573
 1376/60000 [..............................] - ETA: 1:51 - loss: 1.2966 - categorical_accuracy: 0.5603
 1408/60000 [..............................] - ETA: 1:51 - loss: 1.2802 - categorical_accuracy: 0.5668
 1440/60000 [..............................] - ETA: 1:50 - loss: 1.2700 - categorical_accuracy: 0.5694
 1472/60000 [..............................] - ETA: 1:50 - loss: 1.2632 - categorical_accuracy: 0.5720
 1504/60000 [..............................] - ETA: 1:50 - loss: 1.2495 - categorical_accuracy: 0.5765
 1536/60000 [..............................] - ETA: 1:50 - loss: 1.2304 - categorical_accuracy: 0.5827
 1568/60000 [..............................] - ETA: 1:49 - loss: 1.2260 - categorical_accuracy: 0.5874
 1600/60000 [..............................] - ETA: 1:49 - loss: 1.2117 - categorical_accuracy: 0.5919
 1632/60000 [..............................] - ETA: 1:49 - loss: 1.2002 - categorical_accuracy: 0.5956
 1664/60000 [..............................] - ETA: 1:49 - loss: 1.1897 - categorical_accuracy: 0.5986
 1696/60000 [..............................] - ETA: 1:48 - loss: 1.1744 - categorical_accuracy: 0.6044
 1728/60000 [..............................] - ETA: 1:48 - loss: 1.1635 - categorical_accuracy: 0.6082
 1760/60000 [..............................] - ETA: 1:48 - loss: 1.1523 - categorical_accuracy: 0.6125
 1792/60000 [..............................] - ETA: 1:48 - loss: 1.1430 - categorical_accuracy: 0.6166
 1824/60000 [..............................] - ETA: 1:48 - loss: 1.1316 - categorical_accuracy: 0.6206
 1856/60000 [..............................] - ETA: 1:48 - loss: 1.1199 - categorical_accuracy: 0.6239
 1888/60000 [..............................] - ETA: 1:47 - loss: 1.1081 - categorical_accuracy: 0.6276
 1920/60000 [..............................] - ETA: 1:47 - loss: 1.0956 - categorical_accuracy: 0.6313
 1952/60000 [..............................] - ETA: 1:47 - loss: 1.0904 - categorical_accuracy: 0.6342
 1984/60000 [..............................] - ETA: 1:47 - loss: 1.0814 - categorical_accuracy: 0.6381
 2016/60000 [>.............................] - ETA: 1:47 - loss: 1.0740 - categorical_accuracy: 0.6399
 2048/60000 [>.............................] - ETA: 1:47 - loss: 1.0657 - categorical_accuracy: 0.6431
 2080/60000 [>.............................] - ETA: 1:47 - loss: 1.0570 - categorical_accuracy: 0.6462
 2112/60000 [>.............................] - ETA: 1:46 - loss: 1.0504 - categorical_accuracy: 0.6482
 2144/60000 [>.............................] - ETA: 1:46 - loss: 1.0470 - categorical_accuracy: 0.6493
 2176/60000 [>.............................] - ETA: 1:46 - loss: 1.0417 - categorical_accuracy: 0.6526
 2208/60000 [>.............................] - ETA: 1:46 - loss: 1.0330 - categorical_accuracy: 0.6562
 2240/60000 [>.............................] - ETA: 1:46 - loss: 1.0252 - categorical_accuracy: 0.6594
 2272/60000 [>.............................] - ETA: 1:46 - loss: 1.0168 - categorical_accuracy: 0.6629
 2304/60000 [>.............................] - ETA: 1:46 - loss: 1.0087 - categorical_accuracy: 0.6654
 2336/60000 [>.............................] - ETA: 1:45 - loss: 0.9995 - categorical_accuracy: 0.6687
 2368/60000 [>.............................] - ETA: 1:45 - loss: 0.9908 - categorical_accuracy: 0.6719
 2400/60000 [>.............................] - ETA: 1:45 - loss: 0.9820 - categorical_accuracy: 0.6750
 2432/60000 [>.............................] - ETA: 1:45 - loss: 0.9768 - categorical_accuracy: 0.6760
 2464/60000 [>.............................] - ETA: 1:45 - loss: 0.9676 - categorical_accuracy: 0.6798
 2496/60000 [>.............................] - ETA: 1:45 - loss: 0.9585 - categorical_accuracy: 0.6827
 2528/60000 [>.............................] - ETA: 1:44 - loss: 0.9549 - categorical_accuracy: 0.6847
 2560/60000 [>.............................] - ETA: 1:44 - loss: 0.9508 - categorical_accuracy: 0.6859
 2592/60000 [>.............................] - ETA: 1:44 - loss: 0.9443 - categorical_accuracy: 0.6883
 2624/60000 [>.............................] - ETA: 1:44 - loss: 0.9399 - categorical_accuracy: 0.6894
 2656/60000 [>.............................] - ETA: 1:44 - loss: 0.9335 - categorical_accuracy: 0.6920
 2688/60000 [>.............................] - ETA: 1:44 - loss: 0.9291 - categorical_accuracy: 0.6938
 2720/60000 [>.............................] - ETA: 1:44 - loss: 0.9206 - categorical_accuracy: 0.6967
 2752/60000 [>.............................] - ETA: 1:43 - loss: 0.9159 - categorical_accuracy: 0.6980
 2784/60000 [>.............................] - ETA: 1:43 - loss: 0.9092 - categorical_accuracy: 0.7004
 2816/60000 [>.............................] - ETA: 1:43 - loss: 0.9046 - categorical_accuracy: 0.7021
 2848/60000 [>.............................] - ETA: 1:43 - loss: 0.8979 - categorical_accuracy: 0.7044
 2880/60000 [>.............................] - ETA: 1:43 - loss: 0.8965 - categorical_accuracy: 0.7052
 2912/60000 [>.............................] - ETA: 1:43 - loss: 0.8911 - categorical_accuracy: 0.7071
 2944/60000 [>.............................] - ETA: 1:43 - loss: 0.8862 - categorical_accuracy: 0.7086
 3008/60000 [>.............................] - ETA: 1:42 - loss: 0.8788 - categorical_accuracy: 0.7108
 3040/60000 [>.............................] - ETA: 1:42 - loss: 0.8727 - categorical_accuracy: 0.7128
 3072/60000 [>.............................] - ETA: 1:42 - loss: 0.8689 - categorical_accuracy: 0.7142
 3104/60000 [>.............................] - ETA: 1:42 - loss: 0.8627 - categorical_accuracy: 0.7165
 3136/60000 [>.............................] - ETA: 1:42 - loss: 0.8587 - categorical_accuracy: 0.7178
 3168/60000 [>.............................] - ETA: 1:42 - loss: 0.8529 - categorical_accuracy: 0.7194
 3200/60000 [>.............................] - ETA: 1:42 - loss: 0.8469 - categorical_accuracy: 0.7216
 3232/60000 [>.............................] - ETA: 1:42 - loss: 0.8408 - categorical_accuracy: 0.7237
 3264/60000 [>.............................] - ETA: 1:41 - loss: 0.8342 - categorical_accuracy: 0.7255
 3328/60000 [>.............................] - ETA: 1:41 - loss: 0.8283 - categorical_accuracy: 0.7281
 3360/60000 [>.............................] - ETA: 1:41 - loss: 0.8242 - categorical_accuracy: 0.7295
 3392/60000 [>.............................] - ETA: 1:41 - loss: 0.8179 - categorical_accuracy: 0.7317
 3424/60000 [>.............................] - ETA: 1:41 - loss: 0.8141 - categorical_accuracy: 0.7331
 3456/60000 [>.............................] - ETA: 1:41 - loss: 0.8087 - categorical_accuracy: 0.7350
 3488/60000 [>.............................] - ETA: 1:41 - loss: 0.8023 - categorical_accuracy: 0.7371
 3520/60000 [>.............................] - ETA: 1:41 - loss: 0.7977 - categorical_accuracy: 0.7384
 3552/60000 [>.............................] - ETA: 1:41 - loss: 0.7945 - categorical_accuracy: 0.7399
 3584/60000 [>.............................] - ETA: 1:41 - loss: 0.7912 - categorical_accuracy: 0.7411
 3616/60000 [>.............................] - ETA: 1:40 - loss: 0.7868 - categorical_accuracy: 0.7425
 3648/60000 [>.............................] - ETA: 1:40 - loss: 0.7844 - categorical_accuracy: 0.7434
 3680/60000 [>.............................] - ETA: 1:40 - loss: 0.7818 - categorical_accuracy: 0.7443
 3712/60000 [>.............................] - ETA: 1:40 - loss: 0.7799 - categorical_accuracy: 0.7454
 3744/60000 [>.............................] - ETA: 1:40 - loss: 0.7767 - categorical_accuracy: 0.7463
 3776/60000 [>.............................] - ETA: 1:40 - loss: 0.7717 - categorical_accuracy: 0.7479
 3808/60000 [>.............................] - ETA: 1:40 - loss: 0.7687 - categorical_accuracy: 0.7487
 3840/60000 [>.............................] - ETA: 1:40 - loss: 0.7644 - categorical_accuracy: 0.7500
 3872/60000 [>.............................] - ETA: 1:40 - loss: 0.7603 - categorical_accuracy: 0.7515
 3904/60000 [>.............................] - ETA: 1:40 - loss: 0.7575 - categorical_accuracy: 0.7520
 3936/60000 [>.............................] - ETA: 1:39 - loss: 0.7539 - categorical_accuracy: 0.7533
 3968/60000 [>.............................] - ETA: 1:39 - loss: 0.7492 - categorical_accuracy: 0.7553
 4000/60000 [=>............................] - ETA: 1:39 - loss: 0.7456 - categorical_accuracy: 0.7567
 4032/60000 [=>............................] - ETA: 1:39 - loss: 0.7414 - categorical_accuracy: 0.7584
 4064/60000 [=>............................] - ETA: 1:39 - loss: 0.7379 - categorical_accuracy: 0.7598
 4096/60000 [=>............................] - ETA: 1:39 - loss: 0.7379 - categorical_accuracy: 0.7607
 4128/60000 [=>............................] - ETA: 1:39 - loss: 0.7354 - categorical_accuracy: 0.7619
 4160/60000 [=>............................] - ETA: 1:39 - loss: 0.7354 - categorical_accuracy: 0.7625
 4192/60000 [=>............................] - ETA: 1:39 - loss: 0.7328 - categorical_accuracy: 0.7629
 4224/60000 [=>............................] - ETA: 1:39 - loss: 0.7308 - categorical_accuracy: 0.7635
 4256/60000 [=>............................] - ETA: 1:38 - loss: 0.7280 - categorical_accuracy: 0.7646
 4288/60000 [=>............................] - ETA: 1:38 - loss: 0.7246 - categorical_accuracy: 0.7656
 4320/60000 [=>............................] - ETA: 1:38 - loss: 0.7206 - categorical_accuracy: 0.7671
 4352/60000 [=>............................] - ETA: 1:38 - loss: 0.7175 - categorical_accuracy: 0.7682
 4384/60000 [=>............................] - ETA: 1:38 - loss: 0.7137 - categorical_accuracy: 0.7692
 4416/60000 [=>............................] - ETA: 1:38 - loss: 0.7105 - categorical_accuracy: 0.7702
 4448/60000 [=>............................] - ETA: 1:38 - loss: 0.7057 - categorical_accuracy: 0.7718
 4480/60000 [=>............................] - ETA: 1:38 - loss: 0.7019 - categorical_accuracy: 0.7732
 4512/60000 [=>............................] - ETA: 1:38 - loss: 0.6980 - categorical_accuracy: 0.7744
 4544/60000 [=>............................] - ETA: 1:38 - loss: 0.6972 - categorical_accuracy: 0.7749
 4576/60000 [=>............................] - ETA: 1:38 - loss: 0.6946 - categorical_accuracy: 0.7758
 4608/60000 [=>............................] - ETA: 1:38 - loss: 0.6921 - categorical_accuracy: 0.7765
 4640/60000 [=>............................] - ETA: 1:38 - loss: 0.6905 - categorical_accuracy: 0.7769
 4672/60000 [=>............................] - ETA: 1:37 - loss: 0.6872 - categorical_accuracy: 0.7780
 4704/60000 [=>............................] - ETA: 1:37 - loss: 0.6848 - categorical_accuracy: 0.7787
 4736/60000 [=>............................] - ETA: 1:37 - loss: 0.6810 - categorical_accuracy: 0.7798
 4768/60000 [=>............................] - ETA: 1:37 - loss: 0.6778 - categorical_accuracy: 0.7808
 4800/60000 [=>............................] - ETA: 1:37 - loss: 0.6750 - categorical_accuracy: 0.7817
 4832/60000 [=>............................] - ETA: 1:37 - loss: 0.6711 - categorical_accuracy: 0.7827
 4864/60000 [=>............................] - ETA: 1:37 - loss: 0.6684 - categorical_accuracy: 0.7837
 4896/60000 [=>............................] - ETA: 1:37 - loss: 0.6658 - categorical_accuracy: 0.7845
 4928/60000 [=>............................] - ETA: 1:37 - loss: 0.6628 - categorical_accuracy: 0.7853
 4960/60000 [=>............................] - ETA: 1:37 - loss: 0.6596 - categorical_accuracy: 0.7867
 4992/60000 [=>............................] - ETA: 1:37 - loss: 0.6571 - categorical_accuracy: 0.7875
 5024/60000 [=>............................] - ETA: 1:37 - loss: 0.6559 - categorical_accuracy: 0.7876
 5056/60000 [=>............................] - ETA: 1:37 - loss: 0.6524 - categorical_accuracy: 0.7886
 5088/60000 [=>............................] - ETA: 1:36 - loss: 0.6515 - categorical_accuracy: 0.7889
 5120/60000 [=>............................] - ETA: 1:36 - loss: 0.6501 - categorical_accuracy: 0.7893
 5152/60000 [=>............................] - ETA: 1:36 - loss: 0.6473 - categorical_accuracy: 0.7902
 5216/60000 [=>............................] - ETA: 1:36 - loss: 0.6431 - categorical_accuracy: 0.7918
 5248/60000 [=>............................] - ETA: 1:36 - loss: 0.6409 - categorical_accuracy: 0.7925
 5280/60000 [=>............................] - ETA: 1:36 - loss: 0.6399 - categorical_accuracy: 0.7930
 5312/60000 [=>............................] - ETA: 1:36 - loss: 0.6398 - categorical_accuracy: 0.7935
 5344/60000 [=>............................] - ETA: 1:36 - loss: 0.6374 - categorical_accuracy: 0.7943
 5376/60000 [=>............................] - ETA: 1:36 - loss: 0.6351 - categorical_accuracy: 0.7950
 5408/60000 [=>............................] - ETA: 1:36 - loss: 0.6331 - categorical_accuracy: 0.7957
 5440/60000 [=>............................] - ETA: 1:36 - loss: 0.6308 - categorical_accuracy: 0.7965
 5472/60000 [=>............................] - ETA: 1:36 - loss: 0.6296 - categorical_accuracy: 0.7970
 5504/60000 [=>............................] - ETA: 1:36 - loss: 0.6279 - categorical_accuracy: 0.7978
 5536/60000 [=>............................] - ETA: 1:36 - loss: 0.6279 - categorical_accuracy: 0.7982
 5568/60000 [=>............................] - ETA: 1:35 - loss: 0.6256 - categorical_accuracy: 0.7990
 5600/60000 [=>............................] - ETA: 1:35 - loss: 0.6249 - categorical_accuracy: 0.7995
 5632/60000 [=>............................] - ETA: 1:35 - loss: 0.6227 - categorical_accuracy: 0.7999
 5664/60000 [=>............................] - ETA: 1:35 - loss: 0.6208 - categorical_accuracy: 0.8005
 5696/60000 [=>............................] - ETA: 1:35 - loss: 0.6183 - categorical_accuracy: 0.8011
 5728/60000 [=>............................] - ETA: 1:35 - loss: 0.6156 - categorical_accuracy: 0.8019
 5760/60000 [=>............................] - ETA: 1:35 - loss: 0.6129 - categorical_accuracy: 0.8030
 5792/60000 [=>............................] - ETA: 1:35 - loss: 0.6110 - categorical_accuracy: 0.8033
 5824/60000 [=>............................] - ETA: 1:35 - loss: 0.6091 - categorical_accuracy: 0.8037
 5856/60000 [=>............................] - ETA: 1:35 - loss: 0.6062 - categorical_accuracy: 0.8048
 5888/60000 [=>............................] - ETA: 1:35 - loss: 0.6049 - categorical_accuracy: 0.8055
 5920/60000 [=>............................] - ETA: 1:35 - loss: 0.6039 - categorical_accuracy: 0.8056
 5952/60000 [=>............................] - ETA: 1:35 - loss: 0.6018 - categorical_accuracy: 0.8065
 5984/60000 [=>............................] - ETA: 1:35 - loss: 0.6009 - categorical_accuracy: 0.8072
 6016/60000 [==>...........................] - ETA: 1:34 - loss: 0.5988 - categorical_accuracy: 0.8080
 6048/60000 [==>...........................] - ETA: 1:34 - loss: 0.5982 - categorical_accuracy: 0.8080
 6080/60000 [==>...........................] - ETA: 1:34 - loss: 0.5979 - categorical_accuracy: 0.8079
 6112/60000 [==>...........................] - ETA: 1:34 - loss: 0.5967 - categorical_accuracy: 0.8084
 6144/60000 [==>...........................] - ETA: 1:34 - loss: 0.5957 - categorical_accuracy: 0.8089
 6176/60000 [==>...........................] - ETA: 1:34 - loss: 0.5938 - categorical_accuracy: 0.8096
 6208/60000 [==>...........................] - ETA: 1:34 - loss: 0.5925 - categorical_accuracy: 0.8098
 6240/60000 [==>...........................] - ETA: 1:34 - loss: 0.5913 - categorical_accuracy: 0.8104
 6272/60000 [==>...........................] - ETA: 1:34 - loss: 0.5897 - categorical_accuracy: 0.8111
 6304/60000 [==>...........................] - ETA: 1:34 - loss: 0.5880 - categorical_accuracy: 0.8117
 6336/60000 [==>...........................] - ETA: 1:34 - loss: 0.5860 - categorical_accuracy: 0.8125
 6368/60000 [==>...........................] - ETA: 1:34 - loss: 0.5847 - categorical_accuracy: 0.8130
 6400/60000 [==>...........................] - ETA: 1:34 - loss: 0.5849 - categorical_accuracy: 0.8131
 6432/60000 [==>...........................] - ETA: 1:34 - loss: 0.5833 - categorical_accuracy: 0.8136
 6464/60000 [==>...........................] - ETA: 1:34 - loss: 0.5812 - categorical_accuracy: 0.8144
 6496/60000 [==>...........................] - ETA: 1:33 - loss: 0.5793 - categorical_accuracy: 0.8150
 6528/60000 [==>...........................] - ETA: 1:33 - loss: 0.5777 - categorical_accuracy: 0.8151
 6560/60000 [==>...........................] - ETA: 1:33 - loss: 0.5758 - categorical_accuracy: 0.8157
 6592/60000 [==>...........................] - ETA: 1:33 - loss: 0.5746 - categorical_accuracy: 0.8161
 6624/60000 [==>...........................] - ETA: 1:33 - loss: 0.5732 - categorical_accuracy: 0.8164
 6656/60000 [==>...........................] - ETA: 1:33 - loss: 0.5714 - categorical_accuracy: 0.8170
 6688/60000 [==>...........................] - ETA: 1:33 - loss: 0.5704 - categorical_accuracy: 0.8174
 6720/60000 [==>...........................] - ETA: 1:33 - loss: 0.5685 - categorical_accuracy: 0.8182
 6752/60000 [==>...........................] - ETA: 1:33 - loss: 0.5668 - categorical_accuracy: 0.8186
 6784/60000 [==>...........................] - ETA: 1:33 - loss: 0.5648 - categorical_accuracy: 0.8191
 6816/60000 [==>...........................] - ETA: 1:33 - loss: 0.5630 - categorical_accuracy: 0.8198
 6848/60000 [==>...........................] - ETA: 1:33 - loss: 0.5622 - categorical_accuracy: 0.8202
 6880/60000 [==>...........................] - ETA: 1:33 - loss: 0.5606 - categorical_accuracy: 0.8208
 6912/60000 [==>...........................] - ETA: 1:33 - loss: 0.5591 - categorical_accuracy: 0.8210
 6944/60000 [==>...........................] - ETA: 1:33 - loss: 0.5586 - categorical_accuracy: 0.8216
 6976/60000 [==>...........................] - ETA: 1:32 - loss: 0.5567 - categorical_accuracy: 0.8224
 7008/60000 [==>...........................] - ETA: 1:32 - loss: 0.5550 - categorical_accuracy: 0.8231
 7040/60000 [==>...........................] - ETA: 1:32 - loss: 0.5528 - categorical_accuracy: 0.8239
 7072/60000 [==>...........................] - ETA: 1:32 - loss: 0.5517 - categorical_accuracy: 0.8244
 7104/60000 [==>...........................] - ETA: 1:32 - loss: 0.5515 - categorical_accuracy: 0.8247
 7136/60000 [==>...........................] - ETA: 1:32 - loss: 0.5507 - categorical_accuracy: 0.8253
 7168/60000 [==>...........................] - ETA: 1:32 - loss: 0.5501 - categorical_accuracy: 0.8253
 7200/60000 [==>...........................] - ETA: 1:32 - loss: 0.5502 - categorical_accuracy: 0.8254
 7232/60000 [==>...........................] - ETA: 1:32 - loss: 0.5484 - categorical_accuracy: 0.8259
 7264/60000 [==>...........................] - ETA: 1:32 - loss: 0.5471 - categorical_accuracy: 0.8263
 7296/60000 [==>...........................] - ETA: 1:32 - loss: 0.5452 - categorical_accuracy: 0.8270
 7328/60000 [==>...........................] - ETA: 1:32 - loss: 0.5437 - categorical_accuracy: 0.8274
 7360/60000 [==>...........................] - ETA: 1:32 - loss: 0.5425 - categorical_accuracy: 0.8279
 7392/60000 [==>...........................] - ETA: 1:32 - loss: 0.5420 - categorical_accuracy: 0.8279
 7424/60000 [==>...........................] - ETA: 1:31 - loss: 0.5406 - categorical_accuracy: 0.8284
 7456/60000 [==>...........................] - ETA: 1:31 - loss: 0.5385 - categorical_accuracy: 0.8291
 7488/60000 [==>...........................] - ETA: 1:31 - loss: 0.5377 - categorical_accuracy: 0.8295
 7520/60000 [==>...........................] - ETA: 1:31 - loss: 0.5361 - categorical_accuracy: 0.8299
 7552/60000 [==>...........................] - ETA: 1:31 - loss: 0.5353 - categorical_accuracy: 0.8305
 7584/60000 [==>...........................] - ETA: 1:31 - loss: 0.5335 - categorical_accuracy: 0.8310
 7616/60000 [==>...........................] - ETA: 1:31 - loss: 0.5322 - categorical_accuracy: 0.8314
 7648/60000 [==>...........................] - ETA: 1:31 - loss: 0.5302 - categorical_accuracy: 0.8321
 7680/60000 [==>...........................] - ETA: 1:31 - loss: 0.5292 - categorical_accuracy: 0.8326
 7712/60000 [==>...........................] - ETA: 1:31 - loss: 0.5282 - categorical_accuracy: 0.8330
 7744/60000 [==>...........................] - ETA: 1:31 - loss: 0.5273 - categorical_accuracy: 0.8334
 7776/60000 [==>...........................] - ETA: 1:31 - loss: 0.5261 - categorical_accuracy: 0.8338
 7808/60000 [==>...........................] - ETA: 1:31 - loss: 0.5243 - categorical_accuracy: 0.8345
 7840/60000 [==>...........................] - ETA: 1:31 - loss: 0.5226 - categorical_accuracy: 0.8351
 7872/60000 [==>...........................] - ETA: 1:31 - loss: 0.5211 - categorical_accuracy: 0.8355
 7904/60000 [==>...........................] - ETA: 1:31 - loss: 0.5199 - categorical_accuracy: 0.8359
 7936/60000 [==>...........................] - ETA: 1:31 - loss: 0.5187 - categorical_accuracy: 0.8362
 7968/60000 [==>...........................] - ETA: 1:31 - loss: 0.5180 - categorical_accuracy: 0.8366
 8000/60000 [===>..........................] - ETA: 1:30 - loss: 0.5165 - categorical_accuracy: 0.8370
 8032/60000 [===>..........................] - ETA: 1:30 - loss: 0.5156 - categorical_accuracy: 0.8372
 8064/60000 [===>..........................] - ETA: 1:30 - loss: 0.5140 - categorical_accuracy: 0.8378
 8096/60000 [===>..........................] - ETA: 1:30 - loss: 0.5138 - categorical_accuracy: 0.8379
 8128/60000 [===>..........................] - ETA: 1:30 - loss: 0.5129 - categorical_accuracy: 0.8382
 8160/60000 [===>..........................] - ETA: 1:30 - loss: 0.5113 - categorical_accuracy: 0.8387
 8192/60000 [===>..........................] - ETA: 1:30 - loss: 0.5101 - categorical_accuracy: 0.8392
 8224/60000 [===>..........................] - ETA: 1:30 - loss: 0.5104 - categorical_accuracy: 0.8394
 8256/60000 [===>..........................] - ETA: 1:30 - loss: 0.5090 - categorical_accuracy: 0.8400
 8320/60000 [===>..........................] - ETA: 1:30 - loss: 0.5075 - categorical_accuracy: 0.8406
 8352/60000 [===>..........................] - ETA: 1:30 - loss: 0.5063 - categorical_accuracy: 0.8410
 8384/60000 [===>..........................] - ETA: 1:30 - loss: 0.5051 - categorical_accuracy: 0.8414
 8416/60000 [===>..........................] - ETA: 1:30 - loss: 0.5036 - categorical_accuracy: 0.8418
 8448/60000 [===>..........................] - ETA: 1:30 - loss: 0.5018 - categorical_accuracy: 0.8424
 8480/60000 [===>..........................] - ETA: 1:29 - loss: 0.5030 - categorical_accuracy: 0.8425
 8512/60000 [===>..........................] - ETA: 1:29 - loss: 0.5021 - categorical_accuracy: 0.8427
 8544/60000 [===>..........................] - ETA: 1:29 - loss: 0.5004 - categorical_accuracy: 0.8433
 8576/60000 [===>..........................] - ETA: 1:29 - loss: 0.4993 - categorical_accuracy: 0.8436
 8608/60000 [===>..........................] - ETA: 1:29 - loss: 0.4991 - categorical_accuracy: 0.8436
 8640/60000 [===>..........................] - ETA: 1:29 - loss: 0.4992 - categorical_accuracy: 0.8434
 8672/60000 [===>..........................] - ETA: 1:29 - loss: 0.4980 - categorical_accuracy: 0.8438
 8704/60000 [===>..........................] - ETA: 1:29 - loss: 0.4967 - categorical_accuracy: 0.8441
 8736/60000 [===>..........................] - ETA: 1:29 - loss: 0.4955 - categorical_accuracy: 0.8444
 8800/60000 [===>..........................] - ETA: 1:29 - loss: 0.4946 - categorical_accuracy: 0.8444
 8864/60000 [===>..........................] - ETA: 1:29 - loss: 0.4929 - categorical_accuracy: 0.8452
 8896/60000 [===>..........................] - ETA: 1:29 - loss: 0.4932 - categorical_accuracy: 0.8451
 8928/60000 [===>..........................] - ETA: 1:28 - loss: 0.4923 - categorical_accuracy: 0.8452
 8960/60000 [===>..........................] - ETA: 1:28 - loss: 0.4909 - categorical_accuracy: 0.8458
 8992/60000 [===>..........................] - ETA: 1:28 - loss: 0.4895 - categorical_accuracy: 0.8462
 9024/60000 [===>..........................] - ETA: 1:28 - loss: 0.4891 - categorical_accuracy: 0.8464
 9056/60000 [===>..........................] - ETA: 1:28 - loss: 0.4879 - categorical_accuracy: 0.8468
 9088/60000 [===>..........................] - ETA: 1:28 - loss: 0.4872 - categorical_accuracy: 0.8473
 9120/60000 [===>..........................] - ETA: 1:28 - loss: 0.4865 - categorical_accuracy: 0.8473
 9152/60000 [===>..........................] - ETA: 1:28 - loss: 0.4857 - categorical_accuracy: 0.8477
 9184/60000 [===>..........................] - ETA: 1:28 - loss: 0.4849 - categorical_accuracy: 0.8479
 9216/60000 [===>..........................] - ETA: 1:28 - loss: 0.4844 - categorical_accuracy: 0.8482
 9248/60000 [===>..........................] - ETA: 1:28 - loss: 0.4841 - categorical_accuracy: 0.8484
 9280/60000 [===>..........................] - ETA: 1:28 - loss: 0.4830 - categorical_accuracy: 0.8488
 9312/60000 [===>..........................] - ETA: 1:28 - loss: 0.4815 - categorical_accuracy: 0.8493
 9376/60000 [===>..........................] - ETA: 1:28 - loss: 0.4799 - categorical_accuracy: 0.8499
 9408/60000 [===>..........................] - ETA: 1:27 - loss: 0.4785 - categorical_accuracy: 0.8504
 9440/60000 [===>..........................] - ETA: 1:27 - loss: 0.4780 - categorical_accuracy: 0.8506
 9472/60000 [===>..........................] - ETA: 1:27 - loss: 0.4780 - categorical_accuracy: 0.8507
 9504/60000 [===>..........................] - ETA: 1:27 - loss: 0.4773 - categorical_accuracy: 0.8509
 9536/60000 [===>..........................] - ETA: 1:27 - loss: 0.4761 - categorical_accuracy: 0.8512
 9568/60000 [===>..........................] - ETA: 1:27 - loss: 0.4748 - categorical_accuracy: 0.8516
 9600/60000 [===>..........................] - ETA: 1:27 - loss: 0.4735 - categorical_accuracy: 0.8520
 9632/60000 [===>..........................] - ETA: 1:27 - loss: 0.4725 - categorical_accuracy: 0.8523
 9664/60000 [===>..........................] - ETA: 1:27 - loss: 0.4719 - categorical_accuracy: 0.8524
 9728/60000 [===>..........................] - ETA: 1:27 - loss: 0.4712 - categorical_accuracy: 0.8527
 9792/60000 [===>..........................] - ETA: 1:27 - loss: 0.4695 - categorical_accuracy: 0.8532
 9824/60000 [===>..........................] - ETA: 1:27 - loss: 0.4683 - categorical_accuracy: 0.8537
 9856/60000 [===>..........................] - ETA: 1:26 - loss: 0.4673 - categorical_accuracy: 0.8540
 9888/60000 [===>..........................] - ETA: 1:26 - loss: 0.4663 - categorical_accuracy: 0.8542
 9920/60000 [===>..........................] - ETA: 1:26 - loss: 0.4658 - categorical_accuracy: 0.8543
 9952/60000 [===>..........................] - ETA: 1:26 - loss: 0.4646 - categorical_accuracy: 0.8547
 9984/60000 [===>..........................] - ETA: 1:26 - loss: 0.4645 - categorical_accuracy: 0.8550
10016/60000 [====>.........................] - ETA: 1:26 - loss: 0.4641 - categorical_accuracy: 0.8550
10048/60000 [====>.........................] - ETA: 1:26 - loss: 0.4628 - categorical_accuracy: 0.8555
10080/60000 [====>.........................] - ETA: 1:26 - loss: 0.4624 - categorical_accuracy: 0.8556
10112/60000 [====>.........................] - ETA: 1:26 - loss: 0.4613 - categorical_accuracy: 0.8559
10144/60000 [====>.........................] - ETA: 1:26 - loss: 0.4604 - categorical_accuracy: 0.8563
10176/60000 [====>.........................] - ETA: 1:26 - loss: 0.4594 - categorical_accuracy: 0.8566
10208/60000 [====>.........................] - ETA: 1:26 - loss: 0.4589 - categorical_accuracy: 0.8567
10240/60000 [====>.........................] - ETA: 1:26 - loss: 0.4590 - categorical_accuracy: 0.8568
10272/60000 [====>.........................] - ETA: 1:26 - loss: 0.4583 - categorical_accuracy: 0.8571
10304/60000 [====>.........................] - ETA: 1:26 - loss: 0.4577 - categorical_accuracy: 0.8573
10336/60000 [====>.........................] - ETA: 1:26 - loss: 0.4572 - categorical_accuracy: 0.8576
10368/60000 [====>.........................] - ETA: 1:25 - loss: 0.4565 - categorical_accuracy: 0.8577
10400/60000 [====>.........................] - ETA: 1:25 - loss: 0.4556 - categorical_accuracy: 0.8580
10432/60000 [====>.........................] - ETA: 1:25 - loss: 0.4550 - categorical_accuracy: 0.8582
10464/60000 [====>.........................] - ETA: 1:25 - loss: 0.4539 - categorical_accuracy: 0.8586
10496/60000 [====>.........................] - ETA: 1:25 - loss: 0.4526 - categorical_accuracy: 0.8590
10528/60000 [====>.........................] - ETA: 1:25 - loss: 0.4517 - categorical_accuracy: 0.8592
10560/60000 [====>.........................] - ETA: 1:25 - loss: 0.4509 - categorical_accuracy: 0.8595
10592/60000 [====>.........................] - ETA: 1:25 - loss: 0.4510 - categorical_accuracy: 0.8594
10624/60000 [====>.........................] - ETA: 1:25 - loss: 0.4501 - categorical_accuracy: 0.8597
10656/60000 [====>.........................] - ETA: 1:25 - loss: 0.4495 - categorical_accuracy: 0.8600
10688/60000 [====>.........................] - ETA: 1:25 - loss: 0.4485 - categorical_accuracy: 0.8602
10752/60000 [====>.........................] - ETA: 1:25 - loss: 0.4474 - categorical_accuracy: 0.8601
10816/60000 [====>.........................] - ETA: 1:25 - loss: 0.4457 - categorical_accuracy: 0.8607
10880/60000 [====>.........................] - ETA: 1:24 - loss: 0.4437 - categorical_accuracy: 0.8613
10912/60000 [====>.........................] - ETA: 1:24 - loss: 0.4428 - categorical_accuracy: 0.8616
10944/60000 [====>.........................] - ETA: 1:24 - loss: 0.4417 - categorical_accuracy: 0.8620
10976/60000 [====>.........................] - ETA: 1:24 - loss: 0.4417 - categorical_accuracy: 0.8622
11008/60000 [====>.........................] - ETA: 1:24 - loss: 0.4409 - categorical_accuracy: 0.8624
11040/60000 [====>.........................] - ETA: 1:24 - loss: 0.4399 - categorical_accuracy: 0.8627
11072/60000 [====>.........................] - ETA: 1:24 - loss: 0.4388 - categorical_accuracy: 0.8631
11104/60000 [====>.........................] - ETA: 1:24 - loss: 0.4377 - categorical_accuracy: 0.8635
11136/60000 [====>.........................] - ETA: 1:24 - loss: 0.4370 - categorical_accuracy: 0.8638
11168/60000 [====>.........................] - ETA: 1:24 - loss: 0.4370 - categorical_accuracy: 0.8637
11200/60000 [====>.........................] - ETA: 1:24 - loss: 0.4368 - categorical_accuracy: 0.8638
11232/60000 [====>.........................] - ETA: 1:24 - loss: 0.4358 - categorical_accuracy: 0.8641
11264/60000 [====>.........................] - ETA: 1:24 - loss: 0.4348 - categorical_accuracy: 0.8644
11296/60000 [====>.........................] - ETA: 1:24 - loss: 0.4338 - categorical_accuracy: 0.8646
11328/60000 [====>.........................] - ETA: 1:24 - loss: 0.4336 - categorical_accuracy: 0.8646
11360/60000 [====>.........................] - ETA: 1:23 - loss: 0.4326 - categorical_accuracy: 0.8649
11392/60000 [====>.........................] - ETA: 1:23 - loss: 0.4320 - categorical_accuracy: 0.8651
11424/60000 [====>.........................] - ETA: 1:23 - loss: 0.4308 - categorical_accuracy: 0.8655
11456/60000 [====>.........................] - ETA: 1:23 - loss: 0.4306 - categorical_accuracy: 0.8655
11488/60000 [====>.........................] - ETA: 1:23 - loss: 0.4301 - categorical_accuracy: 0.8657
11552/60000 [====>.........................] - ETA: 1:23 - loss: 0.4284 - categorical_accuracy: 0.8662
11584/60000 [====>.........................] - ETA: 1:23 - loss: 0.4288 - categorical_accuracy: 0.8661
11648/60000 [====>.........................] - ETA: 1:23 - loss: 0.4285 - categorical_accuracy: 0.8662
11712/60000 [====>.........................] - ETA: 1:23 - loss: 0.4274 - categorical_accuracy: 0.8665
11744/60000 [====>.........................] - ETA: 1:23 - loss: 0.4268 - categorical_accuracy: 0.8667
11776/60000 [====>.........................] - ETA: 1:23 - loss: 0.4260 - categorical_accuracy: 0.8668
11808/60000 [====>.........................] - ETA: 1:23 - loss: 0.4261 - categorical_accuracy: 0.8669
11840/60000 [====>.........................] - ETA: 1:23 - loss: 0.4260 - categorical_accuracy: 0.8670
11872/60000 [====>.........................] - ETA: 1:23 - loss: 0.4250 - categorical_accuracy: 0.8673
11904/60000 [====>.........................] - ETA: 1:22 - loss: 0.4242 - categorical_accuracy: 0.8675
11936/60000 [====>.........................] - ETA: 1:22 - loss: 0.4237 - categorical_accuracy: 0.8677
11968/60000 [====>.........................] - ETA: 1:22 - loss: 0.4226 - categorical_accuracy: 0.8681
12032/60000 [=====>........................] - ETA: 1:22 - loss: 0.4211 - categorical_accuracy: 0.8685
12096/60000 [=====>........................] - ETA: 1:22 - loss: 0.4196 - categorical_accuracy: 0.8690
12160/60000 [=====>........................] - ETA: 1:22 - loss: 0.4184 - categorical_accuracy: 0.8693
12192/60000 [=====>........................] - ETA: 1:22 - loss: 0.4179 - categorical_accuracy: 0.8695
12224/60000 [=====>........................] - ETA: 1:22 - loss: 0.4174 - categorical_accuracy: 0.8698
12256/60000 [=====>........................] - ETA: 1:22 - loss: 0.4166 - categorical_accuracy: 0.8700
12288/60000 [=====>........................] - ETA: 1:22 - loss: 0.4159 - categorical_accuracy: 0.8702
12320/60000 [=====>........................] - ETA: 1:22 - loss: 0.4151 - categorical_accuracy: 0.8705
12352/60000 [=====>........................] - ETA: 1:22 - loss: 0.4142 - categorical_accuracy: 0.8707
12384/60000 [=====>........................] - ETA: 1:21 - loss: 0.4141 - categorical_accuracy: 0.8708
12416/60000 [=====>........................] - ETA: 1:21 - loss: 0.4134 - categorical_accuracy: 0.8711
12448/60000 [=====>........................] - ETA: 1:21 - loss: 0.4125 - categorical_accuracy: 0.8713
12480/60000 [=====>........................] - ETA: 1:21 - loss: 0.4119 - categorical_accuracy: 0.8716
12512/60000 [=====>........................] - ETA: 1:21 - loss: 0.4114 - categorical_accuracy: 0.8716
12544/60000 [=====>........................] - ETA: 1:21 - loss: 0.4105 - categorical_accuracy: 0.8719
12576/60000 [=====>........................] - ETA: 1:21 - loss: 0.4098 - categorical_accuracy: 0.8721
12608/60000 [=====>........................] - ETA: 1:21 - loss: 0.4090 - categorical_accuracy: 0.8724
12640/60000 [=====>........................] - ETA: 1:21 - loss: 0.4088 - categorical_accuracy: 0.8725
12672/60000 [=====>........................] - ETA: 1:21 - loss: 0.4081 - categorical_accuracy: 0.8727
12704/60000 [=====>........................] - ETA: 1:21 - loss: 0.4079 - categorical_accuracy: 0.8728
12736/60000 [=====>........................] - ETA: 1:21 - loss: 0.4071 - categorical_accuracy: 0.8730
12800/60000 [=====>........................] - ETA: 1:21 - loss: 0.4065 - categorical_accuracy: 0.8731
12864/60000 [=====>........................] - ETA: 1:21 - loss: 0.4052 - categorical_accuracy: 0.8737
12896/60000 [=====>........................] - ETA: 1:21 - loss: 0.4044 - categorical_accuracy: 0.8740
12928/60000 [=====>........................] - ETA: 1:20 - loss: 0.4040 - categorical_accuracy: 0.8740
12960/60000 [=====>........................] - ETA: 1:20 - loss: 0.4033 - categorical_accuracy: 0.8743
12992/60000 [=====>........................] - ETA: 1:20 - loss: 0.4029 - categorical_accuracy: 0.8744
13024/60000 [=====>........................] - ETA: 1:20 - loss: 0.4021 - categorical_accuracy: 0.8746
13056/60000 [=====>........................] - ETA: 1:20 - loss: 0.4014 - categorical_accuracy: 0.8748
13088/60000 [=====>........................] - ETA: 1:20 - loss: 0.4009 - categorical_accuracy: 0.8749
13120/60000 [=====>........................] - ETA: 1:20 - loss: 0.4005 - categorical_accuracy: 0.8751
13152/60000 [=====>........................] - ETA: 1:20 - loss: 0.4005 - categorical_accuracy: 0.8752
13184/60000 [=====>........................] - ETA: 1:20 - loss: 0.4002 - categorical_accuracy: 0.8754
13216/60000 [=====>........................] - ETA: 1:20 - loss: 0.3996 - categorical_accuracy: 0.8755
13248/60000 [=====>........................] - ETA: 1:20 - loss: 0.3989 - categorical_accuracy: 0.8757
13280/60000 [=====>........................] - ETA: 1:20 - loss: 0.3986 - categorical_accuracy: 0.8757
13312/60000 [=====>........................] - ETA: 1:20 - loss: 0.3987 - categorical_accuracy: 0.8756
13344/60000 [=====>........................] - ETA: 1:20 - loss: 0.3982 - categorical_accuracy: 0.8756
13376/60000 [=====>........................] - ETA: 1:20 - loss: 0.3975 - categorical_accuracy: 0.8758
13408/60000 [=====>........................] - ETA: 1:20 - loss: 0.3969 - categorical_accuracy: 0.8760
13440/60000 [=====>........................] - ETA: 1:20 - loss: 0.3961 - categorical_accuracy: 0.8763
13472/60000 [=====>........................] - ETA: 1:20 - loss: 0.3957 - categorical_accuracy: 0.8764
13504/60000 [=====>........................] - ETA: 1:20 - loss: 0.3950 - categorical_accuracy: 0.8766
13536/60000 [=====>........................] - ETA: 1:20 - loss: 0.3943 - categorical_accuracy: 0.8768
13568/60000 [=====>........................] - ETA: 1:19 - loss: 0.3947 - categorical_accuracy: 0.8769
13600/60000 [=====>........................] - ETA: 1:19 - loss: 0.3941 - categorical_accuracy: 0.8771
13632/60000 [=====>........................] - ETA: 1:19 - loss: 0.3938 - categorical_accuracy: 0.8771
13664/60000 [=====>........................] - ETA: 1:19 - loss: 0.3932 - categorical_accuracy: 0.8773
13696/60000 [=====>........................] - ETA: 1:19 - loss: 0.3924 - categorical_accuracy: 0.8776
13728/60000 [=====>........................] - ETA: 1:19 - loss: 0.3918 - categorical_accuracy: 0.8778
13760/60000 [=====>........................] - ETA: 1:19 - loss: 0.3913 - categorical_accuracy: 0.8780
13792/60000 [=====>........................] - ETA: 1:19 - loss: 0.3905 - categorical_accuracy: 0.8783
13824/60000 [=====>........................] - ETA: 1:19 - loss: 0.3898 - categorical_accuracy: 0.8785
13856/60000 [=====>........................] - ETA: 1:19 - loss: 0.3892 - categorical_accuracy: 0.8787
13888/60000 [=====>........................] - ETA: 1:19 - loss: 0.3884 - categorical_accuracy: 0.8789
13920/60000 [=====>........................] - ETA: 1:19 - loss: 0.3879 - categorical_accuracy: 0.8791
13952/60000 [=====>........................] - ETA: 1:19 - loss: 0.3876 - categorical_accuracy: 0.8793
13984/60000 [=====>........................] - ETA: 1:19 - loss: 0.3870 - categorical_accuracy: 0.8795
14016/60000 [======>.......................] - ETA: 1:19 - loss: 0.3864 - categorical_accuracy: 0.8796
14048/60000 [======>.......................] - ETA: 1:19 - loss: 0.3859 - categorical_accuracy: 0.8797
14080/60000 [======>.......................] - ETA: 1:19 - loss: 0.3852 - categorical_accuracy: 0.8799
14112/60000 [======>.......................] - ETA: 1:19 - loss: 0.3846 - categorical_accuracy: 0.8801
14144/60000 [======>.......................] - ETA: 1:18 - loss: 0.3838 - categorical_accuracy: 0.8804
14176/60000 [======>.......................] - ETA: 1:18 - loss: 0.3831 - categorical_accuracy: 0.8806
14208/60000 [======>.......................] - ETA: 1:18 - loss: 0.3825 - categorical_accuracy: 0.8808
14240/60000 [======>.......................] - ETA: 1:18 - loss: 0.3823 - categorical_accuracy: 0.8809
14272/60000 [======>.......................] - ETA: 1:18 - loss: 0.3816 - categorical_accuracy: 0.8811
14304/60000 [======>.......................] - ETA: 1:18 - loss: 0.3810 - categorical_accuracy: 0.8813
14336/60000 [======>.......................] - ETA: 1:18 - loss: 0.3803 - categorical_accuracy: 0.8815
14368/60000 [======>.......................] - ETA: 1:18 - loss: 0.3802 - categorical_accuracy: 0.8815
14400/60000 [======>.......................] - ETA: 1:18 - loss: 0.3803 - categorical_accuracy: 0.8815
14432/60000 [======>.......................] - ETA: 1:18 - loss: 0.3802 - categorical_accuracy: 0.8816
14464/60000 [======>.......................] - ETA: 1:18 - loss: 0.3795 - categorical_accuracy: 0.8817
14528/60000 [======>.......................] - ETA: 1:18 - loss: 0.3785 - categorical_accuracy: 0.8820
14560/60000 [======>.......................] - ETA: 1:18 - loss: 0.3777 - categorical_accuracy: 0.8823
14592/60000 [======>.......................] - ETA: 1:18 - loss: 0.3774 - categorical_accuracy: 0.8824
14624/60000 [======>.......................] - ETA: 1:18 - loss: 0.3767 - categorical_accuracy: 0.8827
14656/60000 [======>.......................] - ETA: 1:18 - loss: 0.3762 - categorical_accuracy: 0.8828
14688/60000 [======>.......................] - ETA: 1:17 - loss: 0.3754 - categorical_accuracy: 0.8831
14720/60000 [======>.......................] - ETA: 1:17 - loss: 0.3751 - categorical_accuracy: 0.8832
14752/60000 [======>.......................] - ETA: 1:17 - loss: 0.3745 - categorical_accuracy: 0.8834
14784/60000 [======>.......................] - ETA: 1:17 - loss: 0.3743 - categorical_accuracy: 0.8834
14816/60000 [======>.......................] - ETA: 1:17 - loss: 0.3736 - categorical_accuracy: 0.8836
14848/60000 [======>.......................] - ETA: 1:17 - loss: 0.3735 - categorical_accuracy: 0.8837
14880/60000 [======>.......................] - ETA: 1:17 - loss: 0.3737 - categorical_accuracy: 0.8836
14912/60000 [======>.......................] - ETA: 1:17 - loss: 0.3731 - categorical_accuracy: 0.8839
14944/60000 [======>.......................] - ETA: 1:17 - loss: 0.3725 - categorical_accuracy: 0.8840
14976/60000 [======>.......................] - ETA: 1:17 - loss: 0.3718 - categorical_accuracy: 0.8843
15008/60000 [======>.......................] - ETA: 1:17 - loss: 0.3716 - categorical_accuracy: 0.8844
15040/60000 [======>.......................] - ETA: 1:17 - loss: 0.3713 - categorical_accuracy: 0.8844
15072/60000 [======>.......................] - ETA: 1:17 - loss: 0.3708 - categorical_accuracy: 0.8846
15104/60000 [======>.......................] - ETA: 1:17 - loss: 0.3704 - categorical_accuracy: 0.8847
15136/60000 [======>.......................] - ETA: 1:17 - loss: 0.3699 - categorical_accuracy: 0.8848
15168/60000 [======>.......................] - ETA: 1:17 - loss: 0.3693 - categorical_accuracy: 0.8850
15200/60000 [======>.......................] - ETA: 1:17 - loss: 0.3689 - categorical_accuracy: 0.8851
15232/60000 [======>.......................] - ETA: 1:16 - loss: 0.3689 - categorical_accuracy: 0.8851
15264/60000 [======>.......................] - ETA: 1:16 - loss: 0.3681 - categorical_accuracy: 0.8854
15296/60000 [======>.......................] - ETA: 1:16 - loss: 0.3676 - categorical_accuracy: 0.8855
15360/60000 [======>.......................] - ETA: 1:16 - loss: 0.3668 - categorical_accuracy: 0.8857
15392/60000 [======>.......................] - ETA: 1:16 - loss: 0.3665 - categorical_accuracy: 0.8857
15424/60000 [======>.......................] - ETA: 1:16 - loss: 0.3667 - categorical_accuracy: 0.8858
15456/60000 [======>.......................] - ETA: 1:16 - loss: 0.3662 - categorical_accuracy: 0.8859
15488/60000 [======>.......................] - ETA: 1:16 - loss: 0.3658 - categorical_accuracy: 0.8860
15520/60000 [======>.......................] - ETA: 1:16 - loss: 0.3653 - categorical_accuracy: 0.8862
15552/60000 [======>.......................] - ETA: 1:16 - loss: 0.3651 - categorical_accuracy: 0.8863
15616/60000 [======>.......................] - ETA: 1:16 - loss: 0.3646 - categorical_accuracy: 0.8863
15648/60000 [======>.......................] - ETA: 1:16 - loss: 0.3642 - categorical_accuracy: 0.8865
15680/60000 [======>.......................] - ETA: 1:16 - loss: 0.3641 - categorical_accuracy: 0.8865
15712/60000 [======>.......................] - ETA: 1:16 - loss: 0.3635 - categorical_accuracy: 0.8868
15744/60000 [======>.......................] - ETA: 1:15 - loss: 0.3634 - categorical_accuracy: 0.8868
15776/60000 [======>.......................] - ETA: 1:15 - loss: 0.3628 - categorical_accuracy: 0.8870
15808/60000 [======>.......................] - ETA: 1:15 - loss: 0.3622 - categorical_accuracy: 0.8872
15840/60000 [======>.......................] - ETA: 1:15 - loss: 0.3618 - categorical_accuracy: 0.8872
15872/60000 [======>.......................] - ETA: 1:15 - loss: 0.3617 - categorical_accuracy: 0.8874
15904/60000 [======>.......................] - ETA: 1:15 - loss: 0.3614 - categorical_accuracy: 0.8875
15936/60000 [======>.......................] - ETA: 1:15 - loss: 0.3607 - categorical_accuracy: 0.8877
15968/60000 [======>.......................] - ETA: 1:15 - loss: 0.3602 - categorical_accuracy: 0.8879
16000/60000 [=======>......................] - ETA: 1:15 - loss: 0.3600 - categorical_accuracy: 0.8879
16032/60000 [=======>......................] - ETA: 1:15 - loss: 0.3594 - categorical_accuracy: 0.8882
16064/60000 [=======>......................] - ETA: 1:15 - loss: 0.3588 - categorical_accuracy: 0.8884
16096/60000 [=======>......................] - ETA: 1:15 - loss: 0.3586 - categorical_accuracy: 0.8885
16128/60000 [=======>......................] - ETA: 1:15 - loss: 0.3583 - categorical_accuracy: 0.8886
16160/60000 [=======>......................] - ETA: 1:15 - loss: 0.3580 - categorical_accuracy: 0.8888
16192/60000 [=======>......................] - ETA: 1:15 - loss: 0.3576 - categorical_accuracy: 0.8889
16224/60000 [=======>......................] - ETA: 1:15 - loss: 0.3572 - categorical_accuracy: 0.8891
16256/60000 [=======>......................] - ETA: 1:15 - loss: 0.3565 - categorical_accuracy: 0.8893
16288/60000 [=======>......................] - ETA: 1:14 - loss: 0.3560 - categorical_accuracy: 0.8895
16320/60000 [=======>......................] - ETA: 1:14 - loss: 0.3563 - categorical_accuracy: 0.8895
16352/60000 [=======>......................] - ETA: 1:14 - loss: 0.3557 - categorical_accuracy: 0.8897
16384/60000 [=======>......................] - ETA: 1:14 - loss: 0.3554 - categorical_accuracy: 0.8898
16416/60000 [=======>......................] - ETA: 1:14 - loss: 0.3549 - categorical_accuracy: 0.8899
16448/60000 [=======>......................] - ETA: 1:14 - loss: 0.3542 - categorical_accuracy: 0.8901
16480/60000 [=======>......................] - ETA: 1:14 - loss: 0.3537 - categorical_accuracy: 0.8903
16512/60000 [=======>......................] - ETA: 1:14 - loss: 0.3537 - categorical_accuracy: 0.8904
16544/60000 [=======>......................] - ETA: 1:14 - loss: 0.3533 - categorical_accuracy: 0.8905
16576/60000 [=======>......................] - ETA: 1:14 - loss: 0.3528 - categorical_accuracy: 0.8907
16608/60000 [=======>......................] - ETA: 1:14 - loss: 0.3525 - categorical_accuracy: 0.8907
16640/60000 [=======>......................] - ETA: 1:14 - loss: 0.3521 - categorical_accuracy: 0.8907
16672/60000 [=======>......................] - ETA: 1:14 - loss: 0.3516 - categorical_accuracy: 0.8910
16736/60000 [=======>......................] - ETA: 1:14 - loss: 0.3508 - categorical_accuracy: 0.8912
16768/60000 [=======>......................] - ETA: 1:14 - loss: 0.3505 - categorical_accuracy: 0.8913
16800/60000 [=======>......................] - ETA: 1:14 - loss: 0.3500 - categorical_accuracy: 0.8914
16832/60000 [=======>......................] - ETA: 1:13 - loss: 0.3498 - categorical_accuracy: 0.8915
16864/60000 [=======>......................] - ETA: 1:13 - loss: 0.3494 - categorical_accuracy: 0.8916
16896/60000 [=======>......................] - ETA: 1:13 - loss: 0.3494 - categorical_accuracy: 0.8917
16928/60000 [=======>......................] - ETA: 1:13 - loss: 0.3498 - categorical_accuracy: 0.8917
16960/60000 [=======>......................] - ETA: 1:13 - loss: 0.3494 - categorical_accuracy: 0.8918
16992/60000 [=======>......................] - ETA: 1:13 - loss: 0.3491 - categorical_accuracy: 0.8919
17024/60000 [=======>......................] - ETA: 1:13 - loss: 0.3496 - categorical_accuracy: 0.8919
17056/60000 [=======>......................] - ETA: 1:13 - loss: 0.3490 - categorical_accuracy: 0.8921
17088/60000 [=======>......................] - ETA: 1:13 - loss: 0.3486 - categorical_accuracy: 0.8922
17120/60000 [=======>......................] - ETA: 1:13 - loss: 0.3481 - categorical_accuracy: 0.8923
17152/60000 [=======>......................] - ETA: 1:13 - loss: 0.3477 - categorical_accuracy: 0.8924
17184/60000 [=======>......................] - ETA: 1:13 - loss: 0.3476 - categorical_accuracy: 0.8923
17216/60000 [=======>......................] - ETA: 1:13 - loss: 0.3472 - categorical_accuracy: 0.8925
17248/60000 [=======>......................] - ETA: 1:13 - loss: 0.3467 - categorical_accuracy: 0.8926
17280/60000 [=======>......................] - ETA: 1:13 - loss: 0.3464 - categorical_accuracy: 0.8928
17312/60000 [=======>......................] - ETA: 1:13 - loss: 0.3459 - categorical_accuracy: 0.8929
17344/60000 [=======>......................] - ETA: 1:13 - loss: 0.3454 - categorical_accuracy: 0.8930
17376/60000 [=======>......................] - ETA: 1:13 - loss: 0.3456 - categorical_accuracy: 0.8930
17408/60000 [=======>......................] - ETA: 1:12 - loss: 0.3455 - categorical_accuracy: 0.8931
17440/60000 [=======>......................] - ETA: 1:12 - loss: 0.3449 - categorical_accuracy: 0.8933
17472/60000 [=======>......................] - ETA: 1:12 - loss: 0.3446 - categorical_accuracy: 0.8934
17504/60000 [=======>......................] - ETA: 1:12 - loss: 0.3441 - categorical_accuracy: 0.8936
17536/60000 [=======>......................] - ETA: 1:12 - loss: 0.3436 - categorical_accuracy: 0.8938
17568/60000 [=======>......................] - ETA: 1:12 - loss: 0.3432 - categorical_accuracy: 0.8938
17600/60000 [=======>......................] - ETA: 1:12 - loss: 0.3427 - categorical_accuracy: 0.8940
17632/60000 [=======>......................] - ETA: 1:12 - loss: 0.3426 - categorical_accuracy: 0.8941
17664/60000 [=======>......................] - ETA: 1:12 - loss: 0.3426 - categorical_accuracy: 0.8941
17696/60000 [=======>......................] - ETA: 1:12 - loss: 0.3421 - categorical_accuracy: 0.8943
17728/60000 [=======>......................] - ETA: 1:12 - loss: 0.3417 - categorical_accuracy: 0.8944
17760/60000 [=======>......................] - ETA: 1:12 - loss: 0.3412 - categorical_accuracy: 0.8945
17792/60000 [=======>......................] - ETA: 1:12 - loss: 0.3408 - categorical_accuracy: 0.8947
17824/60000 [=======>......................] - ETA: 1:12 - loss: 0.3405 - categorical_accuracy: 0.8947
17856/60000 [=======>......................] - ETA: 1:12 - loss: 0.3403 - categorical_accuracy: 0.8949
17888/60000 [=======>......................] - ETA: 1:12 - loss: 0.3404 - categorical_accuracy: 0.8949
17920/60000 [=======>......................] - ETA: 1:12 - loss: 0.3399 - categorical_accuracy: 0.8951
17952/60000 [=======>......................] - ETA: 1:12 - loss: 0.3394 - categorical_accuracy: 0.8952
17984/60000 [=======>......................] - ETA: 1:11 - loss: 0.3396 - categorical_accuracy: 0.8952
18016/60000 [========>.....................] - ETA: 1:11 - loss: 0.3392 - categorical_accuracy: 0.8953
18048/60000 [========>.....................] - ETA: 1:11 - loss: 0.3392 - categorical_accuracy: 0.8953
18080/60000 [========>.....................] - ETA: 1:11 - loss: 0.3388 - categorical_accuracy: 0.8954
18112/60000 [========>.....................] - ETA: 1:11 - loss: 0.3385 - categorical_accuracy: 0.8955
18144/60000 [========>.....................] - ETA: 1:11 - loss: 0.3384 - categorical_accuracy: 0.8956
18176/60000 [========>.....................] - ETA: 1:11 - loss: 0.3380 - categorical_accuracy: 0.8957
18208/60000 [========>.....................] - ETA: 1:11 - loss: 0.3377 - categorical_accuracy: 0.8958
18240/60000 [========>.....................] - ETA: 1:11 - loss: 0.3375 - categorical_accuracy: 0.8959
18272/60000 [========>.....................] - ETA: 1:11 - loss: 0.3374 - categorical_accuracy: 0.8961
18304/60000 [========>.....................] - ETA: 1:11 - loss: 0.3370 - categorical_accuracy: 0.8962
18336/60000 [========>.....................] - ETA: 1:11 - loss: 0.3370 - categorical_accuracy: 0.8962
18368/60000 [========>.....................] - ETA: 1:11 - loss: 0.3366 - categorical_accuracy: 0.8963
18400/60000 [========>.....................] - ETA: 1:11 - loss: 0.3363 - categorical_accuracy: 0.8964
18464/60000 [========>.....................] - ETA: 1:11 - loss: 0.3356 - categorical_accuracy: 0.8964
18496/60000 [========>.....................] - ETA: 1:11 - loss: 0.3351 - categorical_accuracy: 0.8966
18528/60000 [========>.....................] - ETA: 1:11 - loss: 0.3346 - categorical_accuracy: 0.8968
18560/60000 [========>.....................] - ETA: 1:11 - loss: 0.3342 - categorical_accuracy: 0.8969
18592/60000 [========>.....................] - ETA: 1:10 - loss: 0.3346 - categorical_accuracy: 0.8967
18624/60000 [========>.....................] - ETA: 1:10 - loss: 0.3342 - categorical_accuracy: 0.8969
18656/60000 [========>.....................] - ETA: 1:10 - loss: 0.3338 - categorical_accuracy: 0.8970
18688/60000 [========>.....................] - ETA: 1:10 - loss: 0.3339 - categorical_accuracy: 0.8969
18720/60000 [========>.....................] - ETA: 1:10 - loss: 0.3335 - categorical_accuracy: 0.8971
18752/60000 [========>.....................] - ETA: 1:10 - loss: 0.3332 - categorical_accuracy: 0.8972
18784/60000 [========>.....................] - ETA: 1:10 - loss: 0.3331 - categorical_accuracy: 0.8973
18816/60000 [========>.....................] - ETA: 1:10 - loss: 0.3328 - categorical_accuracy: 0.8974
18848/60000 [========>.....................] - ETA: 1:10 - loss: 0.3322 - categorical_accuracy: 0.8975
18880/60000 [========>.....................] - ETA: 1:10 - loss: 0.3320 - categorical_accuracy: 0.8976
18912/60000 [========>.....................] - ETA: 1:10 - loss: 0.3320 - categorical_accuracy: 0.8976
18944/60000 [========>.....................] - ETA: 1:10 - loss: 0.3316 - categorical_accuracy: 0.8978
18976/60000 [========>.....................] - ETA: 1:10 - loss: 0.3314 - categorical_accuracy: 0.8979
19008/60000 [========>.....................] - ETA: 1:10 - loss: 0.3311 - categorical_accuracy: 0.8979
19040/60000 [========>.....................] - ETA: 1:10 - loss: 0.3308 - categorical_accuracy: 0.8981
19072/60000 [========>.....................] - ETA: 1:10 - loss: 0.3307 - categorical_accuracy: 0.8982
19104/60000 [========>.....................] - ETA: 1:10 - loss: 0.3302 - categorical_accuracy: 0.8983
19136/60000 [========>.....................] - ETA: 1:10 - loss: 0.3297 - categorical_accuracy: 0.8985
19168/60000 [========>.....................] - ETA: 1:10 - loss: 0.3293 - categorical_accuracy: 0.8987
19200/60000 [========>.....................] - ETA: 1:09 - loss: 0.3289 - categorical_accuracy: 0.8988
19232/60000 [========>.....................] - ETA: 1:09 - loss: 0.3284 - categorical_accuracy: 0.8989
19264/60000 [========>.....................] - ETA: 1:09 - loss: 0.3281 - categorical_accuracy: 0.8990
19296/60000 [========>.....................] - ETA: 1:09 - loss: 0.3277 - categorical_accuracy: 0.8992
19328/60000 [========>.....................] - ETA: 1:09 - loss: 0.3273 - categorical_accuracy: 0.8993
19360/60000 [========>.....................] - ETA: 1:09 - loss: 0.3268 - categorical_accuracy: 0.8994
19424/60000 [========>.....................] - ETA: 1:09 - loss: 0.3259 - categorical_accuracy: 0.8997
19456/60000 [========>.....................] - ETA: 1:09 - loss: 0.3255 - categorical_accuracy: 0.8997
19488/60000 [========>.....................] - ETA: 1:09 - loss: 0.3254 - categorical_accuracy: 0.8997
19520/60000 [========>.....................] - ETA: 1:09 - loss: 0.3250 - categorical_accuracy: 0.8998
19552/60000 [========>.....................] - ETA: 1:09 - loss: 0.3246 - categorical_accuracy: 0.9000
19584/60000 [========>.....................] - ETA: 1:09 - loss: 0.3242 - categorical_accuracy: 0.9001
19616/60000 [========>.....................] - ETA: 1:09 - loss: 0.3238 - categorical_accuracy: 0.9002
19648/60000 [========>.....................] - ETA: 1:09 - loss: 0.3241 - categorical_accuracy: 0.9002
19680/60000 [========>.....................] - ETA: 1:09 - loss: 0.3238 - categorical_accuracy: 0.9003
19712/60000 [========>.....................] - ETA: 1:09 - loss: 0.3234 - categorical_accuracy: 0.9004
19744/60000 [========>.....................] - ETA: 1:09 - loss: 0.3229 - categorical_accuracy: 0.9005
19776/60000 [========>.....................] - ETA: 1:08 - loss: 0.3227 - categorical_accuracy: 0.9005
19808/60000 [========>.....................] - ETA: 1:08 - loss: 0.3225 - categorical_accuracy: 0.9005
19840/60000 [========>.....................] - ETA: 1:08 - loss: 0.3221 - categorical_accuracy: 0.9007
19904/60000 [========>.....................] - ETA: 1:08 - loss: 0.3213 - categorical_accuracy: 0.9009
19936/60000 [========>.....................] - ETA: 1:08 - loss: 0.3209 - categorical_accuracy: 0.9010
19968/60000 [========>.....................] - ETA: 1:08 - loss: 0.3208 - categorical_accuracy: 0.9010
20000/60000 [=========>....................] - ETA: 1:08 - loss: 0.3210 - categorical_accuracy: 0.9009
20032/60000 [=========>....................] - ETA: 1:08 - loss: 0.3205 - categorical_accuracy: 0.9011
20064/60000 [=========>....................] - ETA: 1:08 - loss: 0.3204 - categorical_accuracy: 0.9011
20096/60000 [=========>....................] - ETA: 1:08 - loss: 0.3202 - categorical_accuracy: 0.9012
20128/60000 [=========>....................] - ETA: 1:08 - loss: 0.3197 - categorical_accuracy: 0.9014
20160/60000 [=========>....................] - ETA: 1:08 - loss: 0.3197 - categorical_accuracy: 0.9014
20192/60000 [=========>....................] - ETA: 1:08 - loss: 0.3194 - categorical_accuracy: 0.9015
20224/60000 [=========>....................] - ETA: 1:08 - loss: 0.3191 - categorical_accuracy: 0.9016
20288/60000 [=========>....................] - ETA: 1:08 - loss: 0.3182 - categorical_accuracy: 0.9019
20320/60000 [=========>....................] - ETA: 1:08 - loss: 0.3179 - categorical_accuracy: 0.9019
20384/60000 [=========>....................] - ETA: 1:07 - loss: 0.3174 - categorical_accuracy: 0.9021
20416/60000 [=========>....................] - ETA: 1:07 - loss: 0.3170 - categorical_accuracy: 0.9022
20448/60000 [=========>....................] - ETA: 1:07 - loss: 0.3166 - categorical_accuracy: 0.9023
20480/60000 [=========>....................] - ETA: 1:07 - loss: 0.3170 - categorical_accuracy: 0.9023
20512/60000 [=========>....................] - ETA: 1:07 - loss: 0.3168 - categorical_accuracy: 0.9024
20544/60000 [=========>....................] - ETA: 1:07 - loss: 0.3164 - categorical_accuracy: 0.9026
20576/60000 [=========>....................] - ETA: 1:07 - loss: 0.3161 - categorical_accuracy: 0.9027
20608/60000 [=========>....................] - ETA: 1:07 - loss: 0.3157 - categorical_accuracy: 0.9029
20640/60000 [=========>....................] - ETA: 1:07 - loss: 0.3152 - categorical_accuracy: 0.9030
20672/60000 [=========>....................] - ETA: 1:07 - loss: 0.3149 - categorical_accuracy: 0.9032
20704/60000 [=========>....................] - ETA: 1:07 - loss: 0.3146 - categorical_accuracy: 0.9032
20736/60000 [=========>....................] - ETA: 1:07 - loss: 0.3142 - categorical_accuracy: 0.9033
20768/60000 [=========>....................] - ETA: 1:07 - loss: 0.3138 - categorical_accuracy: 0.9034
20832/60000 [=========>....................] - ETA: 1:07 - loss: 0.3133 - categorical_accuracy: 0.9036
20864/60000 [=========>....................] - ETA: 1:07 - loss: 0.3128 - categorical_accuracy: 0.9037
20896/60000 [=========>....................] - ETA: 1:07 - loss: 0.3125 - categorical_accuracy: 0.9038
20928/60000 [=========>....................] - ETA: 1:06 - loss: 0.3121 - categorical_accuracy: 0.9039
20960/60000 [=========>....................] - ETA: 1:06 - loss: 0.3118 - categorical_accuracy: 0.9040
20992/60000 [=========>....................] - ETA: 1:06 - loss: 0.3117 - categorical_accuracy: 0.9041
21024/60000 [=========>....................] - ETA: 1:06 - loss: 0.3113 - categorical_accuracy: 0.9042
21056/60000 [=========>....................] - ETA: 1:06 - loss: 0.3110 - categorical_accuracy: 0.9042
21088/60000 [=========>....................] - ETA: 1:06 - loss: 0.3108 - categorical_accuracy: 0.9043
21120/60000 [=========>....................] - ETA: 1:06 - loss: 0.3107 - categorical_accuracy: 0.9043
21152/60000 [=========>....................] - ETA: 1:06 - loss: 0.3104 - categorical_accuracy: 0.9043
21184/60000 [=========>....................] - ETA: 1:06 - loss: 0.3104 - categorical_accuracy: 0.9043
21216/60000 [=========>....................] - ETA: 1:06 - loss: 0.3102 - categorical_accuracy: 0.9044
21248/60000 [=========>....................] - ETA: 1:06 - loss: 0.3099 - categorical_accuracy: 0.9044
21312/60000 [=========>....................] - ETA: 1:06 - loss: 0.3093 - categorical_accuracy: 0.9046
21344/60000 [=========>....................] - ETA: 1:06 - loss: 0.3091 - categorical_accuracy: 0.9046
21376/60000 [=========>....................] - ETA: 1:06 - loss: 0.3088 - categorical_accuracy: 0.9047
21408/60000 [=========>....................] - ETA: 1:06 - loss: 0.3086 - categorical_accuracy: 0.9048
21440/60000 [=========>....................] - ETA: 1:06 - loss: 0.3082 - categorical_accuracy: 0.9049
21472/60000 [=========>....................] - ETA: 1:06 - loss: 0.3082 - categorical_accuracy: 0.9049
21504/60000 [=========>....................] - ETA: 1:05 - loss: 0.3078 - categorical_accuracy: 0.9050
21536/60000 [=========>....................] - ETA: 1:05 - loss: 0.3082 - categorical_accuracy: 0.9050
21568/60000 [=========>....................] - ETA: 1:05 - loss: 0.3080 - categorical_accuracy: 0.9051
21600/60000 [=========>....................] - ETA: 1:05 - loss: 0.3081 - categorical_accuracy: 0.9051
21632/60000 [=========>....................] - ETA: 1:05 - loss: 0.3078 - categorical_accuracy: 0.9052
21664/60000 [=========>....................] - ETA: 1:05 - loss: 0.3074 - categorical_accuracy: 0.9053
21728/60000 [=========>....................] - ETA: 1:05 - loss: 0.3069 - categorical_accuracy: 0.9054
21760/60000 [=========>....................] - ETA: 1:05 - loss: 0.3066 - categorical_accuracy: 0.9055
21792/60000 [=========>....................] - ETA: 1:05 - loss: 0.3064 - categorical_accuracy: 0.9055
21824/60000 [=========>....................] - ETA: 1:05 - loss: 0.3063 - categorical_accuracy: 0.9055
21856/60000 [=========>....................] - ETA: 1:05 - loss: 0.3063 - categorical_accuracy: 0.9056
21888/60000 [=========>....................] - ETA: 1:05 - loss: 0.3059 - categorical_accuracy: 0.9057
21920/60000 [=========>....................] - ETA: 1:05 - loss: 0.3055 - categorical_accuracy: 0.9059
21952/60000 [=========>....................] - ETA: 1:05 - loss: 0.3052 - categorical_accuracy: 0.9060
21984/60000 [=========>....................] - ETA: 1:05 - loss: 0.3048 - categorical_accuracy: 0.9061
22016/60000 [==========>...................] - ETA: 1:05 - loss: 0.3048 - categorical_accuracy: 0.9062
22048/60000 [==========>...................] - ETA: 1:05 - loss: 0.3046 - categorical_accuracy: 0.9062
22080/60000 [==========>...................] - ETA: 1:04 - loss: 0.3044 - categorical_accuracy: 0.9062
22112/60000 [==========>...................] - ETA: 1:04 - loss: 0.3041 - categorical_accuracy: 0.9063
22144/60000 [==========>...................] - ETA: 1:04 - loss: 0.3038 - categorical_accuracy: 0.9064
22176/60000 [==========>...................] - ETA: 1:04 - loss: 0.3035 - categorical_accuracy: 0.9065
22208/60000 [==========>...................] - ETA: 1:04 - loss: 0.3032 - categorical_accuracy: 0.9066
22240/60000 [==========>...................] - ETA: 1:04 - loss: 0.3030 - categorical_accuracy: 0.9067
22272/60000 [==========>...................] - ETA: 1:04 - loss: 0.3027 - categorical_accuracy: 0.9067
22304/60000 [==========>...................] - ETA: 1:04 - loss: 0.3024 - categorical_accuracy: 0.9068
22336/60000 [==========>...................] - ETA: 1:04 - loss: 0.3022 - categorical_accuracy: 0.9068
22368/60000 [==========>...................] - ETA: 1:04 - loss: 0.3019 - categorical_accuracy: 0.9069
22400/60000 [==========>...................] - ETA: 1:04 - loss: 0.3015 - categorical_accuracy: 0.9071
22432/60000 [==========>...................] - ETA: 1:04 - loss: 0.3015 - categorical_accuracy: 0.9071
22464/60000 [==========>...................] - ETA: 1:04 - loss: 0.3012 - categorical_accuracy: 0.9071
22496/60000 [==========>...................] - ETA: 1:04 - loss: 0.3008 - categorical_accuracy: 0.9073
22528/60000 [==========>...................] - ETA: 1:04 - loss: 0.3006 - categorical_accuracy: 0.9073
22560/60000 [==========>...................] - ETA: 1:04 - loss: 0.3004 - categorical_accuracy: 0.9074
22592/60000 [==========>...................] - ETA: 1:04 - loss: 0.3003 - categorical_accuracy: 0.9074
22624/60000 [==========>...................] - ETA: 1:04 - loss: 0.3003 - categorical_accuracy: 0.9074
22656/60000 [==========>...................] - ETA: 1:03 - loss: 0.3000 - categorical_accuracy: 0.9075
22688/60000 [==========>...................] - ETA: 1:03 - loss: 0.2997 - categorical_accuracy: 0.9076
22752/60000 [==========>...................] - ETA: 1:03 - loss: 0.2994 - categorical_accuracy: 0.9078
22784/60000 [==========>...................] - ETA: 1:03 - loss: 0.2991 - categorical_accuracy: 0.9078
22848/60000 [==========>...................] - ETA: 1:03 - loss: 0.2988 - categorical_accuracy: 0.9079
22912/60000 [==========>...................] - ETA: 1:03 - loss: 0.2982 - categorical_accuracy: 0.9080
22944/60000 [==========>...................] - ETA: 1:03 - loss: 0.2979 - categorical_accuracy: 0.9081
22976/60000 [==========>...................] - ETA: 1:03 - loss: 0.2978 - categorical_accuracy: 0.9081
23008/60000 [==========>...................] - ETA: 1:03 - loss: 0.2974 - categorical_accuracy: 0.9082
23040/60000 [==========>...................] - ETA: 1:03 - loss: 0.2974 - categorical_accuracy: 0.9083
23072/60000 [==========>...................] - ETA: 1:03 - loss: 0.2974 - categorical_accuracy: 0.9084
23104/60000 [==========>...................] - ETA: 1:03 - loss: 0.2972 - categorical_accuracy: 0.9084
23136/60000 [==========>...................] - ETA: 1:03 - loss: 0.2971 - categorical_accuracy: 0.9084
23168/60000 [==========>...................] - ETA: 1:03 - loss: 0.2969 - categorical_accuracy: 0.9085
23200/60000 [==========>...................] - ETA: 1:03 - loss: 0.2966 - categorical_accuracy: 0.9086
23232/60000 [==========>...................] - ETA: 1:02 - loss: 0.2964 - categorical_accuracy: 0.9087
23264/60000 [==========>...................] - ETA: 1:02 - loss: 0.2962 - categorical_accuracy: 0.9087
23296/60000 [==========>...................] - ETA: 1:02 - loss: 0.2963 - categorical_accuracy: 0.9087
23328/60000 [==========>...................] - ETA: 1:02 - loss: 0.2960 - categorical_accuracy: 0.9089
23360/60000 [==========>...................] - ETA: 1:02 - loss: 0.2957 - categorical_accuracy: 0.9090
23392/60000 [==========>...................] - ETA: 1:02 - loss: 0.2955 - categorical_accuracy: 0.9091
23424/60000 [==========>...................] - ETA: 1:02 - loss: 0.2953 - categorical_accuracy: 0.9091
23456/60000 [==========>...................] - ETA: 1:02 - loss: 0.2950 - categorical_accuracy: 0.9091
23488/60000 [==========>...................] - ETA: 1:02 - loss: 0.2950 - categorical_accuracy: 0.9091
23520/60000 [==========>...................] - ETA: 1:02 - loss: 0.2947 - categorical_accuracy: 0.9092
23552/60000 [==========>...................] - ETA: 1:02 - loss: 0.2945 - categorical_accuracy: 0.9093
23616/60000 [==========>...................] - ETA: 1:02 - loss: 0.2938 - categorical_accuracy: 0.9095
23648/60000 [==========>...................] - ETA: 1:02 - loss: 0.2935 - categorical_accuracy: 0.9096
23680/60000 [==========>...................] - ETA: 1:02 - loss: 0.2932 - categorical_accuracy: 0.9097
23712/60000 [==========>...................] - ETA: 1:02 - loss: 0.2930 - categorical_accuracy: 0.9097
23744/60000 [==========>...................] - ETA: 1:02 - loss: 0.2928 - categorical_accuracy: 0.9098
23776/60000 [==========>...................] - ETA: 1:02 - loss: 0.2927 - categorical_accuracy: 0.9099
23808/60000 [==========>...................] - ETA: 1:01 - loss: 0.2927 - categorical_accuracy: 0.9099
23840/60000 [==========>...................] - ETA: 1:01 - loss: 0.2923 - categorical_accuracy: 0.9100
23872/60000 [==========>...................] - ETA: 1:01 - loss: 0.2920 - categorical_accuracy: 0.9101
23904/60000 [==========>...................] - ETA: 1:01 - loss: 0.2917 - categorical_accuracy: 0.9102
23936/60000 [==========>...................] - ETA: 1:01 - loss: 0.2914 - categorical_accuracy: 0.9102
23968/60000 [==========>...................] - ETA: 1:01 - loss: 0.2915 - categorical_accuracy: 0.9102
24000/60000 [===========>..................] - ETA: 1:01 - loss: 0.2911 - categorical_accuracy: 0.9103
24032/60000 [===========>..................] - ETA: 1:01 - loss: 0.2913 - categorical_accuracy: 0.9103
24064/60000 [===========>..................] - ETA: 1:01 - loss: 0.2910 - categorical_accuracy: 0.9104
24096/60000 [===========>..................] - ETA: 1:01 - loss: 0.2906 - categorical_accuracy: 0.9105
24128/60000 [===========>..................] - ETA: 1:01 - loss: 0.2903 - categorical_accuracy: 0.9106
24192/60000 [===========>..................] - ETA: 1:01 - loss: 0.2896 - categorical_accuracy: 0.9109
24224/60000 [===========>..................] - ETA: 1:01 - loss: 0.2894 - categorical_accuracy: 0.9110
24256/60000 [===========>..................] - ETA: 1:01 - loss: 0.2890 - categorical_accuracy: 0.9111
24288/60000 [===========>..................] - ETA: 1:01 - loss: 0.2891 - categorical_accuracy: 0.9111
24320/60000 [===========>..................] - ETA: 1:01 - loss: 0.2889 - categorical_accuracy: 0.9111
24352/60000 [===========>..................] - ETA: 1:01 - loss: 0.2888 - categorical_accuracy: 0.9112
24384/60000 [===========>..................] - ETA: 1:00 - loss: 0.2886 - categorical_accuracy: 0.9113
24416/60000 [===========>..................] - ETA: 1:00 - loss: 0.2883 - categorical_accuracy: 0.9114
24448/60000 [===========>..................] - ETA: 1:00 - loss: 0.2882 - categorical_accuracy: 0.9114
24480/60000 [===========>..................] - ETA: 1:00 - loss: 0.2881 - categorical_accuracy: 0.9114
24512/60000 [===========>..................] - ETA: 1:00 - loss: 0.2878 - categorical_accuracy: 0.9115
24544/60000 [===========>..................] - ETA: 1:00 - loss: 0.2879 - categorical_accuracy: 0.9114
24576/60000 [===========>..................] - ETA: 1:00 - loss: 0.2877 - categorical_accuracy: 0.9114
24608/60000 [===========>..................] - ETA: 1:00 - loss: 0.2875 - categorical_accuracy: 0.9115
24640/60000 [===========>..................] - ETA: 1:00 - loss: 0.2875 - categorical_accuracy: 0.9115
24672/60000 [===========>..................] - ETA: 1:00 - loss: 0.2874 - categorical_accuracy: 0.9115
24704/60000 [===========>..................] - ETA: 1:00 - loss: 0.2874 - categorical_accuracy: 0.9115
24736/60000 [===========>..................] - ETA: 1:00 - loss: 0.2872 - categorical_accuracy: 0.9115
24800/60000 [===========>..................] - ETA: 1:00 - loss: 0.2866 - categorical_accuracy: 0.9117
24832/60000 [===========>..................] - ETA: 1:00 - loss: 0.2863 - categorical_accuracy: 0.9118
24864/60000 [===========>..................] - ETA: 1:00 - loss: 0.2860 - categorical_accuracy: 0.9119
24896/60000 [===========>..................] - ETA: 1:00 - loss: 0.2857 - categorical_accuracy: 0.9120
24928/60000 [===========>..................] - ETA: 1:00 - loss: 0.2856 - categorical_accuracy: 0.9120
24960/60000 [===========>..................] - ETA: 59s - loss: 0.2852 - categorical_accuracy: 0.9121 
24992/60000 [===========>..................] - ETA: 59s - loss: 0.2849 - categorical_accuracy: 0.9123
25024/60000 [===========>..................] - ETA: 59s - loss: 0.2848 - categorical_accuracy: 0.9123
25056/60000 [===========>..................] - ETA: 59s - loss: 0.2846 - categorical_accuracy: 0.9124
25088/60000 [===========>..................] - ETA: 59s - loss: 0.2843 - categorical_accuracy: 0.9124
25152/60000 [===========>..................] - ETA: 59s - loss: 0.2839 - categorical_accuracy: 0.9125
25184/60000 [===========>..................] - ETA: 59s - loss: 0.2838 - categorical_accuracy: 0.9126
25216/60000 [===========>..................] - ETA: 59s - loss: 0.2836 - categorical_accuracy: 0.9126
25248/60000 [===========>..................] - ETA: 59s - loss: 0.2833 - categorical_accuracy: 0.9127
25280/60000 [===========>..................] - ETA: 59s - loss: 0.2832 - categorical_accuracy: 0.9127
25312/60000 [===========>..................] - ETA: 59s - loss: 0.2831 - categorical_accuracy: 0.9127
25344/60000 [===========>..................] - ETA: 59s - loss: 0.2829 - categorical_accuracy: 0.9127
25376/60000 [===========>..................] - ETA: 59s - loss: 0.2829 - categorical_accuracy: 0.9127
25408/60000 [===========>..................] - ETA: 59s - loss: 0.2827 - categorical_accuracy: 0.9127
25440/60000 [===========>..................] - ETA: 59s - loss: 0.2824 - categorical_accuracy: 0.9128
25472/60000 [===========>..................] - ETA: 59s - loss: 0.2822 - categorical_accuracy: 0.9129
25536/60000 [===========>..................] - ETA: 58s - loss: 0.2820 - categorical_accuracy: 0.9130
25568/60000 [===========>..................] - ETA: 58s - loss: 0.2819 - categorical_accuracy: 0.9130
25600/60000 [===========>..................] - ETA: 58s - loss: 0.2817 - categorical_accuracy: 0.9130
25632/60000 [===========>..................] - ETA: 58s - loss: 0.2817 - categorical_accuracy: 0.9131
25664/60000 [===========>..................] - ETA: 58s - loss: 0.2815 - categorical_accuracy: 0.9131
25696/60000 [===========>..................] - ETA: 58s - loss: 0.2813 - categorical_accuracy: 0.9131
25728/60000 [===========>..................] - ETA: 58s - loss: 0.2814 - categorical_accuracy: 0.9131
25760/60000 [===========>..................] - ETA: 58s - loss: 0.2811 - categorical_accuracy: 0.9132
25824/60000 [===========>..................] - ETA: 58s - loss: 0.2809 - categorical_accuracy: 0.9133
25888/60000 [===========>..................] - ETA: 58s - loss: 0.2804 - categorical_accuracy: 0.9134
25920/60000 [===========>..................] - ETA: 58s - loss: 0.2803 - categorical_accuracy: 0.9135
25984/60000 [===========>..................] - ETA: 58s - loss: 0.2799 - categorical_accuracy: 0.9136
26016/60000 [============>.................] - ETA: 58s - loss: 0.2799 - categorical_accuracy: 0.9136
26048/60000 [============>.................] - ETA: 58s - loss: 0.2797 - categorical_accuracy: 0.9137
26080/60000 [============>.................] - ETA: 57s - loss: 0.2796 - categorical_accuracy: 0.9137
26112/60000 [============>.................] - ETA: 57s - loss: 0.2793 - categorical_accuracy: 0.9138
26144/60000 [============>.................] - ETA: 57s - loss: 0.2790 - categorical_accuracy: 0.9139
26176/60000 [============>.................] - ETA: 57s - loss: 0.2787 - categorical_accuracy: 0.9140
26208/60000 [============>.................] - ETA: 57s - loss: 0.2787 - categorical_accuracy: 0.9141
26240/60000 [============>.................] - ETA: 57s - loss: 0.2787 - categorical_accuracy: 0.9141
26272/60000 [============>.................] - ETA: 57s - loss: 0.2785 - categorical_accuracy: 0.9141
26304/60000 [============>.................] - ETA: 57s - loss: 0.2786 - categorical_accuracy: 0.9141
26336/60000 [============>.................] - ETA: 57s - loss: 0.2787 - categorical_accuracy: 0.9140
26368/60000 [============>.................] - ETA: 57s - loss: 0.2784 - categorical_accuracy: 0.9141
26400/60000 [============>.................] - ETA: 57s - loss: 0.2784 - categorical_accuracy: 0.9142
26432/60000 [============>.................] - ETA: 57s - loss: 0.2781 - categorical_accuracy: 0.9143
26464/60000 [============>.................] - ETA: 57s - loss: 0.2778 - categorical_accuracy: 0.9143
26496/60000 [============>.................] - ETA: 57s - loss: 0.2776 - categorical_accuracy: 0.9144
26528/60000 [============>.................] - ETA: 57s - loss: 0.2776 - categorical_accuracy: 0.9144
26560/60000 [============>.................] - ETA: 57s - loss: 0.2775 - categorical_accuracy: 0.9144
26592/60000 [============>.................] - ETA: 57s - loss: 0.2773 - categorical_accuracy: 0.9144
26624/60000 [============>.................] - ETA: 57s - loss: 0.2771 - categorical_accuracy: 0.9145
26656/60000 [============>.................] - ETA: 57s - loss: 0.2768 - categorical_accuracy: 0.9146
26688/60000 [============>.................] - ETA: 56s - loss: 0.2765 - categorical_accuracy: 0.9147
26720/60000 [============>.................] - ETA: 56s - loss: 0.2764 - categorical_accuracy: 0.9147
26752/60000 [============>.................] - ETA: 56s - loss: 0.2761 - categorical_accuracy: 0.9148
26784/60000 [============>.................] - ETA: 56s - loss: 0.2762 - categorical_accuracy: 0.9148
26816/60000 [============>.................] - ETA: 56s - loss: 0.2759 - categorical_accuracy: 0.9149
26880/60000 [============>.................] - ETA: 56s - loss: 0.2755 - categorical_accuracy: 0.9150
26912/60000 [============>.................] - ETA: 56s - loss: 0.2753 - categorical_accuracy: 0.9151
26976/60000 [============>.................] - ETA: 56s - loss: 0.2747 - categorical_accuracy: 0.9153
27008/60000 [============>.................] - ETA: 56s - loss: 0.2746 - categorical_accuracy: 0.9154
27040/60000 [============>.................] - ETA: 56s - loss: 0.2745 - categorical_accuracy: 0.9154
27072/60000 [============>.................] - ETA: 56s - loss: 0.2742 - categorical_accuracy: 0.9155
27104/60000 [============>.................] - ETA: 56s - loss: 0.2740 - categorical_accuracy: 0.9155
27136/60000 [============>.................] - ETA: 56s - loss: 0.2738 - categorical_accuracy: 0.9156
27168/60000 [============>.................] - ETA: 56s - loss: 0.2737 - categorical_accuracy: 0.9156
27200/60000 [============>.................] - ETA: 56s - loss: 0.2735 - categorical_accuracy: 0.9157
27232/60000 [============>.................] - ETA: 55s - loss: 0.2733 - categorical_accuracy: 0.9157
27296/60000 [============>.................] - ETA: 55s - loss: 0.2727 - categorical_accuracy: 0.9159
27328/60000 [============>.................] - ETA: 55s - loss: 0.2724 - categorical_accuracy: 0.9160
27360/60000 [============>.................] - ETA: 55s - loss: 0.2722 - categorical_accuracy: 0.9160
27392/60000 [============>.................] - ETA: 55s - loss: 0.2721 - categorical_accuracy: 0.9161
27424/60000 [============>.................] - ETA: 55s - loss: 0.2721 - categorical_accuracy: 0.9161
27456/60000 [============>.................] - ETA: 55s - loss: 0.2718 - categorical_accuracy: 0.9161
27488/60000 [============>.................] - ETA: 55s - loss: 0.2716 - categorical_accuracy: 0.9162
27520/60000 [============>.................] - ETA: 55s - loss: 0.2716 - categorical_accuracy: 0.9162
27552/60000 [============>.................] - ETA: 55s - loss: 0.2713 - categorical_accuracy: 0.9163
27584/60000 [============>.................] - ETA: 55s - loss: 0.2712 - categorical_accuracy: 0.9164
27616/60000 [============>.................] - ETA: 55s - loss: 0.2712 - categorical_accuracy: 0.9163
27648/60000 [============>.................] - ETA: 55s - loss: 0.2711 - categorical_accuracy: 0.9163
27680/60000 [============>.................] - ETA: 55s - loss: 0.2708 - categorical_accuracy: 0.9164
27712/60000 [============>.................] - ETA: 55s - loss: 0.2706 - categorical_accuracy: 0.9165
27744/60000 [============>.................] - ETA: 55s - loss: 0.2705 - categorical_accuracy: 0.9165
27776/60000 [============>.................] - ETA: 55s - loss: 0.2704 - categorical_accuracy: 0.9165
27808/60000 [============>.................] - ETA: 54s - loss: 0.2701 - categorical_accuracy: 0.9166
27840/60000 [============>.................] - ETA: 54s - loss: 0.2700 - categorical_accuracy: 0.9167
27872/60000 [============>.................] - ETA: 54s - loss: 0.2697 - categorical_accuracy: 0.9167
27904/60000 [============>.................] - ETA: 54s - loss: 0.2695 - categorical_accuracy: 0.9168
27936/60000 [============>.................] - ETA: 54s - loss: 0.2694 - categorical_accuracy: 0.9168
27968/60000 [============>.................] - ETA: 54s - loss: 0.2692 - categorical_accuracy: 0.9169
28000/60000 [=============>................] - ETA: 54s - loss: 0.2689 - categorical_accuracy: 0.9170
28032/60000 [=============>................] - ETA: 54s - loss: 0.2686 - categorical_accuracy: 0.9171
28064/60000 [=============>................] - ETA: 54s - loss: 0.2686 - categorical_accuracy: 0.9170
28096/60000 [=============>................] - ETA: 54s - loss: 0.2684 - categorical_accuracy: 0.9171
28128/60000 [=============>................] - ETA: 54s - loss: 0.2684 - categorical_accuracy: 0.9172
28160/60000 [=============>................] - ETA: 54s - loss: 0.2684 - categorical_accuracy: 0.9172
28192/60000 [=============>................] - ETA: 54s - loss: 0.2683 - categorical_accuracy: 0.9172
28224/60000 [=============>................] - ETA: 54s - loss: 0.2684 - categorical_accuracy: 0.9172
28256/60000 [=============>................] - ETA: 54s - loss: 0.2683 - categorical_accuracy: 0.9172
28288/60000 [=============>................] - ETA: 54s - loss: 0.2682 - categorical_accuracy: 0.9173
28320/60000 [=============>................] - ETA: 54s - loss: 0.2681 - categorical_accuracy: 0.9173
28352/60000 [=============>................] - ETA: 54s - loss: 0.2679 - categorical_accuracy: 0.9174
28416/60000 [=============>................] - ETA: 53s - loss: 0.2675 - categorical_accuracy: 0.9175
28448/60000 [=============>................] - ETA: 53s - loss: 0.2673 - categorical_accuracy: 0.9176
28480/60000 [=============>................] - ETA: 53s - loss: 0.2671 - categorical_accuracy: 0.9177
28512/60000 [=============>................] - ETA: 53s - loss: 0.2669 - categorical_accuracy: 0.9177
28544/60000 [=============>................] - ETA: 53s - loss: 0.2669 - categorical_accuracy: 0.9177
28576/60000 [=============>................] - ETA: 53s - loss: 0.2666 - categorical_accuracy: 0.9178
28608/60000 [=============>................] - ETA: 53s - loss: 0.2664 - categorical_accuracy: 0.9179
28640/60000 [=============>................] - ETA: 53s - loss: 0.2664 - categorical_accuracy: 0.9179
28672/60000 [=============>................] - ETA: 53s - loss: 0.2664 - categorical_accuracy: 0.9180
28704/60000 [=============>................] - ETA: 53s - loss: 0.2662 - categorical_accuracy: 0.9180
28736/60000 [=============>................] - ETA: 53s - loss: 0.2659 - categorical_accuracy: 0.9181
28768/60000 [=============>................] - ETA: 53s - loss: 0.2657 - categorical_accuracy: 0.9182
28800/60000 [=============>................] - ETA: 53s - loss: 0.2654 - categorical_accuracy: 0.9183
28832/60000 [=============>................] - ETA: 53s - loss: 0.2652 - categorical_accuracy: 0.9184
28864/60000 [=============>................] - ETA: 53s - loss: 0.2651 - categorical_accuracy: 0.9184
28896/60000 [=============>................] - ETA: 53s - loss: 0.2650 - categorical_accuracy: 0.9185
28928/60000 [=============>................] - ETA: 53s - loss: 0.2647 - categorical_accuracy: 0.9186
28992/60000 [=============>................] - ETA: 52s - loss: 0.2643 - categorical_accuracy: 0.9187
29024/60000 [=============>................] - ETA: 52s - loss: 0.2641 - categorical_accuracy: 0.9188
29056/60000 [=============>................] - ETA: 52s - loss: 0.2638 - categorical_accuracy: 0.9188
29088/60000 [=============>................] - ETA: 52s - loss: 0.2639 - categorical_accuracy: 0.9189
29120/60000 [=============>................] - ETA: 52s - loss: 0.2638 - categorical_accuracy: 0.9189
29152/60000 [=============>................] - ETA: 52s - loss: 0.2636 - categorical_accuracy: 0.9190
29184/60000 [=============>................] - ETA: 52s - loss: 0.2634 - categorical_accuracy: 0.9191
29216/60000 [=============>................] - ETA: 52s - loss: 0.2632 - categorical_accuracy: 0.9192
29248/60000 [=============>................] - ETA: 52s - loss: 0.2629 - categorical_accuracy: 0.9192
29280/60000 [=============>................] - ETA: 52s - loss: 0.2628 - categorical_accuracy: 0.9193
29344/60000 [=============>................] - ETA: 52s - loss: 0.2624 - categorical_accuracy: 0.9194
29376/60000 [=============>................] - ETA: 52s - loss: 0.2622 - categorical_accuracy: 0.9194
29408/60000 [=============>................] - ETA: 52s - loss: 0.2622 - categorical_accuracy: 0.9194
29440/60000 [=============>................] - ETA: 52s - loss: 0.2622 - categorical_accuracy: 0.9194
29472/60000 [=============>................] - ETA: 52s - loss: 0.2621 - categorical_accuracy: 0.9194
29504/60000 [=============>................] - ETA: 52s - loss: 0.2618 - categorical_accuracy: 0.9195
29536/60000 [=============>................] - ETA: 52s - loss: 0.2617 - categorical_accuracy: 0.9196
29568/60000 [=============>................] - ETA: 51s - loss: 0.2614 - categorical_accuracy: 0.9196
29600/60000 [=============>................] - ETA: 51s - loss: 0.2615 - categorical_accuracy: 0.9197
29632/60000 [=============>................] - ETA: 51s - loss: 0.2614 - categorical_accuracy: 0.9196
29664/60000 [=============>................] - ETA: 51s - loss: 0.2612 - categorical_accuracy: 0.9197
29696/60000 [=============>................] - ETA: 51s - loss: 0.2610 - categorical_accuracy: 0.9198
29728/60000 [=============>................] - ETA: 51s - loss: 0.2608 - categorical_accuracy: 0.9198
29760/60000 [=============>................] - ETA: 51s - loss: 0.2606 - categorical_accuracy: 0.9199
29792/60000 [=============>................] - ETA: 51s - loss: 0.2604 - categorical_accuracy: 0.9200
29824/60000 [=============>................] - ETA: 51s - loss: 0.2602 - categorical_accuracy: 0.9200
29856/60000 [=============>................] - ETA: 51s - loss: 0.2602 - categorical_accuracy: 0.9200
29888/60000 [=============>................] - ETA: 51s - loss: 0.2601 - categorical_accuracy: 0.9201
29920/60000 [=============>................] - ETA: 51s - loss: 0.2599 - categorical_accuracy: 0.9201
29952/60000 [=============>................] - ETA: 51s - loss: 0.2600 - categorical_accuracy: 0.9201
29984/60000 [=============>................] - ETA: 51s - loss: 0.2598 - categorical_accuracy: 0.9201
30016/60000 [==============>...............] - ETA: 51s - loss: 0.2596 - categorical_accuracy: 0.9202
30048/60000 [==============>...............] - ETA: 51s - loss: 0.2595 - categorical_accuracy: 0.9202
30080/60000 [==============>...............] - ETA: 51s - loss: 0.2593 - categorical_accuracy: 0.9203
30112/60000 [==============>...............] - ETA: 51s - loss: 0.2590 - categorical_accuracy: 0.9204
30144/60000 [==============>...............] - ETA: 51s - loss: 0.2590 - categorical_accuracy: 0.9204
30176/60000 [==============>...............] - ETA: 50s - loss: 0.2588 - categorical_accuracy: 0.9205
30208/60000 [==============>...............] - ETA: 50s - loss: 0.2586 - categorical_accuracy: 0.9205
30240/60000 [==============>...............] - ETA: 50s - loss: 0.2584 - categorical_accuracy: 0.9206
30272/60000 [==============>...............] - ETA: 50s - loss: 0.2582 - categorical_accuracy: 0.9207
30304/60000 [==============>...............] - ETA: 50s - loss: 0.2580 - categorical_accuracy: 0.9207
30336/60000 [==============>...............] - ETA: 50s - loss: 0.2578 - categorical_accuracy: 0.9208
30368/60000 [==============>...............] - ETA: 50s - loss: 0.2577 - categorical_accuracy: 0.9208
30400/60000 [==============>...............] - ETA: 50s - loss: 0.2574 - categorical_accuracy: 0.9209
30432/60000 [==============>...............] - ETA: 50s - loss: 0.2576 - categorical_accuracy: 0.9208
30464/60000 [==============>...............] - ETA: 50s - loss: 0.2575 - categorical_accuracy: 0.9209
30528/60000 [==============>...............] - ETA: 50s - loss: 0.2573 - categorical_accuracy: 0.9208
30560/60000 [==============>...............] - ETA: 50s - loss: 0.2572 - categorical_accuracy: 0.9209
30624/60000 [==============>...............] - ETA: 50s - loss: 0.2572 - categorical_accuracy: 0.9209
30656/60000 [==============>...............] - ETA: 50s - loss: 0.2571 - categorical_accuracy: 0.9210
30688/60000 [==============>...............] - ETA: 50s - loss: 0.2569 - categorical_accuracy: 0.9210
30720/60000 [==============>...............] - ETA: 50s - loss: 0.2568 - categorical_accuracy: 0.9210
30784/60000 [==============>...............] - ETA: 49s - loss: 0.2565 - categorical_accuracy: 0.9211
30816/60000 [==============>...............] - ETA: 49s - loss: 0.2563 - categorical_accuracy: 0.9212
30880/60000 [==============>...............] - ETA: 49s - loss: 0.2563 - categorical_accuracy: 0.9212
30944/60000 [==============>...............] - ETA: 49s - loss: 0.2560 - categorical_accuracy: 0.9212
30976/60000 [==============>...............] - ETA: 49s - loss: 0.2559 - categorical_accuracy: 0.9212
31008/60000 [==============>...............] - ETA: 49s - loss: 0.2558 - categorical_accuracy: 0.9213
31040/60000 [==============>...............] - ETA: 49s - loss: 0.2555 - categorical_accuracy: 0.9214
31072/60000 [==============>...............] - ETA: 49s - loss: 0.2553 - categorical_accuracy: 0.9214
31104/60000 [==============>...............] - ETA: 49s - loss: 0.2551 - categorical_accuracy: 0.9215
31136/60000 [==============>...............] - ETA: 49s - loss: 0.2549 - categorical_accuracy: 0.9215
31168/60000 [==============>...............] - ETA: 49s - loss: 0.2547 - categorical_accuracy: 0.9216
31200/60000 [==============>...............] - ETA: 49s - loss: 0.2546 - categorical_accuracy: 0.9216
31264/60000 [==============>...............] - ETA: 49s - loss: 0.2543 - categorical_accuracy: 0.9217
31296/60000 [==============>...............] - ETA: 48s - loss: 0.2540 - categorical_accuracy: 0.9218
31328/60000 [==============>...............] - ETA: 48s - loss: 0.2539 - categorical_accuracy: 0.9219
31360/60000 [==============>...............] - ETA: 48s - loss: 0.2537 - categorical_accuracy: 0.9219
31392/60000 [==============>...............] - ETA: 48s - loss: 0.2537 - categorical_accuracy: 0.9219
31424/60000 [==============>...............] - ETA: 48s - loss: 0.2538 - categorical_accuracy: 0.9218
31456/60000 [==============>...............] - ETA: 48s - loss: 0.2536 - categorical_accuracy: 0.9219
31488/60000 [==============>...............] - ETA: 48s - loss: 0.2537 - categorical_accuracy: 0.9219
31520/60000 [==============>...............] - ETA: 48s - loss: 0.2536 - categorical_accuracy: 0.9219
31552/60000 [==============>...............] - ETA: 48s - loss: 0.2534 - categorical_accuracy: 0.9219
31584/60000 [==============>...............] - ETA: 48s - loss: 0.2533 - categorical_accuracy: 0.9220
31616/60000 [==============>...............] - ETA: 48s - loss: 0.2530 - categorical_accuracy: 0.9221
31648/60000 [==============>...............] - ETA: 48s - loss: 0.2528 - categorical_accuracy: 0.9221
31680/60000 [==============>...............] - ETA: 48s - loss: 0.2526 - categorical_accuracy: 0.9222
31712/60000 [==============>...............] - ETA: 48s - loss: 0.2525 - categorical_accuracy: 0.9222
31744/60000 [==============>...............] - ETA: 48s - loss: 0.2525 - categorical_accuracy: 0.9223
31776/60000 [==============>...............] - ETA: 48s - loss: 0.2523 - categorical_accuracy: 0.9223
31808/60000 [==============>...............] - ETA: 48s - loss: 0.2522 - categorical_accuracy: 0.9224
31840/60000 [==============>...............] - ETA: 48s - loss: 0.2521 - categorical_accuracy: 0.9224
31872/60000 [==============>...............] - ETA: 48s - loss: 0.2519 - categorical_accuracy: 0.9225
31904/60000 [==============>...............] - ETA: 47s - loss: 0.2518 - categorical_accuracy: 0.9225
31936/60000 [==============>...............] - ETA: 47s - loss: 0.2516 - categorical_accuracy: 0.9226
31968/60000 [==============>...............] - ETA: 47s - loss: 0.2514 - categorical_accuracy: 0.9226
32000/60000 [===============>..............] - ETA: 47s - loss: 0.2512 - categorical_accuracy: 0.9227
32032/60000 [===============>..............] - ETA: 47s - loss: 0.2510 - categorical_accuracy: 0.9228
32096/60000 [===============>..............] - ETA: 47s - loss: 0.2506 - categorical_accuracy: 0.9229
32128/60000 [===============>..............] - ETA: 47s - loss: 0.2504 - categorical_accuracy: 0.9229
32160/60000 [===============>..............] - ETA: 47s - loss: 0.2502 - categorical_accuracy: 0.9230
32192/60000 [===============>..............] - ETA: 47s - loss: 0.2500 - categorical_accuracy: 0.9231
32224/60000 [===============>..............] - ETA: 47s - loss: 0.2499 - categorical_accuracy: 0.9231
32256/60000 [===============>..............] - ETA: 47s - loss: 0.2497 - categorical_accuracy: 0.9232
32288/60000 [===============>..............] - ETA: 47s - loss: 0.2495 - categorical_accuracy: 0.9233
32320/60000 [===============>..............] - ETA: 47s - loss: 0.2494 - categorical_accuracy: 0.9233
32352/60000 [===============>..............] - ETA: 47s - loss: 0.2494 - categorical_accuracy: 0.9233
32384/60000 [===============>..............] - ETA: 47s - loss: 0.2493 - categorical_accuracy: 0.9233
32416/60000 [===============>..............] - ETA: 47s - loss: 0.2491 - categorical_accuracy: 0.9234
32448/60000 [===============>..............] - ETA: 47s - loss: 0.2489 - categorical_accuracy: 0.9234
32480/60000 [===============>..............] - ETA: 46s - loss: 0.2487 - categorical_accuracy: 0.9235
32512/60000 [===============>..............] - ETA: 46s - loss: 0.2486 - categorical_accuracy: 0.9236
32544/60000 [===============>..............] - ETA: 46s - loss: 0.2488 - categorical_accuracy: 0.9235
32576/60000 [===============>..............] - ETA: 46s - loss: 0.2488 - categorical_accuracy: 0.9235
32608/60000 [===============>..............] - ETA: 46s - loss: 0.2491 - categorical_accuracy: 0.9235
32640/60000 [===============>..............] - ETA: 46s - loss: 0.2489 - categorical_accuracy: 0.9236
32672/60000 [===============>..............] - ETA: 46s - loss: 0.2487 - categorical_accuracy: 0.9237
32704/60000 [===============>..............] - ETA: 46s - loss: 0.2486 - categorical_accuracy: 0.9237
32736/60000 [===============>..............] - ETA: 46s - loss: 0.2485 - categorical_accuracy: 0.9237
32768/60000 [===============>..............] - ETA: 46s - loss: 0.2484 - categorical_accuracy: 0.9237
32800/60000 [===============>..............] - ETA: 46s - loss: 0.2484 - categorical_accuracy: 0.9237
32832/60000 [===============>..............] - ETA: 46s - loss: 0.2485 - categorical_accuracy: 0.9237
32896/60000 [===============>..............] - ETA: 46s - loss: 0.2483 - categorical_accuracy: 0.9238
32928/60000 [===============>..............] - ETA: 46s - loss: 0.2481 - categorical_accuracy: 0.9238
32960/60000 [===============>..............] - ETA: 46s - loss: 0.2481 - categorical_accuracy: 0.9238
32992/60000 [===============>..............] - ETA: 46s - loss: 0.2480 - categorical_accuracy: 0.9239
33024/60000 [===============>..............] - ETA: 46s - loss: 0.2481 - categorical_accuracy: 0.9239
33056/60000 [===============>..............] - ETA: 46s - loss: 0.2480 - categorical_accuracy: 0.9240
33088/60000 [===============>..............] - ETA: 45s - loss: 0.2478 - categorical_accuracy: 0.9241
33120/60000 [===============>..............] - ETA: 45s - loss: 0.2476 - categorical_accuracy: 0.9241
33152/60000 [===============>..............] - ETA: 45s - loss: 0.2475 - categorical_accuracy: 0.9241
33184/60000 [===============>..............] - ETA: 45s - loss: 0.2473 - categorical_accuracy: 0.9242
33216/60000 [===============>..............] - ETA: 45s - loss: 0.2471 - categorical_accuracy: 0.9243
33248/60000 [===============>..............] - ETA: 45s - loss: 0.2469 - categorical_accuracy: 0.9243
33280/60000 [===============>..............] - ETA: 45s - loss: 0.2467 - categorical_accuracy: 0.9244
33312/60000 [===============>..............] - ETA: 45s - loss: 0.2465 - categorical_accuracy: 0.9245
33344/60000 [===============>..............] - ETA: 45s - loss: 0.2463 - categorical_accuracy: 0.9245
33376/60000 [===============>..............] - ETA: 45s - loss: 0.2461 - categorical_accuracy: 0.9246
33408/60000 [===============>..............] - ETA: 45s - loss: 0.2462 - categorical_accuracy: 0.9245
33440/60000 [===============>..............] - ETA: 45s - loss: 0.2460 - categorical_accuracy: 0.9246
33472/60000 [===============>..............] - ETA: 45s - loss: 0.2458 - categorical_accuracy: 0.9247
33504/60000 [===============>..............] - ETA: 45s - loss: 0.2456 - categorical_accuracy: 0.9248
33536/60000 [===============>..............] - ETA: 45s - loss: 0.2455 - categorical_accuracy: 0.9248
33568/60000 [===============>..............] - ETA: 45s - loss: 0.2454 - categorical_accuracy: 0.9248
33600/60000 [===============>..............] - ETA: 45s - loss: 0.2452 - categorical_accuracy: 0.9249
33632/60000 [===============>..............] - ETA: 45s - loss: 0.2451 - categorical_accuracy: 0.9249
33664/60000 [===============>..............] - ETA: 44s - loss: 0.2449 - categorical_accuracy: 0.9250
33696/60000 [===============>..............] - ETA: 44s - loss: 0.2447 - categorical_accuracy: 0.9250
33728/60000 [===============>..............] - ETA: 44s - loss: 0.2447 - categorical_accuracy: 0.9251
33760/60000 [===============>..............] - ETA: 44s - loss: 0.2445 - categorical_accuracy: 0.9251
33824/60000 [===============>..............] - ETA: 44s - loss: 0.2443 - categorical_accuracy: 0.9252
33856/60000 [===============>..............] - ETA: 44s - loss: 0.2443 - categorical_accuracy: 0.9252
33888/60000 [===============>..............] - ETA: 44s - loss: 0.2442 - categorical_accuracy: 0.9253
33920/60000 [===============>..............] - ETA: 44s - loss: 0.2440 - categorical_accuracy: 0.9253
33952/60000 [===============>..............] - ETA: 44s - loss: 0.2439 - categorical_accuracy: 0.9253
33984/60000 [===============>..............] - ETA: 44s - loss: 0.2440 - categorical_accuracy: 0.9253
34016/60000 [================>.............] - ETA: 44s - loss: 0.2439 - categorical_accuracy: 0.9254
34048/60000 [================>.............] - ETA: 44s - loss: 0.2440 - categorical_accuracy: 0.9253
34080/60000 [================>.............] - ETA: 44s - loss: 0.2437 - categorical_accuracy: 0.9254
34112/60000 [================>.............] - ETA: 44s - loss: 0.2435 - categorical_accuracy: 0.9254
34144/60000 [================>.............] - ETA: 44s - loss: 0.2433 - categorical_accuracy: 0.9255
34176/60000 [================>.............] - ETA: 44s - loss: 0.2433 - categorical_accuracy: 0.9255
34208/60000 [================>.............] - ETA: 44s - loss: 0.2432 - categorical_accuracy: 0.9256
34240/60000 [================>.............] - ETA: 43s - loss: 0.2431 - categorical_accuracy: 0.9256
34272/60000 [================>.............] - ETA: 43s - loss: 0.2429 - categorical_accuracy: 0.9256
34304/60000 [================>.............] - ETA: 43s - loss: 0.2427 - categorical_accuracy: 0.9257
34368/60000 [================>.............] - ETA: 43s - loss: 0.2425 - categorical_accuracy: 0.9257
34400/60000 [================>.............] - ETA: 43s - loss: 0.2424 - categorical_accuracy: 0.9258
34432/60000 [================>.............] - ETA: 43s - loss: 0.2423 - categorical_accuracy: 0.9258
34464/60000 [================>.............] - ETA: 43s - loss: 0.2421 - categorical_accuracy: 0.9258
34496/60000 [================>.............] - ETA: 43s - loss: 0.2420 - categorical_accuracy: 0.9258
34528/60000 [================>.............] - ETA: 43s - loss: 0.2418 - categorical_accuracy: 0.9259
34560/60000 [================>.............] - ETA: 43s - loss: 0.2416 - categorical_accuracy: 0.9259
34592/60000 [================>.............] - ETA: 43s - loss: 0.2415 - categorical_accuracy: 0.9259
34624/60000 [================>.............] - ETA: 43s - loss: 0.2413 - categorical_accuracy: 0.9260
34656/60000 [================>.............] - ETA: 43s - loss: 0.2412 - categorical_accuracy: 0.9260
34688/60000 [================>.............] - ETA: 43s - loss: 0.2411 - categorical_accuracy: 0.9260
34720/60000 [================>.............] - ETA: 43s - loss: 0.2409 - categorical_accuracy: 0.9261
34784/60000 [================>.............] - ETA: 43s - loss: 0.2407 - categorical_accuracy: 0.9261
34816/60000 [================>.............] - ETA: 43s - loss: 0.2405 - categorical_accuracy: 0.9262
34848/60000 [================>.............] - ETA: 42s - loss: 0.2404 - categorical_accuracy: 0.9263
34880/60000 [================>.............] - ETA: 42s - loss: 0.2404 - categorical_accuracy: 0.9262
34912/60000 [================>.............] - ETA: 42s - loss: 0.2405 - categorical_accuracy: 0.9262
34944/60000 [================>.............] - ETA: 42s - loss: 0.2403 - categorical_accuracy: 0.9263
34976/60000 [================>.............] - ETA: 42s - loss: 0.2403 - categorical_accuracy: 0.9263
35008/60000 [================>.............] - ETA: 42s - loss: 0.2402 - categorical_accuracy: 0.9264
35040/60000 [================>.............] - ETA: 42s - loss: 0.2401 - categorical_accuracy: 0.9264
35072/60000 [================>.............] - ETA: 42s - loss: 0.2399 - categorical_accuracy: 0.9265
35104/60000 [================>.............] - ETA: 42s - loss: 0.2397 - categorical_accuracy: 0.9265
35136/60000 [================>.............] - ETA: 42s - loss: 0.2398 - categorical_accuracy: 0.9265
35168/60000 [================>.............] - ETA: 42s - loss: 0.2399 - categorical_accuracy: 0.9265
35200/60000 [================>.............] - ETA: 42s - loss: 0.2397 - categorical_accuracy: 0.9265
35232/60000 [================>.............] - ETA: 42s - loss: 0.2396 - categorical_accuracy: 0.9265
35264/60000 [================>.............] - ETA: 42s - loss: 0.2395 - categorical_accuracy: 0.9266
35296/60000 [================>.............] - ETA: 42s - loss: 0.2393 - categorical_accuracy: 0.9266
35328/60000 [================>.............] - ETA: 42s - loss: 0.2391 - categorical_accuracy: 0.9267
35360/60000 [================>.............] - ETA: 42s - loss: 0.2389 - categorical_accuracy: 0.9268
35392/60000 [================>.............] - ETA: 42s - loss: 0.2388 - categorical_accuracy: 0.9268
35424/60000 [================>.............] - ETA: 41s - loss: 0.2387 - categorical_accuracy: 0.9269
35456/60000 [================>.............] - ETA: 41s - loss: 0.2385 - categorical_accuracy: 0.9270
35488/60000 [================>.............] - ETA: 41s - loss: 0.2384 - categorical_accuracy: 0.9270
35520/60000 [================>.............] - ETA: 41s - loss: 0.2382 - categorical_accuracy: 0.9271
35552/60000 [================>.............] - ETA: 41s - loss: 0.2380 - categorical_accuracy: 0.9271
35584/60000 [================>.............] - ETA: 41s - loss: 0.2379 - categorical_accuracy: 0.9271
35616/60000 [================>.............] - ETA: 41s - loss: 0.2377 - categorical_accuracy: 0.9272
35648/60000 [================>.............] - ETA: 41s - loss: 0.2375 - categorical_accuracy: 0.9272
35680/60000 [================>.............] - ETA: 41s - loss: 0.2375 - categorical_accuracy: 0.9272
35712/60000 [================>.............] - ETA: 41s - loss: 0.2375 - categorical_accuracy: 0.9273
35744/60000 [================>.............] - ETA: 41s - loss: 0.2374 - categorical_accuracy: 0.9273
35776/60000 [================>.............] - ETA: 41s - loss: 0.2373 - categorical_accuracy: 0.9273
35808/60000 [================>.............] - ETA: 41s - loss: 0.2372 - categorical_accuracy: 0.9274
35840/60000 [================>.............] - ETA: 41s - loss: 0.2370 - categorical_accuracy: 0.9274
35872/60000 [================>.............] - ETA: 41s - loss: 0.2369 - categorical_accuracy: 0.9274
35904/60000 [================>.............] - ETA: 41s - loss: 0.2367 - categorical_accuracy: 0.9275
35936/60000 [================>.............] - ETA: 41s - loss: 0.2366 - categorical_accuracy: 0.9275
35968/60000 [================>.............] - ETA: 41s - loss: 0.2364 - categorical_accuracy: 0.9276
36000/60000 [=================>............] - ETA: 41s - loss: 0.2363 - categorical_accuracy: 0.9276
36032/60000 [=================>............] - ETA: 40s - loss: 0.2361 - categorical_accuracy: 0.9276
36064/60000 [=================>............] - ETA: 40s - loss: 0.2360 - categorical_accuracy: 0.9277
36096/60000 [=================>............] - ETA: 40s - loss: 0.2360 - categorical_accuracy: 0.9277
36128/60000 [=================>............] - ETA: 40s - loss: 0.2358 - categorical_accuracy: 0.9278
36160/60000 [=================>............] - ETA: 40s - loss: 0.2357 - categorical_accuracy: 0.9278
36192/60000 [=================>............] - ETA: 40s - loss: 0.2355 - categorical_accuracy: 0.9279
36224/60000 [=================>............] - ETA: 40s - loss: 0.2353 - categorical_accuracy: 0.9279
36256/60000 [=================>............] - ETA: 40s - loss: 0.2351 - categorical_accuracy: 0.9280
36320/60000 [=================>............] - ETA: 40s - loss: 0.2349 - categorical_accuracy: 0.9281
36352/60000 [=================>............] - ETA: 40s - loss: 0.2347 - categorical_accuracy: 0.9281
36384/60000 [=================>............] - ETA: 40s - loss: 0.2346 - categorical_accuracy: 0.9281
36416/60000 [=================>............] - ETA: 40s - loss: 0.2346 - categorical_accuracy: 0.9281
36448/60000 [=================>............] - ETA: 40s - loss: 0.2349 - categorical_accuracy: 0.9281
36480/60000 [=================>............] - ETA: 40s - loss: 0.2347 - categorical_accuracy: 0.9282
36512/60000 [=================>............] - ETA: 40s - loss: 0.2347 - categorical_accuracy: 0.9282
36544/60000 [=================>............] - ETA: 40s - loss: 0.2345 - categorical_accuracy: 0.9283
36576/60000 [=================>............] - ETA: 40s - loss: 0.2349 - categorical_accuracy: 0.9282
36608/60000 [=================>............] - ETA: 39s - loss: 0.2349 - categorical_accuracy: 0.9282
36640/60000 [=================>............] - ETA: 39s - loss: 0.2349 - categorical_accuracy: 0.9282
36672/60000 [=================>............] - ETA: 39s - loss: 0.2347 - categorical_accuracy: 0.9283
36704/60000 [=================>............] - ETA: 39s - loss: 0.2346 - categorical_accuracy: 0.9283
36736/60000 [=================>............] - ETA: 39s - loss: 0.2345 - categorical_accuracy: 0.9283
36800/60000 [=================>............] - ETA: 39s - loss: 0.2342 - categorical_accuracy: 0.9284
36832/60000 [=================>............] - ETA: 39s - loss: 0.2341 - categorical_accuracy: 0.9284
36864/60000 [=================>............] - ETA: 39s - loss: 0.2340 - categorical_accuracy: 0.9285
36928/60000 [=================>............] - ETA: 39s - loss: 0.2337 - categorical_accuracy: 0.9286
36992/60000 [=================>............] - ETA: 39s - loss: 0.2337 - categorical_accuracy: 0.9286
37024/60000 [=================>............] - ETA: 39s - loss: 0.2338 - categorical_accuracy: 0.9286
37056/60000 [=================>............] - ETA: 39s - loss: 0.2336 - categorical_accuracy: 0.9286
37088/60000 [=================>............] - ETA: 39s - loss: 0.2335 - categorical_accuracy: 0.9287
37120/60000 [=================>............] - ETA: 39s - loss: 0.2335 - categorical_accuracy: 0.9287
37152/60000 [=================>............] - ETA: 38s - loss: 0.2333 - categorical_accuracy: 0.9287
37184/60000 [=================>............] - ETA: 38s - loss: 0.2333 - categorical_accuracy: 0.9287
37216/60000 [=================>............] - ETA: 38s - loss: 0.2331 - categorical_accuracy: 0.9288
37248/60000 [=================>............] - ETA: 38s - loss: 0.2329 - categorical_accuracy: 0.9289
37280/60000 [=================>............] - ETA: 38s - loss: 0.2328 - categorical_accuracy: 0.9289
37312/60000 [=================>............] - ETA: 38s - loss: 0.2327 - categorical_accuracy: 0.9289
37344/60000 [=================>............] - ETA: 38s - loss: 0.2326 - categorical_accuracy: 0.9289
37376/60000 [=================>............] - ETA: 38s - loss: 0.2326 - categorical_accuracy: 0.9290
37408/60000 [=================>............] - ETA: 38s - loss: 0.2324 - categorical_accuracy: 0.9290
37440/60000 [=================>............] - ETA: 38s - loss: 0.2323 - categorical_accuracy: 0.9291
37472/60000 [=================>............] - ETA: 38s - loss: 0.2322 - categorical_accuracy: 0.9291
37504/60000 [=================>............] - ETA: 38s - loss: 0.2320 - categorical_accuracy: 0.9292
37536/60000 [=================>............] - ETA: 38s - loss: 0.2318 - categorical_accuracy: 0.9292
37568/60000 [=================>............] - ETA: 38s - loss: 0.2317 - categorical_accuracy: 0.9292
37600/60000 [=================>............] - ETA: 38s - loss: 0.2316 - categorical_accuracy: 0.9293
37664/60000 [=================>............] - ETA: 38s - loss: 0.2314 - categorical_accuracy: 0.9293
37728/60000 [=================>............] - ETA: 37s - loss: 0.2312 - categorical_accuracy: 0.9294
37760/60000 [=================>............] - ETA: 37s - loss: 0.2310 - categorical_accuracy: 0.9294
37824/60000 [=================>............] - ETA: 37s - loss: 0.2308 - categorical_accuracy: 0.9295
37856/60000 [=================>............] - ETA: 37s - loss: 0.2309 - categorical_accuracy: 0.9295
37888/60000 [=================>............] - ETA: 37s - loss: 0.2308 - categorical_accuracy: 0.9296
37920/60000 [=================>............] - ETA: 37s - loss: 0.2306 - categorical_accuracy: 0.9296
37952/60000 [=================>............] - ETA: 37s - loss: 0.2305 - categorical_accuracy: 0.9296
38016/60000 [==================>...........] - ETA: 37s - loss: 0.2302 - categorical_accuracy: 0.9297
38048/60000 [==================>...........] - ETA: 37s - loss: 0.2303 - categorical_accuracy: 0.9297
38080/60000 [==================>...........] - ETA: 37s - loss: 0.2301 - categorical_accuracy: 0.9298
38112/60000 [==================>...........] - ETA: 37s - loss: 0.2300 - categorical_accuracy: 0.9298
38144/60000 [==================>...........] - ETA: 37s - loss: 0.2299 - categorical_accuracy: 0.9298
38176/60000 [==================>...........] - ETA: 37s - loss: 0.2297 - categorical_accuracy: 0.9299
38208/60000 [==================>...........] - ETA: 37s - loss: 0.2297 - categorical_accuracy: 0.9299
38240/60000 [==================>...........] - ETA: 37s - loss: 0.2295 - categorical_accuracy: 0.9299
38272/60000 [==================>...........] - ETA: 37s - loss: 0.2295 - categorical_accuracy: 0.9299
38304/60000 [==================>...........] - ETA: 36s - loss: 0.2294 - categorical_accuracy: 0.9300
38336/60000 [==================>...........] - ETA: 36s - loss: 0.2293 - categorical_accuracy: 0.9300
38400/60000 [==================>...........] - ETA: 36s - loss: 0.2291 - categorical_accuracy: 0.9301
38432/60000 [==================>...........] - ETA: 36s - loss: 0.2289 - categorical_accuracy: 0.9301
38464/60000 [==================>...........] - ETA: 36s - loss: 0.2288 - categorical_accuracy: 0.9301
38496/60000 [==================>...........] - ETA: 36s - loss: 0.2287 - categorical_accuracy: 0.9301
38528/60000 [==================>...........] - ETA: 36s - loss: 0.2286 - categorical_accuracy: 0.9302
38560/60000 [==================>...........] - ETA: 36s - loss: 0.2285 - categorical_accuracy: 0.9302
38592/60000 [==================>...........] - ETA: 36s - loss: 0.2283 - categorical_accuracy: 0.9303
38624/60000 [==================>...........] - ETA: 36s - loss: 0.2282 - categorical_accuracy: 0.9303
38656/60000 [==================>...........] - ETA: 36s - loss: 0.2281 - categorical_accuracy: 0.9303
38688/60000 [==================>...........] - ETA: 36s - loss: 0.2279 - categorical_accuracy: 0.9304
38720/60000 [==================>...........] - ETA: 36s - loss: 0.2277 - categorical_accuracy: 0.9304
38752/60000 [==================>...........] - ETA: 36s - loss: 0.2276 - categorical_accuracy: 0.9305
38784/60000 [==================>...........] - ETA: 36s - loss: 0.2274 - categorical_accuracy: 0.9306
38816/60000 [==================>...........] - ETA: 36s - loss: 0.2273 - categorical_accuracy: 0.9306
38848/60000 [==================>...........] - ETA: 36s - loss: 0.2271 - categorical_accuracy: 0.9306
38880/60000 [==================>...........] - ETA: 36s - loss: 0.2270 - categorical_accuracy: 0.9307
38912/60000 [==================>...........] - ETA: 35s - loss: 0.2271 - categorical_accuracy: 0.9306
38944/60000 [==================>...........] - ETA: 35s - loss: 0.2270 - categorical_accuracy: 0.9307
38976/60000 [==================>...........] - ETA: 35s - loss: 0.2268 - categorical_accuracy: 0.9307
39008/60000 [==================>...........] - ETA: 35s - loss: 0.2266 - categorical_accuracy: 0.9308
39040/60000 [==================>...........] - ETA: 35s - loss: 0.2267 - categorical_accuracy: 0.9307
39072/60000 [==================>...........] - ETA: 35s - loss: 0.2266 - categorical_accuracy: 0.9307
39104/60000 [==================>...........] - ETA: 35s - loss: 0.2265 - categorical_accuracy: 0.9308
39136/60000 [==================>...........] - ETA: 35s - loss: 0.2265 - categorical_accuracy: 0.9308
39168/60000 [==================>...........] - ETA: 35s - loss: 0.2264 - categorical_accuracy: 0.9308
39200/60000 [==================>...........] - ETA: 35s - loss: 0.2262 - categorical_accuracy: 0.9309
39232/60000 [==================>...........] - ETA: 35s - loss: 0.2261 - categorical_accuracy: 0.9309
39264/60000 [==================>...........] - ETA: 35s - loss: 0.2260 - categorical_accuracy: 0.9310
39296/60000 [==================>...........] - ETA: 35s - loss: 0.2262 - categorical_accuracy: 0.9309
39328/60000 [==================>...........] - ETA: 35s - loss: 0.2262 - categorical_accuracy: 0.9309
39360/60000 [==================>...........] - ETA: 35s - loss: 0.2262 - categorical_accuracy: 0.9309
39392/60000 [==================>...........] - ETA: 35s - loss: 0.2261 - categorical_accuracy: 0.9310
39424/60000 [==================>...........] - ETA: 35s - loss: 0.2260 - categorical_accuracy: 0.9310
39456/60000 [==================>...........] - ETA: 35s - loss: 0.2259 - categorical_accuracy: 0.9310
39488/60000 [==================>...........] - ETA: 34s - loss: 0.2258 - categorical_accuracy: 0.9310
39520/60000 [==================>...........] - ETA: 34s - loss: 0.2257 - categorical_accuracy: 0.9311
39552/60000 [==================>...........] - ETA: 34s - loss: 0.2256 - categorical_accuracy: 0.9311
39584/60000 [==================>...........] - ETA: 34s - loss: 0.2255 - categorical_accuracy: 0.9311
39616/60000 [==================>...........] - ETA: 34s - loss: 0.2254 - categorical_accuracy: 0.9312
39648/60000 [==================>...........] - ETA: 34s - loss: 0.2254 - categorical_accuracy: 0.9312
39680/60000 [==================>...........] - ETA: 34s - loss: 0.2254 - categorical_accuracy: 0.9312
39712/60000 [==================>...........] - ETA: 34s - loss: 0.2252 - categorical_accuracy: 0.9312
39744/60000 [==================>...........] - ETA: 34s - loss: 0.2251 - categorical_accuracy: 0.9313
39776/60000 [==================>...........] - ETA: 34s - loss: 0.2250 - categorical_accuracy: 0.9313
39840/60000 [==================>...........] - ETA: 34s - loss: 0.2247 - categorical_accuracy: 0.9314
39872/60000 [==================>...........] - ETA: 34s - loss: 0.2247 - categorical_accuracy: 0.9314
39936/60000 [==================>...........] - ETA: 34s - loss: 0.2244 - categorical_accuracy: 0.9315
39968/60000 [==================>...........] - ETA: 34s - loss: 0.2243 - categorical_accuracy: 0.9315
40000/60000 [===================>..........] - ETA: 34s - loss: 0.2243 - categorical_accuracy: 0.9315
40032/60000 [===================>..........] - ETA: 34s - loss: 0.2242 - categorical_accuracy: 0.9316
40064/60000 [===================>..........] - ETA: 33s - loss: 0.2241 - categorical_accuracy: 0.9316
40096/60000 [===================>..........] - ETA: 33s - loss: 0.2240 - categorical_accuracy: 0.9316
40128/60000 [===================>..........] - ETA: 33s - loss: 0.2239 - categorical_accuracy: 0.9316
40160/60000 [===================>..........] - ETA: 33s - loss: 0.2239 - categorical_accuracy: 0.9316
40192/60000 [===================>..........] - ETA: 33s - loss: 0.2238 - categorical_accuracy: 0.9317
40224/60000 [===================>..........] - ETA: 33s - loss: 0.2237 - categorical_accuracy: 0.9317
40256/60000 [===================>..........] - ETA: 33s - loss: 0.2236 - categorical_accuracy: 0.9317
40288/60000 [===================>..........] - ETA: 33s - loss: 0.2235 - categorical_accuracy: 0.9317
40320/60000 [===================>..........] - ETA: 33s - loss: 0.2234 - categorical_accuracy: 0.9318
40352/60000 [===================>..........] - ETA: 33s - loss: 0.2233 - categorical_accuracy: 0.9318
40416/60000 [===================>..........] - ETA: 33s - loss: 0.2230 - categorical_accuracy: 0.9318
40448/60000 [===================>..........] - ETA: 33s - loss: 0.2230 - categorical_accuracy: 0.9318
40480/60000 [===================>..........] - ETA: 33s - loss: 0.2230 - categorical_accuracy: 0.9318
40512/60000 [===================>..........] - ETA: 33s - loss: 0.2229 - categorical_accuracy: 0.9319
40544/60000 [===================>..........] - ETA: 33s - loss: 0.2227 - categorical_accuracy: 0.9319
40576/60000 [===================>..........] - ETA: 33s - loss: 0.2226 - categorical_accuracy: 0.9320
40608/60000 [===================>..........] - ETA: 33s - loss: 0.2226 - categorical_accuracy: 0.9320
40640/60000 [===================>..........] - ETA: 33s - loss: 0.2224 - categorical_accuracy: 0.9320
40672/60000 [===================>..........] - ETA: 32s - loss: 0.2225 - categorical_accuracy: 0.9320
40704/60000 [===================>..........] - ETA: 32s - loss: 0.2224 - categorical_accuracy: 0.9320
40736/60000 [===================>..........] - ETA: 32s - loss: 0.2222 - categorical_accuracy: 0.9321
40768/60000 [===================>..........] - ETA: 32s - loss: 0.2223 - categorical_accuracy: 0.9321
40800/60000 [===================>..........] - ETA: 32s - loss: 0.2222 - categorical_accuracy: 0.9321
40832/60000 [===================>..........] - ETA: 32s - loss: 0.2221 - categorical_accuracy: 0.9322
40864/60000 [===================>..........] - ETA: 32s - loss: 0.2219 - categorical_accuracy: 0.9322
40896/60000 [===================>..........] - ETA: 32s - loss: 0.2218 - categorical_accuracy: 0.9323
40928/60000 [===================>..........] - ETA: 32s - loss: 0.2218 - categorical_accuracy: 0.9323
40960/60000 [===================>..........] - ETA: 32s - loss: 0.2216 - categorical_accuracy: 0.9323
40992/60000 [===================>..........] - ETA: 32s - loss: 0.2215 - categorical_accuracy: 0.9324
41024/60000 [===================>..........] - ETA: 32s - loss: 0.2215 - categorical_accuracy: 0.9324
41056/60000 [===================>..........] - ETA: 32s - loss: 0.2214 - categorical_accuracy: 0.9324
41088/60000 [===================>..........] - ETA: 32s - loss: 0.2212 - categorical_accuracy: 0.9324
41120/60000 [===================>..........] - ETA: 32s - loss: 0.2211 - categorical_accuracy: 0.9325
41152/60000 [===================>..........] - ETA: 32s - loss: 0.2210 - categorical_accuracy: 0.9325
41184/60000 [===================>..........] - ETA: 32s - loss: 0.2209 - categorical_accuracy: 0.9325
41216/60000 [===================>..........] - ETA: 32s - loss: 0.2208 - categorical_accuracy: 0.9326
41248/60000 [===================>..........] - ETA: 31s - loss: 0.2206 - categorical_accuracy: 0.9326
41280/60000 [===================>..........] - ETA: 31s - loss: 0.2205 - categorical_accuracy: 0.9327
41312/60000 [===================>..........] - ETA: 31s - loss: 0.2204 - categorical_accuracy: 0.9327
41344/60000 [===================>..........] - ETA: 31s - loss: 0.2203 - categorical_accuracy: 0.9327
41376/60000 [===================>..........] - ETA: 31s - loss: 0.2201 - categorical_accuracy: 0.9327
41408/60000 [===================>..........] - ETA: 31s - loss: 0.2200 - categorical_accuracy: 0.9328
41440/60000 [===================>..........] - ETA: 31s - loss: 0.2199 - categorical_accuracy: 0.9328
41472/60000 [===================>..........] - ETA: 31s - loss: 0.2198 - categorical_accuracy: 0.9328
41504/60000 [===================>..........] - ETA: 31s - loss: 0.2197 - categorical_accuracy: 0.9328
41536/60000 [===================>..........] - ETA: 31s - loss: 0.2195 - categorical_accuracy: 0.9329
41568/60000 [===================>..........] - ETA: 31s - loss: 0.2194 - categorical_accuracy: 0.9329
41600/60000 [===================>..........] - ETA: 31s - loss: 0.2196 - categorical_accuracy: 0.9329
41632/60000 [===================>..........] - ETA: 31s - loss: 0.2196 - categorical_accuracy: 0.9329
41664/60000 [===================>..........] - ETA: 31s - loss: 0.2194 - categorical_accuracy: 0.9329
41696/60000 [===================>..........] - ETA: 31s - loss: 0.2194 - categorical_accuracy: 0.9330
41728/60000 [===================>..........] - ETA: 31s - loss: 0.2192 - categorical_accuracy: 0.9330
41760/60000 [===================>..........] - ETA: 31s - loss: 0.2190 - categorical_accuracy: 0.9331
41792/60000 [===================>..........] - ETA: 31s - loss: 0.2190 - categorical_accuracy: 0.9330
41824/60000 [===================>..........] - ETA: 30s - loss: 0.2188 - categorical_accuracy: 0.9331
41856/60000 [===================>..........] - ETA: 30s - loss: 0.2187 - categorical_accuracy: 0.9332
41888/60000 [===================>..........] - ETA: 30s - loss: 0.2187 - categorical_accuracy: 0.9332
41920/60000 [===================>..........] - ETA: 30s - loss: 0.2186 - categorical_accuracy: 0.9332
41952/60000 [===================>..........] - ETA: 30s - loss: 0.2185 - categorical_accuracy: 0.9332
41984/60000 [===================>..........] - ETA: 30s - loss: 0.2184 - categorical_accuracy: 0.9333
42016/60000 [====================>.........] - ETA: 30s - loss: 0.2183 - categorical_accuracy: 0.9333
42048/60000 [====================>.........] - ETA: 30s - loss: 0.2182 - categorical_accuracy: 0.9333
42080/60000 [====================>.........] - ETA: 30s - loss: 0.2182 - categorical_accuracy: 0.9333
42112/60000 [====================>.........] - ETA: 30s - loss: 0.2181 - categorical_accuracy: 0.9333
42144/60000 [====================>.........] - ETA: 30s - loss: 0.2180 - categorical_accuracy: 0.9334
42176/60000 [====================>.........] - ETA: 30s - loss: 0.2181 - categorical_accuracy: 0.9334
42208/60000 [====================>.........] - ETA: 30s - loss: 0.2180 - categorical_accuracy: 0.9334
42240/60000 [====================>.........] - ETA: 30s - loss: 0.2178 - categorical_accuracy: 0.9335
42272/60000 [====================>.........] - ETA: 30s - loss: 0.2177 - categorical_accuracy: 0.9335
42304/60000 [====================>.........] - ETA: 30s - loss: 0.2175 - categorical_accuracy: 0.9336
42336/60000 [====================>.........] - ETA: 30s - loss: 0.2175 - categorical_accuracy: 0.9335
42368/60000 [====================>.........] - ETA: 30s - loss: 0.2174 - categorical_accuracy: 0.9336
42400/60000 [====================>.........] - ETA: 30s - loss: 0.2175 - categorical_accuracy: 0.9335
42432/60000 [====================>.........] - ETA: 29s - loss: 0.2174 - categorical_accuracy: 0.9336
42464/60000 [====================>.........] - ETA: 29s - loss: 0.2173 - categorical_accuracy: 0.9336
42496/60000 [====================>.........] - ETA: 29s - loss: 0.2172 - categorical_accuracy: 0.9336
42528/60000 [====================>.........] - ETA: 29s - loss: 0.2171 - categorical_accuracy: 0.9336
42560/60000 [====================>.........] - ETA: 29s - loss: 0.2170 - categorical_accuracy: 0.9337
42592/60000 [====================>.........] - ETA: 29s - loss: 0.2168 - categorical_accuracy: 0.9337
42624/60000 [====================>.........] - ETA: 29s - loss: 0.2167 - categorical_accuracy: 0.9337
42656/60000 [====================>.........] - ETA: 29s - loss: 0.2166 - categorical_accuracy: 0.9338
42688/60000 [====================>.........] - ETA: 29s - loss: 0.2167 - categorical_accuracy: 0.9338
42720/60000 [====================>.........] - ETA: 29s - loss: 0.2167 - categorical_accuracy: 0.9338
42752/60000 [====================>.........] - ETA: 29s - loss: 0.2167 - categorical_accuracy: 0.9338
42784/60000 [====================>.........] - ETA: 29s - loss: 0.2166 - categorical_accuracy: 0.9339
42816/60000 [====================>.........] - ETA: 29s - loss: 0.2165 - categorical_accuracy: 0.9339
42848/60000 [====================>.........] - ETA: 29s - loss: 0.2164 - categorical_accuracy: 0.9339
42880/60000 [====================>.........] - ETA: 29s - loss: 0.2164 - categorical_accuracy: 0.9339
42912/60000 [====================>.........] - ETA: 29s - loss: 0.2164 - categorical_accuracy: 0.9339
42944/60000 [====================>.........] - ETA: 29s - loss: 0.2164 - categorical_accuracy: 0.9339
42976/60000 [====================>.........] - ETA: 29s - loss: 0.2163 - categorical_accuracy: 0.9339
43008/60000 [====================>.........] - ETA: 28s - loss: 0.2161 - categorical_accuracy: 0.9339
43040/60000 [====================>.........] - ETA: 28s - loss: 0.2160 - categorical_accuracy: 0.9340
43072/60000 [====================>.........] - ETA: 28s - loss: 0.2160 - categorical_accuracy: 0.9340
43104/60000 [====================>.........] - ETA: 28s - loss: 0.2159 - categorical_accuracy: 0.9340
43136/60000 [====================>.........] - ETA: 28s - loss: 0.2159 - categorical_accuracy: 0.9340
43168/60000 [====================>.........] - ETA: 28s - loss: 0.2158 - categorical_accuracy: 0.9340
43200/60000 [====================>.........] - ETA: 28s - loss: 0.2157 - categorical_accuracy: 0.9341
43232/60000 [====================>.........] - ETA: 28s - loss: 0.2156 - categorical_accuracy: 0.9341
43264/60000 [====================>.........] - ETA: 28s - loss: 0.2155 - categorical_accuracy: 0.9341
43296/60000 [====================>.........] - ETA: 28s - loss: 0.2154 - categorical_accuracy: 0.9341
43328/60000 [====================>.........] - ETA: 28s - loss: 0.2153 - categorical_accuracy: 0.9342
43360/60000 [====================>.........] - ETA: 28s - loss: 0.2153 - categorical_accuracy: 0.9342
43392/60000 [====================>.........] - ETA: 28s - loss: 0.2152 - categorical_accuracy: 0.9342
43424/60000 [====================>.........] - ETA: 28s - loss: 0.2151 - categorical_accuracy: 0.9343
43456/60000 [====================>.........] - ETA: 28s - loss: 0.2149 - categorical_accuracy: 0.9343
43488/60000 [====================>.........] - ETA: 28s - loss: 0.2148 - categorical_accuracy: 0.9343
43520/60000 [====================>.........] - ETA: 28s - loss: 0.2148 - categorical_accuracy: 0.9344
43552/60000 [====================>.........] - ETA: 28s - loss: 0.2146 - categorical_accuracy: 0.9344
43584/60000 [====================>.........] - ETA: 27s - loss: 0.2147 - categorical_accuracy: 0.9344
43616/60000 [====================>.........] - ETA: 27s - loss: 0.2147 - categorical_accuracy: 0.9344
43648/60000 [====================>.........] - ETA: 27s - loss: 0.2147 - categorical_accuracy: 0.9344
43680/60000 [====================>.........] - ETA: 27s - loss: 0.2146 - categorical_accuracy: 0.9344
43712/60000 [====================>.........] - ETA: 27s - loss: 0.2145 - categorical_accuracy: 0.9345
43744/60000 [====================>.........] - ETA: 27s - loss: 0.2144 - categorical_accuracy: 0.9345
43776/60000 [====================>.........] - ETA: 27s - loss: 0.2143 - categorical_accuracy: 0.9345
43808/60000 [====================>.........] - ETA: 27s - loss: 0.2142 - categorical_accuracy: 0.9345
43840/60000 [====================>.........] - ETA: 27s - loss: 0.2142 - categorical_accuracy: 0.9345
43872/60000 [====================>.........] - ETA: 27s - loss: 0.2140 - categorical_accuracy: 0.9346
43904/60000 [====================>.........] - ETA: 27s - loss: 0.2139 - categorical_accuracy: 0.9346
43936/60000 [====================>.........] - ETA: 27s - loss: 0.2138 - categorical_accuracy: 0.9347
43968/60000 [====================>.........] - ETA: 27s - loss: 0.2137 - categorical_accuracy: 0.9347
44000/60000 [=====================>........] - ETA: 27s - loss: 0.2135 - categorical_accuracy: 0.9348
44032/60000 [=====================>........] - ETA: 27s - loss: 0.2135 - categorical_accuracy: 0.9348
44064/60000 [=====================>........] - ETA: 27s - loss: 0.2134 - categorical_accuracy: 0.9348
44096/60000 [=====================>........] - ETA: 27s - loss: 0.2133 - categorical_accuracy: 0.9348
44128/60000 [=====================>........] - ETA: 27s - loss: 0.2132 - categorical_accuracy: 0.9348
44160/60000 [=====================>........] - ETA: 26s - loss: 0.2131 - categorical_accuracy: 0.9349
44192/60000 [=====================>........] - ETA: 26s - loss: 0.2130 - categorical_accuracy: 0.9349
44224/60000 [=====================>........] - ETA: 26s - loss: 0.2130 - categorical_accuracy: 0.9349
44256/60000 [=====================>........] - ETA: 26s - loss: 0.2129 - categorical_accuracy: 0.9349
44288/60000 [=====================>........] - ETA: 26s - loss: 0.2128 - categorical_accuracy: 0.9349
44320/60000 [=====================>........] - ETA: 26s - loss: 0.2127 - categorical_accuracy: 0.9350
44352/60000 [=====================>........] - ETA: 26s - loss: 0.2125 - categorical_accuracy: 0.9350
44384/60000 [=====================>........] - ETA: 26s - loss: 0.2126 - categorical_accuracy: 0.9350
44416/60000 [=====================>........] - ETA: 26s - loss: 0.2125 - categorical_accuracy: 0.9350
44448/60000 [=====================>........] - ETA: 26s - loss: 0.2124 - categorical_accuracy: 0.9351
44480/60000 [=====================>........] - ETA: 26s - loss: 0.2123 - categorical_accuracy: 0.9351
44512/60000 [=====================>........] - ETA: 26s - loss: 0.2121 - categorical_accuracy: 0.9352
44544/60000 [=====================>........] - ETA: 26s - loss: 0.2120 - categorical_accuracy: 0.9352
44576/60000 [=====================>........] - ETA: 26s - loss: 0.2120 - categorical_accuracy: 0.9352
44608/60000 [=====================>........] - ETA: 26s - loss: 0.2121 - categorical_accuracy: 0.9352
44640/60000 [=====================>........] - ETA: 26s - loss: 0.2122 - categorical_accuracy: 0.9352
44672/60000 [=====================>........] - ETA: 26s - loss: 0.2120 - categorical_accuracy: 0.9352
44704/60000 [=====================>........] - ETA: 26s - loss: 0.2119 - categorical_accuracy: 0.9352
44736/60000 [=====================>........] - ETA: 26s - loss: 0.2118 - categorical_accuracy: 0.9353
44768/60000 [=====================>........] - ETA: 25s - loss: 0.2117 - categorical_accuracy: 0.9353
44800/60000 [=====================>........] - ETA: 25s - loss: 0.2117 - categorical_accuracy: 0.9353
44832/60000 [=====================>........] - ETA: 25s - loss: 0.2116 - categorical_accuracy: 0.9354
44864/60000 [=====================>........] - ETA: 25s - loss: 0.2114 - categorical_accuracy: 0.9354
44896/60000 [=====================>........] - ETA: 25s - loss: 0.2114 - categorical_accuracy: 0.9354
44928/60000 [=====================>........] - ETA: 25s - loss: 0.2114 - categorical_accuracy: 0.9354
44960/60000 [=====================>........] - ETA: 25s - loss: 0.2112 - categorical_accuracy: 0.9355
44992/60000 [=====================>........] - ETA: 25s - loss: 0.2111 - categorical_accuracy: 0.9355
45024/60000 [=====================>........] - ETA: 25s - loss: 0.2110 - categorical_accuracy: 0.9355
45056/60000 [=====================>........] - ETA: 25s - loss: 0.2110 - categorical_accuracy: 0.9355
45088/60000 [=====================>........] - ETA: 25s - loss: 0.2109 - categorical_accuracy: 0.9356
45120/60000 [=====================>........] - ETA: 25s - loss: 0.2109 - categorical_accuracy: 0.9356
45152/60000 [=====================>........] - ETA: 25s - loss: 0.2109 - categorical_accuracy: 0.9356
45184/60000 [=====================>........] - ETA: 25s - loss: 0.2108 - categorical_accuracy: 0.9356
45216/60000 [=====================>........] - ETA: 25s - loss: 0.2107 - categorical_accuracy: 0.9356
45248/60000 [=====================>........] - ETA: 25s - loss: 0.2107 - categorical_accuracy: 0.9356
45280/60000 [=====================>........] - ETA: 25s - loss: 0.2107 - categorical_accuracy: 0.9356
45312/60000 [=====================>........] - ETA: 25s - loss: 0.2107 - categorical_accuracy: 0.9356
45344/60000 [=====================>........] - ETA: 24s - loss: 0.2106 - categorical_accuracy: 0.9357
45376/60000 [=====================>........] - ETA: 24s - loss: 0.2107 - categorical_accuracy: 0.9356
45408/60000 [=====================>........] - ETA: 24s - loss: 0.2108 - categorical_accuracy: 0.9356
45440/60000 [=====================>........] - ETA: 24s - loss: 0.2107 - categorical_accuracy: 0.9357
45472/60000 [=====================>........] - ETA: 24s - loss: 0.2106 - categorical_accuracy: 0.9357
45504/60000 [=====================>........] - ETA: 24s - loss: 0.2106 - categorical_accuracy: 0.9357
45536/60000 [=====================>........] - ETA: 24s - loss: 0.2105 - categorical_accuracy: 0.9357
45568/60000 [=====================>........] - ETA: 24s - loss: 0.2103 - categorical_accuracy: 0.9358
45600/60000 [=====================>........] - ETA: 24s - loss: 0.2102 - categorical_accuracy: 0.9358
45632/60000 [=====================>........] - ETA: 24s - loss: 0.2101 - categorical_accuracy: 0.9359
45664/60000 [=====================>........] - ETA: 24s - loss: 0.2100 - categorical_accuracy: 0.9359
45696/60000 [=====================>........] - ETA: 24s - loss: 0.2100 - categorical_accuracy: 0.9359
45728/60000 [=====================>........] - ETA: 24s - loss: 0.2098 - categorical_accuracy: 0.9359
45760/60000 [=====================>........] - ETA: 24s - loss: 0.2099 - categorical_accuracy: 0.9360
45792/60000 [=====================>........] - ETA: 24s - loss: 0.2097 - categorical_accuracy: 0.9360
45824/60000 [=====================>........] - ETA: 24s - loss: 0.2097 - categorical_accuracy: 0.9360
45856/60000 [=====================>........] - ETA: 24s - loss: 0.2096 - categorical_accuracy: 0.9360
45888/60000 [=====================>........] - ETA: 24s - loss: 0.2094 - categorical_accuracy: 0.9361
45920/60000 [=====================>........] - ETA: 24s - loss: 0.2093 - categorical_accuracy: 0.9361
45952/60000 [=====================>........] - ETA: 23s - loss: 0.2092 - categorical_accuracy: 0.9362
45984/60000 [=====================>........] - ETA: 23s - loss: 0.2092 - categorical_accuracy: 0.9362
46016/60000 [======================>.......] - ETA: 23s - loss: 0.2091 - categorical_accuracy: 0.9362
46080/60000 [======================>.......] - ETA: 23s - loss: 0.2091 - categorical_accuracy: 0.9362
46112/60000 [======================>.......] - ETA: 23s - loss: 0.2090 - categorical_accuracy: 0.9362
46144/60000 [======================>.......] - ETA: 23s - loss: 0.2092 - categorical_accuracy: 0.9362
46208/60000 [======================>.......] - ETA: 23s - loss: 0.2090 - categorical_accuracy: 0.9362
46240/60000 [======================>.......] - ETA: 23s - loss: 0.2089 - categorical_accuracy: 0.9362
46272/60000 [======================>.......] - ETA: 23s - loss: 0.2088 - categorical_accuracy: 0.9362
46304/60000 [======================>.......] - ETA: 23s - loss: 0.2088 - categorical_accuracy: 0.9363
46336/60000 [======================>.......] - ETA: 23s - loss: 0.2087 - categorical_accuracy: 0.9363
46368/60000 [======================>.......] - ETA: 23s - loss: 0.2086 - categorical_accuracy: 0.9363
46432/60000 [======================>.......] - ETA: 23s - loss: 0.2084 - categorical_accuracy: 0.9364
46464/60000 [======================>.......] - ETA: 23s - loss: 0.2083 - categorical_accuracy: 0.9364
46496/60000 [======================>.......] - ETA: 23s - loss: 0.2081 - categorical_accuracy: 0.9364
46528/60000 [======================>.......] - ETA: 22s - loss: 0.2081 - categorical_accuracy: 0.9365
46592/60000 [======================>.......] - ETA: 22s - loss: 0.2079 - categorical_accuracy: 0.9365
46624/60000 [======================>.......] - ETA: 22s - loss: 0.2078 - categorical_accuracy: 0.9366
46688/60000 [======================>.......] - ETA: 22s - loss: 0.2077 - categorical_accuracy: 0.9366
46720/60000 [======================>.......] - ETA: 22s - loss: 0.2076 - categorical_accuracy: 0.9367
46752/60000 [======================>.......] - ETA: 22s - loss: 0.2075 - categorical_accuracy: 0.9367
46784/60000 [======================>.......] - ETA: 22s - loss: 0.2073 - categorical_accuracy: 0.9367
46816/60000 [======================>.......] - ETA: 22s - loss: 0.2072 - categorical_accuracy: 0.9368
46848/60000 [======================>.......] - ETA: 22s - loss: 0.2071 - categorical_accuracy: 0.9368
46880/60000 [======================>.......] - ETA: 22s - loss: 0.2072 - categorical_accuracy: 0.9368
46912/60000 [======================>.......] - ETA: 22s - loss: 0.2071 - categorical_accuracy: 0.9368
46976/60000 [======================>.......] - ETA: 22s - loss: 0.2069 - categorical_accuracy: 0.9369
47008/60000 [======================>.......] - ETA: 22s - loss: 0.2068 - categorical_accuracy: 0.9369
47040/60000 [======================>.......] - ETA: 22s - loss: 0.2067 - categorical_accuracy: 0.9369
47104/60000 [======================>.......] - ETA: 21s - loss: 0.2066 - categorical_accuracy: 0.9369
47136/60000 [======================>.......] - ETA: 21s - loss: 0.2065 - categorical_accuracy: 0.9370
47168/60000 [======================>.......] - ETA: 21s - loss: 0.2064 - categorical_accuracy: 0.9370
47200/60000 [======================>.......] - ETA: 21s - loss: 0.2065 - categorical_accuracy: 0.9370
47232/60000 [======================>.......] - ETA: 21s - loss: 0.2064 - categorical_accuracy: 0.9370
47296/60000 [======================>.......] - ETA: 21s - loss: 0.2065 - categorical_accuracy: 0.9370
47328/60000 [======================>.......] - ETA: 21s - loss: 0.2066 - categorical_accuracy: 0.9369
47392/60000 [======================>.......] - ETA: 21s - loss: 0.2064 - categorical_accuracy: 0.9370
47424/60000 [======================>.......] - ETA: 21s - loss: 0.2064 - categorical_accuracy: 0.9370
47456/60000 [======================>.......] - ETA: 21s - loss: 0.2064 - categorical_accuracy: 0.9370
47488/60000 [======================>.......] - ETA: 21s - loss: 0.2064 - categorical_accuracy: 0.9370
47520/60000 [======================>.......] - ETA: 21s - loss: 0.2063 - categorical_accuracy: 0.9370
47552/60000 [======================>.......] - ETA: 21s - loss: 0.2062 - categorical_accuracy: 0.9370
47584/60000 [======================>.......] - ETA: 21s - loss: 0.2061 - categorical_accuracy: 0.9371
47616/60000 [======================>.......] - ETA: 21s - loss: 0.2060 - categorical_accuracy: 0.9371
47648/60000 [======================>.......] - ETA: 21s - loss: 0.2059 - categorical_accuracy: 0.9371
47680/60000 [======================>.......] - ETA: 20s - loss: 0.2058 - categorical_accuracy: 0.9372
47712/60000 [======================>.......] - ETA: 20s - loss: 0.2057 - categorical_accuracy: 0.9372
47744/60000 [======================>.......] - ETA: 20s - loss: 0.2056 - categorical_accuracy: 0.9372
47776/60000 [======================>.......] - ETA: 20s - loss: 0.2055 - categorical_accuracy: 0.9372
47808/60000 [======================>.......] - ETA: 20s - loss: 0.2054 - categorical_accuracy: 0.9372
47840/60000 [======================>.......] - ETA: 20s - loss: 0.2055 - categorical_accuracy: 0.9372
47904/60000 [======================>.......] - ETA: 20s - loss: 0.2053 - categorical_accuracy: 0.9373
47968/60000 [======================>.......] - ETA: 20s - loss: 0.2053 - categorical_accuracy: 0.9373
48032/60000 [=======================>......] - ETA: 20s - loss: 0.2051 - categorical_accuracy: 0.9374
48064/60000 [=======================>......] - ETA: 20s - loss: 0.2050 - categorical_accuracy: 0.9374
48096/60000 [=======================>......] - ETA: 20s - loss: 0.2049 - categorical_accuracy: 0.9374
48128/60000 [=======================>......] - ETA: 20s - loss: 0.2049 - categorical_accuracy: 0.9374
48192/60000 [=======================>......] - ETA: 20s - loss: 0.2047 - categorical_accuracy: 0.9375
48256/60000 [=======================>......] - ETA: 20s - loss: 0.2045 - categorical_accuracy: 0.9375
48288/60000 [=======================>......] - ETA: 19s - loss: 0.2045 - categorical_accuracy: 0.9375
48320/60000 [=======================>......] - ETA: 19s - loss: 0.2045 - categorical_accuracy: 0.9375
48352/60000 [=======================>......] - ETA: 19s - loss: 0.2044 - categorical_accuracy: 0.9376
48416/60000 [=======================>......] - ETA: 19s - loss: 0.2042 - categorical_accuracy: 0.9376
48448/60000 [=======================>......] - ETA: 19s - loss: 0.2043 - categorical_accuracy: 0.9376
48480/60000 [=======================>......] - ETA: 19s - loss: 0.2042 - categorical_accuracy: 0.9376
48512/60000 [=======================>......] - ETA: 19s - loss: 0.2041 - categorical_accuracy: 0.9376
48544/60000 [=======================>......] - ETA: 19s - loss: 0.2040 - categorical_accuracy: 0.9377
48576/60000 [=======================>......] - ETA: 19s - loss: 0.2039 - categorical_accuracy: 0.9377
48608/60000 [=======================>......] - ETA: 19s - loss: 0.2038 - categorical_accuracy: 0.9377
48640/60000 [=======================>......] - ETA: 19s - loss: 0.2037 - categorical_accuracy: 0.9378
48672/60000 [=======================>......] - ETA: 19s - loss: 0.2036 - categorical_accuracy: 0.9378
48704/60000 [=======================>......] - ETA: 19s - loss: 0.2036 - categorical_accuracy: 0.9378
48736/60000 [=======================>......] - ETA: 19s - loss: 0.2035 - categorical_accuracy: 0.9378
48768/60000 [=======================>......] - ETA: 19s - loss: 0.2034 - categorical_accuracy: 0.9378
48800/60000 [=======================>......] - ETA: 19s - loss: 0.2033 - categorical_accuracy: 0.9379
48832/60000 [=======================>......] - ETA: 19s - loss: 0.2032 - categorical_accuracy: 0.9379
48864/60000 [=======================>......] - ETA: 18s - loss: 0.2031 - categorical_accuracy: 0.9380
48896/60000 [=======================>......] - ETA: 18s - loss: 0.2031 - categorical_accuracy: 0.9379
48928/60000 [=======================>......] - ETA: 18s - loss: 0.2029 - categorical_accuracy: 0.9380
48960/60000 [=======================>......] - ETA: 18s - loss: 0.2028 - categorical_accuracy: 0.9380
48992/60000 [=======================>......] - ETA: 18s - loss: 0.2028 - categorical_accuracy: 0.9380
49024/60000 [=======================>......] - ETA: 18s - loss: 0.2027 - categorical_accuracy: 0.9380
49056/60000 [=======================>......] - ETA: 18s - loss: 0.2026 - categorical_accuracy: 0.9381
49088/60000 [=======================>......] - ETA: 18s - loss: 0.2026 - categorical_accuracy: 0.9381
49120/60000 [=======================>......] - ETA: 18s - loss: 0.2025 - categorical_accuracy: 0.9381
49152/60000 [=======================>......] - ETA: 18s - loss: 0.2024 - categorical_accuracy: 0.9382
49184/60000 [=======================>......] - ETA: 18s - loss: 0.2024 - categorical_accuracy: 0.9382
49216/60000 [=======================>......] - ETA: 18s - loss: 0.2026 - categorical_accuracy: 0.9382
49248/60000 [=======================>......] - ETA: 18s - loss: 0.2024 - categorical_accuracy: 0.9382
49280/60000 [=======================>......] - ETA: 18s - loss: 0.2024 - categorical_accuracy: 0.9382
49312/60000 [=======================>......] - ETA: 18s - loss: 0.2023 - categorical_accuracy: 0.9382
49376/60000 [=======================>......] - ETA: 18s - loss: 0.2023 - categorical_accuracy: 0.9382
49408/60000 [=======================>......] - ETA: 18s - loss: 0.2023 - categorical_accuracy: 0.9383
49440/60000 [=======================>......] - ETA: 17s - loss: 0.2022 - categorical_accuracy: 0.9383
49504/60000 [=======================>......] - ETA: 17s - loss: 0.2020 - categorical_accuracy: 0.9383
49568/60000 [=======================>......] - ETA: 17s - loss: 0.2019 - categorical_accuracy: 0.9384
49600/60000 [=======================>......] - ETA: 17s - loss: 0.2018 - categorical_accuracy: 0.9384
49632/60000 [=======================>......] - ETA: 17s - loss: 0.2017 - categorical_accuracy: 0.9384
49664/60000 [=======================>......] - ETA: 17s - loss: 0.2016 - categorical_accuracy: 0.9384
49696/60000 [=======================>......] - ETA: 17s - loss: 0.2015 - categorical_accuracy: 0.9384
49728/60000 [=======================>......] - ETA: 17s - loss: 0.2014 - categorical_accuracy: 0.9385
49760/60000 [=======================>......] - ETA: 17s - loss: 0.2013 - categorical_accuracy: 0.9385
49792/60000 [=======================>......] - ETA: 17s - loss: 0.2012 - categorical_accuracy: 0.9385
49824/60000 [=======================>......] - ETA: 17s - loss: 0.2011 - categorical_accuracy: 0.9385
49856/60000 [=======================>......] - ETA: 17s - loss: 0.2011 - categorical_accuracy: 0.9386
49888/60000 [=======================>......] - ETA: 17s - loss: 0.2010 - categorical_accuracy: 0.9386
49920/60000 [=======================>......] - ETA: 17s - loss: 0.2011 - categorical_accuracy: 0.9386
49952/60000 [=======================>......] - ETA: 17s - loss: 0.2010 - categorical_accuracy: 0.9386
49984/60000 [=======================>......] - ETA: 17s - loss: 0.2009 - categorical_accuracy: 0.9387
50016/60000 [========================>.....] - ETA: 16s - loss: 0.2009 - categorical_accuracy: 0.9387
50048/60000 [========================>.....] - ETA: 16s - loss: 0.2010 - categorical_accuracy: 0.9387
50112/60000 [========================>.....] - ETA: 16s - loss: 0.2009 - categorical_accuracy: 0.9387
50176/60000 [========================>.....] - ETA: 16s - loss: 0.2007 - categorical_accuracy: 0.9388
50208/60000 [========================>.....] - ETA: 16s - loss: 0.2006 - categorical_accuracy: 0.9388
50240/60000 [========================>.....] - ETA: 16s - loss: 0.2005 - categorical_accuracy: 0.9388
50272/60000 [========================>.....] - ETA: 16s - loss: 0.2005 - categorical_accuracy: 0.9388
50304/60000 [========================>.....] - ETA: 16s - loss: 0.2004 - categorical_accuracy: 0.9388
50336/60000 [========================>.....] - ETA: 16s - loss: 0.2003 - categorical_accuracy: 0.9388
50368/60000 [========================>.....] - ETA: 16s - loss: 0.2003 - categorical_accuracy: 0.9389
50400/60000 [========================>.....] - ETA: 16s - loss: 0.2002 - categorical_accuracy: 0.9389
50432/60000 [========================>.....] - ETA: 16s - loss: 0.2001 - categorical_accuracy: 0.9389
50464/60000 [========================>.....] - ETA: 16s - loss: 0.2000 - categorical_accuracy: 0.9389
50496/60000 [========================>.....] - ETA: 16s - loss: 0.2000 - categorical_accuracy: 0.9389
50528/60000 [========================>.....] - ETA: 16s - loss: 0.2000 - categorical_accuracy: 0.9390
50560/60000 [========================>.....] - ETA: 16s - loss: 0.1999 - categorical_accuracy: 0.9390
50592/60000 [========================>.....] - ETA: 16s - loss: 0.1998 - categorical_accuracy: 0.9390
50624/60000 [========================>.....] - ETA: 15s - loss: 0.1997 - categorical_accuracy: 0.9390
50656/60000 [========================>.....] - ETA: 15s - loss: 0.1996 - categorical_accuracy: 0.9391
50688/60000 [========================>.....] - ETA: 15s - loss: 0.1995 - categorical_accuracy: 0.9391
50720/60000 [========================>.....] - ETA: 15s - loss: 0.1995 - categorical_accuracy: 0.9391
50752/60000 [========================>.....] - ETA: 15s - loss: 0.1994 - categorical_accuracy: 0.9391
50784/60000 [========================>.....] - ETA: 15s - loss: 0.1993 - categorical_accuracy: 0.9392
50816/60000 [========================>.....] - ETA: 15s - loss: 0.1992 - categorical_accuracy: 0.9392
50848/60000 [========================>.....] - ETA: 15s - loss: 0.1991 - categorical_accuracy: 0.9392
50880/60000 [========================>.....] - ETA: 15s - loss: 0.1991 - categorical_accuracy: 0.9392
50912/60000 [========================>.....] - ETA: 15s - loss: 0.1989 - categorical_accuracy: 0.9393
50944/60000 [========================>.....] - ETA: 15s - loss: 0.1988 - categorical_accuracy: 0.9393
50976/60000 [========================>.....] - ETA: 15s - loss: 0.1988 - categorical_accuracy: 0.9393
51008/60000 [========================>.....] - ETA: 15s - loss: 0.1987 - categorical_accuracy: 0.9394
51040/60000 [========================>.....] - ETA: 15s - loss: 0.1986 - categorical_accuracy: 0.9394
51072/60000 [========================>.....] - ETA: 15s - loss: 0.1985 - categorical_accuracy: 0.9394
51104/60000 [========================>.....] - ETA: 15s - loss: 0.1984 - categorical_accuracy: 0.9395
51136/60000 [========================>.....] - ETA: 15s - loss: 0.1984 - categorical_accuracy: 0.9395
51168/60000 [========================>.....] - ETA: 15s - loss: 0.1985 - categorical_accuracy: 0.9395
51200/60000 [========================>.....] - ETA: 14s - loss: 0.1984 - categorical_accuracy: 0.9395
51232/60000 [========================>.....] - ETA: 14s - loss: 0.1983 - categorical_accuracy: 0.9395
51264/60000 [========================>.....] - ETA: 14s - loss: 0.1982 - categorical_accuracy: 0.9395
51296/60000 [========================>.....] - ETA: 14s - loss: 0.1982 - categorical_accuracy: 0.9395
51328/60000 [========================>.....] - ETA: 14s - loss: 0.1982 - categorical_accuracy: 0.9395
51360/60000 [========================>.....] - ETA: 14s - loss: 0.1982 - categorical_accuracy: 0.9396
51392/60000 [========================>.....] - ETA: 14s - loss: 0.1982 - categorical_accuracy: 0.9396
51424/60000 [========================>.....] - ETA: 14s - loss: 0.1982 - categorical_accuracy: 0.9396
51488/60000 [========================>.....] - ETA: 14s - loss: 0.1980 - categorical_accuracy: 0.9397
51520/60000 [========================>.....] - ETA: 14s - loss: 0.1979 - categorical_accuracy: 0.9397
51552/60000 [========================>.....] - ETA: 14s - loss: 0.1978 - categorical_accuracy: 0.9397
51584/60000 [========================>.....] - ETA: 14s - loss: 0.1977 - categorical_accuracy: 0.9398
51616/60000 [========================>.....] - ETA: 14s - loss: 0.1977 - categorical_accuracy: 0.9398
51648/60000 [========================>.....] - ETA: 14s - loss: 0.1977 - categorical_accuracy: 0.9398
51680/60000 [========================>.....] - ETA: 14s - loss: 0.1976 - categorical_accuracy: 0.9398
51712/60000 [========================>.....] - ETA: 14s - loss: 0.1975 - categorical_accuracy: 0.9398
51744/60000 [========================>.....] - ETA: 14s - loss: 0.1974 - categorical_accuracy: 0.9399
51776/60000 [========================>.....] - ETA: 14s - loss: 0.1973 - categorical_accuracy: 0.9399
51808/60000 [========================>.....] - ETA: 13s - loss: 0.1973 - categorical_accuracy: 0.9399
51872/60000 [========================>.....] - ETA: 13s - loss: 0.1971 - categorical_accuracy: 0.9400
51904/60000 [========================>.....] - ETA: 13s - loss: 0.1970 - categorical_accuracy: 0.9400
51936/60000 [========================>.....] - ETA: 13s - loss: 0.1969 - categorical_accuracy: 0.9400
51968/60000 [========================>.....] - ETA: 13s - loss: 0.1968 - categorical_accuracy: 0.9400
52000/60000 [=========================>....] - ETA: 13s - loss: 0.1967 - categorical_accuracy: 0.9400
52032/60000 [=========================>....] - ETA: 13s - loss: 0.1966 - categorical_accuracy: 0.9401
52064/60000 [=========================>....] - ETA: 13s - loss: 0.1965 - categorical_accuracy: 0.9401
52096/60000 [=========================>....] - ETA: 13s - loss: 0.1965 - categorical_accuracy: 0.9401
52128/60000 [=========================>....] - ETA: 13s - loss: 0.1964 - categorical_accuracy: 0.9401
52160/60000 [=========================>....] - ETA: 13s - loss: 0.1963 - categorical_accuracy: 0.9401
52192/60000 [=========================>....] - ETA: 13s - loss: 0.1962 - categorical_accuracy: 0.9402
52224/60000 [=========================>....] - ETA: 13s - loss: 0.1961 - categorical_accuracy: 0.9402
52256/60000 [=========================>....] - ETA: 13s - loss: 0.1961 - categorical_accuracy: 0.9402
52288/60000 [=========================>....] - ETA: 13s - loss: 0.1961 - categorical_accuracy: 0.9402
52320/60000 [=========================>....] - ETA: 13s - loss: 0.1961 - categorical_accuracy: 0.9403
52352/60000 [=========================>....] - ETA: 13s - loss: 0.1960 - categorical_accuracy: 0.9403
52384/60000 [=========================>....] - ETA: 12s - loss: 0.1959 - categorical_accuracy: 0.9403
52416/60000 [=========================>....] - ETA: 12s - loss: 0.1959 - categorical_accuracy: 0.9403
52448/60000 [=========================>....] - ETA: 12s - loss: 0.1958 - categorical_accuracy: 0.9403
52480/60000 [=========================>....] - ETA: 12s - loss: 0.1957 - categorical_accuracy: 0.9404
52512/60000 [=========================>....] - ETA: 12s - loss: 0.1956 - categorical_accuracy: 0.9404
52544/60000 [=========================>....] - ETA: 12s - loss: 0.1955 - categorical_accuracy: 0.9404
52576/60000 [=========================>....] - ETA: 12s - loss: 0.1955 - categorical_accuracy: 0.9404
52608/60000 [=========================>....] - ETA: 12s - loss: 0.1954 - categorical_accuracy: 0.9405
52672/60000 [=========================>....] - ETA: 12s - loss: 0.1955 - categorical_accuracy: 0.9404
52736/60000 [=========================>....] - ETA: 12s - loss: 0.1955 - categorical_accuracy: 0.9405
52768/60000 [=========================>....] - ETA: 12s - loss: 0.1956 - categorical_accuracy: 0.9405
52800/60000 [=========================>....] - ETA: 12s - loss: 0.1955 - categorical_accuracy: 0.9405
52832/60000 [=========================>....] - ETA: 12s - loss: 0.1954 - categorical_accuracy: 0.9405
52864/60000 [=========================>....] - ETA: 12s - loss: 0.1954 - categorical_accuracy: 0.9405
52896/60000 [=========================>....] - ETA: 12s - loss: 0.1953 - categorical_accuracy: 0.9405
52928/60000 [=========================>....] - ETA: 12s - loss: 0.1952 - categorical_accuracy: 0.9406
52960/60000 [=========================>....] - ETA: 11s - loss: 0.1952 - categorical_accuracy: 0.9406
52992/60000 [=========================>....] - ETA: 11s - loss: 0.1952 - categorical_accuracy: 0.9406
53024/60000 [=========================>....] - ETA: 11s - loss: 0.1951 - categorical_accuracy: 0.9406
53056/60000 [=========================>....] - ETA: 11s - loss: 0.1950 - categorical_accuracy: 0.9407
53088/60000 [=========================>....] - ETA: 11s - loss: 0.1950 - categorical_accuracy: 0.9407
53152/60000 [=========================>....] - ETA: 11s - loss: 0.1948 - categorical_accuracy: 0.9407
53184/60000 [=========================>....] - ETA: 11s - loss: 0.1947 - categorical_accuracy: 0.9408
53216/60000 [=========================>....] - ETA: 11s - loss: 0.1948 - categorical_accuracy: 0.9408
53248/60000 [=========================>....] - ETA: 11s - loss: 0.1947 - categorical_accuracy: 0.9408
53280/60000 [=========================>....] - ETA: 11s - loss: 0.1946 - categorical_accuracy: 0.9408
53312/60000 [=========================>....] - ETA: 11s - loss: 0.1945 - categorical_accuracy: 0.9408
53344/60000 [=========================>....] - ETA: 11s - loss: 0.1944 - categorical_accuracy: 0.9409
53376/60000 [=========================>....] - ETA: 11s - loss: 0.1944 - categorical_accuracy: 0.9409
53408/60000 [=========================>....] - ETA: 11s - loss: 0.1943 - categorical_accuracy: 0.9409
53440/60000 [=========================>....] - ETA: 11s - loss: 0.1944 - categorical_accuracy: 0.9409
53472/60000 [=========================>....] - ETA: 11s - loss: 0.1943 - categorical_accuracy: 0.9409
53504/60000 [=========================>....] - ETA: 11s - loss: 0.1942 - categorical_accuracy: 0.9410
53536/60000 [=========================>....] - ETA: 11s - loss: 0.1941 - categorical_accuracy: 0.9410
53568/60000 [=========================>....] - ETA: 10s - loss: 0.1940 - categorical_accuracy: 0.9410
53600/60000 [=========================>....] - ETA: 10s - loss: 0.1939 - categorical_accuracy: 0.9410
53632/60000 [=========================>....] - ETA: 10s - loss: 0.1938 - categorical_accuracy: 0.9411
53664/60000 [=========================>....] - ETA: 10s - loss: 0.1937 - categorical_accuracy: 0.9411
53696/60000 [=========================>....] - ETA: 10s - loss: 0.1936 - categorical_accuracy: 0.9412
53728/60000 [=========================>....] - ETA: 10s - loss: 0.1935 - categorical_accuracy: 0.9412
53760/60000 [=========================>....] - ETA: 10s - loss: 0.1935 - categorical_accuracy: 0.9412
53792/60000 [=========================>....] - ETA: 10s - loss: 0.1934 - categorical_accuracy: 0.9412
53824/60000 [=========================>....] - ETA: 10s - loss: 0.1934 - categorical_accuracy: 0.9412
53856/60000 [=========================>....] - ETA: 10s - loss: 0.1932 - categorical_accuracy: 0.9413
53888/60000 [=========================>....] - ETA: 10s - loss: 0.1931 - categorical_accuracy: 0.9413
53920/60000 [=========================>....] - ETA: 10s - loss: 0.1931 - categorical_accuracy: 0.9413
53952/60000 [=========================>....] - ETA: 10s - loss: 0.1932 - categorical_accuracy: 0.9413
53984/60000 [=========================>....] - ETA: 10s - loss: 0.1931 - categorical_accuracy: 0.9413
54016/60000 [==========================>...] - ETA: 10s - loss: 0.1930 - categorical_accuracy: 0.9413
54048/60000 [==========================>...] - ETA: 10s - loss: 0.1929 - categorical_accuracy: 0.9414
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1928 - categorical_accuracy: 0.9414
54112/60000 [==========================>...] - ETA: 10s - loss: 0.1927 - categorical_accuracy: 0.9414
54144/60000 [==========================>...] - ETA: 9s - loss: 0.1927 - categorical_accuracy: 0.9414 
54176/60000 [==========================>...] - ETA: 9s - loss: 0.1927 - categorical_accuracy: 0.9414
54240/60000 [==========================>...] - ETA: 9s - loss: 0.1926 - categorical_accuracy: 0.9415
54304/60000 [==========================>...] - ETA: 9s - loss: 0.1924 - categorical_accuracy: 0.9415
54368/60000 [==========================>...] - ETA: 9s - loss: 0.1923 - categorical_accuracy: 0.9416
54400/60000 [==========================>...] - ETA: 9s - loss: 0.1922 - categorical_accuracy: 0.9416
54464/60000 [==========================>...] - ETA: 9s - loss: 0.1921 - categorical_accuracy: 0.9416
54496/60000 [==========================>...] - ETA: 9s - loss: 0.1920 - categorical_accuracy: 0.9416
54528/60000 [==========================>...] - ETA: 9s - loss: 0.1919 - categorical_accuracy: 0.9417
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1918 - categorical_accuracy: 0.9417
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1918 - categorical_accuracy: 0.9417
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1917 - categorical_accuracy: 0.9418
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1916 - categorical_accuracy: 0.9418
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1916 - categorical_accuracy: 0.9418
54720/60000 [==========================>...] - ETA: 8s - loss: 0.1915 - categorical_accuracy: 0.9418
54752/60000 [==========================>...] - ETA: 8s - loss: 0.1914 - categorical_accuracy: 0.9418
54784/60000 [==========================>...] - ETA: 8s - loss: 0.1913 - categorical_accuracy: 0.9418
54848/60000 [==========================>...] - ETA: 8s - loss: 0.1911 - categorical_accuracy: 0.9419
54880/60000 [==========================>...] - ETA: 8s - loss: 0.1911 - categorical_accuracy: 0.9419
54912/60000 [==========================>...] - ETA: 8s - loss: 0.1911 - categorical_accuracy: 0.9419
54944/60000 [==========================>...] - ETA: 8s - loss: 0.1911 - categorical_accuracy: 0.9419
54976/60000 [==========================>...] - ETA: 8s - loss: 0.1910 - categorical_accuracy: 0.9419
55008/60000 [==========================>...] - ETA: 8s - loss: 0.1909 - categorical_accuracy: 0.9420
55040/60000 [==========================>...] - ETA: 8s - loss: 0.1908 - categorical_accuracy: 0.9420
55072/60000 [==========================>...] - ETA: 8s - loss: 0.1909 - categorical_accuracy: 0.9419
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1908 - categorical_accuracy: 0.9419
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1907 - categorical_accuracy: 0.9420
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1906 - categorical_accuracy: 0.9420
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1905 - categorical_accuracy: 0.9420
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1904 - categorical_accuracy: 0.9421
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1904 - categorical_accuracy: 0.9421
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1905 - categorical_accuracy: 0.9421
55328/60000 [==========================>...] - ETA: 7s - loss: 0.1904 - categorical_accuracy: 0.9421
55360/60000 [==========================>...] - ETA: 7s - loss: 0.1904 - categorical_accuracy: 0.9421
55392/60000 [==========================>...] - ETA: 7s - loss: 0.1904 - categorical_accuracy: 0.9421
55424/60000 [==========================>...] - ETA: 7s - loss: 0.1903 - categorical_accuracy: 0.9421
55456/60000 [==========================>...] - ETA: 7s - loss: 0.1903 - categorical_accuracy: 0.9421
55520/60000 [==========================>...] - ETA: 7s - loss: 0.1903 - categorical_accuracy: 0.9421
55584/60000 [==========================>...] - ETA: 7s - loss: 0.1903 - categorical_accuracy: 0.9421
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1902 - categorical_accuracy: 0.9421
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1903 - categorical_accuracy: 0.9422
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1902 - categorical_accuracy: 0.9422
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1902 - categorical_accuracy: 0.9422
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1902 - categorical_accuracy: 0.9421
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1902 - categorical_accuracy: 0.9422
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1901 - categorical_accuracy: 0.9422
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1900 - categorical_accuracy: 0.9422
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1899 - categorical_accuracy: 0.9422
55904/60000 [==========================>...] - ETA: 6s - loss: 0.1898 - categorical_accuracy: 0.9423
55936/60000 [==========================>...] - ETA: 6s - loss: 0.1898 - categorical_accuracy: 0.9423
55968/60000 [==========================>...] - ETA: 6s - loss: 0.1897 - categorical_accuracy: 0.9423
56000/60000 [===========================>..] - ETA: 6s - loss: 0.1896 - categorical_accuracy: 0.9423
56032/60000 [===========================>..] - ETA: 6s - loss: 0.1896 - categorical_accuracy: 0.9423
56064/60000 [===========================>..] - ETA: 6s - loss: 0.1895 - categorical_accuracy: 0.9424
56096/60000 [===========================>..] - ETA: 6s - loss: 0.1895 - categorical_accuracy: 0.9424
56128/60000 [===========================>..] - ETA: 6s - loss: 0.1895 - categorical_accuracy: 0.9424
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1894 - categorical_accuracy: 0.9424
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1894 - categorical_accuracy: 0.9424
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1892 - categorical_accuracy: 0.9425
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1891 - categorical_accuracy: 0.9425
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1891 - categorical_accuracy: 0.9425
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1890 - categorical_accuracy: 0.9425
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1889 - categorical_accuracy: 0.9426
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1888 - categorical_accuracy: 0.9426
56480/60000 [===========================>..] - ETA: 5s - loss: 0.1887 - categorical_accuracy: 0.9426
56512/60000 [===========================>..] - ETA: 5s - loss: 0.1887 - categorical_accuracy: 0.9426
56544/60000 [===========================>..] - ETA: 5s - loss: 0.1886 - categorical_accuracy: 0.9427
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1886 - categorical_accuracy: 0.9427
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1885 - categorical_accuracy: 0.9427
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1884 - categorical_accuracy: 0.9427
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1883 - categorical_accuracy: 0.9427
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1882 - categorical_accuracy: 0.9427
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1882 - categorical_accuracy: 0.9428
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1882 - categorical_accuracy: 0.9428
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1880 - categorical_accuracy: 0.9428
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1879 - categorical_accuracy: 0.9428
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1877 - categorical_accuracy: 0.9429
57088/60000 [===========================>..] - ETA: 4s - loss: 0.1876 - categorical_accuracy: 0.9429
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1876 - categorical_accuracy: 0.9429
57152/60000 [===========================>..] - ETA: 4s - loss: 0.1875 - categorical_accuracy: 0.9429
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1874 - categorical_accuracy: 0.9429
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1874 - categorical_accuracy: 0.9429
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1873 - categorical_accuracy: 0.9430
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1872 - categorical_accuracy: 0.9430
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1872 - categorical_accuracy: 0.9430
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1871 - categorical_accuracy: 0.9430
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1870 - categorical_accuracy: 0.9431
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1869 - categorical_accuracy: 0.9431
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1869 - categorical_accuracy: 0.9431
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1869 - categorical_accuracy: 0.9431
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1868 - categorical_accuracy: 0.9432
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1868 - categorical_accuracy: 0.9432
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1868 - categorical_accuracy: 0.9432
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1868 - categorical_accuracy: 0.9431
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1867 - categorical_accuracy: 0.9432
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1867 - categorical_accuracy: 0.9432
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1866 - categorical_accuracy: 0.9432
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1869 - categorical_accuracy: 0.9432
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1868 - categorical_accuracy: 0.9432
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1868 - categorical_accuracy: 0.9432
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1867 - categorical_accuracy: 0.9432
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1867 - categorical_accuracy: 0.9432
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1866 - categorical_accuracy: 0.9432
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1865 - categorical_accuracy: 0.9433
58016/60000 [============================>.] - ETA: 3s - loss: 0.1865 - categorical_accuracy: 0.9433
58048/60000 [============================>.] - ETA: 3s - loss: 0.1864 - categorical_accuracy: 0.9433
58080/60000 [============================>.] - ETA: 3s - loss: 0.1863 - categorical_accuracy: 0.9433
58112/60000 [============================>.] - ETA: 3s - loss: 0.1863 - categorical_accuracy: 0.9433
58144/60000 [============================>.] - ETA: 3s - loss: 0.1862 - categorical_accuracy: 0.9433
58176/60000 [============================>.] - ETA: 3s - loss: 0.1862 - categorical_accuracy: 0.9433
58208/60000 [============================>.] - ETA: 3s - loss: 0.1861 - categorical_accuracy: 0.9433
58240/60000 [============================>.] - ETA: 2s - loss: 0.1861 - categorical_accuracy: 0.9433
58272/60000 [============================>.] - ETA: 2s - loss: 0.1860 - categorical_accuracy: 0.9434
58304/60000 [============================>.] - ETA: 2s - loss: 0.1861 - categorical_accuracy: 0.9434
58336/60000 [============================>.] - ETA: 2s - loss: 0.1861 - categorical_accuracy: 0.9434
58368/60000 [============================>.] - ETA: 2s - loss: 0.1862 - categorical_accuracy: 0.9433
58400/60000 [============================>.] - ETA: 2s - loss: 0.1861 - categorical_accuracy: 0.9434
58432/60000 [============================>.] - ETA: 2s - loss: 0.1860 - categorical_accuracy: 0.9434
58464/60000 [============================>.] - ETA: 2s - loss: 0.1862 - categorical_accuracy: 0.9433
58496/60000 [============================>.] - ETA: 2s - loss: 0.1861 - categorical_accuracy: 0.9434
58528/60000 [============================>.] - ETA: 2s - loss: 0.1860 - categorical_accuracy: 0.9434
58560/60000 [============================>.] - ETA: 2s - loss: 0.1860 - categorical_accuracy: 0.9434
58592/60000 [============================>.] - ETA: 2s - loss: 0.1860 - categorical_accuracy: 0.9434
58624/60000 [============================>.] - ETA: 2s - loss: 0.1859 - categorical_accuracy: 0.9434
58656/60000 [============================>.] - ETA: 2s - loss: 0.1859 - categorical_accuracy: 0.9434
58688/60000 [============================>.] - ETA: 2s - loss: 0.1858 - categorical_accuracy: 0.9434
58752/60000 [============================>.] - ETA: 2s - loss: 0.1857 - categorical_accuracy: 0.9435
58784/60000 [============================>.] - ETA: 2s - loss: 0.1856 - categorical_accuracy: 0.9435
58816/60000 [============================>.] - ETA: 2s - loss: 0.1856 - categorical_accuracy: 0.9435
58848/60000 [============================>.] - ETA: 1s - loss: 0.1855 - categorical_accuracy: 0.9435
58880/60000 [============================>.] - ETA: 1s - loss: 0.1854 - categorical_accuracy: 0.9435
58912/60000 [============================>.] - ETA: 1s - loss: 0.1854 - categorical_accuracy: 0.9435
58944/60000 [============================>.] - ETA: 1s - loss: 0.1853 - categorical_accuracy: 0.9436
58976/60000 [============================>.] - ETA: 1s - loss: 0.1852 - categorical_accuracy: 0.9436
59008/60000 [============================>.] - ETA: 1s - loss: 0.1852 - categorical_accuracy: 0.9436
59040/60000 [============================>.] - ETA: 1s - loss: 0.1852 - categorical_accuracy: 0.9436
59072/60000 [============================>.] - ETA: 1s - loss: 0.1852 - categorical_accuracy: 0.9436
59104/60000 [============================>.] - ETA: 1s - loss: 0.1852 - categorical_accuracy: 0.9436
59136/60000 [============================>.] - ETA: 1s - loss: 0.1853 - categorical_accuracy: 0.9436
59168/60000 [============================>.] - ETA: 1s - loss: 0.1852 - categorical_accuracy: 0.9436
59200/60000 [============================>.] - ETA: 1s - loss: 0.1851 - categorical_accuracy: 0.9436
59232/60000 [============================>.] - ETA: 1s - loss: 0.1850 - categorical_accuracy: 0.9437
59264/60000 [============================>.] - ETA: 1s - loss: 0.1849 - categorical_accuracy: 0.9437
59296/60000 [============================>.] - ETA: 1s - loss: 0.1849 - categorical_accuracy: 0.9437
59328/60000 [============================>.] - ETA: 1s - loss: 0.1849 - categorical_accuracy: 0.9437
59360/60000 [============================>.] - ETA: 1s - loss: 0.1848 - categorical_accuracy: 0.9437
59392/60000 [============================>.] - ETA: 1s - loss: 0.1850 - categorical_accuracy: 0.9437
59424/60000 [============================>.] - ETA: 0s - loss: 0.1849 - categorical_accuracy: 0.9437
59456/60000 [============================>.] - ETA: 0s - loss: 0.1848 - categorical_accuracy: 0.9437
59488/60000 [============================>.] - ETA: 0s - loss: 0.1847 - categorical_accuracy: 0.9438
59520/60000 [============================>.] - ETA: 0s - loss: 0.1847 - categorical_accuracy: 0.9438
59552/60000 [============================>.] - ETA: 0s - loss: 0.1846 - categorical_accuracy: 0.9438
59584/60000 [============================>.] - ETA: 0s - loss: 0.1845 - categorical_accuracy: 0.9438
59616/60000 [============================>.] - ETA: 0s - loss: 0.1845 - categorical_accuracy: 0.9438
59648/60000 [============================>.] - ETA: 0s - loss: 0.1845 - categorical_accuracy: 0.9438
59680/60000 [============================>.] - ETA: 0s - loss: 0.1845 - categorical_accuracy: 0.9438
59712/60000 [============================>.] - ETA: 0s - loss: 0.1844 - categorical_accuracy: 0.9438
59744/60000 [============================>.] - ETA: 0s - loss: 0.1843 - categorical_accuracy: 0.9439
59776/60000 [============================>.] - ETA: 0s - loss: 0.1842 - categorical_accuracy: 0.9439
59808/60000 [============================>.] - ETA: 0s - loss: 0.1842 - categorical_accuracy: 0.9439
59840/60000 [============================>.] - ETA: 0s - loss: 0.1841 - categorical_accuracy: 0.9439
59872/60000 [============================>.] - ETA: 0s - loss: 0.1841 - categorical_accuracy: 0.9439
59904/60000 [============================>.] - ETA: 0s - loss: 0.1840 - categorical_accuracy: 0.9439
59968/60000 [============================>.] - ETA: 0s - loss: 0.1839 - categorical_accuracy: 0.9440
60000/60000 [==============================] - 105s 2ms/step - loss: 0.1838 - categorical_accuracy: 0.9440 - val_loss: 0.0423 - val_categorical_accuracy: 0.9855

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 17s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 3s
  672/10000 [=>............................] - ETA: 3s
  864/10000 [=>............................] - ETA: 3s
 1056/10000 [==>...........................] - ETA: 3s
 1216/10000 [==>...........................] - ETA: 3s
 1376/10000 [===>..........................] - ETA: 3s
 1536/10000 [===>..........................] - ETA: 3s
 1664/10000 [===>..........................] - ETA: 3s
 1792/10000 [====>.........................] - ETA: 3s
 1952/10000 [====>.........................] - ETA: 2s
 2112/10000 [=====>........................] - ETA: 2s
 2272/10000 [=====>........................] - ETA: 2s
 2432/10000 [======>.......................] - ETA: 2s
 2592/10000 [======>.......................] - ETA: 2s
 2784/10000 [=======>......................] - ETA: 2s
 2976/10000 [=======>......................] - ETA: 2s
 3136/10000 [========>.....................] - ETA: 2s
 3328/10000 [========>.....................] - ETA: 2s
 3488/10000 [=========>....................] - ETA: 2s
 3648/10000 [=========>....................] - ETA: 2s
 3808/10000 [==========>...................] - ETA: 2s
 3968/10000 [==========>...................] - ETA: 2s
 4128/10000 [===========>..................] - ETA: 2s
 4288/10000 [===========>..................] - ETA: 1s
 4448/10000 [============>.................] - ETA: 1s
 4608/10000 [============>.................] - ETA: 1s
 4768/10000 [=============>................] - ETA: 1s
 4960/10000 [=============>................] - ETA: 1s
 5152/10000 [==============>...............] - ETA: 1s
 5312/10000 [==============>...............] - ETA: 1s
 5504/10000 [===============>..............] - ETA: 1s
 5664/10000 [===============>..............] - ETA: 1s
 5824/10000 [================>.............] - ETA: 1s
 5984/10000 [================>.............] - ETA: 1s
 6144/10000 [=================>............] - ETA: 1s
 6304/10000 [=================>............] - ETA: 1s
 6464/10000 [==================>...........] - ETA: 1s
 6624/10000 [==================>...........] - ETA: 1s
 6784/10000 [===================>..........] - ETA: 1s
 6976/10000 [===================>..........] - ETA: 1s
 7168/10000 [====================>.........] - ETA: 0s
 7328/10000 [====================>.........] - ETA: 0s
 7520/10000 [=====================>........] - ETA: 0s
 7680/10000 [======================>.......] - ETA: 0s
 7872/10000 [======================>.......] - ETA: 0s
 8064/10000 [=======================>......] - ETA: 0s
 8256/10000 [=======================>......] - ETA: 0s
 8416/10000 [========================>.....] - ETA: 0s
 8608/10000 [========================>.....] - ETA: 0s
 8768/10000 [=========================>....] - ETA: 0s
 8928/10000 [=========================>....] - ETA: 0s
 9088/10000 [==========================>...] - ETA: 0s
 9248/10000 [==========================>...] - ETA: 0s
 9408/10000 [===========================>..] - ETA: 0s
 9568/10000 [===========================>..] - ETA: 0s
 9728/10000 [============================>.] - ETA: 0s
 9888/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 334us/step
[[9.05316355e-09 6.95060578e-08 6.78746233e-07 ... 9.99995828e-01
  5.15026644e-09 2.02978140e-06]
 [2.46794184e-06 5.11531141e-07 9.99992013e-01 ... 3.68035202e-08
  3.05404205e-06 8.97265595e-10]
 [6.51969276e-06 9.99639630e-01 6.10523348e-05 ... 1.13178648e-04
  7.09455853e-05 4.67755262e-06]
 ...
 [1.94400518e-09 6.43232170e-08 4.25134106e-09 ... 5.97491237e-07
  1.22673146e-06 1.10780247e-05]
 [8.62728712e-07 2.58729411e-08 1.14902265e-08 ... 5.77804826e-08
  1.55064394e-03 1.13798505e-05]
 [6.02917407e-06 5.26349311e-07 6.28927864e-06 ... 4.54931648e-09
  3.53245497e-07 6.42034266e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04232305201796116, 'accuracy_test:': 0.9854999780654907}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   a8c5ebe..3f0ed83  master     -> origin/master
Updating a8c5ebe..3f0ed83
Fast-forward
 error_list/20200515/list_log_benchmark_20200515.md | 180 ++++++++++-----------
 .../20200515/list_log_dataloader_20200515.md       |   2 +-
 2 files changed, 86 insertions(+), 96 deletions(-)
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
[master fa51b3b] ml_store
 1 file changed, 1933 insertions(+)
To github.com:arita37/mlmodels_store.git
   3f0ed83..fa51b3b  master -> master





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
{'loss': 0.43129290640354156, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 08:33:50.354447: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master c5490ba] ml_store
 1 file changed, 233 insertions(+)
To github.com:arita37/mlmodels_store.git
   fa51b3b..c5490ba  master -> master





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
