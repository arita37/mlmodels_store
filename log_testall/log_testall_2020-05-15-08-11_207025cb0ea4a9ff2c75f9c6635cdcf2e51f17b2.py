
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
[master 9d6f06d] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   c5490ba..9d6f06d  master -> master





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
	Data preprocessing and feature engineering runtime = 0.29s ...
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
 40%|████      | 2/5 [00:17<00:26,  8.75s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.9733590063989046, 'learning_rate': 0.01558370990297927, 'min_data_in_leaf': 28, 'num_leaves': 37} and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef%\xc1\xc9x\x02\xdeX\r\x00\x00\x00learning_rateq\x02G?\x8f\xeaZ#\x11\xb5\xadX\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K%u.' and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef%\xc1\xc9x\x02\xdeX\r\x00\x00\x00learning_rateq\x02G?\x8f\xeaZ#\x11\xb5\xadX\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K%u.' and reward: 0.39
 60%|██████    | 3/5 [00:36<00:23, 11.70s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.8726864378823265, 'learning_rate': 0.022122351489801146, 'min_data_in_leaf': 17, 'num_leaves': 45} and reward: 0.3918
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\xed\x0c\x1b\xcb\xc1&X\r\x00\x00\x00learning_rateq\x02G?\x96\xa7=\xe0\xa3"\xe5X\x10\x00\x00\x00min_data_in_leafq\x03K\x11X\n\x00\x00\x00num_leavesq\x04K-u.' and reward: 0.3918
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\xed\x0c\x1b\xcb\xc1&X\r\x00\x00\x00learning_rateq\x02G?\x96\xa7=\xe0\xa3"\xe5X\x10\x00\x00\x00min_data_in_leafq\x03K\x11X\n\x00\x00\x00num_leavesq\x04K-u.' and reward: 0.3918
 80%|████████  | 4/5 [00:56<00:14, 14.27s/it] 80%|████████  | 4/5 [00:56<00:14, 14.09s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8369108808726807, 'learning_rate': 0.0056632088321769535, 'min_data_in_leaf': 23, 'num_leaves': 33} and reward: 0.384
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xc7\xf9S\xe0x\x8dX\r\x00\x00\x00learning_rateq\x02G?w2N\x0b\x98\tnX\x10\x00\x00\x00min_data_in_leafq\x03K\x17X\n\x00\x00\x00num_leavesq\x04K!u.' and reward: 0.384
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xc7\xf9S\xe0x\x8dX\r\x00\x00\x00learning_rateq\x02G?w2N\x0b\x98\tnX\x10\x00\x00\x00min_data_in_leafq\x03K\x17X\n\x00\x00\x00num_leavesq\x04K!u.' and reward: 0.384
Time for Gradient Boosting hyperparameter optimization: 73.5320143699646
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.8726864378823265, 'learning_rate': 0.022122351489801146, 'min_data_in_leaf': 17, 'num_leaves': 45}
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
 40%|████      | 2/5 [00:55<01:23, 27.99s/it] 40%|████      | 2/5 [00:55<01:23, 27.99s/it]
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
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.32664652346859674, 'embedding_size_factor': 0.9376008295002153, 'layers.choice': 1, 'learning_rate': 0.002903183387470723, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 2.995783495691961e-08} and reward: 0.3788
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\xe7\xc6\xd1\xe9\x95\x01X\x15\x00\x00\x00embedding_size_factorq\x03G?\xee\x00\xd3tl\xfd\x1bX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?g\xc8j\xb6\x82\xe2\xbaX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>`\x15_\x9c\xb2\x979u.' and reward: 0.3788
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\xe7\xc6\xd1\xe9\x95\x01X\x15\x00\x00\x00embedding_size_factorq\x03G?\xee\x00\xd3tl\xfd\x1bX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?g\xc8j\xb6\x82\xe2\xbaX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>`\x15_\x9c\xb2\x979u.' and reward: 0.3788
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 158.01817965507507
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.71s of the -115.33s of remaining time.
Ensemble size: 74
Ensemble weights: 
[0.97297297 0.01351351 0.01351351 0.         0.         0.        ]
	0.392	 = Validation accuracy score
	1.57s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 236.97s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f8a65b99b00>

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   9d6f06d..a161df8  master     -> origin/master
Updating 9d6f06d..a161df8
Fast-forward
 .../20200515/list_log_dataloader_20200515.md       |    2 +-
 .../20200515/list_log_pullrequest_20200515.md      |    2 +-
 ...-12_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 2489 ++++++++++++++++++++
 3 files changed, 2491 insertions(+), 2 deletions(-)
 create mode 100644 log_benchmark/log_benchmark_2020-05-15-08-12_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
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
[master 066a9a5] ml_store
 1 file changed, 236 insertions(+)
To github.com:arita37/mlmodels_store.git
   a161df8..066a9a5  master -> master





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
[master c2c6a29] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   066a9a5..c2c6a29  master -> master





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
[master bdf43d0] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   c2c6a29..bdf43d0  master -> master





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
{'roc_auc_score': 0.9545454545454546}

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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7fa4a6de8f98> 

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
[master e4587ea] ml_store
 1 file changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   bdf43d0..e4587ea  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 3.54133613e-01  2.11124755e-01  9.21450069e-01  1.65275673e-02
   9.03945451e-01  1.77187720e-01  9.54250872e-02 -1.11647002e+00
   8.09271010e-02  6.07501958e-02]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 6.67591795e-01 -4.52524973e-01 -6.05981321e-01  1.16128569e+00
  -1.44620987e+00  1.06996554e+00  1.92381543e+00 -1.04553425e+00
   3.55284507e-01  1.80358898e+00]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 5.63077902e-01 -1.17598267e+00 -1.74180344e-01  1.01012718e+00
   1.06796368e+00  9.20017933e-01 -1.68198840e-01 -1.95057341e-01
   8.05393424e-01  4.61164100e-01]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 1.46893146e+00 -1.47115693e+00  5.85910431e-01 -8.30171895e-01
   1.03345052e+00 -8.80577600e-01 -9.55425262e-01 -2.79097722e-01
   1.62284909e+00  2.06578332e+00]
 [ 1.24549398e+00 -7.22391905e-01  1.11813340e+00  1.09899633e+00
   1.00277655e+00 -9.01634490e-01 -5.32234021e-01 -8.22467189e-01
   7.21711292e-01  6.74396105e-01]
 [ 1.64661853e+00 -1.52568032e+00 -6.06998398e-01  7.95026094e-01
   1.08480038e+00 -3.74438319e-01  4.29526140e-01  1.34048197e-01
   1.20205486e+00  1.06222724e-01]
 [ 6.81889336e-01 -1.15498263e+00  1.22895559e+00 -1.77632196e-01
   9.98545187e-01 -1.51045638e+00 -2.75846063e-01  1.01120706e+00
  -1.47656266e+00  1.30970591e+00]
 [ 9.26869810e-01  3.92334911e-01 -4.23478297e-01  4.48380651e-01
  -1.09230828e+00  1.12532350e+00 -9.48439656e-01  1.04053390e-01
   5.28003422e-01  1.00796648e+00]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 9.97855163e-01 -6.00138799e-01  4.57947076e-01  1.46765263e-01
  -9.33557290e-01  5.71804879e-01  5.72962726e-01 -3.68176565e-02
   1.12368489e-01 -1.78175491e-02]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 1.34728643e+00 -3.64538050e-01  8.07509886e-02 -4.59717681e-01
  -8.89487596e-01  1.70548352e+00  9.49961101e-02  2.40505552e-01
  -9.99426501e-01 -7.67803746e-01]
 [ 6.23688521e-01  1.20660790e+00  9.03999174e-01 -2.82863552e-01
  -1.18913787e+00 -2.66326884e-01  1.42361443e+00  1.06897162e+00
   4.03714310e-02  1.57546791e+00]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 7.73703613e-01  1.27852808e+00 -2.11416392e+00 -4.42229280e-01
   1.06821044e+00  3.23527354e-01 -2.50644065e+00 -1.09991490e-01
   8.54894544e-03 -4.11639163e-01]
 [ 1.77547698e+00 -2.03394449e-01 -1.98837863e-01  2.42669441e-01
   9.64350564e-01  2.01830179e-01 -5.45774168e-01  6.61020288e-01
   1.79215821e+00 -7.00398505e-01]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
   8.84908814e-01  2.76499305e-01]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7fb32d9d7f28>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7fb347d61710> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]
 [ 0.69174373  1.00978733 -1.21333813 -1.55694156 -1.20257258 -0.61244213
  -2.69836174 -0.13935181 -0.72853749  0.0722519 ]
 [ 1.02817479 -0.50845713  1.7653351   0.77741921  0.61771419 -0.11877117
   0.45015551 -0.19899818  1.86647138  0.8709698 ]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]
 [ 0.55853873 -0.51634791 -0.51814555  0.3511169   0.82550695 -0.06877046
  -0.9520621  -1.34776494  1.47073986 -1.4614036 ]
 [ 0.35413361  0.21112476  0.92145007  0.01652757  0.90394545  0.17718772
   0.09542509 -1.11647002  0.0809271   0.0607502 ]
 [ 1.838294    0.50274088  0.12910158  1.55880554  1.32551412  0.1094027
   1.40754    -1.2197444   2.44936865  1.6169496 ]
 [ 1.77547698 -0.20339445 -0.19883786  0.24266944  0.96435056  0.20183018
  -0.54577417  0.66102029  1.79215821 -0.7003985 ]
 [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
  -0.17961229  0.98495167  1.07125243 -0.59334375]
 [ 0.44118981  0.47985237 -0.1920037  -1.55269878 -1.88873982  0.57846442
   0.39859839 -0.9612636  -1.45832446 -3.05376438]
 [ 0.77528533  1.47016034  1.03298378 -0.87000822  0.78655651  0.36919047
  -0.14319575  0.85328219 -0.13971173 -0.22241403]
 [ 0.87122579 -0.20975294 -0.45698786  0.93514778 -0.87353582  1.81252782
   0.92550121  0.14010988 -1.41914878  1.06898597]
 [ 0.99785516 -0.6001388   0.45794708  0.14676526 -0.93355729  0.57180488
   0.57296273 -0.03681766  0.11236849 -0.01781755]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 0.44689516  0.38653915  1.35010682 -0.85145566  0.85063796  1.00088142
  -1.1601701  -0.38483225  1.45810824 -0.33128317]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
   1.10223673  0.81434313  0.78046993  1.10861676]
 [ 0.93621125  0.20437739 -1.49419377  0.61223252 -0.98437725  0.74488454
   0.49434165 -0.03628129 -0.83239535 -0.4466992 ]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 0.89551051  0.92061512  0.79452824 -0.03536792  0.8780991   2.11060505
  -1.02188594 -1.30653407  0.07638048 -1.87316098]]
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
[[ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 8.88389445e-01  2.82995534e-01  1.79558917e-02  1.08030817e-01
  -8.49671873e-01  2.94176190e-02 -5.03973949e-01 -1.34793129e-01
   1.04921829e+00 -1.27046078e+00]
 [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
   1.40925339e+00  9.66539250e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 7.90323893e-01  1.61336137e+00 -2.09424782e+00 -3.74804687e-01
   9.15884042e-01 -7.49969617e-01  3.10272288e-01  2.05462410e+00
   5.34095368e-02 -2.28765829e-01]
 [ 1.58463774e+00  5.71209961e-02 -1.77183179e-02 -7.99547491e-01
   1.32970299e+00 -2.91594596e-01 -1.10771250e+00 -2.58982853e-01
   1.89293198e-01 -1.71939447e+00]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]
 [ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
   1.10417433e+00 -4.22856961e-01]
 [ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
  -1.14870423e+00 -9.50909251e-01]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 6.21530991e-01 -1.50957268e+00 -1.01932039e-01 -1.08071069e+00
  -1.13742855e+00  7.25474004e-01  7.98063795e-01 -3.91782562e-02
  -2.28754171e-01  7.43356544e-01]
 [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
  -1.41914878e+00  1.06898597e+00]
 [ 3.54133613e-01  2.11124755e-01  9.21450069e-01  1.65275673e-02
   9.03945451e-01  1.77187720e-01  9.54250872e-02 -1.11647002e+00
   8.09271010e-02  6.07501958e-02]
 [ 8.95510508e-01  9.20615118e-01  7.94528240e-01 -3.53679249e-02
   8.78099103e-01  2.11060505e+00 -1.02188594e+00 -1.30653407e+00
   7.63804802e-02 -1.87316098e+00]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]
 [ 1.34740825e+00  7.33023232e-01  8.38634747e-01 -1.89881206e+00
  -5.42459922e-01 -1.11711069e+00 -1.09715436e+00 -5.08972278e-01
  -1.66485955e-01 -1.03918232e+00]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 1.77547698e+00 -2.03394449e-01 -1.98837863e-01  2.42669441e-01
   9.64350564e-01  2.01830179e-01 -5.45774168e-01  6.61020288e-01
   1.79215821e+00 -7.00398505e-01]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 6.92114488e-01 -6.06524918e-02  2.05635552e+00 -2.41350300e+00
   1.17456965e+00 -1.77756638e+00 -2.81736269e-01 -7.77858827e-01
   1.11584111e+00  1.76024923e+00]
 [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 1.34728643e+00 -3.64538050e-01  8.07509886e-02 -4.59717681e-01
  -8.89487596e-01  1.70548352e+00  9.49961101e-02  2.40505552e-01
  -9.99426501e-01 -7.67803746e-01]]
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
[master 7541238] ml_store
 1 file changed, 296 insertions(+)
To github.com:arita37/mlmodels_store.git
   e4587ea..7541238  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140633767636888
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140633767636048
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140633767635600
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140633767635096
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140633767634592
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140633767634256

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
grad_step = 000000, loss = 0.404534
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.294715
grad_step = 000002, loss = 0.223214
grad_step = 000003, loss = 0.148475
grad_step = 000004, loss = 0.082481
grad_step = 000005, loss = 0.044169
grad_step = 000006, loss = 0.041182
grad_step = 000007, loss = 0.051980
grad_step = 000008, loss = 0.044752
grad_step = 000009, loss = 0.028690
grad_step = 000010, loss = 0.017772
grad_step = 000011, loss = 0.015482
grad_step = 000012, loss = 0.017956
grad_step = 000013, loss = 0.020466
grad_step = 000014, loss = 0.020690
grad_step = 000015, loss = 0.019182
grad_step = 000016, loss = 0.016990
grad_step = 000017, loss = 0.014919
grad_step = 000018, loss = 0.013102
grad_step = 000019, loss = 0.011436
grad_step = 000020, loss = 0.010175
grad_step = 000021, loss = 0.009649
grad_step = 000022, loss = 0.009704
grad_step = 000023, loss = 0.009816
grad_step = 000024, loss = 0.009554
grad_step = 000025, loss = 0.008986
grad_step = 000026, loss = 0.008421
grad_step = 000027, loss = 0.008147
grad_step = 000028, loss = 0.008183
grad_step = 000029, loss = 0.008355
grad_step = 000030, loss = 0.008428
grad_step = 000031, loss = 0.008252
grad_step = 000032, loss = 0.007829
grad_step = 000033, loss = 0.007290
grad_step = 000034, loss = 0.006841
grad_step = 000035, loss = 0.006643
grad_step = 000036, loss = 0.006730
grad_step = 000037, loss = 0.006977
grad_step = 000038, loss = 0.007182
grad_step = 000039, loss = 0.007198
grad_step = 000040, loss = 0.006996
grad_step = 000041, loss = 0.006686
grad_step = 000042, loss = 0.006412
grad_step = 000043, loss = 0.006291
grad_step = 000044, loss = 0.006340
grad_step = 000045, loss = 0.006478
grad_step = 000046, loss = 0.006574
grad_step = 000047, loss = 0.006543
grad_step = 000048, loss = 0.006401
grad_step = 000049, loss = 0.006228
grad_step = 000050, loss = 0.006108
grad_step = 000051, loss = 0.006071
grad_step = 000052, loss = 0.006089
grad_step = 000053, loss = 0.006110
grad_step = 000054, loss = 0.006098
grad_step = 000055, loss = 0.006055
grad_step = 000056, loss = 0.005991
grad_step = 000057, loss = 0.005933
grad_step = 000058, loss = 0.005897
grad_step = 000059, loss = 0.005875
grad_step = 000060, loss = 0.005855
grad_step = 000061, loss = 0.005824
grad_step = 000062, loss = 0.005780
grad_step = 000063, loss = 0.005726
grad_step = 000064, loss = 0.005675
grad_step = 000065, loss = 0.005637
grad_step = 000066, loss = 0.005612
grad_step = 000067, loss = 0.005589
grad_step = 000068, loss = 0.005554
grad_step = 000069, loss = 0.005506
grad_step = 000070, loss = 0.005453
grad_step = 000071, loss = 0.005408
grad_step = 000072, loss = 0.005373
grad_step = 000073, loss = 0.005341
grad_step = 000074, loss = 0.005302
grad_step = 000075, loss = 0.005256
grad_step = 000076, loss = 0.005204
grad_step = 000077, loss = 0.005152
grad_step = 000078, loss = 0.005103
grad_step = 000079, loss = 0.005059
grad_step = 000080, loss = 0.005016
grad_step = 000081, loss = 0.004969
grad_step = 000082, loss = 0.004914
grad_step = 000083, loss = 0.004854
grad_step = 000084, loss = 0.004796
grad_step = 000085, loss = 0.004741
grad_step = 000086, loss = 0.004685
grad_step = 000087, loss = 0.004627
grad_step = 000088, loss = 0.004564
grad_step = 000089, loss = 0.004497
grad_step = 000090, loss = 0.004428
grad_step = 000091, loss = 0.004361
grad_step = 000092, loss = 0.004295
grad_step = 000093, loss = 0.004227
grad_step = 000094, loss = 0.004153
grad_step = 000095, loss = 0.004076
grad_step = 000096, loss = 0.003999
grad_step = 000097, loss = 0.003922
grad_step = 000098, loss = 0.003846
grad_step = 000099, loss = 0.003767
grad_step = 000100, loss = 0.003686
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003605
grad_step = 000102, loss = 0.003522
grad_step = 000103, loss = 0.003440
grad_step = 000104, loss = 0.003357
grad_step = 000105, loss = 0.003277
grad_step = 000106, loss = 0.003195
grad_step = 000107, loss = 0.003114
grad_step = 000108, loss = 0.003035
grad_step = 000109, loss = 0.002958
grad_step = 000110, loss = 0.002885
grad_step = 000111, loss = 0.002813
grad_step = 000112, loss = 0.002743
grad_step = 000113, loss = 0.002678
grad_step = 000114, loss = 0.002619
grad_step = 000115, loss = 0.002562
grad_step = 000116, loss = 0.002507
grad_step = 000117, loss = 0.002459
grad_step = 000118, loss = 0.002414
grad_step = 000119, loss = 0.002374
grad_step = 000120, loss = 0.002337
grad_step = 000121, loss = 0.002312
grad_step = 000122, loss = 0.002306
grad_step = 000123, loss = 0.002276
grad_step = 000124, loss = 0.002203
grad_step = 000125, loss = 0.002186
grad_step = 000126, loss = 0.002165
grad_step = 000127, loss = 0.002105
grad_step = 000128, loss = 0.002090
grad_step = 000129, loss = 0.002055
grad_step = 000130, loss = 0.002012
grad_step = 000131, loss = 0.001999
grad_step = 000132, loss = 0.001955
grad_step = 000133, loss = 0.001933
grad_step = 000134, loss = 0.001907
grad_step = 000135, loss = 0.001871
grad_step = 000136, loss = 0.001854
grad_step = 000137, loss = 0.001819
grad_step = 000138, loss = 0.001796
grad_step = 000139, loss = 0.001769
grad_step = 000140, loss = 0.001739
grad_step = 000141, loss = 0.001718
grad_step = 000142, loss = 0.001685
grad_step = 000143, loss = 0.001662
grad_step = 000144, loss = 0.001634
grad_step = 000145, loss = 0.001606
grad_step = 000146, loss = 0.001583
grad_step = 000147, loss = 0.001552
grad_step = 000148, loss = 0.001526
grad_step = 000149, loss = 0.001500
grad_step = 000150, loss = 0.001471
grad_step = 000151, loss = 0.001445
grad_step = 000152, loss = 0.001418
grad_step = 000153, loss = 0.001389
grad_step = 000154, loss = 0.001361
grad_step = 000155, loss = 0.001333
grad_step = 000156, loss = 0.001304
grad_step = 000157, loss = 0.001275
grad_step = 000158, loss = 0.001246
grad_step = 000159, loss = 0.001217
grad_step = 000160, loss = 0.001188
grad_step = 000161, loss = 0.001158
grad_step = 000162, loss = 0.001129
grad_step = 000163, loss = 0.001101
grad_step = 000164, loss = 0.001072
grad_step = 000165, loss = 0.001045
grad_step = 000166, loss = 0.001018
grad_step = 000167, loss = 0.000993
grad_step = 000168, loss = 0.000968
grad_step = 000169, loss = 0.000945
grad_step = 000170, loss = 0.000923
grad_step = 000171, loss = 0.000902
grad_step = 000172, loss = 0.000882
grad_step = 000173, loss = 0.000864
grad_step = 000174, loss = 0.000846
grad_step = 000175, loss = 0.000831
grad_step = 000176, loss = 0.000818
grad_step = 000177, loss = 0.000807
grad_step = 000178, loss = 0.000798
grad_step = 000179, loss = 0.000789
grad_step = 000180, loss = 0.000777
grad_step = 000181, loss = 0.000765
grad_step = 000182, loss = 0.000754
grad_step = 000183, loss = 0.000747
grad_step = 000184, loss = 0.000742
grad_step = 000185, loss = 0.000740
grad_step = 000186, loss = 0.000741
grad_step = 000187, loss = 0.000735
grad_step = 000188, loss = 0.000717
grad_step = 000189, loss = 0.000710
grad_step = 000190, loss = 0.000714
grad_step = 000191, loss = 0.000708
grad_step = 000192, loss = 0.000698
grad_step = 000193, loss = 0.000694
grad_step = 000194, loss = 0.000687
grad_step = 000195, loss = 0.000681
grad_step = 000196, loss = 0.000682
grad_step = 000197, loss = 0.000675
grad_step = 000198, loss = 0.000666
grad_step = 000199, loss = 0.000663
grad_step = 000200, loss = 0.000657
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000653
grad_step = 000202, loss = 0.000651
grad_step = 000203, loss = 0.000645
grad_step = 000204, loss = 0.000639
grad_step = 000205, loss = 0.000635
grad_step = 000206, loss = 0.000629
grad_step = 000207, loss = 0.000625
grad_step = 000208, loss = 0.000622
grad_step = 000209, loss = 0.000617
grad_step = 000210, loss = 0.000613
grad_step = 000211, loss = 0.000610
grad_step = 000212, loss = 0.000605
grad_step = 000213, loss = 0.000602
grad_step = 000214, loss = 0.000600
grad_step = 000215, loss = 0.000598
grad_step = 000216, loss = 0.000598
grad_step = 000217, loss = 0.000600
grad_step = 000218, loss = 0.000599
grad_step = 000219, loss = 0.000597
grad_step = 000220, loss = 0.000589
grad_step = 000221, loss = 0.000578
grad_step = 000222, loss = 0.000567
grad_step = 000223, loss = 0.000559
grad_step = 000224, loss = 0.000557
grad_step = 000225, loss = 0.000558
grad_step = 000226, loss = 0.000561
grad_step = 000227, loss = 0.000564
grad_step = 000228, loss = 0.000565
grad_step = 000229, loss = 0.000562
grad_step = 000230, loss = 0.000552
grad_step = 000231, loss = 0.000539
grad_step = 000232, loss = 0.000527
grad_step = 000233, loss = 0.000521
grad_step = 000234, loss = 0.000520
grad_step = 000235, loss = 0.000522
grad_step = 000236, loss = 0.000524
grad_step = 000237, loss = 0.000524
grad_step = 000238, loss = 0.000519
grad_step = 000239, loss = 0.000510
grad_step = 000240, loss = 0.000501
grad_step = 000241, loss = 0.000493
grad_step = 000242, loss = 0.000489
grad_step = 000243, loss = 0.000488
grad_step = 000244, loss = 0.000489
grad_step = 000245, loss = 0.000491
grad_step = 000246, loss = 0.000495
grad_step = 000247, loss = 0.000501
grad_step = 000248, loss = 0.000510
grad_step = 000249, loss = 0.000515
grad_step = 000250, loss = 0.000514
grad_step = 000251, loss = 0.000498
grad_step = 000252, loss = 0.000478
grad_step = 000253, loss = 0.000461
grad_step = 000254, loss = 0.000457
grad_step = 000255, loss = 0.000463
grad_step = 000256, loss = 0.000471
grad_step = 000257, loss = 0.000473
grad_step = 000258, loss = 0.000465
grad_step = 000259, loss = 0.000453
grad_step = 000260, loss = 0.000442
grad_step = 000261, loss = 0.000437
grad_step = 000262, loss = 0.000437
grad_step = 000263, loss = 0.000441
grad_step = 000264, loss = 0.000446
grad_step = 000265, loss = 0.000448
grad_step = 000266, loss = 0.000448
grad_step = 000267, loss = 0.000444
grad_step = 000268, loss = 0.000438
grad_step = 000269, loss = 0.000429
grad_step = 000270, loss = 0.000421
grad_step = 000271, loss = 0.000414
grad_step = 000272, loss = 0.000410
grad_step = 000273, loss = 0.000409
grad_step = 000274, loss = 0.000409
grad_step = 000275, loss = 0.000410
grad_step = 000276, loss = 0.000412
grad_step = 000277, loss = 0.000415
grad_step = 000278, loss = 0.000417
grad_step = 000279, loss = 0.000420
grad_step = 000280, loss = 0.000421
grad_step = 000281, loss = 0.000422
grad_step = 000282, loss = 0.000418
grad_step = 000283, loss = 0.000412
grad_step = 000284, loss = 0.000401
grad_step = 000285, loss = 0.000390
grad_step = 000286, loss = 0.000382
grad_step = 000287, loss = 0.000379
grad_step = 000288, loss = 0.000379
grad_step = 000289, loss = 0.000383
grad_step = 000290, loss = 0.000388
grad_step = 000291, loss = 0.000393
grad_step = 000292, loss = 0.000398
grad_step = 000293, loss = 0.000400
grad_step = 000294, loss = 0.000401
grad_step = 000295, loss = 0.000393
grad_step = 000296, loss = 0.000383
grad_step = 000297, loss = 0.000370
grad_step = 000298, loss = 0.000361
grad_step = 000299, loss = 0.000358
grad_step = 000300, loss = 0.000362
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000373
grad_step = 000302, loss = 0.000390
grad_step = 000303, loss = 0.000409
grad_step = 000304, loss = 0.000407
grad_step = 000305, loss = 0.000382
grad_step = 000306, loss = 0.000351
grad_step = 000307, loss = 0.000347
grad_step = 000308, loss = 0.000365
grad_step = 000309, loss = 0.000371
grad_step = 000310, loss = 0.000356
grad_step = 000311, loss = 0.000340
grad_step = 000312, loss = 0.000344
grad_step = 000313, loss = 0.000358
grad_step = 000314, loss = 0.000355
grad_step = 000315, loss = 0.000344
grad_step = 000316, loss = 0.000341
grad_step = 000317, loss = 0.000352
grad_step = 000318, loss = 0.000363
grad_step = 000319, loss = 0.000361
grad_step = 000320, loss = 0.000364
grad_step = 000321, loss = 0.000376
grad_step = 000322, loss = 0.000386
grad_step = 000323, loss = 0.000373
grad_step = 000324, loss = 0.000350
grad_step = 000325, loss = 0.000337
grad_step = 000326, loss = 0.000331
grad_step = 000327, loss = 0.000323
grad_step = 000328, loss = 0.000320
grad_step = 000329, loss = 0.000328
grad_step = 000330, loss = 0.000337
grad_step = 000331, loss = 0.000339
grad_step = 000332, loss = 0.000329
grad_step = 000333, loss = 0.000322
grad_step = 000334, loss = 0.000321
grad_step = 000335, loss = 0.000318
grad_step = 000336, loss = 0.000311
grad_step = 000337, loss = 0.000304
grad_step = 000338, loss = 0.000303
grad_step = 000339, loss = 0.000306
grad_step = 000340, loss = 0.000307
grad_step = 000341, loss = 0.000307
grad_step = 000342, loss = 0.000308
grad_step = 000343, loss = 0.000313
grad_step = 000344, loss = 0.000325
grad_step = 000345, loss = 0.000337
grad_step = 000346, loss = 0.000352
grad_step = 000347, loss = 0.000362
grad_step = 000348, loss = 0.000370
grad_step = 000349, loss = 0.000354
grad_step = 000350, loss = 0.000331
grad_step = 000351, loss = 0.000304
grad_step = 000352, loss = 0.000288
grad_step = 000353, loss = 0.000290
grad_step = 000354, loss = 0.000302
grad_step = 000355, loss = 0.000314
grad_step = 000356, loss = 0.000318
grad_step = 000357, loss = 0.000312
grad_step = 000358, loss = 0.000299
grad_step = 000359, loss = 0.000288
grad_step = 000360, loss = 0.000281
grad_step = 000361, loss = 0.000280
grad_step = 000362, loss = 0.000284
grad_step = 000363, loss = 0.000289
grad_step = 000364, loss = 0.000295
grad_step = 000365, loss = 0.000297
grad_step = 000366, loss = 0.000298
grad_step = 000367, loss = 0.000295
grad_step = 000368, loss = 0.000290
grad_step = 000369, loss = 0.000282
grad_step = 000370, loss = 0.000275
grad_step = 000371, loss = 0.000271
grad_step = 000372, loss = 0.000269
grad_step = 000373, loss = 0.000269
grad_step = 000374, loss = 0.000272
grad_step = 000375, loss = 0.000276
grad_step = 000376, loss = 0.000282
grad_step = 000377, loss = 0.000292
grad_step = 000378, loss = 0.000301
grad_step = 000379, loss = 0.000310
grad_step = 000380, loss = 0.000310
grad_step = 000381, loss = 0.000305
grad_step = 000382, loss = 0.000289
grad_step = 000383, loss = 0.000274
grad_step = 000384, loss = 0.000264
grad_step = 000385, loss = 0.000264
grad_step = 000386, loss = 0.000271
grad_step = 000387, loss = 0.000283
grad_step = 000388, loss = 0.000296
grad_step = 000389, loss = 0.000304
grad_step = 000390, loss = 0.000307
grad_step = 000391, loss = 0.000298
grad_step = 000392, loss = 0.000284
grad_step = 000393, loss = 0.000269
grad_step = 000394, loss = 0.000265
grad_step = 000395, loss = 0.000272
grad_step = 000396, loss = 0.000284
grad_step = 000397, loss = 0.000294
grad_step = 000398, loss = 0.000293
grad_step = 000399, loss = 0.000282
grad_step = 000400, loss = 0.000263
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000249
grad_step = 000402, loss = 0.000244
grad_step = 000403, loss = 0.000250
grad_step = 000404, loss = 0.000260
grad_step = 000405, loss = 0.000266
grad_step = 000406, loss = 0.000270
grad_step = 000407, loss = 0.000271
grad_step = 000408, loss = 0.000269
grad_step = 000409, loss = 0.000261
grad_step = 000410, loss = 0.000253
grad_step = 000411, loss = 0.000247
grad_step = 000412, loss = 0.000246
grad_step = 000413, loss = 0.000250
grad_step = 000414, loss = 0.000258
grad_step = 000415, loss = 0.000265
grad_step = 000416, loss = 0.000269
grad_step = 000417, loss = 0.000271
grad_step = 000418, loss = 0.000267
grad_step = 000419, loss = 0.000260
grad_step = 000420, loss = 0.000249
grad_step = 000421, loss = 0.000241
grad_step = 000422, loss = 0.000236
grad_step = 000423, loss = 0.000236
grad_step = 000424, loss = 0.000239
grad_step = 000425, loss = 0.000244
grad_step = 000426, loss = 0.000250
grad_step = 000427, loss = 0.000251
grad_step = 000428, loss = 0.000252
grad_step = 000429, loss = 0.000246
grad_step = 000430, loss = 0.000238
grad_step = 000431, loss = 0.000229
grad_step = 000432, loss = 0.000224
grad_step = 000433, loss = 0.000223
grad_step = 000434, loss = 0.000226
grad_step = 000435, loss = 0.000232
grad_step = 000436, loss = 0.000239
grad_step = 000437, loss = 0.000251
grad_step = 000438, loss = 0.000264
grad_step = 000439, loss = 0.000282
grad_step = 000440, loss = 0.000295
grad_step = 000441, loss = 0.000306
grad_step = 000442, loss = 0.000297
grad_step = 000443, loss = 0.000272
grad_step = 000444, loss = 0.000256
grad_step = 000445, loss = 0.000267
grad_step = 000446, loss = 0.000303
grad_step = 000447, loss = 0.000317
grad_step = 000448, loss = 0.000299
grad_step = 000449, loss = 0.000255
grad_step = 000450, loss = 0.000229
grad_step = 000451, loss = 0.000230
grad_step = 000452, loss = 0.000242
grad_step = 000453, loss = 0.000248
grad_step = 000454, loss = 0.000242
grad_step = 000455, loss = 0.000239
grad_step = 000456, loss = 0.000240
grad_step = 000457, loss = 0.000242
grad_step = 000458, loss = 0.000234
grad_step = 000459, loss = 0.000221
grad_step = 000460, loss = 0.000211
grad_step = 000461, loss = 0.000211
grad_step = 000462, loss = 0.000220
grad_step = 000463, loss = 0.000227
grad_step = 000464, loss = 0.000228
grad_step = 000465, loss = 0.000221
grad_step = 000466, loss = 0.000213
grad_step = 000467, loss = 0.000209
grad_step = 000468, loss = 0.000209
grad_step = 000469, loss = 0.000211
grad_step = 000470, loss = 0.000212
grad_step = 000471, loss = 0.000211
grad_step = 000472, loss = 0.000207
grad_step = 000473, loss = 0.000203
grad_step = 000474, loss = 0.000203
grad_step = 000475, loss = 0.000205
grad_step = 000476, loss = 0.000209
grad_step = 000477, loss = 0.000214
grad_step = 000478, loss = 0.000217
grad_step = 000479, loss = 0.000220
grad_step = 000480, loss = 0.000221
grad_step = 000481, loss = 0.000222
grad_step = 000482, loss = 0.000223
grad_step = 000483, loss = 0.000224
grad_step = 000484, loss = 0.000224
grad_step = 000485, loss = 0.000224
grad_step = 000486, loss = 0.000221
grad_step = 000487, loss = 0.000217
grad_step = 000488, loss = 0.000210
grad_step = 000489, loss = 0.000202
grad_step = 000490, loss = 0.000196
grad_step = 000491, loss = 0.000193
grad_step = 000492, loss = 0.000193
grad_step = 000493, loss = 0.000196
grad_step = 000494, loss = 0.000201
grad_step = 000495, loss = 0.000208
grad_step = 000496, loss = 0.000219
grad_step = 000497, loss = 0.000231
grad_step = 000498, loss = 0.000242
grad_step = 000499, loss = 0.000248
grad_step = 000500, loss = 0.000250
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
[[0.8575932  0.83671534 0.9244338  0.9503681  1.0205892 ]
 [0.83528864 0.8999127  0.9493058  1.0048704  0.99675   ]
 [0.8781531  0.89537346 0.9872731  0.96924627 0.93903774]
 [0.90215766 0.98727095 0.97068524 0.94673365 0.9124046 ]
 [0.9862151  0.9873425  0.94933736 0.9229957  0.8645598 ]
 [0.96170247 0.94443357 0.9030098  0.8512816  0.866452  ]
 [0.9162189  0.90188587 0.8474883  0.8446806  0.8269189 ]
 [0.8855971  0.8336258  0.8466648  0.8064397  0.84547067]
 [0.81357026 0.8318243  0.813305   0.83957475 0.8596699 ]
 [0.8297967  0.8047838  0.83372414 0.8450683  0.82687026]
 [0.7961546  0.8228463  0.8490058  0.8143859  0.92981493]
 [0.8179513  0.8267423  0.82853615 0.9228246  0.94179094]
 [0.85080874 0.83128536 0.9206625  0.94965976 1.018323  ]
 [0.8261385  0.90716803 0.94868433 0.9998719  0.9872788 ]
 [0.89308834 0.9119493  0.99026287 0.96691895 0.9240047 ]
 [0.9145936  0.994655   0.9633639  0.93367827 0.89497507]
 [0.9866998  0.9781327  0.93785655 0.89958364 0.84335953]
 [0.95851505 0.9283422  0.88744485 0.82915384 0.847087  ]
 [0.90748036 0.8934685  0.83136976 0.83591086 0.8215788 ]
 [0.8883821  0.83567846 0.8342699  0.8053425  0.84561026]
 [0.82404387 0.840274   0.81182146 0.83966315 0.8649872 ]
 [0.844882   0.815336   0.83619344 0.85141796 0.83156574]
 [0.8063134  0.8343626  0.8601645  0.8173224  0.9325194 ]
 [0.82480717 0.83460474 0.8346236  0.92034125 0.94391733]
 [0.8637639  0.8436198  0.9255091  0.95554364 1.0279107 ]
 [0.843245   0.9049326  0.95304644 1.011499   1.0095155 ]
 [0.8856007  0.90225995 0.9947264  0.9803542  0.9507545 ]
 [0.91325426 0.9978181  0.98247397 0.9584907  0.92170554]
 [0.99451005 0.99953544 0.96105754 0.93107224 0.87009954]
 [0.97270644 0.9538616  0.9138807  0.85718423 0.8696474 ]
 [0.9234979  0.911579   0.8543956  0.8505384  0.83475006]]

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
[master 4ed9e8b] ml_store
 1 file changed, 1122 insertions(+)
To github.com:arita37/mlmodels_store.git
   7541238..4ed9e8b  master -> master





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
[master 435067a] ml_store
 1 file changed, 37 insertions(+)
To github.com:arita37/mlmodels_store.git
   4ed9e8b..435067a  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s] 23%|██▎       | 52224/231508 [00:00<00:00, 391678.97B/s] 53%|█████▎    | 121856/231508 [00:00<00:00, 419775.23B/s]100%|██████████| 231508/231508 [00:00<00:00, 675355.22B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 169526.15B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  0%|          | 44032/440473133 [00:00<21:19, 344305.85B/s]  0%|          | 131072/440473133 [00:00<18:13, 402779.24B/s]  0%|          | 374784/440473133 [00:00<13:55, 526919.04B/s]  0%|          | 1332224/440473133 [00:00<10:01, 730312.20B/s]  1%|          | 3921920/440473133 [00:00<07:03, 1030844.12B/s]  2%|▏         | 8182784/440473133 [00:00<04:56, 1457279.01B/s]  3%|▎         | 12789760/440473133 [00:00<03:28, 2053976.58B/s]  4%|▍         | 17293312/440473133 [00:00<02:27, 2877997.57B/s]  5%|▌         | 22158336/440473133 [00:01<01:44, 4008662.88B/s]  6%|▌         | 26254336/440473133 [00:01<01:15, 5495846.27B/s]  7%|▋         | 30881792/440473133 [00:01<00:54, 7470886.97B/s]  8%|▊         | 35404800/440473133 [00:01<00:40, 9967082.42B/s]  9%|▉         | 40016896/440473133 [00:01<00:30, 12984448.24B/s] 10%|█         | 44557312/440473133 [00:01<00:23, 16523782.55B/s] 11%|█         | 48911360/440473133 [00:01<00:19, 20174788.35B/s] 12%|█▏        | 53434368/440473133 [00:01<00:15, 24195763.16B/s] 13%|█▎        | 57851904/440473133 [00:01<00:13, 27993670.21B/s] 14%|█▍        | 62429184/440473133 [00:01<00:11, 31684399.94B/s] 15%|█▌        | 66869248/440473133 [00:02<00:10, 34454448.01B/s] 16%|█▌        | 71353344/440473133 [00:02<00:09, 37025882.71B/s] 17%|█▋        | 75783168/440473133 [00:02<00:09, 38924217.04B/s] 18%|█▊        | 80210944/440473133 [00:02<00:08, 40375229.89B/s] 19%|█▉        | 84636672/440473133 [00:02<00:08, 41053432.06B/s] 20%|██        | 89016320/440473133 [00:02<00:08, 41125487.93B/s] 21%|██        | 93395968/440473133 [00:02<00:08, 41869064.47B/s] 22%|██▏       | 97950720/440473133 [00:02<00:07, 42904775.05B/s] 23%|██▎       | 102565888/440473133 [00:02<00:07, 43829109.73B/s] 24%|██▍       | 107110400/440473133 [00:02<00:07, 44301295.01B/s] 25%|██▌       | 111777792/440473133 [00:03<00:07, 44984769.45B/s] 26%|██▋       | 116415488/440473133 [00:03<00:07, 44980617.37B/s] 28%|██▊       | 121281536/440473133 [00:03<00:06, 46014862.20B/s] 29%|██▊       | 125910016/440473133 [00:03<00:06, 45701484.90B/s] 30%|██▉       | 130678784/440473133 [00:03<00:06, 46275492.01B/s] 31%|███       | 135322624/440473133 [00:03<00:06, 46316061.98B/s] 32%|███▏      | 139965440/440473133 [00:03<00:06, 46322657.31B/s] 33%|███▎      | 144730112/440473133 [00:03<00:06, 46711671.57B/s] 34%|███▍      | 149414912/440473133 [00:03<00:06, 46751332.08B/s] 35%|███▍      | 154094592/440473133 [00:03<00:06, 46724465.73B/s] 36%|███▌      | 158770176/440473133 [00:04<00:06, 46490591.75B/s] 37%|███▋      | 163422208/440473133 [00:04<00:05, 46478485.49B/s] 38%|███▊      | 168101888/440473133 [00:04<00:05, 46572741.82B/s] 39%|███▉      | 172761088/440473133 [00:04<00:05, 46394810.21B/s] 40%|████      | 177457152/440473133 [00:04<00:05, 46559965.49B/s] 41%|████▏     | 182114304/440473133 [00:04<00:05, 45908506.34B/s] 42%|████▏     | 187054080/440473133 [00:04<00:05, 46901206.32B/s] 44%|████▎     | 191752192/440473133 [00:04<00:05, 45447110.64B/s] 45%|████▍     | 196613120/440473133 [00:04<00:05, 46350356.24B/s] 46%|████▌     | 201263104/440473133 [00:04<00:05, 46029898.42B/s] 47%|████▋     | 205936640/440473133 [00:05<00:05, 46237896.56B/s] 48%|████▊     | 210745344/440473133 [00:05<00:04, 46775430.39B/s] 49%|████▉     | 215430144/440473133 [00:05<00:04, 46636650.67B/s] 50%|████▉     | 220170240/440473133 [00:05<00:04, 46541579.60B/s] 51%|█████     | 224975872/440473133 [00:05<00:04, 46979256.30B/s] 52%|█████▏    | 229682176/440473133 [00:05<00:04, 47001855.27B/s] 53%|█████▎    | 234494976/440473133 [00:05<00:04, 47131627.85B/s] 54%|█████▍    | 239212544/440473133 [00:05<00:04, 47143500.43B/s] 55%|█████▌    | 243929088/440473133 [00:05<00:04, 46680834.16B/s] 56%|█████▋    | 248599552/440473133 [00:05<00:04, 46222162.77B/s] 58%|█████▊    | 253428736/440473133 [00:06<00:03, 46823851.77B/s] 59%|█████▊    | 258115584/440473133 [00:06<00:03, 46027812.97B/s] 60%|█████▉    | 262895616/440473133 [00:06<00:03, 46544744.79B/s] 61%|██████    | 267555840/440473133 [00:06<00:03, 46559263.37B/s] 62%|██████▏   | 272216064/440473133 [00:06<00:03, 46404323.50B/s] 63%|██████▎   | 276859904/440473133 [00:06<00:03, 46345904.65B/s] 64%|██████▍   | 281496576/440473133 [00:06<00:03, 46253867.72B/s] 65%|██████▍   | 286124032/440473133 [00:06<00:03, 45566620.08B/s] 66%|██████▌   | 290954240/440473133 [00:06<00:03, 46281046.71B/s] 67%|██████▋   | 295731200/440473133 [00:06<00:03, 46716224.45B/s] 68%|██████▊   | 300407808/440473133 [00:07<00:03, 46509728.26B/s] 69%|██████▉   | 305112064/440473133 [00:07<00:02, 46666928.56B/s] 70%|███████   | 309781504/440473133 [00:07<00:02, 46612493.50B/s] 71%|███████▏  | 314568704/440473133 [00:07<00:02, 46982058.18B/s] 72%|███████▏  | 319268864/440473133 [00:07<00:02, 46890095.83B/s] 74%|███████▎  | 324009984/440473133 [00:07<00:02, 47044499.12B/s] 75%|███████▍  | 328716288/440473133 [00:07<00:02, 46318769.62B/s] 76%|███████▌  | 333351936/440473133 [00:07<00:02, 39365298.99B/s] 77%|███████▋  | 338068480/440473133 [00:07<00:02, 41419326.84B/s] 78%|███████▊  | 342367232/440473133 [00:08<00:02, 39384921.77B/s] 79%|███████▊  | 346436608/440473133 [00:08<00:02, 38537765.37B/s] 80%|███████▉  | 350493696/440473133 [00:08<00:02, 39120456.34B/s] 80%|████████  | 354475008/440473133 [00:08<00:02, 38326872.44B/s] 81%|████████▏ | 358360064/440473133 [00:08<00:02, 37277732.78B/s] 82%|████████▏ | 362736640/440473133 [00:08<00:01, 39011649.94B/s] 83%|████████▎ | 366689280/440473133 [00:08<00:01, 38095262.39B/s] 84%|████████▍ | 370613248/440473133 [00:08<00:01, 36993062.13B/s] 85%|████████▌ | 374756352/440473133 [00:08<00:01, 38220790.77B/s] 86%|████████▌ | 378613760/440473133 [00:09<00:01, 37612270.36B/s] 87%|████████▋ | 383215616/440473133 [00:09<00:01, 39792430.16B/s] 88%|████████▊ | 387305472/440473133 [00:09<00:01, 40116820.77B/s] 89%|████████▉ | 391912448/440473133 [00:09<00:01, 40478877.49B/s] 90%|█████████ | 396468224/440473133 [00:09<00:01, 41878379.37B/s] 91%|█████████ | 400688128/440473133 [00:09<00:00, 41440967.32B/s] 92%|█████████▏| 404855808/440473133 [00:09<00:00, 41261031.37B/s] 93%|█████████▎| 409225216/440473133 [00:09<00:00, 41961504.27B/s] 94%|█████████▍| 413435904/440473133 [00:09<00:00, 40358313.70B/s] 95%|█████████▍| 417995776/440473133 [00:09<00:00, 41285306.06B/s] 96%|█████████▌| 422194176/440473133 [00:10<00:00, 41492178.62B/s] 97%|█████████▋| 426580992/440473133 [00:10<00:00, 42174642.58B/s] 98%|█████████▊| 430843904/440473133 [00:10<00:00, 42307934.71B/s] 99%|█████████▉| 435183616/440473133 [00:10<00:00, 42627091.14B/s]100%|█████████▉| 439771136/440473133 [00:10<00:00, 43550802.58B/s]100%|██████████| 440473133/440473133 [00:10<00:00, 41938564.34B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
  16384/7094233 [..............................] - ETA: 31s
  57344/7094233 [..............................] - ETA: 19s
  81920/7094233 [..............................] - ETA: 21s
 106496/7094233 [..............................] - ETA: 23s
 131072/7094233 [..............................] - ETA: 22s
 147456/7094233 [..............................] - ETA: 23s
 172032/7094233 [..............................] - ETA: 24s
 196608/7094233 [..............................] - ETA: 24s
 212992/7094233 [..............................] - ETA: 24s
 237568/7094233 [>.............................] - ETA: 24s
 262144/7094233 [>.............................] - ETA: 24s
 278528/7094233 [>.............................] - ETA: 24s
 303104/7094233 [>.............................] - ETA: 24s
 319488/7094233 [>.............................] - ETA: 24s
 327680/7094233 [>.............................] - ETA: 25s
 360448/7094233 [>.............................] - ETA: 24s
 368640/7094233 [>.............................] - ETA: 25s
 393216/7094233 [>.............................] - ETA: 24s
 409600/7094233 [>.............................] - ETA: 25s
 434176/7094233 [>.............................] - ETA: 25s
 458752/7094233 [>.............................] - ETA: 25s
 491520/7094233 [=>............................] - ETA: 24s
 499712/7094233 [=>............................] - ETA: 25s
 524288/7094233 [=>............................] - ETA: 25s
 557056/7094233 [=>............................] - ETA: 46s
1024000/7094233 [===>..........................] - ETA: 24s
1048576/7094233 [===>..........................] - ETA: 24s
1064960/7094233 [===>..........................] - ETA: 24s
1089536/7094233 [===>..........................] - ETA: 23s
1114112/7094233 [===>..........................] - ETA: 23s
1130496/7094233 [===>..........................] - ETA: 24s
1155072/7094233 [===>..........................] - ETA: 23s
1179648/7094233 [===>..........................] - ETA: 24s
1212416/7094233 [====>.........................] - ETA: 23s
1236992/7094233 [====>.........................] - ETA: 23s
1253376/7094233 [====>.........................] - ETA: 23s
1277952/7094233 [====>.........................] - ETA: 23s
1302528/7094233 [====>.........................] - ETA: 23s
1318912/7094233 [====>.........................] - ETA: 23s
1343488/7094233 [====>.........................] - ETA: 23s
1359872/7094233 [====>.........................] - ETA: 23s
1384448/7094233 [====>.........................] - ETA: 23s
1400832/7094233 [====>.........................] - ETA: 23s
1409024/7094233 [====>.........................] - ETA: 23s
1433600/7094233 [=====>........................] - ETA: 23s
1449984/7094233 [=====>........................] - ETA: 23s
1474560/7094233 [=====>........................] - ETA: 23s
1490944/7094233 [=====>........................] - ETA: 23s
1540096/7094233 [=====>........................] - ETA: 23s
1556480/7094233 [=====>........................] - ETA: 23s
1581056/7094233 [=====>........................] - ETA: 23s
1605632/7094233 [=====>........................] - ETA: 23s
1630208/7094233 [=====>........................] - ETA: 22s
1646592/7094233 [=====>........................] - ETA: 22s
1671168/7094233 [======>.......................] - ETA: 22s
1695744/7094233 [======>.......................] - ETA: 22s
1712128/7094233 [======>.......................] - ETA: 22s
1736704/7094233 [======>.......................] - ETA: 22s
1761280/7094233 [======>.......................] - ETA: 22s
1777664/7094233 [======>.......................] - ETA: 22s
1802240/7094233 [======>.......................] - ETA: 22s
1826816/7094233 [======>.......................] - ETA: 22s
1843200/7094233 [======>.......................] - ETA: 22s
1867776/7094233 [======>.......................] - ETA: 21s
1892352/7094233 [=======>......................] - ETA: 21s
1908736/7094233 [=======>......................] - ETA: 21s
1933312/7094233 [=======>......................] - ETA: 21s
1957888/7094233 [=======>......................] - ETA: 21s
1974272/7094233 [=======>......................] - ETA: 21s
1998848/7094233 [=======>......................] - ETA: 21s
2023424/7094233 [=======>......................] - ETA: 21s
2039808/7094233 [=======>......................] - ETA: 21s
2064384/7094233 [=======>......................] - ETA: 21s
2088960/7094233 [=======>......................] - ETA: 21s
2121728/7094233 [=======>......................] - ETA: 20s
2154496/7094233 [========>.....................] - ETA: 20s
2170880/7094233 [========>.....................] - ETA: 20s
2195456/7094233 [========>.....................] - ETA: 20s
2220032/7094233 [========>.....................] - ETA: 20s
2252800/7094233 [========>.....................] - ETA: 20s
2260992/7094233 [========>.....................] - ETA: 20s
2285568/7094233 [========>.....................] - ETA: 20s
2301952/7094233 [========>.....................] - ETA: 20s
2326528/7094233 [========>.....................] - ETA: 20s
2351104/7094233 [========>.....................] - ETA: 20s
2383872/7094233 [=========>....................] - ETA: 19s
2392064/7094233 [=========>....................] - ETA: 19s
2416640/7094233 [=========>....................] - ETA: 19s
2433024/7094233 [=========>....................] - ETA: 19s
2457600/7094233 [=========>....................] - ETA: 19s
2482176/7094233 [=========>....................] - ETA: 19s
2514944/7094233 [=========>....................] - ETA: 19s
2539520/7094233 [=========>....................] - ETA: 18s
2564096/7094233 [=========>....................] - ETA: 18s
2596864/7094233 [=========>....................] - ETA: 18s
2605056/7094233 [==========>...................] - ETA: 18s
2637824/7094233 [==========>...................] - ETA: 18s
2646016/7094233 [==========>...................] - ETA: 18s
2670592/7094233 [==========>...................] - ETA: 18s
2703360/7094233 [==========>...................] - ETA: 17s
2711552/7094233 [==========>...................] - ETA: 18s
2736128/7094233 [==========>...................] - ETA: 17s
2752512/7094233 [==========>...................] - ETA: 17s
2768896/7094233 [==========>...................] - ETA: 17s
2777088/7094233 [==========>...................] - ETA: 17s
2801664/7094233 [==========>...................] - ETA: 17s
2818048/7094233 [==========>...................] - ETA: 17s
2834432/7094233 [==========>...................] - ETA: 17s
2842624/7094233 [===========>..................] - ETA: 17s
2867200/7094233 [===========>..................] - ETA: 17s
2883584/7094233 [===========>..................] - ETA: 17s
2899968/7094233 [===========>..................] - ETA: 17s
2924544/7094233 [===========>..................] - ETA: 17s
2932736/7094233 [===========>..................] - ETA: 17s
2949120/7094233 [===========>..................] - ETA: 17s
2973696/7094233 [===========>..................] - ETA: 17s
2990080/7094233 [===========>..................] - ETA: 16s
2998272/7094233 [===========>..................] - ETA: 16s
3031040/7094233 [===========>..................] - ETA: 16s
3039232/7094233 [===========>..................] - ETA: 16s
3063808/7094233 [===========>..................] - ETA: 16s
3080192/7094233 [============>.................] - ETA: 16s
3104768/7094233 [============>.................] - ETA: 16s
3129344/7094233 [============>.................] - ETA: 16s
3145728/7094233 [============>.................] - ETA: 16s
3162112/7094233 [============>.................] - ETA: 16s
3186688/7094233 [============>.................] - ETA: 16s
3211264/7094233 [============>.................] - ETA: 16s
3235840/7094233 [============>.................] - ETA: 16s
3260416/7094233 [============>.................] - ETA: 16s
3276800/7094233 [============>.................] - ETA: 15s
3301376/7094233 [============>.................] - ETA: 15s
3325952/7094233 [=============>................] - ETA: 15s
3366912/7094233 [=============>................] - ETA: 15s
3391488/7094233 [=============>................] - ETA: 15s
3407872/7094233 [=============>................] - ETA: 15s
3432448/7094233 [=============>................] - ETA: 15s
3457024/7094233 [=============>................] - ETA: 15s
3489792/7094233 [=============>................] - ETA: 15s
3497984/7094233 [=============>................] - ETA: 15s
3522560/7094233 [=============>................] - ETA: 14s
3555328/7094233 [==============>...............] - ETA: 14s
3563520/7094233 [==============>...............] - ETA: 14s
3588096/7094233 [==============>...............] - ETA: 14s
3604480/7094233 [==============>...............] - ETA: 14s
3629056/7094233 [==============>...............] - ETA: 14s
3653632/7094233 [==============>...............] - ETA: 14s
3670016/7094233 [==============>...............] - ETA: 14s
3694592/7094233 [==============>...............] - ETA: 14s
3719168/7094233 [==============>...............] - ETA: 14s
3735552/7094233 [==============>...............] - ETA: 13s
3760128/7094233 [==============>...............] - ETA: 13s
3784704/7094233 [===============>..............] - ETA: 13s
3801088/7094233 [===============>..............] - ETA: 13s
3825664/7094233 [===============>..............] - ETA: 13s
3850240/7094233 [===============>..............] - ETA: 13s
3866624/7094233 [===============>..............] - ETA: 13s
3891200/7094233 [===============>..............] - ETA: 13s
3915776/7094233 [===============>..............] - ETA: 13s
3932160/7094233 [===============>..............] - ETA: 13s
3956736/7094233 [===============>..............] - ETA: 13s
3973120/7094233 [===============>..............] - ETA: 13s
3989504/7094233 [===============>..............] - ETA: 12s
4014080/7094233 [===============>..............] - ETA: 12s
4038656/7094233 [================>.............] - ETA: 12s
4055040/7094233 [================>.............] - ETA: 12s
4079616/7094233 [================>.............] - ETA: 12s
4104192/7094233 [================>.............] - ETA: 12s
4120576/7094233 [================>.............] - ETA: 12s
4145152/7094233 [================>.............] - ETA: 12s
4169728/7094233 [================>.............] - ETA: 12s
4186112/7094233 [================>.............] - ETA: 12s
4210688/7094233 [================>.............] - ETA: 11s
4235264/7094233 [================>.............] - ETA: 11s
4251648/7094233 [================>.............] - ETA: 11s
4276224/7094233 [=================>............] - ETA: 11s
4300800/7094233 [=================>............] - ETA: 11s
4317184/7094233 [=================>............] - ETA: 11s
4341760/7094233 [=================>............] - ETA: 11s
4366336/7094233 [=================>............] - ETA: 11s
4382720/7094233 [=================>............] - ETA: 11s
4407296/7094233 [=================>............] - ETA: 11s
4431872/7094233 [=================>............] - ETA: 11s
4448256/7094233 [=================>............] - ETA: 10s
4472832/7094233 [=================>............] - ETA: 10s
4497408/7094233 [==================>...........] - ETA: 10s
4513792/7094233 [==================>...........] - ETA: 10s
4538368/7094233 [==================>...........] - ETA: 10s
4562944/7094233 [==================>...........] - ETA: 10s
4579328/7094233 [==================>...........] - ETA: 10s
4603904/7094233 [==================>...........] - ETA: 10s
4628480/7094233 [==================>...........] - ETA: 10s
4644864/7094233 [==================>...........] - ETA: 10s
4669440/7094233 [==================>...........] - ETA: 10s
4694016/7094233 [==================>...........] - ETA: 9s 
4710400/7094233 [==================>...........] - ETA: 9s
4734976/7094233 [===================>..........] - ETA: 9s
4759552/7094233 [===================>..........] - ETA: 9s
4775936/7094233 [===================>..........] - ETA: 9s
4800512/7094233 [===================>..........] - ETA: 9s
4825088/7094233 [===================>..........] - ETA: 9s
4841472/7094233 [===================>..........] - ETA: 9s
4866048/7094233 [===================>..........] - ETA: 9s
4890624/7094233 [===================>..........] - ETA: 9s
4923392/7094233 [===================>..........] - ETA: 8s
4931584/7094233 [===================>..........] - ETA: 8s
4956160/7094233 [===================>..........] - ETA: 8s
4972544/7094233 [====================>.........] - ETA: 8s
4997120/7094233 [====================>.........] - ETA: 8s
5021696/7094233 [====================>.........] - ETA: 8s
5038080/7094233 [====================>.........] - ETA: 8s
5062656/7094233 [====================>.........] - ETA: 8s
5087232/7094233 [====================>.........] - ETA: 8s
5103616/7094233 [====================>.........] - ETA: 8s
5144576/7094233 [====================>.........] - ETA: 8s
5169152/7094233 [====================>.........] - ETA: 7s
5193728/7094233 [====================>.........] - ETA: 7s
5218304/7094233 [=====================>........] - ETA: 7s
5234688/7094233 [=====================>........] - ETA: 7s
5259264/7094233 [=====================>........] - ETA: 7s
5283840/7094233 [=====================>........] - ETA: 7s
5300224/7094233 [=====================>........] - ETA: 7s
5316608/7094233 [=====================>........] - ETA: 7s
5324800/7094233 [=====================>........] - ETA: 7s
5349376/7094233 [=====================>........] - ETA: 7s
5382144/7094233 [=====================>........] - ETA: 7s
5406720/7094233 [=====================>........] - ETA: 6s
5439488/7094233 [======================>.......] - ETA: 6s
5447680/7094233 [======================>.......] - ETA: 6s
5472256/7094233 [======================>.......] - ETA: 6s
5488640/7094233 [======================>.......] - ETA: 6s
5505024/7094233 [======================>.......] - ETA: 6s
5529600/7094233 [======================>.......] - ETA: 6s
5537792/7094233 [======================>.......] - ETA: 6s
5554176/7094233 [======================>.......] - ETA: 6s
5578752/7094233 [======================>.......] - ETA: 6s
5603328/7094233 [======================>.......] - ETA: 6s
5619712/7094233 [======================>.......] - ETA: 6s
5644288/7094233 [======================>.......] - ETA: 5s
5668864/7094233 [======================>.......] - ETA: 5s
5701632/7094233 [=======================>......] - ETA: 5s
5709824/7094233 [=======================>......] - ETA: 5s
5734400/7094233 [=======================>......] - ETA: 5s
5750784/7094233 [=======================>......] - ETA: 5s
5775360/7094233 [=======================>......] - ETA: 5s
5799936/7094233 [=======================>......] - ETA: 5s
5832704/7094233 [=======================>......] - ETA: 5s
5857280/7094233 [=======================>......] - ETA: 5s
5881856/7094233 [=======================>......] - ETA: 4s
5906432/7094233 [=======================>......] - ETA: 4s
5931008/7094233 [========================>.....] - ETA: 4s
5947392/7094233 [========================>.....] - ETA: 4s
5971968/7094233 [========================>.....] - ETA: 4s
5996544/7094233 [========================>.....] - ETA: 4s
6012928/7094233 [========================>.....] - ETA: 4s
6037504/7094233 [========================>.....] - ETA: 4s
6053888/7094233 [========================>.....] - ETA: 4s
6078464/7094233 [========================>.....] - ETA: 4s
6094848/7094233 [========================>.....] - ETA: 4s
6103040/7094233 [========================>.....] - ETA: 4s
6127616/7094233 [========================>.....] - ETA: 3s
6144000/7094233 [========================>.....] - ETA: 3s
6168576/7094233 [=========================>....] - ETA: 3s
6193152/7094233 [=========================>....] - ETA: 3s
6225920/7094233 [=========================>....] - ETA: 3s
6234112/7094233 [=========================>....] - ETA: 3s
6258688/7094233 [=========================>....] - ETA: 3s
6275072/7094233 [=========================>....] - ETA: 3s
6299648/7094233 [=========================>....] - ETA: 3s
6316032/7094233 [=========================>....] - ETA: 3s
6324224/7094233 [=========================>....] - ETA: 3s
6356992/7094233 [=========================>....] - ETA: 3s
6365184/7094233 [=========================>....] - ETA: 3s
6389760/7094233 [==========================>...] - ETA: 2s
6422528/7094233 [==========================>...] - ETA: 2s
6430720/7094233 [==========================>...] - ETA: 2s
6455296/7094233 [==========================>...] - ETA: 2s
6471680/7094233 [==========================>...] - ETA: 2s
6488064/7094233 [==========================>...] - ETA: 2s
6496256/7094233 [==========================>...] - ETA: 2s
6520832/7094233 [==========================>...] - ETA: 2s
6537216/7094233 [==========================>...] - ETA: 2s
6561792/7094233 [==========================>...] - ETA: 2s
6586368/7094233 [==========================>...] - ETA: 2s
6619136/7094233 [==========================>...] - ETA: 1s
6627328/7094233 [===========================>..] - ETA: 1s
6643712/7094233 [===========================>..] - ETA: 1s
6651904/7094233 [===========================>..] - ETA: 1s
6684672/7094233 [===========================>..] - ETA: 1s
6717440/7094233 [===========================>..] - ETA: 1s
6733824/7094233 [===========================>..] - ETA: 1s
6750208/7094233 [===========================>..] - ETA: 1s
6774784/7094233 [===========================>..] - ETA: 1s
6791168/7094233 [===========================>..] - ETA: 1s
6807552/7094233 [===========================>..] - ETA: 1s
6815744/7094233 [===========================>..] - ETA: 1s
6832128/7094233 [===========================>..] - ETA: 1s
6856704/7094233 [===========================>..] - ETA: 0s
6881280/7094233 [============================>.] - ETA: 0s
6905856/7094233 [============================>.] - ETA: 0s
6922240/7094233 [============================>.] - ETA: 0s
6946816/7094233 [============================>.] - ETA: 0s
6971392/7094233 [============================>.] - ETA: 0s
6987776/7094233 [============================>.] - ETA: 0s
7012352/7094233 [============================>.] - ETA: 0s
7036928/7094233 [============================>.] - ETA: 0s
7053312/7094233 [============================>.] - ETA: 0s
7077888/7094233 [============================>.] - ETA: 0s
7094272/7094233 [==============================] - 30s 4us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   8%|▊         | 179/2118 [00:00<00:01, 1786.69it/s]Processing text_left with encode:  27%|██▋       | 563/2118 [00:00<00:00, 2127.47it/s]Processing text_left with encode:  46%|████▌     | 964/2118 [00:00<00:00, 2475.62it/s]Processing text_left with encode:  68%|██████▊   | 1434/2118 [00:00<00:00, 2884.68it/s]Processing text_left with encode:  87%|████████▋ | 1837/2118 [00:00<00:00, 3153.12it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3693.10it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 158/18841 [00:00<00:11, 1576.46it/s]Processing text_right with encode:   2%|▏         | 323/18841 [00:00<00:11, 1597.65it/s]Processing text_right with encode:   3%|▎         | 474/18841 [00:00<00:11, 1569.57it/s]Processing text_right with encode:   3%|▎         | 654/18841 [00:00<00:11, 1632.12it/s]Processing text_right with encode:   4%|▍         | 817/18841 [00:00<00:11, 1630.03it/s]Processing text_right with encode:   5%|▌         | 956/18841 [00:00<00:11, 1546.63it/s]Processing text_right with encode:   6%|▌         | 1094/18841 [00:00<00:11, 1479.03it/s]Processing text_right with encode:   7%|▋         | 1231/18841 [00:00<00:12, 1433.70it/s]Processing text_right with encode:   7%|▋         | 1388/18841 [00:00<00:11, 1471.35it/s]Processing text_right with encode:   8%|▊         | 1530/18841 [00:01<00:11, 1449.74it/s]Processing text_right with encode:   9%|▉         | 1682/18841 [00:01<00:11, 1469.24it/s]Processing text_right with encode:  10%|▉         | 1843/18841 [00:01<00:11, 1506.75it/s]Processing text_right with encode:  11%|█         | 2015/18841 [00:01<00:10, 1564.84it/s]Processing text_right with encode:  12%|█▏        | 2199/18841 [00:01<00:10, 1637.86it/s]Processing text_right with encode:  13%|█▎        | 2364/18841 [00:01<00:10, 1578.97it/s]Processing text_right with encode:  13%|█▎        | 2523/18841 [00:01<00:10, 1527.91it/s]Processing text_right with encode:  14%|█▍        | 2677/18841 [00:01<00:10, 1506.79it/s]Processing text_right with encode:  15%|█▌        | 2844/18841 [00:01<00:10, 1551.34it/s]Processing text_right with encode:  16%|█▌        | 3016/18841 [00:01<00:09, 1597.45it/s]Processing text_right with encode:  17%|█▋        | 3179/18841 [00:02<00:09, 1604.34it/s]Processing text_right with encode:  18%|█▊        | 3360/18841 [00:02<00:09, 1660.83it/s]Processing text_right with encode:  19%|█▊        | 3530/18841 [00:02<00:09, 1671.59it/s]Processing text_right with encode:  20%|█▉        | 3700/18841 [00:02<00:09, 1677.44it/s]Processing text_right with encode:  21%|██        | 3883/18841 [00:02<00:08, 1719.88it/s]Processing text_right with encode:  22%|██▏       | 4056/18841 [00:02<00:08, 1719.60it/s]Processing text_right with encode:  22%|██▏       | 4229/18841 [00:02<00:08, 1713.67it/s]Processing text_right with encode:  23%|██▎       | 4408/18841 [00:02<00:08, 1732.37it/s]Processing text_right with encode:  24%|██▍       | 4582/18841 [00:02<00:08, 1683.84it/s]Processing text_right with encode:  25%|██▌       | 4766/18841 [00:02<00:08, 1725.29it/s]Processing text_right with encode:  26%|██▌       | 4945/18841 [00:03<00:07, 1742.99it/s]Processing text_right with encode:  27%|██▋       | 5130/18841 [00:03<00:07, 1772.35it/s]Processing text_right with encode:  28%|██▊       | 5325/18841 [00:03<00:07, 1822.05it/s]Processing text_right with encode:  29%|██▉       | 5508/18841 [00:03<00:07, 1821.17it/s]Processing text_right with encode:  30%|███       | 5691/18841 [00:03<00:07, 1774.41it/s]Processing text_right with encode:  31%|███       | 5870/18841 [00:03<00:07, 1777.23it/s]Processing text_right with encode:  32%|███▏      | 6049/18841 [00:03<00:07, 1733.79it/s]Processing text_right with encode:  33%|███▎      | 6225/18841 [00:03<00:07, 1739.39it/s]Processing text_right with encode:  34%|███▍      | 6401/18841 [00:03<00:07, 1744.34it/s]Processing text_right with encode:  35%|███▍      | 6578/18841 [00:03<00:07, 1749.31it/s]Processing text_right with encode:  36%|███▌      | 6754/18841 [00:04<00:07, 1664.40it/s]Processing text_right with encode:  37%|███▋      | 6922/18841 [00:04<00:07, 1662.70it/s]Processing text_right with encode:  38%|███▊      | 7089/18841 [00:04<00:07, 1617.12it/s]Processing text_right with encode:  39%|███▊      | 7273/18841 [00:04<00:06, 1675.31it/s]Processing text_right with encode:  40%|███▉      | 7469/18841 [00:04<00:06, 1747.43it/s]Processing text_right with encode:  41%|████      | 7646/18841 [00:04<00:06, 1661.42it/s]Processing text_right with encode:  42%|████▏     | 7821/18841 [00:04<00:06, 1686.74it/s]Processing text_right with encode:  42%|████▏     | 7992/18841 [00:04<00:06, 1674.63it/s]Processing text_right with encode:  43%|████▎     | 8175/18841 [00:04<00:06, 1713.69it/s]Processing text_right with encode:  44%|████▍     | 8362/18841 [00:05<00:05, 1756.77it/s]Processing text_right with encode:  45%|████▌     | 8539/18841 [00:05<00:06, 1603.07it/s]Processing text_right with encode:  46%|████▌     | 8703/18841 [00:05<00:06, 1563.55it/s]Processing text_right with encode:  47%|████▋     | 8862/18841 [00:05<00:06, 1512.79it/s]Processing text_right with encode:  48%|████▊     | 9030/18841 [00:05<00:06, 1558.10it/s]Processing text_right with encode:  49%|████▉     | 9207/18841 [00:05<00:05, 1614.26it/s]Processing text_right with encode:  50%|████▉     | 9387/18841 [00:05<00:05, 1663.82it/s]Processing text_right with encode:  51%|█████     | 9577/18841 [00:05<00:05, 1725.81it/s]Processing text_right with encode:  52%|█████▏    | 9752/18841 [00:05<00:05, 1728.64it/s]Processing text_right with encode:  53%|█████▎    | 9941/18841 [00:05<00:05, 1771.97it/s]Processing text_right with encode:  54%|█████▎    | 10127/18841 [00:06<00:04, 1795.61it/s]Processing text_right with encode:  55%|█████▍    | 10308/18841 [00:06<00:05, 1693.96it/s]Processing text_right with encode:  56%|█████▌    | 10490/18841 [00:06<00:04, 1727.99it/s]Processing text_right with encode:  57%|█████▋    | 10665/18841 [00:06<00:05, 1595.69it/s]Processing text_right with encode:  57%|█████▋    | 10828/18841 [00:06<00:05, 1600.55it/s]Processing text_right with encode:  58%|█████▊    | 11000/18841 [00:06<00:04, 1632.55it/s]Processing text_right with encode:  59%|█████▉    | 11180/18841 [00:06<00:04, 1679.32it/s]Processing text_right with encode:  60%|██████    | 11360/18841 [00:06<00:04, 1712.54it/s]Processing text_right with encode:  61%|██████▏   | 11542/18841 [00:06<00:04, 1743.26it/s]Processing text_right with encode:  62%|██████▏   | 11720/18841 [00:07<00:04, 1752.47it/s]Processing text_right with encode:  63%|██████▎   | 11898/18841 [00:07<00:03, 1759.33it/s]Processing text_right with encode:  64%|██████▍   | 12081/18841 [00:07<00:03, 1779.81it/s]Processing text_right with encode:  65%|██████▌   | 12260/18841 [00:07<00:03, 1773.25it/s]Processing text_right with encode:  66%|██████▌   | 12439/18841 [00:07<00:03, 1777.42it/s]Processing text_right with encode:  67%|██████▋   | 12617/18841 [00:07<00:03, 1732.76it/s]Processing text_right with encode:  68%|██████▊   | 12791/18841 [00:07<00:03, 1650.07it/s]Processing text_right with encode:  69%|██████▉   | 12958/18841 [00:07<00:03, 1570.55it/s]Processing text_right with encode:  70%|██████▉   | 13117/18841 [00:07<00:03, 1526.64it/s]Processing text_right with encode:  71%|███████   | 13289/18841 [00:08<00:03, 1579.76it/s]Processing text_right with encode:  71%|███████▏  | 13459/18841 [00:08<00:03, 1611.02it/s]Processing text_right with encode:  72%|███████▏  | 13626/18841 [00:08<00:03, 1623.89it/s]Processing text_right with encode:  73%|███████▎  | 13790/18841 [00:08<00:03, 1599.55it/s]Processing text_right with encode:  74%|███████▍  | 13967/18841 [00:08<00:02, 1645.14it/s]Processing text_right with encode:  75%|███████▌  | 14135/18841 [00:08<00:02, 1655.09it/s]Processing text_right with encode:  76%|███████▌  | 14308/18841 [00:08<00:02, 1676.83it/s]Processing text_right with encode:  77%|███████▋  | 14481/18841 [00:08<00:02, 1690.19it/s]Processing text_right with encode:  78%|███████▊  | 14676/18841 [00:08<00:02, 1757.14it/s]Processing text_right with encode:  79%|███████▉  | 14853/18841 [00:08<00:02, 1662.62it/s]Processing text_right with encode:  80%|███████▉  | 15021/18841 [00:09<00:02, 1582.60it/s]Processing text_right with encode:  81%|████████  | 15182/18841 [00:09<00:02, 1545.44it/s]Processing text_right with encode:  81%|████████▏ | 15339/18841 [00:09<00:02, 1509.64it/s]Processing text_right with encode:  82%|████████▏ | 15492/18841 [00:09<00:02, 1488.04it/s]Processing text_right with encode:  83%|████████▎ | 15645/18841 [00:09<00:02, 1499.80it/s]Processing text_right with encode:  84%|████████▍ | 15827/18841 [00:09<00:01, 1580.98it/s]Processing text_right with encode:  85%|████████▍ | 15999/18841 [00:09<00:01, 1617.10it/s]Processing text_right with encode:  86%|████████▌ | 16163/18841 [00:09<00:01, 1577.97it/s]Processing text_right with encode:  87%|████████▋ | 16322/18841 [00:09<00:01, 1529.42it/s]Processing text_right with encode:  87%|████████▋ | 16477/18841 [00:10<00:01, 1517.45it/s]Processing text_right with encode:  88%|████████▊ | 16630/18841 [00:10<00:01, 1441.45it/s]Processing text_right with encode:  89%|████████▉ | 16814/18841 [00:10<00:01, 1538.69it/s]Processing text_right with encode:  90%|█████████ | 16975/18841 [00:10<00:01, 1557.77it/s]Processing text_right with encode:  91%|█████████ | 17150/18841 [00:10<00:01, 1610.23it/s]Processing text_right with encode:  92%|█████████▏| 17332/18841 [00:10<00:00, 1667.48it/s]Processing text_right with encode:  93%|█████████▎| 17501/18841 [00:10<00:00, 1664.02it/s]Processing text_right with encode:  94%|█████████▍| 17682/18841 [00:10<00:00, 1704.36it/s]Processing text_right with encode:  95%|█████████▍| 17854/18841 [00:10<00:00, 1642.84it/s]Processing text_right with encode:  96%|█████████▌| 18020/18841 [00:10<00:00, 1634.75it/s]Processing text_right with encode:  97%|█████████▋| 18212/18841 [00:11<00:00, 1710.84it/s]Processing text_right with encode:  98%|█████████▊| 18388/18841 [00:11<00:00, 1724.60it/s]Processing text_right with encode:  99%|█████████▊| 18578/18841 [00:11<00:00, 1772.66it/s]Processing text_right with encode: 100%|█████████▉| 18757/18841 [00:11<00:00, 1773.77it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:11<00:00, 1653.94it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 697513.81it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 801128.14it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  61%|██████▏   | 389/633 [00:00<00:00, 3878.66it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4024.77it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 183/5961 [00:00<00:03, 1826.70it/s]Processing text_right with encode:   6%|▌         | 358/5961 [00:00<00:03, 1802.84it/s]Processing text_right with encode:   8%|▊         | 497/5961 [00:00<00:03, 1654.87it/s]Processing text_right with encode:  11%|█         | 649/5961 [00:00<00:03, 1611.49it/s]Processing text_right with encode:  13%|█▎        | 796/5961 [00:00<00:03, 1566.19it/s]Processing text_right with encode:  16%|█▌        | 961/5961 [00:00<00:03, 1587.83it/s]Processing text_right with encode:  19%|█▉        | 1154/5961 [00:00<00:02, 1674.67it/s]Processing text_right with encode:  22%|██▏       | 1339/5961 [00:00<00:02, 1722.08it/s]Processing text_right with encode:  26%|██▌       | 1525/5961 [00:00<00:02, 1759.88it/s]Processing text_right with encode:  28%|██▊       | 1696/5961 [00:01<00:02, 1701.16it/s]Processing text_right with encode:  31%|███▏      | 1871/5961 [00:01<00:02, 1715.47it/s]Processing text_right with encode:  35%|███▍      | 2057/5961 [00:01<00:02, 1753.98it/s]Processing text_right with encode:  38%|███▊      | 2257/5961 [00:01<00:02, 1813.90it/s]Processing text_right with encode:  41%|████      | 2438/5961 [00:01<00:01, 1775.15it/s]Processing text_right with encode:  44%|████▍     | 2626/5961 [00:01<00:01, 1801.86it/s]Processing text_right with encode:  47%|████▋     | 2820/5961 [00:01<00:01, 1837.46it/s]Processing text_right with encode:  50%|█████     | 3004/5961 [00:01<00:01, 1810.45it/s]Processing text_right with encode:  53%|█████▎    | 3189/5961 [00:01<00:01, 1822.08it/s]Processing text_right with encode:  57%|█████▋    | 3372/5961 [00:01<00:01, 1792.32it/s]Processing text_right with encode:  60%|█████▉    | 3553/5961 [00:02<00:01, 1796.90it/s]Processing text_right with encode:  63%|██████▎   | 3733/5961 [00:02<00:01, 1794.31it/s]Processing text_right with encode:  66%|██████▌   | 3934/5961 [00:02<00:01, 1852.29it/s]Processing text_right with encode:  69%|██████▉   | 4129/5961 [00:02<00:00, 1879.85it/s]Processing text_right with encode:  72%|███████▏  | 4318/5961 [00:02<00:00, 1856.47it/s]Processing text_right with encode:  76%|███████▌  | 4505/5961 [00:02<00:00, 1814.16it/s]Processing text_right with encode:  79%|███████▊  | 4687/5961 [00:02<00:00, 1779.17it/s]Processing text_right with encode:  82%|████████▏ | 4866/5961 [00:02<00:00, 1679.34it/s]Processing text_right with encode:  85%|████████▍ | 5038/5961 [00:02<00:00, 1690.50it/s]Processing text_right with encode:  88%|████████▊ | 5221/5961 [00:02<00:00, 1729.86it/s]Processing text_right with encode:  91%|█████████ | 5395/5961 [00:03<00:00, 1709.83it/s]Processing text_right with encode:  93%|█████████▎| 5567/5961 [00:03<00:00, 1709.14it/s]Processing text_right with encode:  96%|█████████▋| 5739/5961 [00:03<00:00, 1683.20it/s]Processing text_right with encode: 100%|█████████▉| 5936/5961 [00:03<00:00, 1759.76it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1756.50it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 450533.59it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 857092.53it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:22<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:22<?, ?it/s, loss=0.832]Epoch 1/1:   1%|          | 1/102 [00:22<38:28, 22.86s/it, loss=0.832]Epoch 1/1:   1%|          | 1/102 [01:12<38:28, 22.86s/it, loss=0.832]Epoch 1/1:   1%|          | 1/102 [01:12<38:28, 22.86s/it, loss=0.889]Epoch 1/1:   2%|▏         | 2/102 [01:12<51:31, 30.91s/it, loss=0.889]Epoch 1/1:   2%|▏         | 2/102 [01:41<51:31, 30.91s/it, loss=0.889]Epoch 1/1:   2%|▏         | 2/102 [01:41<51:31, 30.91s/it, loss=0.900]Epoch 1/1:   3%|▎         | 3/102 [01:41<49:52, 30.23s/it, loss=0.900]Epoch 1/1:   3%|▎         | 3/102 [02:19<49:52, 30.23s/it, loss=0.900]Epoch 1/1:   3%|▎         | 3/102 [02:19<49:52, 30.23s/it, loss=0.972]Epoch 1/1:   4%|▍         | 4/102 [02:19<53:34, 32.80s/it, loss=0.972]Epoch 1/1:   4%|▍         | 4/102 [02:52<53:34, 32.80s/it, loss=0.972]Epoch 1/1:   4%|▍         | 4/102 [02:52<53:34, 32.80s/it, loss=0.914]Epoch 1/1:   5%|▍         | 5/102 [02:52<52:44, 32.63s/it, loss=0.914]Epoch 1/1:   5%|▍         | 5/102 [03:50<52:44, 32.63s/it, loss=0.914]Epoch 1/1:   5%|▍         | 5/102 [03:50<52:44, 32.63s/it, loss=0.772]Epoch 1/1:   6%|▌         | 6/102 [03:50<1:04:19, 40.20s/it, loss=0.772]Killed

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   435067a..9ba1257  master     -> origin/master
Updating 435067a..9ba1257
Fast-forward
 error_list/20200515/list_log_benchmark_20200515.md | 164 ++++++------
 .../20200515/list_log_dataloader_20200515.md       |   2 +-
 error_list/20200515/list_log_json_20200515.md      | 276 ++++++++++-----------
 .../20200515/list_log_pullrequest_20200515.md      |   2 +-
 4 files changed, 222 insertions(+), 222 deletions(-)
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
[master 123f14d] ml_store
 1 file changed, 373 insertions(+)
To github.com:arita37/mlmodels_store.git
   9ba1257..123f14d  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'dataset': 'torchvision.datasets:MNIST', 'transform_uri': 'mlmodels.preprocess.image.py:torch_transform_mnist', '2nd___transform_uri': '/mnt/hgfs/d/gitdev/mlmodels/mlmodels/preprocess/image.py:torch_transform_mnist', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 144588.26it/s]  1%|          | 98304/9912422 [00:00<00:52, 188143.08it/s]  4%|▍         | 434176/9912422 [00:00<00:36, 260135.90it/s] 18%|█▊        | 1753088/9912422 [00:00<00:22, 367445.40it/s] 44%|████▍     | 4407296/9912422 [00:00<00:10, 520689.82it/s] 74%|███████▍  | 7315456/9912422 [00:01<00:03, 736248.13it/s]9920512it [00:01, 8794125.69it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 136375.63it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160605.31it/s]  6%|▌         | 98304/1648877 [00:00<00:07, 206151.92it/s] 26%|██▋       | 434176/1648877 [00:00<00:04, 284269.54it/s]1654784it [00:00, 2576938.60it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 50173.09it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
[master 764224b] ml_store
 1 file changed, 84 insertions(+)
To github.com:arita37/mlmodels_store.git
   123f14d..764224b  master -> master





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
[master 6e85b9e] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   764224b..6e85b9e  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
Iteration:   0%|          | 0/29440 [00:00<?, ?it/s][A
Iteration:   0%|          | 1/29440 [00:15<125:29:46, 15.35s/it][A
Iteration:   0%|          | 2/29440 [00:25<113:11:48, 13.84s/it][A
Iteration:   0%|          | 3/29440 [00:40<115:36:07, 14.14s/it][A
Iteration:   0%|          | 4/29440 [00:53<111:49:59, 13.68s/it][A
Iteration:   0%|          | 5/29440 [02:06<257:11:48, 31.46s/it][A
Iteration:   0%|          | 6/29440 [02:37<256:08:23, 31.33s/it][A
Iteration:   0%|          | 7/29440 [04:49<503:02:32, 61.53s/it][A
Iteration:   0%|          | 8/29440 [05:54<511:44:41, 62.59s/it][A
Iteration:   0%|          | 9/29440 [08:12<697:50:06, 85.36s/it][A
Iteration:   0%|          | 10/29440 [12:36<1134:48:58, 138.82s/it][A
Iteration:   0%|          | 11/29440 [13:55<989:48:38, 121.08s/it] [A
Iteration:   0%|          | 12/29440 [14:24<762:50:13, 93.32s/it] [A
Iteration:   0%|          | 13/29440 [17:32<996:14:19, 121.88s/it][A
Iteration:   0%|          | 14/29440 [18:53<894:26:18, 109.43s/it][AKilled

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   6e85b9e..3ccf6fa  master     -> origin/master
Updating 6e85b9e..3ccf6fa
Fast-forward
 error_list/20200515/list_log_benchmark_20200515.md | 160 +++---
 error_list/20200515/list_log_json_20200515.md      | 276 ++++-----
 .../20200515/list_log_pullrequest_20200515.md      |   2 +-
 error_list/20200515/list_log_testall_20200515.md   | 290 +++++-----
 ...-10_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py | 620 +++++++++++++++++++++
 5 files changed, 972 insertions(+), 376 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-15-09-10_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py
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
[master fdfd65f] ml_store
 1 file changed, 65 insertions(+)
To github.com:arita37/mlmodels_store.git
   3ccf6fa..fdfd65f  master -> master





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
[master 2827216] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   fdfd65f..2827216  master -> master





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
   2827216..664cfaa  master     -> origin/master
Updating 2827216..664cfaa
Fast-forward
 error_list/20200515/list_log_benchmark_20200515.md | 162 ++++++------
 .../20200515/list_log_dataloader_20200515.md       |   2 +-
 error_list/20200515/list_log_testall_20200515.md   | 290 +++++++++++----------
 3 files changed, 239 insertions(+), 215 deletions(-)
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
