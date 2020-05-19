
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '73f54da32a5da4768415eb9105ad096255137679', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/73f54da32a5da4768415eb9105ad096255137679

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679

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
From github.com:arita37/mlmodels_store
   9c7c43d..2c69a3e  master     -> origin/master
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..2c69a3e
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..2c69a3e
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..2c69a3e
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_mean (E (None, 1, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-19 00:16:08.549991: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-19 00:16:08.555024: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-19 00:16:08.555745: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5606197f4150 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-19 00:16:08.555760: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 153
Trainable params: 153
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.4700 - binary_crossentropy: 7.2497500/500 [==============================] - 1s 1ms/sample - loss: 0.5020 - binary_crossentropy: 7.7433 - val_loss: 0.4800 - val_binary_crossentropy: 7.4040

  #### metrics   #################################################### 
{'MSE': 0.491}

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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_mean (E (None, 1, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
Total params: 153
Trainable params: 153
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_max[0][0]               
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
Total params: 438
Trainable params: 438
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2792 - binary_crossentropy: 0.7599500/500 [==============================] - 1s 1ms/sample - loss: 0.2716 - binary_crossentropy: 0.7689 - val_loss: 0.2716 - val_binary_crossentropy: 0.7422

  #### metrics   #################################################### 
{'MSE': 0.2709151558164216}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_max[0][0]               
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
Total params: 438
Trainable params: 438
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
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
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
Total params: 687
Trainable params: 687
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2501 - binary_crossentropy: 0.6934 - val_loss: 0.2500 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.2498696945810919}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
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
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
Total params: 687
Trainable params: 687
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
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
weighted_sequence_layer_9 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
Total params: 458
Trainable params: 458
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2838 - binary_crossentropy: 0.7651500/500 [==============================] - 1s 2ms/sample - loss: 0.2676 - binary_crossentropy: 0.7304 - val_loss: 0.2543 - val_binary_crossentropy: 0.7019

  #### metrics   #################################################### 
{'MSE': 0.25568608352000827}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
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
weighted_sequence_layer_9 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
Total params: 458
Trainable params: 458
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 158
Trainable params: 158
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3226 - binary_crossentropy: 1.6199500/500 [==============================] - 2s 3ms/sample - loss: 0.2850 - binary_crossentropy: 1.3628 - val_loss: 0.2772 - val_binary_crossentropy: 1.3479

  #### metrics   #################################################### 
{'MSE': 0.2784670542902204}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 158
Trainable params: 158
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-19 00:17:23.774697: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:17:23.776784: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:17:23.782838: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-19 00:17:23.793638: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-19 00:17:23.795375: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-19 00:17:23.797159: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:17:23.798809: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2523 - val_binary_crossentropy: 0.6977
2020-05-19 00:17:25.041545: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:17:25.043198: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:17:25.047895: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-19 00:17:25.058402: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-19 00:17:25.060822: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-19 00:17:25.062570: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:17:25.063857: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2528671434929389}

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
2020-05-19 00:17:47.188051: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:17:47.189389: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:17:47.192996: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-19 00:17:47.199269: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-19 00:17:47.200418: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-19 00:17:47.201362: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:17:47.202418: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2503 - val_binary_crossentropy: 0.6937
2020-05-19 00:17:48.573482: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:17:48.574485: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:17:48.577233: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-19 00:17:48.582874: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-19 00:17:48.583958: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-19 00:17:48.585052: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:17:48.586090: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2503076822290368}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-19 00:18:20.720644: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:18:20.725873: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:18:20.740603: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-19 00:18:20.765310: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-19 00:18:20.770224: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-19 00:18:20.774311: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:18:20.778247: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.3362 - binary_crossentropy: 0.8671 - val_loss: 0.2613 - val_binary_crossentropy: 0.7162
2020-05-19 00:18:22.917828: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:18:22.922549: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:18:22.934281: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-19 00:18:22.960795: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-19 00:18:22.965488: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-19 00:18:22.969580: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 00:18:22.973197: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2261619936106586}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
Total params: 675
Trainable params: 675
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3214 - binary_crossentropy: 0.8646500/500 [==============================] - 4s 9ms/sample - loss: 0.3039 - binary_crossentropy: 0.8481 - val_loss: 0.3086 - val_binary_crossentropy: 0.8604

  #### metrics   #################################################### 
{'MSE': 0.3053109940189321}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
Total params: 675
Trainable params: 675
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 2)         10          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 2)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         18          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         12          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         14          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_5[0][0]           
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
Total params: 284
Trainable params: 284
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3449 - binary_crossentropy: 1.4639500/500 [==============================] - 5s 9ms/sample - loss: 0.3560 - binary_crossentropy: 1.9137 - val_loss: 0.3359 - val_binary_crossentropy: 1.5392

  #### metrics   #################################################### 
{'MSE': 0.34705407464742016}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 2)         10          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 2)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         18          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         12          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         14          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_5[0][0]           
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
Total params: 284
Trainable params: 284
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_max[0][0]               
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
Total params: 1,919
Trainable params: 1,919
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2623 - binary_crossentropy: 0.8452500/500 [==============================] - 5s 10ms/sample - loss: 0.2612 - binary_crossentropy: 0.9506 - val_loss: 0.2629 - val_binary_crossentropy: 0.8766

  #### metrics   #################################################### 
{'MSE': 0.2616987336026372}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_max[0][0]               
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
Total params: 1,919
Trainable params: 1,919
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
regionsequence_sum (InputLayer) [(None, 6)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 8)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 6)]          0                                            
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
region_10sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         9           regionsequence_max[0][0]         
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
Total params: 200
Trainable params: 200
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2430 - binary_crossentropy: 0.6790500/500 [==============================] - 6s 12ms/sample - loss: 0.2495 - binary_crossentropy: 0.6920 - val_loss: 0.2564 - val_binary_crossentropy: 0.7058

  #### metrics   #################################################### 
{'MSE': 0.2527286910391308}

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
regionsequence_sum (InputLayer) [(None, 6)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 8)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 6)]          0                                            
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
region_10sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         9           regionsequence_max[0][0]         
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
Total params: 200
Trainable params: 200
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 1,402
Trainable params: 1,402
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2867 - binary_crossentropy: 0.7771500/500 [==============================] - 6s 12ms/sample - loss: 0.2732 - binary_crossentropy: 0.7451 - val_loss: 0.2590 - val_binary_crossentropy: 0.7123

  #### metrics   #################################################### 
{'MSE': 0.26366519005475014}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 1,402
Trainable params: 1,402
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         24          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         24          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         24          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         24          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           hash_10[0][0]                    
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
Total params: 3,273
Trainable params: 3,193
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2392 - binary_crossentropy: 0.6720500/500 [==============================] - 7s 14ms/sample - loss: 0.2644 - binary_crossentropy: 0.7247 - val_loss: 0.2655 - val_binary_crossentropy: 0.7259

  #### metrics   #################################################### 
{'MSE': 0.2641639279603802}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         24          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         24          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         24          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         24          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           hash_10[0][0]                    
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
Total params: 3,273
Trainable params: 3,193
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
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   2c69a3e..c8ba147  master     -> origin/master
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..c8ba147
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..c8ba147
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..c8ba147
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..c8ba147
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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

2020-05-19 00:27:31.737280: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-19 00:27:31.742166: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-19 00:27:31.742516: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555a202c8e80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-19 00:27:31.742536: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3878
256/354 [====================>.........] - ETA: 3s - loss: 1.3121
354/354 [==============================] - 15s 43ms/step - loss: 1.2863 - val_loss: 1.8369

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
From github.com:arita37/mlmodels_store
   c8ba147..03b14c2  master     -> origin/master
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1220608/17464789 [=>............................] - ETA: 0s
 3571712/17464789 [=====>........................] - ETA: 0s
 7143424/17464789 [===========>..................] - ETA: 0s
11452416/17464789 [==================>...........] - ETA: 0s
16416768/17464789 [===========================>..] - ETA: 0s
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
2020-05-19 00:28:28.165310: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-19 00:28:28.170229: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-19 00:28:28.170396: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c12f7b7c50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-19 00:28:28.170410: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5133 - accuracy: 0.5100
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8966 - accuracy: 0.4850 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7331 - accuracy: 0.4957
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7011 - accuracy: 0.4978
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7096 - accuracy: 0.4972
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6232 - accuracy: 0.5028
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6403 - accuracy: 0.5017
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6417 - accuracy: 0.5016
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6632 - accuracy: 0.5002
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7019 - accuracy: 0.4977
11000/25000 [============>.................] - ETA: 4s - loss: 7.7140 - accuracy: 0.4969
12000/25000 [=============>................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7138 - accuracy: 0.4969
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6809 - accuracy: 0.4991
15000/25000 [=================>............] - ETA: 3s - loss: 7.6809 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6762 - accuracy: 0.4994
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6892 - accuracy: 0.4985
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6777 - accuracy: 0.4993
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6674 - accuracy: 0.4999
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6973 - accuracy: 0.4980
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6951 - accuracy: 0.4981
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6896 - accuracy: 0.4985
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6753 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6698 - accuracy: 0.4998
25000/25000 [==============================] - 9s 379us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7ffaf96226a0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7ffb06cfe2b0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8353 - accuracy: 0.4890 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7944 - accuracy: 0.4917
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8736 - accuracy: 0.4865
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8353 - accuracy: 0.4890
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7382 - accuracy: 0.4953
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7170 - accuracy: 0.4967
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7107 - accuracy: 0.4971
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7211 - accuracy: 0.4964
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6942 - accuracy: 0.4982
11000/25000 [============>.................] - ETA: 4s - loss: 7.7238 - accuracy: 0.4963
12000/25000 [=============>................] - ETA: 4s - loss: 7.6947 - accuracy: 0.4982
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7055 - accuracy: 0.4975
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7181 - accuracy: 0.4966
15000/25000 [=================>............] - ETA: 3s - loss: 7.7116 - accuracy: 0.4971
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7088 - accuracy: 0.4972
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6982 - accuracy: 0.4979
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6862 - accuracy: 0.4987
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6774 - accuracy: 0.4993
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6688 - accuracy: 0.4999
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6896 - accuracy: 0.4985
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6780 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
25000/25000 [==============================] - 9s 373us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8506 - accuracy: 0.4880
 2000/25000 [=>............................] - ETA: 9s - loss: 8.0116 - accuracy: 0.4775 
 3000/25000 [==>...........................] - ETA: 8s - loss: 8.0551 - accuracy: 0.4747
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8238 - accuracy: 0.4897
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8782 - accuracy: 0.4862
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8302 - accuracy: 0.4893
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7893 - accuracy: 0.4920
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7605 - accuracy: 0.4939
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7774 - accuracy: 0.4928
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7632 - accuracy: 0.4937
11000/25000 [============>.................] - ETA: 4s - loss: 7.7544 - accuracy: 0.4943
12000/25000 [=============>................] - ETA: 4s - loss: 7.7484 - accuracy: 0.4947
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7551 - accuracy: 0.4942
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7345 - accuracy: 0.4956
15000/25000 [=================>............] - ETA: 3s - loss: 7.7290 - accuracy: 0.4959
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7433 - accuracy: 0.4950
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7262 - accuracy: 0.4961
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7007 - accuracy: 0.4978
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6997 - accuracy: 0.4978
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6728 - accuracy: 0.4996
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6652 - accuracy: 0.5001
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6694 - accuracy: 0.4998
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6626 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6711 - accuracy: 0.4997
25000/25000 [==============================] - 10s 382us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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

13/13 [==============================] - 2s 116ms/step - loss: nan
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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
 2097152/11490434 [====>.........................] - ETA: 0s
 8806400/11490434 [=====================>........] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:10 - loss: 2.3334 - categorical_accuracy: 0.0625
   64/60000 [..............................] - ETA: 4:34 - loss: 2.3151 - categorical_accuracy: 0.0938
   96/60000 [..............................] - ETA: 3:38 - loss: 2.3005 - categorical_accuracy: 0.0833
  128/60000 [..............................] - ETA: 3:09 - loss: 2.2962 - categorical_accuracy: 0.0938
  160/60000 [..............................] - ETA: 2:51 - loss: 2.2771 - categorical_accuracy: 0.1063
  192/60000 [..............................] - ETA: 2:38 - loss: 2.2466 - categorical_accuracy: 0.1458
  224/60000 [..............................] - ETA: 2:30 - loss: 2.2309 - categorical_accuracy: 0.1607
  256/60000 [..............................] - ETA: 2:23 - loss: 2.2053 - categorical_accuracy: 0.1875
  288/60000 [..............................] - ETA: 2:18 - loss: 2.1469 - categorical_accuracy: 0.2257
  320/60000 [..............................] - ETA: 2:14 - loss: 2.1129 - categorical_accuracy: 0.2469
  352/60000 [..............................] - ETA: 2:11 - loss: 2.0982 - categorical_accuracy: 0.2614
  384/60000 [..............................] - ETA: 2:09 - loss: 2.0503 - categorical_accuracy: 0.2891
  416/60000 [..............................] - ETA: 2:07 - loss: 2.0176 - categorical_accuracy: 0.3053
  448/60000 [..............................] - ETA: 2:05 - loss: 1.9972 - categorical_accuracy: 0.3170
  480/60000 [..............................] - ETA: 2:04 - loss: 1.9504 - categorical_accuracy: 0.3354
  512/60000 [..............................] - ETA: 2:02 - loss: 1.9456 - categorical_accuracy: 0.3398
  544/60000 [..............................] - ETA: 2:01 - loss: 1.9203 - categorical_accuracy: 0.3529
  576/60000 [..............................] - ETA: 2:00 - loss: 1.8860 - categorical_accuracy: 0.3628
  608/60000 [..............................] - ETA: 1:59 - loss: 1.8639 - categorical_accuracy: 0.3717
  640/60000 [..............................] - ETA: 1:58 - loss: 1.8444 - categorical_accuracy: 0.3781
  672/60000 [..............................] - ETA: 1:57 - loss: 1.8291 - categorical_accuracy: 0.3795
  704/60000 [..............................] - ETA: 1:57 - loss: 1.8210 - categorical_accuracy: 0.3807
  736/60000 [..............................] - ETA: 1:56 - loss: 1.7860 - categorical_accuracy: 0.3967
  768/60000 [..............................] - ETA: 1:55 - loss: 1.7642 - categorical_accuracy: 0.4023
  800/60000 [..............................] - ETA: 1:54 - loss: 1.7354 - categorical_accuracy: 0.4100
  832/60000 [..............................] - ETA: 1:54 - loss: 1.7241 - categorical_accuracy: 0.4111
  864/60000 [..............................] - ETA: 1:54 - loss: 1.7002 - categorical_accuracy: 0.4167
  896/60000 [..............................] - ETA: 1:54 - loss: 1.6807 - categorical_accuracy: 0.4230
  928/60000 [..............................] - ETA: 1:55 - loss: 1.6649 - categorical_accuracy: 0.4300
  960/60000 [..............................] - ETA: 1:54 - loss: 1.6437 - categorical_accuracy: 0.4396
  992/60000 [..............................] - ETA: 1:54 - loss: 1.6167 - categorical_accuracy: 0.4506
 1024/60000 [..............................] - ETA: 1:53 - loss: 1.5950 - categorical_accuracy: 0.4590
 1056/60000 [..............................] - ETA: 1:53 - loss: 1.5759 - categorical_accuracy: 0.4669
 1088/60000 [..............................] - ETA: 1:52 - loss: 1.5547 - categorical_accuracy: 0.4743
 1120/60000 [..............................] - ETA: 1:52 - loss: 1.5329 - categorical_accuracy: 0.4830
 1152/60000 [..............................] - ETA: 1:51 - loss: 1.5103 - categorical_accuracy: 0.4931
 1184/60000 [..............................] - ETA: 1:51 - loss: 1.4852 - categorical_accuracy: 0.5025
 1216/60000 [..............................] - ETA: 1:51 - loss: 1.4713 - categorical_accuracy: 0.5090
 1248/60000 [..............................] - ETA: 1:51 - loss: 1.4861 - categorical_accuracy: 0.5048
 1280/60000 [..............................] - ETA: 1:50 - loss: 1.4744 - categorical_accuracy: 0.5094
 1312/60000 [..............................] - ETA: 1:50 - loss: 1.4559 - categorical_accuracy: 0.5160
 1344/60000 [..............................] - ETA: 1:50 - loss: 1.4500 - categorical_accuracy: 0.5186
 1376/60000 [..............................] - ETA: 1:49 - loss: 1.4362 - categorical_accuracy: 0.5254
 1408/60000 [..............................] - ETA: 1:49 - loss: 1.4238 - categorical_accuracy: 0.5291
 1440/60000 [..............................] - ETA: 1:49 - loss: 1.4108 - categorical_accuracy: 0.5340
 1472/60000 [..............................] - ETA: 1:49 - loss: 1.3914 - categorical_accuracy: 0.5401
 1504/60000 [..............................] - ETA: 1:49 - loss: 1.3774 - categorical_accuracy: 0.5445
 1536/60000 [..............................] - ETA: 1:49 - loss: 1.3590 - categorical_accuracy: 0.5514
 1568/60000 [..............................] - ETA: 1:49 - loss: 1.3423 - categorical_accuracy: 0.5555
 1600/60000 [..............................] - ETA: 1:48 - loss: 1.3302 - categorical_accuracy: 0.5581
 1632/60000 [..............................] - ETA: 1:48 - loss: 1.3160 - categorical_accuracy: 0.5637
 1664/60000 [..............................] - ETA: 1:48 - loss: 1.2988 - categorical_accuracy: 0.5697
 1696/60000 [..............................] - ETA: 1:48 - loss: 1.2818 - categorical_accuracy: 0.5755
 1728/60000 [..............................] - ETA: 1:47 - loss: 1.2653 - categorical_accuracy: 0.5810
 1760/60000 [..............................] - ETA: 1:47 - loss: 1.2511 - categorical_accuracy: 0.5852
 1792/60000 [..............................] - ETA: 1:47 - loss: 1.2373 - categorical_accuracy: 0.5893
 1824/60000 [..............................] - ETA: 1:47 - loss: 1.2260 - categorical_accuracy: 0.5938
 1856/60000 [..............................] - ETA: 1:47 - loss: 1.2128 - categorical_accuracy: 0.5986
 1888/60000 [..............................] - ETA: 1:47 - loss: 1.1997 - categorical_accuracy: 0.6038
 1920/60000 [..............................] - ETA: 1:47 - loss: 1.1923 - categorical_accuracy: 0.6073
 1952/60000 [..............................] - ETA: 1:47 - loss: 1.1935 - categorical_accuracy: 0.6091
 1984/60000 [..............................] - ETA: 1:46 - loss: 1.1850 - categorical_accuracy: 0.6119
 2016/60000 [>.............................] - ETA: 1:46 - loss: 1.1749 - categorical_accuracy: 0.6166
 2048/60000 [>.............................] - ETA: 1:46 - loss: 1.1674 - categorical_accuracy: 0.6191
 2080/60000 [>.............................] - ETA: 1:46 - loss: 1.1581 - categorical_accuracy: 0.6226
 2112/60000 [>.............................] - ETA: 1:46 - loss: 1.1468 - categorical_accuracy: 0.6264
 2144/60000 [>.............................] - ETA: 1:47 - loss: 1.1358 - categorical_accuracy: 0.6306
 2176/60000 [>.............................] - ETA: 1:47 - loss: 1.1312 - categorical_accuracy: 0.6328
 2208/60000 [>.............................] - ETA: 1:46 - loss: 1.1206 - categorical_accuracy: 0.6359
 2240/60000 [>.............................] - ETA: 1:46 - loss: 1.1142 - categorical_accuracy: 0.6384
 2272/60000 [>.............................] - ETA: 1:46 - loss: 1.1067 - categorical_accuracy: 0.6400
 2304/60000 [>.............................] - ETA: 1:46 - loss: 1.1036 - categorical_accuracy: 0.6406
 2336/60000 [>.............................] - ETA: 1:46 - loss: 1.0989 - categorical_accuracy: 0.6430
 2368/60000 [>.............................] - ETA: 1:46 - loss: 1.0898 - categorical_accuracy: 0.6461
 2400/60000 [>.............................] - ETA: 1:46 - loss: 1.0827 - categorical_accuracy: 0.6475
 2432/60000 [>.............................] - ETA: 1:46 - loss: 1.0749 - categorical_accuracy: 0.6497
 2464/60000 [>.............................] - ETA: 1:46 - loss: 1.0657 - categorical_accuracy: 0.6530
 2496/60000 [>.............................] - ETA: 1:46 - loss: 1.0594 - categorical_accuracy: 0.6554
 2528/60000 [>.............................] - ETA: 1:46 - loss: 1.0529 - categorical_accuracy: 0.6570
 2560/60000 [>.............................] - ETA: 1:46 - loss: 1.0458 - categorical_accuracy: 0.6590
 2592/60000 [>.............................] - ETA: 1:45 - loss: 1.0359 - categorical_accuracy: 0.6620
 2624/60000 [>.............................] - ETA: 1:46 - loss: 1.0314 - categorical_accuracy: 0.6643
 2656/60000 [>.............................] - ETA: 1:46 - loss: 1.0235 - categorical_accuracy: 0.6668
 2688/60000 [>.............................] - ETA: 1:46 - loss: 1.0137 - categorical_accuracy: 0.6704
 2720/60000 [>.............................] - ETA: 1:46 - loss: 1.0086 - categorical_accuracy: 0.6713
 2752/60000 [>.............................] - ETA: 1:46 - loss: 1.0036 - categorical_accuracy: 0.6744
 2784/60000 [>.............................] - ETA: 1:46 - loss: 0.9966 - categorical_accuracy: 0.6764
 2816/60000 [>.............................] - ETA: 1:45 - loss: 0.9889 - categorical_accuracy: 0.6790
 2848/60000 [>.............................] - ETA: 1:45 - loss: 0.9812 - categorical_accuracy: 0.6815
 2880/60000 [>.............................] - ETA: 1:45 - loss: 0.9777 - categorical_accuracy: 0.6819
 2912/60000 [>.............................] - ETA: 1:45 - loss: 0.9704 - categorical_accuracy: 0.6837
 2944/60000 [>.............................] - ETA: 1:45 - loss: 0.9658 - categorical_accuracy: 0.6855
 2976/60000 [>.............................] - ETA: 1:45 - loss: 0.9596 - categorical_accuracy: 0.6875
 3008/60000 [>.............................] - ETA: 1:45 - loss: 0.9526 - categorical_accuracy: 0.6895
 3040/60000 [>.............................] - ETA: 1:45 - loss: 0.9472 - categorical_accuracy: 0.6908
 3072/60000 [>.............................] - ETA: 1:44 - loss: 0.9434 - categorical_accuracy: 0.6917
 3104/60000 [>.............................] - ETA: 1:44 - loss: 0.9390 - categorical_accuracy: 0.6933
 3136/60000 [>.............................] - ETA: 1:44 - loss: 0.9326 - categorical_accuracy: 0.6948
 3168/60000 [>.............................] - ETA: 1:44 - loss: 0.9262 - categorical_accuracy: 0.6970
 3200/60000 [>.............................] - ETA: 1:44 - loss: 0.9215 - categorical_accuracy: 0.6988
 3232/60000 [>.............................] - ETA: 1:44 - loss: 0.9213 - categorical_accuracy: 0.6983
 3264/60000 [>.............................] - ETA: 1:44 - loss: 0.9162 - categorical_accuracy: 0.6994
 3296/60000 [>.............................] - ETA: 1:44 - loss: 0.9134 - categorical_accuracy: 0.7005
 3328/60000 [>.............................] - ETA: 1:44 - loss: 0.9084 - categorical_accuracy: 0.7019
 3360/60000 [>.............................] - ETA: 1:43 - loss: 0.9011 - categorical_accuracy: 0.7048
 3392/60000 [>.............................] - ETA: 1:43 - loss: 0.8956 - categorical_accuracy: 0.7070
 3424/60000 [>.............................] - ETA: 1:43 - loss: 0.8889 - categorical_accuracy: 0.7088
 3456/60000 [>.............................] - ETA: 1:43 - loss: 0.8843 - categorical_accuracy: 0.7104
 3488/60000 [>.............................] - ETA: 1:43 - loss: 0.8788 - categorical_accuracy: 0.7124
 3520/60000 [>.............................] - ETA: 1:43 - loss: 0.8738 - categorical_accuracy: 0.7142
 3552/60000 [>.............................] - ETA: 1:43 - loss: 0.8718 - categorical_accuracy: 0.7151
 3584/60000 [>.............................] - ETA: 1:43 - loss: 0.8663 - categorical_accuracy: 0.7171
 3616/60000 [>.............................] - ETA: 1:43 - loss: 0.8609 - categorical_accuracy: 0.7193
 3648/60000 [>.............................] - ETA: 1:43 - loss: 0.8551 - categorical_accuracy: 0.7212
 3680/60000 [>.............................] - ETA: 1:43 - loss: 0.8489 - categorical_accuracy: 0.7234
 3712/60000 [>.............................] - ETA: 1:43 - loss: 0.8443 - categorical_accuracy: 0.7249
 3744/60000 [>.............................] - ETA: 1:43 - loss: 0.8388 - categorical_accuracy: 0.7268
 3776/60000 [>.............................] - ETA: 1:43 - loss: 0.8333 - categorical_accuracy: 0.7288
 3808/60000 [>.............................] - ETA: 1:42 - loss: 0.8302 - categorical_accuracy: 0.7298
 3840/60000 [>.............................] - ETA: 1:42 - loss: 0.8279 - categorical_accuracy: 0.7305
 3872/60000 [>.............................] - ETA: 1:42 - loss: 0.8232 - categorical_accuracy: 0.7317
 3904/60000 [>.............................] - ETA: 1:42 - loss: 0.8205 - categorical_accuracy: 0.7326
 3936/60000 [>.............................] - ETA: 1:42 - loss: 0.8154 - categorical_accuracy: 0.7345
 3968/60000 [>.............................] - ETA: 1:42 - loss: 0.8133 - categorical_accuracy: 0.7351
 4000/60000 [=>............................] - ETA: 1:42 - loss: 0.8135 - categorical_accuracy: 0.7358
 4032/60000 [=>............................] - ETA: 1:42 - loss: 0.8081 - categorical_accuracy: 0.7378
 4064/60000 [=>............................] - ETA: 1:42 - loss: 0.8050 - categorical_accuracy: 0.7384
 4096/60000 [=>............................] - ETA: 1:41 - loss: 0.8014 - categorical_accuracy: 0.7390
 4128/60000 [=>............................] - ETA: 1:41 - loss: 0.7981 - categorical_accuracy: 0.7403
 4160/60000 [=>............................] - ETA: 1:41 - loss: 0.7945 - categorical_accuracy: 0.7416
 4192/60000 [=>............................] - ETA: 1:41 - loss: 0.7904 - categorical_accuracy: 0.7433
 4224/60000 [=>............................] - ETA: 1:41 - loss: 0.7868 - categorical_accuracy: 0.7446
 4256/60000 [=>............................] - ETA: 1:41 - loss: 0.7840 - categorical_accuracy: 0.7455
 4288/60000 [=>............................] - ETA: 1:41 - loss: 0.7858 - categorical_accuracy: 0.7456
 4320/60000 [=>............................] - ETA: 1:41 - loss: 0.7814 - categorical_accuracy: 0.7472
 4352/60000 [=>............................] - ETA: 1:41 - loss: 0.7773 - categorical_accuracy: 0.7484
 4384/60000 [=>............................] - ETA: 1:41 - loss: 0.7740 - categorical_accuracy: 0.7493
 4416/60000 [=>............................] - ETA: 1:41 - loss: 0.7751 - categorical_accuracy: 0.7495
 4448/60000 [=>............................] - ETA: 1:41 - loss: 0.7732 - categorical_accuracy: 0.7502
 4480/60000 [=>............................] - ETA: 1:40 - loss: 0.7687 - categorical_accuracy: 0.7518
 4512/60000 [=>............................] - ETA: 1:40 - loss: 0.7651 - categorical_accuracy: 0.7531
 4544/60000 [=>............................] - ETA: 1:40 - loss: 0.7618 - categorical_accuracy: 0.7544
 4576/60000 [=>............................] - ETA: 1:40 - loss: 0.7613 - categorical_accuracy: 0.7550
 4608/60000 [=>............................] - ETA: 1:40 - loss: 0.7574 - categorical_accuracy: 0.7561
 4640/60000 [=>............................] - ETA: 1:40 - loss: 0.7555 - categorical_accuracy: 0.7569
 4672/60000 [=>............................] - ETA: 1:40 - loss: 0.7525 - categorical_accuracy: 0.7577
 4704/60000 [=>............................] - ETA: 1:40 - loss: 0.7488 - categorical_accuracy: 0.7589
 4736/60000 [=>............................] - ETA: 1:40 - loss: 0.7471 - categorical_accuracy: 0.7591
 4768/60000 [=>............................] - ETA: 1:40 - loss: 0.7433 - categorical_accuracy: 0.7601
 4800/60000 [=>............................] - ETA: 1:40 - loss: 0.7400 - categorical_accuracy: 0.7610
 4832/60000 [=>............................] - ETA: 1:40 - loss: 0.7382 - categorical_accuracy: 0.7614
 4864/60000 [=>............................] - ETA: 1:40 - loss: 0.7349 - categorical_accuracy: 0.7623
 4896/60000 [=>............................] - ETA: 1:40 - loss: 0.7358 - categorical_accuracy: 0.7627
 4928/60000 [=>............................] - ETA: 1:39 - loss: 0.7336 - categorical_accuracy: 0.7634
 4960/60000 [=>............................] - ETA: 1:39 - loss: 0.7311 - categorical_accuracy: 0.7639
 4992/60000 [=>............................] - ETA: 1:39 - loss: 0.7299 - categorical_accuracy: 0.7644
 5024/60000 [=>............................] - ETA: 1:39 - loss: 0.7281 - categorical_accuracy: 0.7651
 5056/60000 [=>............................] - ETA: 1:39 - loss: 0.7258 - categorical_accuracy: 0.7656
 5088/60000 [=>............................] - ETA: 1:39 - loss: 0.7219 - categorical_accuracy: 0.7669
 5120/60000 [=>............................] - ETA: 1:39 - loss: 0.7188 - categorical_accuracy: 0.7680
 5152/60000 [=>............................] - ETA: 1:39 - loss: 0.7153 - categorical_accuracy: 0.7690
 5184/60000 [=>............................] - ETA: 1:39 - loss: 0.7123 - categorical_accuracy: 0.7699
 5216/60000 [=>............................] - ETA: 1:39 - loss: 0.7093 - categorical_accuracy: 0.7709
 5248/60000 [=>............................] - ETA: 1:39 - loss: 0.7055 - categorical_accuracy: 0.7723
 5280/60000 [=>............................] - ETA: 1:39 - loss: 0.7026 - categorical_accuracy: 0.7735
 5312/60000 [=>............................] - ETA: 1:39 - loss: 0.7012 - categorical_accuracy: 0.7739
 5344/60000 [=>............................] - ETA: 1:38 - loss: 0.6990 - categorical_accuracy: 0.7745
 5376/60000 [=>............................] - ETA: 1:38 - loss: 0.6976 - categorical_accuracy: 0.7747
 5408/60000 [=>............................] - ETA: 1:38 - loss: 0.6971 - categorical_accuracy: 0.7748
 5440/60000 [=>............................] - ETA: 1:38 - loss: 0.6942 - categorical_accuracy: 0.7757
 5472/60000 [=>............................] - ETA: 1:38 - loss: 0.6917 - categorical_accuracy: 0.7765
 5504/60000 [=>............................] - ETA: 1:38 - loss: 0.6897 - categorical_accuracy: 0.7773
 5536/60000 [=>............................] - ETA: 1:38 - loss: 0.6886 - categorical_accuracy: 0.7778
 5568/60000 [=>............................] - ETA: 1:38 - loss: 0.6861 - categorical_accuracy: 0.7789
 5600/60000 [=>............................] - ETA: 1:38 - loss: 0.6834 - categorical_accuracy: 0.7796
 5632/60000 [=>............................] - ETA: 1:38 - loss: 0.6809 - categorical_accuracy: 0.7804
 5664/60000 [=>............................] - ETA: 1:38 - loss: 0.6791 - categorical_accuracy: 0.7809
 5696/60000 [=>............................] - ETA: 1:38 - loss: 0.6764 - categorical_accuracy: 0.7818
 5728/60000 [=>............................] - ETA: 1:38 - loss: 0.6732 - categorical_accuracy: 0.7826
 5760/60000 [=>............................] - ETA: 1:38 - loss: 0.6700 - categorical_accuracy: 0.7837
 5792/60000 [=>............................] - ETA: 1:37 - loss: 0.6686 - categorical_accuracy: 0.7844
 5824/60000 [=>............................] - ETA: 1:37 - loss: 0.6669 - categorical_accuracy: 0.7847
 5856/60000 [=>............................] - ETA: 1:37 - loss: 0.6670 - categorical_accuracy: 0.7848
 5888/60000 [=>............................] - ETA: 1:37 - loss: 0.6654 - categorical_accuracy: 0.7852
 5920/60000 [=>............................] - ETA: 1:37 - loss: 0.6629 - categorical_accuracy: 0.7860
 5952/60000 [=>............................] - ETA: 1:37 - loss: 0.6601 - categorical_accuracy: 0.7868
 5984/60000 [=>............................] - ETA: 1:37 - loss: 0.6571 - categorical_accuracy: 0.7878
 6016/60000 [==>...........................] - ETA: 1:37 - loss: 0.6564 - categorical_accuracy: 0.7881
 6048/60000 [==>...........................] - ETA: 1:37 - loss: 0.6547 - categorical_accuracy: 0.7885
 6080/60000 [==>...........................] - ETA: 1:37 - loss: 0.6522 - categorical_accuracy: 0.7893
 6112/60000 [==>...........................] - ETA: 1:37 - loss: 0.6513 - categorical_accuracy: 0.7898
 6144/60000 [==>...........................] - ETA: 1:37 - loss: 0.6489 - categorical_accuracy: 0.7905
 6176/60000 [==>...........................] - ETA: 1:37 - loss: 0.6462 - categorical_accuracy: 0.7915
 6208/60000 [==>...........................] - ETA: 1:37 - loss: 0.6435 - categorical_accuracy: 0.7924
 6240/60000 [==>...........................] - ETA: 1:37 - loss: 0.6407 - categorical_accuracy: 0.7931
 6272/60000 [==>...........................] - ETA: 1:37 - loss: 0.6400 - categorical_accuracy: 0.7935
 6304/60000 [==>...........................] - ETA: 1:37 - loss: 0.6376 - categorical_accuracy: 0.7943
 6336/60000 [==>...........................] - ETA: 1:36 - loss: 0.6407 - categorical_accuracy: 0.7943
 6368/60000 [==>...........................] - ETA: 1:36 - loss: 0.6401 - categorical_accuracy: 0.7944
 6400/60000 [==>...........................] - ETA: 1:36 - loss: 0.6394 - categorical_accuracy: 0.7948
 6432/60000 [==>...........................] - ETA: 1:36 - loss: 0.6372 - categorical_accuracy: 0.7954
 6464/60000 [==>...........................] - ETA: 1:36 - loss: 0.6365 - categorical_accuracy: 0.7956
 6496/60000 [==>...........................] - ETA: 1:36 - loss: 0.6354 - categorical_accuracy: 0.7957
 6528/60000 [==>...........................] - ETA: 1:36 - loss: 0.6343 - categorical_accuracy: 0.7961
 6560/60000 [==>...........................] - ETA: 1:36 - loss: 0.6326 - categorical_accuracy: 0.7966
 6592/60000 [==>...........................] - ETA: 1:36 - loss: 0.6310 - categorical_accuracy: 0.7972
 6624/60000 [==>...........................] - ETA: 1:36 - loss: 0.6301 - categorical_accuracy: 0.7974
 6656/60000 [==>...........................] - ETA: 1:36 - loss: 0.6288 - categorical_accuracy: 0.7978
 6688/60000 [==>...........................] - ETA: 1:36 - loss: 0.6282 - categorical_accuracy: 0.7980
 6720/60000 [==>...........................] - ETA: 1:36 - loss: 0.6269 - categorical_accuracy: 0.7982
 6752/60000 [==>...........................] - ETA: 1:36 - loss: 0.6257 - categorical_accuracy: 0.7990
 6784/60000 [==>...........................] - ETA: 1:36 - loss: 0.6252 - categorical_accuracy: 0.7992
 6816/60000 [==>...........................] - ETA: 1:36 - loss: 0.6237 - categorical_accuracy: 0.7999
 6848/60000 [==>...........................] - ETA: 1:36 - loss: 0.6217 - categorical_accuracy: 0.8007
 6880/60000 [==>...........................] - ETA: 1:35 - loss: 0.6193 - categorical_accuracy: 0.8015
 6912/60000 [==>...........................] - ETA: 1:35 - loss: 0.6181 - categorical_accuracy: 0.8019
 6944/60000 [==>...........................] - ETA: 1:35 - loss: 0.6173 - categorical_accuracy: 0.8024
 6976/60000 [==>...........................] - ETA: 1:35 - loss: 0.6155 - categorical_accuracy: 0.8029
 7008/60000 [==>...........................] - ETA: 1:35 - loss: 0.6141 - categorical_accuracy: 0.8034
 7040/60000 [==>...........................] - ETA: 1:35 - loss: 0.6135 - categorical_accuracy: 0.8040
 7072/60000 [==>...........................] - ETA: 1:35 - loss: 0.6111 - categorical_accuracy: 0.8049
 7104/60000 [==>...........................] - ETA: 1:35 - loss: 0.6088 - categorical_accuracy: 0.8057
 7136/60000 [==>...........................] - ETA: 1:35 - loss: 0.6070 - categorical_accuracy: 0.8063
 7168/60000 [==>...........................] - ETA: 1:35 - loss: 0.6051 - categorical_accuracy: 0.8071
 7200/60000 [==>...........................] - ETA: 1:35 - loss: 0.6028 - categorical_accuracy: 0.8078
 7232/60000 [==>...........................] - ETA: 1:35 - loss: 0.6019 - categorical_accuracy: 0.8082
 7264/60000 [==>...........................] - ETA: 1:35 - loss: 0.6010 - categorical_accuracy: 0.8088
 7296/60000 [==>...........................] - ETA: 1:35 - loss: 0.5988 - categorical_accuracy: 0.8096
 7328/60000 [==>...........................] - ETA: 1:34 - loss: 0.5967 - categorical_accuracy: 0.8103
 7360/60000 [==>...........................] - ETA: 1:34 - loss: 0.5944 - categorical_accuracy: 0.8111
 7392/60000 [==>...........................] - ETA: 1:34 - loss: 0.5923 - categorical_accuracy: 0.8118
 7424/60000 [==>...........................] - ETA: 1:34 - loss: 0.5906 - categorical_accuracy: 0.8124
 7456/60000 [==>...........................] - ETA: 1:34 - loss: 0.5892 - categorical_accuracy: 0.8129
 7488/60000 [==>...........................] - ETA: 1:34 - loss: 0.5875 - categorical_accuracy: 0.8133
 7520/60000 [==>...........................] - ETA: 1:34 - loss: 0.5856 - categorical_accuracy: 0.8140
 7552/60000 [==>...........................] - ETA: 1:34 - loss: 0.5846 - categorical_accuracy: 0.8142
 7584/60000 [==>...........................] - ETA: 1:34 - loss: 0.5837 - categorical_accuracy: 0.8146
 7616/60000 [==>...........................] - ETA: 1:34 - loss: 0.5836 - categorical_accuracy: 0.8149
 7648/60000 [==>...........................] - ETA: 1:34 - loss: 0.5822 - categorical_accuracy: 0.8152
 7680/60000 [==>...........................] - ETA: 1:34 - loss: 0.5806 - categorical_accuracy: 0.8158
 7712/60000 [==>...........................] - ETA: 1:34 - loss: 0.5797 - categorical_accuracy: 0.8161
 7744/60000 [==>...........................] - ETA: 1:34 - loss: 0.5779 - categorical_accuracy: 0.8166
 7776/60000 [==>...........................] - ETA: 1:34 - loss: 0.5767 - categorical_accuracy: 0.8170
 7808/60000 [==>...........................] - ETA: 1:34 - loss: 0.5755 - categorical_accuracy: 0.8174
 7840/60000 [==>...........................] - ETA: 1:34 - loss: 0.5733 - categorical_accuracy: 0.8181
 7872/60000 [==>...........................] - ETA: 1:33 - loss: 0.5716 - categorical_accuracy: 0.8187
 7904/60000 [==>...........................] - ETA: 1:33 - loss: 0.5704 - categorical_accuracy: 0.8192
 7936/60000 [==>...........................] - ETA: 1:33 - loss: 0.5697 - categorical_accuracy: 0.8197
 7968/60000 [==>...........................] - ETA: 1:33 - loss: 0.5676 - categorical_accuracy: 0.8204
 8000/60000 [===>..........................] - ETA: 1:33 - loss: 0.5677 - categorical_accuracy: 0.8205
 8032/60000 [===>..........................] - ETA: 1:33 - loss: 0.5657 - categorical_accuracy: 0.8212
 8064/60000 [===>..........................] - ETA: 1:33 - loss: 0.5651 - categorical_accuracy: 0.8211
 8096/60000 [===>..........................] - ETA: 1:33 - loss: 0.5636 - categorical_accuracy: 0.8214
 8128/60000 [===>..........................] - ETA: 1:33 - loss: 0.5618 - categorical_accuracy: 0.8220
 8160/60000 [===>..........................] - ETA: 1:33 - loss: 0.5603 - categorical_accuracy: 0.8224
 8192/60000 [===>..........................] - ETA: 1:33 - loss: 0.5589 - categorical_accuracy: 0.8229
 8224/60000 [===>..........................] - ETA: 1:33 - loss: 0.5577 - categorical_accuracy: 0.8230
 8256/60000 [===>..........................] - ETA: 1:33 - loss: 0.5563 - categorical_accuracy: 0.8234
 8288/60000 [===>..........................] - ETA: 1:33 - loss: 0.5552 - categorical_accuracy: 0.8238
 8320/60000 [===>..........................] - ETA: 1:33 - loss: 0.5539 - categorical_accuracy: 0.8242
 8352/60000 [===>..........................] - ETA: 1:33 - loss: 0.5535 - categorical_accuracy: 0.8241
 8384/60000 [===>..........................] - ETA: 1:33 - loss: 0.5524 - categorical_accuracy: 0.8244
 8416/60000 [===>..........................] - ETA: 1:33 - loss: 0.5512 - categorical_accuracy: 0.8249
 8448/60000 [===>..........................] - ETA: 1:33 - loss: 0.5504 - categorical_accuracy: 0.8252
 8480/60000 [===>..........................] - ETA: 1:33 - loss: 0.5485 - categorical_accuracy: 0.8258
 8512/60000 [===>..........................] - ETA: 1:32 - loss: 0.5465 - categorical_accuracy: 0.8265
 8544/60000 [===>..........................] - ETA: 1:32 - loss: 0.5453 - categorical_accuracy: 0.8270
 8576/60000 [===>..........................] - ETA: 1:32 - loss: 0.5435 - categorical_accuracy: 0.8277
 8608/60000 [===>..........................] - ETA: 1:32 - loss: 0.5423 - categorical_accuracy: 0.8281
 8640/60000 [===>..........................] - ETA: 1:32 - loss: 0.5407 - categorical_accuracy: 0.8286
 8672/60000 [===>..........................] - ETA: 1:32 - loss: 0.5391 - categorical_accuracy: 0.8291
 8704/60000 [===>..........................] - ETA: 1:32 - loss: 0.5377 - categorical_accuracy: 0.8296
 8736/60000 [===>..........................] - ETA: 1:32 - loss: 0.5363 - categorical_accuracy: 0.8300
 8768/60000 [===>..........................] - ETA: 1:32 - loss: 0.5348 - categorical_accuracy: 0.8305
 8800/60000 [===>..........................] - ETA: 1:32 - loss: 0.5348 - categorical_accuracy: 0.8308
 8832/60000 [===>..........................] - ETA: 1:32 - loss: 0.5337 - categorical_accuracy: 0.8312
 8864/60000 [===>..........................] - ETA: 1:32 - loss: 0.5326 - categorical_accuracy: 0.8315
 8896/60000 [===>..........................] - ETA: 1:32 - loss: 0.5329 - categorical_accuracy: 0.8316
 8928/60000 [===>..........................] - ETA: 1:32 - loss: 0.5313 - categorical_accuracy: 0.8321
 8960/60000 [===>..........................] - ETA: 1:32 - loss: 0.5303 - categorical_accuracy: 0.8325
 8992/60000 [===>..........................] - ETA: 1:32 - loss: 0.5306 - categorical_accuracy: 0.8325
 9024/60000 [===>..........................] - ETA: 1:32 - loss: 0.5295 - categorical_accuracy: 0.8329
 9056/60000 [===>..........................] - ETA: 1:32 - loss: 0.5283 - categorical_accuracy: 0.8333
 9088/60000 [===>..........................] - ETA: 1:31 - loss: 0.5278 - categorical_accuracy: 0.8336
 9120/60000 [===>..........................] - ETA: 1:31 - loss: 0.5263 - categorical_accuracy: 0.8342
 9152/60000 [===>..........................] - ETA: 1:31 - loss: 0.5250 - categorical_accuracy: 0.8345
 9184/60000 [===>..........................] - ETA: 1:31 - loss: 0.5245 - categorical_accuracy: 0.8346
 9216/60000 [===>..........................] - ETA: 1:31 - loss: 0.5236 - categorical_accuracy: 0.8349
 9248/60000 [===>..........................] - ETA: 1:31 - loss: 0.5229 - categorical_accuracy: 0.8350
 9280/60000 [===>..........................] - ETA: 1:31 - loss: 0.5219 - categorical_accuracy: 0.8353
 9312/60000 [===>..........................] - ETA: 1:31 - loss: 0.5221 - categorical_accuracy: 0.8354
 9344/60000 [===>..........................] - ETA: 1:31 - loss: 0.5214 - categorical_accuracy: 0.8356
 9376/60000 [===>..........................] - ETA: 1:31 - loss: 0.5212 - categorical_accuracy: 0.8356
 9408/60000 [===>..........................] - ETA: 1:31 - loss: 0.5203 - categorical_accuracy: 0.8361
 9440/60000 [===>..........................] - ETA: 1:31 - loss: 0.5194 - categorical_accuracy: 0.8362
 9472/60000 [===>..........................] - ETA: 1:31 - loss: 0.5179 - categorical_accuracy: 0.8367
 9504/60000 [===>..........................] - ETA: 1:31 - loss: 0.5165 - categorical_accuracy: 0.8372
 9536/60000 [===>..........................] - ETA: 1:31 - loss: 0.5149 - categorical_accuracy: 0.8378
 9568/60000 [===>..........................] - ETA: 1:31 - loss: 0.5135 - categorical_accuracy: 0.8382
 9600/60000 [===>..........................] - ETA: 1:31 - loss: 0.5125 - categorical_accuracy: 0.8384
 9632/60000 [===>..........................] - ETA: 1:31 - loss: 0.5114 - categorical_accuracy: 0.8388
 9664/60000 [===>..........................] - ETA: 1:31 - loss: 0.5110 - categorical_accuracy: 0.8389
 9696/60000 [===>..........................] - ETA: 1:30 - loss: 0.5096 - categorical_accuracy: 0.8393
 9728/60000 [===>..........................] - ETA: 1:30 - loss: 0.5096 - categorical_accuracy: 0.8394
 9760/60000 [===>..........................] - ETA: 1:30 - loss: 0.5085 - categorical_accuracy: 0.8399
 9792/60000 [===>..........................] - ETA: 1:30 - loss: 0.5074 - categorical_accuracy: 0.8402
 9824/60000 [===>..........................] - ETA: 1:30 - loss: 0.5075 - categorical_accuracy: 0.8402
 9856/60000 [===>..........................] - ETA: 1:30 - loss: 0.5067 - categorical_accuracy: 0.8404
 9888/60000 [===>..........................] - ETA: 1:30 - loss: 0.5055 - categorical_accuracy: 0.8408
 9920/60000 [===>..........................] - ETA: 1:30 - loss: 0.5044 - categorical_accuracy: 0.8411
 9952/60000 [===>..........................] - ETA: 1:30 - loss: 0.5033 - categorical_accuracy: 0.8414
 9984/60000 [===>..........................] - ETA: 1:30 - loss: 0.5024 - categorical_accuracy: 0.8417
10016/60000 [====>.........................] - ETA: 1:30 - loss: 0.5021 - categorical_accuracy: 0.8419
10048/60000 [====>.........................] - ETA: 1:30 - loss: 0.5015 - categorical_accuracy: 0.8420
10080/60000 [====>.........................] - ETA: 1:30 - loss: 0.5002 - categorical_accuracy: 0.8424
10112/60000 [====>.........................] - ETA: 1:30 - loss: 0.5002 - categorical_accuracy: 0.8426
10144/60000 [====>.........................] - ETA: 1:30 - loss: 0.4993 - categorical_accuracy: 0.8428
10176/60000 [====>.........................] - ETA: 1:30 - loss: 0.4979 - categorical_accuracy: 0.8433
10208/60000 [====>.........................] - ETA: 1:30 - loss: 0.4968 - categorical_accuracy: 0.8437
10240/60000 [====>.........................] - ETA: 1:30 - loss: 0.4965 - categorical_accuracy: 0.8438
10272/60000 [====>.........................] - ETA: 1:29 - loss: 0.4954 - categorical_accuracy: 0.8442
10304/60000 [====>.........................] - ETA: 1:29 - loss: 0.4944 - categorical_accuracy: 0.8445
10336/60000 [====>.........................] - ETA: 1:29 - loss: 0.4933 - categorical_accuracy: 0.8449
10368/60000 [====>.........................] - ETA: 1:29 - loss: 0.4928 - categorical_accuracy: 0.8451
10400/60000 [====>.........................] - ETA: 1:29 - loss: 0.4918 - categorical_accuracy: 0.8455
10432/60000 [====>.........................] - ETA: 1:29 - loss: 0.4913 - categorical_accuracy: 0.8456
10464/60000 [====>.........................] - ETA: 1:29 - loss: 0.4901 - categorical_accuracy: 0.8459
10496/60000 [====>.........................] - ETA: 1:29 - loss: 0.4889 - categorical_accuracy: 0.8462
10528/60000 [====>.........................] - ETA: 1:29 - loss: 0.4893 - categorical_accuracy: 0.8464
10560/60000 [====>.........................] - ETA: 1:29 - loss: 0.4885 - categorical_accuracy: 0.8466
10592/60000 [====>.........................] - ETA: 1:29 - loss: 0.4877 - categorical_accuracy: 0.8468
10624/60000 [====>.........................] - ETA: 1:29 - loss: 0.4863 - categorical_accuracy: 0.8472
10656/60000 [====>.........................] - ETA: 1:29 - loss: 0.4851 - categorical_accuracy: 0.8477
10688/60000 [====>.........................] - ETA: 1:29 - loss: 0.4842 - categorical_accuracy: 0.8480
10720/60000 [====>.........................] - ETA: 1:28 - loss: 0.4834 - categorical_accuracy: 0.8481
10752/60000 [====>.........................] - ETA: 1:28 - loss: 0.4824 - categorical_accuracy: 0.8484
10784/60000 [====>.........................] - ETA: 1:28 - loss: 0.4815 - categorical_accuracy: 0.8487
10816/60000 [====>.........................] - ETA: 1:28 - loss: 0.4804 - categorical_accuracy: 0.8489
10848/60000 [====>.........................] - ETA: 1:28 - loss: 0.4796 - categorical_accuracy: 0.8491
10880/60000 [====>.........................] - ETA: 1:28 - loss: 0.4793 - categorical_accuracy: 0.8490
10912/60000 [====>.........................] - ETA: 1:28 - loss: 0.4784 - categorical_accuracy: 0.8492
10944/60000 [====>.........................] - ETA: 1:28 - loss: 0.4775 - categorical_accuracy: 0.8495
10976/60000 [====>.........................] - ETA: 1:28 - loss: 0.4765 - categorical_accuracy: 0.8498
11008/60000 [====>.........................] - ETA: 1:28 - loss: 0.4754 - categorical_accuracy: 0.8500
11040/60000 [====>.........................] - ETA: 1:28 - loss: 0.4754 - categorical_accuracy: 0.8502
11072/60000 [====>.........................] - ETA: 1:28 - loss: 0.4744 - categorical_accuracy: 0.8505
11104/60000 [====>.........................] - ETA: 1:28 - loss: 0.4738 - categorical_accuracy: 0.8508
11136/60000 [====>.........................] - ETA: 1:28 - loss: 0.4729 - categorical_accuracy: 0.8509
11168/60000 [====>.........................] - ETA: 1:28 - loss: 0.4717 - categorical_accuracy: 0.8514
11200/60000 [====>.........................] - ETA: 1:28 - loss: 0.4711 - categorical_accuracy: 0.8516
11232/60000 [====>.........................] - ETA: 1:28 - loss: 0.4707 - categorical_accuracy: 0.8516
11264/60000 [====>.........................] - ETA: 1:28 - loss: 0.4696 - categorical_accuracy: 0.8519
11296/60000 [====>.........................] - ETA: 1:27 - loss: 0.4701 - categorical_accuracy: 0.8520
11328/60000 [====>.........................] - ETA: 1:27 - loss: 0.4690 - categorical_accuracy: 0.8524
11360/60000 [====>.........................] - ETA: 1:27 - loss: 0.4679 - categorical_accuracy: 0.8527
11392/60000 [====>.........................] - ETA: 1:27 - loss: 0.4670 - categorical_accuracy: 0.8530
11424/60000 [====>.........................] - ETA: 1:27 - loss: 0.4661 - categorical_accuracy: 0.8532
11456/60000 [====>.........................] - ETA: 1:27 - loss: 0.4655 - categorical_accuracy: 0.8534
11488/60000 [====>.........................] - ETA: 1:27 - loss: 0.4643 - categorical_accuracy: 0.8538
11520/60000 [====>.........................] - ETA: 1:27 - loss: 0.4633 - categorical_accuracy: 0.8541
11552/60000 [====>.........................] - ETA: 1:27 - loss: 0.4627 - categorical_accuracy: 0.8542
11584/60000 [====>.........................] - ETA: 1:27 - loss: 0.4617 - categorical_accuracy: 0.8545
11616/60000 [====>.........................] - ETA: 1:27 - loss: 0.4610 - categorical_accuracy: 0.8549
11648/60000 [====>.........................] - ETA: 1:27 - loss: 0.4599 - categorical_accuracy: 0.8553
11680/60000 [====>.........................] - ETA: 1:27 - loss: 0.4593 - categorical_accuracy: 0.8556
11712/60000 [====>.........................] - ETA: 1:27 - loss: 0.4584 - categorical_accuracy: 0.8559
11744/60000 [====>.........................] - ETA: 1:27 - loss: 0.4577 - categorical_accuracy: 0.8561
11776/60000 [====>.........................] - ETA: 1:27 - loss: 0.4578 - categorical_accuracy: 0.8562
11808/60000 [====>.........................] - ETA: 1:27 - loss: 0.4566 - categorical_accuracy: 0.8566
11840/60000 [====>.........................] - ETA: 1:26 - loss: 0.4558 - categorical_accuracy: 0.8569
11872/60000 [====>.........................] - ETA: 1:26 - loss: 0.4553 - categorical_accuracy: 0.8571
11904/60000 [====>.........................] - ETA: 1:26 - loss: 0.4549 - categorical_accuracy: 0.8572
11936/60000 [====>.........................] - ETA: 1:26 - loss: 0.4542 - categorical_accuracy: 0.8574
11968/60000 [====>.........................] - ETA: 1:26 - loss: 0.4534 - categorical_accuracy: 0.8576
12000/60000 [=====>........................] - ETA: 1:26 - loss: 0.4531 - categorical_accuracy: 0.8579
12032/60000 [=====>........................] - ETA: 1:26 - loss: 0.4522 - categorical_accuracy: 0.8582
12064/60000 [=====>........................] - ETA: 1:26 - loss: 0.4516 - categorical_accuracy: 0.8583
12096/60000 [=====>........................] - ETA: 1:26 - loss: 0.4505 - categorical_accuracy: 0.8587
12128/60000 [=====>........................] - ETA: 1:26 - loss: 0.4496 - categorical_accuracy: 0.8589
12160/60000 [=====>........................] - ETA: 1:26 - loss: 0.4487 - categorical_accuracy: 0.8591
12192/60000 [=====>........................] - ETA: 1:26 - loss: 0.4481 - categorical_accuracy: 0.8593
12224/60000 [=====>........................] - ETA: 1:26 - loss: 0.4472 - categorical_accuracy: 0.8597
12256/60000 [=====>........................] - ETA: 1:26 - loss: 0.4461 - categorical_accuracy: 0.8601
12288/60000 [=====>........................] - ETA: 1:26 - loss: 0.4456 - categorical_accuracy: 0.8602
12320/60000 [=====>........................] - ETA: 1:26 - loss: 0.4453 - categorical_accuracy: 0.8603
12352/60000 [=====>........................] - ETA: 1:26 - loss: 0.4449 - categorical_accuracy: 0.8605
12384/60000 [=====>........................] - ETA: 1:26 - loss: 0.4438 - categorical_accuracy: 0.8609
12416/60000 [=====>........................] - ETA: 1:25 - loss: 0.4436 - categorical_accuracy: 0.8611
12448/60000 [=====>........................] - ETA: 1:25 - loss: 0.4427 - categorical_accuracy: 0.8613
12480/60000 [=====>........................] - ETA: 1:25 - loss: 0.4422 - categorical_accuracy: 0.8615
12512/60000 [=====>........................] - ETA: 1:25 - loss: 0.4427 - categorical_accuracy: 0.8616
12544/60000 [=====>........................] - ETA: 1:25 - loss: 0.4418 - categorical_accuracy: 0.8618
12576/60000 [=====>........................] - ETA: 1:25 - loss: 0.4413 - categorical_accuracy: 0.8619
12608/60000 [=====>........................] - ETA: 1:25 - loss: 0.4408 - categorical_accuracy: 0.8621
12640/60000 [=====>........................] - ETA: 1:25 - loss: 0.4403 - categorical_accuracy: 0.8623
12672/60000 [=====>........................] - ETA: 1:25 - loss: 0.4401 - categorical_accuracy: 0.8625
12704/60000 [=====>........................] - ETA: 1:25 - loss: 0.4393 - categorical_accuracy: 0.8627
12736/60000 [=====>........................] - ETA: 1:25 - loss: 0.4384 - categorical_accuracy: 0.8630
12768/60000 [=====>........................] - ETA: 1:25 - loss: 0.4378 - categorical_accuracy: 0.8633
12800/60000 [=====>........................] - ETA: 1:25 - loss: 0.4369 - categorical_accuracy: 0.8636
12832/60000 [=====>........................] - ETA: 1:25 - loss: 0.4365 - categorical_accuracy: 0.8637
12864/60000 [=====>........................] - ETA: 1:25 - loss: 0.4359 - categorical_accuracy: 0.8639
12896/60000 [=====>........................] - ETA: 1:25 - loss: 0.4354 - categorical_accuracy: 0.8641
12928/60000 [=====>........................] - ETA: 1:24 - loss: 0.4345 - categorical_accuracy: 0.8644
12960/60000 [=====>........................] - ETA: 1:24 - loss: 0.4336 - categorical_accuracy: 0.8647
12992/60000 [=====>........................] - ETA: 1:24 - loss: 0.4327 - categorical_accuracy: 0.8650
13024/60000 [=====>........................] - ETA: 1:24 - loss: 0.4323 - categorical_accuracy: 0.8650
13056/60000 [=====>........................] - ETA: 1:24 - loss: 0.4313 - categorical_accuracy: 0.8653
13088/60000 [=====>........................] - ETA: 1:24 - loss: 0.4304 - categorical_accuracy: 0.8657
13120/60000 [=====>........................] - ETA: 1:24 - loss: 0.4294 - categorical_accuracy: 0.8660
13152/60000 [=====>........................] - ETA: 1:24 - loss: 0.4294 - categorical_accuracy: 0.8661
13184/60000 [=====>........................] - ETA: 1:24 - loss: 0.4285 - categorical_accuracy: 0.8664
13216/60000 [=====>........................] - ETA: 1:24 - loss: 0.4275 - categorical_accuracy: 0.8668
13248/60000 [=====>........................] - ETA: 1:24 - loss: 0.4271 - categorical_accuracy: 0.8669
13280/60000 [=====>........................] - ETA: 1:24 - loss: 0.4266 - categorical_accuracy: 0.8671
13312/60000 [=====>........................] - ETA: 1:24 - loss: 0.4266 - categorical_accuracy: 0.8671
13344/60000 [=====>........................] - ETA: 1:24 - loss: 0.4260 - categorical_accuracy: 0.8673
13376/60000 [=====>........................] - ETA: 1:24 - loss: 0.4255 - categorical_accuracy: 0.8674
13408/60000 [=====>........................] - ETA: 1:24 - loss: 0.4249 - categorical_accuracy: 0.8675
13440/60000 [=====>........................] - ETA: 1:23 - loss: 0.4242 - categorical_accuracy: 0.8678
13472/60000 [=====>........................] - ETA: 1:23 - loss: 0.4238 - categorical_accuracy: 0.8679
13504/60000 [=====>........................] - ETA: 1:23 - loss: 0.4228 - categorical_accuracy: 0.8683
13536/60000 [=====>........................] - ETA: 1:23 - loss: 0.4223 - categorical_accuracy: 0.8684
13568/60000 [=====>........................] - ETA: 1:23 - loss: 0.4221 - categorical_accuracy: 0.8684
13600/60000 [=====>........................] - ETA: 1:23 - loss: 0.4213 - categorical_accuracy: 0.8687
13632/60000 [=====>........................] - ETA: 1:23 - loss: 0.4211 - categorical_accuracy: 0.8689
13664/60000 [=====>........................] - ETA: 1:23 - loss: 0.4205 - categorical_accuracy: 0.8691
13696/60000 [=====>........................] - ETA: 1:23 - loss: 0.4199 - categorical_accuracy: 0.8692
13728/60000 [=====>........................] - ETA: 1:23 - loss: 0.4197 - categorical_accuracy: 0.8692
13760/60000 [=====>........................] - ETA: 1:23 - loss: 0.4199 - categorical_accuracy: 0.8691
13792/60000 [=====>........................] - ETA: 1:23 - loss: 0.4193 - categorical_accuracy: 0.8693
13824/60000 [=====>........................] - ETA: 1:23 - loss: 0.4186 - categorical_accuracy: 0.8696
13856/60000 [=====>........................] - ETA: 1:23 - loss: 0.4178 - categorical_accuracy: 0.8698
13888/60000 [=====>........................] - ETA: 1:23 - loss: 0.4173 - categorical_accuracy: 0.8700
13920/60000 [=====>........................] - ETA: 1:23 - loss: 0.4167 - categorical_accuracy: 0.8701
13952/60000 [=====>........................] - ETA: 1:23 - loss: 0.4161 - categorical_accuracy: 0.8703
13984/60000 [=====>........................] - ETA: 1:23 - loss: 0.4159 - categorical_accuracy: 0.8703
14016/60000 [======>.......................] - ETA: 1:23 - loss: 0.4154 - categorical_accuracy: 0.8704
14048/60000 [======>.......................] - ETA: 1:23 - loss: 0.4150 - categorical_accuracy: 0.8706
14080/60000 [======>.......................] - ETA: 1:23 - loss: 0.4142 - categorical_accuracy: 0.8708
14112/60000 [======>.......................] - ETA: 1:22 - loss: 0.4137 - categorical_accuracy: 0.8709
14144/60000 [======>.......................] - ETA: 1:22 - loss: 0.4130 - categorical_accuracy: 0.8711
14176/60000 [======>.......................] - ETA: 1:22 - loss: 0.4127 - categorical_accuracy: 0.8712
14208/60000 [======>.......................] - ETA: 1:22 - loss: 0.4121 - categorical_accuracy: 0.8714
14240/60000 [======>.......................] - ETA: 1:22 - loss: 0.4113 - categorical_accuracy: 0.8717
14272/60000 [======>.......................] - ETA: 1:22 - loss: 0.4112 - categorical_accuracy: 0.8718
14304/60000 [======>.......................] - ETA: 1:22 - loss: 0.4107 - categorical_accuracy: 0.8720
14336/60000 [======>.......................] - ETA: 1:22 - loss: 0.4100 - categorical_accuracy: 0.8722
14368/60000 [======>.......................] - ETA: 1:22 - loss: 0.4095 - categorical_accuracy: 0.8724
14400/60000 [======>.......................] - ETA: 1:22 - loss: 0.4090 - categorical_accuracy: 0.8726
14432/60000 [======>.......................] - ETA: 1:22 - loss: 0.4087 - categorical_accuracy: 0.8727
14464/60000 [======>.......................] - ETA: 1:22 - loss: 0.4082 - categorical_accuracy: 0.8729
14496/60000 [======>.......................] - ETA: 1:22 - loss: 0.4078 - categorical_accuracy: 0.8729
14528/60000 [======>.......................] - ETA: 1:22 - loss: 0.4071 - categorical_accuracy: 0.8731
14560/60000 [======>.......................] - ETA: 1:22 - loss: 0.4067 - categorical_accuracy: 0.8732
14592/60000 [======>.......................] - ETA: 1:22 - loss: 0.4059 - categorical_accuracy: 0.8735
14624/60000 [======>.......................] - ETA: 1:22 - loss: 0.4052 - categorical_accuracy: 0.8737
14656/60000 [======>.......................] - ETA: 1:22 - loss: 0.4047 - categorical_accuracy: 0.8739
14688/60000 [======>.......................] - ETA: 1:21 - loss: 0.4044 - categorical_accuracy: 0.8740
14720/60000 [======>.......................] - ETA: 1:21 - loss: 0.4038 - categorical_accuracy: 0.8741
14752/60000 [======>.......................] - ETA: 1:21 - loss: 0.4032 - categorical_accuracy: 0.8743
14784/60000 [======>.......................] - ETA: 1:21 - loss: 0.4024 - categorical_accuracy: 0.8745
14816/60000 [======>.......................] - ETA: 1:21 - loss: 0.4019 - categorical_accuracy: 0.8747
14848/60000 [======>.......................] - ETA: 1:21 - loss: 0.4016 - categorical_accuracy: 0.8747
14880/60000 [======>.......................] - ETA: 1:21 - loss: 0.4016 - categorical_accuracy: 0.8748
14912/60000 [======>.......................] - ETA: 1:21 - loss: 0.4012 - categorical_accuracy: 0.8749
14944/60000 [======>.......................] - ETA: 1:21 - loss: 0.4006 - categorical_accuracy: 0.8751
14976/60000 [======>.......................] - ETA: 1:21 - loss: 0.3999 - categorical_accuracy: 0.8753
15008/60000 [======>.......................] - ETA: 1:21 - loss: 0.3993 - categorical_accuracy: 0.8755
15040/60000 [======>.......................] - ETA: 1:21 - loss: 0.3990 - categorical_accuracy: 0.8757
15072/60000 [======>.......................] - ETA: 1:21 - loss: 0.3986 - categorical_accuracy: 0.8758
15104/60000 [======>.......................] - ETA: 1:21 - loss: 0.3983 - categorical_accuracy: 0.8759
15136/60000 [======>.......................] - ETA: 1:21 - loss: 0.3979 - categorical_accuracy: 0.8759
15168/60000 [======>.......................] - ETA: 1:21 - loss: 0.3978 - categorical_accuracy: 0.8759
15200/60000 [======>.......................] - ETA: 1:21 - loss: 0.3980 - categorical_accuracy: 0.8758
15232/60000 [======>.......................] - ETA: 1:21 - loss: 0.3975 - categorical_accuracy: 0.8760
15264/60000 [======>.......................] - ETA: 1:20 - loss: 0.3969 - categorical_accuracy: 0.8762
15296/60000 [======>.......................] - ETA: 1:20 - loss: 0.3963 - categorical_accuracy: 0.8764
15328/60000 [======>.......................] - ETA: 1:20 - loss: 0.3958 - categorical_accuracy: 0.8765
15360/60000 [======>.......................] - ETA: 1:20 - loss: 0.3956 - categorical_accuracy: 0.8766
15392/60000 [======>.......................] - ETA: 1:20 - loss: 0.3951 - categorical_accuracy: 0.8767
15424/60000 [======>.......................] - ETA: 1:20 - loss: 0.3947 - categorical_accuracy: 0.8768
15456/60000 [======>.......................] - ETA: 1:20 - loss: 0.3940 - categorical_accuracy: 0.8771
15488/60000 [======>.......................] - ETA: 1:20 - loss: 0.3934 - categorical_accuracy: 0.8773
15520/60000 [======>.......................] - ETA: 1:20 - loss: 0.3927 - categorical_accuracy: 0.8775
15552/60000 [======>.......................] - ETA: 1:20 - loss: 0.3921 - categorical_accuracy: 0.8776
15584/60000 [======>.......................] - ETA: 1:20 - loss: 0.3917 - categorical_accuracy: 0.8778
15616/60000 [======>.......................] - ETA: 1:20 - loss: 0.3914 - categorical_accuracy: 0.8779
15648/60000 [======>.......................] - ETA: 1:20 - loss: 0.3912 - categorical_accuracy: 0.8781
15680/60000 [======>.......................] - ETA: 1:20 - loss: 0.3910 - categorical_accuracy: 0.8781
15712/60000 [======>.......................] - ETA: 1:20 - loss: 0.3904 - categorical_accuracy: 0.8784
15744/60000 [======>.......................] - ETA: 1:20 - loss: 0.3903 - categorical_accuracy: 0.8784
15776/60000 [======>.......................] - ETA: 1:20 - loss: 0.3897 - categorical_accuracy: 0.8786
15808/60000 [======>.......................] - ETA: 1:20 - loss: 0.3898 - categorical_accuracy: 0.8786
15840/60000 [======>.......................] - ETA: 1:20 - loss: 0.3893 - categorical_accuracy: 0.8787
15872/60000 [======>.......................] - ETA: 1:20 - loss: 0.3889 - categorical_accuracy: 0.8788
15904/60000 [======>.......................] - ETA: 1:19 - loss: 0.3888 - categorical_accuracy: 0.8788
15936/60000 [======>.......................] - ETA: 1:19 - loss: 0.3881 - categorical_accuracy: 0.8791
15968/60000 [======>.......................] - ETA: 1:19 - loss: 0.3874 - categorical_accuracy: 0.8793
16000/60000 [=======>......................] - ETA: 1:19 - loss: 0.3868 - categorical_accuracy: 0.8796
16032/60000 [=======>......................] - ETA: 1:19 - loss: 0.3862 - categorical_accuracy: 0.8797
16064/60000 [=======>......................] - ETA: 1:19 - loss: 0.3856 - categorical_accuracy: 0.8799
16096/60000 [=======>......................] - ETA: 1:19 - loss: 0.3849 - categorical_accuracy: 0.8801
16128/60000 [=======>......................] - ETA: 1:19 - loss: 0.3842 - categorical_accuracy: 0.8803
16160/60000 [=======>......................] - ETA: 1:19 - loss: 0.3843 - categorical_accuracy: 0.8804
16192/60000 [=======>......................] - ETA: 1:19 - loss: 0.3839 - categorical_accuracy: 0.8806
16224/60000 [=======>......................] - ETA: 1:19 - loss: 0.3834 - categorical_accuracy: 0.8807
16256/60000 [=======>......................] - ETA: 1:19 - loss: 0.3829 - categorical_accuracy: 0.8808
16288/60000 [=======>......................] - ETA: 1:19 - loss: 0.3826 - categorical_accuracy: 0.8809
16320/60000 [=======>......................] - ETA: 1:19 - loss: 0.3822 - categorical_accuracy: 0.8810
16352/60000 [=======>......................] - ETA: 1:19 - loss: 0.3816 - categorical_accuracy: 0.8812
16384/60000 [=======>......................] - ETA: 1:19 - loss: 0.3809 - categorical_accuracy: 0.8814
16416/60000 [=======>......................] - ETA: 1:19 - loss: 0.3804 - categorical_accuracy: 0.8816
16448/60000 [=======>......................] - ETA: 1:19 - loss: 0.3797 - categorical_accuracy: 0.8818
16480/60000 [=======>......................] - ETA: 1:18 - loss: 0.3792 - categorical_accuracy: 0.8820
16512/60000 [=======>......................] - ETA: 1:18 - loss: 0.3787 - categorical_accuracy: 0.8821
16544/60000 [=======>......................] - ETA: 1:18 - loss: 0.3782 - categorical_accuracy: 0.8823
16576/60000 [=======>......................] - ETA: 1:18 - loss: 0.3776 - categorical_accuracy: 0.8824
16608/60000 [=======>......................] - ETA: 1:18 - loss: 0.3771 - categorical_accuracy: 0.8825
16640/60000 [=======>......................] - ETA: 1:18 - loss: 0.3764 - categorical_accuracy: 0.8828
16672/60000 [=======>......................] - ETA: 1:18 - loss: 0.3761 - categorical_accuracy: 0.8829
16704/60000 [=======>......................] - ETA: 1:18 - loss: 0.3758 - categorical_accuracy: 0.8829
16736/60000 [=======>......................] - ETA: 1:18 - loss: 0.3758 - categorical_accuracy: 0.8829
16768/60000 [=======>......................] - ETA: 1:18 - loss: 0.3755 - categorical_accuracy: 0.8831
16800/60000 [=======>......................] - ETA: 1:18 - loss: 0.3749 - categorical_accuracy: 0.8833
16832/60000 [=======>......................] - ETA: 1:18 - loss: 0.3750 - categorical_accuracy: 0.8833
16864/60000 [=======>......................] - ETA: 1:18 - loss: 0.3744 - categorical_accuracy: 0.8835
16896/60000 [=======>......................] - ETA: 1:18 - loss: 0.3742 - categorical_accuracy: 0.8836
16928/60000 [=======>......................] - ETA: 1:18 - loss: 0.3738 - categorical_accuracy: 0.8837
16960/60000 [=======>......................] - ETA: 1:17 - loss: 0.3737 - categorical_accuracy: 0.8837
16992/60000 [=======>......................] - ETA: 1:17 - loss: 0.3734 - categorical_accuracy: 0.8838
17024/60000 [=======>......................] - ETA: 1:17 - loss: 0.3734 - categorical_accuracy: 0.8838
17056/60000 [=======>......................] - ETA: 1:17 - loss: 0.3732 - categorical_accuracy: 0.8839
17088/60000 [=======>......................] - ETA: 1:17 - loss: 0.3725 - categorical_accuracy: 0.8841
17120/60000 [=======>......................] - ETA: 1:17 - loss: 0.3721 - categorical_accuracy: 0.8841
17152/60000 [=======>......................] - ETA: 1:17 - loss: 0.3718 - categorical_accuracy: 0.8842
17184/60000 [=======>......................] - ETA: 1:17 - loss: 0.3715 - categorical_accuracy: 0.8843
17216/60000 [=======>......................] - ETA: 1:17 - loss: 0.3711 - categorical_accuracy: 0.8844
17248/60000 [=======>......................] - ETA: 1:17 - loss: 0.3710 - categorical_accuracy: 0.8845
17280/60000 [=======>......................] - ETA: 1:17 - loss: 0.3707 - categorical_accuracy: 0.8846
17312/60000 [=======>......................] - ETA: 1:17 - loss: 0.3701 - categorical_accuracy: 0.8848
17344/60000 [=======>......................] - ETA: 1:17 - loss: 0.3696 - categorical_accuracy: 0.8849
17376/60000 [=======>......................] - ETA: 1:17 - loss: 0.3691 - categorical_accuracy: 0.8850
17408/60000 [=======>......................] - ETA: 1:17 - loss: 0.3686 - categorical_accuracy: 0.8852
17440/60000 [=======>......................] - ETA: 1:17 - loss: 0.3683 - categorical_accuracy: 0.8852
17472/60000 [=======>......................] - ETA: 1:17 - loss: 0.3682 - categorical_accuracy: 0.8852
17504/60000 [=======>......................] - ETA: 1:16 - loss: 0.3677 - categorical_accuracy: 0.8853
17536/60000 [=======>......................] - ETA: 1:16 - loss: 0.3673 - categorical_accuracy: 0.8854
17568/60000 [=======>......................] - ETA: 1:16 - loss: 0.3669 - categorical_accuracy: 0.8856
17600/60000 [=======>......................] - ETA: 1:16 - loss: 0.3665 - categorical_accuracy: 0.8857
17632/60000 [=======>......................] - ETA: 1:16 - loss: 0.3661 - categorical_accuracy: 0.8858
17664/60000 [=======>......................] - ETA: 1:16 - loss: 0.3655 - categorical_accuracy: 0.8860
17696/60000 [=======>......................] - ETA: 1:16 - loss: 0.3656 - categorical_accuracy: 0.8861
17728/60000 [=======>......................] - ETA: 1:16 - loss: 0.3653 - categorical_accuracy: 0.8862
17760/60000 [=======>......................] - ETA: 1:16 - loss: 0.3648 - categorical_accuracy: 0.8863
17792/60000 [=======>......................] - ETA: 1:16 - loss: 0.3645 - categorical_accuracy: 0.8864
17824/60000 [=======>......................] - ETA: 1:16 - loss: 0.3640 - categorical_accuracy: 0.8866
17856/60000 [=======>......................] - ETA: 1:16 - loss: 0.3639 - categorical_accuracy: 0.8866
17888/60000 [=======>......................] - ETA: 1:16 - loss: 0.3636 - categorical_accuracy: 0.8866
17920/60000 [=======>......................] - ETA: 1:16 - loss: 0.3632 - categorical_accuracy: 0.8867
17952/60000 [=======>......................] - ETA: 1:16 - loss: 0.3628 - categorical_accuracy: 0.8868
17984/60000 [=======>......................] - ETA: 1:16 - loss: 0.3622 - categorical_accuracy: 0.8870
18016/60000 [========>.....................] - ETA: 1:16 - loss: 0.3619 - categorical_accuracy: 0.8870
18048/60000 [========>.....................] - ETA: 1:15 - loss: 0.3616 - categorical_accuracy: 0.8871
18080/60000 [========>.....................] - ETA: 1:15 - loss: 0.3615 - categorical_accuracy: 0.8871
18112/60000 [========>.....................] - ETA: 1:15 - loss: 0.3613 - categorical_accuracy: 0.8871
18144/60000 [========>.....................] - ETA: 1:15 - loss: 0.3610 - categorical_accuracy: 0.8872
18176/60000 [========>.....................] - ETA: 1:15 - loss: 0.3606 - categorical_accuracy: 0.8873
18208/60000 [========>.....................] - ETA: 1:15 - loss: 0.3605 - categorical_accuracy: 0.8874
18240/60000 [========>.....................] - ETA: 1:15 - loss: 0.3603 - categorical_accuracy: 0.8874
18272/60000 [========>.....................] - ETA: 1:15 - loss: 0.3601 - categorical_accuracy: 0.8875
18304/60000 [========>.....................] - ETA: 1:15 - loss: 0.3596 - categorical_accuracy: 0.8877
18336/60000 [========>.....................] - ETA: 1:15 - loss: 0.3591 - categorical_accuracy: 0.8878
18368/60000 [========>.....................] - ETA: 1:15 - loss: 0.3586 - categorical_accuracy: 0.8880
18400/60000 [========>.....................] - ETA: 1:15 - loss: 0.3581 - categorical_accuracy: 0.8882
18432/60000 [========>.....................] - ETA: 1:15 - loss: 0.3577 - categorical_accuracy: 0.8882
18464/60000 [========>.....................] - ETA: 1:15 - loss: 0.3574 - categorical_accuracy: 0.8884
18496/60000 [========>.....................] - ETA: 1:15 - loss: 0.3572 - categorical_accuracy: 0.8884
18528/60000 [========>.....................] - ETA: 1:15 - loss: 0.3566 - categorical_accuracy: 0.8885
18560/60000 [========>.....................] - ETA: 1:15 - loss: 0.3561 - categorical_accuracy: 0.8887
18592/60000 [========>.....................] - ETA: 1:14 - loss: 0.3558 - categorical_accuracy: 0.8888
18624/60000 [========>.....................] - ETA: 1:14 - loss: 0.3554 - categorical_accuracy: 0.8889
18656/60000 [========>.....................] - ETA: 1:14 - loss: 0.3549 - categorical_accuracy: 0.8890
18688/60000 [========>.....................] - ETA: 1:14 - loss: 0.3545 - categorical_accuracy: 0.8891
18720/60000 [========>.....................] - ETA: 1:14 - loss: 0.3542 - categorical_accuracy: 0.8893
18752/60000 [========>.....................] - ETA: 1:14 - loss: 0.3537 - categorical_accuracy: 0.8895
18784/60000 [========>.....................] - ETA: 1:14 - loss: 0.3536 - categorical_accuracy: 0.8895
18816/60000 [========>.....................] - ETA: 1:14 - loss: 0.3532 - categorical_accuracy: 0.8896
18848/60000 [========>.....................] - ETA: 1:14 - loss: 0.3529 - categorical_accuracy: 0.8897
18880/60000 [========>.....................] - ETA: 1:14 - loss: 0.3524 - categorical_accuracy: 0.8898
18912/60000 [========>.....................] - ETA: 1:14 - loss: 0.3524 - categorical_accuracy: 0.8899
18944/60000 [========>.....................] - ETA: 1:14 - loss: 0.3518 - categorical_accuracy: 0.8900
18976/60000 [========>.....................] - ETA: 1:14 - loss: 0.3515 - categorical_accuracy: 0.8902
19008/60000 [========>.....................] - ETA: 1:14 - loss: 0.3513 - categorical_accuracy: 0.8903
19040/60000 [========>.....................] - ETA: 1:14 - loss: 0.3511 - categorical_accuracy: 0.8903
19072/60000 [========>.....................] - ETA: 1:14 - loss: 0.3509 - categorical_accuracy: 0.8904
19104/60000 [========>.....................] - ETA: 1:13 - loss: 0.3504 - categorical_accuracy: 0.8905
19136/60000 [========>.....................] - ETA: 1:13 - loss: 0.3501 - categorical_accuracy: 0.8906
19168/60000 [========>.....................] - ETA: 1:13 - loss: 0.3495 - categorical_accuracy: 0.8908
19200/60000 [========>.....................] - ETA: 1:13 - loss: 0.3493 - categorical_accuracy: 0.8909
19232/60000 [========>.....................] - ETA: 1:13 - loss: 0.3487 - categorical_accuracy: 0.8911
19264/60000 [========>.....................] - ETA: 1:13 - loss: 0.3484 - categorical_accuracy: 0.8912
19296/60000 [========>.....................] - ETA: 1:13 - loss: 0.3480 - categorical_accuracy: 0.8914
19328/60000 [========>.....................] - ETA: 1:13 - loss: 0.3476 - categorical_accuracy: 0.8915
19360/60000 [========>.....................] - ETA: 1:13 - loss: 0.3471 - categorical_accuracy: 0.8916
19392/60000 [========>.....................] - ETA: 1:13 - loss: 0.3466 - categorical_accuracy: 0.8918
19424/60000 [========>.....................] - ETA: 1:13 - loss: 0.3469 - categorical_accuracy: 0.8917
19456/60000 [========>.....................] - ETA: 1:13 - loss: 0.3464 - categorical_accuracy: 0.8919
19488/60000 [========>.....................] - ETA: 1:13 - loss: 0.3462 - categorical_accuracy: 0.8920
19520/60000 [========>.....................] - ETA: 1:13 - loss: 0.3456 - categorical_accuracy: 0.8922
19552/60000 [========>.....................] - ETA: 1:13 - loss: 0.3452 - categorical_accuracy: 0.8923
19584/60000 [========>.....................] - ETA: 1:13 - loss: 0.3449 - categorical_accuracy: 0.8924
19616/60000 [========>.....................] - ETA: 1:13 - loss: 0.3446 - categorical_accuracy: 0.8924
19648/60000 [========>.....................] - ETA: 1:12 - loss: 0.3441 - categorical_accuracy: 0.8926
19680/60000 [========>.....................] - ETA: 1:12 - loss: 0.3437 - categorical_accuracy: 0.8927
19712/60000 [========>.....................] - ETA: 1:12 - loss: 0.3432 - categorical_accuracy: 0.8929
19744/60000 [========>.....................] - ETA: 1:12 - loss: 0.3428 - categorical_accuracy: 0.8930
19776/60000 [========>.....................] - ETA: 1:12 - loss: 0.3427 - categorical_accuracy: 0.8931
19808/60000 [========>.....................] - ETA: 1:12 - loss: 0.3423 - categorical_accuracy: 0.8932
19840/60000 [========>.....................] - ETA: 1:12 - loss: 0.3422 - categorical_accuracy: 0.8932
19872/60000 [========>.....................] - ETA: 1:12 - loss: 0.3418 - categorical_accuracy: 0.8934
19904/60000 [========>.....................] - ETA: 1:12 - loss: 0.3414 - categorical_accuracy: 0.8935
19936/60000 [========>.....................] - ETA: 1:12 - loss: 0.3411 - categorical_accuracy: 0.8936
19968/60000 [========>.....................] - ETA: 1:12 - loss: 0.3416 - categorical_accuracy: 0.8935
20000/60000 [=========>....................] - ETA: 1:12 - loss: 0.3414 - categorical_accuracy: 0.8936
20032/60000 [=========>....................] - ETA: 1:12 - loss: 0.3414 - categorical_accuracy: 0.8937
20064/60000 [=========>....................] - ETA: 1:12 - loss: 0.3410 - categorical_accuracy: 0.8938
20096/60000 [=========>....................] - ETA: 1:12 - loss: 0.3407 - categorical_accuracy: 0.8939
20128/60000 [=========>....................] - ETA: 1:12 - loss: 0.3403 - categorical_accuracy: 0.8941
20160/60000 [=========>....................] - ETA: 1:12 - loss: 0.3401 - categorical_accuracy: 0.8940
20192/60000 [=========>....................] - ETA: 1:11 - loss: 0.3397 - categorical_accuracy: 0.8942
20224/60000 [=========>....................] - ETA: 1:11 - loss: 0.3395 - categorical_accuracy: 0.8943
20256/60000 [=========>....................] - ETA: 1:11 - loss: 0.3392 - categorical_accuracy: 0.8944
20288/60000 [=========>....................] - ETA: 1:11 - loss: 0.3389 - categorical_accuracy: 0.8945
20320/60000 [=========>....................] - ETA: 1:11 - loss: 0.3385 - categorical_accuracy: 0.8946
20352/60000 [=========>....................] - ETA: 1:11 - loss: 0.3384 - categorical_accuracy: 0.8947
20384/60000 [=========>....................] - ETA: 1:11 - loss: 0.3381 - categorical_accuracy: 0.8948
20416/60000 [=========>....................] - ETA: 1:11 - loss: 0.3378 - categorical_accuracy: 0.8949
20448/60000 [=========>....................] - ETA: 1:11 - loss: 0.3375 - categorical_accuracy: 0.8950
20480/60000 [=========>....................] - ETA: 1:11 - loss: 0.3371 - categorical_accuracy: 0.8950
20512/60000 [=========>....................] - ETA: 1:11 - loss: 0.3368 - categorical_accuracy: 0.8951
20544/60000 [=========>....................] - ETA: 1:11 - loss: 0.3367 - categorical_accuracy: 0.8952
20576/60000 [=========>....................] - ETA: 1:11 - loss: 0.3363 - categorical_accuracy: 0.8954
20608/60000 [=========>....................] - ETA: 1:11 - loss: 0.3359 - categorical_accuracy: 0.8955
20640/60000 [=========>....................] - ETA: 1:11 - loss: 0.3355 - categorical_accuracy: 0.8956
20672/60000 [=========>....................] - ETA: 1:11 - loss: 0.3350 - categorical_accuracy: 0.8958
20704/60000 [=========>....................] - ETA: 1:10 - loss: 0.3347 - categorical_accuracy: 0.8959
20736/60000 [=========>....................] - ETA: 1:10 - loss: 0.3344 - categorical_accuracy: 0.8960
20768/60000 [=========>....................] - ETA: 1:10 - loss: 0.3341 - categorical_accuracy: 0.8961
20800/60000 [=========>....................] - ETA: 1:10 - loss: 0.3338 - categorical_accuracy: 0.8962
20832/60000 [=========>....................] - ETA: 1:10 - loss: 0.3341 - categorical_accuracy: 0.8962
20864/60000 [=========>....................] - ETA: 1:10 - loss: 0.3339 - categorical_accuracy: 0.8962
20896/60000 [=========>....................] - ETA: 1:10 - loss: 0.3335 - categorical_accuracy: 0.8963
20928/60000 [=========>....................] - ETA: 1:10 - loss: 0.3330 - categorical_accuracy: 0.8965
20960/60000 [=========>....................] - ETA: 1:10 - loss: 0.3327 - categorical_accuracy: 0.8966
20992/60000 [=========>....................] - ETA: 1:10 - loss: 0.3324 - categorical_accuracy: 0.8966
21024/60000 [=========>....................] - ETA: 1:10 - loss: 0.3320 - categorical_accuracy: 0.8968
21056/60000 [=========>....................] - ETA: 1:10 - loss: 0.3319 - categorical_accuracy: 0.8967
21088/60000 [=========>....................] - ETA: 1:10 - loss: 0.3315 - categorical_accuracy: 0.8969
21120/60000 [=========>....................] - ETA: 1:10 - loss: 0.3312 - categorical_accuracy: 0.8969
21152/60000 [=========>....................] - ETA: 1:10 - loss: 0.3307 - categorical_accuracy: 0.8971
21184/60000 [=========>....................] - ETA: 1:10 - loss: 0.3304 - categorical_accuracy: 0.8972
21216/60000 [=========>....................] - ETA: 1:10 - loss: 0.3300 - categorical_accuracy: 0.8973
21248/60000 [=========>....................] - ETA: 1:09 - loss: 0.3302 - categorical_accuracy: 0.8973
21280/60000 [=========>....................] - ETA: 1:09 - loss: 0.3312 - categorical_accuracy: 0.8973
21312/60000 [=========>....................] - ETA: 1:09 - loss: 0.3309 - categorical_accuracy: 0.8974
21344/60000 [=========>....................] - ETA: 1:09 - loss: 0.3306 - categorical_accuracy: 0.8975
21376/60000 [=========>....................] - ETA: 1:09 - loss: 0.3303 - categorical_accuracy: 0.8976
21408/60000 [=========>....................] - ETA: 1:09 - loss: 0.3305 - categorical_accuracy: 0.8977
21440/60000 [=========>....................] - ETA: 1:09 - loss: 0.3304 - categorical_accuracy: 0.8977
21472/60000 [=========>....................] - ETA: 1:09 - loss: 0.3300 - categorical_accuracy: 0.8978
21504/60000 [=========>....................] - ETA: 1:09 - loss: 0.3297 - categorical_accuracy: 0.8979
21536/60000 [=========>....................] - ETA: 1:09 - loss: 0.3294 - categorical_accuracy: 0.8979
21568/60000 [=========>....................] - ETA: 1:09 - loss: 0.3291 - categorical_accuracy: 0.8980
21600/60000 [=========>....................] - ETA: 1:09 - loss: 0.3293 - categorical_accuracy: 0.8981
21632/60000 [=========>....................] - ETA: 1:09 - loss: 0.3291 - categorical_accuracy: 0.8982
21664/60000 [=========>....................] - ETA: 1:09 - loss: 0.3287 - categorical_accuracy: 0.8983
21696/60000 [=========>....................] - ETA: 1:09 - loss: 0.3282 - categorical_accuracy: 0.8985
21728/60000 [=========>....................] - ETA: 1:09 - loss: 0.3279 - categorical_accuracy: 0.8985
21760/60000 [=========>....................] - ETA: 1:08 - loss: 0.3276 - categorical_accuracy: 0.8986
21792/60000 [=========>....................] - ETA: 1:08 - loss: 0.3273 - categorical_accuracy: 0.8987
21824/60000 [=========>....................] - ETA: 1:08 - loss: 0.3269 - categorical_accuracy: 0.8988
21856/60000 [=========>....................] - ETA: 1:08 - loss: 0.3265 - categorical_accuracy: 0.8990
21888/60000 [=========>....................] - ETA: 1:08 - loss: 0.3263 - categorical_accuracy: 0.8990
21920/60000 [=========>....................] - ETA: 1:08 - loss: 0.3262 - categorical_accuracy: 0.8990
21952/60000 [=========>....................] - ETA: 1:08 - loss: 0.3260 - categorical_accuracy: 0.8990
21984/60000 [=========>....................] - ETA: 1:08 - loss: 0.3260 - categorical_accuracy: 0.8989
22016/60000 [==========>...................] - ETA: 1:08 - loss: 0.3257 - categorical_accuracy: 0.8990
22048/60000 [==========>...................] - ETA: 1:08 - loss: 0.3253 - categorical_accuracy: 0.8992
22080/60000 [==========>...................] - ETA: 1:08 - loss: 0.3249 - categorical_accuracy: 0.8993
22112/60000 [==========>...................] - ETA: 1:08 - loss: 0.3246 - categorical_accuracy: 0.8995
22144/60000 [==========>...................] - ETA: 1:08 - loss: 0.3243 - categorical_accuracy: 0.8995
22176/60000 [==========>...................] - ETA: 1:08 - loss: 0.3239 - categorical_accuracy: 0.8996
22208/60000 [==========>...................] - ETA: 1:08 - loss: 0.3235 - categorical_accuracy: 0.8997
22240/60000 [==========>...................] - ETA: 1:08 - loss: 0.3232 - categorical_accuracy: 0.8998
22272/60000 [==========>...................] - ETA: 1:08 - loss: 0.3229 - categorical_accuracy: 0.8999
22304/60000 [==========>...................] - ETA: 1:07 - loss: 0.3225 - categorical_accuracy: 0.9001
22336/60000 [==========>...................] - ETA: 1:07 - loss: 0.3225 - categorical_accuracy: 0.9001
22368/60000 [==========>...................] - ETA: 1:07 - loss: 0.3225 - categorical_accuracy: 0.9001
22400/60000 [==========>...................] - ETA: 1:07 - loss: 0.3225 - categorical_accuracy: 0.9002
22432/60000 [==========>...................] - ETA: 1:07 - loss: 0.3223 - categorical_accuracy: 0.9002
22464/60000 [==========>...................] - ETA: 1:07 - loss: 0.3221 - categorical_accuracy: 0.9003
22496/60000 [==========>...................] - ETA: 1:07 - loss: 0.3220 - categorical_accuracy: 0.9004
22528/60000 [==========>...................] - ETA: 1:07 - loss: 0.3220 - categorical_accuracy: 0.9004
22560/60000 [==========>...................] - ETA: 1:07 - loss: 0.3216 - categorical_accuracy: 0.9005
22592/60000 [==========>...................] - ETA: 1:07 - loss: 0.3218 - categorical_accuracy: 0.9005
22624/60000 [==========>...................] - ETA: 1:07 - loss: 0.3215 - categorical_accuracy: 0.9006
22656/60000 [==========>...................] - ETA: 1:07 - loss: 0.3211 - categorical_accuracy: 0.9008
22688/60000 [==========>...................] - ETA: 1:07 - loss: 0.3207 - categorical_accuracy: 0.9009
22720/60000 [==========>...................] - ETA: 1:07 - loss: 0.3205 - categorical_accuracy: 0.9010
22752/60000 [==========>...................] - ETA: 1:07 - loss: 0.3201 - categorical_accuracy: 0.9011
22784/60000 [==========>...................] - ETA: 1:07 - loss: 0.3199 - categorical_accuracy: 0.9012
22816/60000 [==========>...................] - ETA: 1:07 - loss: 0.3198 - categorical_accuracy: 0.9013
22848/60000 [==========>...................] - ETA: 1:06 - loss: 0.3194 - categorical_accuracy: 0.9014
22880/60000 [==========>...................] - ETA: 1:06 - loss: 0.3191 - categorical_accuracy: 0.9015
22912/60000 [==========>...................] - ETA: 1:06 - loss: 0.3187 - categorical_accuracy: 0.9016
22944/60000 [==========>...................] - ETA: 1:06 - loss: 0.3185 - categorical_accuracy: 0.9016
22976/60000 [==========>...................] - ETA: 1:06 - loss: 0.3182 - categorical_accuracy: 0.9018
23008/60000 [==========>...................] - ETA: 1:06 - loss: 0.3178 - categorical_accuracy: 0.9019
23040/60000 [==========>...................] - ETA: 1:06 - loss: 0.3175 - categorical_accuracy: 0.9020
23072/60000 [==========>...................] - ETA: 1:06 - loss: 0.3173 - categorical_accuracy: 0.9020
23104/60000 [==========>...................] - ETA: 1:06 - loss: 0.3170 - categorical_accuracy: 0.9021
23136/60000 [==========>...................] - ETA: 1:06 - loss: 0.3167 - categorical_accuracy: 0.9022
23168/60000 [==========>...................] - ETA: 1:06 - loss: 0.3168 - categorical_accuracy: 0.9022
23200/60000 [==========>...................] - ETA: 1:06 - loss: 0.3170 - categorical_accuracy: 0.9023
23232/60000 [==========>...................] - ETA: 1:06 - loss: 0.3169 - categorical_accuracy: 0.9023
23264/60000 [==========>...................] - ETA: 1:06 - loss: 0.3165 - categorical_accuracy: 0.9025
23296/60000 [==========>...................] - ETA: 1:06 - loss: 0.3165 - categorical_accuracy: 0.9026
23328/60000 [==========>...................] - ETA: 1:06 - loss: 0.3162 - categorical_accuracy: 0.9026
23360/60000 [==========>...................] - ETA: 1:05 - loss: 0.3161 - categorical_accuracy: 0.9027
23392/60000 [==========>...................] - ETA: 1:05 - loss: 0.3157 - categorical_accuracy: 0.9029
23424/60000 [==========>...................] - ETA: 1:05 - loss: 0.3154 - categorical_accuracy: 0.9030
23456/60000 [==========>...................] - ETA: 1:05 - loss: 0.3152 - categorical_accuracy: 0.9031
23488/60000 [==========>...................] - ETA: 1:05 - loss: 0.3149 - categorical_accuracy: 0.9032
23520/60000 [==========>...................] - ETA: 1:05 - loss: 0.3146 - categorical_accuracy: 0.9033
23552/60000 [==========>...................] - ETA: 1:05 - loss: 0.3143 - categorical_accuracy: 0.9033
23584/60000 [==========>...................] - ETA: 1:05 - loss: 0.3139 - categorical_accuracy: 0.9035
23616/60000 [==========>...................] - ETA: 1:05 - loss: 0.3139 - categorical_accuracy: 0.9034
23648/60000 [==========>...................] - ETA: 1:05 - loss: 0.3135 - categorical_accuracy: 0.9035
23680/60000 [==========>...................] - ETA: 1:05 - loss: 0.3132 - categorical_accuracy: 0.9036
23712/60000 [==========>...................] - ETA: 1:05 - loss: 0.3131 - categorical_accuracy: 0.9037
23744/60000 [==========>...................] - ETA: 1:05 - loss: 0.3128 - categorical_accuracy: 0.9038
23776/60000 [==========>...................] - ETA: 1:05 - loss: 0.3124 - categorical_accuracy: 0.9039
23808/60000 [==========>...................] - ETA: 1:05 - loss: 0.3123 - categorical_accuracy: 0.9040
23840/60000 [==========>...................] - ETA: 1:05 - loss: 0.3121 - categorical_accuracy: 0.9041
23872/60000 [==========>...................] - ETA: 1:05 - loss: 0.3119 - categorical_accuracy: 0.9041
23904/60000 [==========>...................] - ETA: 1:04 - loss: 0.3122 - categorical_accuracy: 0.9041
23936/60000 [==========>...................] - ETA: 1:04 - loss: 0.3120 - categorical_accuracy: 0.9041
23968/60000 [==========>...................] - ETA: 1:04 - loss: 0.3117 - categorical_accuracy: 0.9042
24000/60000 [===========>..................] - ETA: 1:04 - loss: 0.3114 - categorical_accuracy: 0.9043
24032/60000 [===========>..................] - ETA: 1:04 - loss: 0.3111 - categorical_accuracy: 0.9044
24064/60000 [===========>..................] - ETA: 1:04 - loss: 0.3109 - categorical_accuracy: 0.9045
24096/60000 [===========>..................] - ETA: 1:04 - loss: 0.3106 - categorical_accuracy: 0.9046
24128/60000 [===========>..................] - ETA: 1:04 - loss: 0.3108 - categorical_accuracy: 0.9046
24160/60000 [===========>..................] - ETA: 1:04 - loss: 0.3107 - categorical_accuracy: 0.9046
24192/60000 [===========>..................] - ETA: 1:04 - loss: 0.3105 - categorical_accuracy: 0.9047
24224/60000 [===========>..................] - ETA: 1:04 - loss: 0.3104 - categorical_accuracy: 0.9047
24256/60000 [===========>..................] - ETA: 1:04 - loss: 0.3103 - categorical_accuracy: 0.9048
24288/60000 [===========>..................] - ETA: 1:04 - loss: 0.3099 - categorical_accuracy: 0.9049
24320/60000 [===========>..................] - ETA: 1:04 - loss: 0.3102 - categorical_accuracy: 0.9049
24352/60000 [===========>..................] - ETA: 1:04 - loss: 0.3099 - categorical_accuracy: 0.9050
24384/60000 [===========>..................] - ETA: 1:04 - loss: 0.3096 - categorical_accuracy: 0.9051
24416/60000 [===========>..................] - ETA: 1:03 - loss: 0.3096 - categorical_accuracy: 0.9051
24448/60000 [===========>..................] - ETA: 1:03 - loss: 0.3093 - categorical_accuracy: 0.9052
24480/60000 [===========>..................] - ETA: 1:03 - loss: 0.3092 - categorical_accuracy: 0.9052
24512/60000 [===========>..................] - ETA: 1:03 - loss: 0.3088 - categorical_accuracy: 0.9054
24544/60000 [===========>..................] - ETA: 1:03 - loss: 0.3085 - categorical_accuracy: 0.9055
24576/60000 [===========>..................] - ETA: 1:03 - loss: 0.3082 - categorical_accuracy: 0.9056
24608/60000 [===========>..................] - ETA: 1:03 - loss: 0.3079 - categorical_accuracy: 0.9057
24640/60000 [===========>..................] - ETA: 1:03 - loss: 0.3078 - categorical_accuracy: 0.9057
24672/60000 [===========>..................] - ETA: 1:03 - loss: 0.3076 - categorical_accuracy: 0.9058
24704/60000 [===========>..................] - ETA: 1:03 - loss: 0.3075 - categorical_accuracy: 0.9057
24736/60000 [===========>..................] - ETA: 1:03 - loss: 0.3072 - categorical_accuracy: 0.9058
24768/60000 [===========>..................] - ETA: 1:03 - loss: 0.3069 - categorical_accuracy: 0.9059
24800/60000 [===========>..................] - ETA: 1:03 - loss: 0.3066 - categorical_accuracy: 0.9060
24832/60000 [===========>..................] - ETA: 1:03 - loss: 0.3062 - categorical_accuracy: 0.9062
24864/60000 [===========>..................] - ETA: 1:03 - loss: 0.3058 - categorical_accuracy: 0.9063
24896/60000 [===========>..................] - ETA: 1:03 - loss: 0.3059 - categorical_accuracy: 0.9063
24928/60000 [===========>..................] - ETA: 1:02 - loss: 0.3058 - categorical_accuracy: 0.9063
24960/60000 [===========>..................] - ETA: 1:02 - loss: 0.3055 - categorical_accuracy: 0.9065
24992/60000 [===========>..................] - ETA: 1:02 - loss: 0.3052 - categorical_accuracy: 0.9065
25024/60000 [===========>..................] - ETA: 1:02 - loss: 0.3049 - categorical_accuracy: 0.9066
25056/60000 [===========>..................] - ETA: 1:02 - loss: 0.3050 - categorical_accuracy: 0.9066
25088/60000 [===========>..................] - ETA: 1:02 - loss: 0.3048 - categorical_accuracy: 0.9067
25120/60000 [===========>..................] - ETA: 1:02 - loss: 0.3047 - categorical_accuracy: 0.9067
25152/60000 [===========>..................] - ETA: 1:02 - loss: 0.3044 - categorical_accuracy: 0.9068
25184/60000 [===========>..................] - ETA: 1:02 - loss: 0.3045 - categorical_accuracy: 0.9068
25216/60000 [===========>..................] - ETA: 1:02 - loss: 0.3042 - categorical_accuracy: 0.9069
25248/60000 [===========>..................] - ETA: 1:02 - loss: 0.3043 - categorical_accuracy: 0.9070
25280/60000 [===========>..................] - ETA: 1:02 - loss: 0.3040 - categorical_accuracy: 0.9070
25312/60000 [===========>..................] - ETA: 1:02 - loss: 0.3039 - categorical_accuracy: 0.9071
25344/60000 [===========>..................] - ETA: 1:02 - loss: 0.3036 - categorical_accuracy: 0.9072
25376/60000 [===========>..................] - ETA: 1:02 - loss: 0.3039 - categorical_accuracy: 0.9072
25408/60000 [===========>..................] - ETA: 1:02 - loss: 0.3037 - categorical_accuracy: 0.9072
25440/60000 [===========>..................] - ETA: 1:01 - loss: 0.3037 - categorical_accuracy: 0.9073
25472/60000 [===========>..................] - ETA: 1:01 - loss: 0.3039 - categorical_accuracy: 0.9073
25504/60000 [===========>..................] - ETA: 1:01 - loss: 0.3035 - categorical_accuracy: 0.9074
25536/60000 [===========>..................] - ETA: 1:01 - loss: 0.3033 - categorical_accuracy: 0.9075
25568/60000 [===========>..................] - ETA: 1:01 - loss: 0.3033 - categorical_accuracy: 0.9075
25600/60000 [===========>..................] - ETA: 1:01 - loss: 0.3029 - categorical_accuracy: 0.9076
25632/60000 [===========>..................] - ETA: 1:01 - loss: 0.3029 - categorical_accuracy: 0.9076
25664/60000 [===========>..................] - ETA: 1:01 - loss: 0.3029 - categorical_accuracy: 0.9076
25696/60000 [===========>..................] - ETA: 1:01 - loss: 0.3026 - categorical_accuracy: 0.9077
25728/60000 [===========>..................] - ETA: 1:01 - loss: 0.3026 - categorical_accuracy: 0.9076
25760/60000 [===========>..................] - ETA: 1:01 - loss: 0.3025 - categorical_accuracy: 0.9076
25792/60000 [===========>..................] - ETA: 1:01 - loss: 0.3021 - categorical_accuracy: 0.9078
25824/60000 [===========>..................] - ETA: 1:01 - loss: 0.3019 - categorical_accuracy: 0.9079
25856/60000 [===========>..................] - ETA: 1:01 - loss: 0.3016 - categorical_accuracy: 0.9080
25888/60000 [===========>..................] - ETA: 1:01 - loss: 0.3014 - categorical_accuracy: 0.9081
25920/60000 [===========>..................] - ETA: 1:01 - loss: 0.3011 - categorical_accuracy: 0.9081
25952/60000 [===========>..................] - ETA: 1:01 - loss: 0.3008 - categorical_accuracy: 0.9082
25984/60000 [===========>..................] - ETA: 1:00 - loss: 0.3009 - categorical_accuracy: 0.9082
26016/60000 [============>.................] - ETA: 1:00 - loss: 0.3008 - categorical_accuracy: 0.9082
26048/60000 [============>.................] - ETA: 1:00 - loss: 0.3008 - categorical_accuracy: 0.9083
26080/60000 [============>.................] - ETA: 1:00 - loss: 0.3006 - categorical_accuracy: 0.9083
26112/60000 [============>.................] - ETA: 1:00 - loss: 0.3004 - categorical_accuracy: 0.9084
26144/60000 [============>.................] - ETA: 1:00 - loss: 0.3003 - categorical_accuracy: 0.9084
26176/60000 [============>.................] - ETA: 1:00 - loss: 0.3001 - categorical_accuracy: 0.9084
26208/60000 [============>.................] - ETA: 1:00 - loss: 0.2997 - categorical_accuracy: 0.9085
26240/60000 [============>.................] - ETA: 1:00 - loss: 0.2995 - categorical_accuracy: 0.9086
26272/60000 [============>.................] - ETA: 1:00 - loss: 0.2992 - categorical_accuracy: 0.9087
26304/60000 [============>.................] - ETA: 1:00 - loss: 0.2990 - categorical_accuracy: 0.9088
26336/60000 [============>.................] - ETA: 1:00 - loss: 0.2990 - categorical_accuracy: 0.9088
26368/60000 [============>.................] - ETA: 1:00 - loss: 0.2987 - categorical_accuracy: 0.9089
26400/60000 [============>.................] - ETA: 1:00 - loss: 0.2985 - categorical_accuracy: 0.9090
26432/60000 [============>.................] - ETA: 1:00 - loss: 0.2983 - categorical_accuracy: 0.9090
26464/60000 [============>.................] - ETA: 1:00 - loss: 0.2983 - categorical_accuracy: 0.9090
26496/60000 [============>.................] - ETA: 1:00 - loss: 0.2980 - categorical_accuracy: 0.9090
26528/60000 [============>.................] - ETA: 59s - loss: 0.2978 - categorical_accuracy: 0.9092 
26560/60000 [============>.................] - ETA: 59s - loss: 0.2977 - categorical_accuracy: 0.9092
26592/60000 [============>.................] - ETA: 59s - loss: 0.2975 - categorical_accuracy: 0.9093
26624/60000 [============>.................] - ETA: 59s - loss: 0.2973 - categorical_accuracy: 0.9094
26656/60000 [============>.................] - ETA: 59s - loss: 0.2973 - categorical_accuracy: 0.9094
26688/60000 [============>.................] - ETA: 59s - loss: 0.2972 - categorical_accuracy: 0.9094
26720/60000 [============>.................] - ETA: 59s - loss: 0.2970 - categorical_accuracy: 0.9095
26752/60000 [============>.................] - ETA: 59s - loss: 0.2968 - categorical_accuracy: 0.9096
26784/60000 [============>.................] - ETA: 59s - loss: 0.2966 - categorical_accuracy: 0.9096
26816/60000 [============>.................] - ETA: 59s - loss: 0.2965 - categorical_accuracy: 0.9096
26848/60000 [============>.................] - ETA: 59s - loss: 0.2962 - categorical_accuracy: 0.9098
26880/60000 [============>.................] - ETA: 59s - loss: 0.2962 - categorical_accuracy: 0.9098
26912/60000 [============>.................] - ETA: 59s - loss: 0.2960 - categorical_accuracy: 0.9099
26944/60000 [============>.................] - ETA: 59s - loss: 0.2957 - categorical_accuracy: 0.9099
26976/60000 [============>.................] - ETA: 59s - loss: 0.2954 - categorical_accuracy: 0.9100
27008/60000 [============>.................] - ETA: 59s - loss: 0.2951 - categorical_accuracy: 0.9101
27040/60000 [============>.................] - ETA: 59s - loss: 0.2953 - categorical_accuracy: 0.9102
27072/60000 [============>.................] - ETA: 58s - loss: 0.2950 - categorical_accuracy: 0.9103
27104/60000 [============>.................] - ETA: 58s - loss: 0.2949 - categorical_accuracy: 0.9103
27136/60000 [============>.................] - ETA: 58s - loss: 0.2946 - categorical_accuracy: 0.9104
27168/60000 [============>.................] - ETA: 58s - loss: 0.2946 - categorical_accuracy: 0.9105
27200/60000 [============>.................] - ETA: 58s - loss: 0.2944 - categorical_accuracy: 0.9106
27232/60000 [============>.................] - ETA: 58s - loss: 0.2941 - categorical_accuracy: 0.9107
27264/60000 [============>.................] - ETA: 58s - loss: 0.2938 - categorical_accuracy: 0.9107
27296/60000 [============>.................] - ETA: 58s - loss: 0.2937 - categorical_accuracy: 0.9108
27328/60000 [============>.................] - ETA: 58s - loss: 0.2935 - categorical_accuracy: 0.9109
27360/60000 [============>.................] - ETA: 58s - loss: 0.2934 - categorical_accuracy: 0.9109
27392/60000 [============>.................] - ETA: 58s - loss: 0.2931 - categorical_accuracy: 0.9110
27424/60000 [============>.................] - ETA: 58s - loss: 0.2930 - categorical_accuracy: 0.9110
27456/60000 [============>.................] - ETA: 58s - loss: 0.2927 - categorical_accuracy: 0.9111
27488/60000 [============>.................] - ETA: 58s - loss: 0.2924 - categorical_accuracy: 0.9112
27520/60000 [============>.................] - ETA: 58s - loss: 0.2922 - categorical_accuracy: 0.9112
27552/60000 [============>.................] - ETA: 58s - loss: 0.2921 - categorical_accuracy: 0.9113
27584/60000 [============>.................] - ETA: 58s - loss: 0.2918 - categorical_accuracy: 0.9114
27616/60000 [============>.................] - ETA: 57s - loss: 0.2915 - categorical_accuracy: 0.9115
27648/60000 [============>.................] - ETA: 57s - loss: 0.2912 - categorical_accuracy: 0.9115
27680/60000 [============>.................] - ETA: 57s - loss: 0.2909 - categorical_accuracy: 0.9116
27712/60000 [============>.................] - ETA: 57s - loss: 0.2906 - categorical_accuracy: 0.9117
27744/60000 [============>.................] - ETA: 57s - loss: 0.2904 - categorical_accuracy: 0.9118
27776/60000 [============>.................] - ETA: 57s - loss: 0.2902 - categorical_accuracy: 0.9118
27808/60000 [============>.................] - ETA: 57s - loss: 0.2900 - categorical_accuracy: 0.9119
27840/60000 [============>.................] - ETA: 57s - loss: 0.2897 - categorical_accuracy: 0.9119
27872/60000 [============>.................] - ETA: 57s - loss: 0.2894 - categorical_accuracy: 0.9120
27904/60000 [============>.................] - ETA: 57s - loss: 0.2892 - categorical_accuracy: 0.9121
27936/60000 [============>.................] - ETA: 57s - loss: 0.2889 - categorical_accuracy: 0.9122
27968/60000 [============>.................] - ETA: 57s - loss: 0.2889 - categorical_accuracy: 0.9123
28000/60000 [=============>................] - ETA: 57s - loss: 0.2887 - categorical_accuracy: 0.9123
28032/60000 [=============>................] - ETA: 57s - loss: 0.2886 - categorical_accuracy: 0.9124
28064/60000 [=============>................] - ETA: 57s - loss: 0.2883 - categorical_accuracy: 0.9125
28096/60000 [=============>................] - ETA: 57s - loss: 0.2880 - categorical_accuracy: 0.9125
28128/60000 [=============>................] - ETA: 57s - loss: 0.2878 - categorical_accuracy: 0.9126
28160/60000 [=============>................] - ETA: 56s - loss: 0.2877 - categorical_accuracy: 0.9126
28192/60000 [=============>................] - ETA: 56s - loss: 0.2875 - categorical_accuracy: 0.9126
28224/60000 [=============>................] - ETA: 56s - loss: 0.2874 - categorical_accuracy: 0.9126
28256/60000 [=============>................] - ETA: 56s - loss: 0.2871 - categorical_accuracy: 0.9127
28288/60000 [=============>................] - ETA: 56s - loss: 0.2872 - categorical_accuracy: 0.9126
28320/60000 [=============>................] - ETA: 56s - loss: 0.2871 - categorical_accuracy: 0.9126
28352/60000 [=============>................] - ETA: 56s - loss: 0.2869 - categorical_accuracy: 0.9127
28384/60000 [=============>................] - ETA: 56s - loss: 0.2866 - categorical_accuracy: 0.9128
28416/60000 [=============>................] - ETA: 56s - loss: 0.2865 - categorical_accuracy: 0.9129
28448/60000 [=============>................] - ETA: 56s - loss: 0.2863 - categorical_accuracy: 0.9129
28480/60000 [=============>................] - ETA: 56s - loss: 0.2860 - categorical_accuracy: 0.9130
28512/60000 [=============>................] - ETA: 56s - loss: 0.2859 - categorical_accuracy: 0.9131
28544/60000 [=============>................] - ETA: 56s - loss: 0.2857 - categorical_accuracy: 0.9132
28576/60000 [=============>................] - ETA: 56s - loss: 0.2856 - categorical_accuracy: 0.9132
28608/60000 [=============>................] - ETA: 56s - loss: 0.2856 - categorical_accuracy: 0.9132
28640/60000 [=============>................] - ETA: 56s - loss: 0.2856 - categorical_accuracy: 0.9132
28672/60000 [=============>................] - ETA: 56s - loss: 0.2854 - categorical_accuracy: 0.9133
28704/60000 [=============>................] - ETA: 55s - loss: 0.2852 - categorical_accuracy: 0.9133
28736/60000 [=============>................] - ETA: 55s - loss: 0.2855 - categorical_accuracy: 0.9133
28768/60000 [=============>................] - ETA: 55s - loss: 0.2854 - categorical_accuracy: 0.9133
28800/60000 [=============>................] - ETA: 55s - loss: 0.2854 - categorical_accuracy: 0.9133
28832/60000 [=============>................] - ETA: 55s - loss: 0.2852 - categorical_accuracy: 0.9133
28864/60000 [=============>................] - ETA: 55s - loss: 0.2851 - categorical_accuracy: 0.9134
28896/60000 [=============>................] - ETA: 55s - loss: 0.2848 - categorical_accuracy: 0.9134
28928/60000 [=============>................] - ETA: 55s - loss: 0.2847 - categorical_accuracy: 0.9135
28960/60000 [=============>................] - ETA: 55s - loss: 0.2844 - categorical_accuracy: 0.9136
28992/60000 [=============>................] - ETA: 55s - loss: 0.2843 - categorical_accuracy: 0.9136
29024/60000 [=============>................] - ETA: 55s - loss: 0.2841 - categorical_accuracy: 0.9137
29056/60000 [=============>................] - ETA: 55s - loss: 0.2839 - categorical_accuracy: 0.9137
29088/60000 [=============>................] - ETA: 55s - loss: 0.2836 - categorical_accuracy: 0.9138
29120/60000 [=============>................] - ETA: 55s - loss: 0.2838 - categorical_accuracy: 0.9138
29152/60000 [=============>................] - ETA: 55s - loss: 0.2837 - categorical_accuracy: 0.9138
29184/60000 [=============>................] - ETA: 55s - loss: 0.2834 - categorical_accuracy: 0.9139
29216/60000 [=============>................] - ETA: 55s - loss: 0.2834 - categorical_accuracy: 0.9139
29248/60000 [=============>................] - ETA: 54s - loss: 0.2833 - categorical_accuracy: 0.9139
29280/60000 [=============>................] - ETA: 54s - loss: 0.2831 - categorical_accuracy: 0.9139
29312/60000 [=============>................] - ETA: 54s - loss: 0.2830 - categorical_accuracy: 0.9139
29344/60000 [=============>................] - ETA: 54s - loss: 0.2827 - categorical_accuracy: 0.9140
29376/60000 [=============>................] - ETA: 54s - loss: 0.2826 - categorical_accuracy: 0.9140
29408/60000 [=============>................] - ETA: 54s - loss: 0.2825 - categorical_accuracy: 0.9141
29440/60000 [=============>................] - ETA: 54s - loss: 0.2823 - categorical_accuracy: 0.9142
29472/60000 [=============>................] - ETA: 54s - loss: 0.2821 - categorical_accuracy: 0.9143
29504/60000 [=============>................] - ETA: 54s - loss: 0.2818 - categorical_accuracy: 0.9144
29536/60000 [=============>................] - ETA: 54s - loss: 0.2815 - categorical_accuracy: 0.9144
29568/60000 [=============>................] - ETA: 54s - loss: 0.2813 - categorical_accuracy: 0.9145
29600/60000 [=============>................] - ETA: 54s - loss: 0.2816 - categorical_accuracy: 0.9146
29632/60000 [=============>................] - ETA: 54s - loss: 0.2813 - categorical_accuracy: 0.9147
29664/60000 [=============>................] - ETA: 54s - loss: 0.2811 - categorical_accuracy: 0.9147
29696/60000 [=============>................] - ETA: 54s - loss: 0.2809 - categorical_accuracy: 0.9148
29728/60000 [=============>................] - ETA: 54s - loss: 0.2806 - categorical_accuracy: 0.9149
29760/60000 [=============>................] - ETA: 54s - loss: 0.2805 - categorical_accuracy: 0.9149
29792/60000 [=============>................] - ETA: 54s - loss: 0.2803 - categorical_accuracy: 0.9149
29824/60000 [=============>................] - ETA: 53s - loss: 0.2801 - categorical_accuracy: 0.9150
29856/60000 [=============>................] - ETA: 53s - loss: 0.2798 - categorical_accuracy: 0.9151
29888/60000 [=============>................] - ETA: 53s - loss: 0.2795 - categorical_accuracy: 0.9151
29920/60000 [=============>................] - ETA: 53s - loss: 0.2793 - categorical_accuracy: 0.9152
29952/60000 [=============>................] - ETA: 53s - loss: 0.2791 - categorical_accuracy: 0.9153
29984/60000 [=============>................] - ETA: 53s - loss: 0.2791 - categorical_accuracy: 0.9153
30016/60000 [==============>...............] - ETA: 53s - loss: 0.2790 - categorical_accuracy: 0.9154
30048/60000 [==============>...............] - ETA: 53s - loss: 0.2790 - categorical_accuracy: 0.9154
30080/60000 [==============>...............] - ETA: 53s - loss: 0.2788 - categorical_accuracy: 0.9155
30112/60000 [==============>...............] - ETA: 53s - loss: 0.2785 - categorical_accuracy: 0.9155
30144/60000 [==============>...............] - ETA: 53s - loss: 0.2784 - categorical_accuracy: 0.9156
30176/60000 [==============>...............] - ETA: 53s - loss: 0.2782 - categorical_accuracy: 0.9156
30208/60000 [==============>...............] - ETA: 53s - loss: 0.2782 - categorical_accuracy: 0.9156
30240/60000 [==============>...............] - ETA: 53s - loss: 0.2781 - categorical_accuracy: 0.9156
30272/60000 [==============>...............] - ETA: 53s - loss: 0.2780 - categorical_accuracy: 0.9156
30304/60000 [==============>...............] - ETA: 53s - loss: 0.2778 - categorical_accuracy: 0.9157
30336/60000 [==============>...............] - ETA: 53s - loss: 0.2778 - categorical_accuracy: 0.9157
30368/60000 [==============>...............] - ETA: 52s - loss: 0.2776 - categorical_accuracy: 0.9157
30400/60000 [==============>...............] - ETA: 52s - loss: 0.2773 - categorical_accuracy: 0.9158
30432/60000 [==============>...............] - ETA: 52s - loss: 0.2771 - categorical_accuracy: 0.9159
30464/60000 [==============>...............] - ETA: 52s - loss: 0.2769 - categorical_accuracy: 0.9160
30496/60000 [==============>...............] - ETA: 52s - loss: 0.2766 - categorical_accuracy: 0.9161
30528/60000 [==============>...............] - ETA: 52s - loss: 0.2763 - categorical_accuracy: 0.9161
30560/60000 [==============>...............] - ETA: 52s - loss: 0.2761 - categorical_accuracy: 0.9162
30592/60000 [==============>...............] - ETA: 52s - loss: 0.2760 - categorical_accuracy: 0.9162
30624/60000 [==============>...............] - ETA: 52s - loss: 0.2757 - categorical_accuracy: 0.9163
30656/60000 [==============>...............] - ETA: 52s - loss: 0.2758 - categorical_accuracy: 0.9163
30688/60000 [==============>...............] - ETA: 52s - loss: 0.2756 - categorical_accuracy: 0.9164
30720/60000 [==============>...............] - ETA: 52s - loss: 0.2754 - categorical_accuracy: 0.9165
30752/60000 [==============>...............] - ETA: 52s - loss: 0.2753 - categorical_accuracy: 0.9165
30784/60000 [==============>...............] - ETA: 52s - loss: 0.2750 - categorical_accuracy: 0.9165
30816/60000 [==============>...............] - ETA: 52s - loss: 0.2751 - categorical_accuracy: 0.9165
30848/60000 [==============>...............] - ETA: 52s - loss: 0.2750 - categorical_accuracy: 0.9166
30880/60000 [==============>...............] - ETA: 52s - loss: 0.2747 - categorical_accuracy: 0.9166
30912/60000 [==============>...............] - ETA: 51s - loss: 0.2746 - categorical_accuracy: 0.9167
30944/60000 [==============>...............] - ETA: 51s - loss: 0.2745 - categorical_accuracy: 0.9167
30976/60000 [==============>...............] - ETA: 51s - loss: 0.2743 - categorical_accuracy: 0.9167
31008/60000 [==============>...............] - ETA: 51s - loss: 0.2741 - categorical_accuracy: 0.9168
31040/60000 [==============>...............] - ETA: 51s - loss: 0.2741 - categorical_accuracy: 0.9167
31072/60000 [==============>...............] - ETA: 51s - loss: 0.2740 - categorical_accuracy: 0.9168
31104/60000 [==============>...............] - ETA: 51s - loss: 0.2738 - categorical_accuracy: 0.9169
31168/60000 [==============>...............] - ETA: 51s - loss: 0.2733 - categorical_accuracy: 0.9170
31200/60000 [==============>...............] - ETA: 51s - loss: 0.2730 - categorical_accuracy: 0.9171
31232/60000 [==============>...............] - ETA: 51s - loss: 0.2729 - categorical_accuracy: 0.9172
31264/60000 [==============>...............] - ETA: 51s - loss: 0.2729 - categorical_accuracy: 0.9172
31296/60000 [==============>...............] - ETA: 51s - loss: 0.2728 - categorical_accuracy: 0.9172
31328/60000 [==============>...............] - ETA: 51s - loss: 0.2728 - categorical_accuracy: 0.9173
31360/60000 [==============>...............] - ETA: 51s - loss: 0.2725 - categorical_accuracy: 0.9173
31392/60000 [==============>...............] - ETA: 51s - loss: 0.2723 - categorical_accuracy: 0.9174
31424/60000 [==============>...............] - ETA: 51s - loss: 0.2721 - categorical_accuracy: 0.9175
31456/60000 [==============>...............] - ETA: 50s - loss: 0.2719 - categorical_accuracy: 0.9176
31488/60000 [==============>...............] - ETA: 50s - loss: 0.2717 - categorical_accuracy: 0.9176
31520/60000 [==============>...............] - ETA: 50s - loss: 0.2715 - categorical_accuracy: 0.9177
31552/60000 [==============>...............] - ETA: 50s - loss: 0.2713 - categorical_accuracy: 0.9178
31584/60000 [==============>...............] - ETA: 50s - loss: 0.2713 - categorical_accuracy: 0.9177
31616/60000 [==============>...............] - ETA: 50s - loss: 0.2713 - categorical_accuracy: 0.9177
31648/60000 [==============>...............] - ETA: 50s - loss: 0.2713 - categorical_accuracy: 0.9177
31680/60000 [==============>...............] - ETA: 50s - loss: 0.2712 - categorical_accuracy: 0.9177
31712/60000 [==============>...............] - ETA: 50s - loss: 0.2711 - categorical_accuracy: 0.9178
31744/60000 [==============>...............] - ETA: 50s - loss: 0.2710 - categorical_accuracy: 0.9178
31776/60000 [==============>...............] - ETA: 50s - loss: 0.2709 - categorical_accuracy: 0.9179
31808/60000 [==============>...............] - ETA: 50s - loss: 0.2707 - categorical_accuracy: 0.9179
31840/60000 [==============>...............] - ETA: 50s - loss: 0.2706 - categorical_accuracy: 0.9180
31872/60000 [==============>...............] - ETA: 50s - loss: 0.2703 - categorical_accuracy: 0.9181
31904/60000 [==============>...............] - ETA: 50s - loss: 0.2702 - categorical_accuracy: 0.9181
31936/60000 [==============>...............] - ETA: 50s - loss: 0.2700 - categorical_accuracy: 0.9181
31968/60000 [==============>...............] - ETA: 50s - loss: 0.2699 - categorical_accuracy: 0.9181
32000/60000 [===============>..............] - ETA: 50s - loss: 0.2698 - categorical_accuracy: 0.9182
32032/60000 [===============>..............] - ETA: 49s - loss: 0.2695 - categorical_accuracy: 0.9183
32064/60000 [===============>..............] - ETA: 49s - loss: 0.2694 - categorical_accuracy: 0.9183
32096/60000 [===============>..............] - ETA: 49s - loss: 0.2694 - categorical_accuracy: 0.9183
32128/60000 [===============>..............] - ETA: 49s - loss: 0.2692 - categorical_accuracy: 0.9183
32160/60000 [===============>..............] - ETA: 49s - loss: 0.2690 - categorical_accuracy: 0.9183
32192/60000 [===============>..............] - ETA: 49s - loss: 0.2687 - categorical_accuracy: 0.9184
32224/60000 [===============>..............] - ETA: 49s - loss: 0.2685 - categorical_accuracy: 0.9185
32256/60000 [===============>..............] - ETA: 49s - loss: 0.2684 - categorical_accuracy: 0.9185
32288/60000 [===============>..............] - ETA: 49s - loss: 0.2686 - categorical_accuracy: 0.9185
32320/60000 [===============>..............] - ETA: 49s - loss: 0.2685 - categorical_accuracy: 0.9186
32352/60000 [===============>..............] - ETA: 49s - loss: 0.2683 - categorical_accuracy: 0.9186
32384/60000 [===============>..............] - ETA: 49s - loss: 0.2680 - categorical_accuracy: 0.9187
32416/60000 [===============>..............] - ETA: 49s - loss: 0.2678 - categorical_accuracy: 0.9187
32448/60000 [===============>..............] - ETA: 49s - loss: 0.2676 - categorical_accuracy: 0.9188
32480/60000 [===============>..............] - ETA: 49s - loss: 0.2674 - categorical_accuracy: 0.9189
32544/60000 [===============>..............] - ETA: 49s - loss: 0.2672 - categorical_accuracy: 0.9189
32576/60000 [===============>..............] - ETA: 48s - loss: 0.2670 - categorical_accuracy: 0.9190
32608/60000 [===============>..............] - ETA: 48s - loss: 0.2669 - categorical_accuracy: 0.9190
32640/60000 [===============>..............] - ETA: 48s - loss: 0.2667 - categorical_accuracy: 0.9191
32672/60000 [===============>..............] - ETA: 48s - loss: 0.2665 - categorical_accuracy: 0.9191
32704/60000 [===============>..............] - ETA: 48s - loss: 0.2662 - categorical_accuracy: 0.9192
32736/60000 [===============>..............] - ETA: 48s - loss: 0.2664 - categorical_accuracy: 0.9193
32768/60000 [===============>..............] - ETA: 48s - loss: 0.2663 - categorical_accuracy: 0.9193
32800/60000 [===============>..............] - ETA: 48s - loss: 0.2662 - categorical_accuracy: 0.9193
32832/60000 [===============>..............] - ETA: 48s - loss: 0.2664 - categorical_accuracy: 0.9193
32864/60000 [===============>..............] - ETA: 48s - loss: 0.2663 - categorical_accuracy: 0.9193
32896/60000 [===============>..............] - ETA: 48s - loss: 0.2661 - categorical_accuracy: 0.9193
32928/60000 [===============>..............] - ETA: 48s - loss: 0.2659 - categorical_accuracy: 0.9194
32960/60000 [===============>..............] - ETA: 48s - loss: 0.2657 - categorical_accuracy: 0.9194
32992/60000 [===============>..............] - ETA: 48s - loss: 0.2657 - categorical_accuracy: 0.9194
33024/60000 [===============>..............] - ETA: 48s - loss: 0.2658 - categorical_accuracy: 0.9194
33056/60000 [===============>..............] - ETA: 48s - loss: 0.2657 - categorical_accuracy: 0.9194
33088/60000 [===============>..............] - ETA: 48s - loss: 0.2655 - categorical_accuracy: 0.9195
33120/60000 [===============>..............] - ETA: 47s - loss: 0.2655 - categorical_accuracy: 0.9195
33152/60000 [===============>..............] - ETA: 47s - loss: 0.2653 - categorical_accuracy: 0.9196
33184/60000 [===============>..............] - ETA: 47s - loss: 0.2651 - categorical_accuracy: 0.9197
33216/60000 [===============>..............] - ETA: 47s - loss: 0.2652 - categorical_accuracy: 0.9196
33248/60000 [===============>..............] - ETA: 47s - loss: 0.2650 - categorical_accuracy: 0.9197
33280/60000 [===============>..............] - ETA: 47s - loss: 0.2649 - categorical_accuracy: 0.9197
33312/60000 [===============>..............] - ETA: 47s - loss: 0.2648 - categorical_accuracy: 0.9197
33344/60000 [===============>..............] - ETA: 47s - loss: 0.2648 - categorical_accuracy: 0.9197
33376/60000 [===============>..............] - ETA: 47s - loss: 0.2646 - categorical_accuracy: 0.9198
33408/60000 [===============>..............] - ETA: 47s - loss: 0.2643 - categorical_accuracy: 0.9199
33440/60000 [===============>..............] - ETA: 47s - loss: 0.2642 - categorical_accuracy: 0.9199
33472/60000 [===============>..............] - ETA: 47s - loss: 0.2640 - categorical_accuracy: 0.9200
33504/60000 [===============>..............] - ETA: 47s - loss: 0.2639 - categorical_accuracy: 0.9200
33536/60000 [===============>..............] - ETA: 47s - loss: 0.2637 - categorical_accuracy: 0.9201
33568/60000 [===============>..............] - ETA: 47s - loss: 0.2636 - categorical_accuracy: 0.9201
33600/60000 [===============>..............] - ETA: 47s - loss: 0.2635 - categorical_accuracy: 0.9201
33632/60000 [===============>..............] - ETA: 47s - loss: 0.2633 - categorical_accuracy: 0.9202
33664/60000 [===============>..............] - ETA: 46s - loss: 0.2631 - categorical_accuracy: 0.9202
33696/60000 [===============>..............] - ETA: 46s - loss: 0.2631 - categorical_accuracy: 0.9203
33728/60000 [===============>..............] - ETA: 46s - loss: 0.2629 - categorical_accuracy: 0.9204
33760/60000 [===============>..............] - ETA: 46s - loss: 0.2627 - categorical_accuracy: 0.9204
33792/60000 [===============>..............] - ETA: 46s - loss: 0.2626 - categorical_accuracy: 0.9204
33824/60000 [===============>..............] - ETA: 46s - loss: 0.2625 - categorical_accuracy: 0.9204
33856/60000 [===============>..............] - ETA: 46s - loss: 0.2623 - categorical_accuracy: 0.9205
33888/60000 [===============>..............] - ETA: 46s - loss: 0.2623 - categorical_accuracy: 0.9204
33920/60000 [===============>..............] - ETA: 46s - loss: 0.2622 - categorical_accuracy: 0.9205
33952/60000 [===============>..............] - ETA: 46s - loss: 0.2619 - categorical_accuracy: 0.9205
33984/60000 [===============>..............] - ETA: 46s - loss: 0.2620 - categorical_accuracy: 0.9205
34016/60000 [================>.............] - ETA: 46s - loss: 0.2619 - categorical_accuracy: 0.9205
34048/60000 [================>.............] - ETA: 46s - loss: 0.2617 - categorical_accuracy: 0.9206
34080/60000 [================>.............] - ETA: 46s - loss: 0.2615 - categorical_accuracy: 0.9207
34112/60000 [================>.............] - ETA: 46s - loss: 0.2613 - categorical_accuracy: 0.9207
34144/60000 [================>.............] - ETA: 46s - loss: 0.2613 - categorical_accuracy: 0.9207
34176/60000 [================>.............] - ETA: 46s - loss: 0.2611 - categorical_accuracy: 0.9208
34208/60000 [================>.............] - ETA: 45s - loss: 0.2611 - categorical_accuracy: 0.9208
34240/60000 [================>.............] - ETA: 45s - loss: 0.2608 - categorical_accuracy: 0.9209
34272/60000 [================>.............] - ETA: 45s - loss: 0.2606 - categorical_accuracy: 0.9210
34304/60000 [================>.............] - ETA: 45s - loss: 0.2604 - categorical_accuracy: 0.9210
34336/60000 [================>.............] - ETA: 45s - loss: 0.2603 - categorical_accuracy: 0.9210
34368/60000 [================>.............] - ETA: 45s - loss: 0.2601 - categorical_accuracy: 0.9211
34400/60000 [================>.............] - ETA: 45s - loss: 0.2599 - categorical_accuracy: 0.9212
34432/60000 [================>.............] - ETA: 45s - loss: 0.2597 - categorical_accuracy: 0.9212
34464/60000 [================>.............] - ETA: 45s - loss: 0.2597 - categorical_accuracy: 0.9213
34496/60000 [================>.............] - ETA: 45s - loss: 0.2597 - categorical_accuracy: 0.9212
34528/60000 [================>.............] - ETA: 45s - loss: 0.2596 - categorical_accuracy: 0.9213
34560/60000 [================>.............] - ETA: 45s - loss: 0.2595 - categorical_accuracy: 0.9213
34592/60000 [================>.............] - ETA: 45s - loss: 0.2593 - categorical_accuracy: 0.9213
34624/60000 [================>.............] - ETA: 45s - loss: 0.2591 - categorical_accuracy: 0.9214
34656/60000 [================>.............] - ETA: 45s - loss: 0.2589 - categorical_accuracy: 0.9214
34688/60000 [================>.............] - ETA: 45s - loss: 0.2588 - categorical_accuracy: 0.9214
34720/60000 [================>.............] - ETA: 45s - loss: 0.2586 - categorical_accuracy: 0.9215
34752/60000 [================>.............] - ETA: 44s - loss: 0.2585 - categorical_accuracy: 0.9216
34784/60000 [================>.............] - ETA: 44s - loss: 0.2583 - categorical_accuracy: 0.9216
34816/60000 [================>.............] - ETA: 44s - loss: 0.2581 - categorical_accuracy: 0.9217
34848/60000 [================>.............] - ETA: 44s - loss: 0.2580 - categorical_accuracy: 0.9217
34880/60000 [================>.............] - ETA: 44s - loss: 0.2579 - categorical_accuracy: 0.9218
34912/60000 [================>.............] - ETA: 44s - loss: 0.2577 - categorical_accuracy: 0.9218
34944/60000 [================>.............] - ETA: 44s - loss: 0.2575 - categorical_accuracy: 0.9219
34976/60000 [================>.............] - ETA: 44s - loss: 0.2574 - categorical_accuracy: 0.9219
35008/60000 [================>.............] - ETA: 44s - loss: 0.2574 - categorical_accuracy: 0.9219
35040/60000 [================>.............] - ETA: 44s - loss: 0.2571 - categorical_accuracy: 0.9220
35072/60000 [================>.............] - ETA: 44s - loss: 0.2570 - categorical_accuracy: 0.9220
35104/60000 [================>.............] - ETA: 44s - loss: 0.2569 - categorical_accuracy: 0.9220
35136/60000 [================>.............] - ETA: 44s - loss: 0.2567 - categorical_accuracy: 0.9221
35168/60000 [================>.............] - ETA: 44s - loss: 0.2565 - categorical_accuracy: 0.9221
35200/60000 [================>.............] - ETA: 44s - loss: 0.2565 - categorical_accuracy: 0.9222
35232/60000 [================>.............] - ETA: 44s - loss: 0.2563 - categorical_accuracy: 0.9222
35264/60000 [================>.............] - ETA: 44s - loss: 0.2561 - categorical_accuracy: 0.9223
35296/60000 [================>.............] - ETA: 43s - loss: 0.2561 - categorical_accuracy: 0.9223
35328/60000 [================>.............] - ETA: 43s - loss: 0.2561 - categorical_accuracy: 0.9223
35360/60000 [================>.............] - ETA: 43s - loss: 0.2560 - categorical_accuracy: 0.9223
35392/60000 [================>.............] - ETA: 43s - loss: 0.2558 - categorical_accuracy: 0.9224
35424/60000 [================>.............] - ETA: 43s - loss: 0.2556 - categorical_accuracy: 0.9225
35456/60000 [================>.............] - ETA: 43s - loss: 0.2554 - categorical_accuracy: 0.9225
35488/60000 [================>.............] - ETA: 43s - loss: 0.2553 - categorical_accuracy: 0.9225
35520/60000 [================>.............] - ETA: 43s - loss: 0.2550 - categorical_accuracy: 0.9226
35552/60000 [================>.............] - ETA: 43s - loss: 0.2549 - categorical_accuracy: 0.9227
35584/60000 [================>.............] - ETA: 43s - loss: 0.2546 - categorical_accuracy: 0.9227
35616/60000 [================>.............] - ETA: 43s - loss: 0.2546 - categorical_accuracy: 0.9228
35648/60000 [================>.............] - ETA: 43s - loss: 0.2544 - categorical_accuracy: 0.9228
35680/60000 [================>.............] - ETA: 43s - loss: 0.2543 - categorical_accuracy: 0.9229
35712/60000 [================>.............] - ETA: 43s - loss: 0.2544 - categorical_accuracy: 0.9229
35744/60000 [================>.............] - ETA: 43s - loss: 0.2542 - categorical_accuracy: 0.9230
35776/60000 [================>.............] - ETA: 43s - loss: 0.2540 - categorical_accuracy: 0.9230
35808/60000 [================>.............] - ETA: 43s - loss: 0.2538 - categorical_accuracy: 0.9231
35840/60000 [================>.............] - ETA: 43s - loss: 0.2536 - categorical_accuracy: 0.9231
35872/60000 [================>.............] - ETA: 42s - loss: 0.2534 - categorical_accuracy: 0.9232
35904/60000 [================>.............] - ETA: 42s - loss: 0.2533 - categorical_accuracy: 0.9232
35936/60000 [================>.............] - ETA: 42s - loss: 0.2531 - categorical_accuracy: 0.9233
35968/60000 [================>.............] - ETA: 42s - loss: 0.2530 - categorical_accuracy: 0.9233
36000/60000 [=================>............] - ETA: 42s - loss: 0.2530 - categorical_accuracy: 0.9233
36032/60000 [=================>............] - ETA: 42s - loss: 0.2529 - categorical_accuracy: 0.9233
36064/60000 [=================>............] - ETA: 42s - loss: 0.2528 - categorical_accuracy: 0.9234
36096/60000 [=================>............] - ETA: 42s - loss: 0.2527 - categorical_accuracy: 0.9234
36128/60000 [=================>............] - ETA: 42s - loss: 0.2525 - categorical_accuracy: 0.9235
36160/60000 [=================>............] - ETA: 42s - loss: 0.2524 - categorical_accuracy: 0.9235
36192/60000 [=================>............] - ETA: 42s - loss: 0.2522 - categorical_accuracy: 0.9235
36224/60000 [=================>............] - ETA: 42s - loss: 0.2520 - categorical_accuracy: 0.9236
36256/60000 [=================>............] - ETA: 42s - loss: 0.2518 - categorical_accuracy: 0.9236
36288/60000 [=================>............] - ETA: 42s - loss: 0.2517 - categorical_accuracy: 0.9237
36320/60000 [=================>............] - ETA: 42s - loss: 0.2516 - categorical_accuracy: 0.9237
36352/60000 [=================>............] - ETA: 42s - loss: 0.2514 - categorical_accuracy: 0.9237
36384/60000 [=================>............] - ETA: 42s - loss: 0.2512 - categorical_accuracy: 0.9238
36416/60000 [=================>............] - ETA: 41s - loss: 0.2511 - categorical_accuracy: 0.9239
36448/60000 [=================>............] - ETA: 41s - loss: 0.2512 - categorical_accuracy: 0.9238
36480/60000 [=================>............] - ETA: 41s - loss: 0.2512 - categorical_accuracy: 0.9238
36512/60000 [=================>............] - ETA: 41s - loss: 0.2511 - categorical_accuracy: 0.9239
36544/60000 [=================>............] - ETA: 41s - loss: 0.2510 - categorical_accuracy: 0.9239
36576/60000 [=================>............] - ETA: 41s - loss: 0.2508 - categorical_accuracy: 0.9239
36608/60000 [=================>............] - ETA: 41s - loss: 0.2511 - categorical_accuracy: 0.9239
36640/60000 [=================>............] - ETA: 41s - loss: 0.2509 - categorical_accuracy: 0.9240
36672/60000 [=================>............] - ETA: 41s - loss: 0.2509 - categorical_accuracy: 0.9239
36704/60000 [=================>............] - ETA: 41s - loss: 0.2508 - categorical_accuracy: 0.9240
36736/60000 [=================>............] - ETA: 41s - loss: 0.2507 - categorical_accuracy: 0.9240
36768/60000 [=================>............] - ETA: 41s - loss: 0.2505 - categorical_accuracy: 0.9240
36800/60000 [=================>............] - ETA: 41s - loss: 0.2505 - categorical_accuracy: 0.9240
36832/60000 [=================>............] - ETA: 41s - loss: 0.2504 - categorical_accuracy: 0.9241
36864/60000 [=================>............] - ETA: 41s - loss: 0.2502 - categorical_accuracy: 0.9241
36896/60000 [=================>............] - ETA: 41s - loss: 0.2500 - categorical_accuracy: 0.9242
36928/60000 [=================>............] - ETA: 41s - loss: 0.2499 - categorical_accuracy: 0.9242
36960/60000 [=================>............] - ETA: 40s - loss: 0.2498 - categorical_accuracy: 0.9242
36992/60000 [=================>............] - ETA: 40s - loss: 0.2496 - categorical_accuracy: 0.9243
37024/60000 [=================>............] - ETA: 40s - loss: 0.2496 - categorical_accuracy: 0.9242
37056/60000 [=================>............] - ETA: 40s - loss: 0.2494 - categorical_accuracy: 0.9243
37088/60000 [=================>............] - ETA: 40s - loss: 0.2492 - categorical_accuracy: 0.9243
37120/60000 [=================>............] - ETA: 40s - loss: 0.2491 - categorical_accuracy: 0.9244
37152/60000 [=================>............] - ETA: 40s - loss: 0.2492 - categorical_accuracy: 0.9243
37184/60000 [=================>............] - ETA: 40s - loss: 0.2489 - categorical_accuracy: 0.9244
37216/60000 [=================>............] - ETA: 40s - loss: 0.2488 - categorical_accuracy: 0.9245
37248/60000 [=================>............] - ETA: 40s - loss: 0.2486 - categorical_accuracy: 0.9245
37280/60000 [=================>............] - ETA: 40s - loss: 0.2484 - categorical_accuracy: 0.9246
37312/60000 [=================>............] - ETA: 40s - loss: 0.2483 - categorical_accuracy: 0.9246
37344/60000 [=================>............] - ETA: 40s - loss: 0.2483 - categorical_accuracy: 0.9246
37376/60000 [=================>............] - ETA: 40s - loss: 0.2485 - categorical_accuracy: 0.9246
37408/60000 [=================>............] - ETA: 40s - loss: 0.2483 - categorical_accuracy: 0.9247
37440/60000 [=================>............] - ETA: 40s - loss: 0.2482 - categorical_accuracy: 0.9247
37472/60000 [=================>............] - ETA: 40s - loss: 0.2481 - categorical_accuracy: 0.9247
37504/60000 [=================>............] - ETA: 40s - loss: 0.2480 - categorical_accuracy: 0.9248
37536/60000 [=================>............] - ETA: 39s - loss: 0.2478 - categorical_accuracy: 0.9249
37568/60000 [=================>............] - ETA: 39s - loss: 0.2476 - categorical_accuracy: 0.9249
37600/60000 [=================>............] - ETA: 39s - loss: 0.2475 - categorical_accuracy: 0.9249
37632/60000 [=================>............] - ETA: 39s - loss: 0.2473 - categorical_accuracy: 0.9250
37664/60000 [=================>............] - ETA: 39s - loss: 0.2472 - categorical_accuracy: 0.9250
37696/60000 [=================>............] - ETA: 39s - loss: 0.2470 - categorical_accuracy: 0.9251
37728/60000 [=================>............] - ETA: 39s - loss: 0.2468 - categorical_accuracy: 0.9251
37760/60000 [=================>............] - ETA: 39s - loss: 0.2466 - categorical_accuracy: 0.9252
37792/60000 [=================>............] - ETA: 39s - loss: 0.2465 - categorical_accuracy: 0.9252
37824/60000 [=================>............] - ETA: 39s - loss: 0.2464 - categorical_accuracy: 0.9252
37856/60000 [=================>............] - ETA: 39s - loss: 0.2464 - categorical_accuracy: 0.9252
37888/60000 [=================>............] - ETA: 39s - loss: 0.2463 - categorical_accuracy: 0.9253
37920/60000 [=================>............] - ETA: 39s - loss: 0.2462 - categorical_accuracy: 0.9253
37952/60000 [=================>............] - ETA: 39s - loss: 0.2460 - categorical_accuracy: 0.9253
37984/60000 [=================>............] - ETA: 39s - loss: 0.2459 - categorical_accuracy: 0.9253
38016/60000 [==================>...........] - ETA: 39s - loss: 0.2457 - categorical_accuracy: 0.9254
38048/60000 [==================>...........] - ETA: 39s - loss: 0.2456 - categorical_accuracy: 0.9254
38080/60000 [==================>...........] - ETA: 38s - loss: 0.2455 - categorical_accuracy: 0.9254
38112/60000 [==================>...........] - ETA: 38s - loss: 0.2455 - categorical_accuracy: 0.9255
38144/60000 [==================>...........] - ETA: 38s - loss: 0.2454 - categorical_accuracy: 0.9255
38176/60000 [==================>...........] - ETA: 38s - loss: 0.2453 - categorical_accuracy: 0.9255
38208/60000 [==================>...........] - ETA: 38s - loss: 0.2452 - categorical_accuracy: 0.9255
38240/60000 [==================>...........] - ETA: 38s - loss: 0.2452 - categorical_accuracy: 0.9255
38272/60000 [==================>...........] - ETA: 38s - loss: 0.2451 - categorical_accuracy: 0.9256
38304/60000 [==================>...........] - ETA: 38s - loss: 0.2449 - categorical_accuracy: 0.9256
38336/60000 [==================>...........] - ETA: 38s - loss: 0.2448 - categorical_accuracy: 0.9256
38368/60000 [==================>...........] - ETA: 38s - loss: 0.2446 - categorical_accuracy: 0.9257
38400/60000 [==================>...........] - ETA: 38s - loss: 0.2445 - categorical_accuracy: 0.9257
38432/60000 [==================>...........] - ETA: 38s - loss: 0.2443 - categorical_accuracy: 0.9258
38464/60000 [==================>...........] - ETA: 38s - loss: 0.2442 - categorical_accuracy: 0.9258
38496/60000 [==================>...........] - ETA: 38s - loss: 0.2443 - categorical_accuracy: 0.9258
38528/60000 [==================>...........] - ETA: 38s - loss: 0.2441 - categorical_accuracy: 0.9258
38560/60000 [==================>...........] - ETA: 38s - loss: 0.2440 - categorical_accuracy: 0.9259
38592/60000 [==================>...........] - ETA: 38s - loss: 0.2438 - categorical_accuracy: 0.9259
38624/60000 [==================>...........] - ETA: 38s - loss: 0.2436 - categorical_accuracy: 0.9260
38656/60000 [==================>...........] - ETA: 37s - loss: 0.2435 - categorical_accuracy: 0.9260
38688/60000 [==================>...........] - ETA: 37s - loss: 0.2434 - categorical_accuracy: 0.9260
38720/60000 [==================>...........] - ETA: 37s - loss: 0.2432 - categorical_accuracy: 0.9261
38752/60000 [==================>...........] - ETA: 37s - loss: 0.2430 - categorical_accuracy: 0.9261
38784/60000 [==================>...........] - ETA: 37s - loss: 0.2429 - categorical_accuracy: 0.9262
38816/60000 [==================>...........] - ETA: 37s - loss: 0.2428 - categorical_accuracy: 0.9262
38848/60000 [==================>...........] - ETA: 37s - loss: 0.2426 - categorical_accuracy: 0.9263
38880/60000 [==================>...........] - ETA: 37s - loss: 0.2426 - categorical_accuracy: 0.9262
38912/60000 [==================>...........] - ETA: 37s - loss: 0.2424 - categorical_accuracy: 0.9263
38944/60000 [==================>...........] - ETA: 37s - loss: 0.2422 - categorical_accuracy: 0.9263
38976/60000 [==================>...........] - ETA: 37s - loss: 0.2420 - categorical_accuracy: 0.9264
39008/60000 [==================>...........] - ETA: 37s - loss: 0.2419 - categorical_accuracy: 0.9264
39040/60000 [==================>...........] - ETA: 37s - loss: 0.2418 - categorical_accuracy: 0.9264
39072/60000 [==================>...........] - ETA: 37s - loss: 0.2416 - categorical_accuracy: 0.9265
39104/60000 [==================>...........] - ETA: 37s - loss: 0.2415 - categorical_accuracy: 0.9266
39136/60000 [==================>...........] - ETA: 37s - loss: 0.2413 - categorical_accuracy: 0.9266
39168/60000 [==================>...........] - ETA: 37s - loss: 0.2412 - categorical_accuracy: 0.9266
39200/60000 [==================>...........] - ETA: 36s - loss: 0.2411 - categorical_accuracy: 0.9267
39232/60000 [==================>...........] - ETA: 36s - loss: 0.2410 - categorical_accuracy: 0.9267
39264/60000 [==================>...........] - ETA: 36s - loss: 0.2408 - categorical_accuracy: 0.9268
39296/60000 [==================>...........] - ETA: 36s - loss: 0.2409 - categorical_accuracy: 0.9267
39328/60000 [==================>...........] - ETA: 36s - loss: 0.2408 - categorical_accuracy: 0.9267
39360/60000 [==================>...........] - ETA: 36s - loss: 0.2406 - categorical_accuracy: 0.9268
39392/60000 [==================>...........] - ETA: 36s - loss: 0.2406 - categorical_accuracy: 0.9268
39424/60000 [==================>...........] - ETA: 36s - loss: 0.2404 - categorical_accuracy: 0.9268
39456/60000 [==================>...........] - ETA: 36s - loss: 0.2402 - categorical_accuracy: 0.9269
39488/60000 [==================>...........] - ETA: 36s - loss: 0.2401 - categorical_accuracy: 0.9269
39520/60000 [==================>...........] - ETA: 36s - loss: 0.2402 - categorical_accuracy: 0.9269
39552/60000 [==================>...........] - ETA: 36s - loss: 0.2403 - categorical_accuracy: 0.9269
39584/60000 [==================>...........] - ETA: 36s - loss: 0.2402 - categorical_accuracy: 0.9269
39616/60000 [==================>...........] - ETA: 36s - loss: 0.2400 - categorical_accuracy: 0.9270
39648/60000 [==================>...........] - ETA: 36s - loss: 0.2400 - categorical_accuracy: 0.9270
39680/60000 [==================>...........] - ETA: 36s - loss: 0.2399 - categorical_accuracy: 0.9270
39712/60000 [==================>...........] - ETA: 36s - loss: 0.2397 - categorical_accuracy: 0.9271
39744/60000 [==================>...........] - ETA: 36s - loss: 0.2395 - categorical_accuracy: 0.9272
39776/60000 [==================>...........] - ETA: 35s - loss: 0.2394 - categorical_accuracy: 0.9272
39808/60000 [==================>...........] - ETA: 35s - loss: 0.2395 - categorical_accuracy: 0.9272
39840/60000 [==================>...........] - ETA: 35s - loss: 0.2396 - categorical_accuracy: 0.9272
39872/60000 [==================>...........] - ETA: 35s - loss: 0.2395 - categorical_accuracy: 0.9272
39904/60000 [==================>...........] - ETA: 35s - loss: 0.2394 - categorical_accuracy: 0.9273
39936/60000 [==================>...........] - ETA: 35s - loss: 0.2392 - categorical_accuracy: 0.9273
39968/60000 [==================>...........] - ETA: 35s - loss: 0.2391 - categorical_accuracy: 0.9273
40000/60000 [===================>..........] - ETA: 35s - loss: 0.2389 - categorical_accuracy: 0.9273
40032/60000 [===================>..........] - ETA: 35s - loss: 0.2388 - categorical_accuracy: 0.9274
40064/60000 [===================>..........] - ETA: 35s - loss: 0.2387 - categorical_accuracy: 0.9274
40096/60000 [===================>..........] - ETA: 35s - loss: 0.2385 - categorical_accuracy: 0.9275
40128/60000 [===================>..........] - ETA: 35s - loss: 0.2385 - categorical_accuracy: 0.9275
40160/60000 [===================>..........] - ETA: 35s - loss: 0.2384 - categorical_accuracy: 0.9275
40192/60000 [===================>..........] - ETA: 35s - loss: 0.2383 - categorical_accuracy: 0.9276
40224/60000 [===================>..........] - ETA: 35s - loss: 0.2381 - categorical_accuracy: 0.9276
40256/60000 [===================>..........] - ETA: 35s - loss: 0.2380 - categorical_accuracy: 0.9277
40288/60000 [===================>..........] - ETA: 35s - loss: 0.2378 - categorical_accuracy: 0.9277
40320/60000 [===================>..........] - ETA: 34s - loss: 0.2379 - categorical_accuracy: 0.9277
40352/60000 [===================>..........] - ETA: 34s - loss: 0.2377 - categorical_accuracy: 0.9278
40384/60000 [===================>..........] - ETA: 34s - loss: 0.2377 - categorical_accuracy: 0.9278
40416/60000 [===================>..........] - ETA: 34s - loss: 0.2376 - categorical_accuracy: 0.9278
40448/60000 [===================>..........] - ETA: 34s - loss: 0.2374 - categorical_accuracy: 0.9279
40480/60000 [===================>..........] - ETA: 34s - loss: 0.2373 - categorical_accuracy: 0.9279
40512/60000 [===================>..........] - ETA: 34s - loss: 0.2374 - categorical_accuracy: 0.9279
40544/60000 [===================>..........] - ETA: 34s - loss: 0.2372 - categorical_accuracy: 0.9280
40576/60000 [===================>..........] - ETA: 34s - loss: 0.2372 - categorical_accuracy: 0.9280
40608/60000 [===================>..........] - ETA: 34s - loss: 0.2371 - categorical_accuracy: 0.9280
40640/60000 [===================>..........] - ETA: 34s - loss: 0.2369 - categorical_accuracy: 0.9281
40672/60000 [===================>..........] - ETA: 34s - loss: 0.2368 - categorical_accuracy: 0.9281
40704/60000 [===================>..........] - ETA: 34s - loss: 0.2369 - categorical_accuracy: 0.9281
40736/60000 [===================>..........] - ETA: 34s - loss: 0.2369 - categorical_accuracy: 0.9281
40768/60000 [===================>..........] - ETA: 34s - loss: 0.2367 - categorical_accuracy: 0.9281
40800/60000 [===================>..........] - ETA: 34s - loss: 0.2366 - categorical_accuracy: 0.9281
40832/60000 [===================>..........] - ETA: 34s - loss: 0.2366 - categorical_accuracy: 0.9281
40864/60000 [===================>..........] - ETA: 34s - loss: 0.2364 - categorical_accuracy: 0.9282
40896/60000 [===================>..........] - ETA: 33s - loss: 0.2363 - categorical_accuracy: 0.9282
40928/60000 [===================>..........] - ETA: 33s - loss: 0.2363 - categorical_accuracy: 0.9281
40960/60000 [===================>..........] - ETA: 33s - loss: 0.2362 - categorical_accuracy: 0.9282
40992/60000 [===================>..........] - ETA: 33s - loss: 0.2360 - categorical_accuracy: 0.9282
41024/60000 [===================>..........] - ETA: 33s - loss: 0.2359 - categorical_accuracy: 0.9282
41056/60000 [===================>..........] - ETA: 33s - loss: 0.2358 - categorical_accuracy: 0.9283
41088/60000 [===================>..........] - ETA: 33s - loss: 0.2358 - categorical_accuracy: 0.9283
41120/60000 [===================>..........] - ETA: 33s - loss: 0.2357 - categorical_accuracy: 0.9282
41152/60000 [===================>..........] - ETA: 33s - loss: 0.2356 - categorical_accuracy: 0.9283
41184/60000 [===================>..........] - ETA: 33s - loss: 0.2354 - categorical_accuracy: 0.9283
41216/60000 [===================>..........] - ETA: 33s - loss: 0.2356 - categorical_accuracy: 0.9283
41248/60000 [===================>..........] - ETA: 33s - loss: 0.2355 - categorical_accuracy: 0.9283
41280/60000 [===================>..........] - ETA: 33s - loss: 0.2355 - categorical_accuracy: 0.9283
41312/60000 [===================>..........] - ETA: 33s - loss: 0.2354 - categorical_accuracy: 0.9283
41344/60000 [===================>..........] - ETA: 33s - loss: 0.2353 - categorical_accuracy: 0.9283
41376/60000 [===================>..........] - ETA: 33s - loss: 0.2352 - categorical_accuracy: 0.9284
41408/60000 [===================>..........] - ETA: 33s - loss: 0.2351 - categorical_accuracy: 0.9284
41440/60000 [===================>..........] - ETA: 33s - loss: 0.2349 - categorical_accuracy: 0.9285
41472/60000 [===================>..........] - ETA: 32s - loss: 0.2348 - categorical_accuracy: 0.9285
41504/60000 [===================>..........] - ETA: 32s - loss: 0.2346 - categorical_accuracy: 0.9285
41536/60000 [===================>..........] - ETA: 32s - loss: 0.2345 - categorical_accuracy: 0.9286
41568/60000 [===================>..........] - ETA: 32s - loss: 0.2343 - categorical_accuracy: 0.9286
41600/60000 [===================>..........] - ETA: 32s - loss: 0.2342 - categorical_accuracy: 0.9287
41632/60000 [===================>..........] - ETA: 32s - loss: 0.2341 - categorical_accuracy: 0.9287
41664/60000 [===================>..........] - ETA: 32s - loss: 0.2339 - categorical_accuracy: 0.9288
41696/60000 [===================>..........] - ETA: 32s - loss: 0.2338 - categorical_accuracy: 0.9288
41728/60000 [===================>..........] - ETA: 32s - loss: 0.2336 - categorical_accuracy: 0.9288
41760/60000 [===================>..........] - ETA: 32s - loss: 0.2336 - categorical_accuracy: 0.9288
41792/60000 [===================>..........] - ETA: 32s - loss: 0.2335 - categorical_accuracy: 0.9289
41824/60000 [===================>..........] - ETA: 32s - loss: 0.2333 - categorical_accuracy: 0.9289
41856/60000 [===================>..........] - ETA: 32s - loss: 0.2332 - categorical_accuracy: 0.9289
41888/60000 [===================>..........] - ETA: 32s - loss: 0.2330 - categorical_accuracy: 0.9290
41920/60000 [===================>..........] - ETA: 32s - loss: 0.2330 - categorical_accuracy: 0.9290
41952/60000 [===================>..........] - ETA: 32s - loss: 0.2331 - categorical_accuracy: 0.9290
41984/60000 [===================>..........] - ETA: 32s - loss: 0.2332 - categorical_accuracy: 0.9289
42016/60000 [====================>.........] - ETA: 31s - loss: 0.2333 - categorical_accuracy: 0.9289
42048/60000 [====================>.........] - ETA: 31s - loss: 0.2332 - categorical_accuracy: 0.9289
42080/60000 [====================>.........] - ETA: 31s - loss: 0.2332 - categorical_accuracy: 0.9289
42112/60000 [====================>.........] - ETA: 31s - loss: 0.2332 - categorical_accuracy: 0.9290
42144/60000 [====================>.........] - ETA: 31s - loss: 0.2330 - categorical_accuracy: 0.9290
42176/60000 [====================>.........] - ETA: 31s - loss: 0.2331 - categorical_accuracy: 0.9290
42208/60000 [====================>.........] - ETA: 31s - loss: 0.2330 - categorical_accuracy: 0.9290
42240/60000 [====================>.........] - ETA: 31s - loss: 0.2330 - categorical_accuracy: 0.9290
42272/60000 [====================>.........] - ETA: 31s - loss: 0.2328 - categorical_accuracy: 0.9291
42304/60000 [====================>.........] - ETA: 31s - loss: 0.2326 - categorical_accuracy: 0.9291
42336/60000 [====================>.........] - ETA: 31s - loss: 0.2325 - categorical_accuracy: 0.9291
42368/60000 [====================>.........] - ETA: 31s - loss: 0.2324 - categorical_accuracy: 0.9292
42400/60000 [====================>.........] - ETA: 31s - loss: 0.2323 - categorical_accuracy: 0.9292
42432/60000 [====================>.........] - ETA: 31s - loss: 0.2322 - categorical_accuracy: 0.9293
42464/60000 [====================>.........] - ETA: 31s - loss: 0.2321 - categorical_accuracy: 0.9293
42496/60000 [====================>.........] - ETA: 31s - loss: 0.2321 - categorical_accuracy: 0.9292
42528/60000 [====================>.........] - ETA: 31s - loss: 0.2320 - categorical_accuracy: 0.9293
42560/60000 [====================>.........] - ETA: 30s - loss: 0.2318 - categorical_accuracy: 0.9293
42592/60000 [====================>.........] - ETA: 30s - loss: 0.2318 - categorical_accuracy: 0.9294
42624/60000 [====================>.........] - ETA: 30s - loss: 0.2317 - categorical_accuracy: 0.9294
42656/60000 [====================>.........] - ETA: 30s - loss: 0.2316 - categorical_accuracy: 0.9295
42688/60000 [====================>.........] - ETA: 30s - loss: 0.2314 - categorical_accuracy: 0.9295
42720/60000 [====================>.........] - ETA: 30s - loss: 0.2313 - categorical_accuracy: 0.9296
42752/60000 [====================>.........] - ETA: 30s - loss: 0.2313 - categorical_accuracy: 0.9296
42784/60000 [====================>.........] - ETA: 30s - loss: 0.2313 - categorical_accuracy: 0.9296
42816/60000 [====================>.........] - ETA: 30s - loss: 0.2312 - categorical_accuracy: 0.9296
42848/60000 [====================>.........] - ETA: 30s - loss: 0.2311 - categorical_accuracy: 0.9297
42880/60000 [====================>.........] - ETA: 30s - loss: 0.2311 - categorical_accuracy: 0.9297
42912/60000 [====================>.........] - ETA: 30s - loss: 0.2310 - categorical_accuracy: 0.9298
42944/60000 [====================>.........] - ETA: 30s - loss: 0.2309 - categorical_accuracy: 0.9298
42976/60000 [====================>.........] - ETA: 30s - loss: 0.2308 - categorical_accuracy: 0.9298
43008/60000 [====================>.........] - ETA: 30s - loss: 0.2307 - categorical_accuracy: 0.9298
43040/60000 [====================>.........] - ETA: 30s - loss: 0.2306 - categorical_accuracy: 0.9298
43072/60000 [====================>.........] - ETA: 30s - loss: 0.2305 - categorical_accuracy: 0.9299
43104/60000 [====================>.........] - ETA: 30s - loss: 0.2305 - categorical_accuracy: 0.9299
43136/60000 [====================>.........] - ETA: 29s - loss: 0.2303 - categorical_accuracy: 0.9299
43168/60000 [====================>.........] - ETA: 29s - loss: 0.2303 - categorical_accuracy: 0.9299
43200/60000 [====================>.........] - ETA: 29s - loss: 0.2302 - categorical_accuracy: 0.9300
43232/60000 [====================>.........] - ETA: 29s - loss: 0.2301 - categorical_accuracy: 0.9300
43264/60000 [====================>.........] - ETA: 29s - loss: 0.2300 - categorical_accuracy: 0.9300
43296/60000 [====================>.........] - ETA: 29s - loss: 0.2299 - categorical_accuracy: 0.9301
43328/60000 [====================>.........] - ETA: 29s - loss: 0.2298 - categorical_accuracy: 0.9301
43360/60000 [====================>.........] - ETA: 29s - loss: 0.2296 - categorical_accuracy: 0.9302
43392/60000 [====================>.........] - ETA: 29s - loss: 0.2294 - categorical_accuracy: 0.9302
43424/60000 [====================>.........] - ETA: 29s - loss: 0.2293 - categorical_accuracy: 0.9303
43456/60000 [====================>.........] - ETA: 29s - loss: 0.2291 - categorical_accuracy: 0.9303
43488/60000 [====================>.........] - ETA: 29s - loss: 0.2291 - categorical_accuracy: 0.9303
43520/60000 [====================>.........] - ETA: 29s - loss: 0.2290 - categorical_accuracy: 0.9304
43552/60000 [====================>.........] - ETA: 29s - loss: 0.2289 - categorical_accuracy: 0.9304
43584/60000 [====================>.........] - ETA: 29s - loss: 0.2288 - categorical_accuracy: 0.9304
43616/60000 [====================>.........] - ETA: 29s - loss: 0.2288 - categorical_accuracy: 0.9304
43648/60000 [====================>.........] - ETA: 29s - loss: 0.2288 - categorical_accuracy: 0.9304
43680/60000 [====================>.........] - ETA: 29s - loss: 0.2287 - categorical_accuracy: 0.9304
43712/60000 [====================>.........] - ETA: 28s - loss: 0.2286 - categorical_accuracy: 0.9305
43744/60000 [====================>.........] - ETA: 28s - loss: 0.2285 - categorical_accuracy: 0.9305
43776/60000 [====================>.........] - ETA: 28s - loss: 0.2283 - categorical_accuracy: 0.9305
43808/60000 [====================>.........] - ETA: 28s - loss: 0.2282 - categorical_accuracy: 0.9306
43840/60000 [====================>.........] - ETA: 28s - loss: 0.2281 - categorical_accuracy: 0.9306
43872/60000 [====================>.........] - ETA: 28s - loss: 0.2281 - categorical_accuracy: 0.9306
43904/60000 [====================>.........] - ETA: 28s - loss: 0.2280 - categorical_accuracy: 0.9306
43936/60000 [====================>.........] - ETA: 28s - loss: 0.2278 - categorical_accuracy: 0.9307
43968/60000 [====================>.........] - ETA: 28s - loss: 0.2277 - categorical_accuracy: 0.9307
44000/60000 [=====================>........] - ETA: 28s - loss: 0.2277 - categorical_accuracy: 0.9307
44032/60000 [=====================>........] - ETA: 28s - loss: 0.2277 - categorical_accuracy: 0.9308
44064/60000 [=====================>........] - ETA: 28s - loss: 0.2276 - categorical_accuracy: 0.9308
44096/60000 [=====================>........] - ETA: 28s - loss: 0.2275 - categorical_accuracy: 0.9308
44128/60000 [=====================>........] - ETA: 28s - loss: 0.2275 - categorical_accuracy: 0.9308
44160/60000 [=====================>........] - ETA: 28s - loss: 0.2274 - categorical_accuracy: 0.9308
44192/60000 [=====================>........] - ETA: 28s - loss: 0.2273 - categorical_accuracy: 0.9308
44224/60000 [=====================>........] - ETA: 28s - loss: 0.2272 - categorical_accuracy: 0.9309
44256/60000 [=====================>........] - ETA: 28s - loss: 0.2271 - categorical_accuracy: 0.9309
44288/60000 [=====================>........] - ETA: 27s - loss: 0.2270 - categorical_accuracy: 0.9309
44320/60000 [=====================>........] - ETA: 27s - loss: 0.2268 - categorical_accuracy: 0.9309
44352/60000 [=====================>........] - ETA: 27s - loss: 0.2267 - categorical_accuracy: 0.9310
44384/60000 [=====================>........] - ETA: 27s - loss: 0.2267 - categorical_accuracy: 0.9310
44416/60000 [=====================>........] - ETA: 27s - loss: 0.2268 - categorical_accuracy: 0.9310
44448/60000 [=====================>........] - ETA: 27s - loss: 0.2268 - categorical_accuracy: 0.9310
44480/60000 [=====================>........] - ETA: 27s - loss: 0.2268 - categorical_accuracy: 0.9310
44512/60000 [=====================>........] - ETA: 27s - loss: 0.2267 - categorical_accuracy: 0.9311
44544/60000 [=====================>........] - ETA: 27s - loss: 0.2266 - categorical_accuracy: 0.9311
44576/60000 [=====================>........] - ETA: 27s - loss: 0.2265 - categorical_accuracy: 0.9312
44608/60000 [=====================>........] - ETA: 27s - loss: 0.2263 - categorical_accuracy: 0.9312
44640/60000 [=====================>........] - ETA: 27s - loss: 0.2263 - categorical_accuracy: 0.9312
44672/60000 [=====================>........] - ETA: 27s - loss: 0.2261 - categorical_accuracy: 0.9312
44704/60000 [=====================>........] - ETA: 27s - loss: 0.2260 - categorical_accuracy: 0.9313
44736/60000 [=====================>........] - ETA: 27s - loss: 0.2259 - categorical_accuracy: 0.9313
44768/60000 [=====================>........] - ETA: 27s - loss: 0.2258 - categorical_accuracy: 0.9314
44800/60000 [=====================>........] - ETA: 27s - loss: 0.2256 - categorical_accuracy: 0.9314
44832/60000 [=====================>........] - ETA: 26s - loss: 0.2255 - categorical_accuracy: 0.9314
44864/60000 [=====================>........] - ETA: 26s - loss: 0.2254 - categorical_accuracy: 0.9314
44896/60000 [=====================>........] - ETA: 26s - loss: 0.2254 - categorical_accuracy: 0.9315
44928/60000 [=====================>........] - ETA: 26s - loss: 0.2253 - categorical_accuracy: 0.9315
44960/60000 [=====================>........] - ETA: 26s - loss: 0.2251 - categorical_accuracy: 0.9316
44992/60000 [=====================>........] - ETA: 26s - loss: 0.2253 - categorical_accuracy: 0.9315
45024/60000 [=====================>........] - ETA: 26s - loss: 0.2252 - categorical_accuracy: 0.9316
45056/60000 [=====================>........] - ETA: 26s - loss: 0.2251 - categorical_accuracy: 0.9316
45088/60000 [=====================>........] - ETA: 26s - loss: 0.2250 - categorical_accuracy: 0.9316
45120/60000 [=====================>........] - ETA: 26s - loss: 0.2249 - categorical_accuracy: 0.9317
45152/60000 [=====================>........] - ETA: 26s - loss: 0.2247 - categorical_accuracy: 0.9317
45184/60000 [=====================>........] - ETA: 26s - loss: 0.2246 - categorical_accuracy: 0.9318
45216/60000 [=====================>........] - ETA: 26s - loss: 0.2246 - categorical_accuracy: 0.9318
45248/60000 [=====================>........] - ETA: 26s - loss: 0.2245 - categorical_accuracy: 0.9318
45280/60000 [=====================>........] - ETA: 26s - loss: 0.2244 - categorical_accuracy: 0.9318
45312/60000 [=====================>........] - ETA: 26s - loss: 0.2244 - categorical_accuracy: 0.9319
45344/60000 [=====================>........] - ETA: 26s - loss: 0.2243 - categorical_accuracy: 0.9319
45376/60000 [=====================>........] - ETA: 26s - loss: 0.2242 - categorical_accuracy: 0.9319
45408/60000 [=====================>........] - ETA: 25s - loss: 0.2241 - categorical_accuracy: 0.9320
45440/60000 [=====================>........] - ETA: 25s - loss: 0.2240 - categorical_accuracy: 0.9320
45472/60000 [=====================>........] - ETA: 25s - loss: 0.2239 - categorical_accuracy: 0.9320
45504/60000 [=====================>........] - ETA: 25s - loss: 0.2237 - categorical_accuracy: 0.9321
45536/60000 [=====================>........] - ETA: 25s - loss: 0.2236 - categorical_accuracy: 0.9321
45568/60000 [=====================>........] - ETA: 25s - loss: 0.2235 - categorical_accuracy: 0.9321
45600/60000 [=====================>........] - ETA: 25s - loss: 0.2234 - categorical_accuracy: 0.9322
45632/60000 [=====================>........] - ETA: 25s - loss: 0.2234 - categorical_accuracy: 0.9322
45664/60000 [=====================>........] - ETA: 25s - loss: 0.2234 - categorical_accuracy: 0.9322
45696/60000 [=====================>........] - ETA: 25s - loss: 0.2232 - categorical_accuracy: 0.9322
45728/60000 [=====================>........] - ETA: 25s - loss: 0.2232 - categorical_accuracy: 0.9323
45760/60000 [=====================>........] - ETA: 25s - loss: 0.2231 - categorical_accuracy: 0.9323
45792/60000 [=====================>........] - ETA: 25s - loss: 0.2229 - categorical_accuracy: 0.9324
45824/60000 [=====================>........] - ETA: 25s - loss: 0.2229 - categorical_accuracy: 0.9324
45856/60000 [=====================>........] - ETA: 25s - loss: 0.2228 - categorical_accuracy: 0.9324
45888/60000 [=====================>........] - ETA: 25s - loss: 0.2227 - categorical_accuracy: 0.9325
45920/60000 [=====================>........] - ETA: 25s - loss: 0.2226 - categorical_accuracy: 0.9325
45952/60000 [=====================>........] - ETA: 24s - loss: 0.2225 - categorical_accuracy: 0.9325
45984/60000 [=====================>........] - ETA: 24s - loss: 0.2224 - categorical_accuracy: 0.9325
46016/60000 [======================>.......] - ETA: 24s - loss: 0.2223 - categorical_accuracy: 0.9326
46048/60000 [======================>.......] - ETA: 24s - loss: 0.2222 - categorical_accuracy: 0.9326
46080/60000 [======================>.......] - ETA: 24s - loss: 0.2220 - categorical_accuracy: 0.9327
46112/60000 [======================>.......] - ETA: 24s - loss: 0.2220 - categorical_accuracy: 0.9327
46144/60000 [======================>.......] - ETA: 24s - loss: 0.2218 - categorical_accuracy: 0.9327
46176/60000 [======================>.......] - ETA: 24s - loss: 0.2218 - categorical_accuracy: 0.9327
46208/60000 [======================>.......] - ETA: 24s - loss: 0.2217 - categorical_accuracy: 0.9328
46240/60000 [======================>.......] - ETA: 24s - loss: 0.2215 - categorical_accuracy: 0.9328
46272/60000 [======================>.......] - ETA: 24s - loss: 0.2214 - categorical_accuracy: 0.9328
46304/60000 [======================>.......] - ETA: 24s - loss: 0.2213 - categorical_accuracy: 0.9329
46336/60000 [======================>.......] - ETA: 24s - loss: 0.2213 - categorical_accuracy: 0.9329
46368/60000 [======================>.......] - ETA: 24s - loss: 0.2214 - categorical_accuracy: 0.9329
46400/60000 [======================>.......] - ETA: 24s - loss: 0.2213 - categorical_accuracy: 0.9329
46432/60000 [======================>.......] - ETA: 24s - loss: 0.2213 - categorical_accuracy: 0.9329
46464/60000 [======================>.......] - ETA: 24s - loss: 0.2213 - categorical_accuracy: 0.9329
46496/60000 [======================>.......] - ETA: 24s - loss: 0.2212 - categorical_accuracy: 0.9330
46528/60000 [======================>.......] - ETA: 23s - loss: 0.2211 - categorical_accuracy: 0.9330
46560/60000 [======================>.......] - ETA: 23s - loss: 0.2210 - categorical_accuracy: 0.9330
46592/60000 [======================>.......] - ETA: 23s - loss: 0.2208 - categorical_accuracy: 0.9330
46624/60000 [======================>.......] - ETA: 23s - loss: 0.2207 - categorical_accuracy: 0.9331
46656/60000 [======================>.......] - ETA: 23s - loss: 0.2206 - categorical_accuracy: 0.9331
46688/60000 [======================>.......] - ETA: 23s - loss: 0.2205 - categorical_accuracy: 0.9332
46720/60000 [======================>.......] - ETA: 23s - loss: 0.2203 - categorical_accuracy: 0.9332
46752/60000 [======================>.......] - ETA: 23s - loss: 0.2202 - categorical_accuracy: 0.9332
46784/60000 [======================>.......] - ETA: 23s - loss: 0.2201 - categorical_accuracy: 0.9333
46816/60000 [======================>.......] - ETA: 23s - loss: 0.2201 - categorical_accuracy: 0.9333
46848/60000 [======================>.......] - ETA: 23s - loss: 0.2200 - categorical_accuracy: 0.9333
46880/60000 [======================>.......] - ETA: 23s - loss: 0.2199 - categorical_accuracy: 0.9333
46912/60000 [======================>.......] - ETA: 23s - loss: 0.2198 - categorical_accuracy: 0.9333
46944/60000 [======================>.......] - ETA: 23s - loss: 0.2197 - categorical_accuracy: 0.9333
46976/60000 [======================>.......] - ETA: 23s - loss: 0.2197 - categorical_accuracy: 0.9333
47008/60000 [======================>.......] - ETA: 23s - loss: 0.2195 - categorical_accuracy: 0.9334
47040/60000 [======================>.......] - ETA: 23s - loss: 0.2194 - categorical_accuracy: 0.9334
47072/60000 [======================>.......] - ETA: 22s - loss: 0.2193 - categorical_accuracy: 0.9334
47104/60000 [======================>.......] - ETA: 22s - loss: 0.2192 - categorical_accuracy: 0.9335
47136/60000 [======================>.......] - ETA: 22s - loss: 0.2190 - categorical_accuracy: 0.9335
47168/60000 [======================>.......] - ETA: 22s - loss: 0.2189 - categorical_accuracy: 0.9336
47200/60000 [======================>.......] - ETA: 22s - loss: 0.2189 - categorical_accuracy: 0.9336
47232/60000 [======================>.......] - ETA: 22s - loss: 0.2188 - categorical_accuracy: 0.9336
47264/60000 [======================>.......] - ETA: 22s - loss: 0.2187 - categorical_accuracy: 0.9336
47296/60000 [======================>.......] - ETA: 22s - loss: 0.2186 - categorical_accuracy: 0.9337
47328/60000 [======================>.......] - ETA: 22s - loss: 0.2185 - categorical_accuracy: 0.9337
47360/60000 [======================>.......] - ETA: 22s - loss: 0.2185 - categorical_accuracy: 0.9337
47392/60000 [======================>.......] - ETA: 22s - loss: 0.2183 - categorical_accuracy: 0.9337
47424/60000 [======================>.......] - ETA: 22s - loss: 0.2182 - categorical_accuracy: 0.9338
47456/60000 [======================>.......] - ETA: 22s - loss: 0.2182 - categorical_accuracy: 0.9338
47488/60000 [======================>.......] - ETA: 22s - loss: 0.2181 - categorical_accuracy: 0.9338
47520/60000 [======================>.......] - ETA: 22s - loss: 0.2181 - categorical_accuracy: 0.9338
47552/60000 [======================>.......] - ETA: 22s - loss: 0.2181 - categorical_accuracy: 0.9338
47584/60000 [======================>.......] - ETA: 22s - loss: 0.2180 - categorical_accuracy: 0.9339
47616/60000 [======================>.......] - ETA: 22s - loss: 0.2179 - categorical_accuracy: 0.9339
47648/60000 [======================>.......] - ETA: 21s - loss: 0.2178 - categorical_accuracy: 0.9339
47680/60000 [======================>.......] - ETA: 21s - loss: 0.2177 - categorical_accuracy: 0.9340
47712/60000 [======================>.......] - ETA: 21s - loss: 0.2176 - categorical_accuracy: 0.9340
47744/60000 [======================>.......] - ETA: 21s - loss: 0.2175 - categorical_accuracy: 0.9340
47776/60000 [======================>.......] - ETA: 21s - loss: 0.2175 - categorical_accuracy: 0.9340
47808/60000 [======================>.......] - ETA: 21s - loss: 0.2175 - categorical_accuracy: 0.9340
47840/60000 [======================>.......] - ETA: 21s - loss: 0.2174 - categorical_accuracy: 0.9341
47872/60000 [======================>.......] - ETA: 21s - loss: 0.2173 - categorical_accuracy: 0.9341
47904/60000 [======================>.......] - ETA: 21s - loss: 0.2172 - categorical_accuracy: 0.9341
47936/60000 [======================>.......] - ETA: 21s - loss: 0.2171 - categorical_accuracy: 0.9341
47968/60000 [======================>.......] - ETA: 21s - loss: 0.2170 - categorical_accuracy: 0.9342
48000/60000 [=======================>......] - ETA: 21s - loss: 0.2169 - categorical_accuracy: 0.9342
48032/60000 [=======================>......] - ETA: 21s - loss: 0.2169 - categorical_accuracy: 0.9342
48064/60000 [=======================>......] - ETA: 21s - loss: 0.2168 - categorical_accuracy: 0.9342
48096/60000 [=======================>......] - ETA: 21s - loss: 0.2168 - categorical_accuracy: 0.9342
48128/60000 [=======================>......] - ETA: 21s - loss: 0.2167 - categorical_accuracy: 0.9342
48160/60000 [=======================>......] - ETA: 21s - loss: 0.2166 - categorical_accuracy: 0.9343
48192/60000 [=======================>......] - ETA: 21s - loss: 0.2166 - categorical_accuracy: 0.9342
48224/60000 [=======================>......] - ETA: 20s - loss: 0.2165 - categorical_accuracy: 0.9343
48256/60000 [=======================>......] - ETA: 20s - loss: 0.2165 - categorical_accuracy: 0.9343
48288/60000 [=======================>......] - ETA: 20s - loss: 0.2165 - categorical_accuracy: 0.9342
48320/60000 [=======================>......] - ETA: 20s - loss: 0.2165 - categorical_accuracy: 0.9342
48352/60000 [=======================>......] - ETA: 20s - loss: 0.2163 - categorical_accuracy: 0.9343
48384/60000 [=======================>......] - ETA: 20s - loss: 0.2164 - categorical_accuracy: 0.9343
48416/60000 [=======================>......] - ETA: 20s - loss: 0.2163 - categorical_accuracy: 0.9343
48448/60000 [=======================>......] - ETA: 20s - loss: 0.2162 - categorical_accuracy: 0.9343
48480/60000 [=======================>......] - ETA: 20s - loss: 0.2161 - categorical_accuracy: 0.9343
48512/60000 [=======================>......] - ETA: 20s - loss: 0.2160 - categorical_accuracy: 0.9343
48544/60000 [=======================>......] - ETA: 20s - loss: 0.2159 - categorical_accuracy: 0.9344
48576/60000 [=======================>......] - ETA: 20s - loss: 0.2158 - categorical_accuracy: 0.9344
48608/60000 [=======================>......] - ETA: 20s - loss: 0.2158 - categorical_accuracy: 0.9344
48640/60000 [=======================>......] - ETA: 20s - loss: 0.2159 - categorical_accuracy: 0.9344
48672/60000 [=======================>......] - ETA: 20s - loss: 0.2159 - categorical_accuracy: 0.9344
48704/60000 [=======================>......] - ETA: 20s - loss: 0.2159 - categorical_accuracy: 0.9344
48736/60000 [=======================>......] - ETA: 20s - loss: 0.2158 - categorical_accuracy: 0.9345
48768/60000 [=======================>......] - ETA: 19s - loss: 0.2156 - categorical_accuracy: 0.9345
48800/60000 [=======================>......] - ETA: 19s - loss: 0.2155 - categorical_accuracy: 0.9346
48832/60000 [=======================>......] - ETA: 19s - loss: 0.2154 - categorical_accuracy: 0.9346
48864/60000 [=======================>......] - ETA: 19s - loss: 0.2153 - categorical_accuracy: 0.9346
48896/60000 [=======================>......] - ETA: 19s - loss: 0.2155 - categorical_accuracy: 0.9346
48928/60000 [=======================>......] - ETA: 19s - loss: 0.2154 - categorical_accuracy: 0.9346
48960/60000 [=======================>......] - ETA: 19s - loss: 0.2153 - categorical_accuracy: 0.9346
48992/60000 [=======================>......] - ETA: 19s - loss: 0.2153 - categorical_accuracy: 0.9346
49024/60000 [=======================>......] - ETA: 19s - loss: 0.2152 - categorical_accuracy: 0.9347
49056/60000 [=======================>......] - ETA: 19s - loss: 0.2151 - categorical_accuracy: 0.9347
49088/60000 [=======================>......] - ETA: 19s - loss: 0.2151 - categorical_accuracy: 0.9347
49120/60000 [=======================>......] - ETA: 19s - loss: 0.2150 - categorical_accuracy: 0.9348
49152/60000 [=======================>......] - ETA: 19s - loss: 0.2150 - categorical_accuracy: 0.9348
49184/60000 [=======================>......] - ETA: 19s - loss: 0.2150 - categorical_accuracy: 0.9348
49216/60000 [=======================>......] - ETA: 19s - loss: 0.2148 - categorical_accuracy: 0.9348
49248/60000 [=======================>......] - ETA: 19s - loss: 0.2148 - categorical_accuracy: 0.9348
49280/60000 [=======================>......] - ETA: 19s - loss: 0.2147 - categorical_accuracy: 0.9348
49312/60000 [=======================>......] - ETA: 19s - loss: 0.2146 - categorical_accuracy: 0.9349
49344/60000 [=======================>......] - ETA: 18s - loss: 0.2145 - categorical_accuracy: 0.9349
49376/60000 [=======================>......] - ETA: 18s - loss: 0.2144 - categorical_accuracy: 0.9349
49408/60000 [=======================>......] - ETA: 18s - loss: 0.2143 - categorical_accuracy: 0.9350
49440/60000 [=======================>......] - ETA: 18s - loss: 0.2143 - categorical_accuracy: 0.9350
49472/60000 [=======================>......] - ETA: 18s - loss: 0.2143 - categorical_accuracy: 0.9350
49504/60000 [=======================>......] - ETA: 18s - loss: 0.2143 - categorical_accuracy: 0.9350
49536/60000 [=======================>......] - ETA: 18s - loss: 0.2142 - categorical_accuracy: 0.9350
49568/60000 [=======================>......] - ETA: 18s - loss: 0.2140 - categorical_accuracy: 0.9351
49600/60000 [=======================>......] - ETA: 18s - loss: 0.2140 - categorical_accuracy: 0.9350
49632/60000 [=======================>......] - ETA: 18s - loss: 0.2139 - categorical_accuracy: 0.9351
49664/60000 [=======================>......] - ETA: 18s - loss: 0.2139 - categorical_accuracy: 0.9351
49696/60000 [=======================>......] - ETA: 18s - loss: 0.2138 - categorical_accuracy: 0.9351
49728/60000 [=======================>......] - ETA: 18s - loss: 0.2138 - categorical_accuracy: 0.9351
49760/60000 [=======================>......] - ETA: 18s - loss: 0.2137 - categorical_accuracy: 0.9351
49792/60000 [=======================>......] - ETA: 18s - loss: 0.2136 - categorical_accuracy: 0.9351
49824/60000 [=======================>......] - ETA: 18s - loss: 0.2135 - categorical_accuracy: 0.9352
49856/60000 [=======================>......] - ETA: 18s - loss: 0.2134 - categorical_accuracy: 0.9352
49888/60000 [=======================>......] - ETA: 17s - loss: 0.2134 - categorical_accuracy: 0.9352
49920/60000 [=======================>......] - ETA: 17s - loss: 0.2133 - categorical_accuracy: 0.9353
49952/60000 [=======================>......] - ETA: 17s - loss: 0.2132 - categorical_accuracy: 0.9353
49984/60000 [=======================>......] - ETA: 17s - loss: 0.2131 - categorical_accuracy: 0.9353
50016/60000 [========================>.....] - ETA: 17s - loss: 0.2130 - categorical_accuracy: 0.9353
50048/60000 [========================>.....] - ETA: 17s - loss: 0.2129 - categorical_accuracy: 0.9354
50080/60000 [========================>.....] - ETA: 17s - loss: 0.2128 - categorical_accuracy: 0.9354
50112/60000 [========================>.....] - ETA: 17s - loss: 0.2127 - categorical_accuracy: 0.9354
50144/60000 [========================>.....] - ETA: 17s - loss: 0.2127 - categorical_accuracy: 0.9354
50176/60000 [========================>.....] - ETA: 17s - loss: 0.2127 - categorical_accuracy: 0.9354
50208/60000 [========================>.....] - ETA: 17s - loss: 0.2127 - categorical_accuracy: 0.9354
50240/60000 [========================>.....] - ETA: 17s - loss: 0.2125 - categorical_accuracy: 0.9354
50272/60000 [========================>.....] - ETA: 17s - loss: 0.2125 - categorical_accuracy: 0.9355
50304/60000 [========================>.....] - ETA: 17s - loss: 0.2124 - categorical_accuracy: 0.9355
50336/60000 [========================>.....] - ETA: 17s - loss: 0.2123 - categorical_accuracy: 0.9356
50368/60000 [========================>.....] - ETA: 17s - loss: 0.2121 - categorical_accuracy: 0.9356
50400/60000 [========================>.....] - ETA: 17s - loss: 0.2121 - categorical_accuracy: 0.9356
50432/60000 [========================>.....] - ETA: 17s - loss: 0.2120 - categorical_accuracy: 0.9357
50464/60000 [========================>.....] - ETA: 16s - loss: 0.2119 - categorical_accuracy: 0.9357
50496/60000 [========================>.....] - ETA: 16s - loss: 0.2119 - categorical_accuracy: 0.9357
50528/60000 [========================>.....] - ETA: 16s - loss: 0.2119 - categorical_accuracy: 0.9357
50560/60000 [========================>.....] - ETA: 16s - loss: 0.2119 - categorical_accuracy: 0.9357
50592/60000 [========================>.....] - ETA: 16s - loss: 0.2119 - categorical_accuracy: 0.9357
50624/60000 [========================>.....] - ETA: 16s - loss: 0.2118 - categorical_accuracy: 0.9357
50656/60000 [========================>.....] - ETA: 16s - loss: 0.2119 - categorical_accuracy: 0.9357
50688/60000 [========================>.....] - ETA: 16s - loss: 0.2119 - categorical_accuracy: 0.9357
50720/60000 [========================>.....] - ETA: 16s - loss: 0.2118 - categorical_accuracy: 0.9358
50752/60000 [========================>.....] - ETA: 16s - loss: 0.2118 - categorical_accuracy: 0.9357
50784/60000 [========================>.....] - ETA: 16s - loss: 0.2118 - categorical_accuracy: 0.9357
50816/60000 [========================>.....] - ETA: 16s - loss: 0.2117 - categorical_accuracy: 0.9358
50848/60000 [========================>.....] - ETA: 16s - loss: 0.2116 - categorical_accuracy: 0.9358
50880/60000 [========================>.....] - ETA: 16s - loss: 0.2115 - categorical_accuracy: 0.9358
50912/60000 [========================>.....] - ETA: 16s - loss: 0.2114 - categorical_accuracy: 0.9359
50944/60000 [========================>.....] - ETA: 16s - loss: 0.2113 - categorical_accuracy: 0.9359
50976/60000 [========================>.....] - ETA: 16s - loss: 0.2112 - categorical_accuracy: 0.9360
51008/60000 [========================>.....] - ETA: 15s - loss: 0.2111 - categorical_accuracy: 0.9360
51040/60000 [========================>.....] - ETA: 15s - loss: 0.2110 - categorical_accuracy: 0.9360
51072/60000 [========================>.....] - ETA: 15s - loss: 0.2111 - categorical_accuracy: 0.9360
51104/60000 [========================>.....] - ETA: 15s - loss: 0.2109 - categorical_accuracy: 0.9360
51136/60000 [========================>.....] - ETA: 15s - loss: 0.2108 - categorical_accuracy: 0.9361
51168/60000 [========================>.....] - ETA: 15s - loss: 0.2107 - categorical_accuracy: 0.9361
51200/60000 [========================>.....] - ETA: 15s - loss: 0.2107 - categorical_accuracy: 0.9361
51232/60000 [========================>.....] - ETA: 15s - loss: 0.2107 - categorical_accuracy: 0.9361
51264/60000 [========================>.....] - ETA: 15s - loss: 0.2107 - categorical_accuracy: 0.9361
51296/60000 [========================>.....] - ETA: 15s - loss: 0.2106 - categorical_accuracy: 0.9361
51328/60000 [========================>.....] - ETA: 15s - loss: 0.2104 - categorical_accuracy: 0.9362
51360/60000 [========================>.....] - ETA: 15s - loss: 0.2105 - categorical_accuracy: 0.9361
51392/60000 [========================>.....] - ETA: 15s - loss: 0.2104 - categorical_accuracy: 0.9362
51424/60000 [========================>.....] - ETA: 15s - loss: 0.2104 - categorical_accuracy: 0.9362
51456/60000 [========================>.....] - ETA: 15s - loss: 0.2105 - categorical_accuracy: 0.9362
51488/60000 [========================>.....] - ETA: 15s - loss: 0.2104 - categorical_accuracy: 0.9362
51520/60000 [========================>.....] - ETA: 15s - loss: 0.2104 - categorical_accuracy: 0.9362
51552/60000 [========================>.....] - ETA: 15s - loss: 0.2103 - categorical_accuracy: 0.9363
51584/60000 [========================>.....] - ETA: 14s - loss: 0.2103 - categorical_accuracy: 0.9363
51616/60000 [========================>.....] - ETA: 14s - loss: 0.2102 - categorical_accuracy: 0.9363
51648/60000 [========================>.....] - ETA: 14s - loss: 0.2101 - categorical_accuracy: 0.9363
51680/60000 [========================>.....] - ETA: 14s - loss: 0.2102 - categorical_accuracy: 0.9363
51712/60000 [========================>.....] - ETA: 14s - loss: 0.2101 - categorical_accuracy: 0.9363
51744/60000 [========================>.....] - ETA: 14s - loss: 0.2101 - categorical_accuracy: 0.9363
51776/60000 [========================>.....] - ETA: 14s - loss: 0.2100 - categorical_accuracy: 0.9363
51808/60000 [========================>.....] - ETA: 14s - loss: 0.2099 - categorical_accuracy: 0.9364
51840/60000 [========================>.....] - ETA: 14s - loss: 0.2101 - categorical_accuracy: 0.9364
51872/60000 [========================>.....] - ETA: 14s - loss: 0.2100 - categorical_accuracy: 0.9364
51904/60000 [========================>.....] - ETA: 14s - loss: 0.2100 - categorical_accuracy: 0.9364
51936/60000 [========================>.....] - ETA: 14s - loss: 0.2099 - categorical_accuracy: 0.9364
51968/60000 [========================>.....] - ETA: 14s - loss: 0.2098 - categorical_accuracy: 0.9364
52000/60000 [=========================>....] - ETA: 14s - loss: 0.2097 - categorical_accuracy: 0.9364
52032/60000 [=========================>....] - ETA: 14s - loss: 0.2096 - categorical_accuracy: 0.9365
52064/60000 [=========================>....] - ETA: 14s - loss: 0.2096 - categorical_accuracy: 0.9365
52096/60000 [=========================>....] - ETA: 14s - loss: 0.2095 - categorical_accuracy: 0.9365
52128/60000 [=========================>....] - ETA: 14s - loss: 0.2094 - categorical_accuracy: 0.9365
52160/60000 [=========================>....] - ETA: 13s - loss: 0.2093 - categorical_accuracy: 0.9365
52192/60000 [=========================>....] - ETA: 13s - loss: 0.2093 - categorical_accuracy: 0.9365
52224/60000 [=========================>....] - ETA: 13s - loss: 0.2092 - categorical_accuracy: 0.9366
52256/60000 [=========================>....] - ETA: 13s - loss: 0.2092 - categorical_accuracy: 0.9366
52288/60000 [=========================>....] - ETA: 13s - loss: 0.2092 - categorical_accuracy: 0.9366
52320/60000 [=========================>....] - ETA: 13s - loss: 0.2091 - categorical_accuracy: 0.9366
52352/60000 [=========================>....] - ETA: 13s - loss: 0.2090 - categorical_accuracy: 0.9366
52384/60000 [=========================>....] - ETA: 13s - loss: 0.2089 - categorical_accuracy: 0.9367
52416/60000 [=========================>....] - ETA: 13s - loss: 0.2088 - categorical_accuracy: 0.9367
52448/60000 [=========================>....] - ETA: 13s - loss: 0.2087 - categorical_accuracy: 0.9367
52480/60000 [=========================>....] - ETA: 13s - loss: 0.2086 - categorical_accuracy: 0.9367
52512/60000 [=========================>....] - ETA: 13s - loss: 0.2085 - categorical_accuracy: 0.9368
52544/60000 [=========================>....] - ETA: 13s - loss: 0.2084 - categorical_accuracy: 0.9368
52576/60000 [=========================>....] - ETA: 13s - loss: 0.2085 - categorical_accuracy: 0.9368
52608/60000 [=========================>....] - ETA: 13s - loss: 0.2084 - categorical_accuracy: 0.9368
52640/60000 [=========================>....] - ETA: 13s - loss: 0.2083 - categorical_accuracy: 0.9368
52672/60000 [=========================>....] - ETA: 13s - loss: 0.2082 - categorical_accuracy: 0.9369
52704/60000 [=========================>....] - ETA: 12s - loss: 0.2081 - categorical_accuracy: 0.9369
52736/60000 [=========================>....] - ETA: 12s - loss: 0.2080 - categorical_accuracy: 0.9369
52768/60000 [=========================>....] - ETA: 12s - loss: 0.2079 - categorical_accuracy: 0.9369
52800/60000 [=========================>....] - ETA: 12s - loss: 0.2079 - categorical_accuracy: 0.9369
52832/60000 [=========================>....] - ETA: 12s - loss: 0.2078 - categorical_accuracy: 0.9370
52864/60000 [=========================>....] - ETA: 12s - loss: 0.2077 - categorical_accuracy: 0.9370
52896/60000 [=========================>....] - ETA: 12s - loss: 0.2076 - categorical_accuracy: 0.9370
52928/60000 [=========================>....] - ETA: 12s - loss: 0.2076 - categorical_accuracy: 0.9370
52960/60000 [=========================>....] - ETA: 12s - loss: 0.2075 - categorical_accuracy: 0.9371
52992/60000 [=========================>....] - ETA: 12s - loss: 0.2074 - categorical_accuracy: 0.9371
53024/60000 [=========================>....] - ETA: 12s - loss: 0.2073 - categorical_accuracy: 0.9372
53056/60000 [=========================>....] - ETA: 12s - loss: 0.2073 - categorical_accuracy: 0.9372
53088/60000 [=========================>....] - ETA: 12s - loss: 0.2072 - categorical_accuracy: 0.9372
53120/60000 [=========================>....] - ETA: 12s - loss: 0.2071 - categorical_accuracy: 0.9372
53152/60000 [=========================>....] - ETA: 12s - loss: 0.2070 - categorical_accuracy: 0.9373
53184/60000 [=========================>....] - ETA: 12s - loss: 0.2070 - categorical_accuracy: 0.9373
53216/60000 [=========================>....] - ETA: 12s - loss: 0.2069 - categorical_accuracy: 0.9373
53248/60000 [=========================>....] - ETA: 12s - loss: 0.2068 - categorical_accuracy: 0.9373
53280/60000 [=========================>....] - ETA: 11s - loss: 0.2067 - categorical_accuracy: 0.9373
53312/60000 [=========================>....] - ETA: 11s - loss: 0.2066 - categorical_accuracy: 0.9374
53344/60000 [=========================>....] - ETA: 11s - loss: 0.2065 - categorical_accuracy: 0.9374
53376/60000 [=========================>....] - ETA: 11s - loss: 0.2065 - categorical_accuracy: 0.9374
53408/60000 [=========================>....] - ETA: 11s - loss: 0.2066 - categorical_accuracy: 0.9374
53440/60000 [=========================>....] - ETA: 11s - loss: 0.2065 - categorical_accuracy: 0.9374
53472/60000 [=========================>....] - ETA: 11s - loss: 0.2064 - categorical_accuracy: 0.9374
53504/60000 [=========================>....] - ETA: 11s - loss: 0.2064 - categorical_accuracy: 0.9375
53536/60000 [=========================>....] - ETA: 11s - loss: 0.2062 - categorical_accuracy: 0.9375
53568/60000 [=========================>....] - ETA: 11s - loss: 0.2061 - categorical_accuracy: 0.9375
53600/60000 [=========================>....] - ETA: 11s - loss: 0.2061 - categorical_accuracy: 0.9376
53632/60000 [=========================>....] - ETA: 11s - loss: 0.2060 - categorical_accuracy: 0.9376
53664/60000 [=========================>....] - ETA: 11s - loss: 0.2059 - categorical_accuracy: 0.9376
53696/60000 [=========================>....] - ETA: 11s - loss: 0.2059 - categorical_accuracy: 0.9376
53728/60000 [=========================>....] - ETA: 11s - loss: 0.2058 - categorical_accuracy: 0.9377
53760/60000 [=========================>....] - ETA: 11s - loss: 0.2056 - categorical_accuracy: 0.9377
53792/60000 [=========================>....] - ETA: 11s - loss: 0.2055 - categorical_accuracy: 0.9377
53824/60000 [=========================>....] - ETA: 10s - loss: 0.2055 - categorical_accuracy: 0.9378
53856/60000 [=========================>....] - ETA: 10s - loss: 0.2054 - categorical_accuracy: 0.9378
53888/60000 [=========================>....] - ETA: 10s - loss: 0.2053 - categorical_accuracy: 0.9378
53920/60000 [=========================>....] - ETA: 10s - loss: 0.2052 - categorical_accuracy: 0.9379
53952/60000 [=========================>....] - ETA: 10s - loss: 0.2051 - categorical_accuracy: 0.9379
53984/60000 [=========================>....] - ETA: 10s - loss: 0.2051 - categorical_accuracy: 0.9379
54016/60000 [==========================>...] - ETA: 10s - loss: 0.2050 - categorical_accuracy: 0.9380
54048/60000 [==========================>...] - ETA: 10s - loss: 0.2049 - categorical_accuracy: 0.9380
54080/60000 [==========================>...] - ETA: 10s - loss: 0.2048 - categorical_accuracy: 0.9380
54112/60000 [==========================>...] - ETA: 10s - loss: 0.2049 - categorical_accuracy: 0.9380
54144/60000 [==========================>...] - ETA: 10s - loss: 0.2048 - categorical_accuracy: 0.9380
54176/60000 [==========================>...] - ETA: 10s - loss: 0.2047 - categorical_accuracy: 0.9380
54208/60000 [==========================>...] - ETA: 10s - loss: 0.2046 - categorical_accuracy: 0.9381
54240/60000 [==========================>...] - ETA: 10s - loss: 0.2046 - categorical_accuracy: 0.9381
54272/60000 [==========================>...] - ETA: 10s - loss: 0.2045 - categorical_accuracy: 0.9381
54304/60000 [==========================>...] - ETA: 10s - loss: 0.2044 - categorical_accuracy: 0.9382
54336/60000 [==========================>...] - ETA: 10s - loss: 0.2043 - categorical_accuracy: 0.9382
54368/60000 [==========================>...] - ETA: 10s - loss: 0.2042 - categorical_accuracy: 0.9382
54400/60000 [==========================>...] - ETA: 9s - loss: 0.2042 - categorical_accuracy: 0.9382 
54432/60000 [==========================>...] - ETA: 9s - loss: 0.2041 - categorical_accuracy: 0.9383
54464/60000 [==========================>...] - ETA: 9s - loss: 0.2040 - categorical_accuracy: 0.9383
54496/60000 [==========================>...] - ETA: 9s - loss: 0.2038 - categorical_accuracy: 0.9383
54528/60000 [==========================>...] - ETA: 9s - loss: 0.2039 - categorical_accuracy: 0.9383
54560/60000 [==========================>...] - ETA: 9s - loss: 0.2038 - categorical_accuracy: 0.9383
54592/60000 [==========================>...] - ETA: 9s - loss: 0.2037 - categorical_accuracy: 0.9384
54624/60000 [==========================>...] - ETA: 9s - loss: 0.2037 - categorical_accuracy: 0.9384
54656/60000 [==========================>...] - ETA: 9s - loss: 0.2036 - categorical_accuracy: 0.9384
54688/60000 [==========================>...] - ETA: 9s - loss: 0.2035 - categorical_accuracy: 0.9384
54720/60000 [==========================>...] - ETA: 9s - loss: 0.2035 - categorical_accuracy: 0.9384
54752/60000 [==========================>...] - ETA: 9s - loss: 0.2036 - categorical_accuracy: 0.9384
54784/60000 [==========================>...] - ETA: 9s - loss: 0.2036 - categorical_accuracy: 0.9384
54816/60000 [==========================>...] - ETA: 9s - loss: 0.2037 - categorical_accuracy: 0.9384
54848/60000 [==========================>...] - ETA: 9s - loss: 0.2036 - categorical_accuracy: 0.9385
54880/60000 [==========================>...] - ETA: 9s - loss: 0.2035 - categorical_accuracy: 0.9385
54912/60000 [==========================>...] - ETA: 9s - loss: 0.2035 - categorical_accuracy: 0.9385
54944/60000 [==========================>...] - ETA: 8s - loss: 0.2034 - categorical_accuracy: 0.9385
54976/60000 [==========================>...] - ETA: 8s - loss: 0.2033 - categorical_accuracy: 0.9385
55008/60000 [==========================>...] - ETA: 8s - loss: 0.2032 - categorical_accuracy: 0.9386
55040/60000 [==========================>...] - ETA: 8s - loss: 0.2033 - categorical_accuracy: 0.9385
55072/60000 [==========================>...] - ETA: 8s - loss: 0.2033 - categorical_accuracy: 0.9386
55104/60000 [==========================>...] - ETA: 8s - loss: 0.2033 - categorical_accuracy: 0.9386
55136/60000 [==========================>...] - ETA: 8s - loss: 0.2032 - categorical_accuracy: 0.9386
55168/60000 [==========================>...] - ETA: 8s - loss: 0.2031 - categorical_accuracy: 0.9386
55200/60000 [==========================>...] - ETA: 8s - loss: 0.2031 - categorical_accuracy: 0.9386
55232/60000 [==========================>...] - ETA: 8s - loss: 0.2029 - categorical_accuracy: 0.9387
55264/60000 [==========================>...] - ETA: 8s - loss: 0.2030 - categorical_accuracy: 0.9387
55296/60000 [==========================>...] - ETA: 8s - loss: 0.2030 - categorical_accuracy: 0.9387
55328/60000 [==========================>...] - ETA: 8s - loss: 0.2029 - categorical_accuracy: 0.9387
55360/60000 [==========================>...] - ETA: 8s - loss: 0.2029 - categorical_accuracy: 0.9387
55392/60000 [==========================>...] - ETA: 8s - loss: 0.2028 - categorical_accuracy: 0.9387
55424/60000 [==========================>...] - ETA: 8s - loss: 0.2028 - categorical_accuracy: 0.9387
55456/60000 [==========================>...] - ETA: 8s - loss: 0.2027 - categorical_accuracy: 0.9388
55488/60000 [==========================>...] - ETA: 8s - loss: 0.2028 - categorical_accuracy: 0.9388
55520/60000 [==========================>...] - ETA: 7s - loss: 0.2027 - categorical_accuracy: 0.9388
55552/60000 [==========================>...] - ETA: 7s - loss: 0.2027 - categorical_accuracy: 0.9388
55584/60000 [==========================>...] - ETA: 7s - loss: 0.2026 - categorical_accuracy: 0.9388
55616/60000 [==========================>...] - ETA: 7s - loss: 0.2026 - categorical_accuracy: 0.9388
55648/60000 [==========================>...] - ETA: 7s - loss: 0.2025 - categorical_accuracy: 0.9388
55680/60000 [==========================>...] - ETA: 7s - loss: 0.2024 - categorical_accuracy: 0.9389
55712/60000 [==========================>...] - ETA: 7s - loss: 0.2024 - categorical_accuracy: 0.9389
55744/60000 [==========================>...] - ETA: 7s - loss: 0.2023 - categorical_accuracy: 0.9389
55776/60000 [==========================>...] - ETA: 7s - loss: 0.2023 - categorical_accuracy: 0.9389
55808/60000 [==========================>...] - ETA: 7s - loss: 0.2022 - categorical_accuracy: 0.9389
55840/60000 [==========================>...] - ETA: 7s - loss: 0.2021 - categorical_accuracy: 0.9390
55872/60000 [==========================>...] - ETA: 7s - loss: 0.2021 - categorical_accuracy: 0.9390
55904/60000 [==========================>...] - ETA: 7s - loss: 0.2021 - categorical_accuracy: 0.9390
55936/60000 [==========================>...] - ETA: 7s - loss: 0.2021 - categorical_accuracy: 0.9390
55968/60000 [==========================>...] - ETA: 7s - loss: 0.2021 - categorical_accuracy: 0.9390
56000/60000 [===========================>..] - ETA: 7s - loss: 0.2022 - categorical_accuracy: 0.9390
56032/60000 [===========================>..] - ETA: 7s - loss: 0.2021 - categorical_accuracy: 0.9390
56064/60000 [===========================>..] - ETA: 7s - loss: 0.2020 - categorical_accuracy: 0.9390
56096/60000 [===========================>..] - ETA: 6s - loss: 0.2020 - categorical_accuracy: 0.9391
56128/60000 [===========================>..] - ETA: 6s - loss: 0.2020 - categorical_accuracy: 0.9391
56160/60000 [===========================>..] - ETA: 6s - loss: 0.2019 - categorical_accuracy: 0.9391
56192/60000 [===========================>..] - ETA: 6s - loss: 0.2018 - categorical_accuracy: 0.9391
56224/60000 [===========================>..] - ETA: 6s - loss: 0.2018 - categorical_accuracy: 0.9391
56256/60000 [===========================>..] - ETA: 6s - loss: 0.2017 - categorical_accuracy: 0.9392
56288/60000 [===========================>..] - ETA: 6s - loss: 0.2016 - categorical_accuracy: 0.9392
56320/60000 [===========================>..] - ETA: 6s - loss: 0.2015 - categorical_accuracy: 0.9392
56352/60000 [===========================>..] - ETA: 6s - loss: 0.2015 - categorical_accuracy: 0.9392
56384/60000 [===========================>..] - ETA: 6s - loss: 0.2015 - categorical_accuracy: 0.9392
56416/60000 [===========================>..] - ETA: 6s - loss: 0.2014 - categorical_accuracy: 0.9393
56448/60000 [===========================>..] - ETA: 6s - loss: 0.2013 - categorical_accuracy: 0.9393
56480/60000 [===========================>..] - ETA: 6s - loss: 0.2012 - categorical_accuracy: 0.9393
56512/60000 [===========================>..] - ETA: 6s - loss: 0.2011 - categorical_accuracy: 0.9393
56544/60000 [===========================>..] - ETA: 6s - loss: 0.2010 - categorical_accuracy: 0.9394
56576/60000 [===========================>..] - ETA: 6s - loss: 0.2009 - categorical_accuracy: 0.9394
56608/60000 [===========================>..] - ETA: 6s - loss: 0.2008 - categorical_accuracy: 0.9394
56640/60000 [===========================>..] - ETA: 5s - loss: 0.2008 - categorical_accuracy: 0.9395
56672/60000 [===========================>..] - ETA: 5s - loss: 0.2007 - categorical_accuracy: 0.9395
56704/60000 [===========================>..] - ETA: 5s - loss: 0.2006 - categorical_accuracy: 0.9395
56736/60000 [===========================>..] - ETA: 5s - loss: 0.2005 - categorical_accuracy: 0.9395
56768/60000 [===========================>..] - ETA: 5s - loss: 0.2004 - categorical_accuracy: 0.9396
56800/60000 [===========================>..] - ETA: 5s - loss: 0.2004 - categorical_accuracy: 0.9396
56832/60000 [===========================>..] - ETA: 5s - loss: 0.2003 - categorical_accuracy: 0.9396
56864/60000 [===========================>..] - ETA: 5s - loss: 0.2003 - categorical_accuracy: 0.9396
56896/60000 [===========================>..] - ETA: 5s - loss: 0.2001 - categorical_accuracy: 0.9396
56928/60000 [===========================>..] - ETA: 5s - loss: 0.2002 - categorical_accuracy: 0.9396
56960/60000 [===========================>..] - ETA: 5s - loss: 0.2002 - categorical_accuracy: 0.9396
56992/60000 [===========================>..] - ETA: 5s - loss: 0.2002 - categorical_accuracy: 0.9396
57024/60000 [===========================>..] - ETA: 5s - loss: 0.2001 - categorical_accuracy: 0.9397
57056/60000 [===========================>..] - ETA: 5s - loss: 0.2001 - categorical_accuracy: 0.9397
57088/60000 [===========================>..] - ETA: 5s - loss: 0.2000 - categorical_accuracy: 0.9397
57120/60000 [===========================>..] - ETA: 5s - loss: 0.2000 - categorical_accuracy: 0.9397
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1999 - categorical_accuracy: 0.9397
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1998 - categorical_accuracy: 0.9398
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1999 - categorical_accuracy: 0.9398
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1998 - categorical_accuracy: 0.9398
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1997 - categorical_accuracy: 0.9398
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1997 - categorical_accuracy: 0.9398
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1996 - categorical_accuracy: 0.9399
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1995 - categorical_accuracy: 0.9399
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1994 - categorical_accuracy: 0.9399
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1993 - categorical_accuracy: 0.9399
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1992 - categorical_accuracy: 0.9399
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1993 - categorical_accuracy: 0.9400
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1992 - categorical_accuracy: 0.9400
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1991 - categorical_accuracy: 0.9400
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1989 - categorical_accuracy: 0.9401
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1988 - categorical_accuracy: 0.9401
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1988 - categorical_accuracy: 0.9401
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1987 - categorical_accuracy: 0.9401
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1986 - categorical_accuracy: 0.9401
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1985 - categorical_accuracy: 0.9402
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1985 - categorical_accuracy: 0.9402
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1985 - categorical_accuracy: 0.9402
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1984 - categorical_accuracy: 0.9402
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1983 - categorical_accuracy: 0.9402
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1982 - categorical_accuracy: 0.9402
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1982 - categorical_accuracy: 0.9402
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1981 - categorical_accuracy: 0.9403
58016/60000 [============================>.] - ETA: 3s - loss: 0.1980 - categorical_accuracy: 0.9403
58048/60000 [============================>.] - ETA: 3s - loss: 0.1979 - categorical_accuracy: 0.9403
58080/60000 [============================>.] - ETA: 3s - loss: 0.1978 - categorical_accuracy: 0.9403
58112/60000 [============================>.] - ETA: 3s - loss: 0.1978 - categorical_accuracy: 0.9404
58144/60000 [============================>.] - ETA: 3s - loss: 0.1977 - categorical_accuracy: 0.9404
58176/60000 [============================>.] - ETA: 3s - loss: 0.1976 - categorical_accuracy: 0.9404
58208/60000 [============================>.] - ETA: 3s - loss: 0.1975 - categorical_accuracy: 0.9404
58240/60000 [============================>.] - ETA: 3s - loss: 0.1975 - categorical_accuracy: 0.9405
58272/60000 [============================>.] - ETA: 3s - loss: 0.1975 - categorical_accuracy: 0.9405
58304/60000 [============================>.] - ETA: 3s - loss: 0.1974 - categorical_accuracy: 0.9405
58336/60000 [============================>.] - ETA: 2s - loss: 0.1974 - categorical_accuracy: 0.9405
58368/60000 [============================>.] - ETA: 2s - loss: 0.1975 - categorical_accuracy: 0.9405
58400/60000 [============================>.] - ETA: 2s - loss: 0.1974 - categorical_accuracy: 0.9405
58432/60000 [============================>.] - ETA: 2s - loss: 0.1974 - categorical_accuracy: 0.9405
58464/60000 [============================>.] - ETA: 2s - loss: 0.1975 - categorical_accuracy: 0.9405
58496/60000 [============================>.] - ETA: 2s - loss: 0.1974 - categorical_accuracy: 0.9405
58528/60000 [============================>.] - ETA: 2s - loss: 0.1973 - categorical_accuracy: 0.9406
58560/60000 [============================>.] - ETA: 2s - loss: 0.1972 - categorical_accuracy: 0.9406
58592/60000 [============================>.] - ETA: 2s - loss: 0.1972 - categorical_accuracy: 0.9406
58624/60000 [============================>.] - ETA: 2s - loss: 0.1971 - categorical_accuracy: 0.9406
58656/60000 [============================>.] - ETA: 2s - loss: 0.1971 - categorical_accuracy: 0.9406
58688/60000 [============================>.] - ETA: 2s - loss: 0.1970 - categorical_accuracy: 0.9406
58720/60000 [============================>.] - ETA: 2s - loss: 0.1969 - categorical_accuracy: 0.9406
58752/60000 [============================>.] - ETA: 2s - loss: 0.1969 - categorical_accuracy: 0.9406
58784/60000 [============================>.] - ETA: 2s - loss: 0.1969 - categorical_accuracy: 0.9407
58816/60000 [============================>.] - ETA: 2s - loss: 0.1968 - categorical_accuracy: 0.9407
58848/60000 [============================>.] - ETA: 2s - loss: 0.1967 - categorical_accuracy: 0.9407
58880/60000 [============================>.] - ETA: 1s - loss: 0.1967 - categorical_accuracy: 0.9407
58912/60000 [============================>.] - ETA: 1s - loss: 0.1966 - categorical_accuracy: 0.9408
58944/60000 [============================>.] - ETA: 1s - loss: 0.1965 - categorical_accuracy: 0.9408
58976/60000 [============================>.] - ETA: 1s - loss: 0.1965 - categorical_accuracy: 0.9408
59008/60000 [============================>.] - ETA: 1s - loss: 0.1964 - categorical_accuracy: 0.9408
59040/60000 [============================>.] - ETA: 1s - loss: 0.1963 - categorical_accuracy: 0.9409
59072/60000 [============================>.] - ETA: 1s - loss: 0.1963 - categorical_accuracy: 0.9409
59104/60000 [============================>.] - ETA: 1s - loss: 0.1963 - categorical_accuracy: 0.9409
59136/60000 [============================>.] - ETA: 1s - loss: 0.1962 - categorical_accuracy: 0.9409
59168/60000 [============================>.] - ETA: 1s - loss: 0.1961 - categorical_accuracy: 0.9409
59200/60000 [============================>.] - ETA: 1s - loss: 0.1960 - categorical_accuracy: 0.9409
59232/60000 [============================>.] - ETA: 1s - loss: 0.1960 - categorical_accuracy: 0.9410
59264/60000 [============================>.] - ETA: 1s - loss: 0.1959 - categorical_accuracy: 0.9410
59296/60000 [============================>.] - ETA: 1s - loss: 0.1959 - categorical_accuracy: 0.9410
59328/60000 [============================>.] - ETA: 1s - loss: 0.1958 - categorical_accuracy: 0.9410
59360/60000 [============================>.] - ETA: 1s - loss: 0.1957 - categorical_accuracy: 0.9410
59392/60000 [============================>.] - ETA: 1s - loss: 0.1956 - categorical_accuracy: 0.9411
59424/60000 [============================>.] - ETA: 1s - loss: 0.1955 - categorical_accuracy: 0.9411
59456/60000 [============================>.] - ETA: 0s - loss: 0.1957 - categorical_accuracy: 0.9411
59488/60000 [============================>.] - ETA: 0s - loss: 0.1956 - categorical_accuracy: 0.9411
59520/60000 [============================>.] - ETA: 0s - loss: 0.1956 - categorical_accuracy: 0.9411
59552/60000 [============================>.] - ETA: 0s - loss: 0.1956 - categorical_accuracy: 0.9411
59584/60000 [============================>.] - ETA: 0s - loss: 0.1955 - categorical_accuracy: 0.9411
59616/60000 [============================>.] - ETA: 0s - loss: 0.1955 - categorical_accuracy: 0.9411
59648/60000 [============================>.] - ETA: 0s - loss: 0.1954 - categorical_accuracy: 0.9411
59680/60000 [============================>.] - ETA: 0s - loss: 0.1954 - categorical_accuracy: 0.9412
59712/60000 [============================>.] - ETA: 0s - loss: 0.1953 - categorical_accuracy: 0.9412
59744/60000 [============================>.] - ETA: 0s - loss: 0.1952 - categorical_accuracy: 0.9412
59776/60000 [============================>.] - ETA: 0s - loss: 0.1951 - categorical_accuracy: 0.9412
59808/60000 [============================>.] - ETA: 0s - loss: 0.1951 - categorical_accuracy: 0.9412
59840/60000 [============================>.] - ETA: 0s - loss: 0.1950 - categorical_accuracy: 0.9413
59872/60000 [============================>.] - ETA: 0s - loss: 0.1950 - categorical_accuracy: 0.9413
59904/60000 [============================>.] - ETA: 0s - loss: 0.1949 - categorical_accuracy: 0.9413
59936/60000 [============================>.] - ETA: 0s - loss: 0.1949 - categorical_accuracy: 0.9413
59968/60000 [============================>.] - ETA: 0s - loss: 0.1948 - categorical_accuracy: 0.9413
60000/60000 [==============================] - 110s 2ms/step - loss: 0.1947 - categorical_accuracy: 0.9413 - val_loss: 0.0511 - val_categorical_accuracy: 0.9840

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
 1632/10000 [===>..........................] - ETA: 2s
 1792/10000 [====>.........................] - ETA: 2s
 1952/10000 [====>.........................] - ETA: 2s
 2112/10000 [=====>........................] - ETA: 2s
 2272/10000 [=====>........................] - ETA: 2s
 2432/10000 [======>.......................] - ETA: 2s
 2592/10000 [======>.......................] - ETA: 2s
 2752/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3072/10000 [========>.....................] - ETA: 2s
 3232/10000 [========>.....................] - ETA: 2s
 3392/10000 [=========>....................] - ETA: 2s
 3552/10000 [=========>....................] - ETA: 2s
 3712/10000 [==========>...................] - ETA: 2s
 3840/10000 [==========>...................] - ETA: 2s
 4000/10000 [===========>..................] - ETA: 2s
 4160/10000 [===========>..................] - ETA: 2s
 4320/10000 [===========>..................] - ETA: 1s
 4480/10000 [============>.................] - ETA: 1s
 4640/10000 [============>.................] - ETA: 1s
 4800/10000 [=============>................] - ETA: 1s
 4960/10000 [=============>................] - ETA: 1s
 5120/10000 [==============>...............] - ETA: 1s
 5248/10000 [==============>...............] - ETA: 1s
 5408/10000 [===============>..............] - ETA: 1s
 5568/10000 [===============>..............] - ETA: 1s
 5728/10000 [================>.............] - ETA: 1s
 5888/10000 [================>.............] - ETA: 1s
 6048/10000 [=================>............] - ETA: 1s
 6208/10000 [=================>............] - ETA: 1s
 6368/10000 [==================>...........] - ETA: 1s
 6528/10000 [==================>...........] - ETA: 1s
 6688/10000 [===================>..........] - ETA: 1s
 6848/10000 [===================>..........] - ETA: 1s
 7008/10000 [====================>.........] - ETA: 1s
 7168/10000 [====================>.........] - ETA: 0s
 7328/10000 [====================>.........] - ETA: 0s
 7488/10000 [=====================>........] - ETA: 0s
 7616/10000 [=====================>........] - ETA: 0s
 7776/10000 [======================>.......] - ETA: 0s
 7936/10000 [======================>.......] - ETA: 0s
 8096/10000 [=======================>......] - ETA: 0s
 8256/10000 [=======================>......] - ETA: 0s
 8416/10000 [========================>.....] - ETA: 0s
 8576/10000 [========================>.....] - ETA: 0s
 8768/10000 [=========================>....] - ETA: 0s
 8928/10000 [=========================>....] - ETA: 0s
 9088/10000 [==========================>...] - ETA: 0s
 9248/10000 [==========================>...] - ETA: 0s
 9408/10000 [===========================>..] - ETA: 0s
 9568/10000 [===========================>..] - ETA: 0s
 9728/10000 [============================>.] - ETA: 0s
 9888/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 341us/step
[[1.1756786e-08 1.2388913e-09 1.5518091e-07 ... 9.9999952e-01
  3.1809352e-10 1.5986491e-07]
 [5.1839510e-07 2.8849323e-05 9.9996889e-01 ... 1.2449560e-08
  4.6441926e-07 1.8499445e-11]
 [5.1946324e-07 9.9991477e-01 2.1012265e-05 ... 1.1127202e-05
  2.3422217e-05 5.7597026e-07]
 ...
 [3.8005693e-10 1.8420327e-06 1.9760975e-08 ... 9.0115718e-06
  8.8902989e-06 5.9209367e-05]
 [2.0375278e-07 8.9529557e-09 2.5201114e-08 ... 4.9064351e-08
  7.5109280e-04 4.4335629e-06]
 [8.6509363e-06 4.1246499e-06 6.1984829e-05 ... 1.8357126e-08
  7.0693463e-06 5.7490258e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.0511030982013559, 'accuracy_test:': 0.984000027179718}

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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
{'loss': 0.39737941697239876, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-19 00:32:06.300440: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
 40%|████      | 2/5 [00:20<00:31, 10.35s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.785144763730315, 'learning_rate': 0.10465383401388063, 'min_data_in_leaf': 28, 'num_leaves': 41} and reward: 0.389
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\x1f\xe7\xe9[\x1d~X\r\x00\x00\x00learning_rateq\x02G?\xba\xca\x97\xfa}\x99\xe9X\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K)u.' and reward: 0.389
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\x1f\xe7\xe9[\x1d~X\r\x00\x00\x00learning_rateq\x02G?\xba\xca\x97\xfa}\x99\xe9X\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K)u.' and reward: 0.389
 60%|██████    | 3/5 [00:41<00:27, 13.56s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.8345319309089222, 'learning_rate': 0.13237621990051868, 'min_data_in_leaf': 22, 'num_leaves': 40} and reward: 0.3918
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xb4|N\xd7\x16\xf5X\r\x00\x00\x00learning_rateq\x02G?\xc0\xf1\xb47\x9e\xd3\x96X\x10\x00\x00\x00min_data_in_leafq\x03K\x16X\n\x00\x00\x00num_leavesq\x04K(u.' and reward: 0.3918
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xb4|N\xd7\x16\xf5X\r\x00\x00\x00learning_rateq\x02G?\xc0\xf1\xb47\x9e\xd3\x96X\x10\x00\x00\x00min_data_in_leafq\x03K\x16X\n\x00\x00\x00num_leavesq\x04K(u.' and reward: 0.3918
 80%|████████  | 4/5 [01:02<00:15, 15.81s/it] 80%|████████  | 4/5 [01:02<00:15, 15.70s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.855623797894829, 'learning_rate': 0.011294354005093404, 'min_data_in_leaf': 22, 'num_leaves': 30} and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xebaE(\xb4gAX\r\x00\x00\x00learning_rateq\x02G?\x87!~\x88\xa6\x8c\x90X\x10\x00\x00\x00min_data_in_leafq\x03K\x16X\n\x00\x00\x00num_leavesq\x04K\x1eu.' and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xebaE(\xb4gAX\r\x00\x00\x00learning_rateq\x02G?\x87!~\x88\xa6\x8c\x90X\x10\x00\x00\x00min_data_in_leafq\x03K\x16X\n\x00\x00\x00num_leavesq\x04K\x1eu.' and reward: 0.3888
Time for Gradient Boosting hyperparameter optimization: 80.80080389976501
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.8345319309089222, 'learning_rate': 0.13237621990051868, 'min_data_in_leaf': 22, 'num_leaves': 40}
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
 40%|████      | 2/5 [00:47<01:11, 23.73s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.4260917393388405, 'embedding_size_factor': 1.4528447856644346, 'layers.choice': 3, 'learning_rate': 0.0008785824615521918, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 1.8542260090706785e-11} and reward: 0.3646
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdbE\x16Ic\x96\xc2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7>\xda,\x89|lX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?L\xca\x15x7\xe5\xa2X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xb4c.\xa6q\x01Tu.' and reward: 0.3646
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdbE\x16Ic\x96\xc2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7>\xda,\x89|lX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?L\xca\x15x7\xe5\xa2X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xb4c.\xa6q\x01Tu.' and reward: 0.3646
 60%|██████    | 3/5 [01:39<01:04, 32.28s/it] 60%|██████    | 3/5 [01:39<01:06, 33.23s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.27743954123336523, 'embedding_size_factor': 0.9580159427937195, 'layers.choice': 0, 'learning_rate': 0.0007146674469304247, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 0.0011461428975953515} and reward: 0.3608
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd1\xc1\x91\xc7\r\xbb$X\x15\x00\x00\x00embedding_size_factorq\x03G?\xee\xa8\x11\x0c\xeb\x0fkX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?Gk\x10\xa7\xf2MzX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?R\xc7E\x90\xbf\xef\x10u.' and reward: 0.3608
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd1\xc1\x91\xc7\r\xbb$X\x15\x00\x00\x00embedding_size_factorq\x03G?\xee\xa8\x11\x0c\xeb\x0fkX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?Gk\x10\xa7\xf2MzX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?R\xc7E\x90\xbf\xef\x10u.' and reward: 0.3608
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 153.11590480804443
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -117.8s of remaining time.
Ensemble size: 63
Ensemble weights: 
[0.25396825 0.20634921 0.36507937 0.03174603 0.03174603 0.
 0.11111111]
	0.4006	 = Validation accuracy score
	1.69s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 239.55s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f5deffe5cf8>

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
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
100%|██████████| 10/10 [00:02<00:00,  3.66it/s, avg_epoch_loss=5.27]
INFO:root:Epoch[0] Elapsed time 2.736 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.273047
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.273046970367432 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f14fb552400>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f14fb552400>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 98.33it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1028.89404296875,
    "abs_error": 363.2300109863281,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.4067438265876313,
    "sMAPE": 0.5058417824204394,
    "MSIS": 96.2697514458417,
    "QuantileLoss[0.5]": 363.2300109863281,
    "Coverage[0.5]": 1.0,
    "RMSE": 32.07637827075791,
    "NRMSE": 0.6752921741212191,
    "ND": 0.6372456333093476,
    "wQuantileLoss[0.5]": 0.6372456333093476,
    "mean_wQuantileLoss": 0.6372456333093476,
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
100%|██████████| 10/10 [00:01<00:00,  7.40it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.351 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f14cf038d30>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f14cf038d30>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 142.29it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.30it/s, avg_epoch_loss=5.21]
INFO:root:Epoch[0] Elapsed time 1.888 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.206671
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.206670618057251 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f14ceffcf28>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f14ceffcf28>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 131.24it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 308.6103922526042,
    "abs_error": 186.87319946289062,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.2382124426908099,
    "sMAPE": 0.3038222056117716,
    "MSIS": 49.528499325295925,
    "QuantileLoss[0.5]": 186.8732032775879,
    "Coverage[0.5]": 0.75,
    "RMSE": 17.567310330628427,
    "NRMSE": 0.36983811222375634,
    "ND": 0.32784771835594845,
    "wQuantileLoss[0.5]": 0.3278477250483998,
    "mean_wQuantileLoss": 0.3278477250483998,
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
 30%|███       | 3/10 [00:12<00:30,  4.30s/it, avg_epoch_loss=6.94] 60%|██████    | 6/10 [00:24<00:16,  4.16s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:35<00:04,  4.06s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:39<00:00,  3.96s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 39.645 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.865111
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.86511116027832 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f14cf038d30>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f14cf038d30>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 121.95it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 54404.671875,
    "abs_error": 2758.6904296875,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 18.278944361143875,
    "sMAPE": 1.4181813273936041,
    "MSIS": 731.157722680522,
    "QuantileLoss[0.5]": 2758.6903686523438,
    "Coverage[0.5]": 1.0,
    "RMSE": 233.24809082819948,
    "NRMSE": 4.910486122698936,
    "ND": 4.839807771381579,
    "wQuantileLoss[0.5]": 4.8398076643023575,
    "mean_wQuantileLoss": 4.8398076643023575,
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
100%|██████████| 10/10 [00:00<00:00, 49.49it/s, avg_epoch_loss=5.07]
INFO:root:Epoch[0] Elapsed time 0.203 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.068160
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.068159675598144 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f14abd68f60>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f14abd68f60>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 127.87it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 329.48236083984375,
    "abs_error": 200.0093231201172,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.325251738897289,
    "sMAPE": 0.32966060846636164,
    "MSIS": 53.01006389406921,
    "QuantileLoss[0.5]": 200.00930786132812,
    "Coverage[0.5]": 0.75,
    "RMSE": 18.151648984041195,
    "NRMSE": 0.3821399786113936,
    "ND": 0.35089354933353895,
    "wQuantileLoss[0.5]": 0.3508935225637336,
    "mean_wQuantileLoss": 0.3508935225637336,
    "MAE_Coverage": 0.25
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
100%|██████████| 10/10 [00:01<00:00,  8.06it/s, avg_epoch_loss=144]
INFO:root:Epoch[0] Elapsed time 1.241 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=144.412806
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 144.41280616614463 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f14cef61198>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f14cef61198>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 146.72it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 2434.5670123443265,
    "abs_error": 573.7377910258112,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 3.8015578142395565,
    "sMAPE": 1.9586154749802402,
    "MSIS": 152.0623125695823,
    "QuantileLoss[0.5]": 573.7377910258112,
    "Coverage[0.5]": 0.0,
    "RMSE": 49.34133168393742,
    "NRMSE": 1.0387648775565772,
    "ND": 1.0065575281154584,
    "wQuantileLoss[0.5]": 1.0065575281154584,
    "mean_wQuantileLoss": 1.0065575281154584,
    "MAE_Coverage": 0.5
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
 10%|█         | 1/10 [02:02<18:22, 122.49s/it, avg_epoch_loss=0.488] 20%|██        | 2/10 [05:05<18:46, 140.75s/it, avg_epoch_loss=0.47]  30%|███       | 3/10 [08:35<18:49, 161.33s/it, avg_epoch_loss=0.453] 40%|████      | 4/10 [11:59<17:25, 174.21s/it, avg_epoch_loss=0.437] 50%|█████     | 5/10 [15:09<14:55, 179.08s/it, avg_epoch_loss=0.424] 60%|██████    | 6/10 [18:25<12:16, 184.06s/it, avg_epoch_loss=0.414] 70%|███████   | 7/10 [21:52<09:32, 190.86s/it, avg_epoch_loss=0.407] 80%|████████  | 8/10 [25:18<06:30, 195.48s/it, avg_epoch_loss=0.404] 90%|█████████ | 9/10 [28:34<03:15, 195.63s/it, avg_epoch_loss=0.401]100%|██████████| 10/10 [32:18<00:00, 203.99s/it, avg_epoch_loss=0.399]100%|██████████| 10/10 [32:18<00:00, 193.81s/it, avg_epoch_loss=0.399]
INFO:root:Epoch[0] Elapsed time 1938.100 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.398938
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3989381194114685 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f14abd52048>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f14abd52048>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 18.97it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 137.49435424804688,
    "abs_error": 102.26634979248047,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 0.6776117021893543,
    "sMAPE": 0.177204681090722,
    "MSIS": 27.104465256662994,
    "QuantileLoss[0.5]": 102.26634216308594,
    "Coverage[0.5]": 0.4166666666666667,
    "RMSE": 11.725798661415217,
    "NRMSE": 0.2468589191876888,
    "ND": 0.17941464875873767,
    "wQuantileLoss[0.5]": 0.17941463537383498,
    "mean_wQuantileLoss": 0.17941463537383498,
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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f1e42926780> 

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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
   1.40925339e+00  9.66539250e-01]
 [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
   8.84908814e-01  2.76499305e-01]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]
 [ 1.06702918e+00 -4.29142278e-01  3.50167159e-01  1.20845633e+00
   7.51480619e-01  1.11570180e+00 -4.79157099e-01  8.40861558e-01
  -1.02887218e-01  1.71647264e-02]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 8.57296491e-01  9.56121704e-01 -8.26097432e-01 -7.05840507e-01
   1.13872896e+00  1.19268607e+00  2.82675712e-01 -2.37941936e-01
   1.15528789e+00  6.21082701e-01]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 7.22978007e-01  1.85535621e-01  9.15499268e-01  3.94428030e-01
  -8.49830738e-01  7.25522558e-01 -1.50504326e-01  1.49588477e+00
   6.75453809e-01 -4.38200267e-01]
 [ 1.77547698e+00 -2.03394449e-01 -1.98837863e-01  2.42669441e-01
   9.64350564e-01  2.01830179e-01 -5.45774168e-01  6.61020288e-01
   1.79215821e+00 -7.00398505e-01]
 [ 8.78740711e-01 -1.92316341e-02  3.19656942e-01  1.50016279e-01
  -1.46662161e+00  4.63534322e-01 -8.98683193e-01  3.97880425e-01
  -9.96010889e-01  3.18154200e-01]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 1.27991386e+00 -8.71422066e-01 -3.24032329e-01 -8.64829941e-01
  -9.68539694e-01  6.08749082e-01  5.07984337e-01  5.61638097e-01
   1.51475038e+00 -1.51107661e+00]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
  -2.43250851e+00 -2.27729800e+00]
 [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
   1.40395436e-01  1.91979229e+00]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 9.97855163e-01 -6.00138799e-01  4.57947076e-01  1.46765263e-01
  -9.33557290e-01  5.71804879e-01  5.72962726e-01 -3.68176565e-02
   1.12368489e-01 -1.78175491e-02]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 1.21619061e+00 -1.90005215e-02  8.60891241e-01 -2.26760192e-01
  -1.36419132e+00 -1.56450785e+00  1.63169151e+00  9.31255679e-01
   9.49808815e-01 -8.80189065e-01]
 [ 6.67591795e-01 -4.52524973e-01 -6.05981321e-01  1.16128569e+00
  -1.44620987e+00  1.06996554e+00  1.92381543e+00 -1.04553425e+00
   3.55284507e-01  1.80358898e+00]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f8aa97c2da0>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f8ac3b315f8> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 0.85771953  0.09811225 -0.26046606  1.06032751 -1.39003042 -1.71116766
   0.2656424   1.65712464  1.41767401  0.44509671]
 [ 1.1437713   0.7278135   0.35249436  0.51507361  1.17718111 -2.78253447
  -1.94332341  0.58464661  0.32427424 -0.23643695]
 [ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]
 [ 0.46739791 -0.23787527 -0.15449119 -0.75566277 -0.54706224  1.85143789
  -1.46405357  0.20909668  1.55501599 -0.09243232]
 [ 1.34728643 -0.36453805  0.08075099 -0.45971768 -0.8894876   1.70548352
   0.09499611  0.24050555 -0.9994265  -0.76780375]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
   1.10223673  0.81434313  0.78046993  1.10861676]
 [ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
  -0.87728152 -0.78191168 -0.43750898 -1.44087602]
 [ 1.06523311 -0.66486777  1.00806543 -1.94504696 -1.23017555 -0.91542437
   0.33722094  1.22515585 -1.05354607  0.78522692]
 [ 0.76170668 -1.48515645  1.30253554 -0.59246129 -1.64162479 -2.30490794
  -1.34869645 -0.03181717  0.11248774 -0.36261209]
 [ 0.87874071 -0.01923163  0.31965694  0.15001628 -1.46662161  0.46353432
  -0.89868319  0.39788042 -0.99601089  0.3181542 ]
 [ 1.21619061 -0.01900052  0.86089124 -0.22676019 -1.36419132 -1.56450785
   1.63169151  0.93125568  0.94980882 -0.88018906]
 [ 0.89562312 -2.29820588 -0.01952256  1.45652739 -1.85064099  0.31663724
   0.11133727 -2.66412594 -0.42642862 -0.83998891]
 [ 0.70017571  0.55607351  0.08968641  1.69380911  0.88239331  0.19686978
  -0.56378873  0.16986926 -1.16400797 -0.6011568 ]
 [ 0.55853873 -0.51634791 -0.51814555  0.3511169   0.82550695 -0.06877046
  -0.9520621  -1.34776494  1.47073986 -1.4614036 ]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 0.84806927  0.45194604  0.63019567 -1.57915629  0.82798737 -0.82862798
  -0.10534471  0.52887975 -2.23708651 -0.4148469 ]
 [ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 0.6109426  -2.79099641 -1.33520272 -0.45611756 -0.94495995 -0.97989025
  -0.15699367  0.69257435 -0.47867236 -0.10646012]
 [ 1.37661405 -0.60022533  0.72591685 -0.37951752 -0.62754626 -1.01480369
   0.96622086  0.4359862  -0.68748739  3.32107876]
 [ 1.06702918 -0.42914228  0.35016716  1.20845633  0.75148062  1.1157018
  -0.4791571   0.84086156 -0.10288722  0.01716473]
 [ 0.87226739 -2.51630386 -0.77507029 -0.59566788  1.02600767 -0.30912132
   1.74643509  0.51093777  1.71066184  0.14164054]]
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
[[ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 1.16777676 -0.66575452 -1.23312074 -1.67419581  1.01313574  0.82502982
  -0.12046457 -0.49821356 -0.31098498 -1.18231813]
 [ 0.69174373  1.00978733 -1.21333813 -1.55694156 -1.20257258 -0.61244213
  -2.69836174 -0.13935181 -0.72853749  0.0722519 ]
 [ 1.02817479 -0.50845713  1.7653351   0.77741921  0.61771419 -0.11877117
   0.45015551 -0.19899818  1.86647138  0.8709698 ]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]
 [ 1.838294    0.50274088  0.12910158  1.55880554  1.32551412  0.1094027
   1.40754    -1.2197444   2.44936865  1.6169496 ]
 [ 1.34728643 -0.36453805  0.08075099 -0.45971768 -0.8894876   1.70548352
   0.09499611  0.24050555 -0.9994265  -0.76780375]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 0.85877496  2.29371761 -1.47023709 -0.83001099 -0.67204982 -1.01951985
   0.59921324 -0.21465384  1.02124813  0.60640394]
 [ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
  -1.09715436 -0.50897228 -0.16648595 -1.03918232]
 [ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]
 [ 1.14809657 -0.7332716   0.26246745  0.83600472  1.17353145  1.54335911
   0.28474811  0.75880566  0.88490881  0.2764993 ]
 [ 1.06702918 -0.42914228  0.35016716  1.20845633  0.75148062  1.1157018
  -0.4791571   0.84086156 -0.10288722  0.01716473]
 [ 1.09488485 -0.06962454 -0.11644415  0.35387043 -1.44189096 -0.18695502
   1.2911889  -0.15323616 -2.43250851 -2.277298  ]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 1.06523311 -0.66486777  1.00806543 -1.94504696 -1.23017555 -0.91542437
   0.33722094  1.22515585 -1.05354607  0.78522692]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 0.85729649  0.9561217  -0.82609743 -0.70584051  1.13872896  1.19268607
   0.28267571 -0.23794194  1.15528789  0.6210827 ]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 2.07582971 -1.40232915 -0.47918492  0.45112294  1.03436581 -0.6949209
  -0.4189379   0.5154138  -1.11487105 -1.95210529]]
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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140448139874256
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140448139874032
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140448139872800
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140448139872352
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140448139871848
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140448139871512

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
grad_step = 000000, loss = 1.264046
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 1.080827
grad_step = 000002, loss = 0.921981
grad_step = 000003, loss = 0.741690
grad_step = 000004, loss = 0.529293
grad_step = 000005, loss = 0.304055
grad_step = 000006, loss = 0.112105
grad_step = 000007, loss = 0.037512
grad_step = 000008, loss = 0.147109
grad_step = 000009, loss = 0.220878
grad_step = 000010, loss = 0.156196
grad_step = 000011, loss = 0.066761
grad_step = 000012, loss = 0.017752
grad_step = 000013, loss = 0.013559
grad_step = 000014, loss = 0.032717
grad_step = 000015, loss = 0.054656
grad_step = 000016, loss = 0.069304
grad_step = 000017, loss = 0.073033
grad_step = 000018, loss = 0.065988
grad_step = 000019, loss = 0.050936
grad_step = 000020, loss = 0.032614
grad_step = 000021, loss = 0.016627
grad_step = 000022, loss = 0.007903
grad_step = 000023, loss = 0.008638
grad_step = 000024, loss = 0.016377
grad_step = 000025, loss = 0.024982
grad_step = 000026, loss = 0.028612
grad_step = 000027, loss = 0.025518
grad_step = 000028, loss = 0.018183
grad_step = 000029, loss = 0.010873
grad_step = 000030, loss = 0.006657
grad_step = 000031, loss = 0.006239
grad_step = 000032, loss = 0.008415
grad_step = 000033, loss = 0.011321
grad_step = 000034, loss = 0.013399
grad_step = 000035, loss = 0.013846
grad_step = 000036, loss = 0.012636
grad_step = 000037, loss = 0.010292
grad_step = 000038, loss = 0.007714
grad_step = 000039, loss = 0.005763
grad_step = 000040, loss = 0.004989
grad_step = 000041, loss = 0.005415
grad_step = 000042, loss = 0.006519
grad_step = 000043, loss = 0.007529
grad_step = 000044, loss = 0.007851
grad_step = 000045, loss = 0.007350
grad_step = 000046, loss = 0.006334
grad_step = 000047, loss = 0.005304
grad_step = 000048, loss = 0.004673
grad_step = 000049, loss = 0.004587
grad_step = 000050, loss = 0.004919
grad_step = 000051, loss = 0.005366
grad_step = 000052, loss = 0.005639
grad_step = 000053, loss = 0.005574
grad_step = 000054, loss = 0.005202
grad_step = 000055, loss = 0.004710
grad_step = 000056, loss = 0.004332
grad_step = 000057, loss = 0.004214
grad_step = 000058, loss = 0.004340
grad_step = 000059, loss = 0.004566
grad_step = 000060, loss = 0.004710
grad_step = 000061, loss = 0.004673
grad_step = 000062, loss = 0.004472
grad_step = 000063, loss = 0.004221
grad_step = 000064, loss = 0.004042
grad_step = 000065, loss = 0.003996
grad_step = 000066, loss = 0.004058
grad_step = 000067, loss = 0.004150
grad_step = 000068, loss = 0.004195
grad_step = 000069, loss = 0.004163
grad_step = 000070, loss = 0.004070
grad_step = 000071, loss = 0.003959
grad_step = 000072, loss = 0.003871
grad_step = 000073, loss = 0.003829
grad_step = 000074, loss = 0.003832
grad_step = 000075, loss = 0.003855
grad_step = 000076, loss = 0.003868
grad_step = 000077, loss = 0.003849
grad_step = 000078, loss = 0.003797
grad_step = 000079, loss = 0.003732
grad_step = 000080, loss = 0.003680
grad_step = 000081, loss = 0.003655
grad_step = 000082, loss = 0.003652
grad_step = 000083, loss = 0.003653
grad_step = 000084, loss = 0.003640
grad_step = 000085, loss = 0.003608
grad_step = 000086, loss = 0.003566
grad_step = 000087, loss = 0.003528
grad_step = 000088, loss = 0.003502
grad_step = 000089, loss = 0.003487
grad_step = 000090, loss = 0.003475
grad_step = 000091, loss = 0.003458
grad_step = 000092, loss = 0.003432
grad_step = 000093, loss = 0.003401
grad_step = 000094, loss = 0.003371
grad_step = 000095, loss = 0.003346
grad_step = 000096, loss = 0.003325
grad_step = 000097, loss = 0.003306
grad_step = 000098, loss = 0.003287
grad_step = 000099, loss = 0.003264
grad_step = 000100, loss = 0.003237
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003209
grad_step = 000102, loss = 0.003182
grad_step = 000103, loss = 0.003159
grad_step = 000104, loss = 0.003137
grad_step = 000105, loss = 0.003114
grad_step = 000106, loss = 0.003088
grad_step = 000107, loss = 0.003061
grad_step = 000108, loss = 0.003033
grad_step = 000109, loss = 0.003006
grad_step = 000110, loss = 0.002980
grad_step = 000111, loss = 0.002953
grad_step = 000112, loss = 0.002926
grad_step = 000113, loss = 0.002898
grad_step = 000114, loss = 0.002870
grad_step = 000115, loss = 0.002841
grad_step = 000116, loss = 0.002812
grad_step = 000117, loss = 0.002783
grad_step = 000118, loss = 0.002755
grad_step = 000119, loss = 0.002727
grad_step = 000120, loss = 0.002698
grad_step = 000121, loss = 0.002668
grad_step = 000122, loss = 0.002639
grad_step = 000123, loss = 0.002611
grad_step = 000124, loss = 0.002583
grad_step = 000125, loss = 0.002556
grad_step = 000126, loss = 0.002528
grad_step = 000127, loss = 0.002500
grad_step = 000128, loss = 0.002474
grad_step = 000129, loss = 0.002448
grad_step = 000130, loss = 0.002423
grad_step = 000131, loss = 0.002398
grad_step = 000132, loss = 0.002374
grad_step = 000133, loss = 0.002351
grad_step = 000134, loss = 0.002329
grad_step = 000135, loss = 0.002307
grad_step = 000136, loss = 0.002287
grad_step = 000137, loss = 0.002267
grad_step = 000138, loss = 0.002248
grad_step = 000139, loss = 0.002230
grad_step = 000140, loss = 0.002212
grad_step = 000141, loss = 0.002196
grad_step = 000142, loss = 0.002180
grad_step = 000143, loss = 0.002165
grad_step = 000144, loss = 0.002150
grad_step = 000145, loss = 0.002136
grad_step = 000146, loss = 0.002123
grad_step = 000147, loss = 0.002110
grad_step = 000148, loss = 0.002097
grad_step = 000149, loss = 0.002084
grad_step = 000150, loss = 0.002072
grad_step = 000151, loss = 0.002060
grad_step = 000152, loss = 0.002048
grad_step = 000153, loss = 0.002036
grad_step = 000154, loss = 0.002025
grad_step = 000155, loss = 0.002014
grad_step = 000156, loss = 0.002002
grad_step = 000157, loss = 0.001991
grad_step = 000158, loss = 0.001979
grad_step = 000159, loss = 0.001968
grad_step = 000160, loss = 0.001956
grad_step = 000161, loss = 0.001945
grad_step = 000162, loss = 0.001933
grad_step = 000163, loss = 0.001922
grad_step = 000164, loss = 0.001911
grad_step = 000165, loss = 0.001899
grad_step = 000166, loss = 0.001888
grad_step = 000167, loss = 0.001877
grad_step = 000168, loss = 0.001866
grad_step = 000169, loss = 0.001854
grad_step = 000170, loss = 0.001843
grad_step = 000171, loss = 0.001832
grad_step = 000172, loss = 0.001820
grad_step = 000173, loss = 0.001809
grad_step = 000174, loss = 0.001797
grad_step = 000175, loss = 0.001786
grad_step = 000176, loss = 0.001774
grad_step = 000177, loss = 0.001763
grad_step = 000178, loss = 0.001752
grad_step = 000179, loss = 0.001740
grad_step = 000180, loss = 0.001728
grad_step = 000181, loss = 0.001717
grad_step = 000182, loss = 0.001705
grad_step = 000183, loss = 0.001693
grad_step = 000184, loss = 0.001682
grad_step = 000185, loss = 0.001670
grad_step = 000186, loss = 0.001658
grad_step = 000187, loss = 0.001647
grad_step = 000188, loss = 0.001635
grad_step = 000189, loss = 0.001623
grad_step = 000190, loss = 0.001611
grad_step = 000191, loss = 0.001599
grad_step = 000192, loss = 0.001587
grad_step = 000193, loss = 0.001574
grad_step = 000194, loss = 0.001562
grad_step = 000195, loss = 0.001550
grad_step = 000196, loss = 0.001538
grad_step = 000197, loss = 0.001525
grad_step = 000198, loss = 0.001513
grad_step = 000199, loss = 0.001501
grad_step = 000200, loss = 0.001488
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001476
grad_step = 000202, loss = 0.001463
grad_step = 000203, loss = 0.001451
grad_step = 000204, loss = 0.001438
grad_step = 000205, loss = 0.001425
grad_step = 000206, loss = 0.001412
grad_step = 000207, loss = 0.001400
grad_step = 000208, loss = 0.001387
grad_step = 000209, loss = 0.001375
grad_step = 000210, loss = 0.001363
grad_step = 000211, loss = 0.001351
grad_step = 000212, loss = 0.001340
grad_step = 000213, loss = 0.001329
grad_step = 000214, loss = 0.001317
grad_step = 000215, loss = 0.001305
grad_step = 000216, loss = 0.001293
grad_step = 000217, loss = 0.001281
grad_step = 000218, loss = 0.001270
grad_step = 000219, loss = 0.001258
grad_step = 000220, loss = 0.001246
grad_step = 000221, loss = 0.001235
grad_step = 000222, loss = 0.001223
grad_step = 000223, loss = 0.001212
grad_step = 000224, loss = 0.001201
grad_step = 000225, loss = 0.001189
grad_step = 000226, loss = 0.001177
grad_step = 000227, loss = 0.001166
grad_step = 000228, loss = 0.001155
grad_step = 000229, loss = 0.001143
grad_step = 000230, loss = 0.001133
grad_step = 000231, loss = 0.001123
grad_step = 000232, loss = 0.001112
grad_step = 000233, loss = 0.001102
grad_step = 000234, loss = 0.001091
grad_step = 000235, loss = 0.001082
grad_step = 000236, loss = 0.001072
grad_step = 000237, loss = 0.001062
grad_step = 000238, loss = 0.001053
grad_step = 000239, loss = 0.001044
grad_step = 000240, loss = 0.001034
grad_step = 000241, loss = 0.001025
grad_step = 000242, loss = 0.001016
grad_step = 000243, loss = 0.001007
grad_step = 000244, loss = 0.000998
grad_step = 000245, loss = 0.000990
grad_step = 000246, loss = 0.000981
grad_step = 000247, loss = 0.000972
grad_step = 000248, loss = 0.000963
grad_step = 000249, loss = 0.000955
grad_step = 000250, loss = 0.000947
grad_step = 000251, loss = 0.000938
grad_step = 000252, loss = 0.000930
grad_step = 000253, loss = 0.000922
grad_step = 000254, loss = 0.000914
grad_step = 000255, loss = 0.000906
grad_step = 000256, loss = 0.000898
grad_step = 000257, loss = 0.000890
grad_step = 000258, loss = 0.000882
grad_step = 000259, loss = 0.000875
grad_step = 000260, loss = 0.000867
grad_step = 000261, loss = 0.000860
grad_step = 000262, loss = 0.000852
grad_step = 000263, loss = 0.000845
grad_step = 000264, loss = 0.000839
grad_step = 000265, loss = 0.000832
grad_step = 000266, loss = 0.000825
grad_step = 000267, loss = 0.000818
grad_step = 000268, loss = 0.000811
grad_step = 000269, loss = 0.000804
grad_step = 000270, loss = 0.000799
grad_step = 000271, loss = 0.000792
grad_step = 000272, loss = 0.000785
grad_step = 000273, loss = 0.000779
grad_step = 000274, loss = 0.000773
grad_step = 000275, loss = 0.000767
grad_step = 000276, loss = 0.000761
grad_step = 000277, loss = 0.000755
grad_step = 000278, loss = 0.000749
grad_step = 000279, loss = 0.000743
grad_step = 000280, loss = 0.000738
grad_step = 000281, loss = 0.000732
grad_step = 000282, loss = 0.000726
grad_step = 000283, loss = 0.000720
grad_step = 000284, loss = 0.000714
grad_step = 000285, loss = 0.000709
grad_step = 000286, loss = 0.000704
grad_step = 000287, loss = 0.000698
grad_step = 000288, loss = 0.000693
grad_step = 000289, loss = 0.000689
grad_step = 000290, loss = 0.000686
grad_step = 000291, loss = 0.000683
grad_step = 000292, loss = 0.000677
grad_step = 000293, loss = 0.000668
grad_step = 000294, loss = 0.000665
grad_step = 000295, loss = 0.000663
grad_step = 000296, loss = 0.000657
grad_step = 000297, loss = 0.000650
grad_step = 000298, loss = 0.000647
grad_step = 000299, loss = 0.000644
grad_step = 000300, loss = 0.000638
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000633
grad_step = 000302, loss = 0.000631
grad_step = 000303, loss = 0.000626
grad_step = 000304, loss = 0.000620
grad_step = 000305, loss = 0.000617
grad_step = 000306, loss = 0.000614
grad_step = 000307, loss = 0.000609
grad_step = 000308, loss = 0.000605
grad_step = 000309, loss = 0.000601
grad_step = 000310, loss = 0.000597
grad_step = 000311, loss = 0.000593
grad_step = 000312, loss = 0.000589
grad_step = 000313, loss = 0.000586
grad_step = 000314, loss = 0.000582
grad_step = 000315, loss = 0.000579
grad_step = 000316, loss = 0.000575
grad_step = 000317, loss = 0.000571
grad_step = 000318, loss = 0.000568
grad_step = 000319, loss = 0.000565
grad_step = 000320, loss = 0.000562
grad_step = 000321, loss = 0.000558
grad_step = 000322, loss = 0.000555
grad_step = 000323, loss = 0.000551
grad_step = 000324, loss = 0.000548
grad_step = 000325, loss = 0.000546
grad_step = 000326, loss = 0.000543
grad_step = 000327, loss = 0.000540
grad_step = 000328, loss = 0.000536
grad_step = 000329, loss = 0.000533
grad_step = 000330, loss = 0.000530
grad_step = 000331, loss = 0.000526
grad_step = 000332, loss = 0.000524
grad_step = 000333, loss = 0.000522
grad_step = 000334, loss = 0.000519
grad_step = 000335, loss = 0.000516
grad_step = 000336, loss = 0.000513
grad_step = 000337, loss = 0.000510
grad_step = 000338, loss = 0.000507
grad_step = 000339, loss = 0.000505
grad_step = 000340, loss = 0.000503
grad_step = 000341, loss = 0.000502
grad_step = 000342, loss = 0.000499
grad_step = 000343, loss = 0.000496
grad_step = 000344, loss = 0.000492
grad_step = 000345, loss = 0.000489
grad_step = 000346, loss = 0.000487
grad_step = 000347, loss = 0.000485
grad_step = 000348, loss = 0.000483
grad_step = 000349, loss = 0.000481
grad_step = 000350, loss = 0.000479
grad_step = 000351, loss = 0.000477
grad_step = 000352, loss = 0.000474
grad_step = 000353, loss = 0.000471
grad_step = 000354, loss = 0.000469
grad_step = 000355, loss = 0.000467
grad_step = 000356, loss = 0.000465
grad_step = 000357, loss = 0.000463
grad_step = 000358, loss = 0.000462
grad_step = 000359, loss = 0.000461
grad_step = 000360, loss = 0.000460
grad_step = 000361, loss = 0.000462
grad_step = 000362, loss = 0.000463
grad_step = 000363, loss = 0.000463
grad_step = 000364, loss = 0.000458
grad_step = 000365, loss = 0.000451
grad_step = 000366, loss = 0.000447
grad_step = 000367, loss = 0.000447
grad_step = 000368, loss = 0.000450
grad_step = 000369, loss = 0.000449
grad_step = 000370, loss = 0.000443
grad_step = 000371, loss = 0.000438
grad_step = 000372, loss = 0.000437
grad_step = 000373, loss = 0.000439
grad_step = 000374, loss = 0.000438
grad_step = 000375, loss = 0.000434
grad_step = 000376, loss = 0.000430
grad_step = 000377, loss = 0.000429
grad_step = 000378, loss = 0.000430
grad_step = 000379, loss = 0.000429
grad_step = 000380, loss = 0.000426
grad_step = 000381, loss = 0.000423
grad_step = 000382, loss = 0.000422
grad_step = 000383, loss = 0.000420
grad_step = 000384, loss = 0.000420
grad_step = 000385, loss = 0.000419
grad_step = 000386, loss = 0.000417
grad_step = 000387, loss = 0.000416
grad_step = 000388, loss = 0.000414
grad_step = 000389, loss = 0.000412
grad_step = 000390, loss = 0.000411
grad_step = 000391, loss = 0.000409
grad_step = 000392, loss = 0.000408
grad_step = 000393, loss = 0.000407
grad_step = 000394, loss = 0.000407
grad_step = 000395, loss = 0.000406
grad_step = 000396, loss = 0.000405
grad_step = 000397, loss = 0.000406
grad_step = 000398, loss = 0.000407
grad_step = 000399, loss = 0.000408
grad_step = 000400, loss = 0.000411
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000412
grad_step = 000402, loss = 0.000411
grad_step = 000403, loss = 0.000405
grad_step = 000404, loss = 0.000398
grad_step = 000405, loss = 0.000393
grad_step = 000406, loss = 0.000395
grad_step = 000407, loss = 0.000398
grad_step = 000408, loss = 0.000398
grad_step = 000409, loss = 0.000394
grad_step = 000410, loss = 0.000390
grad_step = 000411, loss = 0.000387
grad_step = 000412, loss = 0.000388
grad_step = 000413, loss = 0.000390
grad_step = 000414, loss = 0.000390
grad_step = 000415, loss = 0.000388
grad_step = 000416, loss = 0.000385
grad_step = 000417, loss = 0.000382
grad_step = 000418, loss = 0.000381
grad_step = 000419, loss = 0.000381
grad_step = 000420, loss = 0.000381
grad_step = 000421, loss = 0.000381
grad_step = 000422, loss = 0.000381
grad_step = 000423, loss = 0.000381
grad_step = 000424, loss = 0.000380
grad_step = 000425, loss = 0.000378
grad_step = 000426, loss = 0.000376
grad_step = 000427, loss = 0.000374
grad_step = 000428, loss = 0.000373
grad_step = 000429, loss = 0.000371
grad_step = 000430, loss = 0.000370
grad_step = 000431, loss = 0.000369
grad_step = 000432, loss = 0.000369
grad_step = 000433, loss = 0.000368
grad_step = 000434, loss = 0.000368
grad_step = 000435, loss = 0.000369
grad_step = 000436, loss = 0.000373
grad_step = 000437, loss = 0.000381
grad_step = 000438, loss = 0.000396
grad_step = 000439, loss = 0.000412
grad_step = 000440, loss = 0.000417
grad_step = 000441, loss = 0.000404
grad_step = 000442, loss = 0.000383
grad_step = 000443, loss = 0.000373
grad_step = 000444, loss = 0.000374
grad_step = 000445, loss = 0.000380
grad_step = 000446, loss = 0.000381
grad_step = 000447, loss = 0.000373
grad_step = 000448, loss = 0.000364
grad_step = 000449, loss = 0.000361
grad_step = 000450, loss = 0.000368
grad_step = 000451, loss = 0.000371
grad_step = 000452, loss = 0.000360
grad_step = 000453, loss = 0.000354
grad_step = 000454, loss = 0.000358
grad_step = 000455, loss = 0.000362
grad_step = 000456, loss = 0.000358
grad_step = 000457, loss = 0.000351
grad_step = 000458, loss = 0.000351
grad_step = 000459, loss = 0.000354
grad_step = 000460, loss = 0.000353
grad_step = 000461, loss = 0.000350
grad_step = 000462, loss = 0.000349
grad_step = 000463, loss = 0.000348
grad_step = 000464, loss = 0.000347
grad_step = 000465, loss = 0.000345
grad_step = 000466, loss = 0.000346
grad_step = 000467, loss = 0.000346
grad_step = 000468, loss = 0.000345
grad_step = 000469, loss = 0.000342
grad_step = 000470, loss = 0.000341
grad_step = 000471, loss = 0.000341
grad_step = 000472, loss = 0.000341
grad_step = 000473, loss = 0.000343
grad_step = 000474, loss = 0.000342
grad_step = 000475, loss = 0.000341
grad_step = 000476, loss = 0.000340
grad_step = 000477, loss = 0.000338
grad_step = 000478, loss = 0.000338
grad_step = 000479, loss = 0.000336
grad_step = 000480, loss = 0.000334
grad_step = 000481, loss = 0.000334
grad_step = 000482, loss = 0.000333
grad_step = 000483, loss = 0.000333
grad_step = 000484, loss = 0.000334
grad_step = 000485, loss = 0.000333
grad_step = 000486, loss = 0.000333
grad_step = 000487, loss = 0.000334
grad_step = 000488, loss = 0.000337
grad_step = 000489, loss = 0.000347
grad_step = 000490, loss = 0.000366
grad_step = 000491, loss = 0.000390
grad_step = 000492, loss = 0.000402
grad_step = 000493, loss = 0.000393
grad_step = 000494, loss = 0.000377
grad_step = 000495, loss = 0.000353
grad_step = 000496, loss = 0.000334
grad_step = 000497, loss = 0.000340
grad_step = 000498, loss = 0.000360
grad_step = 000499, loss = 0.000362
grad_step = 000500, loss = 0.000335
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000321
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
[[0.85373074 0.8834134  0.94472927 0.9606887  1.0192645 ]
 [0.8619802  0.9132054  0.9559214  1.0231112  1.003846  ]
 [0.9069703  0.9305068  1.0091195  0.9881401  0.9600017 ]
 [0.9193827  0.98234576 1.0037845  0.93596566 0.9243565 ]
 [0.97952485 0.9844659  0.9639025  0.9134026  0.8520766 ]
 [0.97142315 0.9530362  0.9280174  0.8441962  0.85526794]
 [0.9365634  0.9197234  0.860021   0.85154426 0.82018363]
 [0.9066441  0.85688746 0.8596046  0.8174726  0.839567  ]
 [0.82021445 0.8385164  0.82603335 0.8335718  0.86075824]
 [0.8218155  0.8179079  0.8478414  0.86704534 0.83474195]
 [0.80873704 0.8363024  0.8627648  0.8342846  0.92282635]
 [0.8065992  0.84452283 0.84275246 0.89880705 0.94256896]
 [0.8417888  0.8684709  0.9336381  0.96506876 1.0144719 ]
 [0.8571911  0.9157262  0.95260406 1.0237542  0.99042815]
 [0.9115667  0.93546796 1.0043813  0.9756944  0.93943363]
 [0.92645097 0.98616105 0.99701643 0.9156303  0.90516394]
 [0.98056906 0.9817439  0.9487231  0.8914903  0.82589865]
 [0.9615528  0.9321101  0.9035499  0.83000576 0.83973473]
 [0.92809093 0.9044132  0.8331607  0.84096694 0.814603  ]
 [0.9090503  0.8508289  0.8420596  0.81622034 0.83790785]
 [0.8326025  0.8438301  0.8226127  0.83341646 0.86582273]
 [0.8358022  0.82676476 0.84658206 0.8757945  0.8410413 ]
 [0.82311106 0.8454672  0.8692751  0.84422374 0.92412406]
 [0.81408304 0.8485455  0.8477774  0.9020652  0.94049454]
 [0.8566991  0.88631713 0.94399303 0.96369517 1.0262483 ]
 [0.8667174  0.9168944  0.95822895 1.0336375  1.015336  ]
 [0.9131503  0.9383724  1.0151234  1.0023075  0.97244924]
 [0.9292633  0.993112   1.0169408  0.9487517  0.93635696]
 [0.98836285 0.9982685  0.97801596 0.92470616 0.8597138 ]
 [0.9790905  0.9614935  0.9381688  0.853384   0.8599784 ]
 [0.9431729  0.92710143 0.865065   0.8576619  0.82750416]]

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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..03b14c2
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 11104293.33B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 409131.25B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 4256768/440473133 [00:00<00:10, 42556255.38B/s]  2%|▏         | 9434112/440473133 [00:00<00:09, 44953391.65B/s]  3%|▎         | 14769152/440473133 [00:00<00:09, 47180104.03B/s]  5%|▍         | 20096000/440473133 [00:00<00:08, 48854185.05B/s]  6%|▌         | 25521152/440473133 [00:00<00:08, 50355863.12B/s]  7%|▋         | 30856192/440473133 [00:00<00:07, 51215698.82B/s]  8%|▊         | 36382720/440473133 [00:00<00:07, 52366304.68B/s]  9%|▉         | 41682944/440473133 [00:00<00:07, 52553591.18B/s] 11%|█         | 47066112/440473133 [00:00<00:07, 52929832.88B/s] 12%|█▏        | 52556800/440473133 [00:01<00:07, 53502702.75B/s] 13%|█▎        | 58009600/440473133 [00:01<00:07, 53804602.69B/s] 14%|█▍        | 63364096/440473133 [00:01<00:07, 53726191.39B/s] 16%|█▌        | 68773888/440473133 [00:01<00:06, 53836459.68B/s] 17%|█▋        | 74301440/440473133 [00:01<00:06, 54256606.37B/s] 18%|█▊        | 79795200/440473133 [00:01<00:06, 54458329.72B/s] 19%|█▉        | 85239808/440473133 [00:01<00:06, 54451826.23B/s] 21%|██        | 90673152/440473133 [00:01<00:06, 54202370.25B/s] 22%|██▏       | 96086016/440473133 [00:01<00:06, 53488192.62B/s] 23%|██▎       | 101435392/440473133 [00:01<00:06, 53487965.28B/s] 24%|██▍       | 107034624/440473133 [00:02<00:06, 54214881.14B/s] 26%|██▌       | 112457728/440473133 [00:02<00:06, 54218618.91B/s] 27%|██▋       | 117880832/440473133 [00:02<00:05, 54004769.22B/s] 28%|██▊       | 123535360/440473133 [00:02<00:05, 54741692.63B/s] 29%|██▉       | 129102848/440473133 [00:02<00:05, 55011597.38B/s] 31%|███       | 134643712/440473133 [00:02<00:05, 55129538.60B/s] 32%|███▏      | 140184576/440473133 [00:02<00:05, 55212831.67B/s] 33%|███▎      | 145708032/440473133 [00:02<00:05, 55072751.44B/s] 34%|███▍      | 151361536/440473133 [00:02<00:05, 55500292.02B/s] 36%|███▌      | 156963840/440473133 [00:02<00:05, 55651913.30B/s] 37%|███▋      | 162530304/440473133 [00:03<00:05, 55459818.24B/s] 38%|███▊      | 168077312/440473133 [00:03<00:04, 54643734.17B/s] 39%|███▉      | 173545472/440473133 [00:03<00:04, 54559268.95B/s] 41%|████      | 179087360/440473133 [00:03<00:04, 54809352.65B/s] 42%|████▏     | 184726528/440473133 [00:03<00:04, 55273152.02B/s] 43%|████▎     | 190346240/440473133 [00:03<00:04, 55542557.78B/s] 44%|████▍     | 195902464/440473133 [00:03<00:04, 55224109.49B/s] 46%|████▌     | 201426944/440473133 [00:03<00:04, 53437709.20B/s] 47%|████▋     | 206784512/440473133 [00:03<00:04, 53111748.31B/s] 48%|████▊     | 212322304/440473133 [00:03<00:04, 53771823.43B/s] 49%|████▉     | 217765888/440473133 [00:04<00:04, 53965739.27B/s] 51%|█████     | 223212544/440473133 [00:04<00:04, 54111031.98B/s] 52%|█████▏    | 228628480/440473133 [00:04<00:03, 54057721.93B/s] 53%|█████▎    | 234060800/440473133 [00:04<00:03, 54135718.76B/s] 54%|█████▍    | 239613952/440473133 [00:04<00:03, 54545173.09B/s] 56%|█████▌    | 245070848/440473133 [00:04<00:03, 53569437.70B/s] 57%|█████▋    | 250433536/440473133 [00:04<00:03, 52363723.87B/s] 58%|█████▊    | 255680512/440473133 [00:04<00:03, 51375468.31B/s] 59%|█████▉    | 260829184/440473133 [00:04<00:03, 51160463.00B/s] 60%|██████    | 265953280/440473133 [00:04<00:03, 50328649.71B/s] 62%|██████▏   | 270995456/440473133 [00:05<00:03, 45283341.72B/s] 63%|██████▎   | 275625984/440473133 [00:05<00:03, 45535438.50B/s] 64%|██████▎   | 280667136/440473133 [00:05<00:03, 46893556.18B/s] 65%|██████▍   | 285706240/440473133 [00:05<00:03, 47889288.05B/s] 66%|██████▌   | 290542592/440473133 [00:05<00:03, 47999667.07B/s] 67%|██████▋   | 295611392/440473133 [00:05<00:02, 48773941.54B/s] 68%|██████▊   | 300931072/440473133 [00:05<00:02, 50017074.02B/s] 69%|██████▉   | 305959936/440473133 [00:05<00:02, 49764715.32B/s] 71%|███████   | 311063552/440473133 [00:05<00:02, 50137523.27B/s] 72%|███████▏  | 316421120/440473133 [00:05<00:02, 51114449.32B/s] 73%|███████▎  | 321955840/440473133 [00:06<00:02, 52312203.31B/s] 74%|███████▍  | 327383040/440473133 [00:06<00:02, 52883363.49B/s] 76%|███████▌  | 332974080/440473133 [00:06<00:01, 53754028.11B/s] 77%|███████▋  | 338416640/440473133 [00:06<00:01, 53948527.72B/s] 78%|███████▊  | 343821312/440473133 [00:06<00:01, 53398653.64B/s] 79%|███████▉  | 349169664/440473133 [00:06<00:01, 53131728.34B/s] 80%|████████  | 354489344/440473133 [00:06<00:01, 53070843.66B/s] 82%|████████▏ | 359800832/440473133 [00:06<00:01, 52669195.77B/s] 83%|████████▎ | 365071360/440473133 [00:06<00:01, 52535206.85B/s] 84%|████████▍ | 370327552/440473133 [00:07<00:01, 52246771.65B/s] 85%|████████▌ | 375843840/440473133 [00:07<00:01, 53087226.07B/s] 87%|████████▋ | 381389824/440473133 [00:07<00:01, 53755883.11B/s] 88%|████████▊ | 386770944/440473133 [00:07<00:01, 52697778.51B/s] 89%|████████▉ | 392213504/440473133 [00:07<00:00, 53200306.72B/s] 90%|█████████ | 397736960/440473133 [00:07<00:00, 53794351.40B/s] 92%|█████████▏| 403195904/440473133 [00:07<00:00, 54028194.09B/s] 93%|█████████▎| 408637440/440473133 [00:07<00:00, 54138693.62B/s] 94%|█████████▍| 414055424/440473133 [00:07<00:00, 53671365.34B/s] 95%|█████████▌| 419639296/440473133 [00:07<00:00, 54303334.75B/s] 97%|█████████▋| 425194496/440473133 [00:08<00:00, 54666820.70B/s] 98%|█████████▊| 430664704/440473133 [00:08<00:00, 54578260.02B/s] 99%|█████████▉| 436213760/440473133 [00:08<00:00, 54848606.28B/s]100%|██████████| 440473133/440473133 [00:08<00:00, 53065627.33B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
  16384/7094233 [..............................] - ETA: 25s
  65536/7094233 [..............................] - ETA: 13s
 139264/7094233 [..............................] - ETA: 8s 
 303104/7094233 [>.............................] - ETA: 5s
 655360/7094233 [=>............................] - ETA: 2s
1302528/7094233 [====>.........................] - ETA: 1s
2498560/7094233 [=========>....................] - ETA: 0s
4866048/7094233 [===================>..........] - ETA: 0s
7094272/7094233 [==============================] - 1s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   0%|          | 2/2118 [00:00<02:02, 17.32it/s]Processing text_left with encode:  22%|██▏       | 465/2118 [00:00<01:06, 24.71it/s]Processing text_left with encode:  43%|████▎     | 911/2118 [00:00<00:34, 35.21it/s]Processing text_left with encode:  60%|██████    | 1281/2118 [00:00<00:16, 50.10it/s]Processing text_left with encode:  82%|████████▏ | 1737/2118 [00:00<00:05, 71.23it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3504.97it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 150/18841 [00:00<00:29, 638.26it/s]Processing text_right with encode:   2%|▏         | 324/18841 [00:00<00:23, 787.57it/s]Processing text_right with encode:   3%|▎         | 498/18841 [00:00<00:19, 942.20it/s]Processing text_right with encode:   4%|▎         | 673/18841 [00:00<00:16, 1093.27it/s]Processing text_right with encode:   4%|▍         | 835/18841 [00:00<00:14, 1210.67it/s]Processing text_right with encode:   5%|▌         | 994/18841 [00:00<00:13, 1301.16it/s]Processing text_right with encode:   6%|▌         | 1155/18841 [00:00<00:12, 1377.94it/s]Processing text_right with encode:   7%|▋         | 1329/18841 [00:00<00:11, 1468.01it/s]Processing text_right with encode:   8%|▊         | 1490/18841 [00:01<00:11, 1507.38it/s]Processing text_right with encode:   9%|▉         | 1671/18841 [00:01<00:10, 1586.89it/s]Processing text_right with encode:  10%|▉         | 1835/18841 [00:01<00:10, 1595.25it/s]Processing text_right with encode:  11%|█         | 2005/18841 [00:01<00:10, 1625.01it/s]Processing text_right with encode:  12%|█▏        | 2183/18841 [00:01<00:10, 1665.74it/s]Processing text_right with encode:  13%|█▎        | 2360/18841 [00:01<00:09, 1694.63it/s]Processing text_right with encode:  13%|█▎        | 2532/18841 [00:01<00:09, 1647.30it/s]Processing text_right with encode:  14%|█▍        | 2699/18841 [00:01<00:09, 1636.08it/s]Processing text_right with encode:  15%|█▌        | 2884/18841 [00:01<00:09, 1692.58it/s]Processing text_right with encode:  16%|█▌        | 3055/18841 [00:01<00:09, 1630.76it/s]Processing text_right with encode:  17%|█▋        | 3220/18841 [00:02<00:09, 1634.14it/s]Processing text_right with encode:  18%|█▊        | 3385/18841 [00:02<00:09, 1635.97it/s]Processing text_right with encode:  19%|█▉        | 3550/18841 [00:02<00:09, 1598.73it/s]Processing text_right with encode:  20%|█▉        | 3711/18841 [00:02<00:09, 1594.06it/s]Processing text_right with encode:  21%|██        | 3886/18841 [00:02<00:09, 1636.04it/s]Processing text_right with encode:  22%|██▏       | 4058/18841 [00:02<00:08, 1658.32it/s]Processing text_right with encode:  22%|██▏       | 4225/18841 [00:02<00:09, 1613.40it/s]Processing text_right with encode:  23%|██▎       | 4391/18841 [00:02<00:08, 1625.39it/s]Processing text_right with encode:  24%|██▍       | 4554/18841 [00:02<00:08, 1623.95it/s]Processing text_right with encode:  25%|██▌       | 4722/18841 [00:02<00:08, 1639.99it/s]Processing text_right with encode:  26%|██▌       | 4904/18841 [00:03<00:08, 1686.30it/s]Processing text_right with encode:  27%|██▋       | 5084/18841 [00:03<00:08, 1718.34it/s]Processing text_right with encode:  28%|██▊       | 5264/18841 [00:03<00:07, 1739.97it/s]Processing text_right with encode:  29%|██▉       | 5439/18841 [00:03<00:08, 1672.01it/s]Processing text_right with encode:  30%|██▉       | 5608/18841 [00:03<00:08, 1627.26it/s]Processing text_right with encode:  31%|███       | 5778/18841 [00:03<00:07, 1647.89it/s]Processing text_right with encode:  32%|███▏      | 5946/18841 [00:03<00:07, 1656.37it/s]Processing text_right with encode:  32%|███▏      | 6113/18841 [00:03<00:07, 1634.69it/s]Processing text_right with encode:  33%|███▎      | 6278/18841 [00:03<00:07, 1638.57it/s]Processing text_right with encode:  34%|███▍      | 6462/18841 [00:04<00:07, 1690.94it/s]Processing text_right with encode:  35%|███▌      | 6641/18841 [00:04<00:07, 1716.85it/s]Processing text_right with encode:  36%|███▌      | 6814/18841 [00:04<00:07, 1686.13it/s]Processing text_right with encode:  37%|███▋      | 6984/18841 [00:04<00:07, 1626.07it/s]Processing text_right with encode:  38%|███▊      | 7154/18841 [00:04<00:07, 1645.90it/s]Processing text_right with encode:  39%|███▉      | 7334/18841 [00:04<00:06, 1684.16it/s]Processing text_right with encode:  40%|███▉      | 7504/18841 [00:04<00:06, 1672.24it/s]Processing text_right with encode:  41%|████      | 7672/18841 [00:04<00:06, 1627.21it/s]Processing text_right with encode:  42%|████▏     | 7836/18841 [00:04<00:06, 1610.07it/s]Processing text_right with encode:  42%|████▏     | 7998/18841 [00:04<00:06, 1564.26it/s]Processing text_right with encode:  43%|████▎     | 8157/18841 [00:05<00:06, 1571.64it/s]Processing text_right with encode:  44%|████▍     | 8317/18841 [00:05<00:06, 1578.40it/s]Processing text_right with encode:  45%|████▌     | 8482/18841 [00:05<00:06, 1598.85it/s]Processing text_right with encode:  46%|████▌     | 8643/18841 [00:05<00:06, 1597.24it/s]Processing text_right with encode:  47%|████▋     | 8821/18841 [00:05<00:06, 1647.59it/s]Processing text_right with encode:  48%|████▊     | 8987/18841 [00:05<00:06, 1622.15it/s]Processing text_right with encode:  49%|████▊     | 9159/18841 [00:05<00:05, 1648.03it/s]Processing text_right with encode:  50%|████▉     | 9330/18841 [00:05<00:05, 1665.17it/s]Processing text_right with encode:  50%|█████     | 9505/18841 [00:05<00:05, 1686.24it/s]Processing text_right with encode:  51%|█████▏    | 9675/18841 [00:05<00:05, 1687.39it/s]Processing text_right with encode:  52%|█████▏    | 9844/18841 [00:06<00:05, 1679.52it/s]Processing text_right with encode:  53%|█████▎    | 10036/18841 [00:06<00:05, 1741.69it/s]Processing text_right with encode:  54%|█████▍    | 10211/18841 [00:06<00:05, 1683.43it/s]Processing text_right with encode:  55%|█████▌    | 10388/18841 [00:06<00:04, 1707.11it/s]Processing text_right with encode:  56%|█████▌    | 10585/18841 [00:06<00:04, 1776.34it/s]Processing text_right with encode:  57%|█████▋    | 10764/18841 [00:06<00:04, 1749.43it/s]Processing text_right with encode:  58%|█████▊    | 10940/18841 [00:06<00:04, 1726.48it/s]Processing text_right with encode:  59%|█████▉    | 11114/18841 [00:06<00:04, 1654.04it/s]Processing text_right with encode:  60%|█████▉    | 11281/18841 [00:06<00:04, 1633.23it/s]Processing text_right with encode:  61%|██████    | 11446/18841 [00:07<00:04, 1626.61it/s]Processing text_right with encode:  62%|██████▏   | 11610/18841 [00:07<00:04, 1560.65it/s]Processing text_right with encode:  63%|██████▎   | 11784/18841 [00:07<00:04, 1609.51it/s]Processing text_right with encode:  63%|██████▎   | 11948/18841 [00:07<00:04, 1617.48it/s]Processing text_right with encode:  64%|██████▍   | 12118/18841 [00:07<00:04, 1638.94it/s]Processing text_right with encode:  65%|██████▌   | 12283/18841 [00:07<00:04, 1628.75it/s]Processing text_right with encode:  66%|██████▌   | 12453/18841 [00:07<00:03, 1647.90it/s]Processing text_right with encode:  67%|██████▋   | 12619/18841 [00:07<00:03, 1644.74it/s]Processing text_right with encode:  68%|██████▊   | 12785/18841 [00:07<00:03, 1648.00it/s]Processing text_right with encode:  69%|██████▉   | 12954/18841 [00:07<00:03, 1659.36it/s]Processing text_right with encode:  70%|██████▉   | 13124/18841 [00:08<00:03, 1669.18it/s]Processing text_right with encode:  71%|███████   | 13306/18841 [00:08<00:03, 1709.22it/s]Processing text_right with encode:  72%|███████▏  | 13478/18841 [00:08<00:03, 1707.99it/s]Processing text_right with encode:  72%|███████▏  | 13655/18841 [00:08<00:03, 1726.08it/s]Processing text_right with encode:  73%|███████▎  | 13828/18841 [00:08<00:02, 1722.41it/s]Processing text_right with encode:  74%|███████▍  | 14002/18841 [00:08<00:02, 1726.28it/s]Processing text_right with encode:  75%|███████▌  | 14175/18841 [00:08<00:02, 1713.66it/s]Processing text_right with encode:  76%|███████▌  | 14347/18841 [00:08<00:02, 1693.65it/s]Processing text_right with encode:  77%|███████▋  | 14517/18841 [00:08<00:02, 1677.30it/s]Processing text_right with encode:  78%|███████▊  | 14705/18841 [00:08<00:02, 1730.22it/s]Processing text_right with encode:  79%|███████▉  | 14879/18841 [00:09<00:02, 1688.81it/s]Processing text_right with encode:  80%|███████▉  | 15049/18841 [00:09<00:02, 1685.37it/s]Processing text_right with encode:  81%|████████  | 15222/18841 [00:09<00:02, 1697.91it/s]Processing text_right with encode:  82%|████████▏ | 15393/18841 [00:09<00:02, 1661.64it/s]Processing text_right with encode:  83%|████████▎ | 15567/18841 [00:09<00:01, 1683.23it/s]Processing text_right with encode:  84%|████████▎ | 15736/18841 [00:09<00:01, 1639.50it/s]Processing text_right with encode:  84%|████████▍ | 15901/18841 [00:09<00:01, 1628.52it/s]Processing text_right with encode:  85%|████████▌ | 16067/18841 [00:09<00:01, 1634.24it/s]Processing text_right with encode:  86%|████████▌ | 16233/18841 [00:09<00:01, 1641.06it/s]Processing text_right with encode:  87%|████████▋ | 16405/18841 [00:10<00:01, 1663.67it/s]Processing text_right with encode:  88%|████████▊ | 16575/18841 [00:10<00:01, 1673.15it/s]Processing text_right with encode:  89%|████████▉ | 16747/18841 [00:10<00:01, 1686.29it/s]Processing text_right with encode:  90%|████████▉ | 16916/18841 [00:10<00:01, 1653.67it/s]Processing text_right with encode:  91%|█████████ | 17084/18841 [00:10<00:01, 1660.38it/s]Processing text_right with encode:  92%|█████████▏| 17256/18841 [00:10<00:00, 1672.04it/s]Processing text_right with encode:  92%|█████████▏| 17424/18841 [00:10<00:00, 1670.60it/s]Processing text_right with encode:  93%|█████████▎| 17599/18841 [00:10<00:00, 1691.25it/s]Processing text_right with encode:  94%|█████████▍| 17776/18841 [00:10<00:00, 1712.61it/s]Processing text_right with encode:  95%|█████████▌| 17948/18841 [00:10<00:00, 1702.49it/s]Processing text_right with encode:  96%|█████████▋| 18147/18841 [00:11<00:00, 1778.02it/s]Processing text_right with encode:  97%|█████████▋| 18326/18841 [00:11<00:00, 1730.55it/s]Processing text_right with encode:  98%|█████████▊| 18513/18841 [00:11<00:00, 1768.67it/s]Processing text_right with encode:  99%|█████████▉| 18691/18841 [00:11<00:00, 1756.85it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:11<00:00, 1650.38it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 629770.02it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 832042.30it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  71%|███████   | 447/633 [00:00<00:00, 4465.34it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4380.56it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 164/5961 [00:00<00:03, 1625.11it/s]Processing text_right with encode:   6%|▌         | 341/5961 [00:00<00:03, 1664.20it/s]Processing text_right with encode:   8%|▊         | 506/5961 [00:00<00:03, 1658.97it/s]Processing text_right with encode:  11%|█▏        | 680/5961 [00:00<00:03, 1681.19it/s]Processing text_right with encode:  14%|█▍        | 853/5961 [00:00<00:03, 1691.26it/s]Processing text_right with encode:  17%|█▋        | 1032/5961 [00:00<00:02, 1717.29it/s]Processing text_right with encode:  20%|██        | 1211/5961 [00:00<00:02, 1738.10it/s]Processing text_right with encode:  23%|██▎       | 1386/5961 [00:00<00:02, 1738.27it/s]Processing text_right with encode:  26%|██▌       | 1560/5961 [00:00<00:02, 1737.43it/s]Processing text_right with encode:  29%|██▉       | 1728/5961 [00:01<00:02, 1683.05it/s]Processing text_right with encode:  32%|███▏      | 1908/5961 [00:01<00:02, 1714.73it/s]Processing text_right with encode:  35%|███▍      | 2077/5961 [00:01<00:02, 1695.71it/s]Processing text_right with encode:  38%|███▊      | 2258/5961 [00:01<00:02, 1720.31it/s]Processing text_right with encode:  41%|████      | 2429/5961 [00:01<00:02, 1654.63it/s]Processing text_right with encode:  44%|████▍     | 2614/5961 [00:01<00:01, 1706.99it/s]Processing text_right with encode:  47%|████▋     | 2801/5961 [00:01<00:01, 1752.64it/s]Processing text_right with encode:  50%|████▉     | 2977/5961 [00:01<00:01, 1748.95it/s]Processing text_right with encode:  53%|█████▎    | 3163/5961 [00:01<00:01, 1775.68it/s]Processing text_right with encode:  56%|█████▌    | 3341/5961 [00:01<00:01, 1735.37it/s]Processing text_right with encode:  59%|█████▉    | 3515/5961 [00:02<00:01, 1686.30it/s]Processing text_right with encode:  62%|██████▏   | 3686/5961 [00:02<00:01, 1691.45it/s]Processing text_right with encode:  65%|██████▍   | 3856/5961 [00:02<00:01, 1669.37it/s]Processing text_right with encode:  68%|██████▊   | 4035/5961 [00:02<00:01, 1699.92it/s]Processing text_right with encode:  71%|███████   | 4206/5961 [00:02<00:01, 1685.21it/s]Processing text_right with encode:  73%|███████▎  | 4376/5961 [00:02<00:00, 1689.15it/s]Processing text_right with encode:  76%|███████▋  | 4546/5961 [00:02<00:00, 1647.45it/s]Processing text_right with encode:  79%|███████▉  | 4712/5961 [00:02<00:00, 1647.42it/s]Processing text_right with encode:  82%|████████▏ | 4878/5961 [00:02<00:00, 1632.33it/s]Processing text_right with encode:  85%|████████▍ | 5042/5961 [00:02<00:00, 1617.94it/s]Processing text_right with encode:  88%|████████▊ | 5226/5961 [00:03<00:00, 1678.48it/s]Processing text_right with encode:  91%|█████████ | 5395/5961 [00:03<00:00, 1679.36it/s]Processing text_right with encode:  93%|█████████▎| 5564/5961 [00:03<00:00, 1678.87it/s]Processing text_right with encode:  96%|█████████▌| 5733/5961 [00:03<00:00, 1639.53it/s]Processing text_right with encode:  99%|█████████▉| 5908/5961 [00:03<00:00, 1670.84it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1698.35it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 447420.70it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 720567.36it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:52<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:52<?, ?it/s, loss=1.098]Epoch 1/1:   1%|          | 1/102 [00:52<1:28:50, 52.78s/it, loss=1.098]Epoch 1/1:   1%|          | 1/102 [01:31<1:28:50, 52.78s/it, loss=1.098]Epoch 1/1:   1%|          | 1/102 [01:31<1:28:50, 52.78s/it, loss=1.129]Epoch 1/1:   2%|▏         | 2/102 [01:31<1:20:44, 48.44s/it, loss=1.129]Epoch 1/1:   2%|▏         | 2/102 [01:51<1:20:44, 48.44s/it, loss=1.129]Epoch 1/1:   2%|▏         | 2/102 [01:51<1:20:44, 48.44s/it, loss=1.197]Epoch 1/1:   3%|▎         | 3/102 [01:51<1:05:50, 39.90s/it, loss=1.197]Killed

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
   03b14c2..025478a  master     -> origin/master
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..025478a
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fadcfaaad90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fadcfaaad90>

  function with postional parmater data_info <function get_dataset_torch at 0x7fadcfaaad90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▌      | 3579904/9912422 [00:00<00:00, 35795319.06it/s]9920512it [00:00, 35303547.07it/s]                             
0it [00:00, ?it/s]32768it [00:00, 589317.90it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 452976.98it/s]1654784it [00:00, 11384002.83it/s]                         
0it [00:00, ?it/s]8192it [00:00, 195213.59it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fadcc103ae8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fadcc103ae8>

  function with postional parmater data_info <function get_dataset_torch at 0x7fadcc103ae8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.002027112583319346 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.010682535409927368 	 Accuracy: 1
model saves at 1 accuracy
Train Epoch: 2 	 Loss: 0.00119727556904157 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.009358981251716613 	 Accuracy: 2
model saves at 2 accuracy

  #### Predict   ##################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fadcc1038c8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fadcc1038c8>

  function with postional parmater data_info <function get_dataset_torch at 0x7fadcc1038c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  #### metrics   ##################################################### 
None

  #### Plot   ######################################################## 

  #### Save  ######################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 

  #### Load   ######################################################## 
<__main__.Model object at 0x7fadcf4515c0>

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
<__main__.Model object at 0x7fadcc14c3c8>

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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..025478a
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py", line 9, in <module>
    from dataloader import DataLoader
ModuleNotFoundError: No module named 'dataloader'

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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..025478a
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
Iteration:   0%|          | 0/29440 [00:00<?, ?it/s][A
Iteration:   0%|          | 1/29440 [00:20<170:10:25, 20.81s/it][A
Iteration:   0%|          | 2/29440 [00:34<152:24:02, 18.64s/it][A
Iteration:   0%|          | 3/29440 [00:48<140:13:37, 17.15s/it][A
Iteration:   0%|          | 4/29440 [01:32<207:49:38, 25.42s/it][A
Iteration:   0%|          | 5/29440 [03:30<433:44:32, 53.05s/it][A
Iteration:   0%|          | 6/29440 [04:31<453:07:05, 55.42s/it][A
Iteration:   0%|          | 7/29440 [05:45<499:14:35, 61.06s/it][A
Iteration:   0%|          | 8/29440 [06:17<427:27:47, 52.29s/it][A
Iteration:   0%|          | 9/29440 [07:14<439:27:54, 53.76s/it][A
Iteration:   0%|          | 10/29440 [10:42<818:36:20, 100.14s/it][A
Iteration:   0%|          | 11/29440 [11:08<635:41:09, 77.76s/it] [A
Iteration:   0%|          | 12/29440 [11:30<498:49:09, 61.02s/it][A
Iteration:   0%|          | 13/29440 [13:07<587:27:57, 71.87s/it][A
Iteration:   0%|          | 14/29440 [14:07<557:53:29, 68.25s/it][A
Iteration:   0%|          | 15/29440 [14:49<493:42:48, 60.40s/it][A
Iteration:   0%|          | 16/29440 [15:16<412:39:02, 50.49s/it][A
Iteration:   0%|          | 17/29440 [17:01<544:28:24, 66.62s/it][A
Iteration:   0%|          | 18/29440 [19:18<718:34:59, 87.92s/it][AKilled

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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..025478a
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py", line 34, in <module>
    "beta_vae": md.model.beta_vae,
AttributeError: module 'mlmodels.model_tch.raw.pytorch_vae' has no attribute 'model'

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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..025478a
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 9c7c43d..025478a
To github.com:arita37/mlmodels_store.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:arita37/mlmodels_store.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.





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
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=c3b940c898889f255e7f4b777655a2970edbf9a2f6e25e1099c925db05f903bf
  Stored in directory: /tmp/pip-ephem-wheel-cache-972zx26n/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
Updating 9c7c43d..025478a
Fast-forward
 deps.txt                                           |    7 +-
 error_list/20200519/list_log_jupyter_20200519.md   | 1816 +++++++++---------
 .../20200519/list_log_pullrequest_20200519.md      |    2 +-
 log_dataloader/log_dataloader.py                   |   38 +-
 log_jupyter/log_jupyter.py                         | 1938 +++++++++-----------
 ...-10_73f54da32a5da4768415eb9105ad096255137679.py |  628 +++++++
 6 files changed, 2361 insertions(+), 2068 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-19-01-10_73f54da32a5da4768415eb9105ad096255137679.py
