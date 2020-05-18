
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
Warning: Permanently added the RSA host key for IP address '140.82.118.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   66f9634..625ab75  master     -> origin/master
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 66f9634..625ab75
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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 66f9634..625ab75
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
Updating 66f9634..625ab75
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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         24          sequence_max[0][0]               
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
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-18 12:15:08.258657: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-18 12:15:08.279969: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-18 12:15:08.280800: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56461ce53ec0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 12:15:08.280818: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 223
Trainable params: 223
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2975 - binary_crossentropy: 2.0901500/500 [==============================] - 1s 1ms/sample - loss: 0.2881 - binary_crossentropy: 1.8109 - val_loss: 0.3021 - val_binary_crossentropy: 2.0774

  #### metrics   #################################################### 
{'MSE': 0.29495231545987016}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         24          sequence_max[0][0]               
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
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
Total params: 223
Trainable params: 223
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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 1s - loss: 0.2723 - binary_crossentropy: 1.5142500/500 [==============================] - 1s 2ms/sample - loss: 0.2772 - binary_crossentropy: 1.6794 - val_loss: 0.2877 - val_binary_crossentropy: 1.9109

  #### metrics   #################################################### 
{'MSE': 0.2822795839190101}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
Total params: 627
Trainable params: 627
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 2ms/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2496 - val_binary_crossentropy: 0.6920

  #### metrics   #################################################### 
{'MSE': 0.24964797355606827}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
Total params: 627
Trainable params: 627
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 418
Trainable params: 418
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3051 - binary_crossentropy: 0.8217500/500 [==============================] - 1s 3ms/sample - loss: 0.2863 - binary_crossentropy: 0.7796 - val_loss: 0.2563 - val_binary_crossentropy: 0.7098

  #### metrics   #################################################### 
{'MSE': 0.2700901773656757}

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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
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
Total params: 203
Trainable params: 203
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2738 - binary_crossentropy: 0.7441500/500 [==============================] - 2s 4ms/sample - loss: 0.2709 - binary_crossentropy: 0.8151 - val_loss: 0.2698 - val_binary_crossentropy: 0.8130

  #### metrics   #################################################### 
{'MSE': 0.26852598074594813}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
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
Total params: 203
Trainable params: 203
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-18 12:16:33.221909: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:16:33.224180: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:16:33.231059: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-18 12:16:33.242310: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-18 12:16:33.244222: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 12:16:33.245984: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:16:33.247628: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2544 - val_binary_crossentropy: 0.7019
2020-05-18 12:16:34.546120: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:16:34.547854: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:16:34.552153: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-18 12:16:34.561197: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-18 12:16:34.563091: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 12:16:34.564616: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:16:34.565881: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25552643831173}

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
2020-05-18 12:16:59.294625: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:16:59.295894: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:16:59.299642: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-18 12:16:59.306369: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-18 12:16:59.307551: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 12:16:59.308552: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:16:59.309481: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2496 - val_binary_crossentropy: 0.6923
2020-05-18 12:17:00.891633: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:17:00.893592: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:17:00.896924: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-18 12:17:00.902561: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-18 12:17:00.903460: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 12:17:00.904244: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:17:00.904991: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2493187289369049}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-18 12:17:36.625805: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:17:36.630608: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:17:36.645632: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-18 12:17:36.670519: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-18 12:17:36.675228: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 12:17:36.679451: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:17:36.683770: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.6985 - binary_crossentropy: 1.8065 - val_loss: 0.2509 - val_binary_crossentropy: 0.6949
2020-05-18 12:17:39.080174: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:17:39.085656: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:17:39.099203: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-18 12:17:39.124842: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-18 12:17:39.129408: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 12:17:39.133432: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 12:17:39.137217: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24212890271087772}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 690
Trainable params: 690
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3085 - binary_crossentropy: 1.5998500/500 [==============================] - 5s 9ms/sample - loss: 0.3081 - binary_crossentropy: 1.4686 - val_loss: 0.2826 - val_binary_crossentropy: 1.0972

  #### metrics   #################################################### 
{'MSE': 0.2942945543764732}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 690
Trainable params: 690
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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 2)         14          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 2)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         16          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         14          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         16          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_5[0][0]           
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
Total params: 278
Trainable params: 278
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3188 - binary_crossentropy: 1.1140500/500 [==============================] - 5s 10ms/sample - loss: 0.3102 - binary_crossentropy: 1.0473 - val_loss: 0.2950 - val_binary_crossentropy: 0.9608

  #### metrics   #################################################### 
{'MSE': 0.2966770286972326}

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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 2)         14          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 2)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         16          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         14          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         16          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_5[0][0]           
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
Total params: 278
Trainable params: 278
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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 6s - loss: 0.2699 - binary_crossentropy: 0.7394500/500 [==============================] - 5s 10ms/sample - loss: 0.2869 - binary_crossentropy: 0.7783 - val_loss: 0.2791 - val_binary_crossentropy: 0.7578

  #### metrics   #################################################### 
{'MSE': 0.28107031961444656}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
regionsequence_sum (InputLayer) [(None, 6)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 9)]          0                                            
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
region_10sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         6           regionsequence_max[0][0]         
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
Total params: 144
Trainable params: 144
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2447 - binary_crossentropy: 0.6805500/500 [==============================] - 6s 13ms/sample - loss: 0.2529 - binary_crossentropy: 0.7509 - val_loss: 0.2502 - val_binary_crossentropy: 0.7193

  #### metrics   #################################################### 
{'MSE': 0.25133568164252224}

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
regionsequence_mean (InputLayer [(None, 9)]          0                                            
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
region_10sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         6           regionsequence_max[0][0]         
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
Total params: 144
Trainable params: 144
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 8s - loss: 0.3054 - binary_crossentropy: 0.8291500/500 [==============================] - 6s 12ms/sample - loss: 0.3014 - binary_crossentropy: 0.8179 - val_loss: 0.2920 - val_binary_crossentropy: 0.7935

  #### metrics   #################################################### 
{'MSE': 0.29480155968075594}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         4           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_11[0][0]                    
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
Total params: 2,916
Trainable params: 2,836
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2956 - binary_crossentropy: 0.8113500/500 [==============================] - 7s 14ms/sample - loss: 0.3032 - binary_crossentropy: 0.8217 - val_loss: 0.3180 - val_binary_crossentropy: 0.8755

  #### metrics   #################################################### 
{'MSE': 0.30668691675503645}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         4           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_11[0][0]                    
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
Total params: 2,916
Trainable params: 2,836
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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 66f9634..625ab75
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
Updating 66f9634..625ab75
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
Updating 66f9634..625ab75
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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 66f9634..625ab75
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

2020-05-18 12:27:10.805823: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-18 12:27:10.811214: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-18 12:27:10.811356: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55df884e2180 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 12:27:10.811371: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3847
256/354 [====================>.........] - ETA: 3s - loss: 1.2227
354/354 [==============================] - 16s 44ms/step - loss: 1.4134 - val_loss: 2.1136

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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 66f9634..625ab75
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
Updating 66f9634..625ab75
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
Updating 66f9634..625ab75
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
 1294336/17464789 [=>............................] - ETA: 0s
 4505600/17464789 [======>.......................] - ETA: 0s
12181504/17464789 [===================>..........] - ETA: 0s
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
2020-05-18 12:28:07.073654: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-18 12:28:07.078167: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-18 12:28:07.078294: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f2a1376820 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 12:28:07.078309: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7280 - accuracy: 0.4960
 2000/25000 [=>............................] - ETA: 9s - loss: 7.3906 - accuracy: 0.5180 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.3702 - accuracy: 0.5193
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.3906 - accuracy: 0.5180
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.4581 - accuracy: 0.5136
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5388 - accuracy: 0.5083
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5900 - accuracy: 0.5050
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6417 - accuracy: 0.5016
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6479 - accuracy: 0.5012
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6283 - accuracy: 0.5025
11000/25000 [============>.................] - ETA: 4s - loss: 7.6276 - accuracy: 0.5025
12000/25000 [=============>................] - ETA: 4s - loss: 7.6487 - accuracy: 0.5012
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6713 - accuracy: 0.4997
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6765 - accuracy: 0.4994
15000/25000 [=================>............] - ETA: 3s - loss: 7.6952 - accuracy: 0.4981
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6887 - accuracy: 0.4986
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6747 - accuracy: 0.4995
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6521 - accuracy: 0.5009
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6567 - accuracy: 0.5006
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6528 - accuracy: 0.5009
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6541 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6615 - accuracy: 0.5003
25000/25000 [==============================] - 10s 397us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f413c967cc0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f413c9d18d0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.3600 - accuracy: 0.5200
 2000/25000 [=>............................] - ETA: 10s - loss: 7.2986 - accuracy: 0.5240
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.5082 - accuracy: 0.5103 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5631 - accuracy: 0.5067
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6176 - accuracy: 0.5032
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5925 - accuracy: 0.5048
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5505 - accuracy: 0.5076
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5401 - accuracy: 0.5082
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5593 - accuracy: 0.5070
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5532 - accuracy: 0.5074
11000/25000 [============>.................] - ETA: 4s - loss: 7.5802 - accuracy: 0.5056
12000/25000 [=============>................] - ETA: 4s - loss: 7.5631 - accuracy: 0.5067
13000/25000 [==============>...............] - ETA: 4s - loss: 7.5510 - accuracy: 0.5075
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5407 - accuracy: 0.5082
15000/25000 [=================>............] - ETA: 3s - loss: 7.5818 - accuracy: 0.5055
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5919 - accuracy: 0.5049
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6035 - accuracy: 0.5041
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6181 - accuracy: 0.5032
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6343 - accuracy: 0.5021
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6390 - accuracy: 0.5018
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6447 - accuracy: 0.5014
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6367 - accuracy: 0.5020
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6380 - accuracy: 0.5019
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6532 - accuracy: 0.5009
25000/25000 [==============================] - 10s 392us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8353 - accuracy: 0.4890
 2000/25000 [=>............................] - ETA: 9s - loss: 7.9733 - accuracy: 0.4800 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.9426 - accuracy: 0.4820
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8123 - accuracy: 0.4905
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7556 - accuracy: 0.4942
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7101 - accuracy: 0.4972
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6820 - accuracy: 0.4990
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7548 - accuracy: 0.4942
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7416 - accuracy: 0.4951
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7218 - accuracy: 0.4964
11000/25000 [============>.................] - ETA: 4s - loss: 7.7015 - accuracy: 0.4977
12000/25000 [=============>................] - ETA: 4s - loss: 7.7420 - accuracy: 0.4951
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7280 - accuracy: 0.4960
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7203 - accuracy: 0.4965
15000/25000 [=================>............] - ETA: 3s - loss: 7.6901 - accuracy: 0.4985
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6839 - accuracy: 0.4989
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6590 - accuracy: 0.5005
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6481 - accuracy: 0.5012
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6689 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6644 - accuracy: 0.5001
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6694 - accuracy: 0.4998
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6756 - accuracy: 0.4994
25000/25000 [==============================] - 10s 391us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 66f9634..625ab75
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

13/13 [==============================] - 2s 123ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
Updating 66f9634..625ab75
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
 1884160/11490434 [===>..........................] - ETA: 0s
 7774208/11490434 [===================>..........] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:43 - loss: 2.3008 - categorical_accuracy: 0.0312
   64/60000 [..............................] - ETA: 4:53 - loss: 2.2847 - categorical_accuracy: 0.0625
   96/60000 [..............................] - ETA: 3:52 - loss: 2.2713 - categorical_accuracy: 0.0938
  128/60000 [..............................] - ETA: 3:23 - loss: 2.2805 - categorical_accuracy: 0.1094
  160/60000 [..............................] - ETA: 3:05 - loss: 2.2659 - categorical_accuracy: 0.1312
  192/60000 [..............................] - ETA: 2:53 - loss: 2.2307 - categorical_accuracy: 0.1875
  224/60000 [..............................] - ETA: 2:45 - loss: 2.1938 - categorical_accuracy: 0.2188
  256/60000 [..............................] - ETA: 2:38 - loss: 2.1537 - categorical_accuracy: 0.2422
  288/60000 [..............................] - ETA: 2:32 - loss: 2.1268 - categorical_accuracy: 0.2500
  320/60000 [..............................] - ETA: 2:28 - loss: 2.0739 - categorical_accuracy: 0.2719
  352/60000 [..............................] - ETA: 2:24 - loss: 2.0598 - categorical_accuracy: 0.2784
  384/60000 [..............................] - ETA: 2:21 - loss: 2.0196 - categorical_accuracy: 0.2995
  416/60000 [..............................] - ETA: 2:19 - loss: 1.9687 - categorical_accuracy: 0.3269
  448/60000 [..............................] - ETA: 2:17 - loss: 1.9044 - categorical_accuracy: 0.3460
  480/60000 [..............................] - ETA: 2:16 - loss: 1.8649 - categorical_accuracy: 0.3625
  512/60000 [..............................] - ETA: 2:15 - loss: 1.8597 - categorical_accuracy: 0.3633
  544/60000 [..............................] - ETA: 2:14 - loss: 1.8387 - categorical_accuracy: 0.3732
  576/60000 [..............................] - ETA: 2:12 - loss: 1.8047 - categorical_accuracy: 0.3872
  608/60000 [..............................] - ETA: 2:11 - loss: 1.7675 - categorical_accuracy: 0.4046
  640/60000 [..............................] - ETA: 2:10 - loss: 1.8005 - categorical_accuracy: 0.4047
  672/60000 [..............................] - ETA: 2:09 - loss: 1.7844 - categorical_accuracy: 0.4122
  704/60000 [..............................] - ETA: 2:08 - loss: 1.7613 - categorical_accuracy: 0.4190
  736/60000 [..............................] - ETA: 2:07 - loss: 1.7382 - categorical_accuracy: 0.4253
  768/60000 [..............................] - ETA: 2:07 - loss: 1.7068 - categorical_accuracy: 0.4388
  800/60000 [..............................] - ETA: 2:06 - loss: 1.6803 - categorical_accuracy: 0.4475
  832/60000 [..............................] - ETA: 2:05 - loss: 1.6574 - categorical_accuracy: 0.4567
  864/60000 [..............................] - ETA: 2:04 - loss: 1.6352 - categorical_accuracy: 0.4595
  896/60000 [..............................] - ETA: 2:04 - loss: 1.6149 - categorical_accuracy: 0.4654
  928/60000 [..............................] - ETA: 2:03 - loss: 1.5949 - categorical_accuracy: 0.4731
  960/60000 [..............................] - ETA: 2:02 - loss: 1.5738 - categorical_accuracy: 0.4812
  992/60000 [..............................] - ETA: 2:02 - loss: 1.5481 - categorical_accuracy: 0.4899
 1024/60000 [..............................] - ETA: 2:01 - loss: 1.5135 - categorical_accuracy: 0.5020
 1056/60000 [..............................] - ETA: 2:01 - loss: 1.4919 - categorical_accuracy: 0.5076
 1088/60000 [..............................] - ETA: 2:00 - loss: 1.4818 - categorical_accuracy: 0.5138
 1120/60000 [..............................] - ETA: 2:00 - loss: 1.4666 - categorical_accuracy: 0.5196
 1152/60000 [..............................] - ETA: 2:00 - loss: 1.4361 - categorical_accuracy: 0.5321
 1184/60000 [..............................] - ETA: 1:59 - loss: 1.4202 - categorical_accuracy: 0.5380
 1216/60000 [..............................] - ETA: 1:59 - loss: 1.3949 - categorical_accuracy: 0.5469
 1248/60000 [..............................] - ETA: 1:59 - loss: 1.3740 - categorical_accuracy: 0.5545
 1280/60000 [..............................] - ETA: 1:58 - loss: 1.3570 - categorical_accuracy: 0.5594
 1312/60000 [..............................] - ETA: 1:58 - loss: 1.3413 - categorical_accuracy: 0.5640
 1344/60000 [..............................] - ETA: 1:58 - loss: 1.3262 - categorical_accuracy: 0.5677
 1376/60000 [..............................] - ETA: 1:58 - loss: 1.3269 - categorical_accuracy: 0.5661
 1408/60000 [..............................] - ETA: 1:58 - loss: 1.3098 - categorical_accuracy: 0.5710
 1440/60000 [..............................] - ETA: 1:57 - loss: 1.2950 - categorical_accuracy: 0.5757
 1472/60000 [..............................] - ETA: 1:57 - loss: 1.2793 - categorical_accuracy: 0.5815
 1504/60000 [..............................] - ETA: 1:57 - loss: 1.2718 - categorical_accuracy: 0.5851
 1536/60000 [..............................] - ETA: 1:57 - loss: 1.2545 - categorical_accuracy: 0.5905
 1568/60000 [..............................] - ETA: 1:57 - loss: 1.2393 - categorical_accuracy: 0.5950
 1600/60000 [..............................] - ETA: 1:56 - loss: 1.2207 - categorical_accuracy: 0.6012
 1632/60000 [..............................] - ETA: 1:56 - loss: 1.2056 - categorical_accuracy: 0.6060
 1664/60000 [..............................] - ETA: 1:56 - loss: 1.1917 - categorical_accuracy: 0.6112
 1696/60000 [..............................] - ETA: 1:55 - loss: 1.1788 - categorical_accuracy: 0.6156
 1728/60000 [..............................] - ETA: 1:55 - loss: 1.1694 - categorical_accuracy: 0.6181
 1760/60000 [..............................] - ETA: 1:55 - loss: 1.1607 - categorical_accuracy: 0.6199
 1792/60000 [..............................] - ETA: 1:55 - loss: 1.1584 - categorical_accuracy: 0.6200
 1824/60000 [..............................] - ETA: 1:55 - loss: 1.1503 - categorical_accuracy: 0.6217
 1856/60000 [..............................] - ETA: 1:55 - loss: 1.1426 - categorical_accuracy: 0.6250
 1888/60000 [..............................] - ETA: 1:55 - loss: 1.1351 - categorical_accuracy: 0.6287
 1920/60000 [..............................] - ETA: 1:55 - loss: 1.1319 - categorical_accuracy: 0.6292
 1952/60000 [..............................] - ETA: 1:55 - loss: 1.1205 - categorical_accuracy: 0.6317
 1984/60000 [..............................] - ETA: 1:55 - loss: 1.1113 - categorical_accuracy: 0.6361
 2016/60000 [>.............................] - ETA: 1:55 - loss: 1.1021 - categorical_accuracy: 0.6404
 2048/60000 [>.............................] - ETA: 1:55 - loss: 1.0904 - categorical_accuracy: 0.6450
 2080/60000 [>.............................] - ETA: 1:54 - loss: 1.0830 - categorical_accuracy: 0.6481
 2112/60000 [>.............................] - ETA: 1:54 - loss: 1.0749 - categorical_accuracy: 0.6510
 2144/60000 [>.............................] - ETA: 1:55 - loss: 1.0686 - categorical_accuracy: 0.6535
 2176/60000 [>.............................] - ETA: 1:55 - loss: 1.0601 - categorical_accuracy: 0.6558
 2208/60000 [>.............................] - ETA: 1:54 - loss: 1.0531 - categorical_accuracy: 0.6581
 2240/60000 [>.............................] - ETA: 1:54 - loss: 1.0426 - categorical_accuracy: 0.6616
 2272/60000 [>.............................] - ETA: 1:54 - loss: 1.0324 - categorical_accuracy: 0.6646
 2304/60000 [>.............................] - ETA: 1:54 - loss: 1.0260 - categorical_accuracy: 0.6671
 2336/60000 [>.............................] - ETA: 1:54 - loss: 1.0237 - categorical_accuracy: 0.6691
 2368/60000 [>.............................] - ETA: 1:54 - loss: 1.0179 - categorical_accuracy: 0.6710
 2400/60000 [>.............................] - ETA: 1:54 - loss: 1.0114 - categorical_accuracy: 0.6733
 2432/60000 [>.............................] - ETA: 1:54 - loss: 1.0073 - categorical_accuracy: 0.6748
 2464/60000 [>.............................] - ETA: 1:53 - loss: 0.9996 - categorical_accuracy: 0.6778
 2496/60000 [>.............................] - ETA: 1:53 - loss: 0.9957 - categorical_accuracy: 0.6787
 2528/60000 [>.............................] - ETA: 1:53 - loss: 0.9912 - categorical_accuracy: 0.6804
 2560/60000 [>.............................] - ETA: 1:53 - loss: 0.9845 - categorical_accuracy: 0.6820
 2592/60000 [>.............................] - ETA: 1:53 - loss: 0.9774 - categorical_accuracy: 0.6844
 2624/60000 [>.............................] - ETA: 1:53 - loss: 0.9697 - categorical_accuracy: 0.6867
 2656/60000 [>.............................] - ETA: 1:53 - loss: 0.9638 - categorical_accuracy: 0.6886
 2688/60000 [>.............................] - ETA: 1:53 - loss: 0.9572 - categorical_accuracy: 0.6908
 2720/60000 [>.............................] - ETA: 1:53 - loss: 0.9523 - categorical_accuracy: 0.6926
 2752/60000 [>.............................] - ETA: 1:53 - loss: 0.9466 - categorical_accuracy: 0.6940
 2784/60000 [>.............................] - ETA: 1:53 - loss: 0.9421 - categorical_accuracy: 0.6961
 2816/60000 [>.............................] - ETA: 1:53 - loss: 0.9364 - categorical_accuracy: 0.6978
 2848/60000 [>.............................] - ETA: 1:52 - loss: 0.9294 - categorical_accuracy: 0.7005
 2880/60000 [>.............................] - ETA: 1:52 - loss: 0.9250 - categorical_accuracy: 0.7028
 2912/60000 [>.............................] - ETA: 1:52 - loss: 0.9218 - categorical_accuracy: 0.7047
 2944/60000 [>.............................] - ETA: 1:52 - loss: 0.9194 - categorical_accuracy: 0.7055
 2976/60000 [>.............................] - ETA: 1:52 - loss: 0.9155 - categorical_accuracy: 0.7067
 3008/60000 [>.............................] - ETA: 1:52 - loss: 0.9091 - categorical_accuracy: 0.7088
 3040/60000 [>.............................] - ETA: 1:52 - loss: 0.9037 - categorical_accuracy: 0.7105
 3072/60000 [>.............................] - ETA: 1:52 - loss: 0.8969 - categorical_accuracy: 0.7126
 3104/60000 [>.............................] - ETA: 1:51 - loss: 0.8921 - categorical_accuracy: 0.7139
 3136/60000 [>.............................] - ETA: 1:51 - loss: 0.8887 - categorical_accuracy: 0.7146
 3168/60000 [>.............................] - ETA: 1:51 - loss: 0.8836 - categorical_accuracy: 0.7159
 3200/60000 [>.............................] - ETA: 1:51 - loss: 0.8771 - categorical_accuracy: 0.7181
 3232/60000 [>.............................] - ETA: 1:51 - loss: 0.8753 - categorical_accuracy: 0.7197
 3264/60000 [>.............................] - ETA: 1:51 - loss: 0.8710 - categorical_accuracy: 0.7209
 3296/60000 [>.............................] - ETA: 1:51 - loss: 0.8656 - categorical_accuracy: 0.7227
 3328/60000 [>.............................] - ETA: 1:51 - loss: 0.8618 - categorical_accuracy: 0.7239
 3360/60000 [>.............................] - ETA: 1:51 - loss: 0.8586 - categorical_accuracy: 0.7247
 3392/60000 [>.............................] - ETA: 1:51 - loss: 0.8563 - categorical_accuracy: 0.7252
 3424/60000 [>.............................] - ETA: 1:51 - loss: 0.8530 - categorical_accuracy: 0.7258
 3456/60000 [>.............................] - ETA: 1:51 - loss: 0.8482 - categorical_accuracy: 0.7274
 3488/60000 [>.............................] - ETA: 1:50 - loss: 0.8435 - categorical_accuracy: 0.7285
 3520/60000 [>.............................] - ETA: 1:50 - loss: 0.8403 - categorical_accuracy: 0.7293
 3552/60000 [>.............................] - ETA: 1:50 - loss: 0.8349 - categorical_accuracy: 0.7311
 3584/60000 [>.............................] - ETA: 1:50 - loss: 0.8307 - categorical_accuracy: 0.7327
 3616/60000 [>.............................] - ETA: 1:50 - loss: 0.8277 - categorical_accuracy: 0.7342
 3648/60000 [>.............................] - ETA: 1:50 - loss: 0.8243 - categorical_accuracy: 0.7355
 3680/60000 [>.............................] - ETA: 1:50 - loss: 0.8210 - categorical_accuracy: 0.7367
 3712/60000 [>.............................] - ETA: 1:50 - loss: 0.8164 - categorical_accuracy: 0.7381
 3744/60000 [>.............................] - ETA: 1:50 - loss: 0.8120 - categorical_accuracy: 0.7393
 3776/60000 [>.............................] - ETA: 1:50 - loss: 0.8073 - categorical_accuracy: 0.7410
 3808/60000 [>.............................] - ETA: 1:50 - loss: 0.8034 - categorical_accuracy: 0.7419
 3840/60000 [>.............................] - ETA: 1:49 - loss: 0.8004 - categorical_accuracy: 0.7430
 3872/60000 [>.............................] - ETA: 1:49 - loss: 0.7969 - categorical_accuracy: 0.7438
 3904/60000 [>.............................] - ETA: 1:49 - loss: 0.7935 - categorical_accuracy: 0.7446
 3936/60000 [>.............................] - ETA: 1:49 - loss: 0.7903 - categorical_accuracy: 0.7459
 3968/60000 [>.............................] - ETA: 1:49 - loss: 0.7845 - categorical_accuracy: 0.7480
 4000/60000 [=>............................] - ETA: 1:49 - loss: 0.7796 - categorical_accuracy: 0.7498
 4032/60000 [=>............................] - ETA: 1:49 - loss: 0.7764 - categorical_accuracy: 0.7507
 4064/60000 [=>............................] - ETA: 1:49 - loss: 0.7734 - categorical_accuracy: 0.7515
 4096/60000 [=>............................] - ETA: 1:49 - loss: 0.7694 - categorical_accuracy: 0.7524
 4128/60000 [=>............................] - ETA: 1:49 - loss: 0.7668 - categorical_accuracy: 0.7529
 4160/60000 [=>............................] - ETA: 1:49 - loss: 0.7633 - categorical_accuracy: 0.7538
 4192/60000 [=>............................] - ETA: 1:48 - loss: 0.7588 - categorical_accuracy: 0.7552
 4224/60000 [=>............................] - ETA: 1:48 - loss: 0.7556 - categorical_accuracy: 0.7562
 4256/60000 [=>............................] - ETA: 1:48 - loss: 0.7549 - categorical_accuracy: 0.7559
 4288/60000 [=>............................] - ETA: 1:48 - loss: 0.7527 - categorical_accuracy: 0.7568
 4320/60000 [=>............................] - ETA: 1:48 - loss: 0.7483 - categorical_accuracy: 0.7583
 4352/60000 [=>............................] - ETA: 1:48 - loss: 0.7445 - categorical_accuracy: 0.7597
 4384/60000 [=>............................] - ETA: 1:48 - loss: 0.7429 - categorical_accuracy: 0.7603
 4416/60000 [=>............................] - ETA: 1:48 - loss: 0.7396 - categorical_accuracy: 0.7613
 4448/60000 [=>............................] - ETA: 1:48 - loss: 0.7355 - categorical_accuracy: 0.7626
 4480/60000 [=>............................] - ETA: 1:48 - loss: 0.7318 - categorical_accuracy: 0.7636
 4512/60000 [=>............................] - ETA: 1:48 - loss: 0.7301 - categorical_accuracy: 0.7640
 4544/60000 [=>............................] - ETA: 1:48 - loss: 0.7261 - categorical_accuracy: 0.7652
 4576/60000 [=>............................] - ETA: 1:48 - loss: 0.7248 - categorical_accuracy: 0.7655
 4608/60000 [=>............................] - ETA: 1:48 - loss: 0.7231 - categorical_accuracy: 0.7663
 4640/60000 [=>............................] - ETA: 1:47 - loss: 0.7210 - categorical_accuracy: 0.7670
 4672/60000 [=>............................] - ETA: 1:47 - loss: 0.7189 - categorical_accuracy: 0.7678
 4704/60000 [=>............................] - ETA: 1:47 - loss: 0.7168 - categorical_accuracy: 0.7689
 4736/60000 [=>............................] - ETA: 1:47 - loss: 0.7144 - categorical_accuracy: 0.7696
 4768/60000 [=>............................] - ETA: 1:47 - loss: 0.7107 - categorical_accuracy: 0.7712
 4800/60000 [=>............................] - ETA: 1:47 - loss: 0.7073 - categorical_accuracy: 0.7723
 4832/60000 [=>............................] - ETA: 1:47 - loss: 0.7048 - categorical_accuracy: 0.7734
 4864/60000 [=>............................] - ETA: 1:47 - loss: 0.7011 - categorical_accuracy: 0.7747
 4896/60000 [=>............................] - ETA: 1:47 - loss: 0.7001 - categorical_accuracy: 0.7755
 4928/60000 [=>............................] - ETA: 1:47 - loss: 0.6974 - categorical_accuracy: 0.7764
 4960/60000 [=>............................] - ETA: 1:47 - loss: 0.6965 - categorical_accuracy: 0.7770
 4992/60000 [=>............................] - ETA: 1:46 - loss: 0.6927 - categorical_accuracy: 0.7782
 5024/60000 [=>............................] - ETA: 1:46 - loss: 0.6891 - categorical_accuracy: 0.7795
 5056/60000 [=>............................] - ETA: 1:46 - loss: 0.6862 - categorical_accuracy: 0.7803
 5088/60000 [=>............................] - ETA: 1:46 - loss: 0.6836 - categorical_accuracy: 0.7811
 5120/60000 [=>............................] - ETA: 1:46 - loss: 0.6817 - categorical_accuracy: 0.7818
 5152/60000 [=>............................] - ETA: 1:46 - loss: 0.6785 - categorical_accuracy: 0.7826
 5184/60000 [=>............................] - ETA: 1:46 - loss: 0.6747 - categorical_accuracy: 0.7838
 5216/60000 [=>............................] - ETA: 1:46 - loss: 0.6721 - categorical_accuracy: 0.7845
 5248/60000 [=>............................] - ETA: 1:46 - loss: 0.6703 - categorical_accuracy: 0.7849
 5280/60000 [=>............................] - ETA: 1:46 - loss: 0.6676 - categorical_accuracy: 0.7858
 5312/60000 [=>............................] - ETA: 1:46 - loss: 0.6646 - categorical_accuracy: 0.7869
 5344/60000 [=>............................] - ETA: 1:46 - loss: 0.6625 - categorical_accuracy: 0.7872
 5376/60000 [=>............................] - ETA: 1:46 - loss: 0.6596 - categorical_accuracy: 0.7879
 5408/60000 [=>............................] - ETA: 1:46 - loss: 0.6564 - categorical_accuracy: 0.7888
 5440/60000 [=>............................] - ETA: 1:45 - loss: 0.6535 - categorical_accuracy: 0.7899
 5472/60000 [=>............................] - ETA: 1:45 - loss: 0.6517 - categorical_accuracy: 0.7904
 5504/60000 [=>............................] - ETA: 1:45 - loss: 0.6503 - categorical_accuracy: 0.7909
 5536/60000 [=>............................] - ETA: 1:45 - loss: 0.6482 - categorical_accuracy: 0.7915
 5568/60000 [=>............................] - ETA: 1:45 - loss: 0.6459 - categorical_accuracy: 0.7922
 5600/60000 [=>............................] - ETA: 1:45 - loss: 0.6445 - categorical_accuracy: 0.7929
 5632/60000 [=>............................] - ETA: 1:45 - loss: 0.6423 - categorical_accuracy: 0.7935
 5664/60000 [=>............................] - ETA: 1:45 - loss: 0.6397 - categorical_accuracy: 0.7943
 5696/60000 [=>............................] - ETA: 1:45 - loss: 0.6374 - categorical_accuracy: 0.7951
 5728/60000 [=>............................] - ETA: 1:45 - loss: 0.6355 - categorical_accuracy: 0.7954
 5760/60000 [=>............................] - ETA: 1:45 - loss: 0.6336 - categorical_accuracy: 0.7962
 5792/60000 [=>............................] - ETA: 1:45 - loss: 0.6310 - categorical_accuracy: 0.7973
 5824/60000 [=>............................] - ETA: 1:45 - loss: 0.6293 - categorical_accuracy: 0.7979
 5856/60000 [=>............................] - ETA: 1:45 - loss: 0.6283 - categorical_accuracy: 0.7982
 5888/60000 [=>............................] - ETA: 1:44 - loss: 0.6272 - categorical_accuracy: 0.7986
 5920/60000 [=>............................] - ETA: 1:44 - loss: 0.6260 - categorical_accuracy: 0.7992
 5952/60000 [=>............................] - ETA: 1:44 - loss: 0.6252 - categorical_accuracy: 0.7996
 5984/60000 [=>............................] - ETA: 1:44 - loss: 0.6233 - categorical_accuracy: 0.8001
 6016/60000 [==>...........................] - ETA: 1:44 - loss: 0.6213 - categorical_accuracy: 0.8007
 6048/60000 [==>...........................] - ETA: 1:44 - loss: 0.6190 - categorical_accuracy: 0.8016
 6080/60000 [==>...........................] - ETA: 1:44 - loss: 0.6181 - categorical_accuracy: 0.8016
 6112/60000 [==>...........................] - ETA: 1:44 - loss: 0.6171 - categorical_accuracy: 0.8020
 6144/60000 [==>...........................] - ETA: 1:44 - loss: 0.6147 - categorical_accuracy: 0.8027
 6176/60000 [==>...........................] - ETA: 1:44 - loss: 0.6123 - categorical_accuracy: 0.8034
 6208/60000 [==>...........................] - ETA: 1:44 - loss: 0.6107 - categorical_accuracy: 0.8040
 6240/60000 [==>...........................] - ETA: 1:44 - loss: 0.6085 - categorical_accuracy: 0.8048
 6272/60000 [==>...........................] - ETA: 1:44 - loss: 0.6060 - categorical_accuracy: 0.8056
 6304/60000 [==>...........................] - ETA: 1:43 - loss: 0.6040 - categorical_accuracy: 0.8065
 6336/60000 [==>...........................] - ETA: 1:43 - loss: 0.6018 - categorical_accuracy: 0.8073
 6368/60000 [==>...........................] - ETA: 1:43 - loss: 0.5992 - categorical_accuracy: 0.8083
 6400/60000 [==>...........................] - ETA: 1:43 - loss: 0.5966 - categorical_accuracy: 0.8092
 6432/60000 [==>...........................] - ETA: 1:43 - loss: 0.5945 - categorical_accuracy: 0.8097
 6464/60000 [==>...........................] - ETA: 1:43 - loss: 0.5923 - categorical_accuracy: 0.8105
 6496/60000 [==>...........................] - ETA: 1:43 - loss: 0.5911 - categorical_accuracy: 0.8110
 6528/60000 [==>...........................] - ETA: 1:43 - loss: 0.5883 - categorical_accuracy: 0.8119
 6560/60000 [==>...........................] - ETA: 1:43 - loss: 0.5874 - categorical_accuracy: 0.8123
 6592/60000 [==>...........................] - ETA: 1:43 - loss: 0.5862 - categorical_accuracy: 0.8127
 6624/60000 [==>...........................] - ETA: 1:43 - loss: 0.5855 - categorical_accuracy: 0.8131
 6656/60000 [==>...........................] - ETA: 1:43 - loss: 0.5834 - categorical_accuracy: 0.8136
 6688/60000 [==>...........................] - ETA: 1:43 - loss: 0.5823 - categorical_accuracy: 0.8138
 6720/60000 [==>...........................] - ETA: 1:43 - loss: 0.5804 - categorical_accuracy: 0.8144
 6752/60000 [==>...........................] - ETA: 1:42 - loss: 0.5784 - categorical_accuracy: 0.8150
 6784/60000 [==>...........................] - ETA: 1:42 - loss: 0.5768 - categorical_accuracy: 0.8157
 6816/60000 [==>...........................] - ETA: 1:42 - loss: 0.5760 - categorical_accuracy: 0.8162
 6848/60000 [==>...........................] - ETA: 1:42 - loss: 0.5739 - categorical_accuracy: 0.8169
 6880/60000 [==>...........................] - ETA: 1:42 - loss: 0.5724 - categorical_accuracy: 0.8173
 6912/60000 [==>...........................] - ETA: 1:42 - loss: 0.5712 - categorical_accuracy: 0.8176
 6944/60000 [==>...........................] - ETA: 1:42 - loss: 0.5701 - categorical_accuracy: 0.8180
 6976/60000 [==>...........................] - ETA: 1:42 - loss: 0.5680 - categorical_accuracy: 0.8187
 7008/60000 [==>...........................] - ETA: 1:42 - loss: 0.5669 - categorical_accuracy: 0.8191
 7040/60000 [==>...........................] - ETA: 1:42 - loss: 0.5651 - categorical_accuracy: 0.8195
 7072/60000 [==>...........................] - ETA: 1:42 - loss: 0.5631 - categorical_accuracy: 0.8201
 7104/60000 [==>...........................] - ETA: 1:42 - loss: 0.5611 - categorical_accuracy: 0.8207
 7136/60000 [==>...........................] - ETA: 1:42 - loss: 0.5590 - categorical_accuracy: 0.8213
 7168/60000 [==>...........................] - ETA: 1:42 - loss: 0.5570 - categorical_accuracy: 0.8218
 7200/60000 [==>...........................] - ETA: 1:42 - loss: 0.5554 - categorical_accuracy: 0.8225
 7232/60000 [==>...........................] - ETA: 1:41 - loss: 0.5540 - categorical_accuracy: 0.8231
 7264/60000 [==>...........................] - ETA: 1:41 - loss: 0.5525 - categorical_accuracy: 0.8237
 7296/60000 [==>...........................] - ETA: 1:41 - loss: 0.5512 - categorical_accuracy: 0.8242
 7328/60000 [==>...........................] - ETA: 1:41 - loss: 0.5491 - categorical_accuracy: 0.8248
 7360/60000 [==>...........................] - ETA: 1:41 - loss: 0.5468 - categorical_accuracy: 0.8255
 7392/60000 [==>...........................] - ETA: 1:41 - loss: 0.5488 - categorical_accuracy: 0.8252
 7424/60000 [==>...........................] - ETA: 1:41 - loss: 0.5474 - categorical_accuracy: 0.8258
 7456/60000 [==>...........................] - ETA: 1:41 - loss: 0.5452 - categorical_accuracy: 0.8266
 7488/60000 [==>...........................] - ETA: 1:41 - loss: 0.5430 - categorical_accuracy: 0.8273
 7520/60000 [==>...........................] - ETA: 1:41 - loss: 0.5421 - categorical_accuracy: 0.8275
 7552/60000 [==>...........................] - ETA: 1:41 - loss: 0.5414 - categorical_accuracy: 0.8279
 7584/60000 [==>...........................] - ETA: 1:41 - loss: 0.5404 - categorical_accuracy: 0.8283
 7616/60000 [==>...........................] - ETA: 1:41 - loss: 0.5393 - categorical_accuracy: 0.8287
 7648/60000 [==>...........................] - ETA: 1:41 - loss: 0.5371 - categorical_accuracy: 0.8294
 7680/60000 [==>...........................] - ETA: 1:41 - loss: 0.5356 - categorical_accuracy: 0.8299
 7712/60000 [==>...........................] - ETA: 1:40 - loss: 0.5353 - categorical_accuracy: 0.8301
 7744/60000 [==>...........................] - ETA: 1:40 - loss: 0.5349 - categorical_accuracy: 0.8303
 7776/60000 [==>...........................] - ETA: 1:40 - loss: 0.5332 - categorical_accuracy: 0.8309
 7808/60000 [==>...........................] - ETA: 1:40 - loss: 0.5327 - categorical_accuracy: 0.8312
 7840/60000 [==>...........................] - ETA: 1:40 - loss: 0.5310 - categorical_accuracy: 0.8316
 7872/60000 [==>...........................] - ETA: 1:40 - loss: 0.5299 - categorical_accuracy: 0.8318
 7904/60000 [==>...........................] - ETA: 1:40 - loss: 0.5285 - categorical_accuracy: 0.8321
 7936/60000 [==>...........................] - ETA: 1:40 - loss: 0.5268 - categorical_accuracy: 0.8327
 7968/60000 [==>...........................] - ETA: 1:40 - loss: 0.5261 - categorical_accuracy: 0.8331
 8000/60000 [===>..........................] - ETA: 1:40 - loss: 0.5243 - categorical_accuracy: 0.8338
 8032/60000 [===>..........................] - ETA: 1:40 - loss: 0.5238 - categorical_accuracy: 0.8339
 8064/60000 [===>..........................] - ETA: 1:40 - loss: 0.5221 - categorical_accuracy: 0.8344
 8096/60000 [===>..........................] - ETA: 1:40 - loss: 0.5212 - categorical_accuracy: 0.8347
 8128/60000 [===>..........................] - ETA: 1:40 - loss: 0.5196 - categorical_accuracy: 0.8353
 8160/60000 [===>..........................] - ETA: 1:40 - loss: 0.5195 - categorical_accuracy: 0.8357
 8192/60000 [===>..........................] - ETA: 1:39 - loss: 0.5196 - categorical_accuracy: 0.8359
 8224/60000 [===>..........................] - ETA: 1:39 - loss: 0.5179 - categorical_accuracy: 0.8365
 8256/60000 [===>..........................] - ETA: 1:39 - loss: 0.5177 - categorical_accuracy: 0.8366
 8288/60000 [===>..........................] - ETA: 1:39 - loss: 0.5166 - categorical_accuracy: 0.8370
 8320/60000 [===>..........................] - ETA: 1:39 - loss: 0.5148 - categorical_accuracy: 0.8376
 8352/60000 [===>..........................] - ETA: 1:39 - loss: 0.5130 - categorical_accuracy: 0.8381
 8384/60000 [===>..........................] - ETA: 1:39 - loss: 0.5131 - categorical_accuracy: 0.8383
 8416/60000 [===>..........................] - ETA: 1:39 - loss: 0.5123 - categorical_accuracy: 0.8384
 8448/60000 [===>..........................] - ETA: 1:39 - loss: 0.5106 - categorical_accuracy: 0.8390
 8480/60000 [===>..........................] - ETA: 1:39 - loss: 0.5088 - categorical_accuracy: 0.8395
 8512/60000 [===>..........................] - ETA: 1:39 - loss: 0.5080 - categorical_accuracy: 0.8398
 8544/60000 [===>..........................] - ETA: 1:39 - loss: 0.5071 - categorical_accuracy: 0.8401
 8576/60000 [===>..........................] - ETA: 1:39 - loss: 0.5059 - categorical_accuracy: 0.8405
 8608/60000 [===>..........................] - ETA: 1:39 - loss: 0.5046 - categorical_accuracy: 0.8410
 8640/60000 [===>..........................] - ETA: 1:39 - loss: 0.5032 - categorical_accuracy: 0.8414
 8672/60000 [===>..........................] - ETA: 1:38 - loss: 0.5020 - categorical_accuracy: 0.8418
 8704/60000 [===>..........................] - ETA: 1:38 - loss: 0.5009 - categorical_accuracy: 0.8420
 8736/60000 [===>..........................] - ETA: 1:38 - loss: 0.5005 - categorical_accuracy: 0.8423
 8768/60000 [===>..........................] - ETA: 1:38 - loss: 0.4991 - categorical_accuracy: 0.8427
 8800/60000 [===>..........................] - ETA: 1:38 - loss: 0.4993 - categorical_accuracy: 0.8427
 8832/60000 [===>..........................] - ETA: 1:38 - loss: 0.4982 - categorical_accuracy: 0.8431
 8864/60000 [===>..........................] - ETA: 1:38 - loss: 0.4972 - categorical_accuracy: 0.8433
 8896/60000 [===>..........................] - ETA: 1:38 - loss: 0.4960 - categorical_accuracy: 0.8438
 8928/60000 [===>..........................] - ETA: 1:38 - loss: 0.4950 - categorical_accuracy: 0.8440
 8960/60000 [===>..........................] - ETA: 1:38 - loss: 0.4938 - categorical_accuracy: 0.8443
 8992/60000 [===>..........................] - ETA: 1:38 - loss: 0.4934 - categorical_accuracy: 0.8444
 9024/60000 [===>..........................] - ETA: 1:38 - loss: 0.4922 - categorical_accuracy: 0.8447
 9056/60000 [===>..........................] - ETA: 1:38 - loss: 0.4917 - categorical_accuracy: 0.8451
 9088/60000 [===>..........................] - ETA: 1:38 - loss: 0.4920 - categorical_accuracy: 0.8452
 9120/60000 [===>..........................] - ETA: 1:37 - loss: 0.4910 - categorical_accuracy: 0.8455
 9152/60000 [===>..........................] - ETA: 1:37 - loss: 0.4907 - categorical_accuracy: 0.8457
 9184/60000 [===>..........................] - ETA: 1:37 - loss: 0.4899 - categorical_accuracy: 0.8460
 9216/60000 [===>..........................] - ETA: 1:37 - loss: 0.4885 - categorical_accuracy: 0.8466
 9248/60000 [===>..........................] - ETA: 1:37 - loss: 0.4872 - categorical_accuracy: 0.8470
 9280/60000 [===>..........................] - ETA: 1:37 - loss: 0.4863 - categorical_accuracy: 0.8473
 9312/60000 [===>..........................] - ETA: 1:37 - loss: 0.4850 - categorical_accuracy: 0.8478
 9344/60000 [===>..........................] - ETA: 1:37 - loss: 0.4844 - categorical_accuracy: 0.8481
 9376/60000 [===>..........................] - ETA: 1:37 - loss: 0.4846 - categorical_accuracy: 0.8482
 9408/60000 [===>..........................] - ETA: 1:37 - loss: 0.4839 - categorical_accuracy: 0.8485
 9440/60000 [===>..........................] - ETA: 1:37 - loss: 0.4826 - categorical_accuracy: 0.8489
 9472/60000 [===>..........................] - ETA: 1:37 - loss: 0.4816 - categorical_accuracy: 0.8493
 9504/60000 [===>..........................] - ETA: 1:37 - loss: 0.4806 - categorical_accuracy: 0.8497
 9536/60000 [===>..........................] - ETA: 1:37 - loss: 0.4798 - categorical_accuracy: 0.8498
 9568/60000 [===>..........................] - ETA: 1:37 - loss: 0.4785 - categorical_accuracy: 0.8501
 9600/60000 [===>..........................] - ETA: 1:36 - loss: 0.4776 - categorical_accuracy: 0.8505
 9632/60000 [===>..........................] - ETA: 1:36 - loss: 0.4766 - categorical_accuracy: 0.8509
 9664/60000 [===>..........................] - ETA: 1:36 - loss: 0.4751 - categorical_accuracy: 0.8514
 9696/60000 [===>..........................] - ETA: 1:36 - loss: 0.4743 - categorical_accuracy: 0.8516
 9728/60000 [===>..........................] - ETA: 1:36 - loss: 0.4737 - categorical_accuracy: 0.8519
 9760/60000 [===>..........................] - ETA: 1:36 - loss: 0.4727 - categorical_accuracy: 0.8522
 9792/60000 [===>..........................] - ETA: 1:36 - loss: 0.4714 - categorical_accuracy: 0.8525
 9824/60000 [===>..........................] - ETA: 1:36 - loss: 0.4706 - categorical_accuracy: 0.8528
 9856/60000 [===>..........................] - ETA: 1:36 - loss: 0.4699 - categorical_accuracy: 0.8529
 9888/60000 [===>..........................] - ETA: 1:36 - loss: 0.4695 - categorical_accuracy: 0.8529
 9920/60000 [===>..........................] - ETA: 1:36 - loss: 0.4685 - categorical_accuracy: 0.8531
 9952/60000 [===>..........................] - ETA: 1:36 - loss: 0.4678 - categorical_accuracy: 0.8534
 9984/60000 [===>..........................] - ETA: 1:36 - loss: 0.4667 - categorical_accuracy: 0.8536
10016/60000 [====>.........................] - ETA: 1:36 - loss: 0.4660 - categorical_accuracy: 0.8539
10048/60000 [====>.........................] - ETA: 1:35 - loss: 0.4661 - categorical_accuracy: 0.8540
10080/60000 [====>.........................] - ETA: 1:35 - loss: 0.4648 - categorical_accuracy: 0.8544
10112/60000 [====>.........................] - ETA: 1:35 - loss: 0.4637 - categorical_accuracy: 0.8546
10144/60000 [====>.........................] - ETA: 1:35 - loss: 0.4635 - categorical_accuracy: 0.8549
10176/60000 [====>.........................] - ETA: 1:35 - loss: 0.4631 - categorical_accuracy: 0.8551
10208/60000 [====>.........................] - ETA: 1:35 - loss: 0.4620 - categorical_accuracy: 0.8553
10240/60000 [====>.........................] - ETA: 1:35 - loss: 0.4609 - categorical_accuracy: 0.8557
10272/60000 [====>.........................] - ETA: 1:35 - loss: 0.4603 - categorical_accuracy: 0.8558
10304/60000 [====>.........................] - ETA: 1:35 - loss: 0.4593 - categorical_accuracy: 0.8562
10336/60000 [====>.........................] - ETA: 1:35 - loss: 0.4583 - categorical_accuracy: 0.8565
10368/60000 [====>.........................] - ETA: 1:35 - loss: 0.4576 - categorical_accuracy: 0.8569
10400/60000 [====>.........................] - ETA: 1:35 - loss: 0.4566 - categorical_accuracy: 0.8571
10432/60000 [====>.........................] - ETA: 1:35 - loss: 0.4558 - categorical_accuracy: 0.8574
10464/60000 [====>.........................] - ETA: 1:35 - loss: 0.4546 - categorical_accuracy: 0.8577
10496/60000 [====>.........................] - ETA: 1:35 - loss: 0.4548 - categorical_accuracy: 0.8578
10528/60000 [====>.........................] - ETA: 1:35 - loss: 0.4537 - categorical_accuracy: 0.8582
10560/60000 [====>.........................] - ETA: 1:34 - loss: 0.4529 - categorical_accuracy: 0.8583
10592/60000 [====>.........................] - ETA: 1:34 - loss: 0.4516 - categorical_accuracy: 0.8588
10624/60000 [====>.........................] - ETA: 1:34 - loss: 0.4520 - categorical_accuracy: 0.8588
10656/60000 [====>.........................] - ETA: 1:34 - loss: 0.4510 - categorical_accuracy: 0.8591
10688/60000 [====>.........................] - ETA: 1:34 - loss: 0.4500 - categorical_accuracy: 0.8594
10720/60000 [====>.........................] - ETA: 1:34 - loss: 0.4490 - categorical_accuracy: 0.8597
10752/60000 [====>.........................] - ETA: 1:34 - loss: 0.4479 - categorical_accuracy: 0.8601
10784/60000 [====>.........................] - ETA: 1:34 - loss: 0.4471 - categorical_accuracy: 0.8603
10816/60000 [====>.........................] - ETA: 1:34 - loss: 0.4468 - categorical_accuracy: 0.8604
10848/60000 [====>.........................] - ETA: 1:34 - loss: 0.4460 - categorical_accuracy: 0.8607
10880/60000 [====>.........................] - ETA: 1:34 - loss: 0.4451 - categorical_accuracy: 0.8609
10912/60000 [====>.........................] - ETA: 1:34 - loss: 0.4445 - categorical_accuracy: 0.8611
10944/60000 [====>.........................] - ETA: 1:34 - loss: 0.4437 - categorical_accuracy: 0.8615
10976/60000 [====>.........................] - ETA: 1:34 - loss: 0.4430 - categorical_accuracy: 0.8617
11008/60000 [====>.........................] - ETA: 1:33 - loss: 0.4435 - categorical_accuracy: 0.8616
11040/60000 [====>.........................] - ETA: 1:33 - loss: 0.4428 - categorical_accuracy: 0.8620
11072/60000 [====>.........................] - ETA: 1:33 - loss: 0.4419 - categorical_accuracy: 0.8622
11104/60000 [====>.........................] - ETA: 1:33 - loss: 0.4411 - categorical_accuracy: 0.8624
11136/60000 [====>.........................] - ETA: 1:33 - loss: 0.4405 - categorical_accuracy: 0.8625
11168/60000 [====>.........................] - ETA: 1:33 - loss: 0.4395 - categorical_accuracy: 0.8628
11200/60000 [====>.........................] - ETA: 1:33 - loss: 0.4385 - categorical_accuracy: 0.8632
11232/60000 [====>.........................] - ETA: 1:33 - loss: 0.4377 - categorical_accuracy: 0.8635
11264/60000 [====>.........................] - ETA: 1:33 - loss: 0.4366 - categorical_accuracy: 0.8639
11296/60000 [====>.........................] - ETA: 1:33 - loss: 0.4359 - categorical_accuracy: 0.8639
11328/60000 [====>.........................] - ETA: 1:33 - loss: 0.4353 - categorical_accuracy: 0.8641
11360/60000 [====>.........................] - ETA: 1:33 - loss: 0.4343 - categorical_accuracy: 0.8643
11392/60000 [====>.........................] - ETA: 1:33 - loss: 0.4333 - categorical_accuracy: 0.8647
11424/60000 [====>.........................] - ETA: 1:33 - loss: 0.4327 - categorical_accuracy: 0.8648
11456/60000 [====>.........................] - ETA: 1:32 - loss: 0.4322 - categorical_accuracy: 0.8650
11488/60000 [====>.........................] - ETA: 1:32 - loss: 0.4313 - categorical_accuracy: 0.8653
11520/60000 [====>.........................] - ETA: 1:32 - loss: 0.4306 - categorical_accuracy: 0.8655
11552/60000 [====>.........................] - ETA: 1:32 - loss: 0.4296 - categorical_accuracy: 0.8659
11584/60000 [====>.........................] - ETA: 1:32 - loss: 0.4288 - categorical_accuracy: 0.8661
11616/60000 [====>.........................] - ETA: 1:32 - loss: 0.4282 - categorical_accuracy: 0.8662
11648/60000 [====>.........................] - ETA: 1:32 - loss: 0.4275 - categorical_accuracy: 0.8664
11680/60000 [====>.........................] - ETA: 1:32 - loss: 0.4270 - categorical_accuracy: 0.8665
11712/60000 [====>.........................] - ETA: 1:32 - loss: 0.4262 - categorical_accuracy: 0.8668
11744/60000 [====>.........................] - ETA: 1:32 - loss: 0.4253 - categorical_accuracy: 0.8671
11776/60000 [====>.........................] - ETA: 1:32 - loss: 0.4244 - categorical_accuracy: 0.8674
11808/60000 [====>.........................] - ETA: 1:32 - loss: 0.4233 - categorical_accuracy: 0.8677
11840/60000 [====>.........................] - ETA: 1:32 - loss: 0.4222 - categorical_accuracy: 0.8681
11872/60000 [====>.........................] - ETA: 1:32 - loss: 0.4212 - categorical_accuracy: 0.8683
11904/60000 [====>.........................] - ETA: 1:32 - loss: 0.4207 - categorical_accuracy: 0.8685
11936/60000 [====>.........................] - ETA: 1:32 - loss: 0.4202 - categorical_accuracy: 0.8685
11968/60000 [====>.........................] - ETA: 1:31 - loss: 0.4192 - categorical_accuracy: 0.8689
12000/60000 [=====>........................] - ETA: 1:31 - loss: 0.4182 - categorical_accuracy: 0.8692
12032/60000 [=====>........................] - ETA: 1:31 - loss: 0.4181 - categorical_accuracy: 0.8691
12064/60000 [=====>........................] - ETA: 1:31 - loss: 0.4175 - categorical_accuracy: 0.8692
12096/60000 [=====>........................] - ETA: 1:31 - loss: 0.4171 - categorical_accuracy: 0.8693
12128/60000 [=====>........................] - ETA: 1:31 - loss: 0.4161 - categorical_accuracy: 0.8696
12160/60000 [=====>........................] - ETA: 1:31 - loss: 0.4152 - categorical_accuracy: 0.8699
12192/60000 [=====>........................] - ETA: 1:31 - loss: 0.4143 - categorical_accuracy: 0.8702
12224/60000 [=====>........................] - ETA: 1:31 - loss: 0.4134 - categorical_accuracy: 0.8704
12256/60000 [=====>........................] - ETA: 1:31 - loss: 0.4130 - categorical_accuracy: 0.8706
12288/60000 [=====>........................] - ETA: 1:31 - loss: 0.4123 - categorical_accuracy: 0.8707
12320/60000 [=====>........................] - ETA: 1:31 - loss: 0.4117 - categorical_accuracy: 0.8709
12352/60000 [=====>........................] - ETA: 1:31 - loss: 0.4110 - categorical_accuracy: 0.8711
12384/60000 [=====>........................] - ETA: 1:31 - loss: 0.4103 - categorical_accuracy: 0.8713
12416/60000 [=====>........................] - ETA: 1:31 - loss: 0.4097 - categorical_accuracy: 0.8715
12448/60000 [=====>........................] - ETA: 1:30 - loss: 0.4094 - categorical_accuracy: 0.8716
12480/60000 [=====>........................] - ETA: 1:30 - loss: 0.4088 - categorical_accuracy: 0.8719
12512/60000 [=====>........................] - ETA: 1:30 - loss: 0.4082 - categorical_accuracy: 0.8720
12544/60000 [=====>........................] - ETA: 1:30 - loss: 0.4074 - categorical_accuracy: 0.8724
12576/60000 [=====>........................] - ETA: 1:30 - loss: 0.4071 - categorical_accuracy: 0.8725
12608/60000 [=====>........................] - ETA: 1:30 - loss: 0.4064 - categorical_accuracy: 0.8728
12640/60000 [=====>........................] - ETA: 1:30 - loss: 0.4057 - categorical_accuracy: 0.8730
12672/60000 [=====>........................] - ETA: 1:30 - loss: 0.4050 - categorical_accuracy: 0.8733
12704/60000 [=====>........................] - ETA: 1:30 - loss: 0.4043 - categorical_accuracy: 0.8734
12736/60000 [=====>........................] - ETA: 1:30 - loss: 0.4045 - categorical_accuracy: 0.8734
12768/60000 [=====>........................] - ETA: 1:30 - loss: 0.4040 - categorical_accuracy: 0.8736
12800/60000 [=====>........................] - ETA: 1:30 - loss: 0.4031 - categorical_accuracy: 0.8739
12832/60000 [=====>........................] - ETA: 1:30 - loss: 0.4024 - categorical_accuracy: 0.8741
12864/60000 [=====>........................] - ETA: 1:30 - loss: 0.4018 - categorical_accuracy: 0.8742
12896/60000 [=====>........................] - ETA: 1:30 - loss: 0.4013 - categorical_accuracy: 0.8743
12928/60000 [=====>........................] - ETA: 1:29 - loss: 0.4014 - categorical_accuracy: 0.8744
12960/60000 [=====>........................] - ETA: 1:29 - loss: 0.4010 - categorical_accuracy: 0.8745
12992/60000 [=====>........................] - ETA: 1:29 - loss: 0.4006 - categorical_accuracy: 0.8746
13024/60000 [=====>........................] - ETA: 1:29 - loss: 0.4000 - categorical_accuracy: 0.8747
13056/60000 [=====>........................] - ETA: 1:29 - loss: 0.3992 - categorical_accuracy: 0.8750
13088/60000 [=====>........................] - ETA: 1:29 - loss: 0.3990 - categorical_accuracy: 0.8750
13120/60000 [=====>........................] - ETA: 1:29 - loss: 0.3984 - categorical_accuracy: 0.8752
13152/60000 [=====>........................] - ETA: 1:29 - loss: 0.3977 - categorical_accuracy: 0.8755
13184/60000 [=====>........................] - ETA: 1:29 - loss: 0.3969 - categorical_accuracy: 0.8758
13216/60000 [=====>........................] - ETA: 1:29 - loss: 0.3960 - categorical_accuracy: 0.8761
13248/60000 [=====>........................] - ETA: 1:29 - loss: 0.3955 - categorical_accuracy: 0.8761
13280/60000 [=====>........................] - ETA: 1:29 - loss: 0.3955 - categorical_accuracy: 0.8762
13312/60000 [=====>........................] - ETA: 1:29 - loss: 0.3947 - categorical_accuracy: 0.8765
13344/60000 [=====>........................] - ETA: 1:29 - loss: 0.3940 - categorical_accuracy: 0.8767
13376/60000 [=====>........................] - ETA: 1:29 - loss: 0.3934 - categorical_accuracy: 0.8768
13408/60000 [=====>........................] - ETA: 1:29 - loss: 0.3935 - categorical_accuracy: 0.8769
13440/60000 [=====>........................] - ETA: 1:29 - loss: 0.3927 - categorical_accuracy: 0.8772
13472/60000 [=====>........................] - ETA: 1:28 - loss: 0.3925 - categorical_accuracy: 0.8774
13504/60000 [=====>........................] - ETA: 1:28 - loss: 0.3916 - categorical_accuracy: 0.8777
13536/60000 [=====>........................] - ETA: 1:28 - loss: 0.3910 - categorical_accuracy: 0.8778
13568/60000 [=====>........................] - ETA: 1:28 - loss: 0.3911 - categorical_accuracy: 0.8777
13600/60000 [=====>........................] - ETA: 1:28 - loss: 0.3907 - categorical_accuracy: 0.8779
13632/60000 [=====>........................] - ETA: 1:28 - loss: 0.3900 - categorical_accuracy: 0.8781
13664/60000 [=====>........................] - ETA: 1:28 - loss: 0.3897 - categorical_accuracy: 0.8781
13696/60000 [=====>........................] - ETA: 1:28 - loss: 0.3893 - categorical_accuracy: 0.8783
13728/60000 [=====>........................] - ETA: 1:28 - loss: 0.3887 - categorical_accuracy: 0.8785
13760/60000 [=====>........................] - ETA: 1:28 - loss: 0.3882 - categorical_accuracy: 0.8786
13792/60000 [=====>........................] - ETA: 1:28 - loss: 0.3876 - categorical_accuracy: 0.8788
13824/60000 [=====>........................] - ETA: 1:28 - loss: 0.3872 - categorical_accuracy: 0.8790
13856/60000 [=====>........................] - ETA: 1:28 - loss: 0.3864 - categorical_accuracy: 0.8793
13888/60000 [=====>........................] - ETA: 1:28 - loss: 0.3857 - categorical_accuracy: 0.8795
13920/60000 [=====>........................] - ETA: 1:28 - loss: 0.3849 - categorical_accuracy: 0.8797
13952/60000 [=====>........................] - ETA: 1:27 - loss: 0.3850 - categorical_accuracy: 0.8797
13984/60000 [=====>........................] - ETA: 1:27 - loss: 0.3842 - categorical_accuracy: 0.8799
14016/60000 [======>.......................] - ETA: 1:27 - loss: 0.3838 - categorical_accuracy: 0.8801
14048/60000 [======>.......................] - ETA: 1:27 - loss: 0.3834 - categorical_accuracy: 0.8801
14080/60000 [======>.......................] - ETA: 1:27 - loss: 0.3832 - categorical_accuracy: 0.8802
14112/60000 [======>.......................] - ETA: 1:27 - loss: 0.3828 - categorical_accuracy: 0.8803
14144/60000 [======>.......................] - ETA: 1:27 - loss: 0.3824 - categorical_accuracy: 0.8804
14176/60000 [======>.......................] - ETA: 1:27 - loss: 0.3818 - categorical_accuracy: 0.8806
14208/60000 [======>.......................] - ETA: 1:27 - loss: 0.3814 - categorical_accuracy: 0.8808
14240/60000 [======>.......................] - ETA: 1:27 - loss: 0.3808 - categorical_accuracy: 0.8810
14272/60000 [======>.......................] - ETA: 1:27 - loss: 0.3802 - categorical_accuracy: 0.8812
14304/60000 [======>.......................] - ETA: 1:27 - loss: 0.3796 - categorical_accuracy: 0.8814
14336/60000 [======>.......................] - ETA: 1:27 - loss: 0.3789 - categorical_accuracy: 0.8816
14368/60000 [======>.......................] - ETA: 1:27 - loss: 0.3782 - categorical_accuracy: 0.8818
14400/60000 [======>.......................] - ETA: 1:27 - loss: 0.3778 - categorical_accuracy: 0.8819
14432/60000 [======>.......................] - ETA: 1:26 - loss: 0.3771 - categorical_accuracy: 0.8821
14464/60000 [======>.......................] - ETA: 1:26 - loss: 0.3766 - categorical_accuracy: 0.8823
14496/60000 [======>.......................] - ETA: 1:26 - loss: 0.3760 - categorical_accuracy: 0.8825
14528/60000 [======>.......................] - ETA: 1:26 - loss: 0.3756 - categorical_accuracy: 0.8826
14560/60000 [======>.......................] - ETA: 1:26 - loss: 0.3753 - categorical_accuracy: 0.8828
14592/60000 [======>.......................] - ETA: 1:26 - loss: 0.3750 - categorical_accuracy: 0.8829
14624/60000 [======>.......................] - ETA: 1:26 - loss: 0.3746 - categorical_accuracy: 0.8831
14656/60000 [======>.......................] - ETA: 1:26 - loss: 0.3742 - categorical_accuracy: 0.8832
14688/60000 [======>.......................] - ETA: 1:26 - loss: 0.3735 - categorical_accuracy: 0.8834
14720/60000 [======>.......................] - ETA: 1:26 - loss: 0.3731 - categorical_accuracy: 0.8836
14752/60000 [======>.......................] - ETA: 1:26 - loss: 0.3726 - categorical_accuracy: 0.8837
14784/60000 [======>.......................] - ETA: 1:26 - loss: 0.3720 - categorical_accuracy: 0.8839
14816/60000 [======>.......................] - ETA: 1:26 - loss: 0.3713 - categorical_accuracy: 0.8840
14848/60000 [======>.......................] - ETA: 1:26 - loss: 0.3709 - categorical_accuracy: 0.8842
14880/60000 [======>.......................] - ETA: 1:26 - loss: 0.3702 - categorical_accuracy: 0.8844
14912/60000 [======>.......................] - ETA: 1:25 - loss: 0.3695 - categorical_accuracy: 0.8847
14944/60000 [======>.......................] - ETA: 1:25 - loss: 0.3695 - categorical_accuracy: 0.8846
14976/60000 [======>.......................] - ETA: 1:25 - loss: 0.3693 - categorical_accuracy: 0.8847
15008/60000 [======>.......................] - ETA: 1:25 - loss: 0.3687 - categorical_accuracy: 0.8849
15040/60000 [======>.......................] - ETA: 1:25 - loss: 0.3683 - categorical_accuracy: 0.8850
15072/60000 [======>.......................] - ETA: 1:25 - loss: 0.3681 - categorical_accuracy: 0.8850
15104/60000 [======>.......................] - ETA: 1:25 - loss: 0.3675 - categorical_accuracy: 0.8852
15136/60000 [======>.......................] - ETA: 1:25 - loss: 0.3671 - categorical_accuracy: 0.8852
15168/60000 [======>.......................] - ETA: 1:25 - loss: 0.3667 - categorical_accuracy: 0.8853
15200/60000 [======>.......................] - ETA: 1:25 - loss: 0.3662 - categorical_accuracy: 0.8854
15232/60000 [======>.......................] - ETA: 1:25 - loss: 0.3656 - categorical_accuracy: 0.8856
15264/60000 [======>.......................] - ETA: 1:25 - loss: 0.3650 - categorical_accuracy: 0.8857
15296/60000 [======>.......................] - ETA: 1:25 - loss: 0.3644 - categorical_accuracy: 0.8859
15328/60000 [======>.......................] - ETA: 1:25 - loss: 0.3639 - categorical_accuracy: 0.8860
15360/60000 [======>.......................] - ETA: 1:25 - loss: 0.3642 - categorical_accuracy: 0.8860
15392/60000 [======>.......................] - ETA: 1:25 - loss: 0.3639 - categorical_accuracy: 0.8861
15424/60000 [======>.......................] - ETA: 1:24 - loss: 0.3632 - categorical_accuracy: 0.8863
15456/60000 [======>.......................] - ETA: 1:24 - loss: 0.3630 - categorical_accuracy: 0.8864
15488/60000 [======>.......................] - ETA: 1:24 - loss: 0.3627 - categorical_accuracy: 0.8866
15520/60000 [======>.......................] - ETA: 1:24 - loss: 0.3620 - categorical_accuracy: 0.8868
15552/60000 [======>.......................] - ETA: 1:24 - loss: 0.3619 - categorical_accuracy: 0.8868
15584/60000 [======>.......................] - ETA: 1:24 - loss: 0.3617 - categorical_accuracy: 0.8869
15616/60000 [======>.......................] - ETA: 1:24 - loss: 0.3614 - categorical_accuracy: 0.8870
15648/60000 [======>.......................] - ETA: 1:24 - loss: 0.3607 - categorical_accuracy: 0.8872
15680/60000 [======>.......................] - ETA: 1:24 - loss: 0.3602 - categorical_accuracy: 0.8874
15712/60000 [======>.......................] - ETA: 1:24 - loss: 0.3599 - categorical_accuracy: 0.8875
15744/60000 [======>.......................] - ETA: 1:24 - loss: 0.3595 - categorical_accuracy: 0.8876
15776/60000 [======>.......................] - ETA: 1:24 - loss: 0.3591 - categorical_accuracy: 0.8877
15808/60000 [======>.......................] - ETA: 1:24 - loss: 0.3590 - categorical_accuracy: 0.8877
15840/60000 [======>.......................] - ETA: 1:24 - loss: 0.3586 - categorical_accuracy: 0.8878
15872/60000 [======>.......................] - ETA: 1:24 - loss: 0.3583 - categorical_accuracy: 0.8879
15904/60000 [======>.......................] - ETA: 1:24 - loss: 0.3578 - categorical_accuracy: 0.8880
15936/60000 [======>.......................] - ETA: 1:24 - loss: 0.3577 - categorical_accuracy: 0.8882
15968/60000 [======>.......................] - ETA: 1:23 - loss: 0.3573 - categorical_accuracy: 0.8883
16000/60000 [=======>......................] - ETA: 1:23 - loss: 0.3571 - categorical_accuracy: 0.8884
16032/60000 [=======>......................] - ETA: 1:23 - loss: 0.3566 - categorical_accuracy: 0.8886
16064/60000 [=======>......................] - ETA: 1:23 - loss: 0.3561 - categorical_accuracy: 0.8888
16096/60000 [=======>......................] - ETA: 1:23 - loss: 0.3555 - categorical_accuracy: 0.8889
16128/60000 [=======>......................] - ETA: 1:23 - loss: 0.3550 - categorical_accuracy: 0.8891
16160/60000 [=======>......................] - ETA: 1:23 - loss: 0.3547 - categorical_accuracy: 0.8892
16192/60000 [=======>......................] - ETA: 1:23 - loss: 0.3540 - categorical_accuracy: 0.8894
16224/60000 [=======>......................] - ETA: 1:23 - loss: 0.3536 - categorical_accuracy: 0.8895
16256/60000 [=======>......................] - ETA: 1:23 - loss: 0.3535 - categorical_accuracy: 0.8896
16288/60000 [=======>......................] - ETA: 1:23 - loss: 0.3531 - categorical_accuracy: 0.8897
16320/60000 [=======>......................] - ETA: 1:23 - loss: 0.3525 - categorical_accuracy: 0.8900
16352/60000 [=======>......................] - ETA: 1:23 - loss: 0.3522 - categorical_accuracy: 0.8901
16384/60000 [=======>......................] - ETA: 1:23 - loss: 0.3518 - categorical_accuracy: 0.8902
16416/60000 [=======>......................] - ETA: 1:23 - loss: 0.3513 - categorical_accuracy: 0.8902
16448/60000 [=======>......................] - ETA: 1:23 - loss: 0.3510 - categorical_accuracy: 0.8903
16480/60000 [=======>......................] - ETA: 1:22 - loss: 0.3505 - categorical_accuracy: 0.8905
16512/60000 [=======>......................] - ETA: 1:22 - loss: 0.3499 - categorical_accuracy: 0.8906
16544/60000 [=======>......................] - ETA: 1:22 - loss: 0.3493 - categorical_accuracy: 0.8908
16576/60000 [=======>......................] - ETA: 1:22 - loss: 0.3493 - categorical_accuracy: 0.8909
16608/60000 [=======>......................] - ETA: 1:22 - loss: 0.3489 - categorical_accuracy: 0.8910
16640/60000 [=======>......................] - ETA: 1:22 - loss: 0.3487 - categorical_accuracy: 0.8911
16672/60000 [=======>......................] - ETA: 1:22 - loss: 0.3486 - categorical_accuracy: 0.8911
16704/60000 [=======>......................] - ETA: 1:22 - loss: 0.3480 - categorical_accuracy: 0.8913
16736/60000 [=======>......................] - ETA: 1:22 - loss: 0.3477 - categorical_accuracy: 0.8915
16768/60000 [=======>......................] - ETA: 1:22 - loss: 0.3472 - categorical_accuracy: 0.8916
16800/60000 [=======>......................] - ETA: 1:22 - loss: 0.3470 - categorical_accuracy: 0.8917
16832/60000 [=======>......................] - ETA: 1:22 - loss: 0.3468 - categorical_accuracy: 0.8916
16864/60000 [=======>......................] - ETA: 1:22 - loss: 0.3463 - categorical_accuracy: 0.8918
16896/60000 [=======>......................] - ETA: 1:22 - loss: 0.3458 - categorical_accuracy: 0.8919
16928/60000 [=======>......................] - ETA: 1:22 - loss: 0.3455 - categorical_accuracy: 0.8920
16960/60000 [=======>......................] - ETA: 1:22 - loss: 0.3452 - categorical_accuracy: 0.8920
16992/60000 [=======>......................] - ETA: 1:21 - loss: 0.3448 - categorical_accuracy: 0.8921
17024/60000 [=======>......................] - ETA: 1:21 - loss: 0.3442 - categorical_accuracy: 0.8923
17056/60000 [=======>......................] - ETA: 1:21 - loss: 0.3437 - categorical_accuracy: 0.8924
17088/60000 [=======>......................] - ETA: 1:21 - loss: 0.3432 - categorical_accuracy: 0.8925
17120/60000 [=======>......................] - ETA: 1:21 - loss: 0.3428 - categorical_accuracy: 0.8926
17152/60000 [=======>......................] - ETA: 1:21 - loss: 0.3423 - categorical_accuracy: 0.8927
17184/60000 [=======>......................] - ETA: 1:21 - loss: 0.3419 - categorical_accuracy: 0.8928
17216/60000 [=======>......................] - ETA: 1:21 - loss: 0.3419 - categorical_accuracy: 0.8928
17248/60000 [=======>......................] - ETA: 1:21 - loss: 0.3416 - categorical_accuracy: 0.8929
17280/60000 [=======>......................] - ETA: 1:21 - loss: 0.3410 - categorical_accuracy: 0.8931
17312/60000 [=======>......................] - ETA: 1:21 - loss: 0.3407 - categorical_accuracy: 0.8931
17344/60000 [=======>......................] - ETA: 1:21 - loss: 0.3406 - categorical_accuracy: 0.8933
17376/60000 [=======>......................] - ETA: 1:21 - loss: 0.3400 - categorical_accuracy: 0.8934
17408/60000 [=======>......................] - ETA: 1:21 - loss: 0.3398 - categorical_accuracy: 0.8936
17440/60000 [=======>......................] - ETA: 1:21 - loss: 0.3392 - categorical_accuracy: 0.8938
17472/60000 [=======>......................] - ETA: 1:21 - loss: 0.3387 - categorical_accuracy: 0.8939
17504/60000 [=======>......................] - ETA: 1:21 - loss: 0.3382 - categorical_accuracy: 0.8941
17536/60000 [=======>......................] - ETA: 1:20 - loss: 0.3376 - categorical_accuracy: 0.8943
17568/60000 [=======>......................] - ETA: 1:20 - loss: 0.3371 - categorical_accuracy: 0.8945
17600/60000 [=======>......................] - ETA: 1:20 - loss: 0.3371 - categorical_accuracy: 0.8946
17632/60000 [=======>......................] - ETA: 1:20 - loss: 0.3366 - categorical_accuracy: 0.8948
17664/60000 [=======>......................] - ETA: 1:20 - loss: 0.3363 - categorical_accuracy: 0.8949
17696/60000 [=======>......................] - ETA: 1:20 - loss: 0.3360 - categorical_accuracy: 0.8949
17728/60000 [=======>......................] - ETA: 1:20 - loss: 0.3355 - categorical_accuracy: 0.8951
17760/60000 [=======>......................] - ETA: 1:20 - loss: 0.3349 - categorical_accuracy: 0.8953
17792/60000 [=======>......................] - ETA: 1:20 - loss: 0.3344 - categorical_accuracy: 0.8955
17824/60000 [=======>......................] - ETA: 1:20 - loss: 0.3339 - categorical_accuracy: 0.8957
17856/60000 [=======>......................] - ETA: 1:20 - loss: 0.3334 - categorical_accuracy: 0.8958
17888/60000 [=======>......................] - ETA: 1:20 - loss: 0.3329 - categorical_accuracy: 0.8959
17920/60000 [=======>......................] - ETA: 1:20 - loss: 0.3325 - categorical_accuracy: 0.8960
17952/60000 [=======>......................] - ETA: 1:20 - loss: 0.3320 - categorical_accuracy: 0.8962
17984/60000 [=======>......................] - ETA: 1:20 - loss: 0.3315 - categorical_accuracy: 0.8964
18016/60000 [========>.....................] - ETA: 1:20 - loss: 0.3312 - categorical_accuracy: 0.8965
18048/60000 [========>.....................] - ETA: 1:20 - loss: 0.3309 - categorical_accuracy: 0.8966
18080/60000 [========>.....................] - ETA: 1:19 - loss: 0.3305 - categorical_accuracy: 0.8967
18112/60000 [========>.....................] - ETA: 1:19 - loss: 0.3305 - categorical_accuracy: 0.8968
18144/60000 [========>.....................] - ETA: 1:19 - loss: 0.3304 - categorical_accuracy: 0.8967
18176/60000 [========>.....................] - ETA: 1:19 - loss: 0.3303 - categorical_accuracy: 0.8966
18208/60000 [========>.....................] - ETA: 1:19 - loss: 0.3298 - categorical_accuracy: 0.8968
18240/60000 [========>.....................] - ETA: 1:19 - loss: 0.3294 - categorical_accuracy: 0.8969
18272/60000 [========>.....................] - ETA: 1:19 - loss: 0.3292 - categorical_accuracy: 0.8969
18304/60000 [========>.....................] - ETA: 1:19 - loss: 0.3288 - categorical_accuracy: 0.8971
18336/60000 [========>.....................] - ETA: 1:19 - loss: 0.3288 - categorical_accuracy: 0.8971
18368/60000 [========>.....................] - ETA: 1:19 - loss: 0.3284 - categorical_accuracy: 0.8972
18400/60000 [========>.....................] - ETA: 1:19 - loss: 0.3283 - categorical_accuracy: 0.8973
18432/60000 [========>.....................] - ETA: 1:19 - loss: 0.3279 - categorical_accuracy: 0.8974
18464/60000 [========>.....................] - ETA: 1:19 - loss: 0.3274 - categorical_accuracy: 0.8976
18496/60000 [========>.....................] - ETA: 1:19 - loss: 0.3270 - categorical_accuracy: 0.8977
18528/60000 [========>.....................] - ETA: 1:19 - loss: 0.3268 - categorical_accuracy: 0.8978
18560/60000 [========>.....................] - ETA: 1:19 - loss: 0.3266 - categorical_accuracy: 0.8979
18592/60000 [========>.....................] - ETA: 1:18 - loss: 0.3263 - categorical_accuracy: 0.8980
18624/60000 [========>.....................] - ETA: 1:18 - loss: 0.3260 - categorical_accuracy: 0.8981
18656/60000 [========>.....................] - ETA: 1:18 - loss: 0.3259 - categorical_accuracy: 0.8982
18688/60000 [========>.....................] - ETA: 1:18 - loss: 0.3254 - categorical_accuracy: 0.8983
18720/60000 [========>.....................] - ETA: 1:18 - loss: 0.3251 - categorical_accuracy: 0.8983
18752/60000 [========>.....................] - ETA: 1:18 - loss: 0.3248 - categorical_accuracy: 0.8984
18784/60000 [========>.....................] - ETA: 1:18 - loss: 0.3243 - categorical_accuracy: 0.8986
18816/60000 [========>.....................] - ETA: 1:18 - loss: 0.3242 - categorical_accuracy: 0.8986
18848/60000 [========>.....................] - ETA: 1:18 - loss: 0.3239 - categorical_accuracy: 0.8987
18880/60000 [========>.....................] - ETA: 1:18 - loss: 0.3235 - categorical_accuracy: 0.8988
18912/60000 [========>.....................] - ETA: 1:18 - loss: 0.3233 - categorical_accuracy: 0.8989
18944/60000 [========>.....................] - ETA: 1:18 - loss: 0.3234 - categorical_accuracy: 0.8989
18976/60000 [========>.....................] - ETA: 1:18 - loss: 0.3231 - categorical_accuracy: 0.8990
19008/60000 [========>.....................] - ETA: 1:18 - loss: 0.3229 - categorical_accuracy: 0.8991
19040/60000 [========>.....................] - ETA: 1:18 - loss: 0.3227 - categorical_accuracy: 0.8992
19072/60000 [========>.....................] - ETA: 1:18 - loss: 0.3223 - categorical_accuracy: 0.8993
19104/60000 [========>.....................] - ETA: 1:17 - loss: 0.3222 - categorical_accuracy: 0.8994
19136/60000 [========>.....................] - ETA: 1:17 - loss: 0.3218 - categorical_accuracy: 0.8995
19168/60000 [========>.....................] - ETA: 1:17 - loss: 0.3218 - categorical_accuracy: 0.8996
19200/60000 [========>.....................] - ETA: 1:17 - loss: 0.3217 - categorical_accuracy: 0.8997
19232/60000 [========>.....................] - ETA: 1:17 - loss: 0.3215 - categorical_accuracy: 0.8998
19264/60000 [========>.....................] - ETA: 1:17 - loss: 0.3211 - categorical_accuracy: 0.8999
19296/60000 [========>.....................] - ETA: 1:17 - loss: 0.3207 - categorical_accuracy: 0.9000
19328/60000 [========>.....................] - ETA: 1:17 - loss: 0.3203 - categorical_accuracy: 0.9001
19360/60000 [========>.....................] - ETA: 1:17 - loss: 0.3202 - categorical_accuracy: 0.9002
19392/60000 [========>.....................] - ETA: 1:17 - loss: 0.3198 - categorical_accuracy: 0.9002
19424/60000 [========>.....................] - ETA: 1:17 - loss: 0.3198 - categorical_accuracy: 0.9003
19456/60000 [========>.....................] - ETA: 1:17 - loss: 0.3194 - categorical_accuracy: 0.9004
19488/60000 [========>.....................] - ETA: 1:17 - loss: 0.3191 - categorical_accuracy: 0.9005
19520/60000 [========>.....................] - ETA: 1:17 - loss: 0.3189 - categorical_accuracy: 0.9005
19552/60000 [========>.....................] - ETA: 1:17 - loss: 0.3187 - categorical_accuracy: 0.9006
19584/60000 [========>.....................] - ETA: 1:17 - loss: 0.3183 - categorical_accuracy: 0.9007
19616/60000 [========>.....................] - ETA: 1:16 - loss: 0.3180 - categorical_accuracy: 0.9008
19648/60000 [========>.....................] - ETA: 1:16 - loss: 0.3176 - categorical_accuracy: 0.9010
19680/60000 [========>.....................] - ETA: 1:16 - loss: 0.3176 - categorical_accuracy: 0.9010
19712/60000 [========>.....................] - ETA: 1:16 - loss: 0.3175 - categorical_accuracy: 0.9010
19744/60000 [========>.....................] - ETA: 1:16 - loss: 0.3170 - categorical_accuracy: 0.9011
19776/60000 [========>.....................] - ETA: 1:16 - loss: 0.3166 - categorical_accuracy: 0.9013
19808/60000 [========>.....................] - ETA: 1:16 - loss: 0.3161 - categorical_accuracy: 0.9014
19840/60000 [========>.....................] - ETA: 1:16 - loss: 0.3159 - categorical_accuracy: 0.9015
19872/60000 [========>.....................] - ETA: 1:16 - loss: 0.3155 - categorical_accuracy: 0.9016
19904/60000 [========>.....................] - ETA: 1:16 - loss: 0.3153 - categorical_accuracy: 0.9016
19936/60000 [========>.....................] - ETA: 1:16 - loss: 0.3156 - categorical_accuracy: 0.9016
19968/60000 [========>.....................] - ETA: 1:16 - loss: 0.3152 - categorical_accuracy: 0.9017
20000/60000 [=========>....................] - ETA: 1:16 - loss: 0.3152 - categorical_accuracy: 0.9018
20032/60000 [=========>....................] - ETA: 1:16 - loss: 0.3150 - categorical_accuracy: 0.9019
20064/60000 [=========>....................] - ETA: 1:16 - loss: 0.3145 - categorical_accuracy: 0.9021
20096/60000 [=========>....................] - ETA: 1:16 - loss: 0.3143 - categorical_accuracy: 0.9021
20128/60000 [=========>....................] - ETA: 1:16 - loss: 0.3142 - categorical_accuracy: 0.9021
20160/60000 [=========>....................] - ETA: 1:15 - loss: 0.3139 - categorical_accuracy: 0.9022
20192/60000 [=========>....................] - ETA: 1:15 - loss: 0.3136 - categorical_accuracy: 0.9023
20224/60000 [=========>....................] - ETA: 1:15 - loss: 0.3137 - categorical_accuracy: 0.9023
20256/60000 [=========>....................] - ETA: 1:15 - loss: 0.3134 - categorical_accuracy: 0.9023
20288/60000 [=========>....................] - ETA: 1:15 - loss: 0.3133 - categorical_accuracy: 0.9025
20320/60000 [=========>....................] - ETA: 1:15 - loss: 0.3134 - categorical_accuracy: 0.9024
20352/60000 [=========>....................] - ETA: 1:15 - loss: 0.3129 - categorical_accuracy: 0.9025
20384/60000 [=========>....................] - ETA: 1:15 - loss: 0.3129 - categorical_accuracy: 0.9025
20416/60000 [=========>....................] - ETA: 1:15 - loss: 0.3125 - categorical_accuracy: 0.9027
20448/60000 [=========>....................] - ETA: 1:15 - loss: 0.3125 - categorical_accuracy: 0.9027
20480/60000 [=========>....................] - ETA: 1:15 - loss: 0.3122 - categorical_accuracy: 0.9027
20512/60000 [=========>....................] - ETA: 1:15 - loss: 0.3121 - categorical_accuracy: 0.9027
20544/60000 [=========>....................] - ETA: 1:15 - loss: 0.3118 - categorical_accuracy: 0.9028
20576/60000 [=========>....................] - ETA: 1:15 - loss: 0.3117 - categorical_accuracy: 0.9029
20608/60000 [=========>....................] - ETA: 1:15 - loss: 0.3113 - categorical_accuracy: 0.9030
20640/60000 [=========>....................] - ETA: 1:15 - loss: 0.3112 - categorical_accuracy: 0.9031
20672/60000 [=========>....................] - ETA: 1:14 - loss: 0.3110 - categorical_accuracy: 0.9032
20704/60000 [=========>....................] - ETA: 1:14 - loss: 0.3105 - categorical_accuracy: 0.9033
20736/60000 [=========>....................] - ETA: 1:14 - loss: 0.3102 - categorical_accuracy: 0.9035
20768/60000 [=========>....................] - ETA: 1:14 - loss: 0.3099 - categorical_accuracy: 0.9036
20800/60000 [=========>....................] - ETA: 1:14 - loss: 0.3095 - categorical_accuracy: 0.9037
20832/60000 [=========>....................] - ETA: 1:14 - loss: 0.3093 - categorical_accuracy: 0.9038
20864/60000 [=========>....................] - ETA: 1:14 - loss: 0.3091 - categorical_accuracy: 0.9038
20896/60000 [=========>....................] - ETA: 1:14 - loss: 0.3089 - categorical_accuracy: 0.9039
20928/60000 [=========>....................] - ETA: 1:14 - loss: 0.3086 - categorical_accuracy: 0.9040
20960/60000 [=========>....................] - ETA: 1:14 - loss: 0.3082 - categorical_accuracy: 0.9041
20992/60000 [=========>....................] - ETA: 1:14 - loss: 0.3078 - categorical_accuracy: 0.9042
21024/60000 [=========>....................] - ETA: 1:14 - loss: 0.3078 - categorical_accuracy: 0.9042
21056/60000 [=========>....................] - ETA: 1:14 - loss: 0.3073 - categorical_accuracy: 0.9044
21088/60000 [=========>....................] - ETA: 1:14 - loss: 0.3069 - categorical_accuracy: 0.9044
21120/60000 [=========>....................] - ETA: 1:14 - loss: 0.3068 - categorical_accuracy: 0.9044
21152/60000 [=========>....................] - ETA: 1:14 - loss: 0.3064 - categorical_accuracy: 0.9045
21184/60000 [=========>....................] - ETA: 1:13 - loss: 0.3064 - categorical_accuracy: 0.9046
21216/60000 [=========>....................] - ETA: 1:13 - loss: 0.3062 - categorical_accuracy: 0.9046
21248/60000 [=========>....................] - ETA: 1:13 - loss: 0.3058 - categorical_accuracy: 0.9047
21280/60000 [=========>....................] - ETA: 1:13 - loss: 0.3055 - categorical_accuracy: 0.9048
21312/60000 [=========>....................] - ETA: 1:13 - loss: 0.3054 - categorical_accuracy: 0.9048
21344/60000 [=========>....................] - ETA: 1:13 - loss: 0.3049 - categorical_accuracy: 0.9049
21376/60000 [=========>....................] - ETA: 1:13 - loss: 0.3045 - categorical_accuracy: 0.9051
21408/60000 [=========>....................] - ETA: 1:13 - loss: 0.3042 - categorical_accuracy: 0.9052
21440/60000 [=========>....................] - ETA: 1:13 - loss: 0.3037 - categorical_accuracy: 0.9054
21472/60000 [=========>....................] - ETA: 1:13 - loss: 0.3033 - categorical_accuracy: 0.9055
21504/60000 [=========>....................] - ETA: 1:13 - loss: 0.3030 - categorical_accuracy: 0.9056
21536/60000 [=========>....................] - ETA: 1:13 - loss: 0.3029 - categorical_accuracy: 0.9057
21568/60000 [=========>....................] - ETA: 1:13 - loss: 0.3026 - categorical_accuracy: 0.9057
21600/60000 [=========>....................] - ETA: 1:13 - loss: 0.3024 - categorical_accuracy: 0.9058
21632/60000 [=========>....................] - ETA: 1:13 - loss: 0.3019 - categorical_accuracy: 0.9060
21664/60000 [=========>....................] - ETA: 1:13 - loss: 0.3016 - categorical_accuracy: 0.9061
21696/60000 [=========>....................] - ETA: 1:12 - loss: 0.3013 - categorical_accuracy: 0.9062
21728/60000 [=========>....................] - ETA: 1:12 - loss: 0.3008 - categorical_accuracy: 0.9063
21760/60000 [=========>....................] - ETA: 1:12 - loss: 0.3006 - categorical_accuracy: 0.9064
21792/60000 [=========>....................] - ETA: 1:12 - loss: 0.3002 - categorical_accuracy: 0.9065
21824/60000 [=========>....................] - ETA: 1:12 - loss: 0.3001 - categorical_accuracy: 0.9066
21856/60000 [=========>....................] - ETA: 1:12 - loss: 0.2998 - categorical_accuracy: 0.9067
21888/60000 [=========>....................] - ETA: 1:12 - loss: 0.2995 - categorical_accuracy: 0.9068
21920/60000 [=========>....................] - ETA: 1:12 - loss: 0.2993 - categorical_accuracy: 0.9068
21952/60000 [=========>....................] - ETA: 1:12 - loss: 0.2989 - categorical_accuracy: 0.9070
21984/60000 [=========>....................] - ETA: 1:12 - loss: 0.2990 - categorical_accuracy: 0.9071
22016/60000 [==========>...................] - ETA: 1:12 - loss: 0.2989 - categorical_accuracy: 0.9071
22048/60000 [==========>...................] - ETA: 1:12 - loss: 0.2986 - categorical_accuracy: 0.9072
22080/60000 [==========>...................] - ETA: 1:12 - loss: 0.2983 - categorical_accuracy: 0.9073
22112/60000 [==========>...................] - ETA: 1:12 - loss: 0.2982 - categorical_accuracy: 0.9073
22144/60000 [==========>...................] - ETA: 1:12 - loss: 0.2979 - categorical_accuracy: 0.9075
22176/60000 [==========>...................] - ETA: 1:12 - loss: 0.2978 - categorical_accuracy: 0.9075
22208/60000 [==========>...................] - ETA: 1:11 - loss: 0.2975 - categorical_accuracy: 0.9076
22240/60000 [==========>...................] - ETA: 1:11 - loss: 0.2973 - categorical_accuracy: 0.9076
22272/60000 [==========>...................] - ETA: 1:11 - loss: 0.2972 - categorical_accuracy: 0.9077
22304/60000 [==========>...................] - ETA: 1:11 - loss: 0.2970 - categorical_accuracy: 0.9077
22336/60000 [==========>...................] - ETA: 1:11 - loss: 0.2968 - categorical_accuracy: 0.9077
22368/60000 [==========>...................] - ETA: 1:11 - loss: 0.2966 - categorical_accuracy: 0.9078
22400/60000 [==========>...................] - ETA: 1:11 - loss: 0.2964 - categorical_accuracy: 0.9079
22432/60000 [==========>...................] - ETA: 1:11 - loss: 0.2962 - categorical_accuracy: 0.9079
22464/60000 [==========>...................] - ETA: 1:11 - loss: 0.2960 - categorical_accuracy: 0.9079
22496/60000 [==========>...................] - ETA: 1:11 - loss: 0.2962 - categorical_accuracy: 0.9079
22528/60000 [==========>...................] - ETA: 1:11 - loss: 0.2960 - categorical_accuracy: 0.9079
22560/60000 [==========>...................] - ETA: 1:11 - loss: 0.2964 - categorical_accuracy: 0.9078
22592/60000 [==========>...................] - ETA: 1:11 - loss: 0.2963 - categorical_accuracy: 0.9079
22624/60000 [==========>...................] - ETA: 1:11 - loss: 0.2962 - categorical_accuracy: 0.9079
22656/60000 [==========>...................] - ETA: 1:11 - loss: 0.2958 - categorical_accuracy: 0.9081
22688/60000 [==========>...................] - ETA: 1:11 - loss: 0.2954 - categorical_accuracy: 0.9082
22720/60000 [==========>...................] - ETA: 1:10 - loss: 0.2953 - categorical_accuracy: 0.9082
22752/60000 [==========>...................] - ETA: 1:10 - loss: 0.2951 - categorical_accuracy: 0.9083
22784/60000 [==========>...................] - ETA: 1:10 - loss: 0.2947 - categorical_accuracy: 0.9084
22816/60000 [==========>...................] - ETA: 1:10 - loss: 0.2949 - categorical_accuracy: 0.9084
22848/60000 [==========>...................] - ETA: 1:10 - loss: 0.2945 - categorical_accuracy: 0.9084
22880/60000 [==========>...................] - ETA: 1:10 - loss: 0.2944 - categorical_accuracy: 0.9085
22912/60000 [==========>...................] - ETA: 1:10 - loss: 0.2944 - categorical_accuracy: 0.9085
22944/60000 [==========>...................] - ETA: 1:10 - loss: 0.2941 - categorical_accuracy: 0.9085
22976/60000 [==========>...................] - ETA: 1:10 - loss: 0.2938 - categorical_accuracy: 0.9086
23008/60000 [==========>...................] - ETA: 1:10 - loss: 0.2934 - categorical_accuracy: 0.9087
23040/60000 [==========>...................] - ETA: 1:10 - loss: 0.2933 - categorical_accuracy: 0.9087
23072/60000 [==========>...................] - ETA: 1:10 - loss: 0.2933 - categorical_accuracy: 0.9088
23104/60000 [==========>...................] - ETA: 1:10 - loss: 0.2934 - categorical_accuracy: 0.9088
23136/60000 [==========>...................] - ETA: 1:10 - loss: 0.2932 - categorical_accuracy: 0.9089
23168/60000 [==========>...................] - ETA: 1:10 - loss: 0.2929 - categorical_accuracy: 0.9090
23200/60000 [==========>...................] - ETA: 1:10 - loss: 0.2925 - categorical_accuracy: 0.9091
23232/60000 [==========>...................] - ETA: 1:10 - loss: 0.2925 - categorical_accuracy: 0.9092
23264/60000 [==========>...................] - ETA: 1:09 - loss: 0.2922 - categorical_accuracy: 0.9093
23296/60000 [==========>...................] - ETA: 1:09 - loss: 0.2921 - categorical_accuracy: 0.9093
23328/60000 [==========>...................] - ETA: 1:09 - loss: 0.2921 - categorical_accuracy: 0.9093
23360/60000 [==========>...................] - ETA: 1:09 - loss: 0.2920 - categorical_accuracy: 0.9092
23392/60000 [==========>...................] - ETA: 1:09 - loss: 0.2917 - categorical_accuracy: 0.9094
23424/60000 [==========>...................] - ETA: 1:09 - loss: 0.2918 - categorical_accuracy: 0.9094
23456/60000 [==========>...................] - ETA: 1:09 - loss: 0.2914 - categorical_accuracy: 0.9095
23488/60000 [==========>...................] - ETA: 1:09 - loss: 0.2912 - categorical_accuracy: 0.9096
23520/60000 [==========>...................] - ETA: 1:09 - loss: 0.2909 - categorical_accuracy: 0.9097
23552/60000 [==========>...................] - ETA: 1:09 - loss: 0.2905 - categorical_accuracy: 0.9098
23584/60000 [==========>...................] - ETA: 1:09 - loss: 0.2902 - categorical_accuracy: 0.9099
23616/60000 [==========>...................] - ETA: 1:09 - loss: 0.2901 - categorical_accuracy: 0.9099
23648/60000 [==========>...................] - ETA: 1:09 - loss: 0.2897 - categorical_accuracy: 0.9101
23680/60000 [==========>...................] - ETA: 1:09 - loss: 0.2895 - categorical_accuracy: 0.9101
23712/60000 [==========>...................] - ETA: 1:09 - loss: 0.2894 - categorical_accuracy: 0.9102
23744/60000 [==========>...................] - ETA: 1:09 - loss: 0.2892 - categorical_accuracy: 0.9102
23776/60000 [==========>...................] - ETA: 1:08 - loss: 0.2890 - categorical_accuracy: 0.9103
23808/60000 [==========>...................] - ETA: 1:08 - loss: 0.2887 - categorical_accuracy: 0.9104
23840/60000 [==========>...................] - ETA: 1:08 - loss: 0.2885 - categorical_accuracy: 0.9104
23872/60000 [==========>...................] - ETA: 1:08 - loss: 0.2884 - categorical_accuracy: 0.9105
23904/60000 [==========>...................] - ETA: 1:08 - loss: 0.2882 - categorical_accuracy: 0.9106
23936/60000 [==========>...................] - ETA: 1:08 - loss: 0.2879 - categorical_accuracy: 0.9107
23968/60000 [==========>...................] - ETA: 1:08 - loss: 0.2878 - categorical_accuracy: 0.9107
24000/60000 [===========>..................] - ETA: 1:08 - loss: 0.2876 - categorical_accuracy: 0.9107
24032/60000 [===========>..................] - ETA: 1:08 - loss: 0.2873 - categorical_accuracy: 0.9109
24064/60000 [===========>..................] - ETA: 1:08 - loss: 0.2871 - categorical_accuracy: 0.9109
24096/60000 [===========>..................] - ETA: 1:08 - loss: 0.2871 - categorical_accuracy: 0.9109
24128/60000 [===========>..................] - ETA: 1:08 - loss: 0.2871 - categorical_accuracy: 0.9109
24160/60000 [===========>..................] - ETA: 1:08 - loss: 0.2873 - categorical_accuracy: 0.9109
24192/60000 [===========>..................] - ETA: 1:08 - loss: 0.2871 - categorical_accuracy: 0.9109
24224/60000 [===========>..................] - ETA: 1:08 - loss: 0.2868 - categorical_accuracy: 0.9110
24256/60000 [===========>..................] - ETA: 1:08 - loss: 0.2867 - categorical_accuracy: 0.9111
24288/60000 [===========>..................] - ETA: 1:07 - loss: 0.2864 - categorical_accuracy: 0.9111
24320/60000 [===========>..................] - ETA: 1:07 - loss: 0.2862 - categorical_accuracy: 0.9112
24352/60000 [===========>..................] - ETA: 1:07 - loss: 0.2861 - categorical_accuracy: 0.9113
24384/60000 [===========>..................] - ETA: 1:07 - loss: 0.2859 - categorical_accuracy: 0.9114
24416/60000 [===========>..................] - ETA: 1:07 - loss: 0.2858 - categorical_accuracy: 0.9114
24448/60000 [===========>..................] - ETA: 1:07 - loss: 0.2855 - categorical_accuracy: 0.9115
24480/60000 [===========>..................] - ETA: 1:07 - loss: 0.2853 - categorical_accuracy: 0.9116
24512/60000 [===========>..................] - ETA: 1:07 - loss: 0.2850 - categorical_accuracy: 0.9116
24544/60000 [===========>..................] - ETA: 1:07 - loss: 0.2849 - categorical_accuracy: 0.9117
24576/60000 [===========>..................] - ETA: 1:07 - loss: 0.2847 - categorical_accuracy: 0.9118
24608/60000 [===========>..................] - ETA: 1:07 - loss: 0.2846 - categorical_accuracy: 0.9117
24640/60000 [===========>..................] - ETA: 1:07 - loss: 0.2843 - categorical_accuracy: 0.9118
24672/60000 [===========>..................] - ETA: 1:07 - loss: 0.2841 - categorical_accuracy: 0.9119
24704/60000 [===========>..................] - ETA: 1:07 - loss: 0.2842 - categorical_accuracy: 0.9118
24736/60000 [===========>..................] - ETA: 1:07 - loss: 0.2838 - categorical_accuracy: 0.9120
24768/60000 [===========>..................] - ETA: 1:07 - loss: 0.2838 - categorical_accuracy: 0.9120
24800/60000 [===========>..................] - ETA: 1:06 - loss: 0.2836 - categorical_accuracy: 0.9120
24832/60000 [===========>..................] - ETA: 1:06 - loss: 0.2833 - categorical_accuracy: 0.9121
24864/60000 [===========>..................] - ETA: 1:06 - loss: 0.2830 - categorical_accuracy: 0.9122
24896/60000 [===========>..................] - ETA: 1:06 - loss: 0.2827 - categorical_accuracy: 0.9123
24928/60000 [===========>..................] - ETA: 1:06 - loss: 0.2823 - categorical_accuracy: 0.9124
24960/60000 [===========>..................] - ETA: 1:06 - loss: 0.2821 - categorical_accuracy: 0.9125
24992/60000 [===========>..................] - ETA: 1:06 - loss: 0.2819 - categorical_accuracy: 0.9125
25024/60000 [===========>..................] - ETA: 1:06 - loss: 0.2816 - categorical_accuracy: 0.9126
25056/60000 [===========>..................] - ETA: 1:06 - loss: 0.2814 - categorical_accuracy: 0.9127
25088/60000 [===========>..................] - ETA: 1:06 - loss: 0.2811 - categorical_accuracy: 0.9128
25120/60000 [===========>..................] - ETA: 1:06 - loss: 0.2809 - categorical_accuracy: 0.9128
25152/60000 [===========>..................] - ETA: 1:06 - loss: 0.2808 - categorical_accuracy: 0.9128
25184/60000 [===========>..................] - ETA: 1:06 - loss: 0.2805 - categorical_accuracy: 0.9130
25216/60000 [===========>..................] - ETA: 1:06 - loss: 0.2801 - categorical_accuracy: 0.9131
25248/60000 [===========>..................] - ETA: 1:06 - loss: 0.2799 - categorical_accuracy: 0.9131
25280/60000 [===========>..................] - ETA: 1:06 - loss: 0.2798 - categorical_accuracy: 0.9131
25312/60000 [===========>..................] - ETA: 1:05 - loss: 0.2794 - categorical_accuracy: 0.9132
25344/60000 [===========>..................] - ETA: 1:05 - loss: 0.2792 - categorical_accuracy: 0.9133
25376/60000 [===========>..................] - ETA: 1:05 - loss: 0.2788 - categorical_accuracy: 0.9134
25408/60000 [===========>..................] - ETA: 1:05 - loss: 0.2786 - categorical_accuracy: 0.9135
25440/60000 [===========>..................] - ETA: 1:05 - loss: 0.2784 - categorical_accuracy: 0.9136
25472/60000 [===========>..................] - ETA: 1:05 - loss: 0.2788 - categorical_accuracy: 0.9135
25504/60000 [===========>..................] - ETA: 1:05 - loss: 0.2786 - categorical_accuracy: 0.9136
25536/60000 [===========>..................] - ETA: 1:05 - loss: 0.2784 - categorical_accuracy: 0.9137
25568/60000 [===========>..................] - ETA: 1:05 - loss: 0.2784 - categorical_accuracy: 0.9137
25600/60000 [===========>..................] - ETA: 1:05 - loss: 0.2782 - categorical_accuracy: 0.9138
25632/60000 [===========>..................] - ETA: 1:05 - loss: 0.2781 - categorical_accuracy: 0.9138
25664/60000 [===========>..................] - ETA: 1:05 - loss: 0.2778 - categorical_accuracy: 0.9138
25696/60000 [===========>..................] - ETA: 1:05 - loss: 0.2777 - categorical_accuracy: 0.9139
25728/60000 [===========>..................] - ETA: 1:05 - loss: 0.2774 - categorical_accuracy: 0.9140
25760/60000 [===========>..................] - ETA: 1:05 - loss: 0.2773 - categorical_accuracy: 0.9140
25792/60000 [===========>..................] - ETA: 1:05 - loss: 0.2770 - categorical_accuracy: 0.9141
25824/60000 [===========>..................] - ETA: 1:04 - loss: 0.2768 - categorical_accuracy: 0.9141
25856/60000 [===========>..................] - ETA: 1:04 - loss: 0.2765 - categorical_accuracy: 0.9143
25888/60000 [===========>..................] - ETA: 1:04 - loss: 0.2768 - categorical_accuracy: 0.9143
25920/60000 [===========>..................] - ETA: 1:04 - loss: 0.2765 - categorical_accuracy: 0.9144
25952/60000 [===========>..................] - ETA: 1:04 - loss: 0.2764 - categorical_accuracy: 0.9144
25984/60000 [===========>..................] - ETA: 1:04 - loss: 0.2760 - categorical_accuracy: 0.9145
26016/60000 [============>.................] - ETA: 1:04 - loss: 0.2757 - categorical_accuracy: 0.9146
26048/60000 [============>.................] - ETA: 1:04 - loss: 0.2755 - categorical_accuracy: 0.9147
26080/60000 [============>.................] - ETA: 1:04 - loss: 0.2752 - categorical_accuracy: 0.9147
26112/60000 [============>.................] - ETA: 1:04 - loss: 0.2754 - categorical_accuracy: 0.9146
26144/60000 [============>.................] - ETA: 1:04 - loss: 0.2753 - categorical_accuracy: 0.9146
26176/60000 [============>.................] - ETA: 1:04 - loss: 0.2752 - categorical_accuracy: 0.9146
26208/60000 [============>.................] - ETA: 1:04 - loss: 0.2750 - categorical_accuracy: 0.9146
26240/60000 [============>.................] - ETA: 1:04 - loss: 0.2750 - categorical_accuracy: 0.9146
26272/60000 [============>.................] - ETA: 1:04 - loss: 0.2748 - categorical_accuracy: 0.9147
26304/60000 [============>.................] - ETA: 1:04 - loss: 0.2745 - categorical_accuracy: 0.9147
26336/60000 [============>.................] - ETA: 1:03 - loss: 0.2743 - categorical_accuracy: 0.9148
26368/60000 [============>.................] - ETA: 1:03 - loss: 0.2741 - categorical_accuracy: 0.9149
26400/60000 [============>.................] - ETA: 1:03 - loss: 0.2738 - categorical_accuracy: 0.9149
26432/60000 [============>.................] - ETA: 1:03 - loss: 0.2739 - categorical_accuracy: 0.9150
26464/60000 [============>.................] - ETA: 1:03 - loss: 0.2739 - categorical_accuracy: 0.9149
26496/60000 [============>.................] - ETA: 1:03 - loss: 0.2736 - categorical_accuracy: 0.9150
26528/60000 [============>.................] - ETA: 1:03 - loss: 0.2734 - categorical_accuracy: 0.9151
26560/60000 [============>.................] - ETA: 1:03 - loss: 0.2732 - categorical_accuracy: 0.9152
26592/60000 [============>.................] - ETA: 1:03 - loss: 0.2730 - categorical_accuracy: 0.9152
26624/60000 [============>.................] - ETA: 1:03 - loss: 0.2730 - categorical_accuracy: 0.9153
26656/60000 [============>.................] - ETA: 1:03 - loss: 0.2731 - categorical_accuracy: 0.9153
26688/60000 [============>.................] - ETA: 1:03 - loss: 0.2730 - categorical_accuracy: 0.9154
26720/60000 [============>.................] - ETA: 1:03 - loss: 0.2732 - categorical_accuracy: 0.9153
26752/60000 [============>.................] - ETA: 1:03 - loss: 0.2731 - categorical_accuracy: 0.9154
26784/60000 [============>.................] - ETA: 1:03 - loss: 0.2728 - categorical_accuracy: 0.9155
26816/60000 [============>.................] - ETA: 1:03 - loss: 0.2726 - categorical_accuracy: 0.9156
26848/60000 [============>.................] - ETA: 1:03 - loss: 0.2724 - categorical_accuracy: 0.9157
26880/60000 [============>.................] - ETA: 1:02 - loss: 0.2722 - categorical_accuracy: 0.9157
26912/60000 [============>.................] - ETA: 1:02 - loss: 0.2720 - categorical_accuracy: 0.9158
26944/60000 [============>.................] - ETA: 1:02 - loss: 0.2717 - categorical_accuracy: 0.9159
26976/60000 [============>.................] - ETA: 1:02 - loss: 0.2715 - categorical_accuracy: 0.9160
27008/60000 [============>.................] - ETA: 1:02 - loss: 0.2712 - categorical_accuracy: 0.9161
27040/60000 [============>.................] - ETA: 1:02 - loss: 0.2710 - categorical_accuracy: 0.9162
27072/60000 [============>.................] - ETA: 1:02 - loss: 0.2711 - categorical_accuracy: 0.9161
27104/60000 [============>.................] - ETA: 1:02 - loss: 0.2709 - categorical_accuracy: 0.9162
27136/60000 [============>.................] - ETA: 1:02 - loss: 0.2709 - categorical_accuracy: 0.9162
27168/60000 [============>.................] - ETA: 1:02 - loss: 0.2707 - categorical_accuracy: 0.9163
27200/60000 [============>.................] - ETA: 1:02 - loss: 0.2705 - categorical_accuracy: 0.9163
27232/60000 [============>.................] - ETA: 1:02 - loss: 0.2702 - categorical_accuracy: 0.9164
27264/60000 [============>.................] - ETA: 1:02 - loss: 0.2702 - categorical_accuracy: 0.9164
27296/60000 [============>.................] - ETA: 1:02 - loss: 0.2699 - categorical_accuracy: 0.9165
27328/60000 [============>.................] - ETA: 1:02 - loss: 0.2698 - categorical_accuracy: 0.9165
27360/60000 [============>.................] - ETA: 1:01 - loss: 0.2703 - categorical_accuracy: 0.9164
27392/60000 [============>.................] - ETA: 1:01 - loss: 0.2700 - categorical_accuracy: 0.9165
27424/60000 [============>.................] - ETA: 1:01 - loss: 0.2698 - categorical_accuracy: 0.9166
27456/60000 [============>.................] - ETA: 1:01 - loss: 0.2696 - categorical_accuracy: 0.9167
27488/60000 [============>.................] - ETA: 1:01 - loss: 0.2695 - categorical_accuracy: 0.9167
27520/60000 [============>.................] - ETA: 1:01 - loss: 0.2693 - categorical_accuracy: 0.9168
27552/60000 [============>.................] - ETA: 1:01 - loss: 0.2692 - categorical_accuracy: 0.9168
27584/60000 [============>.................] - ETA: 1:01 - loss: 0.2689 - categorical_accuracy: 0.9169
27616/60000 [============>.................] - ETA: 1:01 - loss: 0.2689 - categorical_accuracy: 0.9169
27648/60000 [============>.................] - ETA: 1:01 - loss: 0.2686 - categorical_accuracy: 0.9170
27680/60000 [============>.................] - ETA: 1:01 - loss: 0.2684 - categorical_accuracy: 0.9171
27712/60000 [============>.................] - ETA: 1:01 - loss: 0.2682 - categorical_accuracy: 0.9172
27744/60000 [============>.................] - ETA: 1:01 - loss: 0.2679 - categorical_accuracy: 0.9173
27776/60000 [============>.................] - ETA: 1:01 - loss: 0.2679 - categorical_accuracy: 0.9173
27808/60000 [============>.................] - ETA: 1:01 - loss: 0.2677 - categorical_accuracy: 0.9173
27840/60000 [============>.................] - ETA: 1:01 - loss: 0.2678 - categorical_accuracy: 0.9173
27872/60000 [============>.................] - ETA: 1:00 - loss: 0.2676 - categorical_accuracy: 0.9173
27904/60000 [============>.................] - ETA: 1:00 - loss: 0.2675 - categorical_accuracy: 0.9174
27936/60000 [============>.................] - ETA: 1:00 - loss: 0.2673 - categorical_accuracy: 0.9174
27968/60000 [============>.................] - ETA: 1:00 - loss: 0.2671 - categorical_accuracy: 0.9174
28000/60000 [=============>................] - ETA: 1:00 - loss: 0.2670 - categorical_accuracy: 0.9175
28032/60000 [=============>................] - ETA: 1:00 - loss: 0.2667 - categorical_accuracy: 0.9176
28064/60000 [=============>................] - ETA: 1:00 - loss: 0.2666 - categorical_accuracy: 0.9176
28096/60000 [=============>................] - ETA: 1:00 - loss: 0.2664 - categorical_accuracy: 0.9177
28128/60000 [=============>................] - ETA: 1:00 - loss: 0.2662 - categorical_accuracy: 0.9178
28160/60000 [=============>................] - ETA: 1:00 - loss: 0.2660 - categorical_accuracy: 0.9178
28192/60000 [=============>................] - ETA: 1:00 - loss: 0.2658 - categorical_accuracy: 0.9179
28224/60000 [=============>................] - ETA: 1:00 - loss: 0.2656 - categorical_accuracy: 0.9179
28256/60000 [=============>................] - ETA: 1:00 - loss: 0.2656 - categorical_accuracy: 0.9180
28288/60000 [=============>................] - ETA: 1:00 - loss: 0.2655 - categorical_accuracy: 0.9180
28320/60000 [=============>................] - ETA: 1:00 - loss: 0.2653 - categorical_accuracy: 0.9180
28352/60000 [=============>................] - ETA: 1:00 - loss: 0.2654 - categorical_accuracy: 0.9180
28384/60000 [=============>................] - ETA: 1:00 - loss: 0.2653 - categorical_accuracy: 0.9180
28416/60000 [=============>................] - ETA: 59s - loss: 0.2650 - categorical_accuracy: 0.9181 
28448/60000 [=============>................] - ETA: 59s - loss: 0.2648 - categorical_accuracy: 0.9182
28480/60000 [=============>................] - ETA: 59s - loss: 0.2645 - categorical_accuracy: 0.9183
28512/60000 [=============>................] - ETA: 59s - loss: 0.2643 - categorical_accuracy: 0.9184
28544/60000 [=============>................] - ETA: 59s - loss: 0.2641 - categorical_accuracy: 0.9184
28576/60000 [=============>................] - ETA: 59s - loss: 0.2638 - categorical_accuracy: 0.9185
28608/60000 [=============>................] - ETA: 59s - loss: 0.2637 - categorical_accuracy: 0.9186
28640/60000 [=============>................] - ETA: 59s - loss: 0.2635 - categorical_accuracy: 0.9186
28672/60000 [=============>................] - ETA: 59s - loss: 0.2634 - categorical_accuracy: 0.9187
28704/60000 [=============>................] - ETA: 59s - loss: 0.2632 - categorical_accuracy: 0.9188
28736/60000 [=============>................] - ETA: 59s - loss: 0.2632 - categorical_accuracy: 0.9187
28768/60000 [=============>................] - ETA: 59s - loss: 0.2633 - categorical_accuracy: 0.9188
28800/60000 [=============>................] - ETA: 59s - loss: 0.2635 - categorical_accuracy: 0.9187
28832/60000 [=============>................] - ETA: 59s - loss: 0.2634 - categorical_accuracy: 0.9187
28864/60000 [=============>................] - ETA: 59s - loss: 0.2633 - categorical_accuracy: 0.9187
28896/60000 [=============>................] - ETA: 59s - loss: 0.2630 - categorical_accuracy: 0.9188
28928/60000 [=============>................] - ETA: 59s - loss: 0.2628 - categorical_accuracy: 0.9189
28960/60000 [=============>................] - ETA: 58s - loss: 0.2626 - categorical_accuracy: 0.9189
28992/60000 [=============>................] - ETA: 58s - loss: 0.2624 - categorical_accuracy: 0.9190
29024/60000 [=============>................] - ETA: 58s - loss: 0.2621 - categorical_accuracy: 0.9191
29056/60000 [=============>................] - ETA: 58s - loss: 0.2619 - categorical_accuracy: 0.9192
29088/60000 [=============>................] - ETA: 58s - loss: 0.2616 - categorical_accuracy: 0.9192
29120/60000 [=============>................] - ETA: 58s - loss: 0.2615 - categorical_accuracy: 0.9192
29152/60000 [=============>................] - ETA: 58s - loss: 0.2614 - categorical_accuracy: 0.9193
29184/60000 [=============>................] - ETA: 58s - loss: 0.2613 - categorical_accuracy: 0.9192
29216/60000 [=============>................] - ETA: 58s - loss: 0.2612 - categorical_accuracy: 0.9193
29248/60000 [=============>................] - ETA: 58s - loss: 0.2612 - categorical_accuracy: 0.9193
29280/60000 [=============>................] - ETA: 58s - loss: 0.2612 - categorical_accuracy: 0.9193
29312/60000 [=============>................] - ETA: 58s - loss: 0.2611 - categorical_accuracy: 0.9193
29344/60000 [=============>................] - ETA: 58s - loss: 0.2609 - categorical_accuracy: 0.9194
29376/60000 [=============>................] - ETA: 58s - loss: 0.2606 - categorical_accuracy: 0.9195
29408/60000 [=============>................] - ETA: 58s - loss: 0.2604 - categorical_accuracy: 0.9195
29440/60000 [=============>................] - ETA: 58s - loss: 0.2602 - categorical_accuracy: 0.9196
29472/60000 [=============>................] - ETA: 57s - loss: 0.2602 - categorical_accuracy: 0.9196
29504/60000 [=============>................] - ETA: 57s - loss: 0.2604 - categorical_accuracy: 0.9196
29536/60000 [=============>................] - ETA: 57s - loss: 0.2602 - categorical_accuracy: 0.9197
29568/60000 [=============>................] - ETA: 57s - loss: 0.2600 - categorical_accuracy: 0.9197
29600/60000 [=============>................] - ETA: 57s - loss: 0.2598 - categorical_accuracy: 0.9198
29632/60000 [=============>................] - ETA: 57s - loss: 0.2595 - categorical_accuracy: 0.9199
29664/60000 [=============>................] - ETA: 57s - loss: 0.2595 - categorical_accuracy: 0.9199
29696/60000 [=============>................] - ETA: 57s - loss: 0.2593 - categorical_accuracy: 0.9200
29728/60000 [=============>................] - ETA: 57s - loss: 0.2591 - categorical_accuracy: 0.9200
29760/60000 [=============>................] - ETA: 57s - loss: 0.2588 - categorical_accuracy: 0.9201
29792/60000 [=============>................] - ETA: 57s - loss: 0.2592 - categorical_accuracy: 0.9200
29824/60000 [=============>................] - ETA: 57s - loss: 0.2589 - categorical_accuracy: 0.9201
29856/60000 [=============>................] - ETA: 57s - loss: 0.2591 - categorical_accuracy: 0.9202
29888/60000 [=============>................] - ETA: 57s - loss: 0.2591 - categorical_accuracy: 0.9202
29920/60000 [=============>................] - ETA: 57s - loss: 0.2588 - categorical_accuracy: 0.9203
29952/60000 [=============>................] - ETA: 57s - loss: 0.2587 - categorical_accuracy: 0.9203
29984/60000 [=============>................] - ETA: 56s - loss: 0.2584 - categorical_accuracy: 0.9204
30016/60000 [==============>...............] - ETA: 56s - loss: 0.2585 - categorical_accuracy: 0.9204
30048/60000 [==============>...............] - ETA: 56s - loss: 0.2583 - categorical_accuracy: 0.9205
30080/60000 [==============>...............] - ETA: 56s - loss: 0.2582 - categorical_accuracy: 0.9205
30112/60000 [==============>...............] - ETA: 56s - loss: 0.2580 - categorical_accuracy: 0.9206
30144/60000 [==============>...............] - ETA: 56s - loss: 0.2577 - categorical_accuracy: 0.9207
30176/60000 [==============>...............] - ETA: 56s - loss: 0.2576 - categorical_accuracy: 0.9207
30208/60000 [==============>...............] - ETA: 56s - loss: 0.2576 - categorical_accuracy: 0.9207
30240/60000 [==============>...............] - ETA: 56s - loss: 0.2574 - categorical_accuracy: 0.9208
30272/60000 [==============>...............] - ETA: 56s - loss: 0.2573 - categorical_accuracy: 0.9208
30304/60000 [==============>...............] - ETA: 56s - loss: 0.2571 - categorical_accuracy: 0.9209
30336/60000 [==============>...............] - ETA: 56s - loss: 0.2571 - categorical_accuracy: 0.9208
30368/60000 [==============>...............] - ETA: 56s - loss: 0.2569 - categorical_accuracy: 0.9209
30400/60000 [==============>...............] - ETA: 56s - loss: 0.2567 - categorical_accuracy: 0.9209
30432/60000 [==============>...............] - ETA: 56s - loss: 0.2565 - categorical_accuracy: 0.9210
30464/60000 [==============>...............] - ETA: 56s - loss: 0.2563 - categorical_accuracy: 0.9210
30496/60000 [==============>...............] - ETA: 55s - loss: 0.2566 - categorical_accuracy: 0.9210
30528/60000 [==============>...............] - ETA: 55s - loss: 0.2563 - categorical_accuracy: 0.9211
30560/60000 [==============>...............] - ETA: 55s - loss: 0.2562 - categorical_accuracy: 0.9211
30592/60000 [==============>...............] - ETA: 55s - loss: 0.2559 - categorical_accuracy: 0.9212
30624/60000 [==============>...............] - ETA: 55s - loss: 0.2557 - categorical_accuracy: 0.9213
30656/60000 [==============>...............] - ETA: 55s - loss: 0.2554 - categorical_accuracy: 0.9214
30688/60000 [==============>...............] - ETA: 55s - loss: 0.2553 - categorical_accuracy: 0.9214
30720/60000 [==============>...............] - ETA: 55s - loss: 0.2551 - categorical_accuracy: 0.9215
30752/60000 [==============>...............] - ETA: 55s - loss: 0.2549 - categorical_accuracy: 0.9216
30784/60000 [==============>...............] - ETA: 55s - loss: 0.2547 - categorical_accuracy: 0.9216
30816/60000 [==============>...............] - ETA: 55s - loss: 0.2545 - categorical_accuracy: 0.9217
30848/60000 [==============>...............] - ETA: 55s - loss: 0.2543 - categorical_accuracy: 0.9218
30880/60000 [==============>...............] - ETA: 55s - loss: 0.2542 - categorical_accuracy: 0.9218
30912/60000 [==============>...............] - ETA: 55s - loss: 0.2540 - categorical_accuracy: 0.9218
30944/60000 [==============>...............] - ETA: 55s - loss: 0.2539 - categorical_accuracy: 0.9219
30976/60000 [==============>...............] - ETA: 55s - loss: 0.2538 - categorical_accuracy: 0.9219
31008/60000 [==============>...............] - ETA: 54s - loss: 0.2536 - categorical_accuracy: 0.9220
31040/60000 [==============>...............] - ETA: 54s - loss: 0.2537 - categorical_accuracy: 0.9220
31072/60000 [==============>...............] - ETA: 54s - loss: 0.2535 - categorical_accuracy: 0.9221
31104/60000 [==============>...............] - ETA: 54s - loss: 0.2533 - categorical_accuracy: 0.9221
31136/60000 [==============>...............] - ETA: 54s - loss: 0.2532 - categorical_accuracy: 0.9222
31168/60000 [==============>...............] - ETA: 54s - loss: 0.2529 - categorical_accuracy: 0.9223
31200/60000 [==============>...............] - ETA: 54s - loss: 0.2528 - categorical_accuracy: 0.9223
31232/60000 [==============>...............] - ETA: 54s - loss: 0.2527 - categorical_accuracy: 0.9223
31264/60000 [==============>...............] - ETA: 54s - loss: 0.2525 - categorical_accuracy: 0.9224
31296/60000 [==============>...............] - ETA: 54s - loss: 0.2528 - categorical_accuracy: 0.9223
31328/60000 [==============>...............] - ETA: 54s - loss: 0.2525 - categorical_accuracy: 0.9224
31360/60000 [==============>...............] - ETA: 54s - loss: 0.2525 - categorical_accuracy: 0.9224
31392/60000 [==============>...............] - ETA: 54s - loss: 0.2522 - categorical_accuracy: 0.9225
31424/60000 [==============>...............] - ETA: 54s - loss: 0.2520 - categorical_accuracy: 0.9226
31456/60000 [==============>...............] - ETA: 54s - loss: 0.2518 - categorical_accuracy: 0.9227
31488/60000 [==============>...............] - ETA: 54s - loss: 0.2516 - categorical_accuracy: 0.9227
31520/60000 [==============>...............] - ETA: 54s - loss: 0.2514 - categorical_accuracy: 0.9228
31552/60000 [==============>...............] - ETA: 53s - loss: 0.2512 - categorical_accuracy: 0.9228
31584/60000 [==============>...............] - ETA: 53s - loss: 0.2511 - categorical_accuracy: 0.9228
31616/60000 [==============>...............] - ETA: 53s - loss: 0.2509 - categorical_accuracy: 0.9229
31648/60000 [==============>...............] - ETA: 53s - loss: 0.2508 - categorical_accuracy: 0.9229
31680/60000 [==============>...............] - ETA: 53s - loss: 0.2506 - categorical_accuracy: 0.9230
31712/60000 [==============>...............] - ETA: 53s - loss: 0.2508 - categorical_accuracy: 0.9230
31744/60000 [==============>...............] - ETA: 53s - loss: 0.2507 - categorical_accuracy: 0.9230
31776/60000 [==============>...............] - ETA: 53s - loss: 0.2508 - categorical_accuracy: 0.9230
31808/60000 [==============>...............] - ETA: 53s - loss: 0.2506 - categorical_accuracy: 0.9231
31840/60000 [==============>...............] - ETA: 53s - loss: 0.2506 - categorical_accuracy: 0.9231
31872/60000 [==============>...............] - ETA: 53s - loss: 0.2506 - categorical_accuracy: 0.9231
31904/60000 [==============>...............] - ETA: 53s - loss: 0.2504 - categorical_accuracy: 0.9231
31936/60000 [==============>...............] - ETA: 53s - loss: 0.2504 - categorical_accuracy: 0.9232
31968/60000 [==============>...............] - ETA: 53s - loss: 0.2503 - categorical_accuracy: 0.9231
32000/60000 [===============>..............] - ETA: 53s - loss: 0.2501 - categorical_accuracy: 0.9232
32032/60000 [===============>..............] - ETA: 53s - loss: 0.2499 - categorical_accuracy: 0.9233
32064/60000 [===============>..............] - ETA: 53s - loss: 0.2497 - categorical_accuracy: 0.9233
32096/60000 [===============>..............] - ETA: 52s - loss: 0.2496 - categorical_accuracy: 0.9234
32128/60000 [===============>..............] - ETA: 52s - loss: 0.2494 - categorical_accuracy: 0.9234
32160/60000 [===============>..............] - ETA: 52s - loss: 0.2492 - categorical_accuracy: 0.9235
32192/60000 [===============>..............] - ETA: 52s - loss: 0.2490 - categorical_accuracy: 0.9236
32224/60000 [===============>..............] - ETA: 52s - loss: 0.2489 - categorical_accuracy: 0.9236
32256/60000 [===============>..............] - ETA: 52s - loss: 0.2488 - categorical_accuracy: 0.9236
32288/60000 [===============>..............] - ETA: 52s - loss: 0.2486 - categorical_accuracy: 0.9237
32320/60000 [===============>..............] - ETA: 52s - loss: 0.2484 - categorical_accuracy: 0.9237
32352/60000 [===============>..............] - ETA: 52s - loss: 0.2482 - categorical_accuracy: 0.9238
32384/60000 [===============>..............] - ETA: 52s - loss: 0.2481 - categorical_accuracy: 0.9238
32416/60000 [===============>..............] - ETA: 52s - loss: 0.2479 - categorical_accuracy: 0.9239
32448/60000 [===============>..............] - ETA: 52s - loss: 0.2477 - categorical_accuracy: 0.9239
32480/60000 [===============>..............] - ETA: 52s - loss: 0.2477 - categorical_accuracy: 0.9240
32512/60000 [===============>..............] - ETA: 52s - loss: 0.2477 - categorical_accuracy: 0.9240
32544/60000 [===============>..............] - ETA: 52s - loss: 0.2476 - categorical_accuracy: 0.9240
32576/60000 [===============>..............] - ETA: 52s - loss: 0.2474 - categorical_accuracy: 0.9241
32608/60000 [===============>..............] - ETA: 51s - loss: 0.2475 - categorical_accuracy: 0.9240
32640/60000 [===============>..............] - ETA: 51s - loss: 0.2474 - categorical_accuracy: 0.9241
32672/60000 [===============>..............] - ETA: 51s - loss: 0.2475 - categorical_accuracy: 0.9241
32704/60000 [===============>..............] - ETA: 51s - loss: 0.2473 - categorical_accuracy: 0.9241
32736/60000 [===============>..............] - ETA: 51s - loss: 0.2472 - categorical_accuracy: 0.9242
32768/60000 [===============>..............] - ETA: 51s - loss: 0.2473 - categorical_accuracy: 0.9241
32800/60000 [===============>..............] - ETA: 51s - loss: 0.2471 - categorical_accuracy: 0.9242
32832/60000 [===============>..............] - ETA: 51s - loss: 0.2470 - categorical_accuracy: 0.9242
32864/60000 [===============>..............] - ETA: 51s - loss: 0.2469 - categorical_accuracy: 0.9242
32896/60000 [===============>..............] - ETA: 51s - loss: 0.2468 - categorical_accuracy: 0.9242
32928/60000 [===============>..............] - ETA: 51s - loss: 0.2469 - categorical_accuracy: 0.9243
32960/60000 [===============>..............] - ETA: 51s - loss: 0.2467 - categorical_accuracy: 0.9243
32992/60000 [===============>..............] - ETA: 51s - loss: 0.2467 - categorical_accuracy: 0.9243
33024/60000 [===============>..............] - ETA: 51s - loss: 0.2465 - categorical_accuracy: 0.9244
33056/60000 [===============>..............] - ETA: 51s - loss: 0.2467 - categorical_accuracy: 0.9243
33088/60000 [===============>..............] - ETA: 51s - loss: 0.2466 - categorical_accuracy: 0.9244
33120/60000 [===============>..............] - ETA: 50s - loss: 0.2465 - categorical_accuracy: 0.9244
33152/60000 [===============>..............] - ETA: 50s - loss: 0.2464 - categorical_accuracy: 0.9244
33184/60000 [===============>..............] - ETA: 50s - loss: 0.2463 - categorical_accuracy: 0.9245
33216/60000 [===============>..............] - ETA: 50s - loss: 0.2461 - categorical_accuracy: 0.9245
33248/60000 [===============>..............] - ETA: 50s - loss: 0.2459 - categorical_accuracy: 0.9246
33280/60000 [===============>..............] - ETA: 50s - loss: 0.2457 - categorical_accuracy: 0.9246
33312/60000 [===============>..............] - ETA: 50s - loss: 0.2455 - categorical_accuracy: 0.9247
33344/60000 [===============>..............] - ETA: 50s - loss: 0.2455 - categorical_accuracy: 0.9247
33376/60000 [===============>..............] - ETA: 50s - loss: 0.2453 - categorical_accuracy: 0.9248
33408/60000 [===============>..............] - ETA: 50s - loss: 0.2452 - categorical_accuracy: 0.9248
33440/60000 [===============>..............] - ETA: 50s - loss: 0.2451 - categorical_accuracy: 0.9249
33472/60000 [===============>..............] - ETA: 50s - loss: 0.2452 - categorical_accuracy: 0.9249
33504/60000 [===============>..............] - ETA: 50s - loss: 0.2450 - categorical_accuracy: 0.9250
33536/60000 [===============>..............] - ETA: 50s - loss: 0.2449 - categorical_accuracy: 0.9250
33568/60000 [===============>..............] - ETA: 50s - loss: 0.2447 - categorical_accuracy: 0.9250
33600/60000 [===============>..............] - ETA: 50s - loss: 0.2445 - categorical_accuracy: 0.9251
33632/60000 [===============>..............] - ETA: 49s - loss: 0.2444 - categorical_accuracy: 0.9251
33664/60000 [===============>..............] - ETA: 49s - loss: 0.2443 - categorical_accuracy: 0.9252
33696/60000 [===============>..............] - ETA: 49s - loss: 0.2442 - categorical_accuracy: 0.9252
33728/60000 [===============>..............] - ETA: 49s - loss: 0.2440 - categorical_accuracy: 0.9253
33760/60000 [===============>..............] - ETA: 49s - loss: 0.2437 - categorical_accuracy: 0.9254
33792/60000 [===============>..............] - ETA: 49s - loss: 0.2437 - categorical_accuracy: 0.9254
33824/60000 [===============>..............] - ETA: 49s - loss: 0.2435 - categorical_accuracy: 0.9254
33856/60000 [===============>..............] - ETA: 49s - loss: 0.2435 - categorical_accuracy: 0.9255
33888/60000 [===============>..............] - ETA: 49s - loss: 0.2433 - categorical_accuracy: 0.9255
33920/60000 [===============>..............] - ETA: 49s - loss: 0.2431 - categorical_accuracy: 0.9256
33952/60000 [===============>..............] - ETA: 49s - loss: 0.2432 - categorical_accuracy: 0.9256
33984/60000 [===============>..............] - ETA: 49s - loss: 0.2431 - categorical_accuracy: 0.9257
34016/60000 [================>.............] - ETA: 49s - loss: 0.2429 - categorical_accuracy: 0.9257
34048/60000 [================>.............] - ETA: 49s - loss: 0.2428 - categorical_accuracy: 0.9257
34080/60000 [================>.............] - ETA: 49s - loss: 0.2426 - categorical_accuracy: 0.9258
34112/60000 [================>.............] - ETA: 49s - loss: 0.2427 - categorical_accuracy: 0.9258
34144/60000 [================>.............] - ETA: 48s - loss: 0.2426 - categorical_accuracy: 0.9259
34176/60000 [================>.............] - ETA: 48s - loss: 0.2424 - categorical_accuracy: 0.9259
34208/60000 [================>.............] - ETA: 48s - loss: 0.2426 - categorical_accuracy: 0.9259
34240/60000 [================>.............] - ETA: 48s - loss: 0.2424 - categorical_accuracy: 0.9260
34272/60000 [================>.............] - ETA: 48s - loss: 0.2423 - categorical_accuracy: 0.9260
34304/60000 [================>.............] - ETA: 48s - loss: 0.2421 - categorical_accuracy: 0.9260
34336/60000 [================>.............] - ETA: 48s - loss: 0.2420 - categorical_accuracy: 0.9261
34368/60000 [================>.............] - ETA: 48s - loss: 0.2418 - categorical_accuracy: 0.9261
34400/60000 [================>.............] - ETA: 48s - loss: 0.2416 - categorical_accuracy: 0.9262
34432/60000 [================>.............] - ETA: 48s - loss: 0.2415 - categorical_accuracy: 0.9262
34464/60000 [================>.............] - ETA: 48s - loss: 0.2417 - categorical_accuracy: 0.9262
34496/60000 [================>.............] - ETA: 48s - loss: 0.2417 - categorical_accuracy: 0.9262
34528/60000 [================>.............] - ETA: 48s - loss: 0.2415 - categorical_accuracy: 0.9263
34560/60000 [================>.............] - ETA: 48s - loss: 0.2413 - categorical_accuracy: 0.9263
34592/60000 [================>.............] - ETA: 48s - loss: 0.2412 - categorical_accuracy: 0.9264
34624/60000 [================>.............] - ETA: 48s - loss: 0.2414 - categorical_accuracy: 0.9264
34656/60000 [================>.............] - ETA: 48s - loss: 0.2412 - categorical_accuracy: 0.9264
34688/60000 [================>.............] - ETA: 47s - loss: 0.2410 - categorical_accuracy: 0.9265
34720/60000 [================>.............] - ETA: 47s - loss: 0.2408 - categorical_accuracy: 0.9266
34752/60000 [================>.............] - ETA: 47s - loss: 0.2407 - categorical_accuracy: 0.9266
34784/60000 [================>.............] - ETA: 47s - loss: 0.2405 - categorical_accuracy: 0.9267
34816/60000 [================>.............] - ETA: 47s - loss: 0.2404 - categorical_accuracy: 0.9267
34848/60000 [================>.............] - ETA: 47s - loss: 0.2402 - categorical_accuracy: 0.9267
34880/60000 [================>.............] - ETA: 47s - loss: 0.2400 - categorical_accuracy: 0.9268
34912/60000 [================>.............] - ETA: 47s - loss: 0.2399 - categorical_accuracy: 0.9268
34944/60000 [================>.............] - ETA: 47s - loss: 0.2398 - categorical_accuracy: 0.9269
34976/60000 [================>.............] - ETA: 47s - loss: 0.2396 - categorical_accuracy: 0.9269
35008/60000 [================>.............] - ETA: 47s - loss: 0.2395 - categorical_accuracy: 0.9269
35040/60000 [================>.............] - ETA: 47s - loss: 0.2393 - categorical_accuracy: 0.9270
35072/60000 [================>.............] - ETA: 47s - loss: 0.2393 - categorical_accuracy: 0.9270
35104/60000 [================>.............] - ETA: 47s - loss: 0.2391 - categorical_accuracy: 0.9270
35136/60000 [================>.............] - ETA: 47s - loss: 0.2390 - categorical_accuracy: 0.9271
35168/60000 [================>.............] - ETA: 47s - loss: 0.2388 - categorical_accuracy: 0.9271
35200/60000 [================>.............] - ETA: 46s - loss: 0.2387 - categorical_accuracy: 0.9271
35232/60000 [================>.............] - ETA: 46s - loss: 0.2387 - categorical_accuracy: 0.9271
35264/60000 [================>.............] - ETA: 46s - loss: 0.2386 - categorical_accuracy: 0.9271
35296/60000 [================>.............] - ETA: 46s - loss: 0.2384 - categorical_accuracy: 0.9272
35328/60000 [================>.............] - ETA: 46s - loss: 0.2383 - categorical_accuracy: 0.9272
35360/60000 [================>.............] - ETA: 46s - loss: 0.2381 - categorical_accuracy: 0.9273
35392/60000 [================>.............] - ETA: 46s - loss: 0.2380 - categorical_accuracy: 0.9273
35424/60000 [================>.............] - ETA: 46s - loss: 0.2378 - categorical_accuracy: 0.9273
35456/60000 [================>.............] - ETA: 46s - loss: 0.2379 - categorical_accuracy: 0.9273
35488/60000 [================>.............] - ETA: 46s - loss: 0.2379 - categorical_accuracy: 0.9274
35520/60000 [================>.............] - ETA: 46s - loss: 0.2378 - categorical_accuracy: 0.9274
35552/60000 [================>.............] - ETA: 46s - loss: 0.2376 - categorical_accuracy: 0.9275
35584/60000 [================>.............] - ETA: 46s - loss: 0.2374 - categorical_accuracy: 0.9276
35616/60000 [================>.............] - ETA: 46s - loss: 0.2375 - categorical_accuracy: 0.9276
35648/60000 [================>.............] - ETA: 46s - loss: 0.2374 - categorical_accuracy: 0.9276
35680/60000 [================>.............] - ETA: 46s - loss: 0.2372 - categorical_accuracy: 0.9276
35712/60000 [================>.............] - ETA: 45s - loss: 0.2371 - categorical_accuracy: 0.9277
35744/60000 [================>.............] - ETA: 45s - loss: 0.2369 - categorical_accuracy: 0.9278
35776/60000 [================>.............] - ETA: 45s - loss: 0.2369 - categorical_accuracy: 0.9278
35808/60000 [================>.............] - ETA: 45s - loss: 0.2369 - categorical_accuracy: 0.9278
35840/60000 [================>.............] - ETA: 45s - loss: 0.2368 - categorical_accuracy: 0.9278
35872/60000 [================>.............] - ETA: 45s - loss: 0.2367 - categorical_accuracy: 0.9278
35904/60000 [================>.............] - ETA: 45s - loss: 0.2366 - categorical_accuracy: 0.9279
35936/60000 [================>.............] - ETA: 45s - loss: 0.2365 - categorical_accuracy: 0.9279
35968/60000 [================>.............] - ETA: 45s - loss: 0.2364 - categorical_accuracy: 0.9279
36000/60000 [=================>............] - ETA: 45s - loss: 0.2363 - categorical_accuracy: 0.9279
36032/60000 [=================>............] - ETA: 45s - loss: 0.2362 - categorical_accuracy: 0.9280
36064/60000 [=================>............] - ETA: 45s - loss: 0.2360 - categorical_accuracy: 0.9280
36096/60000 [=================>............] - ETA: 45s - loss: 0.2359 - categorical_accuracy: 0.9281
36128/60000 [=================>............] - ETA: 45s - loss: 0.2358 - categorical_accuracy: 0.9281
36160/60000 [=================>............] - ETA: 45s - loss: 0.2356 - categorical_accuracy: 0.9282
36192/60000 [=================>............] - ETA: 45s - loss: 0.2355 - categorical_accuracy: 0.9282
36224/60000 [=================>............] - ETA: 44s - loss: 0.2354 - categorical_accuracy: 0.9282
36256/60000 [=================>............] - ETA: 44s - loss: 0.2353 - categorical_accuracy: 0.9282
36288/60000 [=================>............] - ETA: 44s - loss: 0.2351 - categorical_accuracy: 0.9282
36320/60000 [=================>............] - ETA: 44s - loss: 0.2350 - categorical_accuracy: 0.9282
36352/60000 [=================>............] - ETA: 44s - loss: 0.2350 - categorical_accuracy: 0.9282
36384/60000 [=================>............] - ETA: 44s - loss: 0.2348 - categorical_accuracy: 0.9283
36416/60000 [=================>............] - ETA: 44s - loss: 0.2349 - categorical_accuracy: 0.9283
36448/60000 [=================>............] - ETA: 44s - loss: 0.2350 - categorical_accuracy: 0.9283
36480/60000 [=================>............] - ETA: 44s - loss: 0.2351 - categorical_accuracy: 0.9283
36512/60000 [=================>............] - ETA: 44s - loss: 0.2349 - categorical_accuracy: 0.9284
36544/60000 [=================>............] - ETA: 44s - loss: 0.2349 - categorical_accuracy: 0.9284
36576/60000 [=================>............] - ETA: 44s - loss: 0.2348 - categorical_accuracy: 0.9285
36608/60000 [=================>............] - ETA: 44s - loss: 0.2346 - categorical_accuracy: 0.9285
36640/60000 [=================>............] - ETA: 44s - loss: 0.2344 - categorical_accuracy: 0.9286
36672/60000 [=================>............] - ETA: 44s - loss: 0.2344 - categorical_accuracy: 0.9286
36704/60000 [=================>............] - ETA: 44s - loss: 0.2343 - categorical_accuracy: 0.9286
36736/60000 [=================>............] - ETA: 44s - loss: 0.2341 - categorical_accuracy: 0.9287
36768/60000 [=================>............] - ETA: 43s - loss: 0.2340 - categorical_accuracy: 0.9287
36800/60000 [=================>............] - ETA: 43s - loss: 0.2339 - categorical_accuracy: 0.9287
36832/60000 [=================>............] - ETA: 43s - loss: 0.2339 - categorical_accuracy: 0.9287
36864/60000 [=================>............] - ETA: 43s - loss: 0.2337 - categorical_accuracy: 0.9288
36896/60000 [=================>............] - ETA: 43s - loss: 0.2335 - categorical_accuracy: 0.9289
36928/60000 [=================>............] - ETA: 43s - loss: 0.2333 - categorical_accuracy: 0.9289
36960/60000 [=================>............] - ETA: 43s - loss: 0.2331 - categorical_accuracy: 0.9290
36992/60000 [=================>............] - ETA: 43s - loss: 0.2330 - categorical_accuracy: 0.9290
37024/60000 [=================>............] - ETA: 43s - loss: 0.2329 - categorical_accuracy: 0.9291
37056/60000 [=================>............] - ETA: 43s - loss: 0.2328 - categorical_accuracy: 0.9291
37088/60000 [=================>............] - ETA: 43s - loss: 0.2328 - categorical_accuracy: 0.9291
37120/60000 [=================>............] - ETA: 43s - loss: 0.2327 - categorical_accuracy: 0.9292
37152/60000 [=================>............] - ETA: 43s - loss: 0.2327 - categorical_accuracy: 0.9292
37184/60000 [=================>............] - ETA: 43s - loss: 0.2326 - categorical_accuracy: 0.9292
37216/60000 [=================>............] - ETA: 43s - loss: 0.2324 - categorical_accuracy: 0.9293
37248/60000 [=================>............] - ETA: 43s - loss: 0.2323 - categorical_accuracy: 0.9293
37280/60000 [=================>............] - ETA: 42s - loss: 0.2321 - categorical_accuracy: 0.9294
37312/60000 [=================>............] - ETA: 42s - loss: 0.2319 - categorical_accuracy: 0.9294
37344/60000 [=================>............] - ETA: 42s - loss: 0.2319 - categorical_accuracy: 0.9295
37376/60000 [=================>............] - ETA: 42s - loss: 0.2317 - categorical_accuracy: 0.9295
37408/60000 [=================>............] - ETA: 42s - loss: 0.2316 - categorical_accuracy: 0.9296
37440/60000 [=================>............] - ETA: 42s - loss: 0.2314 - categorical_accuracy: 0.9296
37472/60000 [=================>............] - ETA: 42s - loss: 0.2313 - categorical_accuracy: 0.9297
37504/60000 [=================>............] - ETA: 42s - loss: 0.2314 - categorical_accuracy: 0.9296
37536/60000 [=================>............] - ETA: 42s - loss: 0.2313 - categorical_accuracy: 0.9297
37568/60000 [=================>............] - ETA: 42s - loss: 0.2312 - categorical_accuracy: 0.9297
37600/60000 [=================>............] - ETA: 42s - loss: 0.2311 - categorical_accuracy: 0.9298
37632/60000 [=================>............] - ETA: 42s - loss: 0.2310 - categorical_accuracy: 0.9298
37664/60000 [=================>............] - ETA: 42s - loss: 0.2308 - categorical_accuracy: 0.9299
37696/60000 [=================>............] - ETA: 42s - loss: 0.2307 - categorical_accuracy: 0.9299
37728/60000 [=================>............] - ETA: 42s - loss: 0.2306 - categorical_accuracy: 0.9299
37760/60000 [=================>............] - ETA: 42s - loss: 0.2307 - categorical_accuracy: 0.9299
37792/60000 [=================>............] - ETA: 42s - loss: 0.2305 - categorical_accuracy: 0.9300
37824/60000 [=================>............] - ETA: 41s - loss: 0.2305 - categorical_accuracy: 0.9300
37856/60000 [=================>............] - ETA: 41s - loss: 0.2304 - categorical_accuracy: 0.9301
37888/60000 [=================>............] - ETA: 41s - loss: 0.2302 - categorical_accuracy: 0.9301
37920/60000 [=================>............] - ETA: 41s - loss: 0.2302 - categorical_accuracy: 0.9301
37952/60000 [=================>............] - ETA: 41s - loss: 0.2301 - categorical_accuracy: 0.9301
37984/60000 [=================>............] - ETA: 41s - loss: 0.2300 - categorical_accuracy: 0.9301
38016/60000 [==================>...........] - ETA: 41s - loss: 0.2299 - categorical_accuracy: 0.9302
38048/60000 [==================>...........] - ETA: 41s - loss: 0.2298 - categorical_accuracy: 0.9302
38080/60000 [==================>...........] - ETA: 41s - loss: 0.2296 - categorical_accuracy: 0.9303
38112/60000 [==================>...........] - ETA: 41s - loss: 0.2298 - categorical_accuracy: 0.9302
38144/60000 [==================>...........] - ETA: 41s - loss: 0.2297 - categorical_accuracy: 0.9302
38176/60000 [==================>...........] - ETA: 41s - loss: 0.2296 - categorical_accuracy: 0.9303
38208/60000 [==================>...........] - ETA: 41s - loss: 0.2295 - categorical_accuracy: 0.9303
38240/60000 [==================>...........] - ETA: 41s - loss: 0.2294 - categorical_accuracy: 0.9304
38272/60000 [==================>...........] - ETA: 41s - loss: 0.2292 - categorical_accuracy: 0.9304
38304/60000 [==================>...........] - ETA: 41s - loss: 0.2291 - categorical_accuracy: 0.9305
38336/60000 [==================>...........] - ETA: 40s - loss: 0.2291 - categorical_accuracy: 0.9304
38368/60000 [==================>...........] - ETA: 40s - loss: 0.2289 - categorical_accuracy: 0.9305
38400/60000 [==================>...........] - ETA: 40s - loss: 0.2289 - categorical_accuracy: 0.9305
38432/60000 [==================>...........] - ETA: 40s - loss: 0.2288 - categorical_accuracy: 0.9305
38464/60000 [==================>...........] - ETA: 40s - loss: 0.2288 - categorical_accuracy: 0.9305
38496/60000 [==================>...........] - ETA: 40s - loss: 0.2286 - categorical_accuracy: 0.9305
38528/60000 [==================>...........] - ETA: 40s - loss: 0.2285 - categorical_accuracy: 0.9306
38560/60000 [==================>...........] - ETA: 40s - loss: 0.2283 - categorical_accuracy: 0.9307
38592/60000 [==================>...........] - ETA: 40s - loss: 0.2282 - categorical_accuracy: 0.9307
38624/60000 [==================>...........] - ETA: 40s - loss: 0.2281 - categorical_accuracy: 0.9307
38656/60000 [==================>...........] - ETA: 40s - loss: 0.2280 - categorical_accuracy: 0.9307
38688/60000 [==================>...........] - ETA: 40s - loss: 0.2279 - categorical_accuracy: 0.9308
38720/60000 [==================>...........] - ETA: 40s - loss: 0.2277 - categorical_accuracy: 0.9308
38752/60000 [==================>...........] - ETA: 40s - loss: 0.2276 - categorical_accuracy: 0.9308
38784/60000 [==================>...........] - ETA: 40s - loss: 0.2275 - categorical_accuracy: 0.9309
38816/60000 [==================>...........] - ETA: 40s - loss: 0.2274 - categorical_accuracy: 0.9309
38848/60000 [==================>...........] - ETA: 40s - loss: 0.2272 - categorical_accuracy: 0.9310
38880/60000 [==================>...........] - ETA: 39s - loss: 0.2271 - categorical_accuracy: 0.9310
38912/60000 [==================>...........] - ETA: 39s - loss: 0.2270 - categorical_accuracy: 0.9310
38944/60000 [==================>...........] - ETA: 39s - loss: 0.2269 - categorical_accuracy: 0.9311
38976/60000 [==================>...........] - ETA: 39s - loss: 0.2268 - categorical_accuracy: 0.9311
39008/60000 [==================>...........] - ETA: 39s - loss: 0.2268 - categorical_accuracy: 0.9311
39040/60000 [==================>...........] - ETA: 39s - loss: 0.2267 - categorical_accuracy: 0.9310
39072/60000 [==================>...........] - ETA: 39s - loss: 0.2265 - categorical_accuracy: 0.9311
39104/60000 [==================>...........] - ETA: 39s - loss: 0.2264 - categorical_accuracy: 0.9311
39136/60000 [==================>...........] - ETA: 39s - loss: 0.2262 - categorical_accuracy: 0.9312
39168/60000 [==================>...........] - ETA: 39s - loss: 0.2261 - categorical_accuracy: 0.9312
39200/60000 [==================>...........] - ETA: 39s - loss: 0.2259 - categorical_accuracy: 0.9313
39232/60000 [==================>...........] - ETA: 39s - loss: 0.2259 - categorical_accuracy: 0.9313
39264/60000 [==================>...........] - ETA: 39s - loss: 0.2259 - categorical_accuracy: 0.9313
39296/60000 [==================>...........] - ETA: 39s - loss: 0.2262 - categorical_accuracy: 0.9313
39328/60000 [==================>...........] - ETA: 39s - loss: 0.2261 - categorical_accuracy: 0.9313
39360/60000 [==================>...........] - ETA: 39s - loss: 0.2260 - categorical_accuracy: 0.9314
39392/60000 [==================>...........] - ETA: 38s - loss: 0.2258 - categorical_accuracy: 0.9314
39424/60000 [==================>...........] - ETA: 38s - loss: 0.2256 - categorical_accuracy: 0.9315
39456/60000 [==================>...........] - ETA: 38s - loss: 0.2255 - categorical_accuracy: 0.9315
39488/60000 [==================>...........] - ETA: 38s - loss: 0.2256 - categorical_accuracy: 0.9315
39520/60000 [==================>...........] - ETA: 38s - loss: 0.2255 - categorical_accuracy: 0.9315
39552/60000 [==================>...........] - ETA: 38s - loss: 0.2254 - categorical_accuracy: 0.9316
39584/60000 [==================>...........] - ETA: 38s - loss: 0.2253 - categorical_accuracy: 0.9316
39616/60000 [==================>...........] - ETA: 38s - loss: 0.2252 - categorical_accuracy: 0.9316
39648/60000 [==================>...........] - ETA: 38s - loss: 0.2251 - categorical_accuracy: 0.9316
39680/60000 [==================>...........] - ETA: 38s - loss: 0.2250 - categorical_accuracy: 0.9317
39712/60000 [==================>...........] - ETA: 38s - loss: 0.2248 - categorical_accuracy: 0.9318
39744/60000 [==================>...........] - ETA: 38s - loss: 0.2247 - categorical_accuracy: 0.9318
39776/60000 [==================>...........] - ETA: 38s - loss: 0.2246 - categorical_accuracy: 0.9318
39808/60000 [==================>...........] - ETA: 38s - loss: 0.2245 - categorical_accuracy: 0.9319
39840/60000 [==================>...........] - ETA: 38s - loss: 0.2243 - categorical_accuracy: 0.9319
39872/60000 [==================>...........] - ETA: 38s - loss: 0.2242 - categorical_accuracy: 0.9319
39904/60000 [==================>...........] - ETA: 38s - loss: 0.2241 - categorical_accuracy: 0.9320
39936/60000 [==================>...........] - ETA: 37s - loss: 0.2239 - categorical_accuracy: 0.9320
39968/60000 [==================>...........] - ETA: 37s - loss: 0.2238 - categorical_accuracy: 0.9321
40000/60000 [===================>..........] - ETA: 37s - loss: 0.2237 - categorical_accuracy: 0.9321
40032/60000 [===================>..........] - ETA: 37s - loss: 0.2236 - categorical_accuracy: 0.9322
40064/60000 [===================>..........] - ETA: 37s - loss: 0.2235 - categorical_accuracy: 0.9322
40096/60000 [===================>..........] - ETA: 37s - loss: 0.2233 - categorical_accuracy: 0.9323
40128/60000 [===================>..........] - ETA: 37s - loss: 0.2232 - categorical_accuracy: 0.9323
40160/60000 [===================>..........] - ETA: 37s - loss: 0.2232 - categorical_accuracy: 0.9323
40192/60000 [===================>..........] - ETA: 37s - loss: 0.2231 - categorical_accuracy: 0.9324
40224/60000 [===================>..........] - ETA: 37s - loss: 0.2229 - categorical_accuracy: 0.9324
40256/60000 [===================>..........] - ETA: 37s - loss: 0.2228 - categorical_accuracy: 0.9325
40288/60000 [===================>..........] - ETA: 37s - loss: 0.2226 - categorical_accuracy: 0.9325
40320/60000 [===================>..........] - ETA: 37s - loss: 0.2225 - categorical_accuracy: 0.9325
40352/60000 [===================>..........] - ETA: 37s - loss: 0.2224 - categorical_accuracy: 0.9326
40384/60000 [===================>..........] - ETA: 37s - loss: 0.2222 - categorical_accuracy: 0.9326
40416/60000 [===================>..........] - ETA: 37s - loss: 0.2222 - categorical_accuracy: 0.9326
40448/60000 [===================>..........] - ETA: 36s - loss: 0.2221 - categorical_accuracy: 0.9327
40480/60000 [===================>..........] - ETA: 36s - loss: 0.2219 - categorical_accuracy: 0.9327
40512/60000 [===================>..........] - ETA: 36s - loss: 0.2218 - categorical_accuracy: 0.9328
40544/60000 [===================>..........] - ETA: 36s - loss: 0.2219 - categorical_accuracy: 0.9327
40576/60000 [===================>..........] - ETA: 36s - loss: 0.2218 - categorical_accuracy: 0.9327
40608/60000 [===================>..........] - ETA: 36s - loss: 0.2217 - categorical_accuracy: 0.9328
40640/60000 [===================>..........] - ETA: 36s - loss: 0.2215 - categorical_accuracy: 0.9328
40672/60000 [===================>..........] - ETA: 36s - loss: 0.2215 - categorical_accuracy: 0.9328
40704/60000 [===================>..........] - ETA: 36s - loss: 0.2216 - categorical_accuracy: 0.9328
40736/60000 [===================>..........] - ETA: 36s - loss: 0.2214 - categorical_accuracy: 0.9328
40768/60000 [===================>..........] - ETA: 36s - loss: 0.2214 - categorical_accuracy: 0.9329
40800/60000 [===================>..........] - ETA: 36s - loss: 0.2214 - categorical_accuracy: 0.9328
40832/60000 [===================>..........] - ETA: 36s - loss: 0.2215 - categorical_accuracy: 0.9328
40864/60000 [===================>..........] - ETA: 36s - loss: 0.2213 - categorical_accuracy: 0.9329
40896/60000 [===================>..........] - ETA: 36s - loss: 0.2213 - categorical_accuracy: 0.9329
40928/60000 [===================>..........] - ETA: 36s - loss: 0.2213 - categorical_accuracy: 0.9328
40960/60000 [===================>..........] - ETA: 36s - loss: 0.2213 - categorical_accuracy: 0.9328
40992/60000 [===================>..........] - ETA: 35s - loss: 0.2212 - categorical_accuracy: 0.9328
41024/60000 [===================>..........] - ETA: 35s - loss: 0.2211 - categorical_accuracy: 0.9328
41056/60000 [===================>..........] - ETA: 35s - loss: 0.2209 - categorical_accuracy: 0.9329
41088/60000 [===================>..........] - ETA: 35s - loss: 0.2208 - categorical_accuracy: 0.9329
41120/60000 [===================>..........] - ETA: 35s - loss: 0.2207 - categorical_accuracy: 0.9330
41152/60000 [===================>..........] - ETA: 35s - loss: 0.2206 - categorical_accuracy: 0.9330
41184/60000 [===================>..........] - ETA: 35s - loss: 0.2205 - categorical_accuracy: 0.9330
41216/60000 [===================>..........] - ETA: 35s - loss: 0.2204 - categorical_accuracy: 0.9330
41248/60000 [===================>..........] - ETA: 35s - loss: 0.2203 - categorical_accuracy: 0.9331
41280/60000 [===================>..........] - ETA: 35s - loss: 0.2201 - categorical_accuracy: 0.9331
41312/60000 [===================>..........] - ETA: 35s - loss: 0.2200 - categorical_accuracy: 0.9331
41344/60000 [===================>..........] - ETA: 35s - loss: 0.2199 - categorical_accuracy: 0.9331
41376/60000 [===================>..........] - ETA: 35s - loss: 0.2198 - categorical_accuracy: 0.9332
41408/60000 [===================>..........] - ETA: 35s - loss: 0.2197 - categorical_accuracy: 0.9332
41440/60000 [===================>..........] - ETA: 35s - loss: 0.2196 - categorical_accuracy: 0.9333
41472/60000 [===================>..........] - ETA: 35s - loss: 0.2194 - categorical_accuracy: 0.9333
41504/60000 [===================>..........] - ETA: 34s - loss: 0.2193 - categorical_accuracy: 0.9333
41536/60000 [===================>..........] - ETA: 34s - loss: 0.2192 - categorical_accuracy: 0.9334
41568/60000 [===================>..........] - ETA: 34s - loss: 0.2192 - categorical_accuracy: 0.9334
41600/60000 [===================>..........] - ETA: 34s - loss: 0.2192 - categorical_accuracy: 0.9334
41632/60000 [===================>..........] - ETA: 34s - loss: 0.2191 - categorical_accuracy: 0.9333
41664/60000 [===================>..........] - ETA: 34s - loss: 0.2191 - categorical_accuracy: 0.9334
41696/60000 [===================>..........] - ETA: 34s - loss: 0.2190 - categorical_accuracy: 0.9334
41728/60000 [===================>..........] - ETA: 34s - loss: 0.2189 - categorical_accuracy: 0.9334
41760/60000 [===================>..........] - ETA: 34s - loss: 0.2187 - categorical_accuracy: 0.9335
41792/60000 [===================>..........] - ETA: 34s - loss: 0.2186 - categorical_accuracy: 0.9335
41824/60000 [===================>..........] - ETA: 34s - loss: 0.2185 - categorical_accuracy: 0.9336
41856/60000 [===================>..........] - ETA: 34s - loss: 0.2185 - categorical_accuracy: 0.9336
41888/60000 [===================>..........] - ETA: 34s - loss: 0.2184 - categorical_accuracy: 0.9336
41920/60000 [===================>..........] - ETA: 34s - loss: 0.2182 - categorical_accuracy: 0.9337
41952/60000 [===================>..........] - ETA: 34s - loss: 0.2182 - categorical_accuracy: 0.9336
41984/60000 [===================>..........] - ETA: 34s - loss: 0.2182 - categorical_accuracy: 0.9336
42016/60000 [====================>.........] - ETA: 33s - loss: 0.2180 - categorical_accuracy: 0.9337
42048/60000 [====================>.........] - ETA: 33s - loss: 0.2180 - categorical_accuracy: 0.9337
42080/60000 [====================>.........] - ETA: 33s - loss: 0.2179 - categorical_accuracy: 0.9337
42112/60000 [====================>.........] - ETA: 33s - loss: 0.2177 - categorical_accuracy: 0.9337
42144/60000 [====================>.........] - ETA: 33s - loss: 0.2177 - categorical_accuracy: 0.9338
42176/60000 [====================>.........] - ETA: 33s - loss: 0.2176 - categorical_accuracy: 0.9338
42208/60000 [====================>.........] - ETA: 33s - loss: 0.2175 - categorical_accuracy: 0.9338
42240/60000 [====================>.........] - ETA: 33s - loss: 0.2174 - categorical_accuracy: 0.9338
42272/60000 [====================>.........] - ETA: 33s - loss: 0.2173 - categorical_accuracy: 0.9338
42304/60000 [====================>.........] - ETA: 33s - loss: 0.2172 - categorical_accuracy: 0.9338
42336/60000 [====================>.........] - ETA: 33s - loss: 0.2171 - categorical_accuracy: 0.9339
42368/60000 [====================>.........] - ETA: 33s - loss: 0.2170 - categorical_accuracy: 0.9339
42400/60000 [====================>.........] - ETA: 33s - loss: 0.2169 - categorical_accuracy: 0.9339
42432/60000 [====================>.........] - ETA: 33s - loss: 0.2168 - categorical_accuracy: 0.9339
42464/60000 [====================>.........] - ETA: 33s - loss: 0.2166 - categorical_accuracy: 0.9340
42496/60000 [====================>.........] - ETA: 33s - loss: 0.2166 - categorical_accuracy: 0.9340
42528/60000 [====================>.........] - ETA: 33s - loss: 0.2165 - categorical_accuracy: 0.9340
42560/60000 [====================>.........] - ETA: 32s - loss: 0.2165 - categorical_accuracy: 0.9340
42592/60000 [====================>.........] - ETA: 32s - loss: 0.2163 - categorical_accuracy: 0.9341
42624/60000 [====================>.........] - ETA: 32s - loss: 0.2162 - categorical_accuracy: 0.9341
42656/60000 [====================>.........] - ETA: 32s - loss: 0.2160 - categorical_accuracy: 0.9342
42688/60000 [====================>.........] - ETA: 32s - loss: 0.2159 - categorical_accuracy: 0.9342
42720/60000 [====================>.........] - ETA: 32s - loss: 0.2158 - categorical_accuracy: 0.9342
42752/60000 [====================>.........] - ETA: 32s - loss: 0.2157 - categorical_accuracy: 0.9343
42784/60000 [====================>.........] - ETA: 32s - loss: 0.2156 - categorical_accuracy: 0.9343
42816/60000 [====================>.........] - ETA: 32s - loss: 0.2155 - categorical_accuracy: 0.9344
42848/60000 [====================>.........] - ETA: 32s - loss: 0.2155 - categorical_accuracy: 0.9343
42880/60000 [====================>.........] - ETA: 32s - loss: 0.2155 - categorical_accuracy: 0.9343
42912/60000 [====================>.........] - ETA: 32s - loss: 0.2153 - categorical_accuracy: 0.9344
42944/60000 [====================>.........] - ETA: 32s - loss: 0.2152 - categorical_accuracy: 0.9344
42976/60000 [====================>.........] - ETA: 32s - loss: 0.2152 - categorical_accuracy: 0.9344
43008/60000 [====================>.........] - ETA: 32s - loss: 0.2150 - categorical_accuracy: 0.9345
43040/60000 [====================>.........] - ETA: 32s - loss: 0.2149 - categorical_accuracy: 0.9345
43072/60000 [====================>.........] - ETA: 31s - loss: 0.2149 - categorical_accuracy: 0.9345
43104/60000 [====================>.........] - ETA: 31s - loss: 0.2147 - categorical_accuracy: 0.9345
43136/60000 [====================>.........] - ETA: 31s - loss: 0.2148 - categorical_accuracy: 0.9345
43168/60000 [====================>.........] - ETA: 31s - loss: 0.2149 - categorical_accuracy: 0.9345
43200/60000 [====================>.........] - ETA: 31s - loss: 0.2147 - categorical_accuracy: 0.9345
43232/60000 [====================>.........] - ETA: 31s - loss: 0.2146 - categorical_accuracy: 0.9346
43264/60000 [====================>.........] - ETA: 31s - loss: 0.2145 - categorical_accuracy: 0.9346
43296/60000 [====================>.........] - ETA: 31s - loss: 0.2143 - categorical_accuracy: 0.9347
43328/60000 [====================>.........] - ETA: 31s - loss: 0.2143 - categorical_accuracy: 0.9347
43360/60000 [====================>.........] - ETA: 31s - loss: 0.2143 - categorical_accuracy: 0.9347
43392/60000 [====================>.........] - ETA: 31s - loss: 0.2142 - categorical_accuracy: 0.9347
43424/60000 [====================>.........] - ETA: 31s - loss: 0.2141 - categorical_accuracy: 0.9347
43456/60000 [====================>.........] - ETA: 31s - loss: 0.2140 - categorical_accuracy: 0.9348
43488/60000 [====================>.........] - ETA: 31s - loss: 0.2139 - categorical_accuracy: 0.9348
43520/60000 [====================>.........] - ETA: 31s - loss: 0.2137 - categorical_accuracy: 0.9349
43552/60000 [====================>.........] - ETA: 31s - loss: 0.2136 - categorical_accuracy: 0.9349
43584/60000 [====================>.........] - ETA: 31s - loss: 0.2135 - categorical_accuracy: 0.9350
43616/60000 [====================>.........] - ETA: 30s - loss: 0.2135 - categorical_accuracy: 0.9349
43648/60000 [====================>.........] - ETA: 30s - loss: 0.2135 - categorical_accuracy: 0.9350
43680/60000 [====================>.........] - ETA: 30s - loss: 0.2135 - categorical_accuracy: 0.9350
43712/60000 [====================>.........] - ETA: 30s - loss: 0.2134 - categorical_accuracy: 0.9350
43744/60000 [====================>.........] - ETA: 30s - loss: 0.2133 - categorical_accuracy: 0.9350
43776/60000 [====================>.........] - ETA: 30s - loss: 0.2131 - categorical_accuracy: 0.9351
43808/60000 [====================>.........] - ETA: 30s - loss: 0.2130 - categorical_accuracy: 0.9351
43840/60000 [====================>.........] - ETA: 30s - loss: 0.2129 - categorical_accuracy: 0.9352
43872/60000 [====================>.........] - ETA: 30s - loss: 0.2127 - categorical_accuracy: 0.9352
43904/60000 [====================>.........] - ETA: 30s - loss: 0.2126 - categorical_accuracy: 0.9352
43936/60000 [====================>.........] - ETA: 30s - loss: 0.2125 - categorical_accuracy: 0.9353
43968/60000 [====================>.........] - ETA: 30s - loss: 0.2124 - categorical_accuracy: 0.9353
44000/60000 [=====================>........] - ETA: 30s - loss: 0.2125 - categorical_accuracy: 0.9352
44032/60000 [=====================>........] - ETA: 30s - loss: 0.2125 - categorical_accuracy: 0.9353
44064/60000 [=====================>........] - ETA: 30s - loss: 0.2124 - categorical_accuracy: 0.9353
44096/60000 [=====================>........] - ETA: 30s - loss: 0.2123 - categorical_accuracy: 0.9353
44128/60000 [=====================>........] - ETA: 29s - loss: 0.2122 - categorical_accuracy: 0.9353
44160/60000 [=====================>........] - ETA: 29s - loss: 0.2123 - categorical_accuracy: 0.9353
44192/60000 [=====================>........] - ETA: 29s - loss: 0.2122 - categorical_accuracy: 0.9353
44224/60000 [=====================>........] - ETA: 29s - loss: 0.2120 - categorical_accuracy: 0.9354
44256/60000 [=====================>........] - ETA: 29s - loss: 0.2119 - categorical_accuracy: 0.9354
44288/60000 [=====================>........] - ETA: 29s - loss: 0.2119 - categorical_accuracy: 0.9354
44320/60000 [=====================>........] - ETA: 29s - loss: 0.2118 - categorical_accuracy: 0.9354
44352/60000 [=====================>........] - ETA: 29s - loss: 0.2118 - categorical_accuracy: 0.9354
44384/60000 [=====================>........] - ETA: 29s - loss: 0.2117 - categorical_accuracy: 0.9354
44416/60000 [=====================>........] - ETA: 29s - loss: 0.2116 - categorical_accuracy: 0.9354
44448/60000 [=====================>........] - ETA: 29s - loss: 0.2115 - categorical_accuracy: 0.9355
44480/60000 [=====================>........] - ETA: 29s - loss: 0.2116 - categorical_accuracy: 0.9354
44512/60000 [=====================>........] - ETA: 29s - loss: 0.2115 - categorical_accuracy: 0.9355
44544/60000 [=====================>........] - ETA: 29s - loss: 0.2113 - categorical_accuracy: 0.9355
44576/60000 [=====================>........] - ETA: 29s - loss: 0.2112 - categorical_accuracy: 0.9355
44608/60000 [=====================>........] - ETA: 29s - loss: 0.2111 - categorical_accuracy: 0.9356
44640/60000 [=====================>........] - ETA: 29s - loss: 0.2110 - categorical_accuracy: 0.9356
44672/60000 [=====================>........] - ETA: 28s - loss: 0.2109 - categorical_accuracy: 0.9356
44704/60000 [=====================>........] - ETA: 28s - loss: 0.2108 - categorical_accuracy: 0.9357
44736/60000 [=====================>........] - ETA: 28s - loss: 0.2108 - categorical_accuracy: 0.9356
44768/60000 [=====================>........] - ETA: 28s - loss: 0.2107 - categorical_accuracy: 0.9357
44800/60000 [=====================>........] - ETA: 28s - loss: 0.2105 - categorical_accuracy: 0.9357
44832/60000 [=====================>........] - ETA: 28s - loss: 0.2104 - categorical_accuracy: 0.9357
44864/60000 [=====================>........] - ETA: 28s - loss: 0.2103 - categorical_accuracy: 0.9358
44896/60000 [=====================>........] - ETA: 28s - loss: 0.2104 - categorical_accuracy: 0.9358
44928/60000 [=====================>........] - ETA: 28s - loss: 0.2103 - categorical_accuracy: 0.9358
44960/60000 [=====================>........] - ETA: 28s - loss: 0.2102 - categorical_accuracy: 0.9358
44992/60000 [=====================>........] - ETA: 28s - loss: 0.2100 - categorical_accuracy: 0.9359
45024/60000 [=====================>........] - ETA: 28s - loss: 0.2099 - categorical_accuracy: 0.9359
45056/60000 [=====================>........] - ETA: 28s - loss: 0.2098 - categorical_accuracy: 0.9360
45088/60000 [=====================>........] - ETA: 28s - loss: 0.2097 - categorical_accuracy: 0.9360
45120/60000 [=====================>........] - ETA: 28s - loss: 0.2096 - categorical_accuracy: 0.9360
45152/60000 [=====================>........] - ETA: 28s - loss: 0.2095 - categorical_accuracy: 0.9361
45184/60000 [=====================>........] - ETA: 27s - loss: 0.2095 - categorical_accuracy: 0.9361
45216/60000 [=====================>........] - ETA: 27s - loss: 0.2094 - categorical_accuracy: 0.9361
45248/60000 [=====================>........] - ETA: 27s - loss: 0.2092 - categorical_accuracy: 0.9362
45280/60000 [=====================>........] - ETA: 27s - loss: 0.2091 - categorical_accuracy: 0.9362
45312/60000 [=====================>........] - ETA: 27s - loss: 0.2089 - categorical_accuracy: 0.9363
45344/60000 [=====================>........] - ETA: 27s - loss: 0.2088 - categorical_accuracy: 0.9363
45376/60000 [=====================>........] - ETA: 27s - loss: 0.2087 - categorical_accuracy: 0.9363
45408/60000 [=====================>........] - ETA: 27s - loss: 0.2087 - categorical_accuracy: 0.9363
45440/60000 [=====================>........] - ETA: 27s - loss: 0.2086 - categorical_accuracy: 0.9364
45472/60000 [=====================>........] - ETA: 27s - loss: 0.2085 - categorical_accuracy: 0.9364
45504/60000 [=====================>........] - ETA: 27s - loss: 0.2084 - categorical_accuracy: 0.9364
45536/60000 [=====================>........] - ETA: 27s - loss: 0.2083 - categorical_accuracy: 0.9365
45568/60000 [=====================>........] - ETA: 27s - loss: 0.2082 - categorical_accuracy: 0.9365
45600/60000 [=====================>........] - ETA: 27s - loss: 0.2081 - categorical_accuracy: 0.9365
45632/60000 [=====================>........] - ETA: 27s - loss: 0.2080 - categorical_accuracy: 0.9365
45664/60000 [=====================>........] - ETA: 27s - loss: 0.2080 - categorical_accuracy: 0.9366
45696/60000 [=====================>........] - ETA: 27s - loss: 0.2079 - categorical_accuracy: 0.9366
45728/60000 [=====================>........] - ETA: 26s - loss: 0.2078 - categorical_accuracy: 0.9366
45760/60000 [=====================>........] - ETA: 26s - loss: 0.2076 - categorical_accuracy: 0.9367
45792/60000 [=====================>........] - ETA: 26s - loss: 0.2076 - categorical_accuracy: 0.9367
45824/60000 [=====================>........] - ETA: 26s - loss: 0.2076 - categorical_accuracy: 0.9367
45856/60000 [=====================>........] - ETA: 26s - loss: 0.2075 - categorical_accuracy: 0.9367
45888/60000 [=====================>........] - ETA: 26s - loss: 0.2075 - categorical_accuracy: 0.9367
45920/60000 [=====================>........] - ETA: 26s - loss: 0.2074 - categorical_accuracy: 0.9367
45952/60000 [=====================>........] - ETA: 26s - loss: 0.2074 - categorical_accuracy: 0.9367
45984/60000 [=====================>........] - ETA: 26s - loss: 0.2073 - categorical_accuracy: 0.9368
46016/60000 [======================>.......] - ETA: 26s - loss: 0.2072 - categorical_accuracy: 0.9368
46048/60000 [======================>.......] - ETA: 26s - loss: 0.2071 - categorical_accuracy: 0.9368
46080/60000 [======================>.......] - ETA: 26s - loss: 0.2070 - categorical_accuracy: 0.9368
46112/60000 [======================>.......] - ETA: 26s - loss: 0.2069 - categorical_accuracy: 0.9368
46144/60000 [======================>.......] - ETA: 26s - loss: 0.2069 - categorical_accuracy: 0.9368
46176/60000 [======================>.......] - ETA: 26s - loss: 0.2068 - categorical_accuracy: 0.9369
46208/60000 [======================>.......] - ETA: 26s - loss: 0.2067 - categorical_accuracy: 0.9369
46240/60000 [======================>.......] - ETA: 25s - loss: 0.2067 - categorical_accuracy: 0.9369
46272/60000 [======================>.......] - ETA: 25s - loss: 0.2066 - categorical_accuracy: 0.9369
46304/60000 [======================>.......] - ETA: 25s - loss: 0.2065 - categorical_accuracy: 0.9370
46336/60000 [======================>.......] - ETA: 25s - loss: 0.2064 - categorical_accuracy: 0.9370
46368/60000 [======================>.......] - ETA: 25s - loss: 0.2064 - categorical_accuracy: 0.9370
46400/60000 [======================>.......] - ETA: 25s - loss: 0.2063 - categorical_accuracy: 0.9370
46432/60000 [======================>.......] - ETA: 25s - loss: 0.2062 - categorical_accuracy: 0.9370
46464/60000 [======================>.......] - ETA: 25s - loss: 0.2061 - categorical_accuracy: 0.9370
46496/60000 [======================>.......] - ETA: 25s - loss: 0.2061 - categorical_accuracy: 0.9370
46528/60000 [======================>.......] - ETA: 25s - loss: 0.2060 - categorical_accuracy: 0.9371
46560/60000 [======================>.......] - ETA: 25s - loss: 0.2059 - categorical_accuracy: 0.9371
46592/60000 [======================>.......] - ETA: 25s - loss: 0.2059 - categorical_accuracy: 0.9371
46624/60000 [======================>.......] - ETA: 25s - loss: 0.2057 - categorical_accuracy: 0.9371
46656/60000 [======================>.......] - ETA: 25s - loss: 0.2056 - categorical_accuracy: 0.9372
46688/60000 [======================>.......] - ETA: 25s - loss: 0.2055 - categorical_accuracy: 0.9372
46720/60000 [======================>.......] - ETA: 25s - loss: 0.2054 - categorical_accuracy: 0.9372
46752/60000 [======================>.......] - ETA: 25s - loss: 0.2052 - categorical_accuracy: 0.9372
46784/60000 [======================>.......] - ETA: 24s - loss: 0.2052 - categorical_accuracy: 0.9373
46816/60000 [======================>.......] - ETA: 24s - loss: 0.2051 - categorical_accuracy: 0.9373
46848/60000 [======================>.......] - ETA: 24s - loss: 0.2051 - categorical_accuracy: 0.9373
46880/60000 [======================>.......] - ETA: 24s - loss: 0.2051 - categorical_accuracy: 0.9373
46912/60000 [======================>.......] - ETA: 24s - loss: 0.2053 - categorical_accuracy: 0.9374
46944/60000 [======================>.......] - ETA: 24s - loss: 0.2053 - categorical_accuracy: 0.9374
46976/60000 [======================>.......] - ETA: 24s - loss: 0.2052 - categorical_accuracy: 0.9374
47008/60000 [======================>.......] - ETA: 24s - loss: 0.2051 - categorical_accuracy: 0.9374
47040/60000 [======================>.......] - ETA: 24s - loss: 0.2050 - categorical_accuracy: 0.9374
47072/60000 [======================>.......] - ETA: 24s - loss: 0.2048 - categorical_accuracy: 0.9375
47104/60000 [======================>.......] - ETA: 24s - loss: 0.2047 - categorical_accuracy: 0.9375
47136/60000 [======================>.......] - ETA: 24s - loss: 0.2046 - categorical_accuracy: 0.9375
47168/60000 [======================>.......] - ETA: 24s - loss: 0.2045 - categorical_accuracy: 0.9376
47200/60000 [======================>.......] - ETA: 24s - loss: 0.2044 - categorical_accuracy: 0.9376
47232/60000 [======================>.......] - ETA: 24s - loss: 0.2043 - categorical_accuracy: 0.9376
47264/60000 [======================>.......] - ETA: 24s - loss: 0.2043 - categorical_accuracy: 0.9376
47296/60000 [======================>.......] - ETA: 23s - loss: 0.2043 - categorical_accuracy: 0.9377
47328/60000 [======================>.......] - ETA: 23s - loss: 0.2042 - categorical_accuracy: 0.9377
47360/60000 [======================>.......] - ETA: 23s - loss: 0.2041 - categorical_accuracy: 0.9377
47392/60000 [======================>.......] - ETA: 23s - loss: 0.2040 - categorical_accuracy: 0.9377
47424/60000 [======================>.......] - ETA: 23s - loss: 0.2040 - categorical_accuracy: 0.9378
47456/60000 [======================>.......] - ETA: 23s - loss: 0.2040 - categorical_accuracy: 0.9378
47488/60000 [======================>.......] - ETA: 23s - loss: 0.2039 - categorical_accuracy: 0.9378
47520/60000 [======================>.......] - ETA: 23s - loss: 0.2041 - categorical_accuracy: 0.9378
47552/60000 [======================>.......] - ETA: 23s - loss: 0.2040 - categorical_accuracy: 0.9378
47584/60000 [======================>.......] - ETA: 23s - loss: 0.2039 - categorical_accuracy: 0.9378
47616/60000 [======================>.......] - ETA: 23s - loss: 0.2038 - categorical_accuracy: 0.9378
47648/60000 [======================>.......] - ETA: 23s - loss: 0.2037 - categorical_accuracy: 0.9379
47680/60000 [======================>.......] - ETA: 23s - loss: 0.2038 - categorical_accuracy: 0.9379
47712/60000 [======================>.......] - ETA: 23s - loss: 0.2038 - categorical_accuracy: 0.9379
47744/60000 [======================>.......] - ETA: 23s - loss: 0.2038 - categorical_accuracy: 0.9379
47776/60000 [======================>.......] - ETA: 23s - loss: 0.2037 - categorical_accuracy: 0.9379
47808/60000 [======================>.......] - ETA: 23s - loss: 0.2036 - categorical_accuracy: 0.9379
47840/60000 [======================>.......] - ETA: 22s - loss: 0.2035 - categorical_accuracy: 0.9380
47872/60000 [======================>.......] - ETA: 22s - loss: 0.2034 - categorical_accuracy: 0.9380
47904/60000 [======================>.......] - ETA: 22s - loss: 0.2034 - categorical_accuracy: 0.9380
47936/60000 [======================>.......] - ETA: 22s - loss: 0.2033 - categorical_accuracy: 0.9380
47968/60000 [======================>.......] - ETA: 22s - loss: 0.2032 - categorical_accuracy: 0.9380
48000/60000 [=======================>......] - ETA: 22s - loss: 0.2031 - categorical_accuracy: 0.9381
48032/60000 [=======================>......] - ETA: 22s - loss: 0.2030 - categorical_accuracy: 0.9381
48064/60000 [=======================>......] - ETA: 22s - loss: 0.2030 - categorical_accuracy: 0.9381
48096/60000 [=======================>......] - ETA: 22s - loss: 0.2029 - categorical_accuracy: 0.9381
48128/60000 [=======================>......] - ETA: 22s - loss: 0.2028 - categorical_accuracy: 0.9382
48160/60000 [=======================>......] - ETA: 22s - loss: 0.2027 - categorical_accuracy: 0.9382
48192/60000 [=======================>......] - ETA: 22s - loss: 0.2026 - categorical_accuracy: 0.9382
48224/60000 [=======================>......] - ETA: 22s - loss: 0.2025 - categorical_accuracy: 0.9383
48256/60000 [=======================>......] - ETA: 22s - loss: 0.2023 - categorical_accuracy: 0.9383
48288/60000 [=======================>......] - ETA: 22s - loss: 0.2022 - categorical_accuracy: 0.9383
48320/60000 [=======================>......] - ETA: 22s - loss: 0.2021 - categorical_accuracy: 0.9384
48352/60000 [=======================>......] - ETA: 21s - loss: 0.2020 - categorical_accuracy: 0.9384
48384/60000 [=======================>......] - ETA: 21s - loss: 0.2019 - categorical_accuracy: 0.9384
48416/60000 [=======================>......] - ETA: 21s - loss: 0.2019 - categorical_accuracy: 0.9384
48448/60000 [=======================>......] - ETA: 21s - loss: 0.2019 - categorical_accuracy: 0.9384
48480/60000 [=======================>......] - ETA: 21s - loss: 0.2017 - categorical_accuracy: 0.9384
48512/60000 [=======================>......] - ETA: 21s - loss: 0.2019 - categorical_accuracy: 0.9384
48544/60000 [=======================>......] - ETA: 21s - loss: 0.2018 - categorical_accuracy: 0.9384
48576/60000 [=======================>......] - ETA: 21s - loss: 0.2018 - categorical_accuracy: 0.9384
48608/60000 [=======================>......] - ETA: 21s - loss: 0.2017 - categorical_accuracy: 0.9385
48640/60000 [=======================>......] - ETA: 21s - loss: 0.2016 - categorical_accuracy: 0.9385
48672/60000 [=======================>......] - ETA: 21s - loss: 0.2016 - categorical_accuracy: 0.9385
48704/60000 [=======================>......] - ETA: 21s - loss: 0.2015 - categorical_accuracy: 0.9386
48736/60000 [=======================>......] - ETA: 21s - loss: 0.2015 - categorical_accuracy: 0.9386
48768/60000 [=======================>......] - ETA: 21s - loss: 0.2015 - categorical_accuracy: 0.9386
48800/60000 [=======================>......] - ETA: 21s - loss: 0.2016 - categorical_accuracy: 0.9385
48832/60000 [=======================>......] - ETA: 21s - loss: 0.2015 - categorical_accuracy: 0.9385
48864/60000 [=======================>......] - ETA: 21s - loss: 0.2014 - categorical_accuracy: 0.9385
48896/60000 [=======================>......] - ETA: 20s - loss: 0.2015 - categorical_accuracy: 0.9385
48928/60000 [=======================>......] - ETA: 20s - loss: 0.2013 - categorical_accuracy: 0.9386
48960/60000 [=======================>......] - ETA: 20s - loss: 0.2013 - categorical_accuracy: 0.9386
48992/60000 [=======================>......] - ETA: 20s - loss: 0.2012 - categorical_accuracy: 0.9386
49024/60000 [=======================>......] - ETA: 20s - loss: 0.2011 - categorical_accuracy: 0.9387
49056/60000 [=======================>......] - ETA: 20s - loss: 0.2010 - categorical_accuracy: 0.9387
49088/60000 [=======================>......] - ETA: 20s - loss: 0.2009 - categorical_accuracy: 0.9387
49120/60000 [=======================>......] - ETA: 20s - loss: 0.2008 - categorical_accuracy: 0.9388
49152/60000 [=======================>......] - ETA: 20s - loss: 0.2007 - categorical_accuracy: 0.9388
49184/60000 [=======================>......] - ETA: 20s - loss: 0.2006 - categorical_accuracy: 0.9388
49216/60000 [=======================>......] - ETA: 20s - loss: 0.2005 - categorical_accuracy: 0.9389
49248/60000 [=======================>......] - ETA: 20s - loss: 0.2005 - categorical_accuracy: 0.9389
49280/60000 [=======================>......] - ETA: 20s - loss: 0.2004 - categorical_accuracy: 0.9389
49312/60000 [=======================>......] - ETA: 20s - loss: 0.2005 - categorical_accuracy: 0.9389
49344/60000 [=======================>......] - ETA: 20s - loss: 0.2003 - categorical_accuracy: 0.9390
49376/60000 [=======================>......] - ETA: 20s - loss: 0.2003 - categorical_accuracy: 0.9390
49408/60000 [=======================>......] - ETA: 19s - loss: 0.2003 - categorical_accuracy: 0.9390
49440/60000 [=======================>......] - ETA: 19s - loss: 0.2003 - categorical_accuracy: 0.9390
49472/60000 [=======================>......] - ETA: 19s - loss: 0.2003 - categorical_accuracy: 0.9390
49504/60000 [=======================>......] - ETA: 19s - loss: 0.2002 - categorical_accuracy: 0.9390
49536/60000 [=======================>......] - ETA: 19s - loss: 0.2001 - categorical_accuracy: 0.9391
49568/60000 [=======================>......] - ETA: 19s - loss: 0.2000 - categorical_accuracy: 0.9391
49600/60000 [=======================>......] - ETA: 19s - loss: 0.2000 - categorical_accuracy: 0.9391
49632/60000 [=======================>......] - ETA: 19s - loss: 0.1999 - categorical_accuracy: 0.9391
49664/60000 [=======================>......] - ETA: 19s - loss: 0.1999 - categorical_accuracy: 0.9391
49696/60000 [=======================>......] - ETA: 19s - loss: 0.1998 - categorical_accuracy: 0.9391
49728/60000 [=======================>......] - ETA: 19s - loss: 0.1997 - categorical_accuracy: 0.9391
49760/60000 [=======================>......] - ETA: 19s - loss: 0.1997 - categorical_accuracy: 0.9392
49792/60000 [=======================>......] - ETA: 19s - loss: 0.1996 - categorical_accuracy: 0.9392
49824/60000 [=======================>......] - ETA: 19s - loss: 0.1996 - categorical_accuracy: 0.9392
49856/60000 [=======================>......] - ETA: 19s - loss: 0.1995 - categorical_accuracy: 0.9392
49888/60000 [=======================>......] - ETA: 19s - loss: 0.1994 - categorical_accuracy: 0.9393
49920/60000 [=======================>......] - ETA: 19s - loss: 0.1993 - categorical_accuracy: 0.9393
49952/60000 [=======================>......] - ETA: 18s - loss: 0.1993 - categorical_accuracy: 0.9393
49984/60000 [=======================>......] - ETA: 18s - loss: 0.1992 - categorical_accuracy: 0.9393
50016/60000 [========================>.....] - ETA: 18s - loss: 0.1991 - categorical_accuracy: 0.9394
50048/60000 [========================>.....] - ETA: 18s - loss: 0.1990 - categorical_accuracy: 0.9394
50080/60000 [========================>.....] - ETA: 18s - loss: 0.1990 - categorical_accuracy: 0.9394
50112/60000 [========================>.....] - ETA: 18s - loss: 0.1989 - categorical_accuracy: 0.9394
50144/60000 [========================>.....] - ETA: 18s - loss: 0.1989 - categorical_accuracy: 0.9394
50176/60000 [========================>.....] - ETA: 18s - loss: 0.1988 - categorical_accuracy: 0.9394
50208/60000 [========================>.....] - ETA: 18s - loss: 0.1988 - categorical_accuracy: 0.9394
50240/60000 [========================>.....] - ETA: 18s - loss: 0.1987 - categorical_accuracy: 0.9395
50272/60000 [========================>.....] - ETA: 18s - loss: 0.1986 - categorical_accuracy: 0.9395
50304/60000 [========================>.....] - ETA: 18s - loss: 0.1985 - categorical_accuracy: 0.9395
50336/60000 [========================>.....] - ETA: 18s - loss: 0.1986 - categorical_accuracy: 0.9395
50368/60000 [========================>.....] - ETA: 18s - loss: 0.1985 - categorical_accuracy: 0.9396
50400/60000 [========================>.....] - ETA: 18s - loss: 0.1985 - categorical_accuracy: 0.9395
50432/60000 [========================>.....] - ETA: 18s - loss: 0.1985 - categorical_accuracy: 0.9396
50464/60000 [========================>.....] - ETA: 17s - loss: 0.1984 - categorical_accuracy: 0.9396
50496/60000 [========================>.....] - ETA: 17s - loss: 0.1983 - categorical_accuracy: 0.9396
50528/60000 [========================>.....] - ETA: 17s - loss: 0.1983 - categorical_accuracy: 0.9396
50560/60000 [========================>.....] - ETA: 17s - loss: 0.1982 - categorical_accuracy: 0.9397
50592/60000 [========================>.....] - ETA: 17s - loss: 0.1981 - categorical_accuracy: 0.9397
50624/60000 [========================>.....] - ETA: 17s - loss: 0.1980 - categorical_accuracy: 0.9397
50656/60000 [========================>.....] - ETA: 17s - loss: 0.1980 - categorical_accuracy: 0.9398
50688/60000 [========================>.....] - ETA: 17s - loss: 0.1979 - categorical_accuracy: 0.9398
50720/60000 [========================>.....] - ETA: 17s - loss: 0.1979 - categorical_accuracy: 0.9398
50752/60000 [========================>.....] - ETA: 17s - loss: 0.1978 - categorical_accuracy: 0.9398
50784/60000 [========================>.....] - ETA: 17s - loss: 0.1978 - categorical_accuracy: 0.9398
50816/60000 [========================>.....] - ETA: 17s - loss: 0.1977 - categorical_accuracy: 0.9399
50848/60000 [========================>.....] - ETA: 17s - loss: 0.1976 - categorical_accuracy: 0.9399
50880/60000 [========================>.....] - ETA: 17s - loss: 0.1975 - categorical_accuracy: 0.9399
50912/60000 [========================>.....] - ETA: 17s - loss: 0.1976 - categorical_accuracy: 0.9399
50944/60000 [========================>.....] - ETA: 17s - loss: 0.1975 - categorical_accuracy: 0.9399
50976/60000 [========================>.....] - ETA: 17s - loss: 0.1974 - categorical_accuracy: 0.9399
51008/60000 [========================>.....] - ETA: 16s - loss: 0.1973 - categorical_accuracy: 0.9399
51040/60000 [========================>.....] - ETA: 16s - loss: 0.1973 - categorical_accuracy: 0.9399
51072/60000 [========================>.....] - ETA: 16s - loss: 0.1972 - categorical_accuracy: 0.9399
51104/60000 [========================>.....] - ETA: 16s - loss: 0.1971 - categorical_accuracy: 0.9400
51136/60000 [========================>.....] - ETA: 16s - loss: 0.1971 - categorical_accuracy: 0.9400
51168/60000 [========================>.....] - ETA: 16s - loss: 0.1970 - categorical_accuracy: 0.9400
51200/60000 [========================>.....] - ETA: 16s - loss: 0.1969 - categorical_accuracy: 0.9401
51232/60000 [========================>.....] - ETA: 16s - loss: 0.1968 - categorical_accuracy: 0.9401
51264/60000 [========================>.....] - ETA: 16s - loss: 0.1967 - categorical_accuracy: 0.9401
51296/60000 [========================>.....] - ETA: 16s - loss: 0.1966 - categorical_accuracy: 0.9402
51328/60000 [========================>.....] - ETA: 16s - loss: 0.1965 - categorical_accuracy: 0.9402
51360/60000 [========================>.....] - ETA: 16s - loss: 0.1964 - categorical_accuracy: 0.9402
51392/60000 [========================>.....] - ETA: 16s - loss: 0.1964 - categorical_accuracy: 0.9402
51424/60000 [========================>.....] - ETA: 16s - loss: 0.1964 - categorical_accuracy: 0.9402
51456/60000 [========================>.....] - ETA: 16s - loss: 0.1963 - categorical_accuracy: 0.9402
51488/60000 [========================>.....] - ETA: 16s - loss: 0.1963 - categorical_accuracy: 0.9402
51520/60000 [========================>.....] - ETA: 15s - loss: 0.1962 - categorical_accuracy: 0.9403
51552/60000 [========================>.....] - ETA: 15s - loss: 0.1961 - categorical_accuracy: 0.9403
51584/60000 [========================>.....] - ETA: 15s - loss: 0.1960 - categorical_accuracy: 0.9403
51616/60000 [========================>.....] - ETA: 15s - loss: 0.1959 - categorical_accuracy: 0.9403
51648/60000 [========================>.....] - ETA: 15s - loss: 0.1958 - categorical_accuracy: 0.9404
51680/60000 [========================>.....] - ETA: 15s - loss: 0.1957 - categorical_accuracy: 0.9404
51712/60000 [========================>.....] - ETA: 15s - loss: 0.1956 - categorical_accuracy: 0.9404
51744/60000 [========================>.....] - ETA: 15s - loss: 0.1955 - categorical_accuracy: 0.9405
51776/60000 [========================>.....] - ETA: 15s - loss: 0.1956 - categorical_accuracy: 0.9405
51808/60000 [========================>.....] - ETA: 15s - loss: 0.1955 - categorical_accuracy: 0.9405
51840/60000 [========================>.....] - ETA: 15s - loss: 0.1955 - categorical_accuracy: 0.9405
51872/60000 [========================>.....] - ETA: 15s - loss: 0.1954 - categorical_accuracy: 0.9405
51904/60000 [========================>.....] - ETA: 15s - loss: 0.1953 - categorical_accuracy: 0.9406
51936/60000 [========================>.....] - ETA: 15s - loss: 0.1952 - categorical_accuracy: 0.9406
51968/60000 [========================>.....] - ETA: 15s - loss: 0.1951 - categorical_accuracy: 0.9406
52000/60000 [=========================>....] - ETA: 15s - loss: 0.1950 - categorical_accuracy: 0.9407
52032/60000 [=========================>....] - ETA: 15s - loss: 0.1949 - categorical_accuracy: 0.9407
52064/60000 [=========================>....] - ETA: 14s - loss: 0.1948 - categorical_accuracy: 0.9407
52096/60000 [=========================>....] - ETA: 14s - loss: 0.1947 - categorical_accuracy: 0.9407
52128/60000 [=========================>....] - ETA: 14s - loss: 0.1946 - categorical_accuracy: 0.9408
52160/60000 [=========================>....] - ETA: 14s - loss: 0.1945 - categorical_accuracy: 0.9408
52192/60000 [=========================>....] - ETA: 14s - loss: 0.1945 - categorical_accuracy: 0.9408
52224/60000 [=========================>....] - ETA: 14s - loss: 0.1944 - categorical_accuracy: 0.9408
52256/60000 [=========================>....] - ETA: 14s - loss: 0.1943 - categorical_accuracy: 0.9408
52288/60000 [=========================>....] - ETA: 14s - loss: 0.1942 - categorical_accuracy: 0.9409
52320/60000 [=========================>....] - ETA: 14s - loss: 0.1941 - categorical_accuracy: 0.9409
52352/60000 [=========================>....] - ETA: 14s - loss: 0.1940 - categorical_accuracy: 0.9409
52384/60000 [=========================>....] - ETA: 14s - loss: 0.1939 - categorical_accuracy: 0.9409
52416/60000 [=========================>....] - ETA: 14s - loss: 0.1939 - categorical_accuracy: 0.9409
52448/60000 [=========================>....] - ETA: 14s - loss: 0.1938 - categorical_accuracy: 0.9410
52480/60000 [=========================>....] - ETA: 14s - loss: 0.1938 - categorical_accuracy: 0.9409
52512/60000 [=========================>....] - ETA: 14s - loss: 0.1939 - categorical_accuracy: 0.9409
52544/60000 [=========================>....] - ETA: 14s - loss: 0.1938 - categorical_accuracy: 0.9410
52576/60000 [=========================>....] - ETA: 14s - loss: 0.1938 - categorical_accuracy: 0.9410
52608/60000 [=========================>....] - ETA: 13s - loss: 0.1936 - categorical_accuracy: 0.9410
52640/60000 [=========================>....] - ETA: 13s - loss: 0.1937 - categorical_accuracy: 0.9410
52672/60000 [=========================>....] - ETA: 13s - loss: 0.1938 - categorical_accuracy: 0.9410
52704/60000 [=========================>....] - ETA: 13s - loss: 0.1937 - categorical_accuracy: 0.9410
52736/60000 [=========================>....] - ETA: 13s - loss: 0.1936 - categorical_accuracy: 0.9411
52768/60000 [=========================>....] - ETA: 13s - loss: 0.1936 - categorical_accuracy: 0.9411
52800/60000 [=========================>....] - ETA: 13s - loss: 0.1935 - categorical_accuracy: 0.9411
52832/60000 [=========================>....] - ETA: 13s - loss: 0.1934 - categorical_accuracy: 0.9411
52864/60000 [=========================>....] - ETA: 13s - loss: 0.1933 - categorical_accuracy: 0.9412
52896/60000 [=========================>....] - ETA: 13s - loss: 0.1932 - categorical_accuracy: 0.9412
52928/60000 [=========================>....] - ETA: 13s - loss: 0.1932 - categorical_accuracy: 0.9412
52960/60000 [=========================>....] - ETA: 13s - loss: 0.1931 - categorical_accuracy: 0.9412
52992/60000 [=========================>....] - ETA: 13s - loss: 0.1931 - categorical_accuracy: 0.9412
53024/60000 [=========================>....] - ETA: 13s - loss: 0.1930 - categorical_accuracy: 0.9413
53056/60000 [=========================>....] - ETA: 13s - loss: 0.1929 - categorical_accuracy: 0.9413
53088/60000 [=========================>....] - ETA: 13s - loss: 0.1929 - categorical_accuracy: 0.9413
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1929 - categorical_accuracy: 0.9413
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1928 - categorical_accuracy: 0.9413
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1927 - categorical_accuracy: 0.9413
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1927 - categorical_accuracy: 0.9414
53248/60000 [=========================>....] - ETA: 12s - loss: 0.1927 - categorical_accuracy: 0.9413
53280/60000 [=========================>....] - ETA: 12s - loss: 0.1926 - categorical_accuracy: 0.9414
53312/60000 [=========================>....] - ETA: 12s - loss: 0.1925 - categorical_accuracy: 0.9414
53344/60000 [=========================>....] - ETA: 12s - loss: 0.1924 - categorical_accuracy: 0.9414
53376/60000 [=========================>....] - ETA: 12s - loss: 0.1924 - categorical_accuracy: 0.9414
53408/60000 [=========================>....] - ETA: 12s - loss: 0.1923 - categorical_accuracy: 0.9415
53440/60000 [=========================>....] - ETA: 12s - loss: 0.1922 - categorical_accuracy: 0.9415
53472/60000 [=========================>....] - ETA: 12s - loss: 0.1922 - categorical_accuracy: 0.9415
53504/60000 [=========================>....] - ETA: 12s - loss: 0.1921 - categorical_accuracy: 0.9415
53536/60000 [=========================>....] - ETA: 12s - loss: 0.1920 - categorical_accuracy: 0.9416
53568/60000 [=========================>....] - ETA: 12s - loss: 0.1919 - categorical_accuracy: 0.9416
53600/60000 [=========================>....] - ETA: 12s - loss: 0.1919 - categorical_accuracy: 0.9416
53632/60000 [=========================>....] - ETA: 12s - loss: 0.1918 - categorical_accuracy: 0.9416
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1920 - categorical_accuracy: 0.9416
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1921 - categorical_accuracy: 0.9416
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1921 - categorical_accuracy: 0.9415
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1921 - categorical_accuracy: 0.9415
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1920 - categorical_accuracy: 0.9415
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1919 - categorical_accuracy: 0.9416
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1919 - categorical_accuracy: 0.9415
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1919 - categorical_accuracy: 0.9415
53920/60000 [=========================>....] - ETA: 11s - loss: 0.1918 - categorical_accuracy: 0.9416
53952/60000 [=========================>....] - ETA: 11s - loss: 0.1917 - categorical_accuracy: 0.9416
53984/60000 [=========================>....] - ETA: 11s - loss: 0.1916 - categorical_accuracy: 0.9416
54016/60000 [==========================>...] - ETA: 11s - loss: 0.1915 - categorical_accuracy: 0.9417
54048/60000 [==========================>...] - ETA: 11s - loss: 0.1914 - categorical_accuracy: 0.9417
54080/60000 [==========================>...] - ETA: 11s - loss: 0.1913 - categorical_accuracy: 0.9417
54112/60000 [==========================>...] - ETA: 11s - loss: 0.1913 - categorical_accuracy: 0.9418
54144/60000 [==========================>...] - ETA: 11s - loss: 0.1914 - categorical_accuracy: 0.9417
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1914 - categorical_accuracy: 0.9417
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1915 - categorical_accuracy: 0.9417
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1914 - categorical_accuracy: 0.9417
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1914 - categorical_accuracy: 0.9418
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1913 - categorical_accuracy: 0.9418
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1912 - categorical_accuracy: 0.9418
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1911 - categorical_accuracy: 0.9418
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1911 - categorical_accuracy: 0.9419
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1910 - categorical_accuracy: 0.9419
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1909 - categorical_accuracy: 0.9419
54496/60000 [==========================>...] - ETA: 10s - loss: 0.1909 - categorical_accuracy: 0.9419
54528/60000 [==========================>...] - ETA: 10s - loss: 0.1908 - categorical_accuracy: 0.9420
54560/60000 [==========================>...] - ETA: 10s - loss: 0.1907 - categorical_accuracy: 0.9420
54592/60000 [==========================>...] - ETA: 10s - loss: 0.1907 - categorical_accuracy: 0.9420
54624/60000 [==========================>...] - ETA: 10s - loss: 0.1907 - categorical_accuracy: 0.9420
54656/60000 [==========================>...] - ETA: 10s - loss: 0.1906 - categorical_accuracy: 0.9420
54688/60000 [==========================>...] - ETA: 10s - loss: 0.1906 - categorical_accuracy: 0.9420
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1906 - categorical_accuracy: 0.9420 
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1905 - categorical_accuracy: 0.9420
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1906 - categorical_accuracy: 0.9420
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1905 - categorical_accuracy: 0.9420
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1904 - categorical_accuracy: 0.9420
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1904 - categorical_accuracy: 0.9420
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1903 - categorical_accuracy: 0.9421
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1902 - categorical_accuracy: 0.9421
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1901 - categorical_accuracy: 0.9421
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1900 - categorical_accuracy: 0.9421
55040/60000 [==========================>...] - ETA: 9s - loss: 0.1900 - categorical_accuracy: 0.9421
55072/60000 [==========================>...] - ETA: 9s - loss: 0.1899 - categorical_accuracy: 0.9421
55104/60000 [==========================>...] - ETA: 9s - loss: 0.1898 - categorical_accuracy: 0.9422
55136/60000 [==========================>...] - ETA: 9s - loss: 0.1897 - categorical_accuracy: 0.9422
55168/60000 [==========================>...] - ETA: 9s - loss: 0.1896 - categorical_accuracy: 0.9422
55200/60000 [==========================>...] - ETA: 9s - loss: 0.1895 - categorical_accuracy: 0.9423
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1894 - categorical_accuracy: 0.9423
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1893 - categorical_accuracy: 0.9423
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1893 - categorical_accuracy: 0.9424
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1892 - categorical_accuracy: 0.9424
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1892 - categorical_accuracy: 0.9424
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1892 - categorical_accuracy: 0.9424
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1891 - categorical_accuracy: 0.9424
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1890 - categorical_accuracy: 0.9424
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1889 - categorical_accuracy: 0.9425
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1889 - categorical_accuracy: 0.9425
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1888 - categorical_accuracy: 0.9425
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1887 - categorical_accuracy: 0.9425
55616/60000 [==========================>...] - ETA: 8s - loss: 0.1887 - categorical_accuracy: 0.9425
55648/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9425
55680/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9426
55712/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9426
55744/60000 [==========================>...] - ETA: 8s - loss: 0.1885 - categorical_accuracy: 0.9426
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1884 - categorical_accuracy: 0.9427
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1883 - categorical_accuracy: 0.9427
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1882 - categorical_accuracy: 0.9427
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1881 - categorical_accuracy: 0.9427
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1881 - categorical_accuracy: 0.9428
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1880 - categorical_accuracy: 0.9428
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1880 - categorical_accuracy: 0.9428
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1880 - categorical_accuracy: 0.9428
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1879 - categorical_accuracy: 0.9428
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1878 - categorical_accuracy: 0.9429
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1877 - categorical_accuracy: 0.9429
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1876 - categorical_accuracy: 0.9429
56160/60000 [===========================>..] - ETA: 7s - loss: 0.1875 - categorical_accuracy: 0.9429
56192/60000 [===========================>..] - ETA: 7s - loss: 0.1874 - categorical_accuracy: 0.9430
56224/60000 [===========================>..] - ETA: 7s - loss: 0.1874 - categorical_accuracy: 0.9430
56256/60000 [===========================>..] - ETA: 7s - loss: 0.1873 - categorical_accuracy: 0.9430
56288/60000 [===========================>..] - ETA: 7s - loss: 0.1872 - categorical_accuracy: 0.9430
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1872 - categorical_accuracy: 0.9430
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1871 - categorical_accuracy: 0.9430
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1871 - categorical_accuracy: 0.9430
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1870 - categorical_accuracy: 0.9430
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9431
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9431
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9430
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9431
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9431
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1867 - categorical_accuracy: 0.9431
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1867 - categorical_accuracy: 0.9431
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1866 - categorical_accuracy: 0.9431
56704/60000 [===========================>..] - ETA: 6s - loss: 0.1865 - categorical_accuracy: 0.9432
56736/60000 [===========================>..] - ETA: 6s - loss: 0.1865 - categorical_accuracy: 0.9432
56768/60000 [===========================>..] - ETA: 6s - loss: 0.1864 - categorical_accuracy: 0.9432
56800/60000 [===========================>..] - ETA: 6s - loss: 0.1864 - categorical_accuracy: 0.9432
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1864 - categorical_accuracy: 0.9432
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1864 - categorical_accuracy: 0.9432
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9432
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9432
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9433
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9433
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9433
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1862 - categorical_accuracy: 0.9433
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1861 - categorical_accuracy: 0.9433
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9433
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1861 - categorical_accuracy: 0.9434
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9434
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9434
57248/60000 [===========================>..] - ETA: 5s - loss: 0.1858 - categorical_accuracy: 0.9435
57280/60000 [===========================>..] - ETA: 5s - loss: 0.1858 - categorical_accuracy: 0.9435
57312/60000 [===========================>..] - ETA: 5s - loss: 0.1857 - categorical_accuracy: 0.9435
57344/60000 [===========================>..] - ETA: 5s - loss: 0.1856 - categorical_accuracy: 0.9435
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1857 - categorical_accuracy: 0.9435
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1857 - categorical_accuracy: 0.9435
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1857 - categorical_accuracy: 0.9435
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1856 - categorical_accuracy: 0.9435
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1855 - categorical_accuracy: 0.9436
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1854 - categorical_accuracy: 0.9436
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1854 - categorical_accuracy: 0.9436
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1854 - categorical_accuracy: 0.9436
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1853 - categorical_accuracy: 0.9436
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1852 - categorical_accuracy: 0.9436
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1852 - categorical_accuracy: 0.9436
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1851 - categorical_accuracy: 0.9436
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9436
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9437
57824/60000 [===========================>..] - ETA: 4s - loss: 0.1849 - categorical_accuracy: 0.9437
57856/60000 [===========================>..] - ETA: 4s - loss: 0.1849 - categorical_accuracy: 0.9437
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1849 - categorical_accuracy: 0.9437
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1848 - categorical_accuracy: 0.9437
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9437
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9438
58016/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9438
58048/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9438
58080/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9438
58112/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9438
58144/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9438
58176/60000 [============================>.] - ETA: 3s - loss: 0.1844 - categorical_accuracy: 0.9438
58208/60000 [============================>.] - ETA: 3s - loss: 0.1844 - categorical_accuracy: 0.9438
58240/60000 [============================>.] - ETA: 3s - loss: 0.1843 - categorical_accuracy: 0.9438
58272/60000 [============================>.] - ETA: 3s - loss: 0.1843 - categorical_accuracy: 0.9438
58304/60000 [============================>.] - ETA: 3s - loss: 0.1842 - categorical_accuracy: 0.9438
58336/60000 [============================>.] - ETA: 3s - loss: 0.1841 - categorical_accuracy: 0.9439
58368/60000 [============================>.] - ETA: 3s - loss: 0.1840 - categorical_accuracy: 0.9439
58400/60000 [============================>.] - ETA: 3s - loss: 0.1840 - categorical_accuracy: 0.9439
58432/60000 [============================>.] - ETA: 2s - loss: 0.1840 - categorical_accuracy: 0.9439
58464/60000 [============================>.] - ETA: 2s - loss: 0.1839 - categorical_accuracy: 0.9439
58496/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9440
58528/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9440
58560/60000 [============================>.] - ETA: 2s - loss: 0.1837 - categorical_accuracy: 0.9440
58592/60000 [============================>.] - ETA: 2s - loss: 0.1837 - categorical_accuracy: 0.9440
58624/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9441
58656/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9440
58688/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9440
58720/60000 [============================>.] - ETA: 2s - loss: 0.1835 - categorical_accuracy: 0.9441
58752/60000 [============================>.] - ETA: 2s - loss: 0.1835 - categorical_accuracy: 0.9441
58784/60000 [============================>.] - ETA: 2s - loss: 0.1834 - categorical_accuracy: 0.9441
58816/60000 [============================>.] - ETA: 2s - loss: 0.1833 - categorical_accuracy: 0.9441
58848/60000 [============================>.] - ETA: 2s - loss: 0.1832 - categorical_accuracy: 0.9442
58880/60000 [============================>.] - ETA: 2s - loss: 0.1831 - categorical_accuracy: 0.9442
58912/60000 [============================>.] - ETA: 2s - loss: 0.1832 - categorical_accuracy: 0.9442
58944/60000 [============================>.] - ETA: 1s - loss: 0.1831 - categorical_accuracy: 0.9442
58976/60000 [============================>.] - ETA: 1s - loss: 0.1830 - categorical_accuracy: 0.9442
59008/60000 [============================>.] - ETA: 1s - loss: 0.1830 - categorical_accuracy: 0.9443
59040/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9443
59072/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9443
59104/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9443
59136/60000 [============================>.] - ETA: 1s - loss: 0.1827 - categorical_accuracy: 0.9443
59168/60000 [============================>.] - ETA: 1s - loss: 0.1827 - categorical_accuracy: 0.9444
59200/60000 [============================>.] - ETA: 1s - loss: 0.1826 - categorical_accuracy: 0.9444
59232/60000 [============================>.] - ETA: 1s - loss: 0.1825 - categorical_accuracy: 0.9444
59264/60000 [============================>.] - ETA: 1s - loss: 0.1825 - categorical_accuracy: 0.9444
59296/60000 [============================>.] - ETA: 1s - loss: 0.1824 - categorical_accuracy: 0.9445
59328/60000 [============================>.] - ETA: 1s - loss: 0.1823 - categorical_accuracy: 0.9445
59360/60000 [============================>.] - ETA: 1s - loss: 0.1822 - categorical_accuracy: 0.9445
59392/60000 [============================>.] - ETA: 1s - loss: 0.1822 - categorical_accuracy: 0.9445
59424/60000 [============================>.] - ETA: 1s - loss: 0.1821 - categorical_accuracy: 0.9445
59456/60000 [============================>.] - ETA: 1s - loss: 0.1821 - categorical_accuracy: 0.9445
59488/60000 [============================>.] - ETA: 0s - loss: 0.1821 - categorical_accuracy: 0.9446
59520/60000 [============================>.] - ETA: 0s - loss: 0.1821 - categorical_accuracy: 0.9446
59552/60000 [============================>.] - ETA: 0s - loss: 0.1821 - categorical_accuracy: 0.9446
59584/60000 [============================>.] - ETA: 0s - loss: 0.1820 - categorical_accuracy: 0.9446
59616/60000 [============================>.] - ETA: 0s - loss: 0.1819 - categorical_accuracy: 0.9446
59648/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9447
59680/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9447
59712/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9447
59744/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9447
59776/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9447
59808/60000 [============================>.] - ETA: 0s - loss: 0.1816 - categorical_accuracy: 0.9447
59840/60000 [============================>.] - ETA: 0s - loss: 0.1816 - categorical_accuracy: 0.9447
59872/60000 [============================>.] - ETA: 0s - loss: 0.1816 - categorical_accuracy: 0.9447
59904/60000 [============================>.] - ETA: 0s - loss: 0.1815 - categorical_accuracy: 0.9448
59936/60000 [============================>.] - ETA: 0s - loss: 0.1814 - categorical_accuracy: 0.9448
59968/60000 [============================>.] - ETA: 0s - loss: 0.1814 - categorical_accuracy: 0.9448
60000/60000 [==============================] - 117s 2ms/step - loss: 0.1813 - categorical_accuracy: 0.9448 - val_loss: 0.0451 - val_categorical_accuracy: 0.9840

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 18s
  192/10000 [..............................] - ETA: 6s 
  320/10000 [..............................] - ETA: 5s
  448/10000 [>.............................] - ETA: 4s
  576/10000 [>.............................] - ETA: 4s
  704/10000 [=>............................] - ETA: 4s
  832/10000 [=>............................] - ETA: 4s
  960/10000 [=>............................] - ETA: 4s
 1120/10000 [==>...........................] - ETA: 3s
 1280/10000 [==>...........................] - ETA: 3s
 1440/10000 [===>..........................] - ETA: 3s
 1600/10000 [===>..........................] - ETA: 3s
 1760/10000 [====>.........................] - ETA: 3s
 1920/10000 [====>.........................] - ETA: 3s
 2048/10000 [=====>........................] - ETA: 3s
 2176/10000 [=====>........................] - ETA: 3s
 2336/10000 [======>.......................] - ETA: 3s
 2464/10000 [======>.......................] - ETA: 3s
 2624/10000 [======>.......................] - ETA: 3s
 2752/10000 [=======>......................] - ETA: 3s
 2880/10000 [=======>......................] - ETA: 2s
 3040/10000 [========>.....................] - ETA: 2s
 3168/10000 [========>.....................] - ETA: 2s
 3296/10000 [========>.....................] - ETA: 2s
 3456/10000 [=========>....................] - ETA: 2s
 3616/10000 [=========>....................] - ETA: 2s
 3776/10000 [==========>...................] - ETA: 2s
 3872/10000 [==========>...................] - ETA: 2s
 4032/10000 [===========>..................] - ETA: 2s
 4192/10000 [===========>..................] - ETA: 2s
 4352/10000 [============>.................] - ETA: 2s
 4480/10000 [============>.................] - ETA: 2s
 4608/10000 [============>.................] - ETA: 2s
 4736/10000 [=============>................] - ETA: 2s
 4896/10000 [=============>................] - ETA: 2s
 5056/10000 [==============>...............] - ETA: 2s
 5216/10000 [==============>...............] - ETA: 1s
 5376/10000 [===============>..............] - ETA: 1s
 5536/10000 [===============>..............] - ETA: 1s
 5632/10000 [===============>..............] - ETA: 1s
 5728/10000 [================>.............] - ETA: 1s
 5888/10000 [================>.............] - ETA: 1s
 6048/10000 [=================>............] - ETA: 1s
 6208/10000 [=================>............] - ETA: 1s
 6368/10000 [==================>...........] - ETA: 1s
 6528/10000 [==================>...........] - ETA: 1s
 6688/10000 [===================>..........] - ETA: 1s
 6848/10000 [===================>..........] - ETA: 1s
 7008/10000 [====================>.........] - ETA: 1s
 7136/10000 [====================>.........] - ETA: 1s
 7296/10000 [====================>.........] - ETA: 1s
 7456/10000 [=====================>........] - ETA: 1s
 7616/10000 [=====================>........] - ETA: 0s
 7744/10000 [======================>.......] - ETA: 0s
 7872/10000 [======================>.......] - ETA: 0s
 8032/10000 [=======================>......] - ETA: 0s
 8160/10000 [=======================>......] - ETA: 0s
 8320/10000 [=======================>......] - ETA: 0s
 8480/10000 [========================>.....] - ETA: 0s
 8640/10000 [========================>.....] - ETA: 0s
 8800/10000 [=========================>....] - ETA: 0s
 8960/10000 [=========================>....] - ETA: 0s
 9120/10000 [==========================>...] - ETA: 0s
 9280/10000 [==========================>...] - ETA: 0s
 9440/10000 [===========================>..] - ETA: 0s
 9600/10000 [===========================>..] - ETA: 0s
 9728/10000 [============================>.] - ETA: 0s
 9888/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 395us/step
[[2.33945627e-08 7.58841256e-09 5.39024086e-07 ... 9.99996662e-01
  3.85155374e-08 2.04047910e-06]
 [4.85883311e-06 1.90489154e-05 9.99945998e-01 ... 1.88289722e-08
  1.07388305e-05 1.35998635e-09]
 [7.75802960e-07 9.99767482e-01 2.98326704e-05 ... 9.02835600e-05
  2.50269331e-05 1.23506481e-06]
 ...
 [8.10757328e-09 8.29275848e-07 2.79526517e-07 ... 1.10353849e-05
  2.29537568e-06 4.71149178e-05]
 [1.61112928e-06 1.17722813e-07 9.92367344e-09 ... 2.36560055e-07
  3.13678570e-03 3.66890185e-06]
 [3.82697226e-06 4.75555225e-07 2.76082810e-05 ... 2.55633537e-09
  8.97130121e-07 4.08059897e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04513382142044138, 'accuracy_test:': 0.984000027179718}

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
Updating 66f9634..625ab75
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
{'loss': 0.5942324548959732, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-18 12:31:56.660110: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 66f9634..625ab75
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
Updating 66f9634..625ab75
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
 40%|████      | 2/5 [00:22<00:33, 11.05s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.8981232947272143, 'learning_rate': 0.16577061152000203, 'min_data_in_leaf': 29, 'num_leaves': 50} and reward: 0.3916
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\xbdm\x10T"\tX\r\x00\x00\x00learning_rateq\x02G?\xc57\xf8\xad\x8e\xe3\xe3X\x10\x00\x00\x00min_data_in_leafq\x03K\x1dX\n\x00\x00\x00num_leavesq\x04K2u.' and reward: 0.3916
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\xbdm\x10T"\tX\r\x00\x00\x00learning_rateq\x02G?\xc57\xf8\xad\x8e\xe3\xe3X\x10\x00\x00\x00min_data_in_leafq\x03K\x1dX\n\x00\x00\x00num_leavesq\x04K2u.' and reward: 0.3916
 60%|██████    | 3/5 [00:48<00:31, 15.69s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9499235422361646, 'learning_rate': 0.010993003826653048, 'min_data_in_leaf': 29, 'num_leaves': 46} and reward: 0.3868
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeee\xc6\x0esZ\x89X\r\x00\x00\x00learning_rateq\x02G?\x86\x83\x7f\xff\\\xba\x8dX\x10\x00\x00\x00min_data_in_leafq\x03K\x1dX\n\x00\x00\x00num_leavesq\x04K.u.' and reward: 0.3868
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeee\xc6\x0esZ\x89X\r\x00\x00\x00learning_rateq\x02G?\x86\x83\x7f\xff\\\xba\x8dX\x10\x00\x00\x00min_data_in_leafq\x03K\x1dX\n\x00\x00\x00num_leavesq\x04K.u.' and reward: 0.3868
 80%|████████  | 4/5 [01:15<00:19, 19.19s/it] 80%|████████  | 4/5 [01:15<00:18, 18.99s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8865756215434635, 'learning_rate': 0.13992442271841185, 'min_data_in_leaf': 12, 'num_leaves': 52} and reward: 0.3864
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec^\xd3\xd6~\xb8\xb8X\r\x00\x00\x00learning_rateq\x02G?\xc1\xe9\x0b!\xbe^{X\x10\x00\x00\x00min_data_in_leafq\x03K\x0cX\n\x00\x00\x00num_leavesq\x04K4u.' and reward: 0.3864
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec^\xd3\xd6~\xb8\xb8X\r\x00\x00\x00learning_rateq\x02G?\xc1\xe9\x0b!\xbe^{X\x10\x00\x00\x00min_data_in_leafq\x03K\x0cX\n\x00\x00\x00num_leavesq\x04K4u.' and reward: 0.3864
Time for Gradient Boosting hyperparameter optimization: 103.00411796569824
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.8981232947272143, 'learning_rate': 0.16577061152000203, 'min_data_in_leaf': 29, 'num_leaves': 50}
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
 40%|████      | 2/5 [00:55<01:22, 27.51s/it] 40%|████      | 2/5 [00:55<01:22, 27.52s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.07132096049285337, 'embedding_size_factor': 0.7176238146847305, 'layers.choice': 1, 'learning_rate': 0.003954994340582744, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 0.00036261503912519283} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb2B\x17(\xd6\x0b\x85X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6\xf6\xc67\xdc\xda\x94X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?p3\x1c\xb5\x94$\xdeX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?7\xc3\xab\xbb\xec\xb2\x80u.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb2B\x17(\xd6\x0b\x85X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6\xf6\xc67\xdc\xda\x94X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?p3\x1c\xb5\x94$\xdeX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?7\xc3\xab\xbb\xec\xb2\x80u.' and reward: 0.3862
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 126.2308030128479
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -112.76s of remaining time.
Ensemble size: 23
Ensemble weights: 
[0.08695652 0.13043478 0.2173913  0.26086957 0.         0.30434783]
	0.3998	 = Validation accuracy score
	1.48s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 234.29s ...
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
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f5fbf670240>

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
Updating 66f9634..625ab75
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
Updating 66f9634..625ab75
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
100%|██████████| 10/10 [00:02<00:00,  3.50it/s, avg_epoch_loss=5.23]
INFO:root:Epoch[0] Elapsed time 2.861 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.225587
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.225586605072022 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4d2bb64400>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4d2bb64400>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 88.36it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1090.52001953125,
    "abs_error": 375.5166015625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.4881541592270353,
    "sMAPE": 0.517936846896718,
    "MSIS": 99.52617283973555,
    "QuantileLoss[0.5]": 375.5166244506836,
    "Coverage[0.5]": 1.0,
    "RMSE": 33.02302256806984,
    "NRMSE": 0.6952215277488387,
    "ND": 0.658801055372807,
    "wQuantileLoss[0.5]": 0.6588010955275151,
    "mean_wQuantileLoss": 0.6588010955275151,
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
100%|██████████| 10/10 [00:01<00:00,  7.66it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.306 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4cff64ad30>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4cff64ad30>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 151.48it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.74it/s, avg_epoch_loss=5.29]
INFO:root:Epoch[0] Elapsed time 1.742 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.293743
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.293742847442627 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4cff570a20>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4cff570a20>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 149.46it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 253.9515584309896,
    "abs_error": 175.0937957763672,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.1601627049325194,
    "sMAPE": 0.29019965548320387,
    "MSIS": 46.40650577080548,
    "QuantileLoss[0.5]": 175.0937957763672,
    "Coverage[0.5]": 0.75,
    "RMSE": 15.935857630858454,
    "NRMSE": 0.33549173959702006,
    "ND": 0.30718209785327577,
    "wQuantileLoss[0.5]": 0.30718209785327577,
    "mean_wQuantileLoss": 0.30718209785327577,
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
 30%|███       | 3/10 [00:12<00:29,  4.24s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:23<00:16,  4.08s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:35<00:03,  3.99s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:39<00:00,  3.90s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 39.016 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.866646
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.8666455268859865 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4cff5f56d8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4cff5f56d8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 139.04it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 52723.322916666664,
    "abs_error": 2690.677001953125,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 17.828290801763472,
    "sMAPE": 1.4078451027033985,
    "MSIS": 713.1316191292308,
    "QuantileLoss[0.5]": 2690.677017211914,
    "Coverage[0.5]": 1.0,
    "RMSE": 229.6155981562809,
    "NRMSE": 4.834012592763808,
    "ND": 4.720485968338815,
    "wQuantileLoss[0.5]": 4.720485995108621,
    "mean_wQuantileLoss": 4.720485995108621,
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
100%|██████████| 10/10 [00:00<00:00, 50.72it/s, avg_epoch_loss=5.19]
INFO:root:Epoch[0] Elapsed time 0.198 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.188961
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.188961029052734 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4cfc381eb8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4cfc381eb8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 146.46it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 493.45654296875,
    "abs_error": 187.95460510253906,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.2453777821961927,
    "sMAPE": 0.3178055843659526,
    "MSIS": 49.815116949670056,
    "QuantileLoss[0.5]": 187.9546241760254,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 22.213881762734534,
    "NRMSE": 0.4676606686891481,
    "ND": 0.3297449212325247,
    "wQuantileLoss[0.5]": 0.3297449546947814,
    "mean_wQuantileLoss": 0.3297449546947814,
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
100%|██████████| 10/10 [00:01<00:00,  8.44it/s, avg_epoch_loss=161]
INFO:root:Epoch[0] Elapsed time 1.186 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=161.108291
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 161.1082910709855 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4cfc55bf28>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4cfc55bf28>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 155.11it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 439.8828247741723,
    "abs_error": 223.62751574072416,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.4817446980843618,
    "sMAPE": 0.42600252840216285,
    "MSIS": 59.26978792337448,
    "QuantileLoss[0.5]": 223.62751574072416,
    "Coverage[0.5]": 0.3333333333333333,
    "RMSE": 20.973383722570194,
    "NRMSE": 0.441544920475162,
    "ND": 0.3923289749837266,
    "wQuantileLoss[0.5]": 0.3923289749837266,
    "mean_wQuantileLoss": 0.3923289749837266,
    "MAE_Coverage": 0.16666666666666669
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
 10%|█         | 1/10 [02:02<18:23, 122.58s/it, avg_epoch_loss=0.582] 20%|██        | 2/10 [05:01<18:35, 139.47s/it, avg_epoch_loss=0.565] 30%|███       | 3/10 [08:30<18:42, 160.29s/it, avg_epoch_loss=0.548] 40%|████      | 4/10 [11:53<17:18, 173.15s/it, avg_epoch_loss=0.531] 50%|█████     | 5/10 [15:42<15:49, 189.98s/it, avg_epoch_loss=0.515] 60%|██████    | 6/10 [19:16<13:09, 197.26s/it, avg_epoch_loss=0.499] 70%|███████   | 7/10 [22:45<10:02, 200.71s/it, avg_epoch_loss=0.484] 80%|████████  | 8/10 [26:34<06:58, 209.18s/it, avg_epoch_loss=0.47]  90%|█████████ | 9/10 [30:11<03:31, 211.56s/it, avg_epoch_loss=0.457]100%|██████████| 10/10 [34:03<00:00, 217.65s/it, avg_epoch_loss=0.447]100%|██████████| 10/10 [34:03<00:00, 204.37s/it, avg_epoch_loss=0.447]
INFO:root:Epoch[0] Elapsed time 2043.695 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.446552
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.4465524971485138 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4cfc49cf98>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f4cfc49cf98>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 15.04it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 164.05573527018228,
    "abs_error": 114.22518920898438,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 0.7568503720907435,
    "sMAPE": 0.20283677998139085,
    "MSIS": 30.274016501293268,
    "QuantileLoss[0.5]": 114.22519302368164,
    "Coverage[0.5]": 0.4166666666666667,
    "RMSE": 12.808424386714483,
    "NRMSE": 0.2696510397203049,
    "ND": 0.2003950687876919,
    "wQuantileLoss[0.5]": 0.20039507548014324,
    "mean_wQuantileLoss": 0.20039507548014324,
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
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
error: Your local changes to the following files would be overwritten by merge:
	deps.txt
Please commit your changes or stash them before you merge.
Aborting
Updating 66f9634..625ab75
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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f9bdd9744a8> 

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
From github.com:arita37/mlmodels_store
   625ab75..029ac41  master     -> origin/master
Updating 66f9634..029ac41
Fast-forward
 .../20200518/list_log_dataloader_20200518.md       |  26 +-
 log_dataloader/log_dataloader.py                   |  46 +-
 ...-11_73f54da32a5da4768415eb9105ad096255137679.py | 627 ++++++++++++++++++++
 ...-09_73f54da32a5da4768415eb9105ad096255137679.py | 636 +++++++++++++++++++++
 4 files changed, 1300 insertions(+), 35 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-18-12-11_73f54da32a5da4768415eb9105ad096255137679.py
 create mode 100644 log_pullrequest/log_pr_2020-05-18-13-09_73f54da32a5da4768415eb9105ad096255137679.py
[master d8d22d9] ml_store
 2 files changed, 2843 insertions(+), 4789 deletions(-)
To github.com:arita37/mlmodels_store.git
   029ac41..d8d22d9  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
   1.40395436e-01  1.91979229e+00]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 7.88018455e-01  3.01960045e-01  7.00982122e-01 -3.94689681e-01
  -1.20376927e+00 -1.17181338e+00  7.55392029e-01  9.84012237e-01
  -5.59681422e-01 -1.98937450e-01]
 [ 1.66752297e+00  1.22372221e+00 -4.59930104e-01 -5.93679025e-02
  -4.93856997e-01  1.44898940e+00 -1.18110317e+00 -4.77580855e-01
   2.59999942e-02 -7.90799954e-01]
 [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 1.22867367e+00  1.34373116e-01 -1.82420406e-01 -2.68371304e-01
  -1.73963799e+00 -1.31675626e-01 -9.26871939e-01  1.01855247e+00
   1.23055820e+00 -4.91125138e-01]
 [ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 8.57296491e-01  9.56121704e-01 -8.26097432e-01 -7.05840507e-01
   1.13872896e+00  1.19268607e+00  2.82675712e-01 -2.37941936e-01
   1.15528789e+00  6.21082701e-01]
 [ 1.02242019e+00  1.85300949e+00  6.44353666e-01  1.42251373e-01
   1.15080755e+00  5.13505480e-01 -4.59942831e-01  3.72456852e-01
  -1.48489803e-01  3.71670291e-01]
 [ 1.64661853e+00 -1.52568032e+00 -6.06998398e-01  7.95026094e-01
   1.08480038e+00 -3.74438319e-01  4.29526140e-01  1.34048197e-01
   1.20205486e+00  1.06222724e-01]
 [ 3.54133613e-01  2.11124755e-01  9.21450069e-01  1.65275673e-02
   9.03945451e-01  1.77187720e-01  9.54250872e-02 -1.11647002e+00
   8.09271010e-02  6.07501958e-02]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 6.67591795e-01 -4.52524973e-01 -6.05981321e-01  1.16128569e+00
  -1.44620987e+00  1.06996554e+00  1.92381543e+00 -1.04553425e+00
   3.55284507e-01  1.80358898e+00]
 [ 1.16755486e+00  3.53600971e-02  7.14789597e-01 -1.53879325e+00
   1.10863359e+00 -4.47895185e-01 -1.75592564e+00  6.17985534e-01
  -1.84176326e-01  8.52704062e-01]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 1.83829400e+00  5.02740882e-01  1.29101580e-01  1.55880554e+00
   1.32551412e+00  1.09402696e-01  1.40754000e+00 -1.21974440e+00
   2.44936865e+00  1.61694960e+00]
 [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 1.06523311e+00 -6.64867767e-01  1.00806543e+00 -1.94504696e+00
  -1.23017555e+00 -9.15424368e-01  3.37220938e-01  1.22515585e+00
  -1.05354607e+00  7.85226920e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 5.69983848e-01 -5.33020326e-01 -1.75458969e-01 -1.42655542e+00
   6.06604307e-01  1.76795995e+00 -1.15985185e-01 -4.75372875e-01
   4.77610182e-01 -9.33914656e-01]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f89c93f8da0>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f89e37675f8> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.16777676 -0.66575452 -1.23312074 -1.67419581  1.01313574  0.82502982
  -0.12046457 -0.49821356 -0.31098498 -1.18231813]
 [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 0.77528533  1.47016034  1.03298378 -0.87000822  0.78655651  0.36919047
  -0.14319575  0.85328219 -0.13971173 -0.22241403]
 [ 1.09488485 -0.06962454 -0.11644415  0.35387043 -1.44189096 -0.18695502
   1.2911889  -0.15323616 -2.43250851 -2.277298  ]
 [ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
  -0.87728152 -0.78191168 -0.43750898 -1.44087602]
 [ 1.21619061 -0.01900052  0.86089124 -0.22676019 -1.36419132 -1.56450785
   1.63169151  0.93125568  0.94980882 -0.88018906]
 [ 0.87874071 -0.01923163  0.31965694  0.15001628 -1.46662161  0.46353432
  -0.89868319  0.39788042 -0.99601089  0.3181542 ]
 [ 1.18559003  0.08646441  1.23289919 -2.14246673  1.033341   -0.83016886
   0.36723181  0.45161595  1.10417433 -0.42285696]
 [ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 0.89562312 -2.29820588 -0.01952256  1.45652739 -1.85064099  0.31663724
   0.11133727 -2.66412594 -0.42642862 -0.83998891]
 [ 0.78344054 -0.05118845  0.82458463 -0.72559712  0.9317172  -0.86776868
   3.03085711 -0.13597733 -0.79726979  0.65458015]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
  -0.3169012   0.43499832  0.9104236   1.21987438]
 [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
   1.10223673  0.81434313  0.78046993  1.10861676]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]
 [ 1.03967316 -0.73153098  0.36184732 -1.56573815  0.95928819  1.01382247
  -1.78791289 -2.22711263 -1.6993336  -0.42449279]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 0.96457205 -0.10679399  1.12232832  1.45142926  1.21828168 -0.61803685
   0.43816635 -2.03720123 -1.94258918 -0.9970198 ]
 [ 0.72297801  0.18553562  0.91549927  0.39442803 -0.84983074  0.72552256
  -0.15050433  1.49588477  0.67545381 -0.43820027]
 [ 1.02242019  1.85300949  0.64435367  0.14225137  1.15080755  0.51350548
  -0.45994283  0.37245685 -0.1484898   0.37167029]]
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
[[ 1.02242019e+00  1.85300949e+00  6.44353666e-01  1.42251373e-01
   1.15080755e+00  5.13505480e-01 -4.59942831e-01  3.72456852e-01
  -1.48489803e-01  3.71670291e-01]
 [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
   8.84908814e-01  2.76499305e-01]
 [ 2.07582971e+00 -1.40232915e+00 -4.79184915e-01  4.51122939e-01
   1.03436581e+00 -6.94920901e-01 -4.18937898e-01  5.15413802e-01
  -1.11487105e+00 -1.95210529e+00]
 [ 7.88018455e-01  3.01960045e-01  7.00982122e-01 -3.94689681e-01
  -1.20376927e+00 -1.17181338e+00  7.55392029e-01  9.84012237e-01
  -5.59681422e-01 -1.98937450e-01]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]
 [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
  -2.43250851e+00 -2.27729800e+00]
 [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 8.72267394e-01 -2.51630386e+00 -7.75070287e-01 -5.95667881e-01
   1.02600767e+00 -3.09121319e-01  1.74643509e+00  5.10937774e-01
   1.71066184e+00  1.41640538e-01]
 [ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 6.92114488e-01 -6.06524918e-02  2.05635552e+00 -2.41350300e+00
   1.17456965e+00 -1.77756638e+00 -2.81736269e-01 -7.77858827e-01
   1.11584111e+00  1.76024923e+00]
 [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
  -3.53409983e-01 -2.51674208e-01]
 [ 8.57719529e-01  9.81122462e-02 -2.60466059e-01  1.06032751e+00
  -1.39003042e+00 -1.71116766e+00  2.65642403e-01  1.65712464e+00
   1.41767401e+00  4.45096710e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 1.44682180e+00  8.07455917e-01  1.49810818e+00  3.12238689e-01
  -6.82430193e-01 -1.93321640e-01  2.88078167e-01 -2.07680202e+00
   9.47501167e-01 -3.00976154e-01]
 [ 1.17867274e+00 -5.99804531e-01 -6.94693595e-01  1.12341216e+00
   1.17899425e+00  3.05267040e-01  1.33526763e-02  1.38877940e+00
  -6.61344243e-01  6.21803504e-01]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 1.64661853e+00 -1.52568032e+00 -6.06998398e-01  7.95026094e-01
   1.08480038e+00 -3.74438319e-01  4.29526140e-01  1.34048197e-01
   1.20205486e+00  1.06222724e-01]
 [ 8.88838813e-01  1.03368687e+00 -4.97025792e-02  8.08844360e-01
   8.14051347e-01  1.78975468e+00  1.14690038e+00  4.51284016e-01
  -1.68405999e+00  4.66643267e-01]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 1.02817479e+00 -5.08457134e-01  1.76533510e+00  7.77419205e-01
   6.17714185e-01 -1.18771172e-01  4.50155513e-01 -1.98998184e-01
   1.86647138e+00  8.70969803e-01]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
   1.10417433e+00 -4.22856961e-01]
 [ 8.61462558e-01  7.43205537e-02 -1.34501002e+00 -1.99560718e-01
  -1.47533915e+00 -6.54603169e-01 -3.14563862e-01  3.18014296e-01
  -8.90271552e-01 -1.29525789e+00]
 [ 1.18468624e+00 -1.00016919e+00 -5.93843067e-01  1.04499441e+00
   9.65482331e-01  6.08514698e-01 -6.25342001e-01 -6.93286967e-02
  -1.08392067e-01 -3.43900709e-01]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]]
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
[master 331e034] ml_store
 1 file changed, 297 insertions(+)
To github.com:arita37/mlmodels_store.git
   d8d22d9..331e034  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140356014038992
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140356014038768
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140356014037536
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140356014037088
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140356014036584
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140356014036248

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
grad_step = 000000, loss = 0.309066
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.187076
grad_step = 000002, loss = 0.096540
grad_step = 000003, loss = 0.031406
grad_step = 000004, loss = 0.027639
grad_step = 000005, loss = 0.049247
grad_step = 000006, loss = 0.032756
grad_step = 000007, loss = 0.014925
grad_step = 000008, loss = 0.012688
grad_step = 000009, loss = 0.017061
grad_step = 000010, loss = 0.019885
grad_step = 000011, loss = 0.018916
grad_step = 000012, loss = 0.015291
grad_step = 000013, loss = 0.011321
grad_step = 000014, loss = 0.009054
grad_step = 000015, loss = 0.009128
grad_step = 000016, loss = 0.010359
grad_step = 000017, loss = 0.010928
grad_step = 000018, loss = 0.010154
grad_step = 000019, loss = 0.008784
grad_step = 000020, loss = 0.007888
grad_step = 000021, loss = 0.007811
grad_step = 000022, loss = 0.008194
grad_step = 000023, loss = 0.008448
grad_step = 000024, loss = 0.008215
grad_step = 000025, loss = 0.007565
grad_step = 000026, loss = 0.006862
grad_step = 000027, loss = 0.006538
grad_step = 000028, loss = 0.006729
grad_step = 000029, loss = 0.007118
grad_step = 000030, loss = 0.007232
grad_step = 000031, loss = 0.006898
grad_step = 000032, loss = 0.006400
grad_step = 000033, loss = 0.006085
grad_step = 000034, loss = 0.006091
grad_step = 000035, loss = 0.006282
grad_step = 000036, loss = 0.006410
grad_step = 000037, loss = 0.006326
grad_step = 000038, loss = 0.006065
grad_step = 000039, loss = 0.005789
grad_step = 000040, loss = 0.005666
grad_step = 000041, loss = 0.005733
grad_step = 000042, loss = 0.005862
grad_step = 000043, loss = 0.005882
grad_step = 000044, loss = 0.005742
grad_step = 000045, loss = 0.005541
grad_step = 000046, loss = 0.005414
grad_step = 000047, loss = 0.005402
grad_step = 000048, loss = 0.005440
grad_step = 000049, loss = 0.005434
grad_step = 000050, loss = 0.005349
grad_step = 000051, loss = 0.005228
grad_step = 000052, loss = 0.005138
grad_step = 000053, loss = 0.005105
grad_step = 000054, loss = 0.005091
grad_step = 000055, loss = 0.005041
grad_step = 000056, loss = 0.004948
grad_step = 000057, loss = 0.004859
grad_step = 000058, loss = 0.004801
grad_step = 000059, loss = 0.004765
grad_step = 000060, loss = 0.004717
grad_step = 000061, loss = 0.004638
grad_step = 000062, loss = 0.004553
grad_step = 000063, loss = 0.004482
grad_step = 000064, loss = 0.004411
grad_step = 000065, loss = 0.004329
grad_step = 000066, loss = 0.004244
grad_step = 000067, loss = 0.004172
grad_step = 000068, loss = 0.004090
grad_step = 000069, loss = 0.003987
grad_step = 000070, loss = 0.003888
grad_step = 000071, loss = 0.003804
grad_step = 000072, loss = 0.003704
grad_step = 000073, loss = 0.003584
grad_step = 000074, loss = 0.003483
grad_step = 000075, loss = 0.003386
grad_step = 000076, loss = 0.003265
grad_step = 000077, loss = 0.003151
grad_step = 000078, loss = 0.003042
grad_step = 000079, loss = 0.002922
grad_step = 000080, loss = 0.002811
grad_step = 000081, loss = 0.002695
grad_step = 000082, loss = 0.002587
grad_step = 000083, loss = 0.002483
grad_step = 000084, loss = 0.002376
grad_step = 000085, loss = 0.002290
grad_step = 000086, loss = 0.002195
grad_step = 000087, loss = 0.002123
grad_step = 000088, loss = 0.002053
grad_step = 000089, loss = 0.001995
grad_step = 000090, loss = 0.001948
grad_step = 000091, loss = 0.001900
grad_step = 000092, loss = 0.001872
grad_step = 000093, loss = 0.001836
grad_step = 000094, loss = 0.001807
grad_step = 000095, loss = 0.001782
grad_step = 000096, loss = 0.001756
grad_step = 000097, loss = 0.001721
grad_step = 000098, loss = 0.001691
grad_step = 000099, loss = 0.001659
grad_step = 000100, loss = 0.001621
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.001586
grad_step = 000102, loss = 0.001550
grad_step = 000103, loss = 0.001519
grad_step = 000104, loss = 0.001490
grad_step = 000105, loss = 0.001460
grad_step = 000106, loss = 0.001433
grad_step = 000107, loss = 0.001408
grad_step = 000108, loss = 0.001386
grad_step = 000109, loss = 0.001368
grad_step = 000110, loss = 0.001351
grad_step = 000111, loss = 0.001335
grad_step = 000112, loss = 0.001314
grad_step = 000113, loss = 0.001289
grad_step = 000114, loss = 0.001267
grad_step = 000115, loss = 0.001251
grad_step = 000116, loss = 0.001238
grad_step = 000117, loss = 0.001227
grad_step = 000118, loss = 0.001210
grad_step = 000119, loss = 0.001189
grad_step = 000120, loss = 0.001170
grad_step = 000121, loss = 0.001156
grad_step = 000122, loss = 0.001147
grad_step = 000123, loss = 0.001139
grad_step = 000124, loss = 0.001128
grad_step = 000125, loss = 0.001112
grad_step = 000126, loss = 0.001096
grad_step = 000127, loss = 0.001080
grad_step = 000128, loss = 0.001064
grad_step = 000129, loss = 0.001049
grad_step = 000130, loss = 0.001036
grad_step = 000131, loss = 0.001022
grad_step = 000132, loss = 0.001010
grad_step = 000133, loss = 0.000998
grad_step = 000134, loss = 0.000988
grad_step = 000135, loss = 0.000981
grad_step = 000136, loss = 0.000979
grad_step = 000137, loss = 0.000987
grad_step = 000138, loss = 0.000995
grad_step = 000139, loss = 0.000979
grad_step = 000140, loss = 0.000932
grad_step = 000141, loss = 0.000905
grad_step = 000142, loss = 0.000912
grad_step = 000143, loss = 0.000919
grad_step = 000144, loss = 0.000901
grad_step = 000145, loss = 0.000870
grad_step = 000146, loss = 0.000858
grad_step = 000147, loss = 0.000866
grad_step = 000148, loss = 0.000873
grad_step = 000149, loss = 0.000856
grad_step = 000150, loss = 0.000829
grad_step = 000151, loss = 0.000814
grad_step = 000152, loss = 0.000816
grad_step = 000153, loss = 0.000819
grad_step = 000154, loss = 0.000807
grad_step = 000155, loss = 0.000787
grad_step = 000156, loss = 0.000776
grad_step = 000157, loss = 0.000773
grad_step = 000158, loss = 0.000771
grad_step = 000159, loss = 0.000767
grad_step = 000160, loss = 0.000757
grad_step = 000161, loss = 0.000744
grad_step = 000162, loss = 0.000735
grad_step = 000163, loss = 0.000729
grad_step = 000164, loss = 0.000725
grad_step = 000165, loss = 0.000723
grad_step = 000166, loss = 0.000721
grad_step = 000167, loss = 0.000716
grad_step = 000168, loss = 0.000712
grad_step = 000169, loss = 0.000708
grad_step = 000170, loss = 0.000701
grad_step = 000171, loss = 0.000694
grad_step = 000172, loss = 0.000688
grad_step = 000173, loss = 0.000682
grad_step = 000174, loss = 0.000676
grad_step = 000175, loss = 0.000672
grad_step = 000176, loss = 0.000669
grad_step = 000177, loss = 0.000665
grad_step = 000178, loss = 0.000664
grad_step = 000179, loss = 0.000667
grad_step = 000180, loss = 0.000675
grad_step = 000181, loss = 0.000692
grad_step = 000182, loss = 0.000716
grad_step = 000183, loss = 0.000736
grad_step = 000184, loss = 0.000725
grad_step = 000185, loss = 0.000679
grad_step = 000186, loss = 0.000633
grad_step = 000187, loss = 0.000625
grad_step = 000188, loss = 0.000648
grad_step = 000189, loss = 0.000669
grad_step = 000190, loss = 0.000656
grad_step = 000191, loss = 0.000625
grad_step = 000192, loss = 0.000606
grad_step = 000193, loss = 0.000611
grad_step = 000194, loss = 0.000625
grad_step = 000195, loss = 0.000625
grad_step = 000196, loss = 0.000608
grad_step = 000197, loss = 0.000592
grad_step = 000198, loss = 0.000587
grad_step = 000199, loss = 0.000594
grad_step = 000200, loss = 0.000597
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000591
grad_step = 000202, loss = 0.000581
grad_step = 000203, loss = 0.000571
grad_step = 000204, loss = 0.000568
grad_step = 000205, loss = 0.000570
grad_step = 000206, loss = 0.000572
grad_step = 000207, loss = 0.000570
grad_step = 000208, loss = 0.000565
grad_step = 000209, loss = 0.000557
grad_step = 000210, loss = 0.000550
grad_step = 000211, loss = 0.000544
grad_step = 000212, loss = 0.000541
grad_step = 000213, loss = 0.000540
grad_step = 000214, loss = 0.000540
grad_step = 000215, loss = 0.000542
grad_step = 000216, loss = 0.000550
grad_step = 000217, loss = 0.000564
grad_step = 000218, loss = 0.000588
grad_step = 000219, loss = 0.000618
grad_step = 000220, loss = 0.000640
grad_step = 000221, loss = 0.000627
grad_step = 000222, loss = 0.000576
grad_step = 000223, loss = 0.000526
grad_step = 000224, loss = 0.000509
grad_step = 000225, loss = 0.000527
grad_step = 000226, loss = 0.000557
grad_step = 000227, loss = 0.000567
grad_step = 000228, loss = 0.000548
grad_step = 000229, loss = 0.000516
grad_step = 000230, loss = 0.000496
grad_step = 000231, loss = 0.000497
grad_step = 000232, loss = 0.000512
grad_step = 000233, loss = 0.000524
grad_step = 000234, loss = 0.000520
grad_step = 000235, loss = 0.000502
grad_step = 000236, loss = 0.000486
grad_step = 000237, loss = 0.000479
grad_step = 000238, loss = 0.000483
grad_step = 000239, loss = 0.000488
grad_step = 000240, loss = 0.000491
grad_step = 000241, loss = 0.000488
grad_step = 000242, loss = 0.000480
grad_step = 000243, loss = 0.000470
grad_step = 000244, loss = 0.000463
grad_step = 000245, loss = 0.000461
grad_step = 000246, loss = 0.000462
grad_step = 000247, loss = 0.000465
grad_step = 000248, loss = 0.000466
grad_step = 000249, loss = 0.000466
grad_step = 000250, loss = 0.000468
grad_step = 000251, loss = 0.000465
grad_step = 000252, loss = 0.000462
grad_step = 000253, loss = 0.000457
grad_step = 000254, loss = 0.000456
grad_step = 000255, loss = 0.000450
grad_step = 000256, loss = 0.000445
grad_step = 000257, loss = 0.000440
grad_step = 000258, loss = 0.000437
grad_step = 000259, loss = 0.000434
grad_step = 000260, loss = 0.000431
grad_step = 000261, loss = 0.000430
grad_step = 000262, loss = 0.000429
grad_step = 000263, loss = 0.000427
grad_step = 000264, loss = 0.000428
grad_step = 000265, loss = 0.000433
grad_step = 000266, loss = 0.000443
grad_step = 000267, loss = 0.000469
grad_step = 000268, loss = 0.000519
grad_step = 000269, loss = 0.000616
grad_step = 000270, loss = 0.000726
grad_step = 000271, loss = 0.000766
grad_step = 000272, loss = 0.000623
grad_step = 000273, loss = 0.000444
grad_step = 000274, loss = 0.000426
grad_step = 000275, loss = 0.000543
grad_step = 000276, loss = 0.000561
grad_step = 000277, loss = 0.000453
grad_step = 000278, loss = 0.000406
grad_step = 000279, loss = 0.000481
grad_step = 000280, loss = 0.000492
grad_step = 000281, loss = 0.000417
grad_step = 000282, loss = 0.000407
grad_step = 000283, loss = 0.000452
grad_step = 000284, loss = 0.000445
grad_step = 000285, loss = 0.000396
grad_step = 000286, loss = 0.000406
grad_step = 000287, loss = 0.000438
grad_step = 000288, loss = 0.000412
grad_step = 000289, loss = 0.000384
grad_step = 000290, loss = 0.000404
grad_step = 000291, loss = 0.000418
grad_step = 000292, loss = 0.000394
grad_step = 000293, loss = 0.000378
grad_step = 000294, loss = 0.000393
grad_step = 000295, loss = 0.000401
grad_step = 000296, loss = 0.000385
grad_step = 000297, loss = 0.000375
grad_step = 000298, loss = 0.000382
grad_step = 000299, loss = 0.000385
grad_step = 000300, loss = 0.000375
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000372
grad_step = 000302, loss = 0.000369
grad_step = 000303, loss = 0.000371
grad_step = 000304, loss = 0.000367
grad_step = 000305, loss = 0.000363
grad_step = 000306, loss = 0.000361
grad_step = 000307, loss = 0.000361
grad_step = 000308, loss = 0.000361
grad_step = 000309, loss = 0.000359
grad_step = 000310, loss = 0.000356
grad_step = 000311, loss = 0.000353
grad_step = 000312, loss = 0.000352
grad_step = 000313, loss = 0.000353
grad_step = 000314, loss = 0.000352
grad_step = 000315, loss = 0.000350
grad_step = 000316, loss = 0.000346
grad_step = 000317, loss = 0.000344
grad_step = 000318, loss = 0.000344
grad_step = 000319, loss = 0.000343
grad_step = 000320, loss = 0.000342
grad_step = 000321, loss = 0.000341
grad_step = 000322, loss = 0.000340
grad_step = 000323, loss = 0.000337
grad_step = 000324, loss = 0.000337
grad_step = 000325, loss = 0.000336
grad_step = 000326, loss = 0.000334
grad_step = 000327, loss = 0.000334
grad_step = 000328, loss = 0.000335
grad_step = 000329, loss = 0.000330
grad_step = 000330, loss = 0.000330
grad_step = 000331, loss = 0.000330
grad_step = 000332, loss = 0.000326
grad_step = 000333, loss = 0.000327
grad_step = 000334, loss = 0.000329
grad_step = 000335, loss = 0.000323
grad_step = 000336, loss = 0.000325
grad_step = 000337, loss = 0.000330
grad_step = 000338, loss = 0.000320
grad_step = 000339, loss = 0.000324
grad_step = 000340, loss = 0.000331
grad_step = 000341, loss = 0.000319
grad_step = 000342, loss = 0.000329
grad_step = 000343, loss = 0.000348
grad_step = 000344, loss = 0.000343
grad_step = 000345, loss = 0.000381
grad_step = 000346, loss = 0.000430
grad_step = 000347, loss = 0.000483
grad_step = 000348, loss = 0.000580
grad_step = 000349, loss = 0.000645
grad_step = 000350, loss = 0.000560
grad_step = 000351, loss = 0.000416
grad_step = 000352, loss = 0.000309
grad_step = 000353, loss = 0.000373
grad_step = 000354, loss = 0.000438
grad_step = 000355, loss = 0.000407
grad_step = 000356, loss = 0.000336
grad_step = 000357, loss = 0.000314
grad_step = 000358, loss = 0.000387
grad_step = 000359, loss = 0.000379
grad_step = 000360, loss = 0.000325
grad_step = 000361, loss = 0.000317
grad_step = 000362, loss = 0.000339
grad_step = 000363, loss = 0.000356
grad_step = 000364, loss = 0.000315
grad_step = 000365, loss = 0.000295
grad_step = 000366, loss = 0.000327
grad_step = 000367, loss = 0.000329
grad_step = 000368, loss = 0.000310
grad_step = 000369, loss = 0.000294
grad_step = 000370, loss = 0.000301
grad_step = 000371, loss = 0.000316
grad_step = 000372, loss = 0.000306
grad_step = 000373, loss = 0.000290
grad_step = 000374, loss = 0.000294
grad_step = 000375, loss = 0.000297
grad_step = 000376, loss = 0.000296
grad_step = 000377, loss = 0.000292
grad_step = 000378, loss = 0.000284
grad_step = 000379, loss = 0.000288
grad_step = 000380, loss = 0.000289
grad_step = 000381, loss = 0.000282
grad_step = 000382, loss = 0.000279
grad_step = 000383, loss = 0.000280
grad_step = 000384, loss = 0.000281
grad_step = 000385, loss = 0.000279
grad_step = 000386, loss = 0.000276
grad_step = 000387, loss = 0.000273
grad_step = 000388, loss = 0.000271
grad_step = 000389, loss = 0.000274
grad_step = 000390, loss = 0.000276
grad_step = 000391, loss = 0.000272
grad_step = 000392, loss = 0.000269
grad_step = 000393, loss = 0.000269
grad_step = 000394, loss = 0.000266
grad_step = 000395, loss = 0.000265
grad_step = 000396, loss = 0.000266
grad_step = 000397, loss = 0.000265
grad_step = 000398, loss = 0.000263
grad_step = 000399, loss = 0.000261
grad_step = 000400, loss = 0.000261
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000262
grad_step = 000402, loss = 0.000262
grad_step = 000403, loss = 0.000263
grad_step = 000404, loss = 0.000267
grad_step = 000405, loss = 0.000273
grad_step = 000406, loss = 0.000284
grad_step = 000407, loss = 0.000302
grad_step = 000408, loss = 0.000323
grad_step = 000409, loss = 0.000337
grad_step = 000410, loss = 0.000333
grad_step = 000411, loss = 0.000306
grad_step = 000412, loss = 0.000282
grad_step = 000413, loss = 0.000264
grad_step = 000414, loss = 0.000277
grad_step = 000415, loss = 0.000305
grad_step = 000416, loss = 0.000296
grad_step = 000417, loss = 0.000279
grad_step = 000418, loss = 0.000271
grad_step = 000419, loss = 0.000256
grad_step = 000420, loss = 0.000263
grad_step = 000421, loss = 0.000274
grad_step = 000422, loss = 0.000279
grad_step = 000423, loss = 0.000262
grad_step = 000424, loss = 0.000254
grad_step = 000425, loss = 0.000251
grad_step = 000426, loss = 0.000258
grad_step = 000427, loss = 0.000265
grad_step = 000428, loss = 0.000269
grad_step = 000429, loss = 0.000259
grad_step = 000430, loss = 0.000250
grad_step = 000431, loss = 0.000246
grad_step = 000432, loss = 0.000245
grad_step = 000433, loss = 0.000247
grad_step = 000434, loss = 0.000251
grad_step = 000435, loss = 0.000251
grad_step = 000436, loss = 0.000247
grad_step = 000437, loss = 0.000244
grad_step = 000438, loss = 0.000244
grad_step = 000439, loss = 0.000249
grad_step = 000440, loss = 0.000255
grad_step = 000441, loss = 0.000268
grad_step = 000442, loss = 0.000286
grad_step = 000443, loss = 0.000316
grad_step = 000444, loss = 0.000338
grad_step = 000445, loss = 0.000368
grad_step = 000446, loss = 0.000385
grad_step = 000447, loss = 0.000375
grad_step = 000448, loss = 0.000325
grad_step = 000449, loss = 0.000271
grad_step = 000450, loss = 0.000236
grad_step = 000451, loss = 0.000244
grad_step = 000452, loss = 0.000263
grad_step = 000453, loss = 0.000282
grad_step = 000454, loss = 0.000276
grad_step = 000455, loss = 0.000253
grad_step = 000456, loss = 0.000231
grad_step = 000457, loss = 0.000242
grad_step = 000458, loss = 0.000250
grad_step = 000459, loss = 0.000248
grad_step = 000460, loss = 0.000232
grad_step = 000461, loss = 0.000241
grad_step = 000462, loss = 0.000252
grad_step = 000463, loss = 0.000251
grad_step = 000464, loss = 0.000238
grad_step = 000465, loss = 0.000237
grad_step = 000466, loss = 0.000227
grad_step = 000467, loss = 0.000224
grad_step = 000468, loss = 0.000228
grad_step = 000469, loss = 0.000237
grad_step = 000470, loss = 0.000232
grad_step = 000471, loss = 0.000222
grad_step = 000472, loss = 0.000223
grad_step = 000473, loss = 0.000230
grad_step = 000474, loss = 0.000227
grad_step = 000475, loss = 0.000219
grad_step = 000476, loss = 0.000220
grad_step = 000477, loss = 0.000218
grad_step = 000478, loss = 0.000217
grad_step = 000479, loss = 0.000212
grad_step = 000480, loss = 0.000218
grad_step = 000481, loss = 0.000220
grad_step = 000482, loss = 0.000218
grad_step = 000483, loss = 0.000208
grad_step = 000484, loss = 0.000212
grad_step = 000485, loss = 0.000213
grad_step = 000486, loss = 0.000212
grad_step = 000487, loss = 0.000207
grad_step = 000488, loss = 0.000212
grad_step = 000489, loss = 0.000216
grad_step = 000490, loss = 0.000213
grad_step = 000491, loss = 0.000209
grad_step = 000492, loss = 0.000212
grad_step = 000493, loss = 0.000225
grad_step = 000494, loss = 0.000235
grad_step = 000495, loss = 0.000256
grad_step = 000496, loss = 0.000298
grad_step = 000497, loss = 0.000392
grad_step = 000498, loss = 0.000507
grad_step = 000499, loss = 0.000617
grad_step = 000500, loss = 0.000542
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000366
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
[[0.83606774 0.8326442  0.93641067 0.94138074 1.0079567 ]
 [0.8292578  0.91414845 0.9515854  1.0094116  0.97529036]
 [0.87476015 0.9029752  0.99187934 0.990751   0.9614378 ]
 [0.8975858  0.9858702  0.99468833 0.95189095 0.91506773]
 [0.97784704 0.98552626 0.9515424  0.916695   0.87355804]
 [0.95825326 0.9382203  0.9109533  0.8694092  0.84860873]
 [0.91206324 0.89820063 0.8413181  0.8709457  0.82196   ]
 [0.8702625  0.827884   0.8454218  0.81448436 0.84016657]
 [0.80809116 0.81847227 0.8220984  0.84441364 0.85098326]
 [0.8244563  0.8177085  0.82227284 0.85310036 0.8537271 ]
 [0.7894219  0.8134424  0.8566785  0.8305845  0.9259361 ]
 [0.8138478  0.8397584  0.8254297  0.927107   0.9357791 ]
 [0.82611    0.83274746 0.9365487  0.9450195  1.0070217 ]
 [0.82017285 0.92496794 0.9554617  1.0133502  0.96980596]
 [0.8933924  0.9243262  0.99700576 0.9825318  0.943378  ]
 [0.9100795  0.99207044 0.98973346 0.93707776 0.8938485 ]
 [0.9835799  0.98204386 0.9351832  0.8957345  0.8526021 ]
 [0.9581381  0.925123   0.89299715 0.8467978  0.838688  ]
 [0.9117236  0.88661206 0.834658   0.8610916  0.8214013 ]
 [0.8776459  0.8319137  0.8441464  0.81751853 0.8409551 ]
 [0.8224704  0.82658863 0.82801324 0.8473062  0.85438323]
 [0.845159   0.8289071  0.8280547  0.86366737 0.85413975]
 [0.80319357 0.8252307  0.8650945  0.8370051  0.92525095]
 [0.8225888  0.84834826 0.83425754 0.9305665  0.93853176]
 [0.84168303 0.8379519  0.9344939  0.9427123  1.009997  ]
 [0.83599555 0.9201889  0.95599    1.0153491  0.98132163]
 [0.88312054 0.9137564  1.0015962  1.0044099  0.9711752 ]
 [0.9111967  0.99779814 1.010373   0.96428514 0.9236959 ]
 [0.987742   0.9972942  0.96466285 0.925326   0.8789096 ]
 [0.96858335 0.9470998  0.92102766 0.8751673  0.8537594 ]
 [0.9217951  0.90542305 0.84913003 0.8760162  0.8289044 ]]

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
Already up to date.
[master 75e3e42] ml_store
 1 file changed, 1123 insertions(+)
To github.com:arita37/mlmodels_store.git
   331e034..75e3e42  master -> master





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
[master 0037c03] ml_store
 1 file changed, 38 insertions(+)
To github.com:arita37/mlmodels_store.git
   75e3e42..0037c03  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 18627154.38B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 519935.19B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 4089856/440473133 [00:00<00:10, 40895283.93B/s]  2%|▏         | 9005056/440473133 [00:00<00:10, 43059377.61B/s]  3%|▎         | 13298688/440473133 [00:00<00:09, 43020702.87B/s]  4%|▍         | 18395136/440473133 [00:00<00:09, 45130472.61B/s]  5%|▌         | 23386112/440473133 [00:00<00:08, 46464980.12B/s]  6%|▋         | 28432384/440473133 [00:00<00:08, 47596155.07B/s]  8%|▊         | 33512448/440473133 [00:00<00:08, 48513251.97B/s]  9%|▉         | 38737920/440473133 [00:00<00:08, 49577406.27B/s] 10%|▉         | 43719680/440473133 [00:00<00:07, 49648633.34B/s] 11%|█         | 48801792/440473133 [00:01<00:07, 49991718.53B/s] 12%|█▏        | 53873664/440473133 [00:01<00:07, 50205609.13B/s] 13%|█▎        | 58838016/440473133 [00:01<00:07, 50034510.60B/s] 14%|█▍        | 63813632/440473133 [00:01<00:07, 49947716.51B/s] 16%|█▌        | 68773888/440473133 [00:01<00:07, 49618401.41B/s] 17%|█▋        | 73873408/440473133 [00:01<00:07, 50022146.27B/s] 18%|█▊        | 78860288/440473133 [00:01<00:07, 49707939.34B/s] 19%|█▉        | 84162560/440473133 [00:01<00:07, 50656514.70B/s] 20%|██        | 89225216/440473133 [00:01<00:06, 50478821.48B/s] 21%|██▏       | 94271488/440473133 [00:01<00:06, 49671639.99B/s] 23%|██▎       | 99434496/440473133 [00:02<00:06, 50239664.27B/s] 24%|██▎       | 104461312/440473133 [00:02<00:06, 49703377.93B/s] 25%|██▍       | 109533184/440473133 [00:02<00:06, 50002959.89B/s] 26%|██▌       | 114536448/440473133 [00:02<00:06, 49988143.52B/s] 27%|██▋       | 119559168/440473133 [00:02<00:06, 50052891.72B/s] 28%|██▊       | 124566528/440473133 [00:02<00:06, 48539861.83B/s] 29%|██▉       | 129431552/440473133 [00:02<00:06, 47700961.81B/s] 31%|███       | 134368256/440473133 [00:02<00:06, 48188533.60B/s] 32%|███▏      | 139546624/440473133 [00:02<00:06, 49212724.18B/s] 33%|███▎      | 144600064/440473133 [00:02<00:05, 49600531.46B/s] 34%|███▍      | 149569536/440473133 [00:03<00:05, 49021372.27B/s] 35%|███▌      | 154612736/440473133 [00:03<00:05, 49436183.81B/s] 36%|███▋      | 159781888/440473133 [00:03<00:05, 50087589.93B/s] 37%|███▋      | 164861952/440473133 [00:03<00:05, 50290362.17B/s] 39%|███▊      | 169934848/440473133 [00:03<00:05, 50420082.11B/s] 40%|███▉      | 174981120/440473133 [00:03<00:05, 50063877.47B/s] 41%|████      | 179990528/440473133 [00:03<00:05, 50032884.97B/s] 42%|████▏     | 185153536/440473133 [00:03<00:05, 50493777.18B/s] 43%|████▎     | 190326784/440473133 [00:03<00:04, 50858878.33B/s] 44%|████▍     | 195462144/440473133 [00:03<00:04, 51005399.69B/s] 46%|████▌     | 200679424/440473133 [00:04<00:04, 51349099.53B/s] 47%|████▋     | 205816832/440473133 [00:04<00:04, 50476553.97B/s] 48%|████▊     | 210884608/440473133 [00:04<00:04, 50531608.17B/s] 49%|████▉     | 215941120/440473133 [00:04<00:04, 49032062.35B/s] 50%|█████     | 220856320/440473133 [00:04<00:04, 48619741.67B/s] 51%|█████▏    | 226128896/440473133 [00:04<00:04, 49779544.61B/s] 53%|█████▎    | 231475200/440473133 [00:04<00:04, 50828286.23B/s] 54%|█████▍    | 236962816/440473133 [00:04<00:03, 51974103.62B/s] 55%|█████▌    | 242520064/440473133 [00:04<00:03, 53001851.78B/s] 56%|█████▋    | 248028160/440473133 [00:04<00:03, 53607213.22B/s] 58%|█████▊    | 253509632/440473133 [00:05<00:03, 53963819.29B/s] 59%|█████▉    | 258916352/440473133 [00:05<00:03, 53926242.84B/s] 60%|██████    | 264315904/440473133 [00:05<00:03, 52822778.21B/s] 61%|██████    | 269608960/440473133 [00:05<00:03, 52701317.56B/s] 62%|██████▏   | 274886656/440473133 [00:05<00:03, 52421051.70B/s] 64%|██████▎   | 280139776/440473133 [00:05<00:03, 52450878.48B/s] 65%|██████▍   | 285388800/440473133 [00:05<00:02, 51741521.21B/s] 66%|██████▌   | 290568192/440473133 [00:05<00:02, 51675359.78B/s] 67%|██████▋   | 295739392/440473133 [00:05<00:02, 51039786.72B/s] 68%|██████▊   | 300848128/440473133 [00:05<00:02, 50104487.22B/s] 69%|██████▉   | 305995776/440473133 [00:06<00:02, 50508074.60B/s] 71%|███████   | 311052288/440473133 [00:06<00:02, 50077634.82B/s] 72%|███████▏  | 316300288/440473133 [00:06<00:02, 50774837.21B/s] 73%|███████▎  | 321551360/440473133 [00:06<00:02, 51280686.72B/s] 74%|███████▍  | 326684672/440473133 [00:06<00:02, 51077049.35B/s] 75%|███████▌  | 331841536/440473133 [00:06<00:02, 51222389.63B/s] 77%|███████▋  | 336966656/440473133 [00:06<00:02, 50871400.12B/s] 78%|███████▊  | 342402048/440473133 [00:06<00:01, 51867403.56B/s] 79%|███████▉  | 347952128/440473133 [00:06<00:01, 52904645.43B/s] 80%|████████  | 353252352/440473133 [00:06<00:01, 52867472.52B/s] 81%|████████▏ | 358657024/440473133 [00:07<00:01, 53215672.93B/s] 83%|████████▎ | 363983872/440473133 [00:07<00:01, 52908074.35B/s] 84%|████████▍ | 369278976/440473133 [00:07<00:01, 52229946.89B/s] 85%|████████▌ | 374507520/440473133 [00:07<00:01, 51917092.45B/s] 86%|████████▌ | 379703296/440473133 [00:07<00:01, 51532803.32B/s] 87%|████████▋ | 384860160/440473133 [00:07<00:01, 49539503.64B/s] 89%|████████▊ | 389833728/440473133 [00:07<00:01, 46524238.27B/s] 90%|████████▉ | 394814464/440473133 [00:07<00:00, 47461045.29B/s] 91%|█████████ | 399742976/440473133 [00:07<00:00, 47992373.66B/s] 92%|█████████▏| 404630528/440473133 [00:08<00:00, 48250486.38B/s] 93%|█████████▎| 409585664/440473133 [00:08<00:00, 48631177.67B/s] 94%|█████████▍| 414465024/440473133 [00:08<00:00, 48496493.99B/s] 95%|█████████▌| 419693568/440473133 [00:08<00:00, 49572871.39B/s] 96%|█████████▋| 424998912/440473133 [00:08<00:00, 50567830.84B/s] 98%|█████████▊| 430464000/440473133 [00:08<00:00, 51725584.37B/s] 99%|█████████▉| 435977216/440473133 [00:08<00:00, 52702461.69B/s]100%|██████████| 440473133/440473133 [00:08<00:00, 50494352.33B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
1064960/7094233 [===>..........................] - ETA: 0s
2105344/7094233 [=======>......................] - ETA: 0s
3145728/7094233 [============>.................] - ETA: 0s
4186112/7094233 [================>.............] - ETA: 0s
5226496/7094233 [=====================>........] - ETA: 0s
5750784/7094233 [=======================>......] - ETA: 0s
6791168/7094233 [===========================>..] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   5%|▌         | 108/2118 [00:00<00:01, 1079.47it/s]Processing text_left with encode:  26%|██▋       | 556/2118 [00:00<00:01, 1397.47it/s]Processing text_left with encode:  45%|████▍     | 943/2118 [00:00<00:00, 1705.96it/s]Processing text_left with encode:  66%|██████▌   | 1398/2118 [00:00<00:00, 2099.66it/s]Processing text_left with encode:  87%|████████▋ | 1851/2118 [00:00<00:00, 2502.27it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3736.42it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 150/18841 [00:00<00:14, 1330.43it/s]Processing text_right with encode:   2%|▏         | 318/18841 [00:00<00:13, 1416.49it/s]Processing text_right with encode:   3%|▎         | 495/18841 [00:00<00:12, 1505.89it/s]Processing text_right with encode:   4%|▎         | 676/18841 [00:00<00:11, 1583.87it/s]Processing text_right with encode:   4%|▍         | 840/18841 [00:00<00:11, 1599.79it/s]Processing text_right with encode:   5%|▌         | 987/18841 [00:00<00:11, 1555.88it/s]Processing text_right with encode:   6%|▌         | 1135/18841 [00:00<00:11, 1529.48it/s]Processing text_right with encode:   7%|▋         | 1303/18841 [00:00<00:11, 1571.71it/s]Processing text_right with encode:   8%|▊         | 1460/18841 [00:00<00:11, 1568.25it/s]Processing text_right with encode:   9%|▊         | 1627/18841 [00:01<00:10, 1594.64it/s]Processing text_right with encode:  10%|▉         | 1798/18841 [00:01<00:10, 1622.49it/s]Processing text_right with encode:  10%|█         | 1959/18841 [00:01<00:10, 1594.46it/s]Processing text_right with encode:  11%|█         | 2118/18841 [00:01<00:10, 1579.40it/s]Processing text_right with encode:  12%|█▏        | 2293/18841 [00:01<00:10, 1625.46it/s]Processing text_right with encode:  13%|█▎        | 2469/18841 [00:01<00:09, 1660.10it/s]Processing text_right with encode:  14%|█▍        | 2635/18841 [00:01<00:10, 1613.87it/s]Processing text_right with encode:  15%|█▌        | 2828/18841 [00:01<00:09, 1695.55it/s]Processing text_right with encode:  16%|█▌        | 2999/18841 [00:01<00:09, 1688.90it/s]Processing text_right with encode:  17%|█▋        | 3177/18841 [00:01<00:09, 1712.07it/s]Processing text_right with encode:  18%|█▊        | 3359/18841 [00:02<00:08, 1731.41it/s]Processing text_right with encode:  19%|█▉        | 3534/18841 [00:02<00:08, 1735.79it/s]Processing text_right with encode:  20%|█▉        | 3708/18841 [00:02<00:09, 1653.48it/s]Processing text_right with encode:  21%|██        | 3875/18841 [00:02<00:09, 1628.71it/s]Processing text_right with encode:  22%|██▏       | 4055/18841 [00:02<00:08, 1673.24it/s]Processing text_right with encode:  22%|██▏       | 4224/18841 [00:02<00:08, 1664.44it/s]Processing text_right with encode:  23%|██▎       | 4408/18841 [00:02<00:08, 1713.30it/s]Processing text_right with encode:  24%|██▍       | 4581/18841 [00:02<00:08, 1653.91it/s]Processing text_right with encode:  25%|██▌       | 4748/18841 [00:02<00:08, 1602.14it/s]Processing text_right with encode:  26%|██▌       | 4918/18841 [00:02<00:08, 1629.20it/s]Processing text_right with encode:  27%|██▋       | 5084/18841 [00:03<00:08, 1634.24it/s]Processing text_right with encode:  28%|██▊       | 5251/18841 [00:03<00:08, 1644.40it/s]Processing text_right with encode:  29%|██▊       | 5416/18841 [00:03<00:08, 1617.55it/s]Processing text_right with encode:  30%|██▉       | 5586/18841 [00:03<00:08, 1637.67it/s]Processing text_right with encode:  31%|███       | 5751/18841 [00:03<00:08, 1629.86it/s]Processing text_right with encode:  31%|███▏      | 5915/18841 [00:03<00:08, 1600.10it/s]Processing text_right with encode:  32%|███▏      | 6076/18841 [00:03<00:08, 1575.92it/s]Processing text_right with encode:  33%|███▎      | 6239/18841 [00:03<00:07, 1589.32it/s]Processing text_right with encode:  34%|███▍      | 6409/18841 [00:03<00:07, 1617.58it/s]Processing text_right with encode:  35%|███▍      | 6579/18841 [00:04<00:07, 1640.43it/s]Processing text_right with encode:  36%|███▌      | 6744/18841 [00:04<00:07, 1621.80it/s]Processing text_right with encode:  37%|███▋      | 6913/18841 [00:04<00:07, 1641.05it/s]Processing text_right with encode:  38%|███▊      | 7082/18841 [00:04<00:07, 1653.66it/s]Processing text_right with encode:  38%|███▊      | 7248/18841 [00:04<00:07, 1649.94it/s]Processing text_right with encode:  39%|███▉      | 7414/18841 [00:04<00:06, 1633.94it/s]Processing text_right with encode:  40%|████      | 7578/18841 [00:04<00:07, 1580.56it/s]Processing text_right with encode:  41%|████      | 7753/18841 [00:04<00:06, 1627.53it/s]Processing text_right with encode:  42%|████▏     | 7920/18841 [00:04<00:06, 1639.29it/s]Processing text_right with encode:  43%|████▎     | 8085/18841 [00:04<00:06, 1634.86it/s]Processing text_right with encode:  44%|████▍     | 8254/18841 [00:05<00:06, 1648.73it/s]Processing text_right with encode:  45%|████▍     | 8420/18841 [00:05<00:06, 1632.14it/s]Processing text_right with encode:  46%|████▌     | 8584/18841 [00:05<00:06, 1600.26it/s]Processing text_right with encode:  46%|████▋     | 8754/18841 [00:05<00:06, 1623.67it/s]Processing text_right with encode:  47%|████▋     | 8933/18841 [00:05<00:05, 1667.19it/s]Processing text_right with encode:  48%|████▊     | 9101/18841 [00:05<00:05, 1655.59it/s]Processing text_right with encode:  49%|████▉     | 9273/18841 [00:05<00:05, 1672.81it/s]Processing text_right with encode:  50%|█████     | 9441/18841 [00:05<00:05, 1664.15it/s]Processing text_right with encode:  51%|█████     | 9618/18841 [00:05<00:05, 1690.21it/s]Processing text_right with encode:  52%|█████▏    | 9788/18841 [00:05<00:05, 1667.65it/s]Processing text_right with encode:  53%|█████▎    | 9975/18841 [00:06<00:05, 1723.03it/s]Processing text_right with encode:  54%|█████▍    | 10148/18841 [00:06<00:05, 1713.02it/s]Processing text_right with encode:  55%|█████▍    | 10320/18841 [00:06<00:04, 1713.86it/s]Processing text_right with encode:  56%|█████▌    | 10526/18841 [00:06<00:04, 1802.58it/s]Processing text_right with encode:  57%|█████▋    | 10708/18841 [00:06<00:04, 1796.71it/s]Processing text_right with encode:  58%|█████▊    | 10889/18841 [00:06<00:04, 1742.63it/s]Processing text_right with encode:  59%|█████▊    | 11065/18841 [00:06<00:04, 1724.76it/s]Processing text_right with encode:  60%|█████▉    | 11240/18841 [00:06<00:04, 1730.57it/s]Processing text_right with encode:  61%|██████    | 11414/18841 [00:06<00:04, 1696.78it/s]Processing text_right with encode:  61%|██████▏   | 11585/18841 [00:06<00:04, 1683.13it/s]Processing text_right with encode:  62%|██████▏   | 11754/18841 [00:07<00:04, 1685.12it/s]Processing text_right with encode:  63%|██████▎   | 11923/18841 [00:07<00:04, 1649.45it/s]Processing text_right with encode:  64%|██████▍   | 12094/18841 [00:07<00:04, 1659.67it/s]Processing text_right with encode:  65%|██████▌   | 12261/18841 [00:07<00:03, 1654.73it/s]Processing text_right with encode:  66%|██████▌   | 12427/18841 [00:07<00:03, 1649.63it/s]Processing text_right with encode:  67%|██████▋   | 12593/18841 [00:07<00:04, 1556.11it/s]Processing text_right with encode:  68%|██████▊   | 12751/18841 [00:07<00:03, 1562.37it/s]Processing text_right with encode:  69%|██████▊   | 12911/18841 [00:07<00:03, 1572.48it/s]Processing text_right with encode:  69%|██████▉   | 13081/18841 [00:07<00:03, 1608.28it/s]Processing text_right with encode:  70%|███████   | 13243/18841 [00:08<00:03, 1611.59it/s]Processing text_right with encode:  71%|███████   | 13418/18841 [00:08<00:03, 1647.84it/s]Processing text_right with encode:  72%|███████▏  | 13584/18841 [00:08<00:03, 1650.74it/s]Processing text_right with encode:  73%|███████▎  | 13763/18841 [00:08<00:03, 1689.81it/s]Processing text_right with encode:  74%|███████▍  | 13933/18841 [00:08<00:02, 1680.26it/s]Processing text_right with encode:  75%|███████▍  | 14102/18841 [00:08<00:02, 1675.55it/s]Processing text_right with encode:  76%|███████▌  | 14271/18841 [00:08<00:02, 1676.81it/s]Processing text_right with encode:  77%|███████▋  | 14439/18841 [00:08<00:02, 1665.44it/s]Processing text_right with encode:  78%|███████▊  | 14619/18841 [00:08<00:02, 1697.82it/s]Processing text_right with encode:  79%|███████▊  | 14801/18841 [00:08<00:02, 1731.19it/s]Processing text_right with encode:  79%|███████▉  | 14975/18841 [00:09<00:02, 1727.38it/s]Processing text_right with encode:  80%|████████  | 15148/18841 [00:09<00:02, 1707.22it/s]Processing text_right with encode:  81%|████████▏ | 15319/18841 [00:09<00:02, 1669.70it/s]Processing text_right with encode:  82%|████████▏ | 15496/18841 [00:09<00:01, 1697.60it/s]Processing text_right with encode:  83%|████████▎ | 15682/18841 [00:09<00:01, 1743.01it/s]Processing text_right with encode:  84%|████████▍ | 15859/18841 [00:09<00:01, 1750.38it/s]Processing text_right with encode:  85%|████████▌ | 16035/18841 [00:09<00:01, 1736.21it/s]Processing text_right with encode:  86%|████████▌ | 16209/18841 [00:09<00:01, 1703.33it/s]Processing text_right with encode:  87%|████████▋ | 16380/18841 [00:09<00:01, 1682.14it/s]Processing text_right with encode:  88%|████████▊ | 16549/18841 [00:09<00:01, 1663.15it/s]Processing text_right with encode:  89%|████████▊ | 16720/18841 [00:10<00:01, 1676.00it/s]Processing text_right with encode:  90%|████████▉ | 16889/18841 [00:10<00:01, 1678.71it/s]Processing text_right with encode:  91%|█████████ | 17058/18841 [00:10<00:01, 1663.44it/s]Processing text_right with encode:  91%|█████████▏| 17227/18841 [00:10<00:00, 1670.69it/s]Processing text_right with encode:  92%|█████████▏| 17399/18841 [00:10<00:00, 1683.55it/s]Processing text_right with encode:  93%|█████████▎| 17574/18841 [00:10<00:00, 1701.59it/s]Processing text_right with encode:  94%|█████████▍| 17751/18841 [00:10<00:00, 1720.84it/s]Processing text_right with encode:  95%|█████████▌| 17924/18841 [00:10<00:00, 1672.19it/s]Processing text_right with encode:  96%|█████████▌| 18119/18841 [00:10<00:00, 1744.79it/s]Processing text_right with encode:  97%|█████████▋| 18295/18841 [00:10<00:00, 1719.69it/s]Processing text_right with encode:  98%|█████████▊| 18490/18841 [00:11<00:00, 1782.62it/s]Processing text_right with encode:  99%|█████████▉| 18670/18841 [00:11<00:00, 1775.27it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:11<00:00, 1667.85it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 645605.80it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 793526.08it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  69%|██████▉   | 436/633 [00:00<00:00, 4353.39it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4315.41it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 173/5961 [00:00<00:03, 1729.26it/s]Processing text_right with encode:   6%|▌         | 344/5961 [00:00<00:03, 1723.01it/s]Processing text_right with encode:   9%|▊         | 508/5961 [00:00<00:03, 1693.70it/s]Processing text_right with encode:  11%|█▏        | 681/5961 [00:00<00:03, 1702.29it/s]Processing text_right with encode:  14%|█▍        | 848/5961 [00:00<00:03, 1692.01it/s]Processing text_right with encode:  17%|█▋        | 1029/5961 [00:00<00:02, 1725.07it/s]Processing text_right with encode:  20%|██        | 1197/5961 [00:00<00:02, 1710.76it/s]Processing text_right with encode:  23%|██▎       | 1373/5961 [00:00<00:02, 1724.65it/s]Processing text_right with encode:  26%|██▌       | 1536/5961 [00:00<00:02, 1673.80it/s]Processing text_right with encode:  28%|██▊       | 1697/5961 [00:01<00:02, 1652.28it/s]Processing text_right with encode:  31%|███       | 1858/5961 [00:01<00:02, 1607.51it/s]Processing text_right with encode:  34%|███▍      | 2042/5961 [00:01<00:02, 1668.06it/s]Processing text_right with encode:  37%|███▋      | 2232/5961 [00:01<00:02, 1728.89it/s]Processing text_right with encode:  40%|████      | 2405/5961 [00:01<00:02, 1658.94it/s]Processing text_right with encode:  43%|████▎     | 2577/5961 [00:01<00:02, 1675.71it/s]Processing text_right with encode:  46%|████▋     | 2759/5961 [00:01<00:01, 1715.84it/s]Processing text_right with encode:  49%|████▉     | 2939/5961 [00:01<00:01, 1739.51it/s]Processing text_right with encode:  52%|█████▏    | 3121/5961 [00:01<00:01, 1761.00it/s]Processing text_right with encode:  55%|█████▌    | 3298/5961 [00:01<00:01, 1731.60it/s]Processing text_right with encode:  58%|█████▊    | 3472/5961 [00:02<00:01, 1669.37it/s]Processing text_right with encode:  61%|██████▏   | 3655/5961 [00:02<00:01, 1712.49it/s]Processing text_right with encode:  64%|██████▍   | 3828/5961 [00:02<00:01, 1685.56it/s]Processing text_right with encode:  67%|██████▋   | 4013/5961 [00:02<00:01, 1730.93it/s]Processing text_right with encode:  70%|███████   | 4187/5961 [00:02<00:01, 1723.67it/s]Processing text_right with encode:  73%|███████▎  | 4360/5961 [00:02<00:00, 1694.51it/s]Processing text_right with encode:  76%|███████▌  | 4536/5961 [00:02<00:00, 1712.49it/s]Processing text_right with encode:  79%|███████▉  | 4711/5961 [00:02<00:00, 1721.82it/s]Processing text_right with encode:  82%|████████▏ | 4884/5961 [00:02<00:00, 1673.95it/s]Processing text_right with encode:  85%|████████▍ | 5052/5961 [00:02<00:00, 1628.73it/s]Processing text_right with encode:  88%|████████▊ | 5228/5961 [00:03<00:00, 1664.22it/s]Processing text_right with encode:  91%|█████████ | 5396/5961 [00:03<00:00, 1610.56it/s]Processing text_right with encode:  93%|█████████▎| 5558/5961 [00:03<00:00, 1584.96it/s]Processing text_right with encode:  96%|█████████▌| 5718/5961 [00:03<00:00, 1585.24it/s]Processing text_right with encode:  99%|█████████▉| 5899/5961 [00:03<00:00, 1645.56it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1686.96it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 507356.09it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 716376.21it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:25<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:25<?, ?it/s, loss=0.917]Epoch 1/1:   1%|          | 1/102 [00:25<43:27, 25.82s/it, loss=0.917]Epoch 1/1:   1%|          | 1/102 [01:59<43:27, 25.82s/it, loss=0.917]Epoch 1/1:   1%|          | 1/102 [01:59<43:27, 25.82s/it, loss=0.871]Epoch 1/1:   2%|▏         | 2/102 [01:59<1:17:03, 46.23s/it, loss=0.871]Epoch 1/1:   2%|▏         | 2/102 [03:49<1:17:03, 46.23s/it, loss=0.871]Epoch 1/1:   2%|▏         | 2/102 [03:49<1:17:03, 46.23s/it, loss=1.000]Epoch 1/1:   3%|▎         | 3/102 [03:49<1:47:45, 65.31s/it, loss=1.000]Epoch 1/1:   3%|▎         | 3/102 [05:47<1:47:45, 65.31s/it, loss=1.000]Epoch 1/1:   3%|▎         | 3/102 [05:47<1:47:45, 65.31s/it, loss=0.985]Epoch 1/1:   4%|▍         | 4/102 [05:47<2:12:26, 81.08s/it, loss=0.985]Epoch 1/1:   4%|▍         | 4/102 [06:55<2:12:26, 81.08s/it, loss=0.985]Epoch 1/1:   4%|▍         | 4/102 [06:55<2:12:26, 81.08s/it, loss=0.944]Epoch 1/1:   5%|▍         | 5/102 [06:55<2:04:47, 77.19s/it, loss=0.944]Epoch 1/1:   5%|▍         | 5/102 [10:02<2:04:47, 77.19s/it, loss=0.944]Epoch 1/1:   5%|▍         | 5/102 [10:02<2:04:47, 77.19s/it, loss=0.933]Epoch 1/1:   6%|▌         | 6/102 [10:02<2:56:08, 110.09s/it, loss=0.933]Epoch 1/1:   6%|▌         | 6/102 [11:41<2:56:08, 110.09s/it, loss=0.933]Epoch 1/1:   6%|▌         | 6/102 [11:41<2:56:08, 110.09s/it, loss=0.755]Epoch 1/1:   7%|▋         | 7/102 [11:41<2:49:15, 106.90s/it, loss=0.755]Epoch 1/1:   7%|▋         | 7/102 [14:14<2:49:15, 106.90s/it, loss=0.755]Epoch 1/1:   7%|▋         | 7/102 [14:14<2:49:15, 106.90s/it, loss=1.043]Epoch 1/1:   8%|▊         | 8/102 [14:14<3:08:46, 120.49s/it, loss=1.043]Epoch 1/1:   8%|▊         | 8/102 [15:06<3:08:46, 120.49s/it, loss=1.043]Epoch 1/1:   8%|▊         | 8/102 [15:06<3:08:46, 120.49s/it, loss=0.730]Epoch 1/1:   9%|▉         | 9/102 [15:06<2:35:15, 100.16s/it, loss=0.730]Epoch 1/1:   9%|▉         | 9/102 [16:22<2:35:15, 100.16s/it, loss=0.730]Epoch 1/1:   9%|▉         | 9/102 [16:22<2:35:15, 100.16s/it, loss=0.744]Epoch 1/1:  10%|▉         | 10/102 [16:22<2:22:09, 92.71s/it, loss=0.744]Killed

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
   0037c03..bfc2b71  master     -> origin/master
Updating 0037c03..bfc2b71
Fast-forward
 .../20200518/list_log_pullrequest_20200518.md      |   2 +-
 error_list/20200518/list_log_testall_20200518.md   | 411 ++++++++++-----------
 2 files changed, 194 insertions(+), 219 deletions(-)
