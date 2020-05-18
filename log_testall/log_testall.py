
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
   d655652..e2bc220  master     -> origin/master
Updating d655652..e2bc220
Fast-forward
 ...-10_73f54da32a5da4768415eb9105ad096255137679.py | 627 +++++++++++++++++++++
 1 file changed, 627 insertions(+)
 create mode 100644 log_pullrequest/log_pr_2020-05-18-20-10_73f54da32a5da4768415eb9105ad096255137679.py
[master 30b274f] ml_store
 2 files changed, 70 insertions(+), 10231 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   e2bc220..30b274f  master -> master





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
[master 50a717a] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   30b274f..50a717a  master -> master





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
[master f299572] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   50a717a..f299572  master -> master





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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-18 20:15:49.774068: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-18 20:15:49.793978: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-18 20:15:49.794268: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55808dfe7420 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 20:15:49.794289: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2501 - binary_crossentropy: 0.6932 - val_loss: 0.2501 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.24992936132331067}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
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
Total params: 463
Trainable params: 463
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2655 - binary_crossentropy: 0.7246500/500 [==============================] - 1s 2ms/sample - loss: 0.2622 - binary_crossentropy: 0.7180 - val_loss: 0.2585 - val_binary_crossentropy: 0.7106

  #### metrics   #################################################### 
{'MSE': 0.2600581702978955}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
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
Total params: 463
Trainable params: 463
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
Total params: 622
Trainable params: 622
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 3ms/sample - loss: 0.2502 - binary_crossentropy: 0.6935 - val_loss: 0.2500 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.2498792410435055}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
Total params: 622
Trainable params: 622
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 468
Trainable params: 468
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2769 - binary_crossentropy: 0.8814500/500 [==============================] - 2s 3ms/sample - loss: 0.2643 - binary_crossentropy: 0.7489 - val_loss: 0.2657 - val_binary_crossentropy: 0.8563

  #### metrics   #################################################### 
{'MSE': 0.26156922255850146}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 468
Trainable params: 468
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
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
100/500 [=====>........................] - ETA: 3s - loss: 0.2520 - binary_crossentropy: 0.6973500/500 [==============================] - 2s 4ms/sample - loss: 0.2533 - binary_crossentropy: 0.6999 - val_loss: 0.2513 - val_binary_crossentropy: 0.6959

  #### metrics   #################################################### 
{'MSE': 0.252625486214919}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-18 20:17:24.428030: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:17:24.430656: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:17:24.437646: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-18 20:17:24.448900: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-18 20:17:24.451182: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 20:17:24.453362: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:17:24.454994: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2502 - val_binary_crossentropy: 0.6936
2020-05-18 20:17:26.002774: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:17:26.004842: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:17:26.009859: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-18 20:17:26.019075: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-18 20:17:26.020791: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 20:17:26.022274: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:17:26.023678: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2502318005986191}

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
2020-05-18 20:17:53.285006: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:17:53.286583: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:17:53.291162: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-18 20:17:53.298299: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-18 20:17:53.299559: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 20:17:53.300689: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:17:53.301702: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2505 - val_binary_crossentropy: 0.6941
2020-05-18 20:17:55.065490: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:17:55.066779: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:17:55.069865: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-18 20:17:55.075471: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-18 20:17:55.076491: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 20:17:55.077295: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:17:55.078062: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25057932318515136}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-18 20:18:33.966323: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:18:33.971823: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:18:33.989278: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-18 20:18:34.019419: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-18 20:18:34.025212: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 20:18:34.030045: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:18:34.034647: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 6s 6s/sample - loss: 0.3866 - binary_crossentropy: 0.9722 - val_loss: 0.2565 - val_binary_crossentropy: 0.7062
2020-05-18 20:18:36.648416: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:18:36.653638: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:18:36.667184: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-18 20:18:36.693719: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-18 20:18:36.698510: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-18 20:18:36.702761: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-18 20:18:36.706515: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.22994278527497394}

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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 650
Trainable params: 650
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.4000 - binary_crossentropy: 6.1700500/500 [==============================] - 5s 10ms/sample - loss: 0.4720 - binary_crossentropy: 7.2806 - val_loss: 0.4680 - val_binary_crossentropy: 7.2189

  #### metrics   #################################################### 
{'MSE': 0.47}

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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 650
Trainable params: 650
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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         10          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 2)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         12          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_5[0][0]           
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
Total params: 242
Trainable params: 242
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2857 - binary_crossentropy: 0.7755500/500 [==============================] - 5s 10ms/sample - loss: 0.2753 - binary_crossentropy: 0.7529 - val_loss: 0.2669 - val_binary_crossentropy: 0.7297

  #### metrics   #################################################### 
{'MSE': 0.2657737525999736}

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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         10          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 2)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         12          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_5[0][0]           
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
Total params: 242
Trainable params: 242
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
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
Total params: 1,909
Trainable params: 1,909
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.5100 - binary_crossentropy: 7.8667500/500 [==============================] - 5s 10ms/sample - loss: 0.4880 - binary_crossentropy: 7.5274 - val_loss: 0.4880 - val_binary_crossentropy: 7.5274

  #### metrics   #################################################### 
{'MSE': 0.488}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
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
Total params: 1,909
Trainable params: 1,909
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
regionsequence_mean (InputLayer [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         3           regionsequence_max[0][0]         
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
Total params: 184
Trainable params: 184
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 10s - loss: 0.2541 - binary_crossentropy: 0.7014500/500 [==============================] - 7s 14ms/sample - loss: 0.2557 - binary_crossentropy: 0.7046 - val_loss: 0.2543 - val_binary_crossentropy: 0.7017

  #### metrics   #################################################### 
{'MSE': 0.2546642827116464}

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
regionsequence_mean (InputLayer [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         8           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         3           regionsequence_max[0][0]         
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
Total params: 184
Trainable params: 184
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
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
Total params: 1,427
Trainable params: 1,427
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2620 - binary_crossentropy: 0.7182500/500 [==============================] - 6s 13ms/sample - loss: 0.2608 - binary_crossentropy: 0.7163 - val_loss: 0.2604 - val_binary_crossentropy: 0.7144

  #### metrics   #################################################### 
{'MSE': 0.2569554781913338}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
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
Total params: 1,427
Trainable params: 1,427
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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         4           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         12          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         20          sequence_mean[0][0]              
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_10[0][0]                    
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
Total params: 2,967
Trainable params: 2,887
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2547 - binary_crossentropy: 0.7031500/500 [==============================] - 7s 14ms/sample - loss: 0.2595 - binary_crossentropy: 0.7131 - val_loss: 0.2610 - val_binary_crossentropy: 0.7156

  #### metrics   #################################################### 
{'MSE': 0.2581752996369687}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         4           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         12          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         12          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         20          sequence_mean[0][0]              
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_10[0][0]                    
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
Total params: 2,967
Trainable params: 2,887
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
   f299572..c451f57  master     -> origin/master
Updating f299572..c451f57
Fast-forward
 error_list/20200518/list_log_testall_20200518.md | 781 +----------------------
 1 file changed, 2 insertions(+), 779 deletions(-)
[master 8725766] ml_store
 1 file changed, 4953 insertions(+)
To github.com:arita37/mlmodels_store.git
   c451f57..8725766  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
Already up to date.
[master f4c74cf] ml_store
 1 file changed, 52 insertions(+)
To github.com:arita37/mlmodels_store.git
   8725766..f4c74cf  master -> master





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
[master 5db13d2] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   f4c74cf..5db13d2  master -> master





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
[master ef08a15] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   5db13d2..ef08a15  master -> master





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

2020-05-18 20:28:37.633701: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-18 20:28:37.639733: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-18 20:28:37.639949: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d707ffca50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 20:28:37.639968: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 9s - loss: 1.3880
256/354 [====================>.........] - ETA: 3s - loss: 1.3040
354/354 [==============================] - 16s 45ms/step - loss: 1.2992 - val_loss: 1.6036

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
[master 67d6af5] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   ef08a15..67d6af5  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
Already up to date.
[master 2ce8c12] ml_store
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   67d6af5..2ce8c12  master -> master





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
[master 833fed5] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   2ce8c12..833fed5  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2285568/17464789 [==>...........................] - ETA: 0s
10100736/17464789 [================>.............] - ETA: 0s
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
2020-05-18 20:29:44.270770: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-18 20:29:44.275549: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-18 20:29:44.275755: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d634b7dea0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 20:29:44.275771: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 14s - loss: 7.3600 - accuracy: 0.5200
 2000/25000 [=>............................] - ETA: 10s - loss: 7.4290 - accuracy: 0.5155
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.5951 - accuracy: 0.5047 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6015 - accuracy: 0.5042
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5225 - accuracy: 0.5094
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.5388 - accuracy: 0.5083
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5067 - accuracy: 0.5104
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.5325 - accuracy: 0.5088
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5576 - accuracy: 0.5071
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5762 - accuracy: 0.5059
11000/25000 [============>.................] - ETA: 4s - loss: 7.5997 - accuracy: 0.5044
12000/25000 [=============>................] - ETA: 4s - loss: 7.6449 - accuracy: 0.5014
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6501 - accuracy: 0.5011
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6305 - accuracy: 0.5024
15000/25000 [=================>............] - ETA: 3s - loss: 7.6329 - accuracy: 0.5022
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6436 - accuracy: 0.5015
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6504 - accuracy: 0.5011
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6632 - accuracy: 0.5002
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6812 - accuracy: 0.4991
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6858 - accuracy: 0.4988
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6725 - accuracy: 0.4996
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6652 - accuracy: 0.5001
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 10s 405us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f192a148cf8>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f192a1b2908> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6666 - accuracy: 0.5000
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6130 - accuracy: 0.5035
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5082 - accuracy: 0.5103 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5593 - accuracy: 0.5070
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5317 - accuracy: 0.5088
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5337 - accuracy: 0.5087
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5702 - accuracy: 0.5063
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6149 - accuracy: 0.5034
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6002 - accuracy: 0.5043
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6268 - accuracy: 0.5026
11000/25000 [============>.................] - ETA: 4s - loss: 7.6499 - accuracy: 0.5011
12000/25000 [=============>................] - ETA: 4s - loss: 7.6743 - accuracy: 0.4995
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6914 - accuracy: 0.4984
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6568 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 3s - loss: 7.6799 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6762 - accuracy: 0.4994
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6910 - accuracy: 0.4984
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6803 - accuracy: 0.4991
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6828 - accuracy: 0.4989
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6739 - accuracy: 0.4995
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6478 - accuracy: 0.5012
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6602 - accuracy: 0.5004
25000/25000 [==============================] - 10s 408us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 14s - loss: 7.6820 - accuracy: 0.4990
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6283 - accuracy: 0.5025
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.5695 - accuracy: 0.5063 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6206 - accuracy: 0.5030
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6912 - accuracy: 0.4984
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6601 - accuracy: 0.5004
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6858 - accuracy: 0.4988
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7143 - accuracy: 0.4969
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7080 - accuracy: 0.4973
11000/25000 [============>.................] - ETA: 4s - loss: 7.6889 - accuracy: 0.4985
12000/25000 [=============>................] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6643 - accuracy: 0.5002
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6622 - accuracy: 0.5003
15000/25000 [=================>............] - ETA: 3s - loss: 7.6421 - accuracy: 0.5016
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6609 - accuracy: 0.5004
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6675 - accuracy: 0.4999
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7126 - accuracy: 0.4970
19000/25000 [=====================>........] - ETA: 2s - loss: 7.7110 - accuracy: 0.4971
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7111 - accuracy: 0.4971
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7112 - accuracy: 0.4971
22000/25000 [=========================>....] - ETA: 1s - loss: 7.7043 - accuracy: 0.4975
23000/25000 [==========================>...] - ETA: 0s - loss: 7.7100 - accuracy: 0.4972
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 10s 407us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
From github.com:arita37/mlmodels_store
   833fed5..1013a65  master     -> origin/master
Updating 833fed5..1013a65
Fast-forward
 error_list/20200518/list_log_testall_20200518.md | 103 +++++++++++++++++++++++
 1 file changed, 103 insertions(+)
[master c04620b] ml_store
 1 file changed, 322 insertions(+)
To github.com:arita37/mlmodels_store.git
   1013a65..c04620b  master -> master





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

13/13 [==============================] - 2s 143ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 8/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master 4b43087] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   c04620b..4b43087  master -> master





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

    8192/11490434 [..............................] - ETA: 1s
 1835008/11490434 [===>..........................] - ETA: 0s
 7602176/11490434 [==================>...........] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:53 - loss: 2.3075 - categorical_accuracy: 0.0938
   64/60000 [..............................] - ETA: 4:58 - loss: 2.2775 - categorical_accuracy: 0.1094
   96/60000 [..............................] - ETA: 3:57 - loss: 2.2425 - categorical_accuracy: 0.1354
  128/60000 [..............................] - ETA: 3:26 - loss: 2.2182 - categorical_accuracy: 0.1172
  160/60000 [..............................] - ETA: 3:07 - loss: 2.2320 - categorical_accuracy: 0.1000
  192/60000 [..............................] - ETA: 2:54 - loss: 2.2039 - categorical_accuracy: 0.1302
  224/60000 [..............................] - ETA: 2:45 - loss: 2.1898 - categorical_accuracy: 0.1518
  256/60000 [..............................] - ETA: 2:39 - loss: 2.1542 - categorical_accuracy: 0.1758
  288/60000 [..............................] - ETA: 2:34 - loss: 2.1423 - categorical_accuracy: 0.1806
  320/60000 [..............................] - ETA: 2:30 - loss: 2.1169 - categorical_accuracy: 0.2062
  352/60000 [..............................] - ETA: 2:27 - loss: 2.0929 - categorical_accuracy: 0.2102
  384/60000 [..............................] - ETA: 2:24 - loss: 2.0533 - categorical_accuracy: 0.2370
  416/60000 [..............................] - ETA: 2:22 - loss: 2.0240 - categorical_accuracy: 0.2524
  448/60000 [..............................] - ETA: 2:20 - loss: 1.9892 - categorical_accuracy: 0.2723
  480/60000 [..............................] - ETA: 2:18 - loss: 1.9932 - categorical_accuracy: 0.2750
  512/60000 [..............................] - ETA: 2:17 - loss: 1.9708 - categorical_accuracy: 0.2793
  544/60000 [..............................] - ETA: 2:15 - loss: 1.9202 - categorical_accuracy: 0.3015
  576/60000 [..............................] - ETA: 2:14 - loss: 1.8761 - categorical_accuracy: 0.3212
  608/60000 [..............................] - ETA: 2:12 - loss: 1.8346 - categorical_accuracy: 0.3339
  640/60000 [..............................] - ETA: 2:11 - loss: 1.8055 - categorical_accuracy: 0.3500
  672/60000 [..............................] - ETA: 2:10 - loss: 1.7672 - categorical_accuracy: 0.3661
  704/60000 [..............................] - ETA: 2:09 - loss: 1.7472 - categorical_accuracy: 0.3778
  736/60000 [..............................] - ETA: 2:09 - loss: 1.7277 - categorical_accuracy: 0.3872
  768/60000 [..............................] - ETA: 2:08 - loss: 1.7120 - categorical_accuracy: 0.3945
  800/60000 [..............................] - ETA: 2:08 - loss: 1.6876 - categorical_accuracy: 0.4025
  832/60000 [..............................] - ETA: 2:07 - loss: 1.6559 - categorical_accuracy: 0.4147
  864/60000 [..............................] - ETA: 2:06 - loss: 1.6336 - categorical_accuracy: 0.4236
  896/60000 [..............................] - ETA: 2:06 - loss: 1.6092 - categorical_accuracy: 0.4308
  928/60000 [..............................] - ETA: 2:05 - loss: 1.5905 - categorical_accuracy: 0.4397
  960/60000 [..............................] - ETA: 2:04 - loss: 1.5599 - categorical_accuracy: 0.4531
  992/60000 [..............................] - ETA: 2:04 - loss: 1.5372 - categorical_accuracy: 0.4587
 1024/60000 [..............................] - ETA: 2:04 - loss: 1.5171 - categorical_accuracy: 0.4668
 1056/60000 [..............................] - ETA: 2:04 - loss: 1.4976 - categorical_accuracy: 0.4744
 1088/60000 [..............................] - ETA: 2:03 - loss: 1.4764 - categorical_accuracy: 0.4816
 1120/60000 [..............................] - ETA: 2:03 - loss: 1.4649 - categorical_accuracy: 0.4848
 1152/60000 [..............................] - ETA: 2:03 - loss: 1.4525 - categorical_accuracy: 0.4878
 1184/60000 [..............................] - ETA: 2:03 - loss: 1.4388 - categorical_accuracy: 0.4932
 1216/60000 [..............................] - ETA: 2:02 - loss: 1.4181 - categorical_accuracy: 0.5016
 1248/60000 [..............................] - ETA: 2:02 - loss: 1.4009 - categorical_accuracy: 0.5088
 1280/60000 [..............................] - ETA: 2:02 - loss: 1.3829 - categorical_accuracy: 0.5164
 1312/60000 [..............................] - ETA: 2:01 - loss: 1.3657 - categorical_accuracy: 0.5236
 1344/60000 [..............................] - ETA: 2:01 - loss: 1.3501 - categorical_accuracy: 0.5298
 1376/60000 [..............................] - ETA: 2:01 - loss: 1.3399 - categorical_accuracy: 0.5320
 1408/60000 [..............................] - ETA: 2:00 - loss: 1.3243 - categorical_accuracy: 0.5384
 1440/60000 [..............................] - ETA: 2:00 - loss: 1.3127 - categorical_accuracy: 0.5437
 1472/60000 [..............................] - ETA: 2:00 - loss: 1.2990 - categorical_accuracy: 0.5476
 1504/60000 [..............................] - ETA: 2:00 - loss: 1.2797 - categorical_accuracy: 0.5545
 1536/60000 [..............................] - ETA: 2:00 - loss: 1.2662 - categorical_accuracy: 0.5579
 1568/60000 [..............................] - ETA: 1:59 - loss: 1.2513 - categorical_accuracy: 0.5631
 1600/60000 [..............................] - ETA: 1:59 - loss: 1.2350 - categorical_accuracy: 0.5688
 1632/60000 [..............................] - ETA: 1:59 - loss: 1.2240 - categorical_accuracy: 0.5729
 1664/60000 [..............................] - ETA: 1:59 - loss: 1.2159 - categorical_accuracy: 0.5775
 1696/60000 [..............................] - ETA: 1:58 - loss: 1.2044 - categorical_accuracy: 0.5820
 1728/60000 [..............................] - ETA: 1:58 - loss: 1.1943 - categorical_accuracy: 0.5862
 1760/60000 [..............................] - ETA: 1:58 - loss: 1.1845 - categorical_accuracy: 0.5892
 1792/60000 [..............................] - ETA: 1:58 - loss: 1.1723 - categorical_accuracy: 0.5938
 1824/60000 [..............................] - ETA: 1:58 - loss: 1.1611 - categorical_accuracy: 0.5987
 1856/60000 [..............................] - ETA: 1:58 - loss: 1.1523 - categorical_accuracy: 0.6024
 1888/60000 [..............................] - ETA: 1:57 - loss: 1.1430 - categorical_accuracy: 0.6059
 1920/60000 [..............................] - ETA: 1:57 - loss: 1.1335 - categorical_accuracy: 0.6094
 1952/60000 [..............................] - ETA: 1:57 - loss: 1.1243 - categorical_accuracy: 0.6127
 1984/60000 [..............................] - ETA: 1:57 - loss: 1.1144 - categorical_accuracy: 0.6159
 2016/60000 [>.............................] - ETA: 1:57 - loss: 1.1068 - categorical_accuracy: 0.6195
 2048/60000 [>.............................] - ETA: 1:56 - loss: 1.1021 - categorical_accuracy: 0.6211
 2080/60000 [>.............................] - ETA: 1:57 - loss: 1.0925 - categorical_accuracy: 0.6250
 2112/60000 [>.............................] - ETA: 1:56 - loss: 1.0818 - categorical_accuracy: 0.6288
 2144/60000 [>.............................] - ETA: 1:56 - loss: 1.0734 - categorical_accuracy: 0.6315
 2176/60000 [>.............................] - ETA: 1:56 - loss: 1.0692 - categorical_accuracy: 0.6347
 2208/60000 [>.............................] - ETA: 1:56 - loss: 1.0615 - categorical_accuracy: 0.6377
 2240/60000 [>.............................] - ETA: 1:56 - loss: 1.0550 - categorical_accuracy: 0.6388
 2272/60000 [>.............................] - ETA: 1:56 - loss: 1.0466 - categorical_accuracy: 0.6408
 2304/60000 [>.............................] - ETA: 1:56 - loss: 1.0365 - categorical_accuracy: 0.6445
 2336/60000 [>.............................] - ETA: 1:55 - loss: 1.0296 - categorical_accuracy: 0.6468
 2368/60000 [>.............................] - ETA: 1:55 - loss: 1.0260 - categorical_accuracy: 0.6486
 2400/60000 [>.............................] - ETA: 1:55 - loss: 1.0187 - categorical_accuracy: 0.6504
 2432/60000 [>.............................] - ETA: 1:55 - loss: 1.0108 - categorical_accuracy: 0.6534
 2464/60000 [>.............................] - ETA: 1:55 - loss: 1.0026 - categorical_accuracy: 0.6558
 2496/60000 [>.............................] - ETA: 1:55 - loss: 0.9985 - categorical_accuracy: 0.6579
 2528/60000 [>.............................] - ETA: 1:55 - loss: 0.9918 - categorical_accuracy: 0.6602
 2560/60000 [>.............................] - ETA: 1:54 - loss: 0.9842 - categorical_accuracy: 0.6625
 2592/60000 [>.............................] - ETA: 1:54 - loss: 0.9766 - categorical_accuracy: 0.6651
 2624/60000 [>.............................] - ETA: 1:54 - loss: 0.9715 - categorical_accuracy: 0.6673
 2656/60000 [>.............................] - ETA: 1:54 - loss: 0.9630 - categorical_accuracy: 0.6706
 2688/60000 [>.............................] - ETA: 1:54 - loss: 0.9591 - categorical_accuracy: 0.6719
 2720/60000 [>.............................] - ETA: 1:54 - loss: 0.9557 - categorical_accuracy: 0.6728
 2752/60000 [>.............................] - ETA: 1:54 - loss: 0.9508 - categorical_accuracy: 0.6755
 2784/60000 [>.............................] - ETA: 1:54 - loss: 0.9438 - categorical_accuracy: 0.6774
 2816/60000 [>.............................] - ETA: 1:53 - loss: 0.9378 - categorical_accuracy: 0.6790
 2848/60000 [>.............................] - ETA: 1:53 - loss: 0.9302 - categorical_accuracy: 0.6819
 2880/60000 [>.............................] - ETA: 1:53 - loss: 0.9249 - categorical_accuracy: 0.6840
 2912/60000 [>.............................] - ETA: 1:53 - loss: 0.9192 - categorical_accuracy: 0.6861
 2944/60000 [>.............................] - ETA: 1:53 - loss: 0.9125 - categorical_accuracy: 0.6889
 2976/60000 [>.............................] - ETA: 1:53 - loss: 0.9097 - categorical_accuracy: 0.6899
 3008/60000 [>.............................] - ETA: 1:53 - loss: 0.9061 - categorical_accuracy: 0.6915
 3040/60000 [>.............................] - ETA: 1:53 - loss: 0.9009 - categorical_accuracy: 0.6934
 3072/60000 [>.............................] - ETA: 1:52 - loss: 0.8958 - categorical_accuracy: 0.6950
 3104/60000 [>.............................] - ETA: 1:52 - loss: 0.8886 - categorical_accuracy: 0.6978
 3136/60000 [>.............................] - ETA: 1:52 - loss: 0.8861 - categorical_accuracy: 0.6990
 3168/60000 [>.............................] - ETA: 1:52 - loss: 0.8820 - categorical_accuracy: 0.7001
 3200/60000 [>.............................] - ETA: 1:52 - loss: 0.8763 - categorical_accuracy: 0.7025
 3232/60000 [>.............................] - ETA: 1:52 - loss: 0.8705 - categorical_accuracy: 0.7051
 3264/60000 [>.............................] - ETA: 1:52 - loss: 0.8655 - categorical_accuracy: 0.7065
 3296/60000 [>.............................] - ETA: 1:52 - loss: 0.8594 - categorical_accuracy: 0.7087
 3328/60000 [>.............................] - ETA: 1:52 - loss: 0.8547 - categorical_accuracy: 0.7103
 3360/60000 [>.............................] - ETA: 1:52 - loss: 0.8520 - categorical_accuracy: 0.7125
 3392/60000 [>.............................] - ETA: 1:51 - loss: 0.8453 - categorical_accuracy: 0.7149
 3424/60000 [>.............................] - ETA: 1:51 - loss: 0.8401 - categorical_accuracy: 0.7167
 3456/60000 [>.............................] - ETA: 1:51 - loss: 0.8375 - categorical_accuracy: 0.7176
 3488/60000 [>.............................] - ETA: 1:51 - loss: 0.8338 - categorical_accuracy: 0.7190
 3520/60000 [>.............................] - ETA: 1:51 - loss: 0.8283 - categorical_accuracy: 0.7210
 3552/60000 [>.............................] - ETA: 1:51 - loss: 0.8255 - categorical_accuracy: 0.7221
 3584/60000 [>.............................] - ETA: 1:51 - loss: 0.8232 - categorical_accuracy: 0.7235
 3616/60000 [>.............................] - ETA: 1:51 - loss: 0.8180 - categorical_accuracy: 0.7254
 3648/60000 [>.............................] - ETA: 1:51 - loss: 0.8154 - categorical_accuracy: 0.7264
 3680/60000 [>.............................] - ETA: 1:51 - loss: 0.8131 - categorical_accuracy: 0.7272
 3712/60000 [>.............................] - ETA: 1:50 - loss: 0.8104 - categorical_accuracy: 0.7282
 3744/60000 [>.............................] - ETA: 1:50 - loss: 0.8063 - categorical_accuracy: 0.7292
 3776/60000 [>.............................] - ETA: 1:50 - loss: 0.8017 - categorical_accuracy: 0.7301
 3808/60000 [>.............................] - ETA: 1:50 - loss: 0.7973 - categorical_accuracy: 0.7316
 3840/60000 [>.............................] - ETA: 1:50 - loss: 0.7926 - categorical_accuracy: 0.7333
 3872/60000 [>.............................] - ETA: 1:50 - loss: 0.7895 - categorical_accuracy: 0.7348
 3904/60000 [>.............................] - ETA: 1:50 - loss: 0.7847 - categorical_accuracy: 0.7367
 3936/60000 [>.............................] - ETA: 1:50 - loss: 0.7837 - categorical_accuracy: 0.7370
 3968/60000 [>.............................] - ETA: 1:50 - loss: 0.7798 - categorical_accuracy: 0.7384
 4000/60000 [=>............................] - ETA: 1:50 - loss: 0.7753 - categorical_accuracy: 0.7400
 4032/60000 [=>............................] - ETA: 1:50 - loss: 0.7727 - categorical_accuracy: 0.7408
 4064/60000 [=>............................] - ETA: 1:50 - loss: 0.7694 - categorical_accuracy: 0.7416
 4096/60000 [=>............................] - ETA: 1:49 - loss: 0.7677 - categorical_accuracy: 0.7427
 4128/60000 [=>............................] - ETA: 1:49 - loss: 0.7670 - categorical_accuracy: 0.7430
 4160/60000 [=>............................] - ETA: 1:49 - loss: 0.7642 - categorical_accuracy: 0.7437
 4192/60000 [=>............................] - ETA: 1:49 - loss: 0.7616 - categorical_accuracy: 0.7448
 4224/60000 [=>............................] - ETA: 1:49 - loss: 0.7585 - categorical_accuracy: 0.7460
 4256/60000 [=>............................] - ETA: 1:49 - loss: 0.7550 - categorical_accuracy: 0.7472
 4288/60000 [=>............................] - ETA: 1:49 - loss: 0.7519 - categorical_accuracy: 0.7488
 4320/60000 [=>............................] - ETA: 1:49 - loss: 0.7491 - categorical_accuracy: 0.7502
 4352/60000 [=>............................] - ETA: 1:49 - loss: 0.7449 - categorical_accuracy: 0.7516
 4384/60000 [=>............................] - ETA: 1:49 - loss: 0.7411 - categorical_accuracy: 0.7532
 4416/60000 [=>............................] - ETA: 1:49 - loss: 0.7381 - categorical_accuracy: 0.7545
 4448/60000 [=>............................] - ETA: 1:49 - loss: 0.7336 - categorical_accuracy: 0.7558
 4480/60000 [=>............................] - ETA: 1:48 - loss: 0.7305 - categorical_accuracy: 0.7574
 4512/60000 [=>............................] - ETA: 1:48 - loss: 0.7268 - categorical_accuracy: 0.7586
 4544/60000 [=>............................] - ETA: 1:48 - loss: 0.7240 - categorical_accuracy: 0.7595
 4576/60000 [=>............................] - ETA: 1:48 - loss: 0.7206 - categorical_accuracy: 0.7609
 4608/60000 [=>............................] - ETA: 1:48 - loss: 0.7180 - categorical_accuracy: 0.7617
 4640/60000 [=>............................] - ETA: 1:48 - loss: 0.7164 - categorical_accuracy: 0.7623
 4672/60000 [=>............................] - ETA: 1:48 - loss: 0.7134 - categorical_accuracy: 0.7633
 4704/60000 [=>............................] - ETA: 1:48 - loss: 0.7107 - categorical_accuracy: 0.7645
 4736/60000 [=>............................] - ETA: 1:48 - loss: 0.7063 - categorical_accuracy: 0.7660
 4768/60000 [=>............................] - ETA: 1:47 - loss: 0.7045 - categorical_accuracy: 0.7668
 4800/60000 [=>............................] - ETA: 1:47 - loss: 0.7013 - categorical_accuracy: 0.7677
 4832/60000 [=>............................] - ETA: 1:47 - loss: 0.6991 - categorical_accuracy: 0.7686
 4864/60000 [=>............................] - ETA: 1:47 - loss: 0.6963 - categorical_accuracy: 0.7695
 4896/60000 [=>............................] - ETA: 1:47 - loss: 0.6950 - categorical_accuracy: 0.7700
 4928/60000 [=>............................] - ETA: 1:47 - loss: 0.6913 - categorical_accuracy: 0.7713
 4960/60000 [=>............................] - ETA: 1:47 - loss: 0.6884 - categorical_accuracy: 0.7726
 4992/60000 [=>............................] - ETA: 1:47 - loss: 0.6861 - categorical_accuracy: 0.7732
 5024/60000 [=>............................] - ETA: 1:47 - loss: 0.6832 - categorical_accuracy: 0.7741
 5056/60000 [=>............................] - ETA: 1:47 - loss: 0.6803 - categorical_accuracy: 0.7749
 5088/60000 [=>............................] - ETA: 1:46 - loss: 0.6779 - categorical_accuracy: 0.7757
 5120/60000 [=>............................] - ETA: 1:46 - loss: 0.6765 - categorical_accuracy: 0.7764
 5152/60000 [=>............................] - ETA: 1:46 - loss: 0.6743 - categorical_accuracy: 0.7772
 5184/60000 [=>............................] - ETA: 1:46 - loss: 0.6710 - categorical_accuracy: 0.7784
 5216/60000 [=>............................] - ETA: 1:46 - loss: 0.6693 - categorical_accuracy: 0.7791
 5248/60000 [=>............................] - ETA: 1:46 - loss: 0.6665 - categorical_accuracy: 0.7799
 5280/60000 [=>............................] - ETA: 1:46 - loss: 0.6634 - categorical_accuracy: 0.7812
 5312/60000 [=>............................] - ETA: 1:46 - loss: 0.6611 - categorical_accuracy: 0.7822
 5344/60000 [=>............................] - ETA: 1:46 - loss: 0.6584 - categorical_accuracy: 0.7829
 5376/60000 [=>............................] - ETA: 1:46 - loss: 0.6573 - categorical_accuracy: 0.7837
 5408/60000 [=>............................] - ETA: 1:46 - loss: 0.6550 - categorical_accuracy: 0.7842
 5440/60000 [=>............................] - ETA: 1:46 - loss: 0.6523 - categorical_accuracy: 0.7851
 5472/60000 [=>............................] - ETA: 1:46 - loss: 0.6507 - categorical_accuracy: 0.7858
 5504/60000 [=>............................] - ETA: 1:46 - loss: 0.6475 - categorical_accuracy: 0.7871
 5536/60000 [=>............................] - ETA: 1:46 - loss: 0.6455 - categorical_accuracy: 0.7878
 5568/60000 [=>............................] - ETA: 1:46 - loss: 0.6432 - categorical_accuracy: 0.7883
 5600/60000 [=>............................] - ETA: 1:46 - loss: 0.6419 - categorical_accuracy: 0.7887
 5632/60000 [=>............................] - ETA: 1:45 - loss: 0.6386 - categorical_accuracy: 0.7900
 5664/60000 [=>............................] - ETA: 1:45 - loss: 0.6369 - categorical_accuracy: 0.7910
 5696/60000 [=>............................] - ETA: 1:45 - loss: 0.6346 - categorical_accuracy: 0.7918
 5728/60000 [=>............................] - ETA: 1:45 - loss: 0.6320 - categorical_accuracy: 0.7928
 5760/60000 [=>............................] - ETA: 1:45 - loss: 0.6312 - categorical_accuracy: 0.7929
 5792/60000 [=>............................] - ETA: 1:45 - loss: 0.6303 - categorical_accuracy: 0.7935
 5824/60000 [=>............................] - ETA: 1:45 - loss: 0.6281 - categorical_accuracy: 0.7943
 5856/60000 [=>............................] - ETA: 1:45 - loss: 0.6266 - categorical_accuracy: 0.7949
 5888/60000 [=>............................] - ETA: 1:45 - loss: 0.6245 - categorical_accuracy: 0.7952
 5920/60000 [=>............................] - ETA: 1:45 - loss: 0.6228 - categorical_accuracy: 0.7954
 5952/60000 [=>............................] - ETA: 1:45 - loss: 0.6214 - categorical_accuracy: 0.7959
 5984/60000 [=>............................] - ETA: 1:45 - loss: 0.6193 - categorical_accuracy: 0.7965
 6016/60000 [==>...........................] - ETA: 1:45 - loss: 0.6169 - categorical_accuracy: 0.7974
 6048/60000 [==>...........................] - ETA: 1:45 - loss: 0.6146 - categorical_accuracy: 0.7979
 6080/60000 [==>...........................] - ETA: 1:44 - loss: 0.6129 - categorical_accuracy: 0.7984
 6112/60000 [==>...........................] - ETA: 1:44 - loss: 0.6104 - categorical_accuracy: 0.7991
 6144/60000 [==>...........................] - ETA: 1:44 - loss: 0.6086 - categorical_accuracy: 0.7998
 6176/60000 [==>...........................] - ETA: 1:44 - loss: 0.6065 - categorical_accuracy: 0.8002
 6208/60000 [==>...........................] - ETA: 1:44 - loss: 0.6056 - categorical_accuracy: 0.8006
 6240/60000 [==>...........................] - ETA: 1:44 - loss: 0.6029 - categorical_accuracy: 0.8014
 6272/60000 [==>...........................] - ETA: 1:44 - loss: 0.6006 - categorical_accuracy: 0.8023
 6304/60000 [==>...........................] - ETA: 1:44 - loss: 0.5998 - categorical_accuracy: 0.8028
 6336/60000 [==>...........................] - ETA: 1:44 - loss: 0.5987 - categorical_accuracy: 0.8030
 6368/60000 [==>...........................] - ETA: 1:44 - loss: 0.5964 - categorical_accuracy: 0.8039
 6400/60000 [==>...........................] - ETA: 1:44 - loss: 0.5948 - categorical_accuracy: 0.8044
 6432/60000 [==>...........................] - ETA: 1:44 - loss: 0.5922 - categorical_accuracy: 0.8053
 6464/60000 [==>...........................] - ETA: 1:44 - loss: 0.5915 - categorical_accuracy: 0.8055
 6496/60000 [==>...........................] - ETA: 1:44 - loss: 0.5893 - categorical_accuracy: 0.8062
 6528/60000 [==>...........................] - ETA: 1:44 - loss: 0.5875 - categorical_accuracy: 0.8067
 6560/60000 [==>...........................] - ETA: 1:43 - loss: 0.5849 - categorical_accuracy: 0.8076
 6592/60000 [==>...........................] - ETA: 1:43 - loss: 0.5836 - categorical_accuracy: 0.8081
 6624/60000 [==>...........................] - ETA: 1:43 - loss: 0.5820 - categorical_accuracy: 0.8087
 6656/60000 [==>...........................] - ETA: 1:43 - loss: 0.5799 - categorical_accuracy: 0.8096
 6688/60000 [==>...........................] - ETA: 1:43 - loss: 0.5792 - categorical_accuracy: 0.8103
 6720/60000 [==>...........................] - ETA: 1:43 - loss: 0.5782 - categorical_accuracy: 0.8107
 6752/60000 [==>...........................] - ETA: 1:43 - loss: 0.5768 - categorical_accuracy: 0.8109
 6784/60000 [==>...........................] - ETA: 1:43 - loss: 0.5747 - categorical_accuracy: 0.8115
 6816/60000 [==>...........................] - ETA: 1:43 - loss: 0.5728 - categorical_accuracy: 0.8121
 6848/60000 [==>...........................] - ETA: 1:43 - loss: 0.5709 - categorical_accuracy: 0.8128
 6880/60000 [==>...........................] - ETA: 1:43 - loss: 0.5702 - categorical_accuracy: 0.8131
 6912/60000 [==>...........................] - ETA: 1:43 - loss: 0.5691 - categorical_accuracy: 0.8134
 6944/60000 [==>...........................] - ETA: 1:43 - loss: 0.5676 - categorical_accuracy: 0.8141
 6976/60000 [==>...........................] - ETA: 1:42 - loss: 0.5658 - categorical_accuracy: 0.8147
 7008/60000 [==>...........................] - ETA: 1:42 - loss: 0.5640 - categorical_accuracy: 0.8152
 7040/60000 [==>...........................] - ETA: 1:42 - loss: 0.5640 - categorical_accuracy: 0.8155
 7072/60000 [==>...........................] - ETA: 1:42 - loss: 0.5624 - categorical_accuracy: 0.8159
 7104/60000 [==>...........................] - ETA: 1:42 - loss: 0.5605 - categorical_accuracy: 0.8164
 7136/60000 [==>...........................] - ETA: 1:42 - loss: 0.5586 - categorical_accuracy: 0.8171
 7168/60000 [==>...........................] - ETA: 1:42 - loss: 0.5581 - categorical_accuracy: 0.8171
 7200/60000 [==>...........................] - ETA: 1:42 - loss: 0.5598 - categorical_accuracy: 0.8169
 7232/60000 [==>...........................] - ETA: 1:42 - loss: 0.5580 - categorical_accuracy: 0.8176
 7264/60000 [==>...........................] - ETA: 1:42 - loss: 0.5571 - categorical_accuracy: 0.8180
 7296/60000 [==>...........................] - ETA: 1:42 - loss: 0.5563 - categorical_accuracy: 0.8184
 7328/60000 [==>...........................] - ETA: 1:42 - loss: 0.5551 - categorical_accuracy: 0.8191
 7360/60000 [==>...........................] - ETA: 1:42 - loss: 0.5536 - categorical_accuracy: 0.8196
 7392/60000 [==>...........................] - ETA: 1:41 - loss: 0.5537 - categorical_accuracy: 0.8197
 7424/60000 [==>...........................] - ETA: 1:41 - loss: 0.5521 - categorical_accuracy: 0.8203
 7456/60000 [==>...........................] - ETA: 1:41 - loss: 0.5507 - categorical_accuracy: 0.8208
 7488/60000 [==>...........................] - ETA: 1:41 - loss: 0.5491 - categorical_accuracy: 0.8214
 7520/60000 [==>...........................] - ETA: 1:41 - loss: 0.5473 - categorical_accuracy: 0.8219
 7552/60000 [==>...........................] - ETA: 1:41 - loss: 0.5469 - categorical_accuracy: 0.8220
 7584/60000 [==>...........................] - ETA: 1:41 - loss: 0.5453 - categorical_accuracy: 0.8227
 7616/60000 [==>...........................] - ETA: 1:41 - loss: 0.5449 - categorical_accuracy: 0.8227
 7648/60000 [==>...........................] - ETA: 1:41 - loss: 0.5438 - categorical_accuracy: 0.8230
 7680/60000 [==>...........................] - ETA: 1:41 - loss: 0.5432 - categorical_accuracy: 0.8229
 7712/60000 [==>...........................] - ETA: 1:41 - loss: 0.5422 - categorical_accuracy: 0.8233
 7744/60000 [==>...........................] - ETA: 1:41 - loss: 0.5403 - categorical_accuracy: 0.8239
 7776/60000 [==>...........................] - ETA: 1:41 - loss: 0.5393 - categorical_accuracy: 0.8243
 7808/60000 [==>...........................] - ETA: 1:41 - loss: 0.5386 - categorical_accuracy: 0.8245
 7840/60000 [==>...........................] - ETA: 1:41 - loss: 0.5378 - categorical_accuracy: 0.8251
 7872/60000 [==>...........................] - ETA: 1:40 - loss: 0.5362 - categorical_accuracy: 0.8256
 7904/60000 [==>...........................] - ETA: 1:40 - loss: 0.5346 - categorical_accuracy: 0.8260
 7936/60000 [==>...........................] - ETA: 1:40 - loss: 0.5337 - categorical_accuracy: 0.8265
 7968/60000 [==>...........................] - ETA: 1:40 - loss: 0.5322 - categorical_accuracy: 0.8269
 8000/60000 [===>..........................] - ETA: 1:40 - loss: 0.5302 - categorical_accuracy: 0.8276
 8032/60000 [===>..........................] - ETA: 1:40 - loss: 0.5297 - categorical_accuracy: 0.8279
 8064/60000 [===>..........................] - ETA: 1:40 - loss: 0.5278 - categorical_accuracy: 0.8285
 8096/60000 [===>..........................] - ETA: 1:40 - loss: 0.5264 - categorical_accuracy: 0.8289
 8128/60000 [===>..........................] - ETA: 1:40 - loss: 0.5251 - categorical_accuracy: 0.8292
 8160/60000 [===>..........................] - ETA: 1:40 - loss: 0.5233 - categorical_accuracy: 0.8299
 8192/60000 [===>..........................] - ETA: 1:40 - loss: 0.5228 - categorical_accuracy: 0.8301
 8224/60000 [===>..........................] - ETA: 1:40 - loss: 0.5216 - categorical_accuracy: 0.8305
 8256/60000 [===>..........................] - ETA: 1:40 - loss: 0.5201 - categorical_accuracy: 0.8310
 8288/60000 [===>..........................] - ETA: 1:39 - loss: 0.5204 - categorical_accuracy: 0.8310
 8320/60000 [===>..........................] - ETA: 1:39 - loss: 0.5187 - categorical_accuracy: 0.8314
 8352/60000 [===>..........................] - ETA: 1:39 - loss: 0.5172 - categorical_accuracy: 0.8319
 8384/60000 [===>..........................] - ETA: 1:39 - loss: 0.5165 - categorical_accuracy: 0.8319
 8416/60000 [===>..........................] - ETA: 1:39 - loss: 0.5148 - categorical_accuracy: 0.8326
 8448/60000 [===>..........................] - ETA: 1:39 - loss: 0.5131 - categorical_accuracy: 0.8331
 8480/60000 [===>..........................] - ETA: 1:39 - loss: 0.5122 - categorical_accuracy: 0.8333
 8512/60000 [===>..........................] - ETA: 1:39 - loss: 0.5110 - categorical_accuracy: 0.8335
 8544/60000 [===>..........................] - ETA: 1:39 - loss: 0.5099 - categorical_accuracy: 0.8337
 8576/60000 [===>..........................] - ETA: 1:39 - loss: 0.5084 - categorical_accuracy: 0.8342
 8608/60000 [===>..........................] - ETA: 1:39 - loss: 0.5068 - categorical_accuracy: 0.8347
 8640/60000 [===>..........................] - ETA: 1:39 - loss: 0.5064 - categorical_accuracy: 0.8348
 8672/60000 [===>..........................] - ETA: 1:39 - loss: 0.5050 - categorical_accuracy: 0.8353
 8704/60000 [===>..........................] - ETA: 1:39 - loss: 0.5057 - categorical_accuracy: 0.8352
 8736/60000 [===>..........................] - ETA: 1:38 - loss: 0.5042 - categorical_accuracy: 0.8357
 8768/60000 [===>..........................] - ETA: 1:38 - loss: 0.5035 - categorical_accuracy: 0.8360
 8800/60000 [===>..........................] - ETA: 1:38 - loss: 0.5023 - categorical_accuracy: 0.8364
 8832/60000 [===>..........................] - ETA: 1:38 - loss: 0.5012 - categorical_accuracy: 0.8367
 8864/60000 [===>..........................] - ETA: 1:38 - loss: 0.5000 - categorical_accuracy: 0.8372
 8896/60000 [===>..........................] - ETA: 1:38 - loss: 0.4992 - categorical_accuracy: 0.8375
 8928/60000 [===>..........................] - ETA: 1:38 - loss: 0.4975 - categorical_accuracy: 0.8380
 8960/60000 [===>..........................] - ETA: 1:38 - loss: 0.4964 - categorical_accuracy: 0.8384
 8992/60000 [===>..........................] - ETA: 1:38 - loss: 0.4956 - categorical_accuracy: 0.8385
 9024/60000 [===>..........................] - ETA: 1:38 - loss: 0.4947 - categorical_accuracy: 0.8389
 9056/60000 [===>..........................] - ETA: 1:38 - loss: 0.4937 - categorical_accuracy: 0.8392
 9088/60000 [===>..........................] - ETA: 1:38 - loss: 0.4925 - categorical_accuracy: 0.8396
 9120/60000 [===>..........................] - ETA: 1:38 - loss: 0.4912 - categorical_accuracy: 0.8400
 9152/60000 [===>..........................] - ETA: 1:37 - loss: 0.4903 - categorical_accuracy: 0.8403
 9184/60000 [===>..........................] - ETA: 1:37 - loss: 0.4892 - categorical_accuracy: 0.8406
 9216/60000 [===>..........................] - ETA: 1:37 - loss: 0.4883 - categorical_accuracy: 0.8408
 9248/60000 [===>..........................] - ETA: 1:37 - loss: 0.4871 - categorical_accuracy: 0.8412
 9280/60000 [===>..........................] - ETA: 1:37 - loss: 0.4858 - categorical_accuracy: 0.8416
 9312/60000 [===>..........................] - ETA: 1:37 - loss: 0.4852 - categorical_accuracy: 0.8417
 9344/60000 [===>..........................] - ETA: 1:37 - loss: 0.4844 - categorical_accuracy: 0.8419
 9376/60000 [===>..........................] - ETA: 1:37 - loss: 0.4851 - categorical_accuracy: 0.8418
 9408/60000 [===>..........................] - ETA: 1:37 - loss: 0.4848 - categorical_accuracy: 0.8422
 9440/60000 [===>..........................] - ETA: 1:37 - loss: 0.4840 - categorical_accuracy: 0.8423
 9472/60000 [===>..........................] - ETA: 1:37 - loss: 0.4839 - categorical_accuracy: 0.8423
 9504/60000 [===>..........................] - ETA: 1:37 - loss: 0.4828 - categorical_accuracy: 0.8426
 9536/60000 [===>..........................] - ETA: 1:37 - loss: 0.4819 - categorical_accuracy: 0.8430
 9568/60000 [===>..........................] - ETA: 1:37 - loss: 0.4811 - categorical_accuracy: 0.8433
 9600/60000 [===>..........................] - ETA: 1:37 - loss: 0.4798 - categorical_accuracy: 0.8438
 9632/60000 [===>..........................] - ETA: 1:36 - loss: 0.4788 - categorical_accuracy: 0.8440
 9664/60000 [===>..........................] - ETA: 1:36 - loss: 0.4778 - categorical_accuracy: 0.8443
 9696/60000 [===>..........................] - ETA: 1:36 - loss: 0.4768 - categorical_accuracy: 0.8447
 9728/60000 [===>..........................] - ETA: 1:36 - loss: 0.4756 - categorical_accuracy: 0.8451
 9760/60000 [===>..........................] - ETA: 1:36 - loss: 0.4743 - categorical_accuracy: 0.8456
 9792/60000 [===>..........................] - ETA: 1:36 - loss: 0.4729 - categorical_accuracy: 0.8460
 9824/60000 [===>..........................] - ETA: 1:36 - loss: 0.4720 - categorical_accuracy: 0.8463
 9856/60000 [===>..........................] - ETA: 1:36 - loss: 0.4714 - categorical_accuracy: 0.8465
 9888/60000 [===>..........................] - ETA: 1:36 - loss: 0.4711 - categorical_accuracy: 0.8468
 9920/60000 [===>..........................] - ETA: 1:36 - loss: 0.4706 - categorical_accuracy: 0.8470
 9952/60000 [===>..........................] - ETA: 1:36 - loss: 0.4703 - categorical_accuracy: 0.8470
 9984/60000 [===>..........................] - ETA: 1:36 - loss: 0.4691 - categorical_accuracy: 0.8474
10016/60000 [====>.........................] - ETA: 1:36 - loss: 0.4678 - categorical_accuracy: 0.8478
10048/60000 [====>.........................] - ETA: 1:36 - loss: 0.4668 - categorical_accuracy: 0.8482
10080/60000 [====>.........................] - ETA: 1:35 - loss: 0.4660 - categorical_accuracy: 0.8485
10112/60000 [====>.........................] - ETA: 1:35 - loss: 0.4648 - categorical_accuracy: 0.8490
10144/60000 [====>.........................] - ETA: 1:35 - loss: 0.4641 - categorical_accuracy: 0.8491
10176/60000 [====>.........................] - ETA: 1:35 - loss: 0.4636 - categorical_accuracy: 0.8494
10208/60000 [====>.........................] - ETA: 1:35 - loss: 0.4626 - categorical_accuracy: 0.8498
10240/60000 [====>.........................] - ETA: 1:35 - loss: 0.4612 - categorical_accuracy: 0.8503
10272/60000 [====>.........................] - ETA: 1:35 - loss: 0.4599 - categorical_accuracy: 0.8508
10304/60000 [====>.........................] - ETA: 1:35 - loss: 0.4587 - categorical_accuracy: 0.8512
10336/60000 [====>.........................] - ETA: 1:35 - loss: 0.4577 - categorical_accuracy: 0.8517
10368/60000 [====>.........................] - ETA: 1:35 - loss: 0.4565 - categorical_accuracy: 0.8520
10400/60000 [====>.........................] - ETA: 1:35 - loss: 0.4560 - categorical_accuracy: 0.8521
10432/60000 [====>.........................] - ETA: 1:35 - loss: 0.4549 - categorical_accuracy: 0.8524
10464/60000 [====>.........................] - ETA: 1:35 - loss: 0.4537 - categorical_accuracy: 0.8528
10496/60000 [====>.........................] - ETA: 1:35 - loss: 0.4530 - categorical_accuracy: 0.8530
10528/60000 [====>.........................] - ETA: 1:35 - loss: 0.4522 - categorical_accuracy: 0.8532
10560/60000 [====>.........................] - ETA: 1:34 - loss: 0.4512 - categorical_accuracy: 0.8536
10592/60000 [====>.........................] - ETA: 1:34 - loss: 0.4505 - categorical_accuracy: 0.8536
10624/60000 [====>.........................] - ETA: 1:34 - loss: 0.4499 - categorical_accuracy: 0.8536
10656/60000 [====>.........................] - ETA: 1:34 - loss: 0.4495 - categorical_accuracy: 0.8536
10688/60000 [====>.........................] - ETA: 1:34 - loss: 0.4490 - categorical_accuracy: 0.8538
10720/60000 [====>.........................] - ETA: 1:34 - loss: 0.4479 - categorical_accuracy: 0.8542
10752/60000 [====>.........................] - ETA: 1:34 - loss: 0.4474 - categorical_accuracy: 0.8544
10784/60000 [====>.........................] - ETA: 1:34 - loss: 0.4466 - categorical_accuracy: 0.8546
10816/60000 [====>.........................] - ETA: 1:34 - loss: 0.4460 - categorical_accuracy: 0.8548
10848/60000 [====>.........................] - ETA: 1:34 - loss: 0.4453 - categorical_accuracy: 0.8550
10880/60000 [====>.........................] - ETA: 1:34 - loss: 0.4441 - categorical_accuracy: 0.8554
10912/60000 [====>.........................] - ETA: 1:34 - loss: 0.4435 - categorical_accuracy: 0.8557
10944/60000 [====>.........................] - ETA: 1:34 - loss: 0.4436 - categorical_accuracy: 0.8557
10976/60000 [====>.........................] - ETA: 1:34 - loss: 0.4425 - categorical_accuracy: 0.8560
11008/60000 [====>.........................] - ETA: 1:34 - loss: 0.4420 - categorical_accuracy: 0.8563
11040/60000 [====>.........................] - ETA: 1:33 - loss: 0.4410 - categorical_accuracy: 0.8567
11072/60000 [====>.........................] - ETA: 1:33 - loss: 0.4407 - categorical_accuracy: 0.8568
11104/60000 [====>.........................] - ETA: 1:33 - loss: 0.4398 - categorical_accuracy: 0.8571
11136/60000 [====>.........................] - ETA: 1:33 - loss: 0.4390 - categorical_accuracy: 0.8573
11168/60000 [====>.........................] - ETA: 1:33 - loss: 0.4380 - categorical_accuracy: 0.8577
11200/60000 [====>.........................] - ETA: 1:33 - loss: 0.4373 - categorical_accuracy: 0.8579
11232/60000 [====>.........................] - ETA: 1:33 - loss: 0.4367 - categorical_accuracy: 0.8580
11264/60000 [====>.........................] - ETA: 1:33 - loss: 0.4357 - categorical_accuracy: 0.8584
11296/60000 [====>.........................] - ETA: 1:33 - loss: 0.4353 - categorical_accuracy: 0.8585
11328/60000 [====>.........................] - ETA: 1:33 - loss: 0.4343 - categorical_accuracy: 0.8589
11360/60000 [====>.........................] - ETA: 1:33 - loss: 0.4333 - categorical_accuracy: 0.8593
11392/60000 [====>.........................] - ETA: 1:33 - loss: 0.4326 - categorical_accuracy: 0.8596
11424/60000 [====>.........................] - ETA: 1:33 - loss: 0.4316 - categorical_accuracy: 0.8599
11456/60000 [====>.........................] - ETA: 1:33 - loss: 0.4310 - categorical_accuracy: 0.8602
11488/60000 [====>.........................] - ETA: 1:33 - loss: 0.4300 - categorical_accuracy: 0.8606
11520/60000 [====>.........................] - ETA: 1:33 - loss: 0.4292 - categorical_accuracy: 0.8609
11552/60000 [====>.........................] - ETA: 1:32 - loss: 0.4281 - categorical_accuracy: 0.8613
11584/60000 [====>.........................] - ETA: 1:32 - loss: 0.4273 - categorical_accuracy: 0.8615
11616/60000 [====>.........................] - ETA: 1:32 - loss: 0.4264 - categorical_accuracy: 0.8618
11648/60000 [====>.........................] - ETA: 1:32 - loss: 0.4254 - categorical_accuracy: 0.8622
11680/60000 [====>.........................] - ETA: 1:32 - loss: 0.4244 - categorical_accuracy: 0.8625
11712/60000 [====>.........................] - ETA: 1:32 - loss: 0.4236 - categorical_accuracy: 0.8628
11744/60000 [====>.........................] - ETA: 1:32 - loss: 0.4230 - categorical_accuracy: 0.8630
11776/60000 [====>.........................] - ETA: 1:32 - loss: 0.4228 - categorical_accuracy: 0.8631
11808/60000 [====>.........................] - ETA: 1:32 - loss: 0.4221 - categorical_accuracy: 0.8632
11840/60000 [====>.........................] - ETA: 1:32 - loss: 0.4213 - categorical_accuracy: 0.8635
11872/60000 [====>.........................] - ETA: 1:32 - loss: 0.4209 - categorical_accuracy: 0.8636
11904/60000 [====>.........................] - ETA: 1:32 - loss: 0.4200 - categorical_accuracy: 0.8640
11936/60000 [====>.........................] - ETA: 1:32 - loss: 0.4190 - categorical_accuracy: 0.8643
11968/60000 [====>.........................] - ETA: 1:32 - loss: 0.4179 - categorical_accuracy: 0.8646
12000/60000 [=====>........................] - ETA: 1:32 - loss: 0.4169 - categorical_accuracy: 0.8650
12032/60000 [=====>........................] - ETA: 1:32 - loss: 0.4164 - categorical_accuracy: 0.8652
12064/60000 [=====>........................] - ETA: 1:31 - loss: 0.4155 - categorical_accuracy: 0.8656
12096/60000 [=====>........................] - ETA: 1:31 - loss: 0.4148 - categorical_accuracy: 0.8658
12128/60000 [=====>........................] - ETA: 1:31 - loss: 0.4139 - categorical_accuracy: 0.8661
12160/60000 [=====>........................] - ETA: 1:31 - loss: 0.4131 - categorical_accuracy: 0.8663
12192/60000 [=====>........................] - ETA: 1:31 - loss: 0.4123 - categorical_accuracy: 0.8666
12224/60000 [=====>........................] - ETA: 1:31 - loss: 0.4125 - categorical_accuracy: 0.8667
12256/60000 [=====>........................] - ETA: 1:31 - loss: 0.4119 - categorical_accuracy: 0.8668
12288/60000 [=====>........................] - ETA: 1:31 - loss: 0.4112 - categorical_accuracy: 0.8671
12320/60000 [=====>........................] - ETA: 1:31 - loss: 0.4105 - categorical_accuracy: 0.8673
12352/60000 [=====>........................] - ETA: 1:31 - loss: 0.4095 - categorical_accuracy: 0.8676
12384/60000 [=====>........................] - ETA: 1:31 - loss: 0.4085 - categorical_accuracy: 0.8680
12416/60000 [=====>........................] - ETA: 1:31 - loss: 0.4079 - categorical_accuracy: 0.8682
12448/60000 [=====>........................] - ETA: 1:31 - loss: 0.4072 - categorical_accuracy: 0.8683
12480/60000 [=====>........................] - ETA: 1:31 - loss: 0.4069 - categorical_accuracy: 0.8685
12512/60000 [=====>........................] - ETA: 1:31 - loss: 0.4068 - categorical_accuracy: 0.8687
12544/60000 [=====>........................] - ETA: 1:31 - loss: 0.4065 - categorical_accuracy: 0.8688
12576/60000 [=====>........................] - ETA: 1:31 - loss: 0.4059 - categorical_accuracy: 0.8690
12608/60000 [=====>........................] - ETA: 1:30 - loss: 0.4053 - categorical_accuracy: 0.8691
12640/60000 [=====>........................] - ETA: 1:30 - loss: 0.4049 - categorical_accuracy: 0.8693
12672/60000 [=====>........................] - ETA: 1:30 - loss: 0.4044 - categorical_accuracy: 0.8695
12704/60000 [=====>........................] - ETA: 1:30 - loss: 0.4034 - categorical_accuracy: 0.8698
12736/60000 [=====>........................] - ETA: 1:30 - loss: 0.4029 - categorical_accuracy: 0.8701
12768/60000 [=====>........................] - ETA: 1:30 - loss: 0.4020 - categorical_accuracy: 0.8703
12800/60000 [=====>........................] - ETA: 1:30 - loss: 0.4012 - categorical_accuracy: 0.8705
12832/60000 [=====>........................] - ETA: 1:30 - loss: 0.4003 - categorical_accuracy: 0.8708
12864/60000 [=====>........................] - ETA: 1:30 - loss: 0.3999 - categorical_accuracy: 0.8710
12896/60000 [=====>........................] - ETA: 1:30 - loss: 0.3992 - categorical_accuracy: 0.8711
12928/60000 [=====>........................] - ETA: 1:30 - loss: 0.3983 - categorical_accuracy: 0.8714
12960/60000 [=====>........................] - ETA: 1:30 - loss: 0.3974 - categorical_accuracy: 0.8718
12992/60000 [=====>........................] - ETA: 1:30 - loss: 0.3968 - categorical_accuracy: 0.8719
13024/60000 [=====>........................] - ETA: 1:30 - loss: 0.3966 - categorical_accuracy: 0.8720
13056/60000 [=====>........................] - ETA: 1:30 - loss: 0.3960 - categorical_accuracy: 0.8722
13088/60000 [=====>........................] - ETA: 1:30 - loss: 0.3954 - categorical_accuracy: 0.8722
13120/60000 [=====>........................] - ETA: 1:29 - loss: 0.3953 - categorical_accuracy: 0.8724
13152/60000 [=====>........................] - ETA: 1:29 - loss: 0.3946 - categorical_accuracy: 0.8726
13184/60000 [=====>........................] - ETA: 1:29 - loss: 0.3938 - categorical_accuracy: 0.8729
13216/60000 [=====>........................] - ETA: 1:29 - loss: 0.3934 - categorical_accuracy: 0.8730
13248/60000 [=====>........................] - ETA: 1:29 - loss: 0.3926 - categorical_accuracy: 0.8733
13280/60000 [=====>........................] - ETA: 1:29 - loss: 0.3917 - categorical_accuracy: 0.8736
13312/60000 [=====>........................] - ETA: 1:29 - loss: 0.3914 - categorical_accuracy: 0.8737
13344/60000 [=====>........................] - ETA: 1:29 - loss: 0.3906 - categorical_accuracy: 0.8740
13376/60000 [=====>........................] - ETA: 1:29 - loss: 0.3898 - categorical_accuracy: 0.8743
13408/60000 [=====>........................] - ETA: 1:29 - loss: 0.3889 - categorical_accuracy: 0.8746
13440/60000 [=====>........................] - ETA: 1:29 - loss: 0.3882 - categorical_accuracy: 0.8749
13472/60000 [=====>........................] - ETA: 1:29 - loss: 0.3881 - categorical_accuracy: 0.8750
13504/60000 [=====>........................] - ETA: 1:29 - loss: 0.3882 - categorical_accuracy: 0.8751
13536/60000 [=====>........................] - ETA: 1:29 - loss: 0.3877 - categorical_accuracy: 0.8752
13568/60000 [=====>........................] - ETA: 1:29 - loss: 0.3871 - categorical_accuracy: 0.8754
13600/60000 [=====>........................] - ETA: 1:28 - loss: 0.3866 - categorical_accuracy: 0.8755
13632/60000 [=====>........................] - ETA: 1:28 - loss: 0.3860 - categorical_accuracy: 0.8757
13664/60000 [=====>........................] - ETA: 1:28 - loss: 0.3854 - categorical_accuracy: 0.8760
13696/60000 [=====>........................] - ETA: 1:28 - loss: 0.3846 - categorical_accuracy: 0.8762
13728/60000 [=====>........................] - ETA: 1:28 - loss: 0.3839 - categorical_accuracy: 0.8765
13760/60000 [=====>........................] - ETA: 1:28 - loss: 0.3832 - categorical_accuracy: 0.8767
13792/60000 [=====>........................] - ETA: 1:28 - loss: 0.3831 - categorical_accuracy: 0.8766
13824/60000 [=====>........................] - ETA: 1:28 - loss: 0.3829 - categorical_accuracy: 0.8766
13856/60000 [=====>........................] - ETA: 1:28 - loss: 0.3831 - categorical_accuracy: 0.8766
13888/60000 [=====>........................] - ETA: 1:28 - loss: 0.3825 - categorical_accuracy: 0.8768
13920/60000 [=====>........................] - ETA: 1:28 - loss: 0.3821 - categorical_accuracy: 0.8769
13952/60000 [=====>........................] - ETA: 1:28 - loss: 0.3816 - categorical_accuracy: 0.8772
13984/60000 [=====>........................] - ETA: 1:28 - loss: 0.3809 - categorical_accuracy: 0.8774
14016/60000 [======>.......................] - ETA: 1:28 - loss: 0.3804 - categorical_accuracy: 0.8776
14048/60000 [======>.......................] - ETA: 1:28 - loss: 0.3806 - categorical_accuracy: 0.8776
14080/60000 [======>.......................] - ETA: 1:28 - loss: 0.3806 - categorical_accuracy: 0.8776
14112/60000 [======>.......................] - ETA: 1:27 - loss: 0.3800 - categorical_accuracy: 0.8778
14144/60000 [======>.......................] - ETA: 1:27 - loss: 0.3796 - categorical_accuracy: 0.8779
14176/60000 [======>.......................] - ETA: 1:27 - loss: 0.3789 - categorical_accuracy: 0.8782
14208/60000 [======>.......................] - ETA: 1:27 - loss: 0.3789 - categorical_accuracy: 0.8783
14240/60000 [======>.......................] - ETA: 1:27 - loss: 0.3782 - categorical_accuracy: 0.8786
14272/60000 [======>.......................] - ETA: 1:27 - loss: 0.3775 - categorical_accuracy: 0.8788
14304/60000 [======>.......................] - ETA: 1:27 - loss: 0.3769 - categorical_accuracy: 0.8790
14336/60000 [======>.......................] - ETA: 1:27 - loss: 0.3766 - categorical_accuracy: 0.8791
14368/60000 [======>.......................] - ETA: 1:27 - loss: 0.3763 - categorical_accuracy: 0.8792
14400/60000 [======>.......................] - ETA: 1:27 - loss: 0.3761 - categorical_accuracy: 0.8792
14432/60000 [======>.......................] - ETA: 1:27 - loss: 0.3754 - categorical_accuracy: 0.8794
14464/60000 [======>.......................] - ETA: 1:27 - loss: 0.3748 - categorical_accuracy: 0.8796
14496/60000 [======>.......................] - ETA: 1:27 - loss: 0.3742 - categorical_accuracy: 0.8796
14528/60000 [======>.......................] - ETA: 1:27 - loss: 0.3735 - categorical_accuracy: 0.8799
14560/60000 [======>.......................] - ETA: 1:27 - loss: 0.3730 - categorical_accuracy: 0.8800
14592/60000 [======>.......................] - ETA: 1:27 - loss: 0.3728 - categorical_accuracy: 0.8801
14624/60000 [======>.......................] - ETA: 1:26 - loss: 0.3722 - categorical_accuracy: 0.8803
14656/60000 [======>.......................] - ETA: 1:26 - loss: 0.3716 - categorical_accuracy: 0.8805
14688/60000 [======>.......................] - ETA: 1:26 - loss: 0.3708 - categorical_accuracy: 0.8807
14720/60000 [======>.......................] - ETA: 1:26 - loss: 0.3702 - categorical_accuracy: 0.8808
14752/60000 [======>.......................] - ETA: 1:26 - loss: 0.3694 - categorical_accuracy: 0.8811
14784/60000 [======>.......................] - ETA: 1:26 - loss: 0.3687 - categorical_accuracy: 0.8813
14816/60000 [======>.......................] - ETA: 1:26 - loss: 0.3683 - categorical_accuracy: 0.8814
14848/60000 [======>.......................] - ETA: 1:26 - loss: 0.3677 - categorical_accuracy: 0.8816
14880/60000 [======>.......................] - ETA: 1:26 - loss: 0.3674 - categorical_accuracy: 0.8817
14912/60000 [======>.......................] - ETA: 1:26 - loss: 0.3671 - categorical_accuracy: 0.8819
14944/60000 [======>.......................] - ETA: 1:26 - loss: 0.3665 - categorical_accuracy: 0.8821
14976/60000 [======>.......................] - ETA: 1:26 - loss: 0.3663 - categorical_accuracy: 0.8821
15008/60000 [======>.......................] - ETA: 1:26 - loss: 0.3657 - categorical_accuracy: 0.8823
15040/60000 [======>.......................] - ETA: 1:26 - loss: 0.3649 - categorical_accuracy: 0.8825
15072/60000 [======>.......................] - ETA: 1:26 - loss: 0.3648 - categorical_accuracy: 0.8825
15104/60000 [======>.......................] - ETA: 1:26 - loss: 0.3647 - categorical_accuracy: 0.8826
15136/60000 [======>.......................] - ETA: 1:25 - loss: 0.3644 - categorical_accuracy: 0.8826
15168/60000 [======>.......................] - ETA: 1:25 - loss: 0.3638 - categorical_accuracy: 0.8828
15200/60000 [======>.......................] - ETA: 1:25 - loss: 0.3636 - categorical_accuracy: 0.8829
15232/60000 [======>.......................] - ETA: 1:25 - loss: 0.3633 - categorical_accuracy: 0.8831
15264/60000 [======>.......................] - ETA: 1:25 - loss: 0.3628 - categorical_accuracy: 0.8833
15296/60000 [======>.......................] - ETA: 1:25 - loss: 0.3622 - categorical_accuracy: 0.8834
15328/60000 [======>.......................] - ETA: 1:25 - loss: 0.3621 - categorical_accuracy: 0.8835
15360/60000 [======>.......................] - ETA: 1:25 - loss: 0.3620 - categorical_accuracy: 0.8837
15392/60000 [======>.......................] - ETA: 1:25 - loss: 0.3614 - categorical_accuracy: 0.8839
15424/60000 [======>.......................] - ETA: 1:25 - loss: 0.3613 - categorical_accuracy: 0.8841
15456/60000 [======>.......................] - ETA: 1:25 - loss: 0.3606 - categorical_accuracy: 0.8843
15488/60000 [======>.......................] - ETA: 1:25 - loss: 0.3602 - categorical_accuracy: 0.8844
15520/60000 [======>.......................] - ETA: 1:25 - loss: 0.3598 - categorical_accuracy: 0.8846
15552/60000 [======>.......................] - ETA: 1:25 - loss: 0.3594 - categorical_accuracy: 0.8847
15584/60000 [======>.......................] - ETA: 1:25 - loss: 0.3588 - categorical_accuracy: 0.8849
15616/60000 [======>.......................] - ETA: 1:24 - loss: 0.3584 - categorical_accuracy: 0.8851
15648/60000 [======>.......................] - ETA: 1:24 - loss: 0.3578 - categorical_accuracy: 0.8852
15680/60000 [======>.......................] - ETA: 1:24 - loss: 0.3572 - categorical_accuracy: 0.8855
15712/60000 [======>.......................] - ETA: 1:24 - loss: 0.3566 - categorical_accuracy: 0.8856
15744/60000 [======>.......................] - ETA: 1:24 - loss: 0.3560 - categorical_accuracy: 0.8859
15776/60000 [======>.......................] - ETA: 1:24 - loss: 0.3558 - categorical_accuracy: 0.8860
15808/60000 [======>.......................] - ETA: 1:24 - loss: 0.3558 - categorical_accuracy: 0.8860
15840/60000 [======>.......................] - ETA: 1:24 - loss: 0.3562 - categorical_accuracy: 0.8860
15872/60000 [======>.......................] - ETA: 1:24 - loss: 0.3560 - categorical_accuracy: 0.8862
15904/60000 [======>.......................] - ETA: 1:24 - loss: 0.3559 - categorical_accuracy: 0.8861
15936/60000 [======>.......................] - ETA: 1:24 - loss: 0.3553 - categorical_accuracy: 0.8864
15968/60000 [======>.......................] - ETA: 1:24 - loss: 0.3553 - categorical_accuracy: 0.8863
16000/60000 [=======>......................] - ETA: 1:24 - loss: 0.3551 - categorical_accuracy: 0.8865
16032/60000 [=======>......................] - ETA: 1:24 - loss: 0.3547 - categorical_accuracy: 0.8866
16064/60000 [=======>......................] - ETA: 1:24 - loss: 0.3541 - categorical_accuracy: 0.8868
16096/60000 [=======>......................] - ETA: 1:24 - loss: 0.3538 - categorical_accuracy: 0.8869
16128/60000 [=======>......................] - ETA: 1:23 - loss: 0.3533 - categorical_accuracy: 0.8872
16160/60000 [=======>......................] - ETA: 1:23 - loss: 0.3530 - categorical_accuracy: 0.8873
16192/60000 [=======>......................] - ETA: 1:23 - loss: 0.3526 - categorical_accuracy: 0.8874
16224/60000 [=======>......................] - ETA: 1:23 - loss: 0.3521 - categorical_accuracy: 0.8875
16256/60000 [=======>......................] - ETA: 1:23 - loss: 0.3516 - categorical_accuracy: 0.8877
16288/60000 [=======>......................] - ETA: 1:23 - loss: 0.3518 - categorical_accuracy: 0.8877
16320/60000 [=======>......................] - ETA: 1:23 - loss: 0.3513 - categorical_accuracy: 0.8879
16352/60000 [=======>......................] - ETA: 1:23 - loss: 0.3508 - categorical_accuracy: 0.8881
16384/60000 [=======>......................] - ETA: 1:23 - loss: 0.3505 - categorical_accuracy: 0.8882
16416/60000 [=======>......................] - ETA: 1:23 - loss: 0.3500 - categorical_accuracy: 0.8883
16448/60000 [=======>......................] - ETA: 1:23 - loss: 0.3499 - categorical_accuracy: 0.8883
16480/60000 [=======>......................] - ETA: 1:23 - loss: 0.3493 - categorical_accuracy: 0.8885
16512/60000 [=======>......................] - ETA: 1:23 - loss: 0.3488 - categorical_accuracy: 0.8886
16544/60000 [=======>......................] - ETA: 1:23 - loss: 0.3485 - categorical_accuracy: 0.8887
16576/60000 [=======>......................] - ETA: 1:23 - loss: 0.3478 - categorical_accuracy: 0.8889
16608/60000 [=======>......................] - ETA: 1:22 - loss: 0.3473 - categorical_accuracy: 0.8891
16640/60000 [=======>......................] - ETA: 1:22 - loss: 0.3468 - categorical_accuracy: 0.8892
16672/60000 [=======>......................] - ETA: 1:22 - loss: 0.3462 - categorical_accuracy: 0.8895
16704/60000 [=======>......................] - ETA: 1:22 - loss: 0.3463 - categorical_accuracy: 0.8895
16736/60000 [=======>......................] - ETA: 1:22 - loss: 0.3457 - categorical_accuracy: 0.8897
16768/60000 [=======>......................] - ETA: 1:22 - loss: 0.3452 - categorical_accuracy: 0.8899
16800/60000 [=======>......................] - ETA: 1:22 - loss: 0.3446 - categorical_accuracy: 0.8901
16832/60000 [=======>......................] - ETA: 1:22 - loss: 0.3445 - categorical_accuracy: 0.8902
16864/60000 [=======>......................] - ETA: 1:22 - loss: 0.3442 - categorical_accuracy: 0.8904
16896/60000 [=======>......................] - ETA: 1:22 - loss: 0.3437 - categorical_accuracy: 0.8905
16928/60000 [=======>......................] - ETA: 1:22 - loss: 0.3432 - categorical_accuracy: 0.8907
16960/60000 [=======>......................] - ETA: 1:22 - loss: 0.3426 - categorical_accuracy: 0.8909
16992/60000 [=======>......................] - ETA: 1:22 - loss: 0.3422 - categorical_accuracy: 0.8909
17024/60000 [=======>......................] - ETA: 1:22 - loss: 0.3422 - categorical_accuracy: 0.8910
17056/60000 [=======>......................] - ETA: 1:22 - loss: 0.3419 - categorical_accuracy: 0.8911
17088/60000 [=======>......................] - ETA: 1:21 - loss: 0.3414 - categorical_accuracy: 0.8913
17120/60000 [=======>......................] - ETA: 1:21 - loss: 0.3410 - categorical_accuracy: 0.8914
17152/60000 [=======>......................] - ETA: 1:21 - loss: 0.3406 - categorical_accuracy: 0.8915
17184/60000 [=======>......................] - ETA: 1:21 - loss: 0.3407 - categorical_accuracy: 0.8916
17216/60000 [=======>......................] - ETA: 1:21 - loss: 0.3409 - categorical_accuracy: 0.8916
17248/60000 [=======>......................] - ETA: 1:21 - loss: 0.3405 - categorical_accuracy: 0.8916
17280/60000 [=======>......................] - ETA: 1:21 - loss: 0.3404 - categorical_accuracy: 0.8917
17312/60000 [=======>......................] - ETA: 1:21 - loss: 0.3404 - categorical_accuracy: 0.8918
17344/60000 [=======>......................] - ETA: 1:21 - loss: 0.3407 - categorical_accuracy: 0.8915
17376/60000 [=======>......................] - ETA: 1:21 - loss: 0.3403 - categorical_accuracy: 0.8917
17408/60000 [=======>......................] - ETA: 1:21 - loss: 0.3398 - categorical_accuracy: 0.8919
17440/60000 [=======>......................] - ETA: 1:21 - loss: 0.3395 - categorical_accuracy: 0.8920
17472/60000 [=======>......................] - ETA: 1:21 - loss: 0.3396 - categorical_accuracy: 0.8921
17504/60000 [=======>......................] - ETA: 1:21 - loss: 0.3390 - categorical_accuracy: 0.8923
17536/60000 [=======>......................] - ETA: 1:21 - loss: 0.3389 - categorical_accuracy: 0.8923
17568/60000 [=======>......................] - ETA: 1:21 - loss: 0.3384 - categorical_accuracy: 0.8925
17600/60000 [=======>......................] - ETA: 1:21 - loss: 0.3380 - categorical_accuracy: 0.8926
17632/60000 [=======>......................] - ETA: 1:20 - loss: 0.3375 - categorical_accuracy: 0.8928
17664/60000 [=======>......................] - ETA: 1:20 - loss: 0.3373 - categorical_accuracy: 0.8928
17696/60000 [=======>......................] - ETA: 1:20 - loss: 0.3371 - categorical_accuracy: 0.8928
17728/60000 [=======>......................] - ETA: 1:20 - loss: 0.3367 - categorical_accuracy: 0.8929
17760/60000 [=======>......................] - ETA: 1:20 - loss: 0.3363 - categorical_accuracy: 0.8930
17792/60000 [=======>......................] - ETA: 1:20 - loss: 0.3364 - categorical_accuracy: 0.8929
17824/60000 [=======>......................] - ETA: 1:20 - loss: 0.3358 - categorical_accuracy: 0.8931
17856/60000 [=======>......................] - ETA: 1:20 - loss: 0.3353 - categorical_accuracy: 0.8933
17888/60000 [=======>......................] - ETA: 1:20 - loss: 0.3349 - categorical_accuracy: 0.8934
17920/60000 [=======>......................] - ETA: 1:20 - loss: 0.3345 - categorical_accuracy: 0.8935
17952/60000 [=======>......................] - ETA: 1:20 - loss: 0.3339 - categorical_accuracy: 0.8937
17984/60000 [=======>......................] - ETA: 1:20 - loss: 0.3335 - categorical_accuracy: 0.8939
18016/60000 [========>.....................] - ETA: 1:20 - loss: 0.3335 - categorical_accuracy: 0.8938
18048/60000 [========>.....................] - ETA: 1:20 - loss: 0.3329 - categorical_accuracy: 0.8940
18080/60000 [========>.....................] - ETA: 1:20 - loss: 0.3328 - categorical_accuracy: 0.8941
18112/60000 [========>.....................] - ETA: 1:20 - loss: 0.3324 - categorical_accuracy: 0.8942
18144/60000 [========>.....................] - ETA: 1:19 - loss: 0.3323 - categorical_accuracy: 0.8943
18176/60000 [========>.....................] - ETA: 1:19 - loss: 0.3318 - categorical_accuracy: 0.8945
18208/60000 [========>.....................] - ETA: 1:19 - loss: 0.3317 - categorical_accuracy: 0.8946
18240/60000 [========>.....................] - ETA: 1:19 - loss: 0.3312 - categorical_accuracy: 0.8947
18272/60000 [========>.....................] - ETA: 1:19 - loss: 0.3308 - categorical_accuracy: 0.8949
18304/60000 [========>.....................] - ETA: 1:19 - loss: 0.3304 - categorical_accuracy: 0.8949
18336/60000 [========>.....................] - ETA: 1:19 - loss: 0.3301 - categorical_accuracy: 0.8950
18368/60000 [========>.....................] - ETA: 1:19 - loss: 0.3298 - categorical_accuracy: 0.8951
18400/60000 [========>.....................] - ETA: 1:19 - loss: 0.3303 - categorical_accuracy: 0.8950
18432/60000 [========>.....................] - ETA: 1:19 - loss: 0.3300 - categorical_accuracy: 0.8951
18464/60000 [========>.....................] - ETA: 1:19 - loss: 0.3297 - categorical_accuracy: 0.8951
18496/60000 [========>.....................] - ETA: 1:19 - loss: 0.3293 - categorical_accuracy: 0.8953
18528/60000 [========>.....................] - ETA: 1:19 - loss: 0.3289 - categorical_accuracy: 0.8953
18560/60000 [========>.....................] - ETA: 1:19 - loss: 0.3287 - categorical_accuracy: 0.8954
18592/60000 [========>.....................] - ETA: 1:19 - loss: 0.3282 - categorical_accuracy: 0.8955
18624/60000 [========>.....................] - ETA: 1:19 - loss: 0.3278 - categorical_accuracy: 0.8957
18656/60000 [========>.....................] - ETA: 1:18 - loss: 0.3273 - categorical_accuracy: 0.8958
18688/60000 [========>.....................] - ETA: 1:18 - loss: 0.3268 - categorical_accuracy: 0.8960
18720/60000 [========>.....................] - ETA: 1:18 - loss: 0.3266 - categorical_accuracy: 0.8960
18752/60000 [========>.....................] - ETA: 1:18 - loss: 0.3262 - categorical_accuracy: 0.8962
18784/60000 [========>.....................] - ETA: 1:18 - loss: 0.3262 - categorical_accuracy: 0.8962
18816/60000 [========>.....................] - ETA: 1:18 - loss: 0.3266 - categorical_accuracy: 0.8962
18848/60000 [========>.....................] - ETA: 1:18 - loss: 0.3262 - categorical_accuracy: 0.8963
18880/60000 [========>.....................] - ETA: 1:18 - loss: 0.3259 - categorical_accuracy: 0.8964
18912/60000 [========>.....................] - ETA: 1:18 - loss: 0.3255 - categorical_accuracy: 0.8965
18944/60000 [========>.....................] - ETA: 1:18 - loss: 0.3250 - categorical_accuracy: 0.8967
18976/60000 [========>.....................] - ETA: 1:18 - loss: 0.3246 - categorical_accuracy: 0.8968
19008/60000 [========>.....................] - ETA: 1:18 - loss: 0.3244 - categorical_accuracy: 0.8969
19040/60000 [========>.....................] - ETA: 1:18 - loss: 0.3239 - categorical_accuracy: 0.8971
19072/60000 [========>.....................] - ETA: 1:18 - loss: 0.3239 - categorical_accuracy: 0.8971
19104/60000 [========>.....................] - ETA: 1:18 - loss: 0.3236 - categorical_accuracy: 0.8971
19136/60000 [========>.....................] - ETA: 1:17 - loss: 0.3232 - categorical_accuracy: 0.8973
19168/60000 [========>.....................] - ETA: 1:17 - loss: 0.3229 - categorical_accuracy: 0.8974
19200/60000 [========>.....................] - ETA: 1:17 - loss: 0.3226 - categorical_accuracy: 0.8974
19232/60000 [========>.....................] - ETA: 1:17 - loss: 0.3225 - categorical_accuracy: 0.8975
19264/60000 [========>.....................] - ETA: 1:17 - loss: 0.3223 - categorical_accuracy: 0.8975
19296/60000 [========>.....................] - ETA: 1:17 - loss: 0.3218 - categorical_accuracy: 0.8977
19328/60000 [========>.....................] - ETA: 1:17 - loss: 0.3218 - categorical_accuracy: 0.8978
19360/60000 [========>.....................] - ETA: 1:17 - loss: 0.3215 - categorical_accuracy: 0.8978
19392/60000 [========>.....................] - ETA: 1:17 - loss: 0.3215 - categorical_accuracy: 0.8979
19424/60000 [========>.....................] - ETA: 1:17 - loss: 0.3214 - categorical_accuracy: 0.8980
19456/60000 [========>.....................] - ETA: 1:17 - loss: 0.3209 - categorical_accuracy: 0.8982
19488/60000 [========>.....................] - ETA: 1:17 - loss: 0.3205 - categorical_accuracy: 0.8983
19520/60000 [========>.....................] - ETA: 1:17 - loss: 0.3201 - categorical_accuracy: 0.8985
19552/60000 [========>.....................] - ETA: 1:17 - loss: 0.3198 - categorical_accuracy: 0.8986
19584/60000 [========>.....................] - ETA: 1:17 - loss: 0.3194 - categorical_accuracy: 0.8987
19616/60000 [========>.....................] - ETA: 1:17 - loss: 0.3191 - categorical_accuracy: 0.8989
19648/60000 [========>.....................] - ETA: 1:16 - loss: 0.3191 - categorical_accuracy: 0.8989
19680/60000 [========>.....................] - ETA: 1:16 - loss: 0.3186 - categorical_accuracy: 0.8990
19712/60000 [========>.....................] - ETA: 1:16 - loss: 0.3182 - categorical_accuracy: 0.8991
19744/60000 [========>.....................] - ETA: 1:16 - loss: 0.3179 - categorical_accuracy: 0.8993
19776/60000 [========>.....................] - ETA: 1:16 - loss: 0.3178 - categorical_accuracy: 0.8993
19808/60000 [========>.....................] - ETA: 1:16 - loss: 0.3177 - categorical_accuracy: 0.8993
19840/60000 [========>.....................] - ETA: 1:16 - loss: 0.3173 - categorical_accuracy: 0.8994
19872/60000 [========>.....................] - ETA: 1:16 - loss: 0.3172 - categorical_accuracy: 0.8994
19904/60000 [========>.....................] - ETA: 1:16 - loss: 0.3171 - categorical_accuracy: 0.8993
19936/60000 [========>.....................] - ETA: 1:16 - loss: 0.3167 - categorical_accuracy: 0.8994
19968/60000 [========>.....................] - ETA: 1:16 - loss: 0.3163 - categorical_accuracy: 0.8996
20000/60000 [=========>....................] - ETA: 1:16 - loss: 0.3162 - categorical_accuracy: 0.8996
20032/60000 [=========>....................] - ETA: 1:16 - loss: 0.3161 - categorical_accuracy: 0.8996
20064/60000 [=========>....................] - ETA: 1:16 - loss: 0.3159 - categorical_accuracy: 0.8996
20096/60000 [=========>....................] - ETA: 1:16 - loss: 0.3155 - categorical_accuracy: 0.8997
20128/60000 [=========>....................] - ETA: 1:16 - loss: 0.3151 - categorical_accuracy: 0.8998
20160/60000 [=========>....................] - ETA: 1:15 - loss: 0.3147 - categorical_accuracy: 0.9000
20192/60000 [=========>....................] - ETA: 1:15 - loss: 0.3147 - categorical_accuracy: 0.8999
20224/60000 [=========>....................] - ETA: 1:15 - loss: 0.3143 - categorical_accuracy: 0.9001
20256/60000 [=========>....................] - ETA: 1:15 - loss: 0.3138 - categorical_accuracy: 0.9002
20288/60000 [=========>....................] - ETA: 1:15 - loss: 0.3140 - categorical_accuracy: 0.9003
20320/60000 [=========>....................] - ETA: 1:15 - loss: 0.3135 - categorical_accuracy: 0.9004
20352/60000 [=========>....................] - ETA: 1:15 - loss: 0.3133 - categorical_accuracy: 0.9006
20384/60000 [=========>....................] - ETA: 1:15 - loss: 0.3132 - categorical_accuracy: 0.9006
20416/60000 [=========>....................] - ETA: 1:15 - loss: 0.3128 - categorical_accuracy: 0.9007
20448/60000 [=========>....................] - ETA: 1:15 - loss: 0.3124 - categorical_accuracy: 0.9008
20480/60000 [=========>....................] - ETA: 1:15 - loss: 0.3120 - categorical_accuracy: 0.9010
20512/60000 [=========>....................] - ETA: 1:15 - loss: 0.3117 - categorical_accuracy: 0.9011
20544/60000 [=========>....................] - ETA: 1:15 - loss: 0.3115 - categorical_accuracy: 0.9012
20576/60000 [=========>....................] - ETA: 1:15 - loss: 0.3111 - categorical_accuracy: 0.9013
20608/60000 [=========>....................] - ETA: 1:15 - loss: 0.3112 - categorical_accuracy: 0.9014
20640/60000 [=========>....................] - ETA: 1:15 - loss: 0.3108 - categorical_accuracy: 0.9015
20672/60000 [=========>....................] - ETA: 1:14 - loss: 0.3105 - categorical_accuracy: 0.9016
20704/60000 [=========>....................] - ETA: 1:14 - loss: 0.3104 - categorical_accuracy: 0.9016
20736/60000 [=========>....................] - ETA: 1:14 - loss: 0.3100 - categorical_accuracy: 0.9018
20768/60000 [=========>....................] - ETA: 1:14 - loss: 0.3096 - categorical_accuracy: 0.9019
20800/60000 [=========>....................] - ETA: 1:14 - loss: 0.3094 - categorical_accuracy: 0.9020
20832/60000 [=========>....................] - ETA: 1:14 - loss: 0.3097 - categorical_accuracy: 0.9020
20864/60000 [=========>....................] - ETA: 1:14 - loss: 0.3095 - categorical_accuracy: 0.9020
20896/60000 [=========>....................] - ETA: 1:14 - loss: 0.3093 - categorical_accuracy: 0.9021
20928/60000 [=========>....................] - ETA: 1:14 - loss: 0.3095 - categorical_accuracy: 0.9021
20960/60000 [=========>....................] - ETA: 1:14 - loss: 0.3093 - categorical_accuracy: 0.9021
20992/60000 [=========>....................] - ETA: 1:14 - loss: 0.3093 - categorical_accuracy: 0.9022
21024/60000 [=========>....................] - ETA: 1:14 - loss: 0.3089 - categorical_accuracy: 0.9023
21056/60000 [=========>....................] - ETA: 1:14 - loss: 0.3085 - categorical_accuracy: 0.9024
21088/60000 [=========>....................] - ETA: 1:14 - loss: 0.3081 - categorical_accuracy: 0.9026
21120/60000 [=========>....................] - ETA: 1:14 - loss: 0.3078 - categorical_accuracy: 0.9027
21152/60000 [=========>....................] - ETA: 1:14 - loss: 0.3074 - categorical_accuracy: 0.9028
21184/60000 [=========>....................] - ETA: 1:13 - loss: 0.3072 - categorical_accuracy: 0.9029
21216/60000 [=========>....................] - ETA: 1:13 - loss: 0.3070 - categorical_accuracy: 0.9029
21248/60000 [=========>....................] - ETA: 1:13 - loss: 0.3068 - categorical_accuracy: 0.9029
21280/60000 [=========>....................] - ETA: 1:13 - loss: 0.3068 - categorical_accuracy: 0.9030
21312/60000 [=========>....................] - ETA: 1:13 - loss: 0.3065 - categorical_accuracy: 0.9031
21344/60000 [=========>....................] - ETA: 1:13 - loss: 0.3062 - categorical_accuracy: 0.9032
21376/60000 [=========>....................] - ETA: 1:13 - loss: 0.3058 - categorical_accuracy: 0.9033
21408/60000 [=========>....................] - ETA: 1:13 - loss: 0.3057 - categorical_accuracy: 0.9034
21440/60000 [=========>....................] - ETA: 1:13 - loss: 0.3053 - categorical_accuracy: 0.9035
21472/60000 [=========>....................] - ETA: 1:13 - loss: 0.3050 - categorical_accuracy: 0.9035
21504/60000 [=========>....................] - ETA: 1:13 - loss: 0.3050 - categorical_accuracy: 0.9036
21536/60000 [=========>....................] - ETA: 1:13 - loss: 0.3046 - categorical_accuracy: 0.9037
21568/60000 [=========>....................] - ETA: 1:13 - loss: 0.3045 - categorical_accuracy: 0.9037
21600/60000 [=========>....................] - ETA: 1:13 - loss: 0.3042 - categorical_accuracy: 0.9038
21632/60000 [=========>....................] - ETA: 1:13 - loss: 0.3041 - categorical_accuracy: 0.9038
21664/60000 [=========>....................] - ETA: 1:13 - loss: 0.3037 - categorical_accuracy: 0.9040
21696/60000 [=========>....................] - ETA: 1:12 - loss: 0.3034 - categorical_accuracy: 0.9041
21728/60000 [=========>....................] - ETA: 1:12 - loss: 0.3031 - categorical_accuracy: 0.9042
21760/60000 [=========>....................] - ETA: 1:12 - loss: 0.3027 - categorical_accuracy: 0.9043
21792/60000 [=========>....................] - ETA: 1:12 - loss: 0.3025 - categorical_accuracy: 0.9044
21824/60000 [=========>....................] - ETA: 1:12 - loss: 0.3023 - categorical_accuracy: 0.9045
21856/60000 [=========>....................] - ETA: 1:12 - loss: 0.3019 - categorical_accuracy: 0.9046
21888/60000 [=========>....................] - ETA: 1:12 - loss: 0.3016 - categorical_accuracy: 0.9046
21920/60000 [=========>....................] - ETA: 1:12 - loss: 0.3015 - categorical_accuracy: 0.9047
21952/60000 [=========>....................] - ETA: 1:12 - loss: 0.3015 - categorical_accuracy: 0.9047
21984/60000 [=========>....................] - ETA: 1:12 - loss: 0.3015 - categorical_accuracy: 0.9047
22016/60000 [==========>...................] - ETA: 1:12 - loss: 0.3012 - categorical_accuracy: 0.9048
22048/60000 [==========>...................] - ETA: 1:12 - loss: 0.3011 - categorical_accuracy: 0.9049
22080/60000 [==========>...................] - ETA: 1:12 - loss: 0.3008 - categorical_accuracy: 0.9050
22112/60000 [==========>...................] - ETA: 1:12 - loss: 0.3005 - categorical_accuracy: 0.9051
22144/60000 [==========>...................] - ETA: 1:12 - loss: 0.3001 - categorical_accuracy: 0.9053
22176/60000 [==========>...................] - ETA: 1:12 - loss: 0.3000 - categorical_accuracy: 0.9053
22208/60000 [==========>...................] - ETA: 1:11 - loss: 0.2999 - categorical_accuracy: 0.9053
22240/60000 [==========>...................] - ETA: 1:11 - loss: 0.2997 - categorical_accuracy: 0.9054
22272/60000 [==========>...................] - ETA: 1:11 - loss: 0.2996 - categorical_accuracy: 0.9054
22304/60000 [==========>...................] - ETA: 1:11 - loss: 0.2994 - categorical_accuracy: 0.9055
22336/60000 [==========>...................] - ETA: 1:11 - loss: 0.2991 - categorical_accuracy: 0.9056
22368/60000 [==========>...................] - ETA: 1:11 - loss: 0.2988 - categorical_accuracy: 0.9057
22400/60000 [==========>...................] - ETA: 1:11 - loss: 0.2987 - categorical_accuracy: 0.9057
22432/60000 [==========>...................] - ETA: 1:11 - loss: 0.2986 - categorical_accuracy: 0.9057
22464/60000 [==========>...................] - ETA: 1:11 - loss: 0.2983 - categorical_accuracy: 0.9058
22496/60000 [==========>...................] - ETA: 1:11 - loss: 0.2980 - categorical_accuracy: 0.9058
22528/60000 [==========>...................] - ETA: 1:11 - loss: 0.2979 - categorical_accuracy: 0.9059
22560/60000 [==========>...................] - ETA: 1:11 - loss: 0.2979 - categorical_accuracy: 0.9059
22592/60000 [==========>...................] - ETA: 1:11 - loss: 0.2975 - categorical_accuracy: 0.9060
22624/60000 [==========>...................] - ETA: 1:11 - loss: 0.2972 - categorical_accuracy: 0.9062
22656/60000 [==========>...................] - ETA: 1:11 - loss: 0.2969 - categorical_accuracy: 0.9062
22688/60000 [==========>...................] - ETA: 1:11 - loss: 0.2967 - categorical_accuracy: 0.9063
22720/60000 [==========>...................] - ETA: 1:11 - loss: 0.2965 - categorical_accuracy: 0.9064
22752/60000 [==========>...................] - ETA: 1:10 - loss: 0.2966 - categorical_accuracy: 0.9065
22784/60000 [==========>...................] - ETA: 1:10 - loss: 0.2963 - categorical_accuracy: 0.9066
22816/60000 [==========>...................] - ETA: 1:10 - loss: 0.2959 - categorical_accuracy: 0.9067
22848/60000 [==========>...................] - ETA: 1:10 - loss: 0.2956 - categorical_accuracy: 0.9068
22880/60000 [==========>...................] - ETA: 1:10 - loss: 0.2954 - categorical_accuracy: 0.9069
22912/60000 [==========>...................] - ETA: 1:10 - loss: 0.2950 - categorical_accuracy: 0.9070
22944/60000 [==========>...................] - ETA: 1:10 - loss: 0.2949 - categorical_accuracy: 0.9071
22976/60000 [==========>...................] - ETA: 1:10 - loss: 0.2945 - categorical_accuracy: 0.9073
23008/60000 [==========>...................] - ETA: 1:10 - loss: 0.2947 - categorical_accuracy: 0.9072
23040/60000 [==========>...................] - ETA: 1:10 - loss: 0.2944 - categorical_accuracy: 0.9073
23072/60000 [==========>...................] - ETA: 1:10 - loss: 0.2941 - categorical_accuracy: 0.9074
23104/60000 [==========>...................] - ETA: 1:10 - loss: 0.2937 - categorical_accuracy: 0.9075
23136/60000 [==========>...................] - ETA: 1:10 - loss: 0.2934 - categorical_accuracy: 0.9076
23168/60000 [==========>...................] - ETA: 1:10 - loss: 0.2930 - categorical_accuracy: 0.9078
23200/60000 [==========>...................] - ETA: 1:10 - loss: 0.2931 - categorical_accuracy: 0.9078
23232/60000 [==========>...................] - ETA: 1:10 - loss: 0.2927 - categorical_accuracy: 0.9079
23264/60000 [==========>...................] - ETA: 1:09 - loss: 0.2925 - categorical_accuracy: 0.9080
23296/60000 [==========>...................] - ETA: 1:09 - loss: 0.2923 - categorical_accuracy: 0.9080
23328/60000 [==========>...................] - ETA: 1:09 - loss: 0.2921 - categorical_accuracy: 0.9080
23360/60000 [==========>...................] - ETA: 1:09 - loss: 0.2918 - categorical_accuracy: 0.9081
23392/60000 [==========>...................] - ETA: 1:09 - loss: 0.2918 - categorical_accuracy: 0.9081
23424/60000 [==========>...................] - ETA: 1:09 - loss: 0.2918 - categorical_accuracy: 0.9082
23456/60000 [==========>...................] - ETA: 1:09 - loss: 0.2917 - categorical_accuracy: 0.9082
23488/60000 [==========>...................] - ETA: 1:09 - loss: 0.2916 - categorical_accuracy: 0.9083
23520/60000 [==========>...................] - ETA: 1:09 - loss: 0.2914 - categorical_accuracy: 0.9083
23552/60000 [==========>...................] - ETA: 1:09 - loss: 0.2913 - categorical_accuracy: 0.9084
23584/60000 [==========>...................] - ETA: 1:09 - loss: 0.2913 - categorical_accuracy: 0.9083
23616/60000 [==========>...................] - ETA: 1:09 - loss: 0.2911 - categorical_accuracy: 0.9084
23648/60000 [==========>...................] - ETA: 1:09 - loss: 0.2912 - categorical_accuracy: 0.9085
23680/60000 [==========>...................] - ETA: 1:09 - loss: 0.2910 - categorical_accuracy: 0.9086
23712/60000 [==========>...................] - ETA: 1:09 - loss: 0.2906 - categorical_accuracy: 0.9087
23744/60000 [==========>...................] - ETA: 1:08 - loss: 0.2904 - categorical_accuracy: 0.9087
23776/60000 [==========>...................] - ETA: 1:08 - loss: 0.2902 - categorical_accuracy: 0.9089
23808/60000 [==========>...................] - ETA: 1:08 - loss: 0.2899 - categorical_accuracy: 0.9089
23840/60000 [==========>...................] - ETA: 1:08 - loss: 0.2895 - categorical_accuracy: 0.9091
23872/60000 [==========>...................] - ETA: 1:08 - loss: 0.2894 - categorical_accuracy: 0.9091
23904/60000 [==========>...................] - ETA: 1:08 - loss: 0.2890 - categorical_accuracy: 0.9093
23936/60000 [==========>...................] - ETA: 1:08 - loss: 0.2887 - categorical_accuracy: 0.9093
23968/60000 [==========>...................] - ETA: 1:08 - loss: 0.2884 - categorical_accuracy: 0.9095
24000/60000 [===========>..................] - ETA: 1:08 - loss: 0.2885 - categorical_accuracy: 0.9095
24032/60000 [===========>..................] - ETA: 1:08 - loss: 0.2884 - categorical_accuracy: 0.9096
24064/60000 [===========>..................] - ETA: 1:08 - loss: 0.2882 - categorical_accuracy: 0.9097
24096/60000 [===========>..................] - ETA: 1:08 - loss: 0.2880 - categorical_accuracy: 0.9097
24128/60000 [===========>..................] - ETA: 1:08 - loss: 0.2877 - categorical_accuracy: 0.9098
24160/60000 [===========>..................] - ETA: 1:08 - loss: 0.2874 - categorical_accuracy: 0.9099
24192/60000 [===========>..................] - ETA: 1:08 - loss: 0.2871 - categorical_accuracy: 0.9100
24224/60000 [===========>..................] - ETA: 1:08 - loss: 0.2869 - categorical_accuracy: 0.9101
24256/60000 [===========>..................] - ETA: 1:08 - loss: 0.2865 - categorical_accuracy: 0.9102
24288/60000 [===========>..................] - ETA: 1:07 - loss: 0.2862 - categorical_accuracy: 0.9103
24320/60000 [===========>..................] - ETA: 1:07 - loss: 0.2859 - categorical_accuracy: 0.9104
24352/60000 [===========>..................] - ETA: 1:07 - loss: 0.2857 - categorical_accuracy: 0.9105
24384/60000 [===========>..................] - ETA: 1:07 - loss: 0.2859 - categorical_accuracy: 0.9106
24416/60000 [===========>..................] - ETA: 1:07 - loss: 0.2856 - categorical_accuracy: 0.9106
24448/60000 [===========>..................] - ETA: 1:07 - loss: 0.2853 - categorical_accuracy: 0.9107
24480/60000 [===========>..................] - ETA: 1:07 - loss: 0.2850 - categorical_accuracy: 0.9109
24512/60000 [===========>..................] - ETA: 1:07 - loss: 0.2847 - categorical_accuracy: 0.9110
24544/60000 [===========>..................] - ETA: 1:07 - loss: 0.2848 - categorical_accuracy: 0.9110
24576/60000 [===========>..................] - ETA: 1:07 - loss: 0.2848 - categorical_accuracy: 0.9110
24608/60000 [===========>..................] - ETA: 1:07 - loss: 0.2850 - categorical_accuracy: 0.9110
24640/60000 [===========>..................] - ETA: 1:07 - loss: 0.2851 - categorical_accuracy: 0.9110
24672/60000 [===========>..................] - ETA: 1:07 - loss: 0.2848 - categorical_accuracy: 0.9112
24704/60000 [===========>..................] - ETA: 1:07 - loss: 0.2847 - categorical_accuracy: 0.9112
24736/60000 [===========>..................] - ETA: 1:07 - loss: 0.2843 - categorical_accuracy: 0.9113
24768/60000 [===========>..................] - ETA: 1:06 - loss: 0.2840 - categorical_accuracy: 0.9114
24800/60000 [===========>..................] - ETA: 1:06 - loss: 0.2837 - categorical_accuracy: 0.9115
24832/60000 [===========>..................] - ETA: 1:06 - loss: 0.2836 - categorical_accuracy: 0.9116
24864/60000 [===========>..................] - ETA: 1:06 - loss: 0.2833 - categorical_accuracy: 0.9117
24896/60000 [===========>..................] - ETA: 1:06 - loss: 0.2832 - categorical_accuracy: 0.9118
24928/60000 [===========>..................] - ETA: 1:06 - loss: 0.2829 - categorical_accuracy: 0.9119
24960/60000 [===========>..................] - ETA: 1:06 - loss: 0.2827 - categorical_accuracy: 0.9119
24992/60000 [===========>..................] - ETA: 1:06 - loss: 0.2823 - categorical_accuracy: 0.9121
25024/60000 [===========>..................] - ETA: 1:06 - loss: 0.2825 - categorical_accuracy: 0.9121
25056/60000 [===========>..................] - ETA: 1:06 - loss: 0.2824 - categorical_accuracy: 0.9122
25088/60000 [===========>..................] - ETA: 1:06 - loss: 0.2821 - categorical_accuracy: 0.9123
25120/60000 [===========>..................] - ETA: 1:06 - loss: 0.2818 - categorical_accuracy: 0.9124
25152/60000 [===========>..................] - ETA: 1:06 - loss: 0.2815 - categorical_accuracy: 0.9125
25184/60000 [===========>..................] - ETA: 1:06 - loss: 0.2816 - categorical_accuracy: 0.9124
25216/60000 [===========>..................] - ETA: 1:06 - loss: 0.2813 - categorical_accuracy: 0.9125
25248/60000 [===========>..................] - ETA: 1:06 - loss: 0.2811 - categorical_accuracy: 0.9125
25280/60000 [===========>..................] - ETA: 1:06 - loss: 0.2808 - categorical_accuracy: 0.9127
25312/60000 [===========>..................] - ETA: 1:05 - loss: 0.2806 - categorical_accuracy: 0.9127
25344/60000 [===========>..................] - ETA: 1:05 - loss: 0.2804 - categorical_accuracy: 0.9128
25376/60000 [===========>..................] - ETA: 1:05 - loss: 0.2804 - categorical_accuracy: 0.9128
25408/60000 [===========>..................] - ETA: 1:05 - loss: 0.2802 - categorical_accuracy: 0.9128
25440/60000 [===========>..................] - ETA: 1:05 - loss: 0.2800 - categorical_accuracy: 0.9129
25472/60000 [===========>..................] - ETA: 1:05 - loss: 0.2798 - categorical_accuracy: 0.9129
25504/60000 [===========>..................] - ETA: 1:05 - loss: 0.2796 - categorical_accuracy: 0.9130
25536/60000 [===========>..................] - ETA: 1:05 - loss: 0.2794 - categorical_accuracy: 0.9130
25568/60000 [===========>..................] - ETA: 1:05 - loss: 0.2793 - categorical_accuracy: 0.9131
25600/60000 [===========>..................] - ETA: 1:05 - loss: 0.2790 - categorical_accuracy: 0.9132
25632/60000 [===========>..................] - ETA: 1:05 - loss: 0.2787 - categorical_accuracy: 0.9133
25664/60000 [===========>..................] - ETA: 1:05 - loss: 0.2786 - categorical_accuracy: 0.9133
25696/60000 [===========>..................] - ETA: 1:05 - loss: 0.2783 - categorical_accuracy: 0.9134
25728/60000 [===========>..................] - ETA: 1:05 - loss: 0.2780 - categorical_accuracy: 0.9135
25760/60000 [===========>..................] - ETA: 1:05 - loss: 0.2777 - categorical_accuracy: 0.9136
25792/60000 [===========>..................] - ETA: 1:05 - loss: 0.2775 - categorical_accuracy: 0.9136
25824/60000 [===========>..................] - ETA: 1:04 - loss: 0.2778 - categorical_accuracy: 0.9135
25856/60000 [===========>..................] - ETA: 1:04 - loss: 0.2775 - categorical_accuracy: 0.9136
25888/60000 [===========>..................] - ETA: 1:04 - loss: 0.2773 - categorical_accuracy: 0.9137
25920/60000 [===========>..................] - ETA: 1:04 - loss: 0.2771 - categorical_accuracy: 0.9138
25952/60000 [===========>..................] - ETA: 1:04 - loss: 0.2769 - categorical_accuracy: 0.9139
25984/60000 [===========>..................] - ETA: 1:04 - loss: 0.2769 - categorical_accuracy: 0.9139
26016/60000 [============>.................] - ETA: 1:04 - loss: 0.2766 - categorical_accuracy: 0.9140
26048/60000 [============>.................] - ETA: 1:04 - loss: 0.2764 - categorical_accuracy: 0.9141
26080/60000 [============>.................] - ETA: 1:04 - loss: 0.2763 - categorical_accuracy: 0.9141
26112/60000 [============>.................] - ETA: 1:04 - loss: 0.2761 - categorical_accuracy: 0.9142
26144/60000 [============>.................] - ETA: 1:04 - loss: 0.2762 - categorical_accuracy: 0.9142
26176/60000 [============>.................] - ETA: 1:04 - loss: 0.2759 - categorical_accuracy: 0.9143
26208/60000 [============>.................] - ETA: 1:04 - loss: 0.2757 - categorical_accuracy: 0.9144
26240/60000 [============>.................] - ETA: 1:04 - loss: 0.2757 - categorical_accuracy: 0.9144
26272/60000 [============>.................] - ETA: 1:04 - loss: 0.2755 - categorical_accuracy: 0.9145
26304/60000 [============>.................] - ETA: 1:04 - loss: 0.2753 - categorical_accuracy: 0.9146
26336/60000 [============>.................] - ETA: 1:03 - loss: 0.2753 - categorical_accuracy: 0.9146
26368/60000 [============>.................] - ETA: 1:03 - loss: 0.2751 - categorical_accuracy: 0.9147
26400/60000 [============>.................] - ETA: 1:03 - loss: 0.2750 - categorical_accuracy: 0.9147
26432/60000 [============>.................] - ETA: 1:03 - loss: 0.2747 - categorical_accuracy: 0.9148
26464/60000 [============>.................] - ETA: 1:03 - loss: 0.2745 - categorical_accuracy: 0.9149
26496/60000 [============>.................] - ETA: 1:03 - loss: 0.2743 - categorical_accuracy: 0.9149
26528/60000 [============>.................] - ETA: 1:03 - loss: 0.2742 - categorical_accuracy: 0.9150
26560/60000 [============>.................] - ETA: 1:03 - loss: 0.2739 - categorical_accuracy: 0.9150
26592/60000 [============>.................] - ETA: 1:03 - loss: 0.2737 - categorical_accuracy: 0.9151
26624/60000 [============>.................] - ETA: 1:03 - loss: 0.2735 - categorical_accuracy: 0.9152
26656/60000 [============>.................] - ETA: 1:03 - loss: 0.2732 - categorical_accuracy: 0.9153
26688/60000 [============>.................] - ETA: 1:03 - loss: 0.2732 - categorical_accuracy: 0.9152
26720/60000 [============>.................] - ETA: 1:03 - loss: 0.2731 - categorical_accuracy: 0.9153
26752/60000 [============>.................] - ETA: 1:03 - loss: 0.2730 - categorical_accuracy: 0.9153
26784/60000 [============>.................] - ETA: 1:03 - loss: 0.2728 - categorical_accuracy: 0.9154
26816/60000 [============>.................] - ETA: 1:03 - loss: 0.2730 - categorical_accuracy: 0.9154
26848/60000 [============>.................] - ETA: 1:02 - loss: 0.2727 - categorical_accuracy: 0.9154
26880/60000 [============>.................] - ETA: 1:02 - loss: 0.2725 - categorical_accuracy: 0.9156
26912/60000 [============>.................] - ETA: 1:02 - loss: 0.2723 - categorical_accuracy: 0.9156
26944/60000 [============>.................] - ETA: 1:02 - loss: 0.2721 - categorical_accuracy: 0.9156
26976/60000 [============>.................] - ETA: 1:02 - loss: 0.2719 - categorical_accuracy: 0.9157
27008/60000 [============>.................] - ETA: 1:02 - loss: 0.2716 - categorical_accuracy: 0.9158
27040/60000 [============>.................] - ETA: 1:02 - loss: 0.2714 - categorical_accuracy: 0.9159
27072/60000 [============>.................] - ETA: 1:02 - loss: 0.2711 - categorical_accuracy: 0.9159
27104/60000 [============>.................] - ETA: 1:02 - loss: 0.2709 - categorical_accuracy: 0.9160
27136/60000 [============>.................] - ETA: 1:02 - loss: 0.2706 - categorical_accuracy: 0.9161
27168/60000 [============>.................] - ETA: 1:02 - loss: 0.2704 - categorical_accuracy: 0.9162
27200/60000 [============>.................] - ETA: 1:02 - loss: 0.2703 - categorical_accuracy: 0.9162
27232/60000 [============>.................] - ETA: 1:02 - loss: 0.2703 - categorical_accuracy: 0.9162
27264/60000 [============>.................] - ETA: 1:02 - loss: 0.2702 - categorical_accuracy: 0.9162
27296/60000 [============>.................] - ETA: 1:02 - loss: 0.2700 - categorical_accuracy: 0.9163
27328/60000 [============>.................] - ETA: 1:02 - loss: 0.2700 - categorical_accuracy: 0.9163
27360/60000 [============>.................] - ETA: 1:02 - loss: 0.2697 - categorical_accuracy: 0.9163
27392/60000 [============>.................] - ETA: 1:01 - loss: 0.2696 - categorical_accuracy: 0.9164
27424/60000 [============>.................] - ETA: 1:01 - loss: 0.2694 - categorical_accuracy: 0.9165
27456/60000 [============>.................] - ETA: 1:01 - loss: 0.2692 - categorical_accuracy: 0.9165
27488/60000 [============>.................] - ETA: 1:01 - loss: 0.2689 - categorical_accuracy: 0.9166
27520/60000 [============>.................] - ETA: 1:01 - loss: 0.2688 - categorical_accuracy: 0.9166
27552/60000 [============>.................] - ETA: 1:01 - loss: 0.2685 - categorical_accuracy: 0.9167
27584/60000 [============>.................] - ETA: 1:01 - loss: 0.2682 - categorical_accuracy: 0.9168
27616/60000 [============>.................] - ETA: 1:01 - loss: 0.2680 - categorical_accuracy: 0.9169
27648/60000 [============>.................] - ETA: 1:01 - loss: 0.2677 - categorical_accuracy: 0.9170
27680/60000 [============>.................] - ETA: 1:01 - loss: 0.2675 - categorical_accuracy: 0.9171
27712/60000 [============>.................] - ETA: 1:01 - loss: 0.2673 - categorical_accuracy: 0.9171
27744/60000 [============>.................] - ETA: 1:01 - loss: 0.2670 - categorical_accuracy: 0.9172
27776/60000 [============>.................] - ETA: 1:01 - loss: 0.2667 - categorical_accuracy: 0.9173
27808/60000 [============>.................] - ETA: 1:01 - loss: 0.2665 - categorical_accuracy: 0.9174
27840/60000 [============>.................] - ETA: 1:01 - loss: 0.2662 - categorical_accuracy: 0.9175
27872/60000 [============>.................] - ETA: 1:01 - loss: 0.2661 - categorical_accuracy: 0.9175
27904/60000 [============>.................] - ETA: 1:00 - loss: 0.2660 - categorical_accuracy: 0.9176
27936/60000 [============>.................] - ETA: 1:00 - loss: 0.2658 - categorical_accuracy: 0.9176
27968/60000 [============>.................] - ETA: 1:00 - loss: 0.2657 - categorical_accuracy: 0.9177
28000/60000 [=============>................] - ETA: 1:00 - loss: 0.2656 - categorical_accuracy: 0.9177
28032/60000 [=============>................] - ETA: 1:00 - loss: 0.2656 - categorical_accuracy: 0.9177
28064/60000 [=============>................] - ETA: 1:00 - loss: 0.2655 - categorical_accuracy: 0.9177
28096/60000 [=============>................] - ETA: 1:00 - loss: 0.2654 - categorical_accuracy: 0.9177
28128/60000 [=============>................] - ETA: 1:00 - loss: 0.2652 - categorical_accuracy: 0.9178
28160/60000 [=============>................] - ETA: 1:00 - loss: 0.2652 - categorical_accuracy: 0.9178
28192/60000 [=============>................] - ETA: 1:00 - loss: 0.2649 - categorical_accuracy: 0.9178
28224/60000 [=============>................] - ETA: 1:00 - loss: 0.2647 - categorical_accuracy: 0.9179
28256/60000 [=============>................] - ETA: 1:00 - loss: 0.2645 - categorical_accuracy: 0.9180
28288/60000 [=============>................] - ETA: 1:00 - loss: 0.2643 - categorical_accuracy: 0.9181
28320/60000 [=============>................] - ETA: 1:00 - loss: 0.2642 - categorical_accuracy: 0.9180
28352/60000 [=============>................] - ETA: 1:00 - loss: 0.2639 - categorical_accuracy: 0.9181
28384/60000 [=============>................] - ETA: 1:00 - loss: 0.2637 - categorical_accuracy: 0.9182
28416/60000 [=============>................] - ETA: 59s - loss: 0.2635 - categorical_accuracy: 0.9182 
28448/60000 [=============>................] - ETA: 59s - loss: 0.2632 - categorical_accuracy: 0.9183
28480/60000 [=============>................] - ETA: 59s - loss: 0.2630 - categorical_accuracy: 0.9184
28512/60000 [=============>................] - ETA: 59s - loss: 0.2631 - categorical_accuracy: 0.9183
28544/60000 [=============>................] - ETA: 59s - loss: 0.2629 - categorical_accuracy: 0.9184
28576/60000 [=============>................] - ETA: 59s - loss: 0.2627 - categorical_accuracy: 0.9184
28608/60000 [=============>................] - ETA: 59s - loss: 0.2624 - categorical_accuracy: 0.9185
28640/60000 [=============>................] - ETA: 59s - loss: 0.2622 - categorical_accuracy: 0.9186
28672/60000 [=============>................] - ETA: 59s - loss: 0.2621 - categorical_accuracy: 0.9186
28704/60000 [=============>................] - ETA: 59s - loss: 0.2618 - categorical_accuracy: 0.9187
28736/60000 [=============>................] - ETA: 59s - loss: 0.2616 - categorical_accuracy: 0.9188
28768/60000 [=============>................] - ETA: 59s - loss: 0.2615 - categorical_accuracy: 0.9188
28800/60000 [=============>................] - ETA: 59s - loss: 0.2613 - categorical_accuracy: 0.9188
28832/60000 [=============>................] - ETA: 59s - loss: 0.2611 - categorical_accuracy: 0.9189
28864/60000 [=============>................] - ETA: 59s - loss: 0.2608 - categorical_accuracy: 0.9190
28896/60000 [=============>................] - ETA: 59s - loss: 0.2606 - categorical_accuracy: 0.9190
28928/60000 [=============>................] - ETA: 58s - loss: 0.2605 - categorical_accuracy: 0.9191
28960/60000 [=============>................] - ETA: 58s - loss: 0.2602 - categorical_accuracy: 0.9192
28992/60000 [=============>................] - ETA: 58s - loss: 0.2599 - categorical_accuracy: 0.9193
29024/60000 [=============>................] - ETA: 58s - loss: 0.2597 - categorical_accuracy: 0.9193
29056/60000 [=============>................] - ETA: 58s - loss: 0.2597 - categorical_accuracy: 0.9192
29088/60000 [=============>................] - ETA: 58s - loss: 0.2594 - categorical_accuracy: 0.9193
29120/60000 [=============>................] - ETA: 58s - loss: 0.2592 - categorical_accuracy: 0.9194
29152/60000 [=============>................] - ETA: 58s - loss: 0.2591 - categorical_accuracy: 0.9194
29184/60000 [=============>................] - ETA: 58s - loss: 0.2591 - categorical_accuracy: 0.9194
29216/60000 [=============>................] - ETA: 58s - loss: 0.2590 - categorical_accuracy: 0.9194
29248/60000 [=============>................] - ETA: 58s - loss: 0.2587 - categorical_accuracy: 0.9195
29280/60000 [=============>................] - ETA: 58s - loss: 0.2586 - categorical_accuracy: 0.9195
29312/60000 [=============>................] - ETA: 58s - loss: 0.2585 - categorical_accuracy: 0.9195
29344/60000 [=============>................] - ETA: 58s - loss: 0.2582 - categorical_accuracy: 0.9196
29376/60000 [=============>................] - ETA: 58s - loss: 0.2580 - categorical_accuracy: 0.9197
29408/60000 [=============>................] - ETA: 58s - loss: 0.2580 - categorical_accuracy: 0.9197
29440/60000 [=============>................] - ETA: 58s - loss: 0.2581 - categorical_accuracy: 0.9197
29472/60000 [=============>................] - ETA: 57s - loss: 0.2581 - categorical_accuracy: 0.9198
29504/60000 [=============>................] - ETA: 57s - loss: 0.2578 - categorical_accuracy: 0.9198
29536/60000 [=============>................] - ETA: 57s - loss: 0.2576 - categorical_accuracy: 0.9199
29568/60000 [=============>................] - ETA: 57s - loss: 0.2574 - categorical_accuracy: 0.9200
29600/60000 [=============>................] - ETA: 57s - loss: 0.2573 - categorical_accuracy: 0.9200
29632/60000 [=============>................] - ETA: 57s - loss: 0.2574 - categorical_accuracy: 0.9200
29664/60000 [=============>................] - ETA: 57s - loss: 0.2573 - categorical_accuracy: 0.9200
29696/60000 [=============>................] - ETA: 57s - loss: 0.2573 - categorical_accuracy: 0.9200
29728/60000 [=============>................] - ETA: 57s - loss: 0.2571 - categorical_accuracy: 0.9200
29760/60000 [=============>................] - ETA: 57s - loss: 0.2569 - categorical_accuracy: 0.9201
29792/60000 [=============>................] - ETA: 57s - loss: 0.2567 - categorical_accuracy: 0.9202
29824/60000 [=============>................] - ETA: 57s - loss: 0.2566 - categorical_accuracy: 0.9202
29856/60000 [=============>................] - ETA: 57s - loss: 0.2564 - categorical_accuracy: 0.9203
29888/60000 [=============>................] - ETA: 57s - loss: 0.2562 - categorical_accuracy: 0.9203
29920/60000 [=============>................] - ETA: 57s - loss: 0.2560 - categorical_accuracy: 0.9204
29952/60000 [=============>................] - ETA: 57s - loss: 0.2558 - categorical_accuracy: 0.9205
29984/60000 [=============>................] - ETA: 56s - loss: 0.2558 - categorical_accuracy: 0.9205
30016/60000 [==============>...............] - ETA: 56s - loss: 0.2555 - categorical_accuracy: 0.9206
30048/60000 [==============>...............] - ETA: 56s - loss: 0.2553 - categorical_accuracy: 0.9207
30080/60000 [==============>...............] - ETA: 56s - loss: 0.2552 - categorical_accuracy: 0.9207
30112/60000 [==============>...............] - ETA: 56s - loss: 0.2552 - categorical_accuracy: 0.9207
30144/60000 [==============>...............] - ETA: 56s - loss: 0.2550 - categorical_accuracy: 0.9207
30176/60000 [==============>...............] - ETA: 56s - loss: 0.2548 - categorical_accuracy: 0.9208
30208/60000 [==============>...............] - ETA: 56s - loss: 0.2547 - categorical_accuracy: 0.9208
30240/60000 [==============>...............] - ETA: 56s - loss: 0.2545 - categorical_accuracy: 0.9209
30272/60000 [==============>...............] - ETA: 56s - loss: 0.2543 - categorical_accuracy: 0.9209
30304/60000 [==============>...............] - ETA: 56s - loss: 0.2546 - categorical_accuracy: 0.9209
30336/60000 [==============>...............] - ETA: 56s - loss: 0.2544 - categorical_accuracy: 0.9210
30368/60000 [==============>...............] - ETA: 56s - loss: 0.2542 - categorical_accuracy: 0.9211
30400/60000 [==============>...............] - ETA: 56s - loss: 0.2541 - categorical_accuracy: 0.9211
30432/60000 [==============>...............] - ETA: 56s - loss: 0.2541 - categorical_accuracy: 0.9211
30464/60000 [==============>...............] - ETA: 56s - loss: 0.2539 - categorical_accuracy: 0.9212
30496/60000 [==============>...............] - ETA: 55s - loss: 0.2537 - categorical_accuracy: 0.9212
30528/60000 [==============>...............] - ETA: 55s - loss: 0.2537 - categorical_accuracy: 0.9213
30560/60000 [==============>...............] - ETA: 55s - loss: 0.2535 - categorical_accuracy: 0.9214
30592/60000 [==============>...............] - ETA: 55s - loss: 0.2534 - categorical_accuracy: 0.9214
30624/60000 [==============>...............] - ETA: 55s - loss: 0.2532 - categorical_accuracy: 0.9214
30656/60000 [==============>...............] - ETA: 55s - loss: 0.2533 - categorical_accuracy: 0.9214
30688/60000 [==============>...............] - ETA: 55s - loss: 0.2533 - categorical_accuracy: 0.9214
30720/60000 [==============>...............] - ETA: 55s - loss: 0.2531 - categorical_accuracy: 0.9215
30752/60000 [==============>...............] - ETA: 55s - loss: 0.2530 - categorical_accuracy: 0.9215
30784/60000 [==============>...............] - ETA: 55s - loss: 0.2529 - categorical_accuracy: 0.9215
30816/60000 [==============>...............] - ETA: 55s - loss: 0.2528 - categorical_accuracy: 0.9216
30848/60000 [==============>...............] - ETA: 55s - loss: 0.2526 - categorical_accuracy: 0.9216
30880/60000 [==============>...............] - ETA: 55s - loss: 0.2525 - categorical_accuracy: 0.9217
30912/60000 [==============>...............] - ETA: 55s - loss: 0.2525 - categorical_accuracy: 0.9216
30944/60000 [==============>...............] - ETA: 55s - loss: 0.2523 - categorical_accuracy: 0.9217
30976/60000 [==============>...............] - ETA: 55s - loss: 0.2521 - categorical_accuracy: 0.9217
31008/60000 [==============>...............] - ETA: 54s - loss: 0.2519 - categorical_accuracy: 0.9218
31040/60000 [==============>...............] - ETA: 54s - loss: 0.2517 - categorical_accuracy: 0.9219
31072/60000 [==============>...............] - ETA: 54s - loss: 0.2515 - categorical_accuracy: 0.9220
31104/60000 [==============>...............] - ETA: 54s - loss: 0.2515 - categorical_accuracy: 0.9220
31136/60000 [==============>...............] - ETA: 54s - loss: 0.2513 - categorical_accuracy: 0.9221
31168/60000 [==============>...............] - ETA: 54s - loss: 0.2511 - categorical_accuracy: 0.9221
31200/60000 [==============>...............] - ETA: 54s - loss: 0.2509 - categorical_accuracy: 0.9221
31232/60000 [==============>...............] - ETA: 54s - loss: 0.2508 - categorical_accuracy: 0.9222
31264/60000 [==============>...............] - ETA: 54s - loss: 0.2511 - categorical_accuracy: 0.9221
31296/60000 [==============>...............] - ETA: 54s - loss: 0.2509 - categorical_accuracy: 0.9222
31328/60000 [==============>...............] - ETA: 54s - loss: 0.2507 - categorical_accuracy: 0.9222
31360/60000 [==============>...............] - ETA: 54s - loss: 0.2506 - categorical_accuracy: 0.9223
31392/60000 [==============>...............] - ETA: 54s - loss: 0.2504 - categorical_accuracy: 0.9224
31424/60000 [==============>...............] - ETA: 54s - loss: 0.2502 - categorical_accuracy: 0.9224
31456/60000 [==============>...............] - ETA: 54s - loss: 0.2502 - categorical_accuracy: 0.9224
31488/60000 [==============>...............] - ETA: 54s - loss: 0.2503 - categorical_accuracy: 0.9224
31520/60000 [==============>...............] - ETA: 54s - loss: 0.2502 - categorical_accuracy: 0.9224
31552/60000 [==============>...............] - ETA: 53s - loss: 0.2500 - categorical_accuracy: 0.9225
31584/60000 [==============>...............] - ETA: 53s - loss: 0.2498 - categorical_accuracy: 0.9225
31616/60000 [==============>...............] - ETA: 53s - loss: 0.2497 - categorical_accuracy: 0.9225
31648/60000 [==============>...............] - ETA: 53s - loss: 0.2495 - categorical_accuracy: 0.9226
31680/60000 [==============>...............] - ETA: 53s - loss: 0.2494 - categorical_accuracy: 0.9226
31712/60000 [==============>...............] - ETA: 53s - loss: 0.2492 - categorical_accuracy: 0.9226
31744/60000 [==============>...............] - ETA: 53s - loss: 0.2495 - categorical_accuracy: 0.9226
31776/60000 [==============>...............] - ETA: 53s - loss: 0.2495 - categorical_accuracy: 0.9226
31808/60000 [==============>...............] - ETA: 53s - loss: 0.2495 - categorical_accuracy: 0.9226
31840/60000 [==============>...............] - ETA: 53s - loss: 0.2496 - categorical_accuracy: 0.9226
31872/60000 [==============>...............] - ETA: 53s - loss: 0.2496 - categorical_accuracy: 0.9226
31904/60000 [==============>...............] - ETA: 53s - loss: 0.2495 - categorical_accuracy: 0.9226
31936/60000 [==============>...............] - ETA: 53s - loss: 0.2493 - categorical_accuracy: 0.9227
31968/60000 [==============>...............] - ETA: 53s - loss: 0.2492 - categorical_accuracy: 0.9227
32000/60000 [===============>..............] - ETA: 53s - loss: 0.2490 - categorical_accuracy: 0.9228
32032/60000 [===============>..............] - ETA: 53s - loss: 0.2488 - categorical_accuracy: 0.9229
32064/60000 [===============>..............] - ETA: 52s - loss: 0.2486 - categorical_accuracy: 0.9230
32096/60000 [===============>..............] - ETA: 52s - loss: 0.2484 - categorical_accuracy: 0.9230
32128/60000 [===============>..............] - ETA: 52s - loss: 0.2483 - categorical_accuracy: 0.9230
32160/60000 [===============>..............] - ETA: 52s - loss: 0.2484 - categorical_accuracy: 0.9230
32192/60000 [===============>..............] - ETA: 52s - loss: 0.2482 - categorical_accuracy: 0.9231
32224/60000 [===============>..............] - ETA: 52s - loss: 0.2483 - categorical_accuracy: 0.9230
32256/60000 [===============>..............] - ETA: 52s - loss: 0.2482 - categorical_accuracy: 0.9231
32288/60000 [===============>..............] - ETA: 52s - loss: 0.2480 - categorical_accuracy: 0.9232
32320/60000 [===============>..............] - ETA: 52s - loss: 0.2479 - categorical_accuracy: 0.9232
32352/60000 [===============>..............] - ETA: 52s - loss: 0.2477 - categorical_accuracy: 0.9232
32384/60000 [===============>..............] - ETA: 52s - loss: 0.2478 - categorical_accuracy: 0.9232
32416/60000 [===============>..............] - ETA: 52s - loss: 0.2477 - categorical_accuracy: 0.9232
32448/60000 [===============>..............] - ETA: 52s - loss: 0.2476 - categorical_accuracy: 0.9232
32480/60000 [===============>..............] - ETA: 52s - loss: 0.2476 - categorical_accuracy: 0.9232
32512/60000 [===============>..............] - ETA: 52s - loss: 0.2475 - categorical_accuracy: 0.9232
32544/60000 [===============>..............] - ETA: 52s - loss: 0.2473 - categorical_accuracy: 0.9232
32576/60000 [===============>..............] - ETA: 51s - loss: 0.2471 - categorical_accuracy: 0.9233
32608/60000 [===============>..............] - ETA: 51s - loss: 0.2469 - categorical_accuracy: 0.9234
32640/60000 [===============>..............] - ETA: 51s - loss: 0.2470 - categorical_accuracy: 0.9234
32672/60000 [===============>..............] - ETA: 51s - loss: 0.2468 - categorical_accuracy: 0.9234
32704/60000 [===============>..............] - ETA: 51s - loss: 0.2469 - categorical_accuracy: 0.9234
32736/60000 [===============>..............] - ETA: 51s - loss: 0.2467 - categorical_accuracy: 0.9234
32768/60000 [===============>..............] - ETA: 51s - loss: 0.2465 - categorical_accuracy: 0.9235
32800/60000 [===============>..............] - ETA: 51s - loss: 0.2466 - categorical_accuracy: 0.9234
32832/60000 [===============>..............] - ETA: 51s - loss: 0.2464 - categorical_accuracy: 0.9235
32864/60000 [===============>..............] - ETA: 51s - loss: 0.2463 - categorical_accuracy: 0.9235
32896/60000 [===============>..............] - ETA: 51s - loss: 0.2461 - categorical_accuracy: 0.9235
32928/60000 [===============>..............] - ETA: 51s - loss: 0.2460 - categorical_accuracy: 0.9236
32960/60000 [===============>..............] - ETA: 51s - loss: 0.2458 - categorical_accuracy: 0.9236
32992/60000 [===============>..............] - ETA: 51s - loss: 0.2456 - categorical_accuracy: 0.9237
33024/60000 [===============>..............] - ETA: 51s - loss: 0.2454 - categorical_accuracy: 0.9238
33056/60000 [===============>..............] - ETA: 51s - loss: 0.2453 - categorical_accuracy: 0.9238
33088/60000 [===============>..............] - ETA: 51s - loss: 0.2452 - categorical_accuracy: 0.9239
33120/60000 [===============>..............] - ETA: 50s - loss: 0.2450 - categorical_accuracy: 0.9239
33152/60000 [===============>..............] - ETA: 50s - loss: 0.2450 - categorical_accuracy: 0.9239
33184/60000 [===============>..............] - ETA: 50s - loss: 0.2451 - categorical_accuracy: 0.9239
33216/60000 [===============>..............] - ETA: 50s - loss: 0.2450 - categorical_accuracy: 0.9240
33248/60000 [===============>..............] - ETA: 50s - loss: 0.2449 - categorical_accuracy: 0.9240
33280/60000 [===============>..............] - ETA: 50s - loss: 0.2447 - categorical_accuracy: 0.9241
33312/60000 [===============>..............] - ETA: 50s - loss: 0.2445 - categorical_accuracy: 0.9241
33344/60000 [===============>..............] - ETA: 50s - loss: 0.2443 - categorical_accuracy: 0.9242
33376/60000 [===============>..............] - ETA: 50s - loss: 0.2442 - categorical_accuracy: 0.9242
33408/60000 [===============>..............] - ETA: 50s - loss: 0.2442 - categorical_accuracy: 0.9242
33440/60000 [===============>..............] - ETA: 50s - loss: 0.2440 - categorical_accuracy: 0.9243
33472/60000 [===============>..............] - ETA: 50s - loss: 0.2437 - categorical_accuracy: 0.9244
33504/60000 [===============>..............] - ETA: 50s - loss: 0.2436 - categorical_accuracy: 0.9245
33536/60000 [===============>..............] - ETA: 50s - loss: 0.2434 - categorical_accuracy: 0.9245
33568/60000 [===============>..............] - ETA: 50s - loss: 0.2432 - categorical_accuracy: 0.9245
33600/60000 [===============>..............] - ETA: 50s - loss: 0.2430 - categorical_accuracy: 0.9246
33632/60000 [===============>..............] - ETA: 49s - loss: 0.2430 - categorical_accuracy: 0.9246
33664/60000 [===============>..............] - ETA: 49s - loss: 0.2433 - categorical_accuracy: 0.9245
33696/60000 [===============>..............] - ETA: 49s - loss: 0.2431 - categorical_accuracy: 0.9246
33728/60000 [===============>..............] - ETA: 49s - loss: 0.2432 - categorical_accuracy: 0.9246
33760/60000 [===============>..............] - ETA: 49s - loss: 0.2430 - categorical_accuracy: 0.9246
33792/60000 [===============>..............] - ETA: 49s - loss: 0.2429 - categorical_accuracy: 0.9247
33824/60000 [===============>..............] - ETA: 49s - loss: 0.2427 - categorical_accuracy: 0.9247
33856/60000 [===============>..............] - ETA: 49s - loss: 0.2425 - categorical_accuracy: 0.9248
33888/60000 [===============>..............] - ETA: 49s - loss: 0.2423 - categorical_accuracy: 0.9248
33920/60000 [===============>..............] - ETA: 49s - loss: 0.2422 - categorical_accuracy: 0.9249
33952/60000 [===============>..............] - ETA: 49s - loss: 0.2420 - categorical_accuracy: 0.9249
33984/60000 [===============>..............] - ETA: 49s - loss: 0.2419 - categorical_accuracy: 0.9250
34016/60000 [================>.............] - ETA: 49s - loss: 0.2417 - categorical_accuracy: 0.9250
34048/60000 [================>.............] - ETA: 49s - loss: 0.2417 - categorical_accuracy: 0.9250
34080/60000 [================>.............] - ETA: 49s - loss: 0.2415 - categorical_accuracy: 0.9251
34112/60000 [================>.............] - ETA: 49s - loss: 0.2414 - categorical_accuracy: 0.9251
34144/60000 [================>.............] - ETA: 48s - loss: 0.2413 - categorical_accuracy: 0.9251
34176/60000 [================>.............] - ETA: 48s - loss: 0.2414 - categorical_accuracy: 0.9251
34208/60000 [================>.............] - ETA: 48s - loss: 0.2412 - categorical_accuracy: 0.9252
34240/60000 [================>.............] - ETA: 48s - loss: 0.2409 - categorical_accuracy: 0.9252
34272/60000 [================>.............] - ETA: 48s - loss: 0.2408 - categorical_accuracy: 0.9253
34304/60000 [================>.............] - ETA: 48s - loss: 0.2406 - categorical_accuracy: 0.9254
34336/60000 [================>.............] - ETA: 48s - loss: 0.2404 - categorical_accuracy: 0.9254
34368/60000 [================>.............] - ETA: 48s - loss: 0.2403 - categorical_accuracy: 0.9255
34400/60000 [================>.............] - ETA: 48s - loss: 0.2402 - categorical_accuracy: 0.9255
34432/60000 [================>.............] - ETA: 48s - loss: 0.2400 - categorical_accuracy: 0.9255
34464/60000 [================>.............] - ETA: 48s - loss: 0.2399 - categorical_accuracy: 0.9256
34496/60000 [================>.............] - ETA: 48s - loss: 0.2397 - categorical_accuracy: 0.9256
34528/60000 [================>.............] - ETA: 48s - loss: 0.2396 - categorical_accuracy: 0.9257
34560/60000 [================>.............] - ETA: 48s - loss: 0.2394 - categorical_accuracy: 0.9257
34592/60000 [================>.............] - ETA: 48s - loss: 0.2393 - categorical_accuracy: 0.9257
34624/60000 [================>.............] - ETA: 48s - loss: 0.2391 - categorical_accuracy: 0.9258
34656/60000 [================>.............] - ETA: 48s - loss: 0.2389 - categorical_accuracy: 0.9259
34688/60000 [================>.............] - ETA: 47s - loss: 0.2389 - categorical_accuracy: 0.9259
34720/60000 [================>.............] - ETA: 47s - loss: 0.2388 - categorical_accuracy: 0.9259
34752/60000 [================>.............] - ETA: 47s - loss: 0.2386 - categorical_accuracy: 0.9260
34784/60000 [================>.............] - ETA: 47s - loss: 0.2384 - categorical_accuracy: 0.9260
34816/60000 [================>.............] - ETA: 47s - loss: 0.2382 - categorical_accuracy: 0.9261
34848/60000 [================>.............] - ETA: 47s - loss: 0.2381 - categorical_accuracy: 0.9261
34880/60000 [================>.............] - ETA: 47s - loss: 0.2379 - categorical_accuracy: 0.9261
34912/60000 [================>.............] - ETA: 47s - loss: 0.2378 - categorical_accuracy: 0.9262
34944/60000 [================>.............] - ETA: 47s - loss: 0.2376 - categorical_accuracy: 0.9262
34976/60000 [================>.............] - ETA: 47s - loss: 0.2374 - categorical_accuracy: 0.9263
35008/60000 [================>.............] - ETA: 47s - loss: 0.2372 - categorical_accuracy: 0.9264
35040/60000 [================>.............] - ETA: 47s - loss: 0.2373 - categorical_accuracy: 0.9263
35072/60000 [================>.............] - ETA: 47s - loss: 0.2371 - categorical_accuracy: 0.9263
35104/60000 [================>.............] - ETA: 47s - loss: 0.2369 - categorical_accuracy: 0.9264
35136/60000 [================>.............] - ETA: 47s - loss: 0.2368 - categorical_accuracy: 0.9264
35168/60000 [================>.............] - ETA: 47s - loss: 0.2369 - categorical_accuracy: 0.9264
35200/60000 [================>.............] - ETA: 46s - loss: 0.2370 - categorical_accuracy: 0.9264
35232/60000 [================>.............] - ETA: 46s - loss: 0.2369 - categorical_accuracy: 0.9265
35264/60000 [================>.............] - ETA: 46s - loss: 0.2368 - categorical_accuracy: 0.9265
35296/60000 [================>.............] - ETA: 46s - loss: 0.2366 - categorical_accuracy: 0.9265
35328/60000 [================>.............] - ETA: 46s - loss: 0.2366 - categorical_accuracy: 0.9265
35360/60000 [================>.............] - ETA: 46s - loss: 0.2367 - categorical_accuracy: 0.9265
35392/60000 [================>.............] - ETA: 46s - loss: 0.2367 - categorical_accuracy: 0.9266
35424/60000 [================>.............] - ETA: 46s - loss: 0.2367 - categorical_accuracy: 0.9265
35456/60000 [================>.............] - ETA: 46s - loss: 0.2366 - categorical_accuracy: 0.9266
35488/60000 [================>.............] - ETA: 46s - loss: 0.2365 - categorical_accuracy: 0.9266
35520/60000 [================>.............] - ETA: 46s - loss: 0.2363 - categorical_accuracy: 0.9267
35552/60000 [================>.............] - ETA: 46s - loss: 0.2362 - categorical_accuracy: 0.9267
35584/60000 [================>.............] - ETA: 46s - loss: 0.2360 - categorical_accuracy: 0.9268
35616/60000 [================>.............] - ETA: 46s - loss: 0.2360 - categorical_accuracy: 0.9267
35648/60000 [================>.............] - ETA: 46s - loss: 0.2358 - categorical_accuracy: 0.9268
35680/60000 [================>.............] - ETA: 46s - loss: 0.2358 - categorical_accuracy: 0.9268
35712/60000 [================>.............] - ETA: 45s - loss: 0.2357 - categorical_accuracy: 0.9268
35744/60000 [================>.............] - ETA: 45s - loss: 0.2356 - categorical_accuracy: 0.9269
35776/60000 [================>.............] - ETA: 45s - loss: 0.2354 - categorical_accuracy: 0.9269
35808/60000 [================>.............] - ETA: 45s - loss: 0.2353 - categorical_accuracy: 0.9270
35840/60000 [================>.............] - ETA: 45s - loss: 0.2352 - categorical_accuracy: 0.9270
35872/60000 [================>.............] - ETA: 45s - loss: 0.2351 - categorical_accuracy: 0.9270
35904/60000 [================>.............] - ETA: 45s - loss: 0.2350 - categorical_accuracy: 0.9270
35936/60000 [================>.............] - ETA: 45s - loss: 0.2349 - categorical_accuracy: 0.9271
35968/60000 [================>.............] - ETA: 45s - loss: 0.2347 - categorical_accuracy: 0.9271
36000/60000 [=================>............] - ETA: 45s - loss: 0.2347 - categorical_accuracy: 0.9272
36032/60000 [=================>............] - ETA: 45s - loss: 0.2347 - categorical_accuracy: 0.9271
36064/60000 [=================>............] - ETA: 45s - loss: 0.2345 - categorical_accuracy: 0.9272
36096/60000 [=================>............] - ETA: 45s - loss: 0.2344 - categorical_accuracy: 0.9272
36128/60000 [=================>............] - ETA: 45s - loss: 0.2343 - categorical_accuracy: 0.9272
36160/60000 [=================>............] - ETA: 45s - loss: 0.2343 - categorical_accuracy: 0.9272
36192/60000 [=================>............] - ETA: 45s - loss: 0.2342 - categorical_accuracy: 0.9272
36224/60000 [=================>............] - ETA: 44s - loss: 0.2341 - categorical_accuracy: 0.9272
36256/60000 [=================>............] - ETA: 44s - loss: 0.2340 - categorical_accuracy: 0.9272
36288/60000 [=================>............] - ETA: 44s - loss: 0.2339 - categorical_accuracy: 0.9272
36320/60000 [=================>............] - ETA: 44s - loss: 0.2339 - categorical_accuracy: 0.9272
36352/60000 [=================>............] - ETA: 44s - loss: 0.2339 - categorical_accuracy: 0.9272
36384/60000 [=================>............] - ETA: 44s - loss: 0.2339 - categorical_accuracy: 0.9272
36416/60000 [=================>............] - ETA: 44s - loss: 0.2339 - categorical_accuracy: 0.9272
36448/60000 [=================>............] - ETA: 44s - loss: 0.2339 - categorical_accuracy: 0.9272
36480/60000 [=================>............] - ETA: 44s - loss: 0.2337 - categorical_accuracy: 0.9272
36512/60000 [=================>............] - ETA: 44s - loss: 0.2335 - categorical_accuracy: 0.9273
36544/60000 [=================>............] - ETA: 44s - loss: 0.2334 - categorical_accuracy: 0.9274
36576/60000 [=================>............] - ETA: 44s - loss: 0.2334 - categorical_accuracy: 0.9274
36608/60000 [=================>............] - ETA: 44s - loss: 0.2333 - categorical_accuracy: 0.9274
36640/60000 [=================>............] - ETA: 44s - loss: 0.2331 - categorical_accuracy: 0.9275
36672/60000 [=================>............] - ETA: 44s - loss: 0.2330 - categorical_accuracy: 0.9275
36704/60000 [=================>............] - ETA: 44s - loss: 0.2329 - categorical_accuracy: 0.9275
36736/60000 [=================>............] - ETA: 44s - loss: 0.2328 - categorical_accuracy: 0.9276
36768/60000 [=================>............] - ETA: 43s - loss: 0.2327 - categorical_accuracy: 0.9276
36800/60000 [=================>............] - ETA: 43s - loss: 0.2326 - categorical_accuracy: 0.9276
36832/60000 [=================>............] - ETA: 43s - loss: 0.2326 - categorical_accuracy: 0.9276
36864/60000 [=================>............] - ETA: 43s - loss: 0.2325 - categorical_accuracy: 0.9276
36896/60000 [=================>............] - ETA: 43s - loss: 0.2325 - categorical_accuracy: 0.9276
36928/60000 [=================>............] - ETA: 43s - loss: 0.2324 - categorical_accuracy: 0.9277
36960/60000 [=================>............] - ETA: 43s - loss: 0.2322 - categorical_accuracy: 0.9277
36992/60000 [=================>............] - ETA: 43s - loss: 0.2321 - categorical_accuracy: 0.9278
37024/60000 [=================>............] - ETA: 43s - loss: 0.2319 - categorical_accuracy: 0.9278
37056/60000 [=================>............] - ETA: 43s - loss: 0.2318 - categorical_accuracy: 0.9279
37088/60000 [=================>............] - ETA: 43s - loss: 0.2317 - categorical_accuracy: 0.9279
37120/60000 [=================>............] - ETA: 43s - loss: 0.2318 - categorical_accuracy: 0.9279
37152/60000 [=================>............] - ETA: 43s - loss: 0.2316 - categorical_accuracy: 0.9280
37184/60000 [=================>............] - ETA: 43s - loss: 0.2315 - categorical_accuracy: 0.9280
37216/60000 [=================>............] - ETA: 43s - loss: 0.2313 - categorical_accuracy: 0.9281
37248/60000 [=================>............] - ETA: 43s - loss: 0.2312 - categorical_accuracy: 0.9281
37280/60000 [=================>............] - ETA: 42s - loss: 0.2310 - categorical_accuracy: 0.9282
37312/60000 [=================>............] - ETA: 42s - loss: 0.2308 - categorical_accuracy: 0.9282
37344/60000 [=================>............] - ETA: 42s - loss: 0.2307 - categorical_accuracy: 0.9283
37376/60000 [=================>............] - ETA: 42s - loss: 0.2306 - categorical_accuracy: 0.9283
37408/60000 [=================>............] - ETA: 42s - loss: 0.2304 - categorical_accuracy: 0.9283
37440/60000 [=================>............] - ETA: 42s - loss: 0.2303 - categorical_accuracy: 0.9284
37472/60000 [=================>............] - ETA: 42s - loss: 0.2302 - categorical_accuracy: 0.9283
37504/60000 [=================>............] - ETA: 42s - loss: 0.2300 - categorical_accuracy: 0.9284
37536/60000 [=================>............] - ETA: 42s - loss: 0.2298 - categorical_accuracy: 0.9285
37568/60000 [=================>............] - ETA: 42s - loss: 0.2297 - categorical_accuracy: 0.9285
37600/60000 [=================>............] - ETA: 42s - loss: 0.2296 - categorical_accuracy: 0.9285
37632/60000 [=================>............] - ETA: 42s - loss: 0.2294 - categorical_accuracy: 0.9286
37664/60000 [=================>............] - ETA: 42s - loss: 0.2295 - categorical_accuracy: 0.9286
37696/60000 [=================>............] - ETA: 42s - loss: 0.2293 - categorical_accuracy: 0.9286
37728/60000 [=================>............] - ETA: 42s - loss: 0.2291 - categorical_accuracy: 0.9287
37760/60000 [=================>............] - ETA: 42s - loss: 0.2290 - categorical_accuracy: 0.9287
37792/60000 [=================>............] - ETA: 42s - loss: 0.2288 - categorical_accuracy: 0.9288
37824/60000 [=================>............] - ETA: 41s - loss: 0.2288 - categorical_accuracy: 0.9288
37856/60000 [=================>............] - ETA: 41s - loss: 0.2286 - categorical_accuracy: 0.9288
37888/60000 [=================>............] - ETA: 41s - loss: 0.2286 - categorical_accuracy: 0.9289
37920/60000 [=================>............] - ETA: 41s - loss: 0.2284 - categorical_accuracy: 0.9289
37952/60000 [=================>............] - ETA: 41s - loss: 0.2283 - categorical_accuracy: 0.9289
37984/60000 [=================>............] - ETA: 41s - loss: 0.2281 - categorical_accuracy: 0.9290
38016/60000 [==================>...........] - ETA: 41s - loss: 0.2279 - categorical_accuracy: 0.9291
38048/60000 [==================>...........] - ETA: 41s - loss: 0.2278 - categorical_accuracy: 0.9291
38080/60000 [==================>...........] - ETA: 41s - loss: 0.2276 - categorical_accuracy: 0.9292
38112/60000 [==================>...........] - ETA: 41s - loss: 0.2275 - categorical_accuracy: 0.9292
38144/60000 [==================>...........] - ETA: 41s - loss: 0.2274 - categorical_accuracy: 0.9292
38176/60000 [==================>...........] - ETA: 41s - loss: 0.2272 - categorical_accuracy: 0.9292
38208/60000 [==================>...........] - ETA: 41s - loss: 0.2271 - categorical_accuracy: 0.9293
38240/60000 [==================>...........] - ETA: 41s - loss: 0.2270 - categorical_accuracy: 0.9293
38272/60000 [==================>...........] - ETA: 41s - loss: 0.2268 - categorical_accuracy: 0.9293
38304/60000 [==================>...........] - ETA: 41s - loss: 0.2269 - categorical_accuracy: 0.9294
38336/60000 [==================>...........] - ETA: 40s - loss: 0.2268 - categorical_accuracy: 0.9294
38368/60000 [==================>...........] - ETA: 40s - loss: 0.2266 - categorical_accuracy: 0.9295
38400/60000 [==================>...........] - ETA: 40s - loss: 0.2264 - categorical_accuracy: 0.9295
38432/60000 [==================>...........] - ETA: 40s - loss: 0.2263 - categorical_accuracy: 0.9296
38464/60000 [==================>...........] - ETA: 40s - loss: 0.2262 - categorical_accuracy: 0.9295
38496/60000 [==================>...........] - ETA: 40s - loss: 0.2260 - categorical_accuracy: 0.9296
38528/60000 [==================>...........] - ETA: 40s - loss: 0.2261 - categorical_accuracy: 0.9296
38560/60000 [==================>...........] - ETA: 40s - loss: 0.2259 - categorical_accuracy: 0.9296
38592/60000 [==================>...........] - ETA: 40s - loss: 0.2257 - categorical_accuracy: 0.9297
38624/60000 [==================>...........] - ETA: 40s - loss: 0.2256 - categorical_accuracy: 0.9298
38656/60000 [==================>...........] - ETA: 40s - loss: 0.2254 - categorical_accuracy: 0.9298
38688/60000 [==================>...........] - ETA: 40s - loss: 0.2252 - categorical_accuracy: 0.9298
38720/60000 [==================>...........] - ETA: 40s - loss: 0.2251 - categorical_accuracy: 0.9299
38752/60000 [==================>...........] - ETA: 40s - loss: 0.2250 - categorical_accuracy: 0.9299
38784/60000 [==================>...........] - ETA: 40s - loss: 0.2249 - categorical_accuracy: 0.9299
38816/60000 [==================>...........] - ETA: 40s - loss: 0.2248 - categorical_accuracy: 0.9300
38848/60000 [==================>...........] - ETA: 40s - loss: 0.2246 - categorical_accuracy: 0.9300
38880/60000 [==================>...........] - ETA: 39s - loss: 0.2245 - categorical_accuracy: 0.9301
38912/60000 [==================>...........] - ETA: 39s - loss: 0.2244 - categorical_accuracy: 0.9301
38944/60000 [==================>...........] - ETA: 39s - loss: 0.2242 - categorical_accuracy: 0.9301
38976/60000 [==================>...........] - ETA: 39s - loss: 0.2243 - categorical_accuracy: 0.9301
39008/60000 [==================>...........] - ETA: 39s - loss: 0.2241 - categorical_accuracy: 0.9301
39040/60000 [==================>...........] - ETA: 39s - loss: 0.2239 - categorical_accuracy: 0.9302
39072/60000 [==================>...........] - ETA: 39s - loss: 0.2242 - categorical_accuracy: 0.9302
39104/60000 [==================>...........] - ETA: 39s - loss: 0.2240 - categorical_accuracy: 0.9302
39136/60000 [==================>...........] - ETA: 39s - loss: 0.2239 - categorical_accuracy: 0.9303
39168/60000 [==================>...........] - ETA: 39s - loss: 0.2238 - categorical_accuracy: 0.9303
39200/60000 [==================>...........] - ETA: 39s - loss: 0.2236 - categorical_accuracy: 0.9304
39232/60000 [==================>...........] - ETA: 39s - loss: 0.2235 - categorical_accuracy: 0.9304
39264/60000 [==================>...........] - ETA: 39s - loss: 0.2233 - categorical_accuracy: 0.9305
39296/60000 [==================>...........] - ETA: 39s - loss: 0.2232 - categorical_accuracy: 0.9305
39328/60000 [==================>...........] - ETA: 39s - loss: 0.2231 - categorical_accuracy: 0.9305
39360/60000 [==================>...........] - ETA: 39s - loss: 0.2230 - categorical_accuracy: 0.9306
39392/60000 [==================>...........] - ETA: 38s - loss: 0.2228 - categorical_accuracy: 0.9306
39424/60000 [==================>...........] - ETA: 38s - loss: 0.2227 - categorical_accuracy: 0.9307
39456/60000 [==================>...........] - ETA: 38s - loss: 0.2227 - categorical_accuracy: 0.9307
39488/60000 [==================>...........] - ETA: 38s - loss: 0.2226 - categorical_accuracy: 0.9307
39520/60000 [==================>...........] - ETA: 38s - loss: 0.2224 - categorical_accuracy: 0.9308
39552/60000 [==================>...........] - ETA: 38s - loss: 0.2223 - categorical_accuracy: 0.9308
39584/60000 [==================>...........] - ETA: 38s - loss: 0.2221 - categorical_accuracy: 0.9309
39616/60000 [==================>...........] - ETA: 38s - loss: 0.2220 - categorical_accuracy: 0.9309
39648/60000 [==================>...........] - ETA: 38s - loss: 0.2221 - categorical_accuracy: 0.9309
39680/60000 [==================>...........] - ETA: 38s - loss: 0.2219 - categorical_accuracy: 0.9309
39712/60000 [==================>...........] - ETA: 38s - loss: 0.2219 - categorical_accuracy: 0.9309
39744/60000 [==================>...........] - ETA: 38s - loss: 0.2218 - categorical_accuracy: 0.9309
39776/60000 [==================>...........] - ETA: 38s - loss: 0.2216 - categorical_accuracy: 0.9310
39808/60000 [==================>...........] - ETA: 38s - loss: 0.2216 - categorical_accuracy: 0.9310
39840/60000 [==================>...........] - ETA: 38s - loss: 0.2214 - categorical_accuracy: 0.9310
39872/60000 [==================>...........] - ETA: 38s - loss: 0.2213 - categorical_accuracy: 0.9311
39904/60000 [==================>...........] - ETA: 37s - loss: 0.2212 - categorical_accuracy: 0.9311
39936/60000 [==================>...........] - ETA: 37s - loss: 0.2210 - categorical_accuracy: 0.9311
39968/60000 [==================>...........] - ETA: 37s - loss: 0.2210 - categorical_accuracy: 0.9311
40000/60000 [===================>..........] - ETA: 37s - loss: 0.2208 - categorical_accuracy: 0.9312
40032/60000 [===================>..........] - ETA: 37s - loss: 0.2207 - categorical_accuracy: 0.9312
40064/60000 [===================>..........] - ETA: 37s - loss: 0.2206 - categorical_accuracy: 0.9312
40096/60000 [===================>..........] - ETA: 37s - loss: 0.2206 - categorical_accuracy: 0.9312
40128/60000 [===================>..........] - ETA: 37s - loss: 0.2205 - categorical_accuracy: 0.9312
40160/60000 [===================>..........] - ETA: 37s - loss: 0.2204 - categorical_accuracy: 0.9313
40192/60000 [===================>..........] - ETA: 37s - loss: 0.2202 - categorical_accuracy: 0.9314
40224/60000 [===================>..........] - ETA: 37s - loss: 0.2201 - categorical_accuracy: 0.9314
40256/60000 [===================>..........] - ETA: 37s - loss: 0.2199 - categorical_accuracy: 0.9314
40288/60000 [===================>..........] - ETA: 37s - loss: 0.2198 - categorical_accuracy: 0.9315
40320/60000 [===================>..........] - ETA: 37s - loss: 0.2197 - categorical_accuracy: 0.9315
40352/60000 [===================>..........] - ETA: 37s - loss: 0.2196 - categorical_accuracy: 0.9315
40384/60000 [===================>..........] - ETA: 37s - loss: 0.2195 - categorical_accuracy: 0.9316
40416/60000 [===================>..........] - ETA: 37s - loss: 0.2195 - categorical_accuracy: 0.9316
40448/60000 [===================>..........] - ETA: 36s - loss: 0.2194 - categorical_accuracy: 0.9316
40480/60000 [===================>..........] - ETA: 36s - loss: 0.2193 - categorical_accuracy: 0.9316
40512/60000 [===================>..........] - ETA: 36s - loss: 0.2191 - categorical_accuracy: 0.9317
40544/60000 [===================>..........] - ETA: 36s - loss: 0.2190 - categorical_accuracy: 0.9317
40576/60000 [===================>..........] - ETA: 36s - loss: 0.2189 - categorical_accuracy: 0.9317
40608/60000 [===================>..........] - ETA: 36s - loss: 0.2188 - categorical_accuracy: 0.9318
40640/60000 [===================>..........] - ETA: 36s - loss: 0.2188 - categorical_accuracy: 0.9318
40672/60000 [===================>..........] - ETA: 36s - loss: 0.2186 - categorical_accuracy: 0.9318
40704/60000 [===================>..........] - ETA: 36s - loss: 0.2185 - categorical_accuracy: 0.9319
40736/60000 [===================>..........] - ETA: 36s - loss: 0.2184 - categorical_accuracy: 0.9319
40768/60000 [===================>..........] - ETA: 36s - loss: 0.2182 - categorical_accuracy: 0.9320
40800/60000 [===================>..........] - ETA: 36s - loss: 0.2182 - categorical_accuracy: 0.9320
40832/60000 [===================>..........] - ETA: 36s - loss: 0.2181 - categorical_accuracy: 0.9320
40864/60000 [===================>..........] - ETA: 36s - loss: 0.2180 - categorical_accuracy: 0.9321
40896/60000 [===================>..........] - ETA: 36s - loss: 0.2178 - categorical_accuracy: 0.9321
40928/60000 [===================>..........] - ETA: 36s - loss: 0.2177 - categorical_accuracy: 0.9321
40960/60000 [===================>..........] - ETA: 35s - loss: 0.2177 - categorical_accuracy: 0.9321
40992/60000 [===================>..........] - ETA: 35s - loss: 0.2175 - categorical_accuracy: 0.9322
41024/60000 [===================>..........] - ETA: 35s - loss: 0.2174 - categorical_accuracy: 0.9322
41056/60000 [===================>..........] - ETA: 35s - loss: 0.2173 - categorical_accuracy: 0.9322
41088/60000 [===================>..........] - ETA: 35s - loss: 0.2171 - categorical_accuracy: 0.9323
41120/60000 [===================>..........] - ETA: 35s - loss: 0.2170 - categorical_accuracy: 0.9323
41152/60000 [===================>..........] - ETA: 35s - loss: 0.2169 - categorical_accuracy: 0.9324
41184/60000 [===================>..........] - ETA: 35s - loss: 0.2168 - categorical_accuracy: 0.9324
41216/60000 [===================>..........] - ETA: 35s - loss: 0.2169 - categorical_accuracy: 0.9324
41248/60000 [===================>..........] - ETA: 35s - loss: 0.2168 - categorical_accuracy: 0.9324
41280/60000 [===================>..........] - ETA: 35s - loss: 0.2168 - categorical_accuracy: 0.9324
41312/60000 [===================>..........] - ETA: 35s - loss: 0.2168 - categorical_accuracy: 0.9323
41344/60000 [===================>..........] - ETA: 35s - loss: 0.2166 - categorical_accuracy: 0.9324
41376/60000 [===================>..........] - ETA: 35s - loss: 0.2165 - categorical_accuracy: 0.9324
41408/60000 [===================>..........] - ETA: 35s - loss: 0.2164 - categorical_accuracy: 0.9325
41440/60000 [===================>..........] - ETA: 35s - loss: 0.2162 - categorical_accuracy: 0.9325
41472/60000 [===================>..........] - ETA: 35s - loss: 0.2161 - categorical_accuracy: 0.9326
41504/60000 [===================>..........] - ETA: 34s - loss: 0.2160 - categorical_accuracy: 0.9326
41536/60000 [===================>..........] - ETA: 34s - loss: 0.2159 - categorical_accuracy: 0.9326
41568/60000 [===================>..........] - ETA: 34s - loss: 0.2158 - categorical_accuracy: 0.9327
41600/60000 [===================>..........] - ETA: 34s - loss: 0.2157 - categorical_accuracy: 0.9327
41632/60000 [===================>..........] - ETA: 34s - loss: 0.2156 - categorical_accuracy: 0.9327
41664/60000 [===================>..........] - ETA: 34s - loss: 0.2156 - categorical_accuracy: 0.9327
41696/60000 [===================>..........] - ETA: 34s - loss: 0.2154 - categorical_accuracy: 0.9328
41728/60000 [===================>..........] - ETA: 34s - loss: 0.2154 - categorical_accuracy: 0.9328
41760/60000 [===================>..........] - ETA: 34s - loss: 0.2152 - categorical_accuracy: 0.9329
41792/60000 [===================>..........] - ETA: 34s - loss: 0.2151 - categorical_accuracy: 0.9329
41824/60000 [===================>..........] - ETA: 34s - loss: 0.2150 - categorical_accuracy: 0.9329
41856/60000 [===================>..........] - ETA: 34s - loss: 0.2151 - categorical_accuracy: 0.9329
41888/60000 [===================>..........] - ETA: 34s - loss: 0.2150 - categorical_accuracy: 0.9329
41920/60000 [===================>..........] - ETA: 34s - loss: 0.2148 - categorical_accuracy: 0.9329
41952/60000 [===================>..........] - ETA: 34s - loss: 0.2148 - categorical_accuracy: 0.9329
41984/60000 [===================>..........] - ETA: 34s - loss: 0.2148 - categorical_accuracy: 0.9330
42016/60000 [====================>.........] - ETA: 33s - loss: 0.2146 - categorical_accuracy: 0.9330
42048/60000 [====================>.........] - ETA: 33s - loss: 0.2145 - categorical_accuracy: 0.9331
42080/60000 [====================>.........] - ETA: 33s - loss: 0.2143 - categorical_accuracy: 0.9331
42112/60000 [====================>.........] - ETA: 33s - loss: 0.2142 - categorical_accuracy: 0.9332
42144/60000 [====================>.........] - ETA: 33s - loss: 0.2141 - categorical_accuracy: 0.9332
42176/60000 [====================>.........] - ETA: 33s - loss: 0.2140 - categorical_accuracy: 0.9332
42208/60000 [====================>.........] - ETA: 33s - loss: 0.2139 - categorical_accuracy: 0.9332
42240/60000 [====================>.........] - ETA: 33s - loss: 0.2141 - categorical_accuracy: 0.9332
42272/60000 [====================>.........] - ETA: 33s - loss: 0.2140 - categorical_accuracy: 0.9333
42304/60000 [====================>.........] - ETA: 33s - loss: 0.2138 - categorical_accuracy: 0.9333
42336/60000 [====================>.........] - ETA: 33s - loss: 0.2138 - categorical_accuracy: 0.9333
42368/60000 [====================>.........] - ETA: 33s - loss: 0.2138 - categorical_accuracy: 0.9333
42400/60000 [====================>.........] - ETA: 33s - loss: 0.2136 - categorical_accuracy: 0.9334
42432/60000 [====================>.........] - ETA: 33s - loss: 0.2135 - categorical_accuracy: 0.9334
42464/60000 [====================>.........] - ETA: 33s - loss: 0.2134 - categorical_accuracy: 0.9334
42496/60000 [====================>.........] - ETA: 33s - loss: 0.2133 - categorical_accuracy: 0.9335
42528/60000 [====================>.........] - ETA: 33s - loss: 0.2132 - categorical_accuracy: 0.9335
42560/60000 [====================>.........] - ETA: 32s - loss: 0.2133 - categorical_accuracy: 0.9335
42592/60000 [====================>.........] - ETA: 32s - loss: 0.2133 - categorical_accuracy: 0.9335
42624/60000 [====================>.........] - ETA: 32s - loss: 0.2132 - categorical_accuracy: 0.9335
42656/60000 [====================>.........] - ETA: 32s - loss: 0.2130 - categorical_accuracy: 0.9335
42688/60000 [====================>.........] - ETA: 32s - loss: 0.2130 - categorical_accuracy: 0.9335
42720/60000 [====================>.........] - ETA: 32s - loss: 0.2129 - categorical_accuracy: 0.9335
42752/60000 [====================>.........] - ETA: 32s - loss: 0.2128 - categorical_accuracy: 0.9335
42784/60000 [====================>.........] - ETA: 32s - loss: 0.2127 - categorical_accuracy: 0.9336
42816/60000 [====================>.........] - ETA: 32s - loss: 0.2126 - categorical_accuracy: 0.9336
42848/60000 [====================>.........] - ETA: 32s - loss: 0.2125 - categorical_accuracy: 0.9336
42880/60000 [====================>.........] - ETA: 32s - loss: 0.2123 - categorical_accuracy: 0.9337
42912/60000 [====================>.........] - ETA: 32s - loss: 0.2122 - categorical_accuracy: 0.9337
42944/60000 [====================>.........] - ETA: 32s - loss: 0.2120 - categorical_accuracy: 0.9338
42976/60000 [====================>.........] - ETA: 32s - loss: 0.2119 - categorical_accuracy: 0.9338
43008/60000 [====================>.........] - ETA: 32s - loss: 0.2119 - categorical_accuracy: 0.9338
43040/60000 [====================>.........] - ETA: 32s - loss: 0.2118 - categorical_accuracy: 0.9339
43072/60000 [====================>.........] - ETA: 32s - loss: 0.2119 - categorical_accuracy: 0.9339
43104/60000 [====================>.........] - ETA: 31s - loss: 0.2119 - categorical_accuracy: 0.9339
43136/60000 [====================>.........] - ETA: 31s - loss: 0.2119 - categorical_accuracy: 0.9338
43168/60000 [====================>.........] - ETA: 31s - loss: 0.2120 - categorical_accuracy: 0.9338
43200/60000 [====================>.........] - ETA: 31s - loss: 0.2119 - categorical_accuracy: 0.9338
43232/60000 [====================>.........] - ETA: 31s - loss: 0.2118 - categorical_accuracy: 0.9338
43264/60000 [====================>.........] - ETA: 31s - loss: 0.2116 - categorical_accuracy: 0.9339
43296/60000 [====================>.........] - ETA: 31s - loss: 0.2115 - categorical_accuracy: 0.9339
43328/60000 [====================>.........] - ETA: 31s - loss: 0.2114 - categorical_accuracy: 0.9340
43360/60000 [====================>.........] - ETA: 31s - loss: 0.2112 - categorical_accuracy: 0.9340
43392/60000 [====================>.........] - ETA: 31s - loss: 0.2111 - categorical_accuracy: 0.9341
43424/60000 [====================>.........] - ETA: 31s - loss: 0.2111 - categorical_accuracy: 0.9341
43456/60000 [====================>.........] - ETA: 31s - loss: 0.2111 - categorical_accuracy: 0.9341
43488/60000 [====================>.........] - ETA: 31s - loss: 0.2110 - categorical_accuracy: 0.9341
43520/60000 [====================>.........] - ETA: 31s - loss: 0.2109 - categorical_accuracy: 0.9341
43552/60000 [====================>.........] - ETA: 31s - loss: 0.2108 - categorical_accuracy: 0.9342
43584/60000 [====================>.........] - ETA: 31s - loss: 0.2107 - categorical_accuracy: 0.9342
43616/60000 [====================>.........] - ETA: 30s - loss: 0.2107 - categorical_accuracy: 0.9341
43648/60000 [====================>.........] - ETA: 30s - loss: 0.2106 - categorical_accuracy: 0.9342
43680/60000 [====================>.........] - ETA: 30s - loss: 0.2105 - categorical_accuracy: 0.9342
43712/60000 [====================>.........] - ETA: 30s - loss: 0.2105 - categorical_accuracy: 0.9342
43744/60000 [====================>.........] - ETA: 30s - loss: 0.2104 - categorical_accuracy: 0.9342
43776/60000 [====================>.........] - ETA: 30s - loss: 0.2103 - categorical_accuracy: 0.9342
43808/60000 [====================>.........] - ETA: 30s - loss: 0.2103 - categorical_accuracy: 0.9342
43840/60000 [====================>.........] - ETA: 30s - loss: 0.2102 - categorical_accuracy: 0.9342
43872/60000 [====================>.........] - ETA: 30s - loss: 0.2101 - categorical_accuracy: 0.9343
43904/60000 [====================>.........] - ETA: 30s - loss: 0.2099 - categorical_accuracy: 0.9343
43936/60000 [====================>.........] - ETA: 30s - loss: 0.2098 - categorical_accuracy: 0.9344
43968/60000 [====================>.........] - ETA: 30s - loss: 0.2098 - categorical_accuracy: 0.9344
44000/60000 [=====================>........] - ETA: 30s - loss: 0.2097 - categorical_accuracy: 0.9344
44032/60000 [=====================>........] - ETA: 30s - loss: 0.2096 - categorical_accuracy: 0.9344
44064/60000 [=====================>........] - ETA: 30s - loss: 0.2095 - categorical_accuracy: 0.9344
44096/60000 [=====================>........] - ETA: 30s - loss: 0.2094 - categorical_accuracy: 0.9345
44128/60000 [=====================>........] - ETA: 29s - loss: 0.2093 - categorical_accuracy: 0.9345
44160/60000 [=====================>........] - ETA: 29s - loss: 0.2092 - categorical_accuracy: 0.9346
44192/60000 [=====================>........] - ETA: 29s - loss: 0.2091 - categorical_accuracy: 0.9346
44224/60000 [=====================>........] - ETA: 29s - loss: 0.2090 - categorical_accuracy: 0.9346
44256/60000 [=====================>........] - ETA: 29s - loss: 0.2089 - categorical_accuracy: 0.9347
44288/60000 [=====================>........] - ETA: 29s - loss: 0.2088 - categorical_accuracy: 0.9347
44320/60000 [=====================>........] - ETA: 29s - loss: 0.2087 - categorical_accuracy: 0.9347
44352/60000 [=====================>........] - ETA: 29s - loss: 0.2086 - categorical_accuracy: 0.9347
44384/60000 [=====================>........] - ETA: 29s - loss: 0.2086 - categorical_accuracy: 0.9347
44416/60000 [=====================>........] - ETA: 29s - loss: 0.2085 - categorical_accuracy: 0.9347
44448/60000 [=====================>........] - ETA: 29s - loss: 0.2084 - categorical_accuracy: 0.9348
44480/60000 [=====================>........] - ETA: 29s - loss: 0.2083 - categorical_accuracy: 0.9348
44512/60000 [=====================>........] - ETA: 29s - loss: 0.2084 - categorical_accuracy: 0.9348
44544/60000 [=====================>........] - ETA: 29s - loss: 0.2084 - categorical_accuracy: 0.9348
44576/60000 [=====================>........] - ETA: 29s - loss: 0.2083 - categorical_accuracy: 0.9348
44608/60000 [=====================>........] - ETA: 29s - loss: 0.2081 - categorical_accuracy: 0.9349
44640/60000 [=====================>........] - ETA: 29s - loss: 0.2080 - categorical_accuracy: 0.9349
44672/60000 [=====================>........] - ETA: 28s - loss: 0.2079 - categorical_accuracy: 0.9349
44704/60000 [=====================>........] - ETA: 28s - loss: 0.2079 - categorical_accuracy: 0.9349
44736/60000 [=====================>........] - ETA: 28s - loss: 0.2077 - categorical_accuracy: 0.9350
44768/60000 [=====================>........] - ETA: 28s - loss: 0.2078 - categorical_accuracy: 0.9350
44800/60000 [=====================>........] - ETA: 28s - loss: 0.2077 - categorical_accuracy: 0.9350
44832/60000 [=====================>........] - ETA: 28s - loss: 0.2076 - categorical_accuracy: 0.9350
44864/60000 [=====================>........] - ETA: 28s - loss: 0.2075 - categorical_accuracy: 0.9351
44896/60000 [=====================>........] - ETA: 28s - loss: 0.2074 - categorical_accuracy: 0.9351
44928/60000 [=====================>........] - ETA: 28s - loss: 0.2073 - categorical_accuracy: 0.9351
44960/60000 [=====================>........] - ETA: 28s - loss: 0.2072 - categorical_accuracy: 0.9352
44992/60000 [=====================>........] - ETA: 28s - loss: 0.2071 - categorical_accuracy: 0.9352
45024/60000 [=====================>........] - ETA: 28s - loss: 0.2072 - categorical_accuracy: 0.9352
45056/60000 [=====================>........] - ETA: 28s - loss: 0.2071 - categorical_accuracy: 0.9352
45088/60000 [=====================>........] - ETA: 28s - loss: 0.2070 - categorical_accuracy: 0.9352
45120/60000 [=====================>........] - ETA: 28s - loss: 0.2069 - categorical_accuracy: 0.9353
45152/60000 [=====================>........] - ETA: 28s - loss: 0.2067 - categorical_accuracy: 0.9353
45184/60000 [=====================>........] - ETA: 28s - loss: 0.2067 - categorical_accuracy: 0.9353
45216/60000 [=====================>........] - ETA: 27s - loss: 0.2067 - categorical_accuracy: 0.9353
45248/60000 [=====================>........] - ETA: 27s - loss: 0.2065 - categorical_accuracy: 0.9354
45280/60000 [=====================>........] - ETA: 27s - loss: 0.2064 - categorical_accuracy: 0.9354
45312/60000 [=====================>........] - ETA: 27s - loss: 0.2063 - categorical_accuracy: 0.9354
45344/60000 [=====================>........] - ETA: 27s - loss: 0.2063 - categorical_accuracy: 0.9355
45376/60000 [=====================>........] - ETA: 27s - loss: 0.2061 - categorical_accuracy: 0.9355
45408/60000 [=====================>........] - ETA: 27s - loss: 0.2061 - categorical_accuracy: 0.9355
45440/60000 [=====================>........] - ETA: 27s - loss: 0.2060 - categorical_accuracy: 0.9355
45472/60000 [=====================>........] - ETA: 27s - loss: 0.2060 - categorical_accuracy: 0.9356
45504/60000 [=====================>........] - ETA: 27s - loss: 0.2058 - categorical_accuracy: 0.9356
45536/60000 [=====================>........] - ETA: 27s - loss: 0.2057 - categorical_accuracy: 0.9356
45568/60000 [=====================>........] - ETA: 27s - loss: 0.2056 - categorical_accuracy: 0.9357
45600/60000 [=====================>........] - ETA: 27s - loss: 0.2056 - categorical_accuracy: 0.9357
45632/60000 [=====================>........] - ETA: 27s - loss: 0.2055 - categorical_accuracy: 0.9357
45664/60000 [=====================>........] - ETA: 27s - loss: 0.2055 - categorical_accuracy: 0.9357
45696/60000 [=====================>........] - ETA: 27s - loss: 0.2055 - categorical_accuracy: 0.9357
45728/60000 [=====================>........] - ETA: 26s - loss: 0.2054 - categorical_accuracy: 0.9358
45760/60000 [=====================>........] - ETA: 26s - loss: 0.2053 - categorical_accuracy: 0.9358
45792/60000 [=====================>........] - ETA: 26s - loss: 0.2052 - categorical_accuracy: 0.9358
45824/60000 [=====================>........] - ETA: 26s - loss: 0.2051 - categorical_accuracy: 0.9358
45856/60000 [=====================>........] - ETA: 26s - loss: 0.2049 - categorical_accuracy: 0.9359
45888/60000 [=====================>........] - ETA: 26s - loss: 0.2048 - categorical_accuracy: 0.9359
45920/60000 [=====================>........] - ETA: 26s - loss: 0.2047 - categorical_accuracy: 0.9360
45952/60000 [=====================>........] - ETA: 26s - loss: 0.2046 - categorical_accuracy: 0.9360
45984/60000 [=====================>........] - ETA: 26s - loss: 0.2045 - categorical_accuracy: 0.9360
46016/60000 [======================>.......] - ETA: 26s - loss: 0.2043 - categorical_accuracy: 0.9361
46048/60000 [======================>.......] - ETA: 26s - loss: 0.2043 - categorical_accuracy: 0.9361
46080/60000 [======================>.......] - ETA: 26s - loss: 0.2042 - categorical_accuracy: 0.9361
46112/60000 [======================>.......] - ETA: 26s - loss: 0.2041 - categorical_accuracy: 0.9361
46144/60000 [======================>.......] - ETA: 26s - loss: 0.2040 - categorical_accuracy: 0.9361
46176/60000 [======================>.......] - ETA: 26s - loss: 0.2039 - categorical_accuracy: 0.9362
46208/60000 [======================>.......] - ETA: 26s - loss: 0.2039 - categorical_accuracy: 0.9362
46240/60000 [======================>.......] - ETA: 25s - loss: 0.2038 - categorical_accuracy: 0.9362
46272/60000 [======================>.......] - ETA: 25s - loss: 0.2037 - categorical_accuracy: 0.9362
46304/60000 [======================>.......] - ETA: 25s - loss: 0.2036 - categorical_accuracy: 0.9363
46336/60000 [======================>.......] - ETA: 25s - loss: 0.2036 - categorical_accuracy: 0.9363
46368/60000 [======================>.......] - ETA: 25s - loss: 0.2035 - categorical_accuracy: 0.9363
46400/60000 [======================>.......] - ETA: 25s - loss: 0.2033 - categorical_accuracy: 0.9364
46432/60000 [======================>.......] - ETA: 25s - loss: 0.2032 - categorical_accuracy: 0.9364
46464/60000 [======================>.......] - ETA: 25s - loss: 0.2031 - categorical_accuracy: 0.9364
46496/60000 [======================>.......] - ETA: 25s - loss: 0.2031 - categorical_accuracy: 0.9364
46528/60000 [======================>.......] - ETA: 25s - loss: 0.2030 - categorical_accuracy: 0.9365
46560/60000 [======================>.......] - ETA: 25s - loss: 0.2029 - categorical_accuracy: 0.9365
46592/60000 [======================>.......] - ETA: 25s - loss: 0.2028 - categorical_accuracy: 0.9365
46624/60000 [======================>.......] - ETA: 25s - loss: 0.2027 - categorical_accuracy: 0.9366
46656/60000 [======================>.......] - ETA: 25s - loss: 0.2026 - categorical_accuracy: 0.9366
46688/60000 [======================>.......] - ETA: 25s - loss: 0.2025 - categorical_accuracy: 0.9366
46720/60000 [======================>.......] - ETA: 25s - loss: 0.2024 - categorical_accuracy: 0.9366
46752/60000 [======================>.......] - ETA: 25s - loss: 0.2023 - categorical_accuracy: 0.9367
46784/60000 [======================>.......] - ETA: 24s - loss: 0.2022 - categorical_accuracy: 0.9367
46816/60000 [======================>.......] - ETA: 24s - loss: 0.2024 - categorical_accuracy: 0.9367
46848/60000 [======================>.......] - ETA: 24s - loss: 0.2023 - categorical_accuracy: 0.9367
46880/60000 [======================>.......] - ETA: 24s - loss: 0.2022 - categorical_accuracy: 0.9367
46912/60000 [======================>.......] - ETA: 24s - loss: 0.2023 - categorical_accuracy: 0.9367
46944/60000 [======================>.......] - ETA: 24s - loss: 0.2022 - categorical_accuracy: 0.9367
46976/60000 [======================>.......] - ETA: 24s - loss: 0.2021 - categorical_accuracy: 0.9368
47008/60000 [======================>.......] - ETA: 24s - loss: 0.2021 - categorical_accuracy: 0.9368
47040/60000 [======================>.......] - ETA: 24s - loss: 0.2020 - categorical_accuracy: 0.9368
47072/60000 [======================>.......] - ETA: 24s - loss: 0.2019 - categorical_accuracy: 0.9368
47104/60000 [======================>.......] - ETA: 24s - loss: 0.2019 - categorical_accuracy: 0.9368
47136/60000 [======================>.......] - ETA: 24s - loss: 0.2018 - categorical_accuracy: 0.9369
47168/60000 [======================>.......] - ETA: 24s - loss: 0.2017 - categorical_accuracy: 0.9369
47200/60000 [======================>.......] - ETA: 24s - loss: 0.2015 - categorical_accuracy: 0.9370
47232/60000 [======================>.......] - ETA: 24s - loss: 0.2015 - categorical_accuracy: 0.9370
47264/60000 [======================>.......] - ETA: 24s - loss: 0.2014 - categorical_accuracy: 0.9370
47296/60000 [======================>.......] - ETA: 24s - loss: 0.2014 - categorical_accuracy: 0.9370
47328/60000 [======================>.......] - ETA: 23s - loss: 0.2014 - categorical_accuracy: 0.9370
47360/60000 [======================>.......] - ETA: 23s - loss: 0.2013 - categorical_accuracy: 0.9371
47392/60000 [======================>.......] - ETA: 23s - loss: 0.2014 - categorical_accuracy: 0.9371
47424/60000 [======================>.......] - ETA: 23s - loss: 0.2013 - categorical_accuracy: 0.9371
47456/60000 [======================>.......] - ETA: 23s - loss: 0.2013 - categorical_accuracy: 0.9371
47488/60000 [======================>.......] - ETA: 23s - loss: 0.2012 - categorical_accuracy: 0.9371
47520/60000 [======================>.......] - ETA: 23s - loss: 0.2011 - categorical_accuracy: 0.9372
47552/60000 [======================>.......] - ETA: 23s - loss: 0.2010 - categorical_accuracy: 0.9372
47584/60000 [======================>.......] - ETA: 23s - loss: 0.2009 - categorical_accuracy: 0.9372
47616/60000 [======================>.......] - ETA: 23s - loss: 0.2008 - categorical_accuracy: 0.9372
47648/60000 [======================>.......] - ETA: 23s - loss: 0.2008 - categorical_accuracy: 0.9372
47680/60000 [======================>.......] - ETA: 23s - loss: 0.2007 - categorical_accuracy: 0.9372
47712/60000 [======================>.......] - ETA: 23s - loss: 0.2007 - categorical_accuracy: 0.9373
47744/60000 [======================>.......] - ETA: 23s - loss: 0.2006 - categorical_accuracy: 0.9373
47776/60000 [======================>.......] - ETA: 23s - loss: 0.2006 - categorical_accuracy: 0.9373
47808/60000 [======================>.......] - ETA: 23s - loss: 0.2007 - categorical_accuracy: 0.9373
47840/60000 [======================>.......] - ETA: 22s - loss: 0.2008 - categorical_accuracy: 0.9373
47872/60000 [======================>.......] - ETA: 22s - loss: 0.2009 - categorical_accuracy: 0.9373
47904/60000 [======================>.......] - ETA: 22s - loss: 0.2010 - categorical_accuracy: 0.9373
47936/60000 [======================>.......] - ETA: 22s - loss: 0.2009 - categorical_accuracy: 0.9373
47968/60000 [======================>.......] - ETA: 22s - loss: 0.2008 - categorical_accuracy: 0.9374
48000/60000 [=======================>......] - ETA: 22s - loss: 0.2007 - categorical_accuracy: 0.9374
48032/60000 [=======================>......] - ETA: 22s - loss: 0.2006 - categorical_accuracy: 0.9374
48064/60000 [=======================>......] - ETA: 22s - loss: 0.2005 - categorical_accuracy: 0.9375
48096/60000 [=======================>......] - ETA: 22s - loss: 0.2004 - categorical_accuracy: 0.9375
48128/60000 [=======================>......] - ETA: 22s - loss: 0.2003 - categorical_accuracy: 0.9375
48160/60000 [=======================>......] - ETA: 22s - loss: 0.2002 - categorical_accuracy: 0.9376
48192/60000 [=======================>......] - ETA: 22s - loss: 0.2001 - categorical_accuracy: 0.9376
48224/60000 [=======================>......] - ETA: 22s - loss: 0.2000 - categorical_accuracy: 0.9376
48256/60000 [=======================>......] - ETA: 22s - loss: 0.1998 - categorical_accuracy: 0.9377
48288/60000 [=======================>......] - ETA: 22s - loss: 0.1998 - categorical_accuracy: 0.9377
48320/60000 [=======================>......] - ETA: 22s - loss: 0.1997 - categorical_accuracy: 0.9377
48352/60000 [=======================>......] - ETA: 21s - loss: 0.1997 - categorical_accuracy: 0.9377
48384/60000 [=======================>......] - ETA: 21s - loss: 0.1996 - categorical_accuracy: 0.9377
48416/60000 [=======================>......] - ETA: 21s - loss: 0.1995 - categorical_accuracy: 0.9377
48448/60000 [=======================>......] - ETA: 21s - loss: 0.1994 - categorical_accuracy: 0.9377
48480/60000 [=======================>......] - ETA: 21s - loss: 0.1993 - categorical_accuracy: 0.9378
48512/60000 [=======================>......] - ETA: 21s - loss: 0.1993 - categorical_accuracy: 0.9378
48544/60000 [=======================>......] - ETA: 21s - loss: 0.1995 - categorical_accuracy: 0.9378
48576/60000 [=======================>......] - ETA: 21s - loss: 0.1993 - categorical_accuracy: 0.9378
48608/60000 [=======================>......] - ETA: 21s - loss: 0.1993 - categorical_accuracy: 0.9378
48640/60000 [=======================>......] - ETA: 21s - loss: 0.1992 - categorical_accuracy: 0.9379
48672/60000 [=======================>......] - ETA: 21s - loss: 0.1991 - categorical_accuracy: 0.9379
48704/60000 [=======================>......] - ETA: 21s - loss: 0.1992 - categorical_accuracy: 0.9379
48736/60000 [=======================>......] - ETA: 21s - loss: 0.1991 - categorical_accuracy: 0.9379
48768/60000 [=======================>......] - ETA: 21s - loss: 0.1990 - categorical_accuracy: 0.9379
48800/60000 [=======================>......] - ETA: 21s - loss: 0.1989 - categorical_accuracy: 0.9380
48832/60000 [=======================>......] - ETA: 21s - loss: 0.1989 - categorical_accuracy: 0.9380
48864/60000 [=======================>......] - ETA: 21s - loss: 0.1988 - categorical_accuracy: 0.9380
48896/60000 [=======================>......] - ETA: 20s - loss: 0.1987 - categorical_accuracy: 0.9381
48928/60000 [=======================>......] - ETA: 20s - loss: 0.1985 - categorical_accuracy: 0.9381
48960/60000 [=======================>......] - ETA: 20s - loss: 0.1985 - categorical_accuracy: 0.9381
48992/60000 [=======================>......] - ETA: 20s - loss: 0.1984 - categorical_accuracy: 0.9381
49024/60000 [=======================>......] - ETA: 20s - loss: 0.1984 - categorical_accuracy: 0.9382
49056/60000 [=======================>......] - ETA: 20s - loss: 0.1983 - categorical_accuracy: 0.9382
49088/60000 [=======================>......] - ETA: 20s - loss: 0.1981 - categorical_accuracy: 0.9382
49120/60000 [=======================>......] - ETA: 20s - loss: 0.1980 - categorical_accuracy: 0.9383
49152/60000 [=======================>......] - ETA: 20s - loss: 0.1979 - categorical_accuracy: 0.9383
49184/60000 [=======================>......] - ETA: 20s - loss: 0.1978 - categorical_accuracy: 0.9383
49216/60000 [=======================>......] - ETA: 20s - loss: 0.1977 - categorical_accuracy: 0.9384
49248/60000 [=======================>......] - ETA: 20s - loss: 0.1976 - categorical_accuracy: 0.9384
49280/60000 [=======================>......] - ETA: 20s - loss: 0.1976 - categorical_accuracy: 0.9384
49312/60000 [=======================>......] - ETA: 20s - loss: 0.1975 - categorical_accuracy: 0.9385
49344/60000 [=======================>......] - ETA: 20s - loss: 0.1974 - categorical_accuracy: 0.9385
49376/60000 [=======================>......] - ETA: 20s - loss: 0.1973 - categorical_accuracy: 0.9385
49408/60000 [=======================>......] - ETA: 19s - loss: 0.1972 - categorical_accuracy: 0.9385
49440/60000 [=======================>......] - ETA: 19s - loss: 0.1974 - categorical_accuracy: 0.9385
49472/60000 [=======================>......] - ETA: 19s - loss: 0.1974 - categorical_accuracy: 0.9385
49504/60000 [=======================>......] - ETA: 19s - loss: 0.1972 - categorical_accuracy: 0.9386
49536/60000 [=======================>......] - ETA: 19s - loss: 0.1971 - categorical_accuracy: 0.9386
49568/60000 [=======================>......] - ETA: 19s - loss: 0.1972 - categorical_accuracy: 0.9386
49600/60000 [=======================>......] - ETA: 19s - loss: 0.1970 - categorical_accuracy: 0.9386
49632/60000 [=======================>......] - ETA: 19s - loss: 0.1970 - categorical_accuracy: 0.9386
49664/60000 [=======================>......] - ETA: 19s - loss: 0.1969 - categorical_accuracy: 0.9387
49696/60000 [=======================>......] - ETA: 19s - loss: 0.1968 - categorical_accuracy: 0.9387
49728/60000 [=======================>......] - ETA: 19s - loss: 0.1967 - categorical_accuracy: 0.9387
49760/60000 [=======================>......] - ETA: 19s - loss: 0.1966 - categorical_accuracy: 0.9388
49792/60000 [=======================>......] - ETA: 19s - loss: 0.1967 - categorical_accuracy: 0.9387
49824/60000 [=======================>......] - ETA: 19s - loss: 0.1966 - categorical_accuracy: 0.9388
49856/60000 [=======================>......] - ETA: 19s - loss: 0.1966 - categorical_accuracy: 0.9388
49888/60000 [=======================>......] - ETA: 19s - loss: 0.1966 - categorical_accuracy: 0.9388
49920/60000 [=======================>......] - ETA: 19s - loss: 0.1965 - categorical_accuracy: 0.9388
49952/60000 [=======================>......] - ETA: 18s - loss: 0.1964 - categorical_accuracy: 0.9388
49984/60000 [=======================>......] - ETA: 18s - loss: 0.1963 - categorical_accuracy: 0.9389
50016/60000 [========================>.....] - ETA: 18s - loss: 0.1962 - categorical_accuracy: 0.9389
50048/60000 [========================>.....] - ETA: 18s - loss: 0.1963 - categorical_accuracy: 0.9389
50080/60000 [========================>.....] - ETA: 18s - loss: 0.1962 - categorical_accuracy: 0.9390
50112/60000 [========================>.....] - ETA: 18s - loss: 0.1961 - categorical_accuracy: 0.9390
50144/60000 [========================>.....] - ETA: 18s - loss: 0.1961 - categorical_accuracy: 0.9390
50176/60000 [========================>.....] - ETA: 18s - loss: 0.1960 - categorical_accuracy: 0.9390
50208/60000 [========================>.....] - ETA: 18s - loss: 0.1959 - categorical_accuracy: 0.9390
50240/60000 [========================>.....] - ETA: 18s - loss: 0.1959 - categorical_accuracy: 0.9390
50272/60000 [========================>.....] - ETA: 18s - loss: 0.1959 - categorical_accuracy: 0.9390
50304/60000 [========================>.....] - ETA: 18s - loss: 0.1959 - categorical_accuracy: 0.9390
50336/60000 [========================>.....] - ETA: 18s - loss: 0.1958 - categorical_accuracy: 0.9391
50368/60000 [========================>.....] - ETA: 18s - loss: 0.1957 - categorical_accuracy: 0.9391
50400/60000 [========================>.....] - ETA: 18s - loss: 0.1957 - categorical_accuracy: 0.9391
50432/60000 [========================>.....] - ETA: 18s - loss: 0.1956 - categorical_accuracy: 0.9391
50464/60000 [========================>.....] - ETA: 17s - loss: 0.1955 - categorical_accuracy: 0.9391
50496/60000 [========================>.....] - ETA: 17s - loss: 0.1956 - categorical_accuracy: 0.9391
50528/60000 [========================>.....] - ETA: 17s - loss: 0.1954 - categorical_accuracy: 0.9392
50560/60000 [========================>.....] - ETA: 17s - loss: 0.1954 - categorical_accuracy: 0.9392
50592/60000 [========================>.....] - ETA: 17s - loss: 0.1953 - categorical_accuracy: 0.9392
50624/60000 [========================>.....] - ETA: 17s - loss: 0.1952 - categorical_accuracy: 0.9392
50656/60000 [========================>.....] - ETA: 17s - loss: 0.1951 - categorical_accuracy: 0.9393
50688/60000 [========================>.....] - ETA: 17s - loss: 0.1950 - categorical_accuracy: 0.9393
50720/60000 [========================>.....] - ETA: 17s - loss: 0.1951 - categorical_accuracy: 0.9393
50752/60000 [========================>.....] - ETA: 17s - loss: 0.1950 - categorical_accuracy: 0.9394
50784/60000 [========================>.....] - ETA: 17s - loss: 0.1949 - categorical_accuracy: 0.9394
50816/60000 [========================>.....] - ETA: 17s - loss: 0.1948 - categorical_accuracy: 0.9394
50848/60000 [========================>.....] - ETA: 17s - loss: 0.1947 - categorical_accuracy: 0.9394
50880/60000 [========================>.....] - ETA: 17s - loss: 0.1946 - categorical_accuracy: 0.9395
50912/60000 [========================>.....] - ETA: 17s - loss: 0.1945 - categorical_accuracy: 0.9395
50944/60000 [========================>.....] - ETA: 17s - loss: 0.1944 - categorical_accuracy: 0.9395
50976/60000 [========================>.....] - ETA: 17s - loss: 0.1943 - categorical_accuracy: 0.9396
51008/60000 [========================>.....] - ETA: 16s - loss: 0.1943 - categorical_accuracy: 0.9396
51040/60000 [========================>.....] - ETA: 16s - loss: 0.1942 - categorical_accuracy: 0.9396
51072/60000 [========================>.....] - ETA: 16s - loss: 0.1945 - categorical_accuracy: 0.9396
51104/60000 [========================>.....] - ETA: 16s - loss: 0.1944 - categorical_accuracy: 0.9396
51136/60000 [========================>.....] - ETA: 16s - loss: 0.1944 - categorical_accuracy: 0.9397
51168/60000 [========================>.....] - ETA: 16s - loss: 0.1943 - categorical_accuracy: 0.9397
51200/60000 [========================>.....] - ETA: 16s - loss: 0.1941 - categorical_accuracy: 0.9397
51232/60000 [========================>.....] - ETA: 16s - loss: 0.1940 - categorical_accuracy: 0.9398
51264/60000 [========================>.....] - ETA: 16s - loss: 0.1940 - categorical_accuracy: 0.9398
51296/60000 [========================>.....] - ETA: 16s - loss: 0.1940 - categorical_accuracy: 0.9398
51328/60000 [========================>.....] - ETA: 16s - loss: 0.1939 - categorical_accuracy: 0.9398
51360/60000 [========================>.....] - ETA: 16s - loss: 0.1939 - categorical_accuracy: 0.9398
51392/60000 [========================>.....] - ETA: 16s - loss: 0.1939 - categorical_accuracy: 0.9398
51424/60000 [========================>.....] - ETA: 16s - loss: 0.1938 - categorical_accuracy: 0.9399
51456/60000 [========================>.....] - ETA: 16s - loss: 0.1937 - categorical_accuracy: 0.9399
51488/60000 [========================>.....] - ETA: 16s - loss: 0.1936 - categorical_accuracy: 0.9399
51520/60000 [========================>.....] - ETA: 15s - loss: 0.1935 - categorical_accuracy: 0.9400
51552/60000 [========================>.....] - ETA: 15s - loss: 0.1934 - categorical_accuracy: 0.9400
51584/60000 [========================>.....] - ETA: 15s - loss: 0.1933 - categorical_accuracy: 0.9400
51616/60000 [========================>.....] - ETA: 15s - loss: 0.1932 - categorical_accuracy: 0.9401
51648/60000 [========================>.....] - ETA: 15s - loss: 0.1934 - categorical_accuracy: 0.9401
51680/60000 [========================>.....] - ETA: 15s - loss: 0.1933 - categorical_accuracy: 0.9401
51712/60000 [========================>.....] - ETA: 15s - loss: 0.1932 - categorical_accuracy: 0.9401
51744/60000 [========================>.....] - ETA: 15s - loss: 0.1931 - categorical_accuracy: 0.9401
51776/60000 [========================>.....] - ETA: 15s - loss: 0.1930 - categorical_accuracy: 0.9402
51808/60000 [========================>.....] - ETA: 15s - loss: 0.1929 - categorical_accuracy: 0.9402
51840/60000 [========================>.....] - ETA: 15s - loss: 0.1928 - categorical_accuracy: 0.9402
51872/60000 [========================>.....] - ETA: 15s - loss: 0.1928 - categorical_accuracy: 0.9402
51904/60000 [========================>.....] - ETA: 15s - loss: 0.1927 - categorical_accuracy: 0.9403
51936/60000 [========================>.....] - ETA: 15s - loss: 0.1926 - categorical_accuracy: 0.9403
51968/60000 [========================>.....] - ETA: 15s - loss: 0.1926 - categorical_accuracy: 0.9403
52000/60000 [=========================>....] - ETA: 15s - loss: 0.1927 - categorical_accuracy: 0.9403
52032/60000 [=========================>....] - ETA: 15s - loss: 0.1927 - categorical_accuracy: 0.9402
52064/60000 [=========================>....] - ETA: 14s - loss: 0.1927 - categorical_accuracy: 0.9403
52096/60000 [=========================>....] - ETA: 14s - loss: 0.1927 - categorical_accuracy: 0.9402
52128/60000 [=========================>....] - ETA: 14s - loss: 0.1926 - categorical_accuracy: 0.9403
52160/60000 [=========================>....] - ETA: 14s - loss: 0.1925 - categorical_accuracy: 0.9403
52192/60000 [=========================>....] - ETA: 14s - loss: 0.1925 - categorical_accuracy: 0.9403
52224/60000 [=========================>....] - ETA: 14s - loss: 0.1924 - categorical_accuracy: 0.9403
52256/60000 [=========================>....] - ETA: 14s - loss: 0.1923 - categorical_accuracy: 0.9404
52288/60000 [=========================>....] - ETA: 14s - loss: 0.1922 - categorical_accuracy: 0.9404
52320/60000 [=========================>....] - ETA: 14s - loss: 0.1921 - categorical_accuracy: 0.9404
52352/60000 [=========================>....] - ETA: 14s - loss: 0.1921 - categorical_accuracy: 0.9404
52384/60000 [=========================>....] - ETA: 14s - loss: 0.1920 - categorical_accuracy: 0.9405
52416/60000 [=========================>....] - ETA: 14s - loss: 0.1919 - categorical_accuracy: 0.9405
52448/60000 [=========================>....] - ETA: 14s - loss: 0.1918 - categorical_accuracy: 0.9405
52480/60000 [=========================>....] - ETA: 14s - loss: 0.1919 - categorical_accuracy: 0.9405
52512/60000 [=========================>....] - ETA: 14s - loss: 0.1920 - categorical_accuracy: 0.9405
52544/60000 [=========================>....] - ETA: 14s - loss: 0.1919 - categorical_accuracy: 0.9405
52576/60000 [=========================>....] - ETA: 13s - loss: 0.1920 - categorical_accuracy: 0.9405
52608/60000 [=========================>....] - ETA: 13s - loss: 0.1920 - categorical_accuracy: 0.9405
52640/60000 [=========================>....] - ETA: 13s - loss: 0.1920 - categorical_accuracy: 0.9405
52672/60000 [=========================>....] - ETA: 13s - loss: 0.1919 - categorical_accuracy: 0.9405
52704/60000 [=========================>....] - ETA: 13s - loss: 0.1919 - categorical_accuracy: 0.9405
52736/60000 [=========================>....] - ETA: 13s - loss: 0.1918 - categorical_accuracy: 0.9406
52768/60000 [=========================>....] - ETA: 13s - loss: 0.1917 - categorical_accuracy: 0.9406
52800/60000 [=========================>....] - ETA: 13s - loss: 0.1917 - categorical_accuracy: 0.9405
52832/60000 [=========================>....] - ETA: 13s - loss: 0.1917 - categorical_accuracy: 0.9405
52864/60000 [=========================>....] - ETA: 13s - loss: 0.1916 - categorical_accuracy: 0.9406
52896/60000 [=========================>....] - ETA: 13s - loss: 0.1915 - categorical_accuracy: 0.9406
52928/60000 [=========================>....] - ETA: 13s - loss: 0.1916 - categorical_accuracy: 0.9406
52960/60000 [=========================>....] - ETA: 13s - loss: 0.1915 - categorical_accuracy: 0.9407
52992/60000 [=========================>....] - ETA: 13s - loss: 0.1914 - categorical_accuracy: 0.9407
53024/60000 [=========================>....] - ETA: 13s - loss: 0.1913 - categorical_accuracy: 0.9407
53056/60000 [=========================>....] - ETA: 13s - loss: 0.1912 - categorical_accuracy: 0.9408
53088/60000 [=========================>....] - ETA: 13s - loss: 0.1911 - categorical_accuracy: 0.9408
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1910 - categorical_accuracy: 0.9408
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1910 - categorical_accuracy: 0.9408
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1909 - categorical_accuracy: 0.9408
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1909 - categorical_accuracy: 0.9408
53248/60000 [=========================>....] - ETA: 12s - loss: 0.1908 - categorical_accuracy: 0.9409
53280/60000 [=========================>....] - ETA: 12s - loss: 0.1907 - categorical_accuracy: 0.9409
53312/60000 [=========================>....] - ETA: 12s - loss: 0.1906 - categorical_accuracy: 0.9409
53344/60000 [=========================>....] - ETA: 12s - loss: 0.1905 - categorical_accuracy: 0.9410
53376/60000 [=========================>....] - ETA: 12s - loss: 0.1905 - categorical_accuracy: 0.9410
53408/60000 [=========================>....] - ETA: 12s - loss: 0.1904 - categorical_accuracy: 0.9410
53440/60000 [=========================>....] - ETA: 12s - loss: 0.1903 - categorical_accuracy: 0.9410
53472/60000 [=========================>....] - ETA: 12s - loss: 0.1902 - categorical_accuracy: 0.9411
53504/60000 [=========================>....] - ETA: 12s - loss: 0.1901 - categorical_accuracy: 0.9411
53536/60000 [=========================>....] - ETA: 12s - loss: 0.1900 - categorical_accuracy: 0.9411
53568/60000 [=========================>....] - ETA: 12s - loss: 0.1900 - categorical_accuracy: 0.9411
53600/60000 [=========================>....] - ETA: 12s - loss: 0.1899 - categorical_accuracy: 0.9412
53632/60000 [=========================>....] - ETA: 12s - loss: 0.1898 - categorical_accuracy: 0.9412
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1898 - categorical_accuracy: 0.9412
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1897 - categorical_accuracy: 0.9412
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1896 - categorical_accuracy: 0.9412
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1895 - categorical_accuracy: 0.9413
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1895 - categorical_accuracy: 0.9413
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1894 - categorical_accuracy: 0.9413
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1894 - categorical_accuracy: 0.9413
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1893 - categorical_accuracy: 0.9413
53920/60000 [=========================>....] - ETA: 11s - loss: 0.1892 - categorical_accuracy: 0.9414
53952/60000 [=========================>....] - ETA: 11s - loss: 0.1891 - categorical_accuracy: 0.9414
53984/60000 [=========================>....] - ETA: 11s - loss: 0.1890 - categorical_accuracy: 0.9414
54016/60000 [==========================>...] - ETA: 11s - loss: 0.1890 - categorical_accuracy: 0.9414
54048/60000 [==========================>...] - ETA: 11s - loss: 0.1888 - categorical_accuracy: 0.9415
54080/60000 [==========================>...] - ETA: 11s - loss: 0.1888 - categorical_accuracy: 0.9415
54112/60000 [==========================>...] - ETA: 11s - loss: 0.1889 - categorical_accuracy: 0.9415
54144/60000 [==========================>...] - ETA: 11s - loss: 0.1889 - categorical_accuracy: 0.9415
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1888 - categorical_accuracy: 0.9415
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1888 - categorical_accuracy: 0.9415
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1887 - categorical_accuracy: 0.9416
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1887 - categorical_accuracy: 0.9416
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1886 - categorical_accuracy: 0.9416
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1886 - categorical_accuracy: 0.9416
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1886 - categorical_accuracy: 0.9416
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1885 - categorical_accuracy: 0.9416
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1885 - categorical_accuracy: 0.9416
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1884 - categorical_accuracy: 0.9416
54496/60000 [==========================>...] - ETA: 10s - loss: 0.1884 - categorical_accuracy: 0.9416
54528/60000 [==========================>...] - ETA: 10s - loss: 0.1883 - categorical_accuracy: 0.9417
54560/60000 [==========================>...] - ETA: 10s - loss: 0.1882 - categorical_accuracy: 0.9417
54592/60000 [==========================>...] - ETA: 10s - loss: 0.1882 - categorical_accuracy: 0.9417
54624/60000 [==========================>...] - ETA: 10s - loss: 0.1882 - categorical_accuracy: 0.9417
54656/60000 [==========================>...] - ETA: 10s - loss: 0.1882 - categorical_accuracy: 0.9417
54688/60000 [==========================>...] - ETA: 10s - loss: 0.1881 - categorical_accuracy: 0.9417
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1880 - categorical_accuracy: 0.9417 
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1879 - categorical_accuracy: 0.9418
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1879 - categorical_accuracy: 0.9418
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1880 - categorical_accuracy: 0.9417
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1880 - categorical_accuracy: 0.9417
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1879 - categorical_accuracy: 0.9418
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1879 - categorical_accuracy: 0.9418
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1879 - categorical_accuracy: 0.9418
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1878 - categorical_accuracy: 0.9418
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1877 - categorical_accuracy: 0.9418
55040/60000 [==========================>...] - ETA: 9s - loss: 0.1877 - categorical_accuracy: 0.9418
55072/60000 [==========================>...] - ETA: 9s - loss: 0.1876 - categorical_accuracy: 0.9419
55104/60000 [==========================>...] - ETA: 9s - loss: 0.1875 - categorical_accuracy: 0.9419
55136/60000 [==========================>...] - ETA: 9s - loss: 0.1875 - categorical_accuracy: 0.9419
55168/60000 [==========================>...] - ETA: 9s - loss: 0.1874 - categorical_accuracy: 0.9419
55200/60000 [==========================>...] - ETA: 9s - loss: 0.1874 - categorical_accuracy: 0.9419
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1873 - categorical_accuracy: 0.9420
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1873 - categorical_accuracy: 0.9420
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1872 - categorical_accuracy: 0.9420
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1872 - categorical_accuracy: 0.9420
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1872 - categorical_accuracy: 0.9420
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1871 - categorical_accuracy: 0.9420
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1871 - categorical_accuracy: 0.9420
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1871 - categorical_accuracy: 0.9420
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1870 - categorical_accuracy: 0.9420
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1869 - categorical_accuracy: 0.9421
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1868 - categorical_accuracy: 0.9421
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1867 - categorical_accuracy: 0.9421
55616/60000 [==========================>...] - ETA: 8s - loss: 0.1866 - categorical_accuracy: 0.9422
55648/60000 [==========================>...] - ETA: 8s - loss: 0.1866 - categorical_accuracy: 0.9422
55680/60000 [==========================>...] - ETA: 8s - loss: 0.1866 - categorical_accuracy: 0.9422
55712/60000 [==========================>...] - ETA: 8s - loss: 0.1866 - categorical_accuracy: 0.9422
55744/60000 [==========================>...] - ETA: 8s - loss: 0.1865 - categorical_accuracy: 0.9422
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1864 - categorical_accuracy: 0.9423
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1863 - categorical_accuracy: 0.9423
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1863 - categorical_accuracy: 0.9423
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1862 - categorical_accuracy: 0.9423
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1861 - categorical_accuracy: 0.9423
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1861 - categorical_accuracy: 0.9423
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1860 - categorical_accuracy: 0.9423
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1859 - categorical_accuracy: 0.9424
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1858 - categorical_accuracy: 0.9424
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1858 - categorical_accuracy: 0.9424
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1857 - categorical_accuracy: 0.9424
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1856 - categorical_accuracy: 0.9424
56160/60000 [===========================>..] - ETA: 7s - loss: 0.1855 - categorical_accuracy: 0.9425
56192/60000 [===========================>..] - ETA: 7s - loss: 0.1855 - categorical_accuracy: 0.9425
56224/60000 [===========================>..] - ETA: 7s - loss: 0.1855 - categorical_accuracy: 0.9425
56256/60000 [===========================>..] - ETA: 7s - loss: 0.1854 - categorical_accuracy: 0.9425
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1853 - categorical_accuracy: 0.9425
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1852 - categorical_accuracy: 0.9425
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1852 - categorical_accuracy: 0.9426
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1851 - categorical_accuracy: 0.9426
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1851 - categorical_accuracy: 0.9426
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1850 - categorical_accuracy: 0.9426
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1850 - categorical_accuracy: 0.9426
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1850 - categorical_accuracy: 0.9426
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1849 - categorical_accuracy: 0.9427
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1848 - categorical_accuracy: 0.9427
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1847 - categorical_accuracy: 0.9427
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1847 - categorical_accuracy: 0.9427
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1846 - categorical_accuracy: 0.9428
56704/60000 [===========================>..] - ETA: 6s - loss: 0.1845 - categorical_accuracy: 0.9428
56736/60000 [===========================>..] - ETA: 6s - loss: 0.1844 - categorical_accuracy: 0.9428
56768/60000 [===========================>..] - ETA: 6s - loss: 0.1843 - categorical_accuracy: 0.9429
56800/60000 [===========================>..] - ETA: 6s - loss: 0.1843 - categorical_accuracy: 0.9429
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1843 - categorical_accuracy: 0.9429
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1842 - categorical_accuracy: 0.9429
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1843 - categorical_accuracy: 0.9429
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1842 - categorical_accuracy: 0.9429
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1842 - categorical_accuracy: 0.9429
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1843 - categorical_accuracy: 0.9429
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1842 - categorical_accuracy: 0.9429
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1842 - categorical_accuracy: 0.9429
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1841 - categorical_accuracy: 0.9430
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1841 - categorical_accuracy: 0.9430
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1840 - categorical_accuracy: 0.9430
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1840 - categorical_accuracy: 0.9430
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1839 - categorical_accuracy: 0.9430
57248/60000 [===========================>..] - ETA: 5s - loss: 0.1838 - categorical_accuracy: 0.9431
57280/60000 [===========================>..] - ETA: 5s - loss: 0.1838 - categorical_accuracy: 0.9431
57312/60000 [===========================>..] - ETA: 5s - loss: 0.1837 - categorical_accuracy: 0.9431
57344/60000 [===========================>..] - ETA: 5s - loss: 0.1837 - categorical_accuracy: 0.9431
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1836 - categorical_accuracy: 0.9431
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1836 - categorical_accuracy: 0.9431
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1837 - categorical_accuracy: 0.9431
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1836 - categorical_accuracy: 0.9431
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1835 - categorical_accuracy: 0.9432
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1834 - categorical_accuracy: 0.9432
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1833 - categorical_accuracy: 0.9432
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1832 - categorical_accuracy: 0.9432
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1832 - categorical_accuracy: 0.9433
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1832 - categorical_accuracy: 0.9433
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1831 - categorical_accuracy: 0.9433
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1830 - categorical_accuracy: 0.9433
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1830 - categorical_accuracy: 0.9434
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1830 - categorical_accuracy: 0.9434
57824/60000 [===========================>..] - ETA: 4s - loss: 0.1829 - categorical_accuracy: 0.9434
57856/60000 [===========================>..] - ETA: 4s - loss: 0.1828 - categorical_accuracy: 0.9434
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1827 - categorical_accuracy: 0.9434
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1827 - categorical_accuracy: 0.9435
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1828 - categorical_accuracy: 0.9435
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1827 - categorical_accuracy: 0.9435
58016/60000 [============================>.] - ETA: 3s - loss: 0.1827 - categorical_accuracy: 0.9435
58048/60000 [============================>.] - ETA: 3s - loss: 0.1826 - categorical_accuracy: 0.9435
58080/60000 [============================>.] - ETA: 3s - loss: 0.1826 - categorical_accuracy: 0.9435
58112/60000 [============================>.] - ETA: 3s - loss: 0.1826 - categorical_accuracy: 0.9435
58144/60000 [============================>.] - ETA: 3s - loss: 0.1826 - categorical_accuracy: 0.9436
58176/60000 [============================>.] - ETA: 3s - loss: 0.1826 - categorical_accuracy: 0.9435
58208/60000 [============================>.] - ETA: 3s - loss: 0.1825 - categorical_accuracy: 0.9435
58240/60000 [============================>.] - ETA: 3s - loss: 0.1826 - categorical_accuracy: 0.9436
58272/60000 [============================>.] - ETA: 3s - loss: 0.1825 - categorical_accuracy: 0.9436
58304/60000 [============================>.] - ETA: 3s - loss: 0.1826 - categorical_accuracy: 0.9436
58336/60000 [============================>.] - ETA: 3s - loss: 0.1826 - categorical_accuracy: 0.9436
58368/60000 [============================>.] - ETA: 3s - loss: 0.1825 - categorical_accuracy: 0.9436
58400/60000 [============================>.] - ETA: 3s - loss: 0.1824 - categorical_accuracy: 0.9436
58432/60000 [============================>.] - ETA: 2s - loss: 0.1824 - categorical_accuracy: 0.9436
58464/60000 [============================>.] - ETA: 2s - loss: 0.1824 - categorical_accuracy: 0.9436
58496/60000 [============================>.] - ETA: 2s - loss: 0.1824 - categorical_accuracy: 0.9437
58528/60000 [============================>.] - ETA: 2s - loss: 0.1823 - categorical_accuracy: 0.9437
58560/60000 [============================>.] - ETA: 2s - loss: 0.1823 - categorical_accuracy: 0.9437
58592/60000 [============================>.] - ETA: 2s - loss: 0.1822 - categorical_accuracy: 0.9437
58624/60000 [============================>.] - ETA: 2s - loss: 0.1821 - categorical_accuracy: 0.9437
58656/60000 [============================>.] - ETA: 2s - loss: 0.1821 - categorical_accuracy: 0.9438
58688/60000 [============================>.] - ETA: 2s - loss: 0.1820 - categorical_accuracy: 0.9438
58720/60000 [============================>.] - ETA: 2s - loss: 0.1819 - categorical_accuracy: 0.9438
58752/60000 [============================>.] - ETA: 2s - loss: 0.1818 - categorical_accuracy: 0.9438
58784/60000 [============================>.] - ETA: 2s - loss: 0.1819 - categorical_accuracy: 0.9438
58816/60000 [============================>.] - ETA: 2s - loss: 0.1818 - categorical_accuracy: 0.9439
58848/60000 [============================>.] - ETA: 2s - loss: 0.1818 - categorical_accuracy: 0.9439
58880/60000 [============================>.] - ETA: 2s - loss: 0.1817 - categorical_accuracy: 0.9439
58912/60000 [============================>.] - ETA: 2s - loss: 0.1817 - categorical_accuracy: 0.9439
58944/60000 [============================>.] - ETA: 1s - loss: 0.1816 - categorical_accuracy: 0.9439
58976/60000 [============================>.] - ETA: 1s - loss: 0.1816 - categorical_accuracy: 0.9439
59008/60000 [============================>.] - ETA: 1s - loss: 0.1816 - categorical_accuracy: 0.9439
59040/60000 [============================>.] - ETA: 1s - loss: 0.1815 - categorical_accuracy: 0.9439
59072/60000 [============================>.] - ETA: 1s - loss: 0.1814 - categorical_accuracy: 0.9440
59104/60000 [============================>.] - ETA: 1s - loss: 0.1814 - categorical_accuracy: 0.9439
59136/60000 [============================>.] - ETA: 1s - loss: 0.1813 - categorical_accuracy: 0.9440
59168/60000 [============================>.] - ETA: 1s - loss: 0.1812 - categorical_accuracy: 0.9440
59200/60000 [============================>.] - ETA: 1s - loss: 0.1814 - categorical_accuracy: 0.9440
59232/60000 [============================>.] - ETA: 1s - loss: 0.1814 - categorical_accuracy: 0.9440
59264/60000 [============================>.] - ETA: 1s - loss: 0.1813 - categorical_accuracy: 0.9440
59296/60000 [============================>.] - ETA: 1s - loss: 0.1812 - categorical_accuracy: 0.9440
59328/60000 [============================>.] - ETA: 1s - loss: 0.1812 - categorical_accuracy: 0.9440
59360/60000 [============================>.] - ETA: 1s - loss: 0.1811 - categorical_accuracy: 0.9441
59392/60000 [============================>.] - ETA: 1s - loss: 0.1811 - categorical_accuracy: 0.9440
59424/60000 [============================>.] - ETA: 1s - loss: 0.1810 - categorical_accuracy: 0.9441
59456/60000 [============================>.] - ETA: 1s - loss: 0.1810 - categorical_accuracy: 0.9441
59488/60000 [============================>.] - ETA: 0s - loss: 0.1809 - categorical_accuracy: 0.9441
59520/60000 [============================>.] - ETA: 0s - loss: 0.1808 - categorical_accuracy: 0.9441
59552/60000 [============================>.] - ETA: 0s - loss: 0.1808 - categorical_accuracy: 0.9441
59584/60000 [============================>.] - ETA: 0s - loss: 0.1807 - categorical_accuracy: 0.9441
59616/60000 [============================>.] - ETA: 0s - loss: 0.1807 - categorical_accuracy: 0.9442
59648/60000 [============================>.] - ETA: 0s - loss: 0.1807 - categorical_accuracy: 0.9442
59680/60000 [============================>.] - ETA: 0s - loss: 0.1806 - categorical_accuracy: 0.9442
59712/60000 [============================>.] - ETA: 0s - loss: 0.1806 - categorical_accuracy: 0.9442
59744/60000 [============================>.] - ETA: 0s - loss: 0.1807 - categorical_accuracy: 0.9442
59776/60000 [============================>.] - ETA: 0s - loss: 0.1806 - categorical_accuracy: 0.9442
59808/60000 [============================>.] - ETA: 0s - loss: 0.1805 - categorical_accuracy: 0.9442
59840/60000 [============================>.] - ETA: 0s - loss: 0.1805 - categorical_accuracy: 0.9443
59872/60000 [============================>.] - ETA: 0s - loss: 0.1809 - categorical_accuracy: 0.9442
59904/60000 [============================>.] - ETA: 0s - loss: 0.1809 - categorical_accuracy: 0.9442
59936/60000 [============================>.] - ETA: 0s - loss: 0.1808 - categorical_accuracy: 0.9442
59968/60000 [============================>.] - ETA: 0s - loss: 0.1809 - categorical_accuracy: 0.9442
60000/60000 [==============================] - 117s 2ms/step - loss: 0.1808 - categorical_accuracy: 0.9442 - val_loss: 0.0588 - val_categorical_accuracy: 0.9817

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 17s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 4s
  672/10000 [=>............................] - ETA: 4s
  832/10000 [=>............................] - ETA: 3s
  992/10000 [=>............................] - ETA: 3s
 1120/10000 [==>...........................] - ETA: 3s
 1280/10000 [==>...........................] - ETA: 3s
 1408/10000 [===>..........................] - ETA: 3s
 1568/10000 [===>..........................] - ETA: 3s
 1696/10000 [====>.........................] - ETA: 3s
 1856/10000 [====>.........................] - ETA: 3s
 2016/10000 [=====>........................] - ETA: 3s
 2176/10000 [=====>........................] - ETA: 3s
 2336/10000 [======>.......................] - ETA: 3s
 2496/10000 [======>.......................] - ETA: 2s
 2656/10000 [======>.......................] - ETA: 2s
 2816/10000 [=======>......................] - ETA: 2s
 2976/10000 [=======>......................] - ETA: 2s
 3136/10000 [========>.....................] - ETA: 2s
 3264/10000 [========>.....................] - ETA: 2s
 3392/10000 [=========>....................] - ETA: 2s
 3552/10000 [=========>....................] - ETA: 2s
 3712/10000 [==========>...................] - ETA: 2s
 3872/10000 [==========>...................] - ETA: 2s
 4000/10000 [===========>..................] - ETA: 2s
 4160/10000 [===========>..................] - ETA: 2s
 4320/10000 [===========>..................] - ETA: 2s
 4480/10000 [============>.................] - ETA: 2s
 4640/10000 [============>.................] - ETA: 2s
 4800/10000 [=============>................] - ETA: 2s
 4928/10000 [=============>................] - ETA: 1s
 5088/10000 [==============>...............] - ETA: 1s
 5248/10000 [==============>...............] - ETA: 1s
 5376/10000 [===============>..............] - ETA: 1s
 5536/10000 [===============>..............] - ETA: 1s
 5696/10000 [================>.............] - ETA: 1s
 5824/10000 [================>.............] - ETA: 1s
 5984/10000 [================>.............] - ETA: 1s
 6144/10000 [=================>............] - ETA: 1s
 6272/10000 [=================>............] - ETA: 1s
 6432/10000 [==================>...........] - ETA: 1s
 6592/10000 [==================>...........] - ETA: 1s
 6752/10000 [===================>..........] - ETA: 1s
 6912/10000 [===================>..........] - ETA: 1s
 7072/10000 [====================>.........] - ETA: 1s
 7232/10000 [====================>.........] - ETA: 1s
 7392/10000 [=====================>........] - ETA: 1s
 7552/10000 [=====================>........] - ETA: 0s
 7712/10000 [======================>.......] - ETA: 0s
 7872/10000 [======================>.......] - ETA: 0s
 8032/10000 [=======================>......] - ETA: 0s
 8192/10000 [=======================>......] - ETA: 0s
 8352/10000 [========================>.....] - ETA: 0s
 8512/10000 [========================>.....] - ETA: 0s
 8672/10000 [=========================>....] - ETA: 0s
 8832/10000 [=========================>....] - ETA: 0s
 8960/10000 [=========================>....] - ETA: 0s
 9120/10000 [==========================>...] - ETA: 0s
 9280/10000 [==========================>...] - ETA: 0s
 9408/10000 [===========================>..] - ETA: 0s
 9568/10000 [===========================>..] - ETA: 0s
 9728/10000 [============================>.] - ETA: 0s
 9888/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 385us/step
[[1.4628224e-06 3.1487542e-07 4.0181763e-05 ... 9.9990237e-01
  3.8045882e-06 3.1714841e-05]
 [1.3309644e-05 2.5932145e-06 9.9995375e-01 ... 4.7643982e-09
  2.2148379e-05 5.2901467e-10]
 [3.7755992e-05 9.9708778e-01 4.1765079e-04 ... 5.0682115e-04
  6.2715140e-04 3.7932041e-05]
 ...
 [1.0670004e-07 7.2008277e-07 9.9656518e-07 ... 5.7449306e-06
  1.0969601e-04 4.1801838e-04]
 [7.4548922e-07 6.5547199e-09 9.9584092e-09 ... 2.1748802e-08
  2.8135208e-03 3.3133580e-07]
 [3.2048232e-05 5.7001148e-07 8.1042177e-05 ... 1.0714413e-08
  1.2649208e-05 1.2078974e-07]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.0587968731591478, 'accuracy_test:': 0.9817000031471252}

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
   4b43087..4c22a53  master     -> origin/master
Updating 4b43087..4c22a53
Fast-forward
 error_list/20200518/list_log_testall_20200518.md | 431 +++++++++++++++++++++++
 1 file changed, 431 insertions(+)
[master 149607c] ml_store
 1 file changed, 2045 insertions(+)
To github.com:arita37/mlmodels_store.git
   4c22a53..149607c  master -> master





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
{'loss': 0.4825948029756546, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-18 20:33:46.455522: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master 643f0ed] ml_store
 1 file changed, 234 insertions(+)
To github.com:arita37/mlmodels_store.git
   149607c..643f0ed  master -> master





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
[master 2334eb4] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   643f0ed..2334eb4  master -> master





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
	Data preprocessing and feature engineering runtime = 0.28s ...
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
 40%|████      | 2/5 [00:23<00:34, 11.51s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.9073568642089626, 'learning_rate': 0.14328832514753953, 'min_data_in_leaf': 7, 'num_leaves': 57} and reward: 0.391
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\t\x11C2\x83\xf3X\r\x00\x00\x00learning_rateq\x02G?\xc2WE\x974"LX\x10\x00\x00\x00min_data_in_leafq\x03K\x07X\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.391
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\t\x11C2\x83\xf3X\r\x00\x00\x00learning_rateq\x02G?\xc2WE\x974"LX\x10\x00\x00\x00min_data_in_leafq\x03K\x07X\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.391
 60%|██████    | 3/5 [00:53<00:34, 17.07s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.7549733495932232, 'learning_rate': 0.016750305454093735, 'min_data_in_leaf': 23, 'num_leaves': 39} and reward: 0.392
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8(\xbd\xde\xbbW\xceX\r\x00\x00\x00learning_rateq\x02G?\x91&\xfd\xf8~M\xa2X\x10\x00\x00\x00min_data_in_leafq\x03K\x17X\n\x00\x00\x00num_leavesq\x04K'u." and reward: 0.392
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8(\xbd\xde\xbbW\xceX\r\x00\x00\x00learning_rateq\x02G?\x91&\xfd\xf8~M\xa2X\x10\x00\x00\x00min_data_in_leafq\x03K\x17X\n\x00\x00\x00num_leavesq\x04K'u." and reward: 0.392
 80%|████████  | 4/5 [01:18<00:19, 19.47s/it] 80%|████████  | 4/5 [01:18<00:19, 19.53s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.9303815627756744, 'learning_rate': 0.04574633039693554, 'min_data_in_leaf': 9, 'num_leaves': 55} and reward: 0.3938
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xc5\xaf\x8e\x1d\x88VX\r\x00\x00\x00learning_rateq\x02G?\xa7l\x10!\xef\x06\xd0X\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K7u.' and reward: 0.3938
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xc5\xaf\x8e\x1d\x88VX\r\x00\x00\x00learning_rateq\x02G?\xa7l\x10!\xef\x06\xd0X\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K7u.' and reward: 0.3938
Time for Gradient Boosting hyperparameter optimization: 109.83776307106018
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.9303815627756744, 'learning_rate': 0.04574633039693554, 'min_data_in_leaf': 9, 'num_leaves': 55}
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
 40%|████      | 2/5 [00:58<01:27, 29.13s/it] 40%|████      | 2/5 [00:58<01:27, 29.13s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.03487357146527587, 'embedding_size_factor': 0.5871184710179056, 'layers.choice': 3, 'learning_rate': 0.0001496251265629178, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 0.00907721161723166} and reward: 0.291
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xa1\xda\xf2\xe1\xe0K\xd9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe2\xc9\xac\xac\xfc\xc8\x1eX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?#\x9c\x96\x0c\xed@\xa3X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x82\x97\x12\xb8Go\xd2u.' and reward: 0.291
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xa1\xda\xf2\xe1\xe0K\xd9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe2\xc9\xac\xac\xfc\xc8\x1eX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?#\x9c\x96\x0c\xed@\xa3X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x82\x97\x12\xb8Go\xd2u.' and reward: 0.291
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 121.13261151313782
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.72s of the -114.85s of remaining time.
Ensemble size: 64
Ensemble weights: 
[0.1875   0.140625 0.       0.140625 0.265625 0.265625]
	0.3998	 = Validation accuracy score
	1.61s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 236.52s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f55e0785940>

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
   2334eb4..13d9f6b  master     -> origin/master
Updating 2334eb4..13d9f6b
Fast-forward
 .../20200518/list_log_pullrequest_20200518.md      |   2 +-
 error_list/20200518/list_log_testall_20200518.md   | 175 +++++++++++++++++++++
 2 files changed, 176 insertions(+), 1 deletion(-)
[master 53c5033] ml_store
 1 file changed, 205 insertions(+)
To github.com:arita37/mlmodels_store.git
   13d9f6b..53c5033  master -> master





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
[master fd65d6f] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   53c5033..fd65d6f  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.34it/s, avg_epoch_loss=5.26]
INFO:root:Epoch[0] Elapsed time 2.996 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.257627
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.257626581192016 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f81804b7400>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f81804b7400>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 70.91it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1091.4256998697917,
    "abs_error": 375.50616455078125,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.4880850041110967,
    "sMAPE": 0.517711823552923,
    "MSIS": 99.52338398780857,
    "QuantileLoss[0.5]": 375.5061492919922,
    "Coverage[0.5]": 1.0,
    "RMSE": 33.036732584651766,
    "NRMSE": 0.6955101596768792,
    "ND": 0.658782744825932,
    "wQuantileLoss[0.5]": 0.6587827180561266,
    "mean_wQuantileLoss": 0.6587827180561266,
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
100%|██████████| 10/10 [00:01<00:00,  6.71it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.492 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f8147f91d30>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f8147f91d30>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 136.65it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.18it/s, avg_epoch_loss=5.27]
INFO:root:Epoch[0] Elapsed time 1.933 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.265795
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.265795183181763 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f8147e8fc18>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f8147e8fc18>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 130.12it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 265.86867268880206,
    "abs_error": 171.6577606201172,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.1373957084010737,
    "sMAPE": 0.284597907378925,
    "MSIS": 45.495832380201776,
    "QuantileLoss[0.5]": 171.65776443481445,
    "Coverage[0.5]": 0.75,
    "RMSE": 16.30547983620237,
    "NRMSE": 0.34327325970952355,
    "ND": 0.3011539660002056,
    "wQuantileLoss[0.5]": 0.3011539726926569,
    "mean_wQuantileLoss": 0.3011539726926569,
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
 30%|███       | 3/10 [00:13<00:31,  4.55s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:25<00:17,  4.39s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:37<00:04,  4.28s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:41<00:00,  4.18s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 41.768 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.862701
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.862701416015625 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f8147f9a860>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f8147f9a860>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 126.73it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 54198.421875,
    "abs_error": 2742.25390625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 18.17003678160124,
    "sMAPE": 1.4157378234091285,
    "MSIS": 726.8014712640495,
    "QuantileLoss[0.5]": 2742.2538452148438,
    "Coverage[0.5]": 1.0,
    "RMSE": 232.8055451981331,
    "NRMSE": 4.901169372592276,
    "ND": 4.810971765350877,
    "wQuantileLoss[0.5]": 4.810971658271655,
    "mean_wQuantileLoss": 4.810971658271655,
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
100%|██████████| 10/10 [00:00<00:00, 44.39it/s, avg_epoch_loss=5.05]
INFO:root:Epoch[0] Elapsed time 0.226 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.053321
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.053320741653442 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f8144db7fd0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f8144db7fd0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 120.95it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 342.1483968098958,
    "abs_error": 201.63221740722656,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.3360049550098527,
    "sMAPE": 0.32929934041508585,
    "MSIS": 53.44019739156235,
    "QuantileLoss[0.5]": 201.63220977783203,
    "Coverage[0.5]": 0.75,
    "RMSE": 18.49725376400226,
    "NRMSE": 0.38941586871583705,
    "ND": 0.35374073229337993,
    "wQuantileLoss[0.5]": 0.35374071890847725,
    "mean_wQuantileLoss": 0.35374071890847725,
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
100%|██████████| 10/10 [00:01<00:00,  7.61it/s, avg_epoch_loss=144]
INFO:root:Epoch[0] Elapsed time 1.315 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=144.412806
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 144.41280616614463 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f8147ea5a20>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f8147ea5a20>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 130.59it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
 10%|█         | 1/10 [01:54<17:06, 114.02s/it, avg_epoch_loss=0.488] 20%|██        | 2/10 [04:44<17:26, 130.82s/it, avg_epoch_loss=0.47]  30%|███       | 3/10 [07:50<17:11, 147.41s/it, avg_epoch_loss=0.453] 40%|████      | 4/10 [11:05<16:10, 161.74s/it, avg_epoch_loss=0.437] 50%|█████     | 5/10 [14:29<14:32, 174.57s/it, avg_epoch_loss=0.424] 60%|██████    | 6/10 [17:41<11:58, 179.57s/it, avg_epoch_loss=0.414] 70%|███████   | 7/10 [20:39<08:58, 179.35s/it, avg_epoch_loss=0.407] 80%|████████  | 8/10 [23:56<06:09, 184.62s/it, avg_epoch_loss=0.404] 90%|█████████ | 9/10 [27:24<03:11, 191.59s/it, avg_epoch_loss=0.401]100%|██████████| 10/10 [31:13<00:00, 202.74s/it, avg_epoch_loss=0.399]100%|██████████| 10/10 [31:13<00:00, 187.35s/it, avg_epoch_loss=0.399]
INFO:root:Epoch[0] Elapsed time 1873.490 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.398938
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3989381194114685 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f8144edb4a8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f8144edb4a8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 19.69it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
From github.com:arita37/mlmodels_store
   fd65d6f..5d727b0  master     -> origin/master
Updating fd65d6f..5d727b0
Fast-forward
 error_list/20200518/list_log_pullrequest_20200518.md | 2 +-
 error_list/20200518/list_log_testall_20200518.md     | 7 +++++++
 2 files changed, 8 insertions(+), 1 deletion(-)
[master 82ee808] ml_store
 1 file changed, 506 insertions(+)
To github.com:arita37/mlmodels_store.git
   5d727b0..82ee808  master -> master





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
{'roc_auc_score': 0.9642857142857143}

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f77cc05e5c0> 

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
