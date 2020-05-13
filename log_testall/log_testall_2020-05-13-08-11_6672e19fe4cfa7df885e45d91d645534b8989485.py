
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
[master 38ca7a0] ml_store
 1 file changed, 59 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py
To github.com:arita37/mlmodels_store.git
   13a75af..38ca7a0  master -> master





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
[master 8939598] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   38ca7a0..8939598  master -> master





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
[master 8ecb6ae] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   8939598..8ecb6ae  master -> master





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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         8           sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-13 08:11:55.481541: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 08:11:55.494843: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 08:11:55.495070: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557ed604a910 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 08:11:55.495088: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 1s - loss: 0.4350 - binary_crossentropy: 5.6344500/500 [==============================] - 1s 1ms/sample - loss: 0.3726 - binary_crossentropy: 4.2465 - val_loss: 0.3760 - val_binary_crossentropy: 4.2055

  #### metrics   #################################################### 
{'MSE': 0.3741492075122274}

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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         8           sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
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
Total params: 463
Trainable params: 463
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2508 - binary_crossentropy: 0.6949500/500 [==============================] - 1s 1ms/sample - loss: 0.2526 - binary_crossentropy: 0.6986 - val_loss: 0.2562 - val_binary_crossentropy: 0.7059

  #### metrics   #################################################### 
{'MSE': 0.25412070274049714}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 627
Trainable params: 627
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2475 - binary_crossentropy: 0.6862500/500 [==============================] - 1s 2ms/sample - loss: 0.2511 - binary_crossentropy: 0.7214 - val_loss: 0.2499 - val_binary_crossentropy: 0.6930

  #### metrics   #################################################### 
{'MSE': 0.25035974974894015}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 398
Trainable params: 398
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2590 - binary_crossentropy: 0.8436500/500 [==============================] - 1s 3ms/sample - loss: 0.2638 - binary_crossentropy: 0.7499 - val_loss: 0.2561 - val_binary_crossentropy: 0.7329

  #### metrics   #################################################### 
{'MSE': 0.2568404094359871}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 398
Trainable params: 398
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 7, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
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
Total params: 158
Trainable params: 158
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2735 - binary_crossentropy: 1.0054500/500 [==============================] - 2s 3ms/sample - loss: 0.2615 - binary_crossentropy: 0.8474 - val_loss: 0.2561 - val_binary_crossentropy: 0.7837

  #### metrics   #################################################### 
{'MSE': 0.2579639074239966}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 7, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-13 08:13:10.623569: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:13:10.625450: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:13:10.631203: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 08:13:10.641319: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 08:13:10.643522: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 08:13:10.645483: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:13:10.647094: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2550 - val_binary_crossentropy: 0.7032
2020-05-13 08:13:11.910988: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:13:11.913150: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:13:11.917223: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 08:13:11.925742: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-13 08:13:11.927107: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 08:13:11.928364: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:13:11.929686: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25629159941146035}

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
2020-05-13 08:13:34.193865: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:13:34.195255: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:13:34.198916: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 08:13:34.204647: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 08:13:34.205862: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 08:13:34.206909: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:13:34.208018: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2488 - val_binary_crossentropy: 0.6908
2020-05-13 08:13:35.573894: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:13:35.574868: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:13:35.577301: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 08:13:35.582928: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-13 08:13:35.584180: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 08:13:35.585051: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:13:35.585940: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2483375839415416}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-13 08:14:08.551107: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:14:08.555555: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:14:08.571170: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 08:14:08.597712: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 08:14:08.602889: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 08:14:08.607246: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:14:08.610851: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.4334 - binary_crossentropy: 1.0740 - val_loss: 0.2518 - val_binary_crossentropy: 0.6967
2020-05-13 08:14:10.705458: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:14:10.710029: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:14:10.720635: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 08:14:10.742067: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-13 08:14:10.746150: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-13 08:14:10.749891: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-13 08:14:10.753904: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.23737034714154248}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
100/500 [=====>........................] - ETA: 6s - loss: 0.5000 - binary_crossentropy: 7.7125500/500 [==============================] - 4s 8ms/sample - loss: 0.5060 - binary_crossentropy: 7.8050 - val_loss: 0.5360 - val_binary_crossentropy: 8.2678

  #### metrics   #################################################### 
{'MSE': 0.521}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 2)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         12          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         6           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_5[0][0]           
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
Total params: 236
Trainable params: 236
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3354 - binary_crossentropy: 0.9224500/500 [==============================] - 5s 9ms/sample - loss: 0.3563 - binary_crossentropy: 0.9943 - val_loss: 0.3286 - val_binary_crossentropy: 0.9300

  #### metrics   #################################################### 
{'MSE': 0.3379721854349682}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 2)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         12          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         6           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_5[0][0]           
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
Total params: 236
Trainable params: 236
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
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
100/500 [=====>........................] - ETA: 6s - loss: 0.2721 - binary_crossentropy: 0.7491500/500 [==============================] - 5s 9ms/sample - loss: 0.2878 - binary_crossentropy: 0.7828 - val_loss: 0.2868 - val_binary_crossentropy: 0.7769

  #### metrics   #################################################### 
{'MSE': 0.2850450493762223}

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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
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
regionsequence_sum (InputLayer) [(None, 7)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 3)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 8s - loss: 0.2619 - binary_crossentropy: 0.7188500/500 [==============================] - 6s 12ms/sample - loss: 0.2705 - binary_crossentropy: 0.7360 - val_loss: 0.2706 - val_binary_crossentropy: 0.7359

  #### metrics   #################################################### 
{'MSE': 0.26955280935582954}

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
regionsequence_sum (InputLayer) [(None, 7)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 3)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 1,377
Trainable params: 1,377
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2822 - binary_crossentropy: 0.7624500/500 [==============================] - 6s 12ms/sample - loss: 0.2839 - binary_crossentropy: 0.8962 - val_loss: 0.2724 - val_binary_crossentropy: 0.8450

  #### metrics   #################################################### 
{'MSE': 0.2758326427374334}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 1,377
Trainable params: 1,377
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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         12          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         4           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_11[0][0]                    
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
Total params: 2,899
Trainable params: 2,819
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2493 - binary_crossentropy: 0.6917500/500 [==============================] - 6s 13ms/sample - loss: 0.2534 - binary_crossentropy: 0.7241 - val_loss: 0.2497 - val_binary_crossentropy: 0.7166

  #### metrics   #################################################### 
{'MSE': 0.25043435028819244}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         12          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         4           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         12          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_11[0][0]                    
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
Total params: 2,899
Trainable params: 2,819
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         32          sequence_max[0][0]               
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
Total params: 449
Trainable params: 449
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2486 - binary_crossentropy: 0.6904500/500 [==============================] - 7s 13ms/sample - loss: 0.2490 - binary_crossentropy: 0.6912 - val_loss: 0.2500 - val_binary_crossentropy: 0.6931

  #### metrics   #################################################### 
{'MSE': 0.25006923044118967}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         32          sequence_max[0][0]               
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
Total params: 449
Trainable params: 449
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 2,014
Trainable params: 2,014
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 7s 13ms/sample - loss: 0.2506 - binary_crossentropy: 0.7200 - val_loss: 0.2501 - val_binary_crossentropy: 0.6934

  #### metrics   #################################################### 
{'MSE': 0.2502382129837185}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_max[0][0]               
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
Total params: 346
Trainable params: 346
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2472 - binary_crossentropy: 0.6873500/500 [==============================] - 7s 14ms/sample - loss: 0.2586 - binary_crossentropy: 0.7107 - val_loss: 0.2536 - val_binary_crossentropy: 0.7004

  #### metrics   #################################################### 
{'MSE': 0.2554812801561057}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_max[0][0]               
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
Total params: 346
Trainable params: 346
Non-trainable params: 0
__________________________________________________________________________________________________

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   8ecb6ae..3275247  master     -> origin/master
   e8ab980..0f8db46  dev        -> origin/dev
Updating 8ecb6ae..3275247
Fast-forward
 error_list/20200513/list_log_test_cli_20200513.md  | 138 ++---
 ...-10_6672e19fe4cfa7df885e45d91d645534b8989485.py | 610 +++++++++++++++++++++
 2 files changed, 679 insertions(+), 69 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-13-08-10_6672e19fe4cfa7df885e45d91d645534b8989485.py
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
[master ede3567] ml_store
 1 file changed, 5669 insertions(+)
To github.com:arita37/mlmodels_store.git
   3275247..ede3567  master -> master





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
[master 747f1e9] ml_store
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   ede3567..747f1e9  master -> master





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
[master 432cdb8] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   747f1e9..432cdb8  master -> master





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
[master 46f544c] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   432cdb8..46f544c  master -> master





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

2020-05-13 08:27:23.808102: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 08:27:23.812182: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 08:27:23.812314: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e6d9a4fb20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 08:27:23.812328: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3837
256/354 [====================>.........] - ETA: 3s - loss: 1.2188
354/354 [==============================] - 15s 42ms/step - loss: 1.4462 - val_loss: 2.1393

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
[master c5eb746] ml_store
 1 file changed, 149 insertions(+)
To github.com:arita37/mlmodels_store.git
   46f544c..c5eb746  master -> master





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
[master b7f7347] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   c5eb746..b7f7347  master -> master





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
[master 5490d92] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   b7f7347..5490d92  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3260416/17464789 [====>.........................] - ETA: 0s
10518528/17464789 [=================>............] - ETA: 0s
16760832/17464789 [===========================>..] - ETA: 0s
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
2020-05-13 08:28:24.390477: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 08:28:24.394389: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 08:28:24.394522: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56059e731140 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 08:28:24.394537: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5516 - accuracy: 0.5075 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6104 - accuracy: 0.5037
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6705 - accuracy: 0.4997
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6053 - accuracy: 0.5040
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5261 - accuracy: 0.5092
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5001 - accuracy: 0.5109
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5420 - accuracy: 0.5081
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5746 - accuracy: 0.5060
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5700 - accuracy: 0.5063
11000/25000 [============>.................] - ETA: 4s - loss: 7.5955 - accuracy: 0.5046
12000/25000 [=============>................] - ETA: 4s - loss: 7.6372 - accuracy: 0.5019
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6371 - accuracy: 0.5019
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6535 - accuracy: 0.5009
15000/25000 [=================>............] - ETA: 3s - loss: 7.6503 - accuracy: 0.5011
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6340 - accuracy: 0.5021
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6486 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6504 - accuracy: 0.5011
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6561 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6506 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6541 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6602 - accuracy: 0.5004
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
(<mlmodels.util.Model_empty object at 0x7f6d619b1320>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f6d604b5cf8> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6513 - accuracy: 0.5010
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4596 - accuracy: 0.5135 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5797 - accuracy: 0.5057
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5785 - accuracy: 0.5058
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6789 - accuracy: 0.4992
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6845 - accuracy: 0.4988
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6316 - accuracy: 0.5023
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6091 - accuracy: 0.5038
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5831 - accuracy: 0.5054
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5516 - accuracy: 0.5075
11000/25000 [============>.................] - ETA: 4s - loss: 7.5858 - accuracy: 0.5053
12000/25000 [=============>................] - ETA: 3s - loss: 7.6053 - accuracy: 0.5040
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5817 - accuracy: 0.5055
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5812 - accuracy: 0.5056
15000/25000 [=================>............] - ETA: 3s - loss: 7.5889 - accuracy: 0.5051
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5842 - accuracy: 0.5054
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6035 - accuracy: 0.5041
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6053 - accuracy: 0.5040
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6174 - accuracy: 0.5032
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6183 - accuracy: 0.5031
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6433 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6457 - accuracy: 0.5014
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6506 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 9s 360us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6436 - accuracy: 0.5015 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6717 - accuracy: 0.4997
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6245 - accuracy: 0.5027
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7402 - accuracy: 0.4952
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7407 - accuracy: 0.4952
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7017 - accuracy: 0.4977
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6743 - accuracy: 0.4995
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6581 - accuracy: 0.5006
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6344 - accuracy: 0.5021
11000/25000 [============>.................] - ETA: 4s - loss: 7.6541 - accuracy: 0.5008
12000/25000 [=============>................] - ETA: 3s - loss: 7.6385 - accuracy: 0.5018
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6230 - accuracy: 0.5028
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6228 - accuracy: 0.5029
15000/25000 [=================>............] - ETA: 3s - loss: 7.6564 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6599 - accuracy: 0.5004
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6558 - accuracy: 0.5007
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6751 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6637 - accuracy: 0.5002
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
25000/25000 [==============================] - 9s 368us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   5490d92..14480f9  master     -> origin/master
Updating 5490d92..14480f9
Fast-forward
 .../20200513/list_log_pullrequest_20200513.md      |   2 +-
 error_list/20200513/list_log_test_cli_20200513.md  | 138 ++++++++++-----------
 2 files changed, 70 insertions(+), 70 deletions(-)
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
[master 26b48c7] ml_store
 1 file changed, 324 insertions(+)
To github.com:arita37/mlmodels_store.git
   14480f9..26b48c7  master -> master





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

13/13 [==============================] - 1s 109ms/step - loss: nan
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
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
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
[master 71aa2de] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   26b48c7..71aa2de  master -> master





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
 2670592/11490434 [=====>........................] - ETA: 0s
10739712/11490434 [===========================>..] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:21 - loss: 2.3334 - categorical_accuracy: 0.0625
   64/60000 [..............................] - ETA: 4:41 - loss: 2.2617 - categorical_accuracy: 0.1719
   96/60000 [..............................] - ETA: 3:45 - loss: 2.2053 - categorical_accuracy: 0.2292
  128/60000 [..............................] - ETA: 3:16 - loss: 2.1488 - categorical_accuracy: 0.3047
  160/60000 [..............................] - ETA: 2:59 - loss: 2.1449 - categorical_accuracy: 0.2937
  192/60000 [..............................] - ETA: 2:46 - loss: 2.1188 - categorical_accuracy: 0.3073
  224/60000 [..............................] - ETA: 2:38 - loss: 2.1338 - categorical_accuracy: 0.2991
  256/60000 [..............................] - ETA: 2:32 - loss: 2.1148 - categorical_accuracy: 0.3125
  288/60000 [..............................] - ETA: 2:27 - loss: 2.0794 - categorical_accuracy: 0.3299
  320/60000 [..............................] - ETA: 2:23 - loss: 2.0668 - categorical_accuracy: 0.3313
  352/60000 [..............................] - ETA: 2:19 - loss: 2.0453 - categorical_accuracy: 0.3409
  384/60000 [..............................] - ETA: 2:16 - loss: 2.0122 - categorical_accuracy: 0.3542
  416/60000 [..............................] - ETA: 2:14 - loss: 1.9867 - categorical_accuracy: 0.3606
  448/60000 [..............................] - ETA: 2:12 - loss: 1.9595 - categorical_accuracy: 0.3638
  480/60000 [..............................] - ETA: 2:11 - loss: 1.9153 - categorical_accuracy: 0.3833
  512/60000 [..............................] - ETA: 2:10 - loss: 1.8671 - categorical_accuracy: 0.4004
  544/60000 [..............................] - ETA: 2:09 - loss: 1.8780 - categorical_accuracy: 0.3952
  576/60000 [..............................] - ETA: 2:08 - loss: 1.8552 - categorical_accuracy: 0.3993
  608/60000 [..............................] - ETA: 2:07 - loss: 1.8311 - categorical_accuracy: 0.4112
  640/60000 [..............................] - ETA: 2:06 - loss: 1.8017 - categorical_accuracy: 0.4141
  672/60000 [..............................] - ETA: 2:05 - loss: 1.7749 - categorical_accuracy: 0.4241
  704/60000 [..............................] - ETA: 2:04 - loss: 1.7472 - categorical_accuracy: 0.4361
  736/60000 [..............................] - ETA: 2:03 - loss: 1.7147 - categorical_accuracy: 0.4470
  768/60000 [..............................] - ETA: 2:02 - loss: 1.6788 - categorical_accuracy: 0.4609
  800/60000 [..............................] - ETA: 2:01 - loss: 1.6521 - categorical_accuracy: 0.4688
  832/60000 [..............................] - ETA: 2:01 - loss: 1.6228 - categorical_accuracy: 0.4796
  864/60000 [..............................] - ETA: 2:00 - loss: 1.6013 - categorical_accuracy: 0.4896
  896/60000 [..............................] - ETA: 1:59 - loss: 1.5778 - categorical_accuracy: 0.4967
  928/60000 [..............................] - ETA: 1:59 - loss: 1.5459 - categorical_accuracy: 0.5075
  960/60000 [..............................] - ETA: 1:58 - loss: 1.5248 - categorical_accuracy: 0.5177
  992/60000 [..............................] - ETA: 1:58 - loss: 1.5097 - categorical_accuracy: 0.5222
 1024/60000 [..............................] - ETA: 1:59 - loss: 1.4885 - categorical_accuracy: 0.5293
 1056/60000 [..............................] - ETA: 1:58 - loss: 1.4710 - categorical_accuracy: 0.5369
 1088/60000 [..............................] - ETA: 1:58 - loss: 1.4483 - categorical_accuracy: 0.5441
 1120/60000 [..............................] - ETA: 1:57 - loss: 1.4360 - categorical_accuracy: 0.5473
 1152/60000 [..............................] - ETA: 1:57 - loss: 1.4215 - categorical_accuracy: 0.5512
 1184/60000 [..............................] - ETA: 1:56 - loss: 1.4097 - categorical_accuracy: 0.5574
 1216/60000 [..............................] - ETA: 1:56 - loss: 1.3928 - categorical_accuracy: 0.5641
 1248/60000 [..............................] - ETA: 1:56 - loss: 1.3799 - categorical_accuracy: 0.5665
 1280/60000 [..............................] - ETA: 1:56 - loss: 1.3699 - categorical_accuracy: 0.5695
 1312/60000 [..............................] - ETA: 1:56 - loss: 1.3619 - categorical_accuracy: 0.5724
 1344/60000 [..............................] - ETA: 1:55 - loss: 1.3462 - categorical_accuracy: 0.5774
 1376/60000 [..............................] - ETA: 1:55 - loss: 1.3292 - categorical_accuracy: 0.5814
 1408/60000 [..............................] - ETA: 1:54 - loss: 1.3139 - categorical_accuracy: 0.5859
 1440/60000 [..............................] - ETA: 1:54 - loss: 1.2989 - categorical_accuracy: 0.5910
 1472/60000 [..............................] - ETA: 1:54 - loss: 1.2833 - categorical_accuracy: 0.5965
 1504/60000 [..............................] - ETA: 1:53 - loss: 1.2677 - categorical_accuracy: 0.6011
 1536/60000 [..............................] - ETA: 1:53 - loss: 1.2611 - categorical_accuracy: 0.6029
 1568/60000 [..............................] - ETA: 1:53 - loss: 1.2564 - categorical_accuracy: 0.6033
 1600/60000 [..............................] - ETA: 1:52 - loss: 1.2453 - categorical_accuracy: 0.6069
 1632/60000 [..............................] - ETA: 1:52 - loss: 1.2388 - categorical_accuracy: 0.6097
 1664/60000 [..............................] - ETA: 1:52 - loss: 1.2283 - categorical_accuracy: 0.6136
 1696/60000 [..............................] - ETA: 1:52 - loss: 1.2158 - categorical_accuracy: 0.6179
 1728/60000 [..............................] - ETA: 1:52 - loss: 1.1988 - categorical_accuracy: 0.6238
 1760/60000 [..............................] - ETA: 1:51 - loss: 1.1883 - categorical_accuracy: 0.6261
 1792/60000 [..............................] - ETA: 1:51 - loss: 1.1760 - categorical_accuracy: 0.6283
 1824/60000 [..............................] - ETA: 1:51 - loss: 1.1651 - categorical_accuracy: 0.6321
 1856/60000 [..............................] - ETA: 1:51 - loss: 1.1552 - categorical_accuracy: 0.6347
 1888/60000 [..............................] - ETA: 1:51 - loss: 1.1465 - categorical_accuracy: 0.6372
 1920/60000 [..............................] - ETA: 1:51 - loss: 1.1412 - categorical_accuracy: 0.6396
 1952/60000 [..............................] - ETA: 1:50 - loss: 1.1344 - categorical_accuracy: 0.6424
 1984/60000 [..............................] - ETA: 1:50 - loss: 1.1199 - categorical_accuracy: 0.6477
 2016/60000 [>.............................] - ETA: 1:50 - loss: 1.1127 - categorical_accuracy: 0.6493
 2048/60000 [>.............................] - ETA: 1:50 - loss: 1.1071 - categorical_accuracy: 0.6509
 2080/60000 [>.............................] - ETA: 1:50 - loss: 1.0969 - categorical_accuracy: 0.6538
 2112/60000 [>.............................] - ETA: 1:49 - loss: 1.0897 - categorical_accuracy: 0.6567
 2144/60000 [>.............................] - ETA: 1:49 - loss: 1.0810 - categorical_accuracy: 0.6581
 2176/60000 [>.............................] - ETA: 1:49 - loss: 1.0711 - categorical_accuracy: 0.6618
 2208/60000 [>.............................] - ETA: 1:49 - loss: 1.0659 - categorical_accuracy: 0.6635
 2240/60000 [>.............................] - ETA: 1:49 - loss: 1.0576 - categorical_accuracy: 0.6661
 2272/60000 [>.............................] - ETA: 1:48 - loss: 1.0483 - categorical_accuracy: 0.6686
 2304/60000 [>.............................] - ETA: 1:48 - loss: 1.0398 - categorical_accuracy: 0.6719
 2336/60000 [>.............................] - ETA: 1:48 - loss: 1.0317 - categorical_accuracy: 0.6751
 2368/60000 [>.............................] - ETA: 1:48 - loss: 1.0277 - categorical_accuracy: 0.6778
 2400/60000 [>.............................] - ETA: 1:48 - loss: 1.0177 - categorical_accuracy: 0.6804
 2432/60000 [>.............................] - ETA: 1:48 - loss: 1.0126 - categorical_accuracy: 0.6817
 2464/60000 [>.............................] - ETA: 1:48 - loss: 1.0057 - categorical_accuracy: 0.6834
 2496/60000 [>.............................] - ETA: 1:48 - loss: 1.0020 - categorical_accuracy: 0.6847
 2528/60000 [>.............................] - ETA: 1:47 - loss: 0.9936 - categorical_accuracy: 0.6871
 2560/60000 [>.............................] - ETA: 1:47 - loss: 0.9895 - categorical_accuracy: 0.6891
 2592/60000 [>.............................] - ETA: 1:47 - loss: 0.9812 - categorical_accuracy: 0.6917
 2624/60000 [>.............................] - ETA: 1:47 - loss: 0.9767 - categorical_accuracy: 0.6944
 2656/60000 [>.............................] - ETA: 1:46 - loss: 0.9721 - categorical_accuracy: 0.6962
 2688/60000 [>.............................] - ETA: 1:46 - loss: 0.9669 - categorical_accuracy: 0.6983
 2720/60000 [>.............................] - ETA: 1:46 - loss: 0.9619 - categorical_accuracy: 0.6996
 2752/60000 [>.............................] - ETA: 1:46 - loss: 0.9584 - categorical_accuracy: 0.7017
 2784/60000 [>.............................] - ETA: 1:46 - loss: 0.9531 - categorical_accuracy: 0.7033
 2816/60000 [>.............................] - ETA: 1:46 - loss: 0.9486 - categorical_accuracy: 0.7049
 2848/60000 [>.............................] - ETA: 1:46 - loss: 0.9402 - categorical_accuracy: 0.7075
 2880/60000 [>.............................] - ETA: 1:46 - loss: 0.9377 - categorical_accuracy: 0.7080
 2912/60000 [>.............................] - ETA: 1:46 - loss: 0.9340 - categorical_accuracy: 0.7098
 2944/60000 [>.............................] - ETA: 1:46 - loss: 0.9269 - categorical_accuracy: 0.7120
 2976/60000 [>.............................] - ETA: 1:46 - loss: 0.9236 - categorical_accuracy: 0.7127
 3008/60000 [>.............................] - ETA: 1:46 - loss: 0.9190 - categorical_accuracy: 0.7138
 3040/60000 [>.............................] - ETA: 1:46 - loss: 0.9159 - categorical_accuracy: 0.7145
 3072/60000 [>.............................] - ETA: 1:45 - loss: 0.9091 - categorical_accuracy: 0.7171
 3104/60000 [>.............................] - ETA: 1:45 - loss: 0.9035 - categorical_accuracy: 0.7191
 3136/60000 [>.............................] - ETA: 1:45 - loss: 0.8989 - categorical_accuracy: 0.7207
 3168/60000 [>.............................] - ETA: 1:45 - loss: 0.8926 - categorical_accuracy: 0.7229
 3200/60000 [>.............................] - ETA: 1:45 - loss: 0.8856 - categorical_accuracy: 0.7247
 3232/60000 [>.............................] - ETA: 1:45 - loss: 0.8787 - categorical_accuracy: 0.7271
 3264/60000 [>.............................] - ETA: 1:45 - loss: 0.8762 - categorical_accuracy: 0.7282
 3296/60000 [>.............................] - ETA: 1:45 - loss: 0.8740 - categorical_accuracy: 0.7294
 3328/60000 [>.............................] - ETA: 1:45 - loss: 0.8707 - categorical_accuracy: 0.7302
 3360/60000 [>.............................] - ETA: 1:45 - loss: 0.8668 - categorical_accuracy: 0.7315
 3392/60000 [>.............................] - ETA: 1:44 - loss: 0.8647 - categorical_accuracy: 0.7320
 3424/60000 [>.............................] - ETA: 1:44 - loss: 0.8592 - categorical_accuracy: 0.7339
 3456/60000 [>.............................] - ETA: 1:44 - loss: 0.8578 - categorical_accuracy: 0.7347
 3488/60000 [>.............................] - ETA: 1:44 - loss: 0.8521 - categorical_accuracy: 0.7365
 3520/60000 [>.............................] - ETA: 1:44 - loss: 0.8482 - categorical_accuracy: 0.7378
 3552/60000 [>.............................] - ETA: 1:44 - loss: 0.8435 - categorical_accuracy: 0.7396
 3584/60000 [>.............................] - ETA: 1:44 - loss: 0.8414 - categorical_accuracy: 0.7400
 3616/60000 [>.............................] - ETA: 1:44 - loss: 0.8361 - categorical_accuracy: 0.7420
 3648/60000 [>.............................] - ETA: 1:44 - loss: 0.8369 - categorical_accuracy: 0.7426
 3680/60000 [>.............................] - ETA: 1:43 - loss: 0.8333 - categorical_accuracy: 0.7437
 3712/60000 [>.............................] - ETA: 1:43 - loss: 0.8285 - categorical_accuracy: 0.7452
 3744/60000 [>.............................] - ETA: 1:43 - loss: 0.8232 - categorical_accuracy: 0.7468
 3776/60000 [>.............................] - ETA: 1:43 - loss: 0.8209 - categorical_accuracy: 0.7476
 3808/60000 [>.............................] - ETA: 1:43 - loss: 0.8171 - categorical_accuracy: 0.7489
 3840/60000 [>.............................] - ETA: 1:43 - loss: 0.8135 - categorical_accuracy: 0.7497
 3872/60000 [>.............................] - ETA: 1:43 - loss: 0.8122 - categorical_accuracy: 0.7505
 3904/60000 [>.............................] - ETA: 1:43 - loss: 0.8095 - categorical_accuracy: 0.7518
 3936/60000 [>.............................] - ETA: 1:43 - loss: 0.8045 - categorical_accuracy: 0.7533
 3968/60000 [>.............................] - ETA: 1:43 - loss: 0.8013 - categorical_accuracy: 0.7543
 4000/60000 [=>............................] - ETA: 1:42 - loss: 0.7969 - categorical_accuracy: 0.7555
 4032/60000 [=>............................] - ETA: 1:42 - loss: 0.7920 - categorical_accuracy: 0.7569
 4064/60000 [=>............................] - ETA: 1:42 - loss: 0.7878 - categorical_accuracy: 0.7584
 4096/60000 [=>............................] - ETA: 1:42 - loss: 0.7831 - categorical_accuracy: 0.7600
 4128/60000 [=>............................] - ETA: 1:42 - loss: 0.7802 - categorical_accuracy: 0.7609
 4160/60000 [=>............................] - ETA: 1:42 - loss: 0.7766 - categorical_accuracy: 0.7620
 4192/60000 [=>............................] - ETA: 1:42 - loss: 0.7722 - categorical_accuracy: 0.7634
 4224/60000 [=>............................] - ETA: 1:42 - loss: 0.7707 - categorical_accuracy: 0.7637
 4256/60000 [=>............................] - ETA: 1:42 - loss: 0.7668 - categorical_accuracy: 0.7653
 4288/60000 [=>............................] - ETA: 1:42 - loss: 0.7657 - categorical_accuracy: 0.7656
 4320/60000 [=>............................] - ETA: 1:42 - loss: 0.7639 - categorical_accuracy: 0.7662
 4352/60000 [=>............................] - ETA: 1:42 - loss: 0.7620 - categorical_accuracy: 0.7670
 4384/60000 [=>............................] - ETA: 1:41 - loss: 0.7589 - categorical_accuracy: 0.7678
 4416/60000 [=>............................] - ETA: 1:41 - loss: 0.7552 - categorical_accuracy: 0.7688
 4448/60000 [=>............................] - ETA: 1:41 - loss: 0.7538 - categorical_accuracy: 0.7689
 4480/60000 [=>............................] - ETA: 1:41 - loss: 0.7533 - categorical_accuracy: 0.7692
 4512/60000 [=>............................] - ETA: 1:41 - loss: 0.7523 - categorical_accuracy: 0.7695
 4544/60000 [=>............................] - ETA: 1:41 - loss: 0.7489 - categorical_accuracy: 0.7705
 4576/60000 [=>............................] - ETA: 1:41 - loss: 0.7462 - categorical_accuracy: 0.7714
 4608/60000 [=>............................] - ETA: 1:41 - loss: 0.7436 - categorical_accuracy: 0.7721
 4640/60000 [=>............................] - ETA: 1:41 - loss: 0.7417 - categorical_accuracy: 0.7728
 4672/60000 [=>............................] - ETA: 1:41 - loss: 0.7392 - categorical_accuracy: 0.7735
 4704/60000 [=>............................] - ETA: 1:41 - loss: 0.7364 - categorical_accuracy: 0.7744
 4736/60000 [=>............................] - ETA: 1:41 - loss: 0.7323 - categorical_accuracy: 0.7760
 4768/60000 [=>............................] - ETA: 1:41 - loss: 0.7297 - categorical_accuracy: 0.7764
 4800/60000 [=>............................] - ETA: 1:41 - loss: 0.7281 - categorical_accuracy: 0.7773
 4832/60000 [=>............................] - ETA: 1:41 - loss: 0.7281 - categorical_accuracy: 0.7779
 4864/60000 [=>............................] - ETA: 1:41 - loss: 0.7265 - categorical_accuracy: 0.7786
 4896/60000 [=>............................] - ETA: 1:40 - loss: 0.7233 - categorical_accuracy: 0.7798
 4928/60000 [=>............................] - ETA: 1:40 - loss: 0.7211 - categorical_accuracy: 0.7804
 4960/60000 [=>............................] - ETA: 1:40 - loss: 0.7173 - categorical_accuracy: 0.7817
 4992/60000 [=>............................] - ETA: 1:40 - loss: 0.7150 - categorical_accuracy: 0.7821
 5024/60000 [=>............................] - ETA: 1:40 - loss: 0.7118 - categorical_accuracy: 0.7828
 5056/60000 [=>............................] - ETA: 1:40 - loss: 0.7096 - categorical_accuracy: 0.7836
 5088/60000 [=>............................] - ETA: 1:40 - loss: 0.7075 - categorical_accuracy: 0.7844
 5120/60000 [=>............................] - ETA: 1:40 - loss: 0.7041 - categorical_accuracy: 0.7854
 5152/60000 [=>............................] - ETA: 1:40 - loss: 0.7025 - categorical_accuracy: 0.7855
 5184/60000 [=>............................] - ETA: 1:40 - loss: 0.7002 - categorical_accuracy: 0.7861
 5216/60000 [=>............................] - ETA: 1:39 - loss: 0.6980 - categorical_accuracy: 0.7866
 5248/60000 [=>............................] - ETA: 1:39 - loss: 0.6954 - categorical_accuracy: 0.7873
 5280/60000 [=>............................] - ETA: 1:39 - loss: 0.6946 - categorical_accuracy: 0.7873
 5312/60000 [=>............................] - ETA: 1:39 - loss: 0.6916 - categorical_accuracy: 0.7882
 5344/60000 [=>............................] - ETA: 1:39 - loss: 0.6887 - categorical_accuracy: 0.7889
 5376/60000 [=>............................] - ETA: 1:39 - loss: 0.6876 - categorical_accuracy: 0.7896
 5408/60000 [=>............................] - ETA: 1:39 - loss: 0.6854 - categorical_accuracy: 0.7905
 5440/60000 [=>............................] - ETA: 1:39 - loss: 0.6830 - categorical_accuracy: 0.7912
 5472/60000 [=>............................] - ETA: 1:39 - loss: 0.6805 - categorical_accuracy: 0.7920
 5504/60000 [=>............................] - ETA: 1:39 - loss: 0.6784 - categorical_accuracy: 0.7927
 5536/60000 [=>............................] - ETA: 1:39 - loss: 0.6769 - categorical_accuracy: 0.7932
 5568/60000 [=>............................] - ETA: 1:38 - loss: 0.6737 - categorical_accuracy: 0.7944
 5600/60000 [=>............................] - ETA: 1:38 - loss: 0.6716 - categorical_accuracy: 0.7950
 5632/60000 [=>............................] - ETA: 1:38 - loss: 0.6689 - categorical_accuracy: 0.7960
 5664/60000 [=>............................] - ETA: 1:38 - loss: 0.6672 - categorical_accuracy: 0.7964
 5696/60000 [=>............................] - ETA: 1:38 - loss: 0.6654 - categorical_accuracy: 0.7969
 5728/60000 [=>............................] - ETA: 1:38 - loss: 0.6649 - categorical_accuracy: 0.7975
 5760/60000 [=>............................] - ETA: 1:38 - loss: 0.6641 - categorical_accuracy: 0.7981
 5792/60000 [=>............................] - ETA: 1:38 - loss: 0.6619 - categorical_accuracy: 0.7989
 5824/60000 [=>............................] - ETA: 1:38 - loss: 0.6612 - categorical_accuracy: 0.7993
 5856/60000 [=>............................] - ETA: 1:38 - loss: 0.6593 - categorical_accuracy: 0.8000
 5888/60000 [=>............................] - ETA: 1:38 - loss: 0.6572 - categorical_accuracy: 0.8008
 5920/60000 [=>............................] - ETA: 1:38 - loss: 0.6544 - categorical_accuracy: 0.8017
 5952/60000 [=>............................] - ETA: 1:37 - loss: 0.6522 - categorical_accuracy: 0.8023
 5984/60000 [=>............................] - ETA: 1:37 - loss: 0.6522 - categorical_accuracy: 0.8025
 6016/60000 [==>...........................] - ETA: 1:37 - loss: 0.6507 - categorical_accuracy: 0.8029
 6048/60000 [==>...........................] - ETA: 1:37 - loss: 0.6500 - categorical_accuracy: 0.8031
 6080/60000 [==>...........................] - ETA: 1:37 - loss: 0.6490 - categorical_accuracy: 0.8031
 6112/60000 [==>...........................] - ETA: 1:37 - loss: 0.6476 - categorical_accuracy: 0.8037
 6144/60000 [==>...........................] - ETA: 1:37 - loss: 0.6460 - categorical_accuracy: 0.8039
 6176/60000 [==>...........................] - ETA: 1:37 - loss: 0.6442 - categorical_accuracy: 0.8044
 6208/60000 [==>...........................] - ETA: 1:37 - loss: 0.6423 - categorical_accuracy: 0.8049
 6240/60000 [==>...........................] - ETA: 1:37 - loss: 0.6400 - categorical_accuracy: 0.8056
 6272/60000 [==>...........................] - ETA: 1:37 - loss: 0.6380 - categorical_accuracy: 0.8063
 6304/60000 [==>...........................] - ETA: 1:37 - loss: 0.6359 - categorical_accuracy: 0.8068
 6336/60000 [==>...........................] - ETA: 1:37 - loss: 0.6351 - categorical_accuracy: 0.8070
 6368/60000 [==>...........................] - ETA: 1:37 - loss: 0.6339 - categorical_accuracy: 0.8075
 6400/60000 [==>...........................] - ETA: 1:37 - loss: 0.6327 - categorical_accuracy: 0.8078
 6432/60000 [==>...........................] - ETA: 1:37 - loss: 0.6309 - categorical_accuracy: 0.8083
 6464/60000 [==>...........................] - ETA: 1:37 - loss: 0.6308 - categorical_accuracy: 0.8085
 6496/60000 [==>...........................] - ETA: 1:36 - loss: 0.6294 - categorical_accuracy: 0.8090
 6528/60000 [==>...........................] - ETA: 1:36 - loss: 0.6272 - categorical_accuracy: 0.8096
 6560/60000 [==>...........................] - ETA: 1:36 - loss: 0.6248 - categorical_accuracy: 0.8102
 6592/60000 [==>...........................] - ETA: 1:36 - loss: 0.6223 - categorical_accuracy: 0.8110
 6624/60000 [==>...........................] - ETA: 1:36 - loss: 0.6206 - categorical_accuracy: 0.8113
 6656/60000 [==>...........................] - ETA: 1:36 - loss: 0.6185 - categorical_accuracy: 0.8119
 6688/60000 [==>...........................] - ETA: 1:36 - loss: 0.6171 - categorical_accuracy: 0.8122
 6720/60000 [==>...........................] - ETA: 1:36 - loss: 0.6161 - categorical_accuracy: 0.8126
 6752/60000 [==>...........................] - ETA: 1:36 - loss: 0.6141 - categorical_accuracy: 0.8132
 6784/60000 [==>...........................] - ETA: 1:36 - loss: 0.6120 - categorical_accuracy: 0.8138
 6816/60000 [==>...........................] - ETA: 1:36 - loss: 0.6103 - categorical_accuracy: 0.8144
 6848/60000 [==>...........................] - ETA: 1:36 - loss: 0.6083 - categorical_accuracy: 0.8150
 6880/60000 [==>...........................] - ETA: 1:36 - loss: 0.6063 - categorical_accuracy: 0.8156
 6912/60000 [==>...........................] - ETA: 1:36 - loss: 0.6044 - categorical_accuracy: 0.8160
 6944/60000 [==>...........................] - ETA: 1:36 - loss: 0.6022 - categorical_accuracy: 0.8167
 6976/60000 [==>...........................] - ETA: 1:35 - loss: 0.6011 - categorical_accuracy: 0.8171
 7008/60000 [==>...........................] - ETA: 1:35 - loss: 0.5988 - categorical_accuracy: 0.8178
 7040/60000 [==>...........................] - ETA: 1:35 - loss: 0.5982 - categorical_accuracy: 0.8180
 7072/60000 [==>...........................] - ETA: 1:35 - loss: 0.5959 - categorical_accuracy: 0.8187
 7104/60000 [==>...........................] - ETA: 1:35 - loss: 0.5945 - categorical_accuracy: 0.8190
 7136/60000 [==>...........................] - ETA: 1:35 - loss: 0.5921 - categorical_accuracy: 0.8198
 7168/60000 [==>...........................] - ETA: 1:35 - loss: 0.5907 - categorical_accuracy: 0.8202
 7200/60000 [==>...........................] - ETA: 1:35 - loss: 0.5892 - categorical_accuracy: 0.8203
 7232/60000 [==>...........................] - ETA: 1:35 - loss: 0.5871 - categorical_accuracy: 0.8209
 7264/60000 [==>...........................] - ETA: 1:35 - loss: 0.5867 - categorical_accuracy: 0.8214
 7296/60000 [==>...........................] - ETA: 1:35 - loss: 0.5851 - categorical_accuracy: 0.8218
 7328/60000 [==>...........................] - ETA: 1:35 - loss: 0.5839 - categorical_accuracy: 0.8223
 7360/60000 [==>...........................] - ETA: 1:34 - loss: 0.5823 - categorical_accuracy: 0.8228
 7392/60000 [==>...........................] - ETA: 1:34 - loss: 0.5804 - categorical_accuracy: 0.8235
 7424/60000 [==>...........................] - ETA: 1:34 - loss: 0.5795 - categorical_accuracy: 0.8238
 7456/60000 [==>...........................] - ETA: 1:34 - loss: 0.5776 - categorical_accuracy: 0.8246
 7488/60000 [==>...........................] - ETA: 1:34 - loss: 0.5758 - categorical_accuracy: 0.8249
 7520/60000 [==>...........................] - ETA: 1:34 - loss: 0.5748 - categorical_accuracy: 0.8253
 7552/60000 [==>...........................] - ETA: 1:34 - loss: 0.5733 - categorical_accuracy: 0.8257
 7584/60000 [==>...........................] - ETA: 1:34 - loss: 0.5718 - categorical_accuracy: 0.8262
 7616/60000 [==>...........................] - ETA: 1:34 - loss: 0.5715 - categorical_accuracy: 0.8263
 7648/60000 [==>...........................] - ETA: 1:34 - loss: 0.5703 - categorical_accuracy: 0.8265
 7680/60000 [==>...........................] - ETA: 1:34 - loss: 0.5704 - categorical_accuracy: 0.8267
 7712/60000 [==>...........................] - ETA: 1:34 - loss: 0.5708 - categorical_accuracy: 0.8264
 7744/60000 [==>...........................] - ETA: 1:34 - loss: 0.5694 - categorical_accuracy: 0.8268
 7776/60000 [==>...........................] - ETA: 1:34 - loss: 0.5682 - categorical_accuracy: 0.8273
 7808/60000 [==>...........................] - ETA: 1:33 - loss: 0.5663 - categorical_accuracy: 0.8279
 7840/60000 [==>...........................] - ETA: 1:33 - loss: 0.5642 - categorical_accuracy: 0.8286
 7872/60000 [==>...........................] - ETA: 1:33 - loss: 0.5647 - categorical_accuracy: 0.8288
 7904/60000 [==>...........................] - ETA: 1:33 - loss: 0.5632 - categorical_accuracy: 0.8292
 7936/60000 [==>...........................] - ETA: 1:33 - loss: 0.5613 - categorical_accuracy: 0.8299
 7968/60000 [==>...........................] - ETA: 1:33 - loss: 0.5595 - categorical_accuracy: 0.8304
 8000/60000 [===>..........................] - ETA: 1:33 - loss: 0.5579 - categorical_accuracy: 0.8310
 8032/60000 [===>..........................] - ETA: 1:33 - loss: 0.5563 - categorical_accuracy: 0.8315
 8064/60000 [===>..........................] - ETA: 1:33 - loss: 0.5553 - categorical_accuracy: 0.8318
 8096/60000 [===>..........................] - ETA: 1:33 - loss: 0.5543 - categorical_accuracy: 0.8321
 8128/60000 [===>..........................] - ETA: 1:33 - loss: 0.5528 - categorical_accuracy: 0.8326
 8160/60000 [===>..........................] - ETA: 1:33 - loss: 0.5513 - categorical_accuracy: 0.8330
 8192/60000 [===>..........................] - ETA: 1:33 - loss: 0.5496 - categorical_accuracy: 0.8335
 8224/60000 [===>..........................] - ETA: 1:33 - loss: 0.5486 - categorical_accuracy: 0.8337
 8256/60000 [===>..........................] - ETA: 1:33 - loss: 0.5470 - categorical_accuracy: 0.8342
 8288/60000 [===>..........................] - ETA: 1:32 - loss: 0.5456 - categorical_accuracy: 0.8345
 8320/60000 [===>..........................] - ETA: 1:32 - loss: 0.5451 - categorical_accuracy: 0.8349
 8352/60000 [===>..........................] - ETA: 1:32 - loss: 0.5449 - categorical_accuracy: 0.8351
 8384/60000 [===>..........................] - ETA: 1:32 - loss: 0.5434 - categorical_accuracy: 0.8356
 8416/60000 [===>..........................] - ETA: 1:32 - loss: 0.5422 - categorical_accuracy: 0.8360
 8448/60000 [===>..........................] - ETA: 1:32 - loss: 0.5406 - categorical_accuracy: 0.8364
 8480/60000 [===>..........................] - ETA: 1:32 - loss: 0.5399 - categorical_accuracy: 0.8367
 8512/60000 [===>..........................] - ETA: 1:32 - loss: 0.5395 - categorical_accuracy: 0.8365
 8544/60000 [===>..........................] - ETA: 1:32 - loss: 0.5390 - categorical_accuracy: 0.8365
 8576/60000 [===>..........................] - ETA: 1:32 - loss: 0.5374 - categorical_accuracy: 0.8371
 8608/60000 [===>..........................] - ETA: 1:32 - loss: 0.5363 - categorical_accuracy: 0.8376
 8640/60000 [===>..........................] - ETA: 1:32 - loss: 0.5348 - categorical_accuracy: 0.8380
 8672/60000 [===>..........................] - ETA: 1:32 - loss: 0.5335 - categorical_accuracy: 0.8382
 8704/60000 [===>..........................] - ETA: 1:32 - loss: 0.5325 - categorical_accuracy: 0.8384
 8736/60000 [===>..........................] - ETA: 1:31 - loss: 0.5318 - categorical_accuracy: 0.8385
 8768/60000 [===>..........................] - ETA: 1:31 - loss: 0.5302 - categorical_accuracy: 0.8391
 8800/60000 [===>..........................] - ETA: 1:31 - loss: 0.5293 - categorical_accuracy: 0.8395
 8832/60000 [===>..........................] - ETA: 1:31 - loss: 0.5288 - categorical_accuracy: 0.8398
 8864/60000 [===>..........................] - ETA: 1:31 - loss: 0.5279 - categorical_accuracy: 0.8401
 8896/60000 [===>..........................] - ETA: 1:31 - loss: 0.5270 - categorical_accuracy: 0.8406
 8928/60000 [===>..........................] - ETA: 1:31 - loss: 0.5258 - categorical_accuracy: 0.8411
 8960/60000 [===>..........................] - ETA: 1:31 - loss: 0.5243 - categorical_accuracy: 0.8415
 8992/60000 [===>..........................] - ETA: 1:31 - loss: 0.5234 - categorical_accuracy: 0.8417
 9024/60000 [===>..........................] - ETA: 1:31 - loss: 0.5220 - categorical_accuracy: 0.8420
 9056/60000 [===>..........................] - ETA: 1:31 - loss: 0.5206 - categorical_accuracy: 0.8423
 9088/60000 [===>..........................] - ETA: 1:31 - loss: 0.5197 - categorical_accuracy: 0.8423
 9120/60000 [===>..........................] - ETA: 1:31 - loss: 0.5187 - categorical_accuracy: 0.8427
 9152/60000 [===>..........................] - ETA: 1:31 - loss: 0.5172 - categorical_accuracy: 0.8430
 9184/60000 [===>..........................] - ETA: 1:31 - loss: 0.5163 - categorical_accuracy: 0.8434
 9216/60000 [===>..........................] - ETA: 1:31 - loss: 0.5161 - categorical_accuracy: 0.8435
 9248/60000 [===>..........................] - ETA: 1:31 - loss: 0.5147 - categorical_accuracy: 0.8441
 9280/60000 [===>..........................] - ETA: 1:31 - loss: 0.5135 - categorical_accuracy: 0.8443
 9312/60000 [===>..........................] - ETA: 1:30 - loss: 0.5143 - categorical_accuracy: 0.8442
 9344/60000 [===>..........................] - ETA: 1:30 - loss: 0.5136 - categorical_accuracy: 0.8443
 9376/60000 [===>..........................] - ETA: 1:30 - loss: 0.5134 - categorical_accuracy: 0.8445
 9408/60000 [===>..........................] - ETA: 1:30 - loss: 0.5121 - categorical_accuracy: 0.8448
 9440/60000 [===>..........................] - ETA: 1:30 - loss: 0.5118 - categorical_accuracy: 0.8448
 9472/60000 [===>..........................] - ETA: 1:30 - loss: 0.5118 - categorical_accuracy: 0.8448
 9504/60000 [===>..........................] - ETA: 1:30 - loss: 0.5109 - categorical_accuracy: 0.8450
 9536/60000 [===>..........................] - ETA: 1:30 - loss: 0.5103 - categorical_accuracy: 0.8452
 9568/60000 [===>..........................] - ETA: 1:30 - loss: 0.5095 - categorical_accuracy: 0.8455
 9600/60000 [===>..........................] - ETA: 1:30 - loss: 0.5085 - categorical_accuracy: 0.8457
 9632/60000 [===>..........................] - ETA: 1:30 - loss: 0.5081 - categorical_accuracy: 0.8460
 9664/60000 [===>..........................] - ETA: 1:30 - loss: 0.5080 - categorical_accuracy: 0.8462
 9696/60000 [===>..........................] - ETA: 1:30 - loss: 0.5071 - categorical_accuracy: 0.8464
 9728/60000 [===>..........................] - ETA: 1:30 - loss: 0.5063 - categorical_accuracy: 0.8467
 9760/60000 [===>..........................] - ETA: 1:30 - loss: 0.5053 - categorical_accuracy: 0.8470
 9792/60000 [===>..........................] - ETA: 1:30 - loss: 0.5044 - categorical_accuracy: 0.8473
 9824/60000 [===>..........................] - ETA: 1:30 - loss: 0.5029 - categorical_accuracy: 0.8478
 9856/60000 [===>..........................] - ETA: 1:30 - loss: 0.5016 - categorical_accuracy: 0.8483
 9888/60000 [===>..........................] - ETA: 1:29 - loss: 0.5002 - categorical_accuracy: 0.8488
 9920/60000 [===>..........................] - ETA: 1:29 - loss: 0.4993 - categorical_accuracy: 0.8491
 9952/60000 [===>..........................] - ETA: 1:29 - loss: 0.4980 - categorical_accuracy: 0.8495
 9984/60000 [===>..........................] - ETA: 1:29 - loss: 0.4973 - categorical_accuracy: 0.8498
10016/60000 [====>.........................] - ETA: 1:29 - loss: 0.4963 - categorical_accuracy: 0.8501
10048/60000 [====>.........................] - ETA: 1:29 - loss: 0.4950 - categorical_accuracy: 0.8505
10080/60000 [====>.........................] - ETA: 1:29 - loss: 0.4936 - categorical_accuracy: 0.8509
10112/60000 [====>.........................] - ETA: 1:29 - loss: 0.4922 - categorical_accuracy: 0.8513
10144/60000 [====>.........................] - ETA: 1:29 - loss: 0.4913 - categorical_accuracy: 0.8516
10176/60000 [====>.........................] - ETA: 1:29 - loss: 0.4900 - categorical_accuracy: 0.8520
10208/60000 [====>.........................] - ETA: 1:29 - loss: 0.4901 - categorical_accuracy: 0.8522
10240/60000 [====>.........................] - ETA: 1:29 - loss: 0.4888 - categorical_accuracy: 0.8525
10272/60000 [====>.........................] - ETA: 1:29 - loss: 0.4882 - categorical_accuracy: 0.8526
10304/60000 [====>.........................] - ETA: 1:29 - loss: 0.4874 - categorical_accuracy: 0.8528
10336/60000 [====>.........................] - ETA: 1:29 - loss: 0.4865 - categorical_accuracy: 0.8531
10368/60000 [====>.........................] - ETA: 1:29 - loss: 0.4857 - categorical_accuracy: 0.8533
10400/60000 [====>.........................] - ETA: 1:29 - loss: 0.4845 - categorical_accuracy: 0.8537
10432/60000 [====>.........................] - ETA: 1:29 - loss: 0.4835 - categorical_accuracy: 0.8539
10464/60000 [====>.........................] - ETA: 1:28 - loss: 0.4826 - categorical_accuracy: 0.8543
10496/60000 [====>.........................] - ETA: 1:28 - loss: 0.4818 - categorical_accuracy: 0.8544
10528/60000 [====>.........................] - ETA: 1:28 - loss: 0.4813 - categorical_accuracy: 0.8546
10560/60000 [====>.........................] - ETA: 1:28 - loss: 0.4805 - categorical_accuracy: 0.8548
10592/60000 [====>.........................] - ETA: 1:28 - loss: 0.4794 - categorical_accuracy: 0.8551
10624/60000 [====>.........................] - ETA: 1:28 - loss: 0.4782 - categorical_accuracy: 0.8554
10656/60000 [====>.........................] - ETA: 1:28 - loss: 0.4774 - categorical_accuracy: 0.8556
10688/60000 [====>.........................] - ETA: 1:28 - loss: 0.4765 - categorical_accuracy: 0.8558
10720/60000 [====>.........................] - ETA: 1:28 - loss: 0.4752 - categorical_accuracy: 0.8562
10752/60000 [====>.........................] - ETA: 1:28 - loss: 0.4746 - categorical_accuracy: 0.8563
10784/60000 [====>.........................] - ETA: 1:28 - loss: 0.4735 - categorical_accuracy: 0.8566
10816/60000 [====>.........................] - ETA: 1:28 - loss: 0.4736 - categorical_accuracy: 0.8563
10848/60000 [====>.........................] - ETA: 1:28 - loss: 0.4737 - categorical_accuracy: 0.8561
10880/60000 [====>.........................] - ETA: 1:28 - loss: 0.4725 - categorical_accuracy: 0.8564
10912/60000 [====>.........................] - ETA: 1:28 - loss: 0.4713 - categorical_accuracy: 0.8568
10944/60000 [====>.........................] - ETA: 1:28 - loss: 0.4707 - categorical_accuracy: 0.8570
10976/60000 [====>.........................] - ETA: 1:28 - loss: 0.4695 - categorical_accuracy: 0.8573
11008/60000 [====>.........................] - ETA: 1:28 - loss: 0.4686 - categorical_accuracy: 0.8576
11040/60000 [====>.........................] - ETA: 1:27 - loss: 0.4677 - categorical_accuracy: 0.8578
11072/60000 [====>.........................] - ETA: 1:27 - loss: 0.4665 - categorical_accuracy: 0.8582
11104/60000 [====>.........................] - ETA: 1:27 - loss: 0.4661 - categorical_accuracy: 0.8583
11136/60000 [====>.........................] - ETA: 1:27 - loss: 0.4652 - categorical_accuracy: 0.8587
11168/60000 [====>.........................] - ETA: 1:27 - loss: 0.4642 - categorical_accuracy: 0.8590
11200/60000 [====>.........................] - ETA: 1:27 - loss: 0.4630 - categorical_accuracy: 0.8594
11232/60000 [====>.........................] - ETA: 1:27 - loss: 0.4632 - categorical_accuracy: 0.8594
11264/60000 [====>.........................] - ETA: 1:27 - loss: 0.4625 - categorical_accuracy: 0.8597
11296/60000 [====>.........................] - ETA: 1:27 - loss: 0.4616 - categorical_accuracy: 0.8600
11328/60000 [====>.........................] - ETA: 1:27 - loss: 0.4611 - categorical_accuracy: 0.8603
11360/60000 [====>.........................] - ETA: 1:27 - loss: 0.4601 - categorical_accuracy: 0.8606
11392/60000 [====>.........................] - ETA: 1:27 - loss: 0.4592 - categorical_accuracy: 0.8608
11424/60000 [====>.........................] - ETA: 1:27 - loss: 0.4584 - categorical_accuracy: 0.8609
11456/60000 [====>.........................] - ETA: 1:27 - loss: 0.4575 - categorical_accuracy: 0.8612
11488/60000 [====>.........................] - ETA: 1:27 - loss: 0.4563 - categorical_accuracy: 0.8616
11520/60000 [====>.........................] - ETA: 1:27 - loss: 0.4556 - categorical_accuracy: 0.8618
11552/60000 [====>.........................] - ETA: 1:27 - loss: 0.4546 - categorical_accuracy: 0.8622
11584/60000 [====>.........................] - ETA: 1:27 - loss: 0.4538 - categorical_accuracy: 0.8623
11616/60000 [====>.........................] - ETA: 1:27 - loss: 0.4527 - categorical_accuracy: 0.8627
11648/60000 [====>.........................] - ETA: 1:27 - loss: 0.4517 - categorical_accuracy: 0.8630
11680/60000 [====>.........................] - ETA: 1:26 - loss: 0.4506 - categorical_accuracy: 0.8634
11712/60000 [====>.........................] - ETA: 1:26 - loss: 0.4496 - categorical_accuracy: 0.8636
11744/60000 [====>.........................] - ETA: 1:26 - loss: 0.4485 - categorical_accuracy: 0.8639
11776/60000 [====>.........................] - ETA: 1:26 - loss: 0.4477 - categorical_accuracy: 0.8642
11808/60000 [====>.........................] - ETA: 1:26 - loss: 0.4474 - categorical_accuracy: 0.8644
11840/60000 [====>.........................] - ETA: 1:26 - loss: 0.4468 - categorical_accuracy: 0.8644
11872/60000 [====>.........................] - ETA: 1:26 - loss: 0.4469 - categorical_accuracy: 0.8646
11904/60000 [====>.........................] - ETA: 1:26 - loss: 0.4460 - categorical_accuracy: 0.8649
11936/60000 [====>.........................] - ETA: 1:26 - loss: 0.4457 - categorical_accuracy: 0.8650
11968/60000 [====>.........................] - ETA: 1:26 - loss: 0.4463 - categorical_accuracy: 0.8649
12000/60000 [=====>........................] - ETA: 1:26 - loss: 0.4453 - categorical_accuracy: 0.8652
12032/60000 [=====>........................] - ETA: 1:26 - loss: 0.4445 - categorical_accuracy: 0.8655
12064/60000 [=====>........................] - ETA: 1:26 - loss: 0.4441 - categorical_accuracy: 0.8656
12096/60000 [=====>........................] - ETA: 1:26 - loss: 0.4431 - categorical_accuracy: 0.8659
12128/60000 [=====>........................] - ETA: 1:26 - loss: 0.4425 - categorical_accuracy: 0.8662
12160/60000 [=====>........................] - ETA: 1:26 - loss: 0.4418 - categorical_accuracy: 0.8664
12192/60000 [=====>........................] - ETA: 1:26 - loss: 0.4409 - categorical_accuracy: 0.8666
12224/60000 [=====>........................] - ETA: 1:26 - loss: 0.4399 - categorical_accuracy: 0.8670
12256/60000 [=====>........................] - ETA: 1:25 - loss: 0.4396 - categorical_accuracy: 0.8670
12288/60000 [=====>........................] - ETA: 1:25 - loss: 0.4391 - categorical_accuracy: 0.8670
12320/60000 [=====>........................] - ETA: 1:25 - loss: 0.4381 - categorical_accuracy: 0.8674
12352/60000 [=====>........................] - ETA: 1:25 - loss: 0.4372 - categorical_accuracy: 0.8676
12384/60000 [=====>........................] - ETA: 1:25 - loss: 0.4367 - categorical_accuracy: 0.8678
12416/60000 [=====>........................] - ETA: 1:25 - loss: 0.4362 - categorical_accuracy: 0.8679
12448/60000 [=====>........................] - ETA: 1:25 - loss: 0.4354 - categorical_accuracy: 0.8682
12480/60000 [=====>........................] - ETA: 1:25 - loss: 0.4343 - categorical_accuracy: 0.8685
12512/60000 [=====>........................] - ETA: 1:25 - loss: 0.4340 - categorical_accuracy: 0.8686
12544/60000 [=====>........................] - ETA: 1:25 - loss: 0.4335 - categorical_accuracy: 0.8688
12576/60000 [=====>........................] - ETA: 1:25 - loss: 0.4328 - categorical_accuracy: 0.8690
12608/60000 [=====>........................] - ETA: 1:25 - loss: 0.4324 - categorical_accuracy: 0.8691
12640/60000 [=====>........................] - ETA: 1:25 - loss: 0.4313 - categorical_accuracy: 0.8695
12672/60000 [=====>........................] - ETA: 1:25 - loss: 0.4305 - categorical_accuracy: 0.8697
12704/60000 [=====>........................] - ETA: 1:25 - loss: 0.4301 - categorical_accuracy: 0.8699
12736/60000 [=====>........................] - ETA: 1:24 - loss: 0.4300 - categorical_accuracy: 0.8701
12768/60000 [=====>........................] - ETA: 1:24 - loss: 0.4290 - categorical_accuracy: 0.8705
12800/60000 [=====>........................] - ETA: 1:24 - loss: 0.4283 - categorical_accuracy: 0.8707
12832/60000 [=====>........................] - ETA: 1:24 - loss: 0.4274 - categorical_accuracy: 0.8710
12864/60000 [=====>........................] - ETA: 1:24 - loss: 0.4265 - categorical_accuracy: 0.8713
12896/60000 [=====>........................] - ETA: 1:24 - loss: 0.4261 - categorical_accuracy: 0.8714
12928/60000 [=====>........................] - ETA: 1:24 - loss: 0.4257 - categorical_accuracy: 0.8714
12960/60000 [=====>........................] - ETA: 1:24 - loss: 0.4250 - categorical_accuracy: 0.8717
12992/60000 [=====>........................] - ETA: 1:24 - loss: 0.4241 - categorical_accuracy: 0.8720
13024/60000 [=====>........................] - ETA: 1:24 - loss: 0.4233 - categorical_accuracy: 0.8722
13056/60000 [=====>........................] - ETA: 1:24 - loss: 0.4224 - categorical_accuracy: 0.8725
13088/60000 [=====>........................] - ETA: 1:24 - loss: 0.4220 - categorical_accuracy: 0.8727
13120/60000 [=====>........................] - ETA: 1:24 - loss: 0.4212 - categorical_accuracy: 0.8729
13152/60000 [=====>........................] - ETA: 1:24 - loss: 0.4206 - categorical_accuracy: 0.8731
13184/60000 [=====>........................] - ETA: 1:24 - loss: 0.4204 - categorical_accuracy: 0.8733
13216/60000 [=====>........................] - ETA: 1:24 - loss: 0.4204 - categorical_accuracy: 0.8733
13248/60000 [=====>........................] - ETA: 1:24 - loss: 0.4196 - categorical_accuracy: 0.8735
13280/60000 [=====>........................] - ETA: 1:24 - loss: 0.4188 - categorical_accuracy: 0.8737
13312/60000 [=====>........................] - ETA: 1:23 - loss: 0.4184 - categorical_accuracy: 0.8737
13344/60000 [=====>........................] - ETA: 1:23 - loss: 0.4182 - categorical_accuracy: 0.8739
13376/60000 [=====>........................] - ETA: 1:23 - loss: 0.4173 - categorical_accuracy: 0.8742
13408/60000 [=====>........................] - ETA: 1:23 - loss: 0.4169 - categorical_accuracy: 0.8743
13440/60000 [=====>........................] - ETA: 1:23 - loss: 0.4161 - categorical_accuracy: 0.8745
13472/60000 [=====>........................] - ETA: 1:23 - loss: 0.4165 - categorical_accuracy: 0.8746
13504/60000 [=====>........................] - ETA: 1:23 - loss: 0.4159 - categorical_accuracy: 0.8748
13536/60000 [=====>........................] - ETA: 1:23 - loss: 0.4152 - categorical_accuracy: 0.8749
13568/60000 [=====>........................] - ETA: 1:23 - loss: 0.4144 - categorical_accuracy: 0.8751
13600/60000 [=====>........................] - ETA: 1:23 - loss: 0.4136 - categorical_accuracy: 0.8754
13632/60000 [=====>........................] - ETA: 1:23 - loss: 0.4127 - categorical_accuracy: 0.8757
13664/60000 [=====>........................] - ETA: 1:23 - loss: 0.4119 - categorical_accuracy: 0.8759
13696/60000 [=====>........................] - ETA: 1:23 - loss: 0.4111 - categorical_accuracy: 0.8761
13728/60000 [=====>........................] - ETA: 1:23 - loss: 0.4104 - categorical_accuracy: 0.8763
13760/60000 [=====>........................] - ETA: 1:23 - loss: 0.4099 - categorical_accuracy: 0.8765
13792/60000 [=====>........................] - ETA: 1:22 - loss: 0.4109 - categorical_accuracy: 0.8765
13824/60000 [=====>........................] - ETA: 1:22 - loss: 0.4104 - categorical_accuracy: 0.8765
13856/60000 [=====>........................] - ETA: 1:22 - loss: 0.4098 - categorical_accuracy: 0.8767
13888/60000 [=====>........................] - ETA: 1:22 - loss: 0.4090 - categorical_accuracy: 0.8769
13920/60000 [=====>........................] - ETA: 1:22 - loss: 0.4086 - categorical_accuracy: 0.8771
13952/60000 [=====>........................] - ETA: 1:22 - loss: 0.4079 - categorical_accuracy: 0.8773
13984/60000 [=====>........................] - ETA: 1:22 - loss: 0.4070 - categorical_accuracy: 0.8776
14016/60000 [======>.......................] - ETA: 1:22 - loss: 0.4066 - categorical_accuracy: 0.8777
14048/60000 [======>.......................] - ETA: 1:22 - loss: 0.4059 - categorical_accuracy: 0.8778
14080/60000 [======>.......................] - ETA: 1:22 - loss: 0.4054 - categorical_accuracy: 0.8781
14112/60000 [======>.......................] - ETA: 1:22 - loss: 0.4049 - categorical_accuracy: 0.8781
14144/60000 [======>.......................] - ETA: 1:22 - loss: 0.4042 - categorical_accuracy: 0.8783
14176/60000 [======>.......................] - ETA: 1:22 - loss: 0.4038 - categorical_accuracy: 0.8784
14208/60000 [======>.......................] - ETA: 1:22 - loss: 0.4031 - categorical_accuracy: 0.8786
14240/60000 [======>.......................] - ETA: 1:22 - loss: 0.4027 - categorical_accuracy: 0.8787
14272/60000 [======>.......................] - ETA: 1:21 - loss: 0.4021 - categorical_accuracy: 0.8789
14304/60000 [======>.......................] - ETA: 1:21 - loss: 0.4014 - categorical_accuracy: 0.8791
14336/60000 [======>.......................] - ETA: 1:21 - loss: 0.4008 - categorical_accuracy: 0.8793
14368/60000 [======>.......................] - ETA: 1:21 - loss: 0.4002 - categorical_accuracy: 0.8795
14400/60000 [======>.......................] - ETA: 1:21 - loss: 0.4000 - categorical_accuracy: 0.8795
14432/60000 [======>.......................] - ETA: 1:21 - loss: 0.3992 - categorical_accuracy: 0.8797
14464/60000 [======>.......................] - ETA: 1:21 - loss: 0.3989 - categorical_accuracy: 0.8798
14496/60000 [======>.......................] - ETA: 1:21 - loss: 0.3983 - categorical_accuracy: 0.8800
14528/60000 [======>.......................] - ETA: 1:21 - loss: 0.3979 - categorical_accuracy: 0.8800
14560/60000 [======>.......................] - ETA: 1:21 - loss: 0.3971 - categorical_accuracy: 0.8803
14592/60000 [======>.......................] - ETA: 1:21 - loss: 0.3964 - categorical_accuracy: 0.8805
14624/60000 [======>.......................] - ETA: 1:21 - loss: 0.3958 - categorical_accuracy: 0.8807
14656/60000 [======>.......................] - ETA: 1:21 - loss: 0.3953 - categorical_accuracy: 0.8809
14688/60000 [======>.......................] - ETA: 1:21 - loss: 0.3946 - categorical_accuracy: 0.8811
14720/60000 [======>.......................] - ETA: 1:21 - loss: 0.3941 - categorical_accuracy: 0.8813
14752/60000 [======>.......................] - ETA: 1:20 - loss: 0.3936 - categorical_accuracy: 0.8814
14784/60000 [======>.......................] - ETA: 1:20 - loss: 0.3935 - categorical_accuracy: 0.8815
14816/60000 [======>.......................] - ETA: 1:20 - loss: 0.3927 - categorical_accuracy: 0.8817
14848/60000 [======>.......................] - ETA: 1:20 - loss: 0.3919 - categorical_accuracy: 0.8820
14880/60000 [======>.......................] - ETA: 1:20 - loss: 0.3913 - categorical_accuracy: 0.8821
14912/60000 [======>.......................] - ETA: 1:20 - loss: 0.3907 - categorical_accuracy: 0.8823
14944/60000 [======>.......................] - ETA: 1:20 - loss: 0.3900 - categorical_accuracy: 0.8825
14976/60000 [======>.......................] - ETA: 1:20 - loss: 0.3894 - categorical_accuracy: 0.8827
15008/60000 [======>.......................] - ETA: 1:20 - loss: 0.3886 - categorical_accuracy: 0.8829
15040/60000 [======>.......................] - ETA: 1:20 - loss: 0.3880 - categorical_accuracy: 0.8831
15072/60000 [======>.......................] - ETA: 1:20 - loss: 0.3877 - categorical_accuracy: 0.8833
15104/60000 [======>.......................] - ETA: 1:20 - loss: 0.3877 - categorical_accuracy: 0.8833
15136/60000 [======>.......................] - ETA: 1:20 - loss: 0.3873 - categorical_accuracy: 0.8834
15168/60000 [======>.......................] - ETA: 1:20 - loss: 0.3876 - categorical_accuracy: 0.8834
15200/60000 [======>.......................] - ETA: 1:20 - loss: 0.3868 - categorical_accuracy: 0.8836
15232/60000 [======>.......................] - ETA: 1:20 - loss: 0.3867 - categorical_accuracy: 0.8836
15264/60000 [======>.......................] - ETA: 1:20 - loss: 0.3863 - categorical_accuracy: 0.8838
15296/60000 [======>.......................] - ETA: 1:20 - loss: 0.3862 - categorical_accuracy: 0.8838
15328/60000 [======>.......................] - ETA: 1:19 - loss: 0.3856 - categorical_accuracy: 0.8841
15360/60000 [======>.......................] - ETA: 1:19 - loss: 0.3856 - categorical_accuracy: 0.8840
15392/60000 [======>.......................] - ETA: 1:19 - loss: 0.3848 - categorical_accuracy: 0.8843
15424/60000 [======>.......................] - ETA: 1:19 - loss: 0.3847 - categorical_accuracy: 0.8843
15456/60000 [======>.......................] - ETA: 1:19 - loss: 0.3840 - categorical_accuracy: 0.8846
15488/60000 [======>.......................] - ETA: 1:19 - loss: 0.3835 - categorical_accuracy: 0.8847
15520/60000 [======>.......................] - ETA: 1:19 - loss: 0.3828 - categorical_accuracy: 0.8849
15552/60000 [======>.......................] - ETA: 1:19 - loss: 0.3822 - categorical_accuracy: 0.8850
15584/60000 [======>.......................] - ETA: 1:19 - loss: 0.3815 - categorical_accuracy: 0.8853
15616/60000 [======>.......................] - ETA: 1:19 - loss: 0.3811 - categorical_accuracy: 0.8854
15648/60000 [======>.......................] - ETA: 1:19 - loss: 0.3806 - categorical_accuracy: 0.8856
15680/60000 [======>.......................] - ETA: 1:19 - loss: 0.3803 - categorical_accuracy: 0.8857
15712/60000 [======>.......................] - ETA: 1:19 - loss: 0.3796 - categorical_accuracy: 0.8859
15744/60000 [======>.......................] - ETA: 1:19 - loss: 0.3793 - categorical_accuracy: 0.8859
15776/60000 [======>.......................] - ETA: 1:19 - loss: 0.3786 - categorical_accuracy: 0.8862
15808/60000 [======>.......................] - ETA: 1:19 - loss: 0.3782 - categorical_accuracy: 0.8862
15840/60000 [======>.......................] - ETA: 1:18 - loss: 0.3776 - categorical_accuracy: 0.8864
15872/60000 [======>.......................] - ETA: 1:18 - loss: 0.3769 - categorical_accuracy: 0.8867
15904/60000 [======>.......................] - ETA: 1:18 - loss: 0.3766 - categorical_accuracy: 0.8868
15936/60000 [======>.......................] - ETA: 1:18 - loss: 0.3760 - categorical_accuracy: 0.8869
15968/60000 [======>.......................] - ETA: 1:18 - loss: 0.3753 - categorical_accuracy: 0.8871
16000/60000 [=======>......................] - ETA: 1:18 - loss: 0.3749 - categorical_accuracy: 0.8873
16032/60000 [=======>......................] - ETA: 1:18 - loss: 0.3742 - categorical_accuracy: 0.8875
16064/60000 [=======>......................] - ETA: 1:18 - loss: 0.3735 - categorical_accuracy: 0.8877
16096/60000 [=======>......................] - ETA: 1:18 - loss: 0.3732 - categorical_accuracy: 0.8878
16128/60000 [=======>......................] - ETA: 1:18 - loss: 0.3728 - categorical_accuracy: 0.8879
16160/60000 [=======>......................] - ETA: 1:18 - loss: 0.3722 - categorical_accuracy: 0.8881
16192/60000 [=======>......................] - ETA: 1:18 - loss: 0.3718 - categorical_accuracy: 0.8881
16224/60000 [=======>......................] - ETA: 1:18 - loss: 0.3716 - categorical_accuracy: 0.8882
16256/60000 [=======>......................] - ETA: 1:18 - loss: 0.3709 - categorical_accuracy: 0.8884
16288/60000 [=======>......................] - ETA: 1:18 - loss: 0.3705 - categorical_accuracy: 0.8884
16320/60000 [=======>......................] - ETA: 1:18 - loss: 0.3699 - categorical_accuracy: 0.8887
16352/60000 [=======>......................] - ETA: 1:18 - loss: 0.3692 - categorical_accuracy: 0.8889
16384/60000 [=======>......................] - ETA: 1:18 - loss: 0.3692 - categorical_accuracy: 0.8889
16416/60000 [=======>......................] - ETA: 1:17 - loss: 0.3688 - categorical_accuracy: 0.8890
16448/60000 [=======>......................] - ETA: 1:17 - loss: 0.3684 - categorical_accuracy: 0.8892
16480/60000 [=======>......................] - ETA: 1:17 - loss: 0.3682 - categorical_accuracy: 0.8893
16512/60000 [=======>......................] - ETA: 1:17 - loss: 0.3677 - categorical_accuracy: 0.8894
16544/60000 [=======>......................] - ETA: 1:17 - loss: 0.3671 - categorical_accuracy: 0.8896
16576/60000 [=======>......................] - ETA: 1:17 - loss: 0.3665 - categorical_accuracy: 0.8898
16608/60000 [=======>......................] - ETA: 1:17 - loss: 0.3660 - categorical_accuracy: 0.8899
16640/60000 [=======>......................] - ETA: 1:17 - loss: 0.3661 - categorical_accuracy: 0.8899
16672/60000 [=======>......................] - ETA: 1:17 - loss: 0.3657 - categorical_accuracy: 0.8900
16704/60000 [=======>......................] - ETA: 1:17 - loss: 0.3653 - categorical_accuracy: 0.8901
16736/60000 [=======>......................] - ETA: 1:17 - loss: 0.3653 - categorical_accuracy: 0.8901
16768/60000 [=======>......................] - ETA: 1:17 - loss: 0.3648 - categorical_accuracy: 0.8902
16800/60000 [=======>......................] - ETA: 1:17 - loss: 0.3645 - categorical_accuracy: 0.8903
16832/60000 [=======>......................] - ETA: 1:17 - loss: 0.3641 - categorical_accuracy: 0.8904
16864/60000 [=======>......................] - ETA: 1:17 - loss: 0.3636 - categorical_accuracy: 0.8905
16896/60000 [=======>......................] - ETA: 1:17 - loss: 0.3631 - categorical_accuracy: 0.8907
16928/60000 [=======>......................] - ETA: 1:17 - loss: 0.3625 - categorical_accuracy: 0.8909
16960/60000 [=======>......................] - ETA: 1:17 - loss: 0.3621 - categorical_accuracy: 0.8909
16992/60000 [=======>......................] - ETA: 1:16 - loss: 0.3616 - categorical_accuracy: 0.8911
17024/60000 [=======>......................] - ETA: 1:16 - loss: 0.3613 - categorical_accuracy: 0.8912
17056/60000 [=======>......................] - ETA: 1:16 - loss: 0.3611 - categorical_accuracy: 0.8913
17088/60000 [=======>......................] - ETA: 1:16 - loss: 0.3604 - categorical_accuracy: 0.8915
17120/60000 [=======>......................] - ETA: 1:16 - loss: 0.3600 - categorical_accuracy: 0.8916
17152/60000 [=======>......................] - ETA: 1:16 - loss: 0.3595 - categorical_accuracy: 0.8917
17184/60000 [=======>......................] - ETA: 1:16 - loss: 0.3591 - categorical_accuracy: 0.8918
17216/60000 [=======>......................] - ETA: 1:16 - loss: 0.3590 - categorical_accuracy: 0.8919
17248/60000 [=======>......................] - ETA: 1:16 - loss: 0.3592 - categorical_accuracy: 0.8919
17280/60000 [=======>......................] - ETA: 1:16 - loss: 0.3589 - categorical_accuracy: 0.8920
17312/60000 [=======>......................] - ETA: 1:16 - loss: 0.3588 - categorical_accuracy: 0.8920
17344/60000 [=======>......................] - ETA: 1:16 - loss: 0.3587 - categorical_accuracy: 0.8921
17376/60000 [=======>......................] - ETA: 1:16 - loss: 0.3582 - categorical_accuracy: 0.8923
17408/60000 [=======>......................] - ETA: 1:16 - loss: 0.3580 - categorical_accuracy: 0.8923
17440/60000 [=======>......................] - ETA: 1:16 - loss: 0.3580 - categorical_accuracy: 0.8923
17472/60000 [=======>......................] - ETA: 1:15 - loss: 0.3575 - categorical_accuracy: 0.8925
17504/60000 [=======>......................] - ETA: 1:15 - loss: 0.3570 - categorical_accuracy: 0.8927
17536/60000 [=======>......................] - ETA: 1:15 - loss: 0.3567 - categorical_accuracy: 0.8928
17568/60000 [=======>......................] - ETA: 1:15 - loss: 0.3561 - categorical_accuracy: 0.8930
17600/60000 [=======>......................] - ETA: 1:15 - loss: 0.3559 - categorical_accuracy: 0.8930
17632/60000 [=======>......................] - ETA: 1:15 - loss: 0.3555 - categorical_accuracy: 0.8931
17664/60000 [=======>......................] - ETA: 1:15 - loss: 0.3553 - categorical_accuracy: 0.8932
17696/60000 [=======>......................] - ETA: 1:15 - loss: 0.3554 - categorical_accuracy: 0.8933
17728/60000 [=======>......................] - ETA: 1:15 - loss: 0.3550 - categorical_accuracy: 0.8934
17760/60000 [=======>......................] - ETA: 1:15 - loss: 0.3547 - categorical_accuracy: 0.8935
17792/60000 [=======>......................] - ETA: 1:15 - loss: 0.3542 - categorical_accuracy: 0.8935
17824/60000 [=======>......................] - ETA: 1:15 - loss: 0.3540 - categorical_accuracy: 0.8936
17856/60000 [=======>......................] - ETA: 1:15 - loss: 0.3536 - categorical_accuracy: 0.8938
17888/60000 [=======>......................] - ETA: 1:15 - loss: 0.3536 - categorical_accuracy: 0.8937
17920/60000 [=======>......................] - ETA: 1:15 - loss: 0.3531 - categorical_accuracy: 0.8938
17952/60000 [=======>......................] - ETA: 1:15 - loss: 0.3530 - categorical_accuracy: 0.8938
17984/60000 [=======>......................] - ETA: 1:14 - loss: 0.3527 - categorical_accuracy: 0.8938
18016/60000 [========>.....................] - ETA: 1:14 - loss: 0.3527 - categorical_accuracy: 0.8938
18048/60000 [========>.....................] - ETA: 1:14 - loss: 0.3521 - categorical_accuracy: 0.8940
18080/60000 [========>.....................] - ETA: 1:14 - loss: 0.3516 - categorical_accuracy: 0.8941
18112/60000 [========>.....................] - ETA: 1:14 - loss: 0.3513 - categorical_accuracy: 0.8943
18144/60000 [========>.....................] - ETA: 1:14 - loss: 0.3508 - categorical_accuracy: 0.8945
18176/60000 [========>.....................] - ETA: 1:14 - loss: 0.3510 - categorical_accuracy: 0.8945
18208/60000 [========>.....................] - ETA: 1:14 - loss: 0.3507 - categorical_accuracy: 0.8947
18240/60000 [========>.....................] - ETA: 1:14 - loss: 0.3505 - categorical_accuracy: 0.8947
18272/60000 [========>.....................] - ETA: 1:14 - loss: 0.3501 - categorical_accuracy: 0.8948
18304/60000 [========>.....................] - ETA: 1:14 - loss: 0.3498 - categorical_accuracy: 0.8949
18336/60000 [========>.....................] - ETA: 1:14 - loss: 0.3498 - categorical_accuracy: 0.8949
18368/60000 [========>.....................] - ETA: 1:14 - loss: 0.3493 - categorical_accuracy: 0.8951
18400/60000 [========>.....................] - ETA: 1:14 - loss: 0.3492 - categorical_accuracy: 0.8951
18432/60000 [========>.....................] - ETA: 1:14 - loss: 0.3488 - categorical_accuracy: 0.8953
18464/60000 [========>.....................] - ETA: 1:14 - loss: 0.3485 - categorical_accuracy: 0.8954
18496/60000 [========>.....................] - ETA: 1:14 - loss: 0.3480 - categorical_accuracy: 0.8955
18528/60000 [========>.....................] - ETA: 1:14 - loss: 0.3476 - categorical_accuracy: 0.8956
18560/60000 [========>.....................] - ETA: 1:13 - loss: 0.3471 - categorical_accuracy: 0.8957
18592/60000 [========>.....................] - ETA: 1:13 - loss: 0.3466 - categorical_accuracy: 0.8959
18624/60000 [========>.....................] - ETA: 1:13 - loss: 0.3465 - categorical_accuracy: 0.8959
18656/60000 [========>.....................] - ETA: 1:13 - loss: 0.3461 - categorical_accuracy: 0.8960
18688/60000 [========>.....................] - ETA: 1:13 - loss: 0.3459 - categorical_accuracy: 0.8960
18720/60000 [========>.....................] - ETA: 1:13 - loss: 0.3456 - categorical_accuracy: 0.8962
18752/60000 [========>.....................] - ETA: 1:13 - loss: 0.3451 - categorical_accuracy: 0.8963
18784/60000 [========>.....................] - ETA: 1:13 - loss: 0.3448 - categorical_accuracy: 0.8964
18816/60000 [========>.....................] - ETA: 1:13 - loss: 0.3444 - categorical_accuracy: 0.8965
18848/60000 [========>.....................] - ETA: 1:13 - loss: 0.3440 - categorical_accuracy: 0.8965
18880/60000 [========>.....................] - ETA: 1:13 - loss: 0.3439 - categorical_accuracy: 0.8967
18912/60000 [========>.....................] - ETA: 1:13 - loss: 0.3436 - categorical_accuracy: 0.8968
18944/60000 [========>.....................] - ETA: 1:13 - loss: 0.3433 - categorical_accuracy: 0.8969
18976/60000 [========>.....................] - ETA: 1:13 - loss: 0.3428 - categorical_accuracy: 0.8970
19008/60000 [========>.....................] - ETA: 1:13 - loss: 0.3427 - categorical_accuracy: 0.8970
19040/60000 [========>.....................] - ETA: 1:13 - loss: 0.3427 - categorical_accuracy: 0.8971
19072/60000 [========>.....................] - ETA: 1:13 - loss: 0.3423 - categorical_accuracy: 0.8972
19104/60000 [========>.....................] - ETA: 1:13 - loss: 0.3419 - categorical_accuracy: 0.8974
19136/60000 [========>.....................] - ETA: 1:12 - loss: 0.3415 - categorical_accuracy: 0.8974
19168/60000 [========>.....................] - ETA: 1:12 - loss: 0.3416 - categorical_accuracy: 0.8973
19200/60000 [========>.....................] - ETA: 1:12 - loss: 0.3414 - categorical_accuracy: 0.8974
19232/60000 [========>.....................] - ETA: 1:12 - loss: 0.3409 - categorical_accuracy: 0.8976
19264/60000 [========>.....................] - ETA: 1:12 - loss: 0.3405 - categorical_accuracy: 0.8977
19296/60000 [========>.....................] - ETA: 1:12 - loss: 0.3404 - categorical_accuracy: 0.8978
19328/60000 [========>.....................] - ETA: 1:12 - loss: 0.3404 - categorical_accuracy: 0.8978
19360/60000 [========>.....................] - ETA: 1:12 - loss: 0.3399 - categorical_accuracy: 0.8979
19392/60000 [========>.....................] - ETA: 1:12 - loss: 0.3395 - categorical_accuracy: 0.8981
19424/60000 [========>.....................] - ETA: 1:12 - loss: 0.3393 - categorical_accuracy: 0.8981
19456/60000 [========>.....................] - ETA: 1:12 - loss: 0.3388 - categorical_accuracy: 0.8982
19488/60000 [========>.....................] - ETA: 1:12 - loss: 0.3383 - categorical_accuracy: 0.8984
19520/60000 [========>.....................] - ETA: 1:12 - loss: 0.3379 - categorical_accuracy: 0.8985
19552/60000 [========>.....................] - ETA: 1:12 - loss: 0.3376 - categorical_accuracy: 0.8985
19584/60000 [========>.....................] - ETA: 1:12 - loss: 0.3371 - categorical_accuracy: 0.8987
19616/60000 [========>.....................] - ETA: 1:12 - loss: 0.3368 - categorical_accuracy: 0.8988
19648/60000 [========>.....................] - ETA: 1:11 - loss: 0.3364 - categorical_accuracy: 0.8989
19680/60000 [========>.....................] - ETA: 1:11 - loss: 0.3359 - categorical_accuracy: 0.8990
19712/60000 [========>.....................] - ETA: 1:11 - loss: 0.3354 - categorical_accuracy: 0.8992
19744/60000 [========>.....................] - ETA: 1:11 - loss: 0.3349 - categorical_accuracy: 0.8993
19776/60000 [========>.....................] - ETA: 1:11 - loss: 0.3347 - categorical_accuracy: 0.8994
19808/60000 [========>.....................] - ETA: 1:11 - loss: 0.3345 - categorical_accuracy: 0.8994
19840/60000 [========>.....................] - ETA: 1:11 - loss: 0.3342 - categorical_accuracy: 0.8995
19872/60000 [========>.....................] - ETA: 1:11 - loss: 0.3338 - categorical_accuracy: 0.8996
19904/60000 [========>.....................] - ETA: 1:11 - loss: 0.3337 - categorical_accuracy: 0.8996
19936/60000 [========>.....................] - ETA: 1:11 - loss: 0.3332 - categorical_accuracy: 0.8998
19968/60000 [========>.....................] - ETA: 1:11 - loss: 0.3329 - categorical_accuracy: 0.8998
20000/60000 [=========>....................] - ETA: 1:11 - loss: 0.3324 - categorical_accuracy: 0.9000
20032/60000 [=========>....................] - ETA: 1:11 - loss: 0.3324 - categorical_accuracy: 0.9001
20064/60000 [=========>....................] - ETA: 1:11 - loss: 0.3320 - categorical_accuracy: 0.9002
20096/60000 [=========>....................] - ETA: 1:11 - loss: 0.3317 - categorical_accuracy: 0.9003
20128/60000 [=========>....................] - ETA: 1:11 - loss: 0.3320 - categorical_accuracy: 0.9002
20160/60000 [=========>....................] - ETA: 1:11 - loss: 0.3316 - categorical_accuracy: 0.9003
20192/60000 [=========>....................] - ETA: 1:11 - loss: 0.3313 - categorical_accuracy: 0.9005
20224/60000 [=========>....................] - ETA: 1:10 - loss: 0.3312 - categorical_accuracy: 0.9005
20256/60000 [=========>....................] - ETA: 1:10 - loss: 0.3310 - categorical_accuracy: 0.9005
20288/60000 [=========>....................] - ETA: 1:10 - loss: 0.3309 - categorical_accuracy: 0.9004
20320/60000 [=========>....................] - ETA: 1:10 - loss: 0.3305 - categorical_accuracy: 0.9006
20352/60000 [=========>....................] - ETA: 1:10 - loss: 0.3302 - categorical_accuracy: 0.9007
20384/60000 [=========>....................] - ETA: 1:10 - loss: 0.3298 - categorical_accuracy: 0.9008
20416/60000 [=========>....................] - ETA: 1:10 - loss: 0.3294 - categorical_accuracy: 0.9009
20448/60000 [=========>....................] - ETA: 1:10 - loss: 0.3291 - categorical_accuracy: 0.9010
20480/60000 [=========>....................] - ETA: 1:10 - loss: 0.3287 - categorical_accuracy: 0.9012
20512/60000 [=========>....................] - ETA: 1:10 - loss: 0.3283 - categorical_accuracy: 0.9013
20544/60000 [=========>....................] - ETA: 1:10 - loss: 0.3287 - categorical_accuracy: 0.9013
20576/60000 [=========>....................] - ETA: 1:10 - loss: 0.3283 - categorical_accuracy: 0.9014
20608/60000 [=========>....................] - ETA: 1:10 - loss: 0.3280 - categorical_accuracy: 0.9015
20640/60000 [=========>....................] - ETA: 1:10 - loss: 0.3276 - categorical_accuracy: 0.9016
20672/60000 [=========>....................] - ETA: 1:10 - loss: 0.3271 - categorical_accuracy: 0.9018
20704/60000 [=========>....................] - ETA: 1:10 - loss: 0.3267 - categorical_accuracy: 0.9019
20736/60000 [=========>....................] - ETA: 1:10 - loss: 0.3263 - categorical_accuracy: 0.9020
20768/60000 [=========>....................] - ETA: 1:09 - loss: 0.3261 - categorical_accuracy: 0.9021
20800/60000 [=========>....................] - ETA: 1:09 - loss: 0.3259 - categorical_accuracy: 0.9022
20832/60000 [=========>....................] - ETA: 1:09 - loss: 0.3256 - categorical_accuracy: 0.9022
20864/60000 [=========>....................] - ETA: 1:09 - loss: 0.3261 - categorical_accuracy: 0.9023
20896/60000 [=========>....................] - ETA: 1:09 - loss: 0.3258 - categorical_accuracy: 0.9023
20928/60000 [=========>....................] - ETA: 1:09 - loss: 0.3254 - categorical_accuracy: 0.9024
20960/60000 [=========>....................] - ETA: 1:09 - loss: 0.3251 - categorical_accuracy: 0.9024
20992/60000 [=========>....................] - ETA: 1:09 - loss: 0.3247 - categorical_accuracy: 0.9026
21024/60000 [=========>....................] - ETA: 1:09 - loss: 0.3244 - categorical_accuracy: 0.9026
21056/60000 [=========>....................] - ETA: 1:09 - loss: 0.3240 - categorical_accuracy: 0.9027
21088/60000 [=========>....................] - ETA: 1:09 - loss: 0.3240 - categorical_accuracy: 0.9028
21120/60000 [=========>....................] - ETA: 1:09 - loss: 0.3235 - categorical_accuracy: 0.9029
21152/60000 [=========>....................] - ETA: 1:09 - loss: 0.3232 - categorical_accuracy: 0.9030
21184/60000 [=========>....................] - ETA: 1:09 - loss: 0.3233 - categorical_accuracy: 0.9030
21216/60000 [=========>....................] - ETA: 1:09 - loss: 0.3230 - categorical_accuracy: 0.9032
21248/60000 [=========>....................] - ETA: 1:09 - loss: 0.3228 - categorical_accuracy: 0.9033
21280/60000 [=========>....................] - ETA: 1:09 - loss: 0.3230 - categorical_accuracy: 0.9033
21312/60000 [=========>....................] - ETA: 1:08 - loss: 0.3231 - categorical_accuracy: 0.9034
21344/60000 [=========>....................] - ETA: 1:08 - loss: 0.3227 - categorical_accuracy: 0.9036
21376/60000 [=========>....................] - ETA: 1:08 - loss: 0.3223 - categorical_accuracy: 0.9037
21408/60000 [=========>....................] - ETA: 1:08 - loss: 0.3219 - categorical_accuracy: 0.9039
21440/60000 [=========>....................] - ETA: 1:08 - loss: 0.3216 - categorical_accuracy: 0.9040
21472/60000 [=========>....................] - ETA: 1:08 - loss: 0.3212 - categorical_accuracy: 0.9041
21504/60000 [=========>....................] - ETA: 1:08 - loss: 0.3208 - categorical_accuracy: 0.9042
21536/60000 [=========>....................] - ETA: 1:08 - loss: 0.3207 - categorical_accuracy: 0.9043
21568/60000 [=========>....................] - ETA: 1:08 - loss: 0.3204 - categorical_accuracy: 0.9043
21600/60000 [=========>....................] - ETA: 1:08 - loss: 0.3201 - categorical_accuracy: 0.9044
21632/60000 [=========>....................] - ETA: 1:08 - loss: 0.3198 - categorical_accuracy: 0.9045
21664/60000 [=========>....................] - ETA: 1:08 - loss: 0.3196 - categorical_accuracy: 0.9045
21696/60000 [=========>....................] - ETA: 1:08 - loss: 0.3194 - categorical_accuracy: 0.9046
21728/60000 [=========>....................] - ETA: 1:08 - loss: 0.3192 - categorical_accuracy: 0.9047
21760/60000 [=========>....................] - ETA: 1:08 - loss: 0.3189 - categorical_accuracy: 0.9047
21792/60000 [=========>....................] - ETA: 1:08 - loss: 0.3186 - categorical_accuracy: 0.9048
21824/60000 [=========>....................] - ETA: 1:08 - loss: 0.3184 - categorical_accuracy: 0.9049
21856/60000 [=========>....................] - ETA: 1:08 - loss: 0.3183 - categorical_accuracy: 0.9049
21888/60000 [=========>....................] - ETA: 1:07 - loss: 0.3180 - categorical_accuracy: 0.9050
21920/60000 [=========>....................] - ETA: 1:07 - loss: 0.3178 - categorical_accuracy: 0.9051
21952/60000 [=========>....................] - ETA: 1:07 - loss: 0.3174 - categorical_accuracy: 0.9052
21984/60000 [=========>....................] - ETA: 1:07 - loss: 0.3170 - categorical_accuracy: 0.9054
22016/60000 [==========>...................] - ETA: 1:07 - loss: 0.3166 - categorical_accuracy: 0.9055
22048/60000 [==========>...................] - ETA: 1:07 - loss: 0.3163 - categorical_accuracy: 0.9056
22080/60000 [==========>...................] - ETA: 1:07 - loss: 0.3161 - categorical_accuracy: 0.9056
22112/60000 [==========>...................] - ETA: 1:07 - loss: 0.3157 - categorical_accuracy: 0.9058
22144/60000 [==========>...................] - ETA: 1:07 - loss: 0.3154 - categorical_accuracy: 0.9058
22176/60000 [==========>...................] - ETA: 1:07 - loss: 0.3152 - categorical_accuracy: 0.9059
22208/60000 [==========>...................] - ETA: 1:07 - loss: 0.3152 - categorical_accuracy: 0.9059
22240/60000 [==========>...................] - ETA: 1:07 - loss: 0.3148 - categorical_accuracy: 0.9060
22272/60000 [==========>...................] - ETA: 1:07 - loss: 0.3144 - categorical_accuracy: 0.9062
22304/60000 [==========>...................] - ETA: 1:07 - loss: 0.3141 - categorical_accuracy: 0.9062
22336/60000 [==========>...................] - ETA: 1:07 - loss: 0.3137 - categorical_accuracy: 0.9064
22368/60000 [==========>...................] - ETA: 1:07 - loss: 0.3134 - categorical_accuracy: 0.9064
22400/60000 [==========>...................] - ETA: 1:07 - loss: 0.3131 - categorical_accuracy: 0.9065
22432/60000 [==========>...................] - ETA: 1:06 - loss: 0.3128 - categorical_accuracy: 0.9066
22464/60000 [==========>...................] - ETA: 1:06 - loss: 0.3127 - categorical_accuracy: 0.9067
22496/60000 [==========>...................] - ETA: 1:06 - loss: 0.3124 - categorical_accuracy: 0.9068
22528/60000 [==========>...................] - ETA: 1:06 - loss: 0.3125 - categorical_accuracy: 0.9068
22560/60000 [==========>...................] - ETA: 1:06 - loss: 0.3122 - categorical_accuracy: 0.9068
22592/60000 [==========>...................] - ETA: 1:06 - loss: 0.3122 - categorical_accuracy: 0.9068
22624/60000 [==========>...................] - ETA: 1:06 - loss: 0.3119 - categorical_accuracy: 0.9069
22656/60000 [==========>...................] - ETA: 1:06 - loss: 0.3118 - categorical_accuracy: 0.9069
22688/60000 [==========>...................] - ETA: 1:06 - loss: 0.3117 - categorical_accuracy: 0.9070
22720/60000 [==========>...................] - ETA: 1:06 - loss: 0.3117 - categorical_accuracy: 0.9069
22752/60000 [==========>...................] - ETA: 1:06 - loss: 0.3113 - categorical_accuracy: 0.9070
22784/60000 [==========>...................] - ETA: 1:06 - loss: 0.3116 - categorical_accuracy: 0.9069
22816/60000 [==========>...................] - ETA: 1:06 - loss: 0.3117 - categorical_accuracy: 0.9069
22848/60000 [==========>...................] - ETA: 1:06 - loss: 0.3113 - categorical_accuracy: 0.9070
22880/60000 [==========>...................] - ETA: 1:06 - loss: 0.3109 - categorical_accuracy: 0.9071
22912/60000 [==========>...................] - ETA: 1:06 - loss: 0.3109 - categorical_accuracy: 0.9071
22944/60000 [==========>...................] - ETA: 1:06 - loss: 0.3107 - categorical_accuracy: 0.9072
22976/60000 [==========>...................] - ETA: 1:06 - loss: 0.3103 - categorical_accuracy: 0.9073
23008/60000 [==========>...................] - ETA: 1:05 - loss: 0.3099 - categorical_accuracy: 0.9075
23040/60000 [==========>...................] - ETA: 1:05 - loss: 0.3097 - categorical_accuracy: 0.9075
23072/60000 [==========>...................] - ETA: 1:05 - loss: 0.3093 - categorical_accuracy: 0.9076
23104/60000 [==========>...................] - ETA: 1:05 - loss: 0.3093 - categorical_accuracy: 0.9077
23136/60000 [==========>...................] - ETA: 1:05 - loss: 0.3091 - categorical_accuracy: 0.9077
23168/60000 [==========>...................] - ETA: 1:05 - loss: 0.3091 - categorical_accuracy: 0.9078
23200/60000 [==========>...................] - ETA: 1:05 - loss: 0.3089 - categorical_accuracy: 0.9078
23232/60000 [==========>...................] - ETA: 1:05 - loss: 0.3089 - categorical_accuracy: 0.9078
23264/60000 [==========>...................] - ETA: 1:05 - loss: 0.3087 - categorical_accuracy: 0.9079
23296/60000 [==========>...................] - ETA: 1:05 - loss: 0.3084 - categorical_accuracy: 0.9080
23328/60000 [==========>...................] - ETA: 1:05 - loss: 0.3080 - categorical_accuracy: 0.9081
23360/60000 [==========>...................] - ETA: 1:05 - loss: 0.3081 - categorical_accuracy: 0.9081
23392/60000 [==========>...................] - ETA: 1:05 - loss: 0.3077 - categorical_accuracy: 0.9083
23424/60000 [==========>...................] - ETA: 1:05 - loss: 0.3076 - categorical_accuracy: 0.9083
23456/60000 [==========>...................] - ETA: 1:05 - loss: 0.3073 - categorical_accuracy: 0.9084
23488/60000 [==========>...................] - ETA: 1:05 - loss: 0.3071 - categorical_accuracy: 0.9084
23520/60000 [==========>...................] - ETA: 1:05 - loss: 0.3070 - categorical_accuracy: 0.9085
23552/60000 [==========>...................] - ETA: 1:04 - loss: 0.3067 - categorical_accuracy: 0.9085
23584/60000 [==========>...................] - ETA: 1:04 - loss: 0.3064 - categorical_accuracy: 0.9086
23616/60000 [==========>...................] - ETA: 1:04 - loss: 0.3060 - categorical_accuracy: 0.9087
23648/60000 [==========>...................] - ETA: 1:04 - loss: 0.3060 - categorical_accuracy: 0.9087
23680/60000 [==========>...................] - ETA: 1:04 - loss: 0.3057 - categorical_accuracy: 0.9089
23712/60000 [==========>...................] - ETA: 1:04 - loss: 0.3055 - categorical_accuracy: 0.9089
23744/60000 [==========>...................] - ETA: 1:04 - loss: 0.3052 - categorical_accuracy: 0.9090
23776/60000 [==========>...................] - ETA: 1:04 - loss: 0.3050 - categorical_accuracy: 0.9091
23808/60000 [==========>...................] - ETA: 1:04 - loss: 0.3046 - categorical_accuracy: 0.9092
23840/60000 [==========>...................] - ETA: 1:04 - loss: 0.3043 - categorical_accuracy: 0.9093
23872/60000 [==========>...................] - ETA: 1:04 - loss: 0.3040 - categorical_accuracy: 0.9094
23904/60000 [==========>...................] - ETA: 1:04 - loss: 0.3036 - categorical_accuracy: 0.9095
23936/60000 [==========>...................] - ETA: 1:04 - loss: 0.3033 - categorical_accuracy: 0.9096
23968/60000 [==========>...................] - ETA: 1:04 - loss: 0.3030 - categorical_accuracy: 0.9097
24000/60000 [===========>..................] - ETA: 1:04 - loss: 0.3028 - categorical_accuracy: 0.9098
24032/60000 [===========>..................] - ETA: 1:04 - loss: 0.3029 - categorical_accuracy: 0.9098
24064/60000 [===========>..................] - ETA: 1:04 - loss: 0.3025 - categorical_accuracy: 0.9099
24096/60000 [===========>..................] - ETA: 1:04 - loss: 0.3023 - categorical_accuracy: 0.9100
24128/60000 [===========>..................] - ETA: 1:03 - loss: 0.3023 - categorical_accuracy: 0.9099
24160/60000 [===========>..................] - ETA: 1:03 - loss: 0.3021 - categorical_accuracy: 0.9100
24192/60000 [===========>..................] - ETA: 1:03 - loss: 0.3018 - categorical_accuracy: 0.9101
24224/60000 [===========>..................] - ETA: 1:03 - loss: 0.3015 - categorical_accuracy: 0.9101
24256/60000 [===========>..................] - ETA: 1:03 - loss: 0.3013 - categorical_accuracy: 0.9101
24288/60000 [===========>..................] - ETA: 1:03 - loss: 0.3012 - categorical_accuracy: 0.9102
24320/60000 [===========>..................] - ETA: 1:03 - loss: 0.3010 - categorical_accuracy: 0.9102
24352/60000 [===========>..................] - ETA: 1:03 - loss: 0.3007 - categorical_accuracy: 0.9103
24384/60000 [===========>..................] - ETA: 1:03 - loss: 0.3004 - categorical_accuracy: 0.9104
24416/60000 [===========>..................] - ETA: 1:03 - loss: 0.3003 - categorical_accuracy: 0.9104
24448/60000 [===========>..................] - ETA: 1:03 - loss: 0.3002 - categorical_accuracy: 0.9104
24480/60000 [===========>..................] - ETA: 1:03 - loss: 0.3000 - categorical_accuracy: 0.9105
24512/60000 [===========>..................] - ETA: 1:03 - loss: 0.2997 - categorical_accuracy: 0.9106
24544/60000 [===========>..................] - ETA: 1:03 - loss: 0.2994 - categorical_accuracy: 0.9107
24576/60000 [===========>..................] - ETA: 1:03 - loss: 0.2991 - categorical_accuracy: 0.9108
24608/60000 [===========>..................] - ETA: 1:03 - loss: 0.2988 - categorical_accuracy: 0.9109
24640/60000 [===========>..................] - ETA: 1:03 - loss: 0.2985 - categorical_accuracy: 0.9110
24672/60000 [===========>..................] - ETA: 1:03 - loss: 0.2985 - categorical_accuracy: 0.9111
24704/60000 [===========>..................] - ETA: 1:02 - loss: 0.2981 - categorical_accuracy: 0.9112
24736/60000 [===========>..................] - ETA: 1:02 - loss: 0.2979 - categorical_accuracy: 0.9113
24768/60000 [===========>..................] - ETA: 1:02 - loss: 0.2980 - categorical_accuracy: 0.9113
24800/60000 [===========>..................] - ETA: 1:02 - loss: 0.2978 - categorical_accuracy: 0.9113
24832/60000 [===========>..................] - ETA: 1:02 - loss: 0.2977 - categorical_accuracy: 0.9114
24864/60000 [===========>..................] - ETA: 1:02 - loss: 0.2976 - categorical_accuracy: 0.9114
24896/60000 [===========>..................] - ETA: 1:02 - loss: 0.2975 - categorical_accuracy: 0.9115
24928/60000 [===========>..................] - ETA: 1:02 - loss: 0.2971 - categorical_accuracy: 0.9116
24960/60000 [===========>..................] - ETA: 1:02 - loss: 0.2969 - categorical_accuracy: 0.9117
24992/60000 [===========>..................] - ETA: 1:02 - loss: 0.2966 - categorical_accuracy: 0.9118
25024/60000 [===========>..................] - ETA: 1:02 - loss: 0.2964 - categorical_accuracy: 0.9118
25056/60000 [===========>..................] - ETA: 1:02 - loss: 0.2961 - categorical_accuracy: 0.9119
25088/60000 [===========>..................] - ETA: 1:02 - loss: 0.2960 - categorical_accuracy: 0.9119
25120/60000 [===========>..................] - ETA: 1:02 - loss: 0.2958 - categorical_accuracy: 0.9120
25152/60000 [===========>..................] - ETA: 1:02 - loss: 0.2955 - categorical_accuracy: 0.9121
25184/60000 [===========>..................] - ETA: 1:02 - loss: 0.2955 - categorical_accuracy: 0.9121
25216/60000 [===========>..................] - ETA: 1:02 - loss: 0.2953 - categorical_accuracy: 0.9122
25248/60000 [===========>..................] - ETA: 1:01 - loss: 0.2950 - categorical_accuracy: 0.9123
25280/60000 [===========>..................] - ETA: 1:01 - loss: 0.2950 - categorical_accuracy: 0.9123
25312/60000 [===========>..................] - ETA: 1:01 - loss: 0.2946 - categorical_accuracy: 0.9124
25344/60000 [===========>..................] - ETA: 1:01 - loss: 0.2943 - categorical_accuracy: 0.9125
25376/60000 [===========>..................] - ETA: 1:01 - loss: 0.2941 - categorical_accuracy: 0.9126
25408/60000 [===========>..................] - ETA: 1:01 - loss: 0.2937 - categorical_accuracy: 0.9127
25440/60000 [===========>..................] - ETA: 1:01 - loss: 0.2936 - categorical_accuracy: 0.9127
25472/60000 [===========>..................] - ETA: 1:01 - loss: 0.2936 - categorical_accuracy: 0.9126
25504/60000 [===========>..................] - ETA: 1:01 - loss: 0.2934 - categorical_accuracy: 0.9127
25536/60000 [===========>..................] - ETA: 1:01 - loss: 0.2932 - categorical_accuracy: 0.9128
25568/60000 [===========>..................] - ETA: 1:01 - loss: 0.2929 - categorical_accuracy: 0.9129
25600/60000 [===========>..................] - ETA: 1:01 - loss: 0.2926 - categorical_accuracy: 0.9130
25632/60000 [===========>..................] - ETA: 1:01 - loss: 0.2923 - categorical_accuracy: 0.9131
25664/60000 [===========>..................] - ETA: 1:01 - loss: 0.2920 - categorical_accuracy: 0.9131
25696/60000 [===========>..................] - ETA: 1:01 - loss: 0.2919 - categorical_accuracy: 0.9131
25728/60000 [===========>..................] - ETA: 1:01 - loss: 0.2917 - categorical_accuracy: 0.9132
25760/60000 [===========>..................] - ETA: 1:01 - loss: 0.2915 - categorical_accuracy: 0.9133
25792/60000 [===========>..................] - ETA: 1:01 - loss: 0.2913 - categorical_accuracy: 0.9133
25824/60000 [===========>..................] - ETA: 1:00 - loss: 0.2916 - categorical_accuracy: 0.9133
25856/60000 [===========>..................] - ETA: 1:00 - loss: 0.2914 - categorical_accuracy: 0.9133
25888/60000 [===========>..................] - ETA: 1:00 - loss: 0.2913 - categorical_accuracy: 0.9134
25920/60000 [===========>..................] - ETA: 1:00 - loss: 0.2911 - categorical_accuracy: 0.9134
25952/60000 [===========>..................] - ETA: 1:00 - loss: 0.2908 - categorical_accuracy: 0.9135
25984/60000 [===========>..................] - ETA: 1:00 - loss: 0.2905 - categorical_accuracy: 0.9136
26016/60000 [============>.................] - ETA: 1:00 - loss: 0.2904 - categorical_accuracy: 0.9137
26048/60000 [============>.................] - ETA: 1:00 - loss: 0.2903 - categorical_accuracy: 0.9137
26080/60000 [============>.................] - ETA: 1:00 - loss: 0.2901 - categorical_accuracy: 0.9137
26112/60000 [============>.................] - ETA: 1:00 - loss: 0.2898 - categorical_accuracy: 0.9138
26144/60000 [============>.................] - ETA: 1:00 - loss: 0.2896 - categorical_accuracy: 0.9139
26176/60000 [============>.................] - ETA: 1:00 - loss: 0.2892 - categorical_accuracy: 0.9140
26208/60000 [============>.................] - ETA: 1:00 - loss: 0.2890 - categorical_accuracy: 0.9140
26240/60000 [============>.................] - ETA: 1:00 - loss: 0.2890 - categorical_accuracy: 0.9141
26272/60000 [============>.................] - ETA: 1:00 - loss: 0.2887 - categorical_accuracy: 0.9141
26304/60000 [============>.................] - ETA: 1:00 - loss: 0.2886 - categorical_accuracy: 0.9141
26336/60000 [============>.................] - ETA: 1:00 - loss: 0.2885 - categorical_accuracy: 0.9142
26368/60000 [============>.................] - ETA: 59s - loss: 0.2882 - categorical_accuracy: 0.9143 
26400/60000 [============>.................] - ETA: 59s - loss: 0.2878 - categorical_accuracy: 0.9144
26432/60000 [============>.................] - ETA: 59s - loss: 0.2876 - categorical_accuracy: 0.9144
26464/60000 [============>.................] - ETA: 59s - loss: 0.2875 - categorical_accuracy: 0.9144
26496/60000 [============>.................] - ETA: 59s - loss: 0.2873 - categorical_accuracy: 0.9145
26528/60000 [============>.................] - ETA: 59s - loss: 0.2870 - categorical_accuracy: 0.9146
26560/60000 [============>.................] - ETA: 59s - loss: 0.2867 - categorical_accuracy: 0.9146
26592/60000 [============>.................] - ETA: 59s - loss: 0.2864 - categorical_accuracy: 0.9147
26624/60000 [============>.................] - ETA: 59s - loss: 0.2863 - categorical_accuracy: 0.9148
26656/60000 [============>.................] - ETA: 59s - loss: 0.2860 - categorical_accuracy: 0.9149
26688/60000 [============>.................] - ETA: 59s - loss: 0.2860 - categorical_accuracy: 0.9149
26720/60000 [============>.................] - ETA: 59s - loss: 0.2859 - categorical_accuracy: 0.9149
26752/60000 [============>.................] - ETA: 59s - loss: 0.2856 - categorical_accuracy: 0.9150
26784/60000 [============>.................] - ETA: 59s - loss: 0.2854 - categorical_accuracy: 0.9150
26816/60000 [============>.................] - ETA: 59s - loss: 0.2852 - categorical_accuracy: 0.9150
26848/60000 [============>.................] - ETA: 59s - loss: 0.2850 - categorical_accuracy: 0.9150
26880/60000 [============>.................] - ETA: 59s - loss: 0.2851 - categorical_accuracy: 0.9150
26912/60000 [============>.................] - ETA: 58s - loss: 0.2848 - categorical_accuracy: 0.9151
26944/60000 [============>.................] - ETA: 58s - loss: 0.2846 - categorical_accuracy: 0.9152
26976/60000 [============>.................] - ETA: 58s - loss: 0.2843 - categorical_accuracy: 0.9153
27008/60000 [============>.................] - ETA: 58s - loss: 0.2842 - categorical_accuracy: 0.9152
27040/60000 [============>.................] - ETA: 58s - loss: 0.2840 - categorical_accuracy: 0.9153
27072/60000 [============>.................] - ETA: 58s - loss: 0.2838 - categorical_accuracy: 0.9153
27104/60000 [============>.................] - ETA: 58s - loss: 0.2838 - categorical_accuracy: 0.9153
27136/60000 [============>.................] - ETA: 58s - loss: 0.2835 - categorical_accuracy: 0.9154
27168/60000 [============>.................] - ETA: 58s - loss: 0.2833 - categorical_accuracy: 0.9154
27200/60000 [============>.................] - ETA: 58s - loss: 0.2835 - categorical_accuracy: 0.9155
27232/60000 [============>.................] - ETA: 58s - loss: 0.2833 - categorical_accuracy: 0.9155
27264/60000 [============>.................] - ETA: 58s - loss: 0.2831 - categorical_accuracy: 0.9156
27296/60000 [============>.................] - ETA: 58s - loss: 0.2829 - categorical_accuracy: 0.9156
27328/60000 [============>.................] - ETA: 58s - loss: 0.2826 - categorical_accuracy: 0.9157
27360/60000 [============>.................] - ETA: 58s - loss: 0.2825 - categorical_accuracy: 0.9157
27392/60000 [============>.................] - ETA: 58s - loss: 0.2823 - categorical_accuracy: 0.9157
27424/60000 [============>.................] - ETA: 58s - loss: 0.2822 - categorical_accuracy: 0.9158
27456/60000 [============>.................] - ETA: 58s - loss: 0.2821 - categorical_accuracy: 0.9158
27488/60000 [============>.................] - ETA: 58s - loss: 0.2821 - categorical_accuracy: 0.9159
27520/60000 [============>.................] - ETA: 57s - loss: 0.2818 - categorical_accuracy: 0.9160
27552/60000 [============>.................] - ETA: 57s - loss: 0.2815 - categorical_accuracy: 0.9160
27584/60000 [============>.................] - ETA: 57s - loss: 0.2812 - categorical_accuracy: 0.9161
27616/60000 [============>.................] - ETA: 57s - loss: 0.2816 - categorical_accuracy: 0.9161
27648/60000 [============>.................] - ETA: 57s - loss: 0.2814 - categorical_accuracy: 0.9161
27680/60000 [============>.................] - ETA: 57s - loss: 0.2812 - categorical_accuracy: 0.9161
27712/60000 [============>.................] - ETA: 57s - loss: 0.2810 - categorical_accuracy: 0.9162
27744/60000 [============>.................] - ETA: 57s - loss: 0.2807 - categorical_accuracy: 0.9163
27776/60000 [============>.................] - ETA: 57s - loss: 0.2806 - categorical_accuracy: 0.9164
27808/60000 [============>.................] - ETA: 57s - loss: 0.2804 - categorical_accuracy: 0.9164
27840/60000 [============>.................] - ETA: 57s - loss: 0.2802 - categorical_accuracy: 0.9165
27872/60000 [============>.................] - ETA: 57s - loss: 0.2801 - categorical_accuracy: 0.9165
27904/60000 [============>.................] - ETA: 57s - loss: 0.2798 - categorical_accuracy: 0.9166
27936/60000 [============>.................] - ETA: 57s - loss: 0.2796 - categorical_accuracy: 0.9166
27968/60000 [============>.................] - ETA: 57s - loss: 0.2793 - categorical_accuracy: 0.9167
28000/60000 [=============>................] - ETA: 57s - loss: 0.2793 - categorical_accuracy: 0.9167
28032/60000 [=============>................] - ETA: 56s - loss: 0.2794 - categorical_accuracy: 0.9167
28064/60000 [=============>................] - ETA: 56s - loss: 0.2794 - categorical_accuracy: 0.9167
28096/60000 [=============>................] - ETA: 56s - loss: 0.2791 - categorical_accuracy: 0.9168
28128/60000 [=============>................] - ETA: 56s - loss: 0.2791 - categorical_accuracy: 0.9168
28160/60000 [=============>................] - ETA: 56s - loss: 0.2789 - categorical_accuracy: 0.9169
28192/60000 [=============>................] - ETA: 56s - loss: 0.2788 - categorical_accuracy: 0.9169
28224/60000 [=============>................] - ETA: 56s - loss: 0.2785 - categorical_accuracy: 0.9170
28256/60000 [=============>................] - ETA: 56s - loss: 0.2784 - categorical_accuracy: 0.9170
28288/60000 [=============>................] - ETA: 56s - loss: 0.2782 - categorical_accuracy: 0.9171
28320/60000 [=============>................] - ETA: 56s - loss: 0.2781 - categorical_accuracy: 0.9171
28352/60000 [=============>................] - ETA: 56s - loss: 0.2780 - categorical_accuracy: 0.9171
28384/60000 [=============>................] - ETA: 56s - loss: 0.2778 - categorical_accuracy: 0.9171
28416/60000 [=============>................] - ETA: 56s - loss: 0.2775 - categorical_accuracy: 0.9172
28448/60000 [=============>................] - ETA: 56s - loss: 0.2772 - categorical_accuracy: 0.9173
28480/60000 [=============>................] - ETA: 56s - loss: 0.2769 - categorical_accuracy: 0.9174
28512/60000 [=============>................] - ETA: 56s - loss: 0.2768 - categorical_accuracy: 0.9174
28544/60000 [=============>................] - ETA: 56s - loss: 0.2771 - categorical_accuracy: 0.9174
28576/60000 [=============>................] - ETA: 55s - loss: 0.2769 - categorical_accuracy: 0.9175
28608/60000 [=============>................] - ETA: 55s - loss: 0.2769 - categorical_accuracy: 0.9175
28640/60000 [=============>................] - ETA: 55s - loss: 0.2769 - categorical_accuracy: 0.9175
28672/60000 [=============>................] - ETA: 55s - loss: 0.2766 - categorical_accuracy: 0.9176
28704/60000 [=============>................] - ETA: 55s - loss: 0.2765 - categorical_accuracy: 0.9176
28736/60000 [=============>................] - ETA: 55s - loss: 0.2765 - categorical_accuracy: 0.9176
28768/60000 [=============>................] - ETA: 55s - loss: 0.2763 - categorical_accuracy: 0.9177
28800/60000 [=============>................] - ETA: 55s - loss: 0.2760 - categorical_accuracy: 0.9177
28832/60000 [=============>................] - ETA: 55s - loss: 0.2758 - categorical_accuracy: 0.9178
28864/60000 [=============>................] - ETA: 55s - loss: 0.2756 - categorical_accuracy: 0.9178
28896/60000 [=============>................] - ETA: 55s - loss: 0.2755 - categorical_accuracy: 0.9178
28928/60000 [=============>................] - ETA: 55s - loss: 0.2756 - categorical_accuracy: 0.9178
28960/60000 [=============>................] - ETA: 55s - loss: 0.2754 - categorical_accuracy: 0.9179
28992/60000 [=============>................] - ETA: 55s - loss: 0.2752 - categorical_accuracy: 0.9179
29024/60000 [=============>................] - ETA: 55s - loss: 0.2750 - categorical_accuracy: 0.9180
29056/60000 [=============>................] - ETA: 55s - loss: 0.2747 - categorical_accuracy: 0.9181
29088/60000 [=============>................] - ETA: 55s - loss: 0.2745 - categorical_accuracy: 0.9182
29120/60000 [=============>................] - ETA: 55s - loss: 0.2743 - categorical_accuracy: 0.9182
29152/60000 [=============>................] - ETA: 54s - loss: 0.2741 - categorical_accuracy: 0.9182
29184/60000 [=============>................] - ETA: 54s - loss: 0.2739 - categorical_accuracy: 0.9183
29216/60000 [=============>................] - ETA: 54s - loss: 0.2736 - categorical_accuracy: 0.9184
29248/60000 [=============>................] - ETA: 54s - loss: 0.2736 - categorical_accuracy: 0.9184
29280/60000 [=============>................] - ETA: 54s - loss: 0.2734 - categorical_accuracy: 0.9185
29312/60000 [=============>................] - ETA: 54s - loss: 0.2732 - categorical_accuracy: 0.9185
29344/60000 [=============>................] - ETA: 54s - loss: 0.2730 - categorical_accuracy: 0.9186
29376/60000 [=============>................] - ETA: 54s - loss: 0.2727 - categorical_accuracy: 0.9187
29408/60000 [=============>................] - ETA: 54s - loss: 0.2724 - categorical_accuracy: 0.9188
29440/60000 [=============>................] - ETA: 54s - loss: 0.2723 - categorical_accuracy: 0.9188
29472/60000 [=============>................] - ETA: 54s - loss: 0.2721 - categorical_accuracy: 0.9189
29504/60000 [=============>................] - ETA: 54s - loss: 0.2718 - categorical_accuracy: 0.9190
29536/60000 [=============>................] - ETA: 54s - loss: 0.2718 - categorical_accuracy: 0.9190
29568/60000 [=============>................] - ETA: 54s - loss: 0.2716 - categorical_accuracy: 0.9191
29600/60000 [=============>................] - ETA: 54s - loss: 0.2714 - categorical_accuracy: 0.9192
29632/60000 [=============>................] - ETA: 54s - loss: 0.2711 - categorical_accuracy: 0.9192
29664/60000 [=============>................] - ETA: 54s - loss: 0.2710 - categorical_accuracy: 0.9192
29696/60000 [=============>................] - ETA: 53s - loss: 0.2709 - categorical_accuracy: 0.9192
29728/60000 [=============>................] - ETA: 53s - loss: 0.2707 - categorical_accuracy: 0.9193
29760/60000 [=============>................] - ETA: 53s - loss: 0.2705 - categorical_accuracy: 0.9194
29792/60000 [=============>................] - ETA: 53s - loss: 0.2702 - categorical_accuracy: 0.9195
29824/60000 [=============>................] - ETA: 53s - loss: 0.2701 - categorical_accuracy: 0.9195
29856/60000 [=============>................] - ETA: 53s - loss: 0.2698 - categorical_accuracy: 0.9196
29888/60000 [=============>................] - ETA: 53s - loss: 0.2696 - categorical_accuracy: 0.9197
29920/60000 [=============>................] - ETA: 53s - loss: 0.2693 - categorical_accuracy: 0.9198
29952/60000 [=============>................] - ETA: 53s - loss: 0.2691 - categorical_accuracy: 0.9198
29984/60000 [=============>................] - ETA: 53s - loss: 0.2688 - categorical_accuracy: 0.9199
30016/60000 [==============>...............] - ETA: 53s - loss: 0.2687 - categorical_accuracy: 0.9199
30048/60000 [==============>...............] - ETA: 53s - loss: 0.2684 - categorical_accuracy: 0.9200
30080/60000 [==============>...............] - ETA: 53s - loss: 0.2684 - categorical_accuracy: 0.9200
30112/60000 [==============>...............] - ETA: 53s - loss: 0.2686 - categorical_accuracy: 0.9200
30144/60000 [==============>...............] - ETA: 53s - loss: 0.2685 - categorical_accuracy: 0.9201
30176/60000 [==============>...............] - ETA: 53s - loss: 0.2684 - categorical_accuracy: 0.9201
30208/60000 [==============>...............] - ETA: 53s - loss: 0.2682 - categorical_accuracy: 0.9202
30240/60000 [==============>...............] - ETA: 52s - loss: 0.2680 - categorical_accuracy: 0.9202
30272/60000 [==============>...............] - ETA: 52s - loss: 0.2678 - categorical_accuracy: 0.9203
30304/60000 [==============>...............] - ETA: 52s - loss: 0.2679 - categorical_accuracy: 0.9203
30336/60000 [==============>...............] - ETA: 52s - loss: 0.2677 - categorical_accuracy: 0.9204
30368/60000 [==============>...............] - ETA: 52s - loss: 0.2674 - categorical_accuracy: 0.9205
30400/60000 [==============>...............] - ETA: 52s - loss: 0.2673 - categorical_accuracy: 0.9205
30432/60000 [==============>...............] - ETA: 52s - loss: 0.2673 - categorical_accuracy: 0.9205
30464/60000 [==============>...............] - ETA: 52s - loss: 0.2671 - categorical_accuracy: 0.9206
30496/60000 [==============>...............] - ETA: 52s - loss: 0.2669 - categorical_accuracy: 0.9206
30528/60000 [==============>...............] - ETA: 52s - loss: 0.2667 - categorical_accuracy: 0.9207
30560/60000 [==============>...............] - ETA: 52s - loss: 0.2666 - categorical_accuracy: 0.9207
30592/60000 [==============>...............] - ETA: 52s - loss: 0.2665 - categorical_accuracy: 0.9207
30624/60000 [==============>...............] - ETA: 52s - loss: 0.2662 - categorical_accuracy: 0.9208
30656/60000 [==============>...............] - ETA: 52s - loss: 0.2660 - categorical_accuracy: 0.9208
30688/60000 [==============>...............] - ETA: 52s - loss: 0.2657 - categorical_accuracy: 0.9209
30720/60000 [==============>...............] - ETA: 52s - loss: 0.2657 - categorical_accuracy: 0.9209
30752/60000 [==============>...............] - ETA: 52s - loss: 0.2654 - categorical_accuracy: 0.9210
30784/60000 [==============>...............] - ETA: 52s - loss: 0.2653 - categorical_accuracy: 0.9210
30816/60000 [==============>...............] - ETA: 51s - loss: 0.2652 - categorical_accuracy: 0.9211
30848/60000 [==============>...............] - ETA: 51s - loss: 0.2650 - categorical_accuracy: 0.9211
30880/60000 [==============>...............] - ETA: 51s - loss: 0.2648 - categorical_accuracy: 0.9212
30912/60000 [==============>...............] - ETA: 51s - loss: 0.2646 - categorical_accuracy: 0.9212
30944/60000 [==============>...............] - ETA: 51s - loss: 0.2645 - categorical_accuracy: 0.9212
30976/60000 [==============>...............] - ETA: 51s - loss: 0.2643 - categorical_accuracy: 0.9213
31008/60000 [==============>...............] - ETA: 51s - loss: 0.2641 - categorical_accuracy: 0.9213
31040/60000 [==============>...............] - ETA: 51s - loss: 0.2639 - categorical_accuracy: 0.9214
31072/60000 [==============>...............] - ETA: 51s - loss: 0.2637 - categorical_accuracy: 0.9215
31104/60000 [==============>...............] - ETA: 51s - loss: 0.2637 - categorical_accuracy: 0.9215
31136/60000 [==============>...............] - ETA: 51s - loss: 0.2636 - categorical_accuracy: 0.9215
31168/60000 [==============>...............] - ETA: 51s - loss: 0.2634 - categorical_accuracy: 0.9216
31200/60000 [==============>...............] - ETA: 51s - loss: 0.2631 - categorical_accuracy: 0.9216
31232/60000 [==============>...............] - ETA: 51s - loss: 0.2629 - categorical_accuracy: 0.9217
31264/60000 [==============>...............] - ETA: 51s - loss: 0.2627 - categorical_accuracy: 0.9218
31296/60000 [==============>...............] - ETA: 51s - loss: 0.2626 - categorical_accuracy: 0.9218
31328/60000 [==============>...............] - ETA: 51s - loss: 0.2626 - categorical_accuracy: 0.9219
31360/60000 [==============>...............] - ETA: 50s - loss: 0.2625 - categorical_accuracy: 0.9219
31392/60000 [==============>...............] - ETA: 50s - loss: 0.2624 - categorical_accuracy: 0.9220
31424/60000 [==============>...............] - ETA: 50s - loss: 0.2622 - categorical_accuracy: 0.9220
31456/60000 [==============>...............] - ETA: 50s - loss: 0.2622 - categorical_accuracy: 0.9220
31488/60000 [==============>...............] - ETA: 50s - loss: 0.2620 - categorical_accuracy: 0.9221
31520/60000 [==============>...............] - ETA: 50s - loss: 0.2618 - categorical_accuracy: 0.9221
31552/60000 [==============>...............] - ETA: 50s - loss: 0.2617 - categorical_accuracy: 0.9222
31584/60000 [==============>...............] - ETA: 50s - loss: 0.2614 - categorical_accuracy: 0.9222
31616/60000 [==============>...............] - ETA: 50s - loss: 0.2613 - categorical_accuracy: 0.9223
31648/60000 [==============>...............] - ETA: 50s - loss: 0.2611 - categorical_accuracy: 0.9223
31680/60000 [==============>...............] - ETA: 50s - loss: 0.2609 - categorical_accuracy: 0.9224
31712/60000 [==============>...............] - ETA: 50s - loss: 0.2607 - categorical_accuracy: 0.9224
31744/60000 [==============>...............] - ETA: 50s - loss: 0.2606 - categorical_accuracy: 0.9225
31776/60000 [==============>...............] - ETA: 50s - loss: 0.2605 - categorical_accuracy: 0.9225
31808/60000 [==============>...............] - ETA: 50s - loss: 0.2605 - categorical_accuracy: 0.9225
31840/60000 [==============>...............] - ETA: 50s - loss: 0.2603 - categorical_accuracy: 0.9226
31872/60000 [==============>...............] - ETA: 50s - loss: 0.2602 - categorical_accuracy: 0.9226
31904/60000 [==============>...............] - ETA: 50s - loss: 0.2601 - categorical_accuracy: 0.9226
31936/60000 [==============>...............] - ETA: 49s - loss: 0.2599 - categorical_accuracy: 0.9227
31968/60000 [==============>...............] - ETA: 49s - loss: 0.2597 - categorical_accuracy: 0.9228
32000/60000 [===============>..............] - ETA: 49s - loss: 0.2594 - categorical_accuracy: 0.9228
32032/60000 [===============>..............] - ETA: 49s - loss: 0.2592 - categorical_accuracy: 0.9229
32064/60000 [===============>..............] - ETA: 49s - loss: 0.2590 - categorical_accuracy: 0.9230
32096/60000 [===============>..............] - ETA: 49s - loss: 0.2588 - categorical_accuracy: 0.9230
32128/60000 [===============>..............] - ETA: 49s - loss: 0.2587 - categorical_accuracy: 0.9231
32160/60000 [===============>..............] - ETA: 49s - loss: 0.2587 - categorical_accuracy: 0.9230
32192/60000 [===============>..............] - ETA: 49s - loss: 0.2586 - categorical_accuracy: 0.9230
32224/60000 [===============>..............] - ETA: 49s - loss: 0.2584 - categorical_accuracy: 0.9231
32256/60000 [===============>..............] - ETA: 49s - loss: 0.2583 - categorical_accuracy: 0.9231
32288/60000 [===============>..............] - ETA: 49s - loss: 0.2581 - categorical_accuracy: 0.9232
32320/60000 [===============>..............] - ETA: 49s - loss: 0.2579 - categorical_accuracy: 0.9232
32352/60000 [===============>..............] - ETA: 49s - loss: 0.2579 - categorical_accuracy: 0.9232
32384/60000 [===============>..............] - ETA: 49s - loss: 0.2577 - categorical_accuracy: 0.9233
32416/60000 [===============>..............] - ETA: 49s - loss: 0.2575 - categorical_accuracy: 0.9234
32448/60000 [===============>..............] - ETA: 49s - loss: 0.2574 - categorical_accuracy: 0.9234
32480/60000 [===============>..............] - ETA: 48s - loss: 0.2572 - categorical_accuracy: 0.9234
32512/60000 [===============>..............] - ETA: 48s - loss: 0.2570 - categorical_accuracy: 0.9235
32544/60000 [===============>..............] - ETA: 48s - loss: 0.2568 - categorical_accuracy: 0.9235
32576/60000 [===============>..............] - ETA: 48s - loss: 0.2567 - categorical_accuracy: 0.9235
32608/60000 [===============>..............] - ETA: 48s - loss: 0.2566 - categorical_accuracy: 0.9236
32640/60000 [===============>..............] - ETA: 48s - loss: 0.2564 - categorical_accuracy: 0.9236
32672/60000 [===============>..............] - ETA: 48s - loss: 0.2562 - categorical_accuracy: 0.9237
32704/60000 [===============>..............] - ETA: 48s - loss: 0.2561 - categorical_accuracy: 0.9237
32736/60000 [===============>..............] - ETA: 48s - loss: 0.2563 - categorical_accuracy: 0.9237
32768/60000 [===============>..............] - ETA: 48s - loss: 0.2561 - categorical_accuracy: 0.9237
32800/60000 [===============>..............] - ETA: 48s - loss: 0.2559 - categorical_accuracy: 0.9238
32832/60000 [===============>..............] - ETA: 48s - loss: 0.2557 - categorical_accuracy: 0.9239
32864/60000 [===============>..............] - ETA: 48s - loss: 0.2556 - categorical_accuracy: 0.9238
32896/60000 [===============>..............] - ETA: 48s - loss: 0.2556 - categorical_accuracy: 0.9239
32928/60000 [===============>..............] - ETA: 48s - loss: 0.2554 - categorical_accuracy: 0.9239
32960/60000 [===============>..............] - ETA: 48s - loss: 0.2552 - categorical_accuracy: 0.9240
32992/60000 [===============>..............] - ETA: 48s - loss: 0.2550 - categorical_accuracy: 0.9240
33024/60000 [===============>..............] - ETA: 48s - loss: 0.2548 - categorical_accuracy: 0.9241
33056/60000 [===============>..............] - ETA: 47s - loss: 0.2548 - categorical_accuracy: 0.9242
33088/60000 [===============>..............] - ETA: 47s - loss: 0.2546 - categorical_accuracy: 0.9242
33120/60000 [===============>..............] - ETA: 47s - loss: 0.2545 - categorical_accuracy: 0.9243
33152/60000 [===============>..............] - ETA: 47s - loss: 0.2545 - categorical_accuracy: 0.9242
33184/60000 [===============>..............] - ETA: 47s - loss: 0.2544 - categorical_accuracy: 0.9243
33216/60000 [===============>..............] - ETA: 47s - loss: 0.2543 - categorical_accuracy: 0.9243
33248/60000 [===============>..............] - ETA: 47s - loss: 0.2542 - categorical_accuracy: 0.9243
33280/60000 [===============>..............] - ETA: 47s - loss: 0.2542 - categorical_accuracy: 0.9242
33312/60000 [===============>..............] - ETA: 47s - loss: 0.2541 - categorical_accuracy: 0.9243
33344/60000 [===============>..............] - ETA: 47s - loss: 0.2539 - categorical_accuracy: 0.9243
33376/60000 [===============>..............] - ETA: 47s - loss: 0.2540 - categorical_accuracy: 0.9243
33408/60000 [===============>..............] - ETA: 47s - loss: 0.2538 - categorical_accuracy: 0.9243
33440/60000 [===============>..............] - ETA: 47s - loss: 0.2536 - categorical_accuracy: 0.9243
33472/60000 [===============>..............] - ETA: 47s - loss: 0.2534 - categorical_accuracy: 0.9244
33504/60000 [===============>..............] - ETA: 47s - loss: 0.2532 - categorical_accuracy: 0.9245
33536/60000 [===============>..............] - ETA: 47s - loss: 0.2532 - categorical_accuracy: 0.9245
33568/60000 [===============>..............] - ETA: 47s - loss: 0.2530 - categorical_accuracy: 0.9245
33600/60000 [===============>..............] - ETA: 46s - loss: 0.2529 - categorical_accuracy: 0.9245
33632/60000 [===============>..............] - ETA: 46s - loss: 0.2529 - categorical_accuracy: 0.9245
33664/60000 [===============>..............] - ETA: 46s - loss: 0.2527 - categorical_accuracy: 0.9246
33696/60000 [===============>..............] - ETA: 46s - loss: 0.2526 - categorical_accuracy: 0.9246
33728/60000 [===============>..............] - ETA: 46s - loss: 0.2527 - categorical_accuracy: 0.9246
33760/60000 [===============>..............] - ETA: 46s - loss: 0.2524 - categorical_accuracy: 0.9246
33792/60000 [===============>..............] - ETA: 46s - loss: 0.2523 - categorical_accuracy: 0.9247
33824/60000 [===============>..............] - ETA: 46s - loss: 0.2521 - categorical_accuracy: 0.9248
33856/60000 [===============>..............] - ETA: 46s - loss: 0.2520 - categorical_accuracy: 0.9248
33888/60000 [===============>..............] - ETA: 46s - loss: 0.2520 - categorical_accuracy: 0.9248
33920/60000 [===============>..............] - ETA: 46s - loss: 0.2520 - categorical_accuracy: 0.9248
33952/60000 [===============>..............] - ETA: 46s - loss: 0.2521 - categorical_accuracy: 0.9247
33984/60000 [===============>..............] - ETA: 46s - loss: 0.2521 - categorical_accuracy: 0.9248
34016/60000 [================>.............] - ETA: 46s - loss: 0.2520 - categorical_accuracy: 0.9248
34048/60000 [================>.............] - ETA: 46s - loss: 0.2519 - categorical_accuracy: 0.9249
34080/60000 [================>.............] - ETA: 46s - loss: 0.2517 - categorical_accuracy: 0.9249
34112/60000 [================>.............] - ETA: 46s - loss: 0.2516 - categorical_accuracy: 0.9250
34144/60000 [================>.............] - ETA: 45s - loss: 0.2513 - categorical_accuracy: 0.9251
34176/60000 [================>.............] - ETA: 45s - loss: 0.2513 - categorical_accuracy: 0.9251
34208/60000 [================>.............] - ETA: 45s - loss: 0.2511 - categorical_accuracy: 0.9251
34240/60000 [================>.............] - ETA: 45s - loss: 0.2510 - categorical_accuracy: 0.9251
34272/60000 [================>.............] - ETA: 45s - loss: 0.2508 - categorical_accuracy: 0.9252
34304/60000 [================>.............] - ETA: 45s - loss: 0.2509 - categorical_accuracy: 0.9252
34336/60000 [================>.............] - ETA: 45s - loss: 0.2509 - categorical_accuracy: 0.9252
34368/60000 [================>.............] - ETA: 45s - loss: 0.2507 - categorical_accuracy: 0.9253
34400/60000 [================>.............] - ETA: 45s - loss: 0.2506 - categorical_accuracy: 0.9253
34432/60000 [================>.............] - ETA: 45s - loss: 0.2506 - categorical_accuracy: 0.9253
34464/60000 [================>.............] - ETA: 45s - loss: 0.2504 - categorical_accuracy: 0.9253
34496/60000 [================>.............] - ETA: 45s - loss: 0.2504 - categorical_accuracy: 0.9254
34528/60000 [================>.............] - ETA: 45s - loss: 0.2503 - categorical_accuracy: 0.9254
34560/60000 [================>.............] - ETA: 45s - loss: 0.2502 - categorical_accuracy: 0.9254
34592/60000 [================>.............] - ETA: 45s - loss: 0.2501 - categorical_accuracy: 0.9254
34624/60000 [================>.............] - ETA: 45s - loss: 0.2500 - categorical_accuracy: 0.9254
34656/60000 [================>.............] - ETA: 45s - loss: 0.2498 - categorical_accuracy: 0.9255
34688/60000 [================>.............] - ETA: 45s - loss: 0.2497 - categorical_accuracy: 0.9255
34720/60000 [================>.............] - ETA: 44s - loss: 0.2497 - categorical_accuracy: 0.9255
34752/60000 [================>.............] - ETA: 44s - loss: 0.2496 - categorical_accuracy: 0.9256
34784/60000 [================>.............] - ETA: 44s - loss: 0.2494 - categorical_accuracy: 0.9256
34816/60000 [================>.............] - ETA: 44s - loss: 0.2492 - categorical_accuracy: 0.9257
34848/60000 [================>.............] - ETA: 44s - loss: 0.2490 - categorical_accuracy: 0.9258
34880/60000 [================>.............] - ETA: 44s - loss: 0.2488 - categorical_accuracy: 0.9258
34912/60000 [================>.............] - ETA: 44s - loss: 0.2486 - categorical_accuracy: 0.9259
34944/60000 [================>.............] - ETA: 44s - loss: 0.2484 - categorical_accuracy: 0.9260
34976/60000 [================>.............] - ETA: 44s - loss: 0.2483 - categorical_accuracy: 0.9260
35008/60000 [================>.............] - ETA: 44s - loss: 0.2483 - categorical_accuracy: 0.9260
35040/60000 [================>.............] - ETA: 44s - loss: 0.2481 - categorical_accuracy: 0.9261
35072/60000 [================>.............] - ETA: 44s - loss: 0.2480 - categorical_accuracy: 0.9261
35104/60000 [================>.............] - ETA: 44s - loss: 0.2479 - categorical_accuracy: 0.9261
35136/60000 [================>.............] - ETA: 44s - loss: 0.2478 - categorical_accuracy: 0.9261
35168/60000 [================>.............] - ETA: 44s - loss: 0.2476 - categorical_accuracy: 0.9262
35200/60000 [================>.............] - ETA: 44s - loss: 0.2474 - categorical_accuracy: 0.9263
35232/60000 [================>.............] - ETA: 44s - loss: 0.2473 - categorical_accuracy: 0.9263
35264/60000 [================>.............] - ETA: 43s - loss: 0.2472 - categorical_accuracy: 0.9264
35296/60000 [================>.............] - ETA: 43s - loss: 0.2470 - categorical_accuracy: 0.9264
35328/60000 [================>.............] - ETA: 43s - loss: 0.2468 - categorical_accuracy: 0.9265
35360/60000 [================>.............] - ETA: 43s - loss: 0.2467 - categorical_accuracy: 0.9265
35392/60000 [================>.............] - ETA: 43s - loss: 0.2467 - categorical_accuracy: 0.9265
35424/60000 [================>.............] - ETA: 43s - loss: 0.2465 - categorical_accuracy: 0.9266
35456/60000 [================>.............] - ETA: 43s - loss: 0.2464 - categorical_accuracy: 0.9266
35488/60000 [================>.............] - ETA: 43s - loss: 0.2463 - categorical_accuracy: 0.9266
35520/60000 [================>.............] - ETA: 43s - loss: 0.2462 - categorical_accuracy: 0.9266
35552/60000 [================>.............] - ETA: 43s - loss: 0.2460 - categorical_accuracy: 0.9267
35584/60000 [================>.............] - ETA: 43s - loss: 0.2459 - categorical_accuracy: 0.9267
35616/60000 [================>.............] - ETA: 43s - loss: 0.2457 - categorical_accuracy: 0.9267
35648/60000 [================>.............] - ETA: 43s - loss: 0.2455 - categorical_accuracy: 0.9268
35680/60000 [================>.............] - ETA: 43s - loss: 0.2454 - categorical_accuracy: 0.9268
35712/60000 [================>.............] - ETA: 43s - loss: 0.2455 - categorical_accuracy: 0.9268
35744/60000 [================>.............] - ETA: 43s - loss: 0.2455 - categorical_accuracy: 0.9268
35776/60000 [================>.............] - ETA: 43s - loss: 0.2453 - categorical_accuracy: 0.9269
35808/60000 [================>.............] - ETA: 43s - loss: 0.2452 - categorical_accuracy: 0.9269
35840/60000 [================>.............] - ETA: 42s - loss: 0.2450 - categorical_accuracy: 0.9269
35872/60000 [================>.............] - ETA: 42s - loss: 0.2448 - categorical_accuracy: 0.9270
35904/60000 [================>.............] - ETA: 42s - loss: 0.2447 - categorical_accuracy: 0.9271
35936/60000 [================>.............] - ETA: 42s - loss: 0.2448 - categorical_accuracy: 0.9271
35968/60000 [================>.............] - ETA: 42s - loss: 0.2446 - categorical_accuracy: 0.9271
36000/60000 [=================>............] - ETA: 42s - loss: 0.2444 - categorical_accuracy: 0.9272
36032/60000 [=================>............] - ETA: 42s - loss: 0.2442 - categorical_accuracy: 0.9272
36064/60000 [=================>............] - ETA: 42s - loss: 0.2441 - categorical_accuracy: 0.9273
36096/60000 [=================>............] - ETA: 42s - loss: 0.2439 - categorical_accuracy: 0.9273
36128/60000 [=================>............] - ETA: 42s - loss: 0.2437 - categorical_accuracy: 0.9274
36160/60000 [=================>............] - ETA: 42s - loss: 0.2435 - categorical_accuracy: 0.9275
36192/60000 [=================>............] - ETA: 42s - loss: 0.2435 - categorical_accuracy: 0.9275
36224/60000 [=================>............] - ETA: 42s - loss: 0.2435 - categorical_accuracy: 0.9275
36256/60000 [=================>............] - ETA: 42s - loss: 0.2433 - categorical_accuracy: 0.9276
36288/60000 [=================>............] - ETA: 42s - loss: 0.2433 - categorical_accuracy: 0.9276
36320/60000 [=================>............] - ETA: 42s - loss: 0.2435 - categorical_accuracy: 0.9276
36352/60000 [=================>............] - ETA: 42s - loss: 0.2436 - categorical_accuracy: 0.9276
36384/60000 [=================>............] - ETA: 42s - loss: 0.2435 - categorical_accuracy: 0.9276
36416/60000 [=================>............] - ETA: 41s - loss: 0.2434 - categorical_accuracy: 0.9276
36448/60000 [=================>............] - ETA: 41s - loss: 0.2433 - categorical_accuracy: 0.9277
36480/60000 [=================>............] - ETA: 41s - loss: 0.2433 - categorical_accuracy: 0.9277
36512/60000 [=================>............] - ETA: 41s - loss: 0.2433 - categorical_accuracy: 0.9277
36544/60000 [=================>............] - ETA: 41s - loss: 0.2432 - categorical_accuracy: 0.9277
36576/60000 [=================>............] - ETA: 41s - loss: 0.2432 - categorical_accuracy: 0.9277
36608/60000 [=================>............] - ETA: 41s - loss: 0.2430 - categorical_accuracy: 0.9278
36640/60000 [=================>............] - ETA: 41s - loss: 0.2428 - categorical_accuracy: 0.9278
36672/60000 [=================>............] - ETA: 41s - loss: 0.2426 - categorical_accuracy: 0.9279
36704/60000 [=================>............] - ETA: 41s - loss: 0.2424 - categorical_accuracy: 0.9279
36736/60000 [=================>............] - ETA: 41s - loss: 0.2424 - categorical_accuracy: 0.9279
36768/60000 [=================>............] - ETA: 41s - loss: 0.2424 - categorical_accuracy: 0.9279
36800/60000 [=================>............] - ETA: 41s - loss: 0.2424 - categorical_accuracy: 0.9279
36832/60000 [=================>............] - ETA: 41s - loss: 0.2423 - categorical_accuracy: 0.9280
36864/60000 [=================>............] - ETA: 41s - loss: 0.2421 - categorical_accuracy: 0.9280
36896/60000 [=================>............] - ETA: 41s - loss: 0.2420 - categorical_accuracy: 0.9280
36928/60000 [=================>............] - ETA: 41s - loss: 0.2418 - categorical_accuracy: 0.9281
36960/60000 [=================>............] - ETA: 40s - loss: 0.2417 - categorical_accuracy: 0.9281
36992/60000 [=================>............] - ETA: 40s - loss: 0.2415 - categorical_accuracy: 0.9282
37024/60000 [=================>............] - ETA: 40s - loss: 0.2416 - categorical_accuracy: 0.9282
37056/60000 [=================>............] - ETA: 40s - loss: 0.2415 - categorical_accuracy: 0.9282
37088/60000 [=================>............] - ETA: 40s - loss: 0.2413 - categorical_accuracy: 0.9283
37120/60000 [=================>............] - ETA: 40s - loss: 0.2415 - categorical_accuracy: 0.9282
37152/60000 [=================>............] - ETA: 40s - loss: 0.2414 - categorical_accuracy: 0.9282
37184/60000 [=================>............] - ETA: 40s - loss: 0.2412 - categorical_accuracy: 0.9283
37216/60000 [=================>............] - ETA: 40s - loss: 0.2411 - categorical_accuracy: 0.9283
37248/60000 [=================>............] - ETA: 40s - loss: 0.2410 - categorical_accuracy: 0.9283
37280/60000 [=================>............] - ETA: 40s - loss: 0.2408 - categorical_accuracy: 0.9284
37312/60000 [=================>............] - ETA: 40s - loss: 0.2406 - categorical_accuracy: 0.9284
37344/60000 [=================>............] - ETA: 40s - loss: 0.2405 - categorical_accuracy: 0.9285
37376/60000 [=================>............] - ETA: 40s - loss: 0.2405 - categorical_accuracy: 0.9285
37408/60000 [=================>............] - ETA: 40s - loss: 0.2403 - categorical_accuracy: 0.9285
37440/60000 [=================>............] - ETA: 40s - loss: 0.2404 - categorical_accuracy: 0.9286
37472/60000 [=================>............] - ETA: 40s - loss: 0.2403 - categorical_accuracy: 0.9285
37504/60000 [=================>............] - ETA: 40s - loss: 0.2404 - categorical_accuracy: 0.9285
37536/60000 [=================>............] - ETA: 39s - loss: 0.2403 - categorical_accuracy: 0.9286
37568/60000 [=================>............] - ETA: 39s - loss: 0.2403 - categorical_accuracy: 0.9286
37600/60000 [=================>............] - ETA: 39s - loss: 0.2401 - categorical_accuracy: 0.9286
37632/60000 [=================>............] - ETA: 39s - loss: 0.2400 - categorical_accuracy: 0.9287
37664/60000 [=================>............] - ETA: 39s - loss: 0.2398 - categorical_accuracy: 0.9287
37696/60000 [=================>............] - ETA: 39s - loss: 0.2398 - categorical_accuracy: 0.9287
37728/60000 [=================>............] - ETA: 39s - loss: 0.2396 - categorical_accuracy: 0.9288
37760/60000 [=================>............] - ETA: 39s - loss: 0.2394 - categorical_accuracy: 0.9288
37792/60000 [=================>............] - ETA: 39s - loss: 0.2393 - categorical_accuracy: 0.9289
37824/60000 [=================>............] - ETA: 39s - loss: 0.2392 - categorical_accuracy: 0.9289
37856/60000 [=================>............] - ETA: 39s - loss: 0.2391 - categorical_accuracy: 0.9289
37888/60000 [=================>............] - ETA: 39s - loss: 0.2390 - categorical_accuracy: 0.9289
37920/60000 [=================>............] - ETA: 39s - loss: 0.2391 - categorical_accuracy: 0.9289
37952/60000 [=================>............] - ETA: 39s - loss: 0.2389 - categorical_accuracy: 0.9290
37984/60000 [=================>............] - ETA: 39s - loss: 0.2388 - categorical_accuracy: 0.9290
38016/60000 [==================>...........] - ETA: 39s - loss: 0.2386 - categorical_accuracy: 0.9291
38048/60000 [==================>...........] - ETA: 39s - loss: 0.2386 - categorical_accuracy: 0.9291
38080/60000 [==================>...........] - ETA: 38s - loss: 0.2385 - categorical_accuracy: 0.9291
38112/60000 [==================>...........] - ETA: 38s - loss: 0.2384 - categorical_accuracy: 0.9291
38144/60000 [==================>...........] - ETA: 38s - loss: 0.2382 - categorical_accuracy: 0.9292
38176/60000 [==================>...........] - ETA: 38s - loss: 0.2381 - categorical_accuracy: 0.9292
38208/60000 [==================>...........] - ETA: 38s - loss: 0.2380 - categorical_accuracy: 0.9293
38240/60000 [==================>...........] - ETA: 38s - loss: 0.2378 - categorical_accuracy: 0.9293
38272/60000 [==================>...........] - ETA: 38s - loss: 0.2378 - categorical_accuracy: 0.9293
38304/60000 [==================>...........] - ETA: 38s - loss: 0.2376 - categorical_accuracy: 0.9294
38336/60000 [==================>...........] - ETA: 38s - loss: 0.2376 - categorical_accuracy: 0.9294
38368/60000 [==================>...........] - ETA: 38s - loss: 0.2374 - categorical_accuracy: 0.9294
38400/60000 [==================>...........] - ETA: 38s - loss: 0.2373 - categorical_accuracy: 0.9295
38432/60000 [==================>...........] - ETA: 38s - loss: 0.2371 - categorical_accuracy: 0.9295
38464/60000 [==================>...........] - ETA: 38s - loss: 0.2371 - categorical_accuracy: 0.9295
38496/60000 [==================>...........] - ETA: 38s - loss: 0.2370 - categorical_accuracy: 0.9295
38528/60000 [==================>...........] - ETA: 38s - loss: 0.2370 - categorical_accuracy: 0.9295
38560/60000 [==================>...........] - ETA: 38s - loss: 0.2371 - categorical_accuracy: 0.9294
38592/60000 [==================>...........] - ETA: 38s - loss: 0.2370 - categorical_accuracy: 0.9294
38624/60000 [==================>...........] - ETA: 38s - loss: 0.2369 - categorical_accuracy: 0.9294
38656/60000 [==================>...........] - ETA: 37s - loss: 0.2369 - categorical_accuracy: 0.9295
38688/60000 [==================>...........] - ETA: 37s - loss: 0.2367 - categorical_accuracy: 0.9295
38720/60000 [==================>...........] - ETA: 37s - loss: 0.2367 - categorical_accuracy: 0.9295
38752/60000 [==================>...........] - ETA: 37s - loss: 0.2368 - categorical_accuracy: 0.9295
38784/60000 [==================>...........] - ETA: 37s - loss: 0.2367 - categorical_accuracy: 0.9295
38816/60000 [==================>...........] - ETA: 37s - loss: 0.2366 - categorical_accuracy: 0.9295
38848/60000 [==================>...........] - ETA: 37s - loss: 0.2365 - categorical_accuracy: 0.9295
38880/60000 [==================>...........] - ETA: 37s - loss: 0.2364 - categorical_accuracy: 0.9296
38912/60000 [==================>...........] - ETA: 37s - loss: 0.2362 - categorical_accuracy: 0.9296
38944/60000 [==================>...........] - ETA: 37s - loss: 0.2363 - categorical_accuracy: 0.9296
38976/60000 [==================>...........] - ETA: 37s - loss: 0.2362 - categorical_accuracy: 0.9296
39008/60000 [==================>...........] - ETA: 37s - loss: 0.2362 - categorical_accuracy: 0.9296
39040/60000 [==================>...........] - ETA: 37s - loss: 0.2360 - categorical_accuracy: 0.9297
39072/60000 [==================>...........] - ETA: 37s - loss: 0.2358 - categorical_accuracy: 0.9297
39104/60000 [==================>...........] - ETA: 37s - loss: 0.2357 - categorical_accuracy: 0.9298
39136/60000 [==================>...........] - ETA: 37s - loss: 0.2358 - categorical_accuracy: 0.9297
39168/60000 [==================>...........] - ETA: 37s - loss: 0.2360 - categorical_accuracy: 0.9298
39200/60000 [==================>...........] - ETA: 36s - loss: 0.2358 - categorical_accuracy: 0.9298
39232/60000 [==================>...........] - ETA: 36s - loss: 0.2357 - categorical_accuracy: 0.9299
39264/60000 [==================>...........] - ETA: 36s - loss: 0.2358 - categorical_accuracy: 0.9299
39296/60000 [==================>...........] - ETA: 36s - loss: 0.2357 - categorical_accuracy: 0.9299
39328/60000 [==================>...........] - ETA: 36s - loss: 0.2356 - categorical_accuracy: 0.9299
39360/60000 [==================>...........] - ETA: 36s - loss: 0.2355 - categorical_accuracy: 0.9299
39392/60000 [==================>...........] - ETA: 36s - loss: 0.2355 - categorical_accuracy: 0.9300
39424/60000 [==================>...........] - ETA: 36s - loss: 0.2353 - categorical_accuracy: 0.9300
39456/60000 [==================>...........] - ETA: 36s - loss: 0.2352 - categorical_accuracy: 0.9300
39488/60000 [==================>...........] - ETA: 36s - loss: 0.2351 - categorical_accuracy: 0.9301
39520/60000 [==================>...........] - ETA: 36s - loss: 0.2352 - categorical_accuracy: 0.9300
39552/60000 [==================>...........] - ETA: 36s - loss: 0.2350 - categorical_accuracy: 0.9301
39584/60000 [==================>...........] - ETA: 36s - loss: 0.2349 - categorical_accuracy: 0.9301
39616/60000 [==================>...........] - ETA: 36s - loss: 0.2347 - categorical_accuracy: 0.9302
39648/60000 [==================>...........] - ETA: 36s - loss: 0.2346 - categorical_accuracy: 0.9302
39680/60000 [==================>...........] - ETA: 36s - loss: 0.2344 - categorical_accuracy: 0.9303
39712/60000 [==================>...........] - ETA: 36s - loss: 0.2343 - categorical_accuracy: 0.9303
39744/60000 [==================>...........] - ETA: 36s - loss: 0.2341 - categorical_accuracy: 0.9304
39776/60000 [==================>...........] - ETA: 35s - loss: 0.2339 - categorical_accuracy: 0.9304
39808/60000 [==================>...........] - ETA: 35s - loss: 0.2339 - categorical_accuracy: 0.9304
39840/60000 [==================>...........] - ETA: 35s - loss: 0.2338 - categorical_accuracy: 0.9305
39872/60000 [==================>...........] - ETA: 35s - loss: 0.2336 - categorical_accuracy: 0.9305
39904/60000 [==================>...........] - ETA: 35s - loss: 0.2335 - categorical_accuracy: 0.9306
39936/60000 [==================>...........] - ETA: 35s - loss: 0.2334 - categorical_accuracy: 0.9306
39968/60000 [==================>...........] - ETA: 35s - loss: 0.2332 - categorical_accuracy: 0.9307
40000/60000 [===================>..........] - ETA: 35s - loss: 0.2330 - categorical_accuracy: 0.9307
40032/60000 [===================>..........] - ETA: 35s - loss: 0.2329 - categorical_accuracy: 0.9308
40064/60000 [===================>..........] - ETA: 35s - loss: 0.2328 - categorical_accuracy: 0.9308
40096/60000 [===================>..........] - ETA: 35s - loss: 0.2327 - categorical_accuracy: 0.9308
40128/60000 [===================>..........] - ETA: 35s - loss: 0.2327 - categorical_accuracy: 0.9308
40160/60000 [===================>..........] - ETA: 35s - loss: 0.2326 - categorical_accuracy: 0.9308
40192/60000 [===================>..........] - ETA: 35s - loss: 0.2324 - categorical_accuracy: 0.9309
40224/60000 [===================>..........] - ETA: 35s - loss: 0.2322 - categorical_accuracy: 0.9309
40256/60000 [===================>..........] - ETA: 35s - loss: 0.2322 - categorical_accuracy: 0.9309
40288/60000 [===================>..........] - ETA: 35s - loss: 0.2320 - categorical_accuracy: 0.9309
40320/60000 [===================>..........] - ETA: 34s - loss: 0.2319 - categorical_accuracy: 0.9310
40352/60000 [===================>..........] - ETA: 34s - loss: 0.2317 - categorical_accuracy: 0.9310
40384/60000 [===================>..........] - ETA: 34s - loss: 0.2317 - categorical_accuracy: 0.9310
40416/60000 [===================>..........] - ETA: 34s - loss: 0.2316 - categorical_accuracy: 0.9310
40448/60000 [===================>..........] - ETA: 34s - loss: 0.2314 - categorical_accuracy: 0.9311
40480/60000 [===================>..........] - ETA: 34s - loss: 0.2313 - categorical_accuracy: 0.9311
40512/60000 [===================>..........] - ETA: 34s - loss: 0.2311 - categorical_accuracy: 0.9312
40544/60000 [===================>..........] - ETA: 34s - loss: 0.2311 - categorical_accuracy: 0.9312
40576/60000 [===================>..........] - ETA: 34s - loss: 0.2309 - categorical_accuracy: 0.9312
40608/60000 [===================>..........] - ETA: 34s - loss: 0.2309 - categorical_accuracy: 0.9312
40640/60000 [===================>..........] - ETA: 34s - loss: 0.2309 - categorical_accuracy: 0.9313
40672/60000 [===================>..........] - ETA: 34s - loss: 0.2308 - categorical_accuracy: 0.9313
40704/60000 [===================>..........] - ETA: 34s - loss: 0.2307 - categorical_accuracy: 0.9313
40736/60000 [===================>..........] - ETA: 34s - loss: 0.2306 - categorical_accuracy: 0.9313
40768/60000 [===================>..........] - ETA: 34s - loss: 0.2305 - categorical_accuracy: 0.9314
40800/60000 [===================>..........] - ETA: 34s - loss: 0.2304 - categorical_accuracy: 0.9314
40832/60000 [===================>..........] - ETA: 34s - loss: 0.2302 - categorical_accuracy: 0.9315
40864/60000 [===================>..........] - ETA: 34s - loss: 0.2301 - categorical_accuracy: 0.9315
40896/60000 [===================>..........] - ETA: 33s - loss: 0.2299 - categorical_accuracy: 0.9315
40928/60000 [===================>..........] - ETA: 33s - loss: 0.2299 - categorical_accuracy: 0.9315
40960/60000 [===================>..........] - ETA: 33s - loss: 0.2298 - categorical_accuracy: 0.9316
40992/60000 [===================>..........] - ETA: 33s - loss: 0.2298 - categorical_accuracy: 0.9315
41024/60000 [===================>..........] - ETA: 33s - loss: 0.2297 - categorical_accuracy: 0.9316
41056/60000 [===================>..........] - ETA: 33s - loss: 0.2296 - categorical_accuracy: 0.9316
41088/60000 [===================>..........] - ETA: 33s - loss: 0.2294 - categorical_accuracy: 0.9316
41120/60000 [===================>..........] - ETA: 33s - loss: 0.2293 - categorical_accuracy: 0.9317
41152/60000 [===================>..........] - ETA: 33s - loss: 0.2292 - categorical_accuracy: 0.9317
41184/60000 [===================>..........] - ETA: 33s - loss: 0.2291 - categorical_accuracy: 0.9317
41216/60000 [===================>..........] - ETA: 33s - loss: 0.2293 - categorical_accuracy: 0.9317
41248/60000 [===================>..........] - ETA: 33s - loss: 0.2291 - categorical_accuracy: 0.9318
41280/60000 [===================>..........] - ETA: 33s - loss: 0.2289 - categorical_accuracy: 0.9318
41312/60000 [===================>..........] - ETA: 33s - loss: 0.2288 - categorical_accuracy: 0.9319
41344/60000 [===================>..........] - ETA: 33s - loss: 0.2287 - categorical_accuracy: 0.9319
41376/60000 [===================>..........] - ETA: 33s - loss: 0.2286 - categorical_accuracy: 0.9319
41408/60000 [===================>..........] - ETA: 33s - loss: 0.2285 - categorical_accuracy: 0.9319
41440/60000 [===================>..........] - ETA: 32s - loss: 0.2284 - categorical_accuracy: 0.9319
41472/60000 [===================>..........] - ETA: 32s - loss: 0.2282 - categorical_accuracy: 0.9320
41504/60000 [===================>..........] - ETA: 32s - loss: 0.2282 - categorical_accuracy: 0.9320
41536/60000 [===================>..........] - ETA: 32s - loss: 0.2280 - categorical_accuracy: 0.9321
41568/60000 [===================>..........] - ETA: 32s - loss: 0.2282 - categorical_accuracy: 0.9320
41600/60000 [===================>..........] - ETA: 32s - loss: 0.2281 - categorical_accuracy: 0.9321
41632/60000 [===================>..........] - ETA: 32s - loss: 0.2279 - categorical_accuracy: 0.9321
41664/60000 [===================>..........] - ETA: 32s - loss: 0.2278 - categorical_accuracy: 0.9322
41696/60000 [===================>..........] - ETA: 32s - loss: 0.2277 - categorical_accuracy: 0.9322
41728/60000 [===================>..........] - ETA: 32s - loss: 0.2275 - categorical_accuracy: 0.9323
41760/60000 [===================>..........] - ETA: 32s - loss: 0.2275 - categorical_accuracy: 0.9323
41792/60000 [===================>..........] - ETA: 32s - loss: 0.2276 - categorical_accuracy: 0.9322
41824/60000 [===================>..........] - ETA: 32s - loss: 0.2275 - categorical_accuracy: 0.9322
41856/60000 [===================>..........] - ETA: 32s - loss: 0.2274 - categorical_accuracy: 0.9323
41888/60000 [===================>..........] - ETA: 32s - loss: 0.2275 - categorical_accuracy: 0.9323
41920/60000 [===================>..........] - ETA: 32s - loss: 0.2274 - categorical_accuracy: 0.9323
41952/60000 [===================>..........] - ETA: 32s - loss: 0.2274 - categorical_accuracy: 0.9323
41984/60000 [===================>..........] - ETA: 32s - loss: 0.2273 - categorical_accuracy: 0.9323
42016/60000 [====================>.........] - ETA: 31s - loss: 0.2272 - categorical_accuracy: 0.9324
42048/60000 [====================>.........] - ETA: 31s - loss: 0.2272 - categorical_accuracy: 0.9324
42080/60000 [====================>.........] - ETA: 31s - loss: 0.2270 - categorical_accuracy: 0.9324
42112/60000 [====================>.........] - ETA: 31s - loss: 0.2269 - categorical_accuracy: 0.9325
42144/60000 [====================>.........] - ETA: 31s - loss: 0.2268 - categorical_accuracy: 0.9325
42176/60000 [====================>.........] - ETA: 31s - loss: 0.2267 - categorical_accuracy: 0.9325
42208/60000 [====================>.........] - ETA: 31s - loss: 0.2265 - categorical_accuracy: 0.9325
42240/60000 [====================>.........] - ETA: 31s - loss: 0.2264 - categorical_accuracy: 0.9326
42272/60000 [====================>.........] - ETA: 31s - loss: 0.2263 - categorical_accuracy: 0.9326
42304/60000 [====================>.........] - ETA: 31s - loss: 0.2261 - categorical_accuracy: 0.9327
42336/60000 [====================>.........] - ETA: 31s - loss: 0.2261 - categorical_accuracy: 0.9327
42368/60000 [====================>.........] - ETA: 31s - loss: 0.2260 - categorical_accuracy: 0.9327
42400/60000 [====================>.........] - ETA: 31s - loss: 0.2259 - categorical_accuracy: 0.9328
42432/60000 [====================>.........] - ETA: 31s - loss: 0.2258 - categorical_accuracy: 0.9328
42464/60000 [====================>.........] - ETA: 31s - loss: 0.2257 - categorical_accuracy: 0.9328
42496/60000 [====================>.........] - ETA: 31s - loss: 0.2256 - categorical_accuracy: 0.9328
42528/60000 [====================>.........] - ETA: 31s - loss: 0.2255 - categorical_accuracy: 0.9329
42560/60000 [====================>.........] - ETA: 30s - loss: 0.2254 - categorical_accuracy: 0.9329
42592/60000 [====================>.........] - ETA: 30s - loss: 0.2253 - categorical_accuracy: 0.9329
42624/60000 [====================>.........] - ETA: 30s - loss: 0.2252 - categorical_accuracy: 0.9329
42656/60000 [====================>.........] - ETA: 30s - loss: 0.2252 - categorical_accuracy: 0.9329
42688/60000 [====================>.........] - ETA: 30s - loss: 0.2251 - categorical_accuracy: 0.9330
42720/60000 [====================>.........] - ETA: 30s - loss: 0.2250 - categorical_accuracy: 0.9330
42752/60000 [====================>.........] - ETA: 30s - loss: 0.2249 - categorical_accuracy: 0.9330
42784/60000 [====================>.........] - ETA: 30s - loss: 0.2247 - categorical_accuracy: 0.9331
42816/60000 [====================>.........] - ETA: 30s - loss: 0.2246 - categorical_accuracy: 0.9331
42848/60000 [====================>.........] - ETA: 30s - loss: 0.2245 - categorical_accuracy: 0.9331
42880/60000 [====================>.........] - ETA: 30s - loss: 0.2246 - categorical_accuracy: 0.9331
42912/60000 [====================>.........] - ETA: 30s - loss: 0.2245 - categorical_accuracy: 0.9331
42944/60000 [====================>.........] - ETA: 30s - loss: 0.2243 - categorical_accuracy: 0.9331
42976/60000 [====================>.........] - ETA: 30s - loss: 0.2242 - categorical_accuracy: 0.9332
43008/60000 [====================>.........] - ETA: 30s - loss: 0.2241 - categorical_accuracy: 0.9332
43040/60000 [====================>.........] - ETA: 30s - loss: 0.2239 - categorical_accuracy: 0.9333
43072/60000 [====================>.........] - ETA: 30s - loss: 0.2238 - categorical_accuracy: 0.9333
43104/60000 [====================>.........] - ETA: 30s - loss: 0.2238 - categorical_accuracy: 0.9333
43136/60000 [====================>.........] - ETA: 29s - loss: 0.2236 - categorical_accuracy: 0.9334
43168/60000 [====================>.........] - ETA: 29s - loss: 0.2236 - categorical_accuracy: 0.9334
43200/60000 [====================>.........] - ETA: 29s - loss: 0.2236 - categorical_accuracy: 0.9334
43232/60000 [====================>.........] - ETA: 29s - loss: 0.2234 - categorical_accuracy: 0.9335
43264/60000 [====================>.........] - ETA: 29s - loss: 0.2233 - categorical_accuracy: 0.9335
43296/60000 [====================>.........] - ETA: 29s - loss: 0.2232 - categorical_accuracy: 0.9336
43328/60000 [====================>.........] - ETA: 29s - loss: 0.2231 - categorical_accuracy: 0.9336
43360/60000 [====================>.........] - ETA: 29s - loss: 0.2229 - categorical_accuracy: 0.9336
43392/60000 [====================>.........] - ETA: 29s - loss: 0.2229 - categorical_accuracy: 0.9336
43424/60000 [====================>.........] - ETA: 29s - loss: 0.2228 - categorical_accuracy: 0.9337
43456/60000 [====================>.........] - ETA: 29s - loss: 0.2227 - categorical_accuracy: 0.9337
43488/60000 [====================>.........] - ETA: 29s - loss: 0.2226 - categorical_accuracy: 0.9337
43520/60000 [====================>.........] - ETA: 29s - loss: 0.2226 - categorical_accuracy: 0.9338
43552/60000 [====================>.........] - ETA: 29s - loss: 0.2224 - categorical_accuracy: 0.9338
43584/60000 [====================>.........] - ETA: 29s - loss: 0.2223 - categorical_accuracy: 0.9339
43616/60000 [====================>.........] - ETA: 29s - loss: 0.2221 - categorical_accuracy: 0.9339
43648/60000 [====================>.........] - ETA: 29s - loss: 0.2220 - categorical_accuracy: 0.9339
43680/60000 [====================>.........] - ETA: 29s - loss: 0.2219 - categorical_accuracy: 0.9340
43712/60000 [====================>.........] - ETA: 28s - loss: 0.2217 - categorical_accuracy: 0.9340
43744/60000 [====================>.........] - ETA: 28s - loss: 0.2217 - categorical_accuracy: 0.9340
43776/60000 [====================>.........] - ETA: 28s - loss: 0.2216 - categorical_accuracy: 0.9340
43808/60000 [====================>.........] - ETA: 28s - loss: 0.2216 - categorical_accuracy: 0.9340
43840/60000 [====================>.........] - ETA: 28s - loss: 0.2216 - categorical_accuracy: 0.9340
43872/60000 [====================>.........] - ETA: 28s - loss: 0.2214 - categorical_accuracy: 0.9340
43904/60000 [====================>.........] - ETA: 28s - loss: 0.2213 - categorical_accuracy: 0.9341
43936/60000 [====================>.........] - ETA: 28s - loss: 0.2212 - categorical_accuracy: 0.9341
43968/60000 [====================>.........] - ETA: 28s - loss: 0.2211 - categorical_accuracy: 0.9341
44000/60000 [=====================>........] - ETA: 28s - loss: 0.2210 - categorical_accuracy: 0.9342
44032/60000 [=====================>........] - ETA: 28s - loss: 0.2209 - categorical_accuracy: 0.9342
44064/60000 [=====================>........] - ETA: 28s - loss: 0.2208 - categorical_accuracy: 0.9342
44096/60000 [=====================>........] - ETA: 28s - loss: 0.2207 - categorical_accuracy: 0.9343
44128/60000 [=====================>........] - ETA: 28s - loss: 0.2205 - categorical_accuracy: 0.9343
44160/60000 [=====================>........] - ETA: 28s - loss: 0.2206 - categorical_accuracy: 0.9343
44192/60000 [=====================>........] - ETA: 28s - loss: 0.2204 - categorical_accuracy: 0.9343
44224/60000 [=====================>........] - ETA: 28s - loss: 0.2203 - categorical_accuracy: 0.9344
44256/60000 [=====================>........] - ETA: 27s - loss: 0.2205 - categorical_accuracy: 0.9344
44288/60000 [=====================>........] - ETA: 27s - loss: 0.2203 - categorical_accuracy: 0.9344
44320/60000 [=====================>........] - ETA: 27s - loss: 0.2202 - categorical_accuracy: 0.9344
44352/60000 [=====================>........] - ETA: 27s - loss: 0.2201 - categorical_accuracy: 0.9344
44384/60000 [=====================>........] - ETA: 27s - loss: 0.2200 - categorical_accuracy: 0.9345
44416/60000 [=====================>........] - ETA: 27s - loss: 0.2199 - categorical_accuracy: 0.9345
44448/60000 [=====================>........] - ETA: 27s - loss: 0.2198 - categorical_accuracy: 0.9345
44480/60000 [=====================>........] - ETA: 27s - loss: 0.2199 - categorical_accuracy: 0.9345
44512/60000 [=====================>........] - ETA: 27s - loss: 0.2199 - categorical_accuracy: 0.9345
44544/60000 [=====================>........] - ETA: 27s - loss: 0.2198 - categorical_accuracy: 0.9345
44576/60000 [=====================>........] - ETA: 27s - loss: 0.2197 - categorical_accuracy: 0.9345
44608/60000 [=====================>........] - ETA: 27s - loss: 0.2196 - categorical_accuracy: 0.9345
44640/60000 [=====================>........] - ETA: 27s - loss: 0.2196 - categorical_accuracy: 0.9346
44672/60000 [=====================>........] - ETA: 27s - loss: 0.2195 - categorical_accuracy: 0.9346
44704/60000 [=====================>........] - ETA: 27s - loss: 0.2193 - categorical_accuracy: 0.9346
44736/60000 [=====================>........] - ETA: 27s - loss: 0.2192 - categorical_accuracy: 0.9347
44768/60000 [=====================>........] - ETA: 27s - loss: 0.2194 - categorical_accuracy: 0.9347
44800/60000 [=====================>........] - ETA: 27s - loss: 0.2192 - categorical_accuracy: 0.9348
44832/60000 [=====================>........] - ETA: 26s - loss: 0.2191 - categorical_accuracy: 0.9348
44864/60000 [=====================>........] - ETA: 26s - loss: 0.2190 - categorical_accuracy: 0.9348
44896/60000 [=====================>........] - ETA: 26s - loss: 0.2189 - categorical_accuracy: 0.9348
44928/60000 [=====================>........] - ETA: 26s - loss: 0.2187 - categorical_accuracy: 0.9349
44960/60000 [=====================>........] - ETA: 26s - loss: 0.2187 - categorical_accuracy: 0.9349
44992/60000 [=====================>........] - ETA: 26s - loss: 0.2186 - categorical_accuracy: 0.9349
45024/60000 [=====================>........] - ETA: 26s - loss: 0.2187 - categorical_accuracy: 0.9349
45056/60000 [=====================>........] - ETA: 26s - loss: 0.2188 - categorical_accuracy: 0.9349
45088/60000 [=====================>........] - ETA: 26s - loss: 0.2187 - categorical_accuracy: 0.9349
45120/60000 [=====================>........] - ETA: 26s - loss: 0.2186 - categorical_accuracy: 0.9350
45152/60000 [=====================>........] - ETA: 26s - loss: 0.2185 - categorical_accuracy: 0.9350
45184/60000 [=====================>........] - ETA: 26s - loss: 0.2184 - categorical_accuracy: 0.9350
45216/60000 [=====================>........] - ETA: 26s - loss: 0.2184 - categorical_accuracy: 0.9351
45248/60000 [=====================>........] - ETA: 26s - loss: 0.2183 - categorical_accuracy: 0.9351
45280/60000 [=====================>........] - ETA: 26s - loss: 0.2182 - categorical_accuracy: 0.9351
45312/60000 [=====================>........] - ETA: 26s - loss: 0.2181 - categorical_accuracy: 0.9351
45344/60000 [=====================>........] - ETA: 26s - loss: 0.2180 - categorical_accuracy: 0.9352
45376/60000 [=====================>........] - ETA: 26s - loss: 0.2179 - categorical_accuracy: 0.9352
45408/60000 [=====================>........] - ETA: 25s - loss: 0.2179 - categorical_accuracy: 0.9352
45440/60000 [=====================>........] - ETA: 25s - loss: 0.2178 - categorical_accuracy: 0.9352
45472/60000 [=====================>........] - ETA: 25s - loss: 0.2177 - categorical_accuracy: 0.9353
45504/60000 [=====================>........] - ETA: 25s - loss: 0.2177 - categorical_accuracy: 0.9353
45536/60000 [=====================>........] - ETA: 25s - loss: 0.2176 - categorical_accuracy: 0.9353
45568/60000 [=====================>........] - ETA: 25s - loss: 0.2176 - categorical_accuracy: 0.9353
45600/60000 [=====================>........] - ETA: 25s - loss: 0.2174 - categorical_accuracy: 0.9353
45632/60000 [=====================>........] - ETA: 25s - loss: 0.2173 - categorical_accuracy: 0.9354
45664/60000 [=====================>........] - ETA: 25s - loss: 0.2172 - categorical_accuracy: 0.9354
45696/60000 [=====================>........] - ETA: 25s - loss: 0.2172 - categorical_accuracy: 0.9354
45728/60000 [=====================>........] - ETA: 25s - loss: 0.2170 - categorical_accuracy: 0.9354
45760/60000 [=====================>........] - ETA: 25s - loss: 0.2169 - categorical_accuracy: 0.9355
45792/60000 [=====================>........] - ETA: 25s - loss: 0.2168 - categorical_accuracy: 0.9355
45824/60000 [=====================>........] - ETA: 25s - loss: 0.2167 - categorical_accuracy: 0.9355
45856/60000 [=====================>........] - ETA: 25s - loss: 0.2166 - categorical_accuracy: 0.9356
45888/60000 [=====================>........] - ETA: 25s - loss: 0.2165 - categorical_accuracy: 0.9356
45920/60000 [=====================>........] - ETA: 25s - loss: 0.2164 - categorical_accuracy: 0.9356
45952/60000 [=====================>........] - ETA: 24s - loss: 0.2163 - categorical_accuracy: 0.9356
45984/60000 [=====================>........] - ETA: 24s - loss: 0.2162 - categorical_accuracy: 0.9357
46016/60000 [======================>.......] - ETA: 24s - loss: 0.2162 - categorical_accuracy: 0.9357
46048/60000 [======================>.......] - ETA: 24s - loss: 0.2161 - categorical_accuracy: 0.9357
46080/60000 [======================>.......] - ETA: 24s - loss: 0.2160 - categorical_accuracy: 0.9357
46112/60000 [======================>.......] - ETA: 24s - loss: 0.2159 - categorical_accuracy: 0.9357
46144/60000 [======================>.......] - ETA: 24s - loss: 0.2158 - categorical_accuracy: 0.9358
46176/60000 [======================>.......] - ETA: 24s - loss: 0.2158 - categorical_accuracy: 0.9358
46208/60000 [======================>.......] - ETA: 24s - loss: 0.2157 - categorical_accuracy: 0.9358
46240/60000 [======================>.......] - ETA: 24s - loss: 0.2157 - categorical_accuracy: 0.9359
46272/60000 [======================>.......] - ETA: 24s - loss: 0.2157 - categorical_accuracy: 0.9359
46304/60000 [======================>.......] - ETA: 24s - loss: 0.2157 - categorical_accuracy: 0.9359
46336/60000 [======================>.......] - ETA: 24s - loss: 0.2157 - categorical_accuracy: 0.9359
46368/60000 [======================>.......] - ETA: 24s - loss: 0.2156 - categorical_accuracy: 0.9359
46400/60000 [======================>.......] - ETA: 24s - loss: 0.2156 - categorical_accuracy: 0.9359
46432/60000 [======================>.......] - ETA: 24s - loss: 0.2155 - categorical_accuracy: 0.9359
46464/60000 [======================>.......] - ETA: 24s - loss: 0.2154 - categorical_accuracy: 0.9359
46496/60000 [======================>.......] - ETA: 24s - loss: 0.2153 - categorical_accuracy: 0.9360
46528/60000 [======================>.......] - ETA: 23s - loss: 0.2152 - categorical_accuracy: 0.9360
46560/60000 [======================>.......] - ETA: 23s - loss: 0.2151 - categorical_accuracy: 0.9361
46592/60000 [======================>.......] - ETA: 23s - loss: 0.2150 - categorical_accuracy: 0.9361
46624/60000 [======================>.......] - ETA: 23s - loss: 0.2149 - categorical_accuracy: 0.9361
46656/60000 [======================>.......] - ETA: 23s - loss: 0.2149 - categorical_accuracy: 0.9361
46688/60000 [======================>.......] - ETA: 23s - loss: 0.2147 - categorical_accuracy: 0.9362
46720/60000 [======================>.......] - ETA: 23s - loss: 0.2146 - categorical_accuracy: 0.9362
46752/60000 [======================>.......] - ETA: 23s - loss: 0.2145 - categorical_accuracy: 0.9362
46784/60000 [======================>.......] - ETA: 23s - loss: 0.2144 - categorical_accuracy: 0.9362
46816/60000 [======================>.......] - ETA: 23s - loss: 0.2144 - categorical_accuracy: 0.9363
46848/60000 [======================>.......] - ETA: 23s - loss: 0.2142 - categorical_accuracy: 0.9363
46880/60000 [======================>.......] - ETA: 23s - loss: 0.2142 - categorical_accuracy: 0.9363
46912/60000 [======================>.......] - ETA: 23s - loss: 0.2141 - categorical_accuracy: 0.9363
46944/60000 [======================>.......] - ETA: 23s - loss: 0.2140 - categorical_accuracy: 0.9364
46976/60000 [======================>.......] - ETA: 23s - loss: 0.2139 - categorical_accuracy: 0.9364
47008/60000 [======================>.......] - ETA: 23s - loss: 0.2138 - categorical_accuracy: 0.9364
47040/60000 [======================>.......] - ETA: 23s - loss: 0.2139 - categorical_accuracy: 0.9364
47072/60000 [======================>.......] - ETA: 22s - loss: 0.2139 - categorical_accuracy: 0.9365
47104/60000 [======================>.......] - ETA: 22s - loss: 0.2139 - categorical_accuracy: 0.9364
47136/60000 [======================>.......] - ETA: 22s - loss: 0.2137 - categorical_accuracy: 0.9365
47168/60000 [======================>.......] - ETA: 22s - loss: 0.2136 - categorical_accuracy: 0.9365
47200/60000 [======================>.......] - ETA: 22s - loss: 0.2135 - categorical_accuracy: 0.9366
47232/60000 [======================>.......] - ETA: 22s - loss: 0.2134 - categorical_accuracy: 0.9366
47264/60000 [======================>.......] - ETA: 22s - loss: 0.2133 - categorical_accuracy: 0.9366
47296/60000 [======================>.......] - ETA: 22s - loss: 0.2132 - categorical_accuracy: 0.9366
47328/60000 [======================>.......] - ETA: 22s - loss: 0.2131 - categorical_accuracy: 0.9367
47360/60000 [======================>.......] - ETA: 22s - loss: 0.2129 - categorical_accuracy: 0.9367
47392/60000 [======================>.......] - ETA: 22s - loss: 0.2128 - categorical_accuracy: 0.9367
47424/60000 [======================>.......] - ETA: 22s - loss: 0.2128 - categorical_accuracy: 0.9368
47456/60000 [======================>.......] - ETA: 22s - loss: 0.2127 - categorical_accuracy: 0.9368
47488/60000 [======================>.......] - ETA: 22s - loss: 0.2128 - categorical_accuracy: 0.9367
47520/60000 [======================>.......] - ETA: 22s - loss: 0.2127 - categorical_accuracy: 0.9367
47552/60000 [======================>.......] - ETA: 22s - loss: 0.2127 - categorical_accuracy: 0.9367
47584/60000 [======================>.......] - ETA: 22s - loss: 0.2126 - categorical_accuracy: 0.9367
47616/60000 [======================>.......] - ETA: 22s - loss: 0.2126 - categorical_accuracy: 0.9368
47648/60000 [======================>.......] - ETA: 21s - loss: 0.2125 - categorical_accuracy: 0.9368
47680/60000 [======================>.......] - ETA: 21s - loss: 0.2123 - categorical_accuracy: 0.9368
47712/60000 [======================>.......] - ETA: 21s - loss: 0.2123 - categorical_accuracy: 0.9369
47744/60000 [======================>.......] - ETA: 21s - loss: 0.2121 - categorical_accuracy: 0.9369
47776/60000 [======================>.......] - ETA: 21s - loss: 0.2121 - categorical_accuracy: 0.9369
47808/60000 [======================>.......] - ETA: 21s - loss: 0.2119 - categorical_accuracy: 0.9370
47840/60000 [======================>.......] - ETA: 21s - loss: 0.2119 - categorical_accuracy: 0.9370
47872/60000 [======================>.......] - ETA: 21s - loss: 0.2117 - categorical_accuracy: 0.9370
47904/60000 [======================>.......] - ETA: 21s - loss: 0.2118 - categorical_accuracy: 0.9370
47936/60000 [======================>.......] - ETA: 21s - loss: 0.2117 - categorical_accuracy: 0.9370
47968/60000 [======================>.......] - ETA: 21s - loss: 0.2116 - categorical_accuracy: 0.9371
48000/60000 [=======================>......] - ETA: 21s - loss: 0.2115 - categorical_accuracy: 0.9371
48032/60000 [=======================>......] - ETA: 21s - loss: 0.2114 - categorical_accuracy: 0.9371
48064/60000 [=======================>......] - ETA: 21s - loss: 0.2112 - categorical_accuracy: 0.9372
48096/60000 [=======================>......] - ETA: 21s - loss: 0.2112 - categorical_accuracy: 0.9372
48128/60000 [=======================>......] - ETA: 21s - loss: 0.2111 - categorical_accuracy: 0.9372
48160/60000 [=======================>......] - ETA: 21s - loss: 0.2109 - categorical_accuracy: 0.9373
48192/60000 [=======================>......] - ETA: 20s - loss: 0.2108 - categorical_accuracy: 0.9373
48224/60000 [=======================>......] - ETA: 20s - loss: 0.2107 - categorical_accuracy: 0.9374
48256/60000 [=======================>......] - ETA: 20s - loss: 0.2105 - categorical_accuracy: 0.9374
48288/60000 [=======================>......] - ETA: 20s - loss: 0.2106 - categorical_accuracy: 0.9374
48320/60000 [=======================>......] - ETA: 20s - loss: 0.2105 - categorical_accuracy: 0.9374
48352/60000 [=======================>......] - ETA: 20s - loss: 0.2104 - categorical_accuracy: 0.9374
48384/60000 [=======================>......] - ETA: 20s - loss: 0.2103 - categorical_accuracy: 0.9375
48416/60000 [=======================>......] - ETA: 20s - loss: 0.2102 - categorical_accuracy: 0.9375
48448/60000 [=======================>......] - ETA: 20s - loss: 0.2101 - categorical_accuracy: 0.9375
48480/60000 [=======================>......] - ETA: 20s - loss: 0.2100 - categorical_accuracy: 0.9375
48512/60000 [=======================>......] - ETA: 20s - loss: 0.2099 - categorical_accuracy: 0.9376
48544/60000 [=======================>......] - ETA: 20s - loss: 0.2098 - categorical_accuracy: 0.9376
48576/60000 [=======================>......] - ETA: 20s - loss: 0.2097 - categorical_accuracy: 0.9376
48608/60000 [=======================>......] - ETA: 20s - loss: 0.2097 - categorical_accuracy: 0.9376
48640/60000 [=======================>......] - ETA: 20s - loss: 0.2096 - categorical_accuracy: 0.9377
48672/60000 [=======================>......] - ETA: 20s - loss: 0.2097 - categorical_accuracy: 0.9376
48704/60000 [=======================>......] - ETA: 20s - loss: 0.2095 - categorical_accuracy: 0.9377
48736/60000 [=======================>......] - ETA: 20s - loss: 0.2097 - categorical_accuracy: 0.9376
48768/60000 [=======================>......] - ETA: 19s - loss: 0.2095 - categorical_accuracy: 0.9377
48800/60000 [=======================>......] - ETA: 19s - loss: 0.2094 - categorical_accuracy: 0.9377
48832/60000 [=======================>......] - ETA: 19s - loss: 0.2094 - categorical_accuracy: 0.9377
48864/60000 [=======================>......] - ETA: 19s - loss: 0.2093 - categorical_accuracy: 0.9377
48896/60000 [=======================>......] - ETA: 19s - loss: 0.2092 - categorical_accuracy: 0.9378
48928/60000 [=======================>......] - ETA: 19s - loss: 0.2091 - categorical_accuracy: 0.9378
48960/60000 [=======================>......] - ETA: 19s - loss: 0.2090 - categorical_accuracy: 0.9378
48992/60000 [=======================>......] - ETA: 19s - loss: 0.2089 - categorical_accuracy: 0.9378
49024/60000 [=======================>......] - ETA: 19s - loss: 0.2088 - categorical_accuracy: 0.9379
49056/60000 [=======================>......] - ETA: 19s - loss: 0.2087 - categorical_accuracy: 0.9379
49088/60000 [=======================>......] - ETA: 19s - loss: 0.2086 - categorical_accuracy: 0.9379
49120/60000 [=======================>......] - ETA: 19s - loss: 0.2085 - categorical_accuracy: 0.9379
49152/60000 [=======================>......] - ETA: 19s - loss: 0.2083 - categorical_accuracy: 0.9380
49184/60000 [=======================>......] - ETA: 19s - loss: 0.2083 - categorical_accuracy: 0.9380
49216/60000 [=======================>......] - ETA: 19s - loss: 0.2081 - categorical_accuracy: 0.9380
49248/60000 [=======================>......] - ETA: 19s - loss: 0.2080 - categorical_accuracy: 0.9381
49280/60000 [=======================>......] - ETA: 19s - loss: 0.2080 - categorical_accuracy: 0.9381
49312/60000 [=======================>......] - ETA: 18s - loss: 0.2079 - categorical_accuracy: 0.9381
49344/60000 [=======================>......] - ETA: 18s - loss: 0.2080 - categorical_accuracy: 0.9381
49376/60000 [=======================>......] - ETA: 18s - loss: 0.2079 - categorical_accuracy: 0.9381
49408/60000 [=======================>......] - ETA: 18s - loss: 0.2079 - categorical_accuracy: 0.9381
49440/60000 [=======================>......] - ETA: 18s - loss: 0.2078 - categorical_accuracy: 0.9382
49472/60000 [=======================>......] - ETA: 18s - loss: 0.2077 - categorical_accuracy: 0.9382
49504/60000 [=======================>......] - ETA: 18s - loss: 0.2076 - categorical_accuracy: 0.9382
49536/60000 [=======================>......] - ETA: 18s - loss: 0.2075 - categorical_accuracy: 0.9383
49568/60000 [=======================>......] - ETA: 18s - loss: 0.2073 - categorical_accuracy: 0.9383
49600/60000 [=======================>......] - ETA: 18s - loss: 0.2073 - categorical_accuracy: 0.9383
49632/60000 [=======================>......] - ETA: 18s - loss: 0.2071 - categorical_accuracy: 0.9384
49664/60000 [=======================>......] - ETA: 18s - loss: 0.2070 - categorical_accuracy: 0.9384
49696/60000 [=======================>......] - ETA: 18s - loss: 0.2069 - categorical_accuracy: 0.9384
49728/60000 [=======================>......] - ETA: 18s - loss: 0.2068 - categorical_accuracy: 0.9385
49760/60000 [=======================>......] - ETA: 18s - loss: 0.2067 - categorical_accuracy: 0.9385
49792/60000 [=======================>......] - ETA: 18s - loss: 0.2066 - categorical_accuracy: 0.9385
49824/60000 [=======================>......] - ETA: 18s - loss: 0.2066 - categorical_accuracy: 0.9385
49856/60000 [=======================>......] - ETA: 18s - loss: 0.2064 - categorical_accuracy: 0.9386
49888/60000 [=======================>......] - ETA: 17s - loss: 0.2063 - categorical_accuracy: 0.9386
49920/60000 [=======================>......] - ETA: 17s - loss: 0.2062 - categorical_accuracy: 0.9387
49952/60000 [=======================>......] - ETA: 17s - loss: 0.2061 - categorical_accuracy: 0.9387
49984/60000 [=======================>......] - ETA: 17s - loss: 0.2060 - categorical_accuracy: 0.9387
50016/60000 [========================>.....] - ETA: 17s - loss: 0.2059 - categorical_accuracy: 0.9387
50048/60000 [========================>.....] - ETA: 17s - loss: 0.2058 - categorical_accuracy: 0.9388
50080/60000 [========================>.....] - ETA: 17s - loss: 0.2058 - categorical_accuracy: 0.9388
50112/60000 [========================>.....] - ETA: 17s - loss: 0.2056 - categorical_accuracy: 0.9388
50144/60000 [========================>.....] - ETA: 17s - loss: 0.2056 - categorical_accuracy: 0.9388
50176/60000 [========================>.....] - ETA: 17s - loss: 0.2055 - categorical_accuracy: 0.9389
50208/60000 [========================>.....] - ETA: 17s - loss: 0.2054 - categorical_accuracy: 0.9389
50240/60000 [========================>.....] - ETA: 17s - loss: 0.2055 - categorical_accuracy: 0.9389
50272/60000 [========================>.....] - ETA: 17s - loss: 0.2054 - categorical_accuracy: 0.9389
50304/60000 [========================>.....] - ETA: 17s - loss: 0.2053 - categorical_accuracy: 0.9389
50336/60000 [========================>.....] - ETA: 17s - loss: 0.2053 - categorical_accuracy: 0.9389
50368/60000 [========================>.....] - ETA: 17s - loss: 0.2053 - categorical_accuracy: 0.9389
50400/60000 [========================>.....] - ETA: 17s - loss: 0.2052 - categorical_accuracy: 0.9389
50432/60000 [========================>.....] - ETA: 16s - loss: 0.2051 - categorical_accuracy: 0.9390
50464/60000 [========================>.....] - ETA: 16s - loss: 0.2050 - categorical_accuracy: 0.9390
50496/60000 [========================>.....] - ETA: 16s - loss: 0.2050 - categorical_accuracy: 0.9390
50528/60000 [========================>.....] - ETA: 16s - loss: 0.2050 - categorical_accuracy: 0.9391
50560/60000 [========================>.....] - ETA: 16s - loss: 0.2048 - categorical_accuracy: 0.9391
50592/60000 [========================>.....] - ETA: 16s - loss: 0.2047 - categorical_accuracy: 0.9391
50624/60000 [========================>.....] - ETA: 16s - loss: 0.2046 - categorical_accuracy: 0.9392
50656/60000 [========================>.....] - ETA: 16s - loss: 0.2045 - categorical_accuracy: 0.9392
50688/60000 [========================>.....] - ETA: 16s - loss: 0.2045 - categorical_accuracy: 0.9392
50720/60000 [========================>.....] - ETA: 16s - loss: 0.2045 - categorical_accuracy: 0.9392
50752/60000 [========================>.....] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9392
50784/60000 [========================>.....] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9392
50816/60000 [========================>.....] - ETA: 16s - loss: 0.2043 - categorical_accuracy: 0.9393
50848/60000 [========================>.....] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9393
50880/60000 [========================>.....] - ETA: 16s - loss: 0.2043 - categorical_accuracy: 0.9393
50912/60000 [========================>.....] - ETA: 16s - loss: 0.2042 - categorical_accuracy: 0.9393
50944/60000 [========================>.....] - ETA: 16s - loss: 0.2041 - categorical_accuracy: 0.9394
50976/60000 [========================>.....] - ETA: 16s - loss: 0.2040 - categorical_accuracy: 0.9394
51008/60000 [========================>.....] - ETA: 15s - loss: 0.2041 - categorical_accuracy: 0.9394
51040/60000 [========================>.....] - ETA: 15s - loss: 0.2040 - categorical_accuracy: 0.9394
51072/60000 [========================>.....] - ETA: 15s - loss: 0.2039 - categorical_accuracy: 0.9394
51104/60000 [========================>.....] - ETA: 15s - loss: 0.2039 - categorical_accuracy: 0.9395
51136/60000 [========================>.....] - ETA: 15s - loss: 0.2038 - categorical_accuracy: 0.9395
51168/60000 [========================>.....] - ETA: 15s - loss: 0.2038 - categorical_accuracy: 0.9395
51200/60000 [========================>.....] - ETA: 15s - loss: 0.2038 - categorical_accuracy: 0.9395
51232/60000 [========================>.....] - ETA: 15s - loss: 0.2038 - categorical_accuracy: 0.9395
51264/60000 [========================>.....] - ETA: 15s - loss: 0.2037 - categorical_accuracy: 0.9395
51296/60000 [========================>.....] - ETA: 15s - loss: 0.2037 - categorical_accuracy: 0.9395
51328/60000 [========================>.....] - ETA: 15s - loss: 0.2035 - categorical_accuracy: 0.9395
51360/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9396
51392/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9396
51424/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9396
51456/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9396
51488/60000 [========================>.....] - ETA: 15s - loss: 0.2033 - categorical_accuracy: 0.9397
51520/60000 [========================>.....] - ETA: 15s - loss: 0.2032 - categorical_accuracy: 0.9397
51552/60000 [========================>.....] - ETA: 15s - loss: 0.2031 - categorical_accuracy: 0.9397
51584/60000 [========================>.....] - ETA: 14s - loss: 0.2032 - categorical_accuracy: 0.9397
51616/60000 [========================>.....] - ETA: 14s - loss: 0.2031 - categorical_accuracy: 0.9397
51648/60000 [========================>.....] - ETA: 14s - loss: 0.2030 - categorical_accuracy: 0.9397
51680/60000 [========================>.....] - ETA: 14s - loss: 0.2029 - categorical_accuracy: 0.9398
51712/60000 [========================>.....] - ETA: 14s - loss: 0.2028 - categorical_accuracy: 0.9398
51744/60000 [========================>.....] - ETA: 14s - loss: 0.2027 - categorical_accuracy: 0.9398
51776/60000 [========================>.....] - ETA: 14s - loss: 0.2029 - categorical_accuracy: 0.9398
51808/60000 [========================>.....] - ETA: 14s - loss: 0.2028 - categorical_accuracy: 0.9398
51840/60000 [========================>.....] - ETA: 14s - loss: 0.2027 - categorical_accuracy: 0.9399
51872/60000 [========================>.....] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9399
51904/60000 [========================>.....] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9399
51936/60000 [========================>.....] - ETA: 14s - loss: 0.2025 - categorical_accuracy: 0.9399
51968/60000 [========================>.....] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9399
52000/60000 [=========================>....] - ETA: 14s - loss: 0.2025 - categorical_accuracy: 0.9399
52032/60000 [=========================>....] - ETA: 14s - loss: 0.2024 - categorical_accuracy: 0.9400
52064/60000 [=========================>....] - ETA: 14s - loss: 0.2024 - categorical_accuracy: 0.9400
52096/60000 [=========================>....] - ETA: 14s - loss: 0.2023 - categorical_accuracy: 0.9400
52128/60000 [=========================>....] - ETA: 13s - loss: 0.2023 - categorical_accuracy: 0.9400
52160/60000 [=========================>....] - ETA: 13s - loss: 0.2022 - categorical_accuracy: 0.9400
52192/60000 [=========================>....] - ETA: 13s - loss: 0.2021 - categorical_accuracy: 0.9400
52224/60000 [=========================>....] - ETA: 13s - loss: 0.2020 - categorical_accuracy: 0.9400
52256/60000 [=========================>....] - ETA: 13s - loss: 0.2019 - categorical_accuracy: 0.9401
52288/60000 [=========================>....] - ETA: 13s - loss: 0.2019 - categorical_accuracy: 0.9401
52320/60000 [=========================>....] - ETA: 13s - loss: 0.2018 - categorical_accuracy: 0.9401
52352/60000 [=========================>....] - ETA: 13s - loss: 0.2017 - categorical_accuracy: 0.9401
52384/60000 [=========================>....] - ETA: 13s - loss: 0.2017 - categorical_accuracy: 0.9402
52416/60000 [=========================>....] - ETA: 13s - loss: 0.2016 - categorical_accuracy: 0.9402
52448/60000 [=========================>....] - ETA: 13s - loss: 0.2015 - categorical_accuracy: 0.9402
52480/60000 [=========================>....] - ETA: 13s - loss: 0.2015 - categorical_accuracy: 0.9402
52512/60000 [=========================>....] - ETA: 13s - loss: 0.2014 - categorical_accuracy: 0.9402
52544/60000 [=========================>....] - ETA: 13s - loss: 0.2013 - categorical_accuracy: 0.9403
52576/60000 [=========================>....] - ETA: 13s - loss: 0.2012 - categorical_accuracy: 0.9403
52608/60000 [=========================>....] - ETA: 13s - loss: 0.2011 - categorical_accuracy: 0.9403
52640/60000 [=========================>....] - ETA: 13s - loss: 0.2010 - categorical_accuracy: 0.9404
52672/60000 [=========================>....] - ETA: 13s - loss: 0.2009 - categorical_accuracy: 0.9404
52704/60000 [=========================>....] - ETA: 12s - loss: 0.2008 - categorical_accuracy: 0.9404
52736/60000 [=========================>....] - ETA: 12s - loss: 0.2007 - categorical_accuracy: 0.9404
52768/60000 [=========================>....] - ETA: 12s - loss: 0.2007 - categorical_accuracy: 0.9404
52800/60000 [=========================>....] - ETA: 12s - loss: 0.2006 - categorical_accuracy: 0.9405
52832/60000 [=========================>....] - ETA: 12s - loss: 0.2005 - categorical_accuracy: 0.9405
52864/60000 [=========================>....] - ETA: 12s - loss: 0.2004 - categorical_accuracy: 0.9405
52896/60000 [=========================>....] - ETA: 12s - loss: 0.2003 - categorical_accuracy: 0.9405
52928/60000 [=========================>....] - ETA: 12s - loss: 0.2002 - categorical_accuracy: 0.9406
52960/60000 [=========================>....] - ETA: 12s - loss: 0.2002 - categorical_accuracy: 0.9406
52992/60000 [=========================>....] - ETA: 12s - loss: 0.2000 - categorical_accuracy: 0.9406
53024/60000 [=========================>....] - ETA: 12s - loss: 0.1999 - categorical_accuracy: 0.9406
53056/60000 [=========================>....] - ETA: 12s - loss: 0.1998 - categorical_accuracy: 0.9407
53088/60000 [=========================>....] - ETA: 12s - loss: 0.1997 - categorical_accuracy: 0.9407
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1998 - categorical_accuracy: 0.9407
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1997 - categorical_accuracy: 0.9407
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1996 - categorical_accuracy: 0.9408
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1995 - categorical_accuracy: 0.9408
53248/60000 [=========================>....] - ETA: 11s - loss: 0.1995 - categorical_accuracy: 0.9408
53280/60000 [=========================>....] - ETA: 11s - loss: 0.1994 - categorical_accuracy: 0.9408
53312/60000 [=========================>....] - ETA: 11s - loss: 0.1994 - categorical_accuracy: 0.9408
53344/60000 [=========================>....] - ETA: 11s - loss: 0.1994 - categorical_accuracy: 0.9408
53376/60000 [=========================>....] - ETA: 11s - loss: 0.1994 - categorical_accuracy: 0.9408
53408/60000 [=========================>....] - ETA: 11s - loss: 0.1994 - categorical_accuracy: 0.9408
53440/60000 [=========================>....] - ETA: 11s - loss: 0.1994 - categorical_accuracy: 0.9408
53472/60000 [=========================>....] - ETA: 11s - loss: 0.1992 - categorical_accuracy: 0.9409
53504/60000 [=========================>....] - ETA: 11s - loss: 0.1991 - categorical_accuracy: 0.9409
53536/60000 [=========================>....] - ETA: 11s - loss: 0.1990 - categorical_accuracy: 0.9409
53568/60000 [=========================>....] - ETA: 11s - loss: 0.1990 - categorical_accuracy: 0.9410
53600/60000 [=========================>....] - ETA: 11s - loss: 0.1989 - categorical_accuracy: 0.9410
53632/60000 [=========================>....] - ETA: 11s - loss: 0.1988 - categorical_accuracy: 0.9410
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1987 - categorical_accuracy: 0.9410
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1985 - categorical_accuracy: 0.9411
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1985 - categorical_accuracy: 0.9411
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1985 - categorical_accuracy: 0.9411
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1985 - categorical_accuracy: 0.9411
53824/60000 [=========================>....] - ETA: 10s - loss: 0.1983 - categorical_accuracy: 0.9411
53856/60000 [=========================>....] - ETA: 10s - loss: 0.1982 - categorical_accuracy: 0.9411
53888/60000 [=========================>....] - ETA: 10s - loss: 0.1983 - categorical_accuracy: 0.9412
53920/60000 [=========================>....] - ETA: 10s - loss: 0.1982 - categorical_accuracy: 0.9412
53952/60000 [=========================>....] - ETA: 10s - loss: 0.1981 - categorical_accuracy: 0.9412
53984/60000 [=========================>....] - ETA: 10s - loss: 0.1980 - categorical_accuracy: 0.9412
54016/60000 [==========================>...] - ETA: 10s - loss: 0.1979 - categorical_accuracy: 0.9413
54048/60000 [==========================>...] - ETA: 10s - loss: 0.1980 - categorical_accuracy: 0.9413
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1979 - categorical_accuracy: 0.9413
54112/60000 [==========================>...] - ETA: 10s - loss: 0.1979 - categorical_accuracy: 0.9413
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1978 - categorical_accuracy: 0.9413
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1979 - categorical_accuracy: 0.9413
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1979 - categorical_accuracy: 0.9413
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1979 - categorical_accuracy: 0.9413
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1978 - categorical_accuracy: 0.9414
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1977 - categorical_accuracy: 0.9414
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1976 - categorical_accuracy: 0.9414
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1975 - categorical_accuracy: 0.9415
54400/60000 [==========================>...] - ETA: 9s - loss: 0.1974 - categorical_accuracy: 0.9415 
54432/60000 [==========================>...] - ETA: 9s - loss: 0.1973 - categorical_accuracy: 0.9415
54464/60000 [==========================>...] - ETA: 9s - loss: 0.1974 - categorical_accuracy: 0.9415
54496/60000 [==========================>...] - ETA: 9s - loss: 0.1973 - categorical_accuracy: 0.9415
54528/60000 [==========================>...] - ETA: 9s - loss: 0.1972 - categorical_accuracy: 0.9416
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1971 - categorical_accuracy: 0.9416
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1970 - categorical_accuracy: 0.9416
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1969 - categorical_accuracy: 0.9417
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1968 - categorical_accuracy: 0.9417
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1967 - categorical_accuracy: 0.9417
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1967 - categorical_accuracy: 0.9417
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1966 - categorical_accuracy: 0.9417
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1966 - categorical_accuracy: 0.9417
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1965 - categorical_accuracy: 0.9417
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1965 - categorical_accuracy: 0.9417
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1964 - categorical_accuracy: 0.9418
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1965 - categorical_accuracy: 0.9417
54944/60000 [==========================>...] - ETA: 8s - loss: 0.1965 - categorical_accuracy: 0.9417
54976/60000 [==========================>...] - ETA: 8s - loss: 0.1964 - categorical_accuracy: 0.9417
55008/60000 [==========================>...] - ETA: 8s - loss: 0.1964 - categorical_accuracy: 0.9418
55040/60000 [==========================>...] - ETA: 8s - loss: 0.1963 - categorical_accuracy: 0.9418
55072/60000 [==========================>...] - ETA: 8s - loss: 0.1962 - categorical_accuracy: 0.9418
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1961 - categorical_accuracy: 0.9418
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1960 - categorical_accuracy: 0.9419
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1959 - categorical_accuracy: 0.9419
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1958 - categorical_accuracy: 0.9419
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1957 - categorical_accuracy: 0.9419
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1957 - categorical_accuracy: 0.9420
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1956 - categorical_accuracy: 0.9420
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1955 - categorical_accuracy: 0.9420
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1954 - categorical_accuracy: 0.9420
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1954 - categorical_accuracy: 0.9420
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1953 - categorical_accuracy: 0.9420
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1952 - categorical_accuracy: 0.9420
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1952 - categorical_accuracy: 0.9421
55520/60000 [==========================>...] - ETA: 7s - loss: 0.1951 - categorical_accuracy: 0.9421
55552/60000 [==========================>...] - ETA: 7s - loss: 0.1951 - categorical_accuracy: 0.9421
55584/60000 [==========================>...] - ETA: 7s - loss: 0.1950 - categorical_accuracy: 0.9421
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1949 - categorical_accuracy: 0.9421
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1948 - categorical_accuracy: 0.9421
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1949 - categorical_accuracy: 0.9421
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1948 - categorical_accuracy: 0.9421
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1947 - categorical_accuracy: 0.9421
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1947 - categorical_accuracy: 0.9421
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1947 - categorical_accuracy: 0.9421
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1946 - categorical_accuracy: 0.9422
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1944 - categorical_accuracy: 0.9422
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1944 - categorical_accuracy: 0.9422
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1943 - categorical_accuracy: 0.9422
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1942 - categorical_accuracy: 0.9422
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1942 - categorical_accuracy: 0.9422
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1941 - categorical_accuracy: 0.9423
56064/60000 [===========================>..] - ETA: 6s - loss: 0.1941 - categorical_accuracy: 0.9423
56096/60000 [===========================>..] - ETA: 6s - loss: 0.1940 - categorical_accuracy: 0.9423
56128/60000 [===========================>..] - ETA: 6s - loss: 0.1939 - categorical_accuracy: 0.9423
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1938 - categorical_accuracy: 0.9423
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1938 - categorical_accuracy: 0.9424
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1938 - categorical_accuracy: 0.9424
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1937 - categorical_accuracy: 0.9424
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1937 - categorical_accuracy: 0.9424
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1936 - categorical_accuracy: 0.9424
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1935 - categorical_accuracy: 0.9425
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1935 - categorical_accuracy: 0.9424
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1935 - categorical_accuracy: 0.9424
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1934 - categorical_accuracy: 0.9425
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1934 - categorical_accuracy: 0.9425
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1933 - categorical_accuracy: 0.9425
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1933 - categorical_accuracy: 0.9425
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1932 - categorical_accuracy: 0.9425
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1932 - categorical_accuracy: 0.9425
56640/60000 [===========================>..] - ETA: 5s - loss: 0.1931 - categorical_accuracy: 0.9425
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1930 - categorical_accuracy: 0.9425
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1929 - categorical_accuracy: 0.9426
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1929 - categorical_accuracy: 0.9426
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1929 - categorical_accuracy: 0.9426
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1928 - categorical_accuracy: 0.9426
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1927 - categorical_accuracy: 0.9426
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1926 - categorical_accuracy: 0.9426
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1925 - categorical_accuracy: 0.9426
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1925 - categorical_accuracy: 0.9427
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1924 - categorical_accuracy: 0.9427
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1923 - categorical_accuracy: 0.9427
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1923 - categorical_accuracy: 0.9427
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1922 - categorical_accuracy: 0.9427
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1921 - categorical_accuracy: 0.9428
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1920 - categorical_accuracy: 0.9428
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1920 - categorical_accuracy: 0.9428
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1919 - categorical_accuracy: 0.9428
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1918 - categorical_accuracy: 0.9428
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1917 - categorical_accuracy: 0.9429
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1917 - categorical_accuracy: 0.9429
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1916 - categorical_accuracy: 0.9429
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1915 - categorical_accuracy: 0.9429
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1915 - categorical_accuracy: 0.9429
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1914 - categorical_accuracy: 0.9429
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1913 - categorical_accuracy: 0.9430
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1913 - categorical_accuracy: 0.9430
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1912 - categorical_accuracy: 0.9430
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1911 - categorical_accuracy: 0.9430
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1910 - categorical_accuracy: 0.9430
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1910 - categorical_accuracy: 0.9431
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1909 - categorical_accuracy: 0.9431
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1908 - categorical_accuracy: 0.9431
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1907 - categorical_accuracy: 0.9431
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1907 - categorical_accuracy: 0.9431
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1906 - categorical_accuracy: 0.9432
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1905 - categorical_accuracy: 0.9432
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1905 - categorical_accuracy: 0.9432
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1904 - categorical_accuracy: 0.9432
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1904 - categorical_accuracy: 0.9432
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1903 - categorical_accuracy: 0.9432
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1904 - categorical_accuracy: 0.9432
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1903 - categorical_accuracy: 0.9432
58016/60000 [============================>.] - ETA: 3s - loss: 0.1902 - categorical_accuracy: 0.9433
58048/60000 [============================>.] - ETA: 3s - loss: 0.1902 - categorical_accuracy: 0.9433
58080/60000 [============================>.] - ETA: 3s - loss: 0.1902 - categorical_accuracy: 0.9433
58112/60000 [============================>.] - ETA: 3s - loss: 0.1901 - categorical_accuracy: 0.9433
58144/60000 [============================>.] - ETA: 3s - loss: 0.1900 - categorical_accuracy: 0.9433
58176/60000 [============================>.] - ETA: 3s - loss: 0.1899 - categorical_accuracy: 0.9433
58208/60000 [============================>.] - ETA: 3s - loss: 0.1899 - categorical_accuracy: 0.9434
58240/60000 [============================>.] - ETA: 3s - loss: 0.1898 - categorical_accuracy: 0.9434
58272/60000 [============================>.] - ETA: 3s - loss: 0.1898 - categorical_accuracy: 0.9434
58304/60000 [============================>.] - ETA: 3s - loss: 0.1897 - categorical_accuracy: 0.9434
58336/60000 [============================>.] - ETA: 2s - loss: 0.1897 - categorical_accuracy: 0.9434
58368/60000 [============================>.] - ETA: 2s - loss: 0.1896 - categorical_accuracy: 0.9434
58400/60000 [============================>.] - ETA: 2s - loss: 0.1898 - categorical_accuracy: 0.9434
58432/60000 [============================>.] - ETA: 2s - loss: 0.1897 - categorical_accuracy: 0.9434
58464/60000 [============================>.] - ETA: 2s - loss: 0.1896 - categorical_accuracy: 0.9434
58496/60000 [============================>.] - ETA: 2s - loss: 0.1895 - categorical_accuracy: 0.9435
58528/60000 [============================>.] - ETA: 2s - loss: 0.1895 - categorical_accuracy: 0.9435
58560/60000 [============================>.] - ETA: 2s - loss: 0.1894 - categorical_accuracy: 0.9435
58592/60000 [============================>.] - ETA: 2s - loss: 0.1893 - categorical_accuracy: 0.9435
58624/60000 [============================>.] - ETA: 2s - loss: 0.1892 - categorical_accuracy: 0.9436
58656/60000 [============================>.] - ETA: 2s - loss: 0.1891 - categorical_accuracy: 0.9436
58688/60000 [============================>.] - ETA: 2s - loss: 0.1891 - categorical_accuracy: 0.9436
58720/60000 [============================>.] - ETA: 2s - loss: 0.1890 - categorical_accuracy: 0.9436
58752/60000 [============================>.] - ETA: 2s - loss: 0.1889 - categorical_accuracy: 0.9437
58784/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9437
58816/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9437
58848/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9437
58880/60000 [============================>.] - ETA: 1s - loss: 0.1889 - categorical_accuracy: 0.9437
58912/60000 [============================>.] - ETA: 1s - loss: 0.1888 - categorical_accuracy: 0.9437
58944/60000 [============================>.] - ETA: 1s - loss: 0.1888 - categorical_accuracy: 0.9437
58976/60000 [============================>.] - ETA: 1s - loss: 0.1887 - categorical_accuracy: 0.9437
59008/60000 [============================>.] - ETA: 1s - loss: 0.1886 - categorical_accuracy: 0.9438
59040/60000 [============================>.] - ETA: 1s - loss: 0.1886 - categorical_accuracy: 0.9438
59072/60000 [============================>.] - ETA: 1s - loss: 0.1885 - categorical_accuracy: 0.9438
59104/60000 [============================>.] - ETA: 1s - loss: 0.1884 - categorical_accuracy: 0.9438
59136/60000 [============================>.] - ETA: 1s - loss: 0.1883 - categorical_accuracy: 0.9439
59168/60000 [============================>.] - ETA: 1s - loss: 0.1883 - categorical_accuracy: 0.9439
59200/60000 [============================>.] - ETA: 1s - loss: 0.1883 - categorical_accuracy: 0.9439
59232/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9439
59264/60000 [============================>.] - ETA: 1s - loss: 0.1884 - categorical_accuracy: 0.9439
59296/60000 [============================>.] - ETA: 1s - loss: 0.1883 - categorical_accuracy: 0.9439
59328/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9440
59360/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9440
59392/60000 [============================>.] - ETA: 1s - loss: 0.1881 - categorical_accuracy: 0.9440
59424/60000 [============================>.] - ETA: 1s - loss: 0.1880 - categorical_accuracy: 0.9440
59456/60000 [============================>.] - ETA: 0s - loss: 0.1879 - categorical_accuracy: 0.9441
59488/60000 [============================>.] - ETA: 0s - loss: 0.1878 - categorical_accuracy: 0.9441
59520/60000 [============================>.] - ETA: 0s - loss: 0.1878 - categorical_accuracy: 0.9441
59552/60000 [============================>.] - ETA: 0s - loss: 0.1877 - categorical_accuracy: 0.9441
59584/60000 [============================>.] - ETA: 0s - loss: 0.1878 - categorical_accuracy: 0.9441
59616/60000 [============================>.] - ETA: 0s - loss: 0.1877 - categorical_accuracy: 0.9441
59648/60000 [============================>.] - ETA: 0s - loss: 0.1877 - categorical_accuracy: 0.9442
59680/60000 [============================>.] - ETA: 0s - loss: 0.1876 - categorical_accuracy: 0.9442
59712/60000 [============================>.] - ETA: 0s - loss: 0.1875 - categorical_accuracy: 0.9442
59744/60000 [============================>.] - ETA: 0s - loss: 0.1874 - categorical_accuracy: 0.9442
59776/60000 [============================>.] - ETA: 0s - loss: 0.1873 - categorical_accuracy: 0.9443
59808/60000 [============================>.] - ETA: 0s - loss: 0.1872 - categorical_accuracy: 0.9443
59840/60000 [============================>.] - ETA: 0s - loss: 0.1872 - categorical_accuracy: 0.9443
59872/60000 [============================>.] - ETA: 0s - loss: 0.1871 - categorical_accuracy: 0.9443
59904/60000 [============================>.] - ETA: 0s - loss: 0.1870 - categorical_accuracy: 0.9444
59936/60000 [============================>.] - ETA: 0s - loss: 0.1869 - categorical_accuracy: 0.9444
59968/60000 [============================>.] - ETA: 0s - loss: 0.1869 - categorical_accuracy: 0.9444
60000/60000 [==============================] - 110s 2ms/step - loss: 0.1868 - categorical_accuracy: 0.9444 - val_loss: 0.0471 - val_categorical_accuracy: 0.9851

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 16s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 4s
  672/10000 [=>............................] - ETA: 3s
  832/10000 [=>............................] - ETA: 3s
  992/10000 [=>............................] - ETA: 3s
 1152/10000 [==>...........................] - ETA: 3s
 1312/10000 [==>...........................] - ETA: 3s
 1472/10000 [===>..........................] - ETA: 3s
 1632/10000 [===>..........................] - ETA: 3s
 1792/10000 [====>.........................] - ETA: 3s
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
 3872/10000 [==========>...................] - ETA: 2s
 4032/10000 [===========>..................] - ETA: 2s
 4192/10000 [===========>..................] - ETA: 2s
 4352/10000 [============>.................] - ETA: 1s
 4512/10000 [============>.................] - ETA: 1s
 4672/10000 [=============>................] - ETA: 1s
 4832/10000 [=============>................] - ETA: 1s
 4992/10000 [=============>................] - ETA: 1s
 5152/10000 [==============>...............] - ETA: 1s
 5312/10000 [==============>...............] - ETA: 1s
 5472/10000 [===============>..............] - ETA: 1s
 5632/10000 [===============>..............] - ETA: 1s
 5792/10000 [================>.............] - ETA: 1s
 5952/10000 [================>.............] - ETA: 1s
 6112/10000 [=================>............] - ETA: 1s
 6240/10000 [=================>............] - ETA: 1s
 6400/10000 [==================>...........] - ETA: 1s
 6560/10000 [==================>...........] - ETA: 1s
 6720/10000 [===================>..........] - ETA: 1s
 6880/10000 [===================>..........] - ETA: 1s
 7040/10000 [====================>.........] - ETA: 1s
 7200/10000 [====================>.........] - ETA: 0s
 7360/10000 [=====================>........] - ETA: 0s
 7520/10000 [=====================>........] - ETA: 0s
 7680/10000 [======================>.......] - ETA: 0s
 7840/10000 [======================>.......] - ETA: 0s
 8000/10000 [=======================>......] - ETA: 0s
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
 9760/10000 [============================>.] - ETA: 0s
 9920/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 345us/step
[[1.85657541e-07 5.60412907e-08 9.71614645e-06 ... 9.99985576e-01
  1.06681071e-07 1.65204483e-06]
 [1.55045054e-05 8.57179330e-05 9.99889374e-01 ... 1.51169068e-08
  2.96833196e-06 9.52794621e-10]
 [1.24073244e-06 9.99808371e-01 5.47072850e-05 ... 6.90290544e-05
  2.21769023e-05 2.67075870e-06]
 ...
 [2.35202169e-09 9.09512039e-07 1.04020605e-08 ... 9.75872808e-07
  1.72000000e-05 3.63709260e-05]
 [5.42411449e-08 9.02871466e-09 1.66750558e-09 ... 5.27525934e-09
  6.08449918e-04 2.19054161e-08]
 [3.01584714e-06 2.62245948e-07 5.16689738e-07 ... 8.97785568e-10
  5.64824347e-07 4.08419770e-10]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04705540933823213, 'accuracy_test:': 0.9850999712944031}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   71aa2de..05d2eb1  master     -> origin/master
Updating 71aa2de..05d2eb1
Fast-forward
 .../20200513/list_log_dataloader_20200513.md       |   2 +-
 error_list/20200513/list_log_json_20200513.md      | 276 ++++++++++-----------
 .../20200513/list_log_pullrequest_20200513.md      |   2 +-
 error_list/20200513/list_log_test_cli_20200513.md  | 138 +++++------
 4 files changed, 209 insertions(+), 209 deletions(-)
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
[master b625ce5] ml_store
 1 file changed, 2045 insertions(+)
To github.com:arita37/mlmodels_store.git
   05d2eb1..b625ce5  master -> master





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
{'loss': 0.49068786203861237, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-13 08:32:04.476230: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master 1eb00c6] ml_store
 1 file changed, 233 insertions(+)
To github.com:arita37/mlmodels_store.git
   b625ce5..1eb00c6  master -> master





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
[master 8d9aa23] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   1eb00c6..8d9aa23  master -> master





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
	Data preprocessing and feature engineering runtime = 0.22s ...
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
 40%|████      | 2/5 [00:19<00:29,  9.98s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.7740682047225033, 'learning_rate': 0.03677913599733817, 'min_data_in_leaf': 24, 'num_leaves': 65} and reward: 0.389
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\xc5*\xaf\x05\x02\xbdX\r\x00\x00\x00learning_rateq\x02G?\xa2\xd4\xb7\x04\x91A\x9fX\x10\x00\x00\x00min_data_in_leafq\x03K\x18X\n\x00\x00\x00num_leavesq\x04KAu.' and reward: 0.389
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\xc5*\xaf\x05\x02\xbdX\r\x00\x00\x00learning_rateq\x02G?\xa2\xd4\xb7\x04\x91A\x9fX\x10\x00\x00\x00min_data_in_leafq\x03K\x18X\n\x00\x00\x00num_leavesq\x04KAu.' and reward: 0.389
 60%|██████    | 3/5 [00:50<00:32, 16.18s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9612393703708337, 'learning_rate': 0.020893373433258117, 'min_data_in_leaf': 16, 'num_leaves': 41} and reward: 0.3906
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee\xc2y\x11k\xda\x02X\r\x00\x00\x00learning_rateq\x02G?\x95e\x12\x8eeUxX\x10\x00\x00\x00min_data_in_leafq\x03K\x10X\n\x00\x00\x00num_leavesq\x04K)u.' and reward: 0.3906
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee\xc2y\x11k\xda\x02X\r\x00\x00\x00learning_rateq\x02G?\x95e\x12\x8eeUxX\x10\x00\x00\x00min_data_in_leafq\x03K\x10X\n\x00\x00\x00num_leavesq\x04K)u.' and reward: 0.3906
 80%|████████  | 4/5 [01:13<00:18, 18.15s/it] 80%|████████  | 4/5 [01:13<00:18, 18.34s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8233227009951549, 'learning_rate': 0.02309915083928766, 'min_data_in_leaf': 19, 'num_leaves': 66} and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeaX\xa8\xd9Z\x83\xb4X\r\x00\x00\x00learning_rateq\x02G?\x97\xa7M\xc5\xae1\xb2X\x10\x00\x00\x00min_data_in_leafq\x03K\x13X\n\x00\x00\x00num_leavesq\x04KBu.' and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeaX\xa8\xd9Z\x83\xb4X\r\x00\x00\x00learning_rateq\x02G?\x97\xa7M\xc5\xae1\xb2X\x10\x00\x00\x00min_data_in_leafq\x03K\x13X\n\x00\x00\x00num_leavesq\x04KBu.' and reward: 0.39
Time for Gradient Boosting hyperparameter optimization: 104.68706059455872
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 1.0, 'learning_rate': 0.1, 'min_data_in_leaf': 20, 'num_leaves': 36}
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
 40%|████      | 2/5 [00:50<01:15, 25.19s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.31095545402818636, 'embedding_size_factor': 1.1475431968749406, 'layers.choice': 0, 'learning_rate': 0.0007534302635605376, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 1.2962376692896608e-11} and reward: 0.381
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd3\xe6\xb1\xb4d\x16\xceX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2\\VAU3\xdaX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?H\xb0;+\xc0g"X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xac\x81+Z\xe1\xbb2u.' and reward: 0.381
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd3\xe6\xb1\xb4d\x16\xceX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2\\VAU3\xdaX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?H\xb0;+\xc0g"X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xac\x81+Z\xe1\xbb2u.' and reward: 0.381
 60%|██████    | 3/5 [01:38<01:04, 32.18s/it] 60%|██████    | 3/5 [01:38<01:05, 32.96s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.42194806339809765, 'embedding_size_factor': 0.9599712770852988, 'layers.choice': 0, 'learning_rate': 0.0006765140396278561, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 5.2694033086119043e-05} and reward: 0.363
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdb\x012s9\xf1|X\x15\x00\x00\x00embedding_size_factorq\x03G?\xee\xb8\x15\xaf\x05\xc8fX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?F+\x02\xd6vW`X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x0b\xa0y0\xbd\xa5\xacu.' and reward: 0.363
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdb\x012s9\xf1|X\x15\x00\x00\x00embedding_size_factorq\x03G?\xee\xb8\x15\xaf\x05\xc8fX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?F+\x02\xd6vW`X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x0b\xa0y0\xbd\xa5\xacu.' and reward: 0.363

Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 150.240318775177
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.78s of the -139.46s of remaining time.
Ensemble size: 77
Ensemble weights: 
[0.22077922 0.16883117 0.06493506 0.35064935 0.03896104 0.01298701
 0.14285714]
	0.4002	 = Validation accuracy score
	1.63s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 261.13s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f83afa71160>

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   8d9aa23..8df238f  master     -> origin/master
Updating 8d9aa23..8df238f
Fast-forward
 error_list/20200513/list_log_benchmark_20200513.md |  166 +-
 .../20200513/list_log_dataloader_20200513.md       |    2 +-
 error_list/20200513/list_log_import_20200513.md    |    2 +-
 error_list/20200513/list_log_jupyter_20200513.md   | 2264 ++++++++++----------
 .../20200513/list_log_pullrequest_20200513.md      |    2 +-
 error_list/20200513/list_log_test_cli_20200513.md  |  152 +-
 error_list/20200513/list_log_testall_20200513.md   |  320 +--
 7 files changed, 1407 insertions(+), 1501 deletions(-)
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
[master 85cdfe6] ml_store
 1 file changed, 221 insertions(+)
To github.com:arita37/mlmodels_store.git
   8df238f..85cdfe6  master -> master





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
[master 4413754] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   85cdfe6..4413754  master -> master





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
[master e7a103c] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   4413754..e7a103c  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f1eea9c5fd0> 

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
[master 0d91ed0] ml_store
 1 file changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   e7a103c..0d91ed0  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 1.16777676e+00 -6.65754518e-01 -1.23312074e+00 -1.67419581e+00
   1.01313574e+00  8.25029824e-01 -1.20464572e-01 -4.98213564e-01
  -3.10984978e-01 -1.18231813e+00]
 [ 8.72267394e-01 -2.51630386e+00 -7.75070287e-01 -5.95667881e-01
   1.02600767e+00 -3.09121319e-01  1.74643509e+00  5.10937774e-01
   1.71066184e+00  1.41640538e-01]
 [ 8.78740711e-01 -1.92316341e-02  3.19656942e-01  1.50016279e-01
  -1.46662161e+00  4.63534322e-01 -8.98683193e-01  3.97880425e-01
  -9.96010889e-01  3.18154200e-01]
 [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 1.07258847e+00 -5.86523939e-01 -1.34267579e+00 -1.23685338e+00
   1.24328724e+00  8.75838928e-01 -3.26499498e-01  6.23362177e-01
  -4.34956683e-01  1.11438298e+00]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 6.92114488e-01 -6.06524918e-02  2.05635552e+00 -2.41350300e+00
   1.17456965e+00 -1.77756638e+00 -2.81736269e-01 -7.77858827e-01
   1.11584111e+00  1.76024923e+00]
 [ 1.44682180e+00  8.07455917e-01  1.49810818e+00  3.12238689e-01
  -6.82430193e-01 -1.93321640e-01  2.88078167e-01 -2.07680202e+00
   9.47501167e-01 -3.00976154e-01]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]
 [ 8.58774962e-01  2.29371761e+00 -1.47023709e+00 -8.30010986e-01
  -6.72049816e-01 -1.01951985e+00  5.99213235e-01 -2.14653842e-01
   1.02124813e+00  6.06403944e-01]
 [ 6.67591795e-01 -4.52524973e-01 -6.05981321e-01  1.16128569e+00
  -1.44620987e+00  1.06996554e+00  1.92381543e+00 -1.04553425e+00
   3.55284507e-01  1.80358898e+00]
 [ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
  -1.14870423e+00 -9.50909251e-01]
 [ 8.61462558e-01  7.43205537e-02 -1.34501002e+00 -1.99560718e-01
  -1.47533915e+00 -6.54603169e-01 -3.14563862e-01  3.18014296e-01
  -8.90271552e-01 -1.29525789e+00]
 [ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 6.81889336e-01 -1.15498263e+00  1.22895559e+00 -1.77632196e-01
   9.98545187e-01 -1.51045638e+00 -2.75846063e-01  1.01120706e+00
  -1.47656266e+00  1.30970591e+00]
 [ 9.47814113e-01 -1.13379204e+00  6.40985866e-01 -1.90548298e-01
  -1.23912256e+00  2.33339126e-01 -3.16901197e-01  4.34998324e-01
   9.10423603e-01  1.21987438e+00]
 [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 6.10942600e-01 -2.79099641e+00 -1.33520272e+00 -4.56117555e-01
  -9.44959948e-01 -9.79890252e-01 -1.56993672e-01  6.92574348e-01
  -4.78672356e-01 -1.06460122e-01]
 [ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
   7.18290736e-01 -5.30119800e-01]
 [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
  -3.53409983e-01 -2.51674208e-01]
 [ 8.95510508e-01  9.20615118e-01  7.94528240e-01 -3.53679249e-02
   8.78099103e-01  2.11060505e+00 -1.02188594e+00 -1.30653407e+00
   7.63804802e-02 -1.87316098e+00]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 6.23688521e-01  1.20660790e+00  9.03999174e-01 -2.82863552e-01
  -1.18913787e+00 -2.66326884e-01  1.42361443e+00  1.06897162e+00
   4.03714310e-02  1.57546791e+00]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f5ced56ff60>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f5d078e6780> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]
 [ 0.44689516  0.38653915  1.35010682 -0.85145566  0.85063796  1.00088142
  -1.1601701  -0.38483225  1.45810824 -0.33128317]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 0.70017571  0.55607351  0.08968641  1.69380911  0.88239331  0.19686978
  -0.56378873  0.16986926 -1.16400797 -0.6011568 ]
 [ 0.99785516 -0.6001388   0.45794708  0.14676526 -0.93355729  0.57180488
   0.57296273 -0.03681766  0.11236849 -0.01781755]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
  -0.3169012   0.43499832  0.9104236   1.21987438]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 0.9292506  -1.10657307 -1.95816909 -0.3592241  -1.21258781  0.5053819
   0.54264529  1.2179409  -1.94068096  0.67780757]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 1.66752297  1.22372221 -0.4599301  -0.0593679  -0.493857    1.4489894
  -1.18110317 -0.47758085  0.02599999 -0.79079995]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]
 [ 0.89551051  0.92061512  0.79452824 -0.03536792  0.8780991   2.11060505
  -1.02188594 -1.30653407  0.07638048 -1.87316098]
 [ 2.07582971 -1.40232915 -0.47918492  0.45112294  1.03436581 -0.6949209
  -0.4189379   0.5154138  -1.11487105 -1.95210529]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]
 [ 1.25704434 -1.82391985 -0.61240697  1.16707517 -0.62373281 -0.0396687
   0.81604368  0.8858258   0.18986165  0.39310924]
 [ 1.838294    0.50274088  0.12910158  1.55880554  1.32551412  0.1094027
   1.40754    -1.2197444   2.44936865  1.6169496 ]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 1.16777676 -0.66575452 -1.23312074 -1.67419581  1.01313574  0.82502982
  -0.12046457 -0.49821356 -0.31098498 -1.18231813]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]]
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
[[ 1.83829400e+00  5.02740882e-01  1.29101580e-01  1.55880554e+00
   1.32551412e+00  1.09402696e-01  1.40754000e+00 -1.21974440e+00
   2.44936865e+00  1.61694960e+00]
 [ 9.97855163e-01 -6.00138799e-01  4.57947076e-01  1.46765263e-01
  -9.33557290e-01  5.71804879e-01  5.72962726e-01 -3.68176565e-02
   1.12368489e-01 -1.78175491e-02]
 [ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
  -4.26428618e-01 -8.39988915e-01]
 [ 1.22867367e+00  1.34373116e-01 -1.82420406e-01 -2.68371304e-01
  -1.73963799e+00 -1.31675626e-01 -9.26871939e-01  1.01855247e+00
   1.23055820e+00 -4.91125138e-01]
 [ 1.27991386e+00 -8.71422066e-01 -3.24032329e-01 -8.64829941e-01
  -9.68539694e-01  6.08749082e-01  5.07984337e-01  5.61638097e-01
   1.51475038e+00 -1.51107661e+00]
 [ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 1.07258847e+00 -5.86523939e-01 -1.34267579e+00 -1.23685338e+00
   1.24328724e+00  8.75838928e-01 -3.26499498e-01  6.23362177e-01
  -4.34956683e-01  1.11438298e+00]
 [ 1.34740825e+00  7.33023232e-01  8.38634747e-01 -1.89881206e+00
  -5.42459922e-01 -1.11711069e+00 -1.09715436e+00 -5.08972278e-01
  -1.66485955e-01 -1.03918232e+00]
 [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 5.63077902e-01 -1.17598267e+00 -1.74180344e-01  1.01012718e+00
   1.06796368e+00  9.20017933e-01 -1.68198840e-01 -1.95057341e-01
   8.05393424e-01  4.61164100e-01]
 [ 8.98917161e-01  5.57439453e-01 -7.58067329e-01  1.81038744e-01
   8.41467206e-01  1.10717545e+00  6.93366226e-01  1.44287693e+00
  -5.39681562e-01 -8.08847196e-01]
 [ 2.07582971e+00 -1.40232915e+00 -4.79184915e-01  4.51122939e-01
   1.03436581e+00 -6.94920901e-01 -4.18937898e-01  5.15413802e-01
  -1.11487105e+00 -1.95210529e+00]
 [ 8.88838813e-01  1.03368687e+00 -4.97025792e-02  8.08844360e-01
   8.14051347e-01  1.78975468e+00  1.14690038e+00  4.51284016e-01
  -1.68405999e+00  4.66643267e-01]
 [ 1.21619061e+00 -1.90005215e-02  8.60891241e-01 -2.26760192e-01
  -1.36419132e+00 -1.56450785e+00  1.63169151e+00  9.31255679e-01
   9.49808815e-01 -8.80189065e-01]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
   8.84908814e-01  2.76499305e-01]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 6.92114488e-01 -6.06524918e-02  2.05635552e+00 -2.41350300e+00
   1.17456965e+00 -1.77756638e+00 -2.81736269e-01 -7.77858827e-01
   1.11584111e+00  1.76024923e+00]
 [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
  -2.43250851e+00 -2.27729800e+00]
 [ 8.58774962e-01  2.29371761e+00 -1.47023709e+00 -8.30010986e-01
  -6.72049816e-01 -1.01951985e+00  5.99213235e-01 -2.14653842e-01
   1.02124813e+00  6.06403944e-01]
 [ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]]
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
[master adb1624] ml_store
 1 file changed, 296 insertions(+)
To github.com:arita37/mlmodels_store.git
   0d91ed0..adb1624  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140350632337360
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140350632337136
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140350632335904
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140350632335456
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140350632334952
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140350632334616

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
grad_step = 000000, loss = 0.442247
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.367756
grad_step = 000002, loss = 0.318408
grad_step = 000003, loss = 0.258203
grad_step = 000004, loss = 0.191439
grad_step = 000005, loss = 0.122721
grad_step = 000006, loss = 0.066537
grad_step = 000007, loss = 0.044178
grad_step = 000008, loss = 0.057762
grad_step = 000009, loss = 0.060512
grad_step = 000010, loss = 0.046059
grad_step = 000011, loss = 0.026112
grad_step = 000012, loss = 0.011606
grad_step = 000013, loss = 0.006998
grad_step = 000014, loss = 0.010061
grad_step = 000015, loss = 0.015789
grad_step = 000016, loss = 0.020388
grad_step = 000017, loss = 0.022226
grad_step = 000018, loss = 0.021387
grad_step = 000019, loss = 0.018903
grad_step = 000020, loss = 0.016157
grad_step = 000021, loss = 0.014294
grad_step = 000022, loss = 0.013628
grad_step = 000023, loss = 0.013470
grad_step = 000024, loss = 0.012818
grad_step = 000025, loss = 0.011327
grad_step = 000026, loss = 0.009492
grad_step = 000027, loss = 0.007992
grad_step = 000028, loss = 0.007201
grad_step = 000029, loss = 0.007091
grad_step = 000030, loss = 0.007372
grad_step = 000031, loss = 0.007723
grad_step = 000032, loss = 0.007941
grad_step = 000033, loss = 0.007961
grad_step = 000034, loss = 0.007813
grad_step = 000035, loss = 0.007565
grad_step = 000036, loss = 0.007285
grad_step = 000037, loss = 0.007024
grad_step = 000038, loss = 0.006800
grad_step = 000039, loss = 0.006595
grad_step = 000040, loss = 0.006402
grad_step = 000041, loss = 0.006221
grad_step = 000042, loss = 0.006074
grad_step = 000043, loss = 0.005983
grad_step = 000044, loss = 0.005944
grad_step = 000045, loss = 0.005928
grad_step = 000046, loss = 0.005896
grad_step = 000047, loss = 0.005833
grad_step = 000048, loss = 0.005750
grad_step = 000049, loss = 0.005670
grad_step = 000050, loss = 0.005608
grad_step = 000051, loss = 0.005563
grad_step = 000052, loss = 0.005527
grad_step = 000053, loss = 0.005492
grad_step = 000054, loss = 0.005454
grad_step = 000055, loss = 0.005411
grad_step = 000056, loss = 0.005364
grad_step = 000057, loss = 0.005313
grad_step = 000058, loss = 0.005259
grad_step = 000059, loss = 0.005201
grad_step = 000060, loss = 0.005136
grad_step = 000061, loss = 0.005066
grad_step = 000062, loss = 0.004999
grad_step = 000063, loss = 0.004943
grad_step = 000064, loss = 0.004903
grad_step = 000065, loss = 0.004871
grad_step = 000066, loss = 0.004839
grad_step = 000067, loss = 0.004797
grad_step = 000068, loss = 0.004745
grad_step = 000069, loss = 0.004690
grad_step = 000070, loss = 0.004634
grad_step = 000071, loss = 0.004576
grad_step = 000072, loss = 0.004512
grad_step = 000073, loss = 0.004442
grad_step = 000074, loss = 0.004377
grad_step = 000075, loss = 0.004319
grad_step = 000076, loss = 0.004265
grad_step = 000077, loss = 0.004209
grad_step = 000078, loss = 0.004152
grad_step = 000079, loss = 0.004092
grad_step = 000080, loss = 0.004031
grad_step = 000081, loss = 0.003968
grad_step = 000082, loss = 0.003905
grad_step = 000083, loss = 0.003841
grad_step = 000084, loss = 0.003776
grad_step = 000085, loss = 0.003713
grad_step = 000086, loss = 0.003654
grad_step = 000087, loss = 0.003592
grad_step = 000088, loss = 0.003528
grad_step = 000089, loss = 0.003463
grad_step = 000090, loss = 0.003398
grad_step = 000091, loss = 0.003333
grad_step = 000092, loss = 0.003268
grad_step = 000093, loss = 0.003203
grad_step = 000094, loss = 0.003139
grad_step = 000095, loss = 0.003078
grad_step = 000096, loss = 0.003015
grad_step = 000097, loss = 0.002953
grad_step = 000098, loss = 0.002894
grad_step = 000099, loss = 0.002833
grad_step = 000100, loss = 0.002776
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002720
grad_step = 000102, loss = 0.002661
grad_step = 000103, loss = 0.002603
grad_step = 000104, loss = 0.002545
grad_step = 000105, loss = 0.002486
grad_step = 000106, loss = 0.002431
grad_step = 000107, loss = 0.002376
grad_step = 000108, loss = 0.002317
grad_step = 000109, loss = 0.002257
grad_step = 000110, loss = 0.002202
grad_step = 000111, loss = 0.002147
grad_step = 000112, loss = 0.002089
grad_step = 000113, loss = 0.002034
grad_step = 000114, loss = 0.001982
grad_step = 000115, loss = 0.001932
grad_step = 000116, loss = 0.001881
grad_step = 000117, loss = 0.001833
grad_step = 000118, loss = 0.001789
grad_step = 000119, loss = 0.001746
grad_step = 000120, loss = 0.001705
grad_step = 000121, loss = 0.001668
grad_step = 000122, loss = 0.001633
grad_step = 000123, loss = 0.001600
grad_step = 000124, loss = 0.001566
grad_step = 000125, loss = 0.001535
grad_step = 000126, loss = 0.001504
grad_step = 000127, loss = 0.001474
grad_step = 000128, loss = 0.001443
grad_step = 000129, loss = 0.001412
grad_step = 000130, loss = 0.001379
grad_step = 000131, loss = 0.001346
grad_step = 000132, loss = 0.001313
grad_step = 000133, loss = 0.001280
grad_step = 000134, loss = 0.001249
grad_step = 000135, loss = 0.001221
grad_step = 000136, loss = 0.001194
grad_step = 000137, loss = 0.001162
grad_step = 000138, loss = 0.001127
grad_step = 000139, loss = 0.001100
grad_step = 000140, loss = 0.001078
grad_step = 000141, loss = 0.001053
grad_step = 000142, loss = 0.001026
grad_step = 000143, loss = 0.001003
grad_step = 000144, loss = 0.000985
grad_step = 000145, loss = 0.000967
grad_step = 000146, loss = 0.000947
grad_step = 000147, loss = 0.000928
grad_step = 000148, loss = 0.000911
grad_step = 000149, loss = 0.000894
grad_step = 000150, loss = 0.000878
grad_step = 000151, loss = 0.000863
grad_step = 000152, loss = 0.000849
grad_step = 000153, loss = 0.000835
grad_step = 000154, loss = 0.000821
grad_step = 000155, loss = 0.000808
grad_step = 000156, loss = 0.000796
grad_step = 000157, loss = 0.000787
grad_step = 000158, loss = 0.000779
grad_step = 000159, loss = 0.000770
grad_step = 000160, loss = 0.000761
grad_step = 000161, loss = 0.000750
grad_step = 000162, loss = 0.000738
grad_step = 000163, loss = 0.000726
grad_step = 000164, loss = 0.000715
grad_step = 000165, loss = 0.000711
grad_step = 000166, loss = 0.000711
grad_step = 000167, loss = 0.000705
grad_step = 000168, loss = 0.000684
grad_step = 000169, loss = 0.000666
grad_step = 000170, loss = 0.000666
grad_step = 000171, loss = 0.000670
grad_step = 000172, loss = 0.000658
grad_step = 000173, loss = 0.000641
grad_step = 000174, loss = 0.000637
grad_step = 000175, loss = 0.000640
grad_step = 000176, loss = 0.000634
grad_step = 000177, loss = 0.000621
grad_step = 000178, loss = 0.000615
grad_step = 000179, loss = 0.000617
grad_step = 000180, loss = 0.000618
grad_step = 000181, loss = 0.000615
grad_step = 000182, loss = 0.000615
grad_step = 000183, loss = 0.000620
grad_step = 000184, loss = 0.000610
grad_step = 000185, loss = 0.000588
grad_step = 000186, loss = 0.000582
grad_step = 000187, loss = 0.000591
grad_step = 000188, loss = 0.000589
grad_step = 000189, loss = 0.000573
grad_step = 000190, loss = 0.000569
grad_step = 000191, loss = 0.000574
grad_step = 000192, loss = 0.000567
grad_step = 000193, loss = 0.000556
grad_step = 000194, loss = 0.000556
grad_step = 000195, loss = 0.000558
grad_step = 000196, loss = 0.000551
grad_step = 000197, loss = 0.000545
grad_step = 000198, loss = 0.000546
grad_step = 000199, loss = 0.000550
grad_step = 000200, loss = 0.000549
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000554
grad_step = 000202, loss = 0.000566
grad_step = 000203, loss = 0.000570
grad_step = 000204, loss = 0.000552
grad_step = 000205, loss = 0.000531
grad_step = 000206, loss = 0.000525
grad_step = 000207, loss = 0.000524
grad_step = 000208, loss = 0.000525
grad_step = 000209, loss = 0.000530
grad_step = 000210, loss = 0.000528
grad_step = 000211, loss = 0.000514
grad_step = 000212, loss = 0.000503
grad_step = 000213, loss = 0.000505
grad_step = 000214, loss = 0.000509
grad_step = 000215, loss = 0.000507
grad_step = 000216, loss = 0.000504
grad_step = 000217, loss = 0.000500
grad_step = 000218, loss = 0.000493
grad_step = 000219, loss = 0.000486
grad_step = 000220, loss = 0.000485
grad_step = 000221, loss = 0.000487
grad_step = 000222, loss = 0.000487
grad_step = 000223, loss = 0.000485
grad_step = 000224, loss = 0.000483
grad_step = 000225, loss = 0.000480
grad_step = 000226, loss = 0.000475
grad_step = 000227, loss = 0.000469
grad_step = 000228, loss = 0.000465
grad_step = 000229, loss = 0.000463
grad_step = 000230, loss = 0.000462
grad_step = 000231, loss = 0.000460
grad_step = 000232, loss = 0.000459
grad_step = 000233, loss = 0.000460
grad_step = 000234, loss = 0.000463
grad_step = 000235, loss = 0.000467
grad_step = 000236, loss = 0.000472
grad_step = 000237, loss = 0.000475
grad_step = 000238, loss = 0.000476
grad_step = 000239, loss = 0.000471
grad_step = 000240, loss = 0.000459
grad_step = 000241, loss = 0.000445
grad_step = 000242, loss = 0.000436
grad_step = 000243, loss = 0.000434
grad_step = 000244, loss = 0.000436
grad_step = 000245, loss = 0.000439
grad_step = 000246, loss = 0.000443
grad_step = 000247, loss = 0.000446
grad_step = 000248, loss = 0.000444
grad_step = 000249, loss = 0.000437
grad_step = 000250, loss = 0.000428
grad_step = 000251, loss = 0.000420
grad_step = 000252, loss = 0.000416
grad_step = 000253, loss = 0.000415
grad_step = 000254, loss = 0.000413
grad_step = 000255, loss = 0.000413
grad_step = 000256, loss = 0.000414
grad_step = 000257, loss = 0.000417
grad_step = 000258, loss = 0.000420
grad_step = 000259, loss = 0.000423
grad_step = 000260, loss = 0.000424
grad_step = 000261, loss = 0.000423
grad_step = 000262, loss = 0.000418
grad_step = 000263, loss = 0.000412
grad_step = 000264, loss = 0.000404
grad_step = 000265, loss = 0.000396
grad_step = 000266, loss = 0.000390
grad_step = 000267, loss = 0.000387
grad_step = 000268, loss = 0.000387
grad_step = 000269, loss = 0.000389
grad_step = 000270, loss = 0.000390
grad_step = 000271, loss = 0.000392
grad_step = 000272, loss = 0.000395
grad_step = 000273, loss = 0.000400
grad_step = 000274, loss = 0.000408
grad_step = 000275, loss = 0.000418
grad_step = 000276, loss = 0.000424
grad_step = 000277, loss = 0.000410
grad_step = 000278, loss = 0.000389
grad_step = 000279, loss = 0.000375
grad_step = 000280, loss = 0.000375
grad_step = 000281, loss = 0.000381
grad_step = 000282, loss = 0.000382
grad_step = 000283, loss = 0.000377
grad_step = 000284, loss = 0.000371
grad_step = 000285, loss = 0.000372
grad_step = 000286, loss = 0.000378
grad_step = 000287, loss = 0.000379
grad_step = 000288, loss = 0.000374
grad_step = 000289, loss = 0.000363
grad_step = 000290, loss = 0.000354
grad_step = 000291, loss = 0.000350
grad_step = 000292, loss = 0.000354
grad_step = 000293, loss = 0.000359
grad_step = 000294, loss = 0.000363
grad_step = 000295, loss = 0.000363
grad_step = 000296, loss = 0.000358
grad_step = 000297, loss = 0.000351
grad_step = 000298, loss = 0.000344
grad_step = 000299, loss = 0.000340
grad_step = 000300, loss = 0.000339
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000340
grad_step = 000302, loss = 0.000342
grad_step = 000303, loss = 0.000345
grad_step = 000304, loss = 0.000346
grad_step = 000305, loss = 0.000346
grad_step = 000306, loss = 0.000345
grad_step = 000307, loss = 0.000342
grad_step = 000308, loss = 0.000338
grad_step = 000309, loss = 0.000334
grad_step = 000310, loss = 0.000330
grad_step = 000311, loss = 0.000328
grad_step = 000312, loss = 0.000328
grad_step = 000313, loss = 0.000338
grad_step = 000314, loss = 0.000377
grad_step = 000315, loss = 0.000454
grad_step = 000316, loss = 0.000586
grad_step = 000317, loss = 0.000468
grad_step = 000318, loss = 0.000409
grad_step = 000319, loss = 0.000503
grad_step = 000320, loss = 0.000487
grad_step = 000321, loss = 0.000345
grad_step = 000322, loss = 0.000431
grad_step = 000323, loss = 0.000416
grad_step = 000324, loss = 0.000348
grad_step = 000325, loss = 0.000416
grad_step = 000326, loss = 0.000355
grad_step = 000327, loss = 0.000390
grad_step = 000328, loss = 0.000345
grad_step = 000329, loss = 0.000422
grad_step = 000330, loss = 0.000370
grad_step = 000331, loss = 0.000394
grad_step = 000332, loss = 0.000374
grad_step = 000333, loss = 0.000379
grad_step = 000334, loss = 0.000350
grad_step = 000335, loss = 0.000352
grad_step = 000336, loss = 0.000348
grad_step = 000337, loss = 0.000347
grad_step = 000338, loss = 0.000336
grad_step = 000339, loss = 0.000327
grad_step = 000340, loss = 0.000336
grad_step = 000341, loss = 0.000325
grad_step = 000342, loss = 0.000335
grad_step = 000343, loss = 0.000312
grad_step = 000344, loss = 0.000330
grad_step = 000345, loss = 0.000309
grad_step = 000346, loss = 0.000326
grad_step = 000347, loss = 0.000310
grad_step = 000348, loss = 0.000316
grad_step = 000349, loss = 0.000309
grad_step = 000350, loss = 0.000308
grad_step = 000351, loss = 0.000311
grad_step = 000352, loss = 0.000304
grad_step = 000353, loss = 0.000309
grad_step = 000354, loss = 0.000300
grad_step = 000355, loss = 0.000304
grad_step = 000356, loss = 0.000299
grad_step = 000357, loss = 0.000302
grad_step = 000358, loss = 0.000300
grad_step = 000359, loss = 0.000299
grad_step = 000360, loss = 0.000298
grad_step = 000361, loss = 0.000295
grad_step = 000362, loss = 0.000296
grad_step = 000363, loss = 0.000293
grad_step = 000364, loss = 0.000294
grad_step = 000365, loss = 0.000292
grad_step = 000366, loss = 0.000292
grad_step = 000367, loss = 0.000292
grad_step = 000368, loss = 0.000290
grad_step = 000369, loss = 0.000290
grad_step = 000370, loss = 0.000289
grad_step = 000371, loss = 0.000289
grad_step = 000372, loss = 0.000288
grad_step = 000373, loss = 0.000288
grad_step = 000374, loss = 0.000288
grad_step = 000375, loss = 0.000288
grad_step = 000376, loss = 0.000290
grad_step = 000377, loss = 0.000291
grad_step = 000378, loss = 0.000296
grad_step = 000379, loss = 0.000299
grad_step = 000380, loss = 0.000306
grad_step = 000381, loss = 0.000305
grad_step = 000382, loss = 0.000305
grad_step = 000383, loss = 0.000295
grad_step = 000384, loss = 0.000287
grad_step = 000385, loss = 0.000281
grad_step = 000386, loss = 0.000281
grad_step = 000387, loss = 0.000284
grad_step = 000388, loss = 0.000287
grad_step = 000389, loss = 0.000285
grad_step = 000390, loss = 0.000281
grad_step = 000391, loss = 0.000280
grad_step = 000392, loss = 0.000282
grad_step = 000393, loss = 0.000289
grad_step = 000394, loss = 0.000299
grad_step = 000395, loss = 0.000310
grad_step = 000396, loss = 0.000323
grad_step = 000397, loss = 0.000336
grad_step = 000398, loss = 0.000343
grad_step = 000399, loss = 0.000335
grad_step = 000400, loss = 0.000311
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000286
grad_step = 000402, loss = 0.000276
grad_step = 000403, loss = 0.000286
grad_step = 000404, loss = 0.000299
grad_step = 000405, loss = 0.000307
grad_step = 000406, loss = 0.000298
grad_step = 000407, loss = 0.000284
grad_step = 000408, loss = 0.000276
grad_step = 000409, loss = 0.000277
grad_step = 000410, loss = 0.000287
grad_step = 000411, loss = 0.000284
grad_step = 000412, loss = 0.000279
grad_step = 000413, loss = 0.000276
grad_step = 000414, loss = 0.000277
grad_step = 000415, loss = 0.000280
grad_step = 000416, loss = 0.000274
grad_step = 000417, loss = 0.000268
grad_step = 000418, loss = 0.000267
grad_step = 000419, loss = 0.000269
grad_step = 000420, loss = 0.000274
grad_step = 000421, loss = 0.000273
grad_step = 000422, loss = 0.000274
grad_step = 000423, loss = 0.000269
grad_step = 000424, loss = 0.000267
grad_step = 000425, loss = 0.000266
grad_step = 000426, loss = 0.000266
grad_step = 000427, loss = 0.000267
grad_step = 000428, loss = 0.000266
grad_step = 000429, loss = 0.000265
grad_step = 000430, loss = 0.000262
grad_step = 000431, loss = 0.000260
grad_step = 000432, loss = 0.000260
grad_step = 000433, loss = 0.000261
grad_step = 000434, loss = 0.000262
grad_step = 000435, loss = 0.000263
grad_step = 000436, loss = 0.000264
grad_step = 000437, loss = 0.000265
grad_step = 000438, loss = 0.000265
grad_step = 000439, loss = 0.000264
grad_step = 000440, loss = 0.000264
grad_step = 000441, loss = 0.000264
grad_step = 000442, loss = 0.000264
grad_step = 000443, loss = 0.000265
grad_step = 000444, loss = 0.000267
grad_step = 000445, loss = 0.000276
grad_step = 000446, loss = 0.000281
grad_step = 000447, loss = 0.000294
grad_step = 000448, loss = 0.000288
grad_step = 000449, loss = 0.000286
grad_step = 000450, loss = 0.000288
grad_step = 000451, loss = 0.000292
grad_step = 000452, loss = 0.000289
grad_step = 000453, loss = 0.000276
grad_step = 000454, loss = 0.000268
grad_step = 000455, loss = 0.000257
grad_step = 000456, loss = 0.000262
grad_step = 000457, loss = 0.000270
grad_step = 000458, loss = 0.000282
grad_step = 000459, loss = 0.000290
grad_step = 000460, loss = 0.000289
grad_step = 000461, loss = 0.000274
grad_step = 000462, loss = 0.000257
grad_step = 000463, loss = 0.000248
grad_step = 000464, loss = 0.000250
grad_step = 000465, loss = 0.000258
grad_step = 000466, loss = 0.000268
grad_step = 000467, loss = 0.000275
grad_step = 000468, loss = 0.000274
grad_step = 000469, loss = 0.000262
grad_step = 000470, loss = 0.000251
grad_step = 000471, loss = 0.000245
grad_step = 000472, loss = 0.000249
grad_step = 000473, loss = 0.000258
grad_step = 000474, loss = 0.000268
grad_step = 000475, loss = 0.000270
grad_step = 000476, loss = 0.000268
grad_step = 000477, loss = 0.000261
grad_step = 000478, loss = 0.000264
grad_step = 000479, loss = 0.000275
grad_step = 000480, loss = 0.000327
grad_step = 000481, loss = 0.000301
grad_step = 000482, loss = 0.000261
grad_step = 000483, loss = 0.000254
grad_step = 000484, loss = 0.000285
grad_step = 000485, loss = 0.000314
grad_step = 000486, loss = 0.000281
grad_step = 000487, loss = 0.000275
grad_step = 000488, loss = 0.000246
grad_step = 000489, loss = 0.000269
grad_step = 000490, loss = 0.000304
grad_step = 000491, loss = 0.000280
grad_step = 000492, loss = 0.000259
grad_step = 000493, loss = 0.000272
grad_step = 000494, loss = 0.000303
grad_step = 000495, loss = 0.000263
grad_step = 000496, loss = 0.000259
grad_step = 000497, loss = 0.000254
grad_step = 000498, loss = 0.000241
grad_step = 000499, loss = 0.000260
grad_step = 000500, loss = 0.000257
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000267
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
[[0.8479406  0.84752685 0.9186009  0.95755583 1.0231929 ]
 [0.84637624 0.90865743 0.9444554  1.0213189  0.98606086]
 [0.8882164  0.91895896 1.000885   0.984856   0.95292753]
 [0.9214096  0.9798085  0.98523265 0.9651366  0.92169565]
 [0.9843102  0.99465585 0.9444321  0.90339947 0.8643256 ]
 [0.9856626  0.9529141  0.91070074 0.8614925  0.86650044]
 [0.92915016 0.9065174  0.84276664 0.8643801  0.8222791 ]
 [0.900803   0.82834995 0.851173   0.8094792  0.8552036 ]
 [0.8252609  0.8359948  0.80436766 0.84879065 0.8556782 ]
 [0.81859404 0.80983686 0.8251096  0.8481731  0.8421659 ]
 [0.81883085 0.8292092  0.85273105 0.83853996 0.9233754 ]
 [0.81218505 0.85549    0.8155823  0.9166431  0.9562912 ]
 [0.84060955 0.84345937 0.91606176 0.95657706 1.0170813 ]
 [0.8500654  0.92243594 0.9484462  1.0197549  0.9752963 ]
 [0.89993495 0.9347239  1.000214   0.97473174 0.93708175]
 [0.93399054 0.9919702  0.97743064 0.94859976 0.90344834]
 [0.9842811  0.9906674  0.92628807 0.884762   0.8399757 ]
 [0.9784156  0.93027866 0.8862349  0.837388   0.84729433]
 [0.92496246 0.8960947  0.8248464  0.8473147  0.815652  ]
 [0.90529263 0.83535147 0.84303415 0.8044296  0.850331  ]
 [0.84010607 0.85287714 0.805045   0.849927   0.8601442 ]
 [0.83779424 0.8258422  0.8327738  0.8530108  0.8445166 ]
 [0.83105654 0.84097093 0.86151046 0.84562474 0.92507756]
 [0.8181873  0.8674619  0.8257262  0.92366695 0.95702684]
 [0.8553442  0.8552786  0.91965723 0.9595855  1.0309759 ]
 [0.8565502  0.9176451  0.9480014  1.0297225  1.0004777 ]
 [0.89920044 0.93209594 1.011431   0.9974283  0.9650333 ]
 [0.93114084 0.99445707 0.99864453 0.9800513  0.9333046 ]
 [0.99230313 1.0095778  0.9574778  0.9169913  0.8697934 ]
 [0.9946577  0.9630605  0.9191522  0.869123   0.8720891 ]
 [0.9382947  0.91289926 0.8492779  0.86832327 0.82930756]]

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
[master cecc2fd] ml_store
 1 file changed, 1122 insertions(+)
To github.com:arita37/mlmodels_store.git
   adb1624..cecc2fd  master -> master





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
[master 494606e] ml_store
 1 file changed, 37 insertions(+)
To github.com:arita37/mlmodels_store.git
   cecc2fd..494606e  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 10715475.19B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 356944.50B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 4295680/440473133 [00:00<00:10, 42952130.29B/s]  2%|▏         | 9390080/440473133 [00:00<00:09, 45072161.71B/s]  3%|▎         | 14653440/440473133 [00:00<00:09, 47098773.08B/s]  5%|▍         | 19880960/440473133 [00:00<00:08, 48538906.16B/s]  6%|▌         | 25121792/440473133 [00:00<00:08, 49637247.28B/s]  7%|▋         | 30520320/440473133 [00:00<00:08, 50865828.97B/s]  8%|▊         | 35614720/440473133 [00:00<00:07, 50887269.91B/s]  9%|▉         | 40712192/440473133 [00:00<00:07, 50912978.76B/s] 11%|█         | 46285824/440473133 [00:00<00:07, 52269302.56B/s] 12%|█▏        | 51452928/440473133 [00:01<00:07, 52087074.73B/s] 13%|█▎        | 56579072/440473133 [00:01<00:07, 51835307.96B/s] 14%|█▍        | 61790208/440473133 [00:01<00:07, 51906586.26B/s] 15%|█▌        | 67231744/440473133 [00:01<00:07, 52633218.91B/s] 16%|█▋        | 72553472/440473133 [00:01<00:06, 52806239.65B/s] 18%|█▊        | 77933568/440473133 [00:01<00:06, 53097825.78B/s] 19%|█▉        | 83227648/440473133 [00:01<00:06, 52313222.13B/s] 20%|██        | 88451072/440473133 [00:01<00:06, 51842593.16B/s] 21%|██▏       | 93630464/440473133 [00:01<00:06, 51765974.30B/s] 22%|██▏       | 98851840/440473133 [00:01<00:06, 51899405.55B/s] 24%|██▎       | 104039424/440473133 [00:02<00:06, 51873086.15B/s] 25%|██▍       | 109225984/440473133 [00:02<00:06, 51772528.72B/s] 26%|██▌       | 114402304/440473133 [00:02<00:06, 51011069.06B/s] 27%|██▋       | 119639040/440473133 [00:02<00:06, 51410103.09B/s] 28%|██▊       | 124916736/440473133 [00:02<00:06, 51811445.56B/s] 30%|██▉       | 130208768/440473133 [00:02<00:05, 52138036.65B/s] 31%|███       | 135560192/440473133 [00:02<00:05, 52543274.87B/s] 32%|███▏      | 140817408/440473133 [00:02<00:05, 52180513.01B/s] 33%|███▎      | 146110464/440473133 [00:02<00:05, 52399441.45B/s] 34%|███▍      | 151586816/440473133 [00:02<00:05, 53086819.86B/s] 36%|███▌      | 157079552/440473133 [00:03<00:05, 53621570.28B/s] 37%|███▋      | 162446336/440473133 [00:03<00:05, 53532292.54B/s] 38%|███▊      | 167802880/440473133 [00:03<00:05, 53023088.84B/s] 39%|███▉      | 173176832/440473133 [00:03<00:05, 53234812.45B/s] 41%|████      | 178726912/440473133 [00:03<00:04, 53891885.10B/s] 42%|████▏     | 184120320/440473133 [00:03<00:04, 53604468.81B/s] 43%|████▎     | 189577216/440473133 [00:03<00:04, 53886045.50B/s] 44%|████▍     | 194968576/440473133 [00:03<00:04, 53489235.34B/s] 45%|████▌     | 200320000/440473133 [00:03<00:04, 53296035.61B/s] 47%|████▋     | 205755392/440473133 [00:03<00:04, 53607743.25B/s] 48%|████▊     | 211118080/440473133 [00:04<00:04, 53607556.47B/s] 49%|████▉     | 216480768/440473133 [00:04<00:04, 53443077.64B/s] 50%|█████     | 221826048/440473133 [00:04<00:04, 53239414.54B/s] 52%|█████▏    | 227170304/440473133 [00:04<00:04, 53296143.02B/s] 53%|█████▎    | 232582144/440473133 [00:04<00:03, 53538797.97B/s] 54%|█████▍    | 238005248/440473133 [00:04<00:03, 53742908.79B/s] 55%|█████▌    | 243403776/440473133 [00:04<00:03, 53813488.22B/s] 56%|█████▋    | 248785920/440473133 [00:04<00:03, 53762289.85B/s] 58%|█████▊    | 254162944/440473133 [00:04<00:03, 53511645.08B/s] 59%|█████▉    | 259621888/440473133 [00:04<00:03, 53826061.59B/s] 60%|██████    | 265013248/440473133 [00:05<00:03, 53849961.81B/s] 61%|██████▏   | 270694400/440473133 [00:05<00:03, 54703888.38B/s] 63%|██████▎   | 276347904/440473133 [00:05<00:02, 55240401.66B/s] 64%|██████▍   | 281876480/440473133 [00:05<00:02, 53900592.26B/s] 65%|██████▌   | 287277056/440473133 [00:05<00:02, 53310142.56B/s] 66%|██████▋   | 292617216/440473133 [00:05<00:02, 52982936.99B/s] 68%|██████▊   | 297922560/440473133 [00:05<00:02, 52519943.22B/s] 69%|██████▉   | 303276032/440473133 [00:05<00:02, 52818688.50B/s] 70%|███████   | 308687872/440473133 [00:05<00:02, 53200779.95B/s] 71%|███████▏  | 314219520/440473133 [00:05<00:02, 53818127.33B/s] 73%|███████▎  | 319804416/440473133 [00:06<00:02, 54410765.48B/s] 74%|███████▍  | 325250048/440473133 [00:06<00:02, 54334950.57B/s] 75%|███████▌  | 330766336/440473133 [00:06<00:02, 54578560.50B/s] 76%|███████▋  | 336227328/440473133 [00:06<00:01, 54321241.01B/s] 78%|███████▊  | 341818368/440473133 [00:06<00:01, 54788244.95B/s] 79%|███████▉  | 347299840/440473133 [00:06<00:01, 54524198.14B/s] 80%|████████  | 352754688/440473133 [00:06<00:01, 54436754.81B/s] 81%|████████▏ | 358200320/440473133 [00:06<00:01, 54176993.69B/s] 83%|████████▎ | 363619328/440473133 [00:06<00:01, 53209766.99B/s] 84%|████████▍ | 368945152/440473133 [00:06<00:01, 51080457.61B/s] 85%|████████▍ | 374133760/440473133 [00:07<00:01, 51317232.57B/s] 86%|████████▌ | 379369472/440473133 [00:07<00:01, 51621245.90B/s] 87%|████████▋ | 384543744/440473133 [00:07<00:01, 51585186.67B/s] 88%|████████▊ | 389710848/440473133 [00:07<00:00, 51492655.41B/s] 90%|████████▉ | 394954752/440473133 [00:07<00:00, 51771004.95B/s] 91%|█████████ | 400324608/440473133 [00:07<00:00, 52333649.22B/s] 92%|█████████▏| 405563392/440473133 [00:07<00:00, 52073929.21B/s] 93%|█████████▎| 410923008/440473133 [00:07<00:00, 52520057.11B/s] 94%|█████████▍| 416179200/440473133 [00:07<00:00, 52254333.86B/s] 96%|█████████▌| 421862400/440473133 [00:07<00:00, 53545737.69B/s] 97%|█████████▋| 427227136/440473133 [00:08<00:00, 52551047.92B/s] 98%|█████████▊| 432596992/440473133 [00:08<00:00, 52889127.64B/s] 99%|█████████▉| 437894144/440473133 [00:08<00:00, 52526310.83B/s]100%|██████████| 440473133/440473133 [00:08<00:00, 52833780.70B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
  65536/7094233 [..............................] - ETA: 5s
 319488/7094233 [>.............................] - ETA: 2s
1212416/7094233 [====>.........................] - ETA: 0s
4603904/7094233 [==================>...........] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:  20%|█▉        | 421/2118 [00:00<00:00, 4205.19it/s]Processing text_left with encode:  41%|████▏     | 878/2118 [00:00<00:00, 4307.63it/s]Processing text_left with encode:  62%|██████▏   | 1309/2118 [00:00<00:00, 4307.64it/s]Processing text_left with encode:  85%|████████▌ | 1802/2118 [00:00<00:00, 4476.77it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 4537.65it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 150/18841 [00:00<00:26, 698.42it/s]Processing text_right with encode:   1%|          | 171/18841 [00:00<02:53, 107.71it/s]Processing text_right with encode:   2%|▏         | 364/18841 [00:00<02:02, 150.24it/s]Processing text_right with encode:   3%|▎         | 535/18841 [00:00<01:28, 206.83it/s]Processing text_right with encode:   4%|▎         | 681/18841 [00:01<01:05, 278.44it/s]Processing text_right with encode:   4%|▍         | 847/18841 [00:01<00:48, 370.97it/s]Processing text_right with encode:   5%|▌         | 1023/18841 [00:01<00:36, 485.96it/s]Processing text_right with encode:   6%|▌         | 1166/18841 [00:01<00:29, 600.75it/s]Processing text_right with encode:   7%|▋         | 1328/18841 [00:01<00:23, 739.89it/s]Processing text_right with encode:   8%|▊         | 1501/18841 [00:01<00:19, 892.28it/s]Processing text_right with encode:   9%|▉         | 1696/18841 [00:01<00:16, 1064.73it/s]Processing text_right with encode:  10%|▉         | 1874/18841 [00:01<00:14, 1209.22it/s]Processing text_right with encode:  11%|█         | 2048/18841 [00:01<00:12, 1329.96it/s]Processing text_right with encode:  12%|█▏        | 2233/18841 [00:02<00:11, 1452.16it/s]Processing text_right with encode:  13%|█▎        | 2419/18841 [00:02<00:10, 1551.81it/s]Processing text_right with encode:  14%|█▍        | 2606/18841 [00:02<00:09, 1634.14it/s]Processing text_right with encode:  15%|█▍        | 2821/18841 [00:02<00:09, 1759.16it/s]Processing text_right with encode:  16%|█▌        | 3012/18841 [00:02<00:08, 1769.56it/s]Processing text_right with encode:  17%|█▋        | 3200/18841 [00:02<00:09, 1671.45it/s]Processing text_right with encode:  18%|█▊        | 3376/18841 [00:02<00:09, 1668.81it/s]Processing text_right with encode:  19%|█▉        | 3549/18841 [00:02<00:09, 1672.27it/s]Processing text_right with encode:  20%|█▉        | 3730/18841 [00:02<00:08, 1709.36it/s]Processing text_right with encode:  21%|██        | 3921/18841 [00:02<00:08, 1762.09it/s]Processing text_right with encode:  22%|██▏       | 4112/18841 [00:03<00:08, 1802.02it/s]Processing text_right with encode:  23%|██▎       | 4295/18841 [00:03<00:08, 1800.89it/s]Processing text_right with encode:  24%|██▍       | 4477/18841 [00:03<00:08, 1793.86it/s]Processing text_right with encode:  25%|██▍       | 4667/18841 [00:03<00:07, 1824.05it/s]Processing text_right with encode:  26%|██▌       | 4861/18841 [00:03<00:07, 1855.71it/s]Processing text_right with encode:  27%|██▋       | 5048/18841 [00:03<00:07, 1832.90it/s]Processing text_right with encode:  28%|██▊       | 5242/18841 [00:03<00:07, 1863.23it/s]Processing text_right with encode:  29%|██▉       | 5429/18841 [00:03<00:07, 1843.27it/s]Processing text_right with encode:  30%|██▉       | 5621/18841 [00:03<00:07, 1863.93it/s]Processing text_right with encode:  31%|███       | 5808/18841 [00:03<00:07, 1858.54it/s]Processing text_right with encode:  32%|███▏      | 5997/18841 [00:04<00:06, 1867.74it/s]Processing text_right with encode:  33%|███▎      | 6184/18841 [00:04<00:07, 1807.56it/s]Processing text_right with encode:  34%|███▍      | 6375/18841 [00:04<00:06, 1833.80it/s]Processing text_right with encode:  35%|███▍      | 6586/18841 [00:04<00:06, 1905.31it/s]Processing text_right with encode:  36%|███▌      | 6778/18841 [00:04<00:06, 1889.62it/s]Processing text_right with encode:  37%|███▋      | 6968/18841 [00:04<00:06, 1837.82it/s]Processing text_right with encode:  38%|███▊      | 7153/18841 [00:04<00:06, 1832.88it/s]Processing text_right with encode:  39%|███▉      | 7349/18841 [00:04<00:06, 1868.19it/s]Processing text_right with encode:  40%|████      | 7538/18841 [00:04<00:06, 1872.99it/s]Processing text_right with encode:  41%|████      | 7736/18841 [00:04<00:05, 1900.59it/s]Processing text_right with encode:  42%|████▏     | 7927/18841 [00:05<00:05, 1849.94it/s]Processing text_right with encode:  43%|████▎     | 8120/18841 [00:05<00:05, 1872.38it/s]Processing text_right with encode:  44%|████▍     | 8314/18841 [00:05<00:05, 1888.76it/s]Processing text_right with encode:  45%|████▌     | 8504/18841 [00:05<00:05, 1855.26it/s]Processing text_right with encode:  46%|████▌     | 8694/18841 [00:05<00:05, 1867.59it/s]Processing text_right with encode:  47%|████▋     | 8887/18841 [00:05<00:05, 1884.91it/s]Processing text_right with encode:  48%|████▊     | 9076/18841 [00:05<00:05, 1865.34it/s]Processing text_right with encode:  49%|████▉     | 9279/18841 [00:05<00:05, 1911.68it/s]Processing text_right with encode:  50%|█████     | 9471/18841 [00:05<00:04, 1906.77it/s]Processing text_right with encode:  51%|█████▏    | 9662/18841 [00:06<00:04, 1896.97it/s]Processing text_right with encode:  52%|█████▏    | 9852/18841 [00:06<00:04, 1817.08it/s]Processing text_right with encode:  53%|█████▎    | 10037/18841 [00:06<00:04, 1825.61it/s]Processing text_right with encode:  54%|█████▍    | 10221/18841 [00:06<00:04, 1772.22it/s]Processing text_right with encode:  55%|█████▌    | 10439/18841 [00:06<00:04, 1876.07it/s]Processing text_right with encode:  56%|█████▋    | 10629/18841 [00:06<00:04, 1805.91it/s]Processing text_right with encode:  57%|█████▋    | 10815/18841 [00:06<00:04, 1820.96it/s]Processing text_right with encode:  58%|█████▊    | 11000/18841 [00:06<00:04, 1828.61it/s]Processing text_right with encode:  59%|█████▉    | 11184/18841 [00:06<00:04, 1820.97it/s]Processing text_right with encode:  60%|██████    | 11367/18841 [00:06<00:04, 1822.23it/s]Processing text_right with encode:  61%|██████▏   | 11550/18841 [00:07<00:04, 1812.71it/s]Processing text_right with encode:  62%|██████▏   | 11732/18841 [00:07<00:03, 1812.30it/s]Processing text_right with encode:  63%|██████▎   | 11914/18841 [00:07<00:03, 1774.85it/s]Processing text_right with encode:  64%|██████▍   | 12110/18841 [00:07<00:03, 1825.77it/s]Processing text_right with encode:  65%|██████▌   | 12294/18841 [00:07<00:03, 1797.20it/s]Processing text_right with encode:  66%|██████▌   | 12475/18841 [00:07<00:03, 1750.31it/s]Processing text_right with encode:  67%|██████▋   | 12651/18841 [00:07<00:03, 1739.83it/s]Processing text_right with encode:  68%|██████▊   | 12840/18841 [00:07<00:03, 1782.00it/s]Processing text_right with encode:  69%|██████▉   | 13019/18841 [00:07<00:03, 1776.61it/s]Processing text_right with encode:  70%|███████   | 13198/18841 [00:07<00:03, 1760.48it/s]Processing text_right with encode:  71%|███████   | 13386/18841 [00:08<00:03, 1792.55it/s]Processing text_right with encode:  72%|███████▏  | 13572/18841 [00:08<00:02, 1809.69it/s]Processing text_right with encode:  73%|███████▎  | 13777/18841 [00:08<00:02, 1875.37it/s]Processing text_right with encode:  74%|███████▍  | 13979/18841 [00:08<00:02, 1916.33it/s]Processing text_right with encode:  75%|███████▌  | 14172/18841 [00:08<00:02, 1846.95it/s]Processing text_right with encode:  76%|███████▌  | 14358/18841 [00:08<00:02, 1772.83it/s]Processing text_right with encode:  77%|███████▋  | 14537/18841 [00:08<00:02, 1759.60it/s]Processing text_right with encode:  78%|███████▊  | 14744/18841 [00:08<00:02, 1841.82it/s]Processing text_right with encode:  79%|███████▉  | 14930/18841 [00:08<00:02, 1804.90it/s]Processing text_right with encode:  80%|████████  | 15115/18841 [00:09<00:02, 1816.83it/s]Processing text_right with encode:  81%|████████  | 15298/18841 [00:09<00:02, 1758.63it/s]Processing text_right with encode:  82%|████████▏ | 15475/18841 [00:09<00:02, 1663.85it/s]Processing text_right with encode:  83%|████████▎ | 15644/18841 [00:09<00:01, 1657.63it/s]Processing text_right with encode:  84%|████████▍ | 15812/18841 [00:09<00:01, 1656.72it/s]Processing text_right with encode:  85%|████████▍ | 15979/18841 [00:09<00:01, 1611.57it/s]Processing text_right with encode:  86%|████████▌ | 16155/18841 [00:09<00:01, 1651.41it/s]Processing text_right with encode:  87%|████████▋ | 16344/18841 [00:09<00:01, 1713.32it/s]Processing text_right with encode:  88%|████████▊ | 16527/18841 [00:09<00:01, 1745.75it/s]Processing text_right with encode:  89%|████████▊ | 16719/18841 [00:09<00:01, 1793.42it/s]Processing text_right with encode:  90%|████████▉ | 16908/18841 [00:10<00:01, 1818.73it/s]Processing text_right with encode:  91%|█████████ | 17091/18841 [00:10<00:00, 1801.27it/s]Processing text_right with encode:  92%|█████████▏| 17276/18841 [00:10<00:00, 1814.93it/s]Processing text_right with encode:  93%|█████████▎| 17458/18841 [00:10<00:00, 1800.02it/s]Processing text_right with encode:  94%|█████████▎| 17642/18841 [00:10<00:00, 1808.14it/s]Processing text_right with encode:  95%|█████████▍| 17824/18841 [00:10<00:00, 1787.33it/s]Processing text_right with encode:  96%|█████████▌| 18003/18841 [00:10<00:00, 1785.10it/s]Processing text_right with encode:  97%|█████████▋| 18219/18841 [00:10<00:00, 1880.91it/s]Processing text_right with encode:  98%|█████████▊| 18420/18841 [00:10<00:00, 1915.70it/s]Processing text_right with encode:  99%|█████████▉| 18613/18841 [00:11<00:00, 1898.76it/s]Processing text_right with encode: 100%|█████████▉| 18804/18841 [00:11<00:00, 1831.52it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:11<00:00, 1691.34it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 643501.33it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 724925.76it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  74%|███████▍  | 467/633 [00:00<00:00, 4661.60it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4651.69it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 192/5961 [00:00<00:03, 1915.81it/s]Processing text_right with encode:   6%|▋         | 385/5961 [00:00<00:02, 1918.50it/s]Processing text_right with encode:  10%|▉         | 581/5961 [00:00<00:02, 1926.06it/s]Processing text_right with encode:  13%|█▎        | 777/5961 [00:00<00:02, 1933.89it/s]Processing text_right with encode:  16%|█▋        | 983/5961 [00:00<00:02, 1968.94it/s]Processing text_right with encode:  20%|█▉        | 1187/5961 [00:00<00:02, 1989.53it/s]Processing text_right with encode:  23%|██▎       | 1391/5961 [00:00<00:02, 2001.60it/s]Processing text_right with encode:  27%|██▋       | 1608/5961 [00:00<00:02, 2047.68it/s]Processing text_right with encode:  30%|███       | 1803/5961 [00:00<00:02, 1992.16it/s]Processing text_right with encode:  34%|███▎      | 2000/5961 [00:01<00:01, 1985.19it/s]Processing text_right with encode:  37%|███▋      | 2221/5961 [00:01<00:01, 2044.40it/s]Processing text_right with encode:  41%|████      | 2423/5961 [00:01<00:01, 1970.17it/s]Processing text_right with encode:  44%|████▍     | 2626/5961 [00:01<00:01, 1986.48it/s]Processing text_right with encode:  48%|████▊     | 2845/5961 [00:01<00:01, 2043.37it/s]Processing text_right with encode:  51%|█████     | 3050/5961 [00:01<00:01, 2006.90it/s]Processing text_right with encode:  55%|█████▍    | 3251/5961 [00:01<00:01, 1926.00it/s]Processing text_right with encode:  58%|█████▊    | 3445/5961 [00:01<00:01, 1803.05it/s]Processing text_right with encode:  61%|██████    | 3633/5961 [00:01<00:01, 1823.02it/s]Processing text_right with encode:  64%|██████▍   | 3823/5961 [00:01<00:01, 1843.08it/s]Processing text_right with encode:  68%|██████▊   | 4040/5961 [00:02<00:00, 1930.03it/s]Processing text_right with encode:  71%|███████   | 4246/5961 [00:02<00:00, 1963.64it/s]Processing text_right with encode:  75%|███████▍  | 4446/5961 [00:02<00:00, 1973.70it/s]Processing text_right with encode:  78%|███████▊  | 4645/5961 [00:02<00:00, 1967.80it/s]Processing text_right with encode:  81%|████████  | 4843/5961 [00:02<00:00, 1882.63it/s]Processing text_right with encode:  84%|████████▍ | 5036/5961 [00:02<00:00, 1895.50it/s]Processing text_right with encode:  88%|████████▊ | 5238/5961 [00:02<00:00, 1928.81it/s]Processing text_right with encode:  91%|█████████ | 5432/5961 [00:02<00:00, 1901.72it/s]Processing text_right with encode:  94%|█████████▍| 5623/5961 [00:02<00:00, 1894.60it/s]Processing text_right with encode:  98%|█████████▊| 5818/5961 [00:02<00:00, 1910.57it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1954.06it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 575047.53it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 896073.62it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:39<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:39<?, ?it/s, loss=0.973]Epoch 1/1:   1%|          | 1/102 [00:39<1:06:42, 39.62s/it, loss=0.973]Epoch 1/1:   1%|          | 1/102 [02:03<1:06:42, 39.62s/it, loss=0.973]Epoch 1/1:   1%|          | 1/102 [02:03<1:06:42, 39.62s/it, loss=1.075]Epoch 1/1:   2%|▏         | 2/102 [02:03<1:28:11, 52.91s/it, loss=1.075]Epoch 1/1:   2%|▏         | 2/102 [03:04<1:28:11, 52.91s/it, loss=1.075]Epoch 1/1:   2%|▏         | 2/102 [03:04<1:28:11, 52.91s/it, loss=1.031]Epoch 1/1:   3%|▎         | 3/102 [03:04<1:31:22, 55.38s/it, loss=1.031]Epoch 1/1:   3%|▎         | 3/102 [03:56<1:31:22, 55.38s/it, loss=1.031]Epoch 1/1:   3%|▎         | 3/102 [03:56<1:31:22, 55.38s/it, loss=0.786]Epoch 1/1:   4%|▍         | 4/102 [03:56<1:28:44, 54.33s/it, loss=0.786]Epoch 1/1:   4%|▍         | 4/102 [04:34<1:28:44, 54.33s/it, loss=0.786]Epoch 1/1:   4%|▍         | 4/102 [04:34<1:28:44, 54.33s/it, loss=0.861]Epoch 1/1:   5%|▍         | 5/102 [04:34<1:20:04, 49.53s/it, loss=0.861]Epoch 1/1:   5%|▍         | 5/102 [07:18<1:20:04, 49.53s/it, loss=0.861]Epoch 1/1:   5%|▍         | 5/102 [07:18<1:20:04, 49.53s/it, loss=1.019]Epoch 1/1:   6%|▌         | 6/102 [07:18<2:13:51, 83.66s/it, loss=1.019]Epoch 1/1:   6%|▌         | 6/102 [08:36<2:13:51, 83.66s/it, loss=1.019]Epoch 1/1:   6%|▌         | 6/102 [08:36<2:13:51, 83.66s/it, loss=0.860]Epoch 1/1:   7%|▋         | 7/102 [08:36<2:09:48, 81.99s/it, loss=0.860]Epoch 1/1:   7%|▋         | 7/102 [11:21<2:09:48, 81.99s/it, loss=0.860]Epoch 1/1:   7%|▋         | 7/102 [11:21<2:09:48, 81.99s/it, loss=0.808]Epoch 1/1:   8%|▊         | 8/102 [11:21<2:47:26, 106.88s/it, loss=0.808]Killed

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   494606e..f0cbead  master     -> origin/master
Updating 494606e..f0cbead
Fast-forward
 error_list/20200513/list_log_benchmark_20200513.md |  166 +-
 error_list/20200513/list_log_import_20200513.md    |    2 +-
 error_list/20200513/list_log_json_20200513.md      |  276 +--
 error_list/20200513/list_log_jupyter_20200513.md   | 2264 +++++++++---------
 .../20200513/list_log_pullrequest_20200513.md      |    2 +-
 error_list/20200513/list_log_test_cli_20200513.md  |  152 +-
 error_list/20200513/list_log_testall_20200513.md   |  320 ++-
 ...-13_6672e19fe4cfa7df885e45d91d645534b8989485.py | 2459 ++++++++++++++++++++
 8 files changed, 4097 insertions(+), 1544 deletions(-)
 create mode 100644 log_benchmark/log_benchmark_2020-05-13-08-13_6672e19fe4cfa7df885e45d91d645534b8989485.py
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
