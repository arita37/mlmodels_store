## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py


### Error 1, [Traceback at line 37](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L37)<br />37..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//keras_gan.py", line 31, in <module>
<br />    'AAE' : kg.aae.aae,
<br />AttributeError: module 'mlmodels.model_keras.raw.keras_gan' has no attribute 'aae'



### Error 2, [Traceback at line 80](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L80)<br />80..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//textcnn_dataloader.py", line 275, in <module>
<br />    test_module(model_uri = MODEL_URI, param_pars= param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras/textcnn_dataloader.py", line 182, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/dataset/json/refactor/textcnn_keras.json'



### Error 3, [Traceback at line 127](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L127)<br />127..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//nbeats.py", line 315, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//nbeats.py", line 278, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//nbeats.py", line 172, in get_dataset
<br />    train_data = Data(data_source= path_norm( data_pars["train_data_source"]) ,
<br />NameError: name 'Data' is not defined



### Error 4, [Traceback at line 5875](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L5875)<br />5875..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//textvae.py", line 356, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//textvae.py", line 327, in test
<br />    xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/dataset/text/quora/train.csv'



### Error 5, [Traceback at line 5923](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L5923)<br />5923..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//namentity_crm_bilstm_dataloader.py", line 306, in <module>
<br />    test_module(model_uri=MODEL_URI, param_pars=param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras/namentity_crm_bilstm_dataloader.py", line 197, in get_params
<br />    cf = json.load(open(data_path, mode="r"))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader.json'



### Error 6, [Traceback at line 5962](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L5962)<br />5962..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 7, [Traceback at line 6101](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L6101)<br />6101..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//charcnn_zhang.py", line 284, in <module>
<br />    test(pars_choice="json", data_path= f"{root_path}/model_keras/charcnn_zhang.json")
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//charcnn_zhang.py", line 268, in test
<br />    model2 = load(out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//charcnn_zhang.py", line 118, in load
<br />    model = load_keras(load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/util.py", line 602, in load_keras
<br />    model.model = load_model(path_file)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/keras/saving/save.py", line 146, in load_model
<br />    loader_impl.parse_saved_model(filepath)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/saved_model/loader_impl.py", line 83, in parse_saved_model
<br />    constants.SAVED_MODEL_FILENAME_PB))
<br />OSError: SavedModel file does not exist at: ztest/ml_keras/charcnn_zhang//model.h5/{saved_model.pbtxt|saved_model.pb}



### Error 8, [Traceback at line 6154](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L6154)<br />6154..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//charcnn.py", line 357, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//charcnn.py", line 320, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//charcnn.py", line 216, in get_dataset
<br />    if data_pars['type'] == "npz":
<br />KeyError: 'type'



### Error 9, [Traceback at line 6198](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L6198)<br />6198..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//namentity_crm_bilstm.py", line 348, in <module>
<br />    test(pars_choice="json", data_path=f"model_keras/namentity_crm_bilstm.json")
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//namentity_crm_bilstm.py", line 311, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//namentity_crm_bilstm.py", line 193, in get_dataset
<br />    raise Exception(f"Not support dataset yet")
<br />Exception: Not support dataset yet
<br />
<br />   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
<br />Fetching origin
<br />Already up to date.
<br />Logs
<br />README.md
<br />README_actions.md
<br />create_error_file.py
<br />create_github_issues.py
<br />error_list
<br />log_benchmark
<br />log_dataloader
<br />log_import
<br />log_json
<br />log_jupyter
<br />log_pullrequest
<br />log_test_cli
<br />log_testall
<br />test_jupyter
<br />[master 5490d92] ml_store
<br /> 1 file changed, 44 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   b7f7347..5490d92  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//textcnn.py 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  #### Path params   ########################################## 
<br />
<br />  #### Loading dataset   ############################################# 
<br />Loading data...
<br />Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz
<br />
<br />    8192/17464789 [..............................] - ETA: 0s
<br /> 3260416/17464789 [====>.........................] - ETA: 0s
<br />10518528/17464789 [=================>............] - ETA: 0s
<br />16760832/17464789 [===========================>..] - ETA: 0s
<br />17465344/17464789 [==============================] - 0s 0us/step
<br />Pad sequences (samples x time)...
<br />
<br />  #### Model init, fit   ############################################# 
<br />Using TensorFlow backend.
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />If using Keras pass *_constraint arguments to layers.
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />Use tf.where in 2.0, which has the same broadcast rule as np.where
<br />2020-05-13 08:28:24.390477: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
<br />2020-05-13 08:28:24.394389: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
<br />2020-05-13 08:28:24.394522: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56059e731140 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
<br />2020-05-13 08:28:24.394537: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
<br />
<br />Model: "model_1"
<br />__________________________________________________________________________________________________
<br />Layer (type)                    Output Shape         Param #     Connected to                     
<br />==================================================================================================
<br />input_1 (InputLayer)            (None, 40)           0                                            
<br />__________________________________________________________________________________________________
<br />embedding_1 (Embedding)         (None, 40, 50)       250         input_1[0][0]                    
<br />__________________________________________________________________________________________________
<br />conv1d_1 (Conv1D)               (None, 38, 128)      19328       embedding_1[0][0]                
<br />__________________________________________________________________________________________________
<br />conv1d_2 (Conv1D)               (None, 37, 128)      25728       embedding_1[0][0]                
<br />__________________________________________________________________________________________________
<br />conv1d_3 (Conv1D)               (None, 36, 128)      32128       embedding_1[0][0]                
<br />__________________________________________________________________________________________________
<br />global_max_pooling1d_1 (GlobalM (None, 128)          0           conv1d_1[0][0]                   
<br />__________________________________________________________________________________________________
<br />global_max_pooling1d_2 (GlobalM (None, 128)          0           conv1d_2[0][0]                   
<br />__________________________________________________________________________________________________
<br />global_max_pooling1d_3 (GlobalM (None, 128)          0           conv1d_3[0][0]                   
<br />__________________________________________________________________________________________________
<br />concatenate_1 (Concatenate)     (None, 384)          0           global_max_pooling1d_1[0][0]     
<br />                                                                 global_max_pooling1d_2[0][0]     
<br />                                                                 global_max_pooling1d_3[0][0]     
<br />__________________________________________________________________________________________________
<br />dense_1 (Dense)                 (None, 1)            385         concatenate_1[0][0]              
<br />==================================================================================================
<br />Total params: 77,819
<br />Trainable params: 77,819
<br />Non-trainable params: 0
<br />__________________________________________________________________________________________________
<br />Loading data...
<br />Pad sequences (samples x time)...
<br />Train on 25000 samples, validate on 25000 samples
<br />Epoch 1/1
<br />
<br /> 1000/25000 [>.............................] - ETA: 12s - loss: 7.6360 - accuracy: 0.5020
<br /> 2000/25000 [=>............................] - ETA: 9s - loss: 7.5516 - accuracy: 0.5075 
<br /> 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6104 - accuracy: 0.5037
<br /> 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6705 - accuracy: 0.4997
<br /> 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6053 - accuracy: 0.5040
<br /> 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5261 - accuracy: 0.5092
<br /> 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5001 - accuracy: 0.5109
<br /> 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5420 - accuracy: 0.5081
<br /> 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5746 - accuracy: 0.5060
<br />10000/25000 [===========>..................] - ETA: 4s - loss: 7.5700 - accuracy: 0.5063
<br />11000/25000 [============>.................] - ETA: 4s - loss: 7.5955 - accuracy: 0.5046
<br />12000/25000 [=============>................] - ETA: 4s - loss: 7.6372 - accuracy: 0.5019
<br />13000/25000 [==============>...............] - ETA: 3s - loss: 7.6371 - accuracy: 0.5019
<br />14000/25000 [===============>..............] - ETA: 3s - loss: 7.6535 - accuracy: 0.5009
<br />15000/25000 [=================>............] - ETA: 3s - loss: 7.6503 - accuracy: 0.5011
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.6340 - accuracy: 0.5021
<br />17000/25000 [===================>..........] - ETA: 2s - loss: 7.6486 - accuracy: 0.5012
<br />18000/25000 [====================>.........] - ETA: 2s - loss: 7.6504 - accuracy: 0.5011
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6561 - accuracy: 0.5007
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
<br />21000/25000 [========================>.....] - ETA: 1s - loss: 7.6506 - accuracy: 0.5010
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6541 - accuracy: 0.5008
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6602 - accuracy: 0.5004
<br />25000/25000 [==============================] - 9s 379us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
<br />
<br />  #### save the trained model  ####################################### 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/ztest/model_keras/textcnn/model.h5'}
<br />
<br />  #### Predict   ##################################################### 
<br />Loading data...
<br />
<br />  #### metrics   ##################################################### 
<br />{}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/keras/initializers.py:119: calling RandomUniform.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />Call initializer instance with the dtype argument instead of passing it to the constructor
<br />{'path': 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/ztest/model_keras/textcnn/model.h5'}
<br />{'path': 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/ztest/model_keras/textcnn/model.h5'}
<br />(<mlmodels.util.Model_empty object at 0x7f6d619b1320>, None)
<br />
<br />  #### Module init   ############################################ 
<br />
<br />  <module 'mlmodels.model_keras.textcnn' from 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras/textcnn.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  #### Path params   ########################################## 
<br />
<br />  #### Model init   ############################################ 
<br />Model: "model_2"
<br />__________________________________________________________________________________________________
<br />Layer (type)                    Output Shape         Param #     Connected to                     
<br />==================================================================================================
<br />input_2 (InputLayer)            (None, 40)           0                                            
<br />__________________________________________________________________________________________________
<br />embedding_2 (Embedding)         (None, 40, 50)       250         input_2[0][0]                    
<br />__________________________________________________________________________________________________
<br />conv1d_4 (Conv1D)               (None, 38, 128)      19328       embedding_2[0][0]                
<br />__________________________________________________________________________________________________
<br />conv1d_5 (Conv1D)               (None, 37, 128)      25728       embedding_2[0][0]                
<br />__________________________________________________________________________________________________
<br />conv1d_6 (Conv1D)               (None, 36, 128)      32128       embedding_2[0][0]                
<br />__________________________________________________________________________________________________
<br />global_max_pooling1d_4 (GlobalM (None, 128)          0           conv1d_4[0][0]                   
<br />__________________________________________________________________________________________________
<br />global_max_pooling1d_5 (GlobalM (None, 128)          0           conv1d_5[0][0]                   
<br />__________________________________________________________________________________________________
<br />global_max_pooling1d_6 (GlobalM (None, 128)          0           conv1d_6[0][0]                   
<br />__________________________________________________________________________________________________
<br />concatenate_2 (Concatenate)     (None, 384)          0           global_max_pooling1d_4[0][0]     
<br />                                                                 global_max_pooling1d_5[0][0]     
<br />                                                                 global_max_pooling1d_6[0][0]     
<br />__________________________________________________________________________________________________
<br />dense_2 (Dense)                 (None, 1)            385         concatenate_2[0][0]              
<br />==================================================================================================
<br />Total params: 77,819
<br />Trainable params: 77,819
<br />Non-trainable params: 0
<br />__________________________________________________________________________________________________
<br />
<br />  <mlmodels.model_keras.textcnn.Model object at 0x7f6d604b5cf8> 
<br />
<br />  #### Fit   ######################################################## 
<br />Loading data...
<br />Pad sequences (samples x time)...
<br />Train on 25000 samples, validate on 25000 samples
<br />Epoch 1/1
<br />
<br /> 1000/25000 [>.............................] - ETA: 12s - loss: 7.6513 - accuracy: 0.5010
<br /> 2000/25000 [=>............................] - ETA: 9s - loss: 7.4596 - accuracy: 0.5135 
<br /> 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5797 - accuracy: 0.5057
<br /> 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5785 - accuracy: 0.5058
<br /> 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6789 - accuracy: 0.4992
<br /> 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6845 - accuracy: 0.4988
<br /> 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6316 - accuracy: 0.5023
<br /> 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6091 - accuracy: 0.5038
<br /> 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5831 - accuracy: 0.5054
<br />10000/25000 [===========>..................] - ETA: 4s - loss: 7.5516 - accuracy: 0.5075
<br />11000/25000 [============>.................] - ETA: 4s - loss: 7.5858 - accuracy: 0.5053
<br />12000/25000 [=============>................] - ETA: 3s - loss: 7.6053 - accuracy: 0.5040
<br />13000/25000 [==============>...............] - ETA: 3s - loss: 7.5817 - accuracy: 0.5055
<br />14000/25000 [===============>..............] - ETA: 3s - loss: 7.5812 - accuracy: 0.5056
<br />15000/25000 [=================>............] - ETA: 3s - loss: 7.5889 - accuracy: 0.5051
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.5842 - accuracy: 0.5054
<br />17000/25000 [===================>..........] - ETA: 2s - loss: 7.6035 - accuracy: 0.5041
<br />18000/25000 [====================>.........] - ETA: 2s - loss: 7.6053 - accuracy: 0.5040
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6174 - accuracy: 0.5032
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6183 - accuracy: 0.5031
<br />21000/25000 [========================>.....] - ETA: 1s - loss: 7.6433 - accuracy: 0.5015
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6457 - accuracy: 0.5014
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6506 - accuracy: 0.5010
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
<br />25000/25000 [==============================] - 9s 360us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
<br />
<br />  #### Predict   #################################################### 
<br />Loading data...
<br />(array([[1.],
<br />       [1.],
<br />       [1.],
<br />       ...,
<br />       [1.],
<br />       [1.],
<br />       [1.]], dtype=float32), None)
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />
<br />  ############ Model preparation   ################################## 
<br />
<br />  #### Module init   ############################################ 
<br />
<br />  <module 'mlmodels.model_keras.textcnn' from 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras/textcnn.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  #### Path params   ########################################## 
<br />
<br />  #### Model init   ############################################ 
<br />Model: "model_3"
<br />__________________________________________________________________________________________________
<br />Layer (type)                    Output Shape         Param #     Connected to                     
<br />==================================================================================================
<br />input_3 (InputLayer)            (None, 40)           0                                            
<br />__________________________________________________________________________________________________
<br />embedding_3 (Embedding)         (None, 40, 50)       250         input_3[0][0]                    
<br />__________________________________________________________________________________________________
<br />conv1d_7 (Conv1D)               (None, 38, 128)      19328       embedding_3[0][0]                
<br />__________________________________________________________________________________________________
<br />conv1d_8 (Conv1D)               (None, 37, 128)      25728       embedding_3[0][0]                
<br />__________________________________________________________________________________________________
<br />conv1d_9 (Conv1D)               (None, 36, 128)      32128       embedding_3[0][0]                
<br />__________________________________________________________________________________________________
<br />global_max_pooling1d_7 (GlobalM (None, 128)          0           conv1d_7[0][0]                   
<br />__________________________________________________________________________________________________
<br />global_max_pooling1d_8 (GlobalM (None, 128)          0           conv1d_8[0][0]                   
<br />__________________________________________________________________________________________________
<br />global_max_pooling1d_9 (GlobalM (None, 128)          0           conv1d_9[0][0]                   
<br />__________________________________________________________________________________________________
<br />concatenate_3 (Concatenate)     (None, 384)          0           global_max_pooling1d_7[0][0]     
<br />                                                                 global_max_pooling1d_8[0][0]     
<br />                                                                 global_max_pooling1d_9[0][0]     
<br />__________________________________________________________________________________________________
<br />dense_3 (Dense)                 (None, 1)            385         concatenate_3[0][0]              
<br />==================================================================================================
<br />Total params: 77,819
<br />Trainable params: 77,819
<br />Non-trainable params: 0
<br />__________________________________________________________________________________________________
<br />
<br />  ############ Model fit   ########################################## 
<br />Loading data...
<br />Pad sequences (samples x time)...
<br />Train on 25000 samples, validate on 25000 samples
<br />Epoch 1/1
<br />
<br /> 1000/25000 [>.............................] - ETA: 11s - loss: 7.8660 - accuracy: 0.4870
<br /> 2000/25000 [=>............................] - ETA: 9s - loss: 7.6436 - accuracy: 0.5015 
<br /> 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6717 - accuracy: 0.4997
<br /> 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6245 - accuracy: 0.5027
<br /> 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7402 - accuracy: 0.4952
<br /> 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7407 - accuracy: 0.4952
<br /> 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7017 - accuracy: 0.4977
<br /> 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6743 - accuracy: 0.4995
<br /> 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6581 - accuracy: 0.5006
<br />10000/25000 [===========>..................] - ETA: 4s - loss: 7.6344 - accuracy: 0.5021
<br />11000/25000 [============>.................] - ETA: 4s - loss: 7.6541 - accuracy: 0.5008
<br />12000/25000 [=============>................] - ETA: 3s - loss: 7.6385 - accuracy: 0.5018
<br />13000/25000 [==============>...............] - ETA: 3s - loss: 7.6230 - accuracy: 0.5028
<br />14000/25000 [===============>..............] - ETA: 3s - loss: 7.6228 - accuracy: 0.5029
<br />15000/25000 [=================>............] - ETA: 3s - loss: 7.6564 - accuracy: 0.5007
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.6599 - accuracy: 0.5004
<br />17000/25000 [===================>..........] - ETA: 2s - loss: 7.6558 - accuracy: 0.5007
<br />18000/25000 [====================>.........] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6751 - accuracy: 0.4994
<br />21000/25000 [========================>.....] - ETA: 1s - loss: 7.6637 - accuracy: 0.5002
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
<br />25000/25000 [==============================] - 9s 368us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
<br />fit success None
<br />
<br />  ############ Prediction############################################ 
<br />Loading data...
<br />(array([[1.],
<br />       [1.],
<br />       [1.],
<br />       ...,
<br />       [1.],
<br />       [1.],
<br />       [1.]], dtype=float32), None)
<br />
<br />  ############ Save/ Load ############################################ 
<br />
<br />   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
<br />Fetching origin
<br />Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
<br />From github.com:arita37/mlmodels_store
<br />   5490d92..14480f9  master     -> origin/master
<br />Updating 5490d92..14480f9
<br />Fast-forward
<br /> .../20200513/list_log_pullrequest_20200513.md      |   2 +-
<br /> error_list/20200513/list_log_test_cli_20200513.md  | 138 ++++++++++-----------
<br /> 2 files changed, 70 insertions(+), 70 deletions(-)
<br />Logs
<br />README.md
<br />README_actions.md
<br />create_error_file.py
<br />create_github_issues.py
<br />error_list
<br />log_benchmark
<br />log_dataloader
<br />log_import
<br />log_json
<br />log_jupyter
<br />log_pullrequest
<br />log_test_cli
<br />log_testall
<br />test_jupyter
<br />[master 26b48c7] ml_store
<br /> 1 file changed, 324 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   14480f9..26b48c7  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//armdn.py 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  #### Model init   ################################################## 
<br />Using TensorFlow backend.
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />If using Keras pass *_constraint arguments to layers.
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_probability/python/distributions/mixture.py:154: Categorical.event_size (from tensorflow_probability.python.distributions.categorical) is deprecated and will be removed after 2019-05-19.
<br />Instructions for updating:
<br />The `event_size` property is deprecated.  Use `num_categories` instead.  They have the same value, but `event_size` is misnamed.
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_ops.py:2509: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />Use tf.where in 2.0, which has the same broadcast rule as np.where
<br />Model: "sequential_1"
<br />_________________________________________________________________
<br />Layer (type)                 Output Shape              Param #   
<br />=================================================================
<br />LSTM_1 (LSTM)                (None, 12, 300)           362400    
<br />_________________________________________________________________
<br />LSTM_2 (LSTM)                (None, 12, 200)           400800    
<br />_________________________________________________________________
<br />LSTM_3 (LSTM)                (None, 12, 24)            21600     
<br />_________________________________________________________________
<br />LSTM_4 (LSTM)                (None, 12)                1776      
<br />_________________________________________________________________
<br />dense_1 (Dense)              (None, 10)                130       
<br />_________________________________________________________________
<br />mdn_1 (MDN)                  (None, 75)                825       
<br />=================================================================
<br />Total params: 787,531
<br />Trainable params: 787,531
<br />Non-trainable params: 0
<br />_________________________________________________________________
<br />
<br />  ### Model Fit ###################################################### 
<br />
<br />  #### Loading dataset   ############################################# 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
<br />
<br />Epoch 1/10
<br />
<br />13/13 [==============================] - 1s 109ms/step - loss: nan
<br />Epoch 2/10
<br />
<br />13/13 [==============================] - 0s 5ms/step - loss: nan
<br />Epoch 3/10
<br />
<br />13/13 [==============================] - 0s 4ms/step - loss: nan
<br />Epoch 4/10
<br />
<br />13/13 [==============================] - 0s 4ms/step - loss: nan
<br />Epoch 5/10
<br />
<br />13/13 [==============================] - 0s 4ms/step - loss: nan
<br />Epoch 6/10
<br />
<br />13/13 [==============================] - 0s 4ms/step - loss: nan
<br />Epoch 7/10
<br />
<br />13/13 [==============================] - 0s 4ms/step - loss: nan
<br />Epoch 8/10
<br />
<br />13/13 [==============================] - 0s 4ms/step - loss: nan
<br />Epoch 9/10
<br />
<br />13/13 [==============================] - 0s 4ms/step - loss: nan
<br />Epoch 10/10
<br />
<br />13/13 [==============================] - 0s 4ms/step - loss: nan
<br />
<br />  fitted metrics {'loss': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]} 
<br />
<br />  #### Predict   ##################################################### 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mdn/__init__.py:209: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.
<br />
<br />[[nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
<br />  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
<br />  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
<br />  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
<br />  nan nan nan]]



### Error 10, [Traceback at line 6642](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L6642)<br />6642..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//armdn.py", line 380, in <module>
<br />    test(pars_choice="json", data_path= "model_keras/armdn.json")
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//armdn.py", line 354, in test
<br />    y_pred, y_test = predict(model=model, model_pars=model_pars, data_pars=data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//armdn.py", line 170, in predict
<br />    model.model_pars["n_mixes"], temp=1.0)
<br />  File "<__array_function__ internals>", line 6, in apply_along_axis
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/shape_base.py", line 379, in apply_along_axis
<br />    res = asanyarray(func1d(inarr_view[ind0], *args, **kwargs))
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mdn/__init__.py", line 237, in sample_from_output
<br />    cov_matrix = np.identity(output_dim) * sig_vector
<br />ValueError: operands could not be broadcast together with shapes (12,12) (0,) 



### Error 11, [Traceback at line 8810](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L8810)<br />8810..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1365, in _do_call
<br />    return fn(*args)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1350, in _run_fn
<br />    target_list, run_metadata)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1443, in _call_tf_sessionrun
<br />    run_metadata)
<br />tensorflow.python.framework.errors_impl.NotFoundError: Key Variable not found in checkpoint
<br />	 [[{{node save_1/RestoreV2}}]]
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 12, [Traceback at line 8822](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L8822)<br />8822..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1290, in restore
<br />    {self.saver_def.filename_tensor_name: save_path})
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 956, in run
<br />    run_metadata_ptr)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1180, in _run
<br />    feed_dict_tensor, options, run_metadata)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1359, in _do_run
<br />    run_metadata)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1384, in _do_call
<br />    raise type(e)(node_def, op, message)
<br />tensorflow.python.framework.errors_impl.NotFoundError: Key Variable not found in checkpoint
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
<br />    test(data_path="", pars_choice="test01", config_mode="test")
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
<br />    session = load(out_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
<br />    return load_tf(load_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
<br />    saver      = tf.compat.v1.train.Saver()
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 828, in __init__
<br />    self.build()
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 840, in build
<br />    self._build(self._filename, build_save=True, build_restore=True)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 878, in _build
<br />    build_restore=build_restore)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 508, in _build_internal
<br />    restore_sequentially, reshape)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 328, in _AddRestoreOps
<br />    restore_sequentially)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 575, in bulk_restore
<br />    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_io_ops.py", line 1696, in restore_v2
<br />    name=name)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
<br />    op_def=op_def)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
<br />    return func(*args, **kwargs)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
<br />    attrs, op_def, compute_device)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
<br />    op_def=op_def)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
<br />    self._traceback = tf_stack.extract_stack()
<br />
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 13, [Traceback at line 8873](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L8873)<br />8873..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1300, in restore
<br />    names_to_keys = object_graph_key_mapping(save_path)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1618, in object_graph_key_mapping
<br />    object_graph_string = reader.get_tensor(trackable.OBJECT_GRAPH_PROTO_KEY)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/pywrap_tensorflow_internal.py", line 915, in get_tensor
<br />    return CheckpointReader_GetTensor(self, compat.as_bytes(tensor_str))
<br />tensorflow.python.framework.errors_impl.NotFoundError: Key _CHECKPOINTABLE_OBJECT_GRAPH not found in checkpoint
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 14, [Traceback at line 8884](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L8884)<br />8884..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tf//1_lstm.py", line 332, in <module>
<br />    test(data_path="", pars_choice="test01", config_mode="test")
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tf//1_lstm.py", line 320, in test
<br />    session = load(out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tf//1_lstm.py", line 199, in load
<br />    return load_tf(load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/util.py", line 477, in load_tf
<br />    saver.restore(sess,  full_name)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1306, in restore
<br />    err, "a Variable name or other graph key that is missing")
<br />tensorflow.python.framework.errors_impl.NotFoundError: Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
<br />
<br />Key Variable not found in checkpoint
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
<br />    test(data_path="", pars_choice="test01", config_mode="test")
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
<br />    session = load(out_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
<br />    return load_tf(load_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
<br />    saver      = tf.compat.v1.train.Saver()
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 828, in __init__
<br />    self.build()
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 840, in build
<br />    self._build(self._filename, build_save=True, build_restore=True)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 878, in _build
<br />    build_restore=build_restore)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 508, in _build_internal
<br />    restore_sequentially, reshape)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 328, in _AddRestoreOps
<br />    restore_sequentially)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 575, in bulk_restore
<br />    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_io_ops.py", line 1696, in restore_v2
<br />    name=name)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
<br />    op_def=op_def)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
<br />    return func(*args, **kwargs)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
<br />    attrs, op_def, compute_device)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
<br />    op_def=op_def)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
<br />    self._traceback = tf_stack.extract_stack()
<br />
<br />
<br />   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
<br />Fetching origin
<br />Already up to date.
<br />Logs
<br />README.md
<br />README_actions.md
<br />create_error_file.py
<br />create_github_issues.py
<br />error_list
<br />log_benchmark
<br />log_dataloader
<br />log_import
<br />log_json
<br />log_jupyter
<br />log_pullrequest
<br />log_test_cli
<br />log_testall
<br />test_jupyter
<br />[master 1eb00c6] ml_store
<br /> 1 file changed, 233 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   b625ce5..1eb00c6  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tf//temporal_fusion_google.py 



### Error 15, [Traceback at line 8965](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L8965)<br />8965..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
<br />    from mlmodels.mode_tf.raw  import temporal_fusion_google
<br />ModuleNotFoundError: No module named 'mlmodels.mode_tf'



### Error 16, [Traceback at line 9230](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L9230)<br />9230..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_gluon//fb_prophet.py", line 160, in <module>
<br />    test(data_path = "model_fb/fbprophet.json", choice="json" )
<br />TypeError: test() got an unexpected keyword argument 'choice'



### Error 17, [Traceback at line 9265](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L9265)<br />9265..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_gluon//gluonts_model.py", line 15, in <module>
<br />    from gluonts.model.deepar import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/__init__.py", line 15, in <module>
<br />    from ._estimator import DeepAREstimator
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/model/deepar/_estimator.py", line 24, in <module>
<br />    from gluonts.distribution import DistributionOutput, StudentTOutput
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/__init__.py", line 15, in <module>
<br />    from . import bijection
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 28, in <module>
<br />    class Bijection:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/distribution/bijection.py", line 36, in Bijection
<br />    @validated()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 398, in validator
<br />    **init_fields,
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 18, [Traceback at line 10839](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L10839)<br />10839..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//transformer_classifier.py", line 522, in <module>
<br />    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//transformer_classifier.py", line 418, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'model_tch/transformer_classifier.json'



### Error 19, [Traceback at line 11265](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L11265)<br />11265..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//torchhub.py", line 406, in <module>
<br />    test(data_path="model_tch/torchhub_cnn_list.json", pars_choice="json", config_mode="resnet18")
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//torchhub.py", line 338, in test
<br />    model, session = fit(model, data_pars, compute_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//torchhub.py", line 207, in fit
<br />    tr_loss, tr_acc = _train(model0, device, train_iter, criterion, optimizer, epoch, epochs, imax=imax_train)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//torchhub.py", line 46, in _train
<br />    for i,batch in enumerate(train_itr):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 346, in __next__
<br />    data = self.dataset_fetcher.fetch(index)  # may raise StopIteration
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
<br />    return self.collate_fn(data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in default_collate
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 80, in <listcomp>
<br />    return [default_collate(samples) for samples in transposed]
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/torch/utils/data/_utils/collate.py", line 82, in default_collate
<br />    raise TypeError(default_collate_err_msg_format.format(elem_type))
<br />TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>



### Error 20, [Traceback at line 11316](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L11316)<br />11316..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//03_nbeats_dataloader.py", line 9, in <module>
<br />    from dataloader import DataLoader
<br />ModuleNotFoundError: No module named 'dataloader'



### Error 21, [Traceback at line 11428](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-08-11_6672e19fe4cfa7df885e45d91d645534b8989485.py#L11428)<br />11428..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//pytorch_vae.py", line 34, in <module>
<br />    "beta_vae": md.model.beta_vae,
<br />AttributeError: module 'mlmodels.model_tch.raw.pytorch_vae' has no attribute 'model'
