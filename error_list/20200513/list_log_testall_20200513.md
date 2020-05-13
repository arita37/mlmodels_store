## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py


### Error 1, [Traceback at line 37](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L37)<br />37..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//keras_gan.py", line 31, in <module>
<br />    'AAE' : kg.aae.aae,
<br />AttributeError: module 'mlmodels.model_keras.raw.keras_gan' has no attribute 'aae'



### Error 2, [Traceback at line 91](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L91)<br />91..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//textcnn_dataloader.py", line 275, in <module>
<br />    test_module(model_uri = MODEL_URI, param_pars= param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras/textcnn_dataloader.py", line 182, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/dataset/json/refactor/textcnn_keras.json'



### Error 3, [Traceback at line 138](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L138)<br />138..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//nbeats.py", line 315, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//nbeats.py", line 278, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//nbeats.py", line 172, in get_dataset
<br />    train_data = Data(data_source= path_norm( data_pars["train_data_source"]) ,
<br />NameError: name 'Data' is not defined



### Error 4, [Traceback at line 5888](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L5888)<br />5888..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//textvae.py", line 356, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//textvae.py", line 327, in test
<br />    xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/dataset/text/quora/train.csv'



### Error 5, [Traceback at line 5936](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L5936)<br />5936..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//namentity_crm_bilstm_dataloader.py", line 306, in <module>
<br />    test_module(model_uri=MODEL_URI, param_pars=param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras/namentity_crm_bilstm_dataloader.py", line 197, in get_params
<br />    cf = json.load(open(data_path, mode="r"))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader.json'



### Error 6, [Traceback at line 5975](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L5975)<br />5975..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 7, [Traceback at line 6114](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L6114)<br />6114..Traceback (most recent call last):
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



### Error 8, [Traceback at line 6168](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L6168)<br />6168..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//charcnn.py", line 357, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//charcnn.py", line 320, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_keras//charcnn.py", line 216, in get_dataset
<br />    if data_pars['type'] == "npz":
<br />KeyError: 'type'



### Error 9, [Traceback at line 6212](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L6212)<br />6212..Traceback (most recent call last):
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
<br />[master 350c602] ml_store
<br /> 1 file changed, 44 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   8f9347c..350c602  master -> master
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
<br />   24576/17464789 [..............................] - ETA: 50s
<br />   49152/17464789 [..............................] - ETA: 50s
<br />   98304/17464789 [..............................] - ETA: 37s
<br />  139264/17464789 [..............................] - ETA: 35s
<br />  204800/17464789 [..............................] - ETA: 29s
<br />  376832/17464789 [..............................] - ETA: 19s
<br />  622592/17464789 [>.............................] - ETA: 13s
<br /> 1179648/17464789 [=>............................] - ETA: 7s 
<br /> 2293760/17464789 [==>...........................] - ETA: 4s
<br /> 4505600/17464789 [======>.......................] - ETA: 2s
<br /> 7487488/17464789 [===========>..................] - ETA: 1s
<br />10223616/17464789 [================>.............] - ETA: 0s
<br />13352960/17464789 [=====================>........] - ETA: 0s
<br />16433152/17464789 [===========================>..] - ETA: 0s
<br />17465344/17464789 [==============================] - 1s 0us/step
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
<br />2020-05-13 12:28:46.629550: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
<br />2020-05-13 12:28:46.633929: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
<br />2020-05-13 12:28:46.634657: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f9f121d600 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
<br />2020-05-13 12:28:46.634675: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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
<br /> 1000/25000 [>.............................] - ETA: 11s - loss: 7.9580 - accuracy: 0.4810
<br /> 2000/25000 [=>............................] - ETA: 8s - loss: 7.6206 - accuracy: 0.5030 
<br /> 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5491 - accuracy: 0.5077
<br /> 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6130 - accuracy: 0.5035
<br /> 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5746 - accuracy: 0.5060
<br /> 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5900 - accuracy: 0.5050
<br /> 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5987 - accuracy: 0.5044
<br /> 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5689 - accuracy: 0.5064
<br /> 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6002 - accuracy: 0.5043
<br />10000/25000 [===========>..................] - ETA: 4s - loss: 7.6038 - accuracy: 0.5041
<br />11000/25000 [============>.................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
<br />12000/25000 [=============>................] - ETA: 3s - loss: 7.5925 - accuracy: 0.5048
<br />13000/25000 [==============>...............] - ETA: 3s - loss: 7.5805 - accuracy: 0.5056
<br />14000/25000 [===============>..............] - ETA: 3s - loss: 7.5921 - accuracy: 0.5049
<br />15000/25000 [=================>............] - ETA: 2s - loss: 7.6155 - accuracy: 0.5033
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.6120 - accuracy: 0.5036
<br />17000/25000 [===================>..........] - ETA: 2s - loss: 7.6405 - accuracy: 0.5017
<br />18000/25000 [====================>.........] - ETA: 1s - loss: 7.6462 - accuracy: 0.5013
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6432 - accuracy: 0.5015
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6590 - accuracy: 0.5005
<br />21000/25000 [========================>.....] - ETA: 1s - loss: 7.6469 - accuracy: 0.5013
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6485 - accuracy: 0.5012
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
<br />25000/25000 [==============================] - 8s 338us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
<br />(<mlmodels.util.Model_empty object at 0x7fc3286ca9e8>, None)
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
<br />  <mlmodels.model_keras.textcnn.Model object at 0x7fc31f0f24a8> 
<br />
<br />  #### Fit   ######################################################## 
<br />Loading data...
<br />Pad sequences (samples x time)...
<br />Train on 25000 samples, validate on 25000 samples
<br />Epoch 1/1
<br />
<br /> 1000/25000 [>.............................] - ETA: 12s - loss: 8.0193 - accuracy: 0.4770
<br /> 2000/25000 [=>............................] - ETA: 9s - loss: 7.8046 - accuracy: 0.4910 
<br /> 3000/25000 [==>...........................] - ETA: 7s - loss: 7.8251 - accuracy: 0.4897
<br /> 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7625 - accuracy: 0.4938
<br /> 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7801 - accuracy: 0.4926
<br /> 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7688 - accuracy: 0.4933
<br /> 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7280 - accuracy: 0.4960
<br /> 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7069 - accuracy: 0.4974
<br /> 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7194 - accuracy: 0.4966
<br />10000/25000 [===========>..................] - ETA: 4s - loss: 7.7188 - accuracy: 0.4966
<br />11000/25000 [============>.................] - ETA: 4s - loss: 7.6875 - accuracy: 0.4986
<br />12000/25000 [=============>................] - ETA: 3s - loss: 7.6909 - accuracy: 0.4984
<br />13000/25000 [==============>...............] - ETA: 3s - loss: 7.6560 - accuracy: 0.5007
<br />14000/25000 [===============>..............] - ETA: 3s - loss: 7.6601 - accuracy: 0.5004
<br />15000/25000 [=================>............] - ETA: 2s - loss: 7.6656 - accuracy: 0.5001
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.6676 - accuracy: 0.4999
<br />17000/25000 [===================>..........] - ETA: 2s - loss: 7.6657 - accuracy: 0.5001
<br />18000/25000 [====================>.........] - ETA: 1s - loss: 7.6556 - accuracy: 0.5007
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6505 - accuracy: 0.5011
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6636 - accuracy: 0.5002
<br />21000/25000 [========================>.....] - ETA: 1s - loss: 7.6557 - accuracy: 0.5007
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6394 - accuracy: 0.5018
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6566 - accuracy: 0.5007
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
<br />25000/25000 [==============================] - 8s 338us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
<br /> 1000/25000 [>.............................] - ETA: 11s - loss: 7.3140 - accuracy: 0.5230
<br /> 2000/25000 [=>............................] - ETA: 8s - loss: 7.7280 - accuracy: 0.4960 
<br /> 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7126 - accuracy: 0.4970
<br /> 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
<br /> 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6912 - accuracy: 0.4984
<br /> 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6717 - accuracy: 0.4997
<br /> 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6535 - accuracy: 0.5009
<br /> 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6206 - accuracy: 0.5030
<br /> 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6445 - accuracy: 0.5014
<br />10000/25000 [===========>..................] - ETA: 4s - loss: 7.6406 - accuracy: 0.5017
<br />11000/25000 [============>.................] - ETA: 3s - loss: 7.6248 - accuracy: 0.5027
<br />12000/25000 [=============>................] - ETA: 3s - loss: 7.5963 - accuracy: 0.5046
<br />13000/25000 [==============>...............] - ETA: 3s - loss: 7.5793 - accuracy: 0.5057
<br />14000/25000 [===============>..............] - ETA: 3s - loss: 7.5790 - accuracy: 0.5057
<br />15000/25000 [=================>............] - ETA: 2s - loss: 7.6043 - accuracy: 0.5041
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.6206 - accuracy: 0.5030
<br />17000/25000 [===================>..........] - ETA: 2s - loss: 7.6269 - accuracy: 0.5026
<br />18000/25000 [====================>.........] - ETA: 1s - loss: 7.6308 - accuracy: 0.5023
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6222 - accuracy: 0.5029
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6597 - accuracy: 0.5005
<br />21000/25000 [========================>.....] - ETA: 1s - loss: 7.6695 - accuracy: 0.4998
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6792 - accuracy: 0.4992
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6766 - accuracy: 0.4993
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6864 - accuracy: 0.4987
<br />25000/25000 [==============================] - 8s 336us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
<br />From github.com:arita37/mlmodels_store
<br />   350c602..c69f43d  master     -> origin/master
<br />Updating 350c602..c69f43d
<br />Fast-forward
<br /> .../20200513/list_log_dataloader_20200513.md       |    2 +-
<br /> error_list/20200513/list_log_jupyter_20200513.md   | 1668 ++++++++++----------
<br /> error_list/20200513/list_log_test_cli_20200513.md  |  152 +-
<br /> 3 files changed, 911 insertions(+), 911 deletions(-)
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
<br />[master c382f84] ml_store
<br /> 1 file changed, 335 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   c69f43d..c382f84  master -> master
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
<br />13/13 [==============================] - 1s 112ms/step - loss: nan
<br />Epoch 2/10
<br />
<br />13/13 [==============================] - 0s 4ms/step - loss: nan
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
<br />13/13 [==============================] - 0s 3ms/step - loss: nan
<br />Epoch 10/10
<br />
<br />13/13 [==============================] - 0s 3ms/step - loss: nan
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



### Error 10, [Traceback at line 6667](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L6667)<br />6667..Traceback (most recent call last):
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



### Error 11, [Traceback at line 8244](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L8244)<br />8244..Traceback (most recent call last):
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



### Error 12, [Traceback at line 8256](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L8256)<br />8256..Traceback (most recent call last):
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



### Error 13, [Traceback at line 8307](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L8307)<br />8307..Traceback (most recent call last):
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



### Error 14, [Traceback at line 8318](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L8318)<br />8318..Traceback (most recent call last):
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
<br />[master efc6832] ml_store
<br /> 1 file changed, 233 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   39dbb65..efc6832  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tf//temporal_fusion_google.py 



### Error 15, [Traceback at line 8399](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L8399)<br />8399..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
<br />    from mlmodels.mode_tf.raw  import temporal_fusion_google
<br />ModuleNotFoundError: No module named 'mlmodels.mode_tf'



### Error 16, [Traceback at line 8661](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L8661)<br />8661..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_gluon//fb_prophet.py", line 160, in <module>
<br />    test(data_path = "model_fb/fbprophet.json", choice="json" )
<br />TypeError: test() got an unexpected keyword argument 'choice'



### Error 17, [Traceback at line 8696](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L8696)<br />8696..Traceback (most recent call last):
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



### Error 18, [Traceback at line 10220](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L10220)<br />10220..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//transformer_classifier.py", line 522, in <module>
<br />    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//transformer_classifier.py", line 418, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'model_tch/transformer_classifier.json'



### Error 19, [Traceback at line 10667](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L10667)<br />10667..Traceback (most recent call last):
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



### Error 20, [Traceback at line 10718](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L10718)<br />10718..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//03_nbeats_dataloader.py", line 9, in <module>
<br />    from dataloader import DataLoader
<br />ModuleNotFoundError: No module named 'dataloader'



### Error 21, [Traceback at line 10841](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L10841)<br />10841..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//pytorch_vae.py", line 34, in <module>
<br />    "beta_vae": md.model.beta_vae,
<br />AttributeError: module 'mlmodels.model_tch.raw.pytorch_vae' has no attribute 'model'



### Error 22, [Traceback at line 10978](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L10978)<br />10978..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//textcnn.py", line 153, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 23, [Traceback at line 10989](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-13-12-12_6672e19fe4cfa7df885e45d91d645534b8989485.py#L10989)<br />10989..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//textcnn.py", line 477, in <module>
<br />    test( data_path="model_tch/textcnn.json", pars_choice = "test01" )
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//textcnn.py", line 442, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//textcnn.py", line 334, in get_dataset
<br />    trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
<br />  File "https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485/mlmodels/model_tch//textcnn.py", line 159, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.
