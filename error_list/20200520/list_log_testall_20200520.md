## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py


### Error 1, [Traceback at line 39](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L39)<br />39..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//keras_gan.py", line 31, in <module>
<br />    'AAE' : kg.aae.aae,
<br />AttributeError: module 'mlmodels.model_keras.raw.keras_gan' has no attribute 'aae'



### Error 2, [Traceback at line 84](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L84)<br />84..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//nbeats.py", line 315, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//nbeats.py", line 278, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//nbeats.py", line 172, in get_dataset
<br />    train_data = Data(data_source= path_norm( data_pars["train_data_source"]) ,
<br />NameError: name 'Data' is not defined



### Error 3, [Traceback at line 5054](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5054)<br />5054..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//01_deepctr.py", line 541, in <module>
<br />    test(pars_choice=5, **{"model_name": model_name})
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//01_deepctr.py", line 517, in test
<br />    module, model = module_load_full("model_keras.01_deepctr", model_pars, data_pars, compute_pars, dataset=dataset)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/models.py", line 101, in module_load_full
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras/01_deepctr.py", line 155, in __init__
<br />    self.model = modeli(feature_columns, **MODEL_PARAMS[model_name])
<br />TypeError: PNN() got an unexpected keyword argument 'embedding_size'



### Error 4, [Traceback at line 5113](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5113)<br />5113..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//textvae.py", line 356, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//textvae.py", line 327, in test
<br />    xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/text/quora/train.csv'



### Error 5, [Traceback at line 5162](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5162)<br />5162..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//namentity_crm_bilstm_dataloader.py", line 306, in <module>
<br />    test_module(model_uri=MODEL_URI, param_pars=param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras/namentity_crm_bilstm_dataloader.py", line 197, in get_params
<br />    cf = json.load(open(data_path, mode="r"))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader.json'



### Error 6, [Traceback at line 5202](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5202)<br />5202..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 7, [Traceback at line 5342](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5342)<br />5342..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//charcnn_zhang.py", line 284, in <module>
<br />    test(pars_choice="json", data_path= f"{root_path}/model_keras/charcnn_zhang.json")
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//charcnn_zhang.py", line 268, in test
<br />    model2 = load(out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//charcnn_zhang.py", line 118, in load
<br />    model = load_keras(load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/util.py", line 602, in load_keras
<br />    model.model = load_model(path_file)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/keras/saving/save.py", line 146, in load_model
<br />    loader_impl.parse_saved_model(filepath)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/saved_model/loader_impl.py", line 83, in parse_saved_model
<br />    constants.SAVED_MODEL_FILENAME_PB))
<br />OSError: SavedModel file does not exist at: ztest/ml_keras/charcnn_zhang//model.h5/{saved_model.pbtxt|saved_model.pb}



### Error 8, [Traceback at line 5396](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5396)<br />5396..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//charcnn.py", line 357, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//charcnn.py", line 320, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//charcnn.py", line 216, in get_dataset
<br />    if data_pars['type'] == "npz":
<br />KeyError: 'type'



### Error 9, [Traceback at line 5441](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5441)<br />5441..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//namentity_crm_bilstm.py", line 348, in <module>
<br />    test(pars_choice="json", data_path=f"model_keras/namentity_crm_bilstm.json")
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//namentity_crm_bilstm.py", line 311, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//namentity_crm_bilstm.py", line 193, in get_dataset
<br />    raise Exception(f"Not support dataset yet")
<br />Exception: Not support dataset yet
<br />
<br />   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
<br />Logs
<br />README.md
<br />README_actions.md
<br />create_error_file.py
<br />create_github_issues.py
<br />deps.txt
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
<br />Fetching origin
<br />Already up to date.
<br />[master 7602461] ml_store
<br /> 1 file changed, 45 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   3d03ad0..7602461  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//textcnn.py 
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
<br />   24576/17464789 [..............................] - ETA: 48s
<br />   57344/17464789 [..............................] - ETA: 41s
<br />  106496/17464789 [..............................] - ETA: 33s
<br />  180224/17464789 [..............................] - ETA: 26s
<br />  352256/17464789 [..............................] - ETA: 16s
<br />  720896/17464789 [>.............................] - ETA: 9s 
<br /> 1433600/17464789 [=>............................] - ETA: 5s
<br /> 2842624/17464789 [===>..........................] - ETA: 2s
<br /> 5693440/17464789 [========>.....................] - ETA: 1s
<br /> 8757248/17464789 [==============>...............] - ETA: 0s
<br />11788288/17464789 [===================>..........] - ETA: 0s
<br />14114816/17464789 [=======================>......] - ETA: 0s
<br />15474688/17464789 [=========================>....] - ETA: 0s
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
<br />2020-05-20 04:23:39.454433: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
<br />2020-05-20 04:23:39.458651: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
<br />2020-05-20 04:23:39.458804: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c1939b2ce0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
<br />2020-05-20 04:23:39.458815: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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
<br /> 1000/25000 [>.............................] - ETA: 10s - loss: 7.7433 - accuracy: 0.4950
<br /> 2000/25000 [=>............................] - ETA: 7s - loss: 7.6590 - accuracy: 0.5005 
<br /> 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
<br /> 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5900 - accuracy: 0.5050
<br /> 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5930 - accuracy: 0.5048
<br /> 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5848 - accuracy: 0.5053
<br /> 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6316 - accuracy: 0.5023
<br /> 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6283 - accuracy: 0.5025
<br /> 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6189 - accuracy: 0.5031
<br />10000/25000 [===========>..................] - ETA: 3s - loss: 7.6130 - accuracy: 0.5035
<br />11000/25000 [============>.................] - ETA: 3s - loss: 7.6220 - accuracy: 0.5029
<br />12000/25000 [=============>................] - ETA: 2s - loss: 7.6232 - accuracy: 0.5028
<br />13000/25000 [==============>...............] - ETA: 2s - loss: 7.6277 - accuracy: 0.5025
<br />14000/25000 [===============>..............] - ETA: 2s - loss: 7.6327 - accuracy: 0.5022
<br />15000/25000 [=================>............] - ETA: 2s - loss: 7.6472 - accuracy: 0.5013
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.6618 - accuracy: 0.5003
<br />17000/25000 [===================>..........] - ETA: 1s - loss: 7.6612 - accuracy: 0.5004
<br />18000/25000 [====================>.........] - ETA: 1s - loss: 7.6521 - accuracy: 0.5009
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6569 - accuracy: 0.5006
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6743 - accuracy: 0.4995
<br />21000/25000 [========================>.....] - ETA: 0s - loss: 7.6681 - accuracy: 0.4999
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6548 - accuracy: 0.5008
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6730 - accuracy: 0.4996
<br />25000/25000 [==============================] - 6s 259us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
<br />
<br />  #### save the trained model  ####################################### 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_keras/textcnn/model.h5'}
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
<br />{'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_keras/textcnn/model.h5'}
<br />{'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_keras/textcnn/model.h5'}
<br />(<mlmodels.util.Model_empty object at 0x7fba9d7dba58>, None)
<br />
<br />  #### Module init   ############################################ 
<br />
<br />  <module 'mlmodels.model_keras.textcnn' from 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras/textcnn.py'> 
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
<br />  <mlmodels.model_keras.textcnn.Model object at 0x7fbab6ce8ba8> 
<br />
<br />  #### Fit   ######################################################## 
<br />Loading data...
<br />Pad sequences (samples x time)...
<br />Train on 25000 samples, validate on 25000 samples
<br />Epoch 1/1
<br />
<br /> 1000/25000 [>.............................] - ETA: 10s - loss: 7.4520 - accuracy: 0.5140
<br /> 2000/25000 [=>............................] - ETA: 7s - loss: 7.5516 - accuracy: 0.5075 
<br /> 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5951 - accuracy: 0.5047
<br /> 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6436 - accuracy: 0.5015
<br /> 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5992 - accuracy: 0.5044
<br /> 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6078 - accuracy: 0.5038
<br /> 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6162 - accuracy: 0.5033
<br /> 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6551 - accuracy: 0.5008
<br /> 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6581 - accuracy: 0.5006
<br />10000/25000 [===========>..................] - ETA: 3s - loss: 7.6774 - accuracy: 0.4993
<br />11000/25000 [============>.................] - ETA: 3s - loss: 7.6708 - accuracy: 0.4997
<br />12000/25000 [=============>................] - ETA: 2s - loss: 7.6577 - accuracy: 0.5006
<br />13000/25000 [==============>...............] - ETA: 2s - loss: 7.6713 - accuracy: 0.4997
<br />14000/25000 [===============>..............] - ETA: 2s - loss: 7.6765 - accuracy: 0.4994
<br />15000/25000 [=================>............] - ETA: 2s - loss: 7.6860 - accuracy: 0.4987
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.6982 - accuracy: 0.4979
<br />17000/25000 [===================>..........] - ETA: 1s - loss: 7.7009 - accuracy: 0.4978
<br />18000/25000 [====================>.........] - ETA: 1s - loss: 7.6956 - accuracy: 0.4981
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6755 - accuracy: 0.4994
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6689 - accuracy: 0.4999
<br />21000/25000 [========================>.....] - ETA: 0s - loss: 7.6528 - accuracy: 0.5009
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6415 - accuracy: 0.5016
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6500 - accuracy: 0.5011
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
<br />25000/25000 [==============================] - 7s 262us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
<br />  <module 'mlmodels.model_keras.textcnn' from 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras/textcnn.py'> 
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
<br /> 1000/25000 [>.............................] - ETA: 10s - loss: 7.8353 - accuracy: 0.4890
<br /> 2000/25000 [=>............................] - ETA: 7s - loss: 7.7280 - accuracy: 0.4960 
<br /> 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6155 - accuracy: 0.5033
<br /> 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7893 - accuracy: 0.4920
<br /> 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8077 - accuracy: 0.4908
<br /> 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7254 - accuracy: 0.4962
<br /> 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7542 - accuracy: 0.4943
<br /> 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7318 - accuracy: 0.4958
<br /> 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7058 - accuracy: 0.4974
<br />10000/25000 [===========>..................] - ETA: 3s - loss: 7.7004 - accuracy: 0.4978
<br />11000/25000 [============>.................] - ETA: 3s - loss: 7.7001 - accuracy: 0.4978
<br />12000/25000 [=============>................] - ETA: 2s - loss: 7.6986 - accuracy: 0.4979
<br />13000/25000 [==============>...............] - ETA: 2s - loss: 7.6737 - accuracy: 0.4995
<br />14000/25000 [===============>..............] - ETA: 2s - loss: 7.7039 - accuracy: 0.4976
<br />15000/25000 [=================>............] - ETA: 2s - loss: 7.6952 - accuracy: 0.4981
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.6829 - accuracy: 0.4989
<br />17000/25000 [===================>..........] - ETA: 1s - loss: 7.6865 - accuracy: 0.4987
<br />18000/25000 [====================>.........] - ETA: 1s - loss: 7.6803 - accuracy: 0.4991
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6900 - accuracy: 0.4985
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6613 - accuracy: 0.5003
<br />21000/25000 [========================>.....] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6715 - accuracy: 0.4997
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6773 - accuracy: 0.4993
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
<br />25000/25000 [==============================] - 7s 262us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
<br />   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
<br />Logs
<br />README.md
<br />README_actions.md
<br />create_error_file.py
<br />create_github_issues.py
<br />deps.txt
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
<br />Fetching origin
<br />Already up to date.
<br />[master 35c95d6] ml_store
<br /> 1 file changed, 328 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   7602461..35c95d6  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//armdn.py 
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
<br />13/13 [==============================] - 2s 121ms/step - loss: nan
<br />Epoch 2/10
<br />
<br />13/13 [==============================] - 0s 6ms/step - loss: nan
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



### Error 10, [Traceback at line 5890](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5890)<br />5890..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//armdn.py", line 380, in <module>
<br />    test(pars_choice="json", data_path= "model_keras/armdn.json")
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//armdn.py", line 354, in test
<br />    y_pred, y_test = predict(model=model, model_pars=model_pars, data_pars=data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//armdn.py", line 170, in predict
<br />    model.model_pars["n_mixes"], temp=1.0)
<br />  File "<__array_function__ internals>", line 6, in apply_along_axis
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/shape_base.py", line 379, in apply_along_axis
<br />    res = asanyarray(func1d(inarr_view[ind0], *args, **kwargs))
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mdn/__init__.py", line 237, in sample_from_output
<br />    cov_matrix = np.identity(output_dim) * sig_vector
<br />ValueError: operands could not be broadcast together with shapes (12,12) (0,) 
