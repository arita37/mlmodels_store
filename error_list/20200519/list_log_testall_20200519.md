## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py


### Error 1, [Traceback at line 39](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L39)<br />39..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//keras_gan.py", line 31, in <module>
<br />    'AAE' : kg.aae.aae,
<br />AttributeError: module 'mlmodels.model_keras.raw.keras_gan' has no attribute 'aae'



### Error 2, [Traceback at line 85](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L85)<br />85..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//nbeats.py", line 315, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//nbeats.py", line 278, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//nbeats.py", line 172, in get_dataset
<br />    train_data = Data(data_source= path_norm( data_pars["train_data_source"]) ,
<br />NameError: name 'Data' is not defined



### Error 3, [Traceback at line 5055](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5055)<br />5055..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//01_deepctr.py", line 541, in <module>
<br />    test(pars_choice=5, **{"model_name": model_name})
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//01_deepctr.py", line 517, in test
<br />    module, model = module_load_full("model_keras.01_deepctr", model_pars, data_pars, compute_pars, dataset=dataset)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/models.py", line 101, in module_load_full
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras/01_deepctr.py", line 155, in __init__
<br />    self.model = modeli(feature_columns, **MODEL_PARAMS[model_name])
<br />TypeError: PNN() got an unexpected keyword argument 'embedding_size'



### Error 4, [Traceback at line 5114](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5114)<br />5114..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//textvae.py", line 356, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//textvae.py", line 327, in test
<br />    xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/text/quora/train.csv'



### Error 5, [Traceback at line 5163](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5163)<br />5163..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//namentity_crm_bilstm_dataloader.py", line 306, in <module>
<br />    test_module(model_uri=MODEL_URI, param_pars=param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras/namentity_crm_bilstm_dataloader.py", line 197, in get_params
<br />    cf = json.load(open(data_path, mode="r"))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader.json'



### Error 6, [Traceback at line 5203](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5203)<br />5203..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 7, [Traceback at line 5344](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5344)<br />5344..Traceback (most recent call last):
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



### Error 8, [Traceback at line 5399](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5399)<br />5399..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//charcnn.py", line 357, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//charcnn.py", line 320, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_keras//charcnn.py", line 216, in get_dataset
<br />    if data_pars['type'] == "npz":
<br />KeyError: 'type'



### Error 9, [Traceback at line 5445](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5445)<br />5445..Traceback (most recent call last):
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
<br />[master e64a6a8] ml_store
<br /> 1 file changed, 45 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   4ac20bc..e64a6a8  master -> master
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
<br /> 2228224/17464789 [==>...........................] - ETA: 0s
<br /> 7421952/17464789 [===========>..................] - ETA: 0s
<br />11542528/17464789 [==================>...........] - ETA: 0s
<br />15163392/17464789 [=========================>....] - ETA: 0s
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
<br />2020-05-19 20:25:26.569740: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
<br />2020-05-19 20:25:26.573956: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
<br />2020-05-19 20:25:26.574096: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55eae1175830 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
<br />2020-05-19 20:25:26.574109: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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
<br /> 1000/25000 [>.............................] - ETA: 13s - loss: 7.7893 - accuracy: 0.4920
<br /> 2000/25000 [=>............................] - ETA: 9s - loss: 7.8046 - accuracy: 0.4910 
<br /> 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8660 - accuracy: 0.4870
<br /> 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8046 - accuracy: 0.4910
<br /> 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8261 - accuracy: 0.4896
<br /> 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6922 - accuracy: 0.4983
<br /> 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7148 - accuracy: 0.4969
<br /> 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7529 - accuracy: 0.4944
<br /> 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7314 - accuracy: 0.4958
<br />10000/25000 [===========>..................] - ETA: 4s - loss: 7.7188 - accuracy: 0.4966
<br />11000/25000 [============>.................] - ETA: 4s - loss: 7.7349 - accuracy: 0.4955
<br />12000/25000 [=============>................] - ETA: 4s - loss: 7.7088 - accuracy: 0.4972
<br />13000/25000 [==============>...............] - ETA: 3s - loss: 7.7221 - accuracy: 0.4964
<br />14000/25000 [===============>..............] - ETA: 3s - loss: 7.7258 - accuracy: 0.4961
<br />15000/25000 [=================>............] - ETA: 3s - loss: 7.7249 - accuracy: 0.4962
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.7260 - accuracy: 0.4961
<br />17000/25000 [===================>..........] - ETA: 2s - loss: 7.7180 - accuracy: 0.4966
<br />18000/25000 [====================>.........] - ETA: 2s - loss: 7.7143 - accuracy: 0.4969
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6900 - accuracy: 0.4985
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6827 - accuracy: 0.4990
<br />21000/25000 [========================>.....] - ETA: 1s - loss: 7.6878 - accuracy: 0.4986
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6766 - accuracy: 0.4993
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
<br />25000/25000 [==============================] - 9s 366us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
<br />(<mlmodels.util.Model_empty object at 0x7f0cc2a439b0>, None)
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
<br />  <mlmodels.model_keras.textcnn.Model object at 0x7f0ca4cd1160> 
<br />
<br />  #### Fit   ######################################################## 
<br />Loading data...
<br />Pad sequences (samples x time)...
<br />Train on 25000 samples, validate on 25000 samples
<br />Epoch 1/1
<br />
<br /> 1000/25000 [>.............................] - ETA: 12s - loss: 8.2646 - accuracy: 0.4610
<br /> 2000/25000 [=>............................] - ETA: 9s - loss: 7.8123 - accuracy: 0.4905 
<br /> 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8251 - accuracy: 0.4897
<br /> 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6858 - accuracy: 0.4988
<br /> 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7218 - accuracy: 0.4964
<br /> 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7612 - accuracy: 0.4938
<br /> 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7258 - accuracy: 0.4961
<br /> 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7107 - accuracy: 0.4971
<br /> 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6871 - accuracy: 0.4987
<br />10000/25000 [===========>..................] - ETA: 4s - loss: 7.6896 - accuracy: 0.4985
<br />11000/25000 [============>.................] - ETA: 4s - loss: 7.6638 - accuracy: 0.5002
<br />12000/25000 [=============>................] - ETA: 4s - loss: 7.6436 - accuracy: 0.5015
<br />13000/25000 [==============>...............] - ETA: 3s - loss: 7.6654 - accuracy: 0.5001
<br />14000/25000 [===============>..............] - ETA: 3s - loss: 7.6579 - accuracy: 0.5006
<br />15000/25000 [=================>............] - ETA: 3s - loss: 7.6574 - accuracy: 0.5006
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.6360 - accuracy: 0.5020
<br />17000/25000 [===================>..........] - ETA: 2s - loss: 7.6305 - accuracy: 0.5024
<br />18000/25000 [====================>.........] - ETA: 2s - loss: 7.6377 - accuracy: 0.5019
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6545 - accuracy: 0.5008
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6544 - accuracy: 0.5008
<br />21000/25000 [========================>.....] - ETA: 1s - loss: 7.6637 - accuracy: 0.5002
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6457 - accuracy: 0.5014
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6613 - accuracy: 0.5003
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6737 - accuracy: 0.4995
<br />25000/25000 [==============================] - 9s 373us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
<br /> 1000/25000 [>.............................] - ETA: 12s - loss: 7.4366 - accuracy: 0.5150
<br /> 2000/25000 [=>............................] - ETA: 9s - loss: 7.5593 - accuracy: 0.5070 
<br /> 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5900 - accuracy: 0.5050
<br /> 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7510 - accuracy: 0.4945
<br /> 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6482 - accuracy: 0.5012
<br /> 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6538 - accuracy: 0.5008
<br /> 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6885 - accuracy: 0.4986
<br /> 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6781 - accuracy: 0.4992
<br /> 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6768 - accuracy: 0.4993
<br />10000/25000 [===========>..................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
<br />11000/25000 [============>.................] - ETA: 4s - loss: 7.6903 - accuracy: 0.4985
<br />12000/25000 [=============>................] - ETA: 4s - loss: 7.7216 - accuracy: 0.4964
<br />13000/25000 [==============>...............] - ETA: 3s - loss: 7.7221 - accuracy: 0.4964
<br />14000/25000 [===============>..............] - ETA: 3s - loss: 7.7225 - accuracy: 0.4964
<br />15000/25000 [=================>............] - ETA: 3s - loss: 7.7239 - accuracy: 0.4963
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.7193 - accuracy: 0.4966
<br />17000/25000 [===================>..........] - ETA: 2s - loss: 7.7063 - accuracy: 0.4974
<br />18000/25000 [====================>.........] - ETA: 2s - loss: 7.6981 - accuracy: 0.4979
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.7021 - accuracy: 0.4977
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6958 - accuracy: 0.4981
<br />21000/25000 [========================>.....] - ETA: 1s - loss: 7.6841 - accuracy: 0.4989
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6875 - accuracy: 0.4986
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6686 - accuracy: 0.4999
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6602 - accuracy: 0.5004
<br />25000/25000 [==============================] - 9s 370us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
<br />From github.com:arita37/mlmodels_store
<br />   e64a6a8..a8427b7  master     -> origin/master
<br />Updating e64a6a8..a8427b7
<br />Fast-forward
<br /> .../20200519/list_log_pullrequest_20200519.md      |   2 +-
<br /> error_list/20200519/list_log_testall_20200519.md   | 103 +++++++++++++++++++++
<br /> 2 files changed, 104 insertions(+), 1 deletion(-)
<br />[master 9e2b244] ml_store
<br /> 1 file changed, 325 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   a8427b7..9e2b244  master -> master
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
<br />13/13 [==============================] - 1s 106ms/step - loss: nan
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
<br />13/13 [==============================] - 0s 5ms/step - loss: nan
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



### Error 10, [Traceback at line 5891](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5891)<br />5891..Traceback (most recent call last):
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



### Error 11, [Traceback at line 8058](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L8058)<br />8058..Traceback (most recent call last):
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



### Error 12, [Traceback at line 8070](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L8070)<br />8070..Traceback (most recent call last):
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



### Error 13, [Traceback at line 8121](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L8121)<br />8121..Traceback (most recent call last):
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



### Error 14, [Traceback at line 8132](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L8132)<br />8132..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tf//1_lstm.py", line 332, in <module>
<br />    test(data_path="", pars_choice="test01", config_mode="test")
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tf//1_lstm.py", line 320, in test
<br />    session = load(out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tf//1_lstm.py", line 199, in load
<br />    return load_tf(load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/util.py", line 477, in load_tf
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
<br />[master 1840f61] ml_store
<br /> 1 file changed, 234 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   abbb2b5..1840f61  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tf//temporal_fusion_google.py 



### Error 15, [Traceback at line 8214](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L8214)<br />8214..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
<br />    from mlmodels.mode_tf.raw  import temporal_fusion_google
<br />ModuleNotFoundError: No module named 'mlmodels.mode_tf'



### Error 16, [Traceback at line 8471](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L8471)<br />8471..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_gluon//fb_prophet.py", line 160, in <module>
<br />    test(data_path = "model_fb/fbprophet.json", choice="json" )
<br />TypeError: test() got an unexpected keyword argument 'choice'



### Error 17, [Traceback at line 10558](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L10558)<br />10558..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tch//transformer_classifier.py", line 522, in <module>
<br />    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tch//transformer_classifier.py", line 418, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'model_tch/transformer_classifier.json'



### Error 18, [Traceback at line 11099](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L11099)<br />11099..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tch//03_nbeats_dataloader.py", line 9, in <module>
<br />    from dataloader import DataLoader
<br />ModuleNotFoundError: No module named 'dataloader'
