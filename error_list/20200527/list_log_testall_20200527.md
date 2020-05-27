## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py


### Error 1, [Traceback at line 39](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L39)<br />39..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//keras_gan.py", line 31, in <module>
<br />    'AAE' : kg.aae.aae,
<br />AttributeError: module 'mlmodels.model_keras.raw.keras_gan' has no attribute 'aae'



### Error 2, [Traceback at line 83](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L83)<br />83..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//nbeats.py", line 315, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//nbeats.py", line 278, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//nbeats.py", line 172, in get_dataset
<br />    train_data = Data(data_source= path_norm( data_pars["train_data_source"]) ,
<br />NameError: name 'Data' is not defined



### Error 3, [Traceback at line 5051](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5051)<br />5051..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//01_deepctr.py", line 541, in <module>
<br />    test(pars_choice=5, **{"model_name": model_name})
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//01_deepctr.py", line 517, in test
<br />    module, model = module_load_full("model_keras.01_deepctr", model_pars, data_pars, compute_pars, dataset=dataset)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 101, in module_load_full
<br />    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras/01_deepctr.py", line 155, in __init__
<br />    self.model = modeli(feature_columns, **MODEL_PARAMS[model_name])
<br />TypeError: PNN() got an unexpected keyword argument 'embedding_size'



### Error 4, [Traceback at line 5100](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5100)<br />5100..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//textvae.py", line 356, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//textvae.py", line 327, in test
<br />    xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/text/quora/train.csv'



### Error 5, [Traceback at line 5147](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5147)<br />5147..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//namentity_crm_bilstm_dataloader.py", line 306, in <module>
<br />    test_module(model_uri=MODEL_URI, param_pars=param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras/namentity_crm_bilstm_dataloader.py", line 197, in get_params
<br />    cf = json.load(open(data_path, mode="r"))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader.json'



### Error 6, [Traceback at line 5185](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5185)<br />5185..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 7, [Traceback at line 5226](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5226)<br />5226..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//charcnn_zhang.py", line 284, in <module>
<br />    test(pars_choice="json", data_path= f"{root_path}/model_keras/charcnn_zhang.json")
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//charcnn_zhang.py", line 248, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//charcnn_zhang.py", line 139, in get_dataset
<br />    train_data.load_data()
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras/raw/char_cnn/data_utils.py", line 41, in load_data
<br />    with open(self.data_source, 'r', encoding='utf-8') as f:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/dataset/text/ag_news_csv/train.csv'



### Error 8, [Traceback at line 5274](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5274)<br />5274..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//charcnn.py", line 357, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//charcnn.py", line 320, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//charcnn.py", line 216, in get_dataset
<br />    if data_pars['type'] == "npz":
<br />KeyError: 'type'



### Error 9, [Traceback at line 5317](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5317)<br />5317..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//namentity_crm_bilstm.py", line 348, in <module>
<br />    test(pars_choice="json", data_path=f"model_keras/namentity_crm_bilstm.json")
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//namentity_crm_bilstm.py", line 311, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//namentity_crm_bilstm.py", line 193, in get_dataset
<br />    raise Exception(f"Not support dataset yet")
<br />Exception: Not support dataset yet
<br />
<br />   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
<br />[master 9cce634] ml_store  && git pull --all
<br /> 1 file changed, 43 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   d6ec4c6..9cce634  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//textcnn.py 
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
<br />   24576/17464789 [..............................] - ETA: 44s
<br />   57344/17464789 [..............................] - ETA: 37s
<br />  106496/17464789 [..............................] - ETA: 30s
<br />  212992/17464789 [..............................] - ETA: 20s
<br />  475136/17464789 [..............................] - ETA: 11s
<br />  958464/17464789 [>.............................] - ETA: 6s 
<br /> 1908736/17464789 [==>...........................] - ETA: 3s
<br /> 3858432/17464789 [=====>........................] - ETA: 1s
<br /> 6938624/17464789 [==========>...................] - ETA: 0s
<br /> 9953280/17464789 [================>.............] - ETA: 0s
<br />13049856/17464789 [=====================>........] - ETA: 0s
<br />16162816/17464789 [==========================>...] - ETA: 0s
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
<br />2020-05-27 09:29:42.450671: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
<br />2020-05-27 09:29:42.454834: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
<br />2020-05-27 09:29:42.454972: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56008971bf00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
<br />2020-05-27 09:29:42.454984: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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
<br /> 1000/25000 [>.............................] - ETA: 10s - loss: 7.8200 - accuracy: 0.4900
<br /> 2000/25000 [=>............................] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010 
<br /> 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7842 - accuracy: 0.4923
<br /> 4000/25000 [===>..........................] - ETA: 5s - loss: 7.8890 - accuracy: 0.4855
<br /> 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8230 - accuracy: 0.4898
<br /> 6000/25000 [======>.......................] - ETA: 4s - loss: 7.8327 - accuracy: 0.4892
<br /> 7000/25000 [=======>......................] - ETA: 4s - loss: 7.8002 - accuracy: 0.4913
<br /> 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7778 - accuracy: 0.4927
<br /> 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7808 - accuracy: 0.4926
<br />10000/25000 [===========>..................] - ETA: 3s - loss: 7.7648 - accuracy: 0.4936
<br />11000/25000 [============>.................] - ETA: 3s - loss: 7.7823 - accuracy: 0.4925
<br />12000/25000 [=============>................] - ETA: 2s - loss: 7.7637 - accuracy: 0.4937
<br />13000/25000 [==============>...............] - ETA: 2s - loss: 7.7645 - accuracy: 0.4936
<br />14000/25000 [===============>..............] - ETA: 2s - loss: 7.7269 - accuracy: 0.4961
<br />15000/25000 [=================>............] - ETA: 2s - loss: 7.7034 - accuracy: 0.4976
<br />16000/25000 [==================>...........] - ETA: 1s - loss: 7.7212 - accuracy: 0.4964
<br />17000/25000 [===================>..........] - ETA: 1s - loss: 7.7090 - accuracy: 0.4972
<br />18000/25000 [====================>.........] - ETA: 1s - loss: 7.6964 - accuracy: 0.4981
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.7037 - accuracy: 0.4976
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.7011 - accuracy: 0.4978
<br />21000/25000 [========================>.....] - ETA: 0s - loss: 7.7104 - accuracy: 0.4971
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6994 - accuracy: 0.4979
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6926 - accuracy: 0.4983
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6915 - accuracy: 0.4984
<br />25000/25000 [==============================] - 6s 256us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
<br />
<br />  #### save the trained model  ####################################### 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/ztest/model_keras/textcnn/model.h5'}
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
<br />{'path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/ztest/model_keras/textcnn/model.h5'}
<br />{'path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/ztest/model_keras/textcnn/model.h5'}
<br />(<mlmodels.util.Model_empty object at 0x7fe9ff2d2160>, None)
<br />
<br />  #### Module init   ############################################ 
<br />
<br />  <module 'mlmodels.model_keras.textcnn' from 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras/textcnn.py'> 
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
<br />  <mlmodels.model_keras.textcnn.Model object at 0x7fe9fe394860> 
<br />
<br />  #### Fit   ######################################################## 
<br />Loading data...
<br />Pad sequences (samples x time)...
<br />Train on 25000 samples, validate on 25000 samples
<br />Epoch 1/1
<br />
<br /> 1000/25000 [>.............................] - ETA: 10s - loss: 7.2066 - accuracy: 0.5300
<br /> 2000/25000 [=>............................] - ETA: 7s - loss: 7.6206 - accuracy: 0.5030 
<br /> 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7433 - accuracy: 0.4950
<br /> 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5976 - accuracy: 0.5045
<br /> 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6053 - accuracy: 0.5040
<br /> 6000/25000 [======>.......................] - ETA: 4s - loss: 7.5721 - accuracy: 0.5062
<br /> 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5659 - accuracy: 0.5066
<br /> 8000/25000 [========>.....................] - ETA: 3s - loss: 7.5861 - accuracy: 0.5052
<br /> 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6155 - accuracy: 0.5033
<br />10000/25000 [===========>..................] - ETA: 3s - loss: 7.6099 - accuracy: 0.5037
<br />11000/25000 [============>.................] - ETA: 3s - loss: 7.6234 - accuracy: 0.5028
<br />12000/25000 [=============>................] - ETA: 2s - loss: 7.6002 - accuracy: 0.5043
<br />13000/25000 [==============>...............] - ETA: 2s - loss: 7.6135 - accuracy: 0.5035
<br />14000/25000 [===============>..............] - ETA: 2s - loss: 7.6162 - accuracy: 0.5033
<br />15000/25000 [=================>............] - ETA: 2s - loss: 7.6390 - accuracy: 0.5018
<br />16000/25000 [==================>...........] - ETA: 1s - loss: 7.6570 - accuracy: 0.5006
<br />17000/25000 [===================>..........] - ETA: 1s - loss: 7.6540 - accuracy: 0.5008
<br />18000/25000 [====================>.........] - ETA: 1s - loss: 7.6538 - accuracy: 0.5008
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
<br />21000/25000 [========================>.....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6882 - accuracy: 0.4986
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6746 - accuracy: 0.4995
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6724 - accuracy: 0.4996
<br />25000/25000 [==============================] - 6s 259us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
<br />  <module 'mlmodels.model_keras.textcnn' from 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras/textcnn.py'> 
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
<br /> 1000/25000 [>.............................] - ETA: 10s - loss: 8.0193 - accuracy: 0.4770
<br /> 2000/25000 [=>............................] - ETA: 7s - loss: 7.7203 - accuracy: 0.4965 
<br /> 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7791 - accuracy: 0.4927
<br /> 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7241 - accuracy: 0.4963
<br /> 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
<br /> 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6743 - accuracy: 0.4995
<br /> 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7017 - accuracy: 0.4977
<br /> 8000/25000 [========>.....................] - ETA: 3s - loss: 7.6225 - accuracy: 0.5029
<br /> 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6462 - accuracy: 0.5013
<br />10000/25000 [===========>..................] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
<br />11000/25000 [============>.................] - ETA: 3s - loss: 7.6262 - accuracy: 0.5026
<br />12000/25000 [=============>................] - ETA: 2s - loss: 7.6487 - accuracy: 0.5012
<br />13000/25000 [==============>...............] - ETA: 2s - loss: 7.6454 - accuracy: 0.5014
<br />14000/25000 [===============>..............] - ETA: 2s - loss: 7.6458 - accuracy: 0.5014
<br />15000/25000 [=================>............] - ETA: 2s - loss: 7.6411 - accuracy: 0.5017
<br />16000/25000 [==================>...........] - ETA: 1s - loss: 7.6618 - accuracy: 0.5003
<br />17000/25000 [===================>..........] - ETA: 1s - loss: 7.6540 - accuracy: 0.5008
<br />18000/25000 [====================>.........] - ETA: 1s - loss: 7.6530 - accuracy: 0.5009
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6497 - accuracy: 0.5011
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6459 - accuracy: 0.5013
<br />21000/25000 [========================>.....] - ETA: 0s - loss: 7.6579 - accuracy: 0.5006
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6562 - accuracy: 0.5007
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6730 - accuracy: 0.4996
<br />25000/25000 [==============================] - 6s 259us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
<br />   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
<br />[master 565c77d] ml_store  && git pull --all
<br /> 1 file changed, 325 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br /> + 89ef6e5...565c77d master -> master (forced update)
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//armdn.py 
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
<br />13/13 [==============================] - 2s 125ms/step - loss: nan
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



### Error 10, [Traceback at line 5761](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L5761)<br />5761..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//armdn.py", line 380, in <module>
<br />    test(pars_choice="json", data_path= "model_keras/armdn.json")
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//armdn.py", line 354, in test
<br />    y_pred, y_test = predict(model=model, model_pars=model_pars, data_pars=data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_keras//armdn.py", line 170, in predict
<br />    model.model_pars["n_mixes"], temp=1.0)
<br />  File "<__array_function__ internals>", line 6, in apply_along_axis
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/shape_base.py", line 379, in apply_along_axis
<br />    res = asanyarray(func1d(inarr_view[ind0], *args, **kwargs))
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mdn/__init__.py", line 237, in sample_from_output
<br />    cov_matrix = np.identity(output_dim) * sig_vector
<br />ValueError: operands could not be broadcast together with shapes (12,12) (0,) 



### Error 11, [Traceback at line 7020](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L7020)<br />7020..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
<br />    from mlmodels.mode_tf.raw  import temporal_fusion_google
<br />ModuleNotFoundError: No module named 'mlmodels.mode_tf'



### Error 12, [Traceback at line 7362](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L7362)<br />7362..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_gluon//fb_prophet.py", line 160, in <module>
<br />    test(data_path = "model_fb/fbprophet.json", choice="json" )
<br />TypeError: test() got an unexpected keyword argument 'choice'



### Error 13, [Traceback at line 9434](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L9434)<br />9434..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch//transformer_classifier.py", line 522, in <module>
<br />    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch//transformer_classifier.py", line 418, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'model_tch/transformer_classifier.json'



### Error 14, [Traceback at line 10087](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L10087)<br />10087..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch//03_nbeats_dataloader.py", line 9, in <module>
<br />    from dataloader import DataLoader
<br />ModuleNotFoundError: No module named 'dataloader'



### Error 15, [Traceback at line 10123](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L10123)<br />10123..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch//transformer_sentence.py", line 487, in <module>
<br />    test(pars_choice="test01", data_path= "model_tch/transformer_sentence.json", config_mode="test")
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch//transformer_sentence.py", line 438, in test
<br />    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
<br />TypeError: 'NoneType' object is not iterable



### Error 16, [Traceback at line 10159](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L10159)<br />10159..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch//pytorch_vae.py", line 34, in <module>
<br />    "beta_vae": md.model.beta_vae,
<br />AttributeError: module 'mlmodels.model_tch.raw.pytorch_vae' has no attribute 'model'



### Error 17, [Traceback at line 10287](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L10287)<br />10287..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch//textcnn.py", line 153, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.



### Error 18, [Traceback at line 10298](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall.py#L10298)<br />10298..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch//textcnn.py", line 477, in <module>
<br />    test( data_path="model_tch/textcnn.json", pars_choice = "test01" )
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch//textcnn.py", line 442, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch//textcnn.py", line 334, in get_dataset
<br />    trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
<br />  File "https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f/mlmodels/model_tch//textcnn.py", line 159, in create_tabular_dataset
<br />    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
<br />    return util.load_model(name, **overrides)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
<br />    raise IOError(Errors.E050.format(name=name))
<br />OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.
