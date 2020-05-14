## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py


### Error 1, [Traceback at line 37](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L37)<br />37..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//keras_gan.py", line 31, in <module>
<br />    'AAE' : kg.aae.aae,
<br />AttributeError: module 'mlmodels.model_keras.raw.keras_gan' has no attribute 'aae'



### Error 2, [Traceback at line 81](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L81)<br />81..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//textcnn_dataloader.py", line 275, in <module>
<br />    test_module(model_uri = MODEL_URI, param_pars= param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras/textcnn_dataloader.py", line 182, in get_params
<br />    cf = json.load(open(data_path, mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/json/refactor/textcnn_keras.json'



### Error 3, [Traceback at line 128](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L128)<br />128..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//nbeats.py", line 315, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//nbeats.py", line 278, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//nbeats.py", line 172, in get_dataset
<br />    train_data = Data(data_source= path_norm( data_pars["train_data_source"]) ,
<br />NameError: name 'Data' is not defined



### Error 4, [Traceback at line 5879](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L5879)<br />5879..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//textvae.py", line 356, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//textvae.py", line 327, in test
<br />    xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/text/quora/train.csv'



### Error 5, [Traceback at line 5927](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L5927)<br />5927..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//namentity_crm_bilstm_dataloader.py", line 306, in <module>
<br />    test_module(model_uri=MODEL_URI, param_pars=param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras/namentity_crm_bilstm_dataloader.py", line 197, in get_params
<br />    cf = json.load(open(data_path, mode="r"))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader.json'



### Error 6, [Traceback at line 5966](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L5966)<br />5966..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 7, [Traceback at line 6106](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L6106)<br />6106..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//charcnn_zhang.py", line 284, in <module>
<br />    test(pars_choice="json", data_path= f"{root_path}/model_keras/charcnn_zhang.json")
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//charcnn_zhang.py", line 268, in test
<br />    model2 = load(out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//charcnn_zhang.py", line 118, in load
<br />    model = load_keras(load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/util.py", line 602, in load_keras
<br />    model.model = load_model(path_file)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/keras/saving/save.py", line 146, in load_model
<br />    loader_impl.parse_saved_model(filepath)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/saved_model/loader_impl.py", line 83, in parse_saved_model
<br />    constants.SAVED_MODEL_FILENAME_PB))
<br />OSError: SavedModel file does not exist at: ztest/ml_keras/charcnn_zhang//model.h5/{saved_model.pbtxt|saved_model.pb}



### Error 8, [Traceback at line 6159](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L6159)<br />6159..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//charcnn.py", line 357, in <module>
<br />    test(pars_choice="test01")
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//charcnn.py", line 320, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//charcnn.py", line 216, in get_dataset
<br />    if data_pars['type'] == "npz":
<br />KeyError: 'type'



### Error 9, [Traceback at line 6203](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L6203)<br />6203..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//namentity_crm_bilstm.py", line 348, in <module>
<br />    test(pars_choice="json", data_path=f"model_keras/namentity_crm_bilstm.json")
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//namentity_crm_bilstm.py", line 311, in test
<br />    Xtuple = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//namentity_crm_bilstm.py", line 193, in get_dataset
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
<br />[master fe0a162] ml_store
<br /> 1 file changed, 44 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   2c95822..fe0a162  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//textcnn.py 
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
<br />   24576/17464789 [..............................] - ETA: 46s
<br />   57344/17464789 [..............................] - ETA: 40s
<br />   90112/17464789 [..............................] - ETA: 38s
<br />  122880/17464789 [..............................] - ETA: 37s
<br />  139264/17464789 [..............................] - ETA: 41s
<br />  212992/17464789 [..............................] - ETA: 32s
<br />  303104/17464789 [..............................] - ETA: 26s
<br />  368640/17464789 [..............................] - ETA: 24s
<br />  458752/17464789 [..............................] - ETA: 22s
<br />  540672/17464789 [..............................] - ETA: 20s
<br />  630784/17464789 [>.............................] - ETA: 19s
<br />  720896/17464789 [>.............................] - ETA: 18s
<br />  802816/17464789 [>.............................] - ETA: 17s
<br />  892928/17464789 [>.............................] - ETA: 17s
<br />  974848/17464789 [>.............................] - ETA: 16s
<br /> 1081344/17464789 [>.............................] - ETA: 16s
<br /> 1171456/17464789 [=>............................] - ETA: 15s
<br /> 1269760/17464789 [=>............................] - ETA: 15s
<br /> 1359872/17464789 [=>............................] - ETA: 14s
<br /> 1466368/17464789 [=>............................] - ETA: 14s
<br /> 1556480/17464789 [=>............................] - ETA: 14s
<br /> 1654784/17464789 [=>............................] - ETA: 13s
<br /> 1761280/17464789 [==>...........................] - ETA: 13s
<br /> 1851392/17464789 [==>...........................] - ETA: 13s
<br /> 1949696/17464789 [==>...........................] - ETA: 13s
<br /> 2056192/17464789 [==>...........................] - ETA: 12s
<br /> 2162688/17464789 [==>...........................] - ETA: 12s
<br /> 2269184/17464789 [==>...........................] - ETA: 12s
<br /> 2367488/17464789 [===>..........................] - ETA: 12s
<br /> 2473984/17464789 [===>..........................] - ETA: 12s
<br /> 2580480/17464789 [===>..........................] - ETA: 11s
<br /> 2686976/17464789 [===>..........................] - ETA: 11s
<br /> 2785280/17464789 [===>..........................] - ETA: 11s
<br /> 2891776/17464789 [===>..........................] - ETA: 11s
<br /> 3014656/17464789 [====>.........................] - ETA: 11s
<br /> 3121152/17464789 [====>.........................] - ETA: 10s
<br /> 3227648/17464789 [====>.........................] - ETA: 10s
<br /> 3325952/17464789 [====>.........................] - ETA: 10s
<br /> 3448832/17464789 [====>.........................] - ETA: 10s
<br /> 3555328/17464789 [=====>........................] - ETA: 10s
<br /> 3661824/17464789 [=====>........................] - ETA: 10s
<br /> 3776512/17464789 [=====>........................] - ETA: 10s
<br /> 3883008/17464789 [=====>........................] - ETA: 9s 
<br /> 3989504/17464789 [=====>........................] - ETA: 9s
<br /> 4112384/17464789 [======>.......................] - ETA: 9s
<br /> 4218880/17464789 [======>.......................] - ETA: 9s
<br /> 4317184/17464789 [======>.......................] - ETA: 9s
<br /> 4440064/17464789 [======>.......................] - ETA: 9s
<br /> 4546560/17464789 [======>.......................] - ETA: 9s
<br /> 4669440/17464789 [=======>......................] - ETA: 9s
<br /> 4775936/17464789 [=======>......................] - ETA: 8s
<br /> 4874240/17464789 [=======>......................] - ETA: 8s
<br /> 4997120/17464789 [=======>......................] - ETA: 8s
<br /> 5103616/17464789 [=======>......................] - ETA: 8s
<br /> 5210112/17464789 [=======>......................] - ETA: 8s
<br /> 5332992/17464789 [========>.....................] - ETA: 8s
<br /> 5431296/17464789 [========>.....................] - ETA: 8s
<br /> 5554176/17464789 [========>.....................] - ETA: 8s
<br /> 5660672/17464789 [========>.....................] - ETA: 8s
<br /> 5783552/17464789 [========>.....................] - ETA: 7s
<br /> 5890048/17464789 [=========>....................] - ETA: 7s
<br /> 6012928/17464789 [=========>....................] - ETA: 7s
<br /> 6127616/17464789 [=========>....................] - ETA: 7s
<br /> 6234112/17464789 [=========>....................] - ETA: 7s
<br /> 6356992/17464789 [=========>....................] - ETA: 7s
<br /> 6479872/17464789 [==========>...................] - ETA: 7s
<br /> 6602752/17464789 [==========>...................] - ETA: 7s
<br /> 6709248/17464789 [==========>...................] - ETA: 7s
<br /> 6823936/17464789 [==========>...................] - ETA: 7s
<br /> 6946816/17464789 [==========>...................] - ETA: 6s
<br /> 7069696/17464789 [===========>..................] - ETA: 6s
<br /> 7192576/17464789 [===========>..................] - ETA: 6s
<br /> 7315456/17464789 [===========>..................] - ETA: 6s
<br /> 7438336/17464789 [===========>..................] - ETA: 6s
<br /> 7561216/17464789 [===========>..................] - ETA: 6s
<br /> 7684096/17464789 [============>.................] - ETA: 6s
<br /> 7798784/17464789 [============>.................] - ETA: 6s
<br /> 7921664/17464789 [============>.................] - ETA: 6s
<br /> 8044544/17464789 [============>.................] - ETA: 6s
<br /> 8183808/17464789 [=============>................] - ETA: 5s
<br /> 8306688/17464789 [=============>................] - ETA: 5s
<br /> 8429568/17464789 [=============>................] - ETA: 5s
<br /> 8552448/17464789 [=============>................] - ETA: 5s
<br /> 8691712/17464789 [=============>................] - ETA: 5s
<br /> 8814592/17464789 [==============>...............] - ETA: 5s
<br /> 8937472/17464789 [==============>...............] - ETA: 5s
<br /> 9076736/17464789 [==============>...............] - ETA: 5s
<br /> 9191424/17464789 [==============>...............] - ETA: 5s
<br /> 9330688/17464789 [===============>..............] - ETA: 5s
<br /> 9453568/17464789 [===============>..............] - ETA: 5s
<br /> 9592832/17464789 [===============>..............] - ETA: 4s
<br /> 9715712/17464789 [===============>..............] - ETA: 4s
<br /> 9854976/17464789 [===============>..............] - ETA: 4s
<br /> 9977856/17464789 [================>.............] - ETA: 4s
<br />10117120/17464789 [================>.............] - ETA: 4s
<br />10256384/17464789 [================>.............] - ETA: 4s
<br />10395648/17464789 [================>.............] - ETA: 4s
<br />10518528/17464789 [=================>............] - ETA: 4s
<br />10657792/17464789 [=================>............] - ETA: 4s
<br />10797056/17464789 [=================>............] - ETA: 4s
<br />10936320/17464789 [=================>............] - ETA: 3s
<br />11075584/17464789 [==================>...........] - ETA: 3s
<br />11214848/17464789 [==================>...........] - ETA: 3s
<br />11354112/17464789 [==================>...........] - ETA: 3s
<br />11493376/17464789 [==================>...........] - ETA: 3s
<br />11649024/17464789 [===================>..........] - ETA: 3s
<br />11788288/17464789 [===================>..........] - ETA: 3s
<br />11927552/17464789 [===================>..........] - ETA: 3s
<br />12066816/17464789 [===================>..........] - ETA: 3s
<br />12222464/17464789 [===================>..........] - ETA: 3s
<br />12361728/17464789 [====================>.........] - ETA: 3s
<br />12517376/17464789 [====================>.........] - ETA: 2s
<br />12656640/17464789 [====================>.........] - ETA: 2s
<br />12812288/17464789 [=====================>........] - ETA: 2s
<br />12951552/17464789 [=====================>........] - ETA: 2s
<br />13115392/17464789 [=====================>........] - ETA: 2s
<br />13254656/17464789 [=====================>........] - ETA: 2s
<br />13410304/17464789 [======================>.......] - ETA: 2s
<br />13565952/17464789 [======================>.......] - ETA: 2s
<br />13705216/17464789 [======================>.......] - ETA: 2s
<br />13860864/17464789 [======================>.......] - ETA: 2s
<br />14016512/17464789 [=======================>......] - ETA: 1s
<br />14172160/17464789 [=======================>......] - ETA: 1s
<br />14327808/17464789 [=======================>......] - ETA: 1s
<br />14483456/17464789 [=======================>......] - ETA: 1s
<br />14647296/17464789 [========================>.....] - ETA: 1s
<br />14802944/17464789 [========================>.....] - ETA: 1s
<br />14958592/17464789 [========================>.....] - ETA: 1s
<br />15114240/17464789 [========================>.....] - ETA: 1s
<br />15269888/17464789 [=========================>....] - ETA: 1s
<br />15425536/17464789 [=========================>....] - ETA: 1s
<br />15597568/17464789 [=========================>....] - ETA: 1s
<br />15761408/17464789 [==========================>...] - ETA: 0s
<br />15917056/17464789 [==========================>...] - ETA: 0s
<br />16089088/17464789 [==========================>...] - ETA: 0s
<br />16261120/17464789 [==========================>...] - ETA: 0s
<br />16433152/17464789 [===========================>..] - ETA: 0s
<br />16613376/17464789 [===========================>..] - ETA: 0s
<br />16785408/17464789 [===========================>..] - ETA: 0s
<br />16957440/17464789 [============================>.] - ETA: 0s
<br />17154048/17464789 [============================>.] - ETA: 0s
<br />17326080/17464789 [============================>.] - ETA: 0s
<br />17465344/17464789 [==============================] - 9s 1us/step
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
<br />2020-05-14 20:29:45.691427: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
<br />2020-05-14 20:29:45.695975: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
<br />2020-05-14 20:29:45.696136: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555632094ed0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
<br />2020-05-14 20:29:45.696151: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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
<br /> 1000/25000 [>.............................] - ETA: 11s - loss: 7.7586 - accuracy: 0.4940
<br /> 2000/25000 [=>............................] - ETA: 8s - loss: 7.9426 - accuracy: 0.4820 
<br /> 3000/25000 [==>...........................] - ETA: 6s - loss: 8.0295 - accuracy: 0.4763
<br /> 4000/25000 [===>..........................] - ETA: 6s - loss: 7.9695 - accuracy: 0.4803
<br /> 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8537 - accuracy: 0.4878
<br /> 6000/25000 [======>.......................] - ETA: 5s - loss: 7.8174 - accuracy: 0.4902
<br /> 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7718 - accuracy: 0.4931
<br /> 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7337 - accuracy: 0.4956
<br /> 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6990 - accuracy: 0.4979
<br />10000/25000 [===========>..................] - ETA: 3s - loss: 7.6620 - accuracy: 0.5003
<br />11000/25000 [============>.................] - ETA: 3s - loss: 7.6457 - accuracy: 0.5014
<br />12000/25000 [=============>................] - ETA: 3s - loss: 7.6232 - accuracy: 0.5028
<br />13000/25000 [==============>...............] - ETA: 2s - loss: 7.6053 - accuracy: 0.5040
<br />14000/25000 [===============>..............] - ETA: 2s - loss: 7.6206 - accuracy: 0.5030
<br />15000/25000 [=================>............] - ETA: 2s - loss: 7.6349 - accuracy: 0.5021
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.6618 - accuracy: 0.5003
<br />17000/25000 [===================>..........] - ETA: 1s - loss: 7.6811 - accuracy: 0.4991
<br />18000/25000 [====================>.........] - ETA: 1s - loss: 7.7015 - accuracy: 0.4977
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6876 - accuracy: 0.4986
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6781 - accuracy: 0.4992
<br />21000/25000 [========================>.....] - ETA: 0s - loss: 7.6761 - accuracy: 0.4994
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6827 - accuracy: 0.4990
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6693 - accuracy: 0.4998
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6564 - accuracy: 0.5007
<br />25000/25000 [==============================] - 7s 276us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
<br />
<br />  #### save the trained model  ####################################### 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_keras/textcnn/model.h5'}
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
<br />{'path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_keras/textcnn/model.h5'}
<br />{'path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_keras/textcnn/model.h5'}
<br />(<mlmodels.util.Model_empty object at 0x7f148d470c50>, None)
<br />
<br />  #### Module init   ############################################ 
<br />
<br />  <module 'mlmodels.model_keras.textcnn' from 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras/textcnn.py'> 
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
<br />  <mlmodels.model_keras.textcnn.Model object at 0x7f1438bb82e8> 
<br />
<br />  #### Fit   ######################################################## 
<br />Loading data...
<br />Pad sequences (samples x time)...
<br />Train on 25000 samples, validate on 25000 samples
<br />Epoch 1/1
<br />
<br /> 1000/25000 [>.............................] - ETA: 10s - loss: 7.8046 - accuracy: 0.4910
<br /> 2000/25000 [=>............................] - ETA: 7s - loss: 7.6743 - accuracy: 0.4995 
<br /> 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7126 - accuracy: 0.4970
<br /> 4000/25000 [===>..........................] - ETA: 5s - loss: 7.8276 - accuracy: 0.4895
<br /> 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7678 - accuracy: 0.4934
<br /> 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7280 - accuracy: 0.4960
<br /> 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6951 - accuracy: 0.4981
<br /> 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7088 - accuracy: 0.4972
<br /> 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7058 - accuracy: 0.4974
<br />10000/25000 [===========>..................] - ETA: 3s - loss: 7.7280 - accuracy: 0.4960
<br />11000/25000 [============>.................] - ETA: 3s - loss: 7.7377 - accuracy: 0.4954
<br />12000/25000 [=============>................] - ETA: 3s - loss: 7.7305 - accuracy: 0.4958
<br />13000/25000 [==============>...............] - ETA: 2s - loss: 7.7197 - accuracy: 0.4965
<br />14000/25000 [===============>..............] - ETA: 2s - loss: 7.7017 - accuracy: 0.4977
<br />15000/25000 [=================>............] - ETA: 2s - loss: 7.6942 - accuracy: 0.4982
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.6781 - accuracy: 0.4992
<br />17000/25000 [===================>..........] - ETA: 1s - loss: 7.7018 - accuracy: 0.4977
<br />18000/25000 [====================>.........] - ETA: 1s - loss: 7.7237 - accuracy: 0.4963
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.7029 - accuracy: 0.4976
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6981 - accuracy: 0.4979
<br />21000/25000 [========================>.....] - ETA: 0s - loss: 7.6827 - accuracy: 0.4990
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
<br />25000/25000 [==============================] - 7s 275us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
<br />  <module 'mlmodels.model_keras.textcnn' from 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras/textcnn.py'> 
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
<br /> 1000/25000 [>.............................] - ETA: 11s - loss: 7.7740 - accuracy: 0.4930
<br /> 2000/25000 [=>............................] - ETA: 8s - loss: 7.7663 - accuracy: 0.4935 
<br /> 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6257 - accuracy: 0.5027
<br /> 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
<br /> 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6820 - accuracy: 0.4990
<br /> 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6615 - accuracy: 0.5003
<br /> 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
<br /> 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7203 - accuracy: 0.4965
<br /> 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7365 - accuracy: 0.4954
<br />10000/25000 [===========>..................] - ETA: 3s - loss: 7.6927 - accuracy: 0.4983
<br />11000/25000 [============>.................] - ETA: 3s - loss: 7.6861 - accuracy: 0.4987
<br />12000/25000 [=============>................] - ETA: 3s - loss: 7.6768 - accuracy: 0.4993
<br />13000/25000 [==============>...............] - ETA: 2s - loss: 7.6678 - accuracy: 0.4999
<br />14000/25000 [===============>..............] - ETA: 2s - loss: 7.6754 - accuracy: 0.4994
<br />15000/25000 [=================>............] - ETA: 2s - loss: 7.6472 - accuracy: 0.5013
<br />16000/25000 [==================>...........] - ETA: 2s - loss: 7.6407 - accuracy: 0.5017
<br />17000/25000 [===================>..........] - ETA: 1s - loss: 7.6585 - accuracy: 0.5005
<br />18000/25000 [====================>.........] - ETA: 1s - loss: 7.6700 - accuracy: 0.4998
<br />19000/25000 [=====================>........] - ETA: 1s - loss: 7.6884 - accuracy: 0.4986
<br />20000/25000 [=======================>......] - ETA: 1s - loss: 7.6689 - accuracy: 0.4999
<br />21000/25000 [========================>.....] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
<br />22000/25000 [=========================>....] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
<br />23000/25000 [==========================>...] - ETA: 0s - loss: 7.6640 - accuracy: 0.5002
<br />24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
<br />25000/25000 [==============================] - 7s 279us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
<br />   fe0a162..99598cf  master     -> origin/master
<br />Updating fe0a162..99598cf
<br />Fast-forward
<br /> error_list/20200514/list_log_benchmark_20200514.md |  166 +-
<br /> error_list/20200514/list_log_json_20200514.md      | 1146 ++++++-------
<br /> error_list/20200514/list_log_jupyter_20200514.md   | 1754 ++++++++++----------
<br /> error_list/20200514/list_log_testall_20200514.md   |   89 +
<br /> 4 files changed, 1617 insertions(+), 1538 deletions(-)
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
<br />[master 78b0ff6] ml_store
<br /> 1 file changed, 464 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   99598cf..78b0ff6  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//armdn.py 
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
<br />13/13 [==============================] - 2s 136ms/step - loss: nan
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



### Error 10, [Traceback at line 6787](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L6787)<br />6787..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//armdn.py", line 380, in <module>
<br />    test(pars_choice="json", data_path= "model_keras/armdn.json")
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//armdn.py", line 354, in test
<br />    y_pred, y_test = predict(model=model, model_pars=model_pars, data_pars=data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras//armdn.py", line 170, in predict
<br />    model.model_pars["n_mixes"], temp=1.0)
<br />  File "<__array_function__ internals>", line 6, in apply_along_axis
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/shape_base.py", line 379, in apply_along_axis
<br />    res = asanyarray(func1d(inarr_view[ind0], *args, **kwargs))
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mdn/__init__.py", line 237, in sample_from_output
<br />    cov_matrix = np.identity(output_dim) * sig_vector
<br />ValueError: operands could not be broadcast together with shapes (12,12) (0,) 



### Error 11, [Traceback at line 8041](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L8041)<br />8041..Traceback (most recent call last):
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



### Error 12, [Traceback at line 8053](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L8053)<br />8053..Traceback (most recent call last):
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



### Error 13, [Traceback at line 8104](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L8104)<br />8104..Traceback (most recent call last):
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



### Error 14, [Traceback at line 8115](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L8115)<br />8115..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_tf//1_lstm.py", line 332, in <module>
<br />    test(data_path="", pars_choice="test01", config_mode="test")
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_tf//1_lstm.py", line 320, in test
<br />    session = load(out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_tf//1_lstm.py", line 199, in load
<br />    return load_tf(load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/util.py", line 477, in load_tf
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
<br />[master 795ed49] ml_store
<br /> 1 file changed, 233 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   375e151..795ed49  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_tf//temporal_fusion_google.py 



### Error 15, [Traceback at line 8196](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-14-20-11_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L8196)<br />8196..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
<br />    from mlmodels.mode_tf.raw  import temporal_fusion_google
<br />ModuleNotFoundError: No module named 'mlmodels.mode_tf'
