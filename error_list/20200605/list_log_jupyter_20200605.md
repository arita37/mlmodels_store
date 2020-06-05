## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py


### Error 1, [Traceback at line 45](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L45)<br />45..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 2, [Traceback at line 57](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L57)<br />57..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 3, [Traceback at line 64](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L64)<br />64..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//sklearn_titanic_svm.py", line 20, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 4, [Traceback at line 80](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L80)<br />80..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 674, in exec_module
<br />  File "<frozen importlib._bootstrap_external>", line 781, in get_code
<br />  File "<frozen importlib._bootstrap_external>", line 741, in source_to_code
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/model_sklearn/model_lightgbm.py", line 316
<br />    else:
<br />       ^
<br />SyntaxError: invalid syntax



### Error 5, [Traceback at line 100](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L100)<br />100..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 6, [Traceback at line 107](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L107)<br />107..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//lightgbm.py", line 23, in <module>
<br />    module        =  module_load( model_uri= model_uri)
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.model_lightgbm notfound, invalid syntax (model_lightgbm.py, line 316), tuple index out of range



### Error 7, [Traceback at line 123](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L123)<br />123..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 8, [Traceback at line 135](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L135)<br />135..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 9, [Traceback at line 142](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L142)<br />142..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//sklearn_titanic_randomForest.py", line 21, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 10, [Traceback at line 183](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L183)<br />183..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//lightgbm_home_retail.py", line 21, in <module>
<br />    pars = json.load(open( data_path , mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'hyper_lightgbm_home_retail.json'



### Error 11, [Traceback at line 197](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L197)<br />197..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//keras_charcnn_reuters.py", line 28, in <module>
<br />    pars = json.load(open( config_path , mode='r'))[config_mode]
<br />FileNotFoundError: [Errno 2] No such file or directory: 'reuters_charcnn.json'



### Error 12, [Traceback at line 213](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L213)<br />213..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/urllib/request.py", line 1318, in do_open
<br />    encode_chunked=req.has_header('Transfer-encoding'))
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/http/client.py", line 1262, in request
<br />    self._send_request(method, url, body, headers, encode_chunked)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/http/client.py", line 1308, in _send_request
<br />    self.endheaders(body, encode_chunked=encode_chunked)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/http/client.py", line 1257, in endheaders
<br />    self._send_output(message_body, encode_chunked=encode_chunked)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/http/client.py", line 1036, in _send_output
<br />    self.send(msg)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/http/client.py", line 974, in send
<br />    self.connect()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/http/client.py", line 1423, in connect
<br />    server_hostname=server_hostname)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/ssl.py", line 407, in wrap_socket
<br />    _context=self, _session=session)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/ssl.py", line 817, in __init__
<br />    self.do_handshake()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/ssl.py", line 1077, in do_handshake
<br />    self._sslobj.do_handshake()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/ssl.py", line 689, in do_handshake
<br />    self._sslobj.do_handshake()
<br />ConnectionResetError: [Errno 104] Connection reset by peer



### Error 13, [Traceback at line 240](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L240)<br />240..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//gluon_automl.py", line 59, in <module>
<br />    model   =  fit(model, data_pars=data_pars, model_pars=model_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/model_gluon/util_autogluon.py", line 128, in fit
<br />    data  = get_dataset(**data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/model_gluon/util_autogluon.py", line 101, in get_dataset
<br />    data, label = _get_dataset_from_aws(**kw)
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/model_gluon/util_autogluon.py", line 41, in _get_dataset_from_aws
<br />    data = tabular_task.Dataset(file_path=URL_INC_TRAIN)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/task/tabular_prediction/dataset.py", line 84, in __init__
<br />    df = load_pd.load(file_path)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/utils/loaders/load_pd.py", line 68, in load
<br />    error_bad_lines=error_bad_lines, low_memory=low_memory, nrows=nrows, skiprows=skiprows, usecols=usecols)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
<br />    return _read(filepath_or_buffer, kwds)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 440, in _read
<br />    filepath_or_buffer, encoding, compression
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/common.py", line 196, in get_filepath_or_buffer
<br />    req = urlopen(filepath_or_buffer)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/urllib/request.py", line 223, in urlopen
<br />    return opener.open(url, data, timeout)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/urllib/request.py", line 526, in open
<br />    response = self._open(req, data)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/urllib/request.py", line 544, in _open
<br />    '_open', req)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/urllib/request.py", line 504, in _call_chain
<br />    result = func(*args)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/urllib/request.py", line 1361, in https_open
<br />    context=self._context, check_hostname=self._check_hostname)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/urllib/request.py", line 1320, in do_open
<br />    raise URLError(err)
<br />urllib.error.URLError: <urlopen error [Errno 104] Connection reset by peer>
<br />
<br />
<br />
<br />
<br />
<br /> ********************************************************************************************************************************************
<br />https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vison_fashion_MNIST.ipynb 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//vison_fashion_MNIST.ipynb to script
<br />[NbConvertApp] Writing 1929 bytes to https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example/vison_fashion_MNIST.txt
<br />python: can't open file 'https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//vison_fashion_MNIST.py': [Errno 2] No such file or directory
<br />No replacement
<br />
<br />
<br />
<br />
<br />
<br /> ********************************************************************************************************************************************
<br />https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//tensorflow_1_lstm.ipynb 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//tensorflow_1_lstm.ipynb to script
<br />[NbConvertApp] Writing 1829 bytes to https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example/tensorflow_1_lstm.py
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />non-resource variables are not supported in the long term
<br />/home/runner/work/mlmodels/mlmodels
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
<br />https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />          0         1         2         3         4         5
<br />0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
<br />1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
<br />2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
<br />3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
<br />4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
<br />https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />          0         1         2         3         4         5
<br />0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
<br />1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
<br />2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
<br />3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
<br />4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
<br />5  0.745928  0.883387  0.838176  0.904464  0.904464  0.370110
<br />6  1.000000  0.881878  0.467996  0.486496  0.486496  1.000000
<br />7  0.216516  0.077549  0.433808  0.329598  0.329598  0.318466
<br />8  0.195249  0.000000  0.000000  0.000000  0.000000  0.671960
<br />9  0.000000  0.173783  0.369041  0.411721  0.411721  0.304384
<br />
<br />
<br />
<br />
<br />
<br /> ********************************************************************************************************************************************
<br />https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vision_mnist.ipynb 
<br />
<br />[NbConvertApp] Converting notebook https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//vision_mnist.ipynb to script
<br />[NbConvertApp] Writing 7429 bytes to https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example/vision_mnist.txt
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//vision_mnist.py", line 15
<br />    !git clone https://github.com/ahmed3bbas/mlmodels.git
<br />    ^
<br />SyntaxError: invalid syntax



### Error 14, [Traceback at line 360](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L360)<br />360..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//lightgbm_glass.py", line 16, in <module>
<br />    print( os.getcwd())
<br />NameError: name 'os' is not defined



### Error 15, [Traceback at line 409](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L409)<br />409..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//keras-textcnn.py", line 37, in <module>
<br />    _, _   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/model_keras/textcnn.py", line 69, in fit
<br />    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/model_keras/textcnn.py", line 143, in get_dataset
<br />    maxlen       = data_pars['data_info']['maxlen']
<br />KeyError: 'data_info'



### Error 16, [Traceback at line 428](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L428)<br />428..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//sklearn_titanic_randomForest_example2.py", line 22, in <module>
<br />    pars = json.load(open( data_path , mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: '../mlmodels/dataset/json/hyper_titanic_randomForest.json'



### Error 17, [Traceback at line 456](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L456)<br />456..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//gluon_automl_titanic.py", line 27, in <module>
<br />    data_path= '../mlmodels/dataset/json/gluon_automl.json'
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/model_gluon/gluon_automl.py", line 82, in get_params
<br />    with open(data_path, encoding='utf-8') as config_f:
<br />FileNotFoundError: [Errno 2] No such file or directory: '../mlmodels/dataset/json/gluon_automl.json'



### Error 18, [Traceback at line 472](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L472)<br />472..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//tensorflow__lstm_json.py", line 13, in <module>
<br />    print( os.getcwd())
<br />NameError: name 'os' is not defined



### Error 19, [Traceback at line 486](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L486)<br />486..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 20, [Traceback at line 498](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L498)<br />498..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 21, [Traceback at line 505](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L505)<br />505..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 22, [Traceback at line 522](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L522)<br />522..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//lightgbm_titanic.py", line 21, in <module>
<br />    pars = json.load(open( data_path , mode='r'))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'hyper_lightgbm_titanic.json'
<br />SyntaxError: invalid syntax



### Error 23, [Traceback at line 554](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L554)<br />554..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//arun_hyper.py", line 2, in <module>
<br />    from jsoncomment import JsonComment ; json = JsonComment(), copy
<br />NameError: name 'copy' is not defined



### Error 24, [Traceback at line 566](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L566)<br />566..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//lightgbm_glass.py", line 16, in <module>
<br />    print( os.getcwd())
<br />NameError: name 'os' is not defined



### Error 25, [Traceback at line 578](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L578)<br />578..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 26, [Traceback at line 590](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L590)<br />590..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 27, [Traceback at line 597](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L597)<br />597..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//benchmark_timeseries_m5.py", line 27, in <module>
<br />    import mxnet as mx
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/__init__.py", line 31, in <module>
<br />    from . import contrib
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/__init__.py", line 31, in <module>
<br />    from . import onnx
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/__init__.py", line 19, in <module>
<br />    from .onnx2mx.import_model import import_model, get_model_metadata
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/__init__.py", line 20, in <module>
<br />    from . import import_model
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_model.py", line 22, in <module>
<br />    from .import_onnx import GraphProto
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_onnx.py", line 26, in <module>
<br />    from ._import_helper import _convert_map as convert_map
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_import_helper.py", line 21, in <module>
<br />    from ._op_translations import identity, random_uniform, random_normal, sample_multinomial
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_op_translations.py", line 22, in <module>
<br />    from . import _translation_utils as translation_utils
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_translation_utils.py", line 23, in <module>
<br />    from .... import  module
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/module/__init__.py", line 22, in <module>
<br />    from .base_module import BaseModule
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/module/base_module.py", line 31, in <module>
<br />    from ..model import BatchEndParam
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/model.py", line 46, in <module>
<br />    from sklearn.base import BaseEstimator
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example/sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range



### Error 28, [Traceback at line 639](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L639)<br />639..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example//arun_model.py", line 27, in <module>
<br />    pars = json.load(open(config_path , mode='r'))[config_mode]
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/model_keras/ardmn.json'



### Error 29, [Traceback at line 659](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L659)<br />659..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_sklearn.sklearn'



### Error 30, [Traceback at line 671](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L671)<br />671..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 31, [Traceback at line 678](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L678)<br />678..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example/benchmark_timeseries_m5.py", line 27, in <module>
<br />    import mxnet as mx
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/__init__.py", line 31, in <module>
<br />    from . import contrib
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/__init__.py", line 31, in <module>
<br />    from . import onnx
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/__init__.py", line 19, in <module>
<br />    from .onnx2mx.import_model import import_model, get_model_metadata
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/__init__.py", line 20, in <module>
<br />    from . import import_model
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_model.py", line 22, in <module>
<br />    from .import_onnx import GraphProto
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/import_onnx.py", line 26, in <module>
<br />    from ._import_helper import _convert_map as convert_map
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_import_helper.py", line 21, in <module>
<br />    from ._op_translations import identity, random_uniform, random_normal, sample_multinomial
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_op_translations.py", line 22, in <module>
<br />    from . import _translation_utils as translation_utils
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/contrib/onnx/onnx2mx/_translation_utils.py", line 23, in <module>
<br />    from .... import  module
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/module/__init__.py", line 22, in <module>
<br />    from .base_module import BaseModule
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/module/base_module.py", line 31, in <module>
<br />    from ..model import BatchEndParam
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/model.py", line 46, in <module>
<br />    from sklearn.base import BaseEstimator
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/example/sklearn.py", line 34, in <module>
<br />    module        =  module_load( model_uri= model_uri )                           # Load file definition
<br />  File "https://github.com/arita37/mlmodels/tree/31d2b88611b4231ac14a6179e72e5d4a4ee459ac/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range
