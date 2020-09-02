## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 445](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L445)<br />445..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/dataloader.py", line 452, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/dataloader.py", line 258, in compute
<br />    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/preprocess/generic.py", line 502, in __init__
<br />    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
<br />    return _read(filepath_or_buffer, kwds)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
<br />    parser = TextFileReader(fp_or_buf, **kwds)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
<br />    self._make_engine(self.engine)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
<br />    self._engine = CParserWrapper(self.f, **self.options)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
<br />    self._reader = parsers.TextReader(src, **kwds)
<br />  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
<br />  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
<br />FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'



### Error 2, [Traceback at line 465](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L465)<br />465..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/dataloader.py", line 452, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/dataloader.py", line 258, in compute
<br />    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/preprocess/generic.py", line 502, in __init__
<br />    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
<br />    return _read(filepath_or_buffer, kwds)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
<br />    parser = TextFileReader(fp_or_buf, **kwds)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
<br />    self._make_engine(self.engine)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
<br />    self._engine = CParserWrapper(self.f, **self.options)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
<br />    self._reader = parsers.TextReader(src, **kwds)
<br />  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
<br />  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
<br />FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'



### Error 3, [Traceback at line 485](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L485)<br />485..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/dataloader.py", line 452, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/dataloader.py", line 258, in compute
<br />    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/preprocess/generic.py", line 607, in __init__
<br />    data            = np.load( file_path,**args.get("numpy_loader_args", {}))
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
<br />    fid = stack.enter_context(open(os_fspath(file), "rb"))
<br />FileNotFoundError: [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'



### Error 4, [Traceback at line 496](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L496)<br />496..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/dataloader.py", line 452, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/dataloader.py", line 301, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/dataloader.py", line 93, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



### Error 5, [Traceback at line 2460](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L2460)<br />2460..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/util.py", line 672, in load_function_uri
<br />    return  getattr(importlib.import_module(package), name)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/model_tch/textcnn.py", line 24, in <module>
<br />    import torchtext
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
<br />    _init_extension()
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
<br />    torch.ops.load_library(ext_specs.origin)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
<br />    ctypes.CDLL(path)
<br />  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
<br />    self._handle = _dlopen(self._name, mode)
<br />OSError: libtorch_cpu.so: cannot open shared object file: No such file or directory



### Error 6, [Traceback at line 2485](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L2485)<br />2485..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/util.py", line 683, in load_function_uri
<br />    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 7, [Traceback at line 2492](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L2492)<br />2492..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/dataloader.py", line 452, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/dataloader.py", line 248, in compute
<br />    preprocessor_func = load_function(uri)
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/util.py", line 688, in load_function_uri
<br />    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
<br />NameError: Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range



### Error 8, [Traceback at line 2500](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L2500)<br />2500..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/util.py", line 672, in load_function_uri
<br />    return  getattr(importlib.import_module(package), name)
<br />AttributeError: module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader'



### Error 9, [Traceback at line 2507](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L2507)<br />2507..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/util.py", line 683, in load_function_uri
<br />    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 10, [Traceback at line 2514](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L2514)<br />2514..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/dataloader.py", line 452, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/dataloader.py", line 275, in compute
<br />    obj_preprocessor.compute(input_tmp)
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/model_tch/util_transformer.py", line 275, in compute
<br />    test_func = load_function(self.test_reader)
<br />  File "https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7/mlmodels/util.py", line 688, in load_function_uri
<br />    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
<br />NameError: Module ['sentence_transformers.readers', 'STSBenchmarkDataReader'] notfound, module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader', tuple index out of range
