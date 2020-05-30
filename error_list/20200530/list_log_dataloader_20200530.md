## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py


### Error 1, [Traceback at line 445](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L445)<br />445..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b/mlmodels/dataloader.py", line 445, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b/mlmodels/dataloader.py", line 257, in compute
<br />    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
<br />  File "https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b/mlmodels/preprocess/generic.py", line 502, in __init__
<br />    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
<br />    return _read(filepath_or_buffer, kwds)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
<br />    parser = TextFileReader(fp_or_buf, **kwds)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
<br />    self._make_engine(self.engine)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
<br />    self._engine = CParserWrapper(self.f, **self.options)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
<br />    self._reader = parsers.TextReader(src, **kwds)
<br />  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
<br />  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
<br />FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'



### Error 2, [Traceback at line 465](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L465)<br />465..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b/mlmodels/dataloader.py", line 445, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b/mlmodels/dataloader.py", line 257, in compute
<br />    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
<br />  File "https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b/mlmodels/preprocess/generic.py", line 502, in __init__
<br />    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
<br />    return _read(filepath_or_buffer, kwds)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
<br />    parser = TextFileReader(fp_or_buf, **kwds)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
<br />    self._make_engine(self.engine)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
<br />    self._engine = CParserWrapper(self.f, **self.options)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
<br />    self._reader = parsers.TextReader(src, **kwds)
<br />  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
<br />  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
<br />FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'



### Error 3, [Traceback at line 485](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L485)<br />485..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b/mlmodels/dataloader.py", line 445, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b/mlmodels/dataloader.py", line 257, in compute
<br />    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
<br />  File "https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b/mlmodels/preprocess/generic.py", line 607, in __init__
<br />    data            = np.load( file_path,**args.get("numpy_loader_args", {}))
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 428, in load
<br />    fid = open(os_fspath(file), "rb")
<br />FileNotFoundError: [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'



### Error 4, [Traceback at line 496](https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py#L496)<br />496..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b/mlmodels/dataloader.py", line 445, in test_json_list
<br />    loader.compute()
<br />  File "https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b/mlmodels/dataloader.py", line 297, in compute
<br />    out_tmp = preprocessor_func(input_tmp, **args)
<br />  File "https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b/mlmodels/dataloader.py", line 92, in pickle_dump
<br />    with open(kwargs["path"], "wb") as fi:
<br />FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
