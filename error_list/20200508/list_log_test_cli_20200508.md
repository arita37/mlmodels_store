## Original File URL: https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt


### Error 1, [Traceback at line 215](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L215)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 941, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.D:\\_devs\\Python01\\gitdev\\mlmodels\\mlmodels\\model_tf'



### Error 2, [Traceback at line 231](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L231)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 3, [Traceback at line 238](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L238)<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 459, in main
<br />    test_module(arg.model_uri, param_pars=param_pars)  # '1_lstm'
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 250, in test_module
<br />    module = module_load(model_uri)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module D:\_devs\Python01\gitdev\mlmodels\mlmodels\model_tf.1_lstm notfound, No module named 'mlmodels.D:\\_devs\\Python01\\gitdev\\mlmodels\\mlmodels\\model_tf', tuple index out of range



### Error 4, [Traceback at line 258](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L258)<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 414, in main
<br />    js = json.load(open(arg.config_file, mode='rb'))  # Config
<br />FileNotFoundError: [Errno 2] No such file or directory: 'optim_config.json'



### Error 5, [Traceback at line 266](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L266)<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/optim.py", line 414, in main
<br />    js = json.load(open(arg.config_file, mode='rb'))  # Config
<br />FileNotFoundError: [Errno 2] No such file or directory: 'optim_config.json'



### Error 6, [Traceback at line 304](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L304)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/fb_prophet.py", line 89, in fit
<br />    train_df, test_df = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/fb_prophet.py", line 32, in get_dataset
<br />    train_df = pd.read_csv(data_pars["train_data_path"], parse_dates=True)[col]
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
<br />FileNotFoundError: [Errno 2] File b'dataset/timeseries/stock/qqq_us_train.csv' does not exist: b'dataset/timeseries/stock/qqq_us_train.csv'



### Error 7, [Traceback at line 1099](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1099)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 8, [Traceback at line 1129](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1129)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 9, [Traceback at line 1136](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1136)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 10, [Traceback at line 1142](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1142)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 11, [Traceback at line 1172](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1172)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 12, [Traceback at line 1179](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1179)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 13, [Traceback at line 1185](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1185)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 14, [Traceback at line 1215](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1215)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 15, [Traceback at line 1237](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1237)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 16, [Traceback at line 1243](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1243)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 17, [Traceback at line 1273](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1273)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 18, [Traceback at line 1280](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1280)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 19, [Traceback at line 1286](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1286)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 20, [Traceback at line 1316](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1316)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 21, [Traceback at line 1323](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1323)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 22, [Traceback at line 1329](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1329)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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
<br />  ({'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}}, NameError('Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range',)) 
<br />  ("### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
<br />  ('#### Model URI and Config JSON',) 
<br />  ({'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}},) 
<br />  ('#### Setup Model   ##############################################',) 
<br />  ({'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}}, NameError('Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range',)) 
<br />  ("### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} #####",) 
<br />  ('#### Model URI and Config JSON',) 
<br />  ({'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}},) 
<br />  ('#### Setup Model   ##############################################',) 
<br />  ({'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}}, NameError('Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range',)) 
<br />  ('benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/',) 
<br />  (                     date_run  ...            metric_name
<br />0  2020-05-07 00:21:12.839090  ...    mean_absolute_error
<br />1  2020-05-07 00:21:12.843547  ...     mean_squared_error
<br />2  2020-05-07 00:21:12.847253  ...  median_absolute_error
<br />3  2020-05-07 00:21:12.850692  ...               r2_score
<br />4  2020-05-07 00:21:30.952214  ...    mean_absolute_error
<br />5  2020-05-07 00:21:30.958663  ...     mean_squared_error
<br />6  2020-05-07 00:21:30.965988  ...  median_absolute_error
<br />7  2020-05-07 00:21:30.971106  ...               r2_score
<br />
<br />[8 rows x 6 columns],) 
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 23, [Traceback at line 1382](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1382)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 24, [Traceback at line 1389](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1389)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 25, [Traceback at line 1395](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1395)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 26, [Traceback at line 1425](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1425)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 27, [Traceback at line 1432](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1432)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 28, [Traceback at line 1438](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1438)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 15, in <module>
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



### Error 29, [Traceback at line 1468](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1468)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 30, [Traceback at line 1475](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L1475)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 31, [Traceback at line 2275](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L2275)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/fb_prophet.py", line 89, in fit
<br />    train_df, test_df = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/fb_prophet.py", line 32, in get_dataset
<br />    train_df = pd.read_csv(data_pars["train_data_path"], parse_dates=True)[col]
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
<br />FileNotFoundError: [Errno 2] File b'dataset/timeseries/stock/qqq_us_train.csv' does not exist: b'dataset/timeseries/stock/qqq_us_train.csv'



### Error 32, [Traceback at line 2295](https://github.com/suyogdahal/mlmodels_store/blob/master/log_test_cli/log_2020-05-07_00:20:39,205.txt#L2295)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 114, in benchmark_run
<br />    model_uri =  model_pars['model_uri']
<br />KeyError: 'model_uri'
