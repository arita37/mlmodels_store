## Original File URL: https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt


### Error 1, [Traceback at line 21](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L21)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
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



### Error 2, [Traceback at line 798](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L798)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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



### Error 3, [Traceback at line 828](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L828)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 4, [Traceback at line 835](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L835)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 5, [Traceback at line 841](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L841)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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



### Error 6, [Traceback at line 871](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L871)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 7, [Traceback at line 878](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L878)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 8, [Traceback at line 884](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L884)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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



### Error 9, [Traceback at line 914](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L914)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 10, [Traceback at line 936](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L936)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 11, [Traceback at line 942](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L942)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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



### Error 12, [Traceback at line 972](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L972)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 13, [Traceback at line 979](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L979)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 14, [Traceback at line 985](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L985)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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



### Error 15, [Traceback at line 1015](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1015)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 16, [Traceback at line 1022](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1022)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 17, [Traceback at line 1028](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1028)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
<br />  ('benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/timeseries/test02/model_list.json',) 
<br />  (                     date_run  ...            metric_name
<br />0  2020-05-06 17:12:06.210428  ...    mean_absolute_error
<br />1  2020-05-06 17:12:06.214948  ...     mean_squared_error
<br />2  2020-05-06 17:12:06.380202  ...  median_absolute_error
<br />3  2020-05-06 17:12:06.384623  ...               r2_score
<br />4  2020-05-06 17:12:30.634413  ...    mean_absolute_error
<br />5  2020-05-06 17:12:30.640197  ...     mean_squared_error
<br />6  2020-05-06 17:12:30.646839  ...  median_absolute_error
<br />7  2020-05-06 17:12:30.652047  ...               r2_score
<br />
<br />[8 rows x 6 columns],) 
<br />  File "pydantic/main.py", line 778, in pydantic.main.create_model
<br />TypeError: create_model() takes exactly 1 positional argument (0 given)



### Error 18, [Traceback at line 1081](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1081)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 19, [Traceback at line 1088](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1088)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 20, [Traceback at line 1094](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1094)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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



### Error 21, [Traceback at line 1124](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1124)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 22, [Traceback at line 1131](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1131)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 23, [Traceback at line 1137](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1137)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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



### Error 24, [Traceback at line 1167](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1167)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 25, [Traceback at line 1174](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1174)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range



### Error 26, [Traceback at line 1266](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1266)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 198, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 175, in get_dataset
<br />    train_loader, valid_loader  = get_dataset_torch(data_pars)
<br />TypeError: get_dataset_torch() missing 1 required positional argument: 'data_info'



### Error 27, [Traceback at line 1275](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1275)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 198, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 175, in get_dataset
<br />    train_loader, valid_loader  = get_dataset_torch(data_pars)
<br />TypeError: get_dataset_torch() missing 1 required positional argument: 'data_info'



### Error 28, [Traceback at line 1284](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1284)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 198, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 175, in get_dataset
<br />    train_loader, valid_loader  = get_dataset_torch(data_pars)
<br />TypeError: get_dataset_torch() missing 1 required positional argument: 'data_info'



### Error 29, [Traceback at line 1293](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1293)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 198, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 175, in get_dataset
<br />    train_loader, valid_loader  = get_dataset_torch(data_pars)
<br />TypeError: get_dataset_torch() missing 1 required positional argument: 'data_info'



### Error 30, [Traceback at line 1302](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1302)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 198, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 175, in get_dataset
<br />    train_loader, valid_loader  = get_dataset_torch(data_pars)
<br />TypeError: get_dataset_torch() missing 1 required positional argument: 'data_info'



### Error 31, [Traceback at line 1311](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1311)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 198, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 175, in get_dataset
<br />    train_loader, valid_loader  = get_dataset_torch(data_pars)
<br />TypeError: get_dataset_torch() missing 1 required positional argument: 'data_info'



### Error 32, [Traceback at line 1320](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1320)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 198, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 175, in get_dataset
<br />    train_loader, valid_loader  = get_dataset_torch(data_pars)
<br />TypeError: get_dataset_torch() missing 1 required positional argument: 'data_info'



### Error 33, [Traceback at line 1329](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1329)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 198, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 175, in get_dataset
<br />    train_loader, valid_loader  = get_dataset_torch(data_pars)
<br />TypeError: get_dataset_torch() missing 1 required positional argument: 'data_info'



### Error 34, [Traceback at line 1338](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1338)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 198, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 175, in get_dataset
<br />    train_loader, valid_loader  = get_dataset_torch(data_pars)
<br />TypeError: get_dataset_torch() missing 1 required positional argument: 'data_info'



### Error 35, [Traceback at line 1347](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1347)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 198, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 175, in get_dataset
<br />    train_loader, valid_loader  = get_dataset_torch(data_pars)
<br />TypeError: get_dataset_torch() missing 1 required positional argument: 'data_info'



### Error 36, [Traceback at line 1356](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1356)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 198, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 175, in get_dataset
<br />    train_loader, valid_loader  = get_dataset_torch(data_pars)
<br />TypeError: get_dataset_torch() missing 1 required positional argument: 'data_info'



### Error 37, [Traceback at line 1367](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1367)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 280, in <module>
<br />    main()
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 277, in main
<br />    raise Exception("No options")
<br />Exception: No options
<br />  ('\n\n\n',) 
<br />  ('python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do text_classification ',) 
<br />  ('text_classification',) 
<br />  ('Model List', [{'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}}, {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}}]) 
<br />  ("### Running {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} #####",) 
<br />  ('#### Model URI and Config JSON',) 
<br />  ({'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2},) 
<br />  ('#### Setup Model   ##############################################',) 
<br />{'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}
<br />  ('#### Fit  #######################################################',) 
<br />>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7eff36c01be0> <class 'mlmodels.model_tch.textcnn.Model'>
<br />Spliting original file to train/valid set...
<br />  ({'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': True}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}}, FileNotFoundError(2, 'No such file or directory')) 
<br />  ("### Running {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} #####",) 
<br />  ('#### Model URI and Config JSON',) 
<br />  ({'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50},) 
<br />  ('#### Setup Model   ##############################################',) 



### Error 38, [Traceback at line 1390](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1390)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 291, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 332, in get_dataset
<br />    split_train_valid( path, data_pars['train_path'], data_pars['valid_path'], frac )
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 119, in split_train_valid
<br />    tr.to_csv(path_train, index=False)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/core/generic.py", line 3228, in to_csv
<br />    formatter.save()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/formats/csvs.py", line 183, in save
<br />    compression=self.compression,
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/common.py", line 399, in _get_handle
<br />    f = open(path_or_buf, mode, encoding=encoding, newline="")
<br />FileNotFoundError: [Errno 2] No such file or directory: 'dataset/recommender/IMDB_train.csv'



### Error 39, [Traceback at line 1526](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1526)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 116, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textvae.py", line 51, in __init__
<br />    texts, embeddings_index = get_dataset(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textvae.py", line 269, in get_dataset
<br />    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
<br />    file = builtins.open(filename, mode, buffering)
<br />FileNotFoundError: [Errno 2] No such file or directory: 'dataset/text/quora/train.csv'



### Error 40, [Traceback at line 1536](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1536)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 116, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.py", line 64, in __init__
<br />    df = get_dataset(data_pars)[0]
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.py", line 191, in get_dataset
<br />    return _preprocess_test(data_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.py", line 207, in _preprocess_test
<br />    df = pd.read_csv(data_pars['path'], encoding="ISO-8859-1")
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
<br />FileNotFoundError: [Errno 2] File b'dataset/text/ner_dataset.csv' does not exist: b'dataset/text/ner_dataset.csv'



### Error 41, [Traceback at line 1558](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1558)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/Autokeras.py", line 12, in <module>
<br />    import autokeras as ak
<br />ModuleNotFoundError: No module named 'autokeras'



### Error 42, [Traceback at line 1575](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1575)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 43, [Traceback at line 1582](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1582)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range



### Error 44, [Traceback at line 1696](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1696)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 136, in benchmark_run
<br />    metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 60, in metric_eval
<br />    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
<br />AttributeError: module 'sklearn.metrics' has no attribute 'accuracy, f1_score'



### Error 45, [Traceback at line 1702](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1702)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
<br />  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
<br />  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
<br />  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_classifier.py", line 39, in <module>
<br />    from util_transformer import (convert_examples_to_features, output_modes,
<br />ModuleNotFoundError: No module named 'util_transformer'



### Error 46, [Traceback at line 1719](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1719)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 47, [Traceback at line 1726](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1726)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 115, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.transformer_classifier notfound, No module named 'util_transformer', tuple index out of range



### Error 48, [Traceback at line 1732](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1732)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.py", line 164, in fit
<br />    output_path      = out_pars["model_path"]
<br />KeyError: 'model_path'



### Error 49, [Traceback at line 1738](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1738)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 122, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 291, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 332, in get_dataset
<br />    split_train_valid( path, data_pars['train_path'], data_pars['valid_path'], frac )
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 119, in split_train_valid
<br />    tr.to_csv(path_train, index=False)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/core/generic.py", line 3228, in to_csv
<br />    formatter.save()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/formats/csvs.py", line 183, in save
<br />    compression=self.compression,
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/common.py", line 399, in _get_handle
<br />    f = open(path_or_buf, mode, encoding=encoding, newline="")
<br />FileNotFoundError: [Errno 2] No such file or directory: 'dataset/recommender/IMDB_train.csv'



### Error 50, [Traceback at line 1754](https://github.com/suyogdahal/mlmodels_store/blob/master/log_benchmark/log_benchmark_2020-05-06_17:11:54,584.txt#L1754)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 116, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
<br />    mpars =json_norm(model_pars['model_pars'])
<br />KeyError: 'model_pars'
