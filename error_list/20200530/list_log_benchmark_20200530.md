## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py


### Error 1, [Traceback at line 212](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L212)<br />212..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.armdn'



### Error 2, [Traceback at line 224](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L224)<br />224..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 3, [Traceback at line 231](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L231)<br />231..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.armdn notfound, No module named 'mlmodels.model_keras.armdn', tuple index out of range



### Error 4, [Traceback at line 237](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L237)<br />237..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.nbeats'



### Error 5, [Traceback at line 249](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L249)<br />249..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 6, [Traceback at line 256](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L256)<br />256..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.nbeats notfound, No module named 'mlmodels.model_tch.nbeats', tuple index out of range



### Error 7, [Traceback at line 290](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L290)<br />290..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 95, in __init__
<br />    if "NegativeBinomialOutput" in  mpars['distr_output'] :
<br />KeyError: 'distr_output'



### Error 8, [Traceback at line 314](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L314)<br />314..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 237, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 139, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 192, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 9, [Traceback at line 343](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L343)<br />343..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 237, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 139, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 192, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 10, [Traceback at line 371](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L371)<br />371..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 237, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 139, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 192, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 11, [Traceback at line 399](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L399)<br />399..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 237, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 139, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 192, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 12, [Traceback at line 427](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L427)<br />427..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 237, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 139, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 192, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 13, [Traceback at line 469](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L469)<br />469..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 237, in fit
<br />    train_ds, test_ds, cardinalities = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 139, in get_dataset
<br />    return get_dataset_single(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 192, in get_dataset_single
<br />    data_path=data_pars['data_path']
<br />KeyError: 'data_path'



### Error 14, [Traceback at line 479](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L479)<br />479..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_gluon/gluonts_model.py", line 90, in __init__
<br />    mpars['encoder'] = MLPEncoder()   #bug in seq2seq
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 424, in init_wrapper
<br />    model = PydanticModel(**{**nmargs, **kwargs})
<br />  File "pydantic/main.py", line 283, in pydantic.main.BaseModel.__init__
<br />pydantic.error_wrappers.ValidationError: 1 validation error for MLPEncoderModel
<br />layer_sizes
<br />  field required (type=value_error.missing)
<br />python https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py --do timeseries 
<br />
<br />
<br />
<br />
<br />
<br /> ********************************************************************************************************************************************
<br />
<br />  vision_mnist 
<br />
<br />  json_path https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/json/benchmark_cnn/mnist 
<br />
<br />  Model List [{'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}}] 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnet18/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1e66aea0f0> <class 'mlmodels.model_tch.torchhub.Model'>
<br />
<br />  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnet18/'}} 'data_info' 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnet34/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1e16002fd0> <class 'mlmodels.model_tch.torchhub.Model'>
<br />
<br />  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnet34/'}} 'data_info' 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1e667c5438> <class 'mlmodels.model_tch.torchhub.Model'>
<br />
<br />  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} 'data_info' 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1e196ed400> <class 'mlmodels.model_tch.torchhub.Model'>
<br />
<br />  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} 'data_info' 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnet101/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1e66aea0f0> <class 'mlmodels.model_tch.torchhub.Model'>
<br />
<br />  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnet101/'}} 'data_info' 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1e16002fd0> <class 'mlmodels.model_tch.torchhub.Model'>
<br />
<br />  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} 'data_info' 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1e667c5438> <class 'mlmodels.model_tch.torchhub.Model'>
<br />
<br />  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} 'data_info' 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1e196ed400> <class 'mlmodels.model_tch.torchhub.Model'>
<br />
<br />  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} 'data_info' 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnet50/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1e66afdf60> <class 'mlmodels.model_tch.torchhub.Model'>
<br />
<br />  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnet50/'}} 'data_info' 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnet152/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1e0a199780> <class 'mlmodels.model_tch.torchhub.Model'>
<br />
<br />  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/resnet152/'}} 'data_info' 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 
<br />Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip



### Error 15, [Traceback at line 677](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L677)<br />677..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 16, [Traceback at line 688](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L688)<br />688..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 17, [Traceback at line 699](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L699)<br />699..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 18, [Traceback at line 710](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L710)<br />710..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 19, [Traceback at line 721](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L721)<br />721..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 20, [Traceback at line 732](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L732)<br />732..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 21, [Traceback at line 743](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L743)<br />743..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 22, [Traceback at line 754](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L754)<br />754..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 23, [Traceback at line 765](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L765)<br />765..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 24, [Traceback at line 776](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L776)<br />776..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 25, [Traceback at line 787](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L787)<br />787..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 222, in fit
<br />    train_iter, valid_iter = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1e679c2438> <class 'mlmodels.model_tch.torchhub.Model'>
<br />
<br />  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} 'data_info' 
<br />
<br />  benchmark file saved at https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/example/benchmark/cnn/mnist 
<br />
<br />  Empty DataFrame
<br />Columns: [date_run, model_uri, json, dataset_uri, metric, metric_name]
<br />Index: [] 
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 26, [Traceback at line 815](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L815)<br />815..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 284, in <module>
<br />    main()
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 281, in main
<br />    raise Exception("No options")
<br />Exception: No options
<br />python https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py --do fashion_vision_mnist 
<br />
<br />
<br />
<br />
<br />
<br /> ********************************************************************************************************************************************
<br />
<br />  text_classification 
<br />
<br />  json_path https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/json/benchmark_text_classification/model_list_bench01.json 
<br />
<br />  Model List [{'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}}, {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}}] 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64} {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />{'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb3e0fb0ac8> <class 'mlmodels.model_tch.textcnn.Model'>
<br />
<br />  {'hypermodel_pars': {}, 'data_pars': {'data_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': True}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} 'data_info' 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'path': 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5} {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'} 
<br />
<br />  #### Setup Model   ############################################## 



### Error 27, [Traceback at line 862](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L862)<br />862..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/textcnn.py", line 361, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/textcnn.py", line 414, in get_dataset
<br />    dataset        = data_pars['data_info'].get('dataset', None)
<br />KeyError: 'data_info'



### Error 28, [Traceback at line 916](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L916)<br />916..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_keras/textcnn.py", line 69, in fit
<br />    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_keras/textcnn.py", line 143, in get_dataset
<br />    maxlen       = data_pars['data_info']['maxlen']
<br />KeyError: 'data_info'



### Error 29, [Traceback at line 978](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L978)<br />978..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/textcnn.py", line 361, in fit
<br />    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/textcnn.py", line 414, in get_dataset
<br />    dataset        = data_pars['data_info'].get('dataset', None)
<br />KeyError: 'data_info'



### Error 30, [Traceback at line 986](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L986)<br />986..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.matchzoo_models'



### Error 31, [Traceback at line 998](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L998)<br />998..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 32, [Traceback at line 1005](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1005)<br />1005..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.matchzoo_models notfound, No module named 'mlmodels.model_tch.matchzoo_models', tuple index out of range



### Error 33, [Traceback at line 1127](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1127)<br />1127..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_keras/textcnn.py", line 69, in fit
<br />    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_keras/textcnn.py", line 143, in get_dataset
<br />    maxlen       = data_pars['data_info']['maxlen']
<br />KeyError: 'data_info'



### Error 34, [Traceback at line 1135](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1135)<br />1135..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.Autokeras'



### Error 35, [Traceback at line 1147](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1147)<br />1147..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 36, [Traceback at line 1154](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1154)<br />1154..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.Autokeras notfound, No module named 'mlmodels.model_keras.Autokeras', tuple index out of range



### Error 37, [Traceback at line 1160](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1160)<br />1160..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_tch.transformer_classifier'



### Error 38, [Traceback at line 1172](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1172)<br />1172..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 39, [Traceback at line 1179](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1179)<br />1179..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_tch.transformer_classifier notfound, No module named 'mlmodels.model_tch.transformer_classifier', tuple index out of range



### Error 40, [Traceback at line 1185](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1185)<br />1185..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_tch/transformer_sentence.py", line 139, in fit
<br />    train_data       = SentencesDataset(train_reader.get_examples(train_fname),  model=model.model)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/sentence_transformers/readers/NLIDataReader.py", line 21, in get_examples
<br />    mode="rt", encoding="utf-8").readlines()
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/gzip.py", line 53, in open
<br />    binary_file = GzipFile(filename, gz_mode, compresslevel)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/gzip.py", line 163, in __init__
<br />    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
<br />FileNotFoundError: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataset/text/AllNLI/s1.train.gz'



### Error 41, [Traceback at line 1197](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1197)<br />1197..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_keras/namentity_crm_bilstm.py", line 66, in __init__
<br />    data_set, internal_states = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/model_keras/namentity_crm_bilstm.py", line 182, in get_dataset
<br />    loader = DataLoader(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/dataloader.py", line 208, in __init__
<br />    self.data_info                = data_pars['data_info']
<br />KeyError: 'data_info'



### Error 42, [Traceback at line 1207](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1207)<br />1207..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 72, in module_load
<br />    module = import_module(f"mlmodels.{model_name}")
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
<br />    return _bootstrap._gcd_import(name[level:], package, level)
<br />  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
<br />  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
<br />  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
<br />ModuleNotFoundError: No module named 'mlmodels.model_keras.textvae'



### Error 43, [Traceback at line 1219](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1219)<br />1219..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 84, in module_load
<br />    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
<br />IndexError: tuple index out of range



### Error 44, [Traceback at line 1226](https://github.com/arita37/mlmodels_store/blob/master/log_benchmark/log_benchmark.py#L1226)<br />1226..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/benchmark.py", line 119, in benchmark_run
<br />    module    = module_load(model_uri)   # "model_tch.torchhub.py"
<br />  File "https://github.com/arita37/mlmodels/tree/cd0e1dbcf68c34dccf0d76405c260752e880d933/mlmodels/models.py", line 89, in module_load
<br />    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
<br />NameError: Module model_keras.textvae notfound, No module named 'mlmodels.model_keras.textvae', tuple index out of range
