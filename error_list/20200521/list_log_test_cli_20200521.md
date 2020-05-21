## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py


### Error 1, [Traceback at line 155](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L155)<br />155..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1365, in _do_call
<br />    return fn(*args)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1350, in _run_fn
<br />    target_list, run_metadata)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1443, in _call_tf_sessionrun
<br />    run_metadata)
<br />tensorflow.python.framework.errors_impl.NotFoundError: Key Variable not found in checkpoint
<br />	 [[{{node save/RestoreV2}}]]
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 2, [Traceback at line 167](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L167)<br />167..Traceback (most recent call last):
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
<br />	 [[node save/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save/RestoreV2':
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 531, in main
<br />    predict_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 442, in predict_cli
<br />    model, session = load(module, load_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 156, in load
<br />    return module.load(load_pars, **kwarg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
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



### Error 3, [Traceback at line 222](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L222)<br />222..Traceback (most recent call last):
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



### Error 4, [Traceback at line 233](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L233)<br />233..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/models.py", line 531, in main
<br />    predict_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/models.py", line 442, in predict_cli
<br />    model, session = load(module, load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/models.py", line 156, in load
<br />    return module.load(load_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tf/1_lstm.py", line 199, in load
<br />    return load_tf(load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/util.py", line 477, in load_tf
<br />    saver.restore(sess,  full_name)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1306, in restore
<br />    err, "a Variable name or other graph key that is missing")
<br />tensorflow.python.framework.errors_impl.NotFoundError: Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
<br />
<br />Key Variable not found in checkpoint
<br />	 [[node save/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save/RestoreV2':
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 531, in main
<br />    predict_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 442, in predict_cli
<br />    model, session = load(module, load_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 156, in load
<br />    return module.load(load_pars, **kwarg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
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
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ml_models  --do test  --model_uri model_tf.1_lstm  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
<br />test
<br />
<br />  #### Module init   ############################################ 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />non-resource variables are not supported in the long term
<br />
<br />  <module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tf/1_lstm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.model_tf.1_lstm.Model object at 0x7faa349fd080> 
<br />
<br />  #### Fit   ######################################################## 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />
<br />  #### Predict   #################################################### 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />[[ 0.          0.          0.          0.          0.          0.        ]
<br /> [-0.03064011 -0.04365268  0.1043158   0.11423741  0.08334577 -0.06460042]
<br /> [ 0.03496298  0.1086756   0.24618623  0.14885113 -0.21408832  0.14911298]
<br /> [ 0.2660425   0.04396567  0.00228554  0.28030851 -0.17419602 -0.0074782 ]
<br /> [ 0.13223289 -0.01447518  0.17716269  0.19756117  0.04483818  0.16930218]
<br /> [ 0.50343776  0.39873841 -0.06771035 -0.44537422 -0.24181873 -0.51603371]
<br /> [ 0.49752903  0.05757993  0.5349192  -0.40100175 -0.18530777 -0.20865031]
<br /> [ 0.31904012  0.27754715  0.26527962 -0.27625281 -0.10801642  0.17567439]
<br /> [ 0.06051001 -0.01349059 -0.31905201  0.21330933  0.24079493  0.47890407]
<br /> [ 0.          0.          0.          0.          0.          0.        ]]
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />model_tf.1_lstm
<br />model_tf.1_lstm
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />
<br />  #### Model init  ############################################# 
<br />
<br />  #### Model fit   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />
<br />  #### Predict   ##################################################### 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />  #### metrics   ##################################################### 
<br />{'loss': 0.48354315012693405, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-19 23:14:37.950607: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
<br />
<br />Key Variable not found in checkpoint
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 523, in main
<br />    test_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 455, in test_cli
<br />    test(arg.model_uri)  # '1_lstm'
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 189, in test
<br />    module.test()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 320, in test
<br />    session = load(out_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
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
<br />model_tf.1_lstm
<br />model_tf.1_lstm
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />
<br />  #### Model init  ############################################# 
<br />
<br />  #### Model fit   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />
<br />  #### Predict   ##################################################### 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />  #### metrics   ##################################################### 
<br />{'loss': 0.3739481046795845, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-19 23:14:39.285753: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
<br />
<br />Key Variable not found in checkpoint
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 523, in main
<br />    test_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 457, in test_cli
<br />    test_global(arg.model_uri)  # '1_lstm'
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 200, in test_global
<br />    module.test()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 320, in test
<br />    session = load(out_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
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
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ml_models --do test  --model_uri "example/custom_model/1_lstm.py"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
<br />test
<br />
<br />  #### Module init   ############################################ 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />non-resource variables are not supported in the long term
<br />
<br />  <module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/example/custom_model/1_lstm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.example.custom_model.1_lstm.Model object at 0x7f66c181c198> 
<br />
<br />  #### Fit   ######################################################## 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />
<br />  #### Predict   #################################################### 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />[[ 0.          0.          0.          0.          0.          0.        ]
<br /> [ 0.07885583 -0.00373171  0.05546467 -0.01432354 -0.09818617 -0.10162313]
<br /> [ 0.09264664 -0.09637666  0.04646406  0.03806468 -0.12252394  0.04087107]
<br /> [ 0.08858925  0.25261655  0.00572366  0.07998239  0.16537589  0.11442566]
<br /> [ 0.37139058  0.20089009  0.10002682  0.18084611 -0.08755662 -0.05118276]
<br /> [-0.03841517  0.43801039 -0.21904576  0.27503201  0.18857005  0.24082805]
<br /> [ 0.32391226  0.22130108  0.04360529  0.44985536  0.29971647 -0.15271735]
<br /> [ 0.0229946   0.11678867  0.39735913  0.06604046 -0.22405595 -0.36381665]
<br /> [-0.16760972  0.51122379  0.25033885 -0.29558462  0.51426446  0.07490502]
<br /> [ 0.          0.          0.          0.          0.          0.        ]]
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />example/custom_model/1_lstm.py
<br />example.custom_model.1_lstm.py
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/example/custom_model/1_lstm.py'>
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/example/custom_model/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />
<br />  #### Model init  ############################################# 
<br />
<br />  #### Model fit   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />
<br />  #### Predict   ##################################################### 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />  #### metrics   ##################################################### 
<br />{'loss': 0.42343443259596825, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />2020-05-19 23:14:44.708412: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />{'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
<br />
<br />Key Variable not found in checkpoint
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 523, in main
<br />    test_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 455, in test_cli
<br />    test(arg.model_uri)  # '1_lstm'
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 189, in test
<br />    module.test()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/example/custom_model/1_lstm.py", line 315, in test
<br />    session = load(out_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/example/custom_model/1_lstm.py", line 199, in load
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
<br />example/custom_model/1_lstm.py
<br />example.custom_model.1_lstm.py
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/example/custom_model/1_lstm.py'>
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/example/custom_model/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />
<br />  #### Model init  ############################################# 
<br />
<br />  #### Model fit   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />
<br />  #### Predict   ##################################################### 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />  #### metrics   ##################################################### 
<br />{'loss': 0.5514167547225952, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />2020-05-19 23:14:45.971240: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />{'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
<br />
<br />Key Variable not found in checkpoint
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 523, in main
<br />    test_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 457, in test_cli
<br />    test_global(arg.model_uri)  # '1_lstm'
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 200, in test_global
<br />    module.test()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/example/custom_model/1_lstm.py", line 315, in test
<br />    session = load(out_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/example/custom_model/1_lstm.py", line 199, in load
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
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ml_optim --do search  --config_file template/optim_config.json  --config_mode "test"  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
<br />Deprecaton set to False
<br />
<br />  ############# OPTIMIZATION Start  ############### 



### Error 5, [Traceback at line 949](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L949)<br />949..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/optim.py", line 388, in main
<br />    optim_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/optim.py", line 259, in optim_cli
<br />    out_pars        = out_pars )
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/optim.py", line 54, in optim
<br />    if hypermodel_pars["engine_pars"]['engine'] == "optuna":
<br />KeyError: 'engine_pars'



### Error 6, [Traceback at line 2125](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2125)<br />2125..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 7, [Traceback at line 2160](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2160)<br />2160..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 8, [Traceback at line 2200](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2200)<br />2200..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 9, [Traceback at line 2235](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2235)<br />2235..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 10, [Traceback at line 2280](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2280)<br />2280..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 11, [Traceback at line 2315](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2315)<br />2315..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 12, [Traceback at line 2372](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2372)<br />2372..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 13, [Traceback at line 2376](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2376)<br />2376..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
<br />    mpars['encoder'] = MLPEncoder()   #bug in seq2seq
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 424, in init_wrapper
<br />    model = PydanticModel(**{**nmargs, **kwargs})
<br />  File "pydantic/main.py", line 283, in pydantic.main.BaseModel.__init__
<br />pydantic.error_wrappers.ValidationError: 1 validation error for MLPEncoderModel
<br />layer_sizes
<br />  field required (type=value_error.missing)
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ml_benchmark  --do  dataset/json/benchmark.json  --path_json  dataset/json/benchmark_timeseries/test01/  2>&1 | tee -a  cd log_cli_$(date +%Y-%m-%d_%H).txt
<br />
<br />  dataset/json/benchmark.json 
<br />
<br />  Custom benchmark 
<br />
<br />  ['mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'r2_score'] 
<br />
<br />  json_path https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/json/benchmark_timeseries/test01/ 
<br />
<br />  Model List [{'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}}, {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}}, {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}, {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}] 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60} {'outpath': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_keras/armdn/'} 
<br />
<br />  #### Setup Model   ############################################## 
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
<br />LSTM_1 (LSTM)                (None, 60, 300)           362400    
<br />_________________________________________________________________
<br />LSTM_2 (LSTM)                (None, 60, 200)           400800    
<br />_________________________________________________________________
<br />LSTM_3 (LSTM)                (None, 60, 24)            21600     
<br />_________________________________________________________________
<br />LSTM_4 (LSTM)                (None, 12)                1776      
<br />_________________________________________________________________
<br />dense_1 (Dense)              (None, 10)                130       
<br />_________________________________________________________________
<br />mdn_1 (MDN)                  (None, 363)               3993      
<br />=================================================================
<br />Total params: 790,699
<br />Trainable params: 790,699
<br />Non-trainable params: 0
<br />_________________________________________________________________
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fb862163f98> <class 'mlmodels.model_keras.armdn.Model'>
<br />
<br />  #### Loading dataset   ############################################# 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
<br />
<br />Epoch 1/10
<br />
<br />1/1 [==============================] - 2s 2s/step - loss: 354235.8750
<br />Epoch 2/10
<br />
<br />1/1 [==============================] - 0s 103ms/step - loss: 257210.4062
<br />Epoch 3/10
<br />
<br />1/1 [==============================] - 0s 107ms/step - loss: 155994.7500
<br />Epoch 4/10
<br />
<br />1/1 [==============================] - 0s 111ms/step - loss: 86977.2500
<br />Epoch 5/10
<br />
<br />1/1 [==============================] - 0s 115ms/step - loss: 46139.7930
<br />Epoch 6/10
<br />
<br />1/1 [==============================] - 0s 115ms/step - loss: 25583.1426
<br />Epoch 7/10
<br />
<br />1/1 [==============================] - 0s 109ms/step - loss: 15446.8965
<br />Epoch 8/10
<br />
<br />1/1 [==============================] - 0s 95ms/step - loss: 10225.6895
<br />Epoch 9/10
<br />
<br />1/1 [==============================] - 0s 101ms/step - loss: 7345.3770
<br />Epoch 10/10
<br />
<br />1/1 [==============================] - 0s 98ms/step - loss: 5630.7637
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />[[-7.28248358e-02 -1.47919476e+00  7.06209600e-01  2.37017822e+00
<br />   4.83762383e-01 -2.27384281e+00 -7.37566650e-01  4.73039448e-01
<br />   1.74618995e+00 -1.29388499e+00 -6.74126685e-01  1.89250541e+00
<br />   1.12782860e+00  1.00749433e-01 -5.48303127e-01  1.49069929e+00
<br />  -6.49441361e-01  1.67148018e+00 -5.46515584e-01  5.51356733e-01
<br />  -1.46843195e-01  1.53443217e-01  6.52796626e-01 -2.22031355e+00
<br />  -5.59123382e-02  1.31797338e+00 -1.08513439e+00 -5.03677845e-01
<br />  -4.67693582e-02 -2.06869155e-01 -3.73073697e-01  2.99305201e-01
<br />  -1.65562475e+00 -1.17411089e+00 -7.72568345e-01  3.69171500e-01
<br />   1.62815666e+00  4.40370411e-01  9.78513002e-01 -1.49918938e+00
<br />  -6.05991960e-01  1.36508334e+00  1.74698770e+00  1.21714175e-01
<br />  -2.81137526e-01  1.09236807e-01  1.10453153e+00 -1.12650549e+00
<br />  -6.28024340e-03 -2.07265645e-01 -6.04180217e-01 -1.25421786e+00
<br />   1.46552399e-02  9.03559744e-01 -8.43556523e-01  4.61064637e-01
<br />   2.28871393e+00 -9.49426293e-01 -9.68208790e-01 -4.49463546e-01
<br />  -4.55023378e-01 -6.83264911e-01  2.80705619e+00  8.76700759e-01
<br />  -1.92169976e+00  1.96423793e+00  1.83281159e+00 -5.92119455e-01
<br />  -1.01919465e-01  3.77492309e-02  6.35229230e-01 -1.27823353e-02
<br />  -2.77596545e+00  1.45810634e-01 -6.74220502e-01  6.47445738e-01
<br />   1.96262896e+00  8.93023074e-01  1.66661233e-01  5.24477512e-02
<br />  -6.20877147e-01  1.22429705e+00  1.56966954e-01  4.06679511e-01
<br />   1.71581537e-01 -1.62060726e+00 -9.29041922e-01 -7.42017150e-01
<br />  -9.05306116e-02 -1.64670336e+00 -1.84566304e-01  5.83983123e-01
<br />   6.69212937e-02 -1.34826863e+00  5.20378470e-01  2.40833473e+00
<br />   9.78827834e-01  9.96527851e-01 -1.11266398e+00 -1.49926460e+00
<br />   1.50933969e+00 -8.54100347e-01  1.42889333e+00  1.24618506e+00
<br />   1.39644885e+00 -9.93853927e-01 -8.75976622e-01  1.80124724e+00
<br />   3.40892881e-01  1.00474930e+00 -3.27370435e-01 -1.31838214e+00
<br />   7.20441341e-04 -9.89506185e-01 -6.33540690e-01 -2.41111815e-01
<br />  -3.66328061e-01 -1.78954077e+00 -3.50467682e-01 -1.62778878e+00
<br />  -5.58581427e-02  1.08039570e+01  1.03479576e+01  8.59070396e+00
<br />   9.02211189e+00  8.72607994e+00  8.95433235e+00  7.81761932e+00
<br />   1.07426071e+01  8.07462788e+00  1.01751223e+01  8.12344360e+00
<br />   8.78417683e+00  1.01401739e+01  9.69964886e+00  1.04100103e+01
<br />   9.51688194e+00  8.39951420e+00  1.06034927e+01  8.99866772e+00
<br />   1.00755520e+01  9.11888123e+00  1.00470304e+01  9.66688824e+00
<br />   9.44434357e+00  9.23408890e+00  9.56870079e+00  1.01953411e+01
<br />   7.87315321e+00  8.52789402e+00  8.02845383e+00  9.31026363e+00
<br />   7.91149044e+00  9.60807991e+00  8.13539791e+00  1.03758593e+01
<br />   9.77257156e+00  8.49588203e+00  8.29405117e+00  9.27227497e+00
<br />   1.07388830e+01  1.09232159e+01  9.56922626e+00  6.98019218e+00
<br />   7.46101665e+00  8.44102859e+00  9.93939495e+00  8.64669323e+00
<br />   7.34976196e+00  9.35154247e+00  9.70558453e+00  8.20082378e+00
<br />   9.00703335e+00  8.53615189e+00  8.67753887e+00  8.28223896e+00
<br />   9.19655228e+00  8.51949120e+00  8.81423855e+00  9.53717422e+00
<br />   1.34717321e+00  5.66424310e-01  8.38868380e-01  1.03028131e+00
<br />   1.48446441e-01  1.56127453e+00  1.22179615e+00  1.29559708e+00
<br />   7.85642624e-01  5.79527617e-01  1.87998152e+00  1.48552489e+00
<br />   1.34398985e+00  1.23846507e+00  7.69099772e-01  4.44179893e-01
<br />   2.25944662e+00  6.13382101e-01  4.37162817e-01  2.14950657e+00
<br />   5.78604937e-01  3.20373178e-01  6.08726263e-01  2.26099849e+00
<br />   1.83395147e-01  1.20985889e+00  9.94343102e-01  1.51615405e+00
<br />   1.60134554e-01  1.24460459e+00  2.56038618e+00  6.49903655e-01
<br />   5.01878321e-01  9.14291620e-01  8.42469752e-01  1.69927859e+00
<br />   4.90938663e-01  1.83635855e+00  6.25247955e-01  1.09814501e+00
<br />   2.40407205e+00  1.25160384e+00  1.37343645e+00  2.32126236e+00
<br />   2.37470770e+00  1.54133999e+00  5.02645433e-01  1.48387671e+00
<br />   1.51900411e+00  1.10942268e+00  9.71776843e-01  4.62319851e-01
<br />   3.15367174e+00  6.98162794e-01  4.70024705e-01  1.70474768e+00
<br />   2.97440886e-01  3.35432589e-01  2.03100300e+00  2.22085536e-01
<br />   1.49181926e+00  4.03639495e-01  2.21437633e-01  5.55086374e-01
<br />   8.03508759e-02  7.87890613e-01  3.41544271e-01  5.43661118e-01
<br />   1.05538070e+00  1.60075331e+00  8.43499601e-01  2.49686766e+00
<br />   1.76241422e+00  2.62997270e-01  1.64579868e-01  1.54364741e+00
<br />   1.86402953e+00  1.59128523e+00  2.18485212e+00  1.58284783e-01
<br />   4.77654874e-01  7.39038467e-01  2.60084963e+00  9.20916200e-01
<br />   1.98499501e-01  9.53031361e-01  4.02840137e-01  1.86046541e+00
<br />   3.11795115e-01  1.87154543e+00  2.42095768e-01  4.08996403e-01
<br />   7.53423810e-01  4.68400180e-01  1.20728457e+00  8.56783390e-02
<br />   2.63007069e+00  1.88874221e+00  2.15818000e+00  3.04904604e+00
<br />   9.04360116e-01  1.82523370e+00  7.52317071e-01  9.26470578e-01
<br />   2.78640556e+00  2.97400069e+00  1.01469994e+00  1.77608252e-01
<br />   1.28408217e+00  1.82655716e+00  3.34737360e-01  6.09509885e-01
<br />   7.40238965e-01  4.40853596e-01  2.36556470e-01  1.82296574e-01
<br />   1.14950740e+00  4.48221862e-01  1.93969333e+00  3.62868786e-01
<br />   1.57920718e-01  1.12373428e+01  9.54489994e+00  9.82826042e+00
<br />   7.05260706e+00  8.92423916e+00  9.78384781e+00  9.58190632e+00
<br />   8.19320202e+00  1.05003090e+01  7.80602217e+00  6.96029472e+00
<br />   7.91663074e+00  1.00615473e+01  9.61744785e+00  9.35241318e+00
<br />   9.62052822e+00  7.29240942e+00  8.00326729e+00  8.51951981e+00
<br />   1.15559664e+01  9.81897640e+00  9.82057667e+00  9.90336990e+00
<br />   1.05050879e+01  9.49212933e+00  8.35439110e+00  9.57020473e+00
<br />   8.76342010e+00  7.65256786e+00  9.06295300e+00  8.17462921e+00
<br />   9.84477520e+00  1.02784452e+01  1.18096914e+01  7.83621120e+00
<br />   9.57862377e+00  8.07893467e+00  9.04792595e+00  1.09220343e+01
<br />   8.86679745e+00  8.22077465e+00  9.08689976e+00  8.46823502e+00
<br />   7.79271984e+00  8.77989483e+00  9.02595043e+00  1.00298023e+01
<br />   8.34031582e+00  8.36321068e+00  7.93623066e+00  1.01233425e+01
<br />   9.38160133e+00  1.07043629e+01  9.05651093e+00  8.47164726e+00
<br />   9.19455719e+00  1.09042988e+01  9.07453632e+00  1.06780624e+01
<br />  -9.63611412e+00 -7.80392408e+00  5.28507710e+00]]
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-19 23:52:09.481992
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   93.6119
<br />metric_name                                  mean_absolute_error
<br />Name: 0, dtype: object 
<br />
<br />  date_run                              2020-05-19 23:52:09.487456
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   8786.64
<br />metric_name                                   mean_squared_error
<br />Name: 1, dtype: object 
<br />
<br />  date_run                              2020-05-19 23:52:09.492036
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   93.4956
<br />metric_name                                median_absolute_error
<br />Name: 2, dtype: object 
<br />
<br />  date_run                              2020-05-19 23:52:09.496052
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -785.906
<br />metric_name                                             r2_score
<br />Name: 3, dtype: object 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />| N-Beats
<br />| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140430003246752
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140427101945024
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140427101945528
<br />| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140427101552880
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140427101553384
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140427101553888
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fb862151fd0> <class 'mlmodels.model_tch.nbeats.Model'>
<br />[[0.40504701]
<br /> [0.40695405]
<br /> [0.39710839]
<br /> ...
<br /> [0.93587014]
<br /> [0.95086039]
<br /> [0.95547277]]
<br />--- fiting ---
<br />grad_step = 000000, loss = 0.455813
<br />plot()
<br />Saved image to .//n_beats_0.png.
<br />grad_step = 000001, loss = 0.431190
<br />grad_step = 000002, loss = 0.415034
<br />grad_step = 000003, loss = 0.399172
<br />grad_step = 000004, loss = 0.385441
<br />grad_step = 000005, loss = 0.375312
<br />grad_step = 000006, loss = 0.368943
<br />grad_step = 000007, loss = 0.362439
<br />grad_step = 000008, loss = 0.352525
<br />grad_step = 000009, loss = 0.342765
<br />grad_step = 000010, loss = 0.334099
<br />grad_step = 000011, loss = 0.326376
<br />grad_step = 000012, loss = 0.318765
<br />grad_step = 000013, loss = 0.310931
<br />grad_step = 000014, loss = 0.302815
<br />grad_step = 000015, loss = 0.294613
<br />grad_step = 000016, loss = 0.286492
<br />grad_step = 000017, loss = 0.278328
<br />grad_step = 000018, loss = 0.270738
<br />grad_step = 000019, loss = 0.263659
<br />grad_step = 000020, loss = 0.256496
<br />grad_step = 000021, loss = 0.249141
<br />grad_step = 000022, loss = 0.241606
<br />grad_step = 000023, loss = 0.234050
<br />grad_step = 000024, loss = 0.226827
<br />grad_step = 000025, loss = 0.220060
<br />grad_step = 000026, loss = 0.213402
<br />grad_step = 000027, loss = 0.206517
<br />grad_step = 000028, loss = 0.199450
<br />grad_step = 000029, loss = 0.192670
<br />grad_step = 000030, loss = 0.186366
<br />grad_step = 000031, loss = 0.180181
<br />grad_step = 000032, loss = 0.173900
<br />grad_step = 000033, loss = 0.167649
<br />grad_step = 000034, loss = 0.161440
<br />grad_step = 000035, loss = 0.155627
<br />grad_step = 000036, loss = 0.149670
<br />grad_step = 000037, loss = 0.143728
<br />grad_step = 000038, loss = 0.137982
<br />grad_step = 000039, loss = 0.132459
<br />grad_step = 000040, loss = 0.127006
<br />grad_step = 000041, loss = 0.121706
<br />grad_step = 000042, loss = 0.116458
<br />grad_step = 000043, loss = 0.111276
<br />grad_step = 000044, loss = 0.106307
<br />grad_step = 000045, loss = 0.101547
<br />grad_step = 000046, loss = 0.096897
<br />grad_step = 000047, loss = 0.092226
<br />grad_step = 000048, loss = 0.087693
<br />grad_step = 000049, loss = 0.083411
<br />grad_step = 000050, loss = 0.079322
<br />grad_step = 000051, loss = 0.075292
<br />grad_step = 000052, loss = 0.071340
<br />grad_step = 000053, loss = 0.067557
<br />grad_step = 000054, loss = 0.063973
<br />grad_step = 000055, loss = 0.060522
<br />grad_step = 000056, loss = 0.057137
<br />grad_step = 000057, loss = 0.053903
<br />grad_step = 000058, loss = 0.050824
<br />grad_step = 000059, loss = 0.047857
<br />grad_step = 000060, loss = 0.045024
<br />grad_step = 000061, loss = 0.042305
<br />grad_step = 000062, loss = 0.039704
<br />grad_step = 000063, loss = 0.037234
<br />grad_step = 000064, loss = 0.034948
<br />grad_step = 000065, loss = 0.032820
<br />grad_step = 000066, loss = 0.030799
<br />grad_step = 000067, loss = 0.028580
<br />grad_step = 000068, loss = 0.026405
<br />grad_step = 000069, loss = 0.024628
<br />grad_step = 000070, loss = 0.023058
<br />grad_step = 000071, loss = 0.021408
<br />grad_step = 000072, loss = 0.019694
<br />grad_step = 000073, loss = 0.018161
<br />grad_step = 000074, loss = 0.016898
<br />grad_step = 000075, loss = 0.015661
<br />grad_step = 000076, loss = 0.014331
<br />grad_step = 000077, loss = 0.013131
<br />grad_step = 000078, loss = 0.012165
<br />grad_step = 000079, loss = 0.011262
<br />grad_step = 000080, loss = 0.010288
<br />grad_step = 000081, loss = 0.009368
<br />grad_step = 000082, loss = 0.008677
<br />grad_step = 000083, loss = 0.008058
<br />grad_step = 000084, loss = 0.007357
<br />grad_step = 000085, loss = 0.006698
<br />grad_step = 000086, loss = 0.006185
<br />grad_step = 000087, loss = 0.005750
<br />grad_step = 000088, loss = 0.005317
<br />grad_step = 000089, loss = 0.004901
<br />grad_step = 000090, loss = 0.004551
<br />grad_step = 000091, loss = 0.004235
<br />grad_step = 000092, loss = 0.003923
<br />grad_step = 000093, loss = 0.003640
<br />grad_step = 000094, loss = 0.003424
<br />grad_step = 000095, loss = 0.003269
<br />grad_step = 000096, loss = 0.003122
<br />grad_step = 000097, loss = 0.002976
<br />grad_step = 000098, loss = 0.002860
<br />grad_step = 000099, loss = 0.002810
<br />grad_step = 000100, loss = 0.002835
<br />plot()
<br />Saved image to .//n_beats_100.png.
<br />grad_step = 000101, loss = 0.002856
<br />grad_step = 000102, loss = 0.002780
<br />grad_step = 000103, loss = 0.002575
<br />grad_step = 000104, loss = 0.002409
<br />grad_step = 000105, loss = 0.002331
<br />grad_step = 000106, loss = 0.002322
<br />grad_step = 000107, loss = 0.002383
<br />grad_step = 000108, loss = 0.002403
<br />grad_step = 000109, loss = 0.002277
<br />grad_step = 000110, loss = 0.002149
<br />grad_step = 000111, loss = 0.002172
<br />grad_step = 000112, loss = 0.002244
<br />grad_step = 000113, loss = 0.002226
<br />grad_step = 000114, loss = 0.002166
<br />grad_step = 000115, loss = 0.002145
<br />grad_step = 000116, loss = 0.002141
<br />grad_step = 000117, loss = 0.002098
<br />grad_step = 000118, loss = 0.002075
<br />grad_step = 000119, loss = 0.002099
<br />grad_step = 000120, loss = 0.002107
<br />grad_step = 000121, loss = 0.002066
<br />grad_step = 000122, loss = 0.002023
<br />grad_step = 000123, loss = 0.002036
<br />grad_step = 000124, loss = 0.002032
<br />grad_step = 000125, loss = 0.001995
<br />grad_step = 000126, loss = 0.002000
<br />grad_step = 000127, loss = 0.002018
<br />grad_step = 000128, loss = 0.001996
<br />grad_step = 000129, loss = 0.001972
<br />grad_step = 000130, loss = 0.001980
<br />grad_step = 000131, loss = 0.001987
<br />grad_step = 000132, loss = 0.001977
<br />grad_step = 000133, loss = 0.001967
<br />grad_step = 000134, loss = 0.001991
<br />grad_step = 000135, loss = 0.002017
<br />grad_step = 000136, loss = 0.002052
<br />grad_step = 000137, loss = 0.002100
<br />grad_step = 000138, loss = 0.002191
<br />grad_step = 000139, loss = 0.002150
<br />grad_step = 000140, loss = 0.002088
<br />grad_step = 000141, loss = 0.001976
<br />grad_step = 000142, loss = 0.001889
<br />grad_step = 000143, loss = 0.001895
<br />grad_step = 000144, loss = 0.001950
<br />grad_step = 000145, loss = 0.001986
<br />grad_step = 000146, loss = 0.002005
<br />grad_step = 000147, loss = 0.002019
<br />grad_step = 000148, loss = 0.001931
<br />grad_step = 000149, loss = 0.001867
<br />grad_step = 000150, loss = 0.001863
<br />grad_step = 000151, loss = 0.001869
<br />grad_step = 000152, loss = 0.001888
<br />grad_step = 000153, loss = 0.001917
<br />grad_step = 000154, loss = 0.001923
<br />grad_step = 000155, loss = 0.001902
<br />grad_step = 000156, loss = 0.001889
<br />grad_step = 000157, loss = 0.001854
<br />grad_step = 000158, loss = 0.001824
<br />grad_step = 000159, loss = 0.001816
<br />grad_step = 000160, loss = 0.001813
<br />grad_step = 000161, loss = 0.001808
<br />grad_step = 000162, loss = 0.001817
<br />grad_step = 000163, loss = 0.001838
<br />grad_step = 000164, loss = 0.001858
<br />grad_step = 000165, loss = 0.001904
<br />grad_step = 000166, loss = 0.001973
<br />grad_step = 000167, loss = 0.002111
<br />grad_step = 000168, loss = 0.002154
<br />grad_step = 000169, loss = 0.002191
<br />grad_step = 000170, loss = 0.002026
<br />grad_step = 000171, loss = 0.001831
<br />grad_step = 000172, loss = 0.001759
<br />grad_step = 000173, loss = 0.001827
<br />grad_step = 000174, loss = 0.001940
<br />grad_step = 000175, loss = 0.001965
<br />grad_step = 000176, loss = 0.001901
<br />grad_step = 000177, loss = 0.001767
<br />grad_step = 000178, loss = 0.001733
<br />grad_step = 000179, loss = 0.001794
<br />grad_step = 000180, loss = 0.001859
<br />grad_step = 000181, loss = 0.001865
<br />grad_step = 000182, loss = 0.001769
<br />grad_step = 000183, loss = 0.001699
<br />grad_step = 000184, loss = 0.001704
<br />grad_step = 000185, loss = 0.001746
<br />grad_step = 000186, loss = 0.001780
<br />grad_step = 000187, loss = 0.001764
<br />grad_step = 000188, loss = 0.001743
<br />grad_step = 000189, loss = 0.001709
<br />grad_step = 000190, loss = 0.001681
<br />grad_step = 000191, loss = 0.001650
<br />grad_step = 000192, loss = 0.001636
<br />grad_step = 000193, loss = 0.001642
<br />grad_step = 000194, loss = 0.001667
<br />grad_step = 000195, loss = 0.001726
<br />grad_step = 000196, loss = 0.001762
<br />grad_step = 000197, loss = 0.001832
<br />grad_step = 000198, loss = 0.001833
<br />grad_step = 000199, loss = 0.001829
<br />grad_step = 000200, loss = 0.001724
<br />plot()
<br />Saved image to .//n_beats_200.png.
<br />grad_step = 000201, loss = 0.001618
<br />grad_step = 000202, loss = 0.001578
<br />grad_step = 000203, loss = 0.001611
<br />grad_step = 000204, loss = 0.001674
<br />grad_step = 000205, loss = 0.001732
<br />grad_step = 000206, loss = 0.001796
<br />grad_step = 000207, loss = 0.001758
<br />grad_step = 000208, loss = 0.001698
<br />grad_step = 000209, loss = 0.001593
<br />grad_step = 000210, loss = 0.001573
<br />grad_step = 000211, loss = 0.001636
<br />grad_step = 000212, loss = 0.001664
<br />grad_step = 000213, loss = 0.001612
<br />grad_step = 000214, loss = 0.001547
<br />grad_step = 000215, loss = 0.001541
<br />grad_step = 000216, loss = 0.001572
<br />grad_step = 000217, loss = 0.001620
<br />grad_step = 000218, loss = 0.001668
<br />grad_step = 000219, loss = 0.001624
<br />grad_step = 000220, loss = 0.001616
<br />grad_step = 000221, loss = 0.001598
<br />grad_step = 000222, loss = 0.001552
<br />grad_step = 000223, loss = 0.001522
<br />grad_step = 000224, loss = 0.001518
<br />grad_step = 000225, loss = 0.001531
<br />grad_step = 000226, loss = 0.001556
<br />grad_step = 000227, loss = 0.001566
<br />grad_step = 000228, loss = 0.001542
<br />grad_step = 000229, loss = 0.001530
<br />grad_step = 000230, loss = 0.001521
<br />grad_step = 000231, loss = 0.001504
<br />grad_step = 000232, loss = 0.001501
<br />grad_step = 000233, loss = 0.001510
<br />grad_step = 000234, loss = 0.001518
<br />grad_step = 000235, loss = 0.001532
<br />grad_step = 000236, loss = 0.001551
<br />grad_step = 000237, loss = 0.001550
<br />grad_step = 000238, loss = 0.001552
<br />grad_step = 000239, loss = 0.001552
<br />grad_step = 000240, loss = 0.001541
<br />grad_step = 000241, loss = 0.001512
<br />grad_step = 000242, loss = 0.001488
<br />grad_step = 000243, loss = 0.001480
<br />grad_step = 000244, loss = 0.001487
<br />grad_step = 000245, loss = 0.001500
<br />grad_step = 000246, loss = 0.001504
<br />grad_step = 000247, loss = 0.001503
<br />grad_step = 000248, loss = 0.001490
<br />grad_step = 000249, loss = 0.001480
<br />grad_step = 000250, loss = 0.001473
<br />grad_step = 000251, loss = 0.001471
<br />grad_step = 000252, loss = 0.001471
<br />grad_step = 000253, loss = 0.001473
<br />grad_step = 000254, loss = 0.001481
<br />grad_step = 000255, loss = 0.001512
<br />grad_step = 000256, loss = 0.001587
<br />grad_step = 000257, loss = 0.001784
<br />grad_step = 000258, loss = 0.001932
<br />grad_step = 000259, loss = 0.002109
<br />grad_step = 000260, loss = 0.001844
<br />grad_step = 000261, loss = 0.001544
<br />grad_step = 000262, loss = 0.001510
<br />grad_step = 000263, loss = 0.001673
<br />grad_step = 000264, loss = 0.001689
<br />grad_step = 000265, loss = 0.001528
<br />grad_step = 000266, loss = 0.001500
<br />grad_step = 000267, loss = 0.001618
<br />grad_step = 000268, loss = 0.001579
<br />grad_step = 000269, loss = 0.001465
<br />grad_step = 000270, loss = 0.001520
<br />grad_step = 000271, loss = 0.001559
<br />grad_step = 000272, loss = 0.001492
<br />grad_step = 000273, loss = 0.001476
<br />grad_step = 000274, loss = 0.001483
<br />grad_step = 000275, loss = 0.001459
<br />grad_step = 000276, loss = 0.001514
<br />grad_step = 000277, loss = 0.001572
<br />grad_step = 000278, loss = 0.001492
<br />grad_step = 000279, loss = 0.001498
<br />grad_step = 000280, loss = 0.001463
<br />grad_step = 000281, loss = 0.001425
<br />grad_step = 000282, loss = 0.001460
<br />grad_step = 000283, loss = 0.001460
<br />grad_step = 000284, loss = 0.001455
<br />grad_step = 000285, loss = 0.001487
<br />grad_step = 000286, loss = 0.001499
<br />grad_step = 000287, loss = 0.001468
<br />grad_step = 000288, loss = 0.001479
<br />grad_step = 000289, loss = 0.001445
<br />grad_step = 000290, loss = 0.001427
<br />grad_step = 000291, loss = 0.001433
<br />grad_step = 000292, loss = 0.001412
<br />grad_step = 000293, loss = 0.001406
<br />grad_step = 000294, loss = 0.001416
<br />grad_step = 000295, loss = 0.001405
<br />grad_step = 000296, loss = 0.001407
<br />grad_step = 000297, loss = 0.001421
<br />grad_step = 000298, loss = 0.001419
<br />grad_step = 000299, loss = 0.001442
<br />grad_step = 000300, loss = 0.001471
<br />plot()
<br />Saved image to .//n_beats_300.png.
<br />grad_step = 000301, loss = 0.001537
<br />grad_step = 000302, loss = 0.001585
<br />grad_step = 000303, loss = 0.001699
<br />grad_step = 000304, loss = 0.001648
<br />grad_step = 000305, loss = 0.001592
<br />grad_step = 000306, loss = 0.001425
<br />grad_step = 000307, loss = 0.001406
<br />grad_step = 000308, loss = 0.001498
<br />grad_step = 000309, loss = 0.001506
<br />grad_step = 000310, loss = 0.001452
<br />grad_step = 000311, loss = 0.001394
<br />grad_step = 000312, loss = 0.001415
<br />grad_step = 000313, loss = 0.001465
<br />grad_step = 000314, loss = 0.001449
<br />grad_step = 000315, loss = 0.001416
<br />grad_step = 000316, loss = 0.001388
<br />grad_step = 000317, loss = 0.001383
<br />grad_step = 000318, loss = 0.001411
<br />grad_step = 000319, loss = 0.001422
<br />grad_step = 000320, loss = 0.001413
<br />grad_step = 000321, loss = 0.001390
<br />grad_step = 000322, loss = 0.001372
<br />grad_step = 000323, loss = 0.001364
<br />grad_step = 000324, loss = 0.001373
<br />grad_step = 000325, loss = 0.001385
<br />grad_step = 000326, loss = 0.001389
<br />grad_step = 000327, loss = 0.001398
<br />grad_step = 000328, loss = 0.001390
<br />grad_step = 000329, loss = 0.001383
<br />grad_step = 000330, loss = 0.001374
<br />grad_step = 000331, loss = 0.001362
<br />grad_step = 000332, loss = 0.001351
<br />grad_step = 000333, loss = 0.001346
<br />grad_step = 000334, loss = 0.001343
<br />grad_step = 000335, loss = 0.001343
<br />grad_step = 000336, loss = 0.001349
<br />grad_step = 000337, loss = 0.001355
<br />grad_step = 000338, loss = 0.001365
<br />grad_step = 000339, loss = 0.001379
<br />grad_step = 000340, loss = 0.001410
<br />grad_step = 000341, loss = 0.001446
<br />grad_step = 000342, loss = 0.001534
<br />grad_step = 000343, loss = 0.001573
<br />grad_step = 000344, loss = 0.001664
<br />grad_step = 000345, loss = 0.001568
<br />grad_step = 000346, loss = 0.001450
<br />grad_step = 000347, loss = 0.001340
<br />grad_step = 000348, loss = 0.001350
<br />grad_step = 000349, loss = 0.001436
<br />grad_step = 000350, loss = 0.001457
<br />grad_step = 000351, loss = 0.001412
<br />grad_step = 000352, loss = 0.001338
<br />grad_step = 000353, loss = 0.001332
<br />grad_step = 000354, loss = 0.001380
<br />grad_step = 000355, loss = 0.001401
<br />grad_step = 000356, loss = 0.001393
<br />grad_step = 000357, loss = 0.001338
<br />grad_step = 000358, loss = 0.001314
<br />grad_step = 000359, loss = 0.001330
<br />grad_step = 000360, loss = 0.001355
<br />grad_step = 000361, loss = 0.001368
<br />grad_step = 000362, loss = 0.001344
<br />grad_step = 000363, loss = 0.001317
<br />grad_step = 000364, loss = 0.001303
<br />grad_step = 000365, loss = 0.001308
<br />grad_step = 000366, loss = 0.001321
<br />grad_step = 000367, loss = 0.001328
<br />grad_step = 000368, loss = 0.001330
<br />grad_step = 000369, loss = 0.001320
<br />grad_step = 000370, loss = 0.001313
<br />grad_step = 000371, loss = 0.001310
<br />grad_step = 000372, loss = 0.001315
<br />grad_step = 000373, loss = 0.001322
<br />grad_step = 000374, loss = 0.001327
<br />grad_step = 000375, loss = 0.001327
<br />grad_step = 000376, loss = 0.001323
<br />grad_step = 000377, loss = 0.001319
<br />grad_step = 000378, loss = 0.001324
<br />grad_step = 000379, loss = 0.001337
<br />grad_step = 000380, loss = 0.001381
<br />grad_step = 000381, loss = 0.001430
<br />grad_step = 000382, loss = 0.001526
<br />grad_step = 000383, loss = 0.001546
<br />grad_step = 000384, loss = 0.001562
<br />grad_step = 000385, loss = 0.001448
<br />grad_step = 000386, loss = 0.001347
<br />grad_step = 000387, loss = 0.001301
<br />grad_step = 000388, loss = 0.001331
<br />grad_step = 000389, loss = 0.001383
<br />grad_step = 000390, loss = 0.001359
<br />grad_step = 000391, loss = 0.001312
<br />grad_step = 000392, loss = 0.001280
<br />grad_step = 000393, loss = 0.001297
<br />grad_step = 000394, loss = 0.001337
<br />grad_step = 000395, loss = 0.001330
<br />grad_step = 000396, loss = 0.001298
<br />grad_step = 000397, loss = 0.001261
<br />grad_step = 000398, loss = 0.001260
<br />grad_step = 000399, loss = 0.001284
<br />grad_step = 000400, loss = 0.001297
<br />plot()
<br />Saved image to .//n_beats_400.png.
<br />grad_step = 000401, loss = 0.001288
<br />grad_step = 000402, loss = 0.001261
<br />grad_step = 000403, loss = 0.001248
<br />grad_step = 000404, loss = 0.001256
<br />grad_step = 000405, loss = 0.001269
<br />grad_step = 000406, loss = 0.001270
<br />grad_step = 000407, loss = 0.001256
<br />grad_step = 000408, loss = 0.001243
<br />grad_step = 000409, loss = 0.001240
<br />grad_step = 000410, loss = 0.001247
<br />grad_step = 000411, loss = 0.001252
<br />grad_step = 000412, loss = 0.001250
<br />grad_step = 000413, loss = 0.001241
<br />grad_step = 000414, loss = 0.001233
<br />grad_step = 000415, loss = 0.001230
<br />grad_step = 000416, loss = 0.001232
<br />grad_step = 000417, loss = 0.001236
<br />grad_step = 000418, loss = 0.001238
<br />grad_step = 000419, loss = 0.001238
<br />grad_step = 000420, loss = 0.001237
<br />grad_step = 000421, loss = 0.001240
<br />grad_step = 000422, loss = 0.001253
<br />grad_step = 000423, loss = 0.001291
<br />grad_step = 000424, loss = 0.001365
<br />grad_step = 000425, loss = 0.001529
<br />grad_step = 000426, loss = 0.001706
<br />grad_step = 000427, loss = 0.001988
<br />grad_step = 000428, loss = 0.001928
<br />grad_step = 000429, loss = 0.001771
<br />grad_step = 000430, loss = 0.001484
<br />grad_step = 000431, loss = 0.001410
<br />grad_step = 000432, loss = 0.001499
<br />grad_step = 000433, loss = 0.001401
<br />grad_step = 000434, loss = 0.001340
<br />grad_step = 000435, loss = 0.001460
<br />grad_step = 000436, loss = 0.001453
<br />grad_step = 000437, loss = 0.001306
<br />grad_step = 000438, loss = 0.001228
<br />grad_step = 000439, loss = 0.001343
<br />grad_step = 000440, loss = 0.001397
<br />grad_step = 000441, loss = 0.001257
<br />grad_step = 000442, loss = 0.001227
<br />grad_step = 000443, loss = 0.001312
<br />grad_step = 000444, loss = 0.001297
<br />grad_step = 000445, loss = 0.001228
<br />grad_step = 000446, loss = 0.001222
<br />grad_step = 000447, loss = 0.001261
<br />grad_step = 000448, loss = 0.001272
<br />grad_step = 000449, loss = 0.001230
<br />grad_step = 000450, loss = 0.001202
<br />grad_step = 000451, loss = 0.001216
<br />grad_step = 000452, loss = 0.001238
<br />grad_step = 000453, loss = 0.001224
<br />grad_step = 000454, loss = 0.001203
<br />grad_step = 000455, loss = 0.001214
<br />grad_step = 000456, loss = 0.001219
<br />grad_step = 000457, loss = 0.001198
<br />grad_step = 000458, loss = 0.001187
<br />grad_step = 000459, loss = 0.001193
<br />grad_step = 000460, loss = 0.001200
<br />grad_step = 000461, loss = 0.001200
<br />grad_step = 000462, loss = 0.001190
<br />grad_step = 000463, loss = 0.001184
<br />grad_step = 000464, loss = 0.001189
<br />grad_step = 000465, loss = 0.001188
<br />grad_step = 000466, loss = 0.001178
<br />grad_step = 000467, loss = 0.001173
<br />grad_step = 000468, loss = 0.001174
<br />grad_step = 000469, loss = 0.001175
<br />grad_step = 000470, loss = 0.001174
<br />grad_step = 000471, loss = 0.001172
<br />grad_step = 000472, loss = 0.001168
<br />grad_step = 000473, loss = 0.001170
<br />grad_step = 000474, loss = 0.001172
<br />grad_step = 000475, loss = 0.001170
<br />grad_step = 000476, loss = 0.001166
<br />grad_step = 000477, loss = 0.001165
<br />grad_step = 000478, loss = 0.001164
<br />grad_step = 000479, loss = 0.001164
<br />grad_step = 000480, loss = 0.001164
<br />grad_step = 000481, loss = 0.001162
<br />grad_step = 000482, loss = 0.001160
<br />grad_step = 000483, loss = 0.001161
<br />grad_step = 000484, loss = 0.001161
<br />grad_step = 000485, loss = 0.001162
<br />grad_step = 000486, loss = 0.001162
<br />grad_step = 000487, loss = 0.001162
<br />grad_step = 000488, loss = 0.001163
<br />grad_step = 000489, loss = 0.001165
<br />grad_step = 000490, loss = 0.001170
<br />grad_step = 000491, loss = 0.001178
<br />grad_step = 000492, loss = 0.001185
<br />grad_step = 000493, loss = 0.001198
<br />grad_step = 000494, loss = 0.001211
<br />grad_step = 000495, loss = 0.001227
<br />grad_step = 000496, loss = 0.001233
<br />grad_step = 000497, loss = 0.001237
<br />grad_step = 000498, loss = 0.001225
<br />grad_step = 000499, loss = 0.001206
<br />grad_step = 000500, loss = 0.001176
<br />plot()
<br />Saved image to .//n_beats_500.png.
<br />grad_step = 000501, loss = 0.001150
<br />Finished.
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />[[0.40504701]
<br /> [0.40695405]
<br /> [0.39710839]
<br /> ...
<br /> [0.93587014]
<br /> [0.95086039]
<br /> [0.95547277]]
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-19 23:52:32.883057
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.218999
<br />metric_name                                  mean_absolute_error
<br />Name: 4, dtype: object 
<br />
<br />  date_run                              2020-05-19 23:52:32.889854
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.114878
<br />metric_name                                   mean_squared_error
<br />Name: 5, dtype: object 
<br />
<br />  date_run                              2020-05-19 23:52:32.898032
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.133731
<br />metric_name                                median_absolute_error
<br />Name: 6, dtype: object 
<br />
<br />  date_run                              2020-05-19 23:52:32.904347
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                 -0.745617
<br />metric_name                                             r2_score
<br />Name: 7, dtype: object 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train_data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'} {'outpath': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_fb/fb_prophet/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 
<br />INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
<br />Initial log joint probability = -192.039
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />      99       9186.38     0.0272386        1207.2           1           1      123   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     199       10269.2     0.0242289       2566.31        0.89        0.89      233   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     299       10621.2     0.0237499       3262.95           1           1      343   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     399       10886.5     0.0339822       1343.14           1           1      459   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     499       11288.1    0.00255943       1266.79           1           1      580   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     599       11498.7     0.0166167       2146.51           1           1      698   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     699       11555.9     0.0104637       2039.91           1           1      812   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     799       11575.2    0.00955805       570.757           1           1      922   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     899       11630.7     0.0178715       1643.41      0.3435      0.3435     1036   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />     999       11700.1      0.034504       2394.16           1           1     1146   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1099       11744.7   0.000237394       144.685           1           1     1258   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1199       11753.1    0.00188838       552.132      0.4814           1     1372   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1299         11758    0.00101299       262.652      0.7415      0.7415     1490   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1399         11761   0.000712302       157.258           1           1     1606   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1499       11781.3     0.0243264       931.457           1           1     1717   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1599       11791.1     0.0025484       550.483      0.7644      0.7644     1834   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1699       11797.7    0.00732868       810.153           1           1     1952   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1799       11802.5   0.000319611       98.1955     0.04871           1     2077   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1818       11803.2   5.97419e-05       246.505   3.588e-07       0.001     2142  LS failed, Hessian reset 
<br />    1855       11803.6   0.000110613       144.447   1.529e-06       0.001     2225  LS failed, Hessian reset 
<br />    1899       11804.3   0.000976631       305.295           1           1     2275   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    1999       11805.4   4.67236e-05       72.2243      0.9487      0.9487     2391   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2033       11806.1   1.47341e-05       111.754   8.766e-08       0.001     2480  LS failed, Hessian reset 
<br />    2099       11806.6   9.53816e-05       108.311      0.9684      0.9684     2563   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2151       11806.8   3.32394e-05       152.834   3.931e-07       0.001     2668  LS failed, Hessian reset 
<br />    2199         11807    0.00273479       216.444           1           1     2723   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2299       11810.9    0.00793685       550.165           1           1     2837   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2399       11818.9     0.0134452       377.542           1           1     2952   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2499       11824.9     0.0041384       130.511           1           1     3060   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2525       11826.5   2.36518e-05       102.803   6.403e-08       0.001     3158  LS failed, Hessian reset 
<br />    2599       11827.9   0.000370724       186.394      0.4637      0.4637     3242   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2606         11828   1.70497e-05       123.589     7.9e-08       0.001     3292  LS failed, Hessian reset 
<br />    2699       11829.1    0.00168243       332.201           1           1     3407   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2709       11829.2   1.92694e-05       146.345   1.034e-07       0.001     3461  LS failed, Hessian reset 
<br />    2746       11829.4   1.61976e-05       125.824   9.572e-08       0.001     3551  LS failed, Hessian reset 
<br />    2799       11829.5    0.00491161       122.515           1           1     3615   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2899       11830.6   0.000250007       100.524           1           1     3742   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    2999       11830.9    0.00236328       193.309           1           1     3889   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    3099       11831.3   0.000309242       194.211      0.7059      0.7059     4015   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    3199       11831.4    1.3396e-05       91.8042      0.9217      0.9217     4136   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    3299       11831.6   0.000373334       77.3538      0.3184           1     4256   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    3399       11831.8   0.000125272       64.7127           1           1     4379   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    3499         11832     0.0010491       69.8273           1           1     4503   
<br />    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
<br />    3553       11832.1   1.09422e-05       89.3197   8.979e-08       0.001     4612  LS failed, Hessian reset 
<br />    3584       11832.1   8.65844e-07       55.9367      0.4252      0.4252     4658   
<br />Optimization terminated normally: 
<br />  Convergence detected: relative gradient magnitude is below tolerance
<br />>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fb862163f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-19 23:52:51.563684
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.3339
<br />metric_name                                  mean_absolute_error
<br />Name: 8, dtype: object 
<br />
<br />  date_run                              2020-05-19 23:52:51.567621
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   215.367
<br />metric_name                                   mean_squared_error
<br />Name: 9, dtype: object 
<br />
<br />  date_run                              2020-05-19 23:52:51.572155
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.4309
<br />metric_name                                median_absolute_error
<br />Name: 10, dtype: object 
<br />
<br />  date_run                              2020-05-19 23:52:51.575889
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -18.2877
<br />metric_name                                             r2_score
<br />Name: 11, dtype: object 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} 'model_uri' 
<br />
<br />  benchmark file saved at https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/example/benchmark/ 
<br />
<br />                        date_run  ...            metric_name
<br />0   2020-05-19 23:52:09.481992  ...    mean_absolute_error
<br />1   2020-05-19 23:52:09.487456  ...     mean_squared_error
<br />2   2020-05-19 23:52:09.492036  ...  median_absolute_error
<br />3   2020-05-19 23:52:09.496052  ...               r2_score
<br />4   2020-05-19 23:52:32.883057  ...    mean_absolute_error
<br />5   2020-05-19 23:52:32.889854  ...     mean_squared_error
<br />6   2020-05-19 23:52:32.898032  ...  median_absolute_error
<br />7   2020-05-19 23:52:32.904347  ...               r2_score
<br />8   2020-05-19 23:52:51.563684  ...    mean_absolute_error
<br />9   2020-05-19 23:52:51.567621  ...     mean_squared_error
<br />10  2020-05-19 23:52:51.572155  ...  median_absolute_error
<br />11  2020-05-19 23:52:51.575889  ...               r2_score
<br />
<br />[12 rows x 6 columns] 



### Error 14, [Traceback at line 3364](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L3364)<br />3364..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42/mlmodels/benchmark.py", line 118, in benchmark_run
<br />    model_uri =  model_pars['model_uri']
<br />KeyError: 'model_uri'
