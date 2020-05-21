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
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 534, in main
<br />    predict_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 445, in predict_cli
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
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/models.py", line 534, in main
<br />    predict_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/models.py", line 445, in predict_cli
<br />    model, session = load(module, load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/models.py", line 156, in load
<br />    return module.load(load_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/model_tf/1_lstm.py", line 199, in load
<br />    return load_tf(load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/util.py", line 477, in load_tf
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
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 534, in main
<br />    predict_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 445, in predict_cli
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
<br />  <module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/model_tf/1_lstm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.model_tf.1_lstm.Model object at 0x7f6895b160f0> 
<br />
<br />  #### Fit   ######################################################## 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br /> [-0.03069999 -0.1822467   0.00696348  0.03670787  0.0106844   0.20348258]
<br /> [ 0.03068816  0.03057728  0.20433538 -0.12523369  0.10438476 -0.07547674]
<br /> [ 0.10906137  0.34426051  0.2851432  -0.10046719 -0.07562888  0.09481577]
<br /> [-0.37409228  0.17084943  0.17283995  0.23940913  0.3191866   0.00088652]
<br /> [-0.12595221  0.15908636  0.41750064  0.49967936  0.65039998 -0.16001183]
<br /> [ 0.36700198  0.20850816  0.56976038  0.27818251  0.01285871 -0.05553106]
<br /> [ 0.31350756 -0.46824601  0.39079225  0.43949685 -0.04415305 -0.29449469]
<br /> [ 0.55916333  0.53047162  0.54817855  0.23728839 -0.52261293  0.35036731]
<br /> [ 0.          0.          0.          0.          0.          0.        ]]
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />model_tf.1_lstm
<br />model_tf.1_lstm
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.3444883469492197, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-21 23:13:25.403631: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
<br />
<br />Key Variable not found in checkpoint
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 526, in main
<br />    test_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 458, in test_cli
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
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.4483088441193104, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-21 23:13:26.567422: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
<br />
<br />Key Variable not found in checkpoint
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 526, in main
<br />    test_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 460, in test_cli
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
<br />  <module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/example/custom_model/1_lstm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.example.custom_model.1_lstm.Model object at 0x7f4a8943c160> 
<br />
<br />  #### Fit   ######################################################## 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />[[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
<br />   0.00000000e+00  0.00000000e+00]
<br /> [-1.46355815e-02  1.57936379e-01 -5.86656183e-02  1.55156776e-01
<br />   6.65646512e-03  1.53786257e-01]
<br /> [ 3.61063182e-02  1.06124431e-01 -8.35187435e-02 -2.31522359e-02
<br />   1.02251321e-01  6.44274503e-02]
<br /> [ 2.19564900e-01  2.50041217e-01 -1.49312377e-01  3.52083109e-02
<br />  -5.42268902e-02  4.17273909e-01]
<br /> [ 4.91373748e-01 -4.75099176e-01  1.50813982e-01  6.24219179e-01
<br />  -7.10538495e-03  3.16861063e-01]
<br /> [-4.33306256e-03 -8.25399011e-02  7.73325935e-02  2.41637424e-01
<br />   2.82679945e-01  4.58430350e-01]
<br /> [-8.00745636e-02 -9.84914899e-02 -8.93045664e-02  6.95714578e-02
<br />  -2.87303906e-02  6.47570252e-01]
<br /> [-6.66429754e-04  6.02769196e-01  3.40615283e-03  1.30700558e-01
<br />  -1.70556046e-02  5.89168310e-01]
<br /> [-7.43779168e-02 -6.55157566e-02 -3.57593924e-01  2.12898344e-01
<br />   2.03557089e-02  7.48671770e-01]
<br /> [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
<br />   0.00000000e+00  0.00000000e+00]]
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />example/custom_model/1_lstm.py
<br />example.custom_model.1_lstm.py
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/example/custom_model/1_lstm.py'>
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/example/custom_model/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.4077007584273815, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />2020-05-21 23:13:31.477563: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />{'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
<br />
<br />Key Variable not found in checkpoint
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 526, in main
<br />    test_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 458, in test_cli
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
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/example/custom_model/1_lstm.py'>
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/example/custom_model/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.44744500145316124, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />2020-05-21 23:13:32.629906: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />{'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
<br />
<br />Key Variable not found in checkpoint
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 526, in main
<br />    test_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 460, in test_cli
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



### Error 5, [Traceback at line 959](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L959)<br />959..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/optim.py", line 388, in main
<br />    optim_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/optim.py", line 259, in optim_cli
<br />    out_pars        = out_pars )
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/optim.py", line 54, in optim
<br />    if hypermodel_pars["engine_pars"]['engine'] == "optuna":
<br />KeyError: 'engine_pars'



### Error 6, [Traceback at line 2135](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2135)<br />2135..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 7, [Traceback at line 2170](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2170)<br />2170..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 8, [Traceback at line 2210](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2210)<br />2210..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 9, [Traceback at line 2245](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2245)<br />2245..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 10, [Traceback at line 2290](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2290)<br />2290..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 11, [Traceback at line 2325](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2325)<br />2325..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 12, [Traceback at line 2382](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2382)<br />2382..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 13, [Traceback at line 2386](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2386)<br />2386..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
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
<br />  json_path https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/json/benchmark_timeseries/test01/ 
<br />
<br />  Model List [{'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}}, {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}}, {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}}, {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}}] 
<br />
<br />  
<br />
<br />
<br />### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />| N-Beats
<br />| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140302377300320
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140301098027384
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140301098027888
<br />| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140301098028392
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140301098028896
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140301098164632
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f9abbf45b38> <class 'mlmodels.model_tch.nbeats.Model'>
<br />[[0.40504701]
<br /> [0.40695405]
<br /> [0.39710839]
<br /> ...
<br /> [0.93587014]
<br /> [0.95086039]
<br /> [0.95547277]]
<br />--- fiting ---
<br />grad_step = 000000, loss = 0.502177
<br />plot()
<br />Saved image to .//n_beats_0.png.
<br />grad_step = 000001, loss = 0.467241
<br />grad_step = 000002, loss = 0.437148
<br />grad_step = 000003, loss = 0.403539
<br />grad_step = 000004, loss = 0.366539
<br />grad_step = 000005, loss = 0.327072
<br />grad_step = 000006, loss = 0.286643
<br />grad_step = 000007, loss = 0.253583
<br />grad_step = 000008, loss = 0.239882
<br />grad_step = 000009, loss = 0.224747
<br />grad_step = 000010, loss = 0.204142
<br />grad_step = 000011, loss = 0.186625
<br />grad_step = 000012, loss = 0.175394
<br />grad_step = 000013, loss = 0.167980
<br />grad_step = 000014, loss = 0.160930
<br />grad_step = 000015, loss = 0.152917
<br />grad_step = 000016, loss = 0.144242
<br />grad_step = 000017, loss = 0.135920
<br />grad_step = 000018, loss = 0.128751
<br />grad_step = 000019, loss = 0.122373
<br />grad_step = 000020, loss = 0.115970
<br />grad_step = 000021, loss = 0.109009
<br />grad_step = 000022, loss = 0.101650
<br />grad_step = 000023, loss = 0.094866
<br />grad_step = 000024, loss = 0.089060
<br />grad_step = 000025, loss = 0.083737
<br />grad_step = 000026, loss = 0.078373
<br />grad_step = 000027, loss = 0.072930
<br />grad_step = 000028, loss = 0.067757
<br />grad_step = 000029, loss = 0.063191
<br />grad_step = 000030, loss = 0.059177
<br />grad_step = 000031, loss = 0.055370
<br />grad_step = 000032, loss = 0.051640
<br />grad_step = 000033, loss = 0.047987
<br />grad_step = 000034, loss = 0.044388
<br />grad_step = 000035, loss = 0.041037
<br />grad_step = 000036, loss = 0.038034
<br />grad_step = 000037, loss = 0.035216
<br />grad_step = 000038, loss = 0.032463
<br />grad_step = 000039, loss = 0.029828
<br />grad_step = 000040, loss = 0.027400
<br />grad_step = 000041, loss = 0.025208
<br />grad_step = 000042, loss = 0.023212
<br />grad_step = 000043, loss = 0.021351
<br />grad_step = 000044, loss = 0.019627
<br />grad_step = 000045, loss = 0.018063
<br />grad_step = 000046, loss = 0.016637
<br />grad_step = 000047, loss = 0.015292
<br />grad_step = 000048, loss = 0.014033
<br />grad_step = 000049, loss = 0.012916
<br />grad_step = 000050, loss = 0.011913
<br />grad_step = 000051, loss = 0.010933
<br />grad_step = 000052, loss = 0.009990
<br />grad_step = 000053, loss = 0.009170
<br />grad_step = 000054, loss = 0.008503
<br />grad_step = 000055, loss = 0.007909
<br />grad_step = 000056, loss = 0.007323
<br />grad_step = 000057, loss = 0.006779
<br />grad_step = 000058, loss = 0.006320
<br />grad_step = 000059, loss = 0.005922
<br />grad_step = 000060, loss = 0.005544
<br />grad_step = 000061, loss = 0.005186
<br />grad_step = 000062, loss = 0.004864
<br />grad_step = 000063, loss = 0.004577
<br />grad_step = 000064, loss = 0.004315
<br />grad_step = 000065, loss = 0.004079
<br />grad_step = 000066, loss = 0.003877
<br />grad_step = 000067, loss = 0.003705
<br />grad_step = 000068, loss = 0.003549
<br />grad_step = 000069, loss = 0.003400
<br />grad_step = 000070, loss = 0.003265
<br />grad_step = 000071, loss = 0.003147
<br />grad_step = 000072, loss = 0.003044
<br />grad_step = 000073, loss = 0.002950
<br />grad_step = 000074, loss = 0.002867
<br />grad_step = 000075, loss = 0.002794
<br />grad_step = 000076, loss = 0.002729
<br />grad_step = 000077, loss = 0.002666
<br />grad_step = 000078, loss = 0.002611
<br />grad_step = 000079, loss = 0.002568
<br />grad_step = 000080, loss = 0.002528
<br />grad_step = 000081, loss = 0.002489
<br />grad_step = 000082, loss = 0.002456
<br />grad_step = 000083, loss = 0.002431
<br />grad_step = 000084, loss = 0.002407
<br />grad_step = 000085, loss = 0.002381
<br />grad_step = 000086, loss = 0.002358
<br />grad_step = 000087, loss = 0.002340
<br />grad_step = 000088, loss = 0.002324
<br />grad_step = 000089, loss = 0.002309
<br />grad_step = 000090, loss = 0.002295
<br />grad_step = 000091, loss = 0.002283
<br />grad_step = 000092, loss = 0.002272
<br />grad_step = 000093, loss = 0.002263
<br />grad_step = 000094, loss = 0.002255
<br />grad_step = 000095, loss = 0.002247
<br />grad_step = 000096, loss = 0.002240
<br />grad_step = 000097, loss = 0.002233
<br />grad_step = 000098, loss = 0.002227
<br />grad_step = 000099, loss = 0.002222
<br />grad_step = 000100, loss = 0.002217
<br />plot()
<br />Saved image to .//n_beats_100.png.
<br />grad_step = 000101, loss = 0.002213
<br />grad_step = 000102, loss = 0.002210
<br />grad_step = 000103, loss = 0.002207
<br />grad_step = 000104, loss = 0.002203
<br />grad_step = 000105, loss = 0.002201
<br />grad_step = 000106, loss = 0.002199
<br />grad_step = 000107, loss = 0.002196
<br />grad_step = 000108, loss = 0.002194
<br />grad_step = 000109, loss = 0.002192
<br />grad_step = 000110, loss = 0.002191
<br />grad_step = 000111, loss = 0.002189
<br />grad_step = 000112, loss = 0.002187
<br />grad_step = 000113, loss = 0.002186
<br />grad_step = 000114, loss = 0.002184
<br />grad_step = 000115, loss = 0.002183
<br />grad_step = 000116, loss = 0.002181
<br />grad_step = 000117, loss = 0.002180
<br />grad_step = 000118, loss = 0.002178
<br />grad_step = 000119, loss = 0.002177
<br />grad_step = 000120, loss = 0.002175
<br />grad_step = 000121, loss = 0.002174
<br />grad_step = 000122, loss = 0.002172
<br />grad_step = 000123, loss = 0.002171
<br />grad_step = 000124, loss = 0.002169
<br />grad_step = 000125, loss = 0.002168
<br />grad_step = 000126, loss = 0.002166
<br />grad_step = 000127, loss = 0.002165
<br />grad_step = 000128, loss = 0.002163
<br />grad_step = 000129, loss = 0.002162
<br />grad_step = 000130, loss = 0.002161
<br />grad_step = 000131, loss = 0.002159
<br />grad_step = 000132, loss = 0.002158
<br />grad_step = 000133, loss = 0.002156
<br />grad_step = 000134, loss = 0.002155
<br />grad_step = 000135, loss = 0.002154
<br />grad_step = 000136, loss = 0.002152
<br />grad_step = 000137, loss = 0.002151
<br />grad_step = 000138, loss = 0.002150
<br />grad_step = 000139, loss = 0.002148
<br />grad_step = 000140, loss = 0.002147
<br />grad_step = 000141, loss = 0.002146
<br />grad_step = 000142, loss = 0.002145
<br />grad_step = 000143, loss = 0.002143
<br />grad_step = 000144, loss = 0.002142
<br />grad_step = 000145, loss = 0.002141
<br />grad_step = 000146, loss = 0.002139
<br />grad_step = 000147, loss = 0.002138
<br />grad_step = 000148, loss = 0.002137
<br />grad_step = 000149, loss = 0.002136
<br />grad_step = 000150, loss = 0.002134
<br />grad_step = 000151, loss = 0.002133
<br />grad_step = 000152, loss = 0.002132
<br />grad_step = 000153, loss = 0.002131
<br />grad_step = 000154, loss = 0.002129
<br />grad_step = 000155, loss = 0.002128
<br />grad_step = 000156, loss = 0.002127
<br />grad_step = 000157, loss = 0.002125
<br />grad_step = 000158, loss = 0.002124
<br />grad_step = 000159, loss = 0.002123
<br />grad_step = 000160, loss = 0.002122
<br />grad_step = 000161, loss = 0.002120
<br />grad_step = 000162, loss = 0.002119
<br />grad_step = 000163, loss = 0.002118
<br />grad_step = 000164, loss = 0.002116
<br />grad_step = 000165, loss = 0.002115
<br />grad_step = 000166, loss = 0.002114
<br />grad_step = 000167, loss = 0.002112
<br />grad_step = 000168, loss = 0.002111
<br />grad_step = 000169, loss = 0.002110
<br />grad_step = 000170, loss = 0.002108
<br />grad_step = 000171, loss = 0.002107
<br />grad_step = 000172, loss = 0.002105
<br />grad_step = 000173, loss = 0.002104
<br />grad_step = 000174, loss = 0.002103
<br />grad_step = 000175, loss = 0.002101
<br />grad_step = 000176, loss = 0.002100
<br />grad_step = 000177, loss = 0.002099
<br />grad_step = 000178, loss = 0.002097
<br />grad_step = 000179, loss = 0.002096
<br />grad_step = 000180, loss = 0.002094
<br />grad_step = 000181, loss = 0.002093
<br />grad_step = 000182, loss = 0.002091
<br />grad_step = 000183, loss = 0.002090
<br />grad_step = 000184, loss = 0.002089
<br />grad_step = 000185, loss = 0.002087
<br />grad_step = 000186, loss = 0.002086
<br />grad_step = 000187, loss = 0.002084
<br />grad_step = 000188, loss = 0.002083
<br />grad_step = 000189, loss = 0.002081
<br />grad_step = 000190, loss = 0.002080
<br />grad_step = 000191, loss = 0.002078
<br />grad_step = 000192, loss = 0.002077
<br />grad_step = 000193, loss = 0.002075
<br />grad_step = 000194, loss = 0.002074
<br />grad_step = 000195, loss = 0.002072
<br />grad_step = 000196, loss = 0.002071
<br />grad_step = 000197, loss = 0.002069
<br />grad_step = 000198, loss = 0.002068
<br />grad_step = 000199, loss = 0.002066
<br />grad_step = 000200, loss = 0.002065
<br />plot()
<br />Saved image to .//n_beats_200.png.
<br />grad_step = 000201, loss = 0.002063
<br />grad_step = 000202, loss = 0.002062
<br />grad_step = 000203, loss = 0.002060
<br />grad_step = 000204, loss = 0.002059
<br />grad_step = 000205, loss = 0.002057
<br />grad_step = 000206, loss = 0.002056
<br />grad_step = 000207, loss = 0.002054
<br />grad_step = 000208, loss = 0.002052
<br />grad_step = 000209, loss = 0.002051
<br />grad_step = 000210, loss = 0.002049
<br />grad_step = 000211, loss = 0.002048
<br />grad_step = 000212, loss = 0.002046
<br />grad_step = 000213, loss = 0.002044
<br />grad_step = 000214, loss = 0.002043
<br />grad_step = 000215, loss = 0.002041
<br />grad_step = 000216, loss = 0.002039
<br />grad_step = 000217, loss = 0.002038
<br />grad_step = 000218, loss = 0.002036
<br />grad_step = 000219, loss = 0.002035
<br />grad_step = 000220, loss = 0.002033
<br />grad_step = 000221, loss = 0.002031
<br />grad_step = 000222, loss = 0.002030
<br />grad_step = 000223, loss = 0.002028
<br />grad_step = 000224, loss = 0.002026
<br />grad_step = 000225, loss = 0.002024
<br />grad_step = 000226, loss = 0.002023
<br />grad_step = 000227, loss = 0.002021
<br />grad_step = 000228, loss = 0.002019
<br />grad_step = 000229, loss = 0.002018
<br />grad_step = 000230, loss = 0.002016
<br />grad_step = 000231, loss = 0.002014
<br />grad_step = 000232, loss = 0.002012
<br />grad_step = 000233, loss = 0.002011
<br />grad_step = 000234, loss = 0.002009
<br />grad_step = 000235, loss = 0.002007
<br />grad_step = 000236, loss = 0.002005
<br />grad_step = 000237, loss = 0.002003
<br />grad_step = 000238, loss = 0.002002
<br />grad_step = 000239, loss = 0.002000
<br />grad_step = 000240, loss = 0.001998
<br />grad_step = 000241, loss = 0.001996
<br />grad_step = 000242, loss = 0.001994
<br />grad_step = 000243, loss = 0.001993
<br />grad_step = 000244, loss = 0.001991
<br />grad_step = 000245, loss = 0.001989
<br />grad_step = 000246, loss = 0.001987
<br />grad_step = 000247, loss = 0.001985
<br />grad_step = 000248, loss = 0.001983
<br />grad_step = 000249, loss = 0.001981
<br />grad_step = 000250, loss = 0.001979
<br />grad_step = 000251, loss = 0.001977
<br />grad_step = 000252, loss = 0.001975
<br />grad_step = 000253, loss = 0.001973
<br />grad_step = 000254, loss = 0.001971
<br />grad_step = 000255, loss = 0.001969
<br />grad_step = 000256, loss = 0.001967
<br />grad_step = 000257, loss = 0.001965
<br />grad_step = 000258, loss = 0.001963
<br />grad_step = 000259, loss = 0.001961
<br />grad_step = 000260, loss = 0.001959
<br />grad_step = 000261, loss = 0.001957
<br />grad_step = 000262, loss = 0.001956
<br />grad_step = 000263, loss = 0.001957
<br />grad_step = 000264, loss = 0.001961
<br />grad_step = 000265, loss = 0.001968
<br />grad_step = 000266, loss = 0.001969
<br />grad_step = 000267, loss = 0.001965
<br />grad_step = 000268, loss = 0.001952
<br />grad_step = 000269, loss = 0.001943
<br />grad_step = 000270, loss = 0.001939
<br />grad_step = 000271, loss = 0.001940
<br />grad_step = 000272, loss = 0.001942
<br />grad_step = 000273, loss = 0.001944
<br />grad_step = 000274, loss = 0.001940
<br />grad_step = 000275, loss = 0.001931
<br />grad_step = 000276, loss = 0.001923
<br />grad_step = 000277, loss = 0.001920
<br />grad_step = 000278, loss = 0.001920
<br />grad_step = 000279, loss = 0.001919
<br />grad_step = 000280, loss = 0.001918
<br />grad_step = 000281, loss = 0.001917
<br />grad_step = 000282, loss = 0.001918
<br />grad_step = 000283, loss = 0.001921
<br />grad_step = 000284, loss = 0.001921
<br />grad_step = 000285, loss = 0.001921
<br />grad_step = 000286, loss = 0.001920
<br />grad_step = 000287, loss = 0.001924
<br />grad_step = 000288, loss = 0.001927
<br />grad_step = 000289, loss = 0.001927
<br />grad_step = 000290, loss = 0.001919
<br />grad_step = 000291, loss = 0.001909
<br />grad_step = 000292, loss = 0.001896
<br />grad_step = 000293, loss = 0.001885
<br />grad_step = 000294, loss = 0.001873
<br />grad_step = 000295, loss = 0.001868
<br />grad_step = 000296, loss = 0.001867
<br />grad_step = 000297, loss = 0.001868
<br />grad_step = 000298, loss = 0.001871
<br />grad_step = 000299, loss = 0.001877
<br />grad_step = 000300, loss = 0.001893
<br />plot()
<br />Saved image to .//n_beats_300.png.
<br />grad_step = 000301, loss = 0.001922
<br />grad_step = 000302, loss = 0.001968
<br />grad_step = 000303, loss = 0.002018
<br />grad_step = 000304, loss = 0.002050
<br />grad_step = 000305, loss = 0.002012
<br />grad_step = 000306, loss = 0.001922
<br />grad_step = 000307, loss = 0.001839
<br />grad_step = 000308, loss = 0.001836
<br />grad_step = 000309, loss = 0.001890
<br />grad_step = 000310, loss = 0.001923
<br />grad_step = 000311, loss = 0.001902
<br />grad_step = 000312, loss = 0.001849
<br />grad_step = 000313, loss = 0.001813
<br />grad_step = 000314, loss = 0.001817
<br />grad_step = 000315, loss = 0.001846
<br />grad_step = 000316, loss = 0.001866
<br />grad_step = 000317, loss = 0.001851
<br />grad_step = 000318, loss = 0.001816
<br />grad_step = 000319, loss = 0.001792
<br />grad_step = 000320, loss = 0.001789
<br />grad_step = 000321, loss = 0.001799
<br />grad_step = 000322, loss = 0.001812
<br />grad_step = 000323, loss = 0.001822
<br />grad_step = 000324, loss = 0.001819
<br />grad_step = 000325, loss = 0.001805
<br />grad_step = 000326, loss = 0.001788
<br />grad_step = 000327, loss = 0.001774
<br />grad_step = 000328, loss = 0.001764
<br />grad_step = 000329, loss = 0.001755
<br />grad_step = 000330, loss = 0.001750
<br />grad_step = 000331, loss = 0.001748
<br />grad_step = 000332, loss = 0.001748
<br />grad_step = 000333, loss = 0.001751
<br />grad_step = 000334, loss = 0.001763
<br />grad_step = 000335, loss = 0.001800
<br />grad_step = 000336, loss = 0.001874
<br />grad_step = 000337, loss = 0.002031
<br />grad_step = 000338, loss = 0.002112
<br />grad_step = 000339, loss = 0.002053
<br />grad_step = 000340, loss = 0.001853
<br />grad_step = 000341, loss = 0.001734
<br />grad_step = 000342, loss = 0.001821
<br />grad_step = 000343, loss = 0.001923
<br />grad_step = 000344, loss = 0.001858
<br />grad_step = 000345, loss = 0.001736
<br />grad_step = 000346, loss = 0.001729
<br />grad_step = 000347, loss = 0.001819
<br />grad_step = 000348, loss = 0.001855
<br />grad_step = 000349, loss = 0.001798
<br />grad_step = 000350, loss = 0.001713
<br />grad_step = 000351, loss = 0.001720
<br />grad_step = 000352, loss = 0.001785
<br />grad_step = 000353, loss = 0.001792
<br />grad_step = 000354, loss = 0.001731
<br />grad_step = 000355, loss = 0.001696
<br />grad_step = 000356, loss = 0.001724
<br />grad_step = 000357, loss = 0.001751
<br />grad_step = 000358, loss = 0.001732
<br />grad_step = 000359, loss = 0.001695
<br />grad_step = 000360, loss = 0.001689
<br />grad_step = 000361, loss = 0.001710
<br />grad_step = 000362, loss = 0.001721
<br />grad_step = 000363, loss = 0.001705
<br />grad_step = 000364, loss = 0.001681
<br />grad_step = 000365, loss = 0.001676
<br />grad_step = 000366, loss = 0.001687
<br />grad_step = 000367, loss = 0.001703
<br />grad_step = 000368, loss = 0.001709
<br />grad_step = 000369, loss = 0.001705
<br />grad_step = 000370, loss = 0.001681
<br />grad_step = 000371, loss = 0.001665
<br />grad_step = 000372, loss = 0.001663
<br />grad_step = 000373, loss = 0.001673
<br />grad_step = 000374, loss = 0.001680
<br />grad_step = 000375, loss = 0.001679
<br />grad_step = 000376, loss = 0.001670
<br />grad_step = 000377, loss = 0.001657
<br />grad_step = 000378, loss = 0.001651
<br />grad_step = 000379, loss = 0.001648
<br />grad_step = 000380, loss = 0.001652
<br />grad_step = 000381, loss = 0.001655
<br />grad_step = 000382, loss = 0.001657
<br />grad_step = 000383, loss = 0.001657
<br />grad_step = 000384, loss = 0.001651
<br />grad_step = 000385, loss = 0.001646
<br />grad_step = 000386, loss = 0.001640
<br />grad_step = 000387, loss = 0.001634
<br />grad_step = 000388, loss = 0.001630
<br />grad_step = 000389, loss = 0.001626
<br />grad_step = 000390, loss = 0.001624
<br />grad_step = 000391, loss = 0.001621
<br />grad_step = 000392, loss = 0.001618
<br />grad_step = 000393, loss = 0.001616
<br />grad_step = 000394, loss = 0.001614
<br />grad_step = 000395, loss = 0.001614
<br />grad_step = 000396, loss = 0.001617
<br />grad_step = 000397, loss = 0.001626
<br />grad_step = 000398, loss = 0.001650
<br />grad_step = 000399, loss = 0.001714
<br />grad_step = 000400, loss = 0.001823
<br />plot()
<br />Saved image to .//n_beats_400.png.
<br />grad_step = 000401, loss = 0.002020
<br />grad_step = 000402, loss = 0.002229
<br />grad_step = 000403, loss = 0.002159
<br />grad_step = 000404, loss = 0.001857
<br />grad_step = 000405, loss = 0.001625
<br />grad_step = 000406, loss = 0.001777
<br />grad_step = 000407, loss = 0.001938
<br />grad_step = 000408, loss = 0.001728
<br />grad_step = 000409, loss = 0.001621
<br />grad_step = 000410, loss = 0.001773
<br />grad_step = 000411, loss = 0.001801
<br />grad_step = 000412, loss = 0.001652
<br />grad_step = 000413, loss = 0.001610
<br />grad_step = 000414, loss = 0.001737
<br />grad_step = 000415, loss = 0.001798
<br />grad_step = 000416, loss = 0.001621
<br />grad_step = 000417, loss = 0.001590
<br />grad_step = 000418, loss = 0.001680
<br />grad_step = 000419, loss = 0.001692
<br />grad_step = 000420, loss = 0.001592
<br />grad_step = 000421, loss = 0.001575
<br />grad_step = 000422, loss = 0.001618
<br />grad_step = 000423, loss = 0.001626
<br />grad_step = 000424, loss = 0.001583
<br />grad_step = 000425, loss = 0.001560
<br />grad_step = 000426, loss = 0.001581
<br />grad_step = 000427, loss = 0.001590
<br />grad_step = 000428, loss = 0.001569
<br />grad_step = 000429, loss = 0.001544
<br />grad_step = 000430, loss = 0.001552
<br />grad_step = 000431, loss = 0.001566
<br />grad_step = 000432, loss = 0.001554
<br />grad_step = 000433, loss = 0.001532
<br />grad_step = 000434, loss = 0.001531
<br />grad_step = 000435, loss = 0.001544
<br />grad_step = 000436, loss = 0.001543
<br />grad_step = 000437, loss = 0.001530
<br />grad_step = 000438, loss = 0.001518
<br />grad_step = 000439, loss = 0.001521
<br />grad_step = 000440, loss = 0.001527
<br />grad_step = 000441, loss = 0.001524
<br />grad_step = 000442, loss = 0.001516
<br />grad_step = 000443, loss = 0.001508
<br />grad_step = 000444, loss = 0.001505
<br />grad_step = 000445, loss = 0.001503
<br />grad_step = 000446, loss = 0.001504
<br />grad_step = 000447, loss = 0.001504
<br />grad_step = 000448, loss = 0.001500
<br />grad_step = 000449, loss = 0.001493
<br />grad_step = 000450, loss = 0.001486
<br />grad_step = 000451, loss = 0.001483
<br />grad_step = 000452, loss = 0.001482
<br />grad_step = 000453, loss = 0.001481
<br />grad_step = 000454, loss = 0.001480
<br />grad_step = 000455, loss = 0.001480
<br />grad_step = 000456, loss = 0.001480
<br />grad_step = 000457, loss = 0.001480
<br />grad_step = 000458, loss = 0.001480
<br />grad_step = 000459, loss = 0.001482
<br />grad_step = 000460, loss = 0.001491
<br />grad_step = 000461, loss = 0.001508
<br />grad_step = 000462, loss = 0.001538
<br />grad_step = 000463, loss = 0.001597
<br />grad_step = 000464, loss = 0.001654
<br />grad_step = 000465, loss = 0.001716
<br />grad_step = 000466, loss = 0.001658
<br />grad_step = 000467, loss = 0.001550
<br />grad_step = 000468, loss = 0.001463
<br />grad_step = 000469, loss = 0.001476
<br />grad_step = 000470, loss = 0.001543
<br />grad_step = 000471, loss = 0.001569
<br />grad_step = 000472, loss = 0.001540
<br />grad_step = 000473, loss = 0.001472
<br />grad_step = 000474, loss = 0.001435
<br />grad_step = 000475, loss = 0.001437
<br />grad_step = 000476, loss = 0.001457
<br />grad_step = 000477, loss = 0.001493
<br />grad_step = 000478, loss = 0.001544
<br />grad_step = 000479, loss = 0.001611
<br />grad_step = 000480, loss = 0.001659
<br />grad_step = 000481, loss = 0.001688
<br />grad_step = 000482, loss = 0.001635
<br />grad_step = 000483, loss = 0.001546
<br />grad_step = 000484, loss = 0.001446
<br />grad_step = 000485, loss = 0.001412
<br />grad_step = 000486, loss = 0.001450
<br />grad_step = 000487, loss = 0.001504
<br />grad_step = 000488, loss = 0.001520
<br />grad_step = 000489, loss = 0.001476
<br />grad_step = 000490, loss = 0.001423
<br />grad_step = 000491, loss = 0.001402
<br />grad_step = 000492, loss = 0.001422
<br />grad_step = 000493, loss = 0.001453
<br />grad_step = 000494, loss = 0.001457
<br />grad_step = 000495, loss = 0.001435
<br />grad_step = 000496, loss = 0.001404
<br />grad_step = 000497, loss = 0.001390
<br />grad_step = 000498, loss = 0.001396
<br />grad_step = 000499, loss = 0.001412
<br />grad_step = 000500, loss = 0.001424
<br />plot()
<br />Saved image to .//n_beats_500.png.
<br />grad_step = 000501, loss = 0.001424
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
<br />  date_run                              2020-05-21 23:48:28.460736
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.200288
<br />metric_name                                  mean_absolute_error
<br />Name: 0, dtype: object 
<br />
<br />  date_run                              2020-05-21 23:48:28.467420
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                 0.0852241
<br />metric_name                                   mean_squared_error
<br />Name: 1, dtype: object 
<br />
<br />  date_run                              2020-05-21 23:48:28.476759
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.133654
<br />metric_name                                median_absolute_error
<br />Name: 2, dtype: object 
<br />
<br />  date_run                              2020-05-21 23:48:28.482038
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -0.29501
<br />metric_name                                             r2_score
<br />Name: 3, dtype: object 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_keras.armdn.py', 'lstm_h_list': [300, 200, 24], 'last_lstm_neuron': 12, 'timesteps': 60, 'dropout_rate': 0.1, 'n_mixes': 3, 'dense_neuron': 10}, 'data_pars': {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60}, 'compute_pars': {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.05, 'patience': 50}, 'out_pars': {'outpath': 'ztest/model_keras/armdn/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60} {'outpath': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_keras/armdn/'} 
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
<br />>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f9abbf45ba8> <class 'mlmodels.model_keras.armdn.Model'>
<br />
<br />  #### Loading dataset   ############################################# 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
<br />
<br />Epoch 1/10
<br />
<br />1/1 [==============================] - 2s 2s/step - loss: 353509.3438
<br />Epoch 2/10
<br />
<br />1/1 [==============================] - 0s 107ms/step - loss: 304378.6562
<br />Epoch 3/10
<br />
<br />1/1 [==============================] - 0s 98ms/step - loss: 222374.7969
<br />Epoch 4/10
<br />
<br />1/1 [==============================] - 0s 105ms/step - loss: 151185.7188
<br />Epoch 5/10
<br />
<br />1/1 [==============================] - 0s 123ms/step - loss: 94533.1719
<br />Epoch 6/10
<br />
<br />1/1 [==============================] - 0s 101ms/step - loss: 55800.2930
<br />Epoch 7/10
<br />
<br />1/1 [==============================] - 0s 101ms/step - loss: 31922.9082
<br />Epoch 8/10
<br />
<br />1/1 [==============================] - 0s 96ms/step - loss: 18975.0918
<br />Epoch 9/10
<br />
<br />1/1 [==============================] - 0s 98ms/step - loss: 12236.0879
<br />Epoch 10/10
<br />
<br />1/1 [==============================] - 0s 105ms/step - loss: 8504.6396
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />[[ 1.03833973e-01  8.21784496e+00  6.96071100e+00  7.26656866e+00
<br />   7.21364880e+00  6.80036688e+00  7.18151569e+00  6.77982378e+00
<br />   8.05128098e+00  6.46238947e+00  8.27004147e+00  7.33505535e+00
<br />   7.71431875e+00  6.89083099e+00  6.06804514e+00  8.18919849e+00
<br />   7.43156481e+00  8.14982700e+00  7.53400850e+00  7.27336597e+00
<br />   7.05945396e+00  7.49302912e+00  6.23757505e+00  6.77064514e+00
<br />   7.82350206e+00  6.18828630e+00  6.78334713e+00  8.42769432e+00
<br />   7.68590736e+00  6.63360643e+00  7.99645042e+00  5.76417255e+00
<br />   8.03851700e+00  8.07216358e+00  7.04125214e+00  7.81087399e+00
<br />   8.97234058e+00  8.09528255e+00  7.69680309e+00  7.21749544e+00
<br />   7.49654293e+00  6.65135241e+00  7.34794044e+00  8.77273655e+00
<br />   7.12544918e+00  6.54900312e+00  7.19441366e+00  8.24079323e+00
<br />   6.46710062e+00  7.10944796e+00  8.54912567e+00  6.89164495e+00
<br />   6.92877579e+00  7.88780451e+00  6.32467127e+00  7.17242098e+00
<br />   5.95299625e+00  8.39504147e+00  6.70234632e+00  5.52969790e+00
<br />   1.94799340e+00 -1.96779281e-01 -5.84490061e-01  3.37745309e-01
<br />  -6.38250053e-01 -9.77267861e-01 -4.54918504e-01  7.75394917e-01
<br />  -1.24619329e+00 -1.22711134e+00  7.19130456e-01 -4.16826040e-01
<br />  -9.83466983e-01 -7.65399992e-01 -1.31168938e+00 -1.64184487e+00
<br />   3.80017936e-01 -4.25969660e-01  2.99796104e-01  7.18704522e-01
<br />  -1.20326793e+00  4.15865153e-01  1.00376999e+00 -2.57861167e-01
<br />  -8.38980377e-01 -8.81874025e-01 -1.78992176e+00 -1.33671594e+00
<br />   4.67875838e-01 -1.48856902e+00  1.61297250e+00 -1.07478368e+00
<br />  -6.28829896e-01 -3.50705795e-02 -8.63890052e-01 -9.00779486e-01
<br />   7.65124977e-01 -5.12779653e-01  1.17101204e+00  7.54172683e-01
<br />  -1.07112896e+00 -8.30212414e-01  1.24471164e+00 -2.26338595e-01
<br />  -1.46957785e-02  1.12332106e+00 -2.14746326e-01  1.24580193e+00
<br />   1.52421105e+00  1.40556037e+00 -1.90188551e+00  1.45758545e+00
<br />   1.36878580e-01 -8.91635954e-01  6.58092141e-01  1.08482122e+00
<br />  -1.04623890e+00  4.84168202e-01  7.16087937e-01  1.50934386e+00
<br />  -3.88022184e-01  4.18048769e-01 -1.17879677e+00 -9.37461853e-03
<br />   8.80011976e-01  6.42779589e-01  2.76236415e-01 -4.49290812e-01
<br />  -7.61635378e-02  6.69319451e-01  2.58538842e-01  2.83422410e-01
<br />  -9.32830930e-01  2.23354936e-01 -1.10513926e+00  2.44487017e-01
<br />  -4.12700564e-01  5.36940992e-01  1.88853383e-01 -1.45879114e+00
<br />  -9.83592510e-01  5.52219689e-01 -3.26123953e-01  1.74273998e-02
<br />   4.85991091e-01  7.81668782e-01  4.88250464e-01 -3.82435322e-03
<br />   2.38054633e-01 -1.38617802e+00  2.09406883e-01  1.09415913e+00
<br />  -1.21354985e+00  1.30831540e+00 -6.31104410e-02 -3.77046704e-01
<br />   3.93435925e-01 -1.29713833e-01  7.90203929e-01 -7.49496222e-02
<br />   4.19987321e-01  1.39782000e+00  7.46711195e-01 -1.98620868e+00
<br />   1.71009350e+00 -1.05886090e+00  1.24991357e+00  1.53949344e+00
<br />  -1.11567700e+00  2.29889795e-01  2.90397108e-01  7.06052005e-01
<br />   7.34313965e-01 -3.97371531e-01  1.05500853e+00 -7.19528913e-01
<br />  -3.33838969e-01 -6.02118373e-01  2.22685188e-01 -1.89758587e+00
<br />   3.33223343e-02  7.59846067e+00  6.50594378e+00  8.02429104e+00
<br />   7.15868902e+00  7.31656837e+00  8.03184414e+00  7.65750027e+00
<br />   6.63083839e+00  7.65104437e+00  8.01774406e+00  8.28887177e+00
<br />   7.69465113e+00  7.56897354e+00  7.48462534e+00  6.83902502e+00
<br />   8.09376049e+00  6.37088060e+00  6.97370815e+00  7.34185553e+00
<br />   6.86364937e+00  8.45744991e+00  7.79504585e+00  7.51083231e+00
<br />   7.56129169e+00  7.63479805e+00  8.16108418e+00  7.51911592e+00
<br />   7.03929710e+00  6.74985361e+00  8.24057579e+00  7.86640596e+00
<br />   8.09479618e+00  8.03914738e+00  6.94259644e+00  8.00867367e+00
<br />   9.43290615e+00  7.56440163e+00  7.57456112e+00  8.07355022e+00
<br />   8.17456436e+00  8.57436180e+00  7.44141006e+00  8.76702690e+00
<br />   8.37048912e+00  8.56655407e+00  6.90152311e+00  7.92394495e+00
<br />   6.93538189e+00  7.53499031e+00  6.96901369e+00  6.67113495e+00
<br />   8.44093704e+00  7.21051073e+00  6.25707817e+00  8.49839401e+00
<br />   7.72885370e+00  7.28320074e+00  6.99450159e+00  8.23169327e+00
<br />   7.48771310e-01  6.04864299e-01  7.60077715e-01  1.74769509e+00
<br />   6.20477140e-01  2.13011336e+00  8.18903387e-01  2.79226446e+00
<br />   2.13128209e-01  1.48300016e+00  1.22743273e+00  1.51631141e+00
<br />   7.96152115e-01  1.68877387e+00  2.13438845e+00  1.06298602e+00
<br />   1.08102584e+00  3.91649127e-01  1.82147896e+00  7.98778951e-01
<br />   9.26722884e-01  1.49695981e+00  3.05306852e-01  2.48718643e+00
<br />   8.50986838e-01  1.82373524e-01  4.58847284e-01  3.63876462e-01
<br />   1.63961995e+00  1.72243357e+00  1.03157997e+00  1.01835740e+00
<br />   1.89173341e-01  8.52050245e-01  2.05303848e-01  2.00970626e+00
<br />   1.36999786e+00  2.91894197e+00  1.14297664e+00  3.60469699e-01
<br />   6.53991103e-01  6.36824846e-01  1.07296109e+00  6.31316245e-01
<br />   1.62606549e+00  1.29191077e+00  1.48254967e+00  1.58716094e+00
<br />   7.71575272e-01  1.06827772e+00  2.66740990e+00  1.06705999e+00
<br />   5.34472704e-01  3.76414776e-01  1.47147536e+00  1.99477696e+00
<br />   1.03101182e+00  4.90424871e-01  1.44678831e+00  2.41127014e-01
<br />   1.87795818e-01  1.87973559e-01  1.69354773e+00  2.94321775e+00
<br />   1.64300442e+00  9.06892896e-01  9.80259895e-01  8.06250453e-01
<br />   9.24299121e-01  4.77846742e-01  1.53404272e+00  1.18369949e+00
<br />   1.69400239e+00  6.54000401e-01  9.93822277e-01  3.17975879e-01
<br />   1.68040395e+00  3.10295701e-01  1.60491157e+00  7.42436647e-01
<br />   1.54079080e+00  8.15715313e-01  7.50772834e-01  2.11407280e+00
<br />   1.04103959e+00  3.51518333e-01  6.96737885e-01  1.17792296e+00
<br />   5.55662692e-01  2.26950788e+00  5.04646182e-01  1.92204201e+00
<br />   4.23407733e-01  1.28505969e+00  3.85845184e-01  7.23940611e-01
<br />   5.93090653e-01  5.16006768e-01  4.39203084e-01  1.96350384e+00
<br />   1.86709142e+00  1.39030004e+00  2.00599909e+00  4.33756888e-01
<br />   1.33698249e+00  6.59699321e-01  9.14391041e-01  1.57469273e+00
<br />   8.71080518e-01  2.93383360e-01  1.00410867e+00  5.37235975e-01
<br />   1.60301065e+00  1.71342409e+00  1.58015442e+00  1.30778539e+00
<br />   4.30021763e-01  1.83395922e-01  6.93468034e-01  2.15767550e+00
<br />   4.64830399e+00 -9.54216671e+00 -4.87201023e+00]]
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-21 23:48:39.553679
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   94.0818
<br />metric_name                                  mean_absolute_error
<br />Name: 4, dtype: object 
<br />
<br />  date_run                              2020-05-21 23:48:39.557775
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   8871.19
<br />metric_name                                   mean_squared_error
<br />Name: 5, dtype: object 
<br />
<br />  date_run                              2020-05-21 23:48:39.563089
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   93.8877
<br />metric_name                                median_absolute_error
<br />Name: 6, dtype: object 
<br />
<br />  date_run                              2020-05-21 23:48:39.566407
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -793.478
<br />metric_name                                             r2_score
<br />Name: 7, dtype: object 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} 'model_uri' 
<br />
<br />  
<br />
<br />
<br />### Running {'model_pars': {'model_uri': 'model_gluon/fb_prophet.py'}, 'data_pars': {'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'}, 'compute_pars': {'dummy': 'dummy'}, 'out_pars': {'outpath': 'ztest/model_fb/fb_prophet/'}} ############################################ 
<br />
<br />  #### Model URI and Config JSON 
<br />
<br />  data_pars out_pars {'train_data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'} {'outpath': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/ztest/model_fb/fb_prophet/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  #### Fit  ####################################################### 



### Error 14, [Traceback at line 3234](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L3234)<br />3234..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7/mlmodels/benchmark.py", line 118, in benchmark_run
<br />    model_uri =  model_pars['model_uri']
<br />KeyError: 'model_uri'
