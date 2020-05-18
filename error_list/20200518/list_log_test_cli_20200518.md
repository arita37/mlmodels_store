## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py


### Error 1, [Traceback at line 154](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L154)<br />154..Traceback (most recent call last):
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



### Error 2, [Traceback at line 166](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L166)<br />166..Traceback (most recent call last):
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



### Error 3, [Traceback at line 221](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L221)<br />221..Traceback (most recent call last):
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



### Error 4, [Traceback at line 232](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L232)<br />232..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/models.py", line 531, in main
<br />    predict_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/models.py", line 442, in predict_cli
<br />    model, session = load(module, load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/models.py", line 156, in load
<br />    return module.load(load_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/model_tf/1_lstm.py", line 199, in load
<br />    return load_tf(load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/util.py", line 477, in load_tf
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
<br />  <module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/model_tf/1_lstm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.model_tf.1_lstm.Model object at 0x7f722dba51d0> 
<br />
<br />  #### Fit   ######################################################## 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br /> [ 0.13440098  0.00596181  0.13462314 -0.00365244  0.04439449  0.10363016]
<br /> [-0.1853438   0.07756391  0.0586026   0.00937159  0.08299484  0.1105232 ]
<br /> [-0.13686585  0.2650966   0.08812129  0.21582873  0.03761189 -0.10310491]
<br /> [-0.30487034  0.11410014 -0.0966932  -0.37476793 -0.20615144 -0.2616975 ]
<br /> [-0.35014221 -0.16949373  0.443194   -0.28434643  0.24643263 -0.02336589]
<br /> [ 0.17060404 -0.20368811  0.08307826  0.43463829 -0.45744151  0.19333874]
<br /> [-0.09773871 -0.43626496 -0.30167884  0.85507333 -0.43463856  0.28897247]
<br /> [ 0.08358903  0.2129696   0.30872372  0.42513341  0.0104204   0.25655288]
<br /> [ 0.          0.          0.          0.          0.          0.        ]]
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />model_tf.1_lstm
<br />model_tf.1_lstm
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.501961212605238, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-17 23:14:55.973812: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/model'}
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
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.5259970277547836, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-17 23:14:57.051186: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/model'}
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
<br />  <module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/example/custom_model/1_lstm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.example.custom_model.1_lstm.Model object at 0x7f10c79c42e8> 
<br />
<br />  #### Fit   ######################################################## 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br /> [-0.01031679 -0.06113058  0.00545259  0.07927281  0.04446095  0.01925521]
<br /> [-0.07230341 -0.15785593  0.09824052 -0.10584588  0.05124531  0.03995899]
<br /> [ 0.29005757 -0.32731706  0.02017794 -0.13972224  0.19514923  0.0115555 ]
<br /> [-0.03859011 -0.25110105  0.07158272 -0.29160327  0.06892191 -0.29304507]
<br /> [ 0.62600207 -0.5131017   0.21852252 -0.06558139  0.2754316  -0.27079201]
<br /> [ 0.1432927   0.3357285   0.47333226  0.02767611  0.06692155  0.1718569 ]
<br /> [ 0.29721543  0.40941313 -0.15886283  0.24339503 -0.78448957 -0.30907494]
<br /> [-0.053553    0.17207038  0.2797524   0.37389165  0.24788219  0.1564755 ]
<br /> [ 0.          0.          0.          0.          0.          0.        ]]
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />example/custom_model/1_lstm.py
<br />example.custom_model.1_lstm.py
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/example/custom_model/1_lstm.py'>
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/example/custom_model/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.45286500453948975, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />2020-05-17 23:15:01.753916: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />{'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/model'}
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
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/example/custom_model/1_lstm.py'>
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/example/custom_model/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.4501103311777115, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />2020-05-17 23:15:02.843138: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />{'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_tf/1_lstm/model'}
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



### Error 5, [Traceback at line 948](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L948)<br />948..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/optim.py", line 388, in main
<br />    optim_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/optim.py", line 259, in optim_cli
<br />    out_pars        = out_pars )
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/optim.py", line 54, in optim
<br />    if hypermodel_pars["engine_pars"]['engine'] == "optuna":
<br />KeyError: 'engine_pars'



### Error 6, [Traceback at line 2124](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2124)<br />2124..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 7, [Traceback at line 2159](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2159)<br />2159..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 8, [Traceback at line 2199](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2199)<br />2199..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 9, [Traceback at line 2234](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2234)<br />2234..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 10, [Traceback at line 2279](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2279)<br />2279..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 11, [Traceback at line 2314](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2314)<br />2314..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 12, [Traceback at line 2371](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2371)<br />2371..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 13, [Traceback at line 2375](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2375)<br />2375..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
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
<br />  json_path https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/json/benchmark_timeseries/test01/ 
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
<br />  data_pars out_pars {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60} {'outpath': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_keras/armdn/'} 
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
<br />>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f90fdeedeb8> <class 'mlmodels.model_keras.armdn.Model'>
<br />
<br />  #### Loading dataset   ############################################# 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
<br />
<br />Epoch 1/10
<br />
<br />1/1 [==============================] - 2s 2s/step - loss: 351382.1250
<br />Epoch 2/10
<br />
<br />1/1 [==============================] - 0s 99ms/step - loss: 185869.5156
<br />Epoch 3/10
<br />
<br />1/1 [==============================] - 0s 101ms/step - loss: 86221.8594
<br />Epoch 4/10
<br />
<br />1/1 [==============================] - 0s 99ms/step - loss: 40535.6250
<br />Epoch 5/10
<br />
<br />1/1 [==============================] - 0s 91ms/step - loss: 21741.8945
<br />Epoch 6/10
<br />
<br />1/1 [==============================] - 0s 96ms/step - loss: 13268.5723
<br />Epoch 7/10
<br />
<br />1/1 [==============================] - 0s 95ms/step - loss: 8920.1621
<br />Epoch 8/10
<br />
<br />1/1 [==============================] - 0s 94ms/step - loss: 6460.2021
<br />Epoch 9/10
<br />
<br />1/1 [==============================] - 0s 93ms/step - loss: 4979.9785
<br />Epoch 10/10
<br />
<br />1/1 [==============================] - 0s 91ms/step - loss: 4043.7019
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />[[ 4.16899808e-02  1.08462782e+01  1.35048847e+01  1.09285793e+01
<br />   1.42438545e+01  1.33449688e+01  1.05183334e+01  1.05492792e+01
<br />   1.11448689e+01  1.03670731e+01  1.24436502e+01  1.36584597e+01
<br />   9.58670330e+00  1.10712500e+01  1.15137110e+01  1.12616673e+01
<br />   1.02870111e+01  1.09561949e+01  1.11660872e+01  9.79597473e+00
<br />   1.06173277e+01  1.26750250e+01  1.15405159e+01  1.19657488e+01
<br />   9.58094311e+00  1.26450281e+01  1.13400736e+01  9.50631142e+00
<br />   8.92709160e+00  1.07038126e+01  1.11585369e+01  1.27083750e+01
<br />   1.05204296e+01  1.16822319e+01  1.05215225e+01  1.27214413e+01
<br />   1.06542435e+01  1.09299736e+01  1.19673872e+01  7.80453348e+00
<br />   1.21495523e+01  1.03679667e+01  1.39951153e+01  1.26492119e+01
<br />   1.00239849e+01  1.37076530e+01  1.20612431e+01  1.05968895e+01
<br />   1.10773830e+01  1.02614822e+01  1.03411636e+01  1.11929865e+01
<br />   1.13720951e+01  1.13367405e+01  1.17916269e+01  1.17366552e+01
<br />   1.02073889e+01  9.19182110e+00  1.17842045e+01  1.27503138e+01
<br />   2.30680037e+00 -2.32591891e+00  1.69095850e+00  1.16982317e+00
<br />   1.06996822e+00  2.95898485e+00  1.36482489e+00  1.21368635e+00
<br />   9.23739001e-02  5.19216895e-01 -2.06112790e+00 -1.11284375e+00
<br />  -5.89695573e-01 -4.17539299e-01 -8.52452219e-01  7.76546955e-01
<br />  -3.53105783e-01  1.36771560e-01  1.85159218e+00 -1.78557408e+00
<br />  -1.29110956e+00 -9.18901324e-01 -8.37078094e-01  2.95307875e+00
<br />  -1.16523683e+00  9.16858733e-01 -2.03746009e+00  4.77505982e-01
<br />  -9.81737018e-01  4.43387687e-01 -1.29815483e+00 -3.49088252e-01
<br />  -1.14581466e+00  1.95785236e+00 -5.39520383e-02 -6.94737673e-01
<br />  -6.15082800e-01  1.23531008e+00  1.31158710e+00 -2.02959824e+00
<br />   1.74116969e+00 -1.37697124e+00  1.38933253e+00 -6.86959267e-01
<br />   1.08642721e+00 -4.07171905e-01  6.51589036e-01  5.19861341e-01
<br />  -1.23625472e-02 -4.00813520e-01 -1.49725652e+00  1.19505072e+00
<br />   1.88563347e+00  4.37552959e-01 -3.69479179e-01  1.11643052e+00
<br />   7.62102067e-01  2.11525410e-01  1.07875995e-01  1.87815651e-01
<br />  -1.47502816e+00  2.74721146e-01 -1.46401322e+00 -2.46811676e+00
<br />  -7.70496726e-01  1.09723949e+00 -1.51994109e+00 -1.14454699e+00
<br />  -1.03741586e+00  1.07595980e+00 -9.39091623e-01  7.65924156e-02
<br />  -1.79015243e+00  4.23542589e-01 -1.12536013e+00  1.55352676e+00
<br />  -1.05480671e+00 -7.13386178e-01  1.27993321e+00  2.19479299e+00
<br />  -6.37848854e-01 -2.15813294e-01  1.10871530e+00  1.67244887e+00
<br />   6.17730677e-01 -2.77651834e+00  1.22056007e+00  1.37897432e-02
<br />   1.02912569e+00 -1.12833846e+00  1.57867360e+00 -7.09626496e-01
<br />  -1.28849566e-01 -2.13623032e-01  6.58020377e-02 -1.72876978e+00
<br />   2.02042866e+00 -1.02351904e+00  1.45678210e+00 -8.41458440e-02
<br />  -3.27042818e-01 -8.87125134e-01  9.86762643e-02 -2.66058147e-02
<br />  -4.24403012e-01  1.45758522e+00 -3.34037781e+00 -1.27089053e-01
<br />   3.69530892e+00  4.52639341e-01  7.80145049e-01  7.35391617e-01
<br />  -1.17161608e+00 -1.99728930e+00  2.44709253e+00 -1.93243289e+00
<br />   5.88753223e-01  1.21840322e+00  1.38016987e+00 -1.86184987e-01
<br />   1.55578434e+00  1.12990026e+01  9.32497597e+00  1.24885807e+01
<br />   1.17744827e+01  1.13759565e+01  1.39107952e+01  1.20989819e+01
<br />   9.07181931e+00  1.14128284e+01  1.24344673e+01  1.02370043e+01
<br />   9.47208691e+00  9.66187477e+00  1.18078747e+01  1.23396978e+01
<br />   1.24667768e+01  1.15475588e+01  1.18485374e+01  8.66812706e+00
<br />   1.04709082e+01  1.23284168e+01  1.15348234e+01  1.08058605e+01
<br />   1.13319645e+01  1.11471548e+01  9.81397629e+00  9.61942482e+00
<br />   1.18541136e+01  1.12099133e+01  1.14211226e+01  1.08528643e+01
<br />   1.08804770e+01  1.11302271e+01  1.01501637e+01  7.13613939e+00
<br />   8.10327816e+00  1.24894934e+01  1.29871616e+01  8.00548077e+00
<br />   1.27156649e+01  1.25421314e+01  1.19615421e+01  9.66070747e+00
<br />   9.02849007e+00  1.26604109e+01  1.03352690e+01  9.99739933e+00
<br />   8.94938469e+00  1.20699358e+01  1.07883282e+01  1.11047115e+01
<br />   1.29354639e+01  1.14638033e+01  1.21131372e+01  1.07762451e+01
<br />   9.91993809e+00  1.26525497e+01  1.38738708e+01  8.34236336e+00
<br />   9.88635719e-01  7.36967921e-02  1.95263147e+00  4.12378168e+00
<br />   4.08609033e-01  3.43823552e-01  1.06179297e-01  3.46448660e-01
<br />   2.57359552e+00  2.04611599e-01  2.16669655e+00  8.10941219e-01
<br />   1.40950441e+00  1.80057859e+00  6.17889166e-02  1.85369086e+00
<br />   4.06373322e-01  8.82572055e-01  1.49548709e-01  1.01872623e-01
<br />   3.10118794e-01  1.86446071e+00  1.83616889e+00  1.85169816e+00
<br />   4.26641464e+00  7.48838067e-01  6.12681150e-01  1.53042793e-01
<br />   8.03608418e-01  3.70528102e-01  1.19610572e+00  1.51018977e-01
<br />   7.24906504e-01  2.75783443e+00  7.24886179e-01  1.98667157e+00
<br />   3.69081438e-01  1.51839757e+00  2.46845937e+00  5.57311654e-01
<br />   1.80683279e+00  1.28766155e+00  1.66906190e+00  2.63773084e-01
<br />   3.64652205e+00  2.32415366e+00  6.85969353e-01  4.39689040e-01
<br />   2.33149576e+00  1.01537359e+00  4.72086191e-01  9.15926099e-02
<br />   1.14739501e+00  9.07108307e-01  1.84570742e+00  1.10653579e+00
<br />   1.00804472e+00  3.53329945e+00  5.50160170e-01  2.59170675e+00
<br />   3.86038005e-01  2.89781904e+00  6.71646833e-01  1.86712182e+00
<br />   2.01169872e+00  3.03031921e-01  4.90428090e-01  3.26550007e-01
<br />   1.44810975e-01  2.72812700e+00  2.90612650e+00  1.26961231e-01
<br />   6.11686945e-01  3.50801563e+00  2.23428488e+00  5.11505246e-01
<br />   2.84086657e+00  1.62708616e+00  2.80445719e+00  9.00381148e-01
<br />   1.08247864e+00  1.37943447e+00  2.64285207e-01  8.60649347e-02
<br />   1.93836284e+00  1.24577928e+00  1.19910109e+00  7.92192876e-01
<br />   7.85824955e-01  2.57206011e+00  4.09979880e-01  6.17701411e-02
<br />   1.71238160e+00  1.53618932e-01  5.62371254e-01  1.61158645e+00
<br />   2.03993702e+00  2.42883742e-01  1.02202177e+00  2.25840998e+00
<br />   4.26075697e-01  2.17155170e+00  1.85849416e+00  7.42139876e-01
<br />   9.13307071e-01  2.37010241e-01  2.31591177e+00  2.12506247e+00
<br />   2.71216869e+00  2.93003678e-01  9.50224876e-01  2.46575379e+00
<br />   4.46472406e-01  1.36901164e+00  1.56982899e+00  1.64465022e+00
<br />   3.65674353e+00  1.48267436e+00  6.14297271e-01  1.26047468e+00
<br />   5.10290861e+00 -1.38166714e+01 -1.35025473e+01]]
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-17 23:50:40.151927
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   91.0376
<br />metric_name                                  mean_absolute_error
<br />Name: 0, dtype: object 
<br />
<br />  date_run                              2020-05-17 23:50:40.156462
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   8318.28
<br />metric_name                                   mean_squared_error
<br />Name: 1, dtype: object 
<br />
<br />  date_run                              2020-05-17 23:50:40.160318
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                    91.526
<br />metric_name                                median_absolute_error
<br />Name: 2, dtype: object 
<br />
<br />  date_run                              2020-05-17 23:50:40.163744
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -743.961
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
<br />  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />| N-Beats
<br />| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140260759780152
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140257918254608
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140257917796424
<br />| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140257917796928
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140257917797432
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140257917797936
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f90fdee4128> <class 'mlmodels.model_tch.nbeats.Model'>
<br />[[0.40504701]
<br /> [0.40695405]
<br /> [0.39710839]
<br /> ...
<br /> [0.93587014]
<br /> [0.95086039]
<br /> [0.95547277]]
<br />--- fiting ---
<br />grad_step = 000000, loss = 0.564838
<br />plot()
<br />Saved image to .//n_beats_0.png.
<br />grad_step = 000001, loss = 0.531568
<br />grad_step = 000002, loss = 0.511128
<br />grad_step = 000003, loss = 0.487336
<br />grad_step = 000004, loss = 0.463216
<br />grad_step = 000005, loss = 0.443453
<br />grad_step = 000006, loss = 0.428290
<br />grad_step = 000007, loss = 0.413212
<br />grad_step = 000008, loss = 0.400529
<br />grad_step = 000009, loss = 0.389527
<br />grad_step = 000010, loss = 0.377512
<br />grad_step = 000011, loss = 0.366070
<br />grad_step = 000012, loss = 0.356305
<br />grad_step = 000013, loss = 0.346076
<br />grad_step = 000014, loss = 0.334388
<br />grad_step = 000015, loss = 0.321157
<br />grad_step = 000016, loss = 0.308476
<br />grad_step = 000017, loss = 0.298105
<br />grad_step = 000018, loss = 0.289194
<br />grad_step = 000019, loss = 0.279217
<br />grad_step = 000020, loss = 0.268373
<br />grad_step = 000021, loss = 0.258437
<br />grad_step = 000022, loss = 0.249377
<br />grad_step = 000023, loss = 0.240121
<br />grad_step = 000024, loss = 0.230467
<br />grad_step = 000025, loss = 0.220878
<br />grad_step = 000026, loss = 0.211798
<br />grad_step = 000027, loss = 0.203121
<br />grad_step = 000028, loss = 0.194374
<br />grad_step = 000029, loss = 0.185643
<br />grad_step = 000030, loss = 0.177464
<br />grad_step = 000031, loss = 0.169669
<br />grad_step = 000032, loss = 0.161807
<br />grad_step = 000033, loss = 0.153974
<br />grad_step = 000034, loss = 0.146397
<br />grad_step = 000035, loss = 0.138997
<br />grad_step = 000036, loss = 0.131800
<br />grad_step = 000037, loss = 0.124867
<br />grad_step = 000038, loss = 0.118126
<br />grad_step = 000039, loss = 0.111579
<br />grad_step = 000040, loss = 0.105237
<br />grad_step = 000041, loss = 0.099121
<br />grad_step = 000042, loss = 0.093209
<br />grad_step = 000043, loss = 0.087454
<br />grad_step = 000044, loss = 0.081963
<br />grad_step = 000045, loss = 0.076677
<br />grad_step = 000046, loss = 0.071538
<br />grad_step = 000047, loss = 0.066715
<br />grad_step = 000048, loss = 0.062129
<br />grad_step = 000049, loss = 0.057728
<br />grad_step = 000050, loss = 0.053588
<br />grad_step = 000051, loss = 0.049617
<br />grad_step = 000052, loss = 0.045876
<br />grad_step = 000053, loss = 0.042425
<br />grad_step = 000054, loss = 0.039167
<br />grad_step = 000055, loss = 0.036083
<br />grad_step = 000056, loss = 0.033218
<br />grad_step = 000057, loss = 0.030589
<br />grad_step = 000058, loss = 0.028125
<br />grad_step = 000059, loss = 0.025833
<br />grad_step = 000060, loss = 0.023748
<br />grad_step = 000061, loss = 0.021823
<br />grad_step = 000062, loss = 0.020038
<br />grad_step = 000063, loss = 0.018421
<br />grad_step = 000064, loss = 0.016942
<br />grad_step = 000065, loss = 0.015572
<br />grad_step = 000066, loss = 0.014336
<br />grad_step = 000067, loss = 0.013219
<br />grad_step = 000068, loss = 0.012175
<br />grad_step = 000069, loss = 0.011238
<br />grad_step = 000070, loss = 0.010387
<br />grad_step = 000071, loss = 0.009612
<br />grad_step = 000072, loss = 0.008905
<br />grad_step = 000073, loss = 0.008256
<br />grad_step = 000074, loss = 0.007661
<br />grad_step = 000075, loss = 0.007129
<br />grad_step = 000076, loss = 0.006637
<br />grad_step = 000077, loss = 0.006191
<br />grad_step = 000078, loss = 0.005786
<br />grad_step = 000079, loss = 0.005413
<br />grad_step = 000080, loss = 0.005073
<br />grad_step = 000081, loss = 0.004767
<br />grad_step = 000082, loss = 0.004487
<br />grad_step = 000083, loss = 0.004234
<br />grad_step = 000084, loss = 0.004004
<br />grad_step = 000085, loss = 0.003792
<br />grad_step = 000086, loss = 0.003605
<br />grad_step = 000087, loss = 0.003433
<br />grad_step = 000088, loss = 0.003281
<br />grad_step = 000089, loss = 0.003142
<br />grad_step = 000090, loss = 0.003016
<br />grad_step = 000091, loss = 0.002905
<br />grad_step = 000092, loss = 0.002804
<br />grad_step = 000093, loss = 0.002715
<br />grad_step = 000094, loss = 0.002636
<br />grad_step = 000095, loss = 0.002564
<br />grad_step = 000096, loss = 0.002501
<br />grad_step = 000097, loss = 0.002444
<br />grad_step = 000098, loss = 0.002395
<br />grad_step = 000099, loss = 0.002350
<br />grad_step = 000100, loss = 0.002311
<br />plot()
<br />Saved image to .//n_beats_100.png.
<br />grad_step = 000101, loss = 0.002276
<br />grad_step = 000102, loss = 0.002245
<br />grad_step = 000103, loss = 0.002218
<br />grad_step = 000104, loss = 0.002194
<br />grad_step = 000105, loss = 0.002172
<br />grad_step = 000106, loss = 0.002153
<br />grad_step = 000107, loss = 0.002136
<br />grad_step = 000108, loss = 0.002120
<br />grad_step = 000109, loss = 0.002107
<br />grad_step = 000110, loss = 0.002094
<br />grad_step = 000111, loss = 0.002083
<br />grad_step = 000112, loss = 0.002072
<br />grad_step = 000113, loss = 0.002063
<br />grad_step = 000114, loss = 0.002054
<br />grad_step = 000115, loss = 0.002046
<br />grad_step = 000116, loss = 0.002038
<br />grad_step = 000117, loss = 0.002031
<br />grad_step = 000118, loss = 0.002024
<br />grad_step = 000119, loss = 0.002017
<br />grad_step = 000120, loss = 0.002010
<br />grad_step = 000121, loss = 0.002004
<br />grad_step = 000122, loss = 0.001998
<br />grad_step = 000123, loss = 0.001992
<br />grad_step = 000124, loss = 0.001987
<br />grad_step = 000125, loss = 0.001986
<br />grad_step = 000126, loss = 0.001998
<br />grad_step = 000127, loss = 0.002039
<br />grad_step = 000128, loss = 0.002108
<br />grad_step = 000129, loss = 0.002116
<br />grad_step = 000130, loss = 0.002048
<br />grad_step = 000131, loss = 0.001970
<br />grad_step = 000132, loss = 0.001963
<br />grad_step = 000133, loss = 0.002018
<br />grad_step = 000134, loss = 0.002036
<br />grad_step = 000135, loss = 0.001972
<br />grad_step = 000136, loss = 0.001931
<br />grad_step = 000137, loss = 0.001959
<br />grad_step = 000138, loss = 0.001982
<br />grad_step = 000139, loss = 0.001961
<br />grad_step = 000140, loss = 0.001932
<br />grad_step = 000141, loss = 0.001920
<br />grad_step = 000142, loss = 0.001930
<br />grad_step = 000143, loss = 0.001947
<br />grad_step = 000144, loss = 0.001934
<br />grad_step = 000145, loss = 0.001906
<br />grad_step = 000146, loss = 0.001896
<br />grad_step = 000147, loss = 0.001909
<br />grad_step = 000148, loss = 0.001917
<br />grad_step = 000149, loss = 0.001906
<br />grad_step = 000150, loss = 0.001893
<br />grad_step = 000151, loss = 0.001882
<br />grad_step = 000152, loss = 0.001877
<br />grad_step = 000153, loss = 0.001882
<br />grad_step = 000154, loss = 0.001888
<br />grad_step = 000155, loss = 0.001890
<br />grad_step = 000156, loss = 0.001886
<br />grad_step = 000157, loss = 0.001878
<br />grad_step = 000158, loss = 0.001868
<br />grad_step = 000159, loss = 0.001862
<br />grad_step = 000160, loss = 0.001856
<br />grad_step = 000161, loss = 0.001849
<br />grad_step = 000162, loss = 0.001843
<br />grad_step = 000163, loss = 0.001839
<br />grad_step = 000164, loss = 0.001838
<br />grad_step = 000165, loss = 0.001837
<br />grad_step = 000166, loss = 0.001836
<br />grad_step = 000167, loss = 0.001838
<br />grad_step = 000168, loss = 0.001846
<br />grad_step = 000169, loss = 0.001866
<br />grad_step = 000170, loss = 0.001915
<br />grad_step = 000171, loss = 0.001977
<br />grad_step = 000172, loss = 0.002064
<br />grad_step = 000173, loss = 0.002043
<br />grad_step = 000174, loss = 0.001941
<br />grad_step = 000175, loss = 0.001839
<br />grad_step = 000176, loss = 0.001831
<br />grad_step = 000177, loss = 0.001892
<br />grad_step = 000178, loss = 0.001922
<br />grad_step = 000179, loss = 0.001886
<br />grad_step = 000180, loss = 0.001821
<br />grad_step = 000181, loss = 0.001797
<br />grad_step = 000182, loss = 0.001833
<br />grad_step = 000183, loss = 0.001875
<br />grad_step = 000184, loss = 0.001872
<br />grad_step = 000185, loss = 0.001824
<br />grad_step = 000186, loss = 0.001792
<br />grad_step = 000187, loss = 0.001788
<br />grad_step = 000188, loss = 0.001792
<br />grad_step = 000189, loss = 0.001787
<br />grad_step = 000190, loss = 0.001791
<br />grad_step = 000191, loss = 0.001811
<br />grad_step = 000192, loss = 0.001819
<br />grad_step = 000193, loss = 0.001806
<br />grad_step = 000194, loss = 0.001777
<br />grad_step = 000195, loss = 0.001761
<br />grad_step = 000196, loss = 0.001757
<br />grad_step = 000197, loss = 0.001754
<br />grad_step = 000198, loss = 0.001751
<br />grad_step = 000199, loss = 0.001755
<br />grad_step = 000200, loss = 0.001766
<br />plot()
<br />Saved image to .//n_beats_200.png.
<br />grad_step = 000201, loss = 0.001774
<br />grad_step = 000202, loss = 0.001776
<br />grad_step = 000203, loss = 0.001775
<br />grad_step = 000204, loss = 0.001783
<br />grad_step = 000205, loss = 0.001791
<br />grad_step = 000206, loss = 0.001802
<br />grad_step = 000207, loss = 0.001797
<br />grad_step = 000208, loss = 0.001788
<br />grad_step = 000209, loss = 0.001766
<br />grad_step = 000210, loss = 0.001746
<br />grad_step = 000211, loss = 0.001726
<br />grad_step = 000212, loss = 0.001711
<br />grad_step = 000213, loss = 0.001707
<br />grad_step = 000214, loss = 0.001711
<br />grad_step = 000215, loss = 0.001722
<br />grad_step = 000216, loss = 0.001737
<br />grad_step = 000217, loss = 0.001758
<br />grad_step = 000218, loss = 0.001784
<br />grad_step = 000219, loss = 0.001830
<br />grad_step = 000220, loss = 0.001869
<br />grad_step = 000221, loss = 0.001913
<br />grad_step = 000222, loss = 0.001876
<br />grad_step = 000223, loss = 0.001798
<br />grad_step = 000224, loss = 0.001710
<br />grad_step = 000225, loss = 0.001686
<br />grad_step = 000226, loss = 0.001725
<br />grad_step = 000227, loss = 0.001771
<br />grad_step = 000228, loss = 0.001782
<br />grad_step = 000229, loss = 0.001736
<br />grad_step = 000230, loss = 0.001687
<br />grad_step = 000231, loss = 0.001672
<br />grad_step = 000232, loss = 0.001693
<br />grad_step = 000233, loss = 0.001725
<br />grad_step = 000234, loss = 0.001739
<br />grad_step = 000235, loss = 0.001739
<br />grad_step = 000236, loss = 0.001713
<br />grad_step = 000237, loss = 0.001687
<br />grad_step = 000238, loss = 0.001665
<br />grad_step = 000239, loss = 0.001656
<br />grad_step = 000240, loss = 0.001659
<br />grad_step = 000241, loss = 0.001670
<br />grad_step = 000242, loss = 0.001683
<br />grad_step = 000243, loss = 0.001692
<br />grad_step = 000244, loss = 0.001699
<br />grad_step = 000245, loss = 0.001697
<br />grad_step = 000246, loss = 0.001691
<br />grad_step = 000247, loss = 0.001677
<br />grad_step = 000248, loss = 0.001663
<br />grad_step = 000249, loss = 0.001648
<br />grad_step = 000250, loss = 0.001639
<br />grad_step = 000251, loss = 0.001635
<br />grad_step = 000252, loss = 0.001636
<br />grad_step = 000253, loss = 0.001640
<br />grad_step = 000254, loss = 0.001647
<br />grad_step = 000255, loss = 0.001658
<br />grad_step = 000256, loss = 0.001672
<br />grad_step = 000257, loss = 0.001697
<br />grad_step = 000258, loss = 0.001728
<br />grad_step = 000259, loss = 0.001779
<br />grad_step = 000260, loss = 0.001813
<br />grad_step = 000261, loss = 0.001840
<br />grad_step = 000262, loss = 0.001792
<br />grad_step = 000263, loss = 0.001713
<br />grad_step = 000264, loss = 0.001638
<br />grad_step = 000265, loss = 0.001623
<br />grad_step = 000266, loss = 0.001661
<br />grad_step = 000267, loss = 0.001701
<br />grad_step = 000268, loss = 0.001708
<br />grad_step = 000269, loss = 0.001671
<br />grad_step = 000270, loss = 0.001628
<br />grad_step = 000271, loss = 0.001605
<br />grad_step = 000272, loss = 0.001612
<br />grad_step = 000273, loss = 0.001637
<br />grad_step = 000274, loss = 0.001664
<br />grad_step = 000275, loss = 0.001689
<br />grad_step = 000276, loss = 0.001692
<br />grad_step = 000277, loss = 0.001682
<br />grad_step = 000278, loss = 0.001653
<br />grad_step = 000279, loss = 0.001622
<br />grad_step = 000280, loss = 0.001598
<br />grad_step = 000281, loss = 0.001588
<br />grad_step = 000282, loss = 0.001592
<br />grad_step = 000283, loss = 0.001603
<br />grad_step = 000284, loss = 0.001617
<br />grad_step = 000285, loss = 0.001626
<br />grad_step = 000286, loss = 0.001632
<br />grad_step = 000287, loss = 0.001628
<br />grad_step = 000288, loss = 0.001621
<br />grad_step = 000289, loss = 0.001608
<br />grad_step = 000290, loss = 0.001595
<br />grad_step = 000291, loss = 0.001584
<br />grad_step = 000292, loss = 0.001575
<br />grad_step = 000293, loss = 0.001570
<br />grad_step = 000294, loss = 0.001567
<br />grad_step = 000295, loss = 0.001566
<br />grad_step = 000296, loss = 0.001567
<br />grad_step = 000297, loss = 0.001569
<br />grad_step = 000298, loss = 0.001574
<br />grad_step = 000299, loss = 0.001584
<br />grad_step = 000300, loss = 0.001602
<br />plot()
<br />Saved image to .//n_beats_300.png.
<br />grad_step = 000301, loss = 0.001638
<br />grad_step = 000302, loss = 0.001697
<br />grad_step = 000303, loss = 0.001801
<br />grad_step = 000304, loss = 0.001920
<br />grad_step = 000305, loss = 0.002040
<br />grad_step = 000306, loss = 0.001978
<br />grad_step = 000307, loss = 0.001778
<br />grad_step = 000308, loss = 0.001586
<br />grad_step = 000309, loss = 0.001597
<br />grad_step = 000310, loss = 0.001733
<br />grad_step = 000311, loss = 0.001765
<br />grad_step = 000312, loss = 0.001652
<br />grad_step = 000313, loss = 0.001552
<br />grad_step = 000314, loss = 0.001594
<br />grad_step = 000315, loss = 0.001686
<br />grad_step = 000316, loss = 0.001678
<br />grad_step = 000317, loss = 0.001595
<br />grad_step = 000318, loss = 0.001534
<br />grad_step = 000319, loss = 0.001551
<br />grad_step = 000320, loss = 0.001606
<br />grad_step = 000321, loss = 0.001621
<br />grad_step = 000322, loss = 0.001585
<br />grad_step = 000323, loss = 0.001536
<br />grad_step = 000324, loss = 0.001523
<br />grad_step = 000325, loss = 0.001549
<br />grad_step = 000326, loss = 0.001573
<br />grad_step = 000327, loss = 0.001569
<br />grad_step = 000328, loss = 0.001540
<br />grad_step = 000329, loss = 0.001517
<br />grad_step = 000330, loss = 0.001516
<br />grad_step = 000331, loss = 0.001531
<br />grad_step = 000332, loss = 0.001542
<br />grad_step = 000333, loss = 0.001536
<br />grad_step = 000334, loss = 0.001520
<br />grad_step = 000335, loss = 0.001506
<br />grad_step = 000336, loss = 0.001504
<br />grad_step = 000337, loss = 0.001511
<br />grad_step = 000338, loss = 0.001517
<br />grad_step = 000339, loss = 0.001517
<br />grad_step = 000340, loss = 0.001511
<br />grad_step = 000341, loss = 0.001501
<br />grad_step = 000342, loss = 0.001495
<br />grad_step = 000343, loss = 0.001493
<br />grad_step = 000344, loss = 0.001494
<br />grad_step = 000345, loss = 0.001496
<br />grad_step = 000346, loss = 0.001498
<br />grad_step = 000347, loss = 0.001496
<br />grad_step = 000348, loss = 0.001493
<br />grad_step = 000349, loss = 0.001488
<br />grad_step = 000350, loss = 0.001484
<br />grad_step = 000351, loss = 0.001481
<br />grad_step = 000352, loss = 0.001478
<br />grad_step = 000353, loss = 0.001477
<br />grad_step = 000354, loss = 0.001477
<br />grad_step = 000355, loss = 0.001478
<br />grad_step = 000356, loss = 0.001480
<br />grad_step = 000357, loss = 0.001482
<br />grad_step = 000358, loss = 0.001487
<br />grad_step = 000359, loss = 0.001493
<br />grad_step = 000360, loss = 0.001506
<br />grad_step = 000361, loss = 0.001524
<br />grad_step = 000362, loss = 0.001565
<br />grad_step = 000363, loss = 0.001633
<br />grad_step = 000364, loss = 0.001760
<br />grad_step = 000365, loss = 0.001933
<br />grad_step = 000366, loss = 0.002069
<br />grad_step = 000367, loss = 0.001992
<br />grad_step = 000368, loss = 0.001704
<br />grad_step = 000369, loss = 0.001485
<br />grad_step = 000370, loss = 0.001577
<br />grad_step = 000371, loss = 0.001762
<br />grad_step = 000372, loss = 0.001699
<br />grad_step = 000373, loss = 0.001502
<br />grad_step = 000374, loss = 0.001482
<br />grad_step = 000375, loss = 0.001617
<br />grad_step = 000376, loss = 0.001657
<br />grad_step = 000377, loss = 0.001526
<br />grad_step = 000378, loss = 0.001447
<br />grad_step = 000379, loss = 0.001509
<br />grad_step = 000380, loss = 0.001578
<br />grad_step = 000381, loss = 0.001549
<br />grad_step = 000382, loss = 0.001462
<br />grad_step = 000383, loss = 0.001447
<br />grad_step = 000384, loss = 0.001501
<br />grad_step = 000385, loss = 0.001518
<br />grad_step = 000386, loss = 0.001473
<br />grad_step = 000387, loss = 0.001434
<br />grad_step = 000388, loss = 0.001452
<br />grad_step = 000389, loss = 0.001483
<br />grad_step = 000390, loss = 0.001470
<br />grad_step = 000391, loss = 0.001436
<br />grad_step = 000392, loss = 0.001427
<br />grad_step = 000393, loss = 0.001446
<br />grad_step = 000394, loss = 0.001456
<br />grad_step = 000395, loss = 0.001438
<br />grad_step = 000396, loss = 0.001420
<br />grad_step = 000397, loss = 0.001421
<br />grad_step = 000398, loss = 0.001432
<br />grad_step = 000399, loss = 0.001433
<br />grad_step = 000400, loss = 0.001421
<br />plot()
<br />Saved image to .//n_beats_400.png.
<br />grad_step = 000401, loss = 0.001411
<br />grad_step = 000402, loss = 0.001411
<br />grad_step = 000403, loss = 0.001417
<br />grad_step = 000404, loss = 0.001416
<br />grad_step = 000405, loss = 0.001409
<br />grad_step = 000406, loss = 0.001402
<br />grad_step = 000407, loss = 0.001401
<br />grad_step = 000408, loss = 0.001403
<br />grad_step = 000409, loss = 0.001403
<br />grad_step = 000410, loss = 0.001399
<br />grad_step = 000411, loss = 0.001394
<br />grad_step = 000412, loss = 0.001392
<br />grad_step = 000413, loss = 0.001392
<br />grad_step = 000414, loss = 0.001392
<br />grad_step = 000415, loss = 0.001390
<br />grad_step = 000416, loss = 0.001385
<br />grad_step = 000417, loss = 0.001382
<br />grad_step = 000418, loss = 0.001380
<br />grad_step = 000419, loss = 0.001380
<br />grad_step = 000420, loss = 0.001380
<br />grad_step = 000421, loss = 0.001377
<br />grad_step = 000422, loss = 0.001374
<br />grad_step = 000423, loss = 0.001371
<br />grad_step = 000424, loss = 0.001369
<br />grad_step = 000425, loss = 0.001368
<br />grad_step = 000426, loss = 0.001367
<br />grad_step = 000427, loss = 0.001365
<br />grad_step = 000428, loss = 0.001363
<br />grad_step = 000429, loss = 0.001360
<br />grad_step = 000430, loss = 0.001358
<br />grad_step = 000431, loss = 0.001356
<br />grad_step = 000432, loss = 0.001356
<br />grad_step = 000433, loss = 0.001357
<br />grad_step = 000434, loss = 0.001361
<br />grad_step = 000435, loss = 0.001371
<br />grad_step = 000436, loss = 0.001392
<br />grad_step = 000437, loss = 0.001434
<br />grad_step = 000438, loss = 0.001509
<br />grad_step = 000439, loss = 0.001628
<br />grad_step = 000440, loss = 0.001733
<br />grad_step = 000441, loss = 0.001772
<br />grad_step = 000442, loss = 0.001655
<br />grad_step = 000443, loss = 0.001452
<br />grad_step = 000444, loss = 0.001358
<br />grad_step = 000445, loss = 0.001426
<br />grad_step = 000446, loss = 0.001510
<br />grad_step = 000447, loss = 0.001484
<br />grad_step = 000448, loss = 0.001373
<br />grad_step = 000449, loss = 0.001342
<br />grad_step = 000450, loss = 0.001411
<br />grad_step = 000451, loss = 0.001454
<br />grad_step = 000452, loss = 0.001427
<br />grad_step = 000453, loss = 0.001359
<br />grad_step = 000454, loss = 0.001325
<br />grad_step = 000455, loss = 0.001339
<br />grad_step = 000456, loss = 0.001372
<br />grad_step = 000457, loss = 0.001386
<br />grad_step = 000458, loss = 0.001359
<br />grad_step = 000459, loss = 0.001318
<br />grad_step = 000460, loss = 0.001308
<br />grad_step = 000461, loss = 0.001329
<br />grad_step = 000462, loss = 0.001343
<br />grad_step = 000463, loss = 0.001334
<br />grad_step = 000464, loss = 0.001313
<br />grad_step = 000465, loss = 0.001302
<br />grad_step = 000466, loss = 0.001303
<br />grad_step = 000467, loss = 0.001308
<br />grad_step = 000468, loss = 0.001311
<br />grad_step = 000469, loss = 0.001307
<br />grad_step = 000470, loss = 0.001296
<br />grad_step = 000471, loss = 0.001287
<br />grad_step = 000472, loss = 0.001285
<br />grad_step = 000473, loss = 0.001289
<br />grad_step = 000474, loss = 0.001293
<br />grad_step = 000475, loss = 0.001289
<br />grad_step = 000476, loss = 0.001280
<br />grad_step = 000477, loss = 0.001273
<br />grad_step = 000478, loss = 0.001270
<br />grad_step = 000479, loss = 0.001272
<br />grad_step = 000480, loss = 0.001273
<br />grad_step = 000481, loss = 0.001272
<br />grad_step = 000482, loss = 0.001267
<br />grad_step = 000483, loss = 0.001262
<br />grad_step = 000484, loss = 0.001258
<br />grad_step = 000485, loss = 0.001256
<br />grad_step = 000486, loss = 0.001254
<br />grad_step = 000487, loss = 0.001254
<br />grad_step = 000488, loss = 0.001252
<br />grad_step = 000489, loss = 0.001251
<br />grad_step = 000490, loss = 0.001248
<br />grad_step = 000491, loss = 0.001245
<br />grad_step = 000492, loss = 0.001242
<br />grad_step = 000493, loss = 0.001239
<br />grad_step = 000494, loss = 0.001236
<br />grad_step = 000495, loss = 0.001234
<br />grad_step = 000496, loss = 0.001232
<br />grad_step = 000497, loss = 0.001230
<br />grad_step = 000498, loss = 0.001228
<br />grad_step = 000499, loss = 0.001226
<br />grad_step = 000500, loss = 0.001224
<br />plot()
<br />Saved image to .//n_beats_500.png.
<br />grad_step = 000501, loss = 0.001222
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
<br />  date_run                              2020-05-17 23:50:58.989094
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.235598
<br />metric_name                                  mean_absolute_error
<br />Name: 4, dtype: object 
<br />
<br />  date_run                              2020-05-17 23:50:58.995273
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.144486
<br />metric_name                                   mean_squared_error
<br />Name: 5, dtype: object 
<br />
<br />  date_run                              2020-05-17 23:50:59.003350
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.131406
<br />metric_name                                median_absolute_error
<br />Name: 6, dtype: object 
<br />
<br />  date_run                              2020-05-17 23:50:59.008271
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -1.19551
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
<br />  data_pars out_pars {'train_data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'} {'outpath': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_fb/fb_prophet/'} 
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
<br />>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f90575fce10> <class 'mlmodels.model_gluon.fb_prophet.Model'>
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-17 23:51:15.032493
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.3339
<br />metric_name                                  mean_absolute_error
<br />Name: 8, dtype: object 
<br />
<br />  date_run                              2020-05-17 23:51:15.035753
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   215.367
<br />metric_name                                   mean_squared_error
<br />Name: 9, dtype: object 
<br />
<br />  date_run                              2020-05-17 23:51:15.038941
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.4309
<br />metric_name                                median_absolute_error
<br />Name: 10, dtype: object 
<br />
<br />  date_run                              2020-05-17 23:51:15.043656
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
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} 'model_uri' 
<br />
<br />  benchmark file saved at https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/example/benchmark/ 
<br />
<br />                        date_run  ...            metric_name
<br />0   2020-05-17 23:50:40.151927  ...    mean_absolute_error
<br />1   2020-05-17 23:50:40.156462  ...     mean_squared_error
<br />2   2020-05-17 23:50:40.160318  ...  median_absolute_error
<br />3   2020-05-17 23:50:40.163744  ...               r2_score
<br />4   2020-05-17 23:50:58.989094  ...    mean_absolute_error
<br />5   2020-05-17 23:50:58.995273  ...     mean_squared_error
<br />6   2020-05-17 23:50:59.003350  ...  median_absolute_error
<br />7   2020-05-17 23:50:59.008271  ...               r2_score
<br />8   2020-05-17 23:51:15.032493  ...    mean_absolute_error
<br />9   2020-05-17 23:51:15.035753  ...     mean_squared_error
<br />10  2020-05-17 23:51:15.038941  ...  median_absolute_error
<br />11  2020-05-17 23:51:15.043656  ...               r2_score
<br />
<br />[12 rows x 6 columns] 



### Error 14, [Traceback at line 3363](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L3363)<br />3363..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/203a72830f23a80c3dd3ee4f0d2ce62ae396cb03/mlmodels/benchmark.py", line 118, in benchmark_run
<br />    model_uri =  model_pars['model_uri']
<br />KeyError: 'model_uri'
