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
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/models.py", line 534, in main
<br />    predict_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/models.py", line 445, in predict_cli
<br />    model, session = load(module, load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/models.py", line 156, in load
<br />    return module.load(load_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/model_tf/1_lstm.py", line 199, in load
<br />    return load_tf(load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/util.py", line 477, in load_tf
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
<br />  <module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/model_tf/1_lstm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.model_tf.1_lstm.Model object at 0x7f6d3c1d50f0> 
<br />
<br />  #### Fit   ######################################################## 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br /> [ 0.17978428 -0.05665762  0.0203147  -0.14453013  0.14176583  0.0812619 ]
<br /> [ 0.07022418  0.046494    0.06529775  0.088366    0.02497934 -0.09640499]
<br /> [ 0.0797433  -0.01239584  0.08903716  0.01959886  0.2715475   0.03006474]
<br /> [ 0.54544979  0.07354705  0.1579908   0.33086652 -0.09356557 -0.07378793]
<br /> [ 0.52547753  0.07371877 -0.07873009 -0.26330516  0.24410746 -0.03754042]
<br /> [ 0.08605129 -0.4304916  -0.3460643  -0.48961118  0.13879535  0.00607062]
<br /> [ 0.1498775   0.0730194  -0.88408184 -0.11237642  0.23210001 -0.47147205]
<br /> [ 0.08969419  0.09647042  0.26332277 -0.23308998 -0.65847504 -0.44515252]
<br /> [ 0.          0.          0.          0.          0.          0.        ]]
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />model_tf.1_lstm
<br />model_tf.1_lstm
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.48894862830638885, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-23 00:36:25.782104: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/model'}
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
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.6432254984974861, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-23 00:36:26.875917: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/model'}
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
<br />  <module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/example/custom_model/1_lstm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.example.custom_model.1_lstm.Model object at 0x7f4f490d91d0> 
<br />
<br />  #### Fit   ######################################################## 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br /> [ 0.07169047  0.07349469 -0.13811514  0.05235035 -0.17422993 -0.02787352]
<br /> [ 0.01132785  0.23125783  0.01689304  0.21431878 -0.02483151 -0.00682053]
<br /> [ 0.0678293   0.08505049 -0.07820809 -0.03827773 -0.2119603  -0.12226772]
<br /> [-0.19526628  0.07591586 -0.08622025 -0.06182463  0.09147284  0.06884826]
<br /> [-0.3618485  -0.23565698  0.35022935  0.54532677 -0.26632774  0.26583442]
<br /> [-0.29658478 -0.00416166  0.47266817  0.02119925 -0.72382462  0.56355828]
<br /> [ 0.06152476  0.31829262 -0.5520522   0.44027469 -0.23268357  0.48522592]
<br /> [ 0.10530004 -0.166632   -0.17561701 -0.09029378  0.25685814  0.08460142]
<br /> [ 0.          0.          0.          0.          0.          0.        ]]
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />example/custom_model/1_lstm.py
<br />example.custom_model.1_lstm.py
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/example/custom_model/1_lstm.py'>
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/example/custom_model/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.4578009620308876, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />2020-05-23 00:36:31.637136: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />{'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/model'}
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
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/example/custom_model/1_lstm.py'>
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/example/custom_model/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.4957592152059078, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />2020-05-23 00:36:32.740410: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />{'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_tf/1_lstm/model'}
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



### Error 5, [Traceback at line 949](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L949)<br />949..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/optim.py", line 388, in main
<br />    optim_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/optim.py", line 259, in optim_cli
<br />    out_pars        = out_pars )
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/optim.py", line 54, in optim
<br />    if hypermodel_pars["engine_pars"]['engine'] == "optuna":
<br />KeyError: 'engine_pars'



### Error 6, [Traceback at line 2125](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2125)<br />2125..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 7, [Traceback at line 2160](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2160)<br />2160..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 8, [Traceback at line 2200](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2200)<br />2200..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 9, [Traceback at line 2235](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2235)<br />2235..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 10, [Traceback at line 2280](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2280)<br />2280..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 11, [Traceback at line 2315](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2315)<br />2315..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 12, [Traceback at line 2372](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2372)<br />2372..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />TypeError: 'Model' object is not iterable



### Error 13, [Traceback at line 2376](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2376)<br />2376..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
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
<br />  json_path https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/json/benchmark_timeseries/test01/ 
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
<br />  data_pars out_pars {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60} {'outpath': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_keras/armdn/'} 
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
<br />>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f31322a5f60> <class 'mlmodels.model_keras.armdn.Model'>
<br />
<br />  #### Loading dataset   ############################################# 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
<br />
<br />Epoch 1/10
<br />
<br />1/1 [==============================] - 2s 2s/step - loss: 357033.7500
<br />Epoch 2/10
<br />
<br />1/1 [==============================] - 0s 95ms/step - loss: 303772.8750
<br />Epoch 3/10
<br />
<br />1/1 [==============================] - 0s 92ms/step - loss: 199017.2812
<br />Epoch 4/10
<br />
<br />1/1 [==============================] - 0s 106ms/step - loss: 129872.7266
<br />Epoch 5/10
<br />
<br />1/1 [==============================] - 0s 117ms/step - loss: 80810.2656
<br />Epoch 6/10
<br />
<br />1/1 [==============================] - 0s 109ms/step - loss: 50032.3672
<br />Epoch 7/10
<br />
<br />1/1 [==============================] - 0s 100ms/step - loss: 32113.5957
<br />Epoch 8/10
<br />
<br />1/1 [==============================] - 0s 98ms/step - loss: 21773.4375
<br />Epoch 9/10
<br />
<br />1/1 [==============================] - 0s 95ms/step - loss: 15541.7090
<br />Epoch 10/10
<br />
<br />1/1 [==============================] - 0s 95ms/step - loss: 11630.6562
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />[[ 3.8706616e-02  5.4731102e+00  4.2977028e+00  6.1461973e+00
<br />   5.5670767e+00  4.9274106e+00  5.4769998e+00  5.6830587e+00
<br />   4.6770077e+00  5.1229553e+00  4.7524052e+00  4.3814106e+00
<br />   5.1050444e+00  5.3672714e+00  5.3672452e+00  5.8264313e+00
<br />   6.5246005e+00  5.1949034e+00  7.6934772e+00  4.6783462e+00
<br />   4.9720697e+00  6.4240670e+00  4.5990734e+00  4.8567762e+00
<br />   5.2937808e+00  4.8389587e+00  5.9450655e+00  6.8785758e+00
<br />   5.6915674e+00  5.2408123e+00  5.0094657e+00  6.2905045e+00
<br />   5.2062578e+00  5.7998400e+00  5.9028139e+00  5.7538633e+00
<br />   5.4057474e+00  7.0064812e+00  5.7333579e+00  5.7169151e+00
<br />   7.2257185e+00  6.5329204e+00  5.0014024e+00  5.2540221e+00
<br />   5.6448469e+00  7.0561004e+00  5.3276191e+00  4.6717086e+00
<br />   5.8638792e+00  5.5188670e+00  7.6569228e+00  6.4298296e+00
<br />   7.0859270e+00  4.0689306e+00  6.1325474e+00  6.0986414e+00
<br />   5.8841968e+00  6.7033348e+00  5.4271369e+00  6.0548239e+00
<br />   4.3486339e-01 -8.8924885e-01 -9.7643536e-01  3.1062976e-01
<br />  -3.6756015e-01  8.5079670e-03 -3.7589842e-01 -7.2100264e-01
<br />   2.5757718e-01  5.2855825e-01  4.0067938e-01  6.8742847e-01
<br />  -1.1383921e-02 -1.8785498e+00 -5.5961430e-02 -2.4400970e-01
<br />  -1.2305063e+00 -2.2537497e-01 -4.2403534e-01  1.7545187e-01
<br />  -5.5635277e-02  1.3644996e+00  9.0750933e-01 -5.3901756e-01
<br />  -6.0082233e-01  4.3186837e-01  5.6961334e-01  6.6153896e-01
<br />  -6.1321545e-01 -1.8610194e-01  6.0355639e-01  5.7476121e-01
<br />   8.1164315e-02  7.4634629e-01 -1.1961377e+00  9.5778555e-01
<br />   1.7448723e-01 -6.3004494e-01 -2.3972198e-01 -1.9899562e-02
<br />  -5.3089142e-01 -5.0962830e-01 -9.9306178e-01  1.1058785e+00
<br />   1.0488698e+00 -6.2134564e-01  3.5110700e-01 -6.4457285e-01
<br />   2.9543570e-01 -9.9800324e-01 -1.6332281e+00  7.7406090e-01
<br />   2.3148663e-01 -1.6892296e-01  3.1466329e-01 -3.7039092e-01
<br />  -6.3041681e-01 -5.6957203e-01  8.6448473e-01 -5.5039710e-01
<br />   4.3487296e-01 -7.8814882e-01 -2.5985432e-01 -9.0934014e-01
<br />  -1.0042347e+00  5.6457186e-01 -4.7314328e-01 -7.5033212e-01
<br />  -5.0592268e-01  9.4171083e-01 -7.4388534e-01 -3.1305516e-01
<br />  -1.8418385e-01  2.9535174e-01  4.3736747e-01 -8.5811514e-01
<br />   5.4916704e-01  7.5911008e-02 -1.8933949e-01 -5.8919609e-01
<br />  -1.0559007e+00 -1.2849137e+00 -7.4297124e-01 -9.7161037e-01
<br />  -4.7257822e-02  4.8729414e-01 -1.3305394e-01  1.1494953e-02
<br />   5.9980261e-01 -3.2966435e-03  3.7537046e-02 -1.9906500e-01
<br />  -1.7390558e-01 -1.1803784e+00  3.3713835e-01  3.2403582e-01
<br />   2.2946879e-01  8.8909984e-01 -7.8271276e-01  9.2032775e-03
<br />  -9.1003813e-02  8.9001232e-01 -4.9999416e-01 -2.8641364e-01
<br />  -6.8264282e-01  3.1365138e-01  5.8321661e-01  1.4941537e-01
<br />   4.2985094e-01 -8.8765490e-01 -7.2244650e-01  7.5023383e-01
<br />   6.9263679e-01 -4.0524098e-01  9.2781955e-01 -3.5216326e-01
<br />   6.4570713e-01  1.5880784e-01 -2.3579475e-01 -2.0095231e-01
<br />   5.1828027e-02  6.2216773e+00  6.0923042e+00  6.2649045e+00
<br />   6.2806854e+00  6.9788494e+00  7.3087397e+00  6.1651039e+00
<br />   6.4903626e+00  5.0367646e+00  6.8180180e+00  6.8433042e+00
<br />   6.4120765e+00  6.3121495e+00  6.6870127e+00  5.3002763e+00
<br />   6.6432352e+00  6.4377489e+00  6.8099556e+00  7.5005164e+00
<br />   6.0327563e+00  5.7383513e+00  5.5214319e+00  7.2877545e+00
<br />   6.2220712e+00  6.9271779e+00  5.8754201e+00  6.5891380e+00
<br />   6.2093167e+00  5.0562820e+00  6.2023473e+00  6.2324238e+00
<br />   7.2964649e+00  5.8021741e+00  6.5086775e+00  5.9588594e+00
<br />   5.9633389e+00  6.4812703e+00  6.6105099e+00  6.4899907e+00
<br />   5.9455099e+00  6.7920408e+00  5.0072412e+00  6.6443744e+00
<br />   5.5020175e+00  6.1872540e+00  6.6629047e+00  6.2329402e+00
<br />   7.2081985e+00  7.2238755e+00  6.2865672e+00  6.5863657e+00
<br />   6.0857472e+00  7.1725922e+00  5.6447759e+00  5.6613417e+00
<br />   6.3934164e+00  5.7149763e+00  6.6782932e+00  5.8926115e+00
<br />   1.3747224e+00  9.6266878e-01  7.6441455e-01  1.1394544e+00
<br />   5.2449816e-01  1.0645413e+00  1.2134386e+00  7.3213536e-01
<br />   9.5869023e-01  9.1843224e-01  1.4379089e+00  1.8346559e+00
<br />   3.2795423e-01  9.3306720e-01  9.4459343e-01  6.5852416e-01
<br />   2.0075827e+00  6.9200587e-01  1.0836370e+00  1.0580320e+00
<br />   1.6163195e+00  1.3158131e+00  8.6708081e-01  2.0120912e+00
<br />   1.0444704e+00  3.1112868e-01  9.5647562e-01  1.5277247e+00
<br />   1.7411436e+00  1.4918755e+00  1.7148317e+00  1.8842316e+00
<br />   1.2871715e+00  1.5417454e+00  2.8836340e-01  2.2028203e+00
<br />   2.1099176e+00  4.2675006e-01  4.3459135e-01  1.4663172e+00
<br />   1.8037291e+00  2.4337220e+00  3.0725026e-01  2.1101201e+00
<br />   7.1023560e-01  1.0737278e+00  4.2379189e-01  1.7202144e+00
<br />   1.7529851e-01  1.3419757e+00  3.7009323e-01  7.5587988e-01
<br />   1.4682513e+00  7.0108747e-01  2.0522912e+00  1.7601855e+00
<br />   4.6813029e-01  5.0663102e-01  5.7884640e-01  9.7457099e-01
<br />   8.0654532e-01  1.0733378e+00  9.9693578e-01  8.0734622e-01
<br />   2.1907144e+00  2.0103610e-01  8.3352447e-01  7.5004232e-01
<br />   5.5712765e-01  2.4315631e-01  1.0935340e+00  1.3072200e+00
<br />   6.2775612e-01  9.9171662e-01  1.9327819e+00  3.9917749e-01
<br />   1.0454532e+00  6.7961502e-01  5.0274509e-01  8.4407705e-01
<br />   6.2175500e-01  1.8040923e+00  1.9134293e+00  7.8312653e-01
<br />   1.3875403e+00  7.9778826e-01  5.4221261e-01  1.5206119e+00
<br />   5.3385168e-01  1.1634347e+00  1.5494859e+00  4.6904349e-01
<br />   1.3343424e-01  6.4280903e-01  1.0107337e+00  6.0661024e-01
<br />   1.1265378e+00  4.9042165e-01  2.2087293e+00  2.2592992e-01
<br />   3.7990010e-01  1.5304949e+00  8.3390290e-01  5.7785541e-01
<br />   6.2842220e-01  7.9012710e-01  5.7400775e-01  1.8335354e+00
<br />   9.9262124e-01  6.5933824e-01  1.2249713e+00  9.0216243e-01
<br />   6.5248871e-01  1.0546095e+00  8.4502345e-01  1.0245783e+00
<br />   1.2587501e+00  2.1604381e+00  1.2569253e+00  1.9936876e+00
<br />   4.3652062e+00 -8.2586079e+00 -4.3412914e+00]]
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-23 01:12:04.805798
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   96.8745
<br />metric_name                                  mean_absolute_error
<br />Name: 0, dtype: object 
<br />
<br />  date_run                              2020-05-23 01:12:04.811065
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   9403.58
<br />metric_name                                   mean_squared_error
<br />Name: 1, dtype: object 
<br />
<br />  date_run                              2020-05-23 01:12:04.814798
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   97.3119
<br />metric_name                                median_absolute_error
<br />Name: 2, dtype: object 
<br />
<br />  date_run                              2020-05-23 01:12:04.817870
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -841.157
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
<br />  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />| N-Beats
<br />| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139849378666144
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139846477499584
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139846477500088
<br />| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139846477095152
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139846477095656
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139846477096160
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f3132295f98> <class 'mlmodels.model_tch.nbeats.Model'>
<br />[[0.40504701]
<br /> [0.40695405]
<br /> [0.39710839]
<br /> ...
<br /> [0.93587014]
<br /> [0.95086039]
<br /> [0.95547277]]
<br />--- fiting ---
<br />grad_step = 000000, loss = 0.484064
<br />plot()
<br />Saved image to .//n_beats_0.png.
<br />grad_step = 000001, loss = 0.455045
<br />grad_step = 000002, loss = 0.440571
<br />grad_step = 000003, loss = 0.426319
<br />grad_step = 000004, loss = 0.411752
<br />grad_step = 000005, loss = 0.394003
<br />grad_step = 000006, loss = 0.374528
<br />grad_step = 000007, loss = 0.355667
<br />grad_step = 000008, loss = 0.340368
<br />grad_step = 000009, loss = 0.327856
<br />grad_step = 000010, loss = 0.319557
<br />grad_step = 000011, loss = 0.310096
<br />grad_step = 000012, loss = 0.298013
<br />grad_step = 000013, loss = 0.286985
<br />grad_step = 000014, loss = 0.277808
<br />grad_step = 000015, loss = 0.269630
<br />grad_step = 000016, loss = 0.261670
<br />grad_step = 000017, loss = 0.253443
<br />grad_step = 000018, loss = 0.244613
<br />grad_step = 000019, loss = 0.233341
<br />grad_step = 000020, loss = 0.222215
<br />grad_step = 000021, loss = 0.212347
<br />grad_step = 000022, loss = 0.204866
<br />grad_step = 000023, loss = 0.197906
<br />grad_step = 000024, loss = 0.189449
<br />grad_step = 000025, loss = 0.180862
<br />grad_step = 000026, loss = 0.173056
<br />grad_step = 000027, loss = 0.165799
<br />grad_step = 000028, loss = 0.158741
<br />grad_step = 000029, loss = 0.151706
<br />grad_step = 000030, loss = 0.144651
<br />grad_step = 000031, loss = 0.137661
<br />grad_step = 000032, loss = 0.130887
<br />grad_step = 000033, loss = 0.124544
<br />grad_step = 000034, loss = 0.118595
<br />grad_step = 000035, loss = 0.112588
<br />grad_step = 000036, loss = 0.106565
<br />grad_step = 000037, loss = 0.100974
<br />grad_step = 000038, loss = 0.095862
<br />grad_step = 000039, loss = 0.090878
<br />grad_step = 000040, loss = 0.085901
<br />grad_step = 000041, loss = 0.081033
<br />grad_step = 000042, loss = 0.076471
<br />grad_step = 000043, loss = 0.072229
<br />grad_step = 000044, loss = 0.068228
<br />grad_step = 000045, loss = 0.064361
<br />grad_step = 000046, loss = 0.060542
<br />grad_step = 000047, loss = 0.056951
<br />grad_step = 000048, loss = 0.053654
<br />grad_step = 000049, loss = 0.050500
<br />grad_step = 000050, loss = 0.047472
<br />grad_step = 000051, loss = 0.044614
<br />grad_step = 000052, loss = 0.041976
<br />grad_step = 000053, loss = 0.039425
<br />grad_step = 000054, loss = 0.036977
<br />grad_step = 000055, loss = 0.034725
<br />grad_step = 000056, loss = 0.032633
<br />grad_step = 000057, loss = 0.030652
<br />grad_step = 000058, loss = 0.028756
<br />grad_step = 000059, loss = 0.026986
<br />grad_step = 000060, loss = 0.025341
<br />grad_step = 000061, loss = 0.023784
<br />grad_step = 000062, loss = 0.022310
<br />grad_step = 000063, loss = 0.020969
<br />grad_step = 000064, loss = 0.019708
<br />grad_step = 000065, loss = 0.018493
<br />grad_step = 000066, loss = 0.017373
<br />grad_step = 000067, loss = 0.016344
<br />grad_step = 000068, loss = 0.015375
<br />grad_step = 000069, loss = 0.014458
<br />grad_step = 000070, loss = 0.013615
<br />grad_step = 000071, loss = 0.012831
<br />grad_step = 000072, loss = 0.012091
<br />grad_step = 000073, loss = 0.011400
<br />grad_step = 000074, loss = 0.010769
<br />grad_step = 000075, loss = 0.010167
<br />grad_step = 000076, loss = 0.009608
<br />grad_step = 000077, loss = 0.009093
<br />grad_step = 000078, loss = 0.008609
<br />grad_step = 000079, loss = 0.008156
<br />grad_step = 000080, loss = 0.007735
<br />grad_step = 000081, loss = 0.007341
<br />grad_step = 000082, loss = 0.006968
<br />grad_step = 000083, loss = 0.006624
<br />grad_step = 000084, loss = 0.006304
<br />grad_step = 000085, loss = 0.005999
<br />grad_step = 000086, loss = 0.005715
<br />grad_step = 000087, loss = 0.005450
<br />grad_step = 000088, loss = 0.005201
<br />grad_step = 000089, loss = 0.004966
<br />grad_step = 000090, loss = 0.004748
<br />grad_step = 000091, loss = 0.004540
<br />grad_step = 000092, loss = 0.004348
<br />grad_step = 000093, loss = 0.004168
<br />grad_step = 000094, loss = 0.003997
<br />grad_step = 000095, loss = 0.003837
<br />grad_step = 000096, loss = 0.003687
<br />grad_step = 000097, loss = 0.003546
<br />grad_step = 000098, loss = 0.003414
<br />grad_step = 000099, loss = 0.003290
<br />grad_step = 000100, loss = 0.003173
<br />plot()
<br />Saved image to .//n_beats_100.png.
<br />grad_step = 000101, loss = 0.003065
<br />grad_step = 000102, loss = 0.002962
<br />grad_step = 000103, loss = 0.002866
<br />grad_step = 000104, loss = 0.002777
<br />grad_step = 000105, loss = 0.002693
<br />grad_step = 000106, loss = 0.002614
<br />grad_step = 000107, loss = 0.002541
<br />grad_step = 000108, loss = 0.002474
<br />grad_step = 000109, loss = 0.002412
<br />grad_step = 000110, loss = 0.002358
<br />grad_step = 000111, loss = 0.002308
<br />grad_step = 000112, loss = 0.002262
<br />grad_step = 000113, loss = 0.002212
<br />grad_step = 000114, loss = 0.002160
<br />grad_step = 000115, loss = 0.002110
<br />grad_step = 000116, loss = 0.002069
<br />grad_step = 000117, loss = 0.002038
<br />grad_step = 000118, loss = 0.002016
<br />grad_step = 000119, loss = 0.002021
<br />grad_step = 000120, loss = 0.002061
<br />grad_step = 000121, loss = 0.002116
<br />grad_step = 000122, loss = 0.002002
<br />grad_step = 000123, loss = 0.001874
<br />grad_step = 000124, loss = 0.001901
<br />grad_step = 000125, loss = 0.001943
<br />grad_step = 000126, loss = 0.001870
<br />grad_step = 000127, loss = 0.001806
<br />grad_step = 000128, loss = 0.001848
<br />grad_step = 000129, loss = 0.001830
<br />grad_step = 000130, loss = 0.001763
<br />grad_step = 000131, loss = 0.001794
<br />grad_step = 000132, loss = 0.001811
<br />grad_step = 000133, loss = 0.001743
<br />grad_step = 000134, loss = 0.001735
<br />grad_step = 000135, loss = 0.001751
<br />grad_step = 000136, loss = 0.001710
<br />grad_step = 000137, loss = 0.001689
<br />grad_step = 000138, loss = 0.001695
<br />grad_step = 000139, loss = 0.001683
<br />grad_step = 000140, loss = 0.001675
<br />grad_step = 000141, loss = 0.001669
<br />grad_step = 000142, loss = 0.001666
<br />grad_step = 000143, loss = 0.001692
<br />grad_step = 000144, loss = 0.001748
<br />grad_step = 000145, loss = 0.001773
<br />grad_step = 000146, loss = 0.001825
<br />grad_step = 000147, loss = 0.001756
<br />grad_step = 000148, loss = 0.001634
<br />grad_step = 000149, loss = 0.001632
<br />grad_step = 000150, loss = 0.001690
<br />grad_step = 000151, loss = 0.001681
<br />grad_step = 000152, loss = 0.001656
<br />grad_step = 000153, loss = 0.001613
<br />grad_step = 000154, loss = 0.001598
<br />grad_step = 000155, loss = 0.001645
<br />grad_step = 000156, loss = 0.001640
<br />grad_step = 000157, loss = 0.001611
<br />grad_step = 000158, loss = 0.001594
<br />grad_step = 000159, loss = 0.001576
<br />grad_step = 000160, loss = 0.001604
<br />grad_step = 000161, loss = 0.001611
<br />grad_step = 000162, loss = 0.001587
<br />grad_step = 000163, loss = 0.001581
<br />grad_step = 000164, loss = 0.001561
<br />grad_step = 000165, loss = 0.001561
<br />grad_step = 000166, loss = 0.001576
<br />grad_step = 000167, loss = 0.001570
<br />grad_step = 000168, loss = 0.001572
<br />grad_step = 000169, loss = 0.001564
<br />grad_step = 000170, loss = 0.001546
<br />grad_step = 000171, loss = 0.001540
<br />grad_step = 000172, loss = 0.001534
<br />grad_step = 000173, loss = 0.001531
<br />grad_step = 000174, loss = 0.001538
<br />grad_step = 000175, loss = 0.001538
<br />grad_step = 000176, loss = 0.001535
<br />grad_step = 000177, loss = 0.001539
<br />grad_step = 000178, loss = 0.001540
<br />grad_step = 000179, loss = 0.001548
<br />grad_step = 000180, loss = 0.001556
<br />grad_step = 000181, loss = 0.001565
<br />grad_step = 000182, loss = 0.001564
<br />grad_step = 000183, loss = 0.001568
<br />grad_step = 000184, loss = 0.001560
<br />grad_step = 000185, loss = 0.001563
<br />grad_step = 000186, loss = 0.001543
<br />grad_step = 000187, loss = 0.001516
<br />grad_step = 000188, loss = 0.001491
<br />grad_step = 000189, loss = 0.001486
<br />grad_step = 000190, loss = 0.001498
<br />grad_step = 000191, loss = 0.001503
<br />grad_step = 000192, loss = 0.001501
<br />grad_step = 000193, loss = 0.001498
<br />grad_step = 000194, loss = 0.001512
<br />grad_step = 000195, loss = 0.001532
<br />grad_step = 000196, loss = 0.001551
<br />grad_step = 000197, loss = 0.001540
<br />grad_step = 000198, loss = 0.001526
<br />grad_step = 000199, loss = 0.001510
<br />grad_step = 000200, loss = 0.001526
<br />plot()
<br />Saved image to .//n_beats_200.png.
<br />grad_step = 000201, loss = 0.001534
<br />grad_step = 000202, loss = 0.001518
<br />grad_step = 000203, loss = 0.001476
<br />grad_step = 000204, loss = 0.001463
<br />grad_step = 000205, loss = 0.001468
<br />grad_step = 000206, loss = 0.001476
<br />grad_step = 000207, loss = 0.001475
<br />grad_step = 000208, loss = 0.001462
<br />grad_step = 000209, loss = 0.001459
<br />grad_step = 000210, loss = 0.001465
<br />grad_step = 000211, loss = 0.001470
<br />grad_step = 000212, loss = 0.001464
<br />grad_step = 000213, loss = 0.001453
<br />grad_step = 000214, loss = 0.001455
<br />grad_step = 000215, loss = 0.001474
<br />grad_step = 000216, loss = 0.001475
<br />grad_step = 000217, loss = 0.001478
<br />grad_step = 000218, loss = 0.001491
<br />grad_step = 000219, loss = 0.001517
<br />grad_step = 000220, loss = 0.001521
<br />grad_step = 000221, loss = 0.001522
<br />grad_step = 000222, loss = 0.001489
<br />grad_step = 000223, loss = 0.001461
<br />grad_step = 000224, loss = 0.001436
<br />grad_step = 000225, loss = 0.001423
<br />grad_step = 000226, loss = 0.001420
<br />grad_step = 000227, loss = 0.001429
<br />grad_step = 000228, loss = 0.001445
<br />grad_step = 000229, loss = 0.001455
<br />grad_step = 000230, loss = 0.001455
<br />grad_step = 000231, loss = 0.001437
<br />grad_step = 000232, loss = 0.001426
<br />grad_step = 000233, loss = 0.001419
<br />grad_step = 000234, loss = 0.001411
<br />grad_step = 000235, loss = 0.001400
<br />grad_step = 000236, loss = 0.001392
<br />grad_step = 000237, loss = 0.001391
<br />grad_step = 000238, loss = 0.001395
<br />grad_step = 000239, loss = 0.001398
<br />grad_step = 000240, loss = 0.001398
<br />grad_step = 000241, loss = 0.001402
<br />grad_step = 000242, loss = 0.001411
<br />grad_step = 000243, loss = 0.001431
<br />grad_step = 000244, loss = 0.001453
<br />grad_step = 000245, loss = 0.001489
<br />grad_step = 000246, loss = 0.001508
<br />grad_step = 000247, loss = 0.001538
<br />grad_step = 000248, loss = 0.001503
<br />grad_step = 000249, loss = 0.001453
<br />grad_step = 000250, loss = 0.001389
<br />grad_step = 000251, loss = 0.001368
<br />grad_step = 000252, loss = 0.001393
<br />grad_step = 000253, loss = 0.001423
<br />grad_step = 000254, loss = 0.001431
<br />grad_step = 000255, loss = 0.001399
<br />grad_step = 000256, loss = 0.001369
<br />grad_step = 000257, loss = 0.001358
<br />grad_step = 000258, loss = 0.001368
<br />grad_step = 000259, loss = 0.001384
<br />grad_step = 000260, loss = 0.001386
<br />grad_step = 000261, loss = 0.001380
<br />grad_step = 000262, loss = 0.001365
<br />grad_step = 000263, loss = 0.001355
<br />grad_step = 000264, loss = 0.001349
<br />grad_step = 000265, loss = 0.001346
<br />grad_step = 000266, loss = 0.001348
<br />grad_step = 000267, loss = 0.001354
<br />grad_step = 000268, loss = 0.001361
<br />grad_step = 000269, loss = 0.001361
<br />grad_step = 000270, loss = 0.001358
<br />grad_step = 000271, loss = 0.001350
<br />grad_step = 000272, loss = 0.001345
<br />grad_step = 000273, loss = 0.001339
<br />grad_step = 000274, loss = 0.001335
<br />grad_step = 000275, loss = 0.001331
<br />grad_step = 000276, loss = 0.001327
<br />grad_step = 000277, loss = 0.001323
<br />grad_step = 000278, loss = 0.001319
<br />grad_step = 000279, loss = 0.001317
<br />grad_step = 000280, loss = 0.001315
<br />grad_step = 000281, loss = 0.001314
<br />grad_step = 000282, loss = 0.001313
<br />grad_step = 000283, loss = 0.001312
<br />grad_step = 000284, loss = 0.001311
<br />grad_step = 000285, loss = 0.001312
<br />grad_step = 000286, loss = 0.001319
<br />grad_step = 000287, loss = 0.001333
<br />grad_step = 000288, loss = 0.001371
<br />grad_step = 000289, loss = 0.001434
<br />grad_step = 000290, loss = 0.001581
<br />grad_step = 000291, loss = 0.001685
<br />grad_step = 000292, loss = 0.001829
<br />grad_step = 000293, loss = 0.001562
<br />grad_step = 000294, loss = 0.001337
<br />grad_step = 000295, loss = 0.001340
<br />grad_step = 000296, loss = 0.001477
<br />grad_step = 000297, loss = 0.001482
<br />grad_step = 000298, loss = 0.001321
<br />grad_step = 000299, loss = 0.001340
<br />grad_step = 000300, loss = 0.001447
<br />plot()
<br />Saved image to .//n_beats_300.png.
<br />grad_step = 000301, loss = 0.001380
<br />grad_step = 000302, loss = 0.001304
<br />grad_step = 000303, loss = 0.001321
<br />grad_step = 000304, loss = 0.001362
<br />grad_step = 000305, loss = 0.001355
<br />grad_step = 000306, loss = 0.001299
<br />grad_step = 000307, loss = 0.001293
<br />grad_step = 000308, loss = 0.001327
<br />grad_step = 000309, loss = 0.001327
<br />grad_step = 000310, loss = 0.001294
<br />grad_step = 000311, loss = 0.001273
<br />grad_step = 000312, loss = 0.001292
<br />grad_step = 000313, loss = 0.001312
<br />grad_step = 000314, loss = 0.001291
<br />grad_step = 000315, loss = 0.001264
<br />grad_step = 000316, loss = 0.001267
<br />grad_step = 000317, loss = 0.001285
<br />grad_step = 000318, loss = 0.001284
<br />grad_step = 000319, loss = 0.001265
<br />grad_step = 000320, loss = 0.001256
<br />grad_step = 000321, loss = 0.001263
<br />grad_step = 000322, loss = 0.001267
<br />grad_step = 000323, loss = 0.001262
<br />grad_step = 000324, loss = 0.001255
<br />grad_step = 000325, loss = 0.001252
<br />grad_step = 000326, loss = 0.001251
<br />grad_step = 000327, loss = 0.001251
<br />grad_step = 000328, loss = 0.001252
<br />grad_step = 000329, loss = 0.001250
<br />grad_step = 000330, loss = 0.001245
<br />grad_step = 000331, loss = 0.001240
<br />grad_step = 000332, loss = 0.001240
<br />grad_step = 000333, loss = 0.001242
<br />grad_step = 000334, loss = 0.001243
<br />grad_step = 000335, loss = 0.001239
<br />grad_step = 000336, loss = 0.001234
<br />grad_step = 000337, loss = 0.001232
<br />grad_step = 000338, loss = 0.001232
<br />grad_step = 000339, loss = 0.001232
<br />grad_step = 000340, loss = 0.001231
<br />grad_step = 000341, loss = 0.001230
<br />grad_step = 000342, loss = 0.001228
<br />grad_step = 000343, loss = 0.001226
<br />grad_step = 000344, loss = 0.001223
<br />grad_step = 000345, loss = 0.001222
<br />grad_step = 000346, loss = 0.001221
<br />grad_step = 000347, loss = 0.001221
<br />grad_step = 000348, loss = 0.001221
<br />grad_step = 000349, loss = 0.001219
<br />grad_step = 000350, loss = 0.001217
<br />grad_step = 000351, loss = 0.001216
<br />grad_step = 000352, loss = 0.001215
<br />grad_step = 000353, loss = 0.001214
<br />grad_step = 000354, loss = 0.001213
<br />grad_step = 000355, loss = 0.001213
<br />grad_step = 000356, loss = 0.001215
<br />grad_step = 000357, loss = 0.001219
<br />grad_step = 000358, loss = 0.001228
<br />grad_step = 000359, loss = 0.001248
<br />grad_step = 000360, loss = 0.001278
<br />grad_step = 000361, loss = 0.001336
<br />grad_step = 000362, loss = 0.001385
<br />grad_step = 000363, loss = 0.001452
<br />grad_step = 000364, loss = 0.001406
<br />grad_step = 000365, loss = 0.001342
<br />grad_step = 000366, loss = 0.001249
<br />grad_step = 000367, loss = 0.001223
<br />grad_step = 000368, loss = 0.001256
<br />grad_step = 000369, loss = 0.001288
<br />grad_step = 000370, loss = 0.001291
<br />grad_step = 000371, loss = 0.001243
<br />grad_step = 000372, loss = 0.001212
<br />grad_step = 000373, loss = 0.001214
<br />grad_step = 000374, loss = 0.001236
<br />grad_step = 000375, loss = 0.001256
<br />grad_step = 000376, loss = 0.001241
<br />grad_step = 000377, loss = 0.001214
<br />grad_step = 000378, loss = 0.001192
<br />grad_step = 000379, loss = 0.001196
<br />grad_step = 000380, loss = 0.001216
<br />grad_step = 000381, loss = 0.001224
<br />grad_step = 000382, loss = 0.001218
<br />grad_step = 000383, loss = 0.001198
<br />grad_step = 000384, loss = 0.001185
<br />grad_step = 000385, loss = 0.001183
<br />grad_step = 000386, loss = 0.001190
<br />grad_step = 000387, loss = 0.001197
<br />grad_step = 000388, loss = 0.001198
<br />grad_step = 000389, loss = 0.001195
<br />grad_step = 000390, loss = 0.001187
<br />grad_step = 000391, loss = 0.001179
<br />grad_step = 000392, loss = 0.001174
<br />grad_step = 000393, loss = 0.001173
<br />grad_step = 000394, loss = 0.001176
<br />grad_step = 000395, loss = 0.001179
<br />grad_step = 000396, loss = 0.001180
<br />grad_step = 000397, loss = 0.001179
<br />grad_step = 000398, loss = 0.001175
<br />grad_step = 000399, loss = 0.001171
<br />grad_step = 000400, loss = 0.001167
<br />plot()
<br />Saved image to .//n_beats_400.png.
<br />grad_step = 000401, loss = 0.001164
<br />grad_step = 000402, loss = 0.001162
<br />grad_step = 000403, loss = 0.001162
<br />grad_step = 000404, loss = 0.001162
<br />grad_step = 000405, loss = 0.001163
<br />grad_step = 000406, loss = 0.001163
<br />grad_step = 000407, loss = 0.001164
<br />grad_step = 000408, loss = 0.001164
<br />grad_step = 000409, loss = 0.001165
<br />grad_step = 000410, loss = 0.001166
<br />grad_step = 000411, loss = 0.001167
<br />grad_step = 000412, loss = 0.001167
<br />grad_step = 000413, loss = 0.001170
<br />grad_step = 000414, loss = 0.001171
<br />grad_step = 000415, loss = 0.001175
<br />grad_step = 000416, loss = 0.001178
<br />grad_step = 000417, loss = 0.001183
<br />grad_step = 000418, loss = 0.001185
<br />grad_step = 000419, loss = 0.001189
<br />grad_step = 000420, loss = 0.001188
<br />grad_step = 000421, loss = 0.001188
<br />grad_step = 000422, loss = 0.001179
<br />grad_step = 000423, loss = 0.001171
<br />grad_step = 000424, loss = 0.001158
<br />grad_step = 000425, loss = 0.001148
<br />grad_step = 000426, loss = 0.001141
<br />grad_step = 000427, loss = 0.001137
<br />grad_step = 000428, loss = 0.001136
<br />grad_step = 000429, loss = 0.001138
<br />grad_step = 000430, loss = 0.001142
<br />grad_step = 000431, loss = 0.001146
<br />grad_step = 000432, loss = 0.001152
<br />grad_step = 000433, loss = 0.001159
<br />grad_step = 000434, loss = 0.001170
<br />grad_step = 000435, loss = 0.001179
<br />grad_step = 000436, loss = 0.001197
<br />grad_step = 000437, loss = 0.001204
<br />grad_step = 000438, loss = 0.001216
<br />grad_step = 000439, loss = 0.001205
<br />grad_step = 000440, loss = 0.001191
<br />grad_step = 000441, loss = 0.001161
<br />grad_step = 000442, loss = 0.001137
<br />grad_step = 000443, loss = 0.001124
<br />grad_step = 000444, loss = 0.001125
<br />grad_step = 000445, loss = 0.001137
<br />grad_step = 000446, loss = 0.001149
<br />grad_step = 000447, loss = 0.001158
<br />grad_step = 000448, loss = 0.001154
<br />grad_step = 000449, loss = 0.001148
<br />grad_step = 000450, loss = 0.001136
<br />grad_step = 000451, loss = 0.001128
<br />grad_step = 000452, loss = 0.001121
<br />grad_step = 000453, loss = 0.001117
<br />grad_step = 000454, loss = 0.001117
<br />grad_step = 000455, loss = 0.001120
<br />grad_step = 000456, loss = 0.001126
<br />grad_step = 000457, loss = 0.001133
<br />grad_step = 000458, loss = 0.001142
<br />grad_step = 000459, loss = 0.001148
<br />grad_step = 000460, loss = 0.001158
<br />grad_step = 000461, loss = 0.001164
<br />grad_step = 000462, loss = 0.001176
<br />grad_step = 000463, loss = 0.001179
<br />grad_step = 000464, loss = 0.001187
<br />grad_step = 000465, loss = 0.001178
<br />grad_step = 000466, loss = 0.001170
<br />grad_step = 000467, loss = 0.001150
<br />grad_step = 000468, loss = 0.001134
<br />grad_step = 000469, loss = 0.001123
<br />grad_step = 000470, loss = 0.001115
<br />grad_step = 000471, loss = 0.001112
<br />grad_step = 000472, loss = 0.001111
<br />grad_step = 000473, loss = 0.001113
<br />grad_step = 000474, loss = 0.001117
<br />grad_step = 000475, loss = 0.001122
<br />grad_step = 000476, loss = 0.001121
<br />grad_step = 000477, loss = 0.001113
<br />grad_step = 000478, loss = 0.001101
<br />grad_step = 000479, loss = 0.001091
<br />grad_step = 000480, loss = 0.001086
<br />grad_step = 000481, loss = 0.001088
<br />grad_step = 000482, loss = 0.001091
<br />grad_step = 000483, loss = 0.001093
<br />grad_step = 000484, loss = 0.001092
<br />grad_step = 000485, loss = 0.001091
<br />grad_step = 000486, loss = 0.001093
<br />grad_step = 000487, loss = 0.001098
<br />grad_step = 000488, loss = 0.001105
<br />grad_step = 000489, loss = 0.001111
<br />grad_step = 000490, loss = 0.001122
<br />grad_step = 000491, loss = 0.001129
<br />grad_step = 000492, loss = 0.001143
<br />grad_step = 000493, loss = 0.001146
<br />grad_step = 000494, loss = 0.001150
<br />grad_step = 000495, loss = 0.001134
<br />grad_step = 000496, loss = 0.001116
<br />grad_step = 000497, loss = 0.001090
<br />grad_step = 000498, loss = 0.001073
<br />grad_step = 000499, loss = 0.001068
<br />grad_step = 000500, loss = 0.001075
<br />plot()
<br />Saved image to .//n_beats_500.png.
<br />grad_step = 000501, loss = 0.001087
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
<br />  date_run                              2020-05-23 01:12:23.424886
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.259609
<br />metric_name                                  mean_absolute_error
<br />Name: 4, dtype: object 
<br />
<br />  date_run                              2020-05-23 01:12:23.430672
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.176683
<br />metric_name                                   mean_squared_error
<br />Name: 5, dtype: object 
<br />
<br />  date_run                              2020-05-23 01:12:23.437248
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.135785
<br />metric_name                                median_absolute_error
<br />Name: 6, dtype: object 
<br />
<br />  date_run                              2020-05-23 01:12:23.442102
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -1.68476
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
<br />  data_pars out_pars {'train_data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'} {'outpath': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_fb/fb_prophet/'} 
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
<br />>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f31322a5f60> <class 'mlmodels.model_gluon.fb_prophet.Model'>
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-23 01:12:39.654730
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.3339
<br />metric_name                                  mean_absolute_error
<br />Name: 8, dtype: object 
<br />
<br />  date_run                              2020-05-23 01:12:39.657988
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   215.367
<br />metric_name                                   mean_squared_error
<br />Name: 9, dtype: object 
<br />
<br />  date_run                              2020-05-23 01:12:39.661074
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.4309
<br />metric_name                                median_absolute_error
<br />Name: 10, dtype: object 
<br />
<br />  date_run                              2020-05-23 01:12:39.664158
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
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} 'model_uri' 
<br />
<br />  benchmark file saved at https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/example/benchmark/ 
<br />
<br />                        date_run  ...            metric_name
<br />0   2020-05-23 01:12:04.805798  ...    mean_absolute_error
<br />1   2020-05-23 01:12:04.811065  ...     mean_squared_error
<br />2   2020-05-23 01:12:04.814798  ...  median_absolute_error
<br />3   2020-05-23 01:12:04.817870  ...               r2_score
<br />4   2020-05-23 01:12:23.424886  ...    mean_absolute_error
<br />5   2020-05-23 01:12:23.430672  ...     mean_squared_error
<br />6   2020-05-23 01:12:23.437248  ...  median_absolute_error
<br />7   2020-05-23 01:12:23.442102  ...               r2_score
<br />8   2020-05-23 01:12:39.654730  ...    mean_absolute_error
<br />9   2020-05-23 01:12:39.657988  ...     mean_squared_error
<br />10  2020-05-23 01:12:39.661074  ...  median_absolute_error
<br />11  2020-05-23 01:12:39.664158  ...               r2_score
<br />
<br />[12 rows x 6 columns] 



### Error 14, [Traceback at line 3364](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L3364)<br />3364..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce/mlmodels/benchmark.py", line 118, in benchmark_run
<br />    model_uri =  model_pars['model_uri']
<br />KeyError: 'model_uri'
