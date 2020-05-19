## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py


### Error 1, [Traceback at line 156](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L156)<br />156..Traceback (most recent call last):
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



### Error 2, [Traceback at line 168](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L168)<br />168..Traceback (most recent call last):
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



### Error 3, [Traceback at line 223](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L223)<br />223..Traceback (most recent call last):
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



### Error 4, [Traceback at line 234](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L234)<br />234..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/models.py", line 531, in main
<br />    predict_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/models.py", line 442, in predict_cli
<br />    model, session = load(module, load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/models.py", line 156, in load
<br />    return module.load(load_pars, **kwarg)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_tf/1_lstm.py", line 199, in load
<br />    return load_tf(load_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/util.py", line 477, in load_tf
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
<br />  <module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_tf/1_lstm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.model_tf.1_lstm.Model object at 0x7f88ed8140b8> 
<br />
<br />  #### Fit   ######################################################## 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br /> [ 0.05474646  0.02520508  0.0338237  -0.02119037  0.02119336  0.01744108]
<br /> [ 0.28258875  0.08274993  0.01459031 -0.01285301 -0.08252246  0.17278814]
<br /> [ 0.04866791  0.04155822 -0.16202316  0.04278629  0.07459117  0.00128272]
<br /> [ 0.03096858 -0.18434891 -0.1656065  -0.16163316 -0.01006034  0.00660182]
<br /> [ 0.11872631  0.24894005 -0.11171068  0.35538924 -0.2634947   0.2242782 ]
<br /> [-0.5055359   0.3458035  -0.23395011  0.03110237  0.27734432  0.57572246]
<br /> [ 0.62201911 -0.04912861 -0.18720469  0.59876877 -0.37671959  0.0679053 ]
<br /> [-0.00920019  0.03391777  0.33457103  0.15898098 -0.19421458 -0.01366312]
<br /> [ 0.          0.          0.          0.          0.          0.        ]]
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />model_tf.1_lstm
<br />model_tf.1_lstm
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.411528991535306, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-19 13:01:15.019623: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/model'}
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
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.535798117518425, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-19 13:01:16.322714: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/model'}
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
<br />  <module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/example/custom_model/1_lstm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.example.custom_model.1_lstm.Model object at 0x7f2037a22198> 
<br />
<br />  #### Fit   ######################################################## 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br /> [ 1.17319785e-01  4.09056023e-02 -4.21057083e-02  3.25592421e-02
<br />   3.55109125e-02  9.79551300e-02]
<br /> [-1.18257955e-01  1.07090108e-01 -1.57314569e-01  1.26011819e-01
<br />   1.84508190e-01  2.24172518e-01]
<br /> [ 1.25343949e-01 -1.64537877e-02  1.77439362e-01  3.48974407e-01
<br />  -5.85007444e-02  3.35837334e-01]
<br /> [-2.59205550e-01  1.77525103e-01 -1.19217560e-02  8.46833810e-02
<br />   2.03216940e-01  2.50900090e-01]
<br /> [-1.97932497e-01 -1.95551187e-01  1.97494581e-01  1.05539508e-01
<br />   6.41370177e-01 -3.08437824e-01]
<br /> [-4.39388677e-05  3.17149222e-01 -5.13712950e-02  4.71269578e-01
<br />  -4.33498621e-01  2.71741569e-01]
<br /> [-1.53063789e-01  2.17422947e-01  2.67824918e-01  4.60827947e-01
<br />   4.00111318e-01  3.18934411e-01]
<br /> [ 6.91152394e-01  7.36106157e-01  5.03316820e-01  1.43736482e-01
<br />   3.55694383e-01  3.62612724e-01]
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
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/example/custom_model/1_lstm.py'>
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/example/custom_model/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.4749623239040375, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />2020-05-19 13:01:21.597576: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />{'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/model'}
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
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/example/custom_model/1_lstm.py'>
<br /><module 'mlmodels.example.custom_model.1_lstm' from 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/example/custom_model/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.5124258175492287, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />2020-05-19 13:01:22.823272: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />{'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_tf/1_lstm/model'}
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



### Error 5, [Traceback at line 960](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L960)<br />960..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_optim", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_optim')()
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/optim.py", line 388, in main
<br />    optim_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/optim.py", line 259, in optim_cli
<br />    out_pars        = out_pars )
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/optim.py", line 54, in optim
<br />    if hypermodel_pars["engine_pars"]['engine'] == "optuna":
<br />KeyError: 'engine_pars'



### Error 6, [Traceback at line 2092](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2092)<br />2092..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 140, in fit
<br />    gluont_ds          = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 106, in get_dataset
<br />    from mlmodels.preprocess.timeseries import pandas_to_gluonts, pd_clean_v1
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/preprocess/timeseries.py", line 378
<br />    train_ds, test_ds, cardinalities   = pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static  pars)
<br />                                                                                                                ^
<br />SyntaxError: invalid syntax



### Error 7, [Traceback at line 2120](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2120)<br />2120..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 140, in fit
<br />    gluont_ds          = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 106, in get_dataset
<br />    from mlmodels.preprocess.timeseries import pandas_to_gluonts, pd_clean_v1
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/preprocess/timeseries.py", line 378
<br />    train_ds, test_ds, cardinalities   = pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static  pars)
<br />                                                                                                                ^
<br />SyntaxError: invalid syntax



### Error 8, [Traceback at line 2149](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2149)<br />2149..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 140, in fit
<br />    gluont_ds          = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 106, in get_dataset
<br />    from mlmodels.preprocess.timeseries import pandas_to_gluonts, pd_clean_v1
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/preprocess/timeseries.py", line 378
<br />    train_ds, test_ds, cardinalities   = pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static  pars)
<br />                                                                                                                ^
<br />SyntaxError: invalid syntax



### Error 9, [Traceback at line 2177](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2177)<br />2177..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 140, in fit
<br />    gluont_ds          = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 106, in get_dataset
<br />    from mlmodels.preprocess.timeseries import pandas_to_gluonts, pd_clean_v1
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/preprocess/timeseries.py", line 378
<br />    train_ds, test_ds, cardinalities   = pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static  pars)
<br />                                                                                                                ^
<br />SyntaxError: invalid syntax



### Error 10, [Traceback at line 2205](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2205)<br />2205..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 140, in fit
<br />    gluont_ds          = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 106, in get_dataset
<br />    from mlmodels.preprocess.timeseries import pandas_to_gluonts, pd_clean_v1
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/preprocess/timeseries.py", line 378
<br />    train_ds, test_ds, cardinalities   = pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static  pars)
<br />                                                                                                                ^
<br />SyntaxError: invalid syntax



### Error 11, [Traceback at line 2233](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2233)<br />2233..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 140, in fit
<br />    gluont_ds          = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 106, in get_dataset
<br />    from mlmodels.preprocess.timeseries import pandas_to_gluonts, pd_clean_v1
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/preprocess/timeseries.py", line 378
<br />    train_ds, test_ds, cardinalities   = pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static  pars)
<br />                                                                                                                ^
<br />SyntaxError: invalid syntax



### Error 12, [Traceback at line 2283](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2283)<br />2283..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/benchmark.py", line 126, in benchmark_run
<br />    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 140, in fit
<br />    gluont_ds          = get_dataset(data_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 106, in get_dataset
<br />    from mlmodels.preprocess.timeseries import pandas_to_gluonts, pd_clean_v1
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/preprocess/timeseries.py", line 378
<br />    train_ds, test_ds, cardinalities   = pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static  pars)
<br />                                                                                                                ^
<br />SyntaxError: invalid syntax



### Error 13, [Traceback at line 2294](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L2294)<br />2294..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/benchmark.py", line 120, in benchmark_run
<br />    model     = module.Model(model_pars, data_pars, compute_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
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
<br />  json_path https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/json/benchmark_timeseries/test01/ 
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
<br />  data_pars out_pars {'train': True, 'col_Xinput': ['Close'], 'col_ytarget': 'Close', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60} {'outpath': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_keras/armdn/'} 
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
<br />>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f16131e2ef0> <class 'mlmodels.model_keras.armdn.Model'>
<br />
<br />  #### Loading dataset   ############################################# 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
<br />
<br />Epoch 1/10
<br />
<br />1/1 [==============================] - 2s 2s/step - loss: 356310.5312
<br />Epoch 2/10
<br />
<br />1/1 [==============================] - 0s 121ms/step - loss: 264154.1875
<br />Epoch 3/10
<br />
<br />1/1 [==============================] - 0s 103ms/step - loss: 165456.4844
<br />Epoch 4/10
<br />
<br />1/1 [==============================] - 0s 124ms/step - loss: 93462.1406
<br />Epoch 5/10
<br />
<br />1/1 [==============================] - 0s 100ms/step - loss: 50609.1016
<br />Epoch 6/10
<br />
<br />1/1 [==============================] - 0s 103ms/step - loss: 28594.1504
<br />Epoch 7/10
<br />
<br />1/1 [==============================] - 0s 99ms/step - loss: 17601.5254
<br />Epoch 8/10
<br />
<br />1/1 [==============================] - 0s 101ms/step - loss: 11856.9707
<br />Epoch 9/10
<br />
<br />1/1 [==============================] - 0s 106ms/step - loss: 8542.5566
<br />Epoch 10/10
<br />
<br />1/1 [==============================] - 0s 107ms/step - loss: 6517.7925
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />[[-2.4050033  -0.31163582  0.78437555 -0.7384667   0.63164246  0.61216116
<br />  -1.5113249  -1.6487643   0.8521124  -0.21959071  1.5732814  -2.0863528
<br />   1.3037391   1.7714931  -1.181113   -2.118582    1.0336534  -0.80246973
<br />  -0.06544949 -1.2428807  -0.25418225  0.15361342 -1.6453866  -0.90499187
<br />  -0.8172206   0.24092421  0.09654251 -1.610411   -1.3658887   0.05165559
<br />   0.35114872 -0.82414865 -1.3227835  -0.49641016 -0.51256734 -0.1469029
<br />  -0.08295119  1.164138   -0.20292532  0.9542413   0.38832226  0.13486046
<br />  -0.6545193   0.18702555  0.69227105 -0.516985    0.2526611   0.44150373
<br />   1.4266453   1.5354451   0.41591424  0.11938981  0.30260772  0.19959065
<br />   0.67209625  1.1075692  -2.186457   -0.37191427 -0.36625993  0.30746105
<br />  -0.21587695  7.804063   10.839045    8.453226    8.120212    7.67237
<br />   9.505883    6.886793    8.384726    8.296047    7.627017    8.736551
<br />  10.115404    7.624832    6.44124     9.009318    9.150042    8.141428
<br />   8.45769     9.860116    8.791417   10.019918    8.941957    7.904322
<br />   9.060243    8.247234    8.044439    7.759871    8.065253    8.709772
<br />   6.8446436   8.079255    8.130889    8.835646    9.523339    8.6657715
<br />   8.619498    9.744658    7.057626    9.778323    8.565583    6.845486
<br />   8.095626    7.1758633   9.117787    8.83833     7.080151    7.0040665
<br />   8.76079     8.042817    6.676881    7.433894    7.021713    7.8996983
<br />   9.043434    7.8386164   6.560085    7.3165026   9.175095    8.462609
<br />  -1.8540467  -2.122372   -0.1462086  -1.2969944   0.39292714  0.54109406
<br />  -0.8537086   0.22648746  1.0484294   1.2363127   1.9332694   1.4203728
<br />   0.86842304  1.8111951  -0.528133    1.2872391  -0.5507606   1.2507027
<br />   1.7554045   0.21931104  0.8822732  -1.5388986   0.73704326  0.9254042
<br />  -0.10533197 -0.26015165  2.4666126   1.8298191  -0.20508933 -0.33215532
<br />   0.25659454  0.10548557  0.67326087  1.4954673  -1.418293    0.19406447
<br />  -0.3369249  -0.8774993   2.1083806  -1.9991118  -0.357097   -0.88062835
<br />  -1.9168615  -0.7141539  -0.8937972  -0.1791921  -1.4419215  -1.3540452
<br />   1.6361778  -0.8014393  -1.0917587  -1.1304971  -1.5088682  -1.2785451
<br />  -0.49565697 -0.16440657 -0.7419968   0.4790538   0.9948838  -1.1981932
<br />   0.8895711   2.4568284   0.44479644  1.0797327   0.60948116  0.59804684
<br />   0.5505984   0.5097213   0.53503114  1.365077    0.6358012   0.5331201
<br />   0.6124841   0.8717066   1.8583373   0.21161044  0.92444843  2.2098675
<br />   2.24053     2.057993    0.19309777  0.40198475  0.46351188  0.4675693
<br />   2.0217195   0.07270336  2.180161    1.4698482   2.9262004   0.7546048
<br />   1.3480294   0.38351828  0.13496405  0.37035358  2.1258702   1.3516132
<br />   1.977484    1.3758277   2.5546634   0.2650869   0.72532004  1.0667804
<br />   0.8171898   1.9095025   1.1354673   2.6317167   0.3309828   0.738201
<br />   0.56728125  0.09172446  2.256311    1.5692803   0.8966676   0.66306055
<br />   0.58836466  0.34213722  1.9393257   2.6127038   0.46515548  1.8328373
<br />   0.09153068  8.261249    6.4854283   8.756653    7.635831    9.06487
<br />  10.322221    8.575523    9.040068   10.138007    8.964354    9.18462
<br />   8.268604   10.303228    9.276009    7.4085636   9.353796    8.420371
<br />   7.77066     9.29776     7.4531646   9.326793    7.091158    7.8635015
<br />   7.2822533   6.6651897   9.20961     6.551423    8.094763    9.507883
<br />   8.238613    7.151999    8.446079   10.378986    9.100535    9.020674
<br />   9.0929      9.337885    8.210714    7.774472    7.9873495   9.852251
<br />   8.117089    9.415473    8.683429    8.120157    9.436607    7.553221
<br />  10.203923    8.010146   10.164826    9.050917    6.2404838   7.08705
<br />   9.235532    9.846936    9.154498    8.976049    8.864508   10.151257
<br />   0.37931287  1.5821621   1.7291744   0.74629974  1.0939419   0.3412478
<br />   1.2252617   0.92781615  0.28989732  0.4375671   0.54128814  1.1636494
<br />   1.0067962   0.13681132  0.50430614  1.5918033   0.3527609   1.263496
<br />   0.9879512   0.37240505  0.6783429   0.7948128   0.1901288   0.62441593
<br />   0.261106    2.0681515   2.3021426   1.1419712   0.18453968  0.31551838
<br />   2.1632304   1.1723582   0.9739594   0.5522996   0.5876693   3.1240067
<br />   1.9166212   1.8587621   3.5689602   1.7307425   2.1035736   0.26056302
<br />   0.3438216   0.59699166  0.91609055  2.135863    0.25372422  0.4649003
<br />   2.3946533   1.2871084   0.7849767   0.66078246  0.4094088   1.4407511
<br />   2.6428642   1.696549    1.4290481   1.8345678   1.8933747   3.0786533
<br />  -9.203916   10.9817705  -4.923316  ]]
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-19 13:02:46.749215
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   93.0968
<br />metric_name                                  mean_absolute_error
<br />Name: 0, dtype: object 
<br />
<br />  date_run                              2020-05-19 13:02:46.753378
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   8686.84
<br />metric_name                                   mean_squared_error
<br />Name: 1, dtype: object 
<br />
<br />  date_run                              2020-05-19 13:02:46.756806
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   93.6946
<br />metric_name                                median_absolute_error
<br />Name: 2, dtype: object 
<br />
<br />  date_run                              2020-05-19 13:02:46.759983
<br />model_uri                                   model_keras.armdn.py
<br />json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  -776.968
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
<br />  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 
<br />
<br />  #### Setup Model   ############################################## 
<br />| N-Beats
<br />| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139732273645832
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139729992523056
<br />     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139729992523560
<br />| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139729992065376
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139729992065880
<br />     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139729992066384
<br />
<br />  #### Fit  ####################################################### 
<br />>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f1623bb4e10> <class 'mlmodels.model_tch.nbeats.Model'>
<br />[[0.40504701]
<br /> [0.40695405]
<br /> [0.39710839]
<br /> ...
<br /> [0.93587014]
<br /> [0.95086039]
<br /> [0.95547277]]
<br />--- fiting ---
<br />grad_step = 000000, loss = 0.447400
<br />plot()
<br />Saved image to .//n_beats_0.png.
<br />grad_step = 000001, loss = 0.417736
<br />grad_step = 000002, loss = 0.389349
<br />grad_step = 000003, loss = 0.356038
<br />grad_step = 000004, loss = 0.318239
<br />grad_step = 000005, loss = 0.281246
<br />grad_step = 000006, loss = 0.256702
<br />grad_step = 000007, loss = 0.250246
<br />grad_step = 000008, loss = 0.235519
<br />grad_step = 000009, loss = 0.222405
<br />grad_step = 000010, loss = 0.209093
<br />grad_step = 000011, loss = 0.194697
<br />grad_step = 000012, loss = 0.183371
<br />grad_step = 000013, loss = 0.174424
<br />grad_step = 000014, loss = 0.165765
<br />grad_step = 000015, loss = 0.157460
<br />grad_step = 000016, loss = 0.149432
<br />grad_step = 000017, loss = 0.140354
<br />grad_step = 000018, loss = 0.130437
<br />grad_step = 000019, loss = 0.121231
<br />grad_step = 000020, loss = 0.113823
<br />grad_step = 000021, loss = 0.107673
<br />grad_step = 000022, loss = 0.101639
<br />grad_step = 000023, loss = 0.095216
<br />grad_step = 000024, loss = 0.088480
<br />grad_step = 000025, loss = 0.081877
<br />grad_step = 000026, loss = 0.076177
<br />grad_step = 000027, loss = 0.071465
<br />grad_step = 000028, loss = 0.067009
<br />grad_step = 000029, loss = 0.062497
<br />grad_step = 000030, loss = 0.058213
<br />grad_step = 000031, loss = 0.054256
<br />grad_step = 000032, loss = 0.050463
<br />grad_step = 000033, loss = 0.046858
<br />grad_step = 000034, loss = 0.043595
<br />grad_step = 000035, loss = 0.040739
<br />grad_step = 000036, loss = 0.038088
<br />grad_step = 000037, loss = 0.035375
<br />grad_step = 000038, loss = 0.032654
<br />grad_step = 000039, loss = 0.030194
<br />grad_step = 000040, loss = 0.028109
<br />grad_step = 000041, loss = 0.026256
<br />grad_step = 000042, loss = 0.024443
<br />grad_step = 000043, loss = 0.022671
<br />grad_step = 000044, loss = 0.021022
<br />grad_step = 000045, loss = 0.019473
<br />grad_step = 000046, loss = 0.018016
<br />grad_step = 000047, loss = 0.016730
<br />grad_step = 000048, loss = 0.015617
<br />grad_step = 000049, loss = 0.014552
<br />grad_step = 000050, loss = 0.013462
<br />grad_step = 000051, loss = 0.012418
<br />grad_step = 000052, loss = 0.011508
<br />grad_step = 000053, loss = 0.010723
<br />grad_step = 000054, loss = 0.009996
<br />grad_step = 000055, loss = 0.009301
<br />grad_step = 000056, loss = 0.008648
<br />grad_step = 000057, loss = 0.008035
<br />grad_step = 000058, loss = 0.007457
<br />grad_step = 000059, loss = 0.006938
<br />grad_step = 000060, loss = 0.006489
<br />grad_step = 000061, loss = 0.006075
<br />grad_step = 000062, loss = 0.005671
<br />grad_step = 000063, loss = 0.005291
<br />grad_step = 000064, loss = 0.004947
<br />grad_step = 000065, loss = 0.004644
<br />grad_step = 000066, loss = 0.004377
<br />grad_step = 000067, loss = 0.004133
<br />grad_step = 000068, loss = 0.003907
<br />grad_step = 000069, loss = 0.003697
<br />grad_step = 000070, loss = 0.003503
<br />grad_step = 000071, loss = 0.003335
<br />grad_step = 000072, loss = 0.003194
<br />grad_step = 000073, loss = 0.003068
<br />grad_step = 000074, loss = 0.002948
<br />grad_step = 000075, loss = 0.002839
<br />grad_step = 000076, loss = 0.002745
<br />grad_step = 000077, loss = 0.002667
<br />grad_step = 000078, loss = 0.002599
<br />grad_step = 000079, loss = 0.002538
<br />grad_step = 000080, loss = 0.002485
<br />grad_step = 000081, loss = 0.002439
<br />grad_step = 000082, loss = 0.002401
<br />grad_step = 000083, loss = 0.002370
<br />grad_step = 000084, loss = 0.002343
<br />grad_step = 000085, loss = 0.002319
<br />grad_step = 000086, loss = 0.002301
<br />grad_step = 000087, loss = 0.002288
<br />grad_step = 000088, loss = 0.002277
<br />grad_step = 000089, loss = 0.002267
<br />grad_step = 000090, loss = 0.002260
<br />grad_step = 000091, loss = 0.002254
<br />grad_step = 000092, loss = 0.002250
<br />grad_step = 000093, loss = 0.002248
<br />grad_step = 000094, loss = 0.002245
<br />grad_step = 000095, loss = 0.002243
<br />grad_step = 000096, loss = 0.002242
<br />grad_step = 000097, loss = 0.002241
<br />grad_step = 000098, loss = 0.002240
<br />grad_step = 000099, loss = 0.002239
<br />grad_step = 000100, loss = 0.002238
<br />plot()
<br />Saved image to .//n_beats_100.png.
<br />grad_step = 000101, loss = 0.002237
<br />grad_step = 000102, loss = 0.002236
<br />grad_step = 000103, loss = 0.002234
<br />grad_step = 000104, loss = 0.002232
<br />grad_step = 000105, loss = 0.002230
<br />grad_step = 000106, loss = 0.002228
<br />grad_step = 000107, loss = 0.002226
<br />grad_step = 000108, loss = 0.002223
<br />grad_step = 000109, loss = 0.002220
<br />grad_step = 000110, loss = 0.002217
<br />grad_step = 000111, loss = 0.002214
<br />grad_step = 000112, loss = 0.002211
<br />grad_step = 000113, loss = 0.002208
<br />grad_step = 000114, loss = 0.002204
<br />grad_step = 000115, loss = 0.002201
<br />grad_step = 000116, loss = 0.002197
<br />grad_step = 000117, loss = 0.002194
<br />grad_step = 000118, loss = 0.002190
<br />grad_step = 000119, loss = 0.002187
<br />grad_step = 000120, loss = 0.002184
<br />grad_step = 000121, loss = 0.002180
<br />grad_step = 000122, loss = 0.002177
<br />grad_step = 000123, loss = 0.002174
<br />grad_step = 000124, loss = 0.002171
<br />grad_step = 000125, loss = 0.002168
<br />grad_step = 000126, loss = 0.002165
<br />grad_step = 000127, loss = 0.002162
<br />grad_step = 000128, loss = 0.002159
<br />grad_step = 000129, loss = 0.002156
<br />grad_step = 000130, loss = 0.002153
<br />grad_step = 000131, loss = 0.002151
<br />grad_step = 000132, loss = 0.002148
<br />grad_step = 000133, loss = 0.002145
<br />grad_step = 000134, loss = 0.002143
<br />grad_step = 000135, loss = 0.002140
<br />grad_step = 000136, loss = 0.002137
<br />grad_step = 000137, loss = 0.002135
<br />grad_step = 000138, loss = 0.002132
<br />grad_step = 000139, loss = 0.002129
<br />grad_step = 000140, loss = 0.002127
<br />grad_step = 000141, loss = 0.002124
<br />grad_step = 000142, loss = 0.002121
<br />grad_step = 000143, loss = 0.002119
<br />grad_step = 000144, loss = 0.002116
<br />grad_step = 000145, loss = 0.002113
<br />grad_step = 000146, loss = 0.002110
<br />grad_step = 000147, loss = 0.002107
<br />grad_step = 000148, loss = 0.002105
<br />grad_step = 000149, loss = 0.002102
<br />grad_step = 000150, loss = 0.002099
<br />grad_step = 000151, loss = 0.002096
<br />grad_step = 000152, loss = 0.002093
<br />grad_step = 000153, loss = 0.002090
<br />grad_step = 000154, loss = 0.002087
<br />grad_step = 000155, loss = 0.002084
<br />grad_step = 000156, loss = 0.002080
<br />grad_step = 000157, loss = 0.002078
<br />grad_step = 000158, loss = 0.002076
<br />grad_step = 000159, loss = 0.002075
<br />grad_step = 000160, loss = 0.002072
<br />grad_step = 000161, loss = 0.002065
<br />grad_step = 000162, loss = 0.002062
<br />grad_step = 000163, loss = 0.002061
<br />grad_step = 000164, loss = 0.002057
<br />grad_step = 000165, loss = 0.002052
<br />grad_step = 000166, loss = 0.002050
<br />grad_step = 000167, loss = 0.002047
<br />grad_step = 000168, loss = 0.002043
<br />grad_step = 000169, loss = 0.002039
<br />grad_step = 000170, loss = 0.002036
<br />grad_step = 000171, loss = 0.002032
<br />grad_step = 000172, loss = 0.002029
<br />grad_step = 000173, loss = 0.002025
<br />grad_step = 000174, loss = 0.002021
<br />grad_step = 000175, loss = 0.002018
<br />grad_step = 000176, loss = 0.002015
<br />grad_step = 000177, loss = 0.002011
<br />grad_step = 000178, loss = 0.002007
<br />grad_step = 000179, loss = 0.002002
<br />grad_step = 000180, loss = 0.001998
<br />grad_step = 000181, loss = 0.001995
<br />grad_step = 000182, loss = 0.001991
<br />grad_step = 000183, loss = 0.001987
<br />grad_step = 000184, loss = 0.001983
<br />grad_step = 000185, loss = 0.001980
<br />grad_step = 000186, loss = 0.001977
<br />grad_step = 000187, loss = 0.001973
<br />grad_step = 000188, loss = 0.001969
<br />grad_step = 000189, loss = 0.001965
<br />grad_step = 000190, loss = 0.001961
<br />grad_step = 000191, loss = 0.001956
<br />grad_step = 000192, loss = 0.001950
<br />grad_step = 000193, loss = 0.001945
<br />grad_step = 000194, loss = 0.001941
<br />grad_step = 000195, loss = 0.001937
<br />grad_step = 000196, loss = 0.001933
<br />grad_step = 000197, loss = 0.001929
<br />grad_step = 000198, loss = 0.001925
<br />grad_step = 000199, loss = 0.001923
<br />grad_step = 000200, loss = 0.001925
<br />plot()
<br />Saved image to .//n_beats_200.png.
<br />grad_step = 000201, loss = 0.001930
<br />grad_step = 000202, loss = 0.001930
<br />grad_step = 000203, loss = 0.001917
<br />grad_step = 000204, loss = 0.001899
<br />grad_step = 000205, loss = 0.001894
<br />grad_step = 000206, loss = 0.001899
<br />grad_step = 000207, loss = 0.001897
<br />grad_step = 000208, loss = 0.001886
<br />grad_step = 000209, loss = 0.001876
<br />grad_step = 000210, loss = 0.001876
<br />grad_step = 000211, loss = 0.001876
<br />grad_step = 000212, loss = 0.001868
<br />grad_step = 000213, loss = 0.001857
<br />grad_step = 000214, loss = 0.001852
<br />grad_step = 000215, loss = 0.001852
<br />grad_step = 000216, loss = 0.001850
<br />grad_step = 000217, loss = 0.001843
<br />grad_step = 000218, loss = 0.001834
<br />grad_step = 000219, loss = 0.001829
<br />grad_step = 000220, loss = 0.001828
<br />grad_step = 000221, loss = 0.001834
<br />grad_step = 000222, loss = 0.001853
<br />grad_step = 000223, loss = 0.001880
<br />grad_step = 000224, loss = 0.001884
<br />grad_step = 000225, loss = 0.001846
<br />grad_step = 000226, loss = 0.001806
<br />grad_step = 000227, loss = 0.001809
<br />grad_step = 000228, loss = 0.001830
<br />grad_step = 000229, loss = 0.001823
<br />grad_step = 000230, loss = 0.001794
<br />grad_step = 000231, loss = 0.001788
<br />grad_step = 000232, loss = 0.001796
<br />grad_step = 000233, loss = 0.001790
<br />grad_step = 000234, loss = 0.001774
<br />grad_step = 000235, loss = 0.001771
<br />grad_step = 000236, loss = 0.001772
<br />grad_step = 000237, loss = 0.001760
<br />grad_step = 000238, loss = 0.001751
<br />grad_step = 000239, loss = 0.001750
<br />grad_step = 000240, loss = 0.001751
<br />grad_step = 000241, loss = 0.001740
<br />grad_step = 000242, loss = 0.001727
<br />grad_step = 000243, loss = 0.001723
<br />grad_step = 000244, loss = 0.001725
<br />grad_step = 000245, loss = 0.001721
<br />grad_step = 000246, loss = 0.001710
<br />grad_step = 000247, loss = 0.001700
<br />grad_step = 000248, loss = 0.001694
<br />grad_step = 000249, loss = 0.001691
<br />grad_step = 000250, loss = 0.001686
<br />grad_step = 000251, loss = 0.001679
<br />grad_step = 000252, loss = 0.001673
<br />grad_step = 000253, loss = 0.001670
<br />grad_step = 000254, loss = 0.001667
<br />grad_step = 000255, loss = 0.001663
<br />grad_step = 000256, loss = 0.001657
<br />grad_step = 000257, loss = 0.001650
<br />grad_step = 000258, loss = 0.001643
<br />grad_step = 000259, loss = 0.001638
<br />grad_step = 000260, loss = 0.001634
<br />grad_step = 000261, loss = 0.001631
<br />grad_step = 000262, loss = 0.001625
<br />grad_step = 000263, loss = 0.001617
<br />grad_step = 000264, loss = 0.001606
<br />grad_step = 000265, loss = 0.001593
<br />grad_step = 000266, loss = 0.001580
<br />grad_step = 000267, loss = 0.001569
<br />grad_step = 000268, loss = 0.001560
<br />grad_step = 000269, loss = 0.001552
<br />grad_step = 000270, loss = 0.001545
<br />grad_step = 000271, loss = 0.001538
<br />grad_step = 000272, loss = 0.001532
<br />grad_step = 000273, loss = 0.001527
<br />grad_step = 000274, loss = 0.001526
<br />grad_step = 000275, loss = 0.001539
<br />grad_step = 000276, loss = 0.001593
<br />grad_step = 000277, loss = 0.001671
<br />grad_step = 000278, loss = 0.001791
<br />grad_step = 000279, loss = 0.001624
<br />grad_step = 000280, loss = 0.001507
<br />grad_step = 000281, loss = 0.001539
<br />grad_step = 000282, loss = 0.001595
<br />grad_step = 000283, loss = 0.001561
<br />grad_step = 000284, loss = 0.001480
<br />grad_step = 000285, loss = 0.001536
<br />grad_step = 000286, loss = 0.001579
<br />grad_step = 000287, loss = 0.001477
<br />grad_step = 000288, loss = 0.001491
<br />grad_step = 000289, loss = 0.001546
<br />grad_step = 000290, loss = 0.001468
<br />grad_step = 000291, loss = 0.001466
<br />grad_step = 000292, loss = 0.001517
<br />grad_step = 000293, loss = 0.001473
<br />grad_step = 000294, loss = 0.001441
<br />grad_step = 000295, loss = 0.001470
<br />grad_step = 000296, loss = 0.001461
<br />grad_step = 000297, loss = 0.001427
<br />grad_step = 000298, loss = 0.001430
<br />grad_step = 000299, loss = 0.001439
<br />grad_step = 000300, loss = 0.001428
<br />plot()
<br />Saved image to .//n_beats_300.png.
<br />grad_step = 000301, loss = 0.001408
<br />grad_step = 000302, loss = 0.001409
<br />grad_step = 000303, loss = 0.001420
<br />grad_step = 000304, loss = 0.001407
<br />grad_step = 000305, loss = 0.001392
<br />grad_step = 000306, loss = 0.001390
<br />grad_step = 000307, loss = 0.001394
<br />grad_step = 000308, loss = 0.001394
<br />grad_step = 000309, loss = 0.001383
<br />grad_step = 000310, loss = 0.001375
<br />grad_step = 000311, loss = 0.001379
<br />grad_step = 000312, loss = 0.001402
<br />grad_step = 000313, loss = 0.001444
<br />grad_step = 000314, loss = 0.001527
<br />grad_step = 000315, loss = 0.001575
<br />grad_step = 000316, loss = 0.001540
<br />grad_step = 000317, loss = 0.001402
<br />grad_step = 000318, loss = 0.001356
<br />grad_step = 000319, loss = 0.001422
<br />grad_step = 000320, loss = 0.001455
<br />grad_step = 000321, loss = 0.001396
<br />grad_step = 000322, loss = 0.001331
<br />grad_step = 000323, loss = 0.001368
<br />grad_step = 000324, loss = 0.001434
<br />grad_step = 000325, loss = 0.001391
<br />grad_step = 000326, loss = 0.001325
<br />grad_step = 000327, loss = 0.001323
<br />grad_step = 000328, loss = 0.001360
<br />grad_step = 000329, loss = 0.001349
<br />grad_step = 000330, loss = 0.001303
<br />grad_step = 000331, loss = 0.001298
<br />grad_step = 000332, loss = 0.001324
<br />grad_step = 000333, loss = 0.001322
<br />grad_step = 000334, loss = 0.001293
<br />grad_step = 000335, loss = 0.001271
<br />grad_step = 000336, loss = 0.001279
<br />grad_step = 000337, loss = 0.001294
<br />grad_step = 000338, loss = 0.001286
<br />grad_step = 000339, loss = 0.001264
<br />grad_step = 000340, loss = 0.001248
<br />grad_step = 000341, loss = 0.001250
<br />grad_step = 000342, loss = 0.001260
<br />grad_step = 000343, loss = 0.001261
<br />grad_step = 000344, loss = 0.001251
<br />grad_step = 000345, loss = 0.001242
<br />grad_step = 000346, loss = 0.001241
<br />grad_step = 000347, loss = 0.001256
<br />grad_step = 000348, loss = 0.001266
<br />grad_step = 000349, loss = 0.001276
<br />grad_step = 000350, loss = 0.001257
<br />grad_step = 000351, loss = 0.001239
<br />grad_step = 000352, loss = 0.001214
<br />grad_step = 000353, loss = 0.001199
<br />grad_step = 000354, loss = 0.001188
<br />grad_step = 000355, loss = 0.001179
<br />grad_step = 000356, loss = 0.001177
<br />grad_step = 000357, loss = 0.001181
<br />grad_step = 000358, loss = 0.001197
<br />grad_step = 000359, loss = 0.001213
<br />grad_step = 000360, loss = 0.001232
<br />grad_step = 000361, loss = 0.001225
<br />grad_step = 000362, loss = 0.001213
<br />grad_step = 000363, loss = 0.001180
<br />grad_step = 000364, loss = 0.001160
<br />grad_step = 000365, loss = 0.001149
<br />grad_step = 000366, loss = 0.001147
<br />grad_step = 000367, loss = 0.001143
<br />grad_step = 000368, loss = 0.001134
<br />grad_step = 000369, loss = 0.001123
<br />grad_step = 000370, loss = 0.001117
<br />grad_step = 000371, loss = 0.001122
<br />grad_step = 000372, loss = 0.001138
<br />grad_step = 000373, loss = 0.001167
<br />grad_step = 000374, loss = 0.001196
<br />grad_step = 000375, loss = 0.001231
<br />grad_step = 000376, loss = 0.001229
<br />grad_step = 000377, loss = 0.001218
<br />grad_step = 000378, loss = 0.001170
<br />grad_step = 000379, loss = 0.001132
<br />grad_step = 000380, loss = 0.001108
<br />grad_step = 000381, loss = 0.001103
<br />grad_step = 000382, loss = 0.001106
<br />grad_step = 000383, loss = 0.001095
<br />grad_step = 000384, loss = 0.001088
<br />grad_step = 000385, loss = 0.001088
<br />grad_step = 000386, loss = 0.001096
<br />grad_step = 000387, loss = 0.001089
<br />grad_step = 000388, loss = 0.001057
<br />grad_step = 000389, loss = 0.001029
<br />grad_step = 000390, loss = 0.001023
<br />grad_step = 000391, loss = 0.001036
<br />grad_step = 000392, loss = 0.001051
<br />grad_step = 000393, loss = 0.001056
<br />grad_step = 000394, loss = 0.001058
<br />grad_step = 000395, loss = 0.001060
<br />grad_step = 000396, loss = 0.001077
<br />grad_step = 000397, loss = 0.001115
<br />grad_step = 000398, loss = 0.001131
<br />grad_step = 000399, loss = 0.001141
<br />grad_step = 000400, loss = 0.001079
<br />plot()
<br />Saved image to .//n_beats_400.png.
<br />grad_step = 000401, loss = 0.001015
<br />grad_step = 000402, loss = 0.000983
<br />grad_step = 000403, loss = 0.000995
<br />grad_step = 000404, loss = 0.001019
<br />grad_step = 000405, loss = 0.001014
<br />grad_step = 000406, loss = 0.000997
<br />grad_step = 000407, loss = 0.000988
<br />grad_step = 000408, loss = 0.000991
<br />grad_step = 000409, loss = 0.000989
<br />grad_step = 000410, loss = 0.000971
<br />grad_step = 000411, loss = 0.000951
<br />grad_step = 000412, loss = 0.000939
<br />grad_step = 000413, loss = 0.000943
<br />grad_step = 000414, loss = 0.000956
<br />grad_step = 000415, loss = 0.000968
<br />grad_step = 000416, loss = 0.000978
<br />grad_step = 000417, loss = 0.000991
<br />grad_step = 000418, loss = 0.001015
<br />grad_step = 000419, loss = 0.001049
<br />grad_step = 000420, loss = 0.001075
<br />grad_step = 000421, loss = 0.001101
<br />grad_step = 000422, loss = 0.001054
<br />grad_step = 000423, loss = 0.000999
<br />grad_step = 000424, loss = 0.000942
<br />grad_step = 000425, loss = 0.000929
<br />grad_step = 000426, loss = 0.000946
<br />grad_step = 000427, loss = 0.000950
<br />grad_step = 000428, loss = 0.000929
<br />grad_step = 000429, loss = 0.000899
<br />grad_step = 000430, loss = 0.000894
<br />grad_step = 000431, loss = 0.000908
<br />grad_step = 000432, loss = 0.000919
<br />grad_step = 000433, loss = 0.000909
<br />grad_step = 000434, loss = 0.000886
<br />grad_step = 000435, loss = 0.000863
<br />grad_step = 000436, loss = 0.000857
<br />grad_step = 000437, loss = 0.000865
<br />grad_step = 000438, loss = 0.000874
<br />grad_step = 000439, loss = 0.000877
<br />grad_step = 000440, loss = 0.000869
<br />grad_step = 000441, loss = 0.000863
<br />grad_step = 000442, loss = 0.000866
<br />grad_step = 000443, loss = 0.000873
<br />grad_step = 000444, loss = 0.000891
<br />grad_step = 000445, loss = 0.000905
<br />grad_step = 000446, loss = 0.000923
<br />grad_step = 000447, loss = 0.000910
<br />grad_step = 000448, loss = 0.000880
<br />grad_step = 000449, loss = 0.000839
<br />grad_step = 000450, loss = 0.000817
<br />grad_step = 000451, loss = 0.000820
<br />grad_step = 000452, loss = 0.000828
<br />grad_step = 000453, loss = 0.000827
<br />grad_step = 000454, loss = 0.000816
<br />grad_step = 000455, loss = 0.000810
<br />grad_step = 000456, loss = 0.000811
<br />grad_step = 000457, loss = 0.000815
<br />grad_step = 000458, loss = 0.000814
<br />grad_step = 000459, loss = 0.000816
<br />grad_step = 000460, loss = 0.000820
<br />grad_step = 000461, loss = 0.000820
<br />grad_step = 000462, loss = 0.000812
<br />grad_step = 000463, loss = 0.000794
<br />grad_step = 000464, loss = 0.000776
<br />grad_step = 000465, loss = 0.000762
<br />grad_step = 000466, loss = 0.000757
<br />grad_step = 000467, loss = 0.000758
<br />grad_step = 000468, loss = 0.000763
<br />grad_step = 000469, loss = 0.000773
<br />grad_step = 000470, loss = 0.000787
<br />grad_step = 000471, loss = 0.000816
<br />grad_step = 000472, loss = 0.000862
<br />grad_step = 000473, loss = 0.000941
<br />grad_step = 000474, loss = 0.001030
<br />grad_step = 000475, loss = 0.001128
<br />grad_step = 000476, loss = 0.001046
<br />grad_step = 000477, loss = 0.000927
<br />grad_step = 000478, loss = 0.000809
<br />grad_step = 000479, loss = 0.000836
<br />grad_step = 000480, loss = 0.000888
<br />grad_step = 000481, loss = 0.000830
<br />grad_step = 000482, loss = 0.000743
<br />grad_step = 000483, loss = 0.000759
<br />grad_step = 000484, loss = 0.000822
<br />grad_step = 000485, loss = 0.000812
<br />grad_step = 000486, loss = 0.000731
<br />grad_step = 000487, loss = 0.000700
<br />grad_step = 000488, loss = 0.000743
<br />grad_step = 000489, loss = 0.000773
<br />grad_step = 000490, loss = 0.000741
<br />grad_step = 000491, loss = 0.000689
<br />grad_step = 000492, loss = 0.000687
<br />grad_step = 000493, loss = 0.000721
<br />grad_step = 000494, loss = 0.000724
<br />grad_step = 000495, loss = 0.000692
<br />grad_step = 000496, loss = 0.000665
<br />grad_step = 000497, loss = 0.000674
<br />grad_step = 000498, loss = 0.000695
<br />grad_step = 000499, loss = 0.000694
<br />grad_step = 000500, loss = 0.000679
<br />plot()
<br />Saved image to .//n_beats_500.png.
<br />grad_step = 000501, loss = 0.000674
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
<br />  date_run                              2020-05-19 13:03:07.585945
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.187912
<br />metric_name                                  mean_absolute_error
<br />Name: 4, dtype: object 
<br />
<br />  date_run                              2020-05-19 13:03:07.591084
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                 0.0919968
<br />metric_name                                   mean_squared_error
<br />Name: 5, dtype: object 
<br />
<br />  date_run                              2020-05-19 13:03:07.600331
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                  0.105072
<br />metric_name                                median_absolute_error
<br />Name: 6, dtype: object 
<br />
<br />  date_run                              2020-05-19 13:03:07.606138
<br />model_uri                                    model_tch.nbeats.py
<br />json           [{'forecast_length': 60, 'backcast_length': 10...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                 -0.397924
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
<br />  data_pars out_pars {'train_data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'date_col': 'Date', 'freq': 'D', 'col_Xinput': 'Close'} {'outpath': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_fb/fb_prophet/'} 
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
<br />>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f156c8aa780> <class 'mlmodels.model_gluon.fb_prophet.Model'>
<br />
<br />  #### Inference Need return ypred, ytrue ######################### 
<br />
<br />  ### Calculate Metrics    ######################################## 
<br />
<br />  date_run                              2020-05-19 13:03:23.233127
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.3339
<br />metric_name                                  mean_absolute_error
<br />Name: 8, dtype: object 
<br />
<br />  date_run                              2020-05-19 13:03:23.236521
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   215.367
<br />metric_name                                   mean_squared_error
<br />Name: 9, dtype: object 
<br />
<br />  date_run                              2020-05-19 13:03:23.239597
<br />model_uri                              model_gluon/fb_prophet.py
<br />json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
<br />dataset_uri    dataset/timeseries//HOBBIES_1_001_CA_1_validat...
<br />metric                                                   14.4309
<br />metric_name                                median_absolute_error
<br />Name: 10, dtype: object 
<br />
<br />  date_run                              2020-05-19 13:03:23.242652
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
<br />  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]} 
<br />
<br />  #### Setup Model   ############################################## 
<br />
<br />  {'model_pars': {'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'prediction_length': 60, 'freq': '1d', 'start': '', 'col_date': 'date', 'col_ytarget': ['Close'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} 'model_uri' 
<br />
<br />  benchmark file saved at https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/example/benchmark/ 
<br />
<br />                        date_run  ...            metric_name
<br />0   2020-05-19 13:02:46.749215  ...    mean_absolute_error
<br />1   2020-05-19 13:02:46.753378  ...     mean_squared_error
<br />2   2020-05-19 13:02:46.756806  ...  median_absolute_error
<br />3   2020-05-19 13:02:46.759983  ...               r2_score
<br />4   2020-05-19 13:03:07.585945  ...    mean_absolute_error
<br />5   2020-05-19 13:03:07.591084  ...     mean_squared_error
<br />6   2020-05-19 13:03:07.600331  ...  median_absolute_error
<br />7   2020-05-19 13:03:07.606138  ...               r2_score
<br />8   2020-05-19 13:03:23.233127  ...    mean_absolute_error
<br />9   2020-05-19 13:03:23.236521  ...     mean_squared_error
<br />10  2020-05-19 13:03:23.239597  ...  median_absolute_error
<br />11  2020-05-19 13:03:23.242652  ...               r2_score
<br />
<br />[12 rows x 6 columns] 



### Error 14, [Traceback at line 3252](https://github.com/arita37/mlmodels_store/blob/master/log_test_cli/log_cli.py#L3252)<br />3252..Traceback (most recent call last):
<br />  File "https://github.com/arita37/mlmodels/tree/4ef62130366793ae7b99c72786bb70d7d21ee062/mlmodels/benchmark.py", line 118, in benchmark_run
<br />    model_uri =  model_pars['model_uri']
<br />KeyError: 'model_uri'
