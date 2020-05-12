
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_tf.temporal_fusion_google', 'model_tf.1_lstm', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_tch.pplm', 'model_tch.transformer_sentence', 'model_tch.nbeats', 'model_tch.textcnn', 'model_tch.mlp', 'model_tch.pytorch_vae', 'model_tch.03_nbeats_dataloader', 'model_tch.torchhub', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_gluon.gluonts_model', 'model_gluon.fb_prophet', 'model_gluon.gluon_automl', 'model_keras.charcnn', 'model_keras.charcnn_zhang', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.01_deepctr', 'model_keras.namentity_crm_bilstm', 'model_keras.nbeats', 'model_keras.textcnn', 'model_keras.keras_gan', 'model_keras.textcnn_dataloader', 'model_keras.02_cnn', 'model_keras.Autokeras', 'model_keras.armdn', 'model_keras.textvae'] 

  Used ['model_tf.temporal_fusion_google', 'model_tf.1_lstm', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_tch.pplm', 'model_tch.transformer_sentence', 'model_tch.nbeats', 'model_tch.textcnn', 'model_tch.mlp', 'model_tch.pytorch_vae', 'model_tch.03_nbeats_dataloader', 'model_tch.torchhub', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_gluon.gluonts_model', 'model_gluon.fb_prophet', 'model_gluon.gluon_automl', 'model_keras.charcnn', 'model_keras.charcnn_zhang', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.01_deepctr', 'model_keras.namentity_crm_bilstm', 'model_keras.nbeats', 'model_keras.textcnn', 'model_keras.keras_gan', 'model_keras.textcnn_dataloader', 'model_keras.02_cnn', 'model_keras.Autokeras', 'model_keras.armdn', 'model_keras.textvae'] 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
    from mlmodels.mode_tf.raw  import temporal_fusion_google
ModuleNotFoundError: No module named 'mlmodels.mode_tf'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   80a182d..d327a88  master     -> origin/master
Updating 80a182d..d327a88
Fast-forward
 ...-09_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py | 167 ++++++
 ...-13_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py | 624 +++++++++++++++++++++
 2 files changed, 791 insertions(+)
 create mode 100644 log_dataloader/log_2020-05-12-00-09_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py
 create mode 100644 log_pullrequest/log_pr_2020-05-12-00-13_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master 071195f] ml_store
 1 file changed, 68 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-12-00-15_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py
To github.com:arita37/mlmodels_store.git
   d327a88..071195f  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
start

  #### Loading params   ############################################## 

  ############# Data, Params preparation   ################# 

  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'} 

  #### Loading dataset   ############################################# 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]

  #### Model init  ############################################# 

  #### Model fit   ############################################# 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]
          0         1         2         3         4         5
0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000

  #### Predict   ##################################################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]
          0         1         2         3         4         5
0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
5  0.745928  0.883387  0.838176  0.904464  0.904464  0.370110
6  1.000000  0.881878  0.467996  0.486496  0.486496  1.000000
7  0.216516  0.077549  0.433808  0.329598  0.329598  0.318466
8  0.195249  0.000000  0.000000  0.000000  0.000000  0.671960
9  0.000000  0.173783  0.369041  0.411721  0.411721  0.304384

  #### metrics   ##################################################### 
{'loss': 0.5461457967758179, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-12 00:15:52.670022: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1365, in _do_call
    return fn(*args)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1350, in _run_fn
    target_list, run_metadata)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1443, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.NotFoundError: Key Variable not found in checkpoint
	 [[{{node save_1/RestoreV2}}]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1290, in restore
    {self.saver_def.filename_tensor_name: save_path})
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 956, in run
    run_metadata_ptr)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1180, in _run
    feed_dict_tensor, options, run_metadata)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1359, in _do_run
    run_metadata)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1384, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.NotFoundError: Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
    test(data_path="", pars_choice="test01", config_mode="test")
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
    saver      = tf.compat.v1.train.Saver()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 828, in __init__
    self.build()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 840, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 878, in _build
    build_restore=build_restore)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 508, in _build_internal
    restore_sequentially, reshape)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 328, in _AddRestoreOps
    restore_sequentially)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 575, in bulk_restore
    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_io_ops.py", line 1696, in restore_v2
    name=name)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
    attrs, op_def, compute_device)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
    self._traceback = tf_stack.extract_stack()


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1300, in restore
    names_to_keys = object_graph_key_mapping(save_path)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1618, in object_graph_key_mapping
    object_graph_string = reader.get_tensor(trackable.OBJECT_GRAPH_PROTO_KEY)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/pywrap_tensorflow_internal.py", line 915, in get_tensor
    return CheckpointReader_GetTensor(self, compat.as_bytes(tensor_str))
tensorflow.python.framework.errors_impl.NotFoundError: Key _CHECKPOINTABLE_OBJECT_GRAPH not found in checkpoint

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
    test(data_path="", pars_choice="test01", config_mode="test")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
    session = load(out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 477, in load_tf
    saver.restore(sess,  full_name)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1306, in restore
    err, "a Variable name or other graph key that is missing")
tensorflow.python.framework.errors_impl.NotFoundError: Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
    test(data_path="", pars_choice="test01", config_mode="test")
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
    saver      = tf.compat.v1.train.Saver()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 828, in __init__
    self.build()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 840, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 878, in _build
    build_restore=build_restore)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 508, in _build_internal
    restore_sequentially, reshape)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 328, in _AddRestoreOps
    restore_sequentially)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 575, in bulk_restore
    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_io_ops.py", line 1696, in restore_v2
    name=name)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
    attrs, op_def, compute_device)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
    self._traceback = tf_stack.extract_stack()


   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master f3c75e0] ml_store
 1 file changed, 234 insertions(+)
To github.com:arita37/mlmodels_store.git
   071195f..f3c75e0  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 2.07582971e+00 -1.40232915e+00 -4.79184915e-01  4.51122939e-01
   1.03436581e+00 -6.94920901e-01 -4.18937898e-01  5.15413802e-01
  -1.11487105e+00 -1.95210529e+00]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 1.06523311e+00 -6.64867767e-01  1.00806543e+00 -1.94504696e+00
  -1.23017555e+00 -9.15424368e-01  3.37220938e-01  1.22515585e+00
  -1.05354607e+00  7.85226920e-01]
 [ 1.24549398e+00 -7.22391905e-01  1.11813340e+00  1.09899633e+00
   1.00277655e+00 -9.01634490e-01 -5.32234021e-01 -8.22467189e-01
   7.21711292e-01  6.74396105e-01]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]
 [ 1.36586461e+00  3.95860270e+00  5.48129585e-01  6.48643644e-01
   8.49176066e-01  1.07343294e-01  1.38631426e+00 -1.39881282e+00
   8.17678188e-02 -1.63744959e+00]
 [ 7.83440538e-01 -5.11884476e-02  8.24584625e-01 -7.25597119e-01
   9.31717197e-01 -8.67768678e-01  3.03085711e+00 -1.35977326e-01
  -7.97269785e-01  6.54580153e-01]
 [ 1.07258847e+00 -5.86523939e-01 -1.34267579e+00 -1.23685338e+00
   1.24328724e+00  8.75838928e-01 -3.26499498e-01  6.23362177e-01
  -4.34956683e-01  1.11438298e+00]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
   8.84908814e-01  2.76499305e-01]
 [ 1.77547698e+00 -2.03394449e-01 -1.98837863e-01  2.42669441e-01
   9.64350564e-01  2.01830179e-01 -5.45774168e-01  6.61020288e-01
   1.79215821e+00 -7.00398505e-01]
 [ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
  -1.14870423e+00 -9.50909251e-01]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
  -7.28537489e-01  7.22518992e-02]
 [ 1.58463774e+00  5.71209961e-02 -1.77183179e-02 -7.99547491e-01
   1.32970299e+00 -2.91594596e-01 -1.10771250e+00 -2.58982853e-01
   1.89293198e-01 -1.71939447e+00]
 [ 1.12062155e+00 -7.02920403e-01 -1.22957425e+00  7.25550518e-01
  -1.18013412e+00 -3.24204219e-01  1.10223673e+00  8.14343129e-01
   7.80469930e-01  1.10861676e+00]
 [ 1.16777676e+00 -6.65754518e-01 -1.23312074e+00 -1.67419581e+00
   1.01313574e+00  8.25029824e-01 -1.20464572e-01 -4.98213564e-01
  -3.10984978e-01 -1.18231813e+00]
 [ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
  -4.26428618e-01 -8.39988915e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 6.81889336e-01 -1.15498263e+00  1.22895559e+00 -1.77632196e-01
   9.98545187e-01 -1.51045638e+00 -2.75846063e-01  1.01120706e+00
  -1.47656266e+00  1.30970591e+00]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f2039f90eb8>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f2039f90f28> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
   1.10417433e+00 -4.22856961e-01]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 1.21619061e+00 -1.90005215e-02  8.60891241e-01 -2.26760192e-01
  -1.36419132e+00 -1.56450785e+00  1.63169151e+00  9.31255679e-01
   9.49808815e-01 -8.80189065e-01]
 [ 6.81889336e-01 -1.15498263e+00  1.22895559e+00 -1.77632196e-01
   9.98545187e-01 -1.51045638e+00 -2.75846063e-01  1.01120706e+00
  -1.47656266e+00  1.30970591e+00]
 [ 1.36586461e+00  3.95860270e+00  5.48129585e-01  6.48643644e-01
   8.49176066e-01  1.07343294e-01  1.38631426e+00 -1.39881282e+00
   8.17678188e-02 -1.63744959e+00]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 9.26869810e-01  3.92334911e-01 -4.23478297e-01  4.48380651e-01
  -1.09230828e+00  1.12532350e+00 -9.48439656e-01  1.04053390e-01
   5.28003422e-01  1.00796648e+00]
 [ 7.90323893e-01  1.61336137e+00 -2.09424782e+00 -3.74804687e-01
   9.15884042e-01 -7.49969617e-01  3.10272288e-01  2.05462410e+00
   5.34095368e-02 -2.28765829e-01]
 [ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 4.41189807e-01  4.79852371e-01 -1.92003697e-01 -1.55269878e+00
  -1.88873982e+00  5.78464420e-01  3.98598388e-01 -9.61263599e-01
  -1.45832446e+00 -3.05376438e+00]
 [ 7.88018455e-01  3.01960045e-01  7.00982122e-01 -3.94689681e-01
  -1.20376927e+00 -1.17181338e+00  7.55392029e-01  9.84012237e-01
  -5.59681422e-01 -1.98937450e-01]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
   1.40395436e-01  1.91979229e+00]
 [ 8.61462558e-01  7.43205537e-02 -1.34501002e+00 -1.99560718e-01
  -1.47533915e+00 -6.54603169e-01 -3.14563862e-01  3.18014296e-01
  -8.90271552e-01 -1.29525789e+00]
 [ 2.07582971e+00 -1.40232915e+00 -4.79184915e-01  4.51122939e-01
   1.03436581e+00 -6.94920901e-01 -4.18937898e-01  5.15413802e-01
  -1.11487105e+00 -1.95210529e+00]
 [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
  -2.43250851e+00 -2.27729800e+00]
 [ 7.73703613e-01  1.27852808e+00 -2.11416392e+00 -4.42229280e-01
   1.06821044e+00  3.23527354e-01 -2.50644065e+00 -1.09991490e-01
   8.54894544e-03 -4.11639163e-01]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 1.66752297e+00  1.22372221e+00 -4.59930104e-01 -5.93679025e-02
  -4.93856997e-01  1.44898940e+00 -1.18110317e+00 -4.77580855e-01
   2.59999942e-02 -7.90799954e-01]
 [ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
  -1.14870423e+00 -9.50909251e-01]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]]
None

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

  ############ Model preparation   ################################## 

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  ############ Model fit   ########################################## 
fit success None

  ############ Prediction############################################ 
[[ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 6.21530991e-01 -1.50957268e+00 -1.01932039e-01 -1.08071069e+00
  -1.13742855e+00  7.25474004e-01  7.98063795e-01 -3.91782562e-02
  -2.28754171e-01  7.43356544e-01]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]
 [ 3.54133613e-01  2.11124755e-01  9.21450069e-01  1.65275673e-02
   9.03945451e-01  1.77187720e-01  9.54250872e-02 -1.11647002e+00
   8.09271010e-02  6.07501958e-02]
 [ 1.66752297e+00  1.22372221e+00 -4.59930104e-01 -5.93679025e-02
  -4.93856997e-01  1.44898940e+00 -1.18110317e+00 -4.77580855e-01
   2.59999942e-02 -7.90799954e-01]
 [ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
  -1.14870423e+00 -9.50909251e-01]
 [ 1.18947778e+00 -6.80678141e-01 -5.68244809e-02 -8.45080274e-02
   8.21783210e-01 -2.97361883e-01 -1.86578994e-01  4.17302005e-01
   7.84770651e-01  4.92336556e-01]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
  -1.41914878e+00  1.06898597e+00]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 1.12062155e+00 -7.02920403e-01 -1.22957425e+00  7.25550518e-01
  -1.18013412e+00 -3.24204219e-01  1.10223673e+00  8.14343129e-01
   7.80469930e-01  1.10861676e+00]
 [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
  -7.28537489e-01  7.22518992e-02]
 [ 1.64661853e+00 -1.52568032e+00 -6.06998398e-01  7.95026094e-01
   1.08480038e+00 -3.74438319e-01  4.29526140e-01  1.34048197e-01
   1.20205486e+00  1.06222724e-01]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 1.46893146e+00 -1.47115693e+00  5.85910431e-01 -8.30171895e-01
   1.03345052e+00 -8.80577600e-01 -9.55425262e-01 -2.79097722e-01
   1.62284909e+00  2.06578332e+00]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 1.06702918e+00 -4.29142278e-01  3.50167159e-01  1.20845633e+00
   7.51480619e-01  1.11570180e+00 -4.79157099e-01  8.40861558e-01
  -1.02887218e-01  1.71647264e-02]
 [ 1.27991386e+00 -8.71422066e-01 -3.24032329e-01 -8.64829941e-01
  -9.68539694e-01  6.08749082e-01  5.07984337e-01  5.61638097e-01
   1.51475038e+00 -1.51107661e+00]
 [ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
   1.10417433e+00 -4.22856961e-01]
 [ 8.57719529e-01  9.81122462e-02 -2.60466059e-01  1.06032751e+00
  -1.39003042e+00 -1.71116766e+00  2.65642403e-01  1.65712464e+00
   1.41767401e+00  4.45096710e-01]]
None

  ############ Save/ Load ############################################ 

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
[master 2e90a74] ml_store
 1 file changed, 321 insertions(+)
To github.com:arita37/mlmodels_store.git
   f3c75e0..2e90a74  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_sklearn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 

  #### metrics   ##################################################### 
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}
{'roc_auc_score': 1.0}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_sklearn/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_sklearn/model.pkl'}
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=4, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}
{'roc_auc_score': 1.0}

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7fed701d3978> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
None

  #### Get  metrics   ################################################ 
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}

  #### Save   ######################################################## 

  #### Load   ######################################################## 

  ############ Model preparation   ################################## 

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  ############ Model fit   ########################################## 
fit success None

  ############ Prediction############################################ 
None

  ############ Save/ Load ############################################ 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
