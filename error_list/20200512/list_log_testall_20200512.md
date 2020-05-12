## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-12-12-12_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py


### Error 1, [Traceback at line 37](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-12-12-12_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py#L37)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
<br />    from mlmodels.mode_tf.raw  import temporal_fusion_google
<br />ModuleNotFoundError: No module named 'mlmodels.mode_tf'



### Error 2, [Traceback at line 158](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-12-12-12_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py#L158)<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1365, in _do_call
<br />    return fn(*args)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1350, in _run_fn
<br />    target_list, run_metadata)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1443, in _call_tf_sessionrun
<br />    run_metadata)
<br />tensorflow.python.framework.errors_impl.NotFoundError: Key Variable not found in checkpoint
<br />	 [[{{node save_1/RestoreV2}}]]
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 3, [Traceback at line 170](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-12-12-12_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py#L170)<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1290, in restore
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
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
<br />    test(data_path="", pars_choice="test01", config_mode="test")
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
<br />    session = load(out_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
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



### Error 4, [Traceback at line 221](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-12-12-12_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py#L221)<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1300, in restore
<br />    names_to_keys = object_graph_key_mapping(save_path)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1618, in object_graph_key_mapping
<br />    object_graph_string = reader.get_tensor(trackable.OBJECT_GRAPH_PROTO_KEY)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/pywrap_tensorflow_internal.py", line 915, in get_tensor
<br />    return CheckpointReader_GetTensor(self, compat.as_bytes(tensor_str))
<br />tensorflow.python.framework.errors_impl.NotFoundError: Key _CHECKPOINTABLE_OBJECT_GRAPH not found in checkpoint
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 5, [Traceback at line 232](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-12-12-12_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py#L232)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
<br />    test(data_path="", pars_choice="test01", config_mode="test")
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
<br />    session = load(out_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
<br />    return load_tf(load_pars)
<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 477, in load_tf
<br />    saver.restore(sess,  full_name)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1306, in restore
<br />    err, "a Variable name or other graph key that is missing")
<br />tensorflow.python.framework.errors_impl.NotFoundError: Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
<br />
<br />Key Variable not found in checkpoint
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
<br />    test(data_path="", pars_choice="test01", config_mode="test")
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
<br />    session = load(out_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
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
<br />   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
<br />Fetching origin
<br />Already up to date.
<br />Logs
<br />README.md
<br />README_actions.md
<br />create_error_file.py
<br />create_github_issues.py
<br />error_list
<br />log_benchmark
<br />log_dataloader
<br />log_import
<br />log_json
<br />log_jupyter
<br />log_pullrequest
<br />log_test_cli
<br />log_testall
<br />test_jupyter
<br />[master 56dc098] ml_store
<br /> 1 file changed, 234 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   90b3fe3..56dc098  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  #### Path params   ########################################## 
<br />
<br />  #### Loading dataset   ############################################# 
<br />
<br />  #### Model init, fit   ############################################# 
<br />
<br />  #### save the trained model  ####################################### 
<br />
<br />  #### Predict   ##################################################### 
<br />[[ 6.81889336e-01 -1.15498263e+00  1.22895559e+00 -1.77632196e-01
<br />   9.98545187e-01 -1.51045638e+00 -2.75846063e-01  1.01120706e+00
<br />  -1.47656266e+00  1.30970591e+00]
<br /> [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
<br />  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
<br />   8.59870972e-01 -1.04906775e+00]
<br /> [ 8.57719529e-01  9.81122462e-02 -2.60466059e-01  1.06032751e+00
<br />  -1.39003042e+00 -1.71116766e+00  2.65642403e-01  1.65712464e+00
<br />   1.41767401e+00  4.45096710e-01]
<br /> [ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
<br />  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
<br />   7.18290736e-01 -5.30119800e-01]
<br /> [ 9.64572049e-01 -1.06793987e-01  1.12232832e+00  1.45142926e+00
<br />   1.21828168e+00 -6.18036848e-01  4.38166347e-01 -2.03720123e+00
<br />  -1.94258918e+00 -9.97019796e-01]
<br /> [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
<br />   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
<br />  -9.18440038e-01  1.46089238e+00]
<br /> [ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
<br />  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
<br />  -1.14870423e+00 -9.50909251e-01]
<br /> [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
<br />  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
<br />   1.12487742e-01 -3.62612088e-01]
<br /> [ 1.77547698e+00 -2.03394449e-01 -1.98837863e-01  2.42669441e-01
<br />   9.64350564e-01  2.01830179e-01 -5.45774168e-01  6.61020288e-01
<br />   1.79215821e+00 -7.00398505e-01]
<br /> [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
<br />   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
<br />   1.40925339e+00  9.66539250e-01]
<br /> [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
<br />   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
<br />   8.84908814e-01  2.76499305e-01]
<br /> [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
<br />   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
<br />  -1.30572692e+00 -8.61316361e-01]
<br /> [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
<br />   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
<br />  -1.39711730e-01 -2.22414029e-01]
<br /> [ 6.10942600e-01 -2.79099641e+00 -1.33520272e+00 -4.56117555e-01
<br />  -9.44959948e-01 -9.79890252e-01 -1.56993672e-01  6.92574348e-01
<br />  -4.78672356e-01 -1.06460122e-01]
<br /> [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
<br />   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
<br />   3.24274243e-01 -2.36436952e-01]
<br /> [ 1.06702918e+00 -4.29142278e-01  3.50167159e-01  1.20845633e+00
<br />   7.51480619e-01  1.11570180e+00 -4.79157099e-01  8.40861558e-01
<br />  -1.02887218e-01  1.71647264e-02]
<br /> [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
<br />  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
<br />  -6.87487393e-01  3.32107876e+00]
<br /> [ 1.24549398e+00 -7.22391905e-01  1.11813340e+00  1.09899633e+00
<br />   1.00277655e+00 -9.01634490e-01 -5.32234021e-01 -8.22467189e-01
<br />   7.21711292e-01  6.74396105e-01]
<br /> [ 9.47814113e-01 -1.13379204e+00  6.40985866e-01 -1.90548298e-01
<br />  -1.23912256e+00  2.33339126e-01 -3.16901197e-01  4.34998324e-01
<br />   9.10423603e-01  1.21987438e+00]
<br /> [ 5.69983848e-01 -5.33020326e-01 -1.75458969e-01 -1.42655542e+00
<br />   6.06604307e-01  1.76795995e+00 -1.15985185e-01 -4.75372875e-01
<br />   4.77610182e-01 -9.33914656e-01]
<br /> [ 1.66752297e+00  1.22372221e+00 -4.59930104e-01 -5.93679025e-02
<br />  -4.93856997e-01  1.44898940e+00 -1.18110317e+00 -4.77580855e-01
<br />   2.59999942e-02 -7.90799954e-01]
<br /> [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
<br />  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
<br />  -7.28537489e-01  7.22518992e-02]
<br /> [ 7.90323893e-01  1.61336137e+00 -2.09424782e+00 -3.74804687e-01
<br />   9.15884042e-01 -7.49969617e-01  3.10272288e-01  2.05462410e+00
<br />   5.34095368e-02 -2.28765829e-01]
<br /> [ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
<br />   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
<br />   1.10417433e+00 -4.22856961e-01]
<br /> [ 7.83440538e-01 -5.11884476e-02  8.24584625e-01 -7.25597119e-01
<br />   9.31717197e-01 -8.67768678e-01  3.03085711e+00 -1.35977326e-01
<br />  -7.97269785e-01  6.54580153e-01]]
<br />
<br />  #### metrics   ##################################################### 
<br />{}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<br />{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<br /><__main__.Model object at 0x7f1ce91a7e10>
<br />
<br />  #### Module init   ############################################ 
<br />
<br />  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  #### Path params   ########################################## 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f1ce91a7e80> 
<br />
<br />  #### Fit   ######################################################## 
<br />
<br />  #### Predict   #################################################### 
<br />[[ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
<br />  -0.31456386  0.3180143  -0.89027155 -1.29525789]
<br /> [ 1.02242019  1.85300949  0.64435367  0.14225137  1.15080755  0.51350548
<br />  -0.45994283  0.37245685 -0.1484898   0.37167029]
<br /> [ 1.64661853 -1.52568032 -0.6069984   0.79502609  1.08480038 -0.37443832
<br />   0.42952614  0.1340482   1.20205486  0.10622272]
<br /> [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
<br />   1.42361443  1.06897162  0.04037143  1.57546791]
<br /> [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
<br />   0.02186284  2.13782807 -0.785534    0.85328122]
<br /> [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
<br />  -0.01745495 -0.85749682 -0.93418184  0.95449567]
<br /> [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
<br />  -0.17961229  0.98495167  1.07125243 -0.59334375]
<br /> [ 0.55853873 -0.51634791 -0.51814555  0.3511169   0.82550695 -0.06877046
<br />  -0.9520621  -1.34776494  1.47073986 -1.4614036 ]
<br /> [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
<br />   0.39378773 -0.18306199 -0.2210289   0.58033011]
<br /> [ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
<br />  -1.09715436 -0.50897228 -0.16648595 -1.03918232]
<br /> [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
<br />   0.05031709  0.30981676  1.05132077  0.6065484 ]
<br /> [ 0.87226739 -2.51630386 -0.77507029 -0.59566788  1.02600767 -0.30912132
<br />   1.74643509  0.51093777  1.71066184  0.14164054]
<br /> [ 1.21619061 -0.01900052  0.86089124 -0.22676019 -1.36419132 -1.56450785
<br />   1.63169151  0.93125568  0.94980882 -0.88018906]
<br /> [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
<br />   0.12837699  0.63658341  1.40925339  0.96653925]
<br /> [ 1.01195228 -1.88141087  1.70018815  0.4972691  -0.91766462  0.2373327
<br />  -1.09033833 -2.14444405 -0.36956243  0.60878366]
<br /> [ 0.88838944  0.28299553  0.01795589  0.10803082 -0.84967187  0.02941762
<br />  -0.50397395 -0.13479313  1.04921829 -1.27046078]
<br /> [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
<br />  -0.27584606  1.01120706 -1.47656266  1.30970591]
<br /> [ 0.61363671  0.3166589   1.34710546 -1.89526695 -0.76045809  0.08972912
<br />  -0.32905155  0.41026575  0.85987097 -1.04906775]
<br /> [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
<br />  -1.19586151  0.75133724  0.14039544  1.91979229]
<br /> [ 0.89551051  0.92061512  0.79452824 -0.03536792  0.8780991   2.11060505
<br />  -1.02188594 -1.30653407  0.07638048 -1.87316098]
<br /> [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
<br />   0.28807817 -2.07680202  0.94750117 -0.30097615]
<br /> [ 0.78801845  0.30196005  0.70098212 -0.39468968 -1.20376927 -1.17181338
<br />   0.75539203  0.98401224 -0.55968142 -0.19893745]
<br /> [ 0.85877496  2.29371761 -1.47023709 -0.83001099 -0.67204982 -1.01951985
<br />   0.59921324 -0.21465384  1.02124813  0.60640394]
<br /> [ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
<br />  -0.1834002   1.16143998 -0.02752939  0.78002714]
<br /> [ 0.87874071 -0.01923163  0.31965694  0.15001628 -1.46662161  0.46353432
<br />  -0.89868319  0.39788042 -0.99601089  0.3181542 ]]
<br />None
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />
<br />  ############ Model preparation   ################################## 
<br />
<br />  #### Module init   ############################################ 
<br />
<br />  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  #### Path params   ########################################## 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  ############ Model fit   ########################################## 
<br />fit success None
<br />
<br />  ############ Prediction############################################ 
<br />[[ 7.73703613e-01  1.27852808e+00 -2.11416392e+00 -4.42229280e-01
<br />   1.06821044e+00  3.23527354e-01 -2.50644065e+00 -1.09991490e-01
<br />   8.54894544e-03 -4.11639163e-01]
<br /> [ 8.95510508e-01  9.20615118e-01  7.94528240e-01 -3.53679249e-02
<br />   8.78099103e-01  2.11060505e+00 -1.02188594e+00 -1.30653407e+00
<br />   7.63804802e-02 -1.87316098e+00]
<br /> [ 1.46893146e+00 -1.47115693e+00  5.85910431e-01 -8.30171895e-01
<br />   1.03345052e+00 -8.80577600e-01 -9.55425262e-01 -2.79097722e-01
<br />   1.62284909e+00  2.06578332e+00]
<br /> [ 4.41189807e-01  4.79852371e-01 -1.92003697e-01 -1.55269878e+00
<br />  -1.88873982e+00  5.78464420e-01  3.98598388e-01 -9.61263599e-01
<br />  -1.45832446e+00 -3.05376438e+00]
<br /> [ 8.98917161e-01  5.57439453e-01 -7.58067329e-01  1.81038744e-01
<br />   8.41467206e-01  1.10717545e+00  6.93366226e-01  1.44287693e+00
<br />  -5.39681562e-01 -8.08847196e-01]
<br /> [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
<br />  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
<br />  -7.66793627e-02  3.55717262e-01]
<br /> [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
<br />  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
<br />  -6.87487393e-01  3.32107876e+00]
<br /> [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
<br />  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
<br />  -1.41914878e+00  1.06898597e+00]
<br /> [ 1.22867367e+00  1.34373116e-01 -1.82420406e-01 -2.68371304e-01
<br />  -1.73963799e+00 -1.31675626e-01 -9.26871939e-01  1.01855247e+00
<br />   1.23055820e+00 -4.91125138e-01]
<br /> [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
<br />  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
<br />   1.05132077e+00  6.06548400e-01]
<br /> [ 1.17867274e+00 -5.99804531e-01 -6.94693595e-01  1.12341216e+00
<br />   1.17899425e+00  3.05267040e-01  1.33526763e-02  1.38877940e+00
<br />  -6.61344243e-01  6.21803504e-01]
<br /> [ 3.54133613e-01  2.11124755e-01  9.21450069e-01  1.65275673e-02
<br />   9.03945451e-01  1.77187720e-01  9.54250872e-02 -1.11647002e+00
<br />   8.09271010e-02  6.07501958e-02]
<br /> [ 1.06523311e+00 -6.64867767e-01  1.00806543e+00 -1.94504696e+00
<br />  -1.23017555e+00 -9.15424368e-01  3.37220938e-01  1.22515585e+00
<br />  -1.05354607e+00  7.85226920e-01]
<br /> [ 1.34740825e+00  7.33023232e-01  8.38634747e-01 -1.89881206e+00
<br />  -5.42459922e-01 -1.11711069e+00 -1.09715436e+00 -5.08972278e-01
<br />  -1.66485955e-01 -1.03918232e+00]
<br /> [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
<br />  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
<br />   1.89861649e-01  3.93109245e-01]
<br /> [ 6.21530991e-01 -1.50957268e+00 -1.01932039e-01 -1.08071069e+00
<br />  -1.13742855e+00  7.25474004e-01  7.98063795e-01 -3.91782562e-02
<br />  -2.28754171e-01  7.43356544e-01]
<br /> [ 5.58538729e-01 -5.16347909e-01 -5.18145552e-01  3.51116897e-01
<br />   8.25506954e-01 -6.87704631e-02 -9.52062101e-01 -1.34776494e+00
<br />   1.47073986e+00 -1.46140360e+00]
<br /> [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
<br />   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
<br />  -3.53409983e-01 -2.51674208e-01]
<br /> [ 1.21619061e+00 -1.90005215e-02  8.60891241e-01 -2.26760192e-01
<br />  -1.36419132e+00 -1.56450785e+00  1.63169151e+00  9.31255679e-01
<br />   9.49808815e-01 -8.80189065e-01]
<br /> [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
<br />   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
<br />   3.24274243e-01 -2.36436952e-01]
<br /> [ 7.00175710e-01  5.56073510e-01  8.96864073e-02  1.69380911e+00
<br />   8.82393314e-01  1.96869779e-01 -5.63788735e-01  1.69869255e-01
<br />  -1.16400797e+00 -6.01156801e-01]
<br /> [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
<br />   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
<br />  -1.30572692e+00 -8.61316361e-01]
<br /> [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
<br />   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
<br />   8.84908814e-01  2.76499305e-01]
<br /> [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
<br />  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
<br />  -7.28537489e-01  7.22518992e-02]
<br /> [ 1.27991386e+00 -8.71422066e-01 -3.24032329e-01 -8.64829941e-01
<br />  -9.68539694e-01  6.08749082e-01  5.07984337e-01  5.61638097e-01
<br />   1.51475038e+00 -1.51107661e+00]]
<br />None
<br />
<br />  ############ Save/ Load ############################################ 
<br />
<br />   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
<br />Fetching origin
<br />Already up to date.
<br />Logs
<br />README.md
<br />README_actions.md
<br />create_error_file.py
<br />create_github_issues.py
<br />error_list
<br />log_benchmark
<br />log_dataloader
<br />log_import
<br />log_json
<br />log_jupyter
<br />log_pullrequest
<br />log_test_cli
<br />log_testall
<br />test_jupyter
<br />[master e01d553] ml_store
<br /> 1 file changed, 296 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   56dc098..e01d553  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_sklearn.py 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  #### Path params   ########################################## 
<br />
<br />  #### Loading dataset   ############################################# 
<br />
<br />  #### Model init, fit   ############################################# 
<br />
<br />  #### save the trained model  ####################################### 
<br />
<br />  #### Predict   ##################################################### 
<br />
<br />  #### metrics   ##################################################### 
<br />{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}
<br />{'roc_auc_score': 1.0}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_sklearn/model.pkl'}
<br />{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_sklearn/model.pkl'}
<br />RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
<br />                       max_depth=4, max_features='auto', max_leaf_nodes=None,
<br />                       min_impurity_decrease=0.0, min_impurity_split=None,
<br />                       min_samples_leaf=1, min_samples_split=2,
<br />                       min_weight_fraction_leaf=0.0, n_estimators=10,
<br />                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
<br />                       warm_start=False)
<br />{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}
<br />{'roc_auc_score': 1.0}
<br />
<br />  #### Module init   ############################################ 
<br />
<br />  <module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  #### Path params   ########################################## 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f34bebcd908> 
<br />
<br />  #### Fit   ######################################################## 
<br />
<br />  #### Predict   #################################################### 
<br />None
<br />
<br />  #### Get  metrics   ################################################ 
<br />{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />
<br />  ############ Model preparation   ################################## 
<br />
<br />  #### Module init   ############################################ 
<br />
<br />  <module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  #### Path params   ########################################## 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  ############ Model fit   ########################################## 
<br />fit success None
<br />
<br />  ############ Prediction############################################ 
<br />None
<br />
<br />  ############ Save/ Load ############################################ 
<br />/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
<br />  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
<br />
<br />   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
<br />Fetching origin
<br />Already up to date.
<br />Logs
<br />README.md
<br />README_actions.md
<br />create_error_file.py
<br />create_github_issues.py
<br />error_list
<br />log_benchmark
<br />log_dataloader
<br />log_import
<br />log_json
<br />log_jupyter
<br />log_pullrequest
<br />log_test_cli
<br />log_testall
<br />test_jupyter
<br />[master 122d57c] ml_store
<br /> 1 file changed, 108 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   e01d553..122d57c  master -> master
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />
<br />  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pplm.py 
<br /> Generating text ... 
<br />= Prefix of sentence =
<br /><|endoftext|>The potato
<br />
<br /> Unperturbed generated text :
<br />
<br /><|endoftext|>The potato-shaped, potato-eating insect of modern times (Ophiocordyceps elegans) has a unique ability to adapt quickly to a wide range of environments. It is able to survive in many different environments, including the Arctic, deserts
<br />
<br /> Perturbed generated text :
<br />
<br /><|endoftext|>The potato bomb is nothing new. It's been on the news a lot since 9/11. In fact, since the bombing in Paris last November, a bomb has been detonated in every major European country in the European Union.
<br />
<br />The bomb in Brussels
<br />
<br />
<br />   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
<br />Fetching origin
<br />Already up to date.
<br />Logs
<br />README.md
<br />README_actions.md
<br />create_error_file.py
<br />create_github_issues.py
<br />error_list
<br />log_benchmark
<br />log_dataloader
<br />log_import
<br />log_json
<br />log_jupyter
<br />log_pullrequest
<br />log_test_cli
<br />log_testall
<br />test_jupyter
