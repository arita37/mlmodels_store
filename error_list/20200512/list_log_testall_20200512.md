## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-11-16-11_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py


### Error 1, [Traceback at line 37](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-11-16-11_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py#L37)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
<br />    from mlmodels.mode_tf.raw  import temporal_fusion_google
<br />ModuleNotFoundError: No module named 'mlmodels.mode_tf'



### Error 2, [Traceback at line 152](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-11-16-11_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py#L152)<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1365, in _do_call
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



### Error 3, [Traceback at line 164](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-11-16-11_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py#L164)<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1290, in restore
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



### Error 4, [Traceback at line 215](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-11-16-11_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py#L215)<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1300, in restore
<br />    names_to_keys = object_graph_key_mapping(save_path)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1618, in object_graph_key_mapping
<br />    object_graph_string = reader.get_tensor(trackable.OBJECT_GRAPH_PROTO_KEY)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/pywrap_tensorflow_internal.py", line 915, in get_tensor
<br />    return CheckpointReader_GetTensor(self, compat.as_bytes(tensor_str))
<br />tensorflow.python.framework.errors_impl.NotFoundError: Key _CHECKPOINTABLE_OBJECT_GRAPH not found in checkpoint
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 5, [Traceback at line 226](https://github.com/arita37/mlmodels_store/blob/master/log_testall/log_testall_2020-05-11-16-11_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py#L226)<br />  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
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
<br />[master 951d2c6] ml_store
<br /> 1 file changed, 234 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   caf7b5a..951d2c6  master -> master
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
<br />[[ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
<br />  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
<br />  -4.26428618e-01 -8.39988915e-01]
<br /> [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
<br />  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
<br />  -2.24235772e-01  1.32247779e-01]
<br /> [ 1.24549398e+00 -7.22391905e-01  1.11813340e+00  1.09899633e+00
<br />   1.00277655e+00 -9.01634490e-01 -5.32234021e-01 -8.22467189e-01
<br />   7.21711292e-01  6.74396105e-01]
<br /> [ 1.21619061e+00 -1.90005215e-02  8.60891241e-01 -2.26760192e-01
<br />  -1.36419132e+00 -1.56450785e+00  1.63169151e+00  9.31255679e-01
<br />   9.49808815e-01 -8.80189065e-01]
<br /> [ 8.95510508e-01  9.20615118e-01  7.94528240e-01 -3.53679249e-02
<br />   8.78099103e-01  2.11060505e+00 -1.02188594e+00 -1.30653407e+00
<br />   7.63804802e-02 -1.87316098e+00]
<br /> [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
<br />   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
<br />  -1.30572692e+00 -8.61316361e-01]
<br /> [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
<br />   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
<br />  -2.75293863e-02  7.80027135e-01]
<br /> [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
<br />  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
<br />   1.40395436e-01  1.91979229e+00]
<br /> [ 1.36586461e+00  3.95860270e+00  5.48129585e-01  6.48643644e-01
<br />   8.49176066e-01  1.07343294e-01  1.38631426e+00 -1.39881282e+00
<br />   8.17678188e-02 -1.63744959e+00]
<br /> [ 9.29250600e-01 -1.10657307e+00 -1.95816909e+00 -3.59224096e-01
<br />  -1.21258781e+00  5.05381903e-01  5.42645295e-01  1.21794090e+00
<br />  -1.94068096e+00  6.77807571e-01]
<br /> [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
<br />   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
<br />   4.95158611e-01  6.61681076e-01]
<br /> [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
<br />  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
<br />  -2.43250851e+00 -2.27729800e+00]
<br /> [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
<br />   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
<br />   9.39168744e-01  1.56263850e-01]
<br /> [ 7.88018455e-01  3.01960045e-01  7.00982122e-01 -3.94689681e-01
<br />  -1.20376927e+00 -1.17181338e+00  7.55392029e-01  9.84012237e-01
<br />  -5.59681422e-01 -1.98937450e-01]
<br /> [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
<br />  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
<br />  -3.69562425e-01  6.08783659e-01]
<br /> [ 6.81889336e-01 -1.15498263e+00  1.22895559e+00 -1.77632196e-01
<br />   9.98545187e-01 -1.51045638e+00 -2.75846063e-01  1.01120706e+00
<br />  -1.47656266e+00  1.30970591e+00]
<br /> [ 1.83829400e+00  5.02740882e-01  1.29101580e-01  1.55880554e+00
<br />   1.32551412e+00  1.09402696e-01  1.40754000e+00 -1.21974440e+00
<br />   2.44936865e+00  1.61694960e+00]
<br /> [ 5.58538729e-01 -5.16347909e-01 -5.18145552e-01  3.51116897e-01
<br />   8.25506954e-01 -6.87704631e-02 -9.52062101e-01 -1.34776494e+00
<br />   1.47073986e+00 -1.46140360e+00]
<br /> [ 1.46893146e+00 -1.47115693e+00  5.85910431e-01 -8.30171895e-01
<br />   1.03345052e+00 -8.80577600e-01 -9.55425262e-01 -2.79097722e-01
<br />   1.62284909e+00  2.06578332e+00]
<br /> [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
<br />   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
<br />  -1.39711730e-01 -2.22414029e-01]
<br /> [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
<br />   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
<br />  -3.53409983e-01 -2.51674208e-01]
<br /> [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
<br />   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
<br />   1.45810824e+00 -3.31283170e-01]
<br /> [ 6.67591795e-01 -4.52524973e-01 -6.05981321e-01  1.16128569e+00
<br />  -1.44620987e+00  1.06996554e+00  1.92381543e+00 -1.04553425e+00
<br />   3.55284507e-01  1.80358898e+00]
<br /> [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
<br />  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
<br />  -8.32395348e-01 -4.46699203e-01]
<br /> [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
<br />  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
<br />  -2.21028902e-01  5.80330113e-01]]
<br />
<br />  #### metrics   ##################################################### 
<br />{}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save/Load   ################################################### 
<br />{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<br />{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<br /><__main__.Model object at 0x7f6b34284eb8>
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
<br />  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f6b4e5f06a0> 
<br />
<br />  #### Fit   ######################################################## 
<br />
<br />  #### Predict   #################################################### 
<br />[[ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
<br />  -0.87728152 -0.78191168 -0.43750898 -1.44087602]
<br /> [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
<br />   1.42361443  1.06897162  0.04037143  1.57546791]
<br /> [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
<br />   1.10223673  0.81434313  0.78046993  1.10861676]
<br /> [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
<br />   0.05031709  0.30981676  1.05132077  0.6065484 ]
<br /> [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
<br />  -0.17961229  0.98495167  1.07125243 -0.59334375]
<br /> [ 0.87122579 -0.20975294 -0.45698786  0.93514778 -0.87353582  1.81252782
<br />   0.92550121  0.14010988 -1.41914878  1.06898597]
<br /> [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
<br />   1.62091229  0.68400133 -0.35340998 -0.25167421]
<br /> [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
<br />  -0.76730983  1.04008915 -0.91844004  1.46089238]
<br /> [ 1.37661405 -0.60022533  0.72591685 -0.37951752 -0.62754626 -1.01480369
<br />   0.96622086  0.4359862  -0.68748739  3.32107876]
<br /> [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
<br />   0.19975956  0.38560229  0.71829074 -0.5301198 ]
<br /> [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
<br />  -2.50644065 -0.10999149  0.00854895 -0.41163916]
<br /> [ 1.838294    0.50274088  0.12910158  1.55880554  1.32551412  0.1094027
<br />   1.40754    -1.2197444   2.44936865  1.6169496 ]
<br /> [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
<br />   0.12837699  0.63658341  1.40925339  0.96653925]
<br /> [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
<br />   0.50798434  0.5616381   1.51475038 -1.51107661]
<br /> [ 0.69174373  1.00978733 -1.21333813 -1.55694156 -1.20257258 -0.61244213
<br />  -2.69836174 -0.13935181 -0.72853749  0.0722519 ]
<br /> [ 1.02242019  1.85300949  0.64435367  0.14225137  1.15080755  0.51350548
<br />  -0.45994283  0.37245685 -0.1484898   0.37167029]
<br /> [ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
<br />   0.59003946  1.53920701 -1.14870423 -0.95090925]
<br /> [ 1.06702918 -0.42914228  0.35016716  1.20845633  0.75148062  1.1157018
<br />  -0.4791571   0.84086156 -0.10288722  0.01716473]
<br /> [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
<br />  -0.27584606  1.01120706 -1.47656266  1.30970591]
<br /> [ 0.78344054 -0.05118845  0.82458463 -0.72559712  0.9317172  -0.86776868
<br />   3.03085711 -0.13597733 -0.79726979  0.65458015]
<br /> [ 0.84806927  0.45194604  0.63019567 -1.57915629  0.82798737 -0.82862798
<br />  -0.10534471  0.52887975 -2.23708651 -0.4148469 ]
<br /> [ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
<br />  -0.3169012   0.43499832  0.9104236   1.21987438]
<br /> [ 0.44118981  0.47985237 -0.1920037  -1.55269878 -1.88873982  0.57846442
<br />   0.39859839 -0.9612636  -1.45832446 -3.05376438]
<br /> [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
<br />   1.92381543 -1.04553425  0.35528451  1.80358898]
<br /> [ 0.72297801  0.18553562  0.91549927  0.39442803 -0.84983074  0.72552256
<br />  -0.15050433  1.49588477  0.67545381 -0.43820027]]
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
<br />[[ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
<br />   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
<br />  -1.30572692e+00 -8.61316361e-01]
<br /> [ 8.57719529e-01  9.81122462e-02 -2.60466059e-01  1.06032751e+00
<br />  -1.39003042e+00 -1.71116766e+00  2.65642403e-01  1.65712464e+00
<br />   1.41767401e+00  4.45096710e-01]
<br /> [ 7.83440538e-01 -5.11884476e-02  8.24584625e-01 -7.25597119e-01
<br />   9.31717197e-01 -8.67768678e-01  3.03085711e+00 -1.35977326e-01
<br />  -7.97269785e-01  6.54580153e-01]
<br /> [ 6.81889336e-01 -1.15498263e+00  1.22895559e+00 -1.77632196e-01
<br />   9.98545187e-01 -1.51045638e+00 -2.75846063e-01  1.01120706e+00
<br />  -1.47656266e+00  1.30970591e+00]
<br /> [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
<br />   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
<br />   4.95158611e-01  6.61681076e-01]
<br /> [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
<br />  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
<br />  -2.24235772e-01  1.32247779e-01]
<br /> [ 1.46893146e+00 -1.47115693e+00  5.85910431e-01 -8.30171895e-01
<br />   1.03345052e+00 -8.80577600e-01 -9.55425262e-01 -2.79097722e-01
<br />   1.62284909e+00  2.06578332e+00]
<br /> [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
<br />   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
<br />   8.84908814e-01  2.76499305e-01]
<br /> [ 7.73703613e-01  1.27852808e+00 -2.11416392e+00 -4.42229280e-01
<br />   1.06821044e+00  3.23527354e-01 -2.50644065e+00 -1.09991490e-01
<br />   8.54894544e-03 -4.11639163e-01]
<br /> [ 1.16755486e+00  3.53600971e-02  7.14789597e-01 -1.53879325e+00
<br />   1.10863359e+00 -4.47895185e-01 -1.75592564e+00  6.17985534e-01
<br />  -1.84176326e-01  8.52704062e-01]
<br /> [ 7.00175710e-01  5.56073510e-01  8.96864073e-02  1.69380911e+00
<br />   8.82393314e-01  1.96869779e-01 -5.63788735e-01  1.69869255e-01
<br />  -1.16400797e+00 -6.01156801e-01]
<br /> [ 9.47814113e-01 -1.13379204e+00  6.40985866e-01 -1.90548298e-01
<br />  -1.23912256e+00  2.33339126e-01 -3.16901197e-01  4.34998324e-01
<br />   9.10423603e-01  1.21987438e+00]
<br /> [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
<br />   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
<br />  -9.18440038e-01  1.46089238e+00]
<br /> [ 1.21619061e+00 -1.90005215e-02  8.60891241e-01 -2.26760192e-01
<br />  -1.36419132e+00 -1.56450785e+00  1.63169151e+00  9.31255679e-01
<br />   9.49808815e-01 -8.80189065e-01]
<br /> [ 6.92114488e-01 -6.06524918e-02  2.05635552e+00 -2.41350300e+00
<br />   1.17456965e+00 -1.77756638e+00 -2.81736269e-01 -7.77858827e-01
<br />   1.11584111e+00  1.76024923e+00]
<br /> [ 1.17867274e+00 -5.99804531e-01 -6.94693595e-01  1.12341216e+00
<br />   1.17899425e+00  3.05267040e-01  1.33526763e-02  1.38877940e+00
<br />  -6.61344243e-01  6.21803504e-01]
<br /> [ 1.36586461e+00  3.95860270e+00  5.48129585e-01  6.48643644e-01
<br />   8.49176066e-01  1.07343294e-01  1.38631426e+00 -1.39881282e+00
<br />   8.17678188e-02 -1.63744959e+00]
<br /> [ 8.88838813e-01  1.03368687e+00 -4.97025792e-02  8.08844360e-01
<br />   8.14051347e-01  1.78975468e+00  1.14690038e+00  4.51284016e-01
<br />  -1.68405999e+00  4.66643267e-01]
<br /> [ 8.95510508e-01  9.20615118e-01  7.94528240e-01 -3.53679249e-02
<br />   8.78099103e-01  2.11060505e+00 -1.02188594e+00 -1.30653407e+00
<br />   7.63804802e-02 -1.87316098e+00]
<br /> [ 1.34728643e+00 -3.64538050e-01  8.07509886e-02 -4.59717681e-01
<br />  -8.89487596e-01  1.70548352e+00  9.49961101e-02  2.40505552e-01
<br />  -9.99426501e-01 -7.67803746e-01]
<br /> [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
<br />   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
<br />  -1.39711730e-01 -2.22414029e-01]
<br /> [ 8.88389445e-01  2.82995534e-01  1.79558917e-02  1.08030817e-01
<br />  -8.49671873e-01  2.94176190e-02 -5.03973949e-01 -1.34793129e-01
<br />   1.04921829e+00 -1.27046078e+00]
<br /> [ 6.21530991e-01 -1.50957268e+00 -1.01932039e-01 -1.08071069e+00
<br />  -1.13742855e+00  7.25474004e-01  7.98063795e-01 -3.91782562e-02
<br />  -2.28754171e-01  7.43356544e-01]
<br /> [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
<br />  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
<br />  -3.69562425e-01  6.08783659e-01]
<br /> [ 5.69983848e-01 -5.33020326e-01 -1.75458969e-01 -1.42655542e+00
<br />   6.06604307e-01  1.76795995e+00 -1.15985185e-01 -4.75372875e-01
<br />   4.77610182e-01 -9.33914656e-01]]
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
<br />[master 9c0bc1b] ml_store
<br /> 1 file changed, 296 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   951d2c6..9c0bc1b  master -> master
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
<br />  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7ff2ba647978> 
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
<br />[master a2b49bf] ml_store
<br /> 1 file changed, 108 insertions(+)
<br />To github.com:arita37/mlmodels_store.git
<br />   9c0bc1b..a2b49bf  master -> master
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
<br />Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
<br />From github.com:arita37/mlmodels_store
<br />   a2b49bf..ea14a7e  master     -> origin/master
<br />Updating a2b49bf..ea14a7e
<br />Fast-forward
<br /> ...-11_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py | 617 +++++++++++++++++++++
<br /> 1 file changed, 617 insertions(+)
<br /> create mode 100644 log_pullrequest/log_pr_2020-05-11-16-11_1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5.py
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
