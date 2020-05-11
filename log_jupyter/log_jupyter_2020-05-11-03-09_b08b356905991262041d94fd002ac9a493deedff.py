
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/b08b356905991262041d94fd002ac9a493deedff', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'b08b356905991262041d94fd002ac9a493deedff', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/b08b356905991262041d94fd002ac9a493deedff

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/b08b356905991262041d94fd002ac9a493deedff

 ************************************************************************************************************************
/home/runner/work/mlmodels/mlmodels/mlmodels/example/
############ List of files ################################
['ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_titanic.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_svm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras_charcnn_reuters.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras-textcnn.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow_1_lstm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_home_retail.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl_titanic.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow__lstm_json.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_model.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m4.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_hyper.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m4.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m5.py']





 ************************************************************************************************************************
############ Running Jupyter files ################################





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_titanic.ipynb 

Deprecaton set to False
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//lightgbm_titanic.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0mdata_path[0m [0;34m=[0m [0;34m'hyper_lightgbm_titanic.json'[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mdata_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpdict[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'hyper_lightgbm_titanic.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn_titanic_svm.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     69[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 70[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     71[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    127[0m [0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m

[0;31mModuleNotFoundError[0m: No module named 'mlmodels.model_sklearn.sklearn'

During handling of the above exception, another exception occurred:

[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     81[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 82[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     83[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: tuple index out of range

During handling of the above exception, another exception occurred:

[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_svm.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;34m[0m[0m
[1;32m      4[0m [0mmodel_uri[0m    [0;34m=[0m [0;34m"model_sklearn.sklearn.py"[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0mmodule[0m        [0;34m=[0m  [0mmodule_load[0m[0;34m([0m [0mmodel_uri[0m[0;34m=[0m [0mmodel_uri[0m [0;34m)[0m                           [0;31m# Load file definition[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={

[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     85[0m [0;34m[0m[0m
[1;32m     86[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 87[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     88[0m [0;34m[0m[0m
[1;32m     89[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//keras_charcnn_reuters.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//keras_charcnn_reuters.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mconfig_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mmodel_pars[0m      [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpars[0m[0;34m[[0m[0;34m'model_pars'[0m[0;34m][0m [0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0mdata_pars[0m       [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpars[0m[0;34m[[0m[0;34m'data_pars'[0m[0;34m][0m [0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mcompute_pars[0m    [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpars[0m[0;34m[[0m[0;34m'compute_pars'[0m[0;34m][0m [0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mout_pars[0m        [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpars[0m[0;34m[[0m[0;34m'out_pars'[0m[0;34m][0m [0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'reuters_charcnn.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//keras-textcnn.ipynb 

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 400)          0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 400, 50)      500         input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 398, 128)     19328       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 397, 128)     25728       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 396, 128)     32128       embedding_1[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 128)          0           conv1d_1[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 128)          0           conv1d_2[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalM (None, 128)          0           conv1d_3[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 384)          0           global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
                                                                 global_max_pooling1d_3[0][0]     
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            385         concatenate_1[0][0]              
==================================================================================================
Total params: 78,069
Trainable params: 78,069
Non-trainable params: 0
__________________________________________________________________________________________________
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 46s
   57344/17464789 [..............................] - ETA: 39s
  106496/17464789 [..............................] - ETA: 32s
  229376/17464789 [..............................] - ETA: 19s
  458752/17464789 [..............................] - ETA: 12s
  909312/17464789 [>.............................] - ETA: 7s 
 1810432/17464789 [==>...........................] - ETA: 3s
 3645440/17464789 [=====>........................] - ETA: 1s
 6643712/17464789 [==========>...................] - ETA: 0s
 9478144/17464789 [===============>..............] - ETA: 0s
12492800/17464789 [====================>.........] - ETA: 0s
15540224/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 03:09:58.555701: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 03:09:58.560264: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-11 03:09:58.560408: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5646f3d2d0e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 03:09:58.560422: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

2020-05-11 03:09:59.030470: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:44 - loss: 6.7083 - accuracy: 0.56252020-05-11 03:09:59.150452: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

   64/25000 [..............................] - ETA: 3:07 - loss: 6.2291 - accuracy: 0.59382020-05-11 03:09:59.264123: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

   96/25000 [..............................] - ETA: 2:34 - loss: 6.7083 - accuracy: 0.56252020-05-11 03:09:59.389547: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

  128/25000 [..............................] - ETA: 2:20 - loss: 6.2291 - accuracy: 0.59382020-05-11 03:09:59.522125: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

  160/25000 [..............................] - ETA: 2:13 - loss: 6.8041 - accuracy: 0.5562
  192/25000 [..............................] - ETA: 2:06 - loss: 6.6284 - accuracy: 0.5677
  224/25000 [..............................] - ETA: 2:01 - loss: 6.9136 - accuracy: 0.5491
  256/25000 [..............................] - ETA: 1:56 - loss: 6.8281 - accuracy: 0.5547
  288/25000 [..............................] - ETA: 1:54 - loss: 6.8680 - accuracy: 0.5521
  320/25000 [..............................] - ETA: 1:52 - loss: 7.1395 - accuracy: 0.5344
  352/25000 [..............................] - ETA: 1:50 - loss: 7.1003 - accuracy: 0.5369
  384/25000 [..............................] - ETA: 1:49 - loss: 7.1076 - accuracy: 0.5365
  416/25000 [..............................] - ETA: 1:47 - loss: 7.1506 - accuracy: 0.5337
  448/25000 [..............................] - ETA: 1:45 - loss: 7.2217 - accuracy: 0.5290
  480/25000 [..............................] - ETA: 1:44 - loss: 7.3152 - accuracy: 0.5229
  512/25000 [..............................] - ETA: 1:42 - loss: 7.3671 - accuracy: 0.5195
  544/25000 [..............................] - ETA: 1:42 - loss: 7.4693 - accuracy: 0.5129
  576/25000 [..............................] - ETA: 1:40 - loss: 7.4270 - accuracy: 0.5156
  608/25000 [..............................] - ETA: 1:40 - loss: 7.3892 - accuracy: 0.5181
  640/25000 [..............................] - ETA: 1:39 - loss: 7.4510 - accuracy: 0.5141
  672/25000 [..............................] - ETA: 1:39 - loss: 7.4613 - accuracy: 0.5134
  704/25000 [..............................] - ETA: 1:38 - loss: 7.4488 - accuracy: 0.5142
  736/25000 [..............................] - ETA: 1:37 - loss: 7.4166 - accuracy: 0.5163
  768/25000 [..............................] - ETA: 1:37 - loss: 7.4869 - accuracy: 0.5117
  800/25000 [..............................] - ETA: 1:37 - loss: 7.4941 - accuracy: 0.5113
  832/25000 [..............................] - ETA: 1:36 - loss: 7.4455 - accuracy: 0.5144
  864/25000 [>.............................] - ETA: 1:36 - loss: 7.4359 - accuracy: 0.5150
  896/25000 [>.............................] - ETA: 1:35 - loss: 7.4270 - accuracy: 0.5156
  928/25000 [>.............................] - ETA: 1:35 - loss: 7.4023 - accuracy: 0.5172
  960/25000 [>.............................] - ETA: 1:34 - loss: 7.4111 - accuracy: 0.5167
  992/25000 [>.............................] - ETA: 1:34 - loss: 7.4502 - accuracy: 0.5141
 1024/25000 [>.............................] - ETA: 1:34 - loss: 7.5319 - accuracy: 0.5088
 1056/25000 [>.............................] - ETA: 1:34 - loss: 7.5069 - accuracy: 0.5104
 1088/25000 [>.............................] - ETA: 1:33 - loss: 7.4411 - accuracy: 0.5147
 1120/25000 [>.............................] - ETA: 1:33 - loss: 7.3928 - accuracy: 0.5179
 1152/25000 [>.............................] - ETA: 1:33 - loss: 7.3871 - accuracy: 0.5182
 1184/25000 [>.............................] - ETA: 1:33 - loss: 7.4076 - accuracy: 0.5169
 1216/25000 [>.............................] - ETA: 1:33 - loss: 7.4270 - accuracy: 0.5156
 1248/25000 [>.............................] - ETA: 1:32 - loss: 7.5069 - accuracy: 0.5104
 1280/25000 [>.............................] - ETA: 1:32 - loss: 7.4750 - accuracy: 0.5125
 1312/25000 [>.............................] - ETA: 1:32 - loss: 7.5264 - accuracy: 0.5091
 1344/25000 [>.............................] - ETA: 1:32 - loss: 7.5754 - accuracy: 0.5060
 1376/25000 [>.............................] - ETA: 1:31 - loss: 7.5998 - accuracy: 0.5044
 1408/25000 [>.............................] - ETA: 1:31 - loss: 7.5686 - accuracy: 0.5064
 1440/25000 [>.............................] - ETA: 1:31 - loss: 7.5601 - accuracy: 0.5069
 1472/25000 [>.............................] - ETA: 1:31 - loss: 7.5520 - accuracy: 0.5075
 1504/25000 [>.............................] - ETA: 1:31 - loss: 7.5953 - accuracy: 0.5047
 1536/25000 [>.............................] - ETA: 1:31 - loss: 7.6067 - accuracy: 0.5039
 1568/25000 [>.............................] - ETA: 1:30 - loss: 7.5786 - accuracy: 0.5057
 1600/25000 [>.............................] - ETA: 1:30 - loss: 7.5516 - accuracy: 0.5075
 1632/25000 [>.............................] - ETA: 1:30 - loss: 7.5257 - accuracy: 0.5092
 1664/25000 [>.............................] - ETA: 1:30 - loss: 7.5284 - accuracy: 0.5090
 1696/25000 [=>............................] - ETA: 1:29 - loss: 7.5220 - accuracy: 0.5094
 1728/25000 [=>............................] - ETA: 1:29 - loss: 7.5158 - accuracy: 0.5098
 1760/25000 [=>............................] - ETA: 1:29 - loss: 7.5359 - accuracy: 0.5085
 1792/25000 [=>............................] - ETA: 1:29 - loss: 7.5297 - accuracy: 0.5089
 1824/25000 [=>............................] - ETA: 1:29 - loss: 7.5237 - accuracy: 0.5093
 1856/25000 [=>............................] - ETA: 1:28 - loss: 7.5179 - accuracy: 0.5097
 1888/25000 [=>............................] - ETA: 1:28 - loss: 7.4879 - accuracy: 0.5117
 1920/25000 [=>............................] - ETA: 1:28 - loss: 7.4909 - accuracy: 0.5115
 1952/25000 [=>............................] - ETA: 1:28 - loss: 7.4702 - accuracy: 0.5128
 1984/25000 [=>............................] - ETA: 1:27 - loss: 7.5043 - accuracy: 0.5106
 2016/25000 [=>............................] - ETA: 1:27 - loss: 7.5525 - accuracy: 0.5074
 2048/25000 [=>............................] - ETA: 1:27 - loss: 7.5768 - accuracy: 0.5059
 2080/25000 [=>............................] - ETA: 1:27 - loss: 7.6003 - accuracy: 0.5043
 2112/25000 [=>............................] - ETA: 1:26 - loss: 7.6231 - accuracy: 0.5028
 2144/25000 [=>............................] - ETA: 1:26 - loss: 7.6309 - accuracy: 0.5023
 2176/25000 [=>............................] - ETA: 1:26 - loss: 7.6384 - accuracy: 0.5018
 2208/25000 [=>............................] - ETA: 1:26 - loss: 7.6458 - accuracy: 0.5014
 2240/25000 [=>............................] - ETA: 1:25 - loss: 7.6529 - accuracy: 0.5009
 2272/25000 [=>............................] - ETA: 1:25 - loss: 7.6936 - accuracy: 0.4982
 2304/25000 [=>............................] - ETA: 1:25 - loss: 7.6866 - accuracy: 0.4987
 2336/25000 [=>............................] - ETA: 1:25 - loss: 7.6732 - accuracy: 0.4996
 2368/25000 [=>............................] - ETA: 1:25 - loss: 7.6990 - accuracy: 0.4979
 2400/25000 [=>............................] - ETA: 1:24 - loss: 7.7113 - accuracy: 0.4971
 2432/25000 [=>............................] - ETA: 1:24 - loss: 7.6918 - accuracy: 0.4984
 2464/25000 [=>............................] - ETA: 1:24 - loss: 7.6791 - accuracy: 0.4992
 2496/25000 [=>............................] - ETA: 1:24 - loss: 7.6912 - accuracy: 0.4984
 2528/25000 [==>...........................] - ETA: 1:24 - loss: 7.6909 - accuracy: 0.4984
 2560/25000 [==>...........................] - ETA: 1:24 - loss: 7.7085 - accuracy: 0.4973
 2592/25000 [==>...........................] - ETA: 1:23 - loss: 7.7080 - accuracy: 0.4973
 2624/25000 [==>...........................] - ETA: 1:23 - loss: 7.7075 - accuracy: 0.4973
 2656/25000 [==>...........................] - ETA: 1:23 - loss: 7.6897 - accuracy: 0.4985
 2688/25000 [==>...........................] - ETA: 1:23 - loss: 7.7180 - accuracy: 0.4967
 2720/25000 [==>...........................] - ETA: 1:23 - loss: 7.7174 - accuracy: 0.4967
 2752/25000 [==>...........................] - ETA: 1:23 - loss: 7.7056 - accuracy: 0.4975
 2784/25000 [==>...........................] - ETA: 1:22 - loss: 7.6887 - accuracy: 0.4986
 2816/25000 [==>...........................] - ETA: 1:22 - loss: 7.6775 - accuracy: 0.4993
 2848/25000 [==>...........................] - ETA: 1:22 - loss: 7.6559 - accuracy: 0.5007
 2880/25000 [==>...........................] - ETA: 1:22 - loss: 7.6506 - accuracy: 0.5010
 2912/25000 [==>...........................] - ETA: 1:22 - loss: 7.6561 - accuracy: 0.5007
 2944/25000 [==>...........................] - ETA: 1:22 - loss: 7.6458 - accuracy: 0.5014
 2976/25000 [==>...........................] - ETA: 1:21 - loss: 7.6409 - accuracy: 0.5017
 3008/25000 [==>...........................] - ETA: 1:21 - loss: 7.6360 - accuracy: 0.5020
 3040/25000 [==>...........................] - ETA: 1:21 - loss: 7.6212 - accuracy: 0.5030
 3072/25000 [==>...........................] - ETA: 1:21 - loss: 7.6167 - accuracy: 0.5033
 3104/25000 [==>...........................] - ETA: 1:21 - loss: 7.5777 - accuracy: 0.5058
 3136/25000 [==>...........................] - ETA: 1:21 - loss: 7.5542 - accuracy: 0.5073
 3168/25000 [==>...........................] - ETA: 1:20 - loss: 7.5892 - accuracy: 0.5051
 3200/25000 [==>...........................] - ETA: 1:20 - loss: 7.5708 - accuracy: 0.5063
 3232/25000 [==>...........................] - ETA: 1:20 - loss: 7.5765 - accuracy: 0.5059
 3264/25000 [==>...........................] - ETA: 1:20 - loss: 7.5821 - accuracy: 0.5055
 3296/25000 [==>...........................] - ETA: 1:20 - loss: 7.5829 - accuracy: 0.5055
 3328/25000 [==>...........................] - ETA: 1:20 - loss: 7.5791 - accuracy: 0.5057
 3360/25000 [===>..........................] - ETA: 1:20 - loss: 7.5936 - accuracy: 0.5048
 3392/25000 [===>..........................] - ETA: 1:19 - loss: 7.5762 - accuracy: 0.5059
 3424/25000 [===>..........................] - ETA: 1:19 - loss: 7.5771 - accuracy: 0.5058
 3456/25000 [===>..........................] - ETA: 1:19 - loss: 7.5779 - accuracy: 0.5058
 3488/25000 [===>..........................] - ETA: 1:19 - loss: 7.5831 - accuracy: 0.5054
 3520/25000 [===>..........................] - ETA: 1:19 - loss: 7.5708 - accuracy: 0.5063
 3552/25000 [===>..........................] - ETA: 1:19 - loss: 7.5846 - accuracy: 0.5053
 3584/25000 [===>..........................] - ETA: 1:19 - loss: 7.5811 - accuracy: 0.5056
 3616/25000 [===>..........................] - ETA: 1:18 - loss: 7.5691 - accuracy: 0.5064
 3648/25000 [===>..........................] - ETA: 1:18 - loss: 7.5699 - accuracy: 0.5063
 3680/25000 [===>..........................] - ETA: 1:18 - loss: 7.5791 - accuracy: 0.5057
 3712/25000 [===>..........................] - ETA: 1:18 - loss: 7.6047 - accuracy: 0.5040
 3744/25000 [===>..........................] - ETA: 1:18 - loss: 7.6175 - accuracy: 0.5032
 3776/25000 [===>..........................] - ETA: 1:18 - loss: 7.6260 - accuracy: 0.5026
 3808/25000 [===>..........................] - ETA: 1:17 - loss: 7.6183 - accuracy: 0.5032
 3840/25000 [===>..........................] - ETA: 1:17 - loss: 7.6307 - accuracy: 0.5023
 3872/25000 [===>..........................] - ETA: 1:17 - loss: 7.6468 - accuracy: 0.5013
 3904/25000 [===>..........................] - ETA: 1:17 - loss: 7.6548 - accuracy: 0.5008
 3936/25000 [===>..........................] - ETA: 1:17 - loss: 7.6666 - accuracy: 0.5000
 3968/25000 [===>..........................] - ETA: 1:17 - loss: 7.6743 - accuracy: 0.4995
 4000/25000 [===>..........................] - ETA: 1:16 - loss: 7.6743 - accuracy: 0.4995
 4032/25000 [===>..........................] - ETA: 1:16 - loss: 7.6894 - accuracy: 0.4985
 4064/25000 [===>..........................] - ETA: 1:16 - loss: 7.6930 - accuracy: 0.4983
 4096/25000 [===>..........................] - ETA: 1:16 - loss: 7.7041 - accuracy: 0.4976
 4128/25000 [===>..........................] - ETA: 1:16 - loss: 7.7075 - accuracy: 0.4973
 4160/25000 [===>..........................] - ETA: 1:16 - loss: 7.6998 - accuracy: 0.4978
 4192/25000 [====>.........................] - ETA: 1:16 - loss: 7.7069 - accuracy: 0.4974
 4224/25000 [====>.........................] - ETA: 1:15 - loss: 7.6920 - accuracy: 0.4983
 4256/25000 [====>.........................] - ETA: 1:15 - loss: 7.6702 - accuracy: 0.4998
 4288/25000 [====>.........................] - ETA: 1:15 - loss: 7.6809 - accuracy: 0.4991
 4320/25000 [====>.........................] - ETA: 1:15 - loss: 7.6844 - accuracy: 0.4988
 4352/25000 [====>.........................] - ETA: 1:15 - loss: 7.6913 - accuracy: 0.4984
 4384/25000 [====>.........................] - ETA: 1:15 - loss: 7.6981 - accuracy: 0.4979
 4416/25000 [====>.........................] - ETA: 1:15 - loss: 7.7013 - accuracy: 0.4977
 4448/25000 [====>.........................] - ETA: 1:14 - loss: 7.6908 - accuracy: 0.4984
 4480/25000 [====>.........................] - ETA: 1:14 - loss: 7.6906 - accuracy: 0.4984
 4512/25000 [====>.........................] - ETA: 1:14 - loss: 7.6700 - accuracy: 0.4998
 4544/25000 [====>.........................] - ETA: 1:14 - loss: 7.6666 - accuracy: 0.5000
 4576/25000 [====>.........................] - ETA: 1:14 - loss: 7.6666 - accuracy: 0.5000
 4608/25000 [====>.........................] - ETA: 1:14 - loss: 7.6633 - accuracy: 0.5002
 4640/25000 [====>.........................] - ETA: 1:14 - loss: 7.6798 - accuracy: 0.4991
 4672/25000 [====>.........................] - ETA: 1:13 - loss: 7.6929 - accuracy: 0.4983
 4704/25000 [====>.........................] - ETA: 1:13 - loss: 7.6992 - accuracy: 0.4979
 4736/25000 [====>.........................] - ETA: 1:13 - loss: 7.7022 - accuracy: 0.4977
 4768/25000 [====>.........................] - ETA: 1:13 - loss: 7.7020 - accuracy: 0.4977
 4800/25000 [====>.........................] - ETA: 1:13 - loss: 7.6954 - accuracy: 0.4981
 4832/25000 [====>.........................] - ETA: 1:13 - loss: 7.7015 - accuracy: 0.4977
 4864/25000 [====>.........................] - ETA: 1:13 - loss: 7.6981 - accuracy: 0.4979
 4896/25000 [====>.........................] - ETA: 1:13 - loss: 7.7011 - accuracy: 0.4978
 4928/25000 [====>.........................] - ETA: 1:12 - loss: 7.7040 - accuracy: 0.4976
 4960/25000 [====>.........................] - ETA: 1:12 - loss: 7.6944 - accuracy: 0.4982
 4992/25000 [====>.........................] - ETA: 1:12 - loss: 7.6943 - accuracy: 0.4982
 5024/25000 [=====>........................] - ETA: 1:12 - loss: 7.6880 - accuracy: 0.4986
 5056/25000 [=====>........................] - ETA: 1:12 - loss: 7.6848 - accuracy: 0.4988
 5088/25000 [=====>........................] - ETA: 1:12 - loss: 7.6817 - accuracy: 0.4990
 5120/25000 [=====>........................] - ETA: 1:12 - loss: 7.6876 - accuracy: 0.4986
 5152/25000 [=====>........................] - ETA: 1:11 - loss: 7.6904 - accuracy: 0.4984
 5184/25000 [=====>........................] - ETA: 1:11 - loss: 7.6903 - accuracy: 0.4985
 5216/25000 [=====>........................] - ETA: 1:11 - loss: 7.7019 - accuracy: 0.4977
 5248/25000 [=====>........................] - ETA: 1:11 - loss: 7.7104 - accuracy: 0.4971
 5280/25000 [=====>........................] - ETA: 1:11 - loss: 7.7044 - accuracy: 0.4975
 5312/25000 [=====>........................] - ETA: 1:11 - loss: 7.7099 - accuracy: 0.4972
 5344/25000 [=====>........................] - ETA: 1:11 - loss: 7.7039 - accuracy: 0.4976
 5376/25000 [=====>........................] - ETA: 1:11 - loss: 7.6980 - accuracy: 0.4980
 5408/25000 [=====>........................] - ETA: 1:10 - loss: 7.7091 - accuracy: 0.4972
 5440/25000 [=====>........................] - ETA: 1:10 - loss: 7.7145 - accuracy: 0.4969
 5472/25000 [=====>........................] - ETA: 1:10 - loss: 7.7115 - accuracy: 0.4971
 5504/25000 [=====>........................] - ETA: 1:10 - loss: 7.7140 - accuracy: 0.4969
 5536/25000 [=====>........................] - ETA: 1:10 - loss: 7.7276 - accuracy: 0.4960
 5568/25000 [=====>........................] - ETA: 1:10 - loss: 7.7244 - accuracy: 0.4962
 5600/25000 [=====>........................] - ETA: 1:10 - loss: 7.7241 - accuracy: 0.4963
 5632/25000 [=====>........................] - ETA: 1:09 - loss: 7.7183 - accuracy: 0.4966
 5664/25000 [=====>........................] - ETA: 1:09 - loss: 7.7208 - accuracy: 0.4965
 5696/25000 [=====>........................] - ETA: 1:09 - loss: 7.7312 - accuracy: 0.4958
 5728/25000 [=====>........................] - ETA: 1:09 - loss: 7.7416 - accuracy: 0.4951
 5760/25000 [=====>........................] - ETA: 1:09 - loss: 7.7518 - accuracy: 0.4944
 5792/25000 [=====>........................] - ETA: 1:09 - loss: 7.7513 - accuracy: 0.4945
 5824/25000 [=====>........................] - ETA: 1:09 - loss: 7.7482 - accuracy: 0.4947
 5856/25000 [======>.......................] - ETA: 1:09 - loss: 7.7530 - accuracy: 0.4944
 5888/25000 [======>.......................] - ETA: 1:08 - loss: 7.7395 - accuracy: 0.4952
 5920/25000 [======>.......................] - ETA: 1:08 - loss: 7.7391 - accuracy: 0.4953
 5952/25000 [======>.......................] - ETA: 1:08 - loss: 7.7336 - accuracy: 0.4956
 5984/25000 [======>.......................] - ETA: 1:08 - loss: 7.7281 - accuracy: 0.4960
 6016/25000 [======>.......................] - ETA: 1:08 - loss: 7.7252 - accuracy: 0.4962
 6048/25000 [======>.......................] - ETA: 1:08 - loss: 7.7249 - accuracy: 0.4962
 6080/25000 [======>.......................] - ETA: 1:08 - loss: 7.7196 - accuracy: 0.4965
 6112/25000 [======>.......................] - ETA: 1:08 - loss: 7.7344 - accuracy: 0.4956
 6144/25000 [======>.......................] - ETA: 1:07 - loss: 7.7240 - accuracy: 0.4963
 6176/25000 [======>.......................] - ETA: 1:07 - loss: 7.7138 - accuracy: 0.4969
 6208/25000 [======>.......................] - ETA: 1:07 - loss: 7.7061 - accuracy: 0.4974
 6240/25000 [======>.......................] - ETA: 1:07 - loss: 7.7059 - accuracy: 0.4974
 6272/25000 [======>.......................] - ETA: 1:07 - loss: 7.7155 - accuracy: 0.4968
 6304/25000 [======>.......................] - ETA: 1:07 - loss: 7.7299 - accuracy: 0.4959
 6336/25000 [======>.......................] - ETA: 1:07 - loss: 7.7392 - accuracy: 0.4953
 6368/25000 [======>.......................] - ETA: 1:07 - loss: 7.7389 - accuracy: 0.4953
 6400/25000 [======>.......................] - ETA: 1:07 - loss: 7.7385 - accuracy: 0.4953
 6432/25000 [======>.......................] - ETA: 1:06 - loss: 7.7453 - accuracy: 0.4949
 6464/25000 [======>.......................] - ETA: 1:06 - loss: 7.7425 - accuracy: 0.4950
 6496/25000 [======>.......................] - ETA: 1:06 - loss: 7.7445 - accuracy: 0.4949
 6528/25000 [======>.......................] - ETA: 1:06 - loss: 7.7559 - accuracy: 0.4942
 6560/25000 [======>.......................] - ETA: 1:06 - loss: 7.7508 - accuracy: 0.4945
 6592/25000 [======>.......................] - ETA: 1:06 - loss: 7.7411 - accuracy: 0.4951
 6624/25000 [======>.......................] - ETA: 1:06 - loss: 7.7407 - accuracy: 0.4952
 6656/25000 [======>.......................] - ETA: 1:05 - loss: 7.7426 - accuracy: 0.4950
 6688/25000 [=======>......................] - ETA: 1:05 - loss: 7.7423 - accuracy: 0.4951
 6720/25000 [=======>......................] - ETA: 1:05 - loss: 7.7510 - accuracy: 0.4945
 6752/25000 [=======>......................] - ETA: 1:05 - loss: 7.7597 - accuracy: 0.4939
 6784/25000 [=======>......................] - ETA: 1:05 - loss: 7.7728 - accuracy: 0.4931
 6816/25000 [=======>......................] - ETA: 1:05 - loss: 7.7701 - accuracy: 0.4933
 6848/25000 [=======>......................] - ETA: 1:05 - loss: 7.7763 - accuracy: 0.4928
 6880/25000 [=======>......................] - ETA: 1:05 - loss: 7.7758 - accuracy: 0.4929
 6912/25000 [=======>......................] - ETA: 1:05 - loss: 7.7687 - accuracy: 0.4933
 6944/25000 [=======>......................] - ETA: 1:04 - loss: 7.7748 - accuracy: 0.4929
 6976/25000 [=======>......................] - ETA: 1:04 - loss: 7.7699 - accuracy: 0.4933
 7008/25000 [=======>......................] - ETA: 1:04 - loss: 7.7629 - accuracy: 0.4937
 7040/25000 [=======>......................] - ETA: 1:04 - loss: 7.7646 - accuracy: 0.4936
 7072/25000 [=======>......................] - ETA: 1:04 - loss: 7.7599 - accuracy: 0.4939
 7104/25000 [=======>......................] - ETA: 1:04 - loss: 7.7508 - accuracy: 0.4945
 7136/25000 [=======>......................] - ETA: 1:04 - loss: 7.7461 - accuracy: 0.4948
 7168/25000 [=======>......................] - ETA: 1:04 - loss: 7.7458 - accuracy: 0.4948
 7200/25000 [=======>......................] - ETA: 1:03 - loss: 7.7433 - accuracy: 0.4950
 7232/25000 [=======>......................] - ETA: 1:03 - loss: 7.7451 - accuracy: 0.4949
 7264/25000 [=======>......................] - ETA: 1:03 - loss: 7.7384 - accuracy: 0.4953
 7296/25000 [=======>......................] - ETA: 1:03 - loss: 7.7318 - accuracy: 0.4958
 7328/25000 [=======>......................] - ETA: 1:03 - loss: 7.7336 - accuracy: 0.4956
 7360/25000 [=======>......................] - ETA: 1:03 - loss: 7.7270 - accuracy: 0.4961
 7392/25000 [=======>......................] - ETA: 1:03 - loss: 7.7309 - accuracy: 0.4958
 7424/25000 [=======>......................] - ETA: 1:03 - loss: 7.7368 - accuracy: 0.4954
 7456/25000 [=======>......................] - ETA: 1:02 - loss: 7.7345 - accuracy: 0.4956
 7488/25000 [=======>......................] - ETA: 1:02 - loss: 7.7321 - accuracy: 0.4957
 7520/25000 [========>.....................] - ETA: 1:02 - loss: 7.7278 - accuracy: 0.4960
 7552/25000 [========>.....................] - ETA: 1:02 - loss: 7.7255 - accuracy: 0.4962
 7584/25000 [========>.....................] - ETA: 1:02 - loss: 7.7293 - accuracy: 0.4959
 7616/25000 [========>.....................] - ETA: 1:02 - loss: 7.7270 - accuracy: 0.4961
 7648/25000 [========>.....................] - ETA: 1:02 - loss: 7.7288 - accuracy: 0.4959
 7680/25000 [========>.....................] - ETA: 1:02 - loss: 7.7285 - accuracy: 0.4960
 7712/25000 [========>.....................] - ETA: 1:02 - loss: 7.7263 - accuracy: 0.4961
 7744/25000 [========>.....................] - ETA: 1:01 - loss: 7.7320 - accuracy: 0.4957
 7776/25000 [========>.....................] - ETA: 1:01 - loss: 7.7317 - accuracy: 0.4958
 7808/25000 [========>.....................] - ETA: 1:01 - loss: 7.7314 - accuracy: 0.4958
 7840/25000 [========>.....................] - ETA: 1:01 - loss: 7.7351 - accuracy: 0.4955
 7872/25000 [========>.....................] - ETA: 1:01 - loss: 7.7367 - accuracy: 0.4954
 7904/25000 [========>.....................] - ETA: 1:01 - loss: 7.7345 - accuracy: 0.4956
 7936/25000 [========>.....................] - ETA: 1:01 - loss: 7.7381 - accuracy: 0.4953
 7968/25000 [========>.....................] - ETA: 1:01 - loss: 7.7455 - accuracy: 0.4949
 8000/25000 [========>.....................] - ETA: 1:00 - loss: 7.7414 - accuracy: 0.4951
 8032/25000 [========>.....................] - ETA: 1:00 - loss: 7.7373 - accuracy: 0.4954
 8064/25000 [========>.....................] - ETA: 1:00 - loss: 7.7351 - accuracy: 0.4955
 8096/25000 [========>.....................] - ETA: 1:00 - loss: 7.7329 - accuracy: 0.4957
 8128/25000 [========>.....................] - ETA: 1:00 - loss: 7.7326 - accuracy: 0.4957
 8160/25000 [========>.....................] - ETA: 1:00 - loss: 7.7305 - accuracy: 0.4958
 8192/25000 [========>.....................] - ETA: 1:00 - loss: 7.7209 - accuracy: 0.4965
 8224/25000 [========>.....................] - ETA: 1:00 - loss: 7.7244 - accuracy: 0.4962
 8256/25000 [========>.....................] - ETA: 59s - loss: 7.7223 - accuracy: 0.4964 
 8288/25000 [========>.....................] - ETA: 59s - loss: 7.7203 - accuracy: 0.4965
 8320/25000 [========>.....................] - ETA: 59s - loss: 7.7201 - accuracy: 0.4965
 8352/25000 [=========>....................] - ETA: 59s - loss: 7.7217 - accuracy: 0.4964
 8384/25000 [=========>....................] - ETA: 59s - loss: 7.7251 - accuracy: 0.4962
 8416/25000 [=========>....................] - ETA: 59s - loss: 7.7176 - accuracy: 0.4967
 8448/25000 [=========>....................] - ETA: 59s - loss: 7.7156 - accuracy: 0.4968
 8480/25000 [=========>....................] - ETA: 59s - loss: 7.7154 - accuracy: 0.4968
 8512/25000 [=========>....................] - ETA: 58s - loss: 7.7171 - accuracy: 0.4967
 8544/25000 [=========>....................] - ETA: 58s - loss: 7.7115 - accuracy: 0.4971
 8576/25000 [=========>....................] - ETA: 58s - loss: 7.7042 - accuracy: 0.4976
 8608/25000 [=========>....................] - ETA: 58s - loss: 7.7058 - accuracy: 0.4974
 8640/25000 [=========>....................] - ETA: 58s - loss: 7.7057 - accuracy: 0.4975
 8672/25000 [=========>....................] - ETA: 58s - loss: 7.6967 - accuracy: 0.4980
 8704/25000 [=========>....................] - ETA: 58s - loss: 7.7036 - accuracy: 0.4976
 8736/25000 [=========>....................] - ETA: 58s - loss: 7.6965 - accuracy: 0.4981
 8768/25000 [=========>....................] - ETA: 58s - loss: 7.6998 - accuracy: 0.4978
 8800/25000 [=========>....................] - ETA: 57s - loss: 7.6997 - accuracy: 0.4978
 8832/25000 [=========>....................] - ETA: 57s - loss: 7.7048 - accuracy: 0.4975
 8864/25000 [=========>....................] - ETA: 57s - loss: 7.7012 - accuracy: 0.4977
 8896/25000 [=========>....................] - ETA: 57s - loss: 7.7011 - accuracy: 0.4978
 8928/25000 [=========>....................] - ETA: 57s - loss: 7.6975 - accuracy: 0.4980
 8960/25000 [=========>....................] - ETA: 57s - loss: 7.6957 - accuracy: 0.4981
 8992/25000 [=========>....................] - ETA: 57s - loss: 7.6871 - accuracy: 0.4987
 9024/25000 [=========>....................] - ETA: 57s - loss: 7.6904 - accuracy: 0.4984
 9056/25000 [=========>....................] - ETA: 56s - loss: 7.6903 - accuracy: 0.4985
 9088/25000 [=========>....................] - ETA: 56s - loss: 7.6852 - accuracy: 0.4988
 9120/25000 [=========>....................] - ETA: 56s - loss: 7.6851 - accuracy: 0.4988
 9152/25000 [=========>....................] - ETA: 56s - loss: 7.6834 - accuracy: 0.4989
 9184/25000 [==========>...................] - ETA: 56s - loss: 7.6850 - accuracy: 0.4988
 9216/25000 [==========>...................] - ETA: 56s - loss: 7.6849 - accuracy: 0.4988
 9248/25000 [==========>...................] - ETA: 56s - loss: 7.6882 - accuracy: 0.4986
 9280/25000 [==========>...................] - ETA: 56s - loss: 7.6898 - accuracy: 0.4985
 9312/25000 [==========>...................] - ETA: 55s - loss: 7.7028 - accuracy: 0.4976
 9344/25000 [==========>...................] - ETA: 55s - loss: 7.7093 - accuracy: 0.4972
 9376/25000 [==========>...................] - ETA: 55s - loss: 7.7042 - accuracy: 0.4975
 9408/25000 [==========>...................] - ETA: 55s - loss: 7.7041 - accuracy: 0.4976
 9440/25000 [==========>...................] - ETA: 55s - loss: 7.6942 - accuracy: 0.4982
 9472/25000 [==========>...................] - ETA: 55s - loss: 7.6860 - accuracy: 0.4987
 9504/25000 [==========>...................] - ETA: 55s - loss: 7.6957 - accuracy: 0.4981
 9536/25000 [==========>...................] - ETA: 55s - loss: 7.6956 - accuracy: 0.4981
 9568/25000 [==========>...................] - ETA: 55s - loss: 7.6955 - accuracy: 0.4981
 9600/25000 [==========>...................] - ETA: 54s - loss: 7.6970 - accuracy: 0.4980
 9632/25000 [==========>...................] - ETA: 54s - loss: 7.6969 - accuracy: 0.4980
 9664/25000 [==========>...................] - ETA: 54s - loss: 7.6999 - accuracy: 0.4978
 9696/25000 [==========>...................] - ETA: 54s - loss: 7.7030 - accuracy: 0.4976
 9728/25000 [==========>...................] - ETA: 54s - loss: 7.7108 - accuracy: 0.4971
 9760/25000 [==========>...................] - ETA: 54s - loss: 7.7090 - accuracy: 0.4972
 9792/25000 [==========>...................] - ETA: 54s - loss: 7.7058 - accuracy: 0.4974
 9824/25000 [==========>...................] - ETA: 54s - loss: 7.7088 - accuracy: 0.4973
 9856/25000 [==========>...................] - ETA: 53s - loss: 7.7117 - accuracy: 0.4971
 9888/25000 [==========>...................] - ETA: 53s - loss: 7.7193 - accuracy: 0.4966
 9920/25000 [==========>...................] - ETA: 53s - loss: 7.7192 - accuracy: 0.4966
 9952/25000 [==========>...................] - ETA: 53s - loss: 7.7221 - accuracy: 0.4964
 9984/25000 [==========>...................] - ETA: 53s - loss: 7.7219 - accuracy: 0.4964
10016/25000 [===========>..................] - ETA: 53s - loss: 7.7233 - accuracy: 0.4963
10048/25000 [===========>..................] - ETA: 53s - loss: 7.7216 - accuracy: 0.4964
10080/25000 [===========>..................] - ETA: 53s - loss: 7.7229 - accuracy: 0.4963
10112/25000 [===========>..................] - ETA: 53s - loss: 7.7151 - accuracy: 0.4968
10144/25000 [===========>..................] - ETA: 52s - loss: 7.7180 - accuracy: 0.4966
10176/25000 [===========>..................] - ETA: 52s - loss: 7.7224 - accuracy: 0.4964
10208/25000 [===========>..................] - ETA: 52s - loss: 7.7192 - accuracy: 0.4966
10240/25000 [===========>..................] - ETA: 52s - loss: 7.7190 - accuracy: 0.4966
10272/25000 [===========>..................] - ETA: 52s - loss: 7.7174 - accuracy: 0.4967
10304/25000 [===========>..................] - ETA: 52s - loss: 7.7157 - accuracy: 0.4968
10336/25000 [===========>..................] - ETA: 52s - loss: 7.7245 - accuracy: 0.4962
10368/25000 [===========>..................] - ETA: 52s - loss: 7.7273 - accuracy: 0.4960
10400/25000 [===========>..................] - ETA: 52s - loss: 7.7256 - accuracy: 0.4962
10432/25000 [===========>..................] - ETA: 51s - loss: 7.7269 - accuracy: 0.4961
10464/25000 [===========>..................] - ETA: 51s - loss: 7.7355 - accuracy: 0.4955
10496/25000 [===========>..................] - ETA: 51s - loss: 7.7309 - accuracy: 0.4958
10528/25000 [===========>..................] - ETA: 51s - loss: 7.7263 - accuracy: 0.4961
10560/25000 [===========>..................] - ETA: 51s - loss: 7.7203 - accuracy: 0.4965
10592/25000 [===========>..................] - ETA: 51s - loss: 7.7187 - accuracy: 0.4966
10624/25000 [===========>..................] - ETA: 51s - loss: 7.7142 - accuracy: 0.4969
10656/25000 [===========>..................] - ETA: 51s - loss: 7.7098 - accuracy: 0.4972
10688/25000 [===========>..................] - ETA: 50s - loss: 7.7025 - accuracy: 0.4977
10720/25000 [===========>..................] - ETA: 50s - loss: 7.7038 - accuracy: 0.4976
10752/25000 [===========>..................] - ETA: 50s - loss: 7.7037 - accuracy: 0.4976
10784/25000 [===========>..................] - ETA: 50s - loss: 7.6993 - accuracy: 0.4979
10816/25000 [===========>..................] - ETA: 50s - loss: 7.7006 - accuracy: 0.4978
10848/25000 [============>.................] - ETA: 50s - loss: 7.6991 - accuracy: 0.4979
10880/25000 [============>.................] - ETA: 50s - loss: 7.6976 - accuracy: 0.4980
10912/25000 [============>.................] - ETA: 50s - loss: 7.6975 - accuracy: 0.4980
10944/25000 [============>.................] - ETA: 50s - loss: 7.6960 - accuracy: 0.4981
10976/25000 [============>.................] - ETA: 49s - loss: 7.6960 - accuracy: 0.4981
11008/25000 [============>.................] - ETA: 49s - loss: 7.6959 - accuracy: 0.4981
11040/25000 [============>.................] - ETA: 49s - loss: 7.6944 - accuracy: 0.4982
11072/25000 [============>.................] - ETA: 49s - loss: 7.7012 - accuracy: 0.4977
11104/25000 [============>.................] - ETA: 49s - loss: 7.7025 - accuracy: 0.4977
11136/25000 [============>.................] - ETA: 49s - loss: 7.7093 - accuracy: 0.4972
11168/25000 [============>.................] - ETA: 49s - loss: 7.7092 - accuracy: 0.4972
11200/25000 [============>.................] - ETA: 49s - loss: 7.7063 - accuracy: 0.4974
11232/25000 [============>.................] - ETA: 48s - loss: 7.7089 - accuracy: 0.4972
11264/25000 [============>.................] - ETA: 48s - loss: 7.7129 - accuracy: 0.4970
11296/25000 [============>.................] - ETA: 48s - loss: 7.7114 - accuracy: 0.4971
11328/25000 [============>.................] - ETA: 48s - loss: 7.7086 - accuracy: 0.4973
11360/25000 [============>.................] - ETA: 48s - loss: 7.7085 - accuracy: 0.4973
11392/25000 [============>.................] - ETA: 48s - loss: 7.7097 - accuracy: 0.4972
11424/25000 [============>.................] - ETA: 48s - loss: 7.7109 - accuracy: 0.4971
11456/25000 [============>.................] - ETA: 48s - loss: 7.7094 - accuracy: 0.4972
11488/25000 [============>.................] - ETA: 48s - loss: 7.7053 - accuracy: 0.4975
11520/25000 [============>.................] - ETA: 47s - loss: 7.7105 - accuracy: 0.4971
11552/25000 [============>.................] - ETA: 47s - loss: 7.7171 - accuracy: 0.4967
11584/25000 [============>.................] - ETA: 47s - loss: 7.7209 - accuracy: 0.4965
11616/25000 [============>.................] - ETA: 47s - loss: 7.7102 - accuracy: 0.4972
11648/25000 [============>.................] - ETA: 47s - loss: 7.7074 - accuracy: 0.4973
11680/25000 [=============>................] - ETA: 47s - loss: 7.7073 - accuracy: 0.4973
11712/25000 [=============>................] - ETA: 47s - loss: 7.7072 - accuracy: 0.4974
11744/25000 [=============>................] - ETA: 47s - loss: 7.7084 - accuracy: 0.4973
11776/25000 [=============>................] - ETA: 46s - loss: 7.7161 - accuracy: 0.4968
11808/25000 [=============>................] - ETA: 46s - loss: 7.7212 - accuracy: 0.4964
11840/25000 [=============>................] - ETA: 46s - loss: 7.7210 - accuracy: 0.4965
11872/25000 [=============>................] - ETA: 46s - loss: 7.7222 - accuracy: 0.4964
11904/25000 [=============>................] - ETA: 46s - loss: 7.7194 - accuracy: 0.4966
11936/25000 [=============>................] - ETA: 46s - loss: 7.7154 - accuracy: 0.4968
11968/25000 [=============>................] - ETA: 46s - loss: 7.7153 - accuracy: 0.4968
12000/25000 [=============>................] - ETA: 46s - loss: 7.7101 - accuracy: 0.4972
12032/25000 [=============>................] - ETA: 46s - loss: 7.7049 - accuracy: 0.4975
12064/25000 [=============>................] - ETA: 45s - loss: 7.7022 - accuracy: 0.4977
12096/25000 [=============>................] - ETA: 45s - loss: 7.6996 - accuracy: 0.4979
12128/25000 [=============>................] - ETA: 45s - loss: 7.6932 - accuracy: 0.4983
12160/25000 [=============>................] - ETA: 45s - loss: 7.6881 - accuracy: 0.4986
12192/25000 [=============>................] - ETA: 45s - loss: 7.6930 - accuracy: 0.4983
12224/25000 [=============>................] - ETA: 45s - loss: 7.6917 - accuracy: 0.4984
12256/25000 [=============>................] - ETA: 45s - loss: 7.6991 - accuracy: 0.4979
12288/25000 [=============>................] - ETA: 45s - loss: 7.6966 - accuracy: 0.4980
12320/25000 [=============>................] - ETA: 45s - loss: 7.6952 - accuracy: 0.4981
12352/25000 [=============>................] - ETA: 44s - loss: 7.6914 - accuracy: 0.4984
12384/25000 [=============>................] - ETA: 44s - loss: 7.6901 - accuracy: 0.4985
12416/25000 [=============>................] - ETA: 44s - loss: 7.6901 - accuracy: 0.4985
12448/25000 [=============>................] - ETA: 44s - loss: 7.6913 - accuracy: 0.4984
12480/25000 [=============>................] - ETA: 44s - loss: 7.6900 - accuracy: 0.4985
12512/25000 [==============>...............] - ETA: 44s - loss: 7.6862 - accuracy: 0.4987
12544/25000 [==============>...............] - ETA: 44s - loss: 7.6874 - accuracy: 0.4986
12576/25000 [==============>...............] - ETA: 44s - loss: 7.6861 - accuracy: 0.4987
12608/25000 [==============>...............] - ETA: 43s - loss: 7.6776 - accuracy: 0.4993
12640/25000 [==============>...............] - ETA: 43s - loss: 7.6763 - accuracy: 0.4994
12672/25000 [==============>...............] - ETA: 43s - loss: 7.6727 - accuracy: 0.4996
12704/25000 [==============>...............] - ETA: 43s - loss: 7.6690 - accuracy: 0.4998
12736/25000 [==============>...............] - ETA: 43s - loss: 7.6726 - accuracy: 0.4996
12768/25000 [==============>...............] - ETA: 43s - loss: 7.6750 - accuracy: 0.4995
12800/25000 [==============>...............] - ETA: 43s - loss: 7.6774 - accuracy: 0.4993
12832/25000 [==============>...............] - ETA: 43s - loss: 7.6798 - accuracy: 0.4991
12864/25000 [==============>...............] - ETA: 43s - loss: 7.6845 - accuracy: 0.4988
12896/25000 [==============>...............] - ETA: 42s - loss: 7.6856 - accuracy: 0.4988
12928/25000 [==============>...............] - ETA: 42s - loss: 7.6856 - accuracy: 0.4988
12960/25000 [==============>...............] - ETA: 42s - loss: 7.6855 - accuracy: 0.4988
12992/25000 [==============>...............] - ETA: 42s - loss: 7.6831 - accuracy: 0.4989
13024/25000 [==============>...............] - ETA: 42s - loss: 7.6831 - accuracy: 0.4989
13056/25000 [==============>...............] - ETA: 42s - loss: 7.6831 - accuracy: 0.4989
13088/25000 [==============>...............] - ETA: 42s - loss: 7.6889 - accuracy: 0.4985
13120/25000 [==============>...............] - ETA: 42s - loss: 7.6923 - accuracy: 0.4983
13152/25000 [==============>...............] - ETA: 41s - loss: 7.6946 - accuracy: 0.4982
13184/25000 [==============>...............] - ETA: 41s - loss: 7.6980 - accuracy: 0.4980
13216/25000 [==============>...............] - ETA: 41s - loss: 7.6945 - accuracy: 0.4982
13248/25000 [==============>...............] - ETA: 41s - loss: 7.7002 - accuracy: 0.4978
13280/25000 [==============>...............] - ETA: 41s - loss: 7.7047 - accuracy: 0.4975
13312/25000 [==============>...............] - ETA: 41s - loss: 7.7058 - accuracy: 0.4974
13344/25000 [===============>..............] - ETA: 41s - loss: 7.7022 - accuracy: 0.4977
13376/25000 [===============>..............] - ETA: 41s - loss: 7.7033 - accuracy: 0.4976
13408/25000 [===============>..............] - ETA: 41s - loss: 7.7044 - accuracy: 0.4975
13440/25000 [===============>..............] - ETA: 40s - loss: 7.7054 - accuracy: 0.4975
13472/25000 [===============>..............] - ETA: 40s - loss: 7.7008 - accuracy: 0.4978
13504/25000 [===============>..............] - ETA: 40s - loss: 7.7052 - accuracy: 0.4975
13536/25000 [===============>..............] - ETA: 40s - loss: 7.7074 - accuracy: 0.4973
13568/25000 [===============>..............] - ETA: 40s - loss: 7.7050 - accuracy: 0.4975
13600/25000 [===============>..............] - ETA: 40s - loss: 7.7083 - accuracy: 0.4973
13632/25000 [===============>..............] - ETA: 40s - loss: 7.7105 - accuracy: 0.4971
13664/25000 [===============>..............] - ETA: 40s - loss: 7.7126 - accuracy: 0.4970
13696/25000 [===============>..............] - ETA: 40s - loss: 7.7159 - accuracy: 0.4968
13728/25000 [===============>..............] - ETA: 39s - loss: 7.7135 - accuracy: 0.4969
13760/25000 [===============>..............] - ETA: 39s - loss: 7.7179 - accuracy: 0.4967
13792/25000 [===============>..............] - ETA: 39s - loss: 7.7144 - accuracy: 0.4969
13824/25000 [===============>..............] - ETA: 39s - loss: 7.7165 - accuracy: 0.4967
13856/25000 [===============>..............] - ETA: 39s - loss: 7.7208 - accuracy: 0.4965
13888/25000 [===============>..............] - ETA: 39s - loss: 7.7152 - accuracy: 0.4968
13920/25000 [===============>..............] - ETA: 39s - loss: 7.7129 - accuracy: 0.4970
13952/25000 [===============>..............] - ETA: 39s - loss: 7.7106 - accuracy: 0.4971
13984/25000 [===============>..............] - ETA: 39s - loss: 7.7149 - accuracy: 0.4969
14016/25000 [===============>..............] - ETA: 38s - loss: 7.7180 - accuracy: 0.4966
14048/25000 [===============>..............] - ETA: 38s - loss: 7.7136 - accuracy: 0.4969
14080/25000 [===============>..............] - ETA: 38s - loss: 7.7167 - accuracy: 0.4967
14112/25000 [===============>..............] - ETA: 38s - loss: 7.7144 - accuracy: 0.4969
14144/25000 [===============>..............] - ETA: 38s - loss: 7.7132 - accuracy: 0.4970
14176/25000 [================>.............] - ETA: 38s - loss: 7.7120 - accuracy: 0.4970
14208/25000 [================>.............] - ETA: 38s - loss: 7.7044 - accuracy: 0.4975
14240/25000 [================>.............] - ETA: 38s - loss: 7.7011 - accuracy: 0.4978
14272/25000 [================>.............] - ETA: 37s - loss: 7.6999 - accuracy: 0.4978
14304/25000 [================>.............] - ETA: 37s - loss: 7.7031 - accuracy: 0.4976
14336/25000 [================>.............] - ETA: 37s - loss: 7.7094 - accuracy: 0.4972
14368/25000 [================>.............] - ETA: 37s - loss: 7.7072 - accuracy: 0.4974
14400/25000 [================>.............] - ETA: 37s - loss: 7.7071 - accuracy: 0.4974
14432/25000 [================>.............] - ETA: 37s - loss: 7.7091 - accuracy: 0.4972
14464/25000 [================>.............] - ETA: 37s - loss: 7.7090 - accuracy: 0.4972
14496/25000 [================>.............] - ETA: 37s - loss: 7.7132 - accuracy: 0.4970
14528/25000 [================>.............] - ETA: 37s - loss: 7.7120 - accuracy: 0.4970
14560/25000 [================>.............] - ETA: 36s - loss: 7.7140 - accuracy: 0.4969
14592/25000 [================>.............] - ETA: 36s - loss: 7.7129 - accuracy: 0.4970
14624/25000 [================>.............] - ETA: 36s - loss: 7.7149 - accuracy: 0.4969
14656/25000 [================>.............] - ETA: 36s - loss: 7.7168 - accuracy: 0.4967
14688/25000 [================>.............] - ETA: 36s - loss: 7.7167 - accuracy: 0.4967
14720/25000 [================>.............] - ETA: 36s - loss: 7.7135 - accuracy: 0.4969
14752/25000 [================>.............] - ETA: 36s - loss: 7.7124 - accuracy: 0.4970
14784/25000 [================>.............] - ETA: 36s - loss: 7.7164 - accuracy: 0.4968
14816/25000 [================>.............] - ETA: 36s - loss: 7.7204 - accuracy: 0.4965
14848/25000 [================>.............] - ETA: 35s - loss: 7.7141 - accuracy: 0.4969
14880/25000 [================>.............] - ETA: 35s - loss: 7.7140 - accuracy: 0.4969
14912/25000 [================>.............] - ETA: 35s - loss: 7.7119 - accuracy: 0.4970
14944/25000 [================>.............] - ETA: 35s - loss: 7.7118 - accuracy: 0.4971
14976/25000 [================>.............] - ETA: 35s - loss: 7.7096 - accuracy: 0.4972
15008/25000 [=================>............] - ETA: 35s - loss: 7.7065 - accuracy: 0.4974
15040/25000 [=================>............] - ETA: 35s - loss: 7.7115 - accuracy: 0.4971
15072/25000 [=================>............] - ETA: 35s - loss: 7.7093 - accuracy: 0.4972
15104/25000 [=================>............] - ETA: 34s - loss: 7.7103 - accuracy: 0.4972
15136/25000 [=================>............] - ETA: 34s - loss: 7.7092 - accuracy: 0.4972
15168/25000 [=================>............] - ETA: 34s - loss: 7.7111 - accuracy: 0.4971
15200/25000 [=================>............] - ETA: 34s - loss: 7.7090 - accuracy: 0.4972
15232/25000 [=================>............] - ETA: 34s - loss: 7.7089 - accuracy: 0.4972
15264/25000 [=================>............] - ETA: 34s - loss: 7.7088 - accuracy: 0.4972
15296/25000 [=================>............] - ETA: 34s - loss: 7.7047 - accuracy: 0.4975
15328/25000 [=================>............] - ETA: 34s - loss: 7.7066 - accuracy: 0.4974
15360/25000 [=================>............] - ETA: 34s - loss: 7.7075 - accuracy: 0.4973
15392/25000 [=================>............] - ETA: 33s - loss: 7.7095 - accuracy: 0.4972
15424/25000 [=================>............] - ETA: 33s - loss: 7.7084 - accuracy: 0.4973
15456/25000 [=================>............] - ETA: 33s - loss: 7.7093 - accuracy: 0.4972
15488/25000 [=================>............] - ETA: 33s - loss: 7.7042 - accuracy: 0.4975
15520/25000 [=================>............] - ETA: 33s - loss: 7.7061 - accuracy: 0.4974
15552/25000 [=================>............] - ETA: 33s - loss: 7.7021 - accuracy: 0.4977
15584/25000 [=================>............] - ETA: 33s - loss: 7.7040 - accuracy: 0.4976
15616/25000 [=================>............] - ETA: 33s - loss: 7.7010 - accuracy: 0.4978
15648/25000 [=================>............] - ETA: 33s - loss: 7.7009 - accuracy: 0.4978
15680/25000 [=================>............] - ETA: 32s - loss: 7.7028 - accuracy: 0.4976
15712/25000 [=================>............] - ETA: 32s - loss: 7.7037 - accuracy: 0.4976
15744/25000 [=================>............] - ETA: 32s - loss: 7.7104 - accuracy: 0.4971
15776/25000 [=================>............] - ETA: 32s - loss: 7.7104 - accuracy: 0.4971
15808/25000 [=================>............] - ETA: 32s - loss: 7.7083 - accuracy: 0.4973
15840/25000 [==================>...........] - ETA: 32s - loss: 7.7111 - accuracy: 0.4971
15872/25000 [==================>...........] - ETA: 32s - loss: 7.7111 - accuracy: 0.4971
15904/25000 [==================>...........] - ETA: 32s - loss: 7.7100 - accuracy: 0.4972
15936/25000 [==================>...........] - ETA: 31s - loss: 7.7061 - accuracy: 0.4974
15968/25000 [==================>...........] - ETA: 31s - loss: 7.7098 - accuracy: 0.4972
16000/25000 [==================>...........] - ETA: 31s - loss: 7.7078 - accuracy: 0.4973
16032/25000 [==================>...........] - ETA: 31s - loss: 7.7077 - accuracy: 0.4973
16064/25000 [==================>...........] - ETA: 31s - loss: 7.7077 - accuracy: 0.4973
16096/25000 [==================>...........] - ETA: 31s - loss: 7.7123 - accuracy: 0.4970
16128/25000 [==================>...........] - ETA: 31s - loss: 7.7180 - accuracy: 0.4967
16160/25000 [==================>...........] - ETA: 31s - loss: 7.7122 - accuracy: 0.4970
16192/25000 [==================>...........] - ETA: 31s - loss: 7.7130 - accuracy: 0.4970
16224/25000 [==================>...........] - ETA: 30s - loss: 7.7139 - accuracy: 0.4969
16256/25000 [==================>...........] - ETA: 30s - loss: 7.7138 - accuracy: 0.4969
16288/25000 [==================>...........] - ETA: 30s - loss: 7.7109 - accuracy: 0.4971
16320/25000 [==================>...........] - ETA: 30s - loss: 7.7117 - accuracy: 0.4971
16352/25000 [==================>...........] - ETA: 30s - loss: 7.7144 - accuracy: 0.4969
16384/25000 [==================>...........] - ETA: 30s - loss: 7.7134 - accuracy: 0.4969
16416/25000 [==================>...........] - ETA: 30s - loss: 7.7124 - accuracy: 0.4970
16448/25000 [==================>...........] - ETA: 30s - loss: 7.7086 - accuracy: 0.4973
16480/25000 [==================>...........] - ETA: 30s - loss: 7.7141 - accuracy: 0.4969
16512/25000 [==================>...........] - ETA: 29s - loss: 7.7131 - accuracy: 0.4970
16544/25000 [==================>...........] - ETA: 29s - loss: 7.7148 - accuracy: 0.4969
16576/25000 [==================>...........] - ETA: 29s - loss: 7.7147 - accuracy: 0.4969
16608/25000 [==================>...........] - ETA: 29s - loss: 7.7211 - accuracy: 0.4964
16640/25000 [==================>...........] - ETA: 29s - loss: 7.7219 - accuracy: 0.4964
16672/25000 [===================>..........] - ETA: 29s - loss: 7.7255 - accuracy: 0.4962
16704/25000 [===================>..........] - ETA: 29s - loss: 7.7263 - accuracy: 0.4961
16736/25000 [===================>..........] - ETA: 29s - loss: 7.7243 - accuracy: 0.4962
16768/25000 [===================>..........] - ETA: 29s - loss: 7.7224 - accuracy: 0.4964
16800/25000 [===================>..........] - ETA: 28s - loss: 7.7196 - accuracy: 0.4965
16832/25000 [===================>..........] - ETA: 28s - loss: 7.7158 - accuracy: 0.4968
16864/25000 [===================>..........] - ETA: 28s - loss: 7.7121 - accuracy: 0.4970
16896/25000 [===================>..........] - ETA: 28s - loss: 7.7147 - accuracy: 0.4969
16928/25000 [===================>..........] - ETA: 28s - loss: 7.7164 - accuracy: 0.4968
16960/25000 [===================>..........] - ETA: 28s - loss: 7.7163 - accuracy: 0.4968
16992/25000 [===================>..........] - ETA: 28s - loss: 7.7199 - accuracy: 0.4965
17024/25000 [===================>..........] - ETA: 28s - loss: 7.7153 - accuracy: 0.4968
17056/25000 [===================>..........] - ETA: 27s - loss: 7.7152 - accuracy: 0.4968
17088/25000 [===================>..........] - ETA: 27s - loss: 7.7205 - accuracy: 0.4965
17120/25000 [===================>..........] - ETA: 27s - loss: 7.7221 - accuracy: 0.4964
17152/25000 [===================>..........] - ETA: 27s - loss: 7.7203 - accuracy: 0.4965
17184/25000 [===================>..........] - ETA: 27s - loss: 7.7210 - accuracy: 0.4965
17216/25000 [===================>..........] - ETA: 27s - loss: 7.7227 - accuracy: 0.4963
17248/25000 [===================>..........] - ETA: 27s - loss: 7.7191 - accuracy: 0.4966
17280/25000 [===================>..........] - ETA: 27s - loss: 7.7199 - accuracy: 0.4965
17312/25000 [===================>..........] - ETA: 27s - loss: 7.7215 - accuracy: 0.4964
17344/25000 [===================>..........] - ETA: 26s - loss: 7.7241 - accuracy: 0.4963
17376/25000 [===================>..........] - ETA: 26s - loss: 7.7240 - accuracy: 0.4963
17408/25000 [===================>..........] - ETA: 26s - loss: 7.7212 - accuracy: 0.4964
17440/25000 [===================>..........] - ETA: 26s - loss: 7.7167 - accuracy: 0.4967
17472/25000 [===================>..........] - ETA: 26s - loss: 7.7123 - accuracy: 0.4970
17504/25000 [====================>.........] - ETA: 26s - loss: 7.7139 - accuracy: 0.4969
17536/25000 [====================>.........] - ETA: 26s - loss: 7.7156 - accuracy: 0.4968
17568/25000 [====================>.........] - ETA: 26s - loss: 7.7138 - accuracy: 0.4969
17600/25000 [====================>.........] - ETA: 26s - loss: 7.7171 - accuracy: 0.4967
17632/25000 [====================>.........] - ETA: 25s - loss: 7.7197 - accuracy: 0.4965
17664/25000 [====================>.........] - ETA: 25s - loss: 7.7178 - accuracy: 0.4967
17696/25000 [====================>.........] - ETA: 25s - loss: 7.7151 - accuracy: 0.4968
17728/25000 [====================>.........] - ETA: 25s - loss: 7.7151 - accuracy: 0.4968
17760/25000 [====================>.........] - ETA: 25s - loss: 7.7141 - accuracy: 0.4969
17792/25000 [====================>.........] - ETA: 25s - loss: 7.7175 - accuracy: 0.4967
17824/25000 [====================>.........] - ETA: 25s - loss: 7.7200 - accuracy: 0.4965
17856/25000 [====================>.........] - ETA: 25s - loss: 7.7190 - accuracy: 0.4966
17888/25000 [====================>.........] - ETA: 25s - loss: 7.7189 - accuracy: 0.4966
17920/25000 [====================>.........] - ETA: 24s - loss: 7.7171 - accuracy: 0.4967
17952/25000 [====================>.........] - ETA: 24s - loss: 7.7196 - accuracy: 0.4965
17984/25000 [====================>.........] - ETA: 24s - loss: 7.7169 - accuracy: 0.4967
18016/25000 [====================>.........] - ETA: 24s - loss: 7.7168 - accuracy: 0.4967
18048/25000 [====================>.........] - ETA: 24s - loss: 7.7116 - accuracy: 0.4971
18080/25000 [====================>.........] - ETA: 24s - loss: 7.7099 - accuracy: 0.4972
18112/25000 [====================>.........] - ETA: 24s - loss: 7.7089 - accuracy: 0.4972
18144/25000 [====================>.........] - ETA: 24s - loss: 7.7072 - accuracy: 0.4974
18176/25000 [====================>.........] - ETA: 24s - loss: 7.7063 - accuracy: 0.4974
18208/25000 [====================>.........] - ETA: 23s - loss: 7.7028 - accuracy: 0.4976
18240/25000 [====================>.........] - ETA: 23s - loss: 7.7002 - accuracy: 0.4978
18272/25000 [====================>.........] - ETA: 23s - loss: 7.7002 - accuracy: 0.4978
18304/25000 [====================>.........] - ETA: 23s - loss: 7.6993 - accuracy: 0.4979
18336/25000 [=====================>........] - ETA: 23s - loss: 7.7051 - accuracy: 0.4975
18368/25000 [=====================>........] - ETA: 23s - loss: 7.7025 - accuracy: 0.4977
18400/25000 [=====================>........] - ETA: 23s - loss: 7.7066 - accuracy: 0.4974
18432/25000 [=====================>........] - ETA: 23s - loss: 7.7065 - accuracy: 0.4974
18464/25000 [=====================>........] - ETA: 23s - loss: 7.7073 - accuracy: 0.4973
18496/25000 [=====================>........] - ETA: 22s - loss: 7.7056 - accuracy: 0.4975
18528/25000 [=====================>........] - ETA: 22s - loss: 7.7030 - accuracy: 0.4976
18560/25000 [=====================>........] - ETA: 22s - loss: 7.7063 - accuracy: 0.4974
18592/25000 [=====================>........] - ETA: 22s - loss: 7.7103 - accuracy: 0.4971
18624/25000 [=====================>........] - ETA: 22s - loss: 7.7127 - accuracy: 0.4970
18656/25000 [=====================>........] - ETA: 22s - loss: 7.7102 - accuracy: 0.4972
18688/25000 [=====================>........] - ETA: 22s - loss: 7.7101 - accuracy: 0.4972
18720/25000 [=====================>........] - ETA: 22s - loss: 7.7125 - accuracy: 0.4970
18752/25000 [=====================>........] - ETA: 21s - loss: 7.7124 - accuracy: 0.4970
18784/25000 [=====================>........] - ETA: 21s - loss: 7.7131 - accuracy: 0.4970
18816/25000 [=====================>........] - ETA: 21s - loss: 7.7123 - accuracy: 0.4970
18848/25000 [=====================>........] - ETA: 21s - loss: 7.7138 - accuracy: 0.4969
18880/25000 [=====================>........] - ETA: 21s - loss: 7.7145 - accuracy: 0.4969
18912/25000 [=====================>........] - ETA: 21s - loss: 7.7169 - accuracy: 0.4967
18944/25000 [=====================>........] - ETA: 21s - loss: 7.7225 - accuracy: 0.4964
18976/25000 [=====================>........] - ETA: 21s - loss: 7.7208 - accuracy: 0.4965
19008/25000 [=====================>........] - ETA: 21s - loss: 7.7199 - accuracy: 0.4965
19040/25000 [=====================>........] - ETA: 20s - loss: 7.7190 - accuracy: 0.4966
19072/25000 [=====================>........] - ETA: 20s - loss: 7.7221 - accuracy: 0.4964
19104/25000 [=====================>........] - ETA: 20s - loss: 7.7212 - accuracy: 0.4964
19136/25000 [=====================>........] - ETA: 20s - loss: 7.7235 - accuracy: 0.4963
19168/25000 [======================>.......] - ETA: 20s - loss: 7.7226 - accuracy: 0.4963
19200/25000 [======================>.......] - ETA: 20s - loss: 7.7265 - accuracy: 0.4961
19232/25000 [======================>.......] - ETA: 20s - loss: 7.7296 - accuracy: 0.4959
19264/25000 [======================>.......] - ETA: 20s - loss: 7.7287 - accuracy: 0.4960
19296/25000 [======================>.......] - ETA: 20s - loss: 7.7286 - accuracy: 0.4960
19328/25000 [======================>.......] - ETA: 19s - loss: 7.7261 - accuracy: 0.4961
19360/25000 [======================>.......] - ETA: 19s - loss: 7.7213 - accuracy: 0.4964
19392/25000 [======================>.......] - ETA: 19s - loss: 7.7251 - accuracy: 0.4962
19424/25000 [======================>.......] - ETA: 19s - loss: 7.7242 - accuracy: 0.4962
19456/25000 [======================>.......] - ETA: 19s - loss: 7.7249 - accuracy: 0.4962
19488/25000 [======================>.......] - ETA: 19s - loss: 7.7248 - accuracy: 0.4962
19520/25000 [======================>.......] - ETA: 19s - loss: 7.7255 - accuracy: 0.4962
19552/25000 [======================>.......] - ETA: 19s - loss: 7.7231 - accuracy: 0.4963
19584/25000 [======================>.......] - ETA: 19s - loss: 7.7246 - accuracy: 0.4962
19616/25000 [======================>.......] - ETA: 18s - loss: 7.7213 - accuracy: 0.4964
19648/25000 [======================>.......] - ETA: 18s - loss: 7.7220 - accuracy: 0.4964
19680/25000 [======================>.......] - ETA: 18s - loss: 7.7235 - accuracy: 0.4963
19712/25000 [======================>.......] - ETA: 18s - loss: 7.7234 - accuracy: 0.4963
19744/25000 [======================>.......] - ETA: 18s - loss: 7.7264 - accuracy: 0.4961
19776/25000 [======================>.......] - ETA: 18s - loss: 7.7286 - accuracy: 0.4960
19808/25000 [======================>.......] - ETA: 18s - loss: 7.7285 - accuracy: 0.4960
19840/25000 [======================>.......] - ETA: 18s - loss: 7.7292 - accuracy: 0.4959
19872/25000 [======================>.......] - ETA: 18s - loss: 7.7299 - accuracy: 0.4959
19904/25000 [======================>.......] - ETA: 17s - loss: 7.7298 - accuracy: 0.4959
19936/25000 [======================>.......] - ETA: 17s - loss: 7.7305 - accuracy: 0.4958
19968/25000 [======================>.......] - ETA: 17s - loss: 7.7304 - accuracy: 0.4958
20000/25000 [=======================>......] - ETA: 17s - loss: 7.7295 - accuracy: 0.4959
20032/25000 [=======================>......] - ETA: 17s - loss: 7.7317 - accuracy: 0.4958
20064/25000 [=======================>......] - ETA: 17s - loss: 7.7308 - accuracy: 0.4958
20096/25000 [=======================>......] - ETA: 17s - loss: 7.7307 - accuracy: 0.4958
20128/25000 [=======================>......] - ETA: 17s - loss: 7.7268 - accuracy: 0.4961
20160/25000 [=======================>......] - ETA: 17s - loss: 7.7282 - accuracy: 0.4960
20192/25000 [=======================>......] - ETA: 16s - loss: 7.7251 - accuracy: 0.4962
20224/25000 [=======================>......] - ETA: 16s - loss: 7.7235 - accuracy: 0.4963
20256/25000 [=======================>......] - ETA: 16s - loss: 7.7196 - accuracy: 0.4965
20288/25000 [=======================>......] - ETA: 16s - loss: 7.7210 - accuracy: 0.4965
20320/25000 [=======================>......] - ETA: 16s - loss: 7.7187 - accuracy: 0.4966
20352/25000 [=======================>......] - ETA: 16s - loss: 7.7156 - accuracy: 0.4968
20384/25000 [=======================>......] - ETA: 16s - loss: 7.7155 - accuracy: 0.4968
20416/25000 [=======================>......] - ETA: 16s - loss: 7.7162 - accuracy: 0.4968
20448/25000 [=======================>......] - ETA: 15s - loss: 7.7146 - accuracy: 0.4969
20480/25000 [=======================>......] - ETA: 15s - loss: 7.7160 - accuracy: 0.4968
20512/25000 [=======================>......] - ETA: 15s - loss: 7.7152 - accuracy: 0.4968
20544/25000 [=======================>......] - ETA: 15s - loss: 7.7174 - accuracy: 0.4967
20576/25000 [=======================>......] - ETA: 15s - loss: 7.7173 - accuracy: 0.4967
20608/25000 [=======================>......] - ETA: 15s - loss: 7.7180 - accuracy: 0.4967
20640/25000 [=======================>......] - ETA: 15s - loss: 7.7194 - accuracy: 0.4966
20672/25000 [=======================>......] - ETA: 15s - loss: 7.7178 - accuracy: 0.4967
20704/25000 [=======================>......] - ETA: 15s - loss: 7.7214 - accuracy: 0.4964
20736/25000 [=======================>......] - ETA: 14s - loss: 7.7213 - accuracy: 0.4964
20768/25000 [=======================>......] - ETA: 14s - loss: 7.7168 - accuracy: 0.4967
20800/25000 [=======================>......] - ETA: 14s - loss: 7.7138 - accuracy: 0.4969
20832/25000 [=======================>......] - ETA: 14s - loss: 7.7152 - accuracy: 0.4968
20864/25000 [========================>.....] - ETA: 14s - loss: 7.7129 - accuracy: 0.4970
20896/25000 [========================>.....] - ETA: 14s - loss: 7.7106 - accuracy: 0.4971
20928/25000 [========================>.....] - ETA: 14s - loss: 7.7128 - accuracy: 0.4970
20960/25000 [========================>.....] - ETA: 14s - loss: 7.7156 - accuracy: 0.4968
20992/25000 [========================>.....] - ETA: 14s - loss: 7.7134 - accuracy: 0.4970
21024/25000 [========================>.....] - ETA: 13s - loss: 7.7169 - accuracy: 0.4967
21056/25000 [========================>.....] - ETA: 13s - loss: 7.7176 - accuracy: 0.4967
21088/25000 [========================>.....] - ETA: 13s - loss: 7.7153 - accuracy: 0.4968
21120/25000 [========================>.....] - ETA: 13s - loss: 7.7145 - accuracy: 0.4969
21152/25000 [========================>.....] - ETA: 13s - loss: 7.7108 - accuracy: 0.4971
21184/25000 [========================>.....] - ETA: 13s - loss: 7.7072 - accuracy: 0.4974
21216/25000 [========================>.....] - ETA: 13s - loss: 7.7071 - accuracy: 0.4974
21248/25000 [========================>.....] - ETA: 13s - loss: 7.7005 - accuracy: 0.4978
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6983 - accuracy: 0.4979
21312/25000 [========================>.....] - ETA: 12s - loss: 7.6997 - accuracy: 0.4978
21344/25000 [========================>.....] - ETA: 12s - loss: 7.6997 - accuracy: 0.4978
21376/25000 [========================>.....] - ETA: 12s - loss: 7.6996 - accuracy: 0.4978
21408/25000 [========================>.....] - ETA: 12s - loss: 7.6981 - accuracy: 0.4979
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6981 - accuracy: 0.4979
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6959 - accuracy: 0.4981
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6944 - accuracy: 0.4982
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6951 - accuracy: 0.4981
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6986 - accuracy: 0.4979
21600/25000 [========================>.....] - ETA: 11s - loss: 7.6971 - accuracy: 0.4980
21632/25000 [========================>.....] - ETA: 11s - loss: 7.6971 - accuracy: 0.4980
21664/25000 [========================>.....] - ETA: 11s - loss: 7.7006 - accuracy: 0.4978
21696/25000 [=========================>....] - ETA: 11s - loss: 7.6984 - accuracy: 0.4979
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6991 - accuracy: 0.4979
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6962 - accuracy: 0.4981
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6927 - accuracy: 0.4983
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6926 - accuracy: 0.4983
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6968 - accuracy: 0.4980
21888/25000 [=========================>....] - ETA: 10s - loss: 7.6953 - accuracy: 0.4981
21920/25000 [=========================>....] - ETA: 10s - loss: 7.6988 - accuracy: 0.4979
21952/25000 [=========================>....] - ETA: 10s - loss: 7.6981 - accuracy: 0.4980
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6980 - accuracy: 0.4980
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6980 - accuracy: 0.4980
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6965 - accuracy: 0.4980
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6958 - accuracy: 0.4981
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6951 - accuracy: 0.4981
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6929 - accuracy: 0.4983
22176/25000 [=========================>....] - ETA: 9s - loss: 7.6915 - accuracy: 0.4984 
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6908 - accuracy: 0.4984
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6908 - accuracy: 0.4984
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6887 - accuracy: 0.4986
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6921 - accuracy: 0.4983
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6927 - accuracy: 0.4983
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6913 - accuracy: 0.4984
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6913 - accuracy: 0.4984
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6912 - accuracy: 0.4984
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6905 - accuracy: 0.4984
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6932 - accuracy: 0.4983
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6952 - accuracy: 0.4981
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6958 - accuracy: 0.4981
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6951 - accuracy: 0.4981
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6951 - accuracy: 0.4981
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6944 - accuracy: 0.4982
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6930 - accuracy: 0.4983
22720/25000 [==========================>...] - ETA: 7s - loss: 7.6923 - accuracy: 0.4983
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6976 - accuracy: 0.4980
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6976 - accuracy: 0.4980
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6982 - accuracy: 0.4979
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6982 - accuracy: 0.4979
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6981 - accuracy: 0.4979
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6987 - accuracy: 0.4979
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6980 - accuracy: 0.4980
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6987 - accuracy: 0.4979
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6966 - accuracy: 0.4980
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6966 - accuracy: 0.4980
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6965 - accuracy: 0.4980
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6965 - accuracy: 0.4981
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6984 - accuracy: 0.4979
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6964 - accuracy: 0.4981
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6964 - accuracy: 0.4981
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6910 - accuracy: 0.4984
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6897 - accuracy: 0.4985
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6910 - accuracy: 0.4984
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6916 - accuracy: 0.4984
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6955 - accuracy: 0.4981
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6994 - accuracy: 0.4979
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6980 - accuracy: 0.4980
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6967 - accuracy: 0.4980
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6980 - accuracy: 0.4980
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6953 - accuracy: 0.4981
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6953 - accuracy: 0.4981
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6939 - accuracy: 0.4982
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6971 - accuracy: 0.4980
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6939 - accuracy: 0.4982
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6925 - accuracy: 0.4983
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6925 - accuracy: 0.4983
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6925 - accuracy: 0.4983
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6911 - accuracy: 0.4984
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6904 - accuracy: 0.4984
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6872 - accuracy: 0.4987
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6872 - accuracy: 0.4987
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6852 - accuracy: 0.4988
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6846 - accuracy: 0.4988
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6884 - accuracy: 0.4986
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6890 - accuracy: 0.4985
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6890 - accuracy: 0.4985
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6889 - accuracy: 0.4985
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6921 - accuracy: 0.4983
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6908 - accuracy: 0.4984
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6888 - accuracy: 0.4986
24192/25000 [============================>.] - ETA: 2s - loss: 7.6888 - accuracy: 0.4986
24224/25000 [============================>.] - ETA: 2s - loss: 7.6856 - accuracy: 0.4988
24256/25000 [============================>.] - ETA: 2s - loss: 7.6862 - accuracy: 0.4987
24288/25000 [============================>.] - ETA: 2s - loss: 7.6856 - accuracy: 0.4988
24320/25000 [============================>.] - ETA: 2s - loss: 7.6843 - accuracy: 0.4988
24352/25000 [============================>.] - ETA: 2s - loss: 7.6836 - accuracy: 0.4989
24384/25000 [============================>.] - ETA: 2s - loss: 7.6823 - accuracy: 0.4990
24416/25000 [============================>.] - ETA: 2s - loss: 7.6823 - accuracy: 0.4990
24448/25000 [============================>.] - ETA: 1s - loss: 7.6829 - accuracy: 0.4989
24480/25000 [============================>.] - ETA: 1s - loss: 7.6810 - accuracy: 0.4991
24512/25000 [============================>.] - ETA: 1s - loss: 7.6810 - accuracy: 0.4991
24544/25000 [============================>.] - ETA: 1s - loss: 7.6797 - accuracy: 0.4991
24576/25000 [============================>.] - ETA: 1s - loss: 7.6785 - accuracy: 0.4992
24608/25000 [============================>.] - ETA: 1s - loss: 7.6760 - accuracy: 0.4994
24640/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24672/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24704/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24736/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24768/25000 [============================>.] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
24800/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24864/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24896/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24928/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24960/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 106s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
Loading data...
Using TensorFlow backend.





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//fashion_MNIST_mlmodels.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//mnist_mlmodels_.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//tensorflow_1_lstm.ipynb 

/home/runner/work/mlmodels/mlmodels
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
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
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
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
test

  #### Module init   ############################################ 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term

  <module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'> 

  #### Loading params   ############################################## 

  ############# Data, Params preparation   ################# 

  #### Model init   ############################################ 

  <mlmodels.model_tf.1_lstm.Model object at 0x7f90031f2a58> 

  #### Fit   ######################################################## 
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

  #### Predict   #################################################### 
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
5  0.745928  0.883387  0.838176  0.904464  0.904464  0.370110
6  1.000000  0.881878  0.467996  0.486496  0.486496  1.000000
7  0.216516  0.077549  0.433808  0.329598  0.329598  0.318466
8  0.195249  0.000000  0.000000  0.000000  0.000000  0.671960
9  0.000000  0.173783  0.369041  0.411721  0.411721  0.304384
[[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00]
 [ 2.46557109e-02  1.77110173e-02 -2.48994119e-02 -2.83730682e-03
   3.26460972e-02  2.49399208e-02]
 [ 6.16239756e-03  5.19947633e-02 -1.81246735e-02 -7.66012818e-06
   4.00829641e-03  2.00418741e-01]
 [-1.02196291e-01  3.02486181e-01  1.13393664e-01  1.38518244e-01
   2.12468533e-03  7.74776787e-02]
 [-2.64209881e-02  9.81474146e-02  9.47099477e-02  3.74387726e-02
  -1.21746920e-02 -4.97656539e-02]
 [-3.47438574e-01  2.76346684e-01  1.63687557e-01 -1.71751305e-01
   2.15977713e-01  2.08587199e-02]
 [ 8.35118666e-02  4.10697222e-01 -2.32657611e-01  2.03155428e-01
   3.31024230e-01 -1.62983641e-01]
 [-4.21184689e-01  4.36074674e-01 -2.38295458e-02 -2.90233850e-01
   4.76553105e-02  1.10260308e-01]
 [-2.03758433e-01  5.67196906e-01  3.58540844e-03  3.72049212e-02
  -4.78368662e-02  5.94163314e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00]]

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 
model_tf/1_lstm.py
model_tf.1_lstm.py
<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>
<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>

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
{'loss': 0.444349929690361, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed [Errno 13] Permission denied: '/model/'
model_tf/1_lstm.py
model_tf.1_lstm.py
<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>
<module 'mlmodels.model_tf.1_lstm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py'>

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
{'loss': 0.6063923984766006, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed [Errno 13] Permission denied: '/model/'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_home_retail.ipynb 

Deprecaton set to False
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//lightgbm_home_retail.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0mdata_path[0m [0;34m=[0m [0;34m'hyper_lightgbm_home_retail.json'[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mdata_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpdict[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'hyper_lightgbm_home_retail.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//gluon_automl.ipynb 

/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB
  Optimizer.opt_registry[name].__name__))
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv | Columns = 15 / 15 | Rows = 39073 -> 39073
Warning: `hyperparameter_tune=True` is currently experimental and may cause the process to hang. Setting `auto_stack=True` instead is recommended to achieve maximum quality models.
Beginning AutoGluon training ... Time limit = 120s
AutoGluon will save models to dataset/
Train Data Rows:    39073
Train Data Columns: 15
Preprocessing data ...
Here are the first 10 unique label values in your data:  [' Tech-support' ' Transport-moving' ' Other-service' ' ?'
 ' Handlers-cleaners' ' Sales' ' Craft-repair' ' Adm-clerical'
 ' Exec-managerial' ' Prof-specialty']
AutoGluon infers your prediction problem is: multiclass  (because dtype of label-column == object)
If this is wrong, please specify `problem_type` argument in fit() instead (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])

Feature Generator processed 39073 data points with 14 features
Original Features:
	int features: 6
	object features: 8
Generated Features:
	int features: 0
All Features:
	int features: 6
	object features: 8
	Data preprocessing and feature engineering runtime = 0.24s ...
AutoGluon will gauge predictive performance using evaluation metric: accuracy
To change this, specify the eval_metric argument of fit()
AutoGluon will early stop models using evaluation metric: accuracy
Saving dataset/learner.pkl
Beginning hyperparameter tuning for Gradient Boosting Model...
Hyperparameter search space for Gradient Boosting Model: 
num_leaves:   Int: lower=26, upper=30
learning_rate:   Real: lower=0.005, upper=0.2
feature_fraction:   Real: lower=0.75, upper=1.0
min_data_in_leaf:   Int: lower=2, upper=30
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 360, in train_single_full
    Y_train=y_train, Y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/models/lgb/lgb_model.py", line 283, in hyperparameter_tune
    directory=directory, lgb_model=self, **params_copy)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/decorator.py", line 69, in register_args
    self.update(**kwvars)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/decorator.py", line 79, in update
    hp = v.get_hp(name=k)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/space.py", line 451, in get_hp
    default_value=self._default)
  File "ConfigSpace/hyperparameters.pyx", line 773, in ConfigSpace.hyperparameters.UniformIntegerHyperparameter.__init__
  File "ConfigSpace/hyperparameters.pyx", line 843, in ConfigSpace.hyperparameters.UniformIntegerHyperparameter.check_default
Warning: Exception caused LightGBMClassifier to fail during hyperparameter tuning... Skipping this model.
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 360, in train_single_full
    Y_train=y_train, Y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/models/lgb/lgb_model.py", line 283, in hyperparameter_tune
    directory=directory, lgb_model=self, **params_copy)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/decorator.py", line 69, in register_args
    self.update(**kwvars)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/decorator.py", line 79, in update
    hp = v.get_hp(name=k)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/space.py", line 451, in get_hp
    default_value=self._default)
  File "ConfigSpace/hyperparameters.pyx", line 773, in ConfigSpace.hyperparameters.UniformIntegerHyperparameter.__init__
  File "ConfigSpace/hyperparameters.pyx", line 843, in ConfigSpace.hyperparameters.UniformIntegerHyperparameter.check_default
ValueError: Illegal default value 36
Saving dataset/models/trainer.pkl
Beginning hyperparameter tuning for Neural Network...
Hyperparameter search space for Neural Network: 
network_type:   Categorical['widedeep', 'feedforward']
layers:   Categorical[[100], [1000], [200, 100], [300, 200, 100]]
activation:   Categorical['relu', 'softrelu', 'tanh']
embedding_size_factor:   Real: lower=0.5, upper=1.5
use_batchnorm:   Categorical[True, False]
dropout_prob:   Real: lower=0.0, upper=0.5
learning_rate:   Real: lower=0.0001, upper=0.01
weight_decay:   Real: lower=1e-12, upper=0.1
AutoGluon Neural Network infers features are of the following types:
{
    "continuous": [
        "age",
        "education-num",
        "hours-per-week"
    ],
    "skewed": [
        "fnlwgt",
        "capital-gain",
        "capital-loss"
    ],
    "onehot": [
        "sex",
        "class"
    ],
    "embed": [
        "workclass",
        "education",
        "marital-status",
        "relationship",
        "race",
        "native-country"
    ],
    "language": []
}


Saving dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Starting Experiments
Num of Finished Tasks is 0
Num of Pending Tasks is 5
  0%|          | 0/5 [00:00<?, ?it/s]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:52<01:18, 26.02s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.2767507736749157, 'embedding_size_factor': 1.0891398106779406, 'layers.choice': 1, 'learning_rate': 0.00186967071228134, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 2.6969258694499784e-09} and reward: 0.3762
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd1\xb6H\xe0\x84\xe4\xbbX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1m\x1d\xdd\xba"ZX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?^\xa1\xf7\xa4\x10\xff\x17X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\'*\x9aJ\xe7\x05Ou.' and reward: 0.3762
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd1\xb6H\xe0\x84\xe4\xbbX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1m\x1d\xdd\xba"ZX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?^\xa1\xf7\xa4\x10\xff\x17X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\'*\x9aJ\xe7\x05Ou.' and reward: 0.3762
 60%|██████    | 3/5 [02:12<01:24, 42.28s/it] 60%|██████    | 3/5 [02:12<01:28, 44.10s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.1361842402665089, 'embedding_size_factor': 0.8921858045779381, 'layers.choice': 1, 'learning_rate': 0.0003551898274821885, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 0.003076979822469271} and reward: 0.3552
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc1n|5\x16n\xfdX\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\x8c\xc9>\x93\xc4 X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?7G\x18\xb1bi3X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?i4\xe4\xf6\xa7\x96Ju.' and reward: 0.3552
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc1n|5\x16n\xfdX\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\x8c\xc9>\x93\xc4 X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?7G\x18\xb1bi3X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?i4\xe4\xf6\xa7\x96Ju.' and reward: 0.3552
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 230.219566822052
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -113.1s of remaining time.
Ensemble size: 30
Ensemble weights: 
[0.66666667 0.23333333 0.1       ]
	0.392	 = Validation accuracy score
	1.06s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 234.19s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
test

  #### Module init   ############################################ 

  <module 'mlmodels.model_gluon.gluon_automl' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluon_automl.py'> 

  #### Loading params   ############################################## 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB
  Optimizer.opt_registry[name].__name__))
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 503, in main
    test_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 433, in test_cli
    test_module(arg.model_uri, param_pars=param_pars)  # '1_lstm'
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 255, in test_module
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluon_automl.py", line 109, in get_params
    return model_pars, data_pars, compute_pars, out_pars
UnboundLocalError: local variable 'model_pars' referenced before assignment





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vision_mnist.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_glass.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      8[0m [0;32mimport[0m [0mjson[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m [0;34m[0m[0m
[0;32m---> 10[0;31m [0mprint[0m[0;34m([0m [0mos[0m[0;34m.[0m[0mgetcwd[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mNameError[0m: name 'os' is not defined





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//gluon_automl_titanic.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//gluon_automl_titanic.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      8[0m     [0mchoice[0m[0;34m=[0m[0;34m'json'[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m     [0mconfig_mode[0m[0;34m=[0m [0;34m'test'[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m     [0mdata_path[0m[0;34m=[0m [0;34m'../mlmodels/dataset/json/gluon_automl.json'[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m )

[0;32m~/work/mlmodels/mlmodels/mlmodels/model_gluon/gluon_automl.py[0m in [0;36mget_params[0;34m(choice, data_path, config_mode, **kw)[0m
[1;32m     80[0m             __file__)).parent.parent / "model_gluon/gluon_automl.json" if data_path == "dataset/" else data_path
[1;32m     81[0m [0;34m[0m[0m
[0;32m---> 82[0;31m         [0;32mwith[0m [0mopen[0m[0;34m([0m[0mdata_path[0m[0;34m,[0m [0mencoding[0m[0;34m=[0m[0;34m'utf-8'[0m[0;34m)[0m [0;32mas[0m [0mconfig_f[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     83[0m             [0mconfig[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mconfig_f[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     84[0m             [0mconfig[0m [0;34m=[0m [0mconfig[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '../mlmodels/dataset/json/gluon_automl.json'
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB
  Optimizer.opt_registry[name].__name__))





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vison_fashion_MNIST.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//tensorflow__lstm_json.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//tensorflow__lstm_json.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      5[0m [0;32mimport[0m [0mjson[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m
[0;32m----> 7[0;31m [0mprint[0m[0;34m([0m [0mos[0m[0;34m.[0m[0mgetcwd[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mNameError[0m: name 'os' is not defined





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//lightgbm.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      4[0m [0mdata_path[0m [0;34m=[0m [0;34m'lightgbm_titanic.json'[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m
[0;32m----> 6[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mdata_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      7[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpdict[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'lightgbm_titanic.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn_titanic_randomForest.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     69[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 70[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     71[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    127[0m [0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m

[0;31mModuleNotFoundError[0m: No module named 'mlmodels.model_sklearn.sklearn'

During handling of the above exception, another exception occurred:

[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     81[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 82[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     83[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: tuple index out of range

During handling of the above exception, another exception occurred:

[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0mmodel_uri[0m    [0;34m=[0m [0;34m"model_sklearn.sklearn.py"[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0mmodule[0m        [0;34m=[0m  [0mmodule_load[0m[0;34m([0m [0mmodel_uri[0m[0;34m=[0m [0mmodel_uri[0m [0;34m)[0m                           [0;31m# Load file definition[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={

[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     85[0m [0;34m[0m[0m
[1;32m     86[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 87[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     88[0m [0;34m[0m[0m
[1;32m     89[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     69[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 70[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     71[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    127[0m [0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m

[0;31mModuleNotFoundError[0m: No module named 'mlmodels.model_sklearn.sklearn'

During handling of the above exception, another exception occurred:

[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     81[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 82[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     83[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: tuple index out of range

During handling of the above exception, another exception occurred:

[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//sklearn.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;32mfrom[0m [0mmlmodels[0m[0;34m.[0m[0mmodels[0m [0;32mimport[0m [0mmodule_load[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0mmodule[0m        [0;34m=[0m  [0mmodule_load[0m[0;34m([0m [0mmodel_uri[0m[0;34m=[0m [0mmodel_uri[0m [0;34m)[0m                           [0;31m# Load file definition[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0mmodel[0m         [0;34m=[0m  [0mmodule[0m[0;34m.[0m[0mModel[0m[0;34m([0m[0mmodel_pars[0m[0;34m=[0m[0mmodel_pars[0m[0;34m,[0m [0mdata_pars[0m[0;34m=[0m[0mdata_pars[0m[0;34m,[0m [0mcompute_pars[0m[0;34m=[0m[0mcompute_pars[0m[0;34m)[0m             [0;31m# Create Model instance[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mmodel[0m[0;34m,[0m [0msess[0m   [0;34m=[0m  [0mmodule[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mdata_pars[0m[0;34m=[0m[0mdata_pars[0m[0;34m,[0m [0mcompute_pars[0m[0;34m=[0m[0mcompute_pars[0m[0;34m,[0m [0mout_pars[0m[0;34m=[0m[0mout_pars[0m[0;34m)[0m          [0;31m# fit the model[0m[0;34m[0m[0;34m[0m[0m

[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     85[0m [0;34m[0m[0m
[1;32m     86[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 87[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     88[0m [0;34m[0m[0m
[1;32m     89[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb 

Deprecaton set to False
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;32mimport[0m [0mjson[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mdata_path[0m [0;34m=[0m [0;34m'../mlmodels/dataset/json/hyper_titanic_randomForest.json'[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mdata_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpdict[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '../mlmodels/dataset/json/hyper_titanic_randomForest.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vision_mnist.py 

[0;36m  File [0;32m"/home/runner/work/mlmodels/mlmodels/mlmodels/example/vision_mnist.py"[0;36m, line [0;32m15[0m
[0;31m    !git clone https://github.com/ahmed3bbas/mlmodels.git[0m
[0m    ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax






 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//arun_model.py 

<module 'mlmodels' from '/home/runner/work/mlmodels/mlmodels/mlmodels/__init__.py'>
/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/ardmn.json
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example/arun_model.py[0m in [0;36m<module>[0;34m[0m
[1;32m     25[0m [0;31m# Model Parameters[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
[1;32m     26[0m [0;31m# model_pars, data_pars, compute_pars, out_pars[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 27[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m[0mconfig_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     28[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpdict[0m   [0;34m)[0m   [0;31m###Normalize path[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/ardmn.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_glass.py 

Deprecaton set to False
/home/runner/work/mlmodels/mlmodels
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example/lightgbm_glass.py[0m in [0;36m<module>[0;34m[0m
[1;32m     20[0m [0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m
[0;32m---> 22[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mconfig_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     23[0m [0mprint[0m[0;34m([0m[0mpars[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'lightgbm_glass.json'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//benchmark_timeseries_m4.py 






 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//arun_hyper.py 

[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example/arun_hyper.py[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;32mfrom[0m [0mmlmodels[0m[0;34m.[0m[0mmodels[0m [0;32mimport[0m [0mmodule_load[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mmlmodels[0m[0;34m.[0m[0mutil[0m [0;32mimport[0m [0mpath_norm_dict[0m[0;34m,[0m [0mpath_norm[0m[0;34m,[0m [0mparams_json_load[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0mprint[0m[0;34m([0m[0mmlmodels[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m

[0;31mNameError[0m: name 'mlmodels' is not defined





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/benchmark_timeseries_m4.py 






 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/benchmark_timeseries_m5.py 

[0;36m  File [0;32m"/home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py"[0;36m, line [0;32m248[0m
[0;31m    We then reshape the forecasts into the correct data shape for submission ...[0m
[0m          ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax






 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//benchmark_timeseries_m5.py 

[0;36m  File [0;32m"/home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py"[0;36m, line [0;32m248[0m
[0;31m    We then reshape the forecasts into the correct data shape for submission ...[0m
[0m          ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax

