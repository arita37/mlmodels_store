
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/f3819ba2d2a4e525f7f9ec0cef0a000f860c50b0', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'f3819ba2d2a4e525f7f9ec0cef0a000f860c50b0', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/f3819ba2d2a4e525f7f9ec0cef0a000f860c50b0

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/f3819ba2d2a4e525f7f9ec0cef0a000f860c50b0

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
 1671168/17464789 [=>............................] - ETA: 0s
 4268032/17464789 [======>.......................] - ETA: 0s
 6873088/17464789 [==========>...................] - ETA: 0s
 9576448/17464789 [===============>..............] - ETA: 0s
12279808/17464789 [====================>.........] - ETA: 0s
14925824/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 06:07:26.181276: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 06:07:26.186141: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-11 06:07:26.186284: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561206569280 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 06:07:26.186300: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

2020-05-11 06:07:26.679112: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:56 - loss: 8.1458 - accuracy: 0.46882020-05-11 06:07:26.811873: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

   64/25000 [..............................] - ETA: 3:16 - loss: 7.9062 - accuracy: 0.48442020-05-11 06:07:26.929655: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

   96/25000 [..............................] - ETA: 2:40 - loss: 7.9861 - accuracy: 0.47922020-05-11 06:07:27.042057: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

  128/25000 [..............................] - ETA: 2:23 - loss: 8.1458 - accuracy: 0.46882020-05-11 06:07:27.163062: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

  160/25000 [..............................] - ETA: 2:13 - loss: 8.2416 - accuracy: 0.4625
  192/25000 [..............................] - ETA: 2:06 - loss: 8.4652 - accuracy: 0.4479
  224/25000 [..............................] - ETA: 2:01 - loss: 8.0089 - accuracy: 0.4777
  256/25000 [..............................] - ETA: 1:57 - loss: 7.9661 - accuracy: 0.4805
  288/25000 [..............................] - ETA: 1:55 - loss: 7.9328 - accuracy: 0.4826
  320/25000 [..............................] - ETA: 1:52 - loss: 8.0020 - accuracy: 0.4781
  352/25000 [..............................] - ETA: 1:50 - loss: 8.0587 - accuracy: 0.4744
  384/25000 [..............................] - ETA: 1:49 - loss: 8.0659 - accuracy: 0.4740
  416/25000 [..............................] - ETA: 1:48 - loss: 7.8878 - accuracy: 0.4856
  448/25000 [..............................] - ETA: 1:47 - loss: 7.7693 - accuracy: 0.4933
  480/25000 [..............................] - ETA: 1:46 - loss: 7.7305 - accuracy: 0.4958
  512/25000 [..............................] - ETA: 1:45 - loss: 7.6966 - accuracy: 0.4980
  544/25000 [..............................] - ETA: 1:44 - loss: 7.5821 - accuracy: 0.5055
  576/25000 [..............................] - ETA: 1:44 - loss: 7.4270 - accuracy: 0.5156
  608/25000 [..............................] - ETA: 1:43 - loss: 7.4396 - accuracy: 0.5148
  640/25000 [..............................] - ETA: 1:42 - loss: 7.4510 - accuracy: 0.5141
  672/25000 [..............................] - ETA: 1:42 - loss: 7.5297 - accuracy: 0.5089
  704/25000 [..............................] - ETA: 1:41 - loss: 7.5359 - accuracy: 0.5085
  736/25000 [..............................] - ETA: 1:40 - loss: 7.5416 - accuracy: 0.5082
  768/25000 [..............................] - ETA: 1:40 - loss: 7.5269 - accuracy: 0.5091
  800/25000 [..............................] - ETA: 1:39 - loss: 7.5900 - accuracy: 0.5050
  832/25000 [..............................] - ETA: 1:39 - loss: 7.5560 - accuracy: 0.5072
  864/25000 [>.............................] - ETA: 1:38 - loss: 7.5069 - accuracy: 0.5104
  896/25000 [>.............................] - ETA: 1:38 - loss: 7.5297 - accuracy: 0.5089
  928/25000 [>.............................] - ETA: 1:38 - loss: 7.5344 - accuracy: 0.5086
  960/25000 [>.............................] - ETA: 1:37 - loss: 7.4909 - accuracy: 0.5115
  992/25000 [>.............................] - ETA: 1:37 - loss: 7.4657 - accuracy: 0.5131
 1024/25000 [>.............................] - ETA: 1:36 - loss: 7.4121 - accuracy: 0.5166
 1056/25000 [>.............................] - ETA: 1:36 - loss: 7.4198 - accuracy: 0.5161
 1088/25000 [>.............................] - ETA: 1:36 - loss: 7.4552 - accuracy: 0.5138
 1120/25000 [>.............................] - ETA: 1:36 - loss: 7.4476 - accuracy: 0.5143
 1152/25000 [>.............................] - ETA: 1:36 - loss: 7.4537 - accuracy: 0.5139
 1184/25000 [>.............................] - ETA: 1:35 - loss: 7.4724 - accuracy: 0.5127
 1216/25000 [>.............................] - ETA: 1:35 - loss: 7.5279 - accuracy: 0.5090
 1248/25000 [>.............................] - ETA: 1:35 - loss: 7.4700 - accuracy: 0.5128
 1280/25000 [>.............................] - ETA: 1:35 - loss: 7.4151 - accuracy: 0.5164
 1312/25000 [>.............................] - ETA: 1:34 - loss: 7.3861 - accuracy: 0.5183
 1344/25000 [>.............................] - ETA: 1:34 - loss: 7.3928 - accuracy: 0.5179
 1376/25000 [>.............................] - ETA: 1:34 - loss: 7.3769 - accuracy: 0.5189
 1408/25000 [>.............................] - ETA: 1:33 - loss: 7.3835 - accuracy: 0.5185
 1440/25000 [>.............................] - ETA: 1:33 - loss: 7.3898 - accuracy: 0.5181
 1472/25000 [>.............................] - ETA: 1:33 - loss: 7.4270 - accuracy: 0.5156
 1504/25000 [>.............................] - ETA: 1:33 - loss: 7.4219 - accuracy: 0.5160
 1536/25000 [>.............................] - ETA: 1:32 - loss: 7.4370 - accuracy: 0.5150
 1568/25000 [>.............................] - ETA: 1:32 - loss: 7.4417 - accuracy: 0.5147
 1600/25000 [>.............................] - ETA: 1:32 - loss: 7.4462 - accuracy: 0.5144
 1632/25000 [>.............................] - ETA: 1:32 - loss: 7.4411 - accuracy: 0.5147
 1664/25000 [>.............................] - ETA: 1:32 - loss: 7.4731 - accuracy: 0.5126
 1696/25000 [=>............................] - ETA: 1:32 - loss: 7.4496 - accuracy: 0.5142
 1728/25000 [=>............................] - ETA: 1:32 - loss: 7.4537 - accuracy: 0.5139
 1760/25000 [=>............................] - ETA: 1:31 - loss: 7.4837 - accuracy: 0.5119
 1792/25000 [=>............................] - ETA: 1:31 - loss: 7.4955 - accuracy: 0.5112
 1824/25000 [=>............................] - ETA: 1:31 - loss: 7.4733 - accuracy: 0.5126
 1856/25000 [=>............................] - ETA: 1:31 - loss: 7.4931 - accuracy: 0.5113
 1888/25000 [=>............................] - ETA: 1:31 - loss: 7.4961 - accuracy: 0.5111
 1920/25000 [=>............................] - ETA: 1:30 - loss: 7.5069 - accuracy: 0.5104
 1952/25000 [=>............................] - ETA: 1:30 - loss: 7.5017 - accuracy: 0.5108
 1984/25000 [=>............................] - ETA: 1:30 - loss: 7.5352 - accuracy: 0.5086
 2016/25000 [=>............................] - ETA: 1:29 - loss: 7.5449 - accuracy: 0.5079
 2048/25000 [=>............................] - ETA: 1:29 - loss: 7.5693 - accuracy: 0.5063
 2080/25000 [=>............................] - ETA: 1:29 - loss: 7.6003 - accuracy: 0.5043
 2112/25000 [=>............................] - ETA: 1:29 - loss: 7.6013 - accuracy: 0.5043
 2144/25000 [=>............................] - ETA: 1:29 - loss: 7.6094 - accuracy: 0.5037
 2176/25000 [=>............................] - ETA: 1:28 - loss: 7.6243 - accuracy: 0.5028
 2208/25000 [=>............................] - ETA: 1:28 - loss: 7.6388 - accuracy: 0.5018
 2240/25000 [=>............................] - ETA: 1:28 - loss: 7.6255 - accuracy: 0.5027
 2272/25000 [=>............................] - ETA: 1:28 - loss: 7.6059 - accuracy: 0.5040
 2304/25000 [=>............................] - ETA: 1:27 - loss: 7.5868 - accuracy: 0.5052
 2336/25000 [=>............................] - ETA: 1:27 - loss: 7.5682 - accuracy: 0.5064
 2368/25000 [=>............................] - ETA: 1:27 - loss: 7.5760 - accuracy: 0.5059
 2400/25000 [=>............................] - ETA: 1:27 - loss: 7.5836 - accuracy: 0.5054
 2432/25000 [=>............................] - ETA: 1:27 - loss: 7.5784 - accuracy: 0.5058
 2464/25000 [=>............................] - ETA: 1:27 - loss: 7.5857 - accuracy: 0.5053
 2496/25000 [=>............................] - ETA: 1:26 - loss: 7.5929 - accuracy: 0.5048
 2528/25000 [==>...........................] - ETA: 1:26 - loss: 7.5635 - accuracy: 0.5067
 2560/25000 [==>...........................] - ETA: 1:26 - loss: 7.5768 - accuracy: 0.5059
 2592/25000 [==>...........................] - ETA: 1:26 - loss: 7.5897 - accuracy: 0.5050
 2624/25000 [==>...........................] - ETA: 1:26 - loss: 7.5790 - accuracy: 0.5057
 2656/25000 [==>...........................] - ETA: 1:26 - loss: 7.5743 - accuracy: 0.5060
 2688/25000 [==>...........................] - ETA: 1:26 - loss: 7.5754 - accuracy: 0.5060
 2720/25000 [==>...........................] - ETA: 1:25 - loss: 7.5933 - accuracy: 0.5048
 2752/25000 [==>...........................] - ETA: 1:25 - loss: 7.6053 - accuracy: 0.5040
 2784/25000 [==>...........................] - ETA: 1:25 - loss: 7.6005 - accuracy: 0.5043
 2816/25000 [==>...........................] - ETA: 1:25 - loss: 7.6013 - accuracy: 0.5043
 2848/25000 [==>...........................] - ETA: 1:25 - loss: 7.6020 - accuracy: 0.5042
 2880/25000 [==>...........................] - ETA: 1:25 - loss: 7.6240 - accuracy: 0.5028
 2912/25000 [==>...........................] - ETA: 1:25 - loss: 7.6350 - accuracy: 0.5021
 2944/25000 [==>...........................] - ETA: 1:24 - loss: 7.6562 - accuracy: 0.5007
 2976/25000 [==>...........................] - ETA: 1:24 - loss: 7.6563 - accuracy: 0.5007
 3008/25000 [==>...........................] - ETA: 1:24 - loss: 7.6666 - accuracy: 0.5000
 3040/25000 [==>...........................] - ETA: 1:24 - loss: 7.6717 - accuracy: 0.4997
 3072/25000 [==>...........................] - ETA: 1:24 - loss: 7.6616 - accuracy: 0.5003
 3104/25000 [==>...........................] - ETA: 1:24 - loss: 7.6567 - accuracy: 0.5006
 3136/25000 [==>...........................] - ETA: 1:24 - loss: 7.6520 - accuracy: 0.5010
 3168/25000 [==>...........................] - ETA: 1:24 - loss: 7.6569 - accuracy: 0.5006
 3200/25000 [==>...........................] - ETA: 1:23 - loss: 7.6475 - accuracy: 0.5013
 3232/25000 [==>...........................] - ETA: 1:23 - loss: 7.6571 - accuracy: 0.5006
 3264/25000 [==>...........................] - ETA: 1:23 - loss: 7.6619 - accuracy: 0.5003
 3296/25000 [==>...........................] - ETA: 1:23 - loss: 7.6806 - accuracy: 0.4991
 3328/25000 [==>...........................] - ETA: 1:23 - loss: 7.6850 - accuracy: 0.4988
 3360/25000 [===>..........................] - ETA: 1:23 - loss: 7.6849 - accuracy: 0.4988
 3392/25000 [===>..........................] - ETA: 1:23 - loss: 7.6711 - accuracy: 0.4997
 3424/25000 [===>..........................] - ETA: 1:22 - loss: 7.6666 - accuracy: 0.5000
 3456/25000 [===>..........................] - ETA: 1:22 - loss: 7.6533 - accuracy: 0.5009
 3488/25000 [===>..........................] - ETA: 1:22 - loss: 7.6446 - accuracy: 0.5014
 3520/25000 [===>..........................] - ETA: 1:22 - loss: 7.6623 - accuracy: 0.5003
 3552/25000 [===>..........................] - ETA: 1:22 - loss: 7.6537 - accuracy: 0.5008
 3584/25000 [===>..........................] - ETA: 1:22 - loss: 7.6409 - accuracy: 0.5017
 3616/25000 [===>..........................] - ETA: 1:22 - loss: 7.6285 - accuracy: 0.5025
 3648/25000 [===>..........................] - ETA: 1:22 - loss: 7.6288 - accuracy: 0.5025
 3680/25000 [===>..........................] - ETA: 1:21 - loss: 7.6250 - accuracy: 0.5027
 3712/25000 [===>..........................] - ETA: 1:21 - loss: 7.6212 - accuracy: 0.5030
 3744/25000 [===>..........................] - ETA: 1:21 - loss: 7.6216 - accuracy: 0.5029
 3776/25000 [===>..........................] - ETA: 1:21 - loss: 7.6098 - accuracy: 0.5037
 3808/25000 [===>..........................] - ETA: 1:21 - loss: 7.6102 - accuracy: 0.5037
 3840/25000 [===>..........................] - ETA: 1:21 - loss: 7.6147 - accuracy: 0.5034
 3872/25000 [===>..........................] - ETA: 1:21 - loss: 7.6151 - accuracy: 0.5034
 3904/25000 [===>..........................] - ETA: 1:21 - loss: 7.6352 - accuracy: 0.5020
 3936/25000 [===>..........................] - ETA: 1:21 - loss: 7.6393 - accuracy: 0.5018
 3968/25000 [===>..........................] - ETA: 1:20 - loss: 7.6434 - accuracy: 0.5015
 4000/25000 [===>..........................] - ETA: 1:20 - loss: 7.6436 - accuracy: 0.5015
 4032/25000 [===>..........................] - ETA: 1:20 - loss: 7.6590 - accuracy: 0.5005
 4064/25000 [===>..........................] - ETA: 1:20 - loss: 7.6628 - accuracy: 0.5002
 4096/25000 [===>..........................] - ETA: 1:20 - loss: 7.6442 - accuracy: 0.5015
 4128/25000 [===>..........................] - ETA: 1:20 - loss: 7.6295 - accuracy: 0.5024
 4160/25000 [===>..........................] - ETA: 1:20 - loss: 7.6334 - accuracy: 0.5022
 4192/25000 [====>.........................] - ETA: 1:19 - loss: 7.6227 - accuracy: 0.5029
 4224/25000 [====>.........................] - ETA: 1:19 - loss: 7.6231 - accuracy: 0.5028
 4256/25000 [====>.........................] - ETA: 1:19 - loss: 7.6378 - accuracy: 0.5019
 4288/25000 [====>.........................] - ETA: 1:19 - loss: 7.6309 - accuracy: 0.5023
 4320/25000 [====>.........................] - ETA: 1:19 - loss: 7.6276 - accuracy: 0.5025
 4352/25000 [====>.........................] - ETA: 1:19 - loss: 7.6349 - accuracy: 0.5021
 4384/25000 [====>.........................] - ETA: 1:19 - loss: 7.6246 - accuracy: 0.5027
 4416/25000 [====>.........................] - ETA: 1:18 - loss: 7.6215 - accuracy: 0.5029
 4448/25000 [====>.........................] - ETA: 1:18 - loss: 7.6011 - accuracy: 0.5043
 4480/25000 [====>.........................] - ETA: 1:18 - loss: 7.6153 - accuracy: 0.5033
 4512/25000 [====>.........................] - ETA: 1:18 - loss: 7.6292 - accuracy: 0.5024
 4544/25000 [====>.........................] - ETA: 1:18 - loss: 7.6295 - accuracy: 0.5024
 4576/25000 [====>.........................] - ETA: 1:18 - loss: 7.6231 - accuracy: 0.5028
 4608/25000 [====>.........................] - ETA: 1:18 - loss: 7.6200 - accuracy: 0.5030
 4640/25000 [====>.........................] - ETA: 1:17 - loss: 7.6137 - accuracy: 0.5034
 4672/25000 [====>.........................] - ETA: 1:17 - loss: 7.6174 - accuracy: 0.5032
 4704/25000 [====>.........................] - ETA: 1:17 - loss: 7.6177 - accuracy: 0.5032
 4736/25000 [====>.........................] - ETA: 1:17 - loss: 7.6440 - accuracy: 0.5015
 4768/25000 [====>.........................] - ETA: 1:17 - loss: 7.6505 - accuracy: 0.5010
 4800/25000 [====>.........................] - ETA: 1:17 - loss: 7.6634 - accuracy: 0.5002
 4832/25000 [====>.........................] - ETA: 1:17 - loss: 7.6666 - accuracy: 0.5000
 4864/25000 [====>.........................] - ETA: 1:17 - loss: 7.6572 - accuracy: 0.5006
 4896/25000 [====>.........................] - ETA: 1:16 - loss: 7.6635 - accuracy: 0.5002
 4928/25000 [====>.........................] - ETA: 1:16 - loss: 7.6511 - accuracy: 0.5010
 4960/25000 [====>.........................] - ETA: 1:16 - loss: 7.6697 - accuracy: 0.4998
 4992/25000 [====>.........................] - ETA: 1:16 - loss: 7.6758 - accuracy: 0.4994
 5024/25000 [=====>........................] - ETA: 1:16 - loss: 7.6727 - accuracy: 0.4996
 5056/25000 [=====>........................] - ETA: 1:16 - loss: 7.6697 - accuracy: 0.4998
 5088/25000 [=====>........................] - ETA: 1:16 - loss: 7.6757 - accuracy: 0.4994
 5120/25000 [=====>........................] - ETA: 1:16 - loss: 7.6876 - accuracy: 0.4986
 5152/25000 [=====>........................] - ETA: 1:15 - loss: 7.6904 - accuracy: 0.4984
 5184/25000 [=====>........................] - ETA: 1:15 - loss: 7.6844 - accuracy: 0.4988
 5216/25000 [=====>........................] - ETA: 1:15 - loss: 7.6813 - accuracy: 0.4990
 5248/25000 [=====>........................] - ETA: 1:15 - loss: 7.6637 - accuracy: 0.5002
 5280/25000 [=====>........................] - ETA: 1:15 - loss: 7.6695 - accuracy: 0.4998
 5312/25000 [=====>........................] - ETA: 1:15 - loss: 7.6839 - accuracy: 0.4989
 5344/25000 [=====>........................] - ETA: 1:15 - loss: 7.6867 - accuracy: 0.4987
 5376/25000 [=====>........................] - ETA: 1:14 - loss: 7.6780 - accuracy: 0.4993
 5408/25000 [=====>........................] - ETA: 1:14 - loss: 7.6751 - accuracy: 0.4994
 5440/25000 [=====>........................] - ETA: 1:14 - loss: 7.6694 - accuracy: 0.4998
 5472/25000 [=====>........................] - ETA: 1:14 - loss: 7.6638 - accuracy: 0.5002
 5504/25000 [=====>........................] - ETA: 1:14 - loss: 7.6638 - accuracy: 0.5002
 5536/25000 [=====>........................] - ETA: 1:14 - loss: 7.6528 - accuracy: 0.5009
 5568/25000 [=====>........................] - ETA: 1:14 - loss: 7.6529 - accuracy: 0.5009
 5600/25000 [=====>........................] - ETA: 1:14 - loss: 7.6502 - accuracy: 0.5011
 5632/25000 [=====>........................] - ETA: 1:13 - loss: 7.6503 - accuracy: 0.5011
 5664/25000 [=====>........................] - ETA: 1:13 - loss: 7.6504 - accuracy: 0.5011
 5696/25000 [=====>........................] - ETA: 1:13 - loss: 7.6505 - accuracy: 0.5011
 5728/25000 [=====>........................] - ETA: 1:13 - loss: 7.6532 - accuracy: 0.5009
 5760/25000 [=====>........................] - ETA: 1:13 - loss: 7.6613 - accuracy: 0.5003
 5792/25000 [=====>........................] - ETA: 1:13 - loss: 7.6719 - accuracy: 0.4997
 5824/25000 [=====>........................] - ETA: 1:13 - loss: 7.6824 - accuracy: 0.4990
 5856/25000 [======>.......................] - ETA: 1:13 - loss: 7.6876 - accuracy: 0.4986
 5888/25000 [======>.......................] - ETA: 1:13 - loss: 7.6901 - accuracy: 0.4985
 5920/25000 [======>.......................] - ETA: 1:12 - loss: 7.6873 - accuracy: 0.4986
 5952/25000 [======>.......................] - ETA: 1:12 - loss: 7.6950 - accuracy: 0.4982
 5984/25000 [======>.......................] - ETA: 1:12 - loss: 7.6922 - accuracy: 0.4983
 6016/25000 [======>.......................] - ETA: 1:12 - loss: 7.6947 - accuracy: 0.4982
 6048/25000 [======>.......................] - ETA: 1:12 - loss: 7.6945 - accuracy: 0.4982
 6080/25000 [======>.......................] - ETA: 1:12 - loss: 7.6944 - accuracy: 0.4982
 6112/25000 [======>.......................] - ETA: 1:12 - loss: 7.6892 - accuracy: 0.4985
 6144/25000 [======>.......................] - ETA: 1:11 - loss: 7.6791 - accuracy: 0.4992
 6176/25000 [======>.......................] - ETA: 1:11 - loss: 7.6790 - accuracy: 0.4992
 6208/25000 [======>.......................] - ETA: 1:11 - loss: 7.6666 - accuracy: 0.5000
 6240/25000 [======>.......................] - ETA: 1:11 - loss: 7.6789 - accuracy: 0.4992
 6272/25000 [======>.......................] - ETA: 1:11 - loss: 7.6813 - accuracy: 0.4990
 6304/25000 [======>.......................] - ETA: 1:11 - loss: 7.6763 - accuracy: 0.4994
 6336/25000 [======>.......................] - ETA: 1:11 - loss: 7.6666 - accuracy: 0.5000
 6368/25000 [======>.......................] - ETA: 1:11 - loss: 7.6690 - accuracy: 0.4998
 6400/25000 [======>.......................] - ETA: 1:10 - loss: 7.6666 - accuracy: 0.5000
 6432/25000 [======>.......................] - ETA: 1:10 - loss: 7.6785 - accuracy: 0.4992
 6464/25000 [======>.......................] - ETA: 1:10 - loss: 7.6714 - accuracy: 0.4997
 6496/25000 [======>.......................] - ETA: 1:10 - loss: 7.6595 - accuracy: 0.5005
 6528/25000 [======>.......................] - ETA: 1:10 - loss: 7.6666 - accuracy: 0.5000
 6560/25000 [======>.......................] - ETA: 1:10 - loss: 7.6783 - accuracy: 0.4992
 6592/25000 [======>.......................] - ETA: 1:10 - loss: 7.6759 - accuracy: 0.4994
 6624/25000 [======>.......................] - ETA: 1:10 - loss: 7.6712 - accuracy: 0.4997
 6656/25000 [======>.......................] - ETA: 1:09 - loss: 7.6689 - accuracy: 0.4998
 6688/25000 [=======>......................] - ETA: 1:09 - loss: 7.6689 - accuracy: 0.4999
 6720/25000 [=======>......................] - ETA: 1:09 - loss: 7.6712 - accuracy: 0.4997
 6752/25000 [=======>......................] - ETA: 1:09 - loss: 7.6689 - accuracy: 0.4999
 6784/25000 [=======>......................] - ETA: 1:09 - loss: 7.6711 - accuracy: 0.4997
 6816/25000 [=======>......................] - ETA: 1:09 - loss: 7.6621 - accuracy: 0.5003
 6848/25000 [=======>......................] - ETA: 1:09 - loss: 7.6577 - accuracy: 0.5006
 6880/25000 [=======>......................] - ETA: 1:08 - loss: 7.6555 - accuracy: 0.5007
 6912/25000 [=======>......................] - ETA: 1:08 - loss: 7.6622 - accuracy: 0.5003
 6944/25000 [=======>......................] - ETA: 1:08 - loss: 7.6556 - accuracy: 0.5007
 6976/25000 [=======>......................] - ETA: 1:08 - loss: 7.6534 - accuracy: 0.5009
 7008/25000 [=======>......................] - ETA: 1:08 - loss: 7.6666 - accuracy: 0.5000
 7040/25000 [=======>......................] - ETA: 1:08 - loss: 7.6666 - accuracy: 0.5000
 7072/25000 [=======>......................] - ETA: 1:08 - loss: 7.6623 - accuracy: 0.5003
 7104/25000 [=======>......................] - ETA: 1:08 - loss: 7.6515 - accuracy: 0.5010
 7136/25000 [=======>......................] - ETA: 1:08 - loss: 7.6559 - accuracy: 0.5007
 7168/25000 [=======>......................] - ETA: 1:07 - loss: 7.6516 - accuracy: 0.5010
 7200/25000 [=======>......................] - ETA: 1:07 - loss: 7.6602 - accuracy: 0.5004
 7232/25000 [=======>......................] - ETA: 1:07 - loss: 7.6497 - accuracy: 0.5011
 7264/25000 [=======>......................] - ETA: 1:07 - loss: 7.6413 - accuracy: 0.5017
 7296/25000 [=======>......................] - ETA: 1:07 - loss: 7.6477 - accuracy: 0.5012
 7328/25000 [=======>......................] - ETA: 1:07 - loss: 7.6436 - accuracy: 0.5015
 7360/25000 [=======>......................] - ETA: 1:07 - loss: 7.6479 - accuracy: 0.5012
 7392/25000 [=======>......................] - ETA: 1:06 - loss: 7.6521 - accuracy: 0.5009
 7424/25000 [=======>......................] - ETA: 1:06 - loss: 7.6646 - accuracy: 0.5001
 7456/25000 [=======>......................] - ETA: 1:06 - loss: 7.6646 - accuracy: 0.5001
 7488/25000 [=======>......................] - ETA: 1:06 - loss: 7.6687 - accuracy: 0.4999
 7520/25000 [========>.....................] - ETA: 1:06 - loss: 7.6748 - accuracy: 0.4995
 7552/25000 [========>.....................] - ETA: 1:06 - loss: 7.6727 - accuracy: 0.4996
 7584/25000 [========>.....................] - ETA: 1:06 - loss: 7.6707 - accuracy: 0.4997
 7616/25000 [========>.....................] - ETA: 1:06 - loss: 7.6586 - accuracy: 0.5005
 7648/25000 [========>.....................] - ETA: 1:05 - loss: 7.6606 - accuracy: 0.5004
 7680/25000 [========>.....................] - ETA: 1:05 - loss: 7.6666 - accuracy: 0.5000
 7712/25000 [========>.....................] - ETA: 1:05 - loss: 7.6547 - accuracy: 0.5008
 7744/25000 [========>.....................] - ETA: 1:05 - loss: 7.6607 - accuracy: 0.5004
 7776/25000 [========>.....................] - ETA: 1:05 - loss: 7.6686 - accuracy: 0.4999
 7808/25000 [========>.....................] - ETA: 1:05 - loss: 7.6548 - accuracy: 0.5008
 7840/25000 [========>.....................] - ETA: 1:05 - loss: 7.6608 - accuracy: 0.5004
 7872/25000 [========>.....................] - ETA: 1:05 - loss: 7.6549 - accuracy: 0.5008
 7904/25000 [========>.....................] - ETA: 1:04 - loss: 7.6589 - accuracy: 0.5005
 7936/25000 [========>.....................] - ETA: 1:04 - loss: 7.6570 - accuracy: 0.5006
 7968/25000 [========>.....................] - ETA: 1:04 - loss: 7.6589 - accuracy: 0.5005
 8000/25000 [========>.....................] - ETA: 1:04 - loss: 7.6628 - accuracy: 0.5002
 8032/25000 [========>.....................] - ETA: 1:04 - loss: 7.6647 - accuracy: 0.5001
 8064/25000 [========>.....................] - ETA: 1:04 - loss: 7.6514 - accuracy: 0.5010
 8096/25000 [========>.....................] - ETA: 1:04 - loss: 7.6458 - accuracy: 0.5014
 8128/25000 [========>.....................] - ETA: 1:04 - loss: 7.6440 - accuracy: 0.5015
 8160/25000 [========>.....................] - ETA: 1:03 - loss: 7.6478 - accuracy: 0.5012
 8192/25000 [========>.....................] - ETA: 1:03 - loss: 7.6404 - accuracy: 0.5017
 8224/25000 [========>.....................] - ETA: 1:03 - loss: 7.6368 - accuracy: 0.5019
 8256/25000 [========>.....................] - ETA: 1:03 - loss: 7.6388 - accuracy: 0.5018
 8288/25000 [========>.....................] - ETA: 1:03 - loss: 7.6389 - accuracy: 0.5018
 8320/25000 [========>.....................] - ETA: 1:03 - loss: 7.6500 - accuracy: 0.5011
 8352/25000 [=========>....................] - ETA: 1:03 - loss: 7.6464 - accuracy: 0.5013
 8384/25000 [=========>....................] - ETA: 1:03 - loss: 7.6410 - accuracy: 0.5017
 8416/25000 [=========>....................] - ETA: 1:02 - loss: 7.6302 - accuracy: 0.5024
 8448/25000 [=========>....................] - ETA: 1:02 - loss: 7.6303 - accuracy: 0.5024
 8480/25000 [=========>....................] - ETA: 1:02 - loss: 7.6232 - accuracy: 0.5028
 8512/25000 [=========>....................] - ETA: 1:02 - loss: 7.6324 - accuracy: 0.5022
 8544/25000 [=========>....................] - ETA: 1:02 - loss: 7.6415 - accuracy: 0.5016
 8576/25000 [=========>....................] - ETA: 1:02 - loss: 7.6255 - accuracy: 0.5027
 8608/25000 [=========>....................] - ETA: 1:02 - loss: 7.6221 - accuracy: 0.5029
 8640/25000 [=========>....................] - ETA: 1:02 - loss: 7.6240 - accuracy: 0.5028
 8672/25000 [=========>....................] - ETA: 1:02 - loss: 7.6242 - accuracy: 0.5028
 8704/25000 [=========>....................] - ETA: 1:01 - loss: 7.6191 - accuracy: 0.5031
 8736/25000 [=========>....................] - ETA: 1:01 - loss: 7.6122 - accuracy: 0.5035
 8768/25000 [=========>....................] - ETA: 1:01 - loss: 7.6054 - accuracy: 0.5040
 8800/25000 [=========>....................] - ETA: 1:01 - loss: 7.6143 - accuracy: 0.5034
 8832/25000 [=========>....................] - ETA: 1:01 - loss: 7.6093 - accuracy: 0.5037
 8864/25000 [=========>....................] - ETA: 1:01 - loss: 7.6147 - accuracy: 0.5034
 8896/25000 [=========>....................] - ETA: 1:01 - loss: 7.6149 - accuracy: 0.5034
 8928/25000 [=========>....................] - ETA: 1:01 - loss: 7.6185 - accuracy: 0.5031
 8960/25000 [=========>....................] - ETA: 1:00 - loss: 7.6204 - accuracy: 0.5030
 8992/25000 [=========>....................] - ETA: 1:00 - loss: 7.6138 - accuracy: 0.5034
 9024/25000 [=========>....................] - ETA: 1:00 - loss: 7.6139 - accuracy: 0.5034
 9056/25000 [=========>....................] - ETA: 1:00 - loss: 7.6124 - accuracy: 0.5035
 9088/25000 [=========>....................] - ETA: 1:00 - loss: 7.6194 - accuracy: 0.5031
 9120/25000 [=========>....................] - ETA: 1:00 - loss: 7.6111 - accuracy: 0.5036
 9152/25000 [=========>....................] - ETA: 1:00 - loss: 7.6046 - accuracy: 0.5040
 9184/25000 [==========>...................] - ETA: 59s - loss: 7.6082 - accuracy: 0.5038 
 9216/25000 [==========>...................] - ETA: 59s - loss: 7.6001 - accuracy: 0.5043
 9248/25000 [==========>...................] - ETA: 59s - loss: 7.5953 - accuracy: 0.5046
 9280/25000 [==========>...................] - ETA: 59s - loss: 7.5972 - accuracy: 0.5045
 9312/25000 [==========>...................] - ETA: 59s - loss: 7.6008 - accuracy: 0.5043
 9344/25000 [==========>...................] - ETA: 59s - loss: 7.5977 - accuracy: 0.5045
 9376/25000 [==========>...................] - ETA: 59s - loss: 7.6028 - accuracy: 0.5042
 9408/25000 [==========>...................] - ETA: 59s - loss: 7.5982 - accuracy: 0.5045
 9440/25000 [==========>...................] - ETA: 58s - loss: 7.6000 - accuracy: 0.5043
 9472/25000 [==========>...................] - ETA: 58s - loss: 7.5970 - accuracy: 0.5045
 9504/25000 [==========>...................] - ETA: 58s - loss: 7.5956 - accuracy: 0.5046
 9536/25000 [==========>...................] - ETA: 58s - loss: 7.6007 - accuracy: 0.5043
 9568/25000 [==========>...................] - ETA: 58s - loss: 7.6009 - accuracy: 0.5043
 9600/25000 [==========>...................] - ETA: 58s - loss: 7.5963 - accuracy: 0.5046
 9632/25000 [==========>...................] - ETA: 58s - loss: 7.5982 - accuracy: 0.5045
 9664/25000 [==========>...................] - ETA: 58s - loss: 7.6016 - accuracy: 0.5042
 9696/25000 [==========>...................] - ETA: 57s - loss: 7.6002 - accuracy: 0.5043
 9728/25000 [==========>...................] - ETA: 57s - loss: 7.6036 - accuracy: 0.5041
 9760/25000 [==========>...................] - ETA: 57s - loss: 7.6101 - accuracy: 0.5037
 9792/25000 [==========>...................] - ETA: 57s - loss: 7.6087 - accuracy: 0.5038
 9824/25000 [==========>...................] - ETA: 57s - loss: 7.6136 - accuracy: 0.5035
 9856/25000 [==========>...................] - ETA: 57s - loss: 7.6091 - accuracy: 0.5038
 9888/25000 [==========>...................] - ETA: 57s - loss: 7.6077 - accuracy: 0.5038
 9920/25000 [==========>...................] - ETA: 57s - loss: 7.6032 - accuracy: 0.5041
 9952/25000 [==========>...................] - ETA: 56s - loss: 7.6050 - accuracy: 0.5040
 9984/25000 [==========>...................] - ETA: 56s - loss: 7.6052 - accuracy: 0.5040
10016/25000 [===========>..................] - ETA: 56s - loss: 7.6039 - accuracy: 0.5041
10048/25000 [===========>..................] - ETA: 56s - loss: 7.6056 - accuracy: 0.5040
10080/25000 [===========>..................] - ETA: 56s - loss: 7.6103 - accuracy: 0.5037
10112/25000 [===========>..................] - ETA: 56s - loss: 7.6181 - accuracy: 0.5032
10144/25000 [===========>..................] - ETA: 56s - loss: 7.6152 - accuracy: 0.5034
10176/25000 [===========>..................] - ETA: 56s - loss: 7.6124 - accuracy: 0.5035
10208/25000 [===========>..................] - ETA: 56s - loss: 7.6155 - accuracy: 0.5033
10240/25000 [===========>..................] - ETA: 55s - loss: 7.6067 - accuracy: 0.5039
10272/25000 [===========>..................] - ETA: 55s - loss: 7.6099 - accuracy: 0.5037
10304/25000 [===========>..................] - ETA: 55s - loss: 7.6041 - accuracy: 0.5041
10336/25000 [===========>..................] - ETA: 55s - loss: 7.6088 - accuracy: 0.5038
10368/25000 [===========>..................] - ETA: 55s - loss: 7.6030 - accuracy: 0.5041
10400/25000 [===========>..................] - ETA: 55s - loss: 7.6047 - accuracy: 0.5040
10432/25000 [===========>..................] - ETA: 55s - loss: 7.6049 - accuracy: 0.5040
10464/25000 [===========>..................] - ETA: 55s - loss: 7.6065 - accuracy: 0.5039
10496/25000 [===========>..................] - ETA: 54s - loss: 7.6053 - accuracy: 0.5040
10528/25000 [===========>..................] - ETA: 54s - loss: 7.6054 - accuracy: 0.5040
10560/25000 [===========>..................] - ETA: 54s - loss: 7.6129 - accuracy: 0.5035
10592/25000 [===========>..................] - ETA: 54s - loss: 7.6145 - accuracy: 0.5034
10624/25000 [===========>..................] - ETA: 54s - loss: 7.6204 - accuracy: 0.5030
10656/25000 [===========>..................] - ETA: 54s - loss: 7.6249 - accuracy: 0.5027
10688/25000 [===========>..................] - ETA: 54s - loss: 7.6308 - accuracy: 0.5023
10720/25000 [===========>..................] - ETA: 54s - loss: 7.6280 - accuracy: 0.5025
10752/25000 [===========>..................] - ETA: 53s - loss: 7.6224 - accuracy: 0.5029
10784/25000 [===========>..................] - ETA: 53s - loss: 7.6183 - accuracy: 0.5032
10816/25000 [===========>..................] - ETA: 53s - loss: 7.6142 - accuracy: 0.5034
10848/25000 [============>.................] - ETA: 53s - loss: 7.6186 - accuracy: 0.5031
10880/25000 [============>.................] - ETA: 53s - loss: 7.6173 - accuracy: 0.5032
10912/25000 [============>.................] - ETA: 53s - loss: 7.6062 - accuracy: 0.5039
10944/25000 [============>.................] - ETA: 53s - loss: 7.6050 - accuracy: 0.5040
10976/25000 [============>.................] - ETA: 53s - loss: 7.6107 - accuracy: 0.5036
11008/25000 [============>.................] - ETA: 53s - loss: 7.6151 - accuracy: 0.5034
11040/25000 [============>.................] - ETA: 52s - loss: 7.6194 - accuracy: 0.5031
11072/25000 [============>.................] - ETA: 52s - loss: 7.6195 - accuracy: 0.5031
11104/25000 [============>.................] - ETA: 52s - loss: 7.6238 - accuracy: 0.5028
11136/25000 [============>.................] - ETA: 52s - loss: 7.6322 - accuracy: 0.5022
11168/25000 [============>.................] - ETA: 52s - loss: 7.6378 - accuracy: 0.5019
11200/25000 [============>.................] - ETA: 52s - loss: 7.6310 - accuracy: 0.5023
11232/25000 [============>.................] - ETA: 52s - loss: 7.6339 - accuracy: 0.5021
11264/25000 [============>.................] - ETA: 52s - loss: 7.6408 - accuracy: 0.5017
11296/25000 [============>.................] - ETA: 51s - loss: 7.6435 - accuracy: 0.5015
11328/25000 [============>.................] - ETA: 51s - loss: 7.6477 - accuracy: 0.5012
11360/25000 [============>.................] - ETA: 51s - loss: 7.6464 - accuracy: 0.5013
11392/25000 [============>.................] - ETA: 51s - loss: 7.6478 - accuracy: 0.5012
11424/25000 [============>.................] - ETA: 51s - loss: 7.6492 - accuracy: 0.5011
11456/25000 [============>.................] - ETA: 51s - loss: 7.6479 - accuracy: 0.5012
11488/25000 [============>.................] - ETA: 51s - loss: 7.6453 - accuracy: 0.5014
11520/25000 [============>.................] - ETA: 51s - loss: 7.6467 - accuracy: 0.5013
11552/25000 [============>.................] - ETA: 50s - loss: 7.6401 - accuracy: 0.5017
11584/25000 [============>.................] - ETA: 50s - loss: 7.6349 - accuracy: 0.5021
11616/25000 [============>.................] - ETA: 50s - loss: 7.6323 - accuracy: 0.5022
11648/25000 [============>.................] - ETA: 50s - loss: 7.6298 - accuracy: 0.5024
11680/25000 [=============>................] - ETA: 50s - loss: 7.6285 - accuracy: 0.5025
11712/25000 [=============>................] - ETA: 50s - loss: 7.6287 - accuracy: 0.5025
11744/25000 [=============>................] - ETA: 50s - loss: 7.6261 - accuracy: 0.5026
11776/25000 [=============>................] - ETA: 50s - loss: 7.6289 - accuracy: 0.5025
11808/25000 [=============>................] - ETA: 49s - loss: 7.6264 - accuracy: 0.5026
11840/25000 [=============>................] - ETA: 49s - loss: 7.6239 - accuracy: 0.5028
11872/25000 [=============>................] - ETA: 49s - loss: 7.6343 - accuracy: 0.5021
11904/25000 [=============>................] - ETA: 49s - loss: 7.6383 - accuracy: 0.5018
11936/25000 [=============>................] - ETA: 49s - loss: 7.6371 - accuracy: 0.5019
11968/25000 [=============>................] - ETA: 49s - loss: 7.6397 - accuracy: 0.5018
12000/25000 [=============>................] - ETA: 49s - loss: 7.6423 - accuracy: 0.5016
12032/25000 [=============>................] - ETA: 49s - loss: 7.6399 - accuracy: 0.5017
12064/25000 [=============>................] - ETA: 48s - loss: 7.6399 - accuracy: 0.5017
12096/25000 [=============>................] - ETA: 48s - loss: 7.6476 - accuracy: 0.5012
12128/25000 [=============>................] - ETA: 48s - loss: 7.6477 - accuracy: 0.5012
12160/25000 [=============>................] - ETA: 48s - loss: 7.6490 - accuracy: 0.5012
12192/25000 [=============>................] - ETA: 48s - loss: 7.6440 - accuracy: 0.5015
12224/25000 [=============>................] - ETA: 48s - loss: 7.6453 - accuracy: 0.5014
12256/25000 [=============>................] - ETA: 48s - loss: 7.6403 - accuracy: 0.5017
12288/25000 [=============>................] - ETA: 48s - loss: 7.6429 - accuracy: 0.5015
12320/25000 [=============>................] - ETA: 48s - loss: 7.6467 - accuracy: 0.5013
12352/25000 [=============>................] - ETA: 47s - loss: 7.6505 - accuracy: 0.5011
12384/25000 [=============>................] - ETA: 47s - loss: 7.6542 - accuracy: 0.5008
12416/25000 [=============>................] - ETA: 47s - loss: 7.6469 - accuracy: 0.5013
12448/25000 [=============>................] - ETA: 47s - loss: 7.6494 - accuracy: 0.5011
12480/25000 [=============>................] - ETA: 47s - loss: 7.6543 - accuracy: 0.5008
12512/25000 [==============>...............] - ETA: 47s - loss: 7.6629 - accuracy: 0.5002
12544/25000 [==============>...............] - ETA: 47s - loss: 7.6581 - accuracy: 0.5006
12576/25000 [==============>...............] - ETA: 47s - loss: 7.6581 - accuracy: 0.5006
12608/25000 [==============>...............] - ETA: 46s - loss: 7.6569 - accuracy: 0.5006
12640/25000 [==============>...............] - ETA: 46s - loss: 7.6569 - accuracy: 0.5006
12672/25000 [==============>...............] - ETA: 46s - loss: 7.6606 - accuracy: 0.5004
12704/25000 [==============>...............] - ETA: 46s - loss: 7.6582 - accuracy: 0.5006
12736/25000 [==============>...............] - ETA: 46s - loss: 7.6606 - accuracy: 0.5004
12768/25000 [==============>...............] - ETA: 46s - loss: 7.6654 - accuracy: 0.5001
12800/25000 [==============>...............] - ETA: 46s - loss: 7.6570 - accuracy: 0.5006
12832/25000 [==============>...............] - ETA: 46s - loss: 7.6571 - accuracy: 0.5006
12864/25000 [==============>...............] - ETA: 45s - loss: 7.6571 - accuracy: 0.5006
12896/25000 [==============>...............] - ETA: 45s - loss: 7.6619 - accuracy: 0.5003
12928/25000 [==============>...............] - ETA: 45s - loss: 7.6654 - accuracy: 0.5001
12960/25000 [==============>...............] - ETA: 45s - loss: 7.6678 - accuracy: 0.4999
12992/25000 [==============>...............] - ETA: 45s - loss: 7.6643 - accuracy: 0.5002
13024/25000 [==============>...............] - ETA: 45s - loss: 7.6654 - accuracy: 0.5001
13056/25000 [==============>...............] - ETA: 45s - loss: 7.6631 - accuracy: 0.5002
13088/25000 [==============>...............] - ETA: 45s - loss: 7.6584 - accuracy: 0.5005
13120/25000 [==============>...............] - ETA: 45s - loss: 7.6619 - accuracy: 0.5003
13152/25000 [==============>...............] - ETA: 44s - loss: 7.6561 - accuracy: 0.5007
13184/25000 [==============>...............] - ETA: 44s - loss: 7.6550 - accuracy: 0.5008
13216/25000 [==============>...............] - ETA: 44s - loss: 7.6562 - accuracy: 0.5007
13248/25000 [==============>...............] - ETA: 44s - loss: 7.6574 - accuracy: 0.5006
13280/25000 [==============>...............] - ETA: 44s - loss: 7.6539 - accuracy: 0.5008
13312/25000 [==============>...............] - ETA: 44s - loss: 7.6574 - accuracy: 0.5006
13344/25000 [===============>..............] - ETA: 44s - loss: 7.6574 - accuracy: 0.5006
13376/25000 [===============>..............] - ETA: 44s - loss: 7.6552 - accuracy: 0.5007
13408/25000 [===============>..............] - ETA: 43s - loss: 7.6540 - accuracy: 0.5008
13440/25000 [===============>..............] - ETA: 43s - loss: 7.6564 - accuracy: 0.5007
13472/25000 [===============>..............] - ETA: 43s - loss: 7.6564 - accuracy: 0.5007
13504/25000 [===============>..............] - ETA: 43s - loss: 7.6598 - accuracy: 0.5004
13536/25000 [===============>..............] - ETA: 43s - loss: 7.6621 - accuracy: 0.5003
13568/25000 [===============>..............] - ETA: 43s - loss: 7.6610 - accuracy: 0.5004
13600/25000 [===============>..............] - ETA: 43s - loss: 7.6576 - accuracy: 0.5006
13632/25000 [===============>..............] - ETA: 43s - loss: 7.6632 - accuracy: 0.5002
13664/25000 [===============>..............] - ETA: 42s - loss: 7.6633 - accuracy: 0.5002
13696/25000 [===============>..............] - ETA: 42s - loss: 7.6655 - accuracy: 0.5001
13728/25000 [===============>..............] - ETA: 42s - loss: 7.6566 - accuracy: 0.5007
13760/25000 [===============>..............] - ETA: 42s - loss: 7.6566 - accuracy: 0.5007
13792/25000 [===============>..............] - ETA: 42s - loss: 7.6511 - accuracy: 0.5010
13824/25000 [===============>..............] - ETA: 42s - loss: 7.6478 - accuracy: 0.5012
13856/25000 [===============>..............] - ETA: 42s - loss: 7.6456 - accuracy: 0.5014
13888/25000 [===============>..............] - ETA: 42s - loss: 7.6467 - accuracy: 0.5013
13920/25000 [===============>..............] - ETA: 42s - loss: 7.6402 - accuracy: 0.5017
13952/25000 [===============>..............] - ETA: 41s - loss: 7.6369 - accuracy: 0.5019
13984/25000 [===============>..............] - ETA: 41s - loss: 7.6271 - accuracy: 0.5026
14016/25000 [===============>..............] - ETA: 41s - loss: 7.6305 - accuracy: 0.5024
14048/25000 [===============>..............] - ETA: 41s - loss: 7.6317 - accuracy: 0.5023
14080/25000 [===============>..............] - ETA: 41s - loss: 7.6307 - accuracy: 0.5023
14112/25000 [===============>..............] - ETA: 41s - loss: 7.6308 - accuracy: 0.5023
14144/25000 [===============>..............] - ETA: 41s - loss: 7.6276 - accuracy: 0.5025
14176/25000 [================>.............] - ETA: 41s - loss: 7.6277 - accuracy: 0.5025
14208/25000 [================>.............] - ETA: 40s - loss: 7.6235 - accuracy: 0.5028
14240/25000 [================>.............] - ETA: 40s - loss: 7.6246 - accuracy: 0.5027
14272/25000 [================>.............] - ETA: 40s - loss: 7.6215 - accuracy: 0.5029
14304/25000 [================>.............] - ETA: 40s - loss: 7.6216 - accuracy: 0.5029
14336/25000 [================>.............] - ETA: 40s - loss: 7.6174 - accuracy: 0.5032
14368/25000 [================>.............] - ETA: 40s - loss: 7.6175 - accuracy: 0.5032
14400/25000 [================>.............] - ETA: 40s - loss: 7.6187 - accuracy: 0.5031
14432/25000 [================>.............] - ETA: 40s - loss: 7.6167 - accuracy: 0.5033
14464/25000 [================>.............] - ETA: 40s - loss: 7.6126 - accuracy: 0.5035
14496/25000 [================>.............] - ETA: 39s - loss: 7.6042 - accuracy: 0.5041
14528/25000 [================>.............] - ETA: 39s - loss: 7.6086 - accuracy: 0.5038
14560/25000 [================>.............] - ETA: 39s - loss: 7.6140 - accuracy: 0.5034
14592/25000 [================>.............] - ETA: 39s - loss: 7.6141 - accuracy: 0.5034
14624/25000 [================>.............] - ETA: 39s - loss: 7.6152 - accuracy: 0.5034
14656/25000 [================>.............] - ETA: 39s - loss: 7.6154 - accuracy: 0.5033
14688/25000 [================>.............] - ETA: 39s - loss: 7.6123 - accuracy: 0.5035
14720/25000 [================>.............] - ETA: 39s - loss: 7.6041 - accuracy: 0.5041
14752/25000 [================>.............] - ETA: 38s - loss: 7.6063 - accuracy: 0.5039
14784/25000 [================>.............] - ETA: 38s - loss: 7.6044 - accuracy: 0.5041
14816/25000 [================>.............] - ETA: 38s - loss: 7.6076 - accuracy: 0.5038
14848/25000 [================>.............] - ETA: 38s - loss: 7.6016 - accuracy: 0.5042
14880/25000 [================>.............] - ETA: 38s - loss: 7.6058 - accuracy: 0.5040
14912/25000 [================>.............] - ETA: 38s - loss: 7.6060 - accuracy: 0.5040
14944/25000 [================>.............] - ETA: 38s - loss: 7.6092 - accuracy: 0.5037
14976/25000 [================>.............] - ETA: 38s - loss: 7.6103 - accuracy: 0.5037
15008/25000 [=================>............] - ETA: 37s - loss: 7.6094 - accuracy: 0.5037
15040/25000 [=================>............] - ETA: 37s - loss: 7.6095 - accuracy: 0.5037
15072/25000 [=================>............] - ETA: 37s - loss: 7.6096 - accuracy: 0.5037
15104/25000 [=================>............] - ETA: 37s - loss: 7.6037 - accuracy: 0.5041
15136/25000 [=================>............] - ETA: 37s - loss: 7.5998 - accuracy: 0.5044
15168/25000 [=================>............] - ETA: 37s - loss: 7.6029 - accuracy: 0.5042
15200/25000 [=================>............] - ETA: 37s - loss: 7.6010 - accuracy: 0.5043
15232/25000 [=================>............] - ETA: 37s - loss: 7.5992 - accuracy: 0.5044
15264/25000 [=================>............] - ETA: 36s - loss: 7.5973 - accuracy: 0.5045
15296/25000 [=================>............] - ETA: 36s - loss: 7.5985 - accuracy: 0.5044
15328/25000 [=================>............] - ETA: 36s - loss: 7.5996 - accuracy: 0.5044
15360/25000 [=================>............] - ETA: 36s - loss: 7.5987 - accuracy: 0.5044
15392/25000 [=================>............] - ETA: 36s - loss: 7.5949 - accuracy: 0.5047
15424/25000 [=================>............] - ETA: 36s - loss: 7.5931 - accuracy: 0.5048
15456/25000 [=================>............] - ETA: 36s - loss: 7.5922 - accuracy: 0.5049
15488/25000 [=================>............] - ETA: 36s - loss: 7.5904 - accuracy: 0.5050
15520/25000 [=================>............] - ETA: 35s - loss: 7.5984 - accuracy: 0.5044
15552/25000 [=================>............] - ETA: 35s - loss: 7.5996 - accuracy: 0.5044
15584/25000 [=================>............] - ETA: 35s - loss: 7.6007 - accuracy: 0.5043
15616/25000 [=================>............] - ETA: 35s - loss: 7.6048 - accuracy: 0.5040
15648/25000 [=================>............] - ETA: 35s - loss: 7.6088 - accuracy: 0.5038
15680/25000 [=================>............] - ETA: 35s - loss: 7.6138 - accuracy: 0.5034
15712/25000 [=================>............] - ETA: 35s - loss: 7.6120 - accuracy: 0.5036
15744/25000 [=================>............] - ETA: 35s - loss: 7.6160 - accuracy: 0.5033
15776/25000 [=================>............] - ETA: 34s - loss: 7.6180 - accuracy: 0.5032
15808/25000 [=================>............] - ETA: 34s - loss: 7.6230 - accuracy: 0.5028
15840/25000 [==================>...........] - ETA: 34s - loss: 7.6221 - accuracy: 0.5029
15872/25000 [==================>...........] - ETA: 34s - loss: 7.6202 - accuracy: 0.5030
15904/25000 [==================>...........] - ETA: 34s - loss: 7.6203 - accuracy: 0.5030
15936/25000 [==================>...........] - ETA: 34s - loss: 7.6224 - accuracy: 0.5029
15968/25000 [==================>...........] - ETA: 34s - loss: 7.6215 - accuracy: 0.5029
16000/25000 [==================>...........] - ETA: 34s - loss: 7.6245 - accuracy: 0.5027
16032/25000 [==================>...........] - ETA: 34s - loss: 7.6341 - accuracy: 0.5021
16064/25000 [==================>...........] - ETA: 33s - loss: 7.6351 - accuracy: 0.5021
16096/25000 [==================>...........] - ETA: 33s - loss: 7.6361 - accuracy: 0.5020
16128/25000 [==================>...........] - ETA: 33s - loss: 7.6352 - accuracy: 0.5020
16160/25000 [==================>...........] - ETA: 33s - loss: 7.6363 - accuracy: 0.5020
16192/25000 [==================>...........] - ETA: 33s - loss: 7.6392 - accuracy: 0.5018
16224/25000 [==================>...........] - ETA: 33s - loss: 7.6439 - accuracy: 0.5015
16256/25000 [==================>...........] - ETA: 33s - loss: 7.6449 - accuracy: 0.5014
16288/25000 [==================>...........] - ETA: 33s - loss: 7.6450 - accuracy: 0.5014
16320/25000 [==================>...........] - ETA: 32s - loss: 7.6441 - accuracy: 0.5015
16352/25000 [==================>...........] - ETA: 32s - loss: 7.6432 - accuracy: 0.5015
16384/25000 [==================>...........] - ETA: 32s - loss: 7.6451 - accuracy: 0.5014
16416/25000 [==================>...........] - ETA: 32s - loss: 7.6489 - accuracy: 0.5012
16448/25000 [==================>...........] - ETA: 32s - loss: 7.6461 - accuracy: 0.5013
16480/25000 [==================>...........] - ETA: 32s - loss: 7.6462 - accuracy: 0.5013
16512/25000 [==================>...........] - ETA: 32s - loss: 7.6434 - accuracy: 0.5015
16544/25000 [==================>...........] - ETA: 32s - loss: 7.6425 - accuracy: 0.5016
16576/25000 [==================>...........] - ETA: 31s - loss: 7.6426 - accuracy: 0.5016
16608/25000 [==================>...........] - ETA: 31s - loss: 7.6445 - accuracy: 0.5014
16640/25000 [==================>...........] - ETA: 31s - loss: 7.6408 - accuracy: 0.5017
16672/25000 [===================>..........] - ETA: 31s - loss: 7.6399 - accuracy: 0.5017
16704/25000 [===================>..........] - ETA: 31s - loss: 7.6418 - accuracy: 0.5016
16736/25000 [===================>..........] - ETA: 31s - loss: 7.6419 - accuracy: 0.5016
16768/25000 [===================>..........] - ETA: 31s - loss: 7.6419 - accuracy: 0.5016
16800/25000 [===================>..........] - ETA: 31s - loss: 7.6438 - accuracy: 0.5015
16832/25000 [===================>..........] - ETA: 30s - loss: 7.6457 - accuracy: 0.5014
16864/25000 [===================>..........] - ETA: 30s - loss: 7.6457 - accuracy: 0.5014
16896/25000 [===================>..........] - ETA: 30s - loss: 7.6421 - accuracy: 0.5016
16928/25000 [===================>..........] - ETA: 30s - loss: 7.6413 - accuracy: 0.5017
16960/25000 [===================>..........] - ETA: 30s - loss: 7.6377 - accuracy: 0.5019
16992/25000 [===================>..........] - ETA: 30s - loss: 7.6414 - accuracy: 0.5016
17024/25000 [===================>..........] - ETA: 30s - loss: 7.6441 - accuracy: 0.5015
17056/25000 [===================>..........] - ETA: 30s - loss: 7.6414 - accuracy: 0.5016
17088/25000 [===================>..........] - ETA: 30s - loss: 7.6397 - accuracy: 0.5018
17120/25000 [===================>..........] - ETA: 29s - loss: 7.6371 - accuracy: 0.5019
17152/25000 [===================>..........] - ETA: 29s - loss: 7.6371 - accuracy: 0.5019
17184/25000 [===================>..........] - ETA: 29s - loss: 7.6381 - accuracy: 0.5019
17216/25000 [===================>..........] - ETA: 29s - loss: 7.6363 - accuracy: 0.5020
17248/25000 [===================>..........] - ETA: 29s - loss: 7.6391 - accuracy: 0.5018
17280/25000 [===================>..........] - ETA: 29s - loss: 7.6400 - accuracy: 0.5017
17312/25000 [===================>..........] - ETA: 29s - loss: 7.6418 - accuracy: 0.5016
17344/25000 [===================>..........] - ETA: 29s - loss: 7.6436 - accuracy: 0.5015
17376/25000 [===================>..........] - ETA: 28s - loss: 7.6410 - accuracy: 0.5017
17408/25000 [===================>..........] - ETA: 28s - loss: 7.6376 - accuracy: 0.5019
17440/25000 [===================>..........] - ETA: 28s - loss: 7.6376 - accuracy: 0.5019
17472/25000 [===================>..........] - ETA: 28s - loss: 7.6350 - accuracy: 0.5021
17504/25000 [====================>.........] - ETA: 28s - loss: 7.6333 - accuracy: 0.5022
17536/25000 [====================>.........] - ETA: 28s - loss: 7.6334 - accuracy: 0.5022
17568/25000 [====================>.........] - ETA: 28s - loss: 7.6335 - accuracy: 0.5022
17600/25000 [====================>.........] - ETA: 28s - loss: 7.6283 - accuracy: 0.5025
17632/25000 [====================>.........] - ETA: 27s - loss: 7.6249 - accuracy: 0.5027
17664/25000 [====================>.........] - ETA: 27s - loss: 7.6310 - accuracy: 0.5023
17696/25000 [====================>.........] - ETA: 27s - loss: 7.6268 - accuracy: 0.5026
17728/25000 [====================>.........] - ETA: 27s - loss: 7.6260 - accuracy: 0.5027
17760/25000 [====================>.........] - ETA: 27s - loss: 7.6278 - accuracy: 0.5025
17792/25000 [====================>.........] - ETA: 27s - loss: 7.6253 - accuracy: 0.5027
17824/25000 [====================>.........] - ETA: 27s - loss: 7.6262 - accuracy: 0.5026
17856/25000 [====================>.........] - ETA: 27s - loss: 7.6228 - accuracy: 0.5029
17888/25000 [====================>.........] - ETA: 26s - loss: 7.6212 - accuracy: 0.5030
17920/25000 [====================>.........] - ETA: 26s - loss: 7.6230 - accuracy: 0.5028
17952/25000 [====================>.........] - ETA: 26s - loss: 7.6214 - accuracy: 0.5030
17984/25000 [====================>.........] - ETA: 26s - loss: 7.6197 - accuracy: 0.5031
18016/25000 [====================>.........] - ETA: 26s - loss: 7.6181 - accuracy: 0.5032
18048/25000 [====================>.........] - ETA: 26s - loss: 7.6139 - accuracy: 0.5034
18080/25000 [====================>.........] - ETA: 26s - loss: 7.6123 - accuracy: 0.5035
18112/25000 [====================>.........] - ETA: 26s - loss: 7.6133 - accuracy: 0.5035
18144/25000 [====================>.........] - ETA: 26s - loss: 7.6159 - accuracy: 0.5033
18176/25000 [====================>.........] - ETA: 25s - loss: 7.6135 - accuracy: 0.5035
18208/25000 [====================>.........] - ETA: 25s - loss: 7.6127 - accuracy: 0.5035
18240/25000 [====================>.........] - ETA: 25s - loss: 7.6128 - accuracy: 0.5035
18272/25000 [====================>.........] - ETA: 25s - loss: 7.6154 - accuracy: 0.5033
18304/25000 [====================>.........] - ETA: 25s - loss: 7.6138 - accuracy: 0.5034
18336/25000 [=====================>........] - ETA: 25s - loss: 7.6123 - accuracy: 0.5035
18368/25000 [=====================>........] - ETA: 25s - loss: 7.6124 - accuracy: 0.5035
18400/25000 [=====================>........] - ETA: 25s - loss: 7.6133 - accuracy: 0.5035
18432/25000 [=====================>........] - ETA: 24s - loss: 7.6150 - accuracy: 0.5034
18464/25000 [=====================>........] - ETA: 24s - loss: 7.6176 - accuracy: 0.5032
18496/25000 [=====================>........] - ETA: 24s - loss: 7.6152 - accuracy: 0.5034
18528/25000 [=====================>........] - ETA: 24s - loss: 7.6137 - accuracy: 0.5035
18560/25000 [=====================>........] - ETA: 24s - loss: 7.6121 - accuracy: 0.5036
18592/25000 [=====================>........] - ETA: 24s - loss: 7.6155 - accuracy: 0.5033
18624/25000 [=====================>........] - ETA: 24s - loss: 7.6164 - accuracy: 0.5033
18656/25000 [=====================>........] - ETA: 24s - loss: 7.6165 - accuracy: 0.5033
18688/25000 [=====================>........] - ETA: 23s - loss: 7.6157 - accuracy: 0.5033
18720/25000 [=====================>........] - ETA: 23s - loss: 7.6167 - accuracy: 0.5033
18752/25000 [=====================>........] - ETA: 23s - loss: 7.6159 - accuracy: 0.5033
18784/25000 [=====================>........] - ETA: 23s - loss: 7.6176 - accuracy: 0.5032
18816/25000 [=====================>........] - ETA: 23s - loss: 7.6177 - accuracy: 0.5032
18848/25000 [=====================>........] - ETA: 23s - loss: 7.6194 - accuracy: 0.5031
18880/25000 [=====================>........] - ETA: 23s - loss: 7.6203 - accuracy: 0.5030
18912/25000 [=====================>........] - ETA: 23s - loss: 7.6220 - accuracy: 0.5029
18944/25000 [=====================>........] - ETA: 22s - loss: 7.6197 - accuracy: 0.5031
18976/25000 [=====================>........] - ETA: 22s - loss: 7.6214 - accuracy: 0.5030
19008/25000 [=====================>........] - ETA: 22s - loss: 7.6198 - accuracy: 0.5031
19040/25000 [=====================>........] - ETA: 22s - loss: 7.6247 - accuracy: 0.5027
19072/25000 [=====================>........] - ETA: 22s - loss: 7.6256 - accuracy: 0.5027
19104/25000 [=====================>........] - ETA: 22s - loss: 7.6241 - accuracy: 0.5028
19136/25000 [=====================>........] - ETA: 22s - loss: 7.6234 - accuracy: 0.5028
19168/25000 [======================>.......] - ETA: 22s - loss: 7.6282 - accuracy: 0.5025
19200/25000 [======================>.......] - ETA: 22s - loss: 7.6307 - accuracy: 0.5023
19232/25000 [======================>.......] - ETA: 21s - loss: 7.6291 - accuracy: 0.5024
19264/25000 [======================>.......] - ETA: 21s - loss: 7.6316 - accuracy: 0.5023
19296/25000 [======================>.......] - ETA: 21s - loss: 7.6309 - accuracy: 0.5023
19328/25000 [======================>.......] - ETA: 21s - loss: 7.6293 - accuracy: 0.5024
19360/25000 [======================>.......] - ETA: 21s - loss: 7.6278 - accuracy: 0.5025
19392/25000 [======================>.......] - ETA: 21s - loss: 7.6310 - accuracy: 0.5023
19424/25000 [======================>.......] - ETA: 21s - loss: 7.6335 - accuracy: 0.5022
19456/25000 [======================>.......] - ETA: 21s - loss: 7.6327 - accuracy: 0.5022
19488/25000 [======================>.......] - ETA: 20s - loss: 7.6328 - accuracy: 0.5022
19520/25000 [======================>.......] - ETA: 20s - loss: 7.6313 - accuracy: 0.5023
19552/25000 [======================>.......] - ETA: 20s - loss: 7.6321 - accuracy: 0.5023
19584/25000 [======================>.......] - ETA: 20s - loss: 7.6314 - accuracy: 0.5023
19616/25000 [======================>.......] - ETA: 20s - loss: 7.6322 - accuracy: 0.5022
19648/25000 [======================>.......] - ETA: 20s - loss: 7.6338 - accuracy: 0.5021
19680/25000 [======================>.......] - ETA: 20s - loss: 7.6347 - accuracy: 0.5021
19712/25000 [======================>.......] - ETA: 20s - loss: 7.6324 - accuracy: 0.5022
19744/25000 [======================>.......] - ETA: 19s - loss: 7.6309 - accuracy: 0.5023
19776/25000 [======================>.......] - ETA: 19s - loss: 7.6310 - accuracy: 0.5023
19808/25000 [======================>.......] - ETA: 19s - loss: 7.6310 - accuracy: 0.5023
19840/25000 [======================>.......] - ETA: 19s - loss: 7.6311 - accuracy: 0.5023
19872/25000 [======================>.......] - ETA: 19s - loss: 7.6319 - accuracy: 0.5023
19904/25000 [======================>.......] - ETA: 19s - loss: 7.6304 - accuracy: 0.5024
19936/25000 [======================>.......] - ETA: 19s - loss: 7.6320 - accuracy: 0.5023
19968/25000 [======================>.......] - ETA: 19s - loss: 7.6313 - accuracy: 0.5023
20000/25000 [=======================>......] - ETA: 18s - loss: 7.6291 - accuracy: 0.5024
20032/25000 [=======================>......] - ETA: 18s - loss: 7.6314 - accuracy: 0.5023
20064/25000 [=======================>......] - ETA: 18s - loss: 7.6261 - accuracy: 0.5026
20096/25000 [=======================>......] - ETA: 18s - loss: 7.6262 - accuracy: 0.5026
20128/25000 [=======================>......] - ETA: 18s - loss: 7.6240 - accuracy: 0.5028
20160/25000 [=======================>......] - ETA: 18s - loss: 7.6240 - accuracy: 0.5028
20192/25000 [=======================>......] - ETA: 18s - loss: 7.6264 - accuracy: 0.5026
20224/25000 [=======================>......] - ETA: 18s - loss: 7.6272 - accuracy: 0.5026
20256/25000 [=======================>......] - ETA: 17s - loss: 7.6295 - accuracy: 0.5024
20288/25000 [=======================>......] - ETA: 17s - loss: 7.6281 - accuracy: 0.5025
20320/25000 [=======================>......] - ETA: 17s - loss: 7.6251 - accuracy: 0.5027
20352/25000 [=======================>......] - ETA: 17s - loss: 7.6274 - accuracy: 0.5026
20384/25000 [=======================>......] - ETA: 17s - loss: 7.6290 - accuracy: 0.5025
20416/25000 [=======================>......] - ETA: 17s - loss: 7.6313 - accuracy: 0.5023
20448/25000 [=======================>......] - ETA: 17s - loss: 7.6336 - accuracy: 0.5022
20480/25000 [=======================>......] - ETA: 17s - loss: 7.6359 - accuracy: 0.5020
20512/25000 [=======================>......] - ETA: 17s - loss: 7.6375 - accuracy: 0.5019
20544/25000 [=======================>......] - ETA: 16s - loss: 7.6375 - accuracy: 0.5019
20576/25000 [=======================>......] - ETA: 16s - loss: 7.6361 - accuracy: 0.5020
20608/25000 [=======================>......] - ETA: 16s - loss: 7.6369 - accuracy: 0.5019
20640/25000 [=======================>......] - ETA: 16s - loss: 7.6369 - accuracy: 0.5019
20672/25000 [=======================>......] - ETA: 16s - loss: 7.6369 - accuracy: 0.5019
20704/25000 [=======================>......] - ETA: 16s - loss: 7.6340 - accuracy: 0.5021
20736/25000 [=======================>......] - ETA: 16s - loss: 7.6333 - accuracy: 0.5022
20768/25000 [=======================>......] - ETA: 16s - loss: 7.6341 - accuracy: 0.5021
20800/25000 [=======================>......] - ETA: 15s - loss: 7.6357 - accuracy: 0.5020
20832/25000 [=======================>......] - ETA: 15s - loss: 7.6342 - accuracy: 0.5021
20864/25000 [========================>.....] - ETA: 15s - loss: 7.6313 - accuracy: 0.5023
20896/25000 [========================>.....] - ETA: 15s - loss: 7.6314 - accuracy: 0.5023
20928/25000 [========================>.....] - ETA: 15s - loss: 7.6322 - accuracy: 0.5022
20960/25000 [========================>.....] - ETA: 15s - loss: 7.6308 - accuracy: 0.5023
20992/25000 [========================>.....] - ETA: 15s - loss: 7.6338 - accuracy: 0.5021
21024/25000 [========================>.....] - ETA: 15s - loss: 7.6331 - accuracy: 0.5022
21056/25000 [========================>.....] - ETA: 14s - loss: 7.6309 - accuracy: 0.5023
21088/25000 [========================>.....] - ETA: 14s - loss: 7.6288 - accuracy: 0.5025
21120/25000 [========================>.....] - ETA: 14s - loss: 7.6332 - accuracy: 0.5022
21152/25000 [========================>.....] - ETA: 14s - loss: 7.6383 - accuracy: 0.5018
21184/25000 [========================>.....] - ETA: 14s - loss: 7.6391 - accuracy: 0.5018
21216/25000 [========================>.....] - ETA: 14s - loss: 7.6399 - accuracy: 0.5017
21248/25000 [========================>.....] - ETA: 14s - loss: 7.6370 - accuracy: 0.5019
21280/25000 [========================>.....] - ETA: 14s - loss: 7.6342 - accuracy: 0.5021
21312/25000 [========================>.....] - ETA: 13s - loss: 7.6350 - accuracy: 0.5021
21344/25000 [========================>.....] - ETA: 13s - loss: 7.6343 - accuracy: 0.5021
21376/25000 [========================>.....] - ETA: 13s - loss: 7.6315 - accuracy: 0.5023
21408/25000 [========================>.....] - ETA: 13s - loss: 7.6322 - accuracy: 0.5022
21440/25000 [========================>.....] - ETA: 13s - loss: 7.6330 - accuracy: 0.5022
21472/25000 [========================>.....] - ETA: 13s - loss: 7.6345 - accuracy: 0.5021
21504/25000 [========================>.....] - ETA: 13s - loss: 7.6345 - accuracy: 0.5021
21536/25000 [========================>.....] - ETA: 13s - loss: 7.6396 - accuracy: 0.5018
21568/25000 [========================>.....] - ETA: 13s - loss: 7.6389 - accuracy: 0.5018
21600/25000 [========================>.....] - ETA: 12s - loss: 7.6396 - accuracy: 0.5018
21632/25000 [========================>.....] - ETA: 12s - loss: 7.6418 - accuracy: 0.5016
21664/25000 [========================>.....] - ETA: 12s - loss: 7.6418 - accuracy: 0.5016
21696/25000 [=========================>....] - ETA: 12s - loss: 7.6412 - accuracy: 0.5017
21728/25000 [=========================>....] - ETA: 12s - loss: 7.6440 - accuracy: 0.5015
21760/25000 [=========================>....] - ETA: 12s - loss: 7.6455 - accuracy: 0.5014
21792/25000 [=========================>....] - ETA: 12s - loss: 7.6434 - accuracy: 0.5015
21824/25000 [=========================>....] - ETA: 12s - loss: 7.6420 - accuracy: 0.5016
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6414 - accuracy: 0.5016
21888/25000 [=========================>....] - ETA: 11s - loss: 7.6400 - accuracy: 0.5017
21920/25000 [=========================>....] - ETA: 11s - loss: 7.6400 - accuracy: 0.5017
21952/25000 [=========================>....] - ETA: 11s - loss: 7.6345 - accuracy: 0.5021
21984/25000 [=========================>....] - ETA: 11s - loss: 7.6352 - accuracy: 0.5020
22016/25000 [=========================>....] - ETA: 11s - loss: 7.6339 - accuracy: 0.5021
22048/25000 [=========================>....] - ETA: 11s - loss: 7.6318 - accuracy: 0.5023
22080/25000 [=========================>....] - ETA: 11s - loss: 7.6305 - accuracy: 0.5024
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6299 - accuracy: 0.5024
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6285 - accuracy: 0.5025
22176/25000 [=========================>....] - ETA: 10s - loss: 7.6272 - accuracy: 0.5026
22208/25000 [=========================>....] - ETA: 10s - loss: 7.6245 - accuracy: 0.5027
22240/25000 [=========================>....] - ETA: 10s - loss: 7.6246 - accuracy: 0.5027
22272/25000 [=========================>....] - ETA: 10s - loss: 7.6219 - accuracy: 0.5029
22304/25000 [=========================>....] - ETA: 10s - loss: 7.6233 - accuracy: 0.5028
22336/25000 [=========================>....] - ETA: 10s - loss: 7.6234 - accuracy: 0.5028
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6255 - accuracy: 0.5027 
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6297 - accuracy: 0.5024
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6338 - accuracy: 0.5021
22464/25000 [=========================>....] - ETA: 9s - loss: 7.6318 - accuracy: 0.5023
22496/25000 [=========================>....] - ETA: 9s - loss: 7.6339 - accuracy: 0.5021
22528/25000 [==========================>...] - ETA: 9s - loss: 7.6380 - accuracy: 0.5019
22560/25000 [==========================>...] - ETA: 9s - loss: 7.6388 - accuracy: 0.5018
22592/25000 [==========================>...] - ETA: 9s - loss: 7.6361 - accuracy: 0.5020
22624/25000 [==========================>...] - ETA: 9s - loss: 7.6354 - accuracy: 0.5020
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6368 - accuracy: 0.5019
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6355 - accuracy: 0.5020
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6376 - accuracy: 0.5019
22752/25000 [==========================>...] - ETA: 8s - loss: 7.6376 - accuracy: 0.5019
22784/25000 [==========================>...] - ETA: 8s - loss: 7.6384 - accuracy: 0.5018
22816/25000 [==========================>...] - ETA: 8s - loss: 7.6397 - accuracy: 0.5018
22848/25000 [==========================>...] - ETA: 8s - loss: 7.6438 - accuracy: 0.5015
22880/25000 [==========================>...] - ETA: 8s - loss: 7.6458 - accuracy: 0.5014
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6452 - accuracy: 0.5014
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6452 - accuracy: 0.5014
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6453 - accuracy: 0.5014
23008/25000 [==========================>...] - ETA: 7s - loss: 7.6446 - accuracy: 0.5014
23040/25000 [==========================>...] - ETA: 7s - loss: 7.6427 - accuracy: 0.5016
23072/25000 [==========================>...] - ETA: 7s - loss: 7.6434 - accuracy: 0.5015
23104/25000 [==========================>...] - ETA: 7s - loss: 7.6474 - accuracy: 0.5013
23136/25000 [==========================>...] - ETA: 7s - loss: 7.6487 - accuracy: 0.5012
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6488 - accuracy: 0.5012
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6501 - accuracy: 0.5011
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6495 - accuracy: 0.5011
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6534 - accuracy: 0.5009
23296/25000 [==========================>...] - ETA: 6s - loss: 7.6502 - accuracy: 0.5011
23328/25000 [==========================>...] - ETA: 6s - loss: 7.6515 - accuracy: 0.5010
23360/25000 [===========================>..] - ETA: 6s - loss: 7.6515 - accuracy: 0.5010
23392/25000 [===========================>..] - ETA: 6s - loss: 7.6535 - accuracy: 0.5009
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6542 - accuracy: 0.5008
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6529 - accuracy: 0.5009
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6516 - accuracy: 0.5010
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6549 - accuracy: 0.5008
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6549 - accuracy: 0.5008
23584/25000 [===========================>..] - ETA: 5s - loss: 7.6536 - accuracy: 0.5008
23616/25000 [===========================>..] - ETA: 5s - loss: 7.6549 - accuracy: 0.5008
23648/25000 [===========================>..] - ETA: 5s - loss: 7.6588 - accuracy: 0.5005
23680/25000 [===========================>..] - ETA: 5s - loss: 7.6588 - accuracy: 0.5005
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6614 - accuracy: 0.5003
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6595 - accuracy: 0.5005
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6634 - accuracy: 0.5002
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6621 - accuracy: 0.5003
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23872/25000 [===========================>..] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23904/25000 [===========================>..] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
23936/25000 [===========================>..] - ETA: 4s - loss: 7.6647 - accuracy: 0.5001
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6628 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6615 - accuracy: 0.5003
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6634 - accuracy: 0.5002
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6634 - accuracy: 0.5002
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6634 - accuracy: 0.5002
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6641 - accuracy: 0.5002
24160/25000 [===========================>..] - ETA: 3s - loss: 7.6641 - accuracy: 0.5002
24192/25000 [============================>.] - ETA: 3s - loss: 7.6647 - accuracy: 0.5001
24224/25000 [============================>.] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24256/25000 [============================>.] - ETA: 2s - loss: 7.6710 - accuracy: 0.4997
24288/25000 [============================>.] - ETA: 2s - loss: 7.6710 - accuracy: 0.4997
24320/25000 [============================>.] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24352/25000 [============================>.] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24384/25000 [============================>.] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24416/25000 [============================>.] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
24448/25000 [============================>.] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24480/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24512/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24544/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
24576/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24608/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24640/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24672/25000 [============================>.] - ETA: 1s - loss: 7.6648 - accuracy: 0.5001
24704/25000 [============================>.] - ETA: 1s - loss: 7.6648 - accuracy: 0.5001
24736/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24768/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24800/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24864/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24896/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
25000/25000 [==============================] - 113s 5ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fb3e5ab1a90> 

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
[[ 0.          0.          0.          0.          0.          0.        ]
 [-0.0116224   0.06613225 -0.01020287 -0.01838858 -0.00097584 -0.00455368]
 [ 0.0015585  -0.10099298 -0.03398979  0.10151986 -0.07060461 -0.04303385]
 [-0.05693438  0.04819415  0.12671171 -0.01246475 -0.02565859  0.02140243]
 [-0.12536401 -0.31803432  0.25888991  0.00549721  0.00379011 -0.32846648]
 [ 0.17376611  0.18279992 -0.28862968 -0.1101452   0.22114471 -0.16341592]
 [-0.01541712 -0.15564336  0.28016657  0.05139209  0.102869   -0.10916903]
 [-0.19775188 -0.57420295 -0.35880551 -0.19376308 -0.18523723  0.39265677]
 [-0.21264592  0.36885923  0.34902817 -0.12493829 -0.06562451  0.14311586]
 [ 0.          0.          0.          0.          0.          0.        ]]

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
{'loss': 0.47592861019074917, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
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
{'loss': 0.4501780718564987, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}





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
	Data preprocessing and feature engineering runtime = 0.26s ...
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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:55<01:22, 27.59s/it] 40%|████      | 2/5 [00:55<01:22, 27.60s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.3619323602834413, 'embedding_size_factor': 1.3424122041266782, 'layers.choice': 1, 'learning_rate': 0.005441789574204518, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.4226664835926741e-05} and reward: 0.3752
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd7)\xe6X\xb2\x03\x99X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5z\x858'\x9b\x12X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?vJ!D\x0b\xf8\x1eX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xed\xd5\xe1\xed+\x9b\xf6u." and reward: 0.3752
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd7)\xe6X\xb2\x03\x99X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5z\x858'\x9b\x12X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?vJ!D\x0b\xf8\x1eX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xed\xd5\xe1\xed+\x9b\xf6u." and reward: 0.3752
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 142.84384202957153
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -24.88s of remaining time.
Ensemble size: 51
Ensemble weights: 
[0.78431373 0.21568627]
	0.3872	 = Validation accuracy score
	1.02s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 145.93s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
test

  #### Module init   ############################################ 

  <module 'mlmodels.model_gluon.gluon_automl' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluon_automl.py'> 

  #### Loading params   ############################################## 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB
  Optimizer.opt_registry[name].__name__))
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 506, in main
    test_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 436, in test_cli
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

