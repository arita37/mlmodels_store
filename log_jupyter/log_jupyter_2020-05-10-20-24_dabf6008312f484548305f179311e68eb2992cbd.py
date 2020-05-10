
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/dabf6008312f484548305f179311e68eb2992cbd', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': 'dabf6008312f484548305f179311e68eb2992cbd', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/dabf6008312f484548305f179311e68eb2992cbd

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/dabf6008312f484548305f179311e68eb2992cbd

 ************************************************************************************************************************
/home/runner/work/mlmodels/mlmodels/mlmodels/example/
############ List of files ################################
['ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_titanic.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_svm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras_charcnn_reuters.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras-textcnn.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow_1_lstm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_home_retail.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl_titanic.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow__lstm_json.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_model.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m4.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_hyper.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m4.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m5.py']





 ************************************************************************************************************************
############ Running Jupyter files ################################





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//lightgbm_titanic.ipynb 

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
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//sklearn_titanic_svm.ipynb 

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
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//keras_charcnn_reuters.ipynb 

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
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//keras-textcnn.ipynb 

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
 1327104/17464789 [=>............................] - ETA: 0s
 7200768/17464789 [===========>..................] - ETA: 0s
11714560/17464789 [===================>..........] - ETA: 0s
16162816/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 20:25:14.330695: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 20:25:14.335245: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-10 20:25:14.335648: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557bacc23a40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 20:25:14.335669: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

2020-05-10 20:25:14.844570: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:53 - loss: 9.5833 - accuracy: 0.37502020-05-10 20:25:14.942196: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

   64/25000 [..............................] - ETA: 3:01 - loss: 8.8645 - accuracy: 0.42192020-05-10 20:25:15.032211: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

   96/25000 [..............................] - ETA: 2:23 - loss: 7.9861 - accuracy: 0.47922020-05-10 20:25:15.115427: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

  128/25000 [..............................] - ETA: 2:04 - loss: 7.7864 - accuracy: 0.49222020-05-10 20:25:15.203215: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

  160/25000 [..............................] - ETA: 1:52 - loss: 7.9541 - accuracy: 0.4812
  192/25000 [..............................] - ETA: 1:44 - loss: 8.3854 - accuracy: 0.4531
  224/25000 [..............................] - ETA: 1:38 - loss: 8.5565 - accuracy: 0.4420
  256/25000 [..............................] - ETA: 1:34 - loss: 8.4453 - accuracy: 0.4492
  288/25000 [..............................] - ETA: 1:31 - loss: 8.3055 - accuracy: 0.4583
  320/25000 [..............................] - ETA: 1:28 - loss: 8.2416 - accuracy: 0.4625
  352/25000 [..............................] - ETA: 1:26 - loss: 8.0151 - accuracy: 0.4773
  384/25000 [..............................] - ETA: 1:25 - loss: 8.0659 - accuracy: 0.4740
  416/25000 [..............................] - ETA: 1:23 - loss: 7.9615 - accuracy: 0.4808
  448/25000 [..............................] - ETA: 1:21 - loss: 7.8035 - accuracy: 0.4911
  480/25000 [..............................] - ETA: 1:20 - loss: 7.5388 - accuracy: 0.5083
  512/25000 [..............................] - ETA: 1:19 - loss: 7.4570 - accuracy: 0.5137
  544/25000 [..............................] - ETA: 1:18 - loss: 7.4975 - accuracy: 0.5110
  576/25000 [..............................] - ETA: 1:17 - loss: 7.6134 - accuracy: 0.5035
  608/25000 [..............................] - ETA: 1:16 - loss: 7.5910 - accuracy: 0.5049
  640/25000 [..............................] - ETA: 1:16 - loss: 7.6666 - accuracy: 0.5000
  672/25000 [..............................] - ETA: 1:16 - loss: 7.6666 - accuracy: 0.5000
  704/25000 [..............................] - ETA: 1:15 - loss: 7.6448 - accuracy: 0.5014
  736/25000 [..............................] - ETA: 1:14 - loss: 7.6041 - accuracy: 0.5041
  768/25000 [..............................] - ETA: 1:14 - loss: 7.5868 - accuracy: 0.5052
  800/25000 [..............................] - ETA: 1:13 - loss: 7.6475 - accuracy: 0.5013
  832/25000 [..............................] - ETA: 1:13 - loss: 7.6482 - accuracy: 0.5012
  864/25000 [>.............................] - ETA: 1:12 - loss: 7.6489 - accuracy: 0.5012
  896/25000 [>.............................] - ETA: 1:11 - loss: 7.5982 - accuracy: 0.5045
  928/25000 [>.............................] - ETA: 1:11 - loss: 7.6171 - accuracy: 0.5032
  960/25000 [>.............................] - ETA: 1:11 - loss: 7.6187 - accuracy: 0.5031
  992/25000 [>.............................] - ETA: 1:11 - loss: 7.6202 - accuracy: 0.5030
 1024/25000 [>.............................] - ETA: 1:10 - loss: 7.5918 - accuracy: 0.5049
 1056/25000 [>.............................] - ETA: 1:10 - loss: 7.5940 - accuracy: 0.5047
 1088/25000 [>.............................] - ETA: 1:10 - loss: 7.5680 - accuracy: 0.5064
 1120/25000 [>.............................] - ETA: 1:09 - loss: 7.4886 - accuracy: 0.5116
 1152/25000 [>.............................] - ETA: 1:09 - loss: 7.5601 - accuracy: 0.5069
 1184/25000 [>.............................] - ETA: 1:09 - loss: 7.5371 - accuracy: 0.5084
 1216/25000 [>.............................] - ETA: 1:08 - loss: 7.5279 - accuracy: 0.5090
 1248/25000 [>.............................] - ETA: 1:08 - loss: 7.5192 - accuracy: 0.5096
 1280/25000 [>.............................] - ETA: 1:08 - loss: 7.5229 - accuracy: 0.5094
 1312/25000 [>.............................] - ETA: 1:08 - loss: 7.4796 - accuracy: 0.5122
 1344/25000 [>.............................] - ETA: 1:07 - loss: 7.4727 - accuracy: 0.5126
 1376/25000 [>.............................] - ETA: 1:07 - loss: 7.4660 - accuracy: 0.5131
 1408/25000 [>.............................] - ETA: 1:07 - loss: 7.5033 - accuracy: 0.5107
 1440/25000 [>.............................] - ETA: 1:07 - loss: 7.5495 - accuracy: 0.5076
 1472/25000 [>.............................] - ETA: 1:06 - loss: 7.5625 - accuracy: 0.5068
 1504/25000 [>.............................] - ETA: 1:06 - loss: 7.5851 - accuracy: 0.5053
 1536/25000 [>.............................] - ETA: 1:06 - loss: 7.5967 - accuracy: 0.5046
 1568/25000 [>.............................] - ETA: 1:06 - loss: 7.5688 - accuracy: 0.5064
 1600/25000 [>.............................] - ETA: 1:05 - loss: 7.5325 - accuracy: 0.5088
 1632/25000 [>.............................] - ETA: 1:05 - loss: 7.5445 - accuracy: 0.5080
 1664/25000 [>.............................] - ETA: 1:05 - loss: 7.5284 - accuracy: 0.5090
 1696/25000 [=>............................] - ETA: 1:05 - loss: 7.5491 - accuracy: 0.5077
 1728/25000 [=>............................] - ETA: 1:05 - loss: 7.5779 - accuracy: 0.5058
 1760/25000 [=>............................] - ETA: 1:05 - loss: 7.6056 - accuracy: 0.5040
 1792/25000 [=>............................] - ETA: 1:05 - loss: 7.5896 - accuracy: 0.5050
 1824/25000 [=>............................] - ETA: 1:05 - loss: 7.5573 - accuracy: 0.5071
 1856/25000 [=>............................] - ETA: 1:04 - loss: 7.5427 - accuracy: 0.5081
 1888/25000 [=>............................] - ETA: 1:04 - loss: 7.5529 - accuracy: 0.5074
 1920/25000 [=>............................] - ETA: 1:04 - loss: 7.5628 - accuracy: 0.5068
 1952/25000 [=>............................] - ETA: 1:04 - loss: 7.5566 - accuracy: 0.5072
 1984/25000 [=>............................] - ETA: 1:04 - loss: 7.5816 - accuracy: 0.5055
 2016/25000 [=>............................] - ETA: 1:04 - loss: 7.6286 - accuracy: 0.5025
 2048/25000 [=>............................] - ETA: 1:04 - loss: 7.5843 - accuracy: 0.5054
 2080/25000 [=>............................] - ETA: 1:03 - loss: 7.6224 - accuracy: 0.5029
 2112/25000 [=>............................] - ETA: 1:03 - loss: 7.6739 - accuracy: 0.4995
 2144/25000 [=>............................] - ETA: 1:03 - loss: 7.6809 - accuracy: 0.4991
 2176/25000 [=>............................] - ETA: 1:03 - loss: 7.6807 - accuracy: 0.4991
 2208/25000 [=>............................] - ETA: 1:03 - loss: 7.6805 - accuracy: 0.4991
 2240/25000 [=>............................] - ETA: 1:02 - loss: 7.6940 - accuracy: 0.4982
 2272/25000 [=>............................] - ETA: 1:02 - loss: 7.7139 - accuracy: 0.4969
 2304/25000 [=>............................] - ETA: 1:02 - loss: 7.7265 - accuracy: 0.4961
 2336/25000 [=>............................] - ETA: 1:02 - loss: 7.7126 - accuracy: 0.4970
 2368/25000 [=>............................] - ETA: 1:02 - loss: 7.7119 - accuracy: 0.4970
 2400/25000 [=>............................] - ETA: 1:02 - loss: 7.7241 - accuracy: 0.4963
 2432/25000 [=>............................] - ETA: 1:02 - loss: 7.7171 - accuracy: 0.4967
 2464/25000 [=>............................] - ETA: 1:02 - loss: 7.7226 - accuracy: 0.4963
 2496/25000 [=>............................] - ETA: 1:01 - loss: 7.6789 - accuracy: 0.4992
 2528/25000 [==>...........................] - ETA: 1:01 - loss: 7.6969 - accuracy: 0.4980
 2560/25000 [==>...........................] - ETA: 1:01 - loss: 7.6966 - accuracy: 0.4980
 2592/25000 [==>...........................] - ETA: 1:01 - loss: 7.6725 - accuracy: 0.4996
 2624/25000 [==>...........................] - ETA: 1:01 - loss: 7.6491 - accuracy: 0.5011
 2656/25000 [==>...........................] - ETA: 1:01 - loss: 7.6262 - accuracy: 0.5026
 2688/25000 [==>...........................] - ETA: 1:01 - loss: 7.6552 - accuracy: 0.5007
 2720/25000 [==>...........................] - ETA: 1:01 - loss: 7.6384 - accuracy: 0.5018
 2752/25000 [==>...........................] - ETA: 1:00 - loss: 7.6555 - accuracy: 0.5007
 2784/25000 [==>...........................] - ETA: 1:00 - loss: 7.6336 - accuracy: 0.5022
 2816/25000 [==>...........................] - ETA: 1:00 - loss: 7.6339 - accuracy: 0.5021
 2848/25000 [==>...........................] - ETA: 1:00 - loss: 7.6128 - accuracy: 0.5035
 2880/25000 [==>...........................] - ETA: 1:00 - loss: 7.6027 - accuracy: 0.5042
 2912/25000 [==>...........................] - ETA: 1:00 - loss: 7.5982 - accuracy: 0.5045
 2944/25000 [==>...........................] - ETA: 1:00 - loss: 7.5833 - accuracy: 0.5054
 2976/25000 [==>...........................] - ETA: 1:00 - loss: 7.5893 - accuracy: 0.5050
 3008/25000 [==>...........................] - ETA: 59s - loss: 7.6004 - accuracy: 0.5043 
 3040/25000 [==>...........................] - ETA: 59s - loss: 7.6010 - accuracy: 0.5043
 3072/25000 [==>...........................] - ETA: 59s - loss: 7.6167 - accuracy: 0.5033
 3104/25000 [==>...........................] - ETA: 59s - loss: 7.5975 - accuracy: 0.5045
 3136/25000 [==>...........................] - ETA: 59s - loss: 7.6128 - accuracy: 0.5035
 3168/25000 [==>...........................] - ETA: 59s - loss: 7.6085 - accuracy: 0.5038
 3200/25000 [==>...........................] - ETA: 59s - loss: 7.6331 - accuracy: 0.5022
 3232/25000 [==>...........................] - ETA: 59s - loss: 7.6429 - accuracy: 0.5015
 3264/25000 [==>...........................] - ETA: 59s - loss: 7.6149 - accuracy: 0.5034
 3296/25000 [==>...........................] - ETA: 58s - loss: 7.6201 - accuracy: 0.5030
 3328/25000 [==>...........................] - ETA: 58s - loss: 7.6344 - accuracy: 0.5021
 3360/25000 [===>..........................] - ETA: 58s - loss: 7.6484 - accuracy: 0.5012
 3392/25000 [===>..........................] - ETA: 58s - loss: 7.6576 - accuracy: 0.5006
 3424/25000 [===>..........................] - ETA: 58s - loss: 7.6442 - accuracy: 0.5015
 3456/25000 [===>..........................] - ETA: 58s - loss: 7.6577 - accuracy: 0.5006
 3488/25000 [===>..........................] - ETA: 58s - loss: 7.6754 - accuracy: 0.4994
 3520/25000 [===>..........................] - ETA: 58s - loss: 7.6753 - accuracy: 0.4994
 3552/25000 [===>..........................] - ETA: 58s - loss: 7.6796 - accuracy: 0.4992
 3584/25000 [===>..........................] - ETA: 57s - loss: 7.7008 - accuracy: 0.4978
 3616/25000 [===>..........................] - ETA: 57s - loss: 7.6836 - accuracy: 0.4989
 3648/25000 [===>..........................] - ETA: 57s - loss: 7.6960 - accuracy: 0.4981
 3680/25000 [===>..........................] - ETA: 57s - loss: 7.6916 - accuracy: 0.4984
 3712/25000 [===>..........................] - ETA: 57s - loss: 7.6955 - accuracy: 0.4981
 3744/25000 [===>..........................] - ETA: 57s - loss: 7.6789 - accuracy: 0.4992
 3776/25000 [===>..........................] - ETA: 57s - loss: 7.6666 - accuracy: 0.5000
 3808/25000 [===>..........................] - ETA: 57s - loss: 7.6706 - accuracy: 0.4997
 3840/25000 [===>..........................] - ETA: 56s - loss: 7.6586 - accuracy: 0.5005
 3872/25000 [===>..........................] - ETA: 56s - loss: 7.6468 - accuracy: 0.5013
 3904/25000 [===>..........................] - ETA: 56s - loss: 7.6391 - accuracy: 0.5018
 3936/25000 [===>..........................] - ETA: 56s - loss: 7.6394 - accuracy: 0.5018
 3968/25000 [===>..........................] - ETA: 56s - loss: 7.6357 - accuracy: 0.5020
 4000/25000 [===>..........................] - ETA: 56s - loss: 7.6513 - accuracy: 0.5010
 4032/25000 [===>..........................] - ETA: 56s - loss: 7.6704 - accuracy: 0.4998
 4064/25000 [===>..........................] - ETA: 56s - loss: 7.6704 - accuracy: 0.4998
 4096/25000 [===>..........................] - ETA: 56s - loss: 7.6704 - accuracy: 0.4998
 4128/25000 [===>..........................] - ETA: 56s - loss: 7.6592 - accuracy: 0.5005
 4160/25000 [===>..........................] - ETA: 55s - loss: 7.6629 - accuracy: 0.5002
 4192/25000 [====>.........................] - ETA: 55s - loss: 7.6593 - accuracy: 0.5005
 4224/25000 [====>.........................] - ETA: 55s - loss: 7.6485 - accuracy: 0.5012
 4256/25000 [====>.........................] - ETA: 55s - loss: 7.6486 - accuracy: 0.5012
 4288/25000 [====>.........................] - ETA: 55s - loss: 7.6487 - accuracy: 0.5012
 4320/25000 [====>.........................] - ETA: 55s - loss: 7.6524 - accuracy: 0.5009
 4352/25000 [====>.........................] - ETA: 55s - loss: 7.6455 - accuracy: 0.5014
 4384/25000 [====>.........................] - ETA: 55s - loss: 7.6281 - accuracy: 0.5025
 4416/25000 [====>.........................] - ETA: 55s - loss: 7.6215 - accuracy: 0.5029
 4448/25000 [====>.........................] - ETA: 55s - loss: 7.6218 - accuracy: 0.5029
 4480/25000 [====>.........................] - ETA: 54s - loss: 7.6187 - accuracy: 0.5031
 4512/25000 [====>.........................] - ETA: 54s - loss: 7.6054 - accuracy: 0.5040
 4544/25000 [====>.........................] - ETA: 54s - loss: 7.6093 - accuracy: 0.5037
 4576/25000 [====>.........................] - ETA: 54s - loss: 7.6164 - accuracy: 0.5033
 4608/25000 [====>.........................] - ETA: 54s - loss: 7.6300 - accuracy: 0.5024
 4640/25000 [====>.........................] - ETA: 54s - loss: 7.6104 - accuracy: 0.5037
 4672/25000 [====>.........................] - ETA: 54s - loss: 7.6108 - accuracy: 0.5036
 4704/25000 [====>.........................] - ETA: 54s - loss: 7.6047 - accuracy: 0.5040
 4736/25000 [====>.........................] - ETA: 54s - loss: 7.5954 - accuracy: 0.5046
 4768/25000 [====>.........................] - ETA: 54s - loss: 7.5959 - accuracy: 0.5046
 4800/25000 [====>.........................] - ETA: 53s - loss: 7.5868 - accuracy: 0.5052
 4832/25000 [====>.........................] - ETA: 53s - loss: 7.5809 - accuracy: 0.5056
 4864/25000 [====>.........................] - ETA: 53s - loss: 7.5720 - accuracy: 0.5062
 4896/25000 [====>.........................] - ETA: 53s - loss: 7.5852 - accuracy: 0.5053
 4928/25000 [====>.........................] - ETA: 53s - loss: 7.5857 - accuracy: 0.5053
 4960/25000 [====>.........................] - ETA: 53s - loss: 7.5708 - accuracy: 0.5063
 4992/25000 [====>.........................] - ETA: 53s - loss: 7.5868 - accuracy: 0.5052
 5024/25000 [=====>........................] - ETA: 53s - loss: 7.5842 - accuracy: 0.5054
 5056/25000 [=====>........................] - ETA: 53s - loss: 7.5878 - accuracy: 0.5051
 5088/25000 [=====>........................] - ETA: 53s - loss: 7.5973 - accuracy: 0.5045
 5120/25000 [=====>........................] - ETA: 52s - loss: 7.5947 - accuracy: 0.5047
 5152/25000 [=====>........................] - ETA: 52s - loss: 7.6101 - accuracy: 0.5037
 5184/25000 [=====>........................] - ETA: 52s - loss: 7.5838 - accuracy: 0.5054
 5216/25000 [=====>........................] - ETA: 52s - loss: 7.5843 - accuracy: 0.5054
 5248/25000 [=====>........................] - ETA: 52s - loss: 7.5819 - accuracy: 0.5055
 5280/25000 [=====>........................] - ETA: 52s - loss: 7.5824 - accuracy: 0.5055
 5312/25000 [=====>........................] - ETA: 52s - loss: 7.5887 - accuracy: 0.5051
 5344/25000 [=====>........................] - ETA: 52s - loss: 7.5863 - accuracy: 0.5052
 5376/25000 [=====>........................] - ETA: 52s - loss: 7.5754 - accuracy: 0.5060
 5408/25000 [=====>........................] - ETA: 51s - loss: 7.5787 - accuracy: 0.5057
 5440/25000 [=====>........................] - ETA: 51s - loss: 7.5764 - accuracy: 0.5059
 5472/25000 [=====>........................] - ETA: 51s - loss: 7.5713 - accuracy: 0.5062
 5504/25000 [=====>........................] - ETA: 51s - loss: 7.5803 - accuracy: 0.5056
 5536/25000 [=====>........................] - ETA: 51s - loss: 7.5835 - accuracy: 0.5054
 5568/25000 [=====>........................] - ETA: 51s - loss: 7.5813 - accuracy: 0.5056
 5600/25000 [=====>........................] - ETA: 51s - loss: 7.5817 - accuracy: 0.5055
 5632/25000 [=====>........................] - ETA: 51s - loss: 7.5741 - accuracy: 0.5060
 5664/25000 [=====>........................] - ETA: 51s - loss: 7.5583 - accuracy: 0.5071
 5696/25000 [=====>........................] - ETA: 51s - loss: 7.5697 - accuracy: 0.5063
 5728/25000 [=====>........................] - ETA: 50s - loss: 7.5863 - accuracy: 0.5052
 5760/25000 [=====>........................] - ETA: 50s - loss: 7.5921 - accuracy: 0.5049
 5792/25000 [=====>........................] - ETA: 50s - loss: 7.5687 - accuracy: 0.5064
 5824/25000 [=====>........................] - ETA: 50s - loss: 7.5613 - accuracy: 0.5069
 5856/25000 [======>.......................] - ETA: 50s - loss: 7.5462 - accuracy: 0.5079
 5888/25000 [======>.......................] - ETA: 50s - loss: 7.5468 - accuracy: 0.5078
 5920/25000 [======>.......................] - ETA: 50s - loss: 7.5423 - accuracy: 0.5081
 5952/25000 [======>.......................] - ETA: 50s - loss: 7.5507 - accuracy: 0.5076
 5984/25000 [======>.......................] - ETA: 50s - loss: 7.5488 - accuracy: 0.5077
 6016/25000 [======>.......................] - ETA: 50s - loss: 7.5647 - accuracy: 0.5066
 6048/25000 [======>.......................] - ETA: 49s - loss: 7.5652 - accuracy: 0.5066
 6080/25000 [======>.......................] - ETA: 49s - loss: 7.5657 - accuracy: 0.5066
 6112/25000 [======>.......................] - ETA: 49s - loss: 7.5713 - accuracy: 0.5062
 6144/25000 [======>.......................] - ETA: 49s - loss: 7.5668 - accuracy: 0.5065
 6176/25000 [======>.......................] - ETA: 49s - loss: 7.5574 - accuracy: 0.5071
 6208/25000 [======>.......................] - ETA: 49s - loss: 7.5530 - accuracy: 0.5074
 6240/25000 [======>.......................] - ETA: 49s - loss: 7.5487 - accuracy: 0.5077
 6272/25000 [======>.......................] - ETA: 49s - loss: 7.5542 - accuracy: 0.5073
 6304/25000 [======>.......................] - ETA: 49s - loss: 7.5450 - accuracy: 0.5079
 6336/25000 [======>.......................] - ETA: 49s - loss: 7.5359 - accuracy: 0.5085
 6368/25000 [======>.......................] - ETA: 49s - loss: 7.5270 - accuracy: 0.5091
 6400/25000 [======>.......................] - ETA: 48s - loss: 7.5277 - accuracy: 0.5091
 6432/25000 [======>.......................] - ETA: 48s - loss: 7.5403 - accuracy: 0.5082
 6464/25000 [======>.......................] - ETA: 48s - loss: 7.5433 - accuracy: 0.5080
 6496/25000 [======>.......................] - ETA: 48s - loss: 7.5462 - accuracy: 0.5079
 6528/25000 [======>.......................] - ETA: 48s - loss: 7.5492 - accuracy: 0.5077
 6560/25000 [======>.......................] - ETA: 48s - loss: 7.5498 - accuracy: 0.5076
 6592/25000 [======>.......................] - ETA: 48s - loss: 7.5457 - accuracy: 0.5079
 6624/25000 [======>.......................] - ETA: 48s - loss: 7.5462 - accuracy: 0.5079
 6656/25000 [======>.......................] - ETA: 48s - loss: 7.5445 - accuracy: 0.5080
 6688/25000 [=======>......................] - ETA: 48s - loss: 7.5497 - accuracy: 0.5076
 6720/25000 [=======>......................] - ETA: 48s - loss: 7.5525 - accuracy: 0.5074
 6752/25000 [=======>......................] - ETA: 47s - loss: 7.5485 - accuracy: 0.5077
 6784/25000 [=======>......................] - ETA: 47s - loss: 7.5446 - accuracy: 0.5080
 6816/25000 [=======>......................] - ETA: 47s - loss: 7.5496 - accuracy: 0.5076
 6848/25000 [=======>......................] - ETA: 47s - loss: 7.5502 - accuracy: 0.5076
 6880/25000 [=======>......................] - ETA: 47s - loss: 7.5619 - accuracy: 0.5068
 6912/25000 [=======>......................] - ETA: 47s - loss: 7.5624 - accuracy: 0.5068
 6944/25000 [=======>......................] - ETA: 47s - loss: 7.5628 - accuracy: 0.5068
 6976/25000 [=======>......................] - ETA: 47s - loss: 7.5677 - accuracy: 0.5065
 7008/25000 [=======>......................] - ETA: 47s - loss: 7.5682 - accuracy: 0.5064
 7040/25000 [=======>......................] - ETA: 47s - loss: 7.5708 - accuracy: 0.5063
 7072/25000 [=======>......................] - ETA: 47s - loss: 7.5669 - accuracy: 0.5065
 7104/25000 [=======>......................] - ETA: 46s - loss: 7.5609 - accuracy: 0.5069
 7136/25000 [=======>......................] - ETA: 46s - loss: 7.5656 - accuracy: 0.5066
 7168/25000 [=======>......................] - ETA: 46s - loss: 7.5575 - accuracy: 0.5071
 7200/25000 [=======>......................] - ETA: 46s - loss: 7.5644 - accuracy: 0.5067
 7232/25000 [=======>......................] - ETA: 46s - loss: 7.5649 - accuracy: 0.5066
 7264/25000 [=======>......................] - ETA: 46s - loss: 7.5737 - accuracy: 0.5061
 7296/25000 [=======>......................] - ETA: 46s - loss: 7.5678 - accuracy: 0.5064
 7328/25000 [=======>......................] - ETA: 46s - loss: 7.5683 - accuracy: 0.5064
 7360/25000 [=======>......................] - ETA: 46s - loss: 7.5791 - accuracy: 0.5057
 7392/25000 [=======>......................] - ETA: 46s - loss: 7.5836 - accuracy: 0.5054
 7424/25000 [=======>......................] - ETA: 46s - loss: 7.5778 - accuracy: 0.5058
 7456/25000 [=======>......................] - ETA: 45s - loss: 7.5720 - accuracy: 0.5062
 7488/25000 [=======>......................] - ETA: 45s - loss: 7.5765 - accuracy: 0.5059
 7520/25000 [========>.....................] - ETA: 45s - loss: 7.5769 - accuracy: 0.5059
 7552/25000 [========>.....................] - ETA: 45s - loss: 7.5874 - accuracy: 0.5052
 7584/25000 [========>.....................] - ETA: 45s - loss: 7.5736 - accuracy: 0.5061
 7616/25000 [========>.....................] - ETA: 45s - loss: 7.5740 - accuracy: 0.5060
 7648/25000 [========>.....................] - ETA: 45s - loss: 7.5664 - accuracy: 0.5065
 7680/25000 [========>.....................] - ETA: 45s - loss: 7.5648 - accuracy: 0.5066
 7712/25000 [========>.....................] - ETA: 45s - loss: 7.5652 - accuracy: 0.5066
 7744/25000 [========>.....................] - ETA: 45s - loss: 7.5696 - accuracy: 0.5063
 7776/25000 [========>.....................] - ETA: 45s - loss: 7.5661 - accuracy: 0.5066
 7808/25000 [========>.....................] - ETA: 44s - loss: 7.5645 - accuracy: 0.5067
 7840/25000 [========>.....................] - ETA: 44s - loss: 7.5610 - accuracy: 0.5069
 7872/25000 [========>.....................] - ETA: 44s - loss: 7.5536 - accuracy: 0.5074
 7904/25000 [========>.....................] - ETA: 44s - loss: 7.5638 - accuracy: 0.5067
 7936/25000 [========>.....................] - ETA: 44s - loss: 7.5681 - accuracy: 0.5064
 7968/25000 [========>.....................] - ETA: 44s - loss: 7.5723 - accuracy: 0.5061
 8000/25000 [========>.....................] - ETA: 44s - loss: 7.5765 - accuracy: 0.5059
 8032/25000 [========>.....................] - ETA: 44s - loss: 7.5750 - accuracy: 0.5060
 8064/25000 [========>.....................] - ETA: 44s - loss: 7.5849 - accuracy: 0.5053
 8096/25000 [========>.....................] - ETA: 44s - loss: 7.5965 - accuracy: 0.5046
 8128/25000 [========>.....................] - ETA: 44s - loss: 7.5836 - accuracy: 0.5054
 8160/25000 [========>.....................] - ETA: 43s - loss: 7.5877 - accuracy: 0.5051
 8192/25000 [========>.....................] - ETA: 43s - loss: 7.5936 - accuracy: 0.5048
 8224/25000 [========>.....................] - ETA: 43s - loss: 7.6032 - accuracy: 0.5041
 8256/25000 [========>.....................] - ETA: 43s - loss: 7.6016 - accuracy: 0.5042
 8288/25000 [========>.....................] - ETA: 43s - loss: 7.6130 - accuracy: 0.5035
 8320/25000 [========>.....................] - ETA: 43s - loss: 7.6187 - accuracy: 0.5031
 8352/25000 [=========>....................] - ETA: 43s - loss: 7.6262 - accuracy: 0.5026
 8384/25000 [=========>....................] - ETA: 43s - loss: 7.6337 - accuracy: 0.5021
 8416/25000 [=========>....................] - ETA: 43s - loss: 7.6466 - accuracy: 0.5013
 8448/25000 [=========>....................] - ETA: 43s - loss: 7.6594 - accuracy: 0.5005
 8480/25000 [=========>....................] - ETA: 43s - loss: 7.6540 - accuracy: 0.5008
 8512/25000 [=========>....................] - ETA: 42s - loss: 7.6468 - accuracy: 0.5013
 8544/25000 [=========>....................] - ETA: 42s - loss: 7.6433 - accuracy: 0.5015
 8576/25000 [=========>....................] - ETA: 42s - loss: 7.6434 - accuracy: 0.5015
 8608/25000 [=========>....................] - ETA: 42s - loss: 7.6417 - accuracy: 0.5016
 8640/25000 [=========>....................] - ETA: 42s - loss: 7.6471 - accuracy: 0.5013
 8672/25000 [=========>....................] - ETA: 42s - loss: 7.6542 - accuracy: 0.5008
 8704/25000 [=========>....................] - ETA: 42s - loss: 7.6525 - accuracy: 0.5009
 8736/25000 [=========>....................] - ETA: 42s - loss: 7.6561 - accuracy: 0.5007
 8768/25000 [=========>....................] - ETA: 42s - loss: 7.6526 - accuracy: 0.5009
 8800/25000 [=========>....................] - ETA: 42s - loss: 7.6509 - accuracy: 0.5010
 8832/25000 [=========>....................] - ETA: 42s - loss: 7.6510 - accuracy: 0.5010
 8864/25000 [=========>....................] - ETA: 41s - loss: 7.6545 - accuracy: 0.5008
 8896/25000 [=========>....................] - ETA: 41s - loss: 7.6580 - accuracy: 0.5006
 8928/25000 [=========>....................] - ETA: 41s - loss: 7.6597 - accuracy: 0.5004
 8960/25000 [=========>....................] - ETA: 41s - loss: 7.6649 - accuracy: 0.5001
 8992/25000 [=========>....................] - ETA: 41s - loss: 7.6632 - accuracy: 0.5002
 9024/25000 [=========>....................] - ETA: 41s - loss: 7.6598 - accuracy: 0.5004
 9056/25000 [=========>....................] - ETA: 41s - loss: 7.6582 - accuracy: 0.5006
 9088/25000 [=========>....................] - ETA: 41s - loss: 7.6565 - accuracy: 0.5007
 9120/25000 [=========>....................] - ETA: 41s - loss: 7.6582 - accuracy: 0.5005
 9152/25000 [=========>....................] - ETA: 41s - loss: 7.6599 - accuracy: 0.5004
 9184/25000 [==========>...................] - ETA: 41s - loss: 7.6566 - accuracy: 0.5007
 9216/25000 [==========>...................] - ETA: 40s - loss: 7.6550 - accuracy: 0.5008
 9248/25000 [==========>...................] - ETA: 40s - loss: 7.6550 - accuracy: 0.5008
 9280/25000 [==========>...................] - ETA: 40s - loss: 7.6600 - accuracy: 0.5004
 9312/25000 [==========>...................] - ETA: 40s - loss: 7.6666 - accuracy: 0.5000
 9344/25000 [==========>...................] - ETA: 40s - loss: 7.6617 - accuracy: 0.5003
 9376/25000 [==========>...................] - ETA: 40s - loss: 7.6633 - accuracy: 0.5002
 9408/25000 [==========>...................] - ETA: 40s - loss: 7.6617 - accuracy: 0.5003
 9440/25000 [==========>...................] - ETA: 40s - loss: 7.6569 - accuracy: 0.5006
 9472/25000 [==========>...................] - ETA: 40s - loss: 7.6699 - accuracy: 0.4998
 9504/25000 [==========>...................] - ETA: 40s - loss: 7.6715 - accuracy: 0.4997
 9536/25000 [==========>...................] - ETA: 40s - loss: 7.6682 - accuracy: 0.4999
 9568/25000 [==========>...................] - ETA: 39s - loss: 7.6698 - accuracy: 0.4998
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.6714 - accuracy: 0.4997
 9632/25000 [==========>...................] - ETA: 39s - loss: 7.6682 - accuracy: 0.4999
 9664/25000 [==========>...................] - ETA: 39s - loss: 7.6746 - accuracy: 0.4995
 9696/25000 [==========>...................] - ETA: 39s - loss: 7.6809 - accuracy: 0.4991
 9728/25000 [==========>...................] - ETA: 39s - loss: 7.6777 - accuracy: 0.4993
 9760/25000 [==========>...................] - ETA: 39s - loss: 7.6745 - accuracy: 0.4995
 9792/25000 [==========>...................] - ETA: 39s - loss: 7.6729 - accuracy: 0.4996
 9824/25000 [==========>...................] - ETA: 39s - loss: 7.6775 - accuracy: 0.4993
 9856/25000 [==========>...................] - ETA: 39s - loss: 7.6822 - accuracy: 0.4990
 9888/25000 [==========>...................] - ETA: 39s - loss: 7.6790 - accuracy: 0.4992
 9920/25000 [==========>...................] - ETA: 39s - loss: 7.6743 - accuracy: 0.4995
 9952/25000 [==========>...................] - ETA: 38s - loss: 7.6774 - accuracy: 0.4993
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.6774 - accuracy: 0.4993
10016/25000 [===========>..................] - ETA: 38s - loss: 7.6804 - accuracy: 0.4991
10048/25000 [===========>..................] - ETA: 38s - loss: 7.6865 - accuracy: 0.4987
10080/25000 [===========>..................] - ETA: 38s - loss: 7.6894 - accuracy: 0.4985
10112/25000 [===========>..................] - ETA: 38s - loss: 7.6863 - accuracy: 0.4987
10144/25000 [===========>..................] - ETA: 38s - loss: 7.6893 - accuracy: 0.4985
10176/25000 [===========>..................] - ETA: 38s - loss: 7.6892 - accuracy: 0.4985
10208/25000 [===========>..................] - ETA: 38s - loss: 7.6861 - accuracy: 0.4987
10240/25000 [===========>..................] - ETA: 38s - loss: 7.6861 - accuracy: 0.4987
10272/25000 [===========>..................] - ETA: 38s - loss: 7.6860 - accuracy: 0.4987
10304/25000 [===========>..................] - ETA: 38s - loss: 7.6875 - accuracy: 0.4986
10336/25000 [===========>..................] - ETA: 37s - loss: 7.6904 - accuracy: 0.4985
10368/25000 [===========>..................] - ETA: 37s - loss: 7.6962 - accuracy: 0.4981
10400/25000 [===========>..................] - ETA: 37s - loss: 7.7020 - accuracy: 0.4977
10432/25000 [===========>..................] - ETA: 37s - loss: 7.7034 - accuracy: 0.4976
10464/25000 [===========>..................] - ETA: 37s - loss: 7.7062 - accuracy: 0.4974
10496/25000 [===========>..................] - ETA: 37s - loss: 7.7046 - accuracy: 0.4975
10528/25000 [===========>..................] - ETA: 37s - loss: 7.6943 - accuracy: 0.4982
10560/25000 [===========>..................] - ETA: 37s - loss: 7.6928 - accuracy: 0.4983
10592/25000 [===========>..................] - ETA: 37s - loss: 7.6941 - accuracy: 0.4982
10624/25000 [===========>..................] - ETA: 37s - loss: 7.6984 - accuracy: 0.4979
10656/25000 [===========>..................] - ETA: 37s - loss: 7.7026 - accuracy: 0.4977
10688/25000 [===========>..................] - ETA: 36s - loss: 7.6939 - accuracy: 0.4982
10720/25000 [===========>..................] - ETA: 36s - loss: 7.6938 - accuracy: 0.4982
10752/25000 [===========>..................] - ETA: 36s - loss: 7.6909 - accuracy: 0.4984
10784/25000 [===========>..................] - ETA: 36s - loss: 7.6879 - accuracy: 0.4986
10816/25000 [===========>..................] - ETA: 36s - loss: 7.6893 - accuracy: 0.4985
10848/25000 [============>.................] - ETA: 36s - loss: 7.6864 - accuracy: 0.4987
10880/25000 [============>.................] - ETA: 36s - loss: 7.6821 - accuracy: 0.4990
10912/25000 [============>.................] - ETA: 36s - loss: 7.6807 - accuracy: 0.4991
10944/25000 [============>.................] - ETA: 36s - loss: 7.6750 - accuracy: 0.4995
10976/25000 [============>.................] - ETA: 36s - loss: 7.6722 - accuracy: 0.4996
11008/25000 [============>.................] - ETA: 36s - loss: 7.6666 - accuracy: 0.5000
11040/25000 [============>.................] - ETA: 36s - loss: 7.6611 - accuracy: 0.5004
11072/25000 [============>.................] - ETA: 35s - loss: 7.6638 - accuracy: 0.5002
11104/25000 [============>.................] - ETA: 35s - loss: 7.6639 - accuracy: 0.5002
11136/25000 [============>.................] - ETA: 35s - loss: 7.6666 - accuracy: 0.5000
11168/25000 [============>.................] - ETA: 35s - loss: 7.6707 - accuracy: 0.4997
11200/25000 [============>.................] - ETA: 35s - loss: 7.6666 - accuracy: 0.5000
11232/25000 [============>.................] - ETA: 35s - loss: 7.6721 - accuracy: 0.4996
11264/25000 [============>.................] - ETA: 35s - loss: 7.6707 - accuracy: 0.4997
11296/25000 [============>.................] - ETA: 35s - loss: 7.6693 - accuracy: 0.4998
11328/25000 [============>.................] - ETA: 35s - loss: 7.6720 - accuracy: 0.4996
11360/25000 [============>.................] - ETA: 35s - loss: 7.6734 - accuracy: 0.4996
11392/25000 [============>.................] - ETA: 35s - loss: 7.6760 - accuracy: 0.4994
11424/25000 [============>.................] - ETA: 35s - loss: 7.6760 - accuracy: 0.4994
11456/25000 [============>.................] - ETA: 34s - loss: 7.6773 - accuracy: 0.4993
11488/25000 [============>.................] - ETA: 34s - loss: 7.6853 - accuracy: 0.4988
11520/25000 [============>.................] - ETA: 34s - loss: 7.6826 - accuracy: 0.4990
11552/25000 [============>.................] - ETA: 34s - loss: 7.6825 - accuracy: 0.4990
11584/25000 [============>.................] - ETA: 34s - loss: 7.6838 - accuracy: 0.4989
11616/25000 [============>.................] - ETA: 34s - loss: 7.6851 - accuracy: 0.4988
11648/25000 [============>.................] - ETA: 34s - loss: 7.6850 - accuracy: 0.4988
11680/25000 [=============>................] - ETA: 34s - loss: 7.6863 - accuracy: 0.4987
11712/25000 [=============>................] - ETA: 34s - loss: 7.6849 - accuracy: 0.4988
11744/25000 [=============>................] - ETA: 34s - loss: 7.6797 - accuracy: 0.4991
11776/25000 [=============>................] - ETA: 34s - loss: 7.6731 - accuracy: 0.4996
11808/25000 [=============>................] - ETA: 34s - loss: 7.6757 - accuracy: 0.4994
11840/25000 [=============>................] - ETA: 33s - loss: 7.6783 - accuracy: 0.4992
11872/25000 [=============>................] - ETA: 33s - loss: 7.6731 - accuracy: 0.4996
11904/25000 [=============>................] - ETA: 33s - loss: 7.6718 - accuracy: 0.4997
11936/25000 [=============>................] - ETA: 33s - loss: 7.6743 - accuracy: 0.4995
11968/25000 [=============>................] - ETA: 33s - loss: 7.6756 - accuracy: 0.4994
12000/25000 [=============>................] - ETA: 33s - loss: 7.6807 - accuracy: 0.4991
12032/25000 [=============>................] - ETA: 33s - loss: 7.6755 - accuracy: 0.4994
12064/25000 [=============>................] - ETA: 33s - loss: 7.6742 - accuracy: 0.4995
12096/25000 [=============>................] - ETA: 33s - loss: 7.6654 - accuracy: 0.5001
12128/25000 [=============>................] - ETA: 33s - loss: 7.6654 - accuracy: 0.5001
12160/25000 [=============>................] - ETA: 33s - loss: 7.6654 - accuracy: 0.5001
12192/25000 [=============>................] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
12224/25000 [=============>................] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
12256/25000 [=============>................] - ETA: 32s - loss: 7.6566 - accuracy: 0.5007
12288/25000 [=============>................] - ETA: 32s - loss: 7.6579 - accuracy: 0.5006
12320/25000 [=============>................] - ETA: 32s - loss: 7.6579 - accuracy: 0.5006
12352/25000 [=============>................] - ETA: 32s - loss: 7.6567 - accuracy: 0.5006
12384/25000 [=============>................] - ETA: 32s - loss: 7.6555 - accuracy: 0.5007
12416/25000 [=============>................] - ETA: 32s - loss: 7.6530 - accuracy: 0.5009
12448/25000 [=============>................] - ETA: 32s - loss: 7.6555 - accuracy: 0.5007
12480/25000 [=============>................] - ETA: 32s - loss: 7.6556 - accuracy: 0.5007
12512/25000 [==============>...............] - ETA: 32s - loss: 7.6568 - accuracy: 0.5006
12544/25000 [==============>...............] - ETA: 32s - loss: 7.6544 - accuracy: 0.5008
12576/25000 [==============>...............] - ETA: 31s - loss: 7.6569 - accuracy: 0.5006
12608/25000 [==============>...............] - ETA: 31s - loss: 7.6581 - accuracy: 0.5006
12640/25000 [==============>...............] - ETA: 31s - loss: 7.6618 - accuracy: 0.5003
12672/25000 [==============>...............] - ETA: 31s - loss: 7.6630 - accuracy: 0.5002
12704/25000 [==============>...............] - ETA: 31s - loss: 7.6606 - accuracy: 0.5004
12736/25000 [==============>...............] - ETA: 31s - loss: 7.6618 - accuracy: 0.5003
12768/25000 [==============>...............] - ETA: 31s - loss: 7.6618 - accuracy: 0.5003
12800/25000 [==============>...............] - ETA: 31s - loss: 7.6630 - accuracy: 0.5002
12832/25000 [==============>...............] - ETA: 31s - loss: 7.6630 - accuracy: 0.5002
12864/25000 [==============>...............] - ETA: 31s - loss: 7.6678 - accuracy: 0.4999
12896/25000 [==============>...............] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
12928/25000 [==============>...............] - ETA: 31s - loss: 7.6714 - accuracy: 0.4997
12960/25000 [==============>...............] - ETA: 30s - loss: 7.6702 - accuracy: 0.4998
12992/25000 [==============>...............] - ETA: 30s - loss: 7.6713 - accuracy: 0.4997
13024/25000 [==============>...............] - ETA: 30s - loss: 7.6713 - accuracy: 0.4997
13056/25000 [==============>...............] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
13088/25000 [==============>...............] - ETA: 30s - loss: 7.6643 - accuracy: 0.5002
13120/25000 [==============>...............] - ETA: 30s - loss: 7.6643 - accuracy: 0.5002
13152/25000 [==============>...............] - ETA: 30s - loss: 7.6643 - accuracy: 0.5002
13184/25000 [==============>...............] - ETA: 30s - loss: 7.6643 - accuracy: 0.5002
13216/25000 [==============>...............] - ETA: 30s - loss: 7.6655 - accuracy: 0.5001
13248/25000 [==============>...............] - ETA: 30s - loss: 7.6678 - accuracy: 0.4999
13280/25000 [==============>...............] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
13312/25000 [==============>...............] - ETA: 30s - loss: 7.6678 - accuracy: 0.4999
13344/25000 [===============>..............] - ETA: 29s - loss: 7.6678 - accuracy: 0.4999
13376/25000 [===============>..............] - ETA: 29s - loss: 7.6712 - accuracy: 0.4997
13408/25000 [===============>..............] - ETA: 29s - loss: 7.6689 - accuracy: 0.4999
13440/25000 [===============>..............] - ETA: 29s - loss: 7.6689 - accuracy: 0.4999
13472/25000 [===============>..............] - ETA: 29s - loss: 7.6700 - accuracy: 0.4998
13504/25000 [===============>..............] - ETA: 29s - loss: 7.6700 - accuracy: 0.4998
13536/25000 [===============>..............] - ETA: 29s - loss: 7.6689 - accuracy: 0.4999
13568/25000 [===============>..............] - ETA: 29s - loss: 7.6700 - accuracy: 0.4998
13600/25000 [===============>..............] - ETA: 29s - loss: 7.6666 - accuracy: 0.5000
13632/25000 [===============>..............] - ETA: 29s - loss: 7.6734 - accuracy: 0.4996
13664/25000 [===============>..............] - ETA: 29s - loss: 7.6722 - accuracy: 0.4996
13696/25000 [===============>..............] - ETA: 29s - loss: 7.6722 - accuracy: 0.4996
13728/25000 [===============>..............] - ETA: 28s - loss: 7.6711 - accuracy: 0.4997
13760/25000 [===============>..............] - ETA: 28s - loss: 7.6722 - accuracy: 0.4996
13792/25000 [===============>..............] - ETA: 28s - loss: 7.6733 - accuracy: 0.4996
13824/25000 [===============>..............] - ETA: 28s - loss: 7.6788 - accuracy: 0.4992
13856/25000 [===============>..............] - ETA: 28s - loss: 7.6810 - accuracy: 0.4991
13888/25000 [===============>..............] - ETA: 28s - loss: 7.6832 - accuracy: 0.4989
13920/25000 [===============>..............] - ETA: 28s - loss: 7.6798 - accuracy: 0.4991
13952/25000 [===============>..............] - ETA: 28s - loss: 7.6820 - accuracy: 0.4990
13984/25000 [===============>..............] - ETA: 28s - loss: 7.6853 - accuracy: 0.4988
14016/25000 [===============>..............] - ETA: 28s - loss: 7.6852 - accuracy: 0.4988
14048/25000 [===============>..............] - ETA: 28s - loss: 7.6863 - accuracy: 0.4987
14080/25000 [===============>..............] - ETA: 28s - loss: 7.6840 - accuracy: 0.4989
14112/25000 [===============>..............] - ETA: 27s - loss: 7.6851 - accuracy: 0.4988
14144/25000 [===============>..............] - ETA: 27s - loss: 7.6861 - accuracy: 0.4987
14176/25000 [================>.............] - ETA: 27s - loss: 7.6872 - accuracy: 0.4987
14208/25000 [================>.............] - ETA: 27s - loss: 7.6828 - accuracy: 0.4989
14240/25000 [================>.............] - ETA: 27s - loss: 7.6871 - accuracy: 0.4987
14272/25000 [================>.............] - ETA: 27s - loss: 7.6913 - accuracy: 0.4984
14304/25000 [================>.............] - ETA: 27s - loss: 7.6923 - accuracy: 0.4983
14336/25000 [================>.............] - ETA: 27s - loss: 7.6923 - accuracy: 0.4983
14368/25000 [================>.............] - ETA: 27s - loss: 7.6976 - accuracy: 0.4980
14400/25000 [================>.............] - ETA: 27s - loss: 7.6954 - accuracy: 0.4981
14432/25000 [================>.............] - ETA: 27s - loss: 7.6868 - accuracy: 0.4987
14464/25000 [================>.............] - ETA: 26s - loss: 7.6868 - accuracy: 0.4987
14496/25000 [================>.............] - ETA: 26s - loss: 7.6888 - accuracy: 0.4986
14528/25000 [================>.............] - ETA: 26s - loss: 7.6909 - accuracy: 0.4984
14560/25000 [================>.............] - ETA: 26s - loss: 7.6887 - accuracy: 0.4986
14592/25000 [================>.............] - ETA: 26s - loss: 7.6908 - accuracy: 0.4984
14624/25000 [================>.............] - ETA: 26s - loss: 7.6918 - accuracy: 0.4984
14656/25000 [================>.............] - ETA: 26s - loss: 7.6917 - accuracy: 0.4984
14688/25000 [================>.............] - ETA: 26s - loss: 7.6927 - accuracy: 0.4983
14720/25000 [================>.............] - ETA: 26s - loss: 7.6937 - accuracy: 0.4982
14752/25000 [================>.............] - ETA: 26s - loss: 7.6905 - accuracy: 0.4984
14784/25000 [================>.............] - ETA: 26s - loss: 7.6884 - accuracy: 0.4986
14816/25000 [================>.............] - ETA: 26s - loss: 7.6894 - accuracy: 0.4985
14848/25000 [================>.............] - ETA: 25s - loss: 7.6883 - accuracy: 0.4986
14880/25000 [================>.............] - ETA: 25s - loss: 7.6893 - accuracy: 0.4985
14912/25000 [================>.............] - ETA: 25s - loss: 7.6903 - accuracy: 0.4985
14944/25000 [================>.............] - ETA: 25s - loss: 7.6912 - accuracy: 0.4984
14976/25000 [================>.............] - ETA: 25s - loss: 7.6902 - accuracy: 0.4985
15008/25000 [=================>............] - ETA: 25s - loss: 7.6922 - accuracy: 0.4983
15040/25000 [=================>............] - ETA: 25s - loss: 7.6911 - accuracy: 0.4984
15072/25000 [=================>............] - ETA: 25s - loss: 7.6900 - accuracy: 0.4985
15104/25000 [=================>............] - ETA: 25s - loss: 7.6879 - accuracy: 0.4986
15136/25000 [=================>............] - ETA: 25s - loss: 7.6859 - accuracy: 0.4987
15168/25000 [=================>............] - ETA: 25s - loss: 7.6828 - accuracy: 0.4989
15200/25000 [=================>............] - ETA: 25s - loss: 7.6807 - accuracy: 0.4991
15232/25000 [=================>............] - ETA: 25s - loss: 7.6837 - accuracy: 0.4989
15264/25000 [=================>............] - ETA: 24s - loss: 7.6837 - accuracy: 0.4989
15296/25000 [=================>............] - ETA: 24s - loss: 7.6827 - accuracy: 0.4990
15328/25000 [=================>............] - ETA: 24s - loss: 7.6786 - accuracy: 0.4992
15360/25000 [=================>............] - ETA: 24s - loss: 7.6746 - accuracy: 0.4995
15392/25000 [=================>............] - ETA: 24s - loss: 7.6736 - accuracy: 0.4995
15424/25000 [=================>............] - ETA: 24s - loss: 7.6726 - accuracy: 0.4996
15456/25000 [=================>............] - ETA: 24s - loss: 7.6736 - accuracy: 0.4995
15488/25000 [=================>............] - ETA: 24s - loss: 7.6745 - accuracy: 0.4995
15520/25000 [=================>............] - ETA: 24s - loss: 7.6725 - accuracy: 0.4996
15552/25000 [=================>............] - ETA: 24s - loss: 7.6735 - accuracy: 0.4995
15584/25000 [=================>............] - ETA: 24s - loss: 7.6696 - accuracy: 0.4998
15616/25000 [=================>............] - ETA: 23s - loss: 7.6696 - accuracy: 0.4998
15648/25000 [=================>............] - ETA: 23s - loss: 7.6745 - accuracy: 0.4995
15680/25000 [=================>............] - ETA: 23s - loss: 7.6764 - accuracy: 0.4994
15712/25000 [=================>............] - ETA: 23s - loss: 7.6725 - accuracy: 0.4996
15744/25000 [=================>............] - ETA: 23s - loss: 7.6734 - accuracy: 0.4996
15776/25000 [=================>............] - ETA: 23s - loss: 7.6763 - accuracy: 0.4994
15808/25000 [=================>............] - ETA: 23s - loss: 7.6773 - accuracy: 0.4993
15840/25000 [==================>...........] - ETA: 23s - loss: 7.6773 - accuracy: 0.4993
15872/25000 [==================>...........] - ETA: 23s - loss: 7.6811 - accuracy: 0.4991
15904/25000 [==================>...........] - ETA: 23s - loss: 7.6801 - accuracy: 0.4991
15936/25000 [==================>...........] - ETA: 23s - loss: 7.6811 - accuracy: 0.4991
15968/25000 [==================>...........] - ETA: 23s - loss: 7.6801 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 22s - loss: 7.6810 - accuracy: 0.4991
16032/25000 [==================>...........] - ETA: 22s - loss: 7.6791 - accuracy: 0.4992
16064/25000 [==================>...........] - ETA: 22s - loss: 7.6771 - accuracy: 0.4993
16096/25000 [==================>...........] - ETA: 22s - loss: 7.6819 - accuracy: 0.4990
16128/25000 [==================>...........] - ETA: 22s - loss: 7.6856 - accuracy: 0.4988
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6884 - accuracy: 0.4986
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6884 - accuracy: 0.4986
16224/25000 [==================>...........] - ETA: 22s - loss: 7.6855 - accuracy: 0.4988
16256/25000 [==================>...........] - ETA: 22s - loss: 7.6883 - accuracy: 0.4986
16288/25000 [==================>...........] - ETA: 22s - loss: 7.6902 - accuracy: 0.4985
16320/25000 [==================>...........] - ETA: 22s - loss: 7.6892 - accuracy: 0.4985
16352/25000 [==================>...........] - ETA: 22s - loss: 7.6938 - accuracy: 0.4982
16384/25000 [==================>...........] - ETA: 22s - loss: 7.6956 - accuracy: 0.4981
16416/25000 [==================>...........] - ETA: 21s - loss: 7.6984 - accuracy: 0.4979
16448/25000 [==================>...........] - ETA: 21s - loss: 7.6955 - accuracy: 0.4981
16480/25000 [==================>...........] - ETA: 21s - loss: 7.7010 - accuracy: 0.4978
16512/25000 [==================>...........] - ETA: 21s - loss: 7.7019 - accuracy: 0.4977
16544/25000 [==================>...........] - ETA: 21s - loss: 7.6972 - accuracy: 0.4980
16576/25000 [==================>...........] - ETA: 21s - loss: 7.6934 - accuracy: 0.4983
16608/25000 [==================>...........] - ETA: 21s - loss: 7.6943 - accuracy: 0.4982
16640/25000 [==================>...........] - ETA: 21s - loss: 7.6897 - accuracy: 0.4985
16672/25000 [===================>..........] - ETA: 21s - loss: 7.6878 - accuracy: 0.4986
16704/25000 [===================>..........] - ETA: 21s - loss: 7.6877 - accuracy: 0.4986
16736/25000 [===================>..........] - ETA: 21s - loss: 7.6859 - accuracy: 0.4987
16768/25000 [===================>..........] - ETA: 21s - loss: 7.6849 - accuracy: 0.4988
16800/25000 [===================>..........] - ETA: 20s - loss: 7.6849 - accuracy: 0.4988
16832/25000 [===================>..........] - ETA: 20s - loss: 7.6903 - accuracy: 0.4985
16864/25000 [===================>..........] - ETA: 20s - loss: 7.6875 - accuracy: 0.4986
16896/25000 [===================>..........] - ETA: 20s - loss: 7.6884 - accuracy: 0.4986
16928/25000 [===================>..........] - ETA: 20s - loss: 7.6893 - accuracy: 0.4985
16960/25000 [===================>..........] - ETA: 20s - loss: 7.6910 - accuracy: 0.4984
16992/25000 [===================>..........] - ETA: 20s - loss: 7.6883 - accuracy: 0.4986
17024/25000 [===================>..........] - ETA: 20s - loss: 7.6882 - accuracy: 0.4986
17056/25000 [===================>..........] - ETA: 20s - loss: 7.6864 - accuracy: 0.4987
17088/25000 [===================>..........] - ETA: 20s - loss: 7.6864 - accuracy: 0.4987
17120/25000 [===================>..........] - ETA: 20s - loss: 7.6854 - accuracy: 0.4988
17152/25000 [===================>..........] - ETA: 20s - loss: 7.6854 - accuracy: 0.4988
17184/25000 [===================>..........] - ETA: 19s - loss: 7.6818 - accuracy: 0.4990
17216/25000 [===================>..........] - ETA: 19s - loss: 7.6853 - accuracy: 0.4988
17248/25000 [===================>..........] - ETA: 19s - loss: 7.6817 - accuracy: 0.4990
17280/25000 [===================>..........] - ETA: 19s - loss: 7.6808 - accuracy: 0.4991
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6764 - accuracy: 0.4994
17344/25000 [===================>..........] - ETA: 19s - loss: 7.6772 - accuracy: 0.4993
17376/25000 [===================>..........] - ETA: 19s - loss: 7.6807 - accuracy: 0.4991
17408/25000 [===================>..........] - ETA: 19s - loss: 7.6790 - accuracy: 0.4992
17440/25000 [===================>..........] - ETA: 19s - loss: 7.6772 - accuracy: 0.4993
17472/25000 [===================>..........] - ETA: 19s - loss: 7.6736 - accuracy: 0.4995
17504/25000 [====================>.........] - ETA: 19s - loss: 7.6763 - accuracy: 0.4994
17536/25000 [====================>.........] - ETA: 19s - loss: 7.6719 - accuracy: 0.4997
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6692 - accuracy: 0.4998
17600/25000 [====================>.........] - ETA: 18s - loss: 7.6675 - accuracy: 0.4999
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6684 - accuracy: 0.4999
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6692 - accuracy: 0.4998
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6658 - accuracy: 0.5001
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6683 - accuracy: 0.4999
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6649 - accuracy: 0.5001
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6640 - accuracy: 0.5002
17824/25000 [====================>.........] - ETA: 18s - loss: 7.6640 - accuracy: 0.5002
17856/25000 [====================>.........] - ETA: 18s - loss: 7.6640 - accuracy: 0.5002
17888/25000 [====================>.........] - ETA: 18s - loss: 7.6640 - accuracy: 0.5002
17920/25000 [====================>.........] - ETA: 18s - loss: 7.6658 - accuracy: 0.5001
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6632 - accuracy: 0.5002
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6632 - accuracy: 0.5002
18016/25000 [====================>.........] - ETA: 17s - loss: 7.6590 - accuracy: 0.5005
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6564 - accuracy: 0.5007
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6607 - accuracy: 0.5004
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6590 - accuracy: 0.5005
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6599 - accuracy: 0.5004
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6582 - accuracy: 0.5006
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6565 - accuracy: 0.5007
18240/25000 [====================>.........] - ETA: 17s - loss: 7.6582 - accuracy: 0.5005
18272/25000 [====================>.........] - ETA: 17s - loss: 7.6557 - accuracy: 0.5007
18304/25000 [====================>.........] - ETA: 17s - loss: 7.6574 - accuracy: 0.5006
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6591 - accuracy: 0.5005
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6558 - accuracy: 0.5007
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6575 - accuracy: 0.5006
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6583 - accuracy: 0.5005
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6583 - accuracy: 0.5005
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6583 - accuracy: 0.5005
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6559 - accuracy: 0.5007
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6584 - accuracy: 0.5005
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6608 - accuracy: 0.5004
18624/25000 [=====================>........] - ETA: 16s - loss: 7.6600 - accuracy: 0.5004
18656/25000 [=====================>........] - ETA: 16s - loss: 7.6576 - accuracy: 0.5006
18688/25000 [=====================>........] - ETA: 16s - loss: 7.6551 - accuracy: 0.5007
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6568 - accuracy: 0.5006
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6593 - accuracy: 0.5005
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6609 - accuracy: 0.5004
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6585 - accuracy: 0.5005
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6593 - accuracy: 0.5005
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6569 - accuracy: 0.5006
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6561 - accuracy: 0.5007
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6561 - accuracy: 0.5007
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6585 - accuracy: 0.5005
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6577 - accuracy: 0.5006
19040/25000 [=====================>........] - ETA: 15s - loss: 7.6594 - accuracy: 0.5005
19072/25000 [=====================>........] - ETA: 15s - loss: 7.6578 - accuracy: 0.5006
19104/25000 [=====================>........] - ETA: 15s - loss: 7.6554 - accuracy: 0.5007
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6562 - accuracy: 0.5007
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6594 - accuracy: 0.5005
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6594 - accuracy: 0.5005
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6571 - accuracy: 0.5006
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6579 - accuracy: 0.5006
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6579 - accuracy: 0.5006
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6579 - accuracy: 0.5006
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6603 - accuracy: 0.5004
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6619 - accuracy: 0.5003
19424/25000 [======================>.......] - ETA: 14s - loss: 7.6579 - accuracy: 0.5006
19456/25000 [======================>.......] - ETA: 14s - loss: 7.6603 - accuracy: 0.5004
19488/25000 [======================>.......] - ETA: 14s - loss: 7.6635 - accuracy: 0.5002
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6650 - accuracy: 0.5001
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6627 - accuracy: 0.5003
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6619 - accuracy: 0.5003
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6611 - accuracy: 0.5004
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6573 - accuracy: 0.5006
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6588 - accuracy: 0.5005
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6596 - accuracy: 0.5005
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6596 - accuracy: 0.5005
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6596 - accuracy: 0.5005
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6581 - accuracy: 0.5006
19840/25000 [======================>.......] - ETA: 13s - loss: 7.6597 - accuracy: 0.5005
19872/25000 [======================>.......] - ETA: 13s - loss: 7.6574 - accuracy: 0.5006
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6597 - accuracy: 0.5005
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6597 - accuracy: 0.5005
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6589 - accuracy: 0.5005
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6636 - accuracy: 0.5002
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6597 - accuracy: 0.5004
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6574 - accuracy: 0.5006
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6598 - accuracy: 0.5004
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6598 - accuracy: 0.5004
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6636 - accuracy: 0.5002
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6613 - accuracy: 0.5003
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6621 - accuracy: 0.5003
20256/25000 [=======================>......] - ETA: 12s - loss: 7.6606 - accuracy: 0.5004
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6583 - accuracy: 0.5005
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6553 - accuracy: 0.5007
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6531 - accuracy: 0.5009
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6546 - accuracy: 0.5008
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6539 - accuracy: 0.5008
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6479 - accuracy: 0.5012
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6479 - accuracy: 0.5012
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6479 - accuracy: 0.5012
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6427 - accuracy: 0.5016
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6420 - accuracy: 0.5016
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6391 - accuracy: 0.5018
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6384 - accuracy: 0.5018
20672/25000 [=======================>......] - ETA: 11s - loss: 7.6407 - accuracy: 0.5017
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6414 - accuracy: 0.5016
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6385 - accuracy: 0.5018
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6400 - accuracy: 0.5017
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6386 - accuracy: 0.5018
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6409 - accuracy: 0.5017
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6402 - accuracy: 0.5017
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6446 - accuracy: 0.5014
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6439 - accuracy: 0.5015
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6403 - accuracy: 0.5017
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6418 - accuracy: 0.5016
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6411 - accuracy: 0.5017
21056/25000 [========================>.....] - ETA: 10s - loss: 7.6397 - accuracy: 0.5018
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6390 - accuracy: 0.5018 
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6398 - accuracy: 0.5018
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6376 - accuracy: 0.5019
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6369 - accuracy: 0.5019
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6392 - accuracy: 0.5018
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6392 - accuracy: 0.5018
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6392 - accuracy: 0.5018
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6422 - accuracy: 0.5016
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6393 - accuracy: 0.5018
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6394 - accuracy: 0.5018
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6380 - accuracy: 0.5019
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6394 - accuracy: 0.5018
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6395 - accuracy: 0.5018
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6395 - accuracy: 0.5018
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6410 - accuracy: 0.5017
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6410 - accuracy: 0.5017
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6404 - accuracy: 0.5017
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6397 - accuracy: 0.5018
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6418 - accuracy: 0.5016
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6412 - accuracy: 0.5017
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6384 - accuracy: 0.5018
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6427 - accuracy: 0.5016
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6392 - accuracy: 0.5018
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6413 - accuracy: 0.5016
21856/25000 [=========================>....] - ETA: 8s - loss: 7.6456 - accuracy: 0.5014
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6491 - accuracy: 0.5011
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6505 - accuracy: 0.5010
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6540 - accuracy: 0.5008
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6541 - accuracy: 0.5008
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6569 - accuracy: 0.5006
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6541 - accuracy: 0.5008
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6520 - accuracy: 0.5010
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6507 - accuracy: 0.5010
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6528 - accuracy: 0.5009
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6528 - accuracy: 0.5009
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6535 - accuracy: 0.5009
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6549 - accuracy: 0.5008
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6563 - accuracy: 0.5007
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6577 - accuracy: 0.5006
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6549 - accuracy: 0.5008
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6598 - accuracy: 0.5004
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6578 - accuracy: 0.5006
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6557 - accuracy: 0.5007
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6564 - accuracy: 0.5007
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6544 - accuracy: 0.5008
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6565 - accuracy: 0.5007
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6565 - accuracy: 0.5007
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6558 - accuracy: 0.5007
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6545 - accuracy: 0.5008
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6565 - accuracy: 0.5007
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6579 - accuracy: 0.5006
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6592 - accuracy: 0.5005
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6613 - accuracy: 0.5003
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6626 - accuracy: 0.5003
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6619 - accuracy: 0.5003
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6646 - accuracy: 0.5001
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6660 - accuracy: 0.5000
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6706 - accuracy: 0.4997
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6706 - accuracy: 0.4997
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6712 - accuracy: 0.4997
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6725 - accuracy: 0.4996
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6712 - accuracy: 0.4997
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6725 - accuracy: 0.4996
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6731 - accuracy: 0.4996
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6731 - accuracy: 0.4996
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6725 - accuracy: 0.4996
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6724 - accuracy: 0.4996
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6718 - accuracy: 0.4997
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6711 - accuracy: 0.4997
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6750 - accuracy: 0.4995
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6794 - accuracy: 0.4992
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6787 - accuracy: 0.4992
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6775 - accuracy: 0.4993
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6762 - accuracy: 0.4994
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6755 - accuracy: 0.4994
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
24192/25000 [============================>.] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
24224/25000 [============================>.] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24256/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24288/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24320/25000 [============================>.] - ETA: 1s - loss: 7.6736 - accuracy: 0.4995
24352/25000 [============================>.] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24384/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4995
24416/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24448/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24480/25000 [============================>.] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24512/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24544/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24576/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24608/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24640/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24672/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24704/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24736/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24768/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24800/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24832/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24864/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24896/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24928/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
25000/25000 [==============================] - 75s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
Loading data...
Using TensorFlow backend.





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//fashion_MNIST_mlmodels.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//mnist_mlmodels_.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//tensorflow_1_lstm.ipynb 

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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f57c7c28a58> 

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
 [ 0.08562652  0.07642957  0.11764943 -0.045536   -0.02780383  0.05554304]
 [-0.18104567 -0.0026969  -0.18287419 -0.14364974 -0.17581998 -0.17880642]
 [ 0.088189    0.23385185  0.02304245  0.0898997  -0.11460979  0.00419524]
 [ 0.4173218  -0.23263063 -0.08013318  0.11932807 -0.07809203  0.23376144]
 [-0.15081312 -0.35113111 -0.03132757  0.08451509 -0.20135492 -0.09957401]
 [-0.61774999  0.24173899 -0.35513872  0.17141539  0.19385974 -0.08625808]
 [ 0.22378469  0.44557217  1.13413322 -0.07157344  0.49915156 -0.40641758]
 [ 0.16407236  0.34678569  0.81725985  0.19834925  0.22461434  0.34715933]
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
{'loss': 0.41297411173582077, 'loss_history': []}

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
{'loss': 0.4836452305316925, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed [Errno 13] Permission denied: '/model/'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//lightgbm_home_retail.ipynb 

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
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//gluon_automl.ipynb 

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
	Data preprocessing and feature engineering runtime = 0.27s ...
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:52<01:19, 26.36s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.22245604744942432, 'embedding_size_factor': 0.6477992993514182, 'layers.choice': 3, 'learning_rate': 0.00522807672475668, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 5.364699496146314e-07} and reward: 0.3866
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xccyp\x94K\xdd\xa6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe4\xba\xc5\x98\xa2\xc0\xf7X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?uj\t(\xdcf\x8cX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xa2\x00=\xe5%\xc1\xedu.' and reward: 0.3866
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xccyp\x94K\xdd\xa6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe4\xba\xc5\x98\xa2\xc0\xf7X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?uj\t(\xdcf\x8cX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xa2\x00=\xe5%\xc1\xedu.' and reward: 0.3866
 60%|██████    | 3/5 [01:46<01:09, 34.62s/it] 60%|██████    | 3/5 [01:46<01:11, 35.54s/it]
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.30436207348289873, 'embedding_size_factor': 0.900305000322708, 'layers.choice': 0, 'learning_rate': 0.0015944497719992015, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 6.067976631717318e-10} and reward: 0.374
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd3z\xab\x0f\xf0\x1dyX\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\xcfLn\x99\xf6,X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?Z\x1f\x9bh\r\xf2NX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x04\xd9r\xddhn|u.' and reward: 0.374
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd3z\xab\x0f\xf0\x1dyX\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\xcfLn\x99\xf6,X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?Z\x1f\x9bh\r\xf2NX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x04\xd9r\xddhn|u.' and reward: 0.374
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 159.934184551239
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.22245604744942432, 'embedding_size_factor': 0.6477992993514182, 'layers.choice': 3, 'learning_rate': 0.00522807672475668, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 5.364699496146314e-07}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.73s of the -42.62s of remaining time.
Ensemble size: 33
Ensemble weights: 
[0.33333333 0.57575758 0.09090909]
	0.3924	 = Validation accuracy score
	1.09s	 = Training runtime
	0.01s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 163.75s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 499, in main
    test_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 429, in test_cli
    test_module(arg.model_uri, param_pars=param_pars)  # '1_lstm'
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 255, in test_module
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluon_automl.py", line 109, in get_params
    return model_pars, data_pars, compute_pars, out_pars
UnboundLocalError: local variable 'model_pars' referenced before assignment





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//vision_mnist.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//lightgbm_glass.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      8[0m [0;32mimport[0m [0mjson[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m [0;34m[0m[0m
[0;32m---> 10[0;31m [0mprint[0m[0;34m([0m [0mos[0m[0;34m.[0m[0mgetcwd[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mNameError[0m: name 'os' is not defined





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//gluon_automl_titanic.ipynb 

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
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//vison_fashion_MNIST.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//tensorflow__lstm_json.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//tensorflow__lstm_json.ipynb[0m in [0;36m<module>[0;34m[0m
[1;32m      5[0m [0;32mimport[0m [0mjson[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m
[0;32m----> 7[0;31m [0mprint[0m[0;34m([0m [0mos[0m[0;34m.[0m[0mgetcwd[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mNameError[0m: name 'os' is not defined





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//lightgbm.ipynb 

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
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//sklearn_titanic_randomForest.ipynb 

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
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//sklearn.ipynb 

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
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb 

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
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//vision_mnist.py 

[0;36m  File [0;32m"/home/runner/work/mlmodels/mlmodels/mlmodels/example/vision_mnist.py"[0;36m, line [0;32m15[0m
[0;31m    !git clone https://github.com/ahmed3bbas/mlmodels.git[0m
[0m    ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax






 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//arun_model.py 

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
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//lightgbm_glass.py 

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
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//benchmark_timeseries_m4.py 






 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//arun_hyper.py 

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
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example/benchmark_timeseries_m4.py 






 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example/benchmark_timeseries_m5.py 

[0;36m  File [0;32m"/home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py"[0;36m, line [0;32m248[0m
[0;31m    We then reshape the forecasts into the correct data shape for submission ...[0m
[0m          ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax






 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/refs/heads/dev/mlmodels/example//benchmark_timeseries_m5.py 

[0;36m  File [0;32m"/home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py"[0;36m, line [0;32m248[0m
[0;31m    We then reshape the forecasts into the correct data shape for submission ...[0m
[0m          ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax

