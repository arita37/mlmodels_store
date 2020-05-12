
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

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
[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     87[0m [0;34m[0m[0m
[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     90[0m [0;34m[0m[0m
[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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
 2375680/17464789 [===>..........................] - ETA: 0s
 8896512/17464789 [==============>...............] - ETA: 0s
16744448/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 10:14:25.764418: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 10:14:25.778924: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-12 10:14:25.779498: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564cad1a2c90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 10:14:25.779813: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 12:45 - loss: 9.1041 - accuracy: 0.4062
   64/25000 [..............................] - ETA: 6:55 - loss: 8.1458 - accuracy: 0.4688 
   96/25000 [..............................] - ETA: 4:56 - loss: 8.3055 - accuracy: 0.4583
  128/25000 [..............................] - ETA: 4:00 - loss: 8.2656 - accuracy: 0.4609
  160/25000 [..............................] - ETA: 3:23 - loss: 8.3374 - accuracy: 0.4563
  192/25000 [..............................] - ETA: 2:59 - loss: 8.3854 - accuracy: 0.4531
  224/25000 [..............................] - ETA: 2:42 - loss: 8.0089 - accuracy: 0.4777
  256/25000 [..............................] - ETA: 2:29 - loss: 8.2057 - accuracy: 0.4648
  288/25000 [..............................] - ETA: 2:19 - loss: 8.1990 - accuracy: 0.4653
  320/25000 [..............................] - ETA: 2:12 - loss: 8.2895 - accuracy: 0.4594
  352/25000 [..............................] - ETA: 2:06 - loss: 8.1893 - accuracy: 0.4659
  384/25000 [..............................] - ETA: 2:00 - loss: 8.0260 - accuracy: 0.4766
  416/25000 [..............................] - ETA: 1:56 - loss: 7.9246 - accuracy: 0.4832
  448/25000 [..............................] - ETA: 1:52 - loss: 8.0773 - accuracy: 0.4732
  480/25000 [..............................] - ETA: 1:48 - loss: 7.9541 - accuracy: 0.4812
  512/25000 [..............................] - ETA: 1:45 - loss: 7.9960 - accuracy: 0.4785
  544/25000 [..............................] - ETA: 1:42 - loss: 8.0049 - accuracy: 0.4779
  576/25000 [..............................] - ETA: 1:40 - loss: 8.0659 - accuracy: 0.4740
  608/25000 [..............................] - ETA: 1:38 - loss: 8.0701 - accuracy: 0.4737
  640/25000 [..............................] - ETA: 1:36 - loss: 8.0260 - accuracy: 0.4766
  672/25000 [..............................] - ETA: 1:34 - loss: 8.0089 - accuracy: 0.4777
  704/25000 [..............................] - ETA: 1:32 - loss: 7.9715 - accuracy: 0.4801
  736/25000 [..............................] - ETA: 1:31 - loss: 7.9583 - accuracy: 0.4810
  768/25000 [..............................] - ETA: 1:29 - loss: 7.9861 - accuracy: 0.4792
  800/25000 [..............................] - ETA: 1:28 - loss: 7.9925 - accuracy: 0.4787
  832/25000 [..............................] - ETA: 1:27 - loss: 7.9615 - accuracy: 0.4808
  864/25000 [>.............................] - ETA: 1:26 - loss: 7.9506 - accuracy: 0.4815
  896/25000 [>.............................] - ETA: 1:25 - loss: 7.9575 - accuracy: 0.4810
  928/25000 [>.............................] - ETA: 1:24 - loss: 8.0136 - accuracy: 0.4774
  960/25000 [>.............................] - ETA: 1:23 - loss: 7.9381 - accuracy: 0.4823
  992/25000 [>.............................] - ETA: 1:22 - loss: 7.9294 - accuracy: 0.4829
 1024/25000 [>.............................] - ETA: 1:21 - loss: 7.8763 - accuracy: 0.4863
 1056/25000 [>.............................] - ETA: 1:20 - loss: 7.7537 - accuracy: 0.4943
 1088/25000 [>.............................] - ETA: 1:20 - loss: 7.7371 - accuracy: 0.4954
 1120/25000 [>.............................] - ETA: 1:19 - loss: 7.7077 - accuracy: 0.4973
 1152/25000 [>.............................] - ETA: 1:18 - loss: 7.7332 - accuracy: 0.4957
 1184/25000 [>.............................] - ETA: 1:18 - loss: 7.7184 - accuracy: 0.4966
 1216/25000 [>.............................] - ETA: 1:17 - loss: 7.7171 - accuracy: 0.4967
 1248/25000 [>.............................] - ETA: 1:17 - loss: 7.6420 - accuracy: 0.5016
 1280/25000 [>.............................] - ETA: 1:16 - loss: 7.6666 - accuracy: 0.5000
 1312/25000 [>.............................] - ETA: 1:16 - loss: 7.6316 - accuracy: 0.5023
 1344/25000 [>.............................] - ETA: 1:15 - loss: 7.6552 - accuracy: 0.5007
 1376/25000 [>.............................] - ETA: 1:15 - loss: 7.6443 - accuracy: 0.5015
 1408/25000 [>.............................] - ETA: 1:14 - loss: 7.6231 - accuracy: 0.5028
 1440/25000 [>.............................] - ETA: 1:14 - loss: 7.6453 - accuracy: 0.5014
 1472/25000 [>.............................] - ETA: 1:13 - loss: 7.6458 - accuracy: 0.5014
 1504/25000 [>.............................] - ETA: 1:13 - loss: 7.6156 - accuracy: 0.5033
 1536/25000 [>.............................] - ETA: 1:13 - loss: 7.6766 - accuracy: 0.4993
 1568/25000 [>.............................] - ETA: 1:12 - loss: 7.6568 - accuracy: 0.5006
 1600/25000 [>.............................] - ETA: 1:12 - loss: 7.6283 - accuracy: 0.5025
 1632/25000 [>.............................] - ETA: 1:12 - loss: 7.6102 - accuracy: 0.5037
 1664/25000 [>.............................] - ETA: 1:11 - loss: 7.6482 - accuracy: 0.5012
 1696/25000 [=>............................] - ETA: 1:11 - loss: 7.6666 - accuracy: 0.5000
 1728/25000 [=>............................] - ETA: 1:11 - loss: 7.6666 - accuracy: 0.5000
 1760/25000 [=>............................] - ETA: 1:10 - loss: 7.6928 - accuracy: 0.4983
 1792/25000 [=>............................] - ETA: 1:10 - loss: 7.6837 - accuracy: 0.4989
 1824/25000 [=>............................] - ETA: 1:10 - loss: 7.6918 - accuracy: 0.4984
 1856/25000 [=>............................] - ETA: 1:09 - loss: 7.6914 - accuracy: 0.4984
 1888/25000 [=>............................] - ETA: 1:09 - loss: 7.7478 - accuracy: 0.4947
 1920/25000 [=>............................] - ETA: 1:09 - loss: 7.7305 - accuracy: 0.4958
 1952/25000 [=>............................] - ETA: 1:08 - loss: 7.7216 - accuracy: 0.4964
 1984/25000 [=>............................] - ETA: 1:08 - loss: 7.7516 - accuracy: 0.4945
 2016/25000 [=>............................] - ETA: 1:08 - loss: 7.7427 - accuracy: 0.4950
 2048/25000 [=>............................] - ETA: 1:08 - loss: 7.7639 - accuracy: 0.4937
 2080/25000 [=>............................] - ETA: 1:07 - loss: 7.7625 - accuracy: 0.4938
 2112/25000 [=>............................] - ETA: 1:07 - loss: 7.7392 - accuracy: 0.4953
 2144/25000 [=>............................] - ETA: 1:07 - loss: 7.7238 - accuracy: 0.4963
 2176/25000 [=>............................] - ETA: 1:07 - loss: 7.7300 - accuracy: 0.4959
 2208/25000 [=>............................] - ETA: 1:06 - loss: 7.7222 - accuracy: 0.4964
 2240/25000 [=>............................] - ETA: 1:06 - loss: 7.7008 - accuracy: 0.4978
 2272/25000 [=>............................] - ETA: 1:06 - loss: 7.7071 - accuracy: 0.4974
 2304/25000 [=>............................] - ETA: 1:06 - loss: 7.6799 - accuracy: 0.4991
 2336/25000 [=>............................] - ETA: 1:05 - loss: 7.6797 - accuracy: 0.4991
 2368/25000 [=>............................] - ETA: 1:05 - loss: 7.7055 - accuracy: 0.4975
 2400/25000 [=>............................] - ETA: 1:05 - loss: 7.7369 - accuracy: 0.4954
 2432/25000 [=>............................] - ETA: 1:05 - loss: 7.7486 - accuracy: 0.4947
 2464/25000 [=>............................] - ETA: 1:04 - loss: 7.7351 - accuracy: 0.4955
 2496/25000 [=>............................] - ETA: 1:04 - loss: 7.7526 - accuracy: 0.4944
 2528/25000 [==>...........................] - ETA: 1:04 - loss: 7.7455 - accuracy: 0.4949
 2560/25000 [==>...........................] - ETA: 1:04 - loss: 7.7565 - accuracy: 0.4941
 2592/25000 [==>...........................] - ETA: 1:04 - loss: 7.7376 - accuracy: 0.4954
 2624/25000 [==>...........................] - ETA: 1:04 - loss: 7.7601 - accuracy: 0.4939
 2656/25000 [==>...........................] - ETA: 1:04 - loss: 7.7532 - accuracy: 0.4944
 2688/25000 [==>...........................] - ETA: 1:03 - loss: 7.7522 - accuracy: 0.4944
 2720/25000 [==>...........................] - ETA: 1:03 - loss: 7.7455 - accuracy: 0.4949
 2752/25000 [==>...........................] - ETA: 1:03 - loss: 7.7446 - accuracy: 0.4949
 2784/25000 [==>...........................] - ETA: 1:03 - loss: 7.7492 - accuracy: 0.4946
 2816/25000 [==>...........................] - ETA: 1:03 - loss: 7.7701 - accuracy: 0.4933
 2848/25000 [==>...........................] - ETA: 1:03 - loss: 7.7635 - accuracy: 0.4937
 2880/25000 [==>...........................] - ETA: 1:03 - loss: 7.7305 - accuracy: 0.4958
 2912/25000 [==>...........................] - ETA: 1:02 - loss: 7.7456 - accuracy: 0.4948
 2944/25000 [==>...........................] - ETA: 1:02 - loss: 7.7395 - accuracy: 0.4952
 2976/25000 [==>...........................] - ETA: 1:02 - loss: 7.7336 - accuracy: 0.4956
 3008/25000 [==>...........................] - ETA: 1:02 - loss: 7.7125 - accuracy: 0.4970
 3040/25000 [==>...........................] - ETA: 1:02 - loss: 7.7171 - accuracy: 0.4967
 3072/25000 [==>...........................] - ETA: 1:02 - loss: 7.7515 - accuracy: 0.4945
 3104/25000 [==>...........................] - ETA: 1:01 - loss: 7.7704 - accuracy: 0.4932
 3136/25000 [==>...........................] - ETA: 1:01 - loss: 7.7986 - accuracy: 0.4914
 3168/25000 [==>...........................] - ETA: 1:01 - loss: 7.7683 - accuracy: 0.4934
 3200/25000 [==>...........................] - ETA: 1:01 - loss: 7.7672 - accuracy: 0.4934
 3232/25000 [==>...........................] - ETA: 1:01 - loss: 7.7615 - accuracy: 0.4938
 3264/25000 [==>...........................] - ETA: 1:01 - loss: 7.7888 - accuracy: 0.4920
 3296/25000 [==>...........................] - ETA: 1:01 - loss: 7.7783 - accuracy: 0.4927
 3328/25000 [==>...........................] - ETA: 1:00 - loss: 7.7818 - accuracy: 0.4925
 3360/25000 [===>..........................] - ETA: 1:00 - loss: 7.7807 - accuracy: 0.4926
 3392/25000 [===>..........................] - ETA: 1:00 - loss: 7.7615 - accuracy: 0.4938
 3424/25000 [===>..........................] - ETA: 1:00 - loss: 7.7607 - accuracy: 0.4939
 3456/25000 [===>..........................] - ETA: 1:00 - loss: 7.7642 - accuracy: 0.4936
 3488/25000 [===>..........................] - ETA: 1:00 - loss: 7.7589 - accuracy: 0.4940
 3520/25000 [===>..........................] - ETA: 1:00 - loss: 7.7625 - accuracy: 0.4938
 3552/25000 [===>..........................] - ETA: 1:00 - loss: 7.7573 - accuracy: 0.4941
 3584/25000 [===>..........................] - ETA: 59s - loss: 7.7607 - accuracy: 0.4939 
 3616/25000 [===>..........................] - ETA: 59s - loss: 7.7557 - accuracy: 0.4942
 3648/25000 [===>..........................] - ETA: 59s - loss: 7.7549 - accuracy: 0.4942
 3680/25000 [===>..........................] - ETA: 59s - loss: 7.7458 - accuracy: 0.4948
 3712/25000 [===>..........................] - ETA: 59s - loss: 7.7534 - accuracy: 0.4943
 3744/25000 [===>..........................] - ETA: 59s - loss: 7.7567 - accuracy: 0.4941
 3776/25000 [===>..........................] - ETA: 59s - loss: 7.7397 - accuracy: 0.4952
 3808/25000 [===>..........................] - ETA: 59s - loss: 7.7230 - accuracy: 0.4963
 3840/25000 [===>..........................] - ETA: 58s - loss: 7.7185 - accuracy: 0.4966
 3872/25000 [===>..........................] - ETA: 58s - loss: 7.7141 - accuracy: 0.4969
 3904/25000 [===>..........................] - ETA: 58s - loss: 7.7177 - accuracy: 0.4967
 3936/25000 [===>..........................] - ETA: 58s - loss: 7.7289 - accuracy: 0.4959
 3968/25000 [===>..........................] - ETA: 58s - loss: 7.7323 - accuracy: 0.4957
 4000/25000 [===>..........................] - ETA: 58s - loss: 7.7356 - accuracy: 0.4955
 4032/25000 [===>..........................] - ETA: 58s - loss: 7.7161 - accuracy: 0.4968
 4064/25000 [===>..........................] - ETA: 58s - loss: 7.7194 - accuracy: 0.4966
 4096/25000 [===>..........................] - ETA: 58s - loss: 7.7153 - accuracy: 0.4968
 4128/25000 [===>..........................] - ETA: 57s - loss: 7.7186 - accuracy: 0.4966
 4160/25000 [===>..........................] - ETA: 57s - loss: 7.7293 - accuracy: 0.4959
 4192/25000 [====>.........................] - ETA: 57s - loss: 7.7215 - accuracy: 0.4964
 4224/25000 [====>.........................] - ETA: 57s - loss: 7.7211 - accuracy: 0.4964
 4256/25000 [====>.........................] - ETA: 57s - loss: 7.7207 - accuracy: 0.4965
 4288/25000 [====>.........................] - ETA: 57s - loss: 7.7060 - accuracy: 0.4974
 4320/25000 [====>.........................] - ETA: 57s - loss: 7.7128 - accuracy: 0.4970
 4352/25000 [====>.........................] - ETA: 57s - loss: 7.7300 - accuracy: 0.4959
 4384/25000 [====>.........................] - ETA: 56s - loss: 7.7261 - accuracy: 0.4961
 4416/25000 [====>.........................] - ETA: 56s - loss: 7.7152 - accuracy: 0.4968
 4448/25000 [====>.........................] - ETA: 56s - loss: 7.7183 - accuracy: 0.4966
 4480/25000 [====>.........................] - ETA: 56s - loss: 7.7077 - accuracy: 0.4973
 4512/25000 [====>.........................] - ETA: 56s - loss: 7.7176 - accuracy: 0.4967
 4544/25000 [====>.........................] - ETA: 56s - loss: 7.7274 - accuracy: 0.4960
 4576/25000 [====>.........................] - ETA: 56s - loss: 7.7303 - accuracy: 0.4958
 4608/25000 [====>.........................] - ETA: 56s - loss: 7.7298 - accuracy: 0.4959
 4640/25000 [====>.........................] - ETA: 55s - loss: 7.7360 - accuracy: 0.4955
 4672/25000 [====>.........................] - ETA: 55s - loss: 7.7355 - accuracy: 0.4955
 4704/25000 [====>.........................] - ETA: 55s - loss: 7.7481 - accuracy: 0.4947
 4736/25000 [====>.........................] - ETA: 55s - loss: 7.7443 - accuracy: 0.4949
 4768/25000 [====>.........................] - ETA: 55s - loss: 7.7309 - accuracy: 0.4958
 4800/25000 [====>.........................] - ETA: 55s - loss: 7.7273 - accuracy: 0.4960
 4832/25000 [====>.........................] - ETA: 55s - loss: 7.7206 - accuracy: 0.4965
 4864/25000 [====>.........................] - ETA: 55s - loss: 7.7297 - accuracy: 0.4959
 4896/25000 [====>.........................] - ETA: 55s - loss: 7.7293 - accuracy: 0.4959
 4928/25000 [====>.........................] - ETA: 54s - loss: 7.7257 - accuracy: 0.4961
 4960/25000 [====>.........................] - ETA: 54s - loss: 7.7284 - accuracy: 0.4960
 4992/25000 [====>.........................] - ETA: 54s - loss: 7.7281 - accuracy: 0.4960
 5024/25000 [=====>........................] - ETA: 54s - loss: 7.7246 - accuracy: 0.4962
 5056/25000 [=====>........................] - ETA: 54s - loss: 7.7121 - accuracy: 0.4970
 5088/25000 [=====>........................] - ETA: 54s - loss: 7.7118 - accuracy: 0.4971
 5120/25000 [=====>........................] - ETA: 54s - loss: 7.7085 - accuracy: 0.4973
 5152/25000 [=====>........................] - ETA: 54s - loss: 7.6994 - accuracy: 0.4979
 5184/25000 [=====>........................] - ETA: 54s - loss: 7.6814 - accuracy: 0.4990
 5216/25000 [=====>........................] - ETA: 54s - loss: 7.6666 - accuracy: 0.5000
 5248/25000 [=====>........................] - ETA: 53s - loss: 7.6637 - accuracy: 0.5002
 5280/25000 [=====>........................] - ETA: 53s - loss: 7.6724 - accuracy: 0.4996
 5312/25000 [=====>........................] - ETA: 53s - loss: 7.6695 - accuracy: 0.4998
 5344/25000 [=====>........................] - ETA: 53s - loss: 7.6666 - accuracy: 0.5000
 5376/25000 [=====>........................] - ETA: 53s - loss: 7.6581 - accuracy: 0.5006
 5408/25000 [=====>........................] - ETA: 53s - loss: 7.6581 - accuracy: 0.5006
 5440/25000 [=====>........................] - ETA: 53s - loss: 7.6610 - accuracy: 0.5004
 5472/25000 [=====>........................] - ETA: 53s - loss: 7.6610 - accuracy: 0.5004
 5504/25000 [=====>........................] - ETA: 53s - loss: 7.6610 - accuracy: 0.5004
 5536/25000 [=====>........................] - ETA: 52s - loss: 7.6694 - accuracy: 0.4998
 5568/25000 [=====>........................] - ETA: 52s - loss: 7.6776 - accuracy: 0.4993
 5600/25000 [=====>........................] - ETA: 52s - loss: 7.6721 - accuracy: 0.4996
 5632/25000 [=====>........................] - ETA: 52s - loss: 7.6775 - accuracy: 0.4993
 5664/25000 [=====>........................] - ETA: 52s - loss: 7.6720 - accuracy: 0.4996
 5696/25000 [=====>........................] - ETA: 52s - loss: 7.6666 - accuracy: 0.5000
 5728/25000 [=====>........................] - ETA: 52s - loss: 7.6693 - accuracy: 0.4998
 5760/25000 [=====>........................] - ETA: 52s - loss: 7.6719 - accuracy: 0.4997
 5792/25000 [=====>........................] - ETA: 52s - loss: 7.6904 - accuracy: 0.4984
 5824/25000 [=====>........................] - ETA: 52s - loss: 7.6903 - accuracy: 0.4985
 5856/25000 [======>.......................] - ETA: 52s - loss: 7.6876 - accuracy: 0.4986
 5888/25000 [======>.......................] - ETA: 51s - loss: 7.6875 - accuracy: 0.4986
 5920/25000 [======>.......................] - ETA: 51s - loss: 7.6848 - accuracy: 0.4988
 5952/25000 [======>.......................] - ETA: 51s - loss: 7.6872 - accuracy: 0.4987
 5984/25000 [======>.......................] - ETA: 51s - loss: 7.6871 - accuracy: 0.4987
 6016/25000 [======>.......................] - ETA: 51s - loss: 7.6845 - accuracy: 0.4988
 6048/25000 [======>.......................] - ETA: 51s - loss: 7.6818 - accuracy: 0.4990
 6080/25000 [======>.......................] - ETA: 51s - loss: 7.6767 - accuracy: 0.4993
 6112/25000 [======>.......................] - ETA: 51s - loss: 7.6767 - accuracy: 0.4993
 6144/25000 [======>.......................] - ETA: 51s - loss: 7.6816 - accuracy: 0.4990
 6176/25000 [======>.......................] - ETA: 51s - loss: 7.6790 - accuracy: 0.4992
 6208/25000 [======>.......................] - ETA: 50s - loss: 7.6790 - accuracy: 0.4992
 6240/25000 [======>.......................] - ETA: 50s - loss: 7.6814 - accuracy: 0.4990
 6272/25000 [======>.......................] - ETA: 50s - loss: 7.6813 - accuracy: 0.4990
 6304/25000 [======>.......................] - ETA: 50s - loss: 7.6958 - accuracy: 0.4981
 6336/25000 [======>.......................] - ETA: 50s - loss: 7.6981 - accuracy: 0.4979
 6368/25000 [======>.......................] - ETA: 50s - loss: 7.7027 - accuracy: 0.4976
 6400/25000 [======>.......................] - ETA: 50s - loss: 7.7145 - accuracy: 0.4969
 6432/25000 [======>.......................] - ETA: 50s - loss: 7.7167 - accuracy: 0.4967
 6464/25000 [======>.......................] - ETA: 50s - loss: 7.7212 - accuracy: 0.4964
 6496/25000 [======>.......................] - ETA: 50s - loss: 7.7280 - accuracy: 0.4960
 6528/25000 [======>.......................] - ETA: 49s - loss: 7.7300 - accuracy: 0.4959
 6560/25000 [======>.......................] - ETA: 49s - loss: 7.7321 - accuracy: 0.4957
 6592/25000 [======>.......................] - ETA: 49s - loss: 7.7271 - accuracy: 0.4961
 6624/25000 [======>.......................] - ETA: 49s - loss: 7.7245 - accuracy: 0.4962
 6656/25000 [======>.......................] - ETA: 49s - loss: 7.7196 - accuracy: 0.4965
 6688/25000 [=======>......................] - ETA: 49s - loss: 7.7331 - accuracy: 0.4957
 6720/25000 [=======>......................] - ETA: 49s - loss: 7.7374 - accuracy: 0.4954
 6752/25000 [=======>......................] - ETA: 49s - loss: 7.7393 - accuracy: 0.4953
 6784/25000 [=======>......................] - ETA: 49s - loss: 7.7412 - accuracy: 0.4951
 6816/25000 [=======>......................] - ETA: 49s - loss: 7.7386 - accuracy: 0.4953
 6848/25000 [=======>......................] - ETA: 49s - loss: 7.7383 - accuracy: 0.4953
 6880/25000 [=======>......................] - ETA: 48s - loss: 7.7335 - accuracy: 0.4956
 6912/25000 [=======>......................] - ETA: 48s - loss: 7.7265 - accuracy: 0.4961
 6944/25000 [=======>......................] - ETA: 48s - loss: 7.7262 - accuracy: 0.4961
 6976/25000 [=======>......................] - ETA: 48s - loss: 7.7282 - accuracy: 0.4960
 7008/25000 [=======>......................] - ETA: 48s - loss: 7.7410 - accuracy: 0.4951
 7040/25000 [=======>......................] - ETA: 48s - loss: 7.7494 - accuracy: 0.4946
 7072/25000 [=======>......................] - ETA: 48s - loss: 7.7447 - accuracy: 0.4949
 7104/25000 [=======>......................] - ETA: 48s - loss: 7.7465 - accuracy: 0.4948
 7136/25000 [=======>......................] - ETA: 48s - loss: 7.7397 - accuracy: 0.4952
 7168/25000 [=======>......................] - ETA: 48s - loss: 7.7393 - accuracy: 0.4953
 7200/25000 [=======>......................] - ETA: 48s - loss: 7.7433 - accuracy: 0.4950
 7232/25000 [=======>......................] - ETA: 48s - loss: 7.7493 - accuracy: 0.4946
 7264/25000 [=======>......................] - ETA: 47s - loss: 7.7489 - accuracy: 0.4946
 7296/25000 [=======>......................] - ETA: 47s - loss: 7.7507 - accuracy: 0.4945
 7328/25000 [=======>......................] - ETA: 47s - loss: 7.7587 - accuracy: 0.4940
 7360/25000 [=======>......................] - ETA: 47s - loss: 7.7541 - accuracy: 0.4943
 7392/25000 [=======>......................] - ETA: 47s - loss: 7.7558 - accuracy: 0.4942
 7424/25000 [=======>......................] - ETA: 47s - loss: 7.7513 - accuracy: 0.4945
 7456/25000 [=======>......................] - ETA: 47s - loss: 7.7489 - accuracy: 0.4946
 7488/25000 [=======>......................] - ETA: 47s - loss: 7.7424 - accuracy: 0.4951
 7520/25000 [========>.....................] - ETA: 47s - loss: 7.7339 - accuracy: 0.4956
 7552/25000 [========>.....................] - ETA: 47s - loss: 7.7296 - accuracy: 0.4959
 7584/25000 [========>.....................] - ETA: 47s - loss: 7.7253 - accuracy: 0.4962
 7616/25000 [========>.....................] - ETA: 46s - loss: 7.7331 - accuracy: 0.4957
 7648/25000 [========>.....................] - ETA: 46s - loss: 7.7348 - accuracy: 0.4956
 7680/25000 [========>.....................] - ETA: 46s - loss: 7.7365 - accuracy: 0.4954
 7712/25000 [========>.....................] - ETA: 46s - loss: 7.7342 - accuracy: 0.4956
 7744/25000 [========>.....................] - ETA: 46s - loss: 7.7399 - accuracy: 0.4952
 7776/25000 [========>.....................] - ETA: 46s - loss: 7.7455 - accuracy: 0.4949
 7808/25000 [========>.....................] - ETA: 46s - loss: 7.7511 - accuracy: 0.4945
 7840/25000 [========>.....................] - ETA: 46s - loss: 7.7507 - accuracy: 0.4945
 7872/25000 [========>.....................] - ETA: 46s - loss: 7.7562 - accuracy: 0.4942
 7904/25000 [========>.....................] - ETA: 46s - loss: 7.7481 - accuracy: 0.4947
 7936/25000 [========>.....................] - ETA: 45s - loss: 7.7478 - accuracy: 0.4947
 7968/25000 [========>.....................] - ETA: 45s - loss: 7.7494 - accuracy: 0.4946
 8000/25000 [========>.....................] - ETA: 45s - loss: 7.7414 - accuracy: 0.4951
 8032/25000 [========>.....................] - ETA: 45s - loss: 7.7411 - accuracy: 0.4951
 8064/25000 [========>.....................] - ETA: 45s - loss: 7.7427 - accuracy: 0.4950
 8096/25000 [========>.....................] - ETA: 45s - loss: 7.7386 - accuracy: 0.4953
 8128/25000 [========>.....................] - ETA: 45s - loss: 7.7326 - accuracy: 0.4957
 8160/25000 [========>.....................] - ETA: 45s - loss: 7.7418 - accuracy: 0.4951
 8192/25000 [========>.....................] - ETA: 45s - loss: 7.7471 - accuracy: 0.4948
 8224/25000 [========>.....................] - ETA: 45s - loss: 7.7412 - accuracy: 0.4951
 8256/25000 [========>.....................] - ETA: 45s - loss: 7.7502 - accuracy: 0.4945
 8288/25000 [========>.....................] - ETA: 45s - loss: 7.7480 - accuracy: 0.4947
 8320/25000 [========>.....................] - ETA: 44s - loss: 7.7532 - accuracy: 0.4944
 8352/25000 [=========>....................] - ETA: 44s - loss: 7.7492 - accuracy: 0.4946
 8384/25000 [=========>....................] - ETA: 44s - loss: 7.7434 - accuracy: 0.4950
 8416/25000 [=========>....................] - ETA: 44s - loss: 7.7395 - accuracy: 0.4952
 8448/25000 [=========>....................] - ETA: 44s - loss: 7.7374 - accuracy: 0.4954
 8480/25000 [=========>....................] - ETA: 44s - loss: 7.7408 - accuracy: 0.4952
 8512/25000 [=========>....................] - ETA: 44s - loss: 7.7369 - accuracy: 0.4954
 8544/25000 [=========>....................] - ETA: 44s - loss: 7.7438 - accuracy: 0.4950
 8576/25000 [=========>....................] - ETA: 44s - loss: 7.7346 - accuracy: 0.4956
 8608/25000 [=========>....................] - ETA: 44s - loss: 7.7272 - accuracy: 0.4961
 8640/25000 [=========>....................] - ETA: 44s - loss: 7.7199 - accuracy: 0.4965
 8672/25000 [=========>....................] - ETA: 43s - loss: 7.7197 - accuracy: 0.4965
 8704/25000 [=========>....................] - ETA: 43s - loss: 7.7195 - accuracy: 0.4966
 8736/25000 [=========>....................] - ETA: 43s - loss: 7.7175 - accuracy: 0.4967
 8768/25000 [=========>....................] - ETA: 43s - loss: 7.7191 - accuracy: 0.4966
 8800/25000 [=========>....................] - ETA: 43s - loss: 7.7189 - accuracy: 0.4966
 8832/25000 [=========>....................] - ETA: 43s - loss: 7.7135 - accuracy: 0.4969
 8864/25000 [=========>....................] - ETA: 43s - loss: 7.7099 - accuracy: 0.4972
 8896/25000 [=========>....................] - ETA: 43s - loss: 7.7028 - accuracy: 0.4976
 8928/25000 [=========>....................] - ETA: 43s - loss: 7.7010 - accuracy: 0.4978
 8960/25000 [=========>....................] - ETA: 43s - loss: 7.6974 - accuracy: 0.4980
 8992/25000 [=========>....................] - ETA: 43s - loss: 7.6990 - accuracy: 0.4979
 9024/25000 [=========>....................] - ETA: 42s - loss: 7.7023 - accuracy: 0.4977
 9056/25000 [=========>....................] - ETA: 42s - loss: 7.7022 - accuracy: 0.4977
 9088/25000 [=========>....................] - ETA: 42s - loss: 7.7054 - accuracy: 0.4975
 9120/25000 [=========>....................] - ETA: 42s - loss: 7.7103 - accuracy: 0.4971
 9152/25000 [=========>....................] - ETA: 42s - loss: 7.7119 - accuracy: 0.4970
 9184/25000 [==========>...................] - ETA: 42s - loss: 7.7067 - accuracy: 0.4974
 9216/25000 [==========>...................] - ETA: 42s - loss: 7.7082 - accuracy: 0.4973
 9248/25000 [==========>...................] - ETA: 42s - loss: 7.7081 - accuracy: 0.4973
 9280/25000 [==========>...................] - ETA: 42s - loss: 7.7046 - accuracy: 0.4975
 9312/25000 [==========>...................] - ETA: 42s - loss: 7.7012 - accuracy: 0.4977
 9344/25000 [==========>...................] - ETA: 42s - loss: 7.7027 - accuracy: 0.4976
 9376/25000 [==========>...................] - ETA: 41s - loss: 7.7108 - accuracy: 0.4971
 9408/25000 [==========>...................] - ETA: 41s - loss: 7.7155 - accuracy: 0.4968
 9440/25000 [==========>...................] - ETA: 41s - loss: 7.7105 - accuracy: 0.4971
 9472/25000 [==========>...................] - ETA: 41s - loss: 7.7087 - accuracy: 0.4973
 9504/25000 [==========>...................] - ETA: 41s - loss: 7.7037 - accuracy: 0.4976
 9536/25000 [==========>...................] - ETA: 41s - loss: 7.7116 - accuracy: 0.4971
 9568/25000 [==========>...................] - ETA: 41s - loss: 7.7067 - accuracy: 0.4974
 9600/25000 [==========>...................] - ETA: 41s - loss: 7.7018 - accuracy: 0.4977
 9632/25000 [==========>...................] - ETA: 41s - loss: 7.7048 - accuracy: 0.4975
 9664/25000 [==========>...................] - ETA: 41s - loss: 7.7015 - accuracy: 0.4977
 9696/25000 [==========>...................] - ETA: 41s - loss: 7.7062 - accuracy: 0.4974
 9728/25000 [==========>...................] - ETA: 40s - loss: 7.7171 - accuracy: 0.4967
 9760/25000 [==========>...................] - ETA: 40s - loss: 7.7122 - accuracy: 0.4970
 9792/25000 [==========>...................] - ETA: 40s - loss: 7.7105 - accuracy: 0.4971
 9824/25000 [==========>...................] - ETA: 40s - loss: 7.7072 - accuracy: 0.4974
 9856/25000 [==========>...................] - ETA: 40s - loss: 7.7117 - accuracy: 0.4971
 9888/25000 [==========>...................] - ETA: 40s - loss: 7.7178 - accuracy: 0.4967
 9920/25000 [==========>...................] - ETA: 40s - loss: 7.7207 - accuracy: 0.4965
 9952/25000 [==========>...................] - ETA: 40s - loss: 7.7221 - accuracy: 0.4964
 9984/25000 [==========>...................] - ETA: 40s - loss: 7.7234 - accuracy: 0.4963
10016/25000 [===========>..................] - ETA: 40s - loss: 7.7263 - accuracy: 0.4961
10048/25000 [===========>..................] - ETA: 40s - loss: 7.7292 - accuracy: 0.4959
10080/25000 [===========>..................] - ETA: 40s - loss: 7.7305 - accuracy: 0.4958
10112/25000 [===========>..................] - ETA: 39s - loss: 7.7349 - accuracy: 0.4955
10144/25000 [===========>..................] - ETA: 39s - loss: 7.7331 - accuracy: 0.4957
10176/25000 [===========>..................] - ETA: 39s - loss: 7.7329 - accuracy: 0.4957
10208/25000 [===========>..................] - ETA: 39s - loss: 7.7282 - accuracy: 0.4960
10240/25000 [===========>..................] - ETA: 39s - loss: 7.7280 - accuracy: 0.4960
10272/25000 [===========>..................] - ETA: 39s - loss: 7.7293 - accuracy: 0.4959
10304/25000 [===========>..................] - ETA: 39s - loss: 7.7247 - accuracy: 0.4962
10336/25000 [===========>..................] - ETA: 39s - loss: 7.7245 - accuracy: 0.4962
10368/25000 [===========>..................] - ETA: 39s - loss: 7.7243 - accuracy: 0.4962
10400/25000 [===========>..................] - ETA: 39s - loss: 7.7212 - accuracy: 0.4964
10432/25000 [===========>..................] - ETA: 39s - loss: 7.7166 - accuracy: 0.4967
10464/25000 [===========>..................] - ETA: 38s - loss: 7.7208 - accuracy: 0.4965
10496/25000 [===========>..................] - ETA: 38s - loss: 7.7192 - accuracy: 0.4966
10528/25000 [===========>..................] - ETA: 38s - loss: 7.7234 - accuracy: 0.4963
10560/25000 [===========>..................] - ETA: 38s - loss: 7.7232 - accuracy: 0.4963
10592/25000 [===========>..................] - ETA: 38s - loss: 7.7231 - accuracy: 0.4963
10624/25000 [===========>..................] - ETA: 38s - loss: 7.7258 - accuracy: 0.4961
10656/25000 [===========>..................] - ETA: 38s - loss: 7.7155 - accuracy: 0.4968
10688/25000 [===========>..................] - ETA: 38s - loss: 7.7154 - accuracy: 0.4968
10720/25000 [===========>..................] - ETA: 38s - loss: 7.7224 - accuracy: 0.4964
10752/25000 [===========>..................] - ETA: 38s - loss: 7.7294 - accuracy: 0.4959
10784/25000 [===========>..................] - ETA: 38s - loss: 7.7334 - accuracy: 0.4956
10816/25000 [===========>..................] - ETA: 37s - loss: 7.7361 - accuracy: 0.4955
10848/25000 [============>.................] - ETA: 37s - loss: 7.7331 - accuracy: 0.4957
10880/25000 [============>.................] - ETA: 37s - loss: 7.7286 - accuracy: 0.4960
10912/25000 [============>.................] - ETA: 37s - loss: 7.7242 - accuracy: 0.4962
10944/25000 [============>.................] - ETA: 37s - loss: 7.7199 - accuracy: 0.4965
10976/25000 [============>.................] - ETA: 37s - loss: 7.7183 - accuracy: 0.4966
11008/25000 [============>.................] - ETA: 37s - loss: 7.7168 - accuracy: 0.4967
11040/25000 [============>.................] - ETA: 37s - loss: 7.7222 - accuracy: 0.4964
11072/25000 [============>.................] - ETA: 37s - loss: 7.7248 - accuracy: 0.4962
11104/25000 [============>.................] - ETA: 37s - loss: 7.7219 - accuracy: 0.4964
11136/25000 [============>.................] - ETA: 37s - loss: 7.7231 - accuracy: 0.4963
11168/25000 [============>.................] - ETA: 37s - loss: 7.7188 - accuracy: 0.4966
11200/25000 [============>.................] - ETA: 36s - loss: 7.7241 - accuracy: 0.4963
11232/25000 [============>.................] - ETA: 36s - loss: 7.7171 - accuracy: 0.4967
11264/25000 [============>.................] - ETA: 36s - loss: 7.7197 - accuracy: 0.4965
11296/25000 [============>.................] - ETA: 36s - loss: 7.7155 - accuracy: 0.4968
11328/25000 [============>.................] - ETA: 36s - loss: 7.7194 - accuracy: 0.4966
11360/25000 [============>.................] - ETA: 36s - loss: 7.7179 - accuracy: 0.4967
11392/25000 [============>.................] - ETA: 36s - loss: 7.7110 - accuracy: 0.4971
11424/25000 [============>.................] - ETA: 36s - loss: 7.7096 - accuracy: 0.4972
11456/25000 [============>.................] - ETA: 36s - loss: 7.7041 - accuracy: 0.4976
11488/25000 [============>.................] - ETA: 36s - loss: 7.7120 - accuracy: 0.4970
11520/25000 [============>.................] - ETA: 36s - loss: 7.7105 - accuracy: 0.4971
11552/25000 [============>.................] - ETA: 35s - loss: 7.7091 - accuracy: 0.4972
11584/25000 [============>.................] - ETA: 35s - loss: 7.7103 - accuracy: 0.4972
11616/25000 [============>.................] - ETA: 35s - loss: 7.7128 - accuracy: 0.4970
11648/25000 [============>.................] - ETA: 35s - loss: 7.7101 - accuracy: 0.4972
11680/25000 [=============>................] - ETA: 35s - loss: 7.7086 - accuracy: 0.4973
11712/25000 [=============>................] - ETA: 35s - loss: 7.7059 - accuracy: 0.4974
11744/25000 [=============>................] - ETA: 35s - loss: 7.7071 - accuracy: 0.4974
11776/25000 [=============>................] - ETA: 35s - loss: 7.7057 - accuracy: 0.4975
11808/25000 [=============>................] - ETA: 35s - loss: 7.7056 - accuracy: 0.4975
11840/25000 [=============>................] - ETA: 35s - loss: 7.7042 - accuracy: 0.4976
11872/25000 [=============>................] - ETA: 35s - loss: 7.7092 - accuracy: 0.4972
11904/25000 [=============>................] - ETA: 34s - loss: 7.7091 - accuracy: 0.4972
11936/25000 [=============>................] - ETA: 34s - loss: 7.7167 - accuracy: 0.4967
11968/25000 [=============>................] - ETA: 34s - loss: 7.7230 - accuracy: 0.4963
12000/25000 [=============>................] - ETA: 34s - loss: 7.7228 - accuracy: 0.4963
12032/25000 [=============>................] - ETA: 34s - loss: 7.7240 - accuracy: 0.4963
12064/25000 [=============>................] - ETA: 34s - loss: 7.7251 - accuracy: 0.4962
12096/25000 [=============>................] - ETA: 34s - loss: 7.7211 - accuracy: 0.4964
12128/25000 [=============>................] - ETA: 34s - loss: 7.7222 - accuracy: 0.4964
12160/25000 [=============>................] - ETA: 34s - loss: 7.7246 - accuracy: 0.4962
12192/25000 [=============>................] - ETA: 34s - loss: 7.7270 - accuracy: 0.4961
12224/25000 [=============>................] - ETA: 34s - loss: 7.7243 - accuracy: 0.4962
12256/25000 [=============>................] - ETA: 33s - loss: 7.7254 - accuracy: 0.4962
12288/25000 [=============>................] - ETA: 33s - loss: 7.7203 - accuracy: 0.4965
12320/25000 [=============>................] - ETA: 33s - loss: 7.7226 - accuracy: 0.4963
12352/25000 [=============>................] - ETA: 33s - loss: 7.7250 - accuracy: 0.4962
12384/25000 [=============>................] - ETA: 33s - loss: 7.7223 - accuracy: 0.4964
12416/25000 [=============>................] - ETA: 33s - loss: 7.7173 - accuracy: 0.4967
12448/25000 [=============>................] - ETA: 33s - loss: 7.7184 - accuracy: 0.4966
12480/25000 [=============>................] - ETA: 33s - loss: 7.7256 - accuracy: 0.4962
12512/25000 [==============>...............] - ETA: 33s - loss: 7.7254 - accuracy: 0.4962
12544/25000 [==============>...............] - ETA: 33s - loss: 7.7302 - accuracy: 0.4959
12576/25000 [==============>...............] - ETA: 33s - loss: 7.7300 - accuracy: 0.4959
12608/25000 [==============>...............] - ETA: 32s - loss: 7.7262 - accuracy: 0.4961
12640/25000 [==============>...............] - ETA: 32s - loss: 7.7212 - accuracy: 0.4964
12672/25000 [==============>...............] - ETA: 32s - loss: 7.7150 - accuracy: 0.4968
12704/25000 [==============>...............] - ETA: 32s - loss: 7.7173 - accuracy: 0.4967
12736/25000 [==============>...............] - ETA: 32s - loss: 7.7256 - accuracy: 0.4962
12768/25000 [==============>...............] - ETA: 32s - loss: 7.7267 - accuracy: 0.4961
12800/25000 [==============>...............] - ETA: 32s - loss: 7.7217 - accuracy: 0.4964
12832/25000 [==============>...............] - ETA: 32s - loss: 7.7216 - accuracy: 0.4964
12864/25000 [==============>...............] - ETA: 32s - loss: 7.7214 - accuracy: 0.4964
12896/25000 [==============>...............] - ETA: 32s - loss: 7.7189 - accuracy: 0.4966
12928/25000 [==============>...............] - ETA: 32s - loss: 7.7164 - accuracy: 0.4968
12960/25000 [==============>...............] - ETA: 31s - loss: 7.7139 - accuracy: 0.4969
12992/25000 [==============>...............] - ETA: 31s - loss: 7.7079 - accuracy: 0.4973
13024/25000 [==============>...............] - ETA: 31s - loss: 7.7114 - accuracy: 0.4971
13056/25000 [==============>...............] - ETA: 31s - loss: 7.7124 - accuracy: 0.4970
13088/25000 [==============>...............] - ETA: 31s - loss: 7.7100 - accuracy: 0.4972
13120/25000 [==============>...............] - ETA: 31s - loss: 7.7134 - accuracy: 0.4970
13152/25000 [==============>...............] - ETA: 31s - loss: 7.7156 - accuracy: 0.4968
13184/25000 [==============>...............] - ETA: 31s - loss: 7.7097 - accuracy: 0.4972
13216/25000 [==============>...............] - ETA: 31s - loss: 7.7049 - accuracy: 0.4975
13248/25000 [==============>...............] - ETA: 31s - loss: 7.7060 - accuracy: 0.4974
13280/25000 [==============>...............] - ETA: 31s - loss: 7.7047 - accuracy: 0.4975
13312/25000 [==============>...............] - ETA: 31s - loss: 7.7058 - accuracy: 0.4974
13344/25000 [===============>..............] - ETA: 30s - loss: 7.7022 - accuracy: 0.4977
13376/25000 [===============>..............] - ETA: 30s - loss: 7.7090 - accuracy: 0.4972
13408/25000 [===============>..............] - ETA: 30s - loss: 7.7066 - accuracy: 0.4974
13440/25000 [===============>..............] - ETA: 30s - loss: 7.7043 - accuracy: 0.4975
13472/25000 [===============>..............] - ETA: 30s - loss: 7.7042 - accuracy: 0.4976
13504/25000 [===============>..............] - ETA: 30s - loss: 7.7007 - accuracy: 0.4978
13536/25000 [===============>..............] - ETA: 30s - loss: 7.6983 - accuracy: 0.4979
13568/25000 [===============>..............] - ETA: 30s - loss: 7.6960 - accuracy: 0.4981
13600/25000 [===============>..............] - ETA: 30s - loss: 7.6926 - accuracy: 0.4983
13632/25000 [===============>..............] - ETA: 30s - loss: 7.6981 - accuracy: 0.4979
13664/25000 [===============>..............] - ETA: 30s - loss: 7.6947 - accuracy: 0.4982
13696/25000 [===============>..............] - ETA: 30s - loss: 7.6946 - accuracy: 0.4982
13728/25000 [===============>..............] - ETA: 29s - loss: 7.6923 - accuracy: 0.4983
13760/25000 [===============>..............] - ETA: 29s - loss: 7.6956 - accuracy: 0.4981
13792/25000 [===============>..............] - ETA: 29s - loss: 7.6955 - accuracy: 0.4981
13824/25000 [===============>..............] - ETA: 29s - loss: 7.6943 - accuracy: 0.4982
13856/25000 [===============>..............] - ETA: 29s - loss: 7.6954 - accuracy: 0.4981
13888/25000 [===============>..............] - ETA: 29s - loss: 7.6953 - accuracy: 0.4981
13920/25000 [===============>..............] - ETA: 29s - loss: 7.6953 - accuracy: 0.4981
13952/25000 [===============>..............] - ETA: 29s - loss: 7.6930 - accuracy: 0.4983
13984/25000 [===============>..............] - ETA: 29s - loss: 7.6995 - accuracy: 0.4979
14016/25000 [===============>..............] - ETA: 29s - loss: 7.6983 - accuracy: 0.4979
14048/25000 [===============>..............] - ETA: 29s - loss: 7.7015 - accuracy: 0.4977
14080/25000 [===============>..............] - ETA: 29s - loss: 7.7058 - accuracy: 0.4974
14112/25000 [===============>..............] - ETA: 28s - loss: 7.7112 - accuracy: 0.4971
14144/25000 [===============>..............] - ETA: 28s - loss: 7.7111 - accuracy: 0.4971
14176/25000 [================>.............] - ETA: 28s - loss: 7.7099 - accuracy: 0.4972
14208/25000 [================>.............] - ETA: 28s - loss: 7.7098 - accuracy: 0.4972
14240/25000 [================>.............] - ETA: 28s - loss: 7.7118 - accuracy: 0.4971
14272/25000 [================>.............] - ETA: 28s - loss: 7.7182 - accuracy: 0.4966
14304/25000 [================>.............] - ETA: 28s - loss: 7.7149 - accuracy: 0.4969
14336/25000 [================>.............] - ETA: 28s - loss: 7.7180 - accuracy: 0.4967
14368/25000 [================>.............] - ETA: 28s - loss: 7.7200 - accuracy: 0.4965
14400/25000 [================>.............] - ETA: 28s - loss: 7.7188 - accuracy: 0.4966
14432/25000 [================>.............] - ETA: 28s - loss: 7.7176 - accuracy: 0.4967
14464/25000 [================>.............] - ETA: 28s - loss: 7.7154 - accuracy: 0.4968
14496/25000 [================>.............] - ETA: 27s - loss: 7.7121 - accuracy: 0.4970
14528/25000 [================>.............] - ETA: 27s - loss: 7.7120 - accuracy: 0.4970
14560/25000 [================>.............] - ETA: 27s - loss: 7.7087 - accuracy: 0.4973
14592/25000 [================>.............] - ETA: 27s - loss: 7.7108 - accuracy: 0.4971
14624/25000 [================>.............] - ETA: 27s - loss: 7.7065 - accuracy: 0.4974
14656/25000 [================>.............] - ETA: 27s - loss: 7.7064 - accuracy: 0.4974
14688/25000 [================>.............] - ETA: 27s - loss: 7.7021 - accuracy: 0.4977
14720/25000 [================>.............] - ETA: 27s - loss: 7.6989 - accuracy: 0.4979
14752/25000 [================>.............] - ETA: 27s - loss: 7.7020 - accuracy: 0.4977
14784/25000 [================>.............] - ETA: 27s - loss: 7.6998 - accuracy: 0.4978
14816/25000 [================>.............] - ETA: 27s - loss: 7.6956 - accuracy: 0.4981
14848/25000 [================>.............] - ETA: 26s - loss: 7.7017 - accuracy: 0.4977
14880/25000 [================>.............] - ETA: 26s - loss: 7.7027 - accuracy: 0.4976
14912/25000 [================>.............] - ETA: 26s - loss: 7.7036 - accuracy: 0.4976
14944/25000 [================>.............] - ETA: 26s - loss: 7.7056 - accuracy: 0.4975
14976/25000 [================>.............] - ETA: 26s - loss: 7.7096 - accuracy: 0.4972
15008/25000 [=================>............] - ETA: 26s - loss: 7.7136 - accuracy: 0.4969
15040/25000 [=================>............] - ETA: 26s - loss: 7.7166 - accuracy: 0.4967
15072/25000 [=================>............] - ETA: 26s - loss: 7.7144 - accuracy: 0.4969
15104/25000 [=================>............] - ETA: 26s - loss: 7.7113 - accuracy: 0.4971
15136/25000 [=================>............] - ETA: 26s - loss: 7.7132 - accuracy: 0.4970
15168/25000 [=================>............] - ETA: 26s - loss: 7.7101 - accuracy: 0.4972
15200/25000 [=================>............] - ETA: 26s - loss: 7.7120 - accuracy: 0.4970
15232/25000 [=================>............] - ETA: 25s - loss: 7.7129 - accuracy: 0.4970
15264/25000 [=================>............] - ETA: 25s - loss: 7.7138 - accuracy: 0.4969
15296/25000 [=================>............] - ETA: 25s - loss: 7.7117 - accuracy: 0.4971
15328/25000 [=================>............] - ETA: 25s - loss: 7.7096 - accuracy: 0.4972
15360/25000 [=================>............] - ETA: 25s - loss: 7.7135 - accuracy: 0.4969
15392/25000 [=================>............] - ETA: 25s - loss: 7.7124 - accuracy: 0.4970
15424/25000 [=================>............] - ETA: 25s - loss: 7.7104 - accuracy: 0.4971
15456/25000 [=================>............] - ETA: 25s - loss: 7.7152 - accuracy: 0.4968
15488/25000 [=================>............] - ETA: 25s - loss: 7.7181 - accuracy: 0.4966
15520/25000 [=================>............] - ETA: 25s - loss: 7.7170 - accuracy: 0.4967
15552/25000 [=================>............] - ETA: 25s - loss: 7.7189 - accuracy: 0.4966
15584/25000 [=================>............] - ETA: 25s - loss: 7.7168 - accuracy: 0.4967
15616/25000 [=================>............] - ETA: 24s - loss: 7.7177 - accuracy: 0.4967
15648/25000 [=================>............] - ETA: 24s - loss: 7.7146 - accuracy: 0.4969
15680/25000 [=================>............] - ETA: 24s - loss: 7.7155 - accuracy: 0.4968
15712/25000 [=================>............] - ETA: 24s - loss: 7.7125 - accuracy: 0.4970
15744/25000 [=================>............] - ETA: 24s - loss: 7.7134 - accuracy: 0.4970
15776/25000 [=================>............] - ETA: 24s - loss: 7.7142 - accuracy: 0.4969
15808/25000 [=================>............] - ETA: 24s - loss: 7.7112 - accuracy: 0.4971
15840/25000 [==================>...........] - ETA: 24s - loss: 7.7111 - accuracy: 0.4971
15872/25000 [==================>...........] - ETA: 24s - loss: 7.7053 - accuracy: 0.4975
15904/25000 [==================>...........] - ETA: 24s - loss: 7.7052 - accuracy: 0.4975
15936/25000 [==================>...........] - ETA: 24s - loss: 7.7003 - accuracy: 0.4978
15968/25000 [==================>...........] - ETA: 23s - loss: 7.7002 - accuracy: 0.4978
16000/25000 [==================>...........] - ETA: 23s - loss: 7.6963 - accuracy: 0.4981
16032/25000 [==================>...........] - ETA: 23s - loss: 7.6953 - accuracy: 0.4981
16064/25000 [==================>...........] - ETA: 23s - loss: 7.6943 - accuracy: 0.4982
16096/25000 [==================>...........] - ETA: 23s - loss: 7.6923 - accuracy: 0.4983
16128/25000 [==================>...........] - ETA: 23s - loss: 7.6961 - accuracy: 0.4981
16160/25000 [==================>...........] - ETA: 23s - loss: 7.6989 - accuracy: 0.4979
16192/25000 [==================>...........] - ETA: 23s - loss: 7.6969 - accuracy: 0.4980
16224/25000 [==================>...........] - ETA: 23s - loss: 7.6959 - accuracy: 0.4981
16256/25000 [==================>...........] - ETA: 23s - loss: 7.6977 - accuracy: 0.4980
16288/25000 [==================>...........] - ETA: 23s - loss: 7.6949 - accuracy: 0.4982
16320/25000 [==================>...........] - ETA: 23s - loss: 7.6967 - accuracy: 0.4980
16352/25000 [==================>...........] - ETA: 22s - loss: 7.6966 - accuracy: 0.4980
16384/25000 [==================>...........] - ETA: 22s - loss: 7.7022 - accuracy: 0.4977
16416/25000 [==================>...........] - ETA: 22s - loss: 7.7030 - accuracy: 0.4976
16448/25000 [==================>...........] - ETA: 22s - loss: 7.7067 - accuracy: 0.4974
16480/25000 [==================>...........] - ETA: 22s - loss: 7.7048 - accuracy: 0.4975
16512/25000 [==================>...........] - ETA: 22s - loss: 7.7010 - accuracy: 0.4978
16544/25000 [==================>...........] - ETA: 22s - loss: 7.6991 - accuracy: 0.4979
16576/25000 [==================>...........] - ETA: 22s - loss: 7.6999 - accuracy: 0.4978
16608/25000 [==================>...........] - ETA: 22s - loss: 7.7026 - accuracy: 0.4977
16640/25000 [==================>...........] - ETA: 22s - loss: 7.7007 - accuracy: 0.4978
16672/25000 [===================>..........] - ETA: 22s - loss: 7.6997 - accuracy: 0.4978
16704/25000 [===================>..........] - ETA: 22s - loss: 7.7015 - accuracy: 0.4977
16736/25000 [===================>..........] - ETA: 21s - loss: 7.6978 - accuracy: 0.4980
16768/25000 [===================>..........] - ETA: 21s - loss: 7.6959 - accuracy: 0.4981
16800/25000 [===================>..........] - ETA: 21s - loss: 7.6977 - accuracy: 0.4980
16832/25000 [===================>..........] - ETA: 21s - loss: 7.6994 - accuracy: 0.4979
16864/25000 [===================>..........] - ETA: 21s - loss: 7.7021 - accuracy: 0.4977
16896/25000 [===================>..........] - ETA: 21s - loss: 7.7029 - accuracy: 0.4976
16928/25000 [===================>..........] - ETA: 21s - loss: 7.6965 - accuracy: 0.4981
16960/25000 [===================>..........] - ETA: 21s - loss: 7.6919 - accuracy: 0.4983
16992/25000 [===================>..........] - ETA: 21s - loss: 7.6946 - accuracy: 0.4982
17024/25000 [===================>..........] - ETA: 21s - loss: 7.6945 - accuracy: 0.4982
17056/25000 [===================>..........] - ETA: 21s - loss: 7.6936 - accuracy: 0.4982
17088/25000 [===================>..........] - ETA: 20s - loss: 7.6908 - accuracy: 0.4984
17120/25000 [===================>..........] - ETA: 20s - loss: 7.6899 - accuracy: 0.4985
17152/25000 [===================>..........] - ETA: 20s - loss: 7.6872 - accuracy: 0.4987
17184/25000 [===================>..........] - ETA: 20s - loss: 7.6818 - accuracy: 0.4990
17216/25000 [===================>..........] - ETA: 20s - loss: 7.6782 - accuracy: 0.4992
17248/25000 [===================>..........] - ETA: 20s - loss: 7.6764 - accuracy: 0.4994
17280/25000 [===================>..........] - ETA: 20s - loss: 7.6737 - accuracy: 0.4995
17312/25000 [===================>..........] - ETA: 20s - loss: 7.6710 - accuracy: 0.4997
17344/25000 [===================>..........] - ETA: 20s - loss: 7.6755 - accuracy: 0.4994
17376/25000 [===================>..........] - ETA: 20s - loss: 7.6754 - accuracy: 0.4994
17408/25000 [===================>..........] - ETA: 20s - loss: 7.6728 - accuracy: 0.4996
17440/25000 [===================>..........] - ETA: 20s - loss: 7.6763 - accuracy: 0.4994
17472/25000 [===================>..........] - ETA: 19s - loss: 7.6728 - accuracy: 0.4996
17504/25000 [====================>.........] - ETA: 19s - loss: 7.6710 - accuracy: 0.4997
17536/25000 [====================>.........] - ETA: 19s - loss: 7.6701 - accuracy: 0.4998
17568/25000 [====================>.........] - ETA: 19s - loss: 7.6719 - accuracy: 0.4997
17600/25000 [====================>.........] - ETA: 19s - loss: 7.6736 - accuracy: 0.4995
17632/25000 [====================>.........] - ETA: 19s - loss: 7.6710 - accuracy: 0.4997
17664/25000 [====================>.........] - ETA: 19s - loss: 7.6692 - accuracy: 0.4998
17696/25000 [====================>.........] - ETA: 19s - loss: 7.6727 - accuracy: 0.4996
17728/25000 [====================>.........] - ETA: 19s - loss: 7.6718 - accuracy: 0.4997
17760/25000 [====================>.........] - ETA: 19s - loss: 7.6701 - accuracy: 0.4998
17792/25000 [====================>.........] - ETA: 19s - loss: 7.6692 - accuracy: 0.4998
17824/25000 [====================>.........] - ETA: 19s - loss: 7.6726 - accuracy: 0.4996
17856/25000 [====================>.........] - ETA: 18s - loss: 7.6735 - accuracy: 0.4996
17888/25000 [====================>.........] - ETA: 18s - loss: 7.6726 - accuracy: 0.4996
17920/25000 [====================>.........] - ETA: 18s - loss: 7.6709 - accuracy: 0.4997
17952/25000 [====================>.........] - ETA: 18s - loss: 7.6726 - accuracy: 0.4996
17984/25000 [====================>.........] - ETA: 18s - loss: 7.6709 - accuracy: 0.4997
18016/25000 [====================>.........] - ETA: 18s - loss: 7.6683 - accuracy: 0.4999
18048/25000 [====================>.........] - ETA: 18s - loss: 7.6709 - accuracy: 0.4997
18080/25000 [====================>.........] - ETA: 18s - loss: 7.6709 - accuracy: 0.4997
18112/25000 [====================>.........] - ETA: 18s - loss: 7.6742 - accuracy: 0.4995
18144/25000 [====================>.........] - ETA: 18s - loss: 7.6785 - accuracy: 0.4992
18176/25000 [====================>.........] - ETA: 18s - loss: 7.6826 - accuracy: 0.4990
18208/25000 [====================>.........] - ETA: 18s - loss: 7.6809 - accuracy: 0.4991
18240/25000 [====================>.........] - ETA: 17s - loss: 7.6775 - accuracy: 0.4993
18272/25000 [====================>.........] - ETA: 17s - loss: 7.6775 - accuracy: 0.4993
18304/25000 [====================>.........] - ETA: 17s - loss: 7.6792 - accuracy: 0.4992
18336/25000 [=====================>........] - ETA: 17s - loss: 7.6808 - accuracy: 0.4991
18368/25000 [=====================>........] - ETA: 17s - loss: 7.6850 - accuracy: 0.4988
18400/25000 [=====================>........] - ETA: 17s - loss: 7.6858 - accuracy: 0.4988
18432/25000 [=====================>........] - ETA: 17s - loss: 7.6841 - accuracy: 0.4989
18464/25000 [=====================>........] - ETA: 17s - loss: 7.6857 - accuracy: 0.4988
18496/25000 [=====================>........] - ETA: 17s - loss: 7.6882 - accuracy: 0.4986
18528/25000 [=====================>........] - ETA: 17s - loss: 7.6848 - accuracy: 0.4988
18560/25000 [=====================>........] - ETA: 17s - loss: 7.6864 - accuracy: 0.4987
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6839 - accuracy: 0.4989
18624/25000 [=====================>........] - ETA: 16s - loss: 7.6856 - accuracy: 0.4988
18656/25000 [=====================>........] - ETA: 16s - loss: 7.6880 - accuracy: 0.4986
18688/25000 [=====================>........] - ETA: 16s - loss: 7.6880 - accuracy: 0.4986
18720/25000 [=====================>........] - ETA: 16s - loss: 7.6871 - accuracy: 0.4987
18752/25000 [=====================>........] - ETA: 16s - loss: 7.6830 - accuracy: 0.4989
18784/25000 [=====================>........] - ETA: 16s - loss: 7.6870 - accuracy: 0.4987
18816/25000 [=====================>........] - ETA: 16s - loss: 7.6878 - accuracy: 0.4986
18848/25000 [=====================>........] - ETA: 16s - loss: 7.6878 - accuracy: 0.4986
18880/25000 [=====================>........] - ETA: 16s - loss: 7.6885 - accuracy: 0.4986
18912/25000 [=====================>........] - ETA: 16s - loss: 7.6893 - accuracy: 0.4985
18944/25000 [=====================>........] - ETA: 16s - loss: 7.6860 - accuracy: 0.4987
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6876 - accuracy: 0.4986
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6868 - accuracy: 0.4987
19040/25000 [=====================>........] - ETA: 15s - loss: 7.6843 - accuracy: 0.4988
19072/25000 [=====================>........] - ETA: 15s - loss: 7.6867 - accuracy: 0.4987
19104/25000 [=====================>........] - ETA: 15s - loss: 7.6875 - accuracy: 0.4986
19136/25000 [=====================>........] - ETA: 15s - loss: 7.6883 - accuracy: 0.4986
19168/25000 [======================>.......] - ETA: 15s - loss: 7.6906 - accuracy: 0.4984
19200/25000 [======================>.......] - ETA: 15s - loss: 7.6914 - accuracy: 0.4984
19232/25000 [======================>.......] - ETA: 15s - loss: 7.6905 - accuracy: 0.4984
19264/25000 [======================>.......] - ETA: 15s - loss: 7.6945 - accuracy: 0.4982
19296/25000 [======================>.......] - ETA: 15s - loss: 7.6920 - accuracy: 0.4983
19328/25000 [======================>.......] - ETA: 15s - loss: 7.6952 - accuracy: 0.4981
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6967 - accuracy: 0.4980
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6982 - accuracy: 0.4979
19424/25000 [======================>.......] - ETA: 14s - loss: 7.6974 - accuracy: 0.4980
19456/25000 [======================>.......] - ETA: 14s - loss: 7.6942 - accuracy: 0.4982
19488/25000 [======================>.......] - ETA: 14s - loss: 7.6926 - accuracy: 0.4983
19520/25000 [======================>.......] - ETA: 14s - loss: 7.6918 - accuracy: 0.4984
19552/25000 [======================>.......] - ETA: 14s - loss: 7.6917 - accuracy: 0.4984
19584/25000 [======================>.......] - ETA: 14s - loss: 7.6917 - accuracy: 0.4984
19616/25000 [======================>.......] - ETA: 14s - loss: 7.6924 - accuracy: 0.4983
19648/25000 [======================>.......] - ETA: 14s - loss: 7.6893 - accuracy: 0.4985
19680/25000 [======================>.......] - ETA: 14s - loss: 7.6939 - accuracy: 0.4982
19712/25000 [======================>.......] - ETA: 14s - loss: 7.6938 - accuracy: 0.4982
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6985 - accuracy: 0.4979
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6930 - accuracy: 0.4983
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6914 - accuracy: 0.4984
19840/25000 [======================>.......] - ETA: 13s - loss: 7.6921 - accuracy: 0.4983
19872/25000 [======================>.......] - ETA: 13s - loss: 7.6944 - accuracy: 0.4982
19904/25000 [======================>.......] - ETA: 13s - loss: 7.6913 - accuracy: 0.4984
19936/25000 [======================>.......] - ETA: 13s - loss: 7.6874 - accuracy: 0.4986
19968/25000 [======================>.......] - ETA: 13s - loss: 7.6850 - accuracy: 0.4988
20000/25000 [=======================>......] - ETA: 13s - loss: 7.6850 - accuracy: 0.4988
20032/25000 [=======================>......] - ETA: 13s - loss: 7.6881 - accuracy: 0.4986
20064/25000 [=======================>......] - ETA: 13s - loss: 7.6926 - accuracy: 0.4983
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6903 - accuracy: 0.4985
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6925 - accuracy: 0.4983
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6910 - accuracy: 0.4984
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6902 - accuracy: 0.4985
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6886 - accuracy: 0.4986
20256/25000 [=======================>......] - ETA: 12s - loss: 7.6855 - accuracy: 0.4988
20288/25000 [=======================>......] - ETA: 12s - loss: 7.6855 - accuracy: 0.4988
20320/25000 [=======================>......] - ETA: 12s - loss: 7.6855 - accuracy: 0.4988
20352/25000 [=======================>......] - ETA: 12s - loss: 7.6824 - accuracy: 0.4990
20384/25000 [=======================>......] - ETA: 12s - loss: 7.6832 - accuracy: 0.4989
20416/25000 [=======================>......] - ETA: 12s - loss: 7.6831 - accuracy: 0.4989
20448/25000 [=======================>......] - ETA: 12s - loss: 7.6801 - accuracy: 0.4991
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6808 - accuracy: 0.4991
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6808 - accuracy: 0.4991
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6786 - accuracy: 0.4992
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6808 - accuracy: 0.4991
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6845 - accuracy: 0.4988
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6859 - accuracy: 0.4987
20672/25000 [=======================>......] - ETA: 11s - loss: 7.6874 - accuracy: 0.4986
20704/25000 [=======================>......] - ETA: 11s - loss: 7.6829 - accuracy: 0.4989
20736/25000 [=======================>......] - ETA: 11s - loss: 7.6851 - accuracy: 0.4988
20768/25000 [=======================>......] - ETA: 11s - loss: 7.6851 - accuracy: 0.4988
20800/25000 [=======================>......] - ETA: 11s - loss: 7.6843 - accuracy: 0.4988
20832/25000 [=======================>......] - ETA: 11s - loss: 7.6828 - accuracy: 0.4989
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6798 - accuracy: 0.4991
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6798 - accuracy: 0.4991
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6769 - accuracy: 0.4993
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6776 - accuracy: 0.4993
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6790 - accuracy: 0.4992
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6819 - accuracy: 0.4990
21056/25000 [========================>.....] - ETA: 10s - loss: 7.6819 - accuracy: 0.4990
21088/25000 [========================>.....] - ETA: 10s - loss: 7.6790 - accuracy: 0.4992
21120/25000 [========================>.....] - ETA: 10s - loss: 7.6782 - accuracy: 0.4992
21152/25000 [========================>.....] - ETA: 10s - loss: 7.6768 - accuracy: 0.4993
21184/25000 [========================>.....] - ETA: 10s - loss: 7.6753 - accuracy: 0.4994
21216/25000 [========================>.....] - ETA: 10s - loss: 7.6767 - accuracy: 0.4993
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6746 - accuracy: 0.4995 
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6731 - accuracy: 0.4996
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6738 - accuracy: 0.4995
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6745 - accuracy: 0.4995
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6759 - accuracy: 0.4994
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6795 - accuracy: 0.4992
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6816 - accuracy: 0.4990
21472/25000 [========================>.....] - ETA: 9s - loss: 7.6845 - accuracy: 0.4988
21504/25000 [========================>.....] - ETA: 9s - loss: 7.6816 - accuracy: 0.4990
21536/25000 [========================>.....] - ETA: 9s - loss: 7.6787 - accuracy: 0.4992
21568/25000 [========================>.....] - ETA: 9s - loss: 7.6801 - accuracy: 0.4991
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6794 - accuracy: 0.4992
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6765 - accuracy: 0.4994
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6787 - accuracy: 0.4992
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6758 - accuracy: 0.4994
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6772 - accuracy: 0.4993
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6779 - accuracy: 0.4993
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6807 - accuracy: 0.4991
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6786 - accuracy: 0.4992
21856/25000 [=========================>....] - ETA: 8s - loss: 7.6814 - accuracy: 0.4990
21888/25000 [=========================>....] - ETA: 8s - loss: 7.6841 - accuracy: 0.4989
21920/25000 [=========================>....] - ETA: 8s - loss: 7.6820 - accuracy: 0.4990
21952/25000 [=========================>....] - ETA: 8s - loss: 7.6841 - accuracy: 0.4989
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6820 - accuracy: 0.4990
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6819 - accuracy: 0.4990
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6798 - accuracy: 0.4991
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6826 - accuracy: 0.4990
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6819 - accuracy: 0.4990
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6853 - accuracy: 0.4988
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6846 - accuracy: 0.4988
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6894 - accuracy: 0.4985
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6894 - accuracy: 0.4985
22272/25000 [=========================>....] - ETA: 7s - loss: 7.6880 - accuracy: 0.4986
22304/25000 [=========================>....] - ETA: 7s - loss: 7.6893 - accuracy: 0.4985
22336/25000 [=========================>....] - ETA: 7s - loss: 7.6913 - accuracy: 0.4984
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6920 - accuracy: 0.4983
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6899 - accuracy: 0.4985
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6899 - accuracy: 0.4985
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6932 - accuracy: 0.4983
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6966 - accuracy: 0.4980
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6938 - accuracy: 0.4982
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6931 - accuracy: 0.4983
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6944 - accuracy: 0.4982
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6944 - accuracy: 0.4982
22656/25000 [==========================>...] - ETA: 6s - loss: 7.6930 - accuracy: 0.4983
22688/25000 [==========================>...] - ETA: 6s - loss: 7.6950 - accuracy: 0.4981
22720/25000 [==========================>...] - ETA: 6s - loss: 7.6963 - accuracy: 0.4981
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6943 - accuracy: 0.4982
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6902 - accuracy: 0.4985
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6888 - accuracy: 0.4986
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6874 - accuracy: 0.4986
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6854 - accuracy: 0.4988
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6834 - accuracy: 0.4989
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6827 - accuracy: 0.4990
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6846 - accuracy: 0.4988
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6846 - accuracy: 0.4988
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6873 - accuracy: 0.4987
23072/25000 [==========================>...] - ETA: 5s - loss: 7.6866 - accuracy: 0.4987
23104/25000 [==========================>...] - ETA: 5s - loss: 7.6865 - accuracy: 0.4987
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6858 - accuracy: 0.4987
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6858 - accuracy: 0.4987
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6858 - accuracy: 0.4988
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6838 - accuracy: 0.4989
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6838 - accuracy: 0.4989
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6837 - accuracy: 0.4989
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6870 - accuracy: 0.4987
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6876 - accuracy: 0.4986
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6889 - accuracy: 0.4985
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6908 - accuracy: 0.4984
23456/25000 [===========================>..] - ETA: 4s - loss: 7.6934 - accuracy: 0.4983
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6921 - accuracy: 0.4983
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6901 - accuracy: 0.4985
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6894 - accuracy: 0.4985
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6900 - accuracy: 0.4985
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6880 - accuracy: 0.4986
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6893 - accuracy: 0.4985
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6912 - accuracy: 0.4984
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6918 - accuracy: 0.4984
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6892 - accuracy: 0.4985
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6905 - accuracy: 0.4984
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6917 - accuracy: 0.4984
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6917 - accuracy: 0.4984
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6949 - accuracy: 0.4982
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6948 - accuracy: 0.4982
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6961 - accuracy: 0.4981
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6948 - accuracy: 0.4982
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6954 - accuracy: 0.4981
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6947 - accuracy: 0.4982
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6934 - accuracy: 0.4983
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6921 - accuracy: 0.4983
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6939 - accuracy: 0.4982
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6958 - accuracy: 0.4981
24192/25000 [============================>.] - ETA: 2s - loss: 7.6964 - accuracy: 0.4981
24224/25000 [============================>.] - ETA: 2s - loss: 7.6964 - accuracy: 0.4981
24256/25000 [============================>.] - ETA: 1s - loss: 7.6944 - accuracy: 0.4982
24288/25000 [============================>.] - ETA: 1s - loss: 7.6925 - accuracy: 0.4983
24320/25000 [============================>.] - ETA: 1s - loss: 7.6899 - accuracy: 0.4985
24352/25000 [============================>.] - ETA: 1s - loss: 7.6880 - accuracy: 0.4986
24384/25000 [============================>.] - ETA: 1s - loss: 7.6855 - accuracy: 0.4988
24416/25000 [============================>.] - ETA: 1s - loss: 7.6848 - accuracy: 0.4988
24448/25000 [============================>.] - ETA: 1s - loss: 7.6879 - accuracy: 0.4986
24480/25000 [============================>.] - ETA: 1s - loss: 7.6885 - accuracy: 0.4986
24512/25000 [============================>.] - ETA: 1s - loss: 7.6873 - accuracy: 0.4987
24544/25000 [============================>.] - ETA: 1s - loss: 7.6854 - accuracy: 0.4988
24576/25000 [============================>.] - ETA: 1s - loss: 7.6841 - accuracy: 0.4989
24608/25000 [============================>.] - ETA: 1s - loss: 7.6828 - accuracy: 0.4989
24640/25000 [============================>.] - ETA: 0s - loss: 7.6816 - accuracy: 0.4990
24672/25000 [============================>.] - ETA: 0s - loss: 7.6784 - accuracy: 0.4992
24704/25000 [============================>.] - ETA: 0s - loss: 7.6784 - accuracy: 0.4992
24736/25000 [============================>.] - ETA: 0s - loss: 7.6765 - accuracy: 0.4994
24768/25000 [============================>.] - ETA: 0s - loss: 7.6734 - accuracy: 0.4996
24800/25000 [============================>.] - ETA: 0s - loss: 7.6747 - accuracy: 0.4995
24832/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24864/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24896/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24928/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24960/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
25000/25000 [==============================] - 78s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f388eb3da58> 

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
 [-0.02652038  0.01526134  0.11876713  0.00579225  0.14340006  0.016548  ]
 [-0.2628977  -0.102519   -0.16173598  0.09139571  0.3977738   0.03578474]
 [-0.11814667 -0.0058542  -0.03195895 -0.10136654 -0.00596349  0.13212365]
 [ 0.41179171  0.05365965 -0.01574416 -0.28176612  0.13658941  0.19773158]
 [ 0.07243663  0.05389145  0.20484215  0.16192487  0.07173517  0.09455655]
 [-0.24371471 -0.0252197   0.04777736  0.1140833  -0.12017541 -0.03273654]
 [-0.35774171  0.38372654  0.28340313  0.12698761 -0.03986476 -0.07802459]
 [-0.05169187 -0.52782458 -0.02215966  0.60361624  0.30862948  0.2640937 ]
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
{'loss': 0.47900019958615303, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-12 10:16:10.435564: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 523, in main
    test_cli(arg)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 455, in test_cli
    test(arg.model_uri)  # '1_lstm'
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 189, in test
    module.test()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
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
{'loss': 0.4187856465578079, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-12 10:16:11.609256: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 523, in main
    test_cli(arg)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 457, in test_cli
    test_global(arg.model_uri)  # '1_lstm'
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 200, in test_global
    module.test()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
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
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:51<01:17, 25.83s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.2859861264578745, 'embedding_size_factor': 0.8964377071284231, 'layers.choice': 2, 'learning_rate': 0.0002941299253048162, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 0.0002048797255854196} and reward: 0.3726
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd2M\x98\xc1\x0f\xc3=X\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\xaf\x9e!`\x91\xd8X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?3F\xaeh\xf3\x14\x83X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?*\xda\x9fq'H|u." and reward: 0.3726
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd2M\x98\xc1\x0f\xc3=X\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\xaf\x9e!`\x91\xd8X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?3F\xaeh\xf3\x14\x83X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?*\xda\x9fq'H|u." and reward: 0.3726
 60%|██████    | 3/5 [01:43<01:07, 33.66s/it] 60%|██████    | 3/5 [01:43<01:09, 34.53s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.48528972868908826, 'embedding_size_factor': 0.5980963687340847, 'layers.choice': 1, 'learning_rate': 0.00617413027019728, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 9.418025045265299e-06} and reward: 0.3766
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdf\x0e\xfc\xa6szlX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe3#\x9a\xfe\xf27MX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?yJ\x0byw\xc5_X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xe3\xc0C\x81\x88\xd8yu.' and reward: 0.3766
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdf\x0e\xfc\xa6szlX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe3#\x9a\xfe\xf27MX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?yJ\x0byw\xc5_X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xe3\xc0C\x81\x88\xd8yu.' and reward: 0.3766
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 198.19964003562927
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -80.89s of remaining time.
Ensemble size: 46
Ensemble weights: 
[0.76086957 0.         0.23913043]
	0.3898	 = Validation accuracy score
	1.08s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 202.02s ...
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 523, in main
    test_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 453, in test_cli
    test_module(arg.model_uri, param_pars=param_pars)  # '1_lstm'
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 257, in test_module
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
[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     87[0m [0;34m[0m[0m
[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     90[0m [0;34m[0m[0m
[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     87[0m [0;34m[0m[0m
[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     90[0m [0;34m[0m[0m
[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

