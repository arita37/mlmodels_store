
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
 1728512/17464789 [=>............................] - ETA: 0s
 4317184/17464789 [======>.......................] - ETA: 0s
 6881280/17464789 [==========>...................] - ETA: 0s
 9609216/17464789 [===============>..............] - ETA: 0s
12312576/17464789 [====================>.........] - ETA: 0s
14991360/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 05:15:18.106191: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-12 05:15:18.119964: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-12 05:15:18.120487: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55864df8e120 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 05:15:18.120776: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 6:18 - loss: 11.0208 - accuracy: 0.2812
   64/25000 [..............................] - ETA: 3:43 - loss: 8.6249 - accuracy: 0.4375 
   96/25000 [..............................] - ETA: 2:49 - loss: 8.3055 - accuracy: 0.4583
  128/25000 [..............................] - ETA: 2:23 - loss: 8.2656 - accuracy: 0.4609
  160/25000 [..............................] - ETA: 2:06 - loss: 7.8583 - accuracy: 0.4875
  192/25000 [..............................] - ETA: 1:56 - loss: 8.0659 - accuracy: 0.4740
  224/25000 [..............................] - ETA: 1:49 - loss: 7.9404 - accuracy: 0.4821
  256/25000 [..............................] - ETA: 1:43 - loss: 7.7864 - accuracy: 0.4922
  288/25000 [..............................] - ETA: 1:38 - loss: 7.6666 - accuracy: 0.5000
  320/25000 [..............................] - ETA: 1:35 - loss: 7.5708 - accuracy: 0.5063
  352/25000 [..............................] - ETA: 1:32 - loss: 7.4053 - accuracy: 0.5170
  384/25000 [..............................] - ETA: 1:29 - loss: 7.1875 - accuracy: 0.5312
  416/25000 [..............................] - ETA: 1:26 - loss: 7.2612 - accuracy: 0.5264
  448/25000 [..............................] - ETA: 1:24 - loss: 7.4270 - accuracy: 0.5156
  480/25000 [..............................] - ETA: 1:22 - loss: 7.4430 - accuracy: 0.5146
  512/25000 [..............................] - ETA: 1:21 - loss: 7.5468 - accuracy: 0.5078
  544/25000 [..............................] - ETA: 1:20 - loss: 7.5257 - accuracy: 0.5092
  576/25000 [..............................] - ETA: 1:19 - loss: 7.6134 - accuracy: 0.5035
  608/25000 [..............................] - ETA: 1:18 - loss: 7.5910 - accuracy: 0.5049
  640/25000 [..............................] - ETA: 1:18 - loss: 7.5947 - accuracy: 0.5047
  672/25000 [..............................] - ETA: 1:17 - loss: 7.5982 - accuracy: 0.5045
  704/25000 [..............................] - ETA: 1:16 - loss: 7.7537 - accuracy: 0.4943
  736/25000 [..............................] - ETA: 1:16 - loss: 7.7916 - accuracy: 0.4918
  768/25000 [..............................] - ETA: 1:15 - loss: 7.8663 - accuracy: 0.4870
  800/25000 [..............................] - ETA: 1:15 - loss: 7.8583 - accuracy: 0.4875
  832/25000 [..............................] - ETA: 1:14 - loss: 7.9246 - accuracy: 0.4832
  864/25000 [>.............................] - ETA: 1:14 - loss: 7.8973 - accuracy: 0.4850
  896/25000 [>.............................] - ETA: 1:13 - loss: 7.8206 - accuracy: 0.4900
  928/25000 [>.............................] - ETA: 1:13 - loss: 7.8318 - accuracy: 0.4892
  960/25000 [>.............................] - ETA: 1:12 - loss: 7.8423 - accuracy: 0.4885
  992/25000 [>.............................] - ETA: 1:12 - loss: 7.8676 - accuracy: 0.4869
 1024/25000 [>.............................] - ETA: 1:12 - loss: 7.8164 - accuracy: 0.4902
 1056/25000 [>.............................] - ETA: 1:12 - loss: 7.8263 - accuracy: 0.4896
 1088/25000 [>.............................] - ETA: 1:11 - loss: 7.8076 - accuracy: 0.4908
 1120/25000 [>.............................] - ETA: 1:11 - loss: 7.8857 - accuracy: 0.4857
 1152/25000 [>.............................] - ETA: 1:10 - loss: 7.8929 - accuracy: 0.4852
 1184/25000 [>.............................] - ETA: 1:10 - loss: 7.9127 - accuracy: 0.4840
 1216/25000 [>.............................] - ETA: 1:10 - loss: 7.9062 - accuracy: 0.4844
 1248/25000 [>.............................] - ETA: 1:09 - loss: 7.8878 - accuracy: 0.4856
 1280/25000 [>.............................] - ETA: 1:09 - loss: 7.9062 - accuracy: 0.4844
 1312/25000 [>.............................] - ETA: 1:09 - loss: 7.9120 - accuracy: 0.4840
 1344/25000 [>.............................] - ETA: 1:09 - loss: 7.9176 - accuracy: 0.4836
 1376/25000 [>.............................] - ETA: 1:08 - loss: 7.8561 - accuracy: 0.4876
 1408/25000 [>.............................] - ETA: 1:08 - loss: 7.8518 - accuracy: 0.4879
 1440/25000 [>.............................] - ETA: 1:08 - loss: 7.8689 - accuracy: 0.4868
 1472/25000 [>.............................] - ETA: 1:07 - loss: 7.8854 - accuracy: 0.4857
 1504/25000 [>.............................] - ETA: 1:07 - loss: 7.8501 - accuracy: 0.4880
 1536/25000 [>.............................] - ETA: 1:07 - loss: 7.8363 - accuracy: 0.4889
 1568/25000 [>.............................] - ETA: 1:07 - loss: 7.8329 - accuracy: 0.4892
 1600/25000 [>.............................] - ETA: 1:06 - loss: 7.8104 - accuracy: 0.4906
 1632/25000 [>.............................] - ETA: 1:06 - loss: 7.7982 - accuracy: 0.4914
 1664/25000 [>.............................] - ETA: 1:06 - loss: 7.8048 - accuracy: 0.4910
 1696/25000 [=>............................] - ETA: 1:06 - loss: 7.7932 - accuracy: 0.4917
 1728/25000 [=>............................] - ETA: 1:06 - loss: 7.8086 - accuracy: 0.4907
 1760/25000 [=>............................] - ETA: 1:05 - loss: 7.8409 - accuracy: 0.4886
 1792/25000 [=>............................] - ETA: 1:05 - loss: 7.8206 - accuracy: 0.4900
 1824/25000 [=>............................] - ETA: 1:05 - loss: 7.7675 - accuracy: 0.4934
 1856/25000 [=>............................] - ETA: 1:05 - loss: 7.7905 - accuracy: 0.4919
 1888/25000 [=>............................] - ETA: 1:04 - loss: 7.8047 - accuracy: 0.4910
 1920/25000 [=>............................] - ETA: 1:04 - loss: 7.8184 - accuracy: 0.4901
 1952/25000 [=>............................] - ETA: 1:04 - loss: 7.8237 - accuracy: 0.4898
 1984/25000 [=>............................] - ETA: 1:04 - loss: 7.8135 - accuracy: 0.4904
 2016/25000 [=>............................] - ETA: 1:04 - loss: 7.8111 - accuracy: 0.4906
 2048/25000 [=>............................] - ETA: 1:04 - loss: 7.8238 - accuracy: 0.4897
 2080/25000 [=>............................] - ETA: 1:04 - loss: 7.8288 - accuracy: 0.4894
 2112/25000 [=>............................] - ETA: 1:04 - loss: 7.8191 - accuracy: 0.4901
 2144/25000 [=>............................] - ETA: 1:03 - loss: 7.8240 - accuracy: 0.4897
 2176/25000 [=>............................] - ETA: 1:03 - loss: 7.7935 - accuracy: 0.4917
 2208/25000 [=>............................] - ETA: 1:03 - loss: 7.7986 - accuracy: 0.4914
 2240/25000 [=>............................] - ETA: 1:03 - loss: 7.7967 - accuracy: 0.4915
 2272/25000 [=>............................] - ETA: 1:03 - loss: 7.8016 - accuracy: 0.4912
 2304/25000 [=>............................] - ETA: 1:03 - loss: 7.7864 - accuracy: 0.4922
 2336/25000 [=>............................] - ETA: 1:02 - loss: 7.7782 - accuracy: 0.4927
 2368/25000 [=>............................] - ETA: 1:02 - loss: 7.7573 - accuracy: 0.4941
 2400/25000 [=>............................] - ETA: 1:02 - loss: 7.7497 - accuracy: 0.4946
 2432/25000 [=>............................] - ETA: 1:02 - loss: 7.7549 - accuracy: 0.4942
 2464/25000 [=>............................] - ETA: 1:02 - loss: 7.7537 - accuracy: 0.4943
 2496/25000 [=>............................] - ETA: 1:02 - loss: 7.7588 - accuracy: 0.4940
 2528/25000 [==>...........................] - ETA: 1:02 - loss: 7.7879 - accuracy: 0.4921
 2560/25000 [==>...........................] - ETA: 1:02 - loss: 7.7804 - accuracy: 0.4926
 2592/25000 [==>...........................] - ETA: 1:01 - loss: 7.7731 - accuracy: 0.4931
 2624/25000 [==>...........................] - ETA: 1:01 - loss: 7.7426 - accuracy: 0.4950
 2656/25000 [==>...........................] - ETA: 1:01 - loss: 7.7244 - accuracy: 0.4962
 2688/25000 [==>...........................] - ETA: 1:01 - loss: 7.7408 - accuracy: 0.4952
 2720/25000 [==>...........................] - ETA: 1:01 - loss: 7.7568 - accuracy: 0.4941
 2752/25000 [==>...........................] - ETA: 1:01 - loss: 7.7781 - accuracy: 0.4927
 2784/25000 [==>...........................] - ETA: 1:00 - loss: 7.7547 - accuracy: 0.4943
 2816/25000 [==>...........................] - ETA: 1:00 - loss: 7.7483 - accuracy: 0.4947
 2848/25000 [==>...........................] - ETA: 1:00 - loss: 7.7312 - accuracy: 0.4958
 2880/25000 [==>...........................] - ETA: 1:00 - loss: 7.7145 - accuracy: 0.4969
 2912/25000 [==>...........................] - ETA: 1:00 - loss: 7.7298 - accuracy: 0.4959
 2944/25000 [==>...........................] - ETA: 1:00 - loss: 7.7447 - accuracy: 0.4949
 2976/25000 [==>...........................] - ETA: 1:00 - loss: 7.7697 - accuracy: 0.4933
 3008/25000 [==>...........................] - ETA: 1:00 - loss: 7.7584 - accuracy: 0.4940
 3040/25000 [==>...........................] - ETA: 59s - loss: 7.7776 - accuracy: 0.4928 
 3072/25000 [==>...........................] - ETA: 59s - loss: 7.7814 - accuracy: 0.4925
 3104/25000 [==>...........................] - ETA: 59s - loss: 7.8049 - accuracy: 0.4910
 3136/25000 [==>...........................] - ETA: 59s - loss: 7.8182 - accuracy: 0.4901
 3168/25000 [==>...........................] - ETA: 59s - loss: 7.8263 - accuracy: 0.4896
 3200/25000 [==>...........................] - ETA: 59s - loss: 7.8343 - accuracy: 0.4891
 3232/25000 [==>...........................] - ETA: 59s - loss: 7.8089 - accuracy: 0.4907
 3264/25000 [==>...........................] - ETA: 59s - loss: 7.7935 - accuracy: 0.4917
 3296/25000 [==>...........................] - ETA: 58s - loss: 7.7829 - accuracy: 0.4924
 3328/25000 [==>...........................] - ETA: 58s - loss: 7.7956 - accuracy: 0.4916
 3360/25000 [===>..........................] - ETA: 58s - loss: 7.7898 - accuracy: 0.4920
 3392/25000 [===>..........................] - ETA: 58s - loss: 7.8022 - accuracy: 0.4912
 3424/25000 [===>..........................] - ETA: 58s - loss: 7.8099 - accuracy: 0.4907
 3456/25000 [===>..........................] - ETA: 58s - loss: 7.8263 - accuracy: 0.4896
 3488/25000 [===>..........................] - ETA: 58s - loss: 7.8161 - accuracy: 0.4903
 3520/25000 [===>..........................] - ETA: 58s - loss: 7.8321 - accuracy: 0.4892
 3552/25000 [===>..........................] - ETA: 57s - loss: 7.8307 - accuracy: 0.4893
 3584/25000 [===>..........................] - ETA: 57s - loss: 7.8292 - accuracy: 0.4894
 3616/25000 [===>..........................] - ETA: 57s - loss: 7.8235 - accuracy: 0.4898
 3648/25000 [===>..........................] - ETA: 57s - loss: 7.8221 - accuracy: 0.4899
 3680/25000 [===>..........................] - ETA: 57s - loss: 7.8083 - accuracy: 0.4908
 3712/25000 [===>..........................] - ETA: 57s - loss: 7.8195 - accuracy: 0.4900
 3744/25000 [===>..........................] - ETA: 57s - loss: 7.8100 - accuracy: 0.4907
 3776/25000 [===>..........................] - ETA: 57s - loss: 7.8128 - accuracy: 0.4905
 3808/25000 [===>..........................] - ETA: 57s - loss: 7.8196 - accuracy: 0.4900
 3840/25000 [===>..........................] - ETA: 57s - loss: 7.8223 - accuracy: 0.4898
 3872/25000 [===>..........................] - ETA: 56s - loss: 7.8409 - accuracy: 0.4886
 3904/25000 [===>..........................] - ETA: 56s - loss: 7.8355 - accuracy: 0.4890
 3936/25000 [===>..........................] - ETA: 56s - loss: 7.8341 - accuracy: 0.4891
 3968/25000 [===>..........................] - ETA: 56s - loss: 7.8405 - accuracy: 0.4887
 4000/25000 [===>..........................] - ETA: 56s - loss: 7.8276 - accuracy: 0.4895
 4032/25000 [===>..........................] - ETA: 56s - loss: 7.8263 - accuracy: 0.4896
 4064/25000 [===>..........................] - ETA: 56s - loss: 7.8100 - accuracy: 0.4906
 4096/25000 [===>..........................] - ETA: 56s - loss: 7.7976 - accuracy: 0.4915
 4128/25000 [===>..........................] - ETA: 55s - loss: 7.7818 - accuracy: 0.4925
 4160/25000 [===>..........................] - ETA: 55s - loss: 7.7846 - accuracy: 0.4923
 4192/25000 [====>.........................] - ETA: 55s - loss: 7.7800 - accuracy: 0.4926
 4224/25000 [====>.........................] - ETA: 55s - loss: 7.7755 - accuracy: 0.4929
 4256/25000 [====>.........................] - ETA: 55s - loss: 7.7855 - accuracy: 0.4922
 4288/25000 [====>.........................] - ETA: 55s - loss: 7.7846 - accuracy: 0.4923
 4320/25000 [====>.........................] - ETA: 55s - loss: 7.7766 - accuracy: 0.4928
 4352/25000 [====>.........................] - ETA: 55s - loss: 7.7653 - accuracy: 0.4936
 4384/25000 [====>.........................] - ETA: 55s - loss: 7.7715 - accuracy: 0.4932
 4416/25000 [====>.........................] - ETA: 54s - loss: 7.7847 - accuracy: 0.4923
 4448/25000 [====>.........................] - ETA: 54s - loss: 7.7942 - accuracy: 0.4917
 4480/25000 [====>.........................] - ETA: 54s - loss: 7.7967 - accuracy: 0.4915
 4512/25000 [====>.........................] - ETA: 54s - loss: 7.8060 - accuracy: 0.4909
 4544/25000 [====>.........................] - ETA: 54s - loss: 7.7847 - accuracy: 0.4923
 4576/25000 [====>.........................] - ETA: 54s - loss: 7.7872 - accuracy: 0.4921
 4608/25000 [====>.........................] - ETA: 54s - loss: 7.7764 - accuracy: 0.4928
 4640/25000 [====>.........................] - ETA: 54s - loss: 7.7823 - accuracy: 0.4925
 4672/25000 [====>.........................] - ETA: 54s - loss: 7.7881 - accuracy: 0.4921
 4704/25000 [====>.........................] - ETA: 53s - loss: 7.7742 - accuracy: 0.4930
 4736/25000 [====>.........................] - ETA: 53s - loss: 7.7832 - accuracy: 0.4924
 4768/25000 [====>.........................] - ETA: 53s - loss: 7.7888 - accuracy: 0.4920
 4800/25000 [====>.........................] - ETA: 53s - loss: 7.7816 - accuracy: 0.4925
 4832/25000 [====>.........................] - ETA: 53s - loss: 7.7872 - accuracy: 0.4921
 4864/25000 [====>.........................] - ETA: 53s - loss: 7.7959 - accuracy: 0.4916
 4896/25000 [====>.........................] - ETA: 53s - loss: 7.7794 - accuracy: 0.4926
 4928/25000 [====>.........................] - ETA: 53s - loss: 7.7786 - accuracy: 0.4927
 4960/25000 [====>.........................] - ETA: 53s - loss: 7.7841 - accuracy: 0.4923
 4992/25000 [====>.........................] - ETA: 53s - loss: 7.7741 - accuracy: 0.4930
 5024/25000 [=====>........................] - ETA: 52s - loss: 7.7734 - accuracy: 0.4930
 5056/25000 [=====>........................] - ETA: 52s - loss: 7.7788 - accuracy: 0.4927
 5088/25000 [=====>........................] - ETA: 52s - loss: 7.7781 - accuracy: 0.4927
 5120/25000 [=====>........................] - ETA: 52s - loss: 7.7864 - accuracy: 0.4922
 5152/25000 [=====>........................] - ETA: 52s - loss: 7.8035 - accuracy: 0.4911
 5184/25000 [=====>........................] - ETA: 52s - loss: 7.7968 - accuracy: 0.4915
 5216/25000 [=====>........................] - ETA: 52s - loss: 7.7960 - accuracy: 0.4916
 5248/25000 [=====>........................] - ETA: 52s - loss: 7.8039 - accuracy: 0.4910
 5280/25000 [=====>........................] - ETA: 52s - loss: 7.8089 - accuracy: 0.4907
 5312/25000 [=====>........................] - ETA: 52s - loss: 7.8052 - accuracy: 0.4910
 5344/25000 [=====>........................] - ETA: 51s - loss: 7.8015 - accuracy: 0.4912
 5376/25000 [=====>........................] - ETA: 51s - loss: 7.7921 - accuracy: 0.4918
 5408/25000 [=====>........................] - ETA: 51s - loss: 7.7829 - accuracy: 0.4924
 5440/25000 [=====>........................] - ETA: 51s - loss: 7.7709 - accuracy: 0.4932
 5472/25000 [=====>........................] - ETA: 51s - loss: 7.7815 - accuracy: 0.4925
 5504/25000 [=====>........................] - ETA: 51s - loss: 7.7725 - accuracy: 0.4931
 5536/25000 [=====>........................] - ETA: 51s - loss: 7.7802 - accuracy: 0.4926
 5568/25000 [=====>........................] - ETA: 51s - loss: 7.7685 - accuracy: 0.4934
 5600/25000 [=====>........................] - ETA: 51s - loss: 7.7707 - accuracy: 0.4932
 5632/25000 [=====>........................] - ETA: 51s - loss: 7.7755 - accuracy: 0.4929
 5664/25000 [=====>........................] - ETA: 51s - loss: 7.7668 - accuracy: 0.4935
 5696/25000 [=====>........................] - ETA: 50s - loss: 7.7716 - accuracy: 0.4932
 5728/25000 [=====>........................] - ETA: 50s - loss: 7.7817 - accuracy: 0.4925
 5760/25000 [=====>........................] - ETA: 50s - loss: 7.7891 - accuracy: 0.4920
 5792/25000 [=====>........................] - ETA: 50s - loss: 7.7857 - accuracy: 0.4922
 5824/25000 [=====>........................] - ETA: 50s - loss: 7.7851 - accuracy: 0.4923
 5856/25000 [======>.......................] - ETA: 50s - loss: 7.7923 - accuracy: 0.4918
 5888/25000 [======>.......................] - ETA: 50s - loss: 7.8046 - accuracy: 0.4910
 5920/25000 [======>.......................] - ETA: 50s - loss: 7.8013 - accuracy: 0.4912
 5952/25000 [======>.......................] - ETA: 50s - loss: 7.8032 - accuracy: 0.4911
 5984/25000 [======>.......................] - ETA: 49s - loss: 7.8076 - accuracy: 0.4908
 6016/25000 [======>.......................] - ETA: 49s - loss: 7.7941 - accuracy: 0.4917
 6048/25000 [======>.......................] - ETA: 49s - loss: 7.7985 - accuracy: 0.4914
 6080/25000 [======>.......................] - ETA: 49s - loss: 7.7851 - accuracy: 0.4923
 6112/25000 [======>.......................] - ETA: 49s - loss: 7.7820 - accuracy: 0.4925
 6144/25000 [======>.......................] - ETA: 49s - loss: 7.7639 - accuracy: 0.4937
 6176/25000 [======>.......................] - ETA: 49s - loss: 7.7659 - accuracy: 0.4935
 6208/25000 [======>.......................] - ETA: 49s - loss: 7.7580 - accuracy: 0.4940
 6240/25000 [======>.......................] - ETA: 49s - loss: 7.7526 - accuracy: 0.4944
 6272/25000 [======>.......................] - ETA: 49s - loss: 7.7400 - accuracy: 0.4952
 6304/25000 [======>.......................] - ETA: 49s - loss: 7.7420 - accuracy: 0.4951
 6336/25000 [======>.......................] - ETA: 49s - loss: 7.7441 - accuracy: 0.4949
 6368/25000 [======>.......................] - ETA: 48s - loss: 7.7413 - accuracy: 0.4951
 6400/25000 [======>.......................] - ETA: 48s - loss: 7.7481 - accuracy: 0.4947
 6432/25000 [======>.......................] - ETA: 48s - loss: 7.7524 - accuracy: 0.4944
 6464/25000 [======>.......................] - ETA: 48s - loss: 7.7473 - accuracy: 0.4947
 6496/25000 [======>.......................] - ETA: 48s - loss: 7.7516 - accuracy: 0.4945
 6528/25000 [======>.......................] - ETA: 48s - loss: 7.7488 - accuracy: 0.4946
 6560/25000 [======>.......................] - ETA: 48s - loss: 7.7438 - accuracy: 0.4950
 6592/25000 [======>.......................] - ETA: 48s - loss: 7.7504 - accuracy: 0.4945
 6624/25000 [======>.......................] - ETA: 48s - loss: 7.7569 - accuracy: 0.4941
 6656/25000 [======>.......................] - ETA: 48s - loss: 7.7542 - accuracy: 0.4943
 6688/25000 [=======>......................] - ETA: 47s - loss: 7.7514 - accuracy: 0.4945
 6720/25000 [=======>......................] - ETA: 47s - loss: 7.7579 - accuracy: 0.4940
 6752/25000 [=======>......................] - ETA: 47s - loss: 7.7597 - accuracy: 0.4939
 6784/25000 [=======>......................] - ETA: 47s - loss: 7.7480 - accuracy: 0.4947
 6816/25000 [=======>......................] - ETA: 47s - loss: 7.7409 - accuracy: 0.4952
 6848/25000 [=======>......................] - ETA: 47s - loss: 7.7383 - accuracy: 0.4953
 6880/25000 [=======>......................] - ETA: 47s - loss: 7.7424 - accuracy: 0.4951
 6912/25000 [=======>......................] - ETA: 47s - loss: 7.7310 - accuracy: 0.4958
 6944/25000 [=======>......................] - ETA: 47s - loss: 7.7329 - accuracy: 0.4957
 6976/25000 [=======>......................] - ETA: 47s - loss: 7.7304 - accuracy: 0.4958
 7008/25000 [=======>......................] - ETA: 47s - loss: 7.7410 - accuracy: 0.4951
 7040/25000 [=======>......................] - ETA: 46s - loss: 7.7450 - accuracy: 0.4949
 7072/25000 [=======>......................] - ETA: 46s - loss: 7.7425 - accuracy: 0.4951
 7104/25000 [=======>......................] - ETA: 46s - loss: 7.7508 - accuracy: 0.4945
 7136/25000 [=======>......................] - ETA: 46s - loss: 7.7418 - accuracy: 0.4951
 7168/25000 [=======>......................] - ETA: 46s - loss: 7.7329 - accuracy: 0.4957
 7200/25000 [=======>......................] - ETA: 46s - loss: 7.7305 - accuracy: 0.4958
 7232/25000 [=======>......................] - ETA: 46s - loss: 7.7281 - accuracy: 0.4960
 7264/25000 [=======>......................] - ETA: 46s - loss: 7.7363 - accuracy: 0.4955
 7296/25000 [=======>......................] - ETA: 46s - loss: 7.7318 - accuracy: 0.4958
 7328/25000 [=======>......................] - ETA: 46s - loss: 7.7419 - accuracy: 0.4951
 7360/25000 [=======>......................] - ETA: 46s - loss: 7.7375 - accuracy: 0.4954
 7392/25000 [=======>......................] - ETA: 45s - loss: 7.7351 - accuracy: 0.4955
 7424/25000 [=======>......................] - ETA: 45s - loss: 7.7410 - accuracy: 0.4952
 7456/25000 [=======>......................] - ETA: 45s - loss: 7.7427 - accuracy: 0.4950
 7488/25000 [=======>......................] - ETA: 45s - loss: 7.7342 - accuracy: 0.4956
 7520/25000 [========>.....................] - ETA: 45s - loss: 7.7359 - accuracy: 0.4955
 7552/25000 [========>.....................] - ETA: 45s - loss: 7.7296 - accuracy: 0.4959
 7584/25000 [========>.....................] - ETA: 45s - loss: 7.7354 - accuracy: 0.4955
 7616/25000 [========>.....................] - ETA: 45s - loss: 7.7310 - accuracy: 0.4958
 7648/25000 [========>.....................] - ETA: 45s - loss: 7.7388 - accuracy: 0.4953
 7680/25000 [========>.....................] - ETA: 45s - loss: 7.7345 - accuracy: 0.4956
 7712/25000 [========>.....................] - ETA: 45s - loss: 7.7283 - accuracy: 0.4960
 7744/25000 [========>.....................] - ETA: 44s - loss: 7.7300 - accuracy: 0.4959
 7776/25000 [========>.....................] - ETA: 44s - loss: 7.7297 - accuracy: 0.4959
 7808/25000 [========>.....................] - ETA: 44s - loss: 7.7196 - accuracy: 0.4965
 7840/25000 [========>.....................] - ETA: 44s - loss: 7.7116 - accuracy: 0.4971
 7872/25000 [========>.....................] - ETA: 44s - loss: 7.7095 - accuracy: 0.4972
 7904/25000 [========>.....................] - ETA: 44s - loss: 7.7093 - accuracy: 0.4972
 7936/25000 [========>.....................] - ETA: 44s - loss: 7.6995 - accuracy: 0.4979
 7968/25000 [========>.....................] - ETA: 44s - loss: 7.6993 - accuracy: 0.4979
 8000/25000 [========>.....................] - ETA: 44s - loss: 7.7011 - accuracy: 0.4978
 8032/25000 [========>.....................] - ETA: 44s - loss: 7.7029 - accuracy: 0.4976
 8064/25000 [========>.....................] - ETA: 44s - loss: 7.6989 - accuracy: 0.4979
 8096/25000 [========>.....................] - ETA: 43s - loss: 7.6912 - accuracy: 0.4984
 8128/25000 [========>.....................] - ETA: 43s - loss: 7.6930 - accuracy: 0.4983
 8160/25000 [========>.....................] - ETA: 43s - loss: 7.6986 - accuracy: 0.4979
 8192/25000 [========>.....................] - ETA: 43s - loss: 7.7022 - accuracy: 0.4977
 8224/25000 [========>.....................] - ETA: 43s - loss: 7.7020 - accuracy: 0.4977
 8256/25000 [========>.....................] - ETA: 43s - loss: 7.6945 - accuracy: 0.4982
 8288/25000 [========>.....................] - ETA: 43s - loss: 7.6870 - accuracy: 0.4987
 8320/25000 [========>.....................] - ETA: 43s - loss: 7.6832 - accuracy: 0.4989
 8352/25000 [=========>....................] - ETA: 43s - loss: 7.6776 - accuracy: 0.4993
 8384/25000 [=========>....................] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
 8416/25000 [=========>....................] - ETA: 43s - loss: 7.6684 - accuracy: 0.4999
 8448/25000 [=========>....................] - ETA: 42s - loss: 7.6702 - accuracy: 0.4998
 8480/25000 [=========>....................] - ETA: 42s - loss: 7.6739 - accuracy: 0.4995
 8512/25000 [=========>....................] - ETA: 42s - loss: 7.6756 - accuracy: 0.4994
 8544/25000 [=========>....................] - ETA: 42s - loss: 7.6738 - accuracy: 0.4995
 8576/25000 [=========>....................] - ETA: 42s - loss: 7.6773 - accuracy: 0.4993
 8608/25000 [=========>....................] - ETA: 42s - loss: 7.6702 - accuracy: 0.4998
 8640/25000 [=========>....................] - ETA: 42s - loss: 7.6666 - accuracy: 0.5000
 8672/25000 [=========>....................] - ETA: 42s - loss: 7.6666 - accuracy: 0.5000
 8704/25000 [=========>....................] - ETA: 42s - loss: 7.6543 - accuracy: 0.5008
 8736/25000 [=========>....................] - ETA: 42s - loss: 7.6631 - accuracy: 0.5002
 8768/25000 [=========>....................] - ETA: 42s - loss: 7.6614 - accuracy: 0.5003
 8800/25000 [=========>....................] - ETA: 41s - loss: 7.6596 - accuracy: 0.5005
 8832/25000 [=========>....................] - ETA: 41s - loss: 7.6579 - accuracy: 0.5006
 8864/25000 [=========>....................] - ETA: 41s - loss: 7.6580 - accuracy: 0.5006
 8896/25000 [=========>....................] - ETA: 41s - loss: 7.6546 - accuracy: 0.5008
 8928/25000 [=========>....................] - ETA: 41s - loss: 7.6580 - accuracy: 0.5006
 8960/25000 [=========>....................] - ETA: 41s - loss: 7.6546 - accuracy: 0.5008
 8992/25000 [=========>....................] - ETA: 41s - loss: 7.6581 - accuracy: 0.5006
 9024/25000 [=========>....................] - ETA: 41s - loss: 7.6581 - accuracy: 0.5006
 9056/25000 [=========>....................] - ETA: 41s - loss: 7.6615 - accuracy: 0.5003
 9088/25000 [=========>....................] - ETA: 41s - loss: 7.6548 - accuracy: 0.5008
 9120/25000 [=========>....................] - ETA: 40s - loss: 7.6616 - accuracy: 0.5003
 9152/25000 [=========>....................] - ETA: 40s - loss: 7.6649 - accuracy: 0.5001
 9184/25000 [==========>...................] - ETA: 40s - loss: 7.6666 - accuracy: 0.5000
 9216/25000 [==========>...................] - ETA: 40s - loss: 7.6699 - accuracy: 0.4998
 9248/25000 [==========>...................] - ETA: 40s - loss: 7.6782 - accuracy: 0.4992
 9280/25000 [==========>...................] - ETA: 40s - loss: 7.6815 - accuracy: 0.4990
 9312/25000 [==========>...................] - ETA: 40s - loss: 7.6847 - accuracy: 0.4988
 9344/25000 [==========>...................] - ETA: 40s - loss: 7.6896 - accuracy: 0.4985
 9376/25000 [==========>...................] - ETA: 40s - loss: 7.6928 - accuracy: 0.4983
 9408/25000 [==========>...................] - ETA: 40s - loss: 7.6960 - accuracy: 0.4981
 9440/25000 [==========>...................] - ETA: 40s - loss: 7.6991 - accuracy: 0.4979
 9472/25000 [==========>...................] - ETA: 39s - loss: 7.6990 - accuracy: 0.4979
 9504/25000 [==========>...................] - ETA: 39s - loss: 7.7021 - accuracy: 0.4977
 9536/25000 [==========>...................] - ETA: 39s - loss: 7.7052 - accuracy: 0.4975
 9568/25000 [==========>...................] - ETA: 39s - loss: 7.7051 - accuracy: 0.4975
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.7034 - accuracy: 0.4976
 9632/25000 [==========>...................] - ETA: 39s - loss: 7.7032 - accuracy: 0.4976
 9664/25000 [==========>...................] - ETA: 39s - loss: 7.7063 - accuracy: 0.4974
 9696/25000 [==========>...................] - ETA: 39s - loss: 7.7062 - accuracy: 0.4974
 9728/25000 [==========>...................] - ETA: 39s - loss: 7.7076 - accuracy: 0.4973
 9760/25000 [==========>...................] - ETA: 39s - loss: 7.7153 - accuracy: 0.4968
 9792/25000 [==========>...................] - ETA: 39s - loss: 7.7105 - accuracy: 0.4971
 9824/25000 [==========>...................] - ETA: 39s - loss: 7.7103 - accuracy: 0.4971
 9856/25000 [==========>...................] - ETA: 38s - loss: 7.7071 - accuracy: 0.4974
 9888/25000 [==========>...................] - ETA: 38s - loss: 7.7054 - accuracy: 0.4975
 9920/25000 [==========>...................] - ETA: 38s - loss: 7.7053 - accuracy: 0.4975
 9952/25000 [==========>...................] - ETA: 38s - loss: 7.7113 - accuracy: 0.4971
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.7050 - accuracy: 0.4975
10016/25000 [===========>..................] - ETA: 38s - loss: 7.7110 - accuracy: 0.4971
10048/25000 [===========>..................] - ETA: 38s - loss: 7.7032 - accuracy: 0.4976
10080/25000 [===========>..................] - ETA: 38s - loss: 7.7031 - accuracy: 0.4976
10112/25000 [===========>..................] - ETA: 38s - loss: 7.7106 - accuracy: 0.4971
10144/25000 [===========>..................] - ETA: 38s - loss: 7.7165 - accuracy: 0.4967
10176/25000 [===========>..................] - ETA: 38s - loss: 7.7209 - accuracy: 0.4965
10208/25000 [===========>..................] - ETA: 37s - loss: 7.7222 - accuracy: 0.4964
10240/25000 [===========>..................] - ETA: 37s - loss: 7.7190 - accuracy: 0.4966
10272/25000 [===========>..................] - ETA: 37s - loss: 7.7189 - accuracy: 0.4966
10304/25000 [===========>..................] - ETA: 37s - loss: 7.7157 - accuracy: 0.4968
10336/25000 [===========>..................] - ETA: 37s - loss: 7.7215 - accuracy: 0.4964
10368/25000 [===========>..................] - ETA: 37s - loss: 7.7243 - accuracy: 0.4962
10400/25000 [===========>..................] - ETA: 37s - loss: 7.7271 - accuracy: 0.4961
10432/25000 [===========>..................] - ETA: 37s - loss: 7.7269 - accuracy: 0.4961
10464/25000 [===========>..................] - ETA: 37s - loss: 7.7252 - accuracy: 0.4962
10496/25000 [===========>..................] - ETA: 37s - loss: 7.7338 - accuracy: 0.4956
10528/25000 [===========>..................] - ETA: 37s - loss: 7.7278 - accuracy: 0.4960
10560/25000 [===========>..................] - ETA: 37s - loss: 7.7276 - accuracy: 0.4960
10592/25000 [===========>..................] - ETA: 36s - loss: 7.7216 - accuracy: 0.4964
10624/25000 [===========>..................] - ETA: 36s - loss: 7.7200 - accuracy: 0.4965
10656/25000 [===========>..................] - ETA: 36s - loss: 7.7199 - accuracy: 0.4965
10688/25000 [===========>..................] - ETA: 36s - loss: 7.7197 - accuracy: 0.4965
10720/25000 [===========>..................] - ETA: 36s - loss: 7.7138 - accuracy: 0.4969
10752/25000 [===========>..................] - ETA: 36s - loss: 7.7108 - accuracy: 0.4971
10784/25000 [===========>..................] - ETA: 36s - loss: 7.7079 - accuracy: 0.4973
10816/25000 [===========>..................] - ETA: 36s - loss: 7.7035 - accuracy: 0.4976
10848/25000 [============>.................] - ETA: 36s - loss: 7.6935 - accuracy: 0.4982
10880/25000 [============>.................] - ETA: 36s - loss: 7.6948 - accuracy: 0.4982
10912/25000 [============>.................] - ETA: 36s - loss: 7.6961 - accuracy: 0.4981
10944/25000 [============>.................] - ETA: 35s - loss: 7.6988 - accuracy: 0.4979
10976/25000 [============>.................] - ETA: 35s - loss: 7.7015 - accuracy: 0.4977
11008/25000 [============>.................] - ETA: 35s - loss: 7.6973 - accuracy: 0.4980
11040/25000 [============>.................] - ETA: 35s - loss: 7.6916 - accuracy: 0.4984
11072/25000 [============>.................] - ETA: 35s - loss: 7.6915 - accuracy: 0.4984
11104/25000 [============>.................] - ETA: 35s - loss: 7.6901 - accuracy: 0.4985
11136/25000 [============>.................] - ETA: 35s - loss: 7.6900 - accuracy: 0.4985
11168/25000 [============>.................] - ETA: 35s - loss: 7.6968 - accuracy: 0.4980
11200/25000 [============>.................] - ETA: 35s - loss: 7.6995 - accuracy: 0.4979
11232/25000 [============>.................] - ETA: 35s - loss: 7.6980 - accuracy: 0.4980
11264/25000 [============>.................] - ETA: 35s - loss: 7.6979 - accuracy: 0.4980
11296/25000 [============>.................] - ETA: 35s - loss: 7.7006 - accuracy: 0.4978
11328/25000 [============>.................] - ETA: 34s - loss: 7.6991 - accuracy: 0.4979
11360/25000 [============>.................] - ETA: 34s - loss: 7.6990 - accuracy: 0.4979
11392/25000 [============>.................] - ETA: 34s - loss: 7.7083 - accuracy: 0.4973
11424/25000 [============>.................] - ETA: 34s - loss: 7.7096 - accuracy: 0.4972
11456/25000 [============>.................] - ETA: 34s - loss: 7.7188 - accuracy: 0.4966
11488/25000 [============>.................] - ETA: 34s - loss: 7.7227 - accuracy: 0.4963
11520/25000 [============>.................] - ETA: 34s - loss: 7.7225 - accuracy: 0.4964
11552/25000 [============>.................] - ETA: 34s - loss: 7.7171 - accuracy: 0.4967
11584/25000 [============>.................] - ETA: 34s - loss: 7.7182 - accuracy: 0.4966
11616/25000 [============>.................] - ETA: 34s - loss: 7.7207 - accuracy: 0.4965
11648/25000 [============>.................] - ETA: 34s - loss: 7.7245 - accuracy: 0.4962
11680/25000 [=============>................] - ETA: 33s - loss: 7.7218 - accuracy: 0.4964
11712/25000 [=============>................] - ETA: 33s - loss: 7.7164 - accuracy: 0.4968
11744/25000 [=============>................] - ETA: 33s - loss: 7.7123 - accuracy: 0.4970
11776/25000 [=============>................] - ETA: 33s - loss: 7.7148 - accuracy: 0.4969
11808/25000 [=============>................] - ETA: 33s - loss: 7.7095 - accuracy: 0.4972
11840/25000 [=============>................] - ETA: 33s - loss: 7.7132 - accuracy: 0.4970
11872/25000 [=============>................] - ETA: 33s - loss: 7.7054 - accuracy: 0.4975
11904/25000 [=============>................] - ETA: 33s - loss: 7.6988 - accuracy: 0.4979
11936/25000 [=============>................] - ETA: 33s - loss: 7.7026 - accuracy: 0.4977
11968/25000 [=============>................] - ETA: 33s - loss: 7.6999 - accuracy: 0.4978
12000/25000 [=============>................] - ETA: 33s - loss: 7.7024 - accuracy: 0.4977
12032/25000 [=============>................] - ETA: 33s - loss: 7.7061 - accuracy: 0.4974
12064/25000 [=============>................] - ETA: 32s - loss: 7.7060 - accuracy: 0.4974
12096/25000 [=============>................] - ETA: 32s - loss: 7.7034 - accuracy: 0.4976
12128/25000 [=============>................] - ETA: 32s - loss: 7.7008 - accuracy: 0.4978
12160/25000 [=============>................] - ETA: 32s - loss: 7.6956 - accuracy: 0.4981
12192/25000 [=============>................] - ETA: 32s - loss: 7.6955 - accuracy: 0.4981
12224/25000 [=============>................] - ETA: 32s - loss: 7.6980 - accuracy: 0.4980
12256/25000 [=============>................] - ETA: 32s - loss: 7.7016 - accuracy: 0.4977
12288/25000 [=============>................] - ETA: 32s - loss: 7.7016 - accuracy: 0.4977
12320/25000 [=============>................] - ETA: 32s - loss: 7.6990 - accuracy: 0.4979
12352/25000 [=============>................] - ETA: 32s - loss: 7.7051 - accuracy: 0.4975
12384/25000 [=============>................] - ETA: 32s - loss: 7.7062 - accuracy: 0.4974
12416/25000 [=============>................] - ETA: 32s - loss: 7.7037 - accuracy: 0.4976
12448/25000 [=============>................] - ETA: 31s - loss: 7.7048 - accuracy: 0.4975
12480/25000 [=============>................] - ETA: 31s - loss: 7.7096 - accuracy: 0.4972
12512/25000 [==============>...............] - ETA: 31s - loss: 7.7071 - accuracy: 0.4974
12544/25000 [==============>...............] - ETA: 31s - loss: 7.7070 - accuracy: 0.4974
12576/25000 [==============>...............] - ETA: 31s - loss: 7.6983 - accuracy: 0.4979
12608/25000 [==============>...............] - ETA: 31s - loss: 7.7007 - accuracy: 0.4978
12640/25000 [==============>...............] - ETA: 31s - loss: 7.6994 - accuracy: 0.4979
12672/25000 [==============>...............] - ETA: 31s - loss: 7.6969 - accuracy: 0.4980
12704/25000 [==============>...............] - ETA: 31s - loss: 7.7040 - accuracy: 0.4976
12736/25000 [==============>...............] - ETA: 31s - loss: 7.7039 - accuracy: 0.4976
12768/25000 [==============>...............] - ETA: 31s - loss: 7.7002 - accuracy: 0.4978
12800/25000 [==============>...............] - ETA: 31s - loss: 7.7014 - accuracy: 0.4977
12832/25000 [==============>...............] - ETA: 30s - loss: 7.7013 - accuracy: 0.4977
12864/25000 [==============>...............] - ETA: 30s - loss: 7.7071 - accuracy: 0.4974
12896/25000 [==============>...............] - ETA: 30s - loss: 7.7094 - accuracy: 0.4972
12928/25000 [==============>...............] - ETA: 30s - loss: 7.7141 - accuracy: 0.4969
12960/25000 [==============>...............] - ETA: 30s - loss: 7.7116 - accuracy: 0.4971
12992/25000 [==============>...............] - ETA: 30s - loss: 7.7091 - accuracy: 0.4972
13024/25000 [==============>...............] - ETA: 30s - loss: 7.7090 - accuracy: 0.4972
13056/25000 [==============>...............] - ETA: 30s - loss: 7.7042 - accuracy: 0.4975
13088/25000 [==============>...............] - ETA: 30s - loss: 7.7065 - accuracy: 0.4974
13120/25000 [==============>...............] - ETA: 30s - loss: 7.7028 - accuracy: 0.4976
13152/25000 [==============>...............] - ETA: 30s - loss: 7.7016 - accuracy: 0.4977
13184/25000 [==============>...............] - ETA: 29s - loss: 7.7038 - accuracy: 0.4976
13216/25000 [==============>...............] - ETA: 29s - loss: 7.7061 - accuracy: 0.4974
13248/25000 [==============>...............] - ETA: 29s - loss: 7.7060 - accuracy: 0.4974
13280/25000 [==============>...............] - ETA: 29s - loss: 7.7047 - accuracy: 0.4975
13312/25000 [==============>...............] - ETA: 29s - loss: 7.7081 - accuracy: 0.4973
13344/25000 [===============>..............] - ETA: 29s - loss: 7.7091 - accuracy: 0.4972
13376/25000 [===============>..............] - ETA: 29s - loss: 7.7056 - accuracy: 0.4975
13408/25000 [===============>..............] - ETA: 29s - loss: 7.7089 - accuracy: 0.4972
13440/25000 [===============>..............] - ETA: 29s - loss: 7.7043 - accuracy: 0.4975
13472/25000 [===============>..............] - ETA: 29s - loss: 7.7076 - accuracy: 0.4973
13504/25000 [===============>..............] - ETA: 29s - loss: 7.7064 - accuracy: 0.4974
13536/25000 [===============>..............] - ETA: 29s - loss: 7.7119 - accuracy: 0.4970
13568/25000 [===============>..............] - ETA: 28s - loss: 7.7141 - accuracy: 0.4969
13600/25000 [===============>..............] - ETA: 28s - loss: 7.7128 - accuracy: 0.4970
13632/25000 [===============>..............] - ETA: 28s - loss: 7.7105 - accuracy: 0.4971
13664/25000 [===============>..............] - ETA: 28s - loss: 7.7070 - accuracy: 0.4974
13696/25000 [===============>..............] - ETA: 28s - loss: 7.7047 - accuracy: 0.4975
13728/25000 [===============>..............] - ETA: 28s - loss: 7.7057 - accuracy: 0.4975
13760/25000 [===============>..............] - ETA: 28s - loss: 7.7056 - accuracy: 0.4975
13792/25000 [===============>..............] - ETA: 28s - loss: 7.7066 - accuracy: 0.4974
13824/25000 [===============>..............] - ETA: 28s - loss: 7.7088 - accuracy: 0.4973
13856/25000 [===============>..............] - ETA: 28s - loss: 7.7109 - accuracy: 0.4971
13888/25000 [===============>..............] - ETA: 28s - loss: 7.7152 - accuracy: 0.4968
13920/25000 [===============>..............] - ETA: 28s - loss: 7.7162 - accuracy: 0.4968
13952/25000 [===============>..............] - ETA: 27s - loss: 7.7194 - accuracy: 0.4966
13984/25000 [===============>..............] - ETA: 27s - loss: 7.7182 - accuracy: 0.4966
14016/25000 [===============>..............] - ETA: 27s - loss: 7.7202 - accuracy: 0.4965
14048/25000 [===============>..............] - ETA: 27s - loss: 7.7168 - accuracy: 0.4967
14080/25000 [===============>..............] - ETA: 27s - loss: 7.7189 - accuracy: 0.4966
14112/25000 [===============>..............] - ETA: 27s - loss: 7.7188 - accuracy: 0.4966
14144/25000 [===============>..............] - ETA: 27s - loss: 7.7197 - accuracy: 0.4965
14176/25000 [================>.............] - ETA: 27s - loss: 7.7250 - accuracy: 0.4962
14208/25000 [================>.............] - ETA: 27s - loss: 7.7271 - accuracy: 0.4961
14240/25000 [================>.............] - ETA: 27s - loss: 7.7323 - accuracy: 0.4957
14272/25000 [================>.............] - ETA: 27s - loss: 7.7279 - accuracy: 0.4960
14304/25000 [================>.............] - ETA: 27s - loss: 7.7213 - accuracy: 0.4964
14336/25000 [================>.............] - ETA: 26s - loss: 7.7201 - accuracy: 0.4965
14368/25000 [================>.............] - ETA: 26s - loss: 7.7146 - accuracy: 0.4969
14400/25000 [================>.............] - ETA: 26s - loss: 7.7135 - accuracy: 0.4969
14432/25000 [================>.............] - ETA: 26s - loss: 7.7112 - accuracy: 0.4971
14464/25000 [================>.............] - ETA: 26s - loss: 7.7143 - accuracy: 0.4969
14496/25000 [================>.............] - ETA: 26s - loss: 7.7163 - accuracy: 0.4968
14528/25000 [================>.............] - ETA: 26s - loss: 7.7162 - accuracy: 0.4968
14560/25000 [================>.............] - ETA: 26s - loss: 7.7161 - accuracy: 0.4968
14592/25000 [================>.............] - ETA: 26s - loss: 7.7150 - accuracy: 0.4968
14624/25000 [================>.............] - ETA: 26s - loss: 7.7138 - accuracy: 0.4969
14656/25000 [================>.............] - ETA: 26s - loss: 7.7168 - accuracy: 0.4967
14688/25000 [================>.............] - ETA: 26s - loss: 7.7167 - accuracy: 0.4967
14720/25000 [================>.............] - ETA: 25s - loss: 7.7104 - accuracy: 0.4971
14752/25000 [================>.............] - ETA: 25s - loss: 7.7103 - accuracy: 0.4972
14784/25000 [================>.............] - ETA: 25s - loss: 7.7091 - accuracy: 0.4972
14816/25000 [================>.............] - ETA: 25s - loss: 7.7070 - accuracy: 0.4974
14848/25000 [================>.............] - ETA: 25s - loss: 7.7038 - accuracy: 0.4976
14880/25000 [================>.............] - ETA: 25s - loss: 7.7027 - accuracy: 0.4976
14912/25000 [================>.............] - ETA: 25s - loss: 7.7036 - accuracy: 0.4976
14944/25000 [================>.............] - ETA: 25s - loss: 7.7025 - accuracy: 0.4977
14976/25000 [================>.............] - ETA: 25s - loss: 7.7055 - accuracy: 0.4975
15008/25000 [=================>............] - ETA: 25s - loss: 7.7065 - accuracy: 0.4974
15040/25000 [=================>............] - ETA: 25s - loss: 7.7033 - accuracy: 0.4976
15072/25000 [=================>............] - ETA: 25s - loss: 7.7002 - accuracy: 0.4978
15104/25000 [=================>............] - ETA: 24s - loss: 7.7011 - accuracy: 0.4977
15136/25000 [=================>............] - ETA: 24s - loss: 7.7031 - accuracy: 0.4976
15168/25000 [=================>............] - ETA: 24s - loss: 7.7071 - accuracy: 0.4974
15200/25000 [=================>............] - ETA: 24s - loss: 7.7050 - accuracy: 0.4975
15232/25000 [=================>............] - ETA: 24s - loss: 7.7079 - accuracy: 0.4973
15264/25000 [=================>............] - ETA: 24s - loss: 7.7118 - accuracy: 0.4971
15296/25000 [=================>............] - ETA: 24s - loss: 7.7117 - accuracy: 0.4971
15328/25000 [=================>............] - ETA: 24s - loss: 7.7136 - accuracy: 0.4969
15360/25000 [=================>............] - ETA: 24s - loss: 7.7165 - accuracy: 0.4967
15392/25000 [=================>............] - ETA: 24s - loss: 7.7105 - accuracy: 0.4971
15424/25000 [=================>............] - ETA: 24s - loss: 7.7123 - accuracy: 0.4970
15456/25000 [=================>............] - ETA: 24s - loss: 7.7132 - accuracy: 0.4970
15488/25000 [=================>............] - ETA: 24s - loss: 7.7092 - accuracy: 0.4972
15520/25000 [=================>............] - ETA: 23s - loss: 7.7052 - accuracy: 0.4975
15552/25000 [=================>............] - ETA: 23s - loss: 7.7041 - accuracy: 0.4976
15584/25000 [=================>............] - ETA: 23s - loss: 7.7020 - accuracy: 0.4977
15616/25000 [=================>............] - ETA: 23s - loss: 7.7039 - accuracy: 0.4976
15648/25000 [=================>............] - ETA: 23s - loss: 7.7048 - accuracy: 0.4975
15680/25000 [=================>............] - ETA: 23s - loss: 7.7057 - accuracy: 0.4974
15712/25000 [=================>............] - ETA: 23s - loss: 7.7037 - accuracy: 0.4976
15744/25000 [=================>............] - ETA: 23s - loss: 7.7027 - accuracy: 0.4976
15776/25000 [=================>............] - ETA: 23s - loss: 7.7036 - accuracy: 0.4976
15808/25000 [=================>............] - ETA: 23s - loss: 7.7044 - accuracy: 0.4975
15840/25000 [==================>...........] - ETA: 23s - loss: 7.7005 - accuracy: 0.4978
15872/25000 [==================>...........] - ETA: 23s - loss: 7.6956 - accuracy: 0.4981
15904/25000 [==================>...........] - ETA: 22s - loss: 7.6965 - accuracy: 0.4981
15936/25000 [==================>...........] - ETA: 22s - loss: 7.6964 - accuracy: 0.4981
15968/25000 [==================>...........] - ETA: 22s - loss: 7.6954 - accuracy: 0.4981
16000/25000 [==================>...........] - ETA: 22s - loss: 7.7002 - accuracy: 0.4978
16032/25000 [==================>...........] - ETA: 22s - loss: 7.6963 - accuracy: 0.4981
16064/25000 [==================>...........] - ETA: 22s - loss: 7.6981 - accuracy: 0.4979
16096/25000 [==================>...........] - ETA: 22s - loss: 7.6981 - accuracy: 0.4979
16128/25000 [==================>...........] - ETA: 22s - loss: 7.6980 - accuracy: 0.4980
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6960 - accuracy: 0.4981
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6931 - accuracy: 0.4983
16224/25000 [==================>...........] - ETA: 22s - loss: 7.6931 - accuracy: 0.4983
16256/25000 [==================>...........] - ETA: 22s - loss: 7.6987 - accuracy: 0.4979
16288/25000 [==================>...........] - ETA: 21s - loss: 7.7005 - accuracy: 0.4978
16320/25000 [==================>...........] - ETA: 21s - loss: 7.6986 - accuracy: 0.4979
16352/25000 [==================>...........] - ETA: 21s - loss: 7.6948 - accuracy: 0.4982
16384/25000 [==================>...........] - ETA: 21s - loss: 7.6975 - accuracy: 0.4980
16416/25000 [==================>...........] - ETA: 21s - loss: 7.6956 - accuracy: 0.4981
16448/25000 [==================>...........] - ETA: 21s - loss: 7.6974 - accuracy: 0.4980
16480/25000 [==================>...........] - ETA: 21s - loss: 7.7029 - accuracy: 0.4976
16512/25000 [==================>...........] - ETA: 21s - loss: 7.7028 - accuracy: 0.4976
16544/25000 [==================>...........] - ETA: 21s - loss: 7.7028 - accuracy: 0.4976
16576/25000 [==================>...........] - ETA: 21s - loss: 7.6990 - accuracy: 0.4979
16608/25000 [==================>...........] - ETA: 21s - loss: 7.6971 - accuracy: 0.4980
16640/25000 [==================>...........] - ETA: 21s - loss: 7.6998 - accuracy: 0.4978
16672/25000 [===================>..........] - ETA: 20s - loss: 7.6951 - accuracy: 0.4981
16704/25000 [===================>..........] - ETA: 20s - loss: 7.6960 - accuracy: 0.4981
16736/25000 [===================>..........] - ETA: 20s - loss: 7.6987 - accuracy: 0.4979
16768/25000 [===================>..........] - ETA: 20s - loss: 7.6959 - accuracy: 0.4981
16800/25000 [===================>..........] - ETA: 20s - loss: 7.6967 - accuracy: 0.4980
16832/25000 [===================>..........] - ETA: 20s - loss: 7.6958 - accuracy: 0.4981
16864/25000 [===================>..........] - ETA: 20s - loss: 7.6957 - accuracy: 0.4981
16896/25000 [===================>..........] - ETA: 20s - loss: 7.6938 - accuracy: 0.4982
16928/25000 [===================>..........] - ETA: 20s - loss: 7.6938 - accuracy: 0.4982
16960/25000 [===================>..........] - ETA: 20s - loss: 7.6928 - accuracy: 0.4983
16992/25000 [===================>..........] - ETA: 20s - loss: 7.6883 - accuracy: 0.4986
17024/25000 [===================>..........] - ETA: 20s - loss: 7.6837 - accuracy: 0.4989
17056/25000 [===================>..........] - ETA: 19s - loss: 7.6837 - accuracy: 0.4989
17088/25000 [===================>..........] - ETA: 19s - loss: 7.6828 - accuracy: 0.4989
17120/25000 [===================>..........] - ETA: 19s - loss: 7.6792 - accuracy: 0.4992
17152/25000 [===================>..........] - ETA: 19s - loss: 7.6773 - accuracy: 0.4993
17184/25000 [===================>..........] - ETA: 19s - loss: 7.6782 - accuracy: 0.4992
17216/25000 [===================>..........] - ETA: 19s - loss: 7.6773 - accuracy: 0.4993
17248/25000 [===================>..........] - ETA: 19s - loss: 7.6791 - accuracy: 0.4992
17280/25000 [===================>..........] - ETA: 19s - loss: 7.6799 - accuracy: 0.4991
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6772 - accuracy: 0.4993
17344/25000 [===================>..........] - ETA: 19s - loss: 7.6772 - accuracy: 0.4993
17376/25000 [===================>..........] - ETA: 19s - loss: 7.6746 - accuracy: 0.4995
17408/25000 [===================>..........] - ETA: 19s - loss: 7.6701 - accuracy: 0.4998
17440/25000 [===================>..........] - ETA: 19s - loss: 7.6701 - accuracy: 0.4998
17472/25000 [===================>..........] - ETA: 18s - loss: 7.6701 - accuracy: 0.4998
17504/25000 [====================>.........] - ETA: 18s - loss: 7.6710 - accuracy: 0.4997
17536/25000 [====================>.........] - ETA: 18s - loss: 7.6675 - accuracy: 0.4999
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6675 - accuracy: 0.4999
17600/25000 [====================>.........] - ETA: 18s - loss: 7.6657 - accuracy: 0.5001
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6666 - accuracy: 0.5000
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6675 - accuracy: 0.4999
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6666 - accuracy: 0.5000
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6683 - accuracy: 0.4999
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6701 - accuracy: 0.4998
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6709 - accuracy: 0.4997
17824/25000 [====================>.........] - ETA: 18s - loss: 7.6709 - accuracy: 0.4997
17856/25000 [====================>.........] - ETA: 17s - loss: 7.6726 - accuracy: 0.4996
17888/25000 [====================>.........] - ETA: 17s - loss: 7.6692 - accuracy: 0.4998
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6709 - accuracy: 0.4997
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6717 - accuracy: 0.4997
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6717 - accuracy: 0.4997
18016/25000 [====================>.........] - ETA: 17s - loss: 7.6717 - accuracy: 0.4997
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6743 - accuracy: 0.4995
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6768 - accuracy: 0.4993
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6793 - accuracy: 0.4992
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6759 - accuracy: 0.4994
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6734 - accuracy: 0.4996
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6750 - accuracy: 0.4995
18240/25000 [====================>.........] - ETA: 16s - loss: 7.6700 - accuracy: 0.4998
18272/25000 [====================>.........] - ETA: 16s - loss: 7.6675 - accuracy: 0.4999
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6700 - accuracy: 0.4998
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6666 - accuracy: 0.5000
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6649 - accuracy: 0.5001
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6641 - accuracy: 0.5002
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6641 - accuracy: 0.5002
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6633 - accuracy: 0.5002
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6650 - accuracy: 0.5001
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6650 - accuracy: 0.5001
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6617 - accuracy: 0.5003
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6625 - accuracy: 0.5003
18624/25000 [=====================>........] - ETA: 16s - loss: 7.6617 - accuracy: 0.5003
18656/25000 [=====================>........] - ETA: 15s - loss: 7.6683 - accuracy: 0.4999
18688/25000 [=====================>........] - ETA: 15s - loss: 7.6683 - accuracy: 0.4999
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6724 - accuracy: 0.4996
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6699 - accuracy: 0.4998
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6601 - accuracy: 0.5004
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6634 - accuracy: 0.5002
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6617 - accuracy: 0.5003
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6593 - accuracy: 0.5005
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6601 - accuracy: 0.5004
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6593 - accuracy: 0.5005
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6586 - accuracy: 0.5005
19040/25000 [=====================>........] - ETA: 14s - loss: 7.6562 - accuracy: 0.5007
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6578 - accuracy: 0.5006
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6578 - accuracy: 0.5006
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6594 - accuracy: 0.5005
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6658 - accuracy: 0.5001
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6626 - accuracy: 0.5003
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6626 - accuracy: 0.5003
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6595 - accuracy: 0.5005
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6611 - accuracy: 0.5004
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6611 - accuracy: 0.5004
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6619 - accuracy: 0.5003
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6587 - accuracy: 0.5005
19424/25000 [======================>.......] - ETA: 13s - loss: 7.6571 - accuracy: 0.5006
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6587 - accuracy: 0.5005
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6580 - accuracy: 0.5006
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6588 - accuracy: 0.5005
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6556 - accuracy: 0.5007
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6541 - accuracy: 0.5008
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6525 - accuracy: 0.5009
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6502 - accuracy: 0.5011
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6503 - accuracy: 0.5011
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6526 - accuracy: 0.5009
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6495 - accuracy: 0.5011
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6480 - accuracy: 0.5012
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6473 - accuracy: 0.5013
19840/25000 [======================>.......] - ETA: 12s - loss: 7.6465 - accuracy: 0.5013
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6450 - accuracy: 0.5014
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6443 - accuracy: 0.5015
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6466 - accuracy: 0.5013
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6467 - accuracy: 0.5013
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6444 - accuracy: 0.5015
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6444 - accuracy: 0.5014
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6429 - accuracy: 0.5015
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6437 - accuracy: 0.5015
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6407 - accuracy: 0.5017
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6423 - accuracy: 0.5016
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6446 - accuracy: 0.5014
20224/25000 [=======================>......] - ETA: 11s - loss: 7.6454 - accuracy: 0.5014
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6469 - accuracy: 0.5013
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6455 - accuracy: 0.5014
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6470 - accuracy: 0.5013
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6470 - accuracy: 0.5013
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6493 - accuracy: 0.5011
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6508 - accuracy: 0.5010
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6554 - accuracy: 0.5007
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6561 - accuracy: 0.5007
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6569 - accuracy: 0.5006
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6569 - accuracy: 0.5006
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6554 - accuracy: 0.5007
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6532 - accuracy: 0.5009
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6503 - accuracy: 0.5011
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6481 - accuracy: 0.5012
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6474 - accuracy: 0.5013
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6489 - accuracy: 0.5012
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6496 - accuracy: 0.5011
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6497 - accuracy: 0.5011
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6467 - accuracy: 0.5013
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6497 - accuracy: 0.5011
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6512 - accuracy: 0.5010
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6512 - accuracy: 0.5010
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6483 - accuracy: 0.5012
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6462 - accuracy: 0.5013
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6433 - accuracy: 0.5015 
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6440 - accuracy: 0.5015
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6477 - accuracy: 0.5012
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6477 - accuracy: 0.5012
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6499 - accuracy: 0.5011
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6500 - accuracy: 0.5011
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6507 - accuracy: 0.5010
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6515 - accuracy: 0.5010
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6493 - accuracy: 0.5011
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6479 - accuracy: 0.5012
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6508 - accuracy: 0.5010
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6523 - accuracy: 0.5009
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6516 - accuracy: 0.5010
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6523 - accuracy: 0.5009
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6566 - accuracy: 0.5007
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6552 - accuracy: 0.5007
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6552 - accuracy: 0.5007
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6552 - accuracy: 0.5007
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6546 - accuracy: 0.5008
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6553 - accuracy: 0.5007
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6539 - accuracy: 0.5008
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6539 - accuracy: 0.5008
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6560 - accuracy: 0.5007
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6546 - accuracy: 0.5008
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6504 - accuracy: 0.5011
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6484 - accuracy: 0.5012
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6484 - accuracy: 0.5012
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6463 - accuracy: 0.5013
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6463 - accuracy: 0.5013
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6478 - accuracy: 0.5012
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6506 - accuracy: 0.5010
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6492 - accuracy: 0.5011
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6527 - accuracy: 0.5009
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6520 - accuracy: 0.5010
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6555 - accuracy: 0.5007
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6562 - accuracy: 0.5007
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6576 - accuracy: 0.5006
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6570 - accuracy: 0.5006
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6611 - accuracy: 0.5004
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6604 - accuracy: 0.5004
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6611 - accuracy: 0.5004
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6598 - accuracy: 0.5004
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6632 - accuracy: 0.5002
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6659 - accuracy: 0.5000
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6659 - accuracy: 0.5000
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6700 - accuracy: 0.4998
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6666 - accuracy: 0.5000
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6680 - accuracy: 0.4999
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6673 - accuracy: 0.5000
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6653 - accuracy: 0.5001
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6687 - accuracy: 0.4999
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6646 - accuracy: 0.5001
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6639 - accuracy: 0.5002
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6626 - accuracy: 0.5003
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6639 - accuracy: 0.5002
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6606 - accuracy: 0.5004
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6639 - accuracy: 0.5002
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6659 - accuracy: 0.5000
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6680 - accuracy: 0.4999
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6686 - accuracy: 0.4999
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6693 - accuracy: 0.4998
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6726 - accuracy: 0.4996
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6726 - accuracy: 0.4996
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6712 - accuracy: 0.4997
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6706 - accuracy: 0.4997
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6686 - accuracy: 0.4999
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6647 - accuracy: 0.5001
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6621 - accuracy: 0.5003
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6608 - accuracy: 0.5004
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6608 - accuracy: 0.5004
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6582 - accuracy: 0.5005
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6602 - accuracy: 0.5004
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6582 - accuracy: 0.5005
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6583 - accuracy: 0.5005
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6589 - accuracy: 0.5005
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6564 - accuracy: 0.5007
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6577 - accuracy: 0.5006
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6570 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6551 - accuracy: 0.5008
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6526 - accuracy: 0.5009
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6501 - accuracy: 0.5011
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6533 - accuracy: 0.5009
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6539 - accuracy: 0.5008
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6565 - accuracy: 0.5007
24192/25000 [============================>.] - ETA: 2s - loss: 7.6552 - accuracy: 0.5007
24224/25000 [============================>.] - ETA: 1s - loss: 7.6527 - accuracy: 0.5009
24256/25000 [============================>.] - ETA: 1s - loss: 7.6527 - accuracy: 0.5009
24288/25000 [============================>.] - ETA: 1s - loss: 7.6527 - accuracy: 0.5009
24320/25000 [============================>.] - ETA: 1s - loss: 7.6534 - accuracy: 0.5009
24352/25000 [============================>.] - ETA: 1s - loss: 7.6540 - accuracy: 0.5008
24384/25000 [============================>.] - ETA: 1s - loss: 7.6566 - accuracy: 0.5007
24416/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24448/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24480/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24512/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5004
24544/25000 [============================>.] - ETA: 1s - loss: 7.6585 - accuracy: 0.5005
24576/25000 [============================>.] - ETA: 1s - loss: 7.6598 - accuracy: 0.5004
24608/25000 [============================>.] - ETA: 0s - loss: 7.6585 - accuracy: 0.5005
24640/25000 [============================>.] - ETA: 0s - loss: 7.6592 - accuracy: 0.5005
24672/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
24704/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24736/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24768/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24800/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24832/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24864/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24896/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24928/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24960/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 74s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f1b19571a58> 

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
 [ 0.03464749  0.05017673 -0.01933245 -0.03724937  0.03488103  0.01434531]
 [ 0.20420744  0.1206034  -0.14331481  0.13477008  0.01684503 -0.01464064]
 [ 0.11488441  0.1526355  -0.04178277 -0.16225302  0.07148037  0.05018704]
 [ 0.17687014  0.20758288  0.08908622 -0.0933501   0.08984999 -0.09302241]
 [ 0.03036438  0.15253487  0.3052364  -0.0769878  -0.43795851  0.05555905]
 [-0.05227506  0.21711031  0.40246499  0.1038297  -0.12773201  0.18347554]
 [ 0.21338747  0.47757506 -0.35580456  0.40308654 -0.40231609  0.58248538]
 [ 0.34363562  0.44311413  0.16359735  0.28287578  0.29554623  0.29402199]
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
{'loss': 0.41512393206357956, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-12 05:16:57.497619: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.4168013110756874, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-12 05:16:58.604855: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
	Data preprocessing and feature engineering runtime = 0.25s ...
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:49<01:14, 25.00s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
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
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1813454118919439, 'embedding_size_factor': 0.6084248817010665, 'layers.choice': 1, 'learning_rate': 0.004093933530686447, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 0.010085506989850328} and reward: 0.3578
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xc76S\x92\xad\x82\x9dX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe3x7u\x1fQ\xeaX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?p\xc4\xcc\xea\x05O\x10X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x84\xa7\xb5\xd5z\xda<u.' and reward: 0.3578
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xc76S\x92\xad\x82\x9dX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe3x7u\x1fQ\xeaX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?p\xc4\xcc\xea\x05O\x10X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x84\xa7\xb5\xd5z\xda<u.' and reward: 0.3578
 60%|██████    | 3/5 [01:52<01:12, 36.39s/it] 60%|██████    | 3/5 [01:52<01:15, 37.65s/it]
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
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.11351531237137344, 'embedding_size_factor': 0.7034322736297254, 'layers.choice': 3, 'learning_rate': 0.00785751220297707, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 2.9801306521571927e-09} and reward: 0.392
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xbd\x0fV\xea:\xf3.X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6\x82\x84fF\x19XX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?\x80\x17\x99o\x84\x85TX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>)\x99`im\x1f+u.' and reward: 0.392
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xbd\x0fV\xea:\xf3.X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6\x82\x84fF\x19XX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?\x80\x17\x99o\x84\x85TX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>)\x99`im\x1f+u.' and reward: 0.392
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 164.7504584789276
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.11351531237137344, 'embedding_size_factor': 0.7034322736297254, 'layers.choice': 3, 'learning_rate': 0.00785751220297707, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 2.9801306521571927e-09}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -47.3s of remaining time.
Ensemble size: 38
Ensemble weights: 
[0.55263158 0.36842105 0.07894737]
	0.3994	 = Validation accuracy score
	1.04s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 168.38s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
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

