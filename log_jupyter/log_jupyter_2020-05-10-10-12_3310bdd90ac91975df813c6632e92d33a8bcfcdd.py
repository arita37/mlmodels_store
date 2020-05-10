
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_jupyter GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/3310bdd90ac91975df813c6632e92d33a8bcfcdd', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '3310bdd90ac91975df813c6632e92d33a8bcfcdd', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/3310bdd90ac91975df813c6632e92d33a8bcfcdd

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/3310bdd90ac91975df813c6632e92d33a8bcfcdd

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
 3424256/17464789 [====>.........................] - ETA: 0s
11223040/17464789 [==================>...........] - ETA: 0s
16285696/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 10:12:51.458928: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 10:12:51.462445: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-10 10:12:51.463030: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5646463ba1d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 10:12:51.463041: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 3:43 - loss: 9.1041 - accuracy: 0.4062
   64/25000 [..............................] - ETA: 2:20 - loss: 7.6666 - accuracy: 0.5000
   96/25000 [..............................] - ETA: 1:52 - loss: 7.8263 - accuracy: 0.4896
  128/25000 [..............................] - ETA: 1:36 - loss: 7.0677 - accuracy: 0.5391
  160/25000 [..............................] - ETA: 1:28 - loss: 7.4750 - accuracy: 0.5125
  192/25000 [..............................] - ETA: 1:21 - loss: 7.8263 - accuracy: 0.4896
  224/25000 [..............................] - ETA: 1:16 - loss: 7.8035 - accuracy: 0.4911
  256/25000 [..............................] - ETA: 1:13 - loss: 7.7864 - accuracy: 0.4922
  288/25000 [..............................] - ETA: 1:12 - loss: 7.9328 - accuracy: 0.4826
  320/25000 [..............................] - ETA: 1:10 - loss: 8.0979 - accuracy: 0.4719
  352/25000 [..............................] - ETA: 1:08 - loss: 8.2329 - accuracy: 0.4631
  384/25000 [..............................] - ETA: 1:06 - loss: 8.1059 - accuracy: 0.4714
  416/25000 [..............................] - ETA: 1:05 - loss: 8.1089 - accuracy: 0.4712
  448/25000 [..............................] - ETA: 1:04 - loss: 8.0431 - accuracy: 0.4754
  480/25000 [..............................] - ETA: 1:03 - loss: 8.0180 - accuracy: 0.4771
  512/25000 [..............................] - ETA: 1:02 - loss: 7.9361 - accuracy: 0.4824
  544/25000 [..............................] - ETA: 1:01 - loss: 7.9203 - accuracy: 0.4835
  576/25000 [..............................] - ETA: 1:01 - loss: 7.9328 - accuracy: 0.4826
  608/25000 [..............................] - ETA: 1:01 - loss: 8.0449 - accuracy: 0.4753
  640/25000 [..............................] - ETA: 1:00 - loss: 8.0020 - accuracy: 0.4781
  672/25000 [..............................] - ETA: 1:00 - loss: 7.8720 - accuracy: 0.4866
  704/25000 [..............................] - ETA: 59s - loss: 7.9062 - accuracy: 0.4844 
  736/25000 [..............................] - ETA: 59s - loss: 7.9166 - accuracy: 0.4837
  768/25000 [..............................] - ETA: 59s - loss: 7.9461 - accuracy: 0.4818
  800/25000 [..............................] - ETA: 58s - loss: 7.9733 - accuracy: 0.4800
  832/25000 [..............................] - ETA: 58s - loss: 7.9062 - accuracy: 0.4844
  864/25000 [>.............................] - ETA: 58s - loss: 7.8441 - accuracy: 0.4884
  896/25000 [>.............................] - ETA: 57s - loss: 7.8720 - accuracy: 0.4866
  928/25000 [>.............................] - ETA: 57s - loss: 7.7988 - accuracy: 0.4914
  960/25000 [>.............................] - ETA: 57s - loss: 7.7784 - accuracy: 0.4927
  992/25000 [>.............................] - ETA: 56s - loss: 7.8057 - accuracy: 0.4909
 1024/25000 [>.............................] - ETA: 56s - loss: 7.8313 - accuracy: 0.4893
 1056/25000 [>.............................] - ETA: 56s - loss: 7.8118 - accuracy: 0.4905
 1088/25000 [>.............................] - ETA: 56s - loss: 7.8216 - accuracy: 0.4899
 1120/25000 [>.............................] - ETA: 56s - loss: 7.8583 - accuracy: 0.4875
 1152/25000 [>.............................] - ETA: 55s - loss: 7.7864 - accuracy: 0.4922
 1184/25000 [>.............................] - ETA: 55s - loss: 7.8220 - accuracy: 0.4899
 1216/25000 [>.............................] - ETA: 55s - loss: 7.7549 - accuracy: 0.4942
 1248/25000 [>.............................] - ETA: 55s - loss: 7.7035 - accuracy: 0.4976
 1280/25000 [>.............................] - ETA: 55s - loss: 7.6786 - accuracy: 0.4992
 1312/25000 [>.............................] - ETA: 55s - loss: 7.6432 - accuracy: 0.5015
 1344/25000 [>.............................] - ETA: 54s - loss: 7.5982 - accuracy: 0.5045
 1376/25000 [>.............................] - ETA: 54s - loss: 7.5552 - accuracy: 0.5073
 1408/25000 [>.............................] - ETA: 54s - loss: 7.6013 - accuracy: 0.5043
 1440/25000 [>.............................] - ETA: 54s - loss: 7.5708 - accuracy: 0.5063
 1472/25000 [>.............................] - ETA: 54s - loss: 7.5312 - accuracy: 0.5088
 1504/25000 [>.............................] - ETA: 54s - loss: 7.5239 - accuracy: 0.5093
 1536/25000 [>.............................] - ETA: 54s - loss: 7.4969 - accuracy: 0.5111
 1568/25000 [>.............................] - ETA: 53s - loss: 7.4808 - accuracy: 0.5121
 1600/25000 [>.............................] - ETA: 53s - loss: 7.5133 - accuracy: 0.5100
 1632/25000 [>.............................] - ETA: 53s - loss: 7.5257 - accuracy: 0.5092
 1664/25000 [>.............................] - ETA: 53s - loss: 7.5560 - accuracy: 0.5072
 1696/25000 [=>............................] - ETA: 53s - loss: 7.5310 - accuracy: 0.5088
 1728/25000 [=>............................] - ETA: 53s - loss: 7.5513 - accuracy: 0.5075
 1760/25000 [=>............................] - ETA: 53s - loss: 7.5534 - accuracy: 0.5074
 1792/25000 [=>............................] - ETA: 52s - loss: 7.5554 - accuracy: 0.5073
 1824/25000 [=>............................] - ETA: 52s - loss: 7.5573 - accuracy: 0.5071
 1856/25000 [=>............................] - ETA: 52s - loss: 7.5675 - accuracy: 0.5065
 1888/25000 [=>............................] - ETA: 52s - loss: 7.6098 - accuracy: 0.5037
 1920/25000 [=>............................] - ETA: 52s - loss: 7.5947 - accuracy: 0.5047
 1952/25000 [=>............................] - ETA: 52s - loss: 7.5881 - accuracy: 0.5051
 1984/25000 [=>............................] - ETA: 52s - loss: 7.5584 - accuracy: 0.5071
 2016/25000 [=>............................] - ETA: 52s - loss: 7.5449 - accuracy: 0.5079
 2048/25000 [=>............................] - ETA: 51s - loss: 7.5393 - accuracy: 0.5083
 2080/25000 [=>............................] - ETA: 51s - loss: 7.5339 - accuracy: 0.5087
 2112/25000 [=>............................] - ETA: 51s - loss: 7.5795 - accuracy: 0.5057
 2144/25000 [=>............................] - ETA: 51s - loss: 7.5951 - accuracy: 0.5047
 2176/25000 [=>............................] - ETA: 51s - loss: 7.5891 - accuracy: 0.5051
 2208/25000 [=>............................] - ETA: 51s - loss: 7.5833 - accuracy: 0.5054
 2240/25000 [=>............................] - ETA: 51s - loss: 7.5297 - accuracy: 0.5089
 2272/25000 [=>............................] - ETA: 51s - loss: 7.5451 - accuracy: 0.5079
 2304/25000 [=>............................] - ETA: 50s - loss: 7.5535 - accuracy: 0.5074
 2336/25000 [=>............................] - ETA: 50s - loss: 7.5550 - accuracy: 0.5073
 2368/25000 [=>............................] - ETA: 50s - loss: 7.5501 - accuracy: 0.5076
 2400/25000 [=>............................] - ETA: 50s - loss: 7.5452 - accuracy: 0.5079
 2432/25000 [=>............................] - ETA: 50s - loss: 7.5342 - accuracy: 0.5086
 2464/25000 [=>............................] - ETA: 50s - loss: 7.5110 - accuracy: 0.5101
 2496/25000 [=>............................] - ETA: 50s - loss: 7.5253 - accuracy: 0.5092
 2528/25000 [==>...........................] - ETA: 50s - loss: 7.5211 - accuracy: 0.5095
 2560/25000 [==>...........................] - ETA: 50s - loss: 7.5289 - accuracy: 0.5090
 2592/25000 [==>...........................] - ETA: 50s - loss: 7.5187 - accuracy: 0.5096
 2624/25000 [==>...........................] - ETA: 50s - loss: 7.4972 - accuracy: 0.5111
 2656/25000 [==>...........................] - ETA: 49s - loss: 7.4934 - accuracy: 0.5113
 2688/25000 [==>...........................] - ETA: 49s - loss: 7.4784 - accuracy: 0.5123
 2720/25000 [==>...........................] - ETA: 49s - loss: 7.4919 - accuracy: 0.5114
 2752/25000 [==>...........................] - ETA: 49s - loss: 7.4883 - accuracy: 0.5116
 2784/25000 [==>...........................] - ETA: 49s - loss: 7.4849 - accuracy: 0.5119
 2816/25000 [==>...........................] - ETA: 49s - loss: 7.4869 - accuracy: 0.5117
 2848/25000 [==>...........................] - ETA: 49s - loss: 7.4997 - accuracy: 0.5109
 2880/25000 [==>...........................] - ETA: 49s - loss: 7.5069 - accuracy: 0.5104
 2912/25000 [==>...........................] - ETA: 49s - loss: 7.5192 - accuracy: 0.5096
 2944/25000 [==>...........................] - ETA: 49s - loss: 7.5104 - accuracy: 0.5102
 2976/25000 [==>...........................] - ETA: 49s - loss: 7.5172 - accuracy: 0.5097
 3008/25000 [==>...........................] - ETA: 48s - loss: 7.4933 - accuracy: 0.5113
 3040/25000 [==>...........................] - ETA: 48s - loss: 7.5002 - accuracy: 0.5109
 3072/25000 [==>...........................] - ETA: 48s - loss: 7.5319 - accuracy: 0.5088
 3104/25000 [==>...........................] - ETA: 48s - loss: 7.5530 - accuracy: 0.5074
 3136/25000 [==>...........................] - ETA: 48s - loss: 7.5542 - accuracy: 0.5073
 3168/25000 [==>...........................] - ETA: 48s - loss: 7.5408 - accuracy: 0.5082
 3200/25000 [==>...........................] - ETA: 48s - loss: 7.5564 - accuracy: 0.5072
 3232/25000 [==>...........................] - ETA: 48s - loss: 7.5622 - accuracy: 0.5068
 3264/25000 [==>...........................] - ETA: 48s - loss: 7.5539 - accuracy: 0.5074
 3296/25000 [==>...........................] - ETA: 48s - loss: 7.5643 - accuracy: 0.5067
 3328/25000 [==>...........................] - ETA: 48s - loss: 7.5607 - accuracy: 0.5069
 3360/25000 [===>..........................] - ETA: 47s - loss: 7.5799 - accuracy: 0.5057
 3392/25000 [===>..........................] - ETA: 48s - loss: 7.5717 - accuracy: 0.5062
 3424/25000 [===>..........................] - ETA: 47s - loss: 7.5591 - accuracy: 0.5070
 3456/25000 [===>..........................] - ETA: 47s - loss: 7.5690 - accuracy: 0.5064
 3488/25000 [===>..........................] - ETA: 47s - loss: 7.5611 - accuracy: 0.5069
 3520/25000 [===>..........................] - ETA: 47s - loss: 7.5882 - accuracy: 0.5051
 3552/25000 [===>..........................] - ETA: 47s - loss: 7.5932 - accuracy: 0.5048
 3584/25000 [===>..........................] - ETA: 47s - loss: 7.5896 - accuracy: 0.5050
 3616/25000 [===>..........................] - ETA: 47s - loss: 7.5988 - accuracy: 0.5044
 3648/25000 [===>..........................] - ETA: 47s - loss: 7.5994 - accuracy: 0.5044
 3680/25000 [===>..........................] - ETA: 47s - loss: 7.6041 - accuracy: 0.5041
 3712/25000 [===>..........................] - ETA: 47s - loss: 7.6088 - accuracy: 0.5038
 3744/25000 [===>..........................] - ETA: 47s - loss: 7.6093 - accuracy: 0.5037
 3776/25000 [===>..........................] - ETA: 46s - loss: 7.6179 - accuracy: 0.5032
 3808/25000 [===>..........................] - ETA: 46s - loss: 7.6102 - accuracy: 0.5037
 3840/25000 [===>..........................] - ETA: 46s - loss: 7.6067 - accuracy: 0.5039
 3872/25000 [===>..........................] - ETA: 46s - loss: 7.6033 - accuracy: 0.5041
 3904/25000 [===>..........................] - ETA: 46s - loss: 7.6156 - accuracy: 0.5033
 3936/25000 [===>..........................] - ETA: 46s - loss: 7.6316 - accuracy: 0.5023
 3968/25000 [===>..........................] - ETA: 46s - loss: 7.6318 - accuracy: 0.5023
 4000/25000 [===>..........................] - ETA: 46s - loss: 7.6513 - accuracy: 0.5010
 4032/25000 [===>..........................] - ETA: 46s - loss: 7.6210 - accuracy: 0.5030
 4064/25000 [===>..........................] - ETA: 46s - loss: 7.6289 - accuracy: 0.5025
 4096/25000 [===>..........................] - ETA: 46s - loss: 7.6367 - accuracy: 0.5020
 4128/25000 [===>..........................] - ETA: 46s - loss: 7.6369 - accuracy: 0.5019
 4160/25000 [===>..........................] - ETA: 46s - loss: 7.6482 - accuracy: 0.5012
 4192/25000 [====>.........................] - ETA: 45s - loss: 7.6410 - accuracy: 0.5017
 4224/25000 [====>.........................] - ETA: 45s - loss: 7.6231 - accuracy: 0.5028
 4256/25000 [====>.........................] - ETA: 45s - loss: 7.6162 - accuracy: 0.5033
 4288/25000 [====>.........................] - ETA: 45s - loss: 7.6094 - accuracy: 0.5037
 4320/25000 [====>.........................] - ETA: 45s - loss: 7.6169 - accuracy: 0.5032
 4352/25000 [====>.........................] - ETA: 45s - loss: 7.6102 - accuracy: 0.5037
 4384/25000 [====>.........................] - ETA: 45s - loss: 7.6177 - accuracy: 0.5032
 4416/25000 [====>.........................] - ETA: 45s - loss: 7.6180 - accuracy: 0.5032
 4448/25000 [====>.........................] - ETA: 45s - loss: 7.6149 - accuracy: 0.5034
 4480/25000 [====>.........................] - ETA: 45s - loss: 7.6187 - accuracy: 0.5031
 4512/25000 [====>.........................] - ETA: 45s - loss: 7.6122 - accuracy: 0.5035
 4544/25000 [====>.........................] - ETA: 45s - loss: 7.6025 - accuracy: 0.5042
 4576/25000 [====>.........................] - ETA: 45s - loss: 7.6063 - accuracy: 0.5039
 4608/25000 [====>.........................] - ETA: 44s - loss: 7.6101 - accuracy: 0.5037
 4640/25000 [====>.........................] - ETA: 44s - loss: 7.5972 - accuracy: 0.5045
 4672/25000 [====>.........................] - ETA: 44s - loss: 7.6141 - accuracy: 0.5034
 4704/25000 [====>.........................] - ETA: 44s - loss: 7.6079 - accuracy: 0.5038
 4736/25000 [====>.........................] - ETA: 44s - loss: 7.6213 - accuracy: 0.5030
 4768/25000 [====>.........................] - ETA: 44s - loss: 7.6248 - accuracy: 0.5027
 4800/25000 [====>.........................] - ETA: 44s - loss: 7.6187 - accuracy: 0.5031
 4832/25000 [====>.........................] - ETA: 44s - loss: 7.6158 - accuracy: 0.5033
 4864/25000 [====>.........................] - ETA: 44s - loss: 7.6193 - accuracy: 0.5031
 4896/25000 [====>.........................] - ETA: 44s - loss: 7.6290 - accuracy: 0.5025
 4928/25000 [====>.........................] - ETA: 44s - loss: 7.6386 - accuracy: 0.5018
 4960/25000 [====>.........................] - ETA: 44s - loss: 7.6357 - accuracy: 0.5020
 4992/25000 [====>.........................] - ETA: 44s - loss: 7.6390 - accuracy: 0.5018
 5024/25000 [=====>........................] - ETA: 44s - loss: 7.6422 - accuracy: 0.5016
 5056/25000 [=====>........................] - ETA: 43s - loss: 7.6484 - accuracy: 0.5012
 5088/25000 [=====>........................] - ETA: 43s - loss: 7.6455 - accuracy: 0.5014
 5120/25000 [=====>........................] - ETA: 43s - loss: 7.6606 - accuracy: 0.5004
 5152/25000 [=====>........................] - ETA: 43s - loss: 7.6577 - accuracy: 0.5006
 5184/25000 [=====>........................] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
 5216/25000 [=====>........................] - ETA: 43s - loss: 7.6725 - accuracy: 0.4996
 5248/25000 [=====>........................] - ETA: 43s - loss: 7.6695 - accuracy: 0.4998
 5280/25000 [=====>........................] - ETA: 43s - loss: 7.6782 - accuracy: 0.4992
 5312/25000 [=====>........................] - ETA: 43s - loss: 7.6724 - accuracy: 0.4996
 5344/25000 [=====>........................] - ETA: 43s - loss: 7.6838 - accuracy: 0.4989
 5376/25000 [=====>........................] - ETA: 43s - loss: 7.6695 - accuracy: 0.4998
 5408/25000 [=====>........................] - ETA: 43s - loss: 7.6553 - accuracy: 0.5007
 5440/25000 [=====>........................] - ETA: 43s - loss: 7.6469 - accuracy: 0.5013
 5472/25000 [=====>........................] - ETA: 42s - loss: 7.6386 - accuracy: 0.5018
 5504/25000 [=====>........................] - ETA: 42s - loss: 7.6304 - accuracy: 0.5024
 5536/25000 [=====>........................] - ETA: 42s - loss: 7.6417 - accuracy: 0.5016
 5568/25000 [=====>........................] - ETA: 42s - loss: 7.6226 - accuracy: 0.5029
 5600/25000 [=====>........................] - ETA: 42s - loss: 7.6173 - accuracy: 0.5032
 5632/25000 [=====>........................] - ETA: 42s - loss: 7.6285 - accuracy: 0.5025
 5664/25000 [=====>........................] - ETA: 42s - loss: 7.6206 - accuracy: 0.5030
 5696/25000 [=====>........................] - ETA: 42s - loss: 7.6235 - accuracy: 0.5028
 5728/25000 [=====>........................] - ETA: 42s - loss: 7.6077 - accuracy: 0.5038
 5760/25000 [=====>........................] - ETA: 42s - loss: 7.6214 - accuracy: 0.5030
 5792/25000 [=====>........................] - ETA: 42s - loss: 7.6243 - accuracy: 0.5028
 5824/25000 [=====>........................] - ETA: 42s - loss: 7.6219 - accuracy: 0.5029
 5856/25000 [======>.......................] - ETA: 41s - loss: 7.6326 - accuracy: 0.5022
 5888/25000 [======>.......................] - ETA: 41s - loss: 7.6380 - accuracy: 0.5019
 5920/25000 [======>.......................] - ETA: 41s - loss: 7.6407 - accuracy: 0.5017
 5952/25000 [======>.......................] - ETA: 41s - loss: 7.6460 - accuracy: 0.5013
 5984/25000 [======>.......................] - ETA: 41s - loss: 7.6538 - accuracy: 0.5008
 6016/25000 [======>.......................] - ETA: 41s - loss: 7.6437 - accuracy: 0.5015
 6048/25000 [======>.......................] - ETA: 41s - loss: 7.6463 - accuracy: 0.5013
 6080/25000 [======>.......................] - ETA: 41s - loss: 7.6263 - accuracy: 0.5026
 6112/25000 [======>.......................] - ETA: 41s - loss: 7.6215 - accuracy: 0.5029
 6144/25000 [======>.......................] - ETA: 41s - loss: 7.6267 - accuracy: 0.5026
 6176/25000 [======>.......................] - ETA: 41s - loss: 7.6319 - accuracy: 0.5023
 6208/25000 [======>.......................] - ETA: 41s - loss: 7.6345 - accuracy: 0.5021
 6240/25000 [======>.......................] - ETA: 41s - loss: 7.6322 - accuracy: 0.5022
 6272/25000 [======>.......................] - ETA: 40s - loss: 7.6397 - accuracy: 0.5018
 6304/25000 [======>.......................] - ETA: 40s - loss: 7.6472 - accuracy: 0.5013
 6336/25000 [======>.......................] - ETA: 40s - loss: 7.6521 - accuracy: 0.5009
 6368/25000 [======>.......................] - ETA: 40s - loss: 7.6618 - accuracy: 0.5003
 6400/25000 [======>.......................] - ETA: 40s - loss: 7.6642 - accuracy: 0.5002
 6432/25000 [======>.......................] - ETA: 40s - loss: 7.6642 - accuracy: 0.5002
 6464/25000 [======>.......................] - ETA: 40s - loss: 7.6595 - accuracy: 0.5005
 6496/25000 [======>.......................] - ETA: 40s - loss: 7.6572 - accuracy: 0.5006
 6528/25000 [======>.......................] - ETA: 40s - loss: 7.6478 - accuracy: 0.5012
 6560/25000 [======>.......................] - ETA: 40s - loss: 7.6456 - accuracy: 0.5014
 6592/25000 [======>.......................] - ETA: 40s - loss: 7.6410 - accuracy: 0.5017
 6624/25000 [======>.......................] - ETA: 40s - loss: 7.6435 - accuracy: 0.5015
 6656/25000 [======>.......................] - ETA: 39s - loss: 7.6413 - accuracy: 0.5017
 6688/25000 [=======>......................] - ETA: 39s - loss: 7.6437 - accuracy: 0.5015
 6720/25000 [=======>......................] - ETA: 39s - loss: 7.6370 - accuracy: 0.5019
 6752/25000 [=======>......................] - ETA: 39s - loss: 7.6303 - accuracy: 0.5024
 6784/25000 [=======>......................] - ETA: 39s - loss: 7.6327 - accuracy: 0.5022
 6816/25000 [=======>......................] - ETA: 39s - loss: 7.6216 - accuracy: 0.5029
 6848/25000 [=======>......................] - ETA: 39s - loss: 7.6151 - accuracy: 0.5034
 6880/25000 [=======>......................] - ETA: 39s - loss: 7.6109 - accuracy: 0.5036
 6912/25000 [=======>......................] - ETA: 39s - loss: 7.6067 - accuracy: 0.5039
 6944/25000 [=======>......................] - ETA: 39s - loss: 7.6070 - accuracy: 0.5039
 6976/25000 [=======>......................] - ETA: 39s - loss: 7.6073 - accuracy: 0.5039
 7008/25000 [=======>......................] - ETA: 39s - loss: 7.6141 - accuracy: 0.5034
 7040/25000 [=======>......................] - ETA: 39s - loss: 7.6122 - accuracy: 0.5036
 7072/25000 [=======>......................] - ETA: 38s - loss: 7.6102 - accuracy: 0.5037
 7104/25000 [=======>......................] - ETA: 38s - loss: 7.6148 - accuracy: 0.5034
 7136/25000 [=======>......................] - ETA: 38s - loss: 7.6151 - accuracy: 0.5034
 7168/25000 [=======>......................] - ETA: 38s - loss: 7.6131 - accuracy: 0.5035
 7200/25000 [=======>......................] - ETA: 38s - loss: 7.6091 - accuracy: 0.5038
 7232/25000 [=======>......................] - ETA: 38s - loss: 7.6073 - accuracy: 0.5039
 7264/25000 [=======>......................] - ETA: 38s - loss: 7.6138 - accuracy: 0.5034
 7296/25000 [=======>......................] - ETA: 38s - loss: 7.6036 - accuracy: 0.5041
 7328/25000 [=======>......................] - ETA: 38s - loss: 7.6080 - accuracy: 0.5038
 7360/25000 [=======>......................] - ETA: 38s - loss: 7.6104 - accuracy: 0.5037
 7392/25000 [=======>......................] - ETA: 38s - loss: 7.6106 - accuracy: 0.5037
 7424/25000 [=======>......................] - ETA: 38s - loss: 7.6129 - accuracy: 0.5035
 7456/25000 [=======>......................] - ETA: 38s - loss: 7.6173 - accuracy: 0.5032
 7488/25000 [=======>......................] - ETA: 37s - loss: 7.6175 - accuracy: 0.5032
 7520/25000 [========>.....................] - ETA: 37s - loss: 7.6299 - accuracy: 0.5024
 7552/25000 [========>.....................] - ETA: 37s - loss: 7.6301 - accuracy: 0.5024
 7584/25000 [========>.....................] - ETA: 37s - loss: 7.6282 - accuracy: 0.5025
 7616/25000 [========>.....................] - ETA: 37s - loss: 7.6243 - accuracy: 0.5028
 7648/25000 [========>.....................] - ETA: 37s - loss: 7.6145 - accuracy: 0.5034
 7680/25000 [========>.....................] - ETA: 37s - loss: 7.6167 - accuracy: 0.5033
 7712/25000 [========>.....................] - ETA: 37s - loss: 7.6189 - accuracy: 0.5031
 7744/25000 [========>.....................] - ETA: 37s - loss: 7.6231 - accuracy: 0.5028
 7776/25000 [========>.....................] - ETA: 37s - loss: 7.6430 - accuracy: 0.5015
 7808/25000 [========>.....................] - ETA: 37s - loss: 7.6411 - accuracy: 0.5017
 7840/25000 [========>.....................] - ETA: 37s - loss: 7.6471 - accuracy: 0.5013
 7872/25000 [========>.....................] - ETA: 37s - loss: 7.6471 - accuracy: 0.5013
 7904/25000 [========>.....................] - ETA: 36s - loss: 7.6453 - accuracy: 0.5014
 7936/25000 [========>.....................] - ETA: 36s - loss: 7.6434 - accuracy: 0.5015
 7968/25000 [========>.....................] - ETA: 36s - loss: 7.6435 - accuracy: 0.5015
 8000/25000 [========>.....................] - ETA: 36s - loss: 7.6513 - accuracy: 0.5010
 8032/25000 [========>.....................] - ETA: 36s - loss: 7.6552 - accuracy: 0.5007
 8064/25000 [========>.....................] - ETA: 36s - loss: 7.6590 - accuracy: 0.5005
 8096/25000 [========>.....................] - ETA: 36s - loss: 7.6628 - accuracy: 0.5002
 8128/25000 [========>.....................] - ETA: 36s - loss: 7.6553 - accuracy: 0.5007
 8160/25000 [========>.....................] - ETA: 36s - loss: 7.6516 - accuracy: 0.5010
 8192/25000 [========>.....................] - ETA: 36s - loss: 7.6498 - accuracy: 0.5011
 8224/25000 [========>.....................] - ETA: 36s - loss: 7.6554 - accuracy: 0.5007
 8256/25000 [========>.....................] - ETA: 36s - loss: 7.6518 - accuracy: 0.5010
 8288/25000 [========>.....................] - ETA: 36s - loss: 7.6537 - accuracy: 0.5008
 8320/25000 [========>.....................] - ETA: 35s - loss: 7.6592 - accuracy: 0.5005
 8352/25000 [=========>....................] - ETA: 35s - loss: 7.6721 - accuracy: 0.4996
 8384/25000 [=========>....................] - ETA: 35s - loss: 7.6776 - accuracy: 0.4993
 8416/25000 [=========>....................] - ETA: 35s - loss: 7.6630 - accuracy: 0.5002
 8448/25000 [=========>....................] - ETA: 35s - loss: 7.6630 - accuracy: 0.5002
 8480/25000 [=========>....................] - ETA: 35s - loss: 7.6648 - accuracy: 0.5001
 8512/25000 [=========>....................] - ETA: 35s - loss: 7.6612 - accuracy: 0.5004
 8544/25000 [=========>....................] - ETA: 35s - loss: 7.6612 - accuracy: 0.5004
 8576/25000 [=========>....................] - ETA: 35s - loss: 7.6630 - accuracy: 0.5002
 8608/25000 [=========>....................] - ETA: 35s - loss: 7.6613 - accuracy: 0.5003
 8640/25000 [=========>....................] - ETA: 35s - loss: 7.6648 - accuracy: 0.5001
 8672/25000 [=========>....................] - ETA: 35s - loss: 7.6631 - accuracy: 0.5002
 8704/25000 [=========>....................] - ETA: 35s - loss: 7.6684 - accuracy: 0.4999
 8736/25000 [=========>....................] - ETA: 35s - loss: 7.6666 - accuracy: 0.5000
 8768/25000 [=========>....................] - ETA: 34s - loss: 7.6701 - accuracy: 0.4998
 8800/25000 [=========>....................] - ETA: 34s - loss: 7.6701 - accuracy: 0.4998
 8832/25000 [=========>....................] - ETA: 34s - loss: 7.6718 - accuracy: 0.4997
 8864/25000 [=========>....................] - ETA: 34s - loss: 7.6683 - accuracy: 0.4999
 8896/25000 [=========>....................] - ETA: 34s - loss: 7.6649 - accuracy: 0.5001
 8928/25000 [=========>....................] - ETA: 34s - loss: 7.6666 - accuracy: 0.5000
 8960/25000 [=========>....................] - ETA: 34s - loss: 7.6718 - accuracy: 0.4997
 8992/25000 [=========>....................] - ETA: 34s - loss: 7.6632 - accuracy: 0.5002
 9024/25000 [=========>....................] - ETA: 34s - loss: 7.6683 - accuracy: 0.4999
 9056/25000 [=========>....................] - ETA: 34s - loss: 7.6819 - accuracy: 0.4990
 9088/25000 [=========>....................] - ETA: 34s - loss: 7.6886 - accuracy: 0.4986
 9120/25000 [=========>....................] - ETA: 34s - loss: 7.6801 - accuracy: 0.4991
 9152/25000 [=========>....................] - ETA: 34s - loss: 7.6800 - accuracy: 0.4991
 9184/25000 [==========>...................] - ETA: 33s - loss: 7.6766 - accuracy: 0.4993
 9216/25000 [==========>...................] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
 9248/25000 [==========>...................] - ETA: 33s - loss: 7.6683 - accuracy: 0.4999
 9280/25000 [==========>...................] - ETA: 33s - loss: 7.6798 - accuracy: 0.4991
 9312/25000 [==========>...................] - ETA: 33s - loss: 7.6765 - accuracy: 0.4994
 9344/25000 [==========>...................] - ETA: 33s - loss: 7.6748 - accuracy: 0.4995
 9376/25000 [==========>...................] - ETA: 33s - loss: 7.6764 - accuracy: 0.4994
 9408/25000 [==========>...................] - ETA: 33s - loss: 7.6813 - accuracy: 0.4990
 9440/25000 [==========>...................] - ETA: 33s - loss: 7.6731 - accuracy: 0.4996
 9472/25000 [==========>...................] - ETA: 33s - loss: 7.6747 - accuracy: 0.4995
 9504/25000 [==========>...................] - ETA: 33s - loss: 7.6715 - accuracy: 0.4997
 9536/25000 [==========>...................] - ETA: 33s - loss: 7.6827 - accuracy: 0.4990
 9568/25000 [==========>...................] - ETA: 33s - loss: 7.6826 - accuracy: 0.4990
 9600/25000 [==========>...................] - ETA: 33s - loss: 7.6858 - accuracy: 0.4988
 9632/25000 [==========>...................] - ETA: 33s - loss: 7.6809 - accuracy: 0.4991
 9664/25000 [==========>...................] - ETA: 32s - loss: 7.6730 - accuracy: 0.4996
 9696/25000 [==========>...................] - ETA: 32s - loss: 7.6729 - accuracy: 0.4996
 9728/25000 [==========>...................] - ETA: 32s - loss: 7.6682 - accuracy: 0.4999
 9760/25000 [==========>...................] - ETA: 32s - loss: 7.6682 - accuracy: 0.4999
 9792/25000 [==========>...................] - ETA: 32s - loss: 7.6698 - accuracy: 0.4998
 9824/25000 [==========>...................] - ETA: 32s - loss: 7.6713 - accuracy: 0.4997
 9856/25000 [==========>...................] - ETA: 32s - loss: 7.6682 - accuracy: 0.4999
 9888/25000 [==========>...................] - ETA: 32s - loss: 7.6651 - accuracy: 0.5001
 9920/25000 [==========>...................] - ETA: 32s - loss: 7.6543 - accuracy: 0.5008
 9952/25000 [==========>...................] - ETA: 32s - loss: 7.6589 - accuracy: 0.5005
 9984/25000 [==========>...................] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
10016/25000 [===========>..................] - ETA: 32s - loss: 7.6620 - accuracy: 0.5003
10048/25000 [===========>..................] - ETA: 32s - loss: 7.6681 - accuracy: 0.4999
10080/25000 [===========>..................] - ETA: 32s - loss: 7.6727 - accuracy: 0.4996
10112/25000 [===========>..................] - ETA: 31s - loss: 7.6727 - accuracy: 0.4996
10144/25000 [===========>..................] - ETA: 31s - loss: 7.6651 - accuracy: 0.5001
10176/25000 [===========>..................] - ETA: 31s - loss: 7.6651 - accuracy: 0.5001
10208/25000 [===========>..................] - ETA: 31s - loss: 7.6591 - accuracy: 0.5005
10240/25000 [===========>..................] - ETA: 31s - loss: 7.6591 - accuracy: 0.5005
10272/25000 [===========>..................] - ETA: 31s - loss: 7.6681 - accuracy: 0.4999
10304/25000 [===========>..................] - ETA: 31s - loss: 7.6681 - accuracy: 0.4999
10336/25000 [===========>..................] - ETA: 31s - loss: 7.6711 - accuracy: 0.4997
10368/25000 [===========>..................] - ETA: 31s - loss: 7.6651 - accuracy: 0.5001
10400/25000 [===========>..................] - ETA: 31s - loss: 7.6592 - accuracy: 0.5005
10432/25000 [===========>..................] - ETA: 31s - loss: 7.6593 - accuracy: 0.5005
10464/25000 [===========>..................] - ETA: 31s - loss: 7.6593 - accuracy: 0.5005
10496/25000 [===========>..................] - ETA: 31s - loss: 7.6564 - accuracy: 0.5007
10528/25000 [===========>..................] - ETA: 31s - loss: 7.6608 - accuracy: 0.5004
10560/25000 [===========>..................] - ETA: 30s - loss: 7.6550 - accuracy: 0.5008
10592/25000 [===========>..................] - ETA: 30s - loss: 7.6521 - accuracy: 0.5009
10624/25000 [===========>..................] - ETA: 30s - loss: 7.6536 - accuracy: 0.5008
10656/25000 [===========>..................] - ETA: 30s - loss: 7.6522 - accuracy: 0.5009
10688/25000 [===========>..................] - ETA: 30s - loss: 7.6537 - accuracy: 0.5008
10720/25000 [===========>..................] - ETA: 30s - loss: 7.6580 - accuracy: 0.5006
10752/25000 [===========>..................] - ETA: 30s - loss: 7.6566 - accuracy: 0.5007
10784/25000 [===========>..................] - ETA: 30s - loss: 7.6595 - accuracy: 0.5005
10816/25000 [===========>..................] - ETA: 30s - loss: 7.6567 - accuracy: 0.5006
10848/25000 [============>.................] - ETA: 30s - loss: 7.6567 - accuracy: 0.5006
10880/25000 [============>.................] - ETA: 30s - loss: 7.6539 - accuracy: 0.5008
10912/25000 [============>.................] - ETA: 30s - loss: 7.6526 - accuracy: 0.5009
10944/25000 [============>.................] - ETA: 30s - loss: 7.6456 - accuracy: 0.5014
10976/25000 [============>.................] - ETA: 30s - loss: 7.6513 - accuracy: 0.5010
11008/25000 [============>.................] - ETA: 29s - loss: 7.6555 - accuracy: 0.5007
11040/25000 [============>.................] - ETA: 29s - loss: 7.6555 - accuracy: 0.5007
11072/25000 [============>.................] - ETA: 29s - loss: 7.6555 - accuracy: 0.5007
11104/25000 [============>.................] - ETA: 29s - loss: 7.6542 - accuracy: 0.5008
11136/25000 [============>.................] - ETA: 29s - loss: 7.6570 - accuracy: 0.5006
11168/25000 [============>.................] - ETA: 29s - loss: 7.6529 - accuracy: 0.5009
11200/25000 [============>.................] - ETA: 29s - loss: 7.6543 - accuracy: 0.5008
11232/25000 [============>.................] - ETA: 29s - loss: 7.6516 - accuracy: 0.5010
11264/25000 [============>.................] - ETA: 29s - loss: 7.6530 - accuracy: 0.5009
11296/25000 [============>.................] - ETA: 29s - loss: 7.6490 - accuracy: 0.5012
11328/25000 [============>.................] - ETA: 29s - loss: 7.6450 - accuracy: 0.5014
11360/25000 [============>.................] - ETA: 29s - loss: 7.6342 - accuracy: 0.5021
11392/25000 [============>.................] - ETA: 29s - loss: 7.6384 - accuracy: 0.5018
11424/25000 [============>.................] - ETA: 29s - loss: 7.6438 - accuracy: 0.5015
11456/25000 [============>.................] - ETA: 28s - loss: 7.6465 - accuracy: 0.5013
11488/25000 [============>.................] - ETA: 28s - loss: 7.6493 - accuracy: 0.5011
11520/25000 [============>.................] - ETA: 28s - loss: 7.6467 - accuracy: 0.5013
11552/25000 [============>.................] - ETA: 28s - loss: 7.6520 - accuracy: 0.5010
11584/25000 [============>.................] - ETA: 28s - loss: 7.6534 - accuracy: 0.5009
11616/25000 [============>.................] - ETA: 28s - loss: 7.6547 - accuracy: 0.5008
11648/25000 [============>.................] - ETA: 28s - loss: 7.6508 - accuracy: 0.5010
11680/25000 [=============>................] - ETA: 28s - loss: 7.6496 - accuracy: 0.5011
11712/25000 [=============>................] - ETA: 28s - loss: 7.6496 - accuracy: 0.5011
11744/25000 [=============>................] - ETA: 28s - loss: 7.6496 - accuracy: 0.5011
11776/25000 [=============>................] - ETA: 28s - loss: 7.6510 - accuracy: 0.5010
11808/25000 [=============>................] - ETA: 28s - loss: 7.6536 - accuracy: 0.5008
11840/25000 [=============>................] - ETA: 28s - loss: 7.6485 - accuracy: 0.5012
11872/25000 [=============>................] - ETA: 28s - loss: 7.6537 - accuracy: 0.5008
11904/25000 [=============>................] - ETA: 27s - loss: 7.6537 - accuracy: 0.5008
11936/25000 [=============>................] - ETA: 27s - loss: 7.6525 - accuracy: 0.5009
11968/25000 [=============>................] - ETA: 27s - loss: 7.6525 - accuracy: 0.5009
12000/25000 [=============>................] - ETA: 27s - loss: 7.6526 - accuracy: 0.5009
12032/25000 [=============>................] - ETA: 27s - loss: 7.6564 - accuracy: 0.5007
12064/25000 [=============>................] - ETA: 27s - loss: 7.6590 - accuracy: 0.5005
12096/25000 [=============>................] - ETA: 27s - loss: 7.6577 - accuracy: 0.5006
12128/25000 [=============>................] - ETA: 27s - loss: 7.6590 - accuracy: 0.5005
12160/25000 [=============>................] - ETA: 27s - loss: 7.6603 - accuracy: 0.5004
12192/25000 [=============>................] - ETA: 27s - loss: 7.6603 - accuracy: 0.5004
12224/25000 [=============>................] - ETA: 27s - loss: 7.6616 - accuracy: 0.5003
12256/25000 [=============>................] - ETA: 27s - loss: 7.6679 - accuracy: 0.4999
12288/25000 [=============>................] - ETA: 27s - loss: 7.6716 - accuracy: 0.4997
12320/25000 [=============>................] - ETA: 27s - loss: 7.6728 - accuracy: 0.4996
12352/25000 [=============>................] - ETA: 26s - loss: 7.6728 - accuracy: 0.4996
12384/25000 [=============>................] - ETA: 26s - loss: 7.6728 - accuracy: 0.4996
12416/25000 [=============>................] - ETA: 26s - loss: 7.6740 - accuracy: 0.4995
12448/25000 [=============>................] - ETA: 26s - loss: 7.6703 - accuracy: 0.4998
12480/25000 [=============>................] - ETA: 26s - loss: 7.6666 - accuracy: 0.5000
12512/25000 [==============>...............] - ETA: 26s - loss: 7.6678 - accuracy: 0.4999
12544/25000 [==============>...............] - ETA: 26s - loss: 7.6642 - accuracy: 0.5002
12576/25000 [==============>...............] - ETA: 26s - loss: 7.6617 - accuracy: 0.5003
12608/25000 [==============>...............] - ETA: 26s - loss: 7.6654 - accuracy: 0.5001
12640/25000 [==============>...............] - ETA: 26s - loss: 7.6666 - accuracy: 0.5000
12672/25000 [==============>...............] - ETA: 26s - loss: 7.6642 - accuracy: 0.5002
12704/25000 [==============>...............] - ETA: 26s - loss: 7.6654 - accuracy: 0.5001
12736/25000 [==============>...............] - ETA: 26s - loss: 7.6678 - accuracy: 0.4999
12768/25000 [==============>...............] - ETA: 26s - loss: 7.6714 - accuracy: 0.4997
12800/25000 [==============>...............] - ETA: 26s - loss: 7.6762 - accuracy: 0.4994
12832/25000 [==============>...............] - ETA: 25s - loss: 7.6738 - accuracy: 0.4995
12864/25000 [==============>...............] - ETA: 25s - loss: 7.6702 - accuracy: 0.4998
12896/25000 [==============>...............] - ETA: 25s - loss: 7.6690 - accuracy: 0.4998
12928/25000 [==============>...............] - ETA: 25s - loss: 7.6666 - accuracy: 0.5000
12960/25000 [==============>...............] - ETA: 25s - loss: 7.6607 - accuracy: 0.5004
12992/25000 [==============>...............] - ETA: 25s - loss: 7.6595 - accuracy: 0.5005
13024/25000 [==============>...............] - ETA: 25s - loss: 7.6619 - accuracy: 0.5003
13056/25000 [==============>...............] - ETA: 25s - loss: 7.6643 - accuracy: 0.5002
13088/25000 [==============>...............] - ETA: 25s - loss: 7.6631 - accuracy: 0.5002
13120/25000 [==============>...............] - ETA: 25s - loss: 7.6619 - accuracy: 0.5003
13152/25000 [==============>...............] - ETA: 25s - loss: 7.6608 - accuracy: 0.5004
13184/25000 [==============>...............] - ETA: 25s - loss: 7.6631 - accuracy: 0.5002
13216/25000 [==============>...............] - ETA: 25s - loss: 7.6631 - accuracy: 0.5002
13248/25000 [==============>...............] - ETA: 25s - loss: 7.6655 - accuracy: 0.5001
13280/25000 [==============>...............] - ETA: 25s - loss: 7.6608 - accuracy: 0.5004
13312/25000 [==============>...............] - ETA: 24s - loss: 7.6643 - accuracy: 0.5002
13344/25000 [===============>..............] - ETA: 24s - loss: 7.6643 - accuracy: 0.5001
13376/25000 [===============>..............] - ETA: 24s - loss: 7.6689 - accuracy: 0.4999
13408/25000 [===============>..............] - ETA: 24s - loss: 7.6701 - accuracy: 0.4998
13440/25000 [===============>..............] - ETA: 24s - loss: 7.6689 - accuracy: 0.4999
13472/25000 [===============>..............] - ETA: 24s - loss: 7.6712 - accuracy: 0.4997
13504/25000 [===============>..............] - ETA: 24s - loss: 7.6757 - accuracy: 0.4994
13536/25000 [===============>..............] - ETA: 24s - loss: 7.6791 - accuracy: 0.4992
13568/25000 [===============>..............] - ETA: 24s - loss: 7.6824 - accuracy: 0.4990
13600/25000 [===============>..............] - ETA: 24s - loss: 7.6858 - accuracy: 0.4988
13632/25000 [===============>..............] - ETA: 24s - loss: 7.6891 - accuracy: 0.4985
13664/25000 [===============>..............] - ETA: 24s - loss: 7.6868 - accuracy: 0.4987
13696/25000 [===============>..............] - ETA: 24s - loss: 7.6901 - accuracy: 0.4985
13728/25000 [===============>..............] - ETA: 24s - loss: 7.6856 - accuracy: 0.4988
13760/25000 [===============>..............] - ETA: 23s - loss: 7.6889 - accuracy: 0.4985
13792/25000 [===============>..............] - ETA: 23s - loss: 7.6855 - accuracy: 0.4988
13824/25000 [===============>..............] - ETA: 23s - loss: 7.6821 - accuracy: 0.4990
13856/25000 [===============>..............] - ETA: 23s - loss: 7.6843 - accuracy: 0.4988
13888/25000 [===============>..............] - ETA: 23s - loss: 7.6810 - accuracy: 0.4991
13920/25000 [===============>..............] - ETA: 23s - loss: 7.6787 - accuracy: 0.4992
13952/25000 [===============>..............] - ETA: 23s - loss: 7.6831 - accuracy: 0.4989
13984/25000 [===============>..............] - ETA: 23s - loss: 7.6809 - accuracy: 0.4991
14016/25000 [===============>..............] - ETA: 23s - loss: 7.6852 - accuracy: 0.4988
14048/25000 [===============>..............] - ETA: 23s - loss: 7.6863 - accuracy: 0.4987
14080/25000 [===============>..............] - ETA: 23s - loss: 7.6873 - accuracy: 0.4987
14112/25000 [===============>..............] - ETA: 23s - loss: 7.6862 - accuracy: 0.4987
14144/25000 [===============>..............] - ETA: 23s - loss: 7.6861 - accuracy: 0.4987
14176/25000 [================>.............] - ETA: 23s - loss: 7.6904 - accuracy: 0.4984
14208/25000 [================>.............] - ETA: 23s - loss: 7.6904 - accuracy: 0.4985
14240/25000 [================>.............] - ETA: 22s - loss: 7.6935 - accuracy: 0.4982
14272/25000 [================>.............] - ETA: 22s - loss: 7.6946 - accuracy: 0.4982
14304/25000 [================>.............] - ETA: 22s - loss: 7.6956 - accuracy: 0.4981
14336/25000 [================>.............] - ETA: 22s - loss: 7.6912 - accuracy: 0.4984
14368/25000 [================>.............] - ETA: 22s - loss: 7.6933 - accuracy: 0.4983
14400/25000 [================>.............] - ETA: 22s - loss: 7.6943 - accuracy: 0.4982
14432/25000 [================>.............] - ETA: 22s - loss: 7.6921 - accuracy: 0.4983
14464/25000 [================>.............] - ETA: 22s - loss: 7.6931 - accuracy: 0.4983
14496/25000 [================>.............] - ETA: 22s - loss: 7.6941 - accuracy: 0.4982
14528/25000 [================>.............] - ETA: 22s - loss: 7.6919 - accuracy: 0.4983
14560/25000 [================>.............] - ETA: 22s - loss: 7.6898 - accuracy: 0.4985
14592/25000 [================>.............] - ETA: 22s - loss: 7.6887 - accuracy: 0.4986
14624/25000 [================>.............] - ETA: 22s - loss: 7.6844 - accuracy: 0.4988
14656/25000 [================>.............] - ETA: 22s - loss: 7.6844 - accuracy: 0.4988
14688/25000 [================>.............] - ETA: 22s - loss: 7.6833 - accuracy: 0.4989
14720/25000 [================>.............] - ETA: 21s - loss: 7.6802 - accuracy: 0.4991
14752/25000 [================>.............] - ETA: 21s - loss: 7.6812 - accuracy: 0.4991
14784/25000 [================>.............] - ETA: 21s - loss: 7.6791 - accuracy: 0.4992
14816/25000 [================>.............] - ETA: 21s - loss: 7.6759 - accuracy: 0.4994
14848/25000 [================>.............] - ETA: 21s - loss: 7.6769 - accuracy: 0.4993
14880/25000 [================>.............] - ETA: 21s - loss: 7.6790 - accuracy: 0.4992
14912/25000 [================>.............] - ETA: 21s - loss: 7.6810 - accuracy: 0.4991
14944/25000 [================>.............] - ETA: 21s - loss: 7.6800 - accuracy: 0.4991
14976/25000 [================>.............] - ETA: 21s - loss: 7.6789 - accuracy: 0.4992
15008/25000 [=================>............] - ETA: 21s - loss: 7.6830 - accuracy: 0.4989
15040/25000 [=================>............] - ETA: 21s - loss: 7.6809 - accuracy: 0.4991
15072/25000 [=================>............] - ETA: 21s - loss: 7.6798 - accuracy: 0.4991
15104/25000 [=================>............] - ETA: 21s - loss: 7.6808 - accuracy: 0.4991
15136/25000 [=================>............] - ETA: 21s - loss: 7.6788 - accuracy: 0.4992
15168/25000 [=================>............] - ETA: 21s - loss: 7.6828 - accuracy: 0.4989
15200/25000 [=================>............] - ETA: 20s - loss: 7.6828 - accuracy: 0.4989
15232/25000 [=================>............] - ETA: 20s - loss: 7.6847 - accuracy: 0.4988
15264/25000 [=================>............] - ETA: 20s - loss: 7.6807 - accuracy: 0.4991
15296/25000 [=================>............] - ETA: 20s - loss: 7.6807 - accuracy: 0.4991
15328/25000 [=================>............] - ETA: 20s - loss: 7.6856 - accuracy: 0.4988
15360/25000 [=================>............] - ETA: 20s - loss: 7.6806 - accuracy: 0.4991
15392/25000 [=================>............] - ETA: 20s - loss: 7.6816 - accuracy: 0.4990
15424/25000 [=================>............] - ETA: 20s - loss: 7.6815 - accuracy: 0.4990
15456/25000 [=================>............] - ETA: 20s - loss: 7.6815 - accuracy: 0.4990
15488/25000 [=================>............] - ETA: 20s - loss: 7.6834 - accuracy: 0.4989
15520/25000 [=================>............] - ETA: 20s - loss: 7.6814 - accuracy: 0.4990
15552/25000 [=================>............] - ETA: 20s - loss: 7.6814 - accuracy: 0.4990
15584/25000 [=================>............] - ETA: 20s - loss: 7.6774 - accuracy: 0.4993
15616/25000 [=================>............] - ETA: 20s - loss: 7.6764 - accuracy: 0.4994
15648/25000 [=================>............] - ETA: 19s - loss: 7.6696 - accuracy: 0.4998
15680/25000 [=================>............] - ETA: 19s - loss: 7.6725 - accuracy: 0.4996
15712/25000 [=================>............] - ETA: 19s - loss: 7.6735 - accuracy: 0.4996
15744/25000 [=================>............] - ETA: 19s - loss: 7.6764 - accuracy: 0.4994
15776/25000 [=================>............] - ETA: 19s - loss: 7.6754 - accuracy: 0.4994
15808/25000 [=================>............] - ETA: 19s - loss: 7.6753 - accuracy: 0.4994
15840/25000 [==================>...........] - ETA: 19s - loss: 7.6792 - accuracy: 0.4992
15872/25000 [==================>...........] - ETA: 19s - loss: 7.6801 - accuracy: 0.4991
15904/25000 [==================>...........] - ETA: 19s - loss: 7.6772 - accuracy: 0.4993
15936/25000 [==================>...........] - ETA: 19s - loss: 7.6782 - accuracy: 0.4992
15968/25000 [==================>...........] - ETA: 19s - loss: 7.6839 - accuracy: 0.4989
16000/25000 [==================>...........] - ETA: 19s - loss: 7.6829 - accuracy: 0.4989
16032/25000 [==================>...........] - ETA: 19s - loss: 7.6800 - accuracy: 0.4991
16064/25000 [==================>...........] - ETA: 19s - loss: 7.6800 - accuracy: 0.4991
16096/25000 [==================>...........] - ETA: 19s - loss: 7.6800 - accuracy: 0.4991
16128/25000 [==================>...........] - ETA: 18s - loss: 7.6790 - accuracy: 0.4992
16160/25000 [==================>...........] - ETA: 18s - loss: 7.6752 - accuracy: 0.4994
16192/25000 [==================>...........] - ETA: 18s - loss: 7.6704 - accuracy: 0.4998
16224/25000 [==================>...........] - ETA: 18s - loss: 7.6685 - accuracy: 0.4999
16256/25000 [==================>...........] - ETA: 18s - loss: 7.6685 - accuracy: 0.4999
16288/25000 [==================>...........] - ETA: 18s - loss: 7.6647 - accuracy: 0.5001
16320/25000 [==================>...........] - ETA: 18s - loss: 7.6694 - accuracy: 0.4998
16352/25000 [==================>...........] - ETA: 18s - loss: 7.6694 - accuracy: 0.4998
16384/25000 [==================>...........] - ETA: 18s - loss: 7.6694 - accuracy: 0.4998
16416/25000 [==================>...........] - ETA: 18s - loss: 7.6722 - accuracy: 0.4996
16448/25000 [==================>...........] - ETA: 18s - loss: 7.6750 - accuracy: 0.4995
16480/25000 [==================>...........] - ETA: 18s - loss: 7.6750 - accuracy: 0.4995
16512/25000 [==================>...........] - ETA: 18s - loss: 7.6768 - accuracy: 0.4993
16544/25000 [==================>...........] - ETA: 18s - loss: 7.6768 - accuracy: 0.4993
16576/25000 [==================>...........] - ETA: 17s - loss: 7.6768 - accuracy: 0.4993
16608/25000 [==================>...........] - ETA: 17s - loss: 7.6768 - accuracy: 0.4993
16640/25000 [==================>...........] - ETA: 17s - loss: 7.6804 - accuracy: 0.4991
16672/25000 [===================>..........] - ETA: 17s - loss: 7.6777 - accuracy: 0.4993
16704/25000 [===================>..........] - ETA: 17s - loss: 7.6776 - accuracy: 0.4993
16736/25000 [===================>..........] - ETA: 17s - loss: 7.6776 - accuracy: 0.4993
16768/25000 [===================>..........] - ETA: 17s - loss: 7.6730 - accuracy: 0.4996
16800/25000 [===================>..........] - ETA: 17s - loss: 7.6712 - accuracy: 0.4997
16832/25000 [===================>..........] - ETA: 17s - loss: 7.6766 - accuracy: 0.4993
16864/25000 [===================>..........] - ETA: 17s - loss: 7.6784 - accuracy: 0.4992
16896/25000 [===================>..........] - ETA: 17s - loss: 7.6757 - accuracy: 0.4994
16928/25000 [===================>..........] - ETA: 17s - loss: 7.6793 - accuracy: 0.4992
16960/25000 [===================>..........] - ETA: 17s - loss: 7.6766 - accuracy: 0.4994
16992/25000 [===================>..........] - ETA: 17s - loss: 7.6720 - accuracy: 0.4996
17024/25000 [===================>..........] - ETA: 16s - loss: 7.6738 - accuracy: 0.4995
17056/25000 [===================>..........] - ETA: 16s - loss: 7.6765 - accuracy: 0.4994
17088/25000 [===================>..........] - ETA: 16s - loss: 7.6747 - accuracy: 0.4995
17120/25000 [===================>..........] - ETA: 16s - loss: 7.6756 - accuracy: 0.4994
17152/25000 [===================>..........] - ETA: 16s - loss: 7.6765 - accuracy: 0.4994
17184/25000 [===================>..........] - ETA: 16s - loss: 7.6782 - accuracy: 0.4992
17216/25000 [===================>..........] - ETA: 16s - loss: 7.6791 - accuracy: 0.4992
17248/25000 [===================>..........] - ETA: 16s - loss: 7.6800 - accuracy: 0.4991
17280/25000 [===================>..........] - ETA: 16s - loss: 7.6817 - accuracy: 0.4990
17312/25000 [===================>..........] - ETA: 16s - loss: 7.6834 - accuracy: 0.4989
17344/25000 [===================>..........] - ETA: 16s - loss: 7.6825 - accuracy: 0.4990
17376/25000 [===================>..........] - ETA: 16s - loss: 7.6852 - accuracy: 0.4988
17408/25000 [===================>..........] - ETA: 16s - loss: 7.6895 - accuracy: 0.4985
17440/25000 [===================>..........] - ETA: 16s - loss: 7.6886 - accuracy: 0.4986
17472/25000 [===================>..........] - ETA: 16s - loss: 7.6886 - accuracy: 0.4986
17504/25000 [====================>.........] - ETA: 15s - loss: 7.6903 - accuracy: 0.4985
17536/25000 [====================>.........] - ETA: 15s - loss: 7.6894 - accuracy: 0.4985
17568/25000 [====================>.........] - ETA: 15s - loss: 7.6902 - accuracy: 0.4985
17600/25000 [====================>.........] - ETA: 15s - loss: 7.6893 - accuracy: 0.4985
17632/25000 [====================>.........] - ETA: 15s - loss: 7.6875 - accuracy: 0.4986
17664/25000 [====================>.........] - ETA: 15s - loss: 7.6857 - accuracy: 0.4988
17696/25000 [====================>.........] - ETA: 15s - loss: 7.6874 - accuracy: 0.4986
17728/25000 [====================>.........] - ETA: 15s - loss: 7.6882 - accuracy: 0.4986
17760/25000 [====================>.........] - ETA: 15s - loss: 7.6839 - accuracy: 0.4989
17792/25000 [====================>.........] - ETA: 15s - loss: 7.6847 - accuracy: 0.4988
17824/25000 [====================>.........] - ETA: 15s - loss: 7.6838 - accuracy: 0.4989
17856/25000 [====================>.........] - ETA: 15s - loss: 7.6829 - accuracy: 0.4989
17888/25000 [====================>.........] - ETA: 15s - loss: 7.6803 - accuracy: 0.4991
17920/25000 [====================>.........] - ETA: 15s - loss: 7.6777 - accuracy: 0.4993
17952/25000 [====================>.........] - ETA: 15s - loss: 7.6769 - accuracy: 0.4993
17984/25000 [====================>.........] - ETA: 14s - loss: 7.6717 - accuracy: 0.4997
18016/25000 [====================>.........] - ETA: 14s - loss: 7.6692 - accuracy: 0.4998
18048/25000 [====================>.........] - ETA: 14s - loss: 7.6700 - accuracy: 0.4998
18080/25000 [====================>.........] - ETA: 14s - loss: 7.6709 - accuracy: 0.4997
18112/25000 [====================>.........] - ETA: 14s - loss: 7.6759 - accuracy: 0.4994
18144/25000 [====================>.........] - ETA: 14s - loss: 7.6742 - accuracy: 0.4995
18176/25000 [====================>.........] - ETA: 14s - loss: 7.6717 - accuracy: 0.4997
18208/25000 [====================>.........] - ETA: 14s - loss: 7.6717 - accuracy: 0.4997
18240/25000 [====================>.........] - ETA: 14s - loss: 7.6683 - accuracy: 0.4999
18272/25000 [====================>.........] - ETA: 14s - loss: 7.6700 - accuracy: 0.4998
18304/25000 [====================>.........] - ETA: 14s - loss: 7.6666 - accuracy: 0.5000
18336/25000 [=====================>........] - ETA: 14s - loss: 7.6658 - accuracy: 0.5001
18368/25000 [=====================>........] - ETA: 14s - loss: 7.6683 - accuracy: 0.4999
18400/25000 [=====================>........] - ETA: 14s - loss: 7.6650 - accuracy: 0.5001
18432/25000 [=====================>........] - ETA: 13s - loss: 7.6675 - accuracy: 0.4999
18464/25000 [=====================>........] - ETA: 13s - loss: 7.6683 - accuracy: 0.4999
18496/25000 [=====================>........] - ETA: 13s - loss: 7.6708 - accuracy: 0.4997
18528/25000 [=====================>........] - ETA: 13s - loss: 7.6732 - accuracy: 0.4996
18560/25000 [=====================>........] - ETA: 13s - loss: 7.6798 - accuracy: 0.4991
18592/25000 [=====================>........] - ETA: 13s - loss: 7.6815 - accuracy: 0.4990
18624/25000 [=====================>........] - ETA: 13s - loss: 7.6831 - accuracy: 0.4989
18656/25000 [=====================>........] - ETA: 13s - loss: 7.6847 - accuracy: 0.4988
18688/25000 [=====================>........] - ETA: 13s - loss: 7.6822 - accuracy: 0.4990
18720/25000 [=====================>........] - ETA: 13s - loss: 7.6822 - accuracy: 0.4990
18752/25000 [=====================>........] - ETA: 13s - loss: 7.6838 - accuracy: 0.4989
18784/25000 [=====================>........] - ETA: 13s - loss: 7.6862 - accuracy: 0.4987
18816/25000 [=====================>........] - ETA: 13s - loss: 7.6845 - accuracy: 0.4988
18848/25000 [=====================>........] - ETA: 13s - loss: 7.6845 - accuracy: 0.4988
18880/25000 [=====================>........] - ETA: 13s - loss: 7.6845 - accuracy: 0.4988
18912/25000 [=====================>........] - ETA: 12s - loss: 7.6885 - accuracy: 0.4986
18944/25000 [=====================>........] - ETA: 12s - loss: 7.6869 - accuracy: 0.4987
18976/25000 [=====================>........] - ETA: 12s - loss: 7.6892 - accuracy: 0.4985
19008/25000 [=====================>........] - ETA: 12s - loss: 7.6900 - accuracy: 0.4985
19040/25000 [=====================>........] - ETA: 12s - loss: 7.6876 - accuracy: 0.4986
19072/25000 [=====================>........] - ETA: 12s - loss: 7.6843 - accuracy: 0.4988
19104/25000 [=====================>........] - ETA: 12s - loss: 7.6867 - accuracy: 0.4987
19136/25000 [=====================>........] - ETA: 12s - loss: 7.6867 - accuracy: 0.4987
19168/25000 [======================>.......] - ETA: 12s - loss: 7.6874 - accuracy: 0.4986
19200/25000 [======================>.......] - ETA: 12s - loss: 7.6834 - accuracy: 0.4989
19232/25000 [======================>.......] - ETA: 12s - loss: 7.6826 - accuracy: 0.4990
19264/25000 [======================>.......] - ETA: 12s - loss: 7.6802 - accuracy: 0.4991
19296/25000 [======================>.......] - ETA: 12s - loss: 7.6777 - accuracy: 0.4993
19328/25000 [======================>.......] - ETA: 12s - loss: 7.6738 - accuracy: 0.4995
19360/25000 [======================>.......] - ETA: 11s - loss: 7.6745 - accuracy: 0.4995
19392/25000 [======================>.......] - ETA: 11s - loss: 7.6753 - accuracy: 0.4994
19424/25000 [======================>.......] - ETA: 11s - loss: 7.6737 - accuracy: 0.4995
19456/25000 [======================>.......] - ETA: 11s - loss: 7.6729 - accuracy: 0.4996
19488/25000 [======================>.......] - ETA: 11s - loss: 7.6666 - accuracy: 0.5000
19520/25000 [======================>.......] - ETA: 11s - loss: 7.6650 - accuracy: 0.5001
19552/25000 [======================>.......] - ETA: 11s - loss: 7.6674 - accuracy: 0.4999
19584/25000 [======================>.......] - ETA: 11s - loss: 7.6658 - accuracy: 0.5001
19616/25000 [======================>.......] - ETA: 11s - loss: 7.6658 - accuracy: 0.5001
19648/25000 [======================>.......] - ETA: 11s - loss: 7.6612 - accuracy: 0.5004
19680/25000 [======================>.......] - ETA: 11s - loss: 7.6557 - accuracy: 0.5007
19712/25000 [======================>.......] - ETA: 11s - loss: 7.6565 - accuracy: 0.5007
19744/25000 [======================>.......] - ETA: 11s - loss: 7.6550 - accuracy: 0.5008
19776/25000 [======================>.......] - ETA: 11s - loss: 7.6527 - accuracy: 0.5009
19808/25000 [======================>.......] - ETA: 11s - loss: 7.6527 - accuracy: 0.5009
19840/25000 [======================>.......] - ETA: 10s - loss: 7.6558 - accuracy: 0.5007
19872/25000 [======================>.......] - ETA: 10s - loss: 7.6543 - accuracy: 0.5008
19904/25000 [======================>.......] - ETA: 10s - loss: 7.6551 - accuracy: 0.5008
19936/25000 [======================>.......] - ETA: 10s - loss: 7.6551 - accuracy: 0.5008
19968/25000 [======================>.......] - ETA: 10s - loss: 7.6582 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 10s - loss: 7.6582 - accuracy: 0.5005
20032/25000 [=======================>......] - ETA: 10s - loss: 7.6620 - accuracy: 0.5003
20064/25000 [=======================>......] - ETA: 10s - loss: 7.6620 - accuracy: 0.5003
20096/25000 [=======================>......] - ETA: 10s - loss: 7.6582 - accuracy: 0.5005
20128/25000 [=======================>......] - ETA: 10s - loss: 7.6560 - accuracy: 0.5007
20160/25000 [=======================>......] - ETA: 10s - loss: 7.6567 - accuracy: 0.5006
20192/25000 [=======================>......] - ETA: 10s - loss: 7.6590 - accuracy: 0.5005
20224/25000 [=======================>......] - ETA: 10s - loss: 7.6552 - accuracy: 0.5007
20256/25000 [=======================>......] - ETA: 10s - loss: 7.6568 - accuracy: 0.5006
20288/25000 [=======================>......] - ETA: 10s - loss: 7.6545 - accuracy: 0.5008
20320/25000 [=======================>......] - ETA: 9s - loss: 7.6576 - accuracy: 0.5006 
20352/25000 [=======================>......] - ETA: 9s - loss: 7.6546 - accuracy: 0.5008
20384/25000 [=======================>......] - ETA: 9s - loss: 7.6538 - accuracy: 0.5008
20416/25000 [=======================>......] - ETA: 9s - loss: 7.6554 - accuracy: 0.5007
20448/25000 [=======================>......] - ETA: 9s - loss: 7.6531 - accuracy: 0.5009
20480/25000 [=======================>......] - ETA: 9s - loss: 7.6524 - accuracy: 0.5009
20512/25000 [=======================>......] - ETA: 9s - loss: 7.6509 - accuracy: 0.5010
20544/25000 [=======================>......] - ETA: 9s - loss: 7.6554 - accuracy: 0.5007
20576/25000 [=======================>......] - ETA: 9s - loss: 7.6547 - accuracy: 0.5008
20608/25000 [=======================>......] - ETA: 9s - loss: 7.6540 - accuracy: 0.5008
20640/25000 [=======================>......] - ETA: 9s - loss: 7.6577 - accuracy: 0.5006
20672/25000 [=======================>......] - ETA: 9s - loss: 7.6577 - accuracy: 0.5006
20704/25000 [=======================>......] - ETA: 9s - loss: 7.6600 - accuracy: 0.5004
20736/25000 [=======================>......] - ETA: 9s - loss: 7.6548 - accuracy: 0.5008
20768/25000 [=======================>......] - ETA: 8s - loss: 7.6548 - accuracy: 0.5008
20800/25000 [=======================>......] - ETA: 8s - loss: 7.6511 - accuracy: 0.5010
20832/25000 [=======================>......] - ETA: 8s - loss: 7.6534 - accuracy: 0.5009
20864/25000 [========================>.....] - ETA: 8s - loss: 7.6556 - accuracy: 0.5007
20896/25000 [========================>.....] - ETA: 8s - loss: 7.6541 - accuracy: 0.5008
20928/25000 [========================>.....] - ETA: 8s - loss: 7.6542 - accuracy: 0.5008
20960/25000 [========================>.....] - ETA: 8s - loss: 7.6520 - accuracy: 0.5010
20992/25000 [========================>.....] - ETA: 8s - loss: 7.6498 - accuracy: 0.5011
21024/25000 [========================>.....] - ETA: 8s - loss: 7.6498 - accuracy: 0.5011
21056/25000 [========================>.....] - ETA: 8s - loss: 7.6521 - accuracy: 0.5009
21088/25000 [========================>.....] - ETA: 8s - loss: 7.6506 - accuracy: 0.5010
21120/25000 [========================>.....] - ETA: 8s - loss: 7.6492 - accuracy: 0.5011
21152/25000 [========================>.....] - ETA: 8s - loss: 7.6470 - accuracy: 0.5013
21184/25000 [========================>.....] - ETA: 8s - loss: 7.6456 - accuracy: 0.5014
21216/25000 [========================>.....] - ETA: 8s - loss: 7.6449 - accuracy: 0.5014
21248/25000 [========================>.....] - ETA: 7s - loss: 7.6435 - accuracy: 0.5015
21280/25000 [========================>.....] - ETA: 7s - loss: 7.6450 - accuracy: 0.5014
21312/25000 [========================>.....] - ETA: 7s - loss: 7.6465 - accuracy: 0.5013
21344/25000 [========================>.....] - ETA: 7s - loss: 7.6451 - accuracy: 0.5014
21376/25000 [========================>.....] - ETA: 7s - loss: 7.6487 - accuracy: 0.5012
21408/25000 [========================>.....] - ETA: 7s - loss: 7.6487 - accuracy: 0.5012
21440/25000 [========================>.....] - ETA: 7s - loss: 7.6473 - accuracy: 0.5013
21472/25000 [========================>.....] - ETA: 7s - loss: 7.6452 - accuracy: 0.5014
21504/25000 [========================>.....] - ETA: 7s - loss: 7.6459 - accuracy: 0.5013
21536/25000 [========================>.....] - ETA: 7s - loss: 7.6460 - accuracy: 0.5013
21568/25000 [========================>.....] - ETA: 7s - loss: 7.6474 - accuracy: 0.5013
21600/25000 [========================>.....] - ETA: 7s - loss: 7.6467 - accuracy: 0.5013
21632/25000 [========================>.....] - ETA: 7s - loss: 7.6439 - accuracy: 0.5015
21664/25000 [========================>.....] - ETA: 7s - loss: 7.6440 - accuracy: 0.5015
21696/25000 [=========================>....] - ETA: 7s - loss: 7.6440 - accuracy: 0.5015
21728/25000 [=========================>....] - ETA: 6s - loss: 7.6440 - accuracy: 0.5015
21760/25000 [=========================>....] - ETA: 6s - loss: 7.6469 - accuracy: 0.5013
21792/25000 [=========================>....] - ETA: 6s - loss: 7.6462 - accuracy: 0.5013
21824/25000 [=========================>....] - ETA: 6s - loss: 7.6484 - accuracy: 0.5012
21856/25000 [=========================>....] - ETA: 6s - loss: 7.6519 - accuracy: 0.5010
21888/25000 [=========================>....] - ETA: 6s - loss: 7.6540 - accuracy: 0.5008
21920/25000 [=========================>....] - ETA: 6s - loss: 7.6575 - accuracy: 0.5006
21952/25000 [=========================>....] - ETA: 6s - loss: 7.6561 - accuracy: 0.5007
21984/25000 [=========================>....] - ETA: 6s - loss: 7.6555 - accuracy: 0.5007
22016/25000 [=========================>....] - ETA: 6s - loss: 7.6555 - accuracy: 0.5007
22048/25000 [=========================>....] - ETA: 6s - loss: 7.6562 - accuracy: 0.5007
22080/25000 [=========================>....] - ETA: 6s - loss: 7.6548 - accuracy: 0.5008
22112/25000 [=========================>....] - ETA: 6s - loss: 7.6528 - accuracy: 0.5009
22144/25000 [=========================>....] - ETA: 6s - loss: 7.6528 - accuracy: 0.5009
22176/25000 [=========================>....] - ETA: 5s - loss: 7.6521 - accuracy: 0.5009
22208/25000 [=========================>....] - ETA: 5s - loss: 7.6507 - accuracy: 0.5010
22240/25000 [=========================>....] - ETA: 5s - loss: 7.6515 - accuracy: 0.5010
22272/25000 [=========================>....] - ETA: 5s - loss: 7.6515 - accuracy: 0.5010
22304/25000 [=========================>....] - ETA: 5s - loss: 7.6542 - accuracy: 0.5008
22336/25000 [=========================>....] - ETA: 5s - loss: 7.6529 - accuracy: 0.5009
22368/25000 [=========================>....] - ETA: 5s - loss: 7.6522 - accuracy: 0.5009
22400/25000 [=========================>....] - ETA: 5s - loss: 7.6557 - accuracy: 0.5007
22432/25000 [=========================>....] - ETA: 5s - loss: 7.6564 - accuracy: 0.5007
22464/25000 [=========================>....] - ETA: 5s - loss: 7.6564 - accuracy: 0.5007
22496/25000 [=========================>....] - ETA: 5s - loss: 7.6584 - accuracy: 0.5005
22528/25000 [==========================>...] - ETA: 5s - loss: 7.6591 - accuracy: 0.5005
22560/25000 [==========================>...] - ETA: 5s - loss: 7.6598 - accuracy: 0.5004
22592/25000 [==========================>...] - ETA: 5s - loss: 7.6625 - accuracy: 0.5003
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6639 - accuracy: 0.5002
22656/25000 [==========================>...] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
22688/25000 [==========================>...] - ETA: 4s - loss: 7.6626 - accuracy: 0.5003
22720/25000 [==========================>...] - ETA: 4s - loss: 7.6572 - accuracy: 0.5006
22752/25000 [==========================>...] - ETA: 4s - loss: 7.6565 - accuracy: 0.5007
22784/25000 [==========================>...] - ETA: 4s - loss: 7.6559 - accuracy: 0.5007
22816/25000 [==========================>...] - ETA: 4s - loss: 7.6565 - accuracy: 0.5007
22848/25000 [==========================>...] - ETA: 4s - loss: 7.6599 - accuracy: 0.5004
22880/25000 [==========================>...] - ETA: 4s - loss: 7.6613 - accuracy: 0.5003
22912/25000 [==========================>...] - ETA: 4s - loss: 7.6579 - accuracy: 0.5006
22944/25000 [==========================>...] - ETA: 4s - loss: 7.6559 - accuracy: 0.5007
22976/25000 [==========================>...] - ETA: 4s - loss: 7.6586 - accuracy: 0.5005
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6560 - accuracy: 0.5007
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6586 - accuracy: 0.5005
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6580 - accuracy: 0.5006
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6567 - accuracy: 0.5006
23136/25000 [==========================>...] - ETA: 3s - loss: 7.6560 - accuracy: 0.5007
23168/25000 [==========================>...] - ETA: 3s - loss: 7.6560 - accuracy: 0.5007
23200/25000 [==========================>...] - ETA: 3s - loss: 7.6574 - accuracy: 0.5006
23232/25000 [==========================>...] - ETA: 3s - loss: 7.6580 - accuracy: 0.5006
23264/25000 [==========================>...] - ETA: 3s - loss: 7.6554 - accuracy: 0.5007
23296/25000 [==========================>...] - ETA: 3s - loss: 7.6548 - accuracy: 0.5008
23328/25000 [==========================>...] - ETA: 3s - loss: 7.6541 - accuracy: 0.5008
23360/25000 [===========================>..] - ETA: 3s - loss: 7.6541 - accuracy: 0.5008
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6529 - accuracy: 0.5009
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6535 - accuracy: 0.5009
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6522 - accuracy: 0.5009
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6542 - accuracy: 0.5008
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6555 - accuracy: 0.5007
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6549 - accuracy: 0.5008
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6536 - accuracy: 0.5008
23616/25000 [===========================>..] - ETA: 2s - loss: 7.6556 - accuracy: 0.5007
23648/25000 [===========================>..] - ETA: 2s - loss: 7.6575 - accuracy: 0.5006
23680/25000 [===========================>..] - ETA: 2s - loss: 7.6601 - accuracy: 0.5004
23712/25000 [===========================>..] - ETA: 2s - loss: 7.6589 - accuracy: 0.5005
23744/25000 [===========================>..] - ETA: 2s - loss: 7.6576 - accuracy: 0.5006
23776/25000 [===========================>..] - ETA: 2s - loss: 7.6557 - accuracy: 0.5007
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6537 - accuracy: 0.5008
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6518 - accuracy: 0.5010
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6544 - accuracy: 0.5008
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6538 - accuracy: 0.5008
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6538 - accuracy: 0.5008
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6545 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6570 - accuracy: 0.5006
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6564 - accuracy: 0.5007
24064/25000 [===========================>..] - ETA: 1s - loss: 7.6577 - accuracy: 0.5006
24096/25000 [===========================>..] - ETA: 1s - loss: 7.6564 - accuracy: 0.5007
24128/25000 [===========================>..] - ETA: 1s - loss: 7.6558 - accuracy: 0.5007
24160/25000 [===========================>..] - ETA: 1s - loss: 7.6577 - accuracy: 0.5006
24192/25000 [============================>.] - ETA: 1s - loss: 7.6596 - accuracy: 0.5005
24224/25000 [============================>.] - ETA: 1s - loss: 7.6609 - accuracy: 0.5004
24256/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24288/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24320/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24352/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24384/25000 [============================>.] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
24416/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24448/25000 [============================>.] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
24480/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24512/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24544/25000 [============================>.] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
24576/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24608/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24640/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
24672/25000 [============================>.] - ETA: 0s - loss: 7.6592 - accuracy: 0.5005
24704/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
24736/25000 [============================>.] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24768/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24800/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24832/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24864/25000 [============================>.] - ETA: 0s - loss: 7.6605 - accuracy: 0.5004
24896/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24928/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24960/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 62s 2ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fdb15b27a90> 

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
 [ 0.02336767 -0.02754975  0.07785628 -0.03454577  0.13681234 -0.03034814]
 [ 0.00500379 -0.00858059 -0.08884755  0.16112067  0.08067913  0.05195656]
 [-0.29306412  0.27297488  0.15999267 -0.07725176  0.00374335  0.220035  ]
 [-0.25332794  0.27475682  0.44016999  0.01857609  0.01766681  0.28302762]
 [ 0.56096631  0.35210493  0.41175768 -0.25287962  0.62359184 -0.01607334]
 [ 0.06895956  0.33671427  0.42134318  0.63433766  0.60917938  0.05152101]
 [-0.14173484 -0.43064111  0.16277489  0.08231883  0.16187964  0.1182865 ]
 [-0.08759314 -0.09263495  0.01058319  0.00997771 -0.07872538  0.13008523]
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
{'loss': 0.42844728380441666, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
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
{'loss': 0.5640210509300232, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}





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
	Data preprocessing and feature engineering runtime = 0.21s ...
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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:42<01:03, 21.29s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.1282415887906346, 'embedding_size_factor': 1.3505215555471533, 'layers.choice': 2, 'learning_rate': 0.002258836105927261, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 4.867325495681328e-12} and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc0j8j\xeb\xe2\xb3X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x9b\xbc}\x99\xe3\x9eX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?b\x81\x1ffv\xa2\x12X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x95h\x1f\x0e\xf8\x16\xecu.' and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc0j8j\xeb\xe2\xb3X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x9b\xbc}\x99\xe3\x9eX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?b\x81\x1ffv\xa2\x12X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x95h\x1f\x0e\xf8\x16\xecu.' and reward: 0.39
 60%|██████    | 3/5 [01:26<00:56, 28.02s/it] 60%|██████    | 3/5 [01:26<00:57, 28.77s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.300392738082621, 'embedding_size_factor': 1.392971702574705, 'layers.choice': 0, 'learning_rate': 0.005433540754484251, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 0.000177741832128413} and reward: 0.3764
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd39\xa2v\x81S\xfcX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6I\x9c\xb2,\xfd\x1fX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?vAz\xfdx\xcd\xadX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?'L\x06\xb6U\xb5qu." and reward: 0.3764
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd39\xa2v\x81S\xfcX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6I\x9c\xb2,\xfd\x1fX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?vAz\xfdx\xcd\xadX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?'L\x06\xb6U\xb5qu." and reward: 0.3764
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 129.57768082618713
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 1, 'dropout_prob': 0.1282415887906346, 'embedding_size_factor': 1.3505215555471533, 'layers.choice': 2, 'learning_rate': 0.002258836105927261, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 4.867325495681328e-12}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.79s of the -11.71s of remaining time.
Ensemble size: 16
Ensemble weights: 
[0.4375 0.375  0.1875]
	0.391	 = Validation accuracy score
	0.89s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 132.64s ...
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

