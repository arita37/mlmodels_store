
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
 2859008/17464789 [===>..........................] - ETA: 0s
12361728/17464789 [====================>.........] - ETA: 0s
17375232/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 10:13:31.385303: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 10:13:31.390591: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-11 10:13:31.391104: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f3c914ad80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 10:13:31.391428: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:54 - loss: 7.6666 - accuracy: 0.5000
   64/25000 [..............................] - ETA: 3:10 - loss: 7.1875 - accuracy: 0.5312
   96/25000 [..............................] - ETA: 2:34 - loss: 7.1875 - accuracy: 0.5312
  128/25000 [..............................] - ETA: 2:17 - loss: 7.3072 - accuracy: 0.5234
  160/25000 [..............................] - ETA: 2:06 - loss: 6.8041 - accuracy: 0.5562
  192/25000 [..............................] - ETA: 1:59 - loss: 6.6284 - accuracy: 0.5677
  224/25000 [..............................] - ETA: 1:54 - loss: 6.7767 - accuracy: 0.5580
  256/25000 [..............................] - ETA: 1:51 - loss: 7.1276 - accuracy: 0.5352
  288/25000 [..............................] - ETA: 1:47 - loss: 7.1875 - accuracy: 0.5312
  320/25000 [..............................] - ETA: 1:45 - loss: 7.3791 - accuracy: 0.5188
  352/25000 [..............................] - ETA: 1:44 - loss: 7.4488 - accuracy: 0.5142
  384/25000 [..............................] - ETA: 1:43 - loss: 7.3871 - accuracy: 0.5182
  416/25000 [..............................] - ETA: 1:41 - loss: 7.4086 - accuracy: 0.5168
  448/25000 [..............................] - ETA: 1:39 - loss: 7.4955 - accuracy: 0.5112
  480/25000 [..............................] - ETA: 1:38 - loss: 7.3472 - accuracy: 0.5208
  512/25000 [..............................] - ETA: 1:37 - loss: 7.4869 - accuracy: 0.5117
  544/25000 [..............................] - ETA: 1:36 - loss: 7.3566 - accuracy: 0.5202
  576/25000 [..............................] - ETA: 1:36 - loss: 7.4004 - accuracy: 0.5174
  608/25000 [..............................] - ETA: 1:35 - loss: 7.4649 - accuracy: 0.5132
  640/25000 [..............................] - ETA: 1:34 - loss: 7.4510 - accuracy: 0.5141
  672/25000 [..............................] - ETA: 1:33 - loss: 7.4841 - accuracy: 0.5119
  704/25000 [..............................] - ETA: 1:33 - loss: 7.4488 - accuracy: 0.5142
  736/25000 [..............................] - ETA: 1:32 - loss: 7.4166 - accuracy: 0.5163
  768/25000 [..............................] - ETA: 1:32 - loss: 7.4869 - accuracy: 0.5117
  800/25000 [..............................] - ETA: 1:31 - loss: 7.4750 - accuracy: 0.5125
  832/25000 [..............................] - ETA: 1:31 - loss: 7.5192 - accuracy: 0.5096
  864/25000 [>.............................] - ETA: 1:30 - loss: 7.5956 - accuracy: 0.5046
  896/25000 [>.............................] - ETA: 1:30 - loss: 7.5982 - accuracy: 0.5045
  928/25000 [>.............................] - ETA: 1:30 - loss: 7.6005 - accuracy: 0.5043
  960/25000 [>.............................] - ETA: 1:29 - loss: 7.5708 - accuracy: 0.5063
  992/25000 [>.............................] - ETA: 1:29 - loss: 7.5739 - accuracy: 0.5060
 1024/25000 [>.............................] - ETA: 1:29 - loss: 7.5618 - accuracy: 0.5068
 1056/25000 [>.............................] - ETA: 1:29 - loss: 7.5214 - accuracy: 0.5095
 1088/25000 [>.............................] - ETA: 1:28 - loss: 7.5257 - accuracy: 0.5092
 1120/25000 [>.............................] - ETA: 1:28 - loss: 7.5434 - accuracy: 0.5080
 1152/25000 [>.............................] - ETA: 1:28 - loss: 7.5335 - accuracy: 0.5087
 1184/25000 [>.............................] - ETA: 1:28 - loss: 7.5371 - accuracy: 0.5084
 1216/25000 [>.............................] - ETA: 1:27 - loss: 7.5531 - accuracy: 0.5074
 1248/25000 [>.............................] - ETA: 1:27 - loss: 7.5560 - accuracy: 0.5072
 1280/25000 [>.............................] - ETA: 1:27 - loss: 7.6187 - accuracy: 0.5031
 1312/25000 [>.............................] - ETA: 1:27 - loss: 7.6199 - accuracy: 0.5030
 1344/25000 [>.............................] - ETA: 1:26 - loss: 7.6210 - accuracy: 0.5030
 1376/25000 [>.............................] - ETA: 1:26 - loss: 7.6443 - accuracy: 0.5015
 1408/25000 [>.............................] - ETA: 1:26 - loss: 7.6557 - accuracy: 0.5007
 1440/25000 [>.............................] - ETA: 1:26 - loss: 7.6134 - accuracy: 0.5035
 1472/25000 [>.............................] - ETA: 1:25 - loss: 7.5833 - accuracy: 0.5054
 1504/25000 [>.............................] - ETA: 1:25 - loss: 7.6258 - accuracy: 0.5027
 1536/25000 [>.............................] - ETA: 1:25 - loss: 7.6367 - accuracy: 0.5020
 1568/25000 [>.............................] - ETA: 1:25 - loss: 7.5688 - accuracy: 0.5064
 1600/25000 [>.............................] - ETA: 1:25 - loss: 7.5612 - accuracy: 0.5069
 1632/25000 [>.............................] - ETA: 1:24 - loss: 7.5351 - accuracy: 0.5086
 1664/25000 [>.............................] - ETA: 1:24 - loss: 7.5284 - accuracy: 0.5090
 1696/25000 [=>............................] - ETA: 1:24 - loss: 7.5310 - accuracy: 0.5088
 1728/25000 [=>............................] - ETA: 1:24 - loss: 7.5335 - accuracy: 0.5087
 1760/25000 [=>............................] - ETA: 1:24 - loss: 7.5708 - accuracy: 0.5063
 1792/25000 [=>............................] - ETA: 1:23 - loss: 7.5811 - accuracy: 0.5056
 1824/25000 [=>............................] - ETA: 1:23 - loss: 7.6330 - accuracy: 0.5022
 1856/25000 [=>............................] - ETA: 1:23 - loss: 7.6501 - accuracy: 0.5011
 1888/25000 [=>............................] - ETA: 1:23 - loss: 7.6666 - accuracy: 0.5000
 1920/25000 [=>............................] - ETA: 1:23 - loss: 7.6506 - accuracy: 0.5010
 1952/25000 [=>............................] - ETA: 1:23 - loss: 7.6352 - accuracy: 0.5020
 1984/25000 [=>............................] - ETA: 1:23 - loss: 7.6434 - accuracy: 0.5015
 2016/25000 [=>............................] - ETA: 1:23 - loss: 7.6970 - accuracy: 0.4980
 2048/25000 [=>............................] - ETA: 1:22 - loss: 7.7265 - accuracy: 0.4961
 2080/25000 [=>............................] - ETA: 1:22 - loss: 7.7256 - accuracy: 0.4962
 2112/25000 [=>............................] - ETA: 1:22 - loss: 7.7174 - accuracy: 0.4967
 2144/25000 [=>............................] - ETA: 1:22 - loss: 7.7024 - accuracy: 0.4977
 2176/25000 [=>............................] - ETA: 1:22 - loss: 7.7230 - accuracy: 0.4963
 2208/25000 [=>............................] - ETA: 1:22 - loss: 7.7013 - accuracy: 0.4977
 2240/25000 [=>............................] - ETA: 1:22 - loss: 7.7077 - accuracy: 0.4973
 2272/25000 [=>............................] - ETA: 1:21 - loss: 7.7341 - accuracy: 0.4956
 2304/25000 [=>............................] - ETA: 1:21 - loss: 7.7199 - accuracy: 0.4965
 2336/25000 [=>............................] - ETA: 1:21 - loss: 7.7454 - accuracy: 0.4949
 2368/25000 [=>............................] - ETA: 1:21 - loss: 7.7443 - accuracy: 0.4949
 2400/25000 [=>............................] - ETA: 1:21 - loss: 7.7305 - accuracy: 0.4958
 2432/25000 [=>............................] - ETA: 1:21 - loss: 7.7297 - accuracy: 0.4959
 2464/25000 [=>............................] - ETA: 1:21 - loss: 7.7164 - accuracy: 0.4968
 2496/25000 [=>............................] - ETA: 1:20 - loss: 7.7219 - accuracy: 0.4964
 2528/25000 [==>...........................] - ETA: 1:20 - loss: 7.7151 - accuracy: 0.4968
 2560/25000 [==>...........................] - ETA: 1:20 - loss: 7.7026 - accuracy: 0.4977
 2592/25000 [==>...........................] - ETA: 1:20 - loss: 7.6903 - accuracy: 0.4985
 2624/25000 [==>...........................] - ETA: 1:20 - loss: 7.6900 - accuracy: 0.4985
 2656/25000 [==>...........................] - ETA: 1:20 - loss: 7.6955 - accuracy: 0.4981
 2688/25000 [==>...........................] - ETA: 1:19 - loss: 7.6780 - accuracy: 0.4993
 2720/25000 [==>...........................] - ETA: 1:19 - loss: 7.6779 - accuracy: 0.4993
 2752/25000 [==>...........................] - ETA: 1:19 - loss: 7.6610 - accuracy: 0.5004
 2784/25000 [==>...........................] - ETA: 1:19 - loss: 7.6391 - accuracy: 0.5018
 2816/25000 [==>...........................] - ETA: 1:19 - loss: 7.6285 - accuracy: 0.5025
 2848/25000 [==>...........................] - ETA: 1:19 - loss: 7.6397 - accuracy: 0.5018
 2880/25000 [==>...........................] - ETA: 1:19 - loss: 7.6294 - accuracy: 0.5024
 2912/25000 [==>...........................] - ETA: 1:18 - loss: 7.5982 - accuracy: 0.5045
 2944/25000 [==>...........................] - ETA: 1:18 - loss: 7.5937 - accuracy: 0.5048
 2976/25000 [==>...........................] - ETA: 1:18 - loss: 7.5945 - accuracy: 0.5047
 3008/25000 [==>...........................] - ETA: 1:18 - loss: 7.5851 - accuracy: 0.5053
 3040/25000 [==>...........................] - ETA: 1:18 - loss: 7.5859 - accuracy: 0.5053
 3072/25000 [==>...........................] - ETA: 1:18 - loss: 7.5618 - accuracy: 0.5068
 3104/25000 [==>...........................] - ETA: 1:18 - loss: 7.5777 - accuracy: 0.5058
 3136/25000 [==>...........................] - ETA: 1:18 - loss: 7.5786 - accuracy: 0.5057
 3168/25000 [==>...........................] - ETA: 1:17 - loss: 7.5601 - accuracy: 0.5069
 3200/25000 [==>...........................] - ETA: 1:17 - loss: 7.5756 - accuracy: 0.5059
 3232/25000 [==>...........................] - ETA: 1:17 - loss: 7.5907 - accuracy: 0.5050
 3264/25000 [==>...........................] - ETA: 1:17 - loss: 7.6009 - accuracy: 0.5043
 3296/25000 [==>...........................] - ETA: 1:17 - loss: 7.6015 - accuracy: 0.5042
 3328/25000 [==>...........................] - ETA: 1:17 - loss: 7.5883 - accuracy: 0.5051
 3360/25000 [===>..........................] - ETA: 1:17 - loss: 7.5982 - accuracy: 0.5045
 3392/25000 [===>..........................] - ETA: 1:17 - loss: 7.6033 - accuracy: 0.5041
 3424/25000 [===>..........................] - ETA: 1:16 - loss: 7.5950 - accuracy: 0.5047
 3456/25000 [===>..........................] - ETA: 1:16 - loss: 7.5956 - accuracy: 0.5046
 3488/25000 [===>..........................] - ETA: 1:16 - loss: 7.5875 - accuracy: 0.5052
 3520/25000 [===>..........................] - ETA: 1:16 - loss: 7.5664 - accuracy: 0.5065
 3552/25000 [===>..........................] - ETA: 1:16 - loss: 7.5717 - accuracy: 0.5062
 3584/25000 [===>..........................] - ETA: 1:16 - loss: 7.5639 - accuracy: 0.5067
 3616/25000 [===>..........................] - ETA: 1:16 - loss: 7.5776 - accuracy: 0.5058
 3648/25000 [===>..........................] - ETA: 1:16 - loss: 7.5784 - accuracy: 0.5058
 3680/25000 [===>..........................] - ETA: 1:15 - loss: 7.5666 - accuracy: 0.5065
 3712/25000 [===>..........................] - ETA: 1:15 - loss: 7.5592 - accuracy: 0.5070
 3744/25000 [===>..........................] - ETA: 1:15 - loss: 7.5683 - accuracy: 0.5064
 3776/25000 [===>..........................] - ETA: 1:15 - loss: 7.5570 - accuracy: 0.5072
 3808/25000 [===>..........................] - ETA: 1:15 - loss: 7.5418 - accuracy: 0.5081
 3840/25000 [===>..........................] - ETA: 1:15 - loss: 7.5428 - accuracy: 0.5081
 3872/25000 [===>..........................] - ETA: 1:15 - loss: 7.5399 - accuracy: 0.5083
 3904/25000 [===>..........................] - ETA: 1:15 - loss: 7.5252 - accuracy: 0.5092
 3936/25000 [===>..........................] - ETA: 1:14 - loss: 7.5381 - accuracy: 0.5084
 3968/25000 [===>..........................] - ETA: 1:14 - loss: 7.5430 - accuracy: 0.5081
 4000/25000 [===>..........................] - ETA: 1:14 - loss: 7.5286 - accuracy: 0.5090
 4032/25000 [===>..........................] - ETA: 1:14 - loss: 7.5259 - accuracy: 0.5092
 4064/25000 [===>..........................] - ETA: 1:14 - loss: 7.5157 - accuracy: 0.5098
 4096/25000 [===>..........................] - ETA: 1:14 - loss: 7.5094 - accuracy: 0.5103
 4128/25000 [===>..........................] - ETA: 1:14 - loss: 7.5218 - accuracy: 0.5094
 4160/25000 [===>..........................] - ETA: 1:14 - loss: 7.5376 - accuracy: 0.5084
 4192/25000 [====>.........................] - ETA: 1:13 - loss: 7.5203 - accuracy: 0.5095
 4224/25000 [====>.........................] - ETA: 1:13 - loss: 7.5142 - accuracy: 0.5099
 4256/25000 [====>.........................] - ETA: 1:13 - loss: 7.5045 - accuracy: 0.5106
 4288/25000 [====>.........................] - ETA: 1:13 - loss: 7.5057 - accuracy: 0.5105
 4320/25000 [====>.........................] - ETA: 1:13 - loss: 7.5033 - accuracy: 0.5106
 4352/25000 [====>.........................] - ETA: 1:13 - loss: 7.5222 - accuracy: 0.5094
 4384/25000 [====>.........................] - ETA: 1:13 - loss: 7.5197 - accuracy: 0.5096
 4416/25000 [====>.........................] - ETA: 1:13 - loss: 7.5277 - accuracy: 0.5091
 4448/25000 [====>.........................] - ETA: 1:12 - loss: 7.5460 - accuracy: 0.5079
 4480/25000 [====>.........................] - ETA: 1:12 - loss: 7.5503 - accuracy: 0.5076
 4512/25000 [====>.........................] - ETA: 1:12 - loss: 7.5545 - accuracy: 0.5073
 4544/25000 [====>.........................] - ETA: 1:12 - loss: 7.5586 - accuracy: 0.5070
 4576/25000 [====>.........................] - ETA: 1:12 - loss: 7.5661 - accuracy: 0.5066
 4608/25000 [====>.........................] - ETA: 1:12 - loss: 7.5502 - accuracy: 0.5076
 4640/25000 [====>.........................] - ETA: 1:12 - loss: 7.5510 - accuracy: 0.5075
 4672/25000 [====>.........................] - ETA: 1:12 - loss: 7.5649 - accuracy: 0.5066
 4704/25000 [====>.........................] - ETA: 1:12 - loss: 7.5754 - accuracy: 0.5060
 4736/25000 [====>.........................] - ETA: 1:11 - loss: 7.5792 - accuracy: 0.5057
 4768/25000 [====>.........................] - ETA: 1:11 - loss: 7.5766 - accuracy: 0.5059
 4800/25000 [====>.........................] - ETA: 1:11 - loss: 7.5804 - accuracy: 0.5056
 4832/25000 [====>.........................] - ETA: 1:11 - loss: 7.5809 - accuracy: 0.5056
 4864/25000 [====>.........................] - ETA: 1:11 - loss: 7.5815 - accuracy: 0.5056
 4896/25000 [====>.........................] - ETA: 1:11 - loss: 7.5852 - accuracy: 0.5053
 4928/25000 [====>.........................] - ETA: 1:11 - loss: 7.5888 - accuracy: 0.5051
 4960/25000 [====>.........................] - ETA: 1:11 - loss: 7.5924 - accuracy: 0.5048
 4992/25000 [====>.........................] - ETA: 1:11 - loss: 7.5990 - accuracy: 0.5044
 5024/25000 [=====>........................] - ETA: 1:11 - loss: 7.5964 - accuracy: 0.5046
 5056/25000 [=====>........................] - ETA: 1:10 - loss: 7.5756 - accuracy: 0.5059
 5088/25000 [=====>........................] - ETA: 1:10 - loss: 7.5792 - accuracy: 0.5057
 5120/25000 [=====>........................] - ETA: 1:10 - loss: 7.5798 - accuracy: 0.5057
 5152/25000 [=====>........................] - ETA: 1:10 - loss: 7.5684 - accuracy: 0.5064
 5184/25000 [=====>........................] - ETA: 1:10 - loss: 7.5808 - accuracy: 0.5056
 5216/25000 [=====>........................] - ETA: 1:10 - loss: 7.5872 - accuracy: 0.5052
 5248/25000 [=====>........................] - ETA: 1:10 - loss: 7.5760 - accuracy: 0.5059
 5280/25000 [=====>........................] - ETA: 1:10 - loss: 7.5679 - accuracy: 0.5064
 5312/25000 [=====>........................] - ETA: 1:09 - loss: 7.5887 - accuracy: 0.5051
 5344/25000 [=====>........................] - ETA: 1:09 - loss: 7.5891 - accuracy: 0.5051
 5376/25000 [=====>........................] - ETA: 1:09 - loss: 7.6010 - accuracy: 0.5043
 5408/25000 [=====>........................] - ETA: 1:09 - loss: 7.5986 - accuracy: 0.5044
 5440/25000 [=====>........................] - ETA: 1:09 - loss: 7.6187 - accuracy: 0.5031
 5472/25000 [=====>........................] - ETA: 1:09 - loss: 7.6274 - accuracy: 0.5026
 5504/25000 [=====>........................] - ETA: 1:09 - loss: 7.6304 - accuracy: 0.5024
 5536/25000 [=====>........................] - ETA: 1:09 - loss: 7.6334 - accuracy: 0.5022
 5568/25000 [=====>........................] - ETA: 1:08 - loss: 7.6363 - accuracy: 0.5020
 5600/25000 [=====>........................] - ETA: 1:08 - loss: 7.6365 - accuracy: 0.5020
 5632/25000 [=====>........................] - ETA: 1:08 - loss: 7.6421 - accuracy: 0.5016
 5664/25000 [=====>........................] - ETA: 1:08 - loss: 7.6341 - accuracy: 0.5021
 5696/25000 [=====>........................] - ETA: 1:08 - loss: 7.6316 - accuracy: 0.5023
 5728/25000 [=====>........................] - ETA: 1:08 - loss: 7.6345 - accuracy: 0.5021
 5760/25000 [=====>........................] - ETA: 1:08 - loss: 7.6373 - accuracy: 0.5019
 5792/25000 [=====>........................] - ETA: 1:08 - loss: 7.6322 - accuracy: 0.5022
 5824/25000 [=====>........................] - ETA: 1:08 - loss: 7.6271 - accuracy: 0.5026
 5856/25000 [======>.......................] - ETA: 1:07 - loss: 7.6273 - accuracy: 0.5026
 5888/25000 [======>.......................] - ETA: 1:07 - loss: 7.6197 - accuracy: 0.5031
 5920/25000 [======>.......................] - ETA: 1:07 - loss: 7.6252 - accuracy: 0.5027
 5952/25000 [======>.......................] - ETA: 1:07 - loss: 7.6202 - accuracy: 0.5030
 5984/25000 [======>.......................] - ETA: 1:07 - loss: 7.6179 - accuracy: 0.5032
 6016/25000 [======>.......................] - ETA: 1:07 - loss: 7.6131 - accuracy: 0.5035
 6048/25000 [======>.......................] - ETA: 1:07 - loss: 7.6210 - accuracy: 0.5030
 6080/25000 [======>.......................] - ETA: 1:07 - loss: 7.6137 - accuracy: 0.5035
 6112/25000 [======>.......................] - ETA: 1:06 - loss: 7.6164 - accuracy: 0.5033
 6144/25000 [======>.......................] - ETA: 1:06 - loss: 7.6167 - accuracy: 0.5033
 6176/25000 [======>.......................] - ETA: 1:06 - loss: 7.6294 - accuracy: 0.5024
 6208/25000 [======>.......................] - ETA: 1:06 - loss: 7.6271 - accuracy: 0.5026
 6240/25000 [======>.......................] - ETA: 1:06 - loss: 7.6224 - accuracy: 0.5029
 6272/25000 [======>.......................] - ETA: 1:06 - loss: 7.6202 - accuracy: 0.5030
 6304/25000 [======>.......................] - ETA: 1:06 - loss: 7.6107 - accuracy: 0.5036
 6336/25000 [======>.......................] - ETA: 1:06 - loss: 7.6061 - accuracy: 0.5039
 6368/25000 [======>.......................] - ETA: 1:05 - loss: 7.6088 - accuracy: 0.5038
 6400/25000 [======>.......................] - ETA: 1:05 - loss: 7.6067 - accuracy: 0.5039
 6432/25000 [======>.......................] - ETA: 1:05 - loss: 7.6118 - accuracy: 0.5036
 6464/25000 [======>.......................] - ETA: 1:05 - loss: 7.6073 - accuracy: 0.5039
 6496/25000 [======>.......................] - ETA: 1:05 - loss: 7.6076 - accuracy: 0.5038
 6528/25000 [======>.......................] - ETA: 1:05 - loss: 7.6126 - accuracy: 0.5035
 6560/25000 [======>.......................] - ETA: 1:05 - loss: 7.6105 - accuracy: 0.5037
 6592/25000 [======>.......................] - ETA: 1:05 - loss: 7.6061 - accuracy: 0.5039
 6624/25000 [======>.......................] - ETA: 1:04 - loss: 7.6134 - accuracy: 0.5035
 6656/25000 [======>.......................] - ETA: 1:04 - loss: 7.6344 - accuracy: 0.5021
 6688/25000 [=======>......................] - ETA: 1:04 - loss: 7.6322 - accuracy: 0.5022
 6720/25000 [=======>......................] - ETA: 1:04 - loss: 7.6278 - accuracy: 0.5025
 6752/25000 [=======>......................] - ETA: 1:04 - loss: 7.6303 - accuracy: 0.5024
 6784/25000 [=======>......................] - ETA: 1:04 - loss: 7.6237 - accuracy: 0.5028
 6816/25000 [=======>......................] - ETA: 1:04 - loss: 7.6239 - accuracy: 0.5028
 6848/25000 [=======>......................] - ETA: 1:04 - loss: 7.6330 - accuracy: 0.5022
 6880/25000 [=======>......................] - ETA: 1:04 - loss: 7.6287 - accuracy: 0.5025
 6912/25000 [=======>......................] - ETA: 1:04 - loss: 7.6200 - accuracy: 0.5030
 6944/25000 [=======>......................] - ETA: 1:03 - loss: 7.6247 - accuracy: 0.5027
 6976/25000 [=======>......................] - ETA: 1:03 - loss: 7.6380 - accuracy: 0.5019
 7008/25000 [=======>......................] - ETA: 1:03 - loss: 7.6338 - accuracy: 0.5021
 7040/25000 [=======>......................] - ETA: 1:03 - loss: 7.6361 - accuracy: 0.5020
 7072/25000 [=======>......................] - ETA: 1:03 - loss: 7.6493 - accuracy: 0.5011
 7104/25000 [=======>......................] - ETA: 1:03 - loss: 7.6472 - accuracy: 0.5013
 7136/25000 [=======>......................] - ETA: 1:03 - loss: 7.6430 - accuracy: 0.5015
 7168/25000 [=======>......................] - ETA: 1:03 - loss: 7.6474 - accuracy: 0.5013
 7200/25000 [=======>......................] - ETA: 1:02 - loss: 7.6517 - accuracy: 0.5010
 7232/25000 [=======>......................] - ETA: 1:02 - loss: 7.6497 - accuracy: 0.5011
 7264/25000 [=======>......................] - ETA: 1:02 - loss: 7.6518 - accuracy: 0.5010
 7296/25000 [=======>......................] - ETA: 1:02 - loss: 7.6393 - accuracy: 0.5018
 7328/25000 [=======>......................] - ETA: 1:02 - loss: 7.6352 - accuracy: 0.5020
 7360/25000 [=======>......................] - ETA: 1:02 - loss: 7.6270 - accuracy: 0.5026
 7392/25000 [=======>......................] - ETA: 1:02 - loss: 7.6314 - accuracy: 0.5023
 7424/25000 [=======>......................] - ETA: 1:02 - loss: 7.6212 - accuracy: 0.5030
 7456/25000 [=======>......................] - ETA: 1:02 - loss: 7.6132 - accuracy: 0.5035
 7488/25000 [=======>......................] - ETA: 1:01 - loss: 7.6175 - accuracy: 0.5032
 7520/25000 [========>.....................] - ETA: 1:01 - loss: 7.6156 - accuracy: 0.5033
 7552/25000 [========>.....................] - ETA: 1:01 - loss: 7.6159 - accuracy: 0.5033
 7584/25000 [========>.....................] - ETA: 1:01 - loss: 7.6100 - accuracy: 0.5037
 7616/25000 [========>.....................] - ETA: 1:01 - loss: 7.6143 - accuracy: 0.5034
 7648/25000 [========>.....................] - ETA: 1:01 - loss: 7.6105 - accuracy: 0.5037
 7680/25000 [========>.....................] - ETA: 1:01 - loss: 7.6047 - accuracy: 0.5040
 7712/25000 [========>.....................] - ETA: 1:01 - loss: 7.6129 - accuracy: 0.5035
 7744/25000 [========>.....................] - ETA: 1:00 - loss: 7.6132 - accuracy: 0.5035
 7776/25000 [========>.....................] - ETA: 1:00 - loss: 7.6075 - accuracy: 0.5039
 7808/25000 [========>.....................] - ETA: 1:00 - loss: 7.6077 - accuracy: 0.5038
 7840/25000 [========>.....................] - ETA: 1:00 - loss: 7.6099 - accuracy: 0.5037
 7872/25000 [========>.....................] - ETA: 1:00 - loss: 7.6160 - accuracy: 0.5033
 7904/25000 [========>.....................] - ETA: 1:00 - loss: 7.6142 - accuracy: 0.5034
 7936/25000 [========>.....................] - ETA: 1:00 - loss: 7.6125 - accuracy: 0.5035
 7968/25000 [========>.....................] - ETA: 1:00 - loss: 7.6108 - accuracy: 0.5036
 8000/25000 [========>.....................] - ETA: 59s - loss: 7.6091 - accuracy: 0.5038 
 8032/25000 [========>.....................] - ETA: 59s - loss: 7.6113 - accuracy: 0.5036
 8064/25000 [========>.....................] - ETA: 59s - loss: 7.6191 - accuracy: 0.5031
 8096/25000 [========>.....................] - ETA: 59s - loss: 7.6212 - accuracy: 0.5030
 8128/25000 [========>.....................] - ETA: 59s - loss: 7.6213 - accuracy: 0.5030
 8160/25000 [========>.....................] - ETA: 59s - loss: 7.6234 - accuracy: 0.5028
 8192/25000 [========>.....................] - ETA: 59s - loss: 7.6217 - accuracy: 0.5029
 8224/25000 [========>.....................] - ETA: 59s - loss: 7.6200 - accuracy: 0.5030
 8256/25000 [========>.....................] - ETA: 59s - loss: 7.6220 - accuracy: 0.5029
 8288/25000 [========>.....................] - ETA: 58s - loss: 7.6296 - accuracy: 0.5024
 8320/25000 [========>.....................] - ETA: 58s - loss: 7.6279 - accuracy: 0.5025
 8352/25000 [=========>....................] - ETA: 58s - loss: 7.6207 - accuracy: 0.5030
 8384/25000 [=========>....................] - ETA: 58s - loss: 7.6136 - accuracy: 0.5035
 8416/25000 [=========>....................] - ETA: 58s - loss: 7.6083 - accuracy: 0.5038
 8448/25000 [=========>....................] - ETA: 58s - loss: 7.5976 - accuracy: 0.5045
 8480/25000 [=========>....................] - ETA: 58s - loss: 7.5997 - accuracy: 0.5044
 8512/25000 [=========>....................] - ETA: 58s - loss: 7.5982 - accuracy: 0.5045
 8544/25000 [=========>....................] - ETA: 57s - loss: 7.6056 - accuracy: 0.5040
 8576/25000 [=========>....................] - ETA: 57s - loss: 7.6040 - accuracy: 0.5041
 8608/25000 [=========>....................] - ETA: 57s - loss: 7.6043 - accuracy: 0.5041
 8640/25000 [=========>....................] - ETA: 57s - loss: 7.6027 - accuracy: 0.5042
 8672/25000 [=========>....................] - ETA: 57s - loss: 7.6012 - accuracy: 0.5043
 8704/25000 [=========>....................] - ETA: 57s - loss: 7.6067 - accuracy: 0.5039
 8736/25000 [=========>....................] - ETA: 57s - loss: 7.6034 - accuracy: 0.5041
 8768/25000 [=========>....................] - ETA: 57s - loss: 7.5932 - accuracy: 0.5048
 8800/25000 [=========>....................] - ETA: 57s - loss: 7.5934 - accuracy: 0.5048
 8832/25000 [=========>....................] - ETA: 57s - loss: 7.5920 - accuracy: 0.5049
 8864/25000 [=========>....................] - ETA: 56s - loss: 7.6026 - accuracy: 0.5042
 8896/25000 [=========>....................] - ETA: 56s - loss: 7.5942 - accuracy: 0.5047
 8928/25000 [=========>....................] - ETA: 56s - loss: 7.5945 - accuracy: 0.5047
 8960/25000 [=========>....................] - ETA: 56s - loss: 7.5913 - accuracy: 0.5049
 8992/25000 [=========>....................] - ETA: 56s - loss: 7.5814 - accuracy: 0.5056
 9024/25000 [=========>....................] - ETA: 56s - loss: 7.5800 - accuracy: 0.5057
 9056/25000 [=========>....................] - ETA: 56s - loss: 7.5803 - accuracy: 0.5056
 9088/25000 [=========>....................] - ETA: 56s - loss: 7.5721 - accuracy: 0.5062
 9120/25000 [=========>....................] - ETA: 56s - loss: 7.5674 - accuracy: 0.5065
 9152/25000 [=========>....................] - ETA: 55s - loss: 7.5728 - accuracy: 0.5061
 9184/25000 [==========>...................] - ETA: 55s - loss: 7.5781 - accuracy: 0.5058
 9216/25000 [==========>...................] - ETA: 55s - loss: 7.5718 - accuracy: 0.5062
 9248/25000 [==========>...................] - ETA: 55s - loss: 7.5688 - accuracy: 0.5064
 9280/25000 [==========>...................] - ETA: 55s - loss: 7.5724 - accuracy: 0.5061
 9312/25000 [==========>...................] - ETA: 55s - loss: 7.5810 - accuracy: 0.5056
 9344/25000 [==========>...................] - ETA: 55s - loss: 7.5780 - accuracy: 0.5058
 9376/25000 [==========>...................] - ETA: 55s - loss: 7.5849 - accuracy: 0.5053
 9408/25000 [==========>...................] - ETA: 55s - loss: 7.5802 - accuracy: 0.5056
 9440/25000 [==========>...................] - ETA: 54s - loss: 7.5870 - accuracy: 0.5052
 9472/25000 [==========>...................] - ETA: 54s - loss: 7.5986 - accuracy: 0.5044
 9504/25000 [==========>...................] - ETA: 54s - loss: 7.5989 - accuracy: 0.5044
 9536/25000 [==========>...................] - ETA: 54s - loss: 7.5991 - accuracy: 0.5044
 9568/25000 [==========>...................] - ETA: 54s - loss: 7.6089 - accuracy: 0.5038
 9600/25000 [==========>...................] - ETA: 54s - loss: 7.6075 - accuracy: 0.5039
 9632/25000 [==========>...................] - ETA: 54s - loss: 7.6029 - accuracy: 0.5042
 9664/25000 [==========>...................] - ETA: 54s - loss: 7.5952 - accuracy: 0.5047
 9696/25000 [==========>...................] - ETA: 54s - loss: 7.5939 - accuracy: 0.5047
 9728/25000 [==========>...................] - ETA: 53s - loss: 7.6004 - accuracy: 0.5043
 9760/25000 [==========>...................] - ETA: 53s - loss: 7.6022 - accuracy: 0.5042
 9792/25000 [==========>...................] - ETA: 53s - loss: 7.6009 - accuracy: 0.5043
 9824/25000 [==========>...................] - ETA: 53s - loss: 7.6011 - accuracy: 0.5043
 9856/25000 [==========>...................] - ETA: 53s - loss: 7.6028 - accuracy: 0.5042
 9888/25000 [==========>...................] - ETA: 53s - loss: 7.6046 - accuracy: 0.5040
 9920/25000 [==========>...................] - ETA: 53s - loss: 7.5971 - accuracy: 0.5045
 9952/25000 [==========>...................] - ETA: 53s - loss: 7.5957 - accuracy: 0.5046
 9984/25000 [==========>...................] - ETA: 53s - loss: 7.5898 - accuracy: 0.5050
10016/25000 [===========>..................] - ETA: 52s - loss: 7.5885 - accuracy: 0.5051
10048/25000 [===========>..................] - ETA: 52s - loss: 7.5903 - accuracy: 0.5050
10080/25000 [===========>..................] - ETA: 52s - loss: 7.5906 - accuracy: 0.5050
10112/25000 [===========>..................] - ETA: 52s - loss: 7.5832 - accuracy: 0.5054
10144/25000 [===========>..................] - ETA: 52s - loss: 7.5850 - accuracy: 0.5053
10176/25000 [===========>..................] - ETA: 52s - loss: 7.5822 - accuracy: 0.5055
10208/25000 [===========>..................] - ETA: 52s - loss: 7.5795 - accuracy: 0.5057
10240/25000 [===========>..................] - ETA: 52s - loss: 7.5783 - accuracy: 0.5058
10272/25000 [===========>..................] - ETA: 51s - loss: 7.5800 - accuracy: 0.5056
10304/25000 [===========>..................] - ETA: 51s - loss: 7.5818 - accuracy: 0.5055
10336/25000 [===========>..................] - ETA: 51s - loss: 7.5732 - accuracy: 0.5061
10368/25000 [===========>..................] - ETA: 51s - loss: 7.5808 - accuracy: 0.5056
10400/25000 [===========>..................] - ETA: 51s - loss: 7.5811 - accuracy: 0.5056
10432/25000 [===========>..................] - ETA: 51s - loss: 7.5858 - accuracy: 0.5053
10464/25000 [===========>..................] - ETA: 51s - loss: 7.5860 - accuracy: 0.5053
10496/25000 [===========>..................] - ETA: 51s - loss: 7.5790 - accuracy: 0.5057
10528/25000 [===========>..................] - ETA: 51s - loss: 7.5792 - accuracy: 0.5057
10560/25000 [===========>..................] - ETA: 50s - loss: 7.5824 - accuracy: 0.5055
10592/25000 [===========>..................] - ETA: 50s - loss: 7.5769 - accuracy: 0.5059
10624/25000 [===========>..................] - ETA: 50s - loss: 7.5757 - accuracy: 0.5059
10656/25000 [===========>..................] - ETA: 50s - loss: 7.5745 - accuracy: 0.5060
10688/25000 [===========>..................] - ETA: 50s - loss: 7.5762 - accuracy: 0.5059
10720/25000 [===========>..................] - ETA: 50s - loss: 7.5765 - accuracy: 0.5059
10752/25000 [===========>..................] - ETA: 50s - loss: 7.5725 - accuracy: 0.5061
10784/25000 [===========>..................] - ETA: 50s - loss: 7.5714 - accuracy: 0.5062
10816/25000 [===========>..................] - ETA: 50s - loss: 7.5745 - accuracy: 0.5060
10848/25000 [============>.................] - ETA: 49s - loss: 7.5747 - accuracy: 0.5060
10880/25000 [============>.................] - ETA: 49s - loss: 7.5694 - accuracy: 0.5063
10912/25000 [============>.................] - ETA: 49s - loss: 7.5612 - accuracy: 0.5069
10944/25000 [============>.................] - ETA: 49s - loss: 7.5671 - accuracy: 0.5065
10976/25000 [============>.................] - ETA: 49s - loss: 7.5632 - accuracy: 0.5067
11008/25000 [============>.................] - ETA: 49s - loss: 7.5608 - accuracy: 0.5069
11040/25000 [============>.................] - ETA: 49s - loss: 7.5611 - accuracy: 0.5069
11072/25000 [============>.................] - ETA: 49s - loss: 7.5697 - accuracy: 0.5063
11104/25000 [============>.................] - ETA: 49s - loss: 7.5755 - accuracy: 0.5059
11136/25000 [============>.................] - ETA: 48s - loss: 7.5799 - accuracy: 0.5057
11168/25000 [============>.................] - ETA: 48s - loss: 7.5760 - accuracy: 0.5059
11200/25000 [============>.................] - ETA: 48s - loss: 7.5763 - accuracy: 0.5059
11232/25000 [============>.................] - ETA: 48s - loss: 7.5779 - accuracy: 0.5058
11264/25000 [============>.................] - ETA: 48s - loss: 7.5822 - accuracy: 0.5055
11296/25000 [============>.................] - ETA: 48s - loss: 7.5838 - accuracy: 0.5054
11328/25000 [============>.................] - ETA: 48s - loss: 7.5841 - accuracy: 0.5054
11360/25000 [============>.................] - ETA: 48s - loss: 7.5829 - accuracy: 0.5055
11392/25000 [============>.................] - ETA: 47s - loss: 7.5845 - accuracy: 0.5054
11424/25000 [============>.................] - ETA: 47s - loss: 7.5874 - accuracy: 0.5052
11456/25000 [============>.................] - ETA: 47s - loss: 7.5850 - accuracy: 0.5053
11488/25000 [============>.................] - ETA: 47s - loss: 7.5785 - accuracy: 0.5057
11520/25000 [============>.................] - ETA: 47s - loss: 7.5734 - accuracy: 0.5061
11552/25000 [============>.................] - ETA: 47s - loss: 7.5644 - accuracy: 0.5067
11584/25000 [============>.................] - ETA: 47s - loss: 7.5607 - accuracy: 0.5069
11616/25000 [============>.................] - ETA: 47s - loss: 7.5584 - accuracy: 0.5071
11648/25000 [============>.................] - ETA: 47s - loss: 7.5574 - accuracy: 0.5071
11680/25000 [=============>................] - ETA: 46s - loss: 7.5590 - accuracy: 0.5070
11712/25000 [=============>................] - ETA: 46s - loss: 7.5658 - accuracy: 0.5066
11744/25000 [=============>................] - ETA: 46s - loss: 7.5726 - accuracy: 0.5061
11776/25000 [=============>................] - ETA: 46s - loss: 7.5690 - accuracy: 0.5064
11808/25000 [=============>................] - ETA: 46s - loss: 7.5718 - accuracy: 0.5062
11840/25000 [=============>................] - ETA: 46s - loss: 7.5786 - accuracy: 0.5057
11872/25000 [=============>................] - ETA: 46s - loss: 7.5775 - accuracy: 0.5058
11904/25000 [=============>................] - ETA: 46s - loss: 7.5765 - accuracy: 0.5059
11936/25000 [=============>................] - ETA: 46s - loss: 7.5793 - accuracy: 0.5057
11968/25000 [=============>................] - ETA: 45s - loss: 7.5795 - accuracy: 0.5057
12000/25000 [=============>................] - ETA: 45s - loss: 7.5810 - accuracy: 0.5056
12032/25000 [=============>................] - ETA: 45s - loss: 7.5838 - accuracy: 0.5054
12064/25000 [=============>................] - ETA: 45s - loss: 7.5878 - accuracy: 0.5051
12096/25000 [=============>................] - ETA: 45s - loss: 7.5906 - accuracy: 0.5050
12128/25000 [=============>................] - ETA: 45s - loss: 7.5933 - accuracy: 0.5048
12160/25000 [=============>................] - ETA: 45s - loss: 7.5973 - accuracy: 0.5045
12192/25000 [=============>................] - ETA: 45s - loss: 7.5949 - accuracy: 0.5047
12224/25000 [=============>................] - ETA: 45s - loss: 7.5888 - accuracy: 0.5051
12256/25000 [=============>................] - ETA: 44s - loss: 7.5941 - accuracy: 0.5047
12288/25000 [=============>................] - ETA: 44s - loss: 7.5893 - accuracy: 0.5050
12320/25000 [=============>................] - ETA: 44s - loss: 7.5882 - accuracy: 0.5051
12352/25000 [=============>................] - ETA: 44s - loss: 7.5959 - accuracy: 0.5046
12384/25000 [=============>................] - ETA: 44s - loss: 7.5973 - accuracy: 0.5045
12416/25000 [=============>................] - ETA: 44s - loss: 7.5938 - accuracy: 0.5048
12448/25000 [=============>................] - ETA: 44s - loss: 7.5939 - accuracy: 0.5047
12480/25000 [=============>................] - ETA: 44s - loss: 7.5904 - accuracy: 0.5050
12512/25000 [==============>...............] - ETA: 44s - loss: 7.5980 - accuracy: 0.5045
12544/25000 [==============>...............] - ETA: 43s - loss: 7.5921 - accuracy: 0.5049
12576/25000 [==============>...............] - ETA: 43s - loss: 7.5922 - accuracy: 0.5049
12608/25000 [==============>...............] - ETA: 43s - loss: 7.5961 - accuracy: 0.5046
12640/25000 [==============>...............] - ETA: 43s - loss: 7.5878 - accuracy: 0.5051
12672/25000 [==============>...............] - ETA: 43s - loss: 7.5892 - accuracy: 0.5051
12704/25000 [==============>...............] - ETA: 43s - loss: 7.5918 - accuracy: 0.5049
12736/25000 [==============>...............] - ETA: 43s - loss: 7.5884 - accuracy: 0.5051
12768/25000 [==============>...............] - ETA: 43s - loss: 7.5874 - accuracy: 0.5052
12800/25000 [==============>...............] - ETA: 42s - loss: 7.5864 - accuracy: 0.5052
12832/25000 [==============>...............] - ETA: 42s - loss: 7.5842 - accuracy: 0.5054
12864/25000 [==============>...............] - ETA: 42s - loss: 7.5844 - accuracy: 0.5054
12896/25000 [==============>...............] - ETA: 42s - loss: 7.5870 - accuracy: 0.5052
12928/25000 [==============>...............] - ETA: 42s - loss: 7.5860 - accuracy: 0.5053
12960/25000 [==============>...............] - ETA: 42s - loss: 7.5803 - accuracy: 0.5056
12992/25000 [==============>...............] - ETA: 42s - loss: 7.5781 - accuracy: 0.5058
13024/25000 [==============>...............] - ETA: 42s - loss: 7.5724 - accuracy: 0.5061
13056/25000 [==============>...............] - ETA: 42s - loss: 7.5727 - accuracy: 0.5061
13088/25000 [==============>...............] - ETA: 41s - loss: 7.5717 - accuracy: 0.5062
13120/25000 [==============>...............] - ETA: 41s - loss: 7.5673 - accuracy: 0.5065
13152/25000 [==============>...............] - ETA: 41s - loss: 7.5699 - accuracy: 0.5063
13184/25000 [==============>...............] - ETA: 41s - loss: 7.5678 - accuracy: 0.5064
13216/25000 [==============>...............] - ETA: 41s - loss: 7.5657 - accuracy: 0.5066
13248/25000 [==============>...............] - ETA: 41s - loss: 7.5625 - accuracy: 0.5068
13280/25000 [==============>...............] - ETA: 41s - loss: 7.5685 - accuracy: 0.5064
13312/25000 [==============>...............] - ETA: 41s - loss: 7.5641 - accuracy: 0.5067
13344/25000 [===============>..............] - ETA: 41s - loss: 7.5632 - accuracy: 0.5067
13376/25000 [===============>..............] - ETA: 40s - loss: 7.5635 - accuracy: 0.5067
13408/25000 [===============>..............] - ETA: 40s - loss: 7.5603 - accuracy: 0.5069
13440/25000 [===============>..............] - ETA: 40s - loss: 7.5571 - accuracy: 0.5071
13472/25000 [===============>..............] - ETA: 40s - loss: 7.5574 - accuracy: 0.5071
13504/25000 [===============>..............] - ETA: 40s - loss: 7.5553 - accuracy: 0.5073
13536/25000 [===============>..............] - ETA: 40s - loss: 7.5522 - accuracy: 0.5075
13568/25000 [===============>..............] - ETA: 40s - loss: 7.5525 - accuracy: 0.5074
13600/25000 [===============>..............] - ETA: 40s - loss: 7.5482 - accuracy: 0.5077
13632/25000 [===============>..............] - ETA: 40s - loss: 7.5485 - accuracy: 0.5077
13664/25000 [===============>..............] - ETA: 39s - loss: 7.5533 - accuracy: 0.5074
13696/25000 [===============>..............] - ETA: 39s - loss: 7.5558 - accuracy: 0.5072
13728/25000 [===============>..............] - ETA: 39s - loss: 7.5538 - accuracy: 0.5074
13760/25000 [===============>..............] - ETA: 39s - loss: 7.5541 - accuracy: 0.5073
13792/25000 [===============>..............] - ETA: 39s - loss: 7.5521 - accuracy: 0.5075
13824/25000 [===============>..............] - ETA: 39s - loss: 7.5502 - accuracy: 0.5076
13856/25000 [===============>..............] - ETA: 39s - loss: 7.5471 - accuracy: 0.5078
13888/25000 [===============>..............] - ETA: 39s - loss: 7.5485 - accuracy: 0.5077
13920/25000 [===============>..............] - ETA: 39s - loss: 7.5543 - accuracy: 0.5073
13952/25000 [===============>..............] - ETA: 38s - loss: 7.5567 - accuracy: 0.5072
13984/25000 [===============>..............] - ETA: 38s - loss: 7.5614 - accuracy: 0.5069
14016/25000 [===============>..............] - ETA: 38s - loss: 7.5583 - accuracy: 0.5071
14048/25000 [===============>..............] - ETA: 38s - loss: 7.5618 - accuracy: 0.5068
14080/25000 [===============>..............] - ETA: 38s - loss: 7.5555 - accuracy: 0.5072
14112/25000 [===============>..............] - ETA: 38s - loss: 7.5558 - accuracy: 0.5072
14144/25000 [===============>..............] - ETA: 38s - loss: 7.5560 - accuracy: 0.5072
14176/25000 [================>.............] - ETA: 38s - loss: 7.5563 - accuracy: 0.5072
14208/25000 [================>.............] - ETA: 37s - loss: 7.5587 - accuracy: 0.5070
14240/25000 [================>.............] - ETA: 37s - loss: 7.5589 - accuracy: 0.5070
14272/25000 [================>.............] - ETA: 37s - loss: 7.5560 - accuracy: 0.5072
14304/25000 [================>.............] - ETA: 37s - loss: 7.5562 - accuracy: 0.5072
14336/25000 [================>.............] - ETA: 37s - loss: 7.5597 - accuracy: 0.5070
14368/25000 [================>.............] - ETA: 37s - loss: 7.5599 - accuracy: 0.5070
14400/25000 [================>.............] - ETA: 37s - loss: 7.5601 - accuracy: 0.5069
14432/25000 [================>.............] - ETA: 37s - loss: 7.5625 - accuracy: 0.5068
14464/25000 [================>.............] - ETA: 37s - loss: 7.5574 - accuracy: 0.5071
14496/25000 [================>.............] - ETA: 36s - loss: 7.5577 - accuracy: 0.5071
14528/25000 [================>.............] - ETA: 36s - loss: 7.5579 - accuracy: 0.5071
14560/25000 [================>.............] - ETA: 36s - loss: 7.5603 - accuracy: 0.5069
14592/25000 [================>.............] - ETA: 36s - loss: 7.5594 - accuracy: 0.5070
14624/25000 [================>.............] - ETA: 36s - loss: 7.5618 - accuracy: 0.5068
14656/25000 [================>.............] - ETA: 36s - loss: 7.5610 - accuracy: 0.5069
14688/25000 [================>.............] - ETA: 36s - loss: 7.5643 - accuracy: 0.5067
14720/25000 [================>.............] - ETA: 36s - loss: 7.5677 - accuracy: 0.5065
14752/25000 [================>.............] - ETA: 36s - loss: 7.5648 - accuracy: 0.5066
14784/25000 [================>.............] - ETA: 35s - loss: 7.5660 - accuracy: 0.5066
14816/25000 [================>.............] - ETA: 35s - loss: 7.5631 - accuracy: 0.5067
14848/25000 [================>.............] - ETA: 35s - loss: 7.5644 - accuracy: 0.5067
14880/25000 [================>.............] - ETA: 35s - loss: 7.5667 - accuracy: 0.5065
14912/25000 [================>.............] - ETA: 35s - loss: 7.5710 - accuracy: 0.5062
14944/25000 [================>.............] - ETA: 35s - loss: 7.5691 - accuracy: 0.5064
14976/25000 [================>.............] - ETA: 35s - loss: 7.5714 - accuracy: 0.5062
15008/25000 [=================>............] - ETA: 35s - loss: 7.5798 - accuracy: 0.5057
15040/25000 [=================>............] - ETA: 35s - loss: 7.5769 - accuracy: 0.5059
15072/25000 [=================>............] - ETA: 34s - loss: 7.5801 - accuracy: 0.5056
15104/25000 [=================>............] - ETA: 34s - loss: 7.5885 - accuracy: 0.5051
15136/25000 [=================>............] - ETA: 34s - loss: 7.5846 - accuracy: 0.5054
15168/25000 [=================>............] - ETA: 34s - loss: 7.5908 - accuracy: 0.5049
15200/25000 [=================>............] - ETA: 34s - loss: 7.5920 - accuracy: 0.5049
15232/25000 [=================>............] - ETA: 34s - loss: 7.5921 - accuracy: 0.5049
15264/25000 [=================>............] - ETA: 34s - loss: 7.5903 - accuracy: 0.5050
15296/25000 [=================>............] - ETA: 34s - loss: 7.5924 - accuracy: 0.5048
15328/25000 [=================>............] - ETA: 34s - loss: 7.5896 - accuracy: 0.5050
15360/25000 [=================>............] - ETA: 33s - loss: 7.5858 - accuracy: 0.5053
15392/25000 [=================>............] - ETA: 33s - loss: 7.5849 - accuracy: 0.5053
15424/25000 [=================>............] - ETA: 33s - loss: 7.5881 - accuracy: 0.5051
15456/25000 [=================>............] - ETA: 33s - loss: 7.5882 - accuracy: 0.5051
15488/25000 [=================>............] - ETA: 33s - loss: 7.5874 - accuracy: 0.5052
15520/25000 [=================>............] - ETA: 33s - loss: 7.5866 - accuracy: 0.5052
15552/25000 [=================>............] - ETA: 33s - loss: 7.5858 - accuracy: 0.5053
15584/25000 [=================>............] - ETA: 33s - loss: 7.5899 - accuracy: 0.5050
15616/25000 [=================>............] - ETA: 33s - loss: 7.5881 - accuracy: 0.5051
15648/25000 [=================>............] - ETA: 32s - loss: 7.5882 - accuracy: 0.5051
15680/25000 [=================>............] - ETA: 32s - loss: 7.5884 - accuracy: 0.5051
15712/25000 [=================>............] - ETA: 32s - loss: 7.5895 - accuracy: 0.5050
15744/25000 [=================>............] - ETA: 32s - loss: 7.5907 - accuracy: 0.5050
15776/25000 [=================>............] - ETA: 32s - loss: 7.5889 - accuracy: 0.5051
15808/25000 [=================>............] - ETA: 32s - loss: 7.5832 - accuracy: 0.5054
15840/25000 [==================>...........] - ETA: 32s - loss: 7.5853 - accuracy: 0.5053
15872/25000 [==================>...........] - ETA: 32s - loss: 7.5874 - accuracy: 0.5052
15904/25000 [==================>...........] - ETA: 32s - loss: 7.5876 - accuracy: 0.5052
15936/25000 [==================>...........] - ETA: 32s - loss: 7.5858 - accuracy: 0.5053
15968/25000 [==================>...........] - ETA: 31s - loss: 7.5802 - accuracy: 0.5056
16000/25000 [==================>...........] - ETA: 31s - loss: 7.5852 - accuracy: 0.5053
16032/25000 [==================>...........] - ETA: 31s - loss: 7.5872 - accuracy: 0.5052
16064/25000 [==================>...........] - ETA: 31s - loss: 7.5903 - accuracy: 0.5050
16096/25000 [==================>...........] - ETA: 31s - loss: 7.5904 - accuracy: 0.5050
16128/25000 [==================>...........] - ETA: 31s - loss: 7.5915 - accuracy: 0.5049
16160/25000 [==================>...........] - ETA: 31s - loss: 7.5926 - accuracy: 0.5048
16192/25000 [==================>...........] - ETA: 31s - loss: 7.5956 - accuracy: 0.5046
16224/25000 [==================>...........] - ETA: 30s - loss: 7.5976 - accuracy: 0.5045
16256/25000 [==================>...........] - ETA: 30s - loss: 7.5949 - accuracy: 0.5047
16288/25000 [==================>...........] - ETA: 30s - loss: 7.5970 - accuracy: 0.5045
16320/25000 [==================>...........] - ETA: 30s - loss: 7.5990 - accuracy: 0.5044
16352/25000 [==================>...........] - ETA: 30s - loss: 7.5982 - accuracy: 0.5045
16384/25000 [==================>...........] - ETA: 30s - loss: 7.5955 - accuracy: 0.5046
16416/25000 [==================>...........] - ETA: 30s - loss: 7.5938 - accuracy: 0.5048
16448/25000 [==================>...........] - ETA: 30s - loss: 7.5958 - accuracy: 0.5046
16480/25000 [==================>...........] - ETA: 30s - loss: 7.5996 - accuracy: 0.5044
16512/25000 [==================>...........] - ETA: 29s - loss: 7.6016 - accuracy: 0.5042
16544/25000 [==================>...........] - ETA: 29s - loss: 7.6017 - accuracy: 0.5042
16576/25000 [==================>...........] - ETA: 29s - loss: 7.6028 - accuracy: 0.5042
16608/25000 [==================>...........] - ETA: 29s - loss: 7.6066 - accuracy: 0.5039
16640/25000 [==================>...........] - ETA: 29s - loss: 7.6040 - accuracy: 0.5041
16672/25000 [===================>..........] - ETA: 29s - loss: 7.6068 - accuracy: 0.5039
16704/25000 [===================>..........] - ETA: 29s - loss: 7.6051 - accuracy: 0.5040
16736/25000 [===================>..........] - ETA: 29s - loss: 7.6043 - accuracy: 0.5041
16768/25000 [===================>..........] - ETA: 29s - loss: 7.6054 - accuracy: 0.5040
16800/25000 [===================>..........] - ETA: 28s - loss: 7.6000 - accuracy: 0.5043
16832/25000 [===================>..........] - ETA: 28s - loss: 7.6001 - accuracy: 0.5043
16864/25000 [===================>..........] - ETA: 28s - loss: 7.6021 - accuracy: 0.5042
16896/25000 [===================>..........] - ETA: 28s - loss: 7.6031 - accuracy: 0.5041
16928/25000 [===================>..........] - ETA: 28s - loss: 7.6096 - accuracy: 0.5037
16960/25000 [===================>..........] - ETA: 28s - loss: 7.6115 - accuracy: 0.5036
16992/25000 [===================>..........] - ETA: 28s - loss: 7.6116 - accuracy: 0.5036
17024/25000 [===================>..........] - ETA: 28s - loss: 7.6108 - accuracy: 0.5036
17056/25000 [===================>..........] - ETA: 28s - loss: 7.6091 - accuracy: 0.5038
17088/25000 [===================>..........] - ETA: 27s - loss: 7.6065 - accuracy: 0.5039
17120/25000 [===================>..........] - ETA: 27s - loss: 7.6057 - accuracy: 0.5040
17152/25000 [===================>..........] - ETA: 27s - loss: 7.6067 - accuracy: 0.5039
17184/25000 [===================>..........] - ETA: 27s - loss: 7.6086 - accuracy: 0.5038
17216/25000 [===================>..........] - ETA: 27s - loss: 7.6123 - accuracy: 0.5035
17248/25000 [===================>..........] - ETA: 27s - loss: 7.6097 - accuracy: 0.5037
17280/25000 [===================>..........] - ETA: 27s - loss: 7.6081 - accuracy: 0.5038
17312/25000 [===================>..........] - ETA: 27s - loss: 7.6117 - accuracy: 0.5036
17344/25000 [===================>..........] - ETA: 27s - loss: 7.6162 - accuracy: 0.5033
17376/25000 [===================>..........] - ETA: 26s - loss: 7.6172 - accuracy: 0.5032
17408/25000 [===================>..........] - ETA: 26s - loss: 7.6182 - accuracy: 0.5032
17440/25000 [===================>..........] - ETA: 26s - loss: 7.6209 - accuracy: 0.5030
17472/25000 [===================>..........] - ETA: 26s - loss: 7.6210 - accuracy: 0.5030
17504/25000 [====================>.........] - ETA: 26s - loss: 7.6211 - accuracy: 0.5030
17536/25000 [====================>.........] - ETA: 26s - loss: 7.6255 - accuracy: 0.5027
17568/25000 [====================>.........] - ETA: 26s - loss: 7.6291 - accuracy: 0.5024
17600/25000 [====================>.........] - ETA: 26s - loss: 7.6257 - accuracy: 0.5027
17632/25000 [====================>.........] - ETA: 26s - loss: 7.6240 - accuracy: 0.5028
17664/25000 [====================>.........] - ETA: 25s - loss: 7.6250 - accuracy: 0.5027
17696/25000 [====================>.........] - ETA: 25s - loss: 7.6259 - accuracy: 0.5027
17728/25000 [====================>.........] - ETA: 25s - loss: 7.6242 - accuracy: 0.5028
17760/25000 [====================>.........] - ETA: 25s - loss: 7.6200 - accuracy: 0.5030
17792/25000 [====================>.........] - ETA: 25s - loss: 7.6201 - accuracy: 0.5030
17824/25000 [====================>.........] - ETA: 25s - loss: 7.6176 - accuracy: 0.5032
17856/25000 [====================>.........] - ETA: 25s - loss: 7.6168 - accuracy: 0.5032
17888/25000 [====================>.........] - ETA: 25s - loss: 7.6143 - accuracy: 0.5034
17920/25000 [====================>.........] - ETA: 25s - loss: 7.6127 - accuracy: 0.5035
17952/25000 [====================>.........] - ETA: 24s - loss: 7.6128 - accuracy: 0.5035
17984/25000 [====================>.........] - ETA: 24s - loss: 7.6146 - accuracy: 0.5034
18016/25000 [====================>.........] - ETA: 24s - loss: 7.6104 - accuracy: 0.5037
18048/25000 [====================>.........] - ETA: 24s - loss: 7.6114 - accuracy: 0.5036
18080/25000 [====================>.........] - ETA: 24s - loss: 7.6081 - accuracy: 0.5038
18112/25000 [====================>.........] - ETA: 24s - loss: 7.6099 - accuracy: 0.5037
18144/25000 [====================>.........] - ETA: 24s - loss: 7.6117 - accuracy: 0.5036
18176/25000 [====================>.........] - ETA: 24s - loss: 7.6118 - accuracy: 0.5036
18208/25000 [====================>.........] - ETA: 24s - loss: 7.6127 - accuracy: 0.5035
18240/25000 [====================>.........] - ETA: 23s - loss: 7.6128 - accuracy: 0.5035
18272/25000 [====================>.........] - ETA: 23s - loss: 7.6121 - accuracy: 0.5036
18304/25000 [====================>.........] - ETA: 23s - loss: 7.6105 - accuracy: 0.5037
18336/25000 [=====================>........] - ETA: 23s - loss: 7.6131 - accuracy: 0.5035
18368/25000 [=====================>........] - ETA: 23s - loss: 7.6140 - accuracy: 0.5034
18400/25000 [=====================>........] - ETA: 23s - loss: 7.6175 - accuracy: 0.5032
18432/25000 [=====================>........] - ETA: 23s - loss: 7.6192 - accuracy: 0.5031
18464/25000 [=====================>........] - ETA: 23s - loss: 7.6234 - accuracy: 0.5028
18496/25000 [=====================>........] - ETA: 23s - loss: 7.6194 - accuracy: 0.5031
18528/25000 [=====================>........] - ETA: 22s - loss: 7.6178 - accuracy: 0.5032
18560/25000 [=====================>........] - ETA: 22s - loss: 7.6212 - accuracy: 0.5030
18592/25000 [=====================>........] - ETA: 22s - loss: 7.6237 - accuracy: 0.5028
18624/25000 [=====================>........] - ETA: 22s - loss: 7.6238 - accuracy: 0.5028
18656/25000 [=====================>........] - ETA: 22s - loss: 7.6255 - accuracy: 0.5027
18688/25000 [=====================>........] - ETA: 22s - loss: 7.6281 - accuracy: 0.5025
18720/25000 [=====================>........] - ETA: 22s - loss: 7.6306 - accuracy: 0.5024
18752/25000 [=====================>........] - ETA: 22s - loss: 7.6323 - accuracy: 0.5022
18784/25000 [=====================>........] - ETA: 22s - loss: 7.6291 - accuracy: 0.5024
18816/25000 [=====================>........] - ETA: 21s - loss: 7.6275 - accuracy: 0.5026
18848/25000 [=====================>........] - ETA: 21s - loss: 7.6308 - accuracy: 0.5023
18880/25000 [=====================>........] - ETA: 21s - loss: 7.6317 - accuracy: 0.5023
18912/25000 [=====================>........] - ETA: 21s - loss: 7.6350 - accuracy: 0.5021
18944/25000 [=====================>........] - ETA: 21s - loss: 7.6334 - accuracy: 0.5022
18976/25000 [=====================>........] - ETA: 21s - loss: 7.6319 - accuracy: 0.5023
19008/25000 [=====================>........] - ETA: 21s - loss: 7.6335 - accuracy: 0.5022
19040/25000 [=====================>........] - ETA: 21s - loss: 7.6336 - accuracy: 0.5022
19072/25000 [=====================>........] - ETA: 21s - loss: 7.6337 - accuracy: 0.5021
19104/25000 [=====================>........] - ETA: 20s - loss: 7.6353 - accuracy: 0.5020
19136/25000 [=====================>........] - ETA: 20s - loss: 7.6362 - accuracy: 0.5020
19168/25000 [======================>.......] - ETA: 20s - loss: 7.6346 - accuracy: 0.5021
19200/25000 [======================>.......] - ETA: 20s - loss: 7.6307 - accuracy: 0.5023
19232/25000 [======================>.......] - ETA: 20s - loss: 7.6299 - accuracy: 0.5024
19264/25000 [======================>.......] - ETA: 20s - loss: 7.6292 - accuracy: 0.5024
19296/25000 [======================>.......] - ETA: 20s - loss: 7.6309 - accuracy: 0.5023
19328/25000 [======================>.......] - ETA: 20s - loss: 7.6341 - accuracy: 0.5021
19360/25000 [======================>.......] - ETA: 19s - loss: 7.6373 - accuracy: 0.5019
19392/25000 [======================>.......] - ETA: 19s - loss: 7.6366 - accuracy: 0.5020
19424/25000 [======================>.......] - ETA: 19s - loss: 7.6382 - accuracy: 0.5019
19456/25000 [======================>.......] - ETA: 19s - loss: 7.6414 - accuracy: 0.5016
19488/25000 [======================>.......] - ETA: 19s - loss: 7.6399 - accuracy: 0.5017
19520/25000 [======================>.......] - ETA: 19s - loss: 7.6423 - accuracy: 0.5016
19552/25000 [======================>.......] - ETA: 19s - loss: 7.6423 - accuracy: 0.5016
19584/25000 [======================>.......] - ETA: 19s - loss: 7.6423 - accuracy: 0.5016
19616/25000 [======================>.......] - ETA: 19s - loss: 7.6447 - accuracy: 0.5014
19648/25000 [======================>.......] - ETA: 18s - loss: 7.6463 - accuracy: 0.5013
19680/25000 [======================>.......] - ETA: 18s - loss: 7.6479 - accuracy: 0.5012
19712/25000 [======================>.......] - ETA: 18s - loss: 7.6448 - accuracy: 0.5014
19744/25000 [======================>.......] - ETA: 18s - loss: 7.6410 - accuracy: 0.5017
19776/25000 [======================>.......] - ETA: 18s - loss: 7.6395 - accuracy: 0.5018
19808/25000 [======================>.......] - ETA: 18s - loss: 7.6411 - accuracy: 0.5017
19840/25000 [======================>.......] - ETA: 18s - loss: 7.6434 - accuracy: 0.5015
19872/25000 [======================>.......] - ETA: 18s - loss: 7.6466 - accuracy: 0.5013
19904/25000 [======================>.......] - ETA: 18s - loss: 7.6443 - accuracy: 0.5015
19936/25000 [======================>.......] - ETA: 17s - loss: 7.6459 - accuracy: 0.5014
19968/25000 [======================>.......] - ETA: 17s - loss: 7.6428 - accuracy: 0.5016
20000/25000 [=======================>......] - ETA: 17s - loss: 7.6436 - accuracy: 0.5015
20032/25000 [=======================>......] - ETA: 17s - loss: 7.6467 - accuracy: 0.5013
20064/25000 [=======================>......] - ETA: 17s - loss: 7.6475 - accuracy: 0.5012
20096/25000 [=======================>......] - ETA: 17s - loss: 7.6498 - accuracy: 0.5011
20128/25000 [=======================>......] - ETA: 17s - loss: 7.6506 - accuracy: 0.5010
20160/25000 [=======================>......] - ETA: 17s - loss: 7.6499 - accuracy: 0.5011
20192/25000 [=======================>......] - ETA: 17s - loss: 7.6507 - accuracy: 0.5010
20224/25000 [=======================>......] - ETA: 16s - loss: 7.6507 - accuracy: 0.5010
20256/25000 [=======================>......] - ETA: 16s - loss: 7.6522 - accuracy: 0.5009
20288/25000 [=======================>......] - ETA: 16s - loss: 7.6507 - accuracy: 0.5010
20320/25000 [=======================>......] - ETA: 16s - loss: 7.6530 - accuracy: 0.5009
20352/25000 [=======================>......] - ETA: 16s - loss: 7.6538 - accuracy: 0.5008
20384/25000 [=======================>......] - ETA: 16s - loss: 7.6531 - accuracy: 0.5009
20416/25000 [=======================>......] - ETA: 16s - loss: 7.6508 - accuracy: 0.5010
20448/25000 [=======================>......] - ETA: 16s - loss: 7.6516 - accuracy: 0.5010
20480/25000 [=======================>......] - ETA: 16s - loss: 7.6501 - accuracy: 0.5011
20512/25000 [=======================>......] - ETA: 15s - loss: 7.6509 - accuracy: 0.5010
20544/25000 [=======================>......] - ETA: 15s - loss: 7.6509 - accuracy: 0.5010
20576/25000 [=======================>......] - ETA: 15s - loss: 7.6510 - accuracy: 0.5010
20608/25000 [=======================>......] - ETA: 15s - loss: 7.6517 - accuracy: 0.5010
20640/25000 [=======================>......] - ETA: 15s - loss: 7.6547 - accuracy: 0.5008
20672/25000 [=======================>......] - ETA: 15s - loss: 7.6555 - accuracy: 0.5007
20704/25000 [=======================>......] - ETA: 15s - loss: 7.6555 - accuracy: 0.5007
20736/25000 [=======================>......] - ETA: 15s - loss: 7.6600 - accuracy: 0.5004
20768/25000 [=======================>......] - ETA: 14s - loss: 7.6570 - accuracy: 0.5006
20800/25000 [=======================>......] - ETA: 14s - loss: 7.6541 - accuracy: 0.5008
20832/25000 [=======================>......] - ETA: 14s - loss: 7.6519 - accuracy: 0.5010
20864/25000 [========================>.....] - ETA: 14s - loss: 7.6534 - accuracy: 0.5009
20896/25000 [========================>.....] - ETA: 14s - loss: 7.6527 - accuracy: 0.5009
20928/25000 [========================>.....] - ETA: 14s - loss: 7.6527 - accuracy: 0.5009
20960/25000 [========================>.....] - ETA: 14s - loss: 7.6505 - accuracy: 0.5010
20992/25000 [========================>.....] - ETA: 14s - loss: 7.6513 - accuracy: 0.5010
21024/25000 [========================>.....] - ETA: 14s - loss: 7.6535 - accuracy: 0.5009
21056/25000 [========================>.....] - ETA: 13s - loss: 7.6513 - accuracy: 0.5010
21088/25000 [========================>.....] - ETA: 13s - loss: 7.6499 - accuracy: 0.5011
21120/25000 [========================>.....] - ETA: 13s - loss: 7.6485 - accuracy: 0.5012
21152/25000 [========================>.....] - ETA: 13s - loss: 7.6507 - accuracy: 0.5010
21184/25000 [========================>.....] - ETA: 13s - loss: 7.6521 - accuracy: 0.5009
21216/25000 [========================>.....] - ETA: 13s - loss: 7.6493 - accuracy: 0.5011
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6464 - accuracy: 0.5013
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6479 - accuracy: 0.5012
21312/25000 [========================>.....] - ETA: 13s - loss: 7.6450 - accuracy: 0.5014
21344/25000 [========================>.....] - ETA: 12s - loss: 7.6479 - accuracy: 0.5012
21376/25000 [========================>.....] - ETA: 12s - loss: 7.6494 - accuracy: 0.5011
21408/25000 [========================>.....] - ETA: 12s - loss: 7.6501 - accuracy: 0.5011
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6509 - accuracy: 0.5010
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6495 - accuracy: 0.5011
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6545 - accuracy: 0.5008
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6552 - accuracy: 0.5007
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6531 - accuracy: 0.5009
21600/25000 [========================>.....] - ETA: 12s - loss: 7.6524 - accuracy: 0.5009
21632/25000 [========================>.....] - ETA: 11s - loss: 7.6524 - accuracy: 0.5009
21664/25000 [========================>.....] - ETA: 11s - loss: 7.6510 - accuracy: 0.5010
21696/25000 [=========================>....] - ETA: 11s - loss: 7.6490 - accuracy: 0.5012
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6490 - accuracy: 0.5012
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6490 - accuracy: 0.5011
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6476 - accuracy: 0.5012
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6498 - accuracy: 0.5011
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6463 - accuracy: 0.5013
21888/25000 [=========================>....] - ETA: 11s - loss: 7.6477 - accuracy: 0.5012
21920/25000 [=========================>....] - ETA: 10s - loss: 7.6456 - accuracy: 0.5014
21952/25000 [=========================>....] - ETA: 10s - loss: 7.6478 - accuracy: 0.5012
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6457 - accuracy: 0.5014
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6485 - accuracy: 0.5012
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6520 - accuracy: 0.5010
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6513 - accuracy: 0.5010
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6500 - accuracy: 0.5011
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6507 - accuracy: 0.5010
22176/25000 [=========================>....] - ETA: 10s - loss: 7.6549 - accuracy: 0.5008
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6542 - accuracy: 0.5008 
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6549 - accuracy: 0.5008
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6570 - accuracy: 0.5006
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6591 - accuracy: 0.5005
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6598 - accuracy: 0.5004
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6577 - accuracy: 0.5006
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6584 - accuracy: 0.5005
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6577 - accuracy: 0.5006
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6625 - accuracy: 0.5003
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6639 - accuracy: 0.5002
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6625 - accuracy: 0.5003
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6625 - accuracy: 0.5003
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6625 - accuracy: 0.5003
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6612 - accuracy: 0.5004
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6605 - accuracy: 0.5004
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6619 - accuracy: 0.5003
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6619 - accuracy: 0.5003
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6632 - accuracy: 0.5002
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6633 - accuracy: 0.5002
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6646 - accuracy: 0.5001
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6646 - accuracy: 0.5001
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6659 - accuracy: 0.5000
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6660 - accuracy: 0.5000
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6673 - accuracy: 0.5000
23008/25000 [==========================>...] - ETA: 7s - loss: 7.6660 - accuracy: 0.5000
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6686 - accuracy: 0.4999
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6706 - accuracy: 0.4997
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6699 - accuracy: 0.4998
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6699 - accuracy: 0.4998
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6713 - accuracy: 0.4997
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6712 - accuracy: 0.4997
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6699 - accuracy: 0.4998
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6693 - accuracy: 0.4998
23296/25000 [==========================>...] - ETA: 6s - loss: 7.6719 - accuracy: 0.4997
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6745 - accuracy: 0.4995
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6738 - accuracy: 0.4995
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6738 - accuracy: 0.4995
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6712 - accuracy: 0.4997
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6705 - accuracy: 0.4997
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6712 - accuracy: 0.4997
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6699 - accuracy: 0.4998
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
23584/25000 [===========================>..] - ETA: 5s - loss: 7.6699 - accuracy: 0.4998
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6647 - accuracy: 0.5001
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6647 - accuracy: 0.5001
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23872/25000 [===========================>..] - ETA: 4s - loss: 7.6634 - accuracy: 0.5002
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6647 - accuracy: 0.5001
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6685 - accuracy: 0.4999
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6647 - accuracy: 0.5001
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6660 - accuracy: 0.5000
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6685 - accuracy: 0.4999
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
24192/25000 [============================>.] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24224/25000 [============================>.] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
24256/25000 [============================>.] - ETA: 2s - loss: 7.6691 - accuracy: 0.4998
24288/25000 [============================>.] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24320/25000 [============================>.] - ETA: 2s - loss: 7.6691 - accuracy: 0.4998
24352/25000 [============================>.] - ETA: 2s - loss: 7.6672 - accuracy: 0.5000
24384/25000 [============================>.] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24416/25000 [============================>.] - ETA: 2s - loss: 7.6716 - accuracy: 0.4997
24448/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
24480/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24512/25000 [============================>.] - ETA: 1s - loss: 7.6722 - accuracy: 0.4996
24544/25000 [============================>.] - ETA: 1s - loss: 7.6722 - accuracy: 0.4996
24576/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24608/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24640/25000 [============================>.] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
24672/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24704/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24736/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24768/25000 [============================>.] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
24800/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24832/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24864/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24896/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24928/25000 [============================>.] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
24960/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24992/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
25000/25000 [==============================] - 107s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fedfd747b00> 

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
 [-0.02514442 -0.01429228 -0.02797576  0.18052392 -0.01458498  0.09338591]
 [ 0.03192504  0.15017632 -0.05435724 -0.01185001  0.13401276 -0.04857888]
 [ 0.08399395  0.10706617  0.19094761  0.02903662  0.12932141  0.22160976]
 [-0.08482537  0.04142921  0.25001356  0.01511247 -0.121999   -0.18695703]
 [-0.10585364  0.07465064 -0.08767042  0.04525509  0.4081468   0.25540432]
 [ 0.02599592 -0.12724203 -0.40119812 -0.27786779 -0.00754303  0.31531394]
 [ 0.40477461  0.0712465   0.04782471  0.68994415 -0.40794784 -0.4345324 ]
 [ 0.15124083 -0.35569784 -0.11440947 -0.24925047  0.10240182  0.04642869]
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
{'loss': 0.4518723338842392, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-11 10:15:52.158233: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.46508028730750084, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-11 10:15:53.432423: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:57<01:26, 28.97s/it] 40%|████      | 2/5 [00:57<01:26, 28.97s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.36306870675506275, 'embedding_size_factor': 0.9975773791649792, 'layers.choice': 3, 'learning_rate': 0.0001683250986010629, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.2351252642816861e-11} and reward: 0.361
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd7<\x84\x87m\xb2RX\x15\x00\x00\x00embedding_size_factorq\x03G?\xef\xec'eW\xc6yX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?&\x10\r\x96Q\x1b*X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xab)#\x18\xfb\xc9uu." and reward: 0.361
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd7<\x84\x87m\xb2RX\x15\x00\x00\x00embedding_size_factorq\x03G?\xef\xec'eW\xc6yX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?&\x10\r\x96Q\x1b*X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xab)#\x18\xfb\xc9uu." and reward: 0.361
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 118.18373370170593
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the 0.03s of remaining time.
Ensemble size: 27
Ensemble weights: 
[0.74074074 0.25925926]
	0.3912	 = Validation accuracy score
	1.01s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 121.02s ...
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

