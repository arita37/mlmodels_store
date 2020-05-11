
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

    8192/17464789 [..............................] - ETA: 11s
 2564096/17464789 [===>..........................] - ETA: 0s 
 9428992/17464789 [===============>..............] - ETA: 0s
16424960/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 15:13:56.053040: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 15:13:56.067067: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-11 15:13:56.067301: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c78e656190 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 15:13:56.067320: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 5:38 - loss: 8.1458 - accuracy: 0.4688
   64/25000 [..............................] - ETA: 3:20 - loss: 8.1458 - accuracy: 0.4688
   96/25000 [..............................] - ETA: 2:34 - loss: 7.3472 - accuracy: 0.5208
  128/25000 [..............................] - ETA: 2:10 - loss: 7.1875 - accuracy: 0.5312
  160/25000 [..............................] - ETA: 1:55 - loss: 7.3791 - accuracy: 0.5188
  192/25000 [..............................] - ETA: 1:46 - loss: 7.5069 - accuracy: 0.5104
  224/25000 [..............................] - ETA: 1:39 - loss: 7.5982 - accuracy: 0.5045
  256/25000 [..............................] - ETA: 1:34 - loss: 7.7265 - accuracy: 0.4961
  288/25000 [..............................] - ETA: 1:30 - loss: 7.7199 - accuracy: 0.4965
  320/25000 [..............................] - ETA: 1:27 - loss: 7.5708 - accuracy: 0.5063
  352/25000 [..............................] - ETA: 1:24 - loss: 7.6231 - accuracy: 0.5028
  384/25000 [..............................] - ETA: 1:22 - loss: 7.7465 - accuracy: 0.4948
  416/25000 [..............................] - ETA: 1:20 - loss: 7.7772 - accuracy: 0.4928
  448/25000 [..............................] - ETA: 1:19 - loss: 7.7693 - accuracy: 0.4933
  480/25000 [..............................] - ETA: 1:18 - loss: 7.6986 - accuracy: 0.4979
  512/25000 [..............................] - ETA: 1:17 - loss: 7.6367 - accuracy: 0.5020
  544/25000 [..............................] - ETA: 1:16 - loss: 7.6102 - accuracy: 0.5037
  576/25000 [..............................] - ETA: 1:15 - loss: 7.5335 - accuracy: 0.5087
  608/25000 [..............................] - ETA: 1:15 - loss: 7.5657 - accuracy: 0.5066
  640/25000 [..............................] - ETA: 1:14 - loss: 7.6187 - accuracy: 0.5031
  672/25000 [..............................] - ETA: 1:13 - loss: 7.5754 - accuracy: 0.5060
  704/25000 [..............................] - ETA: 1:13 - loss: 7.6231 - accuracy: 0.5028
  736/25000 [..............................] - ETA: 1:12 - loss: 7.6666 - accuracy: 0.5000
  768/25000 [..............................] - ETA: 1:11 - loss: 7.7065 - accuracy: 0.4974
  800/25000 [..............................] - ETA: 1:10 - loss: 7.7816 - accuracy: 0.4925
  832/25000 [..............................] - ETA: 1:10 - loss: 7.8325 - accuracy: 0.4892
  864/25000 [>.............................] - ETA: 1:09 - loss: 7.8618 - accuracy: 0.4873
  896/25000 [>.............................] - ETA: 1:09 - loss: 7.8377 - accuracy: 0.4888
  928/25000 [>.............................] - ETA: 1:08 - loss: 7.8484 - accuracy: 0.4881
  960/25000 [>.............................] - ETA: 1:08 - loss: 7.8104 - accuracy: 0.4906
  992/25000 [>.............................] - ETA: 1:07 - loss: 7.8676 - accuracy: 0.4869
 1024/25000 [>.............................] - ETA: 1:07 - loss: 7.8763 - accuracy: 0.4863
 1056/25000 [>.............................] - ETA: 1:06 - loss: 7.8409 - accuracy: 0.4886
 1088/25000 [>.............................] - ETA: 1:06 - loss: 7.7794 - accuracy: 0.4926
 1120/25000 [>.............................] - ETA: 1:06 - loss: 7.7488 - accuracy: 0.4946
 1152/25000 [>.............................] - ETA: 1:05 - loss: 7.7864 - accuracy: 0.4922
 1184/25000 [>.............................] - ETA: 1:05 - loss: 7.7702 - accuracy: 0.4932
 1216/25000 [>.............................] - ETA: 1:05 - loss: 7.7675 - accuracy: 0.4934
 1248/25000 [>.............................] - ETA: 1:05 - loss: 7.7649 - accuracy: 0.4936
 1280/25000 [>.............................] - ETA: 1:05 - loss: 7.6906 - accuracy: 0.4984
 1312/25000 [>.............................] - ETA: 1:04 - loss: 7.6900 - accuracy: 0.4985
 1344/25000 [>.............................] - ETA: 1:04 - loss: 7.6666 - accuracy: 0.5000
 1376/25000 [>.............................] - ETA: 1:04 - loss: 7.6555 - accuracy: 0.5007
 1408/25000 [>.............................] - ETA: 1:04 - loss: 7.6448 - accuracy: 0.5014
 1440/25000 [>.............................] - ETA: 1:03 - loss: 7.6347 - accuracy: 0.5021
 1472/25000 [>.............................] - ETA: 1:03 - loss: 7.6041 - accuracy: 0.5041
 1504/25000 [>.............................] - ETA: 1:03 - loss: 7.5545 - accuracy: 0.5073
 1536/25000 [>.............................] - ETA: 1:03 - loss: 7.5668 - accuracy: 0.5065
 1568/25000 [>.............................] - ETA: 1:02 - loss: 7.5591 - accuracy: 0.5070
 1600/25000 [>.............................] - ETA: 1:02 - loss: 7.5133 - accuracy: 0.5100
 1632/25000 [>.............................] - ETA: 1:02 - loss: 7.5069 - accuracy: 0.5104
 1664/25000 [>.............................] - ETA: 1:02 - loss: 7.4731 - accuracy: 0.5126
 1696/25000 [=>............................] - ETA: 1:01 - loss: 7.4948 - accuracy: 0.5112
 1728/25000 [=>............................] - ETA: 1:01 - loss: 7.4892 - accuracy: 0.5116
 1760/25000 [=>............................] - ETA: 1:01 - loss: 7.5011 - accuracy: 0.5108
 1792/25000 [=>............................] - ETA: 1:01 - loss: 7.5126 - accuracy: 0.5100
 1824/25000 [=>............................] - ETA: 1:01 - loss: 7.5069 - accuracy: 0.5104
 1856/25000 [=>............................] - ETA: 1:01 - loss: 7.5097 - accuracy: 0.5102
 1888/25000 [=>............................] - ETA: 1:01 - loss: 7.4879 - accuracy: 0.5117
 1920/25000 [=>............................] - ETA: 1:00 - loss: 7.4909 - accuracy: 0.5115
 1952/25000 [=>............................] - ETA: 1:00 - loss: 7.4467 - accuracy: 0.5143
 1984/25000 [=>............................] - ETA: 1:00 - loss: 7.4348 - accuracy: 0.5151
 2016/25000 [=>............................] - ETA: 1:00 - loss: 7.4156 - accuracy: 0.5164
 2048/25000 [=>............................] - ETA: 1:00 - loss: 7.4196 - accuracy: 0.5161
 2080/25000 [=>............................] - ETA: 1:00 - loss: 7.3865 - accuracy: 0.5183
 2112/25000 [=>............................] - ETA: 1:00 - loss: 7.3907 - accuracy: 0.5180
 2144/25000 [=>............................] - ETA: 1:00 - loss: 7.3949 - accuracy: 0.5177
 2176/25000 [=>............................] - ETA: 59s - loss: 7.3848 - accuracy: 0.5184 
 2208/25000 [=>............................] - ETA: 59s - loss: 7.4166 - accuracy: 0.5163
 2240/25000 [=>............................] - ETA: 59s - loss: 7.4202 - accuracy: 0.5161
 2272/25000 [=>............................] - ETA: 59s - loss: 7.4304 - accuracy: 0.5154
 2304/25000 [=>............................] - ETA: 59s - loss: 7.4537 - accuracy: 0.5139
 2336/25000 [=>............................] - ETA: 59s - loss: 7.4631 - accuracy: 0.5133
 2368/25000 [=>............................] - ETA: 59s - loss: 7.4659 - accuracy: 0.5131
 2400/25000 [=>............................] - ETA: 58s - loss: 7.4686 - accuracy: 0.5129
 2432/25000 [=>............................] - ETA: 58s - loss: 7.4586 - accuracy: 0.5136
 2464/25000 [=>............................] - ETA: 58s - loss: 7.4613 - accuracy: 0.5134
 2496/25000 [=>............................] - ETA: 58s - loss: 7.4578 - accuracy: 0.5136
 2528/25000 [==>...........................] - ETA: 58s - loss: 7.4665 - accuracy: 0.5131
 2560/25000 [==>...........................] - ETA: 58s - loss: 7.4570 - accuracy: 0.5137
 2592/25000 [==>...........................] - ETA: 58s - loss: 7.4477 - accuracy: 0.5143
 2624/25000 [==>...........................] - ETA: 58s - loss: 7.4563 - accuracy: 0.5137
 2656/25000 [==>...........................] - ETA: 58s - loss: 7.4530 - accuracy: 0.5139
 2688/25000 [==>...........................] - ETA: 58s - loss: 7.4784 - accuracy: 0.5123
 2720/25000 [==>...........................] - ETA: 57s - loss: 7.4750 - accuracy: 0.5125
 2752/25000 [==>...........................] - ETA: 57s - loss: 7.4605 - accuracy: 0.5134
 2784/25000 [==>...........................] - ETA: 57s - loss: 7.4628 - accuracy: 0.5133
 2816/25000 [==>...........................] - ETA: 57s - loss: 7.4488 - accuracy: 0.5142
 2848/25000 [==>...........................] - ETA: 57s - loss: 7.4351 - accuracy: 0.5151
 2880/25000 [==>...........................] - ETA: 57s - loss: 7.4483 - accuracy: 0.5142
 2912/25000 [==>...........................] - ETA: 57s - loss: 7.4244 - accuracy: 0.5158
 2944/25000 [==>...........................] - ETA: 57s - loss: 7.4166 - accuracy: 0.5163
 2976/25000 [==>...........................] - ETA: 57s - loss: 7.4090 - accuracy: 0.5168
 3008/25000 [==>...........................] - ETA: 56s - loss: 7.4168 - accuracy: 0.5163
 3040/25000 [==>...........................] - ETA: 56s - loss: 7.4043 - accuracy: 0.5171
 3072/25000 [==>...........................] - ETA: 56s - loss: 7.4171 - accuracy: 0.5163
 3104/25000 [==>...........................] - ETA: 56s - loss: 7.4147 - accuracy: 0.5164
 3136/25000 [==>...........................] - ETA: 56s - loss: 7.4173 - accuracy: 0.5163
 3168/25000 [==>...........................] - ETA: 56s - loss: 7.4004 - accuracy: 0.5174
 3200/25000 [==>...........................] - ETA: 56s - loss: 7.4127 - accuracy: 0.5166
 3232/25000 [==>...........................] - ETA: 56s - loss: 7.4104 - accuracy: 0.5167
 3264/25000 [==>...........................] - ETA: 56s - loss: 7.4176 - accuracy: 0.5162
 3296/25000 [==>...........................] - ETA: 56s - loss: 7.4247 - accuracy: 0.5158
 3328/25000 [==>...........................] - ETA: 55s - loss: 7.4316 - accuracy: 0.5153
 3360/25000 [===>..........................] - ETA: 55s - loss: 7.4202 - accuracy: 0.5161
 3392/25000 [===>..........................] - ETA: 55s - loss: 7.4361 - accuracy: 0.5150
 3424/25000 [===>..........................] - ETA: 55s - loss: 7.4382 - accuracy: 0.5149
 3456/25000 [===>..........................] - ETA: 55s - loss: 7.4492 - accuracy: 0.5142
 3488/25000 [===>..........................] - ETA: 55s - loss: 7.4292 - accuracy: 0.5155
 3520/25000 [===>..........................] - ETA: 55s - loss: 7.4183 - accuracy: 0.5162
 3552/25000 [===>..........................] - ETA: 55s - loss: 7.4206 - accuracy: 0.5160
 3584/25000 [===>..........................] - ETA: 55s - loss: 7.4313 - accuracy: 0.5153
 3616/25000 [===>..........................] - ETA: 55s - loss: 7.4673 - accuracy: 0.5130
 3648/25000 [===>..........................] - ETA: 54s - loss: 7.4649 - accuracy: 0.5132
 3680/25000 [===>..........................] - ETA: 54s - loss: 7.4708 - accuracy: 0.5128
 3712/25000 [===>..........................] - ETA: 54s - loss: 7.4601 - accuracy: 0.5135
 3744/25000 [===>..........................] - ETA: 54s - loss: 7.4659 - accuracy: 0.5131
 3776/25000 [===>..........................] - ETA: 54s - loss: 7.4798 - accuracy: 0.5122
 3808/25000 [===>..........................] - ETA: 54s - loss: 7.4774 - accuracy: 0.5123
 3840/25000 [===>..........................] - ETA: 54s - loss: 7.4710 - accuracy: 0.5128
 3872/25000 [===>..........................] - ETA: 54s - loss: 7.4686 - accuracy: 0.5129
 3904/25000 [===>..........................] - ETA: 54s - loss: 7.4938 - accuracy: 0.5113
 3936/25000 [===>..........................] - ETA: 54s - loss: 7.4835 - accuracy: 0.5119
 3968/25000 [===>..........................] - ETA: 54s - loss: 7.4773 - accuracy: 0.5123
 4000/25000 [===>..........................] - ETA: 53s - loss: 7.4673 - accuracy: 0.5130
 4032/25000 [===>..........................] - ETA: 53s - loss: 7.4613 - accuracy: 0.5134
 4064/25000 [===>..........................] - ETA: 53s - loss: 7.4591 - accuracy: 0.5135
 4096/25000 [===>..........................] - ETA: 53s - loss: 7.4420 - accuracy: 0.5146
 4128/25000 [===>..........................] - ETA: 53s - loss: 7.4363 - accuracy: 0.5150
 4160/25000 [===>..........................] - ETA: 53s - loss: 7.4492 - accuracy: 0.5142
 4192/25000 [====>.........................] - ETA: 53s - loss: 7.4472 - accuracy: 0.5143
 4224/25000 [====>.........................] - ETA: 53s - loss: 7.4416 - accuracy: 0.5147
 4256/25000 [====>.........................] - ETA: 53s - loss: 7.4252 - accuracy: 0.5157
 4288/25000 [====>.........................] - ETA: 53s - loss: 7.4235 - accuracy: 0.5159
 4320/25000 [====>.........................] - ETA: 53s - loss: 7.4359 - accuracy: 0.5150
 4352/25000 [====>.........................] - ETA: 52s - loss: 7.4517 - accuracy: 0.5140
 4384/25000 [====>.........................] - ETA: 52s - loss: 7.4533 - accuracy: 0.5139
 4416/25000 [====>.........................] - ETA: 52s - loss: 7.4479 - accuracy: 0.5143
 4448/25000 [====>.........................] - ETA: 52s - loss: 7.4529 - accuracy: 0.5139
 4480/25000 [====>.........................] - ETA: 52s - loss: 7.4613 - accuracy: 0.5134
 4512/25000 [====>.........................] - ETA: 52s - loss: 7.4593 - accuracy: 0.5135
 4544/25000 [====>.........................] - ETA: 52s - loss: 7.4642 - accuracy: 0.5132
 4576/25000 [====>.........................] - ETA: 52s - loss: 7.4756 - accuracy: 0.5125
 4608/25000 [====>.........................] - ETA: 52s - loss: 7.4803 - accuracy: 0.5122
 4640/25000 [====>.........................] - ETA: 52s - loss: 7.4816 - accuracy: 0.5121
 4672/25000 [====>.........................] - ETA: 52s - loss: 7.4861 - accuracy: 0.5118
 4704/25000 [====>.........................] - ETA: 51s - loss: 7.4743 - accuracy: 0.5125
 4736/25000 [====>.........................] - ETA: 51s - loss: 7.4853 - accuracy: 0.5118
 4768/25000 [====>.........................] - ETA: 51s - loss: 7.4865 - accuracy: 0.5117
 4800/25000 [====>.........................] - ETA: 51s - loss: 7.4973 - accuracy: 0.5110
 4832/25000 [====>.........................] - ETA: 51s - loss: 7.4984 - accuracy: 0.5110
 4864/25000 [====>.........................] - ETA: 51s - loss: 7.5027 - accuracy: 0.5107
 4896/25000 [====>.........................] - ETA: 51s - loss: 7.4944 - accuracy: 0.5112
 4928/25000 [====>.........................] - ETA: 51s - loss: 7.4986 - accuracy: 0.5110
 4960/25000 [====>.........................] - ETA: 51s - loss: 7.4997 - accuracy: 0.5109
 4992/25000 [====>.........................] - ETA: 51s - loss: 7.4823 - accuracy: 0.5120
 5024/25000 [=====>........................] - ETA: 51s - loss: 7.4835 - accuracy: 0.5119
 5056/25000 [=====>........................] - ETA: 51s - loss: 7.4786 - accuracy: 0.5123
 5088/25000 [=====>........................] - ETA: 50s - loss: 7.4768 - accuracy: 0.5124
 5120/25000 [=====>........................] - ETA: 50s - loss: 7.4989 - accuracy: 0.5109
 5152/25000 [=====>........................] - ETA: 50s - loss: 7.4940 - accuracy: 0.5113
 5184/25000 [=====>........................] - ETA: 50s - loss: 7.4980 - accuracy: 0.5110
 5216/25000 [=====>........................] - ETA: 50s - loss: 7.4932 - accuracy: 0.5113
 5248/25000 [=====>........................] - ETA: 50s - loss: 7.5030 - accuracy: 0.5107
 5280/25000 [=====>........................] - ETA: 50s - loss: 7.5011 - accuracy: 0.5108
 5312/25000 [=====>........................] - ETA: 50s - loss: 7.4992 - accuracy: 0.5109
 5344/25000 [=====>........................] - ETA: 50s - loss: 7.5002 - accuracy: 0.5109
 5376/25000 [=====>........................] - ETA: 50s - loss: 7.4983 - accuracy: 0.5110
 5408/25000 [=====>........................] - ETA: 50s - loss: 7.4965 - accuracy: 0.5111
 5440/25000 [=====>........................] - ETA: 49s - loss: 7.4919 - accuracy: 0.5114
 5472/25000 [=====>........................] - ETA: 49s - loss: 7.4845 - accuracy: 0.5119
 5504/25000 [=====>........................] - ETA: 49s - loss: 7.5023 - accuracy: 0.5107
 5536/25000 [=====>........................] - ETA: 49s - loss: 7.5087 - accuracy: 0.5103
 5568/25000 [=====>........................] - ETA: 49s - loss: 7.5124 - accuracy: 0.5101
 5600/25000 [=====>........................] - ETA: 49s - loss: 7.5215 - accuracy: 0.5095
 5632/25000 [=====>........................] - ETA: 49s - loss: 7.5142 - accuracy: 0.5099
 5664/25000 [=====>........................] - ETA: 49s - loss: 7.5150 - accuracy: 0.5099
 5696/25000 [=====>........................] - ETA: 49s - loss: 7.5078 - accuracy: 0.5104
 5728/25000 [=====>........................] - ETA: 49s - loss: 7.5114 - accuracy: 0.5101
 5760/25000 [=====>........................] - ETA: 49s - loss: 7.5149 - accuracy: 0.5099
 5792/25000 [=====>........................] - ETA: 49s - loss: 7.5184 - accuracy: 0.5097
 5824/25000 [=====>........................] - ETA: 48s - loss: 7.5166 - accuracy: 0.5098
 5856/25000 [======>.......................] - ETA: 48s - loss: 7.5148 - accuracy: 0.5099
 5888/25000 [======>.......................] - ETA: 48s - loss: 7.5260 - accuracy: 0.5092
 5920/25000 [======>.......................] - ETA: 48s - loss: 7.5319 - accuracy: 0.5088
 5952/25000 [======>.......................] - ETA: 48s - loss: 7.5378 - accuracy: 0.5084
 5984/25000 [======>.......................] - ETA: 48s - loss: 7.5385 - accuracy: 0.5084
 6016/25000 [======>.......................] - ETA: 48s - loss: 7.5494 - accuracy: 0.5076
 6048/25000 [======>.......................] - ETA: 48s - loss: 7.5627 - accuracy: 0.5068
 6080/25000 [======>.......................] - ETA: 48s - loss: 7.5683 - accuracy: 0.5064
 6112/25000 [======>.......................] - ETA: 48s - loss: 7.5713 - accuracy: 0.5062
 6144/25000 [======>.......................] - ETA: 48s - loss: 7.5693 - accuracy: 0.5063
 6176/25000 [======>.......................] - ETA: 48s - loss: 7.5698 - accuracy: 0.5063
 6208/25000 [======>.......................] - ETA: 48s - loss: 7.5802 - accuracy: 0.5056
 6240/25000 [======>.......................] - ETA: 47s - loss: 7.5880 - accuracy: 0.5051
 6272/25000 [======>.......................] - ETA: 47s - loss: 7.5933 - accuracy: 0.5048
 6304/25000 [======>.......................] - ETA: 47s - loss: 7.5985 - accuracy: 0.5044
 6336/25000 [======>.......................] - ETA: 47s - loss: 7.5940 - accuracy: 0.5047
 6368/25000 [======>.......................] - ETA: 47s - loss: 7.5920 - accuracy: 0.5049
 6400/25000 [======>.......................] - ETA: 47s - loss: 7.5995 - accuracy: 0.5044
 6432/25000 [======>.......................] - ETA: 47s - loss: 7.6070 - accuracy: 0.5039
 6464/25000 [======>.......................] - ETA: 47s - loss: 7.6121 - accuracy: 0.5036
 6496/25000 [======>.......................] - ETA: 47s - loss: 7.6123 - accuracy: 0.5035
 6528/25000 [======>.......................] - ETA: 47s - loss: 7.6149 - accuracy: 0.5034
 6560/25000 [======>.......................] - ETA: 46s - loss: 7.6222 - accuracy: 0.5029
 6592/25000 [======>.......................] - ETA: 46s - loss: 7.6154 - accuracy: 0.5033
 6624/25000 [======>.......................] - ETA: 46s - loss: 7.6180 - accuracy: 0.5032
 6656/25000 [======>.......................] - ETA: 46s - loss: 7.6182 - accuracy: 0.5032
 6688/25000 [=======>......................] - ETA: 46s - loss: 7.6139 - accuracy: 0.5034
 6720/25000 [=======>......................] - ETA: 46s - loss: 7.6278 - accuracy: 0.5025
 6752/25000 [=======>......................] - ETA: 46s - loss: 7.6371 - accuracy: 0.5019
 6784/25000 [=======>......................] - ETA: 46s - loss: 7.6372 - accuracy: 0.5019
 6816/25000 [=======>......................] - ETA: 46s - loss: 7.6419 - accuracy: 0.5016
 6848/25000 [=======>......................] - ETA: 46s - loss: 7.6263 - accuracy: 0.5026
 6880/25000 [=======>......................] - ETA: 46s - loss: 7.6287 - accuracy: 0.5025
 6912/25000 [=======>......................] - ETA: 46s - loss: 7.6311 - accuracy: 0.5023
 6944/25000 [=======>......................] - ETA: 45s - loss: 7.6247 - accuracy: 0.5027
 6976/25000 [=======>......................] - ETA: 45s - loss: 7.6161 - accuracy: 0.5033
 7008/25000 [=======>......................] - ETA: 45s - loss: 7.6272 - accuracy: 0.5026
 7040/25000 [=======>......................] - ETA: 45s - loss: 7.6318 - accuracy: 0.5023
 7072/25000 [=======>......................] - ETA: 45s - loss: 7.6384 - accuracy: 0.5018
 7104/25000 [=======>......................] - ETA: 45s - loss: 7.6299 - accuracy: 0.5024
 7136/25000 [=======>......................] - ETA: 45s - loss: 7.6258 - accuracy: 0.5027
 7168/25000 [=======>......................] - ETA: 45s - loss: 7.6217 - accuracy: 0.5029
 7200/25000 [=======>......................] - ETA: 45s - loss: 7.6219 - accuracy: 0.5029
 7232/25000 [=======>......................] - ETA: 45s - loss: 7.6200 - accuracy: 0.5030
 7264/25000 [=======>......................] - ETA: 45s - loss: 7.6202 - accuracy: 0.5030
 7296/25000 [=======>......................] - ETA: 45s - loss: 7.6120 - accuracy: 0.5036
 7328/25000 [=======>......................] - ETA: 45s - loss: 7.5934 - accuracy: 0.5048
 7360/25000 [=======>......................] - ETA: 44s - loss: 7.5895 - accuracy: 0.5050
 7392/25000 [=======>......................] - ETA: 44s - loss: 7.5961 - accuracy: 0.5046
 7424/25000 [=======>......................] - ETA: 44s - loss: 7.5902 - accuracy: 0.5050
 7456/25000 [=======>......................] - ETA: 44s - loss: 7.5926 - accuracy: 0.5048
 7488/25000 [=======>......................] - ETA: 44s - loss: 7.6011 - accuracy: 0.5043
 7520/25000 [========>.....................] - ETA: 44s - loss: 7.6054 - accuracy: 0.5040
 7552/25000 [========>.....................] - ETA: 44s - loss: 7.5996 - accuracy: 0.5044
 7584/25000 [========>.....................] - ETA: 44s - loss: 7.5979 - accuracy: 0.5045
 7616/25000 [========>.....................] - ETA: 44s - loss: 7.6042 - accuracy: 0.5041
 7648/25000 [========>.....................] - ETA: 44s - loss: 7.6065 - accuracy: 0.5039
 7680/25000 [========>.....................] - ETA: 44s - loss: 7.6127 - accuracy: 0.5035
 7712/25000 [========>.....................] - ETA: 44s - loss: 7.6090 - accuracy: 0.5038
 7744/25000 [========>.....................] - ETA: 43s - loss: 7.6033 - accuracy: 0.5041
 7776/25000 [========>.....................] - ETA: 43s - loss: 7.6015 - accuracy: 0.5042
 7808/25000 [========>.....................] - ETA: 43s - loss: 7.6116 - accuracy: 0.5036
 7840/25000 [========>.....................] - ETA: 43s - loss: 7.6138 - accuracy: 0.5034
 7872/25000 [========>.....................] - ETA: 43s - loss: 7.6101 - accuracy: 0.5037
 7904/25000 [========>.....................] - ETA: 43s - loss: 7.6181 - accuracy: 0.5032
 7936/25000 [========>.....................] - ETA: 43s - loss: 7.6067 - accuracy: 0.5039
 7968/25000 [========>.....................] - ETA: 43s - loss: 7.6050 - accuracy: 0.5040
 8000/25000 [========>.....................] - ETA: 43s - loss: 7.5938 - accuracy: 0.5048
 8032/25000 [========>.....................] - ETA: 43s - loss: 7.6017 - accuracy: 0.5042
 8064/25000 [========>.....................] - ETA: 43s - loss: 7.6058 - accuracy: 0.5040
 8096/25000 [========>.....................] - ETA: 43s - loss: 7.5965 - accuracy: 0.5046
 8128/25000 [========>.....................] - ETA: 42s - loss: 7.5949 - accuracy: 0.5047
 8160/25000 [========>.....................] - ETA: 42s - loss: 7.5933 - accuracy: 0.5048
 8192/25000 [========>.....................] - ETA: 42s - loss: 7.5992 - accuracy: 0.5044
 8224/25000 [========>.....................] - ETA: 42s - loss: 7.5958 - accuracy: 0.5046
 8256/25000 [========>.....................] - ETA: 42s - loss: 7.5998 - accuracy: 0.5044
 8288/25000 [========>.....................] - ETA: 42s - loss: 7.5982 - accuracy: 0.5045
 8320/25000 [========>.....................] - ETA: 42s - loss: 7.6021 - accuracy: 0.5042
 8352/25000 [=========>....................] - ETA: 42s - loss: 7.6060 - accuracy: 0.5040
 8384/25000 [=========>....................] - ETA: 42s - loss: 7.6026 - accuracy: 0.5042
 8416/25000 [=========>....................] - ETA: 42s - loss: 7.6120 - accuracy: 0.5036
 8448/25000 [=========>....................] - ETA: 42s - loss: 7.6085 - accuracy: 0.5038
 8480/25000 [=========>....................] - ETA: 42s - loss: 7.6142 - accuracy: 0.5034
 8512/25000 [=========>....................] - ETA: 42s - loss: 7.6162 - accuracy: 0.5033
 8544/25000 [=========>....................] - ETA: 42s - loss: 7.6200 - accuracy: 0.5030
 8576/25000 [=========>....................] - ETA: 41s - loss: 7.6148 - accuracy: 0.5034
 8608/25000 [=========>....................] - ETA: 41s - loss: 7.6114 - accuracy: 0.5036
 8640/25000 [=========>....................] - ETA: 41s - loss: 7.6134 - accuracy: 0.5035
 8672/25000 [=========>....................] - ETA: 41s - loss: 7.6083 - accuracy: 0.5038
 8704/25000 [=========>....................] - ETA: 41s - loss: 7.6120 - accuracy: 0.5036
 8736/25000 [=========>....................] - ETA: 41s - loss: 7.6105 - accuracy: 0.5037
 8768/25000 [=========>....................] - ETA: 41s - loss: 7.6072 - accuracy: 0.5039
 8800/25000 [=========>....................] - ETA: 41s - loss: 7.6056 - accuracy: 0.5040
 8832/25000 [=========>....................] - ETA: 41s - loss: 7.6059 - accuracy: 0.5040
 8864/25000 [=========>....................] - ETA: 41s - loss: 7.6061 - accuracy: 0.5039
 8896/25000 [=========>....................] - ETA: 41s - loss: 7.6046 - accuracy: 0.5040
 8928/25000 [=========>....................] - ETA: 41s - loss: 7.6048 - accuracy: 0.5040
 8960/25000 [=========>....................] - ETA: 40s - loss: 7.6067 - accuracy: 0.5039
 8992/25000 [=========>....................] - ETA: 40s - loss: 7.6103 - accuracy: 0.5037
 9024/25000 [=========>....................] - ETA: 40s - loss: 7.6054 - accuracy: 0.5040
 9056/25000 [=========>....................] - ETA: 40s - loss: 7.6091 - accuracy: 0.5038
 9088/25000 [=========>....................] - ETA: 40s - loss: 7.6059 - accuracy: 0.5040
 9120/25000 [=========>....................] - ETA: 40s - loss: 7.6010 - accuracy: 0.5043
 9152/25000 [=========>....................] - ETA: 40s - loss: 7.6080 - accuracy: 0.5038
 9184/25000 [==========>...................] - ETA: 40s - loss: 7.6082 - accuracy: 0.5038
 9216/25000 [==========>...................] - ETA: 40s - loss: 7.6084 - accuracy: 0.5038
 9248/25000 [==========>...................] - ETA: 40s - loss: 7.6102 - accuracy: 0.5037
 9280/25000 [==========>...................] - ETA: 40s - loss: 7.6022 - accuracy: 0.5042
 9312/25000 [==========>...................] - ETA: 40s - loss: 7.6106 - accuracy: 0.5037
 9344/25000 [==========>...................] - ETA: 39s - loss: 7.6174 - accuracy: 0.5032
 9376/25000 [==========>...................] - ETA: 39s - loss: 7.6159 - accuracy: 0.5033
 9408/25000 [==========>...................] - ETA: 39s - loss: 7.6145 - accuracy: 0.5034
 9440/25000 [==========>...................] - ETA: 39s - loss: 7.6179 - accuracy: 0.5032
 9472/25000 [==========>...................] - ETA: 39s - loss: 7.6132 - accuracy: 0.5035
 9504/25000 [==========>...................] - ETA: 39s - loss: 7.6150 - accuracy: 0.5034
 9536/25000 [==========>...................] - ETA: 39s - loss: 7.6152 - accuracy: 0.5034
 9568/25000 [==========>...................] - ETA: 39s - loss: 7.6169 - accuracy: 0.5032
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.6187 - accuracy: 0.5031
 9632/25000 [==========>...................] - ETA: 39s - loss: 7.6236 - accuracy: 0.5028
 9664/25000 [==========>...................] - ETA: 39s - loss: 7.6254 - accuracy: 0.5027
 9696/25000 [==========>...................] - ETA: 39s - loss: 7.6160 - accuracy: 0.5033
 9728/25000 [==========>...................] - ETA: 39s - loss: 7.6225 - accuracy: 0.5029
 9760/25000 [==========>...................] - ETA: 38s - loss: 7.6179 - accuracy: 0.5032
 9792/25000 [==========>...................] - ETA: 38s - loss: 7.6149 - accuracy: 0.5034
 9824/25000 [==========>...................] - ETA: 38s - loss: 7.6120 - accuracy: 0.5036
 9856/25000 [==========>...................] - ETA: 38s - loss: 7.6215 - accuracy: 0.5029
 9888/25000 [==========>...................] - ETA: 38s - loss: 7.6216 - accuracy: 0.5029
 9920/25000 [==========>...................] - ETA: 38s - loss: 7.6218 - accuracy: 0.5029
 9952/25000 [==========>...................] - ETA: 38s - loss: 7.6296 - accuracy: 0.5024
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.6344 - accuracy: 0.5021
10016/25000 [===========>..................] - ETA: 38s - loss: 7.6391 - accuracy: 0.5018
10048/25000 [===========>..................] - ETA: 38s - loss: 7.6376 - accuracy: 0.5019
10080/25000 [===========>..................] - ETA: 38s - loss: 7.6362 - accuracy: 0.5020
10112/25000 [===========>..................] - ETA: 38s - loss: 7.6302 - accuracy: 0.5024
10144/25000 [===========>..................] - ETA: 37s - loss: 7.6394 - accuracy: 0.5018
10176/25000 [===========>..................] - ETA: 37s - loss: 7.6425 - accuracy: 0.5016
10208/25000 [===========>..................] - ETA: 37s - loss: 7.6411 - accuracy: 0.5017
10240/25000 [===========>..................] - ETA: 37s - loss: 7.6487 - accuracy: 0.5012
10272/25000 [===========>..................] - ETA: 37s - loss: 7.6517 - accuracy: 0.5010
10304/25000 [===========>..................] - ETA: 37s - loss: 7.6517 - accuracy: 0.5010
10336/25000 [===========>..................] - ETA: 37s - loss: 7.6533 - accuracy: 0.5009
10368/25000 [===========>..................] - ETA: 37s - loss: 7.6518 - accuracy: 0.5010
10400/25000 [===========>..................] - ETA: 37s - loss: 7.6534 - accuracy: 0.5009
10432/25000 [===========>..................] - ETA: 37s - loss: 7.6549 - accuracy: 0.5008
10464/25000 [===========>..................] - ETA: 37s - loss: 7.6564 - accuracy: 0.5007
10496/25000 [===========>..................] - ETA: 37s - loss: 7.6549 - accuracy: 0.5008
10528/25000 [===========>..................] - ETA: 36s - loss: 7.6462 - accuracy: 0.5013
10560/25000 [===========>..................] - ETA: 36s - loss: 7.6536 - accuracy: 0.5009
10592/25000 [===========>..................] - ETA: 36s - loss: 7.6550 - accuracy: 0.5008
10624/25000 [===========>..................] - ETA: 36s - loss: 7.6594 - accuracy: 0.5005
10656/25000 [===========>..................] - ETA: 36s - loss: 7.6652 - accuracy: 0.5001
10688/25000 [===========>..................] - ETA: 36s - loss: 7.6638 - accuracy: 0.5002
10720/25000 [===========>..................] - ETA: 36s - loss: 7.6666 - accuracy: 0.5000
10752/25000 [===========>..................] - ETA: 36s - loss: 7.6638 - accuracy: 0.5002
10784/25000 [===========>..................] - ETA: 36s - loss: 7.6680 - accuracy: 0.4999
10816/25000 [===========>..................] - ETA: 36s - loss: 7.6695 - accuracy: 0.4998
10848/25000 [============>.................] - ETA: 36s - loss: 7.6709 - accuracy: 0.4997
10880/25000 [============>.................] - ETA: 36s - loss: 7.6708 - accuracy: 0.4997
10912/25000 [============>.................] - ETA: 35s - loss: 7.6694 - accuracy: 0.4998
10944/25000 [============>.................] - ETA: 35s - loss: 7.6666 - accuracy: 0.5000
10976/25000 [============>.................] - ETA: 35s - loss: 7.6722 - accuracy: 0.4996
11008/25000 [============>.................] - ETA: 35s - loss: 7.6764 - accuracy: 0.4994
11040/25000 [============>.................] - ETA: 35s - loss: 7.6805 - accuracy: 0.4991
11072/25000 [============>.................] - ETA: 35s - loss: 7.6846 - accuracy: 0.4988
11104/25000 [============>.................] - ETA: 35s - loss: 7.6846 - accuracy: 0.4988
11136/25000 [============>.................] - ETA: 35s - loss: 7.6776 - accuracy: 0.4993
11168/25000 [============>.................] - ETA: 35s - loss: 7.6790 - accuracy: 0.4992
11200/25000 [============>.................] - ETA: 35s - loss: 7.6776 - accuracy: 0.4993
11232/25000 [============>.................] - ETA: 35s - loss: 7.6762 - accuracy: 0.4994
11264/25000 [============>.................] - ETA: 35s - loss: 7.6707 - accuracy: 0.4997
11296/25000 [============>.................] - ETA: 35s - loss: 7.6598 - accuracy: 0.5004
11328/25000 [============>.................] - ETA: 34s - loss: 7.6585 - accuracy: 0.5005
11360/25000 [============>.................] - ETA: 34s - loss: 7.6612 - accuracy: 0.5004
11392/25000 [============>.................] - ETA: 34s - loss: 7.6653 - accuracy: 0.5001
11424/25000 [============>.................] - ETA: 34s - loss: 7.6747 - accuracy: 0.4995
11456/25000 [============>.................] - ETA: 34s - loss: 7.6706 - accuracy: 0.4997
11488/25000 [============>.................] - ETA: 34s - loss: 7.6626 - accuracy: 0.5003
11520/25000 [============>.................] - ETA: 34s - loss: 7.6653 - accuracy: 0.5001
11552/25000 [============>.................] - ETA: 34s - loss: 7.6613 - accuracy: 0.5003
11584/25000 [============>.................] - ETA: 34s - loss: 7.6640 - accuracy: 0.5002
11616/25000 [============>.................] - ETA: 34s - loss: 7.6640 - accuracy: 0.5002
11648/25000 [============>.................] - ETA: 34s - loss: 7.6693 - accuracy: 0.4998
11680/25000 [=============>................] - ETA: 33s - loss: 7.6732 - accuracy: 0.4996
11712/25000 [=============>................] - ETA: 33s - loss: 7.6758 - accuracy: 0.4994
11744/25000 [=============>................] - ETA: 33s - loss: 7.6745 - accuracy: 0.4995
11776/25000 [=============>................] - ETA: 33s - loss: 7.6718 - accuracy: 0.4997
11808/25000 [=============>................] - ETA: 33s - loss: 7.6770 - accuracy: 0.4993
11840/25000 [=============>................] - ETA: 33s - loss: 7.6783 - accuracy: 0.4992
11872/25000 [=============>................] - ETA: 33s - loss: 7.6757 - accuracy: 0.4994
11904/25000 [=============>................] - ETA: 33s - loss: 7.6756 - accuracy: 0.4994
11936/25000 [=============>................] - ETA: 33s - loss: 7.6743 - accuracy: 0.4995
11968/25000 [=============>................] - ETA: 33s - loss: 7.6717 - accuracy: 0.4997
12000/25000 [=============>................] - ETA: 33s - loss: 7.6768 - accuracy: 0.4993
12032/25000 [=============>................] - ETA: 33s - loss: 7.6794 - accuracy: 0.4992
12064/25000 [=============>................] - ETA: 33s - loss: 7.6844 - accuracy: 0.4988
12096/25000 [=============>................] - ETA: 32s - loss: 7.6755 - accuracy: 0.4994
12128/25000 [=============>................] - ETA: 32s - loss: 7.6767 - accuracy: 0.4993
12160/25000 [=============>................] - ETA: 32s - loss: 7.6717 - accuracy: 0.4997
12192/25000 [=============>................] - ETA: 32s - loss: 7.6691 - accuracy: 0.4998
12224/25000 [=============>................] - ETA: 32s - loss: 7.6679 - accuracy: 0.4999
12256/25000 [=============>................] - ETA: 32s - loss: 7.6679 - accuracy: 0.4999
12288/25000 [=============>................] - ETA: 32s - loss: 7.6679 - accuracy: 0.4999
12320/25000 [=============>................] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
12352/25000 [=============>................] - ETA: 32s - loss: 7.6703 - accuracy: 0.4998
12384/25000 [=============>................] - ETA: 32s - loss: 7.6679 - accuracy: 0.4999
12416/25000 [=============>................] - ETA: 32s - loss: 7.6617 - accuracy: 0.5003
12448/25000 [=============>................] - ETA: 32s - loss: 7.6642 - accuracy: 0.5002
12480/25000 [=============>................] - ETA: 31s - loss: 7.6580 - accuracy: 0.5006
12512/25000 [==============>...............] - ETA: 31s - loss: 7.6580 - accuracy: 0.5006
12544/25000 [==============>...............] - ETA: 31s - loss: 7.6605 - accuracy: 0.5004
12576/25000 [==============>...............] - ETA: 31s - loss: 7.6593 - accuracy: 0.5005
12608/25000 [==============>...............] - ETA: 31s - loss: 7.6618 - accuracy: 0.5003
12640/25000 [==============>...............] - ETA: 31s - loss: 7.6654 - accuracy: 0.5001
12672/25000 [==============>...............] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
12704/25000 [==============>...............] - ETA: 31s - loss: 7.6678 - accuracy: 0.4999
12736/25000 [==============>...............] - ETA: 31s - loss: 7.6702 - accuracy: 0.4998
12768/25000 [==============>...............] - ETA: 31s - loss: 7.6690 - accuracy: 0.4998
12800/25000 [==============>...............] - ETA: 31s - loss: 7.6702 - accuracy: 0.4998
12832/25000 [==============>...............] - ETA: 31s - loss: 7.6690 - accuracy: 0.4998
12864/25000 [==============>...............] - ETA: 30s - loss: 7.6654 - accuracy: 0.5001
12896/25000 [==============>...............] - ETA: 30s - loss: 7.6642 - accuracy: 0.5002
12928/25000 [==============>...............] - ETA: 30s - loss: 7.6631 - accuracy: 0.5002
12960/25000 [==============>...............] - ETA: 30s - loss: 7.6631 - accuracy: 0.5002
12992/25000 [==============>...............] - ETA: 30s - loss: 7.6595 - accuracy: 0.5005
13024/25000 [==============>...............] - ETA: 30s - loss: 7.6619 - accuracy: 0.5003
13056/25000 [==============>...............] - ETA: 30s - loss: 7.6607 - accuracy: 0.5004
13088/25000 [==============>...............] - ETA: 30s - loss: 7.6596 - accuracy: 0.5005
13120/25000 [==============>...............] - ETA: 30s - loss: 7.6596 - accuracy: 0.5005
13152/25000 [==============>...............] - ETA: 30s - loss: 7.6596 - accuracy: 0.5005
13184/25000 [==============>...............] - ETA: 30s - loss: 7.6678 - accuracy: 0.4999
13216/25000 [==============>...............] - ETA: 30s - loss: 7.6724 - accuracy: 0.4996
13248/25000 [==============>...............] - ETA: 29s - loss: 7.6689 - accuracy: 0.4998
13280/25000 [==============>...............] - ETA: 29s - loss: 7.6678 - accuracy: 0.4999
13312/25000 [==============>...............] - ETA: 29s - loss: 7.6643 - accuracy: 0.5002
13344/25000 [===============>..............] - ETA: 29s - loss: 7.6655 - accuracy: 0.5001
13376/25000 [===============>..............] - ETA: 29s - loss: 7.6689 - accuracy: 0.4999
13408/25000 [===============>..............] - ETA: 29s - loss: 7.6689 - accuracy: 0.4999
13440/25000 [===============>..............] - ETA: 29s - loss: 7.6746 - accuracy: 0.4995
13472/25000 [===============>..............] - ETA: 29s - loss: 7.6746 - accuracy: 0.4995
13504/25000 [===============>..............] - ETA: 29s - loss: 7.6746 - accuracy: 0.4995
13536/25000 [===============>..............] - ETA: 29s - loss: 7.6791 - accuracy: 0.4992
13568/25000 [===============>..............] - ETA: 29s - loss: 7.6824 - accuracy: 0.4990
13600/25000 [===============>..............] - ETA: 29s - loss: 7.6892 - accuracy: 0.4985
13632/25000 [===============>..............] - ETA: 28s - loss: 7.6902 - accuracy: 0.4985
13664/25000 [===============>..............] - ETA: 28s - loss: 7.6902 - accuracy: 0.4985
13696/25000 [===============>..............] - ETA: 28s - loss: 7.6912 - accuracy: 0.4984
13728/25000 [===============>..............] - ETA: 28s - loss: 7.6878 - accuracy: 0.4986
13760/25000 [===============>..............] - ETA: 28s - loss: 7.6878 - accuracy: 0.4986
13792/25000 [===============>..............] - ETA: 28s - loss: 7.6833 - accuracy: 0.4989
13824/25000 [===============>..............] - ETA: 28s - loss: 7.6810 - accuracy: 0.4991
13856/25000 [===============>..............] - ETA: 28s - loss: 7.6799 - accuracy: 0.4991
13888/25000 [===============>..............] - ETA: 28s - loss: 7.6755 - accuracy: 0.4994
13920/25000 [===============>..............] - ETA: 28s - loss: 7.6743 - accuracy: 0.4995
13952/25000 [===============>..............] - ETA: 28s - loss: 7.6809 - accuracy: 0.4991
13984/25000 [===============>..............] - ETA: 28s - loss: 7.6776 - accuracy: 0.4993
14016/25000 [===============>..............] - ETA: 27s - loss: 7.6776 - accuracy: 0.4993
14048/25000 [===============>..............] - ETA: 27s - loss: 7.6764 - accuracy: 0.4994
14080/25000 [===============>..............] - ETA: 27s - loss: 7.6742 - accuracy: 0.4995
14112/25000 [===============>..............] - ETA: 27s - loss: 7.6688 - accuracy: 0.4999
14144/25000 [===============>..............] - ETA: 27s - loss: 7.6710 - accuracy: 0.4997
14176/25000 [================>.............] - ETA: 27s - loss: 7.6742 - accuracy: 0.4995
14208/25000 [================>.............] - ETA: 27s - loss: 7.6666 - accuracy: 0.5000
14240/25000 [================>.............] - ETA: 27s - loss: 7.6655 - accuracy: 0.5001
14272/25000 [================>.............] - ETA: 27s - loss: 7.6634 - accuracy: 0.5002
14304/25000 [================>.............] - ETA: 27s - loss: 7.6666 - accuracy: 0.5000
14336/25000 [================>.............] - ETA: 27s - loss: 7.6688 - accuracy: 0.4999
14368/25000 [================>.............] - ETA: 27s - loss: 7.6698 - accuracy: 0.4998
14400/25000 [================>.............] - ETA: 26s - loss: 7.6730 - accuracy: 0.4996
14432/25000 [================>.............] - ETA: 26s - loss: 7.6698 - accuracy: 0.4998
14464/25000 [================>.............] - ETA: 26s - loss: 7.6645 - accuracy: 0.5001
14496/25000 [================>.............] - ETA: 26s - loss: 7.6666 - accuracy: 0.5000
14528/25000 [================>.............] - ETA: 26s - loss: 7.6624 - accuracy: 0.5003
14560/25000 [================>.............] - ETA: 26s - loss: 7.6603 - accuracy: 0.5004
14592/25000 [================>.............] - ETA: 26s - loss: 7.6635 - accuracy: 0.5002
14624/25000 [================>.............] - ETA: 26s - loss: 7.6614 - accuracy: 0.5003
14656/25000 [================>.............] - ETA: 26s - loss: 7.6624 - accuracy: 0.5003
14688/25000 [================>.............] - ETA: 26s - loss: 7.6677 - accuracy: 0.4999
14720/25000 [================>.............] - ETA: 26s - loss: 7.6677 - accuracy: 0.4999
14752/25000 [================>.............] - ETA: 26s - loss: 7.6687 - accuracy: 0.4999
14784/25000 [================>.............] - ETA: 25s - loss: 7.6666 - accuracy: 0.5000
14816/25000 [================>.............] - ETA: 25s - loss: 7.6645 - accuracy: 0.5001
14848/25000 [================>.............] - ETA: 25s - loss: 7.6594 - accuracy: 0.5005
14880/25000 [================>.............] - ETA: 25s - loss: 7.6635 - accuracy: 0.5002
14912/25000 [================>.............] - ETA: 25s - loss: 7.6584 - accuracy: 0.5005
14944/25000 [================>.............] - ETA: 25s - loss: 7.6574 - accuracy: 0.5006
14976/25000 [================>.............] - ETA: 25s - loss: 7.6595 - accuracy: 0.5005
15008/25000 [=================>............] - ETA: 25s - loss: 7.6595 - accuracy: 0.5005
15040/25000 [=================>............] - ETA: 25s - loss: 7.6564 - accuracy: 0.5007
15072/25000 [=================>............] - ETA: 25s - loss: 7.6524 - accuracy: 0.5009
15104/25000 [=================>............] - ETA: 25s - loss: 7.6595 - accuracy: 0.5005
15136/25000 [=================>............] - ETA: 25s - loss: 7.6605 - accuracy: 0.5004
15168/25000 [=================>............] - ETA: 25s - loss: 7.6656 - accuracy: 0.5001
15200/25000 [=================>............] - ETA: 24s - loss: 7.6676 - accuracy: 0.4999
15232/25000 [=================>............] - ETA: 24s - loss: 7.6626 - accuracy: 0.5003
15264/25000 [=================>............] - ETA: 24s - loss: 7.6586 - accuracy: 0.5005
15296/25000 [=================>............] - ETA: 24s - loss: 7.6566 - accuracy: 0.5007
15328/25000 [=================>............] - ETA: 24s - loss: 7.6606 - accuracy: 0.5004
15360/25000 [=================>............] - ETA: 24s - loss: 7.6576 - accuracy: 0.5006
15392/25000 [=================>............] - ETA: 24s - loss: 7.6596 - accuracy: 0.5005
15424/25000 [=================>............] - ETA: 24s - loss: 7.6587 - accuracy: 0.5005
15456/25000 [=================>............] - ETA: 24s - loss: 7.6567 - accuracy: 0.5006
15488/25000 [=================>............] - ETA: 24s - loss: 7.6557 - accuracy: 0.5007
15520/25000 [=================>............] - ETA: 24s - loss: 7.6567 - accuracy: 0.5006
15552/25000 [=================>............] - ETA: 23s - loss: 7.6577 - accuracy: 0.5006
15584/25000 [=================>............] - ETA: 23s - loss: 7.6587 - accuracy: 0.5005
15616/25000 [=================>............] - ETA: 23s - loss: 7.6509 - accuracy: 0.5010
15648/25000 [=================>............] - ETA: 23s - loss: 7.6519 - accuracy: 0.5010
15680/25000 [=================>............] - ETA: 23s - loss: 7.6480 - accuracy: 0.5012
15712/25000 [=================>............] - ETA: 23s - loss: 7.6491 - accuracy: 0.5011
15744/25000 [=================>............] - ETA: 23s - loss: 7.6491 - accuracy: 0.5011
15776/25000 [=================>............] - ETA: 23s - loss: 7.6511 - accuracy: 0.5010
15808/25000 [=================>............] - ETA: 23s - loss: 7.6463 - accuracy: 0.5013
15840/25000 [==================>...........] - ETA: 23s - loss: 7.6492 - accuracy: 0.5011
15872/25000 [==================>...........] - ETA: 23s - loss: 7.6463 - accuracy: 0.5013
15904/25000 [==================>...........] - ETA: 23s - loss: 7.6473 - accuracy: 0.5013
15936/25000 [==================>...........] - ETA: 23s - loss: 7.6493 - accuracy: 0.5011
15968/25000 [==================>...........] - ETA: 22s - loss: 7.6474 - accuracy: 0.5013
16000/25000 [==================>...........] - ETA: 22s - loss: 7.6455 - accuracy: 0.5014
16032/25000 [==================>...........] - ETA: 22s - loss: 7.6437 - accuracy: 0.5015
16064/25000 [==================>...........] - ETA: 22s - loss: 7.6437 - accuracy: 0.5015
16096/25000 [==================>...........] - ETA: 22s - loss: 7.6399 - accuracy: 0.5017
16128/25000 [==================>...........] - ETA: 22s - loss: 7.6429 - accuracy: 0.5016
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6438 - accuracy: 0.5015
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6401 - accuracy: 0.5017
16224/25000 [==================>...........] - ETA: 22s - loss: 7.6420 - accuracy: 0.5016
16256/25000 [==================>...........] - ETA: 22s - loss: 7.6383 - accuracy: 0.5018
16288/25000 [==================>...........] - ETA: 22s - loss: 7.6356 - accuracy: 0.5020
16320/25000 [==================>...........] - ETA: 21s - loss: 7.6375 - accuracy: 0.5019
16352/25000 [==================>...........] - ETA: 21s - loss: 7.6413 - accuracy: 0.5017
16384/25000 [==================>...........] - ETA: 21s - loss: 7.6404 - accuracy: 0.5017
16416/25000 [==================>...........] - ETA: 21s - loss: 7.6386 - accuracy: 0.5018
16448/25000 [==================>...........] - ETA: 21s - loss: 7.6368 - accuracy: 0.5019
16480/25000 [==================>...........] - ETA: 21s - loss: 7.6331 - accuracy: 0.5022
16512/25000 [==================>...........] - ETA: 21s - loss: 7.6332 - accuracy: 0.5022
16544/25000 [==================>...........] - ETA: 21s - loss: 7.6314 - accuracy: 0.5023
16576/25000 [==================>...........] - ETA: 21s - loss: 7.6324 - accuracy: 0.5022
16608/25000 [==================>...........] - ETA: 21s - loss: 7.6380 - accuracy: 0.5019
16640/25000 [==================>...........] - ETA: 21s - loss: 7.6408 - accuracy: 0.5017
16672/25000 [===================>..........] - ETA: 21s - loss: 7.6381 - accuracy: 0.5019
16704/25000 [===================>..........] - ETA: 20s - loss: 7.6372 - accuracy: 0.5019
16736/25000 [===================>..........] - ETA: 20s - loss: 7.6355 - accuracy: 0.5020
16768/25000 [===================>..........] - ETA: 20s - loss: 7.6310 - accuracy: 0.5023
16800/25000 [===================>..........] - ETA: 20s - loss: 7.6365 - accuracy: 0.5020
16832/25000 [===================>..........] - ETA: 20s - loss: 7.6384 - accuracy: 0.5018
16864/25000 [===================>..........] - ETA: 20s - loss: 7.6421 - accuracy: 0.5016
16896/25000 [===================>..........] - ETA: 20s - loss: 7.6430 - accuracy: 0.5015
16928/25000 [===================>..........] - ETA: 20s - loss: 7.6467 - accuracy: 0.5013
16960/25000 [===================>..........] - ETA: 20s - loss: 7.6458 - accuracy: 0.5014
16992/25000 [===================>..........] - ETA: 20s - loss: 7.6459 - accuracy: 0.5014
17024/25000 [===================>..........] - ETA: 20s - loss: 7.6432 - accuracy: 0.5015
17056/25000 [===================>..........] - ETA: 20s - loss: 7.6414 - accuracy: 0.5016
17088/25000 [===================>..........] - ETA: 20s - loss: 7.6433 - accuracy: 0.5015
17120/25000 [===================>..........] - ETA: 19s - loss: 7.6415 - accuracy: 0.5016
17152/25000 [===================>..........] - ETA: 19s - loss: 7.6398 - accuracy: 0.5017
17184/25000 [===================>..........] - ETA: 19s - loss: 7.6425 - accuracy: 0.5016
17216/25000 [===================>..........] - ETA: 19s - loss: 7.6426 - accuracy: 0.5016
17248/25000 [===================>..........] - ETA: 19s - loss: 7.6399 - accuracy: 0.5017
17280/25000 [===================>..........] - ETA: 19s - loss: 7.6409 - accuracy: 0.5017
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6374 - accuracy: 0.5019
17344/25000 [===================>..........] - ETA: 19s - loss: 7.6427 - accuracy: 0.5016
17376/25000 [===================>..........] - ETA: 19s - loss: 7.6419 - accuracy: 0.5016
17408/25000 [===================>..........] - ETA: 19s - loss: 7.6428 - accuracy: 0.5016
17440/25000 [===================>..........] - ETA: 19s - loss: 7.6455 - accuracy: 0.5014
17472/25000 [===================>..........] - ETA: 19s - loss: 7.6403 - accuracy: 0.5017
17504/25000 [====================>.........] - ETA: 18s - loss: 7.6403 - accuracy: 0.5017
17536/25000 [====================>.........] - ETA: 18s - loss: 7.6439 - accuracy: 0.5015
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6465 - accuracy: 0.5013
17600/25000 [====================>.........] - ETA: 18s - loss: 7.6501 - accuracy: 0.5011
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6510 - accuracy: 0.5010
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6484 - accuracy: 0.5012
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6510 - accuracy: 0.5010
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6528 - accuracy: 0.5009
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6537 - accuracy: 0.5008
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6537 - accuracy: 0.5008
17824/25000 [====================>.........] - ETA: 18s - loss: 7.6546 - accuracy: 0.5008
17856/25000 [====================>.........] - ETA: 18s - loss: 7.6555 - accuracy: 0.5007
17888/25000 [====================>.........] - ETA: 17s - loss: 7.6572 - accuracy: 0.5006
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6564 - accuracy: 0.5007
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6581 - accuracy: 0.5006
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6564 - accuracy: 0.5007
18016/25000 [====================>.........] - ETA: 17s - loss: 7.6564 - accuracy: 0.5007
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6590 - accuracy: 0.5005
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6598 - accuracy: 0.5004
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6573 - accuracy: 0.5006
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6565 - accuracy: 0.5007
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6573 - accuracy: 0.5006
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6607 - accuracy: 0.5004
18240/25000 [====================>.........] - ETA: 17s - loss: 7.6633 - accuracy: 0.5002
18272/25000 [====================>.........] - ETA: 16s - loss: 7.6649 - accuracy: 0.5001
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6666 - accuracy: 0.5000
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6716 - accuracy: 0.4997
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6725 - accuracy: 0.4996
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6733 - accuracy: 0.4996
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6708 - accuracy: 0.4997
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6724 - accuracy: 0.4996
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6733 - accuracy: 0.4996
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6732 - accuracy: 0.4996
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6732 - accuracy: 0.4996
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6691 - accuracy: 0.4998
18624/25000 [=====================>........] - ETA: 16s - loss: 7.6683 - accuracy: 0.4999
18656/25000 [=====================>........] - ETA: 16s - loss: 7.6666 - accuracy: 0.5000
18688/25000 [=====================>........] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6617 - accuracy: 0.5003
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6584 - accuracy: 0.5005
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6593 - accuracy: 0.5005
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6617 - accuracy: 0.5003
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6642 - accuracy: 0.5002
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6634 - accuracy: 0.5002
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6626 - accuracy: 0.5003
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6650 - accuracy: 0.5001
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6674 - accuracy: 0.4999
19040/25000 [=====================>........] - ETA: 15s - loss: 7.6723 - accuracy: 0.4996
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6731 - accuracy: 0.4996
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6714 - accuracy: 0.4997
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6714 - accuracy: 0.4997
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6738 - accuracy: 0.4995
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6722 - accuracy: 0.4996
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6746 - accuracy: 0.4995
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6770 - accuracy: 0.4993
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6785 - accuracy: 0.4992
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6761 - accuracy: 0.4994
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6809 - accuracy: 0.4991
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6864 - accuracy: 0.4987
19424/25000 [======================>.......] - ETA: 14s - loss: 7.6903 - accuracy: 0.4985
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6926 - accuracy: 0.4983
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6957 - accuracy: 0.4981
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6949 - accuracy: 0.4982
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6925 - accuracy: 0.4983
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6917 - accuracy: 0.4984
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6924 - accuracy: 0.4983
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6908 - accuracy: 0.4984
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6908 - accuracy: 0.4984
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6907 - accuracy: 0.4984
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6915 - accuracy: 0.4984
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6945 - accuracy: 0.4982
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6945 - accuracy: 0.4982
19840/25000 [======================>.......] - ETA: 13s - loss: 7.6968 - accuracy: 0.4980
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6952 - accuracy: 0.4981
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6944 - accuracy: 0.4982
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6943 - accuracy: 0.4982
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6950 - accuracy: 0.4981
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6912 - accuracy: 0.4984
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6896 - accuracy: 0.4985
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6903 - accuracy: 0.4985
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6880 - accuracy: 0.4986
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6857 - accuracy: 0.4988
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6834 - accuracy: 0.4989
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6848 - accuracy: 0.4988
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6871 - accuracy: 0.4987
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6886 - accuracy: 0.4986
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6900 - accuracy: 0.4985
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6893 - accuracy: 0.4985
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6907 - accuracy: 0.4984
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6892 - accuracy: 0.4985
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6922 - accuracy: 0.4983
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6929 - accuracy: 0.4983
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6906 - accuracy: 0.4984
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6920 - accuracy: 0.4983
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6853 - accuracy: 0.4988
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6875 - accuracy: 0.4986
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6875 - accuracy: 0.4986
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6889 - accuracy: 0.4985
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6889 - accuracy: 0.4985
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6888 - accuracy: 0.4986
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6918 - accuracy: 0.4984
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6917 - accuracy: 0.4984
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6887 - accuracy: 0.4986
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6887 - accuracy: 0.4986
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6909 - accuracy: 0.4984
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6916 - accuracy: 0.4984
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6930 - accuracy: 0.4983
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6900 - accuracy: 0.4985
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6915 - accuracy: 0.4984
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6921 - accuracy: 0.4983
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6899 - accuracy: 0.4985 
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6906 - accuracy: 0.4984
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6869 - accuracy: 0.4987
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6876 - accuracy: 0.4986
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6876 - accuracy: 0.4986
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6912 - accuracy: 0.4984
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6904 - accuracy: 0.4984
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6897 - accuracy: 0.4985
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6904 - accuracy: 0.4985
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6910 - accuracy: 0.4984
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6939 - accuracy: 0.4982
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6917 - accuracy: 0.4984
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6931 - accuracy: 0.4983
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6916 - accuracy: 0.4984
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6887 - accuracy: 0.4986
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6880 - accuracy: 0.4986
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6872 - accuracy: 0.4987
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6865 - accuracy: 0.4987
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6843 - accuracy: 0.4988
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6843 - accuracy: 0.4988
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6850 - accuracy: 0.4988
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6843 - accuracy: 0.4988
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6828 - accuracy: 0.4989
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6814 - accuracy: 0.4990
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6821 - accuracy: 0.4990
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6792 - accuracy: 0.4992
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6834 - accuracy: 0.4989
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6834 - accuracy: 0.4989
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6876 - accuracy: 0.4986
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6855 - accuracy: 0.4988
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6868 - accuracy: 0.4987
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6840 - accuracy: 0.4989
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6854 - accuracy: 0.4988
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6853 - accuracy: 0.4988
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6832 - accuracy: 0.4989
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6811 - accuracy: 0.4991
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6825 - accuracy: 0.4990
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6832 - accuracy: 0.4989
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6825 - accuracy: 0.4990
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6797 - accuracy: 0.4991
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6783 - accuracy: 0.4992
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6755 - accuracy: 0.4994
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6755 - accuracy: 0.4994
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6741 - accuracy: 0.4995
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6721 - accuracy: 0.4996
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6728 - accuracy: 0.4996
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6727 - accuracy: 0.4996
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6693 - accuracy: 0.4998
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6693 - accuracy: 0.4998
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6687 - accuracy: 0.4999
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6707 - accuracy: 0.4997
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6707 - accuracy: 0.4997
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6734 - accuracy: 0.4996
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6727 - accuracy: 0.4996
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6733 - accuracy: 0.4996
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6733 - accuracy: 0.4996
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6720 - accuracy: 0.4996
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6706 - accuracy: 0.4997
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6706 - accuracy: 0.4997
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6680 - accuracy: 0.4999
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6686 - accuracy: 0.4999
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6646 - accuracy: 0.5001
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6693 - accuracy: 0.4998
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6706 - accuracy: 0.4997
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6725 - accuracy: 0.4996
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6725 - accuracy: 0.4996
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6686 - accuracy: 0.4999
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6686 - accuracy: 0.4999
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6614 - accuracy: 0.5003
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6634 - accuracy: 0.5002
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6640 - accuracy: 0.5002
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6660 - accuracy: 0.5000
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6686 - accuracy: 0.4999
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6640 - accuracy: 0.5002
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6653 - accuracy: 0.5001
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24192/25000 [============================>.] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24224/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24256/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24288/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24320/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24352/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24384/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24416/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24448/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24480/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24512/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24544/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
24576/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24608/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24640/25000 [============================>.] - ETA: 0s - loss: 7.6728 - accuracy: 0.4996
24672/25000 [============================>.] - ETA: 0s - loss: 7.6735 - accuracy: 0.4996
24704/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24736/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24768/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24800/25000 [============================>.] - ETA: 0s - loss: 7.6728 - accuracy: 0.4996
24832/25000 [============================>.] - ETA: 0s - loss: 7.6734 - accuracy: 0.4996
24864/25000 [============================>.] - ETA: 0s - loss: 7.6709 - accuracy: 0.4997
24896/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 73s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f6ca1f97b00> 

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
 [-0.00240272 -0.03540706 -0.03589001  0.01130874 -0.05868539 -0.01373739]
 [ 0.28615496 -0.1448949  -0.24890229  0.18262893 -0.11493639  0.20618913]
 [ 0.00319698 -0.23459727  0.10324322  0.17797814 -0.05098395  0.08664047]
 [-0.08320297  0.1536091   0.07026839 -0.20511287  0.17964476  0.08363621]
 [-0.04555265  0.22149804 -0.09640612  0.04868883  0.30849972  0.30636209]
 [ 0.22251612 -0.20069772  0.15872997  0.26027885 -0.33032343 -0.22165041]
 [ 0.02323676  0.34260213 -0.14194696  0.23076916 -0.47379181 -0.01658661]
 [-0.05217695 -0.2956484  -0.04114044 -0.31006902 -0.1453812  -0.32213578]
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
{'loss': 0.5513646006584167, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-11 15:15:34.536105: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.49388454481959343, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-11 15:15:35.661896: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:49<01:14, 24.73s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.05362596402819182, 'embedding_size_factor': 0.9889371319887312, 'layers.choice': 0, 'learning_rate': 0.0004194954200958002, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 2.3197087482476492e-11} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xabt\xdc\xc3oc\x8aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xef\xa5_{\xf6!\xe2X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?;}\xf7\x1c1\xb15X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xb9\x81fP\x0bq&u.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xabt\xdc\xc3oc\x8aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xef\xa5_{\xf6!\xe2X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?;}\xf7\x1c1\xb15X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xb9\x81fP\x0bq&u.' and reward: 0.3862
 60%|██████    | 3/5 [01:38<01:04, 32.13s/it] 60%|██████    | 3/5 [01:38<01:05, 32.95s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.2975941749187023, 'embedding_size_factor': 0.9103248377548802, 'layers.choice': 0, 'learning_rate': 0.0005114237802545553, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.04420270308531576} and reward: 0.3814
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd3\x0b\xc8p0a(X\x15\x00\x00\x00embedding_size_factorq\x03G?\xed!a\x8d\xdc\x99SX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@\xc2"4\x8e6\x9bX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\xa6\xa1\xbc\x98J\xde\x97u.' and reward: 0.3814
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd3\x0b\xc8p0a(X\x15\x00\x00\x00embedding_size_factorq\x03G?\xed!a\x8d\xdc\x99SX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@\xc2"4\x8e6\x9bX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\xa6\xa1\xbc\x98J\xde\x97u.' and reward: 0.3814
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 149.10772609710693
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -31.65s of remaining time.
Ensemble size: 76
Ensemble weights: 
[0.48684211 0.13157895 0.38157895]
	0.3942	 = Validation accuracy score
	1.06s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 152.75s ...
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

