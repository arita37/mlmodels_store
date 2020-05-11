
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/390cac7ee9f7623cf7fa6990086c5bab3f62fb80', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '390cac7ee9f7623cf7fa6990086c5bab3f62fb80', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/390cac7ee9f7623cf7fa6990086c5bab3f62fb80

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/390cac7ee9f7623cf7fa6990086c5bab3f62fb80

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

    8192/17464789 [..............................] - ETA: 12s
 3686400/17464789 [=====>........................] - ETA: 0s 
12009472/17464789 [===================>..........] - ETA: 0s
17022976/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 03:17:04.416515: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 03:17:04.420791: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-11 03:17:04.420995: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e76b3f6a70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 03:17:04.421011: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:49 - loss: 5.7500 - accuracy: 0.6250
   64/25000 [..............................] - ETA: 3:12 - loss: 6.4687 - accuracy: 0.5781
   96/25000 [..............................] - ETA: 2:37 - loss: 5.9097 - accuracy: 0.6146
  128/25000 [..............................] - ETA: 2:21 - loss: 5.8698 - accuracy: 0.6172
  160/25000 [..............................] - ETA: 2:10 - loss: 6.9000 - accuracy: 0.5500
  192/25000 [..............................] - ETA: 2:03 - loss: 7.1875 - accuracy: 0.5312
  224/25000 [..............................] - ETA: 1:57 - loss: 7.3244 - accuracy: 0.5223
  256/25000 [..............................] - ETA: 1:53 - loss: 7.3072 - accuracy: 0.5234
  288/25000 [..............................] - ETA: 1:50 - loss: 7.3472 - accuracy: 0.5208
  320/25000 [..............................] - ETA: 1:48 - loss: 7.3312 - accuracy: 0.5219
  352/25000 [..............................] - ETA: 1:48 - loss: 7.3181 - accuracy: 0.5227
  384/25000 [..............................] - ETA: 1:46 - loss: 7.3072 - accuracy: 0.5234
  416/25000 [..............................] - ETA: 1:45 - loss: 7.4455 - accuracy: 0.5144
  448/25000 [..............................] - ETA: 1:43 - loss: 7.3928 - accuracy: 0.5179
  480/25000 [..............................] - ETA: 1:42 - loss: 7.3472 - accuracy: 0.5208
  512/25000 [..............................] - ETA: 1:41 - loss: 7.5468 - accuracy: 0.5078
  544/25000 [..............................] - ETA: 1:40 - loss: 7.5821 - accuracy: 0.5055
  576/25000 [..............................] - ETA: 1:39 - loss: 7.4537 - accuracy: 0.5139
  608/25000 [..............................] - ETA: 1:38 - loss: 7.5910 - accuracy: 0.5049
  640/25000 [..............................] - ETA: 1:37 - loss: 7.6187 - accuracy: 0.5031
  672/25000 [..............................] - ETA: 1:36 - loss: 7.4841 - accuracy: 0.5119
  704/25000 [..............................] - ETA: 1:36 - loss: 7.5577 - accuracy: 0.5071
  736/25000 [..............................] - ETA: 1:35 - loss: 7.5208 - accuracy: 0.5095
  768/25000 [..............................] - ETA: 1:35 - loss: 7.4869 - accuracy: 0.5117
  800/25000 [..............................] - ETA: 1:35 - loss: 7.5325 - accuracy: 0.5088
  832/25000 [..............................] - ETA: 1:34 - loss: 7.4455 - accuracy: 0.5144
  864/25000 [>.............................] - ETA: 1:34 - loss: 7.4714 - accuracy: 0.5127
  896/25000 [>.............................] - ETA: 1:33 - loss: 7.5126 - accuracy: 0.5100
  928/25000 [>.............................] - ETA: 1:33 - loss: 7.5510 - accuracy: 0.5075
  960/25000 [>.............................] - ETA: 1:32 - loss: 7.5069 - accuracy: 0.5104
  992/25000 [>.............................] - ETA: 1:32 - loss: 7.5893 - accuracy: 0.5050
 1024/25000 [>.............................] - ETA: 1:32 - loss: 7.6516 - accuracy: 0.5010
 1056/25000 [>.............................] - ETA: 1:31 - loss: 7.5940 - accuracy: 0.5047
 1088/25000 [>.............................] - ETA: 1:31 - loss: 7.5398 - accuracy: 0.5083
 1120/25000 [>.............................] - ETA: 1:31 - loss: 7.5571 - accuracy: 0.5071
 1152/25000 [>.............................] - ETA: 1:30 - loss: 7.6001 - accuracy: 0.5043
 1184/25000 [>.............................] - ETA: 1:30 - loss: 7.6019 - accuracy: 0.5042
 1216/25000 [>.............................] - ETA: 1:30 - loss: 7.6666 - accuracy: 0.5000
 1248/25000 [>.............................] - ETA: 1:30 - loss: 7.6175 - accuracy: 0.5032
 1280/25000 [>.............................] - ETA: 1:30 - loss: 7.6187 - accuracy: 0.5031
 1312/25000 [>.............................] - ETA: 1:30 - loss: 7.6666 - accuracy: 0.5000
 1344/25000 [>.............................] - ETA: 1:29 - loss: 7.6780 - accuracy: 0.4993
 1376/25000 [>.............................] - ETA: 1:29 - loss: 7.6555 - accuracy: 0.5007
 1408/25000 [>.............................] - ETA: 1:29 - loss: 7.6122 - accuracy: 0.5036
 1440/25000 [>.............................] - ETA: 1:29 - loss: 7.6773 - accuracy: 0.4993
 1472/25000 [>.............................] - ETA: 1:28 - loss: 7.6770 - accuracy: 0.4993
 1504/25000 [>.............................] - ETA: 1:28 - loss: 7.6564 - accuracy: 0.5007
 1536/25000 [>.............................] - ETA: 1:28 - loss: 7.6267 - accuracy: 0.5026
 1568/25000 [>.............................] - ETA: 1:28 - loss: 7.5884 - accuracy: 0.5051
 1600/25000 [>.............................] - ETA: 1:27 - loss: 7.5900 - accuracy: 0.5050
 1632/25000 [>.............................] - ETA: 1:27 - loss: 7.5821 - accuracy: 0.5055
 1664/25000 [>.............................] - ETA: 1:27 - loss: 7.5837 - accuracy: 0.5054
 1696/25000 [=>............................] - ETA: 1:27 - loss: 7.4948 - accuracy: 0.5112
 1728/25000 [=>............................] - ETA: 1:27 - loss: 7.4892 - accuracy: 0.5116
 1760/25000 [=>............................] - ETA: 1:26 - loss: 7.4837 - accuracy: 0.5119
 1792/25000 [=>............................] - ETA: 1:26 - loss: 7.4784 - accuracy: 0.5123
 1824/25000 [=>............................] - ETA: 1:26 - loss: 7.4817 - accuracy: 0.5121
 1856/25000 [=>............................] - ETA: 1:26 - loss: 7.4931 - accuracy: 0.5113
 1888/25000 [=>............................] - ETA: 1:26 - loss: 7.5042 - accuracy: 0.5106
 1920/25000 [=>............................] - ETA: 1:26 - loss: 7.5309 - accuracy: 0.5089
 1952/25000 [=>............................] - ETA: 1:26 - loss: 7.5174 - accuracy: 0.5097
 1984/25000 [=>............................] - ETA: 1:25 - loss: 7.5275 - accuracy: 0.5091
 2016/25000 [=>............................] - ETA: 1:25 - loss: 7.5297 - accuracy: 0.5089
 2048/25000 [=>............................] - ETA: 1:25 - loss: 7.5094 - accuracy: 0.5103
 2080/25000 [=>............................] - ETA: 1:25 - loss: 7.5044 - accuracy: 0.5106
 2112/25000 [=>............................] - ETA: 1:25 - loss: 7.4924 - accuracy: 0.5114
 2144/25000 [=>............................] - ETA: 1:25 - loss: 7.4950 - accuracy: 0.5112
 2176/25000 [=>............................] - ETA: 1:24 - loss: 7.5045 - accuracy: 0.5106
 2208/25000 [=>............................] - ETA: 1:24 - loss: 7.5208 - accuracy: 0.5095
 2240/25000 [=>............................] - ETA: 1:24 - loss: 7.5160 - accuracy: 0.5098
 2272/25000 [=>............................] - ETA: 1:24 - loss: 7.5316 - accuracy: 0.5088
 2304/25000 [=>............................] - ETA: 1:24 - loss: 7.5335 - accuracy: 0.5087
 2336/25000 [=>............................] - ETA: 1:23 - loss: 7.5550 - accuracy: 0.5073
 2368/25000 [=>............................] - ETA: 1:23 - loss: 7.5565 - accuracy: 0.5072
 2400/25000 [=>............................] - ETA: 1:23 - loss: 7.5388 - accuracy: 0.5083
 2432/25000 [=>............................] - ETA: 1:23 - loss: 7.5342 - accuracy: 0.5086
 2464/25000 [=>............................] - ETA: 1:23 - loss: 7.5608 - accuracy: 0.5069
 2496/25000 [=>............................] - ETA: 1:23 - loss: 7.5745 - accuracy: 0.5060
 2528/25000 [==>...........................] - ETA: 1:23 - loss: 7.6242 - accuracy: 0.5028
 2560/25000 [==>...........................] - ETA: 1:22 - loss: 7.6247 - accuracy: 0.5027
 2592/25000 [==>...........................] - ETA: 1:22 - loss: 7.6430 - accuracy: 0.5015
 2624/25000 [==>...........................] - ETA: 1:22 - loss: 7.6316 - accuracy: 0.5023
 2656/25000 [==>...........................] - ETA: 1:22 - loss: 7.6378 - accuracy: 0.5019
 2688/25000 [==>...........................] - ETA: 1:22 - loss: 7.6153 - accuracy: 0.5033
 2720/25000 [==>...........................] - ETA: 1:21 - loss: 7.6384 - accuracy: 0.5018
 2752/25000 [==>...........................] - ETA: 1:21 - loss: 7.6555 - accuracy: 0.5007
 2784/25000 [==>...........................] - ETA: 1:21 - loss: 7.6666 - accuracy: 0.5000
 2816/25000 [==>...........................] - ETA: 1:21 - loss: 7.6231 - accuracy: 0.5028
 2848/25000 [==>...........................] - ETA: 1:21 - loss: 7.6020 - accuracy: 0.5042
 2880/25000 [==>...........................] - ETA: 1:21 - loss: 7.5974 - accuracy: 0.5045
 2912/25000 [==>...........................] - ETA: 1:21 - loss: 7.5824 - accuracy: 0.5055
 2944/25000 [==>...........................] - ETA: 1:20 - loss: 7.5729 - accuracy: 0.5061
 2976/25000 [==>...........................] - ETA: 1:20 - loss: 7.5996 - accuracy: 0.5044
 3008/25000 [==>...........................] - ETA: 1:20 - loss: 7.6105 - accuracy: 0.5037
 3040/25000 [==>...........................] - ETA: 1:20 - loss: 7.6162 - accuracy: 0.5033
 3072/25000 [==>...........................] - ETA: 1:20 - loss: 7.6167 - accuracy: 0.5033
 3104/25000 [==>...........................] - ETA: 1:20 - loss: 7.5876 - accuracy: 0.5052
 3136/25000 [==>...........................] - ETA: 1:19 - loss: 7.6031 - accuracy: 0.5041
 3168/25000 [==>...........................] - ETA: 1:19 - loss: 7.5989 - accuracy: 0.5044
 3200/25000 [==>...........................] - ETA: 1:19 - loss: 7.6139 - accuracy: 0.5034
 3232/25000 [==>...........................] - ETA: 1:19 - loss: 7.6524 - accuracy: 0.5009
 3264/25000 [==>...........................] - ETA: 1:19 - loss: 7.6478 - accuracy: 0.5012
 3296/25000 [==>...........................] - ETA: 1:19 - loss: 7.6341 - accuracy: 0.5021
 3328/25000 [==>...........................] - ETA: 1:19 - loss: 7.6344 - accuracy: 0.5021
 3360/25000 [===>..........................] - ETA: 1:18 - loss: 7.6392 - accuracy: 0.5018
 3392/25000 [===>..........................] - ETA: 1:18 - loss: 7.6259 - accuracy: 0.5027
 3424/25000 [===>..........................] - ETA: 1:18 - loss: 7.6218 - accuracy: 0.5029
 3456/25000 [===>..........................] - ETA: 1:18 - loss: 7.6356 - accuracy: 0.5020
 3488/25000 [===>..........................] - ETA: 1:18 - loss: 7.6271 - accuracy: 0.5026
 3520/25000 [===>..........................] - ETA: 1:18 - loss: 7.6100 - accuracy: 0.5037
 3552/25000 [===>..........................] - ETA: 1:17 - loss: 7.6407 - accuracy: 0.5017
 3584/25000 [===>..........................] - ETA: 1:17 - loss: 7.6581 - accuracy: 0.5006
 3616/25000 [===>..........................] - ETA: 1:17 - loss: 7.6581 - accuracy: 0.5006
 3648/25000 [===>..........................] - ETA: 1:17 - loss: 7.6666 - accuracy: 0.5000
 3680/25000 [===>..........................] - ETA: 1:17 - loss: 7.6708 - accuracy: 0.4997
 3712/25000 [===>..........................] - ETA: 1:17 - loss: 7.6501 - accuracy: 0.5011
 3744/25000 [===>..........................] - ETA: 1:17 - loss: 7.6175 - accuracy: 0.5032
 3776/25000 [===>..........................] - ETA: 1:17 - loss: 7.6301 - accuracy: 0.5024
 3808/25000 [===>..........................] - ETA: 1:16 - loss: 7.6505 - accuracy: 0.5011
 3840/25000 [===>..........................] - ETA: 1:16 - loss: 7.6467 - accuracy: 0.5013
 3872/25000 [===>..........................] - ETA: 1:16 - loss: 7.6468 - accuracy: 0.5013
 3904/25000 [===>..........................] - ETA: 1:16 - loss: 7.6588 - accuracy: 0.5005
 3936/25000 [===>..........................] - ETA: 1:16 - loss: 7.6510 - accuracy: 0.5010
 3968/25000 [===>..........................] - ETA: 1:16 - loss: 7.6589 - accuracy: 0.5005
 4000/25000 [===>..........................] - ETA: 1:16 - loss: 7.6666 - accuracy: 0.5000
 4032/25000 [===>..........................] - ETA: 1:15 - loss: 7.6666 - accuracy: 0.5000
 4064/25000 [===>..........................] - ETA: 1:15 - loss: 7.6855 - accuracy: 0.4988
 4096/25000 [===>..........................] - ETA: 1:15 - loss: 7.6928 - accuracy: 0.4983
 4128/25000 [===>..........................] - ETA: 1:15 - loss: 7.7112 - accuracy: 0.4971
 4160/25000 [===>..........................] - ETA: 1:15 - loss: 7.7145 - accuracy: 0.4969
 4192/25000 [====>.........................] - ETA: 1:15 - loss: 7.7215 - accuracy: 0.4964
 4224/25000 [====>.........................] - ETA: 1:15 - loss: 7.7174 - accuracy: 0.4967
 4256/25000 [====>.........................] - ETA: 1:15 - loss: 7.7279 - accuracy: 0.4960
 4288/25000 [====>.........................] - ETA: 1:15 - loss: 7.7381 - accuracy: 0.4953
 4320/25000 [====>.........................] - ETA: 1:14 - loss: 7.7625 - accuracy: 0.4938
 4352/25000 [====>.........................] - ETA: 1:14 - loss: 7.7794 - accuracy: 0.4926
 4384/25000 [====>.........................] - ETA: 1:14 - loss: 7.7750 - accuracy: 0.4929
 4416/25000 [====>.........................] - ETA: 1:14 - loss: 7.7708 - accuracy: 0.4932
 4448/25000 [====>.........................] - ETA: 1:14 - loss: 7.7700 - accuracy: 0.4933
 4480/25000 [====>.........................] - ETA: 1:14 - loss: 7.7693 - accuracy: 0.4933
 4512/25000 [====>.........................] - ETA: 1:14 - loss: 7.7550 - accuracy: 0.4942
 4544/25000 [====>.........................] - ETA: 1:13 - loss: 7.7476 - accuracy: 0.4947
 4576/25000 [====>.........................] - ETA: 1:13 - loss: 7.7470 - accuracy: 0.4948
 4608/25000 [====>.........................] - ETA: 1:13 - loss: 7.7531 - accuracy: 0.4944
 4640/25000 [====>.........................] - ETA: 1:13 - loss: 7.7691 - accuracy: 0.4933
 4672/25000 [====>.........................] - ETA: 1:13 - loss: 7.7749 - accuracy: 0.4929
 4704/25000 [====>.........................] - ETA: 1:13 - loss: 7.7872 - accuracy: 0.4921
 4736/25000 [====>.........................] - ETA: 1:13 - loss: 7.7961 - accuracy: 0.4916
 4768/25000 [====>.........................] - ETA: 1:13 - loss: 7.8274 - accuracy: 0.4895
 4800/25000 [====>.........................] - ETA: 1:12 - loss: 7.8231 - accuracy: 0.4898
 4832/25000 [====>.........................] - ETA: 1:12 - loss: 7.8189 - accuracy: 0.4901
 4864/25000 [====>.........................] - ETA: 1:12 - loss: 7.8148 - accuracy: 0.4903
 4896/25000 [====>.........................] - ETA: 1:12 - loss: 7.8295 - accuracy: 0.4894
 4928/25000 [====>.........................] - ETA: 1:12 - loss: 7.8315 - accuracy: 0.4892
 4960/25000 [====>.........................] - ETA: 1:12 - loss: 7.8336 - accuracy: 0.4891
 4992/25000 [====>.........................] - ETA: 1:12 - loss: 7.8294 - accuracy: 0.4894
 5024/25000 [=====>........................] - ETA: 1:12 - loss: 7.8253 - accuracy: 0.4896
 5056/25000 [=====>........................] - ETA: 1:11 - loss: 7.8304 - accuracy: 0.4893
 5088/25000 [=====>........................] - ETA: 1:11 - loss: 7.8354 - accuracy: 0.4890
 5120/25000 [=====>........................] - ETA: 1:11 - loss: 7.8253 - accuracy: 0.4896
 5152/25000 [=====>........................] - ETA: 1:11 - loss: 7.8244 - accuracy: 0.4897
 5184/25000 [=====>........................] - ETA: 1:11 - loss: 7.8116 - accuracy: 0.4905
 5216/25000 [=====>........................] - ETA: 1:11 - loss: 7.7930 - accuracy: 0.4918
 5248/25000 [=====>........................] - ETA: 1:11 - loss: 7.7893 - accuracy: 0.4920
 5280/25000 [=====>........................] - ETA: 1:10 - loss: 7.8060 - accuracy: 0.4909
 5312/25000 [=====>........................] - ETA: 1:10 - loss: 7.7907 - accuracy: 0.4919
 5344/25000 [=====>........................] - ETA: 1:10 - loss: 7.8101 - accuracy: 0.4906
 5376/25000 [=====>........................] - ETA: 1:10 - loss: 7.8035 - accuracy: 0.4911
 5408/25000 [=====>........................] - ETA: 1:10 - loss: 7.8112 - accuracy: 0.4906
 5440/25000 [=====>........................] - ETA: 1:10 - loss: 7.8245 - accuracy: 0.4897
 5472/25000 [=====>........................] - ETA: 1:10 - loss: 7.8123 - accuracy: 0.4905
 5504/25000 [=====>........................] - ETA: 1:10 - loss: 7.8059 - accuracy: 0.4909
 5536/25000 [=====>........................] - ETA: 1:09 - loss: 7.8051 - accuracy: 0.4910
 5568/25000 [=====>........................] - ETA: 1:09 - loss: 7.8043 - accuracy: 0.4910
 5600/25000 [=====>........................] - ETA: 1:09 - loss: 7.8035 - accuracy: 0.4911
 5632/25000 [=====>........................] - ETA: 1:09 - loss: 7.8082 - accuracy: 0.4908
 5664/25000 [=====>........................] - ETA: 1:09 - loss: 7.7939 - accuracy: 0.4917
 5696/25000 [=====>........................] - ETA: 1:09 - loss: 7.7931 - accuracy: 0.4917
 5728/25000 [=====>........................] - ETA: 1:09 - loss: 7.8085 - accuracy: 0.4907
 5760/25000 [=====>........................] - ETA: 1:09 - loss: 7.8104 - accuracy: 0.4906
 5792/25000 [=====>........................] - ETA: 1:08 - loss: 7.8175 - accuracy: 0.4902
 5824/25000 [=====>........................] - ETA: 1:08 - loss: 7.8272 - accuracy: 0.4895
 5856/25000 [======>.......................] - ETA: 1:08 - loss: 7.8211 - accuracy: 0.4899
 5888/25000 [======>.......................] - ETA: 1:08 - loss: 7.8177 - accuracy: 0.4901
 5920/25000 [======>.......................] - ETA: 1:08 - loss: 7.8091 - accuracy: 0.4907
 5952/25000 [======>.......................] - ETA: 1:08 - loss: 7.7980 - accuracy: 0.4914
 5984/25000 [======>.......................] - ETA: 1:08 - loss: 7.8178 - accuracy: 0.4901
 6016/25000 [======>.......................] - ETA: 1:08 - loss: 7.8170 - accuracy: 0.4902
 6048/25000 [======>.......................] - ETA: 1:08 - loss: 7.8086 - accuracy: 0.4907
 6080/25000 [======>.......................] - ETA: 1:07 - loss: 7.8078 - accuracy: 0.4908
 6112/25000 [======>.......................] - ETA: 1:07 - loss: 7.8046 - accuracy: 0.4910
 6144/25000 [======>.......................] - ETA: 1:07 - loss: 7.8039 - accuracy: 0.4910
 6176/25000 [======>.......................] - ETA: 1:07 - loss: 7.7957 - accuracy: 0.4916
 6208/25000 [======>.......................] - ETA: 1:07 - loss: 7.7827 - accuracy: 0.4924
 6240/25000 [======>.......................] - ETA: 1:07 - loss: 7.7870 - accuracy: 0.4921
 6272/25000 [======>.......................] - ETA: 1:07 - loss: 7.7889 - accuracy: 0.4920
 6304/25000 [======>.......................] - ETA: 1:06 - loss: 7.7809 - accuracy: 0.4925
 6336/25000 [======>.......................] - ETA: 1:06 - loss: 7.7804 - accuracy: 0.4926
 6368/25000 [======>.......................] - ETA: 1:06 - loss: 7.7846 - accuracy: 0.4923
 6400/25000 [======>.......................] - ETA: 1:06 - loss: 7.7864 - accuracy: 0.4922
 6432/25000 [======>.......................] - ETA: 1:06 - loss: 7.7834 - accuracy: 0.4924
 6464/25000 [======>.......................] - ETA: 1:06 - loss: 7.7876 - accuracy: 0.4921
 6496/25000 [======>.......................] - ETA: 1:06 - loss: 7.7799 - accuracy: 0.4926
 6528/25000 [======>.......................] - ETA: 1:06 - loss: 7.7770 - accuracy: 0.4928
 6560/25000 [======>.......................] - ETA: 1:05 - loss: 7.7835 - accuracy: 0.4924
 6592/25000 [======>.......................] - ETA: 1:05 - loss: 7.7666 - accuracy: 0.4935
 6624/25000 [======>.......................] - ETA: 1:05 - loss: 7.7685 - accuracy: 0.4934
 6656/25000 [======>.......................] - ETA: 1:05 - loss: 7.7657 - accuracy: 0.4935
 6688/25000 [=======>......................] - ETA: 1:05 - loss: 7.7721 - accuracy: 0.4931
 6720/25000 [=======>......................] - ETA: 1:05 - loss: 7.7670 - accuracy: 0.4935
 6752/25000 [=======>......................] - ETA: 1:05 - loss: 7.7688 - accuracy: 0.4933
 6784/25000 [=======>......................] - ETA: 1:05 - loss: 7.7842 - accuracy: 0.4923
 6816/25000 [=======>......................] - ETA: 1:05 - loss: 7.7858 - accuracy: 0.4922
 6848/25000 [=======>......................] - ETA: 1:04 - loss: 7.7763 - accuracy: 0.4928
 6880/25000 [=======>......................] - ETA: 1:04 - loss: 7.7803 - accuracy: 0.4926
 6912/25000 [=======>......................] - ETA: 1:04 - loss: 7.7953 - accuracy: 0.4916
 6944/25000 [=======>......................] - ETA: 1:04 - loss: 7.7947 - accuracy: 0.4916
 6976/25000 [=======>......................] - ETA: 1:04 - loss: 7.8051 - accuracy: 0.4910
 7008/25000 [=======>......................] - ETA: 1:04 - loss: 7.7979 - accuracy: 0.4914
 7040/25000 [=======>......................] - ETA: 1:04 - loss: 7.7973 - accuracy: 0.4915
 7072/25000 [=======>......................] - ETA: 1:04 - loss: 7.7924 - accuracy: 0.4918
 7104/25000 [=======>......................] - ETA: 1:03 - loss: 7.7853 - accuracy: 0.4923
 7136/25000 [=======>......................] - ETA: 1:03 - loss: 7.7891 - accuracy: 0.4920
 7168/25000 [=======>......................] - ETA: 1:03 - loss: 7.7928 - accuracy: 0.4918
 7200/25000 [=======>......................] - ETA: 1:03 - loss: 7.8029 - accuracy: 0.4911
 7232/25000 [=======>......................] - ETA: 1:03 - loss: 7.8023 - accuracy: 0.4912
 7264/25000 [=======>......................] - ETA: 1:03 - loss: 7.8059 - accuracy: 0.4909
 7296/25000 [=======>......................] - ETA: 1:03 - loss: 7.8032 - accuracy: 0.4911
 7328/25000 [=======>......................] - ETA: 1:03 - loss: 7.8026 - accuracy: 0.4911
 7360/25000 [=======>......................] - ETA: 1:03 - loss: 7.8062 - accuracy: 0.4909
 7392/25000 [=======>......................] - ETA: 1:02 - loss: 7.8035 - accuracy: 0.4911
 7424/25000 [=======>......................] - ETA: 1:02 - loss: 7.7988 - accuracy: 0.4914
 7456/25000 [=======>......................] - ETA: 1:02 - loss: 7.7941 - accuracy: 0.4917
 7488/25000 [=======>......................] - ETA: 1:02 - loss: 7.7997 - accuracy: 0.4913
 7520/25000 [========>.....................] - ETA: 1:02 - loss: 7.7930 - accuracy: 0.4918
 7552/25000 [========>.....................] - ETA: 1:02 - loss: 7.7905 - accuracy: 0.4919
 7584/25000 [========>.....................] - ETA: 1:02 - loss: 7.7859 - accuracy: 0.4922
 7616/25000 [========>.....................] - ETA: 1:02 - loss: 7.7814 - accuracy: 0.4925
 7648/25000 [========>.....................] - ETA: 1:01 - loss: 7.7729 - accuracy: 0.4931
 7680/25000 [========>.....................] - ETA: 1:01 - loss: 7.7664 - accuracy: 0.4935
 7712/25000 [========>.....................] - ETA: 1:01 - loss: 7.7720 - accuracy: 0.4931
 7744/25000 [========>.....................] - ETA: 1:01 - loss: 7.7696 - accuracy: 0.4933
 7776/25000 [========>.....................] - ETA: 1:01 - loss: 7.7751 - accuracy: 0.4929
 7808/25000 [========>.....................] - ETA: 1:01 - loss: 7.7707 - accuracy: 0.4932
 7840/25000 [========>.....................] - ETA: 1:01 - loss: 7.7761 - accuracy: 0.4929
 7872/25000 [========>.....................] - ETA: 1:01 - loss: 7.7815 - accuracy: 0.4925
 7904/25000 [========>.....................] - ETA: 1:00 - loss: 7.7811 - accuracy: 0.4925
 7936/25000 [========>.....................] - ETA: 1:00 - loss: 7.7787 - accuracy: 0.4927
 7968/25000 [========>.....................] - ETA: 1:00 - loss: 7.7802 - accuracy: 0.4926
 8000/25000 [========>.....................] - ETA: 1:00 - loss: 7.7740 - accuracy: 0.4930
 8032/25000 [========>.....................] - ETA: 1:00 - loss: 7.7716 - accuracy: 0.4932
 8064/25000 [========>.....................] - ETA: 1:00 - loss: 7.7655 - accuracy: 0.4936
 8096/25000 [========>.....................] - ETA: 1:00 - loss: 7.7651 - accuracy: 0.4936
 8128/25000 [========>.....................] - ETA: 1:00 - loss: 7.7647 - accuracy: 0.4936
 8160/25000 [========>.....................] - ETA: 59s - loss: 7.7718 - accuracy: 0.4931 
 8192/25000 [========>.....................] - ETA: 59s - loss: 7.7714 - accuracy: 0.4932
 8224/25000 [========>.....................] - ETA: 59s - loss: 7.7729 - accuracy: 0.4931
 8256/25000 [========>.....................] - ETA: 59s - loss: 7.7651 - accuracy: 0.4936
 8288/25000 [========>.....................] - ETA: 59s - loss: 7.7647 - accuracy: 0.4936
 8320/25000 [========>.....................] - ETA: 59s - loss: 7.7551 - accuracy: 0.4942
 8352/25000 [=========>....................] - ETA: 59s - loss: 7.7602 - accuracy: 0.4939
 8384/25000 [=========>....................] - ETA: 59s - loss: 7.7617 - accuracy: 0.4938
 8416/25000 [=========>....................] - ETA: 58s - loss: 7.7632 - accuracy: 0.4937
 8448/25000 [=========>....................] - ETA: 58s - loss: 7.7610 - accuracy: 0.4938
 8480/25000 [=========>....................] - ETA: 58s - loss: 7.7643 - accuracy: 0.4936
 8512/25000 [=========>....................] - ETA: 58s - loss: 7.7621 - accuracy: 0.4938
 8544/25000 [=========>....................] - ETA: 58s - loss: 7.7599 - accuracy: 0.4939
 8576/25000 [=========>....................] - ETA: 58s - loss: 7.7560 - accuracy: 0.4942
 8608/25000 [=========>....................] - ETA: 58s - loss: 7.7503 - accuracy: 0.4945
 8640/25000 [=========>....................] - ETA: 58s - loss: 7.7465 - accuracy: 0.4948
 8672/25000 [=========>....................] - ETA: 58s - loss: 7.7426 - accuracy: 0.4950
 8704/25000 [=========>....................] - ETA: 57s - loss: 7.7459 - accuracy: 0.4948
 8736/25000 [=========>....................] - ETA: 57s - loss: 7.7403 - accuracy: 0.4952
 8768/25000 [=========>....................] - ETA: 57s - loss: 7.7436 - accuracy: 0.4950
 8800/25000 [=========>....................] - ETA: 57s - loss: 7.7503 - accuracy: 0.4945
 8832/25000 [=========>....................] - ETA: 57s - loss: 7.7500 - accuracy: 0.4946
 8864/25000 [=========>....................] - ETA: 57s - loss: 7.7462 - accuracy: 0.4948
 8896/25000 [=========>....................] - ETA: 57s - loss: 7.7425 - accuracy: 0.4951
 8928/25000 [=========>....................] - ETA: 57s - loss: 7.7508 - accuracy: 0.4945
 8960/25000 [=========>....................] - ETA: 56s - loss: 7.7505 - accuracy: 0.4945
 8992/25000 [=========>....................] - ETA: 56s - loss: 7.7502 - accuracy: 0.4946
 9024/25000 [=========>....................] - ETA: 56s - loss: 7.7448 - accuracy: 0.4949
 9056/25000 [=========>....................] - ETA: 56s - loss: 7.7394 - accuracy: 0.4953
 9088/25000 [=========>....................] - ETA: 56s - loss: 7.7375 - accuracy: 0.4954
 9120/25000 [=========>....................] - ETA: 56s - loss: 7.7423 - accuracy: 0.4951
 9152/25000 [=========>....................] - ETA: 56s - loss: 7.7403 - accuracy: 0.4952
 9184/25000 [==========>...................] - ETA: 56s - loss: 7.7384 - accuracy: 0.4953
 9216/25000 [==========>...................] - ETA: 56s - loss: 7.7365 - accuracy: 0.4954
 9248/25000 [==========>...................] - ETA: 55s - loss: 7.7346 - accuracy: 0.4956
 9280/25000 [==========>...................] - ETA: 55s - loss: 7.7311 - accuracy: 0.4958
 9312/25000 [==========>...................] - ETA: 55s - loss: 7.7341 - accuracy: 0.4956
 9344/25000 [==========>...................] - ETA: 55s - loss: 7.7273 - accuracy: 0.4960
 9376/25000 [==========>...................] - ETA: 55s - loss: 7.7304 - accuracy: 0.4958
 9408/25000 [==========>...................] - ETA: 55s - loss: 7.7220 - accuracy: 0.4964
 9440/25000 [==========>...................] - ETA: 55s - loss: 7.7218 - accuracy: 0.4964
 9472/25000 [==========>...................] - ETA: 55s - loss: 7.7184 - accuracy: 0.4966
 9504/25000 [==========>...................] - ETA: 55s - loss: 7.7166 - accuracy: 0.4967
 9536/25000 [==========>...................] - ETA: 54s - loss: 7.7116 - accuracy: 0.4971
 9568/25000 [==========>...................] - ETA: 54s - loss: 7.7099 - accuracy: 0.4972
 9600/25000 [==========>...................] - ETA: 54s - loss: 7.7034 - accuracy: 0.4976
 9632/25000 [==========>...................] - ETA: 54s - loss: 7.6969 - accuracy: 0.4980
 9664/25000 [==========>...................] - ETA: 54s - loss: 7.7031 - accuracy: 0.4976
 9696/25000 [==========>...................] - ETA: 54s - loss: 7.7046 - accuracy: 0.4975
 9728/25000 [==========>...................] - ETA: 54s - loss: 7.7076 - accuracy: 0.4973
 9760/25000 [==========>...................] - ETA: 54s - loss: 7.7075 - accuracy: 0.4973
 9792/25000 [==========>...................] - ETA: 53s - loss: 7.6995 - accuracy: 0.4979
 9824/25000 [==========>...................] - ETA: 53s - loss: 7.6978 - accuracy: 0.4980
 9856/25000 [==========>...................] - ETA: 53s - loss: 7.6946 - accuracy: 0.4982
 9888/25000 [==========>...................] - ETA: 53s - loss: 7.6868 - accuracy: 0.4987
 9920/25000 [==========>...................] - ETA: 53s - loss: 7.6944 - accuracy: 0.4982
 9952/25000 [==========>...................] - ETA: 53s - loss: 7.6851 - accuracy: 0.4988
 9984/25000 [==========>...................] - ETA: 53s - loss: 7.6881 - accuracy: 0.4986
10016/25000 [===========>..................] - ETA: 53s - loss: 7.6865 - accuracy: 0.4987
10048/25000 [===========>..................] - ETA: 53s - loss: 7.6880 - accuracy: 0.4986
10080/25000 [===========>..................] - ETA: 52s - loss: 7.6849 - accuracy: 0.4988
10112/25000 [===========>..................] - ETA: 52s - loss: 7.6833 - accuracy: 0.4989
10144/25000 [===========>..................] - ETA: 52s - loss: 7.6832 - accuracy: 0.4989
10176/25000 [===========>..................] - ETA: 52s - loss: 7.6772 - accuracy: 0.4993
10208/25000 [===========>..................] - ETA: 52s - loss: 7.6771 - accuracy: 0.4993
10240/25000 [===========>..................] - ETA: 52s - loss: 7.6741 - accuracy: 0.4995
10272/25000 [===========>..................] - ETA: 52s - loss: 7.6771 - accuracy: 0.4993
10304/25000 [===========>..................] - ETA: 52s - loss: 7.6785 - accuracy: 0.4992
10336/25000 [===========>..................] - ETA: 51s - loss: 7.6829 - accuracy: 0.4989
10368/25000 [===========>..................] - ETA: 51s - loss: 7.6755 - accuracy: 0.4994
10400/25000 [===========>..................] - ETA: 51s - loss: 7.6769 - accuracy: 0.4993
10432/25000 [===========>..................] - ETA: 51s - loss: 7.6769 - accuracy: 0.4993
10464/25000 [===========>..................] - ETA: 51s - loss: 7.6813 - accuracy: 0.4990
10496/25000 [===========>..................] - ETA: 51s - loss: 7.6783 - accuracy: 0.4992
10528/25000 [===========>..................] - ETA: 51s - loss: 7.6826 - accuracy: 0.4990
10560/25000 [===========>..................] - ETA: 51s - loss: 7.6884 - accuracy: 0.4986
10592/25000 [===========>..................] - ETA: 51s - loss: 7.6825 - accuracy: 0.4990
10624/25000 [===========>..................] - ETA: 50s - loss: 7.6767 - accuracy: 0.4993
10656/25000 [===========>..................] - ETA: 50s - loss: 7.6753 - accuracy: 0.4994
10688/25000 [===========>..................] - ETA: 50s - loss: 7.6795 - accuracy: 0.4992
10720/25000 [===========>..................] - ETA: 50s - loss: 7.6766 - accuracy: 0.4993
10752/25000 [===========>..................] - ETA: 50s - loss: 7.6752 - accuracy: 0.4994
10784/25000 [===========>..................] - ETA: 50s - loss: 7.6752 - accuracy: 0.4994
10816/25000 [===========>..................] - ETA: 50s - loss: 7.6751 - accuracy: 0.4994
10848/25000 [============>.................] - ETA: 50s - loss: 7.6751 - accuracy: 0.4994
10880/25000 [============>.................] - ETA: 49s - loss: 7.6793 - accuracy: 0.4992
10912/25000 [============>.................] - ETA: 49s - loss: 7.6751 - accuracy: 0.4995
10944/25000 [============>.................] - ETA: 49s - loss: 7.6652 - accuracy: 0.5001
10976/25000 [============>.................] - ETA: 49s - loss: 7.6666 - accuracy: 0.5000
11008/25000 [============>.................] - ETA: 49s - loss: 7.6624 - accuracy: 0.5003
11040/25000 [============>.................] - ETA: 49s - loss: 7.6611 - accuracy: 0.5004
11072/25000 [============>.................] - ETA: 49s - loss: 7.6611 - accuracy: 0.5004
11104/25000 [============>.................] - ETA: 49s - loss: 7.6680 - accuracy: 0.4999
11136/25000 [============>.................] - ETA: 49s - loss: 7.6611 - accuracy: 0.5004
11168/25000 [============>.................] - ETA: 48s - loss: 7.6584 - accuracy: 0.5005
11200/25000 [============>.................] - ETA: 48s - loss: 7.6570 - accuracy: 0.5006
11232/25000 [============>.................] - ETA: 48s - loss: 7.6530 - accuracy: 0.5009
11264/25000 [============>.................] - ETA: 48s - loss: 7.6530 - accuracy: 0.5009
11296/25000 [============>.................] - ETA: 48s - loss: 7.6476 - accuracy: 0.5012
11328/25000 [============>.................] - ETA: 48s - loss: 7.6490 - accuracy: 0.5011
11360/25000 [============>.................] - ETA: 48s - loss: 7.6437 - accuracy: 0.5015
11392/25000 [============>.................] - ETA: 48s - loss: 7.6478 - accuracy: 0.5012
11424/25000 [============>.................] - ETA: 48s - loss: 7.6492 - accuracy: 0.5011
11456/25000 [============>.................] - ETA: 47s - loss: 7.6465 - accuracy: 0.5013
11488/25000 [============>.................] - ETA: 47s - loss: 7.6466 - accuracy: 0.5013
11520/25000 [============>.................] - ETA: 47s - loss: 7.6480 - accuracy: 0.5012
11552/25000 [============>.................] - ETA: 47s - loss: 7.6441 - accuracy: 0.5015
11584/25000 [============>.................] - ETA: 47s - loss: 7.6375 - accuracy: 0.5019
11616/25000 [============>.................] - ETA: 47s - loss: 7.6402 - accuracy: 0.5017
11648/25000 [============>.................] - ETA: 47s - loss: 7.6390 - accuracy: 0.5018
11680/25000 [=============>................] - ETA: 47s - loss: 7.6443 - accuracy: 0.5015
11712/25000 [=============>................] - ETA: 46s - loss: 7.6431 - accuracy: 0.5015
11744/25000 [=============>................] - ETA: 46s - loss: 7.6405 - accuracy: 0.5017
11776/25000 [=============>................] - ETA: 46s - loss: 7.6393 - accuracy: 0.5018
11808/25000 [=============>................] - ETA: 46s - loss: 7.6406 - accuracy: 0.5017
11840/25000 [=============>................] - ETA: 46s - loss: 7.6381 - accuracy: 0.5019
11872/25000 [=============>................] - ETA: 46s - loss: 7.6343 - accuracy: 0.5021
11904/25000 [=============>................] - ETA: 46s - loss: 7.6331 - accuracy: 0.5022
11936/25000 [=============>................] - ETA: 46s - loss: 7.6409 - accuracy: 0.5017
11968/25000 [=============>................] - ETA: 46s - loss: 7.6372 - accuracy: 0.5019
12000/25000 [=============>................] - ETA: 45s - loss: 7.6411 - accuracy: 0.5017
12032/25000 [=============>................] - ETA: 45s - loss: 7.6399 - accuracy: 0.5017
12064/25000 [=============>................] - ETA: 45s - loss: 7.6412 - accuracy: 0.5017
12096/25000 [=============>................] - ETA: 45s - loss: 7.6387 - accuracy: 0.5018
12128/25000 [=============>................] - ETA: 45s - loss: 7.6300 - accuracy: 0.5024
12160/25000 [=============>................] - ETA: 45s - loss: 7.6275 - accuracy: 0.5025
12192/25000 [=============>................] - ETA: 45s - loss: 7.6264 - accuracy: 0.5026
12224/25000 [=============>................] - ETA: 45s - loss: 7.6252 - accuracy: 0.5027
12256/25000 [=============>................] - ETA: 45s - loss: 7.6278 - accuracy: 0.5025
12288/25000 [=============>................] - ETA: 44s - loss: 7.6342 - accuracy: 0.5021
12320/25000 [=============>................] - ETA: 44s - loss: 7.6293 - accuracy: 0.5024
12352/25000 [=============>................] - ETA: 44s - loss: 7.6219 - accuracy: 0.5029
12384/25000 [=============>................] - ETA: 44s - loss: 7.6245 - accuracy: 0.5027
12416/25000 [=============>................] - ETA: 44s - loss: 7.6246 - accuracy: 0.5027
12448/25000 [=============>................] - ETA: 44s - loss: 7.6186 - accuracy: 0.5031
12480/25000 [=============>................] - ETA: 44s - loss: 7.6162 - accuracy: 0.5033
12512/25000 [==============>...............] - ETA: 44s - loss: 7.6213 - accuracy: 0.5030
12544/25000 [==============>...............] - ETA: 44s - loss: 7.6214 - accuracy: 0.5029
12576/25000 [==============>...............] - ETA: 43s - loss: 7.6264 - accuracy: 0.5026
12608/25000 [==============>...............] - ETA: 43s - loss: 7.6216 - accuracy: 0.5029
12640/25000 [==============>...............] - ETA: 43s - loss: 7.6278 - accuracy: 0.5025
12672/25000 [==============>...............] - ETA: 43s - loss: 7.6291 - accuracy: 0.5024
12704/25000 [==============>...............] - ETA: 43s - loss: 7.6268 - accuracy: 0.5026
12736/25000 [==============>...............] - ETA: 43s - loss: 7.6257 - accuracy: 0.5027
12768/25000 [==============>...............] - ETA: 43s - loss: 7.6198 - accuracy: 0.5031
12800/25000 [==============>...............] - ETA: 43s - loss: 7.6151 - accuracy: 0.5034
12832/25000 [==============>...............] - ETA: 42s - loss: 7.6176 - accuracy: 0.5032
12864/25000 [==============>...............] - ETA: 42s - loss: 7.6177 - accuracy: 0.5032
12896/25000 [==============>...............] - ETA: 42s - loss: 7.6191 - accuracy: 0.5031
12928/25000 [==============>...............] - ETA: 42s - loss: 7.6168 - accuracy: 0.5032
12960/25000 [==============>...............] - ETA: 42s - loss: 7.6157 - accuracy: 0.5033
12992/25000 [==============>...............] - ETA: 42s - loss: 7.6194 - accuracy: 0.5031
13024/25000 [==============>...............] - ETA: 42s - loss: 7.6242 - accuracy: 0.5028
13056/25000 [==============>...............] - ETA: 42s - loss: 7.6255 - accuracy: 0.5027
13088/25000 [==============>...............] - ETA: 42s - loss: 7.6221 - accuracy: 0.5029
13120/25000 [==============>...............] - ETA: 41s - loss: 7.6222 - accuracy: 0.5029
13152/25000 [==============>...............] - ETA: 41s - loss: 7.6316 - accuracy: 0.5023
13184/25000 [==============>...............] - ETA: 41s - loss: 7.6352 - accuracy: 0.5020
13216/25000 [==============>...............] - ETA: 41s - loss: 7.6353 - accuracy: 0.5020
13248/25000 [==============>...............] - ETA: 41s - loss: 7.6331 - accuracy: 0.5022
13280/25000 [==============>...............] - ETA: 41s - loss: 7.6343 - accuracy: 0.5021
13312/25000 [==============>...............] - ETA: 41s - loss: 7.6378 - accuracy: 0.5019
13344/25000 [===============>..............] - ETA: 41s - loss: 7.6356 - accuracy: 0.5020
13376/25000 [===============>..............] - ETA: 40s - loss: 7.6322 - accuracy: 0.5022
13408/25000 [===============>..............] - ETA: 40s - loss: 7.6289 - accuracy: 0.5025
13440/25000 [===============>..............] - ETA: 40s - loss: 7.6324 - accuracy: 0.5022
13472/25000 [===============>..............] - ETA: 40s - loss: 7.6256 - accuracy: 0.5027
13504/25000 [===============>..............] - ETA: 40s - loss: 7.6235 - accuracy: 0.5028
13536/25000 [===============>..............] - ETA: 40s - loss: 7.6270 - accuracy: 0.5026
13568/25000 [===============>..............] - ETA: 40s - loss: 7.6259 - accuracy: 0.5027
13600/25000 [===============>..............] - ETA: 40s - loss: 7.6181 - accuracy: 0.5032
13632/25000 [===============>..............] - ETA: 40s - loss: 7.6183 - accuracy: 0.5032
13664/25000 [===============>..............] - ETA: 39s - loss: 7.6184 - accuracy: 0.5031
13696/25000 [===============>..............] - ETA: 39s - loss: 7.6174 - accuracy: 0.5032
13728/25000 [===============>..............] - ETA: 39s - loss: 7.6186 - accuracy: 0.5031
13760/25000 [===============>..............] - ETA: 39s - loss: 7.6198 - accuracy: 0.5031
13792/25000 [===============>..............] - ETA: 39s - loss: 7.6210 - accuracy: 0.5030
13824/25000 [===============>..............] - ETA: 39s - loss: 7.6211 - accuracy: 0.5030
13856/25000 [===============>..............] - ETA: 39s - loss: 7.6212 - accuracy: 0.5030
13888/25000 [===============>..............] - ETA: 39s - loss: 7.6214 - accuracy: 0.5030
13920/25000 [===============>..............] - ETA: 39s - loss: 7.6204 - accuracy: 0.5030
13952/25000 [===============>..............] - ETA: 38s - loss: 7.6183 - accuracy: 0.5032
13984/25000 [===============>..............] - ETA: 38s - loss: 7.6184 - accuracy: 0.5031
14016/25000 [===============>..............] - ETA: 38s - loss: 7.6152 - accuracy: 0.5034
14048/25000 [===============>..............] - ETA: 38s - loss: 7.6197 - accuracy: 0.5031
14080/25000 [===============>..............] - ETA: 38s - loss: 7.6198 - accuracy: 0.5031
14112/25000 [===============>..............] - ETA: 38s - loss: 7.6188 - accuracy: 0.5031
14144/25000 [===============>..............] - ETA: 38s - loss: 7.6200 - accuracy: 0.5030
14176/25000 [================>.............] - ETA: 38s - loss: 7.6212 - accuracy: 0.5030
14208/25000 [================>.............] - ETA: 38s - loss: 7.6213 - accuracy: 0.5030
14240/25000 [================>.............] - ETA: 37s - loss: 7.6257 - accuracy: 0.5027
14272/25000 [================>.............] - ETA: 37s - loss: 7.6269 - accuracy: 0.5026
14304/25000 [================>.............] - ETA: 37s - loss: 7.6291 - accuracy: 0.5024
14336/25000 [================>.............] - ETA: 37s - loss: 7.6303 - accuracy: 0.5024
14368/25000 [================>.............] - ETA: 37s - loss: 7.6335 - accuracy: 0.5022
14400/25000 [================>.............] - ETA: 37s - loss: 7.6294 - accuracy: 0.5024
14432/25000 [================>.............] - ETA: 37s - loss: 7.6294 - accuracy: 0.5024
14464/25000 [================>.............] - ETA: 37s - loss: 7.6242 - accuracy: 0.5028
14496/25000 [================>.............] - ETA: 36s - loss: 7.6211 - accuracy: 0.5030
14528/25000 [================>.............] - ETA: 36s - loss: 7.6223 - accuracy: 0.5029
14560/25000 [================>.............] - ETA: 36s - loss: 7.6203 - accuracy: 0.5030
14592/25000 [================>.............] - ETA: 36s - loss: 7.6162 - accuracy: 0.5033
14624/25000 [================>.............] - ETA: 36s - loss: 7.6163 - accuracy: 0.5033
14656/25000 [================>.............] - ETA: 36s - loss: 7.6112 - accuracy: 0.5036
14688/25000 [================>.............] - ETA: 36s - loss: 7.6113 - accuracy: 0.5036
14720/25000 [================>.............] - ETA: 36s - loss: 7.6145 - accuracy: 0.5034
14752/25000 [================>.............] - ETA: 36s - loss: 7.6178 - accuracy: 0.5032
14784/25000 [================>.............] - ETA: 35s - loss: 7.6199 - accuracy: 0.5030
14816/25000 [================>.............] - ETA: 35s - loss: 7.6221 - accuracy: 0.5029
14848/25000 [================>.............] - ETA: 35s - loss: 7.6263 - accuracy: 0.5026
14880/25000 [================>.............] - ETA: 35s - loss: 7.6254 - accuracy: 0.5027
14912/25000 [================>.............] - ETA: 35s - loss: 7.6245 - accuracy: 0.5027
14944/25000 [================>.............] - ETA: 35s - loss: 7.6204 - accuracy: 0.5030
14976/25000 [================>.............] - ETA: 35s - loss: 7.6144 - accuracy: 0.5034
15008/25000 [=================>............] - ETA: 35s - loss: 7.6114 - accuracy: 0.5036
15040/25000 [=================>............] - ETA: 35s - loss: 7.6105 - accuracy: 0.5037
15072/25000 [=================>............] - ETA: 34s - loss: 7.6066 - accuracy: 0.5039
15104/25000 [=================>............] - ETA: 34s - loss: 7.6067 - accuracy: 0.5039
15136/25000 [=================>............] - ETA: 34s - loss: 7.6089 - accuracy: 0.5038
15168/25000 [=================>............] - ETA: 34s - loss: 7.6100 - accuracy: 0.5037
15200/25000 [=================>............] - ETA: 34s - loss: 7.6061 - accuracy: 0.5039
15232/25000 [=================>............] - ETA: 34s - loss: 7.6032 - accuracy: 0.5041
15264/25000 [=================>............] - ETA: 34s - loss: 7.6033 - accuracy: 0.5041
15296/25000 [=================>............] - ETA: 34s - loss: 7.6025 - accuracy: 0.5042
15328/25000 [=================>............] - ETA: 34s - loss: 7.6016 - accuracy: 0.5042
15360/25000 [=================>............] - ETA: 33s - loss: 7.6037 - accuracy: 0.5041
15392/25000 [=================>............] - ETA: 33s - loss: 7.6059 - accuracy: 0.5040
15424/25000 [=================>............] - ETA: 33s - loss: 7.6050 - accuracy: 0.5040
15456/25000 [=================>............] - ETA: 33s - loss: 7.6081 - accuracy: 0.5038
15488/25000 [=================>............] - ETA: 33s - loss: 7.6112 - accuracy: 0.5036
15520/25000 [=================>............] - ETA: 33s - loss: 7.6123 - accuracy: 0.5035
15552/25000 [=================>............] - ETA: 33s - loss: 7.6104 - accuracy: 0.5037
15584/25000 [=================>............] - ETA: 33s - loss: 7.6125 - accuracy: 0.5035
15616/25000 [=================>............] - ETA: 32s - loss: 7.6107 - accuracy: 0.5037
15648/25000 [=================>............] - ETA: 32s - loss: 7.6117 - accuracy: 0.5036
15680/25000 [=================>............] - ETA: 32s - loss: 7.6109 - accuracy: 0.5036
15712/25000 [=================>............] - ETA: 32s - loss: 7.6100 - accuracy: 0.5037
15744/25000 [=================>............] - ETA: 32s - loss: 7.6092 - accuracy: 0.5037
15776/25000 [=================>............] - ETA: 32s - loss: 7.6064 - accuracy: 0.5039
15808/25000 [=================>............] - ETA: 32s - loss: 7.6094 - accuracy: 0.5037
15840/25000 [==================>...........] - ETA: 32s - loss: 7.6114 - accuracy: 0.5036
15872/25000 [==================>...........] - ETA: 32s - loss: 7.6154 - accuracy: 0.5033
15904/25000 [==================>...........] - ETA: 31s - loss: 7.6146 - accuracy: 0.5034
15936/25000 [==================>...........] - ETA: 31s - loss: 7.6137 - accuracy: 0.5035
15968/25000 [==================>...........] - ETA: 31s - loss: 7.6157 - accuracy: 0.5033
16000/25000 [==================>...........] - ETA: 31s - loss: 7.6206 - accuracy: 0.5030
16032/25000 [==================>...........] - ETA: 31s - loss: 7.6226 - accuracy: 0.5029
16064/25000 [==================>...........] - ETA: 31s - loss: 7.6246 - accuracy: 0.5027
16096/25000 [==================>...........] - ETA: 31s - loss: 7.6257 - accuracy: 0.5027
16128/25000 [==================>...........] - ETA: 31s - loss: 7.6295 - accuracy: 0.5024
16160/25000 [==================>...........] - ETA: 31s - loss: 7.6258 - accuracy: 0.5027
16192/25000 [==================>...........] - ETA: 30s - loss: 7.6268 - accuracy: 0.5026
16224/25000 [==================>...........] - ETA: 30s - loss: 7.6279 - accuracy: 0.5025
16256/25000 [==================>...........] - ETA: 30s - loss: 7.6336 - accuracy: 0.5022
16288/25000 [==================>...........] - ETA: 30s - loss: 7.6365 - accuracy: 0.5020
16320/25000 [==================>...........] - ETA: 30s - loss: 7.6356 - accuracy: 0.5020
16352/25000 [==================>...........] - ETA: 30s - loss: 7.6404 - accuracy: 0.5017
16384/25000 [==================>...........] - ETA: 30s - loss: 7.6385 - accuracy: 0.5018
16416/25000 [==================>...........] - ETA: 30s - loss: 7.6423 - accuracy: 0.5016
16448/25000 [==================>...........] - ETA: 30s - loss: 7.6387 - accuracy: 0.5018
16480/25000 [==================>...........] - ETA: 29s - loss: 7.6368 - accuracy: 0.5019
16512/25000 [==================>...........] - ETA: 29s - loss: 7.6369 - accuracy: 0.5019
16544/25000 [==================>...........] - ETA: 29s - loss: 7.6379 - accuracy: 0.5019
16576/25000 [==================>...........] - ETA: 29s - loss: 7.6379 - accuracy: 0.5019
16608/25000 [==================>...........] - ETA: 29s - loss: 7.6380 - accuracy: 0.5019
16640/25000 [==================>...........] - ETA: 29s - loss: 7.6371 - accuracy: 0.5019
16672/25000 [===================>..........] - ETA: 29s - loss: 7.6455 - accuracy: 0.5014
16704/25000 [===================>..........] - ETA: 29s - loss: 7.6483 - accuracy: 0.5012
16736/25000 [===================>..........] - ETA: 29s - loss: 7.6437 - accuracy: 0.5015
16768/25000 [===================>..........] - ETA: 28s - loss: 7.6428 - accuracy: 0.5016
16800/25000 [===================>..........] - ETA: 28s - loss: 7.6438 - accuracy: 0.5015
16832/25000 [===================>..........] - ETA: 28s - loss: 7.6448 - accuracy: 0.5014
16864/25000 [===================>..........] - ETA: 28s - loss: 7.6475 - accuracy: 0.5012
16896/25000 [===================>..........] - ETA: 28s - loss: 7.6430 - accuracy: 0.5015
16928/25000 [===================>..........] - ETA: 28s - loss: 7.6413 - accuracy: 0.5017
16960/25000 [===================>..........] - ETA: 28s - loss: 7.6413 - accuracy: 0.5017
16992/25000 [===================>..........] - ETA: 28s - loss: 7.6405 - accuracy: 0.5017
17024/25000 [===================>..........] - ETA: 28s - loss: 7.6423 - accuracy: 0.5016
17056/25000 [===================>..........] - ETA: 27s - loss: 7.6396 - accuracy: 0.5018
17088/25000 [===================>..........] - ETA: 27s - loss: 7.6388 - accuracy: 0.5018
17120/25000 [===================>..........] - ETA: 27s - loss: 7.6371 - accuracy: 0.5019
17152/25000 [===================>..........] - ETA: 27s - loss: 7.6362 - accuracy: 0.5020
17184/25000 [===================>..........] - ETA: 27s - loss: 7.6399 - accuracy: 0.5017
17216/25000 [===================>..........] - ETA: 27s - loss: 7.6399 - accuracy: 0.5017
17248/25000 [===================>..........] - ETA: 27s - loss: 7.6426 - accuracy: 0.5016
17280/25000 [===================>..........] - ETA: 27s - loss: 7.6418 - accuracy: 0.5016
17312/25000 [===================>..........] - ETA: 27s - loss: 7.6392 - accuracy: 0.5018
17344/25000 [===================>..........] - ETA: 26s - loss: 7.6410 - accuracy: 0.5017
17376/25000 [===================>..........] - ETA: 26s - loss: 7.6384 - accuracy: 0.5018
17408/25000 [===================>..........] - ETA: 26s - loss: 7.6402 - accuracy: 0.5017
17440/25000 [===================>..........] - ETA: 26s - loss: 7.6376 - accuracy: 0.5019
17472/25000 [===================>..........] - ETA: 26s - loss: 7.6394 - accuracy: 0.5018
17504/25000 [====================>.........] - ETA: 26s - loss: 7.6342 - accuracy: 0.5021
17536/25000 [====================>.........] - ETA: 26s - loss: 7.6308 - accuracy: 0.5023
17568/25000 [====================>.........] - ETA: 26s - loss: 7.6317 - accuracy: 0.5023
17600/25000 [====================>.........] - ETA: 26s - loss: 7.6318 - accuracy: 0.5023
17632/25000 [====================>.........] - ETA: 25s - loss: 7.6327 - accuracy: 0.5022
17664/25000 [====================>.........] - ETA: 25s - loss: 7.6310 - accuracy: 0.5023
17696/25000 [====================>.........] - ETA: 25s - loss: 7.6311 - accuracy: 0.5023
17728/25000 [====================>.........] - ETA: 25s - loss: 7.6303 - accuracy: 0.5024
17760/25000 [====================>.........] - ETA: 25s - loss: 7.6321 - accuracy: 0.5023
17792/25000 [====================>.........] - ETA: 25s - loss: 7.6339 - accuracy: 0.5021
17824/25000 [====================>.........] - ETA: 25s - loss: 7.6313 - accuracy: 0.5023
17856/25000 [====================>.........] - ETA: 25s - loss: 7.6306 - accuracy: 0.5024
17888/25000 [====================>.........] - ETA: 25s - loss: 7.6323 - accuracy: 0.5022
17920/25000 [====================>.........] - ETA: 24s - loss: 7.6315 - accuracy: 0.5023
17952/25000 [====================>.........] - ETA: 24s - loss: 7.6282 - accuracy: 0.5025
17984/25000 [====================>.........] - ETA: 24s - loss: 7.6274 - accuracy: 0.5026
18016/25000 [====================>.........] - ETA: 24s - loss: 7.6249 - accuracy: 0.5027
18048/25000 [====================>.........] - ETA: 24s - loss: 7.6250 - accuracy: 0.5027
18080/25000 [====================>.........] - ETA: 24s - loss: 7.6293 - accuracy: 0.5024
18112/25000 [====================>.........] - ETA: 24s - loss: 7.6251 - accuracy: 0.5027
18144/25000 [====================>.........] - ETA: 24s - loss: 7.6286 - accuracy: 0.5025
18176/25000 [====================>.........] - ETA: 24s - loss: 7.6278 - accuracy: 0.5025
18208/25000 [====================>.........] - ETA: 23s - loss: 7.6211 - accuracy: 0.5030
18240/25000 [====================>.........] - ETA: 23s - loss: 7.6254 - accuracy: 0.5027
18272/25000 [====================>.........] - ETA: 23s - loss: 7.6255 - accuracy: 0.5027
18304/25000 [====================>.........] - ETA: 23s - loss: 7.6272 - accuracy: 0.5026
18336/25000 [=====================>........] - ETA: 23s - loss: 7.6282 - accuracy: 0.5025
18368/25000 [=====================>........] - ETA: 23s - loss: 7.6282 - accuracy: 0.5025
18400/25000 [=====================>........] - ETA: 23s - loss: 7.6300 - accuracy: 0.5024
18432/25000 [=====================>........] - ETA: 23s - loss: 7.6284 - accuracy: 0.5025
18464/25000 [=====================>........] - ETA: 22s - loss: 7.6268 - accuracy: 0.5026
18496/25000 [=====================>........] - ETA: 22s - loss: 7.6285 - accuracy: 0.5025
18528/25000 [=====================>........] - ETA: 22s - loss: 7.6310 - accuracy: 0.5023
18560/25000 [=====================>........] - ETA: 22s - loss: 7.6327 - accuracy: 0.5022
18592/25000 [=====================>........] - ETA: 22s - loss: 7.6345 - accuracy: 0.5021
18624/25000 [=====================>........] - ETA: 22s - loss: 7.6362 - accuracy: 0.5020
18656/25000 [=====================>........] - ETA: 22s - loss: 7.6362 - accuracy: 0.5020
18688/25000 [=====================>........] - ETA: 22s - loss: 7.6379 - accuracy: 0.5019
18720/25000 [=====================>........] - ETA: 22s - loss: 7.6355 - accuracy: 0.5020
18752/25000 [=====================>........] - ETA: 21s - loss: 7.6372 - accuracy: 0.5019
18784/25000 [=====================>........] - ETA: 21s - loss: 7.6364 - accuracy: 0.5020
18816/25000 [=====================>........] - ETA: 21s - loss: 7.6397 - accuracy: 0.5018
18848/25000 [=====================>........] - ETA: 21s - loss: 7.6349 - accuracy: 0.5021
18880/25000 [=====================>........] - ETA: 21s - loss: 7.6358 - accuracy: 0.5020
18912/25000 [=====================>........] - ETA: 21s - loss: 7.6350 - accuracy: 0.5021
18944/25000 [=====================>........] - ETA: 21s - loss: 7.6334 - accuracy: 0.5022
18976/25000 [=====================>........] - ETA: 21s - loss: 7.6367 - accuracy: 0.5019
19008/25000 [=====================>........] - ETA: 21s - loss: 7.6384 - accuracy: 0.5018
19040/25000 [=====================>........] - ETA: 20s - loss: 7.6425 - accuracy: 0.5016
19072/25000 [=====================>........] - ETA: 20s - loss: 7.6417 - accuracy: 0.5016
19104/25000 [=====================>........] - ETA: 20s - loss: 7.6401 - accuracy: 0.5017
19136/25000 [=====================>........] - ETA: 20s - loss: 7.6378 - accuracy: 0.5019
19168/25000 [======================>.......] - ETA: 20s - loss: 7.6394 - accuracy: 0.5018
19200/25000 [======================>.......] - ETA: 20s - loss: 7.6387 - accuracy: 0.5018
19232/25000 [======================>.......] - ETA: 20s - loss: 7.6379 - accuracy: 0.5019
19264/25000 [======================>.......] - ETA: 20s - loss: 7.6388 - accuracy: 0.5018
19296/25000 [======================>.......] - ETA: 20s - loss: 7.6348 - accuracy: 0.5021
19328/25000 [======================>.......] - ETA: 19s - loss: 7.6365 - accuracy: 0.5020
19360/25000 [======================>.......] - ETA: 19s - loss: 7.6373 - accuracy: 0.5019
19392/25000 [======================>.......] - ETA: 19s - loss: 7.6397 - accuracy: 0.5018
19424/25000 [======================>.......] - ETA: 19s - loss: 7.6398 - accuracy: 0.5018
19456/25000 [======================>.......] - ETA: 19s - loss: 7.6422 - accuracy: 0.5016
19488/25000 [======================>.......] - ETA: 19s - loss: 7.6438 - accuracy: 0.5015
19520/25000 [======================>.......] - ETA: 19s - loss: 7.6454 - accuracy: 0.5014
19552/25000 [======================>.......] - ETA: 19s - loss: 7.6462 - accuracy: 0.5013
19584/25000 [======================>.......] - ETA: 19s - loss: 7.6486 - accuracy: 0.5012
19616/25000 [======================>.......] - ETA: 18s - loss: 7.6525 - accuracy: 0.5009
19648/25000 [======================>.......] - ETA: 18s - loss: 7.6549 - accuracy: 0.5008
19680/25000 [======================>.......] - ETA: 18s - loss: 7.6557 - accuracy: 0.5007
19712/25000 [======================>.......] - ETA: 18s - loss: 7.6565 - accuracy: 0.5007
19744/25000 [======================>.......] - ETA: 18s - loss: 7.6573 - accuracy: 0.5006
19776/25000 [======================>.......] - ETA: 18s - loss: 7.6565 - accuracy: 0.5007
19808/25000 [======================>.......] - ETA: 18s - loss: 7.6581 - accuracy: 0.5006
19840/25000 [======================>.......] - ETA: 18s - loss: 7.6612 - accuracy: 0.5004
19872/25000 [======================>.......] - ETA: 18s - loss: 7.6604 - accuracy: 0.5004
19904/25000 [======================>.......] - ETA: 17s - loss: 7.6581 - accuracy: 0.5006
19936/25000 [======================>.......] - ETA: 17s - loss: 7.6582 - accuracy: 0.5006
19968/25000 [======================>.......] - ETA: 17s - loss: 7.6597 - accuracy: 0.5005
20000/25000 [=======================>......] - ETA: 17s - loss: 7.6597 - accuracy: 0.5005
20032/25000 [=======================>......] - ETA: 17s - loss: 7.6620 - accuracy: 0.5003
20064/25000 [=======================>......] - ETA: 17s - loss: 7.6636 - accuracy: 0.5002
20096/25000 [=======================>......] - ETA: 17s - loss: 7.6620 - accuracy: 0.5003
20128/25000 [=======================>......] - ETA: 17s - loss: 7.6628 - accuracy: 0.5002
20160/25000 [=======================>......] - ETA: 17s - loss: 7.6643 - accuracy: 0.5001
20192/25000 [=======================>......] - ETA: 16s - loss: 7.6643 - accuracy: 0.5001
20224/25000 [=======================>......] - ETA: 16s - loss: 7.6628 - accuracy: 0.5002
20256/25000 [=======================>......] - ETA: 16s - loss: 7.6621 - accuracy: 0.5003
20288/25000 [=======================>......] - ETA: 16s - loss: 7.6621 - accuracy: 0.5003
20320/25000 [=======================>......] - ETA: 16s - loss: 7.6628 - accuracy: 0.5002
20352/25000 [=======================>......] - ETA: 16s - loss: 7.6659 - accuracy: 0.5000
20384/25000 [=======================>......] - ETA: 16s - loss: 7.6666 - accuracy: 0.5000
20416/25000 [=======================>......] - ETA: 16s - loss: 7.6674 - accuracy: 0.5000
20448/25000 [=======================>......] - ETA: 16s - loss: 7.6711 - accuracy: 0.4997
20480/25000 [=======================>......] - ETA: 15s - loss: 7.6764 - accuracy: 0.4994
20512/25000 [=======================>......] - ETA: 15s - loss: 7.6756 - accuracy: 0.4994
20544/25000 [=======================>......] - ETA: 15s - loss: 7.6771 - accuracy: 0.4993
20576/25000 [=======================>......] - ETA: 15s - loss: 7.6771 - accuracy: 0.4993
20608/25000 [=======================>......] - ETA: 15s - loss: 7.6726 - accuracy: 0.4996
20640/25000 [=======================>......] - ETA: 15s - loss: 7.6733 - accuracy: 0.4996
20672/25000 [=======================>......] - ETA: 15s - loss: 7.6733 - accuracy: 0.4996
20704/25000 [=======================>......] - ETA: 15s - loss: 7.6733 - accuracy: 0.4996
20736/25000 [=======================>......] - ETA: 15s - loss: 7.6748 - accuracy: 0.4995
20768/25000 [=======================>......] - ETA: 14s - loss: 7.6740 - accuracy: 0.4995
20800/25000 [=======================>......] - ETA: 14s - loss: 7.6740 - accuracy: 0.4995
20832/25000 [=======================>......] - ETA: 14s - loss: 7.6725 - accuracy: 0.4996
20864/25000 [========================>.....] - ETA: 14s - loss: 7.6732 - accuracy: 0.4996
20896/25000 [========================>.....] - ETA: 14s - loss: 7.6725 - accuracy: 0.4996
20928/25000 [========================>.....] - ETA: 14s - loss: 7.6754 - accuracy: 0.4994
20960/25000 [========================>.....] - ETA: 14s - loss: 7.6732 - accuracy: 0.4996
20992/25000 [========================>.....] - ETA: 14s - loss: 7.6717 - accuracy: 0.4997
21024/25000 [========================>.....] - ETA: 13s - loss: 7.6725 - accuracy: 0.4996
21056/25000 [========================>.....] - ETA: 13s - loss: 7.6703 - accuracy: 0.4998
21088/25000 [========================>.....] - ETA: 13s - loss: 7.6717 - accuracy: 0.4997
21120/25000 [========================>.....] - ETA: 13s - loss: 7.6724 - accuracy: 0.4996
21152/25000 [========================>.....] - ETA: 13s - loss: 7.6688 - accuracy: 0.4999
21184/25000 [========================>.....] - ETA: 13s - loss: 7.6688 - accuracy: 0.4999
21216/25000 [========================>.....] - ETA: 13s - loss: 7.6673 - accuracy: 0.5000
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6630 - accuracy: 0.5002
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6652 - accuracy: 0.5001
21312/25000 [========================>.....] - ETA: 12s - loss: 7.6673 - accuracy: 0.5000
21344/25000 [========================>.....] - ETA: 12s - loss: 7.6645 - accuracy: 0.5001
21376/25000 [========================>.....] - ETA: 12s - loss: 7.6602 - accuracy: 0.5004
21408/25000 [========================>.....] - ETA: 12s - loss: 7.6602 - accuracy: 0.5004
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6595 - accuracy: 0.5005
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6609 - accuracy: 0.5004
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6595 - accuracy: 0.5005
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6588 - accuracy: 0.5005
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6616 - accuracy: 0.5003
21600/25000 [========================>.....] - ETA: 11s - loss: 7.6609 - accuracy: 0.5004
21632/25000 [========================>.....] - ETA: 11s - loss: 7.6631 - accuracy: 0.5002
21664/25000 [========================>.....] - ETA: 11s - loss: 7.6631 - accuracy: 0.5002
21696/25000 [=========================>....] - ETA: 11s - loss: 7.6645 - accuracy: 0.5001
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6631 - accuracy: 0.5002
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6624 - accuracy: 0.5003
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6617 - accuracy: 0.5003
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6589 - accuracy: 0.5005
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6568 - accuracy: 0.5006
21888/25000 [=========================>....] - ETA: 10s - loss: 7.6554 - accuracy: 0.5007
21920/25000 [=========================>....] - ETA: 10s - loss: 7.6568 - accuracy: 0.5006
21952/25000 [=========================>....] - ETA: 10s - loss: 7.6589 - accuracy: 0.5005
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6610 - accuracy: 0.5004
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6624 - accuracy: 0.5003
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6631 - accuracy: 0.5002
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6604 - accuracy: 0.5004
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6611 - accuracy: 0.5004
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6583 - accuracy: 0.5005
22176/25000 [=========================>....] - ETA: 9s - loss: 7.6590 - accuracy: 0.5005 
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6590 - accuracy: 0.5005
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6570 - accuracy: 0.5006
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6549 - accuracy: 0.5008
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6584 - accuracy: 0.5005
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6577 - accuracy: 0.5006
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6550 - accuracy: 0.5008
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6557 - accuracy: 0.5007
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6564 - accuracy: 0.5007
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6550 - accuracy: 0.5008
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6550 - accuracy: 0.5008
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6557 - accuracy: 0.5007
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6571 - accuracy: 0.5006
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6592 - accuracy: 0.5005
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6558 - accuracy: 0.5007
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6565 - accuracy: 0.5007
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6578 - accuracy: 0.5006
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6565 - accuracy: 0.5007
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6558 - accuracy: 0.5007
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6572 - accuracy: 0.5006
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6599 - accuracy: 0.5004
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6606 - accuracy: 0.5004
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6633 - accuracy: 0.5002
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6639 - accuracy: 0.5002
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6619 - accuracy: 0.5003
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6633 - accuracy: 0.5002
23008/25000 [==========================>...] - ETA: 7s - loss: 7.6633 - accuracy: 0.5002
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6606 - accuracy: 0.5004
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6613 - accuracy: 0.5003
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6620 - accuracy: 0.5003
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6633 - accuracy: 0.5002
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6660 - accuracy: 0.5000
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6640 - accuracy: 0.5002
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6666 - accuracy: 0.5000
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6633 - accuracy: 0.5002
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6633 - accuracy: 0.5002
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6646 - accuracy: 0.5001
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6620 - accuracy: 0.5003
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6627 - accuracy: 0.5003
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6620 - accuracy: 0.5003
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6620 - accuracy: 0.5003
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6614 - accuracy: 0.5003
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6627 - accuracy: 0.5003
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6640 - accuracy: 0.5002
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6634 - accuracy: 0.5002
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6614 - accuracy: 0.5003
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6647 - accuracy: 0.5001
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6640 - accuracy: 0.5002
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6647 - accuracy: 0.5001
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6724 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6730 - accuracy: 0.4996
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6736 - accuracy: 0.4995
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6698 - accuracy: 0.4998
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6723 - accuracy: 0.4996
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6704 - accuracy: 0.4998
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24192/25000 [============================>.] - ETA: 2s - loss: 7.6723 - accuracy: 0.4996
24224/25000 [============================>.] - ETA: 2s - loss: 7.6729 - accuracy: 0.4996
24256/25000 [============================>.] - ETA: 2s - loss: 7.6755 - accuracy: 0.4994
24288/25000 [============================>.] - ETA: 2s - loss: 7.6742 - accuracy: 0.4995
24320/25000 [============================>.] - ETA: 2s - loss: 7.6742 - accuracy: 0.4995
24352/25000 [============================>.] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
24384/25000 [============================>.] - ETA: 2s - loss: 7.6729 - accuracy: 0.4996
24416/25000 [============================>.] - ETA: 2s - loss: 7.6716 - accuracy: 0.4997
24448/25000 [============================>.] - ETA: 1s - loss: 7.6741 - accuracy: 0.4995
24480/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
24512/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24544/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24576/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24608/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24640/25000 [============================>.] - ETA: 1s - loss: 7.6747 - accuracy: 0.4995
24672/25000 [============================>.] - ETA: 1s - loss: 7.6722 - accuracy: 0.4996
24704/25000 [============================>.] - ETA: 1s - loss: 7.6722 - accuracy: 0.4996
24736/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24768/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24800/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24864/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24896/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24928/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fd44ea3ca90> 

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
 [-0.11160225  0.06428059  0.00966818 -0.01390419 -0.01983005  0.08546215]
 [-0.14851628  0.0807453  -0.04496838  0.07076558 -0.06786522  0.0575561 ]
 [-0.14564158 -0.1217351  -0.01037535  0.06395568  0.20123944  0.06635448]
 [-0.1553279   0.36773485 -0.05887907  0.03124439 -0.15177697  0.42701176]
 [-0.06823408  0.26008886  0.21381564  0.00427368  0.21831189 -0.01424692]
 [-0.10229161  0.20893313  0.35290614  0.24212243 -0.2610659  -0.05930964]
 [ 0.00612947  0.05199882 -0.01239984 -0.14805003 -0.04026802 -0.03519361]
 [-0.01305184  0.62603414  0.81116986  0.28469583 -0.15270881 -0.03140299]
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
{'loss': 0.4820788726210594, 'loss_history': []}

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
{'loss': 0.41347064822912216, 'loss_history': []}

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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:54<01:21, 27.30s/it] 40%|████      | 2/5 [00:54<01:21, 27.30s/it]
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.005325598623064976, 'embedding_size_factor': 1.3675652129565152, 'layers.choice': 3, 'learning_rate': 0.002317252059913027, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.225790831942204e-10} and reward: 0.3732
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?u\xd0K~\xae\xd8AX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\xe1\x8c\x0f\x8c\xbagX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?b\xfb\xa1:\x0b\x97\xa5X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xe0\xd8\xde:\x0f(\xd7u.' and reward: 0.3732
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?u\xd0K~\xae\xd8AX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\xe1\x8c\x0f\x8c\xbagX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?b\xfb\xa1:\x0b\x97\xa5X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xe0\xd8\xde:\x0f(\xd7u.' and reward: 0.3732
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 114.32045531272888
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.73s of the 3.99s of remaining time.
Ensemble size: 25
Ensemble weights: 
[0.96 0.04]
	0.387	 = Validation accuracy score
	0.95s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 116.99s ...
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

