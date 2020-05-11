
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/83c7f33970cca9536c5f95a36b96ffe9d8e7773e', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '83c7f33970cca9536c5f95a36b96ffe9d8e7773e', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/83c7f33970cca9536c5f95a36b96ffe9d8e7773e

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/83c7f33970cca9536c5f95a36b96ffe9d8e7773e

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
  245760/17464789 [..............................] - ETA: 3s
  540672/17464789 [..............................] - ETA: 3s
  942080/17464789 [>.............................] - ETA: 2s
 1376256/17464789 [=>............................] - ETA: 2s
 1810432/17464789 [==>...........................] - ETA: 2s
 2301952/17464789 [==>...........................] - ETA: 2s
 2908160/17464789 [===>..........................] - ETA: 1s
 3538944/17464789 [=====>........................] - ETA: 1s
 4202496/17464789 [======>.......................] - ETA: 1s
 4964352/17464789 [=======>......................] - ETA: 1s
 5554176/17464789 [========>.....................] - ETA: 1s
 6184960/17464789 [=========>....................] - ETA: 1s
 6848512/17464789 [==========>...................] - ETA: 1s
 7626752/17464789 [============>.................] - ETA: 0s
 8429568/17464789 [=============>................] - ETA: 0s
 9314304/17464789 [==============>...............] - ETA: 0s
10207232/17464789 [================>.............] - ETA: 0s
11042816/17464789 [=================>............] - ETA: 0s
12115968/17464789 [===================>..........] - ETA: 0s
13271040/17464789 [=====================>........] - ETA: 0s
14106624/17464789 [=======================>......] - ETA: 0s
15376384/17464789 [=========================>....] - ETA: 0s
16596992/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 03:15:40.670672: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 03:15:40.675065: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-11 03:15:40.675467: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56070cf241b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 03:15:40.675487: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:30 - loss: 7.1875 - accuracy: 0.5312
   64/25000 [..............................] - ETA: 2:47 - loss: 6.4687 - accuracy: 0.5781
   96/25000 [..............................] - ETA: 2:11 - loss: 6.5486 - accuracy: 0.5729
  128/25000 [..............................] - ETA: 1:51 - loss: 6.7083 - accuracy: 0.5625
  160/25000 [..............................] - ETA: 1:40 - loss: 7.1875 - accuracy: 0.5312
  192/25000 [..............................] - ETA: 1:33 - loss: 7.3472 - accuracy: 0.5208
  224/25000 [..............................] - ETA: 1:28 - loss: 7.4613 - accuracy: 0.5134
  256/25000 [..............................] - ETA: 1:24 - loss: 7.5468 - accuracy: 0.5078
  288/25000 [..............................] - ETA: 1:21 - loss: 7.2407 - accuracy: 0.5278
  320/25000 [..............................] - ETA: 1:19 - loss: 7.3791 - accuracy: 0.5188
  352/25000 [..............................] - ETA: 1:16 - loss: 7.3617 - accuracy: 0.5199
  384/25000 [..............................] - ETA: 1:15 - loss: 7.2673 - accuracy: 0.5260
  416/25000 [..............................] - ETA: 1:14 - loss: 7.4086 - accuracy: 0.5168
  448/25000 [..............................] - ETA: 1:12 - loss: 7.2217 - accuracy: 0.5290
  480/25000 [..............................] - ETA: 1:11 - loss: 7.3472 - accuracy: 0.5208
  512/25000 [..............................] - ETA: 1:11 - loss: 7.4270 - accuracy: 0.5156
  544/25000 [..............................] - ETA: 1:10 - loss: 7.6102 - accuracy: 0.5037
  576/25000 [..............................] - ETA: 1:09 - loss: 7.5601 - accuracy: 0.5069
  608/25000 [..............................] - ETA: 1:09 - loss: 7.5405 - accuracy: 0.5082
  640/25000 [..............................] - ETA: 1:09 - loss: 7.5229 - accuracy: 0.5094
  672/25000 [..............................] - ETA: 1:08 - loss: 7.5982 - accuracy: 0.5045
  704/25000 [..............................] - ETA: 1:08 - loss: 7.6448 - accuracy: 0.5014
  736/25000 [..............................] - ETA: 1:07 - loss: 7.5625 - accuracy: 0.5068
  768/25000 [..............................] - ETA: 1:07 - loss: 7.5269 - accuracy: 0.5091
  800/25000 [..............................] - ETA: 1:07 - loss: 7.5133 - accuracy: 0.5100
  832/25000 [..............................] - ETA: 1:06 - loss: 7.5929 - accuracy: 0.5048
  864/25000 [>.............................] - ETA: 1:06 - loss: 7.6489 - accuracy: 0.5012
  896/25000 [>.............................] - ETA: 1:06 - loss: 7.5639 - accuracy: 0.5067
  928/25000 [>.............................] - ETA: 1:05 - loss: 7.5179 - accuracy: 0.5097
  960/25000 [>.............................] - ETA: 1:05 - loss: 7.5229 - accuracy: 0.5094
  992/25000 [>.............................] - ETA: 1:04 - loss: 7.5584 - accuracy: 0.5071
 1024/25000 [>.............................] - ETA: 1:04 - loss: 7.4869 - accuracy: 0.5117
 1056/25000 [>.............................] - ETA: 1:04 - loss: 7.5359 - accuracy: 0.5085
 1088/25000 [>.............................] - ETA: 1:03 - loss: 7.5116 - accuracy: 0.5101
 1120/25000 [>.............................] - ETA: 1:03 - loss: 7.4613 - accuracy: 0.5134
 1152/25000 [>.............................] - ETA: 1:03 - loss: 7.4803 - accuracy: 0.5122
 1184/25000 [>.............................] - ETA: 1:03 - loss: 7.4465 - accuracy: 0.5144
 1216/25000 [>.............................] - ETA: 1:03 - loss: 7.3892 - accuracy: 0.5181
 1248/25000 [>.............................] - ETA: 1:03 - loss: 7.4086 - accuracy: 0.5168
 1280/25000 [>.............................] - ETA: 1:03 - loss: 7.4031 - accuracy: 0.5172
 1312/25000 [>.............................] - ETA: 1:03 - loss: 7.4446 - accuracy: 0.5145
 1344/25000 [>.............................] - ETA: 1:03 - loss: 7.4955 - accuracy: 0.5112
 1376/25000 [>.............................] - ETA: 1:03 - loss: 7.5106 - accuracy: 0.5102
 1408/25000 [>.............................] - ETA: 1:02 - loss: 7.5250 - accuracy: 0.5092
 1440/25000 [>.............................] - ETA: 1:03 - loss: 7.4856 - accuracy: 0.5118
 1472/25000 [>.............................] - ETA: 1:02 - loss: 7.5000 - accuracy: 0.5109
 1504/25000 [>.............................] - ETA: 1:02 - loss: 7.5647 - accuracy: 0.5066
 1536/25000 [>.............................] - ETA: 1:02 - loss: 7.5269 - accuracy: 0.5091
 1568/25000 [>.............................] - ETA: 1:02 - loss: 7.5688 - accuracy: 0.5064
 1600/25000 [>.............................] - ETA: 1:02 - loss: 7.5612 - accuracy: 0.5069
 1632/25000 [>.............................] - ETA: 1:02 - loss: 7.5915 - accuracy: 0.5049
 1664/25000 [>.............................] - ETA: 1:01 - loss: 7.6113 - accuracy: 0.5036
 1696/25000 [=>............................] - ETA: 1:01 - loss: 7.6305 - accuracy: 0.5024
 1728/25000 [=>............................] - ETA: 1:01 - loss: 7.6400 - accuracy: 0.5017
 1760/25000 [=>............................] - ETA: 1:01 - loss: 7.6579 - accuracy: 0.5006
 1792/25000 [=>............................] - ETA: 1:01 - loss: 7.6410 - accuracy: 0.5017
 1824/25000 [=>............................] - ETA: 1:01 - loss: 7.6414 - accuracy: 0.5016
 1856/25000 [=>............................] - ETA: 1:01 - loss: 7.6005 - accuracy: 0.5043
 1888/25000 [=>............................] - ETA: 1:00 - loss: 7.5935 - accuracy: 0.5048
 1920/25000 [=>............................] - ETA: 1:00 - loss: 7.5708 - accuracy: 0.5063
 1952/25000 [=>............................] - ETA: 1:00 - loss: 7.5724 - accuracy: 0.5061
 1984/25000 [=>............................] - ETA: 1:00 - loss: 7.6048 - accuracy: 0.5040
 2016/25000 [=>............................] - ETA: 1:00 - loss: 7.6134 - accuracy: 0.5035
 2048/25000 [=>............................] - ETA: 1:00 - loss: 7.6292 - accuracy: 0.5024
 2080/25000 [=>............................] - ETA: 1:00 - loss: 7.6445 - accuracy: 0.5014
 2112/25000 [=>............................] - ETA: 1:00 - loss: 7.6376 - accuracy: 0.5019
 2144/25000 [=>............................] - ETA: 59s - loss: 7.6738 - accuracy: 0.4995 
 2176/25000 [=>............................] - ETA: 59s - loss: 7.6666 - accuracy: 0.5000
 2208/25000 [=>............................] - ETA: 59s - loss: 7.6666 - accuracy: 0.5000
 2240/25000 [=>............................] - ETA: 59s - loss: 7.6735 - accuracy: 0.4996
 2272/25000 [=>............................] - ETA: 59s - loss: 7.6666 - accuracy: 0.5000
 2304/25000 [=>............................] - ETA: 59s - loss: 7.6533 - accuracy: 0.5009
 2336/25000 [=>............................] - ETA: 59s - loss: 7.6666 - accuracy: 0.5000
 2368/25000 [=>............................] - ETA: 59s - loss: 7.6407 - accuracy: 0.5017
 2400/25000 [=>............................] - ETA: 58s - loss: 7.6794 - accuracy: 0.4992
 2432/25000 [=>............................] - ETA: 58s - loss: 7.6918 - accuracy: 0.4984
 2464/25000 [=>............................] - ETA: 58s - loss: 7.6853 - accuracy: 0.4988
 2496/25000 [=>............................] - ETA: 58s - loss: 7.6973 - accuracy: 0.4980
 2528/25000 [==>...........................] - ETA: 58s - loss: 7.6969 - accuracy: 0.4980
 2560/25000 [==>...........................] - ETA: 58s - loss: 7.6786 - accuracy: 0.4992
 2592/25000 [==>...........................] - ETA: 58s - loss: 7.6725 - accuracy: 0.4996
 2624/25000 [==>...........................] - ETA: 57s - loss: 7.6725 - accuracy: 0.4996
 2656/25000 [==>...........................] - ETA: 57s - loss: 7.6724 - accuracy: 0.4996
 2688/25000 [==>...........................] - ETA: 57s - loss: 7.6609 - accuracy: 0.5004
 2720/25000 [==>...........................] - ETA: 57s - loss: 7.6666 - accuracy: 0.5000
 2752/25000 [==>...........................] - ETA: 57s - loss: 7.6555 - accuracy: 0.5007
 2784/25000 [==>...........................] - ETA: 57s - loss: 7.6611 - accuracy: 0.5004
 2816/25000 [==>...........................] - ETA: 57s - loss: 7.6557 - accuracy: 0.5007
 2848/25000 [==>...........................] - ETA: 57s - loss: 7.6612 - accuracy: 0.5004
 2880/25000 [==>...........................] - ETA: 56s - loss: 7.6187 - accuracy: 0.5031
 2912/25000 [==>...........................] - ETA: 56s - loss: 7.6245 - accuracy: 0.5027
 2944/25000 [==>...........................] - ETA: 56s - loss: 7.6145 - accuracy: 0.5034
 2976/25000 [==>...........................] - ETA: 56s - loss: 7.6254 - accuracy: 0.5027
 3008/25000 [==>...........................] - ETA: 56s - loss: 7.6462 - accuracy: 0.5013
 3040/25000 [==>...........................] - ETA: 56s - loss: 7.6464 - accuracy: 0.5013
 3072/25000 [==>...........................] - ETA: 56s - loss: 7.6217 - accuracy: 0.5029
 3104/25000 [==>...........................] - ETA: 56s - loss: 7.6469 - accuracy: 0.5013
 3136/25000 [==>...........................] - ETA: 56s - loss: 7.6617 - accuracy: 0.5003
 3168/25000 [==>...........................] - ETA: 56s - loss: 7.6811 - accuracy: 0.4991
 3200/25000 [==>...........................] - ETA: 55s - loss: 7.6618 - accuracy: 0.5003
 3232/25000 [==>...........................] - ETA: 55s - loss: 7.6714 - accuracy: 0.4997
 3264/25000 [==>...........................] - ETA: 55s - loss: 7.6901 - accuracy: 0.4985
 3296/25000 [==>...........................] - ETA: 55s - loss: 7.6945 - accuracy: 0.4982
 3328/25000 [==>...........................] - ETA: 55s - loss: 7.6989 - accuracy: 0.4979
 3360/25000 [===>..........................] - ETA: 55s - loss: 7.7077 - accuracy: 0.4973
 3392/25000 [===>..........................] - ETA: 55s - loss: 7.7163 - accuracy: 0.4968
 3424/25000 [===>..........................] - ETA: 55s - loss: 7.7293 - accuracy: 0.4959
 3456/25000 [===>..........................] - ETA: 55s - loss: 7.7376 - accuracy: 0.4954
 3488/25000 [===>..........................] - ETA: 55s - loss: 7.7457 - accuracy: 0.4948
 3520/25000 [===>..........................] - ETA: 54s - loss: 7.7276 - accuracy: 0.4960
 3552/25000 [===>..........................] - ETA: 54s - loss: 7.7443 - accuracy: 0.4949
 3584/25000 [===>..........................] - ETA: 54s - loss: 7.7393 - accuracy: 0.4953
 3616/25000 [===>..........................] - ETA: 54s - loss: 7.7429 - accuracy: 0.4950
 3648/25000 [===>..........................] - ETA: 54s - loss: 7.7339 - accuracy: 0.4956
 3680/25000 [===>..........................] - ETA: 54s - loss: 7.7416 - accuracy: 0.4951
 3712/25000 [===>..........................] - ETA: 54s - loss: 7.7410 - accuracy: 0.4952
 3744/25000 [===>..........................] - ETA: 54s - loss: 7.7281 - accuracy: 0.4960
 3776/25000 [===>..........................] - ETA: 54s - loss: 7.7153 - accuracy: 0.4968
 3808/25000 [===>..........................] - ETA: 54s - loss: 7.6988 - accuracy: 0.4979
 3840/25000 [===>..........................] - ETA: 53s - loss: 7.6866 - accuracy: 0.4987
 3872/25000 [===>..........................] - ETA: 53s - loss: 7.6983 - accuracy: 0.4979
 3904/25000 [===>..........................] - ETA: 53s - loss: 7.6902 - accuracy: 0.4985
 3936/25000 [===>..........................] - ETA: 53s - loss: 7.7017 - accuracy: 0.4977
 3968/25000 [===>..........................] - ETA: 53s - loss: 7.7091 - accuracy: 0.4972
 4000/25000 [===>..........................] - ETA: 53s - loss: 7.6896 - accuracy: 0.4985
 4032/25000 [===>..........................] - ETA: 53s - loss: 7.6666 - accuracy: 0.5000
 4064/25000 [===>..........................] - ETA: 53s - loss: 7.6779 - accuracy: 0.4993
 4096/25000 [===>..........................] - ETA: 53s - loss: 7.6666 - accuracy: 0.5000
 4128/25000 [===>..........................] - ETA: 53s - loss: 7.6629 - accuracy: 0.5002
 4160/25000 [===>..........................] - ETA: 53s - loss: 7.6666 - accuracy: 0.5000
 4192/25000 [====>.........................] - ETA: 53s - loss: 7.6666 - accuracy: 0.5000
 4224/25000 [====>.........................] - ETA: 53s - loss: 7.6666 - accuracy: 0.5000
 4256/25000 [====>.........................] - ETA: 52s - loss: 7.6774 - accuracy: 0.4993
 4288/25000 [====>.........................] - ETA: 52s - loss: 7.6702 - accuracy: 0.4998
 4320/25000 [====>.........................] - ETA: 52s - loss: 7.6773 - accuracy: 0.4993
 4352/25000 [====>.........................] - ETA: 52s - loss: 7.6737 - accuracy: 0.4995
 4384/25000 [====>.........................] - ETA: 52s - loss: 7.6701 - accuracy: 0.4998
 4416/25000 [====>.........................] - ETA: 52s - loss: 7.6666 - accuracy: 0.5000
 4448/25000 [====>.........................] - ETA: 52s - loss: 7.6770 - accuracy: 0.4993
 4480/25000 [====>.........................] - ETA: 52s - loss: 7.6769 - accuracy: 0.4993
 4512/25000 [====>.........................] - ETA: 52s - loss: 7.6734 - accuracy: 0.4996
 4544/25000 [====>.........................] - ETA: 52s - loss: 7.6835 - accuracy: 0.4989
 4576/25000 [====>.........................] - ETA: 52s - loss: 7.6834 - accuracy: 0.4989
 4608/25000 [====>.........................] - ETA: 51s - loss: 7.6733 - accuracy: 0.4996
 4640/25000 [====>.........................] - ETA: 51s - loss: 7.6666 - accuracy: 0.5000
 4672/25000 [====>.........................] - ETA: 51s - loss: 7.6601 - accuracy: 0.5004
 4704/25000 [====>.........................] - ETA: 51s - loss: 7.6601 - accuracy: 0.5004
 4736/25000 [====>.........................] - ETA: 51s - loss: 7.6763 - accuracy: 0.4994
 4768/25000 [====>.........................] - ETA: 51s - loss: 7.6956 - accuracy: 0.4981
 4800/25000 [====>.........................] - ETA: 51s - loss: 7.7050 - accuracy: 0.4975
 4832/25000 [====>.........................] - ETA: 51s - loss: 7.6984 - accuracy: 0.4979
 4864/25000 [====>.........................] - ETA: 51s - loss: 7.7108 - accuracy: 0.4971
 4896/25000 [====>.........................] - ETA: 51s - loss: 7.7199 - accuracy: 0.4965
 4928/25000 [====>.........................] - ETA: 51s - loss: 7.7257 - accuracy: 0.4961
 4960/25000 [====>.........................] - ETA: 51s - loss: 7.7346 - accuracy: 0.4956
 4992/25000 [====>.........................] - ETA: 51s - loss: 7.7219 - accuracy: 0.4964
 5024/25000 [=====>........................] - ETA: 51s - loss: 7.7093 - accuracy: 0.4972
 5056/25000 [=====>........................] - ETA: 50s - loss: 7.7091 - accuracy: 0.4972
 5088/25000 [=====>........................] - ETA: 50s - loss: 7.7088 - accuracy: 0.4972
 5120/25000 [=====>........................] - ETA: 50s - loss: 7.7056 - accuracy: 0.4975
 5152/25000 [=====>........................] - ETA: 50s - loss: 7.7113 - accuracy: 0.4971
 5184/25000 [=====>........................] - ETA: 50s - loss: 7.7110 - accuracy: 0.4971
 5216/25000 [=====>........................] - ETA: 50s - loss: 7.7313 - accuracy: 0.4958
 5248/25000 [=====>........................] - ETA: 50s - loss: 7.7338 - accuracy: 0.4956
 5280/25000 [=====>........................] - ETA: 50s - loss: 7.7160 - accuracy: 0.4968
 5312/25000 [=====>........................] - ETA: 50s - loss: 7.7186 - accuracy: 0.4966
 5344/25000 [=====>........................] - ETA: 50s - loss: 7.7039 - accuracy: 0.4976
 5376/25000 [=====>........................] - ETA: 50s - loss: 7.7123 - accuracy: 0.4970
 5408/25000 [=====>........................] - ETA: 50s - loss: 7.6978 - accuracy: 0.4980
 5440/25000 [=====>........................] - ETA: 49s - loss: 7.6976 - accuracy: 0.4980
 5472/25000 [=====>........................] - ETA: 49s - loss: 7.6974 - accuracy: 0.4980
 5504/25000 [=====>........................] - ETA: 49s - loss: 7.6945 - accuracy: 0.4982
 5536/25000 [=====>........................] - ETA: 49s - loss: 7.6943 - accuracy: 0.4982
 5568/25000 [=====>........................] - ETA: 49s - loss: 7.6969 - accuracy: 0.4980
 5600/25000 [=====>........................] - ETA: 49s - loss: 7.6885 - accuracy: 0.4986
 5632/25000 [=====>........................] - ETA: 49s - loss: 7.6938 - accuracy: 0.4982
 5664/25000 [=====>........................] - ETA: 49s - loss: 7.7045 - accuracy: 0.4975
 5696/25000 [=====>........................] - ETA: 49s - loss: 7.7043 - accuracy: 0.4975
 5728/25000 [=====>........................] - ETA: 49s - loss: 7.7148 - accuracy: 0.4969
 5760/25000 [=====>........................] - ETA: 49s - loss: 7.7012 - accuracy: 0.4977
 5792/25000 [=====>........................] - ETA: 48s - loss: 7.6878 - accuracy: 0.4986
 5824/25000 [=====>........................] - ETA: 48s - loss: 7.6903 - accuracy: 0.4985
 5856/25000 [======>.......................] - ETA: 48s - loss: 7.6849 - accuracy: 0.4988
 5888/25000 [======>.......................] - ETA: 48s - loss: 7.6953 - accuracy: 0.4981
 5920/25000 [======>.......................] - ETA: 48s - loss: 7.6977 - accuracy: 0.4980
 5952/25000 [======>.......................] - ETA: 48s - loss: 7.7156 - accuracy: 0.4968
 5984/25000 [======>.......................] - ETA: 48s - loss: 7.7076 - accuracy: 0.4973
 6016/25000 [======>.......................] - ETA: 48s - loss: 7.6921 - accuracy: 0.4983
 6048/25000 [======>.......................] - ETA: 48s - loss: 7.6869 - accuracy: 0.4987
 6080/25000 [======>.......................] - ETA: 48s - loss: 7.6868 - accuracy: 0.4987
 6112/25000 [======>.......................] - ETA: 48s - loss: 7.6892 - accuracy: 0.4985
 6144/25000 [======>.......................] - ETA: 47s - loss: 7.6816 - accuracy: 0.4990
 6176/25000 [======>.......................] - ETA: 47s - loss: 7.6865 - accuracy: 0.4987
 6208/25000 [======>.......................] - ETA: 47s - loss: 7.6765 - accuracy: 0.4994
 6240/25000 [======>.......................] - ETA: 47s - loss: 7.6789 - accuracy: 0.4992
 6272/25000 [======>.......................] - ETA: 47s - loss: 7.6740 - accuracy: 0.4995
 6304/25000 [======>.......................] - ETA: 47s - loss: 7.6788 - accuracy: 0.4992
 6336/25000 [======>.......................] - ETA: 47s - loss: 7.6884 - accuracy: 0.4986
 6368/25000 [======>.......................] - ETA: 47s - loss: 7.6883 - accuracy: 0.4986
 6400/25000 [======>.......................] - ETA: 47s - loss: 7.6858 - accuracy: 0.4988
 6432/25000 [======>.......................] - ETA: 47s - loss: 7.6976 - accuracy: 0.4980
 6464/25000 [======>.......................] - ETA: 47s - loss: 7.6832 - accuracy: 0.4989
 6496/25000 [======>.......................] - ETA: 46s - loss: 7.6902 - accuracy: 0.4985
 6528/25000 [======>.......................] - ETA: 46s - loss: 7.6925 - accuracy: 0.4983
 6560/25000 [======>.......................] - ETA: 46s - loss: 7.7017 - accuracy: 0.4977
 6592/25000 [======>.......................] - ETA: 46s - loss: 7.6922 - accuracy: 0.4983
 6624/25000 [======>.......................] - ETA: 46s - loss: 7.6990 - accuracy: 0.4979
 6656/25000 [======>.......................] - ETA: 46s - loss: 7.6897 - accuracy: 0.4985
 6688/25000 [=======>......................] - ETA: 46s - loss: 7.6850 - accuracy: 0.4988
 6720/25000 [=======>......................] - ETA: 46s - loss: 7.6735 - accuracy: 0.4996
 6752/25000 [=======>......................] - ETA: 46s - loss: 7.6734 - accuracy: 0.4996
 6784/25000 [=======>......................] - ETA: 46s - loss: 7.6757 - accuracy: 0.4994
 6816/25000 [=======>......................] - ETA: 46s - loss: 7.6689 - accuracy: 0.4999
 6848/25000 [=======>......................] - ETA: 46s - loss: 7.6621 - accuracy: 0.5003
 6880/25000 [=======>......................] - ETA: 45s - loss: 7.6577 - accuracy: 0.5006
 6912/25000 [=======>......................] - ETA: 45s - loss: 7.6600 - accuracy: 0.5004
 6944/25000 [=======>......................] - ETA: 45s - loss: 7.6578 - accuracy: 0.5006
 6976/25000 [=======>......................] - ETA: 45s - loss: 7.6622 - accuracy: 0.5003
 7008/25000 [=======>......................] - ETA: 45s - loss: 7.6601 - accuracy: 0.5004
 7040/25000 [=======>......................] - ETA: 45s - loss: 7.6601 - accuracy: 0.5004
 7072/25000 [=======>......................] - ETA: 45s - loss: 7.6558 - accuracy: 0.5007
 7104/25000 [=======>......................] - ETA: 45s - loss: 7.6580 - accuracy: 0.5006
 7136/25000 [=======>......................] - ETA: 45s - loss: 7.6516 - accuracy: 0.5010
 7168/25000 [=======>......................] - ETA: 45s - loss: 7.6495 - accuracy: 0.5011
 7200/25000 [=======>......................] - ETA: 45s - loss: 7.6581 - accuracy: 0.5006
 7232/25000 [=======>......................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 7264/25000 [=======>......................] - ETA: 44s - loss: 7.6687 - accuracy: 0.4999
 7296/25000 [=======>......................] - ETA: 44s - loss: 7.6729 - accuracy: 0.4996
 7328/25000 [=======>......................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 7360/25000 [=======>......................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 7392/25000 [=======>......................] - ETA: 44s - loss: 7.6583 - accuracy: 0.5005
 7424/25000 [=======>......................] - ETA: 44s - loss: 7.6625 - accuracy: 0.5003
 7456/25000 [=======>......................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 7488/25000 [=======>......................] - ETA: 44s - loss: 7.6584 - accuracy: 0.5005
 7520/25000 [========>.....................] - ETA: 44s - loss: 7.6625 - accuracy: 0.5003
 7552/25000 [========>.....................] - ETA: 44s - loss: 7.6707 - accuracy: 0.4997
 7584/25000 [========>.....................] - ETA: 44s - loss: 7.6707 - accuracy: 0.4997
 7616/25000 [========>.....................] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
 7648/25000 [========>.....................] - ETA: 43s - loss: 7.6746 - accuracy: 0.4995
 7680/25000 [========>.....................] - ETA: 43s - loss: 7.6806 - accuracy: 0.4991
 7712/25000 [========>.....................] - ETA: 43s - loss: 7.6706 - accuracy: 0.4997
 7744/25000 [========>.....................] - ETA: 43s - loss: 7.6745 - accuracy: 0.4995
 7776/25000 [========>.....................] - ETA: 43s - loss: 7.6863 - accuracy: 0.4987
 7808/25000 [========>.....................] - ETA: 43s - loss: 7.6784 - accuracy: 0.4992
 7840/25000 [========>.....................] - ETA: 43s - loss: 7.6744 - accuracy: 0.4995
 7872/25000 [========>.....................] - ETA: 43s - loss: 7.6744 - accuracy: 0.4995
 7904/25000 [========>.....................] - ETA: 43s - loss: 7.6860 - accuracy: 0.4987
 7936/25000 [========>.....................] - ETA: 43s - loss: 7.6975 - accuracy: 0.4980
 7968/25000 [========>.....................] - ETA: 43s - loss: 7.6974 - accuracy: 0.4980
 8000/25000 [========>.....................] - ETA: 43s - loss: 7.6954 - accuracy: 0.4981
 8032/25000 [========>.....................] - ETA: 42s - loss: 7.6991 - accuracy: 0.4979
 8064/25000 [========>.....................] - ETA: 42s - loss: 7.6970 - accuracy: 0.4980
 8096/25000 [========>.....................] - ETA: 42s - loss: 7.7007 - accuracy: 0.4978
 8128/25000 [========>.....................] - ETA: 42s - loss: 7.7043 - accuracy: 0.4975
 8160/25000 [========>.....................] - ETA: 42s - loss: 7.6986 - accuracy: 0.4979
 8192/25000 [========>.....................] - ETA: 42s - loss: 7.6947 - accuracy: 0.4982
 8224/25000 [========>.....................] - ETA: 42s - loss: 7.7002 - accuracy: 0.4978
 8256/25000 [========>.....................] - ETA: 42s - loss: 7.6945 - accuracy: 0.4982
 8288/25000 [========>.....................] - ETA: 42s - loss: 7.6981 - accuracy: 0.4979
 8320/25000 [========>.....................] - ETA: 42s - loss: 7.6998 - accuracy: 0.4978
 8352/25000 [=========>....................] - ETA: 42s - loss: 7.7052 - accuracy: 0.4975
 8384/25000 [=========>....................] - ETA: 42s - loss: 7.7014 - accuracy: 0.4977
 8416/25000 [=========>....................] - ETA: 41s - loss: 7.6958 - accuracy: 0.4981
 8448/25000 [=========>....................] - ETA: 41s - loss: 7.6920 - accuracy: 0.4983
 8480/25000 [=========>....................] - ETA: 41s - loss: 7.6937 - accuracy: 0.4982
 8512/25000 [=========>....................] - ETA: 41s - loss: 7.6864 - accuracy: 0.4987
 8544/25000 [=========>....................] - ETA: 41s - loss: 7.6882 - accuracy: 0.4986
 8576/25000 [=========>....................] - ETA: 41s - loss: 7.6934 - accuracy: 0.4983
 8608/25000 [=========>....................] - ETA: 41s - loss: 7.7022 - accuracy: 0.4977
 8640/25000 [=========>....................] - ETA: 41s - loss: 7.7074 - accuracy: 0.4973
 8672/25000 [=========>....................] - ETA: 41s - loss: 7.7091 - accuracy: 0.4972
 8704/25000 [=========>....................] - ETA: 41s - loss: 7.7089 - accuracy: 0.4972
 8736/25000 [=========>....................] - ETA: 41s - loss: 7.7035 - accuracy: 0.4976
 8768/25000 [=========>....................] - ETA: 40s - loss: 7.6929 - accuracy: 0.4983
 8800/25000 [=========>....................] - ETA: 40s - loss: 7.6858 - accuracy: 0.4988
 8832/25000 [=========>....................] - ETA: 40s - loss: 7.6770 - accuracy: 0.4993
 8864/25000 [=========>....................] - ETA: 40s - loss: 7.6770 - accuracy: 0.4993
 8896/25000 [=========>....................] - ETA: 40s - loss: 7.6804 - accuracy: 0.4991
 8928/25000 [=========>....................] - ETA: 40s - loss: 7.6838 - accuracy: 0.4989
 8960/25000 [=========>....................] - ETA: 40s - loss: 7.6854 - accuracy: 0.4988
 8992/25000 [=========>....................] - ETA: 40s - loss: 7.6820 - accuracy: 0.4990
 9024/25000 [=========>....................] - ETA: 40s - loss: 7.6802 - accuracy: 0.4991
 9056/25000 [=========>....................] - ETA: 40s - loss: 7.6836 - accuracy: 0.4989
 9088/25000 [=========>....................] - ETA: 40s - loss: 7.6835 - accuracy: 0.4989
 9120/25000 [=========>....................] - ETA: 40s - loss: 7.6851 - accuracy: 0.4988
 9152/25000 [=========>....................] - ETA: 40s - loss: 7.6901 - accuracy: 0.4985
 9184/25000 [==========>...................] - ETA: 40s - loss: 7.6883 - accuracy: 0.4986
 9216/25000 [==========>...................] - ETA: 39s - loss: 7.6916 - accuracy: 0.4984
 9248/25000 [==========>...................] - ETA: 39s - loss: 7.6915 - accuracy: 0.4984
 9280/25000 [==========>...................] - ETA: 39s - loss: 7.6931 - accuracy: 0.4983
 9312/25000 [==========>...................] - ETA: 39s - loss: 7.6979 - accuracy: 0.4980
 9344/25000 [==========>...................] - ETA: 39s - loss: 7.6962 - accuracy: 0.4981
 9376/25000 [==========>...................] - ETA: 39s - loss: 7.6961 - accuracy: 0.4981
 9408/25000 [==========>...................] - ETA: 39s - loss: 7.7041 - accuracy: 0.4976
 9440/25000 [==========>...................] - ETA: 39s - loss: 7.7024 - accuracy: 0.4977
 9472/25000 [==========>...................] - ETA: 39s - loss: 7.7006 - accuracy: 0.4978
 9504/25000 [==========>...................] - ETA: 39s - loss: 7.7037 - accuracy: 0.4976
 9536/25000 [==========>...................] - ETA: 39s - loss: 7.6988 - accuracy: 0.4979
 9568/25000 [==========>...................] - ETA: 39s - loss: 7.7003 - accuracy: 0.4978
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.6986 - accuracy: 0.4979
 9632/25000 [==========>...................] - ETA: 38s - loss: 7.7032 - accuracy: 0.4976
 9664/25000 [==========>...................] - ETA: 38s - loss: 7.6952 - accuracy: 0.4981
 9696/25000 [==========>...................] - ETA: 38s - loss: 7.6935 - accuracy: 0.4982
 9728/25000 [==========>...................] - ETA: 38s - loss: 7.6981 - accuracy: 0.4979
 9760/25000 [==========>...................] - ETA: 38s - loss: 7.6996 - accuracy: 0.4978
 9792/25000 [==========>...................] - ETA: 38s - loss: 7.6995 - accuracy: 0.4979
 9824/25000 [==========>...................] - ETA: 38s - loss: 7.6994 - accuracy: 0.4979
 9856/25000 [==========>...................] - ETA: 38s - loss: 7.7008 - accuracy: 0.4978
 9888/25000 [==========>...................] - ETA: 38s - loss: 7.6961 - accuracy: 0.4981
 9920/25000 [==========>...................] - ETA: 38s - loss: 7.6975 - accuracy: 0.4980
 9952/25000 [==========>...................] - ETA: 38s - loss: 7.6928 - accuracy: 0.4983
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.6927 - accuracy: 0.4983
10016/25000 [===========>..................] - ETA: 37s - loss: 7.6972 - accuracy: 0.4980
10048/25000 [===========>..................] - ETA: 37s - loss: 7.6956 - accuracy: 0.4981
10080/25000 [===========>..................] - ETA: 37s - loss: 7.6879 - accuracy: 0.4986
10112/25000 [===========>..................] - ETA: 37s - loss: 7.6909 - accuracy: 0.4984
10144/25000 [===========>..................] - ETA: 37s - loss: 7.6953 - accuracy: 0.4981
10176/25000 [===========>..................] - ETA: 37s - loss: 7.6937 - accuracy: 0.4982
10208/25000 [===========>..................] - ETA: 37s - loss: 7.6922 - accuracy: 0.4983
10240/25000 [===========>..................] - ETA: 37s - loss: 7.6876 - accuracy: 0.4986
10272/25000 [===========>..................] - ETA: 37s - loss: 7.6845 - accuracy: 0.4988
10304/25000 [===========>..................] - ETA: 37s - loss: 7.6845 - accuracy: 0.4988
10336/25000 [===========>..................] - ETA: 37s - loss: 7.6859 - accuracy: 0.4987
10368/25000 [===========>..................] - ETA: 37s - loss: 7.6873 - accuracy: 0.4986
10400/25000 [===========>..................] - ETA: 37s - loss: 7.6828 - accuracy: 0.4989
10432/25000 [===========>..................] - ETA: 36s - loss: 7.6843 - accuracy: 0.4988
10464/25000 [===========>..................] - ETA: 36s - loss: 7.6827 - accuracy: 0.4989
10496/25000 [===========>..................] - ETA: 36s - loss: 7.6812 - accuracy: 0.4990
10528/25000 [===========>..................] - ETA: 36s - loss: 7.6856 - accuracy: 0.4988
10560/25000 [===========>..................] - ETA: 36s - loss: 7.6884 - accuracy: 0.4986
10592/25000 [===========>..................] - ETA: 36s - loss: 7.6869 - accuracy: 0.4987
10624/25000 [===========>..................] - ETA: 36s - loss: 7.6811 - accuracy: 0.4991
10656/25000 [===========>..................] - ETA: 36s - loss: 7.6839 - accuracy: 0.4989
10688/25000 [===========>..................] - ETA: 36s - loss: 7.6853 - accuracy: 0.4988
10720/25000 [===========>..................] - ETA: 36s - loss: 7.6881 - accuracy: 0.4986
10752/25000 [===========>..................] - ETA: 36s - loss: 7.6894 - accuracy: 0.4985
10784/25000 [===========>..................] - ETA: 36s - loss: 7.6865 - accuracy: 0.4987
10816/25000 [===========>..................] - ETA: 35s - loss: 7.6893 - accuracy: 0.4985
10848/25000 [============>.................] - ETA: 35s - loss: 7.6921 - accuracy: 0.4983
10880/25000 [============>.................] - ETA: 35s - loss: 7.6906 - accuracy: 0.4984
10912/25000 [============>.................] - ETA: 35s - loss: 7.6891 - accuracy: 0.4985
10944/25000 [============>.................] - ETA: 35s - loss: 7.6946 - accuracy: 0.4982
10976/25000 [============>.................] - ETA: 35s - loss: 7.6946 - accuracy: 0.4982
11008/25000 [============>.................] - ETA: 35s - loss: 7.6903 - accuracy: 0.4985
11040/25000 [============>.................] - ETA: 35s - loss: 7.6888 - accuracy: 0.4986
11072/25000 [============>.................] - ETA: 35s - loss: 7.6888 - accuracy: 0.4986
11104/25000 [============>.................] - ETA: 35s - loss: 7.6956 - accuracy: 0.4981
11136/25000 [============>.................] - ETA: 35s - loss: 7.6928 - accuracy: 0.4983
11168/25000 [============>.................] - ETA: 35s - loss: 7.6968 - accuracy: 0.4980
11200/25000 [============>.................] - ETA: 34s - loss: 7.6981 - accuracy: 0.4979
11232/25000 [============>.................] - ETA: 34s - loss: 7.6994 - accuracy: 0.4979
11264/25000 [============>.................] - ETA: 34s - loss: 7.6979 - accuracy: 0.4980
11296/25000 [============>.................] - ETA: 34s - loss: 7.6978 - accuracy: 0.4980
11328/25000 [============>.................] - ETA: 34s - loss: 7.6950 - accuracy: 0.4981
11360/25000 [============>.................] - ETA: 34s - loss: 7.7017 - accuracy: 0.4977
11392/25000 [============>.................] - ETA: 34s - loss: 7.6949 - accuracy: 0.4982
11424/25000 [============>.................] - ETA: 34s - loss: 7.6988 - accuracy: 0.4979
11456/25000 [============>.................] - ETA: 34s - loss: 7.7028 - accuracy: 0.4976
11488/25000 [============>.................] - ETA: 34s - loss: 7.7093 - accuracy: 0.4972
11520/25000 [============>.................] - ETA: 34s - loss: 7.7092 - accuracy: 0.4972
11552/25000 [============>.................] - ETA: 34s - loss: 7.7078 - accuracy: 0.4973
11584/25000 [============>.................] - ETA: 33s - loss: 7.7077 - accuracy: 0.4973
11616/25000 [============>.................] - ETA: 33s - loss: 7.7036 - accuracy: 0.4976
11648/25000 [============>.................] - ETA: 33s - loss: 7.7087 - accuracy: 0.4973
11680/25000 [=============>................] - ETA: 33s - loss: 7.7099 - accuracy: 0.4972
11712/25000 [=============>................] - ETA: 33s - loss: 7.7124 - accuracy: 0.4970
11744/25000 [=============>................] - ETA: 33s - loss: 7.7110 - accuracy: 0.4971
11776/25000 [=============>................] - ETA: 33s - loss: 7.7109 - accuracy: 0.4971
11808/25000 [=============>................] - ETA: 33s - loss: 7.7056 - accuracy: 0.4975
11840/25000 [=============>................] - ETA: 33s - loss: 7.7055 - accuracy: 0.4975
11872/25000 [=============>................] - ETA: 33s - loss: 7.7054 - accuracy: 0.4975
11904/25000 [=============>................] - ETA: 33s - loss: 7.7078 - accuracy: 0.4973
11936/25000 [=============>................] - ETA: 33s - loss: 7.7090 - accuracy: 0.4972
11968/25000 [=============>................] - ETA: 32s - loss: 7.7038 - accuracy: 0.4976
12000/25000 [=============>................] - ETA: 32s - loss: 7.6998 - accuracy: 0.4978
12032/25000 [=============>................] - ETA: 32s - loss: 7.7049 - accuracy: 0.4975
12064/25000 [=============>................] - ETA: 32s - loss: 7.7047 - accuracy: 0.4975
12096/25000 [=============>................] - ETA: 32s - loss: 7.7072 - accuracy: 0.4974
12128/25000 [=============>................] - ETA: 32s - loss: 7.7096 - accuracy: 0.4972
12160/25000 [=============>................] - ETA: 32s - loss: 7.7082 - accuracy: 0.4973
12192/25000 [=============>................] - ETA: 32s - loss: 7.7031 - accuracy: 0.4976
12224/25000 [=============>................] - ETA: 32s - loss: 7.7017 - accuracy: 0.4977
12256/25000 [=============>................] - ETA: 32s - loss: 7.6966 - accuracy: 0.4980
12288/25000 [=============>................] - ETA: 32s - loss: 7.6953 - accuracy: 0.4981
12320/25000 [=============>................] - ETA: 32s - loss: 7.6965 - accuracy: 0.4981
12352/25000 [=============>................] - ETA: 31s - loss: 7.6977 - accuracy: 0.4980
12384/25000 [=============>................] - ETA: 31s - loss: 7.6976 - accuracy: 0.4980
12416/25000 [=============>................] - ETA: 31s - loss: 7.6987 - accuracy: 0.4979
12448/25000 [=============>................] - ETA: 31s - loss: 7.6986 - accuracy: 0.4979
12480/25000 [=============>................] - ETA: 31s - loss: 7.6998 - accuracy: 0.4978
12512/25000 [==============>...............] - ETA: 31s - loss: 7.7046 - accuracy: 0.4975
12544/25000 [==============>...............] - ETA: 31s - loss: 7.7070 - accuracy: 0.4974
12576/25000 [==============>...............] - ETA: 31s - loss: 7.7142 - accuracy: 0.4969
12608/25000 [==============>...............] - ETA: 31s - loss: 7.7177 - accuracy: 0.4967
12640/25000 [==============>...............] - ETA: 31s - loss: 7.7115 - accuracy: 0.4971
12672/25000 [==============>...............] - ETA: 31s - loss: 7.7138 - accuracy: 0.4969
12704/25000 [==============>...............] - ETA: 31s - loss: 7.7113 - accuracy: 0.4971
12736/25000 [==============>...............] - ETA: 30s - loss: 7.7136 - accuracy: 0.4969
12768/25000 [==============>...............] - ETA: 30s - loss: 7.7147 - accuracy: 0.4969
12800/25000 [==============>...............] - ETA: 30s - loss: 7.7097 - accuracy: 0.4972
12832/25000 [==============>...............] - ETA: 30s - loss: 7.7013 - accuracy: 0.4977
12864/25000 [==============>...............] - ETA: 30s - loss: 7.7024 - accuracy: 0.4977
12896/25000 [==============>...............] - ETA: 30s - loss: 7.6987 - accuracy: 0.4979
12928/25000 [==============>...............] - ETA: 30s - loss: 7.7022 - accuracy: 0.4977
12960/25000 [==============>...............] - ETA: 30s - loss: 7.7021 - accuracy: 0.4977
12992/25000 [==============>...............] - ETA: 30s - loss: 7.7032 - accuracy: 0.4976
13024/25000 [==============>...............] - ETA: 30s - loss: 7.7043 - accuracy: 0.4975
13056/25000 [==============>...............] - ETA: 30s - loss: 7.7019 - accuracy: 0.4977
13088/25000 [==============>...............] - ETA: 30s - loss: 7.7041 - accuracy: 0.4976
13120/25000 [==============>...............] - ETA: 30s - loss: 7.7064 - accuracy: 0.4974
13152/25000 [==============>...............] - ETA: 29s - loss: 7.6993 - accuracy: 0.4979
13184/25000 [==============>...............] - ETA: 29s - loss: 7.7003 - accuracy: 0.4978
13216/25000 [==============>...............] - ETA: 29s - loss: 7.6979 - accuracy: 0.4980
13248/25000 [==============>...............] - ETA: 29s - loss: 7.7002 - accuracy: 0.4978
13280/25000 [==============>...............] - ETA: 29s - loss: 7.7013 - accuracy: 0.4977
13312/25000 [==============>...............] - ETA: 29s - loss: 7.7023 - accuracy: 0.4977
13344/25000 [===============>..............] - ETA: 29s - loss: 7.7011 - accuracy: 0.4978
13376/25000 [===============>..............] - ETA: 29s - loss: 7.7010 - accuracy: 0.4978
13408/25000 [===============>..............] - ETA: 29s - loss: 7.6964 - accuracy: 0.4981
13440/25000 [===============>..............] - ETA: 29s - loss: 7.6929 - accuracy: 0.4983
13472/25000 [===============>..............] - ETA: 29s - loss: 7.6917 - accuracy: 0.4984
13504/25000 [===============>..............] - ETA: 29s - loss: 7.6927 - accuracy: 0.4983
13536/25000 [===============>..............] - ETA: 28s - loss: 7.6938 - accuracy: 0.4982
13568/25000 [===============>..............] - ETA: 28s - loss: 7.6915 - accuracy: 0.4984
13600/25000 [===============>..............] - ETA: 28s - loss: 7.6937 - accuracy: 0.4982
13632/25000 [===============>..............] - ETA: 28s - loss: 7.6970 - accuracy: 0.4980
13664/25000 [===============>..............] - ETA: 28s - loss: 7.6936 - accuracy: 0.4982
13696/25000 [===============>..............] - ETA: 28s - loss: 7.6890 - accuracy: 0.4985
13728/25000 [===============>..............] - ETA: 28s - loss: 7.6878 - accuracy: 0.4986
13760/25000 [===============>..............] - ETA: 28s - loss: 7.6878 - accuracy: 0.4986
13792/25000 [===============>..............] - ETA: 28s - loss: 7.6811 - accuracy: 0.4991
13824/25000 [===============>..............] - ETA: 28s - loss: 7.6788 - accuracy: 0.4992
13856/25000 [===============>..............] - ETA: 28s - loss: 7.6810 - accuracy: 0.4991
13888/25000 [===============>..............] - ETA: 28s - loss: 7.6810 - accuracy: 0.4991
13920/25000 [===============>..............] - ETA: 27s - loss: 7.6842 - accuracy: 0.4989
13952/25000 [===============>..............] - ETA: 27s - loss: 7.6809 - accuracy: 0.4991
13984/25000 [===============>..............] - ETA: 27s - loss: 7.6798 - accuracy: 0.4991
14016/25000 [===============>..............] - ETA: 27s - loss: 7.6732 - accuracy: 0.4996
14048/25000 [===============>..............] - ETA: 27s - loss: 7.6699 - accuracy: 0.4998
14080/25000 [===============>..............] - ETA: 27s - loss: 7.6688 - accuracy: 0.4999
14112/25000 [===============>..............] - ETA: 27s - loss: 7.6666 - accuracy: 0.5000
14144/25000 [===============>..............] - ETA: 27s - loss: 7.6666 - accuracy: 0.5000
14176/25000 [================>.............] - ETA: 27s - loss: 7.6623 - accuracy: 0.5003
14208/25000 [================>.............] - ETA: 27s - loss: 7.6612 - accuracy: 0.5004
14240/25000 [================>.............] - ETA: 27s - loss: 7.6634 - accuracy: 0.5002
14272/25000 [================>.............] - ETA: 27s - loss: 7.6602 - accuracy: 0.5004
14304/25000 [================>.............] - ETA: 26s - loss: 7.6677 - accuracy: 0.4999
14336/25000 [================>.............] - ETA: 26s - loss: 7.6645 - accuracy: 0.5001
14368/25000 [================>.............] - ETA: 26s - loss: 7.6645 - accuracy: 0.5001
14400/25000 [================>.............] - ETA: 26s - loss: 7.6624 - accuracy: 0.5003
14432/25000 [================>.............] - ETA: 26s - loss: 7.6571 - accuracy: 0.5006
14464/25000 [================>.............] - ETA: 26s - loss: 7.6528 - accuracy: 0.5009
14496/25000 [================>.............] - ETA: 26s - loss: 7.6539 - accuracy: 0.5008
14528/25000 [================>.............] - ETA: 26s - loss: 7.6508 - accuracy: 0.5010
14560/25000 [================>.............] - ETA: 26s - loss: 7.6487 - accuracy: 0.5012
14592/25000 [================>.............] - ETA: 26s - loss: 7.6456 - accuracy: 0.5014
14624/25000 [================>.............] - ETA: 26s - loss: 7.6436 - accuracy: 0.5015
14656/25000 [================>.............] - ETA: 26s - loss: 7.6478 - accuracy: 0.5012
14688/25000 [================>.............] - ETA: 26s - loss: 7.6530 - accuracy: 0.5009
14720/25000 [================>.............] - ETA: 25s - loss: 7.6572 - accuracy: 0.5006
14752/25000 [================>.............] - ETA: 25s - loss: 7.6541 - accuracy: 0.5008
14784/25000 [================>.............] - ETA: 25s - loss: 7.6562 - accuracy: 0.5007
14816/25000 [================>.............] - ETA: 25s - loss: 7.6552 - accuracy: 0.5007
14848/25000 [================>.............] - ETA: 25s - loss: 7.6532 - accuracy: 0.5009
14880/25000 [================>.............] - ETA: 25s - loss: 7.6543 - accuracy: 0.5008
14912/25000 [================>.............] - ETA: 25s - loss: 7.6543 - accuracy: 0.5008
14944/25000 [================>.............] - ETA: 25s - loss: 7.6492 - accuracy: 0.5011
14976/25000 [================>.............] - ETA: 25s - loss: 7.6492 - accuracy: 0.5011
15008/25000 [=================>............] - ETA: 25s - loss: 7.6462 - accuracy: 0.5013
15040/25000 [=================>............] - ETA: 25s - loss: 7.6442 - accuracy: 0.5015
15072/25000 [=================>............] - ETA: 25s - loss: 7.6442 - accuracy: 0.5015
15104/25000 [=================>............] - ETA: 24s - loss: 7.6443 - accuracy: 0.5015
15136/25000 [=================>............] - ETA: 24s - loss: 7.6464 - accuracy: 0.5013
15168/25000 [=================>............] - ETA: 24s - loss: 7.6464 - accuracy: 0.5013
15200/25000 [=================>............] - ETA: 24s - loss: 7.6475 - accuracy: 0.5013
15232/25000 [=================>............] - ETA: 24s - loss: 7.6495 - accuracy: 0.5011
15264/25000 [=================>............] - ETA: 24s - loss: 7.6495 - accuracy: 0.5011
15296/25000 [=================>............] - ETA: 24s - loss: 7.6526 - accuracy: 0.5009
15328/25000 [=================>............] - ETA: 24s - loss: 7.6506 - accuracy: 0.5010
15360/25000 [=================>............] - ETA: 24s - loss: 7.6506 - accuracy: 0.5010
15392/25000 [=================>............] - ETA: 24s - loss: 7.6497 - accuracy: 0.5011
15424/25000 [=================>............] - ETA: 24s - loss: 7.6487 - accuracy: 0.5012
15456/25000 [=================>............] - ETA: 24s - loss: 7.6488 - accuracy: 0.5012
15488/25000 [=================>............] - ETA: 23s - loss: 7.6488 - accuracy: 0.5012
15520/25000 [=================>............] - ETA: 23s - loss: 7.6449 - accuracy: 0.5014
15552/25000 [=================>............] - ETA: 23s - loss: 7.6430 - accuracy: 0.5015
15584/25000 [=================>............] - ETA: 23s - loss: 7.6381 - accuracy: 0.5019
15616/25000 [=================>............] - ETA: 23s - loss: 7.6421 - accuracy: 0.5016
15648/25000 [=================>............] - ETA: 23s - loss: 7.6402 - accuracy: 0.5017
15680/25000 [=================>............] - ETA: 23s - loss: 7.6422 - accuracy: 0.5016
15712/25000 [=================>............] - ETA: 23s - loss: 7.6451 - accuracy: 0.5014
15744/25000 [=================>............] - ETA: 23s - loss: 7.6471 - accuracy: 0.5013
15776/25000 [=================>............] - ETA: 23s - loss: 7.6433 - accuracy: 0.5015
15808/25000 [=================>............] - ETA: 23s - loss: 7.6433 - accuracy: 0.5015
15840/25000 [==================>...........] - ETA: 23s - loss: 7.6434 - accuracy: 0.5015
15872/25000 [==================>...........] - ETA: 22s - loss: 7.6434 - accuracy: 0.5015
15904/25000 [==================>...........] - ETA: 22s - loss: 7.6425 - accuracy: 0.5016
15936/25000 [==================>...........] - ETA: 22s - loss: 7.6406 - accuracy: 0.5017
15968/25000 [==================>...........] - ETA: 22s - loss: 7.6397 - accuracy: 0.5018
16000/25000 [==================>...........] - ETA: 22s - loss: 7.6407 - accuracy: 0.5017
16032/25000 [==================>...........] - ETA: 22s - loss: 7.6446 - accuracy: 0.5014
16064/25000 [==================>...........] - ETA: 22s - loss: 7.6456 - accuracy: 0.5014
16096/25000 [==================>...........] - ETA: 22s - loss: 7.6409 - accuracy: 0.5017
16128/25000 [==================>...........] - ETA: 22s - loss: 7.6381 - accuracy: 0.5019
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6382 - accuracy: 0.5019
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6392 - accuracy: 0.5018
16224/25000 [==================>...........] - ETA: 22s - loss: 7.6383 - accuracy: 0.5018
16256/25000 [==================>...........] - ETA: 21s - loss: 7.6364 - accuracy: 0.5020
16288/25000 [==================>...........] - ETA: 21s - loss: 7.6337 - accuracy: 0.5021
16320/25000 [==================>...........] - ETA: 21s - loss: 7.6337 - accuracy: 0.5021
16352/25000 [==================>...........] - ETA: 21s - loss: 7.6310 - accuracy: 0.5023
16384/25000 [==================>...........] - ETA: 21s - loss: 7.6301 - accuracy: 0.5024
16416/25000 [==================>...........] - ETA: 21s - loss: 7.6330 - accuracy: 0.5022
16448/25000 [==================>...........] - ETA: 21s - loss: 7.6359 - accuracy: 0.5020
16480/25000 [==================>...........] - ETA: 21s - loss: 7.6396 - accuracy: 0.5018
16512/25000 [==================>...........] - ETA: 21s - loss: 7.6415 - accuracy: 0.5016
16544/25000 [==================>...........] - ETA: 21s - loss: 7.6425 - accuracy: 0.5016
16576/25000 [==================>...........] - ETA: 21s - loss: 7.6426 - accuracy: 0.5016
16608/25000 [==================>...........] - ETA: 21s - loss: 7.6463 - accuracy: 0.5013
16640/25000 [==================>...........] - ETA: 20s - loss: 7.6436 - accuracy: 0.5015
16672/25000 [===================>..........] - ETA: 20s - loss: 7.6399 - accuracy: 0.5017
16704/25000 [===================>..........] - ETA: 20s - loss: 7.6428 - accuracy: 0.5016
16736/25000 [===================>..........] - ETA: 20s - loss: 7.6437 - accuracy: 0.5015
16768/25000 [===================>..........] - ETA: 20s - loss: 7.6456 - accuracy: 0.5014
16800/25000 [===================>..........] - ETA: 20s - loss: 7.6456 - accuracy: 0.5014
16832/25000 [===================>..........] - ETA: 20s - loss: 7.6457 - accuracy: 0.5014
16864/25000 [===================>..........] - ETA: 20s - loss: 7.6421 - accuracy: 0.5016
16896/25000 [===================>..........] - ETA: 20s - loss: 7.6448 - accuracy: 0.5014
16928/25000 [===================>..........] - ETA: 20s - loss: 7.6458 - accuracy: 0.5014
16960/25000 [===================>..........] - ETA: 20s - loss: 7.6404 - accuracy: 0.5017
16992/25000 [===================>..........] - ETA: 20s - loss: 7.6432 - accuracy: 0.5015
17024/25000 [===================>..........] - ETA: 20s - loss: 7.6441 - accuracy: 0.5015
17056/25000 [===================>..........] - ETA: 19s - loss: 7.6432 - accuracy: 0.5015
17088/25000 [===================>..........] - ETA: 19s - loss: 7.6451 - accuracy: 0.5014
17120/25000 [===================>..........] - ETA: 19s - loss: 7.6460 - accuracy: 0.5013
17152/25000 [===================>..........] - ETA: 19s - loss: 7.6505 - accuracy: 0.5010
17184/25000 [===================>..........] - ETA: 19s - loss: 7.6497 - accuracy: 0.5011
17216/25000 [===================>..........] - ETA: 19s - loss: 7.6515 - accuracy: 0.5010
17248/25000 [===================>..........] - ETA: 19s - loss: 7.6533 - accuracy: 0.5009
17280/25000 [===================>..........] - ETA: 19s - loss: 7.6560 - accuracy: 0.5007
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6551 - accuracy: 0.5008
17344/25000 [===================>..........] - ETA: 19s - loss: 7.6516 - accuracy: 0.5010
17376/25000 [===================>..........] - ETA: 19s - loss: 7.6516 - accuracy: 0.5010
17408/25000 [===================>..........] - ETA: 19s - loss: 7.6525 - accuracy: 0.5009
17440/25000 [===================>..........] - ETA: 19s - loss: 7.6526 - accuracy: 0.5009
17472/25000 [===================>..........] - ETA: 18s - loss: 7.6508 - accuracy: 0.5010
17504/25000 [====================>.........] - ETA: 18s - loss: 7.6535 - accuracy: 0.5009
17536/25000 [====================>.........] - ETA: 18s - loss: 7.6544 - accuracy: 0.5008
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6518 - accuracy: 0.5010
17600/25000 [====================>.........] - ETA: 18s - loss: 7.6544 - accuracy: 0.5008
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6527 - accuracy: 0.5009
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6493 - accuracy: 0.5011
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6502 - accuracy: 0.5011
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6502 - accuracy: 0.5011
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6511 - accuracy: 0.5010
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6485 - accuracy: 0.5012
17824/25000 [====================>.........] - ETA: 18s - loss: 7.6503 - accuracy: 0.5011
17856/25000 [====================>.........] - ETA: 17s - loss: 7.6503 - accuracy: 0.5011
17888/25000 [====================>.........] - ETA: 17s - loss: 7.6529 - accuracy: 0.5009
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6581 - accuracy: 0.5006
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6589 - accuracy: 0.5005
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6547 - accuracy: 0.5008
18016/25000 [====================>.........] - ETA: 17s - loss: 7.6539 - accuracy: 0.5008
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6496 - accuracy: 0.5011
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6480 - accuracy: 0.5012
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6531 - accuracy: 0.5009
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6565 - accuracy: 0.5007
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6590 - accuracy: 0.5005
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6599 - accuracy: 0.5004
18240/25000 [====================>.........] - ETA: 17s - loss: 7.6582 - accuracy: 0.5005
18272/25000 [====================>.........] - ETA: 16s - loss: 7.6582 - accuracy: 0.5005
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6608 - accuracy: 0.5004
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6616 - accuracy: 0.5003
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6616 - accuracy: 0.5003
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6641 - accuracy: 0.5002
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6658 - accuracy: 0.5001
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6658 - accuracy: 0.5001
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6641 - accuracy: 0.5002
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6617 - accuracy: 0.5003
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6683 - accuracy: 0.4999
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6658 - accuracy: 0.5001
18624/25000 [=====================>........] - ETA: 16s - loss: 7.6658 - accuracy: 0.5001
18656/25000 [=====================>........] - ETA: 15s - loss: 7.6633 - accuracy: 0.5002
18688/25000 [=====================>........] - ETA: 15s - loss: 7.6609 - accuracy: 0.5004
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6633 - accuracy: 0.5002
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6642 - accuracy: 0.5002
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6642 - accuracy: 0.5002
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6682 - accuracy: 0.4999
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6682 - accuracy: 0.4999
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6666 - accuracy: 0.5000
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6682 - accuracy: 0.4999
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6690 - accuracy: 0.4998
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6707 - accuracy: 0.4997
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6707 - accuracy: 0.4997
19040/25000 [=====================>........] - ETA: 14s - loss: 7.6731 - accuracy: 0.4996
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6706 - accuracy: 0.4997
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6722 - accuracy: 0.4996
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6698 - accuracy: 0.4998
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6642 - accuracy: 0.5002
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6610 - accuracy: 0.5004
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6634 - accuracy: 0.5002
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6658 - accuracy: 0.5001
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6634 - accuracy: 0.5002
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6611 - accuracy: 0.5004
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6627 - accuracy: 0.5003
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6642 - accuracy: 0.5002
19424/25000 [======================>.......] - ETA: 14s - loss: 7.6643 - accuracy: 0.5002
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6627 - accuracy: 0.5003
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6611 - accuracy: 0.5004
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6595 - accuracy: 0.5005
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6596 - accuracy: 0.5005
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6549 - accuracy: 0.5008
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6557 - accuracy: 0.5007
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6565 - accuracy: 0.5007
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6565 - accuracy: 0.5007
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6596 - accuracy: 0.5005
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6589 - accuracy: 0.5005
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6573 - accuracy: 0.5006
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6527 - accuracy: 0.5009
19840/25000 [======================>.......] - ETA: 12s - loss: 7.6558 - accuracy: 0.5007
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6604 - accuracy: 0.5004
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6574 - accuracy: 0.5006
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6582 - accuracy: 0.5006
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6566 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6582 - accuracy: 0.5005
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6536 - accuracy: 0.5008
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6567 - accuracy: 0.5006
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6552 - accuracy: 0.5007
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6529 - accuracy: 0.5009
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6499 - accuracy: 0.5011
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6522 - accuracy: 0.5009
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6537 - accuracy: 0.5008
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6538 - accuracy: 0.5008
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6530 - accuracy: 0.5009
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6545 - accuracy: 0.5008
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6538 - accuracy: 0.5008
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6553 - accuracy: 0.5007
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6561 - accuracy: 0.5007
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6569 - accuracy: 0.5006
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6591 - accuracy: 0.5005
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6576 - accuracy: 0.5006
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6592 - accuracy: 0.5005
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6599 - accuracy: 0.5004
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6607 - accuracy: 0.5004
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6622 - accuracy: 0.5003
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6607 - accuracy: 0.5004
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6637 - accuracy: 0.5002
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6659 - accuracy: 0.5000
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6688 - accuracy: 0.4999
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6696 - accuracy: 0.4998
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6688 - accuracy: 0.4999
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6696 - accuracy: 0.4998
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6703 - accuracy: 0.4998
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6747 - accuracy: 0.4995
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6695 - accuracy: 0.4998
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6681 - accuracy: 0.4999 
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6695 - accuracy: 0.4998
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6717 - accuracy: 0.4997
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6732 - accuracy: 0.4996
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6710 - accuracy: 0.4997
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6702 - accuracy: 0.4998
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6695 - accuracy: 0.4998
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6702 - accuracy: 0.4998
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6688 - accuracy: 0.4999
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6681 - accuracy: 0.4999
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6688 - accuracy: 0.4999
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6645 - accuracy: 0.5001
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6638 - accuracy: 0.5002
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6645 - accuracy: 0.5001
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6652 - accuracy: 0.5001
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6645 - accuracy: 0.5001
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6631 - accuracy: 0.5002
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6638 - accuracy: 0.5002
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6652 - accuracy: 0.5001
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6659 - accuracy: 0.5000
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6652 - accuracy: 0.5001
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6631 - accuracy: 0.5002
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6638 - accuracy: 0.5002
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6638 - accuracy: 0.5002
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6645 - accuracy: 0.5001
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6645 - accuracy: 0.5001
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6610 - accuracy: 0.5004
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6596 - accuracy: 0.5005
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6575 - accuracy: 0.5006
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6582 - accuracy: 0.5005
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6590 - accuracy: 0.5005
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6569 - accuracy: 0.5006
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6569 - accuracy: 0.5006
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6534 - accuracy: 0.5009
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6555 - accuracy: 0.5007
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6562 - accuracy: 0.5007
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6570 - accuracy: 0.5006
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6583 - accuracy: 0.5005
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6590 - accuracy: 0.5005
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6577 - accuracy: 0.5006
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6536 - accuracy: 0.5008
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6557 - accuracy: 0.5007
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6537 - accuracy: 0.5008
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6509 - accuracy: 0.5010
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6476 - accuracy: 0.5012
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6462 - accuracy: 0.5013
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6456 - accuracy: 0.5014
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6456 - accuracy: 0.5014
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6490 - accuracy: 0.5011
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6477 - accuracy: 0.5012
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6504 - accuracy: 0.5011
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6525 - accuracy: 0.5009
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6572 - accuracy: 0.5006
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6572 - accuracy: 0.5006
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6586 - accuracy: 0.5005
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6579 - accuracy: 0.5006
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6606 - accuracy: 0.5004
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6646 - accuracy: 0.5001
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6626 - accuracy: 0.5003
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6600 - accuracy: 0.5004
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6613 - accuracy: 0.5003
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6600 - accuracy: 0.5004
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6626 - accuracy: 0.5003
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6607 - accuracy: 0.5004
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6600 - accuracy: 0.5004
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6607 - accuracy: 0.5004
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6640 - accuracy: 0.5002
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6692 - accuracy: 0.4998
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6706 - accuracy: 0.4997
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6712 - accuracy: 0.4997
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6719 - accuracy: 0.4997
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6738 - accuracy: 0.4995
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6738 - accuracy: 0.4995
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6751 - accuracy: 0.4994
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6757 - accuracy: 0.4994
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6744 - accuracy: 0.4995
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6744 - accuracy: 0.4995
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6738 - accuracy: 0.4995
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6737 - accuracy: 0.4995
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6750 - accuracy: 0.4995
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6744 - accuracy: 0.4995
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6737 - accuracy: 0.4995
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6750 - accuracy: 0.4995
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6737 - accuracy: 0.4995
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6750 - accuracy: 0.4995
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6724 - accuracy: 0.4996
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6749 - accuracy: 0.4995
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6711 - accuracy: 0.4997
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6723 - accuracy: 0.4996
24192/25000 [============================>.] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
24224/25000 [============================>.] - ETA: 1s - loss: 7.6748 - accuracy: 0.4995
24256/25000 [============================>.] - ETA: 1s - loss: 7.6736 - accuracy: 0.4995
24288/25000 [============================>.] - ETA: 1s - loss: 7.6736 - accuracy: 0.4995
24320/25000 [============================>.] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24352/25000 [============================>.] - ETA: 1s - loss: 7.6742 - accuracy: 0.4995
24384/25000 [============================>.] - ETA: 1s - loss: 7.6754 - accuracy: 0.4994
24416/25000 [============================>.] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24448/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24480/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24512/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24544/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24576/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24608/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24640/25000 [============================>.] - ETA: 0s - loss: 7.6704 - accuracy: 0.4998
24672/25000 [============================>.] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
24704/25000 [============================>.] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
24736/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24768/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24800/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24832/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24864/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24896/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24928/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24960/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24992/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f657a351a90> 

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
 [ 0.06910693  0.03611922 -0.02392675 -0.00233308  0.13038585 -0.06288359]
 [ 0.02723297  0.01734905  0.24580958 -0.02024018 -0.03736066  0.08731338]
 [ 0.05520063  0.1563978   0.18761533  0.07518766  0.10245229 -0.09936244]
 [ 0.17774884  0.11448228 -0.13108577  0.04458354 -0.02181875 -0.08142143]
 [-0.29576617  0.05447529  0.09443319  0.29916945 -0.22763799  0.15086837]
 [ 0.21396993  0.34682044  0.29324821 -0.6096527   0.31283623  0.22328338]
 [-0.23931675  0.54980898 -0.11345342  0.32864293 -0.18988468  0.34937766]
 [ 0.55028981  0.41960433 -0.02283287  0.26027811  0.20554711  0.0190306 ]
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
{'loss': 0.47149988263845444, 'loss_history': []}

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
{'loss': 0.5271644443273544, 'loss_history': []}

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
 40%|████      | 2/5 [00:49<01:14, 24.71s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.04129664333796585, 'embedding_size_factor': 1.2203576343897926, 'layers.choice': 3, 'learning_rate': 0.0001556842888221405, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 8.513253197364536e-10} and reward: 0.3632
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xa5$\xd5i$liX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\x86\x95\xba\x12\x0c\xa2X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?$g\xe5\xdb\xa4\xda\xa5X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\r@V2E-\xb3u.' and reward: 0.3632
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xa5$\xd5i$liX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\x86\x95\xba\x12\x0c\xa2X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?$g\xe5\xdb\xa4\xda\xa5X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\r@V2E-\xb3u.' and reward: 0.3632
 60%|██████    | 3/5 [01:39<01:04, 32.44s/it] 60%|██████    | 3/5 [01:39<01:06, 33.30s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.2018174116945694, 'embedding_size_factor': 0.7902687199824168, 'layers.choice': 0, 'learning_rate': 0.0005450053632810771, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 2.1161862471164115e-08} and reward: 0.363
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xc9\xd5''~\xea\xe6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe9I\xe1\xa0l\nVX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?A\xdb\xd6\x1b\x10\\\xa5X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>V\xb8\xed\xaf\xc2&\x9bu." and reward: 0.363
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xc9\xd5''~\xea\xe6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe9I\xe1\xa0l\nVX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?A\xdb\xd6\x1b\x10\\\xa5X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>V\xb8\xed\xaf\xc2&\x9bu." and reward: 0.363
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 150.2322826385498
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -32.78s of remaining time.
Ensemble size: 12
Ensemble weights: 
[0.75       0.08333333 0.16666667]
	0.3898	 = Validation accuracy score
	1.06s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 153.87s ...
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

