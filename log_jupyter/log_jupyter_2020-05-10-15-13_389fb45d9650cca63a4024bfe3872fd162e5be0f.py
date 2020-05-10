
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_jupyter GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/389fb45d9650cca63a4024bfe3872fd162e5be0f', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '389fb45d9650cca63a4024bfe3872fd162e5be0f', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/389fb45d9650cca63a4024bfe3872fd162e5be0f

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/389fb45d9650cca63a4024bfe3872fd162e5be0f

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
 1990656/17464789 [==>...........................] - ETA: 0s
 7102464/17464789 [===========>..................] - ETA: 0s
16515072/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 15:13:34.945641: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 15:13:34.960036: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-10 15:13:34.960194: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564ac02241e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 15:13:34.960210: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 6:01 - loss: 7.1875 - accuracy: 0.5312
   64/25000 [..............................] - ETA: 3:31 - loss: 7.9062 - accuracy: 0.4844
   96/25000 [..............................] - ETA: 2:40 - loss: 7.9861 - accuracy: 0.4792
  128/25000 [..............................] - ETA: 2:15 - loss: 8.5052 - accuracy: 0.4453
  160/25000 [..............................] - ETA: 1:59 - loss: 8.6249 - accuracy: 0.4375
  192/25000 [..............................] - ETA: 1:52 - loss: 8.4652 - accuracy: 0.4479
  224/25000 [..............................] - ETA: 1:44 - loss: 8.4880 - accuracy: 0.4464
  256/25000 [..............................] - ETA: 1:39 - loss: 8.5651 - accuracy: 0.4414
  288/25000 [..............................] - ETA: 1:34 - loss: 8.4652 - accuracy: 0.4479
  320/25000 [..............................] - ETA: 1:31 - loss: 8.3374 - accuracy: 0.4563
  352/25000 [..............................] - ETA: 1:28 - loss: 8.1458 - accuracy: 0.4688
  384/25000 [..............................] - ETA: 1:25 - loss: 8.0659 - accuracy: 0.4740
  416/25000 [..............................] - ETA: 1:23 - loss: 7.8141 - accuracy: 0.4904
  448/25000 [..............................] - ETA: 1:22 - loss: 7.8377 - accuracy: 0.4888
  480/25000 [..............................] - ETA: 1:20 - loss: 7.9541 - accuracy: 0.4812
  512/25000 [..............................] - ETA: 1:19 - loss: 7.9361 - accuracy: 0.4824
  544/25000 [..............................] - ETA: 1:18 - loss: 7.9767 - accuracy: 0.4798
  576/25000 [..............................] - ETA: 1:16 - loss: 8.0127 - accuracy: 0.4774
  608/25000 [..............................] - ETA: 1:16 - loss: 7.9945 - accuracy: 0.4786
  640/25000 [..............................] - ETA: 1:15 - loss: 8.0020 - accuracy: 0.4781
  672/25000 [..............................] - ETA: 1:14 - loss: 8.0089 - accuracy: 0.4777
  704/25000 [..............................] - ETA: 1:13 - loss: 8.1022 - accuracy: 0.4716
  736/25000 [..............................] - ETA: 1:12 - loss: 8.0833 - accuracy: 0.4728
  768/25000 [..............................] - ETA: 1:12 - loss: 8.0060 - accuracy: 0.4779
  800/25000 [..............................] - ETA: 1:11 - loss: 8.0691 - accuracy: 0.4737
  832/25000 [..............................] - ETA: 1:11 - loss: 8.0352 - accuracy: 0.4760
  864/25000 [>.............................] - ETA: 1:10 - loss: 8.0748 - accuracy: 0.4734
  896/25000 [>.............................] - ETA: 1:10 - loss: 7.9575 - accuracy: 0.4810
  928/25000 [>.............................] - ETA: 1:09 - loss: 7.9475 - accuracy: 0.4817
  960/25000 [>.............................] - ETA: 1:09 - loss: 7.9701 - accuracy: 0.4802
  992/25000 [>.............................] - ETA: 1:09 - loss: 7.9139 - accuracy: 0.4839
 1024/25000 [>.............................] - ETA: 1:08 - loss: 7.8613 - accuracy: 0.4873
 1056/25000 [>.............................] - ETA: 1:08 - loss: 7.8118 - accuracy: 0.4905
 1088/25000 [>.............................] - ETA: 1:08 - loss: 7.8216 - accuracy: 0.4899
 1120/25000 [>.............................] - ETA: 1:07 - loss: 7.8172 - accuracy: 0.4902
 1152/25000 [>.............................] - ETA: 1:07 - loss: 7.7997 - accuracy: 0.4913
 1184/25000 [>.............................] - ETA: 1:07 - loss: 7.8479 - accuracy: 0.4882
 1216/25000 [>.............................] - ETA: 1:06 - loss: 7.9062 - accuracy: 0.4844
 1248/25000 [>.............................] - ETA: 1:06 - loss: 7.9369 - accuracy: 0.4824
 1280/25000 [>.............................] - ETA: 1:06 - loss: 7.9541 - accuracy: 0.4812
 1312/25000 [>.............................] - ETA: 1:05 - loss: 7.9588 - accuracy: 0.4809
 1344/25000 [>.............................] - ETA: 1:05 - loss: 8.0203 - accuracy: 0.4769
 1376/25000 [>.............................] - ETA: 1:05 - loss: 8.0009 - accuracy: 0.4782
 1408/25000 [>.............................] - ETA: 1:05 - loss: 8.0587 - accuracy: 0.4744
 1440/25000 [>.............................] - ETA: 1:05 - loss: 8.0606 - accuracy: 0.4743
 1472/25000 [>.............................] - ETA: 1:05 - loss: 8.0520 - accuracy: 0.4749
 1504/25000 [>.............................] - ETA: 1:04 - loss: 8.0438 - accuracy: 0.4754
 1536/25000 [>.............................] - ETA: 1:04 - loss: 8.0959 - accuracy: 0.4720
 1568/25000 [>.............................] - ETA: 1:04 - loss: 8.0578 - accuracy: 0.4745
 1600/25000 [>.............................] - ETA: 1:04 - loss: 8.0979 - accuracy: 0.4719
 1632/25000 [>.............................] - ETA: 1:03 - loss: 8.0988 - accuracy: 0.4718
 1664/25000 [>.............................] - ETA: 1:03 - loss: 8.0813 - accuracy: 0.4730
 1696/25000 [=>............................] - ETA: 1:03 - loss: 8.0644 - accuracy: 0.4741
 1728/25000 [=>............................] - ETA: 1:03 - loss: 8.0304 - accuracy: 0.4763
 1760/25000 [=>............................] - ETA: 1:03 - loss: 8.0412 - accuracy: 0.4756
 1792/25000 [=>............................] - ETA: 1:03 - loss: 8.0345 - accuracy: 0.4760
 1824/25000 [=>............................] - ETA: 1:03 - loss: 8.0365 - accuracy: 0.4759
 1856/25000 [=>............................] - ETA: 1:02 - loss: 8.0384 - accuracy: 0.4758
 1888/25000 [=>............................] - ETA: 1:02 - loss: 8.0321 - accuracy: 0.4762
 1920/25000 [=>............................] - ETA: 1:02 - loss: 8.0260 - accuracy: 0.4766
 1952/25000 [=>............................] - ETA: 1:02 - loss: 8.0201 - accuracy: 0.4769
 1984/25000 [=>............................] - ETA: 1:02 - loss: 7.9835 - accuracy: 0.4793
 2016/25000 [=>............................] - ETA: 1:02 - loss: 7.9709 - accuracy: 0.4802
 2048/25000 [=>............................] - ETA: 1:01 - loss: 7.9436 - accuracy: 0.4819
 2080/25000 [=>............................] - ETA: 1:01 - loss: 7.9099 - accuracy: 0.4841
 2112/25000 [=>............................] - ETA: 1:01 - loss: 7.8917 - accuracy: 0.4853
 2144/25000 [=>............................] - ETA: 1:01 - loss: 7.8812 - accuracy: 0.4860
 2176/25000 [=>............................] - ETA: 1:01 - loss: 7.8992 - accuracy: 0.4848
 2208/25000 [=>............................] - ETA: 1:01 - loss: 7.8888 - accuracy: 0.4855
 2240/25000 [=>............................] - ETA: 1:01 - loss: 7.8925 - accuracy: 0.4853
 2272/25000 [=>............................] - ETA: 1:00 - loss: 7.9028 - accuracy: 0.4846
 2304/25000 [=>............................] - ETA: 1:00 - loss: 7.8729 - accuracy: 0.4865
 2336/25000 [=>............................] - ETA: 1:00 - loss: 7.8701 - accuracy: 0.4867
 2368/25000 [=>............................] - ETA: 1:00 - loss: 7.8479 - accuracy: 0.4882
 2400/25000 [=>............................] - ETA: 1:00 - loss: 7.8072 - accuracy: 0.4908
 2432/25000 [=>............................] - ETA: 1:00 - loss: 7.7801 - accuracy: 0.4926
 2464/25000 [=>............................] - ETA: 1:00 - loss: 7.8097 - accuracy: 0.4907
 2496/25000 [=>............................] - ETA: 1:00 - loss: 7.8141 - accuracy: 0.4904
 2528/25000 [==>...........................] - ETA: 59s - loss: 7.8001 - accuracy: 0.4913 
 2560/25000 [==>...........................] - ETA: 59s - loss: 7.8164 - accuracy: 0.4902
 2592/25000 [==>...........................] - ETA: 59s - loss: 7.8027 - accuracy: 0.4911
 2624/25000 [==>...........................] - ETA: 59s - loss: 7.7835 - accuracy: 0.4924
 2656/25000 [==>...........................] - ETA: 59s - loss: 7.7763 - accuracy: 0.4928
 2688/25000 [==>...........................] - ETA: 59s - loss: 7.7522 - accuracy: 0.4944
 2720/25000 [==>...........................] - ETA: 59s - loss: 7.7568 - accuracy: 0.4941
 2752/25000 [==>...........................] - ETA: 59s - loss: 7.7558 - accuracy: 0.4942
 2784/25000 [==>...........................] - ETA: 59s - loss: 7.7547 - accuracy: 0.4943
 2816/25000 [==>...........................] - ETA: 59s - loss: 7.7646 - accuracy: 0.4936
 2848/25000 [==>...........................] - ETA: 58s - loss: 7.7743 - accuracy: 0.4930
 2880/25000 [==>...........................] - ETA: 58s - loss: 7.7678 - accuracy: 0.4934
 2912/25000 [==>...........................] - ETA: 58s - loss: 7.7614 - accuracy: 0.4938
 2944/25000 [==>...........................] - ETA: 58s - loss: 7.7656 - accuracy: 0.4935
 2976/25000 [==>...........................] - ETA: 58s - loss: 7.7594 - accuracy: 0.4940
 3008/25000 [==>...........................] - ETA: 58s - loss: 7.7533 - accuracy: 0.4943
 3040/25000 [==>...........................] - ETA: 58s - loss: 7.7524 - accuracy: 0.4944
 3072/25000 [==>...........................] - ETA: 58s - loss: 7.7565 - accuracy: 0.4941
 3104/25000 [==>...........................] - ETA: 57s - loss: 7.7555 - accuracy: 0.4942
 3136/25000 [==>...........................] - ETA: 57s - loss: 7.7302 - accuracy: 0.4959
 3168/25000 [==>...........................] - ETA: 57s - loss: 7.7295 - accuracy: 0.4959
 3200/25000 [==>...........................] - ETA: 57s - loss: 7.7241 - accuracy: 0.4963
 3232/25000 [==>...........................] - ETA: 57s - loss: 7.7330 - accuracy: 0.4957
 3264/25000 [==>...........................] - ETA: 57s - loss: 7.7230 - accuracy: 0.4963
 3296/25000 [==>...........................] - ETA: 57s - loss: 7.7271 - accuracy: 0.4961
 3328/25000 [==>...........................] - ETA: 57s - loss: 7.7311 - accuracy: 0.4958
 3360/25000 [===>..........................] - ETA: 56s - loss: 7.7168 - accuracy: 0.4967
 3392/25000 [===>..........................] - ETA: 56s - loss: 7.7344 - accuracy: 0.4956
 3424/25000 [===>..........................] - ETA: 56s - loss: 7.7338 - accuracy: 0.4956
 3456/25000 [===>..........................] - ETA: 56s - loss: 7.7287 - accuracy: 0.4959
 3488/25000 [===>..........................] - ETA: 56s - loss: 7.7326 - accuracy: 0.4957
 3520/25000 [===>..........................] - ETA: 56s - loss: 7.7189 - accuracy: 0.4966
 3552/25000 [===>..........................] - ETA: 56s - loss: 7.7141 - accuracy: 0.4969
 3584/25000 [===>..........................] - ETA: 56s - loss: 7.7351 - accuracy: 0.4955
 3616/25000 [===>..........................] - ETA: 56s - loss: 7.7345 - accuracy: 0.4956
 3648/25000 [===>..........................] - ETA: 56s - loss: 7.7213 - accuracy: 0.4964
 3680/25000 [===>..........................] - ETA: 56s - loss: 7.7208 - accuracy: 0.4965
 3712/25000 [===>..........................] - ETA: 56s - loss: 7.7121 - accuracy: 0.4970
 3744/25000 [===>..........................] - ETA: 55s - loss: 7.7117 - accuracy: 0.4971
 3776/25000 [===>..........................] - ETA: 55s - loss: 7.7113 - accuracy: 0.4971
 3808/25000 [===>..........................] - ETA: 55s - loss: 7.7230 - accuracy: 0.4963
 3840/25000 [===>..........................] - ETA: 55s - loss: 7.7026 - accuracy: 0.4977
 3872/25000 [===>..........................] - ETA: 55s - loss: 7.6983 - accuracy: 0.4979
 3904/25000 [===>..........................] - ETA: 55s - loss: 7.6902 - accuracy: 0.4985
 3936/25000 [===>..........................] - ETA: 55s - loss: 7.6861 - accuracy: 0.4987
 3968/25000 [===>..........................] - ETA: 55s - loss: 7.6859 - accuracy: 0.4987
 4000/25000 [===>..........................] - ETA: 55s - loss: 7.6973 - accuracy: 0.4980
 4032/25000 [===>..........................] - ETA: 55s - loss: 7.6894 - accuracy: 0.4985
 4064/25000 [===>..........................] - ETA: 55s - loss: 7.7043 - accuracy: 0.4975
 4096/25000 [===>..........................] - ETA: 54s - loss: 7.7078 - accuracy: 0.4973
 4128/25000 [===>..........................] - ETA: 54s - loss: 7.7038 - accuracy: 0.4976
 4160/25000 [===>..........................] - ETA: 54s - loss: 7.7072 - accuracy: 0.4974
 4192/25000 [====>.........................] - ETA: 54s - loss: 7.7105 - accuracy: 0.4971
 4224/25000 [====>.........................] - ETA: 54s - loss: 7.7138 - accuracy: 0.4969
 4256/25000 [====>.........................] - ETA: 54s - loss: 7.7062 - accuracy: 0.4974
 4288/25000 [====>.........................] - ETA: 54s - loss: 7.6952 - accuracy: 0.4981
 4320/25000 [====>.........................] - ETA: 54s - loss: 7.7021 - accuracy: 0.4977
 4352/25000 [====>.........................] - ETA: 54s - loss: 7.7089 - accuracy: 0.4972
 4384/25000 [====>.........................] - ETA: 54s - loss: 7.7086 - accuracy: 0.4973
 4416/25000 [====>.........................] - ETA: 53s - loss: 7.7222 - accuracy: 0.4964
 4448/25000 [====>.........................] - ETA: 53s - loss: 7.7149 - accuracy: 0.4969
 4480/25000 [====>.........................] - ETA: 53s - loss: 7.7180 - accuracy: 0.4967
 4512/25000 [====>.........................] - ETA: 53s - loss: 7.7244 - accuracy: 0.4962
 4544/25000 [====>.........................] - ETA: 53s - loss: 7.7071 - accuracy: 0.4974
 4576/25000 [====>.........................] - ETA: 53s - loss: 7.7102 - accuracy: 0.4972
 4608/25000 [====>.........................] - ETA: 53s - loss: 7.7065 - accuracy: 0.4974
 4640/25000 [====>.........................] - ETA: 53s - loss: 7.7162 - accuracy: 0.4968
 4672/25000 [====>.........................] - ETA: 53s - loss: 7.7158 - accuracy: 0.4968
 4704/25000 [====>.........................] - ETA: 53s - loss: 7.7057 - accuracy: 0.4974
 4736/25000 [====>.........................] - ETA: 52s - loss: 7.7217 - accuracy: 0.4964
 4768/25000 [====>.........................] - ETA: 52s - loss: 7.7181 - accuracy: 0.4966
 4800/25000 [====>.........................] - ETA: 52s - loss: 7.7113 - accuracy: 0.4971
 4832/25000 [====>.........................] - ETA: 52s - loss: 7.7079 - accuracy: 0.4973
 4864/25000 [====>.........................] - ETA: 52s - loss: 7.6950 - accuracy: 0.4981
 4896/25000 [====>.........................] - ETA: 52s - loss: 7.6885 - accuracy: 0.4986
 4928/25000 [====>.........................] - ETA: 52s - loss: 7.6884 - accuracy: 0.4986
 4960/25000 [====>.........................] - ETA: 52s - loss: 7.6852 - accuracy: 0.4988
 4992/25000 [====>.........................] - ETA: 52s - loss: 7.6697 - accuracy: 0.4998
 5024/25000 [=====>........................] - ETA: 52s - loss: 7.6575 - accuracy: 0.5006
 5056/25000 [=====>........................] - ETA: 52s - loss: 7.6606 - accuracy: 0.5004
 5088/25000 [=====>........................] - ETA: 51s - loss: 7.6606 - accuracy: 0.5004
 5120/25000 [=====>........................] - ETA: 51s - loss: 7.6666 - accuracy: 0.5000
 5152/25000 [=====>........................] - ETA: 51s - loss: 7.6726 - accuracy: 0.4996
 5184/25000 [=====>........................] - ETA: 51s - loss: 7.6873 - accuracy: 0.4986
 5216/25000 [=====>........................] - ETA: 51s - loss: 7.6725 - accuracy: 0.4996
 5248/25000 [=====>........................] - ETA: 51s - loss: 7.6725 - accuracy: 0.4996
 5280/25000 [=====>........................] - ETA: 51s - loss: 7.6666 - accuracy: 0.5000
 5312/25000 [=====>........................] - ETA: 51s - loss: 7.6666 - accuracy: 0.5000
 5344/25000 [=====>........................] - ETA: 51s - loss: 7.6695 - accuracy: 0.4998
 5376/25000 [=====>........................] - ETA: 51s - loss: 7.6780 - accuracy: 0.4993
 5408/25000 [=====>........................] - ETA: 51s - loss: 7.6893 - accuracy: 0.4985
 5440/25000 [=====>........................] - ETA: 50s - loss: 7.6976 - accuracy: 0.4980
 5472/25000 [=====>........................] - ETA: 50s - loss: 7.7115 - accuracy: 0.4971
 5504/25000 [=====>........................] - ETA: 50s - loss: 7.7084 - accuracy: 0.4973
 5536/25000 [=====>........................] - ETA: 50s - loss: 7.7026 - accuracy: 0.4977
 5568/25000 [=====>........................] - ETA: 50s - loss: 7.6969 - accuracy: 0.4980
 5600/25000 [=====>........................] - ETA: 50s - loss: 7.6940 - accuracy: 0.4982
 5632/25000 [=====>........................] - ETA: 50s - loss: 7.6938 - accuracy: 0.4982
 5664/25000 [=====>........................] - ETA: 50s - loss: 7.6883 - accuracy: 0.4986
 5696/25000 [=====>........................] - ETA: 50s - loss: 7.6962 - accuracy: 0.4981
 5728/25000 [=====>........................] - ETA: 50s - loss: 7.6934 - accuracy: 0.4983
 5760/25000 [=====>........................] - ETA: 50s - loss: 7.6879 - accuracy: 0.4986
 5792/25000 [=====>........................] - ETA: 49s - loss: 7.6931 - accuracy: 0.4983
 5824/25000 [=====>........................] - ETA: 49s - loss: 7.6798 - accuracy: 0.4991
 5856/25000 [======>.......................] - ETA: 49s - loss: 7.6849 - accuracy: 0.4988
 5888/25000 [======>.......................] - ETA: 49s - loss: 7.6848 - accuracy: 0.4988
 5920/25000 [======>.......................] - ETA: 49s - loss: 7.6796 - accuracy: 0.4992
 5952/25000 [======>.......................] - ETA: 49s - loss: 7.6821 - accuracy: 0.4990
 5984/25000 [======>.......................] - ETA: 49s - loss: 7.6846 - accuracy: 0.4988
 6016/25000 [======>.......................] - ETA: 49s - loss: 7.6870 - accuracy: 0.4987
 6048/25000 [======>.......................] - ETA: 49s - loss: 7.6894 - accuracy: 0.4985
 6080/25000 [======>.......................] - ETA: 49s - loss: 7.6994 - accuracy: 0.4979
 6112/25000 [======>.......................] - ETA: 49s - loss: 7.7043 - accuracy: 0.4975
 6144/25000 [======>.......................] - ETA: 48s - loss: 7.7065 - accuracy: 0.4974
 6176/25000 [======>.......................] - ETA: 48s - loss: 7.7088 - accuracy: 0.4972
 6208/25000 [======>.......................] - ETA: 48s - loss: 7.7037 - accuracy: 0.4976
 6240/25000 [======>.......................] - ETA: 48s - loss: 7.7108 - accuracy: 0.4971
 6272/25000 [======>.......................] - ETA: 48s - loss: 7.7033 - accuracy: 0.4976
 6304/25000 [======>.......................] - ETA: 48s - loss: 7.7080 - accuracy: 0.4973
 6336/25000 [======>.......................] - ETA: 48s - loss: 7.7102 - accuracy: 0.4972
 6368/25000 [======>.......................] - ETA: 48s - loss: 7.7076 - accuracy: 0.4973
 6400/25000 [======>.......................] - ETA: 48s - loss: 7.6978 - accuracy: 0.4980
 6432/25000 [======>.......................] - ETA: 48s - loss: 7.6976 - accuracy: 0.4980
 6464/25000 [======>.......................] - ETA: 48s - loss: 7.6975 - accuracy: 0.4980
 6496/25000 [======>.......................] - ETA: 47s - loss: 7.6926 - accuracy: 0.4983
 6528/25000 [======>.......................] - ETA: 47s - loss: 7.6854 - accuracy: 0.4988
 6560/25000 [======>.......................] - ETA: 47s - loss: 7.6806 - accuracy: 0.4991
 6592/25000 [======>.......................] - ETA: 47s - loss: 7.6829 - accuracy: 0.4989
 6624/25000 [======>.......................] - ETA: 47s - loss: 7.6759 - accuracy: 0.4994
 6656/25000 [======>.......................] - ETA: 47s - loss: 7.6920 - accuracy: 0.4983
 6688/25000 [=======>......................] - ETA: 47s - loss: 7.6850 - accuracy: 0.4988
 6720/25000 [=======>......................] - ETA: 47s - loss: 7.6780 - accuracy: 0.4993
 6752/25000 [=======>......................] - ETA: 47s - loss: 7.6871 - accuracy: 0.4987
 6784/25000 [=======>......................] - ETA: 47s - loss: 7.6915 - accuracy: 0.4984
 6816/25000 [=======>......................] - ETA: 47s - loss: 7.6891 - accuracy: 0.4985
 6848/25000 [=======>......................] - ETA: 46s - loss: 7.6845 - accuracy: 0.4988
 6880/25000 [=======>......................] - ETA: 46s - loss: 7.6911 - accuracy: 0.4984
 6912/25000 [=======>......................] - ETA: 46s - loss: 7.6866 - accuracy: 0.4987
 6944/25000 [=======>......................] - ETA: 46s - loss: 7.6975 - accuracy: 0.4980
 6976/25000 [=======>......................] - ETA: 46s - loss: 7.6952 - accuracy: 0.4981
 7008/25000 [=======>......................] - ETA: 46s - loss: 7.6973 - accuracy: 0.4980
 7040/25000 [=======>......................] - ETA: 46s - loss: 7.6862 - accuracy: 0.4987
 7072/25000 [=======>......................] - ETA: 46s - loss: 7.6883 - accuracy: 0.4986
 7104/25000 [=======>......................] - ETA: 46s - loss: 7.6968 - accuracy: 0.4980
 7136/25000 [=======>......................] - ETA: 46s - loss: 7.6924 - accuracy: 0.4983
 7168/25000 [=======>......................] - ETA: 46s - loss: 7.6987 - accuracy: 0.4979
 7200/25000 [=======>......................] - ETA: 46s - loss: 7.6922 - accuracy: 0.4983
 7232/25000 [=======>......................] - ETA: 45s - loss: 7.6878 - accuracy: 0.4986
 7264/25000 [=======>......................] - ETA: 45s - loss: 7.6856 - accuracy: 0.4988
 7296/25000 [=======>......................] - ETA: 45s - loss: 7.6813 - accuracy: 0.4990
 7328/25000 [=======>......................] - ETA: 45s - loss: 7.6750 - accuracy: 0.4995
 7360/25000 [=======>......................] - ETA: 45s - loss: 7.6770 - accuracy: 0.4993
 7392/25000 [=======>......................] - ETA: 45s - loss: 7.6811 - accuracy: 0.4991
 7424/25000 [=======>......................] - ETA: 45s - loss: 7.6831 - accuracy: 0.4989
 7456/25000 [=======>......................] - ETA: 45s - loss: 7.6872 - accuracy: 0.4987
 7488/25000 [=======>......................] - ETA: 45s - loss: 7.6769 - accuracy: 0.4993
 7520/25000 [========>.....................] - ETA: 45s - loss: 7.6768 - accuracy: 0.4993
 7552/25000 [========>.....................] - ETA: 45s - loss: 7.6768 - accuracy: 0.4993
 7584/25000 [========>.....................] - ETA: 45s - loss: 7.6767 - accuracy: 0.4993
 7616/25000 [========>.....................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 7648/25000 [========>.....................] - ETA: 44s - loss: 7.6626 - accuracy: 0.5003
 7680/25000 [========>.....................] - ETA: 44s - loss: 7.6606 - accuracy: 0.5004
 7712/25000 [========>.....................] - ETA: 44s - loss: 7.6607 - accuracy: 0.5004
 7744/25000 [========>.....................] - ETA: 44s - loss: 7.6587 - accuracy: 0.5005
 7776/25000 [========>.....................] - ETA: 44s - loss: 7.6706 - accuracy: 0.4997
 7808/25000 [========>.....................] - ETA: 44s - loss: 7.6647 - accuracy: 0.5001
 7840/25000 [========>.....................] - ETA: 44s - loss: 7.6725 - accuracy: 0.4996
 7872/25000 [========>.....................] - ETA: 44s - loss: 7.6744 - accuracy: 0.4995
 7904/25000 [========>.....................] - ETA: 44s - loss: 7.6744 - accuracy: 0.4995
 7936/25000 [========>.....................] - ETA: 44s - loss: 7.6724 - accuracy: 0.4996
 7968/25000 [========>.....................] - ETA: 43s - loss: 7.6705 - accuracy: 0.4997
 8000/25000 [========>.....................] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
 8032/25000 [========>.....................] - ETA: 43s - loss: 7.6552 - accuracy: 0.5007
 8064/25000 [========>.....................] - ETA: 43s - loss: 7.6590 - accuracy: 0.5005
 8096/25000 [========>.....................] - ETA: 43s - loss: 7.6571 - accuracy: 0.5006
 8128/25000 [========>.....................] - ETA: 43s - loss: 7.6572 - accuracy: 0.5006
 8160/25000 [========>.....................] - ETA: 43s - loss: 7.6516 - accuracy: 0.5010
 8192/25000 [========>.....................] - ETA: 43s - loss: 7.6498 - accuracy: 0.5011
 8224/25000 [========>.....................] - ETA: 43s - loss: 7.6461 - accuracy: 0.5013
 8256/25000 [========>.....................] - ETA: 43s - loss: 7.6573 - accuracy: 0.5006
 8288/25000 [========>.....................] - ETA: 43s - loss: 7.6592 - accuracy: 0.5005
 8320/25000 [========>.....................] - ETA: 43s - loss: 7.6482 - accuracy: 0.5012
 8352/25000 [=========>....................] - ETA: 43s - loss: 7.6501 - accuracy: 0.5011
 8384/25000 [=========>....................] - ETA: 42s - loss: 7.6447 - accuracy: 0.5014
 8416/25000 [=========>....................] - ETA: 42s - loss: 7.6429 - accuracy: 0.5015
 8448/25000 [=========>....................] - ETA: 42s - loss: 7.6485 - accuracy: 0.5012
 8480/25000 [=========>....................] - ETA: 42s - loss: 7.6503 - accuracy: 0.5011
 8512/25000 [=========>....................] - ETA: 42s - loss: 7.6594 - accuracy: 0.5005
 8544/25000 [=========>....................] - ETA: 42s - loss: 7.6648 - accuracy: 0.5001
 8576/25000 [=========>....................] - ETA: 42s - loss: 7.6809 - accuracy: 0.4991
 8608/25000 [=========>....................] - ETA: 42s - loss: 7.6791 - accuracy: 0.4992
 8640/25000 [=========>....................] - ETA: 42s - loss: 7.6773 - accuracy: 0.4993
 8672/25000 [=========>....................] - ETA: 42s - loss: 7.6790 - accuracy: 0.4992
 8704/25000 [=========>....................] - ETA: 41s - loss: 7.6842 - accuracy: 0.4989
 8736/25000 [=========>....................] - ETA: 41s - loss: 7.6877 - accuracy: 0.4986
 8768/25000 [=========>....................] - ETA: 41s - loss: 7.6859 - accuracy: 0.4987
 8800/25000 [=========>....................] - ETA: 41s - loss: 7.6858 - accuracy: 0.4988
 8832/25000 [=========>....................] - ETA: 41s - loss: 7.6875 - accuracy: 0.4986
 8864/25000 [=========>....................] - ETA: 41s - loss: 7.6753 - accuracy: 0.4994
 8896/25000 [=========>....................] - ETA: 41s - loss: 7.6804 - accuracy: 0.4991
 8928/25000 [=========>....................] - ETA: 41s - loss: 7.6838 - accuracy: 0.4989
 8960/25000 [=========>....................] - ETA: 41s - loss: 7.6786 - accuracy: 0.4992
 8992/25000 [=========>....................] - ETA: 41s - loss: 7.6854 - accuracy: 0.4988
 9024/25000 [=========>....................] - ETA: 41s - loss: 7.6870 - accuracy: 0.4987
 9056/25000 [=========>....................] - ETA: 41s - loss: 7.6852 - accuracy: 0.4988
 9088/25000 [=========>....................] - ETA: 40s - loss: 7.6852 - accuracy: 0.4988
 9120/25000 [=========>....................] - ETA: 40s - loss: 7.6818 - accuracy: 0.4990
 9152/25000 [=========>....................] - ETA: 40s - loss: 7.6817 - accuracy: 0.4990
 9184/25000 [==========>...................] - ETA: 40s - loss: 7.6816 - accuracy: 0.4990
 9216/25000 [==========>...................] - ETA: 40s - loss: 7.6816 - accuracy: 0.4990
 9248/25000 [==========>...................] - ETA: 40s - loss: 7.6832 - accuracy: 0.4989
 9280/25000 [==========>...................] - ETA: 40s - loss: 7.6848 - accuracy: 0.4988
 9312/25000 [==========>...................] - ETA: 40s - loss: 7.6847 - accuracy: 0.4988
 9344/25000 [==========>...................] - ETA: 40s - loss: 7.6797 - accuracy: 0.4991
 9376/25000 [==========>...................] - ETA: 40s - loss: 7.6732 - accuracy: 0.4996
 9408/25000 [==========>...................] - ETA: 40s - loss: 7.6650 - accuracy: 0.5001
 9440/25000 [==========>...................] - ETA: 40s - loss: 7.6650 - accuracy: 0.5001
 9472/25000 [==========>...................] - ETA: 39s - loss: 7.6666 - accuracy: 0.5000
 9504/25000 [==========>...................] - ETA: 39s - loss: 7.6682 - accuracy: 0.4999
 9536/25000 [==========>...................] - ETA: 39s - loss: 7.6650 - accuracy: 0.5001
 9568/25000 [==========>...................] - ETA: 39s - loss: 7.6650 - accuracy: 0.5001
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.6618 - accuracy: 0.5003
 9632/25000 [==========>...................] - ETA: 39s - loss: 7.6650 - accuracy: 0.5001
 9664/25000 [==========>...................] - ETA: 39s - loss: 7.6603 - accuracy: 0.5004
 9696/25000 [==========>...................] - ETA: 39s - loss: 7.6587 - accuracy: 0.5005
 9728/25000 [==========>...................] - ETA: 39s - loss: 7.6493 - accuracy: 0.5011
 9760/25000 [==========>...................] - ETA: 39s - loss: 7.6446 - accuracy: 0.5014
 9792/25000 [==========>...................] - ETA: 39s - loss: 7.6400 - accuracy: 0.5017
 9824/25000 [==========>...................] - ETA: 39s - loss: 7.6338 - accuracy: 0.5021
 9856/25000 [==========>...................] - ETA: 38s - loss: 7.6339 - accuracy: 0.5021
 9888/25000 [==========>...................] - ETA: 38s - loss: 7.6372 - accuracy: 0.5019
 9920/25000 [==========>...................] - ETA: 38s - loss: 7.6373 - accuracy: 0.5019
 9952/25000 [==========>...................] - ETA: 38s - loss: 7.6389 - accuracy: 0.5018
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.6451 - accuracy: 0.5014
10016/25000 [===========>..................] - ETA: 38s - loss: 7.6452 - accuracy: 0.5014
10048/25000 [===========>..................] - ETA: 38s - loss: 7.6514 - accuracy: 0.5010
10080/25000 [===========>..................] - ETA: 38s - loss: 7.6560 - accuracy: 0.5007
10112/25000 [===========>..................] - ETA: 38s - loss: 7.6560 - accuracy: 0.5007
10144/25000 [===========>..................] - ETA: 38s - loss: 7.6530 - accuracy: 0.5009
10176/25000 [===========>..................] - ETA: 38s - loss: 7.6516 - accuracy: 0.5010
10208/25000 [===========>..................] - ETA: 38s - loss: 7.6546 - accuracy: 0.5008
10240/25000 [===========>..................] - ETA: 37s - loss: 7.6636 - accuracy: 0.5002
10272/25000 [===========>..................] - ETA: 37s - loss: 7.6636 - accuracy: 0.5002
10304/25000 [===========>..................] - ETA: 37s - loss: 7.6607 - accuracy: 0.5004
10336/25000 [===========>..................] - ETA: 37s - loss: 7.6562 - accuracy: 0.5007
10368/25000 [===========>..................] - ETA: 37s - loss: 7.6607 - accuracy: 0.5004
10400/25000 [===========>..................] - ETA: 37s - loss: 7.6637 - accuracy: 0.5002
10432/25000 [===========>..................] - ETA: 37s - loss: 7.6578 - accuracy: 0.5006
10464/25000 [===========>..................] - ETA: 37s - loss: 7.6549 - accuracy: 0.5008
10496/25000 [===========>..................] - ETA: 37s - loss: 7.6505 - accuracy: 0.5010
10528/25000 [===========>..................] - ETA: 37s - loss: 7.6521 - accuracy: 0.5009
10560/25000 [===========>..................] - ETA: 37s - loss: 7.6565 - accuracy: 0.5007
10592/25000 [===========>..................] - ETA: 37s - loss: 7.6565 - accuracy: 0.5007
10624/25000 [===========>..................] - ETA: 37s - loss: 7.6565 - accuracy: 0.5007
10656/25000 [===========>..................] - ETA: 36s - loss: 7.6551 - accuracy: 0.5008
10688/25000 [===========>..................] - ETA: 36s - loss: 7.6551 - accuracy: 0.5007
10720/25000 [===========>..................] - ETA: 36s - loss: 7.6566 - accuracy: 0.5007
10752/25000 [===========>..................] - ETA: 36s - loss: 7.6623 - accuracy: 0.5003
10784/25000 [===========>..................] - ETA: 36s - loss: 7.6680 - accuracy: 0.4999
10816/25000 [===========>..................] - ETA: 36s - loss: 7.6695 - accuracy: 0.4998
10848/25000 [============>.................] - ETA: 36s - loss: 7.6666 - accuracy: 0.5000
10880/25000 [============>.................] - ETA: 36s - loss: 7.6680 - accuracy: 0.4999
10912/25000 [============>.................] - ETA: 36s - loss: 7.6610 - accuracy: 0.5004
10944/25000 [============>.................] - ETA: 36s - loss: 7.6638 - accuracy: 0.5002
10976/25000 [============>.................] - ETA: 36s - loss: 7.6610 - accuracy: 0.5004
11008/25000 [============>.................] - ETA: 36s - loss: 7.6610 - accuracy: 0.5004
11040/25000 [============>.................] - ETA: 35s - loss: 7.6652 - accuracy: 0.5001
11072/25000 [============>.................] - ETA: 35s - loss: 7.6583 - accuracy: 0.5005
11104/25000 [============>.................] - ETA: 35s - loss: 7.6556 - accuracy: 0.5007
11136/25000 [============>.................] - ETA: 35s - loss: 7.6556 - accuracy: 0.5007
11168/25000 [============>.................] - ETA: 35s - loss: 7.6570 - accuracy: 0.5006
11200/25000 [============>.................] - ETA: 35s - loss: 7.6543 - accuracy: 0.5008
11232/25000 [============>.................] - ETA: 35s - loss: 7.6584 - accuracy: 0.5005
11264/25000 [============>.................] - ETA: 35s - loss: 7.6612 - accuracy: 0.5004
11296/25000 [============>.................] - ETA: 35s - loss: 7.6585 - accuracy: 0.5005
11328/25000 [============>.................] - ETA: 35s - loss: 7.6626 - accuracy: 0.5003
11360/25000 [============>.................] - ETA: 35s - loss: 7.6653 - accuracy: 0.5001
11392/25000 [============>.................] - ETA: 34s - loss: 7.6666 - accuracy: 0.5000
11424/25000 [============>.................] - ETA: 34s - loss: 7.6653 - accuracy: 0.5001
11456/25000 [============>.................] - ETA: 34s - loss: 7.6639 - accuracy: 0.5002
11488/25000 [============>.................] - ETA: 34s - loss: 7.6746 - accuracy: 0.4995
11520/25000 [============>.................] - ETA: 34s - loss: 7.6746 - accuracy: 0.4995
11552/25000 [============>.................] - ETA: 34s - loss: 7.6759 - accuracy: 0.4994
11584/25000 [============>.................] - ETA: 34s - loss: 7.6812 - accuracy: 0.4991
11616/25000 [============>.................] - ETA: 34s - loss: 7.6759 - accuracy: 0.4994
11648/25000 [============>.................] - ETA: 34s - loss: 7.6772 - accuracy: 0.4993
11680/25000 [=============>................] - ETA: 34s - loss: 7.6797 - accuracy: 0.4991
11712/25000 [=============>................] - ETA: 34s - loss: 7.6732 - accuracy: 0.4996
11744/25000 [=============>................] - ETA: 34s - loss: 7.6692 - accuracy: 0.4998
11776/25000 [=============>................] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
11808/25000 [=============>................] - ETA: 33s - loss: 7.6679 - accuracy: 0.4999
11840/25000 [=============>................] - ETA: 33s - loss: 7.6627 - accuracy: 0.5003
11872/25000 [=============>................] - ETA: 33s - loss: 7.6653 - accuracy: 0.5001
11904/25000 [=============>................] - ETA: 33s - loss: 7.6679 - accuracy: 0.4999
11936/25000 [=============>................] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
11968/25000 [=============>................] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
12000/25000 [=============>................] - ETA: 33s - loss: 7.6641 - accuracy: 0.5002
12032/25000 [=============>................] - ETA: 33s - loss: 7.6615 - accuracy: 0.5003
12064/25000 [=============>................] - ETA: 33s - loss: 7.6577 - accuracy: 0.5006
12096/25000 [=============>................] - ETA: 33s - loss: 7.6577 - accuracy: 0.5006
12128/25000 [=============>................] - ETA: 33s - loss: 7.6565 - accuracy: 0.5007
12160/25000 [=============>................] - ETA: 32s - loss: 7.6578 - accuracy: 0.5006
12192/25000 [=============>................] - ETA: 32s - loss: 7.6591 - accuracy: 0.5005
12224/25000 [=============>................] - ETA: 32s - loss: 7.6578 - accuracy: 0.5006
12256/25000 [=============>................] - ETA: 32s - loss: 7.6616 - accuracy: 0.5003
12288/25000 [=============>................] - ETA: 32s - loss: 7.6579 - accuracy: 0.5006
12320/25000 [=============>................] - ETA: 32s - loss: 7.6579 - accuracy: 0.5006
12352/25000 [=============>................] - ETA: 32s - loss: 7.6579 - accuracy: 0.5006
12384/25000 [=============>................] - ETA: 32s - loss: 7.6580 - accuracy: 0.5006
12416/25000 [=============>................] - ETA: 32s - loss: 7.6543 - accuracy: 0.5008
12448/25000 [=============>................] - ETA: 32s - loss: 7.6592 - accuracy: 0.5005
12480/25000 [=============>................] - ETA: 32s - loss: 7.6580 - accuracy: 0.5006
12512/25000 [==============>...............] - ETA: 32s - loss: 7.6556 - accuracy: 0.5007
12544/25000 [==============>...............] - ETA: 31s - loss: 7.6520 - accuracy: 0.5010
12576/25000 [==============>...............] - ETA: 31s - loss: 7.6544 - accuracy: 0.5008
12608/25000 [==============>...............] - ETA: 31s - loss: 7.6569 - accuracy: 0.5006
12640/25000 [==============>...............] - ETA: 31s - loss: 7.6557 - accuracy: 0.5007
12672/25000 [==============>...............] - ETA: 31s - loss: 7.6557 - accuracy: 0.5007
12704/25000 [==============>...............] - ETA: 31s - loss: 7.6606 - accuracy: 0.5004
12736/25000 [==============>...............] - ETA: 31s - loss: 7.6630 - accuracy: 0.5002
12768/25000 [==============>...............] - ETA: 31s - loss: 7.6654 - accuracy: 0.5001
12800/25000 [==============>...............] - ETA: 31s - loss: 7.6654 - accuracy: 0.5001
12832/25000 [==============>...............] - ETA: 31s - loss: 7.6630 - accuracy: 0.5002
12864/25000 [==============>...............] - ETA: 31s - loss: 7.6630 - accuracy: 0.5002
12896/25000 [==============>...............] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
12928/25000 [==============>...............] - ETA: 30s - loss: 7.6690 - accuracy: 0.4998
12960/25000 [==============>...............] - ETA: 30s - loss: 7.6702 - accuracy: 0.4998
12992/25000 [==============>...............] - ETA: 30s - loss: 7.6713 - accuracy: 0.4997
13024/25000 [==============>...............] - ETA: 30s - loss: 7.6702 - accuracy: 0.4998
13056/25000 [==============>...............] - ETA: 30s - loss: 7.6678 - accuracy: 0.4999
13088/25000 [==============>...............] - ETA: 30s - loss: 7.6619 - accuracy: 0.5003
13120/25000 [==============>...............] - ETA: 30s - loss: 7.6619 - accuracy: 0.5003
13152/25000 [==============>...............] - ETA: 30s - loss: 7.6620 - accuracy: 0.5003
13184/25000 [==============>...............] - ETA: 30s - loss: 7.6631 - accuracy: 0.5002
13216/25000 [==============>...............] - ETA: 30s - loss: 7.6597 - accuracy: 0.5005
13248/25000 [==============>...............] - ETA: 30s - loss: 7.6585 - accuracy: 0.5005
13280/25000 [==============>...............] - ETA: 30s - loss: 7.6597 - accuracy: 0.5005
13312/25000 [==============>...............] - ETA: 29s - loss: 7.6574 - accuracy: 0.5006
13344/25000 [===============>..............] - ETA: 29s - loss: 7.6517 - accuracy: 0.5010
13376/25000 [===============>..............] - ETA: 29s - loss: 7.6483 - accuracy: 0.5012
13408/25000 [===============>..............] - ETA: 29s - loss: 7.6472 - accuracy: 0.5013
13440/25000 [===============>..............] - ETA: 29s - loss: 7.6495 - accuracy: 0.5011
13472/25000 [===============>..............] - ETA: 29s - loss: 7.6530 - accuracy: 0.5009
13504/25000 [===============>..............] - ETA: 29s - loss: 7.6519 - accuracy: 0.5010
13536/25000 [===============>..............] - ETA: 29s - loss: 7.6530 - accuracy: 0.5009
13568/25000 [===============>..............] - ETA: 29s - loss: 7.6519 - accuracy: 0.5010
13600/25000 [===============>..............] - ETA: 29s - loss: 7.6463 - accuracy: 0.5013
13632/25000 [===============>..............] - ETA: 29s - loss: 7.6475 - accuracy: 0.5012
13664/25000 [===============>..............] - ETA: 29s - loss: 7.6442 - accuracy: 0.5015
13696/25000 [===============>..............] - ETA: 29s - loss: 7.6420 - accuracy: 0.5016
13728/25000 [===============>..............] - ETA: 28s - loss: 7.6487 - accuracy: 0.5012
13760/25000 [===============>..............] - ETA: 28s - loss: 7.6432 - accuracy: 0.5015
13792/25000 [===============>..............] - ETA: 28s - loss: 7.6444 - accuracy: 0.5015
13824/25000 [===============>..............] - ETA: 28s - loss: 7.6455 - accuracy: 0.5014
13856/25000 [===============>..............] - ETA: 28s - loss: 7.6423 - accuracy: 0.5016
13888/25000 [===============>..............] - ETA: 28s - loss: 7.6401 - accuracy: 0.5017
13920/25000 [===============>..............] - ETA: 28s - loss: 7.6413 - accuracy: 0.5017
13952/25000 [===============>..............] - ETA: 28s - loss: 7.6380 - accuracy: 0.5019
13984/25000 [===============>..............] - ETA: 28s - loss: 7.6425 - accuracy: 0.5016
14016/25000 [===============>..............] - ETA: 28s - loss: 7.6447 - accuracy: 0.5014
14048/25000 [===============>..............] - ETA: 28s - loss: 7.6459 - accuracy: 0.5014
14080/25000 [===============>..............] - ETA: 28s - loss: 7.6470 - accuracy: 0.5013
14112/25000 [===============>..............] - ETA: 28s - loss: 7.6471 - accuracy: 0.5013
14144/25000 [===============>..............] - ETA: 27s - loss: 7.6471 - accuracy: 0.5013
14176/25000 [================>.............] - ETA: 27s - loss: 7.6472 - accuracy: 0.5013
14208/25000 [================>.............] - ETA: 27s - loss: 7.6494 - accuracy: 0.5011
14240/25000 [================>.............] - ETA: 27s - loss: 7.6451 - accuracy: 0.5014
14272/25000 [================>.............] - ETA: 27s - loss: 7.6462 - accuracy: 0.5013
14304/25000 [================>.............] - ETA: 27s - loss: 7.6495 - accuracy: 0.5011
14336/25000 [================>.............] - ETA: 27s - loss: 7.6516 - accuracy: 0.5010
14368/25000 [================>.............] - ETA: 27s - loss: 7.6495 - accuracy: 0.5011
14400/25000 [================>.............] - ETA: 27s - loss: 7.6453 - accuracy: 0.5014
14432/25000 [================>.............] - ETA: 27s - loss: 7.6464 - accuracy: 0.5013
14464/25000 [================>.............] - ETA: 27s - loss: 7.6475 - accuracy: 0.5012
14496/25000 [================>.............] - ETA: 27s - loss: 7.6455 - accuracy: 0.5014
14528/25000 [================>.............] - ETA: 26s - loss: 7.6434 - accuracy: 0.5015
14560/25000 [================>.............] - ETA: 26s - loss: 7.6413 - accuracy: 0.5016
14592/25000 [================>.............] - ETA: 26s - loss: 7.6361 - accuracy: 0.5020
14624/25000 [================>.............] - ETA: 26s - loss: 7.6404 - accuracy: 0.5017
14656/25000 [================>.............] - ETA: 26s - loss: 7.6415 - accuracy: 0.5016
14688/25000 [================>.............] - ETA: 26s - loss: 7.6395 - accuracy: 0.5018
14720/25000 [================>.............] - ETA: 26s - loss: 7.6406 - accuracy: 0.5017
14752/25000 [================>.............] - ETA: 26s - loss: 7.6427 - accuracy: 0.5016
14784/25000 [================>.............] - ETA: 26s - loss: 7.6397 - accuracy: 0.5018
14816/25000 [================>.............] - ETA: 26s - loss: 7.6449 - accuracy: 0.5014
14848/25000 [================>.............] - ETA: 26s - loss: 7.6460 - accuracy: 0.5013
14880/25000 [================>.............] - ETA: 26s - loss: 7.6470 - accuracy: 0.5013
14912/25000 [================>.............] - ETA: 25s - loss: 7.6512 - accuracy: 0.5010
14944/25000 [================>.............] - ETA: 25s - loss: 7.6502 - accuracy: 0.5011
14976/25000 [================>.............] - ETA: 25s - loss: 7.6492 - accuracy: 0.5011
15008/25000 [=================>............] - ETA: 25s - loss: 7.6441 - accuracy: 0.5015
15040/25000 [=================>............] - ETA: 25s - loss: 7.6432 - accuracy: 0.5015
15072/25000 [=================>............] - ETA: 25s - loss: 7.6371 - accuracy: 0.5019
15104/25000 [=================>............] - ETA: 25s - loss: 7.6392 - accuracy: 0.5018
15136/25000 [=================>............] - ETA: 25s - loss: 7.6453 - accuracy: 0.5014
15168/25000 [=================>............] - ETA: 25s - loss: 7.6515 - accuracy: 0.5010
15200/25000 [=================>............] - ETA: 25s - loss: 7.6515 - accuracy: 0.5010
15232/25000 [=================>............] - ETA: 25s - loss: 7.6495 - accuracy: 0.5011
15264/25000 [=================>............] - ETA: 25s - loss: 7.6546 - accuracy: 0.5008
15296/25000 [=================>............] - ETA: 25s - loss: 7.6556 - accuracy: 0.5007
15328/25000 [=================>............] - ETA: 24s - loss: 7.6546 - accuracy: 0.5008
15360/25000 [=================>............] - ETA: 24s - loss: 7.6526 - accuracy: 0.5009
15392/25000 [=================>............] - ETA: 24s - loss: 7.6527 - accuracy: 0.5009
15424/25000 [=================>............] - ETA: 24s - loss: 7.6467 - accuracy: 0.5013
15456/25000 [=================>............] - ETA: 24s - loss: 7.6488 - accuracy: 0.5012
15488/25000 [=================>............] - ETA: 24s - loss: 7.6478 - accuracy: 0.5012
15520/25000 [=================>............] - ETA: 24s - loss: 7.6488 - accuracy: 0.5012
15552/25000 [=================>............] - ETA: 24s - loss: 7.6499 - accuracy: 0.5011
15584/25000 [=================>............] - ETA: 24s - loss: 7.6528 - accuracy: 0.5009
15616/25000 [=================>............] - ETA: 24s - loss: 7.6489 - accuracy: 0.5012
15648/25000 [=================>............] - ETA: 24s - loss: 7.6441 - accuracy: 0.5015
15680/25000 [=================>............] - ETA: 24s - loss: 7.6451 - accuracy: 0.5014
15712/25000 [=================>............] - ETA: 23s - loss: 7.6442 - accuracy: 0.5015
15744/25000 [=================>............] - ETA: 23s - loss: 7.6442 - accuracy: 0.5015
15776/25000 [=================>............] - ETA: 23s - loss: 7.6443 - accuracy: 0.5015
15808/25000 [=================>............] - ETA: 23s - loss: 7.6424 - accuracy: 0.5016
15840/25000 [==================>...........] - ETA: 23s - loss: 7.6395 - accuracy: 0.5018
15872/25000 [==================>...........] - ETA: 23s - loss: 7.6415 - accuracy: 0.5016
15904/25000 [==================>...........] - ETA: 23s - loss: 7.6377 - accuracy: 0.5019
15936/25000 [==================>...........] - ETA: 23s - loss: 7.6339 - accuracy: 0.5021
15968/25000 [==================>...........] - ETA: 23s - loss: 7.6359 - accuracy: 0.5020
16000/25000 [==================>...........] - ETA: 23s - loss: 7.6388 - accuracy: 0.5018
16032/25000 [==================>...........] - ETA: 23s - loss: 7.6379 - accuracy: 0.5019
16064/25000 [==================>...........] - ETA: 23s - loss: 7.6389 - accuracy: 0.5018
16096/25000 [==================>...........] - ETA: 22s - loss: 7.6457 - accuracy: 0.5014
16128/25000 [==================>...........] - ETA: 22s - loss: 7.6429 - accuracy: 0.5016
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6429 - accuracy: 0.5015
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6420 - accuracy: 0.5016
16224/25000 [==================>...........] - ETA: 22s - loss: 7.6402 - accuracy: 0.5017
16256/25000 [==================>...........] - ETA: 22s - loss: 7.6393 - accuracy: 0.5018
16288/25000 [==================>...........] - ETA: 22s - loss: 7.6403 - accuracy: 0.5017
16320/25000 [==================>...........] - ETA: 22s - loss: 7.6403 - accuracy: 0.5017
16352/25000 [==================>...........] - ETA: 22s - loss: 7.6347 - accuracy: 0.5021
16384/25000 [==================>...........] - ETA: 22s - loss: 7.6339 - accuracy: 0.5021
16416/25000 [==================>...........] - ETA: 22s - loss: 7.6302 - accuracy: 0.5024
16448/25000 [==================>...........] - ETA: 22s - loss: 7.6303 - accuracy: 0.5024
16480/25000 [==================>...........] - ETA: 21s - loss: 7.6248 - accuracy: 0.5027
16512/25000 [==================>...........] - ETA: 21s - loss: 7.6258 - accuracy: 0.5027
16544/25000 [==================>...........] - ETA: 21s - loss: 7.6249 - accuracy: 0.5027
16576/25000 [==================>...........] - ETA: 21s - loss: 7.6250 - accuracy: 0.5027
16608/25000 [==================>...........] - ETA: 21s - loss: 7.6269 - accuracy: 0.5026
16640/25000 [==================>...........] - ETA: 21s - loss: 7.6270 - accuracy: 0.5026
16672/25000 [===================>..........] - ETA: 21s - loss: 7.6225 - accuracy: 0.5029
16704/25000 [===================>..........] - ETA: 21s - loss: 7.6226 - accuracy: 0.5029
16736/25000 [===================>..........] - ETA: 21s - loss: 7.6226 - accuracy: 0.5029
16768/25000 [===================>..........] - ETA: 21s - loss: 7.6264 - accuracy: 0.5026
16800/25000 [===================>..........] - ETA: 21s - loss: 7.6246 - accuracy: 0.5027
16832/25000 [===================>..........] - ETA: 21s - loss: 7.6293 - accuracy: 0.5024
16864/25000 [===================>..........] - ETA: 20s - loss: 7.6312 - accuracy: 0.5023
16896/25000 [===================>..........] - ETA: 20s - loss: 7.6303 - accuracy: 0.5024
16928/25000 [===================>..........] - ETA: 20s - loss: 7.6331 - accuracy: 0.5022
16960/25000 [===================>..........] - ETA: 20s - loss: 7.6350 - accuracy: 0.5021
16992/25000 [===================>..........] - ETA: 20s - loss: 7.6305 - accuracy: 0.5024
17024/25000 [===================>..........] - ETA: 20s - loss: 7.6324 - accuracy: 0.5022
17056/25000 [===================>..........] - ETA: 20s - loss: 7.6298 - accuracy: 0.5024
17088/25000 [===================>..........] - ETA: 20s - loss: 7.6307 - accuracy: 0.5023
17120/25000 [===================>..........] - ETA: 20s - loss: 7.6335 - accuracy: 0.5022
17152/25000 [===================>..........] - ETA: 20s - loss: 7.6344 - accuracy: 0.5021
17184/25000 [===================>..........] - ETA: 20s - loss: 7.6345 - accuracy: 0.5021
17216/25000 [===================>..........] - ETA: 20s - loss: 7.6319 - accuracy: 0.5023
17248/25000 [===================>..........] - ETA: 20s - loss: 7.6302 - accuracy: 0.5024
17280/25000 [===================>..........] - ETA: 19s - loss: 7.6285 - accuracy: 0.5025
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6294 - accuracy: 0.5024
17344/25000 [===================>..........] - ETA: 19s - loss: 7.6313 - accuracy: 0.5023
17376/25000 [===================>..........] - ETA: 19s - loss: 7.6331 - accuracy: 0.5022
17408/25000 [===================>..........] - ETA: 19s - loss: 7.6331 - accuracy: 0.5022
17440/25000 [===================>..........] - ETA: 19s - loss: 7.6315 - accuracy: 0.5023
17472/25000 [===================>..........] - ETA: 19s - loss: 7.6298 - accuracy: 0.5024
17504/25000 [====================>.........] - ETA: 19s - loss: 7.6351 - accuracy: 0.5021
17536/25000 [====================>.........] - ETA: 19s - loss: 7.6343 - accuracy: 0.5021
17568/25000 [====================>.........] - ETA: 19s - loss: 7.6369 - accuracy: 0.5019
17600/25000 [====================>.........] - ETA: 19s - loss: 7.6344 - accuracy: 0.5021
17632/25000 [====================>.........] - ETA: 19s - loss: 7.6327 - accuracy: 0.5022
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6345 - accuracy: 0.5021
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6320 - accuracy: 0.5023
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6320 - accuracy: 0.5023
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6329 - accuracy: 0.5022
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6356 - accuracy: 0.5020
17824/25000 [====================>.........] - ETA: 18s - loss: 7.6357 - accuracy: 0.5020
17856/25000 [====================>.........] - ETA: 18s - loss: 7.6331 - accuracy: 0.5022
17888/25000 [====================>.........] - ETA: 18s - loss: 7.6349 - accuracy: 0.5021
17920/25000 [====================>.........] - ETA: 18s - loss: 7.6367 - accuracy: 0.5020
17952/25000 [====================>.........] - ETA: 18s - loss: 7.6384 - accuracy: 0.5018
17984/25000 [====================>.........] - ETA: 18s - loss: 7.6427 - accuracy: 0.5016
18016/25000 [====================>.........] - ETA: 18s - loss: 7.6377 - accuracy: 0.5019
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6386 - accuracy: 0.5018
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6335 - accuracy: 0.5022
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6302 - accuracy: 0.5024
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6269 - accuracy: 0.5026
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6295 - accuracy: 0.5024
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6338 - accuracy: 0.5021
18240/25000 [====================>.........] - ETA: 17s - loss: 7.6322 - accuracy: 0.5022
18272/25000 [====================>.........] - ETA: 17s - loss: 7.6322 - accuracy: 0.5022
18304/25000 [====================>.........] - ETA: 17s - loss: 7.6323 - accuracy: 0.5022
18336/25000 [=====================>........] - ETA: 17s - loss: 7.6348 - accuracy: 0.5021
18368/25000 [=====================>........] - ETA: 17s - loss: 7.6357 - accuracy: 0.5020
18400/25000 [=====================>........] - ETA: 17s - loss: 7.6358 - accuracy: 0.5020
18432/25000 [=====================>........] - ETA: 17s - loss: 7.6325 - accuracy: 0.5022
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6301 - accuracy: 0.5024
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6277 - accuracy: 0.5025
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6236 - accuracy: 0.5028
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6228 - accuracy: 0.5029
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6254 - accuracy: 0.5027
18624/25000 [=====================>........] - ETA: 16s - loss: 7.6238 - accuracy: 0.5028
18656/25000 [=====================>........] - ETA: 16s - loss: 7.6255 - accuracy: 0.5027
18688/25000 [=====================>........] - ETA: 16s - loss: 7.6256 - accuracy: 0.5027
18720/25000 [=====================>........] - ETA: 16s - loss: 7.6289 - accuracy: 0.5025
18752/25000 [=====================>........] - ETA: 16s - loss: 7.6306 - accuracy: 0.5023
18784/25000 [=====================>........] - ETA: 16s - loss: 7.6340 - accuracy: 0.5021
18816/25000 [=====================>........] - ETA: 16s - loss: 7.6324 - accuracy: 0.5022
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6365 - accuracy: 0.5020
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6374 - accuracy: 0.5019
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6366 - accuracy: 0.5020
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6359 - accuracy: 0.5020
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6335 - accuracy: 0.5022
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6344 - accuracy: 0.5021
19040/25000 [=====================>........] - ETA: 15s - loss: 7.6328 - accuracy: 0.5022
19072/25000 [=====================>........] - ETA: 15s - loss: 7.6353 - accuracy: 0.5020
19104/25000 [=====================>........] - ETA: 15s - loss: 7.6361 - accuracy: 0.5020
19136/25000 [=====================>........] - ETA: 15s - loss: 7.6378 - accuracy: 0.5019
19168/25000 [======================>.......] - ETA: 15s - loss: 7.6346 - accuracy: 0.5021
19200/25000 [======================>.......] - ETA: 15s - loss: 7.6355 - accuracy: 0.5020
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6347 - accuracy: 0.5021
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6372 - accuracy: 0.5019
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6348 - accuracy: 0.5021
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6373 - accuracy: 0.5019
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6381 - accuracy: 0.5019
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6389 - accuracy: 0.5018
19424/25000 [======================>.......] - ETA: 14s - loss: 7.6382 - accuracy: 0.5019
19456/25000 [======================>.......] - ETA: 14s - loss: 7.6390 - accuracy: 0.5018
19488/25000 [======================>.......] - ETA: 14s - loss: 7.6407 - accuracy: 0.5017
19520/25000 [======================>.......] - ETA: 14s - loss: 7.6399 - accuracy: 0.5017
19552/25000 [======================>.......] - ETA: 14s - loss: 7.6392 - accuracy: 0.5018
19584/25000 [======================>.......] - ETA: 14s - loss: 7.6377 - accuracy: 0.5019
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6385 - accuracy: 0.5018
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6370 - accuracy: 0.5019
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6378 - accuracy: 0.5019
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6371 - accuracy: 0.5019
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6418 - accuracy: 0.5016
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6434 - accuracy: 0.5015
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6488 - accuracy: 0.5012
19840/25000 [======================>.......] - ETA: 13s - loss: 7.6488 - accuracy: 0.5012
19872/25000 [======================>.......] - ETA: 13s - loss: 7.6512 - accuracy: 0.5010
19904/25000 [======================>.......] - ETA: 13s - loss: 7.6489 - accuracy: 0.5012
19936/25000 [======================>.......] - ETA: 13s - loss: 7.6512 - accuracy: 0.5010
19968/25000 [======================>.......] - ETA: 13s - loss: 7.6513 - accuracy: 0.5010
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6544 - accuracy: 0.5008
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6528 - accuracy: 0.5009
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6544 - accuracy: 0.5008
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6590 - accuracy: 0.5005
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6620 - accuracy: 0.5003
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6605 - accuracy: 0.5004
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6628 - accuracy: 0.5002
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6598 - accuracy: 0.5004
20256/25000 [=======================>......] - ETA: 12s - loss: 7.6606 - accuracy: 0.5004
20288/25000 [=======================>......] - ETA: 12s - loss: 7.6606 - accuracy: 0.5004
20320/25000 [=======================>......] - ETA: 12s - loss: 7.6598 - accuracy: 0.5004
20352/25000 [=======================>......] - ETA: 12s - loss: 7.6613 - accuracy: 0.5003
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6591 - accuracy: 0.5005
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6606 - accuracy: 0.5004
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6621 - accuracy: 0.5003
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6644 - accuracy: 0.5001
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6644 - accuracy: 0.5001
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6659 - accuracy: 0.5000
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6666 - accuracy: 0.5000
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6681 - accuracy: 0.4999
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6681 - accuracy: 0.4999
20672/25000 [=======================>......] - ETA: 11s - loss: 7.6696 - accuracy: 0.4998
20704/25000 [=======================>......] - ETA: 11s - loss: 7.6725 - accuracy: 0.4996
20736/25000 [=======================>......] - ETA: 11s - loss: 7.6725 - accuracy: 0.4996
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6718 - accuracy: 0.4997
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6703 - accuracy: 0.4998
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6725 - accuracy: 0.4996
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6710 - accuracy: 0.4997
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6718 - accuracy: 0.4997
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6681 - accuracy: 0.4999
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6674 - accuracy: 0.5000
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6673 - accuracy: 0.5000
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6695 - accuracy: 0.4998
21056/25000 [========================>.....] - ETA: 10s - loss: 7.6688 - accuracy: 0.4999
21088/25000 [========================>.....] - ETA: 10s - loss: 7.6688 - accuracy: 0.4999
21120/25000 [========================>.....] - ETA: 10s - loss: 7.6644 - accuracy: 0.5001
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6637 - accuracy: 0.5002 
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6637 - accuracy: 0.5002
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6637 - accuracy: 0.5002
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6652 - accuracy: 0.5001
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6652 - accuracy: 0.5001
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6645 - accuracy: 0.5001
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6630 - accuracy: 0.5002
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6652 - accuracy: 0.5001
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6630 - accuracy: 0.5002
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6630 - accuracy: 0.5002
21472/25000 [========================>.....] - ETA: 9s - loss: 7.6616 - accuracy: 0.5003
21504/25000 [========================>.....] - ETA: 9s - loss: 7.6616 - accuracy: 0.5003
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6595 - accuracy: 0.5005
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6602 - accuracy: 0.5004
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6560 - accuracy: 0.5007
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6567 - accuracy: 0.5006
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6553 - accuracy: 0.5007
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6546 - accuracy: 0.5008
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6532 - accuracy: 0.5009
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6539 - accuracy: 0.5008
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6533 - accuracy: 0.5009
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6554 - accuracy: 0.5007
21856/25000 [=========================>....] - ETA: 8s - loss: 7.6526 - accuracy: 0.5009
21888/25000 [=========================>....] - ETA: 8s - loss: 7.6512 - accuracy: 0.5010
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6498 - accuracy: 0.5011
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6485 - accuracy: 0.5012
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6499 - accuracy: 0.5011
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6479 - accuracy: 0.5012
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6486 - accuracy: 0.5012
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6479 - accuracy: 0.5012
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6480 - accuracy: 0.5012
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6445 - accuracy: 0.5014
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6439 - accuracy: 0.5015
22272/25000 [=========================>....] - ETA: 7s - loss: 7.6453 - accuracy: 0.5014
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6460 - accuracy: 0.5013
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6508 - accuracy: 0.5010
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6502 - accuracy: 0.5011
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6502 - accuracy: 0.5011
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6495 - accuracy: 0.5011
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6502 - accuracy: 0.5011
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6550 - accuracy: 0.5008
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6550 - accuracy: 0.5008
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6530 - accuracy: 0.5009
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6537 - accuracy: 0.5008
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6504 - accuracy: 0.5011
22656/25000 [==========================>...] - ETA: 6s - loss: 7.6524 - accuracy: 0.5009
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6511 - accuracy: 0.5010
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6511 - accuracy: 0.5010
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6504 - accuracy: 0.5011
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6511 - accuracy: 0.5010
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6512 - accuracy: 0.5010
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6512 - accuracy: 0.5010
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6519 - accuracy: 0.5010
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6526 - accuracy: 0.5009
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6539 - accuracy: 0.5008
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6506 - accuracy: 0.5010
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6493 - accuracy: 0.5011
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6487 - accuracy: 0.5012
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6487 - accuracy: 0.5012
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6474 - accuracy: 0.5013
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6488 - accuracy: 0.5012
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6527 - accuracy: 0.5009
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6528 - accuracy: 0.5009
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6534 - accuracy: 0.5009
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6528 - accuracy: 0.5009
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6548 - accuracy: 0.5008
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6509 - accuracy: 0.5010
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6476 - accuracy: 0.5012
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6450 - accuracy: 0.5014
23456/25000 [===========================>..] - ETA: 4s - loss: 7.6450 - accuracy: 0.5014
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6425 - accuracy: 0.5016
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6445 - accuracy: 0.5014
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6458 - accuracy: 0.5014
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6439 - accuracy: 0.5015
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6439 - accuracy: 0.5015
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6459 - accuracy: 0.5014
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6472 - accuracy: 0.5013
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6459 - accuracy: 0.5013
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6447 - accuracy: 0.5014
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6447 - accuracy: 0.5014
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6460 - accuracy: 0.5013
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6454 - accuracy: 0.5014
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6422 - accuracy: 0.5016
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6448 - accuracy: 0.5014
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6474 - accuracy: 0.5013
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6461 - accuracy: 0.5013
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6487 - accuracy: 0.5012
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6488 - accuracy: 0.5012
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6532 - accuracy: 0.5009
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6513 - accuracy: 0.5010
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6495 - accuracy: 0.5011
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6501 - accuracy: 0.5011
24192/25000 [============================>.] - ETA: 2s - loss: 7.6514 - accuracy: 0.5010
24224/25000 [============================>.] - ETA: 2s - loss: 7.6521 - accuracy: 0.5009
24256/25000 [============================>.] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
24288/25000 [============================>.] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
24320/25000 [============================>.] - ETA: 1s - loss: 7.6546 - accuracy: 0.5008
24352/25000 [============================>.] - ETA: 1s - loss: 7.6565 - accuracy: 0.5007
24384/25000 [============================>.] - ETA: 1s - loss: 7.6572 - accuracy: 0.5006
24416/25000 [============================>.] - ETA: 1s - loss: 7.6566 - accuracy: 0.5007
24448/25000 [============================>.] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
24480/25000 [============================>.] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
24512/25000 [============================>.] - ETA: 1s - loss: 7.6560 - accuracy: 0.5007
24544/25000 [============================>.] - ETA: 1s - loss: 7.6591 - accuracy: 0.5005
24576/25000 [============================>.] - ETA: 1s - loss: 7.6598 - accuracy: 0.5004
24608/25000 [============================>.] - ETA: 1s - loss: 7.6579 - accuracy: 0.5006
24640/25000 [============================>.] - ETA: 0s - loss: 7.6585 - accuracy: 0.5005
24672/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24704/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24736/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
24768/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
24800/25000 [============================>.] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24832/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24864/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24896/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24960/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 76s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7ff51e89ba58> 

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
 [ 0.00552712 -0.07886589 -0.0211259   0.04092107 -0.02957537 -0.00329356]
 [-0.01042536  0.05779052  0.0080185  -0.08009491 -0.16316153  0.03037735]
 [ 0.0229214  -0.02249796 -0.11436159 -0.25632495 -0.02188831  0.25018793]
 [-0.1886348  -0.2598092   0.02653416 -0.37177029  0.22514717  0.02212927]
 [-0.68072712  0.39762172 -0.03173476 -0.60844254 -0.66942477  0.15364869]
 [ 0.0974791   0.13519472  0.11603656 -0.09300988  0.05757046 -0.00721707]
 [-0.22397274 -0.33602002  0.17322545 -0.0053841   0.19417776  0.11993199]
 [ 0.19000971  0.34657857  0.27343473 -0.65507567 -0.00863771 -0.00842206]
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
{'loss': 0.4317092038691044, 'loss_history': []}

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
{'loss': 0.48900385200977325, 'loss_history': []}

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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:52<01:19, 26.38s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.062179653174668184, 'embedding_size_factor': 0.5619475330610555, 'layers.choice': 0, 'learning_rate': 0.00023676458647999143, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 8.395772650765917e-08} and reward: 0.3806
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xaf\xd6\x02\xf1\xb9@\xcaX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe1\xfbyd\x92\x15fX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?/\x08\x80O\xc2\xba5X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>v\x89\x87\xf1\xd0\x94\xa0u.' and reward: 0.3806
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xaf\xd6\x02\xf1\xb9@\xcaX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe1\xfbyd\x92\x15fX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?/\x08\x80O\xc2\xba5X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>v\x89\x87\xf1\xd0\x94\xa0u.' and reward: 0.3806
 60%|██████    | 3/5 [01:44<01:08, 34.13s/it] 60%|██████    | 3/5 [01:44<01:09, 34.99s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.0990772717169359, 'embedding_size_factor': 0.6929843089542498, 'layers.choice': 0, 'learning_rate': 0.00031735386907862204, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 0.0003247310625077135} and reward: 0.3494
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xb9] \xc9\xcd\x17\xdeX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6,\xedm\xf30pX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?4\xccP}+\xf1\x14X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?5H\x15K \x95\xd2u.' and reward: 0.3494
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xb9] \xc9\xcd\x17\xdeX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6,\xedm\xf30pX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?4\xccP}+\xf1\x14X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?5H\x15K \x95\xd2u.' and reward: 0.3494
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 158.39549493789673
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -40.96s of remaining time.
Ensemble size: 28
Ensemble weights: 
[0.78571429 0.10714286 0.10714286]
	0.3888	 = Validation accuracy score
	1.1s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 162.1s ...
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

