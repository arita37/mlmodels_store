
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_jupyter GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/1d5d13d83f013fb4dab2460dee95031ee3f22079', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '1d5d13d83f013fb4dab2460dee95031ee3f22079', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/1d5d13d83f013fb4dab2460dee95031ee3f22079

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/1d5d13d83f013fb4dab2460dee95031ee3f22079

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
 2539520/17464789 [===>..........................] - ETA: 0s
11673600/17464789 [===================>..........] - ETA: 0s
16687104/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 19:53:27.744568: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 19:53:27.749280: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-10 19:53:27.749475: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56459af847a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 19:53:27.749491: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

2020-05-10 19:53:28.215162: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:39 - loss: 9.1041 - accuracy: 0.40622020-05-10 19:53:28.338718: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

   64/25000 [..............................] - ETA: 3:04 - loss: 8.3854 - accuracy: 0.45312020-05-10 19:53:28.455018: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

   96/25000 [..............................] - ETA: 2:33 - loss: 7.5069 - accuracy: 0.51042020-05-10 19:53:28.575200: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

  128/25000 [..............................] - ETA: 2:20 - loss: 7.6666 - accuracy: 0.50002020-05-10 19:53:28.702549: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 17424000 exceeds 10% of system memory.

  160/25000 [..............................] - ETA: 2:10 - loss: 7.6666 - accuracy: 0.5000
  192/25000 [..............................] - ETA: 2:04 - loss: 7.4270 - accuracy: 0.5156
  224/25000 [..............................] - ETA: 1:59 - loss: 7.5297 - accuracy: 0.5089
  256/25000 [..............................] - ETA: 1:55 - loss: 7.7265 - accuracy: 0.4961
  288/25000 [..............................] - ETA: 1:53 - loss: 7.8263 - accuracy: 0.4896
  320/25000 [..............................] - ETA: 1:50 - loss: 7.8583 - accuracy: 0.4875
  352/25000 [..............................] - ETA: 1:47 - loss: 7.9280 - accuracy: 0.4830
  384/25000 [..............................] - ETA: 1:46 - loss: 8.1458 - accuracy: 0.4688
  416/25000 [..............................] - ETA: 1:45 - loss: 8.1089 - accuracy: 0.4712
  448/25000 [..............................] - ETA: 1:43 - loss: 8.0773 - accuracy: 0.4732
  480/25000 [..............................] - ETA: 1:42 - loss: 8.0180 - accuracy: 0.4771
  512/25000 [..............................] - ETA: 1:40 - loss: 7.9062 - accuracy: 0.4844
  544/25000 [..............................] - ETA: 1:39 - loss: 7.8076 - accuracy: 0.4908
  576/25000 [..............................] - ETA: 1:38 - loss: 7.8796 - accuracy: 0.4861
  608/25000 [..............................] - ETA: 1:37 - loss: 7.7927 - accuracy: 0.4918
  640/25000 [..............................] - ETA: 1:37 - loss: 7.8583 - accuracy: 0.4875
  672/25000 [..............................] - ETA: 1:36 - loss: 7.7351 - accuracy: 0.4955
  704/25000 [..............................] - ETA: 1:36 - loss: 7.6884 - accuracy: 0.4986
  736/25000 [..............................] - ETA: 1:35 - loss: 7.6875 - accuracy: 0.4986
  768/25000 [..............................] - ETA: 1:35 - loss: 7.6467 - accuracy: 0.5013
  800/25000 [..............................] - ETA: 1:34 - loss: 7.7241 - accuracy: 0.4963
  832/25000 [..............................] - ETA: 1:34 - loss: 7.6298 - accuracy: 0.5024
  864/25000 [>.............................] - ETA: 1:34 - loss: 7.6311 - accuracy: 0.5023
  896/25000 [>.............................] - ETA: 1:33 - loss: 7.5468 - accuracy: 0.5078
  928/25000 [>.............................] - ETA: 1:33 - loss: 7.5675 - accuracy: 0.5065
  960/25000 [>.............................] - ETA: 1:32 - loss: 7.5708 - accuracy: 0.5063
  992/25000 [>.............................] - ETA: 1:32 - loss: 7.5584 - accuracy: 0.5071
 1024/25000 [>.............................] - ETA: 1:31 - loss: 7.5468 - accuracy: 0.5078
 1056/25000 [>.............................] - ETA: 1:31 - loss: 7.5795 - accuracy: 0.5057
 1088/25000 [>.............................] - ETA: 1:31 - loss: 7.5821 - accuracy: 0.5055
 1120/25000 [>.............................] - ETA: 1:31 - loss: 7.6666 - accuracy: 0.5000
 1152/25000 [>.............................] - ETA: 1:30 - loss: 7.6400 - accuracy: 0.5017
 1184/25000 [>.............................] - ETA: 1:30 - loss: 7.7184 - accuracy: 0.4966
 1216/25000 [>.............................] - ETA: 1:29 - loss: 7.6792 - accuracy: 0.4992
 1248/25000 [>.............................] - ETA: 1:29 - loss: 7.7035 - accuracy: 0.4976
 1280/25000 [>.............................] - ETA: 1:29 - loss: 7.7145 - accuracy: 0.4969
 1312/25000 [>.............................] - ETA: 1:28 - loss: 7.7251 - accuracy: 0.4962
 1344/25000 [>.............................] - ETA: 1:28 - loss: 7.7123 - accuracy: 0.4970
 1376/25000 [>.............................] - ETA: 1:27 - loss: 7.6443 - accuracy: 0.5015
 1408/25000 [>.............................] - ETA: 1:27 - loss: 7.6122 - accuracy: 0.5036
 1440/25000 [>.............................] - ETA: 1:27 - loss: 7.6027 - accuracy: 0.5042
 1472/25000 [>.............................] - ETA: 1:26 - loss: 7.6354 - accuracy: 0.5020
 1504/25000 [>.............................] - ETA: 1:26 - loss: 7.6360 - accuracy: 0.5020
 1536/25000 [>.............................] - ETA: 1:26 - loss: 7.5967 - accuracy: 0.5046
 1568/25000 [>.............................] - ETA: 1:26 - loss: 7.5493 - accuracy: 0.5077
 1600/25000 [>.............................] - ETA: 1:25 - loss: 7.5229 - accuracy: 0.5094
 1632/25000 [>.............................] - ETA: 1:25 - loss: 7.5351 - accuracy: 0.5086
 1664/25000 [>.............................] - ETA: 1:25 - loss: 7.5192 - accuracy: 0.5096
 1696/25000 [=>............................] - ETA: 1:25 - loss: 7.5039 - accuracy: 0.5106
 1728/25000 [=>............................] - ETA: 1:24 - loss: 7.4892 - accuracy: 0.5116
 1760/25000 [=>............................] - ETA: 1:24 - loss: 7.4488 - accuracy: 0.5142
 1792/25000 [=>............................] - ETA: 1:24 - loss: 7.4613 - accuracy: 0.5134
 1824/25000 [=>............................] - ETA: 1:24 - loss: 7.4901 - accuracy: 0.5115
 1856/25000 [=>............................] - ETA: 1:23 - loss: 7.4931 - accuracy: 0.5113
 1888/25000 [=>............................] - ETA: 1:23 - loss: 7.4717 - accuracy: 0.5127
 1920/25000 [=>............................] - ETA: 1:23 - loss: 7.4590 - accuracy: 0.5135
 1952/25000 [=>............................] - ETA: 1:22 - loss: 7.4231 - accuracy: 0.5159
 1984/25000 [=>............................] - ETA: 1:22 - loss: 7.3807 - accuracy: 0.5186
 2016/25000 [=>............................] - ETA: 1:22 - loss: 7.4156 - accuracy: 0.5164
 2048/25000 [=>............................] - ETA: 1:22 - loss: 7.4345 - accuracy: 0.5151
 2080/25000 [=>............................] - ETA: 1:21 - loss: 7.4602 - accuracy: 0.5135
 2112/25000 [=>............................] - ETA: 1:21 - loss: 7.4706 - accuracy: 0.5128
 2144/25000 [=>............................] - ETA: 1:21 - loss: 7.4807 - accuracy: 0.5121
 2176/25000 [=>............................] - ETA: 1:21 - loss: 7.4623 - accuracy: 0.5133
 2208/25000 [=>............................] - ETA: 1:21 - loss: 7.4652 - accuracy: 0.5131
 2240/25000 [=>............................] - ETA: 1:20 - loss: 7.4681 - accuracy: 0.5129
 2272/25000 [=>............................] - ETA: 1:20 - loss: 7.4912 - accuracy: 0.5114
 2304/25000 [=>............................] - ETA: 1:20 - loss: 7.4869 - accuracy: 0.5117
 2336/25000 [=>............................] - ETA: 1:20 - loss: 7.4828 - accuracy: 0.5120
 2368/25000 [=>............................] - ETA: 1:20 - loss: 7.4724 - accuracy: 0.5127
 2400/25000 [=>............................] - ETA: 1:19 - loss: 7.4558 - accuracy: 0.5138
 2432/25000 [=>............................] - ETA: 1:19 - loss: 7.4712 - accuracy: 0.5127
 2464/25000 [=>............................] - ETA: 1:19 - loss: 7.4862 - accuracy: 0.5118
 2496/25000 [=>............................] - ETA: 1:19 - loss: 7.4823 - accuracy: 0.5120
 2528/25000 [==>...........................] - ETA: 1:19 - loss: 7.4847 - accuracy: 0.5119
 2560/25000 [==>...........................] - ETA: 1:19 - loss: 7.4750 - accuracy: 0.5125
 2592/25000 [==>...........................] - ETA: 1:18 - loss: 7.5010 - accuracy: 0.5108
 2624/25000 [==>...........................] - ETA: 1:18 - loss: 7.5147 - accuracy: 0.5099
 2656/25000 [==>...........................] - ETA: 1:18 - loss: 7.5050 - accuracy: 0.5105
 2688/25000 [==>...........................] - ETA: 1:18 - loss: 7.4955 - accuracy: 0.5112
 2720/25000 [==>...........................] - ETA: 1:18 - loss: 7.5088 - accuracy: 0.5103
 2752/25000 [==>...........................] - ETA: 1:17 - loss: 7.5329 - accuracy: 0.5087
 2784/25000 [==>...........................] - ETA: 1:17 - loss: 7.5455 - accuracy: 0.5079
 2816/25000 [==>...........................] - ETA: 1:17 - loss: 7.5632 - accuracy: 0.5067
 2848/25000 [==>...........................] - ETA: 1:17 - loss: 7.5482 - accuracy: 0.5077
 2880/25000 [==>...........................] - ETA: 1:17 - loss: 7.5868 - accuracy: 0.5052
 2912/25000 [==>...........................] - ETA: 1:17 - loss: 7.5718 - accuracy: 0.5062
 2944/25000 [==>...........................] - ETA: 1:17 - loss: 7.5781 - accuracy: 0.5058
 2976/25000 [==>...........................] - ETA: 1:16 - loss: 7.5842 - accuracy: 0.5054
 3008/25000 [==>...........................] - ETA: 1:16 - loss: 7.6156 - accuracy: 0.5033
 3040/25000 [==>...........................] - ETA: 1:16 - loss: 7.6313 - accuracy: 0.5023
 3072/25000 [==>...........................] - ETA: 1:16 - loss: 7.6516 - accuracy: 0.5010
 3104/25000 [==>...........................] - ETA: 1:16 - loss: 7.6765 - accuracy: 0.4994
 3136/25000 [==>...........................] - ETA: 1:16 - loss: 7.6715 - accuracy: 0.4997
 3168/25000 [==>...........................] - ETA: 1:15 - loss: 7.6811 - accuracy: 0.4991
 3200/25000 [==>...........................] - ETA: 1:15 - loss: 7.6858 - accuracy: 0.4988
 3232/25000 [==>...........................] - ETA: 1:15 - loss: 7.6903 - accuracy: 0.4985
 3264/25000 [==>...........................] - ETA: 1:15 - loss: 7.7183 - accuracy: 0.4966
 3296/25000 [==>...........................] - ETA: 1:15 - loss: 7.7131 - accuracy: 0.4970
 3328/25000 [==>...........................] - ETA: 1:15 - loss: 7.7173 - accuracy: 0.4967
 3360/25000 [===>..........................] - ETA: 1:15 - loss: 7.7351 - accuracy: 0.4955
 3392/25000 [===>..........................] - ETA: 1:15 - loss: 7.7435 - accuracy: 0.4950
 3424/25000 [===>..........................] - ETA: 1:15 - loss: 7.7383 - accuracy: 0.4953
 3456/25000 [===>..........................] - ETA: 1:14 - loss: 7.7243 - accuracy: 0.4962
 3488/25000 [===>..........................] - ETA: 1:14 - loss: 7.7501 - accuracy: 0.4946
 3520/25000 [===>..........................] - ETA: 1:14 - loss: 7.7363 - accuracy: 0.4955
 3552/25000 [===>..........................] - ETA: 1:14 - loss: 7.7400 - accuracy: 0.4952
 3584/25000 [===>..........................] - ETA: 1:14 - loss: 7.7393 - accuracy: 0.4953
 3616/25000 [===>..........................] - ETA: 1:14 - loss: 7.7429 - accuracy: 0.4950
 3648/25000 [===>..........................] - ETA: 1:14 - loss: 7.7381 - accuracy: 0.4953
 3680/25000 [===>..........................] - ETA: 1:13 - loss: 7.7416 - accuracy: 0.4951
 3712/25000 [===>..........................] - ETA: 1:13 - loss: 7.7410 - accuracy: 0.4952
 3744/25000 [===>..........................] - ETA: 1:13 - loss: 7.7690 - accuracy: 0.4933
 3776/25000 [===>..........................] - ETA: 1:13 - loss: 7.7641 - accuracy: 0.4936
 3808/25000 [===>..........................] - ETA: 1:13 - loss: 7.7472 - accuracy: 0.4947
 3840/25000 [===>..........................] - ETA: 1:13 - loss: 7.7505 - accuracy: 0.4945
 3872/25000 [===>..........................] - ETA: 1:13 - loss: 7.7577 - accuracy: 0.4941
 3904/25000 [===>..........................] - ETA: 1:13 - loss: 7.7295 - accuracy: 0.4959
 3936/25000 [===>..........................] - ETA: 1:12 - loss: 7.7251 - accuracy: 0.4962
 3968/25000 [===>..........................] - ETA: 1:12 - loss: 7.7323 - accuracy: 0.4957
 4000/25000 [===>..........................] - ETA: 1:12 - loss: 7.7318 - accuracy: 0.4958
 4032/25000 [===>..........................] - ETA: 1:12 - loss: 7.7389 - accuracy: 0.4953
 4064/25000 [===>..........................] - ETA: 1:12 - loss: 7.7496 - accuracy: 0.4946
 4096/25000 [===>..........................] - ETA: 1:12 - loss: 7.7639 - accuracy: 0.4937
 4128/25000 [===>..........................] - ETA: 1:12 - loss: 7.7632 - accuracy: 0.4937
 4160/25000 [===>..........................] - ETA: 1:12 - loss: 7.7735 - accuracy: 0.4930
 4192/25000 [====>.........................] - ETA: 1:11 - loss: 7.7910 - accuracy: 0.4919
 4224/25000 [====>.........................] - ETA: 1:11 - loss: 7.7755 - accuracy: 0.4929
 4256/25000 [====>.........................] - ETA: 1:11 - loss: 7.7711 - accuracy: 0.4932
 4288/25000 [====>.........................] - ETA: 1:11 - loss: 7.7560 - accuracy: 0.4942
 4320/25000 [====>.........................] - ETA: 1:11 - loss: 7.7483 - accuracy: 0.4947
 4352/25000 [====>.........................] - ETA: 1:11 - loss: 7.7582 - accuracy: 0.4940
 4384/25000 [====>.........................] - ETA: 1:11 - loss: 7.7680 - accuracy: 0.4934
 4416/25000 [====>.........................] - ETA: 1:11 - loss: 7.7673 - accuracy: 0.4934
 4448/25000 [====>.........................] - ETA: 1:11 - loss: 7.7562 - accuracy: 0.4942
 4480/25000 [====>.........................] - ETA: 1:10 - loss: 7.7488 - accuracy: 0.4946
 4512/25000 [====>.........................] - ETA: 1:10 - loss: 7.7482 - accuracy: 0.4947
 4544/25000 [====>.........................] - ETA: 1:10 - loss: 7.7476 - accuracy: 0.4947
 4576/25000 [====>.........................] - ETA: 1:10 - loss: 7.7437 - accuracy: 0.4950
 4608/25000 [====>.........................] - ETA: 1:10 - loss: 7.7465 - accuracy: 0.4948
 4640/25000 [====>.........................] - ETA: 1:10 - loss: 7.7591 - accuracy: 0.4940
 4672/25000 [====>.........................] - ETA: 1:10 - loss: 7.7716 - accuracy: 0.4932
 4704/25000 [====>.........................] - ETA: 1:10 - loss: 7.7611 - accuracy: 0.4938
 4736/25000 [====>.........................] - ETA: 1:10 - loss: 7.7540 - accuracy: 0.4943
 4768/25000 [====>.........................] - ETA: 1:09 - loss: 7.7406 - accuracy: 0.4952
 4800/25000 [====>.........................] - ETA: 1:09 - loss: 7.7369 - accuracy: 0.4954
 4832/25000 [====>.........................] - ETA: 1:09 - loss: 7.7237 - accuracy: 0.4963
 4864/25000 [====>.........................] - ETA: 1:09 - loss: 7.7139 - accuracy: 0.4969
 4896/25000 [====>.........................] - ETA: 1:09 - loss: 7.7293 - accuracy: 0.4959
 4928/25000 [====>.........................] - ETA: 1:09 - loss: 7.7320 - accuracy: 0.4957
 4960/25000 [====>.........................] - ETA: 1:09 - loss: 7.7315 - accuracy: 0.4958
 4992/25000 [====>.........................] - ETA: 1:09 - loss: 7.7342 - accuracy: 0.4956
 5024/25000 [=====>........................] - ETA: 1:08 - loss: 7.7368 - accuracy: 0.4954
 5056/25000 [=====>........................] - ETA: 1:08 - loss: 7.7273 - accuracy: 0.4960
 5088/25000 [=====>........................] - ETA: 1:08 - loss: 7.7420 - accuracy: 0.4951
 5120/25000 [=====>........................] - ETA: 1:08 - loss: 7.7355 - accuracy: 0.4955
 5152/25000 [=====>........................] - ETA: 1:08 - loss: 7.7321 - accuracy: 0.4957
 5184/25000 [=====>........................] - ETA: 1:08 - loss: 7.7317 - accuracy: 0.4958
 5216/25000 [=====>........................] - ETA: 1:08 - loss: 7.7401 - accuracy: 0.4952
 5248/25000 [=====>........................] - ETA: 1:08 - loss: 7.7397 - accuracy: 0.4952
 5280/25000 [=====>........................] - ETA: 1:08 - loss: 7.7421 - accuracy: 0.4951
 5312/25000 [=====>........................] - ETA: 1:07 - loss: 7.7474 - accuracy: 0.4947
 5344/25000 [=====>........................] - ETA: 1:07 - loss: 7.7470 - accuracy: 0.4948
 5376/25000 [=====>........................] - ETA: 1:07 - loss: 7.7522 - accuracy: 0.4944
 5408/25000 [=====>........................] - ETA: 1:07 - loss: 7.7573 - accuracy: 0.4941
 5440/25000 [=====>........................] - ETA: 1:07 - loss: 7.7540 - accuracy: 0.4943
 5472/25000 [=====>........................] - ETA: 1:07 - loss: 7.7563 - accuracy: 0.4942
 5504/25000 [=====>........................] - ETA: 1:07 - loss: 7.7530 - accuracy: 0.4944
 5536/25000 [=====>........................] - ETA: 1:07 - loss: 7.7442 - accuracy: 0.4949
 5568/25000 [=====>........................] - ETA: 1:06 - loss: 7.7492 - accuracy: 0.4946
 5600/25000 [=====>........................] - ETA: 1:06 - loss: 7.7515 - accuracy: 0.4945
 5632/25000 [=====>........................] - ETA: 1:06 - loss: 7.7429 - accuracy: 0.4950
 5664/25000 [=====>........................] - ETA: 1:06 - loss: 7.7451 - accuracy: 0.4949
 5696/25000 [=====>........................] - ETA: 1:06 - loss: 7.7528 - accuracy: 0.4944
 5728/25000 [=====>........................] - ETA: 1:06 - loss: 7.7576 - accuracy: 0.4941
 5760/25000 [=====>........................] - ETA: 1:06 - loss: 7.7571 - accuracy: 0.4941
 5792/25000 [=====>........................] - ETA: 1:06 - loss: 7.7646 - accuracy: 0.4936
 5824/25000 [=====>........................] - ETA: 1:05 - loss: 7.7640 - accuracy: 0.4936
 5856/25000 [======>.......................] - ETA: 1:05 - loss: 7.7635 - accuracy: 0.4937
 5888/25000 [======>.......................] - ETA: 1:05 - loss: 7.7656 - accuracy: 0.4935
 5920/25000 [======>.......................] - ETA: 1:05 - loss: 7.7754 - accuracy: 0.4929
 5952/25000 [======>.......................] - ETA: 1:05 - loss: 7.7748 - accuracy: 0.4929
 5984/25000 [======>.......................] - ETA: 1:05 - loss: 7.7691 - accuracy: 0.4933
 6016/25000 [======>.......................] - ETA: 1:05 - loss: 7.7686 - accuracy: 0.4934
 6048/25000 [======>.......................] - ETA: 1:05 - loss: 7.7579 - accuracy: 0.4940
 6080/25000 [======>.......................] - ETA: 1:05 - loss: 7.7498 - accuracy: 0.4946
 6112/25000 [======>.......................] - ETA: 1:04 - loss: 7.7620 - accuracy: 0.4938
 6144/25000 [======>.......................] - ETA: 1:04 - loss: 7.7615 - accuracy: 0.4938
 6176/25000 [======>.......................] - ETA: 1:04 - loss: 7.7610 - accuracy: 0.4938
 6208/25000 [======>.......................] - ETA: 1:04 - loss: 7.7629 - accuracy: 0.4937
 6240/25000 [======>.......................] - ETA: 1:04 - loss: 7.7551 - accuracy: 0.4942
 6272/25000 [======>.......................] - ETA: 1:04 - loss: 7.7815 - accuracy: 0.4925
 6304/25000 [======>.......................] - ETA: 1:04 - loss: 7.7809 - accuracy: 0.4925
 6336/25000 [======>.......................] - ETA: 1:04 - loss: 7.7828 - accuracy: 0.4924
 6368/25000 [======>.......................] - ETA: 1:04 - loss: 7.7702 - accuracy: 0.4932
 6400/25000 [======>.......................] - ETA: 1:03 - loss: 7.7768 - accuracy: 0.4928
 6432/25000 [======>.......................] - ETA: 1:03 - loss: 7.7787 - accuracy: 0.4927
 6464/25000 [======>.......................] - ETA: 1:03 - loss: 7.7971 - accuracy: 0.4915
 6496/25000 [======>.......................] - ETA: 1:03 - loss: 7.7988 - accuracy: 0.4914
 6528/25000 [======>.......................] - ETA: 1:03 - loss: 7.8075 - accuracy: 0.4908
 6560/25000 [======>.......................] - ETA: 1:03 - loss: 7.8069 - accuracy: 0.4909
 6592/25000 [======>.......................] - ETA: 1:03 - loss: 7.8085 - accuracy: 0.4907
 6624/25000 [======>.......................] - ETA: 1:03 - loss: 7.8101 - accuracy: 0.4906
 6656/25000 [======>.......................] - ETA: 1:02 - loss: 7.8025 - accuracy: 0.4911
 6688/25000 [=======>......................] - ETA: 1:02 - loss: 7.8088 - accuracy: 0.4907
 6720/25000 [=======>......................] - ETA: 1:02 - loss: 7.8058 - accuracy: 0.4909
 6752/25000 [=======>......................] - ETA: 1:02 - loss: 7.8051 - accuracy: 0.4910
 6784/25000 [=======>......................] - ETA: 1:02 - loss: 7.8068 - accuracy: 0.4909
 6816/25000 [=======>......................] - ETA: 1:02 - loss: 7.8106 - accuracy: 0.4906
 6848/25000 [=======>......................] - ETA: 1:02 - loss: 7.8211 - accuracy: 0.4899
 6880/25000 [=======>......................] - ETA: 1:02 - loss: 7.8182 - accuracy: 0.4901
 6912/25000 [=======>......................] - ETA: 1:02 - loss: 7.8019 - accuracy: 0.4912
 6944/25000 [=======>......................] - ETA: 1:01 - loss: 7.7991 - accuracy: 0.4914
 6976/25000 [=======>......................] - ETA: 1:01 - loss: 7.8051 - accuracy: 0.4910
 7008/25000 [=======>......................] - ETA: 1:01 - loss: 7.8023 - accuracy: 0.4912
 7040/25000 [=======>......................] - ETA: 1:01 - loss: 7.8017 - accuracy: 0.4912
 7072/25000 [=======>......................] - ETA: 1:01 - loss: 7.8032 - accuracy: 0.4911
 7104/25000 [=======>......................] - ETA: 1:01 - loss: 7.8134 - accuracy: 0.4904
 7136/25000 [=======>......................] - ETA: 1:01 - loss: 7.8149 - accuracy: 0.4903
 7168/25000 [=======>......................] - ETA: 1:01 - loss: 7.8206 - accuracy: 0.4900
 7200/25000 [=======>......................] - ETA: 1:01 - loss: 7.8157 - accuracy: 0.4903
 7232/25000 [=======>......................] - ETA: 1:01 - loss: 7.8172 - accuracy: 0.4902
 7264/25000 [=======>......................] - ETA: 1:00 - loss: 7.8080 - accuracy: 0.4908
 7296/25000 [=======>......................] - ETA: 1:00 - loss: 7.8116 - accuracy: 0.4905
 7328/25000 [=======>......................] - ETA: 1:00 - loss: 7.8005 - accuracy: 0.4913
 7360/25000 [=======>......................] - ETA: 1:00 - loss: 7.8000 - accuracy: 0.4913
 7392/25000 [=======>......................] - ETA: 1:00 - loss: 7.7973 - accuracy: 0.4915
 7424/25000 [=======>......................] - ETA: 1:00 - loss: 7.7967 - accuracy: 0.4915
 7456/25000 [=======>......................] - ETA: 1:00 - loss: 7.8003 - accuracy: 0.4913
 7488/25000 [=======>......................] - ETA: 1:00 - loss: 7.8018 - accuracy: 0.4912
 7520/25000 [========>.....................] - ETA: 1:00 - loss: 7.8053 - accuracy: 0.4910
 7552/25000 [========>.....................] - ETA: 1:00 - loss: 7.8169 - accuracy: 0.4902
 7584/25000 [========>.....................] - ETA: 59s - loss: 7.8183 - accuracy: 0.4901 
 7616/25000 [========>.....................] - ETA: 59s - loss: 7.8196 - accuracy: 0.4900
 7648/25000 [========>.....................] - ETA: 59s - loss: 7.8170 - accuracy: 0.4902
 7680/25000 [========>.....................] - ETA: 59s - loss: 7.8084 - accuracy: 0.4908
 7712/25000 [========>.....................] - ETA: 59s - loss: 7.8137 - accuracy: 0.4904
 7744/25000 [========>.....................] - ETA: 59s - loss: 7.8092 - accuracy: 0.4907
 7776/25000 [========>.....................] - ETA: 59s - loss: 7.8185 - accuracy: 0.4901
 7808/25000 [========>.....................] - ETA: 59s - loss: 7.8178 - accuracy: 0.4901
 7840/25000 [========>.....................] - ETA: 59s - loss: 7.8113 - accuracy: 0.4906
 7872/25000 [========>.....................] - ETA: 59s - loss: 7.8166 - accuracy: 0.4902
 7904/25000 [========>.....................] - ETA: 59s - loss: 7.8082 - accuracy: 0.4908
 7936/25000 [========>.....................] - ETA: 58s - loss: 7.8057 - accuracy: 0.4909
 7968/25000 [========>.....................] - ETA: 58s - loss: 7.7975 - accuracy: 0.4915
 8000/25000 [========>.....................] - ETA: 58s - loss: 7.7950 - accuracy: 0.4916
 8032/25000 [========>.....................] - ETA: 58s - loss: 7.7907 - accuracy: 0.4919
 8064/25000 [========>.....................] - ETA: 58s - loss: 7.7883 - accuracy: 0.4921
 8096/25000 [========>.....................] - ETA: 58s - loss: 7.7859 - accuracy: 0.4922
 8128/25000 [========>.....................] - ETA: 58s - loss: 7.7836 - accuracy: 0.4924
 8160/25000 [========>.....................] - ETA: 58s - loss: 7.7794 - accuracy: 0.4926
 8192/25000 [========>.....................] - ETA: 58s - loss: 7.7827 - accuracy: 0.4924
 8224/25000 [========>.....................] - ETA: 58s - loss: 7.7934 - accuracy: 0.4917
 8256/25000 [========>.....................] - ETA: 57s - loss: 7.7966 - accuracy: 0.4915
 8288/25000 [========>.....................] - ETA: 57s - loss: 7.7924 - accuracy: 0.4918
 8320/25000 [========>.....................] - ETA: 57s - loss: 7.8012 - accuracy: 0.4912
 8352/25000 [=========>....................] - ETA: 57s - loss: 7.7915 - accuracy: 0.4919
 8384/25000 [=========>....................] - ETA: 57s - loss: 7.7873 - accuracy: 0.4921
 8416/25000 [=========>....................] - ETA: 57s - loss: 7.7887 - accuracy: 0.4920
 8448/25000 [=========>....................] - ETA: 57s - loss: 7.7955 - accuracy: 0.4916
 8480/25000 [=========>....................] - ETA: 57s - loss: 7.7950 - accuracy: 0.4916
 8512/25000 [=========>....................] - ETA: 57s - loss: 7.7891 - accuracy: 0.4920
 8544/25000 [=========>....................] - ETA: 56s - loss: 7.7869 - accuracy: 0.4922
 8576/25000 [=========>....................] - ETA: 56s - loss: 7.7882 - accuracy: 0.4921
 8608/25000 [=========>....................] - ETA: 56s - loss: 7.8020 - accuracy: 0.4912
 8640/25000 [=========>....................] - ETA: 56s - loss: 7.7997 - accuracy: 0.4913
 8672/25000 [=========>....................] - ETA: 56s - loss: 7.7992 - accuracy: 0.4914
 8704/25000 [=========>....................] - ETA: 56s - loss: 7.7935 - accuracy: 0.4917
 8736/25000 [=========>....................] - ETA: 56s - loss: 7.7912 - accuracy: 0.4919
 8768/25000 [=========>....................] - ETA: 56s - loss: 7.7925 - accuracy: 0.4918
 8800/25000 [=========>....................] - ETA: 56s - loss: 7.7886 - accuracy: 0.4920
 8832/25000 [=========>....................] - ETA: 56s - loss: 7.7881 - accuracy: 0.4921
 8864/25000 [=========>....................] - ETA: 55s - loss: 7.7912 - accuracy: 0.4919
 8896/25000 [=========>....................] - ETA: 55s - loss: 7.7924 - accuracy: 0.4918
 8928/25000 [=========>....................] - ETA: 55s - loss: 7.7903 - accuracy: 0.4919
 8960/25000 [=========>....................] - ETA: 55s - loss: 7.7864 - accuracy: 0.4922
 8992/25000 [=========>....................] - ETA: 55s - loss: 7.7911 - accuracy: 0.4919
 9024/25000 [=========>....................] - ETA: 55s - loss: 7.7975 - accuracy: 0.4915
 9056/25000 [=========>....................] - ETA: 55s - loss: 7.7902 - accuracy: 0.4919
 9088/25000 [=========>....................] - ETA: 55s - loss: 7.7932 - accuracy: 0.4917
 9120/25000 [=========>....................] - ETA: 55s - loss: 7.7894 - accuracy: 0.4920
 9152/25000 [=========>....................] - ETA: 54s - loss: 7.7872 - accuracy: 0.4921
 9184/25000 [==========>...................] - ETA: 54s - loss: 7.7818 - accuracy: 0.4925
 9216/25000 [==========>...................] - ETA: 54s - loss: 7.7731 - accuracy: 0.4931
 9248/25000 [==========>...................] - ETA: 54s - loss: 7.7760 - accuracy: 0.4929
 9280/25000 [==========>...................] - ETA: 54s - loss: 7.7691 - accuracy: 0.4933
 9312/25000 [==========>...................] - ETA: 54s - loss: 7.7638 - accuracy: 0.4937
 9344/25000 [==========>...................] - ETA: 54s - loss: 7.7667 - accuracy: 0.4935
 9376/25000 [==========>...................] - ETA: 54s - loss: 7.7696 - accuracy: 0.4933
 9408/25000 [==========>...................] - ETA: 54s - loss: 7.7660 - accuracy: 0.4935
 9440/25000 [==========>...................] - ETA: 53s - loss: 7.7608 - accuracy: 0.4939
 9472/25000 [==========>...................] - ETA: 53s - loss: 7.7621 - accuracy: 0.4938
 9504/25000 [==========>...................] - ETA: 53s - loss: 7.7650 - accuracy: 0.4936
 9536/25000 [==========>...................] - ETA: 53s - loss: 7.7599 - accuracy: 0.4939
 9568/25000 [==========>...................] - ETA: 53s - loss: 7.7660 - accuracy: 0.4935
 9600/25000 [==========>...................] - ETA: 53s - loss: 7.7593 - accuracy: 0.4940
 9632/25000 [==========>...................] - ETA: 53s - loss: 7.7590 - accuracy: 0.4940
 9664/25000 [==========>...................] - ETA: 53s - loss: 7.7571 - accuracy: 0.4941
 9696/25000 [==========>...................] - ETA: 53s - loss: 7.7552 - accuracy: 0.4942
 9728/25000 [==========>...................] - ETA: 52s - loss: 7.7549 - accuracy: 0.4942
 9760/25000 [==========>...................] - ETA: 52s - loss: 7.7577 - accuracy: 0.4941
 9792/25000 [==========>...................] - ETA: 52s - loss: 7.7559 - accuracy: 0.4942
 9824/25000 [==========>...................] - ETA: 52s - loss: 7.7509 - accuracy: 0.4945
 9856/25000 [==========>...................] - ETA: 52s - loss: 7.7506 - accuracy: 0.4945
 9888/25000 [==========>...................] - ETA: 52s - loss: 7.7504 - accuracy: 0.4945
 9920/25000 [==========>...................] - ETA: 52s - loss: 7.7563 - accuracy: 0.4942
 9952/25000 [==========>...................] - ETA: 52s - loss: 7.7621 - accuracy: 0.4938
 9984/25000 [==========>...................] - ETA: 52s - loss: 7.7618 - accuracy: 0.4938
10016/25000 [===========>..................] - ETA: 51s - loss: 7.7585 - accuracy: 0.4940
10048/25000 [===========>..................] - ETA: 51s - loss: 7.7612 - accuracy: 0.4938
10080/25000 [===========>..................] - ETA: 51s - loss: 7.7564 - accuracy: 0.4941
10112/25000 [===========>..................] - ETA: 51s - loss: 7.7500 - accuracy: 0.4946
10144/25000 [===========>..................] - ETA: 51s - loss: 7.7558 - accuracy: 0.4942
10176/25000 [===========>..................] - ETA: 51s - loss: 7.7555 - accuracy: 0.4942
10208/25000 [===========>..................] - ETA: 51s - loss: 7.7522 - accuracy: 0.4944
10240/25000 [===========>..................] - ETA: 51s - loss: 7.7520 - accuracy: 0.4944
10272/25000 [===========>..................] - ETA: 50s - loss: 7.7547 - accuracy: 0.4943
10304/25000 [===========>..................] - ETA: 50s - loss: 7.7559 - accuracy: 0.4942
10336/25000 [===========>..................] - ETA: 50s - loss: 7.7556 - accuracy: 0.4942
10368/25000 [===========>..................] - ETA: 50s - loss: 7.7554 - accuracy: 0.4942
10400/25000 [===========>..................] - ETA: 50s - loss: 7.7595 - accuracy: 0.4939
10432/25000 [===========>..................] - ETA: 50s - loss: 7.7680 - accuracy: 0.4934
10464/25000 [===========>..................] - ETA: 50s - loss: 7.7677 - accuracy: 0.4934
10496/25000 [===========>..................] - ETA: 50s - loss: 7.7660 - accuracy: 0.4935
10528/25000 [===========>..................] - ETA: 50s - loss: 7.7671 - accuracy: 0.4934
10560/25000 [===========>..................] - ETA: 50s - loss: 7.7668 - accuracy: 0.4935
10592/25000 [===========>..................] - ETA: 49s - loss: 7.7651 - accuracy: 0.4936
10624/25000 [===========>..................] - ETA: 49s - loss: 7.7532 - accuracy: 0.4944
10656/25000 [===========>..................] - ETA: 49s - loss: 7.7515 - accuracy: 0.4945
10688/25000 [===========>..................] - ETA: 49s - loss: 7.7441 - accuracy: 0.4949
10720/25000 [===========>..................] - ETA: 49s - loss: 7.7439 - accuracy: 0.4950
10752/25000 [===========>..................] - ETA: 49s - loss: 7.7422 - accuracy: 0.4951
10784/25000 [===========>..................] - ETA: 49s - loss: 7.7377 - accuracy: 0.4954
10816/25000 [===========>..................] - ETA: 49s - loss: 7.7418 - accuracy: 0.4951
10848/25000 [============>.................] - ETA: 49s - loss: 7.7316 - accuracy: 0.4958
10880/25000 [============>.................] - ETA: 48s - loss: 7.7357 - accuracy: 0.4955
10912/25000 [============>.................] - ETA: 48s - loss: 7.7284 - accuracy: 0.4960
10944/25000 [============>.................] - ETA: 48s - loss: 7.7255 - accuracy: 0.4962
10976/25000 [============>.................] - ETA: 48s - loss: 7.7211 - accuracy: 0.4964
11008/25000 [============>.................] - ETA: 48s - loss: 7.7237 - accuracy: 0.4963
11040/25000 [============>.................] - ETA: 48s - loss: 7.7277 - accuracy: 0.4960
11072/25000 [============>.................] - ETA: 48s - loss: 7.7262 - accuracy: 0.4961
11104/25000 [============>.................] - ETA: 48s - loss: 7.7288 - accuracy: 0.4959
11136/25000 [============>.................] - ETA: 48s - loss: 7.7272 - accuracy: 0.4960
11168/25000 [============>.................] - ETA: 48s - loss: 7.7160 - accuracy: 0.4968
11200/25000 [============>.................] - ETA: 47s - loss: 7.7159 - accuracy: 0.4968
11232/25000 [============>.................] - ETA: 47s - loss: 7.7171 - accuracy: 0.4967
11264/25000 [============>.................] - ETA: 47s - loss: 7.7102 - accuracy: 0.4972
11296/25000 [============>.................] - ETA: 47s - loss: 7.7087 - accuracy: 0.4973
11328/25000 [============>.................] - ETA: 47s - loss: 7.7126 - accuracy: 0.4970
11360/25000 [============>.................] - ETA: 47s - loss: 7.7085 - accuracy: 0.4973
11392/25000 [============>.................] - ETA: 47s - loss: 7.7083 - accuracy: 0.4973
11424/25000 [============>.................] - ETA: 47s - loss: 7.7096 - accuracy: 0.4972
11456/25000 [============>.................] - ETA: 47s - loss: 7.7121 - accuracy: 0.4970
11488/25000 [============>.................] - ETA: 46s - loss: 7.7067 - accuracy: 0.4974
11520/25000 [============>.................] - ETA: 46s - loss: 7.6999 - accuracy: 0.4978
11552/25000 [============>.................] - ETA: 46s - loss: 7.6918 - accuracy: 0.4984
11584/25000 [============>.................] - ETA: 46s - loss: 7.6891 - accuracy: 0.4985
11616/25000 [============>.................] - ETA: 46s - loss: 7.6904 - accuracy: 0.4985
11648/25000 [============>.................] - ETA: 46s - loss: 7.6864 - accuracy: 0.4987
11680/25000 [=============>................] - ETA: 46s - loss: 7.6863 - accuracy: 0.4987
11712/25000 [=============>................] - ETA: 46s - loss: 7.6823 - accuracy: 0.4990
11744/25000 [=============>................] - ETA: 46s - loss: 7.6810 - accuracy: 0.4991
11776/25000 [=============>................] - ETA: 45s - loss: 7.6783 - accuracy: 0.4992
11808/25000 [=============>................] - ETA: 45s - loss: 7.6809 - accuracy: 0.4991
11840/25000 [=============>................] - ETA: 45s - loss: 7.6783 - accuracy: 0.4992
11872/25000 [=============>................] - ETA: 45s - loss: 7.6808 - accuracy: 0.4991
11904/25000 [=============>................] - ETA: 45s - loss: 7.6834 - accuracy: 0.4989
11936/25000 [=============>................] - ETA: 45s - loss: 7.6885 - accuracy: 0.4986
11968/25000 [=============>................] - ETA: 45s - loss: 7.6922 - accuracy: 0.4983
12000/25000 [=============>................] - ETA: 45s - loss: 7.6909 - accuracy: 0.4984
12032/25000 [=============>................] - ETA: 45s - loss: 7.6870 - accuracy: 0.4987
12064/25000 [=============>................] - ETA: 44s - loss: 7.6882 - accuracy: 0.4986
12096/25000 [=============>................] - ETA: 44s - loss: 7.6882 - accuracy: 0.4986
12128/25000 [=============>................] - ETA: 44s - loss: 7.6906 - accuracy: 0.4984
12160/25000 [=============>................] - ETA: 44s - loss: 7.6893 - accuracy: 0.4985
12192/25000 [=============>................] - ETA: 44s - loss: 7.6893 - accuracy: 0.4985
12224/25000 [=============>................] - ETA: 44s - loss: 7.6892 - accuracy: 0.4985
12256/25000 [=============>................] - ETA: 44s - loss: 7.6866 - accuracy: 0.4987
12288/25000 [=============>................] - ETA: 44s - loss: 7.6841 - accuracy: 0.4989
12320/25000 [=============>................] - ETA: 44s - loss: 7.6840 - accuracy: 0.4989
12352/25000 [=============>................] - ETA: 43s - loss: 7.6815 - accuracy: 0.4990
12384/25000 [=============>................] - ETA: 43s - loss: 7.6852 - accuracy: 0.4988
12416/25000 [=============>................] - ETA: 43s - loss: 7.6864 - accuracy: 0.4987
12448/25000 [=============>................] - ETA: 43s - loss: 7.6876 - accuracy: 0.4986
12480/25000 [=============>................] - ETA: 43s - loss: 7.6900 - accuracy: 0.4985
12512/25000 [==============>...............] - ETA: 43s - loss: 7.6936 - accuracy: 0.4982
12544/25000 [==============>...............] - ETA: 43s - loss: 7.6935 - accuracy: 0.4982
12576/25000 [==============>...............] - ETA: 43s - loss: 7.6873 - accuracy: 0.4986
12608/25000 [==============>...............] - ETA: 43s - loss: 7.6873 - accuracy: 0.4987
12640/25000 [==============>...............] - ETA: 42s - loss: 7.6872 - accuracy: 0.4987
12672/25000 [==============>...............] - ETA: 42s - loss: 7.6896 - accuracy: 0.4985
12704/25000 [==============>...............] - ETA: 42s - loss: 7.6908 - accuracy: 0.4984
12736/25000 [==============>...............] - ETA: 42s - loss: 7.6895 - accuracy: 0.4985
12768/25000 [==============>...............] - ETA: 42s - loss: 7.6918 - accuracy: 0.4984
12800/25000 [==============>...............] - ETA: 42s - loss: 7.6894 - accuracy: 0.4985
12832/25000 [==============>...............] - ETA: 42s - loss: 7.6893 - accuracy: 0.4985
12864/25000 [==============>...............] - ETA: 42s - loss: 7.6881 - accuracy: 0.4986
12896/25000 [==============>...............] - ETA: 42s - loss: 7.6892 - accuracy: 0.4985
12928/25000 [==============>...............] - ETA: 41s - loss: 7.6915 - accuracy: 0.4984
12960/25000 [==============>...............] - ETA: 41s - loss: 7.6962 - accuracy: 0.4981
12992/25000 [==============>...............] - ETA: 41s - loss: 7.6973 - accuracy: 0.4980
13024/25000 [==============>...............] - ETA: 41s - loss: 7.6984 - accuracy: 0.4979
13056/25000 [==============>...............] - ETA: 41s - loss: 7.6913 - accuracy: 0.4984
13088/25000 [==============>...............] - ETA: 41s - loss: 7.7006 - accuracy: 0.4978
13120/25000 [==============>...............] - ETA: 41s - loss: 7.7005 - accuracy: 0.4978
13152/25000 [==============>...............] - ETA: 41s - loss: 7.6993 - accuracy: 0.4979
13184/25000 [==============>...............] - ETA: 40s - loss: 7.6969 - accuracy: 0.4980
13216/25000 [==============>...............] - ETA: 40s - loss: 7.6945 - accuracy: 0.4982
13248/25000 [==============>...............] - ETA: 40s - loss: 7.6932 - accuracy: 0.4983
13280/25000 [==============>...............] - ETA: 40s - loss: 7.6920 - accuracy: 0.4983
13312/25000 [==============>...............] - ETA: 40s - loss: 7.6977 - accuracy: 0.4980
13344/25000 [===============>..............] - ETA: 40s - loss: 7.6965 - accuracy: 0.4981
13376/25000 [===============>..............] - ETA: 40s - loss: 7.6953 - accuracy: 0.4981
13408/25000 [===============>..............] - ETA: 40s - loss: 7.6952 - accuracy: 0.4981
13440/25000 [===============>..............] - ETA: 40s - loss: 7.6951 - accuracy: 0.4981
13472/25000 [===============>..............] - ETA: 40s - loss: 7.6962 - accuracy: 0.4981
13504/25000 [===============>..............] - ETA: 39s - loss: 7.6927 - accuracy: 0.4983
13536/25000 [===============>..............] - ETA: 39s - loss: 7.6927 - accuracy: 0.4983
13568/25000 [===============>..............] - ETA: 39s - loss: 7.6904 - accuracy: 0.4985
13600/25000 [===============>..............] - ETA: 39s - loss: 7.6903 - accuracy: 0.4985
13632/25000 [===============>..............] - ETA: 39s - loss: 7.6846 - accuracy: 0.4988
13664/25000 [===============>..............] - ETA: 39s - loss: 7.6868 - accuracy: 0.4987
13696/25000 [===============>..............] - ETA: 39s - loss: 7.6890 - accuracy: 0.4985
13728/25000 [===============>..............] - ETA: 39s - loss: 7.6878 - accuracy: 0.4986
13760/25000 [===============>..............] - ETA: 39s - loss: 7.6889 - accuracy: 0.4985
13792/25000 [===============>..............] - ETA: 38s - loss: 7.6844 - accuracy: 0.4988
13824/25000 [===============>..............] - ETA: 38s - loss: 7.6866 - accuracy: 0.4987
13856/25000 [===============>..............] - ETA: 38s - loss: 7.6843 - accuracy: 0.4988
13888/25000 [===============>..............] - ETA: 38s - loss: 7.6887 - accuracy: 0.4986
13920/25000 [===============>..............] - ETA: 38s - loss: 7.6875 - accuracy: 0.4986
13952/25000 [===============>..............] - ETA: 38s - loss: 7.6809 - accuracy: 0.4991
13984/25000 [===============>..............] - ETA: 38s - loss: 7.6776 - accuracy: 0.4993
14016/25000 [===============>..............] - ETA: 38s - loss: 7.6787 - accuracy: 0.4992
14048/25000 [===============>..............] - ETA: 38s - loss: 7.6841 - accuracy: 0.4989
14080/25000 [===============>..............] - ETA: 37s - loss: 7.6884 - accuracy: 0.4986
14112/25000 [===============>..............] - ETA: 37s - loss: 7.6851 - accuracy: 0.4988
14144/25000 [===============>..............] - ETA: 37s - loss: 7.6861 - accuracy: 0.4987
14176/25000 [================>.............] - ETA: 37s - loss: 7.6861 - accuracy: 0.4987
14208/25000 [================>.............] - ETA: 37s - loss: 7.6904 - accuracy: 0.4985
14240/25000 [================>.............] - ETA: 37s - loss: 7.6892 - accuracy: 0.4985
14272/25000 [================>.............] - ETA: 37s - loss: 7.6860 - accuracy: 0.4987
14304/25000 [================>.............] - ETA: 37s - loss: 7.6870 - accuracy: 0.4987
14336/25000 [================>.............] - ETA: 37s - loss: 7.6848 - accuracy: 0.4988
14368/25000 [================>.............] - ETA: 36s - loss: 7.6826 - accuracy: 0.4990
14400/25000 [================>.............] - ETA: 36s - loss: 7.6815 - accuracy: 0.4990
14432/25000 [================>.............] - ETA: 36s - loss: 7.6836 - accuracy: 0.4989
14464/25000 [================>.............] - ETA: 36s - loss: 7.6836 - accuracy: 0.4989
14496/25000 [================>.............] - ETA: 36s - loss: 7.6867 - accuracy: 0.4987
14528/25000 [================>.............] - ETA: 36s - loss: 7.6825 - accuracy: 0.4990
14560/25000 [================>.............] - ETA: 36s - loss: 7.6782 - accuracy: 0.4992
14592/25000 [================>.............] - ETA: 36s - loss: 7.6834 - accuracy: 0.4989
14624/25000 [================>.............] - ETA: 36s - loss: 7.6792 - accuracy: 0.4992
14656/25000 [================>.............] - ETA: 35s - loss: 7.6792 - accuracy: 0.4992
14688/25000 [================>.............] - ETA: 35s - loss: 7.6771 - accuracy: 0.4993
14720/25000 [================>.............] - ETA: 35s - loss: 7.6781 - accuracy: 0.4993
14752/25000 [================>.............] - ETA: 35s - loss: 7.6781 - accuracy: 0.4993
14784/25000 [================>.............] - ETA: 35s - loss: 7.6822 - accuracy: 0.4990
14816/25000 [================>.............] - ETA: 35s - loss: 7.6842 - accuracy: 0.4989
14848/25000 [================>.............] - ETA: 35s - loss: 7.6821 - accuracy: 0.4990
14880/25000 [================>.............] - ETA: 35s - loss: 7.6810 - accuracy: 0.4991
14912/25000 [================>.............] - ETA: 35s - loss: 7.6841 - accuracy: 0.4989
14944/25000 [================>.............] - ETA: 34s - loss: 7.6830 - accuracy: 0.4989
14976/25000 [================>.............] - ETA: 34s - loss: 7.6789 - accuracy: 0.4992
15008/25000 [=================>............] - ETA: 34s - loss: 7.6758 - accuracy: 0.4994
15040/25000 [=================>............] - ETA: 34s - loss: 7.6768 - accuracy: 0.4993
15072/25000 [=================>............] - ETA: 34s - loss: 7.6707 - accuracy: 0.4997
15104/25000 [=================>............] - ETA: 34s - loss: 7.6768 - accuracy: 0.4993
15136/25000 [=================>............] - ETA: 34s - loss: 7.6778 - accuracy: 0.4993
15168/25000 [=================>............] - ETA: 34s - loss: 7.6828 - accuracy: 0.4989
15200/25000 [=================>............] - ETA: 33s - loss: 7.6787 - accuracy: 0.4992
15232/25000 [=================>............] - ETA: 33s - loss: 7.6727 - accuracy: 0.4996
15264/25000 [=================>............] - ETA: 33s - loss: 7.6747 - accuracy: 0.4995
15296/25000 [=================>............] - ETA: 33s - loss: 7.6736 - accuracy: 0.4995
15328/25000 [=================>............] - ETA: 33s - loss: 7.6676 - accuracy: 0.4999
15360/25000 [=================>............] - ETA: 33s - loss: 7.6696 - accuracy: 0.4998
15392/25000 [=================>............] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
15424/25000 [=================>............] - ETA: 33s - loss: 7.6607 - accuracy: 0.5004
15456/25000 [=================>............] - ETA: 33s - loss: 7.6567 - accuracy: 0.5006
15488/25000 [=================>............] - ETA: 32s - loss: 7.6557 - accuracy: 0.5007
15520/25000 [=================>............] - ETA: 32s - loss: 7.6567 - accuracy: 0.5006
15552/25000 [=================>............] - ETA: 32s - loss: 7.6538 - accuracy: 0.5008
15584/25000 [=================>............] - ETA: 32s - loss: 7.6558 - accuracy: 0.5007
15616/25000 [=================>............] - ETA: 32s - loss: 7.6499 - accuracy: 0.5011
15648/25000 [=================>............] - ETA: 32s - loss: 7.6519 - accuracy: 0.5010
15680/25000 [=================>............] - ETA: 32s - loss: 7.6549 - accuracy: 0.5008
15712/25000 [=================>............] - ETA: 32s - loss: 7.6520 - accuracy: 0.5010
15744/25000 [=================>............] - ETA: 32s - loss: 7.6481 - accuracy: 0.5012
15776/25000 [=================>............] - ETA: 32s - loss: 7.6472 - accuracy: 0.5013
15808/25000 [=================>............] - ETA: 31s - loss: 7.6482 - accuracy: 0.5012
15840/25000 [==================>...........] - ETA: 31s - loss: 7.6511 - accuracy: 0.5010
15872/25000 [==================>...........] - ETA: 31s - loss: 7.6531 - accuracy: 0.5009
15904/25000 [==================>...........] - ETA: 31s - loss: 7.6551 - accuracy: 0.5008
15936/25000 [==================>...........] - ETA: 31s - loss: 7.6493 - accuracy: 0.5011
15968/25000 [==================>...........] - ETA: 31s - loss: 7.6513 - accuracy: 0.5010
16000/25000 [==================>...........] - ETA: 31s - loss: 7.6513 - accuracy: 0.5010
16032/25000 [==================>...........] - ETA: 31s - loss: 7.6571 - accuracy: 0.5006
16064/25000 [==================>...........] - ETA: 31s - loss: 7.6599 - accuracy: 0.5004
16096/25000 [==================>...........] - ETA: 30s - loss: 7.6600 - accuracy: 0.5004
16128/25000 [==================>...........] - ETA: 30s - loss: 7.6552 - accuracy: 0.5007
16160/25000 [==================>...........] - ETA: 30s - loss: 7.6543 - accuracy: 0.5008
16192/25000 [==================>...........] - ETA: 30s - loss: 7.6571 - accuracy: 0.5006
16224/25000 [==================>...........] - ETA: 30s - loss: 7.6553 - accuracy: 0.5007
16256/25000 [==================>...........] - ETA: 30s - loss: 7.6515 - accuracy: 0.5010
16288/25000 [==================>...........] - ETA: 30s - loss: 7.6497 - accuracy: 0.5011
16320/25000 [==================>...........] - ETA: 30s - loss: 7.6525 - accuracy: 0.5009
16352/25000 [==================>...........] - ETA: 30s - loss: 7.6488 - accuracy: 0.5012
16384/25000 [==================>...........] - ETA: 29s - loss: 7.6488 - accuracy: 0.5012
16416/25000 [==================>...........] - ETA: 29s - loss: 7.6470 - accuracy: 0.5013
16448/25000 [==================>...........] - ETA: 29s - loss: 7.6442 - accuracy: 0.5015
16480/25000 [==================>...........] - ETA: 29s - loss: 7.6462 - accuracy: 0.5013
16512/25000 [==================>...........] - ETA: 29s - loss: 7.6518 - accuracy: 0.5010
16544/25000 [==================>...........] - ETA: 29s - loss: 7.6536 - accuracy: 0.5008
16576/25000 [==================>...........] - ETA: 29s - loss: 7.6481 - accuracy: 0.5012
16608/25000 [==================>...........] - ETA: 29s - loss: 7.6500 - accuracy: 0.5011
16640/25000 [==================>...........] - ETA: 28s - loss: 7.6510 - accuracy: 0.5010
16672/25000 [===================>..........] - ETA: 28s - loss: 7.6491 - accuracy: 0.5011
16704/25000 [===================>..........] - ETA: 28s - loss: 7.6501 - accuracy: 0.5011
16736/25000 [===================>..........] - ETA: 28s - loss: 7.6492 - accuracy: 0.5011
16768/25000 [===================>..........] - ETA: 28s - loss: 7.6511 - accuracy: 0.5010
16800/25000 [===================>..........] - ETA: 28s - loss: 7.6475 - accuracy: 0.5013
16832/25000 [===================>..........] - ETA: 28s - loss: 7.6511 - accuracy: 0.5010
16864/25000 [===================>..........] - ETA: 28s - loss: 7.6475 - accuracy: 0.5012
16896/25000 [===================>..........] - ETA: 28s - loss: 7.6494 - accuracy: 0.5011
16928/25000 [===================>..........] - ETA: 27s - loss: 7.6494 - accuracy: 0.5011
16960/25000 [===================>..........] - ETA: 27s - loss: 7.6503 - accuracy: 0.5011
16992/25000 [===================>..........] - ETA: 27s - loss: 7.6477 - accuracy: 0.5012
17024/25000 [===================>..........] - ETA: 27s - loss: 7.6513 - accuracy: 0.5010
17056/25000 [===================>..........] - ETA: 27s - loss: 7.6540 - accuracy: 0.5008
17088/25000 [===================>..........] - ETA: 27s - loss: 7.6514 - accuracy: 0.5010
17120/25000 [===================>..........] - ETA: 27s - loss: 7.6505 - accuracy: 0.5011
17152/25000 [===================>..........] - ETA: 27s - loss: 7.6487 - accuracy: 0.5012
17184/25000 [===================>..........] - ETA: 27s - loss: 7.6470 - accuracy: 0.5013
17216/25000 [===================>..........] - ETA: 26s - loss: 7.6452 - accuracy: 0.5014
17248/25000 [===================>..........] - ETA: 26s - loss: 7.6497 - accuracy: 0.5011
17280/25000 [===================>..........] - ETA: 26s - loss: 7.6524 - accuracy: 0.5009
17312/25000 [===================>..........] - ETA: 26s - loss: 7.6516 - accuracy: 0.5010
17344/25000 [===================>..........] - ETA: 26s - loss: 7.6498 - accuracy: 0.5011
17376/25000 [===================>..........] - ETA: 26s - loss: 7.6534 - accuracy: 0.5009
17408/25000 [===================>..........] - ETA: 26s - loss: 7.6525 - accuracy: 0.5009
17440/25000 [===================>..........] - ETA: 26s - loss: 7.6552 - accuracy: 0.5007
17472/25000 [===================>..........] - ETA: 26s - loss: 7.6570 - accuracy: 0.5006
17504/25000 [====================>.........] - ETA: 25s - loss: 7.6526 - accuracy: 0.5009
17536/25000 [====================>.........] - ETA: 25s - loss: 7.6509 - accuracy: 0.5010
17568/25000 [====================>.........] - ETA: 25s - loss: 7.6465 - accuracy: 0.5013
17600/25000 [====================>.........] - ETA: 25s - loss: 7.6466 - accuracy: 0.5013
17632/25000 [====================>.........] - ETA: 25s - loss: 7.6484 - accuracy: 0.5012
17664/25000 [====================>.........] - ETA: 25s - loss: 7.6527 - accuracy: 0.5009
17696/25000 [====================>.........] - ETA: 25s - loss: 7.6493 - accuracy: 0.5011
17728/25000 [====================>.........] - ETA: 25s - loss: 7.6519 - accuracy: 0.5010
17760/25000 [====================>.........] - ETA: 25s - loss: 7.6519 - accuracy: 0.5010
17792/25000 [====================>.........] - ETA: 25s - loss: 7.6468 - accuracy: 0.5013
17824/25000 [====================>.........] - ETA: 24s - loss: 7.6494 - accuracy: 0.5011
17856/25000 [====================>.........] - ETA: 24s - loss: 7.6494 - accuracy: 0.5011
17888/25000 [====================>.........] - ETA: 24s - loss: 7.6512 - accuracy: 0.5010
17920/25000 [====================>.........] - ETA: 24s - loss: 7.6512 - accuracy: 0.5010
17952/25000 [====================>.........] - ETA: 24s - loss: 7.6495 - accuracy: 0.5011
17984/25000 [====================>.........] - ETA: 24s - loss: 7.6479 - accuracy: 0.5012
18016/25000 [====================>.........] - ETA: 24s - loss: 7.6462 - accuracy: 0.5013
18048/25000 [====================>.........] - ETA: 24s - loss: 7.6471 - accuracy: 0.5013
18080/25000 [====================>.........] - ETA: 23s - loss: 7.6480 - accuracy: 0.5012
18112/25000 [====================>.........] - ETA: 23s - loss: 7.6497 - accuracy: 0.5011
18144/25000 [====================>.........] - ETA: 23s - loss: 7.6497 - accuracy: 0.5011
18176/25000 [====================>.........] - ETA: 23s - loss: 7.6481 - accuracy: 0.5012
18208/25000 [====================>.........] - ETA: 23s - loss: 7.6481 - accuracy: 0.5012
18240/25000 [====================>.........] - ETA: 23s - loss: 7.6473 - accuracy: 0.5013
18272/25000 [====================>.........] - ETA: 23s - loss: 7.6465 - accuracy: 0.5013
18304/25000 [====================>.........] - ETA: 23s - loss: 7.6474 - accuracy: 0.5013
18336/25000 [=====================>........] - ETA: 23s - loss: 7.6449 - accuracy: 0.5014
18368/25000 [=====================>........] - ETA: 22s - loss: 7.6441 - accuracy: 0.5015
18400/25000 [=====================>........] - ETA: 22s - loss: 7.6433 - accuracy: 0.5015
18432/25000 [=====================>........] - ETA: 22s - loss: 7.6442 - accuracy: 0.5015
18464/25000 [=====================>........] - ETA: 22s - loss: 7.6425 - accuracy: 0.5016
18496/25000 [=====================>........] - ETA: 22s - loss: 7.6434 - accuracy: 0.5015
18528/25000 [=====================>........] - ETA: 22s - loss: 7.6451 - accuracy: 0.5014
18560/25000 [=====================>........] - ETA: 22s - loss: 7.6451 - accuracy: 0.5014
18592/25000 [=====================>........] - ETA: 22s - loss: 7.6444 - accuracy: 0.5015
18624/25000 [=====================>........] - ETA: 22s - loss: 7.6436 - accuracy: 0.5015
18656/25000 [=====================>........] - ETA: 21s - loss: 7.6453 - accuracy: 0.5014
18688/25000 [=====================>........] - ETA: 21s - loss: 7.6445 - accuracy: 0.5014
18720/25000 [=====================>........] - ETA: 21s - loss: 7.6453 - accuracy: 0.5014
18752/25000 [=====================>........] - ETA: 21s - loss: 7.6454 - accuracy: 0.5014
18784/25000 [=====================>........] - ETA: 21s - loss: 7.6462 - accuracy: 0.5013
18816/25000 [=====================>........] - ETA: 21s - loss: 7.6471 - accuracy: 0.5013
18848/25000 [=====================>........] - ETA: 21s - loss: 7.6455 - accuracy: 0.5014
18880/25000 [=====================>........] - ETA: 21s - loss: 7.6463 - accuracy: 0.5013
18912/25000 [=====================>........] - ETA: 21s - loss: 7.6464 - accuracy: 0.5013
18944/25000 [=====================>........] - ETA: 21s - loss: 7.6448 - accuracy: 0.5014
18976/25000 [=====================>........] - ETA: 20s - loss: 7.6488 - accuracy: 0.5012
19008/25000 [=====================>........] - ETA: 20s - loss: 7.6513 - accuracy: 0.5010
19040/25000 [=====================>........] - ETA: 20s - loss: 7.6529 - accuracy: 0.5009
19072/25000 [=====================>........] - ETA: 20s - loss: 7.6546 - accuracy: 0.5008
19104/25000 [=====================>........] - ETA: 20s - loss: 7.6538 - accuracy: 0.5008
19136/25000 [=====================>........] - ETA: 20s - loss: 7.6538 - accuracy: 0.5008
19168/25000 [======================>.......] - ETA: 20s - loss: 7.6562 - accuracy: 0.5007
19200/25000 [======================>.......] - ETA: 20s - loss: 7.6562 - accuracy: 0.5007
19232/25000 [======================>.......] - ETA: 20s - loss: 7.6547 - accuracy: 0.5008
19264/25000 [======================>.......] - ETA: 19s - loss: 7.6595 - accuracy: 0.5005
19296/25000 [======================>.......] - ETA: 19s - loss: 7.6571 - accuracy: 0.5006
19328/25000 [======================>.......] - ETA: 19s - loss: 7.6571 - accuracy: 0.5006
19360/25000 [======================>.......] - ETA: 19s - loss: 7.6571 - accuracy: 0.5006
19392/25000 [======================>.......] - ETA: 19s - loss: 7.6563 - accuracy: 0.5007
19424/25000 [======================>.......] - ETA: 19s - loss: 7.6548 - accuracy: 0.5008
19456/25000 [======================>.......] - ETA: 19s - loss: 7.6516 - accuracy: 0.5010
19488/25000 [======================>.......] - ETA: 19s - loss: 7.6517 - accuracy: 0.5010
19520/25000 [======================>.......] - ETA: 19s - loss: 7.6525 - accuracy: 0.5009
19552/25000 [======================>.......] - ETA: 18s - loss: 7.6541 - accuracy: 0.5008
19584/25000 [======================>.......] - ETA: 18s - loss: 7.6510 - accuracy: 0.5010
19616/25000 [======================>.......] - ETA: 18s - loss: 7.6557 - accuracy: 0.5007
19648/25000 [======================>.......] - ETA: 18s - loss: 7.6557 - accuracy: 0.5007
19680/25000 [======================>.......] - ETA: 18s - loss: 7.6565 - accuracy: 0.5007
19712/25000 [======================>.......] - ETA: 18s - loss: 7.6542 - accuracy: 0.5008
19744/25000 [======================>.......] - ETA: 18s - loss: 7.6581 - accuracy: 0.5006
19776/25000 [======================>.......] - ETA: 18s - loss: 7.6596 - accuracy: 0.5005
19808/25000 [======================>.......] - ETA: 18s - loss: 7.6604 - accuracy: 0.5004
19840/25000 [======================>.......] - ETA: 17s - loss: 7.6612 - accuracy: 0.5004
19872/25000 [======================>.......] - ETA: 17s - loss: 7.6604 - accuracy: 0.5004
19904/25000 [======================>.......] - ETA: 17s - loss: 7.6628 - accuracy: 0.5003
19936/25000 [======================>.......] - ETA: 17s - loss: 7.6620 - accuracy: 0.5003
19968/25000 [======================>.......] - ETA: 17s - loss: 7.6620 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 17s - loss: 7.6636 - accuracy: 0.5002
20032/25000 [=======================>......] - ETA: 17s - loss: 7.6651 - accuracy: 0.5001
20064/25000 [=======================>......] - ETA: 17s - loss: 7.6628 - accuracy: 0.5002
20096/25000 [=======================>......] - ETA: 17s - loss: 7.6636 - accuracy: 0.5002
20128/25000 [=======================>......] - ETA: 16s - loss: 7.6636 - accuracy: 0.5002
20160/25000 [=======================>......] - ETA: 16s - loss: 7.6651 - accuracy: 0.5001
20192/25000 [=======================>......] - ETA: 16s - loss: 7.6636 - accuracy: 0.5002
20224/25000 [=======================>......] - ETA: 16s - loss: 7.6651 - accuracy: 0.5001
20256/25000 [=======================>......] - ETA: 16s - loss: 7.6651 - accuracy: 0.5001
20288/25000 [=======================>......] - ETA: 16s - loss: 7.6636 - accuracy: 0.5002
20320/25000 [=======================>......] - ETA: 16s - loss: 7.6598 - accuracy: 0.5004
20352/25000 [=======================>......] - ETA: 16s - loss: 7.6606 - accuracy: 0.5004
20384/25000 [=======================>......] - ETA: 16s - loss: 7.6561 - accuracy: 0.5007
20416/25000 [=======================>......] - ETA: 15s - loss: 7.6554 - accuracy: 0.5007
20448/25000 [=======================>......] - ETA: 15s - loss: 7.6539 - accuracy: 0.5008
20480/25000 [=======================>......] - ETA: 15s - loss: 7.6584 - accuracy: 0.5005
20512/25000 [=======================>......] - ETA: 15s - loss: 7.6539 - accuracy: 0.5008
20544/25000 [=======================>......] - ETA: 15s - loss: 7.6539 - accuracy: 0.5008
20576/25000 [=======================>......] - ETA: 15s - loss: 7.6517 - accuracy: 0.5010
20608/25000 [=======================>......] - ETA: 15s - loss: 7.6510 - accuracy: 0.5010
20640/25000 [=======================>......] - ETA: 15s - loss: 7.6525 - accuracy: 0.5009
20672/25000 [=======================>......] - ETA: 15s - loss: 7.6488 - accuracy: 0.5012
20704/25000 [=======================>......] - ETA: 14s - loss: 7.6481 - accuracy: 0.5012
20736/25000 [=======================>......] - ETA: 14s - loss: 7.6467 - accuracy: 0.5013
20768/25000 [=======================>......] - ETA: 14s - loss: 7.6482 - accuracy: 0.5012
20800/25000 [=======================>......] - ETA: 14s - loss: 7.6438 - accuracy: 0.5015
20832/25000 [=======================>......] - ETA: 14s - loss: 7.6460 - accuracy: 0.5013
20864/25000 [========================>.....] - ETA: 14s - loss: 7.6460 - accuracy: 0.5013
20896/25000 [========================>.....] - ETA: 14s - loss: 7.6439 - accuracy: 0.5015
20928/25000 [========================>.....] - ETA: 14s - loss: 7.6483 - accuracy: 0.5012
20960/25000 [========================>.....] - ETA: 14s - loss: 7.6483 - accuracy: 0.5012
20992/25000 [========================>.....] - ETA: 13s - loss: 7.6491 - accuracy: 0.5011
21024/25000 [========================>.....] - ETA: 13s - loss: 7.6520 - accuracy: 0.5010
21056/25000 [========================>.....] - ETA: 13s - loss: 7.6535 - accuracy: 0.5009
21088/25000 [========================>.....] - ETA: 13s - loss: 7.6550 - accuracy: 0.5008
21120/25000 [========================>.....] - ETA: 13s - loss: 7.6550 - accuracy: 0.5008
21152/25000 [========================>.....] - ETA: 13s - loss: 7.6557 - accuracy: 0.5007
21184/25000 [========================>.....] - ETA: 13s - loss: 7.6558 - accuracy: 0.5007
21216/25000 [========================>.....] - ETA: 13s - loss: 7.6522 - accuracy: 0.5009
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6515 - accuracy: 0.5010
21280/25000 [========================>.....] - ETA: 12s - loss: 7.6522 - accuracy: 0.5009
21312/25000 [========================>.....] - ETA: 12s - loss: 7.6537 - accuracy: 0.5008
21344/25000 [========================>.....] - ETA: 12s - loss: 7.6523 - accuracy: 0.5009
21376/25000 [========================>.....] - ETA: 12s - loss: 7.6544 - accuracy: 0.5008
21408/25000 [========================>.....] - ETA: 12s - loss: 7.6537 - accuracy: 0.5008
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6509 - accuracy: 0.5010
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6509 - accuracy: 0.5010
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6495 - accuracy: 0.5011
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6495 - accuracy: 0.5011
21568/25000 [========================>.....] - ETA: 11s - loss: 7.6453 - accuracy: 0.5014
21600/25000 [========================>.....] - ETA: 11s - loss: 7.6453 - accuracy: 0.5014
21632/25000 [========================>.....] - ETA: 11s - loss: 7.6425 - accuracy: 0.5016
21664/25000 [========================>.....] - ETA: 11s - loss: 7.6461 - accuracy: 0.5013
21696/25000 [=========================>....] - ETA: 11s - loss: 7.6461 - accuracy: 0.5013
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6462 - accuracy: 0.5013
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6462 - accuracy: 0.5013
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6441 - accuracy: 0.5015
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6448 - accuracy: 0.5014
21856/25000 [=========================>....] - ETA: 10s - loss: 7.6435 - accuracy: 0.5015
21888/25000 [=========================>....] - ETA: 10s - loss: 7.6435 - accuracy: 0.5015
21920/25000 [=========================>....] - ETA: 10s - loss: 7.6442 - accuracy: 0.5015
21952/25000 [=========================>....] - ETA: 10s - loss: 7.6415 - accuracy: 0.5016
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6436 - accuracy: 0.5015
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6471 - accuracy: 0.5013
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6471 - accuracy: 0.5013
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6451 - accuracy: 0.5014
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6458 - accuracy: 0.5014
22144/25000 [=========================>....] - ETA: 9s - loss: 7.6417 - accuracy: 0.5016 
22176/25000 [=========================>....] - ETA: 9s - loss: 7.6417 - accuracy: 0.5016
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6418 - accuracy: 0.5016
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6432 - accuracy: 0.5015
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6467 - accuracy: 0.5013
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6467 - accuracy: 0.5013
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6453 - accuracy: 0.5014
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6447 - accuracy: 0.5014
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6440 - accuracy: 0.5015
22432/25000 [=========================>....] - ETA: 8s - loss: 7.6461 - accuracy: 0.5013
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6448 - accuracy: 0.5014
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6462 - accuracy: 0.5013
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6462 - accuracy: 0.5013
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6462 - accuracy: 0.5013
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6442 - accuracy: 0.5015
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6443 - accuracy: 0.5015
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6429 - accuracy: 0.5015
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6436 - accuracy: 0.5015
22720/25000 [==========================>...] - ETA: 7s - loss: 7.6416 - accuracy: 0.5016
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6417 - accuracy: 0.5016
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6397 - accuracy: 0.5018
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6418 - accuracy: 0.5016
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6398 - accuracy: 0.5018
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6438 - accuracy: 0.5015
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6425 - accuracy: 0.5016
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6432 - accuracy: 0.5015
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6459 - accuracy: 0.5013
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6486 - accuracy: 0.5012
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6493 - accuracy: 0.5011
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6500 - accuracy: 0.5011
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6480 - accuracy: 0.5012
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6494 - accuracy: 0.5011
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6501 - accuracy: 0.5011
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6514 - accuracy: 0.5010
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6521 - accuracy: 0.5009
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6495 - accuracy: 0.5011
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6488 - accuracy: 0.5012
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6476 - accuracy: 0.5012
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6443 - accuracy: 0.5015
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6411 - accuracy: 0.5017
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6424 - accuracy: 0.5016
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6398 - accuracy: 0.5017
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6418 - accuracy: 0.5016
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6425 - accuracy: 0.5016
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6412 - accuracy: 0.5017
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6413 - accuracy: 0.5017
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6406 - accuracy: 0.5017
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6361 - accuracy: 0.5020
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6368 - accuracy: 0.5019
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6369 - accuracy: 0.5019
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6376 - accuracy: 0.5019
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6370 - accuracy: 0.5019
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6402 - accuracy: 0.5017
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6402 - accuracy: 0.5017
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6384 - accuracy: 0.5018
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6390 - accuracy: 0.5018
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6404 - accuracy: 0.5017
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6423 - accuracy: 0.5016
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6430 - accuracy: 0.5015
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6456 - accuracy: 0.5014
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6475 - accuracy: 0.5012
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6494 - accuracy: 0.5011
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6507 - accuracy: 0.5010
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6527 - accuracy: 0.5009
24192/25000 [============================>.] - ETA: 2s - loss: 7.6539 - accuracy: 0.5008
24224/25000 [============================>.] - ETA: 2s - loss: 7.6578 - accuracy: 0.5006
24256/25000 [============================>.] - ETA: 2s - loss: 7.6571 - accuracy: 0.5006
24288/25000 [============================>.] - ETA: 2s - loss: 7.6540 - accuracy: 0.5008
24320/25000 [============================>.] - ETA: 2s - loss: 7.6597 - accuracy: 0.5005
24352/25000 [============================>.] - ETA: 2s - loss: 7.6591 - accuracy: 0.5005
24384/25000 [============================>.] - ETA: 2s - loss: 7.6603 - accuracy: 0.5004
24416/25000 [============================>.] - ETA: 2s - loss: 7.6585 - accuracy: 0.5005
24448/25000 [============================>.] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
24480/25000 [============================>.] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
24512/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24544/25000 [============================>.] - ETA: 1s - loss: 7.6629 - accuracy: 0.5002
24576/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24608/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24640/25000 [============================>.] - ETA: 1s - loss: 7.6629 - accuracy: 0.5002
24672/25000 [============================>.] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
24704/25000 [============================>.] - ETA: 1s - loss: 7.6579 - accuracy: 0.5006
24736/25000 [============================>.] - ETA: 0s - loss: 7.6561 - accuracy: 0.5007
24768/25000 [============================>.] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24800/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24832/25000 [============================>.] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
24864/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24896/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24928/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24960/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 104s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f3d9ac919e8> 

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
 [-0.05529152 -0.00278106  0.00868782 -0.04748154  0.09322448  0.00656089]
 [-0.03190015  0.06958697 -0.0612773   0.07044359  0.03899764 -0.00147164]
 [ 0.01750891  0.04860398 -0.04736517 -0.00505755 -0.15415274  0.08610851]
 [ 0.0274651   0.02919315 -0.06429131  0.01673975  0.08018266 -0.1796083 ]
 [-0.58491689 -0.32552323  0.12330666 -0.15749013  0.08739173 -0.1403258 ]
 [ 0.09899717  0.05405192  0.10785206 -0.19337337  0.06765942  0.14415532]
 [-0.07578518  0.63706392  0.1485918   0.06590094  0.7007643   0.24473272]
 [ 0.06780232 -0.33575255  0.35339475 -0.42080921  0.19223973  0.14227535]
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
{'loss': 0.5265209600329399, 'loss_history': []}

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
{'loss': 0.5176071748137474, 'loss_history': []}

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
	Data preprocessing and feature engineering runtime = 0.22s ...
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
 40%|████      | 2/5 [00:46<01:10, 23.46s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.28794432494795125, 'embedding_size_factor': 1.2786826185023015, 'layers.choice': 0, 'learning_rate': 0.0018424331594209206, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 0.002064889855086772} and reward: 0.3772
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd2m\xae\x08\xae\x18\x89X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4u{\xe7\xc6\xe5\x7fX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?^/\xb9\x8a\x8b\xbcjX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?`\xeacL\xb7\xd3\xcfu.' and reward: 0.3772
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd2m\xae\x08\xae\x18\x89X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4u{\xe7\xc6\xe5\x7fX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?^/\xb9\x8a\x8b\xbcjX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?`\xeacL\xb7\xd3\xcfu.' and reward: 0.3772
 60%|██████    | 3/5 [01:33<01:01, 30.54s/it] 60%|██████    | 3/5 [01:33<01:02, 31.33s/it]
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.3004249916234195, 'embedding_size_factor': 0.5031530722760926, 'layers.choice': 0, 'learning_rate': 0.002363590152670403, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 4.472140596684212e-09} and reward: 0.376
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd3:)\xbe{\x1dBX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe0\x19\xd4x\xc9\xd9\x06X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?c\\\xce\xcd\x05\x0e\x00X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>35+\xab\x98\xae.u.' and reward: 0.376
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd3:)\xbe{\x1dBX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe0\x19\xd4x\xc9\xd9\x06X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?c\\\xce\xcd\x05\x0e\x00X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>35+\xab\x98\xae.u.' and reward: 0.376
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 142.3050947189331
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.78s of the -24.69s of remaining time.
Ensemble size: 36
Ensemble weights: 
[0.63888889 0.25       0.11111111]
	0.39	 = Validation accuracy score
	0.95s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 145.67s ...
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

