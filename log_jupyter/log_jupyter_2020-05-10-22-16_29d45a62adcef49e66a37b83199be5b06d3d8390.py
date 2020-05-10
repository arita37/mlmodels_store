
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/29d45a62adcef49e66a37b83199be5b06d3d8390', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '29d45a62adcef49e66a37b83199be5b06d3d8390', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/29d45a62adcef49e66a37b83199be5b06d3d8390

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/29d45a62adcef49e66a37b83199be5b06d3d8390

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
   24576/17464789 [..............................] - ETA: 45s
   57344/17464789 [..............................] - ETA: 39s
   90112/17464789 [..............................] - ETA: 37s
  163840/17464789 [..............................] - ETA: 27s
  319488/17464789 [..............................] - ETA: 17s
  630784/17464789 [>.............................] - ETA: 10s
 1277952/17464789 [=>............................] - ETA: 5s 
 2523136/17464789 [===>..........................] - ETA: 3s
 5029888/17464789 [=======>......................] - ETA: 1s
 7700480/17464789 [============>.................] - ETA: 0s
10567680/17464789 [=================>............] - ETA: 0s
13369344/17464789 [=====================>........] - ETA: 0s
16154624/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 22:16:26.817486: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 22:16:26.821055: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-10 22:16:26.821183: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559ad9033700 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 22:16:26.821199: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:12 - loss: 8.1458 - accuracy: 0.4688
   64/25000 [..............................] - ETA: 2:43 - loss: 8.1458 - accuracy: 0.4688
   96/25000 [..............................] - ETA: 2:12 - loss: 7.0277 - accuracy: 0.5417
  128/25000 [..............................] - ETA: 1:57 - loss: 6.5885 - accuracy: 0.5703
  160/25000 [..............................] - ETA: 1:47 - loss: 6.9000 - accuracy: 0.5500
  192/25000 [..............................] - ETA: 1:41 - loss: 6.9479 - accuracy: 0.5469
  224/25000 [..............................] - ETA: 1:37 - loss: 7.1190 - accuracy: 0.5357
  256/25000 [..............................] - ETA: 1:33 - loss: 7.1875 - accuracy: 0.5312
  288/25000 [..............................] - ETA: 1:31 - loss: 7.2939 - accuracy: 0.5243
  320/25000 [..............................] - ETA: 1:30 - loss: 7.4750 - accuracy: 0.5125
  352/25000 [..............................] - ETA: 1:28 - loss: 7.5359 - accuracy: 0.5085
  384/25000 [..............................] - ETA: 1:26 - loss: 7.3871 - accuracy: 0.5182
  416/25000 [..............................] - ETA: 1:25 - loss: 7.4455 - accuracy: 0.5144
  448/25000 [..............................] - ETA: 1:24 - loss: 7.4955 - accuracy: 0.5112
  480/25000 [..............................] - ETA: 1:23 - loss: 7.5708 - accuracy: 0.5063
  512/25000 [..............................] - ETA: 1:22 - loss: 7.5169 - accuracy: 0.5098
  544/25000 [..............................] - ETA: 1:21 - loss: 7.5821 - accuracy: 0.5055
  576/25000 [..............................] - ETA: 1:20 - loss: 7.6666 - accuracy: 0.5000
  608/25000 [..............................] - ETA: 1:20 - loss: 7.5153 - accuracy: 0.5099
  640/25000 [..............................] - ETA: 1:19 - loss: 7.4989 - accuracy: 0.5109
  672/25000 [..............................] - ETA: 1:19 - loss: 7.4841 - accuracy: 0.5119
  704/25000 [..............................] - ETA: 1:18 - loss: 7.4706 - accuracy: 0.5128
  736/25000 [..............................] - ETA: 1:18 - loss: 7.4583 - accuracy: 0.5136
  768/25000 [..............................] - ETA: 1:18 - loss: 7.3871 - accuracy: 0.5182
  800/25000 [..............................] - ETA: 1:17 - loss: 7.4750 - accuracy: 0.5125
  832/25000 [..............................] - ETA: 1:17 - loss: 7.5008 - accuracy: 0.5108
  864/25000 [>.............................] - ETA: 1:17 - loss: 7.5069 - accuracy: 0.5104
  896/25000 [>.............................] - ETA: 1:16 - loss: 7.5297 - accuracy: 0.5089
  928/25000 [>.............................] - ETA: 1:16 - loss: 7.5014 - accuracy: 0.5108
  960/25000 [>.............................] - ETA: 1:16 - loss: 7.5069 - accuracy: 0.5104
  992/25000 [>.............................] - ETA: 1:15 - loss: 7.5275 - accuracy: 0.5091
 1024/25000 [>.............................] - ETA: 1:15 - loss: 7.5468 - accuracy: 0.5078
 1056/25000 [>.............................] - ETA: 1:15 - loss: 7.5505 - accuracy: 0.5076
 1088/25000 [>.............................] - ETA: 1:14 - loss: 7.4975 - accuracy: 0.5110
 1120/25000 [>.............................] - ETA: 1:14 - loss: 7.5160 - accuracy: 0.5098
 1152/25000 [>.............................] - ETA: 1:14 - loss: 7.5468 - accuracy: 0.5078
 1184/25000 [>.............................] - ETA: 1:14 - loss: 7.5889 - accuracy: 0.5051
 1216/25000 [>.............................] - ETA: 1:13 - loss: 7.5405 - accuracy: 0.5082
 1248/25000 [>.............................] - ETA: 1:13 - loss: 7.5315 - accuracy: 0.5088
 1280/25000 [>.............................] - ETA: 1:13 - loss: 7.5229 - accuracy: 0.5094
 1312/25000 [>.............................] - ETA: 1:13 - loss: 7.5381 - accuracy: 0.5084
 1344/25000 [>.............................] - ETA: 1:13 - loss: 7.4727 - accuracy: 0.5126
 1376/25000 [>.............................] - ETA: 1:13 - loss: 7.4883 - accuracy: 0.5116
 1408/25000 [>.............................] - ETA: 1:13 - loss: 7.4706 - accuracy: 0.5128
 1440/25000 [>.............................] - ETA: 1:12 - loss: 7.4856 - accuracy: 0.5118
 1472/25000 [>.............................] - ETA: 1:12 - loss: 7.4895 - accuracy: 0.5115
 1504/25000 [>.............................] - ETA: 1:12 - loss: 7.4831 - accuracy: 0.5120
 1536/25000 [>.............................] - ETA: 1:12 - loss: 7.5169 - accuracy: 0.5098
 1568/25000 [>.............................] - ETA: 1:12 - loss: 7.5688 - accuracy: 0.5064
 1600/25000 [>.............................] - ETA: 1:11 - loss: 7.5516 - accuracy: 0.5075
 1632/25000 [>.............................] - ETA: 1:11 - loss: 7.5539 - accuracy: 0.5074
 1664/25000 [>.............................] - ETA: 1:11 - loss: 7.5929 - accuracy: 0.5048
 1696/25000 [=>............................] - ETA: 1:11 - loss: 7.5853 - accuracy: 0.5053
 1728/25000 [=>............................] - ETA: 1:11 - loss: 7.6577 - accuracy: 0.5006
 1760/25000 [=>............................] - ETA: 1:10 - loss: 7.6928 - accuracy: 0.4983
 1792/25000 [=>............................] - ETA: 1:10 - loss: 7.7008 - accuracy: 0.4978
 1824/25000 [=>............................] - ETA: 1:10 - loss: 7.6918 - accuracy: 0.4984
 1856/25000 [=>............................] - ETA: 1:10 - loss: 7.6997 - accuracy: 0.4978
 1888/25000 [=>............................] - ETA: 1:10 - loss: 7.7072 - accuracy: 0.4974
 1920/25000 [=>............................] - ETA: 1:09 - loss: 7.7225 - accuracy: 0.4964
 1952/25000 [=>............................] - ETA: 1:09 - loss: 7.6980 - accuracy: 0.4980
 1984/25000 [=>............................] - ETA: 1:09 - loss: 7.6512 - accuracy: 0.5010
 2016/25000 [=>............................] - ETA: 1:09 - loss: 7.6210 - accuracy: 0.5030
 2048/25000 [=>............................] - ETA: 1:09 - loss: 7.6292 - accuracy: 0.5024
 2080/25000 [=>............................] - ETA: 1:09 - loss: 7.6666 - accuracy: 0.5000
 2112/25000 [=>............................] - ETA: 1:09 - loss: 7.6739 - accuracy: 0.4995
 2144/25000 [=>............................] - ETA: 1:08 - loss: 7.7095 - accuracy: 0.4972
 2176/25000 [=>............................] - ETA: 1:08 - loss: 7.7089 - accuracy: 0.4972
 2208/25000 [=>............................] - ETA: 1:08 - loss: 7.6944 - accuracy: 0.4982
 2240/25000 [=>............................] - ETA: 1:08 - loss: 7.7077 - accuracy: 0.4973
 2272/25000 [=>............................] - ETA: 1:08 - loss: 7.7206 - accuracy: 0.4965
 2304/25000 [=>............................] - ETA: 1:08 - loss: 7.7065 - accuracy: 0.4974
 2336/25000 [=>............................] - ETA: 1:07 - loss: 7.7323 - accuracy: 0.4957
 2368/25000 [=>............................] - ETA: 1:07 - loss: 7.7119 - accuracy: 0.4970
 2400/25000 [=>............................] - ETA: 1:07 - loss: 7.7177 - accuracy: 0.4967
 2432/25000 [=>............................] - ETA: 1:07 - loss: 7.6855 - accuracy: 0.4988
 2464/25000 [=>............................] - ETA: 1:07 - loss: 7.7102 - accuracy: 0.4972
 2496/25000 [=>............................] - ETA: 1:07 - loss: 7.7035 - accuracy: 0.4976
 2528/25000 [==>...........................] - ETA: 1:07 - loss: 7.7151 - accuracy: 0.4968
 2560/25000 [==>...........................] - ETA: 1:07 - loss: 7.7085 - accuracy: 0.4973
 2592/25000 [==>...........................] - ETA: 1:07 - loss: 7.7139 - accuracy: 0.4969
 2624/25000 [==>...........................] - ETA: 1:07 - loss: 7.7426 - accuracy: 0.4950
 2656/25000 [==>...........................] - ETA: 1:07 - loss: 7.7417 - accuracy: 0.4951
 2688/25000 [==>...........................] - ETA: 1:06 - loss: 7.7522 - accuracy: 0.4944
 2720/25000 [==>...........................] - ETA: 1:06 - loss: 7.7512 - accuracy: 0.4945
 2752/25000 [==>...........................] - ETA: 1:06 - loss: 7.7223 - accuracy: 0.4964
 2784/25000 [==>...........................] - ETA: 1:06 - loss: 7.7382 - accuracy: 0.4953
 2816/25000 [==>...........................] - ETA: 1:06 - loss: 7.7483 - accuracy: 0.4947
 2848/25000 [==>...........................] - ETA: 1:06 - loss: 7.7474 - accuracy: 0.4947
 2880/25000 [==>...........................] - ETA: 1:06 - loss: 7.7412 - accuracy: 0.4951
 2912/25000 [==>...........................] - ETA: 1:06 - loss: 7.7614 - accuracy: 0.4938
 2944/25000 [==>...........................] - ETA: 1:05 - loss: 7.7604 - accuracy: 0.4939
 2976/25000 [==>...........................] - ETA: 1:05 - loss: 7.7491 - accuracy: 0.4946
 3008/25000 [==>...........................] - ETA: 1:05 - loss: 7.7431 - accuracy: 0.4950
 3040/25000 [==>...........................] - ETA: 1:05 - loss: 7.7271 - accuracy: 0.4961
 3072/25000 [==>...........................] - ETA: 1:05 - loss: 7.7515 - accuracy: 0.4945
 3104/25000 [==>...........................] - ETA: 1:05 - loss: 7.7308 - accuracy: 0.4958
 3136/25000 [==>...........................] - ETA: 1:05 - loss: 7.7497 - accuracy: 0.4946
 3168/25000 [==>...........................] - ETA: 1:05 - loss: 7.7441 - accuracy: 0.4949
 3200/25000 [==>...........................] - ETA: 1:05 - loss: 7.7672 - accuracy: 0.4934
 3232/25000 [==>...........................] - ETA: 1:04 - loss: 7.7757 - accuracy: 0.4929
 3264/25000 [==>...........................] - ETA: 1:04 - loss: 7.7747 - accuracy: 0.4930
 3296/25000 [==>...........................] - ETA: 1:04 - loss: 7.7969 - accuracy: 0.4915
 3328/25000 [==>...........................] - ETA: 1:04 - loss: 7.7818 - accuracy: 0.4925
 3360/25000 [===>..........................] - ETA: 1:04 - loss: 7.7533 - accuracy: 0.4943
 3392/25000 [===>..........................] - ETA: 1:04 - loss: 7.7435 - accuracy: 0.4950
 3424/25000 [===>..........................] - ETA: 1:04 - loss: 7.7562 - accuracy: 0.4942
 3456/25000 [===>..........................] - ETA: 1:04 - loss: 7.7332 - accuracy: 0.4957
 3488/25000 [===>..........................] - ETA: 1:04 - loss: 7.7326 - accuracy: 0.4957
 3520/25000 [===>..........................] - ETA: 1:04 - loss: 7.7407 - accuracy: 0.4952
 3552/25000 [===>..........................] - ETA: 1:03 - loss: 7.7573 - accuracy: 0.4941
 3584/25000 [===>..........................] - ETA: 1:03 - loss: 7.7779 - accuracy: 0.4927
 3616/25000 [===>..........................] - ETA: 1:03 - loss: 7.7938 - accuracy: 0.4917
 3648/25000 [===>..........................] - ETA: 1:03 - loss: 7.7969 - accuracy: 0.4915
 3680/25000 [===>..........................] - ETA: 1:03 - loss: 7.8041 - accuracy: 0.4910
 3712/25000 [===>..........................] - ETA: 1:03 - loss: 7.7905 - accuracy: 0.4919
 3744/25000 [===>..........................] - ETA: 1:03 - loss: 7.7936 - accuracy: 0.4917
 3776/25000 [===>..........................] - ETA: 1:03 - loss: 7.8047 - accuracy: 0.4910
 3808/25000 [===>..........................] - ETA: 1:03 - loss: 7.7995 - accuracy: 0.4913
 3840/25000 [===>..........................] - ETA: 1:03 - loss: 7.8064 - accuracy: 0.4909
 3872/25000 [===>..........................] - ETA: 1:02 - loss: 7.7933 - accuracy: 0.4917
 3904/25000 [===>..........................] - ETA: 1:02 - loss: 7.7844 - accuracy: 0.4923
 3936/25000 [===>..........................] - ETA: 1:02 - loss: 7.8147 - accuracy: 0.4903
 3968/25000 [===>..........................] - ETA: 1:02 - loss: 7.8251 - accuracy: 0.4897
 4000/25000 [===>..........................] - ETA: 1:02 - loss: 7.8276 - accuracy: 0.4895
 4032/25000 [===>..........................] - ETA: 1:02 - loss: 7.8073 - accuracy: 0.4908
 4064/25000 [===>..........................] - ETA: 1:02 - loss: 7.8024 - accuracy: 0.4911
 4096/25000 [===>..........................] - ETA: 1:02 - loss: 7.8051 - accuracy: 0.4910
 4128/25000 [===>..........................] - ETA: 1:02 - loss: 7.7892 - accuracy: 0.4920
 4160/25000 [===>..........................] - ETA: 1:02 - loss: 7.7772 - accuracy: 0.4928
 4192/25000 [====>.........................] - ETA: 1:01 - loss: 7.7617 - accuracy: 0.4938
 4224/25000 [====>.........................] - ETA: 1:01 - loss: 7.7574 - accuracy: 0.4941
 4256/25000 [====>.........................] - ETA: 1:01 - loss: 7.7495 - accuracy: 0.4946
 4288/25000 [====>.........................] - ETA: 1:01 - loss: 7.7417 - accuracy: 0.4951
 4320/25000 [====>.........................] - ETA: 1:01 - loss: 7.7376 - accuracy: 0.4954
 4352/25000 [====>.........................] - ETA: 1:01 - loss: 7.7300 - accuracy: 0.4959
 4384/25000 [====>.........................] - ETA: 1:01 - loss: 7.7226 - accuracy: 0.4964
 4416/25000 [====>.........................] - ETA: 1:01 - loss: 7.7187 - accuracy: 0.4966
 4448/25000 [====>.........................] - ETA: 1:01 - loss: 7.7080 - accuracy: 0.4973
 4480/25000 [====>.........................] - ETA: 1:00 - loss: 7.7008 - accuracy: 0.4978
 4512/25000 [====>.........................] - ETA: 1:00 - loss: 7.6870 - accuracy: 0.4987
 4544/25000 [====>.........................] - ETA: 1:00 - loss: 7.6801 - accuracy: 0.4991
 4576/25000 [====>.........................] - ETA: 1:00 - loss: 7.6666 - accuracy: 0.5000
 4608/25000 [====>.........................] - ETA: 1:00 - loss: 7.6600 - accuracy: 0.5004
 4640/25000 [====>.........................] - ETA: 1:00 - loss: 7.6501 - accuracy: 0.5011
 4672/25000 [====>.........................] - ETA: 1:00 - loss: 7.6535 - accuracy: 0.5009
 4704/25000 [====>.........................] - ETA: 1:00 - loss: 7.6438 - accuracy: 0.5015
 4736/25000 [====>.........................] - ETA: 1:00 - loss: 7.6472 - accuracy: 0.5013
 4768/25000 [====>.........................] - ETA: 59s - loss: 7.6441 - accuracy: 0.5015 
 4800/25000 [====>.........................] - ETA: 59s - loss: 7.6506 - accuracy: 0.5010
 4832/25000 [====>.........................] - ETA: 59s - loss: 7.6476 - accuracy: 0.5012
 4864/25000 [====>.........................] - ETA: 59s - loss: 7.6509 - accuracy: 0.5010
 4896/25000 [====>.........................] - ETA: 59s - loss: 7.6666 - accuracy: 0.5000
 4928/25000 [====>.........................] - ETA: 59s - loss: 7.6697 - accuracy: 0.4998
 4960/25000 [====>.........................] - ETA: 59s - loss: 7.6604 - accuracy: 0.5004
 4992/25000 [====>.........................] - ETA: 59s - loss: 7.6697 - accuracy: 0.4998
 5024/25000 [=====>........................] - ETA: 59s - loss: 7.6636 - accuracy: 0.5002
 5056/25000 [=====>........................] - ETA: 59s - loss: 7.6727 - accuracy: 0.4996
 5088/25000 [=====>........................] - ETA: 58s - loss: 7.6787 - accuracy: 0.4992
 5120/25000 [=====>........................] - ETA: 58s - loss: 7.6756 - accuracy: 0.4994
 5152/25000 [=====>........................] - ETA: 58s - loss: 7.6785 - accuracy: 0.4992
 5184/25000 [=====>........................] - ETA: 58s - loss: 7.6785 - accuracy: 0.4992
 5216/25000 [=====>........................] - ETA: 58s - loss: 7.6725 - accuracy: 0.4996
 5248/25000 [=====>........................] - ETA: 58s - loss: 7.6725 - accuracy: 0.4996
 5280/25000 [=====>........................] - ETA: 58s - loss: 7.6724 - accuracy: 0.4996
 5312/25000 [=====>........................] - ETA: 58s - loss: 7.6782 - accuracy: 0.4992
 5344/25000 [=====>........................] - ETA: 58s - loss: 7.6924 - accuracy: 0.4983
 5376/25000 [=====>........................] - ETA: 57s - loss: 7.7008 - accuracy: 0.4978
 5408/25000 [=====>........................] - ETA: 57s - loss: 7.7006 - accuracy: 0.4978
 5440/25000 [=====>........................] - ETA: 57s - loss: 7.6948 - accuracy: 0.4982
 5472/25000 [=====>........................] - ETA: 57s - loss: 7.6918 - accuracy: 0.4984
 5504/25000 [=====>........................] - ETA: 57s - loss: 7.7028 - accuracy: 0.4976
 5536/25000 [=====>........................] - ETA: 57s - loss: 7.7109 - accuracy: 0.4971
 5568/25000 [=====>........................] - ETA: 57s - loss: 7.7217 - accuracy: 0.4964
 5600/25000 [=====>........................] - ETA: 57s - loss: 7.7241 - accuracy: 0.4963
 5632/25000 [=====>........................] - ETA: 57s - loss: 7.7156 - accuracy: 0.4968
 5664/25000 [=====>........................] - ETA: 56s - loss: 7.7126 - accuracy: 0.4970
 5696/25000 [=====>........................] - ETA: 56s - loss: 7.7043 - accuracy: 0.4975
 5728/25000 [=====>........................] - ETA: 56s - loss: 7.7121 - accuracy: 0.4970
 5760/25000 [=====>........................] - ETA: 56s - loss: 7.7092 - accuracy: 0.4972
 5792/25000 [=====>........................] - ETA: 56s - loss: 7.7037 - accuracy: 0.4976
 5824/25000 [=====>........................] - ETA: 56s - loss: 7.7087 - accuracy: 0.4973
 5856/25000 [======>.......................] - ETA: 56s - loss: 7.7111 - accuracy: 0.4971
 5888/25000 [======>.......................] - ETA: 56s - loss: 7.7109 - accuracy: 0.4971
 5920/25000 [======>.......................] - ETA: 56s - loss: 7.7210 - accuracy: 0.4965
 5952/25000 [======>.......................] - ETA: 56s - loss: 7.7181 - accuracy: 0.4966
 5984/25000 [======>.......................] - ETA: 55s - loss: 7.7076 - accuracy: 0.4973
 6016/25000 [======>.......................] - ETA: 55s - loss: 7.7099 - accuracy: 0.4972
 6048/25000 [======>.......................] - ETA: 55s - loss: 7.7148 - accuracy: 0.4969
 6080/25000 [======>.......................] - ETA: 55s - loss: 7.7196 - accuracy: 0.4965
 6112/25000 [======>.......................] - ETA: 55s - loss: 7.7218 - accuracy: 0.4964
 6144/25000 [======>.......................] - ETA: 55s - loss: 7.7115 - accuracy: 0.4971
 6176/25000 [======>.......................] - ETA: 55s - loss: 7.7088 - accuracy: 0.4972
 6208/25000 [======>.......................] - ETA: 55s - loss: 7.7160 - accuracy: 0.4968
 6240/25000 [======>.......................] - ETA: 55s - loss: 7.7010 - accuracy: 0.4978
 6272/25000 [======>.......................] - ETA: 55s - loss: 7.7106 - accuracy: 0.4971
 6304/25000 [======>.......................] - ETA: 54s - loss: 7.7055 - accuracy: 0.4975
 6336/25000 [======>.......................] - ETA: 54s - loss: 7.7053 - accuracy: 0.4975
 6368/25000 [======>.......................] - ETA: 54s - loss: 7.7003 - accuracy: 0.4978
 6400/25000 [======>.......................] - ETA: 54s - loss: 7.6978 - accuracy: 0.4980
 6432/25000 [======>.......................] - ETA: 54s - loss: 7.6881 - accuracy: 0.4986
 6464/25000 [======>.......................] - ETA: 54s - loss: 7.7022 - accuracy: 0.4977
 6496/25000 [======>.......................] - ETA: 54s - loss: 7.6973 - accuracy: 0.4980
 6528/25000 [======>.......................] - ETA: 54s - loss: 7.7019 - accuracy: 0.4977
 6560/25000 [======>.......................] - ETA: 54s - loss: 7.6947 - accuracy: 0.4982
 6592/25000 [======>.......................] - ETA: 54s - loss: 7.6969 - accuracy: 0.4980
 6624/25000 [======>.......................] - ETA: 54s - loss: 7.6967 - accuracy: 0.4980
 6656/25000 [======>.......................] - ETA: 53s - loss: 7.6989 - accuracy: 0.4979
 6688/25000 [=======>......................] - ETA: 53s - loss: 7.6918 - accuracy: 0.4984
 6720/25000 [=======>......................] - ETA: 53s - loss: 7.6940 - accuracy: 0.4982
 6752/25000 [=======>......................] - ETA: 53s - loss: 7.6893 - accuracy: 0.4985
 6784/25000 [=======>......................] - ETA: 53s - loss: 7.6847 - accuracy: 0.4988
 6816/25000 [=======>......................] - ETA: 53s - loss: 7.6801 - accuracy: 0.4991
 6848/25000 [=======>......................] - ETA: 53s - loss: 7.6890 - accuracy: 0.4985
 6880/25000 [=======>......................] - ETA: 53s - loss: 7.6889 - accuracy: 0.4985
 6912/25000 [=======>......................] - ETA: 53s - loss: 7.6888 - accuracy: 0.4986
 6944/25000 [=======>......................] - ETA: 53s - loss: 7.6887 - accuracy: 0.4986
 6976/25000 [=======>......................] - ETA: 52s - loss: 7.6886 - accuracy: 0.4986
 7008/25000 [=======>......................] - ETA: 52s - loss: 7.6797 - accuracy: 0.4991
 7040/25000 [=======>......................] - ETA: 52s - loss: 7.6775 - accuracy: 0.4993
 7072/25000 [=======>......................] - ETA: 52s - loss: 7.6775 - accuracy: 0.4993
 7104/25000 [=======>......................] - ETA: 52s - loss: 7.6645 - accuracy: 0.5001
 7136/25000 [=======>......................] - ETA: 52s - loss: 7.6516 - accuracy: 0.5010
 7168/25000 [=======>......................] - ETA: 52s - loss: 7.6431 - accuracy: 0.5015
 7200/25000 [=======>......................] - ETA: 52s - loss: 7.6475 - accuracy: 0.5013
 7232/25000 [=======>......................] - ETA: 52s - loss: 7.6581 - accuracy: 0.5006
 7264/25000 [=======>......................] - ETA: 52s - loss: 7.6455 - accuracy: 0.5014
 7296/25000 [=======>......................] - ETA: 51s - loss: 7.6498 - accuracy: 0.5011
 7328/25000 [=======>......................] - ETA: 51s - loss: 7.6415 - accuracy: 0.5016
 7360/25000 [=======>......................] - ETA: 51s - loss: 7.6416 - accuracy: 0.5016
 7392/25000 [=======>......................] - ETA: 51s - loss: 7.6334 - accuracy: 0.5022
 7424/25000 [=======>......................] - ETA: 51s - loss: 7.6377 - accuracy: 0.5019
 7456/25000 [=======>......................] - ETA: 51s - loss: 7.6358 - accuracy: 0.5020
 7488/25000 [=======>......................] - ETA: 51s - loss: 7.6339 - accuracy: 0.5021
 7520/25000 [========>.....................] - ETA: 51s - loss: 7.6320 - accuracy: 0.5023
 7552/25000 [========>.....................] - ETA: 51s - loss: 7.6240 - accuracy: 0.5028
 7584/25000 [========>.....................] - ETA: 50s - loss: 7.6100 - accuracy: 0.5037
 7616/25000 [========>.....................] - ETA: 50s - loss: 7.6102 - accuracy: 0.5037
 7648/25000 [========>.....................] - ETA: 50s - loss: 7.6165 - accuracy: 0.5033
 7680/25000 [========>.....................] - ETA: 50s - loss: 7.6187 - accuracy: 0.5031
 7712/25000 [========>.....................] - ETA: 50s - loss: 7.6209 - accuracy: 0.5030
 7744/25000 [========>.....................] - ETA: 50s - loss: 7.6191 - accuracy: 0.5031
 7776/25000 [========>.....................] - ETA: 50s - loss: 7.6154 - accuracy: 0.5033
 7808/25000 [========>.....................] - ETA: 50s - loss: 7.6234 - accuracy: 0.5028
 7840/25000 [========>.....................] - ETA: 50s - loss: 7.6197 - accuracy: 0.5031
 7872/25000 [========>.....................] - ETA: 50s - loss: 7.6121 - accuracy: 0.5036
 7904/25000 [========>.....................] - ETA: 50s - loss: 7.6104 - accuracy: 0.5037
 7936/25000 [========>.....................] - ETA: 50s - loss: 7.6048 - accuracy: 0.5040
 7968/25000 [========>.....................] - ETA: 49s - loss: 7.6050 - accuracy: 0.5040
 8000/25000 [========>.....................] - ETA: 49s - loss: 7.6149 - accuracy: 0.5034
 8032/25000 [========>.....................] - ETA: 49s - loss: 7.6093 - accuracy: 0.5037
 8064/25000 [========>.....................] - ETA: 49s - loss: 7.5982 - accuracy: 0.5045
 8096/25000 [========>.....................] - ETA: 49s - loss: 7.5984 - accuracy: 0.5044
 8128/25000 [========>.....................] - ETA: 49s - loss: 7.5987 - accuracy: 0.5044
 8160/25000 [========>.....................] - ETA: 49s - loss: 7.5896 - accuracy: 0.5050
 8192/25000 [========>.....................] - ETA: 49s - loss: 7.5955 - accuracy: 0.5046
 8224/25000 [========>.....................] - ETA: 49s - loss: 7.6032 - accuracy: 0.5041
 8256/25000 [========>.....................] - ETA: 49s - loss: 7.6109 - accuracy: 0.5036
 8288/25000 [========>.....................] - ETA: 48s - loss: 7.6185 - accuracy: 0.5031
 8320/25000 [========>.....................] - ETA: 48s - loss: 7.6169 - accuracy: 0.5032
 8352/25000 [=========>....................] - ETA: 48s - loss: 7.6207 - accuracy: 0.5030
 8384/25000 [=========>....................] - ETA: 48s - loss: 7.6191 - accuracy: 0.5031
 8416/25000 [=========>....................] - ETA: 48s - loss: 7.6265 - accuracy: 0.5026
 8448/25000 [=========>....................] - ETA: 48s - loss: 7.6176 - accuracy: 0.5032
 8480/25000 [=========>....................] - ETA: 48s - loss: 7.6178 - accuracy: 0.5032
 8512/25000 [=========>....................] - ETA: 48s - loss: 7.6198 - accuracy: 0.5031
 8544/25000 [=========>....................] - ETA: 48s - loss: 7.6289 - accuracy: 0.5025
 8576/25000 [=========>....................] - ETA: 48s - loss: 7.6362 - accuracy: 0.5020
 8608/25000 [=========>....................] - ETA: 48s - loss: 7.6363 - accuracy: 0.5020
 8640/25000 [=========>....................] - ETA: 47s - loss: 7.6364 - accuracy: 0.5020
 8672/25000 [=========>....................] - ETA: 47s - loss: 7.6419 - accuracy: 0.5016
 8704/25000 [=========>....................] - ETA: 47s - loss: 7.6420 - accuracy: 0.5016
 8736/25000 [=========>....................] - ETA: 47s - loss: 7.6473 - accuracy: 0.5013
 8768/25000 [=========>....................] - ETA: 47s - loss: 7.6456 - accuracy: 0.5014
 8800/25000 [=========>....................] - ETA: 47s - loss: 7.6475 - accuracy: 0.5013
 8832/25000 [=========>....................] - ETA: 47s - loss: 7.6406 - accuracy: 0.5017
 8864/25000 [=========>....................] - ETA: 47s - loss: 7.6424 - accuracy: 0.5016
 8896/25000 [=========>....................] - ETA: 47s - loss: 7.6494 - accuracy: 0.5011
 8928/25000 [=========>....................] - ETA: 47s - loss: 7.6460 - accuracy: 0.5013
 8960/25000 [=========>....................] - ETA: 46s - loss: 7.6444 - accuracy: 0.5015
 8992/25000 [=========>....................] - ETA: 46s - loss: 7.6479 - accuracy: 0.5012
 9024/25000 [=========>....................] - ETA: 46s - loss: 7.6530 - accuracy: 0.5009
 9056/25000 [=========>....................] - ETA: 46s - loss: 7.6514 - accuracy: 0.5010
 9088/25000 [=========>....................] - ETA: 46s - loss: 7.6464 - accuracy: 0.5013
 9120/25000 [=========>....................] - ETA: 46s - loss: 7.6380 - accuracy: 0.5019
 9152/25000 [=========>....................] - ETA: 46s - loss: 7.6432 - accuracy: 0.5015
 9184/25000 [==========>...................] - ETA: 46s - loss: 7.6432 - accuracy: 0.5015
 9216/25000 [==========>...................] - ETA: 46s - loss: 7.6417 - accuracy: 0.5016
 9248/25000 [==========>...................] - ETA: 46s - loss: 7.6434 - accuracy: 0.5015
 9280/25000 [==========>...................] - ETA: 45s - loss: 7.6435 - accuracy: 0.5015
 9312/25000 [==========>...................] - ETA: 45s - loss: 7.6386 - accuracy: 0.5018
 9344/25000 [==========>...................] - ETA: 45s - loss: 7.6354 - accuracy: 0.5020
 9376/25000 [==========>...................] - ETA: 45s - loss: 7.6323 - accuracy: 0.5022
 9408/25000 [==========>...................] - ETA: 45s - loss: 7.6308 - accuracy: 0.5023
 9440/25000 [==========>...................] - ETA: 45s - loss: 7.6374 - accuracy: 0.5019
 9472/25000 [==========>...................] - ETA: 45s - loss: 7.6391 - accuracy: 0.5018
 9504/25000 [==========>...................] - ETA: 45s - loss: 7.6408 - accuracy: 0.5017
 9536/25000 [==========>...................] - ETA: 45s - loss: 7.6409 - accuracy: 0.5017
 9568/25000 [==========>...................] - ETA: 45s - loss: 7.6506 - accuracy: 0.5010
 9600/25000 [==========>...................] - ETA: 45s - loss: 7.6522 - accuracy: 0.5009
 9632/25000 [==========>...................] - ETA: 44s - loss: 7.6539 - accuracy: 0.5008
 9664/25000 [==========>...................] - ETA: 44s - loss: 7.6587 - accuracy: 0.5005
 9696/25000 [==========>...................] - ETA: 44s - loss: 7.6603 - accuracy: 0.5004
 9728/25000 [==========>...................] - ETA: 44s - loss: 7.6587 - accuracy: 0.5005
 9760/25000 [==========>...................] - ETA: 44s - loss: 7.6603 - accuracy: 0.5004
 9792/25000 [==========>...................] - ETA: 44s - loss: 7.6604 - accuracy: 0.5004
 9824/25000 [==========>...................] - ETA: 44s - loss: 7.6619 - accuracy: 0.5003
 9856/25000 [==========>...................] - ETA: 44s - loss: 7.6573 - accuracy: 0.5006
 9888/25000 [==========>...................] - ETA: 44s - loss: 7.6558 - accuracy: 0.5007
 9920/25000 [==========>...................] - ETA: 44s - loss: 7.6620 - accuracy: 0.5003
 9952/25000 [==========>...................] - ETA: 43s - loss: 7.6620 - accuracy: 0.5003
 9984/25000 [==========>...................] - ETA: 43s - loss: 7.6589 - accuracy: 0.5005
10016/25000 [===========>..................] - ETA: 43s - loss: 7.6528 - accuracy: 0.5009
10048/25000 [===========>..................] - ETA: 43s - loss: 7.6559 - accuracy: 0.5007
10080/25000 [===========>..................] - ETA: 43s - loss: 7.6605 - accuracy: 0.5004
10112/25000 [===========>..................] - ETA: 43s - loss: 7.6621 - accuracy: 0.5003
10144/25000 [===========>..................] - ETA: 43s - loss: 7.6651 - accuracy: 0.5001
10176/25000 [===========>..................] - ETA: 43s - loss: 7.6651 - accuracy: 0.5001
10208/25000 [===========>..................] - ETA: 43s - loss: 7.6651 - accuracy: 0.5001
10240/25000 [===========>..................] - ETA: 43s - loss: 7.6651 - accuracy: 0.5001
10272/25000 [===========>..................] - ETA: 43s - loss: 7.6621 - accuracy: 0.5003
10304/25000 [===========>..................] - ETA: 42s - loss: 7.6592 - accuracy: 0.5005
10336/25000 [===========>..................] - ETA: 42s - loss: 7.6681 - accuracy: 0.4999
10368/25000 [===========>..................] - ETA: 42s - loss: 7.6740 - accuracy: 0.4995
10400/25000 [===========>..................] - ETA: 42s - loss: 7.6696 - accuracy: 0.4998
10432/25000 [===========>..................] - ETA: 42s - loss: 7.6637 - accuracy: 0.5002
10464/25000 [===========>..................] - ETA: 42s - loss: 7.6564 - accuracy: 0.5007
10496/25000 [===========>..................] - ETA: 42s - loss: 7.6520 - accuracy: 0.5010
10528/25000 [===========>..................] - ETA: 42s - loss: 7.6521 - accuracy: 0.5009
10560/25000 [===========>..................] - ETA: 42s - loss: 7.6565 - accuracy: 0.5007
10592/25000 [===========>..................] - ETA: 42s - loss: 7.6536 - accuracy: 0.5008
10624/25000 [===========>..................] - ETA: 41s - loss: 7.6507 - accuracy: 0.5010
10656/25000 [===========>..................] - ETA: 41s - loss: 7.6508 - accuracy: 0.5010
10688/25000 [===========>..................] - ETA: 41s - loss: 7.6537 - accuracy: 0.5008
10720/25000 [===========>..................] - ETA: 41s - loss: 7.6523 - accuracy: 0.5009
10752/25000 [===========>..................] - ETA: 41s - loss: 7.6538 - accuracy: 0.5008
10784/25000 [===========>..................] - ETA: 41s - loss: 7.6581 - accuracy: 0.5006
10816/25000 [===========>..................] - ETA: 41s - loss: 7.6595 - accuracy: 0.5005
10848/25000 [============>.................] - ETA: 41s - loss: 7.6638 - accuracy: 0.5002
10880/25000 [============>.................] - ETA: 41s - loss: 7.6610 - accuracy: 0.5004
10912/25000 [============>.................] - ETA: 41s - loss: 7.6596 - accuracy: 0.5005
10944/25000 [============>.................] - ETA: 40s - loss: 7.6582 - accuracy: 0.5005
10976/25000 [============>.................] - ETA: 40s - loss: 7.6540 - accuracy: 0.5008
11008/25000 [============>.................] - ETA: 40s - loss: 7.6513 - accuracy: 0.5010
11040/25000 [============>.................] - ETA: 40s - loss: 7.6513 - accuracy: 0.5010
11072/25000 [============>.................] - ETA: 40s - loss: 7.6486 - accuracy: 0.5012
11104/25000 [============>.................] - ETA: 40s - loss: 7.6459 - accuracy: 0.5014
11136/25000 [============>.................] - ETA: 40s - loss: 7.6487 - accuracy: 0.5012
11168/25000 [============>.................] - ETA: 40s - loss: 7.6460 - accuracy: 0.5013
11200/25000 [============>.................] - ETA: 40s - loss: 7.6543 - accuracy: 0.5008
11232/25000 [============>.................] - ETA: 40s - loss: 7.6543 - accuracy: 0.5008
11264/25000 [============>.................] - ETA: 40s - loss: 7.6516 - accuracy: 0.5010
11296/25000 [============>.................] - ETA: 39s - loss: 7.6517 - accuracy: 0.5010
11328/25000 [============>.................] - ETA: 39s - loss: 7.6544 - accuracy: 0.5008
11360/25000 [============>.................] - ETA: 39s - loss: 7.6545 - accuracy: 0.5008
11392/25000 [============>.................] - ETA: 39s - loss: 7.6572 - accuracy: 0.5006
11424/25000 [============>.................] - ETA: 39s - loss: 7.6519 - accuracy: 0.5010
11456/25000 [============>.................] - ETA: 39s - loss: 7.6532 - accuracy: 0.5009
11488/25000 [============>.................] - ETA: 39s - loss: 7.6479 - accuracy: 0.5012
11520/25000 [============>.................] - ETA: 39s - loss: 7.6453 - accuracy: 0.5014
11552/25000 [============>.................] - ETA: 39s - loss: 7.6441 - accuracy: 0.5015
11584/25000 [============>.................] - ETA: 39s - loss: 7.6415 - accuracy: 0.5016
11616/25000 [============>.................] - ETA: 39s - loss: 7.6415 - accuracy: 0.5016
11648/25000 [============>.................] - ETA: 38s - loss: 7.6390 - accuracy: 0.5018
11680/25000 [=============>................] - ETA: 38s - loss: 7.6351 - accuracy: 0.5021
11712/25000 [=============>................] - ETA: 38s - loss: 7.6352 - accuracy: 0.5020
11744/25000 [=============>................] - ETA: 38s - loss: 7.6327 - accuracy: 0.5022
11776/25000 [=============>................] - ETA: 38s - loss: 7.6315 - accuracy: 0.5023
11808/25000 [=============>................] - ETA: 38s - loss: 7.6290 - accuracy: 0.5025
11840/25000 [=============>................] - ETA: 38s - loss: 7.6329 - accuracy: 0.5022
11872/25000 [=============>................] - ETA: 38s - loss: 7.6317 - accuracy: 0.5023
11904/25000 [=============>................] - ETA: 38s - loss: 7.6370 - accuracy: 0.5019
11936/25000 [=============>................] - ETA: 38s - loss: 7.6345 - accuracy: 0.5021
11968/25000 [=============>................] - ETA: 37s - loss: 7.6372 - accuracy: 0.5019
12000/25000 [=============>................] - ETA: 37s - loss: 7.6321 - accuracy: 0.5023
12032/25000 [=============>................] - ETA: 37s - loss: 7.6360 - accuracy: 0.5020
12064/25000 [=============>................] - ETA: 37s - loss: 7.6361 - accuracy: 0.5020
12096/25000 [=============>................] - ETA: 37s - loss: 7.6349 - accuracy: 0.5021
12128/25000 [=============>................] - ETA: 37s - loss: 7.6363 - accuracy: 0.5020
12160/25000 [=============>................] - ETA: 37s - loss: 7.6351 - accuracy: 0.5021
12192/25000 [=============>................] - ETA: 37s - loss: 7.6390 - accuracy: 0.5018
12224/25000 [=============>................] - ETA: 37s - loss: 7.6328 - accuracy: 0.5022
12256/25000 [=============>................] - ETA: 37s - loss: 7.6316 - accuracy: 0.5023
12288/25000 [=============>................] - ETA: 37s - loss: 7.6329 - accuracy: 0.5022
12320/25000 [=============>................] - ETA: 36s - loss: 7.6343 - accuracy: 0.5021
12352/25000 [=============>................] - ETA: 36s - loss: 7.6343 - accuracy: 0.5021
12384/25000 [=============>................] - ETA: 36s - loss: 7.6307 - accuracy: 0.5023
12416/25000 [=============>................] - ETA: 36s - loss: 7.6296 - accuracy: 0.5024
12448/25000 [=============>................] - ETA: 36s - loss: 7.6297 - accuracy: 0.5024
12480/25000 [=============>................] - ETA: 36s - loss: 7.6236 - accuracy: 0.5028
12512/25000 [==============>...............] - ETA: 36s - loss: 7.6311 - accuracy: 0.5023
12544/25000 [==============>...............] - ETA: 36s - loss: 7.6348 - accuracy: 0.5021
12576/25000 [==============>...............] - ETA: 36s - loss: 7.6410 - accuracy: 0.5017
12608/25000 [==============>...............] - ETA: 36s - loss: 7.6374 - accuracy: 0.5019
12640/25000 [==============>...............] - ETA: 36s - loss: 7.6411 - accuracy: 0.5017
12672/25000 [==============>...............] - ETA: 35s - loss: 7.6376 - accuracy: 0.5019
12704/25000 [==============>...............] - ETA: 35s - loss: 7.6352 - accuracy: 0.5020
12736/25000 [==============>...............] - ETA: 35s - loss: 7.6401 - accuracy: 0.5017
12768/25000 [==============>...............] - ETA: 35s - loss: 7.6378 - accuracy: 0.5019
12800/25000 [==============>...............] - ETA: 35s - loss: 7.6355 - accuracy: 0.5020
12832/25000 [==============>...............] - ETA: 35s - loss: 7.6344 - accuracy: 0.5021
12864/25000 [==============>...............] - ETA: 35s - loss: 7.6368 - accuracy: 0.5019
12896/25000 [==============>...............] - ETA: 35s - loss: 7.6345 - accuracy: 0.5021
12928/25000 [==============>...............] - ETA: 35s - loss: 7.6322 - accuracy: 0.5022
12960/25000 [==============>...............] - ETA: 35s - loss: 7.6323 - accuracy: 0.5022
12992/25000 [==============>...............] - ETA: 34s - loss: 7.6277 - accuracy: 0.5025
13024/25000 [==============>...............] - ETA: 34s - loss: 7.6289 - accuracy: 0.5025
13056/25000 [==============>...............] - ETA: 34s - loss: 7.6267 - accuracy: 0.5026
13088/25000 [==============>...............] - ETA: 34s - loss: 7.6291 - accuracy: 0.5024
13120/25000 [==============>...............] - ETA: 34s - loss: 7.6269 - accuracy: 0.5026
13152/25000 [==============>...............] - ETA: 34s - loss: 7.6188 - accuracy: 0.5031
13184/25000 [==============>...............] - ETA: 34s - loss: 7.6201 - accuracy: 0.5030
13216/25000 [==============>...............] - ETA: 34s - loss: 7.6156 - accuracy: 0.5033
13248/25000 [==============>...............] - ETA: 34s - loss: 7.6192 - accuracy: 0.5031
13280/25000 [==============>...............] - ETA: 34s - loss: 7.6170 - accuracy: 0.5032
13312/25000 [==============>...............] - ETA: 34s - loss: 7.6159 - accuracy: 0.5033
13344/25000 [===============>..............] - ETA: 33s - loss: 7.6126 - accuracy: 0.5035
13376/25000 [===============>..............] - ETA: 33s - loss: 7.6162 - accuracy: 0.5033
13408/25000 [===============>..............] - ETA: 33s - loss: 7.6163 - accuracy: 0.5033
13440/25000 [===============>..............] - ETA: 33s - loss: 7.6153 - accuracy: 0.5033
13472/25000 [===============>..............] - ETA: 33s - loss: 7.6120 - accuracy: 0.5036
13504/25000 [===============>..............] - ETA: 33s - loss: 7.6189 - accuracy: 0.5031
13536/25000 [===============>..............] - ETA: 33s - loss: 7.6213 - accuracy: 0.5030
13568/25000 [===============>..............] - ETA: 33s - loss: 7.6214 - accuracy: 0.5029
13600/25000 [===============>..............] - ETA: 33s - loss: 7.6204 - accuracy: 0.5030
13632/25000 [===============>..............] - ETA: 33s - loss: 7.6183 - accuracy: 0.5032
13664/25000 [===============>..............] - ETA: 32s - loss: 7.6161 - accuracy: 0.5033
13696/25000 [===============>..............] - ETA: 32s - loss: 7.6162 - accuracy: 0.5033
13728/25000 [===============>..............] - ETA: 32s - loss: 7.6108 - accuracy: 0.5036
13760/25000 [===============>..............] - ETA: 32s - loss: 7.6064 - accuracy: 0.5039
13792/25000 [===============>..............] - ETA: 32s - loss: 7.6055 - accuracy: 0.5040
13824/25000 [===============>..............] - ETA: 32s - loss: 7.6101 - accuracy: 0.5037
13856/25000 [===============>..............] - ETA: 32s - loss: 7.6046 - accuracy: 0.5040
13888/25000 [===============>..............] - ETA: 32s - loss: 7.6037 - accuracy: 0.5041
13920/25000 [===============>..............] - ETA: 32s - loss: 7.6049 - accuracy: 0.5040
13952/25000 [===============>..............] - ETA: 32s - loss: 7.6029 - accuracy: 0.5042
13984/25000 [===============>..............] - ETA: 32s - loss: 7.5986 - accuracy: 0.5044
14016/25000 [===============>..............] - ETA: 31s - loss: 7.6021 - accuracy: 0.5042
14048/25000 [===============>..............] - ETA: 31s - loss: 7.5989 - accuracy: 0.5044
14080/25000 [===============>..............] - ETA: 31s - loss: 7.6035 - accuracy: 0.5041
14112/25000 [===============>..............] - ETA: 31s - loss: 7.6003 - accuracy: 0.5043
14144/25000 [===============>..............] - ETA: 31s - loss: 7.6037 - accuracy: 0.5041
14176/25000 [================>.............] - ETA: 31s - loss: 7.6017 - accuracy: 0.5042
14208/25000 [================>.............] - ETA: 31s - loss: 7.6008 - accuracy: 0.5043
14240/25000 [================>.............] - ETA: 31s - loss: 7.6042 - accuracy: 0.5041
14272/25000 [================>.............] - ETA: 31s - loss: 7.6032 - accuracy: 0.5041
14304/25000 [================>.............] - ETA: 31s - loss: 7.6055 - accuracy: 0.5040
14336/25000 [================>.............] - ETA: 31s - loss: 7.6121 - accuracy: 0.5036
14368/25000 [================>.............] - ETA: 30s - loss: 7.6133 - accuracy: 0.5035
14400/25000 [================>.............] - ETA: 30s - loss: 7.6144 - accuracy: 0.5034
14432/25000 [================>.............] - ETA: 30s - loss: 7.6156 - accuracy: 0.5033
14464/25000 [================>.............] - ETA: 30s - loss: 7.6168 - accuracy: 0.5032
14496/25000 [================>.............] - ETA: 30s - loss: 7.6233 - accuracy: 0.5028
14528/25000 [================>.............] - ETA: 30s - loss: 7.6223 - accuracy: 0.5029
14560/25000 [================>.............] - ETA: 30s - loss: 7.6234 - accuracy: 0.5028
14592/25000 [================>.............] - ETA: 30s - loss: 7.6246 - accuracy: 0.5027
14624/25000 [================>.............] - ETA: 30s - loss: 7.6268 - accuracy: 0.5026
14656/25000 [================>.............] - ETA: 30s - loss: 7.6321 - accuracy: 0.5023
14688/25000 [================>.............] - ETA: 29s - loss: 7.6343 - accuracy: 0.5021
14720/25000 [================>.............] - ETA: 29s - loss: 7.6333 - accuracy: 0.5022
14752/25000 [================>.............] - ETA: 29s - loss: 7.6365 - accuracy: 0.5020
14784/25000 [================>.............] - ETA: 29s - loss: 7.6365 - accuracy: 0.5020
14816/25000 [================>.............] - ETA: 29s - loss: 7.6366 - accuracy: 0.5020
14848/25000 [================>.............] - ETA: 29s - loss: 7.6367 - accuracy: 0.5020
14880/25000 [================>.............] - ETA: 29s - loss: 7.6378 - accuracy: 0.5019
14912/25000 [================>.............] - ETA: 29s - loss: 7.6389 - accuracy: 0.5018
14944/25000 [================>.............] - ETA: 29s - loss: 7.6420 - accuracy: 0.5016
14976/25000 [================>.............] - ETA: 29s - loss: 7.6441 - accuracy: 0.5015
15008/25000 [=================>............] - ETA: 29s - loss: 7.6411 - accuracy: 0.5017
15040/25000 [=================>............] - ETA: 28s - loss: 7.6391 - accuracy: 0.5018
15072/25000 [=================>............] - ETA: 28s - loss: 7.6381 - accuracy: 0.5019
15104/25000 [=================>............] - ETA: 28s - loss: 7.6382 - accuracy: 0.5019
15136/25000 [=================>............] - ETA: 28s - loss: 7.6362 - accuracy: 0.5020
15168/25000 [=================>............] - ETA: 28s - loss: 7.6363 - accuracy: 0.5020
15200/25000 [=================>............] - ETA: 28s - loss: 7.6313 - accuracy: 0.5023
15232/25000 [=================>............] - ETA: 28s - loss: 7.6314 - accuracy: 0.5023
15264/25000 [=================>............] - ETA: 28s - loss: 7.6325 - accuracy: 0.5022
15296/25000 [=================>............] - ETA: 28s - loss: 7.6345 - accuracy: 0.5021
15328/25000 [=================>............] - ETA: 28s - loss: 7.6396 - accuracy: 0.5018
15360/25000 [=================>............] - ETA: 28s - loss: 7.6367 - accuracy: 0.5020
15392/25000 [=================>............] - ETA: 27s - loss: 7.6298 - accuracy: 0.5024
15424/25000 [=================>............] - ETA: 27s - loss: 7.6358 - accuracy: 0.5020
15456/25000 [=================>............] - ETA: 27s - loss: 7.6339 - accuracy: 0.5021
15488/25000 [=================>............] - ETA: 27s - loss: 7.6330 - accuracy: 0.5022
15520/25000 [=================>............] - ETA: 27s - loss: 7.6350 - accuracy: 0.5021
15552/25000 [=================>............] - ETA: 27s - loss: 7.6361 - accuracy: 0.5020
15584/25000 [=================>............] - ETA: 27s - loss: 7.6361 - accuracy: 0.5020
15616/25000 [=================>............] - ETA: 27s - loss: 7.6293 - accuracy: 0.5024
15648/25000 [=================>............] - ETA: 27s - loss: 7.6304 - accuracy: 0.5024
15680/25000 [=================>............] - ETA: 27s - loss: 7.6343 - accuracy: 0.5021
15712/25000 [=================>............] - ETA: 27s - loss: 7.6334 - accuracy: 0.5022
15744/25000 [=================>............] - ETA: 26s - loss: 7.6355 - accuracy: 0.5020
15776/25000 [=================>............] - ETA: 26s - loss: 7.6394 - accuracy: 0.5018
15808/25000 [=================>............] - ETA: 26s - loss: 7.6385 - accuracy: 0.5018
15840/25000 [==================>...........] - ETA: 26s - loss: 7.6385 - accuracy: 0.5018
15872/25000 [==================>...........] - ETA: 26s - loss: 7.6376 - accuracy: 0.5019
15904/25000 [==================>...........] - ETA: 26s - loss: 7.6338 - accuracy: 0.5021
15936/25000 [==================>...........] - ETA: 26s - loss: 7.6349 - accuracy: 0.5021
15968/25000 [==================>...........] - ETA: 26s - loss: 7.6369 - accuracy: 0.5019
16000/25000 [==================>...........] - ETA: 26s - loss: 7.6340 - accuracy: 0.5021
16032/25000 [==================>...........] - ETA: 26s - loss: 7.6379 - accuracy: 0.5019
16064/25000 [==================>...........] - ETA: 25s - loss: 7.6370 - accuracy: 0.5019
16096/25000 [==================>...........] - ETA: 25s - loss: 7.6380 - accuracy: 0.5019
16128/25000 [==================>...........] - ETA: 25s - loss: 7.6381 - accuracy: 0.5019
16160/25000 [==================>...........] - ETA: 25s - loss: 7.6372 - accuracy: 0.5019
16192/25000 [==================>...........] - ETA: 25s - loss: 7.6354 - accuracy: 0.5020
16224/25000 [==================>...........] - ETA: 25s - loss: 7.6298 - accuracy: 0.5024
16256/25000 [==================>...........] - ETA: 25s - loss: 7.6317 - accuracy: 0.5023
16288/25000 [==================>...........] - ETA: 25s - loss: 7.6337 - accuracy: 0.5021
16320/25000 [==================>...........] - ETA: 25s - loss: 7.6366 - accuracy: 0.5020
16352/25000 [==================>...........] - ETA: 25s - loss: 7.6385 - accuracy: 0.5018
16384/25000 [==================>...........] - ETA: 25s - loss: 7.6385 - accuracy: 0.5018
16416/25000 [==================>...........] - ETA: 24s - loss: 7.6358 - accuracy: 0.5020
16448/25000 [==================>...........] - ETA: 24s - loss: 7.6359 - accuracy: 0.5020
16480/25000 [==================>...........] - ETA: 24s - loss: 7.6313 - accuracy: 0.5023
16512/25000 [==================>...........] - ETA: 24s - loss: 7.6313 - accuracy: 0.5023
16544/25000 [==================>...........] - ETA: 24s - loss: 7.6333 - accuracy: 0.5022
16576/25000 [==================>...........] - ETA: 24s - loss: 7.6296 - accuracy: 0.5024
16608/25000 [==================>...........] - ETA: 24s - loss: 7.6288 - accuracy: 0.5025
16640/25000 [==================>...........] - ETA: 24s - loss: 7.6261 - accuracy: 0.5026
16672/25000 [===================>..........] - ETA: 24s - loss: 7.6262 - accuracy: 0.5026
16704/25000 [===================>..........] - ETA: 24s - loss: 7.6271 - accuracy: 0.5026
16736/25000 [===================>..........] - ETA: 24s - loss: 7.6300 - accuracy: 0.5024
16768/25000 [===================>..........] - ETA: 23s - loss: 7.6236 - accuracy: 0.5028
16800/25000 [===================>..........] - ETA: 23s - loss: 7.6246 - accuracy: 0.5027
16832/25000 [===================>..........] - ETA: 23s - loss: 7.6211 - accuracy: 0.5030
16864/25000 [===================>..........] - ETA: 23s - loss: 7.6175 - accuracy: 0.5032
16896/25000 [===================>..........] - ETA: 23s - loss: 7.6167 - accuracy: 0.5033
16928/25000 [===================>..........] - ETA: 23s - loss: 7.6123 - accuracy: 0.5035
16960/25000 [===================>..........] - ETA: 23s - loss: 7.6124 - accuracy: 0.5035
16992/25000 [===================>..........] - ETA: 23s - loss: 7.6116 - accuracy: 0.5036
17024/25000 [===================>..........] - ETA: 23s - loss: 7.6090 - accuracy: 0.5038
17056/25000 [===================>..........] - ETA: 23s - loss: 7.6091 - accuracy: 0.5038
17088/25000 [===================>..........] - ETA: 23s - loss: 7.6083 - accuracy: 0.5038
17120/25000 [===================>..........] - ETA: 22s - loss: 7.6138 - accuracy: 0.5034
17152/25000 [===================>..........] - ETA: 22s - loss: 7.6148 - accuracy: 0.5034
17184/25000 [===================>..........] - ETA: 22s - loss: 7.6140 - accuracy: 0.5034
17216/25000 [===================>..........] - ETA: 22s - loss: 7.6194 - accuracy: 0.5031
17248/25000 [===================>..........] - ETA: 22s - loss: 7.6213 - accuracy: 0.5030
17280/25000 [===================>..........] - ETA: 22s - loss: 7.6231 - accuracy: 0.5028
17312/25000 [===================>..........] - ETA: 22s - loss: 7.6276 - accuracy: 0.5025
17344/25000 [===================>..........] - ETA: 22s - loss: 7.6277 - accuracy: 0.5025
17376/25000 [===================>..........] - ETA: 22s - loss: 7.6207 - accuracy: 0.5030
17408/25000 [===================>..........] - ETA: 22s - loss: 7.6191 - accuracy: 0.5031
17440/25000 [===================>..........] - ETA: 21s - loss: 7.6191 - accuracy: 0.5031
17472/25000 [===================>..........] - ETA: 21s - loss: 7.6201 - accuracy: 0.5030
17504/25000 [====================>.........] - ETA: 21s - loss: 7.6176 - accuracy: 0.5032
17536/25000 [====================>.........] - ETA: 21s - loss: 7.6203 - accuracy: 0.5030
17568/25000 [====================>.........] - ETA: 21s - loss: 7.6212 - accuracy: 0.5030
17600/25000 [====================>.........] - ETA: 21s - loss: 7.6196 - accuracy: 0.5031
17632/25000 [====================>.........] - ETA: 21s - loss: 7.6257 - accuracy: 0.5027
17664/25000 [====================>.........] - ETA: 21s - loss: 7.6232 - accuracy: 0.5028
17696/25000 [====================>.........] - ETA: 21s - loss: 7.6224 - accuracy: 0.5029
17728/25000 [====================>.........] - ETA: 21s - loss: 7.6225 - accuracy: 0.5029
17760/25000 [====================>.........] - ETA: 21s - loss: 7.6260 - accuracy: 0.5026
17792/25000 [====================>.........] - ETA: 20s - loss: 7.6261 - accuracy: 0.5026
17824/25000 [====================>.........] - ETA: 20s - loss: 7.6262 - accuracy: 0.5026
17856/25000 [====================>.........] - ETA: 20s - loss: 7.6288 - accuracy: 0.5025
17888/25000 [====================>.........] - ETA: 20s - loss: 7.6272 - accuracy: 0.5026
17920/25000 [====================>.........] - ETA: 20s - loss: 7.6315 - accuracy: 0.5023
17952/25000 [====================>.........] - ETA: 20s - loss: 7.6307 - accuracy: 0.5023
17984/25000 [====================>.........] - ETA: 20s - loss: 7.6283 - accuracy: 0.5025
18016/25000 [====================>.........] - ETA: 20s - loss: 7.6275 - accuracy: 0.5026
18048/25000 [====================>.........] - ETA: 20s - loss: 7.6318 - accuracy: 0.5023
18080/25000 [====================>.........] - ETA: 20s - loss: 7.6335 - accuracy: 0.5022
18112/25000 [====================>.........] - ETA: 20s - loss: 7.6328 - accuracy: 0.5022
18144/25000 [====================>.........] - ETA: 19s - loss: 7.6328 - accuracy: 0.5022
18176/25000 [====================>.........] - ETA: 19s - loss: 7.6362 - accuracy: 0.5020
18208/25000 [====================>.........] - ETA: 19s - loss: 7.6371 - accuracy: 0.5019
18240/25000 [====================>.........] - ETA: 19s - loss: 7.6372 - accuracy: 0.5019
18272/25000 [====================>.........] - ETA: 19s - loss: 7.6356 - accuracy: 0.5020
18304/25000 [====================>.........] - ETA: 19s - loss: 7.6373 - accuracy: 0.5019
18336/25000 [=====================>........] - ETA: 19s - loss: 7.6357 - accuracy: 0.5020
18368/25000 [=====================>........] - ETA: 19s - loss: 7.6349 - accuracy: 0.5021
18400/25000 [=====================>........] - ETA: 19s - loss: 7.6316 - accuracy: 0.5023
18432/25000 [=====================>........] - ETA: 19s - loss: 7.6325 - accuracy: 0.5022
18464/25000 [=====================>........] - ETA: 18s - loss: 7.6309 - accuracy: 0.5023
18496/25000 [=====================>........] - ETA: 18s - loss: 7.6301 - accuracy: 0.5024
18528/25000 [=====================>........] - ETA: 18s - loss: 7.6277 - accuracy: 0.5025
18560/25000 [=====================>........] - ETA: 18s - loss: 7.6278 - accuracy: 0.5025
18592/25000 [=====================>........] - ETA: 18s - loss: 7.6279 - accuracy: 0.5025
18624/25000 [=====================>........] - ETA: 18s - loss: 7.6287 - accuracy: 0.5025
18656/25000 [=====================>........] - ETA: 18s - loss: 7.6288 - accuracy: 0.5025
18688/25000 [=====================>........] - ETA: 18s - loss: 7.6305 - accuracy: 0.5024
18720/25000 [=====================>........] - ETA: 18s - loss: 7.6273 - accuracy: 0.5026
18752/25000 [=====================>........] - ETA: 18s - loss: 7.6290 - accuracy: 0.5025
18784/25000 [=====================>........] - ETA: 18s - loss: 7.6274 - accuracy: 0.5026
18816/25000 [=====================>........] - ETA: 17s - loss: 7.6275 - accuracy: 0.5026
18848/25000 [=====================>........] - ETA: 17s - loss: 7.6243 - accuracy: 0.5028
18880/25000 [=====================>........] - ETA: 17s - loss: 7.6293 - accuracy: 0.5024
18912/25000 [=====================>........] - ETA: 17s - loss: 7.6309 - accuracy: 0.5023
18944/25000 [=====================>........] - ETA: 17s - loss: 7.6286 - accuracy: 0.5025
18976/25000 [=====================>........] - ETA: 17s - loss: 7.6286 - accuracy: 0.5025
19008/25000 [=====================>........] - ETA: 17s - loss: 7.6279 - accuracy: 0.5025
19040/25000 [=====================>........] - ETA: 17s - loss: 7.6280 - accuracy: 0.5025
19072/25000 [=====================>........] - ETA: 17s - loss: 7.6296 - accuracy: 0.5024
19104/25000 [=====================>........] - ETA: 17s - loss: 7.6289 - accuracy: 0.5025
19136/25000 [=====================>........] - ETA: 17s - loss: 7.6306 - accuracy: 0.5024
19168/25000 [======================>.......] - ETA: 16s - loss: 7.6322 - accuracy: 0.5022
19200/25000 [======================>.......] - ETA: 16s - loss: 7.6315 - accuracy: 0.5023
19232/25000 [======================>.......] - ETA: 16s - loss: 7.6299 - accuracy: 0.5024
19264/25000 [======================>.......] - ETA: 16s - loss: 7.6276 - accuracy: 0.5025
19296/25000 [======================>.......] - ETA: 16s - loss: 7.6277 - accuracy: 0.5025
19328/25000 [======================>.......] - ETA: 16s - loss: 7.6262 - accuracy: 0.5026
19360/25000 [======================>.......] - ETA: 16s - loss: 7.6246 - accuracy: 0.5027
19392/25000 [======================>.......] - ETA: 16s - loss: 7.6255 - accuracy: 0.5027
19424/25000 [======================>.......] - ETA: 16s - loss: 7.6256 - accuracy: 0.5027
19456/25000 [======================>.......] - ETA: 16s - loss: 7.6241 - accuracy: 0.5028
19488/25000 [======================>.......] - ETA: 15s - loss: 7.6265 - accuracy: 0.5026
19520/25000 [======================>.......] - ETA: 15s - loss: 7.6258 - accuracy: 0.5027
19552/25000 [======================>.......] - ETA: 15s - loss: 7.6274 - accuracy: 0.5026
19584/25000 [======================>.......] - ETA: 15s - loss: 7.6236 - accuracy: 0.5028
19616/25000 [======================>.......] - ETA: 15s - loss: 7.6252 - accuracy: 0.5027
19648/25000 [======================>.......] - ETA: 15s - loss: 7.6221 - accuracy: 0.5029
19680/25000 [======================>.......] - ETA: 15s - loss: 7.6207 - accuracy: 0.5030
19712/25000 [======================>.......] - ETA: 15s - loss: 7.6192 - accuracy: 0.5031
19744/25000 [======================>.......] - ETA: 15s - loss: 7.6192 - accuracy: 0.5031
19776/25000 [======================>.......] - ETA: 15s - loss: 7.6248 - accuracy: 0.5027
19808/25000 [======================>.......] - ETA: 15s - loss: 7.6248 - accuracy: 0.5027
19840/25000 [======================>.......] - ETA: 14s - loss: 7.6241 - accuracy: 0.5028
19872/25000 [======================>.......] - ETA: 14s - loss: 7.6242 - accuracy: 0.5028
19904/25000 [======================>.......] - ETA: 14s - loss: 7.6258 - accuracy: 0.5027
19936/25000 [======================>.......] - ETA: 14s - loss: 7.6289 - accuracy: 0.5025
19968/25000 [======================>.......] - ETA: 14s - loss: 7.6267 - accuracy: 0.5026
20000/25000 [=======================>......] - ETA: 14s - loss: 7.6275 - accuracy: 0.5026
20032/25000 [=======================>......] - ETA: 14s - loss: 7.6268 - accuracy: 0.5026
20064/25000 [=======================>......] - ETA: 14s - loss: 7.6330 - accuracy: 0.5022
20096/25000 [=======================>......] - ETA: 14s - loss: 7.6330 - accuracy: 0.5022
20128/25000 [=======================>......] - ETA: 14s - loss: 7.6308 - accuracy: 0.5023
20160/25000 [=======================>......] - ETA: 14s - loss: 7.6294 - accuracy: 0.5024
20192/25000 [=======================>......] - ETA: 13s - loss: 7.6324 - accuracy: 0.5022
20224/25000 [=======================>......] - ETA: 13s - loss: 7.6363 - accuracy: 0.5020
20256/25000 [=======================>......] - ETA: 13s - loss: 7.6379 - accuracy: 0.5019
20288/25000 [=======================>......] - ETA: 13s - loss: 7.6334 - accuracy: 0.5022
20320/25000 [=======================>......] - ETA: 13s - loss: 7.6342 - accuracy: 0.5021
20352/25000 [=======================>......] - ETA: 13s - loss: 7.6335 - accuracy: 0.5022
20384/25000 [=======================>......] - ETA: 13s - loss: 7.6328 - accuracy: 0.5022
20416/25000 [=======================>......] - ETA: 13s - loss: 7.6366 - accuracy: 0.5020
20448/25000 [=======================>......] - ETA: 13s - loss: 7.6381 - accuracy: 0.5019
20480/25000 [=======================>......] - ETA: 13s - loss: 7.6382 - accuracy: 0.5019
20512/25000 [=======================>......] - ETA: 13s - loss: 7.6367 - accuracy: 0.5020
20544/25000 [=======================>......] - ETA: 12s - loss: 7.6383 - accuracy: 0.5018
20576/25000 [=======================>......] - ETA: 12s - loss: 7.6368 - accuracy: 0.5019
20608/25000 [=======================>......] - ETA: 12s - loss: 7.6383 - accuracy: 0.5018
20640/25000 [=======================>......] - ETA: 12s - loss: 7.6399 - accuracy: 0.5017
20672/25000 [=======================>......] - ETA: 12s - loss: 7.6384 - accuracy: 0.5018
20704/25000 [=======================>......] - ETA: 12s - loss: 7.6370 - accuracy: 0.5019
20736/25000 [=======================>......] - ETA: 12s - loss: 7.6341 - accuracy: 0.5021
20768/25000 [=======================>......] - ETA: 12s - loss: 7.6363 - accuracy: 0.5020
20800/25000 [=======================>......] - ETA: 12s - loss: 7.6379 - accuracy: 0.5019
20832/25000 [=======================>......] - ETA: 12s - loss: 7.6386 - accuracy: 0.5018
20864/25000 [========================>.....] - ETA: 11s - loss: 7.6402 - accuracy: 0.5017
20896/25000 [========================>.....] - ETA: 11s - loss: 7.6439 - accuracy: 0.5015
20928/25000 [========================>.....] - ETA: 11s - loss: 7.6468 - accuracy: 0.5013
20960/25000 [========================>.....] - ETA: 11s - loss: 7.6454 - accuracy: 0.5014
20992/25000 [========================>.....] - ETA: 11s - loss: 7.6447 - accuracy: 0.5014
21024/25000 [========================>.....] - ETA: 11s - loss: 7.6404 - accuracy: 0.5017
21056/25000 [========================>.....] - ETA: 11s - loss: 7.6419 - accuracy: 0.5016
21088/25000 [========================>.....] - ETA: 11s - loss: 7.6390 - accuracy: 0.5018
21120/25000 [========================>.....] - ETA: 11s - loss: 7.6434 - accuracy: 0.5015
21152/25000 [========================>.....] - ETA: 11s - loss: 7.6441 - accuracy: 0.5015
21184/25000 [========================>.....] - ETA: 11s - loss: 7.6406 - accuracy: 0.5017
21216/25000 [========================>.....] - ETA: 10s - loss: 7.6420 - accuracy: 0.5016
21248/25000 [========================>.....] - ETA: 10s - loss: 7.6450 - accuracy: 0.5014
21280/25000 [========================>.....] - ETA: 10s - loss: 7.6486 - accuracy: 0.5012
21312/25000 [========================>.....] - ETA: 10s - loss: 7.6494 - accuracy: 0.5011
21344/25000 [========================>.....] - ETA: 10s - loss: 7.6472 - accuracy: 0.5013
21376/25000 [========================>.....] - ETA: 10s - loss: 7.6465 - accuracy: 0.5013
21408/25000 [========================>.....] - ETA: 10s - loss: 7.6473 - accuracy: 0.5013
21440/25000 [========================>.....] - ETA: 10s - loss: 7.6466 - accuracy: 0.5013
21472/25000 [========================>.....] - ETA: 10s - loss: 7.6445 - accuracy: 0.5014
21504/25000 [========================>.....] - ETA: 10s - loss: 7.6459 - accuracy: 0.5013
21536/25000 [========================>.....] - ETA: 10s - loss: 7.6460 - accuracy: 0.5013
21568/25000 [========================>.....] - ETA: 9s - loss: 7.6453 - accuracy: 0.5014 
21600/25000 [========================>.....] - ETA: 9s - loss: 7.6446 - accuracy: 0.5014
21632/25000 [========================>.....] - ETA: 9s - loss: 7.6446 - accuracy: 0.5014
21664/25000 [========================>.....] - ETA: 9s - loss: 7.6426 - accuracy: 0.5016
21696/25000 [=========================>....] - ETA: 9s - loss: 7.6419 - accuracy: 0.5016
21728/25000 [=========================>....] - ETA: 9s - loss: 7.6412 - accuracy: 0.5017
21760/25000 [=========================>....] - ETA: 9s - loss: 7.6413 - accuracy: 0.5017
21792/25000 [=========================>....] - ETA: 9s - loss: 7.6434 - accuracy: 0.5015
21824/25000 [=========================>....] - ETA: 9s - loss: 7.6434 - accuracy: 0.5015
21856/25000 [=========================>....] - ETA: 9s - loss: 7.6407 - accuracy: 0.5017
21888/25000 [=========================>....] - ETA: 9s - loss: 7.6428 - accuracy: 0.5016
21920/25000 [=========================>....] - ETA: 8s - loss: 7.6442 - accuracy: 0.5015
21952/25000 [=========================>....] - ETA: 8s - loss: 7.6464 - accuracy: 0.5013
21984/25000 [=========================>....] - ETA: 8s - loss: 7.6436 - accuracy: 0.5015
22016/25000 [=========================>....] - ETA: 8s - loss: 7.6422 - accuracy: 0.5016
22048/25000 [=========================>....] - ETA: 8s - loss: 7.6423 - accuracy: 0.5016
22080/25000 [=========================>....] - ETA: 8s - loss: 7.6437 - accuracy: 0.5015
22112/25000 [=========================>....] - ETA: 8s - loss: 7.6430 - accuracy: 0.5015
22144/25000 [=========================>....] - ETA: 8s - loss: 7.6465 - accuracy: 0.5013
22176/25000 [=========================>....] - ETA: 8s - loss: 7.6466 - accuracy: 0.5013
22208/25000 [=========================>....] - ETA: 8s - loss: 7.6473 - accuracy: 0.5013
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6446 - accuracy: 0.5014
22272/25000 [=========================>....] - ETA: 7s - loss: 7.6453 - accuracy: 0.5014
22304/25000 [=========================>....] - ETA: 7s - loss: 7.6460 - accuracy: 0.5013
22336/25000 [=========================>....] - ETA: 7s - loss: 7.6467 - accuracy: 0.5013
22368/25000 [=========================>....] - ETA: 7s - loss: 7.6447 - accuracy: 0.5014
22400/25000 [=========================>....] - ETA: 7s - loss: 7.6454 - accuracy: 0.5014
22432/25000 [=========================>....] - ETA: 7s - loss: 7.6482 - accuracy: 0.5012
22464/25000 [=========================>....] - ETA: 7s - loss: 7.6475 - accuracy: 0.5012
22496/25000 [=========================>....] - ETA: 7s - loss: 7.6469 - accuracy: 0.5013
22528/25000 [==========================>...] - ETA: 7s - loss: 7.6442 - accuracy: 0.5015
22560/25000 [==========================>...] - ETA: 7s - loss: 7.6442 - accuracy: 0.5015
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6435 - accuracy: 0.5015
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6443 - accuracy: 0.5015
22656/25000 [==========================>...] - ETA: 6s - loss: 7.6477 - accuracy: 0.5012
22688/25000 [==========================>...] - ETA: 6s - loss: 7.6477 - accuracy: 0.5012
22720/25000 [==========================>...] - ETA: 6s - loss: 7.6504 - accuracy: 0.5011
22752/25000 [==========================>...] - ETA: 6s - loss: 7.6518 - accuracy: 0.5010
22784/25000 [==========================>...] - ETA: 6s - loss: 7.6538 - accuracy: 0.5008
22816/25000 [==========================>...] - ETA: 6s - loss: 7.6545 - accuracy: 0.5008
22848/25000 [==========================>...] - ETA: 6s - loss: 7.6525 - accuracy: 0.5009
22880/25000 [==========================>...] - ETA: 6s - loss: 7.6485 - accuracy: 0.5012
22912/25000 [==========================>...] - ETA: 6s - loss: 7.6465 - accuracy: 0.5013
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6432 - accuracy: 0.5015
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6453 - accuracy: 0.5014
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6446 - accuracy: 0.5014
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6467 - accuracy: 0.5013
23072/25000 [==========================>...] - ETA: 5s - loss: 7.6473 - accuracy: 0.5013
23104/25000 [==========================>...] - ETA: 5s - loss: 7.6520 - accuracy: 0.5010
23136/25000 [==========================>...] - ETA: 5s - loss: 7.6494 - accuracy: 0.5011
23168/25000 [==========================>...] - ETA: 5s - loss: 7.6501 - accuracy: 0.5011
23200/25000 [==========================>...] - ETA: 5s - loss: 7.6541 - accuracy: 0.5008
23232/25000 [==========================>...] - ETA: 5s - loss: 7.6541 - accuracy: 0.5008
23264/25000 [==========================>...] - ETA: 5s - loss: 7.6521 - accuracy: 0.5009
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6482 - accuracy: 0.5012
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6495 - accuracy: 0.5011
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6456 - accuracy: 0.5014
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6463 - accuracy: 0.5013
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6476 - accuracy: 0.5012
23456/25000 [===========================>..] - ETA: 4s - loss: 7.6490 - accuracy: 0.5012
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6490 - accuracy: 0.5011
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6464 - accuracy: 0.5013
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6458 - accuracy: 0.5014
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6478 - accuracy: 0.5012
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6517 - accuracy: 0.5010
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6530 - accuracy: 0.5009
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6485 - accuracy: 0.5012
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6485 - accuracy: 0.5012
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6505 - accuracy: 0.5011
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6505 - accuracy: 0.5011
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6525 - accuracy: 0.5009
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6525 - accuracy: 0.5009
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6525 - accuracy: 0.5009
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6519 - accuracy: 0.5010
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6500 - accuracy: 0.5011
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6506 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6519 - accuracy: 0.5010
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6539 - accuracy: 0.5008
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6526 - accuracy: 0.5009
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6520 - accuracy: 0.5010
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6507 - accuracy: 0.5010
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6539 - accuracy: 0.5008
24192/25000 [============================>.] - ETA: 2s - loss: 7.6533 - accuracy: 0.5009
24224/25000 [============================>.] - ETA: 2s - loss: 7.6540 - accuracy: 0.5008
24256/25000 [============================>.] - ETA: 2s - loss: 7.6559 - accuracy: 0.5007
24288/25000 [============================>.] - ETA: 2s - loss: 7.6578 - accuracy: 0.5006
24320/25000 [============================>.] - ETA: 1s - loss: 7.6591 - accuracy: 0.5005
24352/25000 [============================>.] - ETA: 1s - loss: 7.6578 - accuracy: 0.5006
24384/25000 [============================>.] - ETA: 1s - loss: 7.6584 - accuracy: 0.5005
24416/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5005
24448/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24480/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24512/25000 [============================>.] - ETA: 1s - loss: 7.6629 - accuracy: 0.5002
24544/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24576/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24608/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24640/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24672/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24704/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24736/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24768/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24800/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24832/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24864/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24896/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24928/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24960/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 87s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f02b7367a20> 

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
 [ 0.06566851  0.08561962 -0.07648948  0.18992606  0.08480231  0.0414596 ]
 [ 0.18621415  0.18253948  0.16103394  0.17011482  0.26272434  0.19859365]
 [ 0.28082773 -0.13323139  0.03674018  0.44642857 -0.29572946  0.22290175]
 [ 0.22368465 -0.03685287  0.41049889  0.31863078 -0.05118754 -0.07992152]
 [ 0.14897829 -0.10550877  0.12953161  0.3377777   0.15066433  0.10810257]
 [-0.19541225  0.17948474  0.40073472  0.2750192  -0.23189278 -0.17559187]
 [ 0.04594968 -0.27457327  0.38028318  0.10918602 -0.23479119 -0.27709725]
 [ 0.11199249  0.01800496  0.12840229  0.18630987  0.1773078   0.44319379]
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
{'loss': 0.5201285183429718, 'loss_history': []}

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
{'loss': 0.4298148714005947, 'loss_history': []}

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
 40%|████      | 2/5 [00:48<01:13, 24.34s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.4532244855402505, 'embedding_size_factor': 0.7582536680931207, 'layers.choice': 0, 'learning_rate': 0.0010064123215242343, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 2.544809443026569e-09} and reward: 0.3718
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdd\x01\xa1E\xc9\x13DX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe8C\x9d2Q\x06\x14X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?P}3\x00v\x96xX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>%\xdc\x18[\x7f\xbd\xaeu.' and reward: 0.3718
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdd\x01\xa1E\xc9\x13DX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe8C\x9d2Q\x06\x14X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?P}3\x00v\x96xX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>%\xdc\x18[\x7f\xbd\xaeu.' and reward: 0.3718
 60%|██████    | 3/5 [01:37<01:03, 31.73s/it] 60%|██████    | 3/5 [01:37<01:05, 32.55s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.28416688522328964, 'embedding_size_factor': 1.346672863159119, 'layers.choice': 2, 'learning_rate': 0.00805622332312914, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 0.0033304311046649577} and reward: 0.3694
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd2/\xcaM\xa8\xf9GX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x8b\xf8\xd8\x1a\xdd\x97X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x80\x7f\xc7\xfd\x9d\xec\xf2X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?kHk\x95\xa2\xc0mu.' and reward: 0.3694
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd2/\xcaM\xa8\xf9GX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x8b\xf8\xd8\x1a\xdd\x97X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x80\x7f\xc7\xfd\x9d\xec\xf2X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?kHk\x95\xa2\xc0mu.' and reward: 0.3694
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 148.60415387153625
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.78s of the -30.89s of remaining time.
Ensemble size: 79
Ensemble weights: 
[0.59493671 0.15189873 0.25316456]
	0.39	 = Validation accuracy score
	0.98s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 151.91s ...
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

