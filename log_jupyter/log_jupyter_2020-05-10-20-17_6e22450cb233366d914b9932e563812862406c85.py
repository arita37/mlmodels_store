
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6e22450cb233366d914b9932e563812862406c85', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '6e22450cb233366d914b9932e563812862406c85', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6e22450cb233366d914b9932e563812862406c85

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6e22450cb233366d914b9932e563812862406c85

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
 3006464/17464789 [====>.........................] - ETA: 0s
 9158656/17464789 [==============>...............] - ETA: 0s
16465920/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 20:17:57.773695: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 20:17:57.777638: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095230000 Hz
2020-05-10 20:17:57.777821: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fa52b1cb90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 20:17:57.777842: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:44 - loss: 10.0624 - accuracy: 0.3438
   64/25000 [..............................] - ETA: 2:57 - loss: 8.8645 - accuracy: 0.4219 
   96/25000 [..............................] - ETA: 2:20 - loss: 8.1458 - accuracy: 0.4688
  128/25000 [..............................] - ETA: 2:02 - loss: 8.1458 - accuracy: 0.4688
  160/25000 [..............................] - ETA: 1:50 - loss: 7.7625 - accuracy: 0.4938
  192/25000 [..............................] - ETA: 1:43 - loss: 7.7465 - accuracy: 0.4948
  224/25000 [..............................] - ETA: 1:38 - loss: 7.9404 - accuracy: 0.4821
  256/25000 [..............................] - ETA: 1:33 - loss: 7.9062 - accuracy: 0.4844
  288/25000 [..............................] - ETA: 1:29 - loss: 7.6666 - accuracy: 0.5000
  320/25000 [..............................] - ETA: 1:27 - loss: 7.5229 - accuracy: 0.5094
  352/25000 [..............................] - ETA: 1:25 - loss: 7.6231 - accuracy: 0.5028
  384/25000 [..............................] - ETA: 1:23 - loss: 7.7465 - accuracy: 0.4948
  416/25000 [..............................] - ETA: 1:21 - loss: 7.4823 - accuracy: 0.5120
  448/25000 [..............................] - ETA: 1:20 - loss: 7.4270 - accuracy: 0.5156
  480/25000 [..............................] - ETA: 1:19 - loss: 7.5708 - accuracy: 0.5063
  512/25000 [..............................] - ETA: 1:18 - loss: 7.6367 - accuracy: 0.5020
  544/25000 [..............................] - ETA: 1:17 - loss: 7.6948 - accuracy: 0.4982
  576/25000 [..............................] - ETA: 1:16 - loss: 7.7199 - accuracy: 0.4965
  608/25000 [..............................] - ETA: 1:15 - loss: 7.7675 - accuracy: 0.4934
  640/25000 [..............................] - ETA: 1:15 - loss: 7.7385 - accuracy: 0.4953
  672/25000 [..............................] - ETA: 1:14 - loss: 7.7579 - accuracy: 0.4940
  704/25000 [..............................] - ETA: 1:13 - loss: 7.7973 - accuracy: 0.4915
  736/25000 [..............................] - ETA: 1:13 - loss: 7.7708 - accuracy: 0.4932
  768/25000 [..............................] - ETA: 1:12 - loss: 7.8263 - accuracy: 0.4896
  800/25000 [..............................] - ETA: 1:12 - loss: 7.7433 - accuracy: 0.4950
  832/25000 [..............................] - ETA: 1:12 - loss: 7.7035 - accuracy: 0.4976
  864/25000 [>.............................] - ETA: 1:11 - loss: 7.6666 - accuracy: 0.5000
  896/25000 [>.............................] - ETA: 1:11 - loss: 7.6666 - accuracy: 0.5000
  928/25000 [>.............................] - ETA: 1:10 - loss: 7.6831 - accuracy: 0.4989
  960/25000 [>.............................] - ETA: 1:10 - loss: 7.6666 - accuracy: 0.5000
  992/25000 [>.............................] - ETA: 1:10 - loss: 7.5584 - accuracy: 0.5071
 1024/25000 [>.............................] - ETA: 1:09 - loss: 7.5918 - accuracy: 0.5049
 1056/25000 [>.............................] - ETA: 1:09 - loss: 7.5940 - accuracy: 0.5047
 1088/25000 [>.............................] - ETA: 1:09 - loss: 7.5680 - accuracy: 0.5064
 1120/25000 [>.............................] - ETA: 1:09 - loss: 7.5023 - accuracy: 0.5107
 1152/25000 [>.............................] - ETA: 1:09 - loss: 7.5202 - accuracy: 0.5095
 1184/25000 [>.............................] - ETA: 1:08 - loss: 7.5501 - accuracy: 0.5076
 1216/25000 [>.............................] - ETA: 1:08 - loss: 7.5657 - accuracy: 0.5066
 1248/25000 [>.............................] - ETA: 1:08 - loss: 7.5683 - accuracy: 0.5064
 1280/25000 [>.............................] - ETA: 1:08 - loss: 7.5588 - accuracy: 0.5070
 1312/25000 [>.............................] - ETA: 1:08 - loss: 7.6082 - accuracy: 0.5038
 1344/25000 [>.............................] - ETA: 1:07 - loss: 7.6324 - accuracy: 0.5022
 1376/25000 [>.............................] - ETA: 1:07 - loss: 7.5886 - accuracy: 0.5051
 1408/25000 [>.............................] - ETA: 1:07 - loss: 7.5686 - accuracy: 0.5064
 1440/25000 [>.............................] - ETA: 1:07 - loss: 7.5921 - accuracy: 0.5049
 1472/25000 [>.............................] - ETA: 1:07 - loss: 7.5625 - accuracy: 0.5068
 1504/25000 [>.............................] - ETA: 1:07 - loss: 7.5239 - accuracy: 0.5093
 1536/25000 [>.............................] - ETA: 1:06 - loss: 7.4869 - accuracy: 0.5117
 1568/25000 [>.............................] - ETA: 1:06 - loss: 7.4515 - accuracy: 0.5140
 1600/25000 [>.............................] - ETA: 1:06 - loss: 7.4750 - accuracy: 0.5125
 1632/25000 [>.............................] - ETA: 1:06 - loss: 7.5257 - accuracy: 0.5092
 1664/25000 [>.............................] - ETA: 1:06 - loss: 7.4639 - accuracy: 0.5132
 1696/25000 [=>............................] - ETA: 1:05 - loss: 7.4677 - accuracy: 0.5130
 1728/25000 [=>............................] - ETA: 1:05 - loss: 7.5069 - accuracy: 0.5104
 1760/25000 [=>............................] - ETA: 1:05 - loss: 7.4924 - accuracy: 0.5114
 1792/25000 [=>............................] - ETA: 1:05 - loss: 7.5212 - accuracy: 0.5095
 1824/25000 [=>............................] - ETA: 1:05 - loss: 7.5321 - accuracy: 0.5088
 1856/25000 [=>............................] - ETA: 1:05 - loss: 7.5262 - accuracy: 0.5092
 1888/25000 [=>............................] - ETA: 1:05 - loss: 7.5286 - accuracy: 0.5090
 1920/25000 [=>............................] - ETA: 1:04 - loss: 7.5149 - accuracy: 0.5099
 1952/25000 [=>............................] - ETA: 1:04 - loss: 7.5252 - accuracy: 0.5092
 1984/25000 [=>............................] - ETA: 1:04 - loss: 7.5739 - accuracy: 0.5060
 2016/25000 [=>............................] - ETA: 1:04 - loss: 7.5830 - accuracy: 0.5055
 2048/25000 [=>............................] - ETA: 1:04 - loss: 7.5918 - accuracy: 0.5049
 2080/25000 [=>............................] - ETA: 1:03 - loss: 7.6003 - accuracy: 0.5043
 2112/25000 [=>............................] - ETA: 1:03 - loss: 7.6448 - accuracy: 0.5014
 2144/25000 [=>............................] - ETA: 1:03 - loss: 7.6452 - accuracy: 0.5014
 2176/25000 [=>............................] - ETA: 1:03 - loss: 7.6596 - accuracy: 0.5005
 2208/25000 [=>............................] - ETA: 1:03 - loss: 7.6319 - accuracy: 0.5023
 2240/25000 [=>............................] - ETA: 1:03 - loss: 7.6324 - accuracy: 0.5022
 2272/25000 [=>............................] - ETA: 1:03 - loss: 7.6126 - accuracy: 0.5035
 2304/25000 [=>............................] - ETA: 1:03 - loss: 7.5801 - accuracy: 0.5056
 2336/25000 [=>............................] - ETA: 1:03 - loss: 7.5813 - accuracy: 0.5056
 2368/25000 [=>............................] - ETA: 1:02 - loss: 7.6019 - accuracy: 0.5042
 2400/25000 [=>............................] - ETA: 1:02 - loss: 7.6027 - accuracy: 0.5042
 2432/25000 [=>............................] - ETA: 1:02 - loss: 7.6225 - accuracy: 0.5029
 2464/25000 [=>............................] - ETA: 1:02 - loss: 7.6168 - accuracy: 0.5032
 2496/25000 [=>............................] - ETA: 1:02 - loss: 7.6236 - accuracy: 0.5028
 2528/25000 [==>...........................] - ETA: 1:02 - loss: 7.6181 - accuracy: 0.5032
 2560/25000 [==>...........................] - ETA: 1:02 - loss: 7.6127 - accuracy: 0.5035
 2592/25000 [==>...........................] - ETA: 1:02 - loss: 7.5956 - accuracy: 0.5046
 2624/25000 [==>...........................] - ETA: 1:01 - loss: 7.5965 - accuracy: 0.5046
 2656/25000 [==>...........................] - ETA: 1:01 - loss: 7.5973 - accuracy: 0.5045
 2688/25000 [==>...........................] - ETA: 1:01 - loss: 7.5925 - accuracy: 0.5048
 2720/25000 [==>...........................] - ETA: 1:01 - loss: 7.5821 - accuracy: 0.5055
 2752/25000 [==>...........................] - ETA: 1:01 - loss: 7.5608 - accuracy: 0.5069
 2784/25000 [==>...........................] - ETA: 1:01 - loss: 7.5675 - accuracy: 0.5065
 2816/25000 [==>...........................] - ETA: 1:01 - loss: 7.6013 - accuracy: 0.5043
 2848/25000 [==>...........................] - ETA: 1:01 - loss: 7.6128 - accuracy: 0.5035
 2880/25000 [==>...........................] - ETA: 1:01 - loss: 7.6134 - accuracy: 0.5035
 2912/25000 [==>...........................] - ETA: 1:01 - loss: 7.6192 - accuracy: 0.5031
 2944/25000 [==>...........................] - ETA: 1:00 - loss: 7.6354 - accuracy: 0.5020
 2976/25000 [==>...........................] - ETA: 1:00 - loss: 7.6357 - accuracy: 0.5020
 3008/25000 [==>...........................] - ETA: 1:00 - loss: 7.6207 - accuracy: 0.5030
 3040/25000 [==>...........................] - ETA: 1:00 - loss: 7.6313 - accuracy: 0.5023
 3072/25000 [==>...........................] - ETA: 1:00 - loss: 7.6367 - accuracy: 0.5020
 3104/25000 [==>...........................] - ETA: 1:00 - loss: 7.6123 - accuracy: 0.5035
 3136/25000 [==>...........................] - ETA: 1:00 - loss: 7.6275 - accuracy: 0.5026
 3168/25000 [==>...........................] - ETA: 1:00 - loss: 7.6327 - accuracy: 0.5022
 3200/25000 [==>...........................] - ETA: 1:00 - loss: 7.6475 - accuracy: 0.5013
 3232/25000 [==>...........................] - ETA: 1:00 - loss: 7.6619 - accuracy: 0.5003
 3264/25000 [==>...........................] - ETA: 59s - loss: 7.6666 - accuracy: 0.5000 
 3296/25000 [==>...........................] - ETA: 59s - loss: 7.6620 - accuracy: 0.5003
 3328/25000 [==>...........................] - ETA: 59s - loss: 7.6574 - accuracy: 0.5006
 3360/25000 [===>..........................] - ETA: 59s - loss: 7.6392 - accuracy: 0.5018
 3392/25000 [===>..........................] - ETA: 59s - loss: 7.6531 - accuracy: 0.5009
 3424/25000 [===>..........................] - ETA: 59s - loss: 7.6353 - accuracy: 0.5020
 3456/25000 [===>..........................] - ETA: 59s - loss: 7.6489 - accuracy: 0.5012
 3488/25000 [===>..........................] - ETA: 59s - loss: 7.6490 - accuracy: 0.5011
 3520/25000 [===>..........................] - ETA: 59s - loss: 7.6448 - accuracy: 0.5014
 3552/25000 [===>..........................] - ETA: 59s - loss: 7.6623 - accuracy: 0.5003
 3584/25000 [===>..........................] - ETA: 58s - loss: 7.6752 - accuracy: 0.4994
 3616/25000 [===>..........................] - ETA: 58s - loss: 7.6751 - accuracy: 0.4994
 3648/25000 [===>..........................] - ETA: 58s - loss: 7.6708 - accuracy: 0.4997
 3680/25000 [===>..........................] - ETA: 58s - loss: 7.6750 - accuracy: 0.4995
 3712/25000 [===>..........................] - ETA: 58s - loss: 7.6418 - accuracy: 0.5016
 3744/25000 [===>..........................] - ETA: 58s - loss: 7.6380 - accuracy: 0.5019
 3776/25000 [===>..........................] - ETA: 58s - loss: 7.6463 - accuracy: 0.5013
 3808/25000 [===>..........................] - ETA: 58s - loss: 7.6465 - accuracy: 0.5013
 3840/25000 [===>..........................] - ETA: 57s - loss: 7.6427 - accuracy: 0.5016
 3872/25000 [===>..........................] - ETA: 57s - loss: 7.6191 - accuracy: 0.5031
 3904/25000 [===>..........................] - ETA: 57s - loss: 7.6273 - accuracy: 0.5026
 3936/25000 [===>..........................] - ETA: 57s - loss: 7.6199 - accuracy: 0.5030
 3968/25000 [===>..........................] - ETA: 57s - loss: 7.6318 - accuracy: 0.5023
 4000/25000 [===>..........................] - ETA: 57s - loss: 7.6398 - accuracy: 0.5017
 4032/25000 [===>..........................] - ETA: 57s - loss: 7.6400 - accuracy: 0.5017
 4064/25000 [===>..........................] - ETA: 57s - loss: 7.6327 - accuracy: 0.5022
 4096/25000 [===>..........................] - ETA: 57s - loss: 7.6254 - accuracy: 0.5027
 4128/25000 [===>..........................] - ETA: 57s - loss: 7.6183 - accuracy: 0.5031
 4160/25000 [===>..........................] - ETA: 57s - loss: 7.6334 - accuracy: 0.5022
 4192/25000 [====>.........................] - ETA: 57s - loss: 7.6300 - accuracy: 0.5024
 4224/25000 [====>.........................] - ETA: 56s - loss: 7.6376 - accuracy: 0.5019
 4256/25000 [====>.........................] - ETA: 56s - loss: 7.6378 - accuracy: 0.5019
 4288/25000 [====>.........................] - ETA: 56s - loss: 7.6166 - accuracy: 0.5033
 4320/25000 [====>.........................] - ETA: 56s - loss: 7.6311 - accuracy: 0.5023
 4352/25000 [====>.........................] - ETA: 56s - loss: 7.6349 - accuracy: 0.5021
 4384/25000 [====>.........................] - ETA: 56s - loss: 7.6386 - accuracy: 0.5018
 4416/25000 [====>.........................] - ETA: 56s - loss: 7.6597 - accuracy: 0.5005
 4448/25000 [====>.........................] - ETA: 56s - loss: 7.6701 - accuracy: 0.4998
 4480/25000 [====>.........................] - ETA: 56s - loss: 7.6564 - accuracy: 0.5007
 4512/25000 [====>.........................] - ETA: 55s - loss: 7.6394 - accuracy: 0.5018
 4544/25000 [====>.........................] - ETA: 55s - loss: 7.6531 - accuracy: 0.5009
 4576/25000 [====>.........................] - ETA: 55s - loss: 7.6331 - accuracy: 0.5022
 4608/25000 [====>.........................] - ETA: 55s - loss: 7.6167 - accuracy: 0.5033
 4640/25000 [====>.........................] - ETA: 55s - loss: 7.6237 - accuracy: 0.5028
 4672/25000 [====>.........................] - ETA: 55s - loss: 7.6272 - accuracy: 0.5026
 4704/25000 [====>.........................] - ETA: 55s - loss: 7.6308 - accuracy: 0.5023
 4736/25000 [====>.........................] - ETA: 55s - loss: 7.6407 - accuracy: 0.5017
 4768/25000 [====>.........................] - ETA: 55s - loss: 7.6602 - accuracy: 0.5004
 4800/25000 [====>.........................] - ETA: 54s - loss: 7.6634 - accuracy: 0.5002
 4832/25000 [====>.........................] - ETA: 54s - loss: 7.6634 - accuracy: 0.5002
 4864/25000 [====>.........................] - ETA: 54s - loss: 7.6666 - accuracy: 0.5000
 4896/25000 [====>.........................] - ETA: 54s - loss: 7.6635 - accuracy: 0.5002
 4928/25000 [====>.........................] - ETA: 54s - loss: 7.6480 - accuracy: 0.5012
 4960/25000 [====>.........................] - ETA: 54s - loss: 7.6697 - accuracy: 0.4998
 4992/25000 [====>.........................] - ETA: 54s - loss: 7.6697 - accuracy: 0.4998
 5024/25000 [=====>........................] - ETA: 54s - loss: 7.6849 - accuracy: 0.4988
 5056/25000 [=====>........................] - ETA: 54s - loss: 7.6939 - accuracy: 0.4982
 5088/25000 [=====>........................] - ETA: 54s - loss: 7.6787 - accuracy: 0.4992
 5120/25000 [=====>........................] - ETA: 53s - loss: 7.6816 - accuracy: 0.4990
 5152/25000 [=====>........................] - ETA: 53s - loss: 7.6845 - accuracy: 0.4988
 5184/25000 [=====>........................] - ETA: 53s - loss: 7.6696 - accuracy: 0.4998
 5216/25000 [=====>........................] - ETA: 53s - loss: 7.6725 - accuracy: 0.4996
 5248/25000 [=====>........................] - ETA: 53s - loss: 7.6783 - accuracy: 0.4992
 5280/25000 [=====>........................] - ETA: 53s - loss: 7.6811 - accuracy: 0.4991
 5312/25000 [=====>........................] - ETA: 53s - loss: 7.6666 - accuracy: 0.5000
 5344/25000 [=====>........................] - ETA: 53s - loss: 7.6551 - accuracy: 0.5007
 5376/25000 [=====>........................] - ETA: 53s - loss: 7.6524 - accuracy: 0.5009
 5408/25000 [=====>........................] - ETA: 53s - loss: 7.6439 - accuracy: 0.5015
 5440/25000 [=====>........................] - ETA: 52s - loss: 7.6384 - accuracy: 0.5018
 5472/25000 [=====>........................] - ETA: 52s - loss: 7.6386 - accuracy: 0.5018
 5504/25000 [=====>........................] - ETA: 52s - loss: 7.6304 - accuracy: 0.5024
 5536/25000 [=====>........................] - ETA: 52s - loss: 7.6195 - accuracy: 0.5031
 5568/25000 [=====>........................] - ETA: 52s - loss: 7.6143 - accuracy: 0.5034
 5600/25000 [=====>........................] - ETA: 52s - loss: 7.6091 - accuracy: 0.5038
 5632/25000 [=====>........................] - ETA: 52s - loss: 7.6203 - accuracy: 0.5030
 5664/25000 [=====>........................] - ETA: 52s - loss: 7.6071 - accuracy: 0.5039
 5696/25000 [=====>........................] - ETA: 52s - loss: 7.6074 - accuracy: 0.5039
 5728/25000 [=====>........................] - ETA: 52s - loss: 7.6184 - accuracy: 0.5031
 5760/25000 [=====>........................] - ETA: 51s - loss: 7.6107 - accuracy: 0.5036
 5792/25000 [=====>........................] - ETA: 51s - loss: 7.6084 - accuracy: 0.5038
 5824/25000 [=====>........................] - ETA: 51s - loss: 7.5982 - accuracy: 0.5045
 5856/25000 [======>.......................] - ETA: 51s - loss: 7.6090 - accuracy: 0.5038
 5888/25000 [======>.......................] - ETA: 51s - loss: 7.6119 - accuracy: 0.5036
 5920/25000 [======>.......................] - ETA: 51s - loss: 7.6174 - accuracy: 0.5032
 5952/25000 [======>.......................] - ETA: 51s - loss: 7.6125 - accuracy: 0.5035
 5984/25000 [======>.......................] - ETA: 51s - loss: 7.6051 - accuracy: 0.5040
 6016/25000 [======>.......................] - ETA: 51s - loss: 7.6131 - accuracy: 0.5035
 6048/25000 [======>.......................] - ETA: 51s - loss: 7.6235 - accuracy: 0.5028
 6080/25000 [======>.......................] - ETA: 51s - loss: 7.6237 - accuracy: 0.5028
 6112/25000 [======>.......................] - ETA: 51s - loss: 7.6365 - accuracy: 0.5020
 6144/25000 [======>.......................] - ETA: 50s - loss: 7.6491 - accuracy: 0.5011
 6176/25000 [======>.......................] - ETA: 50s - loss: 7.6492 - accuracy: 0.5011
 6208/25000 [======>.......................] - ETA: 50s - loss: 7.6395 - accuracy: 0.5018
 6240/25000 [======>.......................] - ETA: 50s - loss: 7.6445 - accuracy: 0.5014
 6272/25000 [======>.......................] - ETA: 50s - loss: 7.6373 - accuracy: 0.5019
 6304/25000 [======>.......................] - ETA: 50s - loss: 7.6423 - accuracy: 0.5016
 6336/25000 [======>.......................] - ETA: 50s - loss: 7.6473 - accuracy: 0.5013
 6368/25000 [======>.......................] - ETA: 50s - loss: 7.6498 - accuracy: 0.5011
 6400/25000 [======>.......................] - ETA: 50s - loss: 7.6403 - accuracy: 0.5017
 6432/25000 [======>.......................] - ETA: 50s - loss: 7.6356 - accuracy: 0.5020
 6464/25000 [======>.......................] - ETA: 50s - loss: 7.6334 - accuracy: 0.5022
 6496/25000 [======>.......................] - ETA: 50s - loss: 7.6241 - accuracy: 0.5028
 6528/25000 [======>.......................] - ETA: 49s - loss: 7.6243 - accuracy: 0.5028
 6560/25000 [======>.......................] - ETA: 49s - loss: 7.6292 - accuracy: 0.5024
 6592/25000 [======>.......................] - ETA: 49s - loss: 7.6410 - accuracy: 0.5017
 6624/25000 [======>.......................] - ETA: 49s - loss: 7.6435 - accuracy: 0.5015
 6656/25000 [======>.......................] - ETA: 49s - loss: 7.6436 - accuracy: 0.5015
 6688/25000 [=======>......................] - ETA: 49s - loss: 7.6483 - accuracy: 0.5012
 6720/25000 [=======>......................] - ETA: 49s - loss: 7.6484 - accuracy: 0.5012
 6752/25000 [=======>......................] - ETA: 49s - loss: 7.6712 - accuracy: 0.4997
 6784/25000 [=======>......................] - ETA: 49s - loss: 7.6711 - accuracy: 0.4997
 6816/25000 [=======>......................] - ETA: 49s - loss: 7.6779 - accuracy: 0.4993
 6848/25000 [=======>......................] - ETA: 49s - loss: 7.6689 - accuracy: 0.4999
 6880/25000 [=======>......................] - ETA: 49s - loss: 7.6644 - accuracy: 0.5001
 6912/25000 [=======>......................] - ETA: 49s - loss: 7.6644 - accuracy: 0.5001
 6944/25000 [=======>......................] - ETA: 48s - loss: 7.6600 - accuracy: 0.5004
 6976/25000 [=======>......................] - ETA: 48s - loss: 7.6556 - accuracy: 0.5007
 7008/25000 [=======>......................] - ETA: 48s - loss: 7.6601 - accuracy: 0.5004
 7040/25000 [=======>......................] - ETA: 48s - loss: 7.6666 - accuracy: 0.5000
 7072/25000 [=======>......................] - ETA: 48s - loss: 7.6688 - accuracy: 0.4999
 7104/25000 [=======>......................] - ETA: 48s - loss: 7.6623 - accuracy: 0.5003
 7136/25000 [=======>......................] - ETA: 48s - loss: 7.6559 - accuracy: 0.5007
 7168/25000 [=======>......................] - ETA: 48s - loss: 7.6623 - accuracy: 0.5003
 7200/25000 [=======>......................] - ETA: 48s - loss: 7.6581 - accuracy: 0.5006
 7232/25000 [=======>......................] - ETA: 48s - loss: 7.6687 - accuracy: 0.4999
 7264/25000 [=======>......................] - ETA: 48s - loss: 7.6666 - accuracy: 0.5000
 7296/25000 [=======>......................] - ETA: 47s - loss: 7.6666 - accuracy: 0.5000
 7328/25000 [=======>......................] - ETA: 47s - loss: 7.6624 - accuracy: 0.5003
 7360/25000 [=======>......................] - ETA: 47s - loss: 7.6645 - accuracy: 0.5001
 7392/25000 [=======>......................] - ETA: 47s - loss: 7.6687 - accuracy: 0.4999
 7424/25000 [=======>......................] - ETA: 47s - loss: 7.6625 - accuracy: 0.5003
 7456/25000 [=======>......................] - ETA: 47s - loss: 7.6625 - accuracy: 0.5003
 7488/25000 [=======>......................] - ETA: 47s - loss: 7.6564 - accuracy: 0.5007
 7520/25000 [========>.....................] - ETA: 47s - loss: 7.6585 - accuracy: 0.5005
 7552/25000 [========>.....................] - ETA: 47s - loss: 7.6504 - accuracy: 0.5011
 7584/25000 [========>.....................] - ETA: 47s - loss: 7.6504 - accuracy: 0.5011
 7616/25000 [========>.....................] - ETA: 47s - loss: 7.6505 - accuracy: 0.5011
 7648/25000 [========>.....................] - ETA: 47s - loss: 7.6586 - accuracy: 0.5005
 7680/25000 [========>.....................] - ETA: 46s - loss: 7.6586 - accuracy: 0.5005
 7712/25000 [========>.....................] - ETA: 46s - loss: 7.6607 - accuracy: 0.5004
 7744/25000 [========>.....................] - ETA: 46s - loss: 7.6508 - accuracy: 0.5010
 7776/25000 [========>.....................] - ETA: 46s - loss: 7.6568 - accuracy: 0.5006
 7808/25000 [========>.....................] - ETA: 46s - loss: 7.6529 - accuracy: 0.5009
 7840/25000 [========>.....................] - ETA: 46s - loss: 7.6588 - accuracy: 0.5005
 7872/25000 [========>.....................] - ETA: 46s - loss: 7.6569 - accuracy: 0.5006
 7904/25000 [========>.....................] - ETA: 46s - loss: 7.6492 - accuracy: 0.5011
 7936/25000 [========>.....................] - ETA: 46s - loss: 7.6570 - accuracy: 0.5006
 7968/25000 [========>.....................] - ETA: 46s - loss: 7.6647 - accuracy: 0.5001
 8000/25000 [========>.....................] - ETA: 46s - loss: 7.6666 - accuracy: 0.5000
 8032/25000 [========>.....................] - ETA: 45s - loss: 7.6647 - accuracy: 0.5001
 8064/25000 [========>.....................] - ETA: 45s - loss: 7.6666 - accuracy: 0.5000
 8096/25000 [========>.....................] - ETA: 45s - loss: 7.6742 - accuracy: 0.4995
 8128/25000 [========>.....................] - ETA: 45s - loss: 7.6723 - accuracy: 0.4996
 8160/25000 [========>.....................] - ETA: 45s - loss: 7.6704 - accuracy: 0.4998
 8192/25000 [========>.....................] - ETA: 45s - loss: 7.6704 - accuracy: 0.4998
 8224/25000 [========>.....................] - ETA: 45s - loss: 7.6741 - accuracy: 0.4995
 8256/25000 [========>.....................] - ETA: 45s - loss: 7.6629 - accuracy: 0.5002
 8288/25000 [========>.....................] - ETA: 45s - loss: 7.6574 - accuracy: 0.5006
 8320/25000 [========>.....................] - ETA: 45s - loss: 7.6519 - accuracy: 0.5010
 8352/25000 [=========>....................] - ETA: 45s - loss: 7.6556 - accuracy: 0.5007
 8384/25000 [=========>....................] - ETA: 45s - loss: 7.6575 - accuracy: 0.5006
 8416/25000 [=========>....................] - ETA: 44s - loss: 7.6557 - accuracy: 0.5007
 8448/25000 [=========>....................] - ETA: 44s - loss: 7.6594 - accuracy: 0.5005
 8480/25000 [=========>....................] - ETA: 44s - loss: 7.6630 - accuracy: 0.5002
 8512/25000 [=========>....................] - ETA: 44s - loss: 7.6576 - accuracy: 0.5006
 8544/25000 [=========>....................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 8576/25000 [=========>....................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 8608/25000 [=========>....................] - ETA: 44s - loss: 7.6702 - accuracy: 0.4998
 8640/25000 [=========>....................] - ETA: 44s - loss: 7.6719 - accuracy: 0.4997
 8672/25000 [=========>....................] - ETA: 44s - loss: 7.6737 - accuracy: 0.4995
 8704/25000 [=========>....................] - ETA: 44s - loss: 7.6719 - accuracy: 0.4997
 8736/25000 [=========>....................] - ETA: 43s - loss: 7.6701 - accuracy: 0.4998
 8768/25000 [=========>....................] - ETA: 43s - loss: 7.6684 - accuracy: 0.4999
 8800/25000 [=========>....................] - ETA: 43s - loss: 7.6736 - accuracy: 0.4995
 8832/25000 [=========>....................] - ETA: 43s - loss: 7.6770 - accuracy: 0.4993
 8864/25000 [=========>....................] - ETA: 43s - loss: 7.6805 - accuracy: 0.4991
 8896/25000 [=========>....................] - ETA: 43s - loss: 7.6787 - accuracy: 0.4992
 8928/25000 [=========>....................] - ETA: 43s - loss: 7.6804 - accuracy: 0.4991
 8960/25000 [=========>....................] - ETA: 43s - loss: 7.6786 - accuracy: 0.4992
 8992/25000 [=========>....................] - ETA: 43s - loss: 7.6734 - accuracy: 0.4996
 9024/25000 [=========>....................] - ETA: 43s - loss: 7.6683 - accuracy: 0.4999
 9056/25000 [=========>....................] - ETA: 43s - loss: 7.6768 - accuracy: 0.4993
 9088/25000 [=========>....................] - ETA: 42s - loss: 7.6784 - accuracy: 0.4992
 9120/25000 [=========>....................] - ETA: 42s - loss: 7.6750 - accuracy: 0.4995
 9152/25000 [=========>....................] - ETA: 42s - loss: 7.6750 - accuracy: 0.4995
 9184/25000 [==========>...................] - ETA: 42s - loss: 7.6783 - accuracy: 0.4992
 9216/25000 [==========>...................] - ETA: 42s - loss: 7.6816 - accuracy: 0.4990
 9248/25000 [==========>...................] - ETA: 42s - loss: 7.6815 - accuracy: 0.4990
 9280/25000 [==========>...................] - ETA: 42s - loss: 7.6831 - accuracy: 0.4989
 9312/25000 [==========>...................] - ETA: 42s - loss: 7.6880 - accuracy: 0.4986
 9344/25000 [==========>...................] - ETA: 42s - loss: 7.6962 - accuracy: 0.4981
 9376/25000 [==========>...................] - ETA: 42s - loss: 7.7010 - accuracy: 0.4978
 9408/25000 [==========>...................] - ETA: 42s - loss: 7.6976 - accuracy: 0.4980
 9440/25000 [==========>...................] - ETA: 41s - loss: 7.6991 - accuracy: 0.4979
 9472/25000 [==========>...................] - ETA: 41s - loss: 7.6941 - accuracy: 0.4982
 9504/25000 [==========>...................] - ETA: 41s - loss: 7.6957 - accuracy: 0.4981
 9536/25000 [==========>...................] - ETA: 41s - loss: 7.6907 - accuracy: 0.4984
 9568/25000 [==========>...................] - ETA: 41s - loss: 7.6875 - accuracy: 0.4986
 9600/25000 [==========>...................] - ETA: 41s - loss: 7.6890 - accuracy: 0.4985
 9632/25000 [==========>...................] - ETA: 41s - loss: 7.6905 - accuracy: 0.4984
 9664/25000 [==========>...................] - ETA: 41s - loss: 7.6952 - accuracy: 0.4981
 9696/25000 [==========>...................] - ETA: 41s - loss: 7.6982 - accuracy: 0.4979
 9728/25000 [==========>...................] - ETA: 41s - loss: 7.6981 - accuracy: 0.4979
 9760/25000 [==========>...................] - ETA: 41s - loss: 7.6949 - accuracy: 0.4982
 9792/25000 [==========>...................] - ETA: 41s - loss: 7.6964 - accuracy: 0.4981
 9824/25000 [==========>...................] - ETA: 40s - loss: 7.6885 - accuracy: 0.4986
 9856/25000 [==========>...................] - ETA: 40s - loss: 7.6837 - accuracy: 0.4989
 9888/25000 [==========>...................] - ETA: 40s - loss: 7.6883 - accuracy: 0.4986
 9920/25000 [==========>...................] - ETA: 40s - loss: 7.6898 - accuracy: 0.4985
 9952/25000 [==========>...................] - ETA: 40s - loss: 7.6866 - accuracy: 0.4987
 9984/25000 [==========>...................] - ETA: 40s - loss: 7.6850 - accuracy: 0.4988
10016/25000 [===========>..................] - ETA: 40s - loss: 7.6819 - accuracy: 0.4990
10048/25000 [===========>..................] - ETA: 40s - loss: 7.6712 - accuracy: 0.4997
10080/25000 [===========>..................] - ETA: 40s - loss: 7.6681 - accuracy: 0.4999
10112/25000 [===========>..................] - ETA: 40s - loss: 7.6636 - accuracy: 0.5002
10144/25000 [===========>..................] - ETA: 40s - loss: 7.6681 - accuracy: 0.4999
10176/25000 [===========>..................] - ETA: 40s - loss: 7.6621 - accuracy: 0.5003
10208/25000 [===========>..................] - ETA: 39s - loss: 7.6606 - accuracy: 0.5004
10240/25000 [===========>..................] - ETA: 39s - loss: 7.6546 - accuracy: 0.5008
10272/25000 [===========>..................] - ETA: 39s - loss: 7.6502 - accuracy: 0.5011
10304/25000 [===========>..................] - ETA: 39s - loss: 7.6503 - accuracy: 0.5011
10336/25000 [===========>..................] - ETA: 39s - loss: 7.6548 - accuracy: 0.5008
10368/25000 [===========>..................] - ETA: 39s - loss: 7.6533 - accuracy: 0.5009
10400/25000 [===========>..................] - ETA: 39s - loss: 7.6563 - accuracy: 0.5007
10432/25000 [===========>..................] - ETA: 39s - loss: 7.6578 - accuracy: 0.5006
10464/25000 [===========>..................] - ETA: 39s - loss: 7.6534 - accuracy: 0.5009
10496/25000 [===========>..................] - ETA: 39s - loss: 7.6535 - accuracy: 0.5009
10528/25000 [===========>..................] - ETA: 39s - loss: 7.6477 - accuracy: 0.5012
10560/25000 [===========>..................] - ETA: 38s - loss: 7.6463 - accuracy: 0.5013
10592/25000 [===========>..................] - ETA: 38s - loss: 7.6478 - accuracy: 0.5012
10624/25000 [===========>..................] - ETA: 38s - loss: 7.6479 - accuracy: 0.5012
10656/25000 [===========>..................] - ETA: 38s - loss: 7.6479 - accuracy: 0.5012
10688/25000 [===========>..................] - ETA: 38s - loss: 7.6480 - accuracy: 0.5012
10720/25000 [===========>..................] - ETA: 38s - loss: 7.6480 - accuracy: 0.5012
10752/25000 [===========>..................] - ETA: 38s - loss: 7.6452 - accuracy: 0.5014
10784/25000 [===========>..................] - ETA: 38s - loss: 7.6453 - accuracy: 0.5014
10816/25000 [===========>..................] - ETA: 38s - loss: 7.6383 - accuracy: 0.5018
10848/25000 [============>.................] - ETA: 38s - loss: 7.6299 - accuracy: 0.5024
10880/25000 [============>.................] - ETA: 38s - loss: 7.6300 - accuracy: 0.5024
10912/25000 [============>.................] - ETA: 37s - loss: 7.6259 - accuracy: 0.5027
10944/25000 [============>.................] - ETA: 37s - loss: 7.6246 - accuracy: 0.5027
10976/25000 [============>.................] - ETA: 37s - loss: 7.6303 - accuracy: 0.5024
11008/25000 [============>.................] - ETA: 37s - loss: 7.6248 - accuracy: 0.5027
11040/25000 [============>.................] - ETA: 37s - loss: 7.6277 - accuracy: 0.5025
11072/25000 [============>.................] - ETA: 37s - loss: 7.6292 - accuracy: 0.5024
11104/25000 [============>.................] - ETA: 37s - loss: 7.6321 - accuracy: 0.5023
11136/25000 [============>.................] - ETA: 37s - loss: 7.6308 - accuracy: 0.5023
11168/25000 [============>.................] - ETA: 37s - loss: 7.6268 - accuracy: 0.5026
11200/25000 [============>.................] - ETA: 37s - loss: 7.6255 - accuracy: 0.5027
11232/25000 [============>.................] - ETA: 37s - loss: 7.6284 - accuracy: 0.5025
11264/25000 [============>.................] - ETA: 37s - loss: 7.6367 - accuracy: 0.5020
11296/25000 [============>.................] - ETA: 36s - loss: 7.6368 - accuracy: 0.5019
11328/25000 [============>.................] - ETA: 36s - loss: 7.6355 - accuracy: 0.5020
11360/25000 [============>.................] - ETA: 36s - loss: 7.6329 - accuracy: 0.5022
11392/25000 [============>.................] - ETA: 36s - loss: 7.6370 - accuracy: 0.5019
11424/25000 [============>.................] - ETA: 36s - loss: 7.6357 - accuracy: 0.5020
11456/25000 [============>.................] - ETA: 36s - loss: 7.6318 - accuracy: 0.5023
11488/25000 [============>.................] - ETA: 36s - loss: 7.6359 - accuracy: 0.5020
11520/25000 [============>.................] - ETA: 36s - loss: 7.6373 - accuracy: 0.5019
11552/25000 [============>.................] - ETA: 36s - loss: 7.6401 - accuracy: 0.5017
11584/25000 [============>.................] - ETA: 36s - loss: 7.6362 - accuracy: 0.5020
11616/25000 [============>.................] - ETA: 36s - loss: 7.6429 - accuracy: 0.5015
11648/25000 [============>.................] - ETA: 35s - loss: 7.6416 - accuracy: 0.5016
11680/25000 [=============>................] - ETA: 35s - loss: 7.6404 - accuracy: 0.5017
11712/25000 [=============>................] - ETA: 35s - loss: 7.6431 - accuracy: 0.5015
11744/25000 [=============>................] - ETA: 35s - loss: 7.6457 - accuracy: 0.5014
11776/25000 [=============>................] - ETA: 35s - loss: 7.6497 - accuracy: 0.5011
11808/25000 [=============>................] - ETA: 35s - loss: 7.6510 - accuracy: 0.5010
11840/25000 [=============>................] - ETA: 35s - loss: 7.6537 - accuracy: 0.5008
11872/25000 [=============>................] - ETA: 35s - loss: 7.6589 - accuracy: 0.5005
11904/25000 [=============>................] - ETA: 35s - loss: 7.6589 - accuracy: 0.5005
11936/25000 [=============>................] - ETA: 35s - loss: 7.6628 - accuracy: 0.5003
11968/25000 [=============>................] - ETA: 35s - loss: 7.6666 - accuracy: 0.5000
12000/25000 [=============>................] - ETA: 35s - loss: 7.6743 - accuracy: 0.4995
12032/25000 [=============>................] - ETA: 34s - loss: 7.6755 - accuracy: 0.4994
12064/25000 [=============>................] - ETA: 34s - loss: 7.6793 - accuracy: 0.4992
12096/25000 [=============>................] - ETA: 34s - loss: 7.6755 - accuracy: 0.4994
12128/25000 [=============>................] - ETA: 34s - loss: 7.6742 - accuracy: 0.4995
12160/25000 [=============>................] - ETA: 34s - loss: 7.6717 - accuracy: 0.4997
12192/25000 [=============>................] - ETA: 34s - loss: 7.6654 - accuracy: 0.5001
12224/25000 [=============>................] - ETA: 34s - loss: 7.6679 - accuracy: 0.4999
12256/25000 [=============>................] - ETA: 34s - loss: 7.6716 - accuracy: 0.4997
12288/25000 [=============>................] - ETA: 34s - loss: 7.6679 - accuracy: 0.4999
12320/25000 [=============>................] - ETA: 34s - loss: 7.6741 - accuracy: 0.4995
12352/25000 [=============>................] - ETA: 34s - loss: 7.6679 - accuracy: 0.4999
12384/25000 [=============>................] - ETA: 33s - loss: 7.6654 - accuracy: 0.5001
12416/25000 [=============>................] - ETA: 33s - loss: 7.6654 - accuracy: 0.5001
12448/25000 [=============>................] - ETA: 33s - loss: 7.6654 - accuracy: 0.5001
12480/25000 [=============>................] - ETA: 33s - loss: 7.6654 - accuracy: 0.5001
12512/25000 [==============>...............] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
12544/25000 [==============>...............] - ETA: 33s - loss: 7.6617 - accuracy: 0.5003
12576/25000 [==============>...............] - ETA: 33s - loss: 7.6630 - accuracy: 0.5002
12608/25000 [==============>...............] - ETA: 33s - loss: 7.6569 - accuracy: 0.5006
12640/25000 [==============>...............] - ETA: 33s - loss: 7.6630 - accuracy: 0.5002
12672/25000 [==============>...............] - ETA: 33s - loss: 7.6618 - accuracy: 0.5003
12704/25000 [==============>...............] - ETA: 33s - loss: 7.6618 - accuracy: 0.5003
12736/25000 [==============>...............] - ETA: 33s - loss: 7.6642 - accuracy: 0.5002
12768/25000 [==============>...............] - ETA: 32s - loss: 7.6642 - accuracy: 0.5002
12800/25000 [==============>...............] - ETA: 32s - loss: 7.6642 - accuracy: 0.5002
12832/25000 [==============>...............] - ETA: 32s - loss: 7.6630 - accuracy: 0.5002
12864/25000 [==============>...............] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
12896/25000 [==============>...............] - ETA: 32s - loss: 7.6690 - accuracy: 0.4998
12928/25000 [==============>...............] - ETA: 32s - loss: 7.6678 - accuracy: 0.4999
12960/25000 [==============>...............] - ETA: 32s - loss: 7.6690 - accuracy: 0.4998
12992/25000 [==============>...............] - ETA: 32s - loss: 7.6713 - accuracy: 0.4997
13024/25000 [==============>...............] - ETA: 32s - loss: 7.6725 - accuracy: 0.4996
13056/25000 [==============>...............] - ETA: 32s - loss: 7.6748 - accuracy: 0.4995
13088/25000 [==============>...............] - ETA: 32s - loss: 7.6736 - accuracy: 0.4995
13120/25000 [==============>...............] - ETA: 31s - loss: 7.6760 - accuracy: 0.4994
13152/25000 [==============>...............] - ETA: 31s - loss: 7.6818 - accuracy: 0.4990
13184/25000 [==============>...............] - ETA: 31s - loss: 7.6817 - accuracy: 0.4990
13216/25000 [==============>...............] - ETA: 31s - loss: 7.6910 - accuracy: 0.4984
13248/25000 [==============>...............] - ETA: 31s - loss: 7.6921 - accuracy: 0.4983
13280/25000 [==============>...............] - ETA: 31s - loss: 7.6920 - accuracy: 0.4983
13312/25000 [==============>...............] - ETA: 31s - loss: 7.6943 - accuracy: 0.4982
13344/25000 [===============>..............] - ETA: 31s - loss: 7.6930 - accuracy: 0.4983
13376/25000 [===============>..............] - ETA: 31s - loss: 7.6930 - accuracy: 0.4983
13408/25000 [===============>..............] - ETA: 31s - loss: 7.6986 - accuracy: 0.4979
13440/25000 [===============>..............] - ETA: 31s - loss: 7.6963 - accuracy: 0.4981
13472/25000 [===============>..............] - ETA: 30s - loss: 7.6962 - accuracy: 0.4981
13504/25000 [===============>..............] - ETA: 30s - loss: 7.6995 - accuracy: 0.4979
13536/25000 [===============>..............] - ETA: 30s - loss: 7.6949 - accuracy: 0.4982
13568/25000 [===============>..............] - ETA: 30s - loss: 7.6960 - accuracy: 0.4981
13600/25000 [===============>..............] - ETA: 30s - loss: 7.6903 - accuracy: 0.4985
13632/25000 [===============>..............] - ETA: 30s - loss: 7.6880 - accuracy: 0.4986
13664/25000 [===============>..............] - ETA: 30s - loss: 7.6846 - accuracy: 0.4988
13696/25000 [===============>..............] - ETA: 30s - loss: 7.6823 - accuracy: 0.4990
13728/25000 [===============>..............] - ETA: 30s - loss: 7.6856 - accuracy: 0.4988
13760/25000 [===============>..............] - ETA: 30s - loss: 7.6889 - accuracy: 0.4985
13792/25000 [===============>..............] - ETA: 30s - loss: 7.6900 - accuracy: 0.4985
13824/25000 [===============>..............] - ETA: 30s - loss: 7.6866 - accuracy: 0.4987
13856/25000 [===============>..............] - ETA: 29s - loss: 7.6876 - accuracy: 0.4986
13888/25000 [===============>..............] - ETA: 29s - loss: 7.6898 - accuracy: 0.4985
13920/25000 [===============>..............] - ETA: 29s - loss: 7.6853 - accuracy: 0.4988
13952/25000 [===============>..............] - ETA: 29s - loss: 7.6864 - accuracy: 0.4987
13984/25000 [===============>..............] - ETA: 29s - loss: 7.6875 - accuracy: 0.4986
14016/25000 [===============>..............] - ETA: 29s - loss: 7.6863 - accuracy: 0.4987
14048/25000 [===============>..............] - ETA: 29s - loss: 7.6852 - accuracy: 0.4988
14080/25000 [===============>..............] - ETA: 29s - loss: 7.6797 - accuracy: 0.4991
14112/25000 [===============>..............] - ETA: 29s - loss: 7.6775 - accuracy: 0.4993
14144/25000 [===============>..............] - ETA: 29s - loss: 7.6785 - accuracy: 0.4992
14176/25000 [================>.............] - ETA: 29s - loss: 7.6774 - accuracy: 0.4993
14208/25000 [================>.............] - ETA: 29s - loss: 7.6774 - accuracy: 0.4993
14240/25000 [================>.............] - ETA: 28s - loss: 7.6785 - accuracy: 0.4992
14272/25000 [================>.............] - ETA: 28s - loss: 7.6774 - accuracy: 0.4993
14304/25000 [================>.............] - ETA: 28s - loss: 7.6795 - accuracy: 0.4992
14336/25000 [================>.............] - ETA: 28s - loss: 7.6773 - accuracy: 0.4993
14368/25000 [================>.............] - ETA: 28s - loss: 7.6794 - accuracy: 0.4992
14400/25000 [================>.............] - ETA: 28s - loss: 7.6751 - accuracy: 0.4994
14432/25000 [================>.............] - ETA: 28s - loss: 7.6751 - accuracy: 0.4994
14464/25000 [================>.............] - ETA: 28s - loss: 7.6751 - accuracy: 0.4994
14496/25000 [================>.............] - ETA: 28s - loss: 7.6761 - accuracy: 0.4994
14528/25000 [================>.............] - ETA: 28s - loss: 7.6740 - accuracy: 0.4995
14560/25000 [================>.............] - ETA: 28s - loss: 7.6708 - accuracy: 0.4997
14592/25000 [================>.............] - ETA: 27s - loss: 7.6750 - accuracy: 0.4995
14624/25000 [================>.............] - ETA: 27s - loss: 7.6740 - accuracy: 0.4995
14656/25000 [================>.............] - ETA: 27s - loss: 7.6750 - accuracy: 0.4995
14688/25000 [================>.............] - ETA: 27s - loss: 7.6718 - accuracy: 0.4997
14720/25000 [================>.............] - ETA: 27s - loss: 7.6708 - accuracy: 0.4997
14752/25000 [================>.............] - ETA: 27s - loss: 7.6687 - accuracy: 0.4999
14784/25000 [================>.............] - ETA: 27s - loss: 7.6687 - accuracy: 0.4999
14816/25000 [================>.............] - ETA: 27s - loss: 7.6677 - accuracy: 0.4999
14848/25000 [================>.............] - ETA: 27s - loss: 7.6615 - accuracy: 0.5003
14880/25000 [================>.............] - ETA: 27s - loss: 7.6573 - accuracy: 0.5006
14912/25000 [================>.............] - ETA: 27s - loss: 7.6533 - accuracy: 0.5009
14944/25000 [================>.............] - ETA: 27s - loss: 7.6543 - accuracy: 0.5008
14976/25000 [================>.............] - ETA: 26s - loss: 7.6533 - accuracy: 0.5009
15008/25000 [=================>............] - ETA: 26s - loss: 7.6574 - accuracy: 0.5006
15040/25000 [=================>............] - ETA: 26s - loss: 7.6564 - accuracy: 0.5007
15072/25000 [=================>............] - ETA: 26s - loss: 7.6554 - accuracy: 0.5007
15104/25000 [=================>............] - ETA: 26s - loss: 7.6585 - accuracy: 0.5005
15136/25000 [=================>............] - ETA: 26s - loss: 7.6575 - accuracy: 0.5006
15168/25000 [=================>............] - ETA: 26s - loss: 7.6585 - accuracy: 0.5005
15200/25000 [=================>............] - ETA: 26s - loss: 7.6606 - accuracy: 0.5004
15232/25000 [=================>............] - ETA: 26s - loss: 7.6576 - accuracy: 0.5006
15264/25000 [=================>............] - ETA: 26s - loss: 7.6546 - accuracy: 0.5008
15296/25000 [=================>............] - ETA: 26s - loss: 7.6536 - accuracy: 0.5008
15328/25000 [=================>............] - ETA: 26s - loss: 7.6526 - accuracy: 0.5009
15360/25000 [=================>............] - ETA: 25s - loss: 7.6516 - accuracy: 0.5010
15392/25000 [=================>............] - ETA: 25s - loss: 7.6497 - accuracy: 0.5011
15424/25000 [=================>............] - ETA: 25s - loss: 7.6517 - accuracy: 0.5010
15456/25000 [=================>............] - ETA: 25s - loss: 7.6517 - accuracy: 0.5010
15488/25000 [=================>............] - ETA: 25s - loss: 7.6508 - accuracy: 0.5010
15520/25000 [=================>............] - ETA: 25s - loss: 7.6518 - accuracy: 0.5010
15552/25000 [=================>............] - ETA: 25s - loss: 7.6558 - accuracy: 0.5007
15584/25000 [=================>............] - ETA: 25s - loss: 7.6568 - accuracy: 0.5006
15616/25000 [=================>............] - ETA: 25s - loss: 7.6607 - accuracy: 0.5004
15648/25000 [=================>............] - ETA: 25s - loss: 7.6588 - accuracy: 0.5005
15680/25000 [=================>............] - ETA: 25s - loss: 7.6549 - accuracy: 0.5008
15712/25000 [=================>............] - ETA: 24s - loss: 7.6539 - accuracy: 0.5008
15744/25000 [=================>............] - ETA: 24s - loss: 7.6559 - accuracy: 0.5007
15776/25000 [=================>............] - ETA: 24s - loss: 7.6550 - accuracy: 0.5008
15808/25000 [=================>............] - ETA: 24s - loss: 7.6550 - accuracy: 0.5008
15840/25000 [==================>...........] - ETA: 24s - loss: 7.6550 - accuracy: 0.5008
15872/25000 [==================>...........] - ETA: 24s - loss: 7.6550 - accuracy: 0.5008
15904/25000 [==================>...........] - ETA: 24s - loss: 7.6599 - accuracy: 0.5004
15936/25000 [==================>...........] - ETA: 24s - loss: 7.6608 - accuracy: 0.5004
15968/25000 [==================>...........] - ETA: 24s - loss: 7.6609 - accuracy: 0.5004
16000/25000 [==================>...........] - ETA: 24s - loss: 7.6628 - accuracy: 0.5002
16032/25000 [==================>...........] - ETA: 24s - loss: 7.6676 - accuracy: 0.4999
16064/25000 [==================>...........] - ETA: 24s - loss: 7.6666 - accuracy: 0.5000
16096/25000 [==================>...........] - ETA: 23s - loss: 7.6676 - accuracy: 0.4999
16128/25000 [==================>...........] - ETA: 23s - loss: 7.6695 - accuracy: 0.4998
16160/25000 [==================>...........] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
16192/25000 [==================>...........] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
16224/25000 [==================>...........] - ETA: 23s - loss: 7.6676 - accuracy: 0.4999
16256/25000 [==================>...........] - ETA: 23s - loss: 7.6685 - accuracy: 0.4999
16288/25000 [==================>...........] - ETA: 23s - loss: 7.6713 - accuracy: 0.4997
16320/25000 [==================>...........] - ETA: 23s - loss: 7.6694 - accuracy: 0.4998
16352/25000 [==================>...........] - ETA: 23s - loss: 7.6676 - accuracy: 0.4999
16384/25000 [==================>...........] - ETA: 23s - loss: 7.6638 - accuracy: 0.5002
16416/25000 [==================>...........] - ETA: 23s - loss: 7.6648 - accuracy: 0.5001
16448/25000 [==================>...........] - ETA: 22s - loss: 7.6629 - accuracy: 0.5002
16480/25000 [==================>...........] - ETA: 22s - loss: 7.6601 - accuracy: 0.5004
16512/25000 [==================>...........] - ETA: 22s - loss: 7.6610 - accuracy: 0.5004
16544/25000 [==================>...........] - ETA: 22s - loss: 7.6620 - accuracy: 0.5003
16576/25000 [==================>...........] - ETA: 22s - loss: 7.6601 - accuracy: 0.5004
16608/25000 [==================>...........] - ETA: 22s - loss: 7.6620 - accuracy: 0.5003
16640/25000 [==================>...........] - ETA: 22s - loss: 7.6620 - accuracy: 0.5003
16672/25000 [===================>..........] - ETA: 22s - loss: 7.6639 - accuracy: 0.5002
16704/25000 [===================>..........] - ETA: 22s - loss: 7.6648 - accuracy: 0.5001
16736/25000 [===================>..........] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
16768/25000 [===================>..........] - ETA: 22s - loss: 7.6657 - accuracy: 0.5001
16800/25000 [===================>..........] - ETA: 21s - loss: 7.6657 - accuracy: 0.5001
16832/25000 [===================>..........] - ETA: 21s - loss: 7.6648 - accuracy: 0.5001
16864/25000 [===================>..........] - ETA: 21s - loss: 7.6666 - accuracy: 0.5000
16896/25000 [===================>..........] - ETA: 21s - loss: 7.6639 - accuracy: 0.5002
16928/25000 [===================>..........] - ETA: 21s - loss: 7.6648 - accuracy: 0.5001
16960/25000 [===================>..........] - ETA: 21s - loss: 7.6684 - accuracy: 0.4999
16992/25000 [===================>..........] - ETA: 21s - loss: 7.6720 - accuracy: 0.4996
17024/25000 [===================>..........] - ETA: 21s - loss: 7.6738 - accuracy: 0.4995
17056/25000 [===================>..........] - ETA: 21s - loss: 7.6720 - accuracy: 0.4996
17088/25000 [===================>..........] - ETA: 21s - loss: 7.6747 - accuracy: 0.4995
17120/25000 [===================>..........] - ETA: 21s - loss: 7.6738 - accuracy: 0.4995
17152/25000 [===================>..........] - ETA: 21s - loss: 7.6756 - accuracy: 0.4994
17184/25000 [===================>..........] - ETA: 20s - loss: 7.6755 - accuracy: 0.4994
17216/25000 [===================>..........] - ETA: 20s - loss: 7.6809 - accuracy: 0.4991
17248/25000 [===================>..........] - ETA: 20s - loss: 7.6782 - accuracy: 0.4992
17280/25000 [===================>..........] - ETA: 20s - loss: 7.6773 - accuracy: 0.4993
17312/25000 [===================>..........] - ETA: 20s - loss: 7.6764 - accuracy: 0.4994
17344/25000 [===================>..........] - ETA: 20s - loss: 7.6746 - accuracy: 0.4995
17376/25000 [===================>..........] - ETA: 20s - loss: 7.6754 - accuracy: 0.4994
17408/25000 [===================>..........] - ETA: 20s - loss: 7.6790 - accuracy: 0.4992
17440/25000 [===================>..........] - ETA: 20s - loss: 7.6798 - accuracy: 0.4991
17472/25000 [===================>..........] - ETA: 20s - loss: 7.6789 - accuracy: 0.4992
17504/25000 [====================>.........] - ETA: 20s - loss: 7.6754 - accuracy: 0.4994
17536/25000 [====================>.........] - ETA: 19s - loss: 7.6771 - accuracy: 0.4993
17568/25000 [====================>.........] - ETA: 19s - loss: 7.6806 - accuracy: 0.4991
17600/25000 [====================>.........] - ETA: 19s - loss: 7.6788 - accuracy: 0.4992
17632/25000 [====================>.........] - ETA: 19s - loss: 7.6779 - accuracy: 0.4993
17664/25000 [====================>.........] - ETA: 19s - loss: 7.6788 - accuracy: 0.4992
17696/25000 [====================>.........] - ETA: 19s - loss: 7.6796 - accuracy: 0.4992
17728/25000 [====================>.........] - ETA: 19s - loss: 7.6796 - accuracy: 0.4992
17760/25000 [====================>.........] - ETA: 19s - loss: 7.6804 - accuracy: 0.4991
17792/25000 [====================>.........] - ETA: 19s - loss: 7.6778 - accuracy: 0.4993
17824/25000 [====================>.........] - ETA: 19s - loss: 7.6795 - accuracy: 0.4992
17856/25000 [====================>.........] - ETA: 19s - loss: 7.6821 - accuracy: 0.4990
17888/25000 [====================>.........] - ETA: 19s - loss: 7.6812 - accuracy: 0.4990
17920/25000 [====================>.........] - ETA: 18s - loss: 7.6795 - accuracy: 0.4992
17952/25000 [====================>.........] - ETA: 18s - loss: 7.6794 - accuracy: 0.4992
17984/25000 [====================>.........] - ETA: 18s - loss: 7.6837 - accuracy: 0.4989
18016/25000 [====================>.........] - ETA: 18s - loss: 7.6853 - accuracy: 0.4988
18048/25000 [====================>.........] - ETA: 18s - loss: 7.6836 - accuracy: 0.4989
18080/25000 [====================>.........] - ETA: 18s - loss: 7.6878 - accuracy: 0.4986
18112/25000 [====================>.........] - ETA: 18s - loss: 7.6869 - accuracy: 0.4987
18144/25000 [====================>.........] - ETA: 18s - loss: 7.6877 - accuracy: 0.4986
18176/25000 [====================>.........] - ETA: 18s - loss: 7.6877 - accuracy: 0.4986
18208/25000 [====================>.........] - ETA: 18s - loss: 7.6885 - accuracy: 0.4986
18240/25000 [====================>.........] - ETA: 18s - loss: 7.6876 - accuracy: 0.4986
18272/25000 [====================>.........] - ETA: 18s - loss: 7.6884 - accuracy: 0.4986
18304/25000 [====================>.........] - ETA: 17s - loss: 7.6901 - accuracy: 0.4985
18336/25000 [=====================>........] - ETA: 17s - loss: 7.6917 - accuracy: 0.4984
18368/25000 [=====================>........] - ETA: 17s - loss: 7.6908 - accuracy: 0.4984
18400/25000 [=====================>........] - ETA: 17s - loss: 7.6891 - accuracy: 0.4985
18432/25000 [=====================>........] - ETA: 17s - loss: 7.6891 - accuracy: 0.4985
18464/25000 [=====================>........] - ETA: 17s - loss: 7.6890 - accuracy: 0.4985
18496/25000 [=====================>........] - ETA: 17s - loss: 7.6890 - accuracy: 0.4985
18528/25000 [=====================>........] - ETA: 17s - loss: 7.6931 - accuracy: 0.4983
18560/25000 [=====================>........] - ETA: 17s - loss: 7.6906 - accuracy: 0.4984
18592/25000 [=====================>........] - ETA: 17s - loss: 7.6914 - accuracy: 0.4984
18624/25000 [=====================>........] - ETA: 17s - loss: 7.6938 - accuracy: 0.4982
18656/25000 [=====================>........] - ETA: 16s - loss: 7.6921 - accuracy: 0.4983
18688/25000 [=====================>........] - ETA: 16s - loss: 7.6896 - accuracy: 0.4985
18720/25000 [=====================>........] - ETA: 16s - loss: 7.6896 - accuracy: 0.4985
18752/25000 [=====================>........] - ETA: 16s - loss: 7.6862 - accuracy: 0.4987
18784/25000 [=====================>........] - ETA: 16s - loss: 7.6870 - accuracy: 0.4987
18816/25000 [=====================>........] - ETA: 16s - loss: 7.6870 - accuracy: 0.4987
18848/25000 [=====================>........] - ETA: 16s - loss: 7.6878 - accuracy: 0.4986
18880/25000 [=====================>........] - ETA: 16s - loss: 7.6902 - accuracy: 0.4985
18912/25000 [=====================>........] - ETA: 16s - loss: 7.6942 - accuracy: 0.4982
18944/25000 [=====================>........] - ETA: 16s - loss: 7.6941 - accuracy: 0.4982
18976/25000 [=====================>........] - ETA: 16s - loss: 7.6965 - accuracy: 0.4981
19008/25000 [=====================>........] - ETA: 16s - loss: 7.6908 - accuracy: 0.4984
19040/25000 [=====================>........] - ETA: 15s - loss: 7.6859 - accuracy: 0.4987
19072/25000 [=====================>........] - ETA: 15s - loss: 7.6843 - accuracy: 0.4988
19104/25000 [=====================>........] - ETA: 15s - loss: 7.6851 - accuracy: 0.4988
19136/25000 [=====================>........] - ETA: 15s - loss: 7.6867 - accuracy: 0.4987
19168/25000 [======================>.......] - ETA: 15s - loss: 7.6882 - accuracy: 0.4986
19200/25000 [======================>.......] - ETA: 15s - loss: 7.6922 - accuracy: 0.4983
19232/25000 [======================>.......] - ETA: 15s - loss: 7.6937 - accuracy: 0.4982
19264/25000 [======================>.......] - ETA: 15s - loss: 7.6929 - accuracy: 0.4983
19296/25000 [======================>.......] - ETA: 15s - loss: 7.6889 - accuracy: 0.4985
19328/25000 [======================>.......] - ETA: 15s - loss: 7.6872 - accuracy: 0.4987
19360/25000 [======================>.......] - ETA: 15s - loss: 7.6872 - accuracy: 0.4987
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6888 - accuracy: 0.4986
19424/25000 [======================>.......] - ETA: 14s - loss: 7.6895 - accuracy: 0.4985
19456/25000 [======================>.......] - ETA: 14s - loss: 7.6895 - accuracy: 0.4985
19488/25000 [======================>.......] - ETA: 14s - loss: 7.6886 - accuracy: 0.4986
19520/25000 [======================>.......] - ETA: 14s - loss: 7.6886 - accuracy: 0.4986
19552/25000 [======================>.......] - ETA: 14s - loss: 7.6870 - accuracy: 0.4987
19584/25000 [======================>.......] - ETA: 14s - loss: 7.6893 - accuracy: 0.4985
19616/25000 [======================>.......] - ETA: 14s - loss: 7.6854 - accuracy: 0.4988
19648/25000 [======================>.......] - ETA: 14s - loss: 7.6846 - accuracy: 0.4988
19680/25000 [======================>.......] - ETA: 14s - loss: 7.6861 - accuracy: 0.4987
19712/25000 [======================>.......] - ETA: 14s - loss: 7.6900 - accuracy: 0.4985
19744/25000 [======================>.......] - ETA: 14s - loss: 7.6899 - accuracy: 0.4985
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6922 - accuracy: 0.4983
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6937 - accuracy: 0.4982
19840/25000 [======================>.......] - ETA: 13s - loss: 7.6952 - accuracy: 0.4981
19872/25000 [======================>.......] - ETA: 13s - loss: 7.6959 - accuracy: 0.4981
19904/25000 [======================>.......] - ETA: 13s - loss: 7.6959 - accuracy: 0.4981
19936/25000 [======================>.......] - ETA: 13s - loss: 7.6943 - accuracy: 0.4982
19968/25000 [======================>.......] - ETA: 13s - loss: 7.6943 - accuracy: 0.4982
20000/25000 [=======================>......] - ETA: 13s - loss: 7.6912 - accuracy: 0.4984
20032/25000 [=======================>......] - ETA: 13s - loss: 7.6926 - accuracy: 0.4983
20064/25000 [=======================>......] - ETA: 13s - loss: 7.6918 - accuracy: 0.4984
20096/25000 [=======================>......] - ETA: 13s - loss: 7.6926 - accuracy: 0.4983
20128/25000 [=======================>......] - ETA: 13s - loss: 7.6918 - accuracy: 0.4984
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6887 - accuracy: 0.4986
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6894 - accuracy: 0.4985
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6909 - accuracy: 0.4984
20256/25000 [=======================>......] - ETA: 12s - loss: 7.6916 - accuracy: 0.4984
20288/25000 [=======================>......] - ETA: 12s - loss: 7.6931 - accuracy: 0.4983
20320/25000 [=======================>......] - ETA: 12s - loss: 7.6930 - accuracy: 0.4983
20352/25000 [=======================>......] - ETA: 12s - loss: 7.6915 - accuracy: 0.4984
20384/25000 [=======================>......] - ETA: 12s - loss: 7.6907 - accuracy: 0.4984
20416/25000 [=======================>......] - ETA: 12s - loss: 7.6929 - accuracy: 0.4983
20448/25000 [=======================>......] - ETA: 12s - loss: 7.6944 - accuracy: 0.4982
20480/25000 [=======================>......] - ETA: 12s - loss: 7.6951 - accuracy: 0.4981
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6920 - accuracy: 0.4983
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6935 - accuracy: 0.4982
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6942 - accuracy: 0.4982
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6941 - accuracy: 0.4982
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6896 - accuracy: 0.4985
20672/25000 [=======================>......] - ETA: 11s - loss: 7.6904 - accuracy: 0.4985
20704/25000 [=======================>......] - ETA: 11s - loss: 7.6933 - accuracy: 0.4983
20736/25000 [=======================>......] - ETA: 11s - loss: 7.6881 - accuracy: 0.4986
20768/25000 [=======================>......] - ETA: 11s - loss: 7.6866 - accuracy: 0.4987
20800/25000 [=======================>......] - ETA: 11s - loss: 7.6821 - accuracy: 0.4990
20832/25000 [=======================>......] - ETA: 11s - loss: 7.6828 - accuracy: 0.4989
20864/25000 [========================>.....] - ETA: 11s - loss: 7.6821 - accuracy: 0.4990
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6820 - accuracy: 0.4990
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6791 - accuracy: 0.4992
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6783 - accuracy: 0.4992
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6820 - accuracy: 0.4990
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6827 - accuracy: 0.4990
21056/25000 [========================>.....] - ETA: 10s - loss: 7.6870 - accuracy: 0.4987
21088/25000 [========================>.....] - ETA: 10s - loss: 7.6863 - accuracy: 0.4987
21120/25000 [========================>.....] - ETA: 10s - loss: 7.6877 - accuracy: 0.4986
21152/25000 [========================>.....] - ETA: 10s - loss: 7.6913 - accuracy: 0.4984
21184/25000 [========================>.....] - ETA: 10s - loss: 7.6927 - accuracy: 0.4983
21216/25000 [========================>.....] - ETA: 10s - loss: 7.6919 - accuracy: 0.4984
21248/25000 [========================>.....] - ETA: 10s - loss: 7.6904 - accuracy: 0.4984
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6897 - accuracy: 0.4985 
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6911 - accuracy: 0.4984
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6889 - accuracy: 0.4985
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6889 - accuracy: 0.4985
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6910 - accuracy: 0.4984
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6881 - accuracy: 0.4986
21472/25000 [========================>.....] - ETA: 9s - loss: 7.6888 - accuracy: 0.4986
21504/25000 [========================>.....] - ETA: 9s - loss: 7.6852 - accuracy: 0.4988
21536/25000 [========================>.....] - ETA: 9s - loss: 7.6858 - accuracy: 0.4987
21568/25000 [========================>.....] - ETA: 9s - loss: 7.6830 - accuracy: 0.4989
21600/25000 [========================>.....] - ETA: 9s - loss: 7.6837 - accuracy: 0.4989
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6829 - accuracy: 0.4989
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6843 - accuracy: 0.4988
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6843 - accuracy: 0.4988
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6843 - accuracy: 0.4988
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6849 - accuracy: 0.4988
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6821 - accuracy: 0.4990
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6814 - accuracy: 0.4990
21856/25000 [=========================>....] - ETA: 8s - loss: 7.6814 - accuracy: 0.4990
21888/25000 [=========================>....] - ETA: 8s - loss: 7.6820 - accuracy: 0.4990
21920/25000 [=========================>....] - ETA: 8s - loss: 7.6827 - accuracy: 0.4990
21952/25000 [=========================>....] - ETA: 8s - loss: 7.6855 - accuracy: 0.4988
21984/25000 [=========================>....] - ETA: 8s - loss: 7.6861 - accuracy: 0.4987
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6875 - accuracy: 0.4986
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6868 - accuracy: 0.4987
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6861 - accuracy: 0.4987
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6826 - accuracy: 0.4990
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6874 - accuracy: 0.4986
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6881 - accuracy: 0.4986
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6901 - accuracy: 0.4985
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6887 - accuracy: 0.4986
22272/25000 [=========================>....] - ETA: 7s - loss: 7.6914 - accuracy: 0.4984
22304/25000 [=========================>....] - ETA: 7s - loss: 7.6921 - accuracy: 0.4983
22336/25000 [=========================>....] - ETA: 7s - loss: 7.6934 - accuracy: 0.4983
22368/25000 [=========================>....] - ETA: 7s - loss: 7.6913 - accuracy: 0.4984
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6892 - accuracy: 0.4985
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6919 - accuracy: 0.4984
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6932 - accuracy: 0.4983
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6952 - accuracy: 0.4981
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6932 - accuracy: 0.4983
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6924 - accuracy: 0.4983
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6944 - accuracy: 0.4982
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6937 - accuracy: 0.4982
22656/25000 [==========================>...] - ETA: 6s - loss: 7.6903 - accuracy: 0.4985
22688/25000 [==========================>...] - ETA: 6s - loss: 7.6889 - accuracy: 0.4985
22720/25000 [==========================>...] - ETA: 6s - loss: 7.6882 - accuracy: 0.4986
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6889 - accuracy: 0.4985
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6882 - accuracy: 0.4986
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6875 - accuracy: 0.4986
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6834 - accuracy: 0.4989
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6840 - accuracy: 0.4989
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6834 - accuracy: 0.4989
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6780 - accuracy: 0.4993
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6746 - accuracy: 0.4995
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6746 - accuracy: 0.4995
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6739 - accuracy: 0.4995
23072/25000 [==========================>...] - ETA: 5s - loss: 7.6726 - accuracy: 0.4996
23104/25000 [==========================>...] - ETA: 5s - loss: 7.6739 - accuracy: 0.4995
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6726 - accuracy: 0.4996
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6706 - accuracy: 0.4997
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6725 - accuracy: 0.4996
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6752 - accuracy: 0.4994
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6752 - accuracy: 0.4994
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6778 - accuracy: 0.4993
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6797 - accuracy: 0.4991
23456/25000 [===========================>..] - ETA: 4s - loss: 7.6777 - accuracy: 0.4993
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6764 - accuracy: 0.4994
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6757 - accuracy: 0.4994
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6757 - accuracy: 0.4994
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6738 - accuracy: 0.4995
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6731 - accuracy: 0.4996
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6718 - accuracy: 0.4997
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6718 - accuracy: 0.4997
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6743 - accuracy: 0.4995
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6737 - accuracy: 0.4995
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6711 - accuracy: 0.4997
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6718 - accuracy: 0.4997
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6737 - accuracy: 0.4995
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6756 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6762 - accuracy: 0.4994
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6711 - accuracy: 0.4997
24192/25000 [============================>.] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24224/25000 [============================>.] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24256/25000 [============================>.] - ETA: 1s - loss: 7.6717 - accuracy: 0.4997
24288/25000 [============================>.] - ETA: 1s - loss: 7.6717 - accuracy: 0.4997
24320/25000 [============================>.] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24352/25000 [============================>.] - ETA: 1s - loss: 7.6717 - accuracy: 0.4997
24384/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24416/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24448/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24480/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24512/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24544/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24576/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24608/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24640/25000 [============================>.] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
24672/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24704/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24736/25000 [============================>.] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
24768/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24800/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24832/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24864/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24896/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24928/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24960/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 78s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f7e509b3a90> 

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
 [ 0.07101095  0.01626763 -0.13739651  0.11002931  0.01652942 -0.06043019]
 [ 0.08055404 -0.16576362 -0.09360932 -0.0047913   0.09944527  0.0227443 ]
 [ 0.01420809 -0.0820155  -0.0320386   0.00107562 -0.03590615  0.07339811]
 [ 0.23458104 -0.11641797 -0.18302414 -0.06333545  0.30538118 -0.04600585]
 [ 0.29220498  0.02013054  0.18172586  0.63446856  0.07514709 -0.16789541]
 [ 0.64834601 -0.21077324 -0.20662025  0.20491502  0.20587151  0.14260201]
 [ 0.26042926  0.04411933  0.14948736  0.12177681  0.13812944  0.07773687]
 [ 0.4978368   0.08001055  0.31848338 -0.37562332  0.45124152  0.35254344]
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
{'loss': 0.5340314581990242, 'loss_history': []}

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
{'loss': 0.44056530110538006, 'loss_history': []}

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
 40%|████      | 2/5 [00:52<01:18, 26.12s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.37096439142733234, 'embedding_size_factor': 1.265718713849886, 'layers.choice': 3, 'learning_rate': 0.006404928783770289, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 2.6634745135403426e-08} and reward: 0.382
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd7\xbd\xe1nJL\xc3X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4@bD\x1e\xba"X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?z<\r\xfa\x8dRpX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\\\x99M\x91\x11Q\x1eu.' and reward: 0.382
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd7\xbd\xe1nJL\xc3X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4@bD\x1e\xba"X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?z<\r\xfa\x8dRpX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\\\x99M\x91\x11Q\x1eu.' and reward: 0.382
 60%|██████    | 3/5 [01:48<01:10, 35.03s/it] 60%|██████    | 3/5 [01:48<01:12, 36.03s/it]
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
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.09062348784691555, 'embedding_size_factor': 1.4525049593071824, 'layers.choice': 3, 'learning_rate': 0.00013978156235994904, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 6.78654678026499e-05} and reward: 0.3558
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb73\x19\xd4\x8dL\xf9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7=u\xd7\x18\x08\xd8X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?"RJzS\xb1\x08X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x11\xca_\xdb\xe4\xd5\xaau.' and reward: 0.3558
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb73\x19\xd4\x8dL\xf9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7=u\xd7\x18\x08\xd8X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?"RJzS\xb1\x08X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x11\xca_\xdb\xe4\xd5\xaau.' and reward: 0.3558
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 160.80022644996643
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -43.47s of remaining time.
Ensemble size: 13
Ensemble weights: 
[0.53846154 0.30769231 0.15384615]
	0.391	 = Validation accuracy score
	1.05s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 164.56s ...
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

