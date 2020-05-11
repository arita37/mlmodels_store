
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/4666f9a3cb70afc04820d03317a1a1d8e2a964a4', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '4666f9a3cb70afc04820d03317a1a1d8e2a964a4', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/4666f9a3cb70afc04820d03317a1a1d8e2a964a4

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/4666f9a3cb70afc04820d03317a1a1d8e2a964a4

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

    8192/17464789 [..............................] - ETA: 3s
   73728/17464789 [..............................] - ETA: 12s
  319488/17464789 [..............................] - ETA: 5s 
 1220608/17464789 [=>............................] - ETA: 2s
 2392064/17464789 [===>..........................] - ETA: 1s
 3481600/17464789 [====>.........................] - ETA: 1s
 4546560/17464789 [======>.......................] - ETA: 0s
 6184960/17464789 [=========>....................] - ETA: 0s
 8265728/17464789 [=============>................] - ETA: 0s
10018816/17464789 [================>.............] - ETA: 0s
11755520/17464789 [===================>..........] - ETA: 0s
14163968/17464789 [=======================>......] - ETA: 0s
16719872/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 00:20:46.346336: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 00:20:46.350384: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-11 00:20:46.350538: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562cd9844020 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 00:20:46.350582: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:42 - loss: 9.5833 - accuracy: 0.3750
   64/25000 [..............................] - ETA: 2:54 - loss: 9.5833 - accuracy: 0.3750
   96/25000 [..............................] - ETA: 2:18 - loss: 8.6249 - accuracy: 0.4375
  128/25000 [..............................] - ETA: 1:59 - loss: 8.1458 - accuracy: 0.4688
  160/25000 [..............................] - ETA: 1:48 - loss: 8.2416 - accuracy: 0.4625
  192/25000 [..............................] - ETA: 1:40 - loss: 8.1458 - accuracy: 0.4688
  224/25000 [..............................] - ETA: 1:34 - loss: 8.0089 - accuracy: 0.4777
  256/25000 [..............................] - ETA: 1:30 - loss: 8.0859 - accuracy: 0.4727
  288/25000 [..............................] - ETA: 1:27 - loss: 8.0925 - accuracy: 0.4722
  320/25000 [..............................] - ETA: 1:24 - loss: 8.0979 - accuracy: 0.4719
  352/25000 [..............................] - ETA: 1:22 - loss: 8.0151 - accuracy: 0.4773
  384/25000 [..............................] - ETA: 1:20 - loss: 8.1059 - accuracy: 0.4714
  416/25000 [..............................] - ETA: 1:18 - loss: 8.1089 - accuracy: 0.4712
  448/25000 [..............................] - ETA: 1:17 - loss: 8.0773 - accuracy: 0.4732
  480/25000 [..............................] - ETA: 1:16 - loss: 8.0180 - accuracy: 0.4771
  512/25000 [..............................] - ETA: 1:16 - loss: 8.0260 - accuracy: 0.4766
  544/25000 [..............................] - ETA: 1:15 - loss: 8.0612 - accuracy: 0.4743
  576/25000 [..............................] - ETA: 1:14 - loss: 8.0925 - accuracy: 0.4722
  608/25000 [..............................] - ETA: 1:14 - loss: 7.9692 - accuracy: 0.4803
  640/25000 [..............................] - ETA: 1:13 - loss: 7.9302 - accuracy: 0.4828
  672/25000 [..............................] - ETA: 1:13 - loss: 7.8720 - accuracy: 0.4866
  704/25000 [..............................] - ETA: 1:12 - loss: 7.8844 - accuracy: 0.4858
  736/25000 [..............................] - ETA: 1:12 - loss: 7.8541 - accuracy: 0.4878
  768/25000 [..............................] - ETA: 1:11 - loss: 7.7265 - accuracy: 0.4961
  800/25000 [..............................] - ETA: 1:10 - loss: 7.7625 - accuracy: 0.4938
  832/25000 [..............................] - ETA: 1:10 - loss: 7.7588 - accuracy: 0.4940
  864/25000 [>.............................] - ETA: 1:10 - loss: 7.7731 - accuracy: 0.4931
  896/25000 [>.............................] - ETA: 1:09 - loss: 7.8035 - accuracy: 0.4911
  928/25000 [>.............................] - ETA: 1:09 - loss: 7.8318 - accuracy: 0.4892
  960/25000 [>.............................] - ETA: 1:09 - loss: 7.8583 - accuracy: 0.4875
  992/25000 [>.............................] - ETA: 1:09 - loss: 7.8521 - accuracy: 0.4879
 1024/25000 [>.............................] - ETA: 1:08 - loss: 7.8313 - accuracy: 0.4893
 1056/25000 [>.............................] - ETA: 1:08 - loss: 7.8118 - accuracy: 0.4905
 1088/25000 [>.............................] - ETA: 1:08 - loss: 7.8216 - accuracy: 0.4899
 1120/25000 [>.............................] - ETA: 1:07 - loss: 7.8446 - accuracy: 0.4884
 1152/25000 [>.............................] - ETA: 1:07 - loss: 7.8796 - accuracy: 0.4861
 1184/25000 [>.............................] - ETA: 1:07 - loss: 7.8350 - accuracy: 0.4890
 1216/25000 [>.............................] - ETA: 1:07 - loss: 7.8053 - accuracy: 0.4910
 1248/25000 [>.............................] - ETA: 1:07 - loss: 7.8755 - accuracy: 0.4864
 1280/25000 [>.............................] - ETA: 1:06 - loss: 7.8583 - accuracy: 0.4875
 1312/25000 [>.............................] - ETA: 1:06 - loss: 7.7952 - accuracy: 0.4916
 1344/25000 [>.............................] - ETA: 1:06 - loss: 7.7123 - accuracy: 0.4970
 1376/25000 [>.............................] - ETA: 1:06 - loss: 7.7558 - accuracy: 0.4942
 1408/25000 [>.............................] - ETA: 1:06 - loss: 7.7973 - accuracy: 0.4915
 1440/25000 [>.............................] - ETA: 1:06 - loss: 7.8370 - accuracy: 0.4889
 1472/25000 [>.............................] - ETA: 1:06 - loss: 7.8854 - accuracy: 0.4857
 1504/25000 [>.............................] - ETA: 1:06 - loss: 7.8603 - accuracy: 0.4874
 1536/25000 [>.............................] - ETA: 1:05 - loss: 7.8363 - accuracy: 0.4889
 1568/25000 [>.............................] - ETA: 1:05 - loss: 7.8133 - accuracy: 0.4904
 1600/25000 [>.............................] - ETA: 1:05 - loss: 7.8008 - accuracy: 0.4913
 1632/25000 [>.............................] - ETA: 1:05 - loss: 7.8169 - accuracy: 0.4902
 1664/25000 [>.............................] - ETA: 1:05 - loss: 7.7496 - accuracy: 0.4946
 1696/25000 [=>............................] - ETA: 1:04 - loss: 7.7661 - accuracy: 0.4935
 1728/25000 [=>............................] - ETA: 1:04 - loss: 7.7642 - accuracy: 0.4936
 1760/25000 [=>............................] - ETA: 1:04 - loss: 7.7886 - accuracy: 0.4920
 1792/25000 [=>............................] - ETA: 1:04 - loss: 7.7950 - accuracy: 0.4916
 1824/25000 [=>............................] - ETA: 1:04 - loss: 7.7759 - accuracy: 0.4929
 1856/25000 [=>............................] - ETA: 1:04 - loss: 7.7575 - accuracy: 0.4941
 1888/25000 [=>............................] - ETA: 1:04 - loss: 7.7884 - accuracy: 0.4921
 1920/25000 [=>............................] - ETA: 1:04 - loss: 7.7625 - accuracy: 0.4938
 1952/25000 [=>............................] - ETA: 1:04 - loss: 7.7687 - accuracy: 0.4933
 1984/25000 [=>............................] - ETA: 1:04 - loss: 7.7748 - accuracy: 0.4929
 2016/25000 [=>............................] - ETA: 1:03 - loss: 7.7579 - accuracy: 0.4940
 2048/25000 [=>............................] - ETA: 1:03 - loss: 7.7789 - accuracy: 0.4927
 2080/25000 [=>............................] - ETA: 1:03 - loss: 7.7846 - accuracy: 0.4923
 2112/25000 [=>............................] - ETA: 1:03 - loss: 7.7828 - accuracy: 0.4924
 2144/25000 [=>............................] - ETA: 1:03 - loss: 7.7524 - accuracy: 0.4944
 2176/25000 [=>............................] - ETA: 1:03 - loss: 7.7582 - accuracy: 0.4940
 2208/25000 [=>............................] - ETA: 1:03 - loss: 7.7986 - accuracy: 0.4914
 2240/25000 [=>............................] - ETA: 1:03 - loss: 7.8104 - accuracy: 0.4906
 2272/25000 [=>............................] - ETA: 1:02 - loss: 7.8286 - accuracy: 0.4894
 2304/25000 [=>............................] - ETA: 1:02 - loss: 7.8130 - accuracy: 0.4905
 2336/25000 [=>............................] - ETA: 1:02 - loss: 7.7979 - accuracy: 0.4914
 2368/25000 [=>............................] - ETA: 1:02 - loss: 7.7896 - accuracy: 0.4920
 2400/25000 [=>............................] - ETA: 1:02 - loss: 7.8136 - accuracy: 0.4904
 2432/25000 [=>............................] - ETA: 1:02 - loss: 7.8116 - accuracy: 0.4905
 2464/25000 [=>............................] - ETA: 1:01 - loss: 7.7973 - accuracy: 0.4915
 2496/25000 [=>............................] - ETA: 1:01 - loss: 7.8079 - accuracy: 0.4908
 2528/25000 [==>...........................] - ETA: 1:01 - loss: 7.8243 - accuracy: 0.4897
 2560/25000 [==>...........................] - ETA: 1:01 - loss: 7.7864 - accuracy: 0.4922
 2592/25000 [==>...........................] - ETA: 1:01 - loss: 7.7849 - accuracy: 0.4923
 2624/25000 [==>...........................] - ETA: 1:01 - loss: 7.7718 - accuracy: 0.4931
 2656/25000 [==>...........................] - ETA: 1:01 - loss: 7.7532 - accuracy: 0.4944
 2688/25000 [==>...........................] - ETA: 1:01 - loss: 7.7579 - accuracy: 0.4940
 2720/25000 [==>...........................] - ETA: 1:01 - loss: 7.7512 - accuracy: 0.4945
 2752/25000 [==>...........................] - ETA: 1:00 - loss: 7.7446 - accuracy: 0.4949
 2784/25000 [==>...........................] - ETA: 1:00 - loss: 7.7492 - accuracy: 0.4946
 2816/25000 [==>...........................] - ETA: 1:00 - loss: 7.7755 - accuracy: 0.4929
 2848/25000 [==>...........................] - ETA: 1:00 - loss: 7.7635 - accuracy: 0.4937
 2880/25000 [==>...........................] - ETA: 1:00 - loss: 7.7358 - accuracy: 0.4955
 2912/25000 [==>...........................] - ETA: 1:00 - loss: 7.7456 - accuracy: 0.4948
 2944/25000 [==>...........................] - ETA: 1:00 - loss: 7.7291 - accuracy: 0.4959
 2976/25000 [==>...........................] - ETA: 59s - loss: 7.7284 - accuracy: 0.4960 
 3008/25000 [==>...........................] - ETA: 59s - loss: 7.7227 - accuracy: 0.4963
 3040/25000 [==>...........................] - ETA: 59s - loss: 7.7221 - accuracy: 0.4964
 3072/25000 [==>...........................] - ETA: 59s - loss: 7.7165 - accuracy: 0.4967
 3104/25000 [==>...........................] - ETA: 59s - loss: 7.7111 - accuracy: 0.4971
 3136/25000 [==>...........................] - ETA: 59s - loss: 7.7008 - accuracy: 0.4978
 3168/25000 [==>...........................] - ETA: 59s - loss: 7.7005 - accuracy: 0.4978
 3200/25000 [==>...........................] - ETA: 59s - loss: 7.7050 - accuracy: 0.4975
 3232/25000 [==>...........................] - ETA: 59s - loss: 7.7093 - accuracy: 0.4972
 3264/25000 [==>...........................] - ETA: 59s - loss: 7.7042 - accuracy: 0.4975
 3296/25000 [==>...........................] - ETA: 59s - loss: 7.6852 - accuracy: 0.4988
 3328/25000 [==>...........................] - ETA: 58s - loss: 7.6943 - accuracy: 0.4982
 3360/25000 [===>..........................] - ETA: 58s - loss: 7.6894 - accuracy: 0.4985
 3392/25000 [===>..........................] - ETA: 58s - loss: 7.7028 - accuracy: 0.4976
 3424/25000 [===>..........................] - ETA: 58s - loss: 7.7159 - accuracy: 0.4968
 3456/25000 [===>..........................] - ETA: 58s - loss: 7.7065 - accuracy: 0.4974
 3488/25000 [===>..........................] - ETA: 58s - loss: 7.7062 - accuracy: 0.4974
 3520/25000 [===>..........................] - ETA: 58s - loss: 7.7102 - accuracy: 0.4972
 3552/25000 [===>..........................] - ETA: 58s - loss: 7.7227 - accuracy: 0.4963
 3584/25000 [===>..........................] - ETA: 58s - loss: 7.7308 - accuracy: 0.4958
 3616/25000 [===>..........................] - ETA: 58s - loss: 7.7260 - accuracy: 0.4961
 3648/25000 [===>..........................] - ETA: 57s - loss: 7.7339 - accuracy: 0.4956
 3680/25000 [===>..........................] - ETA: 57s - loss: 7.7291 - accuracy: 0.4959
 3712/25000 [===>..........................] - ETA: 57s - loss: 7.7327 - accuracy: 0.4957
 3744/25000 [===>..........................] - ETA: 57s - loss: 7.7403 - accuracy: 0.4952
 3776/25000 [===>..........................] - ETA: 57s - loss: 7.7438 - accuracy: 0.4950
 3808/25000 [===>..........................] - ETA: 57s - loss: 7.7431 - accuracy: 0.4950
 3840/25000 [===>..........................] - ETA: 57s - loss: 7.7545 - accuracy: 0.4943
 3872/25000 [===>..........................] - ETA: 57s - loss: 7.7339 - accuracy: 0.4956
 3904/25000 [===>..........................] - ETA: 56s - loss: 7.7295 - accuracy: 0.4959
 3936/25000 [===>..........................] - ETA: 56s - loss: 7.7251 - accuracy: 0.4962
 3968/25000 [===>..........................] - ETA: 56s - loss: 7.7284 - accuracy: 0.4960
 4000/25000 [===>..........................] - ETA: 56s - loss: 7.7318 - accuracy: 0.4958
 4032/25000 [===>..........................] - ETA: 56s - loss: 7.7465 - accuracy: 0.4948
 4064/25000 [===>..........................] - ETA: 56s - loss: 7.7609 - accuracy: 0.4938
 4096/25000 [===>..........................] - ETA: 56s - loss: 7.7565 - accuracy: 0.4941
 4128/25000 [===>..........................] - ETA: 56s - loss: 7.7558 - accuracy: 0.4942
 4160/25000 [===>..........................] - ETA: 56s - loss: 7.7588 - accuracy: 0.4940
 4192/25000 [====>.........................] - ETA: 55s - loss: 7.7581 - accuracy: 0.4940
 4224/25000 [====>.........................] - ETA: 55s - loss: 7.7537 - accuracy: 0.4943
 4256/25000 [====>.........................] - ETA: 55s - loss: 7.7531 - accuracy: 0.4944
 4288/25000 [====>.........................] - ETA: 55s - loss: 7.7596 - accuracy: 0.4939
 4320/25000 [====>.........................] - ETA: 55s - loss: 7.7589 - accuracy: 0.4940
 4352/25000 [====>.........................] - ETA: 55s - loss: 7.7547 - accuracy: 0.4943
 4384/25000 [====>.........................] - ETA: 55s - loss: 7.7296 - accuracy: 0.4959
 4416/25000 [====>.........................] - ETA: 55s - loss: 7.7256 - accuracy: 0.4962
 4448/25000 [====>.........................] - ETA: 55s - loss: 7.7114 - accuracy: 0.4971
 4480/25000 [====>.........................] - ETA: 55s - loss: 7.7214 - accuracy: 0.4964
 4512/25000 [====>.........................] - ETA: 54s - loss: 7.7210 - accuracy: 0.4965
 4544/25000 [====>.........................] - ETA: 54s - loss: 7.7240 - accuracy: 0.4963
 4576/25000 [====>.........................] - ETA: 54s - loss: 7.7135 - accuracy: 0.4969
 4608/25000 [====>.........................] - ETA: 54s - loss: 7.7099 - accuracy: 0.4972
 4640/25000 [====>.........................] - ETA: 54s - loss: 7.7096 - accuracy: 0.4972
 4672/25000 [====>.........................] - ETA: 54s - loss: 7.7093 - accuracy: 0.4972
 4704/25000 [====>.........................] - ETA: 54s - loss: 7.7155 - accuracy: 0.4968
 4736/25000 [====>.........................] - ETA: 54s - loss: 7.7217 - accuracy: 0.4964
 4768/25000 [====>.........................] - ETA: 54s - loss: 7.7213 - accuracy: 0.4964
 4800/25000 [====>.........................] - ETA: 54s - loss: 7.7241 - accuracy: 0.4963
 4832/25000 [====>.........................] - ETA: 53s - loss: 7.7079 - accuracy: 0.4973
 4864/25000 [====>.........................] - ETA: 53s - loss: 7.6981 - accuracy: 0.4979
 4896/25000 [====>.........................] - ETA: 53s - loss: 7.6854 - accuracy: 0.4988
 4928/25000 [====>.........................] - ETA: 53s - loss: 7.6791 - accuracy: 0.4992
 4960/25000 [====>.........................] - ETA: 53s - loss: 7.6944 - accuracy: 0.4982
 4992/25000 [====>.........................] - ETA: 53s - loss: 7.7035 - accuracy: 0.4976
 5024/25000 [=====>........................] - ETA: 53s - loss: 7.7032 - accuracy: 0.4976
 5056/25000 [=====>........................] - ETA: 53s - loss: 7.7121 - accuracy: 0.4970
 5088/25000 [=====>........................] - ETA: 53s - loss: 7.7148 - accuracy: 0.4969
 5120/25000 [=====>........................] - ETA: 53s - loss: 7.7056 - accuracy: 0.4975
 5152/25000 [=====>........................] - ETA: 52s - loss: 7.6994 - accuracy: 0.4979
 5184/25000 [=====>........................] - ETA: 52s - loss: 7.7021 - accuracy: 0.4977
 5216/25000 [=====>........................] - ETA: 52s - loss: 7.6960 - accuracy: 0.4981
 5248/25000 [=====>........................] - ETA: 52s - loss: 7.6929 - accuracy: 0.4983
 5280/25000 [=====>........................] - ETA: 52s - loss: 7.6782 - accuracy: 0.4992
 5312/25000 [=====>........................] - ETA: 52s - loss: 7.6868 - accuracy: 0.4987
 5344/25000 [=====>........................] - ETA: 52s - loss: 7.6752 - accuracy: 0.4994
 5376/25000 [=====>........................] - ETA: 52s - loss: 7.6666 - accuracy: 0.5000
 5408/25000 [=====>........................] - ETA: 52s - loss: 7.6638 - accuracy: 0.5002
 5440/25000 [=====>........................] - ETA: 52s - loss: 7.6666 - accuracy: 0.5000
 5472/25000 [=====>........................] - ETA: 52s - loss: 7.6610 - accuracy: 0.5004
 5504/25000 [=====>........................] - ETA: 51s - loss: 7.6610 - accuracy: 0.5004
 5536/25000 [=====>........................] - ETA: 51s - loss: 7.6777 - accuracy: 0.4993
 5568/25000 [=====>........................] - ETA: 51s - loss: 7.6749 - accuracy: 0.4995
 5600/25000 [=====>........................] - ETA: 51s - loss: 7.6694 - accuracy: 0.4998
 5632/25000 [=====>........................] - ETA: 51s - loss: 7.6666 - accuracy: 0.5000
 5664/25000 [=====>........................] - ETA: 51s - loss: 7.6639 - accuracy: 0.5002
 5696/25000 [=====>........................] - ETA: 51s - loss: 7.6747 - accuracy: 0.4995
 5728/25000 [=====>........................] - ETA: 51s - loss: 7.6827 - accuracy: 0.4990
 5760/25000 [=====>........................] - ETA: 51s - loss: 7.6879 - accuracy: 0.4986
 5792/25000 [=====>........................] - ETA: 51s - loss: 7.6852 - accuracy: 0.4988
 5824/25000 [=====>........................] - ETA: 51s - loss: 7.6719 - accuracy: 0.4997
 5856/25000 [======>.......................] - ETA: 51s - loss: 7.6640 - accuracy: 0.5002
 5888/25000 [======>.......................] - ETA: 50s - loss: 7.6666 - accuracy: 0.5000
 5920/25000 [======>.......................] - ETA: 50s - loss: 7.6640 - accuracy: 0.5002
 5952/25000 [======>.......................] - ETA: 50s - loss: 7.6563 - accuracy: 0.5007
 5984/25000 [======>.......................] - ETA: 50s - loss: 7.6589 - accuracy: 0.5005
 6016/25000 [======>.......................] - ETA: 50s - loss: 7.6564 - accuracy: 0.5007
 6048/25000 [======>.......................] - ETA: 50s - loss: 7.6514 - accuracy: 0.5010
 6080/25000 [======>.......................] - ETA: 50s - loss: 7.6641 - accuracy: 0.5002
 6112/25000 [======>.......................] - ETA: 50s - loss: 7.6666 - accuracy: 0.5000
 6144/25000 [======>.......................] - ETA: 50s - loss: 7.6716 - accuracy: 0.4997
 6176/25000 [======>.......................] - ETA: 50s - loss: 7.6716 - accuracy: 0.4997
 6208/25000 [======>.......................] - ETA: 50s - loss: 7.6765 - accuracy: 0.4994
 6240/25000 [======>.......................] - ETA: 50s - loss: 7.6764 - accuracy: 0.4994
 6272/25000 [======>.......................] - ETA: 49s - loss: 7.6764 - accuracy: 0.4994
 6304/25000 [======>.......................] - ETA: 49s - loss: 7.6763 - accuracy: 0.4994
 6336/25000 [======>.......................] - ETA: 49s - loss: 7.6642 - accuracy: 0.5002
 6368/25000 [======>.......................] - ETA: 49s - loss: 7.6690 - accuracy: 0.4998
 6400/25000 [======>.......................] - ETA: 49s - loss: 7.6666 - accuracy: 0.5000
 6432/25000 [======>.......................] - ETA: 49s - loss: 7.6642 - accuracy: 0.5002
 6464/25000 [======>.......................] - ETA: 49s - loss: 7.6761 - accuracy: 0.4994
 6496/25000 [======>.......................] - ETA: 49s - loss: 7.6666 - accuracy: 0.5000
 6528/25000 [======>.......................] - ETA: 49s - loss: 7.6666 - accuracy: 0.5000
 6560/25000 [======>.......................] - ETA: 49s - loss: 7.6619 - accuracy: 0.5003
 6592/25000 [======>.......................] - ETA: 48s - loss: 7.6666 - accuracy: 0.5000
 6624/25000 [======>.......................] - ETA: 48s - loss: 7.6527 - accuracy: 0.5009
 6656/25000 [======>.......................] - ETA: 48s - loss: 7.6528 - accuracy: 0.5009
 6688/25000 [=======>......................] - ETA: 48s - loss: 7.6529 - accuracy: 0.5009
 6720/25000 [=======>......................] - ETA: 48s - loss: 7.6575 - accuracy: 0.5006
 6752/25000 [=======>......................] - ETA: 48s - loss: 7.6575 - accuracy: 0.5006
 6784/25000 [=======>......................] - ETA: 48s - loss: 7.6644 - accuracy: 0.5001
 6816/25000 [=======>......................] - ETA: 48s - loss: 7.6689 - accuracy: 0.4999
 6848/25000 [=======>......................] - ETA: 48s - loss: 7.6666 - accuracy: 0.5000
 6880/25000 [=======>......................] - ETA: 48s - loss: 7.6599 - accuracy: 0.5004
 6912/25000 [=======>......................] - ETA: 48s - loss: 7.6533 - accuracy: 0.5009
 6944/25000 [=======>......................] - ETA: 48s - loss: 7.6556 - accuracy: 0.5007
 6976/25000 [=======>......................] - ETA: 47s - loss: 7.6534 - accuracy: 0.5009
 7008/25000 [=======>......................] - ETA: 47s - loss: 7.6557 - accuracy: 0.5007
 7040/25000 [=======>......................] - ETA: 47s - loss: 7.6623 - accuracy: 0.5003
 7072/25000 [=======>......................] - ETA: 47s - loss: 7.6601 - accuracy: 0.5004
 7104/25000 [=======>......................] - ETA: 47s - loss: 7.6688 - accuracy: 0.4999
 7136/25000 [=======>......................] - ETA: 47s - loss: 7.6688 - accuracy: 0.4999
 7168/25000 [=======>......................] - ETA: 47s - loss: 7.6709 - accuracy: 0.4997
 7200/25000 [=======>......................] - ETA: 47s - loss: 7.6794 - accuracy: 0.4992
 7232/25000 [=======>......................] - ETA: 47s - loss: 7.6899 - accuracy: 0.4985
 7264/25000 [=======>......................] - ETA: 47s - loss: 7.6919 - accuracy: 0.4983
 7296/25000 [=======>......................] - ETA: 47s - loss: 7.6939 - accuracy: 0.4982
 7328/25000 [=======>......................] - ETA: 47s - loss: 7.6980 - accuracy: 0.4980
 7360/25000 [=======>......................] - ETA: 46s - loss: 7.6958 - accuracy: 0.4981
 7392/25000 [=======>......................] - ETA: 46s - loss: 7.6977 - accuracy: 0.4980
 7424/25000 [=======>......................] - ETA: 46s - loss: 7.6997 - accuracy: 0.4978
 7456/25000 [=======>......................] - ETA: 46s - loss: 7.6913 - accuracy: 0.4984
 7488/25000 [=======>......................] - ETA: 46s - loss: 7.6891 - accuracy: 0.4985
 7520/25000 [========>.....................] - ETA: 46s - loss: 7.6931 - accuracy: 0.4983
 7552/25000 [========>.....................] - ETA: 46s - loss: 7.6910 - accuracy: 0.4984
 7584/25000 [========>.....................] - ETA: 46s - loss: 7.6808 - accuracy: 0.4991
 7616/25000 [========>.....................] - ETA: 46s - loss: 7.6847 - accuracy: 0.4988
 7648/25000 [========>.....................] - ETA: 46s - loss: 7.6847 - accuracy: 0.4988
 7680/25000 [========>.....................] - ETA: 45s - loss: 7.6866 - accuracy: 0.4987
 7712/25000 [========>.....................] - ETA: 45s - loss: 7.6905 - accuracy: 0.4984
 7744/25000 [========>.....................] - ETA: 45s - loss: 7.6943 - accuracy: 0.4982
 7776/25000 [========>.....................] - ETA: 45s - loss: 7.7001 - accuracy: 0.4978
 7808/25000 [========>.....................] - ETA: 45s - loss: 7.6980 - accuracy: 0.4980
 7840/25000 [========>.....................] - ETA: 45s - loss: 7.6960 - accuracy: 0.4981
 7872/25000 [========>.....................] - ETA: 45s - loss: 7.6958 - accuracy: 0.4981
 7904/25000 [========>.....................] - ETA: 45s - loss: 7.6918 - accuracy: 0.4984
 7936/25000 [========>.....................] - ETA: 45s - loss: 7.6879 - accuracy: 0.4986
 7968/25000 [========>.....................] - ETA: 45s - loss: 7.6878 - accuracy: 0.4986
 8000/25000 [========>.....................] - ETA: 45s - loss: 7.6915 - accuracy: 0.4984
 8032/25000 [========>.....................] - ETA: 45s - loss: 7.6838 - accuracy: 0.4989
 8064/25000 [========>.....................] - ETA: 44s - loss: 7.6799 - accuracy: 0.4991
 8096/25000 [========>.....................] - ETA: 44s - loss: 7.6799 - accuracy: 0.4991
 8128/25000 [========>.....................] - ETA: 44s - loss: 7.6798 - accuracy: 0.4991
 8160/25000 [========>.....................] - ETA: 44s - loss: 7.6704 - accuracy: 0.4998
 8192/25000 [========>.....................] - ETA: 44s - loss: 7.6629 - accuracy: 0.5002
 8224/25000 [========>.....................] - ETA: 44s - loss: 7.6685 - accuracy: 0.4999
 8256/25000 [========>.....................] - ETA: 44s - loss: 7.6703 - accuracy: 0.4998
 8288/25000 [========>.....................] - ETA: 44s - loss: 7.6685 - accuracy: 0.4999
 8320/25000 [========>.....................] - ETA: 44s - loss: 7.6648 - accuracy: 0.5001
 8352/25000 [=========>....................] - ETA: 44s - loss: 7.6611 - accuracy: 0.5004
 8384/25000 [=========>....................] - ETA: 43s - loss: 7.6630 - accuracy: 0.5002
 8416/25000 [=========>....................] - ETA: 43s - loss: 7.6575 - accuracy: 0.5006
 8448/25000 [=========>....................] - ETA: 43s - loss: 7.6630 - accuracy: 0.5002
 8480/25000 [=========>....................] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
 8512/25000 [=========>....................] - ETA: 43s - loss: 7.6720 - accuracy: 0.4996
 8544/25000 [=========>....................] - ETA: 43s - loss: 7.6738 - accuracy: 0.4995
 8576/25000 [=========>....................] - ETA: 43s - loss: 7.6773 - accuracy: 0.4993
 8608/25000 [=========>....................] - ETA: 43s - loss: 7.6720 - accuracy: 0.4997
 8640/25000 [=========>....................] - ETA: 43s - loss: 7.6755 - accuracy: 0.4994
 8672/25000 [=========>....................] - ETA: 43s - loss: 7.6808 - accuracy: 0.4991
 8704/25000 [=========>....................] - ETA: 43s - loss: 7.6825 - accuracy: 0.4990
 8736/25000 [=========>....................] - ETA: 42s - loss: 7.6842 - accuracy: 0.4989
 8768/25000 [=========>....................] - ETA: 42s - loss: 7.6806 - accuracy: 0.4991
 8800/25000 [=========>....................] - ETA: 42s - loss: 7.6771 - accuracy: 0.4993
 8832/25000 [=========>....................] - ETA: 42s - loss: 7.6822 - accuracy: 0.4990
 8864/25000 [=========>....................] - ETA: 42s - loss: 7.6839 - accuracy: 0.4989
 8896/25000 [=========>....................] - ETA: 42s - loss: 7.6873 - accuracy: 0.4987
 8928/25000 [=========>....................] - ETA: 42s - loss: 7.6924 - accuracy: 0.4983
 8960/25000 [=========>....................] - ETA: 42s - loss: 7.6786 - accuracy: 0.4992
 8992/25000 [=========>....................] - ETA: 42s - loss: 7.6803 - accuracy: 0.4991
 9024/25000 [=========>....................] - ETA: 42s - loss: 7.6785 - accuracy: 0.4992
 9056/25000 [=========>....................] - ETA: 42s - loss: 7.6802 - accuracy: 0.4991
 9088/25000 [=========>....................] - ETA: 41s - loss: 7.6751 - accuracy: 0.4994
 9120/25000 [=========>....................] - ETA: 41s - loss: 7.6750 - accuracy: 0.4995
 9152/25000 [=========>....................] - ETA: 41s - loss: 7.6783 - accuracy: 0.4992
 9184/25000 [==========>...................] - ETA: 41s - loss: 7.6816 - accuracy: 0.4990
 9216/25000 [==========>...................] - ETA: 41s - loss: 7.6799 - accuracy: 0.4991
 9248/25000 [==========>...................] - ETA: 41s - loss: 7.6832 - accuracy: 0.4989
 9280/25000 [==========>...................] - ETA: 41s - loss: 7.6848 - accuracy: 0.4988
 9312/25000 [==========>...................] - ETA: 41s - loss: 7.6930 - accuracy: 0.4983
 9344/25000 [==========>...................] - ETA: 41s - loss: 7.6978 - accuracy: 0.4980
 9376/25000 [==========>...................] - ETA: 41s - loss: 7.6928 - accuracy: 0.4983
 9408/25000 [==========>...................] - ETA: 41s - loss: 7.6976 - accuracy: 0.4980
 9440/25000 [==========>...................] - ETA: 41s - loss: 7.7024 - accuracy: 0.4977
 9472/25000 [==========>...................] - ETA: 40s - loss: 7.7071 - accuracy: 0.4974
 9504/25000 [==========>...................] - ETA: 40s - loss: 7.7005 - accuracy: 0.4978
 9536/25000 [==========>...................] - ETA: 40s - loss: 7.6988 - accuracy: 0.4979
 9568/25000 [==========>...................] - ETA: 40s - loss: 7.7067 - accuracy: 0.4974
 9600/25000 [==========>...................] - ETA: 40s - loss: 7.6954 - accuracy: 0.4981
 9632/25000 [==========>...................] - ETA: 40s - loss: 7.6969 - accuracy: 0.4980
 9664/25000 [==========>...................] - ETA: 40s - loss: 7.7031 - accuracy: 0.4976
 9696/25000 [==========>...................] - ETA: 40s - loss: 7.7062 - accuracy: 0.4974
 9728/25000 [==========>...................] - ETA: 40s - loss: 7.7092 - accuracy: 0.4972
 9760/25000 [==========>...................] - ETA: 40s - loss: 7.7059 - accuracy: 0.4974
 9792/25000 [==========>...................] - ETA: 40s - loss: 7.7011 - accuracy: 0.4978
 9824/25000 [==========>...................] - ETA: 39s - loss: 7.7056 - accuracy: 0.4975
 9856/25000 [==========>...................] - ETA: 39s - loss: 7.7024 - accuracy: 0.4977
 9888/25000 [==========>...................] - ETA: 39s - loss: 7.7007 - accuracy: 0.4978
 9920/25000 [==========>...................] - ETA: 39s - loss: 7.7006 - accuracy: 0.4978
 9952/25000 [==========>...................] - ETA: 39s - loss: 7.7036 - accuracy: 0.4976
 9984/25000 [==========>...................] - ETA: 39s - loss: 7.7096 - accuracy: 0.4972
10016/25000 [===========>..................] - ETA: 39s - loss: 7.7003 - accuracy: 0.4978
10048/25000 [===========>..................] - ETA: 39s - loss: 7.7032 - accuracy: 0.4976
10080/25000 [===========>..................] - ETA: 39s - loss: 7.7077 - accuracy: 0.4973
10112/25000 [===========>..................] - ETA: 39s - loss: 7.7076 - accuracy: 0.4973
10144/25000 [===========>..................] - ETA: 39s - loss: 7.7014 - accuracy: 0.4977
10176/25000 [===========>..................] - ETA: 38s - loss: 7.7043 - accuracy: 0.4975
10208/25000 [===========>..................] - ETA: 38s - loss: 7.7087 - accuracy: 0.4973
10240/25000 [===========>..................] - ETA: 38s - loss: 7.7100 - accuracy: 0.4972
10272/25000 [===========>..................] - ETA: 38s - loss: 7.7144 - accuracy: 0.4969
10304/25000 [===========>..................] - ETA: 38s - loss: 7.7142 - accuracy: 0.4969
10336/25000 [===========>..................] - ETA: 38s - loss: 7.7156 - accuracy: 0.4968
10368/25000 [===========>..................] - ETA: 38s - loss: 7.7199 - accuracy: 0.4965
10400/25000 [===========>..................] - ETA: 38s - loss: 7.7197 - accuracy: 0.4965
10432/25000 [===========>..................] - ETA: 38s - loss: 7.7181 - accuracy: 0.4966
10464/25000 [===========>..................] - ETA: 38s - loss: 7.7106 - accuracy: 0.4971
10496/25000 [===========>..................] - ETA: 37s - loss: 7.7090 - accuracy: 0.4972
10528/25000 [===========>..................] - ETA: 37s - loss: 7.7132 - accuracy: 0.4970
10560/25000 [===========>..................] - ETA: 37s - loss: 7.7087 - accuracy: 0.4973
10592/25000 [===========>..................] - ETA: 37s - loss: 7.7086 - accuracy: 0.4973
10624/25000 [===========>..................] - ETA: 37s - loss: 7.7142 - accuracy: 0.4969
10656/25000 [===========>..................] - ETA: 37s - loss: 7.7127 - accuracy: 0.4970
10688/25000 [===========>..................] - ETA: 37s - loss: 7.7168 - accuracy: 0.4967
10720/25000 [===========>..................] - ETA: 37s - loss: 7.7052 - accuracy: 0.4975
10752/25000 [===========>..................] - ETA: 37s - loss: 7.7094 - accuracy: 0.4972
10784/25000 [===========>..................] - ETA: 37s - loss: 7.7093 - accuracy: 0.4972
10816/25000 [===========>..................] - ETA: 37s - loss: 7.7120 - accuracy: 0.4970
10848/25000 [============>.................] - ETA: 36s - loss: 7.7104 - accuracy: 0.4971
10880/25000 [============>.................] - ETA: 36s - loss: 7.7061 - accuracy: 0.4974
10912/25000 [============>.................] - ETA: 36s - loss: 7.7003 - accuracy: 0.4978
10944/25000 [============>.................] - ETA: 36s - loss: 7.6960 - accuracy: 0.4981
10976/25000 [============>.................] - ETA: 36s - loss: 7.6960 - accuracy: 0.4981
11008/25000 [============>.................] - ETA: 36s - loss: 7.6945 - accuracy: 0.4982
11040/25000 [============>.................] - ETA: 36s - loss: 7.6972 - accuracy: 0.4980
11072/25000 [============>.................] - ETA: 36s - loss: 7.6957 - accuracy: 0.4981
11104/25000 [============>.................] - ETA: 36s - loss: 7.6942 - accuracy: 0.4982
11136/25000 [============>.................] - ETA: 36s - loss: 7.6886 - accuracy: 0.4986
11168/25000 [============>.................] - ETA: 36s - loss: 7.6955 - accuracy: 0.4981
11200/25000 [============>.................] - ETA: 35s - loss: 7.6872 - accuracy: 0.4987
11232/25000 [============>.................] - ETA: 35s - loss: 7.6926 - accuracy: 0.4983
11264/25000 [============>.................] - ETA: 35s - loss: 7.6870 - accuracy: 0.4987
11296/25000 [============>.................] - ETA: 35s - loss: 7.6897 - accuracy: 0.4985
11328/25000 [============>.................] - ETA: 35s - loss: 7.6829 - accuracy: 0.4989
11360/25000 [============>.................] - ETA: 35s - loss: 7.6896 - accuracy: 0.4985
11392/25000 [============>.................] - ETA: 35s - loss: 7.6895 - accuracy: 0.4985
11424/25000 [============>.................] - ETA: 35s - loss: 7.6881 - accuracy: 0.4986
11456/25000 [============>.................] - ETA: 35s - loss: 7.6880 - accuracy: 0.4986
11488/25000 [============>.................] - ETA: 35s - loss: 7.6880 - accuracy: 0.4986
11520/25000 [============>.................] - ETA: 35s - loss: 7.6866 - accuracy: 0.4987
11552/25000 [============>.................] - ETA: 35s - loss: 7.6852 - accuracy: 0.4988
11584/25000 [============>.................] - ETA: 34s - loss: 7.6878 - accuracy: 0.4986
11616/25000 [============>.................] - ETA: 34s - loss: 7.6864 - accuracy: 0.4987
11648/25000 [============>.................] - ETA: 34s - loss: 7.6890 - accuracy: 0.4985
11680/25000 [=============>................] - ETA: 34s - loss: 7.6876 - accuracy: 0.4986
11712/25000 [=============>................] - ETA: 34s - loss: 7.6863 - accuracy: 0.4987
11744/25000 [=============>................] - ETA: 34s - loss: 7.6901 - accuracy: 0.4985
11776/25000 [=============>................] - ETA: 34s - loss: 7.6940 - accuracy: 0.4982
11808/25000 [=============>................] - ETA: 34s - loss: 7.6965 - accuracy: 0.4981
11840/25000 [=============>................] - ETA: 34s - loss: 7.6951 - accuracy: 0.4981
11872/25000 [=============>................] - ETA: 34s - loss: 7.6976 - accuracy: 0.4980
11904/25000 [=============>................] - ETA: 34s - loss: 7.6988 - accuracy: 0.4979
11936/25000 [=============>................] - ETA: 33s - loss: 7.6962 - accuracy: 0.4981
11968/25000 [=============>................] - ETA: 33s - loss: 7.7012 - accuracy: 0.4977
12000/25000 [=============>................] - ETA: 33s - loss: 7.6986 - accuracy: 0.4979
12032/25000 [=============>................] - ETA: 33s - loss: 7.6934 - accuracy: 0.4983
12064/25000 [=============>................] - ETA: 33s - loss: 7.6920 - accuracy: 0.4983
12096/25000 [=============>................] - ETA: 33s - loss: 7.6945 - accuracy: 0.4982
12128/25000 [=============>................] - ETA: 33s - loss: 7.6982 - accuracy: 0.4979
12160/25000 [=============>................] - ETA: 33s - loss: 7.6981 - accuracy: 0.4979
12192/25000 [=============>................] - ETA: 33s - loss: 7.6943 - accuracy: 0.4982
12224/25000 [=============>................] - ETA: 33s - loss: 7.6905 - accuracy: 0.4984
12256/25000 [=============>................] - ETA: 33s - loss: 7.6866 - accuracy: 0.4987
12288/25000 [=============>................] - ETA: 33s - loss: 7.6891 - accuracy: 0.4985
12320/25000 [=============>................] - ETA: 32s - loss: 7.6840 - accuracy: 0.4989
12352/25000 [=============>................] - ETA: 32s - loss: 7.6778 - accuracy: 0.4993
12384/25000 [=============>................] - ETA: 32s - loss: 7.6728 - accuracy: 0.4996
12416/25000 [=============>................] - ETA: 32s - loss: 7.6691 - accuracy: 0.4998
12448/25000 [=============>................] - ETA: 32s - loss: 7.6715 - accuracy: 0.4997
12480/25000 [=============>................] - ETA: 32s - loss: 7.6728 - accuracy: 0.4996
12512/25000 [==============>...............] - ETA: 32s - loss: 7.6776 - accuracy: 0.4993
12544/25000 [==============>...............] - ETA: 32s - loss: 7.6776 - accuracy: 0.4993
12576/25000 [==============>...............] - ETA: 32s - loss: 7.6764 - accuracy: 0.4994
12608/25000 [==============>...............] - ETA: 32s - loss: 7.6812 - accuracy: 0.4990
12640/25000 [==============>...............] - ETA: 32s - loss: 7.6860 - accuracy: 0.4987
12672/25000 [==============>...............] - ETA: 31s - loss: 7.6848 - accuracy: 0.4988
12704/25000 [==============>...............] - ETA: 31s - loss: 7.6811 - accuracy: 0.4991
12736/25000 [==============>...............] - ETA: 31s - loss: 7.6750 - accuracy: 0.4995
12768/25000 [==============>...............] - ETA: 31s - loss: 7.6750 - accuracy: 0.4995
12800/25000 [==============>...............] - ETA: 31s - loss: 7.6750 - accuracy: 0.4995
12832/25000 [==============>...............] - ETA: 31s - loss: 7.6714 - accuracy: 0.4997
12864/25000 [==============>...............] - ETA: 31s - loss: 7.6678 - accuracy: 0.4999
12896/25000 [==============>...............] - ETA: 31s - loss: 7.6642 - accuracy: 0.5002
12928/25000 [==============>...............] - ETA: 31s - loss: 7.6631 - accuracy: 0.5002
12960/25000 [==============>...............] - ETA: 31s - loss: 7.6643 - accuracy: 0.5002
12992/25000 [==============>...............] - ETA: 31s - loss: 7.6631 - accuracy: 0.5002
13024/25000 [==============>...............] - ETA: 31s - loss: 7.6654 - accuracy: 0.5001
13056/25000 [==============>...............] - ETA: 30s - loss: 7.6607 - accuracy: 0.5004
13088/25000 [==============>...............] - ETA: 30s - loss: 7.6608 - accuracy: 0.5004
13120/25000 [==============>...............] - ETA: 30s - loss: 7.6631 - accuracy: 0.5002
13152/25000 [==============>...............] - ETA: 30s - loss: 7.6596 - accuracy: 0.5005
13184/25000 [==============>...............] - ETA: 30s - loss: 7.6596 - accuracy: 0.5005
13216/25000 [==============>...............] - ETA: 30s - loss: 7.6643 - accuracy: 0.5002
13248/25000 [==============>...............] - ETA: 30s - loss: 7.6585 - accuracy: 0.5005
13280/25000 [==============>...............] - ETA: 30s - loss: 7.6632 - accuracy: 0.5002
13312/25000 [==============>...............] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
13344/25000 [===============>..............] - ETA: 30s - loss: 7.6701 - accuracy: 0.4998
13376/25000 [===============>..............] - ETA: 30s - loss: 7.6689 - accuracy: 0.4999
13408/25000 [===============>..............] - ETA: 30s - loss: 7.6632 - accuracy: 0.5002
13440/25000 [===============>..............] - ETA: 30s - loss: 7.6598 - accuracy: 0.5004
13472/25000 [===============>..............] - ETA: 29s - loss: 7.6609 - accuracy: 0.5004
13504/25000 [===============>..............] - ETA: 29s - loss: 7.6632 - accuracy: 0.5002
13536/25000 [===============>..............] - ETA: 29s - loss: 7.6678 - accuracy: 0.4999
13568/25000 [===============>..............] - ETA: 29s - loss: 7.6598 - accuracy: 0.5004
13600/25000 [===============>..............] - ETA: 29s - loss: 7.6576 - accuracy: 0.5006
13632/25000 [===============>..............] - ETA: 29s - loss: 7.6554 - accuracy: 0.5007
13664/25000 [===============>..............] - ETA: 29s - loss: 7.6565 - accuracy: 0.5007
13696/25000 [===============>..............] - ETA: 29s - loss: 7.6621 - accuracy: 0.5003
13728/25000 [===============>..............] - ETA: 29s - loss: 7.6622 - accuracy: 0.5003
13760/25000 [===============>..............] - ETA: 29s - loss: 7.6633 - accuracy: 0.5002
13792/25000 [===============>..............] - ETA: 29s - loss: 7.6622 - accuracy: 0.5003
13824/25000 [===============>..............] - ETA: 28s - loss: 7.6633 - accuracy: 0.5002
13856/25000 [===============>..............] - ETA: 28s - loss: 7.6666 - accuracy: 0.5000
13888/25000 [===============>..............] - ETA: 28s - loss: 7.6677 - accuracy: 0.4999
13920/25000 [===============>..............] - ETA: 28s - loss: 7.6688 - accuracy: 0.4999
13952/25000 [===============>..............] - ETA: 28s - loss: 7.6688 - accuracy: 0.4999
13984/25000 [===============>..............] - ETA: 28s - loss: 7.6644 - accuracy: 0.5001
14016/25000 [===============>..............] - ETA: 28s - loss: 7.6633 - accuracy: 0.5002
14048/25000 [===============>..............] - ETA: 28s - loss: 7.6666 - accuracy: 0.5000
14080/25000 [===============>..............] - ETA: 28s - loss: 7.6677 - accuracy: 0.4999
14112/25000 [===============>..............] - ETA: 28s - loss: 7.6699 - accuracy: 0.4998
14144/25000 [===============>..............] - ETA: 28s - loss: 7.6731 - accuracy: 0.4996
14176/25000 [================>.............] - ETA: 28s - loss: 7.6720 - accuracy: 0.4996
14208/25000 [================>.............] - ETA: 27s - loss: 7.6742 - accuracy: 0.4995
14240/25000 [================>.............] - ETA: 27s - loss: 7.6720 - accuracy: 0.4996
14272/25000 [================>.............] - ETA: 27s - loss: 7.6677 - accuracy: 0.4999
14304/25000 [================>.............] - ETA: 27s - loss: 7.6655 - accuracy: 0.5001
14336/25000 [================>.............] - ETA: 27s - loss: 7.6645 - accuracy: 0.5001
14368/25000 [================>.............] - ETA: 27s - loss: 7.6656 - accuracy: 0.5001
14400/25000 [================>.............] - ETA: 27s - loss: 7.6656 - accuracy: 0.5001
14432/25000 [================>.............] - ETA: 27s - loss: 7.6645 - accuracy: 0.5001
14464/25000 [================>.............] - ETA: 27s - loss: 7.6687 - accuracy: 0.4999
14496/25000 [================>.............] - ETA: 27s - loss: 7.6709 - accuracy: 0.4997
14528/25000 [================>.............] - ETA: 27s - loss: 7.6677 - accuracy: 0.4999
14560/25000 [================>.............] - ETA: 27s - loss: 7.6666 - accuracy: 0.5000
14592/25000 [================>.............] - ETA: 26s - loss: 7.6698 - accuracy: 0.4998
14624/25000 [================>.............] - ETA: 26s - loss: 7.6719 - accuracy: 0.4997
14656/25000 [================>.............] - ETA: 26s - loss: 7.6677 - accuracy: 0.4999
14688/25000 [================>.............] - ETA: 26s - loss: 7.6666 - accuracy: 0.5000
14720/25000 [================>.............] - ETA: 26s - loss: 7.6677 - accuracy: 0.4999
14752/25000 [================>.............] - ETA: 26s - loss: 7.6625 - accuracy: 0.5003
14784/25000 [================>.............] - ETA: 26s - loss: 7.6594 - accuracy: 0.5005
14816/25000 [================>.............] - ETA: 26s - loss: 7.6532 - accuracy: 0.5009
14848/25000 [================>.............] - ETA: 26s - loss: 7.6553 - accuracy: 0.5007
14880/25000 [================>.............] - ETA: 26s - loss: 7.6573 - accuracy: 0.5006
14912/25000 [================>.............] - ETA: 26s - loss: 7.6604 - accuracy: 0.5004
14944/25000 [================>.............] - ETA: 25s - loss: 7.6625 - accuracy: 0.5003
14976/25000 [================>.............] - ETA: 25s - loss: 7.6646 - accuracy: 0.5001
15008/25000 [=================>............] - ETA: 25s - loss: 7.6656 - accuracy: 0.5001
15040/25000 [=================>............] - ETA: 25s - loss: 7.6646 - accuracy: 0.5001
15072/25000 [=================>............] - ETA: 25s - loss: 7.6646 - accuracy: 0.5001
15104/25000 [=================>............] - ETA: 25s - loss: 7.6676 - accuracy: 0.4999
15136/25000 [=================>............] - ETA: 25s - loss: 7.6676 - accuracy: 0.4999
15168/25000 [=================>............] - ETA: 25s - loss: 7.6666 - accuracy: 0.5000
15200/25000 [=================>............] - ETA: 25s - loss: 7.6636 - accuracy: 0.5002
15232/25000 [=================>............] - ETA: 25s - loss: 7.6586 - accuracy: 0.5005
15264/25000 [=================>............] - ETA: 25s - loss: 7.6586 - accuracy: 0.5005
15296/25000 [=================>............] - ETA: 25s - loss: 7.6626 - accuracy: 0.5003
15328/25000 [=================>............] - ETA: 24s - loss: 7.6616 - accuracy: 0.5003
15360/25000 [=================>............] - ETA: 24s - loss: 7.6646 - accuracy: 0.5001
15392/25000 [=================>............] - ETA: 24s - loss: 7.6616 - accuracy: 0.5003
15424/25000 [=================>............] - ETA: 24s - loss: 7.6597 - accuracy: 0.5005
15456/25000 [=================>............] - ETA: 24s - loss: 7.6567 - accuracy: 0.5006
15488/25000 [=================>............] - ETA: 24s - loss: 7.6547 - accuracy: 0.5008
15520/25000 [=================>............] - ETA: 24s - loss: 7.6548 - accuracy: 0.5008
15552/25000 [=================>............] - ETA: 24s - loss: 7.6508 - accuracy: 0.5010
15584/25000 [=================>............] - ETA: 24s - loss: 7.6489 - accuracy: 0.5012
15616/25000 [=================>............] - ETA: 24s - loss: 7.6450 - accuracy: 0.5014
15648/25000 [=================>............] - ETA: 24s - loss: 7.6421 - accuracy: 0.5016
15680/25000 [=================>............] - ETA: 24s - loss: 7.6392 - accuracy: 0.5018
15712/25000 [=================>............] - ETA: 23s - loss: 7.6412 - accuracy: 0.5017
15744/25000 [=================>............] - ETA: 23s - loss: 7.6413 - accuracy: 0.5017
15776/25000 [=================>............] - ETA: 23s - loss: 7.6443 - accuracy: 0.5015
15808/25000 [=================>............] - ETA: 23s - loss: 7.6463 - accuracy: 0.5013
15840/25000 [==================>...........] - ETA: 23s - loss: 7.6463 - accuracy: 0.5013
15872/25000 [==================>...........] - ETA: 23s - loss: 7.6444 - accuracy: 0.5014
15904/25000 [==================>...........] - ETA: 23s - loss: 7.6416 - accuracy: 0.5016
15936/25000 [==================>...........] - ETA: 23s - loss: 7.6406 - accuracy: 0.5017
15968/25000 [==================>...........] - ETA: 23s - loss: 7.6426 - accuracy: 0.5016
16000/25000 [==================>...........] - ETA: 23s - loss: 7.6407 - accuracy: 0.5017
16032/25000 [==================>...........] - ETA: 23s - loss: 7.6360 - accuracy: 0.5020
16064/25000 [==================>...........] - ETA: 23s - loss: 7.6380 - accuracy: 0.5019
16096/25000 [==================>...........] - ETA: 22s - loss: 7.6380 - accuracy: 0.5019
16128/25000 [==================>...........] - ETA: 22s - loss: 7.6409 - accuracy: 0.5017
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6419 - accuracy: 0.5016
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6411 - accuracy: 0.5017
16224/25000 [==================>...........] - ETA: 22s - loss: 7.6411 - accuracy: 0.5017
16256/25000 [==================>...........] - ETA: 22s - loss: 7.6383 - accuracy: 0.5018
16288/25000 [==================>...........] - ETA: 22s - loss: 7.6421 - accuracy: 0.5016
16320/25000 [==================>...........] - ETA: 22s - loss: 7.6431 - accuracy: 0.5015
16352/25000 [==================>...........] - ETA: 22s - loss: 7.6441 - accuracy: 0.5015
16384/25000 [==================>...........] - ETA: 22s - loss: 7.6423 - accuracy: 0.5016
16416/25000 [==================>...........] - ETA: 22s - loss: 7.6405 - accuracy: 0.5017
16448/25000 [==================>...........] - ETA: 22s - loss: 7.6452 - accuracy: 0.5014
16480/25000 [==================>...........] - ETA: 21s - loss: 7.6443 - accuracy: 0.5015
16512/25000 [==================>...........] - ETA: 21s - loss: 7.6453 - accuracy: 0.5014
16544/25000 [==================>...........] - ETA: 21s - loss: 7.6472 - accuracy: 0.5013
16576/25000 [==================>...........] - ETA: 21s - loss: 7.6481 - accuracy: 0.5012
16608/25000 [==================>...........] - ETA: 21s - loss: 7.6463 - accuracy: 0.5013
16640/25000 [==================>...........] - ETA: 21s - loss: 7.6473 - accuracy: 0.5013
16672/25000 [===================>..........] - ETA: 21s - loss: 7.6519 - accuracy: 0.5010
16704/25000 [===================>..........] - ETA: 21s - loss: 7.6547 - accuracy: 0.5008
16736/25000 [===================>..........] - ETA: 21s - loss: 7.6611 - accuracy: 0.5004
16768/25000 [===================>..........] - ETA: 21s - loss: 7.6620 - accuracy: 0.5003
16800/25000 [===================>..........] - ETA: 21s - loss: 7.6648 - accuracy: 0.5001
16832/25000 [===================>..........] - ETA: 21s - loss: 7.6630 - accuracy: 0.5002
16864/25000 [===================>..........] - ETA: 20s - loss: 7.6648 - accuracy: 0.5001
16896/25000 [===================>..........] - ETA: 20s - loss: 7.6639 - accuracy: 0.5002
16928/25000 [===================>..........] - ETA: 20s - loss: 7.6630 - accuracy: 0.5002
16960/25000 [===================>..........] - ETA: 20s - loss: 7.6594 - accuracy: 0.5005
16992/25000 [===================>..........] - ETA: 20s - loss: 7.6567 - accuracy: 0.5006
17024/25000 [===================>..........] - ETA: 20s - loss: 7.6594 - accuracy: 0.5005
17056/25000 [===================>..........] - ETA: 20s - loss: 7.6612 - accuracy: 0.5004
17088/25000 [===================>..........] - ETA: 20s - loss: 7.6621 - accuracy: 0.5003
17120/25000 [===================>..........] - ETA: 20s - loss: 7.6639 - accuracy: 0.5002
17152/25000 [===================>..........] - ETA: 20s - loss: 7.6595 - accuracy: 0.5005
17184/25000 [===================>..........] - ETA: 20s - loss: 7.6613 - accuracy: 0.5003
17216/25000 [===================>..........] - ETA: 20s - loss: 7.6657 - accuracy: 0.5001
17248/25000 [===================>..........] - ETA: 19s - loss: 7.6640 - accuracy: 0.5002
17280/25000 [===================>..........] - ETA: 19s - loss: 7.6640 - accuracy: 0.5002
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6648 - accuracy: 0.5001
17344/25000 [===================>..........] - ETA: 19s - loss: 7.6666 - accuracy: 0.5000
17376/25000 [===================>..........] - ETA: 19s - loss: 7.6666 - accuracy: 0.5000
17408/25000 [===================>..........] - ETA: 19s - loss: 7.6640 - accuracy: 0.5002
17440/25000 [===================>..........] - ETA: 19s - loss: 7.6649 - accuracy: 0.5001
17472/25000 [===================>..........] - ETA: 19s - loss: 7.6622 - accuracy: 0.5003
17504/25000 [====================>.........] - ETA: 19s - loss: 7.6622 - accuracy: 0.5003
17536/25000 [====================>.........] - ETA: 19s - loss: 7.6579 - accuracy: 0.5006
17568/25000 [====================>.........] - ETA: 19s - loss: 7.6579 - accuracy: 0.5006
17600/25000 [====================>.........] - ETA: 19s - loss: 7.6614 - accuracy: 0.5003
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6588 - accuracy: 0.5005
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6562 - accuracy: 0.5007
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6588 - accuracy: 0.5005
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6545 - accuracy: 0.5008
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6545 - accuracy: 0.5008
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6537 - accuracy: 0.5008
17824/25000 [====================>.........] - ETA: 18s - loss: 7.6520 - accuracy: 0.5010
17856/25000 [====================>.........] - ETA: 18s - loss: 7.6486 - accuracy: 0.5012
17888/25000 [====================>.........] - ETA: 18s - loss: 7.6443 - accuracy: 0.5015
17920/25000 [====================>.........] - ETA: 18s - loss: 7.6427 - accuracy: 0.5016
17952/25000 [====================>.........] - ETA: 18s - loss: 7.6427 - accuracy: 0.5016
17984/25000 [====================>.........] - ETA: 18s - loss: 7.6427 - accuracy: 0.5016
18016/25000 [====================>.........] - ETA: 17s - loss: 7.6419 - accuracy: 0.5016
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6411 - accuracy: 0.5017
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6386 - accuracy: 0.5018
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6387 - accuracy: 0.5018
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6404 - accuracy: 0.5017
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6447 - accuracy: 0.5014
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6414 - accuracy: 0.5016
18240/25000 [====================>.........] - ETA: 17s - loss: 7.6448 - accuracy: 0.5014
18272/25000 [====================>.........] - ETA: 17s - loss: 7.6414 - accuracy: 0.5016
18304/25000 [====================>.........] - ETA: 17s - loss: 7.6398 - accuracy: 0.5017
18336/25000 [=====================>........] - ETA: 17s - loss: 7.6374 - accuracy: 0.5019
18368/25000 [=====================>........] - ETA: 17s - loss: 7.6357 - accuracy: 0.5020
18400/25000 [=====================>........] - ETA: 17s - loss: 7.6425 - accuracy: 0.5016
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6408 - accuracy: 0.5017
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6409 - accuracy: 0.5017
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6417 - accuracy: 0.5016
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6451 - accuracy: 0.5014
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6451 - accuracy: 0.5014
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6485 - accuracy: 0.5012
18624/25000 [=====================>........] - ETA: 16s - loss: 7.6493 - accuracy: 0.5011
18656/25000 [=====================>........] - ETA: 16s - loss: 7.6494 - accuracy: 0.5011
18688/25000 [=====================>........] - ETA: 16s - loss: 7.6527 - accuracy: 0.5009
18720/25000 [=====================>........] - ETA: 16s - loss: 7.6519 - accuracy: 0.5010
18752/25000 [=====================>........] - ETA: 16s - loss: 7.6511 - accuracy: 0.5010
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6519 - accuracy: 0.5010
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6487 - accuracy: 0.5012
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6503 - accuracy: 0.5011
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6496 - accuracy: 0.5011
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6472 - accuracy: 0.5013
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6488 - accuracy: 0.5012
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6448 - accuracy: 0.5014
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6473 - accuracy: 0.5013
19040/25000 [=====================>........] - ETA: 15s - loss: 7.6449 - accuracy: 0.5014
19072/25000 [=====================>........] - ETA: 15s - loss: 7.6425 - accuracy: 0.5016
19104/25000 [=====================>........] - ETA: 15s - loss: 7.6377 - accuracy: 0.5019
19136/25000 [=====================>........] - ETA: 15s - loss: 7.6370 - accuracy: 0.5019
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6370 - accuracy: 0.5019
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6371 - accuracy: 0.5019
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6387 - accuracy: 0.5018
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6388 - accuracy: 0.5018
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6380 - accuracy: 0.5019
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6381 - accuracy: 0.5019
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6389 - accuracy: 0.5018
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6389 - accuracy: 0.5018
19424/25000 [======================>.......] - ETA: 14s - loss: 7.6421 - accuracy: 0.5016
19456/25000 [======================>.......] - ETA: 14s - loss: 7.6398 - accuracy: 0.5017
19488/25000 [======================>.......] - ETA: 14s - loss: 7.6351 - accuracy: 0.5021
19520/25000 [======================>.......] - ETA: 14s - loss: 7.6368 - accuracy: 0.5019
19552/25000 [======================>.......] - ETA: 14s - loss: 7.6345 - accuracy: 0.5021
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6345 - accuracy: 0.5021
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6393 - accuracy: 0.5018
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6424 - accuracy: 0.5016
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6448 - accuracy: 0.5014
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6441 - accuracy: 0.5015
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6449 - accuracy: 0.5014
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6457 - accuracy: 0.5014
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6480 - accuracy: 0.5012
19840/25000 [======================>.......] - ETA: 13s - loss: 7.6496 - accuracy: 0.5011
19872/25000 [======================>.......] - ETA: 13s - loss: 7.6473 - accuracy: 0.5013
19904/25000 [======================>.......] - ETA: 13s - loss: 7.6481 - accuracy: 0.5012
19936/25000 [======================>.......] - ETA: 13s - loss: 7.6528 - accuracy: 0.5009
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6520 - accuracy: 0.5010
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6505 - accuracy: 0.5010
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6482 - accuracy: 0.5012
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6483 - accuracy: 0.5012
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6483 - accuracy: 0.5012
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6468 - accuracy: 0.5013
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6499 - accuracy: 0.5011
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6499 - accuracy: 0.5011
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6454 - accuracy: 0.5014
20256/25000 [=======================>......] - ETA: 12s - loss: 7.6477 - accuracy: 0.5012
20288/25000 [=======================>......] - ETA: 12s - loss: 7.6477 - accuracy: 0.5012
20320/25000 [=======================>......] - ETA: 12s - loss: 7.6478 - accuracy: 0.5012
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6478 - accuracy: 0.5012
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6478 - accuracy: 0.5012
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6478 - accuracy: 0.5012
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6471 - accuracy: 0.5013
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6464 - accuracy: 0.5013
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6502 - accuracy: 0.5011
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6547 - accuracy: 0.5008
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6547 - accuracy: 0.5008
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6562 - accuracy: 0.5007
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6592 - accuracy: 0.5005
20672/25000 [=======================>......] - ETA: 11s - loss: 7.6585 - accuracy: 0.5005
20704/25000 [=======================>......] - ETA: 11s - loss: 7.6607 - accuracy: 0.5004
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6577 - accuracy: 0.5006
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6578 - accuracy: 0.5006
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6563 - accuracy: 0.5007
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6563 - accuracy: 0.5007
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6541 - accuracy: 0.5008
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6549 - accuracy: 0.5008
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6534 - accuracy: 0.5009
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6535 - accuracy: 0.5009
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6549 - accuracy: 0.5008
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6550 - accuracy: 0.5008
21056/25000 [========================>.....] - ETA: 10s - loss: 7.6557 - accuracy: 0.5007
21088/25000 [========================>.....] - ETA: 10s - loss: 7.6564 - accuracy: 0.5007
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6543 - accuracy: 0.5008 
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6521 - accuracy: 0.5009
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6536 - accuracy: 0.5008
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6529 - accuracy: 0.5009
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6515 - accuracy: 0.5010
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6486 - accuracy: 0.5012
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6515 - accuracy: 0.5010
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6508 - accuracy: 0.5010
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6537 - accuracy: 0.5008
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6559 - accuracy: 0.5007
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6566 - accuracy: 0.5007
21472/25000 [========================>.....] - ETA: 9s - loss: 7.6573 - accuracy: 0.5006
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6566 - accuracy: 0.5007
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6552 - accuracy: 0.5007
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6510 - accuracy: 0.5010
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6496 - accuracy: 0.5011
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6510 - accuracy: 0.5010
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6510 - accuracy: 0.5010
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6482 - accuracy: 0.5012
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6476 - accuracy: 0.5012
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6448 - accuracy: 0.5014
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6441 - accuracy: 0.5015
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6406 - accuracy: 0.5017
21856/25000 [=========================>....] - ETA: 8s - loss: 7.6421 - accuracy: 0.5016
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6414 - accuracy: 0.5016
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6400 - accuracy: 0.5017
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6380 - accuracy: 0.5019
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6401 - accuracy: 0.5017
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6395 - accuracy: 0.5018
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6402 - accuracy: 0.5017
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6430 - accuracy: 0.5015
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6396 - accuracy: 0.5018
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6417 - accuracy: 0.5016
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6390 - accuracy: 0.5018
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6376 - accuracy: 0.5019
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6384 - accuracy: 0.5018
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6398 - accuracy: 0.5018
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6405 - accuracy: 0.5017
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6378 - accuracy: 0.5019
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6371 - accuracy: 0.5019
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6372 - accuracy: 0.5019
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6372 - accuracy: 0.5019
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6352 - accuracy: 0.5020
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6380 - accuracy: 0.5019
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6360 - accuracy: 0.5020
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6367 - accuracy: 0.5020
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6381 - accuracy: 0.5019
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6395 - accuracy: 0.5018
22656/25000 [==========================>...] - ETA: 6s - loss: 7.6429 - accuracy: 0.5015
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6409 - accuracy: 0.5017
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6416 - accuracy: 0.5016
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6410 - accuracy: 0.5017
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6410 - accuracy: 0.5017
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6431 - accuracy: 0.5015
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6411 - accuracy: 0.5017
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6358 - accuracy: 0.5020
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6352 - accuracy: 0.5021
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6339 - accuracy: 0.5021
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6339 - accuracy: 0.5021
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6353 - accuracy: 0.5020
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6367 - accuracy: 0.5020
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6367 - accuracy: 0.5020
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6368 - accuracy: 0.5019
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6368 - accuracy: 0.5019
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6375 - accuracy: 0.5019
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6349 - accuracy: 0.5021
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6349 - accuracy: 0.5021
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6343 - accuracy: 0.5021
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6357 - accuracy: 0.5020
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6384 - accuracy: 0.5018
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6371 - accuracy: 0.5019
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6345 - accuracy: 0.5021
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6352 - accuracy: 0.5020
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6352 - accuracy: 0.5020
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6340 - accuracy: 0.5021
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6353 - accuracy: 0.5020
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6347 - accuracy: 0.5021
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6348 - accuracy: 0.5021
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6355 - accuracy: 0.5020
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6348 - accuracy: 0.5021
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6375 - accuracy: 0.5019
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6395 - accuracy: 0.5018
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6401 - accuracy: 0.5017
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6395 - accuracy: 0.5018
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6415 - accuracy: 0.5016
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6402 - accuracy: 0.5017
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6422 - accuracy: 0.5016
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6435 - accuracy: 0.5015
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6448 - accuracy: 0.5014
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6449 - accuracy: 0.5014
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6449 - accuracy: 0.5014
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6462 - accuracy: 0.5013
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6462 - accuracy: 0.5013
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6475 - accuracy: 0.5012
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6476 - accuracy: 0.5012
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6450 - accuracy: 0.5014
24192/25000 [============================>.] - ETA: 2s - loss: 7.6476 - accuracy: 0.5012
24224/25000 [============================>.] - ETA: 1s - loss: 7.6470 - accuracy: 0.5013
24256/25000 [============================>.] - ETA: 1s - loss: 7.6464 - accuracy: 0.5013
24288/25000 [============================>.] - ETA: 1s - loss: 7.6483 - accuracy: 0.5012
24320/25000 [============================>.] - ETA: 1s - loss: 7.6477 - accuracy: 0.5012
24352/25000 [============================>.] - ETA: 1s - loss: 7.6440 - accuracy: 0.5015
24384/25000 [============================>.] - ETA: 1s - loss: 7.6465 - accuracy: 0.5013
24416/25000 [============================>.] - ETA: 1s - loss: 7.6478 - accuracy: 0.5012
24448/25000 [============================>.] - ETA: 1s - loss: 7.6484 - accuracy: 0.5012
24480/25000 [============================>.] - ETA: 1s - loss: 7.6447 - accuracy: 0.5014
24512/25000 [============================>.] - ETA: 1s - loss: 7.6485 - accuracy: 0.5012
24544/25000 [============================>.] - ETA: 1s - loss: 7.6491 - accuracy: 0.5011
24576/25000 [============================>.] - ETA: 1s - loss: 7.6467 - accuracy: 0.5013
24608/25000 [============================>.] - ETA: 1s - loss: 7.6454 - accuracy: 0.5014
24640/25000 [============================>.] - ETA: 0s - loss: 7.6486 - accuracy: 0.5012
24672/25000 [============================>.] - ETA: 0s - loss: 7.6511 - accuracy: 0.5010
24704/25000 [============================>.] - ETA: 0s - loss: 7.6523 - accuracy: 0.5009
24736/25000 [============================>.] - ETA: 0s - loss: 7.6536 - accuracy: 0.5008
24768/25000 [============================>.] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24800/25000 [============================>.] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24832/25000 [============================>.] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24864/25000 [============================>.] - ETA: 0s - loss: 7.6592 - accuracy: 0.5005
24896/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24928/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24960/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
25000/25000 [==============================] - 75s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f1d75407a58> 

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
 [-0.02922283  0.08809307 -0.00145645 -0.14966193 -0.05533531  0.03140925]
 [ 0.07772525  0.12026396 -0.05185551 -0.02552956 -0.09677214  0.10599115]
 [ 0.08687262  0.13938372  0.09996872 -0.01118269  0.0932241   0.06591008]
 [ 0.02522672  0.12651233  0.14039744  0.14519973  0.35623372  0.16453016]
 [ 0.01906509 -0.20874523 -0.17329638 -0.16998018 -0.28899524  0.1517093 ]
 [ 0.06794271 -0.63956374 -0.38569528  0.03762048  0.19941536 -0.20849691]
 [-0.24658179  0.21729361  0.69151342 -0.27418599  0.01271172  0.53868687]
 [-0.32967618  0.42156753  0.28907952  0.32875893 -0.04095228  1.05369473]
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
{'loss': 0.4355686828494072, 'loss_history': []}

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
{'loss': 0.5287717282772064, 'loss_history': []}

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
	Data preprocessing and feature engineering runtime = 0.23s ...
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
 40%|████      | 2/5 [00:50<01:15, 25.33s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.2713256533525157, 'embedding_size_factor': 1.2936089856365611, 'layers.choice': 0, 'learning_rate': 0.0002483733653621097, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 0.019557069884220353} and reward: 0.1946
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd1]fE\xed\xc0\xb8X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4\xb2\x9fU\xf1\xee\xa6X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?0G\x03{>\xd8\xdaX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x94\x06\xc4\xbe;\xb6bu.' and reward: 0.1946
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd1]fE\xed\xc0\xb8X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4\xb2\x9fU\xf1\xee\xa6X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?0G\x03{>\xd8\xdaX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x94\x06\xc4\xbe;\xb6bu.' and reward: 0.1946
 60%|██████    | 3/5 [01:41<01:06, 33.02s/it] 60%|██████    | 3/5 [01:41<01:07, 33.88s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.43190573842667856, 'embedding_size_factor': 0.70825943950877, 'layers.choice': 3, 'learning_rate': 0.004110825603768702, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 1.075767601839044e-05} and reward: 0.371
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdb\xa4W\xf7_\xd4\x02X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6\xaa\x0f\xb38\xc0*X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?p\xd6\x83Xq\xa5\xd2X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xe6\x8f{\xbb\xd7\x97gu.' and reward: 0.371
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdb\xa4W\xf7_\xd4\x02X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6\xaa\x0f\xb38\xc0*X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?p\xd6\x83Xq\xa5\xd2X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xe6\x8f{\xbb\xd7\x97gu.' and reward: 0.371
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 154.80303978919983
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.77s of the -37.33s of remaining time.
Ensemble size: 11
Ensemble weights: 
[0.72727273 0.27272727 0.        ]
	0.3884	 = Validation accuracy score
	1.06s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 158.44s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
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

