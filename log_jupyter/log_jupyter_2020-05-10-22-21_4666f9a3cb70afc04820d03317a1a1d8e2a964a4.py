
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

    8192/17464789 [..............................] - ETA: 0s
  876544/17464789 [>.............................] - ETA: 1s
 2351104/17464789 [===>..........................] - ETA: 0s
 3973120/17464789 [=====>........................] - ETA: 0s
 5660672/17464789 [========>.....................] - ETA: 0s
 7364608/17464789 [===========>..................] - ETA: 0s
 8724480/17464789 [=============>................] - ETA: 0s
10567680/17464789 [=================>............] - ETA: 0s
12115968/17464789 [===================>..........] - ETA: 0s
14000128/17464789 [=======================>......] - ETA: 0s
15876096/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 22:21:38.577748: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 22:21:38.581884: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-10 22:21:38.582386: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55772fb641a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 22:21:38.582404: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:36 - loss: 10.0624 - accuracy: 0.3438
   64/25000 [..............................] - ETA: 2:52 - loss: 9.5833 - accuracy: 0.3750 
   96/25000 [..............................] - ETA: 2:15 - loss: 9.9027 - accuracy: 0.3542
  128/25000 [..............................] - ETA: 1:58 - loss: 9.4635 - accuracy: 0.3828
  160/25000 [..............................] - ETA: 1:46 - loss: 9.5833 - accuracy: 0.3750
  192/25000 [..............................] - ETA: 1:39 - loss: 9.2638 - accuracy: 0.3958
  224/25000 [..............................] - ETA: 1:33 - loss: 9.1726 - accuracy: 0.4018
  256/25000 [..............................] - ETA: 1:29 - loss: 9.0442 - accuracy: 0.4102
  288/25000 [..............................] - ETA: 1:26 - loss: 8.8379 - accuracy: 0.4236
  320/25000 [..............................] - ETA: 1:23 - loss: 8.7208 - accuracy: 0.4313
  352/25000 [..............................] - ETA: 1:21 - loss: 8.7121 - accuracy: 0.4318
  384/25000 [..............................] - ETA: 1:19 - loss: 8.7048 - accuracy: 0.4323
  416/25000 [..............................] - ETA: 1:18 - loss: 8.5881 - accuracy: 0.4399
  448/25000 [..............................] - ETA: 1:17 - loss: 8.3169 - accuracy: 0.4576
  480/25000 [..............................] - ETA: 1:15 - loss: 8.1138 - accuracy: 0.4708
  512/25000 [..............................] - ETA: 1:14 - loss: 8.1158 - accuracy: 0.4707
  544/25000 [..............................] - ETA: 1:13 - loss: 8.0330 - accuracy: 0.4761
  576/25000 [..............................] - ETA: 1:12 - loss: 8.0393 - accuracy: 0.4757
  608/25000 [..............................] - ETA: 1:11 - loss: 8.1206 - accuracy: 0.4704
  640/25000 [..............................] - ETA: 1:11 - loss: 8.1937 - accuracy: 0.4656
  672/25000 [..............................] - ETA: 1:10 - loss: 8.1230 - accuracy: 0.4702
  704/25000 [..............................] - ETA: 1:10 - loss: 8.1240 - accuracy: 0.4702
  736/25000 [..............................] - ETA: 1:09 - loss: 8.1249 - accuracy: 0.4701
  768/25000 [..............................] - ETA: 1:09 - loss: 8.1857 - accuracy: 0.4661
  800/25000 [..............................] - ETA: 1:08 - loss: 8.1075 - accuracy: 0.4712
  832/25000 [..............................] - ETA: 1:08 - loss: 8.0536 - accuracy: 0.4748
  864/25000 [>.............................] - ETA: 1:08 - loss: 8.0038 - accuracy: 0.4780
  896/25000 [>.............................] - ETA: 1:07 - loss: 8.0602 - accuracy: 0.4743
  928/25000 [>.............................] - ETA: 1:07 - loss: 8.0301 - accuracy: 0.4763
  960/25000 [>.............................] - ETA: 1:07 - loss: 7.9541 - accuracy: 0.4812
  992/25000 [>.............................] - ETA: 1:06 - loss: 7.9912 - accuracy: 0.4788
 1024/25000 [>.............................] - ETA: 1:06 - loss: 7.8912 - accuracy: 0.4854
 1056/25000 [>.............................] - ETA: 1:06 - loss: 7.9570 - accuracy: 0.4811
 1088/25000 [>.............................] - ETA: 1:06 - loss: 7.8921 - accuracy: 0.4853
 1120/25000 [>.............................] - ETA: 1:06 - loss: 7.8994 - accuracy: 0.4848
 1152/25000 [>.............................] - ETA: 1:06 - loss: 7.8530 - accuracy: 0.4878
 1184/25000 [>.............................] - ETA: 1:05 - loss: 7.8091 - accuracy: 0.4907
 1216/25000 [>.............................] - ETA: 1:05 - loss: 7.8558 - accuracy: 0.4877
 1248/25000 [>.............................] - ETA: 1:05 - loss: 7.8878 - accuracy: 0.4856
 1280/25000 [>.............................] - ETA: 1:04 - loss: 7.8942 - accuracy: 0.4852
 1312/25000 [>.............................] - ETA: 1:04 - loss: 7.8887 - accuracy: 0.4855
 1344/25000 [>.............................] - ETA: 1:04 - loss: 7.8834 - accuracy: 0.4859
 1376/25000 [>.............................] - ETA: 1:04 - loss: 7.8449 - accuracy: 0.4884
 1408/25000 [>.............................] - ETA: 1:03 - loss: 7.8409 - accuracy: 0.4886
 1440/25000 [>.............................] - ETA: 1:03 - loss: 7.8050 - accuracy: 0.4910
 1472/25000 [>.............................] - ETA: 1:03 - loss: 7.7604 - accuracy: 0.4939
 1504/25000 [>.............................] - ETA: 1:03 - loss: 7.7992 - accuracy: 0.4914
 1536/25000 [>.............................] - ETA: 1:03 - loss: 7.8064 - accuracy: 0.4909
 1568/25000 [>.............................] - ETA: 1:03 - loss: 7.8133 - accuracy: 0.4904
 1600/25000 [>.............................] - ETA: 1:03 - loss: 7.8487 - accuracy: 0.4881
 1632/25000 [>.............................] - ETA: 1:02 - loss: 7.8076 - accuracy: 0.4908
 1664/25000 [>.............................] - ETA: 1:02 - loss: 7.7956 - accuracy: 0.4916
 1696/25000 [=>............................] - ETA: 1:02 - loss: 7.8113 - accuracy: 0.4906
 1728/25000 [=>............................] - ETA: 1:02 - loss: 7.8352 - accuracy: 0.4890
 1760/25000 [=>............................] - ETA: 1:02 - loss: 7.8496 - accuracy: 0.4881
 1792/25000 [=>............................] - ETA: 1:02 - loss: 7.7950 - accuracy: 0.4916
 1824/25000 [=>............................] - ETA: 1:01 - loss: 7.8347 - accuracy: 0.4890
 1856/25000 [=>............................] - ETA: 1:01 - loss: 7.8401 - accuracy: 0.4887
 1888/25000 [=>............................] - ETA: 1:01 - loss: 7.8534 - accuracy: 0.4878
 1920/25000 [=>............................] - ETA: 1:01 - loss: 7.8982 - accuracy: 0.4849
 1952/25000 [=>............................] - ETA: 1:01 - loss: 7.9023 - accuracy: 0.4846
 1984/25000 [=>............................] - ETA: 1:01 - loss: 7.9062 - accuracy: 0.4844
 2016/25000 [=>............................] - ETA: 1:01 - loss: 7.9024 - accuracy: 0.4846
 2048/25000 [=>............................] - ETA: 1:00 - loss: 7.9062 - accuracy: 0.4844
 2080/25000 [=>............................] - ETA: 1:00 - loss: 7.9099 - accuracy: 0.4841
 2112/25000 [=>............................] - ETA: 1:00 - loss: 7.9207 - accuracy: 0.4834
 2144/25000 [=>............................] - ETA: 1:00 - loss: 7.9169 - accuracy: 0.4837
 2176/25000 [=>............................] - ETA: 1:00 - loss: 7.8921 - accuracy: 0.4853
 2208/25000 [=>............................] - ETA: 1:00 - loss: 7.8819 - accuracy: 0.4860
 2240/25000 [=>............................] - ETA: 1:00 - loss: 7.8857 - accuracy: 0.4857
 2272/25000 [=>............................] - ETA: 59s - loss: 7.8961 - accuracy: 0.4850 
 2304/25000 [=>............................] - ETA: 59s - loss: 7.8995 - accuracy: 0.4848
 2336/25000 [=>............................] - ETA: 59s - loss: 7.9095 - accuracy: 0.4842
 2368/25000 [=>............................] - ETA: 59s - loss: 7.8868 - accuracy: 0.4856
 2400/25000 [=>............................] - ETA: 59s - loss: 7.8902 - accuracy: 0.4854
 2432/25000 [=>............................] - ETA: 59s - loss: 7.8621 - accuracy: 0.4873
 2464/25000 [=>............................] - ETA: 59s - loss: 7.8471 - accuracy: 0.4882
 2496/25000 [=>............................] - ETA: 58s - loss: 7.8386 - accuracy: 0.4888
 2528/25000 [==>...........................] - ETA: 58s - loss: 7.8364 - accuracy: 0.4889
 2560/25000 [==>...........................] - ETA: 58s - loss: 7.8223 - accuracy: 0.4898
 2592/25000 [==>...........................] - ETA: 58s - loss: 7.8263 - accuracy: 0.4896
 2624/25000 [==>...........................] - ETA: 58s - loss: 7.7952 - accuracy: 0.4916
 2656/25000 [==>...........................] - ETA: 58s - loss: 7.7532 - accuracy: 0.4944
 2688/25000 [==>...........................] - ETA: 58s - loss: 7.7750 - accuracy: 0.4929
 2720/25000 [==>...........................] - ETA: 58s - loss: 7.7512 - accuracy: 0.4945
 2752/25000 [==>...........................] - ETA: 58s - loss: 7.7502 - accuracy: 0.4945
 2784/25000 [==>...........................] - ETA: 57s - loss: 7.7547 - accuracy: 0.4943
 2816/25000 [==>...........................] - ETA: 57s - loss: 7.7592 - accuracy: 0.4940
 2848/25000 [==>...........................] - ETA: 57s - loss: 7.7581 - accuracy: 0.4940
 2880/25000 [==>...........................] - ETA: 57s - loss: 7.7678 - accuracy: 0.4934
 2912/25000 [==>...........................] - ETA: 57s - loss: 7.7772 - accuracy: 0.4928
 2944/25000 [==>...........................] - ETA: 57s - loss: 7.7604 - accuracy: 0.4939
 2976/25000 [==>...........................] - ETA: 57s - loss: 7.7439 - accuracy: 0.4950
 3008/25000 [==>...........................] - ETA: 57s - loss: 7.7584 - accuracy: 0.4940
 3040/25000 [==>...........................] - ETA: 57s - loss: 7.7574 - accuracy: 0.4941
 3072/25000 [==>...........................] - ETA: 56s - loss: 7.7615 - accuracy: 0.4938
 3104/25000 [==>...........................] - ETA: 56s - loss: 7.7358 - accuracy: 0.4955
 3136/25000 [==>...........................] - ETA: 56s - loss: 7.7106 - accuracy: 0.4971
 3168/25000 [==>...........................] - ETA: 56s - loss: 7.7102 - accuracy: 0.4972
 3200/25000 [==>...........................] - ETA: 56s - loss: 7.7289 - accuracy: 0.4959
 3232/25000 [==>...........................] - ETA: 56s - loss: 7.7425 - accuracy: 0.4950
 3264/25000 [==>...........................] - ETA: 56s - loss: 7.7230 - accuracy: 0.4963
 3296/25000 [==>...........................] - ETA: 56s - loss: 7.7224 - accuracy: 0.4964
 3328/25000 [==>...........................] - ETA: 56s - loss: 7.7311 - accuracy: 0.4958
 3360/25000 [===>..........................] - ETA: 56s - loss: 7.7305 - accuracy: 0.4958
 3392/25000 [===>..........................] - ETA: 56s - loss: 7.7480 - accuracy: 0.4947
 3424/25000 [===>..........................] - ETA: 55s - loss: 7.7427 - accuracy: 0.4950
 3456/25000 [===>..........................] - ETA: 55s - loss: 7.7376 - accuracy: 0.4954
 3488/25000 [===>..........................] - ETA: 55s - loss: 7.7457 - accuracy: 0.4948
 3520/25000 [===>..........................] - ETA: 55s - loss: 7.7450 - accuracy: 0.4949
 3552/25000 [===>..........................] - ETA: 55s - loss: 7.7486 - accuracy: 0.4947
 3584/25000 [===>..........................] - ETA: 55s - loss: 7.7693 - accuracy: 0.4933
 3616/25000 [===>..........................] - ETA: 55s - loss: 7.7769 - accuracy: 0.4928
 3648/25000 [===>..........................] - ETA: 55s - loss: 7.7633 - accuracy: 0.4937
 3680/25000 [===>..........................] - ETA: 55s - loss: 7.7625 - accuracy: 0.4938
 3712/25000 [===>..........................] - ETA: 55s - loss: 7.7699 - accuracy: 0.4933
 3744/25000 [===>..........................] - ETA: 55s - loss: 7.7690 - accuracy: 0.4933
 3776/25000 [===>..........................] - ETA: 54s - loss: 7.7763 - accuracy: 0.4928
 3808/25000 [===>..........................] - ETA: 54s - loss: 7.7673 - accuracy: 0.4934
 3840/25000 [===>..........................] - ETA: 54s - loss: 7.7744 - accuracy: 0.4930
 3872/25000 [===>..........................] - ETA: 54s - loss: 7.7815 - accuracy: 0.4925
 3904/25000 [===>..........................] - ETA: 54s - loss: 7.7648 - accuracy: 0.4936
 3936/25000 [===>..........................] - ETA: 54s - loss: 7.7601 - accuracy: 0.4939
 3968/25000 [===>..........................] - ETA: 54s - loss: 7.7710 - accuracy: 0.4932
 4000/25000 [===>..........................] - ETA: 54s - loss: 7.7701 - accuracy: 0.4933
 4032/25000 [===>..........................] - ETA: 54s - loss: 7.7807 - accuracy: 0.4926
 4064/25000 [===>..........................] - ETA: 54s - loss: 7.7911 - accuracy: 0.4919
 4096/25000 [===>..........................] - ETA: 53s - loss: 7.7902 - accuracy: 0.4919
 4128/25000 [===>..........................] - ETA: 53s - loss: 7.7743 - accuracy: 0.4930
 4160/25000 [===>..........................] - ETA: 53s - loss: 7.7625 - accuracy: 0.4938
 4192/25000 [====>.........................] - ETA: 53s - loss: 7.7654 - accuracy: 0.4936
 4224/25000 [====>.........................] - ETA: 53s - loss: 7.7610 - accuracy: 0.4938
 4256/25000 [====>.........................] - ETA: 53s - loss: 7.7603 - accuracy: 0.4939
 4288/25000 [====>.........................] - ETA: 53s - loss: 7.7524 - accuracy: 0.4944
 4320/25000 [====>.........................] - ETA: 53s - loss: 7.7376 - accuracy: 0.4954
 4352/25000 [====>.........................] - ETA: 53s - loss: 7.7300 - accuracy: 0.4959
 4384/25000 [====>.........................] - ETA: 53s - loss: 7.7436 - accuracy: 0.4950
 4416/25000 [====>.........................] - ETA: 53s - loss: 7.7430 - accuracy: 0.4950
 4448/25000 [====>.........................] - ETA: 53s - loss: 7.7562 - accuracy: 0.4942
 4480/25000 [====>.........................] - ETA: 53s - loss: 7.7625 - accuracy: 0.4938
 4512/25000 [====>.........................] - ETA: 52s - loss: 7.7652 - accuracy: 0.4936
 4544/25000 [====>.........................] - ETA: 52s - loss: 7.7645 - accuracy: 0.4936
 4576/25000 [====>.........................] - ETA: 52s - loss: 7.7805 - accuracy: 0.4926
 4608/25000 [====>.........................] - ETA: 52s - loss: 7.7864 - accuracy: 0.4922
 4640/25000 [====>.........................] - ETA: 52s - loss: 7.7691 - accuracy: 0.4933
 4672/25000 [====>.........................] - ETA: 52s - loss: 7.7848 - accuracy: 0.4923
 4704/25000 [====>.........................] - ETA: 52s - loss: 7.7970 - accuracy: 0.4915
 4736/25000 [====>.........................] - ETA: 52s - loss: 7.8058 - accuracy: 0.4909
 4768/25000 [====>.........................] - ETA: 52s - loss: 7.8049 - accuracy: 0.4910
 4800/25000 [====>.........................] - ETA: 52s - loss: 7.8040 - accuracy: 0.4910
 4832/25000 [====>.........................] - ETA: 52s - loss: 7.8031 - accuracy: 0.4911
 4864/25000 [====>.........................] - ETA: 51s - loss: 7.8022 - accuracy: 0.4912
 4896/25000 [====>.........................] - ETA: 51s - loss: 7.8076 - accuracy: 0.4908
 4928/25000 [====>.........................] - ETA: 51s - loss: 7.8160 - accuracy: 0.4903
 4960/25000 [====>.........................] - ETA: 51s - loss: 7.8026 - accuracy: 0.4911
 4992/25000 [====>.........................] - ETA: 51s - loss: 7.7864 - accuracy: 0.4922
 5024/25000 [=====>........................] - ETA: 51s - loss: 7.8009 - accuracy: 0.4912
 5056/25000 [=====>........................] - ETA: 51s - loss: 7.7879 - accuracy: 0.4921
 5088/25000 [=====>........................] - ETA: 51s - loss: 7.7842 - accuracy: 0.4923
 5120/25000 [=====>........................] - ETA: 51s - loss: 7.7774 - accuracy: 0.4928
 5152/25000 [=====>........................] - ETA: 51s - loss: 7.7916 - accuracy: 0.4918
 5184/25000 [=====>........................] - ETA: 51s - loss: 7.7849 - accuracy: 0.4923
 5216/25000 [=====>........................] - ETA: 50s - loss: 7.7901 - accuracy: 0.4919
 5248/25000 [=====>........................] - ETA: 50s - loss: 7.7806 - accuracy: 0.4926
 5280/25000 [=====>........................] - ETA: 50s - loss: 7.8031 - accuracy: 0.4911
 5312/25000 [=====>........................] - ETA: 50s - loss: 7.8081 - accuracy: 0.4908
 5344/25000 [=====>........................] - ETA: 50s - loss: 7.8072 - accuracy: 0.4908
 5376/25000 [=====>........................] - ETA: 50s - loss: 7.8206 - accuracy: 0.4900
 5408/25000 [=====>........................] - ETA: 50s - loss: 7.8226 - accuracy: 0.4898
 5440/25000 [=====>........................] - ETA: 50s - loss: 7.8245 - accuracy: 0.4897
 5472/25000 [=====>........................] - ETA: 50s - loss: 7.8291 - accuracy: 0.4894
 5504/25000 [=====>........................] - ETA: 50s - loss: 7.8310 - accuracy: 0.4893
 5536/25000 [=====>........................] - ETA: 50s - loss: 7.8273 - accuracy: 0.4895
 5568/25000 [=====>........................] - ETA: 50s - loss: 7.8236 - accuracy: 0.4898
 5600/25000 [=====>........................] - ETA: 49s - loss: 7.8172 - accuracy: 0.4902
 5632/25000 [=====>........................] - ETA: 49s - loss: 7.8164 - accuracy: 0.4902
 5664/25000 [=====>........................] - ETA: 49s - loss: 7.8209 - accuracy: 0.4899
 5696/25000 [=====>........................] - ETA: 49s - loss: 7.8147 - accuracy: 0.4903
 5728/25000 [=====>........................] - ETA: 49s - loss: 7.8112 - accuracy: 0.4906
 5760/25000 [=====>........................] - ETA: 49s - loss: 7.8184 - accuracy: 0.4901
 5792/25000 [=====>........................] - ETA: 49s - loss: 7.8149 - accuracy: 0.4903
 5824/25000 [=====>........................] - ETA: 49s - loss: 7.8088 - accuracy: 0.4907
 5856/25000 [======>.......................] - ETA: 49s - loss: 7.8054 - accuracy: 0.4909
 5888/25000 [======>.......................] - ETA: 49s - loss: 7.7942 - accuracy: 0.4917
 5920/25000 [======>.......................] - ETA: 49s - loss: 7.7909 - accuracy: 0.4919
 5952/25000 [======>.......................] - ETA: 48s - loss: 7.7825 - accuracy: 0.4924
 5984/25000 [======>.......................] - ETA: 48s - loss: 7.7794 - accuracy: 0.4926
 6016/25000 [======>.......................] - ETA: 48s - loss: 7.7737 - accuracy: 0.4930
 6048/25000 [======>.......................] - ETA: 48s - loss: 7.7908 - accuracy: 0.4919
 6080/25000 [======>.......................] - ETA: 48s - loss: 7.7826 - accuracy: 0.4924
 6112/25000 [======>.......................] - ETA: 48s - loss: 7.7971 - accuracy: 0.4915
 6144/25000 [======>.......................] - ETA: 48s - loss: 7.8089 - accuracy: 0.4907
 6176/25000 [======>.......................] - ETA: 48s - loss: 7.8156 - accuracy: 0.4903
 6208/25000 [======>.......................] - ETA: 48s - loss: 7.8025 - accuracy: 0.4911
 6240/25000 [======>.......................] - ETA: 48s - loss: 7.7993 - accuracy: 0.4913
 6272/25000 [======>.......................] - ETA: 48s - loss: 7.7962 - accuracy: 0.4915
 6304/25000 [======>.......................] - ETA: 48s - loss: 7.8053 - accuracy: 0.4910
 6336/25000 [======>.......................] - ETA: 47s - loss: 7.8118 - accuracy: 0.4905
 6368/25000 [======>.......................] - ETA: 47s - loss: 7.8087 - accuracy: 0.4907
 6400/25000 [======>.......................] - ETA: 47s - loss: 7.8176 - accuracy: 0.4902
 6432/25000 [======>.......................] - ETA: 47s - loss: 7.8097 - accuracy: 0.4907
 6464/25000 [======>.......................] - ETA: 47s - loss: 7.8066 - accuracy: 0.4909
 6496/25000 [======>.......................] - ETA: 47s - loss: 7.8082 - accuracy: 0.4908
 6528/25000 [======>.......................] - ETA: 47s - loss: 7.8099 - accuracy: 0.4907
 6560/25000 [======>.......................] - ETA: 47s - loss: 7.8092 - accuracy: 0.4907
 6592/25000 [======>.......................] - ETA: 47s - loss: 7.8132 - accuracy: 0.4904
 6624/25000 [======>.......................] - ETA: 47s - loss: 7.8078 - accuracy: 0.4908
 6656/25000 [======>.......................] - ETA: 47s - loss: 7.8048 - accuracy: 0.4910
 6688/25000 [=======>......................] - ETA: 47s - loss: 7.8088 - accuracy: 0.4907
 6720/25000 [=======>......................] - ETA: 46s - loss: 7.8195 - accuracy: 0.4900
 6752/25000 [=======>......................] - ETA: 46s - loss: 7.8142 - accuracy: 0.4904
 6784/25000 [=======>......................] - ETA: 46s - loss: 7.8113 - accuracy: 0.4906
 6816/25000 [=======>......................] - ETA: 46s - loss: 7.7948 - accuracy: 0.4916
 6848/25000 [=======>......................] - ETA: 46s - loss: 7.7942 - accuracy: 0.4917
 6880/25000 [=======>......................] - ETA: 46s - loss: 7.7981 - accuracy: 0.4914
 6912/25000 [=======>......................] - ETA: 46s - loss: 7.7997 - accuracy: 0.4913
 6944/25000 [=======>......................] - ETA: 46s - loss: 7.8035 - accuracy: 0.4911
 6976/25000 [=======>......................] - ETA: 46s - loss: 7.7985 - accuracy: 0.4914
 7008/25000 [=======>......................] - ETA: 46s - loss: 7.8001 - accuracy: 0.4913
 7040/25000 [=======>......................] - ETA: 46s - loss: 7.7995 - accuracy: 0.4913
 7072/25000 [=======>......................] - ETA: 46s - loss: 7.8054 - accuracy: 0.4910
 7104/25000 [=======>......................] - ETA: 45s - loss: 7.8004 - accuracy: 0.4913
 7136/25000 [=======>......................] - ETA: 45s - loss: 7.7934 - accuracy: 0.4917
 7168/25000 [=======>......................] - ETA: 45s - loss: 7.8014 - accuracy: 0.4912
 7200/25000 [=======>......................] - ETA: 45s - loss: 7.8050 - accuracy: 0.4910
 7232/25000 [=======>......................] - ETA: 45s - loss: 7.8002 - accuracy: 0.4913
 7264/25000 [=======>......................] - ETA: 45s - loss: 7.8080 - accuracy: 0.4908
 7296/25000 [=======>......................] - ETA: 45s - loss: 7.8137 - accuracy: 0.4904
 7328/25000 [=======>......................] - ETA: 45s - loss: 7.8194 - accuracy: 0.4900
 7360/25000 [=======>......................] - ETA: 45s - loss: 7.8145 - accuracy: 0.4904
 7392/25000 [=======>......................] - ETA: 45s - loss: 7.8077 - accuracy: 0.4908
 7424/25000 [=======>......................] - ETA: 45s - loss: 7.8112 - accuracy: 0.4906
 7456/25000 [=======>......................] - ETA: 45s - loss: 7.8188 - accuracy: 0.4901
 7488/25000 [=======>......................] - ETA: 44s - loss: 7.8141 - accuracy: 0.4904
 7520/25000 [========>.....................] - ETA: 44s - loss: 7.8032 - accuracy: 0.4911
 7552/25000 [========>.....................] - ETA: 44s - loss: 7.7986 - accuracy: 0.4914
 7584/25000 [========>.....................] - ETA: 44s - loss: 7.8021 - accuracy: 0.4912
 7616/25000 [========>.....................] - ETA: 44s - loss: 7.8015 - accuracy: 0.4912
 7648/25000 [========>.....................] - ETA: 44s - loss: 7.8030 - accuracy: 0.4911
 7680/25000 [========>.....................] - ETA: 44s - loss: 7.7984 - accuracy: 0.4914
 7712/25000 [========>.....................] - ETA: 44s - loss: 7.7899 - accuracy: 0.4920
 7744/25000 [========>.....................] - ETA: 44s - loss: 7.7973 - accuracy: 0.4915
 7776/25000 [========>.....................] - ETA: 44s - loss: 7.7948 - accuracy: 0.4916
 7808/25000 [========>.....................] - ETA: 44s - loss: 7.7943 - accuracy: 0.4917
 7840/25000 [========>.....................] - ETA: 43s - loss: 7.8016 - accuracy: 0.4912
 7872/25000 [========>.....................] - ETA: 43s - loss: 7.8030 - accuracy: 0.4911
 7904/25000 [========>.....................] - ETA: 43s - loss: 7.8024 - accuracy: 0.4911
 7936/25000 [========>.....................] - ETA: 43s - loss: 7.8019 - accuracy: 0.4912
 7968/25000 [========>.....................] - ETA: 43s - loss: 7.8013 - accuracy: 0.4912
 8000/25000 [========>.....................] - ETA: 43s - loss: 7.8008 - accuracy: 0.4913
 8032/25000 [========>.....................] - ETA: 43s - loss: 7.7983 - accuracy: 0.4914
 8064/25000 [========>.....................] - ETA: 43s - loss: 7.7940 - accuracy: 0.4917
 8096/25000 [========>.....................] - ETA: 43s - loss: 7.7954 - accuracy: 0.4916
 8128/25000 [========>.....................] - ETA: 43s - loss: 7.7930 - accuracy: 0.4918
 8160/25000 [========>.....................] - ETA: 43s - loss: 7.7906 - accuracy: 0.4919
 8192/25000 [========>.....................] - ETA: 43s - loss: 7.7845 - accuracy: 0.4923
 8224/25000 [========>.....................] - ETA: 42s - loss: 7.7841 - accuracy: 0.4923
 8256/25000 [========>.....................] - ETA: 42s - loss: 7.7929 - accuracy: 0.4918
 8288/25000 [========>.....................] - ETA: 42s - loss: 7.7998 - accuracy: 0.4913
 8320/25000 [========>.....................] - ETA: 42s - loss: 7.8030 - accuracy: 0.4911
 8352/25000 [=========>....................] - ETA: 42s - loss: 7.8043 - accuracy: 0.4910
 8384/25000 [=========>....................] - ETA: 42s - loss: 7.8038 - accuracy: 0.4911
 8416/25000 [=========>....................] - ETA: 42s - loss: 7.8051 - accuracy: 0.4910
 8448/25000 [=========>....................] - ETA: 42s - loss: 7.8173 - accuracy: 0.4902
 8480/25000 [=========>....................] - ETA: 42s - loss: 7.8185 - accuracy: 0.4901
 8512/25000 [=========>....................] - ETA: 42s - loss: 7.8107 - accuracy: 0.4906
 8544/25000 [=========>....................] - ETA: 42s - loss: 7.8030 - accuracy: 0.4911
 8576/25000 [=========>....................] - ETA: 42s - loss: 7.8097 - accuracy: 0.4907
 8608/25000 [=========>....................] - ETA: 41s - loss: 7.8109 - accuracy: 0.4906
 8640/25000 [=========>....................] - ETA: 41s - loss: 7.8050 - accuracy: 0.4910
 8672/25000 [=========>....................] - ETA: 41s - loss: 7.8063 - accuracy: 0.4909
 8704/25000 [=========>....................] - ETA: 41s - loss: 7.7970 - accuracy: 0.4915
 8736/25000 [=========>....................] - ETA: 41s - loss: 7.8035 - accuracy: 0.4911
 8768/25000 [=========>....................] - ETA: 41s - loss: 7.8030 - accuracy: 0.4911
 8800/25000 [=========>....................] - ETA: 41s - loss: 7.7990 - accuracy: 0.4914
 8832/25000 [=========>....................] - ETA: 41s - loss: 7.7968 - accuracy: 0.4915
 8864/25000 [=========>....................] - ETA: 41s - loss: 7.7964 - accuracy: 0.4915
 8896/25000 [=========>....................] - ETA: 41s - loss: 7.7924 - accuracy: 0.4918
 8928/25000 [=========>....................] - ETA: 41s - loss: 7.7886 - accuracy: 0.4920
 8960/25000 [=========>....................] - ETA: 41s - loss: 7.7864 - accuracy: 0.4922
 8992/25000 [=========>....................] - ETA: 40s - loss: 7.7860 - accuracy: 0.4922
 9024/25000 [=========>....................] - ETA: 40s - loss: 7.7805 - accuracy: 0.4926
 9056/25000 [=========>....................] - ETA: 40s - loss: 7.7750 - accuracy: 0.4929
 9088/25000 [=========>....................] - ETA: 40s - loss: 7.7712 - accuracy: 0.4932
 9120/25000 [=========>....................] - ETA: 40s - loss: 7.7709 - accuracy: 0.4932
 9152/25000 [=========>....................] - ETA: 40s - loss: 7.7722 - accuracy: 0.4931
 9184/25000 [==========>...................] - ETA: 40s - loss: 7.7685 - accuracy: 0.4934
 9216/25000 [==========>...................] - ETA: 40s - loss: 7.7664 - accuracy: 0.4935
 9248/25000 [==========>...................] - ETA: 40s - loss: 7.7595 - accuracy: 0.4939
 9280/25000 [==========>...................] - ETA: 40s - loss: 7.7509 - accuracy: 0.4945
 9312/25000 [==========>...................] - ETA: 40s - loss: 7.7539 - accuracy: 0.4943
 9344/25000 [==========>...................] - ETA: 40s - loss: 7.7487 - accuracy: 0.4946
 9376/25000 [==========>...................] - ETA: 39s - loss: 7.7533 - accuracy: 0.4943
 9408/25000 [==========>...................] - ETA: 39s - loss: 7.7481 - accuracy: 0.4947
 9440/25000 [==========>...................] - ETA: 39s - loss: 7.7527 - accuracy: 0.4944
 9472/25000 [==========>...................] - ETA: 39s - loss: 7.7605 - accuracy: 0.4939
 9504/25000 [==========>...................] - ETA: 39s - loss: 7.7505 - accuracy: 0.4945
 9536/25000 [==========>...................] - ETA: 39s - loss: 7.7486 - accuracy: 0.4947
 9568/25000 [==========>...................] - ETA: 39s - loss: 7.7435 - accuracy: 0.4950
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.7513 - accuracy: 0.4945
 9632/25000 [==========>...................] - ETA: 39s - loss: 7.7542 - accuracy: 0.4943
 9664/25000 [==========>...................] - ETA: 39s - loss: 7.7460 - accuracy: 0.4948
 9696/25000 [==========>...................] - ETA: 39s - loss: 7.7504 - accuracy: 0.4945
 9728/25000 [==========>...................] - ETA: 39s - loss: 7.7580 - accuracy: 0.4940
 9760/25000 [==========>...................] - ETA: 38s - loss: 7.7593 - accuracy: 0.4940
 9792/25000 [==========>...................] - ETA: 38s - loss: 7.7668 - accuracy: 0.4935
 9824/25000 [==========>...................] - ETA: 38s - loss: 7.7712 - accuracy: 0.4932
 9856/25000 [==========>...................] - ETA: 38s - loss: 7.7631 - accuracy: 0.4937
 9888/25000 [==========>...................] - ETA: 38s - loss: 7.7721 - accuracy: 0.4931
 9920/25000 [==========>...................] - ETA: 38s - loss: 7.7686 - accuracy: 0.4933
 9952/25000 [==========>...................] - ETA: 38s - loss: 7.7652 - accuracy: 0.4936
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.7711 - accuracy: 0.4932
10016/25000 [===========>..................] - ETA: 38s - loss: 7.7707 - accuracy: 0.4932
10048/25000 [===========>..................] - ETA: 38s - loss: 7.7734 - accuracy: 0.4930
10080/25000 [===========>..................] - ETA: 38s - loss: 7.7731 - accuracy: 0.4931
10112/25000 [===========>..................] - ETA: 38s - loss: 7.7728 - accuracy: 0.4931
10144/25000 [===========>..................] - ETA: 37s - loss: 7.7694 - accuracy: 0.4933
10176/25000 [===========>..................] - ETA: 37s - loss: 7.7766 - accuracy: 0.4928
10208/25000 [===========>..................] - ETA: 37s - loss: 7.7763 - accuracy: 0.4928
10240/25000 [===========>..................] - ETA: 37s - loss: 7.7774 - accuracy: 0.4928
10272/25000 [===========>..................] - ETA: 37s - loss: 7.7771 - accuracy: 0.4928
10304/25000 [===========>..................] - ETA: 37s - loss: 7.7767 - accuracy: 0.4928
10336/25000 [===========>..................] - ETA: 37s - loss: 7.7764 - accuracy: 0.4928
10368/25000 [===========>..................] - ETA: 37s - loss: 7.7805 - accuracy: 0.4926
10400/25000 [===========>..................] - ETA: 37s - loss: 7.7801 - accuracy: 0.4926
10432/25000 [===========>..................] - ETA: 37s - loss: 7.7798 - accuracy: 0.4926
10464/25000 [===========>..................] - ETA: 37s - loss: 7.7795 - accuracy: 0.4926
10496/25000 [===========>..................] - ETA: 37s - loss: 7.7879 - accuracy: 0.4921
10528/25000 [===========>..................] - ETA: 36s - loss: 7.7919 - accuracy: 0.4918
10560/25000 [===========>..................] - ETA: 36s - loss: 7.7958 - accuracy: 0.4916
10592/25000 [===========>..................] - ETA: 36s - loss: 7.7897 - accuracy: 0.4920
10624/25000 [===========>..................] - ETA: 36s - loss: 7.7922 - accuracy: 0.4918
10656/25000 [===========>..................] - ETA: 36s - loss: 7.7875 - accuracy: 0.4921
10688/25000 [===========>..................] - ETA: 36s - loss: 7.7886 - accuracy: 0.4920
10720/25000 [===========>..................] - ETA: 36s - loss: 7.7882 - accuracy: 0.4921
10752/25000 [===========>..................] - ETA: 36s - loss: 7.7850 - accuracy: 0.4923
10784/25000 [===========>..................] - ETA: 36s - loss: 7.7846 - accuracy: 0.4923
10816/25000 [===========>..................] - ETA: 36s - loss: 7.7829 - accuracy: 0.4924
10848/25000 [============>.................] - ETA: 36s - loss: 7.7839 - accuracy: 0.4923
10880/25000 [============>.................] - ETA: 36s - loss: 7.7822 - accuracy: 0.4925
10912/25000 [============>.................] - ETA: 36s - loss: 7.7832 - accuracy: 0.4924
10944/25000 [============>.................] - ETA: 35s - loss: 7.7857 - accuracy: 0.4922
10976/25000 [============>.................] - ETA: 35s - loss: 7.7840 - accuracy: 0.4923
11008/25000 [============>.................] - ETA: 35s - loss: 7.7906 - accuracy: 0.4919
11040/25000 [============>.................] - ETA: 35s - loss: 7.7805 - accuracy: 0.4926
11072/25000 [============>.................] - ETA: 35s - loss: 7.7774 - accuracy: 0.4928
11104/25000 [============>.................] - ETA: 35s - loss: 7.7785 - accuracy: 0.4927
11136/25000 [============>.................] - ETA: 35s - loss: 7.7795 - accuracy: 0.4926
11168/25000 [============>.................] - ETA: 35s - loss: 7.7765 - accuracy: 0.4928
11200/25000 [============>.................] - ETA: 35s - loss: 7.7707 - accuracy: 0.4932
11232/25000 [============>.................] - ETA: 35s - loss: 7.7758 - accuracy: 0.4929
11264/25000 [============>.................] - ETA: 35s - loss: 7.7810 - accuracy: 0.4925
11296/25000 [============>.................] - ETA: 35s - loss: 7.7820 - accuracy: 0.4925
11328/25000 [============>.................] - ETA: 34s - loss: 7.7803 - accuracy: 0.4926
11360/25000 [============>.................] - ETA: 34s - loss: 7.7786 - accuracy: 0.4927
11392/25000 [============>.................] - ETA: 34s - loss: 7.7783 - accuracy: 0.4927
11424/25000 [============>.................] - ETA: 34s - loss: 7.7713 - accuracy: 0.4932
11456/25000 [============>.................] - ETA: 34s - loss: 7.7737 - accuracy: 0.4930
11488/25000 [============>.................] - ETA: 34s - loss: 7.7641 - accuracy: 0.4936
11520/25000 [============>.................] - ETA: 34s - loss: 7.7638 - accuracy: 0.4937
11552/25000 [============>.................] - ETA: 34s - loss: 7.7622 - accuracy: 0.4938
11584/25000 [============>.................] - ETA: 34s - loss: 7.7619 - accuracy: 0.4938
11616/25000 [============>.................] - ETA: 34s - loss: 7.7617 - accuracy: 0.4938
11648/25000 [============>.................] - ETA: 34s - loss: 7.7614 - accuracy: 0.4938
11680/25000 [=============>................] - ETA: 34s - loss: 7.7611 - accuracy: 0.4938
11712/25000 [=============>................] - ETA: 33s - loss: 7.7543 - accuracy: 0.4943
11744/25000 [=============>................] - ETA: 33s - loss: 7.7619 - accuracy: 0.4938
11776/25000 [=============>................] - ETA: 33s - loss: 7.7617 - accuracy: 0.4938
11808/25000 [=============>................] - ETA: 33s - loss: 7.7627 - accuracy: 0.4937
11840/25000 [=============>................] - ETA: 33s - loss: 7.7663 - accuracy: 0.4935
11872/25000 [=============>................] - ETA: 33s - loss: 7.7635 - accuracy: 0.4937
11904/25000 [=============>................] - ETA: 33s - loss: 7.7671 - accuracy: 0.4934
11936/25000 [=============>................] - ETA: 33s - loss: 7.7668 - accuracy: 0.4935
11968/25000 [=============>................] - ETA: 33s - loss: 7.7576 - accuracy: 0.4941
12000/25000 [=============>................] - ETA: 33s - loss: 7.7561 - accuracy: 0.4942
12032/25000 [=============>................] - ETA: 33s - loss: 7.7507 - accuracy: 0.4945
12064/25000 [=============>................] - ETA: 33s - loss: 7.7429 - accuracy: 0.4950
12096/25000 [=============>................] - ETA: 32s - loss: 7.7427 - accuracy: 0.4950
12128/25000 [=============>................] - ETA: 32s - loss: 7.7412 - accuracy: 0.4951
12160/25000 [=============>................] - ETA: 32s - loss: 7.7398 - accuracy: 0.4952
12192/25000 [=============>................] - ETA: 32s - loss: 7.7421 - accuracy: 0.4951
12224/25000 [=============>................] - ETA: 32s - loss: 7.7419 - accuracy: 0.4951
12256/25000 [=============>................] - ETA: 32s - loss: 7.7429 - accuracy: 0.4950
12288/25000 [=============>................] - ETA: 32s - loss: 7.7427 - accuracy: 0.4950
12320/25000 [=============>................] - ETA: 32s - loss: 7.7413 - accuracy: 0.4951
12352/25000 [=============>................] - ETA: 32s - loss: 7.7337 - accuracy: 0.4956
12384/25000 [=============>................] - ETA: 32s - loss: 7.7322 - accuracy: 0.4957
12416/25000 [=============>................] - ETA: 32s - loss: 7.7296 - accuracy: 0.4959
12448/25000 [=============>................] - ETA: 32s - loss: 7.7294 - accuracy: 0.4959
12480/25000 [=============>................] - ETA: 31s - loss: 7.7281 - accuracy: 0.4960
12512/25000 [==============>...............] - ETA: 31s - loss: 7.7279 - accuracy: 0.4960
12544/25000 [==============>...............] - ETA: 31s - loss: 7.7228 - accuracy: 0.4963
12576/25000 [==============>...............] - ETA: 31s - loss: 7.7203 - accuracy: 0.4965
12608/25000 [==============>...............] - ETA: 31s - loss: 7.7201 - accuracy: 0.4965
12640/25000 [==============>...............] - ETA: 31s - loss: 7.7200 - accuracy: 0.4965
12672/25000 [==============>...............] - ETA: 31s - loss: 7.7186 - accuracy: 0.4966
12704/25000 [==============>...............] - ETA: 31s - loss: 7.7233 - accuracy: 0.4963
12736/25000 [==============>...............] - ETA: 31s - loss: 7.7196 - accuracy: 0.4965
12768/25000 [==============>...............] - ETA: 31s - loss: 7.7255 - accuracy: 0.4962
12800/25000 [==============>...............] - ETA: 31s - loss: 7.7301 - accuracy: 0.4959
12832/25000 [==============>...............] - ETA: 31s - loss: 7.7323 - accuracy: 0.4957
12864/25000 [==============>...............] - ETA: 31s - loss: 7.7346 - accuracy: 0.4956
12896/25000 [==============>...............] - ETA: 30s - loss: 7.7368 - accuracy: 0.4954
12928/25000 [==============>...............] - ETA: 30s - loss: 7.7342 - accuracy: 0.4956
12960/25000 [==============>...............] - ETA: 30s - loss: 7.7305 - accuracy: 0.4958
12992/25000 [==============>...............] - ETA: 30s - loss: 7.7339 - accuracy: 0.4956
13024/25000 [==============>...............] - ETA: 30s - loss: 7.7384 - accuracy: 0.4953
13056/25000 [==============>...............] - ETA: 30s - loss: 7.7336 - accuracy: 0.4956
13088/25000 [==============>...............] - ETA: 30s - loss: 7.7381 - accuracy: 0.4953
13120/25000 [==============>...............] - ETA: 30s - loss: 7.7402 - accuracy: 0.4952
13152/25000 [==============>...............] - ETA: 30s - loss: 7.7401 - accuracy: 0.4952
13184/25000 [==============>...............] - ETA: 30s - loss: 7.7387 - accuracy: 0.4953
13216/25000 [==============>...............] - ETA: 30s - loss: 7.7339 - accuracy: 0.4956
13248/25000 [==============>...............] - ETA: 30s - loss: 7.7361 - accuracy: 0.4955
13280/25000 [==============>...............] - ETA: 29s - loss: 7.7313 - accuracy: 0.4958
13312/25000 [==============>...............] - ETA: 29s - loss: 7.7334 - accuracy: 0.4956
13344/25000 [===============>..............] - ETA: 29s - loss: 7.7356 - accuracy: 0.4955
13376/25000 [===============>..............] - ETA: 29s - loss: 7.7331 - accuracy: 0.4957
13408/25000 [===============>..............] - ETA: 29s - loss: 7.7352 - accuracy: 0.4955
13440/25000 [===============>..............] - ETA: 29s - loss: 7.7419 - accuracy: 0.4951
13472/25000 [===============>..............] - ETA: 29s - loss: 7.7383 - accuracy: 0.4953
13504/25000 [===============>..............] - ETA: 29s - loss: 7.7370 - accuracy: 0.4954
13536/25000 [===============>..............] - ETA: 29s - loss: 7.7436 - accuracy: 0.4950
13568/25000 [===============>..............] - ETA: 29s - loss: 7.7435 - accuracy: 0.4950
13600/25000 [===============>..............] - ETA: 29s - loss: 7.7410 - accuracy: 0.4951
13632/25000 [===============>..............] - ETA: 29s - loss: 7.7397 - accuracy: 0.4952
13664/25000 [===============>..............] - ETA: 28s - loss: 7.7396 - accuracy: 0.4952
13696/25000 [===============>..............] - ETA: 28s - loss: 7.7416 - accuracy: 0.4951
13728/25000 [===============>..............] - ETA: 28s - loss: 7.7415 - accuracy: 0.4951
13760/25000 [===============>..............] - ETA: 28s - loss: 7.7391 - accuracy: 0.4953
13792/25000 [===============>..............] - ETA: 28s - loss: 7.7378 - accuracy: 0.4954
13824/25000 [===============>..............] - ETA: 28s - loss: 7.7376 - accuracy: 0.4954
13856/25000 [===============>..............] - ETA: 28s - loss: 7.7341 - accuracy: 0.4956
13888/25000 [===============>..............] - ETA: 28s - loss: 7.7296 - accuracy: 0.4959
13920/25000 [===============>..............] - ETA: 28s - loss: 7.7261 - accuracy: 0.4961
13952/25000 [===============>..............] - ETA: 28s - loss: 7.7194 - accuracy: 0.4966
13984/25000 [===============>..............] - ETA: 28s - loss: 7.7214 - accuracy: 0.4964
14016/25000 [===============>..............] - ETA: 28s - loss: 7.7158 - accuracy: 0.4968
14048/25000 [===============>..............] - ETA: 27s - loss: 7.7157 - accuracy: 0.4968
14080/25000 [===============>..............] - ETA: 27s - loss: 7.7167 - accuracy: 0.4967
14112/25000 [===============>..............] - ETA: 27s - loss: 7.7166 - accuracy: 0.4967
14144/25000 [===============>..............] - ETA: 27s - loss: 7.7176 - accuracy: 0.4967
14176/25000 [================>.............] - ETA: 27s - loss: 7.7185 - accuracy: 0.4966
14208/25000 [================>.............] - ETA: 27s - loss: 7.7173 - accuracy: 0.4967
14240/25000 [================>.............] - ETA: 27s - loss: 7.7172 - accuracy: 0.4967
14272/25000 [================>.............] - ETA: 27s - loss: 7.7160 - accuracy: 0.4968
14304/25000 [================>.............] - ETA: 27s - loss: 7.7191 - accuracy: 0.4966
14336/25000 [================>.............] - ETA: 27s - loss: 7.7180 - accuracy: 0.4967
14368/25000 [================>.............] - ETA: 27s - loss: 7.7200 - accuracy: 0.4965
14400/25000 [================>.............] - ETA: 27s - loss: 7.7209 - accuracy: 0.4965
14432/25000 [================>.............] - ETA: 26s - loss: 7.7208 - accuracy: 0.4965
14464/25000 [================>.............] - ETA: 26s - loss: 7.7111 - accuracy: 0.4971
14496/25000 [================>.............] - ETA: 26s - loss: 7.7079 - accuracy: 0.4973
14528/25000 [================>.............] - ETA: 26s - loss: 7.7036 - accuracy: 0.4976
14560/25000 [================>.............] - ETA: 26s - loss: 7.7045 - accuracy: 0.4975
14592/25000 [================>.............] - ETA: 26s - loss: 7.7087 - accuracy: 0.4973
14624/25000 [================>.............] - ETA: 26s - loss: 7.7117 - accuracy: 0.4971
14656/25000 [================>.............] - ETA: 26s - loss: 7.7106 - accuracy: 0.4971
14688/25000 [================>.............] - ETA: 26s - loss: 7.7126 - accuracy: 0.4970
14720/25000 [================>.............] - ETA: 26s - loss: 7.7135 - accuracy: 0.4969
14752/25000 [================>.............] - ETA: 26s - loss: 7.7134 - accuracy: 0.4969
14784/25000 [================>.............] - ETA: 26s - loss: 7.7112 - accuracy: 0.4971
14816/25000 [================>.............] - ETA: 25s - loss: 7.7132 - accuracy: 0.4970
14848/25000 [================>.............] - ETA: 25s - loss: 7.7141 - accuracy: 0.4969
14880/25000 [================>.............] - ETA: 25s - loss: 7.7151 - accuracy: 0.4968
14912/25000 [================>.............] - ETA: 25s - loss: 7.7149 - accuracy: 0.4968
14944/25000 [================>.............] - ETA: 25s - loss: 7.7159 - accuracy: 0.4968
14976/25000 [================>.............] - ETA: 25s - loss: 7.7188 - accuracy: 0.4966
15008/25000 [=================>............] - ETA: 25s - loss: 7.7208 - accuracy: 0.4965
15040/25000 [=================>............] - ETA: 25s - loss: 7.7217 - accuracy: 0.4964
15072/25000 [=================>............] - ETA: 25s - loss: 7.7205 - accuracy: 0.4965
15104/25000 [=================>............] - ETA: 25s - loss: 7.7214 - accuracy: 0.4964
15136/25000 [=================>............] - ETA: 25s - loss: 7.7233 - accuracy: 0.4963
15168/25000 [=================>............] - ETA: 25s - loss: 7.7273 - accuracy: 0.4960
15200/25000 [=================>............] - ETA: 24s - loss: 7.7352 - accuracy: 0.4955
15232/25000 [=================>............] - ETA: 24s - loss: 7.7381 - accuracy: 0.4953
15264/25000 [=================>............] - ETA: 24s - loss: 7.7420 - accuracy: 0.4951
15296/25000 [=================>............] - ETA: 24s - loss: 7.7428 - accuracy: 0.4950
15328/25000 [=================>............] - ETA: 24s - loss: 7.7436 - accuracy: 0.4950
15360/25000 [=================>............] - ETA: 24s - loss: 7.7385 - accuracy: 0.4953
15392/25000 [=================>............] - ETA: 24s - loss: 7.7393 - accuracy: 0.4953
15424/25000 [=================>............] - ETA: 24s - loss: 7.7352 - accuracy: 0.4955
15456/25000 [=================>............] - ETA: 24s - loss: 7.7361 - accuracy: 0.4955
15488/25000 [=================>............] - ETA: 24s - loss: 7.7369 - accuracy: 0.4954
15520/25000 [=================>............] - ETA: 24s - loss: 7.7378 - accuracy: 0.4954
15552/25000 [=================>............] - ETA: 24s - loss: 7.7337 - accuracy: 0.4956
15584/25000 [=================>............] - ETA: 23s - loss: 7.7375 - accuracy: 0.4954
15616/25000 [=================>............] - ETA: 23s - loss: 7.7363 - accuracy: 0.4955
15648/25000 [=================>............] - ETA: 23s - loss: 7.7382 - accuracy: 0.4953
15680/25000 [=================>............] - ETA: 23s - loss: 7.7429 - accuracy: 0.4950
15712/25000 [=================>............] - ETA: 23s - loss: 7.7369 - accuracy: 0.4954
15744/25000 [=================>............] - ETA: 23s - loss: 7.7387 - accuracy: 0.4953
15776/25000 [=================>............] - ETA: 23s - loss: 7.7444 - accuracy: 0.4949
15808/25000 [=================>............] - ETA: 23s - loss: 7.7423 - accuracy: 0.4951
15840/25000 [==================>...........] - ETA: 23s - loss: 7.7441 - accuracy: 0.4949
15872/25000 [==================>...........] - ETA: 23s - loss: 7.7400 - accuracy: 0.4952
15904/25000 [==================>...........] - ETA: 23s - loss: 7.7389 - accuracy: 0.4953
15936/25000 [==================>...........] - ETA: 23s - loss: 7.7407 - accuracy: 0.4952
15968/25000 [==================>...........] - ETA: 23s - loss: 7.7367 - accuracy: 0.4954
16000/25000 [==================>...........] - ETA: 22s - loss: 7.7327 - accuracy: 0.4957
16032/25000 [==================>...........] - ETA: 22s - loss: 7.7317 - accuracy: 0.4958
16064/25000 [==================>...........] - ETA: 22s - loss: 7.7277 - accuracy: 0.4960
16096/25000 [==================>...........] - ETA: 22s - loss: 7.7304 - accuracy: 0.4958
16128/25000 [==================>...........] - ETA: 22s - loss: 7.7275 - accuracy: 0.4960
16160/25000 [==================>...........] - ETA: 22s - loss: 7.7254 - accuracy: 0.4962
16192/25000 [==================>...........] - ETA: 22s - loss: 7.7234 - accuracy: 0.4963
16224/25000 [==================>...........] - ETA: 22s - loss: 7.7205 - accuracy: 0.4965
16256/25000 [==================>...........] - ETA: 22s - loss: 7.7194 - accuracy: 0.4966
16288/25000 [==================>...........] - ETA: 22s - loss: 7.7175 - accuracy: 0.4967
16320/25000 [==================>...........] - ETA: 22s - loss: 7.7136 - accuracy: 0.4969
16352/25000 [==================>...........] - ETA: 22s - loss: 7.7126 - accuracy: 0.4970
16384/25000 [==================>...........] - ETA: 21s - loss: 7.7125 - accuracy: 0.4970
16416/25000 [==================>...........] - ETA: 21s - loss: 7.7133 - accuracy: 0.4970
16448/25000 [==================>...........] - ETA: 21s - loss: 7.7179 - accuracy: 0.4967
16480/25000 [==================>...........] - ETA: 21s - loss: 7.7187 - accuracy: 0.4966
16512/25000 [==================>...........] - ETA: 21s - loss: 7.7177 - accuracy: 0.4967
16544/25000 [==================>...........] - ETA: 21s - loss: 7.7241 - accuracy: 0.4963
16576/25000 [==================>...........] - ETA: 21s - loss: 7.7240 - accuracy: 0.4963
16608/25000 [==================>...........] - ETA: 21s - loss: 7.7202 - accuracy: 0.4965
16640/25000 [==================>...........] - ETA: 21s - loss: 7.7182 - accuracy: 0.4966
16672/25000 [===================>..........] - ETA: 21s - loss: 7.7163 - accuracy: 0.4968
16704/25000 [===================>..........] - ETA: 21s - loss: 7.7153 - accuracy: 0.4968
16736/25000 [===================>..........] - ETA: 21s - loss: 7.7179 - accuracy: 0.4967
16768/25000 [===================>..........] - ETA: 20s - loss: 7.7151 - accuracy: 0.4968
16800/25000 [===================>..........] - ETA: 20s - loss: 7.7168 - accuracy: 0.4967
16832/25000 [===================>..........] - ETA: 20s - loss: 7.7149 - accuracy: 0.4969
16864/25000 [===================>..........] - ETA: 20s - loss: 7.7184 - accuracy: 0.4966
16896/25000 [===================>..........] - ETA: 20s - loss: 7.7174 - accuracy: 0.4967
16928/25000 [===================>..........] - ETA: 20s - loss: 7.7128 - accuracy: 0.4970
16960/25000 [===================>..........] - ETA: 20s - loss: 7.7064 - accuracy: 0.4974
16992/25000 [===================>..........] - ETA: 20s - loss: 7.7054 - accuracy: 0.4975
17024/25000 [===================>..........] - ETA: 20s - loss: 7.7072 - accuracy: 0.4974
17056/25000 [===================>..........] - ETA: 20s - loss: 7.7071 - accuracy: 0.4974
17088/25000 [===================>..........] - ETA: 20s - loss: 7.7061 - accuracy: 0.4974
17120/25000 [===================>..........] - ETA: 20s - loss: 7.7087 - accuracy: 0.4973
17152/25000 [===================>..........] - ETA: 19s - loss: 7.7086 - accuracy: 0.4973
17184/25000 [===================>..........] - ETA: 19s - loss: 7.7094 - accuracy: 0.4972
17216/25000 [===================>..........] - ETA: 19s - loss: 7.7103 - accuracy: 0.4972
17248/25000 [===================>..........] - ETA: 19s - loss: 7.7111 - accuracy: 0.4971
17280/25000 [===================>..........] - ETA: 19s - loss: 7.7128 - accuracy: 0.4970
17312/25000 [===================>..........] - ETA: 19s - loss: 7.7162 - accuracy: 0.4968
17344/25000 [===================>..........] - ETA: 19s - loss: 7.7126 - accuracy: 0.4970
17376/25000 [===================>..........] - ETA: 19s - loss: 7.7125 - accuracy: 0.4970
17408/25000 [===================>..........] - ETA: 19s - loss: 7.7124 - accuracy: 0.4970
17440/25000 [===================>..........] - ETA: 19s - loss: 7.7088 - accuracy: 0.4972
17472/25000 [===================>..........] - ETA: 19s - loss: 7.7061 - accuracy: 0.4974
17504/25000 [====================>.........] - ETA: 19s - loss: 7.7078 - accuracy: 0.4973
17536/25000 [====================>.........] - ETA: 18s - loss: 7.7095 - accuracy: 0.4972
17568/25000 [====================>.........] - ETA: 18s - loss: 7.7103 - accuracy: 0.4972
17600/25000 [====================>.........] - ETA: 18s - loss: 7.7119 - accuracy: 0.4970
17632/25000 [====================>.........] - ETA: 18s - loss: 7.7127 - accuracy: 0.4970
17664/25000 [====================>.........] - ETA: 18s - loss: 7.7100 - accuracy: 0.4972
17696/25000 [====================>.........] - ETA: 18s - loss: 7.7091 - accuracy: 0.4972
17728/25000 [====================>.........] - ETA: 18s - loss: 7.7047 - accuracy: 0.4975
17760/25000 [====================>.........] - ETA: 18s - loss: 7.7072 - accuracy: 0.4974
17792/25000 [====================>.........] - ETA: 18s - loss: 7.7123 - accuracy: 0.4970
17824/25000 [====================>.........] - ETA: 18s - loss: 7.7131 - accuracy: 0.4970
17856/25000 [====================>.........] - ETA: 18s - loss: 7.7130 - accuracy: 0.4970
17888/25000 [====================>.........] - ETA: 18s - loss: 7.7129 - accuracy: 0.4970
17920/25000 [====================>.........] - ETA: 18s - loss: 7.7145 - accuracy: 0.4969
17952/25000 [====================>.........] - ETA: 17s - loss: 7.7127 - accuracy: 0.4970
17984/25000 [====================>.........] - ETA: 17s - loss: 7.7135 - accuracy: 0.4969
18016/25000 [====================>.........] - ETA: 17s - loss: 7.7160 - accuracy: 0.4968
18048/25000 [====================>.........] - ETA: 17s - loss: 7.7142 - accuracy: 0.4969
18080/25000 [====================>.........] - ETA: 17s - loss: 7.7124 - accuracy: 0.4970
18112/25000 [====================>.........] - ETA: 17s - loss: 7.7081 - accuracy: 0.4973
18144/25000 [====================>.........] - ETA: 17s - loss: 7.7080 - accuracy: 0.4973
18176/25000 [====================>.........] - ETA: 17s - loss: 7.7088 - accuracy: 0.4972
18208/25000 [====================>.........] - ETA: 17s - loss: 7.7079 - accuracy: 0.4973
18240/25000 [====================>.........] - ETA: 17s - loss: 7.7070 - accuracy: 0.4974
18272/25000 [====================>.........] - ETA: 17s - loss: 7.7044 - accuracy: 0.4975
18304/25000 [====================>.........] - ETA: 17s - loss: 7.7035 - accuracy: 0.4976
18336/25000 [=====================>........] - ETA: 16s - loss: 7.7043 - accuracy: 0.4975
18368/25000 [=====================>........] - ETA: 16s - loss: 7.7059 - accuracy: 0.4974
18400/25000 [=====================>........] - ETA: 16s - loss: 7.7041 - accuracy: 0.4976
18432/25000 [=====================>........] - ETA: 16s - loss: 7.7016 - accuracy: 0.4977
18464/25000 [=====================>........] - ETA: 16s - loss: 7.7015 - accuracy: 0.4977
18496/25000 [=====================>........] - ETA: 16s - loss: 7.7006 - accuracy: 0.4978
18528/25000 [=====================>........] - ETA: 16s - loss: 7.7055 - accuracy: 0.4975
18560/25000 [=====================>........] - ETA: 16s - loss: 7.7046 - accuracy: 0.4975
18592/25000 [=====================>........] - ETA: 16s - loss: 7.7046 - accuracy: 0.4975
18624/25000 [=====================>........] - ETA: 16s - loss: 7.7045 - accuracy: 0.4975
18656/25000 [=====================>........] - ETA: 16s - loss: 7.7036 - accuracy: 0.4976
18688/25000 [=====================>........] - ETA: 16s - loss: 7.7044 - accuracy: 0.4975
18720/25000 [=====================>........] - ETA: 16s - loss: 7.7076 - accuracy: 0.4973
18752/25000 [=====================>........] - ETA: 15s - loss: 7.7083 - accuracy: 0.4973
18784/25000 [=====================>........] - ETA: 15s - loss: 7.7083 - accuracy: 0.4973
18816/25000 [=====================>........] - ETA: 15s - loss: 7.7098 - accuracy: 0.4972
18848/25000 [=====================>........] - ETA: 15s - loss: 7.7089 - accuracy: 0.4972
18880/25000 [=====================>........] - ETA: 15s - loss: 7.7056 - accuracy: 0.4975
18912/25000 [=====================>........] - ETA: 15s - loss: 7.7055 - accuracy: 0.4975
18944/25000 [=====================>........] - ETA: 15s - loss: 7.7022 - accuracy: 0.4977
18976/25000 [=====================>........] - ETA: 15s - loss: 7.7046 - accuracy: 0.4975
19008/25000 [=====================>........] - ETA: 15s - loss: 7.7094 - accuracy: 0.4972
19040/25000 [=====================>........] - ETA: 15s - loss: 7.7085 - accuracy: 0.4973
19072/25000 [=====================>........] - ETA: 15s - loss: 7.7092 - accuracy: 0.4972
19104/25000 [=====================>........] - ETA: 15s - loss: 7.7084 - accuracy: 0.4973
19136/25000 [=====================>........] - ETA: 14s - loss: 7.7115 - accuracy: 0.4971
19168/25000 [======================>.......] - ETA: 14s - loss: 7.7074 - accuracy: 0.4973
19200/25000 [======================>.......] - ETA: 14s - loss: 7.7097 - accuracy: 0.4972
19232/25000 [======================>.......] - ETA: 14s - loss: 7.7113 - accuracy: 0.4971
19264/25000 [======================>.......] - ETA: 14s - loss: 7.7112 - accuracy: 0.4971
19296/25000 [======================>.......] - ETA: 14s - loss: 7.7087 - accuracy: 0.4973
19328/25000 [======================>.......] - ETA: 14s - loss: 7.7079 - accuracy: 0.4973
19360/25000 [======================>.......] - ETA: 14s - loss: 7.7070 - accuracy: 0.4974
19392/25000 [======================>.......] - ETA: 14s - loss: 7.7022 - accuracy: 0.4977
19424/25000 [======================>.......] - ETA: 14s - loss: 7.7053 - accuracy: 0.4975
19456/25000 [======================>.......] - ETA: 14s - loss: 7.7068 - accuracy: 0.4974
19488/25000 [======================>.......] - ETA: 14s - loss: 7.7091 - accuracy: 0.4972
19520/25000 [======================>.......] - ETA: 13s - loss: 7.7075 - accuracy: 0.4973
19552/25000 [======================>.......] - ETA: 13s - loss: 7.7035 - accuracy: 0.4976
19584/25000 [======================>.......] - ETA: 13s - loss: 7.7026 - accuracy: 0.4977
19616/25000 [======================>.......] - ETA: 13s - loss: 7.7018 - accuracy: 0.4977
19648/25000 [======================>.......] - ETA: 13s - loss: 7.7056 - accuracy: 0.4975
19680/25000 [======================>.......] - ETA: 13s - loss: 7.7064 - accuracy: 0.4974
19712/25000 [======================>.......] - ETA: 13s - loss: 7.7078 - accuracy: 0.4973
19744/25000 [======================>.......] - ETA: 13s - loss: 7.7109 - accuracy: 0.4971
19776/25000 [======================>.......] - ETA: 13s - loss: 7.7139 - accuracy: 0.4969
19808/25000 [======================>.......] - ETA: 13s - loss: 7.7154 - accuracy: 0.4968
19840/25000 [======================>.......] - ETA: 13s - loss: 7.7153 - accuracy: 0.4968
19872/25000 [======================>.......] - ETA: 13s - loss: 7.7191 - accuracy: 0.4966
19904/25000 [======================>.......] - ETA: 12s - loss: 7.7167 - accuracy: 0.4967
19936/25000 [======================>.......] - ETA: 12s - loss: 7.7143 - accuracy: 0.4969
19968/25000 [======================>.......] - ETA: 12s - loss: 7.7112 - accuracy: 0.4971
20000/25000 [=======================>......] - ETA: 12s - loss: 7.7065 - accuracy: 0.4974
20032/25000 [=======================>......] - ETA: 12s - loss: 7.7080 - accuracy: 0.4973
20064/25000 [=======================>......] - ETA: 12s - loss: 7.7094 - accuracy: 0.4972
20096/25000 [=======================>......] - ETA: 12s - loss: 7.7078 - accuracy: 0.4973
20128/25000 [=======================>......] - ETA: 12s - loss: 7.7078 - accuracy: 0.4973
20160/25000 [=======================>......] - ETA: 12s - loss: 7.7054 - accuracy: 0.4975
20192/25000 [=======================>......] - ETA: 12s - loss: 7.7053 - accuracy: 0.4975
20224/25000 [=======================>......] - ETA: 12s - loss: 7.7038 - accuracy: 0.4976
20256/25000 [=======================>......] - ETA: 12s - loss: 7.7045 - accuracy: 0.4975
20288/25000 [=======================>......] - ETA: 11s - loss: 7.7006 - accuracy: 0.4978
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6991 - accuracy: 0.4979
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6990 - accuracy: 0.4979
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6967 - accuracy: 0.4980
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6974 - accuracy: 0.4980
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6974 - accuracy: 0.4980
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6981 - accuracy: 0.4979
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6995 - accuracy: 0.4979
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6965 - accuracy: 0.4981
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6934 - accuracy: 0.4983
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6934 - accuracy: 0.4983
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6934 - accuracy: 0.4983
20672/25000 [=======================>......] - ETA: 11s - loss: 7.6970 - accuracy: 0.4980
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6940 - accuracy: 0.4982
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6918 - accuracy: 0.4984
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6910 - accuracy: 0.4984
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6909 - accuracy: 0.4984
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6924 - accuracy: 0.4983
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6916 - accuracy: 0.4984
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6945 - accuracy: 0.4982
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6967 - accuracy: 0.4980
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6981 - accuracy: 0.4979
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6988 - accuracy: 0.4979
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6980 - accuracy: 0.4980
21056/25000 [========================>.....] - ETA: 10s - loss: 7.7008 - accuracy: 0.4978
21088/25000 [========================>.....] - ETA: 9s - loss: 7.7037 - accuracy: 0.4976 
21120/25000 [========================>.....] - ETA: 9s - loss: 7.7029 - accuracy: 0.4976
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6992 - accuracy: 0.4979
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6941 - accuracy: 0.4982
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6912 - accuracy: 0.4984
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6904 - accuracy: 0.4984
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6904 - accuracy: 0.4984
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6860 - accuracy: 0.4987
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6846 - accuracy: 0.4988
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6853 - accuracy: 0.4988
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6860 - accuracy: 0.4987
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6852 - accuracy: 0.4988
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6838 - accuracy: 0.4989
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6844 - accuracy: 0.4988
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6851 - accuracy: 0.4988
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6851 - accuracy: 0.4988
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6886 - accuracy: 0.4986
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6858 - accuracy: 0.4988
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6836 - accuracy: 0.4989
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6822 - accuracy: 0.4990
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6814 - accuracy: 0.4990
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6821 - accuracy: 0.4990
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6828 - accuracy: 0.4989
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6807 - accuracy: 0.4991
21856/25000 [=========================>....] - ETA: 8s - loss: 7.6814 - accuracy: 0.4990
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6813 - accuracy: 0.4990
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6799 - accuracy: 0.4991
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6813 - accuracy: 0.4990
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6813 - accuracy: 0.4990
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6826 - accuracy: 0.4990
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6819 - accuracy: 0.4990
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6798 - accuracy: 0.4991
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6805 - accuracy: 0.4991
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6812 - accuracy: 0.4991
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6804 - accuracy: 0.4991
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6797 - accuracy: 0.4991
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6797 - accuracy: 0.4991
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6763 - accuracy: 0.4994
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6769 - accuracy: 0.4993
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6742 - accuracy: 0.4995
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6742 - accuracy: 0.4995
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6748 - accuracy: 0.4995
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6776 - accuracy: 0.4993
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6782 - accuracy: 0.4992
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6803 - accuracy: 0.4991
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6802 - accuracy: 0.4991
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6809 - accuracy: 0.4991
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6822 - accuracy: 0.4990
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6815 - accuracy: 0.4990
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6802 - accuracy: 0.4991
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6815 - accuracy: 0.4990
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6828 - accuracy: 0.4989
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6855 - accuracy: 0.4988
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6841 - accuracy: 0.4989
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6827 - accuracy: 0.4989
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6800 - accuracy: 0.4991
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6820 - accuracy: 0.4990
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6800 - accuracy: 0.4991
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6807 - accuracy: 0.4991
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6800 - accuracy: 0.4991
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6806 - accuracy: 0.4991
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6806 - accuracy: 0.4991
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6792 - accuracy: 0.4992
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6832 - accuracy: 0.4989
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6812 - accuracy: 0.4990
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6805 - accuracy: 0.4991
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6798 - accuracy: 0.4991
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6785 - accuracy: 0.4992
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6791 - accuracy: 0.4992
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6811 - accuracy: 0.4991
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6811 - accuracy: 0.4991
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6804 - accuracy: 0.4991
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6797 - accuracy: 0.4991
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6804 - accuracy: 0.4991
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6803 - accuracy: 0.4991
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6810 - accuracy: 0.4991
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6849 - accuracy: 0.4988
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6842 - accuracy: 0.4989
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6809 - accuracy: 0.4991
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6777 - accuracy: 0.4993
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6757 - accuracy: 0.4994
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6737 - accuracy: 0.4995
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6724 - accuracy: 0.4996
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6711 - accuracy: 0.4997
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6622 - accuracy: 0.5003
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6628 - accuracy: 0.5002
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24192/25000 [============================>.] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
24224/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24256/25000 [============================>.] - ETA: 1s - loss: 7.6673 - accuracy: 0.5000
24288/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24320/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24352/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24384/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24416/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24448/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24480/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24512/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24544/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24576/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24608/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24640/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24672/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24704/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24736/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24768/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24800/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24832/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24864/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24896/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24960/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fa6be1029e8> 

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
[[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00]
 [ 8.94751921e-02 -1.87919214e-02  7.94546604e-02  1.07407473e-01
   1.03573985e-02  2.29958072e-02]
 [-1.25965551e-01  6.68798313e-02 -2.63543520e-02  1.12547711e-01
  -4.02143821e-02  3.04396003e-02]
 [ 8.51878896e-02 -9.05381069e-02 -2.08156519e-02  6.60183504e-02
  -1.12451157e-02  1.16730265e-01]
 [ 5.42190135e-01  4.25270572e-02  2.66687840e-01  4.01804689e-04
   9.40652192e-02 -1.92919403e-01]
 [ 2.75072336e-01  2.99311988e-02  3.28245647e-02  2.31140733e-01
  -2.50324845e-01  1.53481394e-01]
 [ 1.10192709e-02 -1.27668649e-01 -1.91641569e-01 -2.52777517e-01
   1.44328758e-01  1.26549020e-01]
 [-1.52307704e-01 -1.30651712e-01  1.48327604e-01  3.34384829e-01
  -3.33901256e-01 -8.72053280e-02]
 [ 9.09582675e-02  7.66155645e-02 -1.95510518e-02 -1.08897164e-01
  -1.77797750e-01 -1.00267321e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00]]

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
{'loss': 0.4438207224011421, 'loss_history': []}

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
{'loss': 0.5683606043457985, 'loss_history': []}

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
 40%|████      | 2/5 [00:50<01:16, 25.34s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.3024436677211062, 'embedding_size_factor': 1.1310484108738286, 'layers.choice': 3, 'learning_rate': 0.002183855816760604, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 0.00018632099451611043} and reward: 0.3814
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd3[<\xafo\xa4\xf0X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2\x18\xc67\xeeUvX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?a\xe3\xe0\xa9\xfdC\xe5X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?(k\xe5'\xf0A\xc4u." and reward: 0.3814
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd3[<\xafo\xa4\xf0X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2\x18\xc67\xeeUvX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?a\xe3\xe0\xa9\xfdC\xe5X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?(k\xe5'\xf0A\xc4u." and reward: 0.3814
 60%|██████    | 3/5 [01:45<01:08, 34.12s/it] 60%|██████    | 3/5 [01:45<01:10, 35.10s/it]
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
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.298475580558665, 'embedding_size_factor': 1.226247051244219, 'layers.choice': 2, 'learning_rate': 0.002863394343831298, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 2.276097943738474e-09} and reward: 0.3706
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd3\x1a9RI\xdcqX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\x9e\xb5:^\x90\xd2X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?gt\xf9!\xfej\xd7X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>#\x8d1;=\xae.u.' and reward: 0.3706
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd3\x1a9RI\xdcqX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\x9e\xb5:^\x90\xd2X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?gt\xf9!\xfej\xd7X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>#\x8d1;=\xae.u.' and reward: 0.3706
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 158.16354608535767
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -40.82s of remaining time.
Ensemble size: 23
Ensemble weights: 
[0.60869565 0.34782609 0.04347826]
	0.3904	 = Validation accuracy score
	1.09s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 161.95s ...
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

