
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/22f2b7c7253266907172fe15dac6b61745a76480', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '22f2b7c7253266907172fe15dac6b61745a76480', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/22f2b7c7253266907172fe15dac6b61745a76480

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/22f2b7c7253266907172fe15dac6b61745a76480

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
 2351104/17464789 [===>..........................] - ETA: 0s
11812864/17464789 [===================>..........] - ETA: 0s
16900096/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 05:54:30.999332: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 05:54:31.003517: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-11 05:54:31.003642: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b5b4bfd250 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 05:54:31.003651: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:21 - loss: 9.1041 - accuracy: 0.4062
   64/25000 [..............................] - ETA: 2:39 - loss: 7.6666 - accuracy: 0.5000
   96/25000 [..............................] - ETA: 2:06 - loss: 7.6666 - accuracy: 0.5000
  128/25000 [..............................] - ETA: 1:49 - loss: 7.0677 - accuracy: 0.5391
  160/25000 [..............................] - ETA: 1:39 - loss: 7.0916 - accuracy: 0.5375
  192/25000 [..............................] - ETA: 1:31 - loss: 6.9479 - accuracy: 0.5469
  224/25000 [..............................] - ETA: 1:26 - loss: 7.2559 - accuracy: 0.5268
  256/25000 [..............................] - ETA: 1:22 - loss: 7.1276 - accuracy: 0.5352
  288/25000 [..............................] - ETA: 1:19 - loss: 7.2407 - accuracy: 0.5278
  320/25000 [..............................] - ETA: 1:17 - loss: 7.4750 - accuracy: 0.5125
  352/25000 [..............................] - ETA: 1:15 - loss: 7.4053 - accuracy: 0.5170
  384/25000 [..............................] - ETA: 1:14 - loss: 7.4270 - accuracy: 0.5156
  416/25000 [..............................] - ETA: 1:12 - loss: 7.5560 - accuracy: 0.5072
  448/25000 [..............................] - ETA: 1:11 - loss: 7.4955 - accuracy: 0.5112
  480/25000 [..............................] - ETA: 1:10 - loss: 7.4430 - accuracy: 0.5146
  512/25000 [..............................] - ETA: 1:09 - loss: 7.4570 - accuracy: 0.5137
  544/25000 [..............................] - ETA: 1:09 - loss: 7.4975 - accuracy: 0.5110
  576/25000 [..............................] - ETA: 1:08 - loss: 7.5601 - accuracy: 0.5069
  608/25000 [..............................] - ETA: 1:07 - loss: 7.5910 - accuracy: 0.5049
  640/25000 [..............................] - ETA: 1:06 - loss: 7.5708 - accuracy: 0.5063
  672/25000 [..............................] - ETA: 1:06 - loss: 7.5525 - accuracy: 0.5074
  704/25000 [..............................] - ETA: 1:06 - loss: 7.6231 - accuracy: 0.5028
  736/25000 [..............................] - ETA: 1:05 - loss: 7.6458 - accuracy: 0.5014
  768/25000 [..............................] - ETA: 1:05 - loss: 7.6067 - accuracy: 0.5039
  800/25000 [..............................] - ETA: 1:04 - loss: 7.5133 - accuracy: 0.5100
  832/25000 [..............................] - ETA: 1:03 - loss: 7.5376 - accuracy: 0.5084
  864/25000 [>.............................] - ETA: 1:03 - loss: 7.5779 - accuracy: 0.5058
  896/25000 [>.............................] - ETA: 1:03 - loss: 7.6153 - accuracy: 0.5033
  928/25000 [>.............................] - ETA: 1:03 - loss: 7.5840 - accuracy: 0.5054
  960/25000 [>.............................] - ETA: 1:02 - loss: 7.5229 - accuracy: 0.5094
  992/25000 [>.............................] - ETA: 1:02 - loss: 7.5275 - accuracy: 0.5091
 1024/25000 [>.............................] - ETA: 1:02 - loss: 7.5918 - accuracy: 0.5049
 1056/25000 [>.............................] - ETA: 1:01 - loss: 7.6231 - accuracy: 0.5028
 1088/25000 [>.............................] - ETA: 1:01 - loss: 7.6948 - accuracy: 0.4982
 1120/25000 [>.............................] - ETA: 1:01 - loss: 7.6940 - accuracy: 0.4982
 1152/25000 [>.............................] - ETA: 1:00 - loss: 7.6799 - accuracy: 0.4991
 1184/25000 [>.............................] - ETA: 1:00 - loss: 7.7314 - accuracy: 0.4958
 1216/25000 [>.............................] - ETA: 1:00 - loss: 7.7044 - accuracy: 0.4975
 1248/25000 [>.............................] - ETA: 1:00 - loss: 7.7158 - accuracy: 0.4968
 1280/25000 [>.............................] - ETA: 59s - loss: 7.7385 - accuracy: 0.4953 
 1312/25000 [>.............................] - ETA: 59s - loss: 7.7484 - accuracy: 0.4947
 1344/25000 [>.............................] - ETA: 59s - loss: 7.7237 - accuracy: 0.4963
 1376/25000 [>.............................] - ETA: 59s - loss: 7.7112 - accuracy: 0.4971
 1408/25000 [>.............................] - ETA: 59s - loss: 7.7320 - accuracy: 0.4957
 1440/25000 [>.............................] - ETA: 58s - loss: 7.7092 - accuracy: 0.4972
 1472/25000 [>.............................] - ETA: 58s - loss: 7.7291 - accuracy: 0.4959
 1504/25000 [>.............................] - ETA: 58s - loss: 7.7584 - accuracy: 0.4940
 1536/25000 [>.............................] - ETA: 58s - loss: 7.7465 - accuracy: 0.4948
 1568/25000 [>.............................] - ETA: 58s - loss: 7.7644 - accuracy: 0.4936
 1600/25000 [>.............................] - ETA: 58s - loss: 7.7816 - accuracy: 0.4925
 1632/25000 [>.............................] - ETA: 57s - loss: 7.7324 - accuracy: 0.4957
 1664/25000 [>.............................] - ETA: 57s - loss: 7.7035 - accuracy: 0.4976
 1696/25000 [=>............................] - ETA: 57s - loss: 7.6757 - accuracy: 0.4994
 1728/25000 [=>............................] - ETA: 57s - loss: 7.6755 - accuracy: 0.4994
 1760/25000 [=>............................] - ETA: 57s - loss: 7.6840 - accuracy: 0.4989
 1792/25000 [=>............................] - ETA: 57s - loss: 7.6837 - accuracy: 0.4989
 1824/25000 [=>............................] - ETA: 57s - loss: 7.6750 - accuracy: 0.4995
 1856/25000 [=>............................] - ETA: 57s - loss: 7.6336 - accuracy: 0.5022
 1888/25000 [=>............................] - ETA: 57s - loss: 7.6585 - accuracy: 0.5005
 1920/25000 [=>............................] - ETA: 57s - loss: 7.6666 - accuracy: 0.5000
 1952/25000 [=>............................] - ETA: 56s - loss: 7.6666 - accuracy: 0.5000
 1984/25000 [=>............................] - ETA: 56s - loss: 7.6589 - accuracy: 0.5005
 2016/25000 [=>............................] - ETA: 56s - loss: 7.6818 - accuracy: 0.4990
 2048/25000 [=>............................] - ETA: 56s - loss: 7.7041 - accuracy: 0.4976
 2080/25000 [=>............................] - ETA: 56s - loss: 7.7330 - accuracy: 0.4957
 2112/25000 [=>............................] - ETA: 56s - loss: 7.7392 - accuracy: 0.4953
 2144/25000 [=>............................] - ETA: 56s - loss: 7.7596 - accuracy: 0.4939
 2176/25000 [=>............................] - ETA: 56s - loss: 7.7723 - accuracy: 0.4931
 2208/25000 [=>............................] - ETA: 56s - loss: 7.7986 - accuracy: 0.4914
 2240/25000 [=>............................] - ETA: 56s - loss: 7.8104 - accuracy: 0.4906
 2272/25000 [=>............................] - ETA: 56s - loss: 7.8353 - accuracy: 0.4890
 2304/25000 [=>............................] - ETA: 56s - loss: 7.8330 - accuracy: 0.4891
 2336/25000 [=>............................] - ETA: 55s - loss: 7.8242 - accuracy: 0.4897
 2368/25000 [=>............................] - ETA: 55s - loss: 7.7896 - accuracy: 0.4920
 2400/25000 [=>............................] - ETA: 55s - loss: 7.7816 - accuracy: 0.4925
 2432/25000 [=>............................] - ETA: 55s - loss: 7.7549 - accuracy: 0.4942
 2464/25000 [=>............................] - ETA: 55s - loss: 7.7413 - accuracy: 0.4951
 2496/25000 [=>............................] - ETA: 55s - loss: 7.7465 - accuracy: 0.4948
 2528/25000 [==>...........................] - ETA: 55s - loss: 7.7637 - accuracy: 0.4937
 2560/25000 [==>...........................] - ETA: 55s - loss: 7.7565 - accuracy: 0.4941
 2592/25000 [==>...........................] - ETA: 54s - loss: 7.7494 - accuracy: 0.4946
 2624/25000 [==>...........................] - ETA: 54s - loss: 7.7367 - accuracy: 0.4954
 2656/25000 [==>...........................] - ETA: 54s - loss: 7.7417 - accuracy: 0.4951
 2688/25000 [==>...........................] - ETA: 54s - loss: 7.7522 - accuracy: 0.4944
 2720/25000 [==>...........................] - ETA: 54s - loss: 7.7681 - accuracy: 0.4934
 2752/25000 [==>...........................] - ETA: 54s - loss: 7.7781 - accuracy: 0.4927
 2784/25000 [==>...........................] - ETA: 54s - loss: 7.7602 - accuracy: 0.4939
 2816/25000 [==>...........................] - ETA: 54s - loss: 7.7646 - accuracy: 0.4936
 2848/25000 [==>...........................] - ETA: 54s - loss: 7.7635 - accuracy: 0.4937
 2880/25000 [==>...........................] - ETA: 54s - loss: 7.7784 - accuracy: 0.4927
 2912/25000 [==>...........................] - ETA: 54s - loss: 7.7614 - accuracy: 0.4938
 2944/25000 [==>...........................] - ETA: 54s - loss: 7.7552 - accuracy: 0.4942
 2976/25000 [==>...........................] - ETA: 54s - loss: 7.7800 - accuracy: 0.4926
 3008/25000 [==>...........................] - ETA: 53s - loss: 7.7788 - accuracy: 0.4927
 3040/25000 [==>...........................] - ETA: 53s - loss: 7.7776 - accuracy: 0.4928
 3072/25000 [==>...........................] - ETA: 53s - loss: 7.7714 - accuracy: 0.4932
 3104/25000 [==>...........................] - ETA: 53s - loss: 7.7654 - accuracy: 0.4936
 3136/25000 [==>...........................] - ETA: 53s - loss: 7.7644 - accuracy: 0.4936
 3168/25000 [==>...........................] - ETA: 53s - loss: 7.7586 - accuracy: 0.4940
 3200/25000 [==>...........................] - ETA: 53s - loss: 7.7385 - accuracy: 0.4953
 3232/25000 [==>...........................] - ETA: 53s - loss: 7.7283 - accuracy: 0.4960
 3264/25000 [==>...........................] - ETA: 53s - loss: 7.7183 - accuracy: 0.4966
 3296/25000 [==>...........................] - ETA: 53s - loss: 7.7271 - accuracy: 0.4961
 3328/25000 [==>...........................] - ETA: 52s - loss: 7.7311 - accuracy: 0.4958
 3360/25000 [===>..........................] - ETA: 52s - loss: 7.7031 - accuracy: 0.4976
 3392/25000 [===>..........................] - ETA: 52s - loss: 7.7163 - accuracy: 0.4968
 3424/25000 [===>..........................] - ETA: 52s - loss: 7.6980 - accuracy: 0.4980
 3456/25000 [===>..........................] - ETA: 52s - loss: 7.7110 - accuracy: 0.4971
 3488/25000 [===>..........................] - ETA: 52s - loss: 7.7062 - accuracy: 0.4974
 3520/25000 [===>..........................] - ETA: 52s - loss: 7.7015 - accuracy: 0.4977
 3552/25000 [===>..........................] - ETA: 52s - loss: 7.7012 - accuracy: 0.4977
 3584/25000 [===>..........................] - ETA: 52s - loss: 7.7222 - accuracy: 0.4964
 3616/25000 [===>..........................] - ETA: 52s - loss: 7.7472 - accuracy: 0.4947
 3648/25000 [===>..........................] - ETA: 52s - loss: 7.7507 - accuracy: 0.4945
 3680/25000 [===>..........................] - ETA: 52s - loss: 7.7500 - accuracy: 0.4946
 3712/25000 [===>..........................] - ETA: 52s - loss: 7.7492 - accuracy: 0.4946
 3744/25000 [===>..........................] - ETA: 51s - loss: 7.7649 - accuracy: 0.4936
 3776/25000 [===>..........................] - ETA: 51s - loss: 7.7681 - accuracy: 0.4934
 3808/25000 [===>..........................] - ETA: 51s - loss: 7.7633 - accuracy: 0.4937
 3840/25000 [===>..........................] - ETA: 51s - loss: 7.7864 - accuracy: 0.4922
 3872/25000 [===>..........................] - ETA: 51s - loss: 7.7577 - accuracy: 0.4941
 3904/25000 [===>..........................] - ETA: 51s - loss: 7.7570 - accuracy: 0.4941
 3936/25000 [===>..........................] - ETA: 51s - loss: 7.7484 - accuracy: 0.4947
 3968/25000 [===>..........................] - ETA: 51s - loss: 7.7362 - accuracy: 0.4955
 4000/25000 [===>..........................] - ETA: 51s - loss: 7.7280 - accuracy: 0.4960
 4032/25000 [===>..........................] - ETA: 50s - loss: 7.7313 - accuracy: 0.4958
 4064/25000 [===>..........................] - ETA: 50s - loss: 7.7308 - accuracy: 0.4958
 4096/25000 [===>..........................] - ETA: 50s - loss: 7.7228 - accuracy: 0.4963
 4128/25000 [===>..........................] - ETA: 50s - loss: 7.7335 - accuracy: 0.4956
 4160/25000 [===>..........................] - ETA: 50s - loss: 7.7514 - accuracy: 0.4945
 4192/25000 [====>.........................] - ETA: 50s - loss: 7.7507 - accuracy: 0.4945
 4224/25000 [====>.........................] - ETA: 50s - loss: 7.7429 - accuracy: 0.4950
 4256/25000 [====>.........................] - ETA: 50s - loss: 7.7171 - accuracy: 0.4967
 4288/25000 [====>.........................] - ETA: 50s - loss: 7.7060 - accuracy: 0.4974
 4320/25000 [====>.........................] - ETA: 50s - loss: 7.7163 - accuracy: 0.4968
 4352/25000 [====>.........................] - ETA: 49s - loss: 7.7265 - accuracy: 0.4961
 4384/25000 [====>.........................] - ETA: 49s - loss: 7.7086 - accuracy: 0.4973
 4416/25000 [====>.........................] - ETA: 49s - loss: 7.7013 - accuracy: 0.4977
 4448/25000 [====>.........................] - ETA: 49s - loss: 7.6908 - accuracy: 0.4984
 4480/25000 [====>.........................] - ETA: 49s - loss: 7.6906 - accuracy: 0.4984
 4512/25000 [====>.........................] - ETA: 49s - loss: 7.6734 - accuracy: 0.4996
 4544/25000 [====>.........................] - ETA: 49s - loss: 7.6666 - accuracy: 0.5000
 4576/25000 [====>.........................] - ETA: 49s - loss: 7.6599 - accuracy: 0.5004
 4608/25000 [====>.........................] - ETA: 49s - loss: 7.6600 - accuracy: 0.5004
 4640/25000 [====>.........................] - ETA: 48s - loss: 7.6732 - accuracy: 0.4996
 4672/25000 [====>.........................] - ETA: 48s - loss: 7.6601 - accuracy: 0.5004
 4704/25000 [====>.........................] - ETA: 48s - loss: 7.6634 - accuracy: 0.5002
 4736/25000 [====>.........................] - ETA: 48s - loss: 7.6601 - accuracy: 0.5004
 4768/25000 [====>.........................] - ETA: 48s - loss: 7.6602 - accuracy: 0.5004
 4800/25000 [====>.........................] - ETA: 48s - loss: 7.6794 - accuracy: 0.4992
 4832/25000 [====>.........................] - ETA: 48s - loss: 7.6793 - accuracy: 0.4992
 4864/25000 [====>.........................] - ETA: 48s - loss: 7.6792 - accuracy: 0.4992
 4896/25000 [====>.........................] - ETA: 48s - loss: 7.6760 - accuracy: 0.4994
 4928/25000 [====>.........................] - ETA: 48s - loss: 7.6573 - accuracy: 0.5006
 4960/25000 [====>.........................] - ETA: 48s - loss: 7.6481 - accuracy: 0.5012
 4992/25000 [====>.........................] - ETA: 48s - loss: 7.6605 - accuracy: 0.5004
 5024/25000 [=====>........................] - ETA: 48s - loss: 7.6544 - accuracy: 0.5008
 5056/25000 [=====>........................] - ETA: 47s - loss: 7.6545 - accuracy: 0.5008
 5088/25000 [=====>........................] - ETA: 47s - loss: 7.6455 - accuracy: 0.5014
 5120/25000 [=====>........................] - ETA: 47s - loss: 7.6397 - accuracy: 0.5018
 5152/25000 [=====>........................] - ETA: 47s - loss: 7.6369 - accuracy: 0.5019
 5184/25000 [=====>........................] - ETA: 47s - loss: 7.6400 - accuracy: 0.5017
 5216/25000 [=====>........................] - ETA: 47s - loss: 7.6372 - accuracy: 0.5019
 5248/25000 [=====>........................] - ETA: 47s - loss: 7.6462 - accuracy: 0.5013
 5280/25000 [=====>........................] - ETA: 47s - loss: 7.6463 - accuracy: 0.5013
 5312/25000 [=====>........................] - ETA: 47s - loss: 7.6493 - accuracy: 0.5011
 5344/25000 [=====>........................] - ETA: 47s - loss: 7.6551 - accuracy: 0.5007
 5376/25000 [=====>........................] - ETA: 47s - loss: 7.6524 - accuracy: 0.5009
 5408/25000 [=====>........................] - ETA: 46s - loss: 7.6496 - accuracy: 0.5011
 5440/25000 [=====>........................] - ETA: 46s - loss: 7.6497 - accuracy: 0.5011
 5472/25000 [=====>........................] - ETA: 46s - loss: 7.6610 - accuracy: 0.5004
 5504/25000 [=====>........................] - ETA: 46s - loss: 7.6555 - accuracy: 0.5007
 5536/25000 [=====>........................] - ETA: 46s - loss: 7.6445 - accuracy: 0.5014
 5568/25000 [=====>........................] - ETA: 46s - loss: 7.6473 - accuracy: 0.5013
 5600/25000 [=====>........................] - ETA: 46s - loss: 7.6447 - accuracy: 0.5014
 5632/25000 [=====>........................] - ETA: 46s - loss: 7.6421 - accuracy: 0.5016
 5664/25000 [=====>........................] - ETA: 46s - loss: 7.6450 - accuracy: 0.5014
 5696/25000 [=====>........................] - ETA: 46s - loss: 7.6505 - accuracy: 0.5011
 5728/25000 [=====>........................] - ETA: 46s - loss: 7.6506 - accuracy: 0.5010
 5760/25000 [=====>........................] - ETA: 45s - loss: 7.6533 - accuracy: 0.5009
 5792/25000 [=====>........................] - ETA: 45s - loss: 7.6507 - accuracy: 0.5010
 5824/25000 [=====>........................] - ETA: 45s - loss: 7.6403 - accuracy: 0.5017
 5856/25000 [======>.......................] - ETA: 45s - loss: 7.6457 - accuracy: 0.5014
 5888/25000 [======>.......................] - ETA: 45s - loss: 7.6510 - accuracy: 0.5010
 5920/25000 [======>.......................] - ETA: 45s - loss: 7.6563 - accuracy: 0.5007
 5952/25000 [======>.......................] - ETA: 45s - loss: 7.6434 - accuracy: 0.5015
 5984/25000 [======>.......................] - ETA: 45s - loss: 7.6410 - accuracy: 0.5017
 6016/25000 [======>.......................] - ETA: 45s - loss: 7.6411 - accuracy: 0.5017
 6048/25000 [======>.......................] - ETA: 45s - loss: 7.6337 - accuracy: 0.5021
 6080/25000 [======>.......................] - ETA: 45s - loss: 7.6389 - accuracy: 0.5018
 6112/25000 [======>.......................] - ETA: 44s - loss: 7.6516 - accuracy: 0.5010
 6144/25000 [======>.......................] - ETA: 44s - loss: 7.6516 - accuracy: 0.5010
 6176/25000 [======>.......................] - ETA: 44s - loss: 7.6492 - accuracy: 0.5011
 6208/25000 [======>.......................] - ETA: 44s - loss: 7.6419 - accuracy: 0.5016
 6240/25000 [======>.......................] - ETA: 44s - loss: 7.6494 - accuracy: 0.5011
 6272/25000 [======>.......................] - ETA: 44s - loss: 7.6568 - accuracy: 0.5006
 6304/25000 [======>.......................] - ETA: 44s - loss: 7.6374 - accuracy: 0.5019
 6336/25000 [======>.......................] - ETA: 44s - loss: 7.6424 - accuracy: 0.5016
 6368/25000 [======>.......................] - ETA: 44s - loss: 7.6401 - accuracy: 0.5017
 6400/25000 [======>.......................] - ETA: 44s - loss: 7.6379 - accuracy: 0.5019
 6432/25000 [======>.......................] - ETA: 44s - loss: 7.6380 - accuracy: 0.5019
 6464/25000 [======>.......................] - ETA: 43s - loss: 7.6382 - accuracy: 0.5019
 6496/25000 [======>.......................] - ETA: 43s - loss: 7.6477 - accuracy: 0.5012
 6528/25000 [======>.......................] - ETA: 43s - loss: 7.6384 - accuracy: 0.5018
 6560/25000 [======>.......................] - ETA: 43s - loss: 7.6386 - accuracy: 0.5018
 6592/25000 [======>.......................] - ETA: 43s - loss: 7.6387 - accuracy: 0.5018
 6624/25000 [======>.......................] - ETA: 43s - loss: 7.6504 - accuracy: 0.5011
 6656/25000 [======>.......................] - ETA: 43s - loss: 7.6528 - accuracy: 0.5009
 6688/25000 [=======>......................] - ETA: 43s - loss: 7.6506 - accuracy: 0.5010
 6720/25000 [=======>......................] - ETA: 43s - loss: 7.6506 - accuracy: 0.5010
 6752/25000 [=======>......................] - ETA: 43s - loss: 7.6348 - accuracy: 0.5021
 6784/25000 [=======>......................] - ETA: 43s - loss: 7.6372 - accuracy: 0.5019
 6816/25000 [=======>......................] - ETA: 42s - loss: 7.6396 - accuracy: 0.5018
 6848/25000 [=======>......................] - ETA: 42s - loss: 7.6420 - accuracy: 0.5016
 6880/25000 [=======>......................] - ETA: 42s - loss: 7.6399 - accuracy: 0.5017
 6912/25000 [=======>......................] - ETA: 42s - loss: 7.6422 - accuracy: 0.5016
 6944/25000 [=======>......................] - ETA: 42s - loss: 7.6445 - accuracy: 0.5014
 6976/25000 [=======>......................] - ETA: 42s - loss: 7.6402 - accuracy: 0.5017
 7008/25000 [=======>......................] - ETA: 42s - loss: 7.6360 - accuracy: 0.5020
 7040/25000 [=======>......................] - ETA: 42s - loss: 7.6427 - accuracy: 0.5016
 7072/25000 [=======>......................] - ETA: 42s - loss: 7.6384 - accuracy: 0.5018
 7104/25000 [=======>......................] - ETA: 42s - loss: 7.6342 - accuracy: 0.5021
 7136/25000 [=======>......................] - ETA: 42s - loss: 7.6301 - accuracy: 0.5024
 7168/25000 [=======>......................] - ETA: 42s - loss: 7.6217 - accuracy: 0.5029
 7200/25000 [=======>......................] - ETA: 41s - loss: 7.6262 - accuracy: 0.5026
 7232/25000 [=======>......................] - ETA: 41s - loss: 7.6285 - accuracy: 0.5025
 7264/25000 [=======>......................] - ETA: 41s - loss: 7.6371 - accuracy: 0.5019
 7296/25000 [=======>......................] - ETA: 41s - loss: 7.6498 - accuracy: 0.5011
 7328/25000 [=======>......................] - ETA: 41s - loss: 7.6541 - accuracy: 0.5008
 7360/25000 [=======>......................] - ETA: 41s - loss: 7.6520 - accuracy: 0.5010
 7392/25000 [=======>......................] - ETA: 41s - loss: 7.6542 - accuracy: 0.5008
 7424/25000 [=======>......................] - ETA: 41s - loss: 7.6480 - accuracy: 0.5012
 7456/25000 [=======>......................] - ETA: 41s - loss: 7.6440 - accuracy: 0.5015
 7488/25000 [=======>......................] - ETA: 41s - loss: 7.6482 - accuracy: 0.5012
 7520/25000 [========>.....................] - ETA: 41s - loss: 7.6544 - accuracy: 0.5008
 7552/25000 [========>.....................] - ETA: 41s - loss: 7.6585 - accuracy: 0.5005
 7584/25000 [========>.....................] - ETA: 41s - loss: 7.6484 - accuracy: 0.5012
 7616/25000 [========>.....................] - ETA: 40s - loss: 7.6465 - accuracy: 0.5013
 7648/25000 [========>.....................] - ETA: 40s - loss: 7.6486 - accuracy: 0.5012
 7680/25000 [========>.....................] - ETA: 40s - loss: 7.6566 - accuracy: 0.5007
 7712/25000 [========>.....................] - ETA: 40s - loss: 7.6646 - accuracy: 0.5001
 7744/25000 [========>.....................] - ETA: 40s - loss: 7.6607 - accuracy: 0.5004
 7776/25000 [========>.....................] - ETA: 40s - loss: 7.6666 - accuracy: 0.5000
 7808/25000 [========>.....................] - ETA: 40s - loss: 7.6647 - accuracy: 0.5001
 7840/25000 [========>.....................] - ETA: 40s - loss: 7.6666 - accuracy: 0.5000
 7872/25000 [========>.....................] - ETA: 40s - loss: 7.6725 - accuracy: 0.4996
 7904/25000 [========>.....................] - ETA: 40s - loss: 7.6705 - accuracy: 0.4997
 7936/25000 [========>.....................] - ETA: 40s - loss: 7.6705 - accuracy: 0.4997
 7968/25000 [========>.....................] - ETA: 39s - loss: 7.6685 - accuracy: 0.4999
 8000/25000 [========>.....................] - ETA: 39s - loss: 7.6666 - accuracy: 0.5000
 8032/25000 [========>.....................] - ETA: 39s - loss: 7.6647 - accuracy: 0.5001
 8064/25000 [========>.....................] - ETA: 39s - loss: 7.6647 - accuracy: 0.5001
 8096/25000 [========>.....................] - ETA: 39s - loss: 7.6628 - accuracy: 0.5002
 8128/25000 [========>.....................] - ETA: 39s - loss: 7.6628 - accuracy: 0.5002
 8160/25000 [========>.....................] - ETA: 39s - loss: 7.6610 - accuracy: 0.5004
 8192/25000 [========>.....................] - ETA: 39s - loss: 7.6629 - accuracy: 0.5002
 8224/25000 [========>.....................] - ETA: 39s - loss: 7.6629 - accuracy: 0.5002
 8256/25000 [========>.....................] - ETA: 39s - loss: 7.6629 - accuracy: 0.5002
 8288/25000 [========>.....................] - ETA: 39s - loss: 7.6722 - accuracy: 0.4996
 8320/25000 [========>.....................] - ETA: 39s - loss: 7.6795 - accuracy: 0.4992
 8352/25000 [=========>....................] - ETA: 39s - loss: 7.6795 - accuracy: 0.4992
 8384/25000 [=========>....................] - ETA: 38s - loss: 7.6904 - accuracy: 0.4984
 8416/25000 [=========>....................] - ETA: 38s - loss: 7.6830 - accuracy: 0.4989
 8448/25000 [=========>....................] - ETA: 38s - loss: 7.6757 - accuracy: 0.4994
 8480/25000 [=========>....................] - ETA: 38s - loss: 7.6793 - accuracy: 0.4992
 8512/25000 [=========>....................] - ETA: 38s - loss: 7.6864 - accuracy: 0.4987
 8544/25000 [=========>....................] - ETA: 38s - loss: 7.6864 - accuracy: 0.4987
 8576/25000 [=========>....................] - ETA: 38s - loss: 7.6917 - accuracy: 0.4984
 8608/25000 [=========>....................] - ETA: 38s - loss: 7.6951 - accuracy: 0.4981
 8640/25000 [=========>....................] - ETA: 38s - loss: 7.6932 - accuracy: 0.4983
 8672/25000 [=========>....................] - ETA: 38s - loss: 7.6967 - accuracy: 0.4980
 8704/25000 [=========>....................] - ETA: 38s - loss: 7.6948 - accuracy: 0.4982
 8736/25000 [=========>....................] - ETA: 38s - loss: 7.6982 - accuracy: 0.4979
 8768/25000 [=========>....................] - ETA: 38s - loss: 7.6963 - accuracy: 0.4981
 8800/25000 [=========>....................] - ETA: 37s - loss: 7.6875 - accuracy: 0.4986
 8832/25000 [=========>....................] - ETA: 37s - loss: 7.6979 - accuracy: 0.4980
 8864/25000 [=========>....................] - ETA: 37s - loss: 7.6926 - accuracy: 0.4983
 8896/25000 [=========>....................] - ETA: 37s - loss: 7.6873 - accuracy: 0.4987
 8928/25000 [=========>....................] - ETA: 37s - loss: 7.6821 - accuracy: 0.4990
 8960/25000 [=========>....................] - ETA: 37s - loss: 7.6752 - accuracy: 0.4994
 8992/25000 [=========>....................] - ETA: 37s - loss: 7.6769 - accuracy: 0.4993
 9024/25000 [=========>....................] - ETA: 37s - loss: 7.6785 - accuracy: 0.4992
 9056/25000 [=========>....................] - ETA: 37s - loss: 7.6869 - accuracy: 0.4987
 9088/25000 [=========>....................] - ETA: 37s - loss: 7.6902 - accuracy: 0.4985
 9120/25000 [=========>....................] - ETA: 37s - loss: 7.6868 - accuracy: 0.4987
 9152/25000 [=========>....................] - ETA: 37s - loss: 7.6918 - accuracy: 0.4984
 9184/25000 [==========>...................] - ETA: 37s - loss: 7.6950 - accuracy: 0.4981
 9216/25000 [==========>...................] - ETA: 37s - loss: 7.6932 - accuracy: 0.4983
 9248/25000 [==========>...................] - ETA: 36s - loss: 7.7014 - accuracy: 0.4977
 9280/25000 [==========>...................] - ETA: 36s - loss: 7.7030 - accuracy: 0.4976
 9312/25000 [==========>...................] - ETA: 36s - loss: 7.7078 - accuracy: 0.4973
 9344/25000 [==========>...................] - ETA: 36s - loss: 7.7060 - accuracy: 0.4974
 9376/25000 [==========>...................] - ETA: 36s - loss: 7.7075 - accuracy: 0.4973
 9408/25000 [==========>...................] - ETA: 36s - loss: 7.7057 - accuracy: 0.4974
 9440/25000 [==========>...................] - ETA: 36s - loss: 7.7072 - accuracy: 0.4974
 9472/25000 [==========>...................] - ETA: 36s - loss: 7.7087 - accuracy: 0.4973
 9504/25000 [==========>...................] - ETA: 36s - loss: 7.7037 - accuracy: 0.4976
 9536/25000 [==========>...................] - ETA: 36s - loss: 7.7100 - accuracy: 0.4972
 9568/25000 [==========>...................] - ETA: 36s - loss: 7.7147 - accuracy: 0.4969
 9600/25000 [==========>...................] - ETA: 36s - loss: 7.7177 - accuracy: 0.4967
 9632/25000 [==========>...................] - ETA: 36s - loss: 7.7176 - accuracy: 0.4967
 9664/25000 [==========>...................] - ETA: 35s - loss: 7.7237 - accuracy: 0.4963
 9696/25000 [==========>...................] - ETA: 35s - loss: 7.7283 - accuracy: 0.4960
 9728/25000 [==========>...................] - ETA: 35s - loss: 7.7265 - accuracy: 0.4961
 9760/25000 [==========>...................] - ETA: 35s - loss: 7.7247 - accuracy: 0.4962
 9792/25000 [==========>...................] - ETA: 35s - loss: 7.7214 - accuracy: 0.4964
 9824/25000 [==========>...................] - ETA: 35s - loss: 7.7197 - accuracy: 0.4965
 9856/25000 [==========>...................] - ETA: 35s - loss: 7.7133 - accuracy: 0.4970
 9888/25000 [==========>...................] - ETA: 35s - loss: 7.7116 - accuracy: 0.4971
 9920/25000 [==========>...................] - ETA: 35s - loss: 7.7114 - accuracy: 0.4971
 9952/25000 [==========>...................] - ETA: 35s - loss: 7.7082 - accuracy: 0.4973
 9984/25000 [==========>...................] - ETA: 35s - loss: 7.7081 - accuracy: 0.4973
10016/25000 [===========>..................] - ETA: 35s - loss: 7.7141 - accuracy: 0.4969
10048/25000 [===========>..................] - ETA: 35s - loss: 7.7124 - accuracy: 0.4970
10080/25000 [===========>..................] - ETA: 34s - loss: 7.7107 - accuracy: 0.4971
10112/25000 [===========>..................] - ETA: 34s - loss: 7.7076 - accuracy: 0.4973
10144/25000 [===========>..................] - ETA: 34s - loss: 7.7089 - accuracy: 0.4972
10176/25000 [===========>..................] - ETA: 34s - loss: 7.7103 - accuracy: 0.4972
10208/25000 [===========>..................] - ETA: 34s - loss: 7.7162 - accuracy: 0.4968
10240/25000 [===========>..................] - ETA: 34s - loss: 7.7145 - accuracy: 0.4969
10272/25000 [===========>..................] - ETA: 34s - loss: 7.7114 - accuracy: 0.4971
10304/25000 [===========>..................] - ETA: 34s - loss: 7.7113 - accuracy: 0.4971
10336/25000 [===========>..................] - ETA: 34s - loss: 7.7096 - accuracy: 0.4972
10368/25000 [===========>..................] - ETA: 34s - loss: 7.7110 - accuracy: 0.4971
10400/25000 [===========>..................] - ETA: 34s - loss: 7.7153 - accuracy: 0.4968
10432/25000 [===========>..................] - ETA: 34s - loss: 7.7181 - accuracy: 0.4966
10464/25000 [===========>..................] - ETA: 33s - loss: 7.7179 - accuracy: 0.4967
10496/25000 [===========>..................] - ETA: 33s - loss: 7.7251 - accuracy: 0.4962
10528/25000 [===========>..................] - ETA: 33s - loss: 7.7292 - accuracy: 0.4959
10560/25000 [===========>..................] - ETA: 33s - loss: 7.7334 - accuracy: 0.4956
10592/25000 [===========>..................] - ETA: 33s - loss: 7.7318 - accuracy: 0.4958
10624/25000 [===========>..................] - ETA: 33s - loss: 7.7359 - accuracy: 0.4955
10656/25000 [===========>..................] - ETA: 33s - loss: 7.7299 - accuracy: 0.4959
10688/25000 [===========>..................] - ETA: 33s - loss: 7.7254 - accuracy: 0.4962
10720/25000 [===========>..................] - ETA: 33s - loss: 7.7353 - accuracy: 0.4955
10752/25000 [===========>..................] - ETA: 33s - loss: 7.7408 - accuracy: 0.4952
10784/25000 [===========>..................] - ETA: 33s - loss: 7.7406 - accuracy: 0.4952
10816/25000 [===========>..................] - ETA: 33s - loss: 7.7446 - accuracy: 0.4949
10848/25000 [============>.................] - ETA: 33s - loss: 7.7444 - accuracy: 0.4949
10880/25000 [============>.................] - ETA: 33s - loss: 7.7385 - accuracy: 0.4953
10912/25000 [============>.................] - ETA: 32s - loss: 7.7369 - accuracy: 0.4954
10944/25000 [============>.................] - ETA: 32s - loss: 7.7367 - accuracy: 0.4954
10976/25000 [============>.................] - ETA: 32s - loss: 7.7421 - accuracy: 0.4951
11008/25000 [============>.................] - ETA: 32s - loss: 7.7446 - accuracy: 0.4949
11040/25000 [============>.................] - ETA: 32s - loss: 7.7472 - accuracy: 0.4947
11072/25000 [============>.................] - ETA: 32s - loss: 7.7469 - accuracy: 0.4948
11104/25000 [============>.................] - ETA: 32s - loss: 7.7481 - accuracy: 0.4947
11136/25000 [============>.................] - ETA: 32s - loss: 7.7465 - accuracy: 0.4948
11168/25000 [============>.................] - ETA: 32s - loss: 7.7435 - accuracy: 0.4950
11200/25000 [============>.................] - ETA: 32s - loss: 7.7460 - accuracy: 0.4948
11232/25000 [============>.................] - ETA: 32s - loss: 7.7485 - accuracy: 0.4947
11264/25000 [============>.................] - ETA: 32s - loss: 7.7510 - accuracy: 0.4945
11296/25000 [============>.................] - ETA: 32s - loss: 7.7467 - accuracy: 0.4948
11328/25000 [============>.................] - ETA: 31s - loss: 7.7465 - accuracy: 0.4948
11360/25000 [============>.................] - ETA: 31s - loss: 7.7503 - accuracy: 0.4945
11392/25000 [============>.................] - ETA: 31s - loss: 7.7487 - accuracy: 0.4946
11424/25000 [============>.................] - ETA: 31s - loss: 7.7472 - accuracy: 0.4947
11456/25000 [============>.................] - ETA: 31s - loss: 7.7416 - accuracy: 0.4951
11488/25000 [============>.................] - ETA: 31s - loss: 7.7374 - accuracy: 0.4954
11520/25000 [============>.................] - ETA: 31s - loss: 7.7318 - accuracy: 0.4957
11552/25000 [============>.................] - ETA: 31s - loss: 7.7303 - accuracy: 0.4958
11584/25000 [============>.................] - ETA: 31s - loss: 7.7328 - accuracy: 0.4957
11616/25000 [============>.................] - ETA: 31s - loss: 7.7339 - accuracy: 0.4956
11648/25000 [============>.................] - ETA: 31s - loss: 7.7351 - accuracy: 0.4955
11680/25000 [=============>................] - ETA: 31s - loss: 7.7375 - accuracy: 0.4954
11712/25000 [=============>................] - ETA: 31s - loss: 7.7321 - accuracy: 0.4957
11744/25000 [=============>................] - ETA: 31s - loss: 7.7345 - accuracy: 0.4956
11776/25000 [=============>................] - ETA: 30s - loss: 7.7356 - accuracy: 0.4955
11808/25000 [=============>................] - ETA: 30s - loss: 7.7328 - accuracy: 0.4957
11840/25000 [=============>................] - ETA: 30s - loss: 7.7314 - accuracy: 0.4958
11872/25000 [=============>................] - ETA: 30s - loss: 7.7299 - accuracy: 0.4959
11904/25000 [=============>................] - ETA: 30s - loss: 7.7272 - accuracy: 0.4961
11936/25000 [=============>................] - ETA: 30s - loss: 7.7270 - accuracy: 0.4961
11968/25000 [=============>................] - ETA: 30s - loss: 7.7268 - accuracy: 0.4961
12000/25000 [=============>................] - ETA: 30s - loss: 7.7267 - accuracy: 0.4961
12032/25000 [=============>................] - ETA: 30s - loss: 7.7291 - accuracy: 0.4959
12064/25000 [=============>................] - ETA: 30s - loss: 7.7289 - accuracy: 0.4959
12096/25000 [=============>................] - ETA: 30s - loss: 7.7249 - accuracy: 0.4962
12128/25000 [=============>................] - ETA: 30s - loss: 7.7273 - accuracy: 0.4960
12160/25000 [=============>................] - ETA: 30s - loss: 7.7221 - accuracy: 0.4964
12192/25000 [=============>................] - ETA: 29s - loss: 7.7207 - accuracy: 0.4965
12224/25000 [=============>................] - ETA: 29s - loss: 7.7243 - accuracy: 0.4962
12256/25000 [=============>................] - ETA: 29s - loss: 7.7292 - accuracy: 0.4959
12288/25000 [=============>................] - ETA: 29s - loss: 7.7352 - accuracy: 0.4955
12320/25000 [=============>................] - ETA: 29s - loss: 7.7400 - accuracy: 0.4952
12352/25000 [=============>................] - ETA: 29s - loss: 7.7436 - accuracy: 0.4950
12384/25000 [=============>................] - ETA: 29s - loss: 7.7347 - accuracy: 0.4956
12416/25000 [=============>................] - ETA: 29s - loss: 7.7345 - accuracy: 0.4956
12448/25000 [=============>................] - ETA: 29s - loss: 7.7294 - accuracy: 0.4959
12480/25000 [=============>................] - ETA: 29s - loss: 7.7244 - accuracy: 0.4962
12512/25000 [==============>...............] - ETA: 29s - loss: 7.7242 - accuracy: 0.4962
12544/25000 [==============>...............] - ETA: 29s - loss: 7.7155 - accuracy: 0.4968
12576/25000 [==============>...............] - ETA: 29s - loss: 7.7117 - accuracy: 0.4971
12608/25000 [==============>...............] - ETA: 28s - loss: 7.7068 - accuracy: 0.4974
12640/25000 [==============>...............] - ETA: 28s - loss: 7.7079 - accuracy: 0.4973
12672/25000 [==============>...............] - ETA: 28s - loss: 7.7065 - accuracy: 0.4974
12704/25000 [==============>...............] - ETA: 28s - loss: 7.7089 - accuracy: 0.4972
12736/25000 [==============>...............] - ETA: 28s - loss: 7.7076 - accuracy: 0.4973
12768/25000 [==============>...............] - ETA: 28s - loss: 7.7062 - accuracy: 0.4974
12800/25000 [==============>...............] - ETA: 28s - loss: 7.7109 - accuracy: 0.4971
12832/25000 [==============>...............] - ETA: 28s - loss: 7.7061 - accuracy: 0.4974
12864/25000 [==============>...............] - ETA: 28s - loss: 7.7060 - accuracy: 0.4974
12896/25000 [==============>...............] - ETA: 28s - loss: 7.7059 - accuracy: 0.4974
12928/25000 [==============>...............] - ETA: 28s - loss: 7.7081 - accuracy: 0.4973
12960/25000 [==============>...............] - ETA: 28s - loss: 7.7116 - accuracy: 0.4971
12992/25000 [==============>...............] - ETA: 28s - loss: 7.7126 - accuracy: 0.4970
13024/25000 [==============>...............] - ETA: 28s - loss: 7.7196 - accuracy: 0.4965
13056/25000 [==============>...............] - ETA: 27s - loss: 7.7183 - accuracy: 0.4966
13088/25000 [==============>...............] - ETA: 27s - loss: 7.7205 - accuracy: 0.4965
13120/25000 [==============>...............] - ETA: 27s - loss: 7.7227 - accuracy: 0.4963
13152/25000 [==============>...............] - ETA: 27s - loss: 7.7226 - accuracy: 0.4964
13184/25000 [==============>...............] - ETA: 27s - loss: 7.7224 - accuracy: 0.4964
13216/25000 [==============>...............] - ETA: 27s - loss: 7.7200 - accuracy: 0.4965
13248/25000 [==============>...............] - ETA: 27s - loss: 7.7175 - accuracy: 0.4967
13280/25000 [==============>...............] - ETA: 27s - loss: 7.7174 - accuracy: 0.4967
13312/25000 [==============>...............] - ETA: 27s - loss: 7.7150 - accuracy: 0.4968
13344/25000 [===============>..............] - ETA: 27s - loss: 7.7137 - accuracy: 0.4969
13376/25000 [===============>..............] - ETA: 27s - loss: 7.7159 - accuracy: 0.4968
13408/25000 [===============>..............] - ETA: 27s - loss: 7.7101 - accuracy: 0.4972
13440/25000 [===============>..............] - ETA: 27s - loss: 7.6997 - accuracy: 0.4978
13472/25000 [===============>..............] - ETA: 26s - loss: 7.6996 - accuracy: 0.4978
13504/25000 [===============>..............] - ETA: 26s - loss: 7.6995 - accuracy: 0.4979
13536/25000 [===============>..............] - ETA: 26s - loss: 7.6983 - accuracy: 0.4979
13568/25000 [===============>..............] - ETA: 26s - loss: 7.6960 - accuracy: 0.4981
13600/25000 [===============>..............] - ETA: 26s - loss: 7.6959 - accuracy: 0.4981
13632/25000 [===============>..............] - ETA: 26s - loss: 7.6914 - accuracy: 0.4984
13664/25000 [===============>..............] - ETA: 26s - loss: 7.6936 - accuracy: 0.4982
13696/25000 [===============>..............] - ETA: 26s - loss: 7.6879 - accuracy: 0.4986
13728/25000 [===============>..............] - ETA: 26s - loss: 7.6912 - accuracy: 0.4984
13760/25000 [===============>..............] - ETA: 26s - loss: 7.6934 - accuracy: 0.4983
13792/25000 [===============>..............] - ETA: 26s - loss: 7.6933 - accuracy: 0.4983
13824/25000 [===============>..............] - ETA: 26s - loss: 7.6899 - accuracy: 0.4985
13856/25000 [===============>..............] - ETA: 26s - loss: 7.6876 - accuracy: 0.4986
13888/25000 [===============>..............] - ETA: 25s - loss: 7.6876 - accuracy: 0.4986
13920/25000 [===============>..............] - ETA: 25s - loss: 7.6864 - accuracy: 0.4987
13952/25000 [===============>..............] - ETA: 25s - loss: 7.6864 - accuracy: 0.4987
13984/25000 [===============>..............] - ETA: 25s - loss: 7.6885 - accuracy: 0.4986
14016/25000 [===============>..............] - ETA: 25s - loss: 7.6885 - accuracy: 0.4986
14048/25000 [===============>..............] - ETA: 25s - loss: 7.6863 - accuracy: 0.4987
14080/25000 [===============>..............] - ETA: 25s - loss: 7.6873 - accuracy: 0.4987
14112/25000 [===============>..............] - ETA: 25s - loss: 7.6884 - accuracy: 0.4986
14144/25000 [===============>..............] - ETA: 25s - loss: 7.6948 - accuracy: 0.4982
14176/25000 [================>.............] - ETA: 25s - loss: 7.6969 - accuracy: 0.4980
14208/25000 [================>.............] - ETA: 25s - loss: 7.6914 - accuracy: 0.4984
14240/25000 [================>.............] - ETA: 25s - loss: 7.6914 - accuracy: 0.4984
14272/25000 [================>.............] - ETA: 25s - loss: 7.6903 - accuracy: 0.4985
14304/25000 [================>.............] - ETA: 24s - loss: 7.6902 - accuracy: 0.4985
14336/25000 [================>.............] - ETA: 24s - loss: 7.6891 - accuracy: 0.4985
14368/25000 [================>.............] - ETA: 24s - loss: 7.6858 - accuracy: 0.4987
14400/25000 [================>.............] - ETA: 24s - loss: 7.6826 - accuracy: 0.4990
14432/25000 [================>.............] - ETA: 24s - loss: 7.6857 - accuracy: 0.4988
14464/25000 [================>.............] - ETA: 24s - loss: 7.6857 - accuracy: 0.4988
14496/25000 [================>.............] - ETA: 24s - loss: 7.6846 - accuracy: 0.4988
14528/25000 [================>.............] - ETA: 24s - loss: 7.6846 - accuracy: 0.4988
14560/25000 [================>.............] - ETA: 24s - loss: 7.6835 - accuracy: 0.4989
14592/25000 [================>.............] - ETA: 24s - loss: 7.6803 - accuracy: 0.4991
14624/25000 [================>.............] - ETA: 24s - loss: 7.6844 - accuracy: 0.4988
14656/25000 [================>.............] - ETA: 24s - loss: 7.6792 - accuracy: 0.4992
14688/25000 [================>.............] - ETA: 24s - loss: 7.6781 - accuracy: 0.4993
14720/25000 [================>.............] - ETA: 23s - loss: 7.6781 - accuracy: 0.4993
14752/25000 [================>.............] - ETA: 23s - loss: 7.6781 - accuracy: 0.4993
14784/25000 [================>.............] - ETA: 23s - loss: 7.6832 - accuracy: 0.4989
14816/25000 [================>.............] - ETA: 23s - loss: 7.6801 - accuracy: 0.4991
14848/25000 [================>.............] - ETA: 23s - loss: 7.6780 - accuracy: 0.4993
14880/25000 [================>.............] - ETA: 23s - loss: 7.6790 - accuracy: 0.4992
14912/25000 [================>.............] - ETA: 23s - loss: 7.6779 - accuracy: 0.4993
14944/25000 [================>.............] - ETA: 23s - loss: 7.6779 - accuracy: 0.4993
14976/25000 [================>.............] - ETA: 23s - loss: 7.6779 - accuracy: 0.4993
15008/25000 [=================>............] - ETA: 23s - loss: 7.6758 - accuracy: 0.4994
15040/25000 [=================>............] - ETA: 23s - loss: 7.6727 - accuracy: 0.4996
15072/25000 [=================>............] - ETA: 23s - loss: 7.6737 - accuracy: 0.4995
15104/25000 [=================>............] - ETA: 23s - loss: 7.6737 - accuracy: 0.4995
15136/25000 [=================>............] - ETA: 22s - loss: 7.6737 - accuracy: 0.4995
15168/25000 [=================>............] - ETA: 22s - loss: 7.6777 - accuracy: 0.4993
15200/25000 [=================>............] - ETA: 22s - loss: 7.6777 - accuracy: 0.4993
15232/25000 [=================>............] - ETA: 22s - loss: 7.6767 - accuracy: 0.4993
15264/25000 [=================>............] - ETA: 22s - loss: 7.6726 - accuracy: 0.4996
15296/25000 [=================>............] - ETA: 22s - loss: 7.6736 - accuracy: 0.4995
15328/25000 [=================>............] - ETA: 22s - loss: 7.6726 - accuracy: 0.4996
15360/25000 [=================>............] - ETA: 22s - loss: 7.6696 - accuracy: 0.4998
15392/25000 [=================>............] - ETA: 22s - loss: 7.6656 - accuracy: 0.5001
15424/25000 [=================>............] - ETA: 22s - loss: 7.6626 - accuracy: 0.5003
15456/25000 [=================>............] - ETA: 22s - loss: 7.6607 - accuracy: 0.5004
15488/25000 [=================>............] - ETA: 22s - loss: 7.6627 - accuracy: 0.5003
15520/25000 [=================>............] - ETA: 22s - loss: 7.6646 - accuracy: 0.5001
15552/25000 [=================>............] - ETA: 21s - loss: 7.6656 - accuracy: 0.5001
15584/25000 [=================>............] - ETA: 21s - loss: 7.6647 - accuracy: 0.5001
15616/25000 [=================>............] - ETA: 21s - loss: 7.6637 - accuracy: 0.5002
15648/25000 [=================>............] - ETA: 21s - loss: 7.6598 - accuracy: 0.5004
15680/25000 [=================>............] - ETA: 21s - loss: 7.6637 - accuracy: 0.5002
15712/25000 [=================>............] - ETA: 21s - loss: 7.6637 - accuracy: 0.5002
15744/25000 [=================>............] - ETA: 21s - loss: 7.6627 - accuracy: 0.5003
15776/25000 [=================>............] - ETA: 21s - loss: 7.6588 - accuracy: 0.5005
15808/25000 [=================>............] - ETA: 21s - loss: 7.6608 - accuracy: 0.5004
15840/25000 [==================>...........] - ETA: 21s - loss: 7.6666 - accuracy: 0.5000
15872/25000 [==================>...........] - ETA: 21s - loss: 7.6666 - accuracy: 0.5000
15904/25000 [==================>...........] - ETA: 21s - loss: 7.6637 - accuracy: 0.5002
15936/25000 [==================>...........] - ETA: 21s - loss: 7.6637 - accuracy: 0.5002
15968/25000 [==================>...........] - ETA: 21s - loss: 7.6685 - accuracy: 0.4999
16000/25000 [==================>...........] - ETA: 20s - loss: 7.6657 - accuracy: 0.5001
16032/25000 [==================>...........] - ETA: 20s - loss: 7.6676 - accuracy: 0.4999
16064/25000 [==================>...........] - ETA: 20s - loss: 7.6695 - accuracy: 0.4998
16096/25000 [==================>...........] - ETA: 20s - loss: 7.6695 - accuracy: 0.4998
16128/25000 [==================>...........] - ETA: 20s - loss: 7.6676 - accuracy: 0.4999
16160/25000 [==================>...........] - ETA: 20s - loss: 7.6666 - accuracy: 0.5000
16192/25000 [==================>...........] - ETA: 20s - loss: 7.6628 - accuracy: 0.5002
16224/25000 [==================>...........] - ETA: 20s - loss: 7.6628 - accuracy: 0.5002
16256/25000 [==================>...........] - ETA: 20s - loss: 7.6628 - accuracy: 0.5002
16288/25000 [==================>...........] - ETA: 20s - loss: 7.6657 - accuracy: 0.5001
16320/25000 [==================>...........] - ETA: 20s - loss: 7.6619 - accuracy: 0.5003
16352/25000 [==================>...........] - ETA: 20s - loss: 7.6657 - accuracy: 0.5001
16384/25000 [==================>...........] - ETA: 20s - loss: 7.6638 - accuracy: 0.5002
16416/25000 [==================>...........] - ETA: 20s - loss: 7.6619 - accuracy: 0.5003
16448/25000 [==================>...........] - ETA: 19s - loss: 7.6601 - accuracy: 0.5004
16480/25000 [==================>...........] - ETA: 19s - loss: 7.6592 - accuracy: 0.5005
16512/25000 [==================>...........] - ETA: 19s - loss: 7.6610 - accuracy: 0.5004
16544/25000 [==================>...........] - ETA: 19s - loss: 7.6592 - accuracy: 0.5005
16576/25000 [==================>...........] - ETA: 19s - loss: 7.6592 - accuracy: 0.5005
16608/25000 [==================>...........] - ETA: 19s - loss: 7.6592 - accuracy: 0.5005
16640/25000 [==================>...........] - ETA: 19s - loss: 7.6611 - accuracy: 0.5004
16672/25000 [===================>..........] - ETA: 19s - loss: 7.6574 - accuracy: 0.5006
16704/25000 [===================>..........] - ETA: 19s - loss: 7.6556 - accuracy: 0.5007
16736/25000 [===================>..........] - ETA: 19s - loss: 7.6584 - accuracy: 0.5005
16768/25000 [===================>..........] - ETA: 19s - loss: 7.6575 - accuracy: 0.5006
16800/25000 [===================>..........] - ETA: 19s - loss: 7.6557 - accuracy: 0.5007
16832/25000 [===================>..........] - ETA: 19s - loss: 7.6548 - accuracy: 0.5008
16864/25000 [===================>..........] - ETA: 18s - loss: 7.6584 - accuracy: 0.5005
16896/25000 [===================>..........] - ETA: 18s - loss: 7.6521 - accuracy: 0.5009
16928/25000 [===================>..........] - ETA: 18s - loss: 7.6512 - accuracy: 0.5010
16960/25000 [===================>..........] - ETA: 18s - loss: 7.6540 - accuracy: 0.5008
16992/25000 [===================>..........] - ETA: 18s - loss: 7.6567 - accuracy: 0.5006
17024/25000 [===================>..........] - ETA: 18s - loss: 7.6603 - accuracy: 0.5004
17056/25000 [===================>..........] - ETA: 18s - loss: 7.6612 - accuracy: 0.5004
17088/25000 [===================>..........] - ETA: 18s - loss: 7.6612 - accuracy: 0.5004
17120/25000 [===================>..........] - ETA: 18s - loss: 7.6621 - accuracy: 0.5003
17152/25000 [===================>..........] - ETA: 18s - loss: 7.6613 - accuracy: 0.5003
17184/25000 [===================>..........] - ETA: 18s - loss: 7.6595 - accuracy: 0.5005
17216/25000 [===================>..........] - ETA: 18s - loss: 7.6586 - accuracy: 0.5005
17248/25000 [===================>..........] - ETA: 18s - loss: 7.6604 - accuracy: 0.5004
17280/25000 [===================>..........] - ETA: 17s - loss: 7.6631 - accuracy: 0.5002
17312/25000 [===================>..........] - ETA: 17s - loss: 7.6631 - accuracy: 0.5002
17344/25000 [===================>..........] - ETA: 17s - loss: 7.6657 - accuracy: 0.5001
17376/25000 [===================>..........] - ETA: 17s - loss: 7.6649 - accuracy: 0.5001
17408/25000 [===================>..........] - ETA: 17s - loss: 7.6640 - accuracy: 0.5002
17440/25000 [===================>..........] - ETA: 17s - loss: 7.6640 - accuracy: 0.5002
17472/25000 [===================>..........] - ETA: 17s - loss: 7.6622 - accuracy: 0.5003
17504/25000 [====================>.........] - ETA: 17s - loss: 7.6631 - accuracy: 0.5002
17536/25000 [====================>.........] - ETA: 17s - loss: 7.6640 - accuracy: 0.5002
17568/25000 [====================>.........] - ETA: 17s - loss: 7.6631 - accuracy: 0.5002
17600/25000 [====================>.........] - ETA: 17s - loss: 7.6605 - accuracy: 0.5004
17632/25000 [====================>.........] - ETA: 17s - loss: 7.6597 - accuracy: 0.5005
17664/25000 [====================>.........] - ETA: 17s - loss: 7.6614 - accuracy: 0.5003
17696/25000 [====================>.........] - ETA: 16s - loss: 7.6606 - accuracy: 0.5004
17728/25000 [====================>.........] - ETA: 16s - loss: 7.6614 - accuracy: 0.5003
17760/25000 [====================>.........] - ETA: 16s - loss: 7.6666 - accuracy: 0.5000
17792/25000 [====================>.........] - ETA: 16s - loss: 7.6623 - accuracy: 0.5003
17824/25000 [====================>.........] - ETA: 16s - loss: 7.6606 - accuracy: 0.5004
17856/25000 [====================>.........] - ETA: 16s - loss: 7.6666 - accuracy: 0.5000
17888/25000 [====================>.........] - ETA: 16s - loss: 7.6666 - accuracy: 0.5000
17920/25000 [====================>.........] - ETA: 16s - loss: 7.6683 - accuracy: 0.4999
17952/25000 [====================>.........] - ETA: 16s - loss: 7.6683 - accuracy: 0.4999
17984/25000 [====================>.........] - ETA: 16s - loss: 7.6709 - accuracy: 0.4997
18016/25000 [====================>.........] - ETA: 16s - loss: 7.6675 - accuracy: 0.4999
18048/25000 [====================>.........] - ETA: 16s - loss: 7.6641 - accuracy: 0.5002
18080/25000 [====================>.........] - ETA: 16s - loss: 7.6624 - accuracy: 0.5003
18112/25000 [====================>.........] - ETA: 16s - loss: 7.6607 - accuracy: 0.5004
18144/25000 [====================>.........] - ETA: 15s - loss: 7.6615 - accuracy: 0.5003
18176/25000 [====================>.........] - ETA: 15s - loss: 7.6599 - accuracy: 0.5004
18208/25000 [====================>.........] - ETA: 15s - loss: 7.6582 - accuracy: 0.5005
18240/25000 [====================>.........] - ETA: 15s - loss: 7.6582 - accuracy: 0.5005
18272/25000 [====================>.........] - ETA: 15s - loss: 7.6591 - accuracy: 0.5005
18304/25000 [====================>.........] - ETA: 15s - loss: 7.6608 - accuracy: 0.5004
18336/25000 [=====================>........] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
18368/25000 [=====================>........] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
18400/25000 [=====================>........] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
18432/25000 [=====================>........] - ETA: 15s - loss: 7.6683 - accuracy: 0.4999
18464/25000 [=====================>........] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
18496/25000 [=====================>........] - ETA: 15s - loss: 7.6650 - accuracy: 0.5001
18528/25000 [=====================>........] - ETA: 15s - loss: 7.6633 - accuracy: 0.5002
18560/25000 [=====================>........] - ETA: 14s - loss: 7.6608 - accuracy: 0.5004
18592/25000 [=====================>........] - ETA: 14s - loss: 7.6600 - accuracy: 0.5004
18624/25000 [=====================>........] - ETA: 14s - loss: 7.6609 - accuracy: 0.5004
18656/25000 [=====================>........] - ETA: 14s - loss: 7.6617 - accuracy: 0.5003
18688/25000 [=====================>........] - ETA: 14s - loss: 7.6650 - accuracy: 0.5001
18720/25000 [=====================>........] - ETA: 14s - loss: 7.6666 - accuracy: 0.5000
18752/25000 [=====================>........] - ETA: 14s - loss: 7.6642 - accuracy: 0.5002
18784/25000 [=====================>........] - ETA: 14s - loss: 7.6674 - accuracy: 0.4999
18816/25000 [=====================>........] - ETA: 14s - loss: 7.6691 - accuracy: 0.4998
18848/25000 [=====================>........] - ETA: 14s - loss: 7.6666 - accuracy: 0.5000
18880/25000 [=====================>........] - ETA: 14s - loss: 7.6674 - accuracy: 0.4999
18912/25000 [=====================>........] - ETA: 14s - loss: 7.6699 - accuracy: 0.4998
18944/25000 [=====================>........] - ETA: 14s - loss: 7.6723 - accuracy: 0.4996
18976/25000 [=====================>........] - ETA: 14s - loss: 7.6731 - accuracy: 0.4996
19008/25000 [=====================>........] - ETA: 13s - loss: 7.6755 - accuracy: 0.4994
19040/25000 [=====================>........] - ETA: 13s - loss: 7.6771 - accuracy: 0.4993
19072/25000 [=====================>........] - ETA: 13s - loss: 7.6787 - accuracy: 0.4992
19104/25000 [=====================>........] - ETA: 13s - loss: 7.6779 - accuracy: 0.4993
19136/25000 [=====================>........] - ETA: 13s - loss: 7.6834 - accuracy: 0.4989
19168/25000 [======================>.......] - ETA: 13s - loss: 7.6850 - accuracy: 0.4988
19200/25000 [======================>.......] - ETA: 13s - loss: 7.6866 - accuracy: 0.4987
19232/25000 [======================>.......] - ETA: 13s - loss: 7.6866 - accuracy: 0.4987
19264/25000 [======================>.......] - ETA: 13s - loss: 7.6865 - accuracy: 0.4987
19296/25000 [======================>.......] - ETA: 13s - loss: 7.6857 - accuracy: 0.4988
19328/25000 [======================>.......] - ETA: 13s - loss: 7.6872 - accuracy: 0.4987
19360/25000 [======================>.......] - ETA: 13s - loss: 7.6864 - accuracy: 0.4987
19392/25000 [======================>.......] - ETA: 13s - loss: 7.6848 - accuracy: 0.4988
19424/25000 [======================>.......] - ETA: 12s - loss: 7.6864 - accuracy: 0.4987
19456/25000 [======================>.......] - ETA: 12s - loss: 7.6863 - accuracy: 0.4987
19488/25000 [======================>.......] - ETA: 12s - loss: 7.6847 - accuracy: 0.4988
19520/25000 [======================>.......] - ETA: 12s - loss: 7.6894 - accuracy: 0.4985
19552/25000 [======================>.......] - ETA: 12s - loss: 7.6901 - accuracy: 0.4985
19584/25000 [======================>.......] - ETA: 12s - loss: 7.6909 - accuracy: 0.4984
19616/25000 [======================>.......] - ETA: 12s - loss: 7.6893 - accuracy: 0.4985
19648/25000 [======================>.......] - ETA: 12s - loss: 7.6877 - accuracy: 0.4986
19680/25000 [======================>.......] - ETA: 12s - loss: 7.6884 - accuracy: 0.4986
19712/25000 [======================>.......] - ETA: 12s - loss: 7.6900 - accuracy: 0.4985
19744/25000 [======================>.......] - ETA: 12s - loss: 7.6891 - accuracy: 0.4985
19776/25000 [======================>.......] - ETA: 12s - loss: 7.6883 - accuracy: 0.4986
19808/25000 [======================>.......] - ETA: 12s - loss: 7.6875 - accuracy: 0.4986
19840/25000 [======================>.......] - ETA: 11s - loss: 7.6898 - accuracy: 0.4985
19872/25000 [======================>.......] - ETA: 11s - loss: 7.6929 - accuracy: 0.4983
19904/25000 [======================>.......] - ETA: 11s - loss: 7.6897 - accuracy: 0.4985
19936/25000 [======================>.......] - ETA: 11s - loss: 7.6912 - accuracy: 0.4984
19968/25000 [======================>.......] - ETA: 11s - loss: 7.6897 - accuracy: 0.4985
20000/25000 [=======================>......] - ETA: 11s - loss: 7.6866 - accuracy: 0.4987
20032/25000 [=======================>......] - ETA: 11s - loss: 7.6842 - accuracy: 0.4989
20064/25000 [=======================>......] - ETA: 11s - loss: 7.6834 - accuracy: 0.4989
20096/25000 [=======================>......] - ETA: 11s - loss: 7.6865 - accuracy: 0.4987
20128/25000 [=======================>......] - ETA: 11s - loss: 7.6879 - accuracy: 0.4986
20160/25000 [=======================>......] - ETA: 11s - loss: 7.6872 - accuracy: 0.4987
20192/25000 [=======================>......] - ETA: 11s - loss: 7.6886 - accuracy: 0.4986
20224/25000 [=======================>......] - ETA: 11s - loss: 7.6863 - accuracy: 0.4987
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6863 - accuracy: 0.4987
20288/25000 [=======================>......] - ETA: 10s - loss: 7.6863 - accuracy: 0.4987
20320/25000 [=======================>......] - ETA: 10s - loss: 7.6847 - accuracy: 0.4988
20352/25000 [=======================>......] - ETA: 10s - loss: 7.6839 - accuracy: 0.4989
20384/25000 [=======================>......] - ETA: 10s - loss: 7.6847 - accuracy: 0.4988
20416/25000 [=======================>......] - ETA: 10s - loss: 7.6846 - accuracy: 0.4988
20448/25000 [=======================>......] - ETA: 10s - loss: 7.6839 - accuracy: 0.4989
20480/25000 [=======================>......] - ETA: 10s - loss: 7.6876 - accuracy: 0.4986
20512/25000 [=======================>......] - ETA: 10s - loss: 7.6853 - accuracy: 0.4988
20544/25000 [=======================>......] - ETA: 10s - loss: 7.6860 - accuracy: 0.4987
20576/25000 [=======================>......] - ETA: 10s - loss: 7.6815 - accuracy: 0.4990
20608/25000 [=======================>......] - ETA: 10s - loss: 7.6830 - accuracy: 0.4989
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6844 - accuracy: 0.4988
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6844 - accuracy: 0.4988
20704/25000 [=======================>......] - ETA: 9s - loss: 7.6829 - accuracy: 0.4989 
20736/25000 [=======================>......] - ETA: 9s - loss: 7.6829 - accuracy: 0.4989
20768/25000 [=======================>......] - ETA: 9s - loss: 7.6843 - accuracy: 0.4988
20800/25000 [=======================>......] - ETA: 9s - loss: 7.6828 - accuracy: 0.4989
20832/25000 [=======================>......] - ETA: 9s - loss: 7.6813 - accuracy: 0.4990
20864/25000 [========================>.....] - ETA: 9s - loss: 7.6872 - accuracy: 0.4987
20896/25000 [========================>.....] - ETA: 9s - loss: 7.6835 - accuracy: 0.4989
20928/25000 [========================>.....] - ETA: 9s - loss: 7.6827 - accuracy: 0.4989
20960/25000 [========================>.....] - ETA: 9s - loss: 7.6842 - accuracy: 0.4989
20992/25000 [========================>.....] - ETA: 9s - loss: 7.6820 - accuracy: 0.4990
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6805 - accuracy: 0.4991
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6783 - accuracy: 0.4992
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6797 - accuracy: 0.4991
21120/25000 [========================>.....] - ETA: 8s - loss: 7.6782 - accuracy: 0.4992
21152/25000 [========================>.....] - ETA: 8s - loss: 7.6789 - accuracy: 0.4992
21184/25000 [========================>.....] - ETA: 8s - loss: 7.6825 - accuracy: 0.4990
21216/25000 [========================>.....] - ETA: 8s - loss: 7.6825 - accuracy: 0.4990
21248/25000 [========================>.....] - ETA: 8s - loss: 7.6825 - accuracy: 0.4990
21280/25000 [========================>.....] - ETA: 8s - loss: 7.6825 - accuracy: 0.4990
21312/25000 [========================>.....] - ETA: 8s - loss: 7.6796 - accuracy: 0.4992
21344/25000 [========================>.....] - ETA: 8s - loss: 7.6774 - accuracy: 0.4993
21376/25000 [========================>.....] - ETA: 8s - loss: 7.6788 - accuracy: 0.4992
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6759 - accuracy: 0.4994
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6773 - accuracy: 0.4993
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6788 - accuracy: 0.4992
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6737 - accuracy: 0.4995
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6716 - accuracy: 0.4997
21568/25000 [========================>.....] - ETA: 7s - loss: 7.6702 - accuracy: 0.4998
21600/25000 [========================>.....] - ETA: 7s - loss: 7.6702 - accuracy: 0.4998
21632/25000 [========================>.....] - ETA: 7s - loss: 7.6702 - accuracy: 0.4998
21664/25000 [========================>.....] - ETA: 7s - loss: 7.6680 - accuracy: 0.4999
21696/25000 [=========================>....] - ETA: 7s - loss: 7.6652 - accuracy: 0.5001
21728/25000 [=========================>....] - ETA: 7s - loss: 7.6680 - accuracy: 0.4999
21760/25000 [=========================>....] - ETA: 7s - loss: 7.6716 - accuracy: 0.4997
21792/25000 [=========================>....] - ETA: 7s - loss: 7.6722 - accuracy: 0.4996
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6701 - accuracy: 0.4998
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6687 - accuracy: 0.4999
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6687 - accuracy: 0.4999
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6687 - accuracy: 0.4999
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6701 - accuracy: 0.4998
21984/25000 [=========================>....] - ETA: 6s - loss: 7.6715 - accuracy: 0.4997
22016/25000 [=========================>....] - ETA: 6s - loss: 7.6722 - accuracy: 0.4996
22048/25000 [=========================>....] - ETA: 6s - loss: 7.6757 - accuracy: 0.4994
22080/25000 [=========================>....] - ETA: 6s - loss: 7.6777 - accuracy: 0.4993
22112/25000 [=========================>....] - ETA: 6s - loss: 7.6791 - accuracy: 0.4992
22144/25000 [=========================>....] - ETA: 6s - loss: 7.6777 - accuracy: 0.4993
22176/25000 [=========================>....] - ETA: 6s - loss: 7.6763 - accuracy: 0.4994
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6770 - accuracy: 0.4993
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6783 - accuracy: 0.4992
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6797 - accuracy: 0.4991
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6817 - accuracy: 0.4990
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6810 - accuracy: 0.4991
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6810 - accuracy: 0.4991
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6810 - accuracy: 0.4991
22432/25000 [=========================>....] - ETA: 5s - loss: 7.6810 - accuracy: 0.4991
22464/25000 [=========================>....] - ETA: 5s - loss: 7.6782 - accuracy: 0.4992
22496/25000 [=========================>....] - ETA: 5s - loss: 7.6789 - accuracy: 0.4992
22528/25000 [==========================>...] - ETA: 5s - loss: 7.6796 - accuracy: 0.4992
22560/25000 [==========================>...] - ETA: 5s - loss: 7.6775 - accuracy: 0.4993
22592/25000 [==========================>...] - ETA: 5s - loss: 7.6782 - accuracy: 0.4992
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6775 - accuracy: 0.4993
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6781 - accuracy: 0.4992
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6768 - accuracy: 0.4993
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6774 - accuracy: 0.4993
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6788 - accuracy: 0.4992
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6787 - accuracy: 0.4992
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6787 - accuracy: 0.4992
22848/25000 [==========================>...] - ETA: 4s - loss: 7.6760 - accuracy: 0.4994
22880/25000 [==========================>...] - ETA: 4s - loss: 7.6773 - accuracy: 0.4993
22912/25000 [==========================>...] - ETA: 4s - loss: 7.6787 - accuracy: 0.4992
22944/25000 [==========================>...] - ETA: 4s - loss: 7.6807 - accuracy: 0.4991
22976/25000 [==========================>...] - ETA: 4s - loss: 7.6793 - accuracy: 0.4992
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6813 - accuracy: 0.4990
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6846 - accuracy: 0.4988
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6806 - accuracy: 0.4991
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6832 - accuracy: 0.4989
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6819 - accuracy: 0.4990
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6799 - accuracy: 0.4991
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6818 - accuracy: 0.4990
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6785 - accuracy: 0.4992
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6791 - accuracy: 0.4992
23296/25000 [==========================>...] - ETA: 3s - loss: 7.6772 - accuracy: 0.4993
23328/25000 [==========================>...] - ETA: 3s - loss: 7.6791 - accuracy: 0.4992
23360/25000 [===========================>..] - ETA: 3s - loss: 7.6784 - accuracy: 0.4992
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6797 - accuracy: 0.4991
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6823 - accuracy: 0.4990
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6771 - accuracy: 0.4993
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6784 - accuracy: 0.4992
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6790 - accuracy: 0.4992
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6803 - accuracy: 0.4991
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6796 - accuracy: 0.4992
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6777 - accuracy: 0.4993
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6809 - accuracy: 0.4991
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6815 - accuracy: 0.4990
23712/25000 [===========================>..] - ETA: 2s - loss: 7.6821 - accuracy: 0.4990
23744/25000 [===========================>..] - ETA: 2s - loss: 7.6802 - accuracy: 0.4991
23776/25000 [===========================>..] - ETA: 2s - loss: 7.6808 - accuracy: 0.4991
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6782 - accuracy: 0.4992
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6776 - accuracy: 0.4993
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6788 - accuracy: 0.4992
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6705 - accuracy: 0.4997
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6711 - accuracy: 0.4997
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24160/25000 [===========================>..] - ETA: 1s - loss: 7.6711 - accuracy: 0.4997
24192/25000 [============================>.] - ETA: 1s - loss: 7.6698 - accuracy: 0.4998
24224/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24256/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24288/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24320/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24352/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24384/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24416/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24448/25000 [============================>.] - ETA: 1s - loss: 7.6629 - accuracy: 0.5002
24480/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24512/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24544/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24576/25000 [============================>.] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
24608/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24640/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24672/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24704/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24736/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24768/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24800/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24864/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24896/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24928/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24960/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
25000/25000 [==============================] - 68s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fd84da63a90> 

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
 [ 1.54638767e-01 -1.51149742e-02 -5.50739206e-02 -3.59312817e-02
  -2.72097290e-02 -7.23208264e-02]
 [ 5.73162213e-02  1.34966388e-01  1.18098557e-01 -1.75640538e-01
   9.63378400e-02  8.53242800e-02]
 [ 8.18052096e-04 -1.52718470e-01  1.90389886e-01  2.75729835e-01
   1.03217140e-01  1.04523487e-01]
 [ 5.16519621e-02  7.74057060e-02  4.91849571e-01 -1.46374732e-01
   9.56035405e-03  3.00739259e-01]
 [-5.93586922e-01  1.51510447e-01  1.51843444e-01  1.11210942e-02
   1.81999877e-01  1.12966947e-01]
 [ 3.82131159e-01  8.26679587e-01  4.76112783e-01  2.21096650e-01
   7.37258554e-01  3.22283626e-01]
 [-1.20241761e-01  3.98921847e-01 -1.86282307e-01 -2.79065847e-01
  -2.67096013e-01  3.75275433e-01]
 [ 4.51894104e-01  2.56181926e-01 -3.52763623e-01  2.28709593e-01
  -8.05628747e-02 -5.97657740e-01]
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
{'loss': 0.395203098654747, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}

  #### Load   ######################################################## 
2020-05-11 05:56:02.873175: W tensorflow/core/util/tensor_slice_reader.cc:95] Could not open /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
2020-05-11 05:56:02.874309: W tensorflow/core/util/tensor_slice_reader.cc:95] Could not open /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
2020-05-11 05:56:02.874343: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_tensor.cc:175 : Data loss: Unable to open table file /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed Unable to open table file /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 506, in main
    test_cli(arg)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 438, in test_cli
    test(arg.model_uri)  # '1_lstm'
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 187, in test
    module.test()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 445, in load_tf
    saver = tf.compat.v1.train.Saver()
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
{'loss': 0.5246107131242752, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}

  #### Load   ######################################################## 
2020-05-11 05:56:03.922920: W tensorflow/core/util/tensor_slice_reader.cc:95] Could not open /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
2020-05-11 05:56:03.924010: W tensorflow/core/util/tensor_slice_reader.cc:95] Could not open /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
2020-05-11 05:56:03.924040: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_tensor.cc:175 : Data loss: Unable to open table file /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed Unable to open table file /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 506, in main
    test_cli(arg)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 440, in test_cli
    test_global(arg.model_uri)  # '1_lstm'
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 198, in test_global
    module.test()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 445, in load_tf
    saver = tf.compat.v1.train.Saver()
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
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:45<01:08, 22.90s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.3251392622157649, 'embedding_size_factor': 1.2153286040990698, 'layers.choice': 3, 'learning_rate': 0.00042970819000298985, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 0.0064388587434363615} and reward: 0.3734
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\xcf\x14\xe8w/\x93X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3q\xfch\x07\xfb<X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?<)N\x9fux?X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?z_\xa1\xfb\xa0?Bu.' and reward: 0.3734
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\xcf\x14\xe8w/\x93X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3q\xfch\x07\xfb<X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?<)N\x9fux?X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?z_\xa1\xfb\xa0?Bu.' and reward: 0.3734
 60%|██████    | 3/5 [01:36<01:02, 31.10s/it] 60%|██████    | 3/5 [01:36<01:04, 32.01s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.21421369023911796, 'embedding_size_factor': 1.0207844191583912, 'layers.choice': 1, 'learning_rate': 0.009714048889916659, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 0.00628149917727097} and reward: 0.366
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xcbkZ\xac\xf7[\xb4X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0U"\x0b\x08\xd3\x8bX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x83\xe4\xf5\x92Y\xfb\xb9X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?y\xba\xa1\x18\x94e\x9eu.' and reward: 0.366
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xcbkZ\xac\xf7[\xb4X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0U"\x0b\x08\xd3\x8bX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x83\xe4\xf5\x92Y\xfb\xb9X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?y\xba\xa1\x18\x94e\x9eu.' and reward: 0.366
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 171.2655050754547
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.77s of the -53.56s of remaining time.
Ensemble size: 27
Ensemble weights: 
[0.85185185 0.14814815 0.        ]
	0.3894	 = Validation accuracy score
	0.94s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 174.53s ...
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

