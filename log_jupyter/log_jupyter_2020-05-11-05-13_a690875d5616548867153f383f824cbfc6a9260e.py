
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/a690875d5616548867153f383f824cbfc6a9260e', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'a690875d5616548867153f383f824cbfc6a9260e', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/a690875d5616548867153f383f824cbfc6a9260e

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/a690875d5616548867153f383f824cbfc6a9260e

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
  925696/17464789 [>.............................] - ETA: 2s
 1343488/17464789 [=>............................] - ETA: 2s
 1835008/17464789 [==>...........................] - ETA: 2s
 2383872/17464789 [===>..........................] - ETA: 2s
 2965504/17464789 [====>.........................] - ETA: 1s
 3604480/17464789 [=====>........................] - ETA: 1s
 4333568/17464789 [======>.......................] - ETA: 1s
 5103616/17464789 [=======>......................] - ETA: 1s
 5939200/17464789 [=========>....................] - ETA: 1s
 6864896/17464789 [==========>...................] - ETA: 0s
 7749632/17464789 [============>.................] - ETA: 0s
 8601600/17464789 [=============>................] - ETA: 0s
 9576448/17464789 [===============>..............] - ETA: 0s
10608640/17464789 [=================>............] - ETA: 0s
11649024/17464789 [===================>..........] - ETA: 0s
12435456/17464789 [====================>.........] - ETA: 0s
13303808/17464789 [=====================>........] - ETA: 0s
14172160/17464789 [=======================>......] - ETA: 0s
14868480/17464789 [========================>.....] - ETA: 0s
15638528/17464789 [=========================>....] - ETA: 0s
16400384/17464789 [===========================>..] - ETA: 0s
17268736/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 05:13:49.071909: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 05:13:49.075994: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-11 05:13:49.076136: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560bca81d360 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 05:13:49.076150: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:23 - loss: 7.6666 - accuracy: 0.5000
   64/25000 [..............................] - ETA: 2:43 - loss: 6.2291 - accuracy: 0.5938
   96/25000 [..............................] - ETA: 2:07 - loss: 6.3888 - accuracy: 0.5833
  128/25000 [..............................] - ETA: 1:50 - loss: 6.9479 - accuracy: 0.5469
  160/25000 [..............................] - ETA: 1:39 - loss: 7.2833 - accuracy: 0.5250
  192/25000 [..............................] - ETA: 1:31 - loss: 7.3472 - accuracy: 0.5208
  224/25000 [..............................] - ETA: 1:26 - loss: 7.4613 - accuracy: 0.5134
  256/25000 [..............................] - ETA: 1:22 - loss: 7.6666 - accuracy: 0.5000
  288/25000 [..............................] - ETA: 1:19 - loss: 7.9328 - accuracy: 0.4826
  320/25000 [..............................] - ETA: 1:17 - loss: 7.9541 - accuracy: 0.4812
  352/25000 [..............................] - ETA: 1:15 - loss: 7.9280 - accuracy: 0.4830
  384/25000 [..............................] - ETA: 1:14 - loss: 7.8263 - accuracy: 0.4896
  416/25000 [..............................] - ETA: 1:12 - loss: 7.8878 - accuracy: 0.4856
  448/25000 [..............................] - ETA: 1:11 - loss: 7.9062 - accuracy: 0.4844
  480/25000 [..............................] - ETA: 1:10 - loss: 7.8583 - accuracy: 0.4875
  512/25000 [..............................] - ETA: 1:09 - loss: 7.8164 - accuracy: 0.4902
  544/25000 [..............................] - ETA: 1:08 - loss: 7.8357 - accuracy: 0.4890
  576/25000 [..............................] - ETA: 1:07 - loss: 7.6666 - accuracy: 0.5000
  608/25000 [..............................] - ETA: 1:06 - loss: 7.6666 - accuracy: 0.5000
  640/25000 [..............................] - ETA: 1:06 - loss: 7.7385 - accuracy: 0.4953
  672/25000 [..............................] - ETA: 1:05 - loss: 7.7351 - accuracy: 0.4955
  704/25000 [..............................] - ETA: 1:05 - loss: 7.8409 - accuracy: 0.4886
  736/25000 [..............................] - ETA: 1:05 - loss: 7.8333 - accuracy: 0.4891
  768/25000 [..............................] - ETA: 1:04 - loss: 7.7664 - accuracy: 0.4935
  800/25000 [..............................] - ETA: 1:04 - loss: 7.7625 - accuracy: 0.4938
  832/25000 [..............................] - ETA: 1:04 - loss: 7.7956 - accuracy: 0.4916
  864/25000 [>.............................] - ETA: 1:03 - loss: 7.7376 - accuracy: 0.4954
  896/25000 [>.............................] - ETA: 1:03 - loss: 7.6837 - accuracy: 0.4989
  928/25000 [>.............................] - ETA: 1:03 - loss: 7.6831 - accuracy: 0.4989
  960/25000 [>.............................] - ETA: 1:03 - loss: 7.6347 - accuracy: 0.5021
  992/25000 [>.............................] - ETA: 1:02 - loss: 7.6666 - accuracy: 0.5000
 1024/25000 [>.............................] - ETA: 1:02 - loss: 7.6966 - accuracy: 0.4980
 1056/25000 [>.............................] - ETA: 1:01 - loss: 7.7392 - accuracy: 0.4953
 1088/25000 [>.............................] - ETA: 1:01 - loss: 7.7935 - accuracy: 0.4917
 1120/25000 [>.............................] - ETA: 1:01 - loss: 7.7351 - accuracy: 0.4955
 1152/25000 [>.............................] - ETA: 1:01 - loss: 7.7065 - accuracy: 0.4974
 1184/25000 [>.............................] - ETA: 1:00 - loss: 7.6925 - accuracy: 0.4983
 1216/25000 [>.............................] - ETA: 1:00 - loss: 7.6540 - accuracy: 0.5008
 1248/25000 [>.............................] - ETA: 1:00 - loss: 7.6666 - accuracy: 0.5000
 1280/25000 [>.............................] - ETA: 1:00 - loss: 7.7026 - accuracy: 0.4977
 1312/25000 [>.............................] - ETA: 59s - loss: 7.7134 - accuracy: 0.4970 
 1344/25000 [>.............................] - ETA: 59s - loss: 7.7351 - accuracy: 0.4955
 1376/25000 [>.............................] - ETA: 59s - loss: 7.6889 - accuracy: 0.4985
 1408/25000 [>.............................] - ETA: 59s - loss: 7.6884 - accuracy: 0.4986
 1440/25000 [>.............................] - ETA: 59s - loss: 7.6773 - accuracy: 0.4993
 1472/25000 [>.............................] - ETA: 58s - loss: 7.6875 - accuracy: 0.4986
 1504/25000 [>.............................] - ETA: 58s - loss: 7.7176 - accuracy: 0.4967
 1536/25000 [>.............................] - ETA: 58s - loss: 7.7165 - accuracy: 0.4967
 1568/25000 [>.............................] - ETA: 58s - loss: 7.7057 - accuracy: 0.4974
 1600/25000 [>.............................] - ETA: 58s - loss: 7.7337 - accuracy: 0.4956
 1632/25000 [>.............................] - ETA: 58s - loss: 7.7324 - accuracy: 0.4957
 1664/25000 [>.............................] - ETA: 58s - loss: 7.7956 - accuracy: 0.4916
 1696/25000 [=>............................] - ETA: 57s - loss: 7.7570 - accuracy: 0.4941
 1728/25000 [=>............................] - ETA: 57s - loss: 7.7997 - accuracy: 0.4913
 1760/25000 [=>............................] - ETA: 57s - loss: 7.8147 - accuracy: 0.4903
 1792/25000 [=>............................] - ETA: 57s - loss: 7.8377 - accuracy: 0.4888
 1824/25000 [=>............................] - ETA: 57s - loss: 7.8095 - accuracy: 0.4907
 1856/25000 [=>............................] - ETA: 57s - loss: 7.8318 - accuracy: 0.4892
 1888/25000 [=>............................] - ETA: 57s - loss: 7.8290 - accuracy: 0.4894
 1920/25000 [=>............................] - ETA: 57s - loss: 7.8184 - accuracy: 0.4901
 1952/25000 [=>............................] - ETA: 57s - loss: 7.8473 - accuracy: 0.4882
 1984/25000 [=>............................] - ETA: 56s - loss: 7.8135 - accuracy: 0.4904
 2016/25000 [=>............................] - ETA: 57s - loss: 7.7883 - accuracy: 0.4921
 2048/25000 [=>............................] - ETA: 56s - loss: 7.7789 - accuracy: 0.4927
 2080/25000 [=>............................] - ETA: 56s - loss: 7.7772 - accuracy: 0.4928
 2112/25000 [=>............................] - ETA: 56s - loss: 7.7683 - accuracy: 0.4934
 2144/25000 [=>............................] - ETA: 56s - loss: 7.7810 - accuracy: 0.4925
 2176/25000 [=>............................] - ETA: 56s - loss: 7.7653 - accuracy: 0.4936
 2208/25000 [=>............................] - ETA: 56s - loss: 7.7361 - accuracy: 0.4955
 2240/25000 [=>............................] - ETA: 56s - loss: 7.7351 - accuracy: 0.4955
 2272/25000 [=>............................] - ETA: 56s - loss: 7.7274 - accuracy: 0.4960
 2304/25000 [=>............................] - ETA: 56s - loss: 7.7132 - accuracy: 0.4970
 2336/25000 [=>............................] - ETA: 56s - loss: 7.7454 - accuracy: 0.4949
 2368/25000 [=>............................] - ETA: 56s - loss: 7.7637 - accuracy: 0.4937
 2400/25000 [=>............................] - ETA: 55s - loss: 7.7752 - accuracy: 0.4929
 2432/25000 [=>............................] - ETA: 55s - loss: 7.7864 - accuracy: 0.4922
 2464/25000 [=>............................] - ETA: 55s - loss: 7.7537 - accuracy: 0.4943
 2496/25000 [=>............................] - ETA: 55s - loss: 7.7772 - accuracy: 0.4928
 2528/25000 [==>...........................] - ETA: 55s - loss: 7.8001 - accuracy: 0.4913
 2560/25000 [==>...........................] - ETA: 55s - loss: 7.7864 - accuracy: 0.4922
 2592/25000 [==>...........................] - ETA: 55s - loss: 7.7731 - accuracy: 0.4931
 2624/25000 [==>...........................] - ETA: 55s - loss: 7.7660 - accuracy: 0.4935
 2656/25000 [==>...........................] - ETA: 54s - loss: 7.7648 - accuracy: 0.4936
 2688/25000 [==>...........................] - ETA: 54s - loss: 7.7750 - accuracy: 0.4929
 2720/25000 [==>...........................] - ETA: 54s - loss: 7.7737 - accuracy: 0.4930
 2752/25000 [==>...........................] - ETA: 54s - loss: 7.7669 - accuracy: 0.4935
 2784/25000 [==>...........................] - ETA: 54s - loss: 7.7547 - accuracy: 0.4943
 2816/25000 [==>...........................] - ETA: 54s - loss: 7.7646 - accuracy: 0.4936
 2848/25000 [==>...........................] - ETA: 54s - loss: 7.7420 - accuracy: 0.4951
 2880/25000 [==>...........................] - ETA: 54s - loss: 7.7625 - accuracy: 0.4938
 2912/25000 [==>...........................] - ETA: 54s - loss: 7.7825 - accuracy: 0.4924
 2944/25000 [==>...........................] - ETA: 54s - loss: 7.8020 - accuracy: 0.4912
 2976/25000 [==>...........................] - ETA: 53s - loss: 7.8006 - accuracy: 0.4913
 3008/25000 [==>...........................] - ETA: 53s - loss: 7.7941 - accuracy: 0.4917
 3040/25000 [==>...........................] - ETA: 53s - loss: 7.7927 - accuracy: 0.4918
 3072/25000 [==>...........................] - ETA: 53s - loss: 7.8014 - accuracy: 0.4912
 3104/25000 [==>...........................] - ETA: 53s - loss: 7.7951 - accuracy: 0.4916
 3136/25000 [==>...........................] - ETA: 53s - loss: 7.7840 - accuracy: 0.4923
 3168/25000 [==>...........................] - ETA: 53s - loss: 7.7731 - accuracy: 0.4931
 3200/25000 [==>...........................] - ETA: 53s - loss: 7.7385 - accuracy: 0.4953
 3232/25000 [==>...........................] - ETA: 53s - loss: 7.7473 - accuracy: 0.4947
 3264/25000 [==>...........................] - ETA: 53s - loss: 7.7418 - accuracy: 0.4951
 3296/25000 [==>...........................] - ETA: 53s - loss: 7.7411 - accuracy: 0.4951
 3328/25000 [==>...........................] - ETA: 52s - loss: 7.7403 - accuracy: 0.4952
 3360/25000 [===>..........................] - ETA: 52s - loss: 7.7168 - accuracy: 0.4967
 3392/25000 [===>..........................] - ETA: 52s - loss: 7.7163 - accuracy: 0.4968
 3424/25000 [===>..........................] - ETA: 52s - loss: 7.7338 - accuracy: 0.4956
 3456/25000 [===>..........................] - ETA: 52s - loss: 7.7554 - accuracy: 0.4942
 3488/25000 [===>..........................] - ETA: 52s - loss: 7.7677 - accuracy: 0.4934
 3520/25000 [===>..........................] - ETA: 52s - loss: 7.7581 - accuracy: 0.4940
 3552/25000 [===>..........................] - ETA: 52s - loss: 7.7486 - accuracy: 0.4947
 3584/25000 [===>..........................] - ETA: 52s - loss: 7.7565 - accuracy: 0.4941
 3616/25000 [===>..........................] - ETA: 52s - loss: 7.7514 - accuracy: 0.4945
 3648/25000 [===>..........................] - ETA: 52s - loss: 7.7465 - accuracy: 0.4948
 3680/25000 [===>..........................] - ETA: 52s - loss: 7.7416 - accuracy: 0.4951
 3712/25000 [===>..........................] - ETA: 51s - loss: 7.7368 - accuracy: 0.4954
 3744/25000 [===>..........................] - ETA: 51s - loss: 7.7281 - accuracy: 0.4960
 3776/25000 [===>..........................] - ETA: 51s - loss: 7.7275 - accuracy: 0.4960
 3808/25000 [===>..........................] - ETA: 51s - loss: 7.7310 - accuracy: 0.4958
 3840/25000 [===>..........................] - ETA: 51s - loss: 7.7225 - accuracy: 0.4964
 3872/25000 [===>..........................] - ETA: 51s - loss: 7.7221 - accuracy: 0.4964
 3904/25000 [===>..........................] - ETA: 51s - loss: 7.7373 - accuracy: 0.4954
 3936/25000 [===>..........................] - ETA: 51s - loss: 7.7134 - accuracy: 0.4970
 3968/25000 [===>..........................] - ETA: 51s - loss: 7.7207 - accuracy: 0.4965
 4000/25000 [===>..........................] - ETA: 51s - loss: 7.7318 - accuracy: 0.4958
 4032/25000 [===>..........................] - ETA: 51s - loss: 7.7275 - accuracy: 0.4960
 4064/25000 [===>..........................] - ETA: 50s - loss: 7.7345 - accuracy: 0.4956
 4096/25000 [===>..........................] - ETA: 50s - loss: 7.7340 - accuracy: 0.4956
 4128/25000 [===>..........................] - ETA: 50s - loss: 7.7335 - accuracy: 0.4956
 4160/25000 [===>..........................] - ETA: 50s - loss: 7.7109 - accuracy: 0.4971
 4192/25000 [====>.........................] - ETA: 50s - loss: 7.7215 - accuracy: 0.4964
 4224/25000 [====>.........................] - ETA: 50s - loss: 7.7138 - accuracy: 0.4969
 4256/25000 [====>.........................] - ETA: 50s - loss: 7.7099 - accuracy: 0.4972
 4288/25000 [====>.........................] - ETA: 50s - loss: 7.6988 - accuracy: 0.4979
 4320/25000 [====>.........................] - ETA: 50s - loss: 7.6844 - accuracy: 0.4988
 4352/25000 [====>.........................] - ETA: 50s - loss: 7.6913 - accuracy: 0.4984
 4384/25000 [====>.........................] - ETA: 50s - loss: 7.6771 - accuracy: 0.4993
 4416/25000 [====>.........................] - ETA: 50s - loss: 7.6909 - accuracy: 0.4984
 4448/25000 [====>.........................] - ETA: 50s - loss: 7.7080 - accuracy: 0.4973
 4480/25000 [====>.........................] - ETA: 49s - loss: 7.7043 - accuracy: 0.4975
 4512/25000 [====>.........................] - ETA: 49s - loss: 7.7006 - accuracy: 0.4978
 4544/25000 [====>.........................] - ETA: 49s - loss: 7.6936 - accuracy: 0.4982
 4576/25000 [====>.........................] - ETA: 49s - loss: 7.6867 - accuracy: 0.4987
 4608/25000 [====>.........................] - ETA: 49s - loss: 7.6866 - accuracy: 0.4987
 4640/25000 [====>.........................] - ETA: 49s - loss: 7.6864 - accuracy: 0.4987
 4672/25000 [====>.........................] - ETA: 49s - loss: 7.6732 - accuracy: 0.4996
 4704/25000 [====>.........................] - ETA: 49s - loss: 7.6731 - accuracy: 0.4996
 4736/25000 [====>.........................] - ETA: 49s - loss: 7.6731 - accuracy: 0.4996
 4768/25000 [====>.........................] - ETA: 49s - loss: 7.6827 - accuracy: 0.4990
 4800/25000 [====>.........................] - ETA: 49s - loss: 7.6986 - accuracy: 0.4979
 4832/25000 [====>.........................] - ETA: 49s - loss: 7.6888 - accuracy: 0.4986
 4864/25000 [====>.........................] - ETA: 48s - loss: 7.6698 - accuracy: 0.4998
 4896/25000 [====>.........................] - ETA: 48s - loss: 7.6604 - accuracy: 0.5004
 4928/25000 [====>.........................] - ETA: 48s - loss: 7.6573 - accuracy: 0.5006
 4960/25000 [====>.........................] - ETA: 48s - loss: 7.6512 - accuracy: 0.5010
 4992/25000 [====>.........................] - ETA: 48s - loss: 7.6451 - accuracy: 0.5014
 5024/25000 [=====>........................] - ETA: 48s - loss: 7.6330 - accuracy: 0.5022
 5056/25000 [=====>........................] - ETA: 48s - loss: 7.6363 - accuracy: 0.5020
 5088/25000 [=====>........................] - ETA: 48s - loss: 7.6335 - accuracy: 0.5022
 5120/25000 [=====>........................] - ETA: 48s - loss: 7.6427 - accuracy: 0.5016
 5152/25000 [=====>........................] - ETA: 48s - loss: 7.6369 - accuracy: 0.5019
 5184/25000 [=====>........................] - ETA: 48s - loss: 7.6430 - accuracy: 0.5015
 5216/25000 [=====>........................] - ETA: 48s - loss: 7.6313 - accuracy: 0.5023
 5248/25000 [=====>........................] - ETA: 47s - loss: 7.6170 - accuracy: 0.5032
 5280/25000 [=====>........................] - ETA: 47s - loss: 7.6027 - accuracy: 0.5042
 5312/25000 [=====>........................] - ETA: 47s - loss: 7.6002 - accuracy: 0.5043
 5344/25000 [=====>........................] - ETA: 47s - loss: 7.5891 - accuracy: 0.5051
 5376/25000 [=====>........................] - ETA: 47s - loss: 7.5896 - accuracy: 0.5050
 5408/25000 [=====>........................] - ETA: 47s - loss: 7.5986 - accuracy: 0.5044
 5440/25000 [=====>........................] - ETA: 47s - loss: 7.6131 - accuracy: 0.5035
 5472/25000 [=====>........................] - ETA: 47s - loss: 7.6134 - accuracy: 0.5035
 5504/25000 [=====>........................] - ETA: 47s - loss: 7.6053 - accuracy: 0.5040
 5536/25000 [=====>........................] - ETA: 47s - loss: 7.6168 - accuracy: 0.5033
 5568/25000 [=====>........................] - ETA: 47s - loss: 7.6115 - accuracy: 0.5036
 5600/25000 [=====>........................] - ETA: 46s - loss: 7.6036 - accuracy: 0.5041
 5632/25000 [=====>........................] - ETA: 46s - loss: 7.6013 - accuracy: 0.5043
 5664/25000 [=====>........................] - ETA: 46s - loss: 7.6016 - accuracy: 0.5042
 5696/25000 [=====>........................] - ETA: 46s - loss: 7.6047 - accuracy: 0.5040
 5728/25000 [=====>........................] - ETA: 46s - loss: 7.6077 - accuracy: 0.5038
 5760/25000 [=====>........................] - ETA: 46s - loss: 7.6107 - accuracy: 0.5036
 5792/25000 [=====>........................] - ETA: 46s - loss: 7.6004 - accuracy: 0.5043
 5824/25000 [=====>........................] - ETA: 46s - loss: 7.6087 - accuracy: 0.5038
 5856/25000 [======>.......................] - ETA: 46s - loss: 7.6143 - accuracy: 0.5034
 5888/25000 [======>.......................] - ETA: 46s - loss: 7.6093 - accuracy: 0.5037
 5920/25000 [======>.......................] - ETA: 46s - loss: 7.6045 - accuracy: 0.5041
 5952/25000 [======>.......................] - ETA: 46s - loss: 7.5996 - accuracy: 0.5044
 5984/25000 [======>.......................] - ETA: 46s - loss: 7.6051 - accuracy: 0.5040
 6016/25000 [======>.......................] - ETA: 46s - loss: 7.6029 - accuracy: 0.5042
 6048/25000 [======>.......................] - ETA: 45s - loss: 7.6007 - accuracy: 0.5043
 6080/25000 [======>.......................] - ETA: 45s - loss: 7.6010 - accuracy: 0.5043
 6112/25000 [======>.......................] - ETA: 45s - loss: 7.6114 - accuracy: 0.5036
 6144/25000 [======>.......................] - ETA: 45s - loss: 7.6092 - accuracy: 0.5037
 6176/25000 [======>.......................] - ETA: 45s - loss: 7.6145 - accuracy: 0.5034
 6208/25000 [======>.......................] - ETA: 45s - loss: 7.6271 - accuracy: 0.5026
 6240/25000 [======>.......................] - ETA: 45s - loss: 7.6322 - accuracy: 0.5022
 6272/25000 [======>.......................] - ETA: 45s - loss: 7.6324 - accuracy: 0.5022
 6304/25000 [======>.......................] - ETA: 45s - loss: 7.6277 - accuracy: 0.5025
 6336/25000 [======>.......................] - ETA: 45s - loss: 7.6255 - accuracy: 0.5027
 6368/25000 [======>.......................] - ETA: 45s - loss: 7.6305 - accuracy: 0.5024
 6400/25000 [======>.......................] - ETA: 45s - loss: 7.6379 - accuracy: 0.5019
 6432/25000 [======>.......................] - ETA: 45s - loss: 7.6285 - accuracy: 0.5025
 6464/25000 [======>.......................] - ETA: 44s - loss: 7.6263 - accuracy: 0.5026
 6496/25000 [======>.......................] - ETA: 44s - loss: 7.6359 - accuracy: 0.5020
 6528/25000 [======>.......................] - ETA: 44s - loss: 7.6384 - accuracy: 0.5018
 6560/25000 [======>.......................] - ETA: 44s - loss: 7.6503 - accuracy: 0.5011
 6592/25000 [======>.......................] - ETA: 44s - loss: 7.6480 - accuracy: 0.5012
 6624/25000 [======>.......................] - ETA: 44s - loss: 7.6412 - accuracy: 0.5017
 6656/25000 [======>.......................] - ETA: 44s - loss: 7.6482 - accuracy: 0.5012
 6688/25000 [=======>......................] - ETA: 44s - loss: 7.6483 - accuracy: 0.5012
 6720/25000 [=======>......................] - ETA: 44s - loss: 7.6461 - accuracy: 0.5013
 6752/25000 [=======>......................] - ETA: 44s - loss: 7.6485 - accuracy: 0.5012
 6784/25000 [=======>......................] - ETA: 44s - loss: 7.6508 - accuracy: 0.5010
 6816/25000 [=======>......................] - ETA: 44s - loss: 7.6554 - accuracy: 0.5007
 6848/25000 [=======>......................] - ETA: 44s - loss: 7.6398 - accuracy: 0.5018
 6880/25000 [=======>......................] - ETA: 44s - loss: 7.6376 - accuracy: 0.5019
 6912/25000 [=======>......................] - ETA: 43s - loss: 7.6422 - accuracy: 0.5016
 6944/25000 [=======>......................] - ETA: 43s - loss: 7.6534 - accuracy: 0.5009
 6976/25000 [=======>......................] - ETA: 43s - loss: 7.6600 - accuracy: 0.5004
 7008/25000 [=======>......................] - ETA: 43s - loss: 7.6535 - accuracy: 0.5009
 7040/25000 [=======>......................] - ETA: 43s - loss: 7.6383 - accuracy: 0.5018
 7072/25000 [=======>......................] - ETA: 43s - loss: 7.6384 - accuracy: 0.5018
 7104/25000 [=======>......................] - ETA: 43s - loss: 7.6515 - accuracy: 0.5010
 7136/25000 [=======>......................] - ETA: 43s - loss: 7.6559 - accuracy: 0.5007
 7168/25000 [=======>......................] - ETA: 43s - loss: 7.6452 - accuracy: 0.5014
 7200/25000 [=======>......................] - ETA: 43s - loss: 7.6453 - accuracy: 0.5014
 7232/25000 [=======>......................] - ETA: 43s - loss: 7.6433 - accuracy: 0.5015
 7264/25000 [=======>......................] - ETA: 43s - loss: 7.6455 - accuracy: 0.5014
 7296/25000 [=======>......................] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
 7328/25000 [=======>......................] - ETA: 42s - loss: 7.6562 - accuracy: 0.5007
 7360/25000 [=======>......................] - ETA: 42s - loss: 7.6541 - accuracy: 0.5008
 7392/25000 [=======>......................] - ETA: 42s - loss: 7.6562 - accuracy: 0.5007
 7424/25000 [=======>......................] - ETA: 42s - loss: 7.6480 - accuracy: 0.5012
 7456/25000 [=======>......................] - ETA: 42s - loss: 7.6502 - accuracy: 0.5011
 7488/25000 [=======>......................] - ETA: 42s - loss: 7.6441 - accuracy: 0.5015
 7520/25000 [========>.....................] - ETA: 42s - loss: 7.6462 - accuracy: 0.5013
 7552/25000 [========>.....................] - ETA: 42s - loss: 7.6402 - accuracy: 0.5017
 7584/25000 [========>.....................] - ETA: 42s - loss: 7.6444 - accuracy: 0.5015
 7616/25000 [========>.....................] - ETA: 42s - loss: 7.6364 - accuracy: 0.5020
 7648/25000 [========>.....................] - ETA: 42s - loss: 7.6386 - accuracy: 0.5018
 7680/25000 [========>.....................] - ETA: 42s - loss: 7.6387 - accuracy: 0.5018
 7712/25000 [========>.....................] - ETA: 42s - loss: 7.6368 - accuracy: 0.5019
 7744/25000 [========>.....................] - ETA: 41s - loss: 7.6389 - accuracy: 0.5018
 7776/25000 [========>.....................] - ETA: 41s - loss: 7.6351 - accuracy: 0.5021
 7808/25000 [========>.....................] - ETA: 41s - loss: 7.6411 - accuracy: 0.5017
 7840/25000 [========>.....................] - ETA: 41s - loss: 7.6451 - accuracy: 0.5014
 7872/25000 [========>.....................] - ETA: 41s - loss: 7.6530 - accuracy: 0.5009
 7904/25000 [========>.....................] - ETA: 41s - loss: 7.6589 - accuracy: 0.5005
 7936/25000 [========>.....................] - ETA: 41s - loss: 7.6628 - accuracy: 0.5003
 7968/25000 [========>.....................] - ETA: 41s - loss: 7.6647 - accuracy: 0.5001
 8000/25000 [========>.....................] - ETA: 41s - loss: 7.6590 - accuracy: 0.5005
 8032/25000 [========>.....................] - ETA: 41s - loss: 7.6704 - accuracy: 0.4998
 8064/25000 [========>.....................] - ETA: 41s - loss: 7.6742 - accuracy: 0.4995
 8096/25000 [========>.....................] - ETA: 41s - loss: 7.6723 - accuracy: 0.4996
 8128/25000 [========>.....................] - ETA: 41s - loss: 7.6761 - accuracy: 0.4994
 8160/25000 [========>.....................] - ETA: 40s - loss: 7.6760 - accuracy: 0.4994
 8192/25000 [========>.....................] - ETA: 40s - loss: 7.6722 - accuracy: 0.4996
 8224/25000 [========>.....................] - ETA: 40s - loss: 7.6666 - accuracy: 0.5000
 8256/25000 [========>.....................] - ETA: 40s - loss: 7.6685 - accuracy: 0.4999
 8288/25000 [========>.....................] - ETA: 40s - loss: 7.6685 - accuracy: 0.4999
 8320/25000 [========>.....................] - ETA: 40s - loss: 7.6666 - accuracy: 0.5000
 8352/25000 [=========>....................] - ETA: 40s - loss: 7.6721 - accuracy: 0.4996
 8384/25000 [=========>....................] - ETA: 40s - loss: 7.6703 - accuracy: 0.4998
 8416/25000 [=========>....................] - ETA: 40s - loss: 7.6684 - accuracy: 0.4999
 8448/25000 [=========>....................] - ETA: 40s - loss: 7.6702 - accuracy: 0.4998
 8480/25000 [=========>....................] - ETA: 40s - loss: 7.6775 - accuracy: 0.4993
 8512/25000 [=========>....................] - ETA: 40s - loss: 7.6774 - accuracy: 0.4993
 8544/25000 [=========>....................] - ETA: 40s - loss: 7.6684 - accuracy: 0.4999
 8576/25000 [=========>....................] - ETA: 40s - loss: 7.6613 - accuracy: 0.5003
 8608/25000 [=========>....................] - ETA: 39s - loss: 7.6702 - accuracy: 0.4998
 8640/25000 [=========>....................] - ETA: 39s - loss: 7.6719 - accuracy: 0.4997
 8672/25000 [=========>....................] - ETA: 39s - loss: 7.6790 - accuracy: 0.4992
 8704/25000 [=========>....................] - ETA: 39s - loss: 7.6772 - accuracy: 0.4993
 8736/25000 [=========>....................] - ETA: 39s - loss: 7.6719 - accuracy: 0.4997
 8768/25000 [=========>....................] - ETA: 39s - loss: 7.6736 - accuracy: 0.4995
 8800/25000 [=========>....................] - ETA: 39s - loss: 7.6649 - accuracy: 0.5001
 8832/25000 [=========>....................] - ETA: 39s - loss: 7.6649 - accuracy: 0.5001
 8864/25000 [=========>....................] - ETA: 39s - loss: 7.6649 - accuracy: 0.5001
 8896/25000 [=========>....................] - ETA: 39s - loss: 7.6632 - accuracy: 0.5002
 8928/25000 [=========>....................] - ETA: 39s - loss: 7.6666 - accuracy: 0.5000
 8960/25000 [=========>....................] - ETA: 39s - loss: 7.6683 - accuracy: 0.4999
 8992/25000 [=========>....................] - ETA: 39s - loss: 7.6717 - accuracy: 0.4997
 9024/25000 [=========>....................] - ETA: 38s - loss: 7.6632 - accuracy: 0.5002
 9056/25000 [=========>....................] - ETA: 38s - loss: 7.6582 - accuracy: 0.5006
 9088/25000 [=========>....................] - ETA: 38s - loss: 7.6582 - accuracy: 0.5006
 9120/25000 [=========>....................] - ETA: 38s - loss: 7.6565 - accuracy: 0.5007
 9152/25000 [=========>....................] - ETA: 38s - loss: 7.6582 - accuracy: 0.5005
 9184/25000 [==========>...................] - ETA: 38s - loss: 7.6599 - accuracy: 0.5004
 9216/25000 [==========>...................] - ETA: 38s - loss: 7.6600 - accuracy: 0.5004
 9248/25000 [==========>...................] - ETA: 38s - loss: 7.6500 - accuracy: 0.5011
 9280/25000 [==========>...................] - ETA: 38s - loss: 7.6501 - accuracy: 0.5011
 9312/25000 [==========>...................] - ETA: 38s - loss: 7.6452 - accuracy: 0.5014
 9344/25000 [==========>...................] - ETA: 38s - loss: 7.6469 - accuracy: 0.5013
 9376/25000 [==========>...................] - ETA: 38s - loss: 7.6503 - accuracy: 0.5011
 9408/25000 [==========>...................] - ETA: 38s - loss: 7.6536 - accuracy: 0.5009
 9440/25000 [==========>...................] - ETA: 37s - loss: 7.6520 - accuracy: 0.5010
 9472/25000 [==========>...................] - ETA: 37s - loss: 7.6521 - accuracy: 0.5010
 9504/25000 [==========>...................] - ETA: 37s - loss: 7.6569 - accuracy: 0.5006
 9536/25000 [==========>...................] - ETA: 37s - loss: 7.6521 - accuracy: 0.5009
 9568/25000 [==========>...................] - ETA: 37s - loss: 7.6490 - accuracy: 0.5011
 9600/25000 [==========>...................] - ETA: 37s - loss: 7.6570 - accuracy: 0.5006
 9632/25000 [==========>...................] - ETA: 37s - loss: 7.6539 - accuracy: 0.5008
 9664/25000 [==========>...................] - ETA: 37s - loss: 7.6555 - accuracy: 0.5007
 9696/25000 [==========>...................] - ETA: 37s - loss: 7.6555 - accuracy: 0.5007
 9728/25000 [==========>...................] - ETA: 37s - loss: 7.6587 - accuracy: 0.5005
 9760/25000 [==========>...................] - ETA: 37s - loss: 7.6556 - accuracy: 0.5007
 9792/25000 [==========>...................] - ETA: 37s - loss: 7.6572 - accuracy: 0.5006
 9824/25000 [==========>...................] - ETA: 37s - loss: 7.6573 - accuracy: 0.5006
 9856/25000 [==========>...................] - ETA: 36s - loss: 7.6557 - accuracy: 0.5007
 9888/25000 [==========>...................] - ETA: 36s - loss: 7.6542 - accuracy: 0.5008
 9920/25000 [==========>...................] - ETA: 36s - loss: 7.6620 - accuracy: 0.5003
 9952/25000 [==========>...................] - ETA: 36s - loss: 7.6697 - accuracy: 0.4998
 9984/25000 [==========>...................] - ETA: 36s - loss: 7.6620 - accuracy: 0.5003
10016/25000 [===========>..................] - ETA: 36s - loss: 7.6620 - accuracy: 0.5003
10048/25000 [===========>..................] - ETA: 36s - loss: 7.6636 - accuracy: 0.5002
10080/25000 [===========>..................] - ETA: 36s - loss: 7.6621 - accuracy: 0.5003
10112/25000 [===========>..................] - ETA: 36s - loss: 7.6666 - accuracy: 0.5000
10144/25000 [===========>..................] - ETA: 36s - loss: 7.6696 - accuracy: 0.4998
10176/25000 [===========>..................] - ETA: 36s - loss: 7.6696 - accuracy: 0.4998
10208/25000 [===========>..................] - ETA: 36s - loss: 7.6666 - accuracy: 0.5000
10240/25000 [===========>..................] - ETA: 36s - loss: 7.6651 - accuracy: 0.5001
10272/25000 [===========>..................] - ETA: 35s - loss: 7.6681 - accuracy: 0.4999
10304/25000 [===========>..................] - ETA: 35s - loss: 7.6726 - accuracy: 0.4996
10336/25000 [===========>..................] - ETA: 35s - loss: 7.6711 - accuracy: 0.4997
10368/25000 [===========>..................] - ETA: 35s - loss: 7.6711 - accuracy: 0.4997
10400/25000 [===========>..................] - ETA: 35s - loss: 7.6725 - accuracy: 0.4996
10432/25000 [===========>..................] - ETA: 35s - loss: 7.6681 - accuracy: 0.4999
10464/25000 [===========>..................] - ETA: 35s - loss: 7.6681 - accuracy: 0.4999
10496/25000 [===========>..................] - ETA: 35s - loss: 7.6666 - accuracy: 0.5000
10528/25000 [===========>..................] - ETA: 35s - loss: 7.6695 - accuracy: 0.4998
10560/25000 [===========>..................] - ETA: 35s - loss: 7.6681 - accuracy: 0.4999
10592/25000 [===========>..................] - ETA: 35s - loss: 7.6724 - accuracy: 0.4996
10624/25000 [===========>..................] - ETA: 35s - loss: 7.6709 - accuracy: 0.4997
10656/25000 [===========>..................] - ETA: 35s - loss: 7.6767 - accuracy: 0.4993
10688/25000 [===========>..................] - ETA: 34s - loss: 7.6738 - accuracy: 0.4995
10720/25000 [===========>..................] - ETA: 34s - loss: 7.6723 - accuracy: 0.4996
10752/25000 [===========>..................] - ETA: 34s - loss: 7.6666 - accuracy: 0.5000
10784/25000 [===========>..................] - ETA: 34s - loss: 7.6652 - accuracy: 0.5001
10816/25000 [===========>..................] - ETA: 34s - loss: 7.6624 - accuracy: 0.5003
10848/25000 [============>.................] - ETA: 34s - loss: 7.6567 - accuracy: 0.5006
10880/25000 [============>.................] - ETA: 34s - loss: 7.6582 - accuracy: 0.5006
10912/25000 [============>.................] - ETA: 34s - loss: 7.6554 - accuracy: 0.5007
10944/25000 [============>.................] - ETA: 34s - loss: 7.6526 - accuracy: 0.5009
10976/25000 [============>.................] - ETA: 34s - loss: 7.6485 - accuracy: 0.5012
11008/25000 [============>.................] - ETA: 34s - loss: 7.6471 - accuracy: 0.5013
11040/25000 [============>.................] - ETA: 34s - loss: 7.6500 - accuracy: 0.5011
11072/25000 [============>.................] - ETA: 33s - loss: 7.6458 - accuracy: 0.5014
11104/25000 [============>.................] - ETA: 33s - loss: 7.6445 - accuracy: 0.5014
11136/25000 [============>.................] - ETA: 33s - loss: 7.6418 - accuracy: 0.5016
11168/25000 [============>.................] - ETA: 33s - loss: 7.6419 - accuracy: 0.5016
11200/25000 [============>.................] - ETA: 33s - loss: 7.6447 - accuracy: 0.5014
11232/25000 [============>.................] - ETA: 33s - loss: 7.6475 - accuracy: 0.5012
11264/25000 [============>.................] - ETA: 33s - loss: 7.6530 - accuracy: 0.5009
11296/25000 [============>.................] - ETA: 33s - loss: 7.6558 - accuracy: 0.5007
11328/25000 [============>.................] - ETA: 33s - loss: 7.6612 - accuracy: 0.5004
11360/25000 [============>.................] - ETA: 33s - loss: 7.6626 - accuracy: 0.5003
11392/25000 [============>.................] - ETA: 33s - loss: 7.6653 - accuracy: 0.5001
11424/25000 [============>.................] - ETA: 33s - loss: 7.6626 - accuracy: 0.5003
11456/25000 [============>.................] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
11488/25000 [============>.................] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
11520/25000 [============>.................] - ETA: 32s - loss: 7.6653 - accuracy: 0.5001
11552/25000 [============>.................] - ETA: 32s - loss: 7.6560 - accuracy: 0.5007
11584/25000 [============>.................] - ETA: 32s - loss: 7.6640 - accuracy: 0.5002
11616/25000 [============>.................] - ETA: 32s - loss: 7.6679 - accuracy: 0.4999
11648/25000 [============>.................] - ETA: 32s - loss: 7.6693 - accuracy: 0.4998
11680/25000 [=============>................] - ETA: 32s - loss: 7.6640 - accuracy: 0.5002
11712/25000 [=============>................] - ETA: 32s - loss: 7.6653 - accuracy: 0.5001
11744/25000 [=============>................] - ETA: 32s - loss: 7.6640 - accuracy: 0.5002
11776/25000 [=============>................] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
11808/25000 [=============>................] - ETA: 32s - loss: 7.6679 - accuracy: 0.4999
11840/25000 [=============>................] - ETA: 32s - loss: 7.6679 - accuracy: 0.4999
11872/25000 [=============>................] - ETA: 32s - loss: 7.6757 - accuracy: 0.4994
11904/25000 [=============>................] - ETA: 31s - loss: 7.6756 - accuracy: 0.4994
11936/25000 [=============>................] - ETA: 31s - loss: 7.6730 - accuracy: 0.4996
11968/25000 [=============>................] - ETA: 31s - loss: 7.6717 - accuracy: 0.4997
12000/25000 [=============>................] - ETA: 31s - loss: 7.6679 - accuracy: 0.4999
12032/25000 [=============>................] - ETA: 31s - loss: 7.6717 - accuracy: 0.4997
12064/25000 [=============>................] - ETA: 31s - loss: 7.6742 - accuracy: 0.4995
12096/25000 [=============>................] - ETA: 31s - loss: 7.6768 - accuracy: 0.4993
12128/25000 [=============>................] - ETA: 31s - loss: 7.6831 - accuracy: 0.4989
12160/25000 [=============>................] - ETA: 31s - loss: 7.6792 - accuracy: 0.4992
12192/25000 [=============>................] - ETA: 31s - loss: 7.6805 - accuracy: 0.4991
12224/25000 [=============>................] - ETA: 31s - loss: 7.6842 - accuracy: 0.4989
12256/25000 [=============>................] - ETA: 31s - loss: 7.6904 - accuracy: 0.4984
12288/25000 [=============>................] - ETA: 31s - loss: 7.6878 - accuracy: 0.4986
12320/25000 [=============>................] - ETA: 30s - loss: 7.6965 - accuracy: 0.4981
12352/25000 [=============>................] - ETA: 30s - loss: 7.7001 - accuracy: 0.4978
12384/25000 [=============>................] - ETA: 30s - loss: 7.7013 - accuracy: 0.4977
12416/25000 [=============>................] - ETA: 30s - loss: 7.7000 - accuracy: 0.4978
12448/25000 [=============>................] - ETA: 30s - loss: 7.6986 - accuracy: 0.4979
12480/25000 [=============>................] - ETA: 30s - loss: 7.6998 - accuracy: 0.4978
12512/25000 [==============>...............] - ETA: 30s - loss: 7.6960 - accuracy: 0.4981
12544/25000 [==============>...............] - ETA: 30s - loss: 7.6935 - accuracy: 0.4982
12576/25000 [==============>...............] - ETA: 30s - loss: 7.6959 - accuracy: 0.4981
12608/25000 [==============>...............] - ETA: 30s - loss: 7.6946 - accuracy: 0.4982
12640/25000 [==============>...............] - ETA: 30s - loss: 7.6945 - accuracy: 0.4982
12672/25000 [==============>...............] - ETA: 30s - loss: 7.6908 - accuracy: 0.4984
12704/25000 [==============>...............] - ETA: 30s - loss: 7.6920 - accuracy: 0.4983
12736/25000 [==============>...............] - ETA: 29s - loss: 7.6895 - accuracy: 0.4985
12768/25000 [==============>...............] - ETA: 29s - loss: 7.6882 - accuracy: 0.4986
12800/25000 [==============>...............] - ETA: 29s - loss: 7.6906 - accuracy: 0.4984
12832/25000 [==============>...............] - ETA: 29s - loss: 7.6893 - accuracy: 0.4985
12864/25000 [==============>...............] - ETA: 29s - loss: 7.6905 - accuracy: 0.4984
12896/25000 [==============>...............] - ETA: 29s - loss: 7.6892 - accuracy: 0.4985
12928/25000 [==============>...............] - ETA: 29s - loss: 7.6903 - accuracy: 0.4985
12960/25000 [==============>...............] - ETA: 29s - loss: 7.6926 - accuracy: 0.4983
12992/25000 [==============>...............] - ETA: 29s - loss: 7.6914 - accuracy: 0.4984
13024/25000 [==============>...............] - ETA: 29s - loss: 7.6902 - accuracy: 0.4985
13056/25000 [==============>...............] - ETA: 29s - loss: 7.6925 - accuracy: 0.4983
13088/25000 [==============>...............] - ETA: 29s - loss: 7.6983 - accuracy: 0.4979
13120/25000 [==============>...............] - ETA: 29s - loss: 7.7040 - accuracy: 0.4976
13152/25000 [==============>...............] - ETA: 28s - loss: 7.7004 - accuracy: 0.4978
13184/25000 [==============>...............] - ETA: 28s - loss: 7.7015 - accuracy: 0.4977
13216/25000 [==============>...............] - ETA: 28s - loss: 7.7003 - accuracy: 0.4978
13248/25000 [==============>...............] - ETA: 28s - loss: 7.6967 - accuracy: 0.4980
13280/25000 [==============>...............] - ETA: 28s - loss: 7.6955 - accuracy: 0.4981
13312/25000 [==============>...............] - ETA: 28s - loss: 7.6931 - accuracy: 0.4983
13344/25000 [===============>..............] - ETA: 28s - loss: 7.6919 - accuracy: 0.4984
13376/25000 [===============>..............] - ETA: 28s - loss: 7.6918 - accuracy: 0.4984
13408/25000 [===============>..............] - ETA: 28s - loss: 7.6838 - accuracy: 0.4989
13440/25000 [===============>..............] - ETA: 28s - loss: 7.6769 - accuracy: 0.4993
13472/25000 [===============>..............] - ETA: 28s - loss: 7.6848 - accuracy: 0.4988
13504/25000 [===============>..............] - ETA: 28s - loss: 7.6882 - accuracy: 0.4986
13536/25000 [===============>..............] - ETA: 28s - loss: 7.6836 - accuracy: 0.4989
13568/25000 [===============>..............] - ETA: 27s - loss: 7.6870 - accuracy: 0.4987
13600/25000 [===============>..............] - ETA: 27s - loss: 7.6880 - accuracy: 0.4986
13632/25000 [===============>..............] - ETA: 27s - loss: 7.6869 - accuracy: 0.4987
13664/25000 [===============>..............] - ETA: 27s - loss: 7.6879 - accuracy: 0.4986
13696/25000 [===============>..............] - ETA: 27s - loss: 7.6857 - accuracy: 0.4988
13728/25000 [===============>..............] - ETA: 27s - loss: 7.6867 - accuracy: 0.4987
13760/25000 [===============>..............] - ETA: 27s - loss: 7.6878 - accuracy: 0.4986
13792/25000 [===============>..............] - ETA: 27s - loss: 7.6855 - accuracy: 0.4988
13824/25000 [===============>..............] - ETA: 27s - loss: 7.6866 - accuracy: 0.4987
13856/25000 [===============>..............] - ETA: 27s - loss: 7.6821 - accuracy: 0.4990
13888/25000 [===============>..............] - ETA: 27s - loss: 7.6788 - accuracy: 0.4992
13920/25000 [===============>..............] - ETA: 27s - loss: 7.6798 - accuracy: 0.4991
13952/25000 [===============>..............] - ETA: 27s - loss: 7.6820 - accuracy: 0.4990
13984/25000 [===============>..............] - ETA: 26s - loss: 7.6820 - accuracy: 0.4990
14016/25000 [===============>..............] - ETA: 26s - loss: 7.6797 - accuracy: 0.4991
14048/25000 [===============>..............] - ETA: 26s - loss: 7.6786 - accuracy: 0.4992
14080/25000 [===============>..............] - ETA: 26s - loss: 7.6830 - accuracy: 0.4989
14112/25000 [===============>..............] - ETA: 26s - loss: 7.6797 - accuracy: 0.4991
14144/25000 [===============>..............] - ETA: 26s - loss: 7.6796 - accuracy: 0.4992
14176/25000 [================>.............] - ETA: 26s - loss: 7.6785 - accuracy: 0.4992
14208/25000 [================>.............] - ETA: 26s - loss: 7.6763 - accuracy: 0.4994
14240/25000 [================>.............] - ETA: 26s - loss: 7.6752 - accuracy: 0.4994
14272/25000 [================>.............] - ETA: 26s - loss: 7.6720 - accuracy: 0.4996
14304/25000 [================>.............] - ETA: 26s - loss: 7.6741 - accuracy: 0.4995
14336/25000 [================>.............] - ETA: 26s - loss: 7.6730 - accuracy: 0.4996
14368/25000 [================>.............] - ETA: 25s - loss: 7.6730 - accuracy: 0.4996
14400/25000 [================>.............] - ETA: 25s - loss: 7.6730 - accuracy: 0.4996
14432/25000 [================>.............] - ETA: 25s - loss: 7.6741 - accuracy: 0.4995
14464/25000 [================>.............] - ETA: 25s - loss: 7.6740 - accuracy: 0.4995
14496/25000 [================>.............] - ETA: 25s - loss: 7.6761 - accuracy: 0.4994
14528/25000 [================>.............] - ETA: 25s - loss: 7.6751 - accuracy: 0.4994
14560/25000 [================>.............] - ETA: 25s - loss: 7.6729 - accuracy: 0.4996
14592/25000 [================>.............] - ETA: 25s - loss: 7.6719 - accuracy: 0.4997
14624/25000 [================>.............] - ETA: 25s - loss: 7.6750 - accuracy: 0.4995
14656/25000 [================>.............] - ETA: 25s - loss: 7.6739 - accuracy: 0.4995
14688/25000 [================>.............] - ETA: 25s - loss: 7.6760 - accuracy: 0.4994
14720/25000 [================>.............] - ETA: 25s - loss: 7.6739 - accuracy: 0.4995
14752/25000 [================>.............] - ETA: 25s - loss: 7.6770 - accuracy: 0.4993
14784/25000 [================>.............] - ETA: 24s - loss: 7.6770 - accuracy: 0.4993
14816/25000 [================>.............] - ETA: 24s - loss: 7.6759 - accuracy: 0.4994
14848/25000 [================>.............] - ETA: 24s - loss: 7.6790 - accuracy: 0.4992
14880/25000 [================>.............] - ETA: 24s - loss: 7.6728 - accuracy: 0.4996
14912/25000 [================>.............] - ETA: 24s - loss: 7.6779 - accuracy: 0.4993
14944/25000 [================>.............] - ETA: 24s - loss: 7.6769 - accuracy: 0.4993
14976/25000 [================>.............] - ETA: 24s - loss: 7.6758 - accuracy: 0.4994
15008/25000 [=================>............] - ETA: 24s - loss: 7.6809 - accuracy: 0.4991
15040/25000 [=================>............] - ETA: 24s - loss: 7.6809 - accuracy: 0.4991
15072/25000 [=================>............] - ETA: 24s - loss: 7.6809 - accuracy: 0.4991
15104/25000 [=================>............] - ETA: 24s - loss: 7.6768 - accuracy: 0.4993
15136/25000 [=================>............] - ETA: 24s - loss: 7.6778 - accuracy: 0.4993
15168/25000 [=================>............] - ETA: 24s - loss: 7.6757 - accuracy: 0.4994
15200/25000 [=================>............] - ETA: 24s - loss: 7.6757 - accuracy: 0.4994
15232/25000 [=================>............] - ETA: 23s - loss: 7.6757 - accuracy: 0.4994
15264/25000 [=================>............] - ETA: 23s - loss: 7.6787 - accuracy: 0.4992
15296/25000 [=================>............] - ETA: 23s - loss: 7.6807 - accuracy: 0.4991
15328/25000 [=================>............] - ETA: 23s - loss: 7.6786 - accuracy: 0.4992
15360/25000 [=================>............] - ETA: 23s - loss: 7.6806 - accuracy: 0.4991
15392/25000 [=================>............] - ETA: 23s - loss: 7.6826 - accuracy: 0.4990
15424/25000 [=================>............] - ETA: 23s - loss: 7.6815 - accuracy: 0.4990
15456/25000 [=================>............] - ETA: 23s - loss: 7.6845 - accuracy: 0.4988
15488/25000 [=================>............] - ETA: 23s - loss: 7.6884 - accuracy: 0.4986
15520/25000 [=================>............] - ETA: 23s - loss: 7.6844 - accuracy: 0.4988
15552/25000 [=================>............] - ETA: 23s - loss: 7.6844 - accuracy: 0.4988
15584/25000 [=================>............] - ETA: 23s - loss: 7.6853 - accuracy: 0.4988
15616/25000 [=================>............] - ETA: 23s - loss: 7.6833 - accuracy: 0.4989
15648/25000 [=================>............] - ETA: 22s - loss: 7.6862 - accuracy: 0.4987
15680/25000 [=================>............] - ETA: 22s - loss: 7.6862 - accuracy: 0.4987
15712/25000 [=================>............] - ETA: 22s - loss: 7.6881 - accuracy: 0.4986
15744/25000 [=================>............] - ETA: 22s - loss: 7.6890 - accuracy: 0.4985
15776/25000 [=================>............] - ETA: 22s - loss: 7.6919 - accuracy: 0.4984
15808/25000 [=================>............] - ETA: 22s - loss: 7.6928 - accuracy: 0.4983
15840/25000 [==================>...........] - ETA: 22s - loss: 7.6937 - accuracy: 0.4982
15872/25000 [==================>...........] - ETA: 22s - loss: 7.6927 - accuracy: 0.4983
15904/25000 [==================>...........] - ETA: 22s - loss: 7.6907 - accuracy: 0.4984
15936/25000 [==================>...........] - ETA: 22s - loss: 7.6916 - accuracy: 0.4984
15968/25000 [==================>...........] - ETA: 22s - loss: 7.6906 - accuracy: 0.4984
16000/25000 [==================>...........] - ETA: 22s - loss: 7.6906 - accuracy: 0.4984
16032/25000 [==================>...........] - ETA: 21s - loss: 7.6886 - accuracy: 0.4986
16064/25000 [==================>...........] - ETA: 21s - loss: 7.6895 - accuracy: 0.4985
16096/25000 [==================>...........] - ETA: 21s - loss: 7.6923 - accuracy: 0.4983
16128/25000 [==================>...........] - ETA: 21s - loss: 7.6894 - accuracy: 0.4985
16160/25000 [==================>...........] - ETA: 21s - loss: 7.6894 - accuracy: 0.4985
16192/25000 [==================>...........] - ETA: 21s - loss: 7.6893 - accuracy: 0.4985
16224/25000 [==================>...........] - ETA: 21s - loss: 7.6893 - accuracy: 0.4985
16256/25000 [==================>...........] - ETA: 21s - loss: 7.6911 - accuracy: 0.4984
16288/25000 [==================>...........] - ETA: 21s - loss: 7.6930 - accuracy: 0.4983
16320/25000 [==================>...........] - ETA: 21s - loss: 7.6929 - accuracy: 0.4983
16352/25000 [==================>...........] - ETA: 21s - loss: 7.6938 - accuracy: 0.4982
16384/25000 [==================>...........] - ETA: 21s - loss: 7.6919 - accuracy: 0.4984
16416/25000 [==================>...........] - ETA: 21s - loss: 7.6881 - accuracy: 0.4986
16448/25000 [==================>...........] - ETA: 20s - loss: 7.6899 - accuracy: 0.4985
16480/25000 [==================>...........] - ETA: 20s - loss: 7.6917 - accuracy: 0.4984
16512/25000 [==================>...........] - ETA: 20s - loss: 7.6945 - accuracy: 0.4982
16544/25000 [==================>...........] - ETA: 20s - loss: 7.6981 - accuracy: 0.4979
16576/25000 [==================>...........] - ETA: 20s - loss: 7.6999 - accuracy: 0.4978
16608/25000 [==================>...........] - ETA: 20s - loss: 7.6999 - accuracy: 0.4978
16640/25000 [==================>...........] - ETA: 20s - loss: 7.7007 - accuracy: 0.4978
16672/25000 [===================>..........] - ETA: 20s - loss: 7.7006 - accuracy: 0.4978
16704/25000 [===================>..........] - ETA: 20s - loss: 7.6987 - accuracy: 0.4979
16736/25000 [===================>..........] - ETA: 20s - loss: 7.7014 - accuracy: 0.4977
16768/25000 [===================>..........] - ETA: 20s - loss: 7.6995 - accuracy: 0.4979
16800/25000 [===================>..........] - ETA: 20s - loss: 7.6986 - accuracy: 0.4979
16832/25000 [===================>..........] - ETA: 20s - loss: 7.6967 - accuracy: 0.4980
16864/25000 [===================>..........] - ETA: 19s - loss: 7.6948 - accuracy: 0.4982
16896/25000 [===================>..........] - ETA: 19s - loss: 7.6920 - accuracy: 0.4983
16928/25000 [===================>..........] - ETA: 19s - loss: 7.6911 - accuracy: 0.4984
16960/25000 [===================>..........] - ETA: 19s - loss: 7.6956 - accuracy: 0.4981
16992/25000 [===================>..........] - ETA: 19s - loss: 7.6955 - accuracy: 0.4981
17024/25000 [===================>..........] - ETA: 19s - loss: 7.6936 - accuracy: 0.4982
17056/25000 [===================>..........] - ETA: 19s - loss: 7.6909 - accuracy: 0.4984
17088/25000 [===================>..........] - ETA: 19s - loss: 7.6935 - accuracy: 0.4982
17120/25000 [===================>..........] - ETA: 19s - loss: 7.6953 - accuracy: 0.4981
17152/25000 [===================>..........] - ETA: 19s - loss: 7.6925 - accuracy: 0.4983
17184/25000 [===================>..........] - ETA: 19s - loss: 7.6898 - accuracy: 0.4985
17216/25000 [===================>..........] - ETA: 19s - loss: 7.6916 - accuracy: 0.4984
17248/25000 [===================>..........] - ETA: 19s - loss: 7.6924 - accuracy: 0.4983
17280/25000 [===================>..........] - ETA: 18s - loss: 7.6897 - accuracy: 0.4985
17312/25000 [===================>..........] - ETA: 18s - loss: 7.6870 - accuracy: 0.4987
17344/25000 [===================>..........] - ETA: 18s - loss: 7.6870 - accuracy: 0.4987
17376/25000 [===================>..........] - ETA: 18s - loss: 7.6860 - accuracy: 0.4987
17408/25000 [===================>..........] - ETA: 18s - loss: 7.6886 - accuracy: 0.4986
17440/25000 [===================>..........] - ETA: 18s - loss: 7.6895 - accuracy: 0.4985
17472/25000 [===================>..........] - ETA: 18s - loss: 7.6903 - accuracy: 0.4985
17504/25000 [====================>.........] - ETA: 18s - loss: 7.6911 - accuracy: 0.4984
17536/25000 [====================>.........] - ETA: 18s - loss: 7.6876 - accuracy: 0.4986
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6919 - accuracy: 0.4983
17600/25000 [====================>.........] - ETA: 18s - loss: 7.6893 - accuracy: 0.4985
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6858 - accuracy: 0.4988
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6857 - accuracy: 0.4988
17696/25000 [====================>.........] - ETA: 17s - loss: 7.6891 - accuracy: 0.4985
17728/25000 [====================>.........] - ETA: 17s - loss: 7.6891 - accuracy: 0.4985
17760/25000 [====================>.........] - ETA: 17s - loss: 7.6908 - accuracy: 0.4984
17792/25000 [====================>.........] - ETA: 17s - loss: 7.6933 - accuracy: 0.4983
17824/25000 [====================>.........] - ETA: 17s - loss: 7.6907 - accuracy: 0.4984
17856/25000 [====================>.........] - ETA: 17s - loss: 7.6907 - accuracy: 0.4984
17888/25000 [====================>.........] - ETA: 17s - loss: 7.6872 - accuracy: 0.4987
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6863 - accuracy: 0.4987
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6897 - accuracy: 0.4985
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6888 - accuracy: 0.4986
18016/25000 [====================>.........] - ETA: 17s - loss: 7.6904 - accuracy: 0.4984
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6904 - accuracy: 0.4984
18080/25000 [====================>.........] - ETA: 16s - loss: 7.6870 - accuracy: 0.4987
18112/25000 [====================>.........] - ETA: 16s - loss: 7.6836 - accuracy: 0.4989
18144/25000 [====================>.........] - ETA: 16s - loss: 7.6852 - accuracy: 0.4988
18176/25000 [====================>.........] - ETA: 16s - loss: 7.6877 - accuracy: 0.4986
18208/25000 [====================>.........] - ETA: 16s - loss: 7.6919 - accuracy: 0.4984
18240/25000 [====================>.........] - ETA: 16s - loss: 7.6893 - accuracy: 0.4985
18272/25000 [====================>.........] - ETA: 16s - loss: 7.6918 - accuracy: 0.4984
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6876 - accuracy: 0.4986
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6850 - accuracy: 0.4988
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6883 - accuracy: 0.4986
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6875 - accuracy: 0.4986
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6874 - accuracy: 0.4986
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6890 - accuracy: 0.4985
18496/25000 [=====================>........] - ETA: 15s - loss: 7.6898 - accuracy: 0.4985
18528/25000 [=====================>........] - ETA: 15s - loss: 7.6873 - accuracy: 0.4987
18560/25000 [=====================>........] - ETA: 15s - loss: 7.6840 - accuracy: 0.4989
18592/25000 [=====================>........] - ETA: 15s - loss: 7.6806 - accuracy: 0.4991
18624/25000 [=====================>........] - ETA: 15s - loss: 7.6814 - accuracy: 0.4990
18656/25000 [=====================>........] - ETA: 15s - loss: 7.6798 - accuracy: 0.4991
18688/25000 [=====================>........] - ETA: 15s - loss: 7.6797 - accuracy: 0.4991
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6805 - accuracy: 0.4991
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6797 - accuracy: 0.4991
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6789 - accuracy: 0.4992
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6821 - accuracy: 0.4990
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6853 - accuracy: 0.4988
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6845 - accuracy: 0.4988
18912/25000 [=====================>........] - ETA: 14s - loss: 7.6820 - accuracy: 0.4990
18944/25000 [=====================>........] - ETA: 14s - loss: 7.6836 - accuracy: 0.4989
18976/25000 [=====================>........] - ETA: 14s - loss: 7.6836 - accuracy: 0.4989
19008/25000 [=====================>........] - ETA: 14s - loss: 7.6828 - accuracy: 0.4989
19040/25000 [=====================>........] - ETA: 14s - loss: 7.6868 - accuracy: 0.4987
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6875 - accuracy: 0.4986
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6851 - accuracy: 0.4988
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6794 - accuracy: 0.4992
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6810 - accuracy: 0.4991
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6826 - accuracy: 0.4990
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6834 - accuracy: 0.4989
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6794 - accuracy: 0.4992
19296/25000 [======================>.......] - ETA: 13s - loss: 7.6825 - accuracy: 0.4990
19328/25000 [======================>.......] - ETA: 13s - loss: 7.6833 - accuracy: 0.4989
19360/25000 [======================>.......] - ETA: 13s - loss: 7.6825 - accuracy: 0.4990
19392/25000 [======================>.......] - ETA: 13s - loss: 7.6824 - accuracy: 0.4990
19424/25000 [======================>.......] - ETA: 13s - loss: 7.6824 - accuracy: 0.4990
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6816 - accuracy: 0.4990
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6847 - accuracy: 0.4988
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6847 - accuracy: 0.4988
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6847 - accuracy: 0.4988
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6878 - accuracy: 0.4986
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6869 - accuracy: 0.4987
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6893 - accuracy: 0.4985
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6861 - accuracy: 0.4987
19712/25000 [======================>.......] - ETA: 12s - loss: 7.6861 - accuracy: 0.4987
19744/25000 [======================>.......] - ETA: 12s - loss: 7.6829 - accuracy: 0.4989
19776/25000 [======================>.......] - ETA: 12s - loss: 7.6852 - accuracy: 0.4988
19808/25000 [======================>.......] - ETA: 12s - loss: 7.6852 - accuracy: 0.4988
19840/25000 [======================>.......] - ETA: 12s - loss: 7.6890 - accuracy: 0.4985
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6905 - accuracy: 0.4984
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6897 - accuracy: 0.4985
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6897 - accuracy: 0.4985
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6927 - accuracy: 0.4983
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6942 - accuracy: 0.4982
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6942 - accuracy: 0.4982
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6949 - accuracy: 0.4982
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6941 - accuracy: 0.4982
20128/25000 [=======================>......] - ETA: 11s - loss: 7.6940 - accuracy: 0.4982
20160/25000 [=======================>......] - ETA: 11s - loss: 7.6955 - accuracy: 0.4981
20192/25000 [=======================>......] - ETA: 11s - loss: 7.6902 - accuracy: 0.4985
20224/25000 [=======================>......] - ETA: 11s - loss: 7.6871 - accuracy: 0.4987
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6855 - accuracy: 0.4988
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6870 - accuracy: 0.4987
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6870 - accuracy: 0.4987
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6839 - accuracy: 0.4989
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6839 - accuracy: 0.4989
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6809 - accuracy: 0.4991
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6816 - accuracy: 0.4990
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6838 - accuracy: 0.4989
20512/25000 [=======================>......] - ETA: 10s - loss: 7.6808 - accuracy: 0.4991
20544/25000 [=======================>......] - ETA: 10s - loss: 7.6823 - accuracy: 0.4990
20576/25000 [=======================>......] - ETA: 10s - loss: 7.6815 - accuracy: 0.4990
20608/25000 [=======================>......] - ETA: 10s - loss: 7.6800 - accuracy: 0.4991
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6815 - accuracy: 0.4990
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6748 - accuracy: 0.4995
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6703 - accuracy: 0.4998
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6696 - accuracy: 0.4998
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6703 - accuracy: 0.4998
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6718 - accuracy: 0.4997
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6718 - accuracy: 0.4997
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6740 - accuracy: 0.4995
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6762 - accuracy: 0.4994
20928/25000 [========================>.....] - ETA: 9s - loss: 7.6805 - accuracy: 0.4991 
20960/25000 [========================>.....] - ETA: 9s - loss: 7.6783 - accuracy: 0.4992
20992/25000 [========================>.....] - ETA: 9s - loss: 7.6805 - accuracy: 0.4991
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6797 - accuracy: 0.4991
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6805 - accuracy: 0.4991
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6841 - accuracy: 0.4989
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6862 - accuracy: 0.4987
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6855 - accuracy: 0.4988
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6862 - accuracy: 0.4987
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6840 - accuracy: 0.4989
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6861 - accuracy: 0.4987
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6868 - accuracy: 0.4987
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6868 - accuracy: 0.4987
21344/25000 [========================>.....] - ETA: 8s - loss: 7.6889 - accuracy: 0.4985
21376/25000 [========================>.....] - ETA: 8s - loss: 7.6881 - accuracy: 0.4986
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6874 - accuracy: 0.4986
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6881 - accuracy: 0.4986
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6859 - accuracy: 0.4987
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6894 - accuracy: 0.4985
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6880 - accuracy: 0.4986
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6879 - accuracy: 0.4986
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6872 - accuracy: 0.4987
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6879 - accuracy: 0.4986
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6893 - accuracy: 0.4985
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6892 - accuracy: 0.4985
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6871 - accuracy: 0.4987
21760/25000 [=========================>....] - ETA: 7s - loss: 7.6849 - accuracy: 0.4988
21792/25000 [=========================>....] - ETA: 7s - loss: 7.6842 - accuracy: 0.4989
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6856 - accuracy: 0.4988
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6842 - accuracy: 0.4989
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6848 - accuracy: 0.4988
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6841 - accuracy: 0.4989
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6841 - accuracy: 0.4989
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6868 - accuracy: 0.4987
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6854 - accuracy: 0.4988
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6854 - accuracy: 0.4988
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6854 - accuracy: 0.4988
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6833 - accuracy: 0.4989
22144/25000 [=========================>....] - ETA: 6s - loss: 7.6839 - accuracy: 0.4989
22176/25000 [=========================>....] - ETA: 6s - loss: 7.6860 - accuracy: 0.4987
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6839 - accuracy: 0.4989
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6845 - accuracy: 0.4988
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6838 - accuracy: 0.4989
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6838 - accuracy: 0.4989
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6831 - accuracy: 0.4989
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6851 - accuracy: 0.4988
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6885 - accuracy: 0.4986
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6892 - accuracy: 0.4985
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6891 - accuracy: 0.4985
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6884 - accuracy: 0.4986
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6898 - accuracy: 0.4985
22560/25000 [==========================>...] - ETA: 5s - loss: 7.6877 - accuracy: 0.4986
22592/25000 [==========================>...] - ETA: 5s - loss: 7.6890 - accuracy: 0.4985
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6856 - accuracy: 0.4988
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6856 - accuracy: 0.4988
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6849 - accuracy: 0.4988
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6828 - accuracy: 0.4989
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6841 - accuracy: 0.4989
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6834 - accuracy: 0.4989
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6861 - accuracy: 0.4987
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6847 - accuracy: 0.4988
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6847 - accuracy: 0.4988
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6867 - accuracy: 0.4987
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6873 - accuracy: 0.4986
22976/25000 [==========================>...] - ETA: 4s - loss: 7.6873 - accuracy: 0.4987
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6886 - accuracy: 0.4986
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6892 - accuracy: 0.4985
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6866 - accuracy: 0.4987
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6825 - accuracy: 0.4990
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6845 - accuracy: 0.4988
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6852 - accuracy: 0.4988
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6878 - accuracy: 0.4986
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6924 - accuracy: 0.4983
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6923 - accuracy: 0.4983
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6910 - accuracy: 0.4984
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6896 - accuracy: 0.4985
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6909 - accuracy: 0.4984
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6902 - accuracy: 0.4985
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6895 - accuracy: 0.4985
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6869 - accuracy: 0.4987
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6856 - accuracy: 0.4988
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6855 - accuracy: 0.4988
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6875 - accuracy: 0.4986
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6900 - accuracy: 0.4985
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6893 - accuracy: 0.4985
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6887 - accuracy: 0.4986
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6873 - accuracy: 0.4986
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6860 - accuracy: 0.4987
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6853 - accuracy: 0.4988
23776/25000 [===========================>..] - ETA: 2s - loss: 7.6905 - accuracy: 0.4984
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6898 - accuracy: 0.4985
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6904 - accuracy: 0.4984
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6904 - accuracy: 0.4985
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6910 - accuracy: 0.4984
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6897 - accuracy: 0.4985
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6903 - accuracy: 0.4985
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6909 - accuracy: 0.4984
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6909 - accuracy: 0.4984
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6908 - accuracy: 0.4984
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6921 - accuracy: 0.4983
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6895 - accuracy: 0.4985
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6876 - accuracy: 0.4986
24192/25000 [============================>.] - ETA: 1s - loss: 7.6856 - accuracy: 0.4988
24224/25000 [============================>.] - ETA: 1s - loss: 7.6831 - accuracy: 0.4989
24256/25000 [============================>.] - ETA: 1s - loss: 7.6805 - accuracy: 0.4991
24288/25000 [============================>.] - ETA: 1s - loss: 7.6805 - accuracy: 0.4991
24320/25000 [============================>.] - ETA: 1s - loss: 7.6786 - accuracy: 0.4992
24352/25000 [============================>.] - ETA: 1s - loss: 7.6773 - accuracy: 0.4993
24384/25000 [============================>.] - ETA: 1s - loss: 7.6761 - accuracy: 0.4994
24416/25000 [============================>.] - ETA: 1s - loss: 7.6786 - accuracy: 0.4992
24448/25000 [============================>.] - ETA: 1s - loss: 7.6760 - accuracy: 0.4994
24480/25000 [============================>.] - ETA: 1s - loss: 7.6760 - accuracy: 0.4994
24512/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
24544/25000 [============================>.] - ETA: 1s - loss: 7.6741 - accuracy: 0.4995
24576/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24608/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24640/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24672/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24704/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24736/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24768/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24800/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24864/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24896/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24928/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24960/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 71s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fb689e10a90> 

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
 [ 0.10450535  0.13639887  0.14199267  0.05659797 -0.05660497  0.12360208]
 [ 0.14676589  0.08232898  0.20580134  0.19547467  0.123201    0.11450935]
 [ 0.06807818 -0.09918907  0.20924649  0.13227622  0.04724217 -0.196064  ]
 [ 0.13438044  0.02262241  0.18964256 -0.04049987 -0.16363446 -0.30003551]
 [ 0.2503815   0.40346113 -0.27995056  0.30871654  0.31710413  0.32276863]
 [-0.0596742  -0.35941818 -0.16372222  0.8334257   0.46321148 -0.2074433 ]
 [ 0.31130734  0.027463    0.09746288  0.57996285 -0.30198351  0.06222152]
 [-0.04586429 -0.17348886 -0.49933454  0.50368059  0.30664867 -0.38133788]
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
{'loss': 0.540978193283081, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
2020-05-11 05:15:25.304544: W tensorflow/core/util/tensor_slice_reader.cc:95] Could not open /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
2020-05-11 05:15:25.305529: W tensorflow/core/util/tensor_slice_reader.cc:95] Could not open /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
2020-05-11 05:15:25.305563: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_tensor.cc:175 : Data loss: Unable to open table file /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
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
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 315, in test
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
{'loss': 0.48145781829953194, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
2020-05-11 05:15:26.404321: W tensorflow/core/util/tensor_slice_reader.cc:95] Could not open /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
2020-05-11 05:15:26.406133: W tensorflow/core/util/tensor_slice_reader.cc:95] Could not open /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
2020-05-11 05:15:26.406304: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_tensor.cc:175 : Data loss: Unable to open table file /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model: Failed precondition: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model; Is a directory: perhaps your file is in a different file format and you need to use a different restore operator?
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
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
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 315, in test
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
 40%|████      | 2/5 [00:48<01:12, 24.11s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.35844892673720996, 'embedding_size_factor': 1.4133857594791484, 'layers.choice': 3, 'learning_rate': 0.005934988384632115, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 7.389970212149201e-10} and reward: 0.376
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd6\xf0\xd3\xc4g\xd8\xf4X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6\x9d:b\xd9\x85bX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?xOIP9\xe5jX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\tdIZ\xcc\xae\xc6u.' and reward: 0.376
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd6\xf0\xd3\xc4g\xd8\xf4X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6\x9d:b\xd9\x85bX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?xOIP9\xe5jX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\tdIZ\xcc\xae\xc6u.' and reward: 0.376
 60%|██████    | 3/5 [01:38<01:04, 32.10s/it] 60%|██████    | 3/5 [01:38<01:05, 32.99s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.3673207305283934, 'embedding_size_factor': 1.463763780276592, 'layers.choice': 3, 'learning_rate': 0.0008475121029618614, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 0.0003148187177457602} and reward: 0.3842
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd7\x82.\xcf0\xc9,X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7k\x93\x91\xd5\xb7xX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?K\xc5ra\xf1\x982X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?4\xa1\xc8\x18\xcd\x91\x02u.' and reward: 0.3842
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd7\x82.\xcf0\xc9,X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7k\x93\x91\xd5\xb7xX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?K\xc5ra\xf1\x982X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?4\xa1\xc8\x18\xcd\x91\x02u.' and reward: 0.3842

Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 150.16805720329285
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -32.4s of remaining time.
Ensemble size: 31
Ensemble weights: 
[0.22580645 0.38709677 0.38709677]
	0.394	 = Validation accuracy score
	1.04s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 153.48s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
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

