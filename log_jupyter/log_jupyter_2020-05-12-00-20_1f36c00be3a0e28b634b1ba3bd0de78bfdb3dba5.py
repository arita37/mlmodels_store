
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

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
[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     87[0m [0;34m[0m[0m
[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     90[0m [0;34m[0m[0m
[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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
 2056192/17464789 [==>...........................] - ETA: 0s
 6356992/17464789 [=========>....................] - ETA: 0s
10518528/17464789 [=================>............] - ETA: 0s
14475264/17464789 [=======================>......] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 00:21:16.035262: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 00:21:16.039619: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-12 00:21:16.039774: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559c0b9d7b70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 00:21:16.039792: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:49 - loss: 10.0624 - accuracy: 0.3438
   64/25000 [..............................] - ETA: 3:07 - loss: 8.1458 - accuracy: 0.4688 
   96/25000 [..............................] - ETA: 2:35 - loss: 7.1875 - accuracy: 0.5312
  128/25000 [..............................] - ETA: 2:19 - loss: 7.7864 - accuracy: 0.4922
  160/25000 [..............................] - ETA: 2:08 - loss: 7.7625 - accuracy: 0.4938
  192/25000 [..............................] - ETA: 2:01 - loss: 7.7465 - accuracy: 0.4948
  224/25000 [..............................] - ETA: 1:56 - loss: 7.8035 - accuracy: 0.4911
  256/25000 [..............................] - ETA: 1:53 - loss: 7.7864 - accuracy: 0.4922
  288/25000 [..............................] - ETA: 1:51 - loss: 7.6666 - accuracy: 0.5000
  320/25000 [..............................] - ETA: 1:49 - loss: 7.5229 - accuracy: 0.5094
  352/25000 [..............................] - ETA: 1:46 - loss: 7.5795 - accuracy: 0.5057
  384/25000 [..............................] - ETA: 1:44 - loss: 7.3871 - accuracy: 0.5182
  416/25000 [..............................] - ETA: 1:42 - loss: 7.5929 - accuracy: 0.5048
  448/25000 [..............................] - ETA: 1:41 - loss: 7.8720 - accuracy: 0.4866
  480/25000 [..............................] - ETA: 1:40 - loss: 7.8263 - accuracy: 0.4896
  512/25000 [..............................] - ETA: 1:38 - loss: 7.6666 - accuracy: 0.5000
  544/25000 [..............................] - ETA: 1:37 - loss: 7.6948 - accuracy: 0.4982
  576/25000 [..............................] - ETA: 1:37 - loss: 7.7199 - accuracy: 0.4965
  608/25000 [..............................] - ETA: 1:36 - loss: 7.8432 - accuracy: 0.4885
  640/25000 [..............................] - ETA: 1:35 - loss: 7.8343 - accuracy: 0.4891
  672/25000 [..............................] - ETA: 1:35 - loss: 7.8263 - accuracy: 0.4896
  704/25000 [..............................] - ETA: 1:34 - loss: 7.8626 - accuracy: 0.4872
  736/25000 [..............................] - ETA: 1:34 - loss: 7.8958 - accuracy: 0.4851
  768/25000 [..............................] - ETA: 1:33 - loss: 7.8663 - accuracy: 0.4870
  800/25000 [..............................] - ETA: 1:32 - loss: 7.8775 - accuracy: 0.4863
  832/25000 [..............................] - ETA: 1:32 - loss: 7.7956 - accuracy: 0.4916
  864/25000 [>.............................] - ETA: 1:31 - loss: 7.8973 - accuracy: 0.4850
  896/25000 [>.............................] - ETA: 1:31 - loss: 7.8720 - accuracy: 0.4866
  928/25000 [>.............................] - ETA: 1:31 - loss: 7.8153 - accuracy: 0.4903
  960/25000 [>.............................] - ETA: 1:30 - loss: 7.8423 - accuracy: 0.4885
  992/25000 [>.............................] - ETA: 1:30 - loss: 7.8985 - accuracy: 0.4849
 1024/25000 [>.............................] - ETA: 1:30 - loss: 7.8763 - accuracy: 0.4863
 1056/25000 [>.............................] - ETA: 1:29 - loss: 7.7973 - accuracy: 0.4915
 1088/25000 [>.............................] - ETA: 1:29 - loss: 7.7935 - accuracy: 0.4917
 1120/25000 [>.............................] - ETA: 1:29 - loss: 7.8035 - accuracy: 0.4911
 1152/25000 [>.............................] - ETA: 1:29 - loss: 7.7332 - accuracy: 0.4957
 1184/25000 [>.............................] - ETA: 1:29 - loss: 7.6537 - accuracy: 0.5008
 1216/25000 [>.............................] - ETA: 1:28 - loss: 7.5531 - accuracy: 0.5074
 1248/25000 [>.............................] - ETA: 1:28 - loss: 7.5683 - accuracy: 0.5064
 1280/25000 [>.............................] - ETA: 1:28 - loss: 7.5708 - accuracy: 0.5063
 1312/25000 [>.............................] - ETA: 1:27 - loss: 7.5498 - accuracy: 0.5076
 1344/25000 [>.............................] - ETA: 1:27 - loss: 7.5754 - accuracy: 0.5060
 1376/25000 [>.............................] - ETA: 1:27 - loss: 7.5663 - accuracy: 0.5065
 1408/25000 [>.............................] - ETA: 1:27 - loss: 7.5795 - accuracy: 0.5057
 1440/25000 [>.............................] - ETA: 1:26 - loss: 7.5921 - accuracy: 0.5049
 1472/25000 [>.............................] - ETA: 1:26 - loss: 7.5520 - accuracy: 0.5075
 1504/25000 [>.............................] - ETA: 1:26 - loss: 7.5647 - accuracy: 0.5066
 1536/25000 [>.............................] - ETA: 1:26 - loss: 7.5368 - accuracy: 0.5085
 1568/25000 [>.............................] - ETA: 1:25 - loss: 7.5591 - accuracy: 0.5070
 1600/25000 [>.............................] - ETA: 1:25 - loss: 7.5229 - accuracy: 0.5094
 1632/25000 [>.............................] - ETA: 1:25 - loss: 7.4881 - accuracy: 0.5116
 1664/25000 [>.............................] - ETA: 1:25 - loss: 7.5284 - accuracy: 0.5090
 1696/25000 [=>............................] - ETA: 1:25 - loss: 7.5400 - accuracy: 0.5083
 1728/25000 [=>............................] - ETA: 1:24 - loss: 7.5424 - accuracy: 0.5081
 1760/25000 [=>............................] - ETA: 1:24 - loss: 7.5882 - accuracy: 0.5051
 1792/25000 [=>............................] - ETA: 1:24 - loss: 7.5982 - accuracy: 0.5045
 1824/25000 [=>............................] - ETA: 1:24 - loss: 7.6078 - accuracy: 0.5038
 1856/25000 [=>............................] - ETA: 1:24 - loss: 7.6253 - accuracy: 0.5027
 1888/25000 [=>............................] - ETA: 1:24 - loss: 7.5854 - accuracy: 0.5053
 1920/25000 [=>............................] - ETA: 1:24 - loss: 7.5708 - accuracy: 0.5063
 1952/25000 [=>............................] - ETA: 1:23 - loss: 7.5331 - accuracy: 0.5087
 1984/25000 [=>............................] - ETA: 1:23 - loss: 7.5352 - accuracy: 0.5086
 2016/25000 [=>............................] - ETA: 1:23 - loss: 7.5525 - accuracy: 0.5074
 2048/25000 [=>............................] - ETA: 1:23 - loss: 7.5618 - accuracy: 0.5068
 2080/25000 [=>............................] - ETA: 1:23 - loss: 7.5855 - accuracy: 0.5053
 2112/25000 [=>............................] - ETA: 1:23 - loss: 7.5940 - accuracy: 0.5047
 2144/25000 [=>............................] - ETA: 1:22 - loss: 7.6094 - accuracy: 0.5037
 2176/25000 [=>............................] - ETA: 1:22 - loss: 7.6525 - accuracy: 0.5009
 2208/25000 [=>............................] - ETA: 1:22 - loss: 7.6666 - accuracy: 0.5000
 2240/25000 [=>............................] - ETA: 1:22 - loss: 7.6324 - accuracy: 0.5022
 2272/25000 [=>............................] - ETA: 1:22 - loss: 7.6396 - accuracy: 0.5018
 2304/25000 [=>............................] - ETA: 1:22 - loss: 7.6267 - accuracy: 0.5026
 2336/25000 [=>............................] - ETA: 1:22 - loss: 7.5944 - accuracy: 0.5047
 2368/25000 [=>............................] - ETA: 1:21 - loss: 7.5695 - accuracy: 0.5063
 2400/25000 [=>............................] - ETA: 1:21 - loss: 7.5772 - accuracy: 0.5058
 2432/25000 [=>............................] - ETA: 1:21 - loss: 7.5910 - accuracy: 0.5049
 2464/25000 [=>............................] - ETA: 1:21 - loss: 7.6044 - accuracy: 0.5041
 2496/25000 [=>............................] - ETA: 1:21 - loss: 7.5868 - accuracy: 0.5052
 2528/25000 [==>...........................] - ETA: 1:20 - loss: 7.5878 - accuracy: 0.5051
 2560/25000 [==>...........................] - ETA: 1:20 - loss: 7.5648 - accuracy: 0.5066
 2592/25000 [==>...........................] - ETA: 1:20 - loss: 7.5661 - accuracy: 0.5066
 2624/25000 [==>...........................] - ETA: 1:20 - loss: 7.5731 - accuracy: 0.5061
 2656/25000 [==>...........................] - ETA: 1:20 - loss: 7.5743 - accuracy: 0.5060
 2688/25000 [==>...........................] - ETA: 1:20 - loss: 7.5982 - accuracy: 0.5045
 2720/25000 [==>...........................] - ETA: 1:20 - loss: 7.6046 - accuracy: 0.5040
 2752/25000 [==>...........................] - ETA: 1:19 - loss: 7.6165 - accuracy: 0.5033
 2784/25000 [==>...........................] - ETA: 1:19 - loss: 7.5895 - accuracy: 0.5050
 2816/25000 [==>...........................] - ETA: 1:19 - loss: 7.5904 - accuracy: 0.5050
 2848/25000 [==>...........................] - ETA: 1:19 - loss: 7.5751 - accuracy: 0.5060
 2880/25000 [==>...........................] - ETA: 1:19 - loss: 7.5814 - accuracy: 0.5056
 2912/25000 [==>...........................] - ETA: 1:19 - loss: 7.5876 - accuracy: 0.5052
 2944/25000 [==>...........................] - ETA: 1:19 - loss: 7.5885 - accuracy: 0.5051
 2976/25000 [==>...........................] - ETA: 1:18 - loss: 7.5739 - accuracy: 0.5060
 3008/25000 [==>...........................] - ETA: 1:18 - loss: 7.5749 - accuracy: 0.5060
 3040/25000 [==>...........................] - ETA: 1:18 - loss: 7.5859 - accuracy: 0.5053
 3072/25000 [==>...........................] - ETA: 1:18 - loss: 7.5868 - accuracy: 0.5052
 3104/25000 [==>...........................] - ETA: 1:18 - loss: 7.5975 - accuracy: 0.5045
 3136/25000 [==>...........................] - ETA: 1:18 - loss: 7.5982 - accuracy: 0.5045
 3168/25000 [==>...........................] - ETA: 1:18 - loss: 7.5795 - accuracy: 0.5057
 3200/25000 [==>...........................] - ETA: 1:17 - loss: 7.5804 - accuracy: 0.5056
 3232/25000 [==>...........................] - ETA: 1:17 - loss: 7.5860 - accuracy: 0.5053
 3264/25000 [==>...........................] - ETA: 1:17 - loss: 7.5727 - accuracy: 0.5061
 3296/25000 [==>...........................] - ETA: 1:17 - loss: 7.5782 - accuracy: 0.5058
 3328/25000 [==>...........................] - ETA: 1:17 - loss: 7.6113 - accuracy: 0.5036
 3360/25000 [===>..........................] - ETA: 1:17 - loss: 7.5936 - accuracy: 0.5048
 3392/25000 [===>..........................] - ETA: 1:17 - loss: 7.5943 - accuracy: 0.5047
 3424/25000 [===>..........................] - ETA: 1:17 - loss: 7.5860 - accuracy: 0.5053
 3456/25000 [===>..........................] - ETA: 1:16 - loss: 7.5912 - accuracy: 0.5049
 3488/25000 [===>..........................] - ETA: 1:16 - loss: 7.6007 - accuracy: 0.5043
 3520/25000 [===>..........................] - ETA: 1:16 - loss: 7.6100 - accuracy: 0.5037
 3552/25000 [===>..........................] - ETA: 1:16 - loss: 7.5976 - accuracy: 0.5045
 3584/25000 [===>..........................] - ETA: 1:16 - loss: 7.6024 - accuracy: 0.5042
 3616/25000 [===>..........................] - ETA: 1:16 - loss: 7.5903 - accuracy: 0.5050
 3648/25000 [===>..........................] - ETA: 1:16 - loss: 7.6036 - accuracy: 0.5041
 3680/25000 [===>..........................] - ETA: 1:16 - loss: 7.6083 - accuracy: 0.5038
 3712/25000 [===>..........................] - ETA: 1:15 - loss: 7.6047 - accuracy: 0.5040
 3744/25000 [===>..........................] - ETA: 1:15 - loss: 7.6052 - accuracy: 0.5040
 3776/25000 [===>..........................] - ETA: 1:15 - loss: 7.6098 - accuracy: 0.5037
 3808/25000 [===>..........................] - ETA: 1:15 - loss: 7.6183 - accuracy: 0.5032
 3840/25000 [===>..........................] - ETA: 1:15 - loss: 7.6107 - accuracy: 0.5036
 3872/25000 [===>..........................] - ETA: 1:15 - loss: 7.6112 - accuracy: 0.5036
 3904/25000 [===>..........................] - ETA: 1:15 - loss: 7.6077 - accuracy: 0.5038
 3936/25000 [===>..........................] - ETA: 1:15 - loss: 7.6199 - accuracy: 0.5030
 3968/25000 [===>..........................] - ETA: 1:14 - loss: 7.6357 - accuracy: 0.5020
 4000/25000 [===>..........................] - ETA: 1:14 - loss: 7.6321 - accuracy: 0.5023
 4032/25000 [===>..........................] - ETA: 1:14 - loss: 7.6096 - accuracy: 0.5037
 4064/25000 [===>..........................] - ETA: 1:14 - loss: 7.6100 - accuracy: 0.5037
 4096/25000 [===>..........................] - ETA: 1:14 - loss: 7.6217 - accuracy: 0.5029
 4128/25000 [===>..........................] - ETA: 1:14 - loss: 7.6183 - accuracy: 0.5031
 4160/25000 [===>..........................] - ETA: 1:14 - loss: 7.6150 - accuracy: 0.5034
 4192/25000 [====>.........................] - ETA: 1:14 - loss: 7.6118 - accuracy: 0.5036
 4224/25000 [====>.........................] - ETA: 1:13 - loss: 7.6194 - accuracy: 0.5031
 4256/25000 [====>.........................] - ETA: 1:13 - loss: 7.6198 - accuracy: 0.5031
 4288/25000 [====>.........................] - ETA: 1:13 - loss: 7.6094 - accuracy: 0.5037
 4320/25000 [====>.........................] - ETA: 1:13 - loss: 7.6063 - accuracy: 0.5039
 4352/25000 [====>.........................] - ETA: 1:13 - loss: 7.5997 - accuracy: 0.5044
 4384/25000 [====>.........................] - ETA: 1:13 - loss: 7.5932 - accuracy: 0.5048
 4416/25000 [====>.........................] - ETA: 1:13 - loss: 7.6111 - accuracy: 0.5036
 4448/25000 [====>.........................] - ETA: 1:13 - loss: 7.6218 - accuracy: 0.5029
 4480/25000 [====>.........................] - ETA: 1:12 - loss: 7.6255 - accuracy: 0.5027
 4512/25000 [====>.........................] - ETA: 1:12 - loss: 7.6122 - accuracy: 0.5035
 4544/25000 [====>.........................] - ETA: 1:12 - loss: 7.6126 - accuracy: 0.5035
 4576/25000 [====>.........................] - ETA: 1:12 - loss: 7.6130 - accuracy: 0.5035
 4608/25000 [====>.........................] - ETA: 1:12 - loss: 7.6101 - accuracy: 0.5037
 4640/25000 [====>.........................] - ETA: 1:12 - loss: 7.5906 - accuracy: 0.5050
 4672/25000 [====>.........................] - ETA: 1:12 - loss: 7.5879 - accuracy: 0.5051
 4704/25000 [====>.........................] - ETA: 1:12 - loss: 7.5884 - accuracy: 0.5051
 4736/25000 [====>.........................] - ETA: 1:12 - loss: 7.5889 - accuracy: 0.5051
 4768/25000 [====>.........................] - ETA: 1:11 - loss: 7.5830 - accuracy: 0.5055
 4800/25000 [====>.........................] - ETA: 1:11 - loss: 7.5868 - accuracy: 0.5052
 4832/25000 [====>.........................] - ETA: 1:11 - loss: 7.5968 - accuracy: 0.5046
 4864/25000 [====>.........................] - ETA: 1:11 - loss: 7.6036 - accuracy: 0.5041
 4896/25000 [====>.........................] - ETA: 1:11 - loss: 7.5946 - accuracy: 0.5047
 4928/25000 [====>.........................] - ETA: 1:11 - loss: 7.5826 - accuracy: 0.5055
 4960/25000 [====>.........................] - ETA: 1:11 - loss: 7.5955 - accuracy: 0.5046
 4992/25000 [====>.........................] - ETA: 1:11 - loss: 7.6021 - accuracy: 0.5042
 5024/25000 [=====>........................] - ETA: 1:10 - loss: 7.6117 - accuracy: 0.5036
 5056/25000 [=====>........................] - ETA: 1:10 - loss: 7.6029 - accuracy: 0.5042
 5088/25000 [=====>........................] - ETA: 1:10 - loss: 7.6033 - accuracy: 0.5041
 5120/25000 [=====>........................] - ETA: 1:10 - loss: 7.5977 - accuracy: 0.5045
 5152/25000 [=====>........................] - ETA: 1:10 - loss: 7.5952 - accuracy: 0.5047
 5184/25000 [=====>........................] - ETA: 1:10 - loss: 7.5897 - accuracy: 0.5050
 5216/25000 [=====>........................] - ETA: 1:10 - loss: 7.5814 - accuracy: 0.5056
 5248/25000 [=====>........................] - ETA: 1:10 - loss: 7.5760 - accuracy: 0.5059
 5280/25000 [=====>........................] - ETA: 1:10 - loss: 7.5737 - accuracy: 0.5061
 5312/25000 [=====>........................] - ETA: 1:09 - loss: 7.5714 - accuracy: 0.5062
 5344/25000 [=====>........................] - ETA: 1:09 - loss: 7.5633 - accuracy: 0.5067
 5376/25000 [=====>........................] - ETA: 1:09 - loss: 7.5639 - accuracy: 0.5067
 5408/25000 [=====>........................] - ETA: 1:09 - loss: 7.5674 - accuracy: 0.5065
 5440/25000 [=====>........................] - ETA: 1:09 - loss: 7.5708 - accuracy: 0.5063
 5472/25000 [=====>........................] - ETA: 1:09 - loss: 7.5826 - accuracy: 0.5055
 5504/25000 [=====>........................] - ETA: 1:09 - loss: 7.5886 - accuracy: 0.5051
 5536/25000 [=====>........................] - ETA: 1:09 - loss: 7.6085 - accuracy: 0.5038
 5568/25000 [=====>........................] - ETA: 1:08 - loss: 7.6005 - accuracy: 0.5043
 5600/25000 [=====>........................] - ETA: 1:08 - loss: 7.6009 - accuracy: 0.5043
 5632/25000 [=====>........................] - ETA: 1:08 - loss: 7.6094 - accuracy: 0.5037
 5664/25000 [=====>........................] - ETA: 1:08 - loss: 7.6044 - accuracy: 0.5041
 5696/25000 [=====>........................] - ETA: 1:08 - loss: 7.5886 - accuracy: 0.5051
 5728/25000 [=====>........................] - ETA: 1:08 - loss: 7.5943 - accuracy: 0.5047
 5760/25000 [=====>........................] - ETA: 1:08 - loss: 7.5894 - accuracy: 0.5050
 5792/25000 [=====>........................] - ETA: 1:08 - loss: 7.5713 - accuracy: 0.5062
 5824/25000 [=====>........................] - ETA: 1:07 - loss: 7.5771 - accuracy: 0.5058
 5856/25000 [======>.......................] - ETA: 1:07 - loss: 7.5750 - accuracy: 0.5060
 5888/25000 [======>.......................] - ETA: 1:07 - loss: 7.5651 - accuracy: 0.5066
 5920/25000 [======>.......................] - ETA: 1:07 - loss: 7.5604 - accuracy: 0.5069
 5952/25000 [======>.......................] - ETA: 1:07 - loss: 7.5713 - accuracy: 0.5062
 5984/25000 [======>.......................] - ETA: 1:07 - loss: 7.5795 - accuracy: 0.5057
 6016/25000 [======>.......................] - ETA: 1:07 - loss: 7.5749 - accuracy: 0.5060
 6048/25000 [======>.......................] - ETA: 1:07 - loss: 7.5804 - accuracy: 0.5056
 6080/25000 [======>.......................] - ETA: 1:07 - loss: 7.5809 - accuracy: 0.5056
 6112/25000 [======>.......................] - ETA: 1:06 - loss: 7.5763 - accuracy: 0.5059
 6144/25000 [======>.......................] - ETA: 1:06 - loss: 7.5743 - accuracy: 0.5060
 6176/25000 [======>.......................] - ETA: 1:06 - loss: 7.5748 - accuracy: 0.5060
 6208/25000 [======>.......................] - ETA: 1:06 - loss: 7.5728 - accuracy: 0.5061
 6240/25000 [======>.......................] - ETA: 1:06 - loss: 7.5782 - accuracy: 0.5058
 6272/25000 [======>.......................] - ETA: 1:06 - loss: 7.5688 - accuracy: 0.5064
 6304/25000 [======>.......................] - ETA: 1:06 - loss: 7.5693 - accuracy: 0.5063
 6336/25000 [======>.......................] - ETA: 1:06 - loss: 7.5650 - accuracy: 0.5066
 6368/25000 [======>.......................] - ETA: 1:06 - loss: 7.5631 - accuracy: 0.5068
 6400/25000 [======>.......................] - ETA: 1:05 - loss: 7.5708 - accuracy: 0.5063
 6432/25000 [======>.......................] - ETA: 1:05 - loss: 7.5736 - accuracy: 0.5061
 6464/25000 [======>.......................] - ETA: 1:05 - loss: 7.5717 - accuracy: 0.5062
 6496/25000 [======>.......................] - ETA: 1:05 - loss: 7.5580 - accuracy: 0.5071
 6528/25000 [======>.......................] - ETA: 1:05 - loss: 7.5539 - accuracy: 0.5074
 6560/25000 [======>.......................] - ETA: 1:05 - loss: 7.5638 - accuracy: 0.5067
 6592/25000 [======>.......................] - ETA: 1:05 - loss: 7.5736 - accuracy: 0.5061
 6624/25000 [======>.......................] - ETA: 1:05 - loss: 7.5694 - accuracy: 0.5063
 6656/25000 [======>.......................] - ETA: 1:05 - loss: 7.5699 - accuracy: 0.5063
 6688/25000 [=======>......................] - ETA: 1:04 - loss: 7.5795 - accuracy: 0.5057
 6720/25000 [=======>......................] - ETA: 1:04 - loss: 7.5845 - accuracy: 0.5054
 6752/25000 [=======>......................] - ETA: 1:04 - loss: 7.5803 - accuracy: 0.5056
 6784/25000 [=======>......................] - ETA: 1:04 - loss: 7.5785 - accuracy: 0.5057
 6816/25000 [=======>......................] - ETA: 1:04 - loss: 7.5856 - accuracy: 0.5053
 6848/25000 [=======>......................] - ETA: 1:04 - loss: 7.5838 - accuracy: 0.5054
 6880/25000 [=======>......................] - ETA: 1:04 - loss: 7.5842 - accuracy: 0.5054
 6912/25000 [=======>......................] - ETA: 1:04 - loss: 7.5801 - accuracy: 0.5056
 6944/25000 [=======>......................] - ETA: 1:04 - loss: 7.5783 - accuracy: 0.5058
 6976/25000 [=======>......................] - ETA: 1:03 - loss: 7.5699 - accuracy: 0.5063
 7008/25000 [=======>......................] - ETA: 1:03 - loss: 7.5747 - accuracy: 0.5060
 7040/25000 [=======>......................] - ETA: 1:03 - loss: 7.5839 - accuracy: 0.5054
 7072/25000 [=======>......................] - ETA: 1:03 - loss: 7.5864 - accuracy: 0.5052
 7104/25000 [=======>......................] - ETA: 1:03 - loss: 7.5803 - accuracy: 0.5056
 7136/25000 [=======>......................] - ETA: 1:03 - loss: 7.5871 - accuracy: 0.5052
 7168/25000 [=======>......................] - ETA: 1:03 - loss: 7.5746 - accuracy: 0.5060
 7200/25000 [=======>......................] - ETA: 1:03 - loss: 7.5772 - accuracy: 0.5058
 7232/25000 [=======>......................] - ETA: 1:02 - loss: 7.5776 - accuracy: 0.5058
 7264/25000 [=======>......................] - ETA: 1:02 - loss: 7.5780 - accuracy: 0.5058
 7296/25000 [=======>......................] - ETA: 1:02 - loss: 7.5952 - accuracy: 0.5047
 7328/25000 [=======>......................] - ETA: 1:02 - loss: 7.5913 - accuracy: 0.5049
 7360/25000 [=======>......................] - ETA: 1:02 - loss: 7.5833 - accuracy: 0.5054
 7392/25000 [=======>......................] - ETA: 1:02 - loss: 7.5795 - accuracy: 0.5057
 7424/25000 [=======>......................] - ETA: 1:02 - loss: 7.5943 - accuracy: 0.5047
 7456/25000 [=======>......................] - ETA: 1:02 - loss: 7.5988 - accuracy: 0.5044
 7488/25000 [=======>......................] - ETA: 1:02 - loss: 7.5949 - accuracy: 0.5047
 7520/25000 [========>.....................] - ETA: 1:01 - loss: 7.6075 - accuracy: 0.5039
 7552/25000 [========>.....................] - ETA: 1:01 - loss: 7.6098 - accuracy: 0.5037
 7584/25000 [========>.....................] - ETA: 1:01 - loss: 7.6141 - accuracy: 0.5034
 7616/25000 [========>.....................] - ETA: 1:01 - loss: 7.6163 - accuracy: 0.5033
 7648/25000 [========>.....................] - ETA: 1:01 - loss: 7.6145 - accuracy: 0.5034
 7680/25000 [========>.....................] - ETA: 1:01 - loss: 7.6107 - accuracy: 0.5036
 7712/25000 [========>.....................] - ETA: 1:01 - loss: 7.6209 - accuracy: 0.5030
 7744/25000 [========>.....................] - ETA: 1:01 - loss: 7.6191 - accuracy: 0.5031
 7776/25000 [========>.....................] - ETA: 1:01 - loss: 7.6213 - accuracy: 0.5030
 7808/25000 [========>.....................] - ETA: 1:00 - loss: 7.6136 - accuracy: 0.5035
 7840/25000 [========>.....................] - ETA: 1:00 - loss: 7.6099 - accuracy: 0.5037
 7872/25000 [========>.....................] - ETA: 1:00 - loss: 7.6140 - accuracy: 0.5034
 7904/25000 [========>.....................] - ETA: 1:00 - loss: 7.6084 - accuracy: 0.5038
 7936/25000 [========>.....................] - ETA: 1:00 - loss: 7.6087 - accuracy: 0.5038
 7968/25000 [========>.....................] - ETA: 1:00 - loss: 7.6127 - accuracy: 0.5035
 8000/25000 [========>.....................] - ETA: 1:00 - loss: 7.6053 - accuracy: 0.5040
 8032/25000 [========>.....................] - ETA: 1:00 - loss: 7.5979 - accuracy: 0.5045
 8064/25000 [========>.....................] - ETA: 59s - loss: 7.6058 - accuracy: 0.5040 
 8096/25000 [========>.....................] - ETA: 59s - loss: 7.6041 - accuracy: 0.5041
 8128/25000 [========>.....................] - ETA: 59s - loss: 7.6119 - accuracy: 0.5036
 8160/25000 [========>.....................] - ETA: 59s - loss: 7.6159 - accuracy: 0.5033
 8192/25000 [========>.....................] - ETA: 59s - loss: 7.6236 - accuracy: 0.5028
 8224/25000 [========>.....................] - ETA: 59s - loss: 7.6200 - accuracy: 0.5030
 8256/25000 [========>.....................] - ETA: 59s - loss: 7.6220 - accuracy: 0.5029
 8288/25000 [========>.....................] - ETA: 59s - loss: 7.6315 - accuracy: 0.5023
 8320/25000 [========>.....................] - ETA: 59s - loss: 7.6445 - accuracy: 0.5014
 8352/25000 [=========>....................] - ETA: 58s - loss: 7.6428 - accuracy: 0.5016
 8384/25000 [=========>....................] - ETA: 58s - loss: 7.6410 - accuracy: 0.5017
 8416/25000 [=========>....................] - ETA: 58s - loss: 7.6429 - accuracy: 0.5015
 8448/25000 [=========>....................] - ETA: 58s - loss: 7.6358 - accuracy: 0.5020
 8480/25000 [=========>....................] - ETA: 58s - loss: 7.6286 - accuracy: 0.5025
 8512/25000 [=========>....................] - ETA: 58s - loss: 7.6252 - accuracy: 0.5027
 8544/25000 [=========>....................] - ETA: 58s - loss: 7.6235 - accuracy: 0.5028
 8576/25000 [=========>....................] - ETA: 58s - loss: 7.6201 - accuracy: 0.5030
 8608/25000 [=========>....................] - ETA: 58s - loss: 7.6185 - accuracy: 0.5031
 8640/25000 [=========>....................] - ETA: 57s - loss: 7.6205 - accuracy: 0.5030
 8672/25000 [=========>....................] - ETA: 57s - loss: 7.6189 - accuracy: 0.5031
 8704/25000 [=========>....................] - ETA: 57s - loss: 7.6261 - accuracy: 0.5026
 8736/25000 [=========>....................] - ETA: 57s - loss: 7.6298 - accuracy: 0.5024
 8768/25000 [=========>....................] - ETA: 57s - loss: 7.6281 - accuracy: 0.5025
 8800/25000 [=========>....................] - ETA: 57s - loss: 7.6283 - accuracy: 0.5025
 8832/25000 [=========>....................] - ETA: 57s - loss: 7.6267 - accuracy: 0.5026
 8864/25000 [=========>....................] - ETA: 57s - loss: 7.6268 - accuracy: 0.5026
 8896/25000 [=========>....................] - ETA: 57s - loss: 7.6356 - accuracy: 0.5020
 8928/25000 [=========>....................] - ETA: 56s - loss: 7.6391 - accuracy: 0.5018
 8960/25000 [=========>....................] - ETA: 56s - loss: 7.6461 - accuracy: 0.5013
 8992/25000 [=========>....................] - ETA: 56s - loss: 7.6530 - accuracy: 0.5009
 9024/25000 [=========>....................] - ETA: 56s - loss: 7.6496 - accuracy: 0.5011
 9056/25000 [=========>....................] - ETA: 56s - loss: 7.6463 - accuracy: 0.5013
 9088/25000 [=========>....................] - ETA: 56s - loss: 7.6481 - accuracy: 0.5012
 9120/25000 [=========>....................] - ETA: 56s - loss: 7.6431 - accuracy: 0.5015
 9152/25000 [=========>....................] - ETA: 56s - loss: 7.6365 - accuracy: 0.5020
 9184/25000 [==========>...................] - ETA: 56s - loss: 7.6432 - accuracy: 0.5015
 9216/25000 [==========>...................] - ETA: 55s - loss: 7.6433 - accuracy: 0.5015
 9248/25000 [==========>...................] - ETA: 55s - loss: 7.6517 - accuracy: 0.5010
 9280/25000 [==========>...................] - ETA: 55s - loss: 7.6517 - accuracy: 0.5010
 9312/25000 [==========>...................] - ETA: 55s - loss: 7.6469 - accuracy: 0.5013
 9344/25000 [==========>...................] - ETA: 55s - loss: 7.6502 - accuracy: 0.5011
 9376/25000 [==========>...................] - ETA: 55s - loss: 7.6437 - accuracy: 0.5015
 9408/25000 [==========>...................] - ETA: 55s - loss: 7.6373 - accuracy: 0.5019
 9440/25000 [==========>...................] - ETA: 55s - loss: 7.6358 - accuracy: 0.5020
 9472/25000 [==========>...................] - ETA: 54s - loss: 7.6342 - accuracy: 0.5021
 9504/25000 [==========>...................] - ETA: 54s - loss: 7.6295 - accuracy: 0.5024
 9536/25000 [==========>...................] - ETA: 54s - loss: 7.6264 - accuracy: 0.5026
 9568/25000 [==========>...................] - ETA: 54s - loss: 7.6234 - accuracy: 0.5028
 9600/25000 [==========>...................] - ETA: 54s - loss: 7.6235 - accuracy: 0.5028
 9632/25000 [==========>...................] - ETA: 54s - loss: 7.6205 - accuracy: 0.5030
 9664/25000 [==========>...................] - ETA: 54s - loss: 7.6238 - accuracy: 0.5028
 9696/25000 [==========>...................] - ETA: 54s - loss: 7.6223 - accuracy: 0.5029
 9728/25000 [==========>...................] - ETA: 54s - loss: 7.6256 - accuracy: 0.5027
 9760/25000 [==========>...................] - ETA: 53s - loss: 7.6305 - accuracy: 0.5024
 9792/25000 [==========>...................] - ETA: 53s - loss: 7.6337 - accuracy: 0.5021
 9824/25000 [==========>...................] - ETA: 53s - loss: 7.6338 - accuracy: 0.5021
 9856/25000 [==========>...................] - ETA: 53s - loss: 7.6308 - accuracy: 0.5023
 9888/25000 [==========>...................] - ETA: 53s - loss: 7.6279 - accuracy: 0.5025
 9920/25000 [==========>...................] - ETA: 53s - loss: 7.6280 - accuracy: 0.5025
 9952/25000 [==========>...................] - ETA: 53s - loss: 7.6312 - accuracy: 0.5023
 9984/25000 [==========>...................] - ETA: 53s - loss: 7.6252 - accuracy: 0.5027
10016/25000 [===========>..................] - ETA: 53s - loss: 7.6176 - accuracy: 0.5032
10048/25000 [===========>..................] - ETA: 52s - loss: 7.6193 - accuracy: 0.5031
10080/25000 [===========>..................] - ETA: 52s - loss: 7.6210 - accuracy: 0.5030
10112/25000 [===========>..................] - ETA: 52s - loss: 7.6211 - accuracy: 0.5030
10144/25000 [===========>..................] - ETA: 52s - loss: 7.6198 - accuracy: 0.5031
10176/25000 [===========>..................] - ETA: 52s - loss: 7.6169 - accuracy: 0.5032
10208/25000 [===========>..................] - ETA: 52s - loss: 7.6155 - accuracy: 0.5033
10240/25000 [===========>..................] - ETA: 52s - loss: 7.6157 - accuracy: 0.5033
10272/25000 [===========>..................] - ETA: 52s - loss: 7.6084 - accuracy: 0.5038
10304/25000 [===========>..................] - ETA: 52s - loss: 7.6116 - accuracy: 0.5036
10336/25000 [===========>..................] - ETA: 51s - loss: 7.6073 - accuracy: 0.5039
10368/25000 [===========>..................] - ETA: 51s - loss: 7.6104 - accuracy: 0.5037
10400/25000 [===========>..................] - ETA: 51s - loss: 7.6076 - accuracy: 0.5038
10432/25000 [===========>..................] - ETA: 51s - loss: 7.6137 - accuracy: 0.5035
10464/25000 [===========>..................] - ETA: 51s - loss: 7.6065 - accuracy: 0.5039
10496/25000 [===========>..................] - ETA: 51s - loss: 7.6053 - accuracy: 0.5040
10528/25000 [===========>..................] - ETA: 51s - loss: 7.6054 - accuracy: 0.5040
10560/25000 [===========>..................] - ETA: 51s - loss: 7.6042 - accuracy: 0.5041
10592/25000 [===========>..................] - ETA: 50s - loss: 7.6073 - accuracy: 0.5039
10624/25000 [===========>..................] - ETA: 50s - loss: 7.6103 - accuracy: 0.5037
10656/25000 [===========>..................] - ETA: 50s - loss: 7.6105 - accuracy: 0.5037
10688/25000 [===========>..................] - ETA: 50s - loss: 7.6164 - accuracy: 0.5033
10720/25000 [===========>..................] - ETA: 50s - loss: 7.6223 - accuracy: 0.5029
10752/25000 [===========>..................] - ETA: 50s - loss: 7.6124 - accuracy: 0.5035
10784/25000 [===========>..................] - ETA: 50s - loss: 7.6097 - accuracy: 0.5037
10816/25000 [===========>..................] - ETA: 50s - loss: 7.6042 - accuracy: 0.5041
10848/25000 [============>.................] - ETA: 50s - loss: 7.6016 - accuracy: 0.5042
10880/25000 [============>.................] - ETA: 49s - loss: 7.6032 - accuracy: 0.5041
10912/25000 [============>.................] - ETA: 49s - loss: 7.6006 - accuracy: 0.5043
10944/25000 [============>.................] - ETA: 49s - loss: 7.6050 - accuracy: 0.5040
10976/25000 [============>.................] - ETA: 49s - loss: 7.6052 - accuracy: 0.5040
11008/25000 [============>.................] - ETA: 49s - loss: 7.6025 - accuracy: 0.5042
11040/25000 [============>.................] - ETA: 49s - loss: 7.6111 - accuracy: 0.5036
11072/25000 [============>.................] - ETA: 49s - loss: 7.6126 - accuracy: 0.5035
11104/25000 [============>.................] - ETA: 49s - loss: 7.6114 - accuracy: 0.5036
11136/25000 [============>.................] - ETA: 49s - loss: 7.6074 - accuracy: 0.5039
11168/25000 [============>.................] - ETA: 48s - loss: 7.6117 - accuracy: 0.5036
11200/25000 [============>.................] - ETA: 48s - loss: 7.6119 - accuracy: 0.5036
11232/25000 [============>.................] - ETA: 48s - loss: 7.6106 - accuracy: 0.5037
11264/25000 [============>.................] - ETA: 48s - loss: 7.6176 - accuracy: 0.5032
11296/25000 [============>.................] - ETA: 48s - loss: 7.6110 - accuracy: 0.5036
11328/25000 [============>.................] - ETA: 48s - loss: 7.6057 - accuracy: 0.5040
11360/25000 [============>.................] - ETA: 48s - loss: 7.6045 - accuracy: 0.5040
11392/25000 [============>.................] - ETA: 48s - loss: 7.6061 - accuracy: 0.5040
11424/25000 [============>.................] - ETA: 48s - loss: 7.5982 - accuracy: 0.5045
11456/25000 [============>.................] - ETA: 47s - loss: 7.5997 - accuracy: 0.5044
11488/25000 [============>.................] - ETA: 47s - loss: 7.5999 - accuracy: 0.5044
11520/25000 [============>.................] - ETA: 47s - loss: 7.6001 - accuracy: 0.5043
11552/25000 [============>.................] - ETA: 47s - loss: 7.6016 - accuracy: 0.5042
11584/25000 [============>.................] - ETA: 47s - loss: 7.6004 - accuracy: 0.5043
11616/25000 [============>.................] - ETA: 47s - loss: 7.6046 - accuracy: 0.5040
11648/25000 [============>.................] - ETA: 47s - loss: 7.6047 - accuracy: 0.5040
11680/25000 [=============>................] - ETA: 47s - loss: 7.6089 - accuracy: 0.5038
11712/25000 [=============>................] - ETA: 47s - loss: 7.6143 - accuracy: 0.5034
11744/25000 [=============>................] - ETA: 46s - loss: 7.6144 - accuracy: 0.5034
11776/25000 [=============>................] - ETA: 46s - loss: 7.6132 - accuracy: 0.5035
11808/25000 [=============>................] - ETA: 46s - loss: 7.6082 - accuracy: 0.5038
11840/25000 [=============>................] - ETA: 46s - loss: 7.6083 - accuracy: 0.5038
11872/25000 [=============>................] - ETA: 46s - loss: 7.6046 - accuracy: 0.5040
11904/25000 [=============>................] - ETA: 46s - loss: 7.6087 - accuracy: 0.5038
11936/25000 [=============>................] - ETA: 46s - loss: 7.6101 - accuracy: 0.5037
11968/25000 [=============>................] - ETA: 46s - loss: 7.6026 - accuracy: 0.5042
12000/25000 [=============>................] - ETA: 45s - loss: 7.5989 - accuracy: 0.5044
12032/25000 [=============>................] - ETA: 45s - loss: 7.6016 - accuracy: 0.5042
12064/25000 [=============>................] - ETA: 45s - loss: 7.6018 - accuracy: 0.5042
12096/25000 [=============>................] - ETA: 45s - loss: 7.6032 - accuracy: 0.5041
12128/25000 [=============>................] - ETA: 45s - loss: 7.5958 - accuracy: 0.5046
12160/25000 [=============>................] - ETA: 45s - loss: 7.5973 - accuracy: 0.5045
12192/25000 [=============>................] - ETA: 45s - loss: 7.5962 - accuracy: 0.5046
12224/25000 [=============>................] - ETA: 45s - loss: 7.5926 - accuracy: 0.5048
12256/25000 [=============>................] - ETA: 45s - loss: 7.5928 - accuracy: 0.5048
12288/25000 [=============>................] - ETA: 44s - loss: 7.5905 - accuracy: 0.5050
12320/25000 [=============>................] - ETA: 44s - loss: 7.5932 - accuracy: 0.5048
12352/25000 [=============>................] - ETA: 44s - loss: 7.5909 - accuracy: 0.5049
12384/25000 [=============>................] - ETA: 44s - loss: 7.5899 - accuracy: 0.5050
12416/25000 [=============>................] - ETA: 44s - loss: 7.5888 - accuracy: 0.5051
12448/25000 [=============>................] - ETA: 44s - loss: 7.5915 - accuracy: 0.5049
12480/25000 [=============>................] - ETA: 44s - loss: 7.5880 - accuracy: 0.5051
12512/25000 [==============>...............] - ETA: 44s - loss: 7.5882 - accuracy: 0.5051
12544/25000 [==============>...............] - ETA: 44s - loss: 7.5872 - accuracy: 0.5052
12576/25000 [==============>...............] - ETA: 43s - loss: 7.5874 - accuracy: 0.5052
12608/25000 [==============>...............] - ETA: 43s - loss: 7.5864 - accuracy: 0.5052
12640/25000 [==============>...............] - ETA: 43s - loss: 7.5878 - accuracy: 0.5051
12672/25000 [==============>...............] - ETA: 43s - loss: 7.5904 - accuracy: 0.5050
12704/25000 [==============>...............] - ETA: 43s - loss: 7.5942 - accuracy: 0.5047
12736/25000 [==============>...............] - ETA: 43s - loss: 7.5968 - accuracy: 0.5046
12768/25000 [==============>...............] - ETA: 43s - loss: 7.6006 - accuracy: 0.5043
12800/25000 [==============>...............] - ETA: 43s - loss: 7.6019 - accuracy: 0.5042
12832/25000 [==============>...............] - ETA: 43s - loss: 7.6021 - accuracy: 0.5042
12864/25000 [==============>...............] - ETA: 42s - loss: 7.5963 - accuracy: 0.5046
12896/25000 [==============>...............] - ETA: 42s - loss: 7.5881 - accuracy: 0.5051
12928/25000 [==============>...............] - ETA: 42s - loss: 7.5836 - accuracy: 0.5054
12960/25000 [==============>...............] - ETA: 42s - loss: 7.5885 - accuracy: 0.5051
12992/25000 [==============>...............] - ETA: 42s - loss: 7.5828 - accuracy: 0.5055
13024/25000 [==============>...............] - ETA: 42s - loss: 7.5783 - accuracy: 0.5058
13056/25000 [==============>...............] - ETA: 42s - loss: 7.5832 - accuracy: 0.5054
13088/25000 [==============>...............] - ETA: 42s - loss: 7.5823 - accuracy: 0.5055
13120/25000 [==============>...............] - ETA: 41s - loss: 7.5766 - accuracy: 0.5059
13152/25000 [==============>...............] - ETA: 41s - loss: 7.5757 - accuracy: 0.5059
13184/25000 [==============>...............] - ETA: 41s - loss: 7.5782 - accuracy: 0.5058
13216/25000 [==============>...............] - ETA: 41s - loss: 7.5750 - accuracy: 0.5060
13248/25000 [==============>...............] - ETA: 41s - loss: 7.5752 - accuracy: 0.5060
13280/25000 [==============>...............] - ETA: 41s - loss: 7.5743 - accuracy: 0.5060
13312/25000 [==============>...............] - ETA: 41s - loss: 7.5791 - accuracy: 0.5057
13344/25000 [===============>..............] - ETA: 41s - loss: 7.5781 - accuracy: 0.5058
13376/25000 [===============>..............] - ETA: 41s - loss: 7.5772 - accuracy: 0.5058
13408/25000 [===============>..............] - ETA: 40s - loss: 7.5740 - accuracy: 0.5060
13440/25000 [===============>..............] - ETA: 40s - loss: 7.5696 - accuracy: 0.5063
13472/25000 [===============>..............] - ETA: 40s - loss: 7.5630 - accuracy: 0.5068
13504/25000 [===============>..............] - ETA: 40s - loss: 7.5633 - accuracy: 0.5067
13536/25000 [===============>..............] - ETA: 40s - loss: 7.5681 - accuracy: 0.5064
13568/25000 [===============>..............] - ETA: 40s - loss: 7.5717 - accuracy: 0.5062
13600/25000 [===============>..............] - ETA: 40s - loss: 7.5742 - accuracy: 0.5060
13632/25000 [===============>..............] - ETA: 40s - loss: 7.5721 - accuracy: 0.5062
13664/25000 [===============>..............] - ETA: 40s - loss: 7.5701 - accuracy: 0.5063
13696/25000 [===============>..............] - ETA: 39s - loss: 7.5737 - accuracy: 0.5061
13728/25000 [===============>..............] - ETA: 39s - loss: 7.5761 - accuracy: 0.5059
13760/25000 [===============>..............] - ETA: 39s - loss: 7.5819 - accuracy: 0.5055
13792/25000 [===============>..............] - ETA: 39s - loss: 7.5832 - accuracy: 0.5054
13824/25000 [===============>..............] - ETA: 39s - loss: 7.5823 - accuracy: 0.5055
13856/25000 [===============>..............] - ETA: 39s - loss: 7.5803 - accuracy: 0.5056
13888/25000 [===============>..............] - ETA: 39s - loss: 7.5849 - accuracy: 0.5053
13920/25000 [===============>..............] - ETA: 39s - loss: 7.5873 - accuracy: 0.5052
13952/25000 [===============>..............] - ETA: 39s - loss: 7.5864 - accuracy: 0.5052
13984/25000 [===============>..............] - ETA: 38s - loss: 7.5866 - accuracy: 0.5052
14016/25000 [===============>..............] - ETA: 38s - loss: 7.5879 - accuracy: 0.5051
14048/25000 [===============>..............] - ETA: 38s - loss: 7.5848 - accuracy: 0.5053
14080/25000 [===============>..............] - ETA: 38s - loss: 7.5817 - accuracy: 0.5055
14112/25000 [===============>..............] - ETA: 38s - loss: 7.5819 - accuracy: 0.5055
14144/25000 [===============>..............] - ETA: 38s - loss: 7.5810 - accuracy: 0.5056
14176/25000 [================>.............] - ETA: 38s - loss: 7.5823 - accuracy: 0.5055
14208/25000 [================>.............] - ETA: 38s - loss: 7.5857 - accuracy: 0.5053
14240/25000 [================>.............] - ETA: 38s - loss: 7.5912 - accuracy: 0.5049
14272/25000 [================>.............] - ETA: 37s - loss: 7.5871 - accuracy: 0.5052
14304/25000 [================>.............] - ETA: 37s - loss: 7.5873 - accuracy: 0.5052
14336/25000 [================>.............] - ETA: 37s - loss: 7.5885 - accuracy: 0.5051
14368/25000 [================>.............] - ETA: 37s - loss: 7.5887 - accuracy: 0.5051
14400/25000 [================>.............] - ETA: 37s - loss: 7.5921 - accuracy: 0.5049
14432/25000 [================>.............] - ETA: 37s - loss: 7.5944 - accuracy: 0.5047
14464/25000 [================>.............] - ETA: 37s - loss: 7.5914 - accuracy: 0.5049
14496/25000 [================>.............] - ETA: 37s - loss: 7.5936 - accuracy: 0.5048
14528/25000 [================>.............] - ETA: 37s - loss: 7.5949 - accuracy: 0.5047
14560/25000 [================>.............] - ETA: 36s - loss: 7.5950 - accuracy: 0.5047
14592/25000 [================>.............] - ETA: 36s - loss: 7.5920 - accuracy: 0.5049
14624/25000 [================>.............] - ETA: 36s - loss: 7.5953 - accuracy: 0.5046
14656/25000 [================>.............] - ETA: 36s - loss: 7.5986 - accuracy: 0.5044
14688/25000 [================>.............] - ETA: 36s - loss: 7.5956 - accuracy: 0.5046
14720/25000 [================>.............] - ETA: 36s - loss: 7.5968 - accuracy: 0.5046
14752/25000 [================>.............] - ETA: 36s - loss: 7.5970 - accuracy: 0.5045
14784/25000 [================>.............] - ETA: 36s - loss: 7.6013 - accuracy: 0.5043
14816/25000 [================>.............] - ETA: 35s - loss: 7.5994 - accuracy: 0.5044
14848/25000 [================>.............] - ETA: 35s - loss: 7.5995 - accuracy: 0.5044
14880/25000 [================>.............] - ETA: 35s - loss: 7.6038 - accuracy: 0.5041
14912/25000 [================>.............] - ETA: 35s - loss: 7.6039 - accuracy: 0.5041
14944/25000 [================>.............] - ETA: 35s - loss: 7.6061 - accuracy: 0.5039
14976/25000 [================>.............] - ETA: 35s - loss: 7.6031 - accuracy: 0.5041
15008/25000 [=================>............] - ETA: 35s - loss: 7.6002 - accuracy: 0.5043
15040/25000 [=================>............] - ETA: 35s - loss: 7.5993 - accuracy: 0.5044
15072/25000 [=================>............] - ETA: 35s - loss: 7.6025 - accuracy: 0.5042
15104/25000 [=================>............] - ETA: 34s - loss: 7.6057 - accuracy: 0.5040
15136/25000 [=================>............] - ETA: 34s - loss: 7.6119 - accuracy: 0.5036
15168/25000 [=================>............] - ETA: 34s - loss: 7.6141 - accuracy: 0.5034
15200/25000 [=================>............] - ETA: 34s - loss: 7.6111 - accuracy: 0.5036
15232/25000 [=================>............] - ETA: 34s - loss: 7.6092 - accuracy: 0.5037
15264/25000 [=================>............] - ETA: 34s - loss: 7.6154 - accuracy: 0.5033
15296/25000 [=================>............] - ETA: 34s - loss: 7.6155 - accuracy: 0.5033
15328/25000 [=================>............] - ETA: 34s - loss: 7.6146 - accuracy: 0.5034
15360/25000 [=================>............] - ETA: 34s - loss: 7.6127 - accuracy: 0.5035
15392/25000 [=================>............] - ETA: 33s - loss: 7.6148 - accuracy: 0.5034
15424/25000 [=================>............] - ETA: 33s - loss: 7.6179 - accuracy: 0.5032
15456/25000 [=================>............] - ETA: 33s - loss: 7.6190 - accuracy: 0.5031
15488/25000 [=================>............] - ETA: 33s - loss: 7.6201 - accuracy: 0.5030
15520/25000 [=================>............] - ETA: 33s - loss: 7.6182 - accuracy: 0.5032
15552/25000 [=================>............] - ETA: 33s - loss: 7.6203 - accuracy: 0.5030
15584/25000 [=================>............] - ETA: 33s - loss: 7.6174 - accuracy: 0.5032
15616/25000 [=================>............] - ETA: 33s - loss: 7.6224 - accuracy: 0.5029
15648/25000 [=================>............] - ETA: 33s - loss: 7.6215 - accuracy: 0.5029
15680/25000 [=================>............] - ETA: 32s - loss: 7.6246 - accuracy: 0.5027
15712/25000 [=================>............] - ETA: 32s - loss: 7.6247 - accuracy: 0.5027
15744/25000 [=================>............] - ETA: 32s - loss: 7.6228 - accuracy: 0.5029
15776/25000 [=================>............] - ETA: 32s - loss: 7.6277 - accuracy: 0.5025
15808/25000 [=================>............] - ETA: 32s - loss: 7.6327 - accuracy: 0.5022
15840/25000 [==================>...........] - ETA: 32s - loss: 7.6356 - accuracy: 0.5020
15872/25000 [==================>...........] - ETA: 32s - loss: 7.6328 - accuracy: 0.5022
15904/25000 [==================>...........] - ETA: 32s - loss: 7.6242 - accuracy: 0.5028
15936/25000 [==================>...........] - ETA: 32s - loss: 7.6272 - accuracy: 0.5026
15968/25000 [==================>...........] - ETA: 31s - loss: 7.6301 - accuracy: 0.5024
16000/25000 [==================>...........] - ETA: 31s - loss: 7.6283 - accuracy: 0.5025
16032/25000 [==================>...........] - ETA: 31s - loss: 7.6312 - accuracy: 0.5023
16064/25000 [==================>...........] - ETA: 31s - loss: 7.6351 - accuracy: 0.5021
16096/25000 [==================>...........] - ETA: 31s - loss: 7.6323 - accuracy: 0.5022
16128/25000 [==================>...........] - ETA: 31s - loss: 7.6333 - accuracy: 0.5022
16160/25000 [==================>...........] - ETA: 31s - loss: 7.6296 - accuracy: 0.5024
16192/25000 [==================>...........] - ETA: 31s - loss: 7.6250 - accuracy: 0.5027
16224/25000 [==================>...........] - ETA: 31s - loss: 7.6260 - accuracy: 0.5027
16256/25000 [==================>...........] - ETA: 30s - loss: 7.6251 - accuracy: 0.5027
16288/25000 [==================>...........] - ETA: 30s - loss: 7.6233 - accuracy: 0.5028
16320/25000 [==================>...........] - ETA: 30s - loss: 7.6243 - accuracy: 0.5028
16352/25000 [==================>...........] - ETA: 30s - loss: 7.6272 - accuracy: 0.5026
16384/25000 [==================>...........] - ETA: 30s - loss: 7.6292 - accuracy: 0.5024
16416/25000 [==================>...........] - ETA: 30s - loss: 7.6255 - accuracy: 0.5027
16448/25000 [==================>...........] - ETA: 30s - loss: 7.6256 - accuracy: 0.5027
16480/25000 [==================>...........] - ETA: 30s - loss: 7.6266 - accuracy: 0.5026
16512/25000 [==================>...........] - ETA: 29s - loss: 7.6276 - accuracy: 0.5025
16544/25000 [==================>...........] - ETA: 29s - loss: 7.6295 - accuracy: 0.5024
16576/25000 [==================>...........] - ETA: 29s - loss: 7.6287 - accuracy: 0.5025
16608/25000 [==================>...........] - ETA: 29s - loss: 7.6278 - accuracy: 0.5025
16640/25000 [==================>...........] - ETA: 29s - loss: 7.6316 - accuracy: 0.5023
16672/25000 [===================>..........] - ETA: 29s - loss: 7.6308 - accuracy: 0.5023
16704/25000 [===================>..........] - ETA: 29s - loss: 7.6271 - accuracy: 0.5026
16736/25000 [===================>..........] - ETA: 29s - loss: 7.6281 - accuracy: 0.5025
16768/25000 [===================>..........] - ETA: 29s - loss: 7.6255 - accuracy: 0.5027
16800/25000 [===================>..........] - ETA: 28s - loss: 7.6265 - accuracy: 0.5026
16832/25000 [===================>..........] - ETA: 28s - loss: 7.6293 - accuracy: 0.5024
16864/25000 [===================>..........] - ETA: 28s - loss: 7.6321 - accuracy: 0.5023
16896/25000 [===================>..........] - ETA: 28s - loss: 7.6285 - accuracy: 0.5025
16928/25000 [===================>..........] - ETA: 28s - loss: 7.6304 - accuracy: 0.5024
16960/25000 [===================>..........] - ETA: 28s - loss: 7.6296 - accuracy: 0.5024
16992/25000 [===================>..........] - ETA: 28s - loss: 7.6278 - accuracy: 0.5025
17024/25000 [===================>..........] - ETA: 28s - loss: 7.6279 - accuracy: 0.5025
17056/25000 [===================>..........] - ETA: 28s - loss: 7.6289 - accuracy: 0.5025
17088/25000 [===================>..........] - ETA: 27s - loss: 7.6253 - accuracy: 0.5027
17120/25000 [===================>..........] - ETA: 27s - loss: 7.6245 - accuracy: 0.5027
17152/25000 [===================>..........] - ETA: 27s - loss: 7.6255 - accuracy: 0.5027
17184/25000 [===================>..........] - ETA: 27s - loss: 7.6238 - accuracy: 0.5028
17216/25000 [===================>..........] - ETA: 27s - loss: 7.6257 - accuracy: 0.5027
17248/25000 [===================>..........] - ETA: 27s - loss: 7.6248 - accuracy: 0.5027
17280/25000 [===================>..........] - ETA: 27s - loss: 7.6240 - accuracy: 0.5028
17312/25000 [===================>..........] - ETA: 27s - loss: 7.6188 - accuracy: 0.5031
17344/25000 [===================>..........] - ETA: 27s - loss: 7.6206 - accuracy: 0.5030
17376/25000 [===================>..........] - ETA: 26s - loss: 7.6199 - accuracy: 0.5031
17408/25000 [===================>..........] - ETA: 26s - loss: 7.6226 - accuracy: 0.5029
17440/25000 [===================>..........] - ETA: 26s - loss: 7.6200 - accuracy: 0.5030
17472/25000 [===================>..........] - ETA: 26s - loss: 7.6254 - accuracy: 0.5027
17504/25000 [====================>.........] - ETA: 26s - loss: 7.6272 - accuracy: 0.5026
17536/25000 [====================>.........] - ETA: 26s - loss: 7.6290 - accuracy: 0.5025
17568/25000 [====================>.........] - ETA: 26s - loss: 7.6265 - accuracy: 0.5026
17600/25000 [====================>.........] - ETA: 26s - loss: 7.6248 - accuracy: 0.5027
17632/25000 [====================>.........] - ETA: 26s - loss: 7.6240 - accuracy: 0.5028
17664/25000 [====================>.........] - ETA: 25s - loss: 7.6232 - accuracy: 0.5028
17696/25000 [====================>.........] - ETA: 25s - loss: 7.6216 - accuracy: 0.5029
17728/25000 [====================>.........] - ETA: 25s - loss: 7.6190 - accuracy: 0.5031
17760/25000 [====================>.........] - ETA: 25s - loss: 7.6174 - accuracy: 0.5032
17792/25000 [====================>.........] - ETA: 25s - loss: 7.6140 - accuracy: 0.5034
17824/25000 [====================>.........] - ETA: 25s - loss: 7.6150 - accuracy: 0.5034
17856/25000 [====================>.........] - ETA: 25s - loss: 7.6168 - accuracy: 0.5032
17888/25000 [====================>.........] - ETA: 25s - loss: 7.6160 - accuracy: 0.5033
17920/25000 [====================>.........] - ETA: 25s - loss: 7.6144 - accuracy: 0.5034
17952/25000 [====================>.........] - ETA: 24s - loss: 7.6137 - accuracy: 0.5035
17984/25000 [====================>.........] - ETA: 24s - loss: 7.6112 - accuracy: 0.5036
18016/25000 [====================>.........] - ETA: 24s - loss: 7.6079 - accuracy: 0.5038
18048/25000 [====================>.........] - ETA: 24s - loss: 7.6080 - accuracy: 0.5038
18080/25000 [====================>.........] - ETA: 24s - loss: 7.6123 - accuracy: 0.5035
18112/25000 [====================>.........] - ETA: 24s - loss: 7.6107 - accuracy: 0.5036
18144/25000 [====================>.........] - ETA: 24s - loss: 7.6100 - accuracy: 0.5037
18176/25000 [====================>.........] - ETA: 24s - loss: 7.6093 - accuracy: 0.5037
18208/25000 [====================>.........] - ETA: 23s - loss: 7.6127 - accuracy: 0.5035
18240/25000 [====================>.........] - ETA: 23s - loss: 7.6145 - accuracy: 0.5034
18272/25000 [====================>.........] - ETA: 23s - loss: 7.6146 - accuracy: 0.5034
18304/25000 [====================>.........] - ETA: 23s - loss: 7.6155 - accuracy: 0.5033
18336/25000 [=====================>........] - ETA: 23s - loss: 7.6131 - accuracy: 0.5035
18368/25000 [=====================>........] - ETA: 23s - loss: 7.6140 - accuracy: 0.5034
18400/25000 [=====================>........] - ETA: 23s - loss: 7.6116 - accuracy: 0.5036
18432/25000 [=====================>........] - ETA: 23s - loss: 7.6109 - accuracy: 0.5036
18464/25000 [=====================>........] - ETA: 23s - loss: 7.6126 - accuracy: 0.5035
18496/25000 [=====================>........] - ETA: 22s - loss: 7.6169 - accuracy: 0.5032
18528/25000 [=====================>........] - ETA: 22s - loss: 7.6186 - accuracy: 0.5031
18560/25000 [=====================>........] - ETA: 22s - loss: 7.6195 - accuracy: 0.5031
18592/25000 [=====================>........] - ETA: 22s - loss: 7.6180 - accuracy: 0.5032
18624/25000 [=====================>........] - ETA: 22s - loss: 7.6205 - accuracy: 0.5030
18656/25000 [=====================>........] - ETA: 22s - loss: 7.6206 - accuracy: 0.5030
18688/25000 [=====================>........] - ETA: 22s - loss: 7.6215 - accuracy: 0.5029
18720/25000 [=====================>........] - ETA: 22s - loss: 7.6208 - accuracy: 0.5030
18752/25000 [=====================>........] - ETA: 22s - loss: 7.6200 - accuracy: 0.5030
18784/25000 [=====================>........] - ETA: 21s - loss: 7.6201 - accuracy: 0.5030
18816/25000 [=====================>........] - ETA: 21s - loss: 7.6185 - accuracy: 0.5031
18848/25000 [=====================>........] - ETA: 21s - loss: 7.6170 - accuracy: 0.5032
18880/25000 [=====================>........] - ETA: 21s - loss: 7.6179 - accuracy: 0.5032
18912/25000 [=====================>........] - ETA: 21s - loss: 7.6180 - accuracy: 0.5032
18944/25000 [=====================>........] - ETA: 21s - loss: 7.6132 - accuracy: 0.5035
18976/25000 [=====================>........] - ETA: 21s - loss: 7.6173 - accuracy: 0.5032
19008/25000 [=====================>........] - ETA: 21s - loss: 7.6142 - accuracy: 0.5034
19040/25000 [=====================>........] - ETA: 21s - loss: 7.6151 - accuracy: 0.5034
19072/25000 [=====================>........] - ETA: 20s - loss: 7.6128 - accuracy: 0.5035
19104/25000 [=====================>........] - ETA: 20s - loss: 7.6144 - accuracy: 0.5034
19136/25000 [=====================>........] - ETA: 20s - loss: 7.6129 - accuracy: 0.5035
19168/25000 [======================>.......] - ETA: 20s - loss: 7.6138 - accuracy: 0.5034
19200/25000 [======================>.......] - ETA: 20s - loss: 7.6147 - accuracy: 0.5034
19232/25000 [======================>.......] - ETA: 20s - loss: 7.6156 - accuracy: 0.5033
19264/25000 [======================>.......] - ETA: 20s - loss: 7.6165 - accuracy: 0.5033
19296/25000 [======================>.......] - ETA: 20s - loss: 7.6197 - accuracy: 0.5031
19328/25000 [======================>.......] - ETA: 20s - loss: 7.6230 - accuracy: 0.5028
19360/25000 [======================>.......] - ETA: 19s - loss: 7.6239 - accuracy: 0.5028
19392/25000 [======================>.......] - ETA: 19s - loss: 7.6263 - accuracy: 0.5026
19424/25000 [======================>.......] - ETA: 19s - loss: 7.6279 - accuracy: 0.5025
19456/25000 [======================>.......] - ETA: 19s - loss: 7.6280 - accuracy: 0.5025
19488/25000 [======================>.......] - ETA: 19s - loss: 7.6296 - accuracy: 0.5024
19520/25000 [======================>.......] - ETA: 19s - loss: 7.6297 - accuracy: 0.5024
19552/25000 [======================>.......] - ETA: 19s - loss: 7.6298 - accuracy: 0.5024
19584/25000 [======================>.......] - ETA: 19s - loss: 7.6314 - accuracy: 0.5023
19616/25000 [======================>.......] - ETA: 19s - loss: 7.6322 - accuracy: 0.5022
19648/25000 [======================>.......] - ETA: 18s - loss: 7.6385 - accuracy: 0.5018
19680/25000 [======================>.......] - ETA: 18s - loss: 7.6425 - accuracy: 0.5016
19712/25000 [======================>.......] - ETA: 18s - loss: 7.6402 - accuracy: 0.5017
19744/25000 [======================>.......] - ETA: 18s - loss: 7.6402 - accuracy: 0.5017
19776/25000 [======================>.......] - ETA: 18s - loss: 7.6379 - accuracy: 0.5019
19808/25000 [======================>.......] - ETA: 18s - loss: 7.6364 - accuracy: 0.5020
19840/25000 [======================>.......] - ETA: 18s - loss: 7.6388 - accuracy: 0.5018
19872/25000 [======================>.......] - ETA: 18s - loss: 7.6412 - accuracy: 0.5017
19904/25000 [======================>.......] - ETA: 17s - loss: 7.6397 - accuracy: 0.5018
19936/25000 [======================>.......] - ETA: 17s - loss: 7.6428 - accuracy: 0.5016
19968/25000 [======================>.......] - ETA: 17s - loss: 7.6413 - accuracy: 0.5017
20000/25000 [=======================>......] - ETA: 17s - loss: 7.6421 - accuracy: 0.5016
20032/25000 [=======================>......] - ETA: 17s - loss: 7.6414 - accuracy: 0.5016
20064/25000 [=======================>......] - ETA: 17s - loss: 7.6445 - accuracy: 0.5014
20096/25000 [=======================>......] - ETA: 17s - loss: 7.6437 - accuracy: 0.5015
20128/25000 [=======================>......] - ETA: 17s - loss: 7.6422 - accuracy: 0.5016
20160/25000 [=======================>......] - ETA: 17s - loss: 7.6430 - accuracy: 0.5015
20192/25000 [=======================>......] - ETA: 16s - loss: 7.6461 - accuracy: 0.5013
20224/25000 [=======================>......] - ETA: 16s - loss: 7.6461 - accuracy: 0.5013
20256/25000 [=======================>......] - ETA: 16s - loss: 7.6485 - accuracy: 0.5012
20288/25000 [=======================>......] - ETA: 16s - loss: 7.6507 - accuracy: 0.5010
20320/25000 [=======================>......] - ETA: 16s - loss: 7.6478 - accuracy: 0.5012
20352/25000 [=======================>......] - ETA: 16s - loss: 7.6500 - accuracy: 0.5011
20384/25000 [=======================>......] - ETA: 16s - loss: 7.6531 - accuracy: 0.5009
20416/25000 [=======================>......] - ETA: 16s - loss: 7.6523 - accuracy: 0.5009
20448/25000 [=======================>......] - ETA: 16s - loss: 7.6539 - accuracy: 0.5008
20480/25000 [=======================>......] - ETA: 15s - loss: 7.6554 - accuracy: 0.5007
20512/25000 [=======================>......] - ETA: 15s - loss: 7.6554 - accuracy: 0.5007
20544/25000 [=======================>......] - ETA: 15s - loss: 7.6562 - accuracy: 0.5007
20576/25000 [=======================>......] - ETA: 15s - loss: 7.6532 - accuracy: 0.5009
20608/25000 [=======================>......] - ETA: 15s - loss: 7.6525 - accuracy: 0.5009
20640/25000 [=======================>......] - ETA: 15s - loss: 7.6532 - accuracy: 0.5009
20672/25000 [=======================>......] - ETA: 15s - loss: 7.6548 - accuracy: 0.5008
20704/25000 [=======================>......] - ETA: 15s - loss: 7.6563 - accuracy: 0.5007
20736/25000 [=======================>......] - ETA: 15s - loss: 7.6533 - accuracy: 0.5009
20768/25000 [=======================>......] - ETA: 14s - loss: 7.6511 - accuracy: 0.5010
20800/25000 [=======================>......] - ETA: 14s - loss: 7.6511 - accuracy: 0.5010
20832/25000 [=======================>......] - ETA: 14s - loss: 7.6512 - accuracy: 0.5010
20864/25000 [========================>.....] - ETA: 14s - loss: 7.6541 - accuracy: 0.5008
20896/25000 [========================>.....] - ETA: 14s - loss: 7.6534 - accuracy: 0.5009
20928/25000 [========================>.....] - ETA: 14s - loss: 7.6549 - accuracy: 0.5008
20960/25000 [========================>.....] - ETA: 14s - loss: 7.6571 - accuracy: 0.5006
20992/25000 [========================>.....] - ETA: 14s - loss: 7.6579 - accuracy: 0.5006
21024/25000 [========================>.....] - ETA: 14s - loss: 7.6550 - accuracy: 0.5008
21056/25000 [========================>.....] - ETA: 13s - loss: 7.6586 - accuracy: 0.5005
21088/25000 [========================>.....] - ETA: 13s - loss: 7.6564 - accuracy: 0.5007
21120/25000 [========================>.....] - ETA: 13s - loss: 7.6550 - accuracy: 0.5008
21152/25000 [========================>.....] - ETA: 13s - loss: 7.6550 - accuracy: 0.5008
21184/25000 [========================>.....] - ETA: 13s - loss: 7.6558 - accuracy: 0.5007
21216/25000 [========================>.....] - ETA: 13s - loss: 7.6529 - accuracy: 0.5009
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6522 - accuracy: 0.5009
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6493 - accuracy: 0.5011
21312/25000 [========================>.....] - ETA: 13s - loss: 7.6494 - accuracy: 0.5011
21344/25000 [========================>.....] - ETA: 12s - loss: 7.6494 - accuracy: 0.5011
21376/25000 [========================>.....] - ETA: 12s - loss: 7.6501 - accuracy: 0.5011
21408/25000 [========================>.....] - ETA: 12s - loss: 7.6537 - accuracy: 0.5008
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6545 - accuracy: 0.5008
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6538 - accuracy: 0.5008
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6552 - accuracy: 0.5007
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6574 - accuracy: 0.5006
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6616 - accuracy: 0.5003
21600/25000 [========================>.....] - ETA: 12s - loss: 7.6609 - accuracy: 0.5004
21632/25000 [========================>.....] - ETA: 11s - loss: 7.6595 - accuracy: 0.5005
21664/25000 [========================>.....] - ETA: 11s - loss: 7.6602 - accuracy: 0.5004
21696/25000 [=========================>....] - ETA: 11s - loss: 7.6588 - accuracy: 0.5005
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6596 - accuracy: 0.5005
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6624 - accuracy: 0.5003
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6624 - accuracy: 0.5003
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6610 - accuracy: 0.5004
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6610 - accuracy: 0.5004
21888/25000 [=========================>....] - ETA: 10s - loss: 7.6624 - accuracy: 0.5003
21920/25000 [=========================>....] - ETA: 10s - loss: 7.6603 - accuracy: 0.5004
21952/25000 [=========================>....] - ETA: 10s - loss: 7.6610 - accuracy: 0.5004
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6589 - accuracy: 0.5005
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6576 - accuracy: 0.5006
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6576 - accuracy: 0.5006
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6583 - accuracy: 0.5005
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6590 - accuracy: 0.5005
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6597 - accuracy: 0.5005
22176/25000 [=========================>....] - ETA: 9s - loss: 7.6611 - accuracy: 0.5004 
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6625 - accuracy: 0.5003
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6618 - accuracy: 0.5003
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6604 - accuracy: 0.5004
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6591 - accuracy: 0.5005
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6604 - accuracy: 0.5004
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6611 - accuracy: 0.5004
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6625 - accuracy: 0.5003
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6598 - accuracy: 0.5004
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6632 - accuracy: 0.5002
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6618 - accuracy: 0.5003
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6639 - accuracy: 0.5002
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6646 - accuracy: 0.5001
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6632 - accuracy: 0.5002
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6592 - accuracy: 0.5005
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6612 - accuracy: 0.5004
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6612 - accuracy: 0.5004
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6592 - accuracy: 0.5005
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6585 - accuracy: 0.5005
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6599 - accuracy: 0.5004
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6612 - accuracy: 0.5004
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6592 - accuracy: 0.5005
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6626 - accuracy: 0.5003
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6626 - accuracy: 0.5003
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6639 - accuracy: 0.5002
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6646 - accuracy: 0.5001
23008/25000 [==========================>...] - ETA: 7s - loss: 7.6620 - accuracy: 0.5003
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6593 - accuracy: 0.5005
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6580 - accuracy: 0.5006
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6600 - accuracy: 0.5004
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6567 - accuracy: 0.5006
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6580 - accuracy: 0.5006
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6613 - accuracy: 0.5003
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6620 - accuracy: 0.5003
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6613 - accuracy: 0.5003
23296/25000 [==========================>...] - ETA: 6s - loss: 7.6627 - accuracy: 0.5003
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6627 - accuracy: 0.5003
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6640 - accuracy: 0.5002
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6627 - accuracy: 0.5003
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6620 - accuracy: 0.5003
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6620 - accuracy: 0.5003
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6647 - accuracy: 0.5001
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6653 - accuracy: 0.5001
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6640 - accuracy: 0.5002
23584/25000 [===========================>..] - ETA: 5s - loss: 7.6627 - accuracy: 0.5003
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6614 - accuracy: 0.5003
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6621 - accuracy: 0.5003
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6634 - accuracy: 0.5002
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6698 - accuracy: 0.4998
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6660 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6628 - accuracy: 0.5002
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6641 - accuracy: 0.5002
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24192/25000 [============================>.] - ETA: 2s - loss: 7.6609 - accuracy: 0.5004
24224/25000 [============================>.] - ETA: 2s - loss: 7.6609 - accuracy: 0.5004
24256/25000 [============================>.] - ETA: 2s - loss: 7.6635 - accuracy: 0.5002
24288/25000 [============================>.] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24320/25000 [============================>.] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
24352/25000 [============================>.] - ETA: 2s - loss: 7.6691 - accuracy: 0.4998
24384/25000 [============================>.] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24416/25000 [============================>.] - ETA: 2s - loss: 7.6672 - accuracy: 0.5000
24448/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24480/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24512/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24544/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24576/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24608/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24640/25000 [============================>.] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
24672/25000 [============================>.] - ETA: 1s - loss: 7.6703 - accuracy: 0.4998
24704/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24736/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24768/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24800/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24832/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24864/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24896/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24928/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
24960/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24992/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f4e76a7aa90> 

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
 [-0.06040791 -0.10146535  0.05892891  0.25840926  0.1073321   0.03800261]
 [ 0.29183483  0.26979324  0.24089599  0.18119404  0.12721241  0.00876994]
 [-0.15238358 -0.00742447 -0.15840846  0.3448984  -0.38144329  0.11227039]
 [-0.27981383 -0.08227225  0.08479659  0.4102636  -0.41063914  0.21904153]
 [ 0.27319622 -0.07297604  0.32164901 -0.04872991  0.3543314   0.67826343]
 [-0.01138831  0.04326198  0.19296314  0.413982   -0.14042771  0.15866084]
 [ 0.28764331 -0.32970452  0.00683084  0.20910314 -0.05704655  0.74323922]
 [ 0.33138621 -0.40753537  0.09830459  0.17700157  0.07091133  0.25510243]
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
{'loss': 0.4720406234264374, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-12 00:23:36.994834: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 523, in main
    test_cli(arg)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 455, in test_cli
    test(arg.model_uri)  # '1_lstm'
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 189, in test
    module.test()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
    saver      = tf.compat.v1.train.Saver()
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
{'loss': 0.5020723938941956, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-12 00:23:38.247221: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 523, in main
    test_cli(arg)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 457, in test_cli
    test_global(arg.model_uri)  # '1_lstm'
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 200, in test_global
    module.test()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
    saver      = tf.compat.v1.train.Saver()
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
 40%|████      | 2/5 [00:56<01:25, 28.42s/it] 40%|████      | 2/5 [00:56<01:25, 28.42s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.0976397033645913, 'embedding_size_factor': 0.6753526152652608, 'layers.choice': 1, 'learning_rate': 0.0002277848158238269, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.2244248548226617e-08} and reward: 0.3598
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xb8\xfe\xead\xbd\xf7\xb2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5\x9c}\x16z\xa2\xbdX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?-\xdb0\xabB\x1b\x01X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>JKX\xc9\x98\x03Xu.' and reward: 0.3598
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xb8\xfe\xead\xbd\xf7\xb2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5\x9c}\x16z\xa2\xbdX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?-\xdb0\xabB\x1b\x01X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>JKX\xc9\x98\x03Xu.' and reward: 0.3598
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 144.59403443336487
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.73s of the -26.63s of remaining time.
Ensemble size: 25
Ensemble weights: 
[0.64 0.36]
	0.3894	 = Validation accuracy score
	0.99s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 147.67s ...
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 523, in main
    test_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 453, in test_cli
    test_module(arg.model_uri, param_pars=param_pars)  # '1_lstm'
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 257, in test_module
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
[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     87[0m [0;34m[0m[0m
[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     90[0m [0;34m[0m[0m
[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range





 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     87[0m [0;34m[0m[0m
[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     90[0m [0;34m[0m[0m
[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

