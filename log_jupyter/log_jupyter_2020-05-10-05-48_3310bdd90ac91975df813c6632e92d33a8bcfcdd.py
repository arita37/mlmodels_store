
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_jupyter GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/3310bdd90ac91975df813c6632e92d33a8bcfcdd', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '3310bdd90ac91975df813c6632e92d33a8bcfcdd', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/3310bdd90ac91975df813c6632e92d33a8bcfcdd

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/3310bdd90ac91975df813c6632e92d33a8bcfcdd

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
 2703360/17464789 [===>..........................] - ETA: 0s
11419648/17464789 [==================>...........] - ETA: 0s
16334848/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 05:48:25.235609: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 05:48:25.240119: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095179999 Hz
2020-05-10 05:48:25.240317: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ddec350630 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 05:48:25.240332: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:18 - loss: 6.2291 - accuracy: 0.5938
   64/25000 [..............................] - ETA: 2:40 - loss: 6.9479 - accuracy: 0.5469
   96/25000 [..............................] - ETA: 2:07 - loss: 6.8680 - accuracy: 0.5521
  128/25000 [..............................] - ETA: 1:50 - loss: 6.9479 - accuracy: 0.5469
  160/25000 [..............................] - ETA: 1:39 - loss: 7.5708 - accuracy: 0.5063
  192/25000 [..............................] - ETA: 1:33 - loss: 7.6666 - accuracy: 0.5000
  224/25000 [..............................] - ETA: 1:27 - loss: 7.8035 - accuracy: 0.4911
  256/25000 [..............................] - ETA: 1:23 - loss: 7.6067 - accuracy: 0.5039
  288/25000 [..............................] - ETA: 1:20 - loss: 7.8263 - accuracy: 0.4896
  320/25000 [..............................] - ETA: 1:19 - loss: 7.7625 - accuracy: 0.4938
  352/25000 [..............................] - ETA: 1:18 - loss: 7.9715 - accuracy: 0.4801
  384/25000 [..............................] - ETA: 1:17 - loss: 7.9062 - accuracy: 0.4844
  416/25000 [..............................] - ETA: 1:16 - loss: 7.9246 - accuracy: 0.4832
  448/25000 [..............................] - ETA: 1:14 - loss: 8.0431 - accuracy: 0.4754
  480/25000 [..............................] - ETA: 1:14 - loss: 7.9861 - accuracy: 0.4792
  512/25000 [..............................] - ETA: 1:12 - loss: 8.1458 - accuracy: 0.4688
  544/25000 [..............................] - ETA: 1:12 - loss: 8.1176 - accuracy: 0.4706
  576/25000 [..............................] - ETA: 1:11 - loss: 8.2256 - accuracy: 0.4635
  608/25000 [..............................] - ETA: 1:11 - loss: 8.2467 - accuracy: 0.4622
  640/25000 [..............................] - ETA: 1:10 - loss: 8.1697 - accuracy: 0.4672
  672/25000 [..............................] - ETA: 1:09 - loss: 8.1230 - accuracy: 0.4702
  704/25000 [..............................] - ETA: 1:08 - loss: 7.9715 - accuracy: 0.4801
  736/25000 [..............................] - ETA: 1:07 - loss: 7.9583 - accuracy: 0.4810
  768/25000 [..............................] - ETA: 1:07 - loss: 7.9262 - accuracy: 0.4831
  800/25000 [..............................] - ETA: 1:06 - loss: 7.9350 - accuracy: 0.4825
  832/25000 [..............................] - ETA: 1:06 - loss: 8.0536 - accuracy: 0.4748
  864/25000 [>.............................] - ETA: 1:05 - loss: 8.0216 - accuracy: 0.4769
  896/25000 [>.............................] - ETA: 1:05 - loss: 8.0431 - accuracy: 0.4754
  928/25000 [>.............................] - ETA: 1:04 - loss: 8.0797 - accuracy: 0.4731
  960/25000 [>.............................] - ETA: 1:04 - loss: 8.1458 - accuracy: 0.4688
  992/25000 [>.............................] - ETA: 1:04 - loss: 8.2076 - accuracy: 0.4647
 1024/25000 [>.............................] - ETA: 1:03 - loss: 8.2207 - accuracy: 0.4639
 1056/25000 [>.............................] - ETA: 1:03 - loss: 8.1167 - accuracy: 0.4706
 1088/25000 [>.............................] - ETA: 1:03 - loss: 8.1176 - accuracy: 0.4706
 1120/25000 [>.............................] - ETA: 1:02 - loss: 8.0363 - accuracy: 0.4759
 1152/25000 [>.............................] - ETA: 1:02 - loss: 8.0526 - accuracy: 0.4748
 1184/25000 [>.............................] - ETA: 1:02 - loss: 8.0422 - accuracy: 0.4755
 1216/25000 [>.............................] - ETA: 1:02 - loss: 7.9819 - accuracy: 0.4794
 1248/25000 [>.............................] - ETA: 1:02 - loss: 7.9615 - accuracy: 0.4808
 1280/25000 [>.............................] - ETA: 1:02 - loss: 7.9661 - accuracy: 0.4805
 1312/25000 [>.............................] - ETA: 1:01 - loss: 8.0289 - accuracy: 0.4764
 1344/25000 [>.............................] - ETA: 1:01 - loss: 8.0089 - accuracy: 0.4777
 1376/25000 [>.............................] - ETA: 1:01 - loss: 8.0009 - accuracy: 0.4782
 1408/25000 [>.............................] - ETA: 1:01 - loss: 7.9389 - accuracy: 0.4822
 1440/25000 [>.............................] - ETA: 1:01 - loss: 7.9648 - accuracy: 0.4806
 1472/25000 [>.............................] - ETA: 1:00 - loss: 7.9583 - accuracy: 0.4810
 1504/25000 [>.............................] - ETA: 1:00 - loss: 7.9011 - accuracy: 0.4847
 1536/25000 [>.............................] - ETA: 1:00 - loss: 7.9162 - accuracy: 0.4837
 1568/25000 [>.............................] - ETA: 1:00 - loss: 7.8915 - accuracy: 0.4853
 1600/25000 [>.............................] - ETA: 1:00 - loss: 7.9445 - accuracy: 0.4819
 1632/25000 [>.............................] - ETA: 1:00 - loss: 7.9203 - accuracy: 0.4835
 1664/25000 [>.............................] - ETA: 59s - loss: 7.8970 - accuracy: 0.4850 
 1696/25000 [=>............................] - ETA: 59s - loss: 7.8926 - accuracy: 0.4853
 1728/25000 [=>............................] - ETA: 59s - loss: 7.9151 - accuracy: 0.4838
 1760/25000 [=>............................] - ETA: 59s - loss: 7.8844 - accuracy: 0.4858
 1792/25000 [=>............................] - ETA: 59s - loss: 7.9148 - accuracy: 0.4838
 1824/25000 [=>............................] - ETA: 59s - loss: 7.8684 - accuracy: 0.4868
 1856/25000 [=>............................] - ETA: 58s - loss: 7.8401 - accuracy: 0.4887
 1888/25000 [=>............................] - ETA: 58s - loss: 7.8615 - accuracy: 0.4873
 1920/25000 [=>............................] - ETA: 58s - loss: 7.8743 - accuracy: 0.4865
 1952/25000 [=>............................] - ETA: 58s - loss: 7.8630 - accuracy: 0.4872
 1984/25000 [=>............................] - ETA: 58s - loss: 7.8521 - accuracy: 0.4879
 2016/25000 [=>............................] - ETA: 58s - loss: 7.8339 - accuracy: 0.4891
 2048/25000 [=>............................] - ETA: 58s - loss: 7.8463 - accuracy: 0.4883
 2080/25000 [=>............................] - ETA: 58s - loss: 7.8804 - accuracy: 0.4861
 2112/25000 [=>............................] - ETA: 57s - loss: 7.8844 - accuracy: 0.4858
 2144/25000 [=>............................] - ETA: 57s - loss: 7.8597 - accuracy: 0.4874
 2176/25000 [=>............................] - ETA: 57s - loss: 7.8569 - accuracy: 0.4876
 2208/25000 [=>............................] - ETA: 57s - loss: 7.8333 - accuracy: 0.4891
 2240/25000 [=>............................] - ETA: 57s - loss: 7.8514 - accuracy: 0.4879
 2272/25000 [=>............................] - ETA: 57s - loss: 7.8691 - accuracy: 0.4868
 2304/25000 [=>............................] - ETA: 57s - loss: 7.8596 - accuracy: 0.4874
 2336/25000 [=>............................] - ETA: 56s - loss: 7.8307 - accuracy: 0.4893
 2368/25000 [=>............................] - ETA: 56s - loss: 7.8220 - accuracy: 0.4899
 2400/25000 [=>............................] - ETA: 56s - loss: 7.8200 - accuracy: 0.4900
 2432/25000 [=>............................] - ETA: 56s - loss: 7.8621 - accuracy: 0.4873
 2464/25000 [=>............................] - ETA: 56s - loss: 7.8595 - accuracy: 0.4874
 2496/25000 [=>............................] - ETA: 56s - loss: 7.8509 - accuracy: 0.4880
 2528/25000 [==>...........................] - ETA: 56s - loss: 7.8364 - accuracy: 0.4889
 2560/25000 [==>...........................] - ETA: 56s - loss: 7.8223 - accuracy: 0.4898
 2592/25000 [==>...........................] - ETA: 55s - loss: 7.8086 - accuracy: 0.4907
 2624/25000 [==>...........................] - ETA: 55s - loss: 7.8127 - accuracy: 0.4905
 2656/25000 [==>...........................] - ETA: 55s - loss: 7.8167 - accuracy: 0.4902
 2688/25000 [==>...........................] - ETA: 55s - loss: 7.8320 - accuracy: 0.4892
 2720/25000 [==>...........................] - ETA: 55s - loss: 7.8132 - accuracy: 0.4904
 2752/25000 [==>...........................] - ETA: 55s - loss: 7.7948 - accuracy: 0.4916
 2784/25000 [==>...........................] - ETA: 55s - loss: 7.8208 - accuracy: 0.4899
 2816/25000 [==>...........................] - ETA: 55s - loss: 7.7864 - accuracy: 0.4922
 2848/25000 [==>...........................] - ETA: 55s - loss: 7.7958 - accuracy: 0.4916
 2880/25000 [==>...........................] - ETA: 55s - loss: 7.8050 - accuracy: 0.4910
 2912/25000 [==>...........................] - ETA: 54s - loss: 7.8088 - accuracy: 0.4907
 2944/25000 [==>...........................] - ETA: 54s - loss: 7.8281 - accuracy: 0.4895
 2976/25000 [==>...........................] - ETA: 54s - loss: 7.8212 - accuracy: 0.4899
 3008/25000 [==>...........................] - ETA: 54s - loss: 7.8246 - accuracy: 0.4897
 3040/25000 [==>...........................] - ETA: 54s - loss: 7.8129 - accuracy: 0.4905
 3072/25000 [==>...........................] - ETA: 54s - loss: 7.7914 - accuracy: 0.4919
 3104/25000 [==>...........................] - ETA: 54s - loss: 7.7654 - accuracy: 0.4936
 3136/25000 [==>...........................] - ETA: 54s - loss: 7.7595 - accuracy: 0.4939
 3168/25000 [==>...........................] - ETA: 54s - loss: 7.7537 - accuracy: 0.4943
 3200/25000 [==>...........................] - ETA: 53s - loss: 7.7289 - accuracy: 0.4959
 3232/25000 [==>...........................] - ETA: 53s - loss: 7.7330 - accuracy: 0.4957
 3264/25000 [==>...........................] - ETA: 53s - loss: 7.7371 - accuracy: 0.4954
 3296/25000 [==>...........................] - ETA: 53s - loss: 7.7457 - accuracy: 0.4948
 3328/25000 [==>...........................] - ETA: 53s - loss: 7.7542 - accuracy: 0.4943
 3360/25000 [===>..........................] - ETA: 53s - loss: 7.7442 - accuracy: 0.4949
 3392/25000 [===>..........................] - ETA: 53s - loss: 7.7254 - accuracy: 0.4962
 3424/25000 [===>..........................] - ETA: 53s - loss: 7.7338 - accuracy: 0.4956
 3456/25000 [===>..........................] - ETA: 53s - loss: 7.7420 - accuracy: 0.4951
 3488/25000 [===>..........................] - ETA: 52s - loss: 7.7501 - accuracy: 0.4946
 3520/25000 [===>..........................] - ETA: 53s - loss: 7.7494 - accuracy: 0.4946
 3552/25000 [===>..........................] - ETA: 52s - loss: 7.7745 - accuracy: 0.4930
 3584/25000 [===>..........................] - ETA: 52s - loss: 7.7565 - accuracy: 0.4941
 3616/25000 [===>..........................] - ETA: 52s - loss: 7.7599 - accuracy: 0.4939
 3648/25000 [===>..........................] - ETA: 52s - loss: 7.7507 - accuracy: 0.4945
 3680/25000 [===>..........................] - ETA: 52s - loss: 7.7458 - accuracy: 0.4948
 3712/25000 [===>..........................] - ETA: 52s - loss: 7.7451 - accuracy: 0.4949
 3744/25000 [===>..........................] - ETA: 52s - loss: 7.7567 - accuracy: 0.4941
 3776/25000 [===>..........................] - ETA: 52s - loss: 7.7641 - accuracy: 0.4936
 3808/25000 [===>..........................] - ETA: 52s - loss: 7.7472 - accuracy: 0.4947
 3840/25000 [===>..........................] - ETA: 52s - loss: 7.7505 - accuracy: 0.4945
 3872/25000 [===>..........................] - ETA: 52s - loss: 7.7537 - accuracy: 0.4943
 3904/25000 [===>..........................] - ETA: 52s - loss: 7.7373 - accuracy: 0.4954
 3936/25000 [===>..........................] - ETA: 51s - loss: 7.7406 - accuracy: 0.4952
 3968/25000 [===>..........................] - ETA: 51s - loss: 7.7284 - accuracy: 0.4960
 4000/25000 [===>..........................] - ETA: 51s - loss: 7.7356 - accuracy: 0.4955
 4032/25000 [===>..........................] - ETA: 51s - loss: 7.7427 - accuracy: 0.4950
 4064/25000 [===>..........................] - ETA: 51s - loss: 7.7308 - accuracy: 0.4958
 4096/25000 [===>..........................] - ETA: 51s - loss: 7.7265 - accuracy: 0.4961
 4128/25000 [===>..........................] - ETA: 51s - loss: 7.7298 - accuracy: 0.4959
 4160/25000 [===>..........................] - ETA: 51s - loss: 7.7403 - accuracy: 0.4952
 4192/25000 [====>.........................] - ETA: 51s - loss: 7.7471 - accuracy: 0.4948
 4224/25000 [====>.........................] - ETA: 51s - loss: 7.7429 - accuracy: 0.4950
 4256/25000 [====>.........................] - ETA: 51s - loss: 7.7243 - accuracy: 0.4962
 4288/25000 [====>.........................] - ETA: 51s - loss: 7.7238 - accuracy: 0.4963
 4320/25000 [====>.........................] - ETA: 50s - loss: 7.7270 - accuracy: 0.4961
 4352/25000 [====>.........................] - ETA: 50s - loss: 7.7195 - accuracy: 0.4966
 4384/25000 [====>.........................] - ETA: 50s - loss: 7.7226 - accuracy: 0.4964
 4416/25000 [====>.........................] - ETA: 50s - loss: 7.7118 - accuracy: 0.4971
 4448/25000 [====>.........................] - ETA: 50s - loss: 7.7011 - accuracy: 0.4978
 4480/25000 [====>.........................] - ETA: 50s - loss: 7.7043 - accuracy: 0.4975
 4512/25000 [====>.........................] - ETA: 50s - loss: 7.6972 - accuracy: 0.4980
 4544/25000 [====>.........................] - ETA: 50s - loss: 7.6936 - accuracy: 0.4982
 4576/25000 [====>.........................] - ETA: 50s - loss: 7.6867 - accuracy: 0.4987
 4608/25000 [====>.........................] - ETA: 50s - loss: 7.6733 - accuracy: 0.4996
 4640/25000 [====>.........................] - ETA: 50s - loss: 7.6732 - accuracy: 0.4996
 4672/25000 [====>.........................] - ETA: 49s - loss: 7.6666 - accuracy: 0.5000
 4704/25000 [====>.........................] - ETA: 49s - loss: 7.6894 - accuracy: 0.4985
 4736/25000 [====>.........................] - ETA: 49s - loss: 7.6925 - accuracy: 0.4983
 4768/25000 [====>.........................] - ETA: 49s - loss: 7.6891 - accuracy: 0.4985
 4800/25000 [====>.........................] - ETA: 49s - loss: 7.6890 - accuracy: 0.4985
 4832/25000 [====>.........................] - ETA: 49s - loss: 7.7015 - accuracy: 0.4977
 4864/25000 [====>.........................] - ETA: 49s - loss: 7.7171 - accuracy: 0.4967
 4896/25000 [====>.........................] - ETA: 49s - loss: 7.7136 - accuracy: 0.4969
 4928/25000 [====>.........................] - ETA: 49s - loss: 7.7195 - accuracy: 0.4966
 4960/25000 [====>.........................] - ETA: 49s - loss: 7.7254 - accuracy: 0.4962
 4992/25000 [====>.........................] - ETA: 49s - loss: 7.7342 - accuracy: 0.4956
 5024/25000 [=====>........................] - ETA: 48s - loss: 7.7338 - accuracy: 0.4956
 5056/25000 [=====>........................] - ETA: 48s - loss: 7.7394 - accuracy: 0.4953
 5088/25000 [=====>........................] - ETA: 48s - loss: 7.7450 - accuracy: 0.4949
 5120/25000 [=====>........................] - ETA: 48s - loss: 7.7505 - accuracy: 0.4945
 5152/25000 [=====>........................] - ETA: 48s - loss: 7.7529 - accuracy: 0.4944
 5184/25000 [=====>........................] - ETA: 48s - loss: 7.7613 - accuracy: 0.4938
 5216/25000 [=====>........................] - ETA: 48s - loss: 7.7666 - accuracy: 0.4935
 5248/25000 [=====>........................] - ETA: 48s - loss: 7.7660 - accuracy: 0.4935
 5280/25000 [=====>........................] - ETA: 48s - loss: 7.7479 - accuracy: 0.4947
 5312/25000 [=====>........................] - ETA: 48s - loss: 7.7503 - accuracy: 0.4945
 5344/25000 [=====>........................] - ETA: 48s - loss: 7.7470 - accuracy: 0.4948
 5376/25000 [=====>........................] - ETA: 47s - loss: 7.7322 - accuracy: 0.4957
 5408/25000 [=====>........................] - ETA: 47s - loss: 7.7375 - accuracy: 0.4954
 5440/25000 [=====>........................] - ETA: 47s - loss: 7.7343 - accuracy: 0.4956
 5472/25000 [=====>........................] - ETA: 47s - loss: 7.7367 - accuracy: 0.4954
 5504/25000 [=====>........................] - ETA: 47s - loss: 7.7418 - accuracy: 0.4951
 5536/25000 [=====>........................] - ETA: 47s - loss: 7.7414 - accuracy: 0.4951
 5568/25000 [=====>........................] - ETA: 47s - loss: 7.7437 - accuracy: 0.4950
 5600/25000 [=====>........................] - ETA: 47s - loss: 7.7378 - accuracy: 0.4954
 5632/25000 [=====>........................] - ETA: 47s - loss: 7.7374 - accuracy: 0.4954
 5664/25000 [=====>........................] - ETA: 47s - loss: 7.7560 - accuracy: 0.4942
 5696/25000 [=====>........................] - ETA: 47s - loss: 7.7581 - accuracy: 0.4940
 5728/25000 [=====>........................] - ETA: 47s - loss: 7.7469 - accuracy: 0.4948
 5760/25000 [=====>........................] - ETA: 46s - loss: 7.7571 - accuracy: 0.4941
 5792/25000 [=====>........................] - ETA: 46s - loss: 7.7460 - accuracy: 0.4948
 5824/25000 [=====>........................] - ETA: 46s - loss: 7.7535 - accuracy: 0.4943
 5856/25000 [======>.......................] - ETA: 46s - loss: 7.7504 - accuracy: 0.4945
 5888/25000 [======>.......................] - ETA: 46s - loss: 7.7421 - accuracy: 0.4951
 5920/25000 [======>.......................] - ETA: 46s - loss: 7.7495 - accuracy: 0.4946
 5952/25000 [======>.......................] - ETA: 46s - loss: 7.7568 - accuracy: 0.4941
 5984/25000 [======>.......................] - ETA: 46s - loss: 7.7486 - accuracy: 0.4947
 6016/25000 [======>.......................] - ETA: 46s - loss: 7.7431 - accuracy: 0.4950
 6048/25000 [======>.......................] - ETA: 46s - loss: 7.7401 - accuracy: 0.4952
 6080/25000 [======>.......................] - ETA: 46s - loss: 7.7423 - accuracy: 0.4951
 6112/25000 [======>.......................] - ETA: 45s - loss: 7.7519 - accuracy: 0.4944
 6144/25000 [======>.......................] - ETA: 45s - loss: 7.7490 - accuracy: 0.4946
 6176/25000 [======>.......................] - ETA: 45s - loss: 7.7485 - accuracy: 0.4947
 6208/25000 [======>.......................] - ETA: 45s - loss: 7.7457 - accuracy: 0.4948
 6240/25000 [======>.......................] - ETA: 45s - loss: 7.7477 - accuracy: 0.4947
 6272/25000 [======>.......................] - ETA: 45s - loss: 7.7473 - accuracy: 0.4947
 6304/25000 [======>.......................] - ETA: 45s - loss: 7.7347 - accuracy: 0.4956
 6336/25000 [======>.......................] - ETA: 45s - loss: 7.7320 - accuracy: 0.4957
 6368/25000 [======>.......................] - ETA: 45s - loss: 7.7316 - accuracy: 0.4958
 6400/25000 [======>.......................] - ETA: 45s - loss: 7.7361 - accuracy: 0.4955
 6432/25000 [======>.......................] - ETA: 44s - loss: 7.7358 - accuracy: 0.4955
 6464/25000 [======>.......................] - ETA: 44s - loss: 7.7354 - accuracy: 0.4955
 6496/25000 [======>.......................] - ETA: 44s - loss: 7.7469 - accuracy: 0.4948
 6528/25000 [======>.......................] - ETA: 44s - loss: 7.7535 - accuracy: 0.4943
 6560/25000 [======>.......................] - ETA: 44s - loss: 7.7601 - accuracy: 0.4939
 6592/25000 [======>.......................] - ETA: 44s - loss: 7.7573 - accuracy: 0.4941
 6624/25000 [======>.......................] - ETA: 44s - loss: 7.7569 - accuracy: 0.4941
 6656/25000 [======>.......................] - ETA: 44s - loss: 7.7449 - accuracy: 0.4949
 6688/25000 [=======>......................] - ETA: 44s - loss: 7.7423 - accuracy: 0.4951
 6720/25000 [=======>......................] - ETA: 44s - loss: 7.7396 - accuracy: 0.4952
 6752/25000 [=======>......................] - ETA: 44s - loss: 7.7347 - accuracy: 0.4956
 6784/25000 [=======>......................] - ETA: 43s - loss: 7.7299 - accuracy: 0.4959
 6816/25000 [=======>......................] - ETA: 43s - loss: 7.7364 - accuracy: 0.4955
 6848/25000 [=======>......................] - ETA: 43s - loss: 7.7360 - accuracy: 0.4955
 6880/25000 [=======>......................] - ETA: 43s - loss: 7.7424 - accuracy: 0.4951
 6912/25000 [=======>......................] - ETA: 43s - loss: 7.7354 - accuracy: 0.4955
 6944/25000 [=======>......................] - ETA: 43s - loss: 7.7329 - accuracy: 0.4957
 6976/25000 [=======>......................] - ETA: 43s - loss: 7.7326 - accuracy: 0.4957
 7008/25000 [=======>......................] - ETA: 43s - loss: 7.7235 - accuracy: 0.4963
 7040/25000 [=======>......................] - ETA: 43s - loss: 7.7211 - accuracy: 0.4964
 7072/25000 [=======>......................] - ETA: 43s - loss: 7.7187 - accuracy: 0.4966
 7104/25000 [=======>......................] - ETA: 43s - loss: 7.7141 - accuracy: 0.4969
 7136/25000 [=======>......................] - ETA: 43s - loss: 7.7117 - accuracy: 0.4971
 7168/25000 [=======>......................] - ETA: 42s - loss: 7.7115 - accuracy: 0.4971
 7200/25000 [=======>......................] - ETA: 42s - loss: 7.7156 - accuracy: 0.4968
 7232/25000 [=======>......................] - ETA: 42s - loss: 7.7133 - accuracy: 0.4970
 7264/25000 [=======>......................] - ETA: 42s - loss: 7.7236 - accuracy: 0.4963
 7296/25000 [=======>......................] - ETA: 42s - loss: 7.7234 - accuracy: 0.4963
 7328/25000 [=======>......................] - ETA: 42s - loss: 7.7273 - accuracy: 0.4960
 7360/25000 [=======>......................] - ETA: 42s - loss: 7.7291 - accuracy: 0.4959
 7392/25000 [=======>......................] - ETA: 42s - loss: 7.7392 - accuracy: 0.4953
 7424/25000 [=======>......................] - ETA: 42s - loss: 7.7265 - accuracy: 0.4961
 7456/25000 [=======>......................] - ETA: 42s - loss: 7.7304 - accuracy: 0.4958
 7488/25000 [=======>......................] - ETA: 42s - loss: 7.7219 - accuracy: 0.4964
 7520/25000 [========>.....................] - ETA: 42s - loss: 7.7156 - accuracy: 0.4968
 7552/25000 [========>.....................] - ETA: 42s - loss: 7.7194 - accuracy: 0.4966
 7584/25000 [========>.....................] - ETA: 41s - loss: 7.7192 - accuracy: 0.4966
 7616/25000 [========>.....................] - ETA: 41s - loss: 7.7230 - accuracy: 0.4963
 7648/25000 [========>.....................] - ETA: 41s - loss: 7.7228 - accuracy: 0.4963
 7680/25000 [========>.....................] - ETA: 41s - loss: 7.7265 - accuracy: 0.4961
 7712/25000 [========>.....................] - ETA: 41s - loss: 7.7183 - accuracy: 0.4966
 7744/25000 [========>.....................] - ETA: 41s - loss: 7.7141 - accuracy: 0.4969
 7776/25000 [========>.....................] - ETA: 41s - loss: 7.7100 - accuracy: 0.4972
 7808/25000 [========>.....................] - ETA: 41s - loss: 7.7177 - accuracy: 0.4967
 7840/25000 [========>.....................] - ETA: 41s - loss: 7.7194 - accuracy: 0.4966
 7872/25000 [========>.....................] - ETA: 41s - loss: 7.7114 - accuracy: 0.4971
 7904/25000 [========>.....................] - ETA: 41s - loss: 7.7054 - accuracy: 0.4975
 7936/25000 [========>.....................] - ETA: 41s - loss: 7.7014 - accuracy: 0.4977
 7968/25000 [========>.....................] - ETA: 40s - loss: 7.6916 - accuracy: 0.4984
 8000/25000 [========>.....................] - ETA: 40s - loss: 7.6896 - accuracy: 0.4985
 8032/25000 [========>.....................] - ETA: 40s - loss: 7.6838 - accuracy: 0.4989
 8064/25000 [========>.....................] - ETA: 40s - loss: 7.6875 - accuracy: 0.4986
 8096/25000 [========>.....................] - ETA: 40s - loss: 7.6931 - accuracy: 0.4983
 8128/25000 [========>.....................] - ETA: 40s - loss: 7.6949 - accuracy: 0.4982
 8160/25000 [========>.....................] - ETA: 40s - loss: 7.6892 - accuracy: 0.4985
 8192/25000 [========>.....................] - ETA: 40s - loss: 7.6891 - accuracy: 0.4985
 8224/25000 [========>.....................] - ETA: 40s - loss: 7.6909 - accuracy: 0.4984
 8256/25000 [========>.....................] - ETA: 40s - loss: 7.6908 - accuracy: 0.4984
 8288/25000 [========>.....................] - ETA: 40s - loss: 7.6962 - accuracy: 0.4981
 8320/25000 [========>.....................] - ETA: 40s - loss: 7.7016 - accuracy: 0.4977
 8352/25000 [=========>....................] - ETA: 40s - loss: 7.6978 - accuracy: 0.4980
 8384/25000 [=========>....................] - ETA: 39s - loss: 7.6977 - accuracy: 0.4980
 8416/25000 [=========>....................] - ETA: 39s - loss: 7.7031 - accuracy: 0.4976
 8448/25000 [=========>....................] - ETA: 39s - loss: 7.7011 - accuracy: 0.4978
 8480/25000 [=========>....................] - ETA: 39s - loss: 7.7046 - accuracy: 0.4975
 8512/25000 [=========>....................] - ETA: 39s - loss: 7.6972 - accuracy: 0.4980
 8544/25000 [=========>....................] - ETA: 39s - loss: 7.6971 - accuracy: 0.4980
 8576/25000 [=========>....................] - ETA: 39s - loss: 7.6970 - accuracy: 0.4980
 8608/25000 [=========>....................] - ETA: 39s - loss: 7.6880 - accuracy: 0.4986
 8640/25000 [=========>....................] - ETA: 39s - loss: 7.6897 - accuracy: 0.4985
 8672/25000 [=========>....................] - ETA: 39s - loss: 7.6861 - accuracy: 0.4987
 8704/25000 [=========>....................] - ETA: 39s - loss: 7.6807 - accuracy: 0.4991
 8736/25000 [=========>....................] - ETA: 39s - loss: 7.6824 - accuracy: 0.4990
 8768/25000 [=========>....................] - ETA: 39s - loss: 7.6859 - accuracy: 0.4987
 8800/25000 [=========>....................] - ETA: 38s - loss: 7.6840 - accuracy: 0.4989
 8832/25000 [=========>....................] - ETA: 38s - loss: 7.6770 - accuracy: 0.4993
 8864/25000 [=========>....................] - ETA: 38s - loss: 7.6753 - accuracy: 0.4994
 8896/25000 [=========>....................] - ETA: 38s - loss: 7.6770 - accuracy: 0.4993
 8928/25000 [=========>....................] - ETA: 38s - loss: 7.6821 - accuracy: 0.4990
 8960/25000 [=========>....................] - ETA: 38s - loss: 7.6786 - accuracy: 0.4992
 8992/25000 [=========>....................] - ETA: 38s - loss: 7.6734 - accuracy: 0.4996
 9024/25000 [=========>....................] - ETA: 38s - loss: 7.6700 - accuracy: 0.4998
 9056/25000 [=========>....................] - ETA: 38s - loss: 7.6632 - accuracy: 0.5002
 9088/25000 [=========>....................] - ETA: 38s - loss: 7.6649 - accuracy: 0.5001
 9120/25000 [=========>....................] - ETA: 38s - loss: 7.6633 - accuracy: 0.5002
 9152/25000 [=========>....................] - ETA: 38s - loss: 7.6599 - accuracy: 0.5004
 9184/25000 [==========>...................] - ETA: 38s - loss: 7.6616 - accuracy: 0.5003
 9216/25000 [==========>...................] - ETA: 37s - loss: 7.6683 - accuracy: 0.4999
 9248/25000 [==========>...................] - ETA: 37s - loss: 7.6716 - accuracy: 0.4997
 9280/25000 [==========>...................] - ETA: 37s - loss: 7.6716 - accuracy: 0.4997
 9312/25000 [==========>...................] - ETA: 37s - loss: 7.6831 - accuracy: 0.4989
 9344/25000 [==========>...................] - ETA: 37s - loss: 7.6847 - accuracy: 0.4988
 9376/25000 [==========>...................] - ETA: 37s - loss: 7.6862 - accuracy: 0.4987
 9408/25000 [==========>...................] - ETA: 37s - loss: 7.6764 - accuracy: 0.4994
 9440/25000 [==========>...................] - ETA: 37s - loss: 7.6747 - accuracy: 0.4995
 9472/25000 [==========>...................] - ETA: 37s - loss: 7.6682 - accuracy: 0.4999
 9504/25000 [==========>...................] - ETA: 37s - loss: 7.6715 - accuracy: 0.4997
 9536/25000 [==========>...................] - ETA: 37s - loss: 7.6698 - accuracy: 0.4998
 9568/25000 [==========>...................] - ETA: 37s - loss: 7.6666 - accuracy: 0.5000
 9600/25000 [==========>...................] - ETA: 37s - loss: 7.6650 - accuracy: 0.5001
 9632/25000 [==========>...................] - ETA: 36s - loss: 7.6618 - accuracy: 0.5003
 9664/25000 [==========>...................] - ETA: 36s - loss: 7.6634 - accuracy: 0.5002
 9696/25000 [==========>...................] - ETA: 36s - loss: 7.6650 - accuracy: 0.5001
 9728/25000 [==========>...................] - ETA: 36s - loss: 7.6682 - accuracy: 0.4999
 9760/25000 [==========>...................] - ETA: 36s - loss: 7.6713 - accuracy: 0.4997
 9792/25000 [==========>...................] - ETA: 36s - loss: 7.6729 - accuracy: 0.4996
 9824/25000 [==========>...................] - ETA: 36s - loss: 7.6744 - accuracy: 0.4995
 9856/25000 [==========>...................] - ETA: 36s - loss: 7.6760 - accuracy: 0.4994
 9888/25000 [==========>...................] - ETA: 36s - loss: 7.6790 - accuracy: 0.4992
 9920/25000 [==========>...................] - ETA: 36s - loss: 7.6759 - accuracy: 0.4994
 9952/25000 [==========>...................] - ETA: 36s - loss: 7.6666 - accuracy: 0.5000
 9984/25000 [==========>...................] - ETA: 36s - loss: 7.6651 - accuracy: 0.5001
10016/25000 [===========>..................] - ETA: 35s - loss: 7.6636 - accuracy: 0.5002
10048/25000 [===========>..................] - ETA: 35s - loss: 7.6605 - accuracy: 0.5004
10080/25000 [===========>..................] - ETA: 35s - loss: 7.6651 - accuracy: 0.5001
10112/25000 [===========>..................] - ETA: 35s - loss: 7.6606 - accuracy: 0.5004
10144/25000 [===========>..................] - ETA: 35s - loss: 7.6500 - accuracy: 0.5011
10176/25000 [===========>..................] - ETA: 35s - loss: 7.6455 - accuracy: 0.5014
10208/25000 [===========>..................] - ETA: 35s - loss: 7.6411 - accuracy: 0.5017
10240/25000 [===========>..................] - ETA: 35s - loss: 7.6382 - accuracy: 0.5019
10272/25000 [===========>..................] - ETA: 35s - loss: 7.6368 - accuracy: 0.5019
10304/25000 [===========>..................] - ETA: 35s - loss: 7.6398 - accuracy: 0.5017
10336/25000 [===========>..................] - ETA: 35s - loss: 7.6459 - accuracy: 0.5014
10368/25000 [===========>..................] - ETA: 35s - loss: 7.6474 - accuracy: 0.5013
10400/25000 [===========>..................] - ETA: 35s - loss: 7.6534 - accuracy: 0.5009
10432/25000 [===========>..................] - ETA: 34s - loss: 7.6431 - accuracy: 0.5015
10464/25000 [===========>..................] - ETA: 34s - loss: 7.6461 - accuracy: 0.5013
10496/25000 [===========>..................] - ETA: 34s - loss: 7.6462 - accuracy: 0.5013
10528/25000 [===========>..................] - ETA: 34s - loss: 7.6462 - accuracy: 0.5013
10560/25000 [===========>..................] - ETA: 34s - loss: 7.6463 - accuracy: 0.5013
10592/25000 [===========>..................] - ETA: 34s - loss: 7.6478 - accuracy: 0.5012
10624/25000 [===========>..................] - ETA: 34s - loss: 7.6435 - accuracy: 0.5015
10656/25000 [===========>..................] - ETA: 34s - loss: 7.6422 - accuracy: 0.5016
10688/25000 [===========>..................] - ETA: 34s - loss: 7.6408 - accuracy: 0.5017
10720/25000 [===========>..................] - ETA: 34s - loss: 7.6409 - accuracy: 0.5017
10752/25000 [===========>..................] - ETA: 34s - loss: 7.6410 - accuracy: 0.5017
10784/25000 [===========>..................] - ETA: 34s - loss: 7.6453 - accuracy: 0.5014
10816/25000 [===========>..................] - ETA: 33s - loss: 7.6439 - accuracy: 0.5015
10848/25000 [============>.................] - ETA: 33s - loss: 7.6426 - accuracy: 0.5016
10880/25000 [============>.................] - ETA: 33s - loss: 7.6497 - accuracy: 0.5011
10912/25000 [============>.................] - ETA: 33s - loss: 7.6540 - accuracy: 0.5008
10944/25000 [============>.................] - ETA: 33s - loss: 7.6540 - accuracy: 0.5008
10976/25000 [============>.................] - ETA: 33s - loss: 7.6513 - accuracy: 0.5010
11008/25000 [============>.................] - ETA: 33s - loss: 7.6541 - accuracy: 0.5008
11040/25000 [============>.................] - ETA: 33s - loss: 7.6583 - accuracy: 0.5005
11072/25000 [============>.................] - ETA: 33s - loss: 7.6611 - accuracy: 0.5004
11104/25000 [============>.................] - ETA: 33s - loss: 7.6639 - accuracy: 0.5002
11136/25000 [============>.................] - ETA: 33s - loss: 7.6611 - accuracy: 0.5004
11168/25000 [============>.................] - ETA: 33s - loss: 7.6625 - accuracy: 0.5003
11200/25000 [============>.................] - ETA: 33s - loss: 7.6625 - accuracy: 0.5003
11232/25000 [============>.................] - ETA: 32s - loss: 7.6612 - accuracy: 0.5004
11264/25000 [============>.................] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
11296/25000 [============>.................] - ETA: 32s - loss: 7.6748 - accuracy: 0.4995
11328/25000 [============>.................] - ETA: 32s - loss: 7.6774 - accuracy: 0.4993
11360/25000 [============>.................] - ETA: 32s - loss: 7.6801 - accuracy: 0.4991
11392/25000 [============>.................] - ETA: 32s - loss: 7.6787 - accuracy: 0.4992
11424/25000 [============>.................] - ETA: 32s - loss: 7.6760 - accuracy: 0.4994
11456/25000 [============>.................] - ETA: 32s - loss: 7.6706 - accuracy: 0.4997
11488/25000 [============>.................] - ETA: 32s - loss: 7.6653 - accuracy: 0.5001
11520/25000 [============>.................] - ETA: 32s - loss: 7.6586 - accuracy: 0.5005
11552/25000 [============>.................] - ETA: 32s - loss: 7.6626 - accuracy: 0.5003
11584/25000 [============>.................] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
11616/25000 [============>.................] - ETA: 31s - loss: 7.6627 - accuracy: 0.5003
11648/25000 [============>.................] - ETA: 31s - loss: 7.6706 - accuracy: 0.4997
11680/25000 [=============>................] - ETA: 31s - loss: 7.6679 - accuracy: 0.4999
11712/25000 [=============>................] - ETA: 31s - loss: 7.6627 - accuracy: 0.5003
11744/25000 [=============>................] - ETA: 31s - loss: 7.6614 - accuracy: 0.5003
11776/25000 [=============>................] - ETA: 31s - loss: 7.6601 - accuracy: 0.5004
11808/25000 [=============>................] - ETA: 31s - loss: 7.6627 - accuracy: 0.5003
11840/25000 [=============>................] - ETA: 31s - loss: 7.6614 - accuracy: 0.5003
11872/25000 [=============>................] - ETA: 31s - loss: 7.6602 - accuracy: 0.5004
11904/25000 [=============>................] - ETA: 31s - loss: 7.6602 - accuracy: 0.5004
11936/25000 [=============>................] - ETA: 31s - loss: 7.6615 - accuracy: 0.5003
11968/25000 [=============>................] - ETA: 31s - loss: 7.6589 - accuracy: 0.5005
12000/25000 [=============>................] - ETA: 31s - loss: 7.6564 - accuracy: 0.5007
12032/25000 [=============>................] - ETA: 30s - loss: 7.6577 - accuracy: 0.5006
12064/25000 [=============>................] - ETA: 30s - loss: 7.6603 - accuracy: 0.5004
12096/25000 [=============>................] - ETA: 30s - loss: 7.6552 - accuracy: 0.5007
12128/25000 [=============>................] - ETA: 30s - loss: 7.6552 - accuracy: 0.5007
12160/25000 [=============>................] - ETA: 30s - loss: 7.6527 - accuracy: 0.5009
12192/25000 [=============>................] - ETA: 30s - loss: 7.6528 - accuracy: 0.5009
12224/25000 [=============>................] - ETA: 30s - loss: 7.6603 - accuracy: 0.5004
12256/25000 [=============>................] - ETA: 30s - loss: 7.6529 - accuracy: 0.5009
12288/25000 [=============>................] - ETA: 30s - loss: 7.6529 - accuracy: 0.5009
12320/25000 [=============>................] - ETA: 30s - loss: 7.6517 - accuracy: 0.5010
12352/25000 [=============>................] - ETA: 30s - loss: 7.6492 - accuracy: 0.5011
12384/25000 [=============>................] - ETA: 30s - loss: 7.6505 - accuracy: 0.5010
12416/25000 [=============>................] - ETA: 30s - loss: 7.6567 - accuracy: 0.5006
12448/25000 [=============>................] - ETA: 29s - loss: 7.6605 - accuracy: 0.5004
12480/25000 [=============>................] - ETA: 29s - loss: 7.6580 - accuracy: 0.5006
12512/25000 [==============>...............] - ETA: 29s - loss: 7.6593 - accuracy: 0.5005
12544/25000 [==============>...............] - ETA: 29s - loss: 7.6617 - accuracy: 0.5003
12576/25000 [==============>...............] - ETA: 29s - loss: 7.6666 - accuracy: 0.5000
12608/25000 [==============>...............] - ETA: 29s - loss: 7.6666 - accuracy: 0.5000
12640/25000 [==============>...............] - ETA: 29s - loss: 7.6654 - accuracy: 0.5001
12672/25000 [==============>...............] - ETA: 29s - loss: 7.6618 - accuracy: 0.5003
12704/25000 [==============>...............] - ETA: 29s - loss: 7.6630 - accuracy: 0.5002
12736/25000 [==============>...............] - ETA: 29s - loss: 7.6606 - accuracy: 0.5004
12768/25000 [==============>...............] - ETA: 29s - loss: 7.6570 - accuracy: 0.5006
12800/25000 [==============>...............] - ETA: 29s - loss: 7.6570 - accuracy: 0.5006
12832/25000 [==============>...............] - ETA: 29s - loss: 7.6511 - accuracy: 0.5010
12864/25000 [==============>...............] - ETA: 28s - loss: 7.6464 - accuracy: 0.5013
12896/25000 [==============>...............] - ETA: 28s - loss: 7.6428 - accuracy: 0.5016
12928/25000 [==============>...............] - ETA: 28s - loss: 7.6405 - accuracy: 0.5017
12960/25000 [==============>...............] - ETA: 28s - loss: 7.6394 - accuracy: 0.5018
12992/25000 [==============>...............] - ETA: 28s - loss: 7.6371 - accuracy: 0.5019
13024/25000 [==============>...............] - ETA: 28s - loss: 7.6348 - accuracy: 0.5021
13056/25000 [==============>...............] - ETA: 28s - loss: 7.6314 - accuracy: 0.5023
13088/25000 [==============>...............] - ETA: 28s - loss: 7.6326 - accuracy: 0.5022
13120/25000 [==============>...............] - ETA: 28s - loss: 7.6351 - accuracy: 0.5021
13152/25000 [==============>...............] - ETA: 28s - loss: 7.6281 - accuracy: 0.5025
13184/25000 [==============>...............] - ETA: 28s - loss: 7.6213 - accuracy: 0.5030
13216/25000 [==============>...............] - ETA: 28s - loss: 7.6295 - accuracy: 0.5024
13248/25000 [==============>...............] - ETA: 27s - loss: 7.6307 - accuracy: 0.5023
13280/25000 [==============>...............] - ETA: 27s - loss: 7.6274 - accuracy: 0.5026
13312/25000 [==============>...............] - ETA: 27s - loss: 7.6252 - accuracy: 0.5027
13344/25000 [===============>..............] - ETA: 27s - loss: 7.6276 - accuracy: 0.5025
13376/25000 [===============>..............] - ETA: 27s - loss: 7.6254 - accuracy: 0.5027
13408/25000 [===============>..............] - ETA: 27s - loss: 7.6266 - accuracy: 0.5026
13440/25000 [===============>..............] - ETA: 27s - loss: 7.6267 - accuracy: 0.5026
13472/25000 [===============>..............] - ETA: 27s - loss: 7.6325 - accuracy: 0.5022
13504/25000 [===============>..............] - ETA: 27s - loss: 7.6303 - accuracy: 0.5024
13536/25000 [===============>..............] - ETA: 27s - loss: 7.6338 - accuracy: 0.5021
13568/25000 [===============>..............] - ETA: 27s - loss: 7.6327 - accuracy: 0.5022
13600/25000 [===============>..............] - ETA: 27s - loss: 7.6260 - accuracy: 0.5026
13632/25000 [===============>..............] - ETA: 27s - loss: 7.6295 - accuracy: 0.5024
13664/25000 [===============>..............] - ETA: 26s - loss: 7.6296 - accuracy: 0.5024
13696/25000 [===============>..............] - ETA: 26s - loss: 7.6230 - accuracy: 0.5028
13728/25000 [===============>..............] - ETA: 26s - loss: 7.6219 - accuracy: 0.5029
13760/25000 [===============>..............] - ETA: 26s - loss: 7.6265 - accuracy: 0.5026
13792/25000 [===============>..............] - ETA: 26s - loss: 7.6255 - accuracy: 0.5027
13824/25000 [===============>..............] - ETA: 26s - loss: 7.6223 - accuracy: 0.5029
13856/25000 [===============>..............] - ETA: 26s - loss: 7.6224 - accuracy: 0.5029
13888/25000 [===============>..............] - ETA: 26s - loss: 7.6247 - accuracy: 0.5027
13920/25000 [===============>..............] - ETA: 26s - loss: 7.6237 - accuracy: 0.5028
13952/25000 [===============>..............] - ETA: 26s - loss: 7.6238 - accuracy: 0.5028
13984/25000 [===============>..............] - ETA: 26s - loss: 7.6228 - accuracy: 0.5029
14016/25000 [===============>..............] - ETA: 26s - loss: 7.6218 - accuracy: 0.5029
14048/25000 [===============>..............] - ETA: 26s - loss: 7.6251 - accuracy: 0.5027
14080/25000 [===============>..............] - ETA: 25s - loss: 7.6231 - accuracy: 0.5028
14112/25000 [===============>..............] - ETA: 25s - loss: 7.6242 - accuracy: 0.5028
14144/25000 [===============>..............] - ETA: 25s - loss: 7.6265 - accuracy: 0.5026
14176/25000 [================>.............] - ETA: 25s - loss: 7.6288 - accuracy: 0.5025
14208/25000 [================>.............] - ETA: 25s - loss: 7.6278 - accuracy: 0.5025
14240/25000 [================>.............] - ETA: 25s - loss: 7.6300 - accuracy: 0.5024
14272/25000 [================>.............] - ETA: 25s - loss: 7.6322 - accuracy: 0.5022
14304/25000 [================>.............] - ETA: 25s - loss: 7.6334 - accuracy: 0.5022
14336/25000 [================>.............] - ETA: 25s - loss: 7.6388 - accuracy: 0.5018
14368/25000 [================>.............] - ETA: 25s - loss: 7.6357 - accuracy: 0.5020
14400/25000 [================>.............] - ETA: 25s - loss: 7.6357 - accuracy: 0.5020
14432/25000 [================>.............] - ETA: 25s - loss: 7.6347 - accuracy: 0.5021
14464/25000 [================>.............] - ETA: 25s - loss: 7.6401 - accuracy: 0.5017
14496/25000 [================>.............] - ETA: 25s - loss: 7.6433 - accuracy: 0.5015
14528/25000 [================>.............] - ETA: 24s - loss: 7.6445 - accuracy: 0.5014
14560/25000 [================>.............] - ETA: 24s - loss: 7.6435 - accuracy: 0.5015
14592/25000 [================>.............] - ETA: 24s - loss: 7.6477 - accuracy: 0.5012
14624/25000 [================>.............] - ETA: 24s - loss: 7.6477 - accuracy: 0.5012
14656/25000 [================>.............] - ETA: 24s - loss: 7.6457 - accuracy: 0.5014
14688/25000 [================>.............] - ETA: 24s - loss: 7.6426 - accuracy: 0.5016
14720/25000 [================>.............] - ETA: 24s - loss: 7.6458 - accuracy: 0.5014
14752/25000 [================>.............] - ETA: 24s - loss: 7.6510 - accuracy: 0.5010
14784/25000 [================>.............] - ETA: 24s - loss: 7.6521 - accuracy: 0.5009
14816/25000 [================>.............] - ETA: 24s - loss: 7.6573 - accuracy: 0.5006
14848/25000 [================>.............] - ETA: 24s - loss: 7.6573 - accuracy: 0.5006
14880/25000 [================>.............] - ETA: 24s - loss: 7.6584 - accuracy: 0.5005
14912/25000 [================>.............] - ETA: 24s - loss: 7.6584 - accuracy: 0.5005
14944/25000 [================>.............] - ETA: 23s - loss: 7.6615 - accuracy: 0.5003
14976/25000 [================>.............] - ETA: 23s - loss: 7.6584 - accuracy: 0.5005
15008/25000 [=================>............] - ETA: 23s - loss: 7.6595 - accuracy: 0.5005
15040/25000 [=================>............] - ETA: 23s - loss: 7.6615 - accuracy: 0.5003
15072/25000 [=================>............] - ETA: 23s - loss: 7.6575 - accuracy: 0.5006
15104/25000 [=================>............] - ETA: 23s - loss: 7.6615 - accuracy: 0.5003
15136/25000 [=================>............] - ETA: 23s - loss: 7.6595 - accuracy: 0.5005
15168/25000 [=================>............] - ETA: 23s - loss: 7.6555 - accuracy: 0.5007
15200/25000 [=================>............] - ETA: 23s - loss: 7.6565 - accuracy: 0.5007
15232/25000 [=================>............] - ETA: 23s - loss: 7.6545 - accuracy: 0.5008
15264/25000 [=================>............] - ETA: 23s - loss: 7.6546 - accuracy: 0.5008
15296/25000 [=================>............] - ETA: 23s - loss: 7.6486 - accuracy: 0.5012
15328/25000 [=================>............] - ETA: 23s - loss: 7.6436 - accuracy: 0.5015
15360/25000 [=================>............] - ETA: 22s - loss: 7.6437 - accuracy: 0.5015
15392/25000 [=================>............] - ETA: 22s - loss: 7.6417 - accuracy: 0.5016
15424/25000 [=================>............] - ETA: 22s - loss: 7.6398 - accuracy: 0.5018
15456/25000 [=================>............] - ETA: 22s - loss: 7.6408 - accuracy: 0.5017
15488/25000 [=================>............] - ETA: 22s - loss: 7.6399 - accuracy: 0.5017
15520/25000 [=================>............] - ETA: 22s - loss: 7.6439 - accuracy: 0.5015
15552/25000 [=================>............] - ETA: 22s - loss: 7.6489 - accuracy: 0.5012
15584/25000 [=================>............] - ETA: 22s - loss: 7.6519 - accuracy: 0.5010
15616/25000 [=================>............] - ETA: 22s - loss: 7.6519 - accuracy: 0.5010
15648/25000 [=================>............] - ETA: 22s - loss: 7.6529 - accuracy: 0.5009
15680/25000 [=================>............] - ETA: 22s - loss: 7.6520 - accuracy: 0.5010
15712/25000 [=================>............] - ETA: 22s - loss: 7.6578 - accuracy: 0.5006
15744/25000 [=================>............] - ETA: 22s - loss: 7.6569 - accuracy: 0.5006
15776/25000 [=================>............] - ETA: 21s - loss: 7.6569 - accuracy: 0.5006
15808/25000 [=================>............] - ETA: 21s - loss: 7.6540 - accuracy: 0.5008
15840/25000 [==================>...........] - ETA: 21s - loss: 7.6521 - accuracy: 0.5009
15872/25000 [==================>...........] - ETA: 21s - loss: 7.6531 - accuracy: 0.5009
15904/25000 [==================>...........] - ETA: 21s - loss: 7.6541 - accuracy: 0.5008
15936/25000 [==================>...........] - ETA: 21s - loss: 7.6503 - accuracy: 0.5011
15968/25000 [==================>...........] - ETA: 21s - loss: 7.6455 - accuracy: 0.5014
16000/25000 [==================>...........] - ETA: 21s - loss: 7.6455 - accuracy: 0.5014
16032/25000 [==================>...........] - ETA: 21s - loss: 7.6456 - accuracy: 0.5014
16064/25000 [==================>...........] - ETA: 21s - loss: 7.6466 - accuracy: 0.5013
16096/25000 [==================>...........] - ETA: 21s - loss: 7.6495 - accuracy: 0.5011
16128/25000 [==================>...........] - ETA: 21s - loss: 7.6476 - accuracy: 0.5012
16160/25000 [==================>...........] - ETA: 21s - loss: 7.6495 - accuracy: 0.5011
16192/25000 [==================>...........] - ETA: 20s - loss: 7.6505 - accuracy: 0.5010
16224/25000 [==================>...........] - ETA: 20s - loss: 7.6534 - accuracy: 0.5009
16256/25000 [==================>...........] - ETA: 20s - loss: 7.6496 - accuracy: 0.5011
16288/25000 [==================>...........] - ETA: 20s - loss: 7.6497 - accuracy: 0.5011
16320/25000 [==================>...........] - ETA: 20s - loss: 7.6525 - accuracy: 0.5009
16352/25000 [==================>...........] - ETA: 20s - loss: 7.6554 - accuracy: 0.5007
16384/25000 [==================>...........] - ETA: 20s - loss: 7.6488 - accuracy: 0.5012
16416/25000 [==================>...........] - ETA: 20s - loss: 7.6433 - accuracy: 0.5015
16448/25000 [==================>...........] - ETA: 20s - loss: 7.6424 - accuracy: 0.5016
16480/25000 [==================>...........] - ETA: 20s - loss: 7.6387 - accuracy: 0.5018
16512/25000 [==================>...........] - ETA: 20s - loss: 7.6406 - accuracy: 0.5017
16544/25000 [==================>...........] - ETA: 20s - loss: 7.6407 - accuracy: 0.5017
16576/25000 [==================>...........] - ETA: 20s - loss: 7.6426 - accuracy: 0.5016
16608/25000 [==================>...........] - ETA: 19s - loss: 7.6408 - accuracy: 0.5017
16640/25000 [==================>...........] - ETA: 19s - loss: 7.6417 - accuracy: 0.5016
16672/25000 [===================>..........] - ETA: 19s - loss: 7.6399 - accuracy: 0.5017
16704/25000 [===================>..........] - ETA: 19s - loss: 7.6400 - accuracy: 0.5017
16736/25000 [===================>..........] - ETA: 19s - loss: 7.6364 - accuracy: 0.5020
16768/25000 [===================>..........] - ETA: 19s - loss: 7.6328 - accuracy: 0.5022
16800/25000 [===================>..........] - ETA: 19s - loss: 7.6319 - accuracy: 0.5023
16832/25000 [===================>..........] - ETA: 19s - loss: 7.6338 - accuracy: 0.5021
16864/25000 [===================>..........] - ETA: 19s - loss: 7.6330 - accuracy: 0.5022
16896/25000 [===================>..........] - ETA: 19s - loss: 7.6367 - accuracy: 0.5020
16928/25000 [===================>..........] - ETA: 19s - loss: 7.6340 - accuracy: 0.5021
16960/25000 [===================>..........] - ETA: 19s - loss: 7.6350 - accuracy: 0.5021
16992/25000 [===================>..........] - ETA: 19s - loss: 7.6341 - accuracy: 0.5021
17024/25000 [===================>..........] - ETA: 19s - loss: 7.6351 - accuracy: 0.5021
17056/25000 [===================>..........] - ETA: 18s - loss: 7.6352 - accuracy: 0.5021
17088/25000 [===================>..........] - ETA: 18s - loss: 7.6361 - accuracy: 0.5020
17120/25000 [===================>..........] - ETA: 18s - loss: 7.6344 - accuracy: 0.5021
17152/25000 [===================>..........] - ETA: 18s - loss: 7.6344 - accuracy: 0.5021
17184/25000 [===================>..........] - ETA: 18s - loss: 7.6354 - accuracy: 0.5020
17216/25000 [===================>..........] - ETA: 18s - loss: 7.6354 - accuracy: 0.5020
17248/25000 [===================>..........] - ETA: 18s - loss: 7.6391 - accuracy: 0.5018
17280/25000 [===================>..........] - ETA: 18s - loss: 7.6409 - accuracy: 0.5017
17312/25000 [===================>..........] - ETA: 18s - loss: 7.6400 - accuracy: 0.5017
17344/25000 [===================>..........] - ETA: 18s - loss: 7.6348 - accuracy: 0.5021
17376/25000 [===================>..........] - ETA: 18s - loss: 7.6313 - accuracy: 0.5023
17408/25000 [===================>..........] - ETA: 18s - loss: 7.6270 - accuracy: 0.5026
17440/25000 [===================>..........] - ETA: 18s - loss: 7.6279 - accuracy: 0.5025
17472/25000 [===================>..........] - ETA: 17s - loss: 7.6298 - accuracy: 0.5024
17504/25000 [====================>.........] - ETA: 17s - loss: 7.6307 - accuracy: 0.5023
17536/25000 [====================>.........] - ETA: 17s - loss: 7.6308 - accuracy: 0.5023
17568/25000 [====================>.........] - ETA: 17s - loss: 7.6300 - accuracy: 0.5024
17600/25000 [====================>.........] - ETA: 17s - loss: 7.6292 - accuracy: 0.5024
17632/25000 [====================>.........] - ETA: 17s - loss: 7.6310 - accuracy: 0.5023
17664/25000 [====================>.........] - ETA: 17s - loss: 7.6293 - accuracy: 0.5024
17696/25000 [====================>.........] - ETA: 17s - loss: 7.6311 - accuracy: 0.5023
17728/25000 [====================>.........] - ETA: 17s - loss: 7.6303 - accuracy: 0.5024
17760/25000 [====================>.........] - ETA: 17s - loss: 7.6243 - accuracy: 0.5028
17792/25000 [====================>.........] - ETA: 17s - loss: 7.6261 - accuracy: 0.5026
17824/25000 [====================>.........] - ETA: 17s - loss: 7.6236 - accuracy: 0.5028
17856/25000 [====================>.........] - ETA: 17s - loss: 7.6228 - accuracy: 0.5029
17888/25000 [====================>.........] - ETA: 17s - loss: 7.6220 - accuracy: 0.5029
17920/25000 [====================>.........] - ETA: 16s - loss: 7.6187 - accuracy: 0.5031
17952/25000 [====================>.........] - ETA: 16s - loss: 7.6179 - accuracy: 0.5032
17984/25000 [====================>.........] - ETA: 16s - loss: 7.6180 - accuracy: 0.5032
18016/25000 [====================>.........] - ETA: 16s - loss: 7.6190 - accuracy: 0.5031
18048/25000 [====================>.........] - ETA: 16s - loss: 7.6224 - accuracy: 0.5029
18080/25000 [====================>.........] - ETA: 16s - loss: 7.6208 - accuracy: 0.5030
18112/25000 [====================>.........] - ETA: 16s - loss: 7.6218 - accuracy: 0.5029
18144/25000 [====================>.........] - ETA: 16s - loss: 7.6201 - accuracy: 0.5030
18176/25000 [====================>.........] - ETA: 16s - loss: 7.6177 - accuracy: 0.5032
18208/25000 [====================>.........] - ETA: 16s - loss: 7.6169 - accuracy: 0.5032
18240/25000 [====================>.........] - ETA: 16s - loss: 7.6128 - accuracy: 0.5035
18272/25000 [====================>.........] - ETA: 16s - loss: 7.6121 - accuracy: 0.5036
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6113 - accuracy: 0.5036
18336/25000 [=====================>........] - ETA: 15s - loss: 7.6131 - accuracy: 0.5035
18368/25000 [=====================>........] - ETA: 15s - loss: 7.6140 - accuracy: 0.5034
18400/25000 [=====================>........] - ETA: 15s - loss: 7.6158 - accuracy: 0.5033
18432/25000 [=====================>........] - ETA: 15s - loss: 7.6175 - accuracy: 0.5032
18464/25000 [=====================>........] - ETA: 15s - loss: 7.6143 - accuracy: 0.5034
18496/25000 [=====================>........] - ETA: 15s - loss: 7.6194 - accuracy: 0.5031
18528/25000 [=====================>........] - ETA: 15s - loss: 7.6194 - accuracy: 0.5031
18560/25000 [=====================>........] - ETA: 15s - loss: 7.6146 - accuracy: 0.5034
18592/25000 [=====================>........] - ETA: 15s - loss: 7.6130 - accuracy: 0.5035
18624/25000 [=====================>........] - ETA: 15s - loss: 7.6115 - accuracy: 0.5036
18656/25000 [=====================>........] - ETA: 15s - loss: 7.6165 - accuracy: 0.5033
18688/25000 [=====================>........] - ETA: 15s - loss: 7.6141 - accuracy: 0.5034
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6085 - accuracy: 0.5038
18752/25000 [=====================>........] - ETA: 14s - loss: 7.6102 - accuracy: 0.5037
18784/25000 [=====================>........] - ETA: 14s - loss: 7.6078 - accuracy: 0.5038
18816/25000 [=====================>........] - ETA: 14s - loss: 7.6096 - accuracy: 0.5037
18848/25000 [=====================>........] - ETA: 14s - loss: 7.6105 - accuracy: 0.5037
18880/25000 [=====================>........] - ETA: 14s - loss: 7.6106 - accuracy: 0.5037
18912/25000 [=====================>........] - ETA: 14s - loss: 7.6123 - accuracy: 0.5035
18944/25000 [=====================>........] - ETA: 14s - loss: 7.6124 - accuracy: 0.5035
18976/25000 [=====================>........] - ETA: 14s - loss: 7.6109 - accuracy: 0.5036
19008/25000 [=====================>........] - ETA: 14s - loss: 7.6093 - accuracy: 0.5037
19040/25000 [=====================>........] - ETA: 14s - loss: 7.6119 - accuracy: 0.5036
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6111 - accuracy: 0.5036
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6104 - accuracy: 0.5037
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6105 - accuracy: 0.5037
19168/25000 [======================>.......] - ETA: 13s - loss: 7.6106 - accuracy: 0.5037
19200/25000 [======================>.......] - ETA: 13s - loss: 7.6131 - accuracy: 0.5035
19232/25000 [======================>.......] - ETA: 13s - loss: 7.6100 - accuracy: 0.5037
19264/25000 [======================>.......] - ETA: 13s - loss: 7.6117 - accuracy: 0.5036
19296/25000 [======================>.......] - ETA: 13s - loss: 7.6134 - accuracy: 0.5035
19328/25000 [======================>.......] - ETA: 13s - loss: 7.6111 - accuracy: 0.5036
19360/25000 [======================>.......] - ETA: 13s - loss: 7.6104 - accuracy: 0.5037
19392/25000 [======================>.......] - ETA: 13s - loss: 7.6089 - accuracy: 0.5038
19424/25000 [======================>.......] - ETA: 13s - loss: 7.6098 - accuracy: 0.5037
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6115 - accuracy: 0.5036
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6092 - accuracy: 0.5037
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6085 - accuracy: 0.5038
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6070 - accuracy: 0.5039
19584/25000 [======================>.......] - ETA: 12s - loss: 7.6071 - accuracy: 0.5039
19616/25000 [======================>.......] - ETA: 12s - loss: 7.6080 - accuracy: 0.5038
19648/25000 [======================>.......] - ETA: 12s - loss: 7.6089 - accuracy: 0.5038
19680/25000 [======================>.......] - ETA: 12s - loss: 7.6097 - accuracy: 0.5037
19712/25000 [======================>.......] - ETA: 12s - loss: 7.6122 - accuracy: 0.5036
19744/25000 [======================>.......] - ETA: 12s - loss: 7.6107 - accuracy: 0.5036
19776/25000 [======================>.......] - ETA: 12s - loss: 7.6147 - accuracy: 0.5034
19808/25000 [======================>.......] - ETA: 12s - loss: 7.6109 - accuracy: 0.5036
19840/25000 [======================>.......] - ETA: 12s - loss: 7.6141 - accuracy: 0.5034
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6157 - accuracy: 0.5033
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6189 - accuracy: 0.5031
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6220 - accuracy: 0.5029
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6205 - accuracy: 0.5030
20000/25000 [=======================>......] - ETA: 11s - loss: 7.6168 - accuracy: 0.5033
20032/25000 [=======================>......] - ETA: 11s - loss: 7.6207 - accuracy: 0.5030
20064/25000 [=======================>......] - ETA: 11s - loss: 7.6231 - accuracy: 0.5028
20096/25000 [=======================>......] - ETA: 11s - loss: 7.6269 - accuracy: 0.5026
20128/25000 [=======================>......] - ETA: 11s - loss: 7.6232 - accuracy: 0.5028
20160/25000 [=======================>......] - ETA: 11s - loss: 7.6225 - accuracy: 0.5029
20192/25000 [=======================>......] - ETA: 11s - loss: 7.6211 - accuracy: 0.5030
20224/25000 [=======================>......] - ETA: 11s - loss: 7.6226 - accuracy: 0.5029
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6227 - accuracy: 0.5029
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6243 - accuracy: 0.5028
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6244 - accuracy: 0.5028
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6252 - accuracy: 0.5027
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6260 - accuracy: 0.5026
20416/25000 [=======================>......] - ETA: 10s - loss: 7.6276 - accuracy: 0.5025
20448/25000 [=======================>......] - ETA: 10s - loss: 7.6291 - accuracy: 0.5024
20480/25000 [=======================>......] - ETA: 10s - loss: 7.6299 - accuracy: 0.5024
20512/25000 [=======================>......] - ETA: 10s - loss: 7.6255 - accuracy: 0.5027
20544/25000 [=======================>......] - ETA: 10s - loss: 7.6256 - accuracy: 0.5027
20576/25000 [=======================>......] - ETA: 10s - loss: 7.6264 - accuracy: 0.5026
20608/25000 [=======================>......] - ETA: 10s - loss: 7.6250 - accuracy: 0.5027
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6250 - accuracy: 0.5027
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6206 - accuracy: 0.5030
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6192 - accuracy: 0.5031
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6215 - accuracy: 0.5029
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6201 - accuracy: 0.5030
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6231 - accuracy: 0.5028
20832/25000 [=======================>......] - ETA: 9s - loss: 7.6225 - accuracy: 0.5029 
20864/25000 [========================>.....] - ETA: 9s - loss: 7.6255 - accuracy: 0.5027
20896/25000 [========================>.....] - ETA: 9s - loss: 7.6292 - accuracy: 0.5024
20928/25000 [========================>.....] - ETA: 9s - loss: 7.6271 - accuracy: 0.5026
20960/25000 [========================>.....] - ETA: 9s - loss: 7.6271 - accuracy: 0.5026
20992/25000 [========================>.....] - ETA: 9s - loss: 7.6250 - accuracy: 0.5027
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6236 - accuracy: 0.5028
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6229 - accuracy: 0.5028
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6237 - accuracy: 0.5028
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6223 - accuracy: 0.5029
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6260 - accuracy: 0.5026
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6239 - accuracy: 0.5028
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6254 - accuracy: 0.5027
21248/25000 [========================>.....] - ETA: 8s - loss: 7.6240 - accuracy: 0.5028
21280/25000 [========================>.....] - ETA: 8s - loss: 7.6241 - accuracy: 0.5028
21312/25000 [========================>.....] - ETA: 8s - loss: 7.6278 - accuracy: 0.5025
21344/25000 [========================>.....] - ETA: 8s - loss: 7.6264 - accuracy: 0.5026
21376/25000 [========================>.....] - ETA: 8s - loss: 7.6243 - accuracy: 0.5028
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6222 - accuracy: 0.5029
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6216 - accuracy: 0.5029
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6245 - accuracy: 0.5027
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6246 - accuracy: 0.5027
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6246 - accuracy: 0.5027
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6247 - accuracy: 0.5027
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6283 - accuracy: 0.5025
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6262 - accuracy: 0.5026
21664/25000 [========================>.....] - ETA: 7s - loss: 7.6263 - accuracy: 0.5026
21696/25000 [=========================>....] - ETA: 7s - loss: 7.6299 - accuracy: 0.5024
21728/25000 [=========================>....] - ETA: 7s - loss: 7.6285 - accuracy: 0.5025
21760/25000 [=========================>....] - ETA: 7s - loss: 7.6307 - accuracy: 0.5023
21792/25000 [=========================>....] - ETA: 7s - loss: 7.6307 - accuracy: 0.5023
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6322 - accuracy: 0.5022
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6336 - accuracy: 0.5022
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6351 - accuracy: 0.5021
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6344 - accuracy: 0.5021
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6366 - accuracy: 0.5020
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6380 - accuracy: 0.5019
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6381 - accuracy: 0.5019
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6374 - accuracy: 0.5019
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6402 - accuracy: 0.5017
22112/25000 [=========================>....] - ETA: 6s - loss: 7.6417 - accuracy: 0.5016
22144/25000 [=========================>....] - ETA: 6s - loss: 7.6431 - accuracy: 0.5015
22176/25000 [=========================>....] - ETA: 6s - loss: 7.6431 - accuracy: 0.5015
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6418 - accuracy: 0.5016
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6425 - accuracy: 0.5016
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6411 - accuracy: 0.5017
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6439 - accuracy: 0.5015
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6433 - accuracy: 0.5015
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6461 - accuracy: 0.5013
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6454 - accuracy: 0.5014
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6461 - accuracy: 0.5013
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6482 - accuracy: 0.5012
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6469 - accuracy: 0.5013
22528/25000 [==========================>...] - ETA: 5s - loss: 7.6503 - accuracy: 0.5011
22560/25000 [==========================>...] - ETA: 5s - loss: 7.6503 - accuracy: 0.5011
22592/25000 [==========================>...] - ETA: 5s - loss: 7.6503 - accuracy: 0.5011
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6490 - accuracy: 0.5011
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6483 - accuracy: 0.5012
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6477 - accuracy: 0.5012
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6511 - accuracy: 0.5010
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6484 - accuracy: 0.5012
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6464 - accuracy: 0.5013
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6465 - accuracy: 0.5013
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6472 - accuracy: 0.5013
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6472 - accuracy: 0.5013
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6465 - accuracy: 0.5013
22944/25000 [==========================>...] - ETA: 4s - loss: 7.6499 - accuracy: 0.5011
22976/25000 [==========================>...] - ETA: 4s - loss: 7.6479 - accuracy: 0.5012
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6486 - accuracy: 0.5012
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6506 - accuracy: 0.5010
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6527 - accuracy: 0.5009
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6547 - accuracy: 0.5008
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6567 - accuracy: 0.5006
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6560 - accuracy: 0.5007
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6541 - accuracy: 0.5008
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6554 - accuracy: 0.5007
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6548 - accuracy: 0.5008
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6581 - accuracy: 0.5006
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6568 - accuracy: 0.5006
23360/25000 [===========================>..] - ETA: 3s - loss: 7.6568 - accuracy: 0.5006
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6581 - accuracy: 0.5006
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6588 - accuracy: 0.5005
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6588 - accuracy: 0.5005
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6614 - accuracy: 0.5003
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6608 - accuracy: 0.5004
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6595 - accuracy: 0.5005
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6614 - accuracy: 0.5003
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6588 - accuracy: 0.5005
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6569 - accuracy: 0.5006
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6569 - accuracy: 0.5006
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6582 - accuracy: 0.5005
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6595 - accuracy: 0.5005
23776/25000 [===========================>..] - ETA: 2s - loss: 7.6602 - accuracy: 0.5004
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6595 - accuracy: 0.5005
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6583 - accuracy: 0.5005
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6628 - accuracy: 0.5002
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6653 - accuracy: 0.5001
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
24192/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24224/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24256/25000 [============================>.] - ETA: 1s - loss: 7.6673 - accuracy: 0.5000
24288/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24320/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24352/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24384/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24416/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24448/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24480/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24512/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24544/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24576/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24608/25000 [============================>.] - ETA: 0s - loss: 7.6616 - accuracy: 0.5003
24640/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24672/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24704/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24736/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24768/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24800/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24832/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24864/25000 [============================>.] - ETA: 0s - loss: 7.6611 - accuracy: 0.5004
24896/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24928/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24960/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 71s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f21278b6a58> 

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
 [-0.07613508  0.06631394 -0.05783654 -0.00583889 -0.04578237 -0.06309772]
 [-0.00559288  0.05951897  0.08409068 -0.06969808  0.20768459 -0.06049733]
 [ 0.14693204 -0.12050503  0.12265566  0.24581191  0.29460818  0.09222275]
 [ 0.21451294 -0.19157378  0.12634599  0.20154497 -0.36093423 -0.17086568]
 [ 0.13588297 -0.09361067 -0.00897046  0.03628718  0.00934435 -0.0743933 ]
 [ 0.28877857  0.55514061 -0.60310823 -0.36443812  0.28151223  0.08806542]
 [-0.38559839 -0.28961599  0.15082598  0.23979926  0.3341563   0.77416182]
 [ 0.35660154 -0.08174607  0.21115376  0.17647083  1.05464101  0.33752322]
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
{'loss': 0.47832459956407547, 'loss_history': []}

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
{'loss': 0.515231229364872, 'loss_history': []}

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
 40%|████      | 2/5 [00:48<01:12, 24.26s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.2512140293589958, 'embedding_size_factor': 0.9101372770193871, 'layers.choice': 1, 'learning_rate': 0.0010205790008293776, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 0.00018001425630896528} and reward: 0.3798
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd0\x13\xe4\x02\x19+YX\x15\x00\x00\x00embedding_size_factorq\x03G?\xed\x1f\xd85\xf5f{X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?P\xb8\x9e[\x9euZX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?'\x98F\xaf\xf4M\xdbu." and reward: 0.3798
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd0\x13\xe4\x02\x19+YX\x15\x00\x00\x00embedding_size_factorq\x03G?\xed\x1f\xd85\xf5f{X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?P\xb8\x9e[\x9euZX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?'\x98F\xaf\xf4M\xdbu." and reward: 0.3798
 60%|██████    | 3/5 [02:36<01:38, 49.26s/it] 60%|██████    | 3/5 [02:36<01:44, 52.04s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.2275148066880115, 'embedding_size_factor': 1.3713080204839772, 'layers.choice': 1, 'learning_rate': 0.00027277788060627104, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 5.785012943588005e-06} and reward: 0.3618
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xcd\x1f4\x87\nV\xb8X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\xf0\xe0\xad\xcb\x89\xf4X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?1\xe0t\x13\x86\xd5\x0eX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xd8C\x9c?\xe4NEu.' and reward: 0.3618
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xcd\x1f4\x87\nV\xb8X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\xf0\xe0\xad\xcb\x89\xf4X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?1\xe0t\x13\x86\xd5\x0eX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xd8C\x9c?\xe4NEu.' and reward: 0.3618
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 244.66006016731262
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.77s of the -127.19s of remaining time.
Ensemble size: 49
Ensemble weights: 
[0.57142857 0.20408163 0.2244898 ]
	0.3908	 = Validation accuracy score
	0.89s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 248.11s ...
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

