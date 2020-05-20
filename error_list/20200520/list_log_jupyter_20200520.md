## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py


### Error 1, [Traceback at line 44](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L44)<br />44..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
<br />[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m    127[0m [0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'mlmodels.model_sklearn.sklearn'
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 2, [Traceback at line 65](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L65)<br />65..[0;31mIndexError[0m                                Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mIndexError[0m: tuple index out of range
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 3, [Traceback at line 75](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L75)<br />75..[0;31mNameError[0m                                 Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_svm.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      3[0m [0;34m[0m[0m
<br />[1;32m      4[0m [0mmodel_uri[0m    [0;34m=[0m [0;34m"model_sklearn.sklearn.py"[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m----> 5[0;31m [0mmodule[0m        [0;34m=[0m  [0mmodule_load[0m[0;34m([0m [0mmodel_uri[0m[0;34m=[0m [0mmodel_uri[0m [0;34m)[0m                           [0;31m# Load file definition[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      6[0m [0;34m[0m[0m
<br />[1;32m      7[0m model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
<br />
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     87[0m [0;34m[0m[0m
<br />[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     90[0m [0;34m[0m[0m
<br />[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm.ipynb 
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 4, [Traceback at line 100](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L100)<br />100..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//lightgbm.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      4[0m [0mdata_path[0m [0;34m=[0m [0;34m'lightgbm_titanic.json'[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      5[0m [0;34m[0m[0m
<br />[0;32m----> 6[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mdata_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      7[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      8[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpdict[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'lightgbm_titanic.json'
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn_titanic_randomForest.ipynb 
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 5, [Traceback at line 118](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L118)<br />118..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
<br />[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m    127[0m [0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'mlmodels.model_sklearn.sklearn'
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 6, [Traceback at line 139](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L139)<br />139..[0;31mIndexError[0m                                Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mIndexError[0m: tuple index out of range
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 7, [Traceback at line 149](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L149)<br />149..[0;31mNameError[0m                                 Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      2[0m [0;34m[0m[0m
<br />[1;32m      3[0m [0mmodel_uri[0m    [0;34m=[0m [0;34m"model_sklearn.sklearn.py"[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m----> 4[0;31m [0mmodule[0m        [0;34m=[0m  [0mmodule_load[0m[0;34m([0m [0mmodel_uri[0m[0;34m=[0m [0mmodel_uri[0m [0;34m)[0m                           [0;31m# Load file definition[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      5[0m [0;34m[0m[0m
<br />[1;32m      6[0m model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
<br />
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     87[0m [0;34m[0m[0m
<br />[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     90[0m [0;34m[0m[0m
<br />[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//timeseries_m5_deepar.ipynb 
<br />
<br />UsageError: Line magic function `%%capture` not found.



### Error 8, [Traceback at line 183](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L183)<br />183..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'google.colab'
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_home_retail.ipynb 
<br />
<br />Deprecaton set to False
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 9, [Traceback at line 199](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L199)<br />199..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//lightgbm_home_retail.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      1[0m [0mdata_path[0m [0;34m=[0m [0;34m'hyper_lightgbm_home_retail.json'[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      2[0m [0;34m[0m[0m
<br />[0;32m----> 3[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mdata_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      4[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      5[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpdict[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'hyper_lightgbm_home_retail.json'
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//keras_charcnn_reuters.ipynb 
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 10, [Traceback at line 217](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L217)<br />217..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//keras_charcnn_reuters.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[0;32m----> 1[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mconfig_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      2[0m [0mmodel_pars[0m      [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpars[0m[0;34m[[0m[0;34m'model_pars'[0m[0;34m][0m [0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      3[0m [0mdata_pars[0m       [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpars[0m[0;34m[[0m[0;34m'data_pars'[0m[0;34m][0m [0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      4[0m [0mcompute_pars[0m    [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpars[0m[0;34m[[0m[0;34m'compute_pars'[0m[0;34m][0m [0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      5[0m [0mout_pars[0m        [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpars[0m[0;34m[[0m[0;34m'out_pars'[0m[0;34m][0m [0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'reuters_charcnn.json'
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//gluon_automl.ipynb 
<br />
<br />/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB
<br />  Optimizer.opt_registry[name].__name__))
<br />Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv | Columns = 15 / 15 | Rows = 39073 -> 39073
<br />Warning: `hyperparameter_tune=True` is currently experimental and may cause the process to hang. Setting `auto_stack=True` instead is recommended to achieve maximum quality models.
<br />Beginning AutoGluon training ... Time limit = 120s
<br />AutoGluon will save models to dataset/
<br />Train Data Rows:    39073
<br />Train Data Columns: 15
<br />Preprocessing data ...
<br />Here are the first 10 unique label values in your data:  [' Tech-support' ' Transport-moving' ' Other-service' ' ?'
<br /> ' Handlers-cleaners' ' Sales' ' Craft-repair' ' Adm-clerical'
<br /> ' Exec-managerial' ' Prof-specialty']
<br />AutoGluon infers your prediction problem is: multiclass  (because dtype of label-column == object)
<br />If this is wrong, please specify `problem_type` argument in fit() instead (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])
<br />
<br />Feature Generator processed 39073 data points with 14 features
<br />Original Features:
<br />	int features: 6
<br />	object features: 8
<br />Generated Features:
<br />	int features: 0
<br />All Features:
<br />	int features: 6
<br />	object features: 8
<br />	Data preprocessing and feature engineering runtime = 0.22s ...
<br />AutoGluon will gauge predictive performance using evaluation metric: accuracy
<br />To change this, specify the eval_metric argument of fit()
<br />AutoGluon will early stop models using evaluation metric: accuracy
<br />Saving dataset/learner.pkl
<br />Beginning hyperparameter tuning for Gradient Boosting Model...
<br />Hyperparameter search space for Gradient Boosting Model: 
<br />num_leaves:   Int: lower=26, upper=30
<br />learning_rate:   Real: lower=0.005, upper=0.2
<br />feature_fraction:   Real: lower=0.75, upper=1.0
<br />min_data_in_leaf:   Int: lower=2, upper=30
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 360, in train_single_full
<br />    Y_train=y_train, Y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/models/lgb/lgb_model.py", line 283, in hyperparameter_tune
<br />    directory=directory, lgb_model=self, **params_copy)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/decorator.py", line 69, in register_args
<br />    self.update(**kwvars)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/decorator.py", line 79, in update
<br />    hp = v.get_hp(name=k)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/space.py", line 451, in get_hp
<br />    default_value=self._default)
<br />  File "ConfigSpace/hyperparameters.pyx", line 773, in ConfigSpace.hyperparameters.UniformIntegerHyperparameter.__init__
<br />  File "ConfigSpace/hyperparameters.pyx", line 843, in ConfigSpace.hyperparameters.UniformIntegerHyperparameter.check_default
<br />Warning: Exception caused LightGBMClassifier to fail during hyperparameter tuning... Skipping this model.



### Error 11, [Traceback at line 282](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L282)<br />282..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/trainer/abstract_trainer.py", line 360, in train_single_full
<br />    Y_train=y_train, Y_test=y_test, scheduler_options=(self.scheduler_func, self.scheduler_options), verbosity=self.verbosity)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/utils/tabular/ml/models/lgb/lgb_model.py", line 283, in hyperparameter_tune
<br />    directory=directory, lgb_model=self, **params_copy)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/decorator.py", line 69, in register_args
<br />    self.update(**kwvars)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/decorator.py", line 79, in update
<br />    hp = v.get_hp(name=k)
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/autogluon/core/space.py", line 451, in get_hp
<br />    default_value=self._default)
<br />  File "ConfigSpace/hyperparameters.pyx", line 773, in ConfigSpace.hyperparameters.UniformIntegerHyperparameter.__init__
<br />  File "ConfigSpace/hyperparameters.pyx", line 843, in ConfigSpace.hyperparameters.UniformIntegerHyperparameter.check_default
<br />ValueError: Illegal default value 36



### Error 12, [Traceback at line 485](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L485)<br />485..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/models.py", line 523, in main
<br />    test_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/models.py", line 453, in test_cli
<br />    test_module(arg.model_uri, param_pars=param_pars)  # '1_lstm'
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_gluon/gluon_automl.py", line 109, in get_params
<br />    return model_pars, data_pars, compute_pars, out_pars
<br />UnboundLocalError: local variable 'model_pars' referenced before assignment



### Error 13, [Traceback at line 506](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L506)<br />506..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'google.colab'
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//tensorflow_1_lstm.ipynb 
<br />
<br />/home/runner/work/mlmodels/mlmodels
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />non-resource variables are not supported in the long term
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
<br />https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />          0         1         2         3         4         5
<br />0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
<br />1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
<br />2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
<br />3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
<br />4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
<br />https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />          0         1         2         3         4         5
<br />0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
<br />1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
<br />2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
<br />3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
<br />4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
<br />5  0.745928  0.883387  0.838176  0.904464  0.904464  0.370110
<br />6  1.000000  0.881878  0.467996  0.486496  0.486496  1.000000
<br />7  0.216516  0.077549  0.433808  0.329598  0.329598  0.318466
<br />8  0.195249  0.000000  0.000000  0.000000  0.000000  0.671960
<br />9  0.000000  0.173783  0.369041  0.411721  0.411721  0.304384
<br />test
<br />
<br />  #### Module init   ############################################ 
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />non-resource variables are not supported in the long term
<br />
<br />  <module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_tf/1_lstm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.model_tf.1_lstm.Model object at 0x7effb789d9b0> 
<br />
<br />  #### Fit   ######################################################## 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />          0         1         2         3         4         5
<br />0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
<br />1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
<br />2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
<br />3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
<br />4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
<br />
<br />  #### Predict   #################################################### 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />          0         1         2         3         4         5
<br />0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
<br />1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
<br />2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
<br />3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
<br />4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
<br />5  0.745928  0.883387  0.838176  0.904464  0.904464  0.370110
<br />6  1.000000  0.881878  0.467996  0.486496  0.486496  1.000000
<br />7  0.216516  0.077549  0.433808  0.329598  0.329598  0.318466
<br />8  0.195249  0.000000  0.000000  0.000000  0.000000  0.671960
<br />9  0.000000  0.173783  0.369041  0.411721  0.411721  0.304384
<br />[[ 0.          0.          0.          0.          0.          0.        ]
<br /> [ 0.04655149  0.02888693  0.0615714   0.00501382  0.00569917 -0.07453486]
<br /> [ 0.21008714  0.02798664 -0.23469928  0.08178039 -0.03203796 -0.14887488]
<br /> [-0.08203344 -0.03239747 -0.16407555  0.11306866  0.23655026  0.06360702]
<br /> [-0.03158426 -0.04173462 -0.07089001 -0.5159812  -0.04174392  0.25128227]
<br /> [ 0.29538631  0.01834104  0.17013898  0.42228481  0.47903964  0.21885738]
<br /> [ 0.21135049  0.17720319  0.47402799  0.01367701 -0.22313513 -0.34312996]
<br /> [ 0.11012827  0.40497413  0.58963829  0.74406642  0.49763009  0.22554512]
<br /> [-0.25714141 -0.18077397  0.05286252 -0.37468287  0.82782304  0.21389796]
<br /> [ 0.          0.          0.          0.          0.          0.        ]]
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />model_tf/1_lstm.py
<br />model_tf.1_lstm.py
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />
<br />  #### Model init  ############################################# 
<br />
<br />  #### Model fit   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />          0         1         2         3         4         5
<br />0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
<br />1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
<br />2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
<br />3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
<br />4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
<br />
<br />  #### Predict   ##################################################### 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />          0         1         2         3         4         5
<br />0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
<br />1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
<br />2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
<br />3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
<br />4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
<br />5  0.745928  0.883387  0.838176  0.904464  0.904464  0.370110
<br />6  1.000000  0.881878  0.467996  0.486496  0.486496  1.000000
<br />7  0.216516  0.077549  0.433808  0.329598  0.329598  0.318466
<br />8  0.195249  0.000000  0.000000  0.000000  0.000000  0.671960
<br />9  0.000000  0.173783  0.369041  0.411721  0.411721  0.304384
<br />
<br />  #### metrics   ##################################################### 
<br />{'loss': 0.42624923400580883, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-20 14:31:57.058628: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
<br />
<br />Key Variable not found in checkpoint
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 523, in main
<br />    test_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 455, in test_cli
<br />    test(arg.model_uri)  # '1_lstm'
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 189, in test
<br />    module.test()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 320, in test
<br />    session = load(out_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
<br />    return load_tf(load_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
<br />    saver      = tf.compat.v1.train.Saver()
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 828, in __init__
<br />    self.build()
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 840, in build
<br />    self._build(self._filename, build_save=True, build_restore=True)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 878, in _build
<br />    build_restore=build_restore)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 508, in _build_internal
<br />    restore_sequentially, reshape)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 328, in _AddRestoreOps
<br />    restore_sequentially)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 575, in bulk_restore
<br />    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_io_ops.py", line 1696, in restore_v2
<br />    name=name)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
<br />    op_def=op_def)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
<br />    return func(*args, **kwargs)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
<br />    attrs, op_def, compute_device)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
<br />    op_def=op_def)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
<br />    self._traceback = tf_stack.extract_stack()
<br />
<br />model_tf/1_lstm.py
<br />model_tf.1_lstm.py
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />
<br />  #### Model init  ############################################# 
<br />
<br />  #### Model fit   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />          0         1         2         3         4         5
<br />0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
<br />1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
<br />2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
<br />3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
<br />4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
<br />
<br />  #### Predict   ##################################################### 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/dataset/timeseries/GOOG-year.csv
<br />         Date        Open        High  ...       Close   Adj Close   Volume
<br />0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
<br />1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
<br />2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
<br />3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
<br />4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800
<br />
<br />[5 rows x 7 columns]
<br />          0         1         2         3         4         5
<br />0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
<br />1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
<br />2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
<br />3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
<br />4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
<br />5  0.745928  0.883387  0.838176  0.904464  0.904464  0.370110
<br />6  1.000000  0.881878  0.467996  0.486496  0.486496  1.000000
<br />7  0.216516  0.077549  0.433808  0.329598  0.329598  0.318466
<br />8  0.195249  0.000000  0.000000  0.000000  0.000000  0.671960
<br />9  0.000000  0.173783  0.369041  0.411721  0.411721  0.304384
<br />
<br />  #### metrics   ##################################################### 
<br />{'loss': 0.5136057361960411, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-20 14:31:58.073114: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/ztest/model_tf/1_lstm/model'}
<br />Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
<br />
<br />Key Variable not found in checkpoint
<br />	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]
<br />
<br />Original stack trace for 'save_1/RestoreV2':
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 523, in main
<br />    test_cli(arg)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 457, in test_cli
<br />    test_global(arg.model_uri)  # '1_lstm'
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 200, in test_global
<br />    module.test()
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 320, in test
<br />    session = load(out_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf/1_lstm.py", line 199, in load
<br />    return load_tf(load_pars)
<br />  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
<br />    saver      = tf.compat.v1.train.Saver()
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 828, in __init__
<br />    self.build()
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 840, in build
<br />    self._build(self._filename, build_save=True, build_restore=True)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 878, in _build
<br />    build_restore=build_restore)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 508, in _build_internal
<br />    restore_sequentially, reshape)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 328, in _AddRestoreOps
<br />    restore_sequentially)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 575, in bulk_restore
<br />    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_io_ops.py", line 1696, in restore_v2
<br />    name=name)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
<br />    op_def=op_def)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
<br />    return func(*args, **kwargs)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
<br />    attrs, op_def, compute_device)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
<br />    op_def=op_def)
<br />  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
<br />    self._traceback = tf_stack.extract_stack()
<br />
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vision_mnist.ipynb 
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 14, [Traceback at line 890](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L890)<br />890..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'google.colab'
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_glass.ipynb 
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 15, [Traceback at line 905](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L905)<br />905..[0;31mNameError[0m                                 Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      8[0m [0;32mimport[0m [0mjson[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      9[0m [0;34m[0m[0m
<br />[0;32m---> 10[0;31m [0mprint[0m[0;34m([0m [0mos[0m[0;34m.[0m[0mgetcwd[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m
<br />[0;31mNameError[0m: name 'os' is not defined
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//keras-textcnn.ipynb 
<br />
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />If using Keras pass *_constraint arguments to layers.
<br />Model: "model_1"
<br />__________________________________________________________________________________________________
<br />Layer (type)                    Output Shape         Param #     Connected to                     
<br />==================================================================================================
<br />input_1 (InputLayer)            (None, 400)          0                                            
<br />__________________________________________________________________________________________________
<br />embedding_1 (Embedding)         (None, 400, 50)      500         input_1[0][0]                    
<br />__________________________________________________________________________________________________
<br />conv1d_1 (Conv1D)               (None, 398, 128)     19328       embedding_1[0][0]                
<br />__________________________________________________________________________________________________
<br />conv1d_2 (Conv1D)               (None, 397, 128)     25728       embedding_1[0][0]                
<br />__________________________________________________________________________________________________
<br />conv1d_3 (Conv1D)               (None, 396, 128)     32128       embedding_1[0][0]                
<br />__________________________________________________________________________________________________
<br />global_max_pooling1d_1 (GlobalM (None, 128)          0           conv1d_1[0][0]                   
<br />__________________________________________________________________________________________________
<br />global_max_pooling1d_2 (GlobalM (None, 128)          0           conv1d_2[0][0]                   
<br />__________________________________________________________________________________________________
<br />global_max_pooling1d_3 (GlobalM (None, 128)          0           conv1d_3[0][0]                   
<br />__________________________________________________________________________________________________
<br />concatenate_1 (Concatenate)     (None, 384)          0           global_max_pooling1d_1[0][0]     
<br />                                                                 global_max_pooling1d_2[0][0]     
<br />                                                                 global_max_pooling1d_3[0][0]     
<br />__________________________________________________________________________________________________
<br />dense_1 (Dense)                 (None, 1)            385         concatenate_1[0][0]              
<br />==================================================================================================
<br />Total params: 78,069
<br />Trainable params: 78,069
<br />Non-trainable params: 0
<br />__________________________________________________________________________________________________
<br />Loading data...
<br />Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz
<br />
<br />    8192/17464789 [..............................] - ETA: 0s
<br /> 1130496/17464789 [>.............................] - ETA: 0s
<br /> 5513216/17464789 [========>.....................] - ETA: 0s
<br />13197312/17464789 [=====================>........] - ETA: 0s
<br />17465344/17464789 [==============================] - 0s 0us/step
<br />Pad sequences (samples x time)...
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />Use tf.where in 2.0, which has the same broadcast rule as np.where
<br />2020-05-20 14:32:08.310264: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
<br />2020-05-20 14:32:08.314697: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
<br />2020-05-20 14:32:08.314849: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561ef22a7020 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
<br />2020-05-20 14:32:08.314864: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
<br />
<br />Train on 25000 samples, validate on 25000 samples
<br />Epoch 1/1
<br />
<br />   32/25000 [..............................] - ETA: 4:06 - loss: 9.1041 - accuracy: 0.4062
<br />   64/25000 [..............................] - ETA: 2:35 - loss: 8.1458 - accuracy: 0.4688
<br />   96/25000 [..............................] - ETA: 2:02 - loss: 8.1458 - accuracy: 0.4688
<br />  128/25000 [..............................] - ETA: 1:45 - loss: 7.7864 - accuracy: 0.4922
<br />  160/25000 [..............................] - ETA: 1:36 - loss: 8.2416 - accuracy: 0.4625
<br />  192/25000 [..............................] - ETA: 1:29 - loss: 8.1458 - accuracy: 0.4688
<br />  224/25000 [..............................] - ETA: 1:24 - loss: 8.0089 - accuracy: 0.4777
<br />  256/25000 [..............................] - ETA: 1:20 - loss: 8.3854 - accuracy: 0.4531
<br />  288/25000 [..............................] - ETA: 1:17 - loss: 8.4652 - accuracy: 0.4479
<br />  320/25000 [..............................] - ETA: 1:15 - loss: 8.2416 - accuracy: 0.4625
<br />  352/25000 [..............................] - ETA: 1:14 - loss: 8.2329 - accuracy: 0.4631
<br />  384/25000 [..............................] - ETA: 1:12 - loss: 8.0659 - accuracy: 0.4740
<br />  416/25000 [..............................] - ETA: 1:11 - loss: 8.0721 - accuracy: 0.4736
<br />  448/25000 [..............................] - ETA: 1:10 - loss: 7.9062 - accuracy: 0.4844
<br />  480/25000 [..............................] - ETA: 1:09 - loss: 7.8902 - accuracy: 0.4854
<br />  512/25000 [..............................] - ETA: 1:08 - loss: 7.9960 - accuracy: 0.4785
<br />  544/25000 [..............................] - ETA: 1:07 - loss: 8.1740 - accuracy: 0.4669
<br />  576/25000 [..............................] - ETA: 1:06 - loss: 8.1458 - accuracy: 0.4688
<br />  608/25000 [..............................] - ETA: 1:06 - loss: 8.0449 - accuracy: 0.4753
<br />  640/25000 [..............................] - ETA: 1:05 - loss: 8.0260 - accuracy: 0.4766
<br />  672/25000 [..............................] - ETA: 1:05 - loss: 8.1230 - accuracy: 0.4702
<br />  704/25000 [..............................] - ETA: 1:04 - loss: 8.0804 - accuracy: 0.4730
<br />  736/25000 [..............................] - ETA: 1:04 - loss: 8.0208 - accuracy: 0.4769
<br />  768/25000 [..............................] - ETA: 1:03 - loss: 8.0260 - accuracy: 0.4766
<br />  800/25000 [..............................] - ETA: 1:03 - loss: 8.0116 - accuracy: 0.4775
<br />  832/25000 [..............................] - ETA: 1:02 - loss: 7.9062 - accuracy: 0.4844
<br />  864/25000 [>.............................] - ETA: 1:02 - loss: 7.8618 - accuracy: 0.4873
<br />  896/25000 [>.............................] - ETA: 1:02 - loss: 7.8035 - accuracy: 0.4911
<br />  928/25000 [>.............................] - ETA: 1:01 - loss: 7.7658 - accuracy: 0.4935
<br />  960/25000 [>.............................] - ETA: 1:01 - loss: 7.7465 - accuracy: 0.4948
<br />  992/25000 [>.............................] - ETA: 1:01 - loss: 7.7594 - accuracy: 0.4940
<br /> 1024/25000 [>.............................] - ETA: 1:01 - loss: 7.7864 - accuracy: 0.4922
<br /> 1056/25000 [>.............................] - ETA: 1:00 - loss: 7.7683 - accuracy: 0.4934
<br /> 1088/25000 [>.............................] - ETA: 1:00 - loss: 7.7935 - accuracy: 0.4917
<br /> 1120/25000 [>.............................] - ETA: 1:00 - loss: 7.8309 - accuracy: 0.4893
<br /> 1152/25000 [>.............................] - ETA: 1:00 - loss: 7.7997 - accuracy: 0.4913
<br /> 1184/25000 [>.............................] - ETA: 59s - loss: 7.8609 - accuracy: 0.4873 
<br /> 1216/25000 [>.............................] - ETA: 59s - loss: 7.8305 - accuracy: 0.4893
<br /> 1248/25000 [>.............................] - ETA: 59s - loss: 7.8141 - accuracy: 0.4904
<br /> 1280/25000 [>.............................] - ETA: 59s - loss: 7.7744 - accuracy: 0.4930
<br /> 1312/25000 [>.............................] - ETA: 59s - loss: 7.7835 - accuracy: 0.4924
<br /> 1344/25000 [>.............................] - ETA: 58s - loss: 7.8377 - accuracy: 0.4888
<br /> 1376/25000 [>.............................] - ETA: 58s - loss: 7.7781 - accuracy: 0.4927
<br /> 1408/25000 [>.............................] - ETA: 58s - loss: 7.8300 - accuracy: 0.4893
<br /> 1440/25000 [>.............................] - ETA: 58s - loss: 7.7944 - accuracy: 0.4917
<br /> 1472/25000 [>.............................] - ETA: 58s - loss: 7.7916 - accuracy: 0.4918
<br /> 1504/25000 [>.............................] - ETA: 58s - loss: 7.7686 - accuracy: 0.4934
<br /> 1536/25000 [>.............................] - ETA: 57s - loss: 7.7664 - accuracy: 0.4935
<br /> 1568/25000 [>.............................] - ETA: 57s - loss: 7.7546 - accuracy: 0.4943
<br /> 1600/25000 [>.............................] - ETA: 57s - loss: 7.7433 - accuracy: 0.4950
<br /> 1632/25000 [>.............................] - ETA: 57s - loss: 7.7136 - accuracy: 0.4969
<br /> 1664/25000 [>.............................] - ETA: 56s - loss: 7.7311 - accuracy: 0.4958
<br /> 1696/25000 [=>............................] - ETA: 56s - loss: 7.7480 - accuracy: 0.4947
<br /> 1728/25000 [=>............................] - ETA: 56s - loss: 7.7642 - accuracy: 0.4936
<br /> 1760/25000 [=>............................] - ETA: 56s - loss: 7.7450 - accuracy: 0.4949
<br /> 1792/25000 [=>............................] - ETA: 56s - loss: 7.7351 - accuracy: 0.4955
<br /> 1824/25000 [=>............................] - ETA: 56s - loss: 7.7255 - accuracy: 0.4962
<br /> 1856/25000 [=>............................] - ETA: 55s - loss: 7.7658 - accuracy: 0.4935
<br /> 1888/25000 [=>............................] - ETA: 55s - loss: 7.7560 - accuracy: 0.4942
<br /> 1920/25000 [=>............................] - ETA: 55s - loss: 7.7305 - accuracy: 0.4958
<br /> 1952/25000 [=>............................] - ETA: 55s - loss: 7.7138 - accuracy: 0.4969
<br /> 1984/25000 [=>............................] - ETA: 55s - loss: 7.7130 - accuracy: 0.4970
<br /> 2016/25000 [=>............................] - ETA: 55s - loss: 7.6742 - accuracy: 0.4995
<br /> 2048/25000 [=>............................] - ETA: 55s - loss: 7.6816 - accuracy: 0.4990
<br /> 2080/25000 [=>............................] - ETA: 54s - loss: 7.6961 - accuracy: 0.4981
<br /> 2112/25000 [=>............................] - ETA: 54s - loss: 7.6739 - accuracy: 0.4995
<br /> 2144/25000 [=>............................] - ETA: 54s - loss: 7.6595 - accuracy: 0.5005
<br /> 2176/25000 [=>............................] - ETA: 54s - loss: 7.6666 - accuracy: 0.5000
<br /> 2208/25000 [=>............................] - ETA: 54s - loss: 7.7083 - accuracy: 0.4973
<br /> 2240/25000 [=>............................] - ETA: 54s - loss: 7.7351 - accuracy: 0.4955
<br /> 2272/25000 [=>............................] - ETA: 54s - loss: 7.7004 - accuracy: 0.4978
<br /> 2304/25000 [=>............................] - ETA: 54s - loss: 7.6666 - accuracy: 0.5000
<br /> 2336/25000 [=>............................] - ETA: 54s - loss: 7.6338 - accuracy: 0.5021
<br /> 2368/25000 [=>............................] - ETA: 53s - loss: 7.6213 - accuracy: 0.5030
<br /> 2400/25000 [=>............................] - ETA: 53s - loss: 7.6411 - accuracy: 0.5017
<br /> 2432/25000 [=>............................] - ETA: 53s - loss: 7.6288 - accuracy: 0.5025
<br /> 2464/25000 [=>............................] - ETA: 53s - loss: 7.6417 - accuracy: 0.5016
<br /> 2496/25000 [=>............................] - ETA: 53s - loss: 7.6482 - accuracy: 0.5012
<br /> 2528/25000 [==>...........................] - ETA: 53s - loss: 7.6363 - accuracy: 0.5020
<br /> 2560/25000 [==>...........................] - ETA: 53s - loss: 7.6367 - accuracy: 0.5020
<br /> 2592/25000 [==>...........................] - ETA: 53s - loss: 7.6134 - accuracy: 0.5035
<br /> 2624/25000 [==>...........................] - ETA: 53s - loss: 7.6199 - accuracy: 0.5030
<br /> 2656/25000 [==>...........................] - ETA: 52s - loss: 7.6262 - accuracy: 0.5026
<br /> 2688/25000 [==>...........................] - ETA: 52s - loss: 7.6267 - accuracy: 0.5026
<br /> 2720/25000 [==>...........................] - ETA: 52s - loss: 7.6046 - accuracy: 0.5040
<br /> 2752/25000 [==>...........................] - ETA: 52s - loss: 7.5942 - accuracy: 0.5047
<br /> 2784/25000 [==>...........................] - ETA: 52s - loss: 7.5840 - accuracy: 0.5054
<br /> 2816/25000 [==>...........................] - ETA: 52s - loss: 7.5795 - accuracy: 0.5057
<br /> 2848/25000 [==>...........................] - ETA: 52s - loss: 7.5966 - accuracy: 0.5046
<br /> 2880/25000 [==>...........................] - ETA: 52s - loss: 7.5548 - accuracy: 0.5073
<br /> 2912/25000 [==>...........................] - ETA: 51s - loss: 7.5613 - accuracy: 0.5069
<br /> 2944/25000 [==>...........................] - ETA: 51s - loss: 7.5625 - accuracy: 0.5068
<br /> 2976/25000 [==>...........................] - ETA: 51s - loss: 7.5739 - accuracy: 0.5060
<br /> 3008/25000 [==>...........................] - ETA: 51s - loss: 7.5800 - accuracy: 0.5057
<br /> 3040/25000 [==>...........................] - ETA: 51s - loss: 7.5758 - accuracy: 0.5059
<br /> 3072/25000 [==>...........................] - ETA: 51s - loss: 7.5768 - accuracy: 0.5059
<br /> 3104/25000 [==>...........................] - ETA: 51s - loss: 7.5728 - accuracy: 0.5061
<br /> 3136/25000 [==>...........................] - ETA: 51s - loss: 7.5786 - accuracy: 0.5057
<br /> 3168/25000 [==>...........................] - ETA: 51s - loss: 7.5698 - accuracy: 0.5063
<br /> 3200/25000 [==>...........................] - ETA: 51s - loss: 7.5564 - accuracy: 0.5072
<br /> 3232/25000 [==>...........................] - ETA: 51s - loss: 7.5717 - accuracy: 0.5062
<br /> 3264/25000 [==>...........................] - ETA: 50s - loss: 7.5727 - accuracy: 0.5061
<br /> 3296/25000 [==>...........................] - ETA: 50s - loss: 7.5968 - accuracy: 0.5046
<br /> 3328/25000 [==>...........................] - ETA: 50s - loss: 7.6205 - accuracy: 0.5030
<br /> 3360/25000 [===>..........................] - ETA: 50s - loss: 7.6255 - accuracy: 0.5027
<br /> 3392/25000 [===>..........................] - ETA: 50s - loss: 7.6033 - accuracy: 0.5041
<br /> 3424/25000 [===>..........................] - ETA: 50s - loss: 7.5994 - accuracy: 0.5044
<br /> 3456/25000 [===>..........................] - ETA: 50s - loss: 7.6001 - accuracy: 0.5043
<br /> 3488/25000 [===>..........................] - ETA: 50s - loss: 7.5875 - accuracy: 0.5052
<br /> 3520/25000 [===>..........................] - ETA: 50s - loss: 7.5926 - accuracy: 0.5048
<br /> 3552/25000 [===>..........................] - ETA: 50s - loss: 7.5803 - accuracy: 0.5056
<br /> 3584/25000 [===>..........................] - ETA: 50s - loss: 7.5853 - accuracy: 0.5053
<br /> 3616/25000 [===>..........................] - ETA: 49s - loss: 7.5818 - accuracy: 0.5055
<br /> 3648/25000 [===>..........................] - ETA: 49s - loss: 7.5826 - accuracy: 0.5055
<br /> 3680/25000 [===>..........................] - ETA: 49s - loss: 7.5833 - accuracy: 0.5054
<br /> 3712/25000 [===>..........................] - ETA: 49s - loss: 7.5881 - accuracy: 0.5051
<br /> 3744/25000 [===>..........................] - ETA: 49s - loss: 7.5765 - accuracy: 0.5059
<br /> 3776/25000 [===>..........................] - ETA: 49s - loss: 7.5813 - accuracy: 0.5056
<br /> 3808/25000 [===>..........................] - ETA: 49s - loss: 7.5861 - accuracy: 0.5053
<br /> 3840/25000 [===>..........................] - ETA: 49s - loss: 7.5828 - accuracy: 0.5055
<br /> 3872/25000 [===>..........................] - ETA: 49s - loss: 7.5676 - accuracy: 0.5065
<br /> 3904/25000 [===>..........................] - ETA: 49s - loss: 7.5763 - accuracy: 0.5059
<br /> 3936/25000 [===>..........................] - ETA: 49s - loss: 7.5770 - accuracy: 0.5058
<br /> 3968/25000 [===>..........................] - ETA: 48s - loss: 7.5700 - accuracy: 0.5063
<br /> 4000/25000 [===>..........................] - ETA: 48s - loss: 7.5708 - accuracy: 0.5063
<br /> 4032/25000 [===>..........................] - ETA: 48s - loss: 7.5906 - accuracy: 0.5050
<br /> 4064/25000 [===>..........................] - ETA: 48s - loss: 7.5874 - accuracy: 0.5052
<br /> 4096/25000 [===>..........................] - ETA: 48s - loss: 7.5880 - accuracy: 0.5051
<br /> 4128/25000 [===>..........................] - ETA: 48s - loss: 7.5998 - accuracy: 0.5044
<br /> 4160/25000 [===>..........................] - ETA: 48s - loss: 7.6113 - accuracy: 0.5036
<br /> 4192/25000 [====>.........................] - ETA: 48s - loss: 7.6008 - accuracy: 0.5043
<br /> 4224/25000 [====>.........................] - ETA: 48s - loss: 7.5904 - accuracy: 0.5050
<br /> 4256/25000 [====>.........................] - ETA: 48s - loss: 7.5802 - accuracy: 0.5056
<br /> 4288/25000 [====>.........................] - ETA: 48s - loss: 7.5880 - accuracy: 0.5051
<br /> 4320/25000 [====>.........................] - ETA: 48s - loss: 7.5992 - accuracy: 0.5044
<br /> 4352/25000 [====>.........................] - ETA: 48s - loss: 7.5997 - accuracy: 0.5044
<br /> 4384/25000 [====>.........................] - ETA: 48s - loss: 7.6072 - accuracy: 0.5039
<br /> 4416/25000 [====>.........................] - ETA: 48s - loss: 7.6006 - accuracy: 0.5043
<br /> 4448/25000 [====>.........................] - ETA: 47s - loss: 7.6115 - accuracy: 0.5036
<br /> 4480/25000 [====>.........................] - ETA: 47s - loss: 7.6290 - accuracy: 0.5025
<br /> 4512/25000 [====>.........................] - ETA: 47s - loss: 7.6258 - accuracy: 0.5027
<br /> 4544/25000 [====>.........................] - ETA: 47s - loss: 7.6430 - accuracy: 0.5015
<br /> 4576/25000 [====>.........................] - ETA: 47s - loss: 7.6398 - accuracy: 0.5017
<br /> 4608/25000 [====>.........................] - ETA: 47s - loss: 7.6333 - accuracy: 0.5022
<br /> 4640/25000 [====>.........................] - ETA: 47s - loss: 7.6402 - accuracy: 0.5017
<br /> 4672/25000 [====>.........................] - ETA: 47s - loss: 7.6338 - accuracy: 0.5021
<br /> 4704/25000 [====>.........................] - ETA: 47s - loss: 7.6373 - accuracy: 0.5019
<br /> 4736/25000 [====>.........................] - ETA: 47s - loss: 7.6278 - accuracy: 0.5025
<br /> 4768/25000 [====>.........................] - ETA: 47s - loss: 7.6087 - accuracy: 0.5038
<br /> 4800/25000 [====>.........................] - ETA: 47s - loss: 7.5868 - accuracy: 0.5052
<br /> 4832/25000 [====>.........................] - ETA: 46s - loss: 7.5809 - accuracy: 0.5056
<br /> 4864/25000 [====>.........................] - ETA: 46s - loss: 7.5847 - accuracy: 0.5053
<br /> 4896/25000 [====>.........................] - ETA: 46s - loss: 7.5727 - accuracy: 0.5061
<br /> 4928/25000 [====>.........................] - ETA: 46s - loss: 7.5733 - accuracy: 0.5061
<br /> 4960/25000 [====>.........................] - ETA: 46s - loss: 7.5708 - accuracy: 0.5063
<br /> 4992/25000 [====>.........................] - ETA: 46s - loss: 7.5622 - accuracy: 0.5068
<br /> 5024/25000 [=====>........................] - ETA: 46s - loss: 7.5690 - accuracy: 0.5064
<br /> 5056/25000 [=====>........................] - ETA: 46s - loss: 7.5726 - accuracy: 0.5061
<br /> 5088/25000 [=====>........................] - ETA: 46s - loss: 7.5792 - accuracy: 0.5057
<br /> 5120/25000 [=====>........................] - ETA: 46s - loss: 7.5888 - accuracy: 0.5051
<br /> 5152/25000 [=====>........................] - ETA: 46s - loss: 7.5803 - accuracy: 0.5056
<br /> 5184/25000 [=====>........................] - ETA: 45s - loss: 7.5661 - accuracy: 0.5066
<br /> 5216/25000 [=====>........................] - ETA: 45s - loss: 7.5490 - accuracy: 0.5077
<br /> 5248/25000 [=====>........................] - ETA: 45s - loss: 7.5614 - accuracy: 0.5069
<br /> 5280/25000 [=====>........................] - ETA: 45s - loss: 7.5563 - accuracy: 0.5072
<br /> 5312/25000 [=====>........................] - ETA: 45s - loss: 7.5512 - accuracy: 0.5075
<br /> 5344/25000 [=====>........................] - ETA: 45s - loss: 7.5432 - accuracy: 0.5080
<br /> 5376/25000 [=====>........................] - ETA: 45s - loss: 7.5440 - accuracy: 0.5080
<br /> 5408/25000 [=====>........................] - ETA: 45s - loss: 7.5447 - accuracy: 0.5080
<br /> 5440/25000 [=====>........................] - ETA: 45s - loss: 7.5539 - accuracy: 0.5074
<br /> 5472/25000 [=====>........................] - ETA: 45s - loss: 7.5517 - accuracy: 0.5075
<br /> 5504/25000 [=====>........................] - ETA: 45s - loss: 7.5524 - accuracy: 0.5074
<br /> 5536/25000 [=====>........................] - ETA: 45s - loss: 7.5614 - accuracy: 0.5069
<br /> 5568/25000 [=====>........................] - ETA: 45s - loss: 7.5730 - accuracy: 0.5061
<br /> 5600/25000 [=====>........................] - ETA: 44s - loss: 7.5708 - accuracy: 0.5063
<br /> 5632/25000 [=====>........................] - ETA: 44s - loss: 7.5686 - accuracy: 0.5064
<br /> 5664/25000 [=====>........................] - ETA: 44s - loss: 7.5583 - accuracy: 0.5071
<br /> 5696/25000 [=====>........................] - ETA: 44s - loss: 7.5589 - accuracy: 0.5070
<br /> 5728/25000 [=====>........................] - ETA: 44s - loss: 7.5676 - accuracy: 0.5065
<br /> 5760/25000 [=====>........................] - ETA: 44s - loss: 7.5655 - accuracy: 0.5066
<br /> 5792/25000 [=====>........................] - ETA: 44s - loss: 7.5687 - accuracy: 0.5064
<br /> 5824/25000 [=====>........................] - ETA: 44s - loss: 7.5745 - accuracy: 0.5060
<br /> 5856/25000 [======>.......................] - ETA: 44s - loss: 7.5750 - accuracy: 0.5060
<br /> 5888/25000 [======>.......................] - ETA: 44s - loss: 7.5859 - accuracy: 0.5053
<br /> 5920/25000 [======>.......................] - ETA: 44s - loss: 7.5915 - accuracy: 0.5049
<br /> 5952/25000 [======>.......................] - ETA: 44s - loss: 7.5945 - accuracy: 0.5047
<br /> 5984/25000 [======>.......................] - ETA: 44s - loss: 7.6000 - accuracy: 0.5043
<br /> 6016/25000 [======>.......................] - ETA: 43s - loss: 7.6105 - accuracy: 0.5037
<br /> 6048/25000 [======>.......................] - ETA: 43s - loss: 7.6184 - accuracy: 0.5031
<br /> 6080/25000 [======>.......................] - ETA: 43s - loss: 7.6237 - accuracy: 0.5028
<br /> 6112/25000 [======>.......................] - ETA: 43s - loss: 7.6265 - accuracy: 0.5026
<br /> 6144/25000 [======>.......................] - ETA: 43s - loss: 7.6292 - accuracy: 0.5024
<br /> 6176/25000 [======>.......................] - ETA: 43s - loss: 7.6319 - accuracy: 0.5023
<br /> 6208/25000 [======>.......................] - ETA: 43s - loss: 7.6296 - accuracy: 0.5024
<br /> 6240/25000 [======>.......................] - ETA: 43s - loss: 7.6298 - accuracy: 0.5024
<br /> 6272/25000 [======>.......................] - ETA: 43s - loss: 7.6128 - accuracy: 0.5035
<br /> 6304/25000 [======>.......................] - ETA: 43s - loss: 7.6228 - accuracy: 0.5029
<br /> 6336/25000 [======>.......................] - ETA: 43s - loss: 7.6279 - accuracy: 0.5025
<br /> 6368/25000 [======>.......................] - ETA: 43s - loss: 7.6305 - accuracy: 0.5024
<br /> 6400/25000 [======>.......................] - ETA: 42s - loss: 7.6475 - accuracy: 0.5013
<br /> 6432/25000 [======>.......................] - ETA: 42s - loss: 7.6309 - accuracy: 0.5023
<br /> 6464/25000 [======>.......................] - ETA: 42s - loss: 7.6263 - accuracy: 0.5026
<br /> 6496/25000 [======>.......................] - ETA: 42s - loss: 7.6218 - accuracy: 0.5029
<br /> 6528/25000 [======>.......................] - ETA: 42s - loss: 7.6196 - accuracy: 0.5031
<br /> 6560/25000 [======>.......................] - ETA: 42s - loss: 7.6245 - accuracy: 0.5027
<br /> 6592/25000 [======>.......................] - ETA: 42s - loss: 7.6224 - accuracy: 0.5029
<br /> 6624/25000 [======>.......................] - ETA: 42s - loss: 7.6180 - accuracy: 0.5032
<br /> 6656/25000 [======>.......................] - ETA: 42s - loss: 7.6205 - accuracy: 0.5030
<br /> 6688/25000 [=======>......................] - ETA: 42s - loss: 7.6231 - accuracy: 0.5028
<br /> 6720/25000 [=======>......................] - ETA: 42s - loss: 7.6301 - accuracy: 0.5024
<br /> 6752/25000 [=======>......................] - ETA: 42s - loss: 7.6394 - accuracy: 0.5018
<br /> 6784/25000 [=======>......................] - ETA: 42s - loss: 7.6350 - accuracy: 0.5021
<br /> 6816/25000 [=======>......................] - ETA: 42s - loss: 7.6396 - accuracy: 0.5018
<br /> 6848/25000 [=======>......................] - ETA: 41s - loss: 7.6398 - accuracy: 0.5018
<br /> 6880/25000 [=======>......................] - ETA: 41s - loss: 7.6310 - accuracy: 0.5023
<br /> 6912/25000 [=======>......................] - ETA: 41s - loss: 7.6311 - accuracy: 0.5023
<br /> 6944/25000 [=======>......................] - ETA: 41s - loss: 7.6247 - accuracy: 0.5027
<br /> 6976/25000 [=======>......................] - ETA: 41s - loss: 7.6161 - accuracy: 0.5033
<br /> 7008/25000 [=======>......................] - ETA: 41s - loss: 7.6141 - accuracy: 0.5034
<br /> 7040/25000 [=======>......................] - ETA: 41s - loss: 7.6122 - accuracy: 0.5036
<br /> 7072/25000 [=======>......................] - ETA: 41s - loss: 7.6081 - accuracy: 0.5038
<br /> 7104/25000 [=======>......................] - ETA: 41s - loss: 7.6019 - accuracy: 0.5042
<br /> 7136/25000 [=======>......................] - ETA: 41s - loss: 7.6086 - accuracy: 0.5038
<br /> 7168/25000 [=======>......................] - ETA: 41s - loss: 7.6110 - accuracy: 0.5036
<br /> 7200/25000 [=======>......................] - ETA: 41s - loss: 7.6176 - accuracy: 0.5032
<br /> 7232/25000 [=======>......................] - ETA: 41s - loss: 7.6136 - accuracy: 0.5035
<br /> 7264/25000 [=======>......................] - ETA: 41s - loss: 7.6138 - accuracy: 0.5034
<br /> 7296/25000 [=======>......................] - ETA: 40s - loss: 7.6057 - accuracy: 0.5040
<br /> 7328/25000 [=======>......................] - ETA: 40s - loss: 7.6059 - accuracy: 0.5040
<br /> 7360/25000 [=======>......................] - ETA: 40s - loss: 7.6083 - accuracy: 0.5038
<br /> 7392/25000 [=======>......................] - ETA: 40s - loss: 7.6065 - accuracy: 0.5039
<br /> 7424/25000 [=======>......................] - ETA: 40s - loss: 7.5964 - accuracy: 0.5046
<br /> 7456/25000 [=======>......................] - ETA: 40s - loss: 7.5967 - accuracy: 0.5046
<br /> 7488/25000 [=======>......................] - ETA: 40s - loss: 7.5929 - accuracy: 0.5048
<br /> 7520/25000 [========>.....................] - ETA: 40s - loss: 7.5851 - accuracy: 0.5053
<br /> 7552/25000 [========>.....................] - ETA: 40s - loss: 7.5834 - accuracy: 0.5054
<br /> 7584/25000 [========>.....................] - ETA: 40s - loss: 7.5716 - accuracy: 0.5062
<br /> 7616/25000 [========>.....................] - ETA: 40s - loss: 7.5660 - accuracy: 0.5066
<br /> 7648/25000 [========>.....................] - ETA: 40s - loss: 7.5644 - accuracy: 0.5067
<br /> 7680/25000 [========>.....................] - ETA: 39s - loss: 7.5588 - accuracy: 0.5070
<br /> 7712/25000 [========>.....................] - ETA: 39s - loss: 7.5652 - accuracy: 0.5066
<br /> 7744/25000 [========>.....................] - ETA: 39s - loss: 7.5637 - accuracy: 0.5067
<br /> 7776/25000 [========>.....................] - ETA: 39s - loss: 7.5641 - accuracy: 0.5067
<br /> 7808/25000 [========>.....................] - ETA: 39s - loss: 7.5625 - accuracy: 0.5068
<br /> 7840/25000 [========>.....................] - ETA: 39s - loss: 7.5669 - accuracy: 0.5065
<br /> 7872/25000 [========>.....................] - ETA: 39s - loss: 7.5692 - accuracy: 0.5064
<br /> 7904/25000 [========>.....................] - ETA: 39s - loss: 7.5793 - accuracy: 0.5057
<br /> 7936/25000 [========>.....................] - ETA: 39s - loss: 7.5739 - accuracy: 0.5060
<br /> 7968/25000 [========>.....................] - ETA: 39s - loss: 7.5685 - accuracy: 0.5064
<br /> 8000/25000 [========>.....................] - ETA: 39s - loss: 7.5746 - accuracy: 0.5060
<br /> 8032/25000 [========>.....................] - ETA: 39s - loss: 7.5750 - accuracy: 0.5060
<br /> 8064/25000 [========>.....................] - ETA: 39s - loss: 7.5830 - accuracy: 0.5055
<br /> 8096/25000 [========>.....................] - ETA: 38s - loss: 7.5890 - accuracy: 0.5051
<br /> 8128/25000 [========>.....................] - ETA: 38s - loss: 7.5968 - accuracy: 0.5046
<br /> 8160/25000 [========>.....................] - ETA: 38s - loss: 7.5933 - accuracy: 0.5048
<br /> 8192/25000 [========>.....................] - ETA: 38s - loss: 7.5918 - accuracy: 0.5049
<br /> 8224/25000 [========>.....................] - ETA: 38s - loss: 7.5976 - accuracy: 0.5045
<br /> 8256/25000 [========>.....................] - ETA: 38s - loss: 7.6016 - accuracy: 0.5042
<br /> 8288/25000 [========>.....................] - ETA: 38s - loss: 7.6111 - accuracy: 0.5036
<br /> 8320/25000 [========>.....................] - ETA: 38s - loss: 7.6040 - accuracy: 0.5041
<br /> 8352/25000 [=========>....................] - ETA: 38s - loss: 7.6079 - accuracy: 0.5038
<br /> 8384/25000 [=========>....................] - ETA: 38s - loss: 7.6118 - accuracy: 0.5036
<br /> 8416/25000 [=========>....................] - ETA: 38s - loss: 7.6101 - accuracy: 0.5037
<br /> 8448/25000 [=========>....................] - ETA: 38s - loss: 7.6122 - accuracy: 0.5036
<br /> 8480/25000 [=========>....................] - ETA: 38s - loss: 7.6160 - accuracy: 0.5033
<br /> 8512/25000 [=========>....................] - ETA: 37s - loss: 7.6126 - accuracy: 0.5035
<br /> 8544/25000 [=========>....................] - ETA: 37s - loss: 7.6092 - accuracy: 0.5037
<br /> 8576/25000 [=========>....................] - ETA: 37s - loss: 7.6112 - accuracy: 0.5036
<br /> 8608/25000 [=========>....................] - ETA: 37s - loss: 7.6114 - accuracy: 0.5036
<br /> 8640/25000 [=========>....................] - ETA: 37s - loss: 7.6152 - accuracy: 0.5034
<br /> 8672/25000 [=========>....................] - ETA: 37s - loss: 7.6189 - accuracy: 0.5031
<br /> 8704/25000 [=========>....................] - ETA: 37s - loss: 7.6226 - accuracy: 0.5029
<br /> 8736/25000 [=========>....................] - ETA: 37s - loss: 7.6192 - accuracy: 0.5031
<br /> 8768/25000 [=========>....................] - ETA: 37s - loss: 7.6177 - accuracy: 0.5032
<br /> 8800/25000 [=========>....................] - ETA: 37s - loss: 7.6178 - accuracy: 0.5032
<br /> 8832/25000 [=========>....................] - ETA: 37s - loss: 7.6145 - accuracy: 0.5034
<br /> 8864/25000 [=========>....................] - ETA: 37s - loss: 7.6216 - accuracy: 0.5029
<br /> 8896/25000 [=========>....................] - ETA: 37s - loss: 7.6201 - accuracy: 0.5030
<br /> 8928/25000 [=========>....................] - ETA: 37s - loss: 7.6168 - accuracy: 0.5032
<br /> 8960/25000 [=========>....................] - ETA: 36s - loss: 7.6204 - accuracy: 0.5030
<br /> 8992/25000 [=========>....................] - ETA: 36s - loss: 7.6240 - accuracy: 0.5028
<br /> 9024/25000 [=========>....................] - ETA: 36s - loss: 7.6224 - accuracy: 0.5029
<br /> 9056/25000 [=========>....................] - ETA: 36s - loss: 7.6209 - accuracy: 0.5030
<br /> 9088/25000 [=========>....................] - ETA: 36s - loss: 7.6177 - accuracy: 0.5032
<br /> 9120/25000 [=========>....................] - ETA: 36s - loss: 7.6212 - accuracy: 0.5030
<br /> 9152/25000 [=========>....................] - ETA: 36s - loss: 7.6197 - accuracy: 0.5031
<br /> 9184/25000 [==========>...................] - ETA: 36s - loss: 7.6232 - accuracy: 0.5028
<br /> 9216/25000 [==========>...................] - ETA: 36s - loss: 7.6250 - accuracy: 0.5027
<br /> 9248/25000 [==========>...................] - ETA: 36s - loss: 7.6285 - accuracy: 0.5025
<br /> 9280/25000 [==========>...................] - ETA: 36s - loss: 7.6270 - accuracy: 0.5026
<br /> 9312/25000 [==========>...................] - ETA: 36s - loss: 7.6304 - accuracy: 0.5024
<br /> 9344/25000 [==========>...................] - ETA: 35s - loss: 7.6322 - accuracy: 0.5022
<br /> 9376/25000 [==========>...................] - ETA: 35s - loss: 7.6372 - accuracy: 0.5019
<br /> 9408/25000 [==========>...................] - ETA: 35s - loss: 7.6389 - accuracy: 0.5018
<br /> 9440/25000 [==========>...................] - ETA: 35s - loss: 7.6439 - accuracy: 0.5015
<br /> 9472/25000 [==========>...................] - ETA: 35s - loss: 7.6456 - accuracy: 0.5014
<br /> 9504/25000 [==========>...................] - ETA: 35s - loss: 7.6440 - accuracy: 0.5015
<br /> 9536/25000 [==========>...................] - ETA: 35s - loss: 7.6393 - accuracy: 0.5018
<br /> 9568/25000 [==========>...................] - ETA: 35s - loss: 7.6410 - accuracy: 0.5017
<br /> 9600/25000 [==========>...................] - ETA: 35s - loss: 7.6443 - accuracy: 0.5015
<br /> 9632/25000 [==========>...................] - ETA: 35s - loss: 7.6507 - accuracy: 0.5010
<br /> 9664/25000 [==========>...................] - ETA: 35s - loss: 7.6428 - accuracy: 0.5016
<br /> 9696/25000 [==========>...................] - ETA: 35s - loss: 7.6413 - accuracy: 0.5017
<br /> 9728/25000 [==========>...................] - ETA: 35s - loss: 7.6367 - accuracy: 0.5020
<br /> 9760/25000 [==========>...................] - ETA: 35s - loss: 7.6399 - accuracy: 0.5017
<br /> 9792/25000 [==========>...................] - ETA: 34s - loss: 7.6322 - accuracy: 0.5022
<br /> 9824/25000 [==========>...................] - ETA: 34s - loss: 7.6307 - accuracy: 0.5023
<br /> 9856/25000 [==========>...................] - ETA: 34s - loss: 7.6215 - accuracy: 0.5029
<br /> 9888/25000 [==========>...................] - ETA: 34s - loss: 7.6248 - accuracy: 0.5027
<br /> 9920/25000 [==========>...................] - ETA: 34s - loss: 7.6249 - accuracy: 0.5027
<br /> 9952/25000 [==========>...................] - ETA: 34s - loss: 7.6312 - accuracy: 0.5023
<br /> 9984/25000 [==========>...................] - ETA: 34s - loss: 7.6313 - accuracy: 0.5023
<br />10016/25000 [===========>..................] - ETA: 34s - loss: 7.6329 - accuracy: 0.5022
<br />10048/25000 [===========>..................] - ETA: 34s - loss: 7.6285 - accuracy: 0.5025
<br />10080/25000 [===========>..................] - ETA: 34s - loss: 7.6286 - accuracy: 0.5025
<br />10112/25000 [===========>..................] - ETA: 34s - loss: 7.6211 - accuracy: 0.5030
<br />10144/25000 [===========>..................] - ETA: 34s - loss: 7.6198 - accuracy: 0.5031
<br />10176/25000 [===========>..................] - ETA: 34s - loss: 7.6199 - accuracy: 0.5030
<br />10208/25000 [===========>..................] - ETA: 33s - loss: 7.6140 - accuracy: 0.5034
<br />10240/25000 [===========>..................] - ETA: 33s - loss: 7.6127 - accuracy: 0.5035
<br />10272/25000 [===========>..................] - ETA: 33s - loss: 7.6189 - accuracy: 0.5031
<br />10304/25000 [===========>..................] - ETA: 33s - loss: 7.6205 - accuracy: 0.5030
<br />10336/25000 [===========>..................] - ETA: 33s - loss: 7.6177 - accuracy: 0.5032
<br />10368/25000 [===========>..................] - ETA: 33s - loss: 7.6252 - accuracy: 0.5027
<br />10400/25000 [===========>..................] - ETA: 33s - loss: 7.6224 - accuracy: 0.5029
<br />10432/25000 [===========>..................] - ETA: 33s - loss: 7.6181 - accuracy: 0.5032
<br />10464/25000 [===========>..................] - ETA: 33s - loss: 7.6197 - accuracy: 0.5031
<br />10496/25000 [===========>..................] - ETA: 33s - loss: 7.6126 - accuracy: 0.5035
<br />10528/25000 [===========>..................] - ETA: 33s - loss: 7.6142 - accuracy: 0.5034
<br />10560/25000 [===========>..................] - ETA: 33s - loss: 7.6114 - accuracy: 0.5036
<br />10592/25000 [===========>..................] - ETA: 33s - loss: 7.6160 - accuracy: 0.5033
<br />10624/25000 [===========>..................] - ETA: 33s - loss: 7.6219 - accuracy: 0.5029
<br />10656/25000 [===========>..................] - ETA: 32s - loss: 7.6206 - accuracy: 0.5030
<br />10688/25000 [===========>..................] - ETA: 32s - loss: 7.6236 - accuracy: 0.5028
<br />10720/25000 [===========>..................] - ETA: 32s - loss: 7.6251 - accuracy: 0.5027
<br />10752/25000 [===========>..................] - ETA: 32s - loss: 7.6310 - accuracy: 0.5023
<br />10784/25000 [===========>..................] - ETA: 32s - loss: 7.6325 - accuracy: 0.5022
<br />10816/25000 [===========>..................] - ETA: 32s - loss: 7.6312 - accuracy: 0.5023
<br />10848/25000 [============>.................] - ETA: 32s - loss: 7.6313 - accuracy: 0.5023
<br />10880/25000 [============>.................] - ETA: 32s - loss: 7.6300 - accuracy: 0.5024
<br />10912/25000 [============>.................] - ETA: 32s - loss: 7.6329 - accuracy: 0.5022
<br />10944/25000 [============>.................] - ETA: 32s - loss: 7.6330 - accuracy: 0.5022
<br />10976/25000 [============>.................] - ETA: 32s - loss: 7.6303 - accuracy: 0.5024
<br />11008/25000 [============>.................] - ETA: 32s - loss: 7.6332 - accuracy: 0.5022
<br />11040/25000 [============>.................] - ETA: 32s - loss: 7.6319 - accuracy: 0.5023
<br />11072/25000 [============>.................] - ETA: 31s - loss: 7.6265 - accuracy: 0.5026
<br />11104/25000 [============>.................] - ETA: 31s - loss: 7.6252 - accuracy: 0.5027
<br />11136/25000 [============>.................] - ETA: 31s - loss: 7.6212 - accuracy: 0.5030
<br />11168/25000 [============>.................] - ETA: 31s - loss: 7.6241 - accuracy: 0.5028
<br />11200/25000 [============>.................] - ETA: 31s - loss: 7.6242 - accuracy: 0.5028
<br />11232/25000 [============>.................] - ETA: 31s - loss: 7.6243 - accuracy: 0.5028
<br />11264/25000 [============>.................] - ETA: 31s - loss: 7.6258 - accuracy: 0.5027
<br />11296/25000 [============>.................] - ETA: 31s - loss: 7.6259 - accuracy: 0.5027
<br />11328/25000 [============>.................] - ETA: 31s - loss: 7.6260 - accuracy: 0.5026
<br />11360/25000 [============>.................] - ETA: 31s - loss: 7.6234 - accuracy: 0.5028
<br />11392/25000 [============>.................] - ETA: 31s - loss: 7.6249 - accuracy: 0.5027
<br />11424/25000 [============>.................] - ETA: 31s - loss: 7.6237 - accuracy: 0.5028
<br />11456/25000 [============>.................] - ETA: 31s - loss: 7.6238 - accuracy: 0.5028
<br />11488/25000 [============>.................] - ETA: 30s - loss: 7.6306 - accuracy: 0.5024
<br />11520/25000 [============>.................] - ETA: 30s - loss: 7.6360 - accuracy: 0.5020
<br />11552/25000 [============>.................] - ETA: 30s - loss: 7.6321 - accuracy: 0.5023
<br />11584/25000 [============>.................] - ETA: 30s - loss: 7.6296 - accuracy: 0.5024
<br />11616/25000 [============>.................] - ETA: 30s - loss: 7.6244 - accuracy: 0.5028
<br />11648/25000 [============>.................] - ETA: 30s - loss: 7.6258 - accuracy: 0.5027
<br />11680/25000 [=============>................] - ETA: 30s - loss: 7.6312 - accuracy: 0.5023
<br />11712/25000 [=============>................] - ETA: 30s - loss: 7.6313 - accuracy: 0.5023
<br />11744/25000 [=============>................] - ETA: 30s - loss: 7.6288 - accuracy: 0.5025
<br />11776/25000 [=============>................] - ETA: 30s - loss: 7.6302 - accuracy: 0.5024
<br />11808/25000 [=============>................] - ETA: 30s - loss: 7.6342 - accuracy: 0.5021
<br />11840/25000 [=============>................] - ETA: 30s - loss: 7.6342 - accuracy: 0.5021
<br />11872/25000 [=============>................] - ETA: 30s - loss: 7.6395 - accuracy: 0.5018
<br />11904/25000 [=============>................] - ETA: 29s - loss: 7.6357 - accuracy: 0.5020
<br />11936/25000 [=============>................] - ETA: 29s - loss: 7.6371 - accuracy: 0.5019
<br />11968/25000 [=============>................] - ETA: 29s - loss: 7.6423 - accuracy: 0.5016
<br />12000/25000 [=============>................] - ETA: 29s - loss: 7.6385 - accuracy: 0.5018
<br />12032/25000 [=============>................] - ETA: 29s - loss: 7.6348 - accuracy: 0.5021
<br />12064/25000 [=============>................] - ETA: 29s - loss: 7.6310 - accuracy: 0.5023
<br />12096/25000 [=============>................] - ETA: 29s - loss: 7.6273 - accuracy: 0.5026
<br />12128/25000 [=============>................] - ETA: 29s - loss: 7.6274 - accuracy: 0.5026
<br />12160/25000 [=============>................] - ETA: 29s - loss: 7.6288 - accuracy: 0.5025
<br />12192/25000 [=============>................] - ETA: 29s - loss: 7.6289 - accuracy: 0.5025
<br />12224/25000 [=============>................] - ETA: 29s - loss: 7.6265 - accuracy: 0.5026
<br />12256/25000 [=============>................] - ETA: 29s - loss: 7.6266 - accuracy: 0.5026
<br />12288/25000 [=============>................] - ETA: 29s - loss: 7.6229 - accuracy: 0.5028
<br />12320/25000 [=============>................] - ETA: 29s - loss: 7.6218 - accuracy: 0.5029
<br />12352/25000 [=============>................] - ETA: 28s - loss: 7.6219 - accuracy: 0.5029
<br />12384/25000 [=============>................] - ETA: 28s - loss: 7.6196 - accuracy: 0.5031
<br />12416/25000 [=============>................] - ETA: 28s - loss: 7.6197 - accuracy: 0.5031
<br />12448/25000 [=============>................] - ETA: 28s - loss: 7.6210 - accuracy: 0.5030
<br />12480/25000 [=============>................] - ETA: 28s - loss: 7.6162 - accuracy: 0.5033
<br />12512/25000 [==============>...............] - ETA: 28s - loss: 7.6164 - accuracy: 0.5033
<br />12544/25000 [==============>...............] - ETA: 28s - loss: 7.6128 - accuracy: 0.5035
<br />12576/25000 [==============>...............] - ETA: 28s - loss: 7.6227 - accuracy: 0.5029
<br />12608/25000 [==============>...............] - ETA: 28s - loss: 7.6192 - accuracy: 0.5031
<br />12640/25000 [==============>...............] - ETA: 28s - loss: 7.6217 - accuracy: 0.5029
<br />12672/25000 [==============>...............] - ETA: 28s - loss: 7.6194 - accuracy: 0.5031
<br />12704/25000 [==============>...............] - ETA: 28s - loss: 7.6171 - accuracy: 0.5032
<br />12736/25000 [==============>...............] - ETA: 28s - loss: 7.6161 - accuracy: 0.5033
<br />12768/25000 [==============>...............] - ETA: 27s - loss: 7.6174 - accuracy: 0.5032
<br />12800/25000 [==============>...............] - ETA: 27s - loss: 7.6175 - accuracy: 0.5032
<br />12832/25000 [==============>...............] - ETA: 27s - loss: 7.6164 - accuracy: 0.5033
<br />12864/25000 [==============>...............] - ETA: 27s - loss: 7.6130 - accuracy: 0.5035
<br />12896/25000 [==============>...............] - ETA: 27s - loss: 7.6167 - accuracy: 0.5033
<br />12928/25000 [==============>...............] - ETA: 27s - loss: 7.6215 - accuracy: 0.5029
<br />12960/25000 [==============>...............] - ETA: 27s - loss: 7.6205 - accuracy: 0.5030
<br />12992/25000 [==============>...............] - ETA: 27s - loss: 7.6159 - accuracy: 0.5033
<br />13024/25000 [==============>...............] - ETA: 27s - loss: 7.6148 - accuracy: 0.5034
<br />13056/25000 [==============>...............] - ETA: 27s - loss: 7.6138 - accuracy: 0.5034
<br />13088/25000 [==============>...............] - ETA: 27s - loss: 7.6162 - accuracy: 0.5033
<br />13120/25000 [==============>...............] - ETA: 27s - loss: 7.6187 - accuracy: 0.5031
<br />13152/25000 [==============>...............] - ETA: 27s - loss: 7.6177 - accuracy: 0.5032
<br />13184/25000 [==============>...............] - ETA: 27s - loss: 7.6166 - accuracy: 0.5033
<br />13216/25000 [==============>...............] - ETA: 26s - loss: 7.6179 - accuracy: 0.5032
<br />13248/25000 [==============>...............] - ETA: 26s - loss: 7.6203 - accuracy: 0.5030
<br />13280/25000 [==============>...............] - ETA: 26s - loss: 7.6227 - accuracy: 0.5029
<br />13312/25000 [==============>...............] - ETA: 26s - loss: 7.6217 - accuracy: 0.5029
<br />13344/25000 [===============>..............] - ETA: 26s - loss: 7.6230 - accuracy: 0.5028
<br />13376/25000 [===============>..............] - ETA: 26s - loss: 7.6173 - accuracy: 0.5032
<br />13408/25000 [===============>..............] - ETA: 26s - loss: 7.6197 - accuracy: 0.5031
<br />13440/25000 [===============>..............] - ETA: 26s - loss: 7.6221 - accuracy: 0.5029
<br />13472/25000 [===============>..............] - ETA: 26s - loss: 7.6211 - accuracy: 0.5030
<br />13504/25000 [===============>..............] - ETA: 26s - loss: 7.6178 - accuracy: 0.5032
<br />13536/25000 [===============>..............] - ETA: 26s - loss: 7.6145 - accuracy: 0.5034
<br />13568/25000 [===============>..............] - ETA: 26s - loss: 7.6135 - accuracy: 0.5035
<br />13600/25000 [===============>..............] - ETA: 26s - loss: 7.6148 - accuracy: 0.5034
<br />13632/25000 [===============>..............] - ETA: 25s - loss: 7.6149 - accuracy: 0.5034
<br />13664/25000 [===============>..............] - ETA: 25s - loss: 7.6128 - accuracy: 0.5035
<br />13696/25000 [===============>..............] - ETA: 25s - loss: 7.6095 - accuracy: 0.5037
<br />13728/25000 [===============>..............] - ETA: 25s - loss: 7.6164 - accuracy: 0.5033
<br />13760/25000 [===============>..............] - ETA: 25s - loss: 7.6165 - accuracy: 0.5033
<br />13792/25000 [===============>..............] - ETA: 25s - loss: 7.6155 - accuracy: 0.5033
<br />13824/25000 [===============>..............] - ETA: 25s - loss: 7.6178 - accuracy: 0.5032
<br />13856/25000 [===============>..............] - ETA: 25s - loss: 7.6246 - accuracy: 0.5027
<br />13888/25000 [===============>..............] - ETA: 25s - loss: 7.6225 - accuracy: 0.5029
<br />13920/25000 [===============>..............] - ETA: 25s - loss: 7.6226 - accuracy: 0.5029
<br />13952/25000 [===============>..............] - ETA: 25s - loss: 7.6216 - accuracy: 0.5029
<br />13984/25000 [===============>..............] - ETA: 25s - loss: 7.6151 - accuracy: 0.5034
<br />14016/25000 [===============>..............] - ETA: 25s - loss: 7.6152 - accuracy: 0.5034
<br />14048/25000 [===============>..............] - ETA: 25s - loss: 7.6175 - accuracy: 0.5032
<br />14080/25000 [===============>..............] - ETA: 24s - loss: 7.6176 - accuracy: 0.5032
<br />14112/25000 [===============>..............] - ETA: 24s - loss: 7.6156 - accuracy: 0.5033
<br />14144/25000 [===============>..............] - ETA: 24s - loss: 7.6222 - accuracy: 0.5029
<br />14176/25000 [================>.............] - ETA: 24s - loss: 7.6212 - accuracy: 0.5030
<br />14208/25000 [================>.............] - ETA: 24s - loss: 7.6202 - accuracy: 0.5030
<br />14240/25000 [================>.............] - ETA: 24s - loss: 7.6203 - accuracy: 0.5030
<br />14272/25000 [================>.............] - ETA: 24s - loss: 7.6193 - accuracy: 0.5031
<br />14304/25000 [================>.............] - ETA: 24s - loss: 7.6152 - accuracy: 0.5034
<br />14336/25000 [================>.............] - ETA: 24s - loss: 7.6153 - accuracy: 0.5033
<br />14368/25000 [================>.............] - ETA: 24s - loss: 7.6186 - accuracy: 0.5031
<br />14400/25000 [================>.............] - ETA: 24s - loss: 7.6198 - accuracy: 0.5031
<br />14432/25000 [================>.............] - ETA: 24s - loss: 7.6177 - accuracy: 0.5032
<br />14464/25000 [================>.............] - ETA: 24s - loss: 7.6210 - accuracy: 0.5030
<br />14496/25000 [================>.............] - ETA: 24s - loss: 7.6285 - accuracy: 0.5025
<br />14528/25000 [================>.............] - ETA: 23s - loss: 7.6297 - accuracy: 0.5024
<br />14560/25000 [================>.............] - ETA: 23s - loss: 7.6340 - accuracy: 0.5021
<br />14592/25000 [================>.............] - ETA: 23s - loss: 7.6340 - accuracy: 0.5021
<br />14624/25000 [================>.............] - ETA: 23s - loss: 7.6362 - accuracy: 0.5020
<br />14656/25000 [================>.............] - ETA: 23s - loss: 7.6373 - accuracy: 0.5019
<br />14688/25000 [================>.............] - ETA: 23s - loss: 7.6384 - accuracy: 0.5018
<br />14720/25000 [================>.............] - ETA: 23s - loss: 7.6312 - accuracy: 0.5023
<br />14752/25000 [================>.............] - ETA: 23s - loss: 7.6334 - accuracy: 0.5022
<br />14784/25000 [================>.............] - ETA: 23s - loss: 7.6345 - accuracy: 0.5021
<br />14816/25000 [================>.............] - ETA: 23s - loss: 7.6345 - accuracy: 0.5021
<br />14848/25000 [================>.............] - ETA: 23s - loss: 7.6367 - accuracy: 0.5020
<br />14880/25000 [================>.............] - ETA: 23s - loss: 7.6347 - accuracy: 0.5021
<br />14912/25000 [================>.............] - ETA: 23s - loss: 7.6317 - accuracy: 0.5023
<br />14944/25000 [================>.............] - ETA: 22s - loss: 7.6317 - accuracy: 0.5023
<br />14976/25000 [================>.............] - ETA: 22s - loss: 7.6318 - accuracy: 0.5023
<br />15008/25000 [=================>............] - ETA: 22s - loss: 7.6309 - accuracy: 0.5023
<br />15040/25000 [=================>............] - ETA: 22s - loss: 7.6309 - accuracy: 0.5023
<br />15072/25000 [=================>............] - ETA: 22s - loss: 7.6259 - accuracy: 0.5027
<br />15104/25000 [=================>............] - ETA: 22s - loss: 7.6260 - accuracy: 0.5026
<br />15136/25000 [=================>............] - ETA: 22s - loss: 7.6261 - accuracy: 0.5026
<br />15168/25000 [=================>............] - ETA: 22s - loss: 7.6262 - accuracy: 0.5026
<br />15200/25000 [=================>............] - ETA: 22s - loss: 7.6253 - accuracy: 0.5027
<br />15232/25000 [=================>............] - ETA: 22s - loss: 7.6233 - accuracy: 0.5028
<br />15264/25000 [=================>............] - ETA: 22s - loss: 7.6274 - accuracy: 0.5026
<br />15296/25000 [=================>............] - ETA: 22s - loss: 7.6245 - accuracy: 0.5027
<br />15328/25000 [=================>............] - ETA: 22s - loss: 7.6266 - accuracy: 0.5026
<br />15360/25000 [=================>............] - ETA: 22s - loss: 7.6257 - accuracy: 0.5027
<br />15392/25000 [=================>............] - ETA: 21s - loss: 7.6278 - accuracy: 0.5025
<br />15424/25000 [=================>............] - ETA: 21s - loss: 7.6219 - accuracy: 0.5029
<br />15456/25000 [=================>............] - ETA: 21s - loss: 7.6230 - accuracy: 0.5028
<br />15488/25000 [=================>............] - ETA: 21s - loss: 7.6270 - accuracy: 0.5026
<br />15520/25000 [=================>............] - ETA: 21s - loss: 7.6301 - accuracy: 0.5024
<br />15552/25000 [=================>............] - ETA: 21s - loss: 7.6272 - accuracy: 0.5026
<br />15584/25000 [=================>............] - ETA: 21s - loss: 7.6312 - accuracy: 0.5023
<br />15616/25000 [=================>............] - ETA: 21s - loss: 7.6332 - accuracy: 0.5022
<br />15648/25000 [=================>............] - ETA: 21s - loss: 7.6333 - accuracy: 0.5022
<br />15680/25000 [=================>............] - ETA: 21s - loss: 7.6353 - accuracy: 0.5020
<br />15712/25000 [=================>............] - ETA: 21s - loss: 7.6364 - accuracy: 0.5020
<br />15744/25000 [=================>............] - ETA: 21s - loss: 7.6374 - accuracy: 0.5019
<br />15776/25000 [=================>............] - ETA: 21s - loss: 7.6365 - accuracy: 0.5020
<br />15808/25000 [=================>............] - ETA: 20s - loss: 7.6336 - accuracy: 0.5022
<br />15840/25000 [==================>...........] - ETA: 20s - loss: 7.6327 - accuracy: 0.5022
<br />15872/25000 [==================>...........] - ETA: 20s - loss: 7.6299 - accuracy: 0.5024
<br />15904/25000 [==================>...........] - ETA: 20s - loss: 7.6338 - accuracy: 0.5021
<br />15936/25000 [==================>...........] - ETA: 20s - loss: 7.6339 - accuracy: 0.5021
<br />15968/25000 [==================>...........] - ETA: 20s - loss: 7.6359 - accuracy: 0.5020
<br />16000/25000 [==================>...........] - ETA: 20s - loss: 7.6321 - accuracy: 0.5023
<br />16032/25000 [==================>...........] - ETA: 20s - loss: 7.6284 - accuracy: 0.5025
<br />16064/25000 [==================>...........] - ETA: 20s - loss: 7.6256 - accuracy: 0.5027
<br />16096/25000 [==================>...........] - ETA: 20s - loss: 7.6257 - accuracy: 0.5027
<br />16128/25000 [==================>...........] - ETA: 20s - loss: 7.6210 - accuracy: 0.5030
<br />16160/25000 [==================>...........] - ETA: 20s - loss: 7.6211 - accuracy: 0.5030
<br />16192/25000 [==================>...........] - ETA: 20s - loss: 7.6193 - accuracy: 0.5031
<br />16224/25000 [==================>...........] - ETA: 20s - loss: 7.6203 - accuracy: 0.5030
<br />16256/25000 [==================>...........] - ETA: 19s - loss: 7.6204 - accuracy: 0.5030
<br />16288/25000 [==================>...........] - ETA: 19s - loss: 7.6214 - accuracy: 0.5029
<br />16320/25000 [==================>...........] - ETA: 19s - loss: 7.6196 - accuracy: 0.5031
<br />16352/25000 [==================>...........] - ETA: 19s - loss: 7.6197 - accuracy: 0.5031
<br />16384/25000 [==================>...........] - ETA: 19s - loss: 7.6170 - accuracy: 0.5032
<br />16416/25000 [==================>...........] - ETA: 19s - loss: 7.6162 - accuracy: 0.5033
<br />16448/25000 [==================>...........] - ETA: 19s - loss: 7.6172 - accuracy: 0.5032
<br />16480/25000 [==================>...........] - ETA: 19s - loss: 7.6127 - accuracy: 0.5035
<br />16512/25000 [==================>...........] - ETA: 19s - loss: 7.6165 - accuracy: 0.5033
<br />16544/25000 [==================>...........] - ETA: 19s - loss: 7.6138 - accuracy: 0.5034
<br />16576/25000 [==================>...........] - ETA: 19s - loss: 7.6130 - accuracy: 0.5035
<br />16608/25000 [==================>...........] - ETA: 19s - loss: 7.6112 - accuracy: 0.5036
<br />16640/25000 [==================>...........] - ETA: 19s - loss: 7.6113 - accuracy: 0.5036
<br />16672/25000 [===================>..........] - ETA: 19s - loss: 7.6096 - accuracy: 0.5037
<br />16704/25000 [===================>..........] - ETA: 18s - loss: 7.6097 - accuracy: 0.5037
<br />16736/25000 [===================>..........] - ETA: 18s - loss: 7.6089 - accuracy: 0.5038
<br />16768/25000 [===================>..........] - ETA: 18s - loss: 7.6118 - accuracy: 0.5036
<br />16800/25000 [===================>..........] - ETA: 18s - loss: 7.6173 - accuracy: 0.5032
<br />16832/25000 [===================>..........] - ETA: 18s - loss: 7.6174 - accuracy: 0.5032
<br />16864/25000 [===================>..........] - ETA: 18s - loss: 7.6221 - accuracy: 0.5029
<br />16896/25000 [===================>..........] - ETA: 18s - loss: 7.6240 - accuracy: 0.5028
<br />16928/25000 [===================>..........] - ETA: 18s - loss: 7.6231 - accuracy: 0.5028
<br />16960/25000 [===================>..........] - ETA: 18s - loss: 7.6205 - accuracy: 0.5030
<br />16992/25000 [===================>..........] - ETA: 18s - loss: 7.6206 - accuracy: 0.5030
<br />17024/25000 [===================>..........] - ETA: 18s - loss: 7.6198 - accuracy: 0.5031
<br />17056/25000 [===================>..........] - ETA: 18s - loss: 7.6226 - accuracy: 0.5029
<br />17088/25000 [===================>..........] - ETA: 18s - loss: 7.6218 - accuracy: 0.5029
<br />17120/25000 [===================>..........] - ETA: 17s - loss: 7.6183 - accuracy: 0.5032
<br />17152/25000 [===================>..........] - ETA: 17s - loss: 7.6175 - accuracy: 0.5032
<br />17184/25000 [===================>..........] - ETA: 17s - loss: 7.6193 - accuracy: 0.5031
<br />17216/25000 [===================>..........] - ETA: 17s - loss: 7.6212 - accuracy: 0.5030
<br />17248/25000 [===================>..........] - ETA: 17s - loss: 7.6266 - accuracy: 0.5026
<br />17280/25000 [===================>..........] - ETA: 17s - loss: 7.6276 - accuracy: 0.5025
<br />17312/25000 [===================>..........] - ETA: 17s - loss: 7.6276 - accuracy: 0.5025
<br />17344/25000 [===================>..........] - ETA: 17s - loss: 7.6295 - accuracy: 0.5024
<br />17376/25000 [===================>..........] - ETA: 17s - loss: 7.6304 - accuracy: 0.5024
<br />17408/25000 [===================>..........] - ETA: 17s - loss: 7.6305 - accuracy: 0.5024
<br />17440/25000 [===================>..........] - ETA: 17s - loss: 7.6306 - accuracy: 0.5024
<br />17472/25000 [===================>..........] - ETA: 17s - loss: 7.6289 - accuracy: 0.5025
<br />17504/25000 [====================>.........] - ETA: 17s - loss: 7.6281 - accuracy: 0.5025
<br />17536/25000 [====================>.........] - ETA: 17s - loss: 7.6343 - accuracy: 0.5021
<br />17568/25000 [====================>.........] - ETA: 16s - loss: 7.6335 - accuracy: 0.5022
<br />17600/25000 [====================>.........] - ETA: 16s - loss: 7.6335 - accuracy: 0.5022
<br />17632/25000 [====================>.........] - ETA: 16s - loss: 7.6362 - accuracy: 0.5020
<br />17664/25000 [====================>.........] - ETA: 16s - loss: 7.6388 - accuracy: 0.5018
<br />17696/25000 [====================>.........] - ETA: 16s - loss: 7.6380 - accuracy: 0.5019
<br />17728/25000 [====================>.........] - ETA: 16s - loss: 7.6363 - accuracy: 0.5020
<br />17760/25000 [====================>.........] - ETA: 16s - loss: 7.6381 - accuracy: 0.5019
<br />17792/25000 [====================>.........] - ETA: 16s - loss: 7.6399 - accuracy: 0.5017
<br />17824/25000 [====================>.........] - ETA: 16s - loss: 7.6374 - accuracy: 0.5019
<br />17856/25000 [====================>.........] - ETA: 16s - loss: 7.6374 - accuracy: 0.5019
<br />17888/25000 [====================>.........] - ETA: 16s - loss: 7.6409 - accuracy: 0.5017
<br />17920/25000 [====================>.........] - ETA: 16s - loss: 7.6384 - accuracy: 0.5018
<br />17952/25000 [====================>.........] - ETA: 16s - loss: 7.6393 - accuracy: 0.5018
<br />17984/25000 [====================>.........] - ETA: 15s - loss: 7.6402 - accuracy: 0.5017
<br />18016/25000 [====================>.........] - ETA: 15s - loss: 7.6445 - accuracy: 0.5014
<br />18048/25000 [====================>.........] - ETA: 15s - loss: 7.6454 - accuracy: 0.5014
<br />18080/25000 [====================>.........] - ETA: 15s - loss: 7.6480 - accuracy: 0.5012
<br />18112/25000 [====================>.........] - ETA: 15s - loss: 7.6514 - accuracy: 0.5010
<br />18144/25000 [====================>.........] - ETA: 15s - loss: 7.6523 - accuracy: 0.5009
<br />18176/25000 [====================>.........] - ETA: 15s - loss: 7.6540 - accuracy: 0.5008
<br />18208/25000 [====================>.........] - ETA: 15s - loss: 7.6540 - accuracy: 0.5008
<br />18240/25000 [====================>.........] - ETA: 15s - loss: 7.6549 - accuracy: 0.5008
<br />18272/25000 [====================>.........] - ETA: 15s - loss: 7.6565 - accuracy: 0.5007
<br />18304/25000 [====================>.........] - ETA: 15s - loss: 7.6582 - accuracy: 0.5005
<br />18336/25000 [=====================>........] - ETA: 15s - loss: 7.6608 - accuracy: 0.5004
<br />18368/25000 [=====================>........] - ETA: 15s - loss: 7.6649 - accuracy: 0.5001
<br />18400/25000 [=====================>........] - ETA: 15s - loss: 7.6625 - accuracy: 0.5003
<br />18432/25000 [=====================>........] - ETA: 14s - loss: 7.6625 - accuracy: 0.5003
<br />18464/25000 [=====================>........] - ETA: 14s - loss: 7.6616 - accuracy: 0.5003
<br />18496/25000 [=====================>........] - ETA: 14s - loss: 7.6625 - accuracy: 0.5003
<br />18528/25000 [=====================>........] - ETA: 14s - loss: 7.6608 - accuracy: 0.5004
<br />18560/25000 [=====================>........] - ETA: 14s - loss: 7.6584 - accuracy: 0.5005
<br />18592/25000 [=====================>........] - ETA: 14s - loss: 7.6567 - accuracy: 0.5006
<br />18624/25000 [=====================>........] - ETA: 14s - loss: 7.6567 - accuracy: 0.5006
<br />18656/25000 [=====================>........] - ETA: 14s - loss: 7.6568 - accuracy: 0.5006
<br />18688/25000 [=====================>........] - ETA: 14s - loss: 7.6527 - accuracy: 0.5009
<br />18720/25000 [=====================>........] - ETA: 14s - loss: 7.6519 - accuracy: 0.5010
<br />18752/25000 [=====================>........] - ETA: 14s - loss: 7.6470 - accuracy: 0.5013
<br />18784/25000 [=====================>........] - ETA: 14s - loss: 7.6438 - accuracy: 0.5015
<br />18816/25000 [=====================>........] - ETA: 14s - loss: 7.6414 - accuracy: 0.5016
<br />18848/25000 [=====================>........] - ETA: 14s - loss: 7.6422 - accuracy: 0.5016
<br />18880/25000 [=====================>........] - ETA: 13s - loss: 7.6414 - accuracy: 0.5016
<br />18912/25000 [=====================>........] - ETA: 13s - loss: 7.6447 - accuracy: 0.5014
<br />18944/25000 [=====================>........] - ETA: 13s - loss: 7.6440 - accuracy: 0.5015
<br />18976/25000 [=====================>........] - ETA: 13s - loss: 7.6448 - accuracy: 0.5014
<br />19008/25000 [=====================>........] - ETA: 13s - loss: 7.6368 - accuracy: 0.5019
<br />19040/25000 [=====================>........] - ETA: 13s - loss: 7.6336 - accuracy: 0.5022
<br />19072/25000 [=====================>........] - ETA: 13s - loss: 7.6345 - accuracy: 0.5021
<br />19104/25000 [=====================>........] - ETA: 13s - loss: 7.6361 - accuracy: 0.5020
<br />19136/25000 [=====================>........] - ETA: 13s - loss: 7.6402 - accuracy: 0.5017
<br />19168/25000 [======================>.......] - ETA: 13s - loss: 7.6370 - accuracy: 0.5019
<br />19200/25000 [======================>.......] - ETA: 13s - loss: 7.6347 - accuracy: 0.5021
<br />19232/25000 [======================>.......] - ETA: 13s - loss: 7.6331 - accuracy: 0.5022
<br />19264/25000 [======================>.......] - ETA: 13s - loss: 7.6340 - accuracy: 0.5021
<br />19296/25000 [======================>.......] - ETA: 12s - loss: 7.6348 - accuracy: 0.5021
<br />19328/25000 [======================>.......] - ETA: 12s - loss: 7.6365 - accuracy: 0.5020
<br />19360/25000 [======================>.......] - ETA: 12s - loss: 7.6341 - accuracy: 0.5021
<br />19392/25000 [======================>.......] - ETA: 12s - loss: 7.6326 - accuracy: 0.5022
<br />19424/25000 [======================>.......] - ETA: 12s - loss: 7.6327 - accuracy: 0.5022
<br />19456/25000 [======================>.......] - ETA: 12s - loss: 7.6343 - accuracy: 0.5021
<br />19488/25000 [======================>.......] - ETA: 12s - loss: 7.6367 - accuracy: 0.5019
<br />19520/25000 [======================>.......] - ETA: 12s - loss: 7.6399 - accuracy: 0.5017
<br />19552/25000 [======================>.......] - ETA: 12s - loss: 7.6400 - accuracy: 0.5017
<br />19584/25000 [======================>.......] - ETA: 12s - loss: 7.6384 - accuracy: 0.5018
<br />19616/25000 [======================>.......] - ETA: 12s - loss: 7.6432 - accuracy: 0.5015
<br />19648/25000 [======================>.......] - ETA: 12s - loss: 7.6416 - accuracy: 0.5016
<br />19680/25000 [======================>.......] - ETA: 12s - loss: 7.6440 - accuracy: 0.5015
<br />19712/25000 [======================>.......] - ETA: 12s - loss: 7.6410 - accuracy: 0.5017
<br />19744/25000 [======================>.......] - ETA: 11s - loss: 7.6402 - accuracy: 0.5017
<br />19776/25000 [======================>.......] - ETA: 11s - loss: 7.6410 - accuracy: 0.5017
<br />19808/25000 [======================>.......] - ETA: 11s - loss: 7.6434 - accuracy: 0.5015
<br />19840/25000 [======================>.......] - ETA: 11s - loss: 7.6380 - accuracy: 0.5019
<br />19872/25000 [======================>.......] - ETA: 11s - loss: 7.6388 - accuracy: 0.5018
<br />19904/25000 [======================>.......] - ETA: 11s - loss: 7.6381 - accuracy: 0.5019
<br />19936/25000 [======================>.......] - ETA: 11s - loss: 7.6351 - accuracy: 0.5021
<br />19968/25000 [======================>.......] - ETA: 11s - loss: 7.6374 - accuracy: 0.5019
<br />20000/25000 [=======================>......] - ETA: 11s - loss: 7.6390 - accuracy: 0.5018
<br />20032/25000 [=======================>......] - ETA: 11s - loss: 7.6383 - accuracy: 0.5018
<br />20064/25000 [=======================>......] - ETA: 11s - loss: 7.6361 - accuracy: 0.5020
<br />20096/25000 [=======================>......] - ETA: 11s - loss: 7.6330 - accuracy: 0.5022
<br />20128/25000 [=======================>......] - ETA: 11s - loss: 7.6331 - accuracy: 0.5022
<br />20160/25000 [=======================>......] - ETA: 11s - loss: 7.6339 - accuracy: 0.5021
<br />20192/25000 [=======================>......] - ETA: 10s - loss: 7.6385 - accuracy: 0.5018
<br />20224/25000 [=======================>......] - ETA: 10s - loss: 7.6371 - accuracy: 0.5019
<br />20256/25000 [=======================>......] - ETA: 10s - loss: 7.6356 - accuracy: 0.5020
<br />20288/25000 [=======================>......] - ETA: 10s - loss: 7.6356 - accuracy: 0.5020
<br />20320/25000 [=======================>......] - ETA: 10s - loss: 7.6364 - accuracy: 0.5020
<br />20352/25000 [=======================>......] - ETA: 10s - loss: 7.6357 - accuracy: 0.5020
<br />20384/25000 [=======================>......] - ETA: 10s - loss: 7.6365 - accuracy: 0.5020
<br />20416/25000 [=======================>......] - ETA: 10s - loss: 7.6358 - accuracy: 0.5020
<br />20448/25000 [=======================>......] - ETA: 10s - loss: 7.6351 - accuracy: 0.5021
<br />20480/25000 [=======================>......] - ETA: 10s - loss: 7.6329 - accuracy: 0.5022
<br />20512/25000 [=======================>......] - ETA: 10s - loss: 7.6360 - accuracy: 0.5020
<br />20544/25000 [=======================>......] - ETA: 10s - loss: 7.6353 - accuracy: 0.5020
<br />20576/25000 [=======================>......] - ETA: 10s - loss: 7.6361 - accuracy: 0.5020
<br />20608/25000 [=======================>......] - ETA: 9s - loss: 7.6331 - accuracy: 0.5022 
<br />20640/25000 [=======================>......] - ETA: 9s - loss: 7.6287 - accuracy: 0.5025
<br />20672/25000 [=======================>......] - ETA: 9s - loss: 7.6325 - accuracy: 0.5022
<br />20704/25000 [=======================>......] - ETA: 9s - loss: 7.6311 - accuracy: 0.5023
<br />20736/25000 [=======================>......] - ETA: 9s - loss: 7.6326 - accuracy: 0.5022
<br />20768/25000 [=======================>......] - ETA: 9s - loss: 7.6334 - accuracy: 0.5022
<br />20800/25000 [=======================>......] - ETA: 9s - loss: 7.6357 - accuracy: 0.5020
<br />20832/25000 [=======================>......] - ETA: 9s - loss: 7.6372 - accuracy: 0.5019
<br />20864/25000 [========================>.....] - ETA: 9s - loss: 7.6416 - accuracy: 0.5016
<br />20896/25000 [========================>.....] - ETA: 9s - loss: 7.6446 - accuracy: 0.5014
<br />20928/25000 [========================>.....] - ETA: 9s - loss: 7.6476 - accuracy: 0.5012
<br />20960/25000 [========================>.....] - ETA: 9s - loss: 7.6498 - accuracy: 0.5011
<br />20992/25000 [========================>.....] - ETA: 9s - loss: 7.6505 - accuracy: 0.5010
<br />21024/25000 [========================>.....] - ETA: 9s - loss: 7.6513 - accuracy: 0.5010
<br />21056/25000 [========================>.....] - ETA: 8s - loss: 7.6535 - accuracy: 0.5009
<br />21088/25000 [========================>.....] - ETA: 8s - loss: 7.6521 - accuracy: 0.5009
<br />21120/25000 [========================>.....] - ETA: 8s - loss: 7.6528 - accuracy: 0.5009
<br />21152/25000 [========================>.....] - ETA: 8s - loss: 7.6543 - accuracy: 0.5008
<br />21184/25000 [========================>.....] - ETA: 8s - loss: 7.6521 - accuracy: 0.5009
<br />21216/25000 [========================>.....] - ETA: 8s - loss: 7.6507 - accuracy: 0.5010
<br />21248/25000 [========================>.....] - ETA: 8s - loss: 7.6515 - accuracy: 0.5010
<br />21280/25000 [========================>.....] - ETA: 8s - loss: 7.6522 - accuracy: 0.5009
<br />21312/25000 [========================>.....] - ETA: 8s - loss: 7.6508 - accuracy: 0.5010
<br />21344/25000 [========================>.....] - ETA: 8s - loss: 7.6515 - accuracy: 0.5010
<br />21376/25000 [========================>.....] - ETA: 8s - loss: 7.6551 - accuracy: 0.5007
<br />21408/25000 [========================>.....] - ETA: 8s - loss: 7.6552 - accuracy: 0.5007
<br />21440/25000 [========================>.....] - ETA: 8s - loss: 7.6573 - accuracy: 0.5006
<br />21472/25000 [========================>.....] - ETA: 8s - loss: 7.6638 - accuracy: 0.5002
<br />21504/25000 [========================>.....] - ETA: 7s - loss: 7.6623 - accuracy: 0.5003
<br />21536/25000 [========================>.....] - ETA: 7s - loss: 7.6631 - accuracy: 0.5002
<br />21568/25000 [========================>.....] - ETA: 7s - loss: 7.6609 - accuracy: 0.5004
<br />21600/25000 [========================>.....] - ETA: 7s - loss: 7.6617 - accuracy: 0.5003
<br />21632/25000 [========================>.....] - ETA: 7s - loss: 7.6602 - accuracy: 0.5004
<br />21664/25000 [========================>.....] - ETA: 7s - loss: 7.6574 - accuracy: 0.5006
<br />21696/25000 [=========================>....] - ETA: 7s - loss: 7.6581 - accuracy: 0.5006
<br />21728/25000 [=========================>....] - ETA: 7s - loss: 7.6582 - accuracy: 0.5006
<br />21760/25000 [=========================>....] - ETA: 7s - loss: 7.6582 - accuracy: 0.5006
<br />21792/25000 [=========================>....] - ETA: 7s - loss: 7.6582 - accuracy: 0.5006
<br />21824/25000 [=========================>....] - ETA: 7s - loss: 7.6575 - accuracy: 0.5006
<br />21856/25000 [=========================>....] - ETA: 7s - loss: 7.6561 - accuracy: 0.5007
<br />21888/25000 [=========================>....] - ETA: 7s - loss: 7.6540 - accuracy: 0.5008
<br />21920/25000 [=========================>....] - ETA: 7s - loss: 7.6547 - accuracy: 0.5008
<br />21952/25000 [=========================>....] - ETA: 6s - loss: 7.6520 - accuracy: 0.5010
<br />21984/25000 [=========================>....] - ETA: 6s - loss: 7.6527 - accuracy: 0.5009
<br />22016/25000 [=========================>....] - ETA: 6s - loss: 7.6492 - accuracy: 0.5011
<br />22048/25000 [=========================>....] - ETA: 6s - loss: 7.6492 - accuracy: 0.5011
<br />22080/25000 [=========================>....] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
<br />22112/25000 [=========================>....] - ETA: 6s - loss: 7.6493 - accuracy: 0.5011
<br />22144/25000 [=========================>....] - ETA: 6s - loss: 7.6445 - accuracy: 0.5014
<br />22176/25000 [=========================>....] - ETA: 6s - loss: 7.6431 - accuracy: 0.5015
<br />22208/25000 [=========================>....] - ETA: 6s - loss: 7.6425 - accuracy: 0.5016
<br />22240/25000 [=========================>....] - ETA: 6s - loss: 7.6411 - accuracy: 0.5017
<br />22272/25000 [=========================>....] - ETA: 6s - loss: 7.6398 - accuracy: 0.5018
<br />22304/25000 [=========================>....] - ETA: 6s - loss: 7.6398 - accuracy: 0.5017
<br />22336/25000 [=========================>....] - ETA: 6s - loss: 7.6447 - accuracy: 0.5014
<br />22368/25000 [=========================>....] - ETA: 5s - loss: 7.6447 - accuracy: 0.5014
<br />22400/25000 [=========================>....] - ETA: 5s - loss: 7.6461 - accuracy: 0.5013
<br />22432/25000 [=========================>....] - ETA: 5s - loss: 7.6482 - accuracy: 0.5012
<br />22464/25000 [=========================>....] - ETA: 5s - loss: 7.6489 - accuracy: 0.5012
<br />22496/25000 [=========================>....] - ETA: 5s - loss: 7.6469 - accuracy: 0.5013
<br />22528/25000 [==========================>...] - ETA: 5s - loss: 7.6448 - accuracy: 0.5014
<br />22560/25000 [==========================>...] - ETA: 5s - loss: 7.6469 - accuracy: 0.5013
<br />22592/25000 [==========================>...] - ETA: 5s - loss: 7.6483 - accuracy: 0.5012
<br />22624/25000 [==========================>...] - ETA: 5s - loss: 7.6470 - accuracy: 0.5013
<br />22656/25000 [==========================>...] - ETA: 5s - loss: 7.6463 - accuracy: 0.5013
<br />22688/25000 [==========================>...] - ETA: 5s - loss: 7.6518 - accuracy: 0.5010
<br />22720/25000 [==========================>...] - ETA: 5s - loss: 7.6504 - accuracy: 0.5011
<br />22752/25000 [==========================>...] - ETA: 5s - loss: 7.6504 - accuracy: 0.5011
<br />22784/25000 [==========================>...] - ETA: 5s - loss: 7.6498 - accuracy: 0.5011
<br />22816/25000 [==========================>...] - ETA: 4s - loss: 7.6525 - accuracy: 0.5009
<br />22848/25000 [==========================>...] - ETA: 4s - loss: 7.6552 - accuracy: 0.5007
<br />22880/25000 [==========================>...] - ETA: 4s - loss: 7.6539 - accuracy: 0.5008
<br />22912/25000 [==========================>...] - ETA: 4s - loss: 7.6532 - accuracy: 0.5009
<br />22944/25000 [==========================>...] - ETA: 4s - loss: 7.6553 - accuracy: 0.5007
<br />22976/25000 [==========================>...] - ETA: 4s - loss: 7.6553 - accuracy: 0.5007
<br />23008/25000 [==========================>...] - ETA: 4s - loss: 7.6553 - accuracy: 0.5007
<br />23040/25000 [==========================>...] - ETA: 4s - loss: 7.6573 - accuracy: 0.5006
<br />23072/25000 [==========================>...] - ETA: 4s - loss: 7.6553 - accuracy: 0.5007
<br />23104/25000 [==========================>...] - ETA: 4s - loss: 7.6573 - accuracy: 0.5006
<br />23136/25000 [==========================>...] - ETA: 4s - loss: 7.6573 - accuracy: 0.5006
<br />23168/25000 [==========================>...] - ETA: 4s - loss: 7.6574 - accuracy: 0.5006
<br />23200/25000 [==========================>...] - ETA: 4s - loss: 7.6567 - accuracy: 0.5006
<br />23232/25000 [==========================>...] - ETA: 4s - loss: 7.6574 - accuracy: 0.5006
<br />23264/25000 [==========================>...] - ETA: 3s - loss: 7.6587 - accuracy: 0.5005
<br />23296/25000 [==========================>...] - ETA: 3s - loss: 7.6594 - accuracy: 0.5005
<br />23328/25000 [==========================>...] - ETA: 3s - loss: 7.6587 - accuracy: 0.5005
<br />23360/25000 [===========================>..] - ETA: 3s - loss: 7.6614 - accuracy: 0.5003
<br />23392/25000 [===========================>..] - ETA: 3s - loss: 7.6620 - accuracy: 0.5003
<br />23424/25000 [===========================>..] - ETA: 3s - loss: 7.6614 - accuracy: 0.5003
<br />23456/25000 [===========================>..] - ETA: 3s - loss: 7.6614 - accuracy: 0.5003
<br />23488/25000 [===========================>..] - ETA: 3s - loss: 7.6620 - accuracy: 0.5003
<br />23520/25000 [===========================>..] - ETA: 3s - loss: 7.6601 - accuracy: 0.5004
<br />23552/25000 [===========================>..] - ETA: 3s - loss: 7.6588 - accuracy: 0.5005
<br />23584/25000 [===========================>..] - ETA: 3s - loss: 7.6569 - accuracy: 0.5006
<br />23616/25000 [===========================>..] - ETA: 3s - loss: 7.6556 - accuracy: 0.5007
<br />23648/25000 [===========================>..] - ETA: 3s - loss: 7.6595 - accuracy: 0.5005
<br />23680/25000 [===========================>..] - ETA: 3s - loss: 7.6576 - accuracy: 0.5006
<br />23712/25000 [===========================>..] - ETA: 2s - loss: 7.6576 - accuracy: 0.5006
<br />23744/25000 [===========================>..] - ETA: 2s - loss: 7.6569 - accuracy: 0.5006
<br />23776/25000 [===========================>..] - ETA: 2s - loss: 7.6576 - accuracy: 0.5006
<br />23808/25000 [===========================>..] - ETA: 2s - loss: 7.6570 - accuracy: 0.5006
<br />23840/25000 [===========================>..] - ETA: 2s - loss: 7.6583 - accuracy: 0.5005
<br />23872/25000 [===========================>..] - ETA: 2s - loss: 7.6589 - accuracy: 0.5005
<br />23904/25000 [===========================>..] - ETA: 2s - loss: 7.6589 - accuracy: 0.5005
<br />23936/25000 [===========================>..] - ETA: 2s - loss: 7.6621 - accuracy: 0.5003
<br />23968/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
<br />24000/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
<br />24032/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
<br />24064/25000 [===========================>..] - ETA: 2s - loss: 7.6622 - accuracy: 0.5003
<br />24096/25000 [===========================>..] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
<br />24128/25000 [===========================>..] - ETA: 1s - loss: 7.6698 - accuracy: 0.4998
<br />24160/25000 [===========================>..] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
<br />24192/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
<br />24224/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
<br />24256/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
<br />24288/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
<br />24320/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
<br />24352/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
<br />24384/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
<br />24416/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
<br />24448/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
<br />24480/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
<br />24512/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
<br />24544/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
<br />24576/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
<br />24608/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
<br />24640/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
<br />24672/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
<br />24704/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
<br />24736/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
<br />24768/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
<br />24800/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
<br />24832/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
<br />24864/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
<br />24896/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
<br />24928/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
<br />24960/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
<br />24992/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
<br />25000/25000 [==============================] - 67s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
<br />Loading data...
<br />Using TensorFlow backend.
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb 
<br />
<br />Deprecaton set to False
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 16, [Traceback at line 1768](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1768)<br />1768..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      3[0m [0;32mimport[0m [0mjson[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      4[0m [0mdata_path[0m [0;34m=[0m [0;34m'../mlmodels/dataset/json/hyper_titanic_randomForest.json'[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m----> 5[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mdata_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      6[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      7[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpdict[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '../mlmodels/dataset/json/hyper_titanic_randomForest.json'
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//mnist_mlmodels_.ipynb 
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 17, [Traceback at line 1786](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1786)<br />1786..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'google.colab'
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//gluon_automl_titanic.ipynb 
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 18, [Traceback at line 1801](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1801)<br />1801..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//gluon_automl_titanic.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      8[0m     [0mchoice[0m[0;34m=[0m[0;34m'json'[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      9[0m     [0mconfig_mode[0m[0;34m=[0m [0;34m'test'[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 10[0;31m     [0mdata_path[0m[0;34m=[0m [0;34m'../mlmodels/dataset/json/gluon_automl.json'[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     11[0m )
<br />
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/model_gluon/gluon_automl.py[0m in [0;36mget_params[0;34m(choice, data_path, config_mode, **kw)[0m
<br />[1;32m     80[0m             __file__)).parent.parent / "model_gluon/gluon_automl.json" if data_path == "dataset/" else data_path
<br />[1;32m     81[0m [0;34m[0m[0m
<br />[0;32m---> 82[0;31m         [0;32mwith[0m [0mopen[0m[0;34m([0m[0mdata_path[0m[0;34m,[0m [0mencoding[0m[0;34m=[0m[0;34m'utf-8'[0m[0;34m)[0m [0;32mas[0m [0mconfig_f[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     83[0m             [0mconfig[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mconfig_f[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m     84[0m             [0mconfig[0m [0;34m=[0m [0mconfig[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '../mlmodels/dataset/json/gluon_automl.json'
<br />/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB
<br />  Optimizer.opt_registry[name].__name__))
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//tensorflow__lstm_json.ipynb 
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 19, [Traceback at line 1827](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1827)<br />1827..[0;31mNameError[0m                                 Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//tensorflow__lstm_json.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      5[0m [0;32mimport[0m [0mjson[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      6[0m [0;34m[0m[0m
<br />[0;32m----> 7[0;31m [0mprint[0m[0;34m([0m [0mos[0m[0;34m.[0m[0mgetcwd[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m
<br />[0;31mNameError[0m: name 'os' is not defined
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//sklearn.ipynb 
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 20, [Traceback at line 1843](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1843)<br />1843..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     71[0m         [0mmodel_name[0m [0;34m=[0m [0mmodel_uri[0m[0;34m.[0m[0mreplace[0m[0;34m([0m[0;34m".py"[0m[0;34m,[0m [0;34m""[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 72[0;31m         [0mmodule[0m [0;34m=[0m [0mimport_module[0m[0;34m([0m[0;34mf"mlmodels.{model_name}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     73[0m         [0;31m# module    = import_module("mlmodels.model_tf.1_lstm")[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
<br />[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m    127[0m [0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m
<br />
<br />[0;31mModuleNotFoundError[0m: No module named 'mlmodels.model_sklearn.sklearn'
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 21, [Traceback at line 1864](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1864)<br />1864..[0;31mIndexError[0m                                Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mIndexError[0m: tuple index out of range
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 22, [Traceback at line 1874](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1874)<br />1874..[0;31mNameError[0m                                 Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//sklearn.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      1[0m [0;32mfrom[0m [0mmlmodels[0m[0;34m.[0m[0mmodels[0m [0;32mimport[0m [0mmodule_load[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      2[0m [0;34m[0m[0m
<br />[0;32m----> 3[0;31m [0mmodule[0m        [0;34m=[0m  [0mmodule_load[0m[0;34m([0m [0mmodel_uri[0m[0;34m=[0m [0mmodel_uri[0m [0;34m)[0m                           [0;31m# Load file definition[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      4[0m [0mmodel[0m         [0;34m=[0m  [0mmodule[0m[0;34m.[0m[0mModel[0m[0;34m([0m[0mmodel_pars[0m[0;34m=[0m[0mmodel_pars[0m[0;34m,[0m [0mdata_pars[0m[0;34m=[0m[0mdata_pars[0m[0;34m,[0m [0mcompute_pars[0m[0;34m=[0m[0mcompute_pars[0m[0;34m)[0m             [0;31m# Create Model instance[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      5[0m [0mmodel[0m[0;34m,[0m [0msess[0m   [0;34m=[0m  [0mmodule[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mdata_pars[0m[0;34m=[0m[0mdata_pars[0m[0;34m,[0m [0mcompute_pars[0m[0;34m=[0m[0mcompute_pars[0m[0;34m,[0m [0mout_pars[0m[0;34m=[0m[0mout_pars[0m[0;34m)[0m          [0;31m# fit the model[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     87[0m [0;34m[0m[0m
<br />[1;32m     88[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me2[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 89[0;31m             [0;32mraise[0m [0mNameError[0m[0;34m([0m[0;34mf"Module {model_name} notfound, {e1}, {e2}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     90[0m [0;34m[0m[0m
<br />[1;32m     91[0m     [0;32mif[0m [0mverbose[0m[0;34m:[0m [0mprint[0m[0;34m([0m[0mmodule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mNameError[0m: Module model_sklearn.sklearn notfound, No module named 'mlmodels.model_sklearn.sklearn', tuple index out of range
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_titanic.ipynb 
<br />
<br />Deprecaton set to False
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 23, [Traceback at line 1900](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1900)<br />1900..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example//lightgbm_titanic.ipynb[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      1[0m [0mdata_path[0m [0;34m=[0m [0;34m'hyper_lightgbm_titanic.json'[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      2[0m [0;34m[0m[0m
<br />[0;32m----> 3[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mdata_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      4[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      5[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpdict[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'hyper_lightgbm_titanic.json'
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vision_mnist.py 
<br />
<br />[0;36m  File [0;32m"https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/example/vision_mnist.py"[0;36m, line [0;32m15[0m
<br />[0;31m    !git clone https://github.com/ahmed3bbas/mlmodels.git[0m
<br />[0m    ^[0m
<br />[0;31mSyntaxError[0m[0;31m:[0m invalid syntax
<br />
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//benchmark_timeseries_m4.py 
<br />
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//arun_hyper.py 
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 24, [Traceback at line 1939](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1939)<br />1939..[0;31mNameError[0m                                 Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example/arun_hyper.py[0m in [0;36m<module>[0;34m[0m
<br />[1;32m      3[0m [0;32mfrom[0m [0mmlmodels[0m[0;34m.[0m[0mmodels[0m [0;32mimport[0m [0mmodule_load[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m      4[0m [0;32mfrom[0m [0mmlmodels[0m[0;34m.[0m[0mutil[0m [0;32mimport[0m [0mpath_norm_dict[0m[0;34m,[0m [0mpath_norm[0m[0;34m,[0m [0mparams_json_load[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m----> 5[0;31m [0mprint[0m[0;34m([0m[0mmlmodels[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m      6[0m [0;34m[0m[0m
<br />[1;32m      7[0m [0;34m[0m[0m
<br />
<br />[0;31mNameError[0m: name 'mlmodels' is not defined
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//lightgbm_glass.py 
<br />
<br />Deprecaton set to False
<br />/home/runner/work/mlmodels/mlmodels
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 25, [Traceback at line 1959](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1959)<br />1959..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example/lightgbm_glass.py[0m in [0;36m<module>[0;34m[0m
<br />[1;32m     20[0m [0;34m[0m[0m
<br />[1;32m     21[0m [0;34m[0m[0m
<br />[0;32m---> 22[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m [0mconfig_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     23[0m [0mprint[0m[0;34m([0m[0mpars[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m     24[0m [0;34m[0m[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'lightgbm_glass.json'
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//benchmark_timeseries_m5.py 
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 26, [Traceback at line 1977](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L1977)<br />1977..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py[0m in [0;36m<module>[0;34m[0m
<br />[1;32m     84[0m [0;34m[0m[0m
<br />[1;32m     85[0m """
<br />[0;32m---> 86[0;31m [0mcalendar[0m               [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/calendar.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     87[0m [0msales_train_val[0m        [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sales_train_val.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m     88[0m [0msample_submission[0m      [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sample_submission.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36mparser_f[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)[0m
<br />[1;32m    683[0m         )
<br />[1;32m    684[0m [0;34m[0m[0m
<br />[0;32m--> 685[0;31m         [0;32mreturn[0m [0m_read[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m    686[0m [0;34m[0m[0m
<br />[1;32m    687[0m     [0mparser_f[0m[0;34m.[0m[0m__name__[0m [0;34m=[0m [0mname[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m_read[0;34m(filepath_or_buffer, kwds)[0m
<br />[1;32m    455[0m [0;34m[0m[0m
<br />[1;32m    456[0m     [0;31m# Create the parser.[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m--> 457[0;31m     [0mparser[0m [0;34m=[0m [0mTextFileReader[0m[0;34m([0m[0mfp_or_buf[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m    458[0m [0;34m[0m[0m
<br />[1;32m    459[0m     [0;32mif[0m [0mchunksize[0m [0;32mor[0m [0miterator[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m__init__[0;34m(self, f, engine, **kwds)[0m
<br />[1;32m    893[0m             [0mself[0m[0;34m.[0m[0moptions[0m[0;34m[[0m[0;34m"has_index_names"[0m[0;34m][0m [0;34m=[0m [0mkwds[0m[0;34m[[0m[0;34m"has_index_names"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m    894[0m [0;34m[0m[0m
<br />[0;32m--> 895[0;31m         [0mself[0m[0;34m.[0m[0m_make_engine[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mengine[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m    896[0m [0;34m[0m[0m
<br />[1;32m    897[0m     [0;32mdef[0m [0mclose[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m_make_engine[0;34m(self, engine)[0m
<br />[1;32m   1133[0m     [0;32mdef[0m [0m_make_engine[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mengine[0m[0;34m=[0m[0;34m"c"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m   1134[0m         [0;32mif[0m [0mengine[0m [0;34m==[0m [0;34m"c"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m-> 1135[0;31m             [0mself[0m[0;34m.[0m[0m_engine[0m [0;34m=[0m [0mCParserWrapper[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mf[0m[0;34m,[0m [0;34m**[0m[0mself[0m[0;34m.[0m[0moptions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m   1136[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m   1137[0m             [0;32mif[0m [0mengine[0m [0;34m==[0m [0;34m"python"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m__init__[0;34m(self, src, **kwds)[0m
<br />[1;32m   1915[0m         [0mkwds[0m[0;34m[[0m[0;34m"usecols"[0m[0;34m][0m [0;34m=[0m [0mself[0m[0;34m.[0m[0musecols[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m   1916[0m [0;34m[0m[0m
<br />[0;32m-> 1917[0;31m         [0mself[0m[0;34m.[0m[0m_reader[0m [0;34m=[0m [0mparsers[0m[0;34m.[0m[0mTextReader[0m[0;34m([0m[0msrc[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m   1918[0m         [0mself[0m[0;34m.[0m[0munnamed_cols[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_reader[0m[0;34m.[0m[0munnamed_cols[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m   1919[0m [0;34m[0m[0m
<br />
<br />[0;32mpandas/_libs/parsers.pyx[0m in [0;36mpandas._libs.parsers.TextReader.__cinit__[0;34m()[0m
<br />
<br />[0;32mpandas/_libs/parsers.pyx[0m in [0;36mpandas._libs.parsers.TextReader._setup_parser_source[0;34m()[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] File b'./m5-forecasting-accuracy/calendar.csv' does not exist: b'./m5-forecasting-accuracy/calendar.csv'
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//arun_model.py 
<br />
<br /><module 'mlmodels' from 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/__init__.py'>
<br />https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras/ardmn.json
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 27, [Traceback at line 2036](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L2036)<br />2036..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example/arun_model.py[0m in [0;36m<module>[0;34m[0m
<br />[1;32m     25[0m [0;31m# Model Parameters[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m     26[0m [0;31m# model_pars, data_pars, compute_pars, out_pars[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 27[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m[0mconfig_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     28[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m     29[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpdict[0m   [0;34m)[0m   [0;31m###Normalize path[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed/mlmodels/model_keras/ardmn.json'
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/benchmark_timeseries_m4.py 
<br />
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/benchmark_timeseries_m5.py 
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 28, [Traceback at line 2062](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter.py#L2062)<br />2062..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py[0m in [0;36m<module>[0;34m[0m
<br />[1;32m     84[0m [0;34m[0m[0m
<br />[1;32m     85[0m """
<br />[0;32m---> 86[0;31m [0mcalendar[0m               [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/calendar.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     87[0m [0msales_train_val[0m        [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sales_train_val.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m     88[0m [0msample_submission[0m      [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sample_submission.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36mparser_f[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)[0m
<br />[1;32m    683[0m         )
<br />[1;32m    684[0m [0;34m[0m[0m
<br />[0;32m--> 685[0;31m         [0;32mreturn[0m [0m_read[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m    686[0m [0;34m[0m[0m
<br />[1;32m    687[0m     [0mparser_f[0m[0;34m.[0m[0m__name__[0m [0;34m=[0m [0mname[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m_read[0;34m(filepath_or_buffer, kwds)[0m
<br />[1;32m    455[0m [0;34m[0m[0m
<br />[1;32m    456[0m     [0;31m# Create the parser.[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m--> 457[0;31m     [0mparser[0m [0;34m=[0m [0mTextFileReader[0m[0;34m([0m[0mfp_or_buf[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m    458[0m [0;34m[0m[0m
<br />[1;32m    459[0m     [0;32mif[0m [0mchunksize[0m [0;32mor[0m [0miterator[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m__init__[0;34m(self, f, engine, **kwds)[0m
<br />[1;32m    893[0m             [0mself[0m[0;34m.[0m[0moptions[0m[0;34m[[0m[0;34m"has_index_names"[0m[0;34m][0m [0;34m=[0m [0mkwds[0m[0;34m[[0m[0;34m"has_index_names"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m    894[0m [0;34m[0m[0m
<br />[0;32m--> 895[0;31m         [0mself[0m[0;34m.[0m[0m_make_engine[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mengine[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m    896[0m [0;34m[0m[0m
<br />[1;32m    897[0m     [0;32mdef[0m [0mclose[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m_make_engine[0;34m(self, engine)[0m
<br />[1;32m   1133[0m     [0;32mdef[0m [0m_make_engine[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mengine[0m[0;34m=[0m[0;34m"c"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m   1134[0m         [0;32mif[0m [0mengine[0m [0;34m==[0m [0;34m"c"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m-> 1135[0;31m             [0mself[0m[0;34m.[0m[0m_engine[0m [0;34m=[0m [0mCParserWrapper[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mf[0m[0;34m,[0m [0;34m**[0m[0mself[0m[0;34m.[0m[0moptions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m   1136[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m   1137[0m             [0;32mif[0m [0mengine[0m [0;34m==[0m [0;34m"python"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m__init__[0;34m(self, src, **kwds)[0m
<br />[1;32m   1915[0m         [0mkwds[0m[0;34m[[0m[0;34m"usecols"[0m[0;34m][0m [0;34m=[0m [0mself[0m[0;34m.[0m[0musecols[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m   1916[0m [0;34m[0m[0m
<br />[0;32m-> 1917[0;31m         [0mself[0m[0;34m.[0m[0m_reader[0m [0;34m=[0m [0mparsers[0m[0;34m.[0m[0mTextReader[0m[0;34m([0m[0msrc[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m   1918[0m         [0mself[0m[0;34m.[0m[0munnamed_cols[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_reader[0m[0;34m.[0m[0munnamed_cols[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m   1919[0m [0;34m[0m[0m
<br />
<br />[0;32mpandas/_libs/parsers.pyx[0m in [0;36mpandas._libs.parsers.TextReader.__cinit__[0;34m()[0m
<br />
<br />[0;32mpandas/_libs/parsers.pyx[0m in [0;36mpandas._libs.parsers.TextReader._setup_parser_source[0;34m()[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] File b'./m5-forecasting-accuracy/calendar.csv' does not exist: b'./m5-forecasting-accuracy/calendar.csv'
