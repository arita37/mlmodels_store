## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py


### Error 1, [Traceback at line 42](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L42)<br />42..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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



### Error 2, [Traceback at line 63](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L63)<br />63..[0;31mIndexError[0m                                Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mIndexError[0m: tuple index out of range
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 3, [Traceback at line 73](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L73)<br />73..[0;31mNameError[0m                                 Traceback (most recent call last)
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



### Error 4, [Traceback at line 98](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L98)<br />98..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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



### Error 5, [Traceback at line 116](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L116)<br />116..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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



### Error 6, [Traceback at line 137](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L137)<br />137..[0;31mIndexError[0m                                Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mIndexError[0m: tuple index out of range
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 7, [Traceback at line 147](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L147)<br />147..[0;31mNameError[0m                                 Traceback (most recent call last)
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
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//fashion_MNIST_mlmodels.ipynb 
<br />
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 8, [Traceback at line 172](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L172)<br />172..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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



### Error 9, [Traceback at line 188](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L188)<br />188..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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



### Error 10, [Traceback at line 206](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L206)<br />206..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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
<br />	Data preprocessing and feature engineering runtime = 0.23s ...
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



### Error 11, [Traceback at line 271](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L271)<br />271..Traceback (most recent call last):
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



### Error 12, [Traceback at line 442](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L442)<br />442..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/models.py", line 523, in main
<br />    test_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/models.py", line 453, in test_cli
<br />    test_module(arg.model_uri, param_pars=param_pars)  # '1_lstm'
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_gluon/gluon_automl.py", line 109, in get_params
<br />    return model_pars, data_pars, compute_pars, out_pars
<br />UnboundLocalError: local variable 'model_pars' referenced before assignment



### Error 13, [Traceback at line 463](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L463)<br />463..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />non-resource variables are not supported in the long term
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
<br />https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
<br />https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />  <module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_tf/1_lstm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.model_tf.1_lstm.Model object at 0x7f5d8cbbfa20> 
<br />
<br />  #### Fit   ######################################################## 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br /> [-0.04966803  0.12734601  0.0713919   0.09146102 -0.02486509 -0.03130038]
<br /> [-0.17437214  0.11320665 -0.03698201 -0.03522266  0.04460657  0.11960361]
<br /> [ 0.32858816  0.18231621  0.0923181  -0.25516546  0.08681861  0.42660075]
<br /> [ 0.01532536  0.19387545  0.21226245 -0.22567159  0.06231602  0.26168162]
<br /> [ 0.25615865 -0.17229925  0.44683635 -0.12695831 -0.08932688 -0.19733104]
<br /> [ 0.11262126 -0.07373199  0.04900286 -0.01673576  0.16331075  0.20091949]
<br /> [-0.06842054  0.13006875 -0.11804266  0.17842568  0.05852774  0.06005628]
<br /> [ 0.05070845 -0.23191056 -0.08528733 -0.24100931  0.20484166  0.26471564]
<br /> [ 0.          0.          0.          0.          0.          0.        ]]
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />model_tf/1_lstm.py
<br />model_tf.1_lstm.py
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.5008648037910461, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-14 15:16:47.166151: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_tf/1_lstm/model'}
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
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.4822178892791271, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-14 15:16:48.297090: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/ztest/model_tf/1_lstm/model'}
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



### Error 14, [Traceback at line 847](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L847)<br />847..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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



### Error 15, [Traceback at line 862](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L862)<br />862..[0;31mNameError[0m                                 Traceback (most recent call last)
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
<br />    8192/17464789 [..............................] - ETA: 2s
<br /> 1556480/17464789 [=>............................] - ETA: 0s
<br /> 7069696/17464789 [===========>..................] - ETA: 0s
<br />15286272/17464789 [=========================>....] - ETA: 0s
<br />17465344/17464789 [==============================] - 0s 0us/step
<br />Pad sequences (samples x time)...
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />Use tf.where in 2.0, which has the same broadcast rule as np.where
<br />2020-05-14 15:16:59.310095: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
<br />2020-05-14 15:16:59.314038: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
<br />2020-05-14 15:16:59.314208: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e9fd1821b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
<br />2020-05-14 15:16:59.314223: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
<br />
<br />Train on 25000 samples, validate on 25000 samples
<br />Epoch 1/1
<br />
<br />   32/25000 [..............................] - ETA: 4:11 - loss: 6.2291 - accuracy: 0.5938
<br />   64/25000 [..............................] - ETA: 2:46 - loss: 7.1875 - accuracy: 0.5312
<br />   96/25000 [..............................] - ETA: 2:17 - loss: 6.8680 - accuracy: 0.5521
<br />  128/25000 [..............................] - ETA: 2:03 - loss: 6.9479 - accuracy: 0.5469
<br />  160/25000 [..............................] - ETA: 1:56 - loss: 6.8041 - accuracy: 0.5562
<br />  192/25000 [..............................] - ETA: 1:50 - loss: 6.4687 - accuracy: 0.5781
<br />  224/25000 [..............................] - ETA: 1:46 - loss: 6.2291 - accuracy: 0.5938
<br />  256/25000 [..............................] - ETA: 1:43 - loss: 6.2890 - accuracy: 0.5898
<br />  288/25000 [..............................] - ETA: 1:40 - loss: 6.4953 - accuracy: 0.5764
<br />  320/25000 [..............................] - ETA: 1:38 - loss: 6.6604 - accuracy: 0.5656
<br />  352/25000 [..............................] - ETA: 1:37 - loss: 6.7954 - accuracy: 0.5568
<br />  384/25000 [..............................] - ETA: 1:36 - loss: 6.7083 - accuracy: 0.5625
<br />  416/25000 [..............................] - ETA: 1:35 - loss: 6.7452 - accuracy: 0.5601
<br />  448/25000 [..............................] - ETA: 1:34 - loss: 6.8110 - accuracy: 0.5558
<br />  480/25000 [..............................] - ETA: 1:33 - loss: 6.9958 - accuracy: 0.5437
<br />  512/25000 [..............................] - ETA: 1:32 - loss: 7.0078 - accuracy: 0.5430
<br />  544/25000 [..............................] - ETA: 1:31 - loss: 7.1029 - accuracy: 0.5368
<br />  576/25000 [..............................] - ETA: 1:31 - loss: 7.0810 - accuracy: 0.5382
<br />  608/25000 [..............................] - ETA: 1:31 - loss: 7.0866 - accuracy: 0.5378
<br />  640/25000 [..............................] - ETA: 1:30 - loss: 7.1635 - accuracy: 0.5328
<br />  672/25000 [..............................] - ETA: 1:30 - loss: 7.1646 - accuracy: 0.5327
<br />  704/25000 [..............................] - ETA: 1:29 - loss: 7.2964 - accuracy: 0.5241
<br />  736/25000 [..............................] - ETA: 1:28 - loss: 7.3333 - accuracy: 0.5217
<br />  768/25000 [..............................] - ETA: 1:28 - loss: 7.3272 - accuracy: 0.5221
<br />  800/25000 [..............................] - ETA: 1:28 - loss: 7.3791 - accuracy: 0.5188
<br />  832/25000 [..............................] - ETA: 1:27 - loss: 7.2980 - accuracy: 0.5240
<br />  864/25000 [>.............................] - ETA: 1:28 - loss: 7.3294 - accuracy: 0.5220
<br />  896/25000 [>.............................] - ETA: 1:27 - loss: 7.3072 - accuracy: 0.5234
<br />  928/25000 [>.............................] - ETA: 1:27 - loss: 7.3196 - accuracy: 0.5226
<br />  960/25000 [>.............................] - ETA: 1:27 - loss: 7.3152 - accuracy: 0.5229
<br />  992/25000 [>.............................] - ETA: 1:26 - loss: 7.3420 - accuracy: 0.5212
<br /> 1024/25000 [>.............................] - ETA: 1:26 - loss: 7.4270 - accuracy: 0.5156
<br /> 1056/25000 [>.............................] - ETA: 1:26 - loss: 7.3472 - accuracy: 0.5208
<br /> 1088/25000 [>.............................] - ETA: 1:26 - loss: 7.3566 - accuracy: 0.5202
<br /> 1120/25000 [>.............................] - ETA: 1:26 - loss: 7.3928 - accuracy: 0.5179
<br /> 1152/25000 [>.............................] - ETA: 1:25 - loss: 7.3738 - accuracy: 0.5191
<br /> 1184/25000 [>.............................] - ETA: 1:25 - loss: 7.3688 - accuracy: 0.5194
<br /> 1216/25000 [>.............................] - ETA: 1:25 - loss: 7.3766 - accuracy: 0.5189
<br /> 1248/25000 [>.............................] - ETA: 1:25 - loss: 7.3840 - accuracy: 0.5184
<br /> 1280/25000 [>.............................] - ETA: 1:24 - loss: 7.3911 - accuracy: 0.5180
<br /> 1312/25000 [>.............................] - ETA: 1:24 - loss: 7.4446 - accuracy: 0.5145
<br /> 1344/25000 [>.............................] - ETA: 1:24 - loss: 7.4499 - accuracy: 0.5141
<br /> 1376/25000 [>.............................] - ETA: 1:24 - loss: 7.4326 - accuracy: 0.5153
<br /> 1408/25000 [>.............................] - ETA: 1:23 - loss: 7.4815 - accuracy: 0.5121
<br /> 1440/25000 [>.............................] - ETA: 1:23 - loss: 7.5175 - accuracy: 0.5097
<br /> 1472/25000 [>.............................] - ETA: 1:23 - loss: 7.4791 - accuracy: 0.5122
<br /> 1504/25000 [>.............................] - ETA: 1:23 - loss: 7.4831 - accuracy: 0.5120
<br /> 1536/25000 [>.............................] - ETA: 1:22 - loss: 7.4869 - accuracy: 0.5117
<br /> 1568/25000 [>.............................] - ETA: 1:22 - loss: 7.4417 - accuracy: 0.5147
<br /> 1600/25000 [>.............................] - ETA: 1:22 - loss: 7.4270 - accuracy: 0.5156
<br /> 1632/25000 [>.............................] - ETA: 1:22 - loss: 7.4317 - accuracy: 0.5153
<br /> 1664/25000 [>.............................] - ETA: 1:22 - loss: 7.4547 - accuracy: 0.5138
<br /> 1696/25000 [=>............................] - ETA: 1:21 - loss: 7.4677 - accuracy: 0.5130
<br /> 1728/25000 [=>............................] - ETA: 1:21 - loss: 7.5069 - accuracy: 0.5104
<br /> 1760/25000 [=>............................] - ETA: 1:21 - loss: 7.5011 - accuracy: 0.5108
<br /> 1792/25000 [=>............................] - ETA: 1:21 - loss: 7.4955 - accuracy: 0.5112
<br /> 1824/25000 [=>............................] - ETA: 1:20 - loss: 7.5321 - accuracy: 0.5088
<br /> 1856/25000 [=>............................] - ETA: 1:20 - loss: 7.5427 - accuracy: 0.5081
<br /> 1888/25000 [=>............................] - ETA: 1:20 - loss: 7.5610 - accuracy: 0.5069
<br /> 1920/25000 [=>............................] - ETA: 1:20 - loss: 7.5468 - accuracy: 0.5078
<br /> 1952/25000 [=>............................] - ETA: 1:20 - loss: 7.5645 - accuracy: 0.5067
<br /> 1984/25000 [=>............................] - ETA: 1:20 - loss: 7.5430 - accuracy: 0.5081
<br /> 2016/25000 [=>............................] - ETA: 1:19 - loss: 7.5221 - accuracy: 0.5094
<br /> 2048/25000 [=>............................] - ETA: 1:19 - loss: 7.5319 - accuracy: 0.5088
<br /> 2080/25000 [=>............................] - ETA: 1:19 - loss: 7.5044 - accuracy: 0.5106
<br /> 2112/25000 [=>............................] - ETA: 1:19 - loss: 7.4996 - accuracy: 0.5109
<br /> 2144/25000 [=>............................] - ETA: 1:19 - loss: 7.5236 - accuracy: 0.5093
<br /> 2176/25000 [=>............................] - ETA: 1:19 - loss: 7.5257 - accuracy: 0.5092
<br /> 2208/25000 [=>............................] - ETA: 1:18 - loss: 7.5347 - accuracy: 0.5086
<br /> 2240/25000 [=>............................] - ETA: 1:18 - loss: 7.5160 - accuracy: 0.5098
<br /> 2272/25000 [=>............................] - ETA: 1:18 - loss: 7.4844 - accuracy: 0.5119
<br /> 2304/25000 [=>............................] - ETA: 1:18 - loss: 7.4803 - accuracy: 0.5122
<br /> 2336/25000 [=>............................] - ETA: 1:18 - loss: 7.5025 - accuracy: 0.5107
<br /> 2368/25000 [=>............................] - ETA: 1:18 - loss: 7.5242 - accuracy: 0.5093
<br /> 2400/25000 [=>............................] - ETA: 1:17 - loss: 7.5197 - accuracy: 0.5096
<br /> 2432/25000 [=>............................] - ETA: 1:17 - loss: 7.5657 - accuracy: 0.5066
<br /> 2464/25000 [=>............................] - ETA: 1:17 - loss: 7.5608 - accuracy: 0.5069
<br /> 2496/25000 [=>............................] - ETA: 1:17 - loss: 7.5622 - accuracy: 0.5068
<br /> 2528/25000 [==>...........................] - ETA: 1:17 - loss: 7.5332 - accuracy: 0.5087
<br /> 2560/25000 [==>...........................] - ETA: 1:17 - loss: 7.4989 - accuracy: 0.5109
<br /> 2592/25000 [==>...........................] - ETA: 1:17 - loss: 7.5187 - accuracy: 0.5096
<br /> 2624/25000 [==>...........................] - ETA: 1:17 - loss: 7.5498 - accuracy: 0.5076
<br /> 2656/25000 [==>...........................] - ETA: 1:17 - loss: 7.5916 - accuracy: 0.5049
<br /> 2688/25000 [==>...........................] - ETA: 1:17 - loss: 7.6096 - accuracy: 0.5037
<br /> 2720/25000 [==>...........................] - ETA: 1:16 - loss: 7.5990 - accuracy: 0.5044
<br /> 2752/25000 [==>...........................] - ETA: 1:16 - loss: 7.6053 - accuracy: 0.5040
<br /> 2784/25000 [==>...........................] - ETA: 1:16 - loss: 7.5950 - accuracy: 0.5047
<br /> 2816/25000 [==>...........................] - ETA: 1:16 - loss: 7.6176 - accuracy: 0.5032
<br /> 2848/25000 [==>...........................] - ETA: 1:16 - loss: 7.6289 - accuracy: 0.5025
<br /> 2880/25000 [==>...........................] - ETA: 1:16 - loss: 7.6294 - accuracy: 0.5024
<br /> 2912/25000 [==>...........................] - ETA: 1:15 - loss: 7.6298 - accuracy: 0.5024
<br /> 2944/25000 [==>...........................] - ETA: 1:15 - loss: 7.6406 - accuracy: 0.5017
<br /> 2976/25000 [==>...........................] - ETA: 1:15 - loss: 7.6460 - accuracy: 0.5013
<br /> 3008/25000 [==>...........................] - ETA: 1:15 - loss: 7.6513 - accuracy: 0.5010
<br /> 3040/25000 [==>...........................] - ETA: 1:15 - loss: 7.6616 - accuracy: 0.5003
<br /> 3072/25000 [==>...........................] - ETA: 1:15 - loss: 7.6766 - accuracy: 0.4993
<br /> 3104/25000 [==>...........................] - ETA: 1:15 - loss: 7.6963 - accuracy: 0.4981
<br /> 3136/25000 [==>...........................] - ETA: 1:15 - loss: 7.6960 - accuracy: 0.4981
<br /> 3168/25000 [==>...........................] - ETA: 1:15 - loss: 7.7005 - accuracy: 0.4978
<br /> 3200/25000 [==>...........................] - ETA: 1:15 - loss: 7.7241 - accuracy: 0.4963
<br /> 3232/25000 [==>...........................] - ETA: 1:15 - loss: 7.7283 - accuracy: 0.4960
<br /> 3264/25000 [==>...........................] - ETA: 1:14 - loss: 7.7230 - accuracy: 0.4963
<br /> 3296/25000 [==>...........................] - ETA: 1:14 - loss: 7.7364 - accuracy: 0.4954
<br /> 3328/25000 [==>...........................] - ETA: 1:14 - loss: 7.7127 - accuracy: 0.4970
<br /> 3360/25000 [===>..........................] - ETA: 1:14 - loss: 7.7123 - accuracy: 0.4970
<br /> 3392/25000 [===>..........................] - ETA: 1:14 - loss: 7.7254 - accuracy: 0.4962
<br /> 3424/25000 [===>..........................] - ETA: 1:14 - loss: 7.7114 - accuracy: 0.4971
<br /> 3456/25000 [===>..........................] - ETA: 1:14 - loss: 7.7332 - accuracy: 0.4957
<br /> 3488/25000 [===>..........................] - ETA: 1:14 - loss: 7.7282 - accuracy: 0.4960
<br /> 3520/25000 [===>..........................] - ETA: 1:14 - loss: 7.7276 - accuracy: 0.4960
<br /> 3552/25000 [===>..........................] - ETA: 1:14 - loss: 7.7357 - accuracy: 0.4955
<br /> 3584/25000 [===>..........................] - ETA: 1:13 - loss: 7.7265 - accuracy: 0.4961
<br /> 3616/25000 [===>..........................] - ETA: 1:13 - loss: 7.7090 - accuracy: 0.4972
<br /> 3648/25000 [===>..........................] - ETA: 1:13 - loss: 7.7087 - accuracy: 0.4973
<br /> 3680/25000 [===>..........................] - ETA: 1:13 - loss: 7.7083 - accuracy: 0.4973
<br /> 3712/25000 [===>..........................] - ETA: 1:13 - loss: 7.7038 - accuracy: 0.4976
<br /> 3744/25000 [===>..........................] - ETA: 1:13 - loss: 7.6912 - accuracy: 0.4984
<br /> 3776/25000 [===>..........................] - ETA: 1:13 - loss: 7.6747 - accuracy: 0.4995
<br /> 3808/25000 [===>..........................] - ETA: 1:13 - loss: 7.6706 - accuracy: 0.4997
<br /> 3840/25000 [===>..........................] - ETA: 1:13 - loss: 7.6427 - accuracy: 0.5016
<br /> 3872/25000 [===>..........................] - ETA: 1:13 - loss: 7.6270 - accuracy: 0.5026
<br /> 3904/25000 [===>..........................] - ETA: 1:12 - loss: 7.6234 - accuracy: 0.5028
<br /> 3936/25000 [===>..........................] - ETA: 1:12 - loss: 7.6160 - accuracy: 0.5033
<br /> 3968/25000 [===>..........................] - ETA: 1:12 - loss: 7.6241 - accuracy: 0.5028
<br /> 4000/25000 [===>..........................] - ETA: 1:12 - loss: 7.6360 - accuracy: 0.5020
<br /> 4032/25000 [===>..........................] - ETA: 1:12 - loss: 7.6438 - accuracy: 0.5015
<br /> 4064/25000 [===>..........................] - ETA: 1:12 - loss: 7.6553 - accuracy: 0.5007
<br /> 4096/25000 [===>..........................] - ETA: 1:12 - loss: 7.6554 - accuracy: 0.5007
<br /> 4128/25000 [===>..........................] - ETA: 1:12 - loss: 7.6518 - accuracy: 0.5010
<br /> 4160/25000 [===>..........................] - ETA: 1:11 - loss: 7.6408 - accuracy: 0.5017
<br /> 4192/25000 [====>.........................] - ETA: 1:11 - loss: 7.6410 - accuracy: 0.5017
<br /> 4224/25000 [====>.........................] - ETA: 1:11 - loss: 7.6231 - accuracy: 0.5028
<br /> 4256/25000 [====>.........................] - ETA: 1:11 - loss: 7.6162 - accuracy: 0.5033
<br /> 4288/25000 [====>.........................] - ETA: 1:11 - loss: 7.6166 - accuracy: 0.5033
<br /> 4320/25000 [====>.........................] - ETA: 1:11 - loss: 7.6276 - accuracy: 0.5025
<br /> 4352/25000 [====>.........................] - ETA: 1:11 - loss: 7.6279 - accuracy: 0.5025
<br /> 4384/25000 [====>.........................] - ETA: 1:11 - loss: 7.6246 - accuracy: 0.5027
<br /> 4416/25000 [====>.........................] - ETA: 1:11 - loss: 7.6284 - accuracy: 0.5025
<br /> 4448/25000 [====>.........................] - ETA: 1:11 - loss: 7.6080 - accuracy: 0.5038
<br /> 4480/25000 [====>.........................] - ETA: 1:10 - loss: 7.6016 - accuracy: 0.5042
<br /> 4512/25000 [====>.........................] - ETA: 1:10 - loss: 7.6122 - accuracy: 0.5035
<br /> 4544/25000 [====>.........................] - ETA: 1:10 - loss: 7.6093 - accuracy: 0.5037
<br /> 4576/25000 [====>.........................] - ETA: 1:10 - loss: 7.6164 - accuracy: 0.5033
<br /> 4608/25000 [====>.........................] - ETA: 1:10 - loss: 7.6200 - accuracy: 0.5030
<br /> 4640/25000 [====>.........................] - ETA: 1:10 - loss: 7.6137 - accuracy: 0.5034
<br /> 4672/25000 [====>.........................] - ETA: 1:10 - loss: 7.6075 - accuracy: 0.5039
<br /> 4704/25000 [====>.........................] - ETA: 1:09 - loss: 7.5949 - accuracy: 0.5047
<br /> 4736/25000 [====>.........................] - ETA: 1:09 - loss: 7.6116 - accuracy: 0.5036
<br /> 4768/25000 [====>.........................] - ETA: 1:09 - loss: 7.6119 - accuracy: 0.5036
<br /> 4800/25000 [====>.........................] - ETA: 1:09 - loss: 7.6123 - accuracy: 0.5035
<br /> 4832/25000 [====>.........................] - ETA: 1:09 - loss: 7.6158 - accuracy: 0.5033
<br /> 4864/25000 [====>.........................] - ETA: 1:09 - loss: 7.6067 - accuracy: 0.5039
<br /> 4896/25000 [====>.........................] - ETA: 1:09 - loss: 7.6228 - accuracy: 0.5029
<br /> 4928/25000 [====>.........................] - ETA: 1:09 - loss: 7.6355 - accuracy: 0.5020
<br /> 4960/25000 [====>.........................] - ETA: 1:09 - loss: 7.6264 - accuracy: 0.5026
<br /> 4992/25000 [====>.........................] - ETA: 1:09 - loss: 7.6359 - accuracy: 0.5020
<br /> 5024/25000 [=====>........................] - ETA: 1:08 - loss: 7.6239 - accuracy: 0.5028
<br /> 5056/25000 [=====>........................] - ETA: 1:08 - loss: 7.6242 - accuracy: 0.5028
<br /> 5088/25000 [=====>........................] - ETA: 1:08 - loss: 7.6335 - accuracy: 0.5022
<br /> 5120/25000 [=====>........................] - ETA: 1:08 - loss: 7.6367 - accuracy: 0.5020
<br /> 5152/25000 [=====>........................] - ETA: 1:08 - loss: 7.6220 - accuracy: 0.5029
<br /> 5184/25000 [=====>........................] - ETA: 1:08 - loss: 7.6193 - accuracy: 0.5031
<br /> 5216/25000 [=====>........................] - ETA: 1:08 - loss: 7.6225 - accuracy: 0.5029
<br /> 5248/25000 [=====>........................] - ETA: 1:08 - loss: 7.6257 - accuracy: 0.5027
<br /> 5280/25000 [=====>........................] - ETA: 1:08 - loss: 7.6289 - accuracy: 0.5025
<br /> 5312/25000 [=====>........................] - ETA: 1:07 - loss: 7.6233 - accuracy: 0.5028
<br /> 5344/25000 [=====>........................] - ETA: 1:07 - loss: 7.6178 - accuracy: 0.5032
<br /> 5376/25000 [=====>........................] - ETA: 1:07 - loss: 7.6096 - accuracy: 0.5037
<br /> 5408/25000 [=====>........................] - ETA: 1:07 - loss: 7.6071 - accuracy: 0.5039
<br /> 5440/25000 [=====>........................] - ETA: 1:07 - loss: 7.6046 - accuracy: 0.5040
<br /> 5472/25000 [=====>........................] - ETA: 1:07 - loss: 7.6050 - accuracy: 0.5040
<br /> 5504/25000 [=====>........................] - ETA: 1:07 - loss: 7.6025 - accuracy: 0.5042
<br /> 5536/25000 [=====>........................] - ETA: 1:07 - loss: 7.6085 - accuracy: 0.5038
<br /> 5568/25000 [=====>........................] - ETA: 1:06 - loss: 7.6143 - accuracy: 0.5034
<br /> 5600/25000 [=====>........................] - ETA: 1:06 - loss: 7.6146 - accuracy: 0.5034
<br /> 5632/25000 [=====>........................] - ETA: 1:06 - loss: 7.6312 - accuracy: 0.5023
<br /> 5664/25000 [=====>........................] - ETA: 1:06 - loss: 7.6125 - accuracy: 0.5035
<br /> 5696/25000 [=====>........................] - ETA: 1:06 - loss: 7.6101 - accuracy: 0.5037
<br /> 5728/25000 [=====>........................] - ETA: 1:06 - loss: 7.6158 - accuracy: 0.5033
<br /> 5760/25000 [=====>........................] - ETA: 1:06 - loss: 7.6160 - accuracy: 0.5033
<br /> 5792/25000 [=====>........................] - ETA: 1:06 - loss: 7.6190 - accuracy: 0.5031
<br /> 5824/25000 [=====>........................] - ETA: 1:05 - loss: 7.6113 - accuracy: 0.5036
<br /> 5856/25000 [======>.......................] - ETA: 1:05 - loss: 7.6169 - accuracy: 0.5032
<br /> 5888/25000 [======>.......................] - ETA: 1:05 - loss: 7.6145 - accuracy: 0.5034
<br /> 5920/25000 [======>.......................] - ETA: 1:05 - loss: 7.6045 - accuracy: 0.5041
<br /> 5952/25000 [======>.......................] - ETA: 1:05 - loss: 7.5971 - accuracy: 0.5045
<br /> 5984/25000 [======>.......................] - ETA: 1:05 - loss: 7.5974 - accuracy: 0.5045
<br /> 6016/25000 [======>.......................] - ETA: 1:05 - loss: 7.6080 - accuracy: 0.5038
<br /> 6048/25000 [======>.......................] - ETA: 1:05 - loss: 7.6108 - accuracy: 0.5036
<br /> 6080/25000 [======>.......................] - ETA: 1:05 - loss: 7.6010 - accuracy: 0.5043
<br /> 6112/25000 [======>.......................] - ETA: 1:04 - loss: 7.6089 - accuracy: 0.5038
<br /> 6144/25000 [======>.......................] - ETA: 1:04 - loss: 7.6067 - accuracy: 0.5039
<br /> 6176/25000 [======>.......................] - ETA: 1:04 - loss: 7.6070 - accuracy: 0.5039
<br /> 6208/25000 [======>.......................] - ETA: 1:04 - loss: 7.6123 - accuracy: 0.5035
<br /> 6240/25000 [======>.......................] - ETA: 1:04 - loss: 7.6027 - accuracy: 0.5042
<br /> 6272/25000 [======>.......................] - ETA: 1:04 - loss: 7.5982 - accuracy: 0.5045
<br /> 6304/25000 [======>.......................] - ETA: 1:04 - loss: 7.5985 - accuracy: 0.5044
<br /> 6336/25000 [======>.......................] - ETA: 1:04 - loss: 7.6061 - accuracy: 0.5039
<br /> 6368/25000 [======>.......................] - ETA: 1:03 - loss: 7.6064 - accuracy: 0.5039
<br /> 6400/25000 [======>.......................] - ETA: 1:03 - loss: 7.6067 - accuracy: 0.5039
<br /> 6432/25000 [======>.......................] - ETA: 1:03 - loss: 7.6166 - accuracy: 0.5033
<br /> 6464/25000 [======>.......................] - ETA: 1:03 - loss: 7.6287 - accuracy: 0.5025
<br /> 6496/25000 [======>.......................] - ETA: 1:03 - loss: 7.6194 - accuracy: 0.5031
<br /> 6528/25000 [======>.......................] - ETA: 1:03 - loss: 7.6243 - accuracy: 0.5028
<br /> 6560/25000 [======>.......................] - ETA: 1:03 - loss: 7.6316 - accuracy: 0.5023
<br /> 6592/25000 [======>.......................] - ETA: 1:03 - loss: 7.6224 - accuracy: 0.5029
<br /> 6624/25000 [======>.......................] - ETA: 1:03 - loss: 7.6273 - accuracy: 0.5026
<br /> 6656/25000 [======>.......................] - ETA: 1:02 - loss: 7.6275 - accuracy: 0.5026
<br /> 6688/25000 [=======>......................] - ETA: 1:02 - loss: 7.6185 - accuracy: 0.5031
<br /> 6720/25000 [=======>......................] - ETA: 1:02 - loss: 7.6255 - accuracy: 0.5027
<br /> 6752/25000 [=======>......................] - ETA: 1:02 - loss: 7.6348 - accuracy: 0.5021
<br /> 6784/25000 [=======>......................] - ETA: 1:02 - loss: 7.6214 - accuracy: 0.5029
<br /> 6816/25000 [=======>......................] - ETA: 1:02 - loss: 7.6171 - accuracy: 0.5032
<br /> 6848/25000 [=======>......................] - ETA: 1:02 - loss: 7.6218 - accuracy: 0.5029
<br /> 6880/25000 [=======>......................] - ETA: 1:02 - loss: 7.6243 - accuracy: 0.5028
<br /> 6912/25000 [=======>......................] - ETA: 1:01 - loss: 7.6200 - accuracy: 0.5030
<br /> 6944/25000 [=======>......................] - ETA: 1:01 - loss: 7.6114 - accuracy: 0.5036
<br /> 6976/25000 [=======>......................] - ETA: 1:01 - loss: 7.6073 - accuracy: 0.5039
<br /> 7008/25000 [=======>......................] - ETA: 1:01 - loss: 7.5922 - accuracy: 0.5049
<br /> 7040/25000 [=======>......................] - ETA: 1:01 - loss: 7.5926 - accuracy: 0.5048
<br /> 7072/25000 [=======>......................] - ETA: 1:01 - loss: 7.5842 - accuracy: 0.5054
<br /> 7104/25000 [=======>......................] - ETA: 1:01 - loss: 7.5932 - accuracy: 0.5048
<br /> 7136/25000 [=======>......................] - ETA: 1:01 - loss: 7.5893 - accuracy: 0.5050
<br /> 7168/25000 [=======>......................] - ETA: 1:01 - loss: 7.5896 - accuracy: 0.5050
<br /> 7200/25000 [=======>......................] - ETA: 1:00 - loss: 7.5921 - accuracy: 0.5049
<br /> 7232/25000 [=======>......................] - ETA: 1:00 - loss: 7.5945 - accuracy: 0.5047
<br /> 7264/25000 [=======>......................] - ETA: 1:00 - loss: 7.5991 - accuracy: 0.5044
<br /> 7296/25000 [=======>......................] - ETA: 1:00 - loss: 7.6015 - accuracy: 0.5042
<br /> 7328/25000 [=======>......................] - ETA: 1:00 - loss: 7.5955 - accuracy: 0.5046
<br /> 7360/25000 [=======>......................] - ETA: 1:00 - loss: 7.5937 - accuracy: 0.5048
<br /> 7392/25000 [=======>......................] - ETA: 1:00 - loss: 7.5940 - accuracy: 0.5047
<br /> 7424/25000 [=======>......................] - ETA: 1:00 - loss: 7.5881 - accuracy: 0.5051
<br /> 7456/25000 [=======>......................] - ETA: 1:00 - loss: 7.5926 - accuracy: 0.5048
<br /> 7488/25000 [=======>......................] - ETA: 59s - loss: 7.6031 - accuracy: 0.5041 
<br /> 7520/25000 [========>.....................] - ETA: 59s - loss: 7.5993 - accuracy: 0.5044
<br /> 7552/25000 [========>.....................] - ETA: 59s - loss: 7.6016 - accuracy: 0.5042
<br /> 7584/25000 [========>.....................] - ETA: 59s - loss: 7.6060 - accuracy: 0.5040
<br /> 7616/25000 [========>.....................] - ETA: 59s - loss: 7.6002 - accuracy: 0.5043
<br /> 7648/25000 [========>.....................] - ETA: 59s - loss: 7.6005 - accuracy: 0.5043
<br /> 7680/25000 [========>.....................] - ETA: 59s - loss: 7.6047 - accuracy: 0.5040
<br /> 7712/25000 [========>.....................] - ETA: 59s - loss: 7.5931 - accuracy: 0.5048
<br /> 7744/25000 [========>.....................] - ETA: 59s - loss: 7.5953 - accuracy: 0.5046
<br /> 7776/25000 [========>.....................] - ETA: 58s - loss: 7.5996 - accuracy: 0.5044
<br /> 7808/25000 [========>.....................] - ETA: 58s - loss: 7.5959 - accuracy: 0.5046
<br /> 7840/25000 [========>.....................] - ETA: 58s - loss: 7.5923 - accuracy: 0.5048
<br /> 7872/25000 [========>.....................] - ETA: 58s - loss: 7.5965 - accuracy: 0.5046
<br /> 7904/25000 [========>.....................] - ETA: 58s - loss: 7.5968 - accuracy: 0.5046
<br /> 7936/25000 [========>.....................] - ETA: 58s - loss: 7.6067 - accuracy: 0.5039
<br /> 7968/25000 [========>.....................] - ETA: 58s - loss: 7.6089 - accuracy: 0.5038
<br /> 8000/25000 [========>.....................] - ETA: 58s - loss: 7.6072 - accuracy: 0.5039
<br /> 8032/25000 [========>.....................] - ETA: 58s - loss: 7.6055 - accuracy: 0.5040
<br /> 8064/25000 [========>.....................] - ETA: 58s - loss: 7.6153 - accuracy: 0.5033
<br /> 8096/25000 [========>.....................] - ETA: 57s - loss: 7.6136 - accuracy: 0.5035
<br /> 8128/25000 [========>.....................] - ETA: 57s - loss: 7.6213 - accuracy: 0.5030
<br /> 8160/25000 [========>.....................] - ETA: 57s - loss: 7.6234 - accuracy: 0.5028
<br /> 8192/25000 [========>.....................] - ETA: 57s - loss: 7.6311 - accuracy: 0.5023
<br /> 8224/25000 [========>.....................] - ETA: 57s - loss: 7.6275 - accuracy: 0.5026
<br /> 8256/25000 [========>.....................] - ETA: 57s - loss: 7.6369 - accuracy: 0.5019
<br /> 8288/25000 [========>.....................] - ETA: 57s - loss: 7.6296 - accuracy: 0.5024
<br /> 8320/25000 [========>.....................] - ETA: 57s - loss: 7.6353 - accuracy: 0.5020
<br /> 8352/25000 [=========>....................] - ETA: 57s - loss: 7.6391 - accuracy: 0.5018
<br /> 8384/25000 [=========>....................] - ETA: 56s - loss: 7.6392 - accuracy: 0.5018
<br /> 8416/25000 [=========>....................] - ETA: 56s - loss: 7.6411 - accuracy: 0.5017
<br /> 8448/25000 [=========>....................] - ETA: 56s - loss: 7.6376 - accuracy: 0.5019
<br /> 8480/25000 [=========>....................] - ETA: 56s - loss: 7.6341 - accuracy: 0.5021
<br /> 8512/25000 [=========>....................] - ETA: 56s - loss: 7.6396 - accuracy: 0.5018
<br /> 8544/25000 [=========>....................] - ETA: 56s - loss: 7.6415 - accuracy: 0.5016
<br /> 8576/25000 [=========>....................] - ETA: 56s - loss: 7.6398 - accuracy: 0.5017
<br /> 8608/25000 [=========>....................] - ETA: 56s - loss: 7.6417 - accuracy: 0.5016
<br /> 8640/25000 [=========>....................] - ETA: 55s - loss: 7.6382 - accuracy: 0.5019
<br /> 8672/25000 [=========>....................] - ETA: 55s - loss: 7.6330 - accuracy: 0.5022
<br /> 8704/25000 [=========>....................] - ETA: 55s - loss: 7.6296 - accuracy: 0.5024
<br /> 8736/25000 [=========>....................] - ETA: 55s - loss: 7.6175 - accuracy: 0.5032
<br /> 8768/25000 [=========>....................] - ETA: 55s - loss: 7.6281 - accuracy: 0.5025
<br /> 8800/25000 [=========>....................] - ETA: 55s - loss: 7.6248 - accuracy: 0.5027
<br /> 8832/25000 [=========>....................] - ETA: 55s - loss: 7.6284 - accuracy: 0.5025
<br /> 8864/25000 [=========>....................] - ETA: 55s - loss: 7.6320 - accuracy: 0.5023
<br /> 8896/25000 [=========>....................] - ETA: 55s - loss: 7.6373 - accuracy: 0.5019
<br /> 8928/25000 [=========>....................] - ETA: 54s - loss: 7.6391 - accuracy: 0.5018
<br /> 8960/25000 [=========>....................] - ETA: 54s - loss: 7.6444 - accuracy: 0.5015
<br /> 8992/25000 [=========>....................] - ETA: 54s - loss: 7.6427 - accuracy: 0.5016
<br /> 9024/25000 [=========>....................] - ETA: 54s - loss: 7.6445 - accuracy: 0.5014
<br /> 9056/25000 [=========>....................] - ETA: 54s - loss: 7.6463 - accuracy: 0.5013
<br /> 9088/25000 [=========>....................] - ETA: 54s - loss: 7.6565 - accuracy: 0.5007
<br /> 9120/25000 [=========>....................] - ETA: 54s - loss: 7.6549 - accuracy: 0.5008
<br /> 9152/25000 [=========>....................] - ETA: 54s - loss: 7.6549 - accuracy: 0.5008
<br /> 9184/25000 [==========>...................] - ETA: 53s - loss: 7.6516 - accuracy: 0.5010
<br /> 9216/25000 [==========>...................] - ETA: 53s - loss: 7.6566 - accuracy: 0.5007
<br /> 9248/25000 [==========>...................] - ETA: 53s - loss: 7.6633 - accuracy: 0.5002
<br /> 9280/25000 [==========>...................] - ETA: 53s - loss: 7.6650 - accuracy: 0.5001
<br /> 9312/25000 [==========>...................] - ETA: 53s - loss: 7.6567 - accuracy: 0.5006
<br /> 9344/25000 [==========>...................] - ETA: 53s - loss: 7.6584 - accuracy: 0.5005
<br /> 9376/25000 [==========>...................] - ETA: 53s - loss: 7.6568 - accuracy: 0.5006
<br /> 9408/25000 [==========>...................] - ETA: 53s - loss: 7.6585 - accuracy: 0.5005
<br /> 9440/25000 [==========>...................] - ETA: 53s - loss: 7.6536 - accuracy: 0.5008
<br /> 9472/25000 [==========>...................] - ETA: 52s - loss: 7.6537 - accuracy: 0.5008
<br /> 9504/25000 [==========>...................] - ETA: 52s - loss: 7.6473 - accuracy: 0.5013
<br /> 9536/25000 [==========>...................] - ETA: 52s - loss: 7.6425 - accuracy: 0.5016
<br /> 9568/25000 [==========>...................] - ETA: 52s - loss: 7.6474 - accuracy: 0.5013
<br /> 9600/25000 [==========>...................] - ETA: 52s - loss: 7.6491 - accuracy: 0.5011
<br /> 9632/25000 [==========>...................] - ETA: 52s - loss: 7.6539 - accuracy: 0.5008
<br /> 9664/25000 [==========>...................] - ETA: 52s - loss: 7.6539 - accuracy: 0.5008
<br /> 9696/25000 [==========>...................] - ETA: 52s - loss: 7.6524 - accuracy: 0.5009
<br /> 9728/25000 [==========>...................] - ETA: 52s - loss: 7.6556 - accuracy: 0.5007
<br /> 9760/25000 [==========>...................] - ETA: 52s - loss: 7.6556 - accuracy: 0.5007
<br /> 9792/25000 [==========>...................] - ETA: 51s - loss: 7.6604 - accuracy: 0.5004
<br /> 9824/25000 [==========>...................] - ETA: 51s - loss: 7.6619 - accuracy: 0.5003
<br /> 9856/25000 [==========>...................] - ETA: 51s - loss: 7.6651 - accuracy: 0.5001
<br /> 9888/25000 [==========>...................] - ETA: 51s - loss: 7.6713 - accuracy: 0.4997
<br /> 9920/25000 [==========>...................] - ETA: 51s - loss: 7.6759 - accuracy: 0.4994
<br /> 9952/25000 [==========>...................] - ETA: 51s - loss: 7.6774 - accuracy: 0.4993
<br /> 9984/25000 [==========>...................] - ETA: 51s - loss: 7.6758 - accuracy: 0.4994
<br />10016/25000 [===========>..................] - ETA: 51s - loss: 7.6789 - accuracy: 0.4992
<br />10048/25000 [===========>..................] - ETA: 51s - loss: 7.6773 - accuracy: 0.4993
<br />10080/25000 [===========>..................] - ETA: 50s - loss: 7.6818 - accuracy: 0.4990
<br />10112/25000 [===========>..................] - ETA: 50s - loss: 7.6833 - accuracy: 0.4989
<br />10144/25000 [===========>..................] - ETA: 50s - loss: 7.6848 - accuracy: 0.4988
<br />10176/25000 [===========>..................] - ETA: 50s - loss: 7.6787 - accuracy: 0.4992
<br />10208/25000 [===========>..................] - ETA: 50s - loss: 7.6696 - accuracy: 0.4998
<br />10240/25000 [===========>..................] - ETA: 50s - loss: 7.6666 - accuracy: 0.5000
<br />10272/25000 [===========>..................] - ETA: 50s - loss: 7.6666 - accuracy: 0.5000
<br />10304/25000 [===========>..................] - ETA: 50s - loss: 7.6755 - accuracy: 0.4994
<br />10336/25000 [===========>..................] - ETA: 50s - loss: 7.6785 - accuracy: 0.4992
<br />10368/25000 [===========>..................] - ETA: 49s - loss: 7.6814 - accuracy: 0.4990
<br />10400/25000 [===========>..................] - ETA: 49s - loss: 7.6843 - accuracy: 0.4988
<br />10432/25000 [===========>..................] - ETA: 49s - loss: 7.6828 - accuracy: 0.4989
<br />10464/25000 [===========>..................] - ETA: 49s - loss: 7.6827 - accuracy: 0.4989
<br />10496/25000 [===========>..................] - ETA: 49s - loss: 7.6783 - accuracy: 0.4992
<br />10528/25000 [===========>..................] - ETA: 49s - loss: 7.6724 - accuracy: 0.4996
<br />10560/25000 [===========>..................] - ETA: 49s - loss: 7.6681 - accuracy: 0.4999
<br />10592/25000 [===========>..................] - ETA: 49s - loss: 7.6681 - accuracy: 0.4999
<br />10624/25000 [===========>..................] - ETA: 49s - loss: 7.6681 - accuracy: 0.4999
<br />10656/25000 [===========>..................] - ETA: 48s - loss: 7.6767 - accuracy: 0.4993
<br />10688/25000 [===========>..................] - ETA: 48s - loss: 7.6752 - accuracy: 0.4994
<br />10720/25000 [===========>..................] - ETA: 48s - loss: 7.6695 - accuracy: 0.4998
<br />10752/25000 [===========>..................] - ETA: 48s - loss: 7.6680 - accuracy: 0.4999
<br />10784/25000 [===========>..................] - ETA: 48s - loss: 7.6695 - accuracy: 0.4998
<br />10816/25000 [===========>..................] - ETA: 48s - loss: 7.6765 - accuracy: 0.4994
<br />10848/25000 [============>.................] - ETA: 48s - loss: 7.6779 - accuracy: 0.4993
<br />10880/25000 [============>.................] - ETA: 48s - loss: 7.6793 - accuracy: 0.4992
<br />10912/25000 [============>.................] - ETA: 48s - loss: 7.6793 - accuracy: 0.4992
<br />10944/25000 [============>.................] - ETA: 48s - loss: 7.6862 - accuracy: 0.4987
<br />10976/25000 [============>.................] - ETA: 47s - loss: 7.6848 - accuracy: 0.4988
<br />11008/25000 [============>.................] - ETA: 47s - loss: 7.6819 - accuracy: 0.4990
<br />11040/25000 [============>.................] - ETA: 47s - loss: 7.6777 - accuracy: 0.4993
<br />11072/25000 [============>.................] - ETA: 47s - loss: 7.6763 - accuracy: 0.4994
<br />11104/25000 [============>.................] - ETA: 47s - loss: 7.6735 - accuracy: 0.4995
<br />11136/25000 [============>.................] - ETA: 47s - loss: 7.6694 - accuracy: 0.4998
<br />11168/25000 [============>.................] - ETA: 47s - loss: 7.6721 - accuracy: 0.4996
<br />11200/25000 [============>.................] - ETA: 47s - loss: 7.6762 - accuracy: 0.4994
<br />11232/25000 [============>.................] - ETA: 47s - loss: 7.6789 - accuracy: 0.4992
<br />11264/25000 [============>.................] - ETA: 46s - loss: 7.6761 - accuracy: 0.4994
<br />11296/25000 [============>.................] - ETA: 46s - loss: 7.6775 - accuracy: 0.4993
<br />11328/25000 [============>.................] - ETA: 46s - loss: 7.6747 - accuracy: 0.4995
<br />11360/25000 [============>.................] - ETA: 46s - loss: 7.6747 - accuracy: 0.4995
<br />11392/25000 [============>.................] - ETA: 46s - loss: 7.6680 - accuracy: 0.4999
<br />11424/25000 [============>.................] - ETA: 46s - loss: 7.6639 - accuracy: 0.5002
<br />11456/25000 [============>.................] - ETA: 46s - loss: 7.6639 - accuracy: 0.5002
<br />11488/25000 [============>.................] - ETA: 46s - loss: 7.6599 - accuracy: 0.5004
<br />11520/25000 [============>.................] - ETA: 46s - loss: 7.6626 - accuracy: 0.5003
<br />11552/25000 [============>.................] - ETA: 46s - loss: 7.6560 - accuracy: 0.5007
<br />11584/25000 [============>.................] - ETA: 45s - loss: 7.6547 - accuracy: 0.5008
<br />11616/25000 [============>.................] - ETA: 45s - loss: 7.6561 - accuracy: 0.5007
<br />11648/25000 [============>.................] - ETA: 45s - loss: 7.6561 - accuracy: 0.5007
<br />11680/25000 [=============>................] - ETA: 45s - loss: 7.6548 - accuracy: 0.5008
<br />11712/25000 [=============>................] - ETA: 45s - loss: 7.6548 - accuracy: 0.5008
<br />11744/25000 [=============>................] - ETA: 45s - loss: 7.6523 - accuracy: 0.5009
<br />11776/25000 [=============>................] - ETA: 45s - loss: 7.6523 - accuracy: 0.5009
<br />11808/25000 [=============>................] - ETA: 45s - loss: 7.6510 - accuracy: 0.5010
<br />11840/25000 [=============>................] - ETA: 44s - loss: 7.6472 - accuracy: 0.5013
<br />11872/25000 [=============>................] - ETA: 44s - loss: 7.6485 - accuracy: 0.5012
<br />11904/25000 [=============>................] - ETA: 44s - loss: 7.6537 - accuracy: 0.5008
<br />11936/25000 [=============>................] - ETA: 44s - loss: 7.6563 - accuracy: 0.5007
<br />11968/25000 [=============>................] - ETA: 44s - loss: 7.6564 - accuracy: 0.5007
<br />12000/25000 [=============>................] - ETA: 44s - loss: 7.6564 - accuracy: 0.5007
<br />12032/25000 [=============>................] - ETA: 44s - loss: 7.6590 - accuracy: 0.5005
<br />12064/25000 [=============>................] - ETA: 44s - loss: 7.6641 - accuracy: 0.5002
<br />12096/25000 [=============>................] - ETA: 44s - loss: 7.6717 - accuracy: 0.4997
<br />12128/25000 [=============>................] - ETA: 44s - loss: 7.6704 - accuracy: 0.4998
<br />12160/25000 [=============>................] - ETA: 43s - loss: 7.6679 - accuracy: 0.4999
<br />12192/25000 [=============>................] - ETA: 43s - loss: 7.6767 - accuracy: 0.4993
<br />12224/25000 [=============>................] - ETA: 43s - loss: 7.6779 - accuracy: 0.4993
<br />12256/25000 [=============>................] - ETA: 43s - loss: 7.6766 - accuracy: 0.4993
<br />12288/25000 [=============>................] - ETA: 43s - loss: 7.6766 - accuracy: 0.4993
<br />12320/25000 [=============>................] - ETA: 43s - loss: 7.6716 - accuracy: 0.4997
<br />12352/25000 [=============>................] - ETA: 43s - loss: 7.6691 - accuracy: 0.4998
<br />12384/25000 [=============>................] - ETA: 43s - loss: 7.6654 - accuracy: 0.5001
<br />12416/25000 [=============>................] - ETA: 43s - loss: 7.6654 - accuracy: 0.5001
<br />12448/25000 [=============>................] - ETA: 42s - loss: 7.6715 - accuracy: 0.4997
<br />12480/25000 [=============>................] - ETA: 42s - loss: 7.6654 - accuracy: 0.5001
<br />12512/25000 [==============>...............] - ETA: 42s - loss: 7.6691 - accuracy: 0.4998
<br />12544/25000 [==============>...............] - ETA: 42s - loss: 7.6654 - accuracy: 0.5001
<br />12576/25000 [==============>...............] - ETA: 42s - loss: 7.6691 - accuracy: 0.4998
<br />12608/25000 [==============>...............] - ETA: 42s - loss: 7.6703 - accuracy: 0.4998
<br />12640/25000 [==============>...............] - ETA: 42s - loss: 7.6751 - accuracy: 0.4994
<br />12672/25000 [==============>...............] - ETA: 42s - loss: 7.6739 - accuracy: 0.4995
<br />12704/25000 [==============>...............] - ETA: 42s - loss: 7.6787 - accuracy: 0.4992
<br />12736/25000 [==============>...............] - ETA: 41s - loss: 7.6702 - accuracy: 0.4998
<br />12768/25000 [==============>...............] - ETA: 41s - loss: 7.6714 - accuracy: 0.4997
<br />12800/25000 [==============>...............] - ETA: 41s - loss: 7.6714 - accuracy: 0.4997
<br />12832/25000 [==============>...............] - ETA: 41s - loss: 7.6726 - accuracy: 0.4996
<br />12864/25000 [==============>...............] - ETA: 41s - loss: 7.6726 - accuracy: 0.4996
<br />12896/25000 [==============>...............] - ETA: 41s - loss: 7.6714 - accuracy: 0.4997
<br />12928/25000 [==============>...............] - ETA: 41s - loss: 7.6737 - accuracy: 0.4995
<br />12960/25000 [==============>...............] - ETA: 41s - loss: 7.6714 - accuracy: 0.4997
<br />12992/25000 [==============>...............] - ETA: 41s - loss: 7.6737 - accuracy: 0.4995
<br />13024/25000 [==============>...............] - ETA: 40s - loss: 7.6725 - accuracy: 0.4996
<br />13056/25000 [==============>...............] - ETA: 40s - loss: 7.6748 - accuracy: 0.4995
<br />13088/25000 [==============>...............] - ETA: 40s - loss: 7.6701 - accuracy: 0.4998
<br />13120/25000 [==============>...............] - ETA: 40s - loss: 7.6678 - accuracy: 0.4999
<br />13152/25000 [==============>...............] - ETA: 40s - loss: 7.6655 - accuracy: 0.5001
<br />13184/25000 [==============>...............] - ETA: 40s - loss: 7.6655 - accuracy: 0.5001
<br />13216/25000 [==============>...............] - ETA: 40s - loss: 7.6689 - accuracy: 0.4998
<br />13248/25000 [==============>...............] - ETA: 40s - loss: 7.6724 - accuracy: 0.4996
<br />13280/25000 [==============>...............] - ETA: 40s - loss: 7.6655 - accuracy: 0.5001
<br />13312/25000 [==============>...............] - ETA: 39s - loss: 7.6632 - accuracy: 0.5002
<br />13344/25000 [===============>..............] - ETA: 39s - loss: 7.6678 - accuracy: 0.4999
<br />13376/25000 [===============>..............] - ETA: 39s - loss: 7.6678 - accuracy: 0.4999
<br />13408/25000 [===============>..............] - ETA: 39s - loss: 7.6632 - accuracy: 0.5002
<br />13440/25000 [===============>..............] - ETA: 39s - loss: 7.6598 - accuracy: 0.5004
<br />13472/25000 [===============>..............] - ETA: 39s - loss: 7.6655 - accuracy: 0.5001
<br />13504/25000 [===============>..............] - ETA: 39s - loss: 7.6689 - accuracy: 0.4999
<br />13536/25000 [===============>..............] - ETA: 39s - loss: 7.6655 - accuracy: 0.5001
<br />13568/25000 [===============>..............] - ETA: 38s - loss: 7.6655 - accuracy: 0.5001
<br />13600/25000 [===============>..............] - ETA: 38s - loss: 7.6700 - accuracy: 0.4998
<br />13632/25000 [===============>..............] - ETA: 38s - loss: 7.6700 - accuracy: 0.4998
<br />13664/25000 [===============>..............] - ETA: 38s - loss: 7.6734 - accuracy: 0.4996
<br />13696/25000 [===============>..............] - ETA: 38s - loss: 7.6745 - accuracy: 0.4995
<br />13728/25000 [===============>..............] - ETA: 38s - loss: 7.6789 - accuracy: 0.4992
<br />13760/25000 [===============>..............] - ETA: 38s - loss: 7.6811 - accuracy: 0.4991
<br />13792/25000 [===============>..............] - ETA: 38s - loss: 7.6766 - accuracy: 0.4993
<br />13824/25000 [===============>..............] - ETA: 38s - loss: 7.6799 - accuracy: 0.4991
<br />13856/25000 [===============>..............] - ETA: 38s - loss: 7.6755 - accuracy: 0.4994
<br />13888/25000 [===============>..............] - ETA: 37s - loss: 7.6766 - accuracy: 0.4994
<br />13920/25000 [===============>..............] - ETA: 37s - loss: 7.6776 - accuracy: 0.4993
<br />13952/25000 [===============>..............] - ETA: 37s - loss: 7.6765 - accuracy: 0.4994
<br />13984/25000 [===============>..............] - ETA: 37s - loss: 7.6732 - accuracy: 0.4996
<br />14016/25000 [===============>..............] - ETA: 37s - loss: 7.6699 - accuracy: 0.4998
<br />14048/25000 [===============>..............] - ETA: 37s - loss: 7.6633 - accuracy: 0.5002
<br />14080/25000 [===============>..............] - ETA: 37s - loss: 7.6655 - accuracy: 0.5001
<br />14112/25000 [===============>..............] - ETA: 37s - loss: 7.6677 - accuracy: 0.4999
<br />14144/25000 [===============>..............] - ETA: 36s - loss: 7.6666 - accuracy: 0.5000
<br />14176/25000 [================>.............] - ETA: 36s - loss: 7.6645 - accuracy: 0.5001
<br />14208/25000 [================>.............] - ETA: 36s - loss: 7.6591 - accuracy: 0.5005
<br />14240/25000 [================>.............] - ETA: 36s - loss: 7.6655 - accuracy: 0.5001
<br />14272/25000 [================>.............] - ETA: 36s - loss: 7.6677 - accuracy: 0.4999
<br />14304/25000 [================>.............] - ETA: 36s - loss: 7.6688 - accuracy: 0.4999
<br />14336/25000 [================>.............] - ETA: 36s - loss: 7.6698 - accuracy: 0.4998
<br />14368/25000 [================>.............] - ETA: 36s - loss: 7.6698 - accuracy: 0.4998
<br />14400/25000 [================>.............] - ETA: 36s - loss: 7.6709 - accuracy: 0.4997
<br />14432/25000 [================>.............] - ETA: 36s - loss: 7.6677 - accuracy: 0.4999
<br />14464/25000 [================>.............] - ETA: 35s - loss: 7.6719 - accuracy: 0.4997
<br />14496/25000 [================>.............] - ETA: 35s - loss: 7.6719 - accuracy: 0.4997
<br />14528/25000 [================>.............] - ETA: 35s - loss: 7.6740 - accuracy: 0.4995
<br />14560/25000 [================>.............] - ETA: 35s - loss: 7.6761 - accuracy: 0.4994
<br />14592/25000 [================>.............] - ETA: 35s - loss: 7.6834 - accuracy: 0.4989
<br />14624/25000 [================>.............] - ETA: 35s - loss: 7.6886 - accuracy: 0.4986
<br />14656/25000 [================>.............] - ETA: 35s - loss: 7.6875 - accuracy: 0.4986
<br />14688/25000 [================>.............] - ETA: 35s - loss: 7.6896 - accuracy: 0.4985
<br />14720/25000 [================>.............] - ETA: 35s - loss: 7.6875 - accuracy: 0.4986
<br />14752/25000 [================>.............] - ETA: 34s - loss: 7.6864 - accuracy: 0.4987
<br />14784/25000 [================>.............] - ETA: 34s - loss: 7.6884 - accuracy: 0.4986
<br />14816/25000 [================>.............] - ETA: 34s - loss: 7.6863 - accuracy: 0.4987
<br />14848/25000 [================>.............] - ETA: 34s - loss: 7.6831 - accuracy: 0.4989
<br />14880/25000 [================>.............] - ETA: 34s - loss: 7.6831 - accuracy: 0.4989
<br />14912/25000 [================>.............] - ETA: 34s - loss: 7.6820 - accuracy: 0.4990
<br />14944/25000 [================>.............] - ETA: 34s - loss: 7.6820 - accuracy: 0.4990
<br />14976/25000 [================>.............] - ETA: 34s - loss: 7.6861 - accuracy: 0.4987
<br />15008/25000 [=================>............] - ETA: 34s - loss: 7.6809 - accuracy: 0.4991
<br />15040/25000 [=================>............] - ETA: 33s - loss: 7.6850 - accuracy: 0.4988
<br />15072/25000 [=================>............] - ETA: 33s - loss: 7.6839 - accuracy: 0.4989
<br />15104/25000 [=================>............] - ETA: 33s - loss: 7.6859 - accuracy: 0.4987
<br />15136/25000 [=================>............] - ETA: 33s - loss: 7.6838 - accuracy: 0.4989
<br />15168/25000 [=================>............] - ETA: 33s - loss: 7.6818 - accuracy: 0.4990
<br />15200/25000 [=================>............] - ETA: 33s - loss: 7.6868 - accuracy: 0.4987
<br />15232/25000 [=================>............] - ETA: 33s - loss: 7.6878 - accuracy: 0.4986
<br />15264/25000 [=================>............] - ETA: 33s - loss: 7.6817 - accuracy: 0.4990
<br />15296/25000 [=================>............] - ETA: 33s - loss: 7.6766 - accuracy: 0.4993
<br />15328/25000 [=================>............] - ETA: 32s - loss: 7.6776 - accuracy: 0.4993
<br />15360/25000 [=================>............] - ETA: 32s - loss: 7.6766 - accuracy: 0.4993
<br />15392/25000 [=================>............] - ETA: 32s - loss: 7.6746 - accuracy: 0.4995
<br />15424/25000 [=================>............] - ETA: 32s - loss: 7.6716 - accuracy: 0.4997
<br />15456/25000 [=================>............] - ETA: 32s - loss: 7.6746 - accuracy: 0.4995
<br />15488/25000 [=================>............] - ETA: 32s - loss: 7.6775 - accuracy: 0.4993
<br />15520/25000 [=================>............] - ETA: 32s - loss: 7.6775 - accuracy: 0.4993
<br />15552/25000 [=================>............] - ETA: 32s - loss: 7.6775 - accuracy: 0.4993
<br />15584/25000 [=================>............] - ETA: 32s - loss: 7.6804 - accuracy: 0.4991
<br />15616/25000 [=================>............] - ETA: 32s - loss: 7.6823 - accuracy: 0.4990
<br />15648/25000 [=================>............] - ETA: 31s - loss: 7.6843 - accuracy: 0.4988
<br />15680/25000 [=================>............] - ETA: 31s - loss: 7.6832 - accuracy: 0.4989
<br />15712/25000 [=================>............] - ETA: 31s - loss: 7.6842 - accuracy: 0.4989
<br />15744/25000 [=================>............] - ETA: 31s - loss: 7.6861 - accuracy: 0.4987
<br />15776/25000 [=================>............] - ETA: 31s - loss: 7.6880 - accuracy: 0.4986
<br />15808/25000 [=================>............] - ETA: 31s - loss: 7.6870 - accuracy: 0.4987
<br />15840/25000 [==================>...........] - ETA: 31s - loss: 7.6869 - accuracy: 0.4987
<br />15872/25000 [==================>...........] - ETA: 31s - loss: 7.6859 - accuracy: 0.4987
<br />15904/25000 [==================>...........] - ETA: 30s - loss: 7.6840 - accuracy: 0.4989
<br />15936/25000 [==================>...........] - ETA: 30s - loss: 7.6801 - accuracy: 0.4991
<br />15968/25000 [==================>...........] - ETA: 30s - loss: 7.6791 - accuracy: 0.4992
<br />16000/25000 [==================>...........] - ETA: 30s - loss: 7.6772 - accuracy: 0.4993
<br />16032/25000 [==================>...........] - ETA: 30s - loss: 7.6800 - accuracy: 0.4991
<br />16064/25000 [==================>...........] - ETA: 30s - loss: 7.6809 - accuracy: 0.4991
<br />16096/25000 [==================>...........] - ETA: 30s - loss: 7.6771 - accuracy: 0.4993
<br />16128/25000 [==================>...........] - ETA: 30s - loss: 7.6771 - accuracy: 0.4993
<br />16160/25000 [==================>...........] - ETA: 30s - loss: 7.6761 - accuracy: 0.4994
<br />16192/25000 [==================>...........] - ETA: 30s - loss: 7.6799 - accuracy: 0.4991
<br />16224/25000 [==================>...........] - ETA: 29s - loss: 7.6770 - accuracy: 0.4993
<br />16256/25000 [==================>...........] - ETA: 29s - loss: 7.6779 - accuracy: 0.4993
<br />16288/25000 [==================>...........] - ETA: 29s - loss: 7.6742 - accuracy: 0.4995
<br />16320/25000 [==================>...........] - ETA: 29s - loss: 7.6770 - accuracy: 0.4993
<br />16352/25000 [==================>...........] - ETA: 29s - loss: 7.6741 - accuracy: 0.4995
<br />16384/25000 [==================>...........] - ETA: 29s - loss: 7.6750 - accuracy: 0.4995
<br />16416/25000 [==================>...........] - ETA: 29s - loss: 7.6769 - accuracy: 0.4993
<br />16448/25000 [==================>...........] - ETA: 29s - loss: 7.6787 - accuracy: 0.4992
<br />16480/25000 [==================>...........] - ETA: 29s - loss: 7.6769 - accuracy: 0.4993
<br />16512/25000 [==================>...........] - ETA: 28s - loss: 7.6796 - accuracy: 0.4992
<br />16544/25000 [==================>...........] - ETA: 28s - loss: 7.6796 - accuracy: 0.4992
<br />16576/25000 [==================>...........] - ETA: 28s - loss: 7.6759 - accuracy: 0.4994
<br />16608/25000 [==================>...........] - ETA: 28s - loss: 7.6740 - accuracy: 0.4995
<br />16640/25000 [==================>...........] - ETA: 28s - loss: 7.6685 - accuracy: 0.4999
<br />16672/25000 [===================>..........] - ETA: 28s - loss: 7.6685 - accuracy: 0.4999
<br />16704/25000 [===================>..........] - ETA: 28s - loss: 7.6666 - accuracy: 0.5000
<br />16736/25000 [===================>..........] - ETA: 28s - loss: 7.6611 - accuracy: 0.5004
<br />16768/25000 [===================>..........] - ETA: 28s - loss: 7.6630 - accuracy: 0.5002
<br />16800/25000 [===================>..........] - ETA: 27s - loss: 7.6630 - accuracy: 0.5002
<br />16832/25000 [===================>..........] - ETA: 27s - loss: 7.6621 - accuracy: 0.5003
<br />16864/25000 [===================>..........] - ETA: 27s - loss: 7.6639 - accuracy: 0.5002
<br />16896/25000 [===================>..........] - ETA: 27s - loss: 7.6648 - accuracy: 0.5001
<br />16928/25000 [===================>..........] - ETA: 27s - loss: 7.6657 - accuracy: 0.5001
<br />16960/25000 [===================>..........] - ETA: 27s - loss: 7.6675 - accuracy: 0.4999
<br />16992/25000 [===================>..........] - ETA: 27s - loss: 7.6648 - accuracy: 0.5001
<br />17024/25000 [===================>..........] - ETA: 27s - loss: 7.6612 - accuracy: 0.5004
<br />17056/25000 [===================>..........] - ETA: 27s - loss: 7.6612 - accuracy: 0.5004
<br />17088/25000 [===================>..........] - ETA: 26s - loss: 7.6657 - accuracy: 0.5001
<br />17120/25000 [===================>..........] - ETA: 26s - loss: 7.6675 - accuracy: 0.4999
<br />17152/25000 [===================>..........] - ETA: 26s - loss: 7.6684 - accuracy: 0.4999
<br />17184/25000 [===================>..........] - ETA: 26s - loss: 7.6702 - accuracy: 0.4998
<br />17216/25000 [===================>..........] - ETA: 26s - loss: 7.6755 - accuracy: 0.4994
<br />17248/25000 [===================>..........] - ETA: 26s - loss: 7.6764 - accuracy: 0.4994
<br />17280/25000 [===================>..........] - ETA: 26s - loss: 7.6764 - accuracy: 0.4994
<br />17312/25000 [===================>..........] - ETA: 26s - loss: 7.6764 - accuracy: 0.4994
<br />17344/25000 [===================>..........] - ETA: 26s - loss: 7.6755 - accuracy: 0.4994
<br />17376/25000 [===================>..........] - ETA: 25s - loss: 7.6772 - accuracy: 0.4993
<br />17408/25000 [===================>..........] - ETA: 25s - loss: 7.6807 - accuracy: 0.4991
<br />17440/25000 [===================>..........] - ETA: 25s - loss: 7.6824 - accuracy: 0.4990
<br />17472/25000 [===================>..........] - ETA: 25s - loss: 7.6798 - accuracy: 0.4991
<br />17504/25000 [====================>.........] - ETA: 25s - loss: 7.6789 - accuracy: 0.4992
<br />17536/25000 [====================>.........] - ETA: 25s - loss: 7.6815 - accuracy: 0.4990
<br />17568/25000 [====================>.........] - ETA: 25s - loss: 7.6788 - accuracy: 0.4992
<br />17600/25000 [====================>.........] - ETA: 25s - loss: 7.6779 - accuracy: 0.4993
<br />17632/25000 [====================>.........] - ETA: 25s - loss: 7.6771 - accuracy: 0.4993
<br />17664/25000 [====================>.........] - ETA: 24s - loss: 7.6736 - accuracy: 0.4995
<br />17696/25000 [====================>.........] - ETA: 24s - loss: 7.6727 - accuracy: 0.4996
<br />17728/25000 [====================>.........] - ETA: 24s - loss: 7.6744 - accuracy: 0.4995
<br />17760/25000 [====================>.........] - ETA: 24s - loss: 7.6761 - accuracy: 0.4994
<br />17792/25000 [====================>.........] - ETA: 24s - loss: 7.6761 - accuracy: 0.4994
<br />17824/25000 [====================>.........] - ETA: 24s - loss: 7.6752 - accuracy: 0.4994
<br />17856/25000 [====================>.........] - ETA: 24s - loss: 7.6761 - accuracy: 0.4994
<br />17888/25000 [====================>.........] - ETA: 24s - loss: 7.6760 - accuracy: 0.4994
<br />17920/25000 [====================>.........] - ETA: 24s - loss: 7.6769 - accuracy: 0.4993
<br />17952/25000 [====================>.........] - ETA: 23s - loss: 7.6743 - accuracy: 0.4995
<br />17984/25000 [====================>.........] - ETA: 23s - loss: 7.6700 - accuracy: 0.4998
<br />18016/25000 [====================>.........] - ETA: 23s - loss: 7.6692 - accuracy: 0.4998
<br />18048/25000 [====================>.........] - ETA: 23s - loss: 7.6709 - accuracy: 0.4997
<br />18080/25000 [====================>.........] - ETA: 23s - loss: 7.6709 - accuracy: 0.4997
<br />18112/25000 [====================>.........] - ETA: 23s - loss: 7.6709 - accuracy: 0.4997
<br />18144/25000 [====================>.........] - ETA: 23s - loss: 7.6734 - accuracy: 0.4996
<br />18176/25000 [====================>.........] - ETA: 23s - loss: 7.6725 - accuracy: 0.4996
<br />18208/25000 [====================>.........] - ETA: 23s - loss: 7.6700 - accuracy: 0.4998
<br />18240/25000 [====================>.........] - ETA: 23s - loss: 7.6683 - accuracy: 0.4999
<br />18272/25000 [====================>.........] - ETA: 22s - loss: 7.6700 - accuracy: 0.4998
<br />18304/25000 [====================>.........] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
<br />18336/25000 [=====================>........] - ETA: 22s - loss: 7.6691 - accuracy: 0.4998
<br />18368/25000 [=====================>........] - ETA: 22s - loss: 7.6691 - accuracy: 0.4998
<br />18400/25000 [=====================>........] - ETA: 22s - loss: 7.6700 - accuracy: 0.4998
<br />18432/25000 [=====================>........] - ETA: 22s - loss: 7.6708 - accuracy: 0.4997
<br />18464/25000 [=====================>........] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
<br />18496/25000 [=====================>........] - ETA: 22s - loss: 7.6658 - accuracy: 0.5001
<br />18528/25000 [=====================>........] - ETA: 22s - loss: 7.6708 - accuracy: 0.4997
<br />18560/25000 [=====================>........] - ETA: 21s - loss: 7.6699 - accuracy: 0.4998
<br />18592/25000 [=====================>........] - ETA: 21s - loss: 7.6699 - accuracy: 0.4998
<br />18624/25000 [=====================>........] - ETA: 21s - loss: 7.6716 - accuracy: 0.4997
<br />18656/25000 [=====================>........] - ETA: 21s - loss: 7.6716 - accuracy: 0.4997
<br />18688/25000 [=====================>........] - ETA: 21s - loss: 7.6691 - accuracy: 0.4998
<br />18720/25000 [=====================>........] - ETA: 21s - loss: 7.6724 - accuracy: 0.4996
<br />18752/25000 [=====================>........] - ETA: 21s - loss: 7.6699 - accuracy: 0.4998
<br />18784/25000 [=====================>........] - ETA: 21s - loss: 7.6658 - accuracy: 0.5001
<br />18816/25000 [=====================>........] - ETA: 21s - loss: 7.6650 - accuracy: 0.5001
<br />18848/25000 [=====================>........] - ETA: 20s - loss: 7.6650 - accuracy: 0.5001
<br />18880/25000 [=====================>........] - ETA: 20s - loss: 7.6642 - accuracy: 0.5002
<br />18912/25000 [=====================>........] - ETA: 20s - loss: 7.6642 - accuracy: 0.5002
<br />18944/25000 [=====================>........] - ETA: 20s - loss: 7.6650 - accuracy: 0.5001
<br />18976/25000 [=====================>........] - ETA: 20s - loss: 7.6626 - accuracy: 0.5003
<br />19008/25000 [=====================>........] - ETA: 20s - loss: 7.6610 - accuracy: 0.5004
<br />19040/25000 [=====================>........] - ETA: 20s - loss: 7.6642 - accuracy: 0.5002
<br />19072/25000 [=====================>........] - ETA: 20s - loss: 7.6626 - accuracy: 0.5003
<br />19104/25000 [=====================>........] - ETA: 20s - loss: 7.6634 - accuracy: 0.5002
<br />19136/25000 [=====================>........] - ETA: 19s - loss: 7.6666 - accuracy: 0.5000
<br />19168/25000 [======================>.......] - ETA: 19s - loss: 7.6658 - accuracy: 0.5001
<br />19200/25000 [======================>.......] - ETA: 19s - loss: 7.6610 - accuracy: 0.5004
<br />19232/25000 [======================>.......] - ETA: 19s - loss: 7.6610 - accuracy: 0.5004
<br />19264/25000 [======================>.......] - ETA: 19s - loss: 7.6595 - accuracy: 0.5005
<br />19296/25000 [======================>.......] - ETA: 19s - loss: 7.6603 - accuracy: 0.5004
<br />19328/25000 [======================>.......] - ETA: 19s - loss: 7.6595 - accuracy: 0.5005
<br />19360/25000 [======================>.......] - ETA: 19s - loss: 7.6587 - accuracy: 0.5005
<br />19392/25000 [======================>.......] - ETA: 19s - loss: 7.6595 - accuracy: 0.5005
<br />19424/25000 [======================>.......] - ETA: 18s - loss: 7.6619 - accuracy: 0.5003
<br />19456/25000 [======================>.......] - ETA: 18s - loss: 7.6627 - accuracy: 0.5003
<br />19488/25000 [======================>.......] - ETA: 18s - loss: 7.6603 - accuracy: 0.5004
<br />19520/25000 [======================>.......] - ETA: 18s - loss: 7.6627 - accuracy: 0.5003
<br />19552/25000 [======================>.......] - ETA: 18s - loss: 7.6619 - accuracy: 0.5003
<br />19584/25000 [======================>.......] - ETA: 18s - loss: 7.6635 - accuracy: 0.5002
<br />19616/25000 [======================>.......] - ETA: 18s - loss: 7.6635 - accuracy: 0.5002
<br />19648/25000 [======================>.......] - ETA: 18s - loss: 7.6627 - accuracy: 0.5003
<br />19680/25000 [======================>.......] - ETA: 18s - loss: 7.6643 - accuracy: 0.5002
<br />19712/25000 [======================>.......] - ETA: 18s - loss: 7.6635 - accuracy: 0.5002
<br />19744/25000 [======================>.......] - ETA: 17s - loss: 7.6635 - accuracy: 0.5002
<br />19776/25000 [======================>.......] - ETA: 17s - loss: 7.6604 - accuracy: 0.5004
<br />19808/25000 [======================>.......] - ETA: 17s - loss: 7.6627 - accuracy: 0.5003
<br />19840/25000 [======================>.......] - ETA: 17s - loss: 7.6635 - accuracy: 0.5002
<br />19872/25000 [======================>.......] - ETA: 17s - loss: 7.6651 - accuracy: 0.5001
<br />19904/25000 [======================>.......] - ETA: 17s - loss: 7.6651 - accuracy: 0.5001
<br />19936/25000 [======================>.......] - ETA: 17s - loss: 7.6712 - accuracy: 0.4997
<br />19968/25000 [======================>.......] - ETA: 17s - loss: 7.6712 - accuracy: 0.4997
<br />20000/25000 [=======================>......] - ETA: 17s - loss: 7.6705 - accuracy: 0.4997
<br />20032/25000 [=======================>......] - ETA: 16s - loss: 7.6697 - accuracy: 0.4998
<br />20064/25000 [=======================>......] - ETA: 16s - loss: 7.6712 - accuracy: 0.4997
<br />20096/25000 [=======================>......] - ETA: 16s - loss: 7.6712 - accuracy: 0.4997
<br />20128/25000 [=======================>......] - ETA: 16s - loss: 7.6704 - accuracy: 0.4998
<br />20160/25000 [=======================>......] - ETA: 16s - loss: 7.6727 - accuracy: 0.4996
<br />20192/25000 [=======================>......] - ETA: 16s - loss: 7.6757 - accuracy: 0.4994
<br />20224/25000 [=======================>......] - ETA: 16s - loss: 7.6750 - accuracy: 0.4995
<br />20256/25000 [=======================>......] - ETA: 16s - loss: 7.6749 - accuracy: 0.4995
<br />20288/25000 [=======================>......] - ETA: 16s - loss: 7.6772 - accuracy: 0.4993
<br />20320/25000 [=======================>......] - ETA: 15s - loss: 7.6772 - accuracy: 0.4993
<br />20352/25000 [=======================>......] - ETA: 15s - loss: 7.6764 - accuracy: 0.4994
<br />20384/25000 [=======================>......] - ETA: 15s - loss: 7.6726 - accuracy: 0.4996
<br />20416/25000 [=======================>......] - ETA: 15s - loss: 7.6719 - accuracy: 0.4997
<br />20448/25000 [=======================>......] - ETA: 15s - loss: 7.6719 - accuracy: 0.4997
<br />20480/25000 [=======================>......] - ETA: 15s - loss: 7.6756 - accuracy: 0.4994
<br />20512/25000 [=======================>......] - ETA: 15s - loss: 7.6748 - accuracy: 0.4995
<br />20544/25000 [=======================>......] - ETA: 15s - loss: 7.6711 - accuracy: 0.4997
<br />20576/25000 [=======================>......] - ETA: 15s - loss: 7.6718 - accuracy: 0.4997
<br />20608/25000 [=======================>......] - ETA: 14s - loss: 7.6703 - accuracy: 0.4998
<br />20640/25000 [=======================>......] - ETA: 14s - loss: 7.6726 - accuracy: 0.4996
<br />20672/25000 [=======================>......] - ETA: 14s - loss: 7.6711 - accuracy: 0.4997
<br />20704/25000 [=======================>......] - ETA: 14s - loss: 7.6681 - accuracy: 0.4999
<br />20736/25000 [=======================>......] - ETA: 14s - loss: 7.6637 - accuracy: 0.5002
<br />20768/25000 [=======================>......] - ETA: 14s - loss: 7.6659 - accuracy: 0.5000
<br />20800/25000 [=======================>......] - ETA: 14s - loss: 7.6688 - accuracy: 0.4999
<br />20832/25000 [=======================>......] - ETA: 14s - loss: 7.6688 - accuracy: 0.4999
<br />20864/25000 [========================>.....] - ETA: 14s - loss: 7.6644 - accuracy: 0.5001
<br />20896/25000 [========================>.....] - ETA: 13s - loss: 7.6622 - accuracy: 0.5003
<br />20928/25000 [========================>.....] - ETA: 13s - loss: 7.6637 - accuracy: 0.5002
<br />20960/25000 [========================>.....] - ETA: 13s - loss: 7.6622 - accuracy: 0.5003
<br />20992/25000 [========================>.....] - ETA: 13s - loss: 7.6637 - accuracy: 0.5002
<br />21024/25000 [========================>.....] - ETA: 13s - loss: 7.6630 - accuracy: 0.5002
<br />21056/25000 [========================>.....] - ETA: 13s - loss: 7.6644 - accuracy: 0.5001
<br />21088/25000 [========================>.....] - ETA: 13s - loss: 7.6608 - accuracy: 0.5004
<br />21120/25000 [========================>.....] - ETA: 13s - loss: 7.6615 - accuracy: 0.5003
<br />21152/25000 [========================>.....] - ETA: 13s - loss: 7.6623 - accuracy: 0.5003
<br />21184/25000 [========================>.....] - ETA: 12s - loss: 7.6623 - accuracy: 0.5003
<br />21216/25000 [========================>.....] - ETA: 12s - loss: 7.6637 - accuracy: 0.5002
<br />21248/25000 [========================>.....] - ETA: 12s - loss: 7.6652 - accuracy: 0.5001
<br />21280/25000 [========================>.....] - ETA: 12s - loss: 7.6681 - accuracy: 0.4999
<br />21312/25000 [========================>.....] - ETA: 12s - loss: 7.6673 - accuracy: 0.5000
<br />21344/25000 [========================>.....] - ETA: 12s - loss: 7.6688 - accuracy: 0.4999
<br />21376/25000 [========================>.....] - ETA: 12s - loss: 7.6688 - accuracy: 0.4999
<br />21408/25000 [========================>.....] - ETA: 12s - loss: 7.6666 - accuracy: 0.5000
<br />21440/25000 [========================>.....] - ETA: 12s - loss: 7.6680 - accuracy: 0.4999
<br />21472/25000 [========================>.....] - ETA: 12s - loss: 7.6673 - accuracy: 0.5000
<br />21504/25000 [========================>.....] - ETA: 11s - loss: 7.6645 - accuracy: 0.5001
<br />21536/25000 [========================>.....] - ETA: 11s - loss: 7.6652 - accuracy: 0.5001
<br />21568/25000 [========================>.....] - ETA: 11s - loss: 7.6680 - accuracy: 0.4999
<br />21600/25000 [========================>.....] - ETA: 11s - loss: 7.6659 - accuracy: 0.5000
<br />21632/25000 [========================>.....] - ETA: 11s - loss: 7.6624 - accuracy: 0.5003
<br />21664/25000 [========================>.....] - ETA: 11s - loss: 7.6638 - accuracy: 0.5002
<br />21696/25000 [=========================>....] - ETA: 11s - loss: 7.6631 - accuracy: 0.5002
<br />21728/25000 [=========================>....] - ETA: 11s - loss: 7.6603 - accuracy: 0.5004
<br />21760/25000 [=========================>....] - ETA: 11s - loss: 7.6596 - accuracy: 0.5005
<br />21792/25000 [=========================>....] - ETA: 10s - loss: 7.6568 - accuracy: 0.5006
<br />21824/25000 [=========================>....] - ETA: 10s - loss: 7.6582 - accuracy: 0.5005
<br />21856/25000 [=========================>....] - ETA: 10s - loss: 7.6568 - accuracy: 0.5006
<br />21888/25000 [=========================>....] - ETA: 10s - loss: 7.6568 - accuracy: 0.5006
<br />21920/25000 [=========================>....] - ETA: 10s - loss: 7.6568 - accuracy: 0.5006
<br />21952/25000 [=========================>....] - ETA: 10s - loss: 7.6582 - accuracy: 0.5005
<br />21984/25000 [=========================>....] - ETA: 10s - loss: 7.6610 - accuracy: 0.5004
<br />22016/25000 [=========================>....] - ETA: 10s - loss: 7.6569 - accuracy: 0.5006
<br />22048/25000 [=========================>....] - ETA: 10s - loss: 7.6569 - accuracy: 0.5006
<br />22080/25000 [=========================>....] - ETA: 9s - loss: 7.6555 - accuracy: 0.5007 
<br />22112/25000 [=========================>....] - ETA: 9s - loss: 7.6555 - accuracy: 0.5007
<br />22144/25000 [=========================>....] - ETA: 9s - loss: 7.6569 - accuracy: 0.5006
<br />22176/25000 [=========================>....] - ETA: 9s - loss: 7.6562 - accuracy: 0.5007
<br />22208/25000 [=========================>....] - ETA: 9s - loss: 7.6535 - accuracy: 0.5009
<br />22240/25000 [=========================>....] - ETA: 9s - loss: 7.6508 - accuracy: 0.5010
<br />22272/25000 [=========================>....] - ETA: 9s - loss: 7.6501 - accuracy: 0.5011
<br />22304/25000 [=========================>....] - ETA: 9s - loss: 7.6522 - accuracy: 0.5009
<br />22336/25000 [=========================>....] - ETA: 9s - loss: 7.6515 - accuracy: 0.5010
<br />22368/25000 [=========================>....] - ETA: 8s - loss: 7.6515 - accuracy: 0.5010
<br />22400/25000 [=========================>....] - ETA: 8s - loss: 7.6481 - accuracy: 0.5012
<br />22432/25000 [=========================>....] - ETA: 8s - loss: 7.6454 - accuracy: 0.5014
<br />22464/25000 [=========================>....] - ETA: 8s - loss: 7.6448 - accuracy: 0.5014
<br />22496/25000 [=========================>....] - ETA: 8s - loss: 7.6421 - accuracy: 0.5016
<br />22528/25000 [==========================>...] - ETA: 8s - loss: 7.6435 - accuracy: 0.5015
<br />22560/25000 [==========================>...] - ETA: 8s - loss: 7.6442 - accuracy: 0.5015
<br />22592/25000 [==========================>...] - ETA: 8s - loss: 7.6442 - accuracy: 0.5015
<br />22624/25000 [==========================>...] - ETA: 8s - loss: 7.6449 - accuracy: 0.5014
<br />22656/25000 [==========================>...] - ETA: 7s - loss: 7.6463 - accuracy: 0.5013
<br />22688/25000 [==========================>...] - ETA: 7s - loss: 7.6470 - accuracy: 0.5013
<br />22720/25000 [==========================>...] - ETA: 7s - loss: 7.6450 - accuracy: 0.5014
<br />22752/25000 [==========================>...] - ETA: 7s - loss: 7.6471 - accuracy: 0.5013
<br />22784/25000 [==========================>...] - ETA: 7s - loss: 7.6484 - accuracy: 0.5012
<br />22816/25000 [==========================>...] - ETA: 7s - loss: 7.6505 - accuracy: 0.5011
<br />22848/25000 [==========================>...] - ETA: 7s - loss: 7.6498 - accuracy: 0.5011
<br />22880/25000 [==========================>...] - ETA: 7s - loss: 7.6525 - accuracy: 0.5009
<br />22912/25000 [==========================>...] - ETA: 7s - loss: 7.6573 - accuracy: 0.5006
<br />22944/25000 [==========================>...] - ETA: 7s - loss: 7.6586 - accuracy: 0.5005
<br />22976/25000 [==========================>...] - ETA: 6s - loss: 7.6593 - accuracy: 0.5005
<br />23008/25000 [==========================>...] - ETA: 6s - loss: 7.6586 - accuracy: 0.5005
<br />23040/25000 [==========================>...] - ETA: 6s - loss: 7.6586 - accuracy: 0.5005
<br />23072/25000 [==========================>...] - ETA: 6s - loss: 7.6593 - accuracy: 0.5005
<br />23104/25000 [==========================>...] - ETA: 6s - loss: 7.6613 - accuracy: 0.5003
<br />23136/25000 [==========================>...] - ETA: 6s - loss: 7.6633 - accuracy: 0.5002
<br />23168/25000 [==========================>...] - ETA: 6s - loss: 7.6626 - accuracy: 0.5003
<br />23200/25000 [==========================>...] - ETA: 6s - loss: 7.6640 - accuracy: 0.5002
<br />23232/25000 [==========================>...] - ETA: 6s - loss: 7.6633 - accuracy: 0.5002
<br />23264/25000 [==========================>...] - ETA: 5s - loss: 7.6646 - accuracy: 0.5001
<br />23296/25000 [==========================>...] - ETA: 5s - loss: 7.6646 - accuracy: 0.5001
<br />23328/25000 [==========================>...] - ETA: 5s - loss: 7.6640 - accuracy: 0.5002
<br />23360/25000 [===========================>..] - ETA: 5s - loss: 7.6633 - accuracy: 0.5002
<br />23392/25000 [===========================>..] - ETA: 5s - loss: 7.6588 - accuracy: 0.5005
<br />23424/25000 [===========================>..] - ETA: 5s - loss: 7.6594 - accuracy: 0.5005
<br />23456/25000 [===========================>..] - ETA: 5s - loss: 7.6588 - accuracy: 0.5005
<br />23488/25000 [===========================>..] - ETA: 5s - loss: 7.6601 - accuracy: 0.5004
<br />23520/25000 [===========================>..] - ETA: 5s - loss: 7.6608 - accuracy: 0.5004
<br />23552/25000 [===========================>..] - ETA: 4s - loss: 7.6614 - accuracy: 0.5003
<br />23584/25000 [===========================>..] - ETA: 4s - loss: 7.6575 - accuracy: 0.5006
<br />23616/25000 [===========================>..] - ETA: 4s - loss: 7.6582 - accuracy: 0.5006
<br />23648/25000 [===========================>..] - ETA: 4s - loss: 7.6575 - accuracy: 0.5006
<br />23680/25000 [===========================>..] - ETA: 4s - loss: 7.6543 - accuracy: 0.5008
<br />23712/25000 [===========================>..] - ETA: 4s - loss: 7.6524 - accuracy: 0.5009
<br />23744/25000 [===========================>..] - ETA: 4s - loss: 7.6518 - accuracy: 0.5010
<br />23776/25000 [===========================>..] - ETA: 4s - loss: 7.6518 - accuracy: 0.5010
<br />23808/25000 [===========================>..] - ETA: 4s - loss: 7.6544 - accuracy: 0.5008
<br />23840/25000 [===========================>..] - ETA: 3s - loss: 7.6544 - accuracy: 0.5008
<br />23872/25000 [===========================>..] - ETA: 3s - loss: 7.6563 - accuracy: 0.5007
<br />23904/25000 [===========================>..] - ETA: 3s - loss: 7.6557 - accuracy: 0.5007
<br />23936/25000 [===========================>..] - ETA: 3s - loss: 7.6551 - accuracy: 0.5008
<br />23968/25000 [===========================>..] - ETA: 3s - loss: 7.6557 - accuracy: 0.5007
<br />24000/25000 [===========================>..] - ETA: 3s - loss: 7.6577 - accuracy: 0.5006
<br />24032/25000 [===========================>..] - ETA: 3s - loss: 7.6577 - accuracy: 0.5006
<br />24064/25000 [===========================>..] - ETA: 3s - loss: 7.6564 - accuracy: 0.5007
<br />24096/25000 [===========================>..] - ETA: 3s - loss: 7.6571 - accuracy: 0.5006
<br />24128/25000 [===========================>..] - ETA: 2s - loss: 7.6545 - accuracy: 0.5008
<br />24160/25000 [===========================>..] - ETA: 2s - loss: 7.6571 - accuracy: 0.5006
<br />24192/25000 [============================>.] - ETA: 2s - loss: 7.6590 - accuracy: 0.5005
<br />24224/25000 [============================>.] - ETA: 2s - loss: 7.6571 - accuracy: 0.5006
<br />24256/25000 [============================>.] - ETA: 2s - loss: 7.6578 - accuracy: 0.5006
<br />24288/25000 [============================>.] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
<br />24320/25000 [============================>.] - ETA: 2s - loss: 7.6591 - accuracy: 0.5005
<br />24352/25000 [============================>.] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
<br />24384/25000 [============================>.] - ETA: 2s - loss: 7.6566 - accuracy: 0.5007
<br />24416/25000 [============================>.] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
<br />24448/25000 [============================>.] - ETA: 1s - loss: 7.6578 - accuracy: 0.5006
<br />24480/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5004
<br />24512/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
<br />24544/25000 [============================>.] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
<br />24576/25000 [============================>.] - ETA: 1s - loss: 7.6579 - accuracy: 0.5006
<br />24608/25000 [============================>.] - ETA: 1s - loss: 7.6573 - accuracy: 0.5006
<br />24640/25000 [============================>.] - ETA: 1s - loss: 7.6598 - accuracy: 0.5004
<br />24672/25000 [============================>.] - ETA: 1s - loss: 7.6585 - accuracy: 0.5005
<br />24704/25000 [============================>.] - ETA: 1s - loss: 7.6586 - accuracy: 0.5005
<br />24736/25000 [============================>.] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
<br />24768/25000 [============================>.] - ETA: 0s - loss: 7.6592 - accuracy: 0.5005
<br />24800/25000 [============================>.] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
<br />24832/25000 [============================>.] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
<br />24864/25000 [============================>.] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
<br />24896/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
<br />24928/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
<br />24960/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
<br />24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
<br />25000/25000 [==============================] - 103s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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



### Error 16, [Traceback at line 1725](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L1725)<br />1725..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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



### Error 17, [Traceback at line 1743](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L1743)<br />1743..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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



### Error 18, [Traceback at line 1758](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L1758)<br />1758..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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



### Error 19, [Traceback at line 1784](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L1784)<br />1784..[0;31mNameError[0m                                 Traceback (most recent call last)
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



### Error 20, [Traceback at line 1800](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L1800)<br />1800..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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



### Error 21, [Traceback at line 1821](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L1821)<br />1821..[0;31mIndexError[0m                                Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mIndexError[0m: tuple index out of range
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 22, [Traceback at line 1831](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L1831)<br />1831..[0;31mNameError[0m                                 Traceback (most recent call last)
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



### Error 23, [Traceback at line 1857](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L1857)<br />1857..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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
<br />[0;36m  File [0;32m"https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/example/vision_mnist.py"[0;36m, line [0;32m15[0m
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



### Error 24, [Traceback at line 1896](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L1896)<br />1896..[0;31mNameError[0m                                 Traceback (most recent call last)
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



### Error 25, [Traceback at line 1916](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L1916)<br />1916..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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
<br />[0;36m  File [0;32m"https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/example/benchmark_timeseries_m5.py"[0;36m, line [0;32m248[0m
<br />[0;31m    We then reshape the forecasts into the correct data shape for submission ...[0m
<br />[0m          ^[0m
<br />[0;31mSyntaxError[0m[0;31m:[0m invalid syntax
<br />
<br />
<br />
<br />
<br />
<br />
<br /> ************************************************************************************************************************
<br />ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//arun_model.py 
<br />
<br /><module 'mlmodels' from 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/__init__.py'>
<br />https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras/ardmn.json
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 26, [Traceback at line 1949](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-14-15-13_207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2.py#L1949)<br />1949..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example/arun_model.py[0m in [0;36m<module>[0;34m[0m
<br />[1;32m     25[0m [0;31m# Model Parameters[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m     26[0m [0;31m# model_pars, data_pars, compute_pars, out_pars[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 27[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m[0mconfig_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     28[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m     29[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpdict[0m   [0;34m)[0m   [0;31m###Normalize path[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/model_keras/ardmn.json'
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
<br />[0;36m  File [0;32m"https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2/mlmodels/example/benchmark_timeseries_m5.py"[0;36m, line [0;32m248[0m
<br />[0;31m    We then reshape the forecasts into the correct data shape for submission ...[0m
<br />[0m          ^[0m
<br />[0;31mSyntaxError[0m[0;31m:[0m invalid syntax
<br />
