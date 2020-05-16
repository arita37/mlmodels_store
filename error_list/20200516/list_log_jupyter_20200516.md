## Original File URL: https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py


### Error 1, [Traceback at line 42](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L42)<br />42..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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



### Error 2, [Traceback at line 63](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L63)<br />63..[0;31mIndexError[0m                                Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mIndexError[0m: tuple index out of range
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 3, [Traceback at line 73](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L73)<br />73..[0;31mNameError[0m                                 Traceback (most recent call last)
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



### Error 4, [Traceback at line 98](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L98)<br />98..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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



### Error 5, [Traceback at line 116](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L116)<br />116..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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



### Error 6, [Traceback at line 137](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L137)<br />137..[0;31mIndexError[0m                                Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mIndexError[0m: tuple index out of range
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 7, [Traceback at line 147](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L147)<br />147..[0;31mNameError[0m                                 Traceback (most recent call last)
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



### Error 8, [Traceback at line 172](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L172)<br />172..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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



### Error 9, [Traceback at line 188](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L188)<br />188..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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



### Error 10, [Traceback at line 206](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L206)<br />206..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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
<br />	Data preprocessing and feature engineering runtime = 0.3s ...
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



### Error 11, [Traceback at line 271](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L271)<br />271..Traceback (most recent call last):
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



### Error 12, [Traceback at line 403](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L403)<br />403..Traceback (most recent call last):
<br />  File "/opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
<br />    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 523, in main
<br />    test_cli(arg)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 453, in test_cli
<br />    test_module(arg.model_uri, param_pars=param_pars)  # '1_lstm'
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/models.py", line 257, in test_module
<br />    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
<br />  File "https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_gluon/gluon_automl.py", line 109, in get_params
<br />    return model_pars, data_pars, compute_pars, out_pars
<br />UnboundLocalError: local variable 'model_pars' referenced before assignment



### Error 13, [Traceback at line 424](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L424)<br />424..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />non-resource variables are not supported in the long term
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
<br />https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
<br />https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />  <module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tf/1_lstm.py'> 
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  #### Model init   ############################################ 
<br />
<br />  <mlmodels.model_tf.1_lstm.Model object at 0x7f8fb6738ac8> 
<br />
<br />  #### Fit   ######################################################## 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br /> [-0.08564074  0.06497264 -0.01655756 -0.00090496 -0.02879354  0.07852428]
<br /> [ 0.00970696  0.19001883 -0.03477718  0.09692499 -0.15627864  0.05839501]
<br /> [-0.03038519 -0.08257481  0.10741445  0.43009552  0.11023194  0.15371193]
<br /> [-0.05999951  0.14061755  0.09784633  0.05734454  0.33897623  0.10788444]
<br /> [-0.03365828  0.31074873 -0.0024698   0.04576865 -0.06958254  0.38334733]
<br /> [-0.06228766  0.09705011 -0.26475406  0.45682052  0.14575714  0.19166605]
<br /> [-0.49674183  0.77981418 -0.43618864  0.34925169  0.5712015   0.41432109]
<br /> [ 0.46336123  0.79205626  0.10868432  0.19246262  0.36018971  0.43245375]
<br /> [ 0.          0.          0.          0.          0.          0.        ]]
<br />
<br />  #### Get  metrics   ################################################ 
<br />
<br />  #### Save   ######################################################## 
<br />
<br />  #### Load   ######################################################## 
<br />model_tf/1_lstm.py
<br />model_tf.1_lstm.py
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.40036983974277973, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-15 20:18:54.089964: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/ztest/model_tf/1_lstm/model'}
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
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tf/1_lstm.py'>
<br /><module 'mlmodels.model_tf.1_lstm' from 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_tf/1_lstm.py'>
<br />
<br />  #### Loading params   ############################################## 
<br />
<br />  ############# Data, Params preparation   ################# 
<br />
<br />  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/ztest/model_tf/1_lstm/model'} 
<br />
<br />  #### Loading dataset   ############################################# 
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
<br />https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'data_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
<br />https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/dataset/timeseries/GOOG-year.csv
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
<br />{'loss': 0.4746616706252098, 'loss_history': []}
<br />
<br />  #### Plot   ######################################################## 
<br />
<br />  #### Save   ######################################################## 
<br />{'path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/ztest/model_tf/1_lstm/'}
<br />Model saved in path: https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt
<br />
<br />  #### Load   ######################################################## 
<br />2020-05-15 20:18:55.279726: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
<br />{'path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/ztest/model_tf/1_lstm/', 'model_path': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/ztest/model_tf/1_lstm/model'}
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



### Error 14, [Traceback at line 808](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L808)<br />808..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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



### Error 15, [Traceback at line 823](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L823)<br />823..[0;31mNameError[0m                                 Traceback (most recent call last)
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
<br /> 2588672/17464789 [===>..........................] - ETA: 0s
<br /> 7716864/17464789 [============>.................] - ETA: 0s
<br />13107200/17464789 [=====================>........] - ETA: 0s
<br />17465344/17464789 [==============================] - 0s 0us/step
<br />Pad sequences (samples x time)...
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
<br />Instructions for updating:
<br />Use tf.where in 2.0, which has the same broadcast rule as np.where
<br />2020-05-15 20:19:07.118651: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
<br />2020-05-15 20:19:07.122799: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
<br />2020-05-15 20:19:07.122946: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5564f04edb30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
<br />2020-05-15 20:19:07.122960: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
<br />WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
<br />
<br />Train on 25000 samples, validate on 25000 samples
<br />Epoch 1/1
<br />
<br />   32/25000 [..............................] - ETA: 4:57 - loss: 8.1458 - accuracy: 0.4688
<br />   64/25000 [..............................] - ETA: 3:13 - loss: 7.9062 - accuracy: 0.4844
<br />   96/25000 [..............................] - ETA: 2:38 - loss: 7.1875 - accuracy: 0.5312
<br />  128/25000 [..............................] - ETA: 2:19 - loss: 7.5468 - accuracy: 0.5078
<br />  160/25000 [..............................] - ETA: 2:07 - loss: 7.3791 - accuracy: 0.5188
<br />  192/25000 [..............................] - ETA: 2:01 - loss: 7.3472 - accuracy: 0.5208
<br />  224/25000 [..............................] - ETA: 1:56 - loss: 7.2559 - accuracy: 0.5268
<br />  256/25000 [..............................] - ETA: 1:52 - loss: 7.1875 - accuracy: 0.5312
<br />  288/25000 [..............................] - ETA: 1:49 - loss: 7.2407 - accuracy: 0.5278
<br />  320/25000 [..............................] - ETA: 1:47 - loss: 7.3791 - accuracy: 0.5188
<br />  352/25000 [..............................] - ETA: 1:45 - loss: 7.3617 - accuracy: 0.5199
<br />  384/25000 [..............................] - ETA: 1:43 - loss: 7.1875 - accuracy: 0.5312
<br />  416/25000 [..............................] - ETA: 1:41 - loss: 7.2243 - accuracy: 0.5288
<br />  448/25000 [..............................] - ETA: 1:40 - loss: 7.2901 - accuracy: 0.5246
<br />  480/25000 [..............................] - ETA: 1:39 - loss: 7.3152 - accuracy: 0.5229
<br />  512/25000 [..............................] - ETA: 1:38 - loss: 7.3372 - accuracy: 0.5215
<br />  544/25000 [..............................] - ETA: 1:38 - loss: 7.3284 - accuracy: 0.5221
<br />  576/25000 [..............................] - ETA: 1:37 - loss: 7.3472 - accuracy: 0.5208
<br />  608/25000 [..............................] - ETA: 1:37 - loss: 7.3388 - accuracy: 0.5214
<br />  640/25000 [..............................] - ETA: 1:36 - loss: 7.4510 - accuracy: 0.5141
<br />  672/25000 [..............................] - ETA: 1:36 - loss: 7.5069 - accuracy: 0.5104
<br />  704/25000 [..............................] - ETA: 1:35 - loss: 7.3835 - accuracy: 0.5185
<br />  736/25000 [..............................] - ETA: 1:35 - loss: 7.3958 - accuracy: 0.5177
<br />  768/25000 [..............................] - ETA: 1:34 - loss: 7.5269 - accuracy: 0.5091
<br />  800/25000 [..............................] - ETA: 1:34 - loss: 7.4558 - accuracy: 0.5138
<br />  832/25000 [..............................] - ETA: 1:34 - loss: 7.3717 - accuracy: 0.5192
<br />  864/25000 [>.............................] - ETA: 1:34 - loss: 7.3649 - accuracy: 0.5197
<br />  896/25000 [>.............................] - ETA: 1:34 - loss: 7.4442 - accuracy: 0.5145
<br />  928/25000 [>.............................] - ETA: 1:34 - loss: 7.4849 - accuracy: 0.5119
<br />  960/25000 [>.............................] - ETA: 1:33 - loss: 7.4909 - accuracy: 0.5115
<br />  992/25000 [>.............................] - ETA: 1:33 - loss: 7.4966 - accuracy: 0.5111
<br /> 1024/25000 [>.............................] - ETA: 1:32 - loss: 7.4720 - accuracy: 0.5127
<br /> 1056/25000 [>.............................] - ETA: 1:32 - loss: 7.4343 - accuracy: 0.5152
<br /> 1088/25000 [>.............................] - ETA: 1:31 - loss: 7.4129 - accuracy: 0.5165
<br /> 1120/25000 [>.............................] - ETA: 1:31 - loss: 7.4202 - accuracy: 0.5161
<br /> 1152/25000 [>.............................] - ETA: 1:31 - loss: 7.4270 - accuracy: 0.5156
<br /> 1184/25000 [>.............................] - ETA: 1:30 - loss: 7.4594 - accuracy: 0.5135
<br /> 1216/25000 [>.............................] - ETA: 1:30 - loss: 7.4018 - accuracy: 0.5173
<br /> 1248/25000 [>.............................] - ETA: 1:30 - loss: 7.3595 - accuracy: 0.5200
<br /> 1280/25000 [>.............................] - ETA: 1:29 - loss: 7.3791 - accuracy: 0.5188
<br /> 1312/25000 [>.............................] - ETA: 1:29 - loss: 7.3978 - accuracy: 0.5175
<br /> 1344/25000 [>.............................] - ETA: 1:29 - loss: 7.3928 - accuracy: 0.5179
<br /> 1376/25000 [>.............................] - ETA: 1:29 - loss: 7.3657 - accuracy: 0.5196
<br /> 1408/25000 [>.............................] - ETA: 1:28 - loss: 7.3399 - accuracy: 0.5213
<br /> 1440/25000 [>.............................] - ETA: 1:28 - loss: 7.3365 - accuracy: 0.5215
<br /> 1472/25000 [>.............................] - ETA: 1:28 - loss: 7.3333 - accuracy: 0.5217
<br /> 1504/25000 [>.............................] - ETA: 1:27 - loss: 7.3812 - accuracy: 0.5186
<br /> 1536/25000 [>.............................] - ETA: 1:27 - loss: 7.4171 - accuracy: 0.5163
<br /> 1568/25000 [>.............................] - ETA: 1:27 - loss: 7.3830 - accuracy: 0.5185
<br /> 1600/25000 [>.............................] - ETA: 1:27 - loss: 7.3983 - accuracy: 0.5175
<br /> 1632/25000 [>.............................] - ETA: 1:27 - loss: 7.4223 - accuracy: 0.5159
<br /> 1664/25000 [>.............................] - ETA: 1:26 - loss: 7.4363 - accuracy: 0.5150
<br /> 1696/25000 [=>............................] - ETA: 1:26 - loss: 7.4406 - accuracy: 0.5147
<br /> 1728/25000 [=>............................] - ETA: 1:26 - loss: 7.4182 - accuracy: 0.5162
<br /> 1760/25000 [=>............................] - ETA: 1:26 - loss: 7.4401 - accuracy: 0.5148
<br /> 1792/25000 [=>............................] - ETA: 1:26 - loss: 7.4356 - accuracy: 0.5151
<br /> 1824/25000 [=>............................] - ETA: 1:25 - loss: 7.4312 - accuracy: 0.5154
<br /> 1856/25000 [=>............................] - ETA: 1:25 - loss: 7.4105 - accuracy: 0.5167
<br /> 1888/25000 [=>............................] - ETA: 1:25 - loss: 7.4067 - accuracy: 0.5169
<br /> 1920/25000 [=>............................] - ETA: 1:25 - loss: 7.4191 - accuracy: 0.5161
<br /> 1952/25000 [=>............................] - ETA: 1:24 - loss: 7.3760 - accuracy: 0.5190
<br /> 1984/25000 [=>............................] - ETA: 1:24 - loss: 7.3729 - accuracy: 0.5192
<br /> 2016/25000 [=>............................] - ETA: 1:24 - loss: 7.3852 - accuracy: 0.5184
<br /> 2048/25000 [=>............................] - ETA: 1:24 - loss: 7.4196 - accuracy: 0.5161
<br /> 2080/25000 [=>............................] - ETA: 1:24 - loss: 7.4234 - accuracy: 0.5159
<br /> 2112/25000 [=>............................] - ETA: 1:24 - loss: 7.4270 - accuracy: 0.5156
<br /> 2144/25000 [=>............................] - ETA: 1:24 - loss: 7.4235 - accuracy: 0.5159
<br /> 2176/25000 [=>............................] - ETA: 1:23 - loss: 7.4905 - accuracy: 0.5115
<br /> 2208/25000 [=>............................] - ETA: 1:23 - loss: 7.4930 - accuracy: 0.5113
<br /> 2240/25000 [=>............................] - ETA: 1:23 - loss: 7.4613 - accuracy: 0.5134
<br /> 2272/25000 [=>............................] - ETA: 1:23 - loss: 7.4642 - accuracy: 0.5132
<br /> 2304/25000 [=>............................] - ETA: 1:23 - loss: 7.4470 - accuracy: 0.5143
<br /> 2336/25000 [=>............................] - ETA: 1:22 - loss: 7.4763 - accuracy: 0.5124
<br /> 2368/25000 [=>............................] - ETA: 1:22 - loss: 7.4918 - accuracy: 0.5114
<br /> 2400/25000 [=>............................] - ETA: 1:22 - loss: 7.5069 - accuracy: 0.5104
<br /> 2432/25000 [=>............................] - ETA: 1:22 - loss: 7.5153 - accuracy: 0.5099
<br /> 2464/25000 [=>............................] - ETA: 1:22 - loss: 7.5110 - accuracy: 0.5101
<br /> 2496/25000 [=>............................] - ETA: 1:22 - loss: 7.5376 - accuracy: 0.5084
<br /> 2528/25000 [==>...........................] - ETA: 1:21 - loss: 7.5271 - accuracy: 0.5091
<br /> 2560/25000 [==>...........................] - ETA: 1:21 - loss: 7.5528 - accuracy: 0.5074
<br /> 2592/25000 [==>...........................] - ETA: 1:21 - loss: 7.5601 - accuracy: 0.5069
<br /> 2624/25000 [==>...........................] - ETA: 1:21 - loss: 7.5731 - accuracy: 0.5061
<br /> 2656/25000 [==>...........................] - ETA: 1:21 - loss: 7.5569 - accuracy: 0.5072
<br /> 2688/25000 [==>...........................] - ETA: 1:21 - loss: 7.5525 - accuracy: 0.5074
<br /> 2720/25000 [==>...........................] - ETA: 1:21 - loss: 7.5651 - accuracy: 0.5066
<br /> 2752/25000 [==>...........................] - ETA: 1:20 - loss: 7.5775 - accuracy: 0.5058
<br /> 2784/25000 [==>...........................] - ETA: 1:20 - loss: 7.5510 - accuracy: 0.5075
<br /> 2816/25000 [==>...........................] - ETA: 1:20 - loss: 7.5686 - accuracy: 0.5064
<br /> 2848/25000 [==>...........................] - ETA: 1:20 - loss: 7.5805 - accuracy: 0.5056
<br /> 2880/25000 [==>...........................] - ETA: 1:20 - loss: 7.6187 - accuracy: 0.5031
<br /> 2912/25000 [==>...........................] - ETA: 1:20 - loss: 7.6245 - accuracy: 0.5027
<br /> 2944/25000 [==>...........................] - ETA: 1:19 - loss: 7.6250 - accuracy: 0.5027
<br /> 2976/25000 [==>...........................] - ETA: 1:19 - loss: 7.6306 - accuracy: 0.5024
<br /> 3008/25000 [==>...........................] - ETA: 1:19 - loss: 7.6462 - accuracy: 0.5013
<br /> 3040/25000 [==>...........................] - ETA: 1:19 - loss: 7.6464 - accuracy: 0.5013
<br /> 3072/25000 [==>...........................] - ETA: 1:19 - loss: 7.6467 - accuracy: 0.5013
<br /> 3104/25000 [==>...........................] - ETA: 1:19 - loss: 7.6419 - accuracy: 0.5016
<br /> 3136/25000 [==>...........................] - ETA: 1:19 - loss: 7.6422 - accuracy: 0.5016
<br /> 3168/25000 [==>...........................] - ETA: 1:18 - loss: 7.6424 - accuracy: 0.5016
<br /> 3200/25000 [==>...........................] - ETA: 1:18 - loss: 7.6618 - accuracy: 0.5003
<br /> 3232/25000 [==>...........................] - ETA: 1:18 - loss: 7.6619 - accuracy: 0.5003
<br /> 3264/25000 [==>...........................] - ETA: 1:18 - loss: 7.6572 - accuracy: 0.5006
<br /> 3296/25000 [==>...........................] - ETA: 1:18 - loss: 7.6666 - accuracy: 0.5000
<br /> 3328/25000 [==>...........................] - ETA: 1:18 - loss: 7.6436 - accuracy: 0.5015
<br /> 3360/25000 [===>..........................] - ETA: 1:18 - loss: 7.6392 - accuracy: 0.5018
<br /> 3392/25000 [===>..........................] - ETA: 1:17 - loss: 7.6440 - accuracy: 0.5015
<br /> 3424/25000 [===>..........................] - ETA: 1:17 - loss: 7.6532 - accuracy: 0.5009
<br /> 3456/25000 [===>..........................] - ETA: 1:17 - loss: 7.6444 - accuracy: 0.5014
<br /> 3488/25000 [===>..........................] - ETA: 1:17 - loss: 7.6446 - accuracy: 0.5014
<br /> 3520/25000 [===>..........................] - ETA: 1:17 - loss: 7.6623 - accuracy: 0.5003
<br /> 3552/25000 [===>..........................] - ETA: 1:17 - loss: 7.6666 - accuracy: 0.5000
<br /> 3584/25000 [===>..........................] - ETA: 1:17 - loss: 7.6752 - accuracy: 0.4994
<br /> 3616/25000 [===>..........................] - ETA: 1:17 - loss: 7.6751 - accuracy: 0.4994
<br /> 3648/25000 [===>..........................] - ETA: 1:16 - loss: 7.6792 - accuracy: 0.4992
<br /> 3680/25000 [===>..........................] - ETA: 1:16 - loss: 7.6833 - accuracy: 0.4989
<br /> 3712/25000 [===>..........................] - ETA: 1:16 - loss: 7.6955 - accuracy: 0.4981
<br /> 3744/25000 [===>..........................] - ETA: 1:16 - loss: 7.6912 - accuracy: 0.4984
<br /> 3776/25000 [===>..........................] - ETA: 1:16 - loss: 7.6910 - accuracy: 0.4984
<br /> 3808/25000 [===>..........................] - ETA: 1:16 - loss: 7.7230 - accuracy: 0.4963
<br /> 3840/25000 [===>..........................] - ETA: 1:16 - loss: 7.7345 - accuracy: 0.4956
<br /> 3872/25000 [===>..........................] - ETA: 1:16 - loss: 7.7300 - accuracy: 0.4959
<br /> 3904/25000 [===>..........................] - ETA: 1:15 - loss: 7.7295 - accuracy: 0.4959
<br /> 3936/25000 [===>..........................] - ETA: 1:15 - loss: 7.7289 - accuracy: 0.4959
<br /> 3968/25000 [===>..........................] - ETA: 1:15 - loss: 7.7400 - accuracy: 0.4952
<br /> 4000/25000 [===>..........................] - ETA: 1:15 - loss: 7.7356 - accuracy: 0.4955
<br /> 4032/25000 [===>..........................] - ETA: 1:15 - loss: 7.7275 - accuracy: 0.4960
<br /> 4064/25000 [===>..........................] - ETA: 1:15 - loss: 7.7157 - accuracy: 0.4968
<br /> 4096/25000 [===>..........................] - ETA: 1:15 - loss: 7.7041 - accuracy: 0.4976
<br /> 4128/25000 [===>..........................] - ETA: 1:15 - loss: 7.7186 - accuracy: 0.4966
<br /> 4160/25000 [===>..........................] - ETA: 1:14 - loss: 7.7219 - accuracy: 0.4964
<br /> 4192/25000 [====>.........................] - ETA: 1:14 - loss: 7.7069 - accuracy: 0.4974
<br /> 4224/25000 [====>.........................] - ETA: 1:14 - loss: 7.7102 - accuracy: 0.4972
<br /> 4256/25000 [====>.........................] - ETA: 1:14 - loss: 7.7171 - accuracy: 0.4967
<br /> 4288/25000 [====>.........................] - ETA: 1:14 - loss: 7.7167 - accuracy: 0.4967
<br /> 4320/25000 [====>.........................] - ETA: 1:14 - loss: 7.7128 - accuracy: 0.4970
<br /> 4352/25000 [====>.........................] - ETA: 1:14 - loss: 7.7089 - accuracy: 0.4972
<br /> 4384/25000 [====>.........................] - ETA: 1:14 - loss: 7.7156 - accuracy: 0.4968
<br /> 4416/25000 [====>.........................] - ETA: 1:13 - loss: 7.7187 - accuracy: 0.4966
<br /> 4448/25000 [====>.........................] - ETA: 1:13 - loss: 7.7045 - accuracy: 0.4975
<br /> 4480/25000 [====>.........................] - ETA: 1:13 - loss: 7.7043 - accuracy: 0.4975
<br /> 4512/25000 [====>.........................] - ETA: 1:13 - loss: 7.7142 - accuracy: 0.4969
<br /> 4544/25000 [====>.........................] - ETA: 1:13 - loss: 7.7071 - accuracy: 0.4974
<br /> 4576/25000 [====>.........................] - ETA: 1:13 - loss: 7.7169 - accuracy: 0.4967
<br /> 4608/25000 [====>.........................] - ETA: 1:13 - loss: 7.7165 - accuracy: 0.4967
<br /> 4640/25000 [====>.........................] - ETA: 1:13 - loss: 7.7096 - accuracy: 0.4972
<br /> 4672/25000 [====>.........................] - ETA: 1:12 - loss: 7.7126 - accuracy: 0.4970
<br /> 4704/25000 [====>.........................] - ETA: 1:12 - loss: 7.7188 - accuracy: 0.4966
<br /> 4736/25000 [====>.........................] - ETA: 1:12 - loss: 7.7055 - accuracy: 0.4975
<br /> 4768/25000 [====>.........................] - ETA: 1:12 - loss: 7.7020 - accuracy: 0.4977
<br /> 4800/25000 [====>.........................] - ETA: 1:12 - loss: 7.7241 - accuracy: 0.4963
<br /> 4832/25000 [====>.........................] - ETA: 1:12 - loss: 7.7174 - accuracy: 0.4967
<br /> 4864/25000 [====>.........................] - ETA: 1:12 - loss: 7.7108 - accuracy: 0.4971
<br /> 4896/25000 [====>.........................] - ETA: 1:12 - loss: 7.7230 - accuracy: 0.4963
<br /> 4928/25000 [====>.........................] - ETA: 1:11 - loss: 7.7257 - accuracy: 0.4961
<br /> 4960/25000 [====>.........................] - ETA: 1:11 - loss: 7.7377 - accuracy: 0.4954
<br /> 4992/25000 [====>.........................] - ETA: 1:11 - loss: 7.7342 - accuracy: 0.4956
<br /> 5024/25000 [=====>........................] - ETA: 1:11 - loss: 7.7216 - accuracy: 0.4964
<br /> 5056/25000 [=====>........................] - ETA: 1:11 - loss: 7.7242 - accuracy: 0.4962
<br /> 5088/25000 [=====>........................] - ETA: 1:11 - loss: 7.7179 - accuracy: 0.4967
<br /> 5120/25000 [=====>........................] - ETA: 1:11 - loss: 7.7145 - accuracy: 0.4969
<br /> 5152/25000 [=====>........................] - ETA: 1:11 - loss: 7.7113 - accuracy: 0.4971
<br /> 5184/25000 [=====>........................] - ETA: 1:11 - loss: 7.7080 - accuracy: 0.4973
<br /> 5216/25000 [=====>........................] - ETA: 1:10 - loss: 7.7195 - accuracy: 0.4965
<br /> 5248/25000 [=====>........................] - ETA: 1:10 - loss: 7.7163 - accuracy: 0.4968
<br /> 5280/25000 [=====>........................] - ETA: 1:10 - loss: 7.7160 - accuracy: 0.4968
<br /> 5312/25000 [=====>........................] - ETA: 1:10 - loss: 7.7330 - accuracy: 0.4957
<br /> 5344/25000 [=====>........................] - ETA: 1:10 - loss: 7.7470 - accuracy: 0.4948
<br /> 5376/25000 [=====>........................] - ETA: 1:10 - loss: 7.7436 - accuracy: 0.4950
<br /> 5408/25000 [=====>........................] - ETA: 1:10 - loss: 7.7545 - accuracy: 0.4943
<br /> 5440/25000 [=====>........................] - ETA: 1:10 - loss: 7.7681 - accuracy: 0.4934
<br /> 5472/25000 [=====>........................] - ETA: 1:09 - loss: 7.7703 - accuracy: 0.4932
<br /> 5504/25000 [=====>........................] - ETA: 1:09 - loss: 7.7781 - accuracy: 0.4927
<br /> 5536/25000 [=====>........................] - ETA: 1:09 - loss: 7.7774 - accuracy: 0.4928
<br /> 5568/25000 [=====>........................] - ETA: 1:09 - loss: 7.7768 - accuracy: 0.4928
<br /> 5600/25000 [=====>........................] - ETA: 1:09 - loss: 7.7871 - accuracy: 0.4921
<br /> 5632/25000 [=====>........................] - ETA: 1:09 - loss: 7.7891 - accuracy: 0.4920
<br /> 5664/25000 [=====>........................] - ETA: 1:09 - loss: 7.7939 - accuracy: 0.4917
<br /> 5696/25000 [=====>........................] - ETA: 1:09 - loss: 7.8039 - accuracy: 0.4910
<br /> 5728/25000 [=====>........................] - ETA: 1:08 - loss: 7.7817 - accuracy: 0.4925
<br /> 5760/25000 [=====>........................] - ETA: 1:08 - loss: 7.7731 - accuracy: 0.4931
<br /> 5792/25000 [=====>........................] - ETA: 1:08 - loss: 7.7805 - accuracy: 0.4926
<br /> 5824/25000 [=====>........................] - ETA: 1:08 - loss: 7.7746 - accuracy: 0.4930
<br /> 5856/25000 [======>.......................] - ETA: 1:08 - loss: 7.7766 - accuracy: 0.4928
<br /> 5888/25000 [======>.......................] - ETA: 1:08 - loss: 7.7760 - accuracy: 0.4929
<br /> 5920/25000 [======>.......................] - ETA: 1:08 - loss: 7.7754 - accuracy: 0.4929
<br /> 5952/25000 [======>.......................] - ETA: 1:08 - loss: 7.7748 - accuracy: 0.4929
<br /> 5984/25000 [======>.......................] - ETA: 1:08 - loss: 7.7742 - accuracy: 0.4930
<br /> 6016/25000 [======>.......................] - ETA: 1:08 - loss: 7.7711 - accuracy: 0.4932
<br /> 6048/25000 [======>.......................] - ETA: 1:07 - loss: 7.7680 - accuracy: 0.4934
<br /> 6080/25000 [======>.......................] - ETA: 1:07 - loss: 7.7826 - accuracy: 0.4924
<br /> 6112/25000 [======>.......................] - ETA: 1:07 - loss: 7.7795 - accuracy: 0.4926
<br /> 6144/25000 [======>.......................] - ETA: 1:07 - loss: 7.7789 - accuracy: 0.4927
<br /> 6176/25000 [======>.......................] - ETA: 1:07 - loss: 7.7883 - accuracy: 0.4921
<br /> 6208/25000 [======>.......................] - ETA: 1:07 - loss: 7.7876 - accuracy: 0.4921
<br /> 6240/25000 [======>.......................] - ETA: 1:07 - loss: 7.7846 - accuracy: 0.4923
<br /> 6272/25000 [======>.......................] - ETA: 1:07 - loss: 7.7815 - accuracy: 0.4925
<br /> 6304/25000 [======>.......................] - ETA: 1:06 - loss: 7.7907 - accuracy: 0.4919
<br /> 6336/25000 [======>.......................] - ETA: 1:06 - loss: 7.7876 - accuracy: 0.4921
<br /> 6368/25000 [======>.......................] - ETA: 1:06 - loss: 7.7991 - accuracy: 0.4914
<br /> 6400/25000 [======>.......................] - ETA: 1:06 - loss: 7.8032 - accuracy: 0.4911
<br /> 6432/25000 [======>.......................] - ETA: 1:06 - loss: 7.7954 - accuracy: 0.4916
<br /> 6464/25000 [======>.......................] - ETA: 1:06 - loss: 7.7805 - accuracy: 0.4926
<br /> 6496/25000 [======>.......................] - ETA: 1:06 - loss: 7.7776 - accuracy: 0.4928
<br /> 6528/25000 [======>.......................] - ETA: 1:06 - loss: 7.7841 - accuracy: 0.4923
<br /> 6560/25000 [======>.......................] - ETA: 1:06 - loss: 7.7928 - accuracy: 0.4918
<br /> 6592/25000 [======>.......................] - ETA: 1:05 - loss: 7.7876 - accuracy: 0.4921
<br /> 6624/25000 [======>.......................] - ETA: 1:05 - loss: 7.7893 - accuracy: 0.4920
<br /> 6656/25000 [======>.......................] - ETA: 1:05 - loss: 7.7841 - accuracy: 0.4923
<br /> 6688/25000 [=======>......................] - ETA: 1:05 - loss: 7.7813 - accuracy: 0.4925
<br /> 6720/25000 [=======>......................] - ETA: 1:05 - loss: 7.7830 - accuracy: 0.4924
<br /> 6752/25000 [=======>......................] - ETA: 1:05 - loss: 7.7802 - accuracy: 0.4926
<br /> 6784/25000 [=======>......................] - ETA: 1:05 - loss: 7.7751 - accuracy: 0.4929
<br /> 6816/25000 [=======>......................] - ETA: 1:05 - loss: 7.7791 - accuracy: 0.4927
<br /> 6848/25000 [=======>......................] - ETA: 1:04 - loss: 7.7674 - accuracy: 0.4934
<br /> 6880/25000 [=======>......................] - ETA: 1:04 - loss: 7.7781 - accuracy: 0.4927
<br /> 6912/25000 [=======>......................] - ETA: 1:04 - loss: 7.7753 - accuracy: 0.4929
<br /> 6944/25000 [=======>......................] - ETA: 1:04 - loss: 7.7814 - accuracy: 0.4925
<br /> 6976/25000 [=======>......................] - ETA: 1:04 - loss: 7.7809 - accuracy: 0.4925
<br /> 7008/25000 [=======>......................] - ETA: 1:04 - loss: 7.7782 - accuracy: 0.4927
<br /> 7040/25000 [=======>......................] - ETA: 1:04 - loss: 7.7755 - accuracy: 0.4929
<br /> 7072/25000 [=======>......................] - ETA: 1:04 - loss: 7.7772 - accuracy: 0.4928
<br /> 7104/25000 [=======>......................] - ETA: 1:04 - loss: 7.7767 - accuracy: 0.4928
<br /> 7136/25000 [=======>......................] - ETA: 1:03 - loss: 7.7805 - accuracy: 0.4926
<br /> 7168/25000 [=======>......................] - ETA: 1:03 - loss: 7.7821 - accuracy: 0.4925
<br /> 7200/25000 [=======>......................] - ETA: 1:03 - loss: 7.7795 - accuracy: 0.4926
<br /> 7232/25000 [=======>......................] - ETA: 1:03 - loss: 7.7769 - accuracy: 0.4928
<br /> 7264/25000 [=======>......................] - ETA: 1:03 - loss: 7.7743 - accuracy: 0.4930
<br /> 7296/25000 [=======>......................] - ETA: 1:03 - loss: 7.7717 - accuracy: 0.4931
<br /> 7328/25000 [=======>......................] - ETA: 1:03 - loss: 7.7754 - accuracy: 0.4929
<br /> 7360/25000 [=======>......................] - ETA: 1:03 - loss: 7.7750 - accuracy: 0.4929
<br /> 7392/25000 [=======>......................] - ETA: 1:02 - loss: 7.7703 - accuracy: 0.4932
<br /> 7424/25000 [=======>......................] - ETA: 1:02 - loss: 7.7678 - accuracy: 0.4934
<br /> 7456/25000 [=======>......................] - ETA: 1:02 - loss: 7.7653 - accuracy: 0.4936
<br /> 7488/25000 [=======>......................] - ETA: 1:02 - loss: 7.7629 - accuracy: 0.4937
<br /> 7520/25000 [========>.....................] - ETA: 1:02 - loss: 7.7706 - accuracy: 0.4932
<br /> 7552/25000 [========>.....................] - ETA: 1:02 - loss: 7.7580 - accuracy: 0.4940
<br /> 7584/25000 [========>.....................] - ETA: 1:02 - loss: 7.7536 - accuracy: 0.4943
<br /> 7616/25000 [========>.....................] - ETA: 1:02 - loss: 7.7552 - accuracy: 0.4942
<br /> 7648/25000 [========>.....................] - ETA: 1:02 - loss: 7.7568 - accuracy: 0.4941
<br /> 7680/25000 [========>.....................] - ETA: 1:01 - loss: 7.7585 - accuracy: 0.4940
<br /> 7712/25000 [========>.....................] - ETA: 1:01 - loss: 7.7581 - accuracy: 0.4940
<br /> 7744/25000 [========>.....................] - ETA: 1:01 - loss: 7.7518 - accuracy: 0.4944
<br /> 7776/25000 [========>.....................] - ETA: 1:01 - loss: 7.7514 - accuracy: 0.4945
<br /> 7808/25000 [========>.....................] - ETA: 1:01 - loss: 7.7491 - accuracy: 0.4946
<br /> 7840/25000 [========>.....................] - ETA: 1:01 - loss: 7.7507 - accuracy: 0.4945
<br /> 7872/25000 [========>.....................] - ETA: 1:01 - loss: 7.7445 - accuracy: 0.4949
<br /> 7904/25000 [========>.....................] - ETA: 1:01 - loss: 7.7423 - accuracy: 0.4951
<br /> 7936/25000 [========>.....................] - ETA: 1:00 - loss: 7.7420 - accuracy: 0.4951
<br /> 7968/25000 [========>.....................] - ETA: 1:00 - loss: 7.7436 - accuracy: 0.4950
<br /> 8000/25000 [========>.....................] - ETA: 1:00 - loss: 7.7490 - accuracy: 0.4946
<br /> 8032/25000 [========>.....................] - ETA: 1:00 - loss: 7.7430 - accuracy: 0.4950
<br /> 8064/25000 [========>.....................] - ETA: 1:00 - loss: 7.7484 - accuracy: 0.4947
<br /> 8096/25000 [========>.....................] - ETA: 1:00 - loss: 7.7500 - accuracy: 0.4946
<br /> 8128/25000 [========>.....................] - ETA: 1:00 - loss: 7.7402 - accuracy: 0.4952
<br /> 8160/25000 [========>.....................] - ETA: 1:00 - loss: 7.7493 - accuracy: 0.4946
<br /> 8192/25000 [========>.....................] - ETA: 1:00 - loss: 7.7527 - accuracy: 0.4944
<br /> 8224/25000 [========>.....................] - ETA: 59s - loss: 7.7505 - accuracy: 0.4945 
<br /> 8256/25000 [========>.....................] - ETA: 59s - loss: 7.7465 - accuracy: 0.4948
<br /> 8288/25000 [========>.....................] - ETA: 59s - loss: 7.7462 - accuracy: 0.4948
<br /> 8320/25000 [========>.....................] - ETA: 59s - loss: 7.7477 - accuracy: 0.4947
<br /> 8352/25000 [=========>....................] - ETA: 59s - loss: 7.7456 - accuracy: 0.4949
<br /> 8384/25000 [=========>....................] - ETA: 59s - loss: 7.7453 - accuracy: 0.4949
<br /> 8416/25000 [=========>....................] - ETA: 59s - loss: 7.7450 - accuracy: 0.4949
<br /> 8448/25000 [=========>....................] - ETA: 59s - loss: 7.7356 - accuracy: 0.4955
<br /> 8480/25000 [=========>....................] - ETA: 58s - loss: 7.7353 - accuracy: 0.4955
<br /> 8512/25000 [=========>....................] - ETA: 58s - loss: 7.7333 - accuracy: 0.4957
<br /> 8544/25000 [=========>....................] - ETA: 58s - loss: 7.7223 - accuracy: 0.4964
<br /> 8576/25000 [=========>....................] - ETA: 58s - loss: 7.7220 - accuracy: 0.4964
<br /> 8608/25000 [=========>....................] - ETA: 58s - loss: 7.7254 - accuracy: 0.4962
<br /> 8640/25000 [=========>....................] - ETA: 58s - loss: 7.7199 - accuracy: 0.4965
<br /> 8672/25000 [=========>....................] - ETA: 58s - loss: 7.7038 - accuracy: 0.4976
<br /> 8704/25000 [=========>....................] - ETA: 58s - loss: 7.6983 - accuracy: 0.4979
<br /> 8736/25000 [=========>....................] - ETA: 58s - loss: 7.6912 - accuracy: 0.4984
<br /> 8768/25000 [=========>....................] - ETA: 57s - loss: 7.6859 - accuracy: 0.4987
<br /> 8800/25000 [=========>....................] - ETA: 57s - loss: 7.6858 - accuracy: 0.4988
<br /> 8832/25000 [=========>....................] - ETA: 57s - loss: 7.6788 - accuracy: 0.4992
<br /> 8864/25000 [=========>....................] - ETA: 57s - loss: 7.6735 - accuracy: 0.4995
<br /> 8896/25000 [=========>....................] - ETA: 57s - loss: 7.6735 - accuracy: 0.4996
<br /> 8928/25000 [=========>....................] - ETA: 57s - loss: 7.6701 - accuracy: 0.4998
<br /> 8960/25000 [=========>....................] - ETA: 57s - loss: 7.6735 - accuracy: 0.4996
<br /> 8992/25000 [=========>....................] - ETA: 57s - loss: 7.6786 - accuracy: 0.4992
<br /> 9024/25000 [=========>....................] - ETA: 57s - loss: 7.6819 - accuracy: 0.4990
<br /> 9056/25000 [=========>....................] - ETA: 56s - loss: 7.6802 - accuracy: 0.4991
<br /> 9088/25000 [=========>....................] - ETA: 56s - loss: 7.6734 - accuracy: 0.4996
<br /> 9120/25000 [=========>....................] - ETA: 56s - loss: 7.6717 - accuracy: 0.4997
<br /> 9152/25000 [=========>....................] - ETA: 56s - loss: 7.6800 - accuracy: 0.4991
<br /> 9184/25000 [==========>...................] - ETA: 56s - loss: 7.6800 - accuracy: 0.4991
<br /> 9216/25000 [==========>...................] - ETA: 56s - loss: 7.6783 - accuracy: 0.4992
<br /> 9248/25000 [==========>...................] - ETA: 56s - loss: 7.6815 - accuracy: 0.4990
<br /> 9280/25000 [==========>...................] - ETA: 56s - loss: 7.6848 - accuracy: 0.4988
<br /> 9312/25000 [==========>...................] - ETA: 55s - loss: 7.6798 - accuracy: 0.4991
<br /> 9344/25000 [==========>...................] - ETA: 55s - loss: 7.6847 - accuracy: 0.4988
<br /> 9376/25000 [==========>...................] - ETA: 55s - loss: 7.6846 - accuracy: 0.4988
<br /> 9408/25000 [==========>...................] - ETA: 55s - loss: 7.6780 - accuracy: 0.4993
<br /> 9440/25000 [==========>...................] - ETA: 55s - loss: 7.6780 - accuracy: 0.4993
<br /> 9472/25000 [==========>...................] - ETA: 55s - loss: 7.6828 - accuracy: 0.4989
<br /> 9504/25000 [==========>...................] - ETA: 55s - loss: 7.6892 - accuracy: 0.4985
<br /> 9536/25000 [==========>...................] - ETA: 55s - loss: 7.6875 - accuracy: 0.4986
<br /> 9568/25000 [==========>...................] - ETA: 55s - loss: 7.6891 - accuracy: 0.4985
<br /> 9600/25000 [==========>...................] - ETA: 54s - loss: 7.7018 - accuracy: 0.4977
<br /> 9632/25000 [==========>...................] - ETA: 54s - loss: 7.7016 - accuracy: 0.4977
<br /> 9664/25000 [==========>...................] - ETA: 54s - loss: 7.6984 - accuracy: 0.4979
<br /> 9696/25000 [==========>...................] - ETA: 54s - loss: 7.6982 - accuracy: 0.4979
<br /> 9728/25000 [==========>...................] - ETA: 54s - loss: 7.6981 - accuracy: 0.4979
<br /> 9760/25000 [==========>...................] - ETA: 54s - loss: 7.6996 - accuracy: 0.4978
<br /> 9792/25000 [==========>...................] - ETA: 54s - loss: 7.6964 - accuracy: 0.4981
<br /> 9824/25000 [==========>...................] - ETA: 54s - loss: 7.6932 - accuracy: 0.4983
<br /> 9856/25000 [==========>...................] - ETA: 53s - loss: 7.6946 - accuracy: 0.4982
<br /> 9888/25000 [==========>...................] - ETA: 53s - loss: 7.6883 - accuracy: 0.4986
<br /> 9920/25000 [==========>...................] - ETA: 53s - loss: 7.6883 - accuracy: 0.4986
<br /> 9952/25000 [==========>...................] - ETA: 53s - loss: 7.6974 - accuracy: 0.4980
<br /> 9984/25000 [==========>...................] - ETA: 53s - loss: 7.6973 - accuracy: 0.4980
<br />10016/25000 [===========>..................] - ETA: 53s - loss: 7.7018 - accuracy: 0.4977
<br />10048/25000 [===========>..................] - ETA: 53s - loss: 7.7017 - accuracy: 0.4977
<br />10080/25000 [===========>..................] - ETA: 53s - loss: 7.6970 - accuracy: 0.4980
<br />10112/25000 [===========>..................] - ETA: 53s - loss: 7.7030 - accuracy: 0.4976
<br />10144/25000 [===========>..................] - ETA: 52s - loss: 7.7029 - accuracy: 0.4976
<br />10176/25000 [===========>..................] - ETA: 52s - loss: 7.6998 - accuracy: 0.4978
<br />10208/25000 [===========>..................] - ETA: 52s - loss: 7.7027 - accuracy: 0.4976
<br />10240/25000 [===========>..................] - ETA: 52s - loss: 7.7056 - accuracy: 0.4975
<br />10272/25000 [===========>..................] - ETA: 52s - loss: 7.6950 - accuracy: 0.4982
<br />10304/25000 [===========>..................] - ETA: 52s - loss: 7.7023 - accuracy: 0.4977
<br />10336/25000 [===========>..................] - ETA: 52s - loss: 7.6993 - accuracy: 0.4979
<br />10368/25000 [===========>..................] - ETA: 52s - loss: 7.6962 - accuracy: 0.4981
<br />10400/25000 [===========>..................] - ETA: 51s - loss: 7.6961 - accuracy: 0.4981
<br />10432/25000 [===========>..................] - ETA: 51s - loss: 7.6975 - accuracy: 0.4980
<br />10464/25000 [===========>..................] - ETA: 51s - loss: 7.7033 - accuracy: 0.4976
<br />10496/25000 [===========>..................] - ETA: 51s - loss: 7.6973 - accuracy: 0.4980
<br />10528/25000 [===========>..................] - ETA: 51s - loss: 7.6972 - accuracy: 0.4980
<br />10560/25000 [===========>..................] - ETA: 51s - loss: 7.6971 - accuracy: 0.4980
<br />10592/25000 [===========>..................] - ETA: 51s - loss: 7.6999 - accuracy: 0.4978
<br />10624/25000 [===========>..................] - ETA: 51s - loss: 7.7056 - accuracy: 0.4975
<br />10656/25000 [===========>..................] - ETA: 51s - loss: 7.7040 - accuracy: 0.4976
<br />10688/25000 [===========>..................] - ETA: 50s - loss: 7.6982 - accuracy: 0.4979
<br />10720/25000 [===========>..................] - ETA: 50s - loss: 7.6981 - accuracy: 0.4979
<br />10752/25000 [===========>..................] - ETA: 50s - loss: 7.7051 - accuracy: 0.4975
<br />10784/25000 [===========>..................] - ETA: 50s - loss: 7.7007 - accuracy: 0.4978
<br />10816/25000 [===========>..................] - ETA: 50s - loss: 7.7049 - accuracy: 0.4975
<br />10848/25000 [============>.................] - ETA: 50s - loss: 7.7005 - accuracy: 0.4978
<br />10880/25000 [============>.................] - ETA: 50s - loss: 7.7019 - accuracy: 0.4977
<br />10912/25000 [============>.................] - ETA: 50s - loss: 7.7074 - accuracy: 0.4973
<br />10944/25000 [============>.................] - ETA: 50s - loss: 7.7115 - accuracy: 0.4971
<br />10976/25000 [============>.................] - ETA: 49s - loss: 7.7057 - accuracy: 0.4974
<br />11008/25000 [============>.................] - ETA: 49s - loss: 7.7000 - accuracy: 0.4978
<br />11040/25000 [============>.................] - ETA: 49s - loss: 7.6986 - accuracy: 0.4979
<br />11072/25000 [============>.................] - ETA: 49s - loss: 7.6999 - accuracy: 0.4978
<br />11104/25000 [============>.................] - ETA: 49s - loss: 7.6984 - accuracy: 0.4979
<br />11136/25000 [============>.................] - ETA: 49s - loss: 7.6983 - accuracy: 0.4979
<br />11168/25000 [============>.................] - ETA: 49s - loss: 7.6927 - accuracy: 0.4983
<br />11200/25000 [============>.................] - ETA: 49s - loss: 7.6926 - accuracy: 0.4983
<br />11232/25000 [============>.................] - ETA: 48s - loss: 7.6980 - accuracy: 0.4980
<br />11264/25000 [============>.................] - ETA: 48s - loss: 7.6979 - accuracy: 0.4980
<br />11296/25000 [============>.................] - ETA: 48s - loss: 7.7006 - accuracy: 0.4978
<br />11328/25000 [============>.................] - ETA: 48s - loss: 7.6937 - accuracy: 0.4982
<br />11360/25000 [============>.................] - ETA: 48s - loss: 7.6990 - accuracy: 0.4979
<br />11392/25000 [============>.................] - ETA: 48s - loss: 7.6949 - accuracy: 0.4982
<br />11424/25000 [============>.................] - ETA: 48s - loss: 7.6894 - accuracy: 0.4985
<br />11456/25000 [============>.................] - ETA: 48s - loss: 7.6907 - accuracy: 0.4984
<br />11488/25000 [============>.................] - ETA: 48s - loss: 7.6946 - accuracy: 0.4982
<br />11520/25000 [============>.................] - ETA: 47s - loss: 7.6999 - accuracy: 0.4978
<br />11552/25000 [============>.................] - ETA: 47s - loss: 7.6971 - accuracy: 0.4980
<br />11584/25000 [============>.................] - ETA: 47s - loss: 7.6918 - accuracy: 0.4984
<br />11616/25000 [============>.................] - ETA: 47s - loss: 7.6943 - accuracy: 0.4982
<br />11648/25000 [============>.................] - ETA: 47s - loss: 7.6943 - accuracy: 0.4982
<br />11680/25000 [=============>................] - ETA: 47s - loss: 7.6876 - accuracy: 0.4986
<br />11712/25000 [=============>................] - ETA: 47s - loss: 7.6928 - accuracy: 0.4983
<br />11744/25000 [=============>................] - ETA: 47s - loss: 7.6927 - accuracy: 0.4983
<br />11776/25000 [=============>................] - ETA: 46s - loss: 7.6862 - accuracy: 0.4987
<br />11808/25000 [=============>................] - ETA: 46s - loss: 7.6874 - accuracy: 0.4986
<br />11840/25000 [=============>................] - ETA: 46s - loss: 7.6860 - accuracy: 0.4987
<br />11872/25000 [=============>................] - ETA: 46s - loss: 7.6808 - accuracy: 0.4991
<br />11904/25000 [=============>................] - ETA: 46s - loss: 7.6769 - accuracy: 0.4993
<br />11936/25000 [=============>................] - ETA: 46s - loss: 7.6833 - accuracy: 0.4989
<br />11968/25000 [=============>................] - ETA: 46s - loss: 7.6871 - accuracy: 0.4987
<br />12000/25000 [=============>................] - ETA: 46s - loss: 7.6807 - accuracy: 0.4991
<br />12032/25000 [=============>................] - ETA: 46s - loss: 7.6794 - accuracy: 0.4992
<br />12064/25000 [=============>................] - ETA: 45s - loss: 7.6755 - accuracy: 0.4994
<br />12096/25000 [=============>................] - ETA: 45s - loss: 7.6704 - accuracy: 0.4998
<br />12128/25000 [=============>................] - ETA: 45s - loss: 7.6641 - accuracy: 0.5002
<br />12160/25000 [=============>................] - ETA: 45s - loss: 7.6666 - accuracy: 0.5000
<br />12192/25000 [=============>................] - ETA: 45s - loss: 7.6691 - accuracy: 0.4998
<br />12224/25000 [=============>................] - ETA: 45s - loss: 7.6654 - accuracy: 0.5001
<br />12256/25000 [=============>................] - ETA: 45s - loss: 7.6716 - accuracy: 0.4997
<br />12288/25000 [=============>................] - ETA: 45s - loss: 7.6766 - accuracy: 0.4993
<br />12320/25000 [=============>................] - ETA: 45s - loss: 7.6778 - accuracy: 0.4993
<br />12352/25000 [=============>................] - ETA: 44s - loss: 7.6766 - accuracy: 0.4994
<br />12384/25000 [=============>................] - ETA: 44s - loss: 7.6740 - accuracy: 0.4995
<br />12416/25000 [=============>................] - ETA: 44s - loss: 7.6740 - accuracy: 0.4995
<br />12448/25000 [=============>................] - ETA: 44s - loss: 7.6715 - accuracy: 0.4997
<br />12480/25000 [=============>................] - ETA: 44s - loss: 7.6678 - accuracy: 0.4999
<br />12512/25000 [==============>...............] - ETA: 44s - loss: 7.6691 - accuracy: 0.4998
<br />12544/25000 [==============>...............] - ETA: 44s - loss: 7.6691 - accuracy: 0.4998
<br />12576/25000 [==============>...............] - ETA: 44s - loss: 7.6703 - accuracy: 0.4998
<br />12608/25000 [==============>...............] - ETA: 44s - loss: 7.6691 - accuracy: 0.4998
<br />12640/25000 [==============>...............] - ETA: 43s - loss: 7.6727 - accuracy: 0.4996
<br />12672/25000 [==============>...............] - ETA: 43s - loss: 7.6763 - accuracy: 0.4994
<br />12704/25000 [==============>...............] - ETA: 43s - loss: 7.6702 - accuracy: 0.4998
<br />12736/25000 [==============>...............] - ETA: 43s - loss: 7.6714 - accuracy: 0.4997
<br />12768/25000 [==============>...............] - ETA: 43s - loss: 7.6702 - accuracy: 0.4998
<br />12800/25000 [==============>...............] - ETA: 43s - loss: 7.6762 - accuracy: 0.4994
<br />12832/25000 [==============>...............] - ETA: 43s - loss: 7.6845 - accuracy: 0.4988
<br />12864/25000 [==============>...............] - ETA: 43s - loss: 7.6845 - accuracy: 0.4988
<br />12896/25000 [==============>...............] - ETA: 43s - loss: 7.6880 - accuracy: 0.4986
<br />12928/25000 [==============>...............] - ETA: 42s - loss: 7.6880 - accuracy: 0.4986
<br />12960/25000 [==============>...............] - ETA: 42s - loss: 7.6891 - accuracy: 0.4985
<br />12992/25000 [==============>...............] - ETA: 42s - loss: 7.6890 - accuracy: 0.4985
<br />13024/25000 [==============>...............] - ETA: 42s - loss: 7.6902 - accuracy: 0.4985
<br />13056/25000 [==============>...............] - ETA: 42s - loss: 7.6960 - accuracy: 0.4981
<br />13088/25000 [==============>...............] - ETA: 42s - loss: 7.6936 - accuracy: 0.4982
<br />13120/25000 [==============>...............] - ETA: 42s - loss: 7.6935 - accuracy: 0.4982
<br />13152/25000 [==============>...............] - ETA: 42s - loss: 7.6934 - accuracy: 0.4983
<br />13184/25000 [==============>...............] - ETA: 41s - loss: 7.6934 - accuracy: 0.4983
<br />13216/25000 [==============>...............] - ETA: 41s - loss: 7.6933 - accuracy: 0.4983
<br />13248/25000 [==============>...............] - ETA: 41s - loss: 7.6909 - accuracy: 0.4984
<br />13280/25000 [==============>...............] - ETA: 41s - loss: 7.6862 - accuracy: 0.4987
<br />13312/25000 [==============>...............] - ETA: 41s - loss: 7.6943 - accuracy: 0.4982
<br />13344/25000 [===============>..............] - ETA: 41s - loss: 7.6862 - accuracy: 0.4987
<br />13376/25000 [===============>..............] - ETA: 41s - loss: 7.6792 - accuracy: 0.4992
<br />13408/25000 [===============>..............] - ETA: 41s - loss: 7.6849 - accuracy: 0.4988
<br />13440/25000 [===============>..............] - ETA: 41s - loss: 7.6872 - accuracy: 0.4987
<br />13472/25000 [===============>..............] - ETA: 40s - loss: 7.6860 - accuracy: 0.4987
<br />13504/25000 [===============>..............] - ETA: 40s - loss: 7.6848 - accuracy: 0.4988
<br />13536/25000 [===============>..............] - ETA: 40s - loss: 7.6870 - accuracy: 0.4987
<br />13568/25000 [===============>..............] - ETA: 40s - loss: 7.6881 - accuracy: 0.4986
<br />13600/25000 [===============>..............] - ETA: 40s - loss: 7.6847 - accuracy: 0.4988
<br />13632/25000 [===============>..............] - ETA: 40s - loss: 7.6846 - accuracy: 0.4988
<br />13664/25000 [===============>..............] - ETA: 40s - loss: 7.6790 - accuracy: 0.4992
<br />13696/25000 [===============>..............] - ETA: 40s - loss: 7.6801 - accuracy: 0.4991
<br />13728/25000 [===============>..............] - ETA: 40s - loss: 7.6834 - accuracy: 0.4989
<br />13760/25000 [===============>..............] - ETA: 39s - loss: 7.6822 - accuracy: 0.4990
<br />13792/25000 [===============>..............] - ETA: 39s - loss: 7.6855 - accuracy: 0.4988
<br />13824/25000 [===============>..............] - ETA: 39s - loss: 7.6866 - accuracy: 0.4987
<br />13856/25000 [===============>..............] - ETA: 39s - loss: 7.6865 - accuracy: 0.4987
<br />13888/25000 [===============>..............] - ETA: 39s - loss: 7.6821 - accuracy: 0.4990
<br />13920/25000 [===============>..............] - ETA: 39s - loss: 7.6842 - accuracy: 0.4989
<br />13952/25000 [===============>..............] - ETA: 39s - loss: 7.6853 - accuracy: 0.4988
<br />13984/25000 [===============>..............] - ETA: 39s - loss: 7.6885 - accuracy: 0.4986
<br />14016/25000 [===============>..............] - ETA: 38s - loss: 7.6896 - accuracy: 0.4985
<br />14048/25000 [===============>..............] - ETA: 38s - loss: 7.6906 - accuracy: 0.4984
<br />14080/25000 [===============>..............] - ETA: 38s - loss: 7.6928 - accuracy: 0.4983
<br />14112/25000 [===============>..............] - ETA: 38s - loss: 7.6960 - accuracy: 0.4981
<br />14144/25000 [===============>..............] - ETA: 38s - loss: 7.6916 - accuracy: 0.4984
<br />14176/25000 [================>.............] - ETA: 38s - loss: 7.6937 - accuracy: 0.4982
<br />14208/25000 [================>.............] - ETA: 38s - loss: 7.6904 - accuracy: 0.4985
<br />14240/25000 [================>.............] - ETA: 38s - loss: 7.6946 - accuracy: 0.4982
<br />14272/25000 [================>.............] - ETA: 38s - loss: 7.6956 - accuracy: 0.4981
<br />14304/25000 [================>.............] - ETA: 37s - loss: 7.6945 - accuracy: 0.4982
<br />14336/25000 [================>.............] - ETA: 37s - loss: 7.6966 - accuracy: 0.4980
<br />14368/25000 [================>.............] - ETA: 37s - loss: 7.6986 - accuracy: 0.4979
<br />14400/25000 [================>.............] - ETA: 37s - loss: 7.7028 - accuracy: 0.4976
<br />14432/25000 [================>.............] - ETA: 37s - loss: 7.6953 - accuracy: 0.4981
<br />14464/25000 [================>.............] - ETA: 37s - loss: 7.6974 - accuracy: 0.4980
<br />14496/25000 [================>.............] - ETA: 37s - loss: 7.6973 - accuracy: 0.4980
<br />14528/25000 [================>.............] - ETA: 37s - loss: 7.6930 - accuracy: 0.4983
<br />14560/25000 [================>.............] - ETA: 37s - loss: 7.6961 - accuracy: 0.4981
<br />14592/25000 [================>.............] - ETA: 36s - loss: 7.6981 - accuracy: 0.4979
<br />14624/25000 [================>.............] - ETA: 36s - loss: 7.7012 - accuracy: 0.4977
<br />14656/25000 [================>.............] - ETA: 36s - loss: 7.7001 - accuracy: 0.4978
<br />14688/25000 [================>.............] - ETA: 36s - loss: 7.6979 - accuracy: 0.4980
<br />14720/25000 [================>.............] - ETA: 36s - loss: 7.6979 - accuracy: 0.4980
<br />14752/25000 [================>.............] - ETA: 36s - loss: 7.7009 - accuracy: 0.4978
<br />14784/25000 [================>.............] - ETA: 36s - loss: 7.7019 - accuracy: 0.4977
<br />14816/25000 [================>.............] - ETA: 36s - loss: 7.6997 - accuracy: 0.4978
<br />14848/25000 [================>.............] - ETA: 35s - loss: 7.6966 - accuracy: 0.4980
<br />14880/25000 [================>.............] - ETA: 35s - loss: 7.6996 - accuracy: 0.4978
<br />14912/25000 [================>.............] - ETA: 35s - loss: 7.6985 - accuracy: 0.4979
<br />14944/25000 [================>.............] - ETA: 35s - loss: 7.6974 - accuracy: 0.4980
<br />14976/25000 [================>.............] - ETA: 35s - loss: 7.7025 - accuracy: 0.4977
<br />15008/25000 [=================>............] - ETA: 35s - loss: 7.7034 - accuracy: 0.4976
<br />15040/25000 [=================>............] - ETA: 35s - loss: 7.7033 - accuracy: 0.4976
<br />15072/25000 [=================>............] - ETA: 35s - loss: 7.7043 - accuracy: 0.4975
<br />15104/25000 [=================>............] - ETA: 35s - loss: 7.7052 - accuracy: 0.4975
<br />15136/25000 [=================>............] - ETA: 34s - loss: 7.7061 - accuracy: 0.4974
<br />15168/25000 [=================>............] - ETA: 34s - loss: 7.7060 - accuracy: 0.4974
<br />15200/25000 [=================>............] - ETA: 34s - loss: 7.7110 - accuracy: 0.4971
<br />15232/25000 [=================>............] - ETA: 34s - loss: 7.7119 - accuracy: 0.4970
<br />15264/25000 [=================>............] - ETA: 34s - loss: 7.7088 - accuracy: 0.4972
<br />15296/25000 [=================>............] - ETA: 34s - loss: 7.7127 - accuracy: 0.4970
<br />15328/25000 [=================>............] - ETA: 34s - loss: 7.7116 - accuracy: 0.4971
<br />15360/25000 [=================>............] - ETA: 34s - loss: 7.7085 - accuracy: 0.4973
<br />15392/25000 [=================>............] - ETA: 34s - loss: 7.7055 - accuracy: 0.4975
<br />15424/25000 [=================>............] - ETA: 33s - loss: 7.7014 - accuracy: 0.4977
<br />15456/25000 [=================>............] - ETA: 33s - loss: 7.7033 - accuracy: 0.4976
<br />15488/25000 [=================>............] - ETA: 33s - loss: 7.7032 - accuracy: 0.4976
<br />15520/25000 [=================>............] - ETA: 33s - loss: 7.7042 - accuracy: 0.4976
<br />15552/25000 [=================>............] - ETA: 33s - loss: 7.7021 - accuracy: 0.4977
<br />15584/25000 [=================>............] - ETA: 33s - loss: 7.7040 - accuracy: 0.4976
<br />15616/25000 [=================>............] - ETA: 33s - loss: 7.7000 - accuracy: 0.4978
<br />15648/25000 [=================>............] - ETA: 33s - loss: 7.6990 - accuracy: 0.4979
<br />15680/25000 [=================>............] - ETA: 33s - loss: 7.6999 - accuracy: 0.4978
<br />15712/25000 [=================>............] - ETA: 32s - loss: 7.6978 - accuracy: 0.4980
<br />15744/25000 [=================>............] - ETA: 32s - loss: 7.6939 - accuracy: 0.4982
<br />15776/25000 [=================>............] - ETA: 32s - loss: 7.6870 - accuracy: 0.4987
<br />15808/25000 [=================>............] - ETA: 32s - loss: 7.6880 - accuracy: 0.4986
<br />15840/25000 [==================>...........] - ETA: 32s - loss: 7.6899 - accuracy: 0.4985
<br />15872/25000 [==================>...........] - ETA: 32s - loss: 7.6869 - accuracy: 0.4987
<br />15904/25000 [==================>...........] - ETA: 32s - loss: 7.6898 - accuracy: 0.4985
<br />15936/25000 [==================>...........] - ETA: 32s - loss: 7.6907 - accuracy: 0.4984
<br />15968/25000 [==================>...........] - ETA: 32s - loss: 7.6916 - accuracy: 0.4984
<br />16000/25000 [==================>...........] - ETA: 31s - loss: 7.6906 - accuracy: 0.4984
<br />16032/25000 [==================>...........] - ETA: 31s - loss: 7.6924 - accuracy: 0.4983
<br />16064/25000 [==================>...........] - ETA: 31s - loss: 7.6933 - accuracy: 0.4983
<br />16096/25000 [==================>...........] - ETA: 31s - loss: 7.6933 - accuracy: 0.4983
<br />16128/25000 [==================>...........] - ETA: 31s - loss: 7.6942 - accuracy: 0.4982
<br />16160/25000 [==================>...........] - ETA: 31s - loss: 7.6932 - accuracy: 0.4983
<br />16192/25000 [==================>...........] - ETA: 31s - loss: 7.6960 - accuracy: 0.4981
<br />16224/25000 [==================>...........] - ETA: 31s - loss: 7.6969 - accuracy: 0.4980
<br />16256/25000 [==================>...........] - ETA: 30s - loss: 7.6977 - accuracy: 0.4980
<br />16288/25000 [==================>...........] - ETA: 30s - loss: 7.7043 - accuracy: 0.4975
<br />16320/25000 [==================>...........] - ETA: 30s - loss: 7.7014 - accuracy: 0.4977
<br />16352/25000 [==================>...........] - ETA: 30s - loss: 7.7069 - accuracy: 0.4974
<br />16384/25000 [==================>...........] - ETA: 30s - loss: 7.7069 - accuracy: 0.4974
<br />16416/25000 [==================>...........] - ETA: 30s - loss: 7.7077 - accuracy: 0.4973
<br />16448/25000 [==================>...........] - ETA: 30s - loss: 7.7104 - accuracy: 0.4971
<br />16480/25000 [==================>...........] - ETA: 30s - loss: 7.7131 - accuracy: 0.4970
<br />16512/25000 [==================>...........] - ETA: 30s - loss: 7.7093 - accuracy: 0.4972
<br />16544/25000 [==================>...........] - ETA: 29s - loss: 7.7083 - accuracy: 0.4973
<br />16576/25000 [==================>...........] - ETA: 29s - loss: 7.7082 - accuracy: 0.4973
<br />16608/25000 [==================>...........] - ETA: 29s - loss: 7.7045 - accuracy: 0.4975
<br />16640/25000 [==================>...........] - ETA: 29s - loss: 7.7090 - accuracy: 0.4972
<br />16672/25000 [===================>..........] - ETA: 29s - loss: 7.7154 - accuracy: 0.4968
<br />16704/25000 [===================>..........] - ETA: 29s - loss: 7.7125 - accuracy: 0.4970
<br />16736/25000 [===================>..........] - ETA: 29s - loss: 7.7115 - accuracy: 0.4971
<br />16768/25000 [===================>..........] - ETA: 29s - loss: 7.7133 - accuracy: 0.4970
<br />16800/25000 [===================>..........] - ETA: 29s - loss: 7.7132 - accuracy: 0.4970
<br />16832/25000 [===================>..........] - ETA: 28s - loss: 7.7149 - accuracy: 0.4969
<br />16864/25000 [===================>..........] - ETA: 28s - loss: 7.7121 - accuracy: 0.4970
<br />16896/25000 [===================>..........] - ETA: 28s - loss: 7.7120 - accuracy: 0.4970
<br />16928/25000 [===================>..........] - ETA: 28s - loss: 7.7137 - accuracy: 0.4969
<br />16960/25000 [===================>..........] - ETA: 28s - loss: 7.7145 - accuracy: 0.4969
<br />16992/25000 [===================>..........] - ETA: 28s - loss: 7.7190 - accuracy: 0.4966
<br />17024/25000 [===================>..........] - ETA: 28s - loss: 7.7117 - accuracy: 0.4971
<br />17056/25000 [===================>..........] - ETA: 28s - loss: 7.7134 - accuracy: 0.4970
<br />17088/25000 [===================>..........] - ETA: 28s - loss: 7.7133 - accuracy: 0.4970
<br />17120/25000 [===================>..........] - ETA: 27s - loss: 7.7078 - accuracy: 0.4973
<br />17152/25000 [===================>..........] - ETA: 27s - loss: 7.7086 - accuracy: 0.4973
<br />17184/25000 [===================>..........] - ETA: 27s - loss: 7.7139 - accuracy: 0.4969
<br />17216/25000 [===================>..........] - ETA: 27s - loss: 7.7183 - accuracy: 0.4966
<br />17248/25000 [===================>..........] - ETA: 27s - loss: 7.7146 - accuracy: 0.4969
<br />17280/25000 [===================>..........] - ETA: 27s - loss: 7.7163 - accuracy: 0.4968
<br />17312/25000 [===================>..........] - ETA: 27s - loss: 7.7144 - accuracy: 0.4969
<br />17344/25000 [===================>..........] - ETA: 27s - loss: 7.7117 - accuracy: 0.4971
<br />17376/25000 [===================>..........] - ETA: 27s - loss: 7.7090 - accuracy: 0.4972
<br />17408/25000 [===================>..........] - ETA: 26s - loss: 7.7107 - accuracy: 0.4971
<br />17440/25000 [===================>..........] - ETA: 26s - loss: 7.7132 - accuracy: 0.4970
<br />17472/25000 [===================>..........] - ETA: 26s - loss: 7.7149 - accuracy: 0.4969
<br />17504/25000 [====================>.........] - ETA: 26s - loss: 7.7113 - accuracy: 0.4971
<br />17536/25000 [====================>.........] - ETA: 26s - loss: 7.7112 - accuracy: 0.4971
<br />17568/25000 [====================>.........] - ETA: 26s - loss: 7.7111 - accuracy: 0.4971
<br />17600/25000 [====================>.........] - ETA: 26s - loss: 7.7163 - accuracy: 0.4968
<br />17632/25000 [====================>.........] - ETA: 26s - loss: 7.7144 - accuracy: 0.4969
<br />17664/25000 [====================>.........] - ETA: 26s - loss: 7.7152 - accuracy: 0.4968
<br />17696/25000 [====================>.........] - ETA: 25s - loss: 7.7160 - accuracy: 0.4968
<br />17728/25000 [====================>.........] - ETA: 25s - loss: 7.7107 - accuracy: 0.4971
<br />17760/25000 [====================>.........] - ETA: 25s - loss: 7.7115 - accuracy: 0.4971
<br />17792/25000 [====================>.........] - ETA: 25s - loss: 7.7088 - accuracy: 0.4972
<br />17824/25000 [====================>.........] - ETA: 25s - loss: 7.7096 - accuracy: 0.4972
<br />17856/25000 [====================>.........] - ETA: 25s - loss: 7.7061 - accuracy: 0.4974
<br />17888/25000 [====================>.........] - ETA: 25s - loss: 7.7069 - accuracy: 0.4974
<br />17920/25000 [====================>.........] - ETA: 25s - loss: 7.7085 - accuracy: 0.4973
<br />17952/25000 [====================>.........] - ETA: 24s - loss: 7.7076 - accuracy: 0.4973
<br />17984/25000 [====================>.........] - ETA: 24s - loss: 7.7058 - accuracy: 0.4974
<br />18016/25000 [====================>.........] - ETA: 24s - loss: 7.7075 - accuracy: 0.4973
<br />18048/25000 [====================>.........] - ETA: 24s - loss: 7.7091 - accuracy: 0.4972
<br />18080/25000 [====================>.........] - ETA: 24s - loss: 7.7065 - accuracy: 0.4974
<br />18112/25000 [====================>.........] - ETA: 24s - loss: 7.7073 - accuracy: 0.4973
<br />18144/25000 [====================>.........] - ETA: 24s - loss: 7.7114 - accuracy: 0.4971
<br />18176/25000 [====================>.........] - ETA: 24s - loss: 7.7096 - accuracy: 0.4972
<br />18208/25000 [====================>.........] - ETA: 24s - loss: 7.7087 - accuracy: 0.4973
<br />18240/25000 [====================>.........] - ETA: 23s - loss: 7.7112 - accuracy: 0.4971
<br />18272/25000 [====================>.........] - ETA: 23s - loss: 7.7094 - accuracy: 0.4972
<br />18304/25000 [====================>.........] - ETA: 23s - loss: 7.7102 - accuracy: 0.4972
<br />18336/25000 [=====================>........] - ETA: 23s - loss: 7.7126 - accuracy: 0.4970
<br />18368/25000 [=====================>........] - ETA: 23s - loss: 7.7134 - accuracy: 0.4970
<br />18400/25000 [=====================>........] - ETA: 23s - loss: 7.7100 - accuracy: 0.4972
<br />18432/25000 [=====================>........] - ETA: 23s - loss: 7.7107 - accuracy: 0.4971
<br />18464/25000 [=====================>........] - ETA: 23s - loss: 7.7081 - accuracy: 0.4973
<br />18496/25000 [=====================>........] - ETA: 23s - loss: 7.7081 - accuracy: 0.4973
<br />18528/25000 [=====================>........] - ETA: 22s - loss: 7.7080 - accuracy: 0.4973
<br />18560/25000 [=====================>........] - ETA: 22s - loss: 7.7096 - accuracy: 0.4972
<br />18592/25000 [=====================>........] - ETA: 22s - loss: 7.7120 - accuracy: 0.4970
<br />18624/25000 [=====================>........] - ETA: 22s - loss: 7.7111 - accuracy: 0.4971
<br />18656/25000 [=====================>........] - ETA: 22s - loss: 7.7143 - accuracy: 0.4969
<br />18688/25000 [=====================>........] - ETA: 22s - loss: 7.7134 - accuracy: 0.4969
<br />18720/25000 [=====================>........] - ETA: 22s - loss: 7.7092 - accuracy: 0.4972
<br />18752/25000 [=====================>........] - ETA: 22s - loss: 7.7091 - accuracy: 0.4972
<br />18784/25000 [=====================>........] - ETA: 22s - loss: 7.7107 - accuracy: 0.4971
<br />18816/25000 [=====================>........] - ETA: 21s - loss: 7.7098 - accuracy: 0.4972
<br />18848/25000 [=====================>........] - ETA: 21s - loss: 7.7081 - accuracy: 0.4973
<br />18880/25000 [=====================>........] - ETA: 21s - loss: 7.7072 - accuracy: 0.4974
<br />18912/25000 [=====================>........] - ETA: 21s - loss: 7.7120 - accuracy: 0.4970
<br />18944/25000 [=====================>........] - ETA: 21s - loss: 7.7136 - accuracy: 0.4969
<br />18976/25000 [=====================>........] - ETA: 21s - loss: 7.7127 - accuracy: 0.4970
<br />19008/25000 [=====================>........] - ETA: 21s - loss: 7.7102 - accuracy: 0.4972
<br />19040/25000 [=====================>........] - ETA: 21s - loss: 7.7117 - accuracy: 0.4971
<br />19072/25000 [=====================>........] - ETA: 20s - loss: 7.7100 - accuracy: 0.4972
<br />19104/25000 [=====================>........] - ETA: 20s - loss: 7.7124 - accuracy: 0.4970
<br />19136/25000 [=====================>........] - ETA: 20s - loss: 7.7099 - accuracy: 0.4972
<br />19168/25000 [======================>.......] - ETA: 20s - loss: 7.7098 - accuracy: 0.4972
<br />19200/25000 [======================>.......] - ETA: 20s - loss: 7.7137 - accuracy: 0.4969
<br />19232/25000 [======================>.......] - ETA: 20s - loss: 7.7113 - accuracy: 0.4971
<br />19264/25000 [======================>.......] - ETA: 20s - loss: 7.7120 - accuracy: 0.4970
<br />19296/25000 [======================>.......] - ETA: 20s - loss: 7.7135 - accuracy: 0.4969
<br />19328/25000 [======================>.......] - ETA: 20s - loss: 7.7095 - accuracy: 0.4972
<br />19360/25000 [======================>.......] - ETA: 19s - loss: 7.7062 - accuracy: 0.4974
<br />19392/25000 [======================>.......] - ETA: 19s - loss: 7.7054 - accuracy: 0.4975
<br />19424/25000 [======================>.......] - ETA: 19s - loss: 7.7029 - accuracy: 0.4976
<br />19456/25000 [======================>.......] - ETA: 19s - loss: 7.7013 - accuracy: 0.4977
<br />19488/25000 [======================>.......] - ETA: 19s - loss: 7.6997 - accuracy: 0.4978
<br />19520/25000 [======================>.......] - ETA: 19s - loss: 7.7028 - accuracy: 0.4976
<br />19552/25000 [======================>.......] - ETA: 19s - loss: 7.7050 - accuracy: 0.4975
<br />19584/25000 [======================>.......] - ETA: 19s - loss: 7.7050 - accuracy: 0.4975
<br />19616/25000 [======================>.......] - ETA: 19s - loss: 7.7049 - accuracy: 0.4975
<br />19648/25000 [======================>.......] - ETA: 18s - loss: 7.7017 - accuracy: 0.4977
<br />19680/25000 [======================>.......] - ETA: 18s - loss: 7.7032 - accuracy: 0.4976
<br />19712/25000 [======================>.......] - ETA: 18s - loss: 7.7001 - accuracy: 0.4978
<br />19744/25000 [======================>.......] - ETA: 18s - loss: 7.6992 - accuracy: 0.4979
<br />19776/25000 [======================>.......] - ETA: 18s - loss: 7.7023 - accuracy: 0.4977
<br />19808/25000 [======================>.......] - ETA: 18s - loss: 7.7038 - accuracy: 0.4976
<br />19840/25000 [======================>.......] - ETA: 18s - loss: 7.7045 - accuracy: 0.4975
<br />19872/25000 [======================>.......] - ETA: 18s - loss: 7.7060 - accuracy: 0.4974
<br />19904/25000 [======================>.......] - ETA: 18s - loss: 7.7051 - accuracy: 0.4975
<br />19936/25000 [======================>.......] - ETA: 17s - loss: 7.7035 - accuracy: 0.4976
<br />19968/25000 [======================>.......] - ETA: 17s - loss: 7.7027 - accuracy: 0.4976
<br />20000/25000 [=======================>......] - ETA: 17s - loss: 7.7042 - accuracy: 0.4976
<br />20032/25000 [=======================>......] - ETA: 17s - loss: 7.7018 - accuracy: 0.4977
<br />20064/25000 [=======================>......] - ETA: 17s - loss: 7.7025 - accuracy: 0.4977
<br />20096/25000 [=======================>......] - ETA: 17s - loss: 7.7032 - accuracy: 0.4976
<br />20128/25000 [=======================>......] - ETA: 17s - loss: 7.7001 - accuracy: 0.4978
<br />20160/25000 [=======================>......] - ETA: 17s - loss: 7.6963 - accuracy: 0.4981
<br />20192/25000 [=======================>......] - ETA: 17s - loss: 7.6962 - accuracy: 0.4981
<br />20224/25000 [=======================>......] - ETA: 16s - loss: 7.6977 - accuracy: 0.4980
<br />20256/25000 [=======================>......] - ETA: 16s - loss: 7.6984 - accuracy: 0.4979
<br />20288/25000 [=======================>......] - ETA: 16s - loss: 7.7006 - accuracy: 0.4978
<br />20320/25000 [=======================>......] - ETA: 16s - loss: 7.6983 - accuracy: 0.4979
<br />20352/25000 [=======================>......] - ETA: 16s - loss: 7.6983 - accuracy: 0.4979
<br />20384/25000 [=======================>......] - ETA: 16s - loss: 7.6982 - accuracy: 0.4979
<br />20416/25000 [=======================>......] - ETA: 16s - loss: 7.6959 - accuracy: 0.4981
<br />20448/25000 [=======================>......] - ETA: 16s - loss: 7.6981 - accuracy: 0.4979
<br />20480/25000 [=======================>......] - ETA: 15s - loss: 7.6966 - accuracy: 0.4980
<br />20512/25000 [=======================>......] - ETA: 15s - loss: 7.6928 - accuracy: 0.4983
<br />20544/25000 [=======================>......] - ETA: 15s - loss: 7.6920 - accuracy: 0.4983
<br />20576/25000 [=======================>......] - ETA: 15s - loss: 7.6942 - accuracy: 0.4982
<br />20608/25000 [=======================>......] - ETA: 15s - loss: 7.6919 - accuracy: 0.4984
<br />20640/25000 [=======================>......] - ETA: 15s - loss: 7.6926 - accuracy: 0.4983
<br />20672/25000 [=======================>......] - ETA: 15s - loss: 7.6911 - accuracy: 0.4984
<br />20704/25000 [=======================>......] - ETA: 15s - loss: 7.6911 - accuracy: 0.4984
<br />20736/25000 [=======================>......] - ETA: 15s - loss: 7.6888 - accuracy: 0.4986
<br />20768/25000 [=======================>......] - ETA: 14s - loss: 7.6880 - accuracy: 0.4986
<br />20800/25000 [=======================>......] - ETA: 14s - loss: 7.6917 - accuracy: 0.4984
<br />20832/25000 [=======================>......] - ETA: 14s - loss: 7.6931 - accuracy: 0.4983
<br />20864/25000 [========================>.....] - ETA: 14s - loss: 7.6909 - accuracy: 0.4984
<br />20896/25000 [========================>.....] - ETA: 14s - loss: 7.6864 - accuracy: 0.4987
<br />20928/25000 [========================>.....] - ETA: 14s - loss: 7.6820 - accuracy: 0.4990
<br />20960/25000 [========================>.....] - ETA: 14s - loss: 7.6805 - accuracy: 0.4991
<br />20992/25000 [========================>.....] - ETA: 14s - loss: 7.6783 - accuracy: 0.4992
<br />21024/25000 [========================>.....] - ETA: 14s - loss: 7.6776 - accuracy: 0.4993
<br />21056/25000 [========================>.....] - ETA: 13s - loss: 7.6783 - accuracy: 0.4992
<br />21088/25000 [========================>.....] - ETA: 13s - loss: 7.6812 - accuracy: 0.4991
<br />21120/25000 [========================>.....] - ETA: 13s - loss: 7.6811 - accuracy: 0.4991
<br />21152/25000 [========================>.....] - ETA: 13s - loss: 7.6782 - accuracy: 0.4992
<br />21184/25000 [========================>.....] - ETA: 13s - loss: 7.6796 - accuracy: 0.4992
<br />21216/25000 [========================>.....] - ETA: 13s - loss: 7.6804 - accuracy: 0.4991
<br />21248/25000 [========================>.....] - ETA: 13s - loss: 7.6803 - accuracy: 0.4991
<br />21280/25000 [========================>.....] - ETA: 13s - loss: 7.6810 - accuracy: 0.4991
<br />21312/25000 [========================>.....] - ETA: 13s - loss: 7.6810 - accuracy: 0.4991
<br />21344/25000 [========================>.....] - ETA: 12s - loss: 7.6817 - accuracy: 0.4990
<br />21376/25000 [========================>.....] - ETA: 12s - loss: 7.6817 - accuracy: 0.4990
<br />21408/25000 [========================>.....] - ETA: 12s - loss: 7.6831 - accuracy: 0.4989
<br />21440/25000 [========================>.....] - ETA: 12s - loss: 7.6838 - accuracy: 0.4989
<br />21472/25000 [========================>.....] - ETA: 12s - loss: 7.6852 - accuracy: 0.4988
<br />21504/25000 [========================>.....] - ETA: 12s - loss: 7.6887 - accuracy: 0.4986
<br />21536/25000 [========================>.....] - ETA: 12s - loss: 7.6887 - accuracy: 0.4986
<br />21568/25000 [========================>.....] - ETA: 12s - loss: 7.6908 - accuracy: 0.4984
<br />21600/25000 [========================>.....] - ETA: 12s - loss: 7.6915 - accuracy: 0.4984
<br />21632/25000 [========================>.....] - ETA: 11s - loss: 7.6943 - accuracy: 0.4982
<br />21664/25000 [========================>.....] - ETA: 11s - loss: 7.6978 - accuracy: 0.4980
<br />21696/25000 [=========================>....] - ETA: 11s - loss: 7.6998 - accuracy: 0.4978
<br />21728/25000 [=========================>....] - ETA: 11s - loss: 7.6984 - accuracy: 0.4979
<br />21760/25000 [=========================>....] - ETA: 11s - loss: 7.6983 - accuracy: 0.4979
<br />21792/25000 [=========================>....] - ETA: 11s - loss: 7.6997 - accuracy: 0.4978
<br />21824/25000 [=========================>....] - ETA: 11s - loss: 7.7010 - accuracy: 0.4978
<br />21856/25000 [=========================>....] - ETA: 11s - loss: 7.7003 - accuracy: 0.4978
<br />21888/25000 [=========================>....] - ETA: 11s - loss: 7.6981 - accuracy: 0.4979
<br />21920/25000 [=========================>....] - ETA: 10s - loss: 7.6981 - accuracy: 0.4979
<br />21952/25000 [=========================>....] - ETA: 10s - loss: 7.6974 - accuracy: 0.4980
<br />21984/25000 [=========================>....] - ETA: 10s - loss: 7.6987 - accuracy: 0.4979
<br />22016/25000 [=========================>....] - ETA: 10s - loss: 7.7021 - accuracy: 0.4977
<br />22048/25000 [=========================>....] - ETA: 10s - loss: 7.7014 - accuracy: 0.4977
<br />22080/25000 [=========================>....] - ETA: 10s - loss: 7.7020 - accuracy: 0.4977
<br />22112/25000 [=========================>....] - ETA: 10s - loss: 7.7006 - accuracy: 0.4978
<br />22144/25000 [=========================>....] - ETA: 10s - loss: 7.6999 - accuracy: 0.4978
<br />22176/25000 [=========================>....] - ETA: 9s - loss: 7.6977 - accuracy: 0.4980 
<br />22208/25000 [=========================>....] - ETA: 9s - loss: 7.6984 - accuracy: 0.4979
<br />22240/25000 [=========================>....] - ETA: 9s - loss: 7.6963 - accuracy: 0.4981
<br />22272/25000 [=========================>....] - ETA: 9s - loss: 7.6928 - accuracy: 0.4983
<br />22304/25000 [=========================>....] - ETA: 9s - loss: 7.6934 - accuracy: 0.4983
<br />22336/25000 [=========================>....] - ETA: 9s - loss: 7.6941 - accuracy: 0.4982
<br />22368/25000 [=========================>....] - ETA: 9s - loss: 7.6934 - accuracy: 0.4983
<br />22400/25000 [=========================>....] - ETA: 9s - loss: 7.6919 - accuracy: 0.4983
<br />22432/25000 [=========================>....] - ETA: 9s - loss: 7.6912 - accuracy: 0.4984
<br />22464/25000 [=========================>....] - ETA: 8s - loss: 7.6926 - accuracy: 0.4983
<br />22496/25000 [=========================>....] - ETA: 8s - loss: 7.6939 - accuracy: 0.4982
<br />22528/25000 [==========================>...] - ETA: 8s - loss: 7.6966 - accuracy: 0.4980
<br />22560/25000 [==========================>...] - ETA: 8s - loss: 7.6958 - accuracy: 0.4981
<br />22592/25000 [==========================>...] - ETA: 8s - loss: 7.6944 - accuracy: 0.4982
<br />22624/25000 [==========================>...] - ETA: 8s - loss: 7.6924 - accuracy: 0.4983
<br />22656/25000 [==========================>...] - ETA: 8s - loss: 7.6917 - accuracy: 0.4984
<br />22688/25000 [==========================>...] - ETA: 8s - loss: 7.6930 - accuracy: 0.4983
<br />22720/25000 [==========================>...] - ETA: 8s - loss: 7.6936 - accuracy: 0.4982
<br />22752/25000 [==========================>...] - ETA: 7s - loss: 7.6963 - accuracy: 0.4981
<br />22784/25000 [==========================>...] - ETA: 7s - loss: 7.6982 - accuracy: 0.4979
<br />22816/25000 [==========================>...] - ETA: 7s - loss: 7.6989 - accuracy: 0.4979
<br />22848/25000 [==========================>...] - ETA: 7s - loss: 7.7002 - accuracy: 0.4978
<br />22880/25000 [==========================>...] - ETA: 7s - loss: 7.6981 - accuracy: 0.4979
<br />22912/25000 [==========================>...] - ETA: 7s - loss: 7.7007 - accuracy: 0.4978
<br />22944/25000 [==========================>...] - ETA: 7s - loss: 7.7034 - accuracy: 0.4976
<br />22976/25000 [==========================>...] - ETA: 7s - loss: 7.7027 - accuracy: 0.4976
<br />23008/25000 [==========================>...] - ETA: 7s - loss: 7.7019 - accuracy: 0.4977
<br />23040/25000 [==========================>...] - ETA: 6s - loss: 7.7026 - accuracy: 0.4977
<br />23072/25000 [==========================>...] - ETA: 6s - loss: 7.6979 - accuracy: 0.4980
<br />23104/25000 [==========================>...] - ETA: 6s - loss: 7.7005 - accuracy: 0.4978
<br />23136/25000 [==========================>...] - ETA: 6s - loss: 7.7011 - accuracy: 0.4978
<br />23168/25000 [==========================>...] - ETA: 6s - loss: 7.6997 - accuracy: 0.4978
<br />23200/25000 [==========================>...] - ETA: 6s - loss: 7.6977 - accuracy: 0.4980
<br />23232/25000 [==========================>...] - ETA: 6s - loss: 7.6943 - accuracy: 0.4982
<br />23264/25000 [==========================>...] - ETA: 6s - loss: 7.6917 - accuracy: 0.4984
<br />23296/25000 [==========================>...] - ETA: 6s - loss: 7.6916 - accuracy: 0.4984
<br />23328/25000 [==========================>...] - ETA: 5s - loss: 7.6949 - accuracy: 0.4982
<br />23360/25000 [===========================>..] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
<br />23392/25000 [===========================>..] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
<br />23424/25000 [===========================>..] - ETA: 5s - loss: 7.6948 - accuracy: 0.4982
<br />23456/25000 [===========================>..] - ETA: 5s - loss: 7.6954 - accuracy: 0.4981
<br />23488/25000 [===========================>..] - ETA: 5s - loss: 7.6940 - accuracy: 0.4982
<br />23520/25000 [===========================>..] - ETA: 5s - loss: 7.6947 - accuracy: 0.4982
<br />23552/25000 [===========================>..] - ETA: 5s - loss: 7.6940 - accuracy: 0.4982
<br />23584/25000 [===========================>..] - ETA: 5s - loss: 7.6959 - accuracy: 0.4981
<br />23616/25000 [===========================>..] - ETA: 4s - loss: 7.6984 - accuracy: 0.4979
<br />23648/25000 [===========================>..] - ETA: 4s - loss: 7.6977 - accuracy: 0.4980
<br />23680/25000 [===========================>..] - ETA: 4s - loss: 7.6990 - accuracy: 0.4979
<br />23712/25000 [===========================>..] - ETA: 4s - loss: 7.6977 - accuracy: 0.4980
<br />23744/25000 [===========================>..] - ETA: 4s - loss: 7.6996 - accuracy: 0.4979
<br />23776/25000 [===========================>..] - ETA: 4s - loss: 7.6976 - accuracy: 0.4980
<br />23808/25000 [===========================>..] - ETA: 4s - loss: 7.6962 - accuracy: 0.4981
<br />23840/25000 [===========================>..] - ETA: 4s - loss: 7.6968 - accuracy: 0.4980
<br />23872/25000 [===========================>..] - ETA: 3s - loss: 7.6949 - accuracy: 0.4982
<br />23904/25000 [===========================>..] - ETA: 3s - loss: 7.6923 - accuracy: 0.4983
<br />23936/25000 [===========================>..] - ETA: 3s - loss: 7.6910 - accuracy: 0.4984
<br />23968/25000 [===========================>..] - ETA: 3s - loss: 7.6922 - accuracy: 0.4983
<br />24000/25000 [===========================>..] - ETA: 3s - loss: 7.6909 - accuracy: 0.4984
<br />24032/25000 [===========================>..] - ETA: 3s - loss: 7.6915 - accuracy: 0.4984
<br />24064/25000 [===========================>..] - ETA: 3s - loss: 7.6902 - accuracy: 0.4985
<br />24096/25000 [===========================>..] - ETA: 3s - loss: 7.6902 - accuracy: 0.4985
<br />24128/25000 [===========================>..] - ETA: 3s - loss: 7.6889 - accuracy: 0.4985
<br />24160/25000 [===========================>..] - ETA: 2s - loss: 7.6882 - accuracy: 0.4986
<br />24192/25000 [============================>.] - ETA: 2s - loss: 7.6882 - accuracy: 0.4986
<br />24224/25000 [============================>.] - ETA: 2s - loss: 7.6894 - accuracy: 0.4985
<br />24256/25000 [============================>.] - ETA: 2s - loss: 7.6887 - accuracy: 0.4986
<br />24288/25000 [============================>.] - ETA: 2s - loss: 7.6868 - accuracy: 0.4987
<br />24320/25000 [============================>.] - ETA: 2s - loss: 7.6824 - accuracy: 0.4990
<br />24352/25000 [============================>.] - ETA: 2s - loss: 7.6805 - accuracy: 0.4991
<br />24384/25000 [============================>.] - ETA: 2s - loss: 7.6798 - accuracy: 0.4991
<br />24416/25000 [============================>.] - ETA: 2s - loss: 7.6811 - accuracy: 0.4991
<br />24448/25000 [============================>.] - ETA: 1s - loss: 7.6817 - accuracy: 0.4990
<br />24480/25000 [============================>.] - ETA: 1s - loss: 7.6804 - accuracy: 0.4991
<br />24512/25000 [============================>.] - ETA: 1s - loss: 7.6791 - accuracy: 0.4992
<br />24544/25000 [============================>.] - ETA: 1s - loss: 7.6785 - accuracy: 0.4992
<br />24576/25000 [============================>.] - ETA: 1s - loss: 7.6760 - accuracy: 0.4994
<br />24608/25000 [============================>.] - ETA: 1s - loss: 7.6753 - accuracy: 0.4994
<br />24640/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
<br />24672/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
<br />24704/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
<br />24736/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
<br />24768/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
<br />24800/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
<br />24832/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
<br />24864/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
<br />24896/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
<br />24928/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
<br />24960/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
<br />24992/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
<br />25000/25000 [==============================] - 106s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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



### Error 16, [Traceback at line 1686](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1686)<br />1686..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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



### Error 17, [Traceback at line 1704](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1704)<br />1704..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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



### Error 18, [Traceback at line 1719](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1719)<br />1719..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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



### Error 19, [Traceback at line 1745](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1745)<br />1745..[0;31mNameError[0m                                 Traceback (most recent call last)
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



### Error 20, [Traceback at line 1761](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1761)<br />1761..[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
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



### Error 21, [Traceback at line 1782](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1782)<br />1782..[0;31mIndexError[0m                                Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/models.py[0m in [0;36mmodule_load[0;34m(model_uri, verbose, env_build)[0m
<br />[1;32m     83[0m             [0mmodel_name[0m [0;34m=[0m [0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mstem[0m  [0;31m# remove .py[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 84[0;31m             [0mmodel_name[0m [0;34m=[0m [0mstr[0m[0;34m([0m[0mPath[0m[0;34m([0m[0mmodel_uri[0m[0;34m)[0m[0;34m.[0m[0mparts[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m][0m[0;34m)[0m [0;34m+[0m [0;34m"."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mmodel_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     85[0m             [0;31m# print(model_name)[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mIndexError[0m: tuple index out of range
<br />
<br />During handling of the above exception, another exception occurred:
<br />



### Error 22, [Traceback at line 1792](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1792)<br />1792..[0;31mNameError[0m                                 Traceback (most recent call last)
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



### Error 23, [Traceback at line 1818](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1818)<br />1818..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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
<br />[0;36m  File [0;32m"https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/example/vision_mnist.py"[0;36m, line [0;32m15[0m
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



### Error 24, [Traceback at line 1857](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1857)<br />1857..[0;31mNameError[0m                                 Traceback (most recent call last)
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



### Error 25, [Traceback at line 1877](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1877)<br />1877..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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



### Error 26, [Traceback at line 1895](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1895)<br />1895..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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
<br /><module 'mlmodels' from 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/__init__.py'>
<br />https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_keras/ardmn.json
<br />[0;31m---------------------------------------------------------------------------[0m



### Error 27, [Traceback at line 1954](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1954)<br />1954..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
<br />[0;32m~/work/mlmodels/mlmodels/mlmodels/example/arun_model.py[0m in [0;36m<module>[0;34m[0m
<br />[1;32m     25[0m [0;31m# Model Parameters[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m     26[0m [0;31m# model_pars, data_pars, compute_pars, out_pars[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
<br />[0;32m---> 27[0;31m [0mpars[0m [0;34m=[0m [0mjson[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mopen[0m[0;34m([0m[0mconfig_path[0m [0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m'r'[0m[0;34m)[0m[0;34m)[0m[0;34m[[0m[0mconfig_mode[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
<br />[0m[1;32m     28[0m [0;32mfor[0m [0mkey[0m[0;34m,[0m [0mpdict[0m [0;32min[0m  [0mpars[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m [0;34m:[0m[0;34m[0m[0;34m[0m[0m
<br />[1;32m     29[0m   [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0mpath_norm_dict[0m[0;34m([0m [0mpdict[0m   [0;34m)[0m   [0;31m###Normalize path[0m[0;34m[0m[0;34m[0m[0m
<br />
<br />[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b/mlmodels/model_keras/ardmn.json'
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



### Error 28, [Traceback at line 1980](https://github.com/arita37/mlmodels_store/blob/master/log_jupyter/log_jupyter_2020-05-15-20-15_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py#L1980)<br />1980..[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
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
