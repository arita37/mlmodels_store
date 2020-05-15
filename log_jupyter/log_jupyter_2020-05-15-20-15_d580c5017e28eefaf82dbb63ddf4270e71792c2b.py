
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'd580c5017e28eefaf82dbb63ddf4270e71792c2b', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/d580c5017e28eefaf82dbb63ddf4270e71792c2b

 ************************************************************************************************************************
/home/runner/work/mlmodels/mlmodels/mlmodels/example/
############ List of files ################################
['ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_svm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_home_retail.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras_charcnn_reuters.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow_1_lstm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras-textcnn.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl_titanic.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow__lstm_json.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_titanic.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m4.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_hyper.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m5.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_model.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m4.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py']





 ************************************************************************************************************************
############ Running Jupyter files ################################





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
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//fashion_MNIST_mlmodels.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





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
	Data preprocessing and feature engineering runtime = 0.3s ...
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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:53<01:19, 26.55s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.4217966727078506, 'embedding_size_factor': 0.6339785094573062, 'layers.choice': 0, 'learning_rate': 0.004727459034074303, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 5.81210806488301e-08} and reward: 0.3716
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xda\xfe\xb7x\xb5\xe0\x9cX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe4I\x8dL\x8f\x8c\xf5X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?s]\x19\x9f\x1c\xcaQX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>o4\x19\xbaN\x0c\xd9u.' and reward: 0.3716
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xda\xfe\xb7x\xb5\xe0\x9cX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe4I\x8dL\x8f\x8c\xf5X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?s]\x19\x9f\x1c\xcaQX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>o4\x19\xbaN\x0c\xd9u.' and reward: 0.3716
 60%|██████    | 3/5 [01:45<01:08, 34.23s/it] 60%|██████    | 3/5 [01:45<01:10, 35.08s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.23845590823991103, 'embedding_size_factor': 0.8024384405028506, 'layers.choice': 2, 'learning_rate': 0.0019489400615534838, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.00016948563155486194} and reward: 0.3868
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xce\x85\xb9#\xb6\xd5\xa4X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe9\xad\x93a`j`X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?_\xeert\xe1\x0e\xceX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?&6\xfe}Ed$u.' and reward: 0.3868
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xce\x85\xb9#\xb6\xd5\xa4X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe9\xad\x93a`j`X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?_\xeert\xe1\x0e\xceX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?&6\xfe}Ed$u.' and reward: 0.3868
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 159.28789043426514
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.23845590823991103, 'embedding_size_factor': 0.8024384405028506, 'layers.choice': 2, 'learning_rate': 0.0019489400615534838, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.00016948563155486194}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.7s of the -41.92s of remaining time.
Ensemble size: 37
Ensemble weights: 
[0.37837838 0.51351351 0.10810811]
	0.3946	 = Validation accuracy score
	1.06s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 163.01s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
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
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vison_fashion_MNIST.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.ipynb[0m in [0;36m<module>[0;34m[0m
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f8fb6738ac8> 

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
 [-0.08564074  0.06497264 -0.01655756 -0.00090496 -0.02879354  0.07852428]
 [ 0.00970696  0.19001883 -0.03477718  0.09692499 -0.15627864  0.05839501]
 [-0.03038519 -0.08257481  0.10741445  0.43009552  0.11023194  0.15371193]
 [-0.05999951  0.14061755  0.09784633  0.05734454  0.33897623  0.10788444]
 [-0.03365828  0.31074873 -0.0024698   0.04576865 -0.06958254  0.38334733]
 [-0.06228766  0.09705011 -0.26475406  0.45682052  0.14575714  0.19166605]
 [-0.49674183  0.77981418 -0.43618864  0.34925169  0.5712015   0.41432109]
 [ 0.46336123  0.79205626  0.10868432  0.19246262  0.36018971  0.43245375]
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
{'loss': 0.40036983974277973, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 20:18:54.089964: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.4746616706252098, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 20:18:55.279726: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 2588672/17464789 [===>..........................] - ETA: 0s
 7716864/17464789 [============>.................] - ETA: 0s
13107200/17464789 [=====================>........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 20:19:07.118651: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 20:19:07.122799: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 20:19:07.122946: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5564f04edb30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 20:19:07.122960: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:57 - loss: 8.1458 - accuracy: 0.4688
   64/25000 [..............................] - ETA: 3:13 - loss: 7.9062 - accuracy: 0.4844
   96/25000 [..............................] - ETA: 2:38 - loss: 7.1875 - accuracy: 0.5312
  128/25000 [..............................] - ETA: 2:19 - loss: 7.5468 - accuracy: 0.5078
  160/25000 [..............................] - ETA: 2:07 - loss: 7.3791 - accuracy: 0.5188
  192/25000 [..............................] - ETA: 2:01 - loss: 7.3472 - accuracy: 0.5208
  224/25000 [..............................] - ETA: 1:56 - loss: 7.2559 - accuracy: 0.5268
  256/25000 [..............................] - ETA: 1:52 - loss: 7.1875 - accuracy: 0.5312
  288/25000 [..............................] - ETA: 1:49 - loss: 7.2407 - accuracy: 0.5278
  320/25000 [..............................] - ETA: 1:47 - loss: 7.3791 - accuracy: 0.5188
  352/25000 [..............................] - ETA: 1:45 - loss: 7.3617 - accuracy: 0.5199
  384/25000 [..............................] - ETA: 1:43 - loss: 7.1875 - accuracy: 0.5312
  416/25000 [..............................] - ETA: 1:41 - loss: 7.2243 - accuracy: 0.5288
  448/25000 [..............................] - ETA: 1:40 - loss: 7.2901 - accuracy: 0.5246
  480/25000 [..............................] - ETA: 1:39 - loss: 7.3152 - accuracy: 0.5229
  512/25000 [..............................] - ETA: 1:38 - loss: 7.3372 - accuracy: 0.5215
  544/25000 [..............................] - ETA: 1:38 - loss: 7.3284 - accuracy: 0.5221
  576/25000 [..............................] - ETA: 1:37 - loss: 7.3472 - accuracy: 0.5208
  608/25000 [..............................] - ETA: 1:37 - loss: 7.3388 - accuracy: 0.5214
  640/25000 [..............................] - ETA: 1:36 - loss: 7.4510 - accuracy: 0.5141
  672/25000 [..............................] - ETA: 1:36 - loss: 7.5069 - accuracy: 0.5104
  704/25000 [..............................] - ETA: 1:35 - loss: 7.3835 - accuracy: 0.5185
  736/25000 [..............................] - ETA: 1:35 - loss: 7.3958 - accuracy: 0.5177
  768/25000 [..............................] - ETA: 1:34 - loss: 7.5269 - accuracy: 0.5091
  800/25000 [..............................] - ETA: 1:34 - loss: 7.4558 - accuracy: 0.5138
  832/25000 [..............................] - ETA: 1:34 - loss: 7.3717 - accuracy: 0.5192
  864/25000 [>.............................] - ETA: 1:34 - loss: 7.3649 - accuracy: 0.5197
  896/25000 [>.............................] - ETA: 1:34 - loss: 7.4442 - accuracy: 0.5145
  928/25000 [>.............................] - ETA: 1:34 - loss: 7.4849 - accuracy: 0.5119
  960/25000 [>.............................] - ETA: 1:33 - loss: 7.4909 - accuracy: 0.5115
  992/25000 [>.............................] - ETA: 1:33 - loss: 7.4966 - accuracy: 0.5111
 1024/25000 [>.............................] - ETA: 1:32 - loss: 7.4720 - accuracy: 0.5127
 1056/25000 [>.............................] - ETA: 1:32 - loss: 7.4343 - accuracy: 0.5152
 1088/25000 [>.............................] - ETA: 1:31 - loss: 7.4129 - accuracy: 0.5165
 1120/25000 [>.............................] - ETA: 1:31 - loss: 7.4202 - accuracy: 0.5161
 1152/25000 [>.............................] - ETA: 1:31 - loss: 7.4270 - accuracy: 0.5156
 1184/25000 [>.............................] - ETA: 1:30 - loss: 7.4594 - accuracy: 0.5135
 1216/25000 [>.............................] - ETA: 1:30 - loss: 7.4018 - accuracy: 0.5173
 1248/25000 [>.............................] - ETA: 1:30 - loss: 7.3595 - accuracy: 0.5200
 1280/25000 [>.............................] - ETA: 1:29 - loss: 7.3791 - accuracy: 0.5188
 1312/25000 [>.............................] - ETA: 1:29 - loss: 7.3978 - accuracy: 0.5175
 1344/25000 [>.............................] - ETA: 1:29 - loss: 7.3928 - accuracy: 0.5179
 1376/25000 [>.............................] - ETA: 1:29 - loss: 7.3657 - accuracy: 0.5196
 1408/25000 [>.............................] - ETA: 1:28 - loss: 7.3399 - accuracy: 0.5213
 1440/25000 [>.............................] - ETA: 1:28 - loss: 7.3365 - accuracy: 0.5215
 1472/25000 [>.............................] - ETA: 1:28 - loss: 7.3333 - accuracy: 0.5217
 1504/25000 [>.............................] - ETA: 1:27 - loss: 7.3812 - accuracy: 0.5186
 1536/25000 [>.............................] - ETA: 1:27 - loss: 7.4171 - accuracy: 0.5163
 1568/25000 [>.............................] - ETA: 1:27 - loss: 7.3830 - accuracy: 0.5185
 1600/25000 [>.............................] - ETA: 1:27 - loss: 7.3983 - accuracy: 0.5175
 1632/25000 [>.............................] - ETA: 1:27 - loss: 7.4223 - accuracy: 0.5159
 1664/25000 [>.............................] - ETA: 1:26 - loss: 7.4363 - accuracy: 0.5150
 1696/25000 [=>............................] - ETA: 1:26 - loss: 7.4406 - accuracy: 0.5147
 1728/25000 [=>............................] - ETA: 1:26 - loss: 7.4182 - accuracy: 0.5162
 1760/25000 [=>............................] - ETA: 1:26 - loss: 7.4401 - accuracy: 0.5148
 1792/25000 [=>............................] - ETA: 1:26 - loss: 7.4356 - accuracy: 0.5151
 1824/25000 [=>............................] - ETA: 1:25 - loss: 7.4312 - accuracy: 0.5154
 1856/25000 [=>............................] - ETA: 1:25 - loss: 7.4105 - accuracy: 0.5167
 1888/25000 [=>............................] - ETA: 1:25 - loss: 7.4067 - accuracy: 0.5169
 1920/25000 [=>............................] - ETA: 1:25 - loss: 7.4191 - accuracy: 0.5161
 1952/25000 [=>............................] - ETA: 1:24 - loss: 7.3760 - accuracy: 0.5190
 1984/25000 [=>............................] - ETA: 1:24 - loss: 7.3729 - accuracy: 0.5192
 2016/25000 [=>............................] - ETA: 1:24 - loss: 7.3852 - accuracy: 0.5184
 2048/25000 [=>............................] - ETA: 1:24 - loss: 7.4196 - accuracy: 0.5161
 2080/25000 [=>............................] - ETA: 1:24 - loss: 7.4234 - accuracy: 0.5159
 2112/25000 [=>............................] - ETA: 1:24 - loss: 7.4270 - accuracy: 0.5156
 2144/25000 [=>............................] - ETA: 1:24 - loss: 7.4235 - accuracy: 0.5159
 2176/25000 [=>............................] - ETA: 1:23 - loss: 7.4905 - accuracy: 0.5115
 2208/25000 [=>............................] - ETA: 1:23 - loss: 7.4930 - accuracy: 0.5113
 2240/25000 [=>............................] - ETA: 1:23 - loss: 7.4613 - accuracy: 0.5134
 2272/25000 [=>............................] - ETA: 1:23 - loss: 7.4642 - accuracy: 0.5132
 2304/25000 [=>............................] - ETA: 1:23 - loss: 7.4470 - accuracy: 0.5143
 2336/25000 [=>............................] - ETA: 1:22 - loss: 7.4763 - accuracy: 0.5124
 2368/25000 [=>............................] - ETA: 1:22 - loss: 7.4918 - accuracy: 0.5114
 2400/25000 [=>............................] - ETA: 1:22 - loss: 7.5069 - accuracy: 0.5104
 2432/25000 [=>............................] - ETA: 1:22 - loss: 7.5153 - accuracy: 0.5099
 2464/25000 [=>............................] - ETA: 1:22 - loss: 7.5110 - accuracy: 0.5101
 2496/25000 [=>............................] - ETA: 1:22 - loss: 7.5376 - accuracy: 0.5084
 2528/25000 [==>...........................] - ETA: 1:21 - loss: 7.5271 - accuracy: 0.5091
 2560/25000 [==>...........................] - ETA: 1:21 - loss: 7.5528 - accuracy: 0.5074
 2592/25000 [==>...........................] - ETA: 1:21 - loss: 7.5601 - accuracy: 0.5069
 2624/25000 [==>...........................] - ETA: 1:21 - loss: 7.5731 - accuracy: 0.5061
 2656/25000 [==>...........................] - ETA: 1:21 - loss: 7.5569 - accuracy: 0.5072
 2688/25000 [==>...........................] - ETA: 1:21 - loss: 7.5525 - accuracy: 0.5074
 2720/25000 [==>...........................] - ETA: 1:21 - loss: 7.5651 - accuracy: 0.5066
 2752/25000 [==>...........................] - ETA: 1:20 - loss: 7.5775 - accuracy: 0.5058
 2784/25000 [==>...........................] - ETA: 1:20 - loss: 7.5510 - accuracy: 0.5075
 2816/25000 [==>...........................] - ETA: 1:20 - loss: 7.5686 - accuracy: 0.5064
 2848/25000 [==>...........................] - ETA: 1:20 - loss: 7.5805 - accuracy: 0.5056
 2880/25000 [==>...........................] - ETA: 1:20 - loss: 7.6187 - accuracy: 0.5031
 2912/25000 [==>...........................] - ETA: 1:20 - loss: 7.6245 - accuracy: 0.5027
 2944/25000 [==>...........................] - ETA: 1:19 - loss: 7.6250 - accuracy: 0.5027
 2976/25000 [==>...........................] - ETA: 1:19 - loss: 7.6306 - accuracy: 0.5024
 3008/25000 [==>...........................] - ETA: 1:19 - loss: 7.6462 - accuracy: 0.5013
 3040/25000 [==>...........................] - ETA: 1:19 - loss: 7.6464 - accuracy: 0.5013
 3072/25000 [==>...........................] - ETA: 1:19 - loss: 7.6467 - accuracy: 0.5013
 3104/25000 [==>...........................] - ETA: 1:19 - loss: 7.6419 - accuracy: 0.5016
 3136/25000 [==>...........................] - ETA: 1:19 - loss: 7.6422 - accuracy: 0.5016
 3168/25000 [==>...........................] - ETA: 1:18 - loss: 7.6424 - accuracy: 0.5016
 3200/25000 [==>...........................] - ETA: 1:18 - loss: 7.6618 - accuracy: 0.5003
 3232/25000 [==>...........................] - ETA: 1:18 - loss: 7.6619 - accuracy: 0.5003
 3264/25000 [==>...........................] - ETA: 1:18 - loss: 7.6572 - accuracy: 0.5006
 3296/25000 [==>...........................] - ETA: 1:18 - loss: 7.6666 - accuracy: 0.5000
 3328/25000 [==>...........................] - ETA: 1:18 - loss: 7.6436 - accuracy: 0.5015
 3360/25000 [===>..........................] - ETA: 1:18 - loss: 7.6392 - accuracy: 0.5018
 3392/25000 [===>..........................] - ETA: 1:17 - loss: 7.6440 - accuracy: 0.5015
 3424/25000 [===>..........................] - ETA: 1:17 - loss: 7.6532 - accuracy: 0.5009
 3456/25000 [===>..........................] - ETA: 1:17 - loss: 7.6444 - accuracy: 0.5014
 3488/25000 [===>..........................] - ETA: 1:17 - loss: 7.6446 - accuracy: 0.5014
 3520/25000 [===>..........................] - ETA: 1:17 - loss: 7.6623 - accuracy: 0.5003
 3552/25000 [===>..........................] - ETA: 1:17 - loss: 7.6666 - accuracy: 0.5000
 3584/25000 [===>..........................] - ETA: 1:17 - loss: 7.6752 - accuracy: 0.4994
 3616/25000 [===>..........................] - ETA: 1:17 - loss: 7.6751 - accuracy: 0.4994
 3648/25000 [===>..........................] - ETA: 1:16 - loss: 7.6792 - accuracy: 0.4992
 3680/25000 [===>..........................] - ETA: 1:16 - loss: 7.6833 - accuracy: 0.4989
 3712/25000 [===>..........................] - ETA: 1:16 - loss: 7.6955 - accuracy: 0.4981
 3744/25000 [===>..........................] - ETA: 1:16 - loss: 7.6912 - accuracy: 0.4984
 3776/25000 [===>..........................] - ETA: 1:16 - loss: 7.6910 - accuracy: 0.4984
 3808/25000 [===>..........................] - ETA: 1:16 - loss: 7.7230 - accuracy: 0.4963
 3840/25000 [===>..........................] - ETA: 1:16 - loss: 7.7345 - accuracy: 0.4956
 3872/25000 [===>..........................] - ETA: 1:16 - loss: 7.7300 - accuracy: 0.4959
 3904/25000 [===>..........................] - ETA: 1:15 - loss: 7.7295 - accuracy: 0.4959
 3936/25000 [===>..........................] - ETA: 1:15 - loss: 7.7289 - accuracy: 0.4959
 3968/25000 [===>..........................] - ETA: 1:15 - loss: 7.7400 - accuracy: 0.4952
 4000/25000 [===>..........................] - ETA: 1:15 - loss: 7.7356 - accuracy: 0.4955
 4032/25000 [===>..........................] - ETA: 1:15 - loss: 7.7275 - accuracy: 0.4960
 4064/25000 [===>..........................] - ETA: 1:15 - loss: 7.7157 - accuracy: 0.4968
 4096/25000 [===>..........................] - ETA: 1:15 - loss: 7.7041 - accuracy: 0.4976
 4128/25000 [===>..........................] - ETA: 1:15 - loss: 7.7186 - accuracy: 0.4966
 4160/25000 [===>..........................] - ETA: 1:14 - loss: 7.7219 - accuracy: 0.4964
 4192/25000 [====>.........................] - ETA: 1:14 - loss: 7.7069 - accuracy: 0.4974
 4224/25000 [====>.........................] - ETA: 1:14 - loss: 7.7102 - accuracy: 0.4972
 4256/25000 [====>.........................] - ETA: 1:14 - loss: 7.7171 - accuracy: 0.4967
 4288/25000 [====>.........................] - ETA: 1:14 - loss: 7.7167 - accuracy: 0.4967
 4320/25000 [====>.........................] - ETA: 1:14 - loss: 7.7128 - accuracy: 0.4970
 4352/25000 [====>.........................] - ETA: 1:14 - loss: 7.7089 - accuracy: 0.4972
 4384/25000 [====>.........................] - ETA: 1:14 - loss: 7.7156 - accuracy: 0.4968
 4416/25000 [====>.........................] - ETA: 1:13 - loss: 7.7187 - accuracy: 0.4966
 4448/25000 [====>.........................] - ETA: 1:13 - loss: 7.7045 - accuracy: 0.4975
 4480/25000 [====>.........................] - ETA: 1:13 - loss: 7.7043 - accuracy: 0.4975
 4512/25000 [====>.........................] - ETA: 1:13 - loss: 7.7142 - accuracy: 0.4969
 4544/25000 [====>.........................] - ETA: 1:13 - loss: 7.7071 - accuracy: 0.4974
 4576/25000 [====>.........................] - ETA: 1:13 - loss: 7.7169 - accuracy: 0.4967
 4608/25000 [====>.........................] - ETA: 1:13 - loss: 7.7165 - accuracy: 0.4967
 4640/25000 [====>.........................] - ETA: 1:13 - loss: 7.7096 - accuracy: 0.4972
 4672/25000 [====>.........................] - ETA: 1:12 - loss: 7.7126 - accuracy: 0.4970
 4704/25000 [====>.........................] - ETA: 1:12 - loss: 7.7188 - accuracy: 0.4966
 4736/25000 [====>.........................] - ETA: 1:12 - loss: 7.7055 - accuracy: 0.4975
 4768/25000 [====>.........................] - ETA: 1:12 - loss: 7.7020 - accuracy: 0.4977
 4800/25000 [====>.........................] - ETA: 1:12 - loss: 7.7241 - accuracy: 0.4963
 4832/25000 [====>.........................] - ETA: 1:12 - loss: 7.7174 - accuracy: 0.4967
 4864/25000 [====>.........................] - ETA: 1:12 - loss: 7.7108 - accuracy: 0.4971
 4896/25000 [====>.........................] - ETA: 1:12 - loss: 7.7230 - accuracy: 0.4963
 4928/25000 [====>.........................] - ETA: 1:11 - loss: 7.7257 - accuracy: 0.4961
 4960/25000 [====>.........................] - ETA: 1:11 - loss: 7.7377 - accuracy: 0.4954
 4992/25000 [====>.........................] - ETA: 1:11 - loss: 7.7342 - accuracy: 0.4956
 5024/25000 [=====>........................] - ETA: 1:11 - loss: 7.7216 - accuracy: 0.4964
 5056/25000 [=====>........................] - ETA: 1:11 - loss: 7.7242 - accuracy: 0.4962
 5088/25000 [=====>........................] - ETA: 1:11 - loss: 7.7179 - accuracy: 0.4967
 5120/25000 [=====>........................] - ETA: 1:11 - loss: 7.7145 - accuracy: 0.4969
 5152/25000 [=====>........................] - ETA: 1:11 - loss: 7.7113 - accuracy: 0.4971
 5184/25000 [=====>........................] - ETA: 1:11 - loss: 7.7080 - accuracy: 0.4973
 5216/25000 [=====>........................] - ETA: 1:10 - loss: 7.7195 - accuracy: 0.4965
 5248/25000 [=====>........................] - ETA: 1:10 - loss: 7.7163 - accuracy: 0.4968
 5280/25000 [=====>........................] - ETA: 1:10 - loss: 7.7160 - accuracy: 0.4968
 5312/25000 [=====>........................] - ETA: 1:10 - loss: 7.7330 - accuracy: 0.4957
 5344/25000 [=====>........................] - ETA: 1:10 - loss: 7.7470 - accuracy: 0.4948
 5376/25000 [=====>........................] - ETA: 1:10 - loss: 7.7436 - accuracy: 0.4950
 5408/25000 [=====>........................] - ETA: 1:10 - loss: 7.7545 - accuracy: 0.4943
 5440/25000 [=====>........................] - ETA: 1:10 - loss: 7.7681 - accuracy: 0.4934
 5472/25000 [=====>........................] - ETA: 1:09 - loss: 7.7703 - accuracy: 0.4932
 5504/25000 [=====>........................] - ETA: 1:09 - loss: 7.7781 - accuracy: 0.4927
 5536/25000 [=====>........................] - ETA: 1:09 - loss: 7.7774 - accuracy: 0.4928
 5568/25000 [=====>........................] - ETA: 1:09 - loss: 7.7768 - accuracy: 0.4928
 5600/25000 [=====>........................] - ETA: 1:09 - loss: 7.7871 - accuracy: 0.4921
 5632/25000 [=====>........................] - ETA: 1:09 - loss: 7.7891 - accuracy: 0.4920
 5664/25000 [=====>........................] - ETA: 1:09 - loss: 7.7939 - accuracy: 0.4917
 5696/25000 [=====>........................] - ETA: 1:09 - loss: 7.8039 - accuracy: 0.4910
 5728/25000 [=====>........................] - ETA: 1:08 - loss: 7.7817 - accuracy: 0.4925
 5760/25000 [=====>........................] - ETA: 1:08 - loss: 7.7731 - accuracy: 0.4931
 5792/25000 [=====>........................] - ETA: 1:08 - loss: 7.7805 - accuracy: 0.4926
 5824/25000 [=====>........................] - ETA: 1:08 - loss: 7.7746 - accuracy: 0.4930
 5856/25000 [======>.......................] - ETA: 1:08 - loss: 7.7766 - accuracy: 0.4928
 5888/25000 [======>.......................] - ETA: 1:08 - loss: 7.7760 - accuracy: 0.4929
 5920/25000 [======>.......................] - ETA: 1:08 - loss: 7.7754 - accuracy: 0.4929
 5952/25000 [======>.......................] - ETA: 1:08 - loss: 7.7748 - accuracy: 0.4929
 5984/25000 [======>.......................] - ETA: 1:08 - loss: 7.7742 - accuracy: 0.4930
 6016/25000 [======>.......................] - ETA: 1:08 - loss: 7.7711 - accuracy: 0.4932
 6048/25000 [======>.......................] - ETA: 1:07 - loss: 7.7680 - accuracy: 0.4934
 6080/25000 [======>.......................] - ETA: 1:07 - loss: 7.7826 - accuracy: 0.4924
 6112/25000 [======>.......................] - ETA: 1:07 - loss: 7.7795 - accuracy: 0.4926
 6144/25000 [======>.......................] - ETA: 1:07 - loss: 7.7789 - accuracy: 0.4927
 6176/25000 [======>.......................] - ETA: 1:07 - loss: 7.7883 - accuracy: 0.4921
 6208/25000 [======>.......................] - ETA: 1:07 - loss: 7.7876 - accuracy: 0.4921
 6240/25000 [======>.......................] - ETA: 1:07 - loss: 7.7846 - accuracy: 0.4923
 6272/25000 [======>.......................] - ETA: 1:07 - loss: 7.7815 - accuracy: 0.4925
 6304/25000 [======>.......................] - ETA: 1:06 - loss: 7.7907 - accuracy: 0.4919
 6336/25000 [======>.......................] - ETA: 1:06 - loss: 7.7876 - accuracy: 0.4921
 6368/25000 [======>.......................] - ETA: 1:06 - loss: 7.7991 - accuracy: 0.4914
 6400/25000 [======>.......................] - ETA: 1:06 - loss: 7.8032 - accuracy: 0.4911
 6432/25000 [======>.......................] - ETA: 1:06 - loss: 7.7954 - accuracy: 0.4916
 6464/25000 [======>.......................] - ETA: 1:06 - loss: 7.7805 - accuracy: 0.4926
 6496/25000 [======>.......................] - ETA: 1:06 - loss: 7.7776 - accuracy: 0.4928
 6528/25000 [======>.......................] - ETA: 1:06 - loss: 7.7841 - accuracy: 0.4923
 6560/25000 [======>.......................] - ETA: 1:06 - loss: 7.7928 - accuracy: 0.4918
 6592/25000 [======>.......................] - ETA: 1:05 - loss: 7.7876 - accuracy: 0.4921
 6624/25000 [======>.......................] - ETA: 1:05 - loss: 7.7893 - accuracy: 0.4920
 6656/25000 [======>.......................] - ETA: 1:05 - loss: 7.7841 - accuracy: 0.4923
 6688/25000 [=======>......................] - ETA: 1:05 - loss: 7.7813 - accuracy: 0.4925
 6720/25000 [=======>......................] - ETA: 1:05 - loss: 7.7830 - accuracy: 0.4924
 6752/25000 [=======>......................] - ETA: 1:05 - loss: 7.7802 - accuracy: 0.4926
 6784/25000 [=======>......................] - ETA: 1:05 - loss: 7.7751 - accuracy: 0.4929
 6816/25000 [=======>......................] - ETA: 1:05 - loss: 7.7791 - accuracy: 0.4927
 6848/25000 [=======>......................] - ETA: 1:04 - loss: 7.7674 - accuracy: 0.4934
 6880/25000 [=======>......................] - ETA: 1:04 - loss: 7.7781 - accuracy: 0.4927
 6912/25000 [=======>......................] - ETA: 1:04 - loss: 7.7753 - accuracy: 0.4929
 6944/25000 [=======>......................] - ETA: 1:04 - loss: 7.7814 - accuracy: 0.4925
 6976/25000 [=======>......................] - ETA: 1:04 - loss: 7.7809 - accuracy: 0.4925
 7008/25000 [=======>......................] - ETA: 1:04 - loss: 7.7782 - accuracy: 0.4927
 7040/25000 [=======>......................] - ETA: 1:04 - loss: 7.7755 - accuracy: 0.4929
 7072/25000 [=======>......................] - ETA: 1:04 - loss: 7.7772 - accuracy: 0.4928
 7104/25000 [=======>......................] - ETA: 1:04 - loss: 7.7767 - accuracy: 0.4928
 7136/25000 [=======>......................] - ETA: 1:03 - loss: 7.7805 - accuracy: 0.4926
 7168/25000 [=======>......................] - ETA: 1:03 - loss: 7.7821 - accuracy: 0.4925
 7200/25000 [=======>......................] - ETA: 1:03 - loss: 7.7795 - accuracy: 0.4926
 7232/25000 [=======>......................] - ETA: 1:03 - loss: 7.7769 - accuracy: 0.4928
 7264/25000 [=======>......................] - ETA: 1:03 - loss: 7.7743 - accuracy: 0.4930
 7296/25000 [=======>......................] - ETA: 1:03 - loss: 7.7717 - accuracy: 0.4931
 7328/25000 [=======>......................] - ETA: 1:03 - loss: 7.7754 - accuracy: 0.4929
 7360/25000 [=======>......................] - ETA: 1:03 - loss: 7.7750 - accuracy: 0.4929
 7392/25000 [=======>......................] - ETA: 1:02 - loss: 7.7703 - accuracy: 0.4932
 7424/25000 [=======>......................] - ETA: 1:02 - loss: 7.7678 - accuracy: 0.4934
 7456/25000 [=======>......................] - ETA: 1:02 - loss: 7.7653 - accuracy: 0.4936
 7488/25000 [=======>......................] - ETA: 1:02 - loss: 7.7629 - accuracy: 0.4937
 7520/25000 [========>.....................] - ETA: 1:02 - loss: 7.7706 - accuracy: 0.4932
 7552/25000 [========>.....................] - ETA: 1:02 - loss: 7.7580 - accuracy: 0.4940
 7584/25000 [========>.....................] - ETA: 1:02 - loss: 7.7536 - accuracy: 0.4943
 7616/25000 [========>.....................] - ETA: 1:02 - loss: 7.7552 - accuracy: 0.4942
 7648/25000 [========>.....................] - ETA: 1:02 - loss: 7.7568 - accuracy: 0.4941
 7680/25000 [========>.....................] - ETA: 1:01 - loss: 7.7585 - accuracy: 0.4940
 7712/25000 [========>.....................] - ETA: 1:01 - loss: 7.7581 - accuracy: 0.4940
 7744/25000 [========>.....................] - ETA: 1:01 - loss: 7.7518 - accuracy: 0.4944
 7776/25000 [========>.....................] - ETA: 1:01 - loss: 7.7514 - accuracy: 0.4945
 7808/25000 [========>.....................] - ETA: 1:01 - loss: 7.7491 - accuracy: 0.4946
 7840/25000 [========>.....................] - ETA: 1:01 - loss: 7.7507 - accuracy: 0.4945
 7872/25000 [========>.....................] - ETA: 1:01 - loss: 7.7445 - accuracy: 0.4949
 7904/25000 [========>.....................] - ETA: 1:01 - loss: 7.7423 - accuracy: 0.4951
 7936/25000 [========>.....................] - ETA: 1:00 - loss: 7.7420 - accuracy: 0.4951
 7968/25000 [========>.....................] - ETA: 1:00 - loss: 7.7436 - accuracy: 0.4950
 8000/25000 [========>.....................] - ETA: 1:00 - loss: 7.7490 - accuracy: 0.4946
 8032/25000 [========>.....................] - ETA: 1:00 - loss: 7.7430 - accuracy: 0.4950
 8064/25000 [========>.....................] - ETA: 1:00 - loss: 7.7484 - accuracy: 0.4947
 8096/25000 [========>.....................] - ETA: 1:00 - loss: 7.7500 - accuracy: 0.4946
 8128/25000 [========>.....................] - ETA: 1:00 - loss: 7.7402 - accuracy: 0.4952
 8160/25000 [========>.....................] - ETA: 1:00 - loss: 7.7493 - accuracy: 0.4946
 8192/25000 [========>.....................] - ETA: 1:00 - loss: 7.7527 - accuracy: 0.4944
 8224/25000 [========>.....................] - ETA: 59s - loss: 7.7505 - accuracy: 0.4945 
 8256/25000 [========>.....................] - ETA: 59s - loss: 7.7465 - accuracy: 0.4948
 8288/25000 [========>.....................] - ETA: 59s - loss: 7.7462 - accuracy: 0.4948
 8320/25000 [========>.....................] - ETA: 59s - loss: 7.7477 - accuracy: 0.4947
 8352/25000 [=========>....................] - ETA: 59s - loss: 7.7456 - accuracy: 0.4949
 8384/25000 [=========>....................] - ETA: 59s - loss: 7.7453 - accuracy: 0.4949
 8416/25000 [=========>....................] - ETA: 59s - loss: 7.7450 - accuracy: 0.4949
 8448/25000 [=========>....................] - ETA: 59s - loss: 7.7356 - accuracy: 0.4955
 8480/25000 [=========>....................] - ETA: 58s - loss: 7.7353 - accuracy: 0.4955
 8512/25000 [=========>....................] - ETA: 58s - loss: 7.7333 - accuracy: 0.4957
 8544/25000 [=========>....................] - ETA: 58s - loss: 7.7223 - accuracy: 0.4964
 8576/25000 [=========>....................] - ETA: 58s - loss: 7.7220 - accuracy: 0.4964
 8608/25000 [=========>....................] - ETA: 58s - loss: 7.7254 - accuracy: 0.4962
 8640/25000 [=========>....................] - ETA: 58s - loss: 7.7199 - accuracy: 0.4965
 8672/25000 [=========>....................] - ETA: 58s - loss: 7.7038 - accuracy: 0.4976
 8704/25000 [=========>....................] - ETA: 58s - loss: 7.6983 - accuracy: 0.4979
 8736/25000 [=========>....................] - ETA: 58s - loss: 7.6912 - accuracy: 0.4984
 8768/25000 [=========>....................] - ETA: 57s - loss: 7.6859 - accuracy: 0.4987
 8800/25000 [=========>....................] - ETA: 57s - loss: 7.6858 - accuracy: 0.4988
 8832/25000 [=========>....................] - ETA: 57s - loss: 7.6788 - accuracy: 0.4992
 8864/25000 [=========>....................] - ETA: 57s - loss: 7.6735 - accuracy: 0.4995
 8896/25000 [=========>....................] - ETA: 57s - loss: 7.6735 - accuracy: 0.4996
 8928/25000 [=========>....................] - ETA: 57s - loss: 7.6701 - accuracy: 0.4998
 8960/25000 [=========>....................] - ETA: 57s - loss: 7.6735 - accuracy: 0.4996
 8992/25000 [=========>....................] - ETA: 57s - loss: 7.6786 - accuracy: 0.4992
 9024/25000 [=========>....................] - ETA: 57s - loss: 7.6819 - accuracy: 0.4990
 9056/25000 [=========>....................] - ETA: 56s - loss: 7.6802 - accuracy: 0.4991
 9088/25000 [=========>....................] - ETA: 56s - loss: 7.6734 - accuracy: 0.4996
 9120/25000 [=========>....................] - ETA: 56s - loss: 7.6717 - accuracy: 0.4997
 9152/25000 [=========>....................] - ETA: 56s - loss: 7.6800 - accuracy: 0.4991
 9184/25000 [==========>...................] - ETA: 56s - loss: 7.6800 - accuracy: 0.4991
 9216/25000 [==========>...................] - ETA: 56s - loss: 7.6783 - accuracy: 0.4992
 9248/25000 [==========>...................] - ETA: 56s - loss: 7.6815 - accuracy: 0.4990
 9280/25000 [==========>...................] - ETA: 56s - loss: 7.6848 - accuracy: 0.4988
 9312/25000 [==========>...................] - ETA: 55s - loss: 7.6798 - accuracy: 0.4991
 9344/25000 [==========>...................] - ETA: 55s - loss: 7.6847 - accuracy: 0.4988
 9376/25000 [==========>...................] - ETA: 55s - loss: 7.6846 - accuracy: 0.4988
 9408/25000 [==========>...................] - ETA: 55s - loss: 7.6780 - accuracy: 0.4993
 9440/25000 [==========>...................] - ETA: 55s - loss: 7.6780 - accuracy: 0.4993
 9472/25000 [==========>...................] - ETA: 55s - loss: 7.6828 - accuracy: 0.4989
 9504/25000 [==========>...................] - ETA: 55s - loss: 7.6892 - accuracy: 0.4985
 9536/25000 [==========>...................] - ETA: 55s - loss: 7.6875 - accuracy: 0.4986
 9568/25000 [==========>...................] - ETA: 55s - loss: 7.6891 - accuracy: 0.4985
 9600/25000 [==========>...................] - ETA: 54s - loss: 7.7018 - accuracy: 0.4977
 9632/25000 [==========>...................] - ETA: 54s - loss: 7.7016 - accuracy: 0.4977
 9664/25000 [==========>...................] - ETA: 54s - loss: 7.6984 - accuracy: 0.4979
 9696/25000 [==========>...................] - ETA: 54s - loss: 7.6982 - accuracy: 0.4979
 9728/25000 [==========>...................] - ETA: 54s - loss: 7.6981 - accuracy: 0.4979
 9760/25000 [==========>...................] - ETA: 54s - loss: 7.6996 - accuracy: 0.4978
 9792/25000 [==========>...................] - ETA: 54s - loss: 7.6964 - accuracy: 0.4981
 9824/25000 [==========>...................] - ETA: 54s - loss: 7.6932 - accuracy: 0.4983
 9856/25000 [==========>...................] - ETA: 53s - loss: 7.6946 - accuracy: 0.4982
 9888/25000 [==========>...................] - ETA: 53s - loss: 7.6883 - accuracy: 0.4986
 9920/25000 [==========>...................] - ETA: 53s - loss: 7.6883 - accuracy: 0.4986
 9952/25000 [==========>...................] - ETA: 53s - loss: 7.6974 - accuracy: 0.4980
 9984/25000 [==========>...................] - ETA: 53s - loss: 7.6973 - accuracy: 0.4980
10016/25000 [===========>..................] - ETA: 53s - loss: 7.7018 - accuracy: 0.4977
10048/25000 [===========>..................] - ETA: 53s - loss: 7.7017 - accuracy: 0.4977
10080/25000 [===========>..................] - ETA: 53s - loss: 7.6970 - accuracy: 0.4980
10112/25000 [===========>..................] - ETA: 53s - loss: 7.7030 - accuracy: 0.4976
10144/25000 [===========>..................] - ETA: 52s - loss: 7.7029 - accuracy: 0.4976
10176/25000 [===========>..................] - ETA: 52s - loss: 7.6998 - accuracy: 0.4978
10208/25000 [===========>..................] - ETA: 52s - loss: 7.7027 - accuracy: 0.4976
10240/25000 [===========>..................] - ETA: 52s - loss: 7.7056 - accuracy: 0.4975
10272/25000 [===========>..................] - ETA: 52s - loss: 7.6950 - accuracy: 0.4982
10304/25000 [===========>..................] - ETA: 52s - loss: 7.7023 - accuracy: 0.4977
10336/25000 [===========>..................] - ETA: 52s - loss: 7.6993 - accuracy: 0.4979
10368/25000 [===========>..................] - ETA: 52s - loss: 7.6962 - accuracy: 0.4981
10400/25000 [===========>..................] - ETA: 51s - loss: 7.6961 - accuracy: 0.4981
10432/25000 [===========>..................] - ETA: 51s - loss: 7.6975 - accuracy: 0.4980
10464/25000 [===========>..................] - ETA: 51s - loss: 7.7033 - accuracy: 0.4976
10496/25000 [===========>..................] - ETA: 51s - loss: 7.6973 - accuracy: 0.4980
10528/25000 [===========>..................] - ETA: 51s - loss: 7.6972 - accuracy: 0.4980
10560/25000 [===========>..................] - ETA: 51s - loss: 7.6971 - accuracy: 0.4980
10592/25000 [===========>..................] - ETA: 51s - loss: 7.6999 - accuracy: 0.4978
10624/25000 [===========>..................] - ETA: 51s - loss: 7.7056 - accuracy: 0.4975
10656/25000 [===========>..................] - ETA: 51s - loss: 7.7040 - accuracy: 0.4976
10688/25000 [===========>..................] - ETA: 50s - loss: 7.6982 - accuracy: 0.4979
10720/25000 [===========>..................] - ETA: 50s - loss: 7.6981 - accuracy: 0.4979
10752/25000 [===========>..................] - ETA: 50s - loss: 7.7051 - accuracy: 0.4975
10784/25000 [===========>..................] - ETA: 50s - loss: 7.7007 - accuracy: 0.4978
10816/25000 [===========>..................] - ETA: 50s - loss: 7.7049 - accuracy: 0.4975
10848/25000 [============>.................] - ETA: 50s - loss: 7.7005 - accuracy: 0.4978
10880/25000 [============>.................] - ETA: 50s - loss: 7.7019 - accuracy: 0.4977
10912/25000 [============>.................] - ETA: 50s - loss: 7.7074 - accuracy: 0.4973
10944/25000 [============>.................] - ETA: 50s - loss: 7.7115 - accuracy: 0.4971
10976/25000 [============>.................] - ETA: 49s - loss: 7.7057 - accuracy: 0.4974
11008/25000 [============>.................] - ETA: 49s - loss: 7.7000 - accuracy: 0.4978
11040/25000 [============>.................] - ETA: 49s - loss: 7.6986 - accuracy: 0.4979
11072/25000 [============>.................] - ETA: 49s - loss: 7.6999 - accuracy: 0.4978
11104/25000 [============>.................] - ETA: 49s - loss: 7.6984 - accuracy: 0.4979
11136/25000 [============>.................] - ETA: 49s - loss: 7.6983 - accuracy: 0.4979
11168/25000 [============>.................] - ETA: 49s - loss: 7.6927 - accuracy: 0.4983
11200/25000 [============>.................] - ETA: 49s - loss: 7.6926 - accuracy: 0.4983
11232/25000 [============>.................] - ETA: 48s - loss: 7.6980 - accuracy: 0.4980
11264/25000 [============>.................] - ETA: 48s - loss: 7.6979 - accuracy: 0.4980
11296/25000 [============>.................] - ETA: 48s - loss: 7.7006 - accuracy: 0.4978
11328/25000 [============>.................] - ETA: 48s - loss: 7.6937 - accuracy: 0.4982
11360/25000 [============>.................] - ETA: 48s - loss: 7.6990 - accuracy: 0.4979
11392/25000 [============>.................] - ETA: 48s - loss: 7.6949 - accuracy: 0.4982
11424/25000 [============>.................] - ETA: 48s - loss: 7.6894 - accuracy: 0.4985
11456/25000 [============>.................] - ETA: 48s - loss: 7.6907 - accuracy: 0.4984
11488/25000 [============>.................] - ETA: 48s - loss: 7.6946 - accuracy: 0.4982
11520/25000 [============>.................] - ETA: 47s - loss: 7.6999 - accuracy: 0.4978
11552/25000 [============>.................] - ETA: 47s - loss: 7.6971 - accuracy: 0.4980
11584/25000 [============>.................] - ETA: 47s - loss: 7.6918 - accuracy: 0.4984
11616/25000 [============>.................] - ETA: 47s - loss: 7.6943 - accuracy: 0.4982
11648/25000 [============>.................] - ETA: 47s - loss: 7.6943 - accuracy: 0.4982
11680/25000 [=============>................] - ETA: 47s - loss: 7.6876 - accuracy: 0.4986
11712/25000 [=============>................] - ETA: 47s - loss: 7.6928 - accuracy: 0.4983
11744/25000 [=============>................] - ETA: 47s - loss: 7.6927 - accuracy: 0.4983
11776/25000 [=============>................] - ETA: 46s - loss: 7.6862 - accuracy: 0.4987
11808/25000 [=============>................] - ETA: 46s - loss: 7.6874 - accuracy: 0.4986
11840/25000 [=============>................] - ETA: 46s - loss: 7.6860 - accuracy: 0.4987
11872/25000 [=============>................] - ETA: 46s - loss: 7.6808 - accuracy: 0.4991
11904/25000 [=============>................] - ETA: 46s - loss: 7.6769 - accuracy: 0.4993
11936/25000 [=============>................] - ETA: 46s - loss: 7.6833 - accuracy: 0.4989
11968/25000 [=============>................] - ETA: 46s - loss: 7.6871 - accuracy: 0.4987
12000/25000 [=============>................] - ETA: 46s - loss: 7.6807 - accuracy: 0.4991
12032/25000 [=============>................] - ETA: 46s - loss: 7.6794 - accuracy: 0.4992
12064/25000 [=============>................] - ETA: 45s - loss: 7.6755 - accuracy: 0.4994
12096/25000 [=============>................] - ETA: 45s - loss: 7.6704 - accuracy: 0.4998
12128/25000 [=============>................] - ETA: 45s - loss: 7.6641 - accuracy: 0.5002
12160/25000 [=============>................] - ETA: 45s - loss: 7.6666 - accuracy: 0.5000
12192/25000 [=============>................] - ETA: 45s - loss: 7.6691 - accuracy: 0.4998
12224/25000 [=============>................] - ETA: 45s - loss: 7.6654 - accuracy: 0.5001
12256/25000 [=============>................] - ETA: 45s - loss: 7.6716 - accuracy: 0.4997
12288/25000 [=============>................] - ETA: 45s - loss: 7.6766 - accuracy: 0.4993
12320/25000 [=============>................] - ETA: 45s - loss: 7.6778 - accuracy: 0.4993
12352/25000 [=============>................] - ETA: 44s - loss: 7.6766 - accuracy: 0.4994
12384/25000 [=============>................] - ETA: 44s - loss: 7.6740 - accuracy: 0.4995
12416/25000 [=============>................] - ETA: 44s - loss: 7.6740 - accuracy: 0.4995
12448/25000 [=============>................] - ETA: 44s - loss: 7.6715 - accuracy: 0.4997
12480/25000 [=============>................] - ETA: 44s - loss: 7.6678 - accuracy: 0.4999
12512/25000 [==============>...............] - ETA: 44s - loss: 7.6691 - accuracy: 0.4998
12544/25000 [==============>...............] - ETA: 44s - loss: 7.6691 - accuracy: 0.4998
12576/25000 [==============>...............] - ETA: 44s - loss: 7.6703 - accuracy: 0.4998
12608/25000 [==============>...............] - ETA: 44s - loss: 7.6691 - accuracy: 0.4998
12640/25000 [==============>...............] - ETA: 43s - loss: 7.6727 - accuracy: 0.4996
12672/25000 [==============>...............] - ETA: 43s - loss: 7.6763 - accuracy: 0.4994
12704/25000 [==============>...............] - ETA: 43s - loss: 7.6702 - accuracy: 0.4998
12736/25000 [==============>...............] - ETA: 43s - loss: 7.6714 - accuracy: 0.4997
12768/25000 [==============>...............] - ETA: 43s - loss: 7.6702 - accuracy: 0.4998
12800/25000 [==============>...............] - ETA: 43s - loss: 7.6762 - accuracy: 0.4994
12832/25000 [==============>...............] - ETA: 43s - loss: 7.6845 - accuracy: 0.4988
12864/25000 [==============>...............] - ETA: 43s - loss: 7.6845 - accuracy: 0.4988
12896/25000 [==============>...............] - ETA: 43s - loss: 7.6880 - accuracy: 0.4986
12928/25000 [==============>...............] - ETA: 42s - loss: 7.6880 - accuracy: 0.4986
12960/25000 [==============>...............] - ETA: 42s - loss: 7.6891 - accuracy: 0.4985
12992/25000 [==============>...............] - ETA: 42s - loss: 7.6890 - accuracy: 0.4985
13024/25000 [==============>...............] - ETA: 42s - loss: 7.6902 - accuracy: 0.4985
13056/25000 [==============>...............] - ETA: 42s - loss: 7.6960 - accuracy: 0.4981
13088/25000 [==============>...............] - ETA: 42s - loss: 7.6936 - accuracy: 0.4982
13120/25000 [==============>...............] - ETA: 42s - loss: 7.6935 - accuracy: 0.4982
13152/25000 [==============>...............] - ETA: 42s - loss: 7.6934 - accuracy: 0.4983
13184/25000 [==============>...............] - ETA: 41s - loss: 7.6934 - accuracy: 0.4983
13216/25000 [==============>...............] - ETA: 41s - loss: 7.6933 - accuracy: 0.4983
13248/25000 [==============>...............] - ETA: 41s - loss: 7.6909 - accuracy: 0.4984
13280/25000 [==============>...............] - ETA: 41s - loss: 7.6862 - accuracy: 0.4987
13312/25000 [==============>...............] - ETA: 41s - loss: 7.6943 - accuracy: 0.4982
13344/25000 [===============>..............] - ETA: 41s - loss: 7.6862 - accuracy: 0.4987
13376/25000 [===============>..............] - ETA: 41s - loss: 7.6792 - accuracy: 0.4992
13408/25000 [===============>..............] - ETA: 41s - loss: 7.6849 - accuracy: 0.4988
13440/25000 [===============>..............] - ETA: 41s - loss: 7.6872 - accuracy: 0.4987
13472/25000 [===============>..............] - ETA: 40s - loss: 7.6860 - accuracy: 0.4987
13504/25000 [===============>..............] - ETA: 40s - loss: 7.6848 - accuracy: 0.4988
13536/25000 [===============>..............] - ETA: 40s - loss: 7.6870 - accuracy: 0.4987
13568/25000 [===============>..............] - ETA: 40s - loss: 7.6881 - accuracy: 0.4986
13600/25000 [===============>..............] - ETA: 40s - loss: 7.6847 - accuracy: 0.4988
13632/25000 [===============>..............] - ETA: 40s - loss: 7.6846 - accuracy: 0.4988
13664/25000 [===============>..............] - ETA: 40s - loss: 7.6790 - accuracy: 0.4992
13696/25000 [===============>..............] - ETA: 40s - loss: 7.6801 - accuracy: 0.4991
13728/25000 [===============>..............] - ETA: 40s - loss: 7.6834 - accuracy: 0.4989
13760/25000 [===============>..............] - ETA: 39s - loss: 7.6822 - accuracy: 0.4990
13792/25000 [===============>..............] - ETA: 39s - loss: 7.6855 - accuracy: 0.4988
13824/25000 [===============>..............] - ETA: 39s - loss: 7.6866 - accuracy: 0.4987
13856/25000 [===============>..............] - ETA: 39s - loss: 7.6865 - accuracy: 0.4987
13888/25000 [===============>..............] - ETA: 39s - loss: 7.6821 - accuracy: 0.4990
13920/25000 [===============>..............] - ETA: 39s - loss: 7.6842 - accuracy: 0.4989
13952/25000 [===============>..............] - ETA: 39s - loss: 7.6853 - accuracy: 0.4988
13984/25000 [===============>..............] - ETA: 39s - loss: 7.6885 - accuracy: 0.4986
14016/25000 [===============>..............] - ETA: 38s - loss: 7.6896 - accuracy: 0.4985
14048/25000 [===============>..............] - ETA: 38s - loss: 7.6906 - accuracy: 0.4984
14080/25000 [===============>..............] - ETA: 38s - loss: 7.6928 - accuracy: 0.4983
14112/25000 [===============>..............] - ETA: 38s - loss: 7.6960 - accuracy: 0.4981
14144/25000 [===============>..............] - ETA: 38s - loss: 7.6916 - accuracy: 0.4984
14176/25000 [================>.............] - ETA: 38s - loss: 7.6937 - accuracy: 0.4982
14208/25000 [================>.............] - ETA: 38s - loss: 7.6904 - accuracy: 0.4985
14240/25000 [================>.............] - ETA: 38s - loss: 7.6946 - accuracy: 0.4982
14272/25000 [================>.............] - ETA: 38s - loss: 7.6956 - accuracy: 0.4981
14304/25000 [================>.............] - ETA: 37s - loss: 7.6945 - accuracy: 0.4982
14336/25000 [================>.............] - ETA: 37s - loss: 7.6966 - accuracy: 0.4980
14368/25000 [================>.............] - ETA: 37s - loss: 7.6986 - accuracy: 0.4979
14400/25000 [================>.............] - ETA: 37s - loss: 7.7028 - accuracy: 0.4976
14432/25000 [================>.............] - ETA: 37s - loss: 7.6953 - accuracy: 0.4981
14464/25000 [================>.............] - ETA: 37s - loss: 7.6974 - accuracy: 0.4980
14496/25000 [================>.............] - ETA: 37s - loss: 7.6973 - accuracy: 0.4980
14528/25000 [================>.............] - ETA: 37s - loss: 7.6930 - accuracy: 0.4983
14560/25000 [================>.............] - ETA: 37s - loss: 7.6961 - accuracy: 0.4981
14592/25000 [================>.............] - ETA: 36s - loss: 7.6981 - accuracy: 0.4979
14624/25000 [================>.............] - ETA: 36s - loss: 7.7012 - accuracy: 0.4977
14656/25000 [================>.............] - ETA: 36s - loss: 7.7001 - accuracy: 0.4978
14688/25000 [================>.............] - ETA: 36s - loss: 7.6979 - accuracy: 0.4980
14720/25000 [================>.............] - ETA: 36s - loss: 7.6979 - accuracy: 0.4980
14752/25000 [================>.............] - ETA: 36s - loss: 7.7009 - accuracy: 0.4978
14784/25000 [================>.............] - ETA: 36s - loss: 7.7019 - accuracy: 0.4977
14816/25000 [================>.............] - ETA: 36s - loss: 7.6997 - accuracy: 0.4978
14848/25000 [================>.............] - ETA: 35s - loss: 7.6966 - accuracy: 0.4980
14880/25000 [================>.............] - ETA: 35s - loss: 7.6996 - accuracy: 0.4978
14912/25000 [================>.............] - ETA: 35s - loss: 7.6985 - accuracy: 0.4979
14944/25000 [================>.............] - ETA: 35s - loss: 7.6974 - accuracy: 0.4980
14976/25000 [================>.............] - ETA: 35s - loss: 7.7025 - accuracy: 0.4977
15008/25000 [=================>............] - ETA: 35s - loss: 7.7034 - accuracy: 0.4976
15040/25000 [=================>............] - ETA: 35s - loss: 7.7033 - accuracy: 0.4976
15072/25000 [=================>............] - ETA: 35s - loss: 7.7043 - accuracy: 0.4975
15104/25000 [=================>............] - ETA: 35s - loss: 7.7052 - accuracy: 0.4975
15136/25000 [=================>............] - ETA: 34s - loss: 7.7061 - accuracy: 0.4974
15168/25000 [=================>............] - ETA: 34s - loss: 7.7060 - accuracy: 0.4974
15200/25000 [=================>............] - ETA: 34s - loss: 7.7110 - accuracy: 0.4971
15232/25000 [=================>............] - ETA: 34s - loss: 7.7119 - accuracy: 0.4970
15264/25000 [=================>............] - ETA: 34s - loss: 7.7088 - accuracy: 0.4972
15296/25000 [=================>............] - ETA: 34s - loss: 7.7127 - accuracy: 0.4970
15328/25000 [=================>............] - ETA: 34s - loss: 7.7116 - accuracy: 0.4971
15360/25000 [=================>............] - ETA: 34s - loss: 7.7085 - accuracy: 0.4973
15392/25000 [=================>............] - ETA: 34s - loss: 7.7055 - accuracy: 0.4975
15424/25000 [=================>............] - ETA: 33s - loss: 7.7014 - accuracy: 0.4977
15456/25000 [=================>............] - ETA: 33s - loss: 7.7033 - accuracy: 0.4976
15488/25000 [=================>............] - ETA: 33s - loss: 7.7032 - accuracy: 0.4976
15520/25000 [=================>............] - ETA: 33s - loss: 7.7042 - accuracy: 0.4976
15552/25000 [=================>............] - ETA: 33s - loss: 7.7021 - accuracy: 0.4977
15584/25000 [=================>............] - ETA: 33s - loss: 7.7040 - accuracy: 0.4976
15616/25000 [=================>............] - ETA: 33s - loss: 7.7000 - accuracy: 0.4978
15648/25000 [=================>............] - ETA: 33s - loss: 7.6990 - accuracy: 0.4979
15680/25000 [=================>............] - ETA: 33s - loss: 7.6999 - accuracy: 0.4978
15712/25000 [=================>............] - ETA: 32s - loss: 7.6978 - accuracy: 0.4980
15744/25000 [=================>............] - ETA: 32s - loss: 7.6939 - accuracy: 0.4982
15776/25000 [=================>............] - ETA: 32s - loss: 7.6870 - accuracy: 0.4987
15808/25000 [=================>............] - ETA: 32s - loss: 7.6880 - accuracy: 0.4986
15840/25000 [==================>...........] - ETA: 32s - loss: 7.6899 - accuracy: 0.4985
15872/25000 [==================>...........] - ETA: 32s - loss: 7.6869 - accuracy: 0.4987
15904/25000 [==================>...........] - ETA: 32s - loss: 7.6898 - accuracy: 0.4985
15936/25000 [==================>...........] - ETA: 32s - loss: 7.6907 - accuracy: 0.4984
15968/25000 [==================>...........] - ETA: 32s - loss: 7.6916 - accuracy: 0.4984
16000/25000 [==================>...........] - ETA: 31s - loss: 7.6906 - accuracy: 0.4984
16032/25000 [==================>...........] - ETA: 31s - loss: 7.6924 - accuracy: 0.4983
16064/25000 [==================>...........] - ETA: 31s - loss: 7.6933 - accuracy: 0.4983
16096/25000 [==================>...........] - ETA: 31s - loss: 7.6933 - accuracy: 0.4983
16128/25000 [==================>...........] - ETA: 31s - loss: 7.6942 - accuracy: 0.4982
16160/25000 [==================>...........] - ETA: 31s - loss: 7.6932 - accuracy: 0.4983
16192/25000 [==================>...........] - ETA: 31s - loss: 7.6960 - accuracy: 0.4981
16224/25000 [==================>...........] - ETA: 31s - loss: 7.6969 - accuracy: 0.4980
16256/25000 [==================>...........] - ETA: 30s - loss: 7.6977 - accuracy: 0.4980
16288/25000 [==================>...........] - ETA: 30s - loss: 7.7043 - accuracy: 0.4975
16320/25000 [==================>...........] - ETA: 30s - loss: 7.7014 - accuracy: 0.4977
16352/25000 [==================>...........] - ETA: 30s - loss: 7.7069 - accuracy: 0.4974
16384/25000 [==================>...........] - ETA: 30s - loss: 7.7069 - accuracy: 0.4974
16416/25000 [==================>...........] - ETA: 30s - loss: 7.7077 - accuracy: 0.4973
16448/25000 [==================>...........] - ETA: 30s - loss: 7.7104 - accuracy: 0.4971
16480/25000 [==================>...........] - ETA: 30s - loss: 7.7131 - accuracy: 0.4970
16512/25000 [==================>...........] - ETA: 30s - loss: 7.7093 - accuracy: 0.4972
16544/25000 [==================>...........] - ETA: 29s - loss: 7.7083 - accuracy: 0.4973
16576/25000 [==================>...........] - ETA: 29s - loss: 7.7082 - accuracy: 0.4973
16608/25000 [==================>...........] - ETA: 29s - loss: 7.7045 - accuracy: 0.4975
16640/25000 [==================>...........] - ETA: 29s - loss: 7.7090 - accuracy: 0.4972
16672/25000 [===================>..........] - ETA: 29s - loss: 7.7154 - accuracy: 0.4968
16704/25000 [===================>..........] - ETA: 29s - loss: 7.7125 - accuracy: 0.4970
16736/25000 [===================>..........] - ETA: 29s - loss: 7.7115 - accuracy: 0.4971
16768/25000 [===================>..........] - ETA: 29s - loss: 7.7133 - accuracy: 0.4970
16800/25000 [===================>..........] - ETA: 29s - loss: 7.7132 - accuracy: 0.4970
16832/25000 [===================>..........] - ETA: 28s - loss: 7.7149 - accuracy: 0.4969
16864/25000 [===================>..........] - ETA: 28s - loss: 7.7121 - accuracy: 0.4970
16896/25000 [===================>..........] - ETA: 28s - loss: 7.7120 - accuracy: 0.4970
16928/25000 [===================>..........] - ETA: 28s - loss: 7.7137 - accuracy: 0.4969
16960/25000 [===================>..........] - ETA: 28s - loss: 7.7145 - accuracy: 0.4969
16992/25000 [===================>..........] - ETA: 28s - loss: 7.7190 - accuracy: 0.4966
17024/25000 [===================>..........] - ETA: 28s - loss: 7.7117 - accuracy: 0.4971
17056/25000 [===================>..........] - ETA: 28s - loss: 7.7134 - accuracy: 0.4970
17088/25000 [===================>..........] - ETA: 28s - loss: 7.7133 - accuracy: 0.4970
17120/25000 [===================>..........] - ETA: 27s - loss: 7.7078 - accuracy: 0.4973
17152/25000 [===================>..........] - ETA: 27s - loss: 7.7086 - accuracy: 0.4973
17184/25000 [===================>..........] - ETA: 27s - loss: 7.7139 - accuracy: 0.4969
17216/25000 [===================>..........] - ETA: 27s - loss: 7.7183 - accuracy: 0.4966
17248/25000 [===================>..........] - ETA: 27s - loss: 7.7146 - accuracy: 0.4969
17280/25000 [===================>..........] - ETA: 27s - loss: 7.7163 - accuracy: 0.4968
17312/25000 [===================>..........] - ETA: 27s - loss: 7.7144 - accuracy: 0.4969
17344/25000 [===================>..........] - ETA: 27s - loss: 7.7117 - accuracy: 0.4971
17376/25000 [===================>..........] - ETA: 27s - loss: 7.7090 - accuracy: 0.4972
17408/25000 [===================>..........] - ETA: 26s - loss: 7.7107 - accuracy: 0.4971
17440/25000 [===================>..........] - ETA: 26s - loss: 7.7132 - accuracy: 0.4970
17472/25000 [===================>..........] - ETA: 26s - loss: 7.7149 - accuracy: 0.4969
17504/25000 [====================>.........] - ETA: 26s - loss: 7.7113 - accuracy: 0.4971
17536/25000 [====================>.........] - ETA: 26s - loss: 7.7112 - accuracy: 0.4971
17568/25000 [====================>.........] - ETA: 26s - loss: 7.7111 - accuracy: 0.4971
17600/25000 [====================>.........] - ETA: 26s - loss: 7.7163 - accuracy: 0.4968
17632/25000 [====================>.........] - ETA: 26s - loss: 7.7144 - accuracy: 0.4969
17664/25000 [====================>.........] - ETA: 26s - loss: 7.7152 - accuracy: 0.4968
17696/25000 [====================>.........] - ETA: 25s - loss: 7.7160 - accuracy: 0.4968
17728/25000 [====================>.........] - ETA: 25s - loss: 7.7107 - accuracy: 0.4971
17760/25000 [====================>.........] - ETA: 25s - loss: 7.7115 - accuracy: 0.4971
17792/25000 [====================>.........] - ETA: 25s - loss: 7.7088 - accuracy: 0.4972
17824/25000 [====================>.........] - ETA: 25s - loss: 7.7096 - accuracy: 0.4972
17856/25000 [====================>.........] - ETA: 25s - loss: 7.7061 - accuracy: 0.4974
17888/25000 [====================>.........] - ETA: 25s - loss: 7.7069 - accuracy: 0.4974
17920/25000 [====================>.........] - ETA: 25s - loss: 7.7085 - accuracy: 0.4973
17952/25000 [====================>.........] - ETA: 24s - loss: 7.7076 - accuracy: 0.4973
17984/25000 [====================>.........] - ETA: 24s - loss: 7.7058 - accuracy: 0.4974
18016/25000 [====================>.........] - ETA: 24s - loss: 7.7075 - accuracy: 0.4973
18048/25000 [====================>.........] - ETA: 24s - loss: 7.7091 - accuracy: 0.4972
18080/25000 [====================>.........] - ETA: 24s - loss: 7.7065 - accuracy: 0.4974
18112/25000 [====================>.........] - ETA: 24s - loss: 7.7073 - accuracy: 0.4973
18144/25000 [====================>.........] - ETA: 24s - loss: 7.7114 - accuracy: 0.4971
18176/25000 [====================>.........] - ETA: 24s - loss: 7.7096 - accuracy: 0.4972
18208/25000 [====================>.........] - ETA: 24s - loss: 7.7087 - accuracy: 0.4973
18240/25000 [====================>.........] - ETA: 23s - loss: 7.7112 - accuracy: 0.4971
18272/25000 [====================>.........] - ETA: 23s - loss: 7.7094 - accuracy: 0.4972
18304/25000 [====================>.........] - ETA: 23s - loss: 7.7102 - accuracy: 0.4972
18336/25000 [=====================>........] - ETA: 23s - loss: 7.7126 - accuracy: 0.4970
18368/25000 [=====================>........] - ETA: 23s - loss: 7.7134 - accuracy: 0.4970
18400/25000 [=====================>........] - ETA: 23s - loss: 7.7100 - accuracy: 0.4972
18432/25000 [=====================>........] - ETA: 23s - loss: 7.7107 - accuracy: 0.4971
18464/25000 [=====================>........] - ETA: 23s - loss: 7.7081 - accuracy: 0.4973
18496/25000 [=====================>........] - ETA: 23s - loss: 7.7081 - accuracy: 0.4973
18528/25000 [=====================>........] - ETA: 22s - loss: 7.7080 - accuracy: 0.4973
18560/25000 [=====================>........] - ETA: 22s - loss: 7.7096 - accuracy: 0.4972
18592/25000 [=====================>........] - ETA: 22s - loss: 7.7120 - accuracy: 0.4970
18624/25000 [=====================>........] - ETA: 22s - loss: 7.7111 - accuracy: 0.4971
18656/25000 [=====================>........] - ETA: 22s - loss: 7.7143 - accuracy: 0.4969
18688/25000 [=====================>........] - ETA: 22s - loss: 7.7134 - accuracy: 0.4969
18720/25000 [=====================>........] - ETA: 22s - loss: 7.7092 - accuracy: 0.4972
18752/25000 [=====================>........] - ETA: 22s - loss: 7.7091 - accuracy: 0.4972
18784/25000 [=====================>........] - ETA: 22s - loss: 7.7107 - accuracy: 0.4971
18816/25000 [=====================>........] - ETA: 21s - loss: 7.7098 - accuracy: 0.4972
18848/25000 [=====================>........] - ETA: 21s - loss: 7.7081 - accuracy: 0.4973
18880/25000 [=====================>........] - ETA: 21s - loss: 7.7072 - accuracy: 0.4974
18912/25000 [=====================>........] - ETA: 21s - loss: 7.7120 - accuracy: 0.4970
18944/25000 [=====================>........] - ETA: 21s - loss: 7.7136 - accuracy: 0.4969
18976/25000 [=====================>........] - ETA: 21s - loss: 7.7127 - accuracy: 0.4970
19008/25000 [=====================>........] - ETA: 21s - loss: 7.7102 - accuracy: 0.4972
19040/25000 [=====================>........] - ETA: 21s - loss: 7.7117 - accuracy: 0.4971
19072/25000 [=====================>........] - ETA: 20s - loss: 7.7100 - accuracy: 0.4972
19104/25000 [=====================>........] - ETA: 20s - loss: 7.7124 - accuracy: 0.4970
19136/25000 [=====================>........] - ETA: 20s - loss: 7.7099 - accuracy: 0.4972
19168/25000 [======================>.......] - ETA: 20s - loss: 7.7098 - accuracy: 0.4972
19200/25000 [======================>.......] - ETA: 20s - loss: 7.7137 - accuracy: 0.4969
19232/25000 [======================>.......] - ETA: 20s - loss: 7.7113 - accuracy: 0.4971
19264/25000 [======================>.......] - ETA: 20s - loss: 7.7120 - accuracy: 0.4970
19296/25000 [======================>.......] - ETA: 20s - loss: 7.7135 - accuracy: 0.4969
19328/25000 [======================>.......] - ETA: 20s - loss: 7.7095 - accuracy: 0.4972
19360/25000 [======================>.......] - ETA: 19s - loss: 7.7062 - accuracy: 0.4974
19392/25000 [======================>.......] - ETA: 19s - loss: 7.7054 - accuracy: 0.4975
19424/25000 [======================>.......] - ETA: 19s - loss: 7.7029 - accuracy: 0.4976
19456/25000 [======================>.......] - ETA: 19s - loss: 7.7013 - accuracy: 0.4977
19488/25000 [======================>.......] - ETA: 19s - loss: 7.6997 - accuracy: 0.4978
19520/25000 [======================>.......] - ETA: 19s - loss: 7.7028 - accuracy: 0.4976
19552/25000 [======================>.......] - ETA: 19s - loss: 7.7050 - accuracy: 0.4975
19584/25000 [======================>.......] - ETA: 19s - loss: 7.7050 - accuracy: 0.4975
19616/25000 [======================>.......] - ETA: 19s - loss: 7.7049 - accuracy: 0.4975
19648/25000 [======================>.......] - ETA: 18s - loss: 7.7017 - accuracy: 0.4977
19680/25000 [======================>.......] - ETA: 18s - loss: 7.7032 - accuracy: 0.4976
19712/25000 [======================>.......] - ETA: 18s - loss: 7.7001 - accuracy: 0.4978
19744/25000 [======================>.......] - ETA: 18s - loss: 7.6992 - accuracy: 0.4979
19776/25000 [======================>.......] - ETA: 18s - loss: 7.7023 - accuracy: 0.4977
19808/25000 [======================>.......] - ETA: 18s - loss: 7.7038 - accuracy: 0.4976
19840/25000 [======================>.......] - ETA: 18s - loss: 7.7045 - accuracy: 0.4975
19872/25000 [======================>.......] - ETA: 18s - loss: 7.7060 - accuracy: 0.4974
19904/25000 [======================>.......] - ETA: 18s - loss: 7.7051 - accuracy: 0.4975
19936/25000 [======================>.......] - ETA: 17s - loss: 7.7035 - accuracy: 0.4976
19968/25000 [======================>.......] - ETA: 17s - loss: 7.7027 - accuracy: 0.4976
20000/25000 [=======================>......] - ETA: 17s - loss: 7.7042 - accuracy: 0.4976
20032/25000 [=======================>......] - ETA: 17s - loss: 7.7018 - accuracy: 0.4977
20064/25000 [=======================>......] - ETA: 17s - loss: 7.7025 - accuracy: 0.4977
20096/25000 [=======================>......] - ETA: 17s - loss: 7.7032 - accuracy: 0.4976
20128/25000 [=======================>......] - ETA: 17s - loss: 7.7001 - accuracy: 0.4978
20160/25000 [=======================>......] - ETA: 17s - loss: 7.6963 - accuracy: 0.4981
20192/25000 [=======================>......] - ETA: 17s - loss: 7.6962 - accuracy: 0.4981
20224/25000 [=======================>......] - ETA: 16s - loss: 7.6977 - accuracy: 0.4980
20256/25000 [=======================>......] - ETA: 16s - loss: 7.6984 - accuracy: 0.4979
20288/25000 [=======================>......] - ETA: 16s - loss: 7.7006 - accuracy: 0.4978
20320/25000 [=======================>......] - ETA: 16s - loss: 7.6983 - accuracy: 0.4979
20352/25000 [=======================>......] - ETA: 16s - loss: 7.6983 - accuracy: 0.4979
20384/25000 [=======================>......] - ETA: 16s - loss: 7.6982 - accuracy: 0.4979
20416/25000 [=======================>......] - ETA: 16s - loss: 7.6959 - accuracy: 0.4981
20448/25000 [=======================>......] - ETA: 16s - loss: 7.6981 - accuracy: 0.4979
20480/25000 [=======================>......] - ETA: 15s - loss: 7.6966 - accuracy: 0.4980
20512/25000 [=======================>......] - ETA: 15s - loss: 7.6928 - accuracy: 0.4983
20544/25000 [=======================>......] - ETA: 15s - loss: 7.6920 - accuracy: 0.4983
20576/25000 [=======================>......] - ETA: 15s - loss: 7.6942 - accuracy: 0.4982
20608/25000 [=======================>......] - ETA: 15s - loss: 7.6919 - accuracy: 0.4984
20640/25000 [=======================>......] - ETA: 15s - loss: 7.6926 - accuracy: 0.4983
20672/25000 [=======================>......] - ETA: 15s - loss: 7.6911 - accuracy: 0.4984
20704/25000 [=======================>......] - ETA: 15s - loss: 7.6911 - accuracy: 0.4984
20736/25000 [=======================>......] - ETA: 15s - loss: 7.6888 - accuracy: 0.4986
20768/25000 [=======================>......] - ETA: 14s - loss: 7.6880 - accuracy: 0.4986
20800/25000 [=======================>......] - ETA: 14s - loss: 7.6917 - accuracy: 0.4984
20832/25000 [=======================>......] - ETA: 14s - loss: 7.6931 - accuracy: 0.4983
20864/25000 [========================>.....] - ETA: 14s - loss: 7.6909 - accuracy: 0.4984
20896/25000 [========================>.....] - ETA: 14s - loss: 7.6864 - accuracy: 0.4987
20928/25000 [========================>.....] - ETA: 14s - loss: 7.6820 - accuracy: 0.4990
20960/25000 [========================>.....] - ETA: 14s - loss: 7.6805 - accuracy: 0.4991
20992/25000 [========================>.....] - ETA: 14s - loss: 7.6783 - accuracy: 0.4992
21024/25000 [========================>.....] - ETA: 14s - loss: 7.6776 - accuracy: 0.4993
21056/25000 [========================>.....] - ETA: 13s - loss: 7.6783 - accuracy: 0.4992
21088/25000 [========================>.....] - ETA: 13s - loss: 7.6812 - accuracy: 0.4991
21120/25000 [========================>.....] - ETA: 13s - loss: 7.6811 - accuracy: 0.4991
21152/25000 [========================>.....] - ETA: 13s - loss: 7.6782 - accuracy: 0.4992
21184/25000 [========================>.....] - ETA: 13s - loss: 7.6796 - accuracy: 0.4992
21216/25000 [========================>.....] - ETA: 13s - loss: 7.6804 - accuracy: 0.4991
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6803 - accuracy: 0.4991
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6810 - accuracy: 0.4991
21312/25000 [========================>.....] - ETA: 13s - loss: 7.6810 - accuracy: 0.4991
21344/25000 [========================>.....] - ETA: 12s - loss: 7.6817 - accuracy: 0.4990
21376/25000 [========================>.....] - ETA: 12s - loss: 7.6817 - accuracy: 0.4990
21408/25000 [========================>.....] - ETA: 12s - loss: 7.6831 - accuracy: 0.4989
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6838 - accuracy: 0.4989
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6852 - accuracy: 0.4988
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6887 - accuracy: 0.4986
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6887 - accuracy: 0.4986
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6908 - accuracy: 0.4984
21600/25000 [========================>.....] - ETA: 12s - loss: 7.6915 - accuracy: 0.4984
21632/25000 [========================>.....] - ETA: 11s - loss: 7.6943 - accuracy: 0.4982
21664/25000 [========================>.....] - ETA: 11s - loss: 7.6978 - accuracy: 0.4980
21696/25000 [=========================>....] - ETA: 11s - loss: 7.6998 - accuracy: 0.4978
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6984 - accuracy: 0.4979
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6983 - accuracy: 0.4979
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6997 - accuracy: 0.4978
21824/25000 [=========================>....] - ETA: 11s - loss: 7.7010 - accuracy: 0.4978
21856/25000 [=========================>....] - ETA: 11s - loss: 7.7003 - accuracy: 0.4978
21888/25000 [=========================>....] - ETA: 11s - loss: 7.6981 - accuracy: 0.4979
21920/25000 [=========================>....] - ETA: 10s - loss: 7.6981 - accuracy: 0.4979
21952/25000 [=========================>....] - ETA: 10s - loss: 7.6974 - accuracy: 0.4980
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6987 - accuracy: 0.4979
22016/25000 [=========================>....] - ETA: 10s - loss: 7.7021 - accuracy: 0.4977
22048/25000 [=========================>....] - ETA: 10s - loss: 7.7014 - accuracy: 0.4977
22080/25000 [=========================>....] - ETA: 10s - loss: 7.7020 - accuracy: 0.4977
22112/25000 [=========================>....] - ETA: 10s - loss: 7.7006 - accuracy: 0.4978
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6999 - accuracy: 0.4978
22176/25000 [=========================>....] - ETA: 9s - loss: 7.6977 - accuracy: 0.4980 
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6984 - accuracy: 0.4979
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6963 - accuracy: 0.4981
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6928 - accuracy: 0.4983
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6934 - accuracy: 0.4983
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6941 - accuracy: 0.4982
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6934 - accuracy: 0.4983
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6919 - accuracy: 0.4983
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6912 - accuracy: 0.4984
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6926 - accuracy: 0.4983
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6939 - accuracy: 0.4982
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6966 - accuracy: 0.4980
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6958 - accuracy: 0.4981
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6944 - accuracy: 0.4982
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6924 - accuracy: 0.4983
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6917 - accuracy: 0.4984
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6930 - accuracy: 0.4983
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6936 - accuracy: 0.4982
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6963 - accuracy: 0.4981
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6982 - accuracy: 0.4979
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6989 - accuracy: 0.4979
22848/25000 [==========================>...] - ETA: 7s - loss: 7.7002 - accuracy: 0.4978
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6981 - accuracy: 0.4979
22912/25000 [==========================>...] - ETA: 7s - loss: 7.7007 - accuracy: 0.4978
22944/25000 [==========================>...] - ETA: 7s - loss: 7.7034 - accuracy: 0.4976
22976/25000 [==========================>...] - ETA: 7s - loss: 7.7027 - accuracy: 0.4976
23008/25000 [==========================>...] - ETA: 7s - loss: 7.7019 - accuracy: 0.4977
23040/25000 [==========================>...] - ETA: 6s - loss: 7.7026 - accuracy: 0.4977
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6979 - accuracy: 0.4980
23104/25000 [==========================>...] - ETA: 6s - loss: 7.7005 - accuracy: 0.4978
23136/25000 [==========================>...] - ETA: 6s - loss: 7.7011 - accuracy: 0.4978
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6997 - accuracy: 0.4978
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6977 - accuracy: 0.4980
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6943 - accuracy: 0.4982
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6917 - accuracy: 0.4984
23296/25000 [==========================>...] - ETA: 6s - loss: 7.6916 - accuracy: 0.4984
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6949 - accuracy: 0.4982
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6948 - accuracy: 0.4982
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6954 - accuracy: 0.4981
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6940 - accuracy: 0.4982
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6947 - accuracy: 0.4982
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6940 - accuracy: 0.4982
23584/25000 [===========================>..] - ETA: 5s - loss: 7.6959 - accuracy: 0.4981
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6984 - accuracy: 0.4979
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6977 - accuracy: 0.4980
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6990 - accuracy: 0.4979
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6977 - accuracy: 0.4980
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6996 - accuracy: 0.4979
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6976 - accuracy: 0.4980
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6962 - accuracy: 0.4981
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6968 - accuracy: 0.4980
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6949 - accuracy: 0.4982
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6923 - accuracy: 0.4983
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6910 - accuracy: 0.4984
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6922 - accuracy: 0.4983
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6909 - accuracy: 0.4984
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6915 - accuracy: 0.4984
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6902 - accuracy: 0.4985
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6902 - accuracy: 0.4985
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6889 - accuracy: 0.4985
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6882 - accuracy: 0.4986
24192/25000 [============================>.] - ETA: 2s - loss: 7.6882 - accuracy: 0.4986
24224/25000 [============================>.] - ETA: 2s - loss: 7.6894 - accuracy: 0.4985
24256/25000 [============================>.] - ETA: 2s - loss: 7.6887 - accuracy: 0.4986
24288/25000 [============================>.] - ETA: 2s - loss: 7.6868 - accuracy: 0.4987
24320/25000 [============================>.] - ETA: 2s - loss: 7.6824 - accuracy: 0.4990
24352/25000 [============================>.] - ETA: 2s - loss: 7.6805 - accuracy: 0.4991
24384/25000 [============================>.] - ETA: 2s - loss: 7.6798 - accuracy: 0.4991
24416/25000 [============================>.] - ETA: 2s - loss: 7.6811 - accuracy: 0.4991
24448/25000 [============================>.] - ETA: 1s - loss: 7.6817 - accuracy: 0.4990
24480/25000 [============================>.] - ETA: 1s - loss: 7.6804 - accuracy: 0.4991
24512/25000 [============================>.] - ETA: 1s - loss: 7.6791 - accuracy: 0.4992
24544/25000 [============================>.] - ETA: 1s - loss: 7.6785 - accuracy: 0.4992
24576/25000 [============================>.] - ETA: 1s - loss: 7.6760 - accuracy: 0.4994
24608/25000 [============================>.] - ETA: 1s - loss: 7.6753 - accuracy: 0.4994
24640/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
24672/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24704/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24736/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24768/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24800/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24832/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24864/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24896/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24928/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
24992/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
25000/25000 [==============================] - 106s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
Loading data...
Using TensorFlow backend.





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
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//mnist_mlmodels_.ipynb 

[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.ipynb[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mcolab[0m [0;32mimport[0m [0mdrive[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdrive[0m[0;34m.[0m[0mmount[0m[0;34m([0m[0;34m'/content/drive'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'google.colab'





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
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//vision_mnist.py 

[0;36m  File [0;32m"/home/runner/work/mlmodels/mlmodels/mlmodels/example/vision_mnist.py"[0;36m, line [0;32m15[0m
[0;31m    !git clone https://github.com/ahmed3bbas/mlmodels.git[0m
[0m    ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax






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
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//benchmark_timeseries_m5.py 

[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py[0m in [0;36m<module>[0;34m[0m
[1;32m     84[0m [0;34m[0m[0m
[1;32m     85[0m """
[0;32m---> 86[0;31m [0mcalendar[0m               [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/calendar.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     87[0m [0msales_train_val[0m        [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sales_train_val.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     88[0m [0msample_submission[0m      [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sample_submission.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36mparser_f[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)[0m
[1;32m    683[0m         )
[1;32m    684[0m [0;34m[0m[0m
[0;32m--> 685[0;31m         [0;32mreturn[0m [0m_read[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    686[0m [0;34m[0m[0m
[1;32m    687[0m     [0mparser_f[0m[0;34m.[0m[0m__name__[0m [0;34m=[0m [0mname[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m_read[0;34m(filepath_or_buffer, kwds)[0m
[1;32m    455[0m [0;34m[0m[0m
[1;32m    456[0m     [0;31m# Create the parser.[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 457[0;31m     [0mparser[0m [0;34m=[0m [0mTextFileReader[0m[0;34m([0m[0mfp_or_buf[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    458[0m [0;34m[0m[0m
[1;32m    459[0m     [0;32mif[0m [0mchunksize[0m [0;32mor[0m [0miterator[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m__init__[0;34m(self, f, engine, **kwds)[0m
[1;32m    893[0m             [0mself[0m[0;34m.[0m[0moptions[0m[0;34m[[0m[0;34m"has_index_names"[0m[0;34m][0m [0;34m=[0m [0mkwds[0m[0;34m[[0m[0;34m"has_index_names"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    894[0m [0;34m[0m[0m
[0;32m--> 895[0;31m         [0mself[0m[0;34m.[0m[0m_make_engine[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mengine[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    896[0m [0;34m[0m[0m
[1;32m    897[0m     [0;32mdef[0m [0mclose[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m_make_engine[0;34m(self, engine)[0m
[1;32m   1133[0m     [0;32mdef[0m [0m_make_engine[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mengine[0m[0;34m=[0m[0;34m"c"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1134[0m         [0;32mif[0m [0mengine[0m [0;34m==[0m [0;34m"c"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1135[0;31m             [0mself[0m[0;34m.[0m[0m_engine[0m [0;34m=[0m [0mCParserWrapper[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mf[0m[0;34m,[0m [0;34m**[0m[0mself[0m[0;34m.[0m[0moptions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1136[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1137[0m             [0;32mif[0m [0mengine[0m [0;34m==[0m [0;34m"python"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m__init__[0;34m(self, src, **kwds)[0m
[1;32m   1915[0m         [0mkwds[0m[0;34m[[0m[0;34m"usecols"[0m[0;34m][0m [0;34m=[0m [0mself[0m[0;34m.[0m[0musecols[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1916[0m [0;34m[0m[0m
[0;32m-> 1917[0;31m         [0mself[0m[0;34m.[0m[0m_reader[0m [0;34m=[0m [0mparsers[0m[0;34m.[0m[0mTextReader[0m[0;34m([0m[0msrc[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1918[0m         [0mself[0m[0;34m.[0m[0munnamed_cols[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_reader[0m[0;34m.[0m[0munnamed_cols[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1919[0m [0;34m[0m[0m

[0;32mpandas/_libs/parsers.pyx[0m in [0;36mpandas._libs.parsers.TextReader.__cinit__[0;34m()[0m

[0;32mpandas/_libs/parsers.pyx[0m in [0;36mpandas._libs.parsers.TextReader._setup_parser_source[0;34m()[0m

[0;31mFileNotFoundError[0m: [Errno 2] File b'./m5-forecasting-accuracy/calendar.csv' does not exist: b'./m5-forecasting-accuracy/calendar.csv'





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
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/benchmark_timeseries_m4.py 






 ************************************************************************************************************************
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example/benchmark_timeseries_m5.py 

[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m~/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py[0m in [0;36m<module>[0;34m[0m
[1;32m     84[0m [0;34m[0m[0m
[1;32m     85[0m """
[0;32m---> 86[0;31m [0mcalendar[0m               [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/calendar.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     87[0m [0msales_train_val[0m        [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sales_train_val.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     88[0m [0msample_submission[0m      [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sample_submission.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36mparser_f[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)[0m
[1;32m    683[0m         )
[1;32m    684[0m [0;34m[0m[0m
[0;32m--> 685[0;31m         [0;32mreturn[0m [0m_read[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    686[0m [0;34m[0m[0m
[1;32m    687[0m     [0mparser_f[0m[0;34m.[0m[0m__name__[0m [0;34m=[0m [0mname[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m_read[0;34m(filepath_or_buffer, kwds)[0m
[1;32m    455[0m [0;34m[0m[0m
[1;32m    456[0m     [0;31m# Create the parser.[0m[0;34m[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 457[0;31m     [0mparser[0m [0;34m=[0m [0mTextFileReader[0m[0;34m([0m[0mfp_or_buf[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    458[0m [0;34m[0m[0m
[1;32m    459[0m     [0;32mif[0m [0mchunksize[0m [0;32mor[0m [0miterator[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m__init__[0;34m(self, f, engine, **kwds)[0m
[1;32m    893[0m             [0mself[0m[0;34m.[0m[0moptions[0m[0;34m[[0m[0;34m"has_index_names"[0m[0;34m][0m [0;34m=[0m [0mkwds[0m[0;34m[[0m[0;34m"has_index_names"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    894[0m [0;34m[0m[0m
[0;32m--> 895[0;31m         [0mself[0m[0;34m.[0m[0m_make_engine[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mengine[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    896[0m [0;34m[0m[0m
[1;32m    897[0m     [0;32mdef[0m [0mclose[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m_make_engine[0;34m(self, engine)[0m
[1;32m   1133[0m     [0;32mdef[0m [0m_make_engine[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mengine[0m[0;34m=[0m[0;34m"c"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1134[0m         [0;32mif[0m [0mengine[0m [0;34m==[0m [0;34m"c"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1135[0;31m             [0mself[0m[0;34m.[0m[0m_engine[0m [0;34m=[0m [0mCParserWrapper[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mf[0m[0;34m,[0m [0;34m**[0m[0mself[0m[0;34m.[0m[0moptions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1136[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1137[0m             [0;32mif[0m [0mengine[0m [0;34m==[0m [0;34m"python"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py[0m in [0;36m__init__[0;34m(self, src, **kwds)[0m
[1;32m   1915[0m         [0mkwds[0m[0;34m[[0m[0;34m"usecols"[0m[0;34m][0m [0;34m=[0m [0mself[0m[0;34m.[0m[0musecols[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1916[0m [0;34m[0m[0m
[0;32m-> 1917[0;31m         [0mself[0m[0;34m.[0m[0m_reader[0m [0;34m=[0m [0mparsers[0m[0;34m.[0m[0mTextReader[0m[0;34m([0m[0msrc[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1918[0m         [0mself[0m[0;34m.[0m[0munnamed_cols[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_reader[0m[0;34m.[0m[0munnamed_cols[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1919[0m [0;34m[0m[0m

[0;32mpandas/_libs/parsers.pyx[0m in [0;36mpandas._libs.parsers.TextReader.__cinit__[0;34m()[0m

[0;32mpandas/_libs/parsers.pyx[0m in [0;36mpandas._libs.parsers.TextReader._setup_parser_source[0;34m()[0m

[0;31mFileNotFoundError[0m: [Errno 2] File b'./m5-forecasting-accuracy/calendar.csv' does not exist: b'./m5-forecasting-accuracy/calendar.csv'
