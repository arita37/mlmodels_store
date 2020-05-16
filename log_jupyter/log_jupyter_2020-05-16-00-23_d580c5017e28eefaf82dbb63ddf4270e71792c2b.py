
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
	Data preprocessing and feature engineering runtime = 0.25s ...
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:57<01:25, 28.51s/it] 40%|████      | 2/5 [00:57<01:25, 28.51s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.10009393133827163, 'embedding_size_factor': 0.8646102668477016, 'layers.choice': 1, 'learning_rate': 0.003620816062460028, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 3.842327773894204e-06} and reward: 0.3678
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x9f\xc1\x81\xa0>\x02X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeb\xaa\xe3&|\xb1\xdcX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?m\xa9f\xd2V\xd0-X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xd0\x1d\xab\x04*\xfb!u.' and reward: 0.3678
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x9f\xc1\x81\xa0>\x02X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeb\xaa\xe3&|\xb1\xdcX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?m\xa9f\xd2V\xd0-X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xd0\x1d\xab\x04*\xfb!u.' and reward: 0.3678
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 156.90940308570862
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -39.02s of remaining time.
Ensemble size: 1
Ensemble weights: 
[1. 0.]
	0.3862	 = Validation accuracy score
	1.01s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 160.07s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f2813bc8a90> 

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
 [ 0.06098526 -0.07736425  0.07599431  0.16009897 -0.13507058  0.15489726]
 [ 0.13042703  0.05666394  0.08185366  0.00787225 -0.16237579  0.04723072]
 [ 0.0306235   0.21829404  0.0310623   0.09234591 -0.08506846  0.01196415]
 [ 0.14348523  0.08995964  0.03140502 -0.04163222  0.05471626 -0.06196846]
 [-0.45861769  0.66748005  0.16865461  0.15147603 -0.17979701  0.05004335]
 [ 0.52660453  0.12559339  0.27088335  0.39385954  0.39230904  0.86018395]
 [-0.06452762  0.74946386  0.21090591  0.51628798  0.34078336 -0.69248611]
 [-0.40593162  0.44347453  0.46567658  0.25386256  0.00949523  0.05278393]
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
{'loss': 0.4162355624139309, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-16 00:27:05.184432: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.5097556933760643, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-16 00:27:06.476503: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 2547712/17464789 [===>..........................] - ETA: 0s
 8962048/17464789 [==============>...............] - ETA: 0s
16490496/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-16 00:27:19.727271: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 00:27:19.731944: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-16 00:27:19.732188: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559cf8cc1760 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 00:27:19.732208: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:55 - loss: 6.2291 - accuracy: 0.5938
   64/25000 [..............................] - ETA: 3:13 - loss: 8.3854 - accuracy: 0.4531
   96/25000 [..............................] - ETA: 2:37 - loss: 8.1458 - accuracy: 0.4688
  128/25000 [..............................] - ETA: 2:19 - loss: 7.6666 - accuracy: 0.5000
  160/25000 [..............................] - ETA: 2:09 - loss: 7.9541 - accuracy: 0.4812
  192/25000 [..............................] - ETA: 2:03 - loss: 7.6666 - accuracy: 0.5000
  224/25000 [..............................] - ETA: 1:59 - loss: 8.0773 - accuracy: 0.4732
  256/25000 [..............................] - ETA: 1:55 - loss: 8.3255 - accuracy: 0.4570
  288/25000 [..............................] - ETA: 1:52 - loss: 8.3587 - accuracy: 0.4549
  320/25000 [..............................] - ETA: 1:50 - loss: 8.1458 - accuracy: 0.4688
  352/25000 [..............................] - ETA: 1:48 - loss: 8.1458 - accuracy: 0.4688
  384/25000 [..............................] - ETA: 1:46 - loss: 8.1857 - accuracy: 0.4661
  416/25000 [..............................] - ETA: 1:44 - loss: 8.2195 - accuracy: 0.4639
  448/25000 [..............................] - ETA: 1:43 - loss: 8.1458 - accuracy: 0.4688
  480/25000 [..............................] - ETA: 1:42 - loss: 8.2736 - accuracy: 0.4604
  512/25000 [..............................] - ETA: 1:41 - loss: 8.2955 - accuracy: 0.4590
  544/25000 [..............................] - ETA: 1:40 - loss: 8.2867 - accuracy: 0.4596
  576/25000 [..............................] - ETA: 1:39 - loss: 8.1458 - accuracy: 0.4688
  608/25000 [..............................] - ETA: 1:38 - loss: 8.0197 - accuracy: 0.4770
  640/25000 [..............................] - ETA: 1:38 - loss: 7.9781 - accuracy: 0.4797
  672/25000 [..............................] - ETA: 1:37 - loss: 7.8948 - accuracy: 0.4851
  704/25000 [..............................] - ETA: 1:36 - loss: 7.8626 - accuracy: 0.4872
  736/25000 [..............................] - ETA: 1:36 - loss: 7.9375 - accuracy: 0.4823
  768/25000 [..............................] - ETA: 1:35 - loss: 7.8663 - accuracy: 0.4870
  800/25000 [..............................] - ETA: 1:35 - loss: 7.8008 - accuracy: 0.4913
  832/25000 [..............................] - ETA: 1:34 - loss: 7.7403 - accuracy: 0.4952
  864/25000 [>.............................] - ETA: 1:34 - loss: 7.7021 - accuracy: 0.4977
  896/25000 [>.............................] - ETA: 1:34 - loss: 7.7522 - accuracy: 0.4944
  928/25000 [>.............................] - ETA: 1:33 - loss: 7.6997 - accuracy: 0.4978
  960/25000 [>.............................] - ETA: 1:33 - loss: 7.6666 - accuracy: 0.5000
  992/25000 [>.............................] - ETA: 1:32 - loss: 7.6357 - accuracy: 0.5020
 1024/25000 [>.............................] - ETA: 1:32 - loss: 7.6367 - accuracy: 0.5020
 1056/25000 [>.............................] - ETA: 1:32 - loss: 7.6811 - accuracy: 0.4991
 1088/25000 [>.............................] - ETA: 1:31 - loss: 7.6807 - accuracy: 0.4991
 1120/25000 [>.............................] - ETA: 1:31 - loss: 7.6255 - accuracy: 0.5027
 1152/25000 [>.............................] - ETA: 1:30 - loss: 7.6533 - accuracy: 0.5009
 1184/25000 [>.............................] - ETA: 1:30 - loss: 7.6019 - accuracy: 0.5042
 1216/25000 [>.............................] - ETA: 1:30 - loss: 7.5657 - accuracy: 0.5066
 1248/25000 [>.............................] - ETA: 1:30 - loss: 7.5929 - accuracy: 0.5048
 1280/25000 [>.............................] - ETA: 1:30 - loss: 7.6067 - accuracy: 0.5039
 1312/25000 [>.............................] - ETA: 1:30 - loss: 7.5498 - accuracy: 0.5076
 1344/25000 [>.............................] - ETA: 1:30 - loss: 7.5411 - accuracy: 0.5082
 1376/25000 [>.............................] - ETA: 1:30 - loss: 7.5440 - accuracy: 0.5080
 1408/25000 [>.............................] - ETA: 1:30 - loss: 7.5686 - accuracy: 0.5064
 1440/25000 [>.............................] - ETA: 1:29 - loss: 7.6134 - accuracy: 0.5035
 1472/25000 [>.............................] - ETA: 1:29 - loss: 7.6250 - accuracy: 0.5027
 1504/25000 [>.............................] - ETA: 1:29 - loss: 7.6258 - accuracy: 0.5027
 1536/25000 [>.............................] - ETA: 1:29 - loss: 7.5768 - accuracy: 0.5059
 1568/25000 [>.............................] - ETA: 1:29 - loss: 7.5591 - accuracy: 0.5070
 1600/25000 [>.............................] - ETA: 1:29 - loss: 7.5612 - accuracy: 0.5069
 1632/25000 [>.............................] - ETA: 1:29 - loss: 7.5821 - accuracy: 0.5055
 1664/25000 [>.............................] - ETA: 1:29 - loss: 7.5745 - accuracy: 0.5060
 1696/25000 [=>............................] - ETA: 1:29 - loss: 7.5762 - accuracy: 0.5059
 1728/25000 [=>............................] - ETA: 1:28 - loss: 7.5335 - accuracy: 0.5087
 1760/25000 [=>............................] - ETA: 1:28 - loss: 7.5708 - accuracy: 0.5063
 1792/25000 [=>............................] - ETA: 1:28 - loss: 7.5554 - accuracy: 0.5073
 1824/25000 [=>............................] - ETA: 1:28 - loss: 7.5237 - accuracy: 0.5093
 1856/25000 [=>............................] - ETA: 1:28 - loss: 7.5179 - accuracy: 0.5097
 1888/25000 [=>............................] - ETA: 1:27 - loss: 7.5367 - accuracy: 0.5085
 1920/25000 [=>............................] - ETA: 1:27 - loss: 7.5149 - accuracy: 0.5099
 1952/25000 [=>............................] - ETA: 1:27 - loss: 7.4938 - accuracy: 0.5113
 1984/25000 [=>............................] - ETA: 1:27 - loss: 7.4425 - accuracy: 0.5146
 2016/25000 [=>............................] - ETA: 1:27 - loss: 7.4461 - accuracy: 0.5144
 2048/25000 [=>............................] - ETA: 1:27 - loss: 7.4420 - accuracy: 0.5146
 2080/25000 [=>............................] - ETA: 1:27 - loss: 7.4528 - accuracy: 0.5139
 2112/25000 [=>............................] - ETA: 1:26 - loss: 7.4779 - accuracy: 0.5123
 2144/25000 [=>............................] - ETA: 1:26 - loss: 7.4878 - accuracy: 0.5117
 2176/25000 [=>............................] - ETA: 1:26 - loss: 7.4693 - accuracy: 0.5129
 2208/25000 [=>............................] - ETA: 1:26 - loss: 7.4930 - accuracy: 0.5113
 2240/25000 [=>............................] - ETA: 1:26 - loss: 7.4886 - accuracy: 0.5116
 2272/25000 [=>............................] - ETA: 1:25 - loss: 7.4507 - accuracy: 0.5141
 2304/25000 [=>............................] - ETA: 1:25 - loss: 7.4736 - accuracy: 0.5126
 2336/25000 [=>............................] - ETA: 1:25 - loss: 7.4566 - accuracy: 0.5137
 2368/25000 [=>............................] - ETA: 1:25 - loss: 7.4853 - accuracy: 0.5118
 2400/25000 [=>............................] - ETA: 1:25 - loss: 7.4877 - accuracy: 0.5117
 2432/25000 [=>............................] - ETA: 1:25 - loss: 7.4964 - accuracy: 0.5111
 2464/25000 [=>............................] - ETA: 1:25 - loss: 7.4737 - accuracy: 0.5126
 2496/25000 [=>............................] - ETA: 1:25 - loss: 7.4455 - accuracy: 0.5144
 2528/25000 [==>...........................] - ETA: 1:24 - loss: 7.4483 - accuracy: 0.5142
 2560/25000 [==>...........................] - ETA: 1:24 - loss: 7.4570 - accuracy: 0.5137
 2592/25000 [==>...........................] - ETA: 1:24 - loss: 7.4655 - accuracy: 0.5131
 2624/25000 [==>...........................] - ETA: 1:24 - loss: 7.4504 - accuracy: 0.5141
 2656/25000 [==>...........................] - ETA: 1:24 - loss: 7.4703 - accuracy: 0.5128
 2688/25000 [==>...........................] - ETA: 1:24 - loss: 7.4556 - accuracy: 0.5138
 2720/25000 [==>...........................] - ETA: 1:24 - loss: 7.4468 - accuracy: 0.5143
 2752/25000 [==>...........................] - ETA: 1:23 - loss: 7.4716 - accuracy: 0.5127
 2784/25000 [==>...........................] - ETA: 1:23 - loss: 7.4628 - accuracy: 0.5133
 2816/25000 [==>...........................] - ETA: 1:23 - loss: 7.4597 - accuracy: 0.5135
 2848/25000 [==>...........................] - ETA: 1:23 - loss: 7.4728 - accuracy: 0.5126
 2880/25000 [==>...........................] - ETA: 1:23 - loss: 7.4909 - accuracy: 0.5115
 2912/25000 [==>...........................] - ETA: 1:23 - loss: 7.5087 - accuracy: 0.5103
 2944/25000 [==>...........................] - ETA: 1:23 - loss: 7.4895 - accuracy: 0.5115
 2976/25000 [==>...........................] - ETA: 1:23 - loss: 7.4863 - accuracy: 0.5118
 3008/25000 [==>...........................] - ETA: 1:22 - loss: 7.4729 - accuracy: 0.5126
 3040/25000 [==>...........................] - ETA: 1:22 - loss: 7.4649 - accuracy: 0.5132
 3072/25000 [==>...........................] - ETA: 1:22 - loss: 7.4570 - accuracy: 0.5137
 3104/25000 [==>...........................] - ETA: 1:22 - loss: 7.4740 - accuracy: 0.5126
 3136/25000 [==>...........................] - ETA: 1:22 - loss: 7.4857 - accuracy: 0.5118
 3168/25000 [==>...........................] - ETA: 1:22 - loss: 7.5117 - accuracy: 0.5101
 3200/25000 [==>...........................] - ETA: 1:22 - loss: 7.5277 - accuracy: 0.5091
 3232/25000 [==>...........................] - ETA: 1:22 - loss: 7.5243 - accuracy: 0.5093
 3264/25000 [==>...........................] - ETA: 1:21 - loss: 7.5163 - accuracy: 0.5098
 3296/25000 [==>...........................] - ETA: 1:21 - loss: 7.5084 - accuracy: 0.5103
 3328/25000 [==>...........................] - ETA: 1:21 - loss: 7.5146 - accuracy: 0.5099
 3360/25000 [===>..........................] - ETA: 1:21 - loss: 7.5069 - accuracy: 0.5104
 3392/25000 [===>..........................] - ETA: 1:21 - loss: 7.4994 - accuracy: 0.5109
 3424/25000 [===>..........................] - ETA: 1:21 - loss: 7.5099 - accuracy: 0.5102
 3456/25000 [===>..........................] - ETA: 1:21 - loss: 7.5246 - accuracy: 0.5093
 3488/25000 [===>..........................] - ETA: 1:21 - loss: 7.5391 - accuracy: 0.5083
 3520/25000 [===>..........................] - ETA: 1:20 - loss: 7.5316 - accuracy: 0.5088
 3552/25000 [===>..........................] - ETA: 1:20 - loss: 7.5328 - accuracy: 0.5087
 3584/25000 [===>..........................] - ETA: 1:20 - loss: 7.5511 - accuracy: 0.5075
 3616/25000 [===>..........................] - ETA: 1:20 - loss: 7.5691 - accuracy: 0.5064
 3648/25000 [===>..........................] - ETA: 1:20 - loss: 7.5952 - accuracy: 0.5047
 3680/25000 [===>..........................] - ETA: 1:20 - loss: 7.6000 - accuracy: 0.5043
 3712/25000 [===>..........................] - ETA: 1:20 - loss: 7.5840 - accuracy: 0.5054
 3744/25000 [===>..........................] - ETA: 1:19 - loss: 7.5847 - accuracy: 0.5053
 3776/25000 [===>..........................] - ETA: 1:19 - loss: 7.5895 - accuracy: 0.5050
 3808/25000 [===>..........................] - ETA: 1:19 - loss: 7.5821 - accuracy: 0.5055
 3840/25000 [===>..........................] - ETA: 1:19 - loss: 7.5868 - accuracy: 0.5052
 3872/25000 [===>..........................] - ETA: 1:19 - loss: 7.6033 - accuracy: 0.5041
 3904/25000 [===>..........................] - ETA: 1:19 - loss: 7.5841 - accuracy: 0.5054
 3936/25000 [===>..........................] - ETA: 1:19 - loss: 7.5887 - accuracy: 0.5051
 3968/25000 [===>..........................] - ETA: 1:19 - loss: 7.5661 - accuracy: 0.5066
 4000/25000 [===>..........................] - ETA: 1:19 - loss: 7.5593 - accuracy: 0.5070
 4032/25000 [===>..........................] - ETA: 1:18 - loss: 7.5677 - accuracy: 0.5064
 4064/25000 [===>..........................] - ETA: 1:18 - loss: 7.5534 - accuracy: 0.5074
 4096/25000 [===>..........................] - ETA: 1:18 - loss: 7.5431 - accuracy: 0.5081
 4128/25000 [===>..........................] - ETA: 1:18 - loss: 7.5478 - accuracy: 0.5078
 4160/25000 [===>..........................] - ETA: 1:18 - loss: 7.5524 - accuracy: 0.5075
 4192/25000 [====>.........................] - ETA: 1:18 - loss: 7.5496 - accuracy: 0.5076
 4224/25000 [====>.........................] - ETA: 1:17 - loss: 7.5541 - accuracy: 0.5073
 4256/25000 [====>.........................] - ETA: 1:17 - loss: 7.5513 - accuracy: 0.5075
 4288/25000 [====>.........................] - ETA: 1:17 - loss: 7.5558 - accuracy: 0.5072
 4320/25000 [====>.........................] - ETA: 1:17 - loss: 7.5495 - accuracy: 0.5076
 4352/25000 [====>.........................] - ETA: 1:17 - loss: 7.5504 - accuracy: 0.5076
 4384/25000 [====>.........................] - ETA: 1:17 - loss: 7.5407 - accuracy: 0.5082
 4416/25000 [====>.........................] - ETA: 1:17 - loss: 7.5347 - accuracy: 0.5086
 4448/25000 [====>.........................] - ETA: 1:16 - loss: 7.5460 - accuracy: 0.5079
 4480/25000 [====>.........................] - ETA: 1:16 - loss: 7.5537 - accuracy: 0.5074
 4512/25000 [====>.........................] - ETA: 1:16 - loss: 7.5749 - accuracy: 0.5060
 4544/25000 [====>.........................] - ETA: 1:16 - loss: 7.5688 - accuracy: 0.5064
 4576/25000 [====>.........................] - ETA: 1:16 - loss: 7.5795 - accuracy: 0.5057
 4608/25000 [====>.........................] - ETA: 1:16 - loss: 7.5934 - accuracy: 0.5048
 4640/25000 [====>.........................] - ETA: 1:16 - loss: 7.5840 - accuracy: 0.5054
 4672/25000 [====>.........................] - ETA: 1:16 - loss: 7.5944 - accuracy: 0.5047
 4704/25000 [====>.........................] - ETA: 1:15 - loss: 7.6079 - accuracy: 0.5038
 4736/25000 [====>.........................] - ETA: 1:15 - loss: 7.6116 - accuracy: 0.5036
 4768/25000 [====>.........................] - ETA: 1:15 - loss: 7.6312 - accuracy: 0.5023
 4800/25000 [====>.........................] - ETA: 1:15 - loss: 7.6379 - accuracy: 0.5019
 4832/25000 [====>.........................] - ETA: 1:15 - loss: 7.6444 - accuracy: 0.5014
 4864/25000 [====>.........................] - ETA: 1:15 - loss: 7.6414 - accuracy: 0.5016
 4896/25000 [====>.........................] - ETA: 1:15 - loss: 7.6635 - accuracy: 0.5002
 4928/25000 [====>.........................] - ETA: 1:14 - loss: 7.6573 - accuracy: 0.5006
 4960/25000 [====>.........................] - ETA: 1:14 - loss: 7.6635 - accuracy: 0.5002
 4992/25000 [====>.........................] - ETA: 1:14 - loss: 7.6728 - accuracy: 0.4996
 5024/25000 [=====>........................] - ETA: 1:14 - loss: 7.6727 - accuracy: 0.4996
 5056/25000 [=====>........................] - ETA: 1:14 - loss: 7.6757 - accuracy: 0.4994
 5088/25000 [=====>........................] - ETA: 1:14 - loss: 7.6817 - accuracy: 0.4990
 5120/25000 [=====>........................] - ETA: 1:13 - loss: 7.6696 - accuracy: 0.4998
 5152/25000 [=====>........................] - ETA: 1:13 - loss: 7.6607 - accuracy: 0.5004
 5184/25000 [=====>........................] - ETA: 1:13 - loss: 7.6696 - accuracy: 0.4998
 5216/25000 [=====>........................] - ETA: 1:13 - loss: 7.6578 - accuracy: 0.5006
 5248/25000 [=====>........................] - ETA: 1:13 - loss: 7.6520 - accuracy: 0.5010
 5280/25000 [=====>........................] - ETA: 1:13 - loss: 7.6405 - accuracy: 0.5017
 5312/25000 [=====>........................] - ETA: 1:13 - loss: 7.6320 - accuracy: 0.5023
 5344/25000 [=====>........................] - ETA: 1:12 - loss: 7.6322 - accuracy: 0.5022
 5376/25000 [=====>........................] - ETA: 1:12 - loss: 7.6324 - accuracy: 0.5022
 5408/25000 [=====>........................] - ETA: 1:12 - loss: 7.6383 - accuracy: 0.5018
 5440/25000 [=====>........................] - ETA: 1:12 - loss: 7.6272 - accuracy: 0.5026
 5472/25000 [=====>........................] - ETA: 1:12 - loss: 7.6274 - accuracy: 0.5026
 5504/25000 [=====>........................] - ETA: 1:12 - loss: 7.6304 - accuracy: 0.5024
 5536/25000 [=====>........................] - ETA: 1:12 - loss: 7.6223 - accuracy: 0.5029
 5568/25000 [=====>........................] - ETA: 1:12 - loss: 7.6171 - accuracy: 0.5032
 5600/25000 [=====>........................] - ETA: 1:11 - loss: 7.6283 - accuracy: 0.5025
 5632/25000 [=====>........................] - ETA: 1:11 - loss: 7.6149 - accuracy: 0.5034
 5664/25000 [=====>........................] - ETA: 1:11 - loss: 7.6260 - accuracy: 0.5026
 5696/25000 [=====>........................] - ETA: 1:11 - loss: 7.6343 - accuracy: 0.5021
 5728/25000 [=====>........................] - ETA: 1:11 - loss: 7.6318 - accuracy: 0.5023
 5760/25000 [=====>........................] - ETA: 1:11 - loss: 7.6400 - accuracy: 0.5017
 5792/25000 [=====>........................] - ETA: 1:11 - loss: 7.6454 - accuracy: 0.5014
 5824/25000 [=====>........................] - ETA: 1:11 - loss: 7.6456 - accuracy: 0.5014
 5856/25000 [======>.......................] - ETA: 1:10 - loss: 7.6457 - accuracy: 0.5014
 5888/25000 [======>.......................] - ETA: 1:10 - loss: 7.6354 - accuracy: 0.5020
 5920/25000 [======>.......................] - ETA: 1:10 - loss: 7.6381 - accuracy: 0.5019
 5952/25000 [======>.......................] - ETA: 1:10 - loss: 7.6383 - accuracy: 0.5018
 5984/25000 [======>.......................] - ETA: 1:10 - loss: 7.6384 - accuracy: 0.5018
 6016/25000 [======>.......................] - ETA: 1:10 - loss: 7.6386 - accuracy: 0.5018
 6048/25000 [======>.......................] - ETA: 1:10 - loss: 7.6337 - accuracy: 0.5021
 6080/25000 [======>.......................] - ETA: 1:10 - loss: 7.6313 - accuracy: 0.5023
 6112/25000 [======>.......................] - ETA: 1:09 - loss: 7.6315 - accuracy: 0.5023
 6144/25000 [======>.......................] - ETA: 1:09 - loss: 7.6342 - accuracy: 0.5021
 6176/25000 [======>.......................] - ETA: 1:09 - loss: 7.6368 - accuracy: 0.5019
 6208/25000 [======>.......................] - ETA: 1:09 - loss: 7.6444 - accuracy: 0.5014
 6240/25000 [======>.......................] - ETA: 1:09 - loss: 7.6494 - accuracy: 0.5011
 6272/25000 [======>.......................] - ETA: 1:09 - loss: 7.6422 - accuracy: 0.5016
 6304/25000 [======>.......................] - ETA: 1:09 - loss: 7.6447 - accuracy: 0.5014
 6336/25000 [======>.......................] - ETA: 1:09 - loss: 7.6400 - accuracy: 0.5017
 6368/25000 [======>.......................] - ETA: 1:09 - loss: 7.6353 - accuracy: 0.5020
 6400/25000 [======>.......................] - ETA: 1:08 - loss: 7.6427 - accuracy: 0.5016
 6432/25000 [======>.......................] - ETA: 1:08 - loss: 7.6404 - accuracy: 0.5017
 6464/25000 [======>.......................] - ETA: 1:08 - loss: 7.6382 - accuracy: 0.5019
 6496/25000 [======>.......................] - ETA: 1:08 - loss: 7.6407 - accuracy: 0.5017
 6528/25000 [======>.......................] - ETA: 1:08 - loss: 7.6431 - accuracy: 0.5015
 6560/25000 [======>.......................] - ETA: 1:08 - loss: 7.6479 - accuracy: 0.5012
 6592/25000 [======>.......................] - ETA: 1:08 - loss: 7.6480 - accuracy: 0.5012
 6624/25000 [======>.......................] - ETA: 1:08 - loss: 7.6435 - accuracy: 0.5015
 6656/25000 [======>.......................] - ETA: 1:07 - loss: 7.6459 - accuracy: 0.5014
 6688/25000 [=======>......................] - ETA: 1:07 - loss: 7.6437 - accuracy: 0.5015
 6720/25000 [=======>......................] - ETA: 1:07 - loss: 7.6347 - accuracy: 0.5021
 6752/25000 [=======>......................] - ETA: 1:07 - loss: 7.6371 - accuracy: 0.5019
 6784/25000 [=======>......................] - ETA: 1:07 - loss: 7.6531 - accuracy: 0.5009
 6816/25000 [=======>......................] - ETA: 1:07 - loss: 7.6644 - accuracy: 0.5001
 6848/25000 [=======>......................] - ETA: 1:07 - loss: 7.6599 - accuracy: 0.5004
 6880/25000 [=======>......................] - ETA: 1:07 - loss: 7.6555 - accuracy: 0.5007
 6912/25000 [=======>......................] - ETA: 1:06 - loss: 7.6622 - accuracy: 0.5003
 6944/25000 [=======>......................] - ETA: 1:06 - loss: 7.6578 - accuracy: 0.5006
 6976/25000 [=======>......................] - ETA: 1:06 - loss: 7.6600 - accuracy: 0.5004
 7008/25000 [=======>......................] - ETA: 1:06 - loss: 7.6732 - accuracy: 0.4996
 7040/25000 [=======>......................] - ETA: 1:06 - loss: 7.6644 - accuracy: 0.5001
 7072/25000 [=======>......................] - ETA: 1:06 - loss: 7.6623 - accuracy: 0.5003
 7104/25000 [=======>......................] - ETA: 1:06 - loss: 7.6645 - accuracy: 0.5001
 7136/25000 [=======>......................] - ETA: 1:06 - loss: 7.6688 - accuracy: 0.4999
 7168/25000 [=======>......................] - ETA: 1:05 - loss: 7.6688 - accuracy: 0.4999
 7200/25000 [=======>......................] - ETA: 1:05 - loss: 7.6687 - accuracy: 0.4999
 7232/25000 [=======>......................] - ETA: 1:05 - loss: 7.6730 - accuracy: 0.4996
 7264/25000 [=======>......................] - ETA: 1:05 - loss: 7.6687 - accuracy: 0.4999
 7296/25000 [=======>......................] - ETA: 1:05 - loss: 7.6792 - accuracy: 0.4992
 7328/25000 [=======>......................] - ETA: 1:05 - loss: 7.6792 - accuracy: 0.4992
 7360/25000 [=======>......................] - ETA: 1:05 - loss: 7.6750 - accuracy: 0.4995
 7392/25000 [=======>......................] - ETA: 1:05 - loss: 7.6811 - accuracy: 0.4991
 7424/25000 [=======>......................] - ETA: 1:04 - loss: 7.6769 - accuracy: 0.4993
 7456/25000 [=======>......................] - ETA: 1:04 - loss: 7.6769 - accuracy: 0.4993
 7488/25000 [=======>......................] - ETA: 1:04 - loss: 7.6789 - accuracy: 0.4992
 7520/25000 [========>.....................] - ETA: 1:04 - loss: 7.6748 - accuracy: 0.4995
 7552/25000 [========>.....................] - ETA: 1:04 - loss: 7.6808 - accuracy: 0.4991
 7584/25000 [========>.....................] - ETA: 1:04 - loss: 7.6808 - accuracy: 0.4991
 7616/25000 [========>.....................] - ETA: 1:04 - loss: 7.6807 - accuracy: 0.4991
 7648/25000 [========>.....................] - ETA: 1:04 - loss: 7.6786 - accuracy: 0.4992
 7680/25000 [========>.....................] - ETA: 1:03 - loss: 7.6686 - accuracy: 0.4999
 7712/25000 [========>.....................] - ETA: 1:03 - loss: 7.6766 - accuracy: 0.4994
 7744/25000 [========>.....................] - ETA: 1:03 - loss: 7.6646 - accuracy: 0.5001
 7776/25000 [========>.....................] - ETA: 1:03 - loss: 7.6607 - accuracy: 0.5004
 7808/25000 [========>.....................] - ETA: 1:03 - loss: 7.6509 - accuracy: 0.5010
 7840/25000 [========>.....................] - ETA: 1:03 - loss: 7.6529 - accuracy: 0.5009
 7872/25000 [========>.....................] - ETA: 1:03 - loss: 7.6510 - accuracy: 0.5010
 7904/25000 [========>.....................] - ETA: 1:03 - loss: 7.6492 - accuracy: 0.5011
 7936/25000 [========>.....................] - ETA: 1:03 - loss: 7.6415 - accuracy: 0.5016
 7968/25000 [========>.....................] - ETA: 1:02 - loss: 7.6416 - accuracy: 0.5016
 8000/25000 [========>.....................] - ETA: 1:02 - loss: 7.6494 - accuracy: 0.5011
 8032/25000 [========>.....................] - ETA: 1:02 - loss: 7.6437 - accuracy: 0.5015
 8064/25000 [========>.....................] - ETA: 1:02 - loss: 7.6476 - accuracy: 0.5012
 8096/25000 [========>.....................] - ETA: 1:02 - loss: 7.6439 - accuracy: 0.5015
 8128/25000 [========>.....................] - ETA: 1:02 - loss: 7.6478 - accuracy: 0.5012
 8160/25000 [========>.....................] - ETA: 1:02 - loss: 7.6553 - accuracy: 0.5007
 8192/25000 [========>.....................] - ETA: 1:02 - loss: 7.6554 - accuracy: 0.5007
 8224/25000 [========>.....................] - ETA: 1:01 - loss: 7.6629 - accuracy: 0.5002
 8256/25000 [========>.....................] - ETA: 1:01 - loss: 7.6610 - accuracy: 0.5004
 8288/25000 [========>.....................] - ETA: 1:01 - loss: 7.6629 - accuracy: 0.5002
 8320/25000 [========>.....................] - ETA: 1:01 - loss: 7.6611 - accuracy: 0.5004
 8352/25000 [=========>....................] - ETA: 1:01 - loss: 7.6519 - accuracy: 0.5010
 8384/25000 [=========>....................] - ETA: 1:01 - loss: 7.6575 - accuracy: 0.5006
 8416/25000 [=========>....................] - ETA: 1:01 - loss: 7.6666 - accuracy: 0.5000
 8448/25000 [=========>....................] - ETA: 1:01 - loss: 7.6666 - accuracy: 0.5000
 8480/25000 [=========>....................] - ETA: 1:00 - loss: 7.6648 - accuracy: 0.5001
 8512/25000 [=========>....................] - ETA: 1:00 - loss: 7.6666 - accuracy: 0.5000
 8544/25000 [=========>....................] - ETA: 1:00 - loss: 7.6702 - accuracy: 0.4998
 8576/25000 [=========>....................] - ETA: 1:00 - loss: 7.6684 - accuracy: 0.4999
 8608/25000 [=========>....................] - ETA: 1:00 - loss: 7.6684 - accuracy: 0.4999
 8640/25000 [=========>....................] - ETA: 1:00 - loss: 7.6666 - accuracy: 0.5000
 8672/25000 [=========>....................] - ETA: 1:00 - loss: 7.6649 - accuracy: 0.5001
 8704/25000 [=========>....................] - ETA: 1:00 - loss: 7.6649 - accuracy: 0.5001
 8736/25000 [=========>....................] - ETA: 59s - loss: 7.6631 - accuracy: 0.5002 
 8768/25000 [=========>....................] - ETA: 59s - loss: 7.6614 - accuracy: 0.5003
 8800/25000 [=========>....................] - ETA: 59s - loss: 7.6614 - accuracy: 0.5003
 8832/25000 [=========>....................] - ETA: 59s - loss: 7.6545 - accuracy: 0.5008
 8864/25000 [=========>....................] - ETA: 59s - loss: 7.6545 - accuracy: 0.5008
 8896/25000 [=========>....................] - ETA: 59s - loss: 7.6477 - accuracy: 0.5012
 8928/25000 [=========>....................] - ETA: 59s - loss: 7.6546 - accuracy: 0.5008
 8960/25000 [=========>....................] - ETA: 59s - loss: 7.6581 - accuracy: 0.5006
 8992/25000 [=========>....................] - ETA: 58s - loss: 7.6666 - accuracy: 0.5000
 9024/25000 [=========>....................] - ETA: 58s - loss: 7.6632 - accuracy: 0.5002
 9056/25000 [=========>....................] - ETA: 58s - loss: 7.6649 - accuracy: 0.5001
 9088/25000 [=========>....................] - ETA: 58s - loss: 7.6683 - accuracy: 0.4999
 9120/25000 [=========>....................] - ETA: 58s - loss: 7.6649 - accuracy: 0.5001
 9152/25000 [=========>....................] - ETA: 58s - loss: 7.6549 - accuracy: 0.5008
 9184/25000 [==========>...................] - ETA: 58s - loss: 7.6483 - accuracy: 0.5012
 9216/25000 [==========>...................] - ETA: 58s - loss: 7.6516 - accuracy: 0.5010
 9248/25000 [==========>...................] - ETA: 57s - loss: 7.6550 - accuracy: 0.5008
 9280/25000 [==========>...................] - ETA: 57s - loss: 7.6517 - accuracy: 0.5010
 9312/25000 [==========>...................] - ETA: 57s - loss: 7.6617 - accuracy: 0.5003
 9344/25000 [==========>...................] - ETA: 57s - loss: 7.6666 - accuracy: 0.5000
 9376/25000 [==========>...................] - ETA: 57s - loss: 7.6764 - accuracy: 0.4994
 9408/25000 [==========>...................] - ETA: 57s - loss: 7.6780 - accuracy: 0.4993
 9440/25000 [==========>...................] - ETA: 57s - loss: 7.6764 - accuracy: 0.4994
 9472/25000 [==========>...................] - ETA: 57s - loss: 7.6763 - accuracy: 0.4994
 9504/25000 [==========>...................] - ETA: 56s - loss: 7.6747 - accuracy: 0.4995
 9536/25000 [==========>...................] - ETA: 56s - loss: 7.6731 - accuracy: 0.4996
 9568/25000 [==========>...................] - ETA: 56s - loss: 7.6762 - accuracy: 0.4994
 9600/25000 [==========>...................] - ETA: 56s - loss: 7.6682 - accuracy: 0.4999
 9632/25000 [==========>...................] - ETA: 56s - loss: 7.6634 - accuracy: 0.5002
 9664/25000 [==========>...................] - ETA: 56s - loss: 7.6603 - accuracy: 0.5004
 9696/25000 [==========>...................] - ETA: 56s - loss: 7.6619 - accuracy: 0.5003
 9728/25000 [==========>...................] - ETA: 56s - loss: 7.6509 - accuracy: 0.5010
 9760/25000 [==========>...................] - ETA: 56s - loss: 7.6493 - accuracy: 0.5011
 9792/25000 [==========>...................] - ETA: 55s - loss: 7.6431 - accuracy: 0.5015
 9824/25000 [==========>...................] - ETA: 55s - loss: 7.6416 - accuracy: 0.5016
 9856/25000 [==========>...................] - ETA: 55s - loss: 7.6464 - accuracy: 0.5013
 9888/25000 [==========>...................] - ETA: 55s - loss: 7.6418 - accuracy: 0.5016
 9920/25000 [==========>...................] - ETA: 55s - loss: 7.6357 - accuracy: 0.5020
 9952/25000 [==========>...................] - ETA: 55s - loss: 7.6312 - accuracy: 0.5023
 9984/25000 [==========>...................] - ETA: 55s - loss: 7.6313 - accuracy: 0.5023
10016/25000 [===========>..................] - ETA: 55s - loss: 7.6360 - accuracy: 0.5020
10048/25000 [===========>..................] - ETA: 54s - loss: 7.6346 - accuracy: 0.5021
10080/25000 [===========>..................] - ETA: 54s - loss: 7.6408 - accuracy: 0.5017
10112/25000 [===========>..................] - ETA: 54s - loss: 7.6363 - accuracy: 0.5020
10144/25000 [===========>..................] - ETA: 54s - loss: 7.6379 - accuracy: 0.5019
10176/25000 [===========>..................] - ETA: 54s - loss: 7.6410 - accuracy: 0.5017
10208/25000 [===========>..................] - ETA: 54s - loss: 7.6336 - accuracy: 0.5022
10240/25000 [===========>..................] - ETA: 54s - loss: 7.6322 - accuracy: 0.5022
10272/25000 [===========>..................] - ETA: 54s - loss: 7.6308 - accuracy: 0.5023
10304/25000 [===========>..................] - ETA: 53s - loss: 7.6264 - accuracy: 0.5026
10336/25000 [===========>..................] - ETA: 53s - loss: 7.6280 - accuracy: 0.5025
10368/25000 [===========>..................] - ETA: 53s - loss: 7.6267 - accuracy: 0.5026
10400/25000 [===========>..................] - ETA: 53s - loss: 7.6239 - accuracy: 0.5028
10432/25000 [===========>..................] - ETA: 53s - loss: 7.6211 - accuracy: 0.5030
10464/25000 [===========>..................] - ETA: 53s - loss: 7.6183 - accuracy: 0.5032
10496/25000 [===========>..................] - ETA: 53s - loss: 7.6184 - accuracy: 0.5031
10528/25000 [===========>..................] - ETA: 53s - loss: 7.6244 - accuracy: 0.5028
10560/25000 [===========>..................] - ETA: 52s - loss: 7.6173 - accuracy: 0.5032
10592/25000 [===========>..................] - ETA: 52s - loss: 7.6145 - accuracy: 0.5034
10624/25000 [===========>..................] - ETA: 52s - loss: 7.6161 - accuracy: 0.5033
10656/25000 [===========>..................] - ETA: 52s - loss: 7.6206 - accuracy: 0.5030
10688/25000 [===========>..................] - ETA: 52s - loss: 7.6207 - accuracy: 0.5030
10720/25000 [===========>..................] - ETA: 52s - loss: 7.6294 - accuracy: 0.5024
10752/25000 [===========>..................] - ETA: 52s - loss: 7.6324 - accuracy: 0.5022
10784/25000 [===========>..................] - ETA: 52s - loss: 7.6311 - accuracy: 0.5023
10816/25000 [===========>..................] - ETA: 52s - loss: 7.6326 - accuracy: 0.5022
10848/25000 [============>.................] - ETA: 51s - loss: 7.6369 - accuracy: 0.5019
10880/25000 [============>.................] - ETA: 51s - loss: 7.6398 - accuracy: 0.5017
10912/25000 [============>.................] - ETA: 51s - loss: 7.6385 - accuracy: 0.5018
10944/25000 [============>.................] - ETA: 51s - loss: 7.6344 - accuracy: 0.5021
10976/25000 [============>.................] - ETA: 51s - loss: 7.6359 - accuracy: 0.5020
11008/25000 [============>.................] - ETA: 51s - loss: 7.6388 - accuracy: 0.5018
11040/25000 [============>.................] - ETA: 51s - loss: 7.6416 - accuracy: 0.5016
11072/25000 [============>.................] - ETA: 51s - loss: 7.6389 - accuracy: 0.5018
11104/25000 [============>.................] - ETA: 50s - loss: 7.6418 - accuracy: 0.5016
11136/25000 [============>.................] - ETA: 50s - loss: 7.6391 - accuracy: 0.5018
11168/25000 [============>.................] - ETA: 50s - loss: 7.6433 - accuracy: 0.5015
11200/25000 [============>.................] - ETA: 50s - loss: 7.6502 - accuracy: 0.5011
11232/25000 [============>.................] - ETA: 50s - loss: 7.6530 - accuracy: 0.5009
11264/25000 [============>.................] - ETA: 50s - loss: 7.6516 - accuracy: 0.5010
11296/25000 [============>.................] - ETA: 50s - loss: 7.6503 - accuracy: 0.5011
11328/25000 [============>.................] - ETA: 50s - loss: 7.6531 - accuracy: 0.5009
11360/25000 [============>.................] - ETA: 49s - loss: 7.6558 - accuracy: 0.5007
11392/25000 [============>.................] - ETA: 49s - loss: 7.6599 - accuracy: 0.5004
11424/25000 [============>.................] - ETA: 49s - loss: 7.6626 - accuracy: 0.5003
11456/25000 [============>.................] - ETA: 49s - loss: 7.6693 - accuracy: 0.4998
11488/25000 [============>.................] - ETA: 49s - loss: 7.6680 - accuracy: 0.4999
11520/25000 [============>.................] - ETA: 49s - loss: 7.6693 - accuracy: 0.4998
11552/25000 [============>.................] - ETA: 49s - loss: 7.6719 - accuracy: 0.4997
11584/25000 [============>.................] - ETA: 49s - loss: 7.6679 - accuracy: 0.4999
11616/25000 [============>.................] - ETA: 48s - loss: 7.6640 - accuracy: 0.5002
11648/25000 [============>.................] - ETA: 48s - loss: 7.6666 - accuracy: 0.5000
11680/25000 [=============>................] - ETA: 48s - loss: 7.6706 - accuracy: 0.4997
11712/25000 [=============>................] - ETA: 48s - loss: 7.6719 - accuracy: 0.4997
11744/25000 [=============>................] - ETA: 48s - loss: 7.6705 - accuracy: 0.4997
11776/25000 [=============>................] - ETA: 48s - loss: 7.6692 - accuracy: 0.4998
11808/25000 [=============>................] - ETA: 48s - loss: 7.6640 - accuracy: 0.5002
11840/25000 [=============>................] - ETA: 48s - loss: 7.6653 - accuracy: 0.5001
11872/25000 [=============>................] - ETA: 48s - loss: 7.6627 - accuracy: 0.5003
11904/25000 [=============>................] - ETA: 47s - loss: 7.6602 - accuracy: 0.5004
11936/25000 [=============>................] - ETA: 47s - loss: 7.6602 - accuracy: 0.5004
11968/25000 [=============>................] - ETA: 47s - loss: 7.6589 - accuracy: 0.5005
12000/25000 [=============>................] - ETA: 47s - loss: 7.6577 - accuracy: 0.5006
12032/25000 [=============>................] - ETA: 47s - loss: 7.6590 - accuracy: 0.5005
12064/25000 [=============>................] - ETA: 47s - loss: 7.6628 - accuracy: 0.5002
12096/25000 [=============>................] - ETA: 47s - loss: 7.6615 - accuracy: 0.5003
12128/25000 [=============>................] - ETA: 47s - loss: 7.6691 - accuracy: 0.4998
12160/25000 [=============>................] - ETA: 46s - loss: 7.6654 - accuracy: 0.5001
12192/25000 [=============>................] - ETA: 46s - loss: 7.6616 - accuracy: 0.5003
12224/25000 [=============>................] - ETA: 46s - loss: 7.6603 - accuracy: 0.5004
12256/25000 [=============>................] - ETA: 46s - loss: 7.6629 - accuracy: 0.5002
12288/25000 [=============>................] - ETA: 46s - loss: 7.6629 - accuracy: 0.5002
12320/25000 [=============>................] - ETA: 46s - loss: 7.6604 - accuracy: 0.5004
12352/25000 [=============>................] - ETA: 46s - loss: 7.6617 - accuracy: 0.5003
12384/25000 [=============>................] - ETA: 46s - loss: 7.6604 - accuracy: 0.5004
12416/25000 [=============>................] - ETA: 46s - loss: 7.6555 - accuracy: 0.5007
12448/25000 [=============>................] - ETA: 45s - loss: 7.6506 - accuracy: 0.5010
12480/25000 [=============>................] - ETA: 45s - loss: 7.6494 - accuracy: 0.5011
12512/25000 [==============>...............] - ETA: 45s - loss: 7.6531 - accuracy: 0.5009
12544/25000 [==============>...............] - ETA: 45s - loss: 7.6568 - accuracy: 0.5006
12576/25000 [==============>...............] - ETA: 45s - loss: 7.6532 - accuracy: 0.5009
12608/25000 [==============>...............] - ETA: 45s - loss: 7.6508 - accuracy: 0.5010
12640/25000 [==============>...............] - ETA: 45s - loss: 7.6521 - accuracy: 0.5009
12672/25000 [==============>...............] - ETA: 45s - loss: 7.6473 - accuracy: 0.5013
12704/25000 [==============>...............] - ETA: 44s - loss: 7.6497 - accuracy: 0.5011
12736/25000 [==============>...............] - ETA: 44s - loss: 7.6510 - accuracy: 0.5010
12768/25000 [==============>...............] - ETA: 44s - loss: 7.6510 - accuracy: 0.5010
12800/25000 [==============>...............] - ETA: 44s - loss: 7.6582 - accuracy: 0.5005
12832/25000 [==============>...............] - ETA: 44s - loss: 7.6630 - accuracy: 0.5002
12864/25000 [==============>...............] - ETA: 44s - loss: 7.6607 - accuracy: 0.5004
12896/25000 [==============>...............] - ETA: 44s - loss: 7.6642 - accuracy: 0.5002
12928/25000 [==============>...............] - ETA: 44s - loss: 7.6690 - accuracy: 0.4998
12960/25000 [==============>...............] - ETA: 44s - loss: 7.6678 - accuracy: 0.4999
12992/25000 [==============>...............] - ETA: 43s - loss: 7.6678 - accuracy: 0.4999
13024/25000 [==============>...............] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
13056/25000 [==============>...............] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
13088/25000 [==============>...............] - ETA: 43s - loss: 7.6678 - accuracy: 0.4999
13120/25000 [==============>...............] - ETA: 43s - loss: 7.6678 - accuracy: 0.4999
13152/25000 [==============>...............] - ETA: 43s - loss: 7.6666 - accuracy: 0.5000
13184/25000 [==============>...............] - ETA: 43s - loss: 7.6748 - accuracy: 0.4995
13216/25000 [==============>...............] - ETA: 43s - loss: 7.6771 - accuracy: 0.4993
13248/25000 [==============>...............] - ETA: 42s - loss: 7.6794 - accuracy: 0.4992
13280/25000 [==============>...............] - ETA: 42s - loss: 7.6793 - accuracy: 0.4992
13312/25000 [==============>...............] - ETA: 42s - loss: 7.6793 - accuracy: 0.4992
13344/25000 [===============>..............] - ETA: 42s - loss: 7.6793 - accuracy: 0.4992
13376/25000 [===============>..............] - ETA: 42s - loss: 7.6804 - accuracy: 0.4991
13408/25000 [===============>..............] - ETA: 42s - loss: 7.6781 - accuracy: 0.4993
13440/25000 [===============>..............] - ETA: 42s - loss: 7.6757 - accuracy: 0.4994
13472/25000 [===============>..............] - ETA: 42s - loss: 7.6757 - accuracy: 0.4994
13504/25000 [===============>..............] - ETA: 42s - loss: 7.6757 - accuracy: 0.4994
13536/25000 [===============>..............] - ETA: 41s - loss: 7.6734 - accuracy: 0.4996
13568/25000 [===============>..............] - ETA: 41s - loss: 7.6791 - accuracy: 0.4992
13600/25000 [===============>..............] - ETA: 41s - loss: 7.6756 - accuracy: 0.4994
13632/25000 [===============>..............] - ETA: 41s - loss: 7.6767 - accuracy: 0.4993
13664/25000 [===============>..............] - ETA: 41s - loss: 7.6778 - accuracy: 0.4993
13696/25000 [===============>..............] - ETA: 41s - loss: 7.6767 - accuracy: 0.4993
13728/25000 [===============>..............] - ETA: 41s - loss: 7.6778 - accuracy: 0.4993
13760/25000 [===============>..............] - ETA: 41s - loss: 7.6688 - accuracy: 0.4999
13792/25000 [===============>..............] - ETA: 40s - loss: 7.6755 - accuracy: 0.4994
13824/25000 [===============>..............] - ETA: 40s - loss: 7.6744 - accuracy: 0.4995
13856/25000 [===============>..............] - ETA: 40s - loss: 7.6688 - accuracy: 0.4999
13888/25000 [===============>..............] - ETA: 40s - loss: 7.6688 - accuracy: 0.4999
13920/25000 [===============>..............] - ETA: 40s - loss: 7.6666 - accuracy: 0.5000
13952/25000 [===============>..............] - ETA: 40s - loss: 7.6710 - accuracy: 0.4997
13984/25000 [===============>..............] - ETA: 40s - loss: 7.6754 - accuracy: 0.4994
14016/25000 [===============>..............] - ETA: 40s - loss: 7.6754 - accuracy: 0.4994
14048/25000 [===============>..............] - ETA: 40s - loss: 7.6764 - accuracy: 0.4994
14080/25000 [===============>..............] - ETA: 39s - loss: 7.6786 - accuracy: 0.4992
14112/25000 [===============>..............] - ETA: 39s - loss: 7.6753 - accuracy: 0.4994
14144/25000 [===============>..............] - ETA: 39s - loss: 7.6753 - accuracy: 0.4994
14176/25000 [================>.............] - ETA: 39s - loss: 7.6720 - accuracy: 0.4996
14208/25000 [================>.............] - ETA: 39s - loss: 7.6742 - accuracy: 0.4995
14240/25000 [================>.............] - ETA: 39s - loss: 7.6785 - accuracy: 0.4992
14272/25000 [================>.............] - ETA: 39s - loss: 7.6795 - accuracy: 0.4992
14304/25000 [================>.............] - ETA: 39s - loss: 7.6773 - accuracy: 0.4993
14336/25000 [================>.............] - ETA: 39s - loss: 7.6784 - accuracy: 0.4992
14368/25000 [================>.............] - ETA: 38s - loss: 7.6720 - accuracy: 0.4997
14400/25000 [================>.............] - ETA: 38s - loss: 7.6730 - accuracy: 0.4996
14432/25000 [================>.............] - ETA: 38s - loss: 7.6719 - accuracy: 0.4997
14464/25000 [================>.............] - ETA: 38s - loss: 7.6719 - accuracy: 0.4997
14496/25000 [================>.............] - ETA: 38s - loss: 7.6730 - accuracy: 0.4996
14528/25000 [================>.............] - ETA: 38s - loss: 7.6730 - accuracy: 0.4996
14560/25000 [================>.............] - ETA: 38s - loss: 7.6729 - accuracy: 0.4996
14592/25000 [================>.............] - ETA: 38s - loss: 7.6708 - accuracy: 0.4997
14624/25000 [================>.............] - ETA: 37s - loss: 7.6666 - accuracy: 0.5000
14656/25000 [================>.............] - ETA: 37s - loss: 7.6614 - accuracy: 0.5003
14688/25000 [================>.............] - ETA: 37s - loss: 7.6645 - accuracy: 0.5001
14720/25000 [================>.............] - ETA: 37s - loss: 7.6614 - accuracy: 0.5003
14752/25000 [================>.............] - ETA: 37s - loss: 7.6656 - accuracy: 0.5001
14784/25000 [================>.............] - ETA: 37s - loss: 7.6645 - accuracy: 0.5001
14816/25000 [================>.............] - ETA: 37s - loss: 7.6614 - accuracy: 0.5003
14848/25000 [================>.............] - ETA: 37s - loss: 7.6563 - accuracy: 0.5007
14880/25000 [================>.............] - ETA: 36s - loss: 7.6594 - accuracy: 0.5005
14912/25000 [================>.............] - ETA: 36s - loss: 7.6615 - accuracy: 0.5003
14944/25000 [================>.............] - ETA: 36s - loss: 7.6646 - accuracy: 0.5001
14976/25000 [================>.............] - ETA: 36s - loss: 7.6697 - accuracy: 0.4998
15008/25000 [=================>............] - ETA: 36s - loss: 7.6758 - accuracy: 0.4994
15040/25000 [=================>............] - ETA: 36s - loss: 7.6727 - accuracy: 0.4996
15072/25000 [=================>............] - ETA: 36s - loss: 7.6707 - accuracy: 0.4997
15104/25000 [=================>............] - ETA: 36s - loss: 7.6717 - accuracy: 0.4997
15136/25000 [=================>............] - ETA: 36s - loss: 7.6727 - accuracy: 0.4996
15168/25000 [=================>............] - ETA: 35s - loss: 7.6707 - accuracy: 0.4997
15200/25000 [=================>............] - ETA: 35s - loss: 7.6707 - accuracy: 0.4997
15232/25000 [=================>............] - ETA: 35s - loss: 7.6727 - accuracy: 0.4996
15264/25000 [=================>............] - ETA: 35s - loss: 7.6706 - accuracy: 0.4997
15296/25000 [=================>............] - ETA: 35s - loss: 7.6686 - accuracy: 0.4999
15328/25000 [=================>............] - ETA: 35s - loss: 7.6676 - accuracy: 0.4999
15360/25000 [=================>............] - ETA: 35s - loss: 7.6676 - accuracy: 0.4999
15392/25000 [=================>............] - ETA: 35s - loss: 7.6676 - accuracy: 0.4999
15424/25000 [=================>............] - ETA: 34s - loss: 7.6656 - accuracy: 0.5001
15456/25000 [=================>............] - ETA: 34s - loss: 7.6696 - accuracy: 0.4998
15488/25000 [=================>............] - ETA: 34s - loss: 7.6706 - accuracy: 0.4997
15520/25000 [=================>............] - ETA: 34s - loss: 7.6706 - accuracy: 0.4997
15552/25000 [=================>............] - ETA: 34s - loss: 7.6765 - accuracy: 0.4994
15584/25000 [=================>............] - ETA: 34s - loss: 7.6745 - accuracy: 0.4995
15616/25000 [=================>............] - ETA: 34s - loss: 7.6755 - accuracy: 0.4994
15648/25000 [=================>............] - ETA: 34s - loss: 7.6754 - accuracy: 0.4994
15680/25000 [=================>............] - ETA: 34s - loss: 7.6735 - accuracy: 0.4996
15712/25000 [=================>............] - ETA: 33s - loss: 7.6764 - accuracy: 0.4994
15744/25000 [=================>............] - ETA: 33s - loss: 7.6783 - accuracy: 0.4992
15776/25000 [=================>............] - ETA: 33s - loss: 7.6802 - accuracy: 0.4991
15808/25000 [=================>............] - ETA: 33s - loss: 7.6753 - accuracy: 0.4994
15840/25000 [==================>...........] - ETA: 33s - loss: 7.6763 - accuracy: 0.4994
15872/25000 [==================>...........] - ETA: 33s - loss: 7.6782 - accuracy: 0.4992
15904/25000 [==================>...........] - ETA: 33s - loss: 7.6763 - accuracy: 0.4994
15936/25000 [==================>...........] - ETA: 33s - loss: 7.6762 - accuracy: 0.4994
15968/25000 [==================>...........] - ETA: 32s - loss: 7.6753 - accuracy: 0.4994
16000/25000 [==================>...........] - ETA: 32s - loss: 7.6733 - accuracy: 0.4996
16032/25000 [==================>...........] - ETA: 32s - loss: 7.6781 - accuracy: 0.4993
16064/25000 [==================>...........] - ETA: 32s - loss: 7.6800 - accuracy: 0.4991
16096/25000 [==================>...........] - ETA: 32s - loss: 7.6838 - accuracy: 0.4989
16128/25000 [==================>...........] - ETA: 32s - loss: 7.6799 - accuracy: 0.4991
16160/25000 [==================>...........] - ETA: 32s - loss: 7.6809 - accuracy: 0.4991
16192/25000 [==================>...........] - ETA: 32s - loss: 7.6780 - accuracy: 0.4993
16224/25000 [==================>...........] - ETA: 32s - loss: 7.6817 - accuracy: 0.4990
16256/25000 [==================>...........] - ETA: 31s - loss: 7.6808 - accuracy: 0.4991
16288/25000 [==================>...........] - ETA: 31s - loss: 7.6845 - accuracy: 0.4988
16320/25000 [==================>...........] - ETA: 31s - loss: 7.6845 - accuracy: 0.4988
16352/25000 [==================>...........] - ETA: 31s - loss: 7.6844 - accuracy: 0.4988
16384/25000 [==================>...........] - ETA: 31s - loss: 7.6844 - accuracy: 0.4988
16416/25000 [==================>...........] - ETA: 31s - loss: 7.6788 - accuracy: 0.4992
16448/25000 [==================>...........] - ETA: 31s - loss: 7.6797 - accuracy: 0.4991
16480/25000 [==================>...........] - ETA: 31s - loss: 7.6778 - accuracy: 0.4993
16512/25000 [==================>...........] - ETA: 30s - loss: 7.6796 - accuracy: 0.4992
16544/25000 [==================>...........] - ETA: 30s - loss: 7.6852 - accuracy: 0.4988
16576/25000 [==================>...........] - ETA: 30s - loss: 7.6796 - accuracy: 0.4992
16608/25000 [==================>...........] - ETA: 30s - loss: 7.6814 - accuracy: 0.4990
16640/25000 [==================>...........] - ETA: 30s - loss: 7.6841 - accuracy: 0.4989
16672/25000 [===================>..........] - ETA: 30s - loss: 7.6878 - accuracy: 0.4986
16704/25000 [===================>..........] - ETA: 30s - loss: 7.6877 - accuracy: 0.4986
16736/25000 [===================>..........] - ETA: 30s - loss: 7.6877 - accuracy: 0.4986
16768/25000 [===================>..........] - ETA: 30s - loss: 7.6877 - accuracy: 0.4986
16800/25000 [===================>..........] - ETA: 29s - loss: 7.6830 - accuracy: 0.4989
16832/25000 [===================>..........] - ETA: 29s - loss: 7.6803 - accuracy: 0.4991
16864/25000 [===================>..........] - ETA: 29s - loss: 7.6757 - accuracy: 0.4994
16896/25000 [===================>..........] - ETA: 29s - loss: 7.6766 - accuracy: 0.4993
16928/25000 [===================>..........] - ETA: 29s - loss: 7.6775 - accuracy: 0.4993
16960/25000 [===================>..........] - ETA: 29s - loss: 7.6757 - accuracy: 0.4994
16992/25000 [===================>..........] - ETA: 29s - loss: 7.6765 - accuracy: 0.4994
17024/25000 [===================>..........] - ETA: 29s - loss: 7.6747 - accuracy: 0.4995
17056/25000 [===================>..........] - ETA: 28s - loss: 7.6765 - accuracy: 0.4994
17088/25000 [===================>..........] - ETA: 28s - loss: 7.6765 - accuracy: 0.4994
17120/25000 [===================>..........] - ETA: 28s - loss: 7.6729 - accuracy: 0.4996
17152/25000 [===================>..........] - ETA: 28s - loss: 7.6729 - accuracy: 0.4996
17184/25000 [===================>..........] - ETA: 28s - loss: 7.6738 - accuracy: 0.4995
17216/25000 [===================>..........] - ETA: 28s - loss: 7.6737 - accuracy: 0.4995
17248/25000 [===================>..........] - ETA: 28s - loss: 7.6702 - accuracy: 0.4998
17280/25000 [===================>..........] - ETA: 28s - loss: 7.6711 - accuracy: 0.4997
17312/25000 [===================>..........] - ETA: 28s - loss: 7.6746 - accuracy: 0.4995
17344/25000 [===================>..........] - ETA: 27s - loss: 7.6737 - accuracy: 0.4995
17376/25000 [===================>..........] - ETA: 27s - loss: 7.6790 - accuracy: 0.4992
17408/25000 [===================>..........] - ETA: 27s - loss: 7.6798 - accuracy: 0.4991
17440/25000 [===================>..........] - ETA: 27s - loss: 7.6833 - accuracy: 0.4989
17472/25000 [===================>..........] - ETA: 27s - loss: 7.6798 - accuracy: 0.4991
17504/25000 [====================>.........] - ETA: 27s - loss: 7.6771 - accuracy: 0.4993
17536/25000 [====================>.........] - ETA: 27s - loss: 7.6780 - accuracy: 0.4993
17568/25000 [====================>.........] - ETA: 27s - loss: 7.6788 - accuracy: 0.4992
17600/25000 [====================>.........] - ETA: 26s - loss: 7.6797 - accuracy: 0.4991
17632/25000 [====================>.........] - ETA: 26s - loss: 7.6771 - accuracy: 0.4993
17664/25000 [====================>.........] - ETA: 26s - loss: 7.6788 - accuracy: 0.4992
17696/25000 [====================>.........] - ETA: 26s - loss: 7.6788 - accuracy: 0.4992
17728/25000 [====================>.........] - ETA: 26s - loss: 7.6753 - accuracy: 0.4994
17760/25000 [====================>.........] - ETA: 26s - loss: 7.6753 - accuracy: 0.4994
17792/25000 [====================>.........] - ETA: 26s - loss: 7.6744 - accuracy: 0.4995
17824/25000 [====================>.........] - ETA: 26s - loss: 7.6709 - accuracy: 0.4997
17856/25000 [====================>.........] - ETA: 26s - loss: 7.6701 - accuracy: 0.4998
17888/25000 [====================>.........] - ETA: 25s - loss: 7.6726 - accuracy: 0.4996
17920/25000 [====================>.........] - ETA: 25s - loss: 7.6692 - accuracy: 0.4998
17952/25000 [====================>.........] - ETA: 25s - loss: 7.6683 - accuracy: 0.4999
17984/25000 [====================>.........] - ETA: 25s - loss: 7.6683 - accuracy: 0.4999
18016/25000 [====================>.........] - ETA: 25s - loss: 7.6700 - accuracy: 0.4998
18048/25000 [====================>.........] - ETA: 25s - loss: 7.6726 - accuracy: 0.4996
18080/25000 [====================>.........] - ETA: 25s - loss: 7.6726 - accuracy: 0.4996
18112/25000 [====================>.........] - ETA: 25s - loss: 7.6709 - accuracy: 0.4997
18144/25000 [====================>.........] - ETA: 24s - loss: 7.6751 - accuracy: 0.4994
18176/25000 [====================>.........] - ETA: 24s - loss: 7.6683 - accuracy: 0.4999
18208/25000 [====================>.........] - ETA: 24s - loss: 7.6700 - accuracy: 0.4998
18240/25000 [====================>.........] - ETA: 24s - loss: 7.6691 - accuracy: 0.4998
18272/25000 [====================>.........] - ETA: 24s - loss: 7.6683 - accuracy: 0.4999
18304/25000 [====================>.........] - ETA: 24s - loss: 7.6675 - accuracy: 0.4999
18336/25000 [=====================>........] - ETA: 24s - loss: 7.6700 - accuracy: 0.4998
18368/25000 [=====================>........] - ETA: 24s - loss: 7.6716 - accuracy: 0.4997
18400/25000 [=====================>........] - ETA: 24s - loss: 7.6691 - accuracy: 0.4998
18432/25000 [=====================>........] - ETA: 23s - loss: 7.6708 - accuracy: 0.4997
18464/25000 [=====================>........] - ETA: 23s - loss: 7.6674 - accuracy: 0.4999
18496/25000 [=====================>........] - ETA: 23s - loss: 7.6691 - accuracy: 0.4998
18528/25000 [=====================>........] - ETA: 23s - loss: 7.6691 - accuracy: 0.4998
18560/25000 [=====================>........] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
18592/25000 [=====================>........] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
18624/25000 [=====================>........] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
18656/25000 [=====================>........] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
18688/25000 [=====================>........] - ETA: 22s - loss: 7.6658 - accuracy: 0.5001
18720/25000 [=====================>........] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
18752/25000 [=====================>........] - ETA: 22s - loss: 7.6642 - accuracy: 0.5002
18784/25000 [=====================>........] - ETA: 22s - loss: 7.6707 - accuracy: 0.4997
18816/25000 [=====================>........] - ETA: 22s - loss: 7.6691 - accuracy: 0.4998
18848/25000 [=====================>........] - ETA: 22s - loss: 7.6715 - accuracy: 0.4997
18880/25000 [=====================>........] - ETA: 22s - loss: 7.6707 - accuracy: 0.4997
18912/25000 [=====================>........] - ETA: 22s - loss: 7.6731 - accuracy: 0.4996
18944/25000 [=====================>........] - ETA: 22s - loss: 7.6715 - accuracy: 0.4997
18976/25000 [=====================>........] - ETA: 21s - loss: 7.6723 - accuracy: 0.4996
19008/25000 [=====================>........] - ETA: 21s - loss: 7.6739 - accuracy: 0.4995
19040/25000 [=====================>........] - ETA: 21s - loss: 7.6706 - accuracy: 0.4997
19072/25000 [=====================>........] - ETA: 21s - loss: 7.6682 - accuracy: 0.4999
19104/25000 [=====================>........] - ETA: 21s - loss: 7.6682 - accuracy: 0.4999
19136/25000 [=====================>........] - ETA: 21s - loss: 7.6642 - accuracy: 0.5002
19168/25000 [======================>.......] - ETA: 21s - loss: 7.6650 - accuracy: 0.5001
19200/25000 [======================>.......] - ETA: 21s - loss: 7.6674 - accuracy: 0.4999
19232/25000 [======================>.......] - ETA: 21s - loss: 7.6666 - accuracy: 0.5000
19264/25000 [======================>.......] - ETA: 20s - loss: 7.6682 - accuracy: 0.4999
19296/25000 [======================>.......] - ETA: 20s - loss: 7.6706 - accuracy: 0.4997
19328/25000 [======================>.......] - ETA: 20s - loss: 7.6682 - accuracy: 0.4999
19360/25000 [======================>.......] - ETA: 20s - loss: 7.6698 - accuracy: 0.4998
19392/25000 [======================>.......] - ETA: 20s - loss: 7.6698 - accuracy: 0.4998
19424/25000 [======================>.......] - ETA: 20s - loss: 7.6698 - accuracy: 0.4998
19456/25000 [======================>.......] - ETA: 20s - loss: 7.6737 - accuracy: 0.4995
19488/25000 [======================>.......] - ETA: 20s - loss: 7.6745 - accuracy: 0.4995
19520/25000 [======================>.......] - ETA: 19s - loss: 7.6721 - accuracy: 0.4996
19552/25000 [======================>.......] - ETA: 19s - loss: 7.6737 - accuracy: 0.4995
19584/25000 [======================>.......] - ETA: 19s - loss: 7.6744 - accuracy: 0.4995
19616/25000 [======================>.......] - ETA: 19s - loss: 7.6729 - accuracy: 0.4996
19648/25000 [======================>.......] - ETA: 19s - loss: 7.6721 - accuracy: 0.4996
19680/25000 [======================>.......] - ETA: 19s - loss: 7.6713 - accuracy: 0.4997
19712/25000 [======================>.......] - ETA: 19s - loss: 7.6752 - accuracy: 0.4994
19744/25000 [======================>.......] - ETA: 19s - loss: 7.6759 - accuracy: 0.4994
19776/25000 [======================>.......] - ETA: 19s - loss: 7.6713 - accuracy: 0.4997
19808/25000 [======================>.......] - ETA: 18s - loss: 7.6682 - accuracy: 0.4999
19840/25000 [======================>.......] - ETA: 18s - loss: 7.6697 - accuracy: 0.4998
19872/25000 [======================>.......] - ETA: 18s - loss: 7.6682 - accuracy: 0.4999
19904/25000 [======================>.......] - ETA: 18s - loss: 7.6697 - accuracy: 0.4998
19936/25000 [======================>.......] - ETA: 18s - loss: 7.6689 - accuracy: 0.4998
19968/25000 [======================>.......] - ETA: 18s - loss: 7.6735 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 18s - loss: 7.6682 - accuracy: 0.4999
20032/25000 [=======================>......] - ETA: 18s - loss: 7.6643 - accuracy: 0.5001
20064/25000 [=======================>......] - ETA: 17s - loss: 7.6704 - accuracy: 0.4998
20096/25000 [=======================>......] - ETA: 17s - loss: 7.6704 - accuracy: 0.4998
20128/25000 [=======================>......] - ETA: 17s - loss: 7.6750 - accuracy: 0.4995
20160/25000 [=======================>......] - ETA: 17s - loss: 7.6719 - accuracy: 0.4997
20192/25000 [=======================>......] - ETA: 17s - loss: 7.6712 - accuracy: 0.4997
20224/25000 [=======================>......] - ETA: 17s - loss: 7.6704 - accuracy: 0.4998
20256/25000 [=======================>......] - ETA: 17s - loss: 7.6696 - accuracy: 0.4998
20288/25000 [=======================>......] - ETA: 17s - loss: 7.6712 - accuracy: 0.4997
20320/25000 [=======================>......] - ETA: 17s - loss: 7.6711 - accuracy: 0.4997
20352/25000 [=======================>......] - ETA: 16s - loss: 7.6704 - accuracy: 0.4998
20384/25000 [=======================>......] - ETA: 16s - loss: 7.6756 - accuracy: 0.4994
20416/25000 [=======================>......] - ETA: 16s - loss: 7.6749 - accuracy: 0.4995
20448/25000 [=======================>......] - ETA: 16s - loss: 7.6764 - accuracy: 0.4994
20480/25000 [=======================>......] - ETA: 16s - loss: 7.6778 - accuracy: 0.4993
20512/25000 [=======================>......] - ETA: 16s - loss: 7.6756 - accuracy: 0.4994
20544/25000 [=======================>......] - ETA: 16s - loss: 7.6771 - accuracy: 0.4993
20576/25000 [=======================>......] - ETA: 16s - loss: 7.6748 - accuracy: 0.4995
20608/25000 [=======================>......] - ETA: 15s - loss: 7.6763 - accuracy: 0.4994
20640/25000 [=======================>......] - ETA: 15s - loss: 7.6770 - accuracy: 0.4993
20672/25000 [=======================>......] - ETA: 15s - loss: 7.6777 - accuracy: 0.4993
20704/25000 [=======================>......] - ETA: 15s - loss: 7.6762 - accuracy: 0.4994
20736/25000 [=======================>......] - ETA: 15s - loss: 7.6733 - accuracy: 0.4996
20768/25000 [=======================>......] - ETA: 15s - loss: 7.6666 - accuracy: 0.5000
20800/25000 [=======================>......] - ETA: 15s - loss: 7.6681 - accuracy: 0.4999
20832/25000 [=======================>......] - ETA: 15s - loss: 7.6674 - accuracy: 0.5000
20864/25000 [========================>.....] - ETA: 15s - loss: 7.6651 - accuracy: 0.5001
20896/25000 [========================>.....] - ETA: 14s - loss: 7.6659 - accuracy: 0.5000
20928/25000 [========================>.....] - ETA: 14s - loss: 7.6666 - accuracy: 0.5000
20960/25000 [========================>.....] - ETA: 14s - loss: 7.6659 - accuracy: 0.5000
20992/25000 [========================>.....] - ETA: 14s - loss: 7.6644 - accuracy: 0.5001
21024/25000 [========================>.....] - ETA: 14s - loss: 7.6637 - accuracy: 0.5002
21056/25000 [========================>.....] - ETA: 14s - loss: 7.6659 - accuracy: 0.5000
21088/25000 [========================>.....] - ETA: 14s - loss: 7.6695 - accuracy: 0.4998
21120/25000 [========================>.....] - ETA: 14s - loss: 7.6681 - accuracy: 0.4999
21152/25000 [========================>.....] - ETA: 14s - loss: 7.6695 - accuracy: 0.4998
21184/25000 [========================>.....] - ETA: 13s - loss: 7.6695 - accuracy: 0.4998
21216/25000 [========================>.....] - ETA: 13s - loss: 7.6717 - accuracy: 0.4997
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6717 - accuracy: 0.4997
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6702 - accuracy: 0.4998
21312/25000 [========================>.....] - ETA: 13s - loss: 7.6695 - accuracy: 0.4998
21344/25000 [========================>.....] - ETA: 13s - loss: 7.6688 - accuracy: 0.4999
21376/25000 [========================>.....] - ETA: 13s - loss: 7.6716 - accuracy: 0.4997
21408/25000 [========================>.....] - ETA: 13s - loss: 7.6723 - accuracy: 0.4996
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6716 - accuracy: 0.4997
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6695 - accuracy: 0.4998
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6688 - accuracy: 0.4999
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6680 - accuracy: 0.4999
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6652 - accuracy: 0.5001
21600/25000 [========================>.....] - ETA: 12s - loss: 7.6609 - accuracy: 0.5004
21632/25000 [========================>.....] - ETA: 12s - loss: 7.6631 - accuracy: 0.5002
21664/25000 [========================>.....] - ETA: 12s - loss: 7.6631 - accuracy: 0.5002
21696/25000 [=========================>....] - ETA: 12s - loss: 7.6603 - accuracy: 0.5004
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6596 - accuracy: 0.5005
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6603 - accuracy: 0.5004
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6603 - accuracy: 0.5004
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6575 - accuracy: 0.5006
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6575 - accuracy: 0.5006
21888/25000 [=========================>....] - ETA: 11s - loss: 7.6589 - accuracy: 0.5005
21920/25000 [=========================>....] - ETA: 11s - loss: 7.6589 - accuracy: 0.5005
21952/25000 [=========================>....] - ETA: 11s - loss: 7.6603 - accuracy: 0.5004
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6589 - accuracy: 0.5005
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6590 - accuracy: 0.5005
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6583 - accuracy: 0.5005
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6576 - accuracy: 0.5006
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6569 - accuracy: 0.5006
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6562 - accuracy: 0.5007
22176/25000 [=========================>....] - ETA: 10s - loss: 7.6521 - accuracy: 0.5009
22208/25000 [=========================>....] - ETA: 10s - loss: 7.6535 - accuracy: 0.5009
22240/25000 [=========================>....] - ETA: 10s - loss: 7.6487 - accuracy: 0.5012
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6487 - accuracy: 0.5012 
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6487 - accuracy: 0.5012
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6488 - accuracy: 0.5012
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6454 - accuracy: 0.5014
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6495 - accuracy: 0.5011
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6482 - accuracy: 0.5012
22464/25000 [=========================>....] - ETA: 9s - loss: 7.6489 - accuracy: 0.5012
22496/25000 [=========================>....] - ETA: 9s - loss: 7.6455 - accuracy: 0.5014
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6469 - accuracy: 0.5013
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6483 - accuracy: 0.5012
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6483 - accuracy: 0.5012
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6470 - accuracy: 0.5013
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6483 - accuracy: 0.5012
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6504 - accuracy: 0.5011
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6518 - accuracy: 0.5010
22752/25000 [==========================>...] - ETA: 8s - loss: 7.6504 - accuracy: 0.5011
22784/25000 [==========================>...] - ETA: 8s - loss: 7.6511 - accuracy: 0.5010
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6532 - accuracy: 0.5009
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6566 - accuracy: 0.5007
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6539 - accuracy: 0.5008
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6552 - accuracy: 0.5007
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6533 - accuracy: 0.5009
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6539 - accuracy: 0.5008
23008/25000 [==========================>...] - ETA: 7s - loss: 7.6560 - accuracy: 0.5007
23040/25000 [==========================>...] - ETA: 7s - loss: 7.6526 - accuracy: 0.5009
23072/25000 [==========================>...] - ETA: 7s - loss: 7.6580 - accuracy: 0.5006
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6573 - accuracy: 0.5006
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6600 - accuracy: 0.5004
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6593 - accuracy: 0.5005
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6640 - accuracy: 0.5002
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6640 - accuracy: 0.5002
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6660 - accuracy: 0.5000
23296/25000 [==========================>...] - ETA: 6s - loss: 7.6653 - accuracy: 0.5001
23328/25000 [==========================>...] - ETA: 6s - loss: 7.6627 - accuracy: 0.5003
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6647 - accuracy: 0.5001
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6686 - accuracy: 0.4999
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6692 - accuracy: 0.4998
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6725 - accuracy: 0.4996
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6686 - accuracy: 0.4999
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6718 - accuracy: 0.4997
23584/25000 [===========================>..] - ETA: 5s - loss: 7.6757 - accuracy: 0.4994
23616/25000 [===========================>..] - ETA: 5s - loss: 7.6764 - accuracy: 0.4994
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6744 - accuracy: 0.4995
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6744 - accuracy: 0.4995
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6744 - accuracy: 0.4995
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6731 - accuracy: 0.4996
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6718 - accuracy: 0.4997
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23872/25000 [===========================>..] - ETA: 4s - loss: 7.6692 - accuracy: 0.4998
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6718 - accuracy: 0.4997
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6698 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6711 - accuracy: 0.4997
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6704 - accuracy: 0.4998
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6704 - accuracy: 0.4998
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6717 - accuracy: 0.4997
24160/25000 [===========================>..] - ETA: 3s - loss: 7.6742 - accuracy: 0.4995
24192/25000 [============================>.] - ETA: 2s - loss: 7.6742 - accuracy: 0.4995
24224/25000 [============================>.] - ETA: 2s - loss: 7.6742 - accuracy: 0.4995
24256/25000 [============================>.] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24288/25000 [============================>.] - ETA: 2s - loss: 7.6723 - accuracy: 0.4996
24320/25000 [============================>.] - ETA: 2s - loss: 7.6729 - accuracy: 0.4996
24352/25000 [============================>.] - ETA: 2s - loss: 7.6735 - accuracy: 0.4995
24384/25000 [============================>.] - ETA: 2s - loss: 7.6742 - accuracy: 0.4995
24416/25000 [============================>.] - ETA: 2s - loss: 7.6735 - accuracy: 0.4995
24448/25000 [============================>.] - ETA: 2s - loss: 7.6716 - accuracy: 0.4997
24480/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24512/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24544/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24576/25000 [============================>.] - ETA: 1s - loss: 7.6722 - accuracy: 0.4996
24608/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
24640/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24672/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24704/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24736/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24768/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24800/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24832/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24864/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24896/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24928/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24960/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
25000/25000 [==============================] - 109s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
