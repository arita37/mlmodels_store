
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '51b64e342c7b2661e79b8abaa33db92672ae95c7', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ************************************************************************************************************************
/home/runner/work/mlmodels/mlmodels/mlmodels/example/
############ List of files ################################
['ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_svm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//timeseries_m5_deepar.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//fashion_MNIST_mlmodels.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_home_retail.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras_charcnn_reuters.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vison_fashion_MNIST.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow_1_lstm.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//keras-textcnn.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn_titanic_randomForest_example2.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//mnist_mlmodels_.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//gluon_automl_titanic.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//tensorflow__lstm_json.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//sklearn.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_titanic.ipynb', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//vision_mnist.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m4.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_hyper.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//lightgbm_glass.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//benchmark_timeseries_m5.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example//arun_model.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m4.py', 'ipython /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py']





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
ipython https://github.com/arita37/mlmodels/blob/dev/mlmodels/example//timeseries_m5_deepar.ipynb 

UsageError: Line magic function `%%capture` not found.





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
	Data preprocessing and feature engineering runtime = 0.22s ...
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
 40%|████      | 2/5 [00:48<01:13, 24.41s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.42985609373526995, 'embedding_size_factor': 1.232819178209116, 'layers.choice': 0, 'learning_rate': 0.0002551480065392433, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.004722855796527332} and reward: 0.3722
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdb\x82\xc3"%\x13+X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\xb9\xa0\x9aD\xa2\xd0X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?0\xb8\xacW\xfeg`X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?sXE\xf3\tu\x14u.' and reward: 0.3722
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdb\x82\xc3"%\x13+X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\xb9\xa0\x9aD\xa2\xd0X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?0\xb8\xacW\xfeg`X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?sXE\xf3\tu\x14u.' and reward: 0.3722
 60%|██████    | 3/5 [01:34<01:01, 30.77s/it] 60%|██████    | 3/5 [01:34<01:02, 31.48s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.07756423375054673, 'embedding_size_factor': 1.4867507743937378, 'layers.choice': 1, 'learning_rate': 0.0035220501374837944, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 5.15186553243304e-06} and reward: 0.371
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xb3\xdb?\xe7LC\x05X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7\xc9\xbb.\x15.2X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?l\xdaFD\xf8\xc0qX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xd5\x9b\xc6\x03\xb0\x83lu.' and reward: 0.371
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xb3\xdb?\xe7LC\x05X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7\xc9\xbb.\x15.2X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?l\xdaFD\xf8\xc0qX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xd5\x9b\xc6\x03\xb0\x83lu.' and reward: 0.371
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 199.80791234970093
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.78s of the -82.21s of remaining time.
Ensemble size: 25
Ensemble weights: 
[0.68 0.2  0.12]
	0.3902	 = Validation accuracy score
	0.91s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 203.16s ...
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 526, in main
    test_cli(arg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 456, in test_cli
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f918ac5d940> 

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
 [-0.03043718 -0.11543512  0.06451724  0.07746897 -0.05372668  0.00130523]
 [-0.25864279 -0.0950458   0.06333264  0.03413699  0.18157877  0.1091811 ]
 [-0.03416693  0.00081687 -0.05651638  0.06262468  0.19049153 -0.14271845]
 [ 0.44146624 -0.07738522 -0.08742023 -0.12874    -0.15293829 -0.15034856]
 [ 0.20669851  0.17606622  0.1427739   0.25902864 -0.13724275  0.46476135]
 [-0.03814562 -0.61588353 -0.50460851  0.02016877  0.61792052 -0.71250319]
 [ 0.77637202 -0.01135493  0.35440829  0.18324141 -0.44233924 -0.19743265]
 [ 0.59394091 -0.05416103 -0.21078907 -0.20511903 -0.39596701 -0.11192165]
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
{'loss': 0.38444624096155167, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-22 22:33:19.589014: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 526, in main
    test_cli(arg)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 458, in test_cli
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
{'loss': 0.39460564032197, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-22 22:33:20.594487: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Failed Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "opt/hostedtoolcache/Python/3.6.10/x64/bin/ml_models", line 11, in <module>
    load_entry_point('mlmodels', 'console_scripts', 'ml_models')()
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 526, in main
    test_cli(arg)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 460, in test_cli
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
 1392640/17464789 [=>............................] - ETA: 0s
 4104192/17464789 [======>.......................] - ETA: 0s
 6856704/17464789 [==========>...................] - ETA: 0s
 9773056/17464789 [===============>..............] - ETA: 0s
13049856/17464789 [=====================>........] - ETA: 0s
15998976/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-22 22:33:31.064084: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-22 22:33:31.067964: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-22 22:33:31.068363: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5604b2fee080 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 22:33:31.068685: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:05 - loss: 7.6666 - accuracy: 0.5000
   64/25000 [..............................] - ETA: 2:45 - loss: 7.6666 - accuracy: 0.5000
   96/25000 [..............................] - ETA: 2:20 - loss: 7.6666 - accuracy: 0.5000
  128/25000 [..............................] - ETA: 2:08 - loss: 7.7864 - accuracy: 0.4922
  160/25000 [..............................] - ETA: 1:57 - loss: 7.7625 - accuracy: 0.4938
  192/25000 [..............................] - ETA: 1:51 - loss: 7.7465 - accuracy: 0.4948
  224/25000 [..............................] - ETA: 1:47 - loss: 7.5297 - accuracy: 0.5089
  256/25000 [..............................] - ETA: 1:43 - loss: 7.7864 - accuracy: 0.4922
  288/25000 [..............................] - ETA: 1:42 - loss: 7.6134 - accuracy: 0.5035
  320/25000 [..............................] - ETA: 1:40 - loss: 7.6666 - accuracy: 0.5000
  352/25000 [..............................] - ETA: 1:37 - loss: 7.8409 - accuracy: 0.4886
  384/25000 [..............................] - ETA: 1:36 - loss: 7.8263 - accuracy: 0.4896
  416/25000 [..............................] - ETA: 1:34 - loss: 7.9615 - accuracy: 0.4808
  448/25000 [..............................] - ETA: 1:32 - loss: 7.8377 - accuracy: 0.4888
  480/25000 [..............................] - ETA: 1:31 - loss: 7.7625 - accuracy: 0.4938
  512/25000 [..............................] - ETA: 1:30 - loss: 7.6966 - accuracy: 0.4980
  544/25000 [..............................] - ETA: 1:30 - loss: 7.6384 - accuracy: 0.5018
  576/25000 [..............................] - ETA: 1:29 - loss: 7.7199 - accuracy: 0.4965
  608/25000 [..............................] - ETA: 1:29 - loss: 7.6918 - accuracy: 0.4984
  640/25000 [..............................] - ETA: 1:28 - loss: 7.7385 - accuracy: 0.4953
  672/25000 [..............................] - ETA: 1:28 - loss: 7.7351 - accuracy: 0.4955
  704/25000 [..............................] - ETA: 1:28 - loss: 7.6884 - accuracy: 0.4986
  736/25000 [..............................] - ETA: 1:27 - loss: 7.6458 - accuracy: 0.5014
  768/25000 [..............................] - ETA: 1:26 - loss: 7.7065 - accuracy: 0.4974
  800/25000 [..............................] - ETA: 1:26 - loss: 7.7816 - accuracy: 0.4925
  832/25000 [..............................] - ETA: 1:26 - loss: 7.7956 - accuracy: 0.4916
  864/25000 [>.............................] - ETA: 1:26 - loss: 7.7554 - accuracy: 0.4942
  896/25000 [>.............................] - ETA: 1:25 - loss: 7.7351 - accuracy: 0.4955
  928/25000 [>.............................] - ETA: 1:25 - loss: 7.6997 - accuracy: 0.4978
  960/25000 [>.............................] - ETA: 1:24 - loss: 7.6187 - accuracy: 0.5031
  992/25000 [>.............................] - ETA: 1:24 - loss: 7.5739 - accuracy: 0.5060
 1024/25000 [>.............................] - ETA: 1:24 - loss: 7.5618 - accuracy: 0.5068
 1056/25000 [>.............................] - ETA: 1:24 - loss: 7.5359 - accuracy: 0.5085
 1088/25000 [>.............................] - ETA: 1:23 - loss: 7.5962 - accuracy: 0.5046
 1120/25000 [>.............................] - ETA: 1:23 - loss: 7.6119 - accuracy: 0.5036
 1152/25000 [>.............................] - ETA: 1:23 - loss: 7.5468 - accuracy: 0.5078
 1184/25000 [>.............................] - ETA: 1:22 - loss: 7.5112 - accuracy: 0.5101
 1216/25000 [>.............................] - ETA: 1:22 - loss: 7.5027 - accuracy: 0.5107
 1248/25000 [>.............................] - ETA: 1:22 - loss: 7.4700 - accuracy: 0.5128
 1280/25000 [>.............................] - ETA: 1:22 - loss: 7.4630 - accuracy: 0.5133
 1312/25000 [>.............................] - ETA: 1:22 - loss: 7.5030 - accuracy: 0.5107
 1344/25000 [>.............................] - ETA: 1:22 - loss: 7.5411 - accuracy: 0.5082
 1376/25000 [>.............................] - ETA: 1:21 - loss: 7.5106 - accuracy: 0.5102
 1408/25000 [>.............................] - ETA: 1:21 - loss: 7.4924 - accuracy: 0.5114
 1440/25000 [>.............................] - ETA: 1:21 - loss: 7.4963 - accuracy: 0.5111
 1472/25000 [>.............................] - ETA: 1:21 - loss: 7.4687 - accuracy: 0.5129
 1504/25000 [>.............................] - ETA: 1:20 - loss: 7.4729 - accuracy: 0.5126
 1536/25000 [>.............................] - ETA: 1:20 - loss: 7.4869 - accuracy: 0.5117
 1568/25000 [>.............................] - ETA: 1:20 - loss: 7.4710 - accuracy: 0.5128
 1600/25000 [>.............................] - ETA: 1:20 - loss: 7.4462 - accuracy: 0.5144
 1632/25000 [>.............................] - ETA: 1:20 - loss: 7.4975 - accuracy: 0.5110
 1664/25000 [>.............................] - ETA: 1:20 - loss: 7.5100 - accuracy: 0.5102
 1696/25000 [=>............................] - ETA: 1:20 - loss: 7.4677 - accuracy: 0.5130
 1728/25000 [=>............................] - ETA: 1:20 - loss: 7.5158 - accuracy: 0.5098
 1760/25000 [=>............................] - ETA: 1:19 - loss: 7.5098 - accuracy: 0.5102
 1792/25000 [=>............................] - ETA: 1:19 - loss: 7.5126 - accuracy: 0.5100
 1824/25000 [=>............................] - ETA: 1:19 - loss: 7.5237 - accuracy: 0.5093
 1856/25000 [=>............................] - ETA: 1:19 - loss: 7.5179 - accuracy: 0.5097
 1888/25000 [=>............................] - ETA: 1:19 - loss: 7.4961 - accuracy: 0.5111
 1920/25000 [=>............................] - ETA: 1:19 - loss: 7.4989 - accuracy: 0.5109
 1952/25000 [=>............................] - ETA: 1:19 - loss: 7.4860 - accuracy: 0.5118
 1984/25000 [=>............................] - ETA: 1:19 - loss: 7.4734 - accuracy: 0.5126
 2016/25000 [=>............................] - ETA: 1:19 - loss: 7.5449 - accuracy: 0.5079
 2048/25000 [=>............................] - ETA: 1:18 - loss: 7.5244 - accuracy: 0.5093
 2080/25000 [=>............................] - ETA: 1:18 - loss: 7.5560 - accuracy: 0.5072
 2112/25000 [=>............................] - ETA: 1:18 - loss: 7.5577 - accuracy: 0.5071
 2144/25000 [=>............................] - ETA: 1:18 - loss: 7.5593 - accuracy: 0.5070
 2176/25000 [=>............................] - ETA: 1:18 - loss: 7.5821 - accuracy: 0.5055
 2208/25000 [=>............................] - ETA: 1:18 - loss: 7.5972 - accuracy: 0.5045
 2240/25000 [=>............................] - ETA: 1:17 - loss: 7.6050 - accuracy: 0.5040
 2272/25000 [=>............................] - ETA: 1:17 - loss: 7.5856 - accuracy: 0.5053
 2304/25000 [=>............................] - ETA: 1:17 - loss: 7.6001 - accuracy: 0.5043
 2336/25000 [=>............................] - ETA: 1:17 - loss: 7.5813 - accuracy: 0.5056
 2368/25000 [=>............................] - ETA: 1:17 - loss: 7.5695 - accuracy: 0.5063
 2400/25000 [=>............................] - ETA: 1:17 - loss: 7.6027 - accuracy: 0.5042
 2432/25000 [=>............................] - ETA: 1:16 - loss: 7.6162 - accuracy: 0.5033
 2464/25000 [=>............................] - ETA: 1:16 - loss: 7.6355 - accuracy: 0.5020
 2496/25000 [=>............................] - ETA: 1:16 - loss: 7.6482 - accuracy: 0.5012
 2528/25000 [==>...........................] - ETA: 1:16 - loss: 7.6545 - accuracy: 0.5008
 2560/25000 [==>...........................] - ETA: 1:16 - loss: 7.6427 - accuracy: 0.5016
 2592/25000 [==>...........................] - ETA: 1:16 - loss: 7.6489 - accuracy: 0.5012
 2624/25000 [==>...........................] - ETA: 1:16 - loss: 7.6666 - accuracy: 0.5000
 2656/25000 [==>...........................] - ETA: 1:15 - loss: 7.7013 - accuracy: 0.4977
 2688/25000 [==>...........................] - ETA: 1:15 - loss: 7.7008 - accuracy: 0.4978
 2720/25000 [==>...........................] - ETA: 1:15 - loss: 7.7117 - accuracy: 0.4971
 2752/25000 [==>...........................] - ETA: 1:15 - loss: 7.7112 - accuracy: 0.4971
 2784/25000 [==>...........................] - ETA: 1:15 - loss: 7.7107 - accuracy: 0.4971
 2816/25000 [==>...........................] - ETA: 1:15 - loss: 7.7156 - accuracy: 0.4968
 2848/25000 [==>...........................] - ETA: 1:15 - loss: 7.7312 - accuracy: 0.4958
 2880/25000 [==>...........................] - ETA: 1:14 - loss: 7.7412 - accuracy: 0.4951
 2912/25000 [==>...........................] - ETA: 1:14 - loss: 7.7509 - accuracy: 0.4945
 2944/25000 [==>...........................] - ETA: 1:14 - loss: 7.7500 - accuracy: 0.4946
 2976/25000 [==>...........................] - ETA: 1:14 - loss: 7.7697 - accuracy: 0.4933
 3008/25000 [==>...........................] - ETA: 1:14 - loss: 7.7788 - accuracy: 0.4927
 3040/25000 [==>...........................] - ETA: 1:14 - loss: 7.7978 - accuracy: 0.4914
 3072/25000 [==>...........................] - ETA: 1:14 - loss: 7.7914 - accuracy: 0.4919
 3104/25000 [==>...........................] - ETA: 1:13 - loss: 7.7654 - accuracy: 0.4936
 3136/25000 [==>...........................] - ETA: 1:13 - loss: 7.7497 - accuracy: 0.4946
 3168/25000 [==>...........................] - ETA: 1:13 - loss: 7.7392 - accuracy: 0.4953
 3200/25000 [==>...........................] - ETA: 1:13 - loss: 7.7433 - accuracy: 0.4950
 3232/25000 [==>...........................] - ETA: 1:13 - loss: 7.7568 - accuracy: 0.4941
 3264/25000 [==>...........................] - ETA: 1:13 - loss: 7.7700 - accuracy: 0.4933
 3296/25000 [==>...........................] - ETA: 1:13 - loss: 7.7783 - accuracy: 0.4927
 3328/25000 [==>...........................] - ETA: 1:13 - loss: 7.7726 - accuracy: 0.4931
 3360/25000 [===>..........................] - ETA: 1:13 - loss: 7.7853 - accuracy: 0.4923
 3392/25000 [===>..........................] - ETA: 1:13 - loss: 7.7706 - accuracy: 0.4932
 3424/25000 [===>..........................] - ETA: 1:12 - loss: 7.7741 - accuracy: 0.4930
 3456/25000 [===>..........................] - ETA: 1:12 - loss: 7.7775 - accuracy: 0.4928
 3488/25000 [===>..........................] - ETA: 1:12 - loss: 7.7853 - accuracy: 0.4923
 3520/25000 [===>..........................] - ETA: 1:12 - loss: 7.8017 - accuracy: 0.4912
 3552/25000 [===>..........................] - ETA: 1:12 - loss: 7.8004 - accuracy: 0.4913
 3584/25000 [===>..........................] - ETA: 1:12 - loss: 7.8121 - accuracy: 0.4905
 3616/25000 [===>..........................] - ETA: 1:12 - loss: 7.8362 - accuracy: 0.4889
 3648/25000 [===>..........................] - ETA: 1:12 - loss: 7.8474 - accuracy: 0.4882
 3680/25000 [===>..........................] - ETA: 1:11 - loss: 7.8416 - accuracy: 0.4886
 3712/25000 [===>..........................] - ETA: 1:11 - loss: 7.8195 - accuracy: 0.4900
 3744/25000 [===>..........................] - ETA: 1:11 - loss: 7.8141 - accuracy: 0.4904
 3776/25000 [===>..........................] - ETA: 1:11 - loss: 7.8169 - accuracy: 0.4902
 3808/25000 [===>..........................] - ETA: 1:11 - loss: 7.8237 - accuracy: 0.4898
 3840/25000 [===>..........................] - ETA: 1:11 - loss: 7.8343 - accuracy: 0.4891
 3872/25000 [===>..........................] - ETA: 1:11 - loss: 7.8329 - accuracy: 0.4892
 3904/25000 [===>..........................] - ETA: 1:11 - loss: 7.8394 - accuracy: 0.4887
 3936/25000 [===>..........................] - ETA: 1:11 - loss: 7.8263 - accuracy: 0.4896
 3968/25000 [===>..........................] - ETA: 1:11 - loss: 7.8289 - accuracy: 0.4894
 4000/25000 [===>..........................] - ETA: 1:11 - loss: 7.8238 - accuracy: 0.4897
 4032/25000 [===>..........................] - ETA: 1:10 - loss: 7.8225 - accuracy: 0.4898
 4064/25000 [===>..........................] - ETA: 1:10 - loss: 7.8138 - accuracy: 0.4904
 4096/25000 [===>..........................] - ETA: 1:10 - loss: 7.8126 - accuracy: 0.4905
 4128/25000 [===>..........................] - ETA: 1:10 - loss: 7.7855 - accuracy: 0.4922
 4160/25000 [===>..........................] - ETA: 1:10 - loss: 7.7735 - accuracy: 0.4930
 4192/25000 [====>.........................] - ETA: 1:10 - loss: 7.7654 - accuracy: 0.4936
 4224/25000 [====>.........................] - ETA: 1:10 - loss: 7.7683 - accuracy: 0.4934
 4256/25000 [====>.........................] - ETA: 1:10 - loss: 7.7531 - accuracy: 0.4944
 4288/25000 [====>.........................] - ETA: 1:09 - loss: 7.7560 - accuracy: 0.4942
 4320/25000 [====>.........................] - ETA: 1:09 - loss: 7.7518 - accuracy: 0.4944
 4352/25000 [====>.........................] - ETA: 1:09 - loss: 7.7512 - accuracy: 0.4945
 4384/25000 [====>.........................] - ETA: 1:09 - loss: 7.7646 - accuracy: 0.4936
 4416/25000 [====>.........................] - ETA: 1:09 - loss: 7.7673 - accuracy: 0.4934
 4448/25000 [====>.........................] - ETA: 1:09 - loss: 7.7528 - accuracy: 0.4944
 4480/25000 [====>.........................] - ETA: 1:09 - loss: 7.7488 - accuracy: 0.4946
 4512/25000 [====>.........................] - ETA: 1:09 - loss: 7.7346 - accuracy: 0.4956
 4544/25000 [====>.........................] - ETA: 1:09 - loss: 7.7442 - accuracy: 0.4949
 4576/25000 [====>.........................] - ETA: 1:08 - loss: 7.7504 - accuracy: 0.4945
 4608/25000 [====>.........................] - ETA: 1:08 - loss: 7.7631 - accuracy: 0.4937
 4640/25000 [====>.........................] - ETA: 1:08 - loss: 7.7658 - accuracy: 0.4935
 4672/25000 [====>.........................] - ETA: 1:08 - loss: 7.7618 - accuracy: 0.4938
 4704/25000 [====>.........................] - ETA: 1:08 - loss: 7.7677 - accuracy: 0.4934
 4736/25000 [====>.........................] - ETA: 1:08 - loss: 7.7637 - accuracy: 0.4937
 4768/25000 [====>.........................] - ETA: 1:08 - loss: 7.7663 - accuracy: 0.4935
 4800/25000 [====>.........................] - ETA: 1:07 - loss: 7.7625 - accuracy: 0.4938
 4832/25000 [====>.........................] - ETA: 1:07 - loss: 7.7745 - accuracy: 0.4930
 4864/25000 [====>.........................] - ETA: 1:07 - loss: 7.7738 - accuracy: 0.4930
 4896/25000 [====>.........................] - ETA: 1:07 - loss: 7.7856 - accuracy: 0.4922
 4928/25000 [====>.........................] - ETA: 1:07 - loss: 7.7849 - accuracy: 0.4923
 4960/25000 [====>.........................] - ETA: 1:07 - loss: 7.7872 - accuracy: 0.4921
 4992/25000 [====>.........................] - ETA: 1:07 - loss: 7.7895 - accuracy: 0.4920
 5024/25000 [=====>........................] - ETA: 1:07 - loss: 7.7795 - accuracy: 0.4926
 5056/25000 [=====>........................] - ETA: 1:07 - loss: 7.7728 - accuracy: 0.4931
 5088/25000 [=====>........................] - ETA: 1:07 - loss: 7.7751 - accuracy: 0.4929
 5120/25000 [=====>........................] - ETA: 1:07 - loss: 7.7924 - accuracy: 0.4918
 5152/25000 [=====>........................] - ETA: 1:06 - loss: 7.7827 - accuracy: 0.4924
 5184/25000 [=====>........................] - ETA: 1:06 - loss: 7.7849 - accuracy: 0.4923
 5216/25000 [=====>........................] - ETA: 1:06 - loss: 7.7754 - accuracy: 0.4929
 5248/25000 [=====>........................] - ETA: 1:06 - loss: 7.7747 - accuracy: 0.4929
 5280/25000 [=====>........................] - ETA: 1:06 - loss: 7.7683 - accuracy: 0.4934
 5312/25000 [=====>........................] - ETA: 1:06 - loss: 7.7907 - accuracy: 0.4919
 5344/25000 [=====>........................] - ETA: 1:06 - loss: 7.7929 - accuracy: 0.4918
 5376/25000 [=====>........................] - ETA: 1:06 - loss: 7.7921 - accuracy: 0.4918
 5408/25000 [=====>........................] - ETA: 1:05 - loss: 7.7914 - accuracy: 0.4919
 5440/25000 [=====>........................] - ETA: 1:05 - loss: 7.7878 - accuracy: 0.4921
 5472/25000 [=====>........................] - ETA: 1:05 - loss: 7.7759 - accuracy: 0.4929
 5504/25000 [=====>........................] - ETA: 1:05 - loss: 7.7753 - accuracy: 0.4929
 5536/25000 [=====>........................] - ETA: 1:05 - loss: 7.7719 - accuracy: 0.4931
 5568/25000 [=====>........................] - ETA: 1:05 - loss: 7.7823 - accuracy: 0.4925
 5600/25000 [=====>........................] - ETA: 1:05 - loss: 7.7980 - accuracy: 0.4914
 5632/25000 [=====>........................] - ETA: 1:04 - loss: 7.8027 - accuracy: 0.4911
 5664/25000 [=====>........................] - ETA: 1:04 - loss: 7.7857 - accuracy: 0.4922
 5696/25000 [=====>........................] - ETA: 1:04 - loss: 7.7797 - accuracy: 0.4926
 5728/25000 [=====>........................] - ETA: 1:04 - loss: 7.7710 - accuracy: 0.4932
 5760/25000 [=====>........................] - ETA: 1:04 - loss: 7.7678 - accuracy: 0.4934
 5792/25000 [=====>........................] - ETA: 1:04 - loss: 7.7699 - accuracy: 0.4933
 5824/25000 [=====>........................] - ETA: 1:04 - loss: 7.7588 - accuracy: 0.4940
 5856/25000 [======>.......................] - ETA: 1:04 - loss: 7.7583 - accuracy: 0.4940
 5888/25000 [======>.......................] - ETA: 1:03 - loss: 7.7473 - accuracy: 0.4947
 5920/25000 [======>.......................] - ETA: 1:03 - loss: 7.7443 - accuracy: 0.4949
 5952/25000 [======>.......................] - ETA: 1:03 - loss: 7.7336 - accuracy: 0.4956
 5984/25000 [======>.......................] - ETA: 1:03 - loss: 7.7307 - accuracy: 0.4958
 6016/25000 [======>.......................] - ETA: 1:03 - loss: 7.7431 - accuracy: 0.4950
 6048/25000 [======>.......................] - ETA: 1:03 - loss: 7.7477 - accuracy: 0.4947
 6080/25000 [======>.......................] - ETA: 1:03 - loss: 7.7473 - accuracy: 0.4947
 6112/25000 [======>.......................] - ETA: 1:03 - loss: 7.7444 - accuracy: 0.4949
 6144/25000 [======>.......................] - ETA: 1:02 - loss: 7.7490 - accuracy: 0.4946
 6176/25000 [======>.......................] - ETA: 1:02 - loss: 7.7510 - accuracy: 0.4945
 6208/25000 [======>.......................] - ETA: 1:02 - loss: 7.7531 - accuracy: 0.4944
 6240/25000 [======>.......................] - ETA: 1:02 - loss: 7.7551 - accuracy: 0.4942
 6272/25000 [======>.......................] - ETA: 1:02 - loss: 7.7644 - accuracy: 0.4936
 6304/25000 [======>.......................] - ETA: 1:02 - loss: 7.7639 - accuracy: 0.4937
 6336/25000 [======>.......................] - ETA: 1:02 - loss: 7.7683 - accuracy: 0.4934
 6368/25000 [======>.......................] - ETA: 1:02 - loss: 7.7605 - accuracy: 0.4939
 6400/25000 [======>.......................] - ETA: 1:01 - loss: 7.7601 - accuracy: 0.4939
 6432/25000 [======>.......................] - ETA: 1:01 - loss: 7.7572 - accuracy: 0.4941
 6464/25000 [======>.......................] - ETA: 1:01 - loss: 7.7591 - accuracy: 0.4940
 6496/25000 [======>.......................] - ETA: 1:01 - loss: 7.7705 - accuracy: 0.4932
 6528/25000 [======>.......................] - ETA: 1:01 - loss: 7.7653 - accuracy: 0.4936
 6560/25000 [======>.......................] - ETA: 1:01 - loss: 7.7625 - accuracy: 0.4938
 6592/25000 [======>.......................] - ETA: 1:01 - loss: 7.7690 - accuracy: 0.4933
 6624/25000 [======>.......................] - ETA: 1:01 - loss: 7.7685 - accuracy: 0.4934
 6656/25000 [======>.......................] - ETA: 1:01 - loss: 7.7703 - accuracy: 0.4932
 6688/25000 [=======>......................] - ETA: 1:00 - loss: 7.7652 - accuracy: 0.4936
 6720/25000 [=======>......................] - ETA: 1:00 - loss: 7.7625 - accuracy: 0.4938
 6752/25000 [=======>......................] - ETA: 1:00 - loss: 7.7597 - accuracy: 0.4939
 6784/25000 [=======>......................] - ETA: 1:00 - loss: 7.7683 - accuracy: 0.4934
 6816/25000 [=======>......................] - ETA: 1:00 - loss: 7.7589 - accuracy: 0.4940
 6848/25000 [=======>......................] - ETA: 1:00 - loss: 7.7562 - accuracy: 0.4942
 6880/25000 [=======>......................] - ETA: 1:00 - loss: 7.7580 - accuracy: 0.4940
 6912/25000 [=======>......................] - ETA: 1:00 - loss: 7.7576 - accuracy: 0.4941
 6944/25000 [=======>......................] - ETA: 59s - loss: 7.7505 - accuracy: 0.4945 
 6976/25000 [=======>......................] - ETA: 59s - loss: 7.7501 - accuracy: 0.4946
 7008/25000 [=======>......................] - ETA: 59s - loss: 7.7366 - accuracy: 0.4954
 7040/25000 [=======>......................] - ETA: 59s - loss: 7.7276 - accuracy: 0.4960
 7072/25000 [=======>......................] - ETA: 59s - loss: 7.7317 - accuracy: 0.4958
 7104/25000 [=======>......................] - ETA: 59s - loss: 7.7292 - accuracy: 0.4959
 7136/25000 [=======>......................] - ETA: 59s - loss: 7.7332 - accuracy: 0.4957
 7168/25000 [=======>......................] - ETA: 59s - loss: 7.7351 - accuracy: 0.4955
 7200/25000 [=======>......................] - ETA: 58s - loss: 7.7433 - accuracy: 0.4950
 7232/25000 [=======>......................] - ETA: 58s - loss: 7.7514 - accuracy: 0.4945
 7264/25000 [=======>......................] - ETA: 58s - loss: 7.7489 - accuracy: 0.4946
 7296/25000 [=======>......................] - ETA: 58s - loss: 7.7402 - accuracy: 0.4952
 7328/25000 [=======>......................] - ETA: 58s - loss: 7.7503 - accuracy: 0.4945
 7360/25000 [=======>......................] - ETA: 58s - loss: 7.7479 - accuracy: 0.4947
 7392/25000 [=======>......................] - ETA: 58s - loss: 7.7537 - accuracy: 0.4943
 7424/25000 [=======>......................] - ETA: 58s - loss: 7.7575 - accuracy: 0.4941
 7456/25000 [=======>......................] - ETA: 58s - loss: 7.7571 - accuracy: 0.4941
 7488/25000 [=======>......................] - ETA: 57s - loss: 7.7588 - accuracy: 0.4940
 7520/25000 [========>.....................] - ETA: 57s - loss: 7.7604 - accuracy: 0.4939
 7552/25000 [========>.....................] - ETA: 57s - loss: 7.7641 - accuracy: 0.4936
 7584/25000 [========>.....................] - ETA: 57s - loss: 7.7576 - accuracy: 0.4941
 7616/25000 [========>.....................] - ETA: 57s - loss: 7.7552 - accuracy: 0.4942
 7648/25000 [========>.....................] - ETA: 57s - loss: 7.7468 - accuracy: 0.4948
 7680/25000 [========>.....................] - ETA: 57s - loss: 7.7485 - accuracy: 0.4947
 7712/25000 [========>.....................] - ETA: 57s - loss: 7.7362 - accuracy: 0.4955
 7744/25000 [========>.....................] - ETA: 57s - loss: 7.7320 - accuracy: 0.4957
 7776/25000 [========>.....................] - ETA: 56s - loss: 7.7356 - accuracy: 0.4955
 7808/25000 [========>.....................] - ETA: 56s - loss: 7.7412 - accuracy: 0.4951
 7840/25000 [========>.....................] - ETA: 56s - loss: 7.7409 - accuracy: 0.4952
 7872/25000 [========>.....................] - ETA: 56s - loss: 7.7367 - accuracy: 0.4954
 7904/25000 [========>.....................] - ETA: 56s - loss: 7.7384 - accuracy: 0.4953
 7936/25000 [========>.....................] - ETA: 56s - loss: 7.7342 - accuracy: 0.4956
 7968/25000 [========>.....................] - ETA: 56s - loss: 7.7244 - accuracy: 0.4962
 8000/25000 [========>.....................] - ETA: 56s - loss: 7.7203 - accuracy: 0.4965
 8032/25000 [========>.....................] - ETA: 56s - loss: 7.7220 - accuracy: 0.4964
 8064/25000 [========>.....................] - ETA: 55s - loss: 7.7256 - accuracy: 0.4962
 8096/25000 [========>.....................] - ETA: 55s - loss: 7.7196 - accuracy: 0.4965
 8128/25000 [========>.....................] - ETA: 55s - loss: 7.7176 - accuracy: 0.4967
 8160/25000 [========>.....................] - ETA: 55s - loss: 7.7249 - accuracy: 0.4962
 8192/25000 [========>.....................] - ETA: 55s - loss: 7.7209 - accuracy: 0.4965
 8224/25000 [========>.....................] - ETA: 55s - loss: 7.7188 - accuracy: 0.4966
 8256/25000 [========>.....................] - ETA: 55s - loss: 7.7205 - accuracy: 0.4965
 8288/25000 [========>.....................] - ETA: 55s - loss: 7.7147 - accuracy: 0.4969
 8320/25000 [========>.....................] - ETA: 55s - loss: 7.7127 - accuracy: 0.4970
 8352/25000 [=========>....................] - ETA: 55s - loss: 7.7199 - accuracy: 0.4965
 8384/25000 [=========>....................] - ETA: 54s - loss: 7.7160 - accuracy: 0.4968
 8416/25000 [=========>....................] - ETA: 54s - loss: 7.7140 - accuracy: 0.4969
 8448/25000 [=========>....................] - ETA: 54s - loss: 7.7156 - accuracy: 0.4968
 8480/25000 [=========>....................] - ETA: 54s - loss: 7.7191 - accuracy: 0.4966
 8512/25000 [=========>....................] - ETA: 54s - loss: 7.7044 - accuracy: 0.4975
 8544/25000 [=========>....................] - ETA: 54s - loss: 7.7025 - accuracy: 0.4977
 8576/25000 [=========>....................] - ETA: 54s - loss: 7.7042 - accuracy: 0.4976
 8608/25000 [=========>....................] - ETA: 54s - loss: 7.7076 - accuracy: 0.4973
 8640/25000 [=========>....................] - ETA: 54s - loss: 7.7021 - accuracy: 0.4977
 8672/25000 [=========>....................] - ETA: 53s - loss: 7.6984 - accuracy: 0.4979
 8704/25000 [=========>....................] - ETA: 53s - loss: 7.6948 - accuracy: 0.4982
 8736/25000 [=========>....................] - ETA: 53s - loss: 7.6929 - accuracy: 0.4983
 8768/25000 [=========>....................] - ETA: 53s - loss: 7.6876 - accuracy: 0.4986
 8800/25000 [=========>....................] - ETA: 53s - loss: 7.6840 - accuracy: 0.4989
 8832/25000 [=========>....................] - ETA: 53s - loss: 7.6805 - accuracy: 0.4991
 8864/25000 [=========>....................] - ETA: 53s - loss: 7.6770 - accuracy: 0.4993
 8896/25000 [=========>....................] - ETA: 53s - loss: 7.6770 - accuracy: 0.4993
 8928/25000 [=========>....................] - ETA: 53s - loss: 7.6769 - accuracy: 0.4993
 8960/25000 [=========>....................] - ETA: 52s - loss: 7.6769 - accuracy: 0.4993
 8992/25000 [=========>....................] - ETA: 52s - loss: 7.6786 - accuracy: 0.4992
 9024/25000 [=========>....................] - ETA: 52s - loss: 7.6802 - accuracy: 0.4991
 9056/25000 [=========>....................] - ETA: 52s - loss: 7.6751 - accuracy: 0.4994
 9088/25000 [=========>....................] - ETA: 52s - loss: 7.6751 - accuracy: 0.4994
 9120/25000 [=========>....................] - ETA: 52s - loss: 7.6750 - accuracy: 0.4995
 9152/25000 [=========>....................] - ETA: 52s - loss: 7.6750 - accuracy: 0.4995
 9184/25000 [==========>...................] - ETA: 52s - loss: 7.6783 - accuracy: 0.4992
 9216/25000 [==========>...................] - ETA: 52s - loss: 7.6749 - accuracy: 0.4995
 9248/25000 [==========>...................] - ETA: 51s - loss: 7.6766 - accuracy: 0.4994
 9280/25000 [==========>...................] - ETA: 51s - loss: 7.6782 - accuracy: 0.4992
 9312/25000 [==========>...................] - ETA: 51s - loss: 7.6765 - accuracy: 0.4994
 9344/25000 [==========>...................] - ETA: 51s - loss: 7.6797 - accuracy: 0.4991
 9376/25000 [==========>...................] - ETA: 51s - loss: 7.6813 - accuracy: 0.4990
 9408/25000 [==========>...................] - ETA: 51s - loss: 7.6829 - accuracy: 0.4989
 9440/25000 [==========>...................] - ETA: 51s - loss: 7.6845 - accuracy: 0.4988
 9472/25000 [==========>...................] - ETA: 51s - loss: 7.6877 - accuracy: 0.4986
 9504/25000 [==========>...................] - ETA: 51s - loss: 7.6908 - accuracy: 0.4984
 9536/25000 [==========>...................] - ETA: 50s - loss: 7.6891 - accuracy: 0.4985
 9568/25000 [==========>...................] - ETA: 50s - loss: 7.6955 - accuracy: 0.4981
 9600/25000 [==========>...................] - ETA: 50s - loss: 7.6938 - accuracy: 0.4982
 9632/25000 [==========>...................] - ETA: 50s - loss: 7.6969 - accuracy: 0.4980
 9664/25000 [==========>...................] - ETA: 50s - loss: 7.6952 - accuracy: 0.4981
 9696/25000 [==========>...................] - ETA: 50s - loss: 7.6919 - accuracy: 0.4983
 9728/25000 [==========>...................] - ETA: 50s - loss: 7.6903 - accuracy: 0.4985
 9760/25000 [==========>...................] - ETA: 50s - loss: 7.6870 - accuracy: 0.4987
 9792/25000 [==========>...................] - ETA: 50s - loss: 7.6823 - accuracy: 0.4990
 9824/25000 [==========>...................] - ETA: 49s - loss: 7.6838 - accuracy: 0.4989
 9856/25000 [==========>...................] - ETA: 49s - loss: 7.6837 - accuracy: 0.4989
 9888/25000 [==========>...................] - ETA: 49s - loss: 7.6899 - accuracy: 0.4985
 9920/25000 [==========>...................] - ETA: 49s - loss: 7.6805 - accuracy: 0.4991
 9952/25000 [==========>...................] - ETA: 49s - loss: 7.6789 - accuracy: 0.4992
 9984/25000 [==========>...................] - ETA: 49s - loss: 7.6789 - accuracy: 0.4992
10016/25000 [===========>..................] - ETA: 49s - loss: 7.6758 - accuracy: 0.4994
10048/25000 [===========>..................] - ETA: 49s - loss: 7.6758 - accuracy: 0.4994
10080/25000 [===========>..................] - ETA: 49s - loss: 7.6742 - accuracy: 0.4995
10112/25000 [===========>..................] - ETA: 49s - loss: 7.6772 - accuracy: 0.4993
10144/25000 [===========>..................] - ETA: 48s - loss: 7.6696 - accuracy: 0.4998
10176/25000 [===========>..................] - ETA: 48s - loss: 7.6621 - accuracy: 0.5003
10208/25000 [===========>..................] - ETA: 48s - loss: 7.6636 - accuracy: 0.5002
10240/25000 [===========>..................] - ETA: 48s - loss: 7.6561 - accuracy: 0.5007
10272/25000 [===========>..................] - ETA: 48s - loss: 7.6487 - accuracy: 0.5012
10304/25000 [===========>..................] - ETA: 48s - loss: 7.6547 - accuracy: 0.5008
10336/25000 [===========>..................] - ETA: 48s - loss: 7.6548 - accuracy: 0.5008
10368/25000 [===========>..................] - ETA: 48s - loss: 7.6518 - accuracy: 0.5010
10400/25000 [===========>..................] - ETA: 47s - loss: 7.6460 - accuracy: 0.5013
10432/25000 [===========>..................] - ETA: 47s - loss: 7.6475 - accuracy: 0.5012
10464/25000 [===========>..................] - ETA: 47s - loss: 7.6534 - accuracy: 0.5009
10496/25000 [===========>..................] - ETA: 47s - loss: 7.6520 - accuracy: 0.5010
10528/25000 [===========>..................] - ETA: 47s - loss: 7.6506 - accuracy: 0.5010
10560/25000 [===========>..................] - ETA: 47s - loss: 7.6477 - accuracy: 0.5012
10592/25000 [===========>..................] - ETA: 47s - loss: 7.6536 - accuracy: 0.5008
10624/25000 [===========>..................] - ETA: 47s - loss: 7.6493 - accuracy: 0.5011
10656/25000 [===========>..................] - ETA: 47s - loss: 7.6422 - accuracy: 0.5016
10688/25000 [===========>..................] - ETA: 47s - loss: 7.6437 - accuracy: 0.5015
10720/25000 [===========>..................] - ETA: 46s - loss: 7.6423 - accuracy: 0.5016
10752/25000 [===========>..................] - ETA: 46s - loss: 7.6452 - accuracy: 0.5014
10784/25000 [===========>..................] - ETA: 46s - loss: 7.6453 - accuracy: 0.5014
10816/25000 [===========>..................] - ETA: 46s - loss: 7.6383 - accuracy: 0.5018
10848/25000 [============>.................] - ETA: 46s - loss: 7.6384 - accuracy: 0.5018
10880/25000 [============>.................] - ETA: 46s - loss: 7.6398 - accuracy: 0.5017
10912/25000 [============>.................] - ETA: 46s - loss: 7.6385 - accuracy: 0.5018
10944/25000 [============>.................] - ETA: 46s - loss: 7.6414 - accuracy: 0.5016
10976/25000 [============>.................] - ETA: 45s - loss: 7.6359 - accuracy: 0.5020
11008/25000 [============>.................] - ETA: 45s - loss: 7.6360 - accuracy: 0.5020
11040/25000 [============>.................] - ETA: 45s - loss: 7.6333 - accuracy: 0.5022
11072/25000 [============>.................] - ETA: 45s - loss: 7.6375 - accuracy: 0.5019
11104/25000 [============>.................] - ETA: 45s - loss: 7.6390 - accuracy: 0.5018
11136/25000 [============>.................] - ETA: 45s - loss: 7.6377 - accuracy: 0.5019
11168/25000 [============>.................] - ETA: 45s - loss: 7.6337 - accuracy: 0.5021
11200/25000 [============>.................] - ETA: 45s - loss: 7.6310 - accuracy: 0.5023
11232/25000 [============>.................] - ETA: 45s - loss: 7.6284 - accuracy: 0.5025
11264/25000 [============>.................] - ETA: 45s - loss: 7.6285 - accuracy: 0.5025
11296/25000 [============>.................] - ETA: 44s - loss: 7.6232 - accuracy: 0.5028
11328/25000 [============>.................] - ETA: 44s - loss: 7.6247 - accuracy: 0.5027
11360/25000 [============>.................] - ETA: 44s - loss: 7.6234 - accuracy: 0.5028
11392/25000 [============>.................] - ETA: 44s - loss: 7.6249 - accuracy: 0.5027
11424/25000 [============>.................] - ETA: 44s - loss: 7.6143 - accuracy: 0.5034
11456/25000 [============>.................] - ETA: 44s - loss: 7.6131 - accuracy: 0.5035
11488/25000 [============>.................] - ETA: 44s - loss: 7.6146 - accuracy: 0.5034
11520/25000 [============>.................] - ETA: 44s - loss: 7.6147 - accuracy: 0.5034
11552/25000 [============>.................] - ETA: 44s - loss: 7.6188 - accuracy: 0.5031
11584/25000 [============>.................] - ETA: 43s - loss: 7.6190 - accuracy: 0.5031
11616/25000 [============>.................] - ETA: 43s - loss: 7.6178 - accuracy: 0.5032
11648/25000 [============>.................] - ETA: 43s - loss: 7.6205 - accuracy: 0.5030
11680/25000 [=============>................] - ETA: 43s - loss: 7.6154 - accuracy: 0.5033
11712/25000 [=============>................] - ETA: 43s - loss: 7.6195 - accuracy: 0.5031
11744/25000 [=============>................] - ETA: 43s - loss: 7.6196 - accuracy: 0.5031
11776/25000 [=============>................] - ETA: 43s - loss: 7.6132 - accuracy: 0.5035
11808/25000 [=============>................] - ETA: 43s - loss: 7.6160 - accuracy: 0.5033
11840/25000 [=============>................] - ETA: 43s - loss: 7.6135 - accuracy: 0.5035
11872/25000 [=============>................] - ETA: 43s - loss: 7.6137 - accuracy: 0.5035
11904/25000 [=============>................] - ETA: 42s - loss: 7.6112 - accuracy: 0.5036
11936/25000 [=============>................] - ETA: 42s - loss: 7.6127 - accuracy: 0.5035
11968/25000 [=============>................] - ETA: 42s - loss: 7.6167 - accuracy: 0.5033
12000/25000 [=============>................] - ETA: 42s - loss: 7.6193 - accuracy: 0.5031
12032/25000 [=============>................] - ETA: 42s - loss: 7.6258 - accuracy: 0.5027
12064/25000 [=============>................] - ETA: 42s - loss: 7.6209 - accuracy: 0.5030
12096/25000 [=============>................] - ETA: 42s - loss: 7.6261 - accuracy: 0.5026
12128/25000 [=============>................] - ETA: 42s - loss: 7.6300 - accuracy: 0.5024
12160/25000 [=============>................] - ETA: 42s - loss: 7.6313 - accuracy: 0.5023
12192/25000 [=============>................] - ETA: 41s - loss: 7.6301 - accuracy: 0.5024
12224/25000 [=============>................] - ETA: 41s - loss: 7.6277 - accuracy: 0.5025
12256/25000 [=============>................] - ETA: 41s - loss: 7.6291 - accuracy: 0.5024
12288/25000 [=============>................] - ETA: 41s - loss: 7.6329 - accuracy: 0.5022
12320/25000 [=============>................] - ETA: 41s - loss: 7.6293 - accuracy: 0.5024
12352/25000 [=============>................] - ETA: 41s - loss: 7.6244 - accuracy: 0.5028
12384/25000 [=============>................] - ETA: 41s - loss: 7.6270 - accuracy: 0.5026
12416/25000 [=============>................] - ETA: 41s - loss: 7.6283 - accuracy: 0.5025
12448/25000 [=============>................] - ETA: 41s - loss: 7.6297 - accuracy: 0.5024
12480/25000 [=============>................] - ETA: 40s - loss: 7.6285 - accuracy: 0.5025
12512/25000 [==============>...............] - ETA: 40s - loss: 7.6311 - accuracy: 0.5023
12544/25000 [==============>...............] - ETA: 40s - loss: 7.6299 - accuracy: 0.5024
12576/25000 [==============>...............] - ETA: 40s - loss: 7.6313 - accuracy: 0.5023
12608/25000 [==============>...............] - ETA: 40s - loss: 7.6314 - accuracy: 0.5023
12640/25000 [==============>...............] - ETA: 40s - loss: 7.6339 - accuracy: 0.5021
12672/25000 [==============>...............] - ETA: 40s - loss: 7.6327 - accuracy: 0.5022
12704/25000 [==============>...............] - ETA: 40s - loss: 7.6280 - accuracy: 0.5025
12736/25000 [==============>...............] - ETA: 40s - loss: 7.6257 - accuracy: 0.5027
12768/25000 [==============>...............] - ETA: 39s - loss: 7.6270 - accuracy: 0.5026
12800/25000 [==============>...............] - ETA: 39s - loss: 7.6295 - accuracy: 0.5024
12832/25000 [==============>...............] - ETA: 39s - loss: 7.6320 - accuracy: 0.5023
12864/25000 [==============>...............] - ETA: 39s - loss: 7.6332 - accuracy: 0.5022
12896/25000 [==============>...............] - ETA: 39s - loss: 7.6321 - accuracy: 0.5022
12928/25000 [==============>...............] - ETA: 39s - loss: 7.6358 - accuracy: 0.5020
12960/25000 [==============>...............] - ETA: 39s - loss: 7.6370 - accuracy: 0.5019
12992/25000 [==============>...............] - ETA: 39s - loss: 7.6371 - accuracy: 0.5019
13024/25000 [==============>...............] - ETA: 39s - loss: 7.6337 - accuracy: 0.5021
13056/25000 [==============>...............] - ETA: 39s - loss: 7.6373 - accuracy: 0.5019
13088/25000 [==============>...............] - ETA: 38s - loss: 7.6408 - accuracy: 0.5017
13120/25000 [==============>...............] - ETA: 38s - loss: 7.6386 - accuracy: 0.5018
13152/25000 [==============>...............] - ETA: 38s - loss: 7.6363 - accuracy: 0.5020
13184/25000 [==============>...............] - ETA: 38s - loss: 7.6341 - accuracy: 0.5021
13216/25000 [==============>...............] - ETA: 38s - loss: 7.6365 - accuracy: 0.5020
13248/25000 [==============>...............] - ETA: 38s - loss: 7.6388 - accuracy: 0.5018
13280/25000 [==============>...............] - ETA: 38s - loss: 7.6389 - accuracy: 0.5018
13312/25000 [==============>...............] - ETA: 38s - loss: 7.6390 - accuracy: 0.5018
13344/25000 [===============>..............] - ETA: 38s - loss: 7.6344 - accuracy: 0.5021
13376/25000 [===============>..............] - ETA: 37s - loss: 7.6311 - accuracy: 0.5023
13408/25000 [===============>..............] - ETA: 37s - loss: 7.6357 - accuracy: 0.5020
13440/25000 [===============>..............] - ETA: 37s - loss: 7.6415 - accuracy: 0.5016
13472/25000 [===============>..............] - ETA: 37s - loss: 7.6404 - accuracy: 0.5017
13504/25000 [===============>..............] - ETA: 37s - loss: 7.6382 - accuracy: 0.5019
13536/25000 [===============>..............] - ETA: 37s - loss: 7.6349 - accuracy: 0.5021
13568/25000 [===============>..............] - ETA: 37s - loss: 7.6350 - accuracy: 0.5021
13600/25000 [===============>..............] - ETA: 37s - loss: 7.6373 - accuracy: 0.5019
13632/25000 [===============>..............] - ETA: 37s - loss: 7.6340 - accuracy: 0.5021
13664/25000 [===============>..............] - ETA: 36s - loss: 7.6341 - accuracy: 0.5021
13696/25000 [===============>..............] - ETA: 36s - loss: 7.6386 - accuracy: 0.5018
13728/25000 [===============>..............] - ETA: 36s - loss: 7.6353 - accuracy: 0.5020
13760/25000 [===============>..............] - ETA: 36s - loss: 7.6321 - accuracy: 0.5023
13792/25000 [===============>..............] - ETA: 36s - loss: 7.6322 - accuracy: 0.5022
13824/25000 [===============>..............] - ETA: 36s - loss: 7.6300 - accuracy: 0.5024
13856/25000 [===============>..............] - ETA: 36s - loss: 7.6334 - accuracy: 0.5022
13888/25000 [===============>..............] - ETA: 36s - loss: 7.6379 - accuracy: 0.5019
13920/25000 [===============>..............] - ETA: 36s - loss: 7.6380 - accuracy: 0.5019
13952/25000 [===============>..............] - ETA: 36s - loss: 7.6358 - accuracy: 0.5020
13984/25000 [===============>..............] - ETA: 35s - loss: 7.6337 - accuracy: 0.5021
14016/25000 [===============>..............] - ETA: 35s - loss: 7.6360 - accuracy: 0.5020
14048/25000 [===============>..............] - ETA: 35s - loss: 7.6361 - accuracy: 0.5020
14080/25000 [===============>..............] - ETA: 35s - loss: 7.6350 - accuracy: 0.5021
14112/25000 [===============>..............] - ETA: 35s - loss: 7.6329 - accuracy: 0.5022
14144/25000 [===============>..............] - ETA: 35s - loss: 7.6287 - accuracy: 0.5025
14176/25000 [================>.............] - ETA: 35s - loss: 7.6255 - accuracy: 0.5027
14208/25000 [================>.............] - ETA: 35s - loss: 7.6224 - accuracy: 0.5029
14240/25000 [================>.............] - ETA: 35s - loss: 7.6182 - accuracy: 0.5032
14272/25000 [================>.............] - ETA: 34s - loss: 7.6193 - accuracy: 0.5031
14304/25000 [================>.............] - ETA: 34s - loss: 7.6227 - accuracy: 0.5029
14336/25000 [================>.............] - ETA: 34s - loss: 7.6217 - accuracy: 0.5029
14368/25000 [================>.............] - ETA: 34s - loss: 7.6207 - accuracy: 0.5030
14400/25000 [================>.............] - ETA: 34s - loss: 7.6208 - accuracy: 0.5030
14432/25000 [================>.............] - ETA: 34s - loss: 7.6220 - accuracy: 0.5029
14464/25000 [================>.............] - ETA: 34s - loss: 7.6179 - accuracy: 0.5032
14496/25000 [================>.............] - ETA: 34s - loss: 7.6190 - accuracy: 0.5031
14528/25000 [================>.............] - ETA: 34s - loss: 7.6191 - accuracy: 0.5031
14560/25000 [================>.............] - ETA: 33s - loss: 7.6213 - accuracy: 0.5030
14592/25000 [================>.............] - ETA: 33s - loss: 7.6193 - accuracy: 0.5031
14624/25000 [================>.............] - ETA: 33s - loss: 7.6236 - accuracy: 0.5028
14656/25000 [================>.............] - ETA: 33s - loss: 7.6269 - accuracy: 0.5026
14688/25000 [================>.............] - ETA: 33s - loss: 7.6259 - accuracy: 0.5027
14720/25000 [================>.............] - ETA: 33s - loss: 7.6218 - accuracy: 0.5029
14752/25000 [================>.............] - ETA: 33s - loss: 7.6198 - accuracy: 0.5031
14784/25000 [================>.............] - ETA: 33s - loss: 7.6199 - accuracy: 0.5030
14816/25000 [================>.............] - ETA: 33s - loss: 7.6211 - accuracy: 0.5030
14848/25000 [================>.............] - ETA: 33s - loss: 7.6222 - accuracy: 0.5029
14880/25000 [================>.............] - ETA: 32s - loss: 7.6223 - accuracy: 0.5029
14912/25000 [================>.............] - ETA: 32s - loss: 7.6234 - accuracy: 0.5028
14944/25000 [================>.............] - ETA: 32s - loss: 7.6256 - accuracy: 0.5027
14976/25000 [================>.............] - ETA: 32s - loss: 7.6298 - accuracy: 0.5024
15008/25000 [=================>............] - ETA: 32s - loss: 7.6309 - accuracy: 0.5023
15040/25000 [=================>............] - ETA: 32s - loss: 7.6320 - accuracy: 0.5023
15072/25000 [=================>............] - ETA: 32s - loss: 7.6300 - accuracy: 0.5024
15104/25000 [=================>............] - ETA: 32s - loss: 7.6280 - accuracy: 0.5025
15136/25000 [=================>............] - ETA: 32s - loss: 7.6271 - accuracy: 0.5026
15168/25000 [=================>............] - ETA: 31s - loss: 7.6282 - accuracy: 0.5025
15200/25000 [=================>............] - ETA: 31s - loss: 7.6343 - accuracy: 0.5021
15232/25000 [=================>............] - ETA: 31s - loss: 7.6364 - accuracy: 0.5020
15264/25000 [=================>............] - ETA: 31s - loss: 7.6385 - accuracy: 0.5018
15296/25000 [=================>............] - ETA: 31s - loss: 7.6386 - accuracy: 0.5018
15328/25000 [=================>............] - ETA: 31s - loss: 7.6386 - accuracy: 0.5018
15360/25000 [=================>............] - ETA: 31s - loss: 7.6377 - accuracy: 0.5019
15392/25000 [=================>............] - ETA: 31s - loss: 7.6387 - accuracy: 0.5018
15424/25000 [=================>............] - ETA: 31s - loss: 7.6438 - accuracy: 0.5015
15456/25000 [=================>............] - ETA: 30s - loss: 7.6418 - accuracy: 0.5016
15488/25000 [=================>............] - ETA: 30s - loss: 7.6478 - accuracy: 0.5012
15520/25000 [=================>............] - ETA: 30s - loss: 7.6439 - accuracy: 0.5015
15552/25000 [=================>............] - ETA: 30s - loss: 7.6449 - accuracy: 0.5014
15584/25000 [=================>............] - ETA: 30s - loss: 7.6420 - accuracy: 0.5016
15616/25000 [=================>............] - ETA: 30s - loss: 7.6372 - accuracy: 0.5019
15648/25000 [=================>............] - ETA: 30s - loss: 7.6402 - accuracy: 0.5017
15680/25000 [=================>............] - ETA: 30s - loss: 7.6402 - accuracy: 0.5017
15712/25000 [=================>............] - ETA: 30s - loss: 7.6364 - accuracy: 0.5020
15744/25000 [=================>............] - ETA: 30s - loss: 7.6355 - accuracy: 0.5020
15776/25000 [=================>............] - ETA: 29s - loss: 7.6365 - accuracy: 0.5020
15808/25000 [=================>............] - ETA: 29s - loss: 7.6385 - accuracy: 0.5018
15840/25000 [==================>...........] - ETA: 29s - loss: 7.6366 - accuracy: 0.5020
15872/25000 [==================>...........] - ETA: 29s - loss: 7.6328 - accuracy: 0.5022
15904/25000 [==================>...........] - ETA: 29s - loss: 7.6358 - accuracy: 0.5020
15936/25000 [==================>...........] - ETA: 29s - loss: 7.6349 - accuracy: 0.5021
15968/25000 [==================>...........] - ETA: 29s - loss: 7.6340 - accuracy: 0.5021
16000/25000 [==================>...........] - ETA: 29s - loss: 7.6350 - accuracy: 0.5021
16032/25000 [==================>...........] - ETA: 29s - loss: 7.6351 - accuracy: 0.5021
16064/25000 [==================>...........] - ETA: 28s - loss: 7.6342 - accuracy: 0.5021
16096/25000 [==================>...........] - ETA: 28s - loss: 7.6333 - accuracy: 0.5022
16128/25000 [==================>...........] - ETA: 28s - loss: 7.6352 - accuracy: 0.5020
16160/25000 [==================>...........] - ETA: 28s - loss: 7.6334 - accuracy: 0.5022
16192/25000 [==================>...........] - ETA: 28s - loss: 7.6316 - accuracy: 0.5023
16224/25000 [==================>...........] - ETA: 28s - loss: 7.6307 - accuracy: 0.5023
16256/25000 [==================>...........] - ETA: 28s - loss: 7.6327 - accuracy: 0.5022
16288/25000 [==================>...........] - ETA: 28s - loss: 7.6356 - accuracy: 0.5020
16320/25000 [==================>...........] - ETA: 28s - loss: 7.6441 - accuracy: 0.5015
16352/25000 [==================>...........] - ETA: 28s - loss: 7.6451 - accuracy: 0.5014
16384/25000 [==================>...........] - ETA: 27s - loss: 7.6414 - accuracy: 0.5016
16416/25000 [==================>...........] - ETA: 27s - loss: 7.6442 - accuracy: 0.5015
16448/25000 [==================>...........] - ETA: 27s - loss: 7.6470 - accuracy: 0.5013
16480/25000 [==================>...........] - ETA: 27s - loss: 7.6480 - accuracy: 0.5012
16512/25000 [==================>...........] - ETA: 27s - loss: 7.6434 - accuracy: 0.5015
16544/25000 [==================>...........] - ETA: 27s - loss: 7.6481 - accuracy: 0.5012
16576/25000 [==================>...........] - ETA: 27s - loss: 7.6444 - accuracy: 0.5014
16608/25000 [==================>...........] - ETA: 27s - loss: 7.6408 - accuracy: 0.5017
16640/25000 [==================>...........] - ETA: 27s - loss: 7.6408 - accuracy: 0.5017
16672/25000 [===================>..........] - ETA: 26s - loss: 7.6409 - accuracy: 0.5017
16704/25000 [===================>..........] - ETA: 26s - loss: 7.6409 - accuracy: 0.5017
16736/25000 [===================>..........] - ETA: 26s - loss: 7.6419 - accuracy: 0.5016
16768/25000 [===================>..........] - ETA: 26s - loss: 7.6410 - accuracy: 0.5017
16800/25000 [===================>..........] - ETA: 26s - loss: 7.6392 - accuracy: 0.5018
16832/25000 [===================>..........] - ETA: 26s - loss: 7.6438 - accuracy: 0.5015
16864/25000 [===================>..........] - ETA: 26s - loss: 7.6430 - accuracy: 0.5015
16896/25000 [===================>..........] - ETA: 26s - loss: 7.6430 - accuracy: 0.5015
16928/25000 [===================>..........] - ETA: 26s - loss: 7.6467 - accuracy: 0.5013
16960/25000 [===================>..........] - ETA: 26s - loss: 7.6503 - accuracy: 0.5011
16992/25000 [===================>..........] - ETA: 25s - loss: 7.6531 - accuracy: 0.5009
17024/25000 [===================>..........] - ETA: 25s - loss: 7.6522 - accuracy: 0.5009
17056/25000 [===================>..........] - ETA: 25s - loss: 7.6558 - accuracy: 0.5007
17088/25000 [===================>..........] - ETA: 25s - loss: 7.6541 - accuracy: 0.5008
17120/25000 [===================>..........] - ETA: 25s - loss: 7.6532 - accuracy: 0.5009
17152/25000 [===================>..........] - ETA: 25s - loss: 7.6514 - accuracy: 0.5010
17184/25000 [===================>..........] - ETA: 25s - loss: 7.6532 - accuracy: 0.5009
17216/25000 [===================>..........] - ETA: 25s - loss: 7.6542 - accuracy: 0.5008
17248/25000 [===================>..........] - ETA: 25s - loss: 7.6533 - accuracy: 0.5009
17280/25000 [===================>..........] - ETA: 25s - loss: 7.6533 - accuracy: 0.5009
17312/25000 [===================>..........] - ETA: 24s - loss: 7.6560 - accuracy: 0.5007
17344/25000 [===================>..........] - ETA: 24s - loss: 7.6542 - accuracy: 0.5008
17376/25000 [===================>..........] - ETA: 24s - loss: 7.6525 - accuracy: 0.5009
17408/25000 [===================>..........] - ETA: 24s - loss: 7.6499 - accuracy: 0.5011
17440/25000 [===================>..........] - ETA: 24s - loss: 7.6473 - accuracy: 0.5013
17472/25000 [===================>..........] - ETA: 24s - loss: 7.6456 - accuracy: 0.5014
17504/25000 [====================>.........] - ETA: 24s - loss: 7.6456 - accuracy: 0.5014
17536/25000 [====================>.........] - ETA: 24s - loss: 7.6483 - accuracy: 0.5012
17568/25000 [====================>.........] - ETA: 24s - loss: 7.6474 - accuracy: 0.5013
17600/25000 [====================>.........] - ETA: 23s - loss: 7.6475 - accuracy: 0.5013
17632/25000 [====================>.........] - ETA: 23s - loss: 7.6449 - accuracy: 0.5014
17664/25000 [====================>.........] - ETA: 23s - loss: 7.6458 - accuracy: 0.5014
17696/25000 [====================>.........] - ETA: 23s - loss: 7.6432 - accuracy: 0.5015
17728/25000 [====================>.........] - ETA: 23s - loss: 7.6450 - accuracy: 0.5014
17760/25000 [====================>.........] - ETA: 23s - loss: 7.6502 - accuracy: 0.5011
17792/25000 [====================>.........] - ETA: 23s - loss: 7.6537 - accuracy: 0.5008
17824/25000 [====================>.........] - ETA: 23s - loss: 7.6589 - accuracy: 0.5005
17856/25000 [====================>.........] - ETA: 23s - loss: 7.6589 - accuracy: 0.5005
17888/25000 [====================>.........] - ETA: 23s - loss: 7.6572 - accuracy: 0.5006
17920/25000 [====================>.........] - ETA: 22s - loss: 7.6572 - accuracy: 0.5006
17952/25000 [====================>.........] - ETA: 22s - loss: 7.6572 - accuracy: 0.5006
17984/25000 [====================>.........] - ETA: 22s - loss: 7.6547 - accuracy: 0.5008
18016/25000 [====================>.........] - ETA: 22s - loss: 7.6522 - accuracy: 0.5009
18048/25000 [====================>.........] - ETA: 22s - loss: 7.6530 - accuracy: 0.5009
18080/25000 [====================>.........] - ETA: 22s - loss: 7.6573 - accuracy: 0.5006
18112/25000 [====================>.........] - ETA: 22s - loss: 7.6539 - accuracy: 0.5008
18144/25000 [====================>.........] - ETA: 22s - loss: 7.6506 - accuracy: 0.5010
18176/25000 [====================>.........] - ETA: 22s - loss: 7.6506 - accuracy: 0.5010
18208/25000 [====================>.........] - ETA: 21s - loss: 7.6548 - accuracy: 0.5008
18240/25000 [====================>.........] - ETA: 21s - loss: 7.6582 - accuracy: 0.5005
18272/25000 [====================>.........] - ETA: 21s - loss: 7.6624 - accuracy: 0.5003
18304/25000 [====================>.........] - ETA: 21s - loss: 7.6633 - accuracy: 0.5002
18336/25000 [=====================>........] - ETA: 21s - loss: 7.6658 - accuracy: 0.5001
18368/25000 [=====================>........] - ETA: 21s - loss: 7.6641 - accuracy: 0.5002
18400/25000 [=====================>........] - ETA: 21s - loss: 7.6608 - accuracy: 0.5004
18432/25000 [=====================>........] - ETA: 21s - loss: 7.6600 - accuracy: 0.5004
18464/25000 [=====================>........] - ETA: 21s - loss: 7.6625 - accuracy: 0.5003
18496/25000 [=====================>........] - ETA: 21s - loss: 7.6650 - accuracy: 0.5001
18528/25000 [=====================>........] - ETA: 20s - loss: 7.6666 - accuracy: 0.5000
18560/25000 [=====================>........] - ETA: 20s - loss: 7.6683 - accuracy: 0.4999
18592/25000 [=====================>........] - ETA: 20s - loss: 7.6699 - accuracy: 0.4998
18624/25000 [=====================>........] - ETA: 20s - loss: 7.6674 - accuracy: 0.4999
18656/25000 [=====================>........] - ETA: 20s - loss: 7.6674 - accuracy: 0.4999
18688/25000 [=====================>........] - ETA: 20s - loss: 7.6674 - accuracy: 0.4999
18720/25000 [=====================>........] - ETA: 20s - loss: 7.6691 - accuracy: 0.4998
18752/25000 [=====================>........] - ETA: 20s - loss: 7.6691 - accuracy: 0.4998
18784/25000 [=====================>........] - ETA: 20s - loss: 7.6666 - accuracy: 0.5000
18816/25000 [=====================>........] - ETA: 20s - loss: 7.6691 - accuracy: 0.4998
18848/25000 [=====================>........] - ETA: 19s - loss: 7.6674 - accuracy: 0.4999
18880/25000 [=====================>........] - ETA: 19s - loss: 7.6682 - accuracy: 0.4999
18912/25000 [=====================>........] - ETA: 19s - loss: 7.6691 - accuracy: 0.4998
18944/25000 [=====================>........] - ETA: 19s - loss: 7.6707 - accuracy: 0.4997
18976/25000 [=====================>........] - ETA: 19s - loss: 7.6715 - accuracy: 0.4997
19008/25000 [=====================>........] - ETA: 19s - loss: 7.6731 - accuracy: 0.4996
19040/25000 [=====================>........] - ETA: 19s - loss: 7.6690 - accuracy: 0.4998
19072/25000 [=====================>........] - ETA: 19s - loss: 7.6690 - accuracy: 0.4998
19104/25000 [=====================>........] - ETA: 19s - loss: 7.6682 - accuracy: 0.4999
19136/25000 [=====================>........] - ETA: 19s - loss: 7.6706 - accuracy: 0.4997
19168/25000 [======================>.......] - ETA: 18s - loss: 7.6698 - accuracy: 0.4998
19200/25000 [======================>.......] - ETA: 18s - loss: 7.6714 - accuracy: 0.4997
19232/25000 [======================>.......] - ETA: 18s - loss: 7.6738 - accuracy: 0.4995
19264/25000 [======================>.......] - ETA: 18s - loss: 7.6682 - accuracy: 0.4999
19296/25000 [======================>.......] - ETA: 18s - loss: 7.6666 - accuracy: 0.5000
19328/25000 [======================>.......] - ETA: 18s - loss: 7.6706 - accuracy: 0.4997
19360/25000 [======================>.......] - ETA: 18s - loss: 7.6745 - accuracy: 0.4995
19392/25000 [======================>.......] - ETA: 18s - loss: 7.6682 - accuracy: 0.4999
19424/25000 [======================>.......] - ETA: 18s - loss: 7.6690 - accuracy: 0.4998
19456/25000 [======================>.......] - ETA: 17s - loss: 7.6666 - accuracy: 0.5000
19488/25000 [======================>.......] - ETA: 17s - loss: 7.6690 - accuracy: 0.4998
19520/25000 [======================>.......] - ETA: 17s - loss: 7.6690 - accuracy: 0.4998
19552/25000 [======================>.......] - ETA: 17s - loss: 7.6682 - accuracy: 0.4999
19584/25000 [======================>.......] - ETA: 17s - loss: 7.6651 - accuracy: 0.5001
19616/25000 [======================>.......] - ETA: 17s - loss: 7.6627 - accuracy: 0.5003
19648/25000 [======================>.......] - ETA: 17s - loss: 7.6612 - accuracy: 0.5004
19680/25000 [======================>.......] - ETA: 17s - loss: 7.6604 - accuracy: 0.5004
19712/25000 [======================>.......] - ETA: 17s - loss: 7.6643 - accuracy: 0.5002
19744/25000 [======================>.......] - ETA: 17s - loss: 7.6658 - accuracy: 0.5001
19776/25000 [======================>.......] - ETA: 16s - loss: 7.6612 - accuracy: 0.5004
19808/25000 [======================>.......] - ETA: 16s - loss: 7.6612 - accuracy: 0.5004
19840/25000 [======================>.......] - ETA: 16s - loss: 7.6643 - accuracy: 0.5002
19872/25000 [======================>.......] - ETA: 16s - loss: 7.6643 - accuracy: 0.5002
19904/25000 [======================>.......] - ETA: 16s - loss: 7.6643 - accuracy: 0.5002
19936/25000 [======================>.......] - ETA: 16s - loss: 7.6643 - accuracy: 0.5002
19968/25000 [======================>.......] - ETA: 16s - loss: 7.6643 - accuracy: 0.5002
20000/25000 [=======================>......] - ETA: 16s - loss: 7.6682 - accuracy: 0.4999
20032/25000 [=======================>......] - ETA: 16s - loss: 7.6689 - accuracy: 0.4999
20064/25000 [=======================>......] - ETA: 16s - loss: 7.6704 - accuracy: 0.4998
20096/25000 [=======================>......] - ETA: 15s - loss: 7.6697 - accuracy: 0.4998
20128/25000 [=======================>......] - ETA: 15s - loss: 7.6750 - accuracy: 0.4995
20160/25000 [=======================>......] - ETA: 15s - loss: 7.6750 - accuracy: 0.4995
20192/25000 [=======================>......] - ETA: 15s - loss: 7.6727 - accuracy: 0.4996
20224/25000 [=======================>......] - ETA: 15s - loss: 7.6750 - accuracy: 0.4995
20256/25000 [=======================>......] - ETA: 15s - loss: 7.6757 - accuracy: 0.4994
20288/25000 [=======================>......] - ETA: 15s - loss: 7.6749 - accuracy: 0.4995
20320/25000 [=======================>......] - ETA: 15s - loss: 7.6719 - accuracy: 0.4997
20352/25000 [=======================>......] - ETA: 15s - loss: 7.6726 - accuracy: 0.4996
20384/25000 [=======================>......] - ETA: 14s - loss: 7.6726 - accuracy: 0.4996
20416/25000 [=======================>......] - ETA: 14s - loss: 7.6719 - accuracy: 0.4997
20448/25000 [=======================>......] - ETA: 14s - loss: 7.6696 - accuracy: 0.4998
20480/25000 [=======================>......] - ETA: 14s - loss: 7.6674 - accuracy: 0.5000
20512/25000 [=======================>......] - ETA: 14s - loss: 7.6651 - accuracy: 0.5001
20544/25000 [=======================>......] - ETA: 14s - loss: 7.6606 - accuracy: 0.5004
20576/25000 [=======================>......] - ETA: 14s - loss: 7.6636 - accuracy: 0.5002
20608/25000 [=======================>......] - ETA: 14s - loss: 7.6629 - accuracy: 0.5002
20640/25000 [=======================>......] - ETA: 14s - loss: 7.6614 - accuracy: 0.5003
20672/25000 [=======================>......] - ETA: 14s - loss: 7.6614 - accuracy: 0.5003
20704/25000 [=======================>......] - ETA: 13s - loss: 7.6644 - accuracy: 0.5001
20736/25000 [=======================>......] - ETA: 13s - loss: 7.6629 - accuracy: 0.5002
20768/25000 [=======================>......] - ETA: 13s - loss: 7.6615 - accuracy: 0.5003
20800/25000 [=======================>......] - ETA: 13s - loss: 7.6637 - accuracy: 0.5002
20832/25000 [=======================>......] - ETA: 13s - loss: 7.6629 - accuracy: 0.5002
20864/25000 [========================>.....] - ETA: 13s - loss: 7.6622 - accuracy: 0.5003
20896/25000 [========================>.....] - ETA: 13s - loss: 7.6622 - accuracy: 0.5003
20928/25000 [========================>.....] - ETA: 13s - loss: 7.6637 - accuracy: 0.5002
20960/25000 [========================>.....] - ETA: 13s - loss: 7.6659 - accuracy: 0.5000
20992/25000 [========================>.....] - ETA: 13s - loss: 7.6652 - accuracy: 0.5001
21024/25000 [========================>.....] - ETA: 12s - loss: 7.6652 - accuracy: 0.5001
21056/25000 [========================>.....] - ETA: 12s - loss: 7.6659 - accuracy: 0.5000
21088/25000 [========================>.....] - ETA: 12s - loss: 7.6673 - accuracy: 0.5000
21120/25000 [========================>.....] - ETA: 12s - loss: 7.6644 - accuracy: 0.5001
21152/25000 [========================>.....] - ETA: 12s - loss: 7.6637 - accuracy: 0.5002
21184/25000 [========================>.....] - ETA: 12s - loss: 7.6630 - accuracy: 0.5002
21216/25000 [========================>.....] - ETA: 12s - loss: 7.6616 - accuracy: 0.5003
21248/25000 [========================>.....] - ETA: 12s - loss: 7.6587 - accuracy: 0.5005
21280/25000 [========================>.....] - ETA: 12s - loss: 7.6580 - accuracy: 0.5006
21312/25000 [========================>.....] - ETA: 11s - loss: 7.6594 - accuracy: 0.5005
21344/25000 [========================>.....] - ETA: 11s - loss: 7.6594 - accuracy: 0.5005
21376/25000 [========================>.....] - ETA: 11s - loss: 7.6602 - accuracy: 0.5004
21408/25000 [========================>.....] - ETA: 11s - loss: 7.6616 - accuracy: 0.5003
21440/25000 [========================>.....] - ETA: 11s - loss: 7.6602 - accuracy: 0.5004
21472/25000 [========================>.....] - ETA: 11s - loss: 7.6588 - accuracy: 0.5005
21504/25000 [========================>.....] - ETA: 11s - loss: 7.6552 - accuracy: 0.5007
21536/25000 [========================>.....] - ETA: 11s - loss: 7.6574 - accuracy: 0.5006
21568/25000 [========================>.....] - ETA: 11s - loss: 7.6616 - accuracy: 0.5003
21600/25000 [========================>.....] - ETA: 11s - loss: 7.6609 - accuracy: 0.5004
21632/25000 [========================>.....] - ETA: 10s - loss: 7.6602 - accuracy: 0.5004
21664/25000 [========================>.....] - ETA: 10s - loss: 7.6602 - accuracy: 0.5004
21696/25000 [=========================>....] - ETA: 10s - loss: 7.6596 - accuracy: 0.5005
21728/25000 [=========================>....] - ETA: 10s - loss: 7.6596 - accuracy: 0.5005
21760/25000 [=========================>....] - ETA: 10s - loss: 7.6582 - accuracy: 0.5006
21792/25000 [=========================>....] - ETA: 10s - loss: 7.6561 - accuracy: 0.5007
21824/25000 [=========================>....] - ETA: 10s - loss: 7.6533 - accuracy: 0.5009
21856/25000 [=========================>....] - ETA: 10s - loss: 7.6547 - accuracy: 0.5008
21888/25000 [=========================>....] - ETA: 10s - loss: 7.6554 - accuracy: 0.5007
21920/25000 [=========================>....] - ETA: 9s - loss: 7.6547 - accuracy: 0.5008 
21952/25000 [=========================>....] - ETA: 9s - loss: 7.6554 - accuracy: 0.5007
21984/25000 [=========================>....] - ETA: 9s - loss: 7.6569 - accuracy: 0.5006
22016/25000 [=========================>....] - ETA: 9s - loss: 7.6576 - accuracy: 0.5006
22048/25000 [=========================>....] - ETA: 9s - loss: 7.6562 - accuracy: 0.5007
22080/25000 [=========================>....] - ETA: 9s - loss: 7.6527 - accuracy: 0.5009
22112/25000 [=========================>....] - ETA: 9s - loss: 7.6534 - accuracy: 0.5009
22144/25000 [=========================>....] - ETA: 9s - loss: 7.6576 - accuracy: 0.5006
22176/25000 [=========================>....] - ETA: 9s - loss: 7.6569 - accuracy: 0.5006
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6528 - accuracy: 0.5009
22240/25000 [=========================>....] - ETA: 8s - loss: 7.6487 - accuracy: 0.5012
22272/25000 [=========================>....] - ETA: 8s - loss: 7.6460 - accuracy: 0.5013
22304/25000 [=========================>....] - ETA: 8s - loss: 7.6474 - accuracy: 0.5013
22336/25000 [=========================>....] - ETA: 8s - loss: 7.6481 - accuracy: 0.5012
22368/25000 [=========================>....] - ETA: 8s - loss: 7.6515 - accuracy: 0.5010
22400/25000 [=========================>....] - ETA: 8s - loss: 7.6516 - accuracy: 0.5010
22432/25000 [=========================>....] - ETA: 8s - loss: 7.6536 - accuracy: 0.5008
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6557 - accuracy: 0.5007
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6557 - accuracy: 0.5007
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6537 - accuracy: 0.5008
22560/25000 [==========================>...] - ETA: 7s - loss: 7.6551 - accuracy: 0.5008
22592/25000 [==========================>...] - ETA: 7s - loss: 7.6578 - accuracy: 0.5006
22624/25000 [==========================>...] - ETA: 7s - loss: 7.6578 - accuracy: 0.5006
22656/25000 [==========================>...] - ETA: 7s - loss: 7.6632 - accuracy: 0.5002
22688/25000 [==========================>...] - ETA: 7s - loss: 7.6612 - accuracy: 0.5004
22720/25000 [==========================>...] - ETA: 7s - loss: 7.6612 - accuracy: 0.5004
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6612 - accuracy: 0.5004
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6606 - accuracy: 0.5004
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6606 - accuracy: 0.5004
22848/25000 [==========================>...] - ETA: 6s - loss: 7.6613 - accuracy: 0.5004
22880/25000 [==========================>...] - ETA: 6s - loss: 7.6653 - accuracy: 0.5001
22912/25000 [==========================>...] - ETA: 6s - loss: 7.6666 - accuracy: 0.5000
22944/25000 [==========================>...] - ETA: 6s - loss: 7.6660 - accuracy: 0.5000
22976/25000 [==========================>...] - ETA: 6s - loss: 7.6673 - accuracy: 0.5000
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6660 - accuracy: 0.5000
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6646 - accuracy: 0.5001
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6653 - accuracy: 0.5001
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6653 - accuracy: 0.5001
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6653 - accuracy: 0.5001
23168/25000 [==========================>...] - ETA: 5s - loss: 7.6653 - accuracy: 0.5001
23200/25000 [==========================>...] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
23232/25000 [==========================>...] - ETA: 5s - loss: 7.6686 - accuracy: 0.4999
23264/25000 [==========================>...] - ETA: 5s - loss: 7.6679 - accuracy: 0.4999
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6679 - accuracy: 0.4999
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6679 - accuracy: 0.4999
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6692 - accuracy: 0.4998
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6718 - accuracy: 0.4997
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6660 - accuracy: 0.5000
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6686 - accuracy: 0.4999
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6685 - accuracy: 0.4999
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6660 - accuracy: 0.5000
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6698 - accuracy: 0.4998
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6647 - accuracy: 0.5001
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6647 - accuracy: 0.5001
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24192/25000 [============================>.] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
24224/25000 [============================>.] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24256/25000 [============================>.] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24288/25000 [============================>.] - ETA: 2s - loss: 7.6691 - accuracy: 0.4998
24320/25000 [============================>.] - ETA: 2s - loss: 7.6710 - accuracy: 0.4997
24352/25000 [============================>.] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24384/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24416/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24448/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24480/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24512/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24544/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24576/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24608/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24640/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
24672/25000 [============================>.] - ETA: 1s - loss: 7.6747 - accuracy: 0.4995
24704/25000 [============================>.] - ETA: 0s - loss: 7.6728 - accuracy: 0.4996
24736/25000 [============================>.] - ETA: 0s - loss: 7.6734 - accuracy: 0.4996
24768/25000 [============================>.] - ETA: 0s - loss: 7.6747 - accuracy: 0.4995
24800/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24832/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24864/25000 [============================>.] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24896/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24928/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24960/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 98s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
