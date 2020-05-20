
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '69b309ad857428cc5a734b8afd99842edf9b2a42', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/69b309ad857428cc5a734b8afd99842edf9b2a42

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42

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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:55<01:23, 27.68s/it] 40%|████      | 2/5 [00:55<01:23, 27.68s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.2874368760592078, 'embedding_size_factor': 0.6646475282137055, 'layers.choice': 0, 'learning_rate': 0.0001940515349155865, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 0.003896469061912527} and reward: 0.332
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd2e]\xa3\x95\xadOX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5D\xca\xe4\xa1q{X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?)oI\xfe\x0e\x10CX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?o\xeb|\xe6\x1a\xf3\xd8u.' and reward: 0.332
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd2e]\xa3\x95\xadOX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5D\xca\xe4\xa1q{X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?)oI\xfe\x0e\x10CX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?o\xeb|\xe6\x1a\xf3\xd8u.' and reward: 0.332
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 113.60384941101074
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the 4.67s of remaining time.
Ensemble size: 49
Ensemble weights: 
[0.65306122 0.34693878]
	0.39	 = Validation accuracy score
	0.94s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 116.31s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f251d301940> 

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
 [-0.01539823  0.00233343  0.10974521  0.00616526  0.0062887   0.03538546]
 [ 0.10351682 -0.08081392  0.32125717  0.01636785  0.05573122 -0.02437549]
 [ 0.15203612  0.34476116 -0.09603886  0.12523054  0.05913395 -0.10845021]
 [ 0.04499478  0.26776621 -0.09971219 -0.21058214 -0.21552148 -0.14600426]
 [ 0.65249914 -0.30902439  0.24062948  0.50135469  0.20664349  0.98861873]
 [ 0.43240047 -0.23101339  0.24341114 -0.00843253  0.22612363  0.45466471]
 [ 0.23849294  0.14145979 -0.12649629  0.03569458 -0.05088076 -0.91511154]
 [-0.02954077  0.4424212   0.12580571  0.66342074  0.34449449 -0.07532088]
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
{'loss': 0.5171691179275513, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-20 00:23:56.458963: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.5333339422941208, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-20 00:23:57.642891: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 4603904/17464789 [======>.......................] - ETA: 0s
11419648/17464789 [==================>...........] - ETA: 0s
15622144/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-20 00:24:09.428552: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-20 00:24:09.433701: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-20 00:24:09.433875: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f73cb6e7b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-20 00:24:09.433893: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:41 - loss: 5.7500 - accuracy: 0.6250
   64/25000 [..............................] - ETA: 3:03 - loss: 5.9895 - accuracy: 0.6094
   96/25000 [..............................] - ETA: 2:32 - loss: 6.5486 - accuracy: 0.5729
  128/25000 [..............................] - ETA: 2:16 - loss: 6.7083 - accuracy: 0.5625
  160/25000 [..............................] - ETA: 2:05 - loss: 7.1875 - accuracy: 0.5312
  192/25000 [..............................] - ETA: 1:59 - loss: 7.5069 - accuracy: 0.5104
  224/25000 [..............................] - ETA: 1:56 - loss: 7.2559 - accuracy: 0.5268
  256/25000 [..............................] - ETA: 1:52 - loss: 7.4270 - accuracy: 0.5156
  288/25000 [..............................] - ETA: 1:49 - loss: 7.4537 - accuracy: 0.5139
  320/25000 [..............................] - ETA: 1:46 - loss: 7.4750 - accuracy: 0.5125
  352/25000 [..............................] - ETA: 1:44 - loss: 7.3181 - accuracy: 0.5227
  384/25000 [..............................] - ETA: 1:43 - loss: 7.3871 - accuracy: 0.5182
  416/25000 [..............................] - ETA: 1:41 - loss: 7.4086 - accuracy: 0.5168
  448/25000 [..............................] - ETA: 1:40 - loss: 7.4613 - accuracy: 0.5134
  480/25000 [..............................] - ETA: 1:39 - loss: 7.6666 - accuracy: 0.5000
  512/25000 [..............................] - ETA: 1:38 - loss: 7.6966 - accuracy: 0.4980
  544/25000 [..............................] - ETA: 1:37 - loss: 7.7230 - accuracy: 0.4963
  576/25000 [..............................] - ETA: 1:37 - loss: 7.7731 - accuracy: 0.4931
  608/25000 [..............................] - ETA: 1:36 - loss: 7.6414 - accuracy: 0.5016
  640/25000 [..............................] - ETA: 1:35 - loss: 7.6906 - accuracy: 0.4984
  672/25000 [..............................] - ETA: 1:35 - loss: 7.5982 - accuracy: 0.5045
  704/25000 [..............................] - ETA: 1:34 - loss: 7.5577 - accuracy: 0.5071
  736/25000 [..............................] - ETA: 1:33 - loss: 7.5000 - accuracy: 0.5109
  768/25000 [..............................] - ETA: 1:33 - loss: 7.3871 - accuracy: 0.5182
  800/25000 [..............................] - ETA: 1:32 - loss: 7.3025 - accuracy: 0.5238
  832/25000 [..............................] - ETA: 1:32 - loss: 7.2980 - accuracy: 0.5240
  864/25000 [>.............................] - ETA: 1:31 - loss: 7.3649 - accuracy: 0.5197
  896/25000 [>.............................] - ETA: 1:31 - loss: 7.4099 - accuracy: 0.5167
  928/25000 [>.............................] - ETA: 1:31 - loss: 7.2866 - accuracy: 0.5248
  960/25000 [>.............................] - ETA: 1:31 - loss: 7.2993 - accuracy: 0.5240
  992/25000 [>.............................] - ETA: 1:31 - loss: 7.3575 - accuracy: 0.5202
 1024/25000 [>.............................] - ETA: 1:30 - loss: 7.3522 - accuracy: 0.5205
 1056/25000 [>.............................] - ETA: 1:30 - loss: 7.3762 - accuracy: 0.5189
 1088/25000 [>.............................] - ETA: 1:30 - loss: 7.3848 - accuracy: 0.5184
 1120/25000 [>.............................] - ETA: 1:30 - loss: 7.4202 - accuracy: 0.5161
 1152/25000 [>.............................] - ETA: 1:29 - loss: 7.3871 - accuracy: 0.5182
 1184/25000 [>.............................] - ETA: 1:29 - loss: 7.4206 - accuracy: 0.5160
 1216/25000 [>.............................] - ETA: 1:29 - loss: 7.4144 - accuracy: 0.5164
 1248/25000 [>.............................] - ETA: 1:29 - loss: 7.4086 - accuracy: 0.5168
 1280/25000 [>.............................] - ETA: 1:29 - loss: 7.4270 - accuracy: 0.5156
 1312/25000 [>.............................] - ETA: 1:28 - loss: 7.3978 - accuracy: 0.5175
 1344/25000 [>.............................] - ETA: 1:28 - loss: 7.4042 - accuracy: 0.5171
 1376/25000 [>.............................] - ETA: 1:28 - loss: 7.4103 - accuracy: 0.5167
 1408/25000 [>.............................] - ETA: 1:28 - loss: 7.3508 - accuracy: 0.5206
 1440/25000 [>.............................] - ETA: 1:28 - loss: 7.4004 - accuracy: 0.5174
 1472/25000 [>.............................] - ETA: 1:27 - loss: 7.4166 - accuracy: 0.5163
 1504/25000 [>.............................] - ETA: 1:27 - loss: 7.3812 - accuracy: 0.5186
 1536/25000 [>.............................] - ETA: 1:27 - loss: 7.3971 - accuracy: 0.5176
 1568/25000 [>.............................] - ETA: 1:27 - loss: 7.4710 - accuracy: 0.5128
 1600/25000 [>.............................] - ETA: 1:27 - loss: 7.4941 - accuracy: 0.5113
 1632/25000 [>.............................] - ETA: 1:26 - loss: 7.4975 - accuracy: 0.5110
 1664/25000 [>.............................] - ETA: 1:26 - loss: 7.5284 - accuracy: 0.5090
 1696/25000 [=>............................] - ETA: 1:26 - loss: 7.5581 - accuracy: 0.5071
 1728/25000 [=>............................] - ETA: 1:26 - loss: 7.5868 - accuracy: 0.5052
 1760/25000 [=>............................] - ETA: 1:26 - loss: 7.6056 - accuracy: 0.5040
 1792/25000 [=>............................] - ETA: 1:26 - loss: 7.6153 - accuracy: 0.5033
 1824/25000 [=>............................] - ETA: 1:25 - loss: 7.5994 - accuracy: 0.5044
 1856/25000 [=>............................] - ETA: 1:25 - loss: 7.6088 - accuracy: 0.5038
 1888/25000 [=>............................] - ETA: 1:25 - loss: 7.6260 - accuracy: 0.5026
 1920/25000 [=>............................] - ETA: 1:25 - loss: 7.6187 - accuracy: 0.5031
 1952/25000 [=>............................] - ETA: 1:25 - loss: 7.6038 - accuracy: 0.5041
 1984/25000 [=>............................] - ETA: 1:25 - loss: 7.6048 - accuracy: 0.5040
 2016/25000 [=>............................] - ETA: 1:25 - loss: 7.6134 - accuracy: 0.5035
 2048/25000 [=>............................] - ETA: 1:25 - loss: 7.6292 - accuracy: 0.5024
 2080/25000 [=>............................] - ETA: 1:24 - loss: 7.6076 - accuracy: 0.5038
 2112/25000 [=>............................] - ETA: 1:24 - loss: 7.6013 - accuracy: 0.5043
 2144/25000 [=>............................] - ETA: 1:24 - loss: 7.6023 - accuracy: 0.5042
 2176/25000 [=>............................] - ETA: 1:24 - loss: 7.6173 - accuracy: 0.5032
 2208/25000 [=>............................] - ETA: 1:24 - loss: 7.6319 - accuracy: 0.5023
 2240/25000 [=>............................] - ETA: 1:24 - loss: 7.6461 - accuracy: 0.5013
 2272/25000 [=>............................] - ETA: 1:23 - loss: 7.6059 - accuracy: 0.5040
 2304/25000 [=>............................] - ETA: 1:23 - loss: 7.6400 - accuracy: 0.5017
 2336/25000 [=>............................] - ETA: 1:23 - loss: 7.6732 - accuracy: 0.4996
 2368/25000 [=>............................] - ETA: 1:23 - loss: 7.6796 - accuracy: 0.4992
 2400/25000 [=>............................] - ETA: 1:23 - loss: 7.6730 - accuracy: 0.4996
 2432/25000 [=>............................] - ETA: 1:23 - loss: 7.6981 - accuracy: 0.4979
 2464/25000 [=>............................] - ETA: 1:22 - loss: 7.6791 - accuracy: 0.4992
 2496/25000 [=>............................] - ETA: 1:22 - loss: 7.6850 - accuracy: 0.4988
 2528/25000 [==>...........................] - ETA: 1:22 - loss: 7.7091 - accuracy: 0.4972
 2560/25000 [==>...........................] - ETA: 1:22 - loss: 7.7085 - accuracy: 0.4973
 2592/25000 [==>...........................] - ETA: 1:22 - loss: 7.7199 - accuracy: 0.4965
 2624/25000 [==>...........................] - ETA: 1:22 - loss: 7.7251 - accuracy: 0.4962
 2656/25000 [==>...........................] - ETA: 1:21 - loss: 7.7070 - accuracy: 0.4974
 2688/25000 [==>...........................] - ETA: 1:21 - loss: 7.7008 - accuracy: 0.4978
 2720/25000 [==>...........................] - ETA: 1:21 - loss: 7.6892 - accuracy: 0.4985
 2752/25000 [==>...........................] - ETA: 1:21 - loss: 7.6833 - accuracy: 0.4989
 2784/25000 [==>...........................] - ETA: 1:21 - loss: 7.6886 - accuracy: 0.4986
 2816/25000 [==>...........................] - ETA: 1:21 - loss: 7.7156 - accuracy: 0.4968
 2848/25000 [==>...........................] - ETA: 1:21 - loss: 7.7043 - accuracy: 0.4975
 2880/25000 [==>...........................] - ETA: 1:20 - loss: 7.6932 - accuracy: 0.4983
 2912/25000 [==>...........................] - ETA: 1:20 - loss: 7.6614 - accuracy: 0.5003
 2944/25000 [==>...........................] - ETA: 1:20 - loss: 7.6458 - accuracy: 0.5014
 2976/25000 [==>...........................] - ETA: 1:20 - loss: 7.6306 - accuracy: 0.5024
 3008/25000 [==>...........................] - ETA: 1:20 - loss: 7.6258 - accuracy: 0.5027
 3040/25000 [==>...........................] - ETA: 1:20 - loss: 7.6263 - accuracy: 0.5026
 3072/25000 [==>...........................] - ETA: 1:19 - loss: 7.6267 - accuracy: 0.5026
 3104/25000 [==>...........................] - ETA: 1:19 - loss: 7.5876 - accuracy: 0.5052
 3136/25000 [==>...........................] - ETA: 1:19 - loss: 7.5835 - accuracy: 0.5054
 3168/25000 [==>...........................] - ETA: 1:19 - loss: 7.5892 - accuracy: 0.5051
 3200/25000 [==>...........................] - ETA: 1:19 - loss: 7.6091 - accuracy: 0.5038
 3232/25000 [==>...........................] - ETA: 1:19 - loss: 7.6192 - accuracy: 0.5031
 3264/25000 [==>...........................] - ETA: 1:19 - loss: 7.6102 - accuracy: 0.5037
 3296/25000 [==>...........................] - ETA: 1:19 - loss: 7.6061 - accuracy: 0.5039
 3328/25000 [==>...........................] - ETA: 1:18 - loss: 7.5929 - accuracy: 0.5048
 3360/25000 [===>..........................] - ETA: 1:18 - loss: 7.5890 - accuracy: 0.5051
 3392/25000 [===>..........................] - ETA: 1:18 - loss: 7.5762 - accuracy: 0.5059
 3424/25000 [===>..........................] - ETA: 1:18 - loss: 7.5681 - accuracy: 0.5064
 3456/25000 [===>..........................] - ETA: 1:18 - loss: 7.5601 - accuracy: 0.5069
 3488/25000 [===>..........................] - ETA: 1:18 - loss: 7.5655 - accuracy: 0.5066
 3520/25000 [===>..........................] - ETA: 1:18 - loss: 7.5795 - accuracy: 0.5057
 3552/25000 [===>..........................] - ETA: 1:18 - loss: 7.5457 - accuracy: 0.5079
 3584/25000 [===>..........................] - ETA: 1:18 - loss: 7.5340 - accuracy: 0.5086
 3616/25000 [===>..........................] - ETA: 1:17 - loss: 7.5394 - accuracy: 0.5083
 3648/25000 [===>..........................] - ETA: 1:17 - loss: 7.5405 - accuracy: 0.5082
 3680/25000 [===>..........................] - ETA: 1:17 - loss: 7.5500 - accuracy: 0.5076
 3712/25000 [===>..........................] - ETA: 1:17 - loss: 7.5551 - accuracy: 0.5073
 3744/25000 [===>..........................] - ETA: 1:17 - loss: 7.5519 - accuracy: 0.5075
 3776/25000 [===>..........................] - ETA: 1:17 - loss: 7.5610 - accuracy: 0.5069
 3808/25000 [===>..........................] - ETA: 1:17 - loss: 7.5660 - accuracy: 0.5066
 3840/25000 [===>..........................] - ETA: 1:17 - loss: 7.5588 - accuracy: 0.5070
 3872/25000 [===>..........................] - ETA: 1:17 - loss: 7.5518 - accuracy: 0.5075
 3904/25000 [===>..........................] - ETA: 1:17 - loss: 7.5527 - accuracy: 0.5074
 3936/25000 [===>..........................] - ETA: 1:16 - loss: 7.5342 - accuracy: 0.5086
 3968/25000 [===>..........................] - ETA: 1:16 - loss: 7.5468 - accuracy: 0.5078
 4000/25000 [===>..........................] - ETA: 1:16 - loss: 7.5286 - accuracy: 0.5090
 4032/25000 [===>..........................] - ETA: 1:16 - loss: 7.5297 - accuracy: 0.5089
 4064/25000 [===>..........................] - ETA: 1:16 - loss: 7.5421 - accuracy: 0.5081
 4096/25000 [===>..........................] - ETA: 1:16 - loss: 7.5393 - accuracy: 0.5083
 4128/25000 [===>..........................] - ETA: 1:16 - loss: 7.5440 - accuracy: 0.5080
 4160/25000 [===>..........................] - ETA: 1:16 - loss: 7.5597 - accuracy: 0.5070
 4192/25000 [====>.........................] - ETA: 1:16 - loss: 7.5679 - accuracy: 0.5064
 4224/25000 [====>.........................] - ETA: 1:16 - loss: 7.5686 - accuracy: 0.5064
 4256/25000 [====>.........................] - ETA: 1:15 - loss: 7.5729 - accuracy: 0.5061
 4288/25000 [====>.........................] - ETA: 1:15 - loss: 7.5629 - accuracy: 0.5068
 4320/25000 [====>.........................] - ETA: 1:15 - loss: 7.5672 - accuracy: 0.5065
 4352/25000 [====>.........................] - ETA: 1:15 - loss: 7.5750 - accuracy: 0.5060
 4384/25000 [====>.........................] - ETA: 1:15 - loss: 7.5862 - accuracy: 0.5052
 4416/25000 [====>.........................] - ETA: 1:15 - loss: 7.5798 - accuracy: 0.5057
 4448/25000 [====>.........................] - ETA: 1:15 - loss: 7.5804 - accuracy: 0.5056
 4480/25000 [====>.........................] - ETA: 1:15 - loss: 7.5776 - accuracy: 0.5058
 4512/25000 [====>.........................] - ETA: 1:15 - loss: 7.5919 - accuracy: 0.5049
 4544/25000 [====>.........................] - ETA: 1:14 - loss: 7.5823 - accuracy: 0.5055
 4576/25000 [====>.........................] - ETA: 1:14 - loss: 7.5795 - accuracy: 0.5057
 4608/25000 [====>.........................] - ETA: 1:14 - loss: 7.5801 - accuracy: 0.5056
 4640/25000 [====>.........................] - ETA: 1:14 - loss: 7.5708 - accuracy: 0.5063
 4672/25000 [====>.........................] - ETA: 1:14 - loss: 7.5714 - accuracy: 0.5062
 4704/25000 [====>.........................] - ETA: 1:14 - loss: 7.5591 - accuracy: 0.5070
 4736/25000 [====>.........................] - ETA: 1:14 - loss: 7.5630 - accuracy: 0.5068
 4768/25000 [====>.........................] - ETA: 1:14 - loss: 7.5637 - accuracy: 0.5067
 4800/25000 [====>.........................] - ETA: 1:14 - loss: 7.5644 - accuracy: 0.5067
 4832/25000 [====>.........................] - ETA: 1:13 - loss: 7.5682 - accuracy: 0.5064
 4864/25000 [====>.........................] - ETA: 1:13 - loss: 7.5657 - accuracy: 0.5066
 4896/25000 [====>.........................] - ETA: 1:13 - loss: 7.5695 - accuracy: 0.5063
 4928/25000 [====>.........................] - ETA: 1:13 - loss: 7.5795 - accuracy: 0.5057
 4960/25000 [====>.........................] - ETA: 1:13 - loss: 7.5893 - accuracy: 0.5050
 4992/25000 [====>.........................] - ETA: 1:13 - loss: 7.5929 - accuracy: 0.5048
 5024/25000 [=====>........................] - ETA: 1:13 - loss: 7.5934 - accuracy: 0.5048
 5056/25000 [=====>........................] - ETA: 1:13 - loss: 7.5969 - accuracy: 0.5045
 5088/25000 [=====>........................] - ETA: 1:13 - loss: 7.5883 - accuracy: 0.5051
 5120/25000 [=====>........................] - ETA: 1:12 - loss: 7.5828 - accuracy: 0.5055
 5152/25000 [=====>........................] - ETA: 1:12 - loss: 7.5892 - accuracy: 0.5050
 5184/25000 [=====>........................] - ETA: 1:12 - loss: 7.5927 - accuracy: 0.5048
 5216/25000 [=====>........................] - ETA: 1:12 - loss: 7.5931 - accuracy: 0.5048
 5248/25000 [=====>........................] - ETA: 1:12 - loss: 7.5877 - accuracy: 0.5051
 5280/25000 [=====>........................] - ETA: 1:12 - loss: 7.5882 - accuracy: 0.5051
 5312/25000 [=====>........................] - ETA: 1:12 - loss: 7.5916 - accuracy: 0.5049
 5344/25000 [=====>........................] - ETA: 1:12 - loss: 7.6035 - accuracy: 0.5041
 5376/25000 [=====>........................] - ETA: 1:12 - loss: 7.5925 - accuracy: 0.5048
 5408/25000 [=====>........................] - ETA: 1:11 - loss: 7.6014 - accuracy: 0.5043
 5440/25000 [=====>........................] - ETA: 1:11 - loss: 7.5962 - accuracy: 0.5046
 5472/25000 [=====>........................] - ETA: 1:11 - loss: 7.5994 - accuracy: 0.5044
 5504/25000 [=====>........................] - ETA: 1:11 - loss: 7.6053 - accuracy: 0.5040
 5536/25000 [=====>........................] - ETA: 1:11 - loss: 7.6085 - accuracy: 0.5038
 5568/25000 [=====>........................] - ETA: 1:11 - loss: 7.5978 - accuracy: 0.5045
 5600/25000 [=====>........................] - ETA: 1:11 - loss: 7.6009 - accuracy: 0.5043
 5632/25000 [=====>........................] - ETA: 1:11 - loss: 7.5986 - accuracy: 0.5044
 5664/25000 [=====>........................] - ETA: 1:11 - loss: 7.5962 - accuracy: 0.5046
 5696/25000 [=====>........................] - ETA: 1:10 - loss: 7.5886 - accuracy: 0.5051
 5728/25000 [=====>........................] - ETA: 1:10 - loss: 7.5836 - accuracy: 0.5054
 5760/25000 [=====>........................] - ETA: 1:10 - loss: 7.5894 - accuracy: 0.5050
 5792/25000 [=====>........................] - ETA: 1:10 - loss: 7.5819 - accuracy: 0.5055
 5824/25000 [=====>........................] - ETA: 1:10 - loss: 7.5850 - accuracy: 0.5053
 5856/25000 [======>.......................] - ETA: 1:10 - loss: 7.5750 - accuracy: 0.5060
 5888/25000 [======>.......................] - ETA: 1:10 - loss: 7.5885 - accuracy: 0.5051
 5920/25000 [======>.......................] - ETA: 1:10 - loss: 7.5786 - accuracy: 0.5057
 5952/25000 [======>.......................] - ETA: 1:10 - loss: 7.5893 - accuracy: 0.5050
 5984/25000 [======>.......................] - ETA: 1:09 - loss: 7.5897 - accuracy: 0.5050
 6016/25000 [======>.......................] - ETA: 1:09 - loss: 7.5876 - accuracy: 0.5052
 6048/25000 [======>.......................] - ETA: 1:09 - loss: 7.5855 - accuracy: 0.5053
 6080/25000 [======>.......................] - ETA: 1:09 - loss: 7.5910 - accuracy: 0.5049
 6112/25000 [======>.......................] - ETA: 1:09 - loss: 7.5989 - accuracy: 0.5044
 6144/25000 [======>.......................] - ETA: 1:09 - loss: 7.6067 - accuracy: 0.5039
 6176/25000 [======>.......................] - ETA: 1:09 - loss: 7.6046 - accuracy: 0.5040
 6208/25000 [======>.......................] - ETA: 1:09 - loss: 7.6049 - accuracy: 0.5040
 6240/25000 [======>.......................] - ETA: 1:08 - loss: 7.6027 - accuracy: 0.5042
 6272/25000 [======>.......................] - ETA: 1:08 - loss: 7.6031 - accuracy: 0.5041
 6304/25000 [======>.......................] - ETA: 1:08 - loss: 7.6034 - accuracy: 0.5041
 6336/25000 [======>.......................] - ETA: 1:08 - loss: 7.5989 - accuracy: 0.5044
 6368/25000 [======>.......................] - ETA: 1:08 - loss: 7.5992 - accuracy: 0.5044
 6400/25000 [======>.......................] - ETA: 1:08 - loss: 7.5947 - accuracy: 0.5047
 6432/25000 [======>.......................] - ETA: 1:08 - loss: 7.5999 - accuracy: 0.5044
 6464/25000 [======>.......................] - ETA: 1:08 - loss: 7.6049 - accuracy: 0.5040
 6496/25000 [======>.......................] - ETA: 1:08 - loss: 7.6029 - accuracy: 0.5042
 6528/25000 [======>.......................] - ETA: 1:07 - loss: 7.5962 - accuracy: 0.5046
 6560/25000 [======>.......................] - ETA: 1:07 - loss: 7.5825 - accuracy: 0.5055
 6592/25000 [======>.......................] - ETA: 1:07 - loss: 7.5829 - accuracy: 0.5055
 6624/25000 [======>.......................] - ETA: 1:07 - loss: 7.5902 - accuracy: 0.5050
 6656/25000 [======>.......................] - ETA: 1:07 - loss: 7.5791 - accuracy: 0.5057
 6688/25000 [=======>......................] - ETA: 1:07 - loss: 7.5657 - accuracy: 0.5066
 6720/25000 [=======>......................] - ETA: 1:07 - loss: 7.5594 - accuracy: 0.5070
 6752/25000 [=======>......................] - ETA: 1:07 - loss: 7.5667 - accuracy: 0.5065
 6784/25000 [=======>......................] - ETA: 1:06 - loss: 7.5649 - accuracy: 0.5066
 6816/25000 [=======>......................] - ETA: 1:06 - loss: 7.5699 - accuracy: 0.5063
 6848/25000 [=======>......................] - ETA: 1:06 - loss: 7.5614 - accuracy: 0.5069
 6880/25000 [=======>......................] - ETA: 1:06 - loss: 7.5619 - accuracy: 0.5068
 6912/25000 [=======>......................] - ETA: 1:06 - loss: 7.5734 - accuracy: 0.5061
 6944/25000 [=======>......................] - ETA: 1:06 - loss: 7.5673 - accuracy: 0.5065
 6976/25000 [=======>......................] - ETA: 1:06 - loss: 7.5721 - accuracy: 0.5062
 7008/25000 [=======>......................] - ETA: 1:06 - loss: 7.5682 - accuracy: 0.5064
 7040/25000 [=======>......................] - ETA: 1:05 - loss: 7.5599 - accuracy: 0.5070
 7072/25000 [=======>......................] - ETA: 1:05 - loss: 7.5625 - accuracy: 0.5068
 7104/25000 [=======>......................] - ETA: 1:05 - loss: 7.5587 - accuracy: 0.5070
 7136/25000 [=======>......................] - ETA: 1:05 - loss: 7.5570 - accuracy: 0.5071
 7168/25000 [=======>......................] - ETA: 1:05 - loss: 7.5597 - accuracy: 0.5070
 7200/25000 [=======>......................] - ETA: 1:05 - loss: 7.5601 - accuracy: 0.5069
 7232/25000 [=======>......................] - ETA: 1:05 - loss: 7.5691 - accuracy: 0.5064
 7264/25000 [=======>......................] - ETA: 1:05 - loss: 7.5674 - accuracy: 0.5065
 7296/25000 [=======>......................] - ETA: 1:04 - loss: 7.5657 - accuracy: 0.5066
 7328/25000 [=======>......................] - ETA: 1:04 - loss: 7.5662 - accuracy: 0.5066
 7360/25000 [=======>......................] - ETA: 1:04 - loss: 7.5666 - accuracy: 0.5065
 7392/25000 [=======>......................] - ETA: 1:04 - loss: 7.5650 - accuracy: 0.5066
 7424/25000 [=======>......................] - ETA: 1:04 - loss: 7.5695 - accuracy: 0.5063
 7456/25000 [=======>......................] - ETA: 1:04 - loss: 7.5720 - accuracy: 0.5062
 7488/25000 [=======>......................] - ETA: 1:04 - loss: 7.5765 - accuracy: 0.5059
 7520/25000 [========>.....................] - ETA: 1:03 - loss: 7.5891 - accuracy: 0.5051
 7552/25000 [========>.....................] - ETA: 1:03 - loss: 7.5854 - accuracy: 0.5053
 7584/25000 [========>.....................] - ETA: 1:03 - loss: 7.5817 - accuracy: 0.5055
 7616/25000 [========>.....................] - ETA: 1:03 - loss: 7.5800 - accuracy: 0.5056
 7648/25000 [========>.....................] - ETA: 1:03 - loss: 7.5664 - accuracy: 0.5065
 7680/25000 [========>.....................] - ETA: 1:03 - loss: 7.5628 - accuracy: 0.5068
 7712/25000 [========>.....................] - ETA: 1:03 - loss: 7.5652 - accuracy: 0.5066
 7744/25000 [========>.....................] - ETA: 1:03 - loss: 7.5597 - accuracy: 0.5070
 7776/25000 [========>.....................] - ETA: 1:03 - loss: 7.5562 - accuracy: 0.5072
 7808/25000 [========>.....................] - ETA: 1:02 - loss: 7.5508 - accuracy: 0.5076
 7840/25000 [========>.....................] - ETA: 1:02 - loss: 7.5512 - accuracy: 0.5075
 7872/25000 [========>.....................] - ETA: 1:02 - loss: 7.5439 - accuracy: 0.5080
 7904/25000 [========>.....................] - ETA: 1:02 - loss: 7.5541 - accuracy: 0.5073
 7936/25000 [========>.....................] - ETA: 1:02 - loss: 7.5468 - accuracy: 0.5078
 7968/25000 [========>.....................] - ETA: 1:02 - loss: 7.5415 - accuracy: 0.5082
 8000/25000 [========>.....................] - ETA: 1:02 - loss: 7.5516 - accuracy: 0.5075
 8032/25000 [========>.....................] - ETA: 1:02 - loss: 7.5559 - accuracy: 0.5072
 8064/25000 [========>.....................] - ETA: 1:01 - loss: 7.5582 - accuracy: 0.5071
 8096/25000 [========>.....................] - ETA: 1:01 - loss: 7.5587 - accuracy: 0.5070
 8128/25000 [========>.....................] - ETA: 1:01 - loss: 7.5515 - accuracy: 0.5075
 8160/25000 [========>.....................] - ETA: 1:01 - loss: 7.5614 - accuracy: 0.5069
 8192/25000 [========>.....................] - ETA: 1:01 - loss: 7.5674 - accuracy: 0.5065
 8224/25000 [========>.....................] - ETA: 1:01 - loss: 7.5697 - accuracy: 0.5063
 8256/25000 [========>.....................] - ETA: 1:01 - loss: 7.5700 - accuracy: 0.5063
 8288/25000 [========>.....................] - ETA: 1:01 - loss: 7.5704 - accuracy: 0.5063
 8320/25000 [========>.....................] - ETA: 1:00 - loss: 7.5634 - accuracy: 0.5067
 8352/25000 [=========>....................] - ETA: 1:00 - loss: 7.5510 - accuracy: 0.5075
 8384/25000 [=========>....................] - ETA: 1:00 - loss: 7.5605 - accuracy: 0.5069
 8416/25000 [=========>....................] - ETA: 1:00 - loss: 7.5555 - accuracy: 0.5072
 8448/25000 [=========>....................] - ETA: 1:00 - loss: 7.5595 - accuracy: 0.5070
 8480/25000 [=========>....................] - ETA: 1:00 - loss: 7.5545 - accuracy: 0.5073
 8512/25000 [=========>....................] - ETA: 1:00 - loss: 7.5441 - accuracy: 0.5080
 8544/25000 [=========>....................] - ETA: 1:00 - loss: 7.5410 - accuracy: 0.5082
 8576/25000 [=========>....................] - ETA: 59s - loss: 7.5343 - accuracy: 0.5086 
 8608/25000 [=========>....................] - ETA: 59s - loss: 7.5366 - accuracy: 0.5085
 8640/25000 [=========>....................] - ETA: 59s - loss: 7.5424 - accuracy: 0.5081
 8672/25000 [=========>....................] - ETA: 59s - loss: 7.5429 - accuracy: 0.5081
 8704/25000 [=========>....................] - ETA: 59s - loss: 7.5415 - accuracy: 0.5082
 8736/25000 [=========>....................] - ETA: 59s - loss: 7.5438 - accuracy: 0.5080
 8768/25000 [=========>....................] - ETA: 59s - loss: 7.5407 - accuracy: 0.5082
 8800/25000 [=========>....................] - ETA: 59s - loss: 7.5429 - accuracy: 0.5081
 8832/25000 [=========>....................] - ETA: 59s - loss: 7.5416 - accuracy: 0.5082
 8864/25000 [=========>....................] - ETA: 58s - loss: 7.5403 - accuracy: 0.5082
 8896/25000 [=========>....................] - ETA: 58s - loss: 7.5356 - accuracy: 0.5085
 8928/25000 [=========>....................] - ETA: 58s - loss: 7.5395 - accuracy: 0.5083
 8960/25000 [=========>....................] - ETA: 58s - loss: 7.5400 - accuracy: 0.5083
 8992/25000 [=========>....................] - ETA: 58s - loss: 7.5455 - accuracy: 0.5079
 9024/25000 [=========>....................] - ETA: 58s - loss: 7.5426 - accuracy: 0.5081
 9056/25000 [=========>....................] - ETA: 58s - loss: 7.5447 - accuracy: 0.5080
 9088/25000 [=========>....................] - ETA: 58s - loss: 7.5451 - accuracy: 0.5079
 9120/25000 [=========>....................] - ETA: 57s - loss: 7.5472 - accuracy: 0.5078
 9152/25000 [=========>....................] - ETA: 57s - loss: 7.5493 - accuracy: 0.5076
 9184/25000 [==========>...................] - ETA: 57s - loss: 7.5464 - accuracy: 0.5078
 9216/25000 [==========>...................] - ETA: 57s - loss: 7.5502 - accuracy: 0.5076
 9248/25000 [==========>...................] - ETA: 57s - loss: 7.5439 - accuracy: 0.5080
 9280/25000 [==========>...................] - ETA: 57s - loss: 7.5394 - accuracy: 0.5083
 9312/25000 [==========>...................] - ETA: 57s - loss: 7.5332 - accuracy: 0.5087
 9344/25000 [==========>...................] - ETA: 57s - loss: 7.5353 - accuracy: 0.5086
 9376/25000 [==========>...................] - ETA: 56s - loss: 7.5358 - accuracy: 0.5085
 9408/25000 [==========>...................] - ETA: 56s - loss: 7.5346 - accuracy: 0.5086
 9440/25000 [==========>...................] - ETA: 56s - loss: 7.5383 - accuracy: 0.5084
 9472/25000 [==========>...................] - ETA: 56s - loss: 7.5306 - accuracy: 0.5089
 9504/25000 [==========>...................] - ETA: 56s - loss: 7.5279 - accuracy: 0.5090
 9536/25000 [==========>...................] - ETA: 56s - loss: 7.5283 - accuracy: 0.5090
 9568/25000 [==========>...................] - ETA: 56s - loss: 7.5240 - accuracy: 0.5093
 9600/25000 [==========>...................] - ETA: 56s - loss: 7.5229 - accuracy: 0.5094
 9632/25000 [==========>...................] - ETA: 55s - loss: 7.5281 - accuracy: 0.5090
 9664/25000 [==========>...................] - ETA: 55s - loss: 7.5286 - accuracy: 0.5090
 9696/25000 [==========>...................] - ETA: 55s - loss: 7.5338 - accuracy: 0.5087
 9728/25000 [==========>...................] - ETA: 55s - loss: 7.5358 - accuracy: 0.5085
 9760/25000 [==========>...................] - ETA: 55s - loss: 7.5347 - accuracy: 0.5086
 9792/25000 [==========>...................] - ETA: 55s - loss: 7.5351 - accuracy: 0.5086
 9824/25000 [==========>...................] - ETA: 55s - loss: 7.5386 - accuracy: 0.5083
 9856/25000 [==========>...................] - ETA: 55s - loss: 7.5437 - accuracy: 0.5080
 9888/25000 [==========>...................] - ETA: 54s - loss: 7.5410 - accuracy: 0.5082
 9920/25000 [==========>...................] - ETA: 54s - loss: 7.5352 - accuracy: 0.5086
 9952/25000 [==========>...................] - ETA: 54s - loss: 7.5357 - accuracy: 0.5085
 9984/25000 [==========>...................] - ETA: 54s - loss: 7.5361 - accuracy: 0.5085
10016/25000 [===========>..................] - ETA: 54s - loss: 7.5350 - accuracy: 0.5086
10048/25000 [===========>..................] - ETA: 54s - loss: 7.5354 - accuracy: 0.5086
10080/25000 [===========>..................] - ETA: 54s - loss: 7.5312 - accuracy: 0.5088
10112/25000 [===========>..................] - ETA: 54s - loss: 7.5377 - accuracy: 0.5084
10144/25000 [===========>..................] - ETA: 53s - loss: 7.5351 - accuracy: 0.5086
10176/25000 [===========>..................] - ETA: 53s - loss: 7.5325 - accuracy: 0.5087
10208/25000 [===========>..................] - ETA: 53s - loss: 7.5299 - accuracy: 0.5089
10240/25000 [===========>..................] - ETA: 53s - loss: 7.5334 - accuracy: 0.5087
10272/25000 [===========>..................] - ETA: 53s - loss: 7.5263 - accuracy: 0.5092
10304/25000 [===========>..................] - ETA: 53s - loss: 7.5253 - accuracy: 0.5092
10336/25000 [===========>..................] - ETA: 53s - loss: 7.5257 - accuracy: 0.5092
10368/25000 [===========>..................] - ETA: 53s - loss: 7.5202 - accuracy: 0.5095
10400/25000 [===========>..................] - ETA: 52s - loss: 7.5162 - accuracy: 0.5098
10432/25000 [===========>..................] - ETA: 52s - loss: 7.5211 - accuracy: 0.5095
10464/25000 [===========>..................] - ETA: 52s - loss: 7.5157 - accuracy: 0.5098
10496/25000 [===========>..................] - ETA: 52s - loss: 7.5308 - accuracy: 0.5089
10528/25000 [===========>..................] - ETA: 52s - loss: 7.5341 - accuracy: 0.5086
10560/25000 [===========>..................] - ETA: 52s - loss: 7.5388 - accuracy: 0.5083
10592/25000 [===========>..................] - ETA: 52s - loss: 7.5407 - accuracy: 0.5082
10624/25000 [===========>..................] - ETA: 52s - loss: 7.5396 - accuracy: 0.5083
10656/25000 [===========>..................] - ETA: 52s - loss: 7.5386 - accuracy: 0.5084
10688/25000 [===========>..................] - ETA: 51s - loss: 7.5432 - accuracy: 0.5080
10720/25000 [===========>..................] - ETA: 51s - loss: 7.5422 - accuracy: 0.5081
10752/25000 [===========>..................] - ETA: 51s - loss: 7.5368 - accuracy: 0.5085
10784/25000 [===========>..................] - ETA: 51s - loss: 7.5401 - accuracy: 0.5083
10816/25000 [===========>..................] - ETA: 51s - loss: 7.5334 - accuracy: 0.5087
10848/25000 [============>.................] - ETA: 51s - loss: 7.5338 - accuracy: 0.5087
10880/25000 [============>.................] - ETA: 51s - loss: 7.5271 - accuracy: 0.5091
10912/25000 [============>.................] - ETA: 51s - loss: 7.5219 - accuracy: 0.5094
10944/25000 [============>.................] - ETA: 50s - loss: 7.5237 - accuracy: 0.5093
10976/25000 [============>.................] - ETA: 50s - loss: 7.5227 - accuracy: 0.5094
11008/25000 [============>.................] - ETA: 50s - loss: 7.5273 - accuracy: 0.5091
11040/25000 [============>.................] - ETA: 50s - loss: 7.5291 - accuracy: 0.5090
11072/25000 [============>.................] - ETA: 50s - loss: 7.5337 - accuracy: 0.5087
11104/25000 [============>.................] - ETA: 50s - loss: 7.5327 - accuracy: 0.5087
11136/25000 [============>.................] - ETA: 50s - loss: 7.5331 - accuracy: 0.5087
11168/25000 [============>.................] - ETA: 50s - loss: 7.5334 - accuracy: 0.5087
11200/25000 [============>.................] - ETA: 49s - loss: 7.5366 - accuracy: 0.5085
11232/25000 [============>.................] - ETA: 49s - loss: 7.5356 - accuracy: 0.5085
11264/25000 [============>.................] - ETA: 49s - loss: 7.5264 - accuracy: 0.5091
11296/25000 [============>.................] - ETA: 49s - loss: 7.5268 - accuracy: 0.5091
11328/25000 [============>.................] - ETA: 49s - loss: 7.5272 - accuracy: 0.5091
11360/25000 [============>.................] - ETA: 49s - loss: 7.5276 - accuracy: 0.5091
11392/25000 [============>.................] - ETA: 49s - loss: 7.5266 - accuracy: 0.5091
11424/25000 [============>.................] - ETA: 49s - loss: 7.5297 - accuracy: 0.5089
11456/25000 [============>.................] - ETA: 48s - loss: 7.5314 - accuracy: 0.5088
11488/25000 [============>.................] - ETA: 48s - loss: 7.5331 - accuracy: 0.5087
11520/25000 [============>.................] - ETA: 48s - loss: 7.5388 - accuracy: 0.5083
11552/25000 [============>.................] - ETA: 48s - loss: 7.5379 - accuracy: 0.5084
11584/25000 [============>.................] - ETA: 48s - loss: 7.5435 - accuracy: 0.5080
11616/25000 [============>.................] - ETA: 48s - loss: 7.5478 - accuracy: 0.5077
11648/25000 [============>.................] - ETA: 48s - loss: 7.5521 - accuracy: 0.5075
11680/25000 [=============>................] - ETA: 48s - loss: 7.5603 - accuracy: 0.5069
11712/25000 [=============>................] - ETA: 47s - loss: 7.5619 - accuracy: 0.5068
11744/25000 [=============>................] - ETA: 47s - loss: 7.5674 - accuracy: 0.5065
11776/25000 [=============>................] - ETA: 47s - loss: 7.5677 - accuracy: 0.5065
11808/25000 [=============>................] - ETA: 47s - loss: 7.5666 - accuracy: 0.5065
11840/25000 [=============>................] - ETA: 47s - loss: 7.5734 - accuracy: 0.5061
11872/25000 [=============>................] - ETA: 47s - loss: 7.5775 - accuracy: 0.5058
11904/25000 [=============>................] - ETA: 47s - loss: 7.5790 - accuracy: 0.5057
11936/25000 [=============>................] - ETA: 47s - loss: 7.5754 - accuracy: 0.5059
11968/25000 [=============>................] - ETA: 47s - loss: 7.5782 - accuracy: 0.5058
12000/25000 [=============>................] - ETA: 46s - loss: 7.5746 - accuracy: 0.5060
12032/25000 [=============>................] - ETA: 46s - loss: 7.5761 - accuracy: 0.5059
12064/25000 [=============>................] - ETA: 46s - loss: 7.5789 - accuracy: 0.5057
12096/25000 [=============>................] - ETA: 46s - loss: 7.5741 - accuracy: 0.5060
12128/25000 [=============>................] - ETA: 46s - loss: 7.5693 - accuracy: 0.5063
12160/25000 [=============>................] - ETA: 46s - loss: 7.5695 - accuracy: 0.5063
12192/25000 [=============>................] - ETA: 46s - loss: 7.5736 - accuracy: 0.5061
12224/25000 [=============>................] - ETA: 46s - loss: 7.5776 - accuracy: 0.5058
12256/25000 [=============>................] - ETA: 45s - loss: 7.5790 - accuracy: 0.5057
12288/25000 [=============>................] - ETA: 45s - loss: 7.5743 - accuracy: 0.5060
12320/25000 [=============>................] - ETA: 45s - loss: 7.5720 - accuracy: 0.5062
12352/25000 [=============>................] - ETA: 45s - loss: 7.5710 - accuracy: 0.5062
12384/25000 [=============>................] - ETA: 45s - loss: 7.5725 - accuracy: 0.5061
12416/25000 [=============>................] - ETA: 45s - loss: 7.5765 - accuracy: 0.5059
12448/25000 [=============>................] - ETA: 45s - loss: 7.5792 - accuracy: 0.5057
12480/25000 [=============>................] - ETA: 45s - loss: 7.5745 - accuracy: 0.5060
12512/25000 [==============>...............] - ETA: 45s - loss: 7.5759 - accuracy: 0.5059
12544/25000 [==============>...............] - ETA: 44s - loss: 7.5737 - accuracy: 0.5061
12576/25000 [==============>...............] - ETA: 44s - loss: 7.5666 - accuracy: 0.5065
12608/25000 [==============>...............] - ETA: 44s - loss: 7.5632 - accuracy: 0.5067
12640/25000 [==============>...............] - ETA: 44s - loss: 7.5599 - accuracy: 0.5070
12672/25000 [==============>...............] - ETA: 44s - loss: 7.5650 - accuracy: 0.5066
12704/25000 [==============>...............] - ETA: 44s - loss: 7.5628 - accuracy: 0.5068
12736/25000 [==============>...............] - ETA: 44s - loss: 7.5667 - accuracy: 0.5065
12768/25000 [==============>...............] - ETA: 44s - loss: 7.5609 - accuracy: 0.5069
12800/25000 [==============>...............] - ETA: 43s - loss: 7.5636 - accuracy: 0.5067
12832/25000 [==============>...............] - ETA: 43s - loss: 7.5639 - accuracy: 0.5067
12864/25000 [==============>...............] - ETA: 43s - loss: 7.5570 - accuracy: 0.5072
12896/25000 [==============>...............] - ETA: 43s - loss: 7.5608 - accuracy: 0.5069
12928/25000 [==============>...............] - ETA: 43s - loss: 7.5634 - accuracy: 0.5067
12960/25000 [==============>...............] - ETA: 43s - loss: 7.5613 - accuracy: 0.5069
12992/25000 [==============>...............] - ETA: 43s - loss: 7.5580 - accuracy: 0.5071
13024/25000 [==============>...............] - ETA: 43s - loss: 7.5618 - accuracy: 0.5068
13056/25000 [==============>...............] - ETA: 42s - loss: 7.5633 - accuracy: 0.5067
13088/25000 [==============>...............] - ETA: 42s - loss: 7.5647 - accuracy: 0.5066
13120/25000 [==============>...............] - ETA: 42s - loss: 7.5696 - accuracy: 0.5063
13152/25000 [==============>...............] - ETA: 42s - loss: 7.5757 - accuracy: 0.5059
13184/25000 [==============>...............] - ETA: 42s - loss: 7.5736 - accuracy: 0.5061
13216/25000 [==============>...............] - ETA: 42s - loss: 7.5784 - accuracy: 0.5058
13248/25000 [==============>...............] - ETA: 42s - loss: 7.5775 - accuracy: 0.5058
13280/25000 [==============>...............] - ETA: 42s - loss: 7.5766 - accuracy: 0.5059
13312/25000 [==============>...............] - ETA: 42s - loss: 7.5768 - accuracy: 0.5059
13344/25000 [===============>..............] - ETA: 41s - loss: 7.5793 - accuracy: 0.5057
13376/25000 [===============>..............] - ETA: 41s - loss: 7.5772 - accuracy: 0.5058
13408/25000 [===============>..............] - ETA: 41s - loss: 7.5774 - accuracy: 0.5058
13440/25000 [===============>..............] - ETA: 41s - loss: 7.5822 - accuracy: 0.5055
13472/25000 [===============>..............] - ETA: 41s - loss: 7.5847 - accuracy: 0.5053
13504/25000 [===============>..............] - ETA: 41s - loss: 7.5883 - accuracy: 0.5051
13536/25000 [===============>..............] - ETA: 41s - loss: 7.5941 - accuracy: 0.5047
13568/25000 [===============>..............] - ETA: 41s - loss: 7.5943 - accuracy: 0.5047
13600/25000 [===============>..............] - ETA: 40s - loss: 7.5967 - accuracy: 0.5046
13632/25000 [===============>..............] - ETA: 40s - loss: 7.5958 - accuracy: 0.5046
13664/25000 [===============>..............] - ETA: 40s - loss: 7.5903 - accuracy: 0.5050
13696/25000 [===============>..............] - ETA: 40s - loss: 7.5883 - accuracy: 0.5051
13728/25000 [===============>..............] - ETA: 40s - loss: 7.5873 - accuracy: 0.5052
13760/25000 [===============>..............] - ETA: 40s - loss: 7.5875 - accuracy: 0.5052
13792/25000 [===============>..............] - ETA: 40s - loss: 7.5977 - accuracy: 0.5045
13824/25000 [===============>..............] - ETA: 40s - loss: 7.5979 - accuracy: 0.5045
13856/25000 [===============>..............] - ETA: 40s - loss: 7.5991 - accuracy: 0.5044
13888/25000 [===============>..............] - ETA: 39s - loss: 7.6015 - accuracy: 0.5042
13920/25000 [===============>..............] - ETA: 39s - loss: 7.5994 - accuracy: 0.5044
13952/25000 [===============>..............] - ETA: 39s - loss: 7.5996 - accuracy: 0.5044
13984/25000 [===============>..............] - ETA: 39s - loss: 7.6041 - accuracy: 0.5041
14016/25000 [===============>..............] - ETA: 39s - loss: 7.6021 - accuracy: 0.5042
14048/25000 [===============>..............] - ETA: 39s - loss: 7.6033 - accuracy: 0.5041
14080/25000 [===============>..............] - ETA: 39s - loss: 7.6078 - accuracy: 0.5038
14112/25000 [===============>..............] - ETA: 39s - loss: 7.6090 - accuracy: 0.5038
14144/25000 [===============>..............] - ETA: 38s - loss: 7.6070 - accuracy: 0.5039
14176/25000 [================>.............] - ETA: 38s - loss: 7.6115 - accuracy: 0.5036
14208/25000 [================>.............] - ETA: 38s - loss: 7.6116 - accuracy: 0.5036
14240/25000 [================>.............] - ETA: 38s - loss: 7.6117 - accuracy: 0.5036
14272/25000 [================>.............] - ETA: 38s - loss: 7.6086 - accuracy: 0.5038
14304/25000 [================>.............] - ETA: 38s - loss: 7.6055 - accuracy: 0.5040
14336/25000 [================>.............] - ETA: 38s - loss: 7.6067 - accuracy: 0.5039
14368/25000 [================>.............] - ETA: 38s - loss: 7.6079 - accuracy: 0.5038
14400/25000 [================>.............] - ETA: 38s - loss: 7.6070 - accuracy: 0.5039
14432/25000 [================>.............] - ETA: 37s - loss: 7.6071 - accuracy: 0.5039
14464/25000 [================>.............] - ETA: 37s - loss: 7.6094 - accuracy: 0.5037
14496/25000 [================>.............] - ETA: 37s - loss: 7.6053 - accuracy: 0.5040
14528/25000 [================>.............] - ETA: 37s - loss: 7.6065 - accuracy: 0.5039
14560/25000 [================>.............] - ETA: 37s - loss: 7.6066 - accuracy: 0.5039
14592/25000 [================>.............] - ETA: 37s - loss: 7.6036 - accuracy: 0.5041
14624/25000 [================>.............] - ETA: 37s - loss: 7.5995 - accuracy: 0.5044
14656/25000 [================>.............] - ETA: 37s - loss: 7.6007 - accuracy: 0.5043
14688/25000 [================>.............] - ETA: 36s - loss: 7.5998 - accuracy: 0.5044
14720/25000 [================>.............] - ETA: 36s - loss: 7.5968 - accuracy: 0.5046
14752/25000 [================>.............] - ETA: 36s - loss: 7.6001 - accuracy: 0.5043
14784/25000 [================>.............] - ETA: 36s - loss: 7.5992 - accuracy: 0.5044
14816/25000 [================>.............] - ETA: 36s - loss: 7.5931 - accuracy: 0.5048
14848/25000 [================>.............] - ETA: 36s - loss: 7.5943 - accuracy: 0.5047
14880/25000 [================>.............] - ETA: 36s - loss: 7.5996 - accuracy: 0.5044
14912/25000 [================>.............] - ETA: 36s - loss: 7.6008 - accuracy: 0.5043
14944/25000 [================>.............] - ETA: 36s - loss: 7.6030 - accuracy: 0.5041
14976/25000 [================>.............] - ETA: 35s - loss: 7.6052 - accuracy: 0.5040
15008/25000 [=================>............] - ETA: 35s - loss: 7.6104 - accuracy: 0.5037
15040/25000 [=================>............] - ETA: 35s - loss: 7.6146 - accuracy: 0.5034
15072/25000 [=================>............] - ETA: 35s - loss: 7.6168 - accuracy: 0.5033
15104/25000 [=================>............] - ETA: 35s - loss: 7.6230 - accuracy: 0.5028
15136/25000 [=================>............] - ETA: 35s - loss: 7.6291 - accuracy: 0.5024
15168/25000 [=================>............] - ETA: 35s - loss: 7.6312 - accuracy: 0.5023
15200/25000 [=================>............] - ETA: 35s - loss: 7.6323 - accuracy: 0.5022
15232/25000 [=================>............] - ETA: 34s - loss: 7.6334 - accuracy: 0.5022
15264/25000 [=================>............] - ETA: 34s - loss: 7.6355 - accuracy: 0.5020
15296/25000 [=================>............] - ETA: 34s - loss: 7.6335 - accuracy: 0.5022
15328/25000 [=================>............] - ETA: 34s - loss: 7.6356 - accuracy: 0.5020
15360/25000 [=================>............] - ETA: 34s - loss: 7.6367 - accuracy: 0.5020
15392/25000 [=================>............] - ETA: 34s - loss: 7.6357 - accuracy: 0.5020
15424/25000 [=================>............] - ETA: 34s - loss: 7.6338 - accuracy: 0.5021
15456/25000 [=================>............] - ETA: 34s - loss: 7.6299 - accuracy: 0.5024
15488/25000 [=================>............] - ETA: 34s - loss: 7.6300 - accuracy: 0.5024
15520/25000 [=================>............] - ETA: 33s - loss: 7.6301 - accuracy: 0.5024
15552/25000 [=================>............] - ETA: 33s - loss: 7.6292 - accuracy: 0.5024
15584/25000 [=================>............] - ETA: 33s - loss: 7.6253 - accuracy: 0.5027
15616/25000 [=================>............] - ETA: 33s - loss: 7.6293 - accuracy: 0.5024
15648/25000 [=================>............] - ETA: 33s - loss: 7.6264 - accuracy: 0.5026
15680/25000 [=================>............] - ETA: 33s - loss: 7.6324 - accuracy: 0.5022
15712/25000 [=================>............] - ETA: 33s - loss: 7.6276 - accuracy: 0.5025
15744/25000 [=================>............] - ETA: 33s - loss: 7.6247 - accuracy: 0.5027
15776/25000 [=================>............] - ETA: 32s - loss: 7.6229 - accuracy: 0.5029
15808/25000 [=================>............] - ETA: 32s - loss: 7.6220 - accuracy: 0.5029
15840/25000 [==================>...........] - ETA: 32s - loss: 7.6260 - accuracy: 0.5027
15872/25000 [==================>...........] - ETA: 32s - loss: 7.6231 - accuracy: 0.5028
15904/25000 [==================>...........] - ETA: 32s - loss: 7.6242 - accuracy: 0.5028
15936/25000 [==================>...........] - ETA: 32s - loss: 7.6204 - accuracy: 0.5030
15968/25000 [==================>...........] - ETA: 32s - loss: 7.6272 - accuracy: 0.5026
16000/25000 [==================>...........] - ETA: 32s - loss: 7.6235 - accuracy: 0.5028
16032/25000 [==================>...........] - ETA: 32s - loss: 7.6255 - accuracy: 0.5027
16064/25000 [==================>...........] - ETA: 31s - loss: 7.6294 - accuracy: 0.5024
16096/25000 [==================>...........] - ETA: 31s - loss: 7.6295 - accuracy: 0.5024
16128/25000 [==================>...........] - ETA: 31s - loss: 7.6324 - accuracy: 0.5022
16160/25000 [==================>...........] - ETA: 31s - loss: 7.6315 - accuracy: 0.5023
16192/25000 [==================>...........] - ETA: 31s - loss: 7.6335 - accuracy: 0.5022
16224/25000 [==================>...........] - ETA: 31s - loss: 7.6345 - accuracy: 0.5021
16256/25000 [==================>...........] - ETA: 31s - loss: 7.6364 - accuracy: 0.5020
16288/25000 [==================>...........] - ETA: 31s - loss: 7.6374 - accuracy: 0.5019
16320/25000 [==================>...........] - ETA: 30s - loss: 7.6384 - accuracy: 0.5018
16352/25000 [==================>...........] - ETA: 30s - loss: 7.6376 - accuracy: 0.5019
16384/25000 [==================>...........] - ETA: 30s - loss: 7.6376 - accuracy: 0.5019
16416/25000 [==================>...........] - ETA: 30s - loss: 7.6386 - accuracy: 0.5018
16448/25000 [==================>...........] - ETA: 30s - loss: 7.6396 - accuracy: 0.5018
16480/25000 [==================>...........] - ETA: 30s - loss: 7.6406 - accuracy: 0.5017
16512/25000 [==================>...........] - ETA: 30s - loss: 7.6415 - accuracy: 0.5016
16544/25000 [==================>...........] - ETA: 30s - loss: 7.6407 - accuracy: 0.5017
16576/25000 [==================>...........] - ETA: 30s - loss: 7.6389 - accuracy: 0.5018
16608/25000 [==================>...........] - ETA: 29s - loss: 7.6380 - accuracy: 0.5019
16640/25000 [==================>...........] - ETA: 29s - loss: 7.6390 - accuracy: 0.5018
16672/25000 [===================>..........] - ETA: 29s - loss: 7.6372 - accuracy: 0.5019
16704/25000 [===================>..........] - ETA: 29s - loss: 7.6372 - accuracy: 0.5019
16736/25000 [===================>..........] - ETA: 29s - loss: 7.6364 - accuracy: 0.5020
16768/25000 [===================>..........] - ETA: 29s - loss: 7.6383 - accuracy: 0.5018
16800/25000 [===================>..........] - ETA: 29s - loss: 7.6392 - accuracy: 0.5018
16832/25000 [===================>..........] - ETA: 29s - loss: 7.6366 - accuracy: 0.5020
16864/25000 [===================>..........] - ETA: 29s - loss: 7.6357 - accuracy: 0.5020
16896/25000 [===================>..........] - ETA: 28s - loss: 7.6294 - accuracy: 0.5024
16928/25000 [===================>..........] - ETA: 28s - loss: 7.6277 - accuracy: 0.5025
16960/25000 [===================>..........] - ETA: 28s - loss: 7.6268 - accuracy: 0.5026
16992/25000 [===================>..........] - ETA: 28s - loss: 7.6260 - accuracy: 0.5026
17024/25000 [===================>..........] - ETA: 28s - loss: 7.6279 - accuracy: 0.5025
17056/25000 [===================>..........] - ETA: 28s - loss: 7.6307 - accuracy: 0.5023
17088/25000 [===================>..........] - ETA: 28s - loss: 7.6316 - accuracy: 0.5023
17120/25000 [===================>..........] - ETA: 28s - loss: 7.6344 - accuracy: 0.5021
17152/25000 [===================>..........] - ETA: 27s - loss: 7.6398 - accuracy: 0.5017
17184/25000 [===================>..........] - ETA: 27s - loss: 7.6363 - accuracy: 0.5020
17216/25000 [===================>..........] - ETA: 27s - loss: 7.6354 - accuracy: 0.5020
17248/25000 [===================>..........] - ETA: 27s - loss: 7.6311 - accuracy: 0.5023
17280/25000 [===================>..........] - ETA: 27s - loss: 7.6320 - accuracy: 0.5023
17312/25000 [===================>..........] - ETA: 27s - loss: 7.6374 - accuracy: 0.5019
17344/25000 [===================>..........] - ETA: 27s - loss: 7.6330 - accuracy: 0.5022
17376/25000 [===================>..........] - ETA: 27s - loss: 7.6331 - accuracy: 0.5022
17408/25000 [===================>..........] - ETA: 27s - loss: 7.6314 - accuracy: 0.5023
17440/25000 [===================>..........] - ETA: 26s - loss: 7.6279 - accuracy: 0.5025
17472/25000 [===================>..........] - ETA: 26s - loss: 7.6263 - accuracy: 0.5026
17504/25000 [====================>.........] - ETA: 26s - loss: 7.6246 - accuracy: 0.5027
17536/25000 [====================>.........] - ETA: 26s - loss: 7.6246 - accuracy: 0.5027
17568/25000 [====================>.........] - ETA: 26s - loss: 7.6230 - accuracy: 0.5028
17600/25000 [====================>.........] - ETA: 26s - loss: 7.6222 - accuracy: 0.5029
17632/25000 [====================>.........] - ETA: 26s - loss: 7.6197 - accuracy: 0.5031
17664/25000 [====================>.........] - ETA: 26s - loss: 7.6171 - accuracy: 0.5032
17696/25000 [====================>.........] - ETA: 26s - loss: 7.6138 - accuracy: 0.5034
17728/25000 [====================>.........] - ETA: 25s - loss: 7.6113 - accuracy: 0.5036
17760/25000 [====================>.........] - ETA: 25s - loss: 7.6105 - accuracy: 0.5037
17792/25000 [====================>.........] - ETA: 25s - loss: 7.6140 - accuracy: 0.5034
17824/25000 [====================>.........] - ETA: 25s - loss: 7.6141 - accuracy: 0.5034
17856/25000 [====================>.........] - ETA: 25s - loss: 7.6091 - accuracy: 0.5038
17888/25000 [====================>.........] - ETA: 25s - loss: 7.6109 - accuracy: 0.5036
17920/25000 [====================>.........] - ETA: 25s - loss: 7.6119 - accuracy: 0.5036
17952/25000 [====================>.........] - ETA: 25s - loss: 7.6094 - accuracy: 0.5037
17984/25000 [====================>.........] - ETA: 24s - loss: 7.6095 - accuracy: 0.5037
18016/25000 [====================>.........] - ETA: 24s - loss: 7.6139 - accuracy: 0.5034
18048/25000 [====================>.........] - ETA: 24s - loss: 7.6148 - accuracy: 0.5034
18080/25000 [====================>.........] - ETA: 24s - loss: 7.6132 - accuracy: 0.5035
18112/25000 [====================>.........] - ETA: 24s - loss: 7.6124 - accuracy: 0.5035
18144/25000 [====================>.........] - ETA: 24s - loss: 7.6083 - accuracy: 0.5038
18176/25000 [====================>.........] - ETA: 24s - loss: 7.6093 - accuracy: 0.5037
18208/25000 [====================>.........] - ETA: 24s - loss: 7.6077 - accuracy: 0.5038
18240/25000 [====================>.........] - ETA: 24s - loss: 7.6111 - accuracy: 0.5036
18272/25000 [====================>.........] - ETA: 23s - loss: 7.6087 - accuracy: 0.5038
18304/25000 [====================>.........] - ETA: 23s - loss: 7.6088 - accuracy: 0.5038
18336/25000 [=====================>........] - ETA: 23s - loss: 7.6081 - accuracy: 0.5038
18368/25000 [=====================>........] - ETA: 23s - loss: 7.6065 - accuracy: 0.5039
18400/25000 [=====================>........] - ETA: 23s - loss: 7.6058 - accuracy: 0.5040
18432/25000 [=====================>........] - ETA: 23s - loss: 7.6084 - accuracy: 0.5038
18464/25000 [=====================>........] - ETA: 23s - loss: 7.6077 - accuracy: 0.5038
18496/25000 [=====================>........] - ETA: 23s - loss: 7.6086 - accuracy: 0.5038
18528/25000 [=====================>........] - ETA: 23s - loss: 7.6087 - accuracy: 0.5038
18560/25000 [=====================>........] - ETA: 22s - loss: 7.6104 - accuracy: 0.5037
18592/25000 [=====================>........] - ETA: 22s - loss: 7.6105 - accuracy: 0.5037
18624/25000 [=====================>........] - ETA: 22s - loss: 7.6115 - accuracy: 0.5036
18656/25000 [=====================>........] - ETA: 22s - loss: 7.6107 - accuracy: 0.5036
18688/25000 [=====================>........] - ETA: 22s - loss: 7.6100 - accuracy: 0.5037
18720/25000 [=====================>........] - ETA: 22s - loss: 7.6134 - accuracy: 0.5035
18752/25000 [=====================>........] - ETA: 22s - loss: 7.6135 - accuracy: 0.5035
18784/25000 [=====================>........] - ETA: 22s - loss: 7.6168 - accuracy: 0.5032
18816/25000 [=====================>........] - ETA: 22s - loss: 7.6161 - accuracy: 0.5033
18848/25000 [=====================>........] - ETA: 21s - loss: 7.6186 - accuracy: 0.5031
18880/25000 [=====================>........] - ETA: 21s - loss: 7.6179 - accuracy: 0.5032
18912/25000 [=====================>........] - ETA: 21s - loss: 7.6188 - accuracy: 0.5031
18944/25000 [=====================>........] - ETA: 21s - loss: 7.6172 - accuracy: 0.5032
18976/25000 [=====================>........] - ETA: 21s - loss: 7.6181 - accuracy: 0.5032
19008/25000 [=====================>........] - ETA: 21s - loss: 7.6182 - accuracy: 0.5032
19040/25000 [=====================>........] - ETA: 21s - loss: 7.6207 - accuracy: 0.5030
19072/25000 [=====================>........] - ETA: 21s - loss: 7.6176 - accuracy: 0.5032
19104/25000 [=====================>........] - ETA: 20s - loss: 7.6153 - accuracy: 0.5034
19136/25000 [=====================>........] - ETA: 20s - loss: 7.6145 - accuracy: 0.5034
19168/25000 [======================>.......] - ETA: 20s - loss: 7.6114 - accuracy: 0.5036
19200/25000 [======================>.......] - ETA: 20s - loss: 7.6091 - accuracy: 0.5038
19232/25000 [======================>.......] - ETA: 20s - loss: 7.6148 - accuracy: 0.5034
19264/25000 [======================>.......] - ETA: 20s - loss: 7.6149 - accuracy: 0.5034
19296/25000 [======================>.......] - ETA: 20s - loss: 7.6174 - accuracy: 0.5032
19328/25000 [======================>.......] - ETA: 20s - loss: 7.6190 - accuracy: 0.5031
19360/25000 [======================>.......] - ETA: 20s - loss: 7.6175 - accuracy: 0.5032
19392/25000 [======================>.......] - ETA: 19s - loss: 7.6176 - accuracy: 0.5032
19424/25000 [======================>.......] - ETA: 19s - loss: 7.6153 - accuracy: 0.5033
19456/25000 [======================>.......] - ETA: 19s - loss: 7.6170 - accuracy: 0.5032
19488/25000 [======================>.......] - ETA: 19s - loss: 7.6210 - accuracy: 0.5030
19520/25000 [======================>.......] - ETA: 19s - loss: 7.6218 - accuracy: 0.5029
19552/25000 [======================>.......] - ETA: 19s - loss: 7.6251 - accuracy: 0.5027
19584/25000 [======================>.......] - ETA: 19s - loss: 7.6251 - accuracy: 0.5027
19616/25000 [======================>.......] - ETA: 19s - loss: 7.6275 - accuracy: 0.5025
19648/25000 [======================>.......] - ETA: 19s - loss: 7.6276 - accuracy: 0.5025
19680/25000 [======================>.......] - ETA: 18s - loss: 7.6308 - accuracy: 0.5023
19712/25000 [======================>.......] - ETA: 18s - loss: 7.6301 - accuracy: 0.5024
19744/25000 [======================>.......] - ETA: 18s - loss: 7.6301 - accuracy: 0.5024
19776/25000 [======================>.......] - ETA: 18s - loss: 7.6248 - accuracy: 0.5027
19808/25000 [======================>.......] - ETA: 18s - loss: 7.6240 - accuracy: 0.5028
19840/25000 [======================>.......] - ETA: 18s - loss: 7.6233 - accuracy: 0.5028
19872/25000 [======================>.......] - ETA: 18s - loss: 7.6250 - accuracy: 0.5027
19904/25000 [======================>.......] - ETA: 18s - loss: 7.6242 - accuracy: 0.5028
19936/25000 [======================>.......] - ETA: 18s - loss: 7.6243 - accuracy: 0.5028
19968/25000 [======================>.......] - ETA: 17s - loss: 7.6252 - accuracy: 0.5027
20000/25000 [=======================>......] - ETA: 17s - loss: 7.6222 - accuracy: 0.5029
20032/25000 [=======================>......] - ETA: 17s - loss: 7.6222 - accuracy: 0.5029
20064/25000 [=======================>......] - ETA: 17s - loss: 7.6238 - accuracy: 0.5028
20096/25000 [=======================>......] - ETA: 17s - loss: 7.6277 - accuracy: 0.5025
20128/25000 [=======================>......] - ETA: 17s - loss: 7.6278 - accuracy: 0.5025
20160/25000 [=======================>......] - ETA: 17s - loss: 7.6263 - accuracy: 0.5026
20192/25000 [=======================>......] - ETA: 17s - loss: 7.6279 - accuracy: 0.5025
20224/25000 [=======================>......] - ETA: 16s - loss: 7.6302 - accuracy: 0.5024
20256/25000 [=======================>......] - ETA: 16s - loss: 7.6288 - accuracy: 0.5025
20288/25000 [=======================>......] - ETA: 16s - loss: 7.6288 - accuracy: 0.5025
20320/25000 [=======================>......] - ETA: 16s - loss: 7.6319 - accuracy: 0.5023
20352/25000 [=======================>......] - ETA: 16s - loss: 7.6365 - accuracy: 0.5020
20384/25000 [=======================>......] - ETA: 16s - loss: 7.6373 - accuracy: 0.5019
20416/25000 [=======================>......] - ETA: 16s - loss: 7.6388 - accuracy: 0.5018
20448/25000 [=======================>......] - ETA: 16s - loss: 7.6419 - accuracy: 0.5016
20480/25000 [=======================>......] - ETA: 16s - loss: 7.6389 - accuracy: 0.5018
20512/25000 [=======================>......] - ETA: 15s - loss: 7.6397 - accuracy: 0.5018
20544/25000 [=======================>......] - ETA: 15s - loss: 7.6412 - accuracy: 0.5017
20576/25000 [=======================>......] - ETA: 15s - loss: 7.6420 - accuracy: 0.5016
20608/25000 [=======================>......] - ETA: 15s - loss: 7.6421 - accuracy: 0.5016
20640/25000 [=======================>......] - ETA: 15s - loss: 7.6436 - accuracy: 0.5015
20672/25000 [=======================>......] - ETA: 15s - loss: 7.6436 - accuracy: 0.5015
20704/25000 [=======================>......] - ETA: 15s - loss: 7.6466 - accuracy: 0.5013
20736/25000 [=======================>......] - ETA: 15s - loss: 7.6444 - accuracy: 0.5014
20768/25000 [=======================>......] - ETA: 15s - loss: 7.6445 - accuracy: 0.5014
20800/25000 [=======================>......] - ETA: 14s - loss: 7.6423 - accuracy: 0.5016
20832/25000 [=======================>......] - ETA: 14s - loss: 7.6416 - accuracy: 0.5016
20864/25000 [========================>.....] - ETA: 14s - loss: 7.6438 - accuracy: 0.5015
20896/25000 [========================>.....] - ETA: 14s - loss: 7.6468 - accuracy: 0.5013
20928/25000 [========================>.....] - ETA: 14s - loss: 7.6446 - accuracy: 0.5014
20960/25000 [========================>.....] - ETA: 14s - loss: 7.6476 - accuracy: 0.5012
20992/25000 [========================>.....] - ETA: 14s - loss: 7.6484 - accuracy: 0.5012
21024/25000 [========================>.....] - ETA: 14s - loss: 7.6469 - accuracy: 0.5013
21056/25000 [========================>.....] - ETA: 14s - loss: 7.6484 - accuracy: 0.5012
21088/25000 [========================>.....] - ETA: 13s - loss: 7.6499 - accuracy: 0.5011
21120/25000 [========================>.....] - ETA: 13s - loss: 7.6485 - accuracy: 0.5012
21152/25000 [========================>.....] - ETA: 13s - loss: 7.6514 - accuracy: 0.5010
21184/25000 [========================>.....] - ETA: 13s - loss: 7.6492 - accuracy: 0.5011
21216/25000 [========================>.....] - ETA: 13s - loss: 7.6486 - accuracy: 0.5012
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6507 - accuracy: 0.5010
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6522 - accuracy: 0.5009
21312/25000 [========================>.....] - ETA: 13s - loss: 7.6494 - accuracy: 0.5011
21344/25000 [========================>.....] - ETA: 12s - loss: 7.6472 - accuracy: 0.5013
21376/25000 [========================>.....] - ETA: 12s - loss: 7.6451 - accuracy: 0.5014
21408/25000 [========================>.....] - ETA: 12s - loss: 7.6451 - accuracy: 0.5014
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6452 - accuracy: 0.5014
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6452 - accuracy: 0.5014
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6438 - accuracy: 0.5015
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6467 - accuracy: 0.5013
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6503 - accuracy: 0.5011
21600/25000 [========================>.....] - ETA: 12s - loss: 7.6524 - accuracy: 0.5009
21632/25000 [========================>.....] - ETA: 11s - loss: 7.6524 - accuracy: 0.5009
21664/25000 [========================>.....] - ETA: 11s - loss: 7.6518 - accuracy: 0.5010
21696/25000 [=========================>....] - ETA: 11s - loss: 7.6539 - accuracy: 0.5008
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6532 - accuracy: 0.5009
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6546 - accuracy: 0.5008
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6554 - accuracy: 0.5007
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6519 - accuracy: 0.5010
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6561 - accuracy: 0.5007
21888/25000 [=========================>....] - ETA: 11s - loss: 7.6575 - accuracy: 0.5006
21920/25000 [=========================>....] - ETA: 10s - loss: 7.6596 - accuracy: 0.5005
21952/25000 [=========================>....] - ETA: 10s - loss: 7.6582 - accuracy: 0.5005
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6589 - accuracy: 0.5005
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6597 - accuracy: 0.5005
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6631 - accuracy: 0.5002
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6638 - accuracy: 0.5002
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6652 - accuracy: 0.5001
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6659 - accuracy: 0.5000
22176/25000 [=========================>....] - ETA: 10s - loss: 7.6687 - accuracy: 0.4999
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6715 - accuracy: 0.4997 
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6687 - accuracy: 0.4999
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6673 - accuracy: 0.5000
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6680 - accuracy: 0.4999
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6666 - accuracy: 0.5000
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6680 - accuracy: 0.4999
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6700 - accuracy: 0.4998
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6700 - accuracy: 0.4998
22464/25000 [=========================>....] - ETA: 9s - loss: 7.6734 - accuracy: 0.4996
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6734 - accuracy: 0.4996
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6761 - accuracy: 0.4994
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6755 - accuracy: 0.4994
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6741 - accuracy: 0.4995
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6761 - accuracy: 0.4994
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6768 - accuracy: 0.4993
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6774 - accuracy: 0.4993
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6794 - accuracy: 0.4992
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6788 - accuracy: 0.4992
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6781 - accuracy: 0.4993
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6821 - accuracy: 0.4990
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6794 - accuracy: 0.4992
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6800 - accuracy: 0.4991
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6793 - accuracy: 0.4992
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6800 - accuracy: 0.4991
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6806 - accuracy: 0.4991
23008/25000 [==========================>...] - ETA: 7s - loss: 7.6806 - accuracy: 0.4991
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6813 - accuracy: 0.4990
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6753 - accuracy: 0.4994
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6739 - accuracy: 0.4995
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6713 - accuracy: 0.4997
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6726 - accuracy: 0.4996
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6712 - accuracy: 0.4997
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6732 - accuracy: 0.4996
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6739 - accuracy: 0.4995
23296/25000 [==========================>...] - ETA: 6s - loss: 7.6745 - accuracy: 0.4995
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6758 - accuracy: 0.4994
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6791 - accuracy: 0.4992
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6791 - accuracy: 0.4992
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6791 - accuracy: 0.4992
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6823 - accuracy: 0.4990
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6797 - accuracy: 0.4991
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6784 - accuracy: 0.4992
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6803 - accuracy: 0.4991
23584/25000 [===========================>..] - ETA: 5s - loss: 7.6796 - accuracy: 0.4992
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6764 - accuracy: 0.4994
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6757 - accuracy: 0.4994
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6783 - accuracy: 0.4992
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6789 - accuracy: 0.4992
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6757 - accuracy: 0.4994
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6750 - accuracy: 0.4995
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6763 - accuracy: 0.4994
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6769 - accuracy: 0.4993
23872/25000 [===========================>..] - ETA: 4s - loss: 7.6782 - accuracy: 0.4992
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6775 - accuracy: 0.4993
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6769 - accuracy: 0.4993
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6743 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6743 - accuracy: 0.4995
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6724 - accuracy: 0.4996
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6704 - accuracy: 0.4998
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6723 - accuracy: 0.4996
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6742 - accuracy: 0.4995
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
24192/25000 [============================>.] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24224/25000 [============================>.] - ETA: 2s - loss: 7.6761 - accuracy: 0.4994
24256/25000 [============================>.] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24288/25000 [============================>.] - ETA: 2s - loss: 7.6761 - accuracy: 0.4994
24320/25000 [============================>.] - ETA: 2s - loss: 7.6742 - accuracy: 0.4995
24352/25000 [============================>.] - ETA: 2s - loss: 7.6754 - accuracy: 0.4994
24384/25000 [============================>.] - ETA: 2s - loss: 7.6735 - accuracy: 0.4995
24416/25000 [============================>.] - ETA: 2s - loss: 7.6735 - accuracy: 0.4995
24448/25000 [============================>.] - ETA: 1s - loss: 7.6741 - accuracy: 0.4995
24480/25000 [============================>.] - ETA: 1s - loss: 7.6754 - accuracy: 0.4994
24512/25000 [============================>.] - ETA: 1s - loss: 7.6766 - accuracy: 0.4993
24544/25000 [============================>.] - ETA: 1s - loss: 7.6779 - accuracy: 0.4993
24576/25000 [============================>.] - ETA: 1s - loss: 7.6772 - accuracy: 0.4993
24608/25000 [============================>.] - ETA: 1s - loss: 7.6753 - accuracy: 0.4994
24640/25000 [============================>.] - ETA: 1s - loss: 7.6778 - accuracy: 0.4993
24672/25000 [============================>.] - ETA: 1s - loss: 7.6766 - accuracy: 0.4994
24704/25000 [============================>.] - ETA: 1s - loss: 7.6734 - accuracy: 0.4996
24736/25000 [============================>.] - ETA: 0s - loss: 7.6747 - accuracy: 0.4995
24768/25000 [============================>.] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24800/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24832/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24864/25000 [============================>.] - ETA: 0s - loss: 7.6709 - accuracy: 0.4997
24896/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24928/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
24960/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 107s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
