
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/76b7a81be9b27c2e92c4951280c0a8da664b997c', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '76b7a81be9b27c2e92c4951280c0a8da664b997c', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/76b7a81be9b27c2e92c4951280c0a8da664b997c

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/76b7a81be9b27c2e92c4951280c0a8da664b997c

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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:48<01:12, 24.32s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.2759329047217108, 'embedding_size_factor': 0.8366681585986646, 'layers.choice': 0, 'learning_rate': 0.0019192536470358775, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 1.909522266579947e-11} and reward: 0.3762
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd1\xa8\xe2|j\xe1\xcdX\x15\x00\x00\x00embedding_size_factorq\x03G?\xea\xc5\xfcMY%RX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?_q\xee\xe9dP\xf9X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xb4\xfe\xd3\xcd\x8b\xe7xu.' and reward: 0.3762
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd1\xa8\xe2|j\xe1\xcdX\x15\x00\x00\x00embedding_size_factorq\x03G?\xea\xc5\xfcMY%RX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?_q\xee\xe9dP\xf9X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xb4\xfe\xd3\xcd\x8b\xe7xu.' and reward: 0.3762
 60%|██████    | 3/5 [01:38<01:03, 31.94s/it] 60%|██████    | 3/5 [01:38<01:05, 32.79s/it]
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.49437081100945696, 'embedding_size_factor': 0.6686478828093482, 'layers.choice': 1, 'learning_rate': 0.00011200732707726717, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 1.1317974427082314e-05} and reward: 0.3668
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdf\xa3\xc5xX|\xc1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5e\x90>\xa6\x95\xd7X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x1d\\\xaf:\x11\xfc\x86X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xe7\xbcJ\x8f\x88\xcb\x1eu.' and reward: 0.3668
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdf\xa3\xc5xX|\xc1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5e\x90>\xa6\x95\xd7X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x1d\\\xaf:\x11\xfc\x86X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xe7\xbcJ\x8f\x88\xcb\x1eu.' and reward: 0.3668

Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 184.27259755134583
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.77s of the -67.04s of remaining time.
Ensemble size: 20
Ensemble weights: 
[0.6  0.35 0.05]
	0.3894	 = Validation accuracy score
	1.05s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 188.12s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f3592fa2ac8> 

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
 [ 0.12050025 -0.03669774  0.01219031 -0.00382647  0.07746646 -0.06484777]
 [-0.01224358  0.06330863 -0.12365095  0.12836596 -0.00974778  0.17839034]
 [-0.06612115 -0.08108819  0.0316787   0.12411465  0.04922161  0.10920994]
 [ 0.31799954  0.32003832  0.02773327  0.4386313   0.4755218   0.19416414]
 [-0.13648193  0.36417231 -0.21797849 -0.19230214  0.4126454  -0.29465646]
 [-0.40430528  0.08581524  0.06737974  0.38297024  0.00174318  0.05586901]
 [ 0.3128821  -0.04393619 -0.21123409  0.0099151   0.39105392  0.32056704]
 [-0.04026484  0.08835644  0.07611439  0.22828516  0.00568217  0.24829875]
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
{'loss': 0.4823354408144951, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-16 14:33:20.377263: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.44971857964992523, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-16 14:33:21.549207: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 3358720/17464789 [====>.........................] - ETA: 0s
10510336/17464789 [=================>............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-16 14:33:33.728348: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 14:33:33.733124: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-16 14:33:33.733297: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5604f010cf00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 14:33:33.733314: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:40 - loss: 7.6666 - accuracy: 0.5000
   64/25000 [..............................] - ETA: 3:07 - loss: 8.6249 - accuracy: 0.4375
   96/25000 [..............................] - ETA: 2:34 - loss: 8.4652 - accuracy: 0.4479
  128/25000 [..............................] - ETA: 2:16 - loss: 8.2656 - accuracy: 0.4609
  160/25000 [..............................] - ETA: 2:07 - loss: 8.2416 - accuracy: 0.4625
  192/25000 [..............................] - ETA: 2:02 - loss: 8.0659 - accuracy: 0.4740
  224/25000 [..............................] - ETA: 1:57 - loss: 8.2142 - accuracy: 0.4643
  256/25000 [..............................] - ETA: 1:53 - loss: 8.2656 - accuracy: 0.4609
  288/25000 [..............................] - ETA: 1:50 - loss: 8.2523 - accuracy: 0.4618
  320/25000 [..............................] - ETA: 1:48 - loss: 8.4333 - accuracy: 0.4500
  352/25000 [..............................] - ETA: 1:45 - loss: 8.3636 - accuracy: 0.4545
  384/25000 [..............................] - ETA: 1:44 - loss: 8.3454 - accuracy: 0.4557
  416/25000 [..............................] - ETA: 1:42 - loss: 8.2564 - accuracy: 0.4615
  448/25000 [..............................] - ETA: 1:41 - loss: 8.3511 - accuracy: 0.4554
  480/25000 [..............................] - ETA: 1:41 - loss: 8.2736 - accuracy: 0.4604
  512/25000 [..............................] - ETA: 1:40 - loss: 8.0859 - accuracy: 0.4727
  544/25000 [..............................] - ETA: 1:39 - loss: 8.0049 - accuracy: 0.4779
  576/25000 [..............................] - ETA: 1:39 - loss: 8.1192 - accuracy: 0.4705
  608/25000 [..............................] - ETA: 1:38 - loss: 8.0197 - accuracy: 0.4770
  640/25000 [..............................] - ETA: 1:37 - loss: 8.0739 - accuracy: 0.4734
  672/25000 [..............................] - ETA: 1:37 - loss: 8.0545 - accuracy: 0.4747
  704/25000 [..............................] - ETA: 1:37 - loss: 8.0369 - accuracy: 0.4759
  736/25000 [..............................] - ETA: 1:36 - loss: 7.9791 - accuracy: 0.4796
  768/25000 [..............................] - ETA: 1:36 - loss: 8.0060 - accuracy: 0.4779
  800/25000 [..............................] - ETA: 1:36 - loss: 7.9925 - accuracy: 0.4787
  832/25000 [..............................] - ETA: 1:36 - loss: 7.9615 - accuracy: 0.4808
  864/25000 [>.............................] - ETA: 1:35 - loss: 7.8973 - accuracy: 0.4850
  896/25000 [>.............................] - ETA: 1:35 - loss: 7.9404 - accuracy: 0.4821
  928/25000 [>.............................] - ETA: 1:35 - loss: 7.9640 - accuracy: 0.4806
  960/25000 [>.............................] - ETA: 1:34 - loss: 7.9381 - accuracy: 0.4823
  992/25000 [>.............................] - ETA: 1:34 - loss: 7.9294 - accuracy: 0.4829
 1024/25000 [>.............................] - ETA: 1:34 - loss: 7.9212 - accuracy: 0.4834
 1056/25000 [>.............................] - ETA: 1:34 - loss: 7.8409 - accuracy: 0.4886
 1088/25000 [>.............................] - ETA: 1:33 - loss: 7.9203 - accuracy: 0.4835
 1120/25000 [>.............................] - ETA: 1:33 - loss: 7.8857 - accuracy: 0.4857
 1152/25000 [>.............................] - ETA: 1:33 - loss: 7.8796 - accuracy: 0.4861
 1184/25000 [>.............................] - ETA: 1:32 - loss: 7.8479 - accuracy: 0.4882
 1216/25000 [>.............................] - ETA: 1:32 - loss: 7.8305 - accuracy: 0.4893
 1248/25000 [>.............................] - ETA: 1:32 - loss: 7.8141 - accuracy: 0.4904
 1280/25000 [>.............................] - ETA: 1:31 - loss: 7.8583 - accuracy: 0.4875
 1312/25000 [>.............................] - ETA: 1:31 - loss: 7.8069 - accuracy: 0.4909
 1344/25000 [>.............................] - ETA: 1:31 - loss: 7.7351 - accuracy: 0.4955
 1376/25000 [>.............................] - ETA: 1:30 - loss: 7.7112 - accuracy: 0.4971
 1408/25000 [>.............................] - ETA: 1:30 - loss: 7.7646 - accuracy: 0.4936
 1440/25000 [>.............................] - ETA: 1:30 - loss: 7.7625 - accuracy: 0.4938
 1472/25000 [>.............................] - ETA: 1:30 - loss: 7.7708 - accuracy: 0.4932
 1504/25000 [>.............................] - ETA: 1:29 - loss: 7.7788 - accuracy: 0.4927
 1536/25000 [>.............................] - ETA: 1:29 - loss: 7.7964 - accuracy: 0.4915
 1568/25000 [>.............................] - ETA: 1:29 - loss: 7.8329 - accuracy: 0.4892
 1600/25000 [>.............................] - ETA: 1:29 - loss: 7.8487 - accuracy: 0.4881
 1632/25000 [>.............................] - ETA: 1:29 - loss: 7.8169 - accuracy: 0.4902
 1664/25000 [>.............................] - ETA: 1:29 - loss: 7.8048 - accuracy: 0.4910
 1696/25000 [=>............................] - ETA: 1:28 - loss: 7.7932 - accuracy: 0.4917
 1728/25000 [=>............................] - ETA: 1:28 - loss: 7.8263 - accuracy: 0.4896
 1760/25000 [=>............................] - ETA: 1:28 - loss: 7.8321 - accuracy: 0.4892
 1792/25000 [=>............................] - ETA: 1:28 - loss: 7.8292 - accuracy: 0.4894
 1824/25000 [=>............................] - ETA: 1:28 - loss: 7.8432 - accuracy: 0.4885
 1856/25000 [=>............................] - ETA: 1:28 - loss: 7.8566 - accuracy: 0.4876
 1888/25000 [=>............................] - ETA: 1:28 - loss: 7.8534 - accuracy: 0.4878
 1920/25000 [=>............................] - ETA: 1:28 - loss: 7.8343 - accuracy: 0.4891
 1952/25000 [=>............................] - ETA: 1:28 - loss: 7.8159 - accuracy: 0.4903
 1984/25000 [=>............................] - ETA: 1:27 - loss: 7.8057 - accuracy: 0.4909
 2016/25000 [=>............................] - ETA: 1:27 - loss: 7.8263 - accuracy: 0.4896
 2048/25000 [=>............................] - ETA: 1:27 - loss: 7.8089 - accuracy: 0.4907
 2080/25000 [=>............................] - ETA: 1:27 - loss: 7.8214 - accuracy: 0.4899
 2112/25000 [=>............................] - ETA: 1:27 - loss: 7.8263 - accuracy: 0.4896
 2144/25000 [=>............................] - ETA: 1:27 - loss: 7.8311 - accuracy: 0.4893
 2176/25000 [=>............................] - ETA: 1:26 - loss: 7.8075 - accuracy: 0.4908
 2208/25000 [=>............................] - ETA: 1:26 - loss: 7.8263 - accuracy: 0.4896
 2240/25000 [=>............................] - ETA: 1:26 - loss: 7.8377 - accuracy: 0.4888
 2272/25000 [=>............................] - ETA: 1:26 - loss: 7.8421 - accuracy: 0.4886
 2304/25000 [=>............................] - ETA: 1:26 - loss: 7.8596 - accuracy: 0.4874
 2336/25000 [=>............................] - ETA: 1:26 - loss: 7.8504 - accuracy: 0.4880
 2368/25000 [=>............................] - ETA: 1:26 - loss: 7.8479 - accuracy: 0.4882
 2400/25000 [=>............................] - ETA: 1:26 - loss: 7.8391 - accuracy: 0.4888
 2432/25000 [=>............................] - ETA: 1:25 - loss: 7.8368 - accuracy: 0.4889
 2464/25000 [=>............................] - ETA: 1:25 - loss: 7.8346 - accuracy: 0.4890
 2496/25000 [=>............................] - ETA: 1:25 - loss: 7.8693 - accuracy: 0.4868
 2528/25000 [==>...........................] - ETA: 1:25 - loss: 7.8668 - accuracy: 0.4869
 2560/25000 [==>...........................] - ETA: 1:25 - loss: 7.8882 - accuracy: 0.4855
 2592/25000 [==>...........................] - ETA: 1:25 - loss: 7.8677 - accuracy: 0.4869
 2624/25000 [==>...........................] - ETA: 1:25 - loss: 7.8887 - accuracy: 0.4855
 2656/25000 [==>...........................] - ETA: 1:25 - loss: 7.8687 - accuracy: 0.4868
 2688/25000 [==>...........................] - ETA: 1:24 - loss: 7.8720 - accuracy: 0.4866
 2720/25000 [==>...........................] - ETA: 1:24 - loss: 7.8696 - accuracy: 0.4868
 2752/25000 [==>...........................] - ETA: 1:24 - loss: 7.8783 - accuracy: 0.4862
 2784/25000 [==>...........................] - ETA: 1:24 - loss: 7.8814 - accuracy: 0.4860
 2816/25000 [==>...........................] - ETA: 1:24 - loss: 7.8681 - accuracy: 0.4869
 2848/25000 [==>...........................] - ETA: 1:24 - loss: 7.9035 - accuracy: 0.4846
 2880/25000 [==>...........................] - ETA: 1:23 - loss: 7.9115 - accuracy: 0.4840
 2912/25000 [==>...........................] - ETA: 1:23 - loss: 7.8983 - accuracy: 0.4849
 2944/25000 [==>...........................] - ETA: 1:23 - loss: 7.8906 - accuracy: 0.4854
 2976/25000 [==>...........................] - ETA: 1:23 - loss: 7.8727 - accuracy: 0.4866
 3008/25000 [==>...........................] - ETA: 1:23 - loss: 7.8807 - accuracy: 0.4860
 3040/25000 [==>...........................] - ETA: 1:22 - loss: 7.8936 - accuracy: 0.4852
 3072/25000 [==>...........................] - ETA: 1:22 - loss: 7.8812 - accuracy: 0.4860
 3104/25000 [==>...........................] - ETA: 1:22 - loss: 7.8939 - accuracy: 0.4852
 3136/25000 [==>...........................] - ETA: 1:22 - loss: 7.8866 - accuracy: 0.4857
 3168/25000 [==>...........................] - ETA: 1:22 - loss: 7.8699 - accuracy: 0.4867
 3200/25000 [==>...........................] - ETA: 1:22 - loss: 7.8727 - accuracy: 0.4866
 3232/25000 [==>...........................] - ETA: 1:22 - loss: 7.8896 - accuracy: 0.4855
 3264/25000 [==>...........................] - ETA: 1:21 - loss: 7.8921 - accuracy: 0.4853
 3296/25000 [==>...........................] - ETA: 1:21 - loss: 7.8992 - accuracy: 0.4848
 3328/25000 [==>...........................] - ETA: 1:21 - loss: 7.8924 - accuracy: 0.4853
 3360/25000 [===>..........................] - ETA: 1:21 - loss: 7.8720 - accuracy: 0.4866
 3392/25000 [===>..........................] - ETA: 1:21 - loss: 7.8791 - accuracy: 0.4861
 3424/25000 [===>..........................] - ETA: 1:21 - loss: 7.8547 - accuracy: 0.4877
 3456/25000 [===>..........................] - ETA: 1:21 - loss: 7.8751 - accuracy: 0.4864
 3488/25000 [===>..........................] - ETA: 1:20 - loss: 7.8644 - accuracy: 0.4871
 3520/25000 [===>..........................] - ETA: 1:20 - loss: 7.8670 - accuracy: 0.4869
 3552/25000 [===>..........................] - ETA: 1:20 - loss: 7.8911 - accuracy: 0.4854
 3584/25000 [===>..........................] - ETA: 1:20 - loss: 7.9148 - accuracy: 0.4838
 3616/25000 [===>..........................] - ETA: 1:20 - loss: 7.9295 - accuracy: 0.4829
 3648/25000 [===>..........................] - ETA: 1:20 - loss: 7.9188 - accuracy: 0.4836
 3680/25000 [===>..........................] - ETA: 1:20 - loss: 7.9166 - accuracy: 0.4837
 3712/25000 [===>..........................] - ETA: 1:20 - loss: 7.9351 - accuracy: 0.4825
 3744/25000 [===>..........................] - ETA: 1:19 - loss: 7.9164 - accuracy: 0.4837
 3776/25000 [===>..........................] - ETA: 1:19 - loss: 7.9021 - accuracy: 0.4846
 3808/25000 [===>..........................] - ETA: 1:19 - loss: 7.9002 - accuracy: 0.4848
 3840/25000 [===>..........................] - ETA: 1:19 - loss: 7.8902 - accuracy: 0.4854
 3872/25000 [===>..........................] - ETA: 1:19 - loss: 7.8805 - accuracy: 0.4861
 3904/25000 [===>..........................] - ETA: 1:19 - loss: 7.8826 - accuracy: 0.4859
 3936/25000 [===>..........................] - ETA: 1:19 - loss: 7.8809 - accuracy: 0.4860
 3968/25000 [===>..........................] - ETA: 1:19 - loss: 7.8830 - accuracy: 0.4859
 4000/25000 [===>..........................] - ETA: 1:18 - loss: 7.8660 - accuracy: 0.4870
 4032/25000 [===>..........................] - ETA: 1:18 - loss: 7.8758 - accuracy: 0.4864
 4064/25000 [===>..........................] - ETA: 1:18 - loss: 7.8741 - accuracy: 0.4865
 4096/25000 [===>..........................] - ETA: 1:18 - loss: 7.8763 - accuracy: 0.4863
 4128/25000 [===>..........................] - ETA: 1:18 - loss: 7.8821 - accuracy: 0.4859
 4160/25000 [===>..........................] - ETA: 1:18 - loss: 7.8878 - accuracy: 0.4856
 4192/25000 [====>.........................] - ETA: 1:18 - loss: 7.8897 - accuracy: 0.4854
 4224/25000 [====>.........................] - ETA: 1:18 - loss: 7.8917 - accuracy: 0.4853
 4256/25000 [====>.........................] - ETA: 1:18 - loss: 7.8828 - accuracy: 0.4859
 4288/25000 [====>.........................] - ETA: 1:17 - loss: 7.8812 - accuracy: 0.4860
 4320/25000 [====>.........................] - ETA: 1:17 - loss: 7.8689 - accuracy: 0.4868
 4352/25000 [====>.........................] - ETA: 1:17 - loss: 7.8710 - accuracy: 0.4867
 4384/25000 [====>.........................] - ETA: 1:17 - loss: 7.8660 - accuracy: 0.4870
 4416/25000 [====>.........................] - ETA: 1:17 - loss: 7.8819 - accuracy: 0.4860
 4448/25000 [====>.........................] - ETA: 1:17 - loss: 7.8735 - accuracy: 0.4865
 4480/25000 [====>.........................] - ETA: 1:17 - loss: 7.8686 - accuracy: 0.4868
 4512/25000 [====>.........................] - ETA: 1:16 - loss: 7.8603 - accuracy: 0.4874
 4544/25000 [====>.........................] - ETA: 1:16 - loss: 7.8691 - accuracy: 0.4868
 4576/25000 [====>.........................] - ETA: 1:16 - loss: 7.8744 - accuracy: 0.4865
 4608/25000 [====>.........................] - ETA: 1:16 - loss: 7.8763 - accuracy: 0.4863
 4640/25000 [====>.........................] - ETA: 1:16 - loss: 7.8649 - accuracy: 0.4871
 4672/25000 [====>.........................] - ETA: 1:16 - loss: 7.8767 - accuracy: 0.4863
 4704/25000 [====>.........................] - ETA: 1:16 - loss: 7.8883 - accuracy: 0.4855
 4736/25000 [====>.........................] - ETA: 1:16 - loss: 7.8900 - accuracy: 0.4854
 4768/25000 [====>.........................] - ETA: 1:15 - loss: 7.9014 - accuracy: 0.4847
 4800/25000 [====>.........................] - ETA: 1:15 - loss: 7.9062 - accuracy: 0.4844
 4832/25000 [====>.........................] - ETA: 1:15 - loss: 7.9078 - accuracy: 0.4843
 4864/25000 [====>.........................] - ETA: 1:15 - loss: 7.9283 - accuracy: 0.4829
 4896/25000 [====>.........................] - ETA: 1:15 - loss: 7.9109 - accuracy: 0.4841
 4928/25000 [====>.........................] - ETA: 1:15 - loss: 7.9093 - accuracy: 0.4842
 4960/25000 [====>.........................] - ETA: 1:15 - loss: 7.9201 - accuracy: 0.4835
 4992/25000 [====>.........................] - ETA: 1:14 - loss: 7.9154 - accuracy: 0.4838
 5024/25000 [=====>........................] - ETA: 1:14 - loss: 7.9230 - accuracy: 0.4833
 5056/25000 [=====>........................] - ETA: 1:14 - loss: 7.9183 - accuracy: 0.4836
 5088/25000 [=====>........................] - ETA: 1:14 - loss: 7.9228 - accuracy: 0.4833
 5120/25000 [=====>........................] - ETA: 1:14 - loss: 7.9152 - accuracy: 0.4838
 5152/25000 [=====>........................] - ETA: 1:14 - loss: 7.9196 - accuracy: 0.4835
 5184/25000 [=====>........................] - ETA: 1:14 - loss: 7.9210 - accuracy: 0.4834
 5216/25000 [=====>........................] - ETA: 1:14 - loss: 7.9136 - accuracy: 0.4839
 5248/25000 [=====>........................] - ETA: 1:13 - loss: 7.9062 - accuracy: 0.4844
 5280/25000 [=====>........................] - ETA: 1:13 - loss: 7.9222 - accuracy: 0.4833
 5312/25000 [=====>........................] - ETA: 1:13 - loss: 7.9149 - accuracy: 0.4838
 5344/25000 [=====>........................] - ETA: 1:13 - loss: 7.9019 - accuracy: 0.4847
 5376/25000 [=====>........................] - ETA: 1:13 - loss: 7.9119 - accuracy: 0.4840
 5408/25000 [=====>........................] - ETA: 1:13 - loss: 7.9161 - accuracy: 0.4837
 5440/25000 [=====>........................] - ETA: 1:13 - loss: 7.9175 - accuracy: 0.4836
 5472/25000 [=====>........................] - ETA: 1:13 - loss: 7.9132 - accuracy: 0.4839
 5504/25000 [=====>........................] - ETA: 1:12 - loss: 7.9006 - accuracy: 0.4847
 5536/25000 [=====>........................] - ETA: 1:12 - loss: 7.8965 - accuracy: 0.4850
 5568/25000 [=====>........................] - ETA: 1:12 - loss: 7.8924 - accuracy: 0.4853
 5600/25000 [=====>........................] - ETA: 1:12 - loss: 7.8857 - accuracy: 0.4857
 5632/25000 [=====>........................] - ETA: 1:12 - loss: 7.8871 - accuracy: 0.4856
 5664/25000 [=====>........................] - ETA: 1:12 - loss: 7.8913 - accuracy: 0.4853
 5696/25000 [=====>........................] - ETA: 1:12 - loss: 7.8927 - accuracy: 0.4853
 5728/25000 [=====>........................] - ETA: 1:12 - loss: 7.8915 - accuracy: 0.4853
 5760/25000 [=====>........................] - ETA: 1:11 - loss: 7.8929 - accuracy: 0.4852
 5792/25000 [=====>........................] - ETA: 1:11 - loss: 7.8890 - accuracy: 0.4855
 5824/25000 [=====>........................] - ETA: 1:11 - loss: 7.8825 - accuracy: 0.4859
 5856/25000 [======>.......................] - ETA: 1:11 - loss: 7.8839 - accuracy: 0.4858
 5888/25000 [======>.......................] - ETA: 1:11 - loss: 7.8776 - accuracy: 0.4862
 5920/25000 [======>.......................] - ETA: 1:11 - loss: 7.8868 - accuracy: 0.4856
 5952/25000 [======>.......................] - ETA: 1:11 - loss: 7.8830 - accuracy: 0.4859
 5984/25000 [======>.......................] - ETA: 1:11 - loss: 7.8742 - accuracy: 0.4865
 6016/25000 [======>.......................] - ETA: 1:10 - loss: 7.8705 - accuracy: 0.4867
 6048/25000 [======>.......................] - ETA: 1:10 - loss: 7.8694 - accuracy: 0.4868
 6080/25000 [======>.......................] - ETA: 1:10 - loss: 7.8684 - accuracy: 0.4868
 6112/25000 [======>.......................] - ETA: 1:10 - loss: 7.8748 - accuracy: 0.4864
 6144/25000 [======>.......................] - ETA: 1:10 - loss: 7.8862 - accuracy: 0.4857
 6176/25000 [======>.......................] - ETA: 1:10 - loss: 7.8727 - accuracy: 0.4866
 6208/25000 [======>.......................] - ETA: 1:10 - loss: 7.8741 - accuracy: 0.4865
 6240/25000 [======>.......................] - ETA: 1:10 - loss: 7.8681 - accuracy: 0.4869
 6272/25000 [======>.......................] - ETA: 1:09 - loss: 7.8671 - accuracy: 0.4869
 6304/25000 [======>.......................] - ETA: 1:09 - loss: 7.8661 - accuracy: 0.4870
 6336/25000 [======>.......................] - ETA: 1:09 - loss: 7.8747 - accuracy: 0.4864
 6368/25000 [======>.......................] - ETA: 1:09 - loss: 7.8665 - accuracy: 0.4870
 6400/25000 [======>.......................] - ETA: 1:09 - loss: 7.8631 - accuracy: 0.4872
 6432/25000 [======>.......................] - ETA: 1:09 - loss: 7.8454 - accuracy: 0.4883
 6464/25000 [======>.......................] - ETA: 1:09 - loss: 7.8540 - accuracy: 0.4878
 6496/25000 [======>.......................] - ETA: 1:08 - loss: 7.8531 - accuracy: 0.4878
 6528/25000 [======>.......................] - ETA: 1:08 - loss: 7.8545 - accuracy: 0.4877
 6560/25000 [======>.......................] - ETA: 1:08 - loss: 7.8536 - accuracy: 0.4878
 6592/25000 [======>.......................] - ETA: 1:08 - loss: 7.8550 - accuracy: 0.4877
 6624/25000 [======>.......................] - ETA: 1:08 - loss: 7.8564 - accuracy: 0.4876
 6656/25000 [======>.......................] - ETA: 1:08 - loss: 7.8486 - accuracy: 0.4881
 6688/25000 [=======>......................] - ETA: 1:08 - loss: 7.8500 - accuracy: 0.4880
 6720/25000 [=======>......................] - ETA: 1:08 - loss: 7.8423 - accuracy: 0.4885
 6752/25000 [=======>......................] - ETA: 1:07 - loss: 7.8460 - accuracy: 0.4883
 6784/25000 [=======>......................] - ETA: 1:07 - loss: 7.8497 - accuracy: 0.4881
 6816/25000 [=======>......................] - ETA: 1:07 - loss: 7.8376 - accuracy: 0.4888
 6848/25000 [=======>......................] - ETA: 1:07 - loss: 7.8256 - accuracy: 0.4896
 6880/25000 [=======>......................] - ETA: 1:07 - loss: 7.8204 - accuracy: 0.4900
 6912/25000 [=======>......................] - ETA: 1:07 - loss: 7.8108 - accuracy: 0.4906
 6944/25000 [=======>......................] - ETA: 1:07 - loss: 7.8101 - accuracy: 0.4906
 6976/25000 [=======>......................] - ETA: 1:07 - loss: 7.8117 - accuracy: 0.4905
 7008/25000 [=======>......................] - ETA: 1:06 - loss: 7.8045 - accuracy: 0.4910
 7040/25000 [=======>......................] - ETA: 1:06 - loss: 7.8060 - accuracy: 0.4909
 7072/25000 [=======>......................] - ETA: 1:06 - loss: 7.8054 - accuracy: 0.4910
 7104/25000 [=======>......................] - ETA: 1:06 - loss: 7.8091 - accuracy: 0.4907
 7136/25000 [=======>......................] - ETA: 1:06 - loss: 7.8084 - accuracy: 0.4908
 7168/25000 [=======>......................] - ETA: 1:06 - loss: 7.8057 - accuracy: 0.4909
 7200/25000 [=======>......................] - ETA: 1:06 - loss: 7.8072 - accuracy: 0.4908
 7232/25000 [=======>......................] - ETA: 1:06 - loss: 7.7960 - accuracy: 0.4916
 7264/25000 [=======>......................] - ETA: 1:05 - loss: 7.7933 - accuracy: 0.4917
 7296/25000 [=======>......................] - ETA: 1:05 - loss: 7.8074 - accuracy: 0.4908
 7328/25000 [=======>......................] - ETA: 1:05 - loss: 7.8215 - accuracy: 0.4899
 7360/25000 [=======>......................] - ETA: 1:05 - loss: 7.8208 - accuracy: 0.4899
 7392/25000 [=======>......................] - ETA: 1:05 - loss: 7.8180 - accuracy: 0.4901
 7424/25000 [=======>......................] - ETA: 1:05 - loss: 7.8174 - accuracy: 0.4902
 7456/25000 [=======>......................] - ETA: 1:05 - loss: 7.8106 - accuracy: 0.4906
 7488/25000 [=======>......................] - ETA: 1:05 - loss: 7.8120 - accuracy: 0.4905
 7520/25000 [========>.....................] - ETA: 1:04 - loss: 7.8114 - accuracy: 0.4906
 7552/25000 [========>.....................] - ETA: 1:04 - loss: 7.8128 - accuracy: 0.4905
 7584/25000 [========>.....................] - ETA: 1:04 - loss: 7.8162 - accuracy: 0.4902
 7616/25000 [========>.....................] - ETA: 1:04 - loss: 7.8156 - accuracy: 0.4903
 7648/25000 [========>.....................] - ETA: 1:04 - loss: 7.8130 - accuracy: 0.4905
 7680/25000 [========>.....................] - ETA: 1:04 - loss: 7.8164 - accuracy: 0.4902
 7712/25000 [========>.....................] - ETA: 1:04 - loss: 7.8217 - accuracy: 0.4899
 7744/25000 [========>.....................] - ETA: 1:04 - loss: 7.8230 - accuracy: 0.4898
 7776/25000 [========>.....................] - ETA: 1:03 - loss: 7.8106 - accuracy: 0.4906
 7808/25000 [========>.....................] - ETA: 1:03 - loss: 7.8041 - accuracy: 0.4910
 7840/25000 [========>.....................] - ETA: 1:03 - loss: 7.8035 - accuracy: 0.4911
 7872/25000 [========>.....................] - ETA: 1:03 - loss: 7.8127 - accuracy: 0.4905
 7904/25000 [========>.....................] - ETA: 1:03 - loss: 7.8005 - accuracy: 0.4913
 7936/25000 [========>.....................] - ETA: 1:03 - loss: 7.7980 - accuracy: 0.4914
 7968/25000 [========>.....................] - ETA: 1:03 - loss: 7.7936 - accuracy: 0.4917
 8000/25000 [========>.....................] - ETA: 1:03 - loss: 7.7950 - accuracy: 0.4916
 8032/25000 [========>.....................] - ETA: 1:02 - loss: 7.8041 - accuracy: 0.4910
 8064/25000 [========>.....................] - ETA: 1:02 - loss: 7.8092 - accuracy: 0.4907
 8096/25000 [========>.....................] - ETA: 1:02 - loss: 7.8143 - accuracy: 0.4904
 8128/25000 [========>.....................] - ETA: 1:02 - loss: 7.8232 - accuracy: 0.4898
 8160/25000 [========>.....................] - ETA: 1:02 - loss: 7.8245 - accuracy: 0.4897
 8192/25000 [========>.....................] - ETA: 1:02 - loss: 7.8313 - accuracy: 0.4893
 8224/25000 [========>.....................] - ETA: 1:02 - loss: 7.8307 - accuracy: 0.4893
 8256/25000 [========>.....................] - ETA: 1:02 - loss: 7.8282 - accuracy: 0.4895
 8288/25000 [========>.....................] - ETA: 1:01 - loss: 7.8368 - accuracy: 0.4889
 8320/25000 [========>.....................] - ETA: 1:01 - loss: 7.8399 - accuracy: 0.4887
 8352/25000 [=========>....................] - ETA: 1:01 - loss: 7.8484 - accuracy: 0.4881
 8384/25000 [=========>....................] - ETA: 1:01 - loss: 7.8385 - accuracy: 0.4888
 8416/25000 [=========>....................] - ETA: 1:01 - loss: 7.8361 - accuracy: 0.4889
 8448/25000 [=========>....................] - ETA: 1:01 - loss: 7.8318 - accuracy: 0.4892
 8480/25000 [=========>....................] - ETA: 1:01 - loss: 7.8330 - accuracy: 0.4892
 8512/25000 [=========>....................] - ETA: 1:01 - loss: 7.8341 - accuracy: 0.4891
 8544/25000 [=========>....................] - ETA: 1:00 - loss: 7.8461 - accuracy: 0.4883
 8576/25000 [=========>....................] - ETA: 1:00 - loss: 7.8400 - accuracy: 0.4887
 8608/25000 [=========>....................] - ETA: 1:00 - loss: 7.8376 - accuracy: 0.4888
 8640/25000 [=========>....................] - ETA: 1:00 - loss: 7.8352 - accuracy: 0.4890
 8672/25000 [=========>....................] - ETA: 1:00 - loss: 7.8364 - accuracy: 0.4889
 8704/25000 [=========>....................] - ETA: 1:00 - loss: 7.8340 - accuracy: 0.4891
 8736/25000 [=========>....................] - ETA: 1:00 - loss: 7.8299 - accuracy: 0.4894
 8768/25000 [=========>....................] - ETA: 1:00 - loss: 7.8275 - accuracy: 0.4895
 8800/25000 [=========>....................] - ETA: 59s - loss: 7.8269 - accuracy: 0.4895 
 8832/25000 [=========>....................] - ETA: 59s - loss: 7.8194 - accuracy: 0.4900
 8864/25000 [=========>....................] - ETA: 59s - loss: 7.8188 - accuracy: 0.4901
 8896/25000 [=========>....................] - ETA: 59s - loss: 7.8148 - accuracy: 0.4903
 8928/25000 [=========>....................] - ETA: 59s - loss: 7.8160 - accuracy: 0.4903
 8960/25000 [=========>....................] - ETA: 59s - loss: 7.8121 - accuracy: 0.4905
 8992/25000 [=========>....................] - ETA: 59s - loss: 7.8082 - accuracy: 0.4908
 9024/25000 [=========>....................] - ETA: 59s - loss: 7.8076 - accuracy: 0.4908
 9056/25000 [=========>....................] - ETA: 59s - loss: 7.8055 - accuracy: 0.4909
 9088/25000 [=========>....................] - ETA: 58s - loss: 7.8083 - accuracy: 0.4908
 9120/25000 [=========>....................] - ETA: 58s - loss: 7.8078 - accuracy: 0.4908
 9152/25000 [=========>....................] - ETA: 58s - loss: 7.8074 - accuracy: 0.4908
 9184/25000 [==========>...................] - ETA: 58s - loss: 7.8069 - accuracy: 0.4909
 9216/25000 [==========>...................] - ETA: 58s - loss: 7.8064 - accuracy: 0.4909
 9248/25000 [==========>...................] - ETA: 58s - loss: 7.7959 - accuracy: 0.4916
 9280/25000 [==========>...................] - ETA: 58s - loss: 7.7889 - accuracy: 0.4920
 9312/25000 [==========>...................] - ETA: 58s - loss: 7.7967 - accuracy: 0.4915
 9344/25000 [==========>...................] - ETA: 58s - loss: 7.7979 - accuracy: 0.4914
 9376/25000 [==========>...................] - ETA: 57s - loss: 7.7942 - accuracy: 0.4917
 9408/25000 [==========>...................] - ETA: 57s - loss: 7.7986 - accuracy: 0.4914
 9440/25000 [==========>...................] - ETA: 57s - loss: 7.7917 - accuracy: 0.4918
 9472/25000 [==========>...................] - ETA: 57s - loss: 7.7816 - accuracy: 0.4925
 9504/25000 [==========>...................] - ETA: 57s - loss: 7.7699 - accuracy: 0.4933
 9536/25000 [==========>...................] - ETA: 57s - loss: 7.7663 - accuracy: 0.4935
 9568/25000 [==========>...................] - ETA: 57s - loss: 7.7660 - accuracy: 0.4935
 9600/25000 [==========>...................] - ETA: 57s - loss: 7.7609 - accuracy: 0.4939
 9632/25000 [==========>...................] - ETA: 56s - loss: 7.7590 - accuracy: 0.4940
 9664/25000 [==========>...................] - ETA: 56s - loss: 7.7650 - accuracy: 0.4936
 9696/25000 [==========>...................] - ETA: 56s - loss: 7.7710 - accuracy: 0.4932
 9728/25000 [==========>...................] - ETA: 56s - loss: 7.7706 - accuracy: 0.4932
 9760/25000 [==========>...................] - ETA: 56s - loss: 7.7719 - accuracy: 0.4931
 9792/25000 [==========>...................] - ETA: 56s - loss: 7.7700 - accuracy: 0.4933
 9824/25000 [==========>...................] - ETA: 56s - loss: 7.7681 - accuracy: 0.4934
 9856/25000 [==========>...................] - ETA: 56s - loss: 7.7677 - accuracy: 0.4934
 9888/25000 [==========>...................] - ETA: 55s - loss: 7.7674 - accuracy: 0.4934
 9920/25000 [==========>...................] - ETA: 55s - loss: 7.7702 - accuracy: 0.4932
 9952/25000 [==========>...................] - ETA: 55s - loss: 7.7652 - accuracy: 0.4936
 9984/25000 [==========>...................] - ETA: 55s - loss: 7.7588 - accuracy: 0.4940
10016/25000 [===========>..................] - ETA: 55s - loss: 7.7523 - accuracy: 0.4944
10048/25000 [===========>..................] - ETA: 55s - loss: 7.7521 - accuracy: 0.4944
10080/25000 [===========>..................] - ETA: 55s - loss: 7.7472 - accuracy: 0.4947
10112/25000 [===========>..................] - ETA: 55s - loss: 7.7470 - accuracy: 0.4948
10144/25000 [===========>..................] - ETA: 54s - loss: 7.7482 - accuracy: 0.4947
10176/25000 [===========>..................] - ETA: 54s - loss: 7.7525 - accuracy: 0.4944
10208/25000 [===========>..................] - ETA: 54s - loss: 7.7507 - accuracy: 0.4945
10240/25000 [===========>..................] - ETA: 54s - loss: 7.7520 - accuracy: 0.4944
10272/25000 [===========>..................] - ETA: 54s - loss: 7.7547 - accuracy: 0.4943
10304/25000 [===========>..................] - ETA: 54s - loss: 7.7544 - accuracy: 0.4943
10336/25000 [===========>..................] - ETA: 54s - loss: 7.7541 - accuracy: 0.4943
10368/25000 [===========>..................] - ETA: 54s - loss: 7.7554 - accuracy: 0.4942
10400/25000 [===========>..................] - ETA: 54s - loss: 7.7580 - accuracy: 0.4940
10432/25000 [===========>..................] - ETA: 53s - loss: 7.7622 - accuracy: 0.4938
10464/25000 [===========>..................] - ETA: 53s - loss: 7.7619 - accuracy: 0.4938
10496/25000 [===========>..................] - ETA: 53s - loss: 7.7703 - accuracy: 0.4932
10528/25000 [===========>..................] - ETA: 53s - loss: 7.7729 - accuracy: 0.4931
10560/25000 [===========>..................] - ETA: 53s - loss: 7.7639 - accuracy: 0.4937
10592/25000 [===========>..................] - ETA: 53s - loss: 7.7665 - accuracy: 0.4935
10624/25000 [===========>..................] - ETA: 53s - loss: 7.7691 - accuracy: 0.4933
10656/25000 [===========>..................] - ETA: 53s - loss: 7.7688 - accuracy: 0.4933
10688/25000 [===========>..................] - ETA: 52s - loss: 7.7656 - accuracy: 0.4935
10720/25000 [===========>..................] - ETA: 52s - loss: 7.7696 - accuracy: 0.4933
10752/25000 [===========>..................] - ETA: 52s - loss: 7.7736 - accuracy: 0.4930
10784/25000 [===========>..................] - ETA: 52s - loss: 7.7690 - accuracy: 0.4933
10816/25000 [===========>..................] - ETA: 52s - loss: 7.7701 - accuracy: 0.4933
10848/25000 [============>.................] - ETA: 52s - loss: 7.7670 - accuracy: 0.4935
10880/25000 [============>.................] - ETA: 52s - loss: 7.7639 - accuracy: 0.4937
10912/25000 [============>.................] - ETA: 52s - loss: 7.7622 - accuracy: 0.4938
10944/25000 [============>.................] - ETA: 52s - loss: 7.7661 - accuracy: 0.4935
10976/25000 [============>.................] - ETA: 51s - loss: 7.7616 - accuracy: 0.4938
11008/25000 [============>.................] - ETA: 51s - loss: 7.7599 - accuracy: 0.4939
11040/25000 [============>.................] - ETA: 51s - loss: 7.7569 - accuracy: 0.4941
11072/25000 [============>.................] - ETA: 51s - loss: 7.7594 - accuracy: 0.4939
11104/25000 [============>.................] - ETA: 51s - loss: 7.7578 - accuracy: 0.4941
11136/25000 [============>.................] - ETA: 51s - loss: 7.7602 - accuracy: 0.4939
11168/25000 [============>.................] - ETA: 51s - loss: 7.7517 - accuracy: 0.4944
11200/25000 [============>.................] - ETA: 51s - loss: 7.7556 - accuracy: 0.4942
11232/25000 [============>.................] - ETA: 51s - loss: 7.7540 - accuracy: 0.4943
11264/25000 [============>.................] - ETA: 50s - loss: 7.7551 - accuracy: 0.4942
11296/25000 [============>.................] - ETA: 50s - loss: 7.7616 - accuracy: 0.4938
11328/25000 [============>.................] - ETA: 50s - loss: 7.7614 - accuracy: 0.4938
11360/25000 [============>.................] - ETA: 50s - loss: 7.7625 - accuracy: 0.4938
11392/25000 [============>.................] - ETA: 50s - loss: 7.7649 - accuracy: 0.4936
11424/25000 [============>.................] - ETA: 50s - loss: 7.7659 - accuracy: 0.4935
11456/25000 [============>.................] - ETA: 50s - loss: 7.7670 - accuracy: 0.4935
11488/25000 [============>.................] - ETA: 50s - loss: 7.7721 - accuracy: 0.4931
11520/25000 [============>.................] - ETA: 49s - loss: 7.7678 - accuracy: 0.4934
11552/25000 [============>.................] - ETA: 49s - loss: 7.7702 - accuracy: 0.4932
11584/25000 [============>.................] - ETA: 49s - loss: 7.7699 - accuracy: 0.4933
11616/25000 [============>.................] - ETA: 49s - loss: 7.7722 - accuracy: 0.4931
11648/25000 [============>.................] - ETA: 49s - loss: 7.7746 - accuracy: 0.4930
11680/25000 [=============>................] - ETA: 49s - loss: 7.7756 - accuracy: 0.4929
11712/25000 [=============>................] - ETA: 49s - loss: 7.7753 - accuracy: 0.4929
11744/25000 [=============>................] - ETA: 49s - loss: 7.7776 - accuracy: 0.4928
11776/25000 [=============>................] - ETA: 48s - loss: 7.7786 - accuracy: 0.4927
11808/25000 [=============>................] - ETA: 48s - loss: 7.7731 - accuracy: 0.4931
11840/25000 [=============>................] - ETA: 48s - loss: 7.7741 - accuracy: 0.4930
11872/25000 [=============>................] - ETA: 48s - loss: 7.7777 - accuracy: 0.4928
11904/25000 [=============>................] - ETA: 48s - loss: 7.7722 - accuracy: 0.4931
11936/25000 [=============>................] - ETA: 48s - loss: 7.7758 - accuracy: 0.4929
11968/25000 [=============>................] - ETA: 48s - loss: 7.7730 - accuracy: 0.4931
12000/25000 [=============>................] - ETA: 48s - loss: 7.7778 - accuracy: 0.4927
12032/25000 [=============>................] - ETA: 48s - loss: 7.7775 - accuracy: 0.4928
12064/25000 [=============>................] - ETA: 47s - loss: 7.7785 - accuracy: 0.4927
12096/25000 [=============>................] - ETA: 47s - loss: 7.7794 - accuracy: 0.4926
12128/25000 [=============>................] - ETA: 47s - loss: 7.7791 - accuracy: 0.4927
12160/25000 [=============>................] - ETA: 47s - loss: 7.7839 - accuracy: 0.4924
12192/25000 [=============>................] - ETA: 47s - loss: 7.7811 - accuracy: 0.4925
12224/25000 [=============>................] - ETA: 47s - loss: 7.7845 - accuracy: 0.4923
12256/25000 [=============>................] - ETA: 47s - loss: 7.7805 - accuracy: 0.4926
12288/25000 [=============>................] - ETA: 47s - loss: 7.7827 - accuracy: 0.4924
12320/25000 [=============>................] - ETA: 46s - loss: 7.7898 - accuracy: 0.4920
12352/25000 [=============>................] - ETA: 46s - loss: 7.7883 - accuracy: 0.4921
12384/25000 [=============>................] - ETA: 46s - loss: 7.7904 - accuracy: 0.4919
12416/25000 [=============>................] - ETA: 46s - loss: 7.7876 - accuracy: 0.4921
12448/25000 [=============>................] - ETA: 46s - loss: 7.7836 - accuracy: 0.4924
12480/25000 [=============>................] - ETA: 46s - loss: 7.7760 - accuracy: 0.4929
12512/25000 [==============>...............] - ETA: 46s - loss: 7.7683 - accuracy: 0.4934
12544/25000 [==============>...............] - ETA: 46s - loss: 7.7620 - accuracy: 0.4938
12576/25000 [==============>...............] - ETA: 46s - loss: 7.7581 - accuracy: 0.4940
12608/25000 [==============>...............] - ETA: 45s - loss: 7.7603 - accuracy: 0.4939
12640/25000 [==============>...............] - ETA: 45s - loss: 7.7588 - accuracy: 0.4940
12672/25000 [==============>...............] - ETA: 45s - loss: 7.7550 - accuracy: 0.4942
12704/25000 [==============>...............] - ETA: 45s - loss: 7.7583 - accuracy: 0.4940
12736/25000 [==============>...............] - ETA: 45s - loss: 7.7617 - accuracy: 0.4938
12768/25000 [==============>...............] - ETA: 45s - loss: 7.7639 - accuracy: 0.4937
12800/25000 [==============>...............] - ETA: 45s - loss: 7.7684 - accuracy: 0.4934
12832/25000 [==============>...............] - ETA: 45s - loss: 7.7730 - accuracy: 0.4931
12864/25000 [==============>...............] - ETA: 44s - loss: 7.7739 - accuracy: 0.4930
12896/25000 [==============>...............] - ETA: 44s - loss: 7.7724 - accuracy: 0.4931
12928/25000 [==============>...............] - ETA: 44s - loss: 7.7769 - accuracy: 0.4928
12960/25000 [==============>...............] - ETA: 44s - loss: 7.7802 - accuracy: 0.4926
12992/25000 [==============>...............] - ETA: 44s - loss: 7.7764 - accuracy: 0.4928
13024/25000 [==============>...............] - ETA: 44s - loss: 7.7738 - accuracy: 0.4930
13056/25000 [==============>...............] - ETA: 44s - loss: 7.7700 - accuracy: 0.4933
13088/25000 [==============>...............] - ETA: 44s - loss: 7.7650 - accuracy: 0.4936
13120/25000 [==============>...............] - ETA: 43s - loss: 7.7578 - accuracy: 0.4941
13152/25000 [==============>...............] - ETA: 43s - loss: 7.7599 - accuracy: 0.4939
13184/25000 [==============>...............] - ETA: 43s - loss: 7.7597 - accuracy: 0.4939
13216/25000 [==============>...............] - ETA: 43s - loss: 7.7618 - accuracy: 0.4938
13248/25000 [==============>...............] - ETA: 43s - loss: 7.7592 - accuracy: 0.4940
13280/25000 [==============>...............] - ETA: 43s - loss: 7.7590 - accuracy: 0.4940
13312/25000 [==============>...............] - ETA: 43s - loss: 7.7599 - accuracy: 0.4939
13344/25000 [===============>..............] - ETA: 43s - loss: 7.7574 - accuracy: 0.4941
13376/25000 [===============>..............] - ETA: 42s - loss: 7.7618 - accuracy: 0.4938
13408/25000 [===============>..............] - ETA: 42s - loss: 7.7615 - accuracy: 0.4938
13440/25000 [===============>..............] - ETA: 42s - loss: 7.7659 - accuracy: 0.4935
13472/25000 [===============>..............] - ETA: 42s - loss: 7.7645 - accuracy: 0.4936
13504/25000 [===============>..............] - ETA: 42s - loss: 7.7643 - accuracy: 0.4936
13536/25000 [===============>..............] - ETA: 42s - loss: 7.7663 - accuracy: 0.4935
13568/25000 [===============>..............] - ETA: 42s - loss: 7.7728 - accuracy: 0.4931
13600/25000 [===============>..............] - ETA: 42s - loss: 7.7703 - accuracy: 0.4932
13632/25000 [===============>..............] - ETA: 41s - loss: 7.7735 - accuracy: 0.4930
13664/25000 [===============>..............] - ETA: 41s - loss: 7.7721 - accuracy: 0.4931
13696/25000 [===============>..............] - ETA: 41s - loss: 7.7741 - accuracy: 0.4930
13728/25000 [===============>..............] - ETA: 41s - loss: 7.7716 - accuracy: 0.4932
13760/25000 [===============>..............] - ETA: 41s - loss: 7.7747 - accuracy: 0.4930
13792/25000 [===============>..............] - ETA: 41s - loss: 7.7722 - accuracy: 0.4931
13824/25000 [===============>..............] - ETA: 41s - loss: 7.7698 - accuracy: 0.4933
13856/25000 [===============>..............] - ETA: 41s - loss: 7.7740 - accuracy: 0.4930
13888/25000 [===============>..............] - ETA: 41s - loss: 7.7726 - accuracy: 0.4931
13920/25000 [===============>..............] - ETA: 40s - loss: 7.7680 - accuracy: 0.4934
13952/25000 [===============>..............] - ETA: 40s - loss: 7.7677 - accuracy: 0.4934
13984/25000 [===============>..............] - ETA: 40s - loss: 7.7730 - accuracy: 0.4931
14016/25000 [===============>..............] - ETA: 40s - loss: 7.7695 - accuracy: 0.4933
14048/25000 [===============>..............] - ETA: 40s - loss: 7.7714 - accuracy: 0.4932
14080/25000 [===============>..............] - ETA: 40s - loss: 7.7723 - accuracy: 0.4931
14112/25000 [===============>..............] - ETA: 40s - loss: 7.7709 - accuracy: 0.4932
14144/25000 [===============>..............] - ETA: 40s - loss: 7.7674 - accuracy: 0.4934
14176/25000 [================>.............] - ETA: 39s - loss: 7.7672 - accuracy: 0.4934
14208/25000 [================>.............] - ETA: 39s - loss: 7.7681 - accuracy: 0.4934
14240/25000 [================>.............] - ETA: 39s - loss: 7.7711 - accuracy: 0.4932
14272/25000 [================>.............] - ETA: 39s - loss: 7.7730 - accuracy: 0.4931
14304/25000 [================>.............] - ETA: 39s - loss: 7.7738 - accuracy: 0.4930
14336/25000 [================>.............] - ETA: 39s - loss: 7.7736 - accuracy: 0.4930
14368/25000 [================>.............] - ETA: 39s - loss: 7.7755 - accuracy: 0.4929
14400/25000 [================>.............] - ETA: 39s - loss: 7.7752 - accuracy: 0.4929
14432/25000 [================>.............] - ETA: 39s - loss: 7.7750 - accuracy: 0.4929
14464/25000 [================>.............] - ETA: 38s - loss: 7.7737 - accuracy: 0.4930
14496/25000 [================>.............] - ETA: 38s - loss: 7.7703 - accuracy: 0.4932
14528/25000 [================>.............] - ETA: 38s - loss: 7.7732 - accuracy: 0.4930
14560/25000 [================>.............] - ETA: 38s - loss: 7.7677 - accuracy: 0.4934
14592/25000 [================>.............] - ETA: 38s - loss: 7.7654 - accuracy: 0.4936
14624/25000 [================>.............] - ETA: 38s - loss: 7.7652 - accuracy: 0.4936
14656/25000 [================>.............] - ETA: 38s - loss: 7.7618 - accuracy: 0.4938
14688/25000 [================>.............] - ETA: 38s - loss: 7.7543 - accuracy: 0.4943
14720/25000 [================>.............] - ETA: 37s - loss: 7.7541 - accuracy: 0.4943
14752/25000 [================>.............] - ETA: 37s - loss: 7.7508 - accuracy: 0.4945
14784/25000 [================>.............] - ETA: 37s - loss: 7.7486 - accuracy: 0.4947
14816/25000 [================>.............] - ETA: 37s - loss: 7.7504 - accuracy: 0.4945
14848/25000 [================>.............] - ETA: 37s - loss: 7.7492 - accuracy: 0.4946
14880/25000 [================>.............] - ETA: 37s - loss: 7.7491 - accuracy: 0.4946
14912/25000 [================>.............] - ETA: 37s - loss: 7.7520 - accuracy: 0.4944
14944/25000 [================>.............] - ETA: 37s - loss: 7.7528 - accuracy: 0.4944
14976/25000 [================>.............] - ETA: 36s - loss: 7.7506 - accuracy: 0.4945
15008/25000 [=================>............] - ETA: 36s - loss: 7.7504 - accuracy: 0.4945
15040/25000 [=================>............] - ETA: 36s - loss: 7.7523 - accuracy: 0.4944
15072/25000 [=================>............] - ETA: 36s - loss: 7.7531 - accuracy: 0.4944
15104/25000 [=================>............] - ETA: 36s - loss: 7.7448 - accuracy: 0.4949
15136/25000 [=================>............] - ETA: 36s - loss: 7.7446 - accuracy: 0.4949
15168/25000 [=================>............] - ETA: 36s - loss: 7.7424 - accuracy: 0.4951
15200/25000 [=================>............] - ETA: 36s - loss: 7.7413 - accuracy: 0.4951
15232/25000 [=================>............] - ETA: 36s - loss: 7.7421 - accuracy: 0.4951
15264/25000 [=================>............] - ETA: 35s - loss: 7.7410 - accuracy: 0.4952
15296/25000 [=================>............] - ETA: 35s - loss: 7.7368 - accuracy: 0.4954
15328/25000 [=================>............] - ETA: 35s - loss: 7.7336 - accuracy: 0.4956
15360/25000 [=================>............] - ETA: 35s - loss: 7.7325 - accuracy: 0.4957
15392/25000 [=================>............] - ETA: 35s - loss: 7.7294 - accuracy: 0.4959
15424/25000 [=================>............] - ETA: 35s - loss: 7.7273 - accuracy: 0.4960
15456/25000 [=================>............] - ETA: 35s - loss: 7.7301 - accuracy: 0.4959
15488/25000 [=================>............] - ETA: 35s - loss: 7.7320 - accuracy: 0.4957
15520/25000 [=================>............] - ETA: 34s - loss: 7.7348 - accuracy: 0.4956
15552/25000 [=================>............] - ETA: 34s - loss: 7.7386 - accuracy: 0.4953
15584/25000 [=================>............] - ETA: 34s - loss: 7.7384 - accuracy: 0.4953
15616/25000 [=================>............] - ETA: 34s - loss: 7.7363 - accuracy: 0.4955
15648/25000 [=================>............] - ETA: 34s - loss: 7.7421 - accuracy: 0.4951
15680/25000 [=================>............] - ETA: 34s - loss: 7.7439 - accuracy: 0.4950
15712/25000 [=================>............] - ETA: 34s - loss: 7.7418 - accuracy: 0.4951
15744/25000 [=================>............] - ETA: 34s - loss: 7.7445 - accuracy: 0.4949
15776/25000 [=================>............] - ETA: 34s - loss: 7.7434 - accuracy: 0.4950
15808/25000 [=================>............] - ETA: 33s - loss: 7.7423 - accuracy: 0.4951
15840/25000 [==================>...........] - ETA: 33s - loss: 7.7402 - accuracy: 0.4952
15872/25000 [==================>...........] - ETA: 33s - loss: 7.7391 - accuracy: 0.4953
15904/25000 [==================>...........] - ETA: 33s - loss: 7.7380 - accuracy: 0.4953
15936/25000 [==================>...........] - ETA: 33s - loss: 7.7378 - accuracy: 0.4954
15968/25000 [==================>...........] - ETA: 33s - loss: 7.7386 - accuracy: 0.4953
16000/25000 [==================>...........] - ETA: 33s - loss: 7.7414 - accuracy: 0.4951
16032/25000 [==================>...........] - ETA: 33s - loss: 7.7412 - accuracy: 0.4951
16064/25000 [==================>...........] - ETA: 32s - loss: 7.7411 - accuracy: 0.4951
16096/25000 [==================>...........] - ETA: 32s - loss: 7.7390 - accuracy: 0.4953
16128/25000 [==================>...........] - ETA: 32s - loss: 7.7417 - accuracy: 0.4951
16160/25000 [==================>...........] - ETA: 32s - loss: 7.7416 - accuracy: 0.4951
16192/25000 [==================>...........] - ETA: 32s - loss: 7.7471 - accuracy: 0.4948
16224/25000 [==================>...........] - ETA: 32s - loss: 7.7460 - accuracy: 0.4948
16256/25000 [==================>...........] - ETA: 32s - loss: 7.7459 - accuracy: 0.4948
16288/25000 [==================>...........] - ETA: 32s - loss: 7.7391 - accuracy: 0.4953
16320/25000 [==================>...........] - ETA: 32s - loss: 7.7399 - accuracy: 0.4952
16352/25000 [==================>...........] - ETA: 31s - loss: 7.7379 - accuracy: 0.4954
16384/25000 [==================>...........] - ETA: 31s - loss: 7.7368 - accuracy: 0.4954
16416/25000 [==================>...........] - ETA: 31s - loss: 7.7376 - accuracy: 0.4954
16448/25000 [==================>...........] - ETA: 31s - loss: 7.7375 - accuracy: 0.4954
16480/25000 [==================>...........] - ETA: 31s - loss: 7.7364 - accuracy: 0.4954
16512/25000 [==================>...........] - ETA: 31s - loss: 7.7353 - accuracy: 0.4955
16544/25000 [==================>...........] - ETA: 31s - loss: 7.7334 - accuracy: 0.4956
16576/25000 [==================>...........] - ETA: 31s - loss: 7.7314 - accuracy: 0.4958
16608/25000 [==================>...........] - ETA: 30s - loss: 7.7340 - accuracy: 0.4956
16640/25000 [==================>...........] - ETA: 30s - loss: 7.7357 - accuracy: 0.4955
16672/25000 [===================>..........] - ETA: 30s - loss: 7.7393 - accuracy: 0.4953
16704/25000 [===================>..........] - ETA: 30s - loss: 7.7364 - accuracy: 0.4955
16736/25000 [===================>..........] - ETA: 30s - loss: 7.7381 - accuracy: 0.4953
16768/25000 [===================>..........] - ETA: 30s - loss: 7.7398 - accuracy: 0.4952
16800/25000 [===================>..........] - ETA: 30s - loss: 7.7342 - accuracy: 0.4956
16832/25000 [===================>..........] - ETA: 30s - loss: 7.7331 - accuracy: 0.4957
16864/25000 [===================>..........] - ETA: 30s - loss: 7.7330 - accuracy: 0.4957
16896/25000 [===================>..........] - ETA: 29s - loss: 7.7329 - accuracy: 0.4957
16928/25000 [===================>..........] - ETA: 29s - loss: 7.7346 - accuracy: 0.4956
16960/25000 [===================>..........] - ETA: 29s - loss: 7.7308 - accuracy: 0.4958
16992/25000 [===================>..........] - ETA: 29s - loss: 7.7289 - accuracy: 0.4959
17024/25000 [===================>..........] - ETA: 29s - loss: 7.7261 - accuracy: 0.4961
17056/25000 [===================>..........] - ETA: 29s - loss: 7.7260 - accuracy: 0.4961
17088/25000 [===================>..........] - ETA: 29s - loss: 7.7232 - accuracy: 0.4963
17120/25000 [===================>..........] - ETA: 29s - loss: 7.7186 - accuracy: 0.4966
17152/25000 [===================>..........] - ETA: 28s - loss: 7.7185 - accuracy: 0.4966
17184/25000 [===================>..........] - ETA: 28s - loss: 7.7175 - accuracy: 0.4967
17216/25000 [===================>..........] - ETA: 28s - loss: 7.7183 - accuracy: 0.4966
17248/25000 [===================>..........] - ETA: 28s - loss: 7.7191 - accuracy: 0.4966
17280/25000 [===================>..........] - ETA: 28s - loss: 7.7234 - accuracy: 0.4963
17312/25000 [===================>..........] - ETA: 28s - loss: 7.7206 - accuracy: 0.4965
17344/25000 [===================>..........] - ETA: 28s - loss: 7.7214 - accuracy: 0.4964
17376/25000 [===================>..........] - ETA: 28s - loss: 7.7231 - accuracy: 0.4963
17408/25000 [===================>..........] - ETA: 27s - loss: 7.7256 - accuracy: 0.4962
17440/25000 [===================>..........] - ETA: 27s - loss: 7.7211 - accuracy: 0.4964
17472/25000 [===================>..........] - ETA: 27s - loss: 7.7193 - accuracy: 0.4966
17504/25000 [====================>.........] - ETA: 27s - loss: 7.7183 - accuracy: 0.4966
17536/25000 [====================>.........] - ETA: 27s - loss: 7.7173 - accuracy: 0.4967
17568/25000 [====================>.........] - ETA: 27s - loss: 7.7199 - accuracy: 0.4965
17600/25000 [====================>.........] - ETA: 27s - loss: 7.7206 - accuracy: 0.4965
17632/25000 [====================>.........] - ETA: 27s - loss: 7.7179 - accuracy: 0.4967
17664/25000 [====================>.........] - ETA: 27s - loss: 7.7170 - accuracy: 0.4967
17696/25000 [====================>.........] - ETA: 26s - loss: 7.7160 - accuracy: 0.4968
17728/25000 [====================>.........] - ETA: 26s - loss: 7.7185 - accuracy: 0.4966
17760/25000 [====================>.........] - ETA: 26s - loss: 7.7158 - accuracy: 0.4968
17792/25000 [====================>.........] - ETA: 26s - loss: 7.7209 - accuracy: 0.4965
17824/25000 [====================>.........] - ETA: 26s - loss: 7.7191 - accuracy: 0.4966
17856/25000 [====================>.........] - ETA: 26s - loss: 7.7181 - accuracy: 0.4966
17888/25000 [====================>.........] - ETA: 26s - loss: 7.7155 - accuracy: 0.4968
17920/25000 [====================>.........] - ETA: 26s - loss: 7.7145 - accuracy: 0.4969
17952/25000 [====================>.........] - ETA: 25s - loss: 7.7119 - accuracy: 0.4970
17984/25000 [====================>.........] - ETA: 25s - loss: 7.7127 - accuracy: 0.4970
18016/25000 [====================>.........] - ETA: 25s - loss: 7.7109 - accuracy: 0.4971
18048/25000 [====================>.........] - ETA: 25s - loss: 7.7091 - accuracy: 0.4972
18080/25000 [====================>.........] - ETA: 25s - loss: 7.7090 - accuracy: 0.4972
18112/25000 [====================>.........] - ETA: 25s - loss: 7.7115 - accuracy: 0.4971
18144/25000 [====================>.........] - ETA: 25s - loss: 7.7089 - accuracy: 0.4972
18176/25000 [====================>.........] - ETA: 25s - loss: 7.7122 - accuracy: 0.4970
18208/25000 [====================>.........] - ETA: 25s - loss: 7.7104 - accuracy: 0.4971
18240/25000 [====================>.........] - ETA: 24s - loss: 7.7087 - accuracy: 0.4973
18272/25000 [====================>.........] - ETA: 24s - loss: 7.7094 - accuracy: 0.4972
18304/25000 [====================>.........] - ETA: 24s - loss: 7.7060 - accuracy: 0.4974
18336/25000 [=====================>........] - ETA: 24s - loss: 7.7034 - accuracy: 0.4976
18368/25000 [=====================>........] - ETA: 24s - loss: 7.7042 - accuracy: 0.4976
18400/25000 [=====================>........] - ETA: 24s - loss: 7.7041 - accuracy: 0.4976
18432/25000 [=====================>........] - ETA: 24s - loss: 7.7016 - accuracy: 0.4977
18464/25000 [=====================>........] - ETA: 24s - loss: 7.7057 - accuracy: 0.4975
18496/25000 [=====================>........] - ETA: 23s - loss: 7.7114 - accuracy: 0.4971
18528/25000 [=====================>........] - ETA: 23s - loss: 7.7105 - accuracy: 0.4971
18560/25000 [=====================>........] - ETA: 23s - loss: 7.7104 - accuracy: 0.4971
18592/25000 [=====================>........] - ETA: 23s - loss: 7.7112 - accuracy: 0.4971
18624/25000 [=====================>........] - ETA: 23s - loss: 7.7135 - accuracy: 0.4969
18656/25000 [=====================>........] - ETA: 23s - loss: 7.7135 - accuracy: 0.4969
18688/25000 [=====================>........] - ETA: 23s - loss: 7.7117 - accuracy: 0.4971
18720/25000 [=====================>........] - ETA: 23s - loss: 7.7117 - accuracy: 0.4971
18752/25000 [=====================>........] - ETA: 22s - loss: 7.7140 - accuracy: 0.4969
18784/25000 [=====================>........] - ETA: 22s - loss: 7.7123 - accuracy: 0.4970
18816/25000 [=====================>........] - ETA: 22s - loss: 7.7106 - accuracy: 0.4971
18848/25000 [=====================>........] - ETA: 22s - loss: 7.7105 - accuracy: 0.4971
18880/25000 [=====================>........] - ETA: 22s - loss: 7.7137 - accuracy: 0.4969
18912/25000 [=====================>........] - ETA: 22s - loss: 7.7080 - accuracy: 0.4973
18944/25000 [=====================>........] - ETA: 22s - loss: 7.7087 - accuracy: 0.4973
18976/25000 [=====================>........] - ETA: 22s - loss: 7.7103 - accuracy: 0.4972
19008/25000 [=====================>........] - ETA: 22s - loss: 7.7118 - accuracy: 0.4971
19040/25000 [=====================>........] - ETA: 21s - loss: 7.7117 - accuracy: 0.4971
19072/25000 [=====================>........] - ETA: 21s - loss: 7.7108 - accuracy: 0.4971
19104/25000 [=====================>........] - ETA: 21s - loss: 7.7140 - accuracy: 0.4969
19136/25000 [=====================>........] - ETA: 21s - loss: 7.7131 - accuracy: 0.4970
19168/25000 [======================>.......] - ETA: 21s - loss: 7.7178 - accuracy: 0.4967
19200/25000 [======================>.......] - ETA: 21s - loss: 7.7161 - accuracy: 0.4968
19232/25000 [======================>.......] - ETA: 21s - loss: 7.7161 - accuracy: 0.4968
19264/25000 [======================>.......] - ETA: 21s - loss: 7.7128 - accuracy: 0.4970
19296/25000 [======================>.......] - ETA: 20s - loss: 7.7095 - accuracy: 0.4972
19328/25000 [======================>.......] - ETA: 20s - loss: 7.7087 - accuracy: 0.4973
19360/25000 [======================>.......] - ETA: 20s - loss: 7.7078 - accuracy: 0.4973
19392/25000 [======================>.......] - ETA: 20s - loss: 7.7038 - accuracy: 0.4976
19424/25000 [======================>.......] - ETA: 20s - loss: 7.7021 - accuracy: 0.4977
19456/25000 [======================>.......] - ETA: 20s - loss: 7.6966 - accuracy: 0.4980
19488/25000 [======================>.......] - ETA: 20s - loss: 7.6949 - accuracy: 0.4982
19520/25000 [======================>.......] - ETA: 20s - loss: 7.6949 - accuracy: 0.4982
19552/25000 [======================>.......] - ETA: 20s - loss: 7.6917 - accuracy: 0.4984
19584/25000 [======================>.......] - ETA: 19s - loss: 7.6932 - accuracy: 0.4983
19616/25000 [======================>.......] - ETA: 19s - loss: 7.6932 - accuracy: 0.4983
19648/25000 [======================>.......] - ETA: 19s - loss: 7.6893 - accuracy: 0.4985
19680/25000 [======================>.......] - ETA: 19s - loss: 7.6923 - accuracy: 0.4983
19712/25000 [======================>.......] - ETA: 19s - loss: 7.6923 - accuracy: 0.4983
19744/25000 [======================>.......] - ETA: 19s - loss: 7.6946 - accuracy: 0.4982
19776/25000 [======================>.......] - ETA: 19s - loss: 7.6976 - accuracy: 0.4980
19808/25000 [======================>.......] - ETA: 19s - loss: 7.6968 - accuracy: 0.4980
19840/25000 [======================>.......] - ETA: 18s - loss: 7.6952 - accuracy: 0.4981
19872/25000 [======================>.......] - ETA: 18s - loss: 7.6975 - accuracy: 0.4980
19904/25000 [======================>.......] - ETA: 18s - loss: 7.6959 - accuracy: 0.4981
19936/25000 [======================>.......] - ETA: 18s - loss: 7.6974 - accuracy: 0.4980
19968/25000 [======================>.......] - ETA: 18s - loss: 7.7004 - accuracy: 0.4978
20000/25000 [=======================>......] - ETA: 18s - loss: 7.6996 - accuracy: 0.4979
20032/25000 [=======================>......] - ETA: 18s - loss: 7.6957 - accuracy: 0.4981
20064/25000 [=======================>......] - ETA: 18s - loss: 7.6926 - accuracy: 0.4983
20096/25000 [=======================>......] - ETA: 18s - loss: 7.6941 - accuracy: 0.4982
20128/25000 [=======================>......] - ETA: 17s - loss: 7.6963 - accuracy: 0.4981
20160/25000 [=======================>......] - ETA: 17s - loss: 7.6955 - accuracy: 0.4981
20192/25000 [=======================>......] - ETA: 17s - loss: 7.6955 - accuracy: 0.4981
20224/25000 [=======================>......] - ETA: 17s - loss: 7.6962 - accuracy: 0.4981
20256/25000 [=======================>......] - ETA: 17s - loss: 7.6954 - accuracy: 0.4981
20288/25000 [=======================>......] - ETA: 17s - loss: 7.6931 - accuracy: 0.4983
20320/25000 [=======================>......] - ETA: 17s - loss: 7.6938 - accuracy: 0.4982
20352/25000 [=======================>......] - ETA: 17s - loss: 7.6922 - accuracy: 0.4983
20384/25000 [=======================>......] - ETA: 16s - loss: 7.6899 - accuracy: 0.4985
20416/25000 [=======================>......] - ETA: 16s - loss: 7.6914 - accuracy: 0.4984
20448/25000 [=======================>......] - ETA: 16s - loss: 7.6921 - accuracy: 0.4983
20480/25000 [=======================>......] - ETA: 16s - loss: 7.6913 - accuracy: 0.4984
20512/25000 [=======================>......] - ETA: 16s - loss: 7.6890 - accuracy: 0.4985
20544/25000 [=======================>......] - ETA: 16s - loss: 7.6890 - accuracy: 0.4985
20576/25000 [=======================>......] - ETA: 16s - loss: 7.6882 - accuracy: 0.4986
20608/25000 [=======================>......] - ETA: 16s - loss: 7.6882 - accuracy: 0.4986
20640/25000 [=======================>......] - ETA: 16s - loss: 7.6844 - accuracy: 0.4988
20672/25000 [=======================>......] - ETA: 15s - loss: 7.6837 - accuracy: 0.4989
20704/25000 [=======================>......] - ETA: 15s - loss: 7.6851 - accuracy: 0.4988
20736/25000 [=======================>......] - ETA: 15s - loss: 7.6836 - accuracy: 0.4989
20768/25000 [=======================>......] - ETA: 15s - loss: 7.6836 - accuracy: 0.4989
20800/25000 [=======================>......] - ETA: 15s - loss: 7.6836 - accuracy: 0.4989
20832/25000 [=======================>......] - ETA: 15s - loss: 7.6828 - accuracy: 0.4989
20864/25000 [========================>.....] - ETA: 15s - loss: 7.6828 - accuracy: 0.4989
20896/25000 [========================>.....] - ETA: 15s - loss: 7.6798 - accuracy: 0.4991
20928/25000 [========================>.....] - ETA: 14s - loss: 7.6798 - accuracy: 0.4991
20960/25000 [========================>.....] - ETA: 14s - loss: 7.6805 - accuracy: 0.4991
20992/25000 [========================>.....] - ETA: 14s - loss: 7.6761 - accuracy: 0.4994
21024/25000 [========================>.....] - ETA: 14s - loss: 7.6768 - accuracy: 0.4993
21056/25000 [========================>.....] - ETA: 14s - loss: 7.6819 - accuracy: 0.4990
21088/25000 [========================>.....] - ETA: 14s - loss: 7.6833 - accuracy: 0.4989
21120/25000 [========================>.....] - ETA: 14s - loss: 7.6848 - accuracy: 0.4988
21152/25000 [========================>.....] - ETA: 14s - loss: 7.6826 - accuracy: 0.4990
21184/25000 [========================>.....] - ETA: 13s - loss: 7.6789 - accuracy: 0.4992
21216/25000 [========================>.....] - ETA: 13s - loss: 7.6782 - accuracy: 0.4992
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6774 - accuracy: 0.4993
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6767 - accuracy: 0.4993
21312/25000 [========================>.....] - ETA: 13s - loss: 7.6760 - accuracy: 0.4994
21344/25000 [========================>.....] - ETA: 13s - loss: 7.6803 - accuracy: 0.4991
21376/25000 [========================>.....] - ETA: 13s - loss: 7.6824 - accuracy: 0.4990
21408/25000 [========================>.....] - ETA: 13s - loss: 7.6824 - accuracy: 0.4990
21440/25000 [========================>.....] - ETA: 13s - loss: 7.6802 - accuracy: 0.4991
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6845 - accuracy: 0.4988
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6809 - accuracy: 0.4991
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6787 - accuracy: 0.4992
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6780 - accuracy: 0.4993
21600/25000 [========================>.....] - ETA: 12s - loss: 7.6751 - accuracy: 0.4994
21632/25000 [========================>.....] - ETA: 12s - loss: 7.6730 - accuracy: 0.4996
21664/25000 [========================>.....] - ETA: 12s - loss: 7.6730 - accuracy: 0.4996
21696/25000 [=========================>....] - ETA: 12s - loss: 7.6765 - accuracy: 0.4994
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6772 - accuracy: 0.4993
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6765 - accuracy: 0.4994
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6765 - accuracy: 0.4994
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6772 - accuracy: 0.4993
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6757 - accuracy: 0.4994
21888/25000 [=========================>....] - ETA: 11s - loss: 7.6736 - accuracy: 0.4995
21920/25000 [=========================>....] - ETA: 11s - loss: 7.6750 - accuracy: 0.4995
21952/25000 [=========================>....] - ETA: 11s - loss: 7.6729 - accuracy: 0.4996
21984/25000 [=========================>....] - ETA: 11s - loss: 7.6715 - accuracy: 0.4997
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6694 - accuracy: 0.4998
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6701 - accuracy: 0.4998
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6652 - accuracy: 0.5001
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6673 - accuracy: 0.5000
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6694 - accuracy: 0.4998
22176/25000 [=========================>....] - ETA: 10s - loss: 7.6701 - accuracy: 0.4998
22208/25000 [=========================>....] - ETA: 10s - loss: 7.6687 - accuracy: 0.4999
22240/25000 [=========================>....] - ETA: 10s - loss: 7.6708 - accuracy: 0.4997
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6701 - accuracy: 0.4998 
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6714 - accuracy: 0.4997
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6728 - accuracy: 0.4996
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6687 - accuracy: 0.4999
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6694 - accuracy: 0.4998
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6680 - accuracy: 0.4999
22464/25000 [=========================>....] - ETA: 9s - loss: 7.6639 - accuracy: 0.5002
22496/25000 [=========================>....] - ETA: 9s - loss: 7.6646 - accuracy: 0.5001
22528/25000 [==========================>...] - ETA: 9s - loss: 7.6666 - accuracy: 0.5000
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6673 - accuracy: 0.5000
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6687 - accuracy: 0.4999
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6680 - accuracy: 0.4999
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6632 - accuracy: 0.5002
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6653 - accuracy: 0.5001
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6619 - accuracy: 0.5003
22752/25000 [==========================>...] - ETA: 8s - loss: 7.6632 - accuracy: 0.5002
22784/25000 [==========================>...] - ETA: 8s - loss: 7.6659 - accuracy: 0.5000
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6680 - accuracy: 0.4999
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6673 - accuracy: 0.5000
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6680 - accuracy: 0.4999
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6673 - accuracy: 0.5000
23008/25000 [==========================>...] - ETA: 7s - loss: 7.6673 - accuracy: 0.5000
23040/25000 [==========================>...] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
23072/25000 [==========================>...] - ETA: 7s - loss: 7.6679 - accuracy: 0.4999
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6673 - accuracy: 0.5000
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6679 - accuracy: 0.4999
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6653 - accuracy: 0.5001
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6633 - accuracy: 0.5002
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6613 - accuracy: 0.5003
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6607 - accuracy: 0.5004
23296/25000 [==========================>...] - ETA: 6s - loss: 7.6594 - accuracy: 0.5005
23328/25000 [==========================>...] - ETA: 6s - loss: 7.6587 - accuracy: 0.5005
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6601 - accuracy: 0.5004
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6607 - accuracy: 0.5004
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6640 - accuracy: 0.5002
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6634 - accuracy: 0.5002
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6620 - accuracy: 0.5003
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6647 - accuracy: 0.5001
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6660 - accuracy: 0.5000
23584/25000 [===========================>..] - ETA: 5s - loss: 7.6647 - accuracy: 0.5001
23616/25000 [===========================>..] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6640 - accuracy: 0.5002
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6660 - accuracy: 0.5000
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6634 - accuracy: 0.5002
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6608 - accuracy: 0.5004
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6621 - accuracy: 0.5003
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6595 - accuracy: 0.5005
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6608 - accuracy: 0.5004
23872/25000 [===========================>..] - ETA: 4s - loss: 7.6615 - accuracy: 0.5003
23904/25000 [===========================>..] - ETA: 4s - loss: 7.6621 - accuracy: 0.5003
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6602 - accuracy: 0.5004
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6641 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6628 - accuracy: 0.5002
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6622 - accuracy: 0.5003
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6634 - accuracy: 0.5002
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6628 - accuracy: 0.5002
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6603 - accuracy: 0.5004
24160/25000 [===========================>..] - ETA: 3s - loss: 7.6603 - accuracy: 0.5004
24192/25000 [============================>.] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
24224/25000 [============================>.] - ETA: 2s - loss: 7.6609 - accuracy: 0.5004
24256/25000 [============================>.] - ETA: 2s - loss: 7.6628 - accuracy: 0.5002
24288/25000 [============================>.] - ETA: 2s - loss: 7.6635 - accuracy: 0.5002
24320/25000 [============================>.] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24352/25000 [============================>.] - ETA: 2s - loss: 7.6635 - accuracy: 0.5002
24384/25000 [============================>.] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24416/25000 [============================>.] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
24448/25000 [============================>.] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24480/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24512/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24544/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24576/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24608/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24640/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24672/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24704/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24736/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24768/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24800/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24832/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24864/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24896/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24928/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24960/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
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
