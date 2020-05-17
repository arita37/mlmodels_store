
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
 40%|████      | 2/5 [00:52<01:18, 26.19s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.1162761560728123, 'embedding_size_factor': 0.6276217573563909, 'layers.choice': 1, 'learning_rate': 0.007149749758951748, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 0.0005379072901484656} and reward: 0.3568
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xbd\xc4F/\xa3'rX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe4\x15z9CQ\xe4X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?}I\x0eV:\x7fcX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?A\xa0K\x1c\x17\x84\x06u." and reward: 0.3568
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xbd\xc4F/\xa3'rX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe4\x15z9CQ\xe4X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?}I\x0eV:\x7fcX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?A\xa0K\x1c\x17\x84\x06u." and reward: 0.3568
 60%|██████    | 3/5 [02:23<01:31, 45.63s/it] 60%|██████    | 3/5 [02:23<01:35, 47.79s/it]
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.4905186484018518, 'embedding_size_factor': 1.4046927325234364, 'layers.choice': 3, 'learning_rate': 0.000761426163409334, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.1228188993142293e-11} and reward: 0.3694
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdfd\xa8T=\xb3lX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6y\x9f\x161\xdf`X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?H\xf3N<.CbX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xa8\xb0\xe8\x91\x89\x15\x02u.' and reward: 0.3694
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdfd\xa8T=\xb3lX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6y\x9f\x161\xdf`X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?H\xf3N<.CbX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xa8\xb0\xe8\x91\x89\x15\x02u.' and reward: 0.3694
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 198.48247027397156
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -81.16s of remaining time.
Ensemble size: 8
Ensemble weights: 
[0.625 0.25  0.125]
	0.3898	 = Validation accuracy score
	1.09s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 202.29s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f985d9eeac8> 

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
 [ 0.13143729 -0.00169529  0.05231509 -0.03051235  0.00699615  0.03079724]
 [-0.03918    -0.02117461  0.03896988 -0.02182996 -0.0029582   0.01422972]
 [ 0.20566837  0.07153352  0.22094499 -0.29042473  0.17067121  0.19025095]
 [ 0.24689506 -0.12899917 -0.17262812 -0.06073889  0.21150377  0.03192602]
 [-0.18549624 -0.01023852  0.19095573  0.11275226  0.00220166  0.23564735]
 [ 0.49169335  0.0124786   0.20282753  0.34061885  0.30318382 -0.28997335]
 [ 0.45882663  0.74235284 -0.35467061  0.47602028  0.07027112  0.03214914]
 [ 0.06400099 -0.29569343 -0.20247696  0.06971137  0.07050303 -0.10340977]
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
{'loss': 0.48379580304026604, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-17 05:18:00.481213: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.5097374357283115, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-17 05:18:01.661159: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 1695744/17464789 [=>............................] - ETA: 0s
 7200768/17464789 [===========>..................] - ETA: 0s
12804096/17464789 [====================>.........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-17 05:18:13.519751: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-17 05:18:13.524754: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095230000 Hz
2020-05-17 05:18:13.524930: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558a82d5e9f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-17 05:18:13.524945: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:42 - loss: 7.1875 - accuracy: 0.5312
   64/25000 [..............................] - ETA: 2:51 - loss: 7.6666 - accuracy: 0.5000
   96/25000 [..............................] - ETA: 2:13 - loss: 7.1875 - accuracy: 0.5312
  128/25000 [..............................] - ETA: 1:56 - loss: 7.3072 - accuracy: 0.5234
  160/25000 [..............................] - ETA: 1:46 - loss: 7.9541 - accuracy: 0.4812
  192/25000 [..............................] - ETA: 1:38 - loss: 7.5868 - accuracy: 0.5052
  224/25000 [..............................] - ETA: 1:32 - loss: 7.7351 - accuracy: 0.4955
  256/25000 [..............................] - ETA: 1:29 - loss: 7.6067 - accuracy: 0.5039
  288/25000 [..............................] - ETA: 1:26 - loss: 7.6666 - accuracy: 0.5000
  320/25000 [..............................] - ETA: 1:24 - loss: 7.6666 - accuracy: 0.5000
  352/25000 [..............................] - ETA: 1:22 - loss: 7.7537 - accuracy: 0.4943
  384/25000 [..............................] - ETA: 1:20 - loss: 7.7864 - accuracy: 0.4922
  416/25000 [..............................] - ETA: 1:19 - loss: 7.7772 - accuracy: 0.4928
  448/25000 [..............................] - ETA: 1:18 - loss: 7.5982 - accuracy: 0.5045
  480/25000 [..............................] - ETA: 1:17 - loss: 7.6027 - accuracy: 0.5042
  512/25000 [..............................] - ETA: 1:16 - loss: 7.5468 - accuracy: 0.5078
  544/25000 [..............................] - ETA: 1:15 - loss: 7.4411 - accuracy: 0.5147
  576/25000 [..............................] - ETA: 1:15 - loss: 7.4004 - accuracy: 0.5174
  608/25000 [..............................] - ETA: 1:14 - loss: 7.4901 - accuracy: 0.5115
  640/25000 [..............................] - ETA: 1:13 - loss: 7.4270 - accuracy: 0.5156
  672/25000 [..............................] - ETA: 1:13 - loss: 7.4156 - accuracy: 0.5164
  704/25000 [..............................] - ETA: 1:12 - loss: 7.4053 - accuracy: 0.5170
  736/25000 [..............................] - ETA: 1:12 - loss: 7.4791 - accuracy: 0.5122
  768/25000 [..............................] - ETA: 1:11 - loss: 7.4470 - accuracy: 0.5143
  800/25000 [..............................] - ETA: 1:11 - loss: 7.4941 - accuracy: 0.5113
  832/25000 [..............................] - ETA: 1:11 - loss: 7.5376 - accuracy: 0.5084
  864/25000 [>.............................] - ETA: 1:10 - loss: 7.5246 - accuracy: 0.5093
  896/25000 [>.............................] - ETA: 1:10 - loss: 7.4784 - accuracy: 0.5123
  928/25000 [>.............................] - ETA: 1:10 - loss: 7.4683 - accuracy: 0.5129
  960/25000 [>.............................] - ETA: 1:10 - loss: 7.4430 - accuracy: 0.5146
  992/25000 [>.............................] - ETA: 1:09 - loss: 7.4348 - accuracy: 0.5151
 1024/25000 [>.............................] - ETA: 1:09 - loss: 7.4570 - accuracy: 0.5137
 1056/25000 [>.............................] - ETA: 1:09 - loss: 7.4343 - accuracy: 0.5152
 1088/25000 [>.............................] - ETA: 1:08 - loss: 7.4411 - accuracy: 0.5147
 1120/25000 [>.............................] - ETA: 1:08 - loss: 7.4476 - accuracy: 0.5143
 1152/25000 [>.............................] - ETA: 1:08 - loss: 7.3605 - accuracy: 0.5200
 1184/25000 [>.............................] - ETA: 1:08 - loss: 7.3558 - accuracy: 0.5203
 1216/25000 [>.............................] - ETA: 1:08 - loss: 7.3388 - accuracy: 0.5214
 1248/25000 [>.............................] - ETA: 1:07 - loss: 7.3226 - accuracy: 0.5224
 1280/25000 [>.............................] - ETA: 1:07 - loss: 7.3552 - accuracy: 0.5203
 1312/25000 [>.............................] - ETA: 1:07 - loss: 7.3978 - accuracy: 0.5175
 1344/25000 [>.............................] - ETA: 1:07 - loss: 7.3700 - accuracy: 0.5193
 1376/25000 [>.............................] - ETA: 1:06 - loss: 7.3657 - accuracy: 0.5196
 1408/25000 [>.............................] - ETA: 1:06 - loss: 7.3726 - accuracy: 0.5192
 1440/25000 [>.............................] - ETA: 1:06 - loss: 7.4217 - accuracy: 0.5160
 1472/25000 [>.............................] - ETA: 1:06 - loss: 7.3854 - accuracy: 0.5183
 1504/25000 [>.............................] - ETA: 1:06 - loss: 7.4016 - accuracy: 0.5173
 1536/25000 [>.............................] - ETA: 1:05 - loss: 7.4270 - accuracy: 0.5156
 1568/25000 [>.............................] - ETA: 1:05 - loss: 7.4124 - accuracy: 0.5166
 1600/25000 [>.............................] - ETA: 1:05 - loss: 7.4079 - accuracy: 0.5169
 1632/25000 [>.............................] - ETA: 1:05 - loss: 7.3848 - accuracy: 0.5184
 1664/25000 [>.............................] - ETA: 1:05 - loss: 7.3994 - accuracy: 0.5174
 1696/25000 [=>............................] - ETA: 1:05 - loss: 7.3954 - accuracy: 0.5177
 1728/25000 [=>............................] - ETA: 1:05 - loss: 7.3915 - accuracy: 0.5179
 1760/25000 [=>............................] - ETA: 1:04 - loss: 7.3965 - accuracy: 0.5176
 1792/25000 [=>............................] - ETA: 1:04 - loss: 7.3843 - accuracy: 0.5184
 1824/25000 [=>............................] - ETA: 1:04 - loss: 7.4060 - accuracy: 0.5170
 1856/25000 [=>............................] - ETA: 1:04 - loss: 7.3857 - accuracy: 0.5183
 1888/25000 [=>............................] - ETA: 1:04 - loss: 7.3661 - accuracy: 0.5196
 1920/25000 [=>............................] - ETA: 1:03 - loss: 7.3791 - accuracy: 0.5188
 1952/25000 [=>............................] - ETA: 1:03 - loss: 7.3917 - accuracy: 0.5179
 1984/25000 [=>............................] - ETA: 1:03 - loss: 7.4348 - accuracy: 0.5151
 2016/25000 [=>............................] - ETA: 1:03 - loss: 7.4613 - accuracy: 0.5134
 2048/25000 [=>............................] - ETA: 1:03 - loss: 7.4495 - accuracy: 0.5142
 2080/25000 [=>............................] - ETA: 1:03 - loss: 7.4602 - accuracy: 0.5135
 2112/25000 [=>............................] - ETA: 1:02 - loss: 7.4633 - accuracy: 0.5133
 2144/25000 [=>............................] - ETA: 1:02 - loss: 7.4664 - accuracy: 0.5131
 2176/25000 [=>............................] - ETA: 1:02 - loss: 7.4905 - accuracy: 0.5115
 2208/25000 [=>............................] - ETA: 1:02 - loss: 7.4861 - accuracy: 0.5118
 2240/25000 [=>............................] - ETA: 1:02 - loss: 7.5023 - accuracy: 0.5107
 2272/25000 [=>............................] - ETA: 1:02 - loss: 7.4979 - accuracy: 0.5110
 2304/25000 [=>............................] - ETA: 1:02 - loss: 7.5468 - accuracy: 0.5078
 2336/25000 [=>............................] - ETA: 1:02 - loss: 7.5485 - accuracy: 0.5077
 2368/25000 [=>............................] - ETA: 1:01 - loss: 7.5565 - accuracy: 0.5072
 2400/25000 [=>............................] - ETA: 1:01 - loss: 7.5388 - accuracy: 0.5083
 2432/25000 [=>............................] - ETA: 1:01 - loss: 7.5342 - accuracy: 0.5086
 2464/25000 [=>............................] - ETA: 1:01 - loss: 7.5235 - accuracy: 0.5093
 2496/25000 [=>............................] - ETA: 1:01 - loss: 7.5130 - accuracy: 0.5100
 2528/25000 [==>...........................] - ETA: 1:01 - loss: 7.5453 - accuracy: 0.5079
 2560/25000 [==>...........................] - ETA: 1:01 - loss: 7.5648 - accuracy: 0.5066
 2592/25000 [==>...........................] - ETA: 1:00 - loss: 7.5897 - accuracy: 0.5050
 2624/25000 [==>...........................] - ETA: 1:00 - loss: 7.6374 - accuracy: 0.5019
 2656/25000 [==>...........................] - ETA: 1:00 - loss: 7.6147 - accuracy: 0.5034
 2688/25000 [==>...........................] - ETA: 1:00 - loss: 7.5982 - accuracy: 0.5045
 2720/25000 [==>...........................] - ETA: 1:00 - loss: 7.5990 - accuracy: 0.5044
 2752/25000 [==>...........................] - ETA: 1:00 - loss: 7.5942 - accuracy: 0.5047
 2784/25000 [==>...........................] - ETA: 1:00 - loss: 7.5840 - accuracy: 0.5054
 2816/25000 [==>...........................] - ETA: 1:00 - loss: 7.5958 - accuracy: 0.5046
 2848/25000 [==>...........................] - ETA: 1:00 - loss: 7.5751 - accuracy: 0.5060
 2880/25000 [==>...........................] - ETA: 59s - loss: 7.5974 - accuracy: 0.5045 
 2912/25000 [==>...........................] - ETA: 59s - loss: 7.5929 - accuracy: 0.5048
 2944/25000 [==>...........................] - ETA: 59s - loss: 7.5885 - accuracy: 0.5051
 2976/25000 [==>...........................] - ETA: 59s - loss: 7.5893 - accuracy: 0.5050
 3008/25000 [==>...........................] - ETA: 59s - loss: 7.5647 - accuracy: 0.5066
 3040/25000 [==>...........................] - ETA: 59s - loss: 7.5607 - accuracy: 0.5069
 3072/25000 [==>...........................] - ETA: 59s - loss: 7.5319 - accuracy: 0.5088
 3104/25000 [==>...........................] - ETA: 59s - loss: 7.5382 - accuracy: 0.5084
 3136/25000 [==>...........................] - ETA: 59s - loss: 7.5346 - accuracy: 0.5086
 3168/25000 [==>...........................] - ETA: 59s - loss: 7.5214 - accuracy: 0.5095
 3200/25000 [==>...........................] - ETA: 58s - loss: 7.5372 - accuracy: 0.5084
 3232/25000 [==>...........................] - ETA: 58s - loss: 7.5385 - accuracy: 0.5084
 3264/25000 [==>...........................] - ETA: 58s - loss: 7.5633 - accuracy: 0.5067
 3296/25000 [==>...........................] - ETA: 58s - loss: 7.5736 - accuracy: 0.5061
 3328/25000 [==>...........................] - ETA: 58s - loss: 7.5929 - accuracy: 0.5048
 3360/25000 [===>..........................] - ETA: 58s - loss: 7.5936 - accuracy: 0.5048
 3392/25000 [===>..........................] - ETA: 58s - loss: 7.6033 - accuracy: 0.5041
 3424/25000 [===>..........................] - ETA: 58s - loss: 7.6308 - accuracy: 0.5023
 3456/25000 [===>..........................] - ETA: 58s - loss: 7.6267 - accuracy: 0.5026
 3488/25000 [===>..........................] - ETA: 58s - loss: 7.6227 - accuracy: 0.5029
 3520/25000 [===>..........................] - ETA: 57s - loss: 7.6143 - accuracy: 0.5034
 3552/25000 [===>..........................] - ETA: 57s - loss: 7.6062 - accuracy: 0.5039
 3584/25000 [===>..........................] - ETA: 57s - loss: 7.6196 - accuracy: 0.5031
 3616/25000 [===>..........................] - ETA: 57s - loss: 7.6412 - accuracy: 0.5017
 3648/25000 [===>..........................] - ETA: 57s - loss: 7.6456 - accuracy: 0.5014
 3680/25000 [===>..........................] - ETA: 57s - loss: 7.6458 - accuracy: 0.5014
 3712/25000 [===>..........................] - ETA: 57s - loss: 7.6584 - accuracy: 0.5005
 3744/25000 [===>..........................] - ETA: 57s - loss: 7.6666 - accuracy: 0.5000
 3776/25000 [===>..........................] - ETA: 57s - loss: 7.6910 - accuracy: 0.4984
 3808/25000 [===>..........................] - ETA: 56s - loss: 7.6827 - accuracy: 0.4989
 3840/25000 [===>..........................] - ETA: 56s - loss: 7.7026 - accuracy: 0.4977
 3872/25000 [===>..........................] - ETA: 56s - loss: 7.6983 - accuracy: 0.4979
 3904/25000 [===>..........................] - ETA: 56s - loss: 7.6823 - accuracy: 0.4990
 3936/25000 [===>..........................] - ETA: 56s - loss: 7.6705 - accuracy: 0.4997
 3968/25000 [===>..........................] - ETA: 56s - loss: 7.6666 - accuracy: 0.5000
 4000/25000 [===>..........................] - ETA: 56s - loss: 7.6705 - accuracy: 0.4997
 4032/25000 [===>..........................] - ETA: 56s - loss: 7.6704 - accuracy: 0.4998
 4064/25000 [===>..........................] - ETA: 56s - loss: 7.6704 - accuracy: 0.4998
 4096/25000 [===>..........................] - ETA: 55s - loss: 7.6741 - accuracy: 0.4995
 4128/25000 [===>..........................] - ETA: 55s - loss: 7.6778 - accuracy: 0.4993
 4160/25000 [===>..........................] - ETA: 55s - loss: 7.6703 - accuracy: 0.4998
 4192/25000 [====>.........................] - ETA: 55s - loss: 7.6666 - accuracy: 0.5000
 4224/25000 [====>.........................] - ETA: 55s - loss: 7.6811 - accuracy: 0.4991
 4256/25000 [====>.........................] - ETA: 55s - loss: 7.6810 - accuracy: 0.4991
 4288/25000 [====>.........................] - ETA: 55s - loss: 7.6809 - accuracy: 0.4991
 4320/25000 [====>.........................] - ETA: 55s - loss: 7.6844 - accuracy: 0.4988
 4352/25000 [====>.........................] - ETA: 55s - loss: 7.6737 - accuracy: 0.4995
 4384/25000 [====>.........................] - ETA: 55s - loss: 7.6701 - accuracy: 0.4998
 4416/25000 [====>.........................] - ETA: 54s - loss: 7.6701 - accuracy: 0.4998
 4448/25000 [====>.........................] - ETA: 54s - loss: 7.6735 - accuracy: 0.4996
 4480/25000 [====>.........................] - ETA: 54s - loss: 7.6803 - accuracy: 0.4991
 4512/25000 [====>.........................] - ETA: 54s - loss: 7.6734 - accuracy: 0.4996
 4544/25000 [====>.........................] - ETA: 54s - loss: 7.6700 - accuracy: 0.4998
 4576/25000 [====>.........................] - ETA: 54s - loss: 7.6666 - accuracy: 0.5000
 4608/25000 [====>.........................] - ETA: 54s - loss: 7.6733 - accuracy: 0.4996
 4640/25000 [====>.........................] - ETA: 54s - loss: 7.6732 - accuracy: 0.4996
 4672/25000 [====>.........................] - ETA: 54s - loss: 7.6633 - accuracy: 0.5002
 4704/25000 [====>.........................] - ETA: 54s - loss: 7.6503 - accuracy: 0.5011
 4736/25000 [====>.........................] - ETA: 53s - loss: 7.6375 - accuracy: 0.5019
 4768/25000 [====>.........................] - ETA: 53s - loss: 7.6538 - accuracy: 0.5008
 4800/25000 [====>.........................] - ETA: 53s - loss: 7.6538 - accuracy: 0.5008
 4832/25000 [====>.........................] - ETA: 53s - loss: 7.6476 - accuracy: 0.5012
 4864/25000 [====>.........................] - ETA: 53s - loss: 7.6477 - accuracy: 0.5012
 4896/25000 [====>.........................] - ETA: 53s - loss: 7.6353 - accuracy: 0.5020
 4928/25000 [====>.........................] - ETA: 53s - loss: 7.6355 - accuracy: 0.5020
 4960/25000 [====>.........................] - ETA: 53s - loss: 7.6233 - accuracy: 0.5028
 4992/25000 [====>.........................] - ETA: 53s - loss: 7.6390 - accuracy: 0.5018
 5024/25000 [=====>........................] - ETA: 52s - loss: 7.6361 - accuracy: 0.5020
 5056/25000 [=====>........................] - ETA: 52s - loss: 7.6424 - accuracy: 0.5016
 5088/25000 [=====>........................] - ETA: 52s - loss: 7.6335 - accuracy: 0.5022
 5120/25000 [=====>........................] - ETA: 52s - loss: 7.6247 - accuracy: 0.5027
 5152/25000 [=====>........................] - ETA: 52s - loss: 7.6309 - accuracy: 0.5023
 5184/25000 [=====>........................] - ETA: 52s - loss: 7.6252 - accuracy: 0.5027
 5216/25000 [=====>........................] - ETA: 52s - loss: 7.6284 - accuracy: 0.5025
 5248/25000 [=====>........................] - ETA: 52s - loss: 7.6316 - accuracy: 0.5023
 5280/25000 [=====>........................] - ETA: 52s - loss: 7.6260 - accuracy: 0.5027
 5312/25000 [=====>........................] - ETA: 52s - loss: 7.6349 - accuracy: 0.5021
 5344/25000 [=====>........................] - ETA: 52s - loss: 7.6293 - accuracy: 0.5024
 5376/25000 [=====>........................] - ETA: 51s - loss: 7.6295 - accuracy: 0.5024
 5408/25000 [=====>........................] - ETA: 51s - loss: 7.6213 - accuracy: 0.5030
 5440/25000 [=====>........................] - ETA: 51s - loss: 7.6215 - accuracy: 0.5029
 5472/25000 [=====>........................] - ETA: 51s - loss: 7.6106 - accuracy: 0.5037
 5504/25000 [=====>........................] - ETA: 51s - loss: 7.6248 - accuracy: 0.5027
 5536/25000 [=====>........................] - ETA: 51s - loss: 7.6251 - accuracy: 0.5027
 5568/25000 [=====>........................] - ETA: 51s - loss: 7.6446 - accuracy: 0.5014
 5600/25000 [=====>........................] - ETA: 51s - loss: 7.6420 - accuracy: 0.5016
 5632/25000 [=====>........................] - ETA: 51s - loss: 7.6448 - accuracy: 0.5014
 5664/25000 [=====>........................] - ETA: 51s - loss: 7.6423 - accuracy: 0.5016
 5696/25000 [=====>........................] - ETA: 50s - loss: 7.6397 - accuracy: 0.5018
 5728/25000 [=====>........................] - ETA: 50s - loss: 7.6345 - accuracy: 0.5021
 5760/25000 [=====>........................] - ETA: 50s - loss: 7.6240 - accuracy: 0.5028
 5792/25000 [=====>........................] - ETA: 50s - loss: 7.6349 - accuracy: 0.5021
 5824/25000 [=====>........................] - ETA: 50s - loss: 7.6350 - accuracy: 0.5021
 5856/25000 [======>.......................] - ETA: 50s - loss: 7.6378 - accuracy: 0.5019
 5888/25000 [======>.......................] - ETA: 50s - loss: 7.6406 - accuracy: 0.5017
 5920/25000 [======>.......................] - ETA: 50s - loss: 7.6459 - accuracy: 0.5014
 5952/25000 [======>.......................] - ETA: 50s - loss: 7.6486 - accuracy: 0.5012
 5984/25000 [======>.......................] - ETA: 50s - loss: 7.6512 - accuracy: 0.5010
 6016/25000 [======>.......................] - ETA: 49s - loss: 7.6539 - accuracy: 0.5008
 6048/25000 [======>.......................] - ETA: 49s - loss: 7.6413 - accuracy: 0.5017
 6080/25000 [======>.......................] - ETA: 49s - loss: 7.6414 - accuracy: 0.5016
 6112/25000 [======>.......................] - ETA: 49s - loss: 7.6365 - accuracy: 0.5020
 6144/25000 [======>.......................] - ETA: 49s - loss: 7.6292 - accuracy: 0.5024
 6176/25000 [======>.......................] - ETA: 49s - loss: 7.6244 - accuracy: 0.5028
 6208/25000 [======>.......................] - ETA: 49s - loss: 7.6246 - accuracy: 0.5027
 6240/25000 [======>.......................] - ETA: 49s - loss: 7.6126 - accuracy: 0.5035
 6272/25000 [======>.......................] - ETA: 49s - loss: 7.6128 - accuracy: 0.5035
 6304/25000 [======>.......................] - ETA: 49s - loss: 7.6204 - accuracy: 0.5030
 6336/25000 [======>.......................] - ETA: 49s - loss: 7.6182 - accuracy: 0.5032
 6368/25000 [======>.......................] - ETA: 48s - loss: 7.6185 - accuracy: 0.5031
 6400/25000 [======>.......................] - ETA: 48s - loss: 7.6235 - accuracy: 0.5028
 6432/25000 [======>.......................] - ETA: 48s - loss: 7.6309 - accuracy: 0.5023
 6464/25000 [======>.......................] - ETA: 48s - loss: 7.6239 - accuracy: 0.5028
 6496/25000 [======>.......................] - ETA: 48s - loss: 7.6336 - accuracy: 0.5022
 6528/25000 [======>.......................] - ETA: 48s - loss: 7.6408 - accuracy: 0.5017
 6560/25000 [======>.......................] - ETA: 48s - loss: 7.6409 - accuracy: 0.5017
 6592/25000 [======>.......................] - ETA: 48s - loss: 7.6364 - accuracy: 0.5020
 6624/25000 [======>.......................] - ETA: 48s - loss: 7.6388 - accuracy: 0.5018
 6656/25000 [======>.......................] - ETA: 48s - loss: 7.6367 - accuracy: 0.5020
 6688/25000 [=======>......................] - ETA: 47s - loss: 7.6437 - accuracy: 0.5015
 6720/25000 [=======>......................] - ETA: 47s - loss: 7.6347 - accuracy: 0.5021
 6752/25000 [=======>......................] - ETA: 47s - loss: 7.6212 - accuracy: 0.5030
 6784/25000 [=======>......................] - ETA: 47s - loss: 7.6192 - accuracy: 0.5031
 6816/25000 [=======>......................] - ETA: 47s - loss: 7.6329 - accuracy: 0.5022
 6848/25000 [=======>......................] - ETA: 47s - loss: 7.6330 - accuracy: 0.5022
 6880/25000 [=======>......................] - ETA: 47s - loss: 7.6310 - accuracy: 0.5023
 6912/25000 [=======>......................] - ETA: 47s - loss: 7.6289 - accuracy: 0.5025
 6944/25000 [=======>......................] - ETA: 47s - loss: 7.6291 - accuracy: 0.5024
 6976/25000 [=======>......................] - ETA: 47s - loss: 7.6293 - accuracy: 0.5024
 7008/25000 [=======>......................] - ETA: 47s - loss: 7.6360 - accuracy: 0.5020
 7040/25000 [=======>......................] - ETA: 46s - loss: 7.6405 - accuracy: 0.5017
 7072/25000 [=======>......................] - ETA: 46s - loss: 7.6558 - accuracy: 0.5007
 7104/25000 [=======>......................] - ETA: 46s - loss: 7.6537 - accuracy: 0.5008
 7136/25000 [=======>......................] - ETA: 46s - loss: 7.6516 - accuracy: 0.5010
 7168/25000 [=======>......................] - ETA: 46s - loss: 7.6495 - accuracy: 0.5011
 7200/25000 [=======>......................] - ETA: 46s - loss: 7.6453 - accuracy: 0.5014
 7232/25000 [=======>......................] - ETA: 46s - loss: 7.6263 - accuracy: 0.5026
 7264/25000 [=======>......................] - ETA: 46s - loss: 7.6244 - accuracy: 0.5028
 7296/25000 [=======>......................] - ETA: 46s - loss: 7.6183 - accuracy: 0.5032
 7328/25000 [=======>......................] - ETA: 46s - loss: 7.6122 - accuracy: 0.5035
 7360/25000 [=======>......................] - ETA: 45s - loss: 7.6166 - accuracy: 0.5033
 7392/25000 [=======>......................] - ETA: 45s - loss: 7.6148 - accuracy: 0.5034
 7424/25000 [=======>......................] - ETA: 45s - loss: 7.6253 - accuracy: 0.5027
 7456/25000 [=======>......................] - ETA: 45s - loss: 7.6193 - accuracy: 0.5031
 7488/25000 [=======>......................] - ETA: 45s - loss: 7.6257 - accuracy: 0.5027
 7520/25000 [========>.....................] - ETA: 45s - loss: 7.6279 - accuracy: 0.5025
 7552/25000 [========>.....................] - ETA: 45s - loss: 7.6341 - accuracy: 0.5021
 7584/25000 [========>.....................] - ETA: 45s - loss: 7.6302 - accuracy: 0.5024
 7616/25000 [========>.....................] - ETA: 45s - loss: 7.6304 - accuracy: 0.5024
 7648/25000 [========>.....................] - ETA: 45s - loss: 7.6245 - accuracy: 0.5027
 7680/25000 [========>.....................] - ETA: 45s - loss: 7.6227 - accuracy: 0.5029
 7712/25000 [========>.....................] - ETA: 44s - loss: 7.6308 - accuracy: 0.5023
 7744/25000 [========>.....................] - ETA: 44s - loss: 7.6389 - accuracy: 0.5018
 7776/25000 [========>.....................] - ETA: 44s - loss: 7.6430 - accuracy: 0.5015
 7808/25000 [========>.....................] - ETA: 44s - loss: 7.6450 - accuracy: 0.5014
 7840/25000 [========>.....................] - ETA: 44s - loss: 7.6392 - accuracy: 0.5018
 7872/25000 [========>.....................] - ETA: 44s - loss: 7.6471 - accuracy: 0.5013
 7904/25000 [========>.....................] - ETA: 44s - loss: 7.6433 - accuracy: 0.5015
 7936/25000 [========>.....................] - ETA: 44s - loss: 7.6512 - accuracy: 0.5010
 7968/25000 [========>.....................] - ETA: 44s - loss: 7.6628 - accuracy: 0.5003
 8000/25000 [========>.....................] - ETA: 44s - loss: 7.6551 - accuracy: 0.5008
 8032/25000 [========>.....................] - ETA: 44s - loss: 7.6552 - accuracy: 0.5007
 8064/25000 [========>.....................] - ETA: 43s - loss: 7.6457 - accuracy: 0.5014
 8096/25000 [========>.....................] - ETA: 43s - loss: 7.6515 - accuracy: 0.5010
 8128/25000 [========>.....................] - ETA: 43s - loss: 7.6515 - accuracy: 0.5010
 8160/25000 [========>.....................] - ETA: 43s - loss: 7.6553 - accuracy: 0.5007
 8192/25000 [========>.....................] - ETA: 43s - loss: 7.6610 - accuracy: 0.5004
 8224/25000 [========>.....................] - ETA: 43s - loss: 7.6573 - accuracy: 0.5006
 8256/25000 [========>.....................] - ETA: 43s - loss: 7.6573 - accuracy: 0.5006
 8288/25000 [========>.....................] - ETA: 43s - loss: 7.6537 - accuracy: 0.5008
 8320/25000 [========>.....................] - ETA: 43s - loss: 7.6556 - accuracy: 0.5007
 8352/25000 [=========>....................] - ETA: 43s - loss: 7.6593 - accuracy: 0.5005
 8384/25000 [=========>....................] - ETA: 43s - loss: 7.6611 - accuracy: 0.5004
 8416/25000 [=========>....................] - ETA: 42s - loss: 7.6666 - accuracy: 0.5000
 8448/25000 [=========>....................] - ETA: 42s - loss: 7.6612 - accuracy: 0.5004
 8480/25000 [=========>....................] - ETA: 42s - loss: 7.6630 - accuracy: 0.5002
 8512/25000 [=========>....................] - ETA: 42s - loss: 7.6648 - accuracy: 0.5001
 8544/25000 [=========>....................] - ETA: 42s - loss: 7.6648 - accuracy: 0.5001
 8576/25000 [=========>....................] - ETA: 42s - loss: 7.6577 - accuracy: 0.5006
 8608/25000 [=========>....................] - ETA: 42s - loss: 7.6542 - accuracy: 0.5008
 8640/25000 [=========>....................] - ETA: 42s - loss: 7.6524 - accuracy: 0.5009
 8672/25000 [=========>....................] - ETA: 42s - loss: 7.6542 - accuracy: 0.5008
 8704/25000 [=========>....................] - ETA: 42s - loss: 7.6560 - accuracy: 0.5007
 8736/25000 [=========>....................] - ETA: 42s - loss: 7.6596 - accuracy: 0.5005
 8768/25000 [=========>....................] - ETA: 41s - loss: 7.6561 - accuracy: 0.5007
 8800/25000 [=========>....................] - ETA: 41s - loss: 7.6509 - accuracy: 0.5010
 8832/25000 [=========>....................] - ETA: 41s - loss: 7.6562 - accuracy: 0.5007
 8864/25000 [=========>....................] - ETA: 41s - loss: 7.6683 - accuracy: 0.4999
 8896/25000 [=========>....................] - ETA: 41s - loss: 7.6649 - accuracy: 0.5001
 8928/25000 [=========>....................] - ETA: 41s - loss: 7.6649 - accuracy: 0.5001
 8960/25000 [=========>....................] - ETA: 41s - loss: 7.6683 - accuracy: 0.4999
 8992/25000 [=========>....................] - ETA: 41s - loss: 7.6632 - accuracy: 0.5002
 9024/25000 [=========>....................] - ETA: 41s - loss: 7.6632 - accuracy: 0.5002
 9056/25000 [=========>....................] - ETA: 41s - loss: 7.6700 - accuracy: 0.4998
 9088/25000 [=========>....................] - ETA: 41s - loss: 7.6683 - accuracy: 0.4999
 9120/25000 [=========>....................] - ETA: 40s - loss: 7.6733 - accuracy: 0.4996
 9152/25000 [=========>....................] - ETA: 40s - loss: 7.6767 - accuracy: 0.4993
 9184/25000 [==========>...................] - ETA: 40s - loss: 7.6783 - accuracy: 0.4992
 9216/25000 [==========>...................] - ETA: 40s - loss: 7.6666 - accuracy: 0.5000
 9248/25000 [==========>...................] - ETA: 40s - loss: 7.6633 - accuracy: 0.5002
 9280/25000 [==========>...................] - ETA: 40s - loss: 7.6584 - accuracy: 0.5005
 9312/25000 [==========>...................] - ETA: 40s - loss: 7.6567 - accuracy: 0.5006
 9344/25000 [==========>...................] - ETA: 40s - loss: 7.6535 - accuracy: 0.5009
 9376/25000 [==========>...................] - ETA: 40s - loss: 7.6535 - accuracy: 0.5009
 9408/25000 [==========>...................] - ETA: 40s - loss: 7.6471 - accuracy: 0.5013
 9440/25000 [==========>...................] - ETA: 40s - loss: 7.6439 - accuracy: 0.5015
 9472/25000 [==========>...................] - ETA: 40s - loss: 7.6504 - accuracy: 0.5011
 9504/25000 [==========>...................] - ETA: 39s - loss: 7.6505 - accuracy: 0.5011
 9536/25000 [==========>...................] - ETA: 39s - loss: 7.6473 - accuracy: 0.5013
 9568/25000 [==========>...................] - ETA: 39s - loss: 7.6522 - accuracy: 0.5009
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.6554 - accuracy: 0.5007
 9632/25000 [==========>...................] - ETA: 39s - loss: 7.6555 - accuracy: 0.5007
 9664/25000 [==========>...................] - ETA: 39s - loss: 7.6555 - accuracy: 0.5007
 9696/25000 [==========>...................] - ETA: 39s - loss: 7.6603 - accuracy: 0.5004
 9728/25000 [==========>...................] - ETA: 39s - loss: 7.6572 - accuracy: 0.5006
 9760/25000 [==========>...................] - ETA: 39s - loss: 7.6588 - accuracy: 0.5005
 9792/25000 [==========>...................] - ETA: 39s - loss: 7.6557 - accuracy: 0.5007
 9824/25000 [==========>...................] - ETA: 39s - loss: 7.6557 - accuracy: 0.5007
 9856/25000 [==========>...................] - ETA: 38s - loss: 7.6542 - accuracy: 0.5008
 9888/25000 [==========>...................] - ETA: 38s - loss: 7.6558 - accuracy: 0.5007
 9920/25000 [==========>...................] - ETA: 38s - loss: 7.6465 - accuracy: 0.5013
 9952/25000 [==========>...................] - ETA: 38s - loss: 7.6450 - accuracy: 0.5014
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.6451 - accuracy: 0.5014
10016/25000 [===========>..................] - ETA: 38s - loss: 7.6482 - accuracy: 0.5012
10048/25000 [===========>..................] - ETA: 38s - loss: 7.6437 - accuracy: 0.5015
10080/25000 [===========>..................] - ETA: 38s - loss: 7.6423 - accuracy: 0.5016
10112/25000 [===========>..................] - ETA: 38s - loss: 7.6424 - accuracy: 0.5016
10144/25000 [===========>..................] - ETA: 38s - loss: 7.6424 - accuracy: 0.5016
10176/25000 [===========>..................] - ETA: 38s - loss: 7.6350 - accuracy: 0.5021
10208/25000 [===========>..................] - ETA: 38s - loss: 7.6366 - accuracy: 0.5020
10240/25000 [===========>..................] - ETA: 37s - loss: 7.6307 - accuracy: 0.5023
10272/25000 [===========>..................] - ETA: 37s - loss: 7.6323 - accuracy: 0.5022
10304/25000 [===========>..................] - ETA: 37s - loss: 7.6294 - accuracy: 0.5024
10336/25000 [===========>..................] - ETA: 37s - loss: 7.6266 - accuracy: 0.5026
10368/25000 [===========>..................] - ETA: 37s - loss: 7.6252 - accuracy: 0.5027
10400/25000 [===========>..................] - ETA: 37s - loss: 7.6283 - accuracy: 0.5025
10432/25000 [===========>..................] - ETA: 37s - loss: 7.6269 - accuracy: 0.5026
10464/25000 [===========>..................] - ETA: 37s - loss: 7.6256 - accuracy: 0.5027
10496/25000 [===========>..................] - ETA: 37s - loss: 7.6286 - accuracy: 0.5025
10528/25000 [===========>..................] - ETA: 37s - loss: 7.6302 - accuracy: 0.5024
10560/25000 [===========>..................] - ETA: 37s - loss: 7.6347 - accuracy: 0.5021
10592/25000 [===========>..................] - ETA: 37s - loss: 7.6420 - accuracy: 0.5016
10624/25000 [===========>..................] - ETA: 36s - loss: 7.6479 - accuracy: 0.5012
10656/25000 [===========>..................] - ETA: 36s - loss: 7.6580 - accuracy: 0.5006
10688/25000 [===========>..................] - ETA: 36s - loss: 7.6494 - accuracy: 0.5011
10720/25000 [===========>..................] - ETA: 36s - loss: 7.6537 - accuracy: 0.5008
10752/25000 [===========>..................] - ETA: 36s - loss: 7.6509 - accuracy: 0.5010
10784/25000 [===========>..................] - ETA: 36s - loss: 7.6453 - accuracy: 0.5014
10816/25000 [===========>..................] - ETA: 36s - loss: 7.6383 - accuracy: 0.5018
10848/25000 [============>.................] - ETA: 36s - loss: 7.6313 - accuracy: 0.5023
10880/25000 [============>.................] - ETA: 36s - loss: 7.6314 - accuracy: 0.5023
10912/25000 [============>.................] - ETA: 36s - loss: 7.6273 - accuracy: 0.5026
10944/25000 [============>.................] - ETA: 36s - loss: 7.6288 - accuracy: 0.5025
10976/25000 [============>.................] - ETA: 35s - loss: 7.6317 - accuracy: 0.5023
11008/25000 [============>.................] - ETA: 35s - loss: 7.6332 - accuracy: 0.5022
11040/25000 [============>.................] - ETA: 35s - loss: 7.6319 - accuracy: 0.5023
11072/25000 [============>.................] - ETA: 35s - loss: 7.6362 - accuracy: 0.5020
11104/25000 [============>.................] - ETA: 35s - loss: 7.6431 - accuracy: 0.5015
11136/25000 [============>.................] - ETA: 35s - loss: 7.6418 - accuracy: 0.5016
11168/25000 [============>.................] - ETA: 35s - loss: 7.6392 - accuracy: 0.5018
11200/25000 [============>.................] - ETA: 35s - loss: 7.6379 - accuracy: 0.5019
11232/25000 [============>.................] - ETA: 35s - loss: 7.6420 - accuracy: 0.5016
11264/25000 [============>.................] - ETA: 35s - loss: 7.6476 - accuracy: 0.5012
11296/25000 [============>.................] - ETA: 35s - loss: 7.6503 - accuracy: 0.5011
11328/25000 [============>.................] - ETA: 35s - loss: 7.6463 - accuracy: 0.5013
11360/25000 [============>.................] - ETA: 34s - loss: 7.6464 - accuracy: 0.5013
11392/25000 [============>.................] - ETA: 34s - loss: 7.6478 - accuracy: 0.5012
11424/25000 [============>.................] - ETA: 34s - loss: 7.6492 - accuracy: 0.5011
11456/25000 [============>.................] - ETA: 34s - loss: 7.6586 - accuracy: 0.5005
11488/25000 [============>.................] - ETA: 34s - loss: 7.6546 - accuracy: 0.5008
11520/25000 [============>.................] - ETA: 34s - loss: 7.6586 - accuracy: 0.5005
11552/25000 [============>.................] - ETA: 34s - loss: 7.6560 - accuracy: 0.5007
11584/25000 [============>.................] - ETA: 34s - loss: 7.6547 - accuracy: 0.5008
11616/25000 [============>.................] - ETA: 34s - loss: 7.6534 - accuracy: 0.5009
11648/25000 [============>.................] - ETA: 34s - loss: 7.6521 - accuracy: 0.5009
11680/25000 [=============>................] - ETA: 34s - loss: 7.6535 - accuracy: 0.5009
11712/25000 [=============>................] - ETA: 34s - loss: 7.6483 - accuracy: 0.5012
11744/25000 [=============>................] - ETA: 33s - loss: 7.6444 - accuracy: 0.5014
11776/25000 [=============>................] - ETA: 33s - loss: 7.6432 - accuracy: 0.5015
11808/25000 [=============>................] - ETA: 33s - loss: 7.6458 - accuracy: 0.5014
11840/25000 [=============>................] - ETA: 33s - loss: 7.6446 - accuracy: 0.5014
11872/25000 [=============>................] - ETA: 33s - loss: 7.6434 - accuracy: 0.5015
11904/25000 [=============>................] - ETA: 33s - loss: 7.6434 - accuracy: 0.5015
11936/25000 [=============>................] - ETA: 33s - loss: 7.6461 - accuracy: 0.5013
11968/25000 [=============>................] - ETA: 33s - loss: 7.6410 - accuracy: 0.5017
12000/25000 [=============>................] - ETA: 33s - loss: 7.6398 - accuracy: 0.5017
12032/25000 [=============>................] - ETA: 33s - loss: 7.6411 - accuracy: 0.5017
12064/25000 [=============>................] - ETA: 33s - loss: 7.6399 - accuracy: 0.5017
12096/25000 [=============>................] - ETA: 32s - loss: 7.6400 - accuracy: 0.5017
12128/25000 [=============>................] - ETA: 32s - loss: 7.6413 - accuracy: 0.5016
12160/25000 [=============>................] - ETA: 32s - loss: 7.6477 - accuracy: 0.5012
12192/25000 [=============>................] - ETA: 32s - loss: 7.6415 - accuracy: 0.5016
12224/25000 [=============>................] - ETA: 32s - loss: 7.6453 - accuracy: 0.5014
12256/25000 [=============>................] - ETA: 32s - loss: 7.6516 - accuracy: 0.5010
12288/25000 [=============>................] - ETA: 32s - loss: 7.6516 - accuracy: 0.5010
12320/25000 [=============>................] - ETA: 32s - loss: 7.6592 - accuracy: 0.5005
12352/25000 [=============>................] - ETA: 32s - loss: 7.6579 - accuracy: 0.5006
12384/25000 [=============>................] - ETA: 32s - loss: 7.6505 - accuracy: 0.5010
12416/25000 [=============>................] - ETA: 32s - loss: 7.6543 - accuracy: 0.5008
12448/25000 [=============>................] - ETA: 32s - loss: 7.6506 - accuracy: 0.5010
12480/25000 [=============>................] - ETA: 31s - loss: 7.6568 - accuracy: 0.5006
12512/25000 [==============>...............] - ETA: 31s - loss: 7.6556 - accuracy: 0.5007
12544/25000 [==============>...............] - ETA: 31s - loss: 7.6605 - accuracy: 0.5004
12576/25000 [==============>...............] - ETA: 31s - loss: 7.6593 - accuracy: 0.5005
12608/25000 [==============>...............] - ETA: 31s - loss: 7.6605 - accuracy: 0.5004
12640/25000 [==============>...............] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
12672/25000 [==============>...............] - ETA: 31s - loss: 7.6618 - accuracy: 0.5003
12704/25000 [==============>...............] - ETA: 31s - loss: 7.6642 - accuracy: 0.5002
12736/25000 [==============>...............] - ETA: 31s - loss: 7.6630 - accuracy: 0.5002
12768/25000 [==============>...............] - ETA: 31s - loss: 7.6642 - accuracy: 0.5002
12800/25000 [==============>...............] - ETA: 31s - loss: 7.6642 - accuracy: 0.5002
12832/25000 [==============>...............] - ETA: 31s - loss: 7.6714 - accuracy: 0.4997
12864/25000 [==============>...............] - ETA: 30s - loss: 7.6785 - accuracy: 0.4992
12896/25000 [==============>...............] - ETA: 30s - loss: 7.6773 - accuracy: 0.4993
12928/25000 [==============>...............] - ETA: 30s - loss: 7.6809 - accuracy: 0.4991
12960/25000 [==============>...............] - ETA: 30s - loss: 7.6808 - accuracy: 0.4991
12992/25000 [==============>...............] - ETA: 30s - loss: 7.6820 - accuracy: 0.4990
13024/25000 [==============>...............] - ETA: 30s - loss: 7.6796 - accuracy: 0.4992
13056/25000 [==============>...............] - ETA: 30s - loss: 7.6784 - accuracy: 0.4992
13088/25000 [==============>...............] - ETA: 30s - loss: 7.6783 - accuracy: 0.4992
13120/25000 [==============>...............] - ETA: 30s - loss: 7.6725 - accuracy: 0.4996
13152/25000 [==============>...............] - ETA: 30s - loss: 7.6690 - accuracy: 0.4998
13184/25000 [==============>...............] - ETA: 30s - loss: 7.6736 - accuracy: 0.4995
13216/25000 [==============>...............] - ETA: 30s - loss: 7.6701 - accuracy: 0.4998
13248/25000 [==============>...............] - ETA: 29s - loss: 7.6678 - accuracy: 0.4999
13280/25000 [==============>...............] - ETA: 29s - loss: 7.6620 - accuracy: 0.5003
13312/25000 [==============>...............] - ETA: 29s - loss: 7.6620 - accuracy: 0.5003
13344/25000 [===============>..............] - ETA: 29s - loss: 7.6689 - accuracy: 0.4999
13376/25000 [===============>..............] - ETA: 29s - loss: 7.6735 - accuracy: 0.4996
13408/25000 [===============>..............] - ETA: 29s - loss: 7.6746 - accuracy: 0.4995
13440/25000 [===============>..............] - ETA: 29s - loss: 7.6735 - accuracy: 0.4996
13472/25000 [===============>..............] - ETA: 29s - loss: 7.6689 - accuracy: 0.4999
13504/25000 [===============>..............] - ETA: 29s - loss: 7.6723 - accuracy: 0.4996
13536/25000 [===============>..............] - ETA: 29s - loss: 7.6734 - accuracy: 0.4996
13568/25000 [===============>..............] - ETA: 29s - loss: 7.6700 - accuracy: 0.4998
13600/25000 [===============>..............] - ETA: 29s - loss: 7.6734 - accuracy: 0.4996
13632/25000 [===============>..............] - ETA: 28s - loss: 7.6734 - accuracy: 0.4996
13664/25000 [===============>..............] - ETA: 28s - loss: 7.6745 - accuracy: 0.4995
13696/25000 [===============>..............] - ETA: 28s - loss: 7.6711 - accuracy: 0.4997
13728/25000 [===============>..............] - ETA: 28s - loss: 7.6677 - accuracy: 0.4999
13760/25000 [===============>..............] - ETA: 28s - loss: 7.6688 - accuracy: 0.4999
13792/25000 [===============>..............] - ETA: 28s - loss: 7.6677 - accuracy: 0.4999
13824/25000 [===============>..............] - ETA: 28s - loss: 7.6611 - accuracy: 0.5004
13856/25000 [===============>..............] - ETA: 28s - loss: 7.6622 - accuracy: 0.5003
13888/25000 [===============>..............] - ETA: 28s - loss: 7.6622 - accuracy: 0.5003
13920/25000 [===============>..............] - ETA: 28s - loss: 7.6567 - accuracy: 0.5006
13952/25000 [===============>..............] - ETA: 28s - loss: 7.6600 - accuracy: 0.5004
13984/25000 [===============>..............] - ETA: 28s - loss: 7.6611 - accuracy: 0.5004
14016/25000 [===============>..............] - ETA: 27s - loss: 7.6590 - accuracy: 0.5005
14048/25000 [===============>..............] - ETA: 27s - loss: 7.6655 - accuracy: 0.5001
14080/25000 [===============>..............] - ETA: 27s - loss: 7.6677 - accuracy: 0.4999
14112/25000 [===============>..............] - ETA: 27s - loss: 7.6688 - accuracy: 0.4999
14144/25000 [===============>..............] - ETA: 27s - loss: 7.6645 - accuracy: 0.5001
14176/25000 [================>.............] - ETA: 27s - loss: 7.6634 - accuracy: 0.5002
14208/25000 [================>.............] - ETA: 27s - loss: 7.6591 - accuracy: 0.5005
14240/25000 [================>.............] - ETA: 27s - loss: 7.6559 - accuracy: 0.5007
14272/25000 [================>.............] - ETA: 27s - loss: 7.6591 - accuracy: 0.5005
14304/25000 [================>.............] - ETA: 27s - loss: 7.6634 - accuracy: 0.5002
14336/25000 [================>.............] - ETA: 27s - loss: 7.6602 - accuracy: 0.5004
14368/25000 [================>.............] - ETA: 27s - loss: 7.6602 - accuracy: 0.5004
14400/25000 [================>.............] - ETA: 26s - loss: 7.6645 - accuracy: 0.5001
14432/25000 [================>.............] - ETA: 26s - loss: 7.6634 - accuracy: 0.5002
14464/25000 [================>.............] - ETA: 26s - loss: 7.6613 - accuracy: 0.5003
14496/25000 [================>.............] - ETA: 26s - loss: 7.6603 - accuracy: 0.5004
14528/25000 [================>.............] - ETA: 26s - loss: 7.6592 - accuracy: 0.5005
14560/25000 [================>.............] - ETA: 26s - loss: 7.6592 - accuracy: 0.5005
14592/25000 [================>.............] - ETA: 26s - loss: 7.6666 - accuracy: 0.5000
14624/25000 [================>.............] - ETA: 26s - loss: 7.6645 - accuracy: 0.5001
14656/25000 [================>.............] - ETA: 26s - loss: 7.6645 - accuracy: 0.5001
14688/25000 [================>.............] - ETA: 26s - loss: 7.6624 - accuracy: 0.5003
14720/25000 [================>.............] - ETA: 26s - loss: 7.6583 - accuracy: 0.5005
14752/25000 [================>.............] - ETA: 26s - loss: 7.6562 - accuracy: 0.5007
14784/25000 [================>.............] - ETA: 25s - loss: 7.6562 - accuracy: 0.5007
14816/25000 [================>.............] - ETA: 25s - loss: 7.6542 - accuracy: 0.5008
14848/25000 [================>.............] - ETA: 25s - loss: 7.6522 - accuracy: 0.5009
14880/25000 [================>.............] - ETA: 25s - loss: 7.6522 - accuracy: 0.5009
14912/25000 [================>.............] - ETA: 25s - loss: 7.6522 - accuracy: 0.5009
14944/25000 [================>.............] - ETA: 25s - loss: 7.6523 - accuracy: 0.5009
14976/25000 [================>.............] - ETA: 25s - loss: 7.6472 - accuracy: 0.5013
15008/25000 [=================>............] - ETA: 25s - loss: 7.6482 - accuracy: 0.5012
15040/25000 [=================>............] - ETA: 25s - loss: 7.6422 - accuracy: 0.5016
15072/25000 [=================>............] - ETA: 25s - loss: 7.6453 - accuracy: 0.5014
15104/25000 [=================>............] - ETA: 25s - loss: 7.6504 - accuracy: 0.5011
15136/25000 [=================>............] - ETA: 25s - loss: 7.6443 - accuracy: 0.5015
15168/25000 [=================>............] - ETA: 24s - loss: 7.6424 - accuracy: 0.5016
15200/25000 [=================>............] - ETA: 24s - loss: 7.6464 - accuracy: 0.5013
15232/25000 [=================>............] - ETA: 24s - loss: 7.6495 - accuracy: 0.5011
15264/25000 [=================>............] - ETA: 24s - loss: 7.6465 - accuracy: 0.5013
15296/25000 [=================>............] - ETA: 24s - loss: 7.6426 - accuracy: 0.5016
15328/25000 [=================>............] - ETA: 24s - loss: 7.6426 - accuracy: 0.5016
15360/25000 [=================>............] - ETA: 24s - loss: 7.6417 - accuracy: 0.5016
15392/25000 [=================>............] - ETA: 24s - loss: 7.6447 - accuracy: 0.5014
15424/25000 [=================>............] - ETA: 24s - loss: 7.6467 - accuracy: 0.5013
15456/25000 [=================>............] - ETA: 24s - loss: 7.6468 - accuracy: 0.5013
15488/25000 [=================>............] - ETA: 24s - loss: 7.6458 - accuracy: 0.5014
15520/25000 [=================>............] - ETA: 24s - loss: 7.6459 - accuracy: 0.5014
15552/25000 [=================>............] - ETA: 23s - loss: 7.6439 - accuracy: 0.5015
15584/25000 [=================>............] - ETA: 23s - loss: 7.6430 - accuracy: 0.5015
15616/25000 [=================>............] - ETA: 23s - loss: 7.6460 - accuracy: 0.5013
15648/25000 [=================>............] - ETA: 23s - loss: 7.6451 - accuracy: 0.5014
15680/25000 [=================>............] - ETA: 23s - loss: 7.6441 - accuracy: 0.5015
15712/25000 [=================>............] - ETA: 23s - loss: 7.6461 - accuracy: 0.5013
15744/25000 [=================>............] - ETA: 23s - loss: 7.6452 - accuracy: 0.5014
15776/25000 [=================>............] - ETA: 23s - loss: 7.6433 - accuracy: 0.5015
15808/25000 [=================>............] - ETA: 23s - loss: 7.6453 - accuracy: 0.5014
15840/25000 [==================>...........] - ETA: 23s - loss: 7.6453 - accuracy: 0.5014
15872/25000 [==================>...........] - ETA: 23s - loss: 7.6473 - accuracy: 0.5013
15904/25000 [==================>...........] - ETA: 23s - loss: 7.6464 - accuracy: 0.5013
15936/25000 [==================>...........] - ETA: 22s - loss: 7.6483 - accuracy: 0.5012
15968/25000 [==================>...........] - ETA: 22s - loss: 7.6484 - accuracy: 0.5012
16000/25000 [==================>...........] - ETA: 22s - loss: 7.6503 - accuracy: 0.5011
16032/25000 [==================>...........] - ETA: 22s - loss: 7.6494 - accuracy: 0.5011
16064/25000 [==================>...........] - ETA: 22s - loss: 7.6466 - accuracy: 0.5013
16096/25000 [==================>...........] - ETA: 22s - loss: 7.6466 - accuracy: 0.5013
16128/25000 [==================>...........] - ETA: 22s - loss: 7.6457 - accuracy: 0.5014
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6438 - accuracy: 0.5015
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6401 - accuracy: 0.5017
16224/25000 [==================>...........] - ETA: 22s - loss: 7.6392 - accuracy: 0.5018
16256/25000 [==================>...........] - ETA: 22s - loss: 7.6412 - accuracy: 0.5017
16288/25000 [==================>...........] - ETA: 22s - loss: 7.6384 - accuracy: 0.5018
16320/25000 [==================>...........] - ETA: 21s - loss: 7.6431 - accuracy: 0.5015
16352/25000 [==================>...........] - ETA: 21s - loss: 7.6441 - accuracy: 0.5015
16384/25000 [==================>...........] - ETA: 21s - loss: 7.6404 - accuracy: 0.5017
16416/25000 [==================>...........] - ETA: 21s - loss: 7.6377 - accuracy: 0.5019
16448/25000 [==================>...........] - ETA: 21s - loss: 7.6396 - accuracy: 0.5018
16480/25000 [==================>...........] - ETA: 21s - loss: 7.6406 - accuracy: 0.5017
16512/25000 [==================>...........] - ETA: 21s - loss: 7.6406 - accuracy: 0.5017
16544/25000 [==================>...........] - ETA: 21s - loss: 7.6434 - accuracy: 0.5015
16576/25000 [==================>...........] - ETA: 21s - loss: 7.6435 - accuracy: 0.5015
16608/25000 [==================>...........] - ETA: 21s - loss: 7.6398 - accuracy: 0.5017
16640/25000 [==================>...........] - ETA: 21s - loss: 7.6399 - accuracy: 0.5017
16672/25000 [===================>..........] - ETA: 21s - loss: 7.6390 - accuracy: 0.5018
16704/25000 [===================>..........] - ETA: 20s - loss: 7.6428 - accuracy: 0.5016
16736/25000 [===================>..........] - ETA: 20s - loss: 7.6428 - accuracy: 0.5016
16768/25000 [===================>..........] - ETA: 20s - loss: 7.6410 - accuracy: 0.5017
16800/25000 [===================>..........] - ETA: 20s - loss: 7.6456 - accuracy: 0.5014
16832/25000 [===================>..........] - ETA: 20s - loss: 7.6502 - accuracy: 0.5011
16864/25000 [===================>..........] - ETA: 20s - loss: 7.6484 - accuracy: 0.5012
16896/25000 [===================>..........] - ETA: 20s - loss: 7.6503 - accuracy: 0.5011
16928/25000 [===================>..........] - ETA: 20s - loss: 7.6548 - accuracy: 0.5008
16960/25000 [===================>..........] - ETA: 20s - loss: 7.6567 - accuracy: 0.5006
16992/25000 [===================>..........] - ETA: 20s - loss: 7.6585 - accuracy: 0.5005
17024/25000 [===================>..........] - ETA: 20s - loss: 7.6531 - accuracy: 0.5009
17056/25000 [===================>..........] - ETA: 20s - loss: 7.6522 - accuracy: 0.5009
17088/25000 [===================>..........] - ETA: 20s - loss: 7.6567 - accuracy: 0.5006
17120/25000 [===================>..........] - ETA: 19s - loss: 7.6604 - accuracy: 0.5004
17152/25000 [===================>..........] - ETA: 19s - loss: 7.6630 - accuracy: 0.5002
17184/25000 [===================>..........] - ETA: 19s - loss: 7.6639 - accuracy: 0.5002
17216/25000 [===================>..........] - ETA: 19s - loss: 7.6675 - accuracy: 0.4999
17248/25000 [===================>..........] - ETA: 19s - loss: 7.6684 - accuracy: 0.4999
17280/25000 [===================>..........] - ETA: 19s - loss: 7.6648 - accuracy: 0.5001
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6702 - accuracy: 0.4998
17344/25000 [===================>..........] - ETA: 19s - loss: 7.6684 - accuracy: 0.4999
17376/25000 [===================>..........] - ETA: 19s - loss: 7.6631 - accuracy: 0.5002
17408/25000 [===================>..........] - ETA: 19s - loss: 7.6622 - accuracy: 0.5003
17440/25000 [===================>..........] - ETA: 19s - loss: 7.6675 - accuracy: 0.4999
17472/25000 [===================>..........] - ETA: 19s - loss: 7.6622 - accuracy: 0.5003
17504/25000 [====================>.........] - ETA: 18s - loss: 7.6640 - accuracy: 0.5002
17536/25000 [====================>.........] - ETA: 18s - loss: 7.6605 - accuracy: 0.5004
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6579 - accuracy: 0.5006
17600/25000 [====================>.........] - ETA: 18s - loss: 7.6605 - accuracy: 0.5004
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6605 - accuracy: 0.5004
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6571 - accuracy: 0.5006
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6597 - accuracy: 0.5005
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6571 - accuracy: 0.5006
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6511 - accuracy: 0.5010
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6554 - accuracy: 0.5007
17824/25000 [====================>.........] - ETA: 18s - loss: 7.6520 - accuracy: 0.5010
17856/25000 [====================>.........] - ETA: 18s - loss: 7.6512 - accuracy: 0.5010
17888/25000 [====================>.........] - ETA: 17s - loss: 7.6486 - accuracy: 0.5012
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6487 - accuracy: 0.5012
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6521 - accuracy: 0.5009
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6496 - accuracy: 0.5011
18016/25000 [====================>.........] - ETA: 17s - loss: 7.6522 - accuracy: 0.5009
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6564 - accuracy: 0.5007
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6581 - accuracy: 0.5006
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6556 - accuracy: 0.5007
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6523 - accuracy: 0.5009
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6531 - accuracy: 0.5009
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6531 - accuracy: 0.5009
18240/25000 [====================>.........] - ETA: 17s - loss: 7.6540 - accuracy: 0.5008
18272/25000 [====================>.........] - ETA: 16s - loss: 7.6490 - accuracy: 0.5011
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6490 - accuracy: 0.5011
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6474 - accuracy: 0.5013
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6457 - accuracy: 0.5014
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6450 - accuracy: 0.5014
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6491 - accuracy: 0.5011
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6467 - accuracy: 0.5013
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6476 - accuracy: 0.5012
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6509 - accuracy: 0.5010
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6526 - accuracy: 0.5009
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6542 - accuracy: 0.5008
18624/25000 [=====================>........] - ETA: 16s - loss: 7.6551 - accuracy: 0.5008
18656/25000 [=====================>........] - ETA: 16s - loss: 7.6568 - accuracy: 0.5006
18688/25000 [=====================>........] - ETA: 15s - loss: 7.6568 - accuracy: 0.5006
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6617 - accuracy: 0.5003
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6642 - accuracy: 0.5002
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6642 - accuracy: 0.5002
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6674 - accuracy: 0.4999
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6650 - accuracy: 0.5001
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6674 - accuracy: 0.4999
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6682 - accuracy: 0.4999
19040/25000 [=====================>........] - ETA: 15s - loss: 7.6706 - accuracy: 0.4997
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6706 - accuracy: 0.4997
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6771 - accuracy: 0.4993
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6778 - accuracy: 0.4993
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6786 - accuracy: 0.4992
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6770 - accuracy: 0.4993
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6770 - accuracy: 0.4993
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6802 - accuracy: 0.4991
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6785 - accuracy: 0.4992
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6753 - accuracy: 0.4994
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6737 - accuracy: 0.4995
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6761 - accuracy: 0.4994
19424/25000 [======================>.......] - ETA: 14s - loss: 7.6729 - accuracy: 0.4996
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6737 - accuracy: 0.4995
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6729 - accuracy: 0.4996
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6737 - accuracy: 0.4995
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6737 - accuracy: 0.4995
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6760 - accuracy: 0.4994
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6760 - accuracy: 0.4994
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6744 - accuracy: 0.4995
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6744 - accuracy: 0.4995
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6736 - accuracy: 0.4995
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6759 - accuracy: 0.4994
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6767 - accuracy: 0.4993
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6767 - accuracy: 0.4993
19840/25000 [======================>.......] - ETA: 13s - loss: 7.6798 - accuracy: 0.4991
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6767 - accuracy: 0.4993
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6774 - accuracy: 0.4993
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6766 - accuracy: 0.4993
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6774 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6751 - accuracy: 0.4994
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6758 - accuracy: 0.4994
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6750 - accuracy: 0.4995
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6735 - accuracy: 0.4996
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6697 - accuracy: 0.4998
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6666 - accuracy: 0.5000
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6659 - accuracy: 0.5000
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6643 - accuracy: 0.5001
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6636 - accuracy: 0.5002
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6644 - accuracy: 0.5001
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6674 - accuracy: 0.5000
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6689 - accuracy: 0.4999
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6696 - accuracy: 0.4998
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6696 - accuracy: 0.4998
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6681 - accuracy: 0.4999
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6636 - accuracy: 0.5002
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6621 - accuracy: 0.5003
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6606 - accuracy: 0.5004
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6607 - accuracy: 0.5004
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6592 - accuracy: 0.5005
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6629 - accuracy: 0.5002
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6614 - accuracy: 0.5003
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6592 - accuracy: 0.5005
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6570 - accuracy: 0.5006
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6570 - accuracy: 0.5006
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6585 - accuracy: 0.5005
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6585 - accuracy: 0.5005
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6593 - accuracy: 0.5005
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6585 - accuracy: 0.5005
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6578 - accuracy: 0.5006
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6586 - accuracy: 0.5005
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6579 - accuracy: 0.5006
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6550 - accuracy: 0.5008
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6564 - accuracy: 0.5007 
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6586 - accuracy: 0.5005
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6572 - accuracy: 0.5006
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6572 - accuracy: 0.5006
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6579 - accuracy: 0.5006
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6565 - accuracy: 0.5007
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6565 - accuracy: 0.5007
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6551 - accuracy: 0.5008
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6544 - accuracy: 0.5008
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6558 - accuracy: 0.5007
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6559 - accuracy: 0.5007
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6595 - accuracy: 0.5005
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6580 - accuracy: 0.5006
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6588 - accuracy: 0.5005
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6595 - accuracy: 0.5005
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6595 - accuracy: 0.5005
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6616 - accuracy: 0.5003
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6602 - accuracy: 0.5004
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6602 - accuracy: 0.5004
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6610 - accuracy: 0.5004
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6567 - accuracy: 0.5006
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6539 - accuracy: 0.5008
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6546 - accuracy: 0.5008
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6504 - accuracy: 0.5011
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6505 - accuracy: 0.5011
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6477 - accuracy: 0.5012
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6498 - accuracy: 0.5011
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6477 - accuracy: 0.5012
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6457 - accuracy: 0.5014
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6471 - accuracy: 0.5013
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6471 - accuracy: 0.5013
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6458 - accuracy: 0.5014
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6458 - accuracy: 0.5014
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6465 - accuracy: 0.5013
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6452 - accuracy: 0.5014
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6473 - accuracy: 0.5013
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6480 - accuracy: 0.5012
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6452 - accuracy: 0.5014
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6460 - accuracy: 0.5013
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6446 - accuracy: 0.5014
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6453 - accuracy: 0.5014
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6433 - accuracy: 0.5015
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6406 - accuracy: 0.5017
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6406 - accuracy: 0.5017
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6380 - accuracy: 0.5019
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6394 - accuracy: 0.5018
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6353 - accuracy: 0.5020
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6367 - accuracy: 0.5020
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6374 - accuracy: 0.5019
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6327 - accuracy: 0.5022
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6389 - accuracy: 0.5018
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6396 - accuracy: 0.5018
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6430 - accuracy: 0.5015
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6444 - accuracy: 0.5015
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6484 - accuracy: 0.5012
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6465 - accuracy: 0.5013
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6465 - accuracy: 0.5013
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6479 - accuracy: 0.5012
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6465 - accuracy: 0.5013
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6499 - accuracy: 0.5011
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6499 - accuracy: 0.5011
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6506 - accuracy: 0.5010
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6480 - accuracy: 0.5012
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6487 - accuracy: 0.5012
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6460 - accuracy: 0.5013
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6447 - accuracy: 0.5014
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6461 - accuracy: 0.5013
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6461 - accuracy: 0.5013
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6462 - accuracy: 0.5013
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6429 - accuracy: 0.5015
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6436 - accuracy: 0.5015
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6430 - accuracy: 0.5015
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6456 - accuracy: 0.5014
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6470 - accuracy: 0.5013
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6509 - accuracy: 0.5010
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6503 - accuracy: 0.5011
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6483 - accuracy: 0.5012
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6477 - accuracy: 0.5012
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6477 - accuracy: 0.5012
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6484 - accuracy: 0.5012
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6517 - accuracy: 0.5010
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6498 - accuracy: 0.5011
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6511 - accuracy: 0.5010
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6524 - accuracy: 0.5009
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6518 - accuracy: 0.5010
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6492 - accuracy: 0.5011
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6505 - accuracy: 0.5011
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6467 - accuracy: 0.5013
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6467 - accuracy: 0.5013
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6480 - accuracy: 0.5012
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6474 - accuracy: 0.5013
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6481 - accuracy: 0.5012
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6500 - accuracy: 0.5011
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6539 - accuracy: 0.5008
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6545 - accuracy: 0.5008
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6564 - accuracy: 0.5007
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6565 - accuracy: 0.5007
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6546 - accuracy: 0.5008
24192/25000 [============================>.] - ETA: 2s - loss: 7.6533 - accuracy: 0.5009
24224/25000 [============================>.] - ETA: 1s - loss: 7.6514 - accuracy: 0.5010
24256/25000 [============================>.] - ETA: 1s - loss: 7.6521 - accuracy: 0.5009
24288/25000 [============================>.] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
24320/25000 [============================>.] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
24352/25000 [============================>.] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
24384/25000 [============================>.] - ETA: 1s - loss: 7.6540 - accuracy: 0.5008
24416/25000 [============================>.] - ETA: 1s - loss: 7.6572 - accuracy: 0.5006
24448/25000 [============================>.] - ETA: 1s - loss: 7.6572 - accuracy: 0.5006
24480/25000 [============================>.] - ETA: 1s - loss: 7.6579 - accuracy: 0.5006
24512/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5004
24544/25000 [============================>.] - ETA: 1s - loss: 7.6585 - accuracy: 0.5005
24576/25000 [============================>.] - ETA: 1s - loss: 7.6598 - accuracy: 0.5004
24608/25000 [============================>.] - ETA: 0s - loss: 7.6585 - accuracy: 0.5005
24640/25000 [============================>.] - ETA: 0s - loss: 7.6592 - accuracy: 0.5005
24672/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24704/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
24736/25000 [============================>.] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24768/25000 [============================>.] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24800/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24832/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24864/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24896/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24928/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 74s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
