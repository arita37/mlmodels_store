
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
	Data preprocessing and feature engineering runtime = 0.26s ...
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
 40%|████      | 2/5 [00:53<01:20, 26.69s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.006632954430099982, 'embedding_size_factor': 0.9415358535164083, 'layers.choice': 1, 'learning_rate': 0.0007322059307774688, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 3.1180619388086894e-12} and reward: 0.3718
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?{+(%\xa6s\xc2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xee!\x0f\xcc[\xa9UX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?G\xfe0Ch\x7f\x01X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x8bm@T\xba\xbb\xccu.' and reward: 0.3718
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?{+(%\xa6s\xc2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xee!\x0f\xcc[\xa9UX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?G\xfe0Ch\x7f\x01X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x8bm@T\xba\xbb\xccu.' and reward: 0.3718
 60%|██████    | 3/5 [02:35<01:38, 49.34s/it] 60%|██████    | 3/5 [02:35<01:43, 51.86s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.4277224658178865, 'embedding_size_factor': 0.5463592601550935, 'layers.choice': 1, 'learning_rate': 0.00013403931638030734, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 2.587630887628693e-12} and reward: 0.3392
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdb_\xce\x0c\x9c\xf2}X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe1{\xc6jGs\xc6X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?!\x91\x9c\xf5\xe0\xeb$X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\x86\xc2\xd3\xa5\xee5/u.' and reward: 0.3392
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdb_\xce\x0c\x9c\xf2}X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe1{\xc6jGs\xc6X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?!\x91\x9c\xf5\xe0\xeb$X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\x86\xc2\xd3\xa5\xee5/u.' and reward: 0.3392
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 224.31980156898499
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -107.23s of remaining time.
Ensemble size: 85
Ensemble weights: 
[0.49411765 0.38823529 0.11764706]
	0.3896	 = Validation accuracy score
	1.06s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 228.33s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f3671b94a20> 

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
 [-0.03366943 -0.02538997  0.04045422 -0.03021277 -0.02330705 -0.13201375]
 [ 0.14260802  0.14906181  0.11005381 -0.08798295 -0.00319643 -0.14885265]
 [-0.01610222  0.13814078  0.21591504  0.14058922  0.11650528  0.0284504 ]
 [ 0.1459929   0.09500794  0.03489681  0.2993488   0.062101    0.06811549]
 [-0.20593664  0.20654325  0.19237946 -0.08680747  0.06382453  0.30498835]
 [ 0.12634926 -0.18684606  0.02760767  0.14807305  0.20781298  0.20589858]
 [ 0.17062393  0.39944577 -0.07041302  0.13621198  0.62445533  0.50281924]
 [ 0.31450227  0.049162    0.05066895 -0.07857487  0.4277643  -0.37533981]
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
{'loss': 0.5048704966902733, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-17 00:27:46.491010: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.5063463225960732, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-17 00:27:47.674135: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 1556480/17464789 [=>............................] - ETA: 0s
 5570560/17464789 [========>.....................] - ETA: 0s
10813440/17464789 [=================>............] - ETA: 0s
13942784/17464789 [======================>.......] - ETA: 0s
17219584/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-17 00:27:59.670959: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-17 00:27:59.675440: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-17 00:27:59.676250: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5613038ab090 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-17 00:27:59.676266: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:58 - loss: 7.6666 - accuracy: 0.5000
   64/25000 [..............................] - ETA: 3:11 - loss: 7.1875 - accuracy: 0.5312
   96/25000 [..............................] - ETA: 2:35 - loss: 8.4652 - accuracy: 0.4479
  128/25000 [..............................] - ETA: 2:18 - loss: 8.6249 - accuracy: 0.4375
  160/25000 [..............................] - ETA: 2:06 - loss: 8.4333 - accuracy: 0.4500
  192/25000 [..............................] - ETA: 1:59 - loss: 8.3854 - accuracy: 0.4531
  224/25000 [..............................] - ETA: 1:54 - loss: 8.2827 - accuracy: 0.4598
  256/25000 [..............................] - ETA: 1:50 - loss: 8.3255 - accuracy: 0.4570
  288/25000 [..............................] - ETA: 1:47 - loss: 8.4652 - accuracy: 0.4479
  320/25000 [..............................] - ETA: 1:45 - loss: 8.3374 - accuracy: 0.4563
  352/25000 [..............................] - ETA: 1:43 - loss: 8.1893 - accuracy: 0.4659
  384/25000 [..............................] - ETA: 1:41 - loss: 8.2656 - accuracy: 0.4609
  416/25000 [..............................] - ETA: 1:39 - loss: 8.1826 - accuracy: 0.4663
  448/25000 [..............................] - ETA: 1:38 - loss: 8.1458 - accuracy: 0.4688
  480/25000 [..............................] - ETA: 1:37 - loss: 8.1777 - accuracy: 0.4667
  512/25000 [..............................] - ETA: 1:36 - loss: 8.1757 - accuracy: 0.4668
  544/25000 [..............................] - ETA: 1:35 - loss: 8.0612 - accuracy: 0.4743
  576/25000 [..............................] - ETA: 1:34 - loss: 8.0659 - accuracy: 0.4740
  608/25000 [..............................] - ETA: 1:34 - loss: 8.2971 - accuracy: 0.4589
  640/25000 [..............................] - ETA: 1:33 - loss: 8.2895 - accuracy: 0.4594
  672/25000 [..............................] - ETA: 1:33 - loss: 8.1230 - accuracy: 0.4702
  704/25000 [..............................] - ETA: 1:32 - loss: 8.0369 - accuracy: 0.4759
  736/25000 [..............................] - ETA: 1:32 - loss: 8.0208 - accuracy: 0.4769
  768/25000 [..............................] - ETA: 1:31 - loss: 8.0060 - accuracy: 0.4779
  800/25000 [..............................] - ETA: 1:31 - loss: 7.9541 - accuracy: 0.4812
  832/25000 [..............................] - ETA: 1:30 - loss: 7.9246 - accuracy: 0.4832
  864/25000 [>.............................] - ETA: 1:30 - loss: 7.9328 - accuracy: 0.4826
  896/25000 [>.............................] - ETA: 1:29 - loss: 7.9404 - accuracy: 0.4821
  928/25000 [>.............................] - ETA: 1:29 - loss: 7.9310 - accuracy: 0.4828
  960/25000 [>.............................] - ETA: 1:29 - loss: 7.9062 - accuracy: 0.4844
  992/25000 [>.............................] - ETA: 1:29 - loss: 7.8830 - accuracy: 0.4859
 1024/25000 [>.............................] - ETA: 1:28 - loss: 7.8463 - accuracy: 0.4883
 1056/25000 [>.............................] - ETA: 1:28 - loss: 7.8409 - accuracy: 0.4886
 1088/25000 [>.............................] - ETA: 1:28 - loss: 7.8216 - accuracy: 0.4899
 1120/25000 [>.............................] - ETA: 1:28 - loss: 7.7898 - accuracy: 0.4920
 1152/25000 [>.............................] - ETA: 1:28 - loss: 7.7731 - accuracy: 0.4931
 1184/25000 [>.............................] - ETA: 1:27 - loss: 7.7055 - accuracy: 0.4975
 1216/25000 [>.............................] - ETA: 1:27 - loss: 7.7297 - accuracy: 0.4959
 1248/25000 [>.............................] - ETA: 1:27 - loss: 7.7649 - accuracy: 0.4936
 1280/25000 [>.............................] - ETA: 1:26 - loss: 7.7625 - accuracy: 0.4938
 1312/25000 [>.............................] - ETA: 1:26 - loss: 7.7601 - accuracy: 0.4939
 1344/25000 [>.............................] - ETA: 1:26 - loss: 7.7579 - accuracy: 0.4940
 1376/25000 [>.............................] - ETA: 1:26 - loss: 7.8003 - accuracy: 0.4913
 1408/25000 [>.............................] - ETA: 1:26 - loss: 7.7537 - accuracy: 0.4943
 1440/25000 [>.............................] - ETA: 1:25 - loss: 7.7518 - accuracy: 0.4944
 1472/25000 [>.............................] - ETA: 1:25 - loss: 7.7604 - accuracy: 0.4939
 1504/25000 [>.............................] - ETA: 1:25 - loss: 7.7788 - accuracy: 0.4927
 1536/25000 [>.............................] - ETA: 1:25 - loss: 7.7365 - accuracy: 0.4954
 1568/25000 [>.............................] - ETA: 1:25 - loss: 7.7546 - accuracy: 0.4943
 1600/25000 [>.............................] - ETA: 1:25 - loss: 7.7337 - accuracy: 0.4956
 1632/25000 [>.............................] - ETA: 1:25 - loss: 7.6948 - accuracy: 0.4982
 1664/25000 [>.............................] - ETA: 1:25 - loss: 7.7219 - accuracy: 0.4964
 1696/25000 [=>............................] - ETA: 1:24 - loss: 7.7209 - accuracy: 0.4965
 1728/25000 [=>............................] - ETA: 1:24 - loss: 7.7376 - accuracy: 0.4954
 1760/25000 [=>............................] - ETA: 1:24 - loss: 7.7276 - accuracy: 0.4960
 1792/25000 [=>............................] - ETA: 1:24 - loss: 7.7351 - accuracy: 0.4955
 1824/25000 [=>............................] - ETA: 1:24 - loss: 7.6918 - accuracy: 0.4984
 1856/25000 [=>............................] - ETA: 1:24 - loss: 7.6914 - accuracy: 0.4984
 1888/25000 [=>............................] - ETA: 1:23 - loss: 7.6747 - accuracy: 0.4995
 1920/25000 [=>............................] - ETA: 1:23 - loss: 7.6826 - accuracy: 0.4990
 1952/25000 [=>............................] - ETA: 1:23 - loss: 7.6823 - accuracy: 0.4990
 1984/25000 [=>............................] - ETA: 1:23 - loss: 7.6821 - accuracy: 0.4990
 2016/25000 [=>............................] - ETA: 1:23 - loss: 7.6742 - accuracy: 0.4995
 2048/25000 [=>............................] - ETA: 1:23 - loss: 7.6666 - accuracy: 0.5000
 2080/25000 [=>............................] - ETA: 1:22 - loss: 7.6445 - accuracy: 0.5014
 2112/25000 [=>............................] - ETA: 1:22 - loss: 7.6594 - accuracy: 0.5005
 2144/25000 [=>............................] - ETA: 1:22 - loss: 7.6738 - accuracy: 0.4995
 2176/25000 [=>............................] - ETA: 1:22 - loss: 7.6525 - accuracy: 0.5009
 2208/25000 [=>............................] - ETA: 1:22 - loss: 7.6527 - accuracy: 0.5009
 2240/25000 [=>............................] - ETA: 1:22 - loss: 7.6529 - accuracy: 0.5009
 2272/25000 [=>............................] - ETA: 1:21 - loss: 7.6531 - accuracy: 0.5009
 2304/25000 [=>............................] - ETA: 1:21 - loss: 7.6666 - accuracy: 0.5000
 2336/25000 [=>............................] - ETA: 1:21 - loss: 7.6863 - accuracy: 0.4987
 2368/25000 [=>............................] - ETA: 1:21 - loss: 7.6601 - accuracy: 0.5004
 2400/25000 [=>............................] - ETA: 1:21 - loss: 7.6347 - accuracy: 0.5021
 2432/25000 [=>............................] - ETA: 1:21 - loss: 7.6225 - accuracy: 0.5029
 2464/25000 [=>............................] - ETA: 1:21 - loss: 7.6231 - accuracy: 0.5028
 2496/25000 [=>............................] - ETA: 1:21 - loss: 7.6113 - accuracy: 0.5036
 2528/25000 [==>...........................] - ETA: 1:21 - loss: 7.6120 - accuracy: 0.5036
 2560/25000 [==>...........................] - ETA: 1:21 - loss: 7.5947 - accuracy: 0.5047
 2592/25000 [==>...........................] - ETA: 1:20 - loss: 7.6015 - accuracy: 0.5042
 2624/25000 [==>...........................] - ETA: 1:20 - loss: 7.5848 - accuracy: 0.5053
 2656/25000 [==>...........................] - ETA: 1:20 - loss: 7.5512 - accuracy: 0.5075
 2688/25000 [==>...........................] - ETA: 1:20 - loss: 7.5639 - accuracy: 0.5067
 2720/25000 [==>...........................] - ETA: 1:20 - loss: 7.5877 - accuracy: 0.5051
 2752/25000 [==>...........................] - ETA: 1:20 - loss: 7.5830 - accuracy: 0.5055
 2784/25000 [==>...........................] - ETA: 1:20 - loss: 7.5565 - accuracy: 0.5072
 2816/25000 [==>...........................] - ETA: 1:19 - loss: 7.5359 - accuracy: 0.5085
 2848/25000 [==>...........................] - ETA: 1:19 - loss: 7.5213 - accuracy: 0.5095
 2880/25000 [==>...........................] - ETA: 1:19 - loss: 7.5016 - accuracy: 0.5108
 2912/25000 [==>...........................] - ETA: 1:19 - loss: 7.5139 - accuracy: 0.5100
 2944/25000 [==>...........................] - ETA: 1:19 - loss: 7.5156 - accuracy: 0.5099
 2976/25000 [==>...........................] - ETA: 1:19 - loss: 7.5172 - accuracy: 0.5097
 3008/25000 [==>...........................] - ETA: 1:18 - loss: 7.5188 - accuracy: 0.5096
 3040/25000 [==>...........................] - ETA: 1:18 - loss: 7.5052 - accuracy: 0.5105
 3072/25000 [==>...........................] - ETA: 1:18 - loss: 7.4869 - accuracy: 0.5117
 3104/25000 [==>...........................] - ETA: 1:18 - loss: 7.4937 - accuracy: 0.5113
 3136/25000 [==>...........................] - ETA: 1:18 - loss: 7.4906 - accuracy: 0.5115
 3168/25000 [==>...........................] - ETA: 1:18 - loss: 7.4779 - accuracy: 0.5123
 3200/25000 [==>...........................] - ETA: 1:18 - loss: 7.4941 - accuracy: 0.5113
 3232/25000 [==>...........................] - ETA: 1:18 - loss: 7.4958 - accuracy: 0.5111
 3264/25000 [==>...........................] - ETA: 1:17 - loss: 7.4928 - accuracy: 0.5113
 3296/25000 [==>...........................] - ETA: 1:17 - loss: 7.4759 - accuracy: 0.5124
 3328/25000 [==>...........................] - ETA: 1:17 - loss: 7.4777 - accuracy: 0.5123
 3360/25000 [===>..........................] - ETA: 1:17 - loss: 7.4841 - accuracy: 0.5119
 3392/25000 [===>..........................] - ETA: 1:17 - loss: 7.4858 - accuracy: 0.5118
 3424/25000 [===>..........................] - ETA: 1:17 - loss: 7.4785 - accuracy: 0.5123
 3456/25000 [===>..........................] - ETA: 1:17 - loss: 7.4936 - accuracy: 0.5113
 3488/25000 [===>..........................] - ETA: 1:17 - loss: 7.4996 - accuracy: 0.5109
 3520/25000 [===>..........................] - ETA: 1:16 - loss: 7.4967 - accuracy: 0.5111
 3552/25000 [===>..........................] - ETA: 1:16 - loss: 7.4939 - accuracy: 0.5113
 3584/25000 [===>..........................] - ETA: 1:16 - loss: 7.4955 - accuracy: 0.5112
 3616/25000 [===>..........................] - ETA: 1:16 - loss: 7.5140 - accuracy: 0.5100
 3648/25000 [===>..........................] - ETA: 1:16 - loss: 7.5195 - accuracy: 0.5096
 3680/25000 [===>..........................] - ETA: 1:16 - loss: 7.5416 - accuracy: 0.5082
 3712/25000 [===>..........................] - ETA: 1:16 - loss: 7.5468 - accuracy: 0.5078
 3744/25000 [===>..........................] - ETA: 1:16 - loss: 7.5479 - accuracy: 0.5077
 3776/25000 [===>..........................] - ETA: 1:16 - loss: 7.5610 - accuracy: 0.5069
 3808/25000 [===>..........................] - ETA: 1:15 - loss: 7.5941 - accuracy: 0.5047
 3840/25000 [===>..........................] - ETA: 1:15 - loss: 7.6027 - accuracy: 0.5042
 3872/25000 [===>..........................] - ETA: 1:15 - loss: 7.5993 - accuracy: 0.5044
 3904/25000 [===>..........................] - ETA: 1:15 - loss: 7.6273 - accuracy: 0.5026
 3936/25000 [===>..........................] - ETA: 1:15 - loss: 7.6238 - accuracy: 0.5028
 3968/25000 [===>..........................] - ETA: 1:15 - loss: 7.6280 - accuracy: 0.5025
 4000/25000 [===>..........................] - ETA: 1:15 - loss: 7.6283 - accuracy: 0.5025
 4032/25000 [===>..........................] - ETA: 1:15 - loss: 7.6362 - accuracy: 0.5020
 4064/25000 [===>..........................] - ETA: 1:15 - loss: 7.6176 - accuracy: 0.5032
 4096/25000 [===>..........................] - ETA: 1:14 - loss: 7.6217 - accuracy: 0.5029
 4128/25000 [===>..........................] - ETA: 1:14 - loss: 7.6072 - accuracy: 0.5039
 4160/25000 [===>..........................] - ETA: 1:14 - loss: 7.6003 - accuracy: 0.5043
 4192/25000 [====>.........................] - ETA: 1:14 - loss: 7.6044 - accuracy: 0.5041
 4224/25000 [====>.........................] - ETA: 1:14 - loss: 7.5940 - accuracy: 0.5047
 4256/25000 [====>.........................] - ETA: 1:14 - loss: 7.6018 - accuracy: 0.5042
 4288/25000 [====>.........................] - ETA: 1:14 - loss: 7.6130 - accuracy: 0.5035
 4320/25000 [====>.........................] - ETA: 1:14 - loss: 7.6134 - accuracy: 0.5035
 4352/25000 [====>.........................] - ETA: 1:14 - loss: 7.6102 - accuracy: 0.5037
 4384/25000 [====>.........................] - ETA: 1:13 - loss: 7.6072 - accuracy: 0.5039
 4416/25000 [====>.........................] - ETA: 1:13 - loss: 7.6076 - accuracy: 0.5038
 4448/25000 [====>.........................] - ETA: 1:13 - loss: 7.6115 - accuracy: 0.5036
 4480/25000 [====>.........................] - ETA: 1:13 - loss: 7.6187 - accuracy: 0.5031
 4512/25000 [====>.........................] - ETA: 1:13 - loss: 7.6258 - accuracy: 0.5027
 4544/25000 [====>.........................] - ETA: 1:13 - loss: 7.6194 - accuracy: 0.5031
 4576/25000 [====>.........................] - ETA: 1:13 - loss: 7.6231 - accuracy: 0.5028
 4608/25000 [====>.........................] - ETA: 1:13 - loss: 7.6234 - accuracy: 0.5028
 4640/25000 [====>.........................] - ETA: 1:12 - loss: 7.6237 - accuracy: 0.5028
 4672/25000 [====>.........................] - ETA: 1:12 - loss: 7.6174 - accuracy: 0.5032
 4704/25000 [====>.........................] - ETA: 1:12 - loss: 7.6145 - accuracy: 0.5034
 4736/25000 [====>.........................] - ETA: 1:12 - loss: 7.6051 - accuracy: 0.5040
 4768/25000 [====>.........................] - ETA: 1:12 - loss: 7.6119 - accuracy: 0.5036
 4800/25000 [====>.........................] - ETA: 1:12 - loss: 7.6219 - accuracy: 0.5029
 4832/25000 [====>.........................] - ETA: 1:12 - loss: 7.6190 - accuracy: 0.5031
 4864/25000 [====>.........................] - ETA: 1:12 - loss: 7.6130 - accuracy: 0.5035
 4896/25000 [====>.........................] - ETA: 1:12 - loss: 7.6259 - accuracy: 0.5027
 4928/25000 [====>.........................] - ETA: 1:11 - loss: 7.6231 - accuracy: 0.5028
 4960/25000 [====>.........................] - ETA: 1:11 - loss: 7.6141 - accuracy: 0.5034
 4992/25000 [====>.........................] - ETA: 1:11 - loss: 7.6144 - accuracy: 0.5034
 5024/25000 [=====>........................] - ETA: 1:11 - loss: 7.6208 - accuracy: 0.5030
 5056/25000 [=====>........................] - ETA: 1:11 - loss: 7.6151 - accuracy: 0.5034
 5088/25000 [=====>........................] - ETA: 1:11 - loss: 7.6305 - accuracy: 0.5024
 5120/25000 [=====>........................] - ETA: 1:11 - loss: 7.6217 - accuracy: 0.5029
 5152/25000 [=====>........................] - ETA: 1:11 - loss: 7.6250 - accuracy: 0.5027
 5184/25000 [=====>........................] - ETA: 1:11 - loss: 7.6282 - accuracy: 0.5025
 5216/25000 [=====>........................] - ETA: 1:10 - loss: 7.6255 - accuracy: 0.5027
 5248/25000 [=====>........................] - ETA: 1:10 - loss: 7.6403 - accuracy: 0.5017
 5280/25000 [=====>........................] - ETA: 1:10 - loss: 7.6434 - accuracy: 0.5015
 5312/25000 [=====>........................] - ETA: 1:10 - loss: 7.6349 - accuracy: 0.5021
 5344/25000 [=====>........................] - ETA: 1:10 - loss: 7.6293 - accuracy: 0.5024
 5376/25000 [=====>........................] - ETA: 1:10 - loss: 7.6295 - accuracy: 0.5024
 5408/25000 [=====>........................] - ETA: 1:10 - loss: 7.6241 - accuracy: 0.5028
 5440/25000 [=====>........................] - ETA: 1:10 - loss: 7.6356 - accuracy: 0.5020
 5472/25000 [=====>........................] - ETA: 1:09 - loss: 7.6470 - accuracy: 0.5013
 5504/25000 [=====>........................] - ETA: 1:09 - loss: 7.6415 - accuracy: 0.5016
 5536/25000 [=====>........................] - ETA: 1:09 - loss: 7.6389 - accuracy: 0.5018
 5568/25000 [=====>........................] - ETA: 1:09 - loss: 7.6391 - accuracy: 0.5018
 5600/25000 [=====>........................] - ETA: 1:09 - loss: 7.6365 - accuracy: 0.5020
 5632/25000 [=====>........................] - ETA: 1:09 - loss: 7.6285 - accuracy: 0.5025
 5664/25000 [=====>........................] - ETA: 1:09 - loss: 7.6233 - accuracy: 0.5028
 5696/25000 [=====>........................] - ETA: 1:09 - loss: 7.6209 - accuracy: 0.5030
 5728/25000 [=====>........................] - ETA: 1:09 - loss: 7.6291 - accuracy: 0.5024
 5760/25000 [=====>........................] - ETA: 1:08 - loss: 7.6267 - accuracy: 0.5026
 5792/25000 [=====>........................] - ETA: 1:08 - loss: 7.6375 - accuracy: 0.5019
 5824/25000 [=====>........................] - ETA: 1:08 - loss: 7.6482 - accuracy: 0.5012
 5856/25000 [======>.......................] - ETA: 1:08 - loss: 7.6561 - accuracy: 0.5007
 5888/25000 [======>.......................] - ETA: 1:08 - loss: 7.6510 - accuracy: 0.5010
 5920/25000 [======>.......................] - ETA: 1:08 - loss: 7.6511 - accuracy: 0.5010
 5952/25000 [======>.......................] - ETA: 1:08 - loss: 7.6512 - accuracy: 0.5010
 5984/25000 [======>.......................] - ETA: 1:08 - loss: 7.6512 - accuracy: 0.5010
 6016/25000 [======>.......................] - ETA: 1:08 - loss: 7.6615 - accuracy: 0.5003
 6048/25000 [======>.......................] - ETA: 1:08 - loss: 7.6438 - accuracy: 0.5015
 6080/25000 [======>.......................] - ETA: 1:07 - loss: 7.6414 - accuracy: 0.5016
 6112/25000 [======>.......................] - ETA: 1:07 - loss: 7.6465 - accuracy: 0.5013
 6144/25000 [======>.......................] - ETA: 1:07 - loss: 7.6591 - accuracy: 0.5005
 6176/25000 [======>.......................] - ETA: 1:07 - loss: 7.6617 - accuracy: 0.5003
 6208/25000 [======>.......................] - ETA: 1:07 - loss: 7.6666 - accuracy: 0.5000
 6240/25000 [======>.......................] - ETA: 1:07 - loss: 7.6691 - accuracy: 0.4998
 6272/25000 [======>.......................] - ETA: 1:07 - loss: 7.6715 - accuracy: 0.4997
 6304/25000 [======>.......................] - ETA: 1:07 - loss: 7.6569 - accuracy: 0.5006
 6336/25000 [======>.......................] - ETA: 1:06 - loss: 7.6594 - accuracy: 0.5005
 6368/25000 [======>.......................] - ETA: 1:06 - loss: 7.6594 - accuracy: 0.5005
 6400/25000 [======>.......................] - ETA: 1:06 - loss: 7.6498 - accuracy: 0.5011
 6432/25000 [======>.......................] - ETA: 1:06 - loss: 7.6380 - accuracy: 0.5019
 6464/25000 [======>.......................] - ETA: 1:06 - loss: 7.6358 - accuracy: 0.5020
 6496/25000 [======>.......................] - ETA: 1:06 - loss: 7.6383 - accuracy: 0.5018
 6528/25000 [======>.......................] - ETA: 1:06 - loss: 7.6431 - accuracy: 0.5015
 6560/25000 [======>.......................] - ETA: 1:06 - loss: 7.6432 - accuracy: 0.5015
 6592/25000 [======>.......................] - ETA: 1:06 - loss: 7.6387 - accuracy: 0.5018
 6624/25000 [======>.......................] - ETA: 1:05 - loss: 7.6412 - accuracy: 0.5017
 6656/25000 [======>.......................] - ETA: 1:05 - loss: 7.6298 - accuracy: 0.5024
 6688/25000 [=======>......................] - ETA: 1:05 - loss: 7.6414 - accuracy: 0.5016
 6720/25000 [=======>......................] - ETA: 1:05 - loss: 7.6461 - accuracy: 0.5013
 6752/25000 [=======>......................] - ETA: 1:05 - loss: 7.6530 - accuracy: 0.5009
 6784/25000 [=======>......................] - ETA: 1:05 - loss: 7.6553 - accuracy: 0.5007
 6816/25000 [=======>......................] - ETA: 1:05 - loss: 7.6351 - accuracy: 0.5021
 6848/25000 [=======>......................] - ETA: 1:05 - loss: 7.6420 - accuracy: 0.5016
 6880/25000 [=======>......................] - ETA: 1:05 - loss: 7.6354 - accuracy: 0.5020
 6912/25000 [=======>......................] - ETA: 1:04 - loss: 7.6311 - accuracy: 0.5023
 6944/25000 [=======>......................] - ETA: 1:04 - loss: 7.6225 - accuracy: 0.5029
 6976/25000 [=======>......................] - ETA: 1:04 - loss: 7.6271 - accuracy: 0.5026
 7008/25000 [=======>......................] - ETA: 1:04 - loss: 7.6294 - accuracy: 0.5024
 7040/25000 [=======>......................] - ETA: 1:04 - loss: 7.6274 - accuracy: 0.5026
 7072/25000 [=======>......................] - ETA: 1:04 - loss: 7.6276 - accuracy: 0.5025
 7104/25000 [=======>......................] - ETA: 1:04 - loss: 7.6148 - accuracy: 0.5034
 7136/25000 [=======>......................] - ETA: 1:04 - loss: 7.6236 - accuracy: 0.5028
 7168/25000 [=======>......................] - ETA: 1:04 - loss: 7.6196 - accuracy: 0.5031
 7200/25000 [=======>......................] - ETA: 1:03 - loss: 7.6112 - accuracy: 0.5036
 7232/25000 [=======>......................] - ETA: 1:03 - loss: 7.6136 - accuracy: 0.5035
 7264/25000 [=======>......................] - ETA: 1:03 - loss: 7.6160 - accuracy: 0.5033
 7296/25000 [=======>......................] - ETA: 1:03 - loss: 7.6225 - accuracy: 0.5029
 7328/25000 [=======>......................] - ETA: 1:03 - loss: 7.6269 - accuracy: 0.5026
 7360/25000 [=======>......................] - ETA: 1:03 - loss: 7.6333 - accuracy: 0.5022
 7392/25000 [=======>......................] - ETA: 1:03 - loss: 7.6417 - accuracy: 0.5016
 7424/25000 [=======>......................] - ETA: 1:03 - loss: 7.6398 - accuracy: 0.5018
 7456/25000 [=======>......................] - ETA: 1:03 - loss: 7.6522 - accuracy: 0.5009
 7488/25000 [=======>......................] - ETA: 1:02 - loss: 7.6523 - accuracy: 0.5009
 7520/25000 [========>.....................] - ETA: 1:02 - loss: 7.6483 - accuracy: 0.5012
 7552/25000 [========>.....................] - ETA: 1:02 - loss: 7.6565 - accuracy: 0.5007
 7584/25000 [========>.....................] - ETA: 1:02 - loss: 7.6504 - accuracy: 0.5011
 7616/25000 [========>.....................] - ETA: 1:02 - loss: 7.6505 - accuracy: 0.5011
 7648/25000 [========>.....................] - ETA: 1:02 - loss: 7.6606 - accuracy: 0.5004
 7680/25000 [========>.....................] - ETA: 1:02 - loss: 7.6566 - accuracy: 0.5007
 7712/25000 [========>.....................] - ETA: 1:02 - loss: 7.6527 - accuracy: 0.5009
 7744/25000 [========>.....................] - ETA: 1:02 - loss: 7.6547 - accuracy: 0.5008
 7776/25000 [========>.....................] - ETA: 1:01 - loss: 7.6548 - accuracy: 0.5008
 7808/25000 [========>.....................] - ETA: 1:01 - loss: 7.6568 - accuracy: 0.5006
 7840/25000 [========>.....................] - ETA: 1:01 - loss: 7.6627 - accuracy: 0.5003
 7872/25000 [========>.....................] - ETA: 1:01 - loss: 7.6666 - accuracy: 0.5000
 7904/25000 [========>.....................] - ETA: 1:01 - loss: 7.6686 - accuracy: 0.4999
 7936/25000 [========>.....................] - ETA: 1:01 - loss: 7.6705 - accuracy: 0.4997
 7968/25000 [========>.....................] - ETA: 1:01 - loss: 7.6724 - accuracy: 0.4996
 8000/25000 [========>.....................] - ETA: 1:01 - loss: 7.6724 - accuracy: 0.4996
 8032/25000 [========>.....................] - ETA: 1:01 - loss: 7.6819 - accuracy: 0.4990
 8064/25000 [========>.....................] - ETA: 1:00 - loss: 7.6780 - accuracy: 0.4993
 8096/25000 [========>.....................] - ETA: 1:00 - loss: 7.6837 - accuracy: 0.4989
 8128/25000 [========>.....................] - ETA: 1:00 - loss: 7.6817 - accuracy: 0.4990
 8160/25000 [========>.....................] - ETA: 1:00 - loss: 7.6835 - accuracy: 0.4989
 8192/25000 [========>.....................] - ETA: 1:00 - loss: 7.6797 - accuracy: 0.4991
 8224/25000 [========>.....................] - ETA: 1:00 - loss: 7.6834 - accuracy: 0.4989
 8256/25000 [========>.....................] - ETA: 1:00 - loss: 7.6796 - accuracy: 0.4992
 8288/25000 [========>.....................] - ETA: 1:00 - loss: 7.6759 - accuracy: 0.4994
 8320/25000 [========>.....................] - ETA: 1:00 - loss: 7.6740 - accuracy: 0.4995
 8352/25000 [=========>....................] - ETA: 59s - loss: 7.6758 - accuracy: 0.4994 
 8384/25000 [=========>....................] - ETA: 59s - loss: 7.6758 - accuracy: 0.4994
 8416/25000 [=========>....................] - ETA: 59s - loss: 7.6776 - accuracy: 0.4993
 8448/25000 [=========>....................] - ETA: 59s - loss: 7.6775 - accuracy: 0.4993
 8480/25000 [=========>....................] - ETA: 59s - loss: 7.6702 - accuracy: 0.4998
 8512/25000 [=========>....................] - ETA: 59s - loss: 7.6738 - accuracy: 0.4995
 8544/25000 [=========>....................] - ETA: 59s - loss: 7.6774 - accuracy: 0.4993
 8576/25000 [=========>....................] - ETA: 59s - loss: 7.6773 - accuracy: 0.4993
 8608/25000 [=========>....................] - ETA: 59s - loss: 7.6755 - accuracy: 0.4994
 8640/25000 [=========>....................] - ETA: 58s - loss: 7.6773 - accuracy: 0.4993
 8672/25000 [=========>....................] - ETA: 58s - loss: 7.6825 - accuracy: 0.4990
 8704/25000 [=========>....................] - ETA: 58s - loss: 7.6895 - accuracy: 0.4985
 8736/25000 [=========>....................] - ETA: 58s - loss: 7.6947 - accuracy: 0.4982
 8768/25000 [=========>....................] - ETA: 58s - loss: 7.6911 - accuracy: 0.4984
 8800/25000 [=========>....................] - ETA: 58s - loss: 7.6910 - accuracy: 0.4984
 8832/25000 [=========>....................] - ETA: 58s - loss: 7.6944 - accuracy: 0.4982
 8864/25000 [=========>....................] - ETA: 58s - loss: 7.6926 - accuracy: 0.4983
 8896/25000 [=========>....................] - ETA: 58s - loss: 7.6925 - accuracy: 0.4983
 8928/25000 [=========>....................] - ETA: 57s - loss: 7.6958 - accuracy: 0.4981
 8960/25000 [=========>....................] - ETA: 57s - loss: 7.6923 - accuracy: 0.4983
 8992/25000 [=========>....................] - ETA: 57s - loss: 7.6854 - accuracy: 0.4988
 9024/25000 [=========>....................] - ETA: 57s - loss: 7.6921 - accuracy: 0.4983
 9056/25000 [=========>....................] - ETA: 57s - loss: 7.6869 - accuracy: 0.4987
 9088/25000 [=========>....................] - ETA: 57s - loss: 7.6801 - accuracy: 0.4991
 9120/25000 [=========>....................] - ETA: 57s - loss: 7.6834 - accuracy: 0.4989
 9152/25000 [=========>....................] - ETA: 57s - loss: 7.6850 - accuracy: 0.4988
 9184/25000 [==========>...................] - ETA: 56s - loss: 7.6833 - accuracy: 0.4989
 9216/25000 [==========>...................] - ETA: 56s - loss: 7.6799 - accuracy: 0.4991
 9248/25000 [==========>...................] - ETA: 56s - loss: 7.6815 - accuracy: 0.4990
 9280/25000 [==========>...................] - ETA: 56s - loss: 7.6765 - accuracy: 0.4994
 9312/25000 [==========>...................] - ETA: 56s - loss: 7.6732 - accuracy: 0.4996
 9344/25000 [==========>...................] - ETA: 56s - loss: 7.6650 - accuracy: 0.5001
 9376/25000 [==========>...................] - ETA: 56s - loss: 7.6715 - accuracy: 0.4997
 9408/25000 [==========>...................] - ETA: 56s - loss: 7.6715 - accuracy: 0.4997
 9440/25000 [==========>...................] - ETA: 56s - loss: 7.6747 - accuracy: 0.4995
 9472/25000 [==========>...................] - ETA: 55s - loss: 7.6763 - accuracy: 0.4994
 9504/25000 [==========>...................] - ETA: 55s - loss: 7.6795 - accuracy: 0.4992
 9536/25000 [==========>...................] - ETA: 55s - loss: 7.6811 - accuracy: 0.4991
 9568/25000 [==========>...................] - ETA: 55s - loss: 7.6810 - accuracy: 0.4991
 9600/25000 [==========>...................] - ETA: 55s - loss: 7.6746 - accuracy: 0.4995
 9632/25000 [==========>...................] - ETA: 55s - loss: 7.6698 - accuracy: 0.4998
 9664/25000 [==========>...................] - ETA: 55s - loss: 7.6666 - accuracy: 0.5000
 9696/25000 [==========>...................] - ETA: 55s - loss: 7.6603 - accuracy: 0.5004
 9728/25000 [==========>...................] - ETA: 54s - loss: 7.6572 - accuracy: 0.5006
 9760/25000 [==========>...................] - ETA: 54s - loss: 7.6666 - accuracy: 0.5000
 9792/25000 [==========>...................] - ETA: 54s - loss: 7.6651 - accuracy: 0.5001
 9824/25000 [==========>...................] - ETA: 54s - loss: 7.6651 - accuracy: 0.5001
 9856/25000 [==========>...................] - ETA: 54s - loss: 7.6620 - accuracy: 0.5003
 9888/25000 [==========>...................] - ETA: 54s - loss: 7.6682 - accuracy: 0.4999
 9920/25000 [==========>...................] - ETA: 54s - loss: 7.6713 - accuracy: 0.4997
 9952/25000 [==========>...................] - ETA: 54s - loss: 7.6728 - accuracy: 0.4996
 9984/25000 [==========>...................] - ETA: 54s - loss: 7.6651 - accuracy: 0.5001
10016/25000 [===========>..................] - ETA: 53s - loss: 7.6636 - accuracy: 0.5002
10048/25000 [===========>..................] - ETA: 53s - loss: 7.6605 - accuracy: 0.5004
10080/25000 [===========>..................] - ETA: 53s - loss: 7.6590 - accuracy: 0.5005
10112/25000 [===========>..................] - ETA: 53s - loss: 7.6681 - accuracy: 0.4999
10144/25000 [===========>..................] - ETA: 53s - loss: 7.6696 - accuracy: 0.4998
10176/25000 [===========>..................] - ETA: 53s - loss: 7.6636 - accuracy: 0.5002
10208/25000 [===========>..................] - ETA: 53s - loss: 7.6651 - accuracy: 0.5001
10240/25000 [===========>..................] - ETA: 53s - loss: 7.6621 - accuracy: 0.5003
10272/25000 [===========>..................] - ETA: 53s - loss: 7.6681 - accuracy: 0.4999
10304/25000 [===========>..................] - ETA: 52s - loss: 7.6755 - accuracy: 0.4994
10336/25000 [===========>..................] - ETA: 52s - loss: 7.6770 - accuracy: 0.4993
10368/25000 [===========>..................] - ETA: 52s - loss: 7.6740 - accuracy: 0.4995
10400/25000 [===========>..................] - ETA: 52s - loss: 7.6784 - accuracy: 0.4992
10432/25000 [===========>..................] - ETA: 52s - loss: 7.6769 - accuracy: 0.4993
10464/25000 [===========>..................] - ETA: 52s - loss: 7.6739 - accuracy: 0.4995
10496/25000 [===========>..................] - ETA: 52s - loss: 7.6695 - accuracy: 0.4998
10528/25000 [===========>..................] - ETA: 52s - loss: 7.6710 - accuracy: 0.4997
10560/25000 [===========>..................] - ETA: 51s - loss: 7.6681 - accuracy: 0.4999
10592/25000 [===========>..................] - ETA: 51s - loss: 7.6681 - accuracy: 0.4999
10624/25000 [===========>..................] - ETA: 51s - loss: 7.6695 - accuracy: 0.4998
10656/25000 [===========>..................] - ETA: 51s - loss: 7.6637 - accuracy: 0.5002
10688/25000 [===========>..................] - ETA: 51s - loss: 7.6724 - accuracy: 0.4996
10720/25000 [===========>..................] - ETA: 51s - loss: 7.6695 - accuracy: 0.4998
10752/25000 [===========>..................] - ETA: 51s - loss: 7.6737 - accuracy: 0.4995
10784/25000 [===========>..................] - ETA: 51s - loss: 7.6666 - accuracy: 0.5000
10816/25000 [===========>..................] - ETA: 51s - loss: 7.6595 - accuracy: 0.5005
10848/25000 [============>.................] - ETA: 50s - loss: 7.6624 - accuracy: 0.5003
10880/25000 [============>.................] - ETA: 50s - loss: 7.6638 - accuracy: 0.5002
10912/25000 [============>.................] - ETA: 50s - loss: 7.6624 - accuracy: 0.5003
10944/25000 [============>.................] - ETA: 50s - loss: 7.6638 - accuracy: 0.5002
10976/25000 [============>.................] - ETA: 50s - loss: 7.6638 - accuracy: 0.5002
11008/25000 [============>.................] - ETA: 50s - loss: 7.6638 - accuracy: 0.5002
11040/25000 [============>.................] - ETA: 50s - loss: 7.6638 - accuracy: 0.5002
11072/25000 [============>.................] - ETA: 50s - loss: 7.6625 - accuracy: 0.5003
11104/25000 [============>.................] - ETA: 50s - loss: 7.6597 - accuracy: 0.5005
11136/25000 [============>.................] - ETA: 49s - loss: 7.6529 - accuracy: 0.5009
11168/25000 [============>.................] - ETA: 49s - loss: 7.6529 - accuracy: 0.5009
11200/25000 [============>.................] - ETA: 49s - loss: 7.6488 - accuracy: 0.5012
11232/25000 [============>.................] - ETA: 49s - loss: 7.6489 - accuracy: 0.5012
11264/25000 [============>.................] - ETA: 49s - loss: 7.6557 - accuracy: 0.5007
11296/25000 [============>.................] - ETA: 49s - loss: 7.6490 - accuracy: 0.5012
11328/25000 [============>.................] - ETA: 49s - loss: 7.6463 - accuracy: 0.5013
11360/25000 [============>.................] - ETA: 49s - loss: 7.6504 - accuracy: 0.5011
11392/25000 [============>.................] - ETA: 49s - loss: 7.6505 - accuracy: 0.5011
11424/25000 [============>.................] - ETA: 48s - loss: 7.6451 - accuracy: 0.5014
11456/25000 [============>.................] - ETA: 48s - loss: 7.6479 - accuracy: 0.5012
11488/25000 [============>.................] - ETA: 48s - loss: 7.6506 - accuracy: 0.5010
11520/25000 [============>.................] - ETA: 48s - loss: 7.6506 - accuracy: 0.5010
11552/25000 [============>.................] - ETA: 48s - loss: 7.6467 - accuracy: 0.5013
11584/25000 [============>.................] - ETA: 48s - loss: 7.6468 - accuracy: 0.5013
11616/25000 [============>.................] - ETA: 48s - loss: 7.6481 - accuracy: 0.5012
11648/25000 [============>.................] - ETA: 48s - loss: 7.6574 - accuracy: 0.5006
11680/25000 [=============>................] - ETA: 48s - loss: 7.6535 - accuracy: 0.5009
11712/25000 [=============>................] - ETA: 47s - loss: 7.6522 - accuracy: 0.5009
11744/25000 [=============>................] - ETA: 47s - loss: 7.6444 - accuracy: 0.5014
11776/25000 [=============>................] - ETA: 47s - loss: 7.6523 - accuracy: 0.5009
11808/25000 [=============>................] - ETA: 47s - loss: 7.6458 - accuracy: 0.5014
11840/25000 [=============>................] - ETA: 47s - loss: 7.6472 - accuracy: 0.5013
11872/25000 [=============>................] - ETA: 47s - loss: 7.6460 - accuracy: 0.5013
11904/25000 [=============>................] - ETA: 47s - loss: 7.6434 - accuracy: 0.5015
11936/25000 [=============>................] - ETA: 47s - loss: 7.6422 - accuracy: 0.5016
11968/25000 [=============>................] - ETA: 46s - loss: 7.6461 - accuracy: 0.5013
12000/25000 [=============>................] - ETA: 46s - loss: 7.6487 - accuracy: 0.5012
12032/25000 [=============>................] - ETA: 46s - loss: 7.6475 - accuracy: 0.5012
12064/25000 [=============>................] - ETA: 46s - loss: 7.6488 - accuracy: 0.5012
12096/25000 [=============>................] - ETA: 46s - loss: 7.6489 - accuracy: 0.5012
12128/25000 [=============>................] - ETA: 46s - loss: 7.6451 - accuracy: 0.5014
12160/25000 [=============>................] - ETA: 46s - loss: 7.6477 - accuracy: 0.5012
12192/25000 [=============>................] - ETA: 46s - loss: 7.6503 - accuracy: 0.5011
12224/25000 [=============>................] - ETA: 46s - loss: 7.6465 - accuracy: 0.5013
12256/25000 [=============>................] - ETA: 45s - loss: 7.6416 - accuracy: 0.5016
12288/25000 [=============>................] - ETA: 45s - loss: 7.6417 - accuracy: 0.5016
12320/25000 [=============>................] - ETA: 45s - loss: 7.6455 - accuracy: 0.5014
12352/25000 [=============>................] - ETA: 45s - loss: 7.6406 - accuracy: 0.5017
12384/25000 [=============>................] - ETA: 45s - loss: 7.6419 - accuracy: 0.5016
12416/25000 [=============>................] - ETA: 45s - loss: 7.6419 - accuracy: 0.5016
12448/25000 [=============>................] - ETA: 45s - loss: 7.6395 - accuracy: 0.5018
12480/25000 [=============>................] - ETA: 45s - loss: 7.6445 - accuracy: 0.5014
12512/25000 [==============>...............] - ETA: 45s - loss: 7.6446 - accuracy: 0.5014
12544/25000 [==============>...............] - ETA: 44s - loss: 7.6471 - accuracy: 0.5013
12576/25000 [==============>...............] - ETA: 44s - loss: 7.6447 - accuracy: 0.5014
12608/25000 [==============>...............] - ETA: 44s - loss: 7.6435 - accuracy: 0.5015
12640/25000 [==============>...............] - ETA: 44s - loss: 7.6472 - accuracy: 0.5013
12672/25000 [==============>...............] - ETA: 44s - loss: 7.6460 - accuracy: 0.5013
12704/25000 [==============>...............] - ETA: 44s - loss: 7.6437 - accuracy: 0.5015
12736/25000 [==============>...............] - ETA: 44s - loss: 7.6474 - accuracy: 0.5013
12768/25000 [==============>...............] - ETA: 44s - loss: 7.6558 - accuracy: 0.5007
12800/25000 [==============>...............] - ETA: 44s - loss: 7.6522 - accuracy: 0.5009
12832/25000 [==============>...............] - ETA: 43s - loss: 7.6463 - accuracy: 0.5013
12864/25000 [==============>...............] - ETA: 43s - loss: 7.6440 - accuracy: 0.5015
12896/25000 [==============>...............] - ETA: 43s - loss: 7.6393 - accuracy: 0.5018
12928/25000 [==============>...............] - ETA: 43s - loss: 7.6334 - accuracy: 0.5022
12960/25000 [==============>...............] - ETA: 43s - loss: 7.6394 - accuracy: 0.5018
12992/25000 [==============>...............] - ETA: 43s - loss: 7.6407 - accuracy: 0.5017
13024/25000 [==============>...............] - ETA: 43s - loss: 7.6384 - accuracy: 0.5018
13056/25000 [==============>...............] - ETA: 43s - loss: 7.6361 - accuracy: 0.5020
13088/25000 [==============>...............] - ETA: 43s - loss: 7.6326 - accuracy: 0.5022
13120/25000 [==============>...............] - ETA: 42s - loss: 7.6374 - accuracy: 0.5019
13152/25000 [==============>...............] - ETA: 42s - loss: 7.6351 - accuracy: 0.5021
13184/25000 [==============>...............] - ETA: 42s - loss: 7.6375 - accuracy: 0.5019
13216/25000 [==============>...............] - ETA: 42s - loss: 7.6399 - accuracy: 0.5017
13248/25000 [==============>...............] - ETA: 42s - loss: 7.6354 - accuracy: 0.5020
13280/25000 [==============>...............] - ETA: 42s - loss: 7.6401 - accuracy: 0.5017
13312/25000 [==============>...............] - ETA: 42s - loss: 7.6413 - accuracy: 0.5017
13344/25000 [===============>..............] - ETA: 42s - loss: 7.6356 - accuracy: 0.5020
13376/25000 [===============>..............] - ETA: 41s - loss: 7.6345 - accuracy: 0.5021
13408/25000 [===============>..............] - ETA: 41s - loss: 7.6335 - accuracy: 0.5022
13440/25000 [===============>..............] - ETA: 41s - loss: 7.6358 - accuracy: 0.5020
13472/25000 [===============>..............] - ETA: 41s - loss: 7.6382 - accuracy: 0.5019
13504/25000 [===============>..............] - ETA: 41s - loss: 7.6371 - accuracy: 0.5019
13536/25000 [===============>..............] - ETA: 41s - loss: 7.6383 - accuracy: 0.5018
13568/25000 [===============>..............] - ETA: 41s - loss: 7.6338 - accuracy: 0.5021
13600/25000 [===============>..............] - ETA: 41s - loss: 7.6384 - accuracy: 0.5018
13632/25000 [===============>..............] - ETA: 41s - loss: 7.6351 - accuracy: 0.5021
13664/25000 [===============>..............] - ETA: 40s - loss: 7.6318 - accuracy: 0.5023
13696/25000 [===============>..............] - ETA: 40s - loss: 7.6308 - accuracy: 0.5023
13728/25000 [===============>..............] - ETA: 40s - loss: 7.6298 - accuracy: 0.5024
13760/25000 [===============>..............] - ETA: 40s - loss: 7.6298 - accuracy: 0.5024
13792/25000 [===============>..............] - ETA: 40s - loss: 7.6310 - accuracy: 0.5023
13824/25000 [===============>..............] - ETA: 40s - loss: 7.6311 - accuracy: 0.5023
13856/25000 [===============>..............] - ETA: 40s - loss: 7.6301 - accuracy: 0.5024
13888/25000 [===============>..............] - ETA: 40s - loss: 7.6291 - accuracy: 0.5024
13920/25000 [===============>..............] - ETA: 39s - loss: 7.6347 - accuracy: 0.5021
13952/25000 [===============>..............] - ETA: 39s - loss: 7.6424 - accuracy: 0.5016
13984/25000 [===============>..............] - ETA: 39s - loss: 7.6414 - accuracy: 0.5016
14016/25000 [===============>..............] - ETA: 39s - loss: 7.6447 - accuracy: 0.5014
14048/25000 [===============>..............] - ETA: 39s - loss: 7.6481 - accuracy: 0.5012
14080/25000 [===============>..............] - ETA: 39s - loss: 7.6503 - accuracy: 0.5011
14112/25000 [===============>..............] - ETA: 39s - loss: 7.6427 - accuracy: 0.5016
14144/25000 [===============>..............] - ETA: 39s - loss: 7.6536 - accuracy: 0.5008
14176/25000 [================>.............] - ETA: 39s - loss: 7.6547 - accuracy: 0.5008
14208/25000 [================>.............] - ETA: 38s - loss: 7.6569 - accuracy: 0.5006
14240/25000 [================>.............] - ETA: 38s - loss: 7.6602 - accuracy: 0.5004
14272/25000 [================>.............] - ETA: 38s - loss: 7.6570 - accuracy: 0.5006
14304/25000 [================>.............] - ETA: 38s - loss: 7.6548 - accuracy: 0.5008
14336/25000 [================>.............] - ETA: 38s - loss: 7.6570 - accuracy: 0.5006
14368/25000 [================>.............] - ETA: 38s - loss: 7.6570 - accuracy: 0.5006
14400/25000 [================>.............] - ETA: 38s - loss: 7.6581 - accuracy: 0.5006
14432/25000 [================>.............] - ETA: 38s - loss: 7.6560 - accuracy: 0.5007
14464/25000 [================>.............] - ETA: 38s - loss: 7.6613 - accuracy: 0.5003
14496/25000 [================>.............] - ETA: 37s - loss: 7.6592 - accuracy: 0.5005
14528/25000 [================>.............] - ETA: 37s - loss: 7.6582 - accuracy: 0.5006
14560/25000 [================>.............] - ETA: 37s - loss: 7.6582 - accuracy: 0.5005
14592/25000 [================>.............] - ETA: 37s - loss: 7.6603 - accuracy: 0.5004
14624/25000 [================>.............] - ETA: 37s - loss: 7.6582 - accuracy: 0.5005
14656/25000 [================>.............] - ETA: 37s - loss: 7.6593 - accuracy: 0.5005
14688/25000 [================>.............] - ETA: 37s - loss: 7.6614 - accuracy: 0.5003
14720/25000 [================>.............] - ETA: 37s - loss: 7.6625 - accuracy: 0.5003
14752/25000 [================>.............] - ETA: 36s - loss: 7.6656 - accuracy: 0.5001
14784/25000 [================>.............] - ETA: 36s - loss: 7.6666 - accuracy: 0.5000
14816/25000 [================>.............] - ETA: 36s - loss: 7.6656 - accuracy: 0.5001
14848/25000 [================>.............] - ETA: 36s - loss: 7.6697 - accuracy: 0.4998
14880/25000 [================>.............] - ETA: 36s - loss: 7.6749 - accuracy: 0.4995
14912/25000 [================>.............] - ETA: 36s - loss: 7.6790 - accuracy: 0.4992
14944/25000 [================>.............] - ETA: 36s - loss: 7.6779 - accuracy: 0.4993
14976/25000 [================>.............] - ETA: 36s - loss: 7.6820 - accuracy: 0.4990
15008/25000 [=================>............] - ETA: 36s - loss: 7.6809 - accuracy: 0.4991
15040/25000 [=================>............] - ETA: 35s - loss: 7.6809 - accuracy: 0.4991
15072/25000 [=================>............] - ETA: 35s - loss: 7.6758 - accuracy: 0.4994
15104/25000 [=================>............] - ETA: 35s - loss: 7.6778 - accuracy: 0.4993
15136/25000 [=================>............] - ETA: 35s - loss: 7.6808 - accuracy: 0.4991
15168/25000 [=================>............] - ETA: 35s - loss: 7.6858 - accuracy: 0.4987
15200/25000 [=================>............] - ETA: 35s - loss: 7.6878 - accuracy: 0.4986
15232/25000 [=================>............] - ETA: 35s - loss: 7.6878 - accuracy: 0.4986
15264/25000 [=================>............] - ETA: 35s - loss: 7.6867 - accuracy: 0.4987
15296/25000 [=================>............] - ETA: 35s - loss: 7.6847 - accuracy: 0.4988
15328/25000 [=================>............] - ETA: 34s - loss: 7.6856 - accuracy: 0.4988
15360/25000 [=================>............] - ETA: 34s - loss: 7.6816 - accuracy: 0.4990
15392/25000 [=================>............] - ETA: 34s - loss: 7.6826 - accuracy: 0.4990
15424/25000 [=================>............] - ETA: 34s - loss: 7.6875 - accuracy: 0.4986
15456/25000 [=================>............] - ETA: 34s - loss: 7.6855 - accuracy: 0.4988
15488/25000 [=================>............] - ETA: 34s - loss: 7.6864 - accuracy: 0.4987
15520/25000 [=================>............] - ETA: 34s - loss: 7.6864 - accuracy: 0.4987
15552/25000 [=================>............] - ETA: 34s - loss: 7.6844 - accuracy: 0.4988
15584/25000 [=================>............] - ETA: 33s - loss: 7.6814 - accuracy: 0.4990
15616/25000 [=================>............] - ETA: 33s - loss: 7.6843 - accuracy: 0.4988
15648/25000 [=================>............] - ETA: 33s - loss: 7.6823 - accuracy: 0.4990
15680/25000 [=================>............] - ETA: 33s - loss: 7.6813 - accuracy: 0.4990
15712/25000 [=================>............] - ETA: 33s - loss: 7.6803 - accuracy: 0.4991
15744/25000 [=================>............] - ETA: 33s - loss: 7.6822 - accuracy: 0.4990
15776/25000 [=================>............] - ETA: 33s - loss: 7.6841 - accuracy: 0.4989
15808/25000 [=================>............] - ETA: 33s - loss: 7.6850 - accuracy: 0.4988
15840/25000 [==================>...........] - ETA: 33s - loss: 7.6879 - accuracy: 0.4986
15872/25000 [==================>...........] - ETA: 32s - loss: 7.6879 - accuracy: 0.4986
15904/25000 [==================>...........] - ETA: 32s - loss: 7.6917 - accuracy: 0.4984
15936/25000 [==================>...........] - ETA: 32s - loss: 7.6907 - accuracy: 0.4984
15968/25000 [==================>...........] - ETA: 32s - loss: 7.6954 - accuracy: 0.4981
16000/25000 [==================>...........] - ETA: 32s - loss: 7.6973 - accuracy: 0.4980
16032/25000 [==================>...........] - ETA: 32s - loss: 7.6953 - accuracy: 0.4981
16064/25000 [==================>...........] - ETA: 32s - loss: 7.6991 - accuracy: 0.4979
16096/25000 [==================>...........] - ETA: 32s - loss: 7.7047 - accuracy: 0.4975
16128/25000 [==================>...........] - ETA: 32s - loss: 7.7065 - accuracy: 0.4974
16160/25000 [==================>...........] - ETA: 31s - loss: 7.7093 - accuracy: 0.4972
16192/25000 [==================>...........] - ETA: 31s - loss: 7.7111 - accuracy: 0.4971
16224/25000 [==================>...........] - ETA: 31s - loss: 7.7073 - accuracy: 0.4973
16256/25000 [==================>...........] - ETA: 31s - loss: 7.7081 - accuracy: 0.4973
16288/25000 [==================>...........] - ETA: 31s - loss: 7.7062 - accuracy: 0.4974
16320/25000 [==================>...........] - ETA: 31s - loss: 7.7051 - accuracy: 0.4975
16352/25000 [==================>...........] - ETA: 31s - loss: 7.7116 - accuracy: 0.4971
16384/25000 [==================>...........] - ETA: 31s - loss: 7.7143 - accuracy: 0.4969
16416/25000 [==================>...........] - ETA: 30s - loss: 7.7152 - accuracy: 0.4968
16448/25000 [==================>...........] - ETA: 30s - loss: 7.7198 - accuracy: 0.4965
16480/25000 [==================>...........] - ETA: 30s - loss: 7.7159 - accuracy: 0.4968
16512/25000 [==================>...........] - ETA: 30s - loss: 7.7149 - accuracy: 0.4969
16544/25000 [==================>...........] - ETA: 30s - loss: 7.7139 - accuracy: 0.4969
16576/25000 [==================>...........] - ETA: 30s - loss: 7.7156 - accuracy: 0.4968
16608/25000 [==================>...........] - ETA: 30s - loss: 7.7156 - accuracy: 0.4968
16640/25000 [==================>...........] - ETA: 30s - loss: 7.7108 - accuracy: 0.4971
16672/25000 [===================>..........] - ETA: 30s - loss: 7.7108 - accuracy: 0.4971
16704/25000 [===================>..........] - ETA: 29s - loss: 7.7070 - accuracy: 0.4974
16736/25000 [===================>..........] - ETA: 29s - loss: 7.7005 - accuracy: 0.4978
16768/25000 [===================>..........] - ETA: 29s - loss: 7.7014 - accuracy: 0.4977
16800/25000 [===================>..........] - ETA: 29s - loss: 7.6995 - accuracy: 0.4979
16832/25000 [===================>..........] - ETA: 29s - loss: 7.6976 - accuracy: 0.4980
16864/25000 [===================>..........] - ETA: 29s - loss: 7.6966 - accuracy: 0.4980
16896/25000 [===================>..........] - ETA: 29s - loss: 7.7011 - accuracy: 0.4978
16928/25000 [===================>..........] - ETA: 29s - loss: 7.6983 - accuracy: 0.4979
16960/25000 [===================>..........] - ETA: 29s - loss: 7.7001 - accuracy: 0.4978
16992/25000 [===================>..........] - ETA: 28s - loss: 7.7018 - accuracy: 0.4977
17024/25000 [===================>..........] - ETA: 28s - loss: 7.7008 - accuracy: 0.4978
17056/25000 [===================>..........] - ETA: 28s - loss: 7.6972 - accuracy: 0.4980
17088/25000 [===================>..........] - ETA: 28s - loss: 7.6962 - accuracy: 0.4981
17120/25000 [===================>..........] - ETA: 28s - loss: 7.6971 - accuracy: 0.4980
17152/25000 [===================>..........] - ETA: 28s - loss: 7.6925 - accuracy: 0.4983
17184/25000 [===================>..........] - ETA: 28s - loss: 7.6943 - accuracy: 0.4982
17216/25000 [===================>..........] - ETA: 28s - loss: 7.6942 - accuracy: 0.4982
17248/25000 [===================>..........] - ETA: 27s - loss: 7.6968 - accuracy: 0.4980
17280/25000 [===================>..........] - ETA: 27s - loss: 7.6950 - accuracy: 0.4981
17312/25000 [===================>..........] - ETA: 27s - loss: 7.6976 - accuracy: 0.4980
17344/25000 [===================>..........] - ETA: 27s - loss: 7.6958 - accuracy: 0.4981
17376/25000 [===================>..........] - ETA: 27s - loss: 7.6913 - accuracy: 0.4984
17408/25000 [===================>..........] - ETA: 27s - loss: 7.6895 - accuracy: 0.4985
17440/25000 [===================>..........] - ETA: 27s - loss: 7.6851 - accuracy: 0.4988
17472/25000 [===================>..........] - ETA: 27s - loss: 7.6859 - accuracy: 0.4987
17504/25000 [====================>.........] - ETA: 27s - loss: 7.6850 - accuracy: 0.4988
17536/25000 [====================>.........] - ETA: 26s - loss: 7.6832 - accuracy: 0.4989
17568/25000 [====================>.........] - ETA: 26s - loss: 7.6841 - accuracy: 0.4989
17600/25000 [====================>.........] - ETA: 26s - loss: 7.6823 - accuracy: 0.4990
17632/25000 [====================>.........] - ETA: 26s - loss: 7.6805 - accuracy: 0.4991
17664/25000 [====================>.........] - ETA: 26s - loss: 7.6822 - accuracy: 0.4990
17696/25000 [====================>.........] - ETA: 26s - loss: 7.6839 - accuracy: 0.4989
17728/25000 [====================>.........] - ETA: 26s - loss: 7.6813 - accuracy: 0.4990
17760/25000 [====================>.........] - ETA: 26s - loss: 7.6813 - accuracy: 0.4990
17792/25000 [====================>.........] - ETA: 25s - loss: 7.6821 - accuracy: 0.4990
17824/25000 [====================>.........] - ETA: 25s - loss: 7.6804 - accuracy: 0.4991
17856/25000 [====================>.........] - ETA: 25s - loss: 7.6786 - accuracy: 0.4992
17888/25000 [====================>.........] - ETA: 25s - loss: 7.6795 - accuracy: 0.4992
17920/25000 [====================>.........] - ETA: 25s - loss: 7.6803 - accuracy: 0.4991
17952/25000 [====================>.........] - ETA: 25s - loss: 7.6820 - accuracy: 0.4990
17984/25000 [====================>.........] - ETA: 25s - loss: 7.6811 - accuracy: 0.4991
18016/25000 [====================>.........] - ETA: 25s - loss: 7.6802 - accuracy: 0.4991
18048/25000 [====================>.........] - ETA: 25s - loss: 7.6802 - accuracy: 0.4991
18080/25000 [====================>.........] - ETA: 24s - loss: 7.6793 - accuracy: 0.4992
18112/25000 [====================>.........] - ETA: 24s - loss: 7.6776 - accuracy: 0.4993
18144/25000 [====================>.........] - ETA: 24s - loss: 7.6742 - accuracy: 0.4995
18176/25000 [====================>.........] - ETA: 24s - loss: 7.6734 - accuracy: 0.4996
18208/25000 [====================>.........] - ETA: 24s - loss: 7.6700 - accuracy: 0.4998
18240/25000 [====================>.........] - ETA: 24s - loss: 7.6700 - accuracy: 0.4998
18272/25000 [====================>.........] - ETA: 24s - loss: 7.6733 - accuracy: 0.4996
18304/25000 [====================>.........] - ETA: 24s - loss: 7.6767 - accuracy: 0.4993
18336/25000 [=====================>........] - ETA: 24s - loss: 7.6775 - accuracy: 0.4993
18368/25000 [=====================>........] - ETA: 23s - loss: 7.6800 - accuracy: 0.4991
18400/25000 [=====================>........] - ETA: 23s - loss: 7.6783 - accuracy: 0.4992
18432/25000 [=====================>........] - ETA: 23s - loss: 7.6783 - accuracy: 0.4992
18464/25000 [=====================>........] - ETA: 23s - loss: 7.6841 - accuracy: 0.4989
18496/25000 [=====================>........] - ETA: 23s - loss: 7.6815 - accuracy: 0.4990
18528/25000 [=====================>........] - ETA: 23s - loss: 7.6832 - accuracy: 0.4989
18560/25000 [=====================>........] - ETA: 23s - loss: 7.6856 - accuracy: 0.4988
18592/25000 [=====================>........] - ETA: 23s - loss: 7.6897 - accuracy: 0.4985
18624/25000 [=====================>........] - ETA: 23s - loss: 7.6880 - accuracy: 0.4986
18656/25000 [=====================>........] - ETA: 22s - loss: 7.6855 - accuracy: 0.4988
18688/25000 [=====================>........] - ETA: 22s - loss: 7.6863 - accuracy: 0.4987
18720/25000 [=====================>........] - ETA: 22s - loss: 7.6846 - accuracy: 0.4988
18752/25000 [=====================>........] - ETA: 22s - loss: 7.6846 - accuracy: 0.4988
18784/25000 [=====================>........] - ETA: 22s - loss: 7.6854 - accuracy: 0.4988
18816/25000 [=====================>........] - ETA: 22s - loss: 7.6837 - accuracy: 0.4989
18848/25000 [=====================>........] - ETA: 22s - loss: 7.6813 - accuracy: 0.4990
18880/25000 [=====================>........] - ETA: 22s - loss: 7.6821 - accuracy: 0.4990
18912/25000 [=====================>........] - ETA: 21s - loss: 7.6796 - accuracy: 0.4992
18944/25000 [=====================>........] - ETA: 21s - loss: 7.6780 - accuracy: 0.4993
18976/25000 [=====================>........] - ETA: 21s - loss: 7.6795 - accuracy: 0.4992
19008/25000 [=====================>........] - ETA: 21s - loss: 7.6803 - accuracy: 0.4991
19040/25000 [=====================>........] - ETA: 21s - loss: 7.6771 - accuracy: 0.4993
19072/25000 [=====================>........] - ETA: 21s - loss: 7.6787 - accuracy: 0.4992
19104/25000 [=====================>........] - ETA: 21s - loss: 7.6787 - accuracy: 0.4992
19136/25000 [=====================>........] - ETA: 21s - loss: 7.6818 - accuracy: 0.4990
19168/25000 [======================>.......] - ETA: 21s - loss: 7.6818 - accuracy: 0.4990
19200/25000 [======================>.......] - ETA: 20s - loss: 7.6786 - accuracy: 0.4992
19232/25000 [======================>.......] - ETA: 20s - loss: 7.6810 - accuracy: 0.4991
19264/25000 [======================>.......] - ETA: 20s - loss: 7.6786 - accuracy: 0.4992
19296/25000 [======================>.......] - ETA: 20s - loss: 7.6793 - accuracy: 0.4992
19328/25000 [======================>.......] - ETA: 20s - loss: 7.6785 - accuracy: 0.4992
19360/25000 [======================>.......] - ETA: 20s - loss: 7.6761 - accuracy: 0.4994
19392/25000 [======================>.......] - ETA: 20s - loss: 7.6761 - accuracy: 0.4994
19424/25000 [======================>.......] - ETA: 20s - loss: 7.6753 - accuracy: 0.4994
19456/25000 [======================>.......] - ETA: 20s - loss: 7.6745 - accuracy: 0.4995
19488/25000 [======================>.......] - ETA: 19s - loss: 7.6745 - accuracy: 0.4995
19520/25000 [======================>.......] - ETA: 19s - loss: 7.6721 - accuracy: 0.4996
19552/25000 [======================>.......] - ETA: 19s - loss: 7.6737 - accuracy: 0.4995
19584/25000 [======================>.......] - ETA: 19s - loss: 7.6682 - accuracy: 0.4999
19616/25000 [======================>.......] - ETA: 19s - loss: 7.6697 - accuracy: 0.4998
19648/25000 [======================>.......] - ETA: 19s - loss: 7.6721 - accuracy: 0.4996
19680/25000 [======================>.......] - ETA: 19s - loss: 7.6697 - accuracy: 0.4998
19712/25000 [======================>.......] - ETA: 19s - loss: 7.6713 - accuracy: 0.4997
19744/25000 [======================>.......] - ETA: 18s - loss: 7.6713 - accuracy: 0.4997
19776/25000 [======================>.......] - ETA: 18s - loss: 7.6689 - accuracy: 0.4998
19808/25000 [======================>.......] - ETA: 18s - loss: 7.6674 - accuracy: 0.4999
19840/25000 [======================>.......] - ETA: 18s - loss: 7.6682 - accuracy: 0.4999
19872/25000 [======================>.......] - ETA: 18s - loss: 7.6689 - accuracy: 0.4998
19904/25000 [======================>.......] - ETA: 18s - loss: 7.6689 - accuracy: 0.4998
19936/25000 [======================>.......] - ETA: 18s - loss: 7.6666 - accuracy: 0.5000
19968/25000 [======================>.......] - ETA: 18s - loss: 7.6659 - accuracy: 0.5001
20000/25000 [=======================>......] - ETA: 18s - loss: 7.6666 - accuracy: 0.5000
20032/25000 [=======================>......] - ETA: 17s - loss: 7.6628 - accuracy: 0.5002
20064/25000 [=======================>......] - ETA: 17s - loss: 7.6605 - accuracy: 0.5004
20096/25000 [=======================>......] - ETA: 17s - loss: 7.6620 - accuracy: 0.5003
20128/25000 [=======================>......] - ETA: 17s - loss: 7.6674 - accuracy: 0.5000
20160/25000 [=======================>......] - ETA: 17s - loss: 7.6704 - accuracy: 0.4998
20192/25000 [=======================>......] - ETA: 17s - loss: 7.6727 - accuracy: 0.4996
20224/25000 [=======================>......] - ETA: 17s - loss: 7.6734 - accuracy: 0.4996
20256/25000 [=======================>......] - ETA: 17s - loss: 7.6727 - accuracy: 0.4996
20288/25000 [=======================>......] - ETA: 17s - loss: 7.6734 - accuracy: 0.4996
20320/25000 [=======================>......] - ETA: 16s - loss: 7.6727 - accuracy: 0.4996
20352/25000 [=======================>......] - ETA: 16s - loss: 7.6726 - accuracy: 0.4996
20384/25000 [=======================>......] - ETA: 16s - loss: 7.6704 - accuracy: 0.4998
20416/25000 [=======================>......] - ETA: 16s - loss: 7.6719 - accuracy: 0.4997
20448/25000 [=======================>......] - ETA: 16s - loss: 7.6681 - accuracy: 0.4999
20480/25000 [=======================>......] - ETA: 16s - loss: 7.6711 - accuracy: 0.4997
20512/25000 [=======================>......] - ETA: 16s - loss: 7.6689 - accuracy: 0.4999
20544/25000 [=======================>......] - ETA: 16s - loss: 7.6674 - accuracy: 0.5000
20576/25000 [=======================>......] - ETA: 15s - loss: 7.6681 - accuracy: 0.4999
20608/25000 [=======================>......] - ETA: 15s - loss: 7.6733 - accuracy: 0.4996
20640/25000 [=======================>......] - ETA: 15s - loss: 7.6726 - accuracy: 0.4996
20672/25000 [=======================>......] - ETA: 15s - loss: 7.6740 - accuracy: 0.4995
20704/25000 [=======================>......] - ETA: 15s - loss: 7.6762 - accuracy: 0.4994
20736/25000 [=======================>......] - ETA: 15s - loss: 7.6762 - accuracy: 0.4994
20768/25000 [=======================>......] - ETA: 15s - loss: 7.6762 - accuracy: 0.4994
20800/25000 [=======================>......] - ETA: 15s - loss: 7.6769 - accuracy: 0.4993
20832/25000 [=======================>......] - ETA: 15s - loss: 7.6799 - accuracy: 0.4991
20864/25000 [========================>.....] - ETA: 14s - loss: 7.6762 - accuracy: 0.4994
20896/25000 [========================>.....] - ETA: 14s - loss: 7.6769 - accuracy: 0.4993
20928/25000 [========================>.....] - ETA: 14s - loss: 7.6798 - accuracy: 0.4991
20960/25000 [========================>.....] - ETA: 14s - loss: 7.6776 - accuracy: 0.4993
20992/25000 [========================>.....] - ETA: 14s - loss: 7.6783 - accuracy: 0.4992
21024/25000 [========================>.....] - ETA: 14s - loss: 7.6790 - accuracy: 0.4992
21056/25000 [========================>.....] - ETA: 14s - loss: 7.6790 - accuracy: 0.4992
21088/25000 [========================>.....] - ETA: 14s - loss: 7.6783 - accuracy: 0.4992
21120/25000 [========================>.....] - ETA: 14s - loss: 7.6790 - accuracy: 0.4992
21152/25000 [========================>.....] - ETA: 13s - loss: 7.6789 - accuracy: 0.4992
21184/25000 [========================>.....] - ETA: 13s - loss: 7.6775 - accuracy: 0.4993
21216/25000 [========================>.....] - ETA: 13s - loss: 7.6782 - accuracy: 0.4992
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6782 - accuracy: 0.4992
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6803 - accuracy: 0.4991
21312/25000 [========================>.....] - ETA: 13s - loss: 7.6824 - accuracy: 0.4990
21344/25000 [========================>.....] - ETA: 13s - loss: 7.6839 - accuracy: 0.4989
21376/25000 [========================>.....] - ETA: 13s - loss: 7.6853 - accuracy: 0.4988
21408/25000 [========================>.....] - ETA: 12s - loss: 7.6874 - accuracy: 0.4986
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6881 - accuracy: 0.4986
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6880 - accuracy: 0.4986
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6866 - accuracy: 0.4987
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6866 - accuracy: 0.4987
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6858 - accuracy: 0.4987
21600/25000 [========================>.....] - ETA: 12s - loss: 7.6865 - accuracy: 0.4987
21632/25000 [========================>.....] - ETA: 12s - loss: 7.6829 - accuracy: 0.4989
21664/25000 [========================>.....] - ETA: 12s - loss: 7.6843 - accuracy: 0.4988
21696/25000 [=========================>....] - ETA: 11s - loss: 7.6857 - accuracy: 0.4988
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6814 - accuracy: 0.4990
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6800 - accuracy: 0.4991
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6842 - accuracy: 0.4989
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6835 - accuracy: 0.4989
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6849 - accuracy: 0.4988
21888/25000 [=========================>....] - ETA: 11s - loss: 7.6820 - accuracy: 0.4990
21920/25000 [=========================>....] - ETA: 11s - loss: 7.6834 - accuracy: 0.4989
21952/25000 [=========================>....] - ETA: 11s - loss: 7.6841 - accuracy: 0.4989
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6834 - accuracy: 0.4989
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6805 - accuracy: 0.4991
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6805 - accuracy: 0.4991
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6819 - accuracy: 0.4990
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6846 - accuracy: 0.4988
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6846 - accuracy: 0.4988
22176/25000 [=========================>....] - ETA: 10s - loss: 7.6811 - accuracy: 0.4991
22208/25000 [=========================>....] - ETA: 10s - loss: 7.6797 - accuracy: 0.4991
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6804 - accuracy: 0.4991 
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6769 - accuracy: 0.4993
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6790 - accuracy: 0.4992
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6803 - accuracy: 0.4991
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6803 - accuracy: 0.4991
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6803 - accuracy: 0.4991
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6803 - accuracy: 0.4991
22464/25000 [=========================>....] - ETA: 9s - loss: 7.6796 - accuracy: 0.4992
22496/25000 [=========================>....] - ETA: 9s - loss: 7.6782 - accuracy: 0.4992
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6775 - accuracy: 0.4993
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6768 - accuracy: 0.4993
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6768 - accuracy: 0.4993
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6748 - accuracy: 0.4995
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6734 - accuracy: 0.4996
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6741 - accuracy: 0.4995
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6761 - accuracy: 0.4994
22752/25000 [==========================>...] - ETA: 8s - loss: 7.6754 - accuracy: 0.4994
22784/25000 [==========================>...] - ETA: 8s - loss: 7.6774 - accuracy: 0.4993
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6767 - accuracy: 0.4993
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6794 - accuracy: 0.4992
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6814 - accuracy: 0.4990
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6807 - accuracy: 0.4991
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6800 - accuracy: 0.4991
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6806 - accuracy: 0.4991
23008/25000 [==========================>...] - ETA: 7s - loss: 7.6779 - accuracy: 0.4993
23040/25000 [==========================>...] - ETA: 7s - loss: 7.6759 - accuracy: 0.4994
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6759 - accuracy: 0.4994
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6759 - accuracy: 0.4994
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6766 - accuracy: 0.4994
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6785 - accuracy: 0.4992
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6765 - accuracy: 0.4994
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6752 - accuracy: 0.4994
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6765 - accuracy: 0.4994
23296/25000 [==========================>...] - ETA: 6s - loss: 7.6758 - accuracy: 0.4994
23328/25000 [==========================>...] - ETA: 6s - loss: 7.6725 - accuracy: 0.4996
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6725 - accuracy: 0.4996
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6712 - accuracy: 0.4997
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6732 - accuracy: 0.4996
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6725 - accuracy: 0.4996
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6725 - accuracy: 0.4996
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6731 - accuracy: 0.4996
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6712 - accuracy: 0.4997
23584/25000 [===========================>..] - ETA: 5s - loss: 7.6692 - accuracy: 0.4998
23616/25000 [===========================>..] - ETA: 5s - loss: 7.6705 - accuracy: 0.4997
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6692 - accuracy: 0.4998
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6718 - accuracy: 0.4997
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6756 - accuracy: 0.4994
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6737 - accuracy: 0.4995
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6737 - accuracy: 0.4995
23872/25000 [===========================>..] - ETA: 4s - loss: 7.6743 - accuracy: 0.4995
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6750 - accuracy: 0.4995
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6749 - accuracy: 0.4995
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6762 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6781 - accuracy: 0.4992
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6794 - accuracy: 0.4992
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6800 - accuracy: 0.4991
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6806 - accuracy: 0.4991
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6819 - accuracy: 0.4990
24160/25000 [===========================>..] - ETA: 3s - loss: 7.6806 - accuracy: 0.4991
24192/25000 [============================>.] - ETA: 2s - loss: 7.6837 - accuracy: 0.4989
24224/25000 [============================>.] - ETA: 2s - loss: 7.6831 - accuracy: 0.4989
24256/25000 [============================>.] - ETA: 2s - loss: 7.6850 - accuracy: 0.4988
24288/25000 [============================>.] - ETA: 2s - loss: 7.6830 - accuracy: 0.4989
24320/25000 [============================>.] - ETA: 2s - loss: 7.6811 - accuracy: 0.4991
24352/25000 [============================>.] - ETA: 2s - loss: 7.6836 - accuracy: 0.4989
24384/25000 [============================>.] - ETA: 2s - loss: 7.6823 - accuracy: 0.4990
24416/25000 [============================>.] - ETA: 2s - loss: 7.6792 - accuracy: 0.4992
24448/25000 [============================>.] - ETA: 1s - loss: 7.6785 - accuracy: 0.4992
24480/25000 [============================>.] - ETA: 1s - loss: 7.6748 - accuracy: 0.4995
24512/25000 [============================>.] - ETA: 1s - loss: 7.6741 - accuracy: 0.4995
24544/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
24576/25000 [============================>.] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
24608/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24640/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24672/25000 [============================>.] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
24704/25000 [============================>.] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
24736/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24768/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24800/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24832/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24864/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24896/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 108s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
