
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077

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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:52<01:18, 26.06s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.34195242501653256, 'embedding_size_factor': 1.2098359454950227, 'layers.choice': 2, 'learning_rate': 0.008685575708743704, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 5.143085733113739e-05} and reward: 0.3742
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd5\xe2\x8cl\x8e\xf86X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3[|\xef\xb6\xd2`X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x81\xc9\xbe<\xed\x8a0X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\n\xf6\xee\xcdK\xed\x81u.' and reward: 0.3742
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd5\xe2\x8cl\x8e\xf86X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3[|\xef\xb6\xd2`X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x81\xc9\xbe<\xed\x8a0X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\n\xf6\xee\xcdK\xed\x81u.' and reward: 0.3742
 60%|██████    | 3/5 [01:44<01:07, 33.93s/it] 60%|██████    | 3/5 [01:44<01:09, 34.80s/it]
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.20034774369166786, 'embedding_size_factor': 0.9302478302669733, 'layers.choice': 0, 'learning_rate': 0.00011217491840596181, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 1.421261913478809e-05} and reward: 0.3666
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc9\xa4\xfe\xaf}\xd6XX\x15\x00\x00\x00embedding_size_factorq\x03G?\xed\xc4\x97\x19\x05}\xd3X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?\x1dg\xeel\x87Y\xf2X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xed\xceW\x7f\xfe\xc7\xe6u.' and reward: 0.3666
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc9\xa4\xfe\xaf}\xd6XX\x15\x00\x00\x00embedding_size_factorq\x03G?\xed\xc4\x97\x19\x05}\xd3X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?\x1dg\xeel\x87Y\xf2X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xed\xceW\x7f\xfe\xc7\xe6u.' and reward: 0.3666
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 158.27992033958435
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -40.93s of remaining time.
Ensemble size: 5
Ensemble weights: 
[0.6 0.4 0. ]
	0.3904	 = Validation accuracy score
	1.08s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 162.05s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7fe7b80f5ba8> 

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
 [ 0.06187677  0.03297938 -0.00574009 -0.03345062 -0.02726668  0.14573424]
 [ 0.01407389  0.11769616  0.11227611  0.07463694  0.06322851 -0.12807919]
 [ 0.21789645  0.02282511  0.00580813  0.08151481 -0.0954612   0.07465629]
 [ 0.35385147  0.15615623  0.05177926 -0.21810924  0.27234718  0.34728745]
 [ 0.08085865  0.20514862  0.02883269 -0.26528794  0.02955835  0.05323559]
 [ 0.24730575  0.23238018 -0.20268372 -0.32653299  0.11607622 -0.08268695]
 [ 0.31336951 -0.26959324 -0.29324129 -0.17924023  0.49110851  0.05447187]
 [ 0.58943737 -0.08419717  0.19143942  0.08411697 -0.28661424  0.1260374 ]
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
{'loss': 0.4830271154642105, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 15:54:34.170298: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.43215595185756683, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 15:54:35.311404: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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

    8192/17464789 [..............................] - ETA: 2s
 2301952/17464789 [==>...........................] - ETA: 0s
 7962624/17464789 [============>.................] - ETA: 0s
11960320/17464789 [===================>..........] - ETA: 0s
15917056/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 15:54:46.885911: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 15:54:46.890492: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-15 15:54:46.890635: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55706bd21680 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 15:54:46.890650: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:42 - loss: 9.5833 - accuracy: 0.3750
   64/25000 [..............................] - ETA: 2:53 - loss: 8.3854 - accuracy: 0.4531
   96/25000 [..............................] - ETA: 2:18 - loss: 8.6249 - accuracy: 0.4375
  128/25000 [..............................] - ETA: 1:59 - loss: 7.4270 - accuracy: 0.5156
  160/25000 [..............................] - ETA: 1:48 - loss: 7.2833 - accuracy: 0.5250
  192/25000 [..............................] - ETA: 1:42 - loss: 7.0277 - accuracy: 0.5417
  224/25000 [..............................] - ETA: 1:37 - loss: 7.4613 - accuracy: 0.5134
  256/25000 [..............................] - ETA: 1:32 - loss: 7.4869 - accuracy: 0.5117
  288/25000 [..............................] - ETA: 1:30 - loss: 7.6666 - accuracy: 0.5000
  320/25000 [..............................] - ETA: 1:27 - loss: 7.7145 - accuracy: 0.4969
  352/25000 [..............................] - ETA: 1:26 - loss: 7.7102 - accuracy: 0.4972
  384/25000 [..............................] - ETA: 1:24 - loss: 7.7864 - accuracy: 0.4922
  416/25000 [..............................] - ETA: 1:22 - loss: 7.7403 - accuracy: 0.4952
  448/25000 [..............................] - ETA: 1:21 - loss: 7.7008 - accuracy: 0.4978
  480/25000 [..............................] - ETA: 1:19 - loss: 7.7305 - accuracy: 0.4958
  512/25000 [..............................] - ETA: 1:18 - loss: 7.8164 - accuracy: 0.4902
  544/25000 [..............................] - ETA: 1:17 - loss: 7.7794 - accuracy: 0.4926
  576/25000 [..............................] - ETA: 1:16 - loss: 7.8530 - accuracy: 0.4878
  608/25000 [..............................] - ETA: 1:15 - loss: 7.8684 - accuracy: 0.4868
  640/25000 [..............................] - ETA: 1:14 - loss: 7.8343 - accuracy: 0.4891
  672/25000 [..............................] - ETA: 1:14 - loss: 7.6894 - accuracy: 0.4985
  704/25000 [..............................] - ETA: 1:13 - loss: 7.7102 - accuracy: 0.4972
  736/25000 [..............................] - ETA: 1:12 - loss: 7.6666 - accuracy: 0.5000
  768/25000 [..............................] - ETA: 1:12 - loss: 7.6666 - accuracy: 0.5000
  800/25000 [..............................] - ETA: 1:11 - loss: 7.6858 - accuracy: 0.4988
  832/25000 [..............................] - ETA: 1:11 - loss: 7.6666 - accuracy: 0.5000
  864/25000 [>.............................] - ETA: 1:11 - loss: 7.7021 - accuracy: 0.4977
  896/25000 [>.............................] - ETA: 1:10 - loss: 7.6666 - accuracy: 0.5000
  928/25000 [>.............................] - ETA: 1:10 - loss: 7.6171 - accuracy: 0.5032
  960/25000 [>.............................] - ETA: 1:10 - loss: 7.6347 - accuracy: 0.5021
  992/25000 [>.............................] - ETA: 1:09 - loss: 7.6512 - accuracy: 0.5010
 1024/25000 [>.............................] - ETA: 1:09 - loss: 7.5768 - accuracy: 0.5059
 1056/25000 [>.............................] - ETA: 1:08 - loss: 7.6376 - accuracy: 0.5019
 1088/25000 [>.............................] - ETA: 1:08 - loss: 7.6102 - accuracy: 0.5037
 1120/25000 [>.............................] - ETA: 1:08 - loss: 7.5982 - accuracy: 0.5045
 1152/25000 [>.............................] - ETA: 1:07 - loss: 7.5601 - accuracy: 0.5069
 1184/25000 [>.............................] - ETA: 1:07 - loss: 7.6019 - accuracy: 0.5042
 1216/25000 [>.............................] - ETA: 1:07 - loss: 7.6666 - accuracy: 0.5000
 1248/25000 [>.............................] - ETA: 1:07 - loss: 7.6789 - accuracy: 0.4992
 1280/25000 [>.............................] - ETA: 1:06 - loss: 7.6307 - accuracy: 0.5023
 1312/25000 [>.............................] - ETA: 1:06 - loss: 7.6082 - accuracy: 0.5038
 1344/25000 [>.............................] - ETA: 1:06 - loss: 7.5868 - accuracy: 0.5052
 1376/25000 [>.............................] - ETA: 1:06 - loss: 7.5775 - accuracy: 0.5058
 1408/25000 [>.............................] - ETA: 1:06 - loss: 7.5795 - accuracy: 0.5057
 1440/25000 [>.............................] - ETA: 1:06 - loss: 7.5921 - accuracy: 0.5049
 1472/25000 [>.............................] - ETA: 1:06 - loss: 7.6145 - accuracy: 0.5034
 1504/25000 [>.............................] - ETA: 1:06 - loss: 7.5749 - accuracy: 0.5060
 1536/25000 [>.............................] - ETA: 1:05 - loss: 7.5768 - accuracy: 0.5059
 1568/25000 [>.............................] - ETA: 1:05 - loss: 7.5982 - accuracy: 0.5045
 1600/25000 [>.............................] - ETA: 1:05 - loss: 7.5900 - accuracy: 0.5050
 1632/25000 [>.............................] - ETA: 1:05 - loss: 7.6009 - accuracy: 0.5043
 1664/25000 [>.............................] - ETA: 1:05 - loss: 7.5929 - accuracy: 0.5048
 1696/25000 [=>............................] - ETA: 1:04 - loss: 7.5762 - accuracy: 0.5059
 1728/25000 [=>............................] - ETA: 1:04 - loss: 7.5868 - accuracy: 0.5052
 1760/25000 [=>............................] - ETA: 1:04 - loss: 7.5534 - accuracy: 0.5074
 1792/25000 [=>............................] - ETA: 1:04 - loss: 7.5383 - accuracy: 0.5084
 1824/25000 [=>............................] - ETA: 1:04 - loss: 7.4985 - accuracy: 0.5110
 1856/25000 [=>............................] - ETA: 1:04 - loss: 7.5427 - accuracy: 0.5081
 1888/25000 [=>............................] - ETA: 1:04 - loss: 7.4961 - accuracy: 0.5111
 1920/25000 [=>............................] - ETA: 1:03 - loss: 7.5309 - accuracy: 0.5089
 1952/25000 [=>............................] - ETA: 1:03 - loss: 7.5488 - accuracy: 0.5077
 1984/25000 [=>............................] - ETA: 1:03 - loss: 7.5661 - accuracy: 0.5066
 2016/25000 [=>............................] - ETA: 1:03 - loss: 7.5449 - accuracy: 0.5079
 2048/25000 [=>............................] - ETA: 1:03 - loss: 7.5768 - accuracy: 0.5059
 2080/25000 [=>............................] - ETA: 1:03 - loss: 7.5782 - accuracy: 0.5058
 2112/25000 [=>............................] - ETA: 1:03 - loss: 7.6085 - accuracy: 0.5038
 2144/25000 [=>............................] - ETA: 1:03 - loss: 7.6094 - accuracy: 0.5037
 2176/25000 [=>............................] - ETA: 1:02 - loss: 7.6102 - accuracy: 0.5037
 2208/25000 [=>............................] - ETA: 1:02 - loss: 7.6180 - accuracy: 0.5032
 2240/25000 [=>............................] - ETA: 1:02 - loss: 7.6050 - accuracy: 0.5040
 2272/25000 [=>............................] - ETA: 1:02 - loss: 7.5991 - accuracy: 0.5044
 2304/25000 [=>............................] - ETA: 1:02 - loss: 7.6067 - accuracy: 0.5039
 2336/25000 [=>............................] - ETA: 1:02 - loss: 7.6075 - accuracy: 0.5039
 2368/25000 [=>............................] - ETA: 1:02 - loss: 7.6278 - accuracy: 0.5025
 2400/25000 [=>............................] - ETA: 1:02 - loss: 7.6219 - accuracy: 0.5029
 2432/25000 [=>............................] - ETA: 1:01 - loss: 7.6036 - accuracy: 0.5041
 2464/25000 [=>............................] - ETA: 1:01 - loss: 7.6044 - accuracy: 0.5041
 2496/25000 [=>............................] - ETA: 1:01 - loss: 7.6052 - accuracy: 0.5040
 2528/25000 [==>...........................] - ETA: 1:01 - loss: 7.5999 - accuracy: 0.5044
 2560/25000 [==>...........................] - ETA: 1:01 - loss: 7.6247 - accuracy: 0.5027
 2592/25000 [==>...........................] - ETA: 1:01 - loss: 7.6311 - accuracy: 0.5023
 2624/25000 [==>...........................] - ETA: 1:01 - loss: 7.6257 - accuracy: 0.5027
 2656/25000 [==>...........................] - ETA: 1:01 - loss: 7.5916 - accuracy: 0.5049
 2688/25000 [==>...........................] - ETA: 1:01 - loss: 7.5811 - accuracy: 0.5056
 2720/25000 [==>...........................] - ETA: 1:01 - loss: 7.5821 - accuracy: 0.5055
 2752/25000 [==>...........................] - ETA: 1:01 - loss: 7.5830 - accuracy: 0.5055
 2784/25000 [==>...........................] - ETA: 1:01 - loss: 7.5840 - accuracy: 0.5054
 2816/25000 [==>...........................] - ETA: 1:01 - loss: 7.5577 - accuracy: 0.5071
 2848/25000 [==>...........................] - ETA: 1:00 - loss: 7.5589 - accuracy: 0.5070
 2880/25000 [==>...........................] - ETA: 1:00 - loss: 7.5708 - accuracy: 0.5063
 2912/25000 [==>...........................] - ETA: 1:00 - loss: 7.5718 - accuracy: 0.5062
 2944/25000 [==>...........................] - ETA: 1:00 - loss: 7.5833 - accuracy: 0.5054
 2976/25000 [==>...........................] - ETA: 1:00 - loss: 7.6048 - accuracy: 0.5040
 3008/25000 [==>...........................] - ETA: 1:00 - loss: 7.5953 - accuracy: 0.5047
 3040/25000 [==>...........................] - ETA: 1:00 - loss: 7.5859 - accuracy: 0.5053
 3072/25000 [==>...........................] - ETA: 1:00 - loss: 7.5967 - accuracy: 0.5046
 3104/25000 [==>...........................] - ETA: 59s - loss: 7.5777 - accuracy: 0.5058 
 3136/25000 [==>...........................] - ETA: 59s - loss: 7.5933 - accuracy: 0.5048
 3168/25000 [==>...........................] - ETA: 59s - loss: 7.5940 - accuracy: 0.5047
 3200/25000 [==>...........................] - ETA: 59s - loss: 7.5947 - accuracy: 0.5047
 3232/25000 [==>...........................] - ETA: 59s - loss: 7.5765 - accuracy: 0.5059
 3264/25000 [==>...........................] - ETA: 59s - loss: 7.5774 - accuracy: 0.5058
 3296/25000 [==>...........................] - ETA: 59s - loss: 7.5829 - accuracy: 0.5055
 3328/25000 [==>...........................] - ETA: 59s - loss: 7.5883 - accuracy: 0.5051
 3360/25000 [===>..........................] - ETA: 59s - loss: 7.6073 - accuracy: 0.5039
 3392/25000 [===>..........................] - ETA: 58s - loss: 7.6033 - accuracy: 0.5041
 3424/25000 [===>..........................] - ETA: 58s - loss: 7.5994 - accuracy: 0.5044
 3456/25000 [===>..........................] - ETA: 58s - loss: 7.5956 - accuracy: 0.5046
 3488/25000 [===>..........................] - ETA: 58s - loss: 7.5963 - accuracy: 0.5046
 3520/25000 [===>..........................] - ETA: 58s - loss: 7.6056 - accuracy: 0.5040
 3552/25000 [===>..........................] - ETA: 58s - loss: 7.5889 - accuracy: 0.5051
 3584/25000 [===>..........................] - ETA: 58s - loss: 7.5982 - accuracy: 0.5045
 3616/25000 [===>..........................] - ETA: 58s - loss: 7.5988 - accuracy: 0.5044
 3648/25000 [===>..........................] - ETA: 58s - loss: 7.5868 - accuracy: 0.5052
 3680/25000 [===>..........................] - ETA: 57s - loss: 7.5750 - accuracy: 0.5060
 3712/25000 [===>..........................] - ETA: 57s - loss: 7.5799 - accuracy: 0.5057
 3744/25000 [===>..........................] - ETA: 57s - loss: 7.5724 - accuracy: 0.5061
 3776/25000 [===>..........................] - ETA: 57s - loss: 7.5692 - accuracy: 0.5064
 3808/25000 [===>..........................] - ETA: 57s - loss: 7.5740 - accuracy: 0.5060
 3840/25000 [===>..........................] - ETA: 57s - loss: 7.5628 - accuracy: 0.5068
 3872/25000 [===>..........................] - ETA: 57s - loss: 7.5637 - accuracy: 0.5067
 3904/25000 [===>..........................] - ETA: 57s - loss: 7.5645 - accuracy: 0.5067
 3936/25000 [===>..........................] - ETA: 57s - loss: 7.5575 - accuracy: 0.5071
 3968/25000 [===>..........................] - ETA: 57s - loss: 7.5546 - accuracy: 0.5073
 4000/25000 [===>..........................] - ETA: 57s - loss: 7.5708 - accuracy: 0.5063
 4032/25000 [===>..........................] - ETA: 56s - loss: 7.5944 - accuracy: 0.5047
 4064/25000 [===>..........................] - ETA: 56s - loss: 7.5874 - accuracy: 0.5052
 4096/25000 [===>..........................] - ETA: 56s - loss: 7.5843 - accuracy: 0.5054
 4128/25000 [===>..........................] - ETA: 56s - loss: 7.5886 - accuracy: 0.5051
 4160/25000 [===>..........................] - ETA: 56s - loss: 7.5892 - accuracy: 0.5050
 4192/25000 [====>.........................] - ETA: 56s - loss: 7.5861 - accuracy: 0.5052
 4224/25000 [====>.........................] - ETA: 56s - loss: 7.5940 - accuracy: 0.5047
 4256/25000 [====>.........................] - ETA: 56s - loss: 7.5874 - accuracy: 0.5052
 4288/25000 [====>.........................] - ETA: 56s - loss: 7.5880 - accuracy: 0.5051
 4320/25000 [====>.........................] - ETA: 56s - loss: 7.5885 - accuracy: 0.5051
 4352/25000 [====>.........................] - ETA: 55s - loss: 7.5891 - accuracy: 0.5051
 4384/25000 [====>.........................] - ETA: 55s - loss: 7.5967 - accuracy: 0.5046
 4416/25000 [====>.........................] - ETA: 55s - loss: 7.5937 - accuracy: 0.5048
 4448/25000 [====>.........................] - ETA: 55s - loss: 7.6046 - accuracy: 0.5040
 4480/25000 [====>.........................] - ETA: 55s - loss: 7.6290 - accuracy: 0.5025
 4512/25000 [====>.........................] - ETA: 55s - loss: 7.6258 - accuracy: 0.5027
 4544/25000 [====>.........................] - ETA: 55s - loss: 7.6329 - accuracy: 0.5022
 4576/25000 [====>.........................] - ETA: 55s - loss: 7.6231 - accuracy: 0.5028
 4608/25000 [====>.........................] - ETA: 55s - loss: 7.6400 - accuracy: 0.5017
 4640/25000 [====>.........................] - ETA: 55s - loss: 7.6501 - accuracy: 0.5011
 4672/25000 [====>.........................] - ETA: 54s - loss: 7.6404 - accuracy: 0.5017
 4704/25000 [====>.........................] - ETA: 54s - loss: 7.6471 - accuracy: 0.5013
 4736/25000 [====>.........................] - ETA: 54s - loss: 7.6375 - accuracy: 0.5019
 4768/25000 [====>.........................] - ETA: 54s - loss: 7.6505 - accuracy: 0.5010
 4800/25000 [====>.........................] - ETA: 54s - loss: 7.6379 - accuracy: 0.5019
 4832/25000 [====>.........................] - ETA: 54s - loss: 7.6381 - accuracy: 0.5019
 4864/25000 [====>.........................] - ETA: 54s - loss: 7.6509 - accuracy: 0.5010
 4896/25000 [====>.........................] - ETA: 54s - loss: 7.6541 - accuracy: 0.5008
 4928/25000 [====>.........................] - ETA: 54s - loss: 7.6511 - accuracy: 0.5010
 4960/25000 [====>.........................] - ETA: 54s - loss: 7.6573 - accuracy: 0.5006
 4992/25000 [====>.........................] - ETA: 54s - loss: 7.6451 - accuracy: 0.5014
 5024/25000 [=====>........................] - ETA: 53s - loss: 7.6514 - accuracy: 0.5010
 5056/25000 [=====>........................] - ETA: 53s - loss: 7.6454 - accuracy: 0.5014
 5088/25000 [=====>........................] - ETA: 53s - loss: 7.6305 - accuracy: 0.5024
 5120/25000 [=====>........................] - ETA: 53s - loss: 7.6247 - accuracy: 0.5027
 5152/25000 [=====>........................] - ETA: 53s - loss: 7.6160 - accuracy: 0.5033
 5184/25000 [=====>........................] - ETA: 53s - loss: 7.6193 - accuracy: 0.5031
 5216/25000 [=====>........................] - ETA: 53s - loss: 7.6166 - accuracy: 0.5033
 5248/25000 [=====>........................] - ETA: 53s - loss: 7.6199 - accuracy: 0.5030
 5280/25000 [=====>........................] - ETA: 53s - loss: 7.6318 - accuracy: 0.5023
 5312/25000 [=====>........................] - ETA: 53s - loss: 7.6291 - accuracy: 0.5024
 5344/25000 [=====>........................] - ETA: 53s - loss: 7.6150 - accuracy: 0.5034
 5376/25000 [=====>........................] - ETA: 53s - loss: 7.6238 - accuracy: 0.5028
 5408/25000 [=====>........................] - ETA: 53s - loss: 7.6326 - accuracy: 0.5022
 5440/25000 [=====>........................] - ETA: 53s - loss: 7.6328 - accuracy: 0.5022
 5472/25000 [=====>........................] - ETA: 52s - loss: 7.6358 - accuracy: 0.5020
 5504/25000 [=====>........................] - ETA: 52s - loss: 7.6415 - accuracy: 0.5016
 5536/25000 [=====>........................] - ETA: 52s - loss: 7.6306 - accuracy: 0.5023
 5568/25000 [=====>........................] - ETA: 52s - loss: 7.6226 - accuracy: 0.5029
 5600/25000 [=====>........................] - ETA: 52s - loss: 7.6365 - accuracy: 0.5020
 5632/25000 [=====>........................] - ETA: 52s - loss: 7.6149 - accuracy: 0.5034
 5664/25000 [=====>........................] - ETA: 52s - loss: 7.6125 - accuracy: 0.5035
 5696/25000 [=====>........................] - ETA: 52s - loss: 7.6047 - accuracy: 0.5040
 5728/25000 [=====>........................] - ETA: 52s - loss: 7.6158 - accuracy: 0.5033
 5760/25000 [=====>........................] - ETA: 51s - loss: 7.6160 - accuracy: 0.5033
 5792/25000 [=====>........................] - ETA: 51s - loss: 7.6084 - accuracy: 0.5038
 5824/25000 [=====>........................] - ETA: 51s - loss: 7.6219 - accuracy: 0.5029
 5856/25000 [======>.......................] - ETA: 51s - loss: 7.6143 - accuracy: 0.5034
 5888/25000 [======>.......................] - ETA: 51s - loss: 7.6171 - accuracy: 0.5032
 5920/25000 [======>.......................] - ETA: 51s - loss: 7.6174 - accuracy: 0.5032
 5952/25000 [======>.......................] - ETA: 51s - loss: 7.6099 - accuracy: 0.5037
 5984/25000 [======>.......................] - ETA: 51s - loss: 7.6128 - accuracy: 0.5035
 6016/25000 [======>.......................] - ETA: 51s - loss: 7.6156 - accuracy: 0.5033
 6048/25000 [======>.......................] - ETA: 51s - loss: 7.6184 - accuracy: 0.5031
 6080/25000 [======>.......................] - ETA: 51s - loss: 7.6137 - accuracy: 0.5035
 6112/25000 [======>.......................] - ETA: 50s - loss: 7.6240 - accuracy: 0.5028
 6144/25000 [======>.......................] - ETA: 50s - loss: 7.6342 - accuracy: 0.5021
 6176/25000 [======>.......................] - ETA: 50s - loss: 7.6343 - accuracy: 0.5021
 6208/25000 [======>.......................] - ETA: 50s - loss: 7.6370 - accuracy: 0.5019
 6240/25000 [======>.......................] - ETA: 50s - loss: 7.6445 - accuracy: 0.5014
 6272/25000 [======>.......................] - ETA: 50s - loss: 7.6446 - accuracy: 0.5014
 6304/25000 [======>.......................] - ETA: 50s - loss: 7.6423 - accuracy: 0.5016
 6336/25000 [======>.......................] - ETA: 50s - loss: 7.6448 - accuracy: 0.5014
 6368/25000 [======>.......................] - ETA: 50s - loss: 7.6474 - accuracy: 0.5013
 6400/25000 [======>.......................] - ETA: 49s - loss: 7.6427 - accuracy: 0.5016
 6432/25000 [======>.......................] - ETA: 49s - loss: 7.6475 - accuracy: 0.5012
 6464/25000 [======>.......................] - ETA: 49s - loss: 7.6405 - accuracy: 0.5017
 6496/25000 [======>.......................] - ETA: 49s - loss: 7.6501 - accuracy: 0.5011
 6528/25000 [======>.......................] - ETA: 49s - loss: 7.6431 - accuracy: 0.5015
 6560/25000 [======>.......................] - ETA: 49s - loss: 7.6503 - accuracy: 0.5011
 6592/25000 [======>.......................] - ETA: 49s - loss: 7.6527 - accuracy: 0.5009
 6624/25000 [======>.......................] - ETA: 49s - loss: 7.6550 - accuracy: 0.5008
 6656/25000 [======>.......................] - ETA: 49s - loss: 7.6620 - accuracy: 0.5003
 6688/25000 [=======>......................] - ETA: 49s - loss: 7.6506 - accuracy: 0.5010
 6720/25000 [=======>......................] - ETA: 48s - loss: 7.6529 - accuracy: 0.5009
 6752/25000 [=======>......................] - ETA: 48s - loss: 7.6553 - accuracy: 0.5007
 6784/25000 [=======>......................] - ETA: 48s - loss: 7.6508 - accuracy: 0.5010
 6816/25000 [=======>......................] - ETA: 48s - loss: 7.6486 - accuracy: 0.5012
 6848/25000 [=======>......................] - ETA: 48s - loss: 7.6554 - accuracy: 0.5007
 6880/25000 [=======>......................] - ETA: 48s - loss: 7.6555 - accuracy: 0.5007
 6912/25000 [=======>......................] - ETA: 48s - loss: 7.6533 - accuracy: 0.5009
 6944/25000 [=======>......................] - ETA: 48s - loss: 7.6490 - accuracy: 0.5012
 6976/25000 [=======>......................] - ETA: 48s - loss: 7.6490 - accuracy: 0.5011
 7008/25000 [=======>......................] - ETA: 48s - loss: 7.6426 - accuracy: 0.5016
 7040/25000 [=======>......................] - ETA: 48s - loss: 7.6448 - accuracy: 0.5014
 7072/25000 [=======>......................] - ETA: 47s - loss: 7.6471 - accuracy: 0.5013
 7104/25000 [=======>......................] - ETA: 47s - loss: 7.6537 - accuracy: 0.5008
 7136/25000 [=======>......................] - ETA: 47s - loss: 7.6602 - accuracy: 0.5004
 7168/25000 [=======>......................] - ETA: 47s - loss: 7.6709 - accuracy: 0.4997
 7200/25000 [=======>......................] - ETA: 47s - loss: 7.6602 - accuracy: 0.5004
 7232/25000 [=======>......................] - ETA: 47s - loss: 7.6539 - accuracy: 0.5008
 7264/25000 [=======>......................] - ETA: 47s - loss: 7.6497 - accuracy: 0.5011
 7296/25000 [=======>......................] - ETA: 47s - loss: 7.6519 - accuracy: 0.5010
 7328/25000 [=======>......................] - ETA: 47s - loss: 7.6562 - accuracy: 0.5007
 7360/25000 [=======>......................] - ETA: 47s - loss: 7.6666 - accuracy: 0.5000
 7392/25000 [=======>......................] - ETA: 47s - loss: 7.6625 - accuracy: 0.5003
 7424/25000 [=======>......................] - ETA: 46s - loss: 7.6584 - accuracy: 0.5005
 7456/25000 [=======>......................] - ETA: 46s - loss: 7.6481 - accuracy: 0.5012
 7488/25000 [=======>......................] - ETA: 46s - loss: 7.6441 - accuracy: 0.5015
 7520/25000 [========>.....................] - ETA: 46s - loss: 7.6422 - accuracy: 0.5016
 7552/25000 [========>.....................] - ETA: 46s - loss: 7.6423 - accuracy: 0.5016
 7584/25000 [========>.....................] - ETA: 46s - loss: 7.6363 - accuracy: 0.5020
 7616/25000 [========>.....................] - ETA: 46s - loss: 7.6344 - accuracy: 0.5021
 7648/25000 [========>.....................] - ETA: 46s - loss: 7.6285 - accuracy: 0.5025
 7680/25000 [========>.....................] - ETA: 46s - loss: 7.6267 - accuracy: 0.5026
 7712/25000 [========>.....................] - ETA: 46s - loss: 7.6408 - accuracy: 0.5017
 7744/25000 [========>.....................] - ETA: 45s - loss: 7.6409 - accuracy: 0.5017
 7776/25000 [========>.....................] - ETA: 45s - loss: 7.6390 - accuracy: 0.5018
 7808/25000 [========>.....................] - ETA: 45s - loss: 7.6450 - accuracy: 0.5014
 7840/25000 [========>.....................] - ETA: 45s - loss: 7.6490 - accuracy: 0.5011
 7872/25000 [========>.....................] - ETA: 45s - loss: 7.6491 - accuracy: 0.5011
 7904/25000 [========>.....................] - ETA: 45s - loss: 7.6492 - accuracy: 0.5011
 7936/25000 [========>.....................] - ETA: 45s - loss: 7.6570 - accuracy: 0.5006
 7968/25000 [========>.....................] - ETA: 45s - loss: 7.6589 - accuracy: 0.5005
 8000/25000 [========>.....................] - ETA: 45s - loss: 7.6570 - accuracy: 0.5006
 8032/25000 [========>.....................] - ETA: 45s - loss: 7.6590 - accuracy: 0.5005
 8064/25000 [========>.....................] - ETA: 45s - loss: 7.6590 - accuracy: 0.5005
 8096/25000 [========>.....................] - ETA: 44s - loss: 7.6571 - accuracy: 0.5006
 8128/25000 [========>.....................] - ETA: 44s - loss: 7.6610 - accuracy: 0.5004
 8160/25000 [========>.....................] - ETA: 44s - loss: 7.6535 - accuracy: 0.5009
 8192/25000 [========>.....................] - ETA: 44s - loss: 7.6647 - accuracy: 0.5001
 8224/25000 [========>.....................] - ETA: 44s - loss: 7.6648 - accuracy: 0.5001
 8256/25000 [========>.....................] - ETA: 44s - loss: 7.6666 - accuracy: 0.5000
 8288/25000 [========>.....................] - ETA: 44s - loss: 7.6740 - accuracy: 0.4995
 8320/25000 [========>.....................] - ETA: 44s - loss: 7.6685 - accuracy: 0.4999
 8352/25000 [=========>....................] - ETA: 44s - loss: 7.6648 - accuracy: 0.5001
 8384/25000 [=========>....................] - ETA: 44s - loss: 7.6703 - accuracy: 0.4998
 8416/25000 [=========>....................] - ETA: 44s - loss: 7.6648 - accuracy: 0.5001
 8448/25000 [=========>....................] - ETA: 43s - loss: 7.6702 - accuracy: 0.4998
 8480/25000 [=========>....................] - ETA: 43s - loss: 7.6720 - accuracy: 0.4996
 8512/25000 [=========>....................] - ETA: 43s - loss: 7.6648 - accuracy: 0.5001
 8544/25000 [=========>....................] - ETA: 43s - loss: 7.6594 - accuracy: 0.5005
 8576/25000 [=========>....................] - ETA: 43s - loss: 7.6613 - accuracy: 0.5003
 8608/25000 [=========>....................] - ETA: 43s - loss: 7.6648 - accuracy: 0.5001
 8640/25000 [=========>....................] - ETA: 43s - loss: 7.6648 - accuracy: 0.5001
 8672/25000 [=========>....................] - ETA: 43s - loss: 7.6684 - accuracy: 0.4999
 8704/25000 [=========>....................] - ETA: 43s - loss: 7.6684 - accuracy: 0.4999
 8736/25000 [=========>....................] - ETA: 43s - loss: 7.6772 - accuracy: 0.4993
 8768/25000 [=========>....................] - ETA: 42s - loss: 7.6719 - accuracy: 0.4997
 8800/25000 [=========>....................] - ETA: 42s - loss: 7.6701 - accuracy: 0.4998
 8832/25000 [=========>....................] - ETA: 42s - loss: 7.6736 - accuracy: 0.4995
 8864/25000 [=========>....................] - ETA: 42s - loss: 7.6787 - accuracy: 0.4992
 8896/25000 [=========>....................] - ETA: 42s - loss: 7.6752 - accuracy: 0.4994
 8928/25000 [=========>....................] - ETA: 42s - loss: 7.6769 - accuracy: 0.4993
 8960/25000 [=========>....................] - ETA: 42s - loss: 7.6735 - accuracy: 0.4996
 8992/25000 [=========>....................] - ETA: 42s - loss: 7.6734 - accuracy: 0.4996
 9024/25000 [=========>....................] - ETA: 42s - loss: 7.6734 - accuracy: 0.4996
 9056/25000 [=========>....................] - ETA: 42s - loss: 7.6819 - accuracy: 0.4990
 9088/25000 [=========>....................] - ETA: 42s - loss: 7.6869 - accuracy: 0.4987
 9120/25000 [=========>....................] - ETA: 42s - loss: 7.6868 - accuracy: 0.4987
 9152/25000 [=========>....................] - ETA: 41s - loss: 7.6783 - accuracy: 0.4992
 9184/25000 [==========>...................] - ETA: 41s - loss: 7.6850 - accuracy: 0.4988
 9216/25000 [==========>...................] - ETA: 41s - loss: 7.6916 - accuracy: 0.4984
 9248/25000 [==========>...................] - ETA: 41s - loss: 7.6915 - accuracy: 0.4984
 9280/25000 [==========>...................] - ETA: 41s - loss: 7.6947 - accuracy: 0.4982
 9312/25000 [==========>...................] - ETA: 41s - loss: 7.6864 - accuracy: 0.4987
 9344/25000 [==========>...................] - ETA: 41s - loss: 7.6962 - accuracy: 0.4981
 9376/25000 [==========>...................] - ETA: 41s - loss: 7.7026 - accuracy: 0.4977
 9408/25000 [==========>...................] - ETA: 41s - loss: 7.6992 - accuracy: 0.4979
 9440/25000 [==========>...................] - ETA: 41s - loss: 7.7056 - accuracy: 0.4975
 9472/25000 [==========>...................] - ETA: 41s - loss: 7.6941 - accuracy: 0.4982
 9504/25000 [==========>...................] - ETA: 40s - loss: 7.7005 - accuracy: 0.4978
 9536/25000 [==========>...................] - ETA: 40s - loss: 7.6972 - accuracy: 0.4980
 9568/25000 [==========>...................] - ETA: 40s - loss: 7.6955 - accuracy: 0.4981
 9600/25000 [==========>...................] - ETA: 40s - loss: 7.6986 - accuracy: 0.4979
 9632/25000 [==========>...................] - ETA: 40s - loss: 7.6937 - accuracy: 0.4982
 9664/25000 [==========>...................] - ETA: 40s - loss: 7.6888 - accuracy: 0.4986
 9696/25000 [==========>...................] - ETA: 40s - loss: 7.6935 - accuracy: 0.4982
 9728/25000 [==========>...................] - ETA: 40s - loss: 7.6903 - accuracy: 0.4985
 9760/25000 [==========>...................] - ETA: 40s - loss: 7.6808 - accuracy: 0.4991
 9792/25000 [==========>...................] - ETA: 40s - loss: 7.6791 - accuracy: 0.4992
 9824/25000 [==========>...................] - ETA: 39s - loss: 7.6807 - accuracy: 0.4991
 9856/25000 [==========>...................] - ETA: 39s - loss: 7.6837 - accuracy: 0.4989
 9888/25000 [==========>...................] - ETA: 39s - loss: 7.6790 - accuracy: 0.4992
 9920/25000 [==========>...................] - ETA: 39s - loss: 7.6790 - accuracy: 0.4992
 9952/25000 [==========>...................] - ETA: 39s - loss: 7.6759 - accuracy: 0.4994
 9984/25000 [==========>...................] - ETA: 39s - loss: 7.6804 - accuracy: 0.4991
10016/25000 [===========>..................] - ETA: 39s - loss: 7.6835 - accuracy: 0.4989
10048/25000 [===========>..................] - ETA: 39s - loss: 7.6849 - accuracy: 0.4988
10080/25000 [===========>..................] - ETA: 39s - loss: 7.6834 - accuracy: 0.4989
10112/25000 [===========>..................] - ETA: 39s - loss: 7.6788 - accuracy: 0.4992
10144/25000 [===========>..................] - ETA: 39s - loss: 7.6817 - accuracy: 0.4990
10176/25000 [===========>..................] - ETA: 39s - loss: 7.6817 - accuracy: 0.4990
10208/25000 [===========>..................] - ETA: 38s - loss: 7.6771 - accuracy: 0.4993
10240/25000 [===========>..................] - ETA: 38s - loss: 7.6786 - accuracy: 0.4992
10272/25000 [===========>..................] - ETA: 38s - loss: 7.6830 - accuracy: 0.4989
10304/25000 [===========>..................] - ETA: 38s - loss: 7.6860 - accuracy: 0.4987
10336/25000 [===========>..................] - ETA: 38s - loss: 7.6859 - accuracy: 0.4987
10368/25000 [===========>..................] - ETA: 38s - loss: 7.6873 - accuracy: 0.4986
10400/25000 [===========>..................] - ETA: 38s - loss: 7.6976 - accuracy: 0.4980
10432/25000 [===========>..................] - ETA: 38s - loss: 7.6990 - accuracy: 0.4979
10464/25000 [===========>..................] - ETA: 38s - loss: 7.6959 - accuracy: 0.4981
10496/25000 [===========>..................] - ETA: 38s - loss: 7.6988 - accuracy: 0.4979
10528/25000 [===========>..................] - ETA: 38s - loss: 7.7001 - accuracy: 0.4978
10560/25000 [===========>..................] - ETA: 37s - loss: 7.6986 - accuracy: 0.4979
10592/25000 [===========>..................] - ETA: 37s - loss: 7.6970 - accuracy: 0.4980
10624/25000 [===========>..................] - ETA: 37s - loss: 7.6984 - accuracy: 0.4979
10656/25000 [===========>..................] - ETA: 37s - loss: 7.6925 - accuracy: 0.4983
10688/25000 [===========>..................] - ETA: 37s - loss: 7.6924 - accuracy: 0.4983
10720/25000 [===========>..................] - ETA: 37s - loss: 7.6952 - accuracy: 0.4981
10752/25000 [===========>..................] - ETA: 37s - loss: 7.6923 - accuracy: 0.4983
10784/25000 [===========>..................] - ETA: 37s - loss: 7.6879 - accuracy: 0.4986
10816/25000 [===========>..................] - ETA: 37s - loss: 7.6879 - accuracy: 0.4986
10848/25000 [============>.................] - ETA: 37s - loss: 7.6850 - accuracy: 0.4988
10880/25000 [============>.................] - ETA: 37s - loss: 7.6863 - accuracy: 0.4987
10912/25000 [============>.................] - ETA: 37s - loss: 7.6891 - accuracy: 0.4985
10944/25000 [============>.................] - ETA: 36s - loss: 7.6890 - accuracy: 0.4985
10976/25000 [============>.................] - ETA: 36s - loss: 7.6862 - accuracy: 0.4987
11008/25000 [============>.................] - ETA: 36s - loss: 7.6917 - accuracy: 0.4984
11040/25000 [============>.................] - ETA: 36s - loss: 7.6916 - accuracy: 0.4984
11072/25000 [============>.................] - ETA: 36s - loss: 7.6929 - accuracy: 0.4983
11104/25000 [============>.................] - ETA: 36s - loss: 7.6942 - accuracy: 0.4982
11136/25000 [============>.................] - ETA: 36s - loss: 7.6983 - accuracy: 0.4979
11168/25000 [============>.................] - ETA: 36s - loss: 7.6996 - accuracy: 0.4979
11200/25000 [============>.................] - ETA: 36s - loss: 7.7008 - accuracy: 0.4978
11232/25000 [============>.................] - ETA: 36s - loss: 7.6994 - accuracy: 0.4979
11264/25000 [============>.................] - ETA: 36s - loss: 7.6979 - accuracy: 0.4980
11296/25000 [============>.................] - ETA: 35s - loss: 7.6951 - accuracy: 0.4981
11328/25000 [============>.................] - ETA: 35s - loss: 7.6937 - accuracy: 0.4982
11360/25000 [============>.................] - ETA: 35s - loss: 7.7017 - accuracy: 0.4977
11392/25000 [============>.................] - ETA: 35s - loss: 7.6989 - accuracy: 0.4979
11424/25000 [============>.................] - ETA: 35s - loss: 7.7015 - accuracy: 0.4977
11456/25000 [============>.................] - ETA: 35s - loss: 7.7081 - accuracy: 0.4973
11488/25000 [============>.................] - ETA: 35s - loss: 7.7080 - accuracy: 0.4973
11520/25000 [============>.................] - ETA: 35s - loss: 7.7079 - accuracy: 0.4973
11552/25000 [============>.................] - ETA: 35s - loss: 7.7091 - accuracy: 0.4972
11584/25000 [============>.................] - ETA: 35s - loss: 7.7103 - accuracy: 0.4972
11616/25000 [============>.................] - ETA: 35s - loss: 7.7075 - accuracy: 0.4973
11648/25000 [============>.................] - ETA: 35s - loss: 7.7114 - accuracy: 0.4971
11680/25000 [=============>................] - ETA: 34s - loss: 7.7139 - accuracy: 0.4969
11712/25000 [=============>................] - ETA: 34s - loss: 7.7216 - accuracy: 0.4964
11744/25000 [=============>................] - ETA: 34s - loss: 7.7254 - accuracy: 0.4962
11776/25000 [=============>................] - ETA: 34s - loss: 7.7304 - accuracy: 0.4958
11808/25000 [=============>................] - ETA: 34s - loss: 7.7315 - accuracy: 0.4958
11840/25000 [=============>................] - ETA: 34s - loss: 7.7340 - accuracy: 0.4956
11872/25000 [=============>................] - ETA: 34s - loss: 7.7325 - accuracy: 0.4957
11904/25000 [=============>................] - ETA: 34s - loss: 7.7297 - accuracy: 0.4959
11936/25000 [=============>................] - ETA: 34s - loss: 7.7309 - accuracy: 0.4958
11968/25000 [=============>................] - ETA: 34s - loss: 7.7358 - accuracy: 0.4955
12000/25000 [=============>................] - ETA: 34s - loss: 7.7369 - accuracy: 0.4954
12032/25000 [=============>................] - ETA: 33s - loss: 7.7342 - accuracy: 0.4956
12064/25000 [=============>................] - ETA: 33s - loss: 7.7365 - accuracy: 0.4954
12096/25000 [=============>................] - ETA: 33s - loss: 7.7376 - accuracy: 0.4954
12128/25000 [=============>................] - ETA: 33s - loss: 7.7324 - accuracy: 0.4957
12160/25000 [=============>................] - ETA: 33s - loss: 7.7234 - accuracy: 0.4963
12192/25000 [=============>................] - ETA: 33s - loss: 7.7207 - accuracy: 0.4965
12224/25000 [=============>................] - ETA: 33s - loss: 7.7093 - accuracy: 0.4972
12256/25000 [=============>................] - ETA: 33s - loss: 7.6979 - accuracy: 0.4980
12288/25000 [=============>................] - ETA: 33s - loss: 7.6953 - accuracy: 0.4981
12320/25000 [=============>................] - ETA: 33s - loss: 7.6990 - accuracy: 0.4979
12352/25000 [=============>................] - ETA: 33s - loss: 7.6964 - accuracy: 0.4981
12384/25000 [=============>................] - ETA: 33s - loss: 7.6976 - accuracy: 0.4980
12416/25000 [=============>................] - ETA: 32s - loss: 7.6938 - accuracy: 0.4982
12448/25000 [=============>................] - ETA: 32s - loss: 7.6913 - accuracy: 0.4984
12480/25000 [=============>................] - ETA: 32s - loss: 7.6949 - accuracy: 0.4982
12512/25000 [==============>...............] - ETA: 32s - loss: 7.6936 - accuracy: 0.4982
12544/25000 [==============>...............] - ETA: 32s - loss: 7.6947 - accuracy: 0.4982
12576/25000 [==============>...............] - ETA: 32s - loss: 7.6983 - accuracy: 0.4979
12608/25000 [==============>...............] - ETA: 32s - loss: 7.6982 - accuracy: 0.4979
12640/25000 [==============>...............] - ETA: 32s - loss: 7.6957 - accuracy: 0.4981
12672/25000 [==============>...............] - ETA: 32s - loss: 7.6944 - accuracy: 0.4982
12704/25000 [==============>...............] - ETA: 32s - loss: 7.6968 - accuracy: 0.4980
12736/25000 [==============>...............] - ETA: 32s - loss: 7.6979 - accuracy: 0.4980
12768/25000 [==============>...............] - ETA: 32s - loss: 7.6966 - accuracy: 0.4980
12800/25000 [==============>...............] - ETA: 31s - loss: 7.7014 - accuracy: 0.4977
12832/25000 [==============>...............] - ETA: 31s - loss: 7.7037 - accuracy: 0.4976
12864/25000 [==============>...............] - ETA: 31s - loss: 7.7036 - accuracy: 0.4976
12896/25000 [==============>...............] - ETA: 31s - loss: 7.7035 - accuracy: 0.4976
12928/25000 [==============>...............] - ETA: 31s - loss: 7.7069 - accuracy: 0.4974
12960/25000 [==============>...............] - ETA: 31s - loss: 7.7080 - accuracy: 0.4973
12992/25000 [==============>...............] - ETA: 31s - loss: 7.7138 - accuracy: 0.4969
13024/25000 [==============>...............] - ETA: 31s - loss: 7.7102 - accuracy: 0.4972
13056/25000 [==============>...............] - ETA: 31s - loss: 7.7112 - accuracy: 0.4971
13088/25000 [==============>...............] - ETA: 31s - loss: 7.7158 - accuracy: 0.4968
13120/25000 [==============>...............] - ETA: 31s - loss: 7.7110 - accuracy: 0.4971
13152/25000 [==============>...............] - ETA: 30s - loss: 7.7133 - accuracy: 0.4970
13184/25000 [==============>...............] - ETA: 30s - loss: 7.7143 - accuracy: 0.4969
13216/25000 [==============>...............] - ETA: 30s - loss: 7.7200 - accuracy: 0.4965
13248/25000 [==============>...............] - ETA: 30s - loss: 7.7129 - accuracy: 0.4970
13280/25000 [==============>...............] - ETA: 30s - loss: 7.7140 - accuracy: 0.4969
13312/25000 [==============>...............] - ETA: 30s - loss: 7.7173 - accuracy: 0.4967
13344/25000 [===============>..............] - ETA: 30s - loss: 7.7206 - accuracy: 0.4965
13376/25000 [===============>..............] - ETA: 30s - loss: 7.7182 - accuracy: 0.4966
13408/25000 [===============>..............] - ETA: 30s - loss: 7.7261 - accuracy: 0.4961
13440/25000 [===============>..............] - ETA: 30s - loss: 7.7248 - accuracy: 0.4962
13472/25000 [===============>..............] - ETA: 30s - loss: 7.7201 - accuracy: 0.4965
13504/25000 [===============>..............] - ETA: 30s - loss: 7.7223 - accuracy: 0.4964
13536/25000 [===============>..............] - ETA: 29s - loss: 7.7210 - accuracy: 0.4965
13568/25000 [===============>..............] - ETA: 29s - loss: 7.7243 - accuracy: 0.4962
13600/25000 [===============>..............] - ETA: 29s - loss: 7.7219 - accuracy: 0.4964
13632/25000 [===============>..............] - ETA: 29s - loss: 7.7195 - accuracy: 0.4966
13664/25000 [===============>..............] - ETA: 29s - loss: 7.7171 - accuracy: 0.4967
13696/25000 [===============>..............] - ETA: 29s - loss: 7.7170 - accuracy: 0.4967
13728/25000 [===============>..............] - ETA: 29s - loss: 7.7180 - accuracy: 0.4966
13760/25000 [===============>..............] - ETA: 29s - loss: 7.7190 - accuracy: 0.4966
13792/25000 [===============>..............] - ETA: 29s - loss: 7.7222 - accuracy: 0.4964
13824/25000 [===============>..............] - ETA: 29s - loss: 7.7199 - accuracy: 0.4965
13856/25000 [===============>..............] - ETA: 29s - loss: 7.7208 - accuracy: 0.4965
13888/25000 [===============>..............] - ETA: 29s - loss: 7.7229 - accuracy: 0.4963
13920/25000 [===============>..............] - ETA: 28s - loss: 7.7294 - accuracy: 0.4959
13952/25000 [===============>..............] - ETA: 28s - loss: 7.7304 - accuracy: 0.4958
13984/25000 [===============>..............] - ETA: 28s - loss: 7.7258 - accuracy: 0.4961
14016/25000 [===============>..............] - ETA: 28s - loss: 7.7268 - accuracy: 0.4961
14048/25000 [===============>..............] - ETA: 28s - loss: 7.7288 - accuracy: 0.4959
14080/25000 [===============>..............] - ETA: 28s - loss: 7.7254 - accuracy: 0.4962
14112/25000 [===============>..............] - ETA: 28s - loss: 7.7231 - accuracy: 0.4963
14144/25000 [===============>..............] - ETA: 28s - loss: 7.7219 - accuracy: 0.4964
14176/25000 [================>.............] - ETA: 28s - loss: 7.7218 - accuracy: 0.4964
14208/25000 [================>.............] - ETA: 28s - loss: 7.7227 - accuracy: 0.4963
14240/25000 [================>.............] - ETA: 28s - loss: 7.7172 - accuracy: 0.4967
14272/25000 [================>.............] - ETA: 27s - loss: 7.7160 - accuracy: 0.4968
14304/25000 [================>.............] - ETA: 27s - loss: 7.7127 - accuracy: 0.4970
14336/25000 [================>.............] - ETA: 27s - loss: 7.7105 - accuracy: 0.4971
14368/25000 [================>.............] - ETA: 27s - loss: 7.7040 - accuracy: 0.4976
14400/25000 [================>.............] - ETA: 27s - loss: 7.7050 - accuracy: 0.4975
14432/25000 [================>.............] - ETA: 27s - loss: 7.7049 - accuracy: 0.4975
14464/25000 [================>.............] - ETA: 27s - loss: 7.7037 - accuracy: 0.4976
14496/25000 [================>.............] - ETA: 27s - loss: 7.7047 - accuracy: 0.4975
14528/25000 [================>.............] - ETA: 27s - loss: 7.7057 - accuracy: 0.4975
14560/25000 [================>.............] - ETA: 27s - loss: 7.7087 - accuracy: 0.4973
14592/25000 [================>.............] - ETA: 27s - loss: 7.7108 - accuracy: 0.4971
14624/25000 [================>.............] - ETA: 27s - loss: 7.7107 - accuracy: 0.4971
14656/25000 [================>.............] - ETA: 26s - loss: 7.7158 - accuracy: 0.4968
14688/25000 [================>.............] - ETA: 26s - loss: 7.7136 - accuracy: 0.4969
14720/25000 [================>.............] - ETA: 26s - loss: 7.7187 - accuracy: 0.4966
14752/25000 [================>.............] - ETA: 26s - loss: 7.7165 - accuracy: 0.4967
14784/25000 [================>.............] - ETA: 26s - loss: 7.7123 - accuracy: 0.4970
14816/25000 [================>.............] - ETA: 26s - loss: 7.7101 - accuracy: 0.4972
14848/25000 [================>.............] - ETA: 26s - loss: 7.7131 - accuracy: 0.4970
14880/25000 [================>.............] - ETA: 26s - loss: 7.7130 - accuracy: 0.4970
14912/25000 [================>.............] - ETA: 26s - loss: 7.7170 - accuracy: 0.4967
14944/25000 [================>.............] - ETA: 26s - loss: 7.7138 - accuracy: 0.4969
14976/25000 [================>.............] - ETA: 26s - loss: 7.7178 - accuracy: 0.4967
15008/25000 [=================>............] - ETA: 26s - loss: 7.7146 - accuracy: 0.4969
15040/25000 [=================>............] - ETA: 25s - loss: 7.7186 - accuracy: 0.4966
15072/25000 [=================>............] - ETA: 25s - loss: 7.7165 - accuracy: 0.4967
15104/25000 [=================>............] - ETA: 25s - loss: 7.7164 - accuracy: 0.4968
15136/25000 [=================>............] - ETA: 25s - loss: 7.7183 - accuracy: 0.4966
15168/25000 [=================>............] - ETA: 25s - loss: 7.7162 - accuracy: 0.4968
15200/25000 [=================>............] - ETA: 25s - loss: 7.7110 - accuracy: 0.4971
15232/25000 [=================>............] - ETA: 25s - loss: 7.7139 - accuracy: 0.4969
15264/25000 [=================>............] - ETA: 25s - loss: 7.7138 - accuracy: 0.4969
15296/25000 [=================>............] - ETA: 25s - loss: 7.7137 - accuracy: 0.4969
15328/25000 [=================>............] - ETA: 25s - loss: 7.7146 - accuracy: 0.4969
15360/25000 [=================>............] - ETA: 25s - loss: 7.7155 - accuracy: 0.4968
15392/25000 [=================>............] - ETA: 25s - loss: 7.7164 - accuracy: 0.4968
15424/25000 [=================>............] - ETA: 24s - loss: 7.7133 - accuracy: 0.4970
15456/25000 [=================>............] - ETA: 24s - loss: 7.7132 - accuracy: 0.4970
15488/25000 [=================>............] - ETA: 24s - loss: 7.7092 - accuracy: 0.4972
15520/25000 [=================>............] - ETA: 24s - loss: 7.7111 - accuracy: 0.4971
15552/25000 [=================>............] - ETA: 24s - loss: 7.7070 - accuracy: 0.4974
15584/25000 [=================>............] - ETA: 24s - loss: 7.7060 - accuracy: 0.4974
15616/25000 [=================>............] - ETA: 24s - loss: 7.7029 - accuracy: 0.4976
15648/25000 [=================>............] - ETA: 24s - loss: 7.7009 - accuracy: 0.4978
15680/25000 [=================>............] - ETA: 24s - loss: 7.6979 - accuracy: 0.4980
15712/25000 [=================>............] - ETA: 24s - loss: 7.6969 - accuracy: 0.4980
15744/25000 [=================>............] - ETA: 24s - loss: 7.6949 - accuracy: 0.4982
15776/25000 [=================>............] - ETA: 24s - loss: 7.6938 - accuracy: 0.4982
15808/25000 [=================>............] - ETA: 23s - loss: 7.6928 - accuracy: 0.4983
15840/25000 [==================>...........] - ETA: 23s - loss: 7.6899 - accuracy: 0.4985
15872/25000 [==================>...........] - ETA: 23s - loss: 7.6927 - accuracy: 0.4983
15904/25000 [==================>...........] - ETA: 23s - loss: 7.6984 - accuracy: 0.4979
15936/25000 [==================>...........] - ETA: 23s - loss: 7.7022 - accuracy: 0.4977
15968/25000 [==================>...........] - ETA: 23s - loss: 7.7031 - accuracy: 0.4976
16000/25000 [==================>...........] - ETA: 23s - loss: 7.7050 - accuracy: 0.4975
16032/25000 [==================>...........] - ETA: 23s - loss: 7.7030 - accuracy: 0.4976
16064/25000 [==================>...........] - ETA: 23s - loss: 7.7019 - accuracy: 0.4977
16096/25000 [==================>...........] - ETA: 23s - loss: 7.7009 - accuracy: 0.4978
16128/25000 [==================>...........] - ETA: 23s - loss: 7.7008 - accuracy: 0.4978
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6960 - accuracy: 0.4981
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6988 - accuracy: 0.4979
16224/25000 [==================>...........] - ETA: 22s - loss: 7.6988 - accuracy: 0.4979
16256/25000 [==================>...........] - ETA: 22s - loss: 7.6996 - accuracy: 0.4978
16288/25000 [==================>...........] - ETA: 22s - loss: 7.6967 - accuracy: 0.4980
16320/25000 [==================>...........] - ETA: 22s - loss: 7.6995 - accuracy: 0.4979
16352/25000 [==================>...........] - ETA: 22s - loss: 7.6976 - accuracy: 0.4980
16384/25000 [==================>...........] - ETA: 22s - loss: 7.6956 - accuracy: 0.4981
16416/25000 [==================>...........] - ETA: 22s - loss: 7.6974 - accuracy: 0.4980
16448/25000 [==================>...........] - ETA: 22s - loss: 7.6965 - accuracy: 0.4981
16480/25000 [==================>...........] - ETA: 22s - loss: 7.6936 - accuracy: 0.4982
16512/25000 [==================>...........] - ETA: 22s - loss: 7.6908 - accuracy: 0.4984
16544/25000 [==================>...........] - ETA: 21s - loss: 7.6870 - accuracy: 0.4987
16576/25000 [==================>...........] - ETA: 21s - loss: 7.6870 - accuracy: 0.4987
16608/25000 [==================>...........] - ETA: 21s - loss: 7.6842 - accuracy: 0.4989
16640/25000 [==================>...........] - ETA: 21s - loss: 7.6832 - accuracy: 0.4989
16672/25000 [===================>..........] - ETA: 21s - loss: 7.6869 - accuracy: 0.4987
16704/25000 [===================>..........] - ETA: 21s - loss: 7.6905 - accuracy: 0.4984
16736/25000 [===================>..........] - ETA: 21s - loss: 7.6886 - accuracy: 0.4986
16768/25000 [===================>..........] - ETA: 21s - loss: 7.6858 - accuracy: 0.4987
16800/25000 [===================>..........] - ETA: 21s - loss: 7.6867 - accuracy: 0.4987
16832/25000 [===================>..........] - ETA: 21s - loss: 7.6848 - accuracy: 0.4988
16864/25000 [===================>..........] - ETA: 21s - loss: 7.6839 - accuracy: 0.4989
16896/25000 [===================>..........] - ETA: 21s - loss: 7.6866 - accuracy: 0.4987
16928/25000 [===================>..........] - ETA: 20s - loss: 7.6838 - accuracy: 0.4989
16960/25000 [===================>..........] - ETA: 20s - loss: 7.6847 - accuracy: 0.4988
16992/25000 [===================>..........] - ETA: 20s - loss: 7.6856 - accuracy: 0.4988
17024/25000 [===================>..........] - ETA: 20s - loss: 7.6873 - accuracy: 0.4986
17056/25000 [===================>..........] - ETA: 20s - loss: 7.6909 - accuracy: 0.4984
17088/25000 [===================>..........] - ETA: 20s - loss: 7.6891 - accuracy: 0.4985
17120/25000 [===================>..........] - ETA: 20s - loss: 7.6899 - accuracy: 0.4985
17152/25000 [===================>..........] - ETA: 20s - loss: 7.6934 - accuracy: 0.4983
17184/25000 [===================>..........] - ETA: 20s - loss: 7.6898 - accuracy: 0.4985
17216/25000 [===================>..........] - ETA: 20s - loss: 7.6889 - accuracy: 0.4985
17248/25000 [===================>..........] - ETA: 20s - loss: 7.6906 - accuracy: 0.4984
17280/25000 [===================>..........] - ETA: 20s - loss: 7.6906 - accuracy: 0.4984
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6941 - accuracy: 0.4982
17344/25000 [===================>..........] - ETA: 19s - loss: 7.6976 - accuracy: 0.4980
17376/25000 [===================>..........] - ETA: 19s - loss: 7.7028 - accuracy: 0.4976
17408/25000 [===================>..........] - ETA: 19s - loss: 7.7027 - accuracy: 0.4976
17440/25000 [===================>..........] - ETA: 19s - loss: 7.7009 - accuracy: 0.4978
17472/25000 [===================>..........] - ETA: 19s - loss: 7.6973 - accuracy: 0.4980
17504/25000 [====================>.........] - ETA: 19s - loss: 7.6982 - accuracy: 0.4979
17536/25000 [====================>.........] - ETA: 19s - loss: 7.7007 - accuracy: 0.4978
17568/25000 [====================>.........] - ETA: 19s - loss: 7.6945 - accuracy: 0.4982
17600/25000 [====================>.........] - ETA: 19s - loss: 7.6945 - accuracy: 0.4982
17632/25000 [====================>.........] - ETA: 19s - loss: 7.6901 - accuracy: 0.4985
17664/25000 [====================>.........] - ETA: 19s - loss: 7.6883 - accuracy: 0.4986
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6909 - accuracy: 0.4984
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6900 - accuracy: 0.4985
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6865 - accuracy: 0.4987
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6847 - accuracy: 0.4988
17824/25000 [====================>.........] - ETA: 18s - loss: 7.6838 - accuracy: 0.4989
17856/25000 [====================>.........] - ETA: 18s - loss: 7.6847 - accuracy: 0.4988
17888/25000 [====================>.........] - ETA: 18s - loss: 7.6906 - accuracy: 0.4984
17920/25000 [====================>.........] - ETA: 18s - loss: 7.6880 - accuracy: 0.4986
17952/25000 [====================>.........] - ETA: 18s - loss: 7.6871 - accuracy: 0.4987
17984/25000 [====================>.........] - ETA: 18s - loss: 7.6862 - accuracy: 0.4987
18016/25000 [====================>.........] - ETA: 18s - loss: 7.6845 - accuracy: 0.4988
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6845 - accuracy: 0.4988
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6802 - accuracy: 0.4991
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6810 - accuracy: 0.4991
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6818 - accuracy: 0.4990
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6801 - accuracy: 0.4991
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6793 - accuracy: 0.4992
18240/25000 [====================>.........] - ETA: 17s - loss: 7.6750 - accuracy: 0.4995
18272/25000 [====================>.........] - ETA: 17s - loss: 7.6750 - accuracy: 0.4995
18304/25000 [====================>.........] - ETA: 17s - loss: 7.6758 - accuracy: 0.4994
18336/25000 [=====================>........] - ETA: 17s - loss: 7.6750 - accuracy: 0.4995
18368/25000 [=====================>........] - ETA: 17s - loss: 7.6750 - accuracy: 0.4995
18400/25000 [=====================>........] - ETA: 17s - loss: 7.6741 - accuracy: 0.4995
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6758 - accuracy: 0.4994
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6741 - accuracy: 0.4995
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6766 - accuracy: 0.4994
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6732 - accuracy: 0.4996
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6757 - accuracy: 0.4994
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6773 - accuracy: 0.4993
18624/25000 [=====================>........] - ETA: 16s - loss: 7.6749 - accuracy: 0.4995
18656/25000 [=====================>........] - ETA: 16s - loss: 7.6732 - accuracy: 0.4996
18688/25000 [=====================>........] - ETA: 16s - loss: 7.6707 - accuracy: 0.4997
18720/25000 [=====================>........] - ETA: 16s - loss: 7.6674 - accuracy: 0.4999
18752/25000 [=====================>........] - ETA: 16s - loss: 7.6666 - accuracy: 0.5000
18784/25000 [=====================>........] - ETA: 16s - loss: 7.6699 - accuracy: 0.4998
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6715 - accuracy: 0.4997
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6707 - accuracy: 0.4997
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6699 - accuracy: 0.4998
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6674 - accuracy: 0.4999
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6682 - accuracy: 0.4999
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6666 - accuracy: 0.5000
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6626 - accuracy: 0.5003
19040/25000 [=====================>........] - ETA: 15s - loss: 7.6586 - accuracy: 0.5005
19072/25000 [=====================>........] - ETA: 15s - loss: 7.6610 - accuracy: 0.5004
19104/25000 [=====================>........] - ETA: 15s - loss: 7.6618 - accuracy: 0.5003
19136/25000 [=====================>........] - ETA: 15s - loss: 7.6610 - accuracy: 0.5004
19168/25000 [======================>.......] - ETA: 15s - loss: 7.6658 - accuracy: 0.5001
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6642 - accuracy: 0.5002
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6642 - accuracy: 0.5002
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6610 - accuracy: 0.5004
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6611 - accuracy: 0.5004
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6619 - accuracy: 0.5003
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6635 - accuracy: 0.5002
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6650 - accuracy: 0.5001
19424/25000 [======================>.......] - ETA: 14s - loss: 7.6627 - accuracy: 0.5003
19456/25000 [======================>.......] - ETA: 14s - loss: 7.6666 - accuracy: 0.5000
19488/25000 [======================>.......] - ETA: 14s - loss: 7.6666 - accuracy: 0.5000
19520/25000 [======================>.......] - ETA: 14s - loss: 7.6666 - accuracy: 0.5000
19552/25000 [======================>.......] - ETA: 14s - loss: 7.6690 - accuracy: 0.4998
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6674 - accuracy: 0.4999
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6674 - accuracy: 0.4999
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6697 - accuracy: 0.4998
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6674 - accuracy: 0.4999
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6635 - accuracy: 0.5002
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6627 - accuracy: 0.5003
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6627 - accuracy: 0.5003
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6627 - accuracy: 0.5003
19840/25000 [======================>.......] - ETA: 13s - loss: 7.6620 - accuracy: 0.5003
19872/25000 [======================>.......] - ETA: 13s - loss: 7.6612 - accuracy: 0.5004
19904/25000 [======================>.......] - ETA: 13s - loss: 7.6574 - accuracy: 0.5006
19936/25000 [======================>.......] - ETA: 13s - loss: 7.6566 - accuracy: 0.5007
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6566 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6513 - accuracy: 0.5010
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6490 - accuracy: 0.5011
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6544 - accuracy: 0.5008
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6536 - accuracy: 0.5008
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6552 - accuracy: 0.5007
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6613 - accuracy: 0.5003
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6598 - accuracy: 0.5004
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6621 - accuracy: 0.5003
20256/25000 [=======================>......] - ETA: 12s - loss: 7.6628 - accuracy: 0.5002
20288/25000 [=======================>......] - ETA: 12s - loss: 7.6613 - accuracy: 0.5003
20320/25000 [=======================>......] - ETA: 12s - loss: 7.6606 - accuracy: 0.5004
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6606 - accuracy: 0.5004
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6606 - accuracy: 0.5004
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6621 - accuracy: 0.5003
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6621 - accuracy: 0.5003
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6629 - accuracy: 0.5002
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6644 - accuracy: 0.5001
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6681 - accuracy: 0.4999
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6674 - accuracy: 0.5000
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6681 - accuracy: 0.4999
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6674 - accuracy: 0.5000
20672/25000 [=======================>......] - ETA: 11s - loss: 7.6651 - accuracy: 0.5001
20704/25000 [=======================>......] - ETA: 11s - loss: 7.6681 - accuracy: 0.4999
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6674 - accuracy: 0.5000
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6659 - accuracy: 0.5000
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6681 - accuracy: 0.4999
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6688 - accuracy: 0.4999
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6659 - accuracy: 0.5000
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6644 - accuracy: 0.5001
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6644 - accuracy: 0.5001
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6615 - accuracy: 0.5003
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6622 - accuracy: 0.5003
21056/25000 [========================>.....] - ETA: 10s - loss: 7.6608 - accuracy: 0.5004
21088/25000 [========================>.....] - ETA: 10s - loss: 7.6601 - accuracy: 0.5004
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6608 - accuracy: 0.5004 
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6586 - accuracy: 0.5005
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6594 - accuracy: 0.5005
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6608 - accuracy: 0.5004
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6616 - accuracy: 0.5003
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6601 - accuracy: 0.5004
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6601 - accuracy: 0.5004
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6594 - accuracy: 0.5005
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6630 - accuracy: 0.5002
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6645 - accuracy: 0.5001
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6666 - accuracy: 0.5000
21472/25000 [========================>.....] - ETA: 9s - loss: 7.6638 - accuracy: 0.5002
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6616 - accuracy: 0.5003
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6645 - accuracy: 0.5001
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6638 - accuracy: 0.5002
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6645 - accuracy: 0.5001
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6659 - accuracy: 0.5000
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6645 - accuracy: 0.5001
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6659 - accuracy: 0.5000
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6603 - accuracy: 0.5004
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6603 - accuracy: 0.5004
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6610 - accuracy: 0.5004
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6610 - accuracy: 0.5004
21856/25000 [=========================>....] - ETA: 8s - loss: 7.6596 - accuracy: 0.5005
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6624 - accuracy: 0.5003
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6638 - accuracy: 0.5002
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6645 - accuracy: 0.5001
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6652 - accuracy: 0.5001
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6652 - accuracy: 0.5001
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6652 - accuracy: 0.5001
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6659 - accuracy: 0.5000
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6645 - accuracy: 0.5001
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6652 - accuracy: 0.5001
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6618 - accuracy: 0.5003
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6604 - accuracy: 0.5004
22272/25000 [=========================>....] - ETA: 7s - loss: 7.6584 - accuracy: 0.5005
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6564 - accuracy: 0.5007
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6557 - accuracy: 0.5007
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6557 - accuracy: 0.5007
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6571 - accuracy: 0.5006
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6585 - accuracy: 0.5005
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6612 - accuracy: 0.5004
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6619 - accuracy: 0.5003
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6626 - accuracy: 0.5003
22656/25000 [==========================>...] - ETA: 6s - loss: 7.6632 - accuracy: 0.5002
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6632 - accuracy: 0.5002
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6639 - accuracy: 0.5002
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6646 - accuracy: 0.5001
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6659 - accuracy: 0.5000
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6653 - accuracy: 0.5001
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6626 - accuracy: 0.5003
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6633 - accuracy: 0.5002
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6633 - accuracy: 0.5002
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6613 - accuracy: 0.5003
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6619 - accuracy: 0.5003
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6626 - accuracy: 0.5003
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6646 - accuracy: 0.5001
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6706 - accuracy: 0.4997
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6706 - accuracy: 0.4997
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6712 - accuracy: 0.4997
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6738 - accuracy: 0.4995
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6725 - accuracy: 0.4996
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6745 - accuracy: 0.4995
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6725 - accuracy: 0.4996
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6718 - accuracy: 0.4997
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6718 - accuracy: 0.4997
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6711 - accuracy: 0.4997
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6705 - accuracy: 0.4997
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6724 - accuracy: 0.4996
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6742 - accuracy: 0.4995
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
24192/25000 [============================>.] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
24224/25000 [============================>.] - ETA: 1s - loss: 7.6736 - accuracy: 0.4995
24256/25000 [============================>.] - ETA: 1s - loss: 7.6748 - accuracy: 0.4995
24288/25000 [============================>.] - ETA: 1s - loss: 7.6736 - accuracy: 0.4995
24320/25000 [============================>.] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24352/25000 [============================>.] - ETA: 1s - loss: 7.6754 - accuracy: 0.4994
24384/25000 [============================>.] - ETA: 1s - loss: 7.6767 - accuracy: 0.4993
24416/25000 [============================>.] - ETA: 1s - loss: 7.6742 - accuracy: 0.4995
24448/25000 [============================>.] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24480/25000 [============================>.] - ETA: 1s - loss: 7.6741 - accuracy: 0.4995
24512/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24544/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
24576/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
24608/25000 [============================>.] - ETA: 1s - loss: 7.6747 - accuracy: 0.4995
24640/25000 [============================>.] - ETA: 0s - loss: 7.6728 - accuracy: 0.4996
24672/25000 [============================>.] - ETA: 0s - loss: 7.6753 - accuracy: 0.4994
24704/25000 [============================>.] - ETA: 0s - loss: 7.6728 - accuracy: 0.4996
24736/25000 [============================>.] - ETA: 0s - loss: 7.6728 - accuracy: 0.4996
24768/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24800/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24832/25000 [============================>.] - ETA: 0s - loss: 7.6709 - accuracy: 0.4997
24864/25000 [============================>.] - ETA: 0s - loss: 7.6709 - accuracy: 0.4997
24896/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24960/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 75s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
