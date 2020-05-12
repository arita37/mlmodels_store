
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/1f36c00be3a0e28b634b1ba3bd0de78bfdb3dba5

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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:49<01:13, 24.61s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.3629239917270541, 'embedding_size_factor': 1.3787844476755993, 'layers.choice': 1, 'learning_rate': 0.0032274120991338077, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 2.3980275545674953e-12} and reward: 0.38
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd7:%\x8c\xd9\xb1\xb7X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6\x0f\x80G\xf0\x00\x80X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?jp_\xadS\xec\x18X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\x85\x17\xe0\xc4\xa5\x7f\xcbu.' and reward: 0.38
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd7:%\x8c\xd9\xb1\xb7X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6\x0f\x80G\xf0\x00\x80X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?jp_\xadS\xec\x18X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\x85\x17\xe0\xc4\xa5\x7f\xcbu.' and reward: 0.38
 60%|██████    | 3/5 [02:07<01:21, 40.82s/it] 60%|██████    | 3/5 [02:07<01:25, 42.63s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.3221347561949043, 'embedding_size_factor': 1.3966064445426436, 'layers.choice': 1, 'learning_rate': 0.0098758220539243, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 4.653701910492732e-10} and reward: 0.3856
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\x9d\xdb\x18\xb0\xc5bX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6X\x7f\xff\xcb\x18\x8bX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x849\xc6e\xf1\x17gX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xff\xfa\xe1\x04\xdd\x1b\xb3u.' and reward: 0.3856
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\x9d\xdb\x18\xb0\xc5bX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6X\x7f\xff\xcb\x18\x8bX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x849\xc6e\xf1\x17gX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xff\xfa\xe1\x04\xdd\x1b\xb3u.' and reward: 0.3856
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 195.35205149650574
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.78s of the -77.9s of remaining time.
Ensemble size: 17
Ensemble weights: 
[0.23529412 0.41176471 0.35294118]
	0.3906	 = Validation accuracy score
	0.99s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 198.93s ...
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
Task exception was never retrieved
future: <Task finished coro=<InProcConnector.connect() done, defined at /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/distributed/comm/inproc.py:285> exception=OSError("no endpoint for inproc address '10.1.0.4/4236/1'",)>
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/distributed/comm/inproc.py", line 288, in connect
    raise IOError("no endpoint for inproc address %r" % (address,))
OSError: no endpoint for inproc address '10.1.0.4/4236/1'





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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f0b83a04a58> 

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
[[ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00]
 [ 3.38518173e-02  1.60558186e-02  4.87436429e-02  4.36629727e-02
  -2.16455907e-02  1.30454544e-04]
 [ 1.92168429e-01 -8.66670832e-02 -6.46990538e-02  2.07293313e-03
   6.76339790e-02  2.62358021e-02]
 [-8.25465247e-02 -3.57868038e-02 -9.73965451e-02  2.98150890e-02
  -1.41274527e-01 -6.31407695e-03]
 [ 2.49993607e-01  1.88802168e-01  5.20375550e-01  1.81266099e-01
   1.06260397e-01  1.31722003e-01]
 [ 2.73971021e-01  2.21005410e-01  2.62112826e-01  5.23619115e-01
   4.87298608e-01  3.78155708e-01]
 [ 5.10931134e-01 -2.99420983e-01 -4.05986786e-01 -1.74110979e-02
  -9.63101014e-02  6.00721002e-01]
 [ 1.81769863e-01 -1.67010099e-01  1.72473937e-01 -3.06292564e-01
  -1.23228185e-01  5.63783884e-01]
 [ 1.68852672e-01  4.01644796e-01  8.44261795e-02  3.52376908e-01
  -5.37219271e-02  6.94995224e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
   0.00000000e+00  0.00000000e+00]]

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
{'loss': 0.47713392972946167, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-12 20:17:48.430423: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.49115361645817757, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-12 20:17:49.533660: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
   24576/17464789 [..............................] - ETA: 49s
   57344/17464789 [..............................] - ETA: 41s
   90112/17464789 [..............................] - ETA: 39s
  180224/17464789 [..............................] - ETA: 26s
  335872/17464789 [..............................] - ETA: 17s
  647168/17464789 [>.............................] - ETA: 10s
 1294336/17464789 [=>............................] - ETA: 6s 
 2564096/17464789 [===>..........................] - ETA: 3s
 4825088/17464789 [=======>......................] - ETA: 1s
 7823360/17464789 [============>.................] - ETA: 0s
10854400/17464789 [=================>............] - ETA: 0s
13606912/17464789 [======================>.......] - ETA: 0s
16621568/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-12 20:18:01.539780: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 20:18:01.544175: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-12 20:18:01.544330: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558894058190 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-12 20:18:01.544345: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 3:55 - loss: 8.1458 - accuracy: 0.4688
   64/25000 [..............................] - ETA: 2:34 - loss: 8.3854 - accuracy: 0.4531
   96/25000 [..............................] - ETA: 2:07 - loss: 8.1458 - accuracy: 0.4688
  128/25000 [..............................] - ETA: 1:53 - loss: 7.7864 - accuracy: 0.4922
  160/25000 [..............................] - ETA: 1:45 - loss: 7.6666 - accuracy: 0.5000
  192/25000 [..............................] - ETA: 1:39 - loss: 7.7465 - accuracy: 0.4948
  224/25000 [..............................] - ETA: 1:35 - loss: 7.7351 - accuracy: 0.4955
  256/25000 [..............................] - ETA: 1:32 - loss: 7.7265 - accuracy: 0.4961
  288/25000 [..............................] - ETA: 1:30 - loss: 7.7731 - accuracy: 0.4931
  320/25000 [..............................] - ETA: 1:27 - loss: 7.6666 - accuracy: 0.5000
  352/25000 [..............................] - ETA: 1:26 - loss: 7.5795 - accuracy: 0.5057
  384/25000 [..............................] - ETA: 1:24 - loss: 7.3472 - accuracy: 0.5208
  416/25000 [..............................] - ETA: 1:24 - loss: 7.3717 - accuracy: 0.5192
  448/25000 [..............................] - ETA: 1:23 - loss: 7.3928 - accuracy: 0.5179
  480/25000 [..............................] - ETA: 1:22 - loss: 7.4430 - accuracy: 0.5146
  512/25000 [..............................] - ETA: 1:21 - loss: 7.4570 - accuracy: 0.5137
  544/25000 [..............................] - ETA: 1:21 - loss: 7.3284 - accuracy: 0.5221
  576/25000 [..............................] - ETA: 1:20 - loss: 7.4004 - accuracy: 0.5174
  608/25000 [..............................] - ETA: 1:19 - loss: 7.5657 - accuracy: 0.5066
  640/25000 [..............................] - ETA: 1:19 - loss: 7.5947 - accuracy: 0.5047
  672/25000 [..............................] - ETA: 1:18 - loss: 7.5525 - accuracy: 0.5074
  704/25000 [..............................] - ETA: 1:18 - loss: 7.6666 - accuracy: 0.5000
  736/25000 [..............................] - ETA: 1:17 - loss: 7.7291 - accuracy: 0.4959
  768/25000 [..............................] - ETA: 1:17 - loss: 7.6467 - accuracy: 0.5013
  800/25000 [..............................] - ETA: 1:17 - loss: 7.7050 - accuracy: 0.4975
  832/25000 [..............................] - ETA: 1:16 - loss: 7.6850 - accuracy: 0.4988
  864/25000 [>.............................] - ETA: 1:16 - loss: 7.7199 - accuracy: 0.4965
  896/25000 [>.............................] - ETA: 1:16 - loss: 7.7864 - accuracy: 0.4922
  928/25000 [>.............................] - ETA: 1:15 - loss: 7.7823 - accuracy: 0.4925
  960/25000 [>.............................] - ETA: 1:15 - loss: 7.6826 - accuracy: 0.4990
  992/25000 [>.............................] - ETA: 1:15 - loss: 7.6821 - accuracy: 0.4990
 1024/25000 [>.............................] - ETA: 1:15 - loss: 7.6966 - accuracy: 0.4980
 1056/25000 [>.............................] - ETA: 1:14 - loss: 7.6376 - accuracy: 0.5019
 1088/25000 [>.............................] - ETA: 1:14 - loss: 7.6384 - accuracy: 0.5018
 1120/25000 [>.............................] - ETA: 1:14 - loss: 7.6119 - accuracy: 0.5036
 1152/25000 [>.............................] - ETA: 1:14 - loss: 7.6400 - accuracy: 0.5017
 1184/25000 [>.............................] - ETA: 1:14 - loss: 7.6407 - accuracy: 0.5017
 1216/25000 [>.............................] - ETA: 1:13 - loss: 7.6540 - accuracy: 0.5008
 1248/25000 [>.............................] - ETA: 1:13 - loss: 7.6175 - accuracy: 0.5032
 1280/25000 [>.............................] - ETA: 1:13 - loss: 7.5947 - accuracy: 0.5047
 1312/25000 [>.............................] - ETA: 1:13 - loss: 7.5731 - accuracy: 0.5061
 1344/25000 [>.............................] - ETA: 1:13 - loss: 7.5639 - accuracy: 0.5067
 1376/25000 [>.............................] - ETA: 1:12 - loss: 7.5775 - accuracy: 0.5058
 1408/25000 [>.............................] - ETA: 1:12 - loss: 7.6122 - accuracy: 0.5036
 1440/25000 [>.............................] - ETA: 1:12 - loss: 7.6240 - accuracy: 0.5028
 1472/25000 [>.............................] - ETA: 1:12 - loss: 7.5833 - accuracy: 0.5054
 1504/25000 [>.............................] - ETA: 1:12 - loss: 7.5647 - accuracy: 0.5066
 1536/25000 [>.............................] - ETA: 1:12 - loss: 7.5568 - accuracy: 0.5072
 1568/25000 [>.............................] - ETA: 1:12 - loss: 7.5493 - accuracy: 0.5077
 1600/25000 [>.............................] - ETA: 1:11 - loss: 7.5708 - accuracy: 0.5063
 1632/25000 [>.............................] - ETA: 1:11 - loss: 7.5727 - accuracy: 0.5061
 1664/25000 [>.............................] - ETA: 1:11 - loss: 7.5745 - accuracy: 0.5060
 1696/25000 [=>............................] - ETA: 1:11 - loss: 7.5400 - accuracy: 0.5083
 1728/25000 [=>............................] - ETA: 1:11 - loss: 7.5424 - accuracy: 0.5081
 1760/25000 [=>............................] - ETA: 1:10 - loss: 7.5621 - accuracy: 0.5068
 1792/25000 [=>............................] - ETA: 1:10 - loss: 7.5554 - accuracy: 0.5073
 1824/25000 [=>............................] - ETA: 1:10 - loss: 7.5489 - accuracy: 0.5077
 1856/25000 [=>............................] - ETA: 1:10 - loss: 7.5344 - accuracy: 0.5086
 1888/25000 [=>............................] - ETA: 1:10 - loss: 7.5286 - accuracy: 0.5090
 1920/25000 [=>............................] - ETA: 1:10 - loss: 7.5309 - accuracy: 0.5089
 1952/25000 [=>............................] - ETA: 1:10 - loss: 7.5488 - accuracy: 0.5077
 1984/25000 [=>............................] - ETA: 1:09 - loss: 7.5739 - accuracy: 0.5060
 2016/25000 [=>............................] - ETA: 1:09 - loss: 7.5906 - accuracy: 0.5050
 2048/25000 [=>............................] - ETA: 1:09 - loss: 7.5768 - accuracy: 0.5059
 2080/25000 [=>............................] - ETA: 1:09 - loss: 7.5855 - accuracy: 0.5053
 2112/25000 [=>............................] - ETA: 1:09 - loss: 7.5868 - accuracy: 0.5052
 2144/25000 [=>............................] - ETA: 1:09 - loss: 7.6023 - accuracy: 0.5042
 2176/25000 [=>............................] - ETA: 1:09 - loss: 7.6173 - accuracy: 0.5032
 2208/25000 [=>............................] - ETA: 1:09 - loss: 7.6250 - accuracy: 0.5027
 2240/25000 [=>............................] - ETA: 1:08 - loss: 7.6461 - accuracy: 0.5013
 2272/25000 [=>............................] - ETA: 1:08 - loss: 7.6396 - accuracy: 0.5018
 2304/25000 [=>............................] - ETA: 1:08 - loss: 7.6467 - accuracy: 0.5013
 2336/25000 [=>............................] - ETA: 1:08 - loss: 7.6469 - accuracy: 0.5013
 2368/25000 [=>............................] - ETA: 1:08 - loss: 7.6666 - accuracy: 0.5000
 2400/25000 [=>............................] - ETA: 1:08 - loss: 7.6602 - accuracy: 0.5004
 2432/25000 [=>............................] - ETA: 1:08 - loss: 7.6414 - accuracy: 0.5016
 2464/25000 [=>............................] - ETA: 1:08 - loss: 7.6231 - accuracy: 0.5028
 2496/25000 [=>............................] - ETA: 1:07 - loss: 7.6113 - accuracy: 0.5036
 2528/25000 [==>...........................] - ETA: 1:07 - loss: 7.5938 - accuracy: 0.5047
 2560/25000 [==>...........................] - ETA: 1:07 - loss: 7.5888 - accuracy: 0.5051
 2592/25000 [==>...........................] - ETA: 1:07 - loss: 7.6134 - accuracy: 0.5035
 2624/25000 [==>...........................] - ETA: 1:07 - loss: 7.5907 - accuracy: 0.5050
 2656/25000 [==>...........................] - ETA: 1:07 - loss: 7.5569 - accuracy: 0.5072
 2688/25000 [==>...........................] - ETA: 1:07 - loss: 7.5411 - accuracy: 0.5082
 2720/25000 [==>...........................] - ETA: 1:07 - loss: 7.5313 - accuracy: 0.5088
 2752/25000 [==>...........................] - ETA: 1:07 - loss: 7.5663 - accuracy: 0.5065
 2784/25000 [==>...........................] - ETA: 1:07 - loss: 7.5785 - accuracy: 0.5057
 2816/25000 [==>...........................] - ETA: 1:07 - loss: 7.5577 - accuracy: 0.5071
 2848/25000 [==>...........................] - ETA: 1:07 - loss: 7.5374 - accuracy: 0.5084
 2880/25000 [==>...........................] - ETA: 1:07 - loss: 7.5282 - accuracy: 0.5090
 2912/25000 [==>...........................] - ETA: 1:07 - loss: 7.5245 - accuracy: 0.5093
 2944/25000 [==>...........................] - ETA: 1:06 - loss: 7.5156 - accuracy: 0.5099
 2976/25000 [==>...........................] - ETA: 1:06 - loss: 7.5378 - accuracy: 0.5084
 3008/25000 [==>...........................] - ETA: 1:06 - loss: 7.5290 - accuracy: 0.5090
 3040/25000 [==>...........................] - ETA: 1:06 - loss: 7.5203 - accuracy: 0.5095
 3072/25000 [==>...........................] - ETA: 1:06 - loss: 7.5169 - accuracy: 0.5098
 3104/25000 [==>...........................] - ETA: 1:06 - loss: 7.5382 - accuracy: 0.5084
 3136/25000 [==>...........................] - ETA: 1:06 - loss: 7.5591 - accuracy: 0.5070
 3168/25000 [==>...........................] - ETA: 1:06 - loss: 7.5311 - accuracy: 0.5088
 3200/25000 [==>...........................] - ETA: 1:06 - loss: 7.5229 - accuracy: 0.5094
 3232/25000 [==>...........................] - ETA: 1:06 - loss: 7.5290 - accuracy: 0.5090
 3264/25000 [==>...........................] - ETA: 1:06 - loss: 7.5304 - accuracy: 0.5089
 3296/25000 [==>...........................] - ETA: 1:06 - loss: 7.5410 - accuracy: 0.5082
 3328/25000 [==>...........................] - ETA: 1:05 - loss: 7.5422 - accuracy: 0.5081
 3360/25000 [===>..........................] - ETA: 1:05 - loss: 7.5571 - accuracy: 0.5071
 3392/25000 [===>..........................] - ETA: 1:05 - loss: 7.5717 - accuracy: 0.5062
 3424/25000 [===>..........................] - ETA: 1:05 - loss: 7.5636 - accuracy: 0.5067
 3456/25000 [===>..........................] - ETA: 1:05 - loss: 7.5779 - accuracy: 0.5058
 3488/25000 [===>..........................] - ETA: 1:05 - loss: 7.5919 - accuracy: 0.5049
 3520/25000 [===>..........................] - ETA: 1:05 - loss: 7.6056 - accuracy: 0.5040
 3552/25000 [===>..........................] - ETA: 1:05 - loss: 7.5976 - accuracy: 0.5045
 3584/25000 [===>..........................] - ETA: 1:05 - loss: 7.5853 - accuracy: 0.5053
 3616/25000 [===>..........................] - ETA: 1:05 - loss: 7.5903 - accuracy: 0.5050
 3648/25000 [===>..........................] - ETA: 1:05 - loss: 7.5868 - accuracy: 0.5052
 3680/25000 [===>..........................] - ETA: 1:04 - loss: 7.5833 - accuracy: 0.5054
 3712/25000 [===>..........................] - ETA: 1:04 - loss: 7.6047 - accuracy: 0.5040
 3744/25000 [===>..........................] - ETA: 1:04 - loss: 7.5970 - accuracy: 0.5045
 3776/25000 [===>..........................] - ETA: 1:04 - loss: 7.6057 - accuracy: 0.5040
 3808/25000 [===>..........................] - ETA: 1:04 - loss: 7.6143 - accuracy: 0.5034
 3840/25000 [===>..........................] - ETA: 1:04 - loss: 7.6187 - accuracy: 0.5031
 3872/25000 [===>..........................] - ETA: 1:04 - loss: 7.6310 - accuracy: 0.5023
 3904/25000 [===>..........................] - ETA: 1:03 - loss: 7.6195 - accuracy: 0.5031
 3936/25000 [===>..........................] - ETA: 1:03 - loss: 7.6160 - accuracy: 0.5033
 3968/25000 [===>..........................] - ETA: 1:03 - loss: 7.6280 - accuracy: 0.5025
 4000/25000 [===>..........................] - ETA: 1:03 - loss: 7.6245 - accuracy: 0.5027
 4032/25000 [===>..........................] - ETA: 1:03 - loss: 7.6324 - accuracy: 0.5022
 4064/25000 [===>..........................] - ETA: 1:03 - loss: 7.6327 - accuracy: 0.5022
 4096/25000 [===>..........................] - ETA: 1:03 - loss: 7.6217 - accuracy: 0.5029
 4128/25000 [===>..........................] - ETA: 1:03 - loss: 7.6109 - accuracy: 0.5036
 4160/25000 [===>..........................] - ETA: 1:02 - loss: 7.6150 - accuracy: 0.5034
 4192/25000 [====>.........................] - ETA: 1:02 - loss: 7.6191 - accuracy: 0.5031
 4224/25000 [====>.........................] - ETA: 1:02 - loss: 7.6303 - accuracy: 0.5024
 4256/25000 [====>.........................] - ETA: 1:02 - loss: 7.6342 - accuracy: 0.5021
 4288/25000 [====>.........................] - ETA: 1:02 - loss: 7.6309 - accuracy: 0.5023
 4320/25000 [====>.........................] - ETA: 1:02 - loss: 7.6169 - accuracy: 0.5032
 4352/25000 [====>.........................] - ETA: 1:02 - loss: 7.5962 - accuracy: 0.5046
 4384/25000 [====>.........................] - ETA: 1:02 - loss: 7.5897 - accuracy: 0.5050
 4416/25000 [====>.........................] - ETA: 1:02 - loss: 7.5937 - accuracy: 0.5048
 4448/25000 [====>.........................] - ETA: 1:02 - loss: 7.5942 - accuracy: 0.5047
 4480/25000 [====>.........................] - ETA: 1:01 - loss: 7.5947 - accuracy: 0.5047
 4512/25000 [====>.........................] - ETA: 1:01 - loss: 7.5851 - accuracy: 0.5053
 4544/25000 [====>.........................] - ETA: 1:01 - loss: 7.5823 - accuracy: 0.5055
 4576/25000 [====>.........................] - ETA: 1:01 - loss: 7.5761 - accuracy: 0.5059
 4608/25000 [====>.........................] - ETA: 1:01 - loss: 7.5834 - accuracy: 0.5054
 4640/25000 [====>.........................] - ETA: 1:01 - loss: 7.5906 - accuracy: 0.5050
 4672/25000 [====>.........................] - ETA: 1:01 - loss: 7.5780 - accuracy: 0.5058
 4704/25000 [====>.........................] - ETA: 1:01 - loss: 7.5851 - accuracy: 0.5053
 4736/25000 [====>.........................] - ETA: 1:01 - loss: 7.5954 - accuracy: 0.5046
 4768/25000 [====>.........................] - ETA: 1:01 - loss: 7.5991 - accuracy: 0.5044
 4800/25000 [====>.........................] - ETA: 1:00 - loss: 7.5900 - accuracy: 0.5050
 4832/25000 [====>.........................] - ETA: 1:00 - loss: 7.5841 - accuracy: 0.5054
 4864/25000 [====>.........................] - ETA: 1:00 - loss: 7.6004 - accuracy: 0.5043
 4896/25000 [====>.........................] - ETA: 1:00 - loss: 7.6040 - accuracy: 0.5041
 4928/25000 [====>.........................] - ETA: 1:00 - loss: 7.5982 - accuracy: 0.5045
 4960/25000 [====>.........................] - ETA: 1:00 - loss: 7.5893 - accuracy: 0.5050
 4992/25000 [====>.........................] - ETA: 1:00 - loss: 7.5775 - accuracy: 0.5058
 5024/25000 [=====>........................] - ETA: 1:00 - loss: 7.5842 - accuracy: 0.5054
 5056/25000 [=====>........................] - ETA: 1:00 - loss: 7.6029 - accuracy: 0.5042
 5088/25000 [=====>........................] - ETA: 1:00 - loss: 7.6033 - accuracy: 0.5041
 5120/25000 [=====>........................] - ETA: 59s - loss: 7.5918 - accuracy: 0.5049 
 5152/25000 [=====>........................] - ETA: 59s - loss: 7.5952 - accuracy: 0.5047
 5184/25000 [=====>........................] - ETA: 59s - loss: 7.6134 - accuracy: 0.5035
 5216/25000 [=====>........................] - ETA: 59s - loss: 7.6225 - accuracy: 0.5029
 5248/25000 [=====>........................] - ETA: 59s - loss: 7.6199 - accuracy: 0.5030
 5280/25000 [=====>........................] - ETA: 59s - loss: 7.6260 - accuracy: 0.5027
 5312/25000 [=====>........................] - ETA: 59s - loss: 7.6175 - accuracy: 0.5032
 5344/25000 [=====>........................] - ETA: 59s - loss: 7.6178 - accuracy: 0.5032
 5376/25000 [=====>........................] - ETA: 59s - loss: 7.6039 - accuracy: 0.5041
 5408/25000 [=====>........................] - ETA: 59s - loss: 7.5929 - accuracy: 0.5048
 5440/25000 [=====>........................] - ETA: 58s - loss: 7.6074 - accuracy: 0.5039
 5472/25000 [=====>........................] - ETA: 58s - loss: 7.6106 - accuracy: 0.5037
 5504/25000 [=====>........................] - ETA: 58s - loss: 7.6248 - accuracy: 0.5027
 5536/25000 [=====>........................] - ETA: 58s - loss: 7.6334 - accuracy: 0.5022
 5568/25000 [=====>........................] - ETA: 58s - loss: 7.6363 - accuracy: 0.5020
 5600/25000 [=====>........................] - ETA: 58s - loss: 7.6502 - accuracy: 0.5011
 5632/25000 [=====>........................] - ETA: 58s - loss: 7.6476 - accuracy: 0.5012
 5664/25000 [=====>........................] - ETA: 58s - loss: 7.6531 - accuracy: 0.5009
 5696/25000 [=====>........................] - ETA: 58s - loss: 7.6505 - accuracy: 0.5011
 5728/25000 [=====>........................] - ETA: 57s - loss: 7.6452 - accuracy: 0.5014
 5760/25000 [=====>........................] - ETA: 57s - loss: 7.6267 - accuracy: 0.5026
 5792/25000 [=====>........................] - ETA: 57s - loss: 7.6216 - accuracy: 0.5029
 5824/25000 [=====>........................] - ETA: 57s - loss: 7.6192 - accuracy: 0.5031
 5856/25000 [======>.......................] - ETA: 57s - loss: 7.6169 - accuracy: 0.5032
 5888/25000 [======>.......................] - ETA: 57s - loss: 7.6145 - accuracy: 0.5034
 5920/25000 [======>.......................] - ETA: 57s - loss: 7.6019 - accuracy: 0.5042
 5952/25000 [======>.......................] - ETA: 57s - loss: 7.5971 - accuracy: 0.5045
 5984/25000 [======>.......................] - ETA: 57s - loss: 7.6051 - accuracy: 0.5040
 6016/25000 [======>.......................] - ETA: 57s - loss: 7.6131 - accuracy: 0.5035
 6048/25000 [======>.......................] - ETA: 56s - loss: 7.6108 - accuracy: 0.5036
 6080/25000 [======>.......................] - ETA: 56s - loss: 7.6010 - accuracy: 0.5043
 6112/25000 [======>.......................] - ETA: 56s - loss: 7.6014 - accuracy: 0.5043
 6144/25000 [======>.......................] - ETA: 56s - loss: 7.6017 - accuracy: 0.5042
 6176/25000 [======>.......................] - ETA: 56s - loss: 7.6120 - accuracy: 0.5036
 6208/25000 [======>.......................] - ETA: 56s - loss: 7.6098 - accuracy: 0.5037
 6240/25000 [======>.......................] - ETA: 56s - loss: 7.6175 - accuracy: 0.5032
 6272/25000 [======>.......................] - ETA: 56s - loss: 7.6202 - accuracy: 0.5030
 6304/25000 [======>.......................] - ETA: 56s - loss: 7.6107 - accuracy: 0.5036
 6336/25000 [======>.......................] - ETA: 56s - loss: 7.6158 - accuracy: 0.5033
 6368/25000 [======>.......................] - ETA: 55s - loss: 7.6136 - accuracy: 0.5035
 6400/25000 [======>.......................] - ETA: 55s - loss: 7.6067 - accuracy: 0.5039
 6432/25000 [======>.......................] - ETA: 55s - loss: 7.6142 - accuracy: 0.5034
 6464/25000 [======>.......................] - ETA: 55s - loss: 7.6144 - accuracy: 0.5034
 6496/25000 [======>.......................] - ETA: 55s - loss: 7.6194 - accuracy: 0.5031
 6528/25000 [======>.......................] - ETA: 55s - loss: 7.6196 - accuracy: 0.5031
 6560/25000 [======>.......................] - ETA: 55s - loss: 7.6316 - accuracy: 0.5023
 6592/25000 [======>.......................] - ETA: 55s - loss: 7.6271 - accuracy: 0.5026
 6624/25000 [======>.......................] - ETA: 55s - loss: 7.6203 - accuracy: 0.5030
 6656/25000 [======>.......................] - ETA: 55s - loss: 7.6228 - accuracy: 0.5029
 6688/25000 [=======>......................] - ETA: 55s - loss: 7.6139 - accuracy: 0.5034
 6720/25000 [=======>......................] - ETA: 54s - loss: 7.6141 - accuracy: 0.5034
 6752/25000 [=======>......................] - ETA: 54s - loss: 7.6121 - accuracy: 0.5036
 6784/25000 [=======>......................] - ETA: 54s - loss: 7.6079 - accuracy: 0.5038
 6816/25000 [=======>......................] - ETA: 54s - loss: 7.6171 - accuracy: 0.5032
 6848/25000 [=======>......................] - ETA: 54s - loss: 7.6106 - accuracy: 0.5037
 6880/25000 [=======>......................] - ETA: 54s - loss: 7.6176 - accuracy: 0.5032
 6912/25000 [=======>......................] - ETA: 54s - loss: 7.6134 - accuracy: 0.5035
 6944/25000 [=======>......................] - ETA: 54s - loss: 7.6180 - accuracy: 0.5032
 6976/25000 [=======>......................] - ETA: 54s - loss: 7.6161 - accuracy: 0.5033
 7008/25000 [=======>......................] - ETA: 54s - loss: 7.6163 - accuracy: 0.5033
 7040/25000 [=======>......................] - ETA: 53s - loss: 7.6143 - accuracy: 0.5034
 7072/25000 [=======>......................] - ETA: 53s - loss: 7.6102 - accuracy: 0.5037
 7104/25000 [=======>......................] - ETA: 53s - loss: 7.6062 - accuracy: 0.5039
 7136/25000 [=======>......................] - ETA: 53s - loss: 7.5936 - accuracy: 0.5048
 7168/25000 [=======>......................] - ETA: 53s - loss: 7.6003 - accuracy: 0.5043
 7200/25000 [=======>......................] - ETA: 53s - loss: 7.5985 - accuracy: 0.5044
 7232/25000 [=======>......................] - ETA: 53s - loss: 7.5967 - accuracy: 0.5046
 7264/25000 [=======>......................] - ETA: 53s - loss: 7.5949 - accuracy: 0.5047
 7296/25000 [=======>......................] - ETA: 53s - loss: 7.5931 - accuracy: 0.5048
 7328/25000 [=======>......................] - ETA: 53s - loss: 7.5955 - accuracy: 0.5046
 7360/25000 [=======>......................] - ETA: 53s - loss: 7.5958 - accuracy: 0.5046
 7392/25000 [=======>......................] - ETA: 52s - loss: 7.6023 - accuracy: 0.5042
 7424/25000 [=======>......................] - ETA: 52s - loss: 7.6150 - accuracy: 0.5034
 7456/25000 [=======>......................] - ETA: 52s - loss: 7.6193 - accuracy: 0.5031
 7488/25000 [=======>......................] - ETA: 52s - loss: 7.6216 - accuracy: 0.5029
 7520/25000 [========>.....................] - ETA: 52s - loss: 7.6136 - accuracy: 0.5035
 7552/25000 [========>.....................] - ETA: 52s - loss: 7.6138 - accuracy: 0.5034
 7584/25000 [========>.....................] - ETA: 52s - loss: 7.6181 - accuracy: 0.5032
 7616/25000 [========>.....................] - ETA: 52s - loss: 7.6123 - accuracy: 0.5035
 7648/25000 [========>.....................] - ETA: 52s - loss: 7.6165 - accuracy: 0.5033
 7680/25000 [========>.....................] - ETA: 52s - loss: 7.6227 - accuracy: 0.5029
 7712/25000 [========>.....................] - ETA: 51s - loss: 7.6269 - accuracy: 0.5026
 7744/25000 [========>.....................] - ETA: 51s - loss: 7.6290 - accuracy: 0.5025
 7776/25000 [========>.....................] - ETA: 51s - loss: 7.6311 - accuracy: 0.5023
 7808/25000 [========>.....................] - ETA: 51s - loss: 7.6372 - accuracy: 0.5019
 7840/25000 [========>.....................] - ETA: 51s - loss: 7.6471 - accuracy: 0.5013
 7872/25000 [========>.....................] - ETA: 51s - loss: 7.6530 - accuracy: 0.5009
 7904/25000 [========>.....................] - ETA: 51s - loss: 7.6589 - accuracy: 0.5005
 7936/25000 [========>.....................] - ETA: 51s - loss: 7.6570 - accuracy: 0.5006
 7968/25000 [========>.....................] - ETA: 51s - loss: 7.6608 - accuracy: 0.5004
 8000/25000 [========>.....................] - ETA: 51s - loss: 7.6551 - accuracy: 0.5008
 8032/25000 [========>.....................] - ETA: 51s - loss: 7.6590 - accuracy: 0.5005
 8064/25000 [========>.....................] - ETA: 50s - loss: 7.6476 - accuracy: 0.5012
 8096/25000 [========>.....................] - ETA: 50s - loss: 7.6477 - accuracy: 0.5012
 8128/25000 [========>.....................] - ETA: 50s - loss: 7.6553 - accuracy: 0.5007
 8160/25000 [========>.....................] - ETA: 50s - loss: 7.6572 - accuracy: 0.5006
 8192/25000 [========>.....................] - ETA: 50s - loss: 7.6666 - accuracy: 0.5000
 8224/25000 [========>.....................] - ETA: 50s - loss: 7.6592 - accuracy: 0.5005
 8256/25000 [========>.....................] - ETA: 50s - loss: 7.6666 - accuracy: 0.5000
 8288/25000 [========>.....................] - ETA: 50s - loss: 7.6703 - accuracy: 0.4998
 8320/25000 [========>.....................] - ETA: 50s - loss: 7.6685 - accuracy: 0.4999
 8352/25000 [=========>....................] - ETA: 50s - loss: 7.6685 - accuracy: 0.4999
 8384/25000 [=========>....................] - ETA: 49s - loss: 7.6739 - accuracy: 0.4995
 8416/25000 [=========>....................] - ETA: 49s - loss: 7.6684 - accuracy: 0.4999
 8448/25000 [=========>....................] - ETA: 49s - loss: 7.6648 - accuracy: 0.5001
 8480/25000 [=========>....................] - ETA: 49s - loss: 7.6666 - accuracy: 0.5000
 8512/25000 [=========>....................] - ETA: 49s - loss: 7.6684 - accuracy: 0.4999
 8544/25000 [=========>....................] - ETA: 49s - loss: 7.6648 - accuracy: 0.5001
 8576/25000 [=========>....................] - ETA: 49s - loss: 7.6684 - accuracy: 0.4999
 8608/25000 [=========>....................] - ETA: 49s - loss: 7.6720 - accuracy: 0.4997
 8640/25000 [=========>....................] - ETA: 49s - loss: 7.6790 - accuracy: 0.4992
 8672/25000 [=========>....................] - ETA: 49s - loss: 7.6790 - accuracy: 0.4992
 8704/25000 [=========>....................] - ETA: 48s - loss: 7.6790 - accuracy: 0.4992
 8736/25000 [=========>....................] - ETA: 48s - loss: 7.6789 - accuracy: 0.4992
 8768/25000 [=========>....................] - ETA: 48s - loss: 7.6806 - accuracy: 0.4991
 8800/25000 [=========>....................] - ETA: 48s - loss: 7.6806 - accuracy: 0.4991
 8832/25000 [=========>....................] - ETA: 48s - loss: 7.6770 - accuracy: 0.4993
 8864/25000 [=========>....................] - ETA: 48s - loss: 7.6770 - accuracy: 0.4993
 8896/25000 [=========>....................] - ETA: 48s - loss: 7.6839 - accuracy: 0.4989
 8928/25000 [=========>....................] - ETA: 48s - loss: 7.6735 - accuracy: 0.4996
 8960/25000 [=========>....................] - ETA: 48s - loss: 7.6735 - accuracy: 0.4996
 8992/25000 [=========>....................] - ETA: 48s - loss: 7.6803 - accuracy: 0.4991
 9024/25000 [=========>....................] - ETA: 47s - loss: 7.6785 - accuracy: 0.4992
 9056/25000 [=========>....................] - ETA: 47s - loss: 7.6734 - accuracy: 0.4996
 9088/25000 [=========>....................] - ETA: 47s - loss: 7.6700 - accuracy: 0.4998
 9120/25000 [=========>....................] - ETA: 47s - loss: 7.6750 - accuracy: 0.4995
 9152/25000 [=========>....................] - ETA: 47s - loss: 7.6716 - accuracy: 0.4997
 9184/25000 [==========>...................] - ETA: 47s - loss: 7.6716 - accuracy: 0.4997
 9216/25000 [==========>...................] - ETA: 47s - loss: 7.6716 - accuracy: 0.4997
 9248/25000 [==========>...................] - ETA: 47s - loss: 7.6699 - accuracy: 0.4998
 9280/25000 [==========>...................] - ETA: 47s - loss: 7.6683 - accuracy: 0.4999
 9312/25000 [==========>...................] - ETA: 47s - loss: 7.6633 - accuracy: 0.5002
 9344/25000 [==========>...................] - ETA: 47s - loss: 7.6601 - accuracy: 0.5004
 9376/25000 [==========>...................] - ETA: 46s - loss: 7.6568 - accuracy: 0.5006
 9408/25000 [==========>...................] - ETA: 46s - loss: 7.6585 - accuracy: 0.5005
 9440/25000 [==========>...................] - ETA: 46s - loss: 7.6585 - accuracy: 0.5005
 9472/25000 [==========>...................] - ETA: 46s - loss: 7.6553 - accuracy: 0.5007
 9504/25000 [==========>...................] - ETA: 46s - loss: 7.6553 - accuracy: 0.5007
 9536/25000 [==========>...................] - ETA: 46s - loss: 7.6538 - accuracy: 0.5008
 9568/25000 [==========>...................] - ETA: 46s - loss: 7.6538 - accuracy: 0.5008
 9600/25000 [==========>...................] - ETA: 46s - loss: 7.6570 - accuracy: 0.5006
 9632/25000 [==========>...................] - ETA: 46s - loss: 7.6603 - accuracy: 0.5004
 9664/25000 [==========>...................] - ETA: 46s - loss: 7.6539 - accuracy: 0.5008
 9696/25000 [==========>...................] - ETA: 45s - loss: 7.6524 - accuracy: 0.5009
 9728/25000 [==========>...................] - ETA: 45s - loss: 7.6540 - accuracy: 0.5008
 9760/25000 [==========>...................] - ETA: 45s - loss: 7.6493 - accuracy: 0.5011
 9792/25000 [==========>...................] - ETA: 45s - loss: 7.6478 - accuracy: 0.5012
 9824/25000 [==========>...................] - ETA: 45s - loss: 7.6416 - accuracy: 0.5016
 9856/25000 [==========>...................] - ETA: 45s - loss: 7.6433 - accuracy: 0.5015
 9888/25000 [==========>...................] - ETA: 45s - loss: 7.6387 - accuracy: 0.5018
 9920/25000 [==========>...................] - ETA: 45s - loss: 7.6388 - accuracy: 0.5018
 9952/25000 [==========>...................] - ETA: 45s - loss: 7.6373 - accuracy: 0.5019
 9984/25000 [==========>...................] - ETA: 45s - loss: 7.6374 - accuracy: 0.5019
10016/25000 [===========>..................] - ETA: 44s - loss: 7.6375 - accuracy: 0.5019
10048/25000 [===========>..................] - ETA: 44s - loss: 7.6407 - accuracy: 0.5017
10080/25000 [===========>..................] - ETA: 44s - loss: 7.6332 - accuracy: 0.5022
10112/25000 [===========>..................] - ETA: 44s - loss: 7.6333 - accuracy: 0.5022
10144/25000 [===========>..................] - ETA: 44s - loss: 7.6334 - accuracy: 0.5022
10176/25000 [===========>..................] - ETA: 44s - loss: 7.6289 - accuracy: 0.5025
10208/25000 [===========>..................] - ETA: 44s - loss: 7.6291 - accuracy: 0.5024
10240/25000 [===========>..................] - ETA: 44s - loss: 7.6337 - accuracy: 0.5021
10272/25000 [===========>..................] - ETA: 44s - loss: 7.6338 - accuracy: 0.5021
10304/25000 [===========>..................] - ETA: 44s - loss: 7.6279 - accuracy: 0.5025
10336/25000 [===========>..................] - ETA: 44s - loss: 7.6266 - accuracy: 0.5026
10368/25000 [===========>..................] - ETA: 43s - loss: 7.6370 - accuracy: 0.5019
10400/25000 [===========>..................] - ETA: 43s - loss: 7.6312 - accuracy: 0.5023
10432/25000 [===========>..................] - ETA: 43s - loss: 7.6343 - accuracy: 0.5021
10464/25000 [===========>..................] - ETA: 43s - loss: 7.6329 - accuracy: 0.5022
10496/25000 [===========>..................] - ETA: 43s - loss: 7.6359 - accuracy: 0.5020
10528/25000 [===========>..................] - ETA: 43s - loss: 7.6360 - accuracy: 0.5020
10560/25000 [===========>..................] - ETA: 43s - loss: 7.6347 - accuracy: 0.5021
10592/25000 [===========>..................] - ETA: 43s - loss: 7.6362 - accuracy: 0.5020
10624/25000 [===========>..................] - ETA: 43s - loss: 7.6305 - accuracy: 0.5024
10656/25000 [===========>..................] - ETA: 42s - loss: 7.6220 - accuracy: 0.5029
10688/25000 [===========>..................] - ETA: 42s - loss: 7.6265 - accuracy: 0.5026
10720/25000 [===========>..................] - ETA: 42s - loss: 7.6251 - accuracy: 0.5027
10752/25000 [===========>..................] - ETA: 42s - loss: 7.6267 - accuracy: 0.5026
10784/25000 [===========>..................] - ETA: 42s - loss: 7.6339 - accuracy: 0.5021
10816/25000 [===========>..................] - ETA: 42s - loss: 7.6368 - accuracy: 0.5019
10848/25000 [============>.................] - ETA: 42s - loss: 7.6313 - accuracy: 0.5023
10880/25000 [============>.................] - ETA: 42s - loss: 7.6370 - accuracy: 0.5019
10912/25000 [============>.................] - ETA: 42s - loss: 7.6343 - accuracy: 0.5021
10944/25000 [============>.................] - ETA: 42s - loss: 7.6316 - accuracy: 0.5023
10976/25000 [============>.................] - ETA: 42s - loss: 7.6303 - accuracy: 0.5024
11008/25000 [============>.................] - ETA: 41s - loss: 7.6304 - accuracy: 0.5024
11040/25000 [============>.................] - ETA: 41s - loss: 7.6319 - accuracy: 0.5023
11072/25000 [============>.................] - ETA: 41s - loss: 7.6348 - accuracy: 0.5021
11104/25000 [============>.................] - ETA: 41s - loss: 7.6376 - accuracy: 0.5019
11136/25000 [============>.................] - ETA: 41s - loss: 7.6377 - accuracy: 0.5019
11168/25000 [============>.................] - ETA: 41s - loss: 7.6405 - accuracy: 0.5017
11200/25000 [============>.................] - ETA: 41s - loss: 7.6420 - accuracy: 0.5016
11232/25000 [============>.................] - ETA: 41s - loss: 7.6420 - accuracy: 0.5016
11264/25000 [============>.................] - ETA: 41s - loss: 7.6421 - accuracy: 0.5016
11296/25000 [============>.................] - ETA: 41s - loss: 7.6476 - accuracy: 0.5012
11328/25000 [============>.................] - ETA: 40s - loss: 7.6450 - accuracy: 0.5014
11360/25000 [============>.................] - ETA: 40s - loss: 7.6450 - accuracy: 0.5014
11392/25000 [============>.................] - ETA: 40s - loss: 7.6491 - accuracy: 0.5011
11424/25000 [============>.................] - ETA: 40s - loss: 7.6505 - accuracy: 0.5011
11456/25000 [============>.................] - ETA: 40s - loss: 7.6573 - accuracy: 0.5006
11488/25000 [============>.................] - ETA: 40s - loss: 7.6586 - accuracy: 0.5005
11520/25000 [============>.................] - ETA: 40s - loss: 7.6560 - accuracy: 0.5007
11552/25000 [============>.................] - ETA: 40s - loss: 7.6547 - accuracy: 0.5008
11584/25000 [============>.................] - ETA: 40s - loss: 7.6574 - accuracy: 0.5006
11616/25000 [============>.................] - ETA: 40s - loss: 7.6561 - accuracy: 0.5007
11648/25000 [============>.................] - ETA: 39s - loss: 7.6561 - accuracy: 0.5007
11680/25000 [=============>................] - ETA: 39s - loss: 7.6574 - accuracy: 0.5006
11712/25000 [=============>................] - ETA: 39s - loss: 7.6535 - accuracy: 0.5009
11744/25000 [=============>................] - ETA: 39s - loss: 7.6496 - accuracy: 0.5011
11776/25000 [=============>................] - ETA: 39s - loss: 7.6471 - accuracy: 0.5013
11808/25000 [=============>................] - ETA: 39s - loss: 7.6497 - accuracy: 0.5011
11840/25000 [=============>................] - ETA: 39s - loss: 7.6407 - accuracy: 0.5017
11872/25000 [=============>................] - ETA: 39s - loss: 7.6421 - accuracy: 0.5016
11904/25000 [=============>................] - ETA: 39s - loss: 7.6460 - accuracy: 0.5013
11936/25000 [=============>................] - ETA: 39s - loss: 7.6525 - accuracy: 0.5009
11968/25000 [=============>................] - ETA: 39s - loss: 7.6538 - accuracy: 0.5008
12000/25000 [=============>................] - ETA: 38s - loss: 7.6538 - accuracy: 0.5008
12032/25000 [=============>................] - ETA: 38s - loss: 7.6564 - accuracy: 0.5007
12064/25000 [=============>................] - ETA: 38s - loss: 7.6603 - accuracy: 0.5004
12096/25000 [=============>................] - ETA: 38s - loss: 7.6590 - accuracy: 0.5005
12128/25000 [=============>................] - ETA: 38s - loss: 7.6578 - accuracy: 0.5006
12160/25000 [=============>................] - ETA: 38s - loss: 7.6591 - accuracy: 0.5005
12192/25000 [=============>................] - ETA: 38s - loss: 7.6628 - accuracy: 0.5002
12224/25000 [=============>................] - ETA: 38s - loss: 7.6679 - accuracy: 0.4999
12256/25000 [=============>................] - ETA: 38s - loss: 7.6691 - accuracy: 0.4998
12288/25000 [=============>................] - ETA: 38s - loss: 7.6716 - accuracy: 0.4997
12320/25000 [=============>................] - ETA: 37s - loss: 7.6728 - accuracy: 0.4996
12352/25000 [=============>................] - ETA: 37s - loss: 7.6753 - accuracy: 0.4994
12384/25000 [=============>................] - ETA: 37s - loss: 7.6778 - accuracy: 0.4993
12416/25000 [=============>................] - ETA: 37s - loss: 7.6728 - accuracy: 0.4996
12448/25000 [=============>................] - ETA: 37s - loss: 7.6703 - accuracy: 0.4998
12480/25000 [=============>................] - ETA: 37s - loss: 7.6728 - accuracy: 0.4996
12512/25000 [==============>...............] - ETA: 37s - loss: 7.6801 - accuracy: 0.4991
12544/25000 [==============>...............] - ETA: 37s - loss: 7.6825 - accuracy: 0.4990
12576/25000 [==============>...............] - ETA: 37s - loss: 7.6800 - accuracy: 0.4991
12608/25000 [==============>...............] - ETA: 37s - loss: 7.6788 - accuracy: 0.4992
12640/25000 [==============>...............] - ETA: 36s - loss: 7.6775 - accuracy: 0.4993
12672/25000 [==============>...............] - ETA: 36s - loss: 7.6787 - accuracy: 0.4992
12704/25000 [==============>...............] - ETA: 36s - loss: 7.6739 - accuracy: 0.4995
12736/25000 [==============>...............] - ETA: 36s - loss: 7.6714 - accuracy: 0.4997
12768/25000 [==============>...............] - ETA: 36s - loss: 7.6678 - accuracy: 0.4999
12800/25000 [==============>...............] - ETA: 36s - loss: 7.6690 - accuracy: 0.4998
12832/25000 [==============>...............] - ETA: 36s - loss: 7.6654 - accuracy: 0.5001
12864/25000 [==============>...............] - ETA: 36s - loss: 7.6642 - accuracy: 0.5002
12896/25000 [==============>...............] - ETA: 36s - loss: 7.6678 - accuracy: 0.4999
12928/25000 [==============>...............] - ETA: 36s - loss: 7.6690 - accuracy: 0.4998
12960/25000 [==============>...............] - ETA: 36s - loss: 7.6690 - accuracy: 0.4998
12992/25000 [==============>...............] - ETA: 35s - loss: 7.6725 - accuracy: 0.4996
13024/25000 [==============>...............] - ETA: 35s - loss: 7.6725 - accuracy: 0.4996
13056/25000 [==============>...............] - ETA: 35s - loss: 7.6725 - accuracy: 0.4996
13088/25000 [==============>...............] - ETA: 35s - loss: 7.6748 - accuracy: 0.4995
13120/25000 [==============>...............] - ETA: 35s - loss: 7.6725 - accuracy: 0.4996
13152/25000 [==============>...............] - ETA: 35s - loss: 7.6690 - accuracy: 0.4998
13184/25000 [==============>...............] - ETA: 35s - loss: 7.6689 - accuracy: 0.4998
13216/25000 [==============>...............] - ETA: 35s - loss: 7.6678 - accuracy: 0.4999
13248/25000 [==============>...............] - ETA: 35s - loss: 7.6736 - accuracy: 0.4995
13280/25000 [==============>...............] - ETA: 35s - loss: 7.6724 - accuracy: 0.4996
13312/25000 [==============>...............] - ETA: 34s - loss: 7.6655 - accuracy: 0.5001
13344/25000 [===============>..............] - ETA: 34s - loss: 7.6701 - accuracy: 0.4998
13376/25000 [===============>..............] - ETA: 34s - loss: 7.6678 - accuracy: 0.4999
13408/25000 [===============>..............] - ETA: 34s - loss: 7.6689 - accuracy: 0.4999
13440/25000 [===============>..............] - ETA: 34s - loss: 7.6712 - accuracy: 0.4997
13472/25000 [===============>..............] - ETA: 34s - loss: 7.6689 - accuracy: 0.4999
13504/25000 [===============>..............] - ETA: 34s - loss: 7.6734 - accuracy: 0.4996
13536/25000 [===============>..............] - ETA: 34s - loss: 7.6757 - accuracy: 0.4994
13568/25000 [===============>..............] - ETA: 34s - loss: 7.6734 - accuracy: 0.4996
13600/25000 [===============>..............] - ETA: 34s - loss: 7.6666 - accuracy: 0.5000
13632/25000 [===============>..............] - ETA: 34s - loss: 7.6644 - accuracy: 0.5001
13664/25000 [===============>..............] - ETA: 33s - loss: 7.6644 - accuracy: 0.5001
13696/25000 [===============>..............] - ETA: 33s - loss: 7.6633 - accuracy: 0.5002
13728/25000 [===============>..............] - ETA: 33s - loss: 7.6666 - accuracy: 0.5000
13760/25000 [===============>..............] - ETA: 33s - loss: 7.6677 - accuracy: 0.4999
13792/25000 [===============>..............] - ETA: 33s - loss: 7.6700 - accuracy: 0.4998
13824/25000 [===============>..............] - ETA: 33s - loss: 7.6744 - accuracy: 0.4995
13856/25000 [===============>..............] - ETA: 33s - loss: 7.6710 - accuracy: 0.4997
13888/25000 [===============>..............] - ETA: 33s - loss: 7.6688 - accuracy: 0.4999
13920/25000 [===============>..............] - ETA: 33s - loss: 7.6688 - accuracy: 0.4999
13952/25000 [===============>..............] - ETA: 33s - loss: 7.6655 - accuracy: 0.5001
13984/25000 [===============>..............] - ETA: 32s - loss: 7.6699 - accuracy: 0.4998
14016/25000 [===============>..............] - ETA: 32s - loss: 7.6754 - accuracy: 0.4994
14048/25000 [===============>..............] - ETA: 32s - loss: 7.6721 - accuracy: 0.4996
14080/25000 [===============>..............] - ETA: 32s - loss: 7.6710 - accuracy: 0.4997
14112/25000 [===============>..............] - ETA: 32s - loss: 7.6677 - accuracy: 0.4999
14144/25000 [===============>..............] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
14176/25000 [================>.............] - ETA: 32s - loss: 7.6688 - accuracy: 0.4999
14208/25000 [================>.............] - ETA: 32s - loss: 7.6699 - accuracy: 0.4998
14240/25000 [================>.............] - ETA: 32s - loss: 7.6677 - accuracy: 0.4999
14272/25000 [================>.............] - ETA: 32s - loss: 7.6720 - accuracy: 0.4996
14304/25000 [================>.............] - ETA: 32s - loss: 7.6709 - accuracy: 0.4997
14336/25000 [================>.............] - ETA: 31s - loss: 7.6709 - accuracy: 0.4997
14368/25000 [================>.............] - ETA: 31s - loss: 7.6688 - accuracy: 0.4999
14400/25000 [================>.............] - ETA: 31s - loss: 7.6687 - accuracy: 0.4999
14432/25000 [================>.............] - ETA: 31s - loss: 7.6719 - accuracy: 0.4997
14464/25000 [================>.............] - ETA: 31s - loss: 7.6730 - accuracy: 0.4996
14496/25000 [================>.............] - ETA: 31s - loss: 7.6761 - accuracy: 0.4994
14528/25000 [================>.............] - ETA: 31s - loss: 7.6761 - accuracy: 0.4994
14560/25000 [================>.............] - ETA: 31s - loss: 7.6750 - accuracy: 0.4995
14592/25000 [================>.............] - ETA: 31s - loss: 7.6771 - accuracy: 0.4993
14624/25000 [================>.............] - ETA: 31s - loss: 7.6708 - accuracy: 0.4997
14656/25000 [================>.............] - ETA: 30s - loss: 7.6677 - accuracy: 0.4999
14688/25000 [================>.............] - ETA: 30s - loss: 7.6656 - accuracy: 0.5001
14720/25000 [================>.............] - ETA: 30s - loss: 7.6677 - accuracy: 0.4999
14752/25000 [================>.............] - ETA: 30s - loss: 7.6708 - accuracy: 0.4997
14784/25000 [================>.............] - ETA: 30s - loss: 7.6697 - accuracy: 0.4998
14816/25000 [================>.............] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
14848/25000 [================>.............] - ETA: 30s - loss: 7.6656 - accuracy: 0.5001
14880/25000 [================>.............] - ETA: 30s - loss: 7.6718 - accuracy: 0.4997
14912/25000 [================>.............] - ETA: 30s - loss: 7.6697 - accuracy: 0.4998
14944/25000 [================>.............] - ETA: 30s - loss: 7.6738 - accuracy: 0.4995
14976/25000 [================>.............] - ETA: 30s - loss: 7.6707 - accuracy: 0.4997
15008/25000 [=================>............] - ETA: 29s - loss: 7.6738 - accuracy: 0.4995
15040/25000 [=================>............] - ETA: 29s - loss: 7.6768 - accuracy: 0.4993
15072/25000 [=================>............] - ETA: 29s - loss: 7.6788 - accuracy: 0.4992
15104/25000 [=================>............] - ETA: 29s - loss: 7.6798 - accuracy: 0.4991
15136/25000 [=================>............] - ETA: 29s - loss: 7.6767 - accuracy: 0.4993
15168/25000 [=================>............] - ETA: 29s - loss: 7.6808 - accuracy: 0.4991
15200/25000 [=================>............] - ETA: 29s - loss: 7.6807 - accuracy: 0.4991
15232/25000 [=================>............] - ETA: 29s - loss: 7.6767 - accuracy: 0.4993
15264/25000 [=================>............] - ETA: 29s - loss: 7.6747 - accuracy: 0.4995
15296/25000 [=================>............] - ETA: 29s - loss: 7.6746 - accuracy: 0.4995
15328/25000 [=================>............] - ETA: 29s - loss: 7.6796 - accuracy: 0.4992
15360/25000 [=================>............] - ETA: 28s - loss: 7.6806 - accuracy: 0.4991
15392/25000 [=================>............] - ETA: 28s - loss: 7.6855 - accuracy: 0.4988
15424/25000 [=================>............] - ETA: 28s - loss: 7.6845 - accuracy: 0.4988
15456/25000 [=================>............] - ETA: 28s - loss: 7.6855 - accuracy: 0.4988
15488/25000 [=================>............] - ETA: 28s - loss: 7.6815 - accuracy: 0.4990
15520/25000 [=================>............] - ETA: 28s - loss: 7.6795 - accuracy: 0.4992
15552/25000 [=================>............] - ETA: 28s - loss: 7.6814 - accuracy: 0.4990
15584/25000 [=================>............] - ETA: 28s - loss: 7.6794 - accuracy: 0.4992
15616/25000 [=================>............] - ETA: 28s - loss: 7.6794 - accuracy: 0.4992
15648/25000 [=================>............] - ETA: 28s - loss: 7.6764 - accuracy: 0.4994
15680/25000 [=================>............] - ETA: 27s - loss: 7.6764 - accuracy: 0.4994
15712/25000 [=================>............] - ETA: 27s - loss: 7.6793 - accuracy: 0.4992
15744/25000 [=================>............] - ETA: 27s - loss: 7.6803 - accuracy: 0.4991
15776/25000 [=================>............] - ETA: 27s - loss: 7.6802 - accuracy: 0.4991
15808/25000 [=================>............] - ETA: 27s - loss: 7.6860 - accuracy: 0.4987
15840/25000 [==================>...........] - ETA: 27s - loss: 7.6840 - accuracy: 0.4989
15872/25000 [==================>...........] - ETA: 27s - loss: 7.6821 - accuracy: 0.4990
15904/25000 [==================>...........] - ETA: 27s - loss: 7.6869 - accuracy: 0.4987
15936/25000 [==================>...........] - ETA: 27s - loss: 7.6859 - accuracy: 0.4987
15968/25000 [==================>...........] - ETA: 27s - loss: 7.6820 - accuracy: 0.4990
16000/25000 [==================>...........] - ETA: 27s - loss: 7.6848 - accuracy: 0.4988
16032/25000 [==================>...........] - ETA: 26s - loss: 7.6838 - accuracy: 0.4989
16064/25000 [==================>...........] - ETA: 26s - loss: 7.6819 - accuracy: 0.4990
16096/25000 [==================>...........] - ETA: 26s - loss: 7.6819 - accuracy: 0.4990
16128/25000 [==================>...........] - ETA: 26s - loss: 7.6828 - accuracy: 0.4989
16160/25000 [==================>...........] - ETA: 26s - loss: 7.6799 - accuracy: 0.4991
16192/25000 [==================>...........] - ETA: 26s - loss: 7.6808 - accuracy: 0.4991
16224/25000 [==================>...........] - ETA: 26s - loss: 7.6761 - accuracy: 0.4994
16256/25000 [==================>...........] - ETA: 26s - loss: 7.6770 - accuracy: 0.4993
16288/25000 [==================>...........] - ETA: 26s - loss: 7.6779 - accuracy: 0.4993
16320/25000 [==================>...........] - ETA: 26s - loss: 7.6770 - accuracy: 0.4993
16352/25000 [==================>...........] - ETA: 25s - loss: 7.6769 - accuracy: 0.4993
16384/25000 [==================>...........] - ETA: 25s - loss: 7.6722 - accuracy: 0.4996
16416/25000 [==================>...........] - ETA: 25s - loss: 7.6722 - accuracy: 0.4996
16448/25000 [==================>...........] - ETA: 25s - loss: 7.6750 - accuracy: 0.4995
16480/25000 [==================>...........] - ETA: 25s - loss: 7.6741 - accuracy: 0.4995
16512/25000 [==================>...........] - ETA: 25s - loss: 7.6759 - accuracy: 0.4994
16544/25000 [==================>...........] - ETA: 25s - loss: 7.6777 - accuracy: 0.4993
16576/25000 [==================>...........] - ETA: 25s - loss: 7.6740 - accuracy: 0.4995
16608/25000 [==================>...........] - ETA: 25s - loss: 7.6749 - accuracy: 0.4995
16640/25000 [==================>...........] - ETA: 25s - loss: 7.6777 - accuracy: 0.4993
16672/25000 [===================>..........] - ETA: 24s - loss: 7.6777 - accuracy: 0.4993
16704/25000 [===================>..........] - ETA: 24s - loss: 7.6767 - accuracy: 0.4993
16736/25000 [===================>..........] - ETA: 24s - loss: 7.6813 - accuracy: 0.4990
16768/25000 [===================>..........] - ETA: 24s - loss: 7.6858 - accuracy: 0.4987
16800/25000 [===================>..........] - ETA: 24s - loss: 7.6913 - accuracy: 0.4984
16832/25000 [===================>..........] - ETA: 24s - loss: 7.6912 - accuracy: 0.4984
16864/25000 [===================>..........] - ETA: 24s - loss: 7.6957 - accuracy: 0.4981
16896/25000 [===================>..........] - ETA: 24s - loss: 7.6993 - accuracy: 0.4979
16928/25000 [===================>..........] - ETA: 24s - loss: 7.6983 - accuracy: 0.4979
16960/25000 [===================>..........] - ETA: 24s - loss: 7.6956 - accuracy: 0.4981
16992/25000 [===================>..........] - ETA: 24s - loss: 7.6928 - accuracy: 0.4983
17024/25000 [===================>..........] - ETA: 23s - loss: 7.6945 - accuracy: 0.4982
17056/25000 [===================>..........] - ETA: 23s - loss: 7.6936 - accuracy: 0.4982
17088/25000 [===================>..........] - ETA: 23s - loss: 7.6971 - accuracy: 0.4980
17120/25000 [===================>..........] - ETA: 23s - loss: 7.6989 - accuracy: 0.4979
17152/25000 [===================>..........] - ETA: 23s - loss: 7.7033 - accuracy: 0.4976
17184/25000 [===================>..........] - ETA: 23s - loss: 7.7014 - accuracy: 0.4977
17216/25000 [===================>..........] - ETA: 23s - loss: 7.7040 - accuracy: 0.4976
17248/25000 [===================>..........] - ETA: 23s - loss: 7.7031 - accuracy: 0.4976
17280/25000 [===================>..........] - ETA: 23s - loss: 7.7021 - accuracy: 0.4977
17312/25000 [===================>..........] - ETA: 23s - loss: 7.7047 - accuracy: 0.4975
17344/25000 [===================>..........] - ETA: 22s - loss: 7.7064 - accuracy: 0.4974
17376/25000 [===================>..........] - ETA: 22s - loss: 7.7037 - accuracy: 0.4976
17408/25000 [===================>..........] - ETA: 22s - loss: 7.7098 - accuracy: 0.4972
17440/25000 [===================>..........] - ETA: 22s - loss: 7.7053 - accuracy: 0.4975
17472/25000 [===================>..........] - ETA: 22s - loss: 7.7044 - accuracy: 0.4975
17504/25000 [====================>.........] - ETA: 22s - loss: 7.7043 - accuracy: 0.4975
17536/25000 [====================>.........] - ETA: 22s - loss: 7.7077 - accuracy: 0.4973
17568/25000 [====================>.........] - ETA: 22s - loss: 7.7076 - accuracy: 0.4973
17600/25000 [====================>.........] - ETA: 22s - loss: 7.7076 - accuracy: 0.4973
17632/25000 [====================>.........] - ETA: 22s - loss: 7.7066 - accuracy: 0.4974
17664/25000 [====================>.........] - ETA: 22s - loss: 7.7039 - accuracy: 0.4976
17696/25000 [====================>.........] - ETA: 21s - loss: 7.7004 - accuracy: 0.4978
17728/25000 [====================>.........] - ETA: 21s - loss: 7.7012 - accuracy: 0.4977
17760/25000 [====================>.........] - ETA: 21s - loss: 7.7003 - accuracy: 0.4978
17792/25000 [====================>.........] - ETA: 21s - loss: 7.7002 - accuracy: 0.4978
17824/25000 [====================>.........] - ETA: 21s - loss: 7.6976 - accuracy: 0.4980
17856/25000 [====================>.........] - ETA: 21s - loss: 7.6967 - accuracy: 0.4980
17888/25000 [====================>.........] - ETA: 21s - loss: 7.6940 - accuracy: 0.4982
17920/25000 [====================>.........] - ETA: 21s - loss: 7.6966 - accuracy: 0.4980
17952/25000 [====================>.........] - ETA: 21s - loss: 7.6957 - accuracy: 0.4981
17984/25000 [====================>.........] - ETA: 21s - loss: 7.6948 - accuracy: 0.4982
18016/25000 [====================>.........] - ETA: 20s - loss: 7.6947 - accuracy: 0.4982
18048/25000 [====================>.........] - ETA: 20s - loss: 7.6964 - accuracy: 0.4981
18080/25000 [====================>.........] - ETA: 20s - loss: 7.6963 - accuracy: 0.4981
18112/25000 [====================>.........] - ETA: 20s - loss: 7.6937 - accuracy: 0.4982
18144/25000 [====================>.........] - ETA: 20s - loss: 7.6937 - accuracy: 0.4982
18176/25000 [====================>.........] - ETA: 20s - loss: 7.6894 - accuracy: 0.4985
18208/25000 [====================>.........] - ETA: 20s - loss: 7.6877 - accuracy: 0.4986
18240/25000 [====================>.........] - ETA: 20s - loss: 7.6893 - accuracy: 0.4985
18272/25000 [====================>.........] - ETA: 20s - loss: 7.6910 - accuracy: 0.4984
18304/25000 [====================>.........] - ETA: 20s - loss: 7.6918 - accuracy: 0.4984
18336/25000 [=====================>........] - ETA: 20s - loss: 7.6892 - accuracy: 0.4985
18368/25000 [=====================>........] - ETA: 19s - loss: 7.6875 - accuracy: 0.4986
18400/25000 [=====================>........] - ETA: 19s - loss: 7.6891 - accuracy: 0.4985
18432/25000 [=====================>........] - ETA: 19s - loss: 7.6899 - accuracy: 0.4985
18464/25000 [=====================>........] - ETA: 19s - loss: 7.6932 - accuracy: 0.4983
18496/25000 [=====================>........] - ETA: 19s - loss: 7.6923 - accuracy: 0.4983
18528/25000 [=====================>........] - ETA: 19s - loss: 7.6890 - accuracy: 0.4985
18560/25000 [=====================>........] - ETA: 19s - loss: 7.6906 - accuracy: 0.4984
18592/25000 [=====================>........] - ETA: 19s - loss: 7.6889 - accuracy: 0.4985
18624/25000 [=====================>........] - ETA: 19s - loss: 7.6954 - accuracy: 0.4981
18656/25000 [=====================>........] - ETA: 19s - loss: 7.6970 - accuracy: 0.4980
18688/25000 [=====================>........] - ETA: 18s - loss: 7.6962 - accuracy: 0.4981
18720/25000 [=====================>........] - ETA: 18s - loss: 7.6969 - accuracy: 0.4980
18752/25000 [=====================>........] - ETA: 18s - loss: 7.6977 - accuracy: 0.4980
18784/25000 [=====================>........] - ETA: 18s - loss: 7.6968 - accuracy: 0.4980
18816/25000 [=====================>........] - ETA: 18s - loss: 7.6960 - accuracy: 0.4981
18848/25000 [=====================>........] - ETA: 18s - loss: 7.6967 - accuracy: 0.4980
18880/25000 [=====================>........] - ETA: 18s - loss: 7.6991 - accuracy: 0.4979
18912/25000 [=====================>........] - ETA: 18s - loss: 7.6950 - accuracy: 0.4981
18944/25000 [=====================>........] - ETA: 18s - loss: 7.6966 - accuracy: 0.4980
18976/25000 [=====================>........] - ETA: 18s - loss: 7.6949 - accuracy: 0.4982
19008/25000 [=====================>........] - ETA: 17s - loss: 7.6981 - accuracy: 0.4979
19040/25000 [=====================>........] - ETA: 17s - loss: 7.6988 - accuracy: 0.4979
19072/25000 [=====================>........] - ETA: 17s - loss: 7.6972 - accuracy: 0.4980
19104/25000 [=====================>........] - ETA: 17s - loss: 7.6955 - accuracy: 0.4981
19136/25000 [=====================>........] - ETA: 17s - loss: 7.6955 - accuracy: 0.4981
19168/25000 [======================>.......] - ETA: 17s - loss: 7.6954 - accuracy: 0.4981
19200/25000 [======================>.......] - ETA: 17s - loss: 7.6954 - accuracy: 0.4981
19232/25000 [======================>.......] - ETA: 17s - loss: 7.6961 - accuracy: 0.4981
19264/25000 [======================>.......] - ETA: 17s - loss: 7.6977 - accuracy: 0.4980
19296/25000 [======================>.......] - ETA: 17s - loss: 7.7008 - accuracy: 0.4978
19328/25000 [======================>.......] - ETA: 17s - loss: 7.6984 - accuracy: 0.4979
19360/25000 [======================>.......] - ETA: 16s - loss: 7.6975 - accuracy: 0.4980
19392/25000 [======================>.......] - ETA: 16s - loss: 7.6990 - accuracy: 0.4979
19424/25000 [======================>.......] - ETA: 16s - loss: 7.6990 - accuracy: 0.4979
19456/25000 [======================>.......] - ETA: 16s - loss: 7.7021 - accuracy: 0.4977
19488/25000 [======================>.......] - ETA: 16s - loss: 7.7020 - accuracy: 0.4977
19520/25000 [======================>.......] - ETA: 16s - loss: 7.7012 - accuracy: 0.4977
19552/25000 [======================>.......] - ETA: 16s - loss: 7.7011 - accuracy: 0.4977
19584/25000 [======================>.......] - ETA: 16s - loss: 7.7003 - accuracy: 0.4978
19616/25000 [======================>.......] - ETA: 16s - loss: 7.6979 - accuracy: 0.4980
19648/25000 [======================>.......] - ETA: 16s - loss: 7.6947 - accuracy: 0.4982
19680/25000 [======================>.......] - ETA: 15s - loss: 7.6947 - accuracy: 0.4982
19712/25000 [======================>.......] - ETA: 15s - loss: 7.6970 - accuracy: 0.4980
19744/25000 [======================>.......] - ETA: 15s - loss: 7.6992 - accuracy: 0.4979
19776/25000 [======================>.......] - ETA: 15s - loss: 7.6984 - accuracy: 0.4979
19808/25000 [======================>.......] - ETA: 15s - loss: 7.6991 - accuracy: 0.4979
19840/25000 [======================>.......] - ETA: 15s - loss: 7.6975 - accuracy: 0.4980
19872/25000 [======================>.......] - ETA: 15s - loss: 7.6952 - accuracy: 0.4981
19904/25000 [======================>.......] - ETA: 15s - loss: 7.6936 - accuracy: 0.4982
19936/25000 [======================>.......] - ETA: 15s - loss: 7.6974 - accuracy: 0.4980
19968/25000 [======================>.......] - ETA: 15s - loss: 7.6996 - accuracy: 0.4978
20000/25000 [=======================>......] - ETA: 15s - loss: 7.7027 - accuracy: 0.4976
20032/25000 [=======================>......] - ETA: 14s - loss: 7.7034 - accuracy: 0.4976
20064/25000 [=======================>......] - ETA: 14s - loss: 7.7002 - accuracy: 0.4978
20096/25000 [=======================>......] - ETA: 14s - loss: 7.6994 - accuracy: 0.4979
20128/25000 [=======================>......] - ETA: 14s - loss: 7.7001 - accuracy: 0.4978
20160/25000 [=======================>......] - ETA: 14s - loss: 7.6986 - accuracy: 0.4979
20192/25000 [=======================>......] - ETA: 14s - loss: 7.6970 - accuracy: 0.4980
20224/25000 [=======================>......] - ETA: 14s - loss: 7.6909 - accuracy: 0.4984
20256/25000 [=======================>......] - ETA: 14s - loss: 7.6924 - accuracy: 0.4983
20288/25000 [=======================>......] - ETA: 14s - loss: 7.6923 - accuracy: 0.4983
20320/25000 [=======================>......] - ETA: 14s - loss: 7.6893 - accuracy: 0.4985
20352/25000 [=======================>......] - ETA: 13s - loss: 7.6885 - accuracy: 0.4986
20384/25000 [=======================>......] - ETA: 13s - loss: 7.6914 - accuracy: 0.4984
20416/25000 [=======================>......] - ETA: 13s - loss: 7.6876 - accuracy: 0.4986
20448/25000 [=======================>......] - ETA: 13s - loss: 7.6854 - accuracy: 0.4988
20480/25000 [=======================>......] - ETA: 13s - loss: 7.6831 - accuracy: 0.4989
20512/25000 [=======================>......] - ETA: 13s - loss: 7.6846 - accuracy: 0.4988
20544/25000 [=======================>......] - ETA: 13s - loss: 7.6830 - accuracy: 0.4989
20576/25000 [=======================>......] - ETA: 13s - loss: 7.6808 - accuracy: 0.4991
20608/25000 [=======================>......] - ETA: 13s - loss: 7.6815 - accuracy: 0.4990
20640/25000 [=======================>......] - ETA: 13s - loss: 7.6852 - accuracy: 0.4988
20672/25000 [=======================>......] - ETA: 13s - loss: 7.6844 - accuracy: 0.4988
20704/25000 [=======================>......] - ETA: 12s - loss: 7.6851 - accuracy: 0.4988
20736/25000 [=======================>......] - ETA: 12s - loss: 7.6836 - accuracy: 0.4989
20768/25000 [=======================>......] - ETA: 12s - loss: 7.6799 - accuracy: 0.4991
20800/25000 [=======================>......] - ETA: 12s - loss: 7.6799 - accuracy: 0.4991
20832/25000 [=======================>......] - ETA: 12s - loss: 7.6806 - accuracy: 0.4991
20864/25000 [========================>.....] - ETA: 12s - loss: 7.6835 - accuracy: 0.4989
20896/25000 [========================>.....] - ETA: 12s - loss: 7.6842 - accuracy: 0.4989
20928/25000 [========================>.....] - ETA: 12s - loss: 7.6857 - accuracy: 0.4988
20960/25000 [========================>.....] - ETA: 12s - loss: 7.6856 - accuracy: 0.4988
20992/25000 [========================>.....] - ETA: 12s - loss: 7.6841 - accuracy: 0.4989
21024/25000 [========================>.....] - ETA: 11s - loss: 7.6849 - accuracy: 0.4988
21056/25000 [========================>.....] - ETA: 11s - loss: 7.6841 - accuracy: 0.4989
21088/25000 [========================>.....] - ETA: 11s - loss: 7.6797 - accuracy: 0.4991
21120/25000 [========================>.....] - ETA: 11s - loss: 7.6790 - accuracy: 0.4992
21152/25000 [========================>.....] - ETA: 11s - loss: 7.6811 - accuracy: 0.4991
21184/25000 [========================>.....] - ETA: 11s - loss: 7.6825 - accuracy: 0.4990
21216/25000 [========================>.....] - ETA: 11s - loss: 7.6840 - accuracy: 0.4989
21248/25000 [========================>.....] - ETA: 11s - loss: 7.6847 - accuracy: 0.4988
21280/25000 [========================>.....] - ETA: 11s - loss: 7.6875 - accuracy: 0.4986
21312/25000 [========================>.....] - ETA: 11s - loss: 7.6853 - accuracy: 0.4988
21344/25000 [========================>.....] - ETA: 11s - loss: 7.6846 - accuracy: 0.4988
21376/25000 [========================>.....] - ETA: 10s - loss: 7.6831 - accuracy: 0.4989
21408/25000 [========================>.....] - ETA: 10s - loss: 7.6845 - accuracy: 0.4988
21440/25000 [========================>.....] - ETA: 10s - loss: 7.6831 - accuracy: 0.4989
21472/25000 [========================>.....] - ETA: 10s - loss: 7.6809 - accuracy: 0.4991
21504/25000 [========================>.....] - ETA: 10s - loss: 7.6816 - accuracy: 0.4990
21536/25000 [========================>.....] - ETA: 10s - loss: 7.6866 - accuracy: 0.4987
21568/25000 [========================>.....] - ETA: 10s - loss: 7.6865 - accuracy: 0.4987
21600/25000 [========================>.....] - ETA: 10s - loss: 7.6886 - accuracy: 0.4986
21632/25000 [========================>.....] - ETA: 10s - loss: 7.6865 - accuracy: 0.4987
21664/25000 [========================>.....] - ETA: 10s - loss: 7.6893 - accuracy: 0.4985
21696/25000 [=========================>....] - ETA: 9s - loss: 7.6885 - accuracy: 0.4986 
21728/25000 [=========================>....] - ETA: 9s - loss: 7.6885 - accuracy: 0.4986
21760/25000 [=========================>....] - ETA: 9s - loss: 7.6920 - accuracy: 0.4983
21792/25000 [=========================>....] - ETA: 9s - loss: 7.6934 - accuracy: 0.4983
21824/25000 [=========================>....] - ETA: 9s - loss: 7.6919 - accuracy: 0.4984
21856/25000 [=========================>....] - ETA: 9s - loss: 7.6912 - accuracy: 0.4984
21888/25000 [=========================>....] - ETA: 9s - loss: 7.6911 - accuracy: 0.4984
21920/25000 [=========================>....] - ETA: 9s - loss: 7.6890 - accuracy: 0.4985
21952/25000 [=========================>....] - ETA: 9s - loss: 7.6904 - accuracy: 0.4985
21984/25000 [=========================>....] - ETA: 9s - loss: 7.6882 - accuracy: 0.4986
22016/25000 [=========================>....] - ETA: 8s - loss: 7.6854 - accuracy: 0.4988
22048/25000 [=========================>....] - ETA: 8s - loss: 7.6847 - accuracy: 0.4988
22080/25000 [=========================>....] - ETA: 8s - loss: 7.6805 - accuracy: 0.4991
22112/25000 [=========================>....] - ETA: 8s - loss: 7.6798 - accuracy: 0.4991
22144/25000 [=========================>....] - ETA: 8s - loss: 7.6756 - accuracy: 0.4994
22176/25000 [=========================>....] - ETA: 8s - loss: 7.6735 - accuracy: 0.4995
22208/25000 [=========================>....] - ETA: 8s - loss: 7.6735 - accuracy: 0.4995
22240/25000 [=========================>....] - ETA: 8s - loss: 7.6714 - accuracy: 0.4997
22272/25000 [=========================>....] - ETA: 8s - loss: 7.6721 - accuracy: 0.4996
22304/25000 [=========================>....] - ETA: 8s - loss: 7.6749 - accuracy: 0.4995
22336/25000 [=========================>....] - ETA: 8s - loss: 7.6755 - accuracy: 0.4994
22368/25000 [=========================>....] - ETA: 7s - loss: 7.6748 - accuracy: 0.4995
22400/25000 [=========================>....] - ETA: 7s - loss: 7.6748 - accuracy: 0.4995
22432/25000 [=========================>....] - ETA: 7s - loss: 7.6769 - accuracy: 0.4993
22464/25000 [=========================>....] - ETA: 7s - loss: 7.6741 - accuracy: 0.4995
22496/25000 [=========================>....] - ETA: 7s - loss: 7.6762 - accuracy: 0.4994
22528/25000 [==========================>...] - ETA: 7s - loss: 7.6741 - accuracy: 0.4995
22560/25000 [==========================>...] - ETA: 7s - loss: 7.6755 - accuracy: 0.4994
22592/25000 [==========================>...] - ETA: 7s - loss: 7.6768 - accuracy: 0.4993
22624/25000 [==========================>...] - ETA: 7s - loss: 7.6781 - accuracy: 0.4992
22656/25000 [==========================>...] - ETA: 7s - loss: 7.6781 - accuracy: 0.4992
22688/25000 [==========================>...] - ETA: 6s - loss: 7.6781 - accuracy: 0.4993
22720/25000 [==========================>...] - ETA: 6s - loss: 7.6754 - accuracy: 0.4994
22752/25000 [==========================>...] - ETA: 6s - loss: 7.6740 - accuracy: 0.4995
22784/25000 [==========================>...] - ETA: 6s - loss: 7.6754 - accuracy: 0.4994
22816/25000 [==========================>...] - ETA: 6s - loss: 7.6754 - accuracy: 0.4994
22848/25000 [==========================>...] - ETA: 6s - loss: 7.6727 - accuracy: 0.4996
22880/25000 [==========================>...] - ETA: 6s - loss: 7.6713 - accuracy: 0.4997
22912/25000 [==========================>...] - ETA: 6s - loss: 7.6706 - accuracy: 0.4997
22944/25000 [==========================>...] - ETA: 6s - loss: 7.6733 - accuracy: 0.4996
22976/25000 [==========================>...] - ETA: 6s - loss: 7.6713 - accuracy: 0.4997
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6693 - accuracy: 0.4998
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6733 - accuracy: 0.4996
23072/25000 [==========================>...] - ETA: 5s - loss: 7.6733 - accuracy: 0.4996
23104/25000 [==========================>...] - ETA: 5s - loss: 7.6686 - accuracy: 0.4999
23136/25000 [==========================>...] - ETA: 5s - loss: 7.6732 - accuracy: 0.4996
23168/25000 [==========================>...] - ETA: 5s - loss: 7.6719 - accuracy: 0.4997
23200/25000 [==========================>...] - ETA: 5s - loss: 7.6746 - accuracy: 0.4995
23232/25000 [==========================>...] - ETA: 5s - loss: 7.6706 - accuracy: 0.4997
23264/25000 [==========================>...] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6712 - accuracy: 0.4997
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6712 - accuracy: 0.4997
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
23456/25000 [===========================>..] - ETA: 4s - loss: 7.6660 - accuracy: 0.5000
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6640 - accuracy: 0.5002
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6647 - accuracy: 0.5001
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6686 - accuracy: 0.4999
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6686 - accuracy: 0.4999
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6634 - accuracy: 0.5002
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6615 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6641 - accuracy: 0.5002
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24192/25000 [============================>.] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24224/25000 [============================>.] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24256/25000 [============================>.] - ETA: 2s - loss: 7.6723 - accuracy: 0.4996
24288/25000 [============================>.] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24320/25000 [============================>.] - ETA: 2s - loss: 7.6723 - accuracy: 0.4996
24352/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24384/25000 [============================>.] - ETA: 1s - loss: 7.6748 - accuracy: 0.4995
24416/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24448/25000 [============================>.] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24480/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24512/25000 [============================>.] - ETA: 1s - loss: 7.6722 - accuracy: 0.4996
24544/25000 [============================>.] - ETA: 1s - loss: 7.6722 - accuracy: 0.4996
24576/25000 [============================>.] - ETA: 1s - loss: 7.6741 - accuracy: 0.4995
24608/25000 [============================>.] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
24640/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24672/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24704/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24736/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24768/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24800/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24832/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24864/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24896/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24928/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 90s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

[0;36m  File [0;32m"/home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py"[0;36m, line [0;32m248[0m
[0;31m    We then reshape the forecasts into the correct data shape for submission ...[0m
[0m          ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax






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

[0;36m  File [0;32m"/home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark_timeseries_m5.py"[0;36m, line [0;32m248[0m
[0;31m    We then reshape the forecasts into the correct data shape for submission ...[0m
[0m          ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax

