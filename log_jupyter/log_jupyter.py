
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/bc1e016c27c40275206592c0baf763e2bb53090d', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'bc1e016c27c40275206592c0baf763e2bb53090d', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/bc1e016c27c40275206592c0baf763e2bb53090d

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/bc1e016c27c40275206592c0baf763e2bb53090d

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/bc1e016c27c40275206592c0baf763e2bb53090d

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
 40%|████      | 2/5 [00:49<01:14, 24.74s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.2480534346039771, 'embedding_size_factor': 1.0103066894276571, 'layers.choice': 3, 'learning_rate': 0.0009519309137631976, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 0.00024654766957727574} and reward: 0.3642
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xcf\xc07\x06\xa4lXX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0*7X\xe0Y\\X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?O1`\x12B\xd3$X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?0(b-\x80T\x88u.' and reward: 0.3642
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xcf\xc07\x06\xa4lXX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0*7X\xe0Y\\X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?O1`\x12B\xd3$X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?0(b-\x80T\x88u.' and reward: 0.3642
 60%|██████    | 3/5 [01:42<01:06, 33.15s/it] 60%|██████    | 3/5 [01:42<01:08, 34.09s/it]
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.45251367274978194, 'embedding_size_factor': 0.8918423719836649, 'layers.choice': 0, 'learning_rate': 0.00293545719098432, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 2.077127423078213e-06} and reward: 0.377
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdc\xf5\xfb\xe8]\x00\x92X\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\x89\xf9\x03\x9bk\xc9X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?h\x0c\x19\x94Jn\x0eX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xc1l\x98\xe0\x84\x9b\xcfu.' and reward: 0.377
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdc\xf5\xfb\xe8]\x00\x92X\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\x89\xf9\x03\x9bk\xc9X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?h\x0c\x19\x94Jn\x0eX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xc1l\x98\xe0\x84\x9b\xcfu.' and reward: 0.377
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 152.97047185897827
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.77s of the -35.49s of remaining time.
Ensemble size: 46
Ensemble weights: 
[0.58695652 0.2826087  0.13043478]
	0.3906	 = Validation accuracy score
	1.05s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 156.57s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f9e9a9c59e8> 

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
 [ 0.06873479 -0.02934621  0.05881377  0.07964936  0.04399765  0.03972627]
 [-0.00892676  0.0523865   0.12941834  0.01662662 -0.02763544  0.04448456]
 [ 0.08428916  0.20267749  0.18661733  0.08454835  0.19703709 -0.00582939]
 [ 0.11147094 -0.09393959 -0.1709775  -0.1761221  -0.05883961 -0.08212345]
 [ 0.22959638 -0.16365677 -0.20476168 -0.4412989  -0.15505637  0.95342821]
 [ 0.05772328  0.34648657 -0.1056451   0.32994387 -0.27583003  0.45290661]
 [-0.35397419 -0.04963214  0.14852127 -0.11158963  0.42082462  0.44815773]
 [ 0.25764081 -0.00969158  0.78867012  0.08015133 -0.21951663 -0.18211442]
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
{'loss': 0.49113836884498596, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-19 13:34:39.201747: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.48041269183158875, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-19 13:34:40.318684: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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

    8192/17464789 [..............................] - ETA: 4s
 1417216/17464789 [=>............................] - ETA: 0s
 4390912/17464789 [======>.......................] - ETA: 0s
12156928/17464789 [===================>..........] - ETA: 0s
17096704/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-19 13:34:51.784325: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-19 13:34:51.788857: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-19 13:34:51.789003: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5566e6278120 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-19 13:34:51.789017: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:44 - loss: 5.7500 - accuracy: 0.6250
   64/25000 [..............................] - ETA: 2:53 - loss: 7.1875 - accuracy: 0.5312
   96/25000 [..............................] - ETA: 2:14 - loss: 7.3472 - accuracy: 0.5208
  128/25000 [..............................] - ETA: 1:55 - loss: 6.8281 - accuracy: 0.5547
  160/25000 [..............................] - ETA: 1:44 - loss: 6.7083 - accuracy: 0.5625
  192/25000 [..............................] - ETA: 1:36 - loss: 7.0277 - accuracy: 0.5417
  224/25000 [..............................] - ETA: 1:32 - loss: 7.1875 - accuracy: 0.5312
  256/25000 [..............................] - ETA: 1:29 - loss: 7.3072 - accuracy: 0.5234
  288/25000 [..............................] - ETA: 1:26 - loss: 7.0277 - accuracy: 0.5417
  320/25000 [..............................] - ETA: 1:23 - loss: 7.1875 - accuracy: 0.5312
  352/25000 [..............................] - ETA: 1:21 - loss: 7.2310 - accuracy: 0.5284
  384/25000 [..............................] - ETA: 1:19 - loss: 7.1475 - accuracy: 0.5339
  416/25000 [..............................] - ETA: 1:18 - loss: 7.2980 - accuracy: 0.5240
  448/25000 [..............................] - ETA: 1:16 - loss: 7.3586 - accuracy: 0.5201
  480/25000 [..............................] - ETA: 1:15 - loss: 7.3791 - accuracy: 0.5188
  512/25000 [..............................] - ETA: 1:14 - loss: 7.2474 - accuracy: 0.5273
  544/25000 [..............................] - ETA: 1:13 - loss: 7.3848 - accuracy: 0.5184
  576/25000 [..............................] - ETA: 1:12 - loss: 7.5069 - accuracy: 0.5104
  608/25000 [..............................] - ETA: 1:12 - loss: 7.5405 - accuracy: 0.5082
  640/25000 [..............................] - ETA: 1:11 - loss: 7.5947 - accuracy: 0.5047
  672/25000 [..............................] - ETA: 1:10 - loss: 7.5754 - accuracy: 0.5060
  704/25000 [..............................] - ETA: 1:10 - loss: 7.6013 - accuracy: 0.5043
  736/25000 [..............................] - ETA: 1:09 - loss: 7.6041 - accuracy: 0.5041
  768/25000 [..............................] - ETA: 1:09 - loss: 7.6267 - accuracy: 0.5026
  800/25000 [..............................] - ETA: 1:08 - loss: 7.6666 - accuracy: 0.5000
  832/25000 [..............................] - ETA: 1:07 - loss: 7.6850 - accuracy: 0.4988
  864/25000 [>.............................] - ETA: 1:07 - loss: 7.5779 - accuracy: 0.5058
  896/25000 [>.............................] - ETA: 1:07 - loss: 7.4955 - accuracy: 0.5112
  928/25000 [>.............................] - ETA: 1:06 - loss: 7.5840 - accuracy: 0.5054
  960/25000 [>.............................] - ETA: 1:06 - loss: 7.5868 - accuracy: 0.5052
  992/25000 [>.............................] - ETA: 1:06 - loss: 7.6357 - accuracy: 0.5020
 1024/25000 [>.............................] - ETA: 1:05 - loss: 7.6666 - accuracy: 0.5000
 1056/25000 [>.............................] - ETA: 1:05 - loss: 7.6521 - accuracy: 0.5009
 1088/25000 [>.............................] - ETA: 1:05 - loss: 7.6525 - accuracy: 0.5009
 1120/25000 [>.............................] - ETA: 1:05 - loss: 7.6255 - accuracy: 0.5027
 1152/25000 [>.............................] - ETA: 1:05 - loss: 7.6134 - accuracy: 0.5035
 1184/25000 [>.............................] - ETA: 1:05 - loss: 7.5630 - accuracy: 0.5068
 1216/25000 [>.............................] - ETA: 1:04 - loss: 7.5657 - accuracy: 0.5066
 1248/25000 [>.............................] - ETA: 1:04 - loss: 7.6052 - accuracy: 0.5040
 1280/25000 [>.............................] - ETA: 1:04 - loss: 7.5708 - accuracy: 0.5063
 1312/25000 [>.............................] - ETA: 1:04 - loss: 7.5848 - accuracy: 0.5053
 1344/25000 [>.............................] - ETA: 1:04 - loss: 7.5868 - accuracy: 0.5052
 1376/25000 [>.............................] - ETA: 1:04 - loss: 7.5440 - accuracy: 0.5080
 1408/25000 [>.............................] - ETA: 1:04 - loss: 7.5142 - accuracy: 0.5099
 1440/25000 [>.............................] - ETA: 1:04 - loss: 7.5069 - accuracy: 0.5104
 1472/25000 [>.............................] - ETA: 1:04 - loss: 7.4895 - accuracy: 0.5115
 1504/25000 [>.............................] - ETA: 1:04 - loss: 7.5035 - accuracy: 0.5106
 1536/25000 [>.............................] - ETA: 1:03 - loss: 7.4869 - accuracy: 0.5117
 1568/25000 [>.............................] - ETA: 1:03 - loss: 7.4808 - accuracy: 0.5121
 1600/25000 [>.............................] - ETA: 1:03 - loss: 7.4750 - accuracy: 0.5125
 1632/25000 [>.............................] - ETA: 1:03 - loss: 7.4505 - accuracy: 0.5141
 1664/25000 [>.............................] - ETA: 1:03 - loss: 7.4547 - accuracy: 0.5138
 1696/25000 [=>............................] - ETA: 1:03 - loss: 7.4316 - accuracy: 0.5153
 1728/25000 [=>............................] - ETA: 1:02 - loss: 7.4359 - accuracy: 0.5150
 1760/25000 [=>............................] - ETA: 1:02 - loss: 7.4314 - accuracy: 0.5153
 1792/25000 [=>............................] - ETA: 1:02 - loss: 7.4185 - accuracy: 0.5162
 1824/25000 [=>............................] - ETA: 1:02 - loss: 7.3724 - accuracy: 0.5192
 1856/25000 [=>............................] - ETA: 1:01 - loss: 7.3857 - accuracy: 0.5183
 1888/25000 [=>............................] - ETA: 1:01 - loss: 7.3824 - accuracy: 0.5185
 1920/25000 [=>............................] - ETA: 1:01 - loss: 7.4031 - accuracy: 0.5172
 1952/25000 [=>............................] - ETA: 1:01 - loss: 7.4153 - accuracy: 0.5164
 1984/25000 [=>............................] - ETA: 1:01 - loss: 7.3961 - accuracy: 0.5176
 2016/25000 [=>............................] - ETA: 1:01 - loss: 7.4004 - accuracy: 0.5174
 2048/25000 [=>............................] - ETA: 1:00 - loss: 7.4121 - accuracy: 0.5166
 2080/25000 [=>............................] - ETA: 1:00 - loss: 7.4307 - accuracy: 0.5154
 2112/25000 [=>............................] - ETA: 1:00 - loss: 7.4488 - accuracy: 0.5142
 2144/25000 [=>............................] - ETA: 1:00 - loss: 7.4664 - accuracy: 0.5131
 2176/25000 [=>............................] - ETA: 1:00 - loss: 7.4693 - accuracy: 0.5129
 2208/25000 [=>............................] - ETA: 1:00 - loss: 7.4583 - accuracy: 0.5136
 2240/25000 [=>............................] - ETA: 1:00 - loss: 7.4818 - accuracy: 0.5121
 2272/25000 [=>............................] - ETA: 1:00 - loss: 7.4642 - accuracy: 0.5132
 2304/25000 [=>............................] - ETA: 59s - loss: 7.4470 - accuracy: 0.5143 
 2336/25000 [=>............................] - ETA: 59s - loss: 7.4697 - accuracy: 0.5128
 2368/25000 [=>............................] - ETA: 59s - loss: 7.4465 - accuracy: 0.5144
 2400/25000 [=>............................] - ETA: 59s - loss: 7.4494 - accuracy: 0.5142
 2432/25000 [=>............................] - ETA: 59s - loss: 7.4270 - accuracy: 0.5156
 2464/25000 [=>............................] - ETA: 59s - loss: 7.4301 - accuracy: 0.5154
 2496/25000 [=>............................] - ETA: 59s - loss: 7.4393 - accuracy: 0.5148
 2528/25000 [==>...........................] - ETA: 58s - loss: 7.4604 - accuracy: 0.5134
 2560/25000 [==>...........................] - ETA: 58s - loss: 7.4510 - accuracy: 0.5141
 2592/25000 [==>...........................] - ETA: 58s - loss: 7.4182 - accuracy: 0.5162
 2624/25000 [==>...........................] - ETA: 58s - loss: 7.4387 - accuracy: 0.5149
 2656/25000 [==>...........................] - ETA: 58s - loss: 7.4357 - accuracy: 0.5151
 2688/25000 [==>...........................] - ETA: 58s - loss: 7.4442 - accuracy: 0.5145
 2720/25000 [==>...........................] - ETA: 58s - loss: 7.4580 - accuracy: 0.5136
 2752/25000 [==>...........................] - ETA: 58s - loss: 7.4716 - accuracy: 0.5127
 2784/25000 [==>...........................] - ETA: 58s - loss: 7.4794 - accuracy: 0.5122
 2816/25000 [==>...........................] - ETA: 58s - loss: 7.5033 - accuracy: 0.5107
 2848/25000 [==>...........................] - ETA: 58s - loss: 7.5266 - accuracy: 0.5091
 2880/25000 [==>...........................] - ETA: 57s - loss: 7.5229 - accuracy: 0.5094
 2912/25000 [==>...........................] - ETA: 57s - loss: 7.5560 - accuracy: 0.5072
 2944/25000 [==>...........................] - ETA: 57s - loss: 7.5260 - accuracy: 0.5092
 2976/25000 [==>...........................] - ETA: 57s - loss: 7.5172 - accuracy: 0.5097
 3008/25000 [==>...........................] - ETA: 57s - loss: 7.5188 - accuracy: 0.5096
 3040/25000 [==>...........................] - ETA: 57s - loss: 7.5153 - accuracy: 0.5099
 3072/25000 [==>...........................] - ETA: 57s - loss: 7.5169 - accuracy: 0.5098
 3104/25000 [==>...........................] - ETA: 57s - loss: 7.5036 - accuracy: 0.5106
 3136/25000 [==>...........................] - ETA: 57s - loss: 7.5053 - accuracy: 0.5105
 3168/25000 [==>...........................] - ETA: 57s - loss: 7.5117 - accuracy: 0.5101
 3200/25000 [==>...........................] - ETA: 56s - loss: 7.4989 - accuracy: 0.5109
 3232/25000 [==>...........................] - ETA: 56s - loss: 7.4863 - accuracy: 0.5118
 3264/25000 [==>...........................] - ETA: 56s - loss: 7.4740 - accuracy: 0.5126
 3296/25000 [==>...........................] - ETA: 56s - loss: 7.4666 - accuracy: 0.5130
 3328/25000 [==>...........................] - ETA: 56s - loss: 7.4593 - accuracy: 0.5135
 3360/25000 [===>..........................] - ETA: 56s - loss: 7.4613 - accuracy: 0.5134
 3392/25000 [===>..........................] - ETA: 56s - loss: 7.4722 - accuracy: 0.5127
 3424/25000 [===>..........................] - ETA: 56s - loss: 7.4651 - accuracy: 0.5131
 3456/25000 [===>..........................] - ETA: 56s - loss: 7.4670 - accuracy: 0.5130
 3488/25000 [===>..........................] - ETA: 55s - loss: 7.4512 - accuracy: 0.5140
 3520/25000 [===>..........................] - ETA: 55s - loss: 7.4750 - accuracy: 0.5125
 3552/25000 [===>..........................] - ETA: 55s - loss: 7.4983 - accuracy: 0.5110
 3584/25000 [===>..........................] - ETA: 55s - loss: 7.4955 - accuracy: 0.5112
 3616/25000 [===>..........................] - ETA: 55s - loss: 7.5097 - accuracy: 0.5102
 3648/25000 [===>..........................] - ETA: 55s - loss: 7.5069 - accuracy: 0.5104
 3680/25000 [===>..........................] - ETA: 55s - loss: 7.5041 - accuracy: 0.5106
 3712/25000 [===>..........................] - ETA: 55s - loss: 7.5138 - accuracy: 0.5100
 3744/25000 [===>..........................] - ETA: 55s - loss: 7.4946 - accuracy: 0.5112
 3776/25000 [===>..........................] - ETA: 55s - loss: 7.4879 - accuracy: 0.5117
 3808/25000 [===>..........................] - ETA: 54s - loss: 7.4774 - accuracy: 0.5123
 3840/25000 [===>..........................] - ETA: 54s - loss: 7.4789 - accuracy: 0.5122
 3872/25000 [===>..........................] - ETA: 54s - loss: 7.4686 - accuracy: 0.5129
 3904/25000 [===>..........................] - ETA: 54s - loss: 7.4624 - accuracy: 0.5133
 3936/25000 [===>..........................] - ETA: 54s - loss: 7.4835 - accuracy: 0.5119
 3968/25000 [===>..........................] - ETA: 54s - loss: 7.4966 - accuracy: 0.5111
 4000/25000 [===>..........................] - ETA: 54s - loss: 7.4941 - accuracy: 0.5113
 4032/25000 [===>..........................] - ETA: 54s - loss: 7.5145 - accuracy: 0.5099
 4064/25000 [===>..........................] - ETA: 54s - loss: 7.5044 - accuracy: 0.5106
 4096/25000 [===>..........................] - ETA: 54s - loss: 7.4944 - accuracy: 0.5112
 4128/25000 [===>..........................] - ETA: 54s - loss: 7.5106 - accuracy: 0.5102
 4160/25000 [===>..........................] - ETA: 54s - loss: 7.5302 - accuracy: 0.5089
 4192/25000 [====>.........................] - ETA: 54s - loss: 7.5386 - accuracy: 0.5083
 4224/25000 [====>.........................] - ETA: 54s - loss: 7.5250 - accuracy: 0.5092
 4256/25000 [====>.........................] - ETA: 53s - loss: 7.5153 - accuracy: 0.5099
 4288/25000 [====>.........................] - ETA: 53s - loss: 7.5164 - accuracy: 0.5098
 4320/25000 [====>.........................] - ETA: 53s - loss: 7.5069 - accuracy: 0.5104
 4352/25000 [====>.........................] - ETA: 53s - loss: 7.5116 - accuracy: 0.5101
 4384/25000 [====>.........................] - ETA: 53s - loss: 7.5022 - accuracy: 0.5107
 4416/25000 [====>.........................] - ETA: 53s - loss: 7.5138 - accuracy: 0.5100
 4448/25000 [====>.........................] - ETA: 53s - loss: 7.5253 - accuracy: 0.5092
 4480/25000 [====>.........................] - ETA: 53s - loss: 7.5229 - accuracy: 0.5094
 4512/25000 [====>.........................] - ETA: 53s - loss: 7.5171 - accuracy: 0.5098
 4544/25000 [====>.........................] - ETA: 53s - loss: 7.5181 - accuracy: 0.5097
 4576/25000 [====>.........................] - ETA: 53s - loss: 7.5259 - accuracy: 0.5092
 4608/25000 [====>.........................] - ETA: 52s - loss: 7.5269 - accuracy: 0.5091
 4640/25000 [====>.........................] - ETA: 52s - loss: 7.5344 - accuracy: 0.5086
 4672/25000 [====>.........................] - ETA: 52s - loss: 7.5321 - accuracy: 0.5088
 4704/25000 [====>.........................] - ETA: 52s - loss: 7.5232 - accuracy: 0.5094
 4736/25000 [====>.........................] - ETA: 52s - loss: 7.5436 - accuracy: 0.5080
 4768/25000 [====>.........................] - ETA: 52s - loss: 7.5412 - accuracy: 0.5082
 4800/25000 [====>.........................] - ETA: 52s - loss: 7.5388 - accuracy: 0.5083
 4832/25000 [====>.........................] - ETA: 52s - loss: 7.5365 - accuracy: 0.5085
 4864/25000 [====>.........................] - ETA: 52s - loss: 7.5342 - accuracy: 0.5086
 4896/25000 [====>.........................] - ETA: 52s - loss: 7.5476 - accuracy: 0.5078
 4928/25000 [====>.........................] - ETA: 52s - loss: 7.5608 - accuracy: 0.5069
 4960/25000 [====>.........................] - ETA: 52s - loss: 7.5584 - accuracy: 0.5071
 4992/25000 [====>.........................] - ETA: 51s - loss: 7.5591 - accuracy: 0.5070
 5024/25000 [=====>........................] - ETA: 51s - loss: 7.5690 - accuracy: 0.5064
 5056/25000 [=====>........................] - ETA: 51s - loss: 7.5817 - accuracy: 0.5055
 5088/25000 [=====>........................] - ETA: 51s - loss: 7.5883 - accuracy: 0.5051
 5120/25000 [=====>........................] - ETA: 51s - loss: 7.5798 - accuracy: 0.5057
 5152/25000 [=====>........................] - ETA: 51s - loss: 7.5982 - accuracy: 0.5045
 5184/25000 [=====>........................] - ETA: 51s - loss: 7.5897 - accuracy: 0.5050
 5216/25000 [=====>........................] - ETA: 51s - loss: 7.5931 - accuracy: 0.5048
 5248/25000 [=====>........................] - ETA: 51s - loss: 7.5994 - accuracy: 0.5044
 5280/25000 [=====>........................] - ETA: 51s - loss: 7.5882 - accuracy: 0.5051
 5312/25000 [=====>........................] - ETA: 51s - loss: 7.5858 - accuracy: 0.5053
 5344/25000 [=====>........................] - ETA: 51s - loss: 7.5805 - accuracy: 0.5056
 5376/25000 [=====>........................] - ETA: 50s - loss: 7.5668 - accuracy: 0.5065
 5408/25000 [=====>........................] - ETA: 50s - loss: 7.5731 - accuracy: 0.5061
 5440/25000 [=====>........................] - ETA: 50s - loss: 7.5877 - accuracy: 0.5051
 5472/25000 [=====>........................] - ETA: 50s - loss: 7.5770 - accuracy: 0.5058
 5504/25000 [=====>........................] - ETA: 50s - loss: 7.5691 - accuracy: 0.5064
 5536/25000 [=====>........................] - ETA: 50s - loss: 7.5586 - accuracy: 0.5070
 5568/25000 [=====>........................] - ETA: 50s - loss: 7.5647 - accuracy: 0.5066
 5600/25000 [=====>........................] - ETA: 50s - loss: 7.5544 - accuracy: 0.5073
 5632/25000 [=====>........................] - ETA: 50s - loss: 7.5632 - accuracy: 0.5067
 5664/25000 [=====>........................] - ETA: 50s - loss: 7.5583 - accuracy: 0.5071
 5696/25000 [=====>........................] - ETA: 50s - loss: 7.5563 - accuracy: 0.5072
 5728/25000 [=====>........................] - ETA: 50s - loss: 7.5595 - accuracy: 0.5070
 5760/25000 [=====>........................] - ETA: 49s - loss: 7.5655 - accuracy: 0.5066
 5792/25000 [=====>........................] - ETA: 49s - loss: 7.5687 - accuracy: 0.5064
 5824/25000 [=====>........................] - ETA: 49s - loss: 7.5745 - accuracy: 0.5060
 5856/25000 [======>.......................] - ETA: 49s - loss: 7.5724 - accuracy: 0.5061
 5888/25000 [======>.......................] - ETA: 49s - loss: 7.5807 - accuracy: 0.5056
 5920/25000 [======>.......................] - ETA: 49s - loss: 7.5786 - accuracy: 0.5057
 5952/25000 [======>.......................] - ETA: 49s - loss: 7.5765 - accuracy: 0.5059
 5984/25000 [======>.......................] - ETA: 49s - loss: 7.5769 - accuracy: 0.5058
 6016/25000 [======>.......................] - ETA: 49s - loss: 7.5851 - accuracy: 0.5053
 6048/25000 [======>.......................] - ETA: 49s - loss: 7.5855 - accuracy: 0.5053
 6080/25000 [======>.......................] - ETA: 49s - loss: 7.5834 - accuracy: 0.5054
 6112/25000 [======>.......................] - ETA: 49s - loss: 7.5888 - accuracy: 0.5051
 6144/25000 [======>.......................] - ETA: 49s - loss: 7.5843 - accuracy: 0.5054
 6176/25000 [======>.......................] - ETA: 48s - loss: 7.5971 - accuracy: 0.5045
 6208/25000 [======>.......................] - ETA: 48s - loss: 7.5999 - accuracy: 0.5043
 6240/25000 [======>.......................] - ETA: 48s - loss: 7.5929 - accuracy: 0.5048
 6272/25000 [======>.......................] - ETA: 48s - loss: 7.5908 - accuracy: 0.5049
 6304/25000 [======>.......................] - ETA: 48s - loss: 7.5912 - accuracy: 0.5049
 6336/25000 [======>.......................] - ETA: 48s - loss: 7.5964 - accuracy: 0.5046
 6368/25000 [======>.......................] - ETA: 48s - loss: 7.6064 - accuracy: 0.5039
 6400/25000 [======>.......................] - ETA: 48s - loss: 7.6139 - accuracy: 0.5034
 6432/25000 [======>.......................] - ETA: 48s - loss: 7.6094 - accuracy: 0.5037
 6464/25000 [======>.......................] - ETA: 48s - loss: 7.6121 - accuracy: 0.5036
 6496/25000 [======>.......................] - ETA: 48s - loss: 7.6123 - accuracy: 0.5035
 6528/25000 [======>.......................] - ETA: 48s - loss: 7.6032 - accuracy: 0.5041
 6560/25000 [======>.......................] - ETA: 47s - loss: 7.5965 - accuracy: 0.5046
 6592/25000 [======>.......................] - ETA: 47s - loss: 7.5945 - accuracy: 0.5047
 6624/25000 [======>.......................] - ETA: 47s - loss: 7.5972 - accuracy: 0.5045
 6656/25000 [======>.......................] - ETA: 47s - loss: 7.5998 - accuracy: 0.5044
 6688/25000 [=======>......................] - ETA: 47s - loss: 7.6093 - accuracy: 0.5037
 6720/25000 [=======>......................] - ETA: 47s - loss: 7.6141 - accuracy: 0.5034
 6752/25000 [=======>......................] - ETA: 47s - loss: 7.6189 - accuracy: 0.5031
 6784/25000 [=======>......................] - ETA: 47s - loss: 7.6169 - accuracy: 0.5032
 6816/25000 [=======>......................] - ETA: 47s - loss: 7.6171 - accuracy: 0.5032
 6848/25000 [=======>......................] - ETA: 47s - loss: 7.6196 - accuracy: 0.5031
 6880/25000 [=======>......................] - ETA: 47s - loss: 7.6265 - accuracy: 0.5026
 6912/25000 [=======>......................] - ETA: 47s - loss: 7.6267 - accuracy: 0.5026
 6944/25000 [=======>......................] - ETA: 46s - loss: 7.6313 - accuracy: 0.5023
 6976/25000 [=======>......................] - ETA: 46s - loss: 7.6227 - accuracy: 0.5029
 7008/25000 [=======>......................] - ETA: 46s - loss: 7.6207 - accuracy: 0.5030
 7040/25000 [=======>......................] - ETA: 46s - loss: 7.6231 - accuracy: 0.5028
 7072/25000 [=======>......................] - ETA: 46s - loss: 7.6233 - accuracy: 0.5028
 7104/25000 [=======>......................] - ETA: 46s - loss: 7.6256 - accuracy: 0.5027
 7136/25000 [=======>......................] - ETA: 46s - loss: 7.6193 - accuracy: 0.5031
 7168/25000 [=======>......................] - ETA: 46s - loss: 7.6238 - accuracy: 0.5028
 7200/25000 [=======>......................] - ETA: 46s - loss: 7.6219 - accuracy: 0.5029
 7232/25000 [=======>......................] - ETA: 46s - loss: 7.6306 - accuracy: 0.5024
 7264/25000 [=======>......................] - ETA: 46s - loss: 7.6286 - accuracy: 0.5025
 7296/25000 [=======>......................] - ETA: 46s - loss: 7.6372 - accuracy: 0.5019
 7328/25000 [=======>......................] - ETA: 46s - loss: 7.6394 - accuracy: 0.5018
 7360/25000 [=======>......................] - ETA: 45s - loss: 7.6437 - accuracy: 0.5015
 7392/25000 [=======>......................] - ETA: 45s - loss: 7.6459 - accuracy: 0.5014
 7424/25000 [=======>......................] - ETA: 45s - loss: 7.6522 - accuracy: 0.5009
 7456/25000 [=======>......................] - ETA: 45s - loss: 7.6440 - accuracy: 0.5015
 7488/25000 [=======>......................] - ETA: 45s - loss: 7.6441 - accuracy: 0.5015
 7520/25000 [========>.....................] - ETA: 45s - loss: 7.6360 - accuracy: 0.5020
 7552/25000 [========>.....................] - ETA: 45s - loss: 7.6321 - accuracy: 0.5023
 7584/25000 [========>.....................] - ETA: 45s - loss: 7.6302 - accuracy: 0.5024
 7616/25000 [========>.....................] - ETA: 45s - loss: 7.6304 - accuracy: 0.5024
 7648/25000 [========>.....................] - ETA: 45s - loss: 7.6365 - accuracy: 0.5020
 7680/25000 [========>.....................] - ETA: 45s - loss: 7.6407 - accuracy: 0.5017
 7712/25000 [========>.....................] - ETA: 45s - loss: 7.6388 - accuracy: 0.5018
 7744/25000 [========>.....................] - ETA: 44s - loss: 7.6429 - accuracy: 0.5015
 7776/25000 [========>.....................] - ETA: 44s - loss: 7.6390 - accuracy: 0.5018
 7808/25000 [========>.....................] - ETA: 44s - loss: 7.6431 - accuracy: 0.5015
 7840/25000 [========>.....................] - ETA: 44s - loss: 7.6412 - accuracy: 0.5017
 7872/25000 [========>.....................] - ETA: 44s - loss: 7.6413 - accuracy: 0.5017
 7904/25000 [========>.....................] - ETA: 44s - loss: 7.6298 - accuracy: 0.5024
 7936/25000 [========>.....................] - ETA: 44s - loss: 7.6318 - accuracy: 0.5023
 7968/25000 [========>.....................] - ETA: 44s - loss: 7.6339 - accuracy: 0.5021
 8000/25000 [========>.....................] - ETA: 44s - loss: 7.6340 - accuracy: 0.5021
 8032/25000 [========>.....................] - ETA: 44s - loss: 7.6342 - accuracy: 0.5021
 8064/25000 [========>.....................] - ETA: 44s - loss: 7.6324 - accuracy: 0.5022
 8096/25000 [========>.....................] - ETA: 44s - loss: 7.6325 - accuracy: 0.5022
 8128/25000 [========>.....................] - ETA: 43s - loss: 7.6364 - accuracy: 0.5020
 8160/25000 [========>.....................] - ETA: 43s - loss: 7.6253 - accuracy: 0.5027
 8192/25000 [========>.....................] - ETA: 43s - loss: 7.6292 - accuracy: 0.5024
 8224/25000 [========>.....................] - ETA: 43s - loss: 7.6387 - accuracy: 0.5018
 8256/25000 [========>.....................] - ETA: 43s - loss: 7.6425 - accuracy: 0.5016
 8288/25000 [========>.....................] - ETA: 43s - loss: 7.6352 - accuracy: 0.5021
 8320/25000 [========>.....................] - ETA: 43s - loss: 7.6334 - accuracy: 0.5022
 8352/25000 [=========>....................] - ETA: 43s - loss: 7.6317 - accuracy: 0.5023
 8384/25000 [=========>....................] - ETA: 43s - loss: 7.6300 - accuracy: 0.5024
 8416/25000 [=========>....................] - ETA: 43s - loss: 7.6320 - accuracy: 0.5023
 8448/25000 [=========>....................] - ETA: 43s - loss: 7.6448 - accuracy: 0.5014
 8480/25000 [=========>....................] - ETA: 42s - loss: 7.6503 - accuracy: 0.5011
 8512/25000 [=========>....................] - ETA: 42s - loss: 7.6522 - accuracy: 0.5009
 8544/25000 [=========>....................] - ETA: 42s - loss: 7.6541 - accuracy: 0.5008
 8576/25000 [=========>....................] - ETA: 42s - loss: 7.6666 - accuracy: 0.5000
 8608/25000 [=========>....................] - ETA: 42s - loss: 7.6720 - accuracy: 0.4997
 8640/25000 [=========>....................] - ETA: 42s - loss: 7.6737 - accuracy: 0.4995
 8672/25000 [=========>....................] - ETA: 42s - loss: 7.6772 - accuracy: 0.4993
 8704/25000 [=========>....................] - ETA: 42s - loss: 7.6790 - accuracy: 0.4992
 8736/25000 [=========>....................] - ETA: 42s - loss: 7.6772 - accuracy: 0.4993
 8768/25000 [=========>....................] - ETA: 42s - loss: 7.6684 - accuracy: 0.4999
 8800/25000 [=========>....................] - ETA: 42s - loss: 7.6666 - accuracy: 0.5000
 8832/25000 [=========>....................] - ETA: 41s - loss: 7.6684 - accuracy: 0.4999
 8864/25000 [=========>....................] - ETA: 41s - loss: 7.6666 - accuracy: 0.5000
 8896/25000 [=========>....................] - ETA: 41s - loss: 7.6735 - accuracy: 0.4996
 8928/25000 [=========>....................] - ETA: 41s - loss: 7.6752 - accuracy: 0.4994
 8960/25000 [=========>....................] - ETA: 41s - loss: 7.6683 - accuracy: 0.4999
 8992/25000 [=========>....................] - ETA: 41s - loss: 7.6683 - accuracy: 0.4999
 9024/25000 [=========>....................] - ETA: 41s - loss: 7.6751 - accuracy: 0.4994
 9056/25000 [=========>....................] - ETA: 41s - loss: 7.6819 - accuracy: 0.4990
 9088/25000 [=========>....................] - ETA: 41s - loss: 7.6902 - accuracy: 0.4985
 9120/25000 [=========>....................] - ETA: 41s - loss: 7.6818 - accuracy: 0.4990
 9152/25000 [=========>....................] - ETA: 41s - loss: 7.6767 - accuracy: 0.4993
 9184/25000 [==========>...................] - ETA: 41s - loss: 7.6783 - accuracy: 0.4992
 9216/25000 [==========>...................] - ETA: 40s - loss: 7.6799 - accuracy: 0.4991
 9248/25000 [==========>...................] - ETA: 40s - loss: 7.6782 - accuracy: 0.4992
 9280/25000 [==========>...................] - ETA: 40s - loss: 7.6815 - accuracy: 0.4990
 9312/25000 [==========>...................] - ETA: 40s - loss: 7.6831 - accuracy: 0.4989
 9344/25000 [==========>...................] - ETA: 40s - loss: 7.6830 - accuracy: 0.4989
 9376/25000 [==========>...................] - ETA: 40s - loss: 7.6879 - accuracy: 0.4986
 9408/25000 [==========>...................] - ETA: 40s - loss: 7.6943 - accuracy: 0.4982
 9440/25000 [==========>...................] - ETA: 40s - loss: 7.6894 - accuracy: 0.4985
 9472/25000 [==========>...................] - ETA: 40s - loss: 7.6909 - accuracy: 0.4984
 9504/25000 [==========>...................] - ETA: 40s - loss: 7.6892 - accuracy: 0.4985
 9536/25000 [==========>...................] - ETA: 40s - loss: 7.6940 - accuracy: 0.4982
 9568/25000 [==========>...................] - ETA: 40s - loss: 7.6939 - accuracy: 0.4982
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.6890 - accuracy: 0.4985
 9632/25000 [==========>...................] - ETA: 39s - loss: 7.6905 - accuracy: 0.4984
 9664/25000 [==========>...................] - ETA: 39s - loss: 7.6984 - accuracy: 0.4979
 9696/25000 [==========>...................] - ETA: 39s - loss: 7.6998 - accuracy: 0.4978
 9728/25000 [==========>...................] - ETA: 39s - loss: 7.6966 - accuracy: 0.4980
 9760/25000 [==========>...................] - ETA: 39s - loss: 7.6949 - accuracy: 0.4982
 9792/25000 [==========>...................] - ETA: 39s - loss: 7.6932 - accuracy: 0.4983
 9824/25000 [==========>...................] - ETA: 39s - loss: 7.6994 - accuracy: 0.4979
 9856/25000 [==========>...................] - ETA: 39s - loss: 7.6962 - accuracy: 0.4981
 9888/25000 [==========>...................] - ETA: 39s - loss: 7.6883 - accuracy: 0.4986
 9920/25000 [==========>...................] - ETA: 39s - loss: 7.6944 - accuracy: 0.4982
 9952/25000 [==========>...................] - ETA: 39s - loss: 7.6974 - accuracy: 0.4980
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.6958 - accuracy: 0.4981
10016/25000 [===========>..................] - ETA: 38s - loss: 7.6896 - accuracy: 0.4985
10048/25000 [===========>..................] - ETA: 38s - loss: 7.6941 - accuracy: 0.4982
10080/25000 [===========>..................] - ETA: 38s - loss: 7.6970 - accuracy: 0.4980
10112/25000 [===========>..................] - ETA: 38s - loss: 7.6985 - accuracy: 0.4979
10144/25000 [===========>..................] - ETA: 38s - loss: 7.6969 - accuracy: 0.4980
10176/25000 [===========>..................] - ETA: 38s - loss: 7.6983 - accuracy: 0.4979
10208/25000 [===========>..................] - ETA: 38s - loss: 7.6907 - accuracy: 0.4984
10240/25000 [===========>..................] - ETA: 38s - loss: 7.6876 - accuracy: 0.4986
10272/25000 [===========>..................] - ETA: 38s - loss: 7.6935 - accuracy: 0.4982
10304/25000 [===========>..................] - ETA: 38s - loss: 7.6934 - accuracy: 0.4983
10336/25000 [===========>..................] - ETA: 38s - loss: 7.6889 - accuracy: 0.4985
10368/25000 [===========>..................] - ETA: 37s - loss: 7.6977 - accuracy: 0.4980
10400/25000 [===========>..................] - ETA: 37s - loss: 7.6961 - accuracy: 0.4981
10432/25000 [===========>..................] - ETA: 37s - loss: 7.7004 - accuracy: 0.4978
10464/25000 [===========>..................] - ETA: 37s - loss: 7.6945 - accuracy: 0.4982
10496/25000 [===========>..................] - ETA: 37s - loss: 7.6929 - accuracy: 0.4983
10528/25000 [===========>..................] - ETA: 37s - loss: 7.6899 - accuracy: 0.4985
10560/25000 [===========>..................] - ETA: 37s - loss: 7.6942 - accuracy: 0.4982
10592/25000 [===========>..................] - ETA: 37s - loss: 7.6970 - accuracy: 0.4980
10624/25000 [===========>..................] - ETA: 37s - loss: 7.6912 - accuracy: 0.4984
10656/25000 [===========>..................] - ETA: 37s - loss: 7.6882 - accuracy: 0.4986
10688/25000 [===========>..................] - ETA: 37s - loss: 7.6867 - accuracy: 0.4987
10720/25000 [===========>..................] - ETA: 37s - loss: 7.6852 - accuracy: 0.4988
10752/25000 [===========>..................] - ETA: 36s - loss: 7.6823 - accuracy: 0.4990
10784/25000 [===========>..................] - ETA: 36s - loss: 7.6794 - accuracy: 0.4992
10816/25000 [===========>..................] - ETA: 36s - loss: 7.6723 - accuracy: 0.4996
10848/25000 [============>.................] - ETA: 36s - loss: 7.6709 - accuracy: 0.4997
10880/25000 [============>.................] - ETA: 36s - loss: 7.6737 - accuracy: 0.4995
10912/25000 [============>.................] - ETA: 36s - loss: 7.6765 - accuracy: 0.4994
10944/25000 [============>.................] - ETA: 36s - loss: 7.6736 - accuracy: 0.4995
10976/25000 [============>.................] - ETA: 36s - loss: 7.6680 - accuracy: 0.4999
11008/25000 [============>.................] - ETA: 36s - loss: 7.6708 - accuracy: 0.4997
11040/25000 [============>.................] - ETA: 36s - loss: 7.6666 - accuracy: 0.5000
11072/25000 [============>.................] - ETA: 36s - loss: 7.6694 - accuracy: 0.4998
11104/25000 [============>.................] - ETA: 36s - loss: 7.6749 - accuracy: 0.4995
11136/25000 [============>.................] - ETA: 35s - loss: 7.6763 - accuracy: 0.4994
11168/25000 [============>.................] - ETA: 35s - loss: 7.6721 - accuracy: 0.4996
11200/25000 [============>.................] - ETA: 35s - loss: 7.6694 - accuracy: 0.4998
11232/25000 [============>.................] - ETA: 35s - loss: 7.6707 - accuracy: 0.4997
11264/25000 [============>.................] - ETA: 35s - loss: 7.6734 - accuracy: 0.4996
11296/25000 [============>.................] - ETA: 35s - loss: 7.6666 - accuracy: 0.5000
11328/25000 [============>.................] - ETA: 35s - loss: 7.6693 - accuracy: 0.4998
11360/25000 [============>.................] - ETA: 35s - loss: 7.6801 - accuracy: 0.4991
11392/25000 [============>.................] - ETA: 35s - loss: 7.6787 - accuracy: 0.4992
11424/25000 [============>.................] - ETA: 35s - loss: 7.6747 - accuracy: 0.4995
11456/25000 [============>.................] - ETA: 35s - loss: 7.6787 - accuracy: 0.4992
11488/25000 [============>.................] - ETA: 35s - loss: 7.6840 - accuracy: 0.4989
11520/25000 [============>.................] - ETA: 34s - loss: 7.6853 - accuracy: 0.4988
11552/25000 [============>.................] - ETA: 34s - loss: 7.6879 - accuracy: 0.4986
11584/25000 [============>.................] - ETA: 34s - loss: 7.6852 - accuracy: 0.4988
11616/25000 [============>.................] - ETA: 34s - loss: 7.6825 - accuracy: 0.4990
11648/25000 [============>.................] - ETA: 34s - loss: 7.6837 - accuracy: 0.4989
11680/25000 [=============>................] - ETA: 34s - loss: 7.6863 - accuracy: 0.4987
11712/25000 [=============>................] - ETA: 34s - loss: 7.6784 - accuracy: 0.4992
11744/25000 [=============>................] - ETA: 34s - loss: 7.6758 - accuracy: 0.4994
11776/25000 [=============>................] - ETA: 34s - loss: 7.6822 - accuracy: 0.4990
11808/25000 [=============>................] - ETA: 34s - loss: 7.6796 - accuracy: 0.4992
11840/25000 [=============>................] - ETA: 34s - loss: 7.6822 - accuracy: 0.4990
11872/25000 [=============>................] - ETA: 34s - loss: 7.6782 - accuracy: 0.4992
11904/25000 [=============>................] - ETA: 33s - loss: 7.6847 - accuracy: 0.4988
11936/25000 [=============>................] - ETA: 33s - loss: 7.6923 - accuracy: 0.4983
11968/25000 [=============>................] - ETA: 33s - loss: 7.6910 - accuracy: 0.4984
12000/25000 [=============>................] - ETA: 33s - loss: 7.6909 - accuracy: 0.4984
12032/25000 [=============>................] - ETA: 33s - loss: 7.6857 - accuracy: 0.4988
12064/25000 [=============>................] - ETA: 33s - loss: 7.6831 - accuracy: 0.4989
12096/25000 [=============>................] - ETA: 33s - loss: 7.6856 - accuracy: 0.4988
12128/25000 [=============>................] - ETA: 33s - loss: 7.6868 - accuracy: 0.4987
12160/25000 [=============>................] - ETA: 33s - loss: 7.6906 - accuracy: 0.4984
12192/25000 [=============>................] - ETA: 33s - loss: 7.6830 - accuracy: 0.4989
12224/25000 [=============>................] - ETA: 33s - loss: 7.6879 - accuracy: 0.4986
12256/25000 [=============>................] - ETA: 33s - loss: 7.6954 - accuracy: 0.4981
12288/25000 [=============>................] - ETA: 32s - loss: 7.7016 - accuracy: 0.4977
12320/25000 [=============>................] - ETA: 32s - loss: 7.7002 - accuracy: 0.4978
12352/25000 [=============>................] - ETA: 32s - loss: 7.7051 - accuracy: 0.4975
12384/25000 [=============>................] - ETA: 32s - loss: 7.7087 - accuracy: 0.4973
12416/25000 [=============>................] - ETA: 32s - loss: 7.7098 - accuracy: 0.4972
12448/25000 [=============>................] - ETA: 32s - loss: 7.7060 - accuracy: 0.4974
12480/25000 [=============>................] - ETA: 32s - loss: 7.7084 - accuracy: 0.4973
12512/25000 [==============>...............] - ETA: 32s - loss: 7.7071 - accuracy: 0.4974
12544/25000 [==============>...............] - ETA: 32s - loss: 7.7082 - accuracy: 0.4973
12576/25000 [==============>...............] - ETA: 32s - loss: 7.7105 - accuracy: 0.4971
12608/25000 [==============>...............] - ETA: 32s - loss: 7.7128 - accuracy: 0.4970
12640/25000 [==============>...............] - ETA: 32s - loss: 7.7164 - accuracy: 0.4968
12672/25000 [==============>...............] - ETA: 31s - loss: 7.7186 - accuracy: 0.4966
12704/25000 [==============>...............] - ETA: 31s - loss: 7.7209 - accuracy: 0.4965
12736/25000 [==============>...............] - ETA: 31s - loss: 7.7268 - accuracy: 0.4961
12768/25000 [==============>...............] - ETA: 31s - loss: 7.7243 - accuracy: 0.4962
12800/25000 [==============>...............] - ETA: 31s - loss: 7.7289 - accuracy: 0.4959
12832/25000 [==============>...............] - ETA: 31s - loss: 7.7300 - accuracy: 0.4959
12864/25000 [==============>...............] - ETA: 31s - loss: 7.7250 - accuracy: 0.4962
12896/25000 [==============>...............] - ETA: 31s - loss: 7.7284 - accuracy: 0.4960
12928/25000 [==============>...............] - ETA: 31s - loss: 7.7259 - accuracy: 0.4961
12960/25000 [==============>...............] - ETA: 31s - loss: 7.7246 - accuracy: 0.4962
12992/25000 [==============>...............] - ETA: 31s - loss: 7.7280 - accuracy: 0.4960
13024/25000 [==============>...............] - ETA: 31s - loss: 7.7267 - accuracy: 0.4961
13056/25000 [==============>...............] - ETA: 30s - loss: 7.7289 - accuracy: 0.4959
13088/25000 [==============>...............] - ETA: 30s - loss: 7.7287 - accuracy: 0.4960
13120/25000 [==============>...............] - ETA: 30s - loss: 7.7297 - accuracy: 0.4959
13152/25000 [==============>...............] - ETA: 30s - loss: 7.7354 - accuracy: 0.4955
13184/25000 [==============>...............] - ETA: 30s - loss: 7.7341 - accuracy: 0.4956
13216/25000 [==============>...............] - ETA: 30s - loss: 7.7293 - accuracy: 0.4959
13248/25000 [==============>...............] - ETA: 30s - loss: 7.7268 - accuracy: 0.4961
13280/25000 [==============>...............] - ETA: 30s - loss: 7.7278 - accuracy: 0.4960
13312/25000 [==============>...............] - ETA: 30s - loss: 7.7219 - accuracy: 0.4964
13344/25000 [===============>..............] - ETA: 30s - loss: 7.7218 - accuracy: 0.4964
13376/25000 [===============>..............] - ETA: 30s - loss: 7.7216 - accuracy: 0.4964
13408/25000 [===============>..............] - ETA: 30s - loss: 7.7227 - accuracy: 0.4963
13440/25000 [===============>..............] - ETA: 29s - loss: 7.7259 - accuracy: 0.4961
13472/25000 [===============>..............] - ETA: 29s - loss: 7.7213 - accuracy: 0.4964
13504/25000 [===============>..............] - ETA: 29s - loss: 7.7223 - accuracy: 0.4964
13536/25000 [===============>..............] - ETA: 29s - loss: 7.7165 - accuracy: 0.4967
13568/25000 [===============>..............] - ETA: 29s - loss: 7.7163 - accuracy: 0.4968
13600/25000 [===============>..............] - ETA: 29s - loss: 7.7196 - accuracy: 0.4965
13632/25000 [===============>..............] - ETA: 29s - loss: 7.7206 - accuracy: 0.4965
13664/25000 [===============>..............] - ETA: 29s - loss: 7.7160 - accuracy: 0.4968
13696/25000 [===============>..............] - ETA: 29s - loss: 7.7080 - accuracy: 0.4973
13728/25000 [===============>..............] - ETA: 29s - loss: 7.7091 - accuracy: 0.4972
13760/25000 [===============>..............] - ETA: 29s - loss: 7.7101 - accuracy: 0.4972
13792/25000 [===============>..............] - ETA: 29s - loss: 7.7144 - accuracy: 0.4969
13824/25000 [===============>..............] - ETA: 28s - loss: 7.7132 - accuracy: 0.4970
13856/25000 [===============>..............] - ETA: 28s - loss: 7.7131 - accuracy: 0.4970
13888/25000 [===============>..............] - ETA: 28s - loss: 7.7119 - accuracy: 0.4970
13920/25000 [===============>..............] - ETA: 28s - loss: 7.7151 - accuracy: 0.4968
13952/25000 [===============>..............] - ETA: 28s - loss: 7.7194 - accuracy: 0.4966
13984/25000 [===============>..............] - ETA: 28s - loss: 7.7127 - accuracy: 0.4970
14016/25000 [===============>..............] - ETA: 28s - loss: 7.7126 - accuracy: 0.4970
14048/25000 [===============>..............] - ETA: 28s - loss: 7.7103 - accuracy: 0.4972
14080/25000 [===============>..............] - ETA: 28s - loss: 7.7080 - accuracy: 0.4973
14112/25000 [===============>..............] - ETA: 28s - loss: 7.7046 - accuracy: 0.4975
14144/25000 [===============>..............] - ETA: 28s - loss: 7.7111 - accuracy: 0.4971
14176/25000 [================>.............] - ETA: 28s - loss: 7.7142 - accuracy: 0.4969
14208/25000 [================>.............] - ETA: 27s - loss: 7.7173 - accuracy: 0.4967
14240/25000 [================>.............] - ETA: 27s - loss: 7.7151 - accuracy: 0.4968
14272/25000 [================>.............] - ETA: 27s - loss: 7.7139 - accuracy: 0.4969
14304/25000 [================>.............] - ETA: 27s - loss: 7.7138 - accuracy: 0.4969
14336/25000 [================>.............] - ETA: 27s - loss: 7.7147 - accuracy: 0.4969
14368/25000 [================>.............] - ETA: 27s - loss: 7.7189 - accuracy: 0.4966
14400/25000 [================>.............] - ETA: 27s - loss: 7.7199 - accuracy: 0.4965
14432/25000 [================>.............] - ETA: 27s - loss: 7.7229 - accuracy: 0.4963
14464/25000 [================>.............] - ETA: 27s - loss: 7.7249 - accuracy: 0.4962
14496/25000 [================>.............] - ETA: 27s - loss: 7.7259 - accuracy: 0.4961
14528/25000 [================>.............] - ETA: 27s - loss: 7.7268 - accuracy: 0.4961
14560/25000 [================>.............] - ETA: 27s - loss: 7.7288 - accuracy: 0.4959
14592/25000 [================>.............] - ETA: 26s - loss: 7.7318 - accuracy: 0.4958
14624/25000 [================>.............] - ETA: 26s - loss: 7.7337 - accuracy: 0.4956
14656/25000 [================>.............] - ETA: 26s - loss: 7.7294 - accuracy: 0.4959
14688/25000 [================>.............] - ETA: 26s - loss: 7.7345 - accuracy: 0.4956
14720/25000 [================>.............] - ETA: 26s - loss: 7.7291 - accuracy: 0.4959
14752/25000 [================>.............] - ETA: 26s - loss: 7.7311 - accuracy: 0.4958
14784/25000 [================>.............] - ETA: 26s - loss: 7.7288 - accuracy: 0.4959
14816/25000 [================>.............] - ETA: 26s - loss: 7.7277 - accuracy: 0.4960
14848/25000 [================>.............] - ETA: 26s - loss: 7.7296 - accuracy: 0.4959
14880/25000 [================>.............] - ETA: 26s - loss: 7.7305 - accuracy: 0.4958
14912/25000 [================>.............] - ETA: 26s - loss: 7.7273 - accuracy: 0.4960
14944/25000 [================>.............] - ETA: 26s - loss: 7.7292 - accuracy: 0.4959
14976/25000 [================>.............] - ETA: 25s - loss: 7.7291 - accuracy: 0.4959
15008/25000 [=================>............] - ETA: 25s - loss: 7.7351 - accuracy: 0.4955
15040/25000 [=================>............] - ETA: 25s - loss: 7.7359 - accuracy: 0.4955
15072/25000 [=================>............] - ETA: 25s - loss: 7.7389 - accuracy: 0.4953
15104/25000 [=================>............] - ETA: 25s - loss: 7.7417 - accuracy: 0.4951
15136/25000 [=================>............] - ETA: 25s - loss: 7.7416 - accuracy: 0.4951
15168/25000 [=================>............] - ETA: 25s - loss: 7.7414 - accuracy: 0.4951
15200/25000 [=================>............] - ETA: 25s - loss: 7.7372 - accuracy: 0.4954
15232/25000 [=================>............] - ETA: 25s - loss: 7.7371 - accuracy: 0.4954
15264/25000 [=================>............] - ETA: 25s - loss: 7.7379 - accuracy: 0.4953
15296/25000 [=================>............] - ETA: 25s - loss: 7.7378 - accuracy: 0.4954
15328/25000 [=================>............] - ETA: 25s - loss: 7.7336 - accuracy: 0.4956
15360/25000 [=================>............] - ETA: 24s - loss: 7.7315 - accuracy: 0.4958
15392/25000 [=================>............] - ETA: 24s - loss: 7.7314 - accuracy: 0.4958
15424/25000 [=================>............] - ETA: 24s - loss: 7.7302 - accuracy: 0.4959
15456/25000 [=================>............] - ETA: 24s - loss: 7.7380 - accuracy: 0.4953
15488/25000 [=================>............] - ETA: 24s - loss: 7.7379 - accuracy: 0.4954
15520/25000 [=================>............] - ETA: 24s - loss: 7.7387 - accuracy: 0.4953
15552/25000 [=================>............] - ETA: 24s - loss: 7.7425 - accuracy: 0.4950
15584/25000 [=================>............] - ETA: 24s - loss: 7.7404 - accuracy: 0.4952
15616/25000 [=================>............] - ETA: 24s - loss: 7.7412 - accuracy: 0.4951
15648/25000 [=================>............] - ETA: 24s - loss: 7.7411 - accuracy: 0.4951
15680/25000 [=================>............] - ETA: 24s - loss: 7.7409 - accuracy: 0.4952
15712/25000 [=================>............] - ETA: 24s - loss: 7.7427 - accuracy: 0.4950
15744/25000 [=================>............] - ETA: 23s - loss: 7.7465 - accuracy: 0.4948
15776/25000 [=================>............] - ETA: 23s - loss: 7.7415 - accuracy: 0.4951
15808/25000 [=================>............] - ETA: 23s - loss: 7.7423 - accuracy: 0.4951
15840/25000 [==================>...........] - ETA: 23s - loss: 7.7421 - accuracy: 0.4951
15872/25000 [==================>...........] - ETA: 23s - loss: 7.7420 - accuracy: 0.4951
15904/25000 [==================>...........] - ETA: 23s - loss: 7.7428 - accuracy: 0.4950
15936/25000 [==================>...........] - ETA: 23s - loss: 7.7417 - accuracy: 0.4951
15968/25000 [==================>...........] - ETA: 23s - loss: 7.7415 - accuracy: 0.4951
16000/25000 [==================>...........] - ETA: 23s - loss: 7.7452 - accuracy: 0.4949
16032/25000 [==================>...........] - ETA: 23s - loss: 7.7412 - accuracy: 0.4951
16064/25000 [==================>...........] - ETA: 23s - loss: 7.7401 - accuracy: 0.4952
16096/25000 [==================>...........] - ETA: 22s - loss: 7.7390 - accuracy: 0.4953
16128/25000 [==================>...........] - ETA: 22s - loss: 7.7370 - accuracy: 0.4954
16160/25000 [==================>...........] - ETA: 22s - loss: 7.7387 - accuracy: 0.4953
16192/25000 [==================>...........] - ETA: 22s - loss: 7.7357 - accuracy: 0.4955
16224/25000 [==================>...........] - ETA: 22s - loss: 7.7375 - accuracy: 0.4954
16256/25000 [==================>...........] - ETA: 22s - loss: 7.7336 - accuracy: 0.4956
16288/25000 [==================>...........] - ETA: 22s - loss: 7.7391 - accuracy: 0.4953
16320/25000 [==================>...........] - ETA: 22s - loss: 7.7361 - accuracy: 0.4955
16352/25000 [==================>...........] - ETA: 22s - loss: 7.7398 - accuracy: 0.4952
16384/25000 [==================>...........] - ETA: 22s - loss: 7.7349 - accuracy: 0.4955
16416/25000 [==================>...........] - ETA: 22s - loss: 7.7339 - accuracy: 0.4956
16448/25000 [==================>...........] - ETA: 22s - loss: 7.7319 - accuracy: 0.4957
16480/25000 [==================>...........] - ETA: 21s - loss: 7.7290 - accuracy: 0.4959
16512/25000 [==================>...........] - ETA: 21s - loss: 7.7270 - accuracy: 0.4961
16544/25000 [==================>...........] - ETA: 21s - loss: 7.7287 - accuracy: 0.4960
16576/25000 [==================>...........] - ETA: 21s - loss: 7.7267 - accuracy: 0.4961
16608/25000 [==================>...........] - ETA: 21s - loss: 7.7276 - accuracy: 0.4960
16640/25000 [==================>...........] - ETA: 21s - loss: 7.7274 - accuracy: 0.4960
16672/25000 [===================>..........] - ETA: 21s - loss: 7.7282 - accuracy: 0.4960
16704/25000 [===================>..........] - ETA: 21s - loss: 7.7309 - accuracy: 0.4958
16736/25000 [===================>..........] - ETA: 21s - loss: 7.7326 - accuracy: 0.4957
16768/25000 [===================>..........] - ETA: 21s - loss: 7.7270 - accuracy: 0.4961
16800/25000 [===================>..........] - ETA: 21s - loss: 7.7223 - accuracy: 0.4964
16832/25000 [===================>..........] - ETA: 21s - loss: 7.7204 - accuracy: 0.4965
16864/25000 [===================>..........] - ETA: 20s - loss: 7.7230 - accuracy: 0.4963
16896/25000 [===================>..........] - ETA: 20s - loss: 7.7238 - accuracy: 0.4963
16928/25000 [===================>..........] - ETA: 20s - loss: 7.7246 - accuracy: 0.4962
16960/25000 [===================>..........] - ETA: 20s - loss: 7.7245 - accuracy: 0.4962
16992/25000 [===================>..........] - ETA: 20s - loss: 7.7208 - accuracy: 0.4965
17024/25000 [===================>..........] - ETA: 20s - loss: 7.7171 - accuracy: 0.4967
17056/25000 [===================>..........] - ETA: 20s - loss: 7.7125 - accuracy: 0.4970
17088/25000 [===================>..........] - ETA: 20s - loss: 7.7151 - accuracy: 0.4968
17120/25000 [===================>..........] - ETA: 20s - loss: 7.7177 - accuracy: 0.4967
17152/25000 [===================>..........] - ETA: 20s - loss: 7.7158 - accuracy: 0.4968
17184/25000 [===================>..........] - ETA: 20s - loss: 7.7175 - accuracy: 0.4967
17216/25000 [===================>..........] - ETA: 20s - loss: 7.7129 - accuracy: 0.4970
17248/25000 [===================>..........] - ETA: 19s - loss: 7.7155 - accuracy: 0.4968
17280/25000 [===================>..........] - ETA: 19s - loss: 7.7190 - accuracy: 0.4966
17312/25000 [===================>..........] - ETA: 19s - loss: 7.7224 - accuracy: 0.4964
17344/25000 [===================>..........] - ETA: 19s - loss: 7.7179 - accuracy: 0.4967
17376/25000 [===================>..........] - ETA: 19s - loss: 7.7160 - accuracy: 0.4968
17408/25000 [===================>..........] - ETA: 19s - loss: 7.7124 - accuracy: 0.4970
17440/25000 [===================>..........] - ETA: 19s - loss: 7.7132 - accuracy: 0.4970
17472/25000 [===================>..........] - ETA: 19s - loss: 7.7149 - accuracy: 0.4969
17504/25000 [====================>.........] - ETA: 19s - loss: 7.7122 - accuracy: 0.4970
17536/25000 [====================>.........] - ETA: 19s - loss: 7.7103 - accuracy: 0.4971
17568/25000 [====================>.........] - ETA: 19s - loss: 7.7155 - accuracy: 0.4968
17600/25000 [====================>.........] - ETA: 19s - loss: 7.7137 - accuracy: 0.4969
17632/25000 [====================>.........] - ETA: 18s - loss: 7.7153 - accuracy: 0.4968
17664/25000 [====================>.........] - ETA: 18s - loss: 7.7170 - accuracy: 0.4967
17696/25000 [====================>.........] - ETA: 18s - loss: 7.7151 - accuracy: 0.4968
17728/25000 [====================>.........] - ETA: 18s - loss: 7.7185 - accuracy: 0.4966
17760/25000 [====================>.........] - ETA: 18s - loss: 7.7193 - accuracy: 0.4966
17792/25000 [====================>.........] - ETA: 18s - loss: 7.7226 - accuracy: 0.4963
17824/25000 [====================>.........] - ETA: 18s - loss: 7.7234 - accuracy: 0.4963
17856/25000 [====================>.........] - ETA: 18s - loss: 7.7233 - accuracy: 0.4963
17888/25000 [====================>.........] - ETA: 18s - loss: 7.7241 - accuracy: 0.4963
17920/25000 [====================>.........] - ETA: 18s - loss: 7.7231 - accuracy: 0.4963
17952/25000 [====================>.........] - ETA: 18s - loss: 7.7247 - accuracy: 0.4962
17984/25000 [====================>.........] - ETA: 18s - loss: 7.7254 - accuracy: 0.4962
18016/25000 [====================>.........] - ETA: 17s - loss: 7.7236 - accuracy: 0.4963
18048/25000 [====================>.........] - ETA: 17s - loss: 7.7235 - accuracy: 0.4963
18080/25000 [====================>.........] - ETA: 17s - loss: 7.7192 - accuracy: 0.4966
18112/25000 [====================>.........] - ETA: 17s - loss: 7.7191 - accuracy: 0.4966
18144/25000 [====================>.........] - ETA: 17s - loss: 7.7190 - accuracy: 0.4966
18176/25000 [====================>.........] - ETA: 17s - loss: 7.7198 - accuracy: 0.4965
18208/25000 [====================>.........] - ETA: 17s - loss: 7.7163 - accuracy: 0.4968
18240/25000 [====================>.........] - ETA: 17s - loss: 7.7137 - accuracy: 0.4969
18272/25000 [====================>.........] - ETA: 17s - loss: 7.7128 - accuracy: 0.4970
18304/25000 [====================>.........] - ETA: 17s - loss: 7.7127 - accuracy: 0.4970
18336/25000 [=====================>........] - ETA: 17s - loss: 7.7101 - accuracy: 0.4972
18368/25000 [=====================>........] - ETA: 17s - loss: 7.7100 - accuracy: 0.4972
18400/25000 [=====================>........] - ETA: 16s - loss: 7.7108 - accuracy: 0.4971
18432/25000 [=====================>........] - ETA: 16s - loss: 7.7132 - accuracy: 0.4970
18464/25000 [=====================>........] - ETA: 16s - loss: 7.7156 - accuracy: 0.4968
18496/25000 [=====================>........] - ETA: 16s - loss: 7.7130 - accuracy: 0.4970
18528/25000 [=====================>........] - ETA: 16s - loss: 7.7097 - accuracy: 0.4972
18560/25000 [=====================>........] - ETA: 16s - loss: 7.7071 - accuracy: 0.4974
18592/25000 [=====================>........] - ETA: 16s - loss: 7.7062 - accuracy: 0.4974
18624/25000 [=====================>........] - ETA: 16s - loss: 7.7078 - accuracy: 0.4973
18656/25000 [=====================>........] - ETA: 16s - loss: 7.7052 - accuracy: 0.4975
18688/25000 [=====================>........] - ETA: 16s - loss: 7.7027 - accuracy: 0.4976
18720/25000 [=====================>........] - ETA: 16s - loss: 7.7043 - accuracy: 0.4975
18752/25000 [=====================>........] - ETA: 16s - loss: 7.7100 - accuracy: 0.4972
18784/25000 [=====================>........] - ETA: 15s - loss: 7.7074 - accuracy: 0.4973
18816/25000 [=====================>........] - ETA: 15s - loss: 7.7057 - accuracy: 0.4974
18848/25000 [=====================>........] - ETA: 15s - loss: 7.7040 - accuracy: 0.4976
18880/25000 [=====================>........] - ETA: 15s - loss: 7.7064 - accuracy: 0.4974
18912/25000 [=====================>........] - ETA: 15s - loss: 7.7063 - accuracy: 0.4974
18944/25000 [=====================>........] - ETA: 15s - loss: 7.7095 - accuracy: 0.4972
18976/25000 [=====================>........] - ETA: 15s - loss: 7.7135 - accuracy: 0.4969
19008/25000 [=====================>........] - ETA: 15s - loss: 7.7134 - accuracy: 0.4969
19040/25000 [=====================>........] - ETA: 15s - loss: 7.7141 - accuracy: 0.4969
19072/25000 [=====================>........] - ETA: 15s - loss: 7.7124 - accuracy: 0.4970
19104/25000 [=====================>........] - ETA: 15s - loss: 7.7124 - accuracy: 0.4970
19136/25000 [=====================>........] - ETA: 15s - loss: 7.7131 - accuracy: 0.4970
19168/25000 [======================>.......] - ETA: 15s - loss: 7.7098 - accuracy: 0.4972
19200/25000 [======================>.......] - ETA: 14s - loss: 7.7113 - accuracy: 0.4971
19232/25000 [======================>.......] - ETA: 14s - loss: 7.7137 - accuracy: 0.4969
19264/25000 [======================>.......] - ETA: 14s - loss: 7.7152 - accuracy: 0.4968
19296/25000 [======================>.......] - ETA: 14s - loss: 7.7119 - accuracy: 0.4970
19328/25000 [======================>.......] - ETA: 14s - loss: 7.7126 - accuracy: 0.4970
19360/25000 [======================>.......] - ETA: 14s - loss: 7.7118 - accuracy: 0.4971
19392/25000 [======================>.......] - ETA: 14s - loss: 7.7093 - accuracy: 0.4972
19424/25000 [======================>.......] - ETA: 14s - loss: 7.7092 - accuracy: 0.4972
19456/25000 [======================>.......] - ETA: 14s - loss: 7.7092 - accuracy: 0.4972
19488/25000 [======================>.......] - ETA: 14s - loss: 7.7115 - accuracy: 0.4971
19520/25000 [======================>.......] - ETA: 14s - loss: 7.7067 - accuracy: 0.4974
19552/25000 [======================>.......] - ETA: 14s - loss: 7.7074 - accuracy: 0.4973
19584/25000 [======================>.......] - ETA: 13s - loss: 7.7081 - accuracy: 0.4973
19616/25000 [======================>.......] - ETA: 13s - loss: 7.7104 - accuracy: 0.4971
19648/25000 [======================>.......] - ETA: 13s - loss: 7.7088 - accuracy: 0.4973
19680/25000 [======================>.......] - ETA: 13s - loss: 7.7110 - accuracy: 0.4971
19712/25000 [======================>.......] - ETA: 13s - loss: 7.7125 - accuracy: 0.4970
19744/25000 [======================>.......] - ETA: 13s - loss: 7.7078 - accuracy: 0.4973
19776/25000 [======================>.......] - ETA: 13s - loss: 7.7093 - accuracy: 0.4972
19808/25000 [======================>.......] - ETA: 13s - loss: 7.7069 - accuracy: 0.4974
19840/25000 [======================>.......] - ETA: 13s - loss: 7.7060 - accuracy: 0.4974
19872/25000 [======================>.......] - ETA: 13s - loss: 7.7021 - accuracy: 0.4977
19904/25000 [======================>.......] - ETA: 13s - loss: 7.7005 - accuracy: 0.4978
19936/25000 [======================>.......] - ETA: 13s - loss: 7.7005 - accuracy: 0.4978
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6981 - accuracy: 0.4979
20000/25000 [=======================>......] - ETA: 12s - loss: 7.7034 - accuracy: 0.4976
20032/25000 [=======================>......] - ETA: 12s - loss: 7.7011 - accuracy: 0.4978
20064/25000 [=======================>......] - ETA: 12s - loss: 7.7002 - accuracy: 0.4978
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6979 - accuracy: 0.4980
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6940 - accuracy: 0.4982
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6932 - accuracy: 0.4983
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6940 - accuracy: 0.4982
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6969 - accuracy: 0.4980
20256/25000 [=======================>......] - ETA: 12s - loss: 7.6939 - accuracy: 0.4982
20288/25000 [=======================>......] - ETA: 12s - loss: 7.6923 - accuracy: 0.4983
20320/25000 [=======================>......] - ETA: 12s - loss: 7.6885 - accuracy: 0.4986
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6900 - accuracy: 0.4985
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6899 - accuracy: 0.4985
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6861 - accuracy: 0.4987
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6861 - accuracy: 0.4987
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6868 - accuracy: 0.4987
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6853 - accuracy: 0.4988
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6875 - accuracy: 0.4986
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6875 - accuracy: 0.4986
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6830 - accuracy: 0.4989
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6837 - accuracy: 0.4989
20672/25000 [=======================>......] - ETA: 11s - loss: 7.6837 - accuracy: 0.4989
20704/25000 [=======================>......] - ETA: 11s - loss: 7.6837 - accuracy: 0.4989
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6821 - accuracy: 0.4990
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6829 - accuracy: 0.4989
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6799 - accuracy: 0.4991
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6784 - accuracy: 0.4992
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6784 - accuracy: 0.4992
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6747 - accuracy: 0.4995
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6739 - accuracy: 0.4995
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6732 - accuracy: 0.4996
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6768 - accuracy: 0.4993
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6754 - accuracy: 0.4994
21056/25000 [========================>.....] - ETA: 10s - loss: 7.6732 - accuracy: 0.4996
21088/25000 [========================>.....] - ETA: 10s - loss: 7.6732 - accuracy: 0.4996
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6732 - accuracy: 0.4996 
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6746 - accuracy: 0.4995
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6739 - accuracy: 0.4995
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6724 - accuracy: 0.4996
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6702 - accuracy: 0.4998
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6724 - accuracy: 0.4996
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6695 - accuracy: 0.4998
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6702 - accuracy: 0.4998
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6724 - accuracy: 0.4996
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6723 - accuracy: 0.4996
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6716 - accuracy: 0.4997
21472/25000 [========================>.....] - ETA: 9s - loss: 7.6702 - accuracy: 0.4998
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6716 - accuracy: 0.4997
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6737 - accuracy: 0.4995
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6759 - accuracy: 0.4994
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6737 - accuracy: 0.4995
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6744 - accuracy: 0.4995
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6758 - accuracy: 0.4994
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6751 - accuracy: 0.4994
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6765 - accuracy: 0.4994
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6751 - accuracy: 0.4994
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6765 - accuracy: 0.4994
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6772 - accuracy: 0.4993
21856/25000 [=========================>....] - ETA: 8s - loss: 7.6750 - accuracy: 0.4995
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6757 - accuracy: 0.4994
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6764 - accuracy: 0.4994
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6743 - accuracy: 0.4995
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6743 - accuracy: 0.4995
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6708 - accuracy: 0.4997
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6729 - accuracy: 0.4996
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6701 - accuracy: 0.4998
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6680 - accuracy: 0.4999
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6708 - accuracy: 0.4997
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6722 - accuracy: 0.4996
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6735 - accuracy: 0.4995
22240/25000 [=========================>....] - ETA: 7s - loss: 7.6721 - accuracy: 0.4996
22272/25000 [=========================>....] - ETA: 7s - loss: 7.6756 - accuracy: 0.4994
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6742 - accuracy: 0.4995
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6769 - accuracy: 0.4993
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6783 - accuracy: 0.4992
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6783 - accuracy: 0.4992
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6769 - accuracy: 0.4993
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6769 - accuracy: 0.4993
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6762 - accuracy: 0.4994
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6775 - accuracy: 0.4993
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6802 - accuracy: 0.4991
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6829 - accuracy: 0.4989
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6795 - accuracy: 0.4992
22656/25000 [==========================>...] - ETA: 6s - loss: 7.6802 - accuracy: 0.4991
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6795 - accuracy: 0.4992
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6767 - accuracy: 0.4993
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6727 - accuracy: 0.4996
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6707 - accuracy: 0.4997
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6680 - accuracy: 0.4999
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6686 - accuracy: 0.4999
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6706 - accuracy: 0.4997
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6733 - accuracy: 0.4996
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6706 - accuracy: 0.4997
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6693 - accuracy: 0.4998
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6706 - accuracy: 0.4997
23040/25000 [==========================>...] - ETA: 5s - loss: 7.6733 - accuracy: 0.4996
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6713 - accuracy: 0.4997
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6733 - accuracy: 0.4996
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6772 - accuracy: 0.4993
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6746 - accuracy: 0.4995
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6746 - accuracy: 0.4995
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6765 - accuracy: 0.4994
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6765 - accuracy: 0.4994
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6778 - accuracy: 0.4993
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6817 - accuracy: 0.4990
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6811 - accuracy: 0.4991
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6791 - accuracy: 0.4992
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6771 - accuracy: 0.4993
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6732 - accuracy: 0.4996
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6640 - accuracy: 0.5002
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6640 - accuracy: 0.5002
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6634 - accuracy: 0.5002
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6634 - accuracy: 0.5002
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6634 - accuracy: 0.5002
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6608 - accuracy: 0.5004
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6628 - accuracy: 0.5003
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6602 - accuracy: 0.5004
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6596 - accuracy: 0.5005
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6583 - accuracy: 0.5005
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6589 - accuracy: 0.5005
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6589 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6596 - accuracy: 0.5005
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6590 - accuracy: 0.5005
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6577 - accuracy: 0.5006
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6558 - accuracy: 0.5007
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6558 - accuracy: 0.5007
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6596 - accuracy: 0.5005
24192/25000 [============================>.] - ETA: 2s - loss: 7.6609 - accuracy: 0.5004
24224/25000 [============================>.] - ETA: 1s - loss: 7.6609 - accuracy: 0.5004
24256/25000 [============================>.] - ETA: 1s - loss: 7.6603 - accuracy: 0.5004
24288/25000 [============================>.] - ETA: 1s - loss: 7.6571 - accuracy: 0.5006
24320/25000 [============================>.] - ETA: 1s - loss: 7.6565 - accuracy: 0.5007
24352/25000 [============================>.] - ETA: 1s - loss: 7.6578 - accuracy: 0.5006
24384/25000 [============================>.] - ETA: 1s - loss: 7.6584 - accuracy: 0.5005
24416/25000 [============================>.] - ETA: 1s - loss: 7.6591 - accuracy: 0.5005
24448/25000 [============================>.] - ETA: 1s - loss: 7.6560 - accuracy: 0.5007
24480/25000 [============================>.] - ETA: 1s - loss: 7.6547 - accuracy: 0.5008
24512/25000 [============================>.] - ETA: 1s - loss: 7.6547 - accuracy: 0.5008
24544/25000 [============================>.] - ETA: 1s - loss: 7.6579 - accuracy: 0.5006
24576/25000 [============================>.] - ETA: 1s - loss: 7.6591 - accuracy: 0.5005
24608/25000 [============================>.] - ETA: 1s - loss: 7.6591 - accuracy: 0.5005
24640/25000 [============================>.] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24672/25000 [============================>.] - ETA: 0s - loss: 7.6585 - accuracy: 0.5005
24704/25000 [============================>.] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24736/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24768/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24800/25000 [============================>.] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24832/25000 [============================>.] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24864/25000 [============================>.] - ETA: 0s - loss: 7.6605 - accuracy: 0.5004
24896/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24928/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24960/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
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
