
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '7d2329693089c1f82c9643c24694005c94b5ebed', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/7d2329693089c1f82c9643c24694005c94b5ebed

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/7d2329693089c1f82c9643c24694005c94b5ebed

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
	Data preprocessing and feature engineering runtime = 0.29s ...
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
 40%|████      | 2/5 [00:59<01:28, 29.62s/it] 40%|████      | 2/5 [00:59<01:28, 29.63s/it]
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.027226694003020864, 'embedding_size_factor': 1.1042576583323314, 'layers.choice': 1, 'learning_rate': 0.009758573305320718, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 7.4425663549630075e-09} and reward: 0.3842
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\x9b\xe1P\x81H\xe1\xd2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\xab\n\x14\x0eQ\x88X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x83\xfcM\x89\x9d\xa3\x07X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>?\xf701\x01\x9a\x9du.' and reward: 0.3842
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\x9b\xe1P\x81H\xe1\xd2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\xab\n\x14\x0eQ\x88X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x83\xfcM\x89\x9d\xa3\x07X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>?\xf701\x01\x9a\x9du.' and reward: 0.3842
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 132.90432047843933
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.71s of the -14.98s of remaining time.
Ensemble size: 61
Ensemble weights: 
[0.54098361 0.45901639]
	0.3904	 = Validation accuracy score
	1.01s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 136.03s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f99168ac908> 

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
 [-0.14640869  0.10365465 -0.01402482 -0.07437767 -0.07435703 -0.05076144]
 [ 0.21179715  0.03994213 -0.17419854  0.28943706 -0.07575951  0.0746849 ]
 [ 0.07237687 -0.00593891  0.23440531  0.14800414 -0.20148949  0.16598727]
 [ 0.13485852  0.04513534 -0.18636307 -0.19087607 -0.03997061  0.00839646]
 [-0.2784723  -0.060878    0.08508789 -0.2128396  -0.01657939  0.14203005]
 [ 0.20782609 -0.01577467  0.09261441  0.0185945   0.08968936 -0.04068655]
 [ 0.41533154  0.45746094 -0.25359517  0.46286777 -0.00816219  0.23748192]
 [ 0.13379037  0.60515904 -0.28016064  0.49432543 -0.47041851 -0.21579374]
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
{'loss': 0.49704546481370926, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-21 00:27:23.792589: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.3815281614661217, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-21 00:27:25.077257: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 2269184/17464789 [==>...........................] - ETA: 0s
10067968/17464789 [================>.............] - ETA: 0s
16392192/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-21 00:27:37.920810: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-21 00:27:37.926466: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-21 00:27:37.926644: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5630e2f9cee0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 00:27:37.926661: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 5:05 - loss: 10.5416 - accuracy: 0.3125
   64/25000 [..............................] - ETA: 3:21 - loss: 9.8229 - accuracy: 0.3594 
   96/25000 [..............................] - ETA: 2:45 - loss: 9.4236 - accuracy: 0.3854
  128/25000 [..............................] - ETA: 2:25 - loss: 8.8645 - accuracy: 0.4219
  160/25000 [..............................] - ETA: 2:14 - loss: 8.6249 - accuracy: 0.4375
  192/25000 [..............................] - ETA: 2:08 - loss: 8.3055 - accuracy: 0.4583
  224/25000 [..............................] - ETA: 2:03 - loss: 7.8035 - accuracy: 0.4911
  256/25000 [..............................] - ETA: 1:59 - loss: 7.7265 - accuracy: 0.4961
  288/25000 [..............................] - ETA: 1:56 - loss: 7.4004 - accuracy: 0.5174
  320/25000 [..............................] - ETA: 1:53 - loss: 7.3312 - accuracy: 0.5219
  352/25000 [..............................] - ETA: 1:51 - loss: 7.3617 - accuracy: 0.5199
  384/25000 [..............................] - ETA: 1:49 - loss: 7.3072 - accuracy: 0.5234
  416/25000 [..............................] - ETA: 1:47 - loss: 7.3717 - accuracy: 0.5192
  448/25000 [..............................] - ETA: 1:46 - loss: 7.4270 - accuracy: 0.5156
  480/25000 [..............................] - ETA: 1:45 - loss: 7.3791 - accuracy: 0.5188
  512/25000 [..............................] - ETA: 1:44 - loss: 7.4270 - accuracy: 0.5156
  544/25000 [..............................] - ETA: 1:43 - loss: 7.3848 - accuracy: 0.5184
  576/25000 [..............................] - ETA: 1:42 - loss: 7.5335 - accuracy: 0.5087
  608/25000 [..............................] - ETA: 1:41 - loss: 7.5405 - accuracy: 0.5082
  640/25000 [..............................] - ETA: 1:40 - loss: 7.4750 - accuracy: 0.5125
  672/25000 [..............................] - ETA: 1:39 - loss: 7.4384 - accuracy: 0.5149
  704/25000 [..............................] - ETA: 1:39 - loss: 7.5142 - accuracy: 0.5099
  736/25000 [..............................] - ETA: 1:38 - loss: 7.5000 - accuracy: 0.5109
  768/25000 [..............................] - ETA: 1:37 - loss: 7.4869 - accuracy: 0.5117
  800/25000 [..............................] - ETA: 1:37 - loss: 7.5133 - accuracy: 0.5100
  832/25000 [..............................] - ETA: 1:36 - loss: 7.5192 - accuracy: 0.5096
  864/25000 [>.............................] - ETA: 1:35 - loss: 7.5246 - accuracy: 0.5093
  896/25000 [>.............................] - ETA: 1:35 - loss: 7.5639 - accuracy: 0.5067
  928/25000 [>.............................] - ETA: 1:35 - loss: 7.5510 - accuracy: 0.5075
  960/25000 [>.............................] - ETA: 1:34 - loss: 7.5388 - accuracy: 0.5083
  992/25000 [>.............................] - ETA: 1:34 - loss: 7.5893 - accuracy: 0.5050
 1024/25000 [>.............................] - ETA: 1:34 - loss: 7.5468 - accuracy: 0.5078
 1056/25000 [>.............................] - ETA: 1:33 - loss: 7.4633 - accuracy: 0.5133
 1088/25000 [>.............................] - ETA: 1:33 - loss: 7.5398 - accuracy: 0.5083
 1120/25000 [>.............................] - ETA: 1:32 - loss: 7.5160 - accuracy: 0.5098
 1152/25000 [>.............................] - ETA: 1:32 - loss: 7.5202 - accuracy: 0.5095
 1184/25000 [>.............................] - ETA: 1:32 - loss: 7.5630 - accuracy: 0.5068
 1216/25000 [>.............................] - ETA: 1:32 - loss: 7.5784 - accuracy: 0.5058
 1248/25000 [>.............................] - ETA: 1:31 - loss: 7.5806 - accuracy: 0.5056
 1280/25000 [>.............................] - ETA: 1:31 - loss: 7.5588 - accuracy: 0.5070
 1312/25000 [>.............................] - ETA: 1:31 - loss: 7.5147 - accuracy: 0.5099
 1344/25000 [>.............................] - ETA: 1:31 - loss: 7.5069 - accuracy: 0.5104
 1376/25000 [>.............................] - ETA: 1:30 - loss: 7.5106 - accuracy: 0.5102
 1408/25000 [>.............................] - ETA: 1:30 - loss: 7.4924 - accuracy: 0.5114
 1440/25000 [>.............................] - ETA: 1:30 - loss: 7.4856 - accuracy: 0.5118
 1472/25000 [>.............................] - ETA: 1:30 - loss: 7.5104 - accuracy: 0.5102
 1504/25000 [>.............................] - ETA: 1:30 - loss: 7.5341 - accuracy: 0.5086
 1536/25000 [>.............................] - ETA: 1:30 - loss: 7.5568 - accuracy: 0.5072
 1568/25000 [>.............................] - ETA: 1:30 - loss: 7.6079 - accuracy: 0.5038
 1600/25000 [>.............................] - ETA: 1:29 - loss: 7.5708 - accuracy: 0.5063
 1632/25000 [>.............................] - ETA: 1:29 - loss: 7.5821 - accuracy: 0.5055
 1664/25000 [>.............................] - ETA: 1:29 - loss: 7.6021 - accuracy: 0.5042
 1696/25000 [=>............................] - ETA: 1:29 - loss: 7.6485 - accuracy: 0.5012
 1728/25000 [=>............................] - ETA: 1:29 - loss: 7.6666 - accuracy: 0.5000
 1760/25000 [=>............................] - ETA: 1:28 - loss: 7.6666 - accuracy: 0.5000
 1792/25000 [=>............................] - ETA: 1:28 - loss: 7.6410 - accuracy: 0.5017
 1824/25000 [=>............................] - ETA: 1:28 - loss: 7.6246 - accuracy: 0.5027
 1856/25000 [=>............................] - ETA: 1:28 - loss: 7.6253 - accuracy: 0.5027
 1888/25000 [=>............................] - ETA: 1:28 - loss: 7.6747 - accuracy: 0.4995
 1920/25000 [=>............................] - ETA: 1:27 - loss: 7.6666 - accuracy: 0.5000
 1952/25000 [=>............................] - ETA: 1:27 - loss: 7.6195 - accuracy: 0.5031
 1984/25000 [=>............................] - ETA: 1:27 - loss: 7.5739 - accuracy: 0.5060
 2016/25000 [=>............................] - ETA: 1:27 - loss: 7.5830 - accuracy: 0.5055
 2048/25000 [=>............................] - ETA: 1:26 - loss: 7.6217 - accuracy: 0.5029
 2080/25000 [=>............................] - ETA: 1:26 - loss: 7.6519 - accuracy: 0.5010
 2112/25000 [=>............................] - ETA: 1:26 - loss: 7.6521 - accuracy: 0.5009
 2144/25000 [=>............................] - ETA: 1:26 - loss: 7.6595 - accuracy: 0.5005
 2176/25000 [=>............................] - ETA: 1:26 - loss: 7.6243 - accuracy: 0.5028
 2208/25000 [=>............................] - ETA: 1:26 - loss: 7.6527 - accuracy: 0.5009
 2240/25000 [=>............................] - ETA: 1:25 - loss: 7.6392 - accuracy: 0.5018
 2272/25000 [=>............................] - ETA: 1:25 - loss: 7.6261 - accuracy: 0.5026
 2304/25000 [=>............................] - ETA: 1:25 - loss: 7.6467 - accuracy: 0.5013
 2336/25000 [=>............................] - ETA: 1:25 - loss: 7.6404 - accuracy: 0.5017
 2368/25000 [=>............................] - ETA: 1:25 - loss: 7.6407 - accuracy: 0.5017
 2400/25000 [=>............................] - ETA: 1:25 - loss: 7.6155 - accuracy: 0.5033
 2432/25000 [=>............................] - ETA: 1:24 - loss: 7.6162 - accuracy: 0.5033
 2464/25000 [=>............................] - ETA: 1:24 - loss: 7.6231 - accuracy: 0.5028
 2496/25000 [=>............................] - ETA: 1:24 - loss: 7.5990 - accuracy: 0.5044
 2528/25000 [==>...........................] - ETA: 1:24 - loss: 7.6060 - accuracy: 0.5040
 2560/25000 [==>...........................] - ETA: 1:24 - loss: 7.6307 - accuracy: 0.5023
 2592/25000 [==>...........................] - ETA: 1:24 - loss: 7.6489 - accuracy: 0.5012
 2624/25000 [==>...........................] - ETA: 1:24 - loss: 7.6608 - accuracy: 0.5004
 2656/25000 [==>...........................] - ETA: 1:23 - loss: 7.6608 - accuracy: 0.5004
 2688/25000 [==>...........................] - ETA: 1:23 - loss: 7.6495 - accuracy: 0.5011
 2720/25000 [==>...........................] - ETA: 1:23 - loss: 7.6666 - accuracy: 0.5000
 2752/25000 [==>...........................] - ETA: 1:23 - loss: 7.6833 - accuracy: 0.4989
 2784/25000 [==>...........................] - ETA: 1:23 - loss: 7.6611 - accuracy: 0.5004
 2816/25000 [==>...........................] - ETA: 1:23 - loss: 7.6830 - accuracy: 0.4989
 2848/25000 [==>...........................] - ETA: 1:23 - loss: 7.6935 - accuracy: 0.4982
 2880/25000 [==>...........................] - ETA: 1:22 - loss: 7.7145 - accuracy: 0.4969
 2912/25000 [==>...........................] - ETA: 1:22 - loss: 7.7193 - accuracy: 0.4966
 2944/25000 [==>...........................] - ETA: 1:22 - loss: 7.6979 - accuracy: 0.4980
 2976/25000 [==>...........................] - ETA: 1:22 - loss: 7.6821 - accuracy: 0.4990
 3008/25000 [==>...........................] - ETA: 1:22 - loss: 7.6768 - accuracy: 0.4993
 3040/25000 [==>...........................] - ETA: 1:22 - loss: 7.6616 - accuracy: 0.5003
 3072/25000 [==>...........................] - ETA: 1:22 - loss: 7.6467 - accuracy: 0.5013
 3104/25000 [==>...........................] - ETA: 1:21 - loss: 7.6567 - accuracy: 0.5006
 3136/25000 [==>...........................] - ETA: 1:21 - loss: 7.6568 - accuracy: 0.5006
 3168/25000 [==>...........................] - ETA: 1:21 - loss: 7.6618 - accuracy: 0.5003
 3200/25000 [==>...........................] - ETA: 1:21 - loss: 7.6906 - accuracy: 0.4984
 3232/25000 [==>...........................] - ETA: 1:21 - loss: 7.7188 - accuracy: 0.4966
 3264/25000 [==>...........................] - ETA: 1:21 - loss: 7.7089 - accuracy: 0.4972
 3296/25000 [==>...........................] - ETA: 1:21 - loss: 7.7317 - accuracy: 0.4958
 3328/25000 [==>...........................] - ETA: 1:20 - loss: 7.7081 - accuracy: 0.4973
 3360/25000 [===>..........................] - ETA: 1:20 - loss: 7.7123 - accuracy: 0.4970
 3392/25000 [===>..........................] - ETA: 1:20 - loss: 7.7028 - accuracy: 0.4976
 3424/25000 [===>..........................] - ETA: 1:20 - loss: 7.7204 - accuracy: 0.4965
 3456/25000 [===>..........................] - ETA: 1:20 - loss: 7.7154 - accuracy: 0.4968
 3488/25000 [===>..........................] - ETA: 1:20 - loss: 7.7062 - accuracy: 0.4974
 3520/25000 [===>..........................] - ETA: 1:20 - loss: 7.7015 - accuracy: 0.4977
 3552/25000 [===>..........................] - ETA: 1:19 - loss: 7.7055 - accuracy: 0.4975
 3584/25000 [===>..........................] - ETA: 1:19 - loss: 7.6966 - accuracy: 0.4980
 3616/25000 [===>..........................] - ETA: 1:19 - loss: 7.6963 - accuracy: 0.4981
 3648/25000 [===>..........................] - ETA: 1:19 - loss: 7.7044 - accuracy: 0.4975
 3680/25000 [===>..........................] - ETA: 1:19 - loss: 7.7166 - accuracy: 0.4967
 3712/25000 [===>..........................] - ETA: 1:19 - loss: 7.7286 - accuracy: 0.4960
 3744/25000 [===>..........................] - ETA: 1:19 - loss: 7.7158 - accuracy: 0.4968
 3776/25000 [===>..........................] - ETA: 1:18 - loss: 7.7153 - accuracy: 0.4968
 3808/25000 [===>..........................] - ETA: 1:18 - loss: 7.7230 - accuracy: 0.4963
 3840/25000 [===>..........................] - ETA: 1:18 - loss: 7.7065 - accuracy: 0.4974
 3872/25000 [===>..........................] - ETA: 1:18 - loss: 7.7141 - accuracy: 0.4969
 3904/25000 [===>..........................] - ETA: 1:18 - loss: 7.6863 - accuracy: 0.4987
 3936/25000 [===>..........................] - ETA: 1:18 - loss: 7.6783 - accuracy: 0.4992
 3968/25000 [===>..........................] - ETA: 1:18 - loss: 7.6859 - accuracy: 0.4987
 4000/25000 [===>..........................] - ETA: 1:17 - loss: 7.6935 - accuracy: 0.4983
 4032/25000 [===>..........................] - ETA: 1:17 - loss: 7.7161 - accuracy: 0.4968
 4064/25000 [===>..........................] - ETA: 1:17 - loss: 7.7081 - accuracy: 0.4973
 4096/25000 [===>..........................] - ETA: 1:17 - loss: 7.7078 - accuracy: 0.4973
 4128/25000 [===>..........................] - ETA: 1:17 - loss: 7.7112 - accuracy: 0.4971
 4160/25000 [===>..........................] - ETA: 1:17 - loss: 7.6998 - accuracy: 0.4978
 4192/25000 [====>.........................] - ETA: 1:17 - loss: 7.7032 - accuracy: 0.4976
 4224/25000 [====>.........................] - ETA: 1:16 - loss: 7.6848 - accuracy: 0.4988
 4256/25000 [====>.........................] - ETA: 1:16 - loss: 7.6810 - accuracy: 0.4991
 4288/25000 [====>.........................] - ETA: 1:16 - loss: 7.6845 - accuracy: 0.4988
 4320/25000 [====>.........................] - ETA: 1:16 - loss: 7.6879 - accuracy: 0.4986
 4352/25000 [====>.........................] - ETA: 1:16 - loss: 7.7054 - accuracy: 0.4975
 4384/25000 [====>.........................] - ETA: 1:16 - loss: 7.6946 - accuracy: 0.4982
 4416/25000 [====>.........................] - ETA: 1:16 - loss: 7.6770 - accuracy: 0.4993
 4448/25000 [====>.........................] - ETA: 1:16 - loss: 7.6770 - accuracy: 0.4993
 4480/25000 [====>.........................] - ETA: 1:15 - loss: 7.6666 - accuracy: 0.5000
 4512/25000 [====>.........................] - ETA: 1:15 - loss: 7.6598 - accuracy: 0.5004
 4544/25000 [====>.........................] - ETA: 1:15 - loss: 7.6767 - accuracy: 0.4993
 4576/25000 [====>.........................] - ETA: 1:15 - loss: 7.6633 - accuracy: 0.5002
 4608/25000 [====>.........................] - ETA: 1:15 - loss: 7.6666 - accuracy: 0.5000
 4640/25000 [====>.........................] - ETA: 1:15 - loss: 7.6600 - accuracy: 0.5004
 4672/25000 [====>.........................] - ETA: 1:15 - loss: 7.6699 - accuracy: 0.4998
 4704/25000 [====>.........................] - ETA: 1:15 - loss: 7.6731 - accuracy: 0.4996
 4736/25000 [====>.........................] - ETA: 1:14 - loss: 7.6731 - accuracy: 0.4996
 4768/25000 [====>.........................] - ETA: 1:14 - loss: 7.6731 - accuracy: 0.4996
 4800/25000 [====>.........................] - ETA: 1:14 - loss: 7.6730 - accuracy: 0.4996
 4832/25000 [====>.........................] - ETA: 1:14 - loss: 7.6698 - accuracy: 0.4998
 4864/25000 [====>.........................] - ETA: 1:14 - loss: 7.6761 - accuracy: 0.4994
 4896/25000 [====>.........................] - ETA: 1:14 - loss: 7.6948 - accuracy: 0.4982
 4928/25000 [====>.........................] - ETA: 1:14 - loss: 7.6822 - accuracy: 0.4990
 4960/25000 [====>.........................] - ETA: 1:14 - loss: 7.6728 - accuracy: 0.4996
 4992/25000 [====>.........................] - ETA: 1:13 - loss: 7.6666 - accuracy: 0.5000
 5024/25000 [=====>........................] - ETA: 1:13 - loss: 7.6666 - accuracy: 0.5000
 5056/25000 [=====>........................] - ETA: 1:13 - loss: 7.6757 - accuracy: 0.4994
 5088/25000 [=====>........................] - ETA: 1:13 - loss: 7.6787 - accuracy: 0.4992
 5120/25000 [=====>........................] - ETA: 1:13 - loss: 7.6786 - accuracy: 0.4992
 5152/25000 [=====>........................] - ETA: 1:13 - loss: 7.6666 - accuracy: 0.5000
 5184/25000 [=====>........................] - ETA: 1:13 - loss: 7.6607 - accuracy: 0.5004
 5216/25000 [=====>........................] - ETA: 1:12 - loss: 7.6666 - accuracy: 0.5000
 5248/25000 [=====>........................] - ETA: 1:12 - loss: 7.6666 - accuracy: 0.5000
 5280/25000 [=====>........................] - ETA: 1:12 - loss: 7.6724 - accuracy: 0.4996
 5312/25000 [=====>........................] - ETA: 1:12 - loss: 7.6811 - accuracy: 0.4991
 5344/25000 [=====>........................] - ETA: 1:12 - loss: 7.6924 - accuracy: 0.4983
 5376/25000 [=====>........................] - ETA: 1:12 - loss: 7.6923 - accuracy: 0.4983
 5408/25000 [=====>........................] - ETA: 1:12 - loss: 7.6865 - accuracy: 0.4987
 5440/25000 [=====>........................] - ETA: 1:12 - loss: 7.6892 - accuracy: 0.4985
 5472/25000 [=====>........................] - ETA: 1:11 - loss: 7.6946 - accuracy: 0.4982
 5504/25000 [=====>........................] - ETA: 1:11 - loss: 7.6889 - accuracy: 0.4985
 5536/25000 [=====>........................] - ETA: 1:11 - loss: 7.6832 - accuracy: 0.4989
 5568/25000 [=====>........................] - ETA: 1:11 - loss: 7.6804 - accuracy: 0.4991
 5600/25000 [=====>........................] - ETA: 1:11 - loss: 7.6694 - accuracy: 0.4998
 5632/25000 [=====>........................] - ETA: 1:11 - loss: 7.6666 - accuracy: 0.5000
 5664/25000 [=====>........................] - ETA: 1:11 - loss: 7.6612 - accuracy: 0.5004
 5696/25000 [=====>........................] - ETA: 1:11 - loss: 7.6532 - accuracy: 0.5009
 5728/25000 [=====>........................] - ETA: 1:11 - loss: 7.6479 - accuracy: 0.5012
 5760/25000 [=====>........................] - ETA: 1:10 - loss: 7.6453 - accuracy: 0.5014
 5792/25000 [=====>........................] - ETA: 1:10 - loss: 7.6454 - accuracy: 0.5014
 5824/25000 [=====>........................] - ETA: 1:10 - loss: 7.6482 - accuracy: 0.5012
 5856/25000 [======>.......................] - ETA: 1:10 - loss: 7.6719 - accuracy: 0.4997
 5888/25000 [======>.......................] - ETA: 1:10 - loss: 7.6822 - accuracy: 0.4990
 5920/25000 [======>.......................] - ETA: 1:10 - loss: 7.6796 - accuracy: 0.4992
 5952/25000 [======>.......................] - ETA: 1:10 - loss: 7.6847 - accuracy: 0.4988
 5984/25000 [======>.......................] - ETA: 1:10 - loss: 7.6871 - accuracy: 0.4987
 6016/25000 [======>.......................] - ETA: 1:09 - loss: 7.6768 - accuracy: 0.4993
 6048/25000 [======>.......................] - ETA: 1:09 - loss: 7.6742 - accuracy: 0.4995
 6080/25000 [======>.......................] - ETA: 1:09 - loss: 7.6818 - accuracy: 0.4990
 6112/25000 [======>.......................] - ETA: 1:09 - loss: 7.6942 - accuracy: 0.4982
 6144/25000 [======>.......................] - ETA: 1:09 - loss: 7.6916 - accuracy: 0.4984
 6176/25000 [======>.......................] - ETA: 1:09 - loss: 7.6865 - accuracy: 0.4987
 6208/25000 [======>.......................] - ETA: 1:09 - loss: 7.6938 - accuracy: 0.4982
 6240/25000 [======>.......................] - ETA: 1:08 - loss: 7.6961 - accuracy: 0.4981
 6272/25000 [======>.......................] - ETA: 1:08 - loss: 7.6813 - accuracy: 0.4990
 6304/25000 [======>.......................] - ETA: 1:08 - loss: 7.6739 - accuracy: 0.4995
 6336/25000 [======>.......................] - ETA: 1:08 - loss: 7.6884 - accuracy: 0.4986
 6368/25000 [======>.......................] - ETA: 1:08 - loss: 7.6883 - accuracy: 0.4986
 6400/25000 [======>.......................] - ETA: 1:08 - loss: 7.6906 - accuracy: 0.4984
 6432/25000 [======>.......................] - ETA: 1:08 - loss: 7.6809 - accuracy: 0.4991
 6464/25000 [======>.......................] - ETA: 1:08 - loss: 7.6832 - accuracy: 0.4989
 6496/25000 [======>.......................] - ETA: 1:08 - loss: 7.6761 - accuracy: 0.4994
 6528/25000 [======>.......................] - ETA: 1:07 - loss: 7.6760 - accuracy: 0.4994
 6560/25000 [======>.......................] - ETA: 1:07 - loss: 7.6900 - accuracy: 0.4985
 6592/25000 [======>.......................] - ETA: 1:07 - loss: 7.6899 - accuracy: 0.4985
 6624/25000 [======>.......................] - ETA: 1:07 - loss: 7.6828 - accuracy: 0.4989
 6656/25000 [======>.......................] - ETA: 1:07 - loss: 7.6781 - accuracy: 0.4992
 6688/25000 [=======>......................] - ETA: 1:07 - loss: 7.6827 - accuracy: 0.4990
 6720/25000 [=======>......................] - ETA: 1:07 - loss: 7.6803 - accuracy: 0.4991
 6752/25000 [=======>......................] - ETA: 1:07 - loss: 7.6825 - accuracy: 0.4990
 6784/25000 [=======>......................] - ETA: 1:06 - loss: 7.6847 - accuracy: 0.4988
 6816/25000 [=======>......................] - ETA: 1:06 - loss: 7.6846 - accuracy: 0.4988
 6848/25000 [=======>......................] - ETA: 1:06 - loss: 7.6845 - accuracy: 0.4988
 6880/25000 [=======>......................] - ETA: 1:06 - loss: 7.6911 - accuracy: 0.4984
 6912/25000 [=======>......................] - ETA: 1:06 - loss: 7.6932 - accuracy: 0.4983
 6944/25000 [=======>......................] - ETA: 1:06 - loss: 7.6953 - accuracy: 0.4981
 6976/25000 [=======>......................] - ETA: 1:06 - loss: 7.6996 - accuracy: 0.4978
 7008/25000 [=======>......................] - ETA: 1:06 - loss: 7.6951 - accuracy: 0.4981
 7040/25000 [=======>......................] - ETA: 1:05 - loss: 7.6949 - accuracy: 0.4982
 7072/25000 [=======>......................] - ETA: 1:05 - loss: 7.6970 - accuracy: 0.4980
 7104/25000 [=======>......................] - ETA: 1:05 - loss: 7.7033 - accuracy: 0.4976
 7136/25000 [=======>......................] - ETA: 1:05 - loss: 7.7031 - accuracy: 0.4976
 7168/25000 [=======>......................] - ETA: 1:05 - loss: 7.7094 - accuracy: 0.4972
 7200/25000 [=======>......................] - ETA: 1:05 - loss: 7.6943 - accuracy: 0.4982
 7232/25000 [=======>......................] - ETA: 1:05 - loss: 7.6963 - accuracy: 0.4981
 7264/25000 [=======>......................] - ETA: 1:05 - loss: 7.6962 - accuracy: 0.4981
 7296/25000 [=======>......................] - ETA: 1:04 - loss: 7.6960 - accuracy: 0.4981
 7328/25000 [=======>......................] - ETA: 1:04 - loss: 7.6917 - accuracy: 0.4984
 7360/25000 [=======>......................] - ETA: 1:04 - loss: 7.7020 - accuracy: 0.4977
 7392/25000 [=======>......................] - ETA: 1:04 - loss: 7.7081 - accuracy: 0.4973
 7424/25000 [=======>......................] - ETA: 1:04 - loss: 7.7121 - accuracy: 0.4970
 7456/25000 [=======>......................] - ETA: 1:04 - loss: 7.7098 - accuracy: 0.4972
 7488/25000 [=======>......................] - ETA: 1:04 - loss: 7.7178 - accuracy: 0.4967
 7520/25000 [========>.....................] - ETA: 1:04 - loss: 7.7094 - accuracy: 0.4972
 7552/25000 [========>.....................] - ETA: 1:03 - loss: 7.7072 - accuracy: 0.4974
 7584/25000 [========>.....................] - ETA: 1:03 - loss: 7.7192 - accuracy: 0.4966
 7616/25000 [========>.....................] - ETA: 1:03 - loss: 7.7290 - accuracy: 0.4959
 7648/25000 [========>.....................] - ETA: 1:03 - loss: 7.7368 - accuracy: 0.4954
 7680/25000 [========>.....................] - ETA: 1:03 - loss: 7.7345 - accuracy: 0.4956
 7712/25000 [========>.....................] - ETA: 1:03 - loss: 7.7263 - accuracy: 0.4961
 7744/25000 [========>.....................] - ETA: 1:03 - loss: 7.7260 - accuracy: 0.4961
 7776/25000 [========>.....................] - ETA: 1:03 - loss: 7.7179 - accuracy: 0.4967
 7808/25000 [========>.....................] - ETA: 1:03 - loss: 7.7177 - accuracy: 0.4967
 7840/25000 [========>.....................] - ETA: 1:02 - loss: 7.7175 - accuracy: 0.4967
 7872/25000 [========>.....................] - ETA: 1:02 - loss: 7.7173 - accuracy: 0.4967
 7904/25000 [========>.....................] - ETA: 1:02 - loss: 7.7132 - accuracy: 0.4970
 7936/25000 [========>.....................] - ETA: 1:02 - loss: 7.7091 - accuracy: 0.4972
 7968/25000 [========>.....................] - ETA: 1:02 - loss: 7.7051 - accuracy: 0.4975
 8000/25000 [========>.....................] - ETA: 1:02 - loss: 7.7107 - accuracy: 0.4971
 8032/25000 [========>.....................] - ETA: 1:02 - loss: 7.7163 - accuracy: 0.4968
 8064/25000 [========>.....................] - ETA: 1:02 - loss: 7.7199 - accuracy: 0.4965
 8096/25000 [========>.....................] - ETA: 1:01 - loss: 7.7178 - accuracy: 0.4967
 8128/25000 [========>.....................] - ETA: 1:01 - loss: 7.7213 - accuracy: 0.4964
 8160/25000 [========>.....................] - ETA: 1:01 - loss: 7.7230 - accuracy: 0.4963
 8192/25000 [========>.....................] - ETA: 1:01 - loss: 7.7228 - accuracy: 0.4963
 8224/25000 [========>.....................] - ETA: 1:01 - loss: 7.7263 - accuracy: 0.4961
 8256/25000 [========>.....................] - ETA: 1:01 - loss: 7.7316 - accuracy: 0.4958
 8288/25000 [========>.....................] - ETA: 1:01 - loss: 7.7369 - accuracy: 0.4954
 8320/25000 [========>.....................] - ETA: 1:01 - loss: 7.7348 - accuracy: 0.4956
 8352/25000 [=========>....................] - ETA: 1:00 - loss: 7.7272 - accuracy: 0.4960
 8384/25000 [=========>....................] - ETA: 1:00 - loss: 7.7288 - accuracy: 0.4959
 8416/25000 [=========>....................] - ETA: 1:00 - loss: 7.7231 - accuracy: 0.4963
 8448/25000 [=========>....................] - ETA: 1:00 - loss: 7.7247 - accuracy: 0.4962
 8480/25000 [=========>....................] - ETA: 1:00 - loss: 7.7209 - accuracy: 0.4965
 8512/25000 [=========>....................] - ETA: 1:00 - loss: 7.7207 - accuracy: 0.4965
 8544/25000 [=========>....................] - ETA: 1:00 - loss: 7.7169 - accuracy: 0.4967
 8576/25000 [=========>....................] - ETA: 1:00 - loss: 7.7167 - accuracy: 0.4967
 8608/25000 [=========>....................] - ETA: 59s - loss: 7.7147 - accuracy: 0.4969 
 8640/25000 [=========>....................] - ETA: 59s - loss: 7.7145 - accuracy: 0.4969
 8672/25000 [=========>....................] - ETA: 59s - loss: 7.7108 - accuracy: 0.4971
 8704/25000 [=========>....................] - ETA: 59s - loss: 7.7089 - accuracy: 0.4972
 8736/25000 [=========>....................] - ETA: 59s - loss: 7.7140 - accuracy: 0.4969
 8768/25000 [=========>....................] - ETA: 59s - loss: 7.7051 - accuracy: 0.4975
 8800/25000 [=========>....................] - ETA: 59s - loss: 7.7067 - accuracy: 0.4974
 8832/25000 [=========>....................] - ETA: 59s - loss: 7.7031 - accuracy: 0.4976
 8864/25000 [=========>....................] - ETA: 58s - loss: 7.7012 - accuracy: 0.4977
 8896/25000 [=========>....................] - ETA: 58s - loss: 7.6976 - accuracy: 0.4980
 8928/25000 [=========>....................] - ETA: 58s - loss: 7.6941 - accuracy: 0.4982
 8960/25000 [=========>....................] - ETA: 58s - loss: 7.6974 - accuracy: 0.4980
 8992/25000 [=========>....................] - ETA: 58s - loss: 7.6956 - accuracy: 0.4981
 9024/25000 [=========>....................] - ETA: 58s - loss: 7.7057 - accuracy: 0.4975
 9056/25000 [=========>....................] - ETA: 58s - loss: 7.7005 - accuracy: 0.4978
 9088/25000 [=========>....................] - ETA: 58s - loss: 7.7004 - accuracy: 0.4978
 9120/25000 [=========>....................] - ETA: 58s - loss: 7.6986 - accuracy: 0.4979
 9152/25000 [=========>....................] - ETA: 57s - loss: 7.7085 - accuracy: 0.4973
 9184/25000 [==========>...................] - ETA: 57s - loss: 7.7084 - accuracy: 0.4973
 9216/25000 [==========>...................] - ETA: 57s - loss: 7.7182 - accuracy: 0.4966
 9248/25000 [==========>...................] - ETA: 57s - loss: 7.7164 - accuracy: 0.4968
 9280/25000 [==========>...................] - ETA: 57s - loss: 7.7112 - accuracy: 0.4971
 9312/25000 [==========>...................] - ETA: 57s - loss: 7.7028 - accuracy: 0.4976
 9344/25000 [==========>...................] - ETA: 57s - loss: 7.7093 - accuracy: 0.4972
 9376/25000 [==========>...................] - ETA: 57s - loss: 7.7140 - accuracy: 0.4969
 9408/25000 [==========>...................] - ETA: 57s - loss: 7.7171 - accuracy: 0.4967
 9440/25000 [==========>...................] - ETA: 56s - loss: 7.7170 - accuracy: 0.4967
 9472/25000 [==========>...................] - ETA: 56s - loss: 7.7152 - accuracy: 0.4968
 9504/25000 [==========>...................] - ETA: 56s - loss: 7.7102 - accuracy: 0.4972
 9536/25000 [==========>...................] - ETA: 56s - loss: 7.7132 - accuracy: 0.4970
 9568/25000 [==========>...................] - ETA: 56s - loss: 7.7131 - accuracy: 0.4970
 9600/25000 [==========>...................] - ETA: 56s - loss: 7.7113 - accuracy: 0.4971
 9632/25000 [==========>...................] - ETA: 56s - loss: 7.7128 - accuracy: 0.4970
 9664/25000 [==========>...................] - ETA: 56s - loss: 7.7158 - accuracy: 0.4968
 9696/25000 [==========>...................] - ETA: 55s - loss: 7.7267 - accuracy: 0.4961
 9728/25000 [==========>...................] - ETA: 55s - loss: 7.7297 - accuracy: 0.4959
 9760/25000 [==========>...................] - ETA: 55s - loss: 7.7247 - accuracy: 0.4962
 9792/25000 [==========>...................] - ETA: 55s - loss: 7.7324 - accuracy: 0.4957
 9824/25000 [==========>...................] - ETA: 55s - loss: 7.7244 - accuracy: 0.4962
 9856/25000 [==========>...................] - ETA: 55s - loss: 7.7226 - accuracy: 0.4963
 9888/25000 [==========>...................] - ETA: 55s - loss: 7.7302 - accuracy: 0.4959
 9920/25000 [==========>...................] - ETA: 55s - loss: 7.7300 - accuracy: 0.4959
 9952/25000 [==========>...................] - ETA: 54s - loss: 7.7298 - accuracy: 0.4959
 9984/25000 [==========>...................] - ETA: 54s - loss: 7.7311 - accuracy: 0.4958
10016/25000 [===========>..................] - ETA: 54s - loss: 7.7294 - accuracy: 0.4959
10048/25000 [===========>..................] - ETA: 54s - loss: 7.7353 - accuracy: 0.4955
10080/25000 [===========>..................] - ETA: 54s - loss: 7.7351 - accuracy: 0.4955
10112/25000 [===========>..................] - ETA: 54s - loss: 7.7379 - accuracy: 0.4954
10144/25000 [===========>..................] - ETA: 54s - loss: 7.7407 - accuracy: 0.4952
10176/25000 [===========>..................] - ETA: 54s - loss: 7.7344 - accuracy: 0.4956
10208/25000 [===========>..................] - ETA: 54s - loss: 7.7387 - accuracy: 0.4953
10240/25000 [===========>..................] - ETA: 53s - loss: 7.7370 - accuracy: 0.4954
10272/25000 [===========>..................] - ETA: 53s - loss: 7.7323 - accuracy: 0.4957
10304/25000 [===========>..................] - ETA: 53s - loss: 7.7291 - accuracy: 0.4959
10336/25000 [===========>..................] - ETA: 53s - loss: 7.7274 - accuracy: 0.4960
10368/25000 [===========>..................] - ETA: 53s - loss: 7.7406 - accuracy: 0.4952
10400/25000 [===========>..................] - ETA: 53s - loss: 7.7359 - accuracy: 0.4955
10432/25000 [===========>..................] - ETA: 53s - loss: 7.7313 - accuracy: 0.4958
10464/25000 [===========>..................] - ETA: 53s - loss: 7.7326 - accuracy: 0.4957
10496/25000 [===========>..................] - ETA: 52s - loss: 7.7294 - accuracy: 0.4959
10528/25000 [===========>..................] - ETA: 52s - loss: 7.7351 - accuracy: 0.4955
10560/25000 [===========>..................] - ETA: 52s - loss: 7.7392 - accuracy: 0.4953
10592/25000 [===========>..................] - ETA: 52s - loss: 7.7361 - accuracy: 0.4955
10624/25000 [===========>..................] - ETA: 52s - loss: 7.7345 - accuracy: 0.4956
10656/25000 [===========>..................] - ETA: 52s - loss: 7.7357 - accuracy: 0.4955
10688/25000 [===========>..................] - ETA: 52s - loss: 7.7312 - accuracy: 0.4958
10720/25000 [===========>..................] - ETA: 52s - loss: 7.7281 - accuracy: 0.4960
10752/25000 [===========>..................] - ETA: 51s - loss: 7.7251 - accuracy: 0.4962
10784/25000 [===========>..................] - ETA: 51s - loss: 7.7235 - accuracy: 0.4963
10816/25000 [===========>..................] - ETA: 51s - loss: 7.7219 - accuracy: 0.4964
10848/25000 [============>.................] - ETA: 51s - loss: 7.7203 - accuracy: 0.4965
10880/25000 [============>.................] - ETA: 51s - loss: 7.7202 - accuracy: 0.4965
10912/25000 [============>.................] - ETA: 51s - loss: 7.7228 - accuracy: 0.4963
10944/25000 [============>.................] - ETA: 51s - loss: 7.7227 - accuracy: 0.4963
10976/25000 [============>.................] - ETA: 51s - loss: 7.7253 - accuracy: 0.4962
11008/25000 [============>.................] - ETA: 51s - loss: 7.7265 - accuracy: 0.4961
11040/25000 [============>.................] - ETA: 50s - loss: 7.7263 - accuracy: 0.4961
11072/25000 [============>.................] - ETA: 50s - loss: 7.7262 - accuracy: 0.4961
11104/25000 [============>.................] - ETA: 50s - loss: 7.7232 - accuracy: 0.4963
11136/25000 [============>.................] - ETA: 50s - loss: 7.7300 - accuracy: 0.4959
11168/25000 [============>.................] - ETA: 50s - loss: 7.7353 - accuracy: 0.4955
11200/25000 [============>.................] - ETA: 50s - loss: 7.7378 - accuracy: 0.4954
11232/25000 [============>.................] - ETA: 50s - loss: 7.7376 - accuracy: 0.4954
11264/25000 [============>.................] - ETA: 50s - loss: 7.7401 - accuracy: 0.4952
11296/25000 [============>.................] - ETA: 49s - loss: 7.7399 - accuracy: 0.4952
11328/25000 [============>.................] - ETA: 49s - loss: 7.7357 - accuracy: 0.4955
11360/25000 [============>.................] - ETA: 49s - loss: 7.7355 - accuracy: 0.4955
11392/25000 [============>.................] - ETA: 49s - loss: 7.7339 - accuracy: 0.4956
11424/25000 [============>.................] - ETA: 49s - loss: 7.7378 - accuracy: 0.4954
11456/25000 [============>.................] - ETA: 49s - loss: 7.7376 - accuracy: 0.4954
11488/25000 [============>.................] - ETA: 49s - loss: 7.7334 - accuracy: 0.4956
11520/25000 [============>.................] - ETA: 49s - loss: 7.7358 - accuracy: 0.4955
11552/25000 [============>.................] - ETA: 48s - loss: 7.7343 - accuracy: 0.4956
11584/25000 [============>.................] - ETA: 48s - loss: 7.7315 - accuracy: 0.4958
11616/25000 [============>.................] - ETA: 48s - loss: 7.7300 - accuracy: 0.4959
11648/25000 [============>.................] - ETA: 48s - loss: 7.7298 - accuracy: 0.4959
11680/25000 [=============>................] - ETA: 48s - loss: 7.7349 - accuracy: 0.4955
11712/25000 [=============>................] - ETA: 48s - loss: 7.7321 - accuracy: 0.4957
11744/25000 [=============>................] - ETA: 48s - loss: 7.7319 - accuracy: 0.4957
11776/25000 [=============>................] - ETA: 48s - loss: 7.7330 - accuracy: 0.4957
11808/25000 [=============>................] - ETA: 48s - loss: 7.7302 - accuracy: 0.4959
11840/25000 [=============>................] - ETA: 47s - loss: 7.7314 - accuracy: 0.4958
11872/25000 [=============>................] - ETA: 47s - loss: 7.7312 - accuracy: 0.4958
11904/25000 [=============>................] - ETA: 47s - loss: 7.7323 - accuracy: 0.4957
11936/25000 [=============>................] - ETA: 47s - loss: 7.7283 - accuracy: 0.4960
11968/25000 [=============>................] - ETA: 47s - loss: 7.7281 - accuracy: 0.4960
12000/25000 [=============>................] - ETA: 47s - loss: 7.7280 - accuracy: 0.4960
12032/25000 [=============>................] - ETA: 47s - loss: 7.7252 - accuracy: 0.4962
12064/25000 [=============>................] - ETA: 47s - loss: 7.7225 - accuracy: 0.4964
12096/25000 [=============>................] - ETA: 46s - loss: 7.7249 - accuracy: 0.4962
12128/25000 [=============>................] - ETA: 46s - loss: 7.7260 - accuracy: 0.4961
12160/25000 [=============>................] - ETA: 46s - loss: 7.7234 - accuracy: 0.4963
12192/25000 [=============>................] - ETA: 46s - loss: 7.7245 - accuracy: 0.4962
12224/25000 [=============>................] - ETA: 46s - loss: 7.7281 - accuracy: 0.4960
12256/25000 [=============>................] - ETA: 46s - loss: 7.7267 - accuracy: 0.4961
12288/25000 [=============>................] - ETA: 46s - loss: 7.7240 - accuracy: 0.4963
12320/25000 [=============>................] - ETA: 46s - loss: 7.7288 - accuracy: 0.4959
12352/25000 [=============>................] - ETA: 46s - loss: 7.7274 - accuracy: 0.4960
12384/25000 [=============>................] - ETA: 45s - loss: 7.7261 - accuracy: 0.4961
12416/25000 [=============>................] - ETA: 45s - loss: 7.7284 - accuracy: 0.4960
12448/25000 [=============>................] - ETA: 45s - loss: 7.7257 - accuracy: 0.4961
12480/25000 [=============>................] - ETA: 45s - loss: 7.7170 - accuracy: 0.4967
12512/25000 [==============>...............] - ETA: 45s - loss: 7.7218 - accuracy: 0.4964
12544/25000 [==============>...............] - ETA: 45s - loss: 7.7241 - accuracy: 0.4963
12576/25000 [==============>...............] - ETA: 45s - loss: 7.7264 - accuracy: 0.4961
12608/25000 [==============>...............] - ETA: 45s - loss: 7.7213 - accuracy: 0.4964
12640/25000 [==============>...............] - ETA: 44s - loss: 7.7248 - accuracy: 0.4962
12672/25000 [==============>...............] - ETA: 44s - loss: 7.7235 - accuracy: 0.4963
12704/25000 [==============>...............] - ETA: 44s - loss: 7.7270 - accuracy: 0.4961
12736/25000 [==============>...............] - ETA: 44s - loss: 7.7220 - accuracy: 0.4964
12768/25000 [==============>...............] - ETA: 44s - loss: 7.7183 - accuracy: 0.4966
12800/25000 [==============>...............] - ETA: 44s - loss: 7.7181 - accuracy: 0.4966
12832/25000 [==============>...............] - ETA: 44s - loss: 7.7204 - accuracy: 0.4965
12864/25000 [==============>...............] - ETA: 44s - loss: 7.7214 - accuracy: 0.4964
12896/25000 [==============>...............] - ETA: 43s - loss: 7.7261 - accuracy: 0.4961
12928/25000 [==============>...............] - ETA: 43s - loss: 7.7295 - accuracy: 0.4959
12960/25000 [==============>...............] - ETA: 43s - loss: 7.7258 - accuracy: 0.4961
12992/25000 [==============>...............] - ETA: 43s - loss: 7.7233 - accuracy: 0.4963
13024/25000 [==============>...............] - ETA: 43s - loss: 7.7231 - accuracy: 0.4963
13056/25000 [==============>...............] - ETA: 43s - loss: 7.7218 - accuracy: 0.4964
13088/25000 [==============>...............] - ETA: 43s - loss: 7.7229 - accuracy: 0.4963
13120/25000 [==============>...............] - ETA: 43s - loss: 7.7262 - accuracy: 0.4961
13152/25000 [==============>...............] - ETA: 43s - loss: 7.7214 - accuracy: 0.4964
13184/25000 [==============>...............] - ETA: 42s - loss: 7.7259 - accuracy: 0.4961
13216/25000 [==============>...............] - ETA: 42s - loss: 7.7246 - accuracy: 0.4962
13248/25000 [==============>...............] - ETA: 42s - loss: 7.7256 - accuracy: 0.4962
13280/25000 [==============>...............] - ETA: 42s - loss: 7.7278 - accuracy: 0.4960
13312/25000 [==============>...............] - ETA: 42s - loss: 7.7265 - accuracy: 0.4961
13344/25000 [===============>..............] - ETA: 42s - loss: 7.7252 - accuracy: 0.4962
13376/25000 [===============>..............] - ETA: 42s - loss: 7.7274 - accuracy: 0.4960
13408/25000 [===============>..............] - ETA: 42s - loss: 7.7295 - accuracy: 0.4959
13440/25000 [===============>..............] - ETA: 42s - loss: 7.7316 - accuracy: 0.4958
13472/25000 [===============>..............] - ETA: 41s - loss: 7.7304 - accuracy: 0.4958
13504/25000 [===============>..............] - ETA: 41s - loss: 7.7279 - accuracy: 0.4960
13536/25000 [===============>..............] - ETA: 41s - loss: 7.7255 - accuracy: 0.4962
13568/25000 [===============>..............] - ETA: 41s - loss: 7.7254 - accuracy: 0.4962
13600/25000 [===============>..............] - ETA: 41s - loss: 7.7207 - accuracy: 0.4965
13632/25000 [===============>..............] - ETA: 41s - loss: 7.7217 - accuracy: 0.4964
13664/25000 [===============>..............] - ETA: 41s - loss: 7.7171 - accuracy: 0.4967
13696/25000 [===============>..............] - ETA: 41s - loss: 7.7192 - accuracy: 0.4966
13728/25000 [===============>..............] - ETA: 40s - loss: 7.7158 - accuracy: 0.4968
13760/25000 [===============>..............] - ETA: 40s - loss: 7.7123 - accuracy: 0.4970
13792/25000 [===============>..............] - ETA: 40s - loss: 7.7144 - accuracy: 0.4969
13824/25000 [===============>..............] - ETA: 40s - loss: 7.7154 - accuracy: 0.4968
13856/25000 [===============>..............] - ETA: 40s - loss: 7.7186 - accuracy: 0.4966
13888/25000 [===============>..............] - ETA: 40s - loss: 7.7229 - accuracy: 0.4963
13920/25000 [===============>..............] - ETA: 40s - loss: 7.7206 - accuracy: 0.4965
13952/25000 [===============>..............] - ETA: 40s - loss: 7.7227 - accuracy: 0.4963
13984/25000 [===============>..............] - ETA: 40s - loss: 7.7214 - accuracy: 0.4964
14016/25000 [===============>..............] - ETA: 39s - loss: 7.7202 - accuracy: 0.4965
14048/25000 [===============>..............] - ETA: 39s - loss: 7.7201 - accuracy: 0.4965
14080/25000 [===============>..............] - ETA: 39s - loss: 7.7200 - accuracy: 0.4965
14112/25000 [===============>..............] - ETA: 39s - loss: 7.7209 - accuracy: 0.4965
14144/25000 [===============>..............] - ETA: 39s - loss: 7.7219 - accuracy: 0.4964
14176/25000 [================>.............] - ETA: 39s - loss: 7.7175 - accuracy: 0.4967
14208/25000 [================>.............] - ETA: 39s - loss: 7.7206 - accuracy: 0.4965
14240/25000 [================>.............] - ETA: 39s - loss: 7.7151 - accuracy: 0.4968
14272/25000 [================>.............] - ETA: 38s - loss: 7.7160 - accuracy: 0.4968
14304/25000 [================>.............] - ETA: 38s - loss: 7.7159 - accuracy: 0.4968
14336/25000 [================>.............] - ETA: 38s - loss: 7.7137 - accuracy: 0.4969
14368/25000 [================>.............] - ETA: 38s - loss: 7.7072 - accuracy: 0.4974
14400/25000 [================>.............] - ETA: 38s - loss: 7.7092 - accuracy: 0.4972
14432/25000 [================>.............] - ETA: 38s - loss: 7.7091 - accuracy: 0.4972
14464/25000 [================>.............] - ETA: 38s - loss: 7.7080 - accuracy: 0.4973
14496/25000 [================>.............] - ETA: 38s - loss: 7.7068 - accuracy: 0.4974
14528/25000 [================>.............] - ETA: 38s - loss: 7.7067 - accuracy: 0.4974
14560/25000 [================>.............] - ETA: 37s - loss: 7.7108 - accuracy: 0.4971
14592/25000 [================>.............] - ETA: 37s - loss: 7.7118 - accuracy: 0.4971
14624/25000 [================>.............] - ETA: 37s - loss: 7.7159 - accuracy: 0.4968
14656/25000 [================>.............] - ETA: 37s - loss: 7.7127 - accuracy: 0.4970
14688/25000 [================>.............] - ETA: 37s - loss: 7.7094 - accuracy: 0.4972
14720/25000 [================>.............] - ETA: 37s - loss: 7.7072 - accuracy: 0.4974
14752/25000 [================>.............] - ETA: 37s - loss: 7.7092 - accuracy: 0.4972
14784/25000 [================>.............] - ETA: 37s - loss: 7.7050 - accuracy: 0.4975
14816/25000 [================>.............] - ETA: 36s - loss: 7.7018 - accuracy: 0.4977
14848/25000 [================>.............] - ETA: 36s - loss: 7.7017 - accuracy: 0.4977
14880/25000 [================>.............] - ETA: 36s - loss: 7.6975 - accuracy: 0.4980
14912/25000 [================>.............] - ETA: 36s - loss: 7.6995 - accuracy: 0.4979
14944/25000 [================>.............] - ETA: 36s - loss: 7.6943 - accuracy: 0.4982
14976/25000 [================>.............] - ETA: 36s - loss: 7.6963 - accuracy: 0.4981
15008/25000 [=================>............] - ETA: 36s - loss: 7.6922 - accuracy: 0.4983
15040/25000 [=================>............] - ETA: 36s - loss: 7.6911 - accuracy: 0.4984
15072/25000 [=================>............] - ETA: 35s - loss: 7.6921 - accuracy: 0.4983
15104/25000 [=================>............] - ETA: 35s - loss: 7.6950 - accuracy: 0.4981
15136/25000 [=================>............] - ETA: 35s - loss: 7.6960 - accuracy: 0.4981
15168/25000 [=================>............] - ETA: 35s - loss: 7.6949 - accuracy: 0.4982
15200/25000 [=================>............] - ETA: 35s - loss: 7.6949 - accuracy: 0.4982
15232/25000 [=================>............] - ETA: 35s - loss: 7.6988 - accuracy: 0.4979
15264/25000 [=================>............] - ETA: 35s - loss: 7.6988 - accuracy: 0.4979
15296/25000 [=================>............] - ETA: 35s - loss: 7.6987 - accuracy: 0.4979
15328/25000 [=================>............] - ETA: 35s - loss: 7.6986 - accuracy: 0.4979
15360/25000 [=================>............] - ETA: 34s - loss: 7.6976 - accuracy: 0.4980
15392/25000 [=================>............] - ETA: 34s - loss: 7.7025 - accuracy: 0.4977
15424/25000 [=================>............] - ETA: 34s - loss: 7.7024 - accuracy: 0.4977
15456/25000 [=================>............] - ETA: 34s - loss: 7.6994 - accuracy: 0.4979
15488/25000 [=================>............] - ETA: 34s - loss: 7.7013 - accuracy: 0.4977
15520/25000 [=================>............] - ETA: 34s - loss: 7.7052 - accuracy: 0.4975
15552/25000 [=================>............] - ETA: 34s - loss: 7.7021 - accuracy: 0.4977
15584/25000 [=================>............] - ETA: 34s - loss: 7.7070 - accuracy: 0.4974
15616/25000 [=================>............] - ETA: 34s - loss: 7.7069 - accuracy: 0.4974
15648/25000 [=================>............] - ETA: 33s - loss: 7.7048 - accuracy: 0.4975
15680/25000 [=================>............] - ETA: 33s - loss: 7.7038 - accuracy: 0.4976
15712/25000 [=================>............] - ETA: 33s - loss: 7.7047 - accuracy: 0.4975
15744/25000 [=================>............] - ETA: 33s - loss: 7.7017 - accuracy: 0.4977
15776/25000 [=================>............] - ETA: 33s - loss: 7.7016 - accuracy: 0.4977
15808/25000 [=================>............] - ETA: 33s - loss: 7.6947 - accuracy: 0.4982
15840/25000 [==================>...........] - ETA: 33s - loss: 7.6908 - accuracy: 0.4984
15872/25000 [==================>...........] - ETA: 33s - loss: 7.6888 - accuracy: 0.4986
15904/25000 [==================>...........] - ETA: 32s - loss: 7.6927 - accuracy: 0.4983
15936/25000 [==================>...........] - ETA: 32s - loss: 7.6955 - accuracy: 0.4981
15968/25000 [==================>...........] - ETA: 32s - loss: 7.6945 - accuracy: 0.4982
16000/25000 [==================>...........] - ETA: 32s - loss: 7.6896 - accuracy: 0.4985
16032/25000 [==================>...........] - ETA: 32s - loss: 7.6886 - accuracy: 0.4986
16064/25000 [==================>...........] - ETA: 32s - loss: 7.6895 - accuracy: 0.4985
16096/25000 [==================>...........] - ETA: 32s - loss: 7.6838 - accuracy: 0.4989
16128/25000 [==================>...........] - ETA: 32s - loss: 7.6847 - accuracy: 0.4988
16160/25000 [==================>...........] - ETA: 32s - loss: 7.6818 - accuracy: 0.4990
16192/25000 [==================>...........] - ETA: 31s - loss: 7.6789 - accuracy: 0.4992
16224/25000 [==================>...........] - ETA: 31s - loss: 7.6789 - accuracy: 0.4992
16256/25000 [==================>...........] - ETA: 31s - loss: 7.6789 - accuracy: 0.4992
16288/25000 [==================>...........] - ETA: 31s - loss: 7.6826 - accuracy: 0.4990
16320/25000 [==================>...........] - ETA: 31s - loss: 7.6845 - accuracy: 0.4988
16352/25000 [==================>...........] - ETA: 31s - loss: 7.6872 - accuracy: 0.4987
16384/25000 [==================>...........] - ETA: 31s - loss: 7.6872 - accuracy: 0.4987
16416/25000 [==================>...........] - ETA: 31s - loss: 7.6844 - accuracy: 0.4988
16448/25000 [==================>...........] - ETA: 30s - loss: 7.6853 - accuracy: 0.4988
16480/25000 [==================>...........] - ETA: 30s - loss: 7.6834 - accuracy: 0.4989
16512/25000 [==================>...........] - ETA: 30s - loss: 7.6815 - accuracy: 0.4990
16544/25000 [==================>...........] - ETA: 30s - loss: 7.6814 - accuracy: 0.4990
16576/25000 [==================>...........] - ETA: 30s - loss: 7.6814 - accuracy: 0.4990
16608/25000 [==================>...........] - ETA: 30s - loss: 7.6832 - accuracy: 0.4989
16640/25000 [==================>...........] - ETA: 30s - loss: 7.6832 - accuracy: 0.4989
16672/25000 [===================>..........] - ETA: 30s - loss: 7.6859 - accuracy: 0.4987
16704/25000 [===================>..........] - ETA: 30s - loss: 7.6859 - accuracy: 0.4987
16736/25000 [===================>..........] - ETA: 29s - loss: 7.6859 - accuracy: 0.4987
16768/25000 [===================>..........] - ETA: 29s - loss: 7.6904 - accuracy: 0.4984
16800/25000 [===================>..........] - ETA: 29s - loss: 7.6885 - accuracy: 0.4986
16832/25000 [===================>..........] - ETA: 29s - loss: 7.6885 - accuracy: 0.4986
16864/25000 [===================>..........] - ETA: 29s - loss: 7.6903 - accuracy: 0.4985
16896/25000 [===================>..........] - ETA: 29s - loss: 7.6920 - accuracy: 0.4983
16928/25000 [===================>..........] - ETA: 29s - loss: 7.6893 - accuracy: 0.4985
16960/25000 [===================>..........] - ETA: 29s - loss: 7.6928 - accuracy: 0.4983
16992/25000 [===================>..........] - ETA: 29s - loss: 7.6937 - accuracy: 0.4982
17024/25000 [===================>..........] - ETA: 28s - loss: 7.6918 - accuracy: 0.4984
17056/25000 [===================>..........] - ETA: 28s - loss: 7.6900 - accuracy: 0.4985
17088/25000 [===================>..........] - ETA: 28s - loss: 7.6864 - accuracy: 0.4987
17120/25000 [===================>..........] - ETA: 28s - loss: 7.6845 - accuracy: 0.4988
17152/25000 [===================>..........] - ETA: 28s - loss: 7.6827 - accuracy: 0.4990
17184/25000 [===================>..........] - ETA: 28s - loss: 7.6800 - accuracy: 0.4991
17216/25000 [===================>..........] - ETA: 28s - loss: 7.6773 - accuracy: 0.4993
17248/25000 [===================>..........] - ETA: 28s - loss: 7.6808 - accuracy: 0.4991
17280/25000 [===================>..........] - ETA: 27s - loss: 7.6808 - accuracy: 0.4991
17312/25000 [===================>..........] - ETA: 27s - loss: 7.6764 - accuracy: 0.4994
17344/25000 [===================>..........] - ETA: 27s - loss: 7.6737 - accuracy: 0.4995
17376/25000 [===================>..........] - ETA: 27s - loss: 7.6746 - accuracy: 0.4995
17408/25000 [===================>..........] - ETA: 27s - loss: 7.6763 - accuracy: 0.4994
17440/25000 [===================>..........] - ETA: 27s - loss: 7.6745 - accuracy: 0.4995
17472/25000 [===================>..........] - ETA: 27s - loss: 7.6772 - accuracy: 0.4993
17504/25000 [====================>.........] - ETA: 27s - loss: 7.6763 - accuracy: 0.4994
17536/25000 [====================>.........] - ETA: 27s - loss: 7.6789 - accuracy: 0.4992
17568/25000 [====================>.........] - ETA: 26s - loss: 7.6849 - accuracy: 0.4988
17600/25000 [====================>.........] - ETA: 26s - loss: 7.6823 - accuracy: 0.4990
17632/25000 [====================>.........] - ETA: 26s - loss: 7.6823 - accuracy: 0.4990
17664/25000 [====================>.........] - ETA: 26s - loss: 7.6866 - accuracy: 0.4987
17696/25000 [====================>.........] - ETA: 26s - loss: 7.6874 - accuracy: 0.4986
17728/25000 [====================>.........] - ETA: 26s - loss: 7.6908 - accuracy: 0.4984
17760/25000 [====================>.........] - ETA: 26s - loss: 7.6942 - accuracy: 0.4982
17792/25000 [====================>.........] - ETA: 26s - loss: 7.6942 - accuracy: 0.4982
17824/25000 [====================>.........] - ETA: 25s - loss: 7.6941 - accuracy: 0.4982
17856/25000 [====================>.........] - ETA: 25s - loss: 7.6932 - accuracy: 0.4983
17888/25000 [====================>.........] - ETA: 25s - loss: 7.6932 - accuracy: 0.4983
17920/25000 [====================>.........] - ETA: 25s - loss: 7.6906 - accuracy: 0.4984
17952/25000 [====================>.........] - ETA: 25s - loss: 7.6922 - accuracy: 0.4983
17984/25000 [====================>.........] - ETA: 25s - loss: 7.6905 - accuracy: 0.4984
18016/25000 [====================>.........] - ETA: 25s - loss: 7.6905 - accuracy: 0.4984
18048/25000 [====================>.........] - ETA: 25s - loss: 7.6887 - accuracy: 0.4986
18080/25000 [====================>.........] - ETA: 25s - loss: 7.6895 - accuracy: 0.4985
18112/25000 [====================>.........] - ETA: 24s - loss: 7.6920 - accuracy: 0.4983
18144/25000 [====================>.........] - ETA: 24s - loss: 7.6954 - accuracy: 0.4981
18176/25000 [====================>.........] - ETA: 24s - loss: 7.6945 - accuracy: 0.4982
18208/25000 [====================>.........] - ETA: 24s - loss: 7.6961 - accuracy: 0.4981
18240/25000 [====================>.........] - ETA: 24s - loss: 7.6977 - accuracy: 0.4980
18272/25000 [====================>.........] - ETA: 24s - loss: 7.7010 - accuracy: 0.4978
18304/25000 [====================>.........] - ETA: 24s - loss: 7.7010 - accuracy: 0.4978
18336/25000 [=====================>........] - ETA: 24s - loss: 7.7017 - accuracy: 0.4977
18368/25000 [=====================>........] - ETA: 23s - loss: 7.6992 - accuracy: 0.4979
18400/25000 [=====================>........] - ETA: 23s - loss: 7.6991 - accuracy: 0.4979
18432/25000 [=====================>........] - ETA: 23s - loss: 7.6991 - accuracy: 0.4979
18464/25000 [=====================>........] - ETA: 23s - loss: 7.6982 - accuracy: 0.4979
18496/25000 [=====================>........] - ETA: 23s - loss: 7.6965 - accuracy: 0.4981
18528/25000 [=====================>........] - ETA: 23s - loss: 7.6956 - accuracy: 0.4981
18560/25000 [=====================>........] - ETA: 23s - loss: 7.6939 - accuracy: 0.4982
18592/25000 [=====================>........] - ETA: 23s - loss: 7.6947 - accuracy: 0.4982
18624/25000 [=====================>........] - ETA: 23s - loss: 7.7020 - accuracy: 0.4977
18656/25000 [=====================>........] - ETA: 22s - loss: 7.7028 - accuracy: 0.4976
18688/25000 [=====================>........] - ETA: 22s - loss: 7.6994 - accuracy: 0.4979
18720/25000 [=====================>........] - ETA: 22s - loss: 7.7027 - accuracy: 0.4976
18752/25000 [=====================>........] - ETA: 22s - loss: 7.7026 - accuracy: 0.4977
18784/25000 [=====================>........] - ETA: 22s - loss: 7.7042 - accuracy: 0.4976
18816/25000 [=====================>........] - ETA: 22s - loss: 7.7049 - accuracy: 0.4975
18848/25000 [=====================>........] - ETA: 22s - loss: 7.7049 - accuracy: 0.4975
18880/25000 [=====================>........] - ETA: 22s - loss: 7.7048 - accuracy: 0.4975
18912/25000 [=====================>........] - ETA: 21s - loss: 7.7055 - accuracy: 0.4975
18944/25000 [=====================>........] - ETA: 21s - loss: 7.7030 - accuracy: 0.4976
18976/25000 [=====================>........] - ETA: 21s - loss: 7.7046 - accuracy: 0.4975
19008/25000 [=====================>........] - ETA: 21s - loss: 7.7053 - accuracy: 0.4975
19040/25000 [=====================>........] - ETA: 21s - loss: 7.7077 - accuracy: 0.4973
19072/25000 [=====================>........] - ETA: 21s - loss: 7.7100 - accuracy: 0.4972
19104/25000 [=====================>........] - ETA: 21s - loss: 7.7092 - accuracy: 0.4972
19136/25000 [=====================>........] - ETA: 21s - loss: 7.7075 - accuracy: 0.4973
19168/25000 [======================>.......] - ETA: 21s - loss: 7.7106 - accuracy: 0.4971
19200/25000 [======================>.......] - ETA: 20s - loss: 7.7097 - accuracy: 0.4972
19232/25000 [======================>.......] - ETA: 20s - loss: 7.7089 - accuracy: 0.4972
19264/25000 [======================>.......] - ETA: 20s - loss: 7.7080 - accuracy: 0.4973
19296/25000 [======================>.......] - ETA: 20s - loss: 7.7095 - accuracy: 0.4972
19328/25000 [======================>.......] - ETA: 20s - loss: 7.7095 - accuracy: 0.4972
19360/25000 [======================>.......] - ETA: 20s - loss: 7.7054 - accuracy: 0.4975
19392/25000 [======================>.......] - ETA: 20s - loss: 7.7022 - accuracy: 0.4977
19424/25000 [======================>.......] - ETA: 20s - loss: 7.7021 - accuracy: 0.4977
19456/25000 [======================>.......] - ETA: 20s - loss: 7.7052 - accuracy: 0.4975
19488/25000 [======================>.......] - ETA: 19s - loss: 7.7044 - accuracy: 0.4975
19520/25000 [======================>.......] - ETA: 19s - loss: 7.7028 - accuracy: 0.4976
19552/25000 [======================>.......] - ETA: 19s - loss: 7.7027 - accuracy: 0.4976
19584/25000 [======================>.......] - ETA: 19s - loss: 7.7042 - accuracy: 0.4975
19616/25000 [======================>.......] - ETA: 19s - loss: 7.7026 - accuracy: 0.4977
19648/25000 [======================>.......] - ETA: 19s - loss: 7.7080 - accuracy: 0.4973
19680/25000 [======================>.......] - ETA: 19s - loss: 7.7095 - accuracy: 0.4972
19712/25000 [======================>.......] - ETA: 19s - loss: 7.7125 - accuracy: 0.4970
19744/25000 [======================>.......] - ETA: 18s - loss: 7.7093 - accuracy: 0.4972
19776/25000 [======================>.......] - ETA: 18s - loss: 7.7100 - accuracy: 0.4972
19808/25000 [======================>.......] - ETA: 18s - loss: 7.7100 - accuracy: 0.4972
19840/25000 [======================>.......] - ETA: 18s - loss: 7.7114 - accuracy: 0.4971
19872/25000 [======================>.......] - ETA: 18s - loss: 7.7091 - accuracy: 0.4972
19904/25000 [======================>.......] - ETA: 18s - loss: 7.7098 - accuracy: 0.4972
19936/25000 [======================>.......] - ETA: 18s - loss: 7.7112 - accuracy: 0.4971
19968/25000 [======================>.......] - ETA: 18s - loss: 7.7112 - accuracy: 0.4971
20000/25000 [=======================>......] - ETA: 18s - loss: 7.7111 - accuracy: 0.4971
20032/25000 [=======================>......] - ETA: 17s - loss: 7.7110 - accuracy: 0.4971
20064/25000 [=======================>......] - ETA: 17s - loss: 7.7109 - accuracy: 0.4971
20096/25000 [=======================>......] - ETA: 17s - loss: 7.7124 - accuracy: 0.4970
20128/25000 [=======================>......] - ETA: 17s - loss: 7.7108 - accuracy: 0.4971
20160/25000 [=======================>......] - ETA: 17s - loss: 7.7100 - accuracy: 0.4972
20192/25000 [=======================>......] - ETA: 17s - loss: 7.7091 - accuracy: 0.4972
20224/25000 [=======================>......] - ETA: 17s - loss: 7.7091 - accuracy: 0.4972
20256/25000 [=======================>......] - ETA: 17s - loss: 7.7105 - accuracy: 0.4971
20288/25000 [=======================>......] - ETA: 17s - loss: 7.7089 - accuracy: 0.4972
20320/25000 [=======================>......] - ETA: 16s - loss: 7.7104 - accuracy: 0.4971
20352/25000 [=======================>......] - ETA: 16s - loss: 7.7081 - accuracy: 0.4973
20384/25000 [=======================>......] - ETA: 16s - loss: 7.7087 - accuracy: 0.4973
20416/25000 [=======================>......] - ETA: 16s - loss: 7.7072 - accuracy: 0.4974
20448/25000 [=======================>......] - ETA: 16s - loss: 7.7071 - accuracy: 0.4974
20480/25000 [=======================>......] - ETA: 16s - loss: 7.7078 - accuracy: 0.4973
20512/25000 [=======================>......] - ETA: 16s - loss: 7.7062 - accuracy: 0.4974
20544/25000 [=======================>......] - ETA: 16s - loss: 7.7047 - accuracy: 0.4975
20576/25000 [=======================>......] - ETA: 15s - loss: 7.7054 - accuracy: 0.4975
20608/25000 [=======================>......] - ETA: 15s - loss: 7.7053 - accuracy: 0.4975
20640/25000 [=======================>......] - ETA: 15s - loss: 7.7030 - accuracy: 0.4976
20672/25000 [=======================>......] - ETA: 15s - loss: 7.7037 - accuracy: 0.4976
20704/25000 [=======================>......] - ETA: 15s - loss: 7.7059 - accuracy: 0.4974
20736/25000 [=======================>......] - ETA: 15s - loss: 7.7058 - accuracy: 0.4974
20768/25000 [=======================>......] - ETA: 15s - loss: 7.7072 - accuracy: 0.4974
20800/25000 [=======================>......] - ETA: 15s - loss: 7.7050 - accuracy: 0.4975
20832/25000 [=======================>......] - ETA: 15s - loss: 7.7049 - accuracy: 0.4975
20864/25000 [========================>.....] - ETA: 14s - loss: 7.7034 - accuracy: 0.4976
20896/25000 [========================>.....] - ETA: 14s - loss: 7.7040 - accuracy: 0.4976
20928/25000 [========================>.....] - ETA: 14s - loss: 7.7076 - accuracy: 0.4973
20960/25000 [========================>.....] - ETA: 14s - loss: 7.7069 - accuracy: 0.4974
20992/25000 [========================>.....] - ETA: 14s - loss: 7.7053 - accuracy: 0.4975
21024/25000 [========================>.....] - ETA: 14s - loss: 7.7082 - accuracy: 0.4973
21056/25000 [========================>.....] - ETA: 14s - loss: 7.7052 - accuracy: 0.4975
21088/25000 [========================>.....] - ETA: 14s - loss: 7.7044 - accuracy: 0.4975
21120/25000 [========================>.....] - ETA: 13s - loss: 7.7044 - accuracy: 0.4975
21152/25000 [========================>.....] - ETA: 13s - loss: 7.7036 - accuracy: 0.4976
21184/25000 [========================>.....] - ETA: 13s - loss: 7.7014 - accuracy: 0.4977
21216/25000 [========================>.....] - ETA: 13s - loss: 7.7020 - accuracy: 0.4977
21248/25000 [========================>.....] - ETA: 13s - loss: 7.7027 - accuracy: 0.4976
21280/25000 [========================>.....] - ETA: 13s - loss: 7.7026 - accuracy: 0.4977
21312/25000 [========================>.....] - ETA: 13s - loss: 7.7033 - accuracy: 0.4976
21344/25000 [========================>.....] - ETA: 13s - loss: 7.7018 - accuracy: 0.4977
21376/25000 [========================>.....] - ETA: 13s - loss: 7.6989 - accuracy: 0.4979
21408/25000 [========================>.....] - ETA: 12s - loss: 7.6967 - accuracy: 0.4980
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6917 - accuracy: 0.4984
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6888 - accuracy: 0.4986
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6837 - accuracy: 0.4989
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6816 - accuracy: 0.4990
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6823 - accuracy: 0.4990
21600/25000 [========================>.....] - ETA: 12s - loss: 7.6801 - accuracy: 0.4991
21632/25000 [========================>.....] - ETA: 12s - loss: 7.6780 - accuracy: 0.4993
21664/25000 [========================>.....] - ETA: 12s - loss: 7.6829 - accuracy: 0.4989
21696/25000 [=========================>....] - ETA: 11s - loss: 7.6836 - accuracy: 0.4989
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6829 - accuracy: 0.4989
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6835 - accuracy: 0.4989
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6779 - accuracy: 0.4993
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6779 - accuracy: 0.4993
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6757 - accuracy: 0.4994
21888/25000 [=========================>....] - ETA: 11s - loss: 7.6799 - accuracy: 0.4991
21920/25000 [=========================>....] - ETA: 11s - loss: 7.6841 - accuracy: 0.4989
21952/25000 [=========================>....] - ETA: 10s - loss: 7.6834 - accuracy: 0.4989
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6820 - accuracy: 0.4990
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6826 - accuracy: 0.4990
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6840 - accuracy: 0.4989
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6826 - accuracy: 0.4990
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6805 - accuracy: 0.4991
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6770 - accuracy: 0.4993
22176/25000 [=========================>....] - ETA: 10s - loss: 7.6777 - accuracy: 0.4993
22208/25000 [=========================>....] - ETA: 10s - loss: 7.6797 - accuracy: 0.4991
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6804 - accuracy: 0.4991 
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6783 - accuracy: 0.4992
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6769 - accuracy: 0.4993
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6769 - accuracy: 0.4993
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6769 - accuracy: 0.4993
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6776 - accuracy: 0.4993
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6789 - accuracy: 0.4992
22464/25000 [=========================>....] - ETA: 9s - loss: 7.6789 - accuracy: 0.4992
22496/25000 [=========================>....] - ETA: 9s - loss: 7.6755 - accuracy: 0.4994
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6734 - accuracy: 0.4996
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6755 - accuracy: 0.4994
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6748 - accuracy: 0.4995
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6734 - accuracy: 0.4996
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6741 - accuracy: 0.4995
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6734 - accuracy: 0.4996
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6747 - accuracy: 0.4995
22752/25000 [==========================>...] - ETA: 8s - loss: 7.6734 - accuracy: 0.4996
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6760 - accuracy: 0.4994
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6747 - accuracy: 0.4995
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6774 - accuracy: 0.4993
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6787 - accuracy: 0.4992
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6773 - accuracy: 0.4993
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6773 - accuracy: 0.4993
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6786 - accuracy: 0.4992
23008/25000 [==========================>...] - ETA: 7s - loss: 7.6759 - accuracy: 0.4994
23040/25000 [==========================>...] - ETA: 7s - loss: 7.6713 - accuracy: 0.4997
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6699 - accuracy: 0.4998
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6713 - accuracy: 0.4997
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6706 - accuracy: 0.4997
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6706 - accuracy: 0.4997
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6699 - accuracy: 0.4998
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6699 - accuracy: 0.4998
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6706 - accuracy: 0.4997
23296/25000 [==========================>...] - ETA: 6s - loss: 7.6739 - accuracy: 0.4995
23328/25000 [==========================>...] - ETA: 6s - loss: 7.6791 - accuracy: 0.4992
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6797 - accuracy: 0.4991
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6824 - accuracy: 0.4990
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6830 - accuracy: 0.4989
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6823 - accuracy: 0.4990
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6856 - accuracy: 0.4988
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6862 - accuracy: 0.4987
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6855 - accuracy: 0.4988
23584/25000 [===========================>..] - ETA: 5s - loss: 7.6822 - accuracy: 0.4990
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6796 - accuracy: 0.4992
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6835 - accuracy: 0.4989
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6848 - accuracy: 0.4988
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6847 - accuracy: 0.4988
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6808 - accuracy: 0.4991
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6802 - accuracy: 0.4991
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6795 - accuracy: 0.4992
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6756 - accuracy: 0.4994
23872/25000 [===========================>..] - ETA: 4s - loss: 7.6756 - accuracy: 0.4994
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6762 - accuracy: 0.4994
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6730 - accuracy: 0.4996
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6724 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6717 - accuracy: 0.4997
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6698 - accuracy: 0.4998
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
24160/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
24192/25000 [============================>.] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24224/25000 [============================>.] - ETA: 2s - loss: 7.6692 - accuracy: 0.4998
24256/25000 [============================>.] - ETA: 2s - loss: 7.6704 - accuracy: 0.4998
24288/25000 [============================>.] - ETA: 2s - loss: 7.6691 - accuracy: 0.4998
24320/25000 [============================>.] - ETA: 2s - loss: 7.6672 - accuracy: 0.5000
24352/25000 [============================>.] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24384/25000 [============================>.] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24416/25000 [============================>.] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24448/25000 [============================>.] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
24480/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24512/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24544/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24576/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24608/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24640/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24672/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24704/25000 [============================>.] - ETA: 1s - loss: 7.6629 - accuracy: 0.5002
24736/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24768/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24800/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24832/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24864/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24896/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24992/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
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
