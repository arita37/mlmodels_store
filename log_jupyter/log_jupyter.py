
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
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
 40%|████      | 2/5 [00:46<01:09, 23.29s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.4115915410773593, 'embedding_size_factor': 0.9982443035427839, 'layers.choice': 0, 'learning_rate': 0.0012846690914539343, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 7.825127055566354e-11} and reward: 0.3818
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdaW\x84\x0c\x0f3.X\x15\x00\x00\x00embedding_size_factorq\x03G?\xef\xf1\x9e\t\xa4NDX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?U\x0cJ\xee\xf9yTX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xd5\x82q\x92X\x81\xe9u.' and reward: 0.3818
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdaW\x84\x0c\x0f3.X\x15\x00\x00\x00embedding_size_factorq\x03G?\xef\xf1\x9e\t\xa4NDX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?U\x0cJ\xee\xf9yTX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xd5\x82q\x92X\x81\xe9u.' and reward: 0.3818
 60%|██████    | 3/5 [01:33<01:00, 30.37s/it] 60%|██████    | 3/5 [01:33<01:02, 31.16s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.015870579771110675, 'embedding_size_factor': 0.9037080797364201, 'layers.choice': 0, 'learning_rate': 0.0004275262025792554, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 9.658792765289759e-09} and reward: 0.3788
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\x90@`\x94W\x87\xf1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\xeb-4\xf3*\x10X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?<\x04\xb3\x0e\xea0\xedX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>D\xbd\xfa;\xfd&\x16u.' and reward: 0.3788
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\x90@`\x94W\x87\xf1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\xeb-4\xf3*\x10X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?<\x04\xb3\x0e\xea0\xedX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>D\xbd\xfa;\xfd&\x16u.' and reward: 0.3788
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 141.71357226371765
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.78s of the -24.0s of remaining time.
Ensemble size: 9
Ensemble weights: 
[0.55555556 0.11111111 0.33333333]
	0.3914	 = Validation accuracy score
	0.94s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 144.97s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7effb789d9b0> 

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
 [ 0.04655149  0.02888693  0.0615714   0.00501382  0.00569917 -0.07453486]
 [ 0.21008714  0.02798664 -0.23469928  0.08178039 -0.03203796 -0.14887488]
 [-0.08203344 -0.03239747 -0.16407555  0.11306866  0.23655026  0.06360702]
 [-0.03158426 -0.04173462 -0.07089001 -0.5159812  -0.04174392  0.25128227]
 [ 0.29538631  0.01834104  0.17013898  0.42228481  0.47903964  0.21885738]
 [ 0.21135049  0.17720319  0.47402799  0.01367701 -0.22313513 -0.34312996]
 [ 0.11012827  0.40497413  0.58963829  0.74406642  0.49763009  0.22554512]
 [-0.25714141 -0.18077397  0.05286252 -0.37468287  0.82782304  0.21389796]
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
{'loss': 0.42624923400580883, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-20 14:31:57.058628: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.5136057361960411, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-20 14:31:58.073114: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 1130496/17464789 [>.............................] - ETA: 0s
 5513216/17464789 [========>.....................] - ETA: 0s
13197312/17464789 [=====================>........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-20 14:32:08.310264: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-20 14:32:08.314697: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-20 14:32:08.314849: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561ef22a7020 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-20 14:32:08.314864: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:06 - loss: 9.1041 - accuracy: 0.4062
   64/25000 [..............................] - ETA: 2:35 - loss: 8.1458 - accuracy: 0.4688
   96/25000 [..............................] - ETA: 2:02 - loss: 8.1458 - accuracy: 0.4688
  128/25000 [..............................] - ETA: 1:45 - loss: 7.7864 - accuracy: 0.4922
  160/25000 [..............................] - ETA: 1:36 - loss: 8.2416 - accuracy: 0.4625
  192/25000 [..............................] - ETA: 1:29 - loss: 8.1458 - accuracy: 0.4688
  224/25000 [..............................] - ETA: 1:24 - loss: 8.0089 - accuracy: 0.4777
  256/25000 [..............................] - ETA: 1:20 - loss: 8.3854 - accuracy: 0.4531
  288/25000 [..............................] - ETA: 1:17 - loss: 8.4652 - accuracy: 0.4479
  320/25000 [..............................] - ETA: 1:15 - loss: 8.2416 - accuracy: 0.4625
  352/25000 [..............................] - ETA: 1:14 - loss: 8.2329 - accuracy: 0.4631
  384/25000 [..............................] - ETA: 1:12 - loss: 8.0659 - accuracy: 0.4740
  416/25000 [..............................] - ETA: 1:11 - loss: 8.0721 - accuracy: 0.4736
  448/25000 [..............................] - ETA: 1:10 - loss: 7.9062 - accuracy: 0.4844
  480/25000 [..............................] - ETA: 1:09 - loss: 7.8902 - accuracy: 0.4854
  512/25000 [..............................] - ETA: 1:08 - loss: 7.9960 - accuracy: 0.4785
  544/25000 [..............................] - ETA: 1:07 - loss: 8.1740 - accuracy: 0.4669
  576/25000 [..............................] - ETA: 1:06 - loss: 8.1458 - accuracy: 0.4688
  608/25000 [..............................] - ETA: 1:06 - loss: 8.0449 - accuracy: 0.4753
  640/25000 [..............................] - ETA: 1:05 - loss: 8.0260 - accuracy: 0.4766
  672/25000 [..............................] - ETA: 1:05 - loss: 8.1230 - accuracy: 0.4702
  704/25000 [..............................] - ETA: 1:04 - loss: 8.0804 - accuracy: 0.4730
  736/25000 [..............................] - ETA: 1:04 - loss: 8.0208 - accuracy: 0.4769
  768/25000 [..............................] - ETA: 1:03 - loss: 8.0260 - accuracy: 0.4766
  800/25000 [..............................] - ETA: 1:03 - loss: 8.0116 - accuracy: 0.4775
  832/25000 [..............................] - ETA: 1:02 - loss: 7.9062 - accuracy: 0.4844
  864/25000 [>.............................] - ETA: 1:02 - loss: 7.8618 - accuracy: 0.4873
  896/25000 [>.............................] - ETA: 1:02 - loss: 7.8035 - accuracy: 0.4911
  928/25000 [>.............................] - ETA: 1:01 - loss: 7.7658 - accuracy: 0.4935
  960/25000 [>.............................] - ETA: 1:01 - loss: 7.7465 - accuracy: 0.4948
  992/25000 [>.............................] - ETA: 1:01 - loss: 7.7594 - accuracy: 0.4940
 1024/25000 [>.............................] - ETA: 1:01 - loss: 7.7864 - accuracy: 0.4922
 1056/25000 [>.............................] - ETA: 1:00 - loss: 7.7683 - accuracy: 0.4934
 1088/25000 [>.............................] - ETA: 1:00 - loss: 7.7935 - accuracy: 0.4917
 1120/25000 [>.............................] - ETA: 1:00 - loss: 7.8309 - accuracy: 0.4893
 1152/25000 [>.............................] - ETA: 1:00 - loss: 7.7997 - accuracy: 0.4913
 1184/25000 [>.............................] - ETA: 59s - loss: 7.8609 - accuracy: 0.4873 
 1216/25000 [>.............................] - ETA: 59s - loss: 7.8305 - accuracy: 0.4893
 1248/25000 [>.............................] - ETA: 59s - loss: 7.8141 - accuracy: 0.4904
 1280/25000 [>.............................] - ETA: 59s - loss: 7.7744 - accuracy: 0.4930
 1312/25000 [>.............................] - ETA: 59s - loss: 7.7835 - accuracy: 0.4924
 1344/25000 [>.............................] - ETA: 58s - loss: 7.8377 - accuracy: 0.4888
 1376/25000 [>.............................] - ETA: 58s - loss: 7.7781 - accuracy: 0.4927
 1408/25000 [>.............................] - ETA: 58s - loss: 7.8300 - accuracy: 0.4893
 1440/25000 [>.............................] - ETA: 58s - loss: 7.7944 - accuracy: 0.4917
 1472/25000 [>.............................] - ETA: 58s - loss: 7.7916 - accuracy: 0.4918
 1504/25000 [>.............................] - ETA: 58s - loss: 7.7686 - accuracy: 0.4934
 1536/25000 [>.............................] - ETA: 57s - loss: 7.7664 - accuracy: 0.4935
 1568/25000 [>.............................] - ETA: 57s - loss: 7.7546 - accuracy: 0.4943
 1600/25000 [>.............................] - ETA: 57s - loss: 7.7433 - accuracy: 0.4950
 1632/25000 [>.............................] - ETA: 57s - loss: 7.7136 - accuracy: 0.4969
 1664/25000 [>.............................] - ETA: 56s - loss: 7.7311 - accuracy: 0.4958
 1696/25000 [=>............................] - ETA: 56s - loss: 7.7480 - accuracy: 0.4947
 1728/25000 [=>............................] - ETA: 56s - loss: 7.7642 - accuracy: 0.4936
 1760/25000 [=>............................] - ETA: 56s - loss: 7.7450 - accuracy: 0.4949
 1792/25000 [=>............................] - ETA: 56s - loss: 7.7351 - accuracy: 0.4955
 1824/25000 [=>............................] - ETA: 56s - loss: 7.7255 - accuracy: 0.4962
 1856/25000 [=>............................] - ETA: 55s - loss: 7.7658 - accuracy: 0.4935
 1888/25000 [=>............................] - ETA: 55s - loss: 7.7560 - accuracy: 0.4942
 1920/25000 [=>............................] - ETA: 55s - loss: 7.7305 - accuracy: 0.4958
 1952/25000 [=>............................] - ETA: 55s - loss: 7.7138 - accuracy: 0.4969
 1984/25000 [=>............................] - ETA: 55s - loss: 7.7130 - accuracy: 0.4970
 2016/25000 [=>............................] - ETA: 55s - loss: 7.6742 - accuracy: 0.4995
 2048/25000 [=>............................] - ETA: 55s - loss: 7.6816 - accuracy: 0.4990
 2080/25000 [=>............................] - ETA: 54s - loss: 7.6961 - accuracy: 0.4981
 2112/25000 [=>............................] - ETA: 54s - loss: 7.6739 - accuracy: 0.4995
 2144/25000 [=>............................] - ETA: 54s - loss: 7.6595 - accuracy: 0.5005
 2176/25000 [=>............................] - ETA: 54s - loss: 7.6666 - accuracy: 0.5000
 2208/25000 [=>............................] - ETA: 54s - loss: 7.7083 - accuracy: 0.4973
 2240/25000 [=>............................] - ETA: 54s - loss: 7.7351 - accuracy: 0.4955
 2272/25000 [=>............................] - ETA: 54s - loss: 7.7004 - accuracy: 0.4978
 2304/25000 [=>............................] - ETA: 54s - loss: 7.6666 - accuracy: 0.5000
 2336/25000 [=>............................] - ETA: 54s - loss: 7.6338 - accuracy: 0.5021
 2368/25000 [=>............................] - ETA: 53s - loss: 7.6213 - accuracy: 0.5030
 2400/25000 [=>............................] - ETA: 53s - loss: 7.6411 - accuracy: 0.5017
 2432/25000 [=>............................] - ETA: 53s - loss: 7.6288 - accuracy: 0.5025
 2464/25000 [=>............................] - ETA: 53s - loss: 7.6417 - accuracy: 0.5016
 2496/25000 [=>............................] - ETA: 53s - loss: 7.6482 - accuracy: 0.5012
 2528/25000 [==>...........................] - ETA: 53s - loss: 7.6363 - accuracy: 0.5020
 2560/25000 [==>...........................] - ETA: 53s - loss: 7.6367 - accuracy: 0.5020
 2592/25000 [==>...........................] - ETA: 53s - loss: 7.6134 - accuracy: 0.5035
 2624/25000 [==>...........................] - ETA: 53s - loss: 7.6199 - accuracy: 0.5030
 2656/25000 [==>...........................] - ETA: 52s - loss: 7.6262 - accuracy: 0.5026
 2688/25000 [==>...........................] - ETA: 52s - loss: 7.6267 - accuracy: 0.5026
 2720/25000 [==>...........................] - ETA: 52s - loss: 7.6046 - accuracy: 0.5040
 2752/25000 [==>...........................] - ETA: 52s - loss: 7.5942 - accuracy: 0.5047
 2784/25000 [==>...........................] - ETA: 52s - loss: 7.5840 - accuracy: 0.5054
 2816/25000 [==>...........................] - ETA: 52s - loss: 7.5795 - accuracy: 0.5057
 2848/25000 [==>...........................] - ETA: 52s - loss: 7.5966 - accuracy: 0.5046
 2880/25000 [==>...........................] - ETA: 52s - loss: 7.5548 - accuracy: 0.5073
 2912/25000 [==>...........................] - ETA: 51s - loss: 7.5613 - accuracy: 0.5069
 2944/25000 [==>...........................] - ETA: 51s - loss: 7.5625 - accuracy: 0.5068
 2976/25000 [==>...........................] - ETA: 51s - loss: 7.5739 - accuracy: 0.5060
 3008/25000 [==>...........................] - ETA: 51s - loss: 7.5800 - accuracy: 0.5057
 3040/25000 [==>...........................] - ETA: 51s - loss: 7.5758 - accuracy: 0.5059
 3072/25000 [==>...........................] - ETA: 51s - loss: 7.5768 - accuracy: 0.5059
 3104/25000 [==>...........................] - ETA: 51s - loss: 7.5728 - accuracy: 0.5061
 3136/25000 [==>...........................] - ETA: 51s - loss: 7.5786 - accuracy: 0.5057
 3168/25000 [==>...........................] - ETA: 51s - loss: 7.5698 - accuracy: 0.5063
 3200/25000 [==>...........................] - ETA: 51s - loss: 7.5564 - accuracy: 0.5072
 3232/25000 [==>...........................] - ETA: 51s - loss: 7.5717 - accuracy: 0.5062
 3264/25000 [==>...........................] - ETA: 50s - loss: 7.5727 - accuracy: 0.5061
 3296/25000 [==>...........................] - ETA: 50s - loss: 7.5968 - accuracy: 0.5046
 3328/25000 [==>...........................] - ETA: 50s - loss: 7.6205 - accuracy: 0.5030
 3360/25000 [===>..........................] - ETA: 50s - loss: 7.6255 - accuracy: 0.5027
 3392/25000 [===>..........................] - ETA: 50s - loss: 7.6033 - accuracy: 0.5041
 3424/25000 [===>..........................] - ETA: 50s - loss: 7.5994 - accuracy: 0.5044
 3456/25000 [===>..........................] - ETA: 50s - loss: 7.6001 - accuracy: 0.5043
 3488/25000 [===>..........................] - ETA: 50s - loss: 7.5875 - accuracy: 0.5052
 3520/25000 [===>..........................] - ETA: 50s - loss: 7.5926 - accuracy: 0.5048
 3552/25000 [===>..........................] - ETA: 50s - loss: 7.5803 - accuracy: 0.5056
 3584/25000 [===>..........................] - ETA: 50s - loss: 7.5853 - accuracy: 0.5053
 3616/25000 [===>..........................] - ETA: 49s - loss: 7.5818 - accuracy: 0.5055
 3648/25000 [===>..........................] - ETA: 49s - loss: 7.5826 - accuracy: 0.5055
 3680/25000 [===>..........................] - ETA: 49s - loss: 7.5833 - accuracy: 0.5054
 3712/25000 [===>..........................] - ETA: 49s - loss: 7.5881 - accuracy: 0.5051
 3744/25000 [===>..........................] - ETA: 49s - loss: 7.5765 - accuracy: 0.5059
 3776/25000 [===>..........................] - ETA: 49s - loss: 7.5813 - accuracy: 0.5056
 3808/25000 [===>..........................] - ETA: 49s - loss: 7.5861 - accuracy: 0.5053
 3840/25000 [===>..........................] - ETA: 49s - loss: 7.5828 - accuracy: 0.5055
 3872/25000 [===>..........................] - ETA: 49s - loss: 7.5676 - accuracy: 0.5065
 3904/25000 [===>..........................] - ETA: 49s - loss: 7.5763 - accuracy: 0.5059
 3936/25000 [===>..........................] - ETA: 49s - loss: 7.5770 - accuracy: 0.5058
 3968/25000 [===>..........................] - ETA: 48s - loss: 7.5700 - accuracy: 0.5063
 4000/25000 [===>..........................] - ETA: 48s - loss: 7.5708 - accuracy: 0.5063
 4032/25000 [===>..........................] - ETA: 48s - loss: 7.5906 - accuracy: 0.5050
 4064/25000 [===>..........................] - ETA: 48s - loss: 7.5874 - accuracy: 0.5052
 4096/25000 [===>..........................] - ETA: 48s - loss: 7.5880 - accuracy: 0.5051
 4128/25000 [===>..........................] - ETA: 48s - loss: 7.5998 - accuracy: 0.5044
 4160/25000 [===>..........................] - ETA: 48s - loss: 7.6113 - accuracy: 0.5036
 4192/25000 [====>.........................] - ETA: 48s - loss: 7.6008 - accuracy: 0.5043
 4224/25000 [====>.........................] - ETA: 48s - loss: 7.5904 - accuracy: 0.5050
 4256/25000 [====>.........................] - ETA: 48s - loss: 7.5802 - accuracy: 0.5056
 4288/25000 [====>.........................] - ETA: 48s - loss: 7.5880 - accuracy: 0.5051
 4320/25000 [====>.........................] - ETA: 48s - loss: 7.5992 - accuracy: 0.5044
 4352/25000 [====>.........................] - ETA: 48s - loss: 7.5997 - accuracy: 0.5044
 4384/25000 [====>.........................] - ETA: 48s - loss: 7.6072 - accuracy: 0.5039
 4416/25000 [====>.........................] - ETA: 48s - loss: 7.6006 - accuracy: 0.5043
 4448/25000 [====>.........................] - ETA: 47s - loss: 7.6115 - accuracy: 0.5036
 4480/25000 [====>.........................] - ETA: 47s - loss: 7.6290 - accuracy: 0.5025
 4512/25000 [====>.........................] - ETA: 47s - loss: 7.6258 - accuracy: 0.5027
 4544/25000 [====>.........................] - ETA: 47s - loss: 7.6430 - accuracy: 0.5015
 4576/25000 [====>.........................] - ETA: 47s - loss: 7.6398 - accuracy: 0.5017
 4608/25000 [====>.........................] - ETA: 47s - loss: 7.6333 - accuracy: 0.5022
 4640/25000 [====>.........................] - ETA: 47s - loss: 7.6402 - accuracy: 0.5017
 4672/25000 [====>.........................] - ETA: 47s - loss: 7.6338 - accuracy: 0.5021
 4704/25000 [====>.........................] - ETA: 47s - loss: 7.6373 - accuracy: 0.5019
 4736/25000 [====>.........................] - ETA: 47s - loss: 7.6278 - accuracy: 0.5025
 4768/25000 [====>.........................] - ETA: 47s - loss: 7.6087 - accuracy: 0.5038
 4800/25000 [====>.........................] - ETA: 47s - loss: 7.5868 - accuracy: 0.5052
 4832/25000 [====>.........................] - ETA: 46s - loss: 7.5809 - accuracy: 0.5056
 4864/25000 [====>.........................] - ETA: 46s - loss: 7.5847 - accuracy: 0.5053
 4896/25000 [====>.........................] - ETA: 46s - loss: 7.5727 - accuracy: 0.5061
 4928/25000 [====>.........................] - ETA: 46s - loss: 7.5733 - accuracy: 0.5061
 4960/25000 [====>.........................] - ETA: 46s - loss: 7.5708 - accuracy: 0.5063
 4992/25000 [====>.........................] - ETA: 46s - loss: 7.5622 - accuracy: 0.5068
 5024/25000 [=====>........................] - ETA: 46s - loss: 7.5690 - accuracy: 0.5064
 5056/25000 [=====>........................] - ETA: 46s - loss: 7.5726 - accuracy: 0.5061
 5088/25000 [=====>........................] - ETA: 46s - loss: 7.5792 - accuracy: 0.5057
 5120/25000 [=====>........................] - ETA: 46s - loss: 7.5888 - accuracy: 0.5051
 5152/25000 [=====>........................] - ETA: 46s - loss: 7.5803 - accuracy: 0.5056
 5184/25000 [=====>........................] - ETA: 45s - loss: 7.5661 - accuracy: 0.5066
 5216/25000 [=====>........................] - ETA: 45s - loss: 7.5490 - accuracy: 0.5077
 5248/25000 [=====>........................] - ETA: 45s - loss: 7.5614 - accuracy: 0.5069
 5280/25000 [=====>........................] - ETA: 45s - loss: 7.5563 - accuracy: 0.5072
 5312/25000 [=====>........................] - ETA: 45s - loss: 7.5512 - accuracy: 0.5075
 5344/25000 [=====>........................] - ETA: 45s - loss: 7.5432 - accuracy: 0.5080
 5376/25000 [=====>........................] - ETA: 45s - loss: 7.5440 - accuracy: 0.5080
 5408/25000 [=====>........................] - ETA: 45s - loss: 7.5447 - accuracy: 0.5080
 5440/25000 [=====>........................] - ETA: 45s - loss: 7.5539 - accuracy: 0.5074
 5472/25000 [=====>........................] - ETA: 45s - loss: 7.5517 - accuracy: 0.5075
 5504/25000 [=====>........................] - ETA: 45s - loss: 7.5524 - accuracy: 0.5074
 5536/25000 [=====>........................] - ETA: 45s - loss: 7.5614 - accuracy: 0.5069
 5568/25000 [=====>........................] - ETA: 45s - loss: 7.5730 - accuracy: 0.5061
 5600/25000 [=====>........................] - ETA: 44s - loss: 7.5708 - accuracy: 0.5063
 5632/25000 [=====>........................] - ETA: 44s - loss: 7.5686 - accuracy: 0.5064
 5664/25000 [=====>........................] - ETA: 44s - loss: 7.5583 - accuracy: 0.5071
 5696/25000 [=====>........................] - ETA: 44s - loss: 7.5589 - accuracy: 0.5070
 5728/25000 [=====>........................] - ETA: 44s - loss: 7.5676 - accuracy: 0.5065
 5760/25000 [=====>........................] - ETA: 44s - loss: 7.5655 - accuracy: 0.5066
 5792/25000 [=====>........................] - ETA: 44s - loss: 7.5687 - accuracy: 0.5064
 5824/25000 [=====>........................] - ETA: 44s - loss: 7.5745 - accuracy: 0.5060
 5856/25000 [======>.......................] - ETA: 44s - loss: 7.5750 - accuracy: 0.5060
 5888/25000 [======>.......................] - ETA: 44s - loss: 7.5859 - accuracy: 0.5053
 5920/25000 [======>.......................] - ETA: 44s - loss: 7.5915 - accuracy: 0.5049
 5952/25000 [======>.......................] - ETA: 44s - loss: 7.5945 - accuracy: 0.5047
 5984/25000 [======>.......................] - ETA: 44s - loss: 7.6000 - accuracy: 0.5043
 6016/25000 [======>.......................] - ETA: 43s - loss: 7.6105 - accuracy: 0.5037
 6048/25000 [======>.......................] - ETA: 43s - loss: 7.6184 - accuracy: 0.5031
 6080/25000 [======>.......................] - ETA: 43s - loss: 7.6237 - accuracy: 0.5028
 6112/25000 [======>.......................] - ETA: 43s - loss: 7.6265 - accuracy: 0.5026
 6144/25000 [======>.......................] - ETA: 43s - loss: 7.6292 - accuracy: 0.5024
 6176/25000 [======>.......................] - ETA: 43s - loss: 7.6319 - accuracy: 0.5023
 6208/25000 [======>.......................] - ETA: 43s - loss: 7.6296 - accuracy: 0.5024
 6240/25000 [======>.......................] - ETA: 43s - loss: 7.6298 - accuracy: 0.5024
 6272/25000 [======>.......................] - ETA: 43s - loss: 7.6128 - accuracy: 0.5035
 6304/25000 [======>.......................] - ETA: 43s - loss: 7.6228 - accuracy: 0.5029
 6336/25000 [======>.......................] - ETA: 43s - loss: 7.6279 - accuracy: 0.5025
 6368/25000 [======>.......................] - ETA: 43s - loss: 7.6305 - accuracy: 0.5024
 6400/25000 [======>.......................] - ETA: 42s - loss: 7.6475 - accuracy: 0.5013
 6432/25000 [======>.......................] - ETA: 42s - loss: 7.6309 - accuracy: 0.5023
 6464/25000 [======>.......................] - ETA: 42s - loss: 7.6263 - accuracy: 0.5026
 6496/25000 [======>.......................] - ETA: 42s - loss: 7.6218 - accuracy: 0.5029
 6528/25000 [======>.......................] - ETA: 42s - loss: 7.6196 - accuracy: 0.5031
 6560/25000 [======>.......................] - ETA: 42s - loss: 7.6245 - accuracy: 0.5027
 6592/25000 [======>.......................] - ETA: 42s - loss: 7.6224 - accuracy: 0.5029
 6624/25000 [======>.......................] - ETA: 42s - loss: 7.6180 - accuracy: 0.5032
 6656/25000 [======>.......................] - ETA: 42s - loss: 7.6205 - accuracy: 0.5030
 6688/25000 [=======>......................] - ETA: 42s - loss: 7.6231 - accuracy: 0.5028
 6720/25000 [=======>......................] - ETA: 42s - loss: 7.6301 - accuracy: 0.5024
 6752/25000 [=======>......................] - ETA: 42s - loss: 7.6394 - accuracy: 0.5018
 6784/25000 [=======>......................] - ETA: 42s - loss: 7.6350 - accuracy: 0.5021
 6816/25000 [=======>......................] - ETA: 42s - loss: 7.6396 - accuracy: 0.5018
 6848/25000 [=======>......................] - ETA: 41s - loss: 7.6398 - accuracy: 0.5018
 6880/25000 [=======>......................] - ETA: 41s - loss: 7.6310 - accuracy: 0.5023
 6912/25000 [=======>......................] - ETA: 41s - loss: 7.6311 - accuracy: 0.5023
 6944/25000 [=======>......................] - ETA: 41s - loss: 7.6247 - accuracy: 0.5027
 6976/25000 [=======>......................] - ETA: 41s - loss: 7.6161 - accuracy: 0.5033
 7008/25000 [=======>......................] - ETA: 41s - loss: 7.6141 - accuracy: 0.5034
 7040/25000 [=======>......................] - ETA: 41s - loss: 7.6122 - accuracy: 0.5036
 7072/25000 [=======>......................] - ETA: 41s - loss: 7.6081 - accuracy: 0.5038
 7104/25000 [=======>......................] - ETA: 41s - loss: 7.6019 - accuracy: 0.5042
 7136/25000 [=======>......................] - ETA: 41s - loss: 7.6086 - accuracy: 0.5038
 7168/25000 [=======>......................] - ETA: 41s - loss: 7.6110 - accuracy: 0.5036
 7200/25000 [=======>......................] - ETA: 41s - loss: 7.6176 - accuracy: 0.5032
 7232/25000 [=======>......................] - ETA: 41s - loss: 7.6136 - accuracy: 0.5035
 7264/25000 [=======>......................] - ETA: 41s - loss: 7.6138 - accuracy: 0.5034
 7296/25000 [=======>......................] - ETA: 40s - loss: 7.6057 - accuracy: 0.5040
 7328/25000 [=======>......................] - ETA: 40s - loss: 7.6059 - accuracy: 0.5040
 7360/25000 [=======>......................] - ETA: 40s - loss: 7.6083 - accuracy: 0.5038
 7392/25000 [=======>......................] - ETA: 40s - loss: 7.6065 - accuracy: 0.5039
 7424/25000 [=======>......................] - ETA: 40s - loss: 7.5964 - accuracy: 0.5046
 7456/25000 [=======>......................] - ETA: 40s - loss: 7.5967 - accuracy: 0.5046
 7488/25000 [=======>......................] - ETA: 40s - loss: 7.5929 - accuracy: 0.5048
 7520/25000 [========>.....................] - ETA: 40s - loss: 7.5851 - accuracy: 0.5053
 7552/25000 [========>.....................] - ETA: 40s - loss: 7.5834 - accuracy: 0.5054
 7584/25000 [========>.....................] - ETA: 40s - loss: 7.5716 - accuracy: 0.5062
 7616/25000 [========>.....................] - ETA: 40s - loss: 7.5660 - accuracy: 0.5066
 7648/25000 [========>.....................] - ETA: 40s - loss: 7.5644 - accuracy: 0.5067
 7680/25000 [========>.....................] - ETA: 39s - loss: 7.5588 - accuracy: 0.5070
 7712/25000 [========>.....................] - ETA: 39s - loss: 7.5652 - accuracy: 0.5066
 7744/25000 [========>.....................] - ETA: 39s - loss: 7.5637 - accuracy: 0.5067
 7776/25000 [========>.....................] - ETA: 39s - loss: 7.5641 - accuracy: 0.5067
 7808/25000 [========>.....................] - ETA: 39s - loss: 7.5625 - accuracy: 0.5068
 7840/25000 [========>.....................] - ETA: 39s - loss: 7.5669 - accuracy: 0.5065
 7872/25000 [========>.....................] - ETA: 39s - loss: 7.5692 - accuracy: 0.5064
 7904/25000 [========>.....................] - ETA: 39s - loss: 7.5793 - accuracy: 0.5057
 7936/25000 [========>.....................] - ETA: 39s - loss: 7.5739 - accuracy: 0.5060
 7968/25000 [========>.....................] - ETA: 39s - loss: 7.5685 - accuracy: 0.5064
 8000/25000 [========>.....................] - ETA: 39s - loss: 7.5746 - accuracy: 0.5060
 8032/25000 [========>.....................] - ETA: 39s - loss: 7.5750 - accuracy: 0.5060
 8064/25000 [========>.....................] - ETA: 39s - loss: 7.5830 - accuracy: 0.5055
 8096/25000 [========>.....................] - ETA: 38s - loss: 7.5890 - accuracy: 0.5051
 8128/25000 [========>.....................] - ETA: 38s - loss: 7.5968 - accuracy: 0.5046
 8160/25000 [========>.....................] - ETA: 38s - loss: 7.5933 - accuracy: 0.5048
 8192/25000 [========>.....................] - ETA: 38s - loss: 7.5918 - accuracy: 0.5049
 8224/25000 [========>.....................] - ETA: 38s - loss: 7.5976 - accuracy: 0.5045
 8256/25000 [========>.....................] - ETA: 38s - loss: 7.6016 - accuracy: 0.5042
 8288/25000 [========>.....................] - ETA: 38s - loss: 7.6111 - accuracy: 0.5036
 8320/25000 [========>.....................] - ETA: 38s - loss: 7.6040 - accuracy: 0.5041
 8352/25000 [=========>....................] - ETA: 38s - loss: 7.6079 - accuracy: 0.5038
 8384/25000 [=========>....................] - ETA: 38s - loss: 7.6118 - accuracy: 0.5036
 8416/25000 [=========>....................] - ETA: 38s - loss: 7.6101 - accuracy: 0.5037
 8448/25000 [=========>....................] - ETA: 38s - loss: 7.6122 - accuracy: 0.5036
 8480/25000 [=========>....................] - ETA: 38s - loss: 7.6160 - accuracy: 0.5033
 8512/25000 [=========>....................] - ETA: 37s - loss: 7.6126 - accuracy: 0.5035
 8544/25000 [=========>....................] - ETA: 37s - loss: 7.6092 - accuracy: 0.5037
 8576/25000 [=========>....................] - ETA: 37s - loss: 7.6112 - accuracy: 0.5036
 8608/25000 [=========>....................] - ETA: 37s - loss: 7.6114 - accuracy: 0.5036
 8640/25000 [=========>....................] - ETA: 37s - loss: 7.6152 - accuracy: 0.5034
 8672/25000 [=========>....................] - ETA: 37s - loss: 7.6189 - accuracy: 0.5031
 8704/25000 [=========>....................] - ETA: 37s - loss: 7.6226 - accuracy: 0.5029
 8736/25000 [=========>....................] - ETA: 37s - loss: 7.6192 - accuracy: 0.5031
 8768/25000 [=========>....................] - ETA: 37s - loss: 7.6177 - accuracy: 0.5032
 8800/25000 [=========>....................] - ETA: 37s - loss: 7.6178 - accuracy: 0.5032
 8832/25000 [=========>....................] - ETA: 37s - loss: 7.6145 - accuracy: 0.5034
 8864/25000 [=========>....................] - ETA: 37s - loss: 7.6216 - accuracy: 0.5029
 8896/25000 [=========>....................] - ETA: 37s - loss: 7.6201 - accuracy: 0.5030
 8928/25000 [=========>....................] - ETA: 37s - loss: 7.6168 - accuracy: 0.5032
 8960/25000 [=========>....................] - ETA: 36s - loss: 7.6204 - accuracy: 0.5030
 8992/25000 [=========>....................] - ETA: 36s - loss: 7.6240 - accuracy: 0.5028
 9024/25000 [=========>....................] - ETA: 36s - loss: 7.6224 - accuracy: 0.5029
 9056/25000 [=========>....................] - ETA: 36s - loss: 7.6209 - accuracy: 0.5030
 9088/25000 [=========>....................] - ETA: 36s - loss: 7.6177 - accuracy: 0.5032
 9120/25000 [=========>....................] - ETA: 36s - loss: 7.6212 - accuracy: 0.5030
 9152/25000 [=========>....................] - ETA: 36s - loss: 7.6197 - accuracy: 0.5031
 9184/25000 [==========>...................] - ETA: 36s - loss: 7.6232 - accuracy: 0.5028
 9216/25000 [==========>...................] - ETA: 36s - loss: 7.6250 - accuracy: 0.5027
 9248/25000 [==========>...................] - ETA: 36s - loss: 7.6285 - accuracy: 0.5025
 9280/25000 [==========>...................] - ETA: 36s - loss: 7.6270 - accuracy: 0.5026
 9312/25000 [==========>...................] - ETA: 36s - loss: 7.6304 - accuracy: 0.5024
 9344/25000 [==========>...................] - ETA: 35s - loss: 7.6322 - accuracy: 0.5022
 9376/25000 [==========>...................] - ETA: 35s - loss: 7.6372 - accuracy: 0.5019
 9408/25000 [==========>...................] - ETA: 35s - loss: 7.6389 - accuracy: 0.5018
 9440/25000 [==========>...................] - ETA: 35s - loss: 7.6439 - accuracy: 0.5015
 9472/25000 [==========>...................] - ETA: 35s - loss: 7.6456 - accuracy: 0.5014
 9504/25000 [==========>...................] - ETA: 35s - loss: 7.6440 - accuracy: 0.5015
 9536/25000 [==========>...................] - ETA: 35s - loss: 7.6393 - accuracy: 0.5018
 9568/25000 [==========>...................] - ETA: 35s - loss: 7.6410 - accuracy: 0.5017
 9600/25000 [==========>...................] - ETA: 35s - loss: 7.6443 - accuracy: 0.5015
 9632/25000 [==========>...................] - ETA: 35s - loss: 7.6507 - accuracy: 0.5010
 9664/25000 [==========>...................] - ETA: 35s - loss: 7.6428 - accuracy: 0.5016
 9696/25000 [==========>...................] - ETA: 35s - loss: 7.6413 - accuracy: 0.5017
 9728/25000 [==========>...................] - ETA: 35s - loss: 7.6367 - accuracy: 0.5020
 9760/25000 [==========>...................] - ETA: 35s - loss: 7.6399 - accuracy: 0.5017
 9792/25000 [==========>...................] - ETA: 34s - loss: 7.6322 - accuracy: 0.5022
 9824/25000 [==========>...................] - ETA: 34s - loss: 7.6307 - accuracy: 0.5023
 9856/25000 [==========>...................] - ETA: 34s - loss: 7.6215 - accuracy: 0.5029
 9888/25000 [==========>...................] - ETA: 34s - loss: 7.6248 - accuracy: 0.5027
 9920/25000 [==========>...................] - ETA: 34s - loss: 7.6249 - accuracy: 0.5027
 9952/25000 [==========>...................] - ETA: 34s - loss: 7.6312 - accuracy: 0.5023
 9984/25000 [==========>...................] - ETA: 34s - loss: 7.6313 - accuracy: 0.5023
10016/25000 [===========>..................] - ETA: 34s - loss: 7.6329 - accuracy: 0.5022
10048/25000 [===========>..................] - ETA: 34s - loss: 7.6285 - accuracy: 0.5025
10080/25000 [===========>..................] - ETA: 34s - loss: 7.6286 - accuracy: 0.5025
10112/25000 [===========>..................] - ETA: 34s - loss: 7.6211 - accuracy: 0.5030
10144/25000 [===========>..................] - ETA: 34s - loss: 7.6198 - accuracy: 0.5031
10176/25000 [===========>..................] - ETA: 34s - loss: 7.6199 - accuracy: 0.5030
10208/25000 [===========>..................] - ETA: 33s - loss: 7.6140 - accuracy: 0.5034
10240/25000 [===========>..................] - ETA: 33s - loss: 7.6127 - accuracy: 0.5035
10272/25000 [===========>..................] - ETA: 33s - loss: 7.6189 - accuracy: 0.5031
10304/25000 [===========>..................] - ETA: 33s - loss: 7.6205 - accuracy: 0.5030
10336/25000 [===========>..................] - ETA: 33s - loss: 7.6177 - accuracy: 0.5032
10368/25000 [===========>..................] - ETA: 33s - loss: 7.6252 - accuracy: 0.5027
10400/25000 [===========>..................] - ETA: 33s - loss: 7.6224 - accuracy: 0.5029
10432/25000 [===========>..................] - ETA: 33s - loss: 7.6181 - accuracy: 0.5032
10464/25000 [===========>..................] - ETA: 33s - loss: 7.6197 - accuracy: 0.5031
10496/25000 [===========>..................] - ETA: 33s - loss: 7.6126 - accuracy: 0.5035
10528/25000 [===========>..................] - ETA: 33s - loss: 7.6142 - accuracy: 0.5034
10560/25000 [===========>..................] - ETA: 33s - loss: 7.6114 - accuracy: 0.5036
10592/25000 [===========>..................] - ETA: 33s - loss: 7.6160 - accuracy: 0.5033
10624/25000 [===========>..................] - ETA: 33s - loss: 7.6219 - accuracy: 0.5029
10656/25000 [===========>..................] - ETA: 32s - loss: 7.6206 - accuracy: 0.5030
10688/25000 [===========>..................] - ETA: 32s - loss: 7.6236 - accuracy: 0.5028
10720/25000 [===========>..................] - ETA: 32s - loss: 7.6251 - accuracy: 0.5027
10752/25000 [===========>..................] - ETA: 32s - loss: 7.6310 - accuracy: 0.5023
10784/25000 [===========>..................] - ETA: 32s - loss: 7.6325 - accuracy: 0.5022
10816/25000 [===========>..................] - ETA: 32s - loss: 7.6312 - accuracy: 0.5023
10848/25000 [============>.................] - ETA: 32s - loss: 7.6313 - accuracy: 0.5023
10880/25000 [============>.................] - ETA: 32s - loss: 7.6300 - accuracy: 0.5024
10912/25000 [============>.................] - ETA: 32s - loss: 7.6329 - accuracy: 0.5022
10944/25000 [============>.................] - ETA: 32s - loss: 7.6330 - accuracy: 0.5022
10976/25000 [============>.................] - ETA: 32s - loss: 7.6303 - accuracy: 0.5024
11008/25000 [============>.................] - ETA: 32s - loss: 7.6332 - accuracy: 0.5022
11040/25000 [============>.................] - ETA: 32s - loss: 7.6319 - accuracy: 0.5023
11072/25000 [============>.................] - ETA: 31s - loss: 7.6265 - accuracy: 0.5026
11104/25000 [============>.................] - ETA: 31s - loss: 7.6252 - accuracy: 0.5027
11136/25000 [============>.................] - ETA: 31s - loss: 7.6212 - accuracy: 0.5030
11168/25000 [============>.................] - ETA: 31s - loss: 7.6241 - accuracy: 0.5028
11200/25000 [============>.................] - ETA: 31s - loss: 7.6242 - accuracy: 0.5028
11232/25000 [============>.................] - ETA: 31s - loss: 7.6243 - accuracy: 0.5028
11264/25000 [============>.................] - ETA: 31s - loss: 7.6258 - accuracy: 0.5027
11296/25000 [============>.................] - ETA: 31s - loss: 7.6259 - accuracy: 0.5027
11328/25000 [============>.................] - ETA: 31s - loss: 7.6260 - accuracy: 0.5026
11360/25000 [============>.................] - ETA: 31s - loss: 7.6234 - accuracy: 0.5028
11392/25000 [============>.................] - ETA: 31s - loss: 7.6249 - accuracy: 0.5027
11424/25000 [============>.................] - ETA: 31s - loss: 7.6237 - accuracy: 0.5028
11456/25000 [============>.................] - ETA: 31s - loss: 7.6238 - accuracy: 0.5028
11488/25000 [============>.................] - ETA: 30s - loss: 7.6306 - accuracy: 0.5024
11520/25000 [============>.................] - ETA: 30s - loss: 7.6360 - accuracy: 0.5020
11552/25000 [============>.................] - ETA: 30s - loss: 7.6321 - accuracy: 0.5023
11584/25000 [============>.................] - ETA: 30s - loss: 7.6296 - accuracy: 0.5024
11616/25000 [============>.................] - ETA: 30s - loss: 7.6244 - accuracy: 0.5028
11648/25000 [============>.................] - ETA: 30s - loss: 7.6258 - accuracy: 0.5027
11680/25000 [=============>................] - ETA: 30s - loss: 7.6312 - accuracy: 0.5023
11712/25000 [=============>................] - ETA: 30s - loss: 7.6313 - accuracy: 0.5023
11744/25000 [=============>................] - ETA: 30s - loss: 7.6288 - accuracy: 0.5025
11776/25000 [=============>................] - ETA: 30s - loss: 7.6302 - accuracy: 0.5024
11808/25000 [=============>................] - ETA: 30s - loss: 7.6342 - accuracy: 0.5021
11840/25000 [=============>................] - ETA: 30s - loss: 7.6342 - accuracy: 0.5021
11872/25000 [=============>................] - ETA: 30s - loss: 7.6395 - accuracy: 0.5018
11904/25000 [=============>................] - ETA: 29s - loss: 7.6357 - accuracy: 0.5020
11936/25000 [=============>................] - ETA: 29s - loss: 7.6371 - accuracy: 0.5019
11968/25000 [=============>................] - ETA: 29s - loss: 7.6423 - accuracy: 0.5016
12000/25000 [=============>................] - ETA: 29s - loss: 7.6385 - accuracy: 0.5018
12032/25000 [=============>................] - ETA: 29s - loss: 7.6348 - accuracy: 0.5021
12064/25000 [=============>................] - ETA: 29s - loss: 7.6310 - accuracy: 0.5023
12096/25000 [=============>................] - ETA: 29s - loss: 7.6273 - accuracy: 0.5026
12128/25000 [=============>................] - ETA: 29s - loss: 7.6274 - accuracy: 0.5026
12160/25000 [=============>................] - ETA: 29s - loss: 7.6288 - accuracy: 0.5025
12192/25000 [=============>................] - ETA: 29s - loss: 7.6289 - accuracy: 0.5025
12224/25000 [=============>................] - ETA: 29s - loss: 7.6265 - accuracy: 0.5026
12256/25000 [=============>................] - ETA: 29s - loss: 7.6266 - accuracy: 0.5026
12288/25000 [=============>................] - ETA: 29s - loss: 7.6229 - accuracy: 0.5028
12320/25000 [=============>................] - ETA: 29s - loss: 7.6218 - accuracy: 0.5029
12352/25000 [=============>................] - ETA: 28s - loss: 7.6219 - accuracy: 0.5029
12384/25000 [=============>................] - ETA: 28s - loss: 7.6196 - accuracy: 0.5031
12416/25000 [=============>................] - ETA: 28s - loss: 7.6197 - accuracy: 0.5031
12448/25000 [=============>................] - ETA: 28s - loss: 7.6210 - accuracy: 0.5030
12480/25000 [=============>................] - ETA: 28s - loss: 7.6162 - accuracy: 0.5033
12512/25000 [==============>...............] - ETA: 28s - loss: 7.6164 - accuracy: 0.5033
12544/25000 [==============>...............] - ETA: 28s - loss: 7.6128 - accuracy: 0.5035
12576/25000 [==============>...............] - ETA: 28s - loss: 7.6227 - accuracy: 0.5029
12608/25000 [==============>...............] - ETA: 28s - loss: 7.6192 - accuracy: 0.5031
12640/25000 [==============>...............] - ETA: 28s - loss: 7.6217 - accuracy: 0.5029
12672/25000 [==============>...............] - ETA: 28s - loss: 7.6194 - accuracy: 0.5031
12704/25000 [==============>...............] - ETA: 28s - loss: 7.6171 - accuracy: 0.5032
12736/25000 [==============>...............] - ETA: 28s - loss: 7.6161 - accuracy: 0.5033
12768/25000 [==============>...............] - ETA: 27s - loss: 7.6174 - accuracy: 0.5032
12800/25000 [==============>...............] - ETA: 27s - loss: 7.6175 - accuracy: 0.5032
12832/25000 [==============>...............] - ETA: 27s - loss: 7.6164 - accuracy: 0.5033
12864/25000 [==============>...............] - ETA: 27s - loss: 7.6130 - accuracy: 0.5035
12896/25000 [==============>...............] - ETA: 27s - loss: 7.6167 - accuracy: 0.5033
12928/25000 [==============>...............] - ETA: 27s - loss: 7.6215 - accuracy: 0.5029
12960/25000 [==============>...............] - ETA: 27s - loss: 7.6205 - accuracy: 0.5030
12992/25000 [==============>...............] - ETA: 27s - loss: 7.6159 - accuracy: 0.5033
13024/25000 [==============>...............] - ETA: 27s - loss: 7.6148 - accuracy: 0.5034
13056/25000 [==============>...............] - ETA: 27s - loss: 7.6138 - accuracy: 0.5034
13088/25000 [==============>...............] - ETA: 27s - loss: 7.6162 - accuracy: 0.5033
13120/25000 [==============>...............] - ETA: 27s - loss: 7.6187 - accuracy: 0.5031
13152/25000 [==============>...............] - ETA: 27s - loss: 7.6177 - accuracy: 0.5032
13184/25000 [==============>...............] - ETA: 27s - loss: 7.6166 - accuracy: 0.5033
13216/25000 [==============>...............] - ETA: 26s - loss: 7.6179 - accuracy: 0.5032
13248/25000 [==============>...............] - ETA: 26s - loss: 7.6203 - accuracy: 0.5030
13280/25000 [==============>...............] - ETA: 26s - loss: 7.6227 - accuracy: 0.5029
13312/25000 [==============>...............] - ETA: 26s - loss: 7.6217 - accuracy: 0.5029
13344/25000 [===============>..............] - ETA: 26s - loss: 7.6230 - accuracy: 0.5028
13376/25000 [===============>..............] - ETA: 26s - loss: 7.6173 - accuracy: 0.5032
13408/25000 [===============>..............] - ETA: 26s - loss: 7.6197 - accuracy: 0.5031
13440/25000 [===============>..............] - ETA: 26s - loss: 7.6221 - accuracy: 0.5029
13472/25000 [===============>..............] - ETA: 26s - loss: 7.6211 - accuracy: 0.5030
13504/25000 [===============>..............] - ETA: 26s - loss: 7.6178 - accuracy: 0.5032
13536/25000 [===============>..............] - ETA: 26s - loss: 7.6145 - accuracy: 0.5034
13568/25000 [===============>..............] - ETA: 26s - loss: 7.6135 - accuracy: 0.5035
13600/25000 [===============>..............] - ETA: 26s - loss: 7.6148 - accuracy: 0.5034
13632/25000 [===============>..............] - ETA: 25s - loss: 7.6149 - accuracy: 0.5034
13664/25000 [===============>..............] - ETA: 25s - loss: 7.6128 - accuracy: 0.5035
13696/25000 [===============>..............] - ETA: 25s - loss: 7.6095 - accuracy: 0.5037
13728/25000 [===============>..............] - ETA: 25s - loss: 7.6164 - accuracy: 0.5033
13760/25000 [===============>..............] - ETA: 25s - loss: 7.6165 - accuracy: 0.5033
13792/25000 [===============>..............] - ETA: 25s - loss: 7.6155 - accuracy: 0.5033
13824/25000 [===============>..............] - ETA: 25s - loss: 7.6178 - accuracy: 0.5032
13856/25000 [===============>..............] - ETA: 25s - loss: 7.6246 - accuracy: 0.5027
13888/25000 [===============>..............] - ETA: 25s - loss: 7.6225 - accuracy: 0.5029
13920/25000 [===============>..............] - ETA: 25s - loss: 7.6226 - accuracy: 0.5029
13952/25000 [===============>..............] - ETA: 25s - loss: 7.6216 - accuracy: 0.5029
13984/25000 [===============>..............] - ETA: 25s - loss: 7.6151 - accuracy: 0.5034
14016/25000 [===============>..............] - ETA: 25s - loss: 7.6152 - accuracy: 0.5034
14048/25000 [===============>..............] - ETA: 25s - loss: 7.6175 - accuracy: 0.5032
14080/25000 [===============>..............] - ETA: 24s - loss: 7.6176 - accuracy: 0.5032
14112/25000 [===============>..............] - ETA: 24s - loss: 7.6156 - accuracy: 0.5033
14144/25000 [===============>..............] - ETA: 24s - loss: 7.6222 - accuracy: 0.5029
14176/25000 [================>.............] - ETA: 24s - loss: 7.6212 - accuracy: 0.5030
14208/25000 [================>.............] - ETA: 24s - loss: 7.6202 - accuracy: 0.5030
14240/25000 [================>.............] - ETA: 24s - loss: 7.6203 - accuracy: 0.5030
14272/25000 [================>.............] - ETA: 24s - loss: 7.6193 - accuracy: 0.5031
14304/25000 [================>.............] - ETA: 24s - loss: 7.6152 - accuracy: 0.5034
14336/25000 [================>.............] - ETA: 24s - loss: 7.6153 - accuracy: 0.5033
14368/25000 [================>.............] - ETA: 24s - loss: 7.6186 - accuracy: 0.5031
14400/25000 [================>.............] - ETA: 24s - loss: 7.6198 - accuracy: 0.5031
14432/25000 [================>.............] - ETA: 24s - loss: 7.6177 - accuracy: 0.5032
14464/25000 [================>.............] - ETA: 24s - loss: 7.6210 - accuracy: 0.5030
14496/25000 [================>.............] - ETA: 24s - loss: 7.6285 - accuracy: 0.5025
14528/25000 [================>.............] - ETA: 23s - loss: 7.6297 - accuracy: 0.5024
14560/25000 [================>.............] - ETA: 23s - loss: 7.6340 - accuracy: 0.5021
14592/25000 [================>.............] - ETA: 23s - loss: 7.6340 - accuracy: 0.5021
14624/25000 [================>.............] - ETA: 23s - loss: 7.6362 - accuracy: 0.5020
14656/25000 [================>.............] - ETA: 23s - loss: 7.6373 - accuracy: 0.5019
14688/25000 [================>.............] - ETA: 23s - loss: 7.6384 - accuracy: 0.5018
14720/25000 [================>.............] - ETA: 23s - loss: 7.6312 - accuracy: 0.5023
14752/25000 [================>.............] - ETA: 23s - loss: 7.6334 - accuracy: 0.5022
14784/25000 [================>.............] - ETA: 23s - loss: 7.6345 - accuracy: 0.5021
14816/25000 [================>.............] - ETA: 23s - loss: 7.6345 - accuracy: 0.5021
14848/25000 [================>.............] - ETA: 23s - loss: 7.6367 - accuracy: 0.5020
14880/25000 [================>.............] - ETA: 23s - loss: 7.6347 - accuracy: 0.5021
14912/25000 [================>.............] - ETA: 23s - loss: 7.6317 - accuracy: 0.5023
14944/25000 [================>.............] - ETA: 22s - loss: 7.6317 - accuracy: 0.5023
14976/25000 [================>.............] - ETA: 22s - loss: 7.6318 - accuracy: 0.5023
15008/25000 [=================>............] - ETA: 22s - loss: 7.6309 - accuracy: 0.5023
15040/25000 [=================>............] - ETA: 22s - loss: 7.6309 - accuracy: 0.5023
15072/25000 [=================>............] - ETA: 22s - loss: 7.6259 - accuracy: 0.5027
15104/25000 [=================>............] - ETA: 22s - loss: 7.6260 - accuracy: 0.5026
15136/25000 [=================>............] - ETA: 22s - loss: 7.6261 - accuracy: 0.5026
15168/25000 [=================>............] - ETA: 22s - loss: 7.6262 - accuracy: 0.5026
15200/25000 [=================>............] - ETA: 22s - loss: 7.6253 - accuracy: 0.5027
15232/25000 [=================>............] - ETA: 22s - loss: 7.6233 - accuracy: 0.5028
15264/25000 [=================>............] - ETA: 22s - loss: 7.6274 - accuracy: 0.5026
15296/25000 [=================>............] - ETA: 22s - loss: 7.6245 - accuracy: 0.5027
15328/25000 [=================>............] - ETA: 22s - loss: 7.6266 - accuracy: 0.5026
15360/25000 [=================>............] - ETA: 22s - loss: 7.6257 - accuracy: 0.5027
15392/25000 [=================>............] - ETA: 21s - loss: 7.6278 - accuracy: 0.5025
15424/25000 [=================>............] - ETA: 21s - loss: 7.6219 - accuracy: 0.5029
15456/25000 [=================>............] - ETA: 21s - loss: 7.6230 - accuracy: 0.5028
15488/25000 [=================>............] - ETA: 21s - loss: 7.6270 - accuracy: 0.5026
15520/25000 [=================>............] - ETA: 21s - loss: 7.6301 - accuracy: 0.5024
15552/25000 [=================>............] - ETA: 21s - loss: 7.6272 - accuracy: 0.5026
15584/25000 [=================>............] - ETA: 21s - loss: 7.6312 - accuracy: 0.5023
15616/25000 [=================>............] - ETA: 21s - loss: 7.6332 - accuracy: 0.5022
15648/25000 [=================>............] - ETA: 21s - loss: 7.6333 - accuracy: 0.5022
15680/25000 [=================>............] - ETA: 21s - loss: 7.6353 - accuracy: 0.5020
15712/25000 [=================>............] - ETA: 21s - loss: 7.6364 - accuracy: 0.5020
15744/25000 [=================>............] - ETA: 21s - loss: 7.6374 - accuracy: 0.5019
15776/25000 [=================>............] - ETA: 21s - loss: 7.6365 - accuracy: 0.5020
15808/25000 [=================>............] - ETA: 20s - loss: 7.6336 - accuracy: 0.5022
15840/25000 [==================>...........] - ETA: 20s - loss: 7.6327 - accuracy: 0.5022
15872/25000 [==================>...........] - ETA: 20s - loss: 7.6299 - accuracy: 0.5024
15904/25000 [==================>...........] - ETA: 20s - loss: 7.6338 - accuracy: 0.5021
15936/25000 [==================>...........] - ETA: 20s - loss: 7.6339 - accuracy: 0.5021
15968/25000 [==================>...........] - ETA: 20s - loss: 7.6359 - accuracy: 0.5020
16000/25000 [==================>...........] - ETA: 20s - loss: 7.6321 - accuracy: 0.5023
16032/25000 [==================>...........] - ETA: 20s - loss: 7.6284 - accuracy: 0.5025
16064/25000 [==================>...........] - ETA: 20s - loss: 7.6256 - accuracy: 0.5027
16096/25000 [==================>...........] - ETA: 20s - loss: 7.6257 - accuracy: 0.5027
16128/25000 [==================>...........] - ETA: 20s - loss: 7.6210 - accuracy: 0.5030
16160/25000 [==================>...........] - ETA: 20s - loss: 7.6211 - accuracy: 0.5030
16192/25000 [==================>...........] - ETA: 20s - loss: 7.6193 - accuracy: 0.5031
16224/25000 [==================>...........] - ETA: 20s - loss: 7.6203 - accuracy: 0.5030
16256/25000 [==================>...........] - ETA: 19s - loss: 7.6204 - accuracy: 0.5030
16288/25000 [==================>...........] - ETA: 19s - loss: 7.6214 - accuracy: 0.5029
16320/25000 [==================>...........] - ETA: 19s - loss: 7.6196 - accuracy: 0.5031
16352/25000 [==================>...........] - ETA: 19s - loss: 7.6197 - accuracy: 0.5031
16384/25000 [==================>...........] - ETA: 19s - loss: 7.6170 - accuracy: 0.5032
16416/25000 [==================>...........] - ETA: 19s - loss: 7.6162 - accuracy: 0.5033
16448/25000 [==================>...........] - ETA: 19s - loss: 7.6172 - accuracy: 0.5032
16480/25000 [==================>...........] - ETA: 19s - loss: 7.6127 - accuracy: 0.5035
16512/25000 [==================>...........] - ETA: 19s - loss: 7.6165 - accuracy: 0.5033
16544/25000 [==================>...........] - ETA: 19s - loss: 7.6138 - accuracy: 0.5034
16576/25000 [==================>...........] - ETA: 19s - loss: 7.6130 - accuracy: 0.5035
16608/25000 [==================>...........] - ETA: 19s - loss: 7.6112 - accuracy: 0.5036
16640/25000 [==================>...........] - ETA: 19s - loss: 7.6113 - accuracy: 0.5036
16672/25000 [===================>..........] - ETA: 19s - loss: 7.6096 - accuracy: 0.5037
16704/25000 [===================>..........] - ETA: 18s - loss: 7.6097 - accuracy: 0.5037
16736/25000 [===================>..........] - ETA: 18s - loss: 7.6089 - accuracy: 0.5038
16768/25000 [===================>..........] - ETA: 18s - loss: 7.6118 - accuracy: 0.5036
16800/25000 [===================>..........] - ETA: 18s - loss: 7.6173 - accuracy: 0.5032
16832/25000 [===================>..........] - ETA: 18s - loss: 7.6174 - accuracy: 0.5032
16864/25000 [===================>..........] - ETA: 18s - loss: 7.6221 - accuracy: 0.5029
16896/25000 [===================>..........] - ETA: 18s - loss: 7.6240 - accuracy: 0.5028
16928/25000 [===================>..........] - ETA: 18s - loss: 7.6231 - accuracy: 0.5028
16960/25000 [===================>..........] - ETA: 18s - loss: 7.6205 - accuracy: 0.5030
16992/25000 [===================>..........] - ETA: 18s - loss: 7.6206 - accuracy: 0.5030
17024/25000 [===================>..........] - ETA: 18s - loss: 7.6198 - accuracy: 0.5031
17056/25000 [===================>..........] - ETA: 18s - loss: 7.6226 - accuracy: 0.5029
17088/25000 [===================>..........] - ETA: 18s - loss: 7.6218 - accuracy: 0.5029
17120/25000 [===================>..........] - ETA: 17s - loss: 7.6183 - accuracy: 0.5032
17152/25000 [===================>..........] - ETA: 17s - loss: 7.6175 - accuracy: 0.5032
17184/25000 [===================>..........] - ETA: 17s - loss: 7.6193 - accuracy: 0.5031
17216/25000 [===================>..........] - ETA: 17s - loss: 7.6212 - accuracy: 0.5030
17248/25000 [===================>..........] - ETA: 17s - loss: 7.6266 - accuracy: 0.5026
17280/25000 [===================>..........] - ETA: 17s - loss: 7.6276 - accuracy: 0.5025
17312/25000 [===================>..........] - ETA: 17s - loss: 7.6276 - accuracy: 0.5025
17344/25000 [===================>..........] - ETA: 17s - loss: 7.6295 - accuracy: 0.5024
17376/25000 [===================>..........] - ETA: 17s - loss: 7.6304 - accuracy: 0.5024
17408/25000 [===================>..........] - ETA: 17s - loss: 7.6305 - accuracy: 0.5024
17440/25000 [===================>..........] - ETA: 17s - loss: 7.6306 - accuracy: 0.5024
17472/25000 [===================>..........] - ETA: 17s - loss: 7.6289 - accuracy: 0.5025
17504/25000 [====================>.........] - ETA: 17s - loss: 7.6281 - accuracy: 0.5025
17536/25000 [====================>.........] - ETA: 17s - loss: 7.6343 - accuracy: 0.5021
17568/25000 [====================>.........] - ETA: 16s - loss: 7.6335 - accuracy: 0.5022
17600/25000 [====================>.........] - ETA: 16s - loss: 7.6335 - accuracy: 0.5022
17632/25000 [====================>.........] - ETA: 16s - loss: 7.6362 - accuracy: 0.5020
17664/25000 [====================>.........] - ETA: 16s - loss: 7.6388 - accuracy: 0.5018
17696/25000 [====================>.........] - ETA: 16s - loss: 7.6380 - accuracy: 0.5019
17728/25000 [====================>.........] - ETA: 16s - loss: 7.6363 - accuracy: 0.5020
17760/25000 [====================>.........] - ETA: 16s - loss: 7.6381 - accuracy: 0.5019
17792/25000 [====================>.........] - ETA: 16s - loss: 7.6399 - accuracy: 0.5017
17824/25000 [====================>.........] - ETA: 16s - loss: 7.6374 - accuracy: 0.5019
17856/25000 [====================>.........] - ETA: 16s - loss: 7.6374 - accuracy: 0.5019
17888/25000 [====================>.........] - ETA: 16s - loss: 7.6409 - accuracy: 0.5017
17920/25000 [====================>.........] - ETA: 16s - loss: 7.6384 - accuracy: 0.5018
17952/25000 [====================>.........] - ETA: 16s - loss: 7.6393 - accuracy: 0.5018
17984/25000 [====================>.........] - ETA: 15s - loss: 7.6402 - accuracy: 0.5017
18016/25000 [====================>.........] - ETA: 15s - loss: 7.6445 - accuracy: 0.5014
18048/25000 [====================>.........] - ETA: 15s - loss: 7.6454 - accuracy: 0.5014
18080/25000 [====================>.........] - ETA: 15s - loss: 7.6480 - accuracy: 0.5012
18112/25000 [====================>.........] - ETA: 15s - loss: 7.6514 - accuracy: 0.5010
18144/25000 [====================>.........] - ETA: 15s - loss: 7.6523 - accuracy: 0.5009
18176/25000 [====================>.........] - ETA: 15s - loss: 7.6540 - accuracy: 0.5008
18208/25000 [====================>.........] - ETA: 15s - loss: 7.6540 - accuracy: 0.5008
18240/25000 [====================>.........] - ETA: 15s - loss: 7.6549 - accuracy: 0.5008
18272/25000 [====================>.........] - ETA: 15s - loss: 7.6565 - accuracy: 0.5007
18304/25000 [====================>.........] - ETA: 15s - loss: 7.6582 - accuracy: 0.5005
18336/25000 [=====================>........] - ETA: 15s - loss: 7.6608 - accuracy: 0.5004
18368/25000 [=====================>........] - ETA: 15s - loss: 7.6649 - accuracy: 0.5001
18400/25000 [=====================>........] - ETA: 15s - loss: 7.6625 - accuracy: 0.5003
18432/25000 [=====================>........] - ETA: 14s - loss: 7.6625 - accuracy: 0.5003
18464/25000 [=====================>........] - ETA: 14s - loss: 7.6616 - accuracy: 0.5003
18496/25000 [=====================>........] - ETA: 14s - loss: 7.6625 - accuracy: 0.5003
18528/25000 [=====================>........] - ETA: 14s - loss: 7.6608 - accuracy: 0.5004
18560/25000 [=====================>........] - ETA: 14s - loss: 7.6584 - accuracy: 0.5005
18592/25000 [=====================>........] - ETA: 14s - loss: 7.6567 - accuracy: 0.5006
18624/25000 [=====================>........] - ETA: 14s - loss: 7.6567 - accuracy: 0.5006
18656/25000 [=====================>........] - ETA: 14s - loss: 7.6568 - accuracy: 0.5006
18688/25000 [=====================>........] - ETA: 14s - loss: 7.6527 - accuracy: 0.5009
18720/25000 [=====================>........] - ETA: 14s - loss: 7.6519 - accuracy: 0.5010
18752/25000 [=====================>........] - ETA: 14s - loss: 7.6470 - accuracy: 0.5013
18784/25000 [=====================>........] - ETA: 14s - loss: 7.6438 - accuracy: 0.5015
18816/25000 [=====================>........] - ETA: 14s - loss: 7.6414 - accuracy: 0.5016
18848/25000 [=====================>........] - ETA: 14s - loss: 7.6422 - accuracy: 0.5016
18880/25000 [=====================>........] - ETA: 13s - loss: 7.6414 - accuracy: 0.5016
18912/25000 [=====================>........] - ETA: 13s - loss: 7.6447 - accuracy: 0.5014
18944/25000 [=====================>........] - ETA: 13s - loss: 7.6440 - accuracy: 0.5015
18976/25000 [=====================>........] - ETA: 13s - loss: 7.6448 - accuracy: 0.5014
19008/25000 [=====================>........] - ETA: 13s - loss: 7.6368 - accuracy: 0.5019
19040/25000 [=====================>........] - ETA: 13s - loss: 7.6336 - accuracy: 0.5022
19072/25000 [=====================>........] - ETA: 13s - loss: 7.6345 - accuracy: 0.5021
19104/25000 [=====================>........] - ETA: 13s - loss: 7.6361 - accuracy: 0.5020
19136/25000 [=====================>........] - ETA: 13s - loss: 7.6402 - accuracy: 0.5017
19168/25000 [======================>.......] - ETA: 13s - loss: 7.6370 - accuracy: 0.5019
19200/25000 [======================>.......] - ETA: 13s - loss: 7.6347 - accuracy: 0.5021
19232/25000 [======================>.......] - ETA: 13s - loss: 7.6331 - accuracy: 0.5022
19264/25000 [======================>.......] - ETA: 13s - loss: 7.6340 - accuracy: 0.5021
19296/25000 [======================>.......] - ETA: 12s - loss: 7.6348 - accuracy: 0.5021
19328/25000 [======================>.......] - ETA: 12s - loss: 7.6365 - accuracy: 0.5020
19360/25000 [======================>.......] - ETA: 12s - loss: 7.6341 - accuracy: 0.5021
19392/25000 [======================>.......] - ETA: 12s - loss: 7.6326 - accuracy: 0.5022
19424/25000 [======================>.......] - ETA: 12s - loss: 7.6327 - accuracy: 0.5022
19456/25000 [======================>.......] - ETA: 12s - loss: 7.6343 - accuracy: 0.5021
19488/25000 [======================>.......] - ETA: 12s - loss: 7.6367 - accuracy: 0.5019
19520/25000 [======================>.......] - ETA: 12s - loss: 7.6399 - accuracy: 0.5017
19552/25000 [======================>.......] - ETA: 12s - loss: 7.6400 - accuracy: 0.5017
19584/25000 [======================>.......] - ETA: 12s - loss: 7.6384 - accuracy: 0.5018
19616/25000 [======================>.......] - ETA: 12s - loss: 7.6432 - accuracy: 0.5015
19648/25000 [======================>.......] - ETA: 12s - loss: 7.6416 - accuracy: 0.5016
19680/25000 [======================>.......] - ETA: 12s - loss: 7.6440 - accuracy: 0.5015
19712/25000 [======================>.......] - ETA: 12s - loss: 7.6410 - accuracy: 0.5017
19744/25000 [======================>.......] - ETA: 11s - loss: 7.6402 - accuracy: 0.5017
19776/25000 [======================>.......] - ETA: 11s - loss: 7.6410 - accuracy: 0.5017
19808/25000 [======================>.......] - ETA: 11s - loss: 7.6434 - accuracy: 0.5015
19840/25000 [======================>.......] - ETA: 11s - loss: 7.6380 - accuracy: 0.5019
19872/25000 [======================>.......] - ETA: 11s - loss: 7.6388 - accuracy: 0.5018
19904/25000 [======================>.......] - ETA: 11s - loss: 7.6381 - accuracy: 0.5019
19936/25000 [======================>.......] - ETA: 11s - loss: 7.6351 - accuracy: 0.5021
19968/25000 [======================>.......] - ETA: 11s - loss: 7.6374 - accuracy: 0.5019
20000/25000 [=======================>......] - ETA: 11s - loss: 7.6390 - accuracy: 0.5018
20032/25000 [=======================>......] - ETA: 11s - loss: 7.6383 - accuracy: 0.5018
20064/25000 [=======================>......] - ETA: 11s - loss: 7.6361 - accuracy: 0.5020
20096/25000 [=======================>......] - ETA: 11s - loss: 7.6330 - accuracy: 0.5022
20128/25000 [=======================>......] - ETA: 11s - loss: 7.6331 - accuracy: 0.5022
20160/25000 [=======================>......] - ETA: 11s - loss: 7.6339 - accuracy: 0.5021
20192/25000 [=======================>......] - ETA: 10s - loss: 7.6385 - accuracy: 0.5018
20224/25000 [=======================>......] - ETA: 10s - loss: 7.6371 - accuracy: 0.5019
20256/25000 [=======================>......] - ETA: 10s - loss: 7.6356 - accuracy: 0.5020
20288/25000 [=======================>......] - ETA: 10s - loss: 7.6356 - accuracy: 0.5020
20320/25000 [=======================>......] - ETA: 10s - loss: 7.6364 - accuracy: 0.5020
20352/25000 [=======================>......] - ETA: 10s - loss: 7.6357 - accuracy: 0.5020
20384/25000 [=======================>......] - ETA: 10s - loss: 7.6365 - accuracy: 0.5020
20416/25000 [=======================>......] - ETA: 10s - loss: 7.6358 - accuracy: 0.5020
20448/25000 [=======================>......] - ETA: 10s - loss: 7.6351 - accuracy: 0.5021
20480/25000 [=======================>......] - ETA: 10s - loss: 7.6329 - accuracy: 0.5022
20512/25000 [=======================>......] - ETA: 10s - loss: 7.6360 - accuracy: 0.5020
20544/25000 [=======================>......] - ETA: 10s - loss: 7.6353 - accuracy: 0.5020
20576/25000 [=======================>......] - ETA: 10s - loss: 7.6361 - accuracy: 0.5020
20608/25000 [=======================>......] - ETA: 9s - loss: 7.6331 - accuracy: 0.5022 
20640/25000 [=======================>......] - ETA: 9s - loss: 7.6287 - accuracy: 0.5025
20672/25000 [=======================>......] - ETA: 9s - loss: 7.6325 - accuracy: 0.5022
20704/25000 [=======================>......] - ETA: 9s - loss: 7.6311 - accuracy: 0.5023
20736/25000 [=======================>......] - ETA: 9s - loss: 7.6326 - accuracy: 0.5022
20768/25000 [=======================>......] - ETA: 9s - loss: 7.6334 - accuracy: 0.5022
20800/25000 [=======================>......] - ETA: 9s - loss: 7.6357 - accuracy: 0.5020
20832/25000 [=======================>......] - ETA: 9s - loss: 7.6372 - accuracy: 0.5019
20864/25000 [========================>.....] - ETA: 9s - loss: 7.6416 - accuracy: 0.5016
20896/25000 [========================>.....] - ETA: 9s - loss: 7.6446 - accuracy: 0.5014
20928/25000 [========================>.....] - ETA: 9s - loss: 7.6476 - accuracy: 0.5012
20960/25000 [========================>.....] - ETA: 9s - loss: 7.6498 - accuracy: 0.5011
20992/25000 [========================>.....] - ETA: 9s - loss: 7.6505 - accuracy: 0.5010
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6513 - accuracy: 0.5010
21056/25000 [========================>.....] - ETA: 8s - loss: 7.6535 - accuracy: 0.5009
21088/25000 [========================>.....] - ETA: 8s - loss: 7.6521 - accuracy: 0.5009
21120/25000 [========================>.....] - ETA: 8s - loss: 7.6528 - accuracy: 0.5009
21152/25000 [========================>.....] - ETA: 8s - loss: 7.6543 - accuracy: 0.5008
21184/25000 [========================>.....] - ETA: 8s - loss: 7.6521 - accuracy: 0.5009
21216/25000 [========================>.....] - ETA: 8s - loss: 7.6507 - accuracy: 0.5010
21248/25000 [========================>.....] - ETA: 8s - loss: 7.6515 - accuracy: 0.5010
21280/25000 [========================>.....] - ETA: 8s - loss: 7.6522 - accuracy: 0.5009
21312/25000 [========================>.....] - ETA: 8s - loss: 7.6508 - accuracy: 0.5010
21344/25000 [========================>.....] - ETA: 8s - loss: 7.6515 - accuracy: 0.5010
21376/25000 [========================>.....] - ETA: 8s - loss: 7.6551 - accuracy: 0.5007
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6552 - accuracy: 0.5007
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6573 - accuracy: 0.5006
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6638 - accuracy: 0.5002
21504/25000 [========================>.....] - ETA: 7s - loss: 7.6623 - accuracy: 0.5003
21536/25000 [========================>.....] - ETA: 7s - loss: 7.6631 - accuracy: 0.5002
21568/25000 [========================>.....] - ETA: 7s - loss: 7.6609 - accuracy: 0.5004
21600/25000 [========================>.....] - ETA: 7s - loss: 7.6617 - accuracy: 0.5003
21632/25000 [========================>.....] - ETA: 7s - loss: 7.6602 - accuracy: 0.5004
21664/25000 [========================>.....] - ETA: 7s - loss: 7.6574 - accuracy: 0.5006
21696/25000 [=========================>....] - ETA: 7s - loss: 7.6581 - accuracy: 0.5006
21728/25000 [=========================>....] - ETA: 7s - loss: 7.6582 - accuracy: 0.5006
21760/25000 [=========================>....] - ETA: 7s - loss: 7.6582 - accuracy: 0.5006
21792/25000 [=========================>....] - ETA: 7s - loss: 7.6582 - accuracy: 0.5006
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6575 - accuracy: 0.5006
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6561 - accuracy: 0.5007
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6540 - accuracy: 0.5008
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6547 - accuracy: 0.5008
21952/25000 [=========================>....] - ETA: 6s - loss: 7.6520 - accuracy: 0.5010
21984/25000 [=========================>....] - ETA: 6s - loss: 7.6527 - accuracy: 0.5009
22016/25000 [=========================>....] - ETA: 6s - loss: 7.6492 - accuracy: 0.5011
22048/25000 [=========================>....] - ETA: 6s - loss: 7.6492 - accuracy: 0.5011
22080/25000 [=========================>....] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
22112/25000 [=========================>....] - ETA: 6s - loss: 7.6493 - accuracy: 0.5011
22144/25000 [=========================>....] - ETA: 6s - loss: 7.6445 - accuracy: 0.5014
22176/25000 [=========================>....] - ETA: 6s - loss: 7.6431 - accuracy: 0.5015
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6425 - accuracy: 0.5016
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6411 - accuracy: 0.5017
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6398 - accuracy: 0.5018
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6398 - accuracy: 0.5017
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6447 - accuracy: 0.5014
22368/25000 [=========================>....] - ETA: 5s - loss: 7.6447 - accuracy: 0.5014
22400/25000 [=========================>....] - ETA: 5s - loss: 7.6461 - accuracy: 0.5013
22432/25000 [=========================>....] - ETA: 5s - loss: 7.6482 - accuracy: 0.5012
22464/25000 [=========================>....] - ETA: 5s - loss: 7.6489 - accuracy: 0.5012
22496/25000 [=========================>....] - ETA: 5s - loss: 7.6469 - accuracy: 0.5013
22528/25000 [==========================>...] - ETA: 5s - loss: 7.6448 - accuracy: 0.5014
22560/25000 [==========================>...] - ETA: 5s - loss: 7.6469 - accuracy: 0.5013
22592/25000 [==========================>...] - ETA: 5s - loss: 7.6483 - accuracy: 0.5012
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6470 - accuracy: 0.5013
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6463 - accuracy: 0.5013
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6518 - accuracy: 0.5010
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6504 - accuracy: 0.5011
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6504 - accuracy: 0.5011
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6498 - accuracy: 0.5011
22816/25000 [==========================>...] - ETA: 4s - loss: 7.6525 - accuracy: 0.5009
22848/25000 [==========================>...] - ETA: 4s - loss: 7.6552 - accuracy: 0.5007
22880/25000 [==========================>...] - ETA: 4s - loss: 7.6539 - accuracy: 0.5008
22912/25000 [==========================>...] - ETA: 4s - loss: 7.6532 - accuracy: 0.5009
22944/25000 [==========================>...] - ETA: 4s - loss: 7.6553 - accuracy: 0.5007
22976/25000 [==========================>...] - ETA: 4s - loss: 7.6553 - accuracy: 0.5007
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6553 - accuracy: 0.5007
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6573 - accuracy: 0.5006
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6553 - accuracy: 0.5007
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6573 - accuracy: 0.5006
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6573 - accuracy: 0.5006
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6574 - accuracy: 0.5006
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6567 - accuracy: 0.5006
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6574 - accuracy: 0.5006
23264/25000 [==========================>...] - ETA: 3s - loss: 7.6587 - accuracy: 0.5005
23296/25000 [==========================>...] - ETA: 3s - loss: 7.6594 - accuracy: 0.5005
23328/25000 [==========================>...] - ETA: 3s - loss: 7.6587 - accuracy: 0.5005
23360/25000 [===========================>..] - ETA: 3s - loss: 7.6614 - accuracy: 0.5003
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6620 - accuracy: 0.5003
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6614 - accuracy: 0.5003
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6614 - accuracy: 0.5003
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6620 - accuracy: 0.5003
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6601 - accuracy: 0.5004
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6588 - accuracy: 0.5005
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6569 - accuracy: 0.5006
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6556 - accuracy: 0.5007
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6595 - accuracy: 0.5005
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6576 - accuracy: 0.5006
23712/25000 [===========================>..] - ETA: 2s - loss: 7.6576 - accuracy: 0.5006
23744/25000 [===========================>..] - ETA: 2s - loss: 7.6569 - accuracy: 0.5006
23776/25000 [===========================>..] - ETA: 2s - loss: 7.6576 - accuracy: 0.5006
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6570 - accuracy: 0.5006
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6583 - accuracy: 0.5005
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6589 - accuracy: 0.5005
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6589 - accuracy: 0.5005
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6621 - accuracy: 0.5003
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6622 - accuracy: 0.5003
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24128/25000 [===========================>..] - ETA: 1s - loss: 7.6698 - accuracy: 0.4998
24160/25000 [===========================>..] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24192/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24224/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24256/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24288/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24320/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24352/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24384/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24416/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24448/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24480/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24512/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24544/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24576/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24608/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
24640/25000 [============================>.] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
24672/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24704/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24736/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24768/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24800/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24832/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24864/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24896/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24960/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24992/25000 [============================>.] - ETA: 0s - loss: 7.6678 - accuracy: 0.4999
25000/25000 [==============================] - 67s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
