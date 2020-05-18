
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/b64972ca3db1fc61767569289991118cc6d5e8ab', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'b64972ca3db1fc61767569289991118cc6d5e8ab', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/b64972ca3db1fc61767569289991118cc6d5e8ab

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/b64972ca3db1fc61767569289991118cc6d5e8ab

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/b64972ca3db1fc61767569289991118cc6d5e8ab

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
 40%|████      | 2/5 [00:52<01:18, 26.32s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.29525874043495387, 'embedding_size_factor': 1.1079042186415948, 'layers.choice': 2, 'learning_rate': 0.003763735920486, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 2.3489616263600543e-09} and reward: 0.3816
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd2\xe5\x84\xea\x81\xae\x91X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\xb9\xf9\xc6"\xa8GX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?n\xd5 V\x0e{gX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>$-k\xd6\x90\xdbbu.' and reward: 0.3816
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd2\xe5\x84\xea\x81\xae\x91X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\xb9\xf9\xc6"\xa8GX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?n\xd5 V\x0e{gX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>$-k\xd6\x90\xdbbu.' and reward: 0.3816
 60%|██████    | 3/5 [01:46<01:08, 34.45s/it] 60%|██████    | 3/5 [01:46<01:10, 35.35s/it]
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
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.0101126773112748, 'embedding_size_factor': 1.060570848099152, 'layers.choice': 0, 'learning_rate': 0.0013954892383522233, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.824535828751345e-08} and reward: 0.3698
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\x84\xb5\xf4\x92\x9d\x8b\xa1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\xf8\x19#:\xd4XX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?V\xdd\x1b)\x00JqX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>S\x97>\xf3#\xb9[u.' and reward: 0.3698
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\x84\xb5\xf4\x92\x9d\x8b\xa1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\xf8\x19#:\xd4XX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?V\xdd\x1b)\x00JqX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>S\x97>\xf3#\xb9[u.' and reward: 0.3698
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 160.39104747772217
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -43.07s of remaining time.
Ensemble size: 64
Ensemble weights: 
[0.578125 0.28125  0.140625]
	0.3926	 = Validation accuracy score
	1.09s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 164.19s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7efff060eac8> 

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
 [-0.0423542   0.03034924 -0.03964816  0.00242972  0.09457052  0.04602471]
 [-0.10547691  0.09357768  0.2268669   0.23297364 -0.1448293   0.13680096]
 [ 0.40913802 -0.14256649  0.32215464  0.14478578 -0.0160309   0.20365344]
 [ 0.02431957 -0.2219262   0.62224537  0.64545238  0.33755791 -0.18663603]
 [-0.06303038  0.03346157  0.20412867  0.1686105   0.52414876  0.22450987]
 [-0.58753902  0.82934684  0.60919797  0.4179576   0.32143223  0.19779198]
 [-0.56686997  0.30886242  0.42298484  0.83579218  0.26003444 -0.32405421]
 [ 0.20317332 -0.1597326   0.45993522  0.81118721  0.30504596  0.99690789]
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
{'loss': 0.5301916673779488, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-18 08:04:46.885502: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.3222058042883873, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-18 08:04:48.052218: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 1327104/17464789 [=>............................] - ETA: 0s
 3432448/17464789 [====>.........................] - ETA: 0s
 6668288/17464789 [==========>...................] - ETA: 0s
10952704/17464789 [=================>............] - ETA: 0s
15368192/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-18 08:05:00.182536: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-18 08:05:00.187422: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-18 08:05:00.187596: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bd395fce60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-18 08:05:00.187612: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:54 - loss: 5.2708 - accuracy: 0.6562
   64/25000 [..............................] - ETA: 3:00 - loss: 6.2291 - accuracy: 0.5938
   96/25000 [..............................] - ETA: 2:22 - loss: 7.3472 - accuracy: 0.5208
  128/25000 [..............................] - ETA: 2:03 - loss: 7.1875 - accuracy: 0.5312
  160/25000 [..............................] - ETA: 1:51 - loss: 6.6125 - accuracy: 0.5688
  192/25000 [..............................] - ETA: 1:43 - loss: 6.4687 - accuracy: 0.5781
  224/25000 [..............................] - ETA: 1:38 - loss: 6.9821 - accuracy: 0.5446
  256/25000 [..............................] - ETA: 1:34 - loss: 7.0078 - accuracy: 0.5430
  288/25000 [..............................] - ETA: 1:30 - loss: 7.1342 - accuracy: 0.5347
  320/25000 [..............................] - ETA: 1:27 - loss: 7.2354 - accuracy: 0.5281
  352/25000 [..............................] - ETA: 1:24 - loss: 7.1439 - accuracy: 0.5341
  384/25000 [..............................] - ETA: 1:22 - loss: 7.1076 - accuracy: 0.5365
  416/25000 [..............................] - ETA: 1:20 - loss: 6.9663 - accuracy: 0.5457
  448/25000 [..............................] - ETA: 1:19 - loss: 6.9821 - accuracy: 0.5446
  480/25000 [..............................] - ETA: 1:17 - loss: 7.0916 - accuracy: 0.5375
  512/25000 [..............................] - ETA: 1:16 - loss: 7.1875 - accuracy: 0.5312
  544/25000 [..............................] - ETA: 1:15 - loss: 7.2156 - accuracy: 0.5294
  576/25000 [..............................] - ETA: 1:14 - loss: 7.2407 - accuracy: 0.5278
  608/25000 [..............................] - ETA: 1:13 - loss: 7.3136 - accuracy: 0.5230
  640/25000 [..............................] - ETA: 1:12 - loss: 7.4510 - accuracy: 0.5141
  672/25000 [..............................] - ETA: 1:11 - loss: 7.4841 - accuracy: 0.5119
  704/25000 [..............................] - ETA: 1:11 - loss: 7.5142 - accuracy: 0.5099
  736/25000 [..............................] - ETA: 1:10 - loss: 7.4791 - accuracy: 0.5122
  768/25000 [..............................] - ETA: 1:09 - loss: 7.4470 - accuracy: 0.5143
  800/25000 [..............................] - ETA: 1:09 - loss: 7.4941 - accuracy: 0.5113
  832/25000 [..............................] - ETA: 1:08 - loss: 7.5008 - accuracy: 0.5108
  864/25000 [>.............................] - ETA: 1:08 - loss: 7.4892 - accuracy: 0.5116
  896/25000 [>.............................] - ETA: 1:08 - loss: 7.4613 - accuracy: 0.5134
  928/25000 [>.............................] - ETA: 1:08 - loss: 7.4518 - accuracy: 0.5140
  960/25000 [>.............................] - ETA: 1:07 - loss: 7.3951 - accuracy: 0.5177
  992/25000 [>.............................] - ETA: 1:07 - loss: 7.3729 - accuracy: 0.5192
 1024/25000 [>.............................] - ETA: 1:07 - loss: 7.4270 - accuracy: 0.5156
 1056/25000 [>.............................] - ETA: 1:07 - loss: 7.4488 - accuracy: 0.5142
 1088/25000 [>.............................] - ETA: 1:06 - loss: 7.3989 - accuracy: 0.5175
 1120/25000 [>.............................] - ETA: 1:06 - loss: 7.5160 - accuracy: 0.5098
 1152/25000 [>.............................] - ETA: 1:06 - loss: 7.5601 - accuracy: 0.5069
 1184/25000 [>.............................] - ETA: 1:05 - loss: 7.5630 - accuracy: 0.5068
 1216/25000 [>.............................] - ETA: 1:05 - loss: 7.5405 - accuracy: 0.5082
 1248/25000 [>.............................] - ETA: 1:05 - loss: 7.5806 - accuracy: 0.5056
 1280/25000 [>.............................] - ETA: 1:04 - loss: 7.5947 - accuracy: 0.5047
 1312/25000 [>.............................] - ETA: 1:04 - loss: 7.5848 - accuracy: 0.5053
 1344/25000 [>.............................] - ETA: 1:04 - loss: 7.5754 - accuracy: 0.5060
 1376/25000 [>.............................] - ETA: 1:04 - loss: 7.6666 - accuracy: 0.5000
 1408/25000 [>.............................] - ETA: 1:04 - loss: 7.6775 - accuracy: 0.4993
 1440/25000 [>.............................] - ETA: 1:03 - loss: 7.6347 - accuracy: 0.5021
 1472/25000 [>.............................] - ETA: 1:03 - loss: 7.6458 - accuracy: 0.5014
 1504/25000 [>.............................] - ETA: 1:03 - loss: 7.6768 - accuracy: 0.4993
 1536/25000 [>.............................] - ETA: 1:03 - loss: 7.6866 - accuracy: 0.4987
 1568/25000 [>.............................] - ETA: 1:02 - loss: 7.6764 - accuracy: 0.4994
 1600/25000 [>.............................] - ETA: 1:02 - loss: 7.6858 - accuracy: 0.4988
 1632/25000 [>.............................] - ETA: 1:02 - loss: 7.7042 - accuracy: 0.4975
 1664/25000 [>.............................] - ETA: 1:02 - loss: 7.7496 - accuracy: 0.4946
 1696/25000 [=>............................] - ETA: 1:02 - loss: 7.7118 - accuracy: 0.4971
 1728/25000 [=>............................] - ETA: 1:02 - loss: 7.6932 - accuracy: 0.4983
 1760/25000 [=>............................] - ETA: 1:01 - loss: 7.7015 - accuracy: 0.4977
 1792/25000 [=>............................] - ETA: 1:01 - loss: 7.6837 - accuracy: 0.4989
 1824/25000 [=>............................] - ETA: 1:01 - loss: 7.7087 - accuracy: 0.4973
 1856/25000 [=>............................] - ETA: 1:01 - loss: 7.7079 - accuracy: 0.4973
 1888/25000 [=>............................] - ETA: 1:01 - loss: 7.7316 - accuracy: 0.4958
 1920/25000 [=>............................] - ETA: 1:01 - loss: 7.7145 - accuracy: 0.4969
 1952/25000 [=>............................] - ETA: 1:01 - loss: 7.7609 - accuracy: 0.4939
 1984/25000 [=>............................] - ETA: 1:00 - loss: 7.7748 - accuracy: 0.4929
 2016/25000 [=>............................] - ETA: 1:00 - loss: 7.8187 - accuracy: 0.4901
 2048/25000 [=>............................] - ETA: 1:00 - loss: 7.8014 - accuracy: 0.4912
 2080/25000 [=>............................] - ETA: 1:00 - loss: 7.8067 - accuracy: 0.4909
 2112/25000 [=>............................] - ETA: 1:00 - loss: 7.8409 - accuracy: 0.4886
 2144/25000 [=>............................] - ETA: 1:00 - loss: 7.8168 - accuracy: 0.4902
 2176/25000 [=>............................] - ETA: 1:00 - loss: 7.7723 - accuracy: 0.4931
 2208/25000 [=>............................] - ETA: 1:00 - loss: 7.7500 - accuracy: 0.4946
 2240/25000 [=>............................] - ETA: 59s - loss: 7.7214 - accuracy: 0.4964 
 2272/25000 [=>............................] - ETA: 59s - loss: 7.7206 - accuracy: 0.4965
 2304/25000 [=>............................] - ETA: 59s - loss: 7.7265 - accuracy: 0.4961
 2336/25000 [=>............................] - ETA: 59s - loss: 7.7191 - accuracy: 0.4966
 2368/25000 [=>............................] - ETA: 59s - loss: 7.7314 - accuracy: 0.4958
 2400/25000 [=>............................] - ETA: 59s - loss: 7.7113 - accuracy: 0.4971
 2432/25000 [=>............................] - ETA: 59s - loss: 7.7171 - accuracy: 0.4967
 2464/25000 [=>............................] - ETA: 59s - loss: 7.7226 - accuracy: 0.4963
 2496/25000 [=>............................] - ETA: 58s - loss: 7.7219 - accuracy: 0.4964
 2528/25000 [==>...........................] - ETA: 58s - loss: 7.7273 - accuracy: 0.4960
 2560/25000 [==>...........................] - ETA: 58s - loss: 7.7265 - accuracy: 0.4961
 2592/25000 [==>...........................] - ETA: 58s - loss: 7.7258 - accuracy: 0.4961
 2624/25000 [==>...........................] - ETA: 58s - loss: 7.7251 - accuracy: 0.4962
 2656/25000 [==>...........................] - ETA: 58s - loss: 7.7301 - accuracy: 0.4959
 2688/25000 [==>...........................] - ETA: 58s - loss: 7.7750 - accuracy: 0.4929
 2720/25000 [==>...........................] - ETA: 57s - loss: 7.7794 - accuracy: 0.4926
 2752/25000 [==>...........................] - ETA: 57s - loss: 7.7781 - accuracy: 0.4927
 2784/25000 [==>...........................] - ETA: 57s - loss: 7.7823 - accuracy: 0.4925
 2816/25000 [==>...........................] - ETA: 57s - loss: 7.8027 - accuracy: 0.4911
 2848/25000 [==>...........................] - ETA: 57s - loss: 7.8066 - accuracy: 0.4909
 2880/25000 [==>...........................] - ETA: 57s - loss: 7.7997 - accuracy: 0.4913
 2912/25000 [==>...........................] - ETA: 57s - loss: 7.8088 - accuracy: 0.4907
 2944/25000 [==>...........................] - ETA: 57s - loss: 7.8020 - accuracy: 0.4912
 2976/25000 [==>...........................] - ETA: 56s - loss: 7.7954 - accuracy: 0.4916
 3008/25000 [==>...........................] - ETA: 56s - loss: 7.7941 - accuracy: 0.4917
 3040/25000 [==>...........................] - ETA: 56s - loss: 7.7826 - accuracy: 0.4924
 3072/25000 [==>...........................] - ETA: 56s - loss: 7.7664 - accuracy: 0.4935
 3104/25000 [==>...........................] - ETA: 56s - loss: 7.7605 - accuracy: 0.4939
 3136/25000 [==>...........................] - ETA: 56s - loss: 7.7840 - accuracy: 0.4923
 3168/25000 [==>...........................] - ETA: 56s - loss: 7.7683 - accuracy: 0.4934
 3200/25000 [==>...........................] - ETA: 56s - loss: 7.7960 - accuracy: 0.4916
 3232/25000 [==>...........................] - ETA: 56s - loss: 7.7947 - accuracy: 0.4916
 3264/25000 [==>...........................] - ETA: 55s - loss: 7.7935 - accuracy: 0.4917
 3296/25000 [==>...........................] - ETA: 55s - loss: 7.7597 - accuracy: 0.4939
 3328/25000 [==>...........................] - ETA: 55s - loss: 7.7634 - accuracy: 0.4937
 3360/25000 [===>..........................] - ETA: 55s - loss: 7.7351 - accuracy: 0.4955
 3392/25000 [===>..........................] - ETA: 55s - loss: 7.7570 - accuracy: 0.4941
 3424/25000 [===>..........................] - ETA: 55s - loss: 7.7472 - accuracy: 0.4947
 3456/25000 [===>..........................] - ETA: 55s - loss: 7.7642 - accuracy: 0.4936
 3488/25000 [===>..........................] - ETA: 55s - loss: 7.7501 - accuracy: 0.4946
 3520/25000 [===>..........................] - ETA: 55s - loss: 7.7232 - accuracy: 0.4963
 3552/25000 [===>..........................] - ETA: 55s - loss: 7.7271 - accuracy: 0.4961
 3584/25000 [===>..........................] - ETA: 55s - loss: 7.7222 - accuracy: 0.4964
 3616/25000 [===>..........................] - ETA: 54s - loss: 7.7260 - accuracy: 0.4961
 3648/25000 [===>..........................] - ETA: 54s - loss: 7.7339 - accuracy: 0.4956
 3680/25000 [===>..........................] - ETA: 54s - loss: 7.7416 - accuracy: 0.4951
 3712/25000 [===>..........................] - ETA: 54s - loss: 7.7286 - accuracy: 0.4960
 3744/25000 [===>..........................] - ETA: 54s - loss: 7.7117 - accuracy: 0.4971
 3776/25000 [===>..........................] - ETA: 54s - loss: 7.7113 - accuracy: 0.4971
 3808/25000 [===>..........................] - ETA: 54s - loss: 7.6948 - accuracy: 0.4982
 3840/25000 [===>..........................] - ETA: 54s - loss: 7.6866 - accuracy: 0.4987
 3872/25000 [===>..........................] - ETA: 54s - loss: 7.7102 - accuracy: 0.4972
 3904/25000 [===>..........................] - ETA: 54s - loss: 7.7059 - accuracy: 0.4974
 3936/25000 [===>..........................] - ETA: 53s - loss: 7.7134 - accuracy: 0.4970
 3968/25000 [===>..........................] - ETA: 53s - loss: 7.7207 - accuracy: 0.4965
 4000/25000 [===>..........................] - ETA: 53s - loss: 7.7241 - accuracy: 0.4963
 4032/25000 [===>..........................] - ETA: 53s - loss: 7.7199 - accuracy: 0.4965
 4064/25000 [===>..........................] - ETA: 53s - loss: 7.7270 - accuracy: 0.4961
 4096/25000 [===>..........................] - ETA: 53s - loss: 7.7265 - accuracy: 0.4961
 4128/25000 [===>..........................] - ETA: 53s - loss: 7.7149 - accuracy: 0.4969
 4160/25000 [===>..........................] - ETA: 53s - loss: 7.7145 - accuracy: 0.4969
 4192/25000 [====>.........................] - ETA: 53s - loss: 7.7142 - accuracy: 0.4969
 4224/25000 [====>.........................] - ETA: 53s - loss: 7.6993 - accuracy: 0.4979
 4256/25000 [====>.........................] - ETA: 52s - loss: 7.7062 - accuracy: 0.4974
 4288/25000 [====>.........................] - ETA: 52s - loss: 7.7095 - accuracy: 0.4972
 4320/25000 [====>.........................] - ETA: 52s - loss: 7.6950 - accuracy: 0.4981
 4352/25000 [====>.........................] - ETA: 52s - loss: 7.7089 - accuracy: 0.4972
 4384/25000 [====>.........................] - ETA: 52s - loss: 7.7191 - accuracy: 0.4966
 4416/25000 [====>.........................] - ETA: 52s - loss: 7.7291 - accuracy: 0.4959
 4448/25000 [====>.........................] - ETA: 52s - loss: 7.7252 - accuracy: 0.4962
 4480/25000 [====>.........................] - ETA: 52s - loss: 7.7214 - accuracy: 0.4964
 4512/25000 [====>.........................] - ETA: 52s - loss: 7.7244 - accuracy: 0.4962
 4544/25000 [====>.........................] - ETA: 52s - loss: 7.7274 - accuracy: 0.4960
 4576/25000 [====>.........................] - ETA: 52s - loss: 7.7269 - accuracy: 0.4961
 4608/25000 [====>.........................] - ETA: 52s - loss: 7.7199 - accuracy: 0.4965
 4640/25000 [====>.........................] - ETA: 51s - loss: 7.7294 - accuracy: 0.4959
 4672/25000 [====>.........................] - ETA: 51s - loss: 7.7290 - accuracy: 0.4959
 4704/25000 [====>.........................] - ETA: 51s - loss: 7.7286 - accuracy: 0.4960
 4736/25000 [====>.........................] - ETA: 51s - loss: 7.7249 - accuracy: 0.4962
 4768/25000 [====>.........................] - ETA: 51s - loss: 7.7149 - accuracy: 0.4969
 4800/25000 [====>.........................] - ETA: 51s - loss: 7.7050 - accuracy: 0.4975
 4832/25000 [====>.........................] - ETA: 51s - loss: 7.7047 - accuracy: 0.4975
 4864/25000 [====>.........................] - ETA: 51s - loss: 7.7108 - accuracy: 0.4971
 4896/25000 [====>.........................] - ETA: 51s - loss: 7.7105 - accuracy: 0.4971
 4928/25000 [====>.........................] - ETA: 51s - loss: 7.7071 - accuracy: 0.4974
 4960/25000 [====>.........................] - ETA: 51s - loss: 7.7223 - accuracy: 0.4964
 4992/25000 [====>.........................] - ETA: 51s - loss: 7.7219 - accuracy: 0.4964
 5024/25000 [=====>........................] - ETA: 51s - loss: 7.7338 - accuracy: 0.4956
 5056/25000 [=====>........................] - ETA: 51s - loss: 7.7212 - accuracy: 0.4964
 5088/25000 [=====>........................] - ETA: 50s - loss: 7.7239 - accuracy: 0.4963
 5120/25000 [=====>........................] - ETA: 50s - loss: 7.7235 - accuracy: 0.4963
 5152/25000 [=====>........................] - ETA: 50s - loss: 7.7261 - accuracy: 0.4961
 5184/25000 [=====>........................] - ETA: 50s - loss: 7.7169 - accuracy: 0.4967
 5216/25000 [=====>........................] - ETA: 50s - loss: 7.7225 - accuracy: 0.4964
 5248/25000 [=====>........................] - ETA: 50s - loss: 7.7163 - accuracy: 0.4968
 5280/25000 [=====>........................] - ETA: 50s - loss: 7.7189 - accuracy: 0.4966
 5312/25000 [=====>........................] - ETA: 50s - loss: 7.7244 - accuracy: 0.4962
 5344/25000 [=====>........................] - ETA: 50s - loss: 7.7154 - accuracy: 0.4968
 5376/25000 [=====>........................] - ETA: 50s - loss: 7.7123 - accuracy: 0.4970
 5408/25000 [=====>........................] - ETA: 50s - loss: 7.7177 - accuracy: 0.4967
 5440/25000 [=====>........................] - ETA: 49s - loss: 7.7258 - accuracy: 0.4961
 5472/25000 [=====>........................] - ETA: 49s - loss: 7.7058 - accuracy: 0.4974
 5504/25000 [=====>........................] - ETA: 49s - loss: 7.7056 - accuracy: 0.4975
 5536/25000 [=====>........................] - ETA: 49s - loss: 7.7137 - accuracy: 0.4969
 5568/25000 [=====>........................] - ETA: 49s - loss: 7.7189 - accuracy: 0.4966
 5600/25000 [=====>........................] - ETA: 49s - loss: 7.7186 - accuracy: 0.4966
 5632/25000 [=====>........................] - ETA: 49s - loss: 7.7102 - accuracy: 0.4972
 5664/25000 [=====>........................] - ETA: 49s - loss: 7.6991 - accuracy: 0.4979
 5696/25000 [=====>........................] - ETA: 49s - loss: 7.7205 - accuracy: 0.4965
 5728/25000 [=====>........................] - ETA: 49s - loss: 7.7228 - accuracy: 0.4963
 5760/25000 [=====>........................] - ETA: 49s - loss: 7.7199 - accuracy: 0.4965
 5792/25000 [=====>........................] - ETA: 49s - loss: 7.7090 - accuracy: 0.4972
 5824/25000 [=====>........................] - ETA: 49s - loss: 7.7114 - accuracy: 0.4971
 5856/25000 [======>.......................] - ETA: 48s - loss: 7.7138 - accuracy: 0.4969
 5888/25000 [======>.......................] - ETA: 48s - loss: 7.7213 - accuracy: 0.4964
 5920/25000 [======>.......................] - ETA: 48s - loss: 7.7132 - accuracy: 0.4970
 5952/25000 [======>.......................] - ETA: 48s - loss: 7.7259 - accuracy: 0.4961
 5984/25000 [======>.......................] - ETA: 48s - loss: 7.7307 - accuracy: 0.4958
 6016/25000 [======>.......................] - ETA: 48s - loss: 7.7354 - accuracy: 0.4955
 6048/25000 [======>.......................] - ETA: 48s - loss: 7.7401 - accuracy: 0.4952
 6080/25000 [======>.......................] - ETA: 48s - loss: 7.7448 - accuracy: 0.4949
 6112/25000 [======>.......................] - ETA: 48s - loss: 7.7444 - accuracy: 0.4949
 6144/25000 [======>.......................] - ETA: 48s - loss: 7.7440 - accuracy: 0.4950
 6176/25000 [======>.......................] - ETA: 48s - loss: 7.7461 - accuracy: 0.4948
 6208/25000 [======>.......................] - ETA: 48s - loss: 7.7506 - accuracy: 0.4945
 6240/25000 [======>.......................] - ETA: 48s - loss: 7.7526 - accuracy: 0.4944
 6272/25000 [======>.......................] - ETA: 47s - loss: 7.7620 - accuracy: 0.4938
 6304/25000 [======>.......................] - ETA: 47s - loss: 7.7639 - accuracy: 0.4937
 6336/25000 [======>.......................] - ETA: 47s - loss: 7.7610 - accuracy: 0.4938
 6368/25000 [======>.......................] - ETA: 47s - loss: 7.7461 - accuracy: 0.4948
 6400/25000 [======>.......................] - ETA: 47s - loss: 7.7433 - accuracy: 0.4950
 6432/25000 [======>.......................] - ETA: 47s - loss: 7.7358 - accuracy: 0.4955
 6464/25000 [======>.......................] - ETA: 47s - loss: 7.7330 - accuracy: 0.4957
 6496/25000 [======>.......................] - ETA: 47s - loss: 7.7445 - accuracy: 0.4949
 6528/25000 [======>.......................] - ETA: 47s - loss: 7.7441 - accuracy: 0.4949
 6560/25000 [======>.......................] - ETA: 47s - loss: 7.7508 - accuracy: 0.4945
 6592/25000 [======>.......................] - ETA: 47s - loss: 7.7411 - accuracy: 0.4951
 6624/25000 [======>.......................] - ETA: 47s - loss: 7.7384 - accuracy: 0.4953
 6656/25000 [======>.......................] - ETA: 47s - loss: 7.7380 - accuracy: 0.4953
 6688/25000 [=======>......................] - ETA: 46s - loss: 7.7285 - accuracy: 0.4960
 6720/25000 [=======>......................] - ETA: 46s - loss: 7.7259 - accuracy: 0.4961
 6752/25000 [=======>......................] - ETA: 46s - loss: 7.7189 - accuracy: 0.4966
 6784/25000 [=======>......................] - ETA: 46s - loss: 7.7141 - accuracy: 0.4969
 6816/25000 [=======>......................] - ETA: 46s - loss: 7.7229 - accuracy: 0.4963
 6848/25000 [=======>......................] - ETA: 46s - loss: 7.7204 - accuracy: 0.4965
 6880/25000 [=======>......................] - ETA: 46s - loss: 7.7179 - accuracy: 0.4967
 6912/25000 [=======>......................] - ETA: 46s - loss: 7.7176 - accuracy: 0.4967
 6944/25000 [=======>......................] - ETA: 46s - loss: 7.7218 - accuracy: 0.4964
 6976/25000 [=======>......................] - ETA: 46s - loss: 7.7326 - accuracy: 0.4957
 7008/25000 [=======>......................] - ETA: 46s - loss: 7.7323 - accuracy: 0.4957
 7040/25000 [=======>......................] - ETA: 46s - loss: 7.7320 - accuracy: 0.4957
 7072/25000 [=======>......................] - ETA: 46s - loss: 7.7208 - accuracy: 0.4965
 7104/25000 [=======>......................] - ETA: 45s - loss: 7.7184 - accuracy: 0.4966
 7136/25000 [=======>......................] - ETA: 45s - loss: 7.7117 - accuracy: 0.4971
 7168/25000 [=======>......................] - ETA: 45s - loss: 7.7051 - accuracy: 0.4975
 7200/25000 [=======>......................] - ETA: 45s - loss: 7.6943 - accuracy: 0.4982
 7232/25000 [=======>......................] - ETA: 45s - loss: 7.6899 - accuracy: 0.4985
 7264/25000 [=======>......................] - ETA: 45s - loss: 7.7004 - accuracy: 0.4978
 7296/25000 [=======>......................] - ETA: 45s - loss: 7.6855 - accuracy: 0.4988
 7328/25000 [=======>......................] - ETA: 45s - loss: 7.6855 - accuracy: 0.4988
 7360/25000 [=======>......................] - ETA: 45s - loss: 7.6875 - accuracy: 0.4986
 7392/25000 [=======>......................] - ETA: 45s - loss: 7.6791 - accuracy: 0.4992
 7424/25000 [=======>......................] - ETA: 45s - loss: 7.6831 - accuracy: 0.4989
 7456/25000 [=======>......................] - ETA: 45s - loss: 7.6872 - accuracy: 0.4987
 7488/25000 [=======>......................] - ETA: 44s - loss: 7.6912 - accuracy: 0.4984
 7520/25000 [========>.....................] - ETA: 44s - loss: 7.6911 - accuracy: 0.4984
 7552/25000 [========>.....................] - ETA: 44s - loss: 7.6910 - accuracy: 0.4984
 7584/25000 [========>.....................] - ETA: 44s - loss: 7.6909 - accuracy: 0.4984
 7616/25000 [========>.....................] - ETA: 44s - loss: 7.6908 - accuracy: 0.4984
 7648/25000 [========>.....................] - ETA: 44s - loss: 7.6867 - accuracy: 0.4987
 7680/25000 [========>.....................] - ETA: 44s - loss: 7.6866 - accuracy: 0.4987
 7712/25000 [========>.....................] - ETA: 44s - loss: 7.6845 - accuracy: 0.4988
 7744/25000 [========>.....................] - ETA: 44s - loss: 7.6844 - accuracy: 0.4988
 7776/25000 [========>.....................] - ETA: 44s - loss: 7.6863 - accuracy: 0.4987
 7808/25000 [========>.....................] - ETA: 44s - loss: 7.6882 - accuracy: 0.4986
 7840/25000 [========>.....................] - ETA: 44s - loss: 7.6862 - accuracy: 0.4987
 7872/25000 [========>.....................] - ETA: 44s - loss: 7.6958 - accuracy: 0.4981
 7904/25000 [========>.....................] - ETA: 43s - loss: 7.6957 - accuracy: 0.4981
 7936/25000 [========>.....................] - ETA: 43s - loss: 7.7014 - accuracy: 0.4977
 7968/25000 [========>.....................] - ETA: 43s - loss: 7.7109 - accuracy: 0.4971
 8000/25000 [========>.....................] - ETA: 43s - loss: 7.7145 - accuracy: 0.4969
 8032/25000 [========>.....................] - ETA: 43s - loss: 7.7143 - accuracy: 0.4969
 8064/25000 [========>.....................] - ETA: 43s - loss: 7.7218 - accuracy: 0.4964
 8096/25000 [========>.....................] - ETA: 43s - loss: 7.7178 - accuracy: 0.4967
 8128/25000 [========>.....................] - ETA: 43s - loss: 7.7100 - accuracy: 0.4972
 8160/25000 [========>.....................] - ETA: 43s - loss: 7.7117 - accuracy: 0.4971
 8192/25000 [========>.....................] - ETA: 43s - loss: 7.7115 - accuracy: 0.4971
 8224/25000 [========>.....................] - ETA: 43s - loss: 7.7095 - accuracy: 0.4972
 8256/25000 [========>.....................] - ETA: 43s - loss: 7.7038 - accuracy: 0.4976
 8288/25000 [========>.....................] - ETA: 42s - loss: 7.6999 - accuracy: 0.4978
 8320/25000 [========>.....................] - ETA: 42s - loss: 7.6887 - accuracy: 0.4986
 8352/25000 [=========>....................] - ETA: 42s - loss: 7.6831 - accuracy: 0.4989
 8384/25000 [=========>....................] - ETA: 42s - loss: 7.6831 - accuracy: 0.4989
 8416/25000 [=========>....................] - ETA: 42s - loss: 7.6812 - accuracy: 0.4990
 8448/25000 [=========>....................] - ETA: 42s - loss: 7.6793 - accuracy: 0.4992
 8480/25000 [=========>....................] - ETA: 42s - loss: 7.6739 - accuracy: 0.4995
 8512/25000 [=========>....................] - ETA: 42s - loss: 7.6738 - accuracy: 0.4995
 8544/25000 [=========>....................] - ETA: 42s - loss: 7.6738 - accuracy: 0.4995
 8576/25000 [=========>....................] - ETA: 42s - loss: 7.6738 - accuracy: 0.4995
 8608/25000 [=========>....................] - ETA: 42s - loss: 7.6684 - accuracy: 0.4999
 8640/25000 [=========>....................] - ETA: 41s - loss: 7.6631 - accuracy: 0.5002
 8672/25000 [=========>....................] - ETA: 41s - loss: 7.6631 - accuracy: 0.5002
 8704/25000 [=========>....................] - ETA: 41s - loss: 7.6631 - accuracy: 0.5002
 8736/25000 [=========>....................] - ETA: 41s - loss: 7.6666 - accuracy: 0.5000
 8768/25000 [=========>....................] - ETA: 41s - loss: 7.6666 - accuracy: 0.5000
 8800/25000 [=========>....................] - ETA: 41s - loss: 7.6614 - accuracy: 0.5003
 8832/25000 [=========>....................] - ETA: 41s - loss: 7.6701 - accuracy: 0.4998
 8864/25000 [=========>....................] - ETA: 41s - loss: 7.6735 - accuracy: 0.4995
 8896/25000 [=========>....................] - ETA: 41s - loss: 7.6683 - accuracy: 0.4999
 8928/25000 [=========>....................] - ETA: 41s - loss: 7.6718 - accuracy: 0.4997
 8960/25000 [=========>....................] - ETA: 41s - loss: 7.6752 - accuracy: 0.4994
 8992/25000 [=========>....................] - ETA: 41s - loss: 7.6854 - accuracy: 0.4988
 9024/25000 [=========>....................] - ETA: 40s - loss: 7.6938 - accuracy: 0.4982
 9056/25000 [=========>....................] - ETA: 40s - loss: 7.6988 - accuracy: 0.4979
 9088/25000 [=========>....................] - ETA: 40s - loss: 7.6987 - accuracy: 0.4979
 9120/25000 [=========>....................] - ETA: 40s - loss: 7.7002 - accuracy: 0.4978
 9152/25000 [=========>....................] - ETA: 40s - loss: 7.6850 - accuracy: 0.4988
 9184/25000 [==========>...................] - ETA: 40s - loss: 7.6867 - accuracy: 0.4987
 9216/25000 [==========>...................] - ETA: 40s - loss: 7.6849 - accuracy: 0.4988
 9248/25000 [==========>...................] - ETA: 40s - loss: 7.6865 - accuracy: 0.4987
 9280/25000 [==========>...................] - ETA: 40s - loss: 7.6898 - accuracy: 0.4985
 9312/25000 [==========>...................] - ETA: 40s - loss: 7.6897 - accuracy: 0.4985
 9344/25000 [==========>...................] - ETA: 40s - loss: 7.6929 - accuracy: 0.4983
 9376/25000 [==========>...................] - ETA: 39s - loss: 7.6961 - accuracy: 0.4981
 9408/25000 [==========>...................] - ETA: 39s - loss: 7.7008 - accuracy: 0.4978
 9440/25000 [==========>...................] - ETA: 39s - loss: 7.7072 - accuracy: 0.4974
 9472/25000 [==========>...................] - ETA: 39s - loss: 7.7039 - accuracy: 0.4976
 9504/25000 [==========>...................] - ETA: 39s - loss: 7.7021 - accuracy: 0.4977
 9536/25000 [==========>...................] - ETA: 39s - loss: 7.7004 - accuracy: 0.4978
 9568/25000 [==========>...................] - ETA: 39s - loss: 7.6971 - accuracy: 0.4980
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.6938 - accuracy: 0.4982
 9632/25000 [==========>...................] - ETA: 39s - loss: 7.6921 - accuracy: 0.4983
 9664/25000 [==========>...................] - ETA: 39s - loss: 7.6904 - accuracy: 0.4984
 9696/25000 [==========>...................] - ETA: 39s - loss: 7.6935 - accuracy: 0.4982
 9728/25000 [==========>...................] - ETA: 39s - loss: 7.6966 - accuracy: 0.4980
 9760/25000 [==========>...................] - ETA: 38s - loss: 7.6980 - accuracy: 0.4980
 9792/25000 [==========>...................] - ETA: 38s - loss: 7.6964 - accuracy: 0.4981
 9824/25000 [==========>...................] - ETA: 38s - loss: 7.6885 - accuracy: 0.4986
 9856/25000 [==========>...................] - ETA: 38s - loss: 7.6946 - accuracy: 0.4982
 9888/25000 [==========>...................] - ETA: 38s - loss: 7.7023 - accuracy: 0.4977
 9920/25000 [==========>...................] - ETA: 38s - loss: 7.6960 - accuracy: 0.4981
 9952/25000 [==========>...................] - ETA: 38s - loss: 7.6990 - accuracy: 0.4979
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.6989 - accuracy: 0.4979
10016/25000 [===========>..................] - ETA: 38s - loss: 7.7034 - accuracy: 0.4976
10048/25000 [===========>..................] - ETA: 38s - loss: 7.7063 - accuracy: 0.4974
10080/25000 [===========>..................] - ETA: 38s - loss: 7.7107 - accuracy: 0.4971
10112/25000 [===========>..................] - ETA: 38s - loss: 7.7060 - accuracy: 0.4974
10144/25000 [===========>..................] - ETA: 37s - loss: 7.7089 - accuracy: 0.4972
10176/25000 [===========>..................] - ETA: 37s - loss: 7.7043 - accuracy: 0.4975
10208/25000 [===========>..................] - ETA: 37s - loss: 7.6997 - accuracy: 0.4978
10240/25000 [===========>..................] - ETA: 37s - loss: 7.6996 - accuracy: 0.4979
10272/25000 [===========>..................] - ETA: 37s - loss: 7.6995 - accuracy: 0.4979
10304/25000 [===========>..................] - ETA: 37s - loss: 7.6979 - accuracy: 0.4980
10336/25000 [===========>..................] - ETA: 37s - loss: 7.6963 - accuracy: 0.4981
10368/25000 [===========>..................] - ETA: 37s - loss: 7.6932 - accuracy: 0.4983
10400/25000 [===========>..................] - ETA: 37s - loss: 7.6902 - accuracy: 0.4985
10432/25000 [===========>..................] - ETA: 37s - loss: 7.6872 - accuracy: 0.4987
10464/25000 [===========>..................] - ETA: 37s - loss: 7.6901 - accuracy: 0.4985
10496/25000 [===========>..................] - ETA: 36s - loss: 7.6929 - accuracy: 0.4983
10528/25000 [===========>..................] - ETA: 36s - loss: 7.6928 - accuracy: 0.4983
10560/25000 [===========>..................] - ETA: 36s - loss: 7.6899 - accuracy: 0.4985
10592/25000 [===========>..................] - ETA: 36s - loss: 7.6898 - accuracy: 0.4985
10624/25000 [===========>..................] - ETA: 36s - loss: 7.6940 - accuracy: 0.4982
10656/25000 [===========>..................] - ETA: 36s - loss: 7.6997 - accuracy: 0.4978
10688/25000 [===========>..................] - ETA: 36s - loss: 7.7025 - accuracy: 0.4977
10720/25000 [===========>..................] - ETA: 36s - loss: 7.7024 - accuracy: 0.4977
10752/25000 [===========>..................] - ETA: 36s - loss: 7.7037 - accuracy: 0.4976
10784/25000 [===========>..................] - ETA: 36s - loss: 7.6993 - accuracy: 0.4979
10816/25000 [===========>..................] - ETA: 36s - loss: 7.6978 - accuracy: 0.4980
10848/25000 [============>.................] - ETA: 36s - loss: 7.6949 - accuracy: 0.4982
10880/25000 [============>.................] - ETA: 35s - loss: 7.6962 - accuracy: 0.4981
10912/25000 [============>.................] - ETA: 35s - loss: 7.6961 - accuracy: 0.4981
10944/25000 [============>.................] - ETA: 35s - loss: 7.6974 - accuracy: 0.4980
10976/25000 [============>.................] - ETA: 35s - loss: 7.6960 - accuracy: 0.4981
11008/25000 [============>.................] - ETA: 35s - loss: 7.6987 - accuracy: 0.4979
11040/25000 [============>.................] - ETA: 35s - loss: 7.7041 - accuracy: 0.4976
11072/25000 [============>.................] - ETA: 35s - loss: 7.7068 - accuracy: 0.4974
11104/25000 [============>.................] - ETA: 35s - loss: 7.7067 - accuracy: 0.4974
11136/25000 [============>.................] - ETA: 35s - loss: 7.7052 - accuracy: 0.4975
11168/25000 [============>.................] - ETA: 35s - loss: 7.7064 - accuracy: 0.4974
11200/25000 [============>.................] - ETA: 35s - loss: 7.7118 - accuracy: 0.4971
11232/25000 [============>.................] - ETA: 35s - loss: 7.7185 - accuracy: 0.4966
11264/25000 [============>.................] - ETA: 34s - loss: 7.7170 - accuracy: 0.4967
11296/25000 [============>.................] - ETA: 34s - loss: 7.7168 - accuracy: 0.4967
11328/25000 [============>.................] - ETA: 34s - loss: 7.7099 - accuracy: 0.4972
11360/25000 [============>.................] - ETA: 34s - loss: 7.7152 - accuracy: 0.4968
11392/25000 [============>.................] - ETA: 34s - loss: 7.7164 - accuracy: 0.4968
11424/25000 [============>.................] - ETA: 34s - loss: 7.7230 - accuracy: 0.4963
11456/25000 [============>.................] - ETA: 34s - loss: 7.7148 - accuracy: 0.4969
11488/25000 [============>.................] - ETA: 34s - loss: 7.7147 - accuracy: 0.4969
11520/25000 [============>.................] - ETA: 34s - loss: 7.7145 - accuracy: 0.4969
11552/25000 [============>.................] - ETA: 34s - loss: 7.7104 - accuracy: 0.4971
11584/25000 [============>.................] - ETA: 34s - loss: 7.7024 - accuracy: 0.4977
11616/25000 [============>.................] - ETA: 34s - loss: 7.7089 - accuracy: 0.4972
11648/25000 [============>.................] - ETA: 33s - loss: 7.7035 - accuracy: 0.4976
11680/25000 [=============>................] - ETA: 33s - loss: 7.7073 - accuracy: 0.4973
11712/25000 [=============>................] - ETA: 33s - loss: 7.7085 - accuracy: 0.4973
11744/25000 [=============>................] - ETA: 33s - loss: 7.7071 - accuracy: 0.4974
11776/25000 [=============>................] - ETA: 33s - loss: 7.7109 - accuracy: 0.4971
11808/25000 [=============>................] - ETA: 33s - loss: 7.7043 - accuracy: 0.4975
11840/25000 [=============>................] - ETA: 33s - loss: 7.7068 - accuracy: 0.4974
11872/25000 [=============>................] - ETA: 33s - loss: 7.7092 - accuracy: 0.4972
11904/25000 [=============>................] - ETA: 33s - loss: 7.7027 - accuracy: 0.4976
11936/25000 [=============>................] - ETA: 33s - loss: 7.7026 - accuracy: 0.4977
11968/25000 [=============>................] - ETA: 33s - loss: 7.7038 - accuracy: 0.4976
12000/25000 [=============>................] - ETA: 33s - loss: 7.7011 - accuracy: 0.4978
12032/25000 [=============>................] - ETA: 32s - loss: 7.7023 - accuracy: 0.4977
12064/25000 [=============>................] - ETA: 32s - loss: 7.7022 - accuracy: 0.4977
12096/25000 [=============>................] - ETA: 32s - loss: 7.7046 - accuracy: 0.4975
12128/25000 [=============>................] - ETA: 32s - loss: 7.7045 - accuracy: 0.4975
12160/25000 [=============>................] - ETA: 32s - loss: 7.7120 - accuracy: 0.4970
12192/25000 [=============>................] - ETA: 32s - loss: 7.7106 - accuracy: 0.4971
12224/25000 [=============>................] - ETA: 32s - loss: 7.7118 - accuracy: 0.4971
12256/25000 [=============>................] - ETA: 32s - loss: 7.7129 - accuracy: 0.4970
12288/25000 [=============>................] - ETA: 32s - loss: 7.7115 - accuracy: 0.4971
12320/25000 [=============>................] - ETA: 32s - loss: 7.7152 - accuracy: 0.4968
12352/25000 [=============>................] - ETA: 32s - loss: 7.7200 - accuracy: 0.4965
12384/25000 [=============>................] - ETA: 32s - loss: 7.7186 - accuracy: 0.4966
12416/25000 [=============>................] - ETA: 31s - loss: 7.7160 - accuracy: 0.4968
12448/25000 [=============>................] - ETA: 31s - loss: 7.7184 - accuracy: 0.4966
12480/25000 [=============>................] - ETA: 31s - loss: 7.7145 - accuracy: 0.4969
12512/25000 [==============>...............] - ETA: 31s - loss: 7.7144 - accuracy: 0.4969
12544/25000 [==============>...............] - ETA: 31s - loss: 7.7094 - accuracy: 0.4972
12576/25000 [==============>...............] - ETA: 31s - loss: 7.7069 - accuracy: 0.4974
12608/25000 [==============>...............] - ETA: 31s - loss: 7.7140 - accuracy: 0.4969
12640/25000 [==============>...............] - ETA: 31s - loss: 7.7067 - accuracy: 0.4974
12672/25000 [==============>...............] - ETA: 31s - loss: 7.7065 - accuracy: 0.4974
12704/25000 [==============>...............] - ETA: 31s - loss: 7.7089 - accuracy: 0.4972
12736/25000 [==============>...............] - ETA: 31s - loss: 7.7112 - accuracy: 0.4971
12768/25000 [==============>...............] - ETA: 31s - loss: 7.7099 - accuracy: 0.4972
12800/25000 [==============>...............] - ETA: 30s - loss: 7.7109 - accuracy: 0.4971
12832/25000 [==============>...............] - ETA: 30s - loss: 7.7072 - accuracy: 0.4974
12864/25000 [==============>...............] - ETA: 30s - loss: 7.7036 - accuracy: 0.4976
12896/25000 [==============>...............] - ETA: 30s - loss: 7.7011 - accuracy: 0.4978
12928/25000 [==============>...............] - ETA: 30s - loss: 7.6963 - accuracy: 0.4981
12960/25000 [==============>...............] - ETA: 30s - loss: 7.6974 - accuracy: 0.4980
12992/25000 [==============>...............] - ETA: 30s - loss: 7.6973 - accuracy: 0.4980
13024/25000 [==============>...............] - ETA: 30s - loss: 7.6949 - accuracy: 0.4982
13056/25000 [==============>...............] - ETA: 30s - loss: 7.6948 - accuracy: 0.4982
13088/25000 [==============>...............] - ETA: 30s - loss: 7.6947 - accuracy: 0.4982
13120/25000 [==============>...............] - ETA: 30s - loss: 7.6958 - accuracy: 0.4981
13152/25000 [==============>...............] - ETA: 30s - loss: 7.6923 - accuracy: 0.4983
13184/25000 [==============>...............] - ETA: 29s - loss: 7.6922 - accuracy: 0.4983
13216/25000 [==============>...............] - ETA: 29s - loss: 7.6921 - accuracy: 0.4983
13248/25000 [==============>...............] - ETA: 29s - loss: 7.6898 - accuracy: 0.4985
13280/25000 [==============>...............] - ETA: 29s - loss: 7.6874 - accuracy: 0.4986
13312/25000 [==============>...............] - ETA: 29s - loss: 7.6920 - accuracy: 0.4983
13344/25000 [===============>..............] - ETA: 29s - loss: 7.6873 - accuracy: 0.4987
13376/25000 [===============>..............] - ETA: 29s - loss: 7.6895 - accuracy: 0.4985
13408/25000 [===============>..............] - ETA: 29s - loss: 7.6838 - accuracy: 0.4989
13440/25000 [===============>..............] - ETA: 29s - loss: 7.6792 - accuracy: 0.4992
13472/25000 [===============>..............] - ETA: 29s - loss: 7.6803 - accuracy: 0.4991
13504/25000 [===============>..............] - ETA: 29s - loss: 7.6825 - accuracy: 0.4990
13536/25000 [===============>..............] - ETA: 29s - loss: 7.6847 - accuracy: 0.4988
13568/25000 [===============>..............] - ETA: 28s - loss: 7.6802 - accuracy: 0.4991
13600/25000 [===============>..............] - ETA: 28s - loss: 7.6801 - accuracy: 0.4991
13632/25000 [===============>..............] - ETA: 28s - loss: 7.6835 - accuracy: 0.4989
13664/25000 [===============>..............] - ETA: 28s - loss: 7.6790 - accuracy: 0.4992
13696/25000 [===============>..............] - ETA: 28s - loss: 7.6733 - accuracy: 0.4996
13728/25000 [===============>..............] - ETA: 28s - loss: 7.6744 - accuracy: 0.4995
13760/25000 [===============>..............] - ETA: 28s - loss: 7.6711 - accuracy: 0.4997
13792/25000 [===============>..............] - ETA: 28s - loss: 7.6755 - accuracy: 0.4994
13824/25000 [===============>..............] - ETA: 28s - loss: 7.6766 - accuracy: 0.4993
13856/25000 [===============>..............] - ETA: 28s - loss: 7.6788 - accuracy: 0.4992
13888/25000 [===============>..............] - ETA: 28s - loss: 7.6777 - accuracy: 0.4993
13920/25000 [===============>..............] - ETA: 28s - loss: 7.6732 - accuracy: 0.4996
13952/25000 [===============>..............] - ETA: 28s - loss: 7.6743 - accuracy: 0.4995
13984/25000 [===============>..............] - ETA: 27s - loss: 7.6721 - accuracy: 0.4996
14016/25000 [===============>..............] - ETA: 27s - loss: 7.6699 - accuracy: 0.4998
14048/25000 [===============>..............] - ETA: 27s - loss: 7.6688 - accuracy: 0.4999
14080/25000 [===============>..............] - ETA: 27s - loss: 7.6677 - accuracy: 0.4999
14112/25000 [===============>..............] - ETA: 27s - loss: 7.6699 - accuracy: 0.4998
14144/25000 [===============>..............] - ETA: 27s - loss: 7.6699 - accuracy: 0.4998
14176/25000 [================>.............] - ETA: 27s - loss: 7.6699 - accuracy: 0.4998
14208/25000 [================>.............] - ETA: 27s - loss: 7.6677 - accuracy: 0.4999
14240/25000 [================>.............] - ETA: 27s - loss: 7.6688 - accuracy: 0.4999
14272/25000 [================>.............] - ETA: 27s - loss: 7.6677 - accuracy: 0.4999
14304/25000 [================>.............] - ETA: 27s - loss: 7.6655 - accuracy: 0.5001
14336/25000 [================>.............] - ETA: 27s - loss: 7.6698 - accuracy: 0.4998
14368/25000 [================>.............] - ETA: 26s - loss: 7.6730 - accuracy: 0.4996
14400/25000 [================>.............] - ETA: 26s - loss: 7.6783 - accuracy: 0.4992
14432/25000 [================>.............] - ETA: 26s - loss: 7.6762 - accuracy: 0.4994
14464/25000 [================>.............] - ETA: 26s - loss: 7.6698 - accuracy: 0.4998
14496/25000 [================>.............] - ETA: 26s - loss: 7.6698 - accuracy: 0.4998
14528/25000 [================>.............] - ETA: 26s - loss: 7.6719 - accuracy: 0.4997
14560/25000 [================>.............] - ETA: 26s - loss: 7.6750 - accuracy: 0.4995
14592/25000 [================>.............] - ETA: 26s - loss: 7.6708 - accuracy: 0.4997
14624/25000 [================>.............] - ETA: 26s - loss: 7.6729 - accuracy: 0.4996
14656/25000 [================>.............] - ETA: 26s - loss: 7.6719 - accuracy: 0.4997
14688/25000 [================>.............] - ETA: 26s - loss: 7.6708 - accuracy: 0.4997
14720/25000 [================>.............] - ETA: 26s - loss: 7.6697 - accuracy: 0.4998
14752/25000 [================>.............] - ETA: 25s - loss: 7.6718 - accuracy: 0.4997
14784/25000 [================>.............] - ETA: 25s - loss: 7.6677 - accuracy: 0.4999
14816/25000 [================>.............] - ETA: 25s - loss: 7.6697 - accuracy: 0.4998
14848/25000 [================>.............] - ETA: 25s - loss: 7.6697 - accuracy: 0.4998
14880/25000 [================>.............] - ETA: 25s - loss: 7.6707 - accuracy: 0.4997
14912/25000 [================>.............] - ETA: 25s - loss: 7.6687 - accuracy: 0.4999
14944/25000 [================>.............] - ETA: 25s - loss: 7.6728 - accuracy: 0.4996
14976/25000 [================>.............] - ETA: 25s - loss: 7.6769 - accuracy: 0.4993
15008/25000 [=================>............] - ETA: 25s - loss: 7.6830 - accuracy: 0.4989
15040/25000 [=================>............] - ETA: 25s - loss: 7.6840 - accuracy: 0.4989
15072/25000 [=================>............] - ETA: 25s - loss: 7.6768 - accuracy: 0.4993
15104/25000 [=================>............] - ETA: 25s - loss: 7.6788 - accuracy: 0.4992
15136/25000 [=================>............] - ETA: 24s - loss: 7.6798 - accuracy: 0.4991
15168/25000 [=================>............] - ETA: 24s - loss: 7.6747 - accuracy: 0.4995
15200/25000 [=================>............] - ETA: 24s - loss: 7.6767 - accuracy: 0.4993
15232/25000 [=================>............] - ETA: 24s - loss: 7.6737 - accuracy: 0.4995
15264/25000 [=================>............] - ETA: 24s - loss: 7.6757 - accuracy: 0.4994
15296/25000 [=================>............] - ETA: 24s - loss: 7.6746 - accuracy: 0.4995
15328/25000 [=================>............] - ETA: 24s - loss: 7.6766 - accuracy: 0.4993
15360/25000 [=================>............] - ETA: 24s - loss: 7.6766 - accuracy: 0.4993
15392/25000 [=================>............] - ETA: 24s - loss: 7.6756 - accuracy: 0.4994
15424/25000 [=================>............] - ETA: 24s - loss: 7.6766 - accuracy: 0.4994
15456/25000 [=================>............] - ETA: 24s - loss: 7.6765 - accuracy: 0.4994
15488/25000 [=================>............] - ETA: 24s - loss: 7.6795 - accuracy: 0.4992
15520/25000 [=================>............] - ETA: 23s - loss: 7.6795 - accuracy: 0.4992
15552/25000 [=================>............] - ETA: 23s - loss: 7.6775 - accuracy: 0.4993
15584/25000 [=================>............] - ETA: 23s - loss: 7.6804 - accuracy: 0.4991
15616/25000 [=================>............] - ETA: 23s - loss: 7.6902 - accuracy: 0.4985
15648/25000 [=================>............] - ETA: 23s - loss: 7.6931 - accuracy: 0.4983
15680/25000 [=================>............] - ETA: 23s - loss: 7.6920 - accuracy: 0.4983
15712/25000 [=================>............] - ETA: 23s - loss: 7.6910 - accuracy: 0.4984
15744/25000 [=================>............] - ETA: 23s - loss: 7.6900 - accuracy: 0.4985
15776/25000 [=================>............] - ETA: 23s - loss: 7.6841 - accuracy: 0.4989
15808/25000 [=================>............] - ETA: 23s - loss: 7.6841 - accuracy: 0.4989
15840/25000 [==================>...........] - ETA: 23s - loss: 7.6889 - accuracy: 0.4985
15872/25000 [==================>...........] - ETA: 23s - loss: 7.6898 - accuracy: 0.4985
15904/25000 [==================>...........] - ETA: 23s - loss: 7.6917 - accuracy: 0.4984
15936/25000 [==================>...........] - ETA: 22s - loss: 7.6907 - accuracy: 0.4984
15968/25000 [==================>...........] - ETA: 22s - loss: 7.6849 - accuracy: 0.4988
16000/25000 [==================>...........] - ETA: 22s - loss: 7.6896 - accuracy: 0.4985
16032/25000 [==================>...........] - ETA: 22s - loss: 7.6905 - accuracy: 0.4984
16064/25000 [==================>...........] - ETA: 22s - loss: 7.6943 - accuracy: 0.4982
16096/25000 [==================>...........] - ETA: 22s - loss: 7.6933 - accuracy: 0.4983
16128/25000 [==================>...........] - ETA: 22s - loss: 7.6942 - accuracy: 0.4982
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6932 - accuracy: 0.4983
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6950 - accuracy: 0.4981
16224/25000 [==================>...........] - ETA: 22s - loss: 7.6969 - accuracy: 0.4980
16256/25000 [==================>...........] - ETA: 22s - loss: 7.6968 - accuracy: 0.4980
16288/25000 [==================>...........] - ETA: 22s - loss: 7.6949 - accuracy: 0.4982
16320/25000 [==================>...........] - ETA: 21s - loss: 7.7014 - accuracy: 0.4977
16352/25000 [==================>...........] - ETA: 21s - loss: 7.7041 - accuracy: 0.4976
16384/25000 [==================>...........] - ETA: 21s - loss: 7.7003 - accuracy: 0.4978
16416/25000 [==================>...........] - ETA: 21s - loss: 7.7040 - accuracy: 0.4976
16448/25000 [==================>...........] - ETA: 21s - loss: 7.7039 - accuracy: 0.4976
16480/25000 [==================>...........] - ETA: 21s - loss: 7.7066 - accuracy: 0.4974
16512/25000 [==================>...........] - ETA: 21s - loss: 7.7056 - accuracy: 0.4975
16544/25000 [==================>...........] - ETA: 21s - loss: 7.7065 - accuracy: 0.4974
16576/25000 [==================>...........] - ETA: 21s - loss: 7.7092 - accuracy: 0.4972
16608/25000 [==================>...........] - ETA: 21s - loss: 7.7082 - accuracy: 0.4973
16640/25000 [==================>...........] - ETA: 21s - loss: 7.7072 - accuracy: 0.4974
16672/25000 [===================>..........] - ETA: 21s - loss: 7.7052 - accuracy: 0.4975
16704/25000 [===================>..........] - ETA: 20s - loss: 7.7079 - accuracy: 0.4973
16736/25000 [===================>..........] - ETA: 20s - loss: 7.7078 - accuracy: 0.4973
16768/25000 [===================>..........] - ETA: 20s - loss: 7.7087 - accuracy: 0.4973
16800/25000 [===================>..........] - ETA: 20s - loss: 7.7077 - accuracy: 0.4973
16832/25000 [===================>..........] - ETA: 20s - loss: 7.7103 - accuracy: 0.4971
16864/25000 [===================>..........] - ETA: 20s - loss: 7.7084 - accuracy: 0.4973
16896/25000 [===================>..........] - ETA: 20s - loss: 7.7111 - accuracy: 0.4971
16928/25000 [===================>..........] - ETA: 20s - loss: 7.7092 - accuracy: 0.4972
16960/25000 [===================>..........] - ETA: 20s - loss: 7.7091 - accuracy: 0.4972
16992/25000 [===================>..........] - ETA: 20s - loss: 7.7054 - accuracy: 0.4975
17024/25000 [===================>..........] - ETA: 20s - loss: 7.7072 - accuracy: 0.4974
17056/25000 [===================>..........] - ETA: 20s - loss: 7.7080 - accuracy: 0.4973
17088/25000 [===================>..........] - ETA: 20s - loss: 7.7061 - accuracy: 0.4974
17120/25000 [===================>..........] - ETA: 19s - loss: 7.7024 - accuracy: 0.4977
17152/25000 [===================>..........] - ETA: 19s - loss: 7.7042 - accuracy: 0.4976
17184/25000 [===================>..........] - ETA: 19s - loss: 7.7050 - accuracy: 0.4975
17216/25000 [===================>..........] - ETA: 19s - loss: 7.7112 - accuracy: 0.4971
17248/25000 [===================>..........] - ETA: 19s - loss: 7.7093 - accuracy: 0.4972
17280/25000 [===================>..........] - ETA: 19s - loss: 7.7110 - accuracy: 0.4971
17312/25000 [===================>..........] - ETA: 19s - loss: 7.7047 - accuracy: 0.4975
17344/25000 [===================>..........] - ETA: 19s - loss: 7.7064 - accuracy: 0.4974
17376/25000 [===================>..........] - ETA: 19s - loss: 7.7072 - accuracy: 0.4974
17408/25000 [===================>..........] - ETA: 19s - loss: 7.7063 - accuracy: 0.4974
17440/25000 [===================>..........] - ETA: 19s - loss: 7.7044 - accuracy: 0.4975
17472/25000 [===================>..........] - ETA: 19s - loss: 7.7035 - accuracy: 0.4976
17504/25000 [====================>.........] - ETA: 18s - loss: 7.6999 - accuracy: 0.4978
17536/25000 [====================>.........] - ETA: 18s - loss: 7.6990 - accuracy: 0.4979
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6989 - accuracy: 0.4979
17600/25000 [====================>.........] - ETA: 18s - loss: 7.6971 - accuracy: 0.4980
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6953 - accuracy: 0.4981
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6970 - accuracy: 0.4980
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6926 - accuracy: 0.4983
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6874 - accuracy: 0.4986
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6882 - accuracy: 0.4986
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6890 - accuracy: 0.4985
17824/25000 [====================>.........] - ETA: 18s - loss: 7.6873 - accuracy: 0.4987
17856/25000 [====================>.........] - ETA: 18s - loss: 7.6958 - accuracy: 0.4981
17888/25000 [====================>.........] - ETA: 17s - loss: 7.6958 - accuracy: 0.4981
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6966 - accuracy: 0.4980
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6982 - accuracy: 0.4979
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6948 - accuracy: 0.4982
18016/25000 [====================>.........] - ETA: 17s - loss: 7.6998 - accuracy: 0.4978
18048/25000 [====================>.........] - ETA: 17s - loss: 7.7049 - accuracy: 0.4975
18080/25000 [====================>.........] - ETA: 17s - loss: 7.7056 - accuracy: 0.4975
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6996 - accuracy: 0.4978
18144/25000 [====================>.........] - ETA: 17s - loss: 7.7004 - accuracy: 0.4978
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6978 - accuracy: 0.4980
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6961 - accuracy: 0.4981
18240/25000 [====================>.........] - ETA: 17s - loss: 7.6935 - accuracy: 0.4982
18272/25000 [====================>.........] - ETA: 17s - loss: 7.6918 - accuracy: 0.4984
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6918 - accuracy: 0.4984
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6909 - accuracy: 0.4984
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6933 - accuracy: 0.4983
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6900 - accuracy: 0.4985
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6891 - accuracy: 0.4985
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6882 - accuracy: 0.4986
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6865 - accuracy: 0.4987
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6840 - accuracy: 0.4989
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6848 - accuracy: 0.4988
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6815 - accuracy: 0.4990
18624/25000 [=====================>........] - ETA: 16s - loss: 7.6839 - accuracy: 0.4989
18656/25000 [=====================>........] - ETA: 16s - loss: 7.6822 - accuracy: 0.4990
18688/25000 [=====================>........] - ETA: 15s - loss: 7.6871 - accuracy: 0.4987
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6904 - accuracy: 0.4985
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6862 - accuracy: 0.4987
18784/25000 [=====================>........] - ETA: 15s - loss: 7.6838 - accuracy: 0.4989
18816/25000 [=====================>........] - ETA: 15s - loss: 7.6862 - accuracy: 0.4987
18848/25000 [=====================>........] - ETA: 15s - loss: 7.6821 - accuracy: 0.4990
18880/25000 [=====================>........] - ETA: 15s - loss: 7.6845 - accuracy: 0.4988
18912/25000 [=====================>........] - ETA: 15s - loss: 7.6845 - accuracy: 0.4988
18944/25000 [=====================>........] - ETA: 15s - loss: 7.6852 - accuracy: 0.4988
18976/25000 [=====================>........] - ETA: 15s - loss: 7.6860 - accuracy: 0.4987
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6860 - accuracy: 0.4987
19040/25000 [=====================>........] - ETA: 15s - loss: 7.6868 - accuracy: 0.4987
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6883 - accuracy: 0.4986
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6899 - accuracy: 0.4985
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6891 - accuracy: 0.4985
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6906 - accuracy: 0.4984
19200/25000 [======================>.......] - ETA: 14s - loss: 7.6906 - accuracy: 0.4984
19232/25000 [======================>.......] - ETA: 14s - loss: 7.6913 - accuracy: 0.4984
19264/25000 [======================>.......] - ETA: 14s - loss: 7.6905 - accuracy: 0.4984
19296/25000 [======================>.......] - ETA: 14s - loss: 7.6905 - accuracy: 0.4984
19328/25000 [======================>.......] - ETA: 14s - loss: 7.6912 - accuracy: 0.4984
19360/25000 [======================>.......] - ETA: 14s - loss: 7.6912 - accuracy: 0.4984
19392/25000 [======================>.......] - ETA: 14s - loss: 7.6895 - accuracy: 0.4985
19424/25000 [======================>.......] - ETA: 14s - loss: 7.6871 - accuracy: 0.4987
19456/25000 [======================>.......] - ETA: 14s - loss: 7.6871 - accuracy: 0.4987
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6887 - accuracy: 0.4986
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6855 - accuracy: 0.4988
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6878 - accuracy: 0.4986
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6925 - accuracy: 0.4983
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6955 - accuracy: 0.4981
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6955 - accuracy: 0.4981
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6978 - accuracy: 0.4980
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6970 - accuracy: 0.4980
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6915 - accuracy: 0.4984
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6868 - accuracy: 0.4987
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6867 - accuracy: 0.4987
19840/25000 [======================>.......] - ETA: 13s - loss: 7.6906 - accuracy: 0.4984
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6867 - accuracy: 0.4987
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6851 - accuracy: 0.4988
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6858 - accuracy: 0.4987
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6850 - accuracy: 0.4988
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6843 - accuracy: 0.4988
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6850 - accuracy: 0.4988
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6834 - accuracy: 0.4989
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6826 - accuracy: 0.4990
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6849 - accuracy: 0.4988
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6818 - accuracy: 0.4990
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6818 - accuracy: 0.4990
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6818 - accuracy: 0.4990
20256/25000 [=======================>......] - ETA: 12s - loss: 7.6818 - accuracy: 0.4990
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6795 - accuracy: 0.4992
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6802 - accuracy: 0.4991
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6794 - accuracy: 0.4992
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6756 - accuracy: 0.4994
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6741 - accuracy: 0.4995
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6711 - accuracy: 0.4997
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6726 - accuracy: 0.4996
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6726 - accuracy: 0.4996
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6711 - accuracy: 0.4997
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6703 - accuracy: 0.4998
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6703 - accuracy: 0.4998
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6696 - accuracy: 0.4998
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6696 - accuracy: 0.4998
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6703 - accuracy: 0.4998
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6681 - accuracy: 0.4999
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6659 - accuracy: 0.5000
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6674 - accuracy: 0.5000
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6688 - accuracy: 0.4999
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6696 - accuracy: 0.4998
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6644 - accuracy: 0.5001
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6637 - accuracy: 0.5002
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6630 - accuracy: 0.5002
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6644 - accuracy: 0.5001
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6630 - accuracy: 0.5002 
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6615 - accuracy: 0.5003
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6637 - accuracy: 0.5002
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6659 - accuracy: 0.5000
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6659 - accuracy: 0.5000
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6630 - accuracy: 0.5002
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6630 - accuracy: 0.5002
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6587 - accuracy: 0.5005
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6580 - accuracy: 0.5006
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6587 - accuracy: 0.5005
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6573 - accuracy: 0.5006
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6544 - accuracy: 0.5008
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6537 - accuracy: 0.5008
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6552 - accuracy: 0.5007
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6581 - accuracy: 0.5006
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6581 - accuracy: 0.5006
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6602 - accuracy: 0.5004
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6609 - accuracy: 0.5004
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6652 - accuracy: 0.5001
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6680 - accuracy: 0.4999
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6694 - accuracy: 0.4998
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6687 - accuracy: 0.4999
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6680 - accuracy: 0.4999
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6652 - accuracy: 0.5001
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6645 - accuracy: 0.5001
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6652 - accuracy: 0.5001
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6603 - accuracy: 0.5004
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6596 - accuracy: 0.5005
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6610 - accuracy: 0.5004
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6694 - accuracy: 0.4998
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6736 - accuracy: 0.4995
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6736 - accuracy: 0.4995
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6749 - accuracy: 0.4995
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6742 - accuracy: 0.4995
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6756 - accuracy: 0.4994
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6742 - accuracy: 0.4995
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6756 - accuracy: 0.4994
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6763 - accuracy: 0.4994
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6804 - accuracy: 0.4991
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6797 - accuracy: 0.4991
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6796 - accuracy: 0.4992
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6810 - accuracy: 0.4991
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6796 - accuracy: 0.4992
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6823 - accuracy: 0.4990
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6837 - accuracy: 0.4989
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6850 - accuracy: 0.4988
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6816 - accuracy: 0.4990
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6849 - accuracy: 0.4988
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6822 - accuracy: 0.4990
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6822 - accuracy: 0.4990
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6815 - accuracy: 0.4990
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6794 - accuracy: 0.4992
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6774 - accuracy: 0.4993
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6760 - accuracy: 0.4994
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6760 - accuracy: 0.4994
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6800 - accuracy: 0.4991
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6787 - accuracy: 0.4992
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6793 - accuracy: 0.4992
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6786 - accuracy: 0.4992
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6786 - accuracy: 0.4992
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6746 - accuracy: 0.4995
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6739 - accuracy: 0.4995
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6693 - accuracy: 0.4998
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6693 - accuracy: 0.4998
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6693 - accuracy: 0.4998
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6673 - accuracy: 0.5000
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6640 - accuracy: 0.5002
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6653 - accuracy: 0.5001
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6633 - accuracy: 0.5002
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6634 - accuracy: 0.5002
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6686 - accuracy: 0.4999
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6718 - accuracy: 0.4997
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6731 - accuracy: 0.4996
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6756 - accuracy: 0.4994
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6737 - accuracy: 0.4995
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6756 - accuracy: 0.4994
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6737 - accuracy: 0.4995
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6756 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6711 - accuracy: 0.4997
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6711 - accuracy: 0.4997
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6742 - accuracy: 0.4995
24192/25000 [============================>.] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
24224/25000 [============================>.] - ETA: 1s - loss: 7.6711 - accuracy: 0.4997
24256/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24288/25000 [============================>.] - ETA: 1s - loss: 7.6704 - accuracy: 0.4998
24320/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24352/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24384/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24416/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24448/25000 [============================>.] - ETA: 1s - loss: 7.6603 - accuracy: 0.5004
24480/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24512/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24544/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24576/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24608/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24640/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24672/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24704/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24736/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24768/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24800/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24864/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24896/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24928/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24960/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
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
Task exception was never retrieved
future: <Task finished coro=<InProcConnector.connect() done, defined at /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/distributed/comm/inproc.py:285> exception=OSError("no endpoint for inproc address '10.1.0.4/4004/1'",)>
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/distributed/comm/inproc.py", line 288, in connect
    raise IOError("no endpoint for inproc address %r" % (address,))
OSError: no endpoint for inproc address '10.1.0.4/4004/1'





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
