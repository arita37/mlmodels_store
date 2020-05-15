
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/0ca7fc10154e30acfd3477806bcaa34404fe1bf2', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '0ca7fc10154e30acfd3477806bcaa34404fe1bf2', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/0ca7fc10154e30acfd3477806bcaa34404fe1bf2

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/0ca7fc10154e30acfd3477806bcaa34404fe1bf2

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
	Data preprocessing and feature engineering runtime = 0.28s ...
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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:56<01:24, 28.12s/it] 40%|████      | 2/5 [00:56<01:24, 28.13s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.3163817645444318, 'embedding_size_factor': 1.0361603867700455, 'layers.choice': 1, 'learning_rate': 0.0010606910914101027, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.2510992671718976e-09} and reward: 0.3872
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd4?\x99L\xf19\x0cX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x94\x1c\xe9\xe9h\xaeX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?Q`\xdcc\x1e\xf6\xf9X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x15~d\x8caMau.' and reward: 0.3872
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd4?\x99L\xf19\x0cX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x94\x1c\xe9\xe9h\xaeX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?Q`\xdcc\x1e\xf6\xf9X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x15~d\x8caMau.' and reward: 0.3872
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 128.83893609046936
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.3163817645444318, 'embedding_size_factor': 1.0361603867700455, 'layers.choice': 1, 'learning_rate': 0.0010606910914101027, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.2510992671718976e-09}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.72s of the -10.77s of remaining time.
Ensemble size: 23
Ensemble weights: 
[0.34782609 0.65217391]
	0.3892	 = Validation accuracy score
	0.97s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 131.78s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f95c8622ba8> 

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
 [-1.16734952e-01  8.07499513e-02 -5.08865528e-02 -1.52565181e-01
   1.10781655e-01  9.52472985e-02]
 [ 2.28853256e-01  7.01136664e-02  7.91653097e-02  1.89897224e-01
  -9.77850705e-02 -6.23661317e-02]
 [ 1.10915862e-04  1.32338936e-03  1.06458887e-01 -3.17638963e-02
   1.16364136e-01  2.51219384e-02]
 [-1.30882472e-01  3.61409277e-01 -8.85773078e-03 -1.42445430e-01
   2.55000979e-01  1.48087367e-02]
 [ 2.39085406e-01 -2.37155229e-01 -9.62603390e-02 -2.97910243e-01
  -3.54750395e-01  3.96091491e-01]
 [ 2.68676996e-01  2.56793022e-01  4.84501064e-01 -2.68437937e-02
  -2.62779951e-01  3.51434439e-01]
 [ 6.79921284e-02  6.72676191e-02 -3.88401262e-02 -4.47019875e-01
   5.99182732e-02 -8.47851858e-03]
 [ 8.81076932e-01  3.76966357e-01 -2.77561218e-01  1.06530480e-01
   7.03281999e-01 -2.36851752e-01]
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
{'loss': 0.49476630985736847, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 14:18:08.222600: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.5259299501776695, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 14:18:09.481632: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 3063808/17464789 [====>.........................] - ETA: 0s
 9666560/17464789 [===============>..............] - ETA: 0s
16130048/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 14:18:22.019326: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 14:18:22.023966: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-15 14:18:22.024219: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559586c94db0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 14:18:22.024240: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:57 - loss: 8.6249 - accuracy: 0.4375
   64/25000 [..............................] - ETA: 3:13 - loss: 9.5833 - accuracy: 0.3750
   96/25000 [..............................] - ETA: 2:34 - loss: 8.7847 - accuracy: 0.4271
  128/25000 [..............................] - ETA: 2:18 - loss: 8.3854 - accuracy: 0.4531
  160/25000 [..............................] - ETA: 2:07 - loss: 8.6249 - accuracy: 0.4375
  192/25000 [..............................] - ETA: 2:00 - loss: 8.2256 - accuracy: 0.4635
  224/25000 [..............................] - ETA: 1:54 - loss: 7.9404 - accuracy: 0.4821
  256/25000 [..............................] - ETA: 1:49 - loss: 8.2057 - accuracy: 0.4648
  288/25000 [..............................] - ETA: 1:45 - loss: 8.0393 - accuracy: 0.4757
  320/25000 [..............................] - ETA: 1:43 - loss: 8.2895 - accuracy: 0.4594
  352/25000 [..............................] - ETA: 1:41 - loss: 8.0587 - accuracy: 0.4744
  384/25000 [..............................] - ETA: 1:39 - loss: 7.9062 - accuracy: 0.4844
  416/25000 [..............................] - ETA: 1:37 - loss: 7.8878 - accuracy: 0.4856
  448/25000 [..............................] - ETA: 1:36 - loss: 7.7693 - accuracy: 0.4933
  480/25000 [..............................] - ETA: 1:35 - loss: 7.7305 - accuracy: 0.4958
  512/25000 [..............................] - ETA: 1:34 - loss: 7.6666 - accuracy: 0.5000
  544/25000 [..............................] - ETA: 1:34 - loss: 7.6948 - accuracy: 0.4982
  576/25000 [..............................] - ETA: 1:33 - loss: 7.7465 - accuracy: 0.4948
  608/25000 [..............................] - ETA: 1:32 - loss: 7.6666 - accuracy: 0.5000
  640/25000 [..............................] - ETA: 1:32 - loss: 7.6427 - accuracy: 0.5016
  672/25000 [..............................] - ETA: 1:31 - loss: 7.6894 - accuracy: 0.4985
  704/25000 [..............................] - ETA: 1:30 - loss: 7.6666 - accuracy: 0.5000
  736/25000 [..............................] - ETA: 1:30 - loss: 7.6666 - accuracy: 0.5000
  768/25000 [..............................] - ETA: 1:29 - loss: 7.6866 - accuracy: 0.4987
  800/25000 [..............................] - ETA: 1:29 - loss: 7.6858 - accuracy: 0.4988
  832/25000 [..............................] - ETA: 1:28 - loss: 7.5929 - accuracy: 0.5048
  864/25000 [>.............................] - ETA: 1:27 - loss: 7.5601 - accuracy: 0.5069
  896/25000 [>.............................] - ETA: 1:27 - loss: 7.4955 - accuracy: 0.5112
  928/25000 [>.............................] - ETA: 1:26 - loss: 7.4849 - accuracy: 0.5119
  960/25000 [>.............................] - ETA: 1:26 - loss: 7.4909 - accuracy: 0.5115
  992/25000 [>.............................] - ETA: 1:26 - loss: 7.4657 - accuracy: 0.5131
 1024/25000 [>.............................] - ETA: 1:26 - loss: 7.5019 - accuracy: 0.5107
 1056/25000 [>.............................] - ETA: 1:25 - loss: 7.5069 - accuracy: 0.5104
 1088/25000 [>.............................] - ETA: 1:25 - loss: 7.4975 - accuracy: 0.5110
 1120/25000 [>.............................] - ETA: 1:25 - loss: 7.4613 - accuracy: 0.5134
 1152/25000 [>.............................] - ETA: 1:25 - loss: 7.4004 - accuracy: 0.5174
 1184/25000 [>.............................] - ETA: 1:24 - loss: 7.4206 - accuracy: 0.5160
 1216/25000 [>.............................] - ETA: 1:24 - loss: 7.4523 - accuracy: 0.5140
 1248/25000 [>.............................] - ETA: 1:24 - loss: 7.4578 - accuracy: 0.5136
 1280/25000 [>.............................] - ETA: 1:23 - loss: 7.4390 - accuracy: 0.5148
 1312/25000 [>.............................] - ETA: 1:23 - loss: 7.4212 - accuracy: 0.5160
 1344/25000 [>.............................] - ETA: 1:23 - loss: 7.3928 - accuracy: 0.5179
 1376/25000 [>.............................] - ETA: 1:23 - loss: 7.4438 - accuracy: 0.5145
 1408/25000 [>.............................] - ETA: 1:23 - loss: 7.4161 - accuracy: 0.5163
 1440/25000 [>.............................] - ETA: 1:22 - loss: 7.3898 - accuracy: 0.5181
 1472/25000 [>.............................] - ETA: 1:22 - loss: 7.4062 - accuracy: 0.5170
 1504/25000 [>.............................] - ETA: 1:22 - loss: 7.4117 - accuracy: 0.5166
 1536/25000 [>.............................] - ETA: 1:22 - loss: 7.4270 - accuracy: 0.5156
 1568/25000 [>.............................] - ETA: 1:22 - loss: 7.4124 - accuracy: 0.5166
 1600/25000 [>.............................] - ETA: 1:22 - loss: 7.4750 - accuracy: 0.5125
 1632/25000 [>.............................] - ETA: 1:21 - loss: 7.5257 - accuracy: 0.5092
 1664/25000 [>.............................] - ETA: 1:21 - loss: 7.5192 - accuracy: 0.5096
 1696/25000 [=>............................] - ETA: 1:21 - loss: 7.5400 - accuracy: 0.5083
 1728/25000 [=>............................] - ETA: 1:21 - loss: 7.5601 - accuracy: 0.5069
 1760/25000 [=>............................] - ETA: 1:21 - loss: 7.5447 - accuracy: 0.5080
 1792/25000 [=>............................] - ETA: 1:21 - loss: 7.5040 - accuracy: 0.5106
 1824/25000 [=>............................] - ETA: 1:21 - loss: 7.4396 - accuracy: 0.5148
 1856/25000 [=>............................] - ETA: 1:20 - loss: 7.4436 - accuracy: 0.5145
 1888/25000 [=>............................] - ETA: 1:20 - loss: 7.4473 - accuracy: 0.5143
 1920/25000 [=>............................] - ETA: 1:20 - loss: 7.4350 - accuracy: 0.5151
 1952/25000 [=>............................] - ETA: 1:20 - loss: 7.4310 - accuracy: 0.5154
 1984/25000 [=>............................] - ETA: 1:20 - loss: 7.4502 - accuracy: 0.5141
 2016/25000 [=>............................] - ETA: 1:20 - loss: 7.4384 - accuracy: 0.5149
 2048/25000 [=>............................] - ETA: 1:19 - loss: 7.4570 - accuracy: 0.5137
 2080/25000 [=>............................] - ETA: 1:19 - loss: 7.4676 - accuracy: 0.5130
 2112/25000 [=>............................] - ETA: 1:19 - loss: 7.4416 - accuracy: 0.5147
 2144/25000 [=>............................] - ETA: 1:19 - loss: 7.4592 - accuracy: 0.5135
 2176/25000 [=>............................] - ETA: 1:19 - loss: 7.4482 - accuracy: 0.5142
 2208/25000 [=>............................] - ETA: 1:19 - loss: 7.4444 - accuracy: 0.5145
 2240/25000 [=>............................] - ETA: 1:19 - loss: 7.4476 - accuracy: 0.5143
 2272/25000 [=>............................] - ETA: 1:19 - loss: 7.4372 - accuracy: 0.5150
 2304/25000 [=>............................] - ETA: 1:19 - loss: 7.4270 - accuracy: 0.5156
 2336/25000 [=>............................] - ETA: 1:18 - loss: 7.4369 - accuracy: 0.5150
 2368/25000 [=>............................] - ETA: 1:18 - loss: 7.4400 - accuracy: 0.5148
 2400/25000 [=>............................] - ETA: 1:18 - loss: 7.4558 - accuracy: 0.5138
 2432/25000 [=>............................] - ETA: 1:18 - loss: 7.4333 - accuracy: 0.5152
 2464/25000 [=>............................] - ETA: 1:18 - loss: 7.4799 - accuracy: 0.5122
 2496/25000 [=>............................] - ETA: 1:18 - loss: 7.4885 - accuracy: 0.5116
 2528/25000 [==>...........................] - ETA: 1:18 - loss: 7.4665 - accuracy: 0.5131
 2560/25000 [==>...........................] - ETA: 1:17 - loss: 7.4450 - accuracy: 0.5145
 2592/25000 [==>...........................] - ETA: 1:17 - loss: 7.4359 - accuracy: 0.5150
 2624/25000 [==>...........................] - ETA: 1:17 - loss: 7.4446 - accuracy: 0.5145
 2656/25000 [==>...........................] - ETA: 1:17 - loss: 7.4703 - accuracy: 0.5128
 2688/25000 [==>...........................] - ETA: 1:17 - loss: 7.4841 - accuracy: 0.5119
 2720/25000 [==>...........................] - ETA: 1:17 - loss: 7.4637 - accuracy: 0.5132
 2752/25000 [==>...........................] - ETA: 1:17 - loss: 7.4605 - accuracy: 0.5134
 2784/25000 [==>...........................] - ETA: 1:17 - loss: 7.4628 - accuracy: 0.5133
 2816/25000 [==>...........................] - ETA: 1:16 - loss: 7.4924 - accuracy: 0.5114
 2848/25000 [==>...........................] - ETA: 1:16 - loss: 7.4782 - accuracy: 0.5123
 2880/25000 [==>...........................] - ETA: 1:16 - loss: 7.4696 - accuracy: 0.5128
 2912/25000 [==>...........................] - ETA: 1:16 - loss: 7.4613 - accuracy: 0.5134
 2944/25000 [==>...........................] - ETA: 1:16 - loss: 7.4583 - accuracy: 0.5136
 2976/25000 [==>...........................] - ETA: 1:16 - loss: 7.4451 - accuracy: 0.5144
 3008/25000 [==>...........................] - ETA: 1:16 - loss: 7.4576 - accuracy: 0.5136
 3040/25000 [==>...........................] - ETA: 1:16 - loss: 7.4598 - accuracy: 0.5135
 3072/25000 [==>...........................] - ETA: 1:15 - loss: 7.4670 - accuracy: 0.5130
 3104/25000 [==>...........................] - ETA: 1:15 - loss: 7.4789 - accuracy: 0.5122
 3136/25000 [==>...........................] - ETA: 1:15 - loss: 7.4613 - accuracy: 0.5134
 3168/25000 [==>...........................] - ETA: 1:15 - loss: 7.4537 - accuracy: 0.5139
 3200/25000 [==>...........................] - ETA: 1:15 - loss: 7.4510 - accuracy: 0.5141
 3232/25000 [==>...........................] - ETA: 1:15 - loss: 7.4531 - accuracy: 0.5139
 3264/25000 [==>...........................] - ETA: 1:15 - loss: 7.4599 - accuracy: 0.5135
 3296/25000 [==>...........................] - ETA: 1:15 - loss: 7.4805 - accuracy: 0.5121
 3328/25000 [==>...........................] - ETA: 1:14 - loss: 7.4639 - accuracy: 0.5132
 3360/25000 [===>..........................] - ETA: 1:14 - loss: 7.4521 - accuracy: 0.5140
 3392/25000 [===>..........................] - ETA: 1:14 - loss: 7.4542 - accuracy: 0.5139
 3424/25000 [===>..........................] - ETA: 1:14 - loss: 7.4382 - accuracy: 0.5149
 3456/25000 [===>..........................] - ETA: 1:14 - loss: 7.4581 - accuracy: 0.5136
 3488/25000 [===>..........................] - ETA: 1:14 - loss: 7.4556 - accuracy: 0.5138
 3520/25000 [===>..........................] - ETA: 1:14 - loss: 7.4357 - accuracy: 0.5151
 3552/25000 [===>..........................] - ETA: 1:14 - loss: 7.4206 - accuracy: 0.5160
 3584/25000 [===>..........................] - ETA: 1:14 - loss: 7.4185 - accuracy: 0.5162
 3616/25000 [===>..........................] - ETA: 1:13 - loss: 7.4249 - accuracy: 0.5158
 3648/25000 [===>..........................] - ETA: 1:13 - loss: 7.4228 - accuracy: 0.5159
 3680/25000 [===>..........................] - ETA: 1:13 - loss: 7.4250 - accuracy: 0.5158
 3712/25000 [===>..........................] - ETA: 1:13 - loss: 7.4270 - accuracy: 0.5156
 3744/25000 [===>..........................] - ETA: 1:13 - loss: 7.4332 - accuracy: 0.5152
 3776/25000 [===>..........................] - ETA: 1:13 - loss: 7.4352 - accuracy: 0.5151
 3808/25000 [===>..........................] - ETA: 1:13 - loss: 7.4291 - accuracy: 0.5155
 3840/25000 [===>..........................] - ETA: 1:12 - loss: 7.4350 - accuracy: 0.5151
 3872/25000 [===>..........................] - ETA: 1:12 - loss: 7.4449 - accuracy: 0.5145
 3904/25000 [===>..........................] - ETA: 1:12 - loss: 7.4624 - accuracy: 0.5133
 3936/25000 [===>..........................] - ETA: 1:12 - loss: 7.4563 - accuracy: 0.5137
 3968/25000 [===>..........................] - ETA: 1:12 - loss: 7.4425 - accuracy: 0.5146
 4000/25000 [===>..........................] - ETA: 1:12 - loss: 7.4328 - accuracy: 0.5153
 4032/25000 [===>..........................] - ETA: 1:12 - loss: 7.4308 - accuracy: 0.5154
 4064/25000 [===>..........................] - ETA: 1:12 - loss: 7.4327 - accuracy: 0.5153
 4096/25000 [===>..........................] - ETA: 1:11 - loss: 7.4270 - accuracy: 0.5156
 4128/25000 [===>..........................] - ETA: 1:11 - loss: 7.4326 - accuracy: 0.5153
 4160/25000 [===>..........................] - ETA: 1:11 - loss: 7.4492 - accuracy: 0.5142
 4192/25000 [====>.........................] - ETA: 1:11 - loss: 7.4362 - accuracy: 0.5150
 4224/25000 [====>.........................] - ETA: 1:11 - loss: 7.4379 - accuracy: 0.5149
 4256/25000 [====>.........................] - ETA: 1:11 - loss: 7.4469 - accuracy: 0.5143
 4288/25000 [====>.........................] - ETA: 1:11 - loss: 7.4342 - accuracy: 0.5152
 4320/25000 [====>.........................] - ETA: 1:10 - loss: 7.4395 - accuracy: 0.5148
 4352/25000 [====>.........................] - ETA: 1:10 - loss: 7.4376 - accuracy: 0.5149
 4384/25000 [====>.........................] - ETA: 1:10 - loss: 7.4323 - accuracy: 0.5153
 4416/25000 [====>.........................] - ETA: 1:10 - loss: 7.4513 - accuracy: 0.5140
 4448/25000 [====>.........................] - ETA: 1:10 - loss: 7.4494 - accuracy: 0.5142
 4480/25000 [====>.........................] - ETA: 1:10 - loss: 7.4407 - accuracy: 0.5147
 4512/25000 [====>.........................] - ETA: 1:10 - loss: 7.4593 - accuracy: 0.5135
 4544/25000 [====>.........................] - ETA: 1:09 - loss: 7.4540 - accuracy: 0.5139
 4576/25000 [====>.........................] - ETA: 1:09 - loss: 7.4723 - accuracy: 0.5127
 4608/25000 [====>.........................] - ETA: 1:09 - loss: 7.4836 - accuracy: 0.5119
 4640/25000 [====>.........................] - ETA: 1:09 - loss: 7.4948 - accuracy: 0.5112
 4672/25000 [====>.........................] - ETA: 1:09 - loss: 7.4992 - accuracy: 0.5109
 4704/25000 [====>.........................] - ETA: 1:09 - loss: 7.4971 - accuracy: 0.5111
 4736/25000 [====>.........................] - ETA: 1:09 - loss: 7.5080 - accuracy: 0.5103
 4768/25000 [====>.........................] - ETA: 1:09 - loss: 7.5026 - accuracy: 0.5107
 4800/25000 [====>.........................] - ETA: 1:09 - loss: 7.5069 - accuracy: 0.5104
 4832/25000 [====>.........................] - ETA: 1:09 - loss: 7.5048 - accuracy: 0.5106
 4864/25000 [====>.........................] - ETA: 1:08 - loss: 7.5058 - accuracy: 0.5105
 4896/25000 [====>.........................] - ETA: 1:08 - loss: 7.5069 - accuracy: 0.5104
 4928/25000 [====>.........................] - ETA: 1:08 - loss: 7.5048 - accuracy: 0.5106
 4960/25000 [====>.........................] - ETA: 1:08 - loss: 7.5059 - accuracy: 0.5105
 4992/25000 [====>.........................] - ETA: 1:08 - loss: 7.5223 - accuracy: 0.5094
 5024/25000 [=====>........................] - ETA: 1:08 - loss: 7.5232 - accuracy: 0.5094
 5056/25000 [=====>........................] - ETA: 1:08 - loss: 7.5241 - accuracy: 0.5093
 5088/25000 [=====>........................] - ETA: 1:08 - loss: 7.5280 - accuracy: 0.5090
 5120/25000 [=====>........................] - ETA: 1:07 - loss: 7.5259 - accuracy: 0.5092
 5152/25000 [=====>........................] - ETA: 1:07 - loss: 7.5327 - accuracy: 0.5087
 5184/25000 [=====>........................] - ETA: 1:07 - loss: 7.5276 - accuracy: 0.5091
 5216/25000 [=====>........................] - ETA: 1:07 - loss: 7.5285 - accuracy: 0.5090
 5248/25000 [=====>........................] - ETA: 1:07 - loss: 7.5410 - accuracy: 0.5082
 5280/25000 [=====>........................] - ETA: 1:07 - loss: 7.5505 - accuracy: 0.5076
 5312/25000 [=====>........................] - ETA: 1:07 - loss: 7.5483 - accuracy: 0.5077
 5344/25000 [=====>........................] - ETA: 1:07 - loss: 7.5490 - accuracy: 0.5077
 5376/25000 [=====>........................] - ETA: 1:07 - loss: 7.5497 - accuracy: 0.5076
 5408/25000 [=====>........................] - ETA: 1:07 - loss: 7.5390 - accuracy: 0.5083
 5440/25000 [=====>........................] - ETA: 1:06 - loss: 7.5370 - accuracy: 0.5085
 5472/25000 [=====>........................] - ETA: 1:06 - loss: 7.5517 - accuracy: 0.5075
 5504/25000 [=====>........................] - ETA: 1:06 - loss: 7.5273 - accuracy: 0.5091
 5536/25000 [=====>........................] - ETA: 1:06 - loss: 7.5309 - accuracy: 0.5089
 5568/25000 [=====>........................] - ETA: 1:06 - loss: 7.5344 - accuracy: 0.5086
 5600/25000 [=====>........................] - ETA: 1:06 - loss: 7.5407 - accuracy: 0.5082
 5632/25000 [=====>........................] - ETA: 1:06 - loss: 7.5332 - accuracy: 0.5087
 5664/25000 [=====>........................] - ETA: 1:06 - loss: 7.5286 - accuracy: 0.5090
 5696/25000 [=====>........................] - ETA: 1:05 - loss: 7.5213 - accuracy: 0.5095
 5728/25000 [=====>........................] - ETA: 1:05 - loss: 7.5274 - accuracy: 0.5091
 5760/25000 [=====>........................] - ETA: 1:05 - loss: 7.5149 - accuracy: 0.5099
 5792/25000 [=====>........................] - ETA: 1:05 - loss: 7.5104 - accuracy: 0.5102
 5824/25000 [=====>........................] - ETA: 1:05 - loss: 7.5245 - accuracy: 0.5093
 5856/25000 [======>.......................] - ETA: 1:05 - loss: 7.5148 - accuracy: 0.5099
 5888/25000 [======>.......................] - ETA: 1:05 - loss: 7.5078 - accuracy: 0.5104
 5920/25000 [======>.......................] - ETA: 1:05 - loss: 7.5164 - accuracy: 0.5098
 5952/25000 [======>.......................] - ETA: 1:05 - loss: 7.5224 - accuracy: 0.5094
 5984/25000 [======>.......................] - ETA: 1:04 - loss: 7.5513 - accuracy: 0.5075
 6016/25000 [======>.......................] - ETA: 1:04 - loss: 7.5494 - accuracy: 0.5076
 6048/25000 [======>.......................] - ETA: 1:04 - loss: 7.5399 - accuracy: 0.5083
 6080/25000 [======>.......................] - ETA: 1:04 - loss: 7.5456 - accuracy: 0.5079
 6112/25000 [======>.......................] - ETA: 1:04 - loss: 7.5512 - accuracy: 0.5075
 6144/25000 [======>.......................] - ETA: 1:04 - loss: 7.5618 - accuracy: 0.5068
 6176/25000 [======>.......................] - ETA: 1:04 - loss: 7.5574 - accuracy: 0.5071
 6208/25000 [======>.......................] - ETA: 1:04 - loss: 7.5678 - accuracy: 0.5064
 6240/25000 [======>.......................] - ETA: 1:04 - loss: 7.5634 - accuracy: 0.5067
 6272/25000 [======>.......................] - ETA: 1:03 - loss: 7.5615 - accuracy: 0.5069
 6304/25000 [======>.......................] - ETA: 1:03 - loss: 7.5547 - accuracy: 0.5073
 6336/25000 [======>.......................] - ETA: 1:03 - loss: 7.5505 - accuracy: 0.5076
 6368/25000 [======>.......................] - ETA: 1:03 - loss: 7.5486 - accuracy: 0.5077
 6400/25000 [======>.......................] - ETA: 1:03 - loss: 7.5564 - accuracy: 0.5072
 6432/25000 [======>.......................] - ETA: 1:03 - loss: 7.5522 - accuracy: 0.5075
 6464/25000 [======>.......................] - ETA: 1:03 - loss: 7.5456 - accuracy: 0.5079
 6496/25000 [======>.......................] - ETA: 1:03 - loss: 7.5462 - accuracy: 0.5079
 6528/25000 [======>.......................] - ETA: 1:03 - loss: 7.5421 - accuracy: 0.5081
 6560/25000 [======>.......................] - ETA: 1:03 - loss: 7.5404 - accuracy: 0.5082
 6592/25000 [======>.......................] - ETA: 1:02 - loss: 7.5433 - accuracy: 0.5080
 6624/25000 [======>.......................] - ETA: 1:02 - loss: 7.5370 - accuracy: 0.5085
 6656/25000 [======>.......................] - ETA: 1:02 - loss: 7.5422 - accuracy: 0.5081
 6688/25000 [=======>......................] - ETA: 1:02 - loss: 7.5405 - accuracy: 0.5082
 6720/25000 [=======>......................] - ETA: 1:02 - loss: 7.5503 - accuracy: 0.5076
 6752/25000 [=======>......................] - ETA: 1:02 - loss: 7.5485 - accuracy: 0.5077
 6784/25000 [=======>......................] - ETA: 1:02 - loss: 7.5446 - accuracy: 0.5080
 6816/25000 [=======>......................] - ETA: 1:02 - loss: 7.5361 - accuracy: 0.5085
 6848/25000 [=======>......................] - ETA: 1:01 - loss: 7.5479 - accuracy: 0.5077
 6880/25000 [=======>......................] - ETA: 1:01 - loss: 7.5463 - accuracy: 0.5078
 6912/25000 [=======>......................] - ETA: 1:01 - loss: 7.5535 - accuracy: 0.5074
 6944/25000 [=======>......................] - ETA: 1:01 - loss: 7.5540 - accuracy: 0.5073
 6976/25000 [=======>......................] - ETA: 1:01 - loss: 7.5655 - accuracy: 0.5066
 7008/25000 [=======>......................] - ETA: 1:01 - loss: 7.5682 - accuracy: 0.5064
 7040/25000 [=======>......................] - ETA: 1:01 - loss: 7.5839 - accuracy: 0.5054
 7072/25000 [=======>......................] - ETA: 1:01 - loss: 7.5821 - accuracy: 0.5055
 7104/25000 [=======>......................] - ETA: 1:00 - loss: 7.5760 - accuracy: 0.5059
 7136/25000 [=======>......................] - ETA: 1:00 - loss: 7.5721 - accuracy: 0.5062
 7168/25000 [=======>......................] - ETA: 1:00 - loss: 7.5704 - accuracy: 0.5063
 7200/25000 [=======>......................] - ETA: 1:00 - loss: 7.5687 - accuracy: 0.5064
 7232/25000 [=======>......................] - ETA: 1:00 - loss: 7.5649 - accuracy: 0.5066
 7264/25000 [=======>......................] - ETA: 1:00 - loss: 7.5674 - accuracy: 0.5065
 7296/25000 [=======>......................] - ETA: 1:00 - loss: 7.5763 - accuracy: 0.5059
 7328/25000 [=======>......................] - ETA: 1:00 - loss: 7.5787 - accuracy: 0.5057
 7360/25000 [=======>......................] - ETA: 1:00 - loss: 7.5812 - accuracy: 0.5056
 7392/25000 [=======>......................] - ETA: 59s - loss: 7.5774 - accuracy: 0.5058 
 7424/25000 [=======>......................] - ETA: 59s - loss: 7.5695 - accuracy: 0.5063
 7456/25000 [=======>......................] - ETA: 59s - loss: 7.5638 - accuracy: 0.5067
 7488/25000 [=======>......................] - ETA: 59s - loss: 7.5601 - accuracy: 0.5069
 7520/25000 [========>.....................] - ETA: 59s - loss: 7.5586 - accuracy: 0.5070
 7552/25000 [========>.....................] - ETA: 59s - loss: 7.5651 - accuracy: 0.5066
 7584/25000 [========>.....................] - ETA: 59s - loss: 7.5574 - accuracy: 0.5071
 7616/25000 [========>.....................] - ETA: 59s - loss: 7.5579 - accuracy: 0.5071
 7648/25000 [========>.....................] - ETA: 58s - loss: 7.5564 - accuracy: 0.5072
 7680/25000 [========>.....................] - ETA: 58s - loss: 7.5508 - accuracy: 0.5076
 7712/25000 [========>.....................] - ETA: 58s - loss: 7.5453 - accuracy: 0.5079
 7744/25000 [========>.....................] - ETA: 58s - loss: 7.5399 - accuracy: 0.5083
 7776/25000 [========>.....................] - ETA: 58s - loss: 7.5444 - accuracy: 0.5080
 7808/25000 [========>.....................] - ETA: 58s - loss: 7.5508 - accuracy: 0.5076
 7840/25000 [========>.....................] - ETA: 58s - loss: 7.5454 - accuracy: 0.5079
 7872/25000 [========>.....................] - ETA: 58s - loss: 7.5361 - accuracy: 0.5085
 7904/25000 [========>.....................] - ETA: 57s - loss: 7.5425 - accuracy: 0.5081
 7936/25000 [========>.....................] - ETA: 57s - loss: 7.5430 - accuracy: 0.5081
 7968/25000 [========>.....................] - ETA: 57s - loss: 7.5338 - accuracy: 0.5087
 8000/25000 [========>.....................] - ETA: 57s - loss: 7.5363 - accuracy: 0.5085
 8032/25000 [========>.....................] - ETA: 57s - loss: 7.5330 - accuracy: 0.5087
 8064/25000 [========>.....................] - ETA: 57s - loss: 7.5297 - accuracy: 0.5089
 8096/25000 [========>.....................] - ETA: 57s - loss: 7.5303 - accuracy: 0.5089
 8128/25000 [========>.....................] - ETA: 57s - loss: 7.5308 - accuracy: 0.5089
 8160/25000 [========>.....................] - ETA: 56s - loss: 7.5370 - accuracy: 0.5085
 8192/25000 [========>.....................] - ETA: 56s - loss: 7.5412 - accuracy: 0.5082
 8224/25000 [========>.....................] - ETA: 56s - loss: 7.5492 - accuracy: 0.5077
 8256/25000 [========>.....................] - ETA: 56s - loss: 7.5533 - accuracy: 0.5074
 8288/25000 [========>.....................] - ETA: 56s - loss: 7.5575 - accuracy: 0.5071
 8320/25000 [========>.....................] - ETA: 56s - loss: 7.5616 - accuracy: 0.5069
 8352/25000 [=========>....................] - ETA: 56s - loss: 7.5620 - accuracy: 0.5068
 8384/25000 [=========>....................] - ETA: 56s - loss: 7.5679 - accuracy: 0.5064
 8416/25000 [=========>....................] - ETA: 55s - loss: 7.5773 - accuracy: 0.5058
 8448/25000 [=========>....................] - ETA: 55s - loss: 7.5795 - accuracy: 0.5057
 8480/25000 [=========>....................] - ETA: 55s - loss: 7.5780 - accuracy: 0.5058
 8512/25000 [=========>....................] - ETA: 55s - loss: 7.5838 - accuracy: 0.5054
 8544/25000 [=========>....................] - ETA: 55s - loss: 7.5823 - accuracy: 0.5055
 8576/25000 [=========>....................] - ETA: 55s - loss: 7.5826 - accuracy: 0.5055
 8608/25000 [=========>....................] - ETA: 55s - loss: 7.5793 - accuracy: 0.5057
 8640/25000 [=========>....................] - ETA: 55s - loss: 7.5761 - accuracy: 0.5059
 8672/25000 [=========>....................] - ETA: 54s - loss: 7.5764 - accuracy: 0.5059
 8704/25000 [=========>....................] - ETA: 54s - loss: 7.5750 - accuracy: 0.5060
 8736/25000 [=========>....................] - ETA: 54s - loss: 7.5789 - accuracy: 0.5057
 8768/25000 [=========>....................] - ETA: 54s - loss: 7.5739 - accuracy: 0.5060
 8800/25000 [=========>....................] - ETA: 54s - loss: 7.5725 - accuracy: 0.5061
 8832/25000 [=========>....................] - ETA: 54s - loss: 7.5694 - accuracy: 0.5063
 8864/25000 [=========>....................] - ETA: 54s - loss: 7.5732 - accuracy: 0.5061
 8896/25000 [=========>....................] - ETA: 54s - loss: 7.5770 - accuracy: 0.5058
 8928/25000 [=========>....................] - ETA: 54s - loss: 7.5756 - accuracy: 0.5059
 8960/25000 [=========>....................] - ETA: 53s - loss: 7.5776 - accuracy: 0.5058
 8992/25000 [=========>....................] - ETA: 53s - loss: 7.5745 - accuracy: 0.5060
 9024/25000 [=========>....................] - ETA: 53s - loss: 7.5817 - accuracy: 0.5055
 9056/25000 [=========>....................] - ETA: 53s - loss: 7.5803 - accuracy: 0.5056
 9088/25000 [=========>....................] - ETA: 53s - loss: 7.5856 - accuracy: 0.5053
 9120/25000 [=========>....................] - ETA: 53s - loss: 7.5876 - accuracy: 0.5052
 9152/25000 [=========>....................] - ETA: 53s - loss: 7.5828 - accuracy: 0.5055
 9184/25000 [==========>...................] - ETA: 53s - loss: 7.5798 - accuracy: 0.5057
 9216/25000 [==========>...................] - ETA: 52s - loss: 7.5818 - accuracy: 0.5055
 9248/25000 [==========>...................] - ETA: 52s - loss: 7.5787 - accuracy: 0.5057
 9280/25000 [==========>...................] - ETA: 52s - loss: 7.5807 - accuracy: 0.5056
 9312/25000 [==========>...................] - ETA: 52s - loss: 7.5793 - accuracy: 0.5057
 9344/25000 [==========>...................] - ETA: 52s - loss: 7.5747 - accuracy: 0.5060
 9376/25000 [==========>...................] - ETA: 52s - loss: 7.5767 - accuracy: 0.5059
 9408/25000 [==========>...................] - ETA: 52s - loss: 7.5851 - accuracy: 0.5053
 9440/25000 [==========>...................] - ETA: 52s - loss: 7.5805 - accuracy: 0.5056
 9472/25000 [==========>...................] - ETA: 52s - loss: 7.5808 - accuracy: 0.5056
 9504/25000 [==========>...................] - ETA: 51s - loss: 7.5747 - accuracy: 0.5060
 9536/25000 [==========>...................] - ETA: 51s - loss: 7.5734 - accuracy: 0.5061
 9568/25000 [==========>...................] - ETA: 51s - loss: 7.5769 - accuracy: 0.5059
 9600/25000 [==========>...................] - ETA: 51s - loss: 7.5836 - accuracy: 0.5054
 9632/25000 [==========>...................] - ETA: 51s - loss: 7.5854 - accuracy: 0.5053
 9664/25000 [==========>...................] - ETA: 51s - loss: 7.5794 - accuracy: 0.5057
 9696/25000 [==========>...................] - ETA: 51s - loss: 7.5749 - accuracy: 0.5060
 9728/25000 [==========>...................] - ETA: 51s - loss: 7.5752 - accuracy: 0.5060
 9760/25000 [==========>...................] - ETA: 50s - loss: 7.5755 - accuracy: 0.5059
 9792/25000 [==========>...................] - ETA: 50s - loss: 7.5821 - accuracy: 0.5055
 9824/25000 [==========>...................] - ETA: 50s - loss: 7.5792 - accuracy: 0.5057
 9856/25000 [==========>...................] - ETA: 50s - loss: 7.5842 - accuracy: 0.5054
 9888/25000 [==========>...................] - ETA: 50s - loss: 7.5844 - accuracy: 0.5054
 9920/25000 [==========>...................] - ETA: 50s - loss: 7.5832 - accuracy: 0.5054
 9952/25000 [==========>...................] - ETA: 50s - loss: 7.5850 - accuracy: 0.5053
 9984/25000 [==========>...................] - ETA: 50s - loss: 7.5898 - accuracy: 0.5050
10016/25000 [===========>..................] - ETA: 50s - loss: 7.5901 - accuracy: 0.5050
10048/25000 [===========>..................] - ETA: 49s - loss: 7.5888 - accuracy: 0.5051
10080/25000 [===========>..................] - ETA: 49s - loss: 7.5860 - accuracy: 0.5053
10112/25000 [===========>..................] - ETA: 49s - loss: 7.5863 - accuracy: 0.5052
10144/25000 [===========>..................] - ETA: 49s - loss: 7.5910 - accuracy: 0.5049
10176/25000 [===========>..................] - ETA: 49s - loss: 7.5943 - accuracy: 0.5047
10208/25000 [===========>..................] - ETA: 49s - loss: 7.5990 - accuracy: 0.5044
10240/25000 [===========>..................] - ETA: 49s - loss: 7.6007 - accuracy: 0.5043
10272/25000 [===========>..................] - ETA: 49s - loss: 7.5980 - accuracy: 0.5045
10304/25000 [===========>..................] - ETA: 48s - loss: 7.5997 - accuracy: 0.5044
10336/25000 [===========>..................] - ETA: 48s - loss: 7.6028 - accuracy: 0.5042
10368/25000 [===========>..................] - ETA: 48s - loss: 7.5986 - accuracy: 0.5044
10400/25000 [===========>..................] - ETA: 48s - loss: 7.5973 - accuracy: 0.5045
10432/25000 [===========>..................] - ETA: 48s - loss: 7.5931 - accuracy: 0.5048
10464/25000 [===========>..................] - ETA: 48s - loss: 7.5919 - accuracy: 0.5049
10496/25000 [===========>..................] - ETA: 48s - loss: 7.5892 - accuracy: 0.5050
10528/25000 [===========>..................] - ETA: 48s - loss: 7.5851 - accuracy: 0.5053
10560/25000 [===========>..................] - ETA: 48s - loss: 7.5868 - accuracy: 0.5052
10592/25000 [===========>..................] - ETA: 47s - loss: 7.5942 - accuracy: 0.5047
10624/25000 [===========>..................] - ETA: 47s - loss: 7.5872 - accuracy: 0.5052
10656/25000 [===========>..................] - ETA: 47s - loss: 7.5875 - accuracy: 0.5052
10688/25000 [===========>..................] - ETA: 47s - loss: 7.5805 - accuracy: 0.5056
10720/25000 [===========>..................] - ETA: 47s - loss: 7.5765 - accuracy: 0.5059
10752/25000 [===========>..................] - ETA: 47s - loss: 7.5796 - accuracy: 0.5057
10784/25000 [===========>..................] - ETA: 47s - loss: 7.5756 - accuracy: 0.5059
10816/25000 [===========>..................] - ETA: 47s - loss: 7.5716 - accuracy: 0.5062
10848/25000 [============>.................] - ETA: 47s - loss: 7.5649 - accuracy: 0.5066
10880/25000 [============>.................] - ETA: 46s - loss: 7.5750 - accuracy: 0.5060
10912/25000 [============>.................] - ETA: 46s - loss: 7.5697 - accuracy: 0.5063
10944/25000 [============>.................] - ETA: 46s - loss: 7.5741 - accuracy: 0.5060
10976/25000 [============>.................] - ETA: 46s - loss: 7.5786 - accuracy: 0.5057
11008/25000 [============>.................] - ETA: 46s - loss: 7.5775 - accuracy: 0.5058
11040/25000 [============>.................] - ETA: 46s - loss: 7.5777 - accuracy: 0.5058
11072/25000 [============>.................] - ETA: 46s - loss: 7.5849 - accuracy: 0.5053
11104/25000 [============>.................] - ETA: 46s - loss: 7.5865 - accuracy: 0.5052
11136/25000 [============>.................] - ETA: 46s - loss: 7.5854 - accuracy: 0.5053
11168/25000 [============>.................] - ETA: 45s - loss: 7.5815 - accuracy: 0.5056
11200/25000 [============>.................] - ETA: 45s - loss: 7.5845 - accuracy: 0.5054
11232/25000 [============>.................] - ETA: 45s - loss: 7.5793 - accuracy: 0.5057
11264/25000 [============>.................] - ETA: 45s - loss: 7.5768 - accuracy: 0.5059
11296/25000 [============>.................] - ETA: 45s - loss: 7.5825 - accuracy: 0.5055
11328/25000 [============>.................] - ETA: 45s - loss: 7.5854 - accuracy: 0.5053
11360/25000 [============>.................] - ETA: 45s - loss: 7.5802 - accuracy: 0.5056
11392/25000 [============>.................] - ETA: 45s - loss: 7.5818 - accuracy: 0.5055
11424/25000 [============>.................] - ETA: 45s - loss: 7.5847 - accuracy: 0.5053
11456/25000 [============>.................] - ETA: 44s - loss: 7.5810 - accuracy: 0.5056
11488/25000 [============>.................] - ETA: 44s - loss: 7.5799 - accuracy: 0.5057
11520/25000 [============>.................] - ETA: 44s - loss: 7.5801 - accuracy: 0.5056
11552/25000 [============>.................] - ETA: 44s - loss: 7.5777 - accuracy: 0.5058
11584/25000 [============>.................] - ETA: 44s - loss: 7.5806 - accuracy: 0.5056
11616/25000 [============>.................] - ETA: 44s - loss: 7.5782 - accuracy: 0.5058
11648/25000 [============>.................] - ETA: 44s - loss: 7.5758 - accuracy: 0.5059
11680/25000 [=============>................] - ETA: 44s - loss: 7.5774 - accuracy: 0.5058
11712/25000 [=============>................] - ETA: 44s - loss: 7.5763 - accuracy: 0.5059
11744/25000 [=============>................] - ETA: 43s - loss: 7.5791 - accuracy: 0.5057
11776/25000 [=============>................] - ETA: 43s - loss: 7.5807 - accuracy: 0.5056
11808/25000 [=============>................] - ETA: 43s - loss: 7.5744 - accuracy: 0.5060
11840/25000 [=============>................] - ETA: 43s - loss: 7.5734 - accuracy: 0.5061
11872/25000 [=============>................] - ETA: 43s - loss: 7.5749 - accuracy: 0.5060
11904/25000 [=============>................] - ETA: 43s - loss: 7.5777 - accuracy: 0.5058
11936/25000 [=============>................] - ETA: 43s - loss: 7.5831 - accuracy: 0.5054
11968/25000 [=============>................] - ETA: 43s - loss: 7.5821 - accuracy: 0.5055
12000/25000 [=============>................] - ETA: 43s - loss: 7.5797 - accuracy: 0.5057
12032/25000 [=============>................] - ETA: 42s - loss: 7.5838 - accuracy: 0.5054
12064/25000 [=============>................] - ETA: 42s - loss: 7.5802 - accuracy: 0.5056
12096/25000 [=============>................] - ETA: 42s - loss: 7.5779 - accuracy: 0.5058
12128/25000 [=============>................] - ETA: 42s - loss: 7.5832 - accuracy: 0.5054
12160/25000 [=============>................] - ETA: 42s - loss: 7.5884 - accuracy: 0.5051
12192/25000 [=============>................] - ETA: 42s - loss: 7.5886 - accuracy: 0.5051
12224/25000 [=============>................] - ETA: 42s - loss: 7.5863 - accuracy: 0.5052
12256/25000 [=============>................] - ETA: 42s - loss: 7.5866 - accuracy: 0.5052
12288/25000 [=============>................] - ETA: 42s - loss: 7.5855 - accuracy: 0.5053
12320/25000 [=============>................] - ETA: 41s - loss: 7.5895 - accuracy: 0.5050
12352/25000 [=============>................] - ETA: 41s - loss: 7.5897 - accuracy: 0.5050
12384/25000 [=============>................] - ETA: 41s - loss: 7.5899 - accuracy: 0.5050
12416/25000 [=============>................] - ETA: 41s - loss: 7.5851 - accuracy: 0.5053
12448/25000 [=============>................] - ETA: 41s - loss: 7.5866 - accuracy: 0.5052
12480/25000 [=============>................] - ETA: 41s - loss: 7.5880 - accuracy: 0.5051
12512/25000 [==============>...............] - ETA: 41s - loss: 7.5882 - accuracy: 0.5051
12544/25000 [==============>...............] - ETA: 41s - loss: 7.5859 - accuracy: 0.5053
12576/25000 [==============>...............] - ETA: 41s - loss: 7.5801 - accuracy: 0.5056
12608/25000 [==============>...............] - ETA: 40s - loss: 7.5864 - accuracy: 0.5052
12640/25000 [==============>...............] - ETA: 40s - loss: 7.5902 - accuracy: 0.5050
12672/25000 [==============>...............] - ETA: 40s - loss: 7.5892 - accuracy: 0.5051
12704/25000 [==============>...............] - ETA: 40s - loss: 7.5858 - accuracy: 0.5053
12736/25000 [==============>...............] - ETA: 40s - loss: 7.5860 - accuracy: 0.5053
12768/25000 [==============>...............] - ETA: 40s - loss: 7.5766 - accuracy: 0.5059
12800/25000 [==============>...............] - ETA: 40s - loss: 7.5756 - accuracy: 0.5059
12832/25000 [==============>...............] - ETA: 40s - loss: 7.5758 - accuracy: 0.5059
12864/25000 [==============>...............] - ETA: 40s - loss: 7.5725 - accuracy: 0.5061
12896/25000 [==============>...............] - ETA: 39s - loss: 7.5739 - accuracy: 0.5060
12928/25000 [==============>...............] - ETA: 39s - loss: 7.5753 - accuracy: 0.5060
12960/25000 [==============>...............] - ETA: 39s - loss: 7.5767 - accuracy: 0.5059
12992/25000 [==============>...............] - ETA: 39s - loss: 7.5781 - accuracy: 0.5058
13024/25000 [==============>...............] - ETA: 39s - loss: 7.5760 - accuracy: 0.5059
13056/25000 [==============>...............] - ETA: 39s - loss: 7.5750 - accuracy: 0.5060
13088/25000 [==============>...............] - ETA: 39s - loss: 7.5764 - accuracy: 0.5059
13120/25000 [==============>...............] - ETA: 39s - loss: 7.5743 - accuracy: 0.5060
13152/25000 [==============>...............] - ETA: 39s - loss: 7.5780 - accuracy: 0.5058
13184/25000 [==============>...............] - ETA: 38s - loss: 7.5782 - accuracy: 0.5058
13216/25000 [==============>...............] - ETA: 38s - loss: 7.5750 - accuracy: 0.5060
13248/25000 [==============>...............] - ETA: 38s - loss: 7.5798 - accuracy: 0.5057
13280/25000 [==============>...............] - ETA: 38s - loss: 7.5800 - accuracy: 0.5056
13312/25000 [==============>...............] - ETA: 38s - loss: 7.5802 - accuracy: 0.5056
13344/25000 [===============>..............] - ETA: 38s - loss: 7.5827 - accuracy: 0.5055
13376/25000 [===============>..............] - ETA: 38s - loss: 7.5795 - accuracy: 0.5057
13408/25000 [===============>..............] - ETA: 38s - loss: 7.5774 - accuracy: 0.5058
13440/25000 [===============>..............] - ETA: 38s - loss: 7.5754 - accuracy: 0.5060
13472/25000 [===============>..............] - ETA: 38s - loss: 7.5744 - accuracy: 0.5060
13504/25000 [===============>..............] - ETA: 37s - loss: 7.5815 - accuracy: 0.5056
13536/25000 [===============>..............] - ETA: 37s - loss: 7.5783 - accuracy: 0.5058
13568/25000 [===============>..............] - ETA: 37s - loss: 7.5728 - accuracy: 0.5061
13600/25000 [===============>..............] - ETA: 37s - loss: 7.5753 - accuracy: 0.5060
13632/25000 [===============>..............] - ETA: 37s - loss: 7.5755 - accuracy: 0.5059
13664/25000 [===============>..............] - ETA: 37s - loss: 7.5768 - accuracy: 0.5059
13696/25000 [===============>..............] - ETA: 37s - loss: 7.5782 - accuracy: 0.5058
13728/25000 [===============>..............] - ETA: 37s - loss: 7.5817 - accuracy: 0.5055
13760/25000 [===============>..............] - ETA: 37s - loss: 7.5752 - accuracy: 0.5060
13792/25000 [===============>..............] - ETA: 36s - loss: 7.5710 - accuracy: 0.5062
13824/25000 [===============>..............] - ETA: 36s - loss: 7.5757 - accuracy: 0.5059
13856/25000 [===============>..............] - ETA: 36s - loss: 7.5759 - accuracy: 0.5059
13888/25000 [===============>..............] - ETA: 36s - loss: 7.5750 - accuracy: 0.5060
13920/25000 [===============>..............] - ETA: 36s - loss: 7.5719 - accuracy: 0.5062
13952/25000 [===============>..............] - ETA: 36s - loss: 7.5776 - accuracy: 0.5058
13984/25000 [===============>..............] - ETA: 36s - loss: 7.5745 - accuracy: 0.5060
14016/25000 [===============>..............] - ETA: 36s - loss: 7.5747 - accuracy: 0.5060
14048/25000 [===============>..............] - ETA: 36s - loss: 7.5749 - accuracy: 0.5060
14080/25000 [===============>..............] - ETA: 35s - loss: 7.5675 - accuracy: 0.5065
14112/25000 [===============>..............] - ETA: 35s - loss: 7.5688 - accuracy: 0.5064
14144/25000 [===============>..............] - ETA: 35s - loss: 7.5734 - accuracy: 0.5061
14176/25000 [================>.............] - ETA: 35s - loss: 7.5747 - accuracy: 0.5060
14208/25000 [================>.............] - ETA: 35s - loss: 7.5738 - accuracy: 0.5061
14240/25000 [================>.............] - ETA: 35s - loss: 7.5794 - accuracy: 0.5057
14272/25000 [================>.............] - ETA: 35s - loss: 7.5828 - accuracy: 0.5055
14304/25000 [================>.............] - ETA: 35s - loss: 7.5830 - accuracy: 0.5055
14336/25000 [================>.............] - ETA: 35s - loss: 7.5843 - accuracy: 0.5054
14368/25000 [================>.............] - ETA: 34s - loss: 7.5866 - accuracy: 0.5052
14400/25000 [================>.............] - ETA: 34s - loss: 7.5900 - accuracy: 0.5050
14432/25000 [================>.............] - ETA: 34s - loss: 7.5859 - accuracy: 0.5053
14464/25000 [================>.............] - ETA: 34s - loss: 7.5892 - accuracy: 0.5050
14496/25000 [================>.............] - ETA: 34s - loss: 7.5894 - accuracy: 0.5050
14528/25000 [================>.............] - ETA: 34s - loss: 7.5875 - accuracy: 0.5052
14560/25000 [================>.............] - ETA: 34s - loss: 7.5824 - accuracy: 0.5055
14592/25000 [================>.............] - ETA: 34s - loss: 7.5763 - accuracy: 0.5059
14624/25000 [================>.............] - ETA: 34s - loss: 7.5775 - accuracy: 0.5058
14656/25000 [================>.............] - ETA: 33s - loss: 7.5798 - accuracy: 0.5057
14688/25000 [================>.............] - ETA: 33s - loss: 7.5831 - accuracy: 0.5054
14720/25000 [================>.............] - ETA: 33s - loss: 7.5812 - accuracy: 0.5056
14752/25000 [================>.............] - ETA: 33s - loss: 7.5824 - accuracy: 0.5055
14784/25000 [================>.............] - ETA: 33s - loss: 7.5847 - accuracy: 0.5053
14816/25000 [================>.............] - ETA: 33s - loss: 7.5869 - accuracy: 0.5052
14848/25000 [================>.............] - ETA: 33s - loss: 7.5871 - accuracy: 0.5052
14880/25000 [================>.............] - ETA: 33s - loss: 7.5862 - accuracy: 0.5052
14912/25000 [================>.............] - ETA: 33s - loss: 7.5864 - accuracy: 0.5052
14944/25000 [================>.............] - ETA: 33s - loss: 7.5815 - accuracy: 0.5056
14976/25000 [================>.............] - ETA: 32s - loss: 7.5806 - accuracy: 0.5056
15008/25000 [=================>............] - ETA: 32s - loss: 7.5818 - accuracy: 0.5055
15040/25000 [=================>............] - ETA: 32s - loss: 7.5881 - accuracy: 0.5051
15072/25000 [=================>............] - ETA: 32s - loss: 7.5863 - accuracy: 0.5052
15104/25000 [=================>............] - ETA: 32s - loss: 7.5864 - accuracy: 0.5052
15136/25000 [=================>............] - ETA: 32s - loss: 7.5866 - accuracy: 0.5052
15168/25000 [=================>............] - ETA: 32s - loss: 7.5908 - accuracy: 0.5049
15200/25000 [=================>............] - ETA: 32s - loss: 7.5960 - accuracy: 0.5046
15232/25000 [=================>............] - ETA: 32s - loss: 7.5972 - accuracy: 0.5045
15264/25000 [=================>............] - ETA: 31s - loss: 7.5943 - accuracy: 0.5047
15296/25000 [=================>............] - ETA: 31s - loss: 7.5924 - accuracy: 0.5048
15328/25000 [=================>............] - ETA: 31s - loss: 7.5946 - accuracy: 0.5047
15360/25000 [=================>............] - ETA: 31s - loss: 7.5947 - accuracy: 0.5047
15392/25000 [=================>............] - ETA: 31s - loss: 7.5949 - accuracy: 0.5047
15424/25000 [=================>............] - ETA: 31s - loss: 7.5921 - accuracy: 0.5049
15456/25000 [=================>............] - ETA: 31s - loss: 7.5892 - accuracy: 0.5050
15488/25000 [=================>............] - ETA: 31s - loss: 7.5904 - accuracy: 0.5050
15520/25000 [=================>............] - ETA: 31s - loss: 7.5935 - accuracy: 0.5048
15552/25000 [=================>............] - ETA: 30s - loss: 7.5897 - accuracy: 0.5050
15584/25000 [=================>............] - ETA: 30s - loss: 7.5850 - accuracy: 0.5053
15616/25000 [=================>............] - ETA: 30s - loss: 7.5822 - accuracy: 0.5055
15648/25000 [=================>............] - ETA: 30s - loss: 7.5823 - accuracy: 0.5055
15680/25000 [=================>............] - ETA: 30s - loss: 7.5796 - accuracy: 0.5057
15712/25000 [=================>............] - ETA: 30s - loss: 7.5788 - accuracy: 0.5057
15744/25000 [=================>............] - ETA: 30s - loss: 7.5799 - accuracy: 0.5057
15776/25000 [=================>............] - ETA: 30s - loss: 7.5762 - accuracy: 0.5059
15808/25000 [=================>............] - ETA: 30s - loss: 7.5725 - accuracy: 0.5061
15840/25000 [==================>...........] - ETA: 30s - loss: 7.5718 - accuracy: 0.5062
15872/25000 [==================>...........] - ETA: 29s - loss: 7.5729 - accuracy: 0.5061
15904/25000 [==================>...........] - ETA: 29s - loss: 7.5760 - accuracy: 0.5059
15936/25000 [==================>...........] - ETA: 29s - loss: 7.5743 - accuracy: 0.5060
15968/25000 [==================>...........] - ETA: 29s - loss: 7.5735 - accuracy: 0.5061
16000/25000 [==================>...........] - ETA: 29s - loss: 7.5737 - accuracy: 0.5061
16032/25000 [==================>...........] - ETA: 29s - loss: 7.5719 - accuracy: 0.5062
16064/25000 [==================>...........] - ETA: 29s - loss: 7.5750 - accuracy: 0.5060
16096/25000 [==================>...........] - ETA: 29s - loss: 7.5771 - accuracy: 0.5058
16128/25000 [==================>...........] - ETA: 29s - loss: 7.5782 - accuracy: 0.5058
16160/25000 [==================>...........] - ETA: 28s - loss: 7.5774 - accuracy: 0.5058
16192/25000 [==================>...........] - ETA: 28s - loss: 7.5823 - accuracy: 0.5055
16224/25000 [==================>...........] - ETA: 28s - loss: 7.5825 - accuracy: 0.5055
16256/25000 [==================>...........] - ETA: 28s - loss: 7.5855 - accuracy: 0.5053
16288/25000 [==================>...........] - ETA: 28s - loss: 7.5866 - accuracy: 0.5052
16320/25000 [==================>...........] - ETA: 28s - loss: 7.5858 - accuracy: 0.5053
16352/25000 [==================>...........] - ETA: 28s - loss: 7.5841 - accuracy: 0.5054
16384/25000 [==================>...........] - ETA: 28s - loss: 7.5852 - accuracy: 0.5053
16416/25000 [==================>...........] - ETA: 28s - loss: 7.5863 - accuracy: 0.5052
16448/25000 [==================>...........] - ETA: 27s - loss: 7.5864 - accuracy: 0.5052
16480/25000 [==================>...........] - ETA: 27s - loss: 7.5866 - accuracy: 0.5052
16512/25000 [==================>...........] - ETA: 27s - loss: 7.5895 - accuracy: 0.5050
16544/25000 [==================>...........] - ETA: 27s - loss: 7.5897 - accuracy: 0.5050
16576/25000 [==================>...........] - ETA: 27s - loss: 7.5908 - accuracy: 0.5049
16608/25000 [==================>...........] - ETA: 27s - loss: 7.5881 - accuracy: 0.5051
16640/25000 [==================>...........] - ETA: 27s - loss: 7.5892 - accuracy: 0.5050
16672/25000 [===================>..........] - ETA: 27s - loss: 7.5848 - accuracy: 0.5053
16704/25000 [===================>..........] - ETA: 27s - loss: 7.5868 - accuracy: 0.5052
16736/25000 [===================>..........] - ETA: 27s - loss: 7.5842 - accuracy: 0.5054
16768/25000 [===================>..........] - ETA: 26s - loss: 7.5807 - accuracy: 0.5056
16800/25000 [===================>..........] - ETA: 26s - loss: 7.5845 - accuracy: 0.5054
16832/25000 [===================>..........] - ETA: 26s - loss: 7.5901 - accuracy: 0.5050
16864/25000 [===================>..........] - ETA: 26s - loss: 7.5939 - accuracy: 0.5047
16896/25000 [===================>..........] - ETA: 26s - loss: 7.5949 - accuracy: 0.5047
16928/25000 [===================>..........] - ETA: 26s - loss: 7.5951 - accuracy: 0.5047
16960/25000 [===================>..........] - ETA: 26s - loss: 7.5934 - accuracy: 0.5048
16992/25000 [===================>..........] - ETA: 26s - loss: 7.5980 - accuracy: 0.5045
17024/25000 [===================>..........] - ETA: 26s - loss: 7.5955 - accuracy: 0.5046
17056/25000 [===================>..........] - ETA: 25s - loss: 7.5920 - accuracy: 0.5049
17088/25000 [===================>..........] - ETA: 25s - loss: 7.5948 - accuracy: 0.5047
17120/25000 [===================>..........] - ETA: 25s - loss: 7.5905 - accuracy: 0.5050
17152/25000 [===================>..........] - ETA: 25s - loss: 7.5969 - accuracy: 0.5045
17184/25000 [===================>..........] - ETA: 25s - loss: 7.5997 - accuracy: 0.5044
17216/25000 [===================>..........] - ETA: 25s - loss: 7.5945 - accuracy: 0.5047
17248/25000 [===================>..........] - ETA: 25s - loss: 7.5973 - accuracy: 0.5045
17280/25000 [===================>..........] - ETA: 25s - loss: 7.6010 - accuracy: 0.5043
17312/25000 [===================>..........] - ETA: 25s - loss: 7.6037 - accuracy: 0.5041
17344/25000 [===================>..........] - ETA: 25s - loss: 7.6021 - accuracy: 0.5042
17376/25000 [===================>..........] - ETA: 24s - loss: 7.6048 - accuracy: 0.5040
17408/25000 [===================>..........] - ETA: 24s - loss: 7.6102 - accuracy: 0.5037
17440/25000 [===================>..........] - ETA: 24s - loss: 7.6112 - accuracy: 0.5036
17472/25000 [===================>..........] - ETA: 24s - loss: 7.6148 - accuracy: 0.5034
17504/25000 [====================>.........] - ETA: 24s - loss: 7.6167 - accuracy: 0.5033
17536/25000 [====================>.........] - ETA: 24s - loss: 7.6159 - accuracy: 0.5033
17568/25000 [====================>.........] - ETA: 24s - loss: 7.6151 - accuracy: 0.5034
17600/25000 [====================>.........] - ETA: 24s - loss: 7.6170 - accuracy: 0.5032
17632/25000 [====================>.........] - ETA: 24s - loss: 7.6153 - accuracy: 0.5033
17664/25000 [====================>.........] - ETA: 23s - loss: 7.6154 - accuracy: 0.5033
17696/25000 [====================>.........] - ETA: 23s - loss: 7.6138 - accuracy: 0.5034
17728/25000 [====================>.........] - ETA: 23s - loss: 7.6147 - accuracy: 0.5034
17760/25000 [====================>.........] - ETA: 23s - loss: 7.6140 - accuracy: 0.5034
17792/25000 [====================>.........] - ETA: 23s - loss: 7.6140 - accuracy: 0.5034
17824/25000 [====================>.........] - ETA: 23s - loss: 7.6150 - accuracy: 0.5034
17856/25000 [====================>.........] - ETA: 23s - loss: 7.6142 - accuracy: 0.5034
17888/25000 [====================>.........] - ETA: 23s - loss: 7.6195 - accuracy: 0.5031
17920/25000 [====================>.........] - ETA: 23s - loss: 7.6178 - accuracy: 0.5032
17952/25000 [====================>.........] - ETA: 23s - loss: 7.6145 - accuracy: 0.5034
17984/25000 [====================>.........] - ETA: 22s - loss: 7.6163 - accuracy: 0.5033
18016/25000 [====================>.........] - ETA: 22s - loss: 7.6190 - accuracy: 0.5031
18048/25000 [====================>.........] - ETA: 22s - loss: 7.6165 - accuracy: 0.5033
18080/25000 [====================>.........] - ETA: 22s - loss: 7.6149 - accuracy: 0.5034
18112/25000 [====================>.........] - ETA: 22s - loss: 7.6158 - accuracy: 0.5033
18144/25000 [====================>.........] - ETA: 22s - loss: 7.6193 - accuracy: 0.5031
18176/25000 [====================>.........] - ETA: 22s - loss: 7.6168 - accuracy: 0.5032
18208/25000 [====================>.........] - ETA: 22s - loss: 7.6144 - accuracy: 0.5034
18240/25000 [====================>.........] - ETA: 22s - loss: 7.6111 - accuracy: 0.5036
18272/25000 [====================>.........] - ETA: 21s - loss: 7.6163 - accuracy: 0.5033
18304/25000 [====================>.........] - ETA: 21s - loss: 7.6155 - accuracy: 0.5033
18336/25000 [=====================>........] - ETA: 21s - loss: 7.6156 - accuracy: 0.5033
18368/25000 [=====================>........] - ETA: 21s - loss: 7.6157 - accuracy: 0.5033
18400/25000 [=====================>........] - ETA: 21s - loss: 7.6158 - accuracy: 0.5033
18432/25000 [=====================>........] - ETA: 21s - loss: 7.6192 - accuracy: 0.5031
18464/25000 [=====================>........] - ETA: 21s - loss: 7.6193 - accuracy: 0.5031
18496/25000 [=====================>........] - ETA: 21s - loss: 7.6210 - accuracy: 0.5030
18528/25000 [=====================>........] - ETA: 21s - loss: 7.6203 - accuracy: 0.5030
18560/25000 [=====================>........] - ETA: 21s - loss: 7.6245 - accuracy: 0.5027
18592/25000 [=====================>........] - ETA: 20s - loss: 7.6262 - accuracy: 0.5026
18624/25000 [=====================>........] - ETA: 20s - loss: 7.6255 - accuracy: 0.5027
18656/25000 [=====================>........] - ETA: 20s - loss: 7.6222 - accuracy: 0.5029
18688/25000 [=====================>........] - ETA: 20s - loss: 7.6223 - accuracy: 0.5029
18720/25000 [=====================>........] - ETA: 20s - loss: 7.6208 - accuracy: 0.5030
18752/25000 [=====================>........] - ETA: 20s - loss: 7.6192 - accuracy: 0.5031
18784/25000 [=====================>........] - ETA: 20s - loss: 7.6234 - accuracy: 0.5028
18816/25000 [=====================>........] - ETA: 20s - loss: 7.6194 - accuracy: 0.5031
18848/25000 [=====================>........] - ETA: 20s - loss: 7.6211 - accuracy: 0.5030
18880/25000 [=====================>........] - ETA: 19s - loss: 7.6220 - accuracy: 0.5029
18912/25000 [=====================>........] - ETA: 19s - loss: 7.6180 - accuracy: 0.5032
18944/25000 [=====================>........] - ETA: 19s - loss: 7.6221 - accuracy: 0.5029
18976/25000 [=====================>........] - ETA: 19s - loss: 7.6230 - accuracy: 0.5028
19008/25000 [=====================>........] - ETA: 19s - loss: 7.6263 - accuracy: 0.5026
19040/25000 [=====================>........] - ETA: 19s - loss: 7.6272 - accuracy: 0.5026
19072/25000 [=====================>........] - ETA: 19s - loss: 7.6280 - accuracy: 0.5025
19104/25000 [=====================>........] - ETA: 19s - loss: 7.6281 - accuracy: 0.5025
19136/25000 [=====================>........] - ETA: 19s - loss: 7.6322 - accuracy: 0.5022
19168/25000 [======================>.......] - ETA: 19s - loss: 7.6314 - accuracy: 0.5023
19200/25000 [======================>.......] - ETA: 18s - loss: 7.6323 - accuracy: 0.5022
19232/25000 [======================>.......] - ETA: 18s - loss: 7.6355 - accuracy: 0.5020
19264/25000 [======================>.......] - ETA: 18s - loss: 7.6380 - accuracy: 0.5019
19296/25000 [======================>.......] - ETA: 18s - loss: 7.6388 - accuracy: 0.5018
19328/25000 [======================>.......] - ETA: 18s - loss: 7.6420 - accuracy: 0.5016
19360/25000 [======================>.......] - ETA: 18s - loss: 7.6421 - accuracy: 0.5016
19392/25000 [======================>.......] - ETA: 18s - loss: 7.6405 - accuracy: 0.5017
19424/25000 [======================>.......] - ETA: 18s - loss: 7.6374 - accuracy: 0.5019
19456/25000 [======================>.......] - ETA: 18s - loss: 7.6375 - accuracy: 0.5019
19488/25000 [======================>.......] - ETA: 17s - loss: 7.6367 - accuracy: 0.5019
19520/25000 [======================>.......] - ETA: 17s - loss: 7.6368 - accuracy: 0.5019
19552/25000 [======================>.......] - ETA: 17s - loss: 7.6360 - accuracy: 0.5020
19584/25000 [======================>.......] - ETA: 17s - loss: 7.6345 - accuracy: 0.5021
19616/25000 [======================>.......] - ETA: 17s - loss: 7.6361 - accuracy: 0.5020
19648/25000 [======================>.......] - ETA: 17s - loss: 7.6370 - accuracy: 0.5019
19680/25000 [======================>.......] - ETA: 17s - loss: 7.6347 - accuracy: 0.5021
19712/25000 [======================>.......] - ETA: 17s - loss: 7.6339 - accuracy: 0.5021
19744/25000 [======================>.......] - ETA: 17s - loss: 7.6371 - accuracy: 0.5019
19776/25000 [======================>.......] - ETA: 17s - loss: 7.6395 - accuracy: 0.5018
19808/25000 [======================>.......] - ETA: 16s - loss: 7.6403 - accuracy: 0.5017
19840/25000 [======================>.......] - ETA: 16s - loss: 7.6396 - accuracy: 0.5018
19872/25000 [======================>.......] - ETA: 16s - loss: 7.6419 - accuracy: 0.5016
19904/25000 [======================>.......] - ETA: 16s - loss: 7.6404 - accuracy: 0.5017
19936/25000 [======================>.......] - ETA: 16s - loss: 7.6382 - accuracy: 0.5019
19968/25000 [======================>.......] - ETA: 16s - loss: 7.6359 - accuracy: 0.5020
20000/25000 [=======================>......] - ETA: 16s - loss: 7.6352 - accuracy: 0.5020
20032/25000 [=======================>......] - ETA: 16s - loss: 7.6352 - accuracy: 0.5020
20064/25000 [=======================>......] - ETA: 16s - loss: 7.6361 - accuracy: 0.5020
20096/25000 [=======================>......] - ETA: 15s - loss: 7.6346 - accuracy: 0.5021
20128/25000 [=======================>......] - ETA: 15s - loss: 7.6377 - accuracy: 0.5019
20160/25000 [=======================>......] - ETA: 15s - loss: 7.6415 - accuracy: 0.5016
20192/25000 [=======================>......] - ETA: 15s - loss: 7.6431 - accuracy: 0.5015
20224/25000 [=======================>......] - ETA: 15s - loss: 7.6431 - accuracy: 0.5015
20256/25000 [=======================>......] - ETA: 15s - loss: 7.6401 - accuracy: 0.5017
20288/25000 [=======================>......] - ETA: 15s - loss: 7.6417 - accuracy: 0.5016
20320/25000 [=======================>......] - ETA: 15s - loss: 7.6402 - accuracy: 0.5017
20352/25000 [=======================>......] - ETA: 15s - loss: 7.6410 - accuracy: 0.5017
20384/25000 [=======================>......] - ETA: 15s - loss: 7.6418 - accuracy: 0.5016
20416/25000 [=======================>......] - ETA: 14s - loss: 7.6411 - accuracy: 0.5017
20448/25000 [=======================>......] - ETA: 14s - loss: 7.6419 - accuracy: 0.5016
20480/25000 [=======================>......] - ETA: 14s - loss: 7.6427 - accuracy: 0.5016
20512/25000 [=======================>......] - ETA: 14s - loss: 7.6412 - accuracy: 0.5017
20544/25000 [=======================>......] - ETA: 14s - loss: 7.6405 - accuracy: 0.5017
20576/25000 [=======================>......] - ETA: 14s - loss: 7.6390 - accuracy: 0.5018
20608/25000 [=======================>......] - ETA: 14s - loss: 7.6428 - accuracy: 0.5016
20640/25000 [=======================>......] - ETA: 14s - loss: 7.6473 - accuracy: 0.5013
20672/25000 [=======================>......] - ETA: 14s - loss: 7.6429 - accuracy: 0.5015
20704/25000 [=======================>......] - ETA: 13s - loss: 7.6429 - accuracy: 0.5015
20736/25000 [=======================>......] - ETA: 13s - loss: 7.6444 - accuracy: 0.5014
20768/25000 [=======================>......] - ETA: 13s - loss: 7.6430 - accuracy: 0.5015
20800/25000 [=======================>......] - ETA: 13s - loss: 7.6393 - accuracy: 0.5018
20832/25000 [=======================>......] - ETA: 13s - loss: 7.6364 - accuracy: 0.5020
20864/25000 [========================>.....] - ETA: 13s - loss: 7.6380 - accuracy: 0.5019
20896/25000 [========================>.....] - ETA: 13s - loss: 7.6373 - accuracy: 0.5019
20928/25000 [========================>.....] - ETA: 13s - loss: 7.6366 - accuracy: 0.5020
20960/25000 [========================>.....] - ETA: 13s - loss: 7.6396 - accuracy: 0.5018
20992/25000 [========================>.....] - ETA: 13s - loss: 7.6411 - accuracy: 0.5017
21024/25000 [========================>.....] - ETA: 12s - loss: 7.6404 - accuracy: 0.5017
21056/25000 [========================>.....] - ETA: 12s - loss: 7.6389 - accuracy: 0.5018
21088/25000 [========================>.....] - ETA: 12s - loss: 7.6383 - accuracy: 0.5018
21120/25000 [========================>.....] - ETA: 12s - loss: 7.6376 - accuracy: 0.5019
21152/25000 [========================>.....] - ETA: 12s - loss: 7.6376 - accuracy: 0.5019
21184/25000 [========================>.....] - ETA: 12s - loss: 7.6391 - accuracy: 0.5018
21216/25000 [========================>.....] - ETA: 12s - loss: 7.6392 - accuracy: 0.5018
21248/25000 [========================>.....] - ETA: 12s - loss: 7.6399 - accuracy: 0.5017
21280/25000 [========================>.....] - ETA: 12s - loss: 7.6400 - accuracy: 0.5017
21312/25000 [========================>.....] - ETA: 12s - loss: 7.6393 - accuracy: 0.5018
21344/25000 [========================>.....] - ETA: 11s - loss: 7.6436 - accuracy: 0.5015
21376/25000 [========================>.....] - ETA: 11s - loss: 7.6408 - accuracy: 0.5017
21408/25000 [========================>.....] - ETA: 11s - loss: 7.6408 - accuracy: 0.5017
21440/25000 [========================>.....] - ETA: 11s - loss: 7.6430 - accuracy: 0.5015
21472/25000 [========================>.....] - ETA: 11s - loss: 7.6423 - accuracy: 0.5016
21504/25000 [========================>.....] - ETA: 11s - loss: 7.6388 - accuracy: 0.5018
21536/25000 [========================>.....] - ETA: 11s - loss: 7.6381 - accuracy: 0.5019
21568/25000 [========================>.....] - ETA: 11s - loss: 7.6382 - accuracy: 0.5019
21600/25000 [========================>.....] - ETA: 11s - loss: 7.6404 - accuracy: 0.5017
21632/25000 [========================>.....] - ETA: 10s - loss: 7.6390 - accuracy: 0.5018
21664/25000 [========================>.....] - ETA: 10s - loss: 7.6390 - accuracy: 0.5018
21696/25000 [=========================>....] - ETA: 10s - loss: 7.6376 - accuracy: 0.5019
21728/25000 [=========================>....] - ETA: 10s - loss: 7.6370 - accuracy: 0.5019
21760/25000 [=========================>....] - ETA: 10s - loss: 7.6356 - accuracy: 0.5020
21792/25000 [=========================>....] - ETA: 10s - loss: 7.6378 - accuracy: 0.5019
21824/25000 [=========================>....] - ETA: 10s - loss: 7.6399 - accuracy: 0.5017
21856/25000 [=========================>....] - ETA: 10s - loss: 7.6365 - accuracy: 0.5020
21888/25000 [=========================>....] - ETA: 10s - loss: 7.6386 - accuracy: 0.5018
21920/25000 [=========================>....] - ETA: 10s - loss: 7.6414 - accuracy: 0.5016
21952/25000 [=========================>....] - ETA: 9s - loss: 7.6401 - accuracy: 0.5017 
21984/25000 [=========================>....] - ETA: 9s - loss: 7.6429 - accuracy: 0.5015
22016/25000 [=========================>....] - ETA: 9s - loss: 7.6443 - accuracy: 0.5015
22048/25000 [=========================>....] - ETA: 9s - loss: 7.6437 - accuracy: 0.5015
22080/25000 [=========================>....] - ETA: 9s - loss: 7.6444 - accuracy: 0.5014
22112/25000 [=========================>....] - ETA: 9s - loss: 7.6444 - accuracy: 0.5014
22144/25000 [=========================>....] - ETA: 9s - loss: 7.6438 - accuracy: 0.5015
22176/25000 [=========================>....] - ETA: 9s - loss: 7.6417 - accuracy: 0.5016
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6459 - accuracy: 0.5014
22240/25000 [=========================>....] - ETA: 8s - loss: 7.6459 - accuracy: 0.5013
22272/25000 [=========================>....] - ETA: 8s - loss: 7.6453 - accuracy: 0.5014
22304/25000 [=========================>....] - ETA: 8s - loss: 7.6467 - accuracy: 0.5013
22336/25000 [=========================>....] - ETA: 8s - loss: 7.6460 - accuracy: 0.5013
22368/25000 [=========================>....] - ETA: 8s - loss: 7.6454 - accuracy: 0.5014
22400/25000 [=========================>....] - ETA: 8s - loss: 7.6447 - accuracy: 0.5014
22432/25000 [=========================>....] - ETA: 8s - loss: 7.6468 - accuracy: 0.5013
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6475 - accuracy: 0.5012
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6455 - accuracy: 0.5014
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6469 - accuracy: 0.5013
22560/25000 [==========================>...] - ETA: 7s - loss: 7.6483 - accuracy: 0.5012
22592/25000 [==========================>...] - ETA: 7s - loss: 7.6483 - accuracy: 0.5012
22624/25000 [==========================>...] - ETA: 7s - loss: 7.6497 - accuracy: 0.5011
22656/25000 [==========================>...] - ETA: 7s - loss: 7.6497 - accuracy: 0.5011
22688/25000 [==========================>...] - ETA: 7s - loss: 7.6497 - accuracy: 0.5011
22720/25000 [==========================>...] - ETA: 7s - loss: 7.6497 - accuracy: 0.5011
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6491 - accuracy: 0.5011
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6471 - accuracy: 0.5013
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6451 - accuracy: 0.5014
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6451 - accuracy: 0.5014
22880/25000 [==========================>...] - ETA: 6s - loss: 7.6472 - accuracy: 0.5013
22912/25000 [==========================>...] - ETA: 6s - loss: 7.6439 - accuracy: 0.5015
22944/25000 [==========================>...] - ETA: 6s - loss: 7.6432 - accuracy: 0.5015
22976/25000 [==========================>...] - ETA: 6s - loss: 7.6419 - accuracy: 0.5016
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6420 - accuracy: 0.5016
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6420 - accuracy: 0.5016
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6434 - accuracy: 0.5015
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6460 - accuracy: 0.5013
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6481 - accuracy: 0.5012
23168/25000 [==========================>...] - ETA: 5s - loss: 7.6488 - accuracy: 0.5012
23200/25000 [==========================>...] - ETA: 5s - loss: 7.6494 - accuracy: 0.5011
23232/25000 [==========================>...] - ETA: 5s - loss: 7.6468 - accuracy: 0.5013
23264/25000 [==========================>...] - ETA: 5s - loss: 7.6455 - accuracy: 0.5014
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6469 - accuracy: 0.5013
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6502 - accuracy: 0.5011
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6509 - accuracy: 0.5010
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6522 - accuracy: 0.5009
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6470 - accuracy: 0.5013
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6477 - accuracy: 0.5012
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6503 - accuracy: 0.5011
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6477 - accuracy: 0.5012
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6471 - accuracy: 0.5013
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6497 - accuracy: 0.5011
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6497 - accuracy: 0.5011
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6517 - accuracy: 0.5010
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6517 - accuracy: 0.5010
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6537 - accuracy: 0.5008
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6543 - accuracy: 0.5008
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6544 - accuracy: 0.5008
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6557 - accuracy: 0.5007
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6576 - accuracy: 0.5006
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6602 - accuracy: 0.5004
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6608 - accuracy: 0.5004
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6596 - accuracy: 0.5005
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6641 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6641 - accuracy: 0.5002
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6641 - accuracy: 0.5002
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6634 - accuracy: 0.5002
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6634 - accuracy: 0.5002
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6647 - accuracy: 0.5001
24192/25000 [============================>.] - ETA: 2s - loss: 7.6628 - accuracy: 0.5002
24224/25000 [============================>.] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24256/25000 [============================>.] - ETA: 2s - loss: 7.6603 - accuracy: 0.5004
24288/25000 [============================>.] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
24320/25000 [============================>.] - ETA: 2s - loss: 7.6578 - accuracy: 0.5006
24352/25000 [============================>.] - ETA: 2s - loss: 7.6610 - accuracy: 0.5004
24384/25000 [============================>.] - ETA: 2s - loss: 7.6622 - accuracy: 0.5003
24416/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24448/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24480/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24512/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24544/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24576/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24608/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24640/25000 [============================>.] - ETA: 1s - loss: 7.6741 - accuracy: 0.4995
24672/25000 [============================>.] - ETA: 1s - loss: 7.6728 - accuracy: 0.4996
24704/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24736/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24768/25000 [============================>.] - ETA: 0s - loss: 7.6691 - accuracy: 0.4998
24800/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24832/25000 [============================>.] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24864/25000 [============================>.] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24896/25000 [============================>.] - ETA: 0s - loss: 7.6759 - accuracy: 0.4994
24928/25000 [============================>.] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24960/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 98s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
[1;32m     83[0m """
[1;32m     84[0m [0;34m[0m[0m
[0;32m---> 85[0;31m [0mcalendar[0m               [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/calendar.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     86[0m [0msales_train_val[0m        [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sales_train_val.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     87[0m [0msample_submission[0m      [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sample_submission.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m     83[0m """
[1;32m     84[0m [0;34m[0m[0m
[0;32m---> 85[0;31m [0mcalendar[0m               [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/calendar.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     86[0m [0msales_train_val[0m        [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sales_train_val.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     87[0m [0msample_submission[0m      [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34mf'{m5_input_path}/sample_submission.csv'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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
