
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
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:51<01:16, 25.53s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 13% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 13% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 13% CPU time recently (threshold: 10%)
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.4876349320595029, 'embedding_size_factor': 1.0651111656008794, 'layers.choice': 1, 'learning_rate': 0.005436578238228051, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 3.879182479137691e-07} and reward: 0.3738
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdf5i%eK\xbaX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\n\xb2\x01m\xc3sX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?vD\xaa[\xc3\xeefX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\x9a\x08b\x7f0Lsu.' and reward: 0.3738
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdf5i%eK\xbaX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\n\xb2\x01m\xc3sX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?vD\xaa[\xc3\xeefX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\x9a\x08b\x7f0Lsu.' and reward: 0.3738
 60%|██████    | 3/5 [02:32<01:36, 48.39s/it] 60%|██████    | 3/5 [02:32<01:41, 50.93s/it]
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.3398708256949031, 'embedding_size_factor': 0.6635569519886655, 'layers.choice': 0, 'learning_rate': 0.004541554563548868, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.012444361822064017} and reward: 0.3568
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd5\xc0q\x90NX\x12X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5;\xdb\xc9\xfadpX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?r\x9a*E,\xd1uX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\x89|m\xf8\\\xe3\x91u.' and reward: 0.3568
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd5\xc0q\x90NX\x12X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5;\xdb\xc9\xfadpX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?r\x9a*E,\xd1uX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\x89|m\xf8\\\xe3\x91u.' and reward: 0.3568
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 204.56919813156128
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -87.43s of remaining time.
Ensemble size: 66
Ensemble weights: 
[0.51515152 0.1969697  0.28787879]
	0.3926	 = Validation accuracy score
	1.05s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 208.52s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f6e5cf4eac8> 

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
 [-0.09715908 -0.16878271  0.14781962  0.13683651  0.00940609 -0.0242458 ]
 [-0.1115071  -0.03682543  0.07107999 -0.14002742 -0.00731012  0.1608748 ]
 [ 0.00319288  0.1098595  -0.07869649  0.42283431 -0.07274604 -0.10872924]
 [ 0.04099433 -0.0124779  -0.0626485  -0.10144817 -0.11898793 -0.09223799]
 [ 0.03681491  0.51416594  0.45633346  0.46477789  0.23054875  0.07825562]
 [-0.20037538  0.21270169 -0.06089731  0.30017552  0.09513601  0.015961  ]
 [-0.70924014  0.24431781  0.36438841  0.02999384 -0.47682789 -0.17512907]
 [ 0.30104887 -0.14269345  0.22111556  0.26486194 -0.19909729 -0.54634809]
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
{'loss': 0.5580978840589523, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 15:17:04.222318: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.4698680080473423, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 15:17:05.365786: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 1835008/17464789 [==>...........................] - ETA: 0s
 7241728/17464789 [===========>..................] - ETA: 0s
15818752/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 15:17:17.014543: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 15:17:17.019081: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-15 15:17:17.019225: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5611b1680400 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 15:17:17.019239: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:35 - loss: 8.1458 - accuracy: 0.4688
   64/25000 [..............................] - ETA: 2:49 - loss: 8.6249 - accuracy: 0.4375
   96/25000 [..............................] - ETA: 2:14 - loss: 7.5069 - accuracy: 0.5104
  128/25000 [..............................] - ETA: 1:56 - loss: 7.0677 - accuracy: 0.5391
  160/25000 [..............................] - ETA: 1:46 - loss: 6.9958 - accuracy: 0.5437
  192/25000 [..............................] - ETA: 1:39 - loss: 7.2673 - accuracy: 0.5260
  224/25000 [..............................] - ETA: 1:35 - loss: 7.1875 - accuracy: 0.5312
  256/25000 [..............................] - ETA: 1:31 - loss: 7.1276 - accuracy: 0.5352
  288/25000 [..............................] - ETA: 1:28 - loss: 7.4004 - accuracy: 0.5174
  320/25000 [..............................] - ETA: 1:25 - loss: 7.4270 - accuracy: 0.5156
  352/25000 [..............................] - ETA: 1:23 - loss: 7.4488 - accuracy: 0.5142
  384/25000 [..............................] - ETA: 1:21 - loss: 7.3871 - accuracy: 0.5182
  416/25000 [..............................] - ETA: 1:20 - loss: 7.3717 - accuracy: 0.5192
  448/25000 [..............................] - ETA: 1:19 - loss: 7.4613 - accuracy: 0.5134
  480/25000 [..............................] - ETA: 1:17 - loss: 7.4430 - accuracy: 0.5146
  512/25000 [..............................] - ETA: 1:17 - loss: 7.3971 - accuracy: 0.5176
  544/25000 [..............................] - ETA: 1:16 - loss: 7.4693 - accuracy: 0.5129
  576/25000 [..............................] - ETA: 1:15 - loss: 7.5335 - accuracy: 0.5087
  608/25000 [..............................] - ETA: 1:15 - loss: 7.5153 - accuracy: 0.5099
  640/25000 [..............................] - ETA: 1:14 - loss: 7.4510 - accuracy: 0.5141
  672/25000 [..............................] - ETA: 1:13 - loss: 7.4841 - accuracy: 0.5119
  704/25000 [..............................] - ETA: 1:13 - loss: 7.4270 - accuracy: 0.5156
  736/25000 [..............................] - ETA: 1:12 - loss: 7.5000 - accuracy: 0.5109
  768/25000 [..............................] - ETA: 1:11 - loss: 7.5069 - accuracy: 0.5104
  800/25000 [..............................] - ETA: 1:11 - loss: 7.4750 - accuracy: 0.5125
  832/25000 [..............................] - ETA: 1:11 - loss: 7.4455 - accuracy: 0.5144
  864/25000 [>.............................] - ETA: 1:10 - loss: 7.4892 - accuracy: 0.5116
  896/25000 [>.............................] - ETA: 1:10 - loss: 7.4784 - accuracy: 0.5123
  928/25000 [>.............................] - ETA: 1:09 - loss: 7.4518 - accuracy: 0.5140
  960/25000 [>.............................] - ETA: 1:09 - loss: 7.5229 - accuracy: 0.5094
  992/25000 [>.............................] - ETA: 1:08 - loss: 7.5739 - accuracy: 0.5060
 1024/25000 [>.............................] - ETA: 1:08 - loss: 7.5169 - accuracy: 0.5098
 1056/25000 [>.............................] - ETA: 1:08 - loss: 7.5214 - accuracy: 0.5095
 1088/25000 [>.............................] - ETA: 1:08 - loss: 7.5398 - accuracy: 0.5083
 1120/25000 [>.............................] - ETA: 1:08 - loss: 7.5571 - accuracy: 0.5071
 1152/25000 [>.............................] - ETA: 1:07 - loss: 7.5868 - accuracy: 0.5052
 1184/25000 [>.............................] - ETA: 1:07 - loss: 7.5889 - accuracy: 0.5051
 1216/25000 [>.............................] - ETA: 1:07 - loss: 7.5657 - accuracy: 0.5066
 1248/25000 [>.............................] - ETA: 1:06 - loss: 7.5929 - accuracy: 0.5048
 1280/25000 [>.............................] - ETA: 1:06 - loss: 7.5229 - accuracy: 0.5094
 1312/25000 [>.............................] - ETA: 1:06 - loss: 7.4563 - accuracy: 0.5137
 1344/25000 [>.............................] - ETA: 1:06 - loss: 7.4499 - accuracy: 0.5141
 1376/25000 [>.............................] - ETA: 1:06 - loss: 7.4438 - accuracy: 0.5145
 1408/25000 [>.............................] - ETA: 1:06 - loss: 7.3835 - accuracy: 0.5185
 1440/25000 [>.............................] - ETA: 1:06 - loss: 7.4111 - accuracy: 0.5167
 1472/25000 [>.............................] - ETA: 1:05 - loss: 7.3854 - accuracy: 0.5183
 1504/25000 [>.............................] - ETA: 1:05 - loss: 7.4117 - accuracy: 0.5166
 1536/25000 [>.............................] - ETA: 1:05 - loss: 7.4270 - accuracy: 0.5156
 1568/25000 [>.............................] - ETA: 1:05 - loss: 7.4221 - accuracy: 0.5159
 1600/25000 [>.............................] - ETA: 1:05 - loss: 7.4366 - accuracy: 0.5150
 1632/25000 [>.............................] - ETA: 1:05 - loss: 7.4223 - accuracy: 0.5159
 1664/25000 [>.............................] - ETA: 1:04 - loss: 7.4639 - accuracy: 0.5132
 1696/25000 [=>............................] - ETA: 1:05 - loss: 7.4948 - accuracy: 0.5112
 1728/25000 [=>............................] - ETA: 1:05 - loss: 7.5246 - accuracy: 0.5093
 1760/25000 [=>............................] - ETA: 1:05 - loss: 7.5534 - accuracy: 0.5074
 1792/25000 [=>............................] - ETA: 1:05 - loss: 7.5639 - accuracy: 0.5067
 1824/25000 [=>............................] - ETA: 1:04 - loss: 7.5826 - accuracy: 0.5055
 1856/25000 [=>............................] - ETA: 1:04 - loss: 7.5840 - accuracy: 0.5054
 1888/25000 [=>............................] - ETA: 1:04 - loss: 7.5610 - accuracy: 0.5069
 1920/25000 [=>............................] - ETA: 1:04 - loss: 7.5708 - accuracy: 0.5063
 1952/25000 [=>............................] - ETA: 1:03 - loss: 7.5409 - accuracy: 0.5082
 1984/25000 [=>............................] - ETA: 1:03 - loss: 7.5661 - accuracy: 0.5066
 2016/25000 [=>............................] - ETA: 1:03 - loss: 7.5601 - accuracy: 0.5069
 2048/25000 [=>............................] - ETA: 1:03 - loss: 7.5393 - accuracy: 0.5083
 2080/25000 [=>............................] - ETA: 1:03 - loss: 7.5487 - accuracy: 0.5077
 2112/25000 [=>............................] - ETA: 1:03 - loss: 7.5650 - accuracy: 0.5066
 2144/25000 [=>............................] - ETA: 1:03 - loss: 7.5880 - accuracy: 0.5051
 2176/25000 [=>............................] - ETA: 1:02 - loss: 7.6032 - accuracy: 0.5041
 2208/25000 [=>............................] - ETA: 1:02 - loss: 7.6180 - accuracy: 0.5032
 2240/25000 [=>............................] - ETA: 1:02 - loss: 7.5708 - accuracy: 0.5063
 2272/25000 [=>............................] - ETA: 1:02 - loss: 7.5654 - accuracy: 0.5066
 2304/25000 [=>............................] - ETA: 1:02 - loss: 7.5934 - accuracy: 0.5048
 2336/25000 [=>............................] - ETA: 1:02 - loss: 7.5813 - accuracy: 0.5056
 2368/25000 [=>............................] - ETA: 1:01 - loss: 7.5565 - accuracy: 0.5072
 2400/25000 [=>............................] - ETA: 1:01 - loss: 7.5772 - accuracy: 0.5058
 2432/25000 [=>............................] - ETA: 1:01 - loss: 7.5847 - accuracy: 0.5053
 2464/25000 [=>............................] - ETA: 1:01 - loss: 7.5919 - accuracy: 0.5049
 2496/25000 [=>............................] - ETA: 1:01 - loss: 7.6052 - accuracy: 0.5040
 2528/25000 [==>...........................] - ETA: 1:01 - loss: 7.6120 - accuracy: 0.5036
 2560/25000 [==>...........................] - ETA: 1:01 - loss: 7.6007 - accuracy: 0.5043
 2592/25000 [==>...........................] - ETA: 1:01 - loss: 7.5897 - accuracy: 0.5050
 2624/25000 [==>...........................] - ETA: 1:00 - loss: 7.5790 - accuracy: 0.5057
 2656/25000 [==>...........................] - ETA: 1:00 - loss: 7.5685 - accuracy: 0.5064
 2688/25000 [==>...........................] - ETA: 1:00 - loss: 7.5811 - accuracy: 0.5056
 2720/25000 [==>...........................] - ETA: 1:00 - loss: 7.5539 - accuracy: 0.5074
 2752/25000 [==>...........................] - ETA: 1:00 - loss: 7.5663 - accuracy: 0.5065
 2784/25000 [==>...........................] - ETA: 1:00 - loss: 7.5510 - accuracy: 0.5075
 2816/25000 [==>...........................] - ETA: 1:00 - loss: 7.5523 - accuracy: 0.5075
 2848/25000 [==>...........................] - ETA: 1:00 - loss: 7.5482 - accuracy: 0.5077
 2880/25000 [==>...........................] - ETA: 59s - loss: 7.5495 - accuracy: 0.5076 
 2912/25000 [==>...........................] - ETA: 59s - loss: 7.5666 - accuracy: 0.5065
 2944/25000 [==>...........................] - ETA: 59s - loss: 7.5781 - accuracy: 0.5058
 2976/25000 [==>...........................] - ETA: 59s - loss: 7.5996 - accuracy: 0.5044
 3008/25000 [==>...........................] - ETA: 59s - loss: 7.5749 - accuracy: 0.5060
 3040/25000 [==>...........................] - ETA: 59s - loss: 7.5758 - accuracy: 0.5059
 3072/25000 [==>...........................] - ETA: 59s - loss: 7.5668 - accuracy: 0.5065
 3104/25000 [==>...........................] - ETA: 58s - loss: 7.5876 - accuracy: 0.5052
 3136/25000 [==>...........................] - ETA: 58s - loss: 7.5982 - accuracy: 0.5045
 3168/25000 [==>...........................] - ETA: 58s - loss: 7.5940 - accuracy: 0.5047
 3200/25000 [==>...........................] - ETA: 58s - loss: 7.5947 - accuracy: 0.5047
 3232/25000 [==>...........................] - ETA: 58s - loss: 7.5765 - accuracy: 0.5059
 3264/25000 [==>...........................] - ETA: 58s - loss: 7.5821 - accuracy: 0.5055
 3296/25000 [==>...........................] - ETA: 58s - loss: 7.5829 - accuracy: 0.5055
 3328/25000 [==>...........................] - ETA: 58s - loss: 7.5837 - accuracy: 0.5054
 3360/25000 [===>..........................] - ETA: 58s - loss: 7.5571 - accuracy: 0.5071
 3392/25000 [===>..........................] - ETA: 57s - loss: 7.5446 - accuracy: 0.5080
 3424/25000 [===>..........................] - ETA: 57s - loss: 7.5368 - accuracy: 0.5085
 3456/25000 [===>..........................] - ETA: 57s - loss: 7.5069 - accuracy: 0.5104
 3488/25000 [===>..........................] - ETA: 57s - loss: 7.5216 - accuracy: 0.5095
 3520/25000 [===>..........................] - ETA: 57s - loss: 7.5185 - accuracy: 0.5097
 3552/25000 [===>..........................] - ETA: 57s - loss: 7.5242 - accuracy: 0.5093
 3584/25000 [===>..........................] - ETA: 57s - loss: 7.5340 - accuracy: 0.5086
 3616/25000 [===>..........................] - ETA: 57s - loss: 7.5224 - accuracy: 0.5094
 3648/25000 [===>..........................] - ETA: 56s - loss: 7.5195 - accuracy: 0.5096
 3680/25000 [===>..........................] - ETA: 56s - loss: 7.5458 - accuracy: 0.5079
 3712/25000 [===>..........................] - ETA: 56s - loss: 7.5468 - accuracy: 0.5078
 3744/25000 [===>..........................] - ETA: 56s - loss: 7.5438 - accuracy: 0.5080
 3776/25000 [===>..........................] - ETA: 56s - loss: 7.5407 - accuracy: 0.5082
 3808/25000 [===>..........................] - ETA: 56s - loss: 7.5539 - accuracy: 0.5074
 3840/25000 [===>..........................] - ETA: 56s - loss: 7.5588 - accuracy: 0.5070
 3872/25000 [===>..........................] - ETA: 56s - loss: 7.5518 - accuracy: 0.5075
 3904/25000 [===>..........................] - ETA: 56s - loss: 7.5645 - accuracy: 0.5067
 3936/25000 [===>..........................] - ETA: 56s - loss: 7.5498 - accuracy: 0.5076
 3968/25000 [===>..........................] - ETA: 56s - loss: 7.5430 - accuracy: 0.5081
 4000/25000 [===>..........................] - ETA: 55s - loss: 7.5516 - accuracy: 0.5075
 4032/25000 [===>..........................] - ETA: 55s - loss: 7.5563 - accuracy: 0.5072
 4064/25000 [===>..........................] - ETA: 55s - loss: 7.5534 - accuracy: 0.5074
 4096/25000 [===>..........................] - ETA: 55s - loss: 7.5618 - accuracy: 0.5068
 4128/25000 [===>..........................] - ETA: 55s - loss: 7.5478 - accuracy: 0.5078
 4160/25000 [===>..........................] - ETA: 55s - loss: 7.5745 - accuracy: 0.5060
 4192/25000 [====>.........................] - ETA: 55s - loss: 7.5825 - accuracy: 0.5055
 4224/25000 [====>.........................] - ETA: 55s - loss: 7.5759 - accuracy: 0.5059
 4256/25000 [====>.........................] - ETA: 55s - loss: 7.5838 - accuracy: 0.5054
 4288/25000 [====>.........................] - ETA: 54s - loss: 7.5951 - accuracy: 0.5047
 4320/25000 [====>.........................] - ETA: 54s - loss: 7.5885 - accuracy: 0.5051
 4352/25000 [====>.........................] - ETA: 54s - loss: 7.5856 - accuracy: 0.5053
 4384/25000 [====>.........................] - ETA: 54s - loss: 7.5792 - accuracy: 0.5057
 4416/25000 [====>.........................] - ETA: 54s - loss: 7.5763 - accuracy: 0.5059
 4448/25000 [====>.........................] - ETA: 54s - loss: 7.5735 - accuracy: 0.5061
 4480/25000 [====>.........................] - ETA: 54s - loss: 7.5776 - accuracy: 0.5058
 4512/25000 [====>.........................] - ETA: 54s - loss: 7.5715 - accuracy: 0.5062
 4544/25000 [====>.........................] - ETA: 54s - loss: 7.5823 - accuracy: 0.5055
 4576/25000 [====>.........................] - ETA: 53s - loss: 7.5728 - accuracy: 0.5061
 4608/25000 [====>.........................] - ETA: 53s - loss: 7.5468 - accuracy: 0.5078
 4640/25000 [====>.........................] - ETA: 53s - loss: 7.5477 - accuracy: 0.5078
 4672/25000 [====>.........................] - ETA: 53s - loss: 7.5386 - accuracy: 0.5083
 4704/25000 [====>.........................] - ETA: 53s - loss: 7.5297 - accuracy: 0.5089
 4736/25000 [====>.........................] - ETA: 53s - loss: 7.5274 - accuracy: 0.5091
 4768/25000 [====>.........................] - ETA: 53s - loss: 7.5316 - accuracy: 0.5088
 4800/25000 [====>.........................] - ETA: 53s - loss: 7.5452 - accuracy: 0.5079
 4832/25000 [====>.........................] - ETA: 53s - loss: 7.5365 - accuracy: 0.5085
 4864/25000 [====>.........................] - ETA: 53s - loss: 7.5405 - accuracy: 0.5082
 4896/25000 [====>.........................] - ETA: 52s - loss: 7.5539 - accuracy: 0.5074
 4928/25000 [====>.........................] - ETA: 52s - loss: 7.5484 - accuracy: 0.5077
 4960/25000 [====>.........................] - ETA: 52s - loss: 7.5553 - accuracy: 0.5073
 4992/25000 [====>.........................] - ETA: 52s - loss: 7.5591 - accuracy: 0.5070
 5024/25000 [=====>........................] - ETA: 52s - loss: 7.5659 - accuracy: 0.5066
 5056/25000 [=====>........................] - ETA: 52s - loss: 7.5787 - accuracy: 0.5057
 5088/25000 [=====>........................] - ETA: 52s - loss: 7.5672 - accuracy: 0.5065
 5120/25000 [=====>........................] - ETA: 52s - loss: 7.5798 - accuracy: 0.5057
 5152/25000 [=====>........................] - ETA: 52s - loss: 7.5833 - accuracy: 0.5054
 5184/25000 [=====>........................] - ETA: 51s - loss: 7.5779 - accuracy: 0.5058
 5216/25000 [=====>........................] - ETA: 51s - loss: 7.5814 - accuracy: 0.5056
 5248/25000 [=====>........................] - ETA: 51s - loss: 7.5819 - accuracy: 0.5055
 5280/25000 [=====>........................] - ETA: 51s - loss: 7.5853 - accuracy: 0.5053
 5312/25000 [=====>........................] - ETA: 51s - loss: 7.5887 - accuracy: 0.5051
 5344/25000 [=====>........................] - ETA: 51s - loss: 7.5949 - accuracy: 0.5047
 5376/25000 [=====>........................] - ETA: 51s - loss: 7.5839 - accuracy: 0.5054
 5408/25000 [=====>........................] - ETA: 51s - loss: 7.6014 - accuracy: 0.5043
 5440/25000 [=====>........................] - ETA: 51s - loss: 7.6046 - accuracy: 0.5040
 5472/25000 [=====>........................] - ETA: 51s - loss: 7.6078 - accuracy: 0.5038
 5504/25000 [=====>........................] - ETA: 50s - loss: 7.5914 - accuracy: 0.5049
 5536/25000 [=====>........................] - ETA: 50s - loss: 7.5891 - accuracy: 0.5051
 5568/25000 [=====>........................] - ETA: 50s - loss: 7.5950 - accuracy: 0.5047
 5600/25000 [=====>........................] - ETA: 50s - loss: 7.5954 - accuracy: 0.5046
 5632/25000 [=====>........................] - ETA: 50s - loss: 7.5958 - accuracy: 0.5046
 5664/25000 [=====>........................] - ETA: 50s - loss: 7.6044 - accuracy: 0.5041
 5696/25000 [=====>........................] - ETA: 50s - loss: 7.6074 - accuracy: 0.5039
 5728/25000 [=====>........................] - ETA: 50s - loss: 7.5997 - accuracy: 0.5044
 5760/25000 [=====>........................] - ETA: 50s - loss: 7.5947 - accuracy: 0.5047
 5792/25000 [=====>........................] - ETA: 50s - loss: 7.5898 - accuracy: 0.5050
 5824/25000 [=====>........................] - ETA: 49s - loss: 7.5982 - accuracy: 0.5045
 5856/25000 [======>.......................] - ETA: 49s - loss: 7.5985 - accuracy: 0.5044
 5888/25000 [======>.......................] - ETA: 49s - loss: 7.5937 - accuracy: 0.5048
 5920/25000 [======>.......................] - ETA: 49s - loss: 7.5863 - accuracy: 0.5052
 5952/25000 [======>.......................] - ETA: 49s - loss: 7.5868 - accuracy: 0.5052
 5984/25000 [======>.......................] - ETA: 49s - loss: 7.5949 - accuracy: 0.5047
 6016/25000 [======>.......................] - ETA: 49s - loss: 7.5927 - accuracy: 0.5048
 6048/25000 [======>.......................] - ETA: 49s - loss: 7.5855 - accuracy: 0.5053
 6080/25000 [======>.......................] - ETA: 49s - loss: 7.5910 - accuracy: 0.5049
 6112/25000 [======>.......................] - ETA: 49s - loss: 7.5989 - accuracy: 0.5044
 6144/25000 [======>.......................] - ETA: 49s - loss: 7.6017 - accuracy: 0.5042
 6176/25000 [======>.......................] - ETA: 48s - loss: 7.6021 - accuracy: 0.5042
 6208/25000 [======>.......................] - ETA: 48s - loss: 7.6098 - accuracy: 0.5037
 6240/25000 [======>.......................] - ETA: 48s - loss: 7.6175 - accuracy: 0.5032
 6272/25000 [======>.......................] - ETA: 48s - loss: 7.6251 - accuracy: 0.5027
 6304/25000 [======>.......................] - ETA: 48s - loss: 7.6228 - accuracy: 0.5029
 6336/25000 [======>.......................] - ETA: 48s - loss: 7.6376 - accuracy: 0.5019
 6368/25000 [======>.......................] - ETA: 48s - loss: 7.6401 - accuracy: 0.5017
 6400/25000 [======>.......................] - ETA: 48s - loss: 7.6403 - accuracy: 0.5017
 6432/25000 [======>.......................] - ETA: 48s - loss: 7.6380 - accuracy: 0.5019
 6464/25000 [======>.......................] - ETA: 48s - loss: 7.6405 - accuracy: 0.5017
 6496/25000 [======>.......................] - ETA: 47s - loss: 7.6336 - accuracy: 0.5022
 6528/25000 [======>.......................] - ETA: 47s - loss: 7.6361 - accuracy: 0.5020
 6560/25000 [======>.......................] - ETA: 47s - loss: 7.6245 - accuracy: 0.5027
 6592/25000 [======>.......................] - ETA: 47s - loss: 7.6364 - accuracy: 0.5020
 6624/25000 [======>.......................] - ETA: 47s - loss: 7.6412 - accuracy: 0.5017
 6656/25000 [======>.......................] - ETA: 47s - loss: 7.6367 - accuracy: 0.5020
 6688/25000 [=======>......................] - ETA: 47s - loss: 7.6322 - accuracy: 0.5022
 6720/25000 [=======>......................] - ETA: 47s - loss: 7.6301 - accuracy: 0.5024
 6752/25000 [=======>......................] - ETA: 47s - loss: 7.6371 - accuracy: 0.5019
 6784/25000 [=======>......................] - ETA: 47s - loss: 7.6305 - accuracy: 0.5024
 6816/25000 [=======>......................] - ETA: 47s - loss: 7.6396 - accuracy: 0.5018
 6848/25000 [=======>......................] - ETA: 46s - loss: 7.6532 - accuracy: 0.5009
 6880/25000 [=======>......................] - ETA: 46s - loss: 7.6555 - accuracy: 0.5007
 6912/25000 [=======>......................] - ETA: 46s - loss: 7.6511 - accuracy: 0.5010
 6944/25000 [=======>......................] - ETA: 46s - loss: 7.6490 - accuracy: 0.5012
 6976/25000 [=======>......................] - ETA: 46s - loss: 7.6380 - accuracy: 0.5019
 7008/25000 [=======>......................] - ETA: 46s - loss: 7.6316 - accuracy: 0.5023
 7040/25000 [=======>......................] - ETA: 46s - loss: 7.6187 - accuracy: 0.5031
 7072/25000 [=======>......................] - ETA: 46s - loss: 7.6102 - accuracy: 0.5037
 7104/25000 [=======>......................] - ETA: 46s - loss: 7.6191 - accuracy: 0.5031
 7136/25000 [=======>......................] - ETA: 46s - loss: 7.6151 - accuracy: 0.5034
 7168/25000 [=======>......................] - ETA: 45s - loss: 7.6196 - accuracy: 0.5031
 7200/25000 [=======>......................] - ETA: 45s - loss: 7.6219 - accuracy: 0.5029
 7232/25000 [=======>......................] - ETA: 45s - loss: 7.6157 - accuracy: 0.5033
 7264/25000 [=======>......................] - ETA: 45s - loss: 7.6160 - accuracy: 0.5033
 7296/25000 [=======>......................] - ETA: 45s - loss: 7.6288 - accuracy: 0.5025
 7328/25000 [=======>......................] - ETA: 45s - loss: 7.6269 - accuracy: 0.5026
 7360/25000 [=======>......................] - ETA: 45s - loss: 7.6250 - accuracy: 0.5027
 7392/25000 [=======>......................] - ETA: 45s - loss: 7.6334 - accuracy: 0.5022
 7424/25000 [=======>......................] - ETA: 45s - loss: 7.6294 - accuracy: 0.5024
 7456/25000 [=======>......................] - ETA: 45s - loss: 7.6317 - accuracy: 0.5023
 7488/25000 [=======>......................] - ETA: 45s - loss: 7.6236 - accuracy: 0.5028
 7520/25000 [========>.....................] - ETA: 45s - loss: 7.6177 - accuracy: 0.5032
 7552/25000 [========>.....................] - ETA: 44s - loss: 7.6179 - accuracy: 0.5032
 7584/25000 [========>.....................] - ETA: 44s - loss: 7.6221 - accuracy: 0.5029
 7616/25000 [========>.....................] - ETA: 44s - loss: 7.6183 - accuracy: 0.5032
 7648/25000 [========>.....................] - ETA: 44s - loss: 7.6085 - accuracy: 0.5038
 7680/25000 [========>.....................] - ETA: 44s - loss: 7.6067 - accuracy: 0.5039
 7712/25000 [========>.....................] - ETA: 44s - loss: 7.6090 - accuracy: 0.5038
 7744/25000 [========>.....................] - ETA: 44s - loss: 7.6092 - accuracy: 0.5037
 7776/25000 [========>.....................] - ETA: 44s - loss: 7.5996 - accuracy: 0.5044
 7808/25000 [========>.....................] - ETA: 44s - loss: 7.5979 - accuracy: 0.5045
 7840/25000 [========>.....................] - ETA: 44s - loss: 7.5923 - accuracy: 0.5048
 7872/25000 [========>.....................] - ETA: 44s - loss: 7.5926 - accuracy: 0.5048
 7904/25000 [========>.....................] - ETA: 44s - loss: 7.5910 - accuracy: 0.5049
 7936/25000 [========>.....................] - ETA: 43s - loss: 7.5913 - accuracy: 0.5049
 7968/25000 [========>.....................] - ETA: 43s - loss: 7.5916 - accuracy: 0.5049
 8000/25000 [========>.....................] - ETA: 43s - loss: 7.6015 - accuracy: 0.5042
 8032/25000 [========>.....................] - ETA: 43s - loss: 7.6017 - accuracy: 0.5042
 8064/25000 [========>.....................] - ETA: 43s - loss: 7.6058 - accuracy: 0.5040
 8096/25000 [========>.....................] - ETA: 43s - loss: 7.6117 - accuracy: 0.5036
 8128/25000 [========>.....................] - ETA: 43s - loss: 7.6176 - accuracy: 0.5032
 8160/25000 [========>.....................] - ETA: 43s - loss: 7.6215 - accuracy: 0.5029
 8192/25000 [========>.....................] - ETA: 43s - loss: 7.6236 - accuracy: 0.5028
 8224/25000 [========>.....................] - ETA: 43s - loss: 7.6237 - accuracy: 0.5028
 8256/25000 [========>.....................] - ETA: 43s - loss: 7.6202 - accuracy: 0.5030
 8288/25000 [========>.....................] - ETA: 43s - loss: 7.6222 - accuracy: 0.5029
 8320/25000 [========>.....................] - ETA: 42s - loss: 7.6132 - accuracy: 0.5035
 8352/25000 [=========>....................] - ETA: 42s - loss: 7.6115 - accuracy: 0.5036
 8384/25000 [=========>....................] - ETA: 42s - loss: 7.6136 - accuracy: 0.5035
 8416/25000 [=========>....................] - ETA: 42s - loss: 7.6120 - accuracy: 0.5036
 8448/25000 [=========>....................] - ETA: 42s - loss: 7.6140 - accuracy: 0.5034
 8480/25000 [=========>....................] - ETA: 42s - loss: 7.6051 - accuracy: 0.5040
 8512/25000 [=========>....................] - ETA: 42s - loss: 7.6000 - accuracy: 0.5043
 8544/25000 [=========>....................] - ETA: 42s - loss: 7.6020 - accuracy: 0.5042
 8576/25000 [=========>....................] - ETA: 42s - loss: 7.5933 - accuracy: 0.5048
 8608/25000 [=========>....................] - ETA: 42s - loss: 7.5811 - accuracy: 0.5056
 8640/25000 [=========>....................] - ETA: 42s - loss: 7.5832 - accuracy: 0.5054
 8672/25000 [=========>....................] - ETA: 41s - loss: 7.5782 - accuracy: 0.5058
 8704/25000 [=========>....................] - ETA: 41s - loss: 7.5803 - accuracy: 0.5056
 8736/25000 [=========>....................] - ETA: 41s - loss: 7.5754 - accuracy: 0.5060
 8768/25000 [=========>....................] - ETA: 41s - loss: 7.5879 - accuracy: 0.5051
 8800/25000 [=========>....................] - ETA: 41s - loss: 7.5952 - accuracy: 0.5047
 8832/25000 [=========>....................] - ETA: 41s - loss: 7.5920 - accuracy: 0.5049
 8864/25000 [=========>....................] - ETA: 41s - loss: 7.5888 - accuracy: 0.5051
 8896/25000 [=========>....................] - ETA: 41s - loss: 7.5908 - accuracy: 0.5049
 8928/25000 [=========>....................] - ETA: 41s - loss: 7.5911 - accuracy: 0.5049
 8960/25000 [=========>....................] - ETA: 41s - loss: 7.5965 - accuracy: 0.5046
 8992/25000 [=========>....................] - ETA: 41s - loss: 7.6001 - accuracy: 0.5043
 9024/25000 [=========>....................] - ETA: 41s - loss: 7.6004 - accuracy: 0.5043
 9056/25000 [=========>....................] - ETA: 40s - loss: 7.6006 - accuracy: 0.5043
 9088/25000 [=========>....................] - ETA: 40s - loss: 7.5958 - accuracy: 0.5046
 9120/25000 [=========>....................] - ETA: 40s - loss: 7.5943 - accuracy: 0.5047
 9152/25000 [=========>....................] - ETA: 40s - loss: 7.5979 - accuracy: 0.5045
 9184/25000 [==========>...................] - ETA: 40s - loss: 7.6032 - accuracy: 0.5041
 9216/25000 [==========>...................] - ETA: 40s - loss: 7.6067 - accuracy: 0.5039
 9248/25000 [==========>...................] - ETA: 40s - loss: 7.6069 - accuracy: 0.5039
 9280/25000 [==========>...................] - ETA: 40s - loss: 7.6071 - accuracy: 0.5039
 9312/25000 [==========>...................] - ETA: 40s - loss: 7.6040 - accuracy: 0.5041
 9344/25000 [==========>...................] - ETA: 40s - loss: 7.6059 - accuracy: 0.5040
 9376/25000 [==========>...................] - ETA: 40s - loss: 7.6061 - accuracy: 0.5039
 9408/25000 [==========>...................] - ETA: 40s - loss: 7.6112 - accuracy: 0.5036
 9440/25000 [==========>...................] - ETA: 39s - loss: 7.6065 - accuracy: 0.5039
 9472/25000 [==========>...................] - ETA: 39s - loss: 7.6002 - accuracy: 0.5043
 9504/25000 [==========>...................] - ETA: 39s - loss: 7.6037 - accuracy: 0.5041
 9536/25000 [==========>...................] - ETA: 39s - loss: 7.6087 - accuracy: 0.5038
 9568/25000 [==========>...................] - ETA: 39s - loss: 7.6121 - accuracy: 0.5036
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.6123 - accuracy: 0.5035
 9632/25000 [==========>...................] - ETA: 39s - loss: 7.6141 - accuracy: 0.5034
 9664/25000 [==========>...................] - ETA: 39s - loss: 7.6158 - accuracy: 0.5033
 9696/25000 [==========>...................] - ETA: 39s - loss: 7.6129 - accuracy: 0.5035
 9728/25000 [==========>...................] - ETA: 39s - loss: 7.6099 - accuracy: 0.5037
 9760/25000 [==========>...................] - ETA: 39s - loss: 7.6085 - accuracy: 0.5038
 9792/25000 [==========>...................] - ETA: 39s - loss: 7.5993 - accuracy: 0.5044
 9824/25000 [==========>...................] - ETA: 38s - loss: 7.6089 - accuracy: 0.5038
 9856/25000 [==========>...................] - ETA: 38s - loss: 7.5997 - accuracy: 0.5044
 9888/25000 [==========>...................] - ETA: 38s - loss: 7.5999 - accuracy: 0.5043
 9920/25000 [==========>...................] - ETA: 38s - loss: 7.6032 - accuracy: 0.5041
 9952/25000 [==========>...................] - ETA: 38s - loss: 7.6050 - accuracy: 0.5040
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.6052 - accuracy: 0.5040
10016/25000 [===========>..................] - ETA: 38s - loss: 7.6100 - accuracy: 0.5037
10048/25000 [===========>..................] - ETA: 38s - loss: 7.6102 - accuracy: 0.5037
10080/25000 [===========>..................] - ETA: 38s - loss: 7.6119 - accuracy: 0.5036
10112/25000 [===========>..................] - ETA: 38s - loss: 7.6181 - accuracy: 0.5032
10144/25000 [===========>..................] - ETA: 38s - loss: 7.6198 - accuracy: 0.5031
10176/25000 [===========>..................] - ETA: 38s - loss: 7.6244 - accuracy: 0.5028
10208/25000 [===========>..................] - ETA: 37s - loss: 7.6276 - accuracy: 0.5025
10240/25000 [===========>..................] - ETA: 37s - loss: 7.6322 - accuracy: 0.5022
10272/25000 [===========>..................] - ETA: 37s - loss: 7.6293 - accuracy: 0.5024
10304/25000 [===========>..................] - ETA: 37s - loss: 7.6324 - accuracy: 0.5022
10336/25000 [===========>..................] - ETA: 37s - loss: 7.6295 - accuracy: 0.5024
10368/25000 [===========>..................] - ETA: 37s - loss: 7.6282 - accuracy: 0.5025
10400/25000 [===========>..................] - ETA: 37s - loss: 7.6298 - accuracy: 0.5024
10432/25000 [===========>..................] - ETA: 37s - loss: 7.6313 - accuracy: 0.5023
10464/25000 [===========>..................] - ETA: 37s - loss: 7.6358 - accuracy: 0.5020
10496/25000 [===========>..................] - ETA: 37s - loss: 7.6403 - accuracy: 0.5017
10528/25000 [===========>..................] - ETA: 37s - loss: 7.6404 - accuracy: 0.5017
10560/25000 [===========>..................] - ETA: 37s - loss: 7.6463 - accuracy: 0.5013
10592/25000 [===========>..................] - ETA: 36s - loss: 7.6464 - accuracy: 0.5013
10624/25000 [===========>..................] - ETA: 36s - loss: 7.6450 - accuracy: 0.5014
10656/25000 [===========>..................] - ETA: 36s - loss: 7.6508 - accuracy: 0.5010
10688/25000 [===========>..................] - ETA: 36s - loss: 7.6508 - accuracy: 0.5010
10720/25000 [===========>..................] - ETA: 36s - loss: 7.6466 - accuracy: 0.5013
10752/25000 [===========>..................] - ETA: 36s - loss: 7.6495 - accuracy: 0.5011
10784/25000 [===========>..................] - ETA: 36s - loss: 7.6481 - accuracy: 0.5012
10816/25000 [===========>..................] - ETA: 36s - loss: 7.6468 - accuracy: 0.5013
10848/25000 [============>.................] - ETA: 36s - loss: 7.6525 - accuracy: 0.5009
10880/25000 [============>.................] - ETA: 36s - loss: 7.6568 - accuracy: 0.5006
10912/25000 [============>.................] - ETA: 36s - loss: 7.6610 - accuracy: 0.5004
10944/25000 [============>.................] - ETA: 36s - loss: 7.6652 - accuracy: 0.5001
10976/25000 [============>.................] - ETA: 36s - loss: 7.6722 - accuracy: 0.4996
11008/25000 [============>.................] - ETA: 35s - loss: 7.6708 - accuracy: 0.4997
11040/25000 [============>.................] - ETA: 35s - loss: 7.6680 - accuracy: 0.4999
11072/25000 [============>.................] - ETA: 35s - loss: 7.6722 - accuracy: 0.4996
11104/25000 [============>.................] - ETA: 35s - loss: 7.6790 - accuracy: 0.4992
11136/25000 [============>.................] - ETA: 35s - loss: 7.6735 - accuracy: 0.4996
11168/25000 [============>.................] - ETA: 35s - loss: 7.6762 - accuracy: 0.4994
11200/25000 [============>.................] - ETA: 35s - loss: 7.6735 - accuracy: 0.4996
11232/25000 [============>.................] - ETA: 35s - loss: 7.6707 - accuracy: 0.4997
11264/25000 [============>.................] - ETA: 35s - loss: 7.6721 - accuracy: 0.4996
11296/25000 [============>.................] - ETA: 35s - loss: 7.6775 - accuracy: 0.4993
11328/25000 [============>.................] - ETA: 35s - loss: 7.6802 - accuracy: 0.4991
11360/25000 [============>.................] - ETA: 35s - loss: 7.6747 - accuracy: 0.4995
11392/25000 [============>.................] - ETA: 34s - loss: 7.6774 - accuracy: 0.4993
11424/25000 [============>.................] - ETA: 34s - loss: 7.6733 - accuracy: 0.4996
11456/25000 [============>.................] - ETA: 34s - loss: 7.6747 - accuracy: 0.4995
11488/25000 [============>.................] - ETA: 34s - loss: 7.6760 - accuracy: 0.4994
11520/25000 [============>.................] - ETA: 34s - loss: 7.6733 - accuracy: 0.4996
11552/25000 [============>.................] - ETA: 34s - loss: 7.6786 - accuracy: 0.4992
11584/25000 [============>.................] - ETA: 34s - loss: 7.6785 - accuracy: 0.4992
11616/25000 [============>.................] - ETA: 34s - loss: 7.6785 - accuracy: 0.4992
11648/25000 [============>.................] - ETA: 34s - loss: 7.6785 - accuracy: 0.4992
11680/25000 [=============>................] - ETA: 34s - loss: 7.6797 - accuracy: 0.4991
11712/25000 [=============>................] - ETA: 34s - loss: 7.6732 - accuracy: 0.4996
11744/25000 [=============>................] - ETA: 33s - loss: 7.6758 - accuracy: 0.4994
11776/25000 [=============>................] - ETA: 33s - loss: 7.6783 - accuracy: 0.4992
11808/25000 [=============>................] - ETA: 33s - loss: 7.6822 - accuracy: 0.4990
11840/25000 [=============>................] - ETA: 33s - loss: 7.6873 - accuracy: 0.4986
11872/25000 [=============>................] - ETA: 33s - loss: 7.6847 - accuracy: 0.4988
11904/25000 [=============>................] - ETA: 33s - loss: 7.6872 - accuracy: 0.4987
11936/25000 [=============>................] - ETA: 33s - loss: 7.6859 - accuracy: 0.4987
11968/25000 [=============>................] - ETA: 33s - loss: 7.6935 - accuracy: 0.4982
12000/25000 [=============>................] - ETA: 33s - loss: 7.6909 - accuracy: 0.4984
12032/25000 [=============>................] - ETA: 33s - loss: 7.6870 - accuracy: 0.4987
12064/25000 [=============>................] - ETA: 33s - loss: 7.6870 - accuracy: 0.4987
12096/25000 [=============>................] - ETA: 33s - loss: 7.6932 - accuracy: 0.4983
12128/25000 [=============>................] - ETA: 33s - loss: 7.6932 - accuracy: 0.4983
12160/25000 [=============>................] - ETA: 32s - loss: 7.6944 - accuracy: 0.4982
12192/25000 [=============>................] - ETA: 32s - loss: 7.6981 - accuracy: 0.4979
12224/25000 [=============>................] - ETA: 32s - loss: 7.6992 - accuracy: 0.4979
12256/25000 [=============>................] - ETA: 32s - loss: 7.7016 - accuracy: 0.4977
12288/25000 [=============>................] - ETA: 32s - loss: 7.7016 - accuracy: 0.4977
12320/25000 [=============>................] - ETA: 32s - loss: 7.7040 - accuracy: 0.4976
12352/25000 [=============>................] - ETA: 32s - loss: 7.7001 - accuracy: 0.4978
12384/25000 [=============>................] - ETA: 32s - loss: 7.7000 - accuracy: 0.4978
12416/25000 [=============>................] - ETA: 32s - loss: 7.7024 - accuracy: 0.4977
12448/25000 [=============>................] - ETA: 32s - loss: 7.7023 - accuracy: 0.4977
12480/25000 [=============>................] - ETA: 32s - loss: 7.6986 - accuracy: 0.4979
12512/25000 [==============>...............] - ETA: 32s - loss: 7.6997 - accuracy: 0.4978
12544/25000 [==============>...............] - ETA: 31s - loss: 7.6996 - accuracy: 0.4978
12576/25000 [==============>...............] - ETA: 31s - loss: 7.6971 - accuracy: 0.4980
12608/25000 [==============>...............] - ETA: 31s - loss: 7.6946 - accuracy: 0.4982
12640/25000 [==============>...............] - ETA: 31s - loss: 7.6933 - accuracy: 0.4983
12672/25000 [==============>...............] - ETA: 31s - loss: 7.6920 - accuracy: 0.4983
12704/25000 [==============>...............] - ETA: 31s - loss: 7.6871 - accuracy: 0.4987
12736/25000 [==============>...............] - ETA: 31s - loss: 7.6799 - accuracy: 0.4991
12768/25000 [==============>...............] - ETA: 31s - loss: 7.6750 - accuracy: 0.4995
12800/25000 [==============>...............] - ETA: 31s - loss: 7.6774 - accuracy: 0.4993
12832/25000 [==============>...............] - ETA: 31s - loss: 7.6762 - accuracy: 0.4994
12864/25000 [==============>...............] - ETA: 31s - loss: 7.6738 - accuracy: 0.4995
12896/25000 [==============>...............] - ETA: 30s - loss: 7.6738 - accuracy: 0.4995
12928/25000 [==============>...............] - ETA: 30s - loss: 7.6761 - accuracy: 0.4994
12960/25000 [==============>...............] - ETA: 30s - loss: 7.6796 - accuracy: 0.4992
12992/25000 [==============>...............] - ETA: 30s - loss: 7.6808 - accuracy: 0.4991
13024/25000 [==============>...............] - ETA: 30s - loss: 7.6749 - accuracy: 0.4995
13056/25000 [==============>...............] - ETA: 30s - loss: 7.6795 - accuracy: 0.4992
13088/25000 [==============>...............] - ETA: 30s - loss: 7.6736 - accuracy: 0.4995
13120/25000 [==============>...............] - ETA: 30s - loss: 7.6771 - accuracy: 0.4993
13152/25000 [==============>...............] - ETA: 30s - loss: 7.6736 - accuracy: 0.4995
13184/25000 [==============>...............] - ETA: 30s - loss: 7.6782 - accuracy: 0.4992
13216/25000 [==============>...............] - ETA: 30s - loss: 7.6771 - accuracy: 0.4993
13248/25000 [==============>...............] - ETA: 30s - loss: 7.6805 - accuracy: 0.4991
13280/25000 [==============>...............] - ETA: 29s - loss: 7.6805 - accuracy: 0.4991
13312/25000 [==============>...............] - ETA: 29s - loss: 7.6781 - accuracy: 0.4992
13344/25000 [===============>..............] - ETA: 29s - loss: 7.6793 - accuracy: 0.4992
13376/25000 [===============>..............] - ETA: 29s - loss: 7.6746 - accuracy: 0.4995
13408/25000 [===============>..............] - ETA: 29s - loss: 7.6746 - accuracy: 0.4995
13440/25000 [===============>..............] - ETA: 29s - loss: 7.6780 - accuracy: 0.4993
13472/25000 [===============>..............] - ETA: 29s - loss: 7.6848 - accuracy: 0.4988
13504/25000 [===============>..............] - ETA: 29s - loss: 7.6837 - accuracy: 0.4989
13536/25000 [===============>..............] - ETA: 29s - loss: 7.6802 - accuracy: 0.4991
13568/25000 [===============>..............] - ETA: 29s - loss: 7.6847 - accuracy: 0.4988
13600/25000 [===============>..............] - ETA: 29s - loss: 7.6790 - accuracy: 0.4992
13632/25000 [===============>..............] - ETA: 29s - loss: 7.6824 - accuracy: 0.4990
13664/25000 [===============>..............] - ETA: 28s - loss: 7.6857 - accuracy: 0.4988
13696/25000 [===============>..............] - ETA: 28s - loss: 7.6857 - accuracy: 0.4988
13728/25000 [===============>..............] - ETA: 28s - loss: 7.6867 - accuracy: 0.4987
13760/25000 [===============>..............] - ETA: 28s - loss: 7.6900 - accuracy: 0.4985
13792/25000 [===============>..............] - ETA: 28s - loss: 7.6922 - accuracy: 0.4983
13824/25000 [===============>..............] - ETA: 28s - loss: 7.6855 - accuracy: 0.4988
13856/25000 [===============>..............] - ETA: 28s - loss: 7.6910 - accuracy: 0.4984
13888/25000 [===============>..............] - ETA: 28s - loss: 7.6953 - accuracy: 0.4981
13920/25000 [===============>..............] - ETA: 28s - loss: 7.6975 - accuracy: 0.4980
13952/25000 [===============>..............] - ETA: 28s - loss: 7.7018 - accuracy: 0.4977
13984/25000 [===============>..............] - ETA: 28s - loss: 7.6962 - accuracy: 0.4981
14016/25000 [===============>..............] - ETA: 28s - loss: 7.7005 - accuracy: 0.4978
14048/25000 [===============>..............] - ETA: 27s - loss: 7.6972 - accuracy: 0.4980
14080/25000 [===============>..............] - ETA: 27s - loss: 7.6960 - accuracy: 0.4981
14112/25000 [===============>..............] - ETA: 27s - loss: 7.6960 - accuracy: 0.4981
14144/25000 [===============>..............] - ETA: 27s - loss: 7.6981 - accuracy: 0.4979
14176/25000 [================>.............] - ETA: 27s - loss: 7.6980 - accuracy: 0.4980
14208/25000 [================>.............] - ETA: 27s - loss: 7.6958 - accuracy: 0.4981
14240/25000 [================>.............] - ETA: 27s - loss: 7.6968 - accuracy: 0.4980
14272/25000 [================>.............] - ETA: 27s - loss: 7.6978 - accuracy: 0.4980
14304/25000 [================>.............] - ETA: 27s - loss: 7.6977 - accuracy: 0.4980
14336/25000 [================>.............] - ETA: 27s - loss: 7.6966 - accuracy: 0.4980
14368/25000 [================>.............] - ETA: 27s - loss: 7.6965 - accuracy: 0.4981
14400/25000 [================>.............] - ETA: 27s - loss: 7.6932 - accuracy: 0.4983
14432/25000 [================>.............] - ETA: 26s - loss: 7.6932 - accuracy: 0.4983
14464/25000 [================>.............] - ETA: 26s - loss: 7.6910 - accuracy: 0.4984
14496/25000 [================>.............] - ETA: 26s - loss: 7.6941 - accuracy: 0.4982
14528/25000 [================>.............] - ETA: 26s - loss: 7.6930 - accuracy: 0.4983
14560/25000 [================>.............] - ETA: 26s - loss: 7.6972 - accuracy: 0.4980
14592/25000 [================>.............] - ETA: 26s - loss: 7.6960 - accuracy: 0.4981
14624/25000 [================>.............] - ETA: 26s - loss: 7.6907 - accuracy: 0.4984
14656/25000 [================>.............] - ETA: 26s - loss: 7.6865 - accuracy: 0.4987
14688/25000 [================>.............] - ETA: 26s - loss: 7.6854 - accuracy: 0.4988
14720/25000 [================>.............] - ETA: 26s - loss: 7.6822 - accuracy: 0.4990
14752/25000 [================>.............] - ETA: 26s - loss: 7.6801 - accuracy: 0.4991
14784/25000 [================>.............] - ETA: 26s - loss: 7.6760 - accuracy: 0.4994
14816/25000 [================>.............] - ETA: 25s - loss: 7.6739 - accuracy: 0.4995
14848/25000 [================>.............] - ETA: 25s - loss: 7.6708 - accuracy: 0.4997
14880/25000 [================>.............] - ETA: 25s - loss: 7.6728 - accuracy: 0.4996
14912/25000 [================>.............] - ETA: 25s - loss: 7.6707 - accuracy: 0.4997
14944/25000 [================>.............] - ETA: 25s - loss: 7.6707 - accuracy: 0.4997
14976/25000 [================>.............] - ETA: 25s - loss: 7.6666 - accuracy: 0.5000
15008/25000 [=================>............] - ETA: 25s - loss: 7.6646 - accuracy: 0.5001
15040/25000 [=================>............] - ETA: 25s - loss: 7.6646 - accuracy: 0.5001
15072/25000 [=================>............] - ETA: 25s - loss: 7.6717 - accuracy: 0.4997
15104/25000 [=================>............] - ETA: 25s - loss: 7.6747 - accuracy: 0.4995
15136/25000 [=================>............] - ETA: 25s - loss: 7.6737 - accuracy: 0.4995
15168/25000 [=================>............] - ETA: 25s - loss: 7.6727 - accuracy: 0.4996
15200/25000 [=================>............] - ETA: 24s - loss: 7.6717 - accuracy: 0.4997
15232/25000 [=================>............] - ETA: 24s - loss: 7.6666 - accuracy: 0.5000
15264/25000 [=================>............] - ETA: 24s - loss: 7.6686 - accuracy: 0.4999
15296/25000 [=================>............] - ETA: 24s - loss: 7.6646 - accuracy: 0.5001
15328/25000 [=================>............] - ETA: 24s - loss: 7.6686 - accuracy: 0.4999
15360/25000 [=================>............] - ETA: 24s - loss: 7.6706 - accuracy: 0.4997
15392/25000 [=================>............] - ETA: 24s - loss: 7.6716 - accuracy: 0.4997
15424/25000 [=================>............] - ETA: 24s - loss: 7.6726 - accuracy: 0.4996
15456/25000 [=================>............] - ETA: 24s - loss: 7.6656 - accuracy: 0.5001
15488/25000 [=================>............] - ETA: 24s - loss: 7.6636 - accuracy: 0.5002
15520/25000 [=================>............] - ETA: 24s - loss: 7.6646 - accuracy: 0.5001
15552/25000 [=================>............] - ETA: 24s - loss: 7.6607 - accuracy: 0.5004
15584/25000 [=================>............] - ETA: 23s - loss: 7.6558 - accuracy: 0.5007
15616/25000 [=================>............] - ETA: 23s - loss: 7.6548 - accuracy: 0.5008
15648/25000 [=================>............] - ETA: 23s - loss: 7.6539 - accuracy: 0.5008
15680/25000 [=================>............] - ETA: 23s - loss: 7.6588 - accuracy: 0.5005
15712/25000 [=================>............] - ETA: 23s - loss: 7.6569 - accuracy: 0.5006
15744/25000 [=================>............] - ETA: 23s - loss: 7.6588 - accuracy: 0.5005
15776/25000 [=================>............] - ETA: 23s - loss: 7.6618 - accuracy: 0.5003
15808/25000 [=================>............] - ETA: 23s - loss: 7.6627 - accuracy: 0.5003
15840/25000 [==================>...........] - ETA: 23s - loss: 7.6608 - accuracy: 0.5004
15872/25000 [==================>...........] - ETA: 23s - loss: 7.6589 - accuracy: 0.5005
15904/25000 [==================>...........] - ETA: 23s - loss: 7.6589 - accuracy: 0.5005
15936/25000 [==================>...........] - ETA: 23s - loss: 7.6599 - accuracy: 0.5004
15968/25000 [==================>...........] - ETA: 23s - loss: 7.6561 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 22s - loss: 7.6561 - accuracy: 0.5007
16032/25000 [==================>...........] - ETA: 22s - loss: 7.6542 - accuracy: 0.5008
16064/25000 [==================>...........] - ETA: 22s - loss: 7.6552 - accuracy: 0.5007
16096/25000 [==================>...........] - ETA: 22s - loss: 7.6542 - accuracy: 0.5008
16128/25000 [==================>...........] - ETA: 22s - loss: 7.6562 - accuracy: 0.5007
16160/25000 [==================>...........] - ETA: 22s - loss: 7.6562 - accuracy: 0.5007
16192/25000 [==================>...........] - ETA: 22s - loss: 7.6571 - accuracy: 0.5006
16224/25000 [==================>...........] - ETA: 22s - loss: 7.6524 - accuracy: 0.5009
16256/25000 [==================>...........] - ETA: 22s - loss: 7.6515 - accuracy: 0.5010
16288/25000 [==================>...........] - ETA: 22s - loss: 7.6506 - accuracy: 0.5010
16320/25000 [==================>...........] - ETA: 22s - loss: 7.6469 - accuracy: 0.5013
16352/25000 [==================>...........] - ETA: 22s - loss: 7.6488 - accuracy: 0.5012
16384/25000 [==================>...........] - ETA: 21s - loss: 7.6535 - accuracy: 0.5009
16416/25000 [==================>...........] - ETA: 21s - loss: 7.6545 - accuracy: 0.5008
16448/25000 [==================>...........] - ETA: 21s - loss: 7.6601 - accuracy: 0.5004
16480/25000 [==================>...........] - ETA: 21s - loss: 7.6648 - accuracy: 0.5001
16512/25000 [==================>...........] - ETA: 21s - loss: 7.6610 - accuracy: 0.5004
16544/25000 [==================>...........] - ETA: 21s - loss: 7.6648 - accuracy: 0.5001
16576/25000 [==================>...........] - ETA: 21s - loss: 7.6694 - accuracy: 0.4998
16608/25000 [==================>...........] - ETA: 21s - loss: 7.6675 - accuracy: 0.4999
16640/25000 [==================>...........] - ETA: 21s - loss: 7.6685 - accuracy: 0.4999
16672/25000 [===================>..........] - ETA: 21s - loss: 7.6675 - accuracy: 0.4999
16704/25000 [===================>..........] - ETA: 21s - loss: 7.6639 - accuracy: 0.5002
16736/25000 [===================>..........] - ETA: 21s - loss: 7.6630 - accuracy: 0.5002
16768/25000 [===================>..........] - ETA: 20s - loss: 7.6648 - accuracy: 0.5001
16800/25000 [===================>..........] - ETA: 20s - loss: 7.6684 - accuracy: 0.4999
16832/25000 [===================>..........] - ETA: 20s - loss: 7.6694 - accuracy: 0.4998
16864/25000 [===================>..........] - ETA: 20s - loss: 7.6703 - accuracy: 0.4998
16896/25000 [===================>..........] - ETA: 20s - loss: 7.6684 - accuracy: 0.4999
16928/25000 [===================>..........] - ETA: 20s - loss: 7.6675 - accuracy: 0.4999
16960/25000 [===================>..........] - ETA: 20s - loss: 7.6711 - accuracy: 0.4997
16992/25000 [===================>..........] - ETA: 20s - loss: 7.6738 - accuracy: 0.4995
17024/25000 [===================>..........] - ETA: 20s - loss: 7.6720 - accuracy: 0.4996
17056/25000 [===================>..........] - ETA: 20s - loss: 7.6756 - accuracy: 0.4994
17088/25000 [===================>..........] - ETA: 20s - loss: 7.6729 - accuracy: 0.4996
17120/25000 [===================>..........] - ETA: 20s - loss: 7.6702 - accuracy: 0.4998
17152/25000 [===================>..........] - ETA: 19s - loss: 7.6675 - accuracy: 0.4999
17184/25000 [===================>..........] - ETA: 19s - loss: 7.6657 - accuracy: 0.5001
17216/25000 [===================>..........] - ETA: 19s - loss: 7.6631 - accuracy: 0.5002
17248/25000 [===================>..........] - ETA: 19s - loss: 7.6648 - accuracy: 0.5001
17280/25000 [===================>..........] - ETA: 19s - loss: 7.6622 - accuracy: 0.5003
17312/25000 [===================>..........] - ETA: 19s - loss: 7.6631 - accuracy: 0.5002
17344/25000 [===================>..........] - ETA: 19s - loss: 7.6640 - accuracy: 0.5002
17376/25000 [===================>..........] - ETA: 19s - loss: 7.6631 - accuracy: 0.5002
17408/25000 [===================>..........] - ETA: 19s - loss: 7.6693 - accuracy: 0.4998
17440/25000 [===================>..........] - ETA: 19s - loss: 7.6710 - accuracy: 0.4997
17472/25000 [===================>..........] - ETA: 19s - loss: 7.6736 - accuracy: 0.4995
17504/25000 [====================>.........] - ETA: 19s - loss: 7.6728 - accuracy: 0.4996
17536/25000 [====================>.........] - ETA: 18s - loss: 7.6736 - accuracy: 0.4995
17568/25000 [====================>.........] - ETA: 18s - loss: 7.6736 - accuracy: 0.4995
17600/25000 [====================>.........] - ETA: 18s - loss: 7.6727 - accuracy: 0.4996
17632/25000 [====================>.........] - ETA: 18s - loss: 7.6744 - accuracy: 0.4995
17664/25000 [====================>.........] - ETA: 18s - loss: 7.6762 - accuracy: 0.4994
17696/25000 [====================>.........] - ETA: 18s - loss: 7.6788 - accuracy: 0.4992
17728/25000 [====================>.........] - ETA: 18s - loss: 7.6839 - accuracy: 0.4989
17760/25000 [====================>.........] - ETA: 18s - loss: 7.6847 - accuracy: 0.4988
17792/25000 [====================>.........] - ETA: 18s - loss: 7.6839 - accuracy: 0.4989
17824/25000 [====================>.........] - ETA: 18s - loss: 7.6847 - accuracy: 0.4988
17856/25000 [====================>.........] - ETA: 18s - loss: 7.6838 - accuracy: 0.4989
17888/25000 [====================>.........] - ETA: 18s - loss: 7.6838 - accuracy: 0.4989
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6880 - accuracy: 0.4986
17952/25000 [====================>.........] - ETA: 17s - loss: 7.6914 - accuracy: 0.4984
17984/25000 [====================>.........] - ETA: 17s - loss: 7.6948 - accuracy: 0.4982
18016/25000 [====================>.........] - ETA: 17s - loss: 7.6973 - accuracy: 0.4980
18048/25000 [====================>.........] - ETA: 17s - loss: 7.6972 - accuracy: 0.4980
18080/25000 [====================>.........] - ETA: 17s - loss: 7.6988 - accuracy: 0.4979
18112/25000 [====================>.........] - ETA: 17s - loss: 7.6979 - accuracy: 0.4980
18144/25000 [====================>.........] - ETA: 17s - loss: 7.6928 - accuracy: 0.4983
18176/25000 [====================>.........] - ETA: 17s - loss: 7.6928 - accuracy: 0.4983
18208/25000 [====================>.........] - ETA: 17s - loss: 7.6936 - accuracy: 0.4982
18240/25000 [====================>.........] - ETA: 17s - loss: 7.6902 - accuracy: 0.4985
18272/25000 [====================>.........] - ETA: 17s - loss: 7.6901 - accuracy: 0.4985
18304/25000 [====================>.........] - ETA: 17s - loss: 7.6884 - accuracy: 0.4986
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6892 - accuracy: 0.4985
18368/25000 [=====================>........] - ETA: 16s - loss: 7.6892 - accuracy: 0.4985
18400/25000 [=====================>........] - ETA: 16s - loss: 7.6916 - accuracy: 0.4984
18432/25000 [=====================>........] - ETA: 16s - loss: 7.6941 - accuracy: 0.4982
18464/25000 [=====================>........] - ETA: 16s - loss: 7.6965 - accuracy: 0.4981
18496/25000 [=====================>........] - ETA: 16s - loss: 7.6973 - accuracy: 0.4980
18528/25000 [=====================>........] - ETA: 16s - loss: 7.6956 - accuracy: 0.4981
18560/25000 [=====================>........] - ETA: 16s - loss: 7.6955 - accuracy: 0.4981
18592/25000 [=====================>........] - ETA: 16s - loss: 7.6963 - accuracy: 0.4981
18624/25000 [=====================>........] - ETA: 16s - loss: 7.6987 - accuracy: 0.4979
18656/25000 [=====================>........] - ETA: 16s - loss: 7.6970 - accuracy: 0.4980
18688/25000 [=====================>........] - ETA: 16s - loss: 7.6962 - accuracy: 0.4981
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6969 - accuracy: 0.4980
18752/25000 [=====================>........] - ETA: 15s - loss: 7.7001 - accuracy: 0.4978
18784/25000 [=====================>........] - ETA: 15s - loss: 7.7034 - accuracy: 0.4976
18816/25000 [=====================>........] - ETA: 15s - loss: 7.7025 - accuracy: 0.4977
18848/25000 [=====================>........] - ETA: 15s - loss: 7.7089 - accuracy: 0.4972
18880/25000 [=====================>........] - ETA: 15s - loss: 7.7080 - accuracy: 0.4973
18912/25000 [=====================>........] - ETA: 15s - loss: 7.7072 - accuracy: 0.4974
18944/25000 [=====================>........] - ETA: 15s - loss: 7.7047 - accuracy: 0.4975
18976/25000 [=====================>........] - ETA: 15s - loss: 7.7014 - accuracy: 0.4977
19008/25000 [=====================>........] - ETA: 15s - loss: 7.6981 - accuracy: 0.4979
19040/25000 [=====================>........] - ETA: 15s - loss: 7.7012 - accuracy: 0.4977
19072/25000 [=====================>........] - ETA: 15s - loss: 7.6996 - accuracy: 0.4979
19104/25000 [=====================>........] - ETA: 14s - loss: 7.7011 - accuracy: 0.4977
19136/25000 [=====================>........] - ETA: 14s - loss: 7.7011 - accuracy: 0.4978
19168/25000 [======================>.......] - ETA: 14s - loss: 7.7066 - accuracy: 0.4974
19200/25000 [======================>.......] - ETA: 14s - loss: 7.7058 - accuracy: 0.4974
19232/25000 [======================>.......] - ETA: 14s - loss: 7.7089 - accuracy: 0.4972
19264/25000 [======================>.......] - ETA: 14s - loss: 7.7088 - accuracy: 0.4972
19296/25000 [======================>.......] - ETA: 14s - loss: 7.7087 - accuracy: 0.4973
19328/25000 [======================>.......] - ETA: 14s - loss: 7.7079 - accuracy: 0.4973
19360/25000 [======================>.......] - ETA: 14s - loss: 7.7078 - accuracy: 0.4973
19392/25000 [======================>.......] - ETA: 14s - loss: 7.7054 - accuracy: 0.4975
19424/25000 [======================>.......] - ETA: 14s - loss: 7.7045 - accuracy: 0.4975
19456/25000 [======================>.......] - ETA: 14s - loss: 7.7021 - accuracy: 0.4977
19488/25000 [======================>.......] - ETA: 13s - loss: 7.7005 - accuracy: 0.4978
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6996 - accuracy: 0.4978
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6988 - accuracy: 0.4979
19584/25000 [======================>.......] - ETA: 13s - loss: 7.6956 - accuracy: 0.4981
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6963 - accuracy: 0.4981
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6971 - accuracy: 0.4980
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6939 - accuracy: 0.4982
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6931 - accuracy: 0.4983
19744/25000 [======================>.......] - ETA: 13s - loss: 7.6946 - accuracy: 0.4982
19776/25000 [======================>.......] - ETA: 13s - loss: 7.6969 - accuracy: 0.4980
19808/25000 [======================>.......] - ETA: 13s - loss: 7.6960 - accuracy: 0.4981
19840/25000 [======================>.......] - ETA: 13s - loss: 7.6929 - accuracy: 0.4983
19872/25000 [======================>.......] - ETA: 13s - loss: 7.6952 - accuracy: 0.4981
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6905 - accuracy: 0.4984
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6912 - accuracy: 0.4984
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6904 - accuracy: 0.4984
20000/25000 [=======================>......] - ETA: 12s - loss: 7.6873 - accuracy: 0.4987
20032/25000 [=======================>......] - ETA: 12s - loss: 7.6865 - accuracy: 0.4987
20064/25000 [=======================>......] - ETA: 12s - loss: 7.6865 - accuracy: 0.4987
20096/25000 [=======================>......] - ETA: 12s - loss: 7.6895 - accuracy: 0.4985
20128/25000 [=======================>......] - ETA: 12s - loss: 7.6879 - accuracy: 0.4986
20160/25000 [=======================>......] - ETA: 12s - loss: 7.6932 - accuracy: 0.4983
20192/25000 [=======================>......] - ETA: 12s - loss: 7.6970 - accuracy: 0.4980
20224/25000 [=======================>......] - ETA: 12s - loss: 7.6962 - accuracy: 0.4981
20256/25000 [=======================>......] - ETA: 12s - loss: 7.6999 - accuracy: 0.4978
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6976 - accuracy: 0.4980
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6976 - accuracy: 0.4980
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6990 - accuracy: 0.4979
20384/25000 [=======================>......] - ETA: 11s - loss: 7.7005 - accuracy: 0.4978
20416/25000 [=======================>......] - ETA: 11s - loss: 7.7012 - accuracy: 0.4977
20448/25000 [=======================>......] - ETA: 11s - loss: 7.7004 - accuracy: 0.4978
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6996 - accuracy: 0.4979
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6995 - accuracy: 0.4979
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6942 - accuracy: 0.4982
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6927 - accuracy: 0.4983
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6919 - accuracy: 0.4984
20640/25000 [=======================>......] - ETA: 11s - loss: 7.6911 - accuracy: 0.4984
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6889 - accuracy: 0.4985
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6866 - accuracy: 0.4987
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6836 - accuracy: 0.4989
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6829 - accuracy: 0.4989
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6850 - accuracy: 0.4988
20832/25000 [=======================>......] - ETA: 10s - loss: 7.6858 - accuracy: 0.4988
20864/25000 [========================>.....] - ETA: 10s - loss: 7.6828 - accuracy: 0.4989
20896/25000 [========================>.....] - ETA: 10s - loss: 7.6828 - accuracy: 0.4989
20928/25000 [========================>.....] - ETA: 10s - loss: 7.6820 - accuracy: 0.4990
20960/25000 [========================>.....] - ETA: 10s - loss: 7.6798 - accuracy: 0.4991
20992/25000 [========================>.....] - ETA: 10s - loss: 7.6768 - accuracy: 0.4993
21024/25000 [========================>.....] - ETA: 10s - loss: 7.6776 - accuracy: 0.4993
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6768 - accuracy: 0.4993 
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6775 - accuracy: 0.4993
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6768 - accuracy: 0.4993
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6775 - accuracy: 0.4993
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6825 - accuracy: 0.4990
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6804 - accuracy: 0.4991
21248/25000 [========================>.....] - ETA: 9s - loss: 7.6803 - accuracy: 0.4991
21280/25000 [========================>.....] - ETA: 9s - loss: 7.6789 - accuracy: 0.4992
21312/25000 [========================>.....] - ETA: 9s - loss: 7.6774 - accuracy: 0.4993
21344/25000 [========================>.....] - ETA: 9s - loss: 7.6781 - accuracy: 0.4993
21376/25000 [========================>.....] - ETA: 9s - loss: 7.6752 - accuracy: 0.4994
21408/25000 [========================>.....] - ETA: 9s - loss: 7.6752 - accuracy: 0.4994
21440/25000 [========================>.....] - ETA: 9s - loss: 7.6752 - accuracy: 0.4994
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6759 - accuracy: 0.4994
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6795 - accuracy: 0.4992
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6837 - accuracy: 0.4989
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6837 - accuracy: 0.4989
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6844 - accuracy: 0.4988
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6829 - accuracy: 0.4989
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6829 - accuracy: 0.4989
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6843 - accuracy: 0.4988
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6829 - accuracy: 0.4989
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6828 - accuracy: 0.4989
21792/25000 [=========================>....] - ETA: 8s - loss: 7.6828 - accuracy: 0.4989
21824/25000 [=========================>....] - ETA: 8s - loss: 7.6835 - accuracy: 0.4989
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6828 - accuracy: 0.4989
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6841 - accuracy: 0.4989
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6848 - accuracy: 0.4988
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6855 - accuracy: 0.4988
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6827 - accuracy: 0.4990
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6833 - accuracy: 0.4989
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6833 - accuracy: 0.4989
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6833 - accuracy: 0.4989
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6833 - accuracy: 0.4989
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6812 - accuracy: 0.4991
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6804 - accuracy: 0.4991
22208/25000 [=========================>....] - ETA: 7s - loss: 7.6818 - accuracy: 0.4990
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6825 - accuracy: 0.4990
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6852 - accuracy: 0.4988
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6859 - accuracy: 0.4987
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6803 - accuracy: 0.4991
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6796 - accuracy: 0.4992
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6755 - accuracy: 0.4994
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6769 - accuracy: 0.4993
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6762 - accuracy: 0.4994
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6796 - accuracy: 0.4992
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6789 - accuracy: 0.4992
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6741 - accuracy: 0.4995
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6775 - accuracy: 0.4993
22624/25000 [==========================>...] - ETA: 6s - loss: 7.6768 - accuracy: 0.4993
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6747 - accuracy: 0.4995
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6720 - accuracy: 0.4996
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6713 - accuracy: 0.4997
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6707 - accuracy: 0.4997
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6693 - accuracy: 0.4998
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6720 - accuracy: 0.4997
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6726 - accuracy: 0.4996
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6740 - accuracy: 0.4995
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6753 - accuracy: 0.4994
23008/25000 [==========================>...] - ETA: 5s - loss: 7.6726 - accuracy: 0.4996
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6699 - accuracy: 0.4998
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6726 - accuracy: 0.4996
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6726 - accuracy: 0.4996
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6746 - accuracy: 0.4995
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6732 - accuracy: 0.4996
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6752 - accuracy: 0.4994
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6732 - accuracy: 0.4996
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6739 - accuracy: 0.4995
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6745 - accuracy: 0.4995
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6732 - accuracy: 0.4996
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6752 - accuracy: 0.4994
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6712 - accuracy: 0.4997
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6725 - accuracy: 0.4996
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6712 - accuracy: 0.4997
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6744 - accuracy: 0.4995
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6738 - accuracy: 0.4995
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6725 - accuracy: 0.4996
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6711 - accuracy: 0.4997
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6731 - accuracy: 0.4996
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6744 - accuracy: 0.4995
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6737 - accuracy: 0.4995
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6763 - accuracy: 0.4994
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6763 - accuracy: 0.4994
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6775 - accuracy: 0.4993
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6782 - accuracy: 0.4992
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6781 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6762 - accuracy: 0.4994
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6749 - accuracy: 0.4995
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6717 - accuracy: 0.4997
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6711 - accuracy: 0.4997
24192/25000 [============================>.] - ETA: 2s - loss: 7.6698 - accuracy: 0.4998
24224/25000 [============================>.] - ETA: 1s - loss: 7.6698 - accuracy: 0.4998
24256/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24288/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24320/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24352/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24384/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24416/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24448/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24480/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24512/25000 [============================>.] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
24544/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24576/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24608/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24640/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24672/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24704/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24736/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24768/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24800/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24864/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24896/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24928/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24960/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
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
