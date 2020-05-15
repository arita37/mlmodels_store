
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '4c47a0bacf53fc18eb078111d70311c747fed1fc', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/4c47a0bacf53fc18eb078111d70311c747fed1fc

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/4c47a0bacf53fc18eb078111d70311c747fed1fc

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
	Data preprocessing and feature engineering runtime = 0.24s ...
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
 40%|████      | 2/5 [00:55<01:22, 27.52s/it] 40%|████      | 2/5 [00:55<01:22, 27.52s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.4317159261957471, 'embedding_size_factor': 0.9497480853272854, 'layers.choice': 2, 'learning_rate': 0.009722541428009338, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 4.039012572695605e-12} and reward: 0.3872
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdb\xa1;\xd6\x0b\x13\xcdX\x15\x00\x00\x00embedding_size_factorq\x03G?\xeedV\x18\xbdj\xf9X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x83\xe9ik\xbd\n\x88X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x91\xc3\x86\x1c\xf4\xbfNu.' and reward: 0.3872
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdb\xa1;\xd6\x0b\x13\xcdX\x15\x00\x00\x00embedding_size_factorq\x03G?\xeedV\x18\xbdj\xf9X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x83\xe9ik\xbd\n\x88X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x91\xc3\x86\x1c\xf4\xbfNu.' and reward: 0.3872
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 110.10516595840454
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 2, 'dropout_prob': 0.4317159261957471, 'embedding_size_factor': 0.9497480853272854, 'layers.choice': 2, 'learning_rate': 0.009722541428009338, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 4.039012572695605e-12}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the 8.14s of remaining time.
Ensemble size: 9
Ensemble weights: 
[0.55555556 0.44444444]
	0.3926	 = Validation accuracy score
	0.89s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 112.79s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f29652adba8> 

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
 [-0.04893396  0.07092552  0.05384652 -0.07364387 -0.13088228 -0.01180258]
 [ 0.06923597  0.1069115   0.0086029  -0.03206631  0.26177403  0.06626451]
 [ 0.10151958  0.25749364  0.1132969   0.11964656  0.26516414  0.14422755]
 [-0.0916056   0.20320328  0.20363452 -0.14680552 -0.32350707  0.4052093 ]
 [ 0.09747619  0.02076602  0.26681924  0.60071832  0.63864273  0.1428992 ]
 [ 0.42955598  0.01686852  0.30976126  0.08986903 -0.38753006  0.10017428]
 [-0.39217162 -0.0421437   0.28333423  0.32279325  0.28516266  0.0464602 ]
 [-0.30120417  0.56390882  0.21822248  0.34746924  0.08855192  0.07925911]
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
{'loss': 0.4797929525375366, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 15:32:27.400459: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.46730610355734825, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 15:32:28.613755: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 2031616/17464789 [==>...........................] - ETA: 0s
 5783552/17464789 [========>.....................] - ETA: 0s
 9707520/17464789 [===============>..............] - ETA: 0s
13533184/17464789 [======================>.......] - ETA: 0s
17301504/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 15:32:40.750014: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 15:32:40.753650: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 15:32:40.753782: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559a9403df40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 15:32:40.753797: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:49 - loss: 10.0624 - accuracy: 0.3438
   64/25000 [..............................] - ETA: 3:06 - loss: 7.9062 - accuracy: 0.4844 
   96/25000 [..............................] - ETA: 2:33 - loss: 7.5069 - accuracy: 0.5104
  128/25000 [..............................] - ETA: 2:17 - loss: 7.6666 - accuracy: 0.5000
  160/25000 [..............................] - ETA: 2:08 - loss: 7.4750 - accuracy: 0.5125
  192/25000 [..............................] - ETA: 2:01 - loss: 7.4270 - accuracy: 0.5156
  224/25000 [..............................] - ETA: 1:56 - loss: 7.3928 - accuracy: 0.5179
  256/25000 [..............................] - ETA: 1:52 - loss: 7.4270 - accuracy: 0.5156
  288/25000 [..............................] - ETA: 1:49 - loss: 7.5601 - accuracy: 0.5069
  320/25000 [..............................] - ETA: 1:47 - loss: 7.5708 - accuracy: 0.5063
  352/25000 [..............................] - ETA: 1:45 - loss: 7.2746 - accuracy: 0.5256
  384/25000 [..............................] - ETA: 1:43 - loss: 7.3072 - accuracy: 0.5234
  416/25000 [..............................] - ETA: 1:42 - loss: 7.5192 - accuracy: 0.5096
  448/25000 [..............................] - ETA: 1:41 - loss: 7.5982 - accuracy: 0.5045
  480/25000 [..............................] - ETA: 1:40 - loss: 7.4430 - accuracy: 0.5146
  512/25000 [..............................] - ETA: 1:39 - loss: 7.5169 - accuracy: 0.5098
  544/25000 [..............................] - ETA: 1:38 - loss: 7.4975 - accuracy: 0.5110
  576/25000 [..............................] - ETA: 1:37 - loss: 7.4537 - accuracy: 0.5139
  608/25000 [..............................] - ETA: 1:37 - loss: 7.5405 - accuracy: 0.5082
  640/25000 [..............................] - ETA: 1:36 - loss: 7.5708 - accuracy: 0.5063
  672/25000 [..............................] - ETA: 1:35 - loss: 7.6438 - accuracy: 0.5015
  704/25000 [..............................] - ETA: 1:35 - loss: 7.7320 - accuracy: 0.4957
  736/25000 [..............................] - ETA: 1:34 - loss: 7.6875 - accuracy: 0.4986
  768/25000 [..............................] - ETA: 1:34 - loss: 7.6267 - accuracy: 0.5026
  800/25000 [..............................] - ETA: 1:34 - loss: 7.6858 - accuracy: 0.4988
  832/25000 [..............................] - ETA: 1:33 - loss: 7.7403 - accuracy: 0.4952
  864/25000 [>.............................] - ETA: 1:33 - loss: 7.7731 - accuracy: 0.4931
  896/25000 [>.............................] - ETA: 1:33 - loss: 7.7180 - accuracy: 0.4967
  928/25000 [>.............................] - ETA: 1:32 - loss: 7.6831 - accuracy: 0.4989
  960/25000 [>.............................] - ETA: 1:32 - loss: 7.6666 - accuracy: 0.5000
  992/25000 [>.............................] - ETA: 1:32 - loss: 7.6512 - accuracy: 0.5010
 1024/25000 [>.............................] - ETA: 1:31 - loss: 7.5918 - accuracy: 0.5049
 1056/25000 [>.............................] - ETA: 1:31 - loss: 7.6231 - accuracy: 0.5028
 1088/25000 [>.............................] - ETA: 1:31 - loss: 7.6666 - accuracy: 0.5000
 1120/25000 [>.............................] - ETA: 1:31 - loss: 7.6666 - accuracy: 0.5000
 1152/25000 [>.............................] - ETA: 1:31 - loss: 7.7199 - accuracy: 0.4965
 1184/25000 [>.............................] - ETA: 1:31 - loss: 7.7055 - accuracy: 0.4975
 1216/25000 [>.............................] - ETA: 1:30 - loss: 7.6540 - accuracy: 0.5008
 1248/25000 [>.............................] - ETA: 1:30 - loss: 7.6298 - accuracy: 0.5024
 1280/25000 [>.............................] - ETA: 1:30 - loss: 7.6666 - accuracy: 0.5000
 1312/25000 [>.............................] - ETA: 1:30 - loss: 7.7017 - accuracy: 0.4977
 1344/25000 [>.............................] - ETA: 1:29 - loss: 7.7123 - accuracy: 0.4970
 1376/25000 [>.............................] - ETA: 1:29 - loss: 7.7000 - accuracy: 0.4978
 1408/25000 [>.............................] - ETA: 1:29 - loss: 7.6666 - accuracy: 0.5000
 1440/25000 [>.............................] - ETA: 1:29 - loss: 7.7092 - accuracy: 0.4972
 1472/25000 [>.............................] - ETA: 1:29 - loss: 7.7604 - accuracy: 0.4939
 1504/25000 [>.............................] - ETA: 1:28 - loss: 7.6972 - accuracy: 0.4980
 1536/25000 [>.............................] - ETA: 1:28 - loss: 7.6666 - accuracy: 0.5000
 1568/25000 [>.............................] - ETA: 1:28 - loss: 7.7155 - accuracy: 0.4968
 1600/25000 [>.............................] - ETA: 1:28 - loss: 7.7145 - accuracy: 0.4969
 1632/25000 [>.............................] - ETA: 1:28 - loss: 7.6948 - accuracy: 0.4982
 1664/25000 [>.............................] - ETA: 1:28 - loss: 7.7403 - accuracy: 0.4952
 1696/25000 [=>............................] - ETA: 1:28 - loss: 7.7209 - accuracy: 0.4965
 1728/25000 [=>............................] - ETA: 1:27 - loss: 7.7021 - accuracy: 0.4977
 1760/25000 [=>............................] - ETA: 1:27 - loss: 7.6928 - accuracy: 0.4983
 1792/25000 [=>............................] - ETA: 1:27 - loss: 7.6581 - accuracy: 0.5006
 1824/25000 [=>............................] - ETA: 1:27 - loss: 7.6666 - accuracy: 0.5000
 1856/25000 [=>............................] - ETA: 1:26 - loss: 7.6666 - accuracy: 0.5000
 1888/25000 [=>............................] - ETA: 1:26 - loss: 7.6423 - accuracy: 0.5016
 1920/25000 [=>............................] - ETA: 1:26 - loss: 7.6267 - accuracy: 0.5026
 1952/25000 [=>............................] - ETA: 1:26 - loss: 7.6431 - accuracy: 0.5015
 1984/25000 [=>............................] - ETA: 1:26 - loss: 7.6357 - accuracy: 0.5020
 2016/25000 [=>............................] - ETA: 1:26 - loss: 7.6666 - accuracy: 0.5000
 2048/25000 [=>............................] - ETA: 1:26 - loss: 7.6591 - accuracy: 0.5005
 2080/25000 [=>............................] - ETA: 1:25 - loss: 7.6519 - accuracy: 0.5010
 2112/25000 [=>............................] - ETA: 1:25 - loss: 7.6594 - accuracy: 0.5005
 2144/25000 [=>............................] - ETA: 1:25 - loss: 7.6523 - accuracy: 0.5009
 2176/25000 [=>............................] - ETA: 1:25 - loss: 7.6666 - accuracy: 0.5000
 2208/25000 [=>............................] - ETA: 1:25 - loss: 7.6875 - accuracy: 0.4986
 2240/25000 [=>............................] - ETA: 1:25 - loss: 7.6735 - accuracy: 0.4996
 2272/25000 [=>............................] - ETA: 1:25 - loss: 7.6734 - accuracy: 0.4996
 2304/25000 [=>............................] - ETA: 1:24 - loss: 7.6666 - accuracy: 0.5000
 2336/25000 [=>............................] - ETA: 1:24 - loss: 7.6797 - accuracy: 0.4991
 2368/25000 [=>............................] - ETA: 1:24 - loss: 7.6731 - accuracy: 0.4996
 2400/25000 [=>............................] - ETA: 1:24 - loss: 7.6538 - accuracy: 0.5008
 2432/25000 [=>............................] - ETA: 1:24 - loss: 7.6603 - accuracy: 0.5004
 2464/25000 [=>............................] - ETA: 1:24 - loss: 7.6542 - accuracy: 0.5008
 2496/25000 [=>............................] - ETA: 1:23 - loss: 7.6666 - accuracy: 0.5000
 2528/25000 [==>...........................] - ETA: 1:23 - loss: 7.6909 - accuracy: 0.4984
 2560/25000 [==>...........................] - ETA: 1:23 - loss: 7.6786 - accuracy: 0.4992
 2592/25000 [==>...........................] - ETA: 1:23 - loss: 7.6489 - accuracy: 0.5012
 2624/25000 [==>...........................] - ETA: 1:23 - loss: 7.6549 - accuracy: 0.5008
 2656/25000 [==>...........................] - ETA: 1:23 - loss: 7.6378 - accuracy: 0.5019
 2688/25000 [==>...........................] - ETA: 1:23 - loss: 7.6324 - accuracy: 0.5022
 2720/25000 [==>...........................] - ETA: 1:23 - loss: 7.6441 - accuracy: 0.5015
 2752/25000 [==>...........................] - ETA: 1:22 - loss: 7.6220 - accuracy: 0.5029
 2784/25000 [==>...........................] - ETA: 1:22 - loss: 7.6171 - accuracy: 0.5032
 2816/25000 [==>...........................] - ETA: 1:22 - loss: 7.6176 - accuracy: 0.5032
 2848/25000 [==>...........................] - ETA: 1:22 - loss: 7.6182 - accuracy: 0.5032
 2880/25000 [==>...........................] - ETA: 1:22 - loss: 7.5974 - accuracy: 0.5045
 2912/25000 [==>...........................] - ETA: 1:22 - loss: 7.5666 - accuracy: 0.5065
 2944/25000 [==>...........................] - ETA: 1:21 - loss: 7.5520 - accuracy: 0.5075
 2976/25000 [==>...........................] - ETA: 1:21 - loss: 7.5584 - accuracy: 0.5071
 3008/25000 [==>...........................] - ETA: 1:21 - loss: 7.5443 - accuracy: 0.5080
 3040/25000 [==>...........................] - ETA: 1:21 - loss: 7.5607 - accuracy: 0.5069
 3072/25000 [==>...........................] - ETA: 1:21 - loss: 7.5618 - accuracy: 0.5068
 3104/25000 [==>...........................] - ETA: 1:21 - loss: 7.5925 - accuracy: 0.5048
 3136/25000 [==>...........................] - ETA: 1:21 - loss: 7.5786 - accuracy: 0.5057
 3168/25000 [==>...........................] - ETA: 1:21 - loss: 7.6037 - accuracy: 0.5041
 3200/25000 [==>...........................] - ETA: 1:21 - loss: 7.5947 - accuracy: 0.5047
 3232/25000 [==>...........................] - ETA: 1:20 - loss: 7.5907 - accuracy: 0.5050
 3264/25000 [==>...........................] - ETA: 1:20 - loss: 7.6055 - accuracy: 0.5040
 3296/25000 [==>...........................] - ETA: 1:20 - loss: 7.6061 - accuracy: 0.5039
 3328/25000 [==>...........................] - ETA: 1:20 - loss: 7.5975 - accuracy: 0.5045
 3360/25000 [===>..........................] - ETA: 1:20 - loss: 7.6027 - accuracy: 0.5042
 3392/25000 [===>..........................] - ETA: 1:20 - loss: 7.6124 - accuracy: 0.5035
 3424/25000 [===>..........................] - ETA: 1:20 - loss: 7.5905 - accuracy: 0.5050
 3456/25000 [===>..........................] - ETA: 1:20 - loss: 7.5912 - accuracy: 0.5049
 3488/25000 [===>..........................] - ETA: 1:20 - loss: 7.5919 - accuracy: 0.5049
 3520/25000 [===>..........................] - ETA: 1:19 - loss: 7.5795 - accuracy: 0.5057
 3552/25000 [===>..........................] - ETA: 1:19 - loss: 7.5760 - accuracy: 0.5059
 3584/25000 [===>..........................] - ETA: 1:19 - loss: 7.5982 - accuracy: 0.5045
 3616/25000 [===>..........................] - ETA: 1:19 - loss: 7.5861 - accuracy: 0.5053
 3648/25000 [===>..........................] - ETA: 1:19 - loss: 7.6036 - accuracy: 0.5041
 3680/25000 [===>..........................] - ETA: 1:19 - loss: 7.6166 - accuracy: 0.5033
 3712/25000 [===>..........................] - ETA: 1:18 - loss: 7.6171 - accuracy: 0.5032
 3744/25000 [===>..........................] - ETA: 1:18 - loss: 7.6134 - accuracy: 0.5035
 3776/25000 [===>..........................] - ETA: 1:18 - loss: 7.6098 - accuracy: 0.5037
 3808/25000 [===>..........................] - ETA: 1:18 - loss: 7.6183 - accuracy: 0.5032
 3840/25000 [===>..........................] - ETA: 1:18 - loss: 7.6147 - accuracy: 0.5034
 3872/25000 [===>..........................] - ETA: 1:18 - loss: 7.6072 - accuracy: 0.5039
 3904/25000 [===>..........................] - ETA: 1:17 - loss: 7.6195 - accuracy: 0.5031
 3936/25000 [===>..........................] - ETA: 1:17 - loss: 7.6394 - accuracy: 0.5018
 3968/25000 [===>..........................] - ETA: 1:17 - loss: 7.6318 - accuracy: 0.5023
 4000/25000 [===>..........................] - ETA: 1:17 - loss: 7.6321 - accuracy: 0.5023
 4032/25000 [===>..........................] - ETA: 1:17 - loss: 7.6400 - accuracy: 0.5017
 4064/25000 [===>..........................] - ETA: 1:17 - loss: 7.6251 - accuracy: 0.5027
 4096/25000 [===>..........................] - ETA: 1:16 - loss: 7.6329 - accuracy: 0.5022
 4128/25000 [===>..........................] - ETA: 1:16 - loss: 7.6369 - accuracy: 0.5019
 4160/25000 [===>..........................] - ETA: 1:16 - loss: 7.6298 - accuracy: 0.5024
 4192/25000 [====>.........................] - ETA: 1:16 - loss: 7.6337 - accuracy: 0.5021
 4224/25000 [====>.........................] - ETA: 1:16 - loss: 7.6376 - accuracy: 0.5019
 4256/25000 [====>.........................] - ETA: 1:16 - loss: 7.6522 - accuracy: 0.5009
 4288/25000 [====>.........................] - ETA: 1:16 - loss: 7.6559 - accuracy: 0.5007
 4320/25000 [====>.........................] - ETA: 1:16 - loss: 7.6595 - accuracy: 0.5005
 4352/25000 [====>.........................] - ETA: 1:15 - loss: 7.6560 - accuracy: 0.5007
 4384/25000 [====>.........................] - ETA: 1:15 - loss: 7.6596 - accuracy: 0.5005
 4416/25000 [====>.........................] - ETA: 1:15 - loss: 7.6631 - accuracy: 0.5002
 4448/25000 [====>.........................] - ETA: 1:15 - loss: 7.6632 - accuracy: 0.5002
 4480/25000 [====>.........................] - ETA: 1:15 - loss: 7.6564 - accuracy: 0.5007
 4512/25000 [====>.........................] - ETA: 1:15 - loss: 7.6666 - accuracy: 0.5000
 4544/25000 [====>.........................] - ETA: 1:15 - loss: 7.6666 - accuracy: 0.5000
 4576/25000 [====>.........................] - ETA: 1:14 - loss: 7.6566 - accuracy: 0.5007
 4608/25000 [====>.........................] - ETA: 1:14 - loss: 7.6566 - accuracy: 0.5007
 4640/25000 [====>.........................] - ETA: 1:14 - loss: 7.6534 - accuracy: 0.5009
 4672/25000 [====>.........................] - ETA: 1:14 - loss: 7.6699 - accuracy: 0.4998
 4704/25000 [====>.........................] - ETA: 1:14 - loss: 7.6666 - accuracy: 0.5000
 4736/25000 [====>.........................] - ETA: 1:14 - loss: 7.6569 - accuracy: 0.5006
 4768/25000 [====>.........................] - ETA: 1:14 - loss: 7.6731 - accuracy: 0.4996
 4800/25000 [====>.........................] - ETA: 1:14 - loss: 7.6762 - accuracy: 0.4994
 4832/25000 [====>.........................] - ETA: 1:13 - loss: 7.6793 - accuracy: 0.4992
 4864/25000 [====>.........................] - ETA: 1:13 - loss: 7.6887 - accuracy: 0.4986
 4896/25000 [====>.........................] - ETA: 1:13 - loss: 7.7073 - accuracy: 0.4973
 4928/25000 [====>.........................] - ETA: 1:13 - loss: 7.7071 - accuracy: 0.4974
 4960/25000 [====>.........................] - ETA: 1:13 - loss: 7.7099 - accuracy: 0.4972
 4992/25000 [====>.........................] - ETA: 1:13 - loss: 7.7096 - accuracy: 0.4972
 5024/25000 [=====>........................] - ETA: 1:13 - loss: 7.7155 - accuracy: 0.4968
 5056/25000 [=====>........................] - ETA: 1:13 - loss: 7.7121 - accuracy: 0.4970
 5088/25000 [=====>........................] - ETA: 1:13 - loss: 7.7028 - accuracy: 0.4976
 5120/25000 [=====>........................] - ETA: 1:12 - loss: 7.7115 - accuracy: 0.4971
 5152/25000 [=====>........................] - ETA: 1:12 - loss: 7.7202 - accuracy: 0.4965
 5184/25000 [=====>........................] - ETA: 1:12 - loss: 7.7139 - accuracy: 0.4969
 5216/25000 [=====>........................] - ETA: 1:12 - loss: 7.7254 - accuracy: 0.4962
 5248/25000 [=====>........................] - ETA: 1:12 - loss: 7.7221 - accuracy: 0.4964
 5280/25000 [=====>........................] - ETA: 1:12 - loss: 7.7102 - accuracy: 0.4972
 5312/25000 [=====>........................] - ETA: 1:12 - loss: 7.7099 - accuracy: 0.4972
 5344/25000 [=====>........................] - ETA: 1:11 - loss: 7.7097 - accuracy: 0.4972
 5376/25000 [=====>........................] - ETA: 1:11 - loss: 7.7065 - accuracy: 0.4974
 5408/25000 [=====>........................] - ETA: 1:11 - loss: 7.7063 - accuracy: 0.4974
 5440/25000 [=====>........................] - ETA: 1:11 - loss: 7.7145 - accuracy: 0.4969
 5472/25000 [=====>........................] - ETA: 1:11 - loss: 7.7283 - accuracy: 0.4960
 5504/25000 [=====>........................] - ETA: 1:11 - loss: 7.7335 - accuracy: 0.4956
 5536/25000 [=====>........................] - ETA: 1:11 - loss: 7.7359 - accuracy: 0.4955
 5568/25000 [=====>........................] - ETA: 1:11 - loss: 7.7355 - accuracy: 0.4955
 5600/25000 [=====>........................] - ETA: 1:11 - loss: 7.7296 - accuracy: 0.4959
 5632/25000 [=====>........................] - ETA: 1:10 - loss: 7.7292 - accuracy: 0.4959
 5664/25000 [=====>........................] - ETA: 1:10 - loss: 7.7316 - accuracy: 0.4958
 5696/25000 [=====>........................] - ETA: 1:10 - loss: 7.7178 - accuracy: 0.4967
 5728/25000 [=====>........................] - ETA: 1:10 - loss: 7.7255 - accuracy: 0.4962
 5760/25000 [=====>........................] - ETA: 1:10 - loss: 7.7252 - accuracy: 0.4962
 5792/25000 [=====>........................] - ETA: 1:10 - loss: 7.7302 - accuracy: 0.4959
 5824/25000 [=====>........................] - ETA: 1:10 - loss: 7.7219 - accuracy: 0.4964
 5856/25000 [======>.......................] - ETA: 1:10 - loss: 7.7216 - accuracy: 0.4964
 5888/25000 [======>.......................] - ETA: 1:09 - loss: 7.7265 - accuracy: 0.4961
 5920/25000 [======>.......................] - ETA: 1:09 - loss: 7.7262 - accuracy: 0.4961
 5952/25000 [======>.......................] - ETA: 1:09 - loss: 7.7233 - accuracy: 0.4963
 5984/25000 [======>.......................] - ETA: 1:09 - loss: 7.7409 - accuracy: 0.4952
 6016/25000 [======>.......................] - ETA: 1:09 - loss: 7.7380 - accuracy: 0.4953
 6048/25000 [======>.......................] - ETA: 1:09 - loss: 7.7351 - accuracy: 0.4955
 6080/25000 [======>.......................] - ETA: 1:09 - loss: 7.7423 - accuracy: 0.4951
 6112/25000 [======>.......................] - ETA: 1:09 - loss: 7.7469 - accuracy: 0.4948
 6144/25000 [======>.......................] - ETA: 1:08 - loss: 7.7440 - accuracy: 0.4950
 6176/25000 [======>.......................] - ETA: 1:08 - loss: 7.7411 - accuracy: 0.4951
 6208/25000 [======>.......................] - ETA: 1:08 - loss: 7.7358 - accuracy: 0.4955
 6240/25000 [======>.......................] - ETA: 1:08 - loss: 7.7379 - accuracy: 0.4954
 6272/25000 [======>.......................] - ETA: 1:08 - loss: 7.7375 - accuracy: 0.4954
 6304/25000 [======>.......................] - ETA: 1:08 - loss: 7.7445 - accuracy: 0.4949
 6336/25000 [======>.......................] - ETA: 1:08 - loss: 7.7489 - accuracy: 0.4946
 6368/25000 [======>.......................] - ETA: 1:08 - loss: 7.7437 - accuracy: 0.4950
 6400/25000 [======>.......................] - ETA: 1:07 - loss: 7.7433 - accuracy: 0.4950
 6432/25000 [======>.......................] - ETA: 1:07 - loss: 7.7405 - accuracy: 0.4952
 6464/25000 [======>.......................] - ETA: 1:07 - loss: 7.7449 - accuracy: 0.4949
 6496/25000 [======>.......................] - ETA: 1:07 - loss: 7.7422 - accuracy: 0.4951
 6528/25000 [======>.......................] - ETA: 1:07 - loss: 7.7535 - accuracy: 0.4943
 6560/25000 [======>.......................] - ETA: 1:07 - loss: 7.7508 - accuracy: 0.4945
 6592/25000 [======>.......................] - ETA: 1:07 - loss: 7.7527 - accuracy: 0.4944
 6624/25000 [======>.......................] - ETA: 1:07 - loss: 7.7662 - accuracy: 0.4935
 6656/25000 [======>.......................] - ETA: 1:07 - loss: 7.7749 - accuracy: 0.4929
 6688/25000 [=======>......................] - ETA: 1:06 - loss: 7.7835 - accuracy: 0.4924
 6720/25000 [=======>......................] - ETA: 1:06 - loss: 7.7784 - accuracy: 0.4927
 6752/25000 [=======>......................] - ETA: 1:06 - loss: 7.7892 - accuracy: 0.4920
 6784/25000 [=======>......................] - ETA: 1:06 - loss: 7.7932 - accuracy: 0.4917
 6816/25000 [=======>......................] - ETA: 1:06 - loss: 7.7948 - accuracy: 0.4916
 6848/25000 [=======>......................] - ETA: 1:06 - loss: 7.8010 - accuracy: 0.4912
 6880/25000 [=======>......................] - ETA: 1:06 - loss: 7.7981 - accuracy: 0.4914
 6912/25000 [=======>......................] - ETA: 1:06 - loss: 7.7997 - accuracy: 0.4913
 6944/25000 [=======>......................] - ETA: 1:05 - loss: 7.8079 - accuracy: 0.4908
 6976/25000 [=======>......................] - ETA: 1:05 - loss: 7.8117 - accuracy: 0.4905
 7008/25000 [=======>......................] - ETA: 1:05 - loss: 7.8023 - accuracy: 0.4912
 7040/25000 [=======>......................] - ETA: 1:05 - loss: 7.7886 - accuracy: 0.4920
 7072/25000 [=======>......................] - ETA: 1:05 - loss: 7.7924 - accuracy: 0.4918
 7104/25000 [=======>......................] - ETA: 1:05 - loss: 7.7918 - accuracy: 0.4918
 7136/25000 [=======>......................] - ETA: 1:05 - loss: 7.7934 - accuracy: 0.4917
 7168/25000 [=======>......................] - ETA: 1:05 - loss: 7.7971 - accuracy: 0.4915
 7200/25000 [=======>......................] - ETA: 1:05 - loss: 7.7965 - accuracy: 0.4915
 7232/25000 [=======>......................] - ETA: 1:04 - loss: 7.7960 - accuracy: 0.4916
 7264/25000 [=======>......................] - ETA: 1:04 - loss: 7.7954 - accuracy: 0.4916
 7296/25000 [=======>......................] - ETA: 1:04 - loss: 7.7927 - accuracy: 0.4918
 7328/25000 [=======>......................] - ETA: 1:04 - loss: 7.7963 - accuracy: 0.4915
 7360/25000 [=======>......................] - ETA: 1:04 - loss: 7.7958 - accuracy: 0.4916
 7392/25000 [=======>......................] - ETA: 1:04 - loss: 7.7973 - accuracy: 0.4915
 7424/25000 [=======>......................] - ETA: 1:04 - loss: 7.7947 - accuracy: 0.4916
 7456/25000 [=======>......................] - ETA: 1:04 - loss: 7.7962 - accuracy: 0.4916
 7488/25000 [=======>......................] - ETA: 1:03 - loss: 7.8018 - accuracy: 0.4912
 7520/25000 [========>.....................] - ETA: 1:03 - loss: 7.7971 - accuracy: 0.4915
 7552/25000 [========>.....................] - ETA: 1:03 - loss: 7.8027 - accuracy: 0.4911
 7584/25000 [========>.....................] - ETA: 1:03 - loss: 7.8102 - accuracy: 0.4906
 7616/25000 [========>.....................] - ETA: 1:03 - loss: 7.8076 - accuracy: 0.4908
 7648/25000 [========>.....................] - ETA: 1:03 - loss: 7.8030 - accuracy: 0.4911
 7680/25000 [========>.....................] - ETA: 1:03 - loss: 7.8084 - accuracy: 0.4908
 7712/25000 [========>.....................] - ETA: 1:03 - loss: 7.8078 - accuracy: 0.4908
 7744/25000 [========>.....................] - ETA: 1:02 - loss: 7.8112 - accuracy: 0.4906
 7776/25000 [========>.....................] - ETA: 1:02 - loss: 7.7968 - accuracy: 0.4915
 7808/25000 [========>.....................] - ETA: 1:02 - loss: 7.8021 - accuracy: 0.4912
 7840/25000 [========>.....................] - ETA: 1:02 - loss: 7.8074 - accuracy: 0.4908
 7872/25000 [========>.....................] - ETA: 1:02 - loss: 7.8108 - accuracy: 0.4906
 7904/25000 [========>.....................] - ETA: 1:02 - loss: 7.8082 - accuracy: 0.4908
 7936/25000 [========>.....................] - ETA: 1:02 - loss: 7.8057 - accuracy: 0.4909
 7968/25000 [========>.....................] - ETA: 1:02 - loss: 7.8090 - accuracy: 0.4907
 8000/25000 [========>.....................] - ETA: 1:02 - loss: 7.8142 - accuracy: 0.4904
 8032/25000 [========>.....................] - ETA: 1:01 - loss: 7.8193 - accuracy: 0.4900
 8064/25000 [========>.....................] - ETA: 1:01 - loss: 7.8168 - accuracy: 0.4902
 8096/25000 [========>.....................] - ETA: 1:01 - loss: 7.8238 - accuracy: 0.4897
 8128/25000 [========>.....................] - ETA: 1:01 - loss: 7.8194 - accuracy: 0.4900
 8160/25000 [========>.....................] - ETA: 1:01 - loss: 7.8169 - accuracy: 0.4902
 8192/25000 [========>.....................] - ETA: 1:01 - loss: 7.8238 - accuracy: 0.4897
 8224/25000 [========>.....................] - ETA: 1:01 - loss: 7.8232 - accuracy: 0.4898
 8256/25000 [========>.....................] - ETA: 1:01 - loss: 7.8282 - accuracy: 0.4895
 8288/25000 [========>.....................] - ETA: 1:01 - loss: 7.8220 - accuracy: 0.4899
 8320/25000 [========>.....................] - ETA: 1:00 - loss: 7.8177 - accuracy: 0.4901
 8352/25000 [=========>....................] - ETA: 1:00 - loss: 7.8227 - accuracy: 0.4898
 8384/25000 [=========>....................] - ETA: 1:00 - loss: 7.8202 - accuracy: 0.4900
 8416/25000 [=========>....................] - ETA: 1:00 - loss: 7.8160 - accuracy: 0.4903
 8448/25000 [=========>....................] - ETA: 1:00 - loss: 7.8173 - accuracy: 0.4902
 8480/25000 [=========>....................] - ETA: 1:00 - loss: 7.8149 - accuracy: 0.4903
 8512/25000 [=========>....................] - ETA: 1:00 - loss: 7.8071 - accuracy: 0.4908
 8544/25000 [=========>....................] - ETA: 1:00 - loss: 7.8120 - accuracy: 0.4905
 8576/25000 [=========>....................] - ETA: 1:00 - loss: 7.8168 - accuracy: 0.4902
 8608/25000 [=========>....................] - ETA: 59s - loss: 7.8162 - accuracy: 0.4902 
 8640/25000 [=========>....................] - ETA: 59s - loss: 7.8210 - accuracy: 0.4899
 8672/25000 [=========>....................] - ETA: 59s - loss: 7.8169 - accuracy: 0.4902
 8704/25000 [=========>....................] - ETA: 59s - loss: 7.8093 - accuracy: 0.4907
 8736/25000 [=========>....................] - ETA: 59s - loss: 7.8158 - accuracy: 0.4903
 8768/25000 [=========>....................] - ETA: 59s - loss: 7.8188 - accuracy: 0.4901
 8800/25000 [=========>....................] - ETA: 59s - loss: 7.8182 - accuracy: 0.4901
 8832/25000 [=========>....................] - ETA: 59s - loss: 7.8211 - accuracy: 0.4899
 8864/25000 [=========>....................] - ETA: 58s - loss: 7.8223 - accuracy: 0.4898
 8896/25000 [=========>....................] - ETA: 58s - loss: 7.8166 - accuracy: 0.4902
 8928/25000 [=========>....................] - ETA: 58s - loss: 7.8143 - accuracy: 0.4904
 8960/25000 [=========>....................] - ETA: 58s - loss: 7.8172 - accuracy: 0.4902
 8992/25000 [=========>....................] - ETA: 58s - loss: 7.8116 - accuracy: 0.4905
 9024/25000 [=========>....................] - ETA: 58s - loss: 7.8026 - accuracy: 0.4911
 9056/25000 [=========>....................] - ETA: 58s - loss: 7.7970 - accuracy: 0.4915
 9088/25000 [=========>....................] - ETA: 58s - loss: 7.7982 - accuracy: 0.4914
 9120/25000 [=========>....................] - ETA: 58s - loss: 7.8028 - accuracy: 0.4911
 9152/25000 [=========>....................] - ETA: 57s - loss: 7.7939 - accuracy: 0.4917
 9184/25000 [==========>...................] - ETA: 57s - loss: 7.7952 - accuracy: 0.4916
 9216/25000 [==========>...................] - ETA: 57s - loss: 7.7997 - accuracy: 0.4913
 9248/25000 [==========>...................] - ETA: 57s - loss: 7.7993 - accuracy: 0.4913
 9280/25000 [==========>...................] - ETA: 57s - loss: 7.8005 - accuracy: 0.4913
 9312/25000 [==========>...................] - ETA: 57s - loss: 7.7983 - accuracy: 0.4914
 9344/25000 [==========>...................] - ETA: 57s - loss: 7.7963 - accuracy: 0.4915
 9376/25000 [==========>...................] - ETA: 57s - loss: 7.8007 - accuracy: 0.4913
 9408/25000 [==========>...................] - ETA: 56s - loss: 7.8052 - accuracy: 0.4910
 9440/25000 [==========>...................] - ETA: 56s - loss: 7.7998 - accuracy: 0.4913
 9472/25000 [==========>...................] - ETA: 56s - loss: 7.7994 - accuracy: 0.4913
 9504/25000 [==========>...................] - ETA: 56s - loss: 7.7941 - accuracy: 0.4917
 9536/25000 [==========>...................] - ETA: 56s - loss: 7.7904 - accuracy: 0.4919
 9568/25000 [==========>...................] - ETA: 56s - loss: 7.7868 - accuracy: 0.4922
 9600/25000 [==========>...................] - ETA: 56s - loss: 7.7912 - accuracy: 0.4919
 9632/25000 [==========>...................] - ETA: 56s - loss: 7.7940 - accuracy: 0.4917
 9664/25000 [==========>...................] - ETA: 56s - loss: 7.7983 - accuracy: 0.4914
 9696/25000 [==========>...................] - ETA: 55s - loss: 7.8010 - accuracy: 0.4912
 9728/25000 [==========>...................] - ETA: 55s - loss: 7.8069 - accuracy: 0.4909
 9760/25000 [==========>...................] - ETA: 55s - loss: 7.8033 - accuracy: 0.4911
 9792/25000 [==========>...................] - ETA: 55s - loss: 7.8013 - accuracy: 0.4912
 9824/25000 [==========>...................] - ETA: 55s - loss: 7.8055 - accuracy: 0.4909
 9856/25000 [==========>...................] - ETA: 55s - loss: 7.8020 - accuracy: 0.4912
 9888/25000 [==========>...................] - ETA: 55s - loss: 7.8015 - accuracy: 0.4912
 9920/25000 [==========>...................] - ETA: 55s - loss: 7.8011 - accuracy: 0.4912
 9952/25000 [==========>...................] - ETA: 54s - loss: 7.8037 - accuracy: 0.4911
 9984/25000 [==========>...................] - ETA: 54s - loss: 7.8048 - accuracy: 0.4910
10016/25000 [===========>..................] - ETA: 54s - loss: 7.8029 - accuracy: 0.4911
10048/25000 [===========>..................] - ETA: 54s - loss: 7.8009 - accuracy: 0.4912
10080/25000 [===========>..................] - ETA: 54s - loss: 7.7959 - accuracy: 0.4916
10112/25000 [===========>..................] - ETA: 54s - loss: 7.7955 - accuracy: 0.4916
10144/25000 [===========>..................] - ETA: 54s - loss: 7.7906 - accuracy: 0.4919
10176/25000 [===========>..................] - ETA: 54s - loss: 7.7887 - accuracy: 0.4920
10208/25000 [===========>..................] - ETA: 54s - loss: 7.7853 - accuracy: 0.4923
10240/25000 [===========>..................] - ETA: 53s - loss: 7.7849 - accuracy: 0.4923
10272/25000 [===========>..................] - ETA: 53s - loss: 7.7860 - accuracy: 0.4922
10304/25000 [===========>..................] - ETA: 53s - loss: 7.7812 - accuracy: 0.4925
10336/25000 [===========>..................] - ETA: 53s - loss: 7.7719 - accuracy: 0.4931
10368/25000 [===========>..................] - ETA: 53s - loss: 7.7716 - accuracy: 0.4932
10400/25000 [===========>..................] - ETA: 53s - loss: 7.7669 - accuracy: 0.4935
10432/25000 [===========>..................] - ETA: 53s - loss: 7.7710 - accuracy: 0.4932
10464/25000 [===========>..................] - ETA: 53s - loss: 7.7707 - accuracy: 0.4932
10496/25000 [===========>..................] - ETA: 52s - loss: 7.7689 - accuracy: 0.4933
10528/25000 [===========>..................] - ETA: 52s - loss: 7.7642 - accuracy: 0.4936
10560/25000 [===========>..................] - ETA: 52s - loss: 7.7683 - accuracy: 0.4934
10592/25000 [===========>..................] - ETA: 52s - loss: 7.7680 - accuracy: 0.4934
10624/25000 [===========>..................] - ETA: 52s - loss: 7.7734 - accuracy: 0.4930
10656/25000 [===========>..................] - ETA: 52s - loss: 7.7789 - accuracy: 0.4927
10688/25000 [===========>..................] - ETA: 52s - loss: 7.7785 - accuracy: 0.4927
10720/25000 [===========>..................] - ETA: 52s - loss: 7.7753 - accuracy: 0.4929
10752/25000 [===========>..................] - ETA: 52s - loss: 7.7679 - accuracy: 0.4934
10784/25000 [===========>..................] - ETA: 51s - loss: 7.7633 - accuracy: 0.4937
10816/25000 [===========>..................] - ETA: 51s - loss: 7.7573 - accuracy: 0.4941
10848/25000 [============>.................] - ETA: 51s - loss: 7.7543 - accuracy: 0.4943
10880/25000 [============>.................] - ETA: 51s - loss: 7.7526 - accuracy: 0.4944
10912/25000 [============>.................] - ETA: 51s - loss: 7.7453 - accuracy: 0.4949
10944/25000 [============>.................] - ETA: 51s - loss: 7.7451 - accuracy: 0.4949
10976/25000 [============>.................] - ETA: 51s - loss: 7.7421 - accuracy: 0.4951
11008/25000 [============>.................] - ETA: 51s - loss: 7.7391 - accuracy: 0.4953
11040/25000 [============>.................] - ETA: 50s - loss: 7.7402 - accuracy: 0.4952
11072/25000 [============>.................] - ETA: 50s - loss: 7.7442 - accuracy: 0.4949
11104/25000 [============>.................] - ETA: 50s - loss: 7.7439 - accuracy: 0.4950
11136/25000 [============>.................] - ETA: 50s - loss: 7.7423 - accuracy: 0.4951
11168/25000 [============>.................] - ETA: 50s - loss: 7.7463 - accuracy: 0.4948
11200/25000 [============>.................] - ETA: 50s - loss: 7.7515 - accuracy: 0.4945
11232/25000 [============>.................] - ETA: 50s - loss: 7.7513 - accuracy: 0.4945
11264/25000 [============>.................] - ETA: 50s - loss: 7.7469 - accuracy: 0.4948
11296/25000 [============>.................] - ETA: 49s - loss: 7.7467 - accuracy: 0.4948
11328/25000 [============>.................] - ETA: 49s - loss: 7.7438 - accuracy: 0.4950
11360/25000 [============>.................] - ETA: 49s - loss: 7.7436 - accuracy: 0.4950
11392/25000 [============>.................] - ETA: 49s - loss: 7.7447 - accuracy: 0.4949
11424/25000 [============>.................] - ETA: 49s - loss: 7.7391 - accuracy: 0.4953
11456/25000 [============>.................] - ETA: 49s - loss: 7.7402 - accuracy: 0.4952
11488/25000 [============>.................] - ETA: 49s - loss: 7.7374 - accuracy: 0.4954
11520/25000 [============>.................] - ETA: 49s - loss: 7.7425 - accuracy: 0.4951
11552/25000 [============>.................] - ETA: 49s - loss: 7.7356 - accuracy: 0.4955
11584/25000 [============>.................] - ETA: 48s - loss: 7.7354 - accuracy: 0.4955
11616/25000 [============>.................] - ETA: 48s - loss: 7.7366 - accuracy: 0.4954
11648/25000 [============>.................] - ETA: 48s - loss: 7.7338 - accuracy: 0.4956
11680/25000 [=============>................] - ETA: 48s - loss: 7.7388 - accuracy: 0.4953
11712/25000 [=============>................] - ETA: 48s - loss: 7.7386 - accuracy: 0.4953
11744/25000 [=============>................] - ETA: 48s - loss: 7.7332 - accuracy: 0.4957
11776/25000 [=============>................] - ETA: 48s - loss: 7.7356 - accuracy: 0.4955
11808/25000 [=============>................] - ETA: 48s - loss: 7.7354 - accuracy: 0.4955
11840/25000 [=============>................] - ETA: 47s - loss: 7.7288 - accuracy: 0.4959
11872/25000 [=============>................] - ETA: 47s - loss: 7.7286 - accuracy: 0.4960
11904/25000 [=============>................] - ETA: 47s - loss: 7.7272 - accuracy: 0.4961
11936/25000 [=============>................] - ETA: 47s - loss: 7.7231 - accuracy: 0.4963
11968/25000 [=============>................] - ETA: 47s - loss: 7.7204 - accuracy: 0.4965
12000/25000 [=============>................] - ETA: 47s - loss: 7.7228 - accuracy: 0.4963
12032/25000 [=============>................] - ETA: 47s - loss: 7.7240 - accuracy: 0.4963
12064/25000 [=============>................] - ETA: 47s - loss: 7.7225 - accuracy: 0.4964
12096/25000 [=============>................] - ETA: 46s - loss: 7.7249 - accuracy: 0.4962
12128/25000 [=============>................] - ETA: 46s - loss: 7.7260 - accuracy: 0.4961
12160/25000 [=============>................] - ETA: 46s - loss: 7.7246 - accuracy: 0.4962
12192/25000 [=============>................] - ETA: 46s - loss: 7.7257 - accuracy: 0.4961
12224/25000 [=============>................] - ETA: 46s - loss: 7.7231 - accuracy: 0.4963
12256/25000 [=============>................] - ETA: 46s - loss: 7.7192 - accuracy: 0.4966
12288/25000 [=============>................] - ETA: 46s - loss: 7.7153 - accuracy: 0.4968
12320/25000 [=============>................] - ETA: 46s - loss: 7.7189 - accuracy: 0.4966
12352/25000 [=============>................] - ETA: 46s - loss: 7.7212 - accuracy: 0.4964
12384/25000 [=============>................] - ETA: 45s - loss: 7.7236 - accuracy: 0.4963
12416/25000 [=============>................] - ETA: 45s - loss: 7.7197 - accuracy: 0.4965
12448/25000 [=============>................] - ETA: 45s - loss: 7.7159 - accuracy: 0.4968
12480/25000 [=============>................] - ETA: 45s - loss: 7.7170 - accuracy: 0.4967
12512/25000 [==============>...............] - ETA: 45s - loss: 7.7193 - accuracy: 0.4966
12544/25000 [==============>...............] - ETA: 45s - loss: 7.7216 - accuracy: 0.4964
12576/25000 [==============>...............] - ETA: 45s - loss: 7.7215 - accuracy: 0.4964
12608/25000 [==============>...............] - ETA: 45s - loss: 7.7226 - accuracy: 0.4964
12640/25000 [==============>...............] - ETA: 44s - loss: 7.7248 - accuracy: 0.4962
12672/25000 [==============>...............] - ETA: 44s - loss: 7.7199 - accuracy: 0.4965
12704/25000 [==============>...............] - ETA: 44s - loss: 7.7221 - accuracy: 0.4964
12736/25000 [==============>...............] - ETA: 44s - loss: 7.7232 - accuracy: 0.4963
12768/25000 [==============>...............] - ETA: 44s - loss: 7.7267 - accuracy: 0.4961
12800/25000 [==============>...............] - ETA: 44s - loss: 7.7241 - accuracy: 0.4963
12832/25000 [==============>...............] - ETA: 44s - loss: 7.7264 - accuracy: 0.4961
12864/25000 [==============>...............] - ETA: 44s - loss: 7.7262 - accuracy: 0.4961
12896/25000 [==============>...............] - ETA: 44s - loss: 7.7249 - accuracy: 0.4962
12928/25000 [==============>...............] - ETA: 43s - loss: 7.7259 - accuracy: 0.4961
12960/25000 [==============>...............] - ETA: 43s - loss: 7.7293 - accuracy: 0.4959
12992/25000 [==============>...............] - ETA: 43s - loss: 7.7304 - accuracy: 0.4958
13024/25000 [==============>...............] - ETA: 43s - loss: 7.7278 - accuracy: 0.4960
13056/25000 [==============>...............] - ETA: 43s - loss: 7.7253 - accuracy: 0.4962
13088/25000 [==============>...............] - ETA: 43s - loss: 7.7275 - accuracy: 0.4960
13120/25000 [==============>...............] - ETA: 43s - loss: 7.7251 - accuracy: 0.4962
13152/25000 [==============>...............] - ETA: 43s - loss: 7.7249 - accuracy: 0.4962
13184/25000 [==============>...............] - ETA: 42s - loss: 7.7271 - accuracy: 0.4961
13216/25000 [==============>...............] - ETA: 42s - loss: 7.7281 - accuracy: 0.4960
13248/25000 [==============>...............] - ETA: 42s - loss: 7.7233 - accuracy: 0.4963
13280/25000 [==============>...............] - ETA: 42s - loss: 7.7220 - accuracy: 0.4964
13312/25000 [==============>...............] - ETA: 42s - loss: 7.7219 - accuracy: 0.4964
13344/25000 [===============>..............] - ETA: 42s - loss: 7.7252 - accuracy: 0.4962
13376/25000 [===============>..............] - ETA: 42s - loss: 7.7159 - accuracy: 0.4968
13408/25000 [===============>..............] - ETA: 42s - loss: 7.7158 - accuracy: 0.4968
13440/25000 [===============>..............] - ETA: 42s - loss: 7.7168 - accuracy: 0.4967
13472/25000 [===============>..............] - ETA: 41s - loss: 7.7133 - accuracy: 0.4970
13504/25000 [===============>..............] - ETA: 41s - loss: 7.7177 - accuracy: 0.4967
13536/25000 [===============>..............] - ETA: 41s - loss: 7.7187 - accuracy: 0.4966
13568/25000 [===============>..............] - ETA: 41s - loss: 7.7186 - accuracy: 0.4966
13600/25000 [===============>..............] - ETA: 41s - loss: 7.7241 - accuracy: 0.4963
13632/25000 [===============>..............] - ETA: 41s - loss: 7.7217 - accuracy: 0.4964
13664/25000 [===============>..............] - ETA: 41s - loss: 7.7238 - accuracy: 0.4963
13696/25000 [===============>..............] - ETA: 41s - loss: 7.7248 - accuracy: 0.4962
13728/25000 [===============>..............] - ETA: 40s - loss: 7.7247 - accuracy: 0.4962
13760/25000 [===============>..............] - ETA: 40s - loss: 7.7212 - accuracy: 0.4964
13792/25000 [===============>..............] - ETA: 40s - loss: 7.7233 - accuracy: 0.4963
13824/25000 [===============>..............] - ETA: 40s - loss: 7.7243 - accuracy: 0.4962
13856/25000 [===============>..............] - ETA: 40s - loss: 7.7264 - accuracy: 0.4961
13888/25000 [===============>..............] - ETA: 40s - loss: 7.7251 - accuracy: 0.4962
13920/25000 [===============>..............] - ETA: 40s - loss: 7.7217 - accuracy: 0.4964
13952/25000 [===============>..............] - ETA: 40s - loss: 7.7216 - accuracy: 0.4964
13984/25000 [===============>..............] - ETA: 40s - loss: 7.7193 - accuracy: 0.4966
14016/25000 [===============>..............] - ETA: 39s - loss: 7.7213 - accuracy: 0.4964
14048/25000 [===============>..............] - ETA: 39s - loss: 7.7234 - accuracy: 0.4963
14080/25000 [===============>..............] - ETA: 39s - loss: 7.7298 - accuracy: 0.4959
14112/25000 [===============>..............] - ETA: 39s - loss: 7.7275 - accuracy: 0.4960
14144/25000 [===============>..............] - ETA: 39s - loss: 7.7284 - accuracy: 0.4960
14176/25000 [================>.............] - ETA: 39s - loss: 7.7294 - accuracy: 0.4959
14208/25000 [================>.............] - ETA: 39s - loss: 7.7314 - accuracy: 0.4958
14240/25000 [================>.............] - ETA: 39s - loss: 7.7269 - accuracy: 0.4961
14272/25000 [================>.............] - ETA: 39s - loss: 7.7279 - accuracy: 0.4960
14304/25000 [================>.............] - ETA: 38s - loss: 7.7224 - accuracy: 0.4964
14336/25000 [================>.............] - ETA: 38s - loss: 7.7222 - accuracy: 0.4964
14368/25000 [================>.............] - ETA: 38s - loss: 7.7200 - accuracy: 0.4965
14400/25000 [================>.............] - ETA: 38s - loss: 7.7177 - accuracy: 0.4967
14432/25000 [================>.............] - ETA: 38s - loss: 7.7134 - accuracy: 0.4970
14464/25000 [================>.............] - ETA: 38s - loss: 7.7069 - accuracy: 0.4974
14496/25000 [================>.............] - ETA: 38s - loss: 7.7036 - accuracy: 0.4976
14528/25000 [================>.............] - ETA: 38s - loss: 7.7036 - accuracy: 0.4976
14560/25000 [================>.............] - ETA: 37s - loss: 7.7056 - accuracy: 0.4975
14592/25000 [================>.............] - ETA: 37s - loss: 7.7044 - accuracy: 0.4975
14624/25000 [================>.............] - ETA: 37s - loss: 7.7033 - accuracy: 0.4976
14656/25000 [================>.............] - ETA: 37s - loss: 7.7001 - accuracy: 0.4978
14688/25000 [================>.............] - ETA: 37s - loss: 7.6990 - accuracy: 0.4979
14720/25000 [================>.............] - ETA: 37s - loss: 7.6979 - accuracy: 0.4980
14752/25000 [================>.............] - ETA: 37s - loss: 7.6947 - accuracy: 0.4982
14784/25000 [================>.............] - ETA: 37s - loss: 7.6936 - accuracy: 0.4982
14816/25000 [================>.............] - ETA: 37s - loss: 7.6925 - accuracy: 0.4983
14848/25000 [================>.............] - ETA: 36s - loss: 7.6904 - accuracy: 0.4985
14880/25000 [================>.............] - ETA: 36s - loss: 7.6893 - accuracy: 0.4985
14912/25000 [================>.............] - ETA: 36s - loss: 7.6851 - accuracy: 0.4988
14944/25000 [================>.............] - ETA: 36s - loss: 7.6882 - accuracy: 0.4986
14976/25000 [================>.............] - ETA: 36s - loss: 7.6789 - accuracy: 0.4992
15008/25000 [=================>............] - ETA: 36s - loss: 7.6779 - accuracy: 0.4993
15040/25000 [=================>............] - ETA: 36s - loss: 7.6758 - accuracy: 0.4994
15072/25000 [=================>............] - ETA: 36s - loss: 7.6727 - accuracy: 0.4996
15104/25000 [=================>............] - ETA: 35s - loss: 7.6717 - accuracy: 0.4997
15136/25000 [=================>............] - ETA: 35s - loss: 7.6727 - accuracy: 0.4996
15168/25000 [=================>............] - ETA: 35s - loss: 7.6727 - accuracy: 0.4996
15200/25000 [=================>............] - ETA: 35s - loss: 7.6686 - accuracy: 0.4999
15232/25000 [=================>............] - ETA: 35s - loss: 7.6717 - accuracy: 0.4997
15264/25000 [=================>............] - ETA: 35s - loss: 7.6747 - accuracy: 0.4995
15296/25000 [=================>............] - ETA: 35s - loss: 7.6746 - accuracy: 0.4995
15328/25000 [=================>............] - ETA: 35s - loss: 7.6736 - accuracy: 0.4995
15360/25000 [=================>............] - ETA: 35s - loss: 7.6686 - accuracy: 0.4999
15392/25000 [=================>............] - ETA: 34s - loss: 7.6676 - accuracy: 0.4999
15424/25000 [=================>............] - ETA: 34s - loss: 7.6636 - accuracy: 0.5002
15456/25000 [=================>............] - ETA: 34s - loss: 7.6646 - accuracy: 0.5001
15488/25000 [=================>............] - ETA: 34s - loss: 7.6627 - accuracy: 0.5003
15520/25000 [=================>............] - ETA: 34s - loss: 7.6587 - accuracy: 0.5005
15552/25000 [=================>............] - ETA: 34s - loss: 7.6607 - accuracy: 0.5004
15584/25000 [=================>............] - ETA: 34s - loss: 7.6548 - accuracy: 0.5008
15616/25000 [=================>............] - ETA: 34s - loss: 7.6519 - accuracy: 0.5010
15648/25000 [=================>............] - ETA: 33s - loss: 7.6500 - accuracy: 0.5011
15680/25000 [=================>............] - ETA: 33s - loss: 7.6549 - accuracy: 0.5008
15712/25000 [=================>............] - ETA: 33s - loss: 7.6520 - accuracy: 0.5010
15744/25000 [=================>............] - ETA: 33s - loss: 7.6501 - accuracy: 0.5011
15776/25000 [=================>............] - ETA: 33s - loss: 7.6501 - accuracy: 0.5011
15808/25000 [=================>............] - ETA: 33s - loss: 7.6530 - accuracy: 0.5009
15840/25000 [==================>...........] - ETA: 33s - loss: 7.6531 - accuracy: 0.5009
15872/25000 [==================>...........] - ETA: 33s - loss: 7.6531 - accuracy: 0.5009
15904/25000 [==================>...........] - ETA: 33s - loss: 7.6541 - accuracy: 0.5008
15936/25000 [==================>...........] - ETA: 32s - loss: 7.6560 - accuracy: 0.5007
15968/25000 [==================>...........] - ETA: 32s - loss: 7.6599 - accuracy: 0.5004
16000/25000 [==================>...........] - ETA: 32s - loss: 7.6628 - accuracy: 0.5002
16032/25000 [==================>...........] - ETA: 32s - loss: 7.6657 - accuracy: 0.5001
16064/25000 [==================>...........] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
16096/25000 [==================>...........] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
16128/25000 [==================>...........] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
16160/25000 [==================>...........] - ETA: 32s - loss: 7.6666 - accuracy: 0.5000
16192/25000 [==================>...........] - ETA: 31s - loss: 7.6676 - accuracy: 0.4999
16224/25000 [==================>...........] - ETA: 31s - loss: 7.6647 - accuracy: 0.5001
16256/25000 [==================>...........] - ETA: 31s - loss: 7.6638 - accuracy: 0.5002
16288/25000 [==================>...........] - ETA: 31s - loss: 7.6638 - accuracy: 0.5002
16320/25000 [==================>...........] - ETA: 31s - loss: 7.6610 - accuracy: 0.5004
16352/25000 [==================>...........] - ETA: 31s - loss: 7.6629 - accuracy: 0.5002
16384/25000 [==================>...........] - ETA: 31s - loss: 7.6629 - accuracy: 0.5002
16416/25000 [==================>...........] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
16448/25000 [==================>...........] - ETA: 31s - loss: 7.6638 - accuracy: 0.5002
16480/25000 [==================>...........] - ETA: 30s - loss: 7.6685 - accuracy: 0.4999
16512/25000 [==================>...........] - ETA: 30s - loss: 7.6685 - accuracy: 0.4999
16544/25000 [==================>...........] - ETA: 30s - loss: 7.6694 - accuracy: 0.4998
16576/25000 [==================>...........] - ETA: 30s - loss: 7.6703 - accuracy: 0.4998
16608/25000 [==================>...........] - ETA: 30s - loss: 7.6685 - accuracy: 0.4999
16640/25000 [==================>...........] - ETA: 30s - loss: 7.6611 - accuracy: 0.5004
16672/25000 [===================>..........] - ETA: 30s - loss: 7.6639 - accuracy: 0.5002
16704/25000 [===================>..........] - ETA: 30s - loss: 7.6648 - accuracy: 0.5001
16736/25000 [===================>..........] - ETA: 30s - loss: 7.6657 - accuracy: 0.5001
16768/25000 [===================>..........] - ETA: 29s - loss: 7.6694 - accuracy: 0.4998
16800/25000 [===================>..........] - ETA: 29s - loss: 7.6684 - accuracy: 0.4999
16832/25000 [===================>..........] - ETA: 29s - loss: 7.6703 - accuracy: 0.4998
16864/25000 [===================>..........] - ETA: 29s - loss: 7.6684 - accuracy: 0.4999
16896/25000 [===================>..........] - ETA: 29s - loss: 7.6684 - accuracy: 0.4999
16928/25000 [===================>..........] - ETA: 29s - loss: 7.6675 - accuracy: 0.4999
16960/25000 [===================>..........] - ETA: 29s - loss: 7.6684 - accuracy: 0.4999
16992/25000 [===================>..........] - ETA: 29s - loss: 7.6648 - accuracy: 0.5001
17024/25000 [===================>..........] - ETA: 28s - loss: 7.6684 - accuracy: 0.4999
17056/25000 [===================>..........] - ETA: 28s - loss: 7.6720 - accuracy: 0.4996
17088/25000 [===================>..........] - ETA: 28s - loss: 7.6738 - accuracy: 0.4995
17120/25000 [===================>..........] - ETA: 28s - loss: 7.6729 - accuracy: 0.4996
17152/25000 [===================>..........] - ETA: 28s - loss: 7.6756 - accuracy: 0.4994
17184/25000 [===================>..........] - ETA: 28s - loss: 7.6755 - accuracy: 0.4994
17216/25000 [===================>..........] - ETA: 28s - loss: 7.6720 - accuracy: 0.4997
17248/25000 [===================>..........] - ETA: 28s - loss: 7.6711 - accuracy: 0.4997
17280/25000 [===================>..........] - ETA: 28s - loss: 7.6719 - accuracy: 0.4997
17312/25000 [===================>..........] - ETA: 27s - loss: 7.6728 - accuracy: 0.4996
17344/25000 [===================>..........] - ETA: 27s - loss: 7.6728 - accuracy: 0.4996
17376/25000 [===================>..........] - ETA: 27s - loss: 7.6719 - accuracy: 0.4997
17408/25000 [===================>..........] - ETA: 27s - loss: 7.6728 - accuracy: 0.4996
17440/25000 [===================>..........] - ETA: 27s - loss: 7.6710 - accuracy: 0.4997
17472/25000 [===================>..........] - ETA: 27s - loss: 7.6701 - accuracy: 0.4998
17504/25000 [====================>.........] - ETA: 27s - loss: 7.6692 - accuracy: 0.4998
17536/25000 [====================>.........] - ETA: 27s - loss: 7.6684 - accuracy: 0.4999
17568/25000 [====================>.........] - ETA: 26s - loss: 7.6701 - accuracy: 0.4998
17600/25000 [====================>.........] - ETA: 26s - loss: 7.6692 - accuracy: 0.4998
17632/25000 [====================>.........] - ETA: 26s - loss: 7.6701 - accuracy: 0.4998
17664/25000 [====================>.........] - ETA: 26s - loss: 7.6701 - accuracy: 0.4998
17696/25000 [====================>.........] - ETA: 26s - loss: 7.6684 - accuracy: 0.4999
17728/25000 [====================>.........] - ETA: 26s - loss: 7.6675 - accuracy: 0.4999
17760/25000 [====================>.........] - ETA: 26s - loss: 7.6649 - accuracy: 0.5001
17792/25000 [====================>.........] - ETA: 26s - loss: 7.6614 - accuracy: 0.5003
17824/25000 [====================>.........] - ETA: 26s - loss: 7.6632 - accuracy: 0.5002
17856/25000 [====================>.........] - ETA: 25s - loss: 7.6623 - accuracy: 0.5003
17888/25000 [====================>.........] - ETA: 25s - loss: 7.6623 - accuracy: 0.5003
17920/25000 [====================>.........] - ETA: 25s - loss: 7.6658 - accuracy: 0.5001
17952/25000 [====================>.........] - ETA: 25s - loss: 7.6675 - accuracy: 0.4999
17984/25000 [====================>.........] - ETA: 25s - loss: 7.6683 - accuracy: 0.4999
18016/25000 [====================>.........] - ETA: 25s - loss: 7.6683 - accuracy: 0.4999
18048/25000 [====================>.........] - ETA: 25s - loss: 7.6692 - accuracy: 0.4998
18080/25000 [====================>.........] - ETA: 25s - loss: 7.6717 - accuracy: 0.4997
18112/25000 [====================>.........] - ETA: 24s - loss: 7.6717 - accuracy: 0.4997
18144/25000 [====================>.........] - ETA: 24s - loss: 7.6725 - accuracy: 0.4996
18176/25000 [====================>.........] - ETA: 24s - loss: 7.6708 - accuracy: 0.4997
18208/25000 [====================>.........] - ETA: 24s - loss: 7.6725 - accuracy: 0.4996
18240/25000 [====================>.........] - ETA: 24s - loss: 7.6725 - accuracy: 0.4996
18272/25000 [====================>.........] - ETA: 24s - loss: 7.6733 - accuracy: 0.4996
18304/25000 [====================>.........] - ETA: 24s - loss: 7.6758 - accuracy: 0.4994
18336/25000 [=====================>........] - ETA: 24s - loss: 7.6758 - accuracy: 0.4994
18368/25000 [=====================>........] - ETA: 24s - loss: 7.6775 - accuracy: 0.4993
18400/25000 [=====================>........] - ETA: 23s - loss: 7.6791 - accuracy: 0.4992
18432/25000 [=====================>........] - ETA: 23s - loss: 7.6758 - accuracy: 0.4994
18464/25000 [=====================>........] - ETA: 23s - loss: 7.6766 - accuracy: 0.4994
18496/25000 [=====================>........] - ETA: 23s - loss: 7.6757 - accuracy: 0.4994
18528/25000 [=====================>........] - ETA: 23s - loss: 7.6757 - accuracy: 0.4994
18560/25000 [=====================>........] - ETA: 23s - loss: 7.6765 - accuracy: 0.4994
18592/25000 [=====================>........] - ETA: 23s - loss: 7.6773 - accuracy: 0.4993
18624/25000 [=====================>........] - ETA: 23s - loss: 7.6814 - accuracy: 0.4990
18656/25000 [=====================>........] - ETA: 22s - loss: 7.6855 - accuracy: 0.4988
18688/25000 [=====================>........] - ETA: 22s - loss: 7.6871 - accuracy: 0.4987
18720/25000 [=====================>........] - ETA: 22s - loss: 7.6887 - accuracy: 0.4986
18752/25000 [=====================>........] - ETA: 22s - loss: 7.6895 - accuracy: 0.4985
18784/25000 [=====================>........] - ETA: 22s - loss: 7.6862 - accuracy: 0.4987
18816/25000 [=====================>........] - ETA: 22s - loss: 7.6870 - accuracy: 0.4987
18848/25000 [=====================>........] - ETA: 22s - loss: 7.6829 - accuracy: 0.4989
18880/25000 [=====================>........] - ETA: 22s - loss: 7.6853 - accuracy: 0.4988
18912/25000 [=====================>........] - ETA: 22s - loss: 7.6845 - accuracy: 0.4988
18944/25000 [=====================>........] - ETA: 21s - loss: 7.6860 - accuracy: 0.4987
18976/25000 [=====================>........] - ETA: 21s - loss: 7.6868 - accuracy: 0.4987
19008/25000 [=====================>........] - ETA: 21s - loss: 7.6876 - accuracy: 0.4986
19040/25000 [=====================>........] - ETA: 21s - loss: 7.6932 - accuracy: 0.4983
19072/25000 [=====================>........] - ETA: 21s - loss: 7.6907 - accuracy: 0.4984
19104/25000 [=====================>........] - ETA: 21s - loss: 7.6883 - accuracy: 0.4986
19136/25000 [=====================>........] - ETA: 21s - loss: 7.6867 - accuracy: 0.4987
19168/25000 [======================>.......] - ETA: 21s - loss: 7.6890 - accuracy: 0.4985
19200/25000 [======================>.......] - ETA: 21s - loss: 7.6914 - accuracy: 0.4984
19232/25000 [======================>.......] - ETA: 20s - loss: 7.6897 - accuracy: 0.4985
19264/25000 [======================>.......] - ETA: 20s - loss: 7.6889 - accuracy: 0.4985
19296/25000 [======================>.......] - ETA: 20s - loss: 7.6928 - accuracy: 0.4983
19328/25000 [======================>.......] - ETA: 20s - loss: 7.6936 - accuracy: 0.4982
19360/25000 [======================>.......] - ETA: 20s - loss: 7.6943 - accuracy: 0.4982
19392/25000 [======================>.......] - ETA: 20s - loss: 7.6967 - accuracy: 0.4980
19424/25000 [======================>.......] - ETA: 20s - loss: 7.7014 - accuracy: 0.4977
19456/25000 [======================>.......] - ETA: 20s - loss: 7.7013 - accuracy: 0.4977
19488/25000 [======================>.......] - ETA: 19s - loss: 7.6989 - accuracy: 0.4979
19520/25000 [======================>.......] - ETA: 19s - loss: 7.6980 - accuracy: 0.4980
19552/25000 [======================>.......] - ETA: 19s - loss: 7.6988 - accuracy: 0.4979
19584/25000 [======================>.......] - ETA: 19s - loss: 7.7019 - accuracy: 0.4977
19616/25000 [======================>.......] - ETA: 19s - loss: 7.7041 - accuracy: 0.4976
19648/25000 [======================>.......] - ETA: 19s - loss: 7.7072 - accuracy: 0.4974
19680/25000 [======================>.......] - ETA: 19s - loss: 7.7110 - accuracy: 0.4971
19712/25000 [======================>.......] - ETA: 19s - loss: 7.7110 - accuracy: 0.4971
19744/25000 [======================>.......] - ETA: 19s - loss: 7.7101 - accuracy: 0.4972
19776/25000 [======================>.......] - ETA: 18s - loss: 7.7124 - accuracy: 0.4970
19808/25000 [======================>.......] - ETA: 18s - loss: 7.7131 - accuracy: 0.4970
19840/25000 [======================>.......] - ETA: 18s - loss: 7.7130 - accuracy: 0.4970
19872/25000 [======================>.......] - ETA: 18s - loss: 7.7152 - accuracy: 0.4968
19904/25000 [======================>.......] - ETA: 18s - loss: 7.7152 - accuracy: 0.4968
19936/25000 [======================>.......] - ETA: 18s - loss: 7.7166 - accuracy: 0.4967
19968/25000 [======================>.......] - ETA: 18s - loss: 7.7165 - accuracy: 0.4967
20000/25000 [=======================>......] - ETA: 18s - loss: 7.7157 - accuracy: 0.4968
20032/25000 [=======================>......] - ETA: 18s - loss: 7.7187 - accuracy: 0.4966
20064/25000 [=======================>......] - ETA: 17s - loss: 7.7194 - accuracy: 0.4966
20096/25000 [=======================>......] - ETA: 17s - loss: 7.7185 - accuracy: 0.4966
20128/25000 [=======================>......] - ETA: 17s - loss: 7.7177 - accuracy: 0.4967
20160/25000 [=======================>......] - ETA: 17s - loss: 7.7168 - accuracy: 0.4967
20192/25000 [=======================>......] - ETA: 17s - loss: 7.7160 - accuracy: 0.4968
20224/25000 [=======================>......] - ETA: 17s - loss: 7.7144 - accuracy: 0.4969
20256/25000 [=======================>......] - ETA: 17s - loss: 7.7143 - accuracy: 0.4969
20288/25000 [=======================>......] - ETA: 17s - loss: 7.7135 - accuracy: 0.4969
20320/25000 [=======================>......] - ETA: 16s - loss: 7.7134 - accuracy: 0.4969
20352/25000 [=======================>......] - ETA: 16s - loss: 7.7133 - accuracy: 0.4970
20384/25000 [=======================>......] - ETA: 16s - loss: 7.7148 - accuracy: 0.4969
20416/25000 [=======================>......] - ETA: 16s - loss: 7.7124 - accuracy: 0.4970
20448/25000 [=======================>......] - ETA: 16s - loss: 7.7116 - accuracy: 0.4971
20480/25000 [=======================>......] - ETA: 16s - loss: 7.7130 - accuracy: 0.4970
20512/25000 [=======================>......] - ETA: 16s - loss: 7.7107 - accuracy: 0.4971
20544/25000 [=======================>......] - ETA: 16s - loss: 7.7107 - accuracy: 0.4971
20576/25000 [=======================>......] - ETA: 16s - loss: 7.7136 - accuracy: 0.4969
20608/25000 [=======================>......] - ETA: 15s - loss: 7.7113 - accuracy: 0.4971
20640/25000 [=======================>......] - ETA: 15s - loss: 7.7082 - accuracy: 0.4973
20672/25000 [=======================>......] - ETA: 15s - loss: 7.7037 - accuracy: 0.4976
20704/25000 [=======================>......] - ETA: 15s - loss: 7.7029 - accuracy: 0.4976
20736/25000 [=======================>......] - ETA: 15s - loss: 7.7029 - accuracy: 0.4976
20768/25000 [=======================>......] - ETA: 15s - loss: 7.6998 - accuracy: 0.4978
20800/25000 [=======================>......] - ETA: 15s - loss: 7.6991 - accuracy: 0.4979
20832/25000 [=======================>......] - ETA: 15s - loss: 7.6975 - accuracy: 0.4980
20864/25000 [========================>.....] - ETA: 15s - loss: 7.6997 - accuracy: 0.4978
20896/25000 [========================>.....] - ETA: 14s - loss: 7.6989 - accuracy: 0.4979
20928/25000 [========================>.....] - ETA: 14s - loss: 7.6974 - accuracy: 0.4980
20960/25000 [========================>.....] - ETA: 14s - loss: 7.6995 - accuracy: 0.4979
20992/25000 [========================>.....] - ETA: 14s - loss: 7.6988 - accuracy: 0.4979
21024/25000 [========================>.....] - ETA: 14s - loss: 7.6958 - accuracy: 0.4981
21056/25000 [========================>.....] - ETA: 14s - loss: 7.6936 - accuracy: 0.4982
21088/25000 [========================>.....] - ETA: 14s - loss: 7.6899 - accuracy: 0.4985
21120/25000 [========================>.....] - ETA: 14s - loss: 7.6877 - accuracy: 0.4986
21152/25000 [========================>.....] - ETA: 13s - loss: 7.6876 - accuracy: 0.4986
21184/25000 [========================>.....] - ETA: 13s - loss: 7.6876 - accuracy: 0.4986
21216/25000 [========================>.....] - ETA: 13s - loss: 7.6854 - accuracy: 0.4988
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6847 - accuracy: 0.4988
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6825 - accuracy: 0.4990
21312/25000 [========================>.....] - ETA: 13s - loss: 7.6810 - accuracy: 0.4991
21344/25000 [========================>.....] - ETA: 13s - loss: 7.6831 - accuracy: 0.4989
21376/25000 [========================>.....] - ETA: 13s - loss: 7.6831 - accuracy: 0.4989
21408/25000 [========================>.....] - ETA: 13s - loss: 7.6817 - accuracy: 0.4990
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6824 - accuracy: 0.4990
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6823 - accuracy: 0.4990
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6795 - accuracy: 0.4992
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6794 - accuracy: 0.4992
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6752 - accuracy: 0.4994
21600/25000 [========================>.....] - ETA: 12s - loss: 7.6787 - accuracy: 0.4992
21632/25000 [========================>.....] - ETA: 12s - loss: 7.6773 - accuracy: 0.4993
21664/25000 [========================>.....] - ETA: 12s - loss: 7.6758 - accuracy: 0.4994
21696/25000 [=========================>....] - ETA: 11s - loss: 7.6786 - accuracy: 0.4992
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6800 - accuracy: 0.4991
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6772 - accuracy: 0.4993
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6772 - accuracy: 0.4993
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6786 - accuracy: 0.4992
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6814 - accuracy: 0.4990
21888/25000 [=========================>....] - ETA: 11s - loss: 7.6792 - accuracy: 0.4992
21920/25000 [=========================>....] - ETA: 11s - loss: 7.6785 - accuracy: 0.4992
21952/25000 [=========================>....] - ETA: 11s - loss: 7.6778 - accuracy: 0.4993
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6785 - accuracy: 0.4992
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6771 - accuracy: 0.4993
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6750 - accuracy: 0.4995
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6701 - accuracy: 0.4998
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6687 - accuracy: 0.4999
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6659 - accuracy: 0.5000
22176/25000 [=========================>....] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
22208/25000 [=========================>....] - ETA: 10s - loss: 7.6680 - accuracy: 0.4999
22240/25000 [=========================>....] - ETA: 10s - loss: 7.6694 - accuracy: 0.4998
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6714 - accuracy: 0.4997 
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6694 - accuracy: 0.4998
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6680 - accuracy: 0.4999
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6714 - accuracy: 0.4997
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6728 - accuracy: 0.4996
22432/25000 [=========================>....] - ETA: 9s - loss: 7.6762 - accuracy: 0.4994
22464/25000 [=========================>....] - ETA: 9s - loss: 7.6775 - accuracy: 0.4993
22496/25000 [=========================>....] - ETA: 9s - loss: 7.6768 - accuracy: 0.4993
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6755 - accuracy: 0.4994
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6789 - accuracy: 0.4992
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6748 - accuracy: 0.4995
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6727 - accuracy: 0.4996
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6727 - accuracy: 0.4996
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6714 - accuracy: 0.4997
22720/25000 [==========================>...] - ETA: 8s - loss: 7.6707 - accuracy: 0.4997
22752/25000 [==========================>...] - ETA: 8s - loss: 7.6693 - accuracy: 0.4998
22784/25000 [==========================>...] - ETA: 8s - loss: 7.6707 - accuracy: 0.4997
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6720 - accuracy: 0.4996
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6686 - accuracy: 0.4999
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6653 - accuracy: 0.5001
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6633 - accuracy: 0.5002
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6626 - accuracy: 0.5003
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6599 - accuracy: 0.5004
23008/25000 [==========================>...] - ETA: 7s - loss: 7.6633 - accuracy: 0.5002
23040/25000 [==========================>...] - ETA: 7s - loss: 7.6646 - accuracy: 0.5001
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6660 - accuracy: 0.5000
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6653 - accuracy: 0.5001
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6640 - accuracy: 0.5002
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6626 - accuracy: 0.5003
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6646 - accuracy: 0.5001
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6627 - accuracy: 0.5003
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6640 - accuracy: 0.5002
23296/25000 [==========================>...] - ETA: 6s - loss: 7.6627 - accuracy: 0.5003
23328/25000 [==========================>...] - ETA: 6s - loss: 7.6607 - accuracy: 0.5004
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6581 - accuracy: 0.5006
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6588 - accuracy: 0.5005
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6614 - accuracy: 0.5003
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6640 - accuracy: 0.5002
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6679 - accuracy: 0.4999
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6686 - accuracy: 0.4999
23584/25000 [===========================>..] - ETA: 5s - loss: 7.6673 - accuracy: 0.5000
23616/25000 [===========================>..] - ETA: 5s - loss: 7.6725 - accuracy: 0.4996
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6738 - accuracy: 0.4995
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6737 - accuracy: 0.4995
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6724 - accuracy: 0.4996
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6698 - accuracy: 0.4998
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6724 - accuracy: 0.4996
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6711 - accuracy: 0.4997
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6718 - accuracy: 0.4997
23872/25000 [===========================>..] - ETA: 4s - loss: 7.6718 - accuracy: 0.4997
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6724 - accuracy: 0.4996
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6685 - accuracy: 0.4999
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6673 - accuracy: 0.5000
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6660 - accuracy: 0.5000
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6641 - accuracy: 0.5002
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6609 - accuracy: 0.5004
24160/25000 [===========================>..] - ETA: 3s - loss: 7.6584 - accuracy: 0.5005
24192/25000 [============================>.] - ETA: 2s - loss: 7.6577 - accuracy: 0.5006
24224/25000 [============================>.] - ETA: 2s - loss: 7.6552 - accuracy: 0.5007
24256/25000 [============================>.] - ETA: 2s - loss: 7.6565 - accuracy: 0.5007
24288/25000 [============================>.] - ETA: 2s - loss: 7.6590 - accuracy: 0.5005
24320/25000 [============================>.] - ETA: 2s - loss: 7.6603 - accuracy: 0.5004
24352/25000 [============================>.] - ETA: 2s - loss: 7.6635 - accuracy: 0.5002
24384/25000 [============================>.] - ETA: 2s - loss: 7.6616 - accuracy: 0.5003
24416/25000 [============================>.] - ETA: 2s - loss: 7.6610 - accuracy: 0.5004
24448/25000 [============================>.] - ETA: 1s - loss: 7.6603 - accuracy: 0.5004
24480/25000 [============================>.] - ETA: 1s - loss: 7.6572 - accuracy: 0.5006
24512/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5004
24544/25000 [============================>.] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
24576/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24608/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24640/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24672/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24704/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24736/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24768/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24800/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24832/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24864/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
24896/25000 [============================>.] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
24928/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24960/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 109s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
