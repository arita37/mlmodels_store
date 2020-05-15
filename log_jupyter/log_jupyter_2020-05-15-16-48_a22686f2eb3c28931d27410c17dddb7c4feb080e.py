
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/a22686f2eb3c28931d27410c17dddb7c4feb080e', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'a22686f2eb3c28931d27410c17dddb7c4feb080e', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/a22686f2eb3c28931d27410c17dddb7c4feb080e

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/a22686f2eb3c28931d27410c17dddb7c4feb080e

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
	Data preprocessing and feature engineering runtime = 0.21s ...
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
 40%|████      | 2/5 [00:49<01:13, 24.53s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.1938442718892946, 'embedding_size_factor': 0.5394556704386952, 'layers.choice': 1, 'learning_rate': 0.0001301550396276345, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.02758505703672164} and reward: 0.3666
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc8\xcf\xe3\x9c$\x06\xa2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe1C8\x89\xc5\xa1bX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?!\x0fGF\xf7\xed\x05X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\x9c?A\xd7R\xf5]u.' and reward: 0.3666
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc8\xcf\xe3\x9c$\x06\xa2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe1C8\x89\xc5\xa1bX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?!\x0fGF\xf7\xed\x05X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\x9c?A\xd7R\xf5]u.' and reward: 0.3666
 60%|██████    | 3/5 [02:33<01:37, 48.63s/it] 60%|██████    | 3/5 [02:33<01:42, 51.31s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.38211451501055504, 'embedding_size_factor': 1.4786078152791395, 'layers.choice': 1, 'learning_rate': 0.0011102016163048022, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1.1593702631706757e-12} and reward: 0.3816
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd8t\x90pS\x05\xe6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7\xa8`\xab#\xbe\x1cX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?R0\x85\xe8\x93a\xb8X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=teV\xe8\x80\x87\x00u.' and reward: 0.3816
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd8t\x90pS\x05\xe6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf7\xa8`\xab#\xbe\x1cX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?R0\x85\xe8\x93a\xb8X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=teV\xe8\x80\x87\x00u.' and reward: 0.3816
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 248.46163821220398
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.79s of the -131.07s of remaining time.
Ensemble size: 20
Ensemble weights: 
[0.6  0.35 0.05]
	0.392	 = Validation accuracy score
	0.98s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 252.09s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f55dd0eaba8> 

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
 [ 0.04799731 -0.02740901  0.02275992  0.02422    -0.02189661  0.04416315]
 [-0.06462751  0.046125   -0.12080213 -0.00854548 -0.13624302  0.10142428]
 [ 0.21420564  0.01396314 -0.15396661  0.16842271  0.1867761   0.20092633]
 [-0.37214038  0.09650123  0.11827571  0.28560954 -0.00151451  0.33816496]
 [-0.12863034  0.23008402 -0.30728489 -0.02180134  0.26603371 -0.24514878]
 [-0.11998782  0.22189766  0.17489091 -0.01035135  0.62964386  0.2575587 ]
 [ 0.50679642 -0.00139893  0.00840601 -0.31553864  0.45788467 -0.0435121 ]
 [-0.17802778  0.33800519  0.0289729   0.16452292  0.3047224   0.03848578]
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
{'loss': 0.49453918635845184, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 16:52:51.103009: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.6084198877215385, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-15 16:52:52.201000: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
  163840/17464789 [..............................] - ETA: 5s
  630784/17464789 [>.............................] - ETA: 2s
 1449984/17464789 [=>............................] - ETA: 1s
 2195456/17464789 [==>...........................] - ETA: 1s
 2531328/17464789 [===>..........................] - ETA: 1s
 2859008/17464789 [===>..........................] - ETA: 1s
 3121152/17464789 [====>.........................] - ETA: 1s
 3645440/17464789 [=====>........................] - ETA: 1s
 3743744/17464789 [=====>........................] - ETA: 1s
 3833856/17464789 [=====>........................] - ETA: 1s
 3940352/17464789 [=====>........................] - ETA: 1s
 4038656/17464789 [=====>........................] - ETA: 2s
 4161536/17464789 [======>.......................] - ETA: 2s
 4317184/17464789 [======>.......................] - ETA: 2s
 4481024/17464789 [======>.......................] - ETA: 2s
 4653056/17464789 [======>.......................] - ETA: 2s
 4825088/17464789 [=======>......................] - ETA: 2s
 4997120/17464789 [=======>......................] - ETA: 2s
 5152768/17464789 [=======>......................] - ETA: 2s
 5398528/17464789 [========>.....................] - ETA: 2s
 5611520/17464789 [========>.....................] - ETA: 2s
 5816320/17464789 [========>.....................] - ETA: 2s
 6078464/17464789 [=========>....................] - ETA: 2s
 6586368/17464789 [==========>...................] - ETA: 2s
 6774784/17464789 [==========>...................] - ETA: 2s
 6914048/17464789 [==========>...................] - ETA: 2s
 7143424/17464789 [===========>..................] - ETA: 2s
 7454720/17464789 [===========>..................] - ETA: 1s
 7798784/17464789 [============>.................] - ETA: 1s
 8118272/17464789 [============>.................] - ETA: 1s
 8380416/17464789 [=============>................] - ETA: 1s
 8568832/17464789 [=============>................] - ETA: 1s
 8830976/17464789 [==============>...............] - ETA: 1s
 9158656/17464789 [==============>...............] - ETA: 1s
 9789440/17464789 [===============>..............] - ETA: 1s
10690560/17464789 [=================>............] - ETA: 1s
11141120/17464789 [==================>...........] - ETA: 1s
11247616/17464789 [==================>...........] - ETA: 1s
11403264/17464789 [==================>...........] - ETA: 1s
11583488/17464789 [==================>...........] - ETA: 1s
11812864/17464789 [===================>..........] - ETA: 1s
12009472/17464789 [===================>..........] - ETA: 1s
12222464/17464789 [===================>..........] - ETA: 0s
12435456/17464789 [====================>.........] - ETA: 0s
12640256/17464789 [====================>.........] - ETA: 0s
12869632/17464789 [=====================>........] - ETA: 0s
13131776/17464789 [=====================>........] - ETA: 0s
13369344/17464789 [=====================>........] - ETA: 0s
13811712/17464789 [======================>.......] - ETA: 0s
14327808/17464789 [=======================>......] - ETA: 0s
14835712/17464789 [========================>.....] - ETA: 0s
15343616/17464789 [=========================>....] - ETA: 0s
15810560/17464789 [==========================>...] - ETA: 0s
16506880/17464789 [===========================>..] - ETA: 0s
16891904/17464789 [============================>.] - ETA: 0s
17186816/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 3s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 16:53:05.631545: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 16:53:05.635352: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-15 16:53:05.635494: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556bfd849e90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 16:53:05.635508: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:01 - loss: 5.7500 - accuracy: 0.6250
   64/25000 [..............................] - ETA: 2:40 - loss: 6.2291 - accuracy: 0.5938
   96/25000 [..............................] - ETA: 2:15 - loss: 6.8680 - accuracy: 0.5521
  128/25000 [..............................] - ETA: 2:02 - loss: 6.8281 - accuracy: 0.5547
  160/25000 [..............................] - ETA: 1:55 - loss: 7.2833 - accuracy: 0.5250
  192/25000 [..............................] - ETA: 1:51 - loss: 7.3472 - accuracy: 0.5208
  224/25000 [..............................] - ETA: 1:48 - loss: 7.3928 - accuracy: 0.5179
  256/25000 [..............................] - ETA: 1:45 - loss: 7.4270 - accuracy: 0.5156
  288/25000 [..............................] - ETA: 1:41 - loss: 7.4004 - accuracy: 0.5174
  320/25000 [..............................] - ETA: 1:40 - loss: 7.6666 - accuracy: 0.5000
  352/25000 [..............................] - ETA: 1:37 - loss: 7.4053 - accuracy: 0.5170
  384/25000 [..............................] - ETA: 1:36 - loss: 7.3072 - accuracy: 0.5234
  416/25000 [..............................] - ETA: 1:36 - loss: 7.4455 - accuracy: 0.5144
  448/25000 [..............................] - ETA: 1:34 - loss: 7.3586 - accuracy: 0.5201
  480/25000 [..............................] - ETA: 1:34 - loss: 7.4430 - accuracy: 0.5146
  512/25000 [..............................] - ETA: 1:34 - loss: 7.4570 - accuracy: 0.5137
  544/25000 [..............................] - ETA: 1:33 - loss: 7.5257 - accuracy: 0.5092
  576/25000 [..............................] - ETA: 1:32 - loss: 7.4803 - accuracy: 0.5122
  608/25000 [..............................] - ETA: 1:31 - loss: 7.6162 - accuracy: 0.5033
  640/25000 [..............................] - ETA: 1:30 - loss: 7.6187 - accuracy: 0.5031
  672/25000 [..............................] - ETA: 1:30 - loss: 7.6210 - accuracy: 0.5030
  704/25000 [..............................] - ETA: 1:29 - loss: 7.5577 - accuracy: 0.5071
  736/25000 [..............................] - ETA: 1:29 - loss: 7.5416 - accuracy: 0.5082
  768/25000 [..............................] - ETA: 1:28 - loss: 7.5868 - accuracy: 0.5052
  800/25000 [..............................] - ETA: 1:29 - loss: 7.5516 - accuracy: 0.5075
  832/25000 [..............................] - ETA: 1:29 - loss: 7.5376 - accuracy: 0.5084
  864/25000 [>.............................] - ETA: 1:29 - loss: 7.5424 - accuracy: 0.5081
  896/25000 [>.............................] - ETA: 1:28 - loss: 7.4442 - accuracy: 0.5145
  928/25000 [>.............................] - ETA: 1:28 - loss: 7.4849 - accuracy: 0.5119
  960/25000 [>.............................] - ETA: 1:27 - loss: 7.4270 - accuracy: 0.5156
  992/25000 [>.............................] - ETA: 1:27 - loss: 7.3111 - accuracy: 0.5232
 1024/25000 [>.............................] - ETA: 1:27 - loss: 7.3222 - accuracy: 0.5225
 1056/25000 [>.............................] - ETA: 1:26 - loss: 7.3327 - accuracy: 0.5218
 1088/25000 [>.............................] - ETA: 1:26 - loss: 7.3284 - accuracy: 0.5221
 1120/25000 [>.............................] - ETA: 1:26 - loss: 7.3244 - accuracy: 0.5223
 1152/25000 [>.............................] - ETA: 1:25 - loss: 7.3072 - accuracy: 0.5234
 1184/25000 [>.............................] - ETA: 1:25 - loss: 7.3299 - accuracy: 0.5220
 1216/25000 [>.............................] - ETA: 1:25 - loss: 7.3388 - accuracy: 0.5214
 1248/25000 [>.............................] - ETA: 1:24 - loss: 7.2980 - accuracy: 0.5240
 1280/25000 [>.............................] - ETA: 1:24 - loss: 7.3312 - accuracy: 0.5219
 1312/25000 [>.............................] - ETA: 1:24 - loss: 7.3043 - accuracy: 0.5236
 1344/25000 [>.............................] - ETA: 1:24 - loss: 7.3586 - accuracy: 0.5201
 1376/25000 [>.............................] - ETA: 1:23 - loss: 7.3546 - accuracy: 0.5203
 1408/25000 [>.............................] - ETA: 1:23 - loss: 7.3290 - accuracy: 0.5220
 1440/25000 [>.............................] - ETA: 1:23 - loss: 7.3472 - accuracy: 0.5208
 1472/25000 [>.............................] - ETA: 1:23 - loss: 7.4062 - accuracy: 0.5170
 1504/25000 [>.............................] - ETA: 1:22 - loss: 7.3812 - accuracy: 0.5186
 1536/25000 [>.............................] - ETA: 1:22 - loss: 7.3871 - accuracy: 0.5182
 1568/25000 [>.............................] - ETA: 1:22 - loss: 7.3537 - accuracy: 0.5204
 1600/25000 [>.............................] - ETA: 1:21 - loss: 7.3216 - accuracy: 0.5225
 1632/25000 [>.............................] - ETA: 1:21 - loss: 7.2908 - accuracy: 0.5245
 1664/25000 [>.............................] - ETA: 1:21 - loss: 7.3072 - accuracy: 0.5234
 1696/25000 [=>............................] - ETA: 1:21 - loss: 7.3140 - accuracy: 0.5230
 1728/25000 [=>............................] - ETA: 1:20 - loss: 7.2584 - accuracy: 0.5266
 1760/25000 [=>............................] - ETA: 1:20 - loss: 7.2920 - accuracy: 0.5244
 1792/25000 [=>............................] - ETA: 1:20 - loss: 7.2645 - accuracy: 0.5262
 1824/25000 [=>............................] - ETA: 1:20 - loss: 7.2883 - accuracy: 0.5247
 1856/25000 [=>............................] - ETA: 1:19 - loss: 7.3196 - accuracy: 0.5226
 1888/25000 [=>............................] - ETA: 1:19 - loss: 7.3174 - accuracy: 0.5228
 1920/25000 [=>............................] - ETA: 1:19 - loss: 7.3072 - accuracy: 0.5234
 1952/25000 [=>............................] - ETA: 1:19 - loss: 7.3131 - accuracy: 0.5231
 1984/25000 [=>............................] - ETA: 1:18 - loss: 7.2802 - accuracy: 0.5252
 2016/25000 [=>............................] - ETA: 1:18 - loss: 7.2863 - accuracy: 0.5248
 2048/25000 [=>............................] - ETA: 1:18 - loss: 7.2923 - accuracy: 0.5244
 2080/25000 [=>............................] - ETA: 1:18 - loss: 7.2907 - accuracy: 0.5245
 2112/25000 [=>............................] - ETA: 1:18 - loss: 7.2891 - accuracy: 0.5246
 2144/25000 [=>............................] - ETA: 1:18 - loss: 7.2876 - accuracy: 0.5247
 2176/25000 [=>............................] - ETA: 1:18 - loss: 7.2932 - accuracy: 0.5244
 2208/25000 [=>............................] - ETA: 1:18 - loss: 7.3125 - accuracy: 0.5231
 2240/25000 [=>............................] - ETA: 1:17 - loss: 7.3038 - accuracy: 0.5237
 2272/25000 [=>............................] - ETA: 1:17 - loss: 7.3427 - accuracy: 0.5211
 2304/25000 [=>............................] - ETA: 1:17 - loss: 7.3671 - accuracy: 0.5195
 2336/25000 [=>............................] - ETA: 1:17 - loss: 7.3450 - accuracy: 0.5210
 2368/25000 [=>............................] - ETA: 1:17 - loss: 7.3558 - accuracy: 0.5203
 2400/25000 [=>............................] - ETA: 1:17 - loss: 7.3536 - accuracy: 0.5204
 2432/25000 [=>............................] - ETA: 1:17 - loss: 7.3577 - accuracy: 0.5201
 2464/25000 [=>............................] - ETA: 1:17 - loss: 7.3741 - accuracy: 0.5191
 2496/25000 [=>............................] - ETA: 1:16 - loss: 7.3533 - accuracy: 0.5204
 2528/25000 [==>...........................] - ETA: 1:16 - loss: 7.3391 - accuracy: 0.5214
 2560/25000 [==>...........................] - ETA: 1:16 - loss: 7.3671 - accuracy: 0.5195
 2592/25000 [==>...........................] - ETA: 1:16 - loss: 7.3649 - accuracy: 0.5197
 2624/25000 [==>...........................] - ETA: 1:16 - loss: 7.3686 - accuracy: 0.5194
 2656/25000 [==>...........................] - ETA: 1:16 - loss: 7.3837 - accuracy: 0.5184
 2688/25000 [==>...........................] - ETA: 1:16 - loss: 7.3529 - accuracy: 0.5205
 2720/25000 [==>...........................] - ETA: 1:15 - loss: 7.3566 - accuracy: 0.5202
 2752/25000 [==>...........................] - ETA: 1:15 - loss: 7.3379 - accuracy: 0.5214
 2784/25000 [==>...........................] - ETA: 1:15 - loss: 7.3362 - accuracy: 0.5216
 2816/25000 [==>...........................] - ETA: 1:15 - loss: 7.3399 - accuracy: 0.5213
 2848/25000 [==>...........................] - ETA: 1:15 - loss: 7.3544 - accuracy: 0.5204
 2880/25000 [==>...........................] - ETA: 1:15 - loss: 7.3685 - accuracy: 0.5194
 2912/25000 [==>...........................] - ETA: 1:14 - loss: 7.3823 - accuracy: 0.5185
 2944/25000 [==>...........................] - ETA: 1:14 - loss: 7.3750 - accuracy: 0.5190
 2976/25000 [==>...........................] - ETA: 1:14 - loss: 7.3729 - accuracy: 0.5192
 3008/25000 [==>...........................] - ETA: 1:14 - loss: 7.3863 - accuracy: 0.5183
 3040/25000 [==>...........................] - ETA: 1:14 - loss: 7.3842 - accuracy: 0.5184
 3072/25000 [==>...........................] - ETA: 1:14 - loss: 7.3671 - accuracy: 0.5195
 3104/25000 [==>...........................] - ETA: 1:13 - loss: 7.3801 - accuracy: 0.5187
 3136/25000 [==>...........................] - ETA: 1:13 - loss: 7.3781 - accuracy: 0.5188
 3168/25000 [==>...........................] - ETA: 1:13 - loss: 7.3859 - accuracy: 0.5183
 3200/25000 [==>...........................] - ETA: 1:13 - loss: 7.3935 - accuracy: 0.5178
 3232/25000 [==>...........................] - ETA: 1:13 - loss: 7.3962 - accuracy: 0.5176
 3264/25000 [==>...........................] - ETA: 1:13 - loss: 7.3989 - accuracy: 0.5175
 3296/25000 [==>...........................] - ETA: 1:13 - loss: 7.3968 - accuracy: 0.5176
 3328/25000 [==>...........................] - ETA: 1:13 - loss: 7.4178 - accuracy: 0.5162
 3360/25000 [===>..........................] - ETA: 1:13 - loss: 7.4156 - accuracy: 0.5164
 3392/25000 [===>..........................] - ETA: 1:13 - loss: 7.4135 - accuracy: 0.5165
 3424/25000 [===>..........................] - ETA: 1:12 - loss: 7.4338 - accuracy: 0.5152
 3456/25000 [===>..........................] - ETA: 1:12 - loss: 7.4315 - accuracy: 0.5153
 3488/25000 [===>..........................] - ETA: 1:12 - loss: 7.4380 - accuracy: 0.5149
 3520/25000 [===>..........................] - ETA: 1:12 - loss: 7.4357 - accuracy: 0.5151
 3552/25000 [===>..........................] - ETA: 1:12 - loss: 7.4249 - accuracy: 0.5158
 3584/25000 [===>..........................] - ETA: 1:12 - loss: 7.4185 - accuracy: 0.5162
 3616/25000 [===>..........................] - ETA: 1:12 - loss: 7.4207 - accuracy: 0.5160
 3648/25000 [===>..........................] - ETA: 1:12 - loss: 7.4607 - accuracy: 0.5134
 3680/25000 [===>..........................] - ETA: 1:12 - loss: 7.4375 - accuracy: 0.5149
 3712/25000 [===>..........................] - ETA: 1:11 - loss: 7.4477 - accuracy: 0.5143
 3744/25000 [===>..........................] - ETA: 1:11 - loss: 7.4578 - accuracy: 0.5136
 3776/25000 [===>..........................] - ETA: 1:11 - loss: 7.4676 - accuracy: 0.5130
 3808/25000 [===>..........................] - ETA: 1:11 - loss: 7.4653 - accuracy: 0.5131
 3840/25000 [===>..........................] - ETA: 1:11 - loss: 7.4550 - accuracy: 0.5138
 3872/25000 [===>..........................] - ETA: 1:11 - loss: 7.4607 - accuracy: 0.5134
 3904/25000 [===>..........................] - ETA: 1:11 - loss: 7.4702 - accuracy: 0.5128
 3936/25000 [===>..........................] - ETA: 1:10 - loss: 7.4757 - accuracy: 0.5124
 3968/25000 [===>..........................] - ETA: 1:10 - loss: 7.4773 - accuracy: 0.5123
 4000/25000 [===>..........................] - ETA: 1:10 - loss: 7.4750 - accuracy: 0.5125
 4032/25000 [===>..........................] - ETA: 1:10 - loss: 7.4727 - accuracy: 0.5126
 4064/25000 [===>..........................] - ETA: 1:10 - loss: 7.4780 - accuracy: 0.5123
 4096/25000 [===>..........................] - ETA: 1:10 - loss: 7.4607 - accuracy: 0.5134
 4128/25000 [===>..........................] - ETA: 1:10 - loss: 7.4698 - accuracy: 0.5128
 4160/25000 [===>..........................] - ETA: 1:10 - loss: 7.4676 - accuracy: 0.5130
 4192/25000 [====>.........................] - ETA: 1:09 - loss: 7.4581 - accuracy: 0.5136
 4224/25000 [====>.........................] - ETA: 1:09 - loss: 7.4561 - accuracy: 0.5137
 4256/25000 [====>.........................] - ETA: 1:09 - loss: 7.4649 - accuracy: 0.5132
 4288/25000 [====>.........................] - ETA: 1:09 - loss: 7.4556 - accuracy: 0.5138
 4320/25000 [====>.........................] - ETA: 1:09 - loss: 7.4608 - accuracy: 0.5134
 4352/25000 [====>.........................] - ETA: 1:09 - loss: 7.4623 - accuracy: 0.5133
 4384/25000 [====>.........................] - ETA: 1:09 - loss: 7.4603 - accuracy: 0.5135
 4416/25000 [====>.........................] - ETA: 1:09 - loss: 7.4548 - accuracy: 0.5138
 4448/25000 [====>.........................] - ETA: 1:08 - loss: 7.4632 - accuracy: 0.5133
 4480/25000 [====>.........................] - ETA: 1:08 - loss: 7.4544 - accuracy: 0.5138
 4512/25000 [====>.........................] - ETA: 1:08 - loss: 7.4491 - accuracy: 0.5142
 4544/25000 [====>.........................] - ETA: 1:08 - loss: 7.4675 - accuracy: 0.5130
 4576/25000 [====>.........................] - ETA: 1:08 - loss: 7.4689 - accuracy: 0.5129
 4608/25000 [====>.........................] - ETA: 1:08 - loss: 7.4736 - accuracy: 0.5126
 4640/25000 [====>.........................] - ETA: 1:08 - loss: 7.4750 - accuracy: 0.5125
 4672/25000 [====>.........................] - ETA: 1:07 - loss: 7.4763 - accuracy: 0.5124
 4704/25000 [====>.........................] - ETA: 1:07 - loss: 7.4808 - accuracy: 0.5121
 4736/25000 [====>.........................] - ETA: 1:07 - loss: 7.4691 - accuracy: 0.5129
 4768/25000 [====>.........................] - ETA: 1:07 - loss: 7.4512 - accuracy: 0.5141
 4800/25000 [====>.........................] - ETA: 1:07 - loss: 7.4366 - accuracy: 0.5150
 4832/25000 [====>.........................] - ETA: 1:07 - loss: 7.4286 - accuracy: 0.5155
 4864/25000 [====>.........................] - ETA: 1:07 - loss: 7.4333 - accuracy: 0.5152
 4896/25000 [====>.........................] - ETA: 1:07 - loss: 7.4380 - accuracy: 0.5149
 4928/25000 [====>.........................] - ETA: 1:06 - loss: 7.4395 - accuracy: 0.5148
 4960/25000 [====>.........................] - ETA: 1:06 - loss: 7.4379 - accuracy: 0.5149
 4992/25000 [====>.........................] - ETA: 1:06 - loss: 7.4270 - accuracy: 0.5156
 5024/25000 [=====>........................] - ETA: 1:06 - loss: 7.4225 - accuracy: 0.5159
 5056/25000 [=====>........................] - ETA: 1:06 - loss: 7.4149 - accuracy: 0.5164
 5088/25000 [=====>........................] - ETA: 1:06 - loss: 7.4195 - accuracy: 0.5161
 5120/25000 [=====>........................] - ETA: 1:06 - loss: 7.4091 - accuracy: 0.5168
 5152/25000 [=====>........................] - ETA: 1:06 - loss: 7.3898 - accuracy: 0.5181
 5184/25000 [=====>........................] - ETA: 1:05 - loss: 7.3915 - accuracy: 0.5179
 5216/25000 [=====>........................] - ETA: 1:05 - loss: 7.3903 - accuracy: 0.5180
 5248/25000 [=====>........................] - ETA: 1:05 - loss: 7.4007 - accuracy: 0.5173
 5280/25000 [=====>........................] - ETA: 1:05 - loss: 7.4053 - accuracy: 0.5170
 5312/25000 [=====>........................] - ETA: 1:05 - loss: 7.4011 - accuracy: 0.5173
 5344/25000 [=====>........................] - ETA: 1:05 - loss: 7.3940 - accuracy: 0.5178
 5376/25000 [=====>........................] - ETA: 1:05 - loss: 7.3928 - accuracy: 0.5179
 5408/25000 [=====>........................] - ETA: 1:05 - loss: 7.4001 - accuracy: 0.5174
 5440/25000 [=====>........................] - ETA: 1:05 - loss: 7.4017 - accuracy: 0.5173
 5472/25000 [=====>........................] - ETA: 1:04 - loss: 7.4004 - accuracy: 0.5174
 5504/25000 [=====>........................] - ETA: 1:04 - loss: 7.4020 - accuracy: 0.5173
 5536/25000 [=====>........................] - ETA: 1:04 - loss: 7.3980 - accuracy: 0.5175
 5568/25000 [=====>........................] - ETA: 1:04 - loss: 7.4023 - accuracy: 0.5172
 5600/25000 [=====>........................] - ETA: 1:04 - loss: 7.4038 - accuracy: 0.5171
 5632/25000 [=====>........................] - ETA: 1:04 - loss: 7.4189 - accuracy: 0.5162
 5664/25000 [=====>........................] - ETA: 1:04 - loss: 7.4203 - accuracy: 0.5161
 5696/25000 [=====>........................] - ETA: 1:04 - loss: 7.4297 - accuracy: 0.5154
 5728/25000 [=====>........................] - ETA: 1:03 - loss: 7.4391 - accuracy: 0.5148
 5760/25000 [=====>........................] - ETA: 1:03 - loss: 7.4377 - accuracy: 0.5149
 5792/25000 [=====>........................] - ETA: 1:03 - loss: 7.4284 - accuracy: 0.5155
 5824/25000 [=====>........................] - ETA: 1:03 - loss: 7.4244 - accuracy: 0.5158
 5856/25000 [======>.......................] - ETA: 1:03 - loss: 7.4231 - accuracy: 0.5159
 5888/25000 [======>.......................] - ETA: 1:03 - loss: 7.4296 - accuracy: 0.5155
 5920/25000 [======>.......................] - ETA: 1:03 - loss: 7.4309 - accuracy: 0.5154
 5952/25000 [======>.......................] - ETA: 1:03 - loss: 7.4425 - accuracy: 0.5146
 5984/25000 [======>.......................] - ETA: 1:02 - loss: 7.4488 - accuracy: 0.5142
 6016/25000 [======>.......................] - ETA: 1:02 - loss: 7.4474 - accuracy: 0.5143
 6048/25000 [======>.......................] - ETA: 1:02 - loss: 7.4435 - accuracy: 0.5146
 6080/25000 [======>.......................] - ETA: 1:02 - loss: 7.4472 - accuracy: 0.5143
 6112/25000 [======>.......................] - ETA: 1:02 - loss: 7.4459 - accuracy: 0.5144
 6144/25000 [======>.......................] - ETA: 1:02 - loss: 7.4520 - accuracy: 0.5140
 6176/25000 [======>.......................] - ETA: 1:02 - loss: 7.4531 - accuracy: 0.5139
 6208/25000 [======>.......................] - ETA: 1:02 - loss: 7.4690 - accuracy: 0.5129
 6240/25000 [======>.......................] - ETA: 1:02 - loss: 7.4700 - accuracy: 0.5128
 6272/25000 [======>.......................] - ETA: 1:01 - loss: 7.4735 - accuracy: 0.5126
 6304/25000 [======>.......................] - ETA: 1:01 - loss: 7.4769 - accuracy: 0.5124
 6336/25000 [======>.......................] - ETA: 1:01 - loss: 7.4706 - accuracy: 0.5128
 6368/25000 [======>.......................] - ETA: 1:01 - loss: 7.4644 - accuracy: 0.5132
 6400/25000 [======>.......................] - ETA: 1:01 - loss: 7.4606 - accuracy: 0.5134
 6432/25000 [======>.......................] - ETA: 1:01 - loss: 7.4664 - accuracy: 0.5131
 6464/25000 [======>.......................] - ETA: 1:01 - loss: 7.4626 - accuracy: 0.5133
 6496/25000 [======>.......................] - ETA: 1:01 - loss: 7.4707 - accuracy: 0.5128
 6528/25000 [======>.......................] - ETA: 1:01 - loss: 7.4717 - accuracy: 0.5127
 6560/25000 [======>.......................] - ETA: 1:00 - loss: 7.4703 - accuracy: 0.5128
 6592/25000 [======>.......................] - ETA: 1:00 - loss: 7.4643 - accuracy: 0.5132
 6624/25000 [======>.......................] - ETA: 1:00 - loss: 7.4629 - accuracy: 0.5133
 6656/25000 [======>.......................] - ETA: 1:00 - loss: 7.4800 - accuracy: 0.5122
 6688/25000 [=======>......................] - ETA: 1:00 - loss: 7.4695 - accuracy: 0.5129
 6720/25000 [=======>......................] - ETA: 1:00 - loss: 7.4658 - accuracy: 0.5131
 6752/25000 [=======>......................] - ETA: 1:00 - loss: 7.4781 - accuracy: 0.5123
 6784/25000 [=======>......................] - ETA: 1:00 - loss: 7.4835 - accuracy: 0.5119
 6816/25000 [=======>......................] - ETA: 1:00 - loss: 7.4912 - accuracy: 0.5114
 6848/25000 [=======>......................] - ETA: 59s - loss: 7.4920 - accuracy: 0.5114 
 6880/25000 [=======>......................] - ETA: 59s - loss: 7.5062 - accuracy: 0.5105
 6912/25000 [=======>......................] - ETA: 59s - loss: 7.5069 - accuracy: 0.5104
 6944/25000 [=======>......................] - ETA: 59s - loss: 7.5076 - accuracy: 0.5104
 6976/25000 [=======>......................] - ETA: 59s - loss: 7.4996 - accuracy: 0.5109
 7008/25000 [=======>......................] - ETA: 59s - loss: 7.5091 - accuracy: 0.5103
 7040/25000 [=======>......................] - ETA: 59s - loss: 7.4967 - accuracy: 0.5111
 7072/25000 [=======>......................] - ETA: 59s - loss: 7.4845 - accuracy: 0.5119
 7104/25000 [=======>......................] - ETA: 59s - loss: 7.4896 - accuracy: 0.5115
 7136/25000 [=======>......................] - ETA: 59s - loss: 7.4947 - accuracy: 0.5112
 7168/25000 [=======>......................] - ETA: 58s - loss: 7.4934 - accuracy: 0.5113
 7200/25000 [=======>......................] - ETA: 58s - loss: 7.4920 - accuracy: 0.5114
 7232/25000 [=======>......................] - ETA: 58s - loss: 7.4949 - accuracy: 0.5112
 7264/25000 [=======>......................] - ETA: 58s - loss: 7.4935 - accuracy: 0.5113
 7296/25000 [=======>......................] - ETA: 58s - loss: 7.4922 - accuracy: 0.5114
 7328/25000 [=======>......................] - ETA: 58s - loss: 7.4929 - accuracy: 0.5113
 7360/25000 [=======>......................] - ETA: 58s - loss: 7.5041 - accuracy: 0.5106
 7392/25000 [=======>......................] - ETA: 58s - loss: 7.5007 - accuracy: 0.5108
 7424/25000 [=======>......................] - ETA: 58s - loss: 7.4849 - accuracy: 0.5119
 7456/25000 [=======>......................] - ETA: 57s - loss: 7.4898 - accuracy: 0.5115
 7488/25000 [=======>......................] - ETA: 57s - loss: 7.5008 - accuracy: 0.5108
 7520/25000 [========>.....................] - ETA: 57s - loss: 7.5015 - accuracy: 0.5108
 7552/25000 [========>.....................] - ETA: 57s - loss: 7.5103 - accuracy: 0.5102
 7584/25000 [========>.....................] - ETA: 57s - loss: 7.5109 - accuracy: 0.5102
 7616/25000 [========>.....................] - ETA: 57s - loss: 7.5056 - accuracy: 0.5105
 7648/25000 [========>.....................] - ETA: 57s - loss: 7.5082 - accuracy: 0.5103
 7680/25000 [========>.....................] - ETA: 57s - loss: 7.5009 - accuracy: 0.5108
 7712/25000 [========>.....................] - ETA: 56s - loss: 7.5056 - accuracy: 0.5105
 7744/25000 [========>.....................] - ETA: 56s - loss: 7.4983 - accuracy: 0.5110
 7776/25000 [========>.....................] - ETA: 56s - loss: 7.5069 - accuracy: 0.5104
 7808/25000 [========>.....................] - ETA: 56s - loss: 7.5193 - accuracy: 0.5096
 7840/25000 [========>.....................] - ETA: 56s - loss: 7.5121 - accuracy: 0.5101
 7872/25000 [========>.....................] - ETA: 56s - loss: 7.5127 - accuracy: 0.5100
 7904/25000 [========>.....................] - ETA: 56s - loss: 7.5153 - accuracy: 0.5099
 7936/25000 [========>.....................] - ETA: 56s - loss: 7.5198 - accuracy: 0.5096
 7968/25000 [========>.....................] - ETA: 56s - loss: 7.5184 - accuracy: 0.5097
 8000/25000 [========>.....................] - ETA: 55s - loss: 7.5229 - accuracy: 0.5094
 8032/25000 [========>.....................] - ETA: 55s - loss: 7.5254 - accuracy: 0.5092
 8064/25000 [========>.....................] - ETA: 55s - loss: 7.5259 - accuracy: 0.5092
 8096/25000 [========>.....................] - ETA: 55s - loss: 7.5303 - accuracy: 0.5089
 8128/25000 [========>.....................] - ETA: 55s - loss: 7.5365 - accuracy: 0.5085
 8160/25000 [========>.....................] - ETA: 55s - loss: 7.5313 - accuracy: 0.5088
 8192/25000 [========>.....................] - ETA: 55s - loss: 7.5412 - accuracy: 0.5082
 8224/25000 [========>.....................] - ETA: 55s - loss: 7.5436 - accuracy: 0.5080
 8256/25000 [========>.....................] - ETA: 55s - loss: 7.5403 - accuracy: 0.5082
 8288/25000 [========>.....................] - ETA: 54s - loss: 7.5427 - accuracy: 0.5081
 8320/25000 [========>.....................] - ETA: 54s - loss: 7.5413 - accuracy: 0.5082
 8352/25000 [=========>....................] - ETA: 54s - loss: 7.5418 - accuracy: 0.5081
 8384/25000 [=========>....................] - ETA: 54s - loss: 7.5404 - accuracy: 0.5082
 8416/25000 [=========>....................] - ETA: 54s - loss: 7.5446 - accuracy: 0.5080
 8448/25000 [=========>....................] - ETA: 54s - loss: 7.5450 - accuracy: 0.5079
 8480/25000 [=========>....................] - ETA: 54s - loss: 7.5419 - accuracy: 0.5081
 8512/25000 [=========>....................] - ETA: 54s - loss: 7.5495 - accuracy: 0.5076
 8544/25000 [=========>....................] - ETA: 54s - loss: 7.5464 - accuracy: 0.5078
 8576/25000 [=========>....................] - ETA: 53s - loss: 7.5486 - accuracy: 0.5077
 8608/25000 [=========>....................] - ETA: 53s - loss: 7.5580 - accuracy: 0.5071
 8640/25000 [=========>....................] - ETA: 53s - loss: 7.5584 - accuracy: 0.5071
 8672/25000 [=========>....................] - ETA: 53s - loss: 7.5588 - accuracy: 0.5070
 8704/25000 [=========>....................] - ETA: 53s - loss: 7.5644 - accuracy: 0.5067
 8736/25000 [=========>....................] - ETA: 53s - loss: 7.5631 - accuracy: 0.5068
 8768/25000 [=========>....................] - ETA: 53s - loss: 7.5599 - accuracy: 0.5070
 8800/25000 [=========>....................] - ETA: 53s - loss: 7.5656 - accuracy: 0.5066
 8832/25000 [=========>....................] - ETA: 53s - loss: 7.5677 - accuracy: 0.5065
 8864/25000 [=========>....................] - ETA: 53s - loss: 7.5715 - accuracy: 0.5062
 8896/25000 [=========>....................] - ETA: 52s - loss: 7.5753 - accuracy: 0.5060
 8928/25000 [=========>....................] - ETA: 52s - loss: 7.5687 - accuracy: 0.5064
 8960/25000 [=========>....................] - ETA: 52s - loss: 7.5691 - accuracy: 0.5064
 8992/25000 [=========>....................] - ETA: 52s - loss: 7.5694 - accuracy: 0.5063
 9024/25000 [=========>....................] - ETA: 52s - loss: 7.5579 - accuracy: 0.5071
 9056/25000 [=========>....................] - ETA: 52s - loss: 7.5566 - accuracy: 0.5072
 9088/25000 [=========>....................] - ETA: 52s - loss: 7.5721 - accuracy: 0.5062
 9120/25000 [=========>....................] - ETA: 52s - loss: 7.5708 - accuracy: 0.5063
 9152/25000 [=========>....................] - ETA: 52s - loss: 7.5678 - accuracy: 0.5064
 9184/25000 [==========>...................] - ETA: 51s - loss: 7.5648 - accuracy: 0.5066
 9216/25000 [==========>...................] - ETA: 51s - loss: 7.5701 - accuracy: 0.5063
 9248/25000 [==========>...................] - ETA: 51s - loss: 7.5688 - accuracy: 0.5064
 9280/25000 [==========>...................] - ETA: 51s - loss: 7.5724 - accuracy: 0.5061
 9312/25000 [==========>...................] - ETA: 51s - loss: 7.5744 - accuracy: 0.5060
 9344/25000 [==========>...................] - ETA: 51s - loss: 7.5780 - accuracy: 0.5058
 9376/25000 [==========>...................] - ETA: 51s - loss: 7.5799 - accuracy: 0.5057
 9408/25000 [==========>...................] - ETA: 51s - loss: 7.5851 - accuracy: 0.5053
 9440/25000 [==========>...................] - ETA: 51s - loss: 7.5838 - accuracy: 0.5054
 9472/25000 [==========>...................] - ETA: 51s - loss: 7.5873 - accuracy: 0.5052
 9504/25000 [==========>...................] - ETA: 50s - loss: 7.5876 - accuracy: 0.5052
 9536/25000 [==========>...................] - ETA: 50s - loss: 7.5862 - accuracy: 0.5052
 9568/25000 [==========>...................] - ETA: 50s - loss: 7.5881 - accuracy: 0.5051
 9600/25000 [==========>...................] - ETA: 50s - loss: 7.5852 - accuracy: 0.5053
 9632/25000 [==========>...................] - ETA: 50s - loss: 7.5870 - accuracy: 0.5052
 9664/25000 [==========>...................] - ETA: 50s - loss: 7.5841 - accuracy: 0.5054
 9696/25000 [==========>...................] - ETA: 50s - loss: 7.5796 - accuracy: 0.5057
 9728/25000 [==========>...................] - ETA: 50s - loss: 7.5752 - accuracy: 0.5060
 9760/25000 [==========>...................] - ETA: 50s - loss: 7.5755 - accuracy: 0.5059
 9792/25000 [==========>...................] - ETA: 49s - loss: 7.5758 - accuracy: 0.5059
 9824/25000 [==========>...................] - ETA: 49s - loss: 7.5761 - accuracy: 0.5059
 9856/25000 [==========>...................] - ETA: 49s - loss: 7.5779 - accuracy: 0.5058
 9888/25000 [==========>...................] - ETA: 49s - loss: 7.5829 - accuracy: 0.5055
 9920/25000 [==========>...................] - ETA: 49s - loss: 7.5816 - accuracy: 0.5055
 9952/25000 [==========>...................] - ETA: 49s - loss: 7.5865 - accuracy: 0.5052
 9984/25000 [==========>...................] - ETA: 49s - loss: 7.6021 - accuracy: 0.5042
10016/25000 [===========>..................] - ETA: 49s - loss: 7.5977 - accuracy: 0.5045
10048/25000 [===========>..................] - ETA: 49s - loss: 7.5964 - accuracy: 0.5046
10080/25000 [===========>..................] - ETA: 49s - loss: 7.5951 - accuracy: 0.5047
10112/25000 [===========>..................] - ETA: 48s - loss: 7.5954 - accuracy: 0.5046
10144/25000 [===========>..................] - ETA: 48s - loss: 7.5956 - accuracy: 0.5046
10176/25000 [===========>..................] - ETA: 48s - loss: 7.6018 - accuracy: 0.5042
10208/25000 [===========>..................] - ETA: 48s - loss: 7.5975 - accuracy: 0.5045
10240/25000 [===========>..................] - ETA: 48s - loss: 7.6007 - accuracy: 0.5043
10272/25000 [===========>..................] - ETA: 48s - loss: 7.5980 - accuracy: 0.5045
10304/25000 [===========>..................] - ETA: 48s - loss: 7.5952 - accuracy: 0.5047
10336/25000 [===========>..................] - ETA: 48s - loss: 7.5969 - accuracy: 0.5045
10368/25000 [===========>..................] - ETA: 48s - loss: 7.5868 - accuracy: 0.5052
10400/25000 [===========>..................] - ETA: 47s - loss: 7.5870 - accuracy: 0.5052
10432/25000 [===========>..................] - ETA: 47s - loss: 7.5799 - accuracy: 0.5057
10464/25000 [===========>..................] - ETA: 47s - loss: 7.5787 - accuracy: 0.5057
10496/25000 [===========>..................] - ETA: 47s - loss: 7.5775 - accuracy: 0.5058
10528/25000 [===========>..................] - ETA: 47s - loss: 7.5749 - accuracy: 0.5060
10560/25000 [===========>..................] - ETA: 47s - loss: 7.5810 - accuracy: 0.5056
10592/25000 [===========>..................] - ETA: 47s - loss: 7.5870 - accuracy: 0.5052
10624/25000 [===========>..................] - ETA: 47s - loss: 7.5829 - accuracy: 0.5055
10656/25000 [===========>..................] - ETA: 47s - loss: 7.5832 - accuracy: 0.5054
10688/25000 [===========>..................] - ETA: 46s - loss: 7.5848 - accuracy: 0.5053
10720/25000 [===========>..................] - ETA: 46s - loss: 7.5851 - accuracy: 0.5053
10752/25000 [===========>..................] - ETA: 46s - loss: 7.5825 - accuracy: 0.5055
10784/25000 [===========>..................] - ETA: 46s - loss: 7.5856 - accuracy: 0.5053
10816/25000 [===========>..................] - ETA: 46s - loss: 7.5929 - accuracy: 0.5048
10848/25000 [============>.................] - ETA: 46s - loss: 7.5917 - accuracy: 0.5049
10880/25000 [============>.................] - ETA: 46s - loss: 7.5877 - accuracy: 0.5051
10912/25000 [============>.................] - ETA: 46s - loss: 7.5879 - accuracy: 0.5051
10944/25000 [============>.................] - ETA: 46s - loss: 7.5854 - accuracy: 0.5053
10976/25000 [============>.................] - ETA: 45s - loss: 7.5926 - accuracy: 0.5048
11008/25000 [============>.................] - ETA: 45s - loss: 7.5942 - accuracy: 0.5047
11040/25000 [============>.................] - ETA: 45s - loss: 7.5958 - accuracy: 0.5046
11072/25000 [============>.................] - ETA: 45s - loss: 7.6043 - accuracy: 0.5041
11104/25000 [============>.................] - ETA: 45s - loss: 7.6059 - accuracy: 0.5040
11136/25000 [============>.................] - ETA: 45s - loss: 7.6088 - accuracy: 0.5038
11168/25000 [============>.................] - ETA: 45s - loss: 7.6076 - accuracy: 0.5039
11200/25000 [============>.................] - ETA: 45s - loss: 7.6146 - accuracy: 0.5034
11232/25000 [============>.................] - ETA: 45s - loss: 7.6066 - accuracy: 0.5039
11264/25000 [============>.................] - ETA: 44s - loss: 7.6067 - accuracy: 0.5039
11296/25000 [============>.................] - ETA: 44s - loss: 7.6123 - accuracy: 0.5035
11328/25000 [============>.................] - ETA: 44s - loss: 7.6111 - accuracy: 0.5036
11360/25000 [============>.................] - ETA: 44s - loss: 7.6099 - accuracy: 0.5037
11392/25000 [============>.................] - ETA: 44s - loss: 7.6101 - accuracy: 0.5037
11424/25000 [============>.................] - ETA: 44s - loss: 7.6089 - accuracy: 0.5038
11456/25000 [============>.................] - ETA: 44s - loss: 7.6077 - accuracy: 0.5038
11488/25000 [============>.................] - ETA: 44s - loss: 7.6146 - accuracy: 0.5034
11520/25000 [============>.................] - ETA: 44s - loss: 7.6174 - accuracy: 0.5032
11552/25000 [============>.................] - ETA: 43s - loss: 7.6162 - accuracy: 0.5033
11584/25000 [============>.................] - ETA: 43s - loss: 7.6110 - accuracy: 0.5036
11616/25000 [============>.................] - ETA: 43s - loss: 7.6085 - accuracy: 0.5038
11648/25000 [============>.................] - ETA: 43s - loss: 7.6126 - accuracy: 0.5035
11680/25000 [=============>................] - ETA: 43s - loss: 7.6167 - accuracy: 0.5033
11712/25000 [=============>................] - ETA: 43s - loss: 7.6182 - accuracy: 0.5032
11744/25000 [=============>................] - ETA: 43s - loss: 7.6222 - accuracy: 0.5029
11776/25000 [=============>................] - ETA: 43s - loss: 7.6263 - accuracy: 0.5026
11808/25000 [=============>................] - ETA: 43s - loss: 7.6251 - accuracy: 0.5027
11840/25000 [=============>................] - ETA: 43s - loss: 7.6213 - accuracy: 0.5030
11872/25000 [=============>................] - ETA: 42s - loss: 7.6227 - accuracy: 0.5029
11904/25000 [=============>................] - ETA: 42s - loss: 7.6280 - accuracy: 0.5025
11936/25000 [=============>................] - ETA: 42s - loss: 7.6281 - accuracy: 0.5025
11968/25000 [=============>................] - ETA: 42s - loss: 7.6256 - accuracy: 0.5027
12000/25000 [=============>................] - ETA: 42s - loss: 7.6257 - accuracy: 0.5027
12032/25000 [=============>................] - ETA: 42s - loss: 7.6207 - accuracy: 0.5030
12064/25000 [=============>................] - ETA: 42s - loss: 7.6171 - accuracy: 0.5032
12096/25000 [=============>................] - ETA: 42s - loss: 7.6159 - accuracy: 0.5033
12128/25000 [=============>................] - ETA: 42s - loss: 7.6148 - accuracy: 0.5034
12160/25000 [=============>................] - ETA: 41s - loss: 7.6200 - accuracy: 0.5030
12192/25000 [=============>................] - ETA: 41s - loss: 7.6176 - accuracy: 0.5032
12224/25000 [=============>................] - ETA: 41s - loss: 7.6177 - accuracy: 0.5032
12256/25000 [=============>................] - ETA: 41s - loss: 7.6178 - accuracy: 0.5032
12288/25000 [=============>................] - ETA: 41s - loss: 7.6142 - accuracy: 0.5034
12320/25000 [=============>................] - ETA: 41s - loss: 7.6106 - accuracy: 0.5037
12352/25000 [=============>................] - ETA: 41s - loss: 7.6170 - accuracy: 0.5032
12384/25000 [=============>................] - ETA: 41s - loss: 7.6159 - accuracy: 0.5033
12416/25000 [=============>................] - ETA: 41s - loss: 7.6160 - accuracy: 0.5033
12448/25000 [=============>................] - ETA: 40s - loss: 7.6149 - accuracy: 0.5034
12480/25000 [=============>................] - ETA: 40s - loss: 7.6089 - accuracy: 0.5038
12512/25000 [==============>...............] - ETA: 40s - loss: 7.6139 - accuracy: 0.5034
12544/25000 [==============>...............] - ETA: 40s - loss: 7.6141 - accuracy: 0.5034
12576/25000 [==============>...............] - ETA: 40s - loss: 7.6105 - accuracy: 0.5037
12608/25000 [==============>...............] - ETA: 40s - loss: 7.6143 - accuracy: 0.5034
12640/25000 [==============>...............] - ETA: 40s - loss: 7.6193 - accuracy: 0.5031
12672/25000 [==============>...............] - ETA: 40s - loss: 7.6182 - accuracy: 0.5032
12704/25000 [==============>...............] - ETA: 40s - loss: 7.6208 - accuracy: 0.5030
12736/25000 [==============>...............] - ETA: 40s - loss: 7.6136 - accuracy: 0.5035
12768/25000 [==============>...............] - ETA: 40s - loss: 7.6126 - accuracy: 0.5035
12800/25000 [==============>...............] - ETA: 39s - loss: 7.6163 - accuracy: 0.5033
12832/25000 [==============>...............] - ETA: 39s - loss: 7.6140 - accuracy: 0.5034
12864/25000 [==============>...............] - ETA: 39s - loss: 7.6106 - accuracy: 0.5037
12896/25000 [==============>...............] - ETA: 39s - loss: 7.6143 - accuracy: 0.5034
12928/25000 [==============>...............] - ETA: 39s - loss: 7.6156 - accuracy: 0.5033
12960/25000 [==============>...............] - ETA: 39s - loss: 7.6122 - accuracy: 0.5035
12992/25000 [==============>...............] - ETA: 39s - loss: 7.6076 - accuracy: 0.5038
13024/25000 [==============>...............] - ETA: 39s - loss: 7.6078 - accuracy: 0.5038
13056/25000 [==============>...............] - ETA: 39s - loss: 7.6055 - accuracy: 0.5040
13088/25000 [==============>...............] - ETA: 38s - loss: 7.6034 - accuracy: 0.5041
13120/25000 [==============>...............] - ETA: 38s - loss: 7.6035 - accuracy: 0.5041
13152/25000 [==============>...............] - ETA: 38s - loss: 7.6107 - accuracy: 0.5036
13184/25000 [==============>...............] - ETA: 38s - loss: 7.6108 - accuracy: 0.5036
13216/25000 [==============>...............] - ETA: 38s - loss: 7.6098 - accuracy: 0.5037
13248/25000 [==============>...............] - ETA: 38s - loss: 7.6134 - accuracy: 0.5035
13280/25000 [==============>...............] - ETA: 38s - loss: 7.6112 - accuracy: 0.5036
13312/25000 [==============>...............] - ETA: 38s - loss: 7.6079 - accuracy: 0.5038
13344/25000 [===============>..............] - ETA: 38s - loss: 7.6126 - accuracy: 0.5035
13376/25000 [===============>..............] - ETA: 38s - loss: 7.6162 - accuracy: 0.5033
13408/25000 [===============>..............] - ETA: 37s - loss: 7.6197 - accuracy: 0.5031
13440/25000 [===============>..............] - ETA: 37s - loss: 7.6176 - accuracy: 0.5032
13472/25000 [===============>..............] - ETA: 37s - loss: 7.6165 - accuracy: 0.5033
13504/25000 [===============>..............] - ETA: 37s - loss: 7.6212 - accuracy: 0.5030
13536/25000 [===============>..............] - ETA: 37s - loss: 7.6247 - accuracy: 0.5027
13568/25000 [===============>..............] - ETA: 37s - loss: 7.6225 - accuracy: 0.5029
13600/25000 [===============>..............] - ETA: 37s - loss: 7.6226 - accuracy: 0.5029
13632/25000 [===============>..............] - ETA: 37s - loss: 7.6228 - accuracy: 0.5029
13664/25000 [===============>..............] - ETA: 37s - loss: 7.6251 - accuracy: 0.5027
13696/25000 [===============>..............] - ETA: 36s - loss: 7.6297 - accuracy: 0.5024
13728/25000 [===============>..............] - ETA: 36s - loss: 7.6275 - accuracy: 0.5025
13760/25000 [===============>..............] - ETA: 36s - loss: 7.6287 - accuracy: 0.5025
13792/25000 [===============>..............] - ETA: 36s - loss: 7.6299 - accuracy: 0.5024
13824/25000 [===============>..............] - ETA: 36s - loss: 7.6234 - accuracy: 0.5028
13856/25000 [===============>..............] - ETA: 36s - loss: 7.6290 - accuracy: 0.5025
13888/25000 [===============>..............] - ETA: 36s - loss: 7.6291 - accuracy: 0.5024
13920/25000 [===============>..............] - ETA: 36s - loss: 7.6303 - accuracy: 0.5024
13952/25000 [===============>..............] - ETA: 36s - loss: 7.6304 - accuracy: 0.5024
13984/25000 [===============>..............] - ETA: 36s - loss: 7.6271 - accuracy: 0.5026
14016/25000 [===============>..............] - ETA: 35s - loss: 7.6250 - accuracy: 0.5027
14048/25000 [===============>..............] - ETA: 35s - loss: 7.6262 - accuracy: 0.5026
14080/25000 [===============>..............] - ETA: 35s - loss: 7.6231 - accuracy: 0.5028
14112/25000 [===============>..............] - ETA: 35s - loss: 7.6242 - accuracy: 0.5028
14144/25000 [===============>..............] - ETA: 35s - loss: 7.6287 - accuracy: 0.5025
14176/25000 [================>.............] - ETA: 35s - loss: 7.6288 - accuracy: 0.5025
14208/25000 [================>.............] - ETA: 35s - loss: 7.6288 - accuracy: 0.5025
14240/25000 [================>.............] - ETA: 35s - loss: 7.6300 - accuracy: 0.5024
14272/25000 [================>.............] - ETA: 35s - loss: 7.6333 - accuracy: 0.5022
14304/25000 [================>.............] - ETA: 35s - loss: 7.6355 - accuracy: 0.5020
14336/25000 [================>.............] - ETA: 34s - loss: 7.6292 - accuracy: 0.5024
14368/25000 [================>.............] - ETA: 34s - loss: 7.6314 - accuracy: 0.5023
14400/25000 [================>.............] - ETA: 34s - loss: 7.6325 - accuracy: 0.5022
14432/25000 [================>.............] - ETA: 34s - loss: 7.6390 - accuracy: 0.5018
14464/25000 [================>.............] - ETA: 34s - loss: 7.6348 - accuracy: 0.5021
14496/25000 [================>.............] - ETA: 34s - loss: 7.6370 - accuracy: 0.5019
14528/25000 [================>.............] - ETA: 34s - loss: 7.6360 - accuracy: 0.5020
14560/25000 [================>.............] - ETA: 34s - loss: 7.6382 - accuracy: 0.5019
14592/25000 [================>.............] - ETA: 34s - loss: 7.6414 - accuracy: 0.5016
14624/25000 [================>.............] - ETA: 34s - loss: 7.6436 - accuracy: 0.5015
14656/25000 [================>.............] - ETA: 33s - loss: 7.6415 - accuracy: 0.5016
14688/25000 [================>.............] - ETA: 33s - loss: 7.6447 - accuracy: 0.5014
14720/25000 [================>.............] - ETA: 33s - loss: 7.6437 - accuracy: 0.5015
14752/25000 [================>.............] - ETA: 33s - loss: 7.6448 - accuracy: 0.5014
14784/25000 [================>.............] - ETA: 33s - loss: 7.6428 - accuracy: 0.5016
14816/25000 [================>.............] - ETA: 33s - loss: 7.6428 - accuracy: 0.5016
14848/25000 [================>.............] - ETA: 33s - loss: 7.6480 - accuracy: 0.5012
14880/25000 [================>.............] - ETA: 33s - loss: 7.6439 - accuracy: 0.5015
14912/25000 [================>.............] - ETA: 33s - loss: 7.6471 - accuracy: 0.5013
14944/25000 [================>.............] - ETA: 32s - loss: 7.6502 - accuracy: 0.5011
14976/25000 [================>.............] - ETA: 32s - loss: 7.6513 - accuracy: 0.5010
15008/25000 [=================>............] - ETA: 32s - loss: 7.6503 - accuracy: 0.5011
15040/25000 [=================>............] - ETA: 32s - loss: 7.6554 - accuracy: 0.5007
15072/25000 [=================>............] - ETA: 32s - loss: 7.6605 - accuracy: 0.5004
15104/25000 [=================>............] - ETA: 32s - loss: 7.6626 - accuracy: 0.5003
15136/25000 [=================>............] - ETA: 32s - loss: 7.6616 - accuracy: 0.5003
15168/25000 [=================>............] - ETA: 32s - loss: 7.6575 - accuracy: 0.5006
15200/25000 [=================>............] - ETA: 32s - loss: 7.6585 - accuracy: 0.5005
15232/25000 [=================>............] - ETA: 32s - loss: 7.6626 - accuracy: 0.5003
15264/25000 [=================>............] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
15296/25000 [=================>............] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
15328/25000 [=================>............] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
15360/25000 [=================>............] - ETA: 31s - loss: 7.6646 - accuracy: 0.5001
15392/25000 [=================>............] - ETA: 31s - loss: 7.6646 - accuracy: 0.5001
15424/25000 [=================>............] - ETA: 31s - loss: 7.6616 - accuracy: 0.5003
15456/25000 [=================>............] - ETA: 31s - loss: 7.6607 - accuracy: 0.5004
15488/25000 [=================>............] - ETA: 31s - loss: 7.6627 - accuracy: 0.5003
15520/25000 [=================>............] - ETA: 31s - loss: 7.6656 - accuracy: 0.5001
15552/25000 [=================>............] - ETA: 30s - loss: 7.6627 - accuracy: 0.5003
15584/25000 [=================>............] - ETA: 30s - loss: 7.6607 - accuracy: 0.5004
15616/25000 [=================>............] - ETA: 30s - loss: 7.6607 - accuracy: 0.5004
15648/25000 [=================>............] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
15680/25000 [=================>............] - ETA: 30s - loss: 7.6647 - accuracy: 0.5001
15712/25000 [=================>............] - ETA: 30s - loss: 7.6666 - accuracy: 0.5000
15744/25000 [=================>............] - ETA: 30s - loss: 7.6656 - accuracy: 0.5001
15776/25000 [=================>............] - ETA: 30s - loss: 7.6676 - accuracy: 0.4999
15808/25000 [=================>............] - ETA: 30s - loss: 7.6695 - accuracy: 0.4998
15840/25000 [==================>...........] - ETA: 30s - loss: 7.6715 - accuracy: 0.4997
15872/25000 [==================>...........] - ETA: 29s - loss: 7.6724 - accuracy: 0.4996
15904/25000 [==================>...........] - ETA: 29s - loss: 7.6772 - accuracy: 0.4993
15936/25000 [==================>...........] - ETA: 29s - loss: 7.6782 - accuracy: 0.4992
15968/25000 [==================>...........] - ETA: 29s - loss: 7.6781 - accuracy: 0.4992
16000/25000 [==================>...........] - ETA: 29s - loss: 7.6800 - accuracy: 0.4991
16032/25000 [==================>...........] - ETA: 29s - loss: 7.6781 - accuracy: 0.4993
16064/25000 [==================>...........] - ETA: 29s - loss: 7.6819 - accuracy: 0.4990
16096/25000 [==================>...........] - ETA: 29s - loss: 7.6761 - accuracy: 0.4994
16128/25000 [==================>...........] - ETA: 29s - loss: 7.6761 - accuracy: 0.4994
16160/25000 [==================>...........] - ETA: 28s - loss: 7.6771 - accuracy: 0.4993
16192/25000 [==================>...........] - ETA: 28s - loss: 7.6827 - accuracy: 0.4990
16224/25000 [==================>...........] - ETA: 28s - loss: 7.6855 - accuracy: 0.4988
16256/25000 [==================>...........] - ETA: 28s - loss: 7.6874 - accuracy: 0.4986
16288/25000 [==================>...........] - ETA: 28s - loss: 7.6854 - accuracy: 0.4988
16320/25000 [==================>...........] - ETA: 28s - loss: 7.6826 - accuracy: 0.4990
16352/25000 [==================>...........] - ETA: 28s - loss: 7.6835 - accuracy: 0.4989
16384/25000 [==================>...........] - ETA: 28s - loss: 7.6797 - accuracy: 0.4991
16416/25000 [==================>...........] - ETA: 28s - loss: 7.6760 - accuracy: 0.4994
16448/25000 [==================>...........] - ETA: 28s - loss: 7.6778 - accuracy: 0.4993
16480/25000 [==================>...........] - ETA: 27s - loss: 7.6703 - accuracy: 0.4998
16512/25000 [==================>...........] - ETA: 27s - loss: 7.6648 - accuracy: 0.5001
16544/25000 [==================>...........] - ETA: 27s - loss: 7.6592 - accuracy: 0.5005
16576/25000 [==================>...........] - ETA: 27s - loss: 7.6574 - accuracy: 0.5006
16608/25000 [==================>...........] - ETA: 27s - loss: 7.6555 - accuracy: 0.5007
16640/25000 [==================>...........] - ETA: 27s - loss: 7.6583 - accuracy: 0.5005
16672/25000 [===================>..........] - ETA: 27s - loss: 7.6602 - accuracy: 0.5004
16704/25000 [===================>..........] - ETA: 27s - loss: 7.6584 - accuracy: 0.5005
16736/25000 [===================>..........] - ETA: 27s - loss: 7.6584 - accuracy: 0.5005
16768/25000 [===================>..........] - ETA: 27s - loss: 7.6575 - accuracy: 0.5006
16800/25000 [===================>..........] - ETA: 26s - loss: 7.6566 - accuracy: 0.5007
16832/25000 [===================>..........] - ETA: 26s - loss: 7.6602 - accuracy: 0.5004
16864/25000 [===================>..........] - ETA: 26s - loss: 7.6575 - accuracy: 0.5006
16896/25000 [===================>..........] - ETA: 26s - loss: 7.6557 - accuracy: 0.5007
16928/25000 [===================>..........] - ETA: 26s - loss: 7.6539 - accuracy: 0.5008
16960/25000 [===================>..........] - ETA: 26s - loss: 7.6503 - accuracy: 0.5011
16992/25000 [===================>..........] - ETA: 26s - loss: 7.6459 - accuracy: 0.5014
17024/25000 [===================>..........] - ETA: 26s - loss: 7.6468 - accuracy: 0.5013
17056/25000 [===================>..........] - ETA: 26s - loss: 7.6486 - accuracy: 0.5012
17088/25000 [===================>..........] - ETA: 26s - loss: 7.6487 - accuracy: 0.5012
17120/25000 [===================>..........] - ETA: 25s - loss: 7.6478 - accuracy: 0.5012
17152/25000 [===================>..........] - ETA: 25s - loss: 7.6496 - accuracy: 0.5011
17184/25000 [===================>..........] - ETA: 25s - loss: 7.6506 - accuracy: 0.5010
17216/25000 [===================>..........] - ETA: 25s - loss: 7.6479 - accuracy: 0.5012
17248/25000 [===================>..........] - ETA: 25s - loss: 7.6462 - accuracy: 0.5013
17280/25000 [===================>..........] - ETA: 25s - loss: 7.6435 - accuracy: 0.5015
17312/25000 [===================>..........] - ETA: 25s - loss: 7.6418 - accuracy: 0.5016
17344/25000 [===================>..........] - ETA: 25s - loss: 7.6427 - accuracy: 0.5016
17376/25000 [===================>..........] - ETA: 25s - loss: 7.6419 - accuracy: 0.5016
17408/25000 [===================>..........] - ETA: 25s - loss: 7.6384 - accuracy: 0.5018
17440/25000 [===================>..........] - ETA: 24s - loss: 7.6402 - accuracy: 0.5017
17472/25000 [===================>..........] - ETA: 24s - loss: 7.6420 - accuracy: 0.5016
17504/25000 [====================>.........] - ETA: 24s - loss: 7.6412 - accuracy: 0.5017
17536/25000 [====================>.........] - ETA: 24s - loss: 7.6439 - accuracy: 0.5015
17568/25000 [====================>.........] - ETA: 24s - loss: 7.6422 - accuracy: 0.5016
17600/25000 [====================>.........] - ETA: 24s - loss: 7.6422 - accuracy: 0.5016
17632/25000 [====================>.........] - ETA: 24s - loss: 7.6371 - accuracy: 0.5019
17664/25000 [====================>.........] - ETA: 24s - loss: 7.6380 - accuracy: 0.5019
17696/25000 [====================>.........] - ETA: 24s - loss: 7.6380 - accuracy: 0.5019
17728/25000 [====================>.........] - ETA: 23s - loss: 7.6355 - accuracy: 0.5020
17760/25000 [====================>.........] - ETA: 23s - loss: 7.6373 - accuracy: 0.5019
17792/25000 [====================>.........] - ETA: 23s - loss: 7.6330 - accuracy: 0.5022
17824/25000 [====================>.........] - ETA: 23s - loss: 7.6296 - accuracy: 0.5024
17856/25000 [====================>.........] - ETA: 23s - loss: 7.6331 - accuracy: 0.5022
17888/25000 [====================>.........] - ETA: 23s - loss: 7.6340 - accuracy: 0.5021
17920/25000 [====================>.........] - ETA: 23s - loss: 7.6409 - accuracy: 0.5017
17952/25000 [====================>.........] - ETA: 23s - loss: 7.6410 - accuracy: 0.5017
17984/25000 [====================>.........] - ETA: 23s - loss: 7.6427 - accuracy: 0.5016
18016/25000 [====================>.........] - ETA: 23s - loss: 7.6445 - accuracy: 0.5014
18048/25000 [====================>.........] - ETA: 22s - loss: 7.6454 - accuracy: 0.5014
18080/25000 [====================>.........] - ETA: 22s - loss: 7.6429 - accuracy: 0.5015
18112/25000 [====================>.........] - ETA: 22s - loss: 7.6455 - accuracy: 0.5014
18144/25000 [====================>.........] - ETA: 22s - loss: 7.6438 - accuracy: 0.5015
18176/25000 [====================>.........] - ETA: 22s - loss: 7.6413 - accuracy: 0.5017
18208/25000 [====================>.........] - ETA: 22s - loss: 7.6447 - accuracy: 0.5014
18240/25000 [====================>.........] - ETA: 22s - loss: 7.6481 - accuracy: 0.5012
18272/25000 [====================>.........] - ETA: 22s - loss: 7.6507 - accuracy: 0.5010
18304/25000 [====================>.........] - ETA: 22s - loss: 7.6465 - accuracy: 0.5013
18336/25000 [=====================>........] - ETA: 21s - loss: 7.6457 - accuracy: 0.5014
18368/25000 [=====================>........] - ETA: 21s - loss: 7.6466 - accuracy: 0.5013
18400/25000 [=====================>........] - ETA: 21s - loss: 7.6483 - accuracy: 0.5012
18432/25000 [=====================>........] - ETA: 21s - loss: 7.6467 - accuracy: 0.5013
18464/25000 [=====================>........] - ETA: 21s - loss: 7.6483 - accuracy: 0.5012
18496/25000 [=====================>........] - ETA: 21s - loss: 7.6451 - accuracy: 0.5014
18528/25000 [=====================>........] - ETA: 21s - loss: 7.6468 - accuracy: 0.5013
18560/25000 [=====================>........] - ETA: 21s - loss: 7.6443 - accuracy: 0.5015
18592/25000 [=====================>........] - ETA: 21s - loss: 7.6452 - accuracy: 0.5014
18624/25000 [=====================>........] - ETA: 21s - loss: 7.6452 - accuracy: 0.5014
18656/25000 [=====================>........] - ETA: 20s - loss: 7.6411 - accuracy: 0.5017
18688/25000 [=====================>........] - ETA: 20s - loss: 7.6461 - accuracy: 0.5013
18720/25000 [=====================>........] - ETA: 20s - loss: 7.6486 - accuracy: 0.5012
18752/25000 [=====================>........] - ETA: 20s - loss: 7.6462 - accuracy: 0.5013
18784/25000 [=====================>........] - ETA: 20s - loss: 7.6478 - accuracy: 0.5012
18816/25000 [=====================>........] - ETA: 20s - loss: 7.6471 - accuracy: 0.5013
18848/25000 [=====================>........] - ETA: 20s - loss: 7.6414 - accuracy: 0.5016
18880/25000 [=====================>........] - ETA: 20s - loss: 7.6414 - accuracy: 0.5016
18912/25000 [=====================>........] - ETA: 20s - loss: 7.6415 - accuracy: 0.5016
18944/25000 [=====================>........] - ETA: 19s - loss: 7.6440 - accuracy: 0.5015
18976/25000 [=====================>........] - ETA: 19s - loss: 7.6408 - accuracy: 0.5017
19008/25000 [=====================>........] - ETA: 19s - loss: 7.6408 - accuracy: 0.5017
19040/25000 [=====================>........] - ETA: 19s - loss: 7.6425 - accuracy: 0.5016
19072/25000 [=====================>........] - ETA: 19s - loss: 7.6401 - accuracy: 0.5017
19104/25000 [=====================>........] - ETA: 19s - loss: 7.6417 - accuracy: 0.5016
19136/25000 [=====================>........] - ETA: 19s - loss: 7.6426 - accuracy: 0.5016
19168/25000 [======================>.......] - ETA: 19s - loss: 7.6394 - accuracy: 0.5018
19200/25000 [======================>.......] - ETA: 19s - loss: 7.6395 - accuracy: 0.5018
19232/25000 [======================>.......] - ETA: 19s - loss: 7.6419 - accuracy: 0.5016
19264/25000 [======================>.......] - ETA: 18s - loss: 7.6435 - accuracy: 0.5015
19296/25000 [======================>.......] - ETA: 18s - loss: 7.6444 - accuracy: 0.5015
19328/25000 [======================>.......] - ETA: 18s - loss: 7.6444 - accuracy: 0.5014
19360/25000 [======================>.......] - ETA: 18s - loss: 7.6421 - accuracy: 0.5016
19392/25000 [======================>.......] - ETA: 18s - loss: 7.6413 - accuracy: 0.5017
19424/25000 [======================>.......] - ETA: 18s - loss: 7.6390 - accuracy: 0.5018
19456/25000 [======================>.......] - ETA: 18s - loss: 7.6406 - accuracy: 0.5017
19488/25000 [======================>.......] - ETA: 18s - loss: 7.6367 - accuracy: 0.5019
19520/25000 [======================>.......] - ETA: 18s - loss: 7.6407 - accuracy: 0.5017
19552/25000 [======================>.......] - ETA: 18s - loss: 7.6439 - accuracy: 0.5015
19584/25000 [======================>.......] - ETA: 17s - loss: 7.6439 - accuracy: 0.5015
19616/25000 [======================>.......] - ETA: 17s - loss: 7.6440 - accuracy: 0.5015
19648/25000 [======================>.......] - ETA: 17s - loss: 7.6416 - accuracy: 0.5016
19680/25000 [======================>.......] - ETA: 17s - loss: 7.6409 - accuracy: 0.5017
19712/25000 [======================>.......] - ETA: 17s - loss: 7.6417 - accuracy: 0.5016
19744/25000 [======================>.......] - ETA: 17s - loss: 7.6410 - accuracy: 0.5017
19776/25000 [======================>.......] - ETA: 17s - loss: 7.6418 - accuracy: 0.5016
19808/25000 [======================>.......] - ETA: 17s - loss: 7.6418 - accuracy: 0.5016
19840/25000 [======================>.......] - ETA: 17s - loss: 7.6411 - accuracy: 0.5017
19872/25000 [======================>.......] - ETA: 16s - loss: 7.6419 - accuracy: 0.5016
19904/25000 [======================>.......] - ETA: 16s - loss: 7.6443 - accuracy: 0.5015
19936/25000 [======================>.......] - ETA: 16s - loss: 7.6420 - accuracy: 0.5016
19968/25000 [======================>.......] - ETA: 16s - loss: 7.6413 - accuracy: 0.5017
20000/25000 [=======================>......] - ETA: 16s - loss: 7.6421 - accuracy: 0.5016
20032/25000 [=======================>......] - ETA: 16s - loss: 7.6414 - accuracy: 0.5016
20064/25000 [=======================>......] - ETA: 16s - loss: 7.6437 - accuracy: 0.5015
20096/25000 [=======================>......] - ETA: 16s - loss: 7.6430 - accuracy: 0.5015
20128/25000 [=======================>......] - ETA: 16s - loss: 7.6415 - accuracy: 0.5016
20160/25000 [=======================>......] - ETA: 16s - loss: 7.6438 - accuracy: 0.5015
20192/25000 [=======================>......] - ETA: 15s - loss: 7.6476 - accuracy: 0.5012
20224/25000 [=======================>......] - ETA: 15s - loss: 7.6454 - accuracy: 0.5014
20256/25000 [=======================>......] - ETA: 15s - loss: 7.6447 - accuracy: 0.5014
20288/25000 [=======================>......] - ETA: 15s - loss: 7.6394 - accuracy: 0.5018
20320/25000 [=======================>......] - ETA: 15s - loss: 7.6379 - accuracy: 0.5019
20352/25000 [=======================>......] - ETA: 15s - loss: 7.6403 - accuracy: 0.5017
20384/25000 [=======================>......] - ETA: 15s - loss: 7.6395 - accuracy: 0.5018
20416/25000 [=======================>......] - ETA: 15s - loss: 7.6411 - accuracy: 0.5017
20448/25000 [=======================>......] - ETA: 15s - loss: 7.6381 - accuracy: 0.5019
20480/25000 [=======================>......] - ETA: 14s - loss: 7.6382 - accuracy: 0.5019
20512/25000 [=======================>......] - ETA: 14s - loss: 7.6434 - accuracy: 0.5015
20544/25000 [=======================>......] - ETA: 14s - loss: 7.6450 - accuracy: 0.5014
20576/25000 [=======================>......] - ETA: 14s - loss: 7.6465 - accuracy: 0.5013
20608/25000 [=======================>......] - ETA: 14s - loss: 7.6465 - accuracy: 0.5013
20640/25000 [=======================>......] - ETA: 14s - loss: 7.6466 - accuracy: 0.5013
20672/25000 [=======================>......] - ETA: 14s - loss: 7.6473 - accuracy: 0.5013
20704/25000 [=======================>......] - ETA: 14s - loss: 7.6488 - accuracy: 0.5012
20736/25000 [=======================>......] - ETA: 14s - loss: 7.6489 - accuracy: 0.5012
20768/25000 [=======================>......] - ETA: 13s - loss: 7.6474 - accuracy: 0.5013
20800/25000 [=======================>......] - ETA: 13s - loss: 7.6467 - accuracy: 0.5013
20832/25000 [=======================>......] - ETA: 13s - loss: 7.6482 - accuracy: 0.5012
20864/25000 [========================>.....] - ETA: 13s - loss: 7.6468 - accuracy: 0.5013
20896/25000 [========================>.....] - ETA: 13s - loss: 7.6505 - accuracy: 0.5011
20928/25000 [========================>.....] - ETA: 13s - loss: 7.6512 - accuracy: 0.5010
20960/25000 [========================>.....] - ETA: 13s - loss: 7.6527 - accuracy: 0.5009
20992/25000 [========================>.....] - ETA: 13s - loss: 7.6505 - accuracy: 0.5010
21024/25000 [========================>.....] - ETA: 13s - loss: 7.6528 - accuracy: 0.5009
21056/25000 [========================>.....] - ETA: 13s - loss: 7.6535 - accuracy: 0.5009
21088/25000 [========================>.....] - ETA: 12s - loss: 7.6521 - accuracy: 0.5009
21120/25000 [========================>.....] - ETA: 12s - loss: 7.6536 - accuracy: 0.5009
21152/25000 [========================>.....] - ETA: 12s - loss: 7.6536 - accuracy: 0.5009
21184/25000 [========================>.....] - ETA: 12s - loss: 7.6536 - accuracy: 0.5008
21216/25000 [========================>.....] - ETA: 12s - loss: 7.6551 - accuracy: 0.5008
21248/25000 [========================>.....] - ETA: 12s - loss: 7.6572 - accuracy: 0.5006
21280/25000 [========================>.....] - ETA: 12s - loss: 7.6551 - accuracy: 0.5008
21312/25000 [========================>.....] - ETA: 12s - loss: 7.6558 - accuracy: 0.5007
21344/25000 [========================>.....] - ETA: 12s - loss: 7.6566 - accuracy: 0.5007
21376/25000 [========================>.....] - ETA: 11s - loss: 7.6580 - accuracy: 0.5006
21408/25000 [========================>.....] - ETA: 11s - loss: 7.6566 - accuracy: 0.5007
21440/25000 [========================>.....] - ETA: 11s - loss: 7.6573 - accuracy: 0.5006
21472/25000 [========================>.....] - ETA: 11s - loss: 7.6595 - accuracy: 0.5005
21504/25000 [========================>.....] - ETA: 11s - loss: 7.6602 - accuracy: 0.5004
21536/25000 [========================>.....] - ETA: 11s - loss: 7.6595 - accuracy: 0.5005
21568/25000 [========================>.....] - ETA: 11s - loss: 7.6588 - accuracy: 0.5005
21600/25000 [========================>.....] - ETA: 11s - loss: 7.6567 - accuracy: 0.5006
21632/25000 [========================>.....] - ETA: 11s - loss: 7.6588 - accuracy: 0.5005
21664/25000 [========================>.....] - ETA: 11s - loss: 7.6539 - accuracy: 0.5008
21696/25000 [=========================>....] - ETA: 10s - loss: 7.6539 - accuracy: 0.5008
21728/25000 [=========================>....] - ETA: 10s - loss: 7.6518 - accuracy: 0.5010
21760/25000 [=========================>....] - ETA: 10s - loss: 7.6525 - accuracy: 0.5009
21792/25000 [=========================>....] - ETA: 10s - loss: 7.6511 - accuracy: 0.5010
21824/25000 [=========================>....] - ETA: 10s - loss: 7.6469 - accuracy: 0.5013
21856/25000 [=========================>....] - ETA: 10s - loss: 7.6477 - accuracy: 0.5012
21888/25000 [=========================>....] - ETA: 10s - loss: 7.6498 - accuracy: 0.5011
21920/25000 [=========================>....] - ETA: 10s - loss: 7.6505 - accuracy: 0.5010
21952/25000 [=========================>....] - ETA: 10s - loss: 7.6513 - accuracy: 0.5010
21984/25000 [=========================>....] - ETA: 9s - loss: 7.6499 - accuracy: 0.5011 
22016/25000 [=========================>....] - ETA: 9s - loss: 7.6520 - accuracy: 0.5010
22048/25000 [=========================>....] - ETA: 9s - loss: 7.6520 - accuracy: 0.5010
22080/25000 [=========================>....] - ETA: 9s - loss: 7.6513 - accuracy: 0.5010
22112/25000 [=========================>....] - ETA: 9s - loss: 7.6514 - accuracy: 0.5010
22144/25000 [=========================>....] - ETA: 9s - loss: 7.6507 - accuracy: 0.5010
22176/25000 [=========================>....] - ETA: 9s - loss: 7.6507 - accuracy: 0.5010
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6507 - accuracy: 0.5010
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6515 - accuracy: 0.5010
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6480 - accuracy: 0.5012
22304/25000 [=========================>....] - ETA: 8s - loss: 7.6508 - accuracy: 0.5010
22336/25000 [=========================>....] - ETA: 8s - loss: 7.6474 - accuracy: 0.5013
22368/25000 [=========================>....] - ETA: 8s - loss: 7.6461 - accuracy: 0.5013
22400/25000 [=========================>....] - ETA: 8s - loss: 7.6481 - accuracy: 0.5012
22432/25000 [=========================>....] - ETA: 8s - loss: 7.6482 - accuracy: 0.5012
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6468 - accuracy: 0.5013
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6482 - accuracy: 0.5012
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6455 - accuracy: 0.5014
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6449 - accuracy: 0.5014
22592/25000 [==========================>...] - ETA: 7s - loss: 7.6463 - accuracy: 0.5013
22624/25000 [==========================>...] - ETA: 7s - loss: 7.6456 - accuracy: 0.5014
22656/25000 [==========================>...] - ETA: 7s - loss: 7.6423 - accuracy: 0.5016
22688/25000 [==========================>...] - ETA: 7s - loss: 7.6416 - accuracy: 0.5016
22720/25000 [==========================>...] - ETA: 7s - loss: 7.6423 - accuracy: 0.5016
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6424 - accuracy: 0.5016
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6437 - accuracy: 0.5015
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6458 - accuracy: 0.5014
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6472 - accuracy: 0.5013
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6458 - accuracy: 0.5014
22912/25000 [==========================>...] - ETA: 6s - loss: 7.6472 - accuracy: 0.5013
22944/25000 [==========================>...] - ETA: 6s - loss: 7.6479 - accuracy: 0.5012
22976/25000 [==========================>...] - ETA: 6s - loss: 7.6466 - accuracy: 0.5013
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6480 - accuracy: 0.5012
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6506 - accuracy: 0.5010
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6520 - accuracy: 0.5010
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6534 - accuracy: 0.5009
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6527 - accuracy: 0.5009
23200/25000 [==========================>...] - ETA: 5s - loss: 7.6534 - accuracy: 0.5009
23232/25000 [==========================>...] - ETA: 5s - loss: 7.6547 - accuracy: 0.5008
23264/25000 [==========================>...] - ETA: 5s - loss: 7.6541 - accuracy: 0.5008
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6548 - accuracy: 0.5008
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6548 - accuracy: 0.5008
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6581 - accuracy: 0.5006
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6548 - accuracy: 0.5008
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6522 - accuracy: 0.5009
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6522 - accuracy: 0.5009
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6523 - accuracy: 0.5009
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6510 - accuracy: 0.5010
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6510 - accuracy: 0.5010
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6491 - accuracy: 0.5011
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6484 - accuracy: 0.5012
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6511 - accuracy: 0.5010
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6524 - accuracy: 0.5009
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6550 - accuracy: 0.5008
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6537 - accuracy: 0.5008
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6550 - accuracy: 0.5008
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6550 - accuracy: 0.5008
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6518 - accuracy: 0.5010
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6512 - accuracy: 0.5010
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6544 - accuracy: 0.5008
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6544 - accuracy: 0.5008
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6564 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6590 - accuracy: 0.5005
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6577 - accuracy: 0.5006
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6552 - accuracy: 0.5007
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6577 - accuracy: 0.5006
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6558 - accuracy: 0.5007
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6571 - accuracy: 0.5006
24192/25000 [============================>.] - ETA: 2s - loss: 7.6558 - accuracy: 0.5007
24224/25000 [============================>.] - ETA: 2s - loss: 7.6578 - accuracy: 0.5006
24256/25000 [============================>.] - ETA: 2s - loss: 7.6571 - accuracy: 0.5006
24288/25000 [============================>.] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
24320/25000 [============================>.] - ETA: 2s - loss: 7.6635 - accuracy: 0.5002
24352/25000 [============================>.] - ETA: 2s - loss: 7.6628 - accuracy: 0.5002
24384/25000 [============================>.] - ETA: 2s - loss: 7.6635 - accuracy: 0.5002
24416/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24448/25000 [============================>.] - ETA: 1s - loss: 7.6629 - accuracy: 0.5002
24480/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24512/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24544/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24576/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24608/25000 [============================>.] - ETA: 1s - loss: 7.6648 - accuracy: 0.5001
24640/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24672/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24704/25000 [============================>.] - ETA: 0s - loss: 7.6703 - accuracy: 0.4998
24736/25000 [============================>.] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
24768/25000 [============================>.] - ETA: 0s - loss: 7.6697 - accuracy: 0.4998
24800/25000 [============================>.] - ETA: 0s - loss: 7.6716 - accuracy: 0.4997
24832/25000 [============================>.] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24864/25000 [============================>.] - ETA: 0s - loss: 7.6734 - accuracy: 0.4996
24896/25000 [============================>.] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
24928/25000 [============================>.] - ETA: 0s - loss: 7.6715 - accuracy: 0.4997
24960/25000 [============================>.] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 99s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
