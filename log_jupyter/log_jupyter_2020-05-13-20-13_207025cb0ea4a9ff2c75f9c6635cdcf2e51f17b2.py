
  test_jupyter /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_jupyter', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_jupyter 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2', 'workflow': 'test_jupyter'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_jupyter

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/207025cb0ea4a9ff2c75f9c6635cdcf2e51f17b2

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
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:51<01:17, 25.72s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.13182881557987688, 'embedding_size_factor': 1.3279912526268234, 'layers.choice': 3, 'learning_rate': 0.008573698585503399, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 3.979780221360213e-08} and reward: 0.3754
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc0\xdf\xc4A\xcb\x01\x9bX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5?s\xc1v\x80 X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?\x81\x8f\x16X>v\x07X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>e]\xc4\xaepM\xc2u.' and reward: 0.3754
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc0\xdf\xc4A\xcb\x01\x9bX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5?s\xc1v\x80 X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?\x81\x8f\x16X>v\x07X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>e]\xc4\xaepM\xc2u.' and reward: 0.3754
 60%|██████    | 3/5 [01:44<01:07, 33.89s/it] 60%|██████    | 3/5 [01:44<01:09, 34.80s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1309202662308265, 'embedding_size_factor': 1.2326605444268077, 'layers.choice': 3, 'learning_rate': 0.0012145712023736773, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 2.6080220111246983e-12} and reward: 0.3924
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xc0\xc1\xfe\xca\xec)cX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\xb8\xfaCV\x1f\x84X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?S\xe6G\xe5\xf1d\x1dX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x86\xf0\xbeT\x9f_[u.' and reward: 0.3924
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xc0\xc1\xfe\xca\xec)cX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf3\xb8\xfaCV\x1f\x84X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?S\xe6G\xe5\xf1d\x1dX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\x86\xf0\xbeT\x9f_[u.' and reward: 0.3924
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 157.69847679138184
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1309202662308265, 'embedding_size_factor': 1.2326605444268077, 'layers.choice': 3, 'learning_rate': 0.0012145712023736773, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 2.6080220111246983e-12}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -40.04s of remaining time.
Ensemble size: 3
Ensemble weights: 
[0.66666667 0.         0.33333333]
	0.3946	 = Validation accuracy score
	1.05s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 161.13s ...
Loading: dataset/models/trainer.pkl
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f316bb05b00> 

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
 [ 0.0666947   0.06565125  0.03289845  0.13282773 -0.03597391  0.13646889]
 [ 0.17838134  0.15489234 -0.05787544  0.06309013  0.16643739  0.25940019]
 [ 0.03260998  0.20459159  0.10546433  0.12630162 -0.00061808 -0.13284385]
 [-0.1308402   0.36333457  0.08979456 -0.03305151 -0.26784182  0.42642576]
 [-0.02057276 -0.03086212  0.12044363 -0.02965911  0.15016139 -0.22976825]
 [-0.33491331 -0.04275338 -0.05725102 -0.00622062 -0.00523075  0.59229428]
 [ 0.15218051  0.02092801 -0.08732225  0.19167982 -0.27128819  0.24521005]
 [ 0.11773756  0.48735514 -0.2056223   0.03628147  0.44518006  0.28589219]
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
{'loss': 0.4409860037267208, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-13 20:17:10.291420: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.4157872311770916, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-13 20:17:11.434625: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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

    8192/17464789 [..............................] - ETA: 10s
  212992/17464789 [..............................] - ETA: 4s 
  385024/17464789 [..............................] - ETA: 4s
  581632/17464789 [..............................] - ETA: 4s
  802816/17464789 [>.............................] - ETA: 4s
 1097728/17464789 [>.............................] - ETA: 4s
 1449984/17464789 [=>............................] - ETA: 3s
 1810432/17464789 [==>...........................] - ETA: 3s
 2211840/17464789 [==>...........................] - ETA: 2s
 2629632/17464789 [===>..........................] - ETA: 2s
 3137536/17464789 [====>.........................] - ETA: 2s
 3645440/17464789 [=====>........................] - ETA: 2s
 4218880/17464789 [======>.......................] - ETA: 1s
 4857856/17464789 [=======>......................] - ETA: 1s
 5521408/17464789 [========>.....................] - ETA: 1s
 6266880/17464789 [=========>....................] - ETA: 1s
 7053312/17464789 [===========>..................] - ETA: 1s
 7905280/17464789 [============>.................] - ETA: 1s
 8798208/17464789 [==============>...............] - ETA: 0s
 9838592/17464789 [===============>..............] - ETA: 0s
10846208/17464789 [=================>............] - ETA: 0s
11927552/17464789 [===================>..........] - ETA: 0s
13058048/17464789 [=====================>........] - ETA: 0s
14229504/17464789 [=======================>......] - ETA: 0s
15499264/17464789 [=========================>....] - ETA: 0s
16695296/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 20:17:24.298203: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 20:17:24.302798: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-13 20:17:24.302938: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5610d660da70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 20:17:24.302953: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:37 - loss: 7.1875 - accuracy: 0.5312
   64/25000 [..............................] - ETA: 2:49 - loss: 7.9062 - accuracy: 0.4844
   96/25000 [..............................] - ETA: 2:14 - loss: 7.8263 - accuracy: 0.4896
  128/25000 [..............................] - ETA: 1:56 - loss: 7.7864 - accuracy: 0.4922
  160/25000 [..............................] - ETA: 1:45 - loss: 7.7625 - accuracy: 0.4938
  192/25000 [..............................] - ETA: 1:38 - loss: 8.0659 - accuracy: 0.4740
  224/25000 [..............................] - ETA: 1:32 - loss: 7.6666 - accuracy: 0.5000
  256/25000 [..............................] - ETA: 1:28 - loss: 7.8463 - accuracy: 0.4883
  288/25000 [..............................] - ETA: 1:25 - loss: 7.9861 - accuracy: 0.4792
  320/25000 [..............................] - ETA: 1:23 - loss: 7.8104 - accuracy: 0.4906
  352/25000 [..............................] - ETA: 1:20 - loss: 7.7102 - accuracy: 0.4972
  384/25000 [..............................] - ETA: 1:19 - loss: 7.7864 - accuracy: 0.4922
  416/25000 [..............................] - ETA: 1:18 - loss: 7.9615 - accuracy: 0.4808
  448/25000 [..............................] - ETA: 1:17 - loss: 7.9747 - accuracy: 0.4799
  480/25000 [..............................] - ETA: 1:16 - loss: 8.0500 - accuracy: 0.4750
  512/25000 [..............................] - ETA: 1:14 - loss: 7.8763 - accuracy: 0.4863
  544/25000 [..............................] - ETA: 1:14 - loss: 7.8357 - accuracy: 0.4890
  576/25000 [..............................] - ETA: 1:13 - loss: 7.7997 - accuracy: 0.4913
  608/25000 [..............................] - ETA: 1:12 - loss: 7.8936 - accuracy: 0.4852
  640/25000 [..............................] - ETA: 1:11 - loss: 7.8822 - accuracy: 0.4859
  672/25000 [..............................] - ETA: 1:11 - loss: 7.8492 - accuracy: 0.4881
  704/25000 [..............................] - ETA: 1:10 - loss: 7.8626 - accuracy: 0.4872
  736/25000 [..............................] - ETA: 1:10 - loss: 7.8750 - accuracy: 0.4864
  768/25000 [..............................] - ETA: 1:09 - loss: 7.8064 - accuracy: 0.4909
  800/25000 [..............................] - ETA: 1:09 - loss: 7.8008 - accuracy: 0.4913
  832/25000 [..............................] - ETA: 1:09 - loss: 7.8878 - accuracy: 0.4856
  864/25000 [>.............................] - ETA: 1:08 - loss: 7.8618 - accuracy: 0.4873
  896/25000 [>.............................] - ETA: 1:08 - loss: 7.8377 - accuracy: 0.4888
  928/25000 [>.............................] - ETA: 1:08 - loss: 7.8814 - accuracy: 0.4860
  960/25000 [>.............................] - ETA: 1:08 - loss: 7.8583 - accuracy: 0.4875
  992/25000 [>.............................] - ETA: 1:07 - loss: 7.8521 - accuracy: 0.4879
 1024/25000 [>.............................] - ETA: 1:07 - loss: 7.8613 - accuracy: 0.4873
 1056/25000 [>.............................] - ETA: 1:07 - loss: 7.8844 - accuracy: 0.4858
 1088/25000 [>.............................] - ETA: 1:06 - loss: 7.8780 - accuracy: 0.4862
 1120/25000 [>.............................] - ETA: 1:06 - loss: 7.8172 - accuracy: 0.4902
 1152/25000 [>.............................] - ETA: 1:06 - loss: 7.8530 - accuracy: 0.4878
 1184/25000 [>.............................] - ETA: 1:06 - loss: 7.8479 - accuracy: 0.4882
 1216/25000 [>.............................] - ETA: 1:05 - loss: 7.8936 - accuracy: 0.4852
 1248/25000 [>.............................] - ETA: 1:05 - loss: 7.8632 - accuracy: 0.4872
 1280/25000 [>.............................] - ETA: 1:05 - loss: 7.8343 - accuracy: 0.4891
 1312/25000 [>.............................] - ETA: 1:05 - loss: 7.8069 - accuracy: 0.4909
 1344/25000 [>.............................] - ETA: 1:05 - loss: 7.8035 - accuracy: 0.4911
 1376/25000 [>.............................] - ETA: 1:05 - loss: 7.8226 - accuracy: 0.4898
 1408/25000 [>.............................] - ETA: 1:04 - loss: 7.8518 - accuracy: 0.4879
 1440/25000 [>.............................] - ETA: 1:04 - loss: 7.8370 - accuracy: 0.4889
 1472/25000 [>.............................] - ETA: 1:04 - loss: 7.8333 - accuracy: 0.4891
 1504/25000 [>.............................] - ETA: 1:04 - loss: 7.8399 - accuracy: 0.4887
 1536/25000 [>.............................] - ETA: 1:04 - loss: 7.8363 - accuracy: 0.4889
 1568/25000 [>.............................] - ETA: 1:03 - loss: 7.8426 - accuracy: 0.4885
 1600/25000 [>.............................] - ETA: 1:03 - loss: 7.8487 - accuracy: 0.4881
 1632/25000 [>.............................] - ETA: 1:03 - loss: 7.8827 - accuracy: 0.4859
 1664/25000 [>.............................] - ETA: 1:03 - loss: 7.8693 - accuracy: 0.4868
 1696/25000 [=>............................] - ETA: 1:03 - loss: 7.9017 - accuracy: 0.4847
 1728/25000 [=>............................] - ETA: 1:02 - loss: 7.8973 - accuracy: 0.4850
 1760/25000 [=>............................] - ETA: 1:02 - loss: 7.8757 - accuracy: 0.4864
 1792/25000 [=>............................] - ETA: 1:02 - loss: 7.9062 - accuracy: 0.4844
 1824/25000 [=>............................] - ETA: 1:02 - loss: 7.8600 - accuracy: 0.4874
 1856/25000 [=>............................] - ETA: 1:02 - loss: 7.8318 - accuracy: 0.4892
 1888/25000 [=>............................] - ETA: 1:02 - loss: 7.8372 - accuracy: 0.4889
 1920/25000 [=>............................] - ETA: 1:02 - loss: 7.8663 - accuracy: 0.4870
 1952/25000 [=>............................] - ETA: 1:02 - loss: 7.8944 - accuracy: 0.4851
 1984/25000 [=>............................] - ETA: 1:01 - loss: 7.8753 - accuracy: 0.4864
 2016/25000 [=>............................] - ETA: 1:01 - loss: 7.8568 - accuracy: 0.4876
 2048/25000 [=>............................] - ETA: 1:01 - loss: 7.8388 - accuracy: 0.4888
 2080/25000 [=>............................] - ETA: 1:01 - loss: 7.8583 - accuracy: 0.4875
 2112/25000 [=>............................] - ETA: 1:01 - loss: 7.8626 - accuracy: 0.4872
 2144/25000 [=>............................] - ETA: 1:01 - loss: 7.8597 - accuracy: 0.4874
 2176/25000 [=>............................] - ETA: 1:01 - loss: 7.8569 - accuracy: 0.4876
 2208/25000 [=>............................] - ETA: 1:01 - loss: 7.8819 - accuracy: 0.4860
 2240/25000 [=>............................] - ETA: 1:01 - loss: 7.8583 - accuracy: 0.4875
 2272/25000 [=>............................] - ETA: 1:01 - loss: 7.8826 - accuracy: 0.4859
 2304/25000 [=>............................] - ETA: 1:00 - loss: 7.9129 - accuracy: 0.4839
 2336/25000 [=>............................] - ETA: 1:00 - loss: 7.8767 - accuracy: 0.4863
 2368/25000 [=>............................] - ETA: 1:00 - loss: 7.8544 - accuracy: 0.4878
 2400/25000 [=>............................] - ETA: 1:00 - loss: 7.8391 - accuracy: 0.4888
 2432/25000 [=>............................] - ETA: 1:00 - loss: 7.8432 - accuracy: 0.4885
 2464/25000 [=>............................] - ETA: 1:00 - loss: 7.8533 - accuracy: 0.4878
 2496/25000 [=>............................] - ETA: 1:00 - loss: 7.8693 - accuracy: 0.4868
 2528/25000 [==>...........................] - ETA: 59s - loss: 7.9092 - accuracy: 0.4842 
 2560/25000 [==>...........................] - ETA: 59s - loss: 7.9481 - accuracy: 0.4816
 2592/25000 [==>...........................] - ETA: 59s - loss: 7.9506 - accuracy: 0.4815
 2624/25000 [==>...........................] - ETA: 59s - loss: 7.9296 - accuracy: 0.4829
 2656/25000 [==>...........................] - ETA: 59s - loss: 7.9206 - accuracy: 0.4834
 2688/25000 [==>...........................] - ETA: 59s - loss: 7.9461 - accuracy: 0.4818
 2720/25000 [==>...........................] - ETA: 59s - loss: 7.9372 - accuracy: 0.4824
 2752/25000 [==>...........................] - ETA: 59s - loss: 7.9396 - accuracy: 0.4822
 2784/25000 [==>...........................] - ETA: 59s - loss: 7.9475 - accuracy: 0.4817
 2816/25000 [==>...........................] - ETA: 58s - loss: 7.9280 - accuracy: 0.4830
 2848/25000 [==>...........................] - ETA: 58s - loss: 7.9089 - accuracy: 0.4842
 2880/25000 [==>...........................] - ETA: 58s - loss: 7.9009 - accuracy: 0.4847
 2912/25000 [==>...........................] - ETA: 58s - loss: 7.9194 - accuracy: 0.4835
 2944/25000 [==>...........................] - ETA: 58s - loss: 7.9375 - accuracy: 0.4823
 2976/25000 [==>...........................] - ETA: 58s - loss: 7.9551 - accuracy: 0.4812
 3008/25000 [==>...........................] - ETA: 58s - loss: 7.9368 - accuracy: 0.4824
 3040/25000 [==>...........................] - ETA: 57s - loss: 7.9188 - accuracy: 0.4836
 3072/25000 [==>...........................] - ETA: 57s - loss: 7.9312 - accuracy: 0.4827
 3104/25000 [==>...........................] - ETA: 57s - loss: 7.9037 - accuracy: 0.4845
 3136/25000 [==>...........................] - ETA: 57s - loss: 7.9111 - accuracy: 0.4841
 3168/25000 [==>...........................] - ETA: 57s - loss: 7.8941 - accuracy: 0.4852
 3200/25000 [==>...........................] - ETA: 57s - loss: 7.8822 - accuracy: 0.4859
 3232/25000 [==>...........................] - ETA: 57s - loss: 7.8659 - accuracy: 0.4870
 3264/25000 [==>...........................] - ETA: 57s - loss: 7.8686 - accuracy: 0.4868
 3296/25000 [==>...........................] - ETA: 57s - loss: 7.8667 - accuracy: 0.4870
 3328/25000 [==>...........................] - ETA: 56s - loss: 7.8555 - accuracy: 0.4877
 3360/25000 [===>..........................] - ETA: 56s - loss: 7.8720 - accuracy: 0.4866
 3392/25000 [===>..........................] - ETA: 56s - loss: 7.8520 - accuracy: 0.4879
 3424/25000 [===>..........................] - ETA: 56s - loss: 7.8502 - accuracy: 0.4880
 3456/25000 [===>..........................] - ETA: 56s - loss: 7.8530 - accuracy: 0.4878
 3488/25000 [===>..........................] - ETA: 56s - loss: 7.8425 - accuracy: 0.4885
 3520/25000 [===>..........................] - ETA: 56s - loss: 7.8496 - accuracy: 0.4881
 3552/25000 [===>..........................] - ETA: 56s - loss: 7.8436 - accuracy: 0.4885
 3584/25000 [===>..........................] - ETA: 56s - loss: 7.8506 - accuracy: 0.4880
 3616/25000 [===>..........................] - ETA: 55s - loss: 7.8362 - accuracy: 0.4889
 3648/25000 [===>..........................] - ETA: 55s - loss: 7.8390 - accuracy: 0.4888
 3680/25000 [===>..........................] - ETA: 55s - loss: 7.8375 - accuracy: 0.4889
 3712/25000 [===>..........................] - ETA: 55s - loss: 7.8236 - accuracy: 0.4898
 3744/25000 [===>..........................] - ETA: 55s - loss: 7.8263 - accuracy: 0.4896
 3776/25000 [===>..........................] - ETA: 55s - loss: 7.8290 - accuracy: 0.4894
 3808/25000 [===>..........................] - ETA: 55s - loss: 7.8196 - accuracy: 0.4900
 3840/25000 [===>..........................] - ETA: 55s - loss: 7.7984 - accuracy: 0.4914
 3872/25000 [===>..........................] - ETA: 55s - loss: 7.8131 - accuracy: 0.4904
 3904/25000 [===>..........................] - ETA: 54s - loss: 7.7962 - accuracy: 0.4915
 3936/25000 [===>..........................] - ETA: 54s - loss: 7.7913 - accuracy: 0.4919
 3968/25000 [===>..........................] - ETA: 54s - loss: 7.7903 - accuracy: 0.4919
 4000/25000 [===>..........................] - ETA: 54s - loss: 7.7625 - accuracy: 0.4938
 4032/25000 [===>..........................] - ETA: 54s - loss: 7.7465 - accuracy: 0.4948
 4064/25000 [===>..........................] - ETA: 54s - loss: 7.7308 - accuracy: 0.4958
 4096/25000 [===>..........................] - ETA: 54s - loss: 7.7340 - accuracy: 0.4956
 4128/25000 [===>..........................] - ETA: 54s - loss: 7.7372 - accuracy: 0.4954
 4160/25000 [===>..........................] - ETA: 54s - loss: 7.7477 - accuracy: 0.4947
 4192/25000 [====>.........................] - ETA: 54s - loss: 7.7544 - accuracy: 0.4943
 4224/25000 [====>.........................] - ETA: 54s - loss: 7.7610 - accuracy: 0.4938
 4256/25000 [====>.........................] - ETA: 53s - loss: 7.7675 - accuracy: 0.4934
 4288/25000 [====>.........................] - ETA: 53s - loss: 7.7703 - accuracy: 0.4932
 4320/25000 [====>.........................] - ETA: 53s - loss: 7.7625 - accuracy: 0.4938
 4352/25000 [====>.........................] - ETA: 53s - loss: 7.7477 - accuracy: 0.4947
 4384/25000 [====>.........................] - ETA: 53s - loss: 7.7436 - accuracy: 0.4950
 4416/25000 [====>.........................] - ETA: 53s - loss: 7.7395 - accuracy: 0.4952
 4448/25000 [====>.........................] - ETA: 53s - loss: 7.7252 - accuracy: 0.4962
 4480/25000 [====>.........................] - ETA: 53s - loss: 7.7248 - accuracy: 0.4962
 4512/25000 [====>.........................] - ETA: 53s - loss: 7.7244 - accuracy: 0.4962
 4544/25000 [====>.........................] - ETA: 53s - loss: 7.7341 - accuracy: 0.4956
 4576/25000 [====>.........................] - ETA: 52s - loss: 7.7269 - accuracy: 0.4961
 4608/25000 [====>.........................] - ETA: 52s - loss: 7.7465 - accuracy: 0.4948
 4640/25000 [====>.........................] - ETA: 52s - loss: 7.7393 - accuracy: 0.4953
 4672/25000 [====>.........................] - ETA: 52s - loss: 7.7454 - accuracy: 0.4949
 4704/25000 [====>.........................] - ETA: 52s - loss: 7.7449 - accuracy: 0.4949
 4736/25000 [====>.........................] - ETA: 52s - loss: 7.7540 - accuracy: 0.4943
 4768/25000 [====>.........................] - ETA: 52s - loss: 7.7599 - accuracy: 0.4939
 4800/25000 [====>.........................] - ETA: 52s - loss: 7.7561 - accuracy: 0.4942
 4832/25000 [====>.........................] - ETA: 52s - loss: 7.7555 - accuracy: 0.4942
 4864/25000 [====>.........................] - ETA: 52s - loss: 7.7675 - accuracy: 0.4934
 4896/25000 [====>.........................] - ETA: 51s - loss: 7.7794 - accuracy: 0.4926
 4928/25000 [====>.........................] - ETA: 51s - loss: 7.7849 - accuracy: 0.4923
 4960/25000 [====>.........................] - ETA: 51s - loss: 7.7903 - accuracy: 0.4919
 4992/25000 [====>.........................] - ETA: 51s - loss: 7.7926 - accuracy: 0.4918
 5024/25000 [=====>........................] - ETA: 51s - loss: 7.7734 - accuracy: 0.4930
 5056/25000 [=====>........................] - ETA: 51s - loss: 7.7728 - accuracy: 0.4931
 5088/25000 [=====>........................] - ETA: 51s - loss: 7.7751 - accuracy: 0.4929
 5120/25000 [=====>........................] - ETA: 51s - loss: 7.7654 - accuracy: 0.4936
 5152/25000 [=====>........................] - ETA: 51s - loss: 7.7678 - accuracy: 0.4934
 5184/25000 [=====>........................] - ETA: 51s - loss: 7.7583 - accuracy: 0.4940
 5216/25000 [=====>........................] - ETA: 50s - loss: 7.7519 - accuracy: 0.4944
 5248/25000 [=====>........................] - ETA: 50s - loss: 7.7397 - accuracy: 0.4952
 5280/25000 [=====>........................] - ETA: 50s - loss: 7.7537 - accuracy: 0.4943
 5312/25000 [=====>........................] - ETA: 50s - loss: 7.7532 - accuracy: 0.4944
 5344/25000 [=====>........................] - ETA: 50s - loss: 7.7412 - accuracy: 0.4951
 5376/25000 [=====>........................] - ETA: 50s - loss: 7.7408 - accuracy: 0.4952
 5408/25000 [=====>........................] - ETA: 50s - loss: 7.7290 - accuracy: 0.4959
 5440/25000 [=====>........................] - ETA: 50s - loss: 7.7314 - accuracy: 0.4958
 5472/25000 [=====>........................] - ETA: 50s - loss: 7.7255 - accuracy: 0.4962
 5504/25000 [=====>........................] - ETA: 50s - loss: 7.7335 - accuracy: 0.4956
 5536/25000 [=====>........................] - ETA: 50s - loss: 7.7303 - accuracy: 0.4958
 5568/25000 [=====>........................] - ETA: 49s - loss: 7.7189 - accuracy: 0.4966
 5600/25000 [=====>........................] - ETA: 49s - loss: 7.7214 - accuracy: 0.4964
 5632/25000 [=====>........................] - ETA: 49s - loss: 7.7129 - accuracy: 0.4970
 5664/25000 [=====>........................] - ETA: 49s - loss: 7.7153 - accuracy: 0.4968
 5696/25000 [=====>........................] - ETA: 49s - loss: 7.7258 - accuracy: 0.4961
 5728/25000 [=====>........................] - ETA: 49s - loss: 7.7255 - accuracy: 0.4962
 5760/25000 [=====>........................] - ETA: 49s - loss: 7.7145 - accuracy: 0.4969
 5792/25000 [=====>........................] - ETA: 49s - loss: 7.7116 - accuracy: 0.4971
 5824/25000 [=====>........................] - ETA: 49s - loss: 7.7114 - accuracy: 0.4971
 5856/25000 [======>.......................] - ETA: 49s - loss: 7.7164 - accuracy: 0.4968
 5888/25000 [======>.......................] - ETA: 49s - loss: 7.7161 - accuracy: 0.4968
 5920/25000 [======>.......................] - ETA: 49s - loss: 7.7055 - accuracy: 0.4975
 5952/25000 [======>.......................] - ETA: 48s - loss: 7.7053 - accuracy: 0.4975
 5984/25000 [======>.......................] - ETA: 48s - loss: 7.7051 - accuracy: 0.4975
 6016/25000 [======>.......................] - ETA: 48s - loss: 7.7074 - accuracy: 0.4973
 6048/25000 [======>.......................] - ETA: 48s - loss: 7.6920 - accuracy: 0.4983
 6080/25000 [======>.......................] - ETA: 48s - loss: 7.6843 - accuracy: 0.4988
 6112/25000 [======>.......................] - ETA: 48s - loss: 7.6842 - accuracy: 0.4989
 6144/25000 [======>.......................] - ETA: 48s - loss: 7.6766 - accuracy: 0.4993
 6176/25000 [======>.......................] - ETA: 48s - loss: 7.6691 - accuracy: 0.4998
 6208/25000 [======>.......................] - ETA: 48s - loss: 7.6543 - accuracy: 0.5008
 6240/25000 [======>.......................] - ETA: 48s - loss: 7.6568 - accuracy: 0.5006
 6272/25000 [======>.......................] - ETA: 48s - loss: 7.6495 - accuracy: 0.5011
 6304/25000 [======>.......................] - ETA: 48s - loss: 7.6520 - accuracy: 0.5010
 6336/25000 [======>.......................] - ETA: 47s - loss: 7.6545 - accuracy: 0.5008
 6368/25000 [======>.......................] - ETA: 47s - loss: 7.6570 - accuracy: 0.5006
 6400/25000 [======>.......................] - ETA: 47s - loss: 7.6690 - accuracy: 0.4998
 6432/25000 [======>.......................] - ETA: 47s - loss: 7.6738 - accuracy: 0.4995
 6464/25000 [======>.......................] - ETA: 47s - loss: 7.6832 - accuracy: 0.4989
 6496/25000 [======>.......................] - ETA: 47s - loss: 7.6831 - accuracy: 0.4989
 6528/25000 [======>.......................] - ETA: 47s - loss: 7.6807 - accuracy: 0.4991
 6560/25000 [======>.......................] - ETA: 47s - loss: 7.6713 - accuracy: 0.4997
 6592/25000 [======>.......................] - ETA: 47s - loss: 7.6736 - accuracy: 0.4995
 6624/25000 [======>.......................] - ETA: 47s - loss: 7.6805 - accuracy: 0.4991
 6656/25000 [======>.......................] - ETA: 47s - loss: 7.6735 - accuracy: 0.4995
 6688/25000 [=======>......................] - ETA: 46s - loss: 7.6689 - accuracy: 0.4999
 6720/25000 [=======>......................] - ETA: 46s - loss: 7.6621 - accuracy: 0.5003
 6752/25000 [=======>......................] - ETA: 46s - loss: 7.6553 - accuracy: 0.5007
 6784/25000 [=======>......................] - ETA: 46s - loss: 7.6576 - accuracy: 0.5006
 6816/25000 [=======>......................] - ETA: 46s - loss: 7.6576 - accuracy: 0.5006
 6848/25000 [=======>......................] - ETA: 46s - loss: 7.6532 - accuracy: 0.5009
 6880/25000 [=======>......................] - ETA: 46s - loss: 7.6488 - accuracy: 0.5012
 6912/25000 [=======>......................] - ETA: 46s - loss: 7.6467 - accuracy: 0.5013
 6944/25000 [=======>......................] - ETA: 46s - loss: 7.6445 - accuracy: 0.5014
 6976/25000 [=======>......................] - ETA: 46s - loss: 7.6446 - accuracy: 0.5014
 7008/25000 [=======>......................] - ETA: 46s - loss: 7.6360 - accuracy: 0.5020
 7040/25000 [=======>......................] - ETA: 46s - loss: 7.6383 - accuracy: 0.5018
 7072/25000 [=======>......................] - ETA: 45s - loss: 7.6493 - accuracy: 0.5011
 7104/25000 [=======>......................] - ETA: 45s - loss: 7.6580 - accuracy: 0.5006
 7136/25000 [=======>......................] - ETA: 45s - loss: 7.6666 - accuracy: 0.5000
 7168/25000 [=======>......................] - ETA: 45s - loss: 7.6645 - accuracy: 0.5001
 7200/25000 [=======>......................] - ETA: 45s - loss: 7.6624 - accuracy: 0.5003
 7232/25000 [=======>......................] - ETA: 45s - loss: 7.6603 - accuracy: 0.5004
 7264/25000 [=======>......................] - ETA: 45s - loss: 7.6687 - accuracy: 0.4999
 7296/25000 [=======>......................] - ETA: 45s - loss: 7.6708 - accuracy: 0.4997
 7328/25000 [=======>......................] - ETA: 45s - loss: 7.6813 - accuracy: 0.4990
 7360/25000 [=======>......................] - ETA: 45s - loss: 7.6791 - accuracy: 0.4992
 7392/25000 [=======>......................] - ETA: 45s - loss: 7.6811 - accuracy: 0.4991
 7424/25000 [=======>......................] - ETA: 45s - loss: 7.6811 - accuracy: 0.4991
 7456/25000 [=======>......................] - ETA: 45s - loss: 7.6851 - accuracy: 0.4988
 7488/25000 [=======>......................] - ETA: 44s - loss: 7.6810 - accuracy: 0.4991
 7520/25000 [========>.....................] - ETA: 44s - loss: 7.6768 - accuracy: 0.4993
 7552/25000 [========>.....................] - ETA: 44s - loss: 7.6829 - accuracy: 0.4989
 7584/25000 [========>.....................] - ETA: 44s - loss: 7.6747 - accuracy: 0.4995
 7616/25000 [========>.....................] - ETA: 44s - loss: 7.6747 - accuracy: 0.4995
 7648/25000 [========>.....................] - ETA: 44s - loss: 7.6766 - accuracy: 0.4993
 7680/25000 [========>.....................] - ETA: 44s - loss: 7.6826 - accuracy: 0.4990
 7712/25000 [========>.....................] - ETA: 44s - loss: 7.6925 - accuracy: 0.4983
 7744/25000 [========>.....................] - ETA: 44s - loss: 7.6983 - accuracy: 0.4979
 7776/25000 [========>.....................] - ETA: 44s - loss: 7.7179 - accuracy: 0.4967
 7808/25000 [========>.....................] - ETA: 43s - loss: 7.7196 - accuracy: 0.4965
 7840/25000 [========>.....................] - ETA: 43s - loss: 7.7194 - accuracy: 0.4966
 7872/25000 [========>.....................] - ETA: 43s - loss: 7.7231 - accuracy: 0.4963
 7904/25000 [========>.....................] - ETA: 43s - loss: 7.7306 - accuracy: 0.4958
 7936/25000 [========>.....................] - ETA: 43s - loss: 7.7304 - accuracy: 0.4958
 7968/25000 [========>.....................] - ETA: 43s - loss: 7.7224 - accuracy: 0.4964
 8000/25000 [========>.....................] - ETA: 43s - loss: 7.7299 - accuracy: 0.4959
 8032/25000 [========>.....................] - ETA: 43s - loss: 7.7239 - accuracy: 0.4963
 8064/25000 [========>.....................] - ETA: 43s - loss: 7.7237 - accuracy: 0.4963
 8096/25000 [========>.....................] - ETA: 43s - loss: 7.7253 - accuracy: 0.4962
 8128/25000 [========>.....................] - ETA: 43s - loss: 7.7194 - accuracy: 0.4966
 8160/25000 [========>.....................] - ETA: 42s - loss: 7.7267 - accuracy: 0.4961
 8192/25000 [========>.....................] - ETA: 42s - loss: 7.7228 - accuracy: 0.4963
 8224/25000 [========>.....................] - ETA: 42s - loss: 7.7226 - accuracy: 0.4964
 8256/25000 [========>.....................] - ETA: 42s - loss: 7.7261 - accuracy: 0.4961
 8288/25000 [========>.....................] - ETA: 42s - loss: 7.7277 - accuracy: 0.4960
 8320/25000 [========>.....................] - ETA: 42s - loss: 7.7311 - accuracy: 0.4958
 8352/25000 [=========>....................] - ETA: 42s - loss: 7.7401 - accuracy: 0.4952
 8384/25000 [=========>....................] - ETA: 42s - loss: 7.7325 - accuracy: 0.4957
 8416/25000 [=========>....................] - ETA: 42s - loss: 7.7213 - accuracy: 0.4964
 8448/25000 [=========>....................] - ETA: 42s - loss: 7.7156 - accuracy: 0.4968
 8480/25000 [=========>....................] - ETA: 42s - loss: 7.7100 - accuracy: 0.4972
 8512/25000 [=========>....................] - ETA: 42s - loss: 7.7026 - accuracy: 0.4977
 8544/25000 [=========>....................] - ETA: 42s - loss: 7.7079 - accuracy: 0.4973
 8576/25000 [=========>....................] - ETA: 41s - loss: 7.7060 - accuracy: 0.4974
 8608/25000 [=========>....................] - ETA: 41s - loss: 7.7112 - accuracy: 0.4971
 8640/25000 [=========>....................] - ETA: 41s - loss: 7.7074 - accuracy: 0.4973
 8672/25000 [=========>....................] - ETA: 41s - loss: 7.7073 - accuracy: 0.4973
 8704/25000 [=========>....................] - ETA: 41s - loss: 7.7124 - accuracy: 0.4970
 8736/25000 [=========>....................] - ETA: 41s - loss: 7.7158 - accuracy: 0.4968
 8768/25000 [=========>....................] - ETA: 41s - loss: 7.7156 - accuracy: 0.4968
 8800/25000 [=========>....................] - ETA: 41s - loss: 7.7189 - accuracy: 0.4966
 8832/25000 [=========>....................] - ETA: 41s - loss: 7.7135 - accuracy: 0.4969
 8864/25000 [=========>....................] - ETA: 41s - loss: 7.7289 - accuracy: 0.4959
 8896/25000 [=========>....................] - ETA: 41s - loss: 7.7287 - accuracy: 0.4960
 8928/25000 [=========>....................] - ETA: 41s - loss: 7.7319 - accuracy: 0.4957
 8960/25000 [=========>....................] - ETA: 40s - loss: 7.7351 - accuracy: 0.4955
 8992/25000 [=========>....................] - ETA: 40s - loss: 7.7365 - accuracy: 0.4954
 9024/25000 [=========>....................] - ETA: 40s - loss: 7.7329 - accuracy: 0.4957
 9056/25000 [=========>....................] - ETA: 40s - loss: 7.7327 - accuracy: 0.4957
 9088/25000 [=========>....................] - ETA: 40s - loss: 7.7341 - accuracy: 0.4956
 9120/25000 [=========>....................] - ETA: 40s - loss: 7.7339 - accuracy: 0.4956
 9152/25000 [=========>....................] - ETA: 40s - loss: 7.7303 - accuracy: 0.4958
 9184/25000 [==========>...................] - ETA: 40s - loss: 7.7267 - accuracy: 0.4961
 9216/25000 [==========>...................] - ETA: 40s - loss: 7.7265 - accuracy: 0.4961
 9248/25000 [==========>...................] - ETA: 40s - loss: 7.7296 - accuracy: 0.4959
 9280/25000 [==========>...................] - ETA: 40s - loss: 7.7278 - accuracy: 0.4960
 9312/25000 [==========>...................] - ETA: 40s - loss: 7.7325 - accuracy: 0.4957
 9344/25000 [==========>...................] - ETA: 39s - loss: 7.7290 - accuracy: 0.4959
 9376/25000 [==========>...................] - ETA: 39s - loss: 7.7288 - accuracy: 0.4959
 9408/25000 [==========>...................] - ETA: 39s - loss: 7.7253 - accuracy: 0.4962
 9440/25000 [==========>...................] - ETA: 39s - loss: 7.7251 - accuracy: 0.4962
 9472/25000 [==========>...................] - ETA: 39s - loss: 7.7200 - accuracy: 0.4965
 9504/25000 [==========>...................] - ETA: 39s - loss: 7.7215 - accuracy: 0.4964
 9536/25000 [==========>...................] - ETA: 39s - loss: 7.7229 - accuracy: 0.4963
 9568/25000 [==========>...................] - ETA: 39s - loss: 7.7307 - accuracy: 0.4958
 9600/25000 [==========>...................] - ETA: 39s - loss: 7.7369 - accuracy: 0.4954
 9632/25000 [==========>...................] - ETA: 39s - loss: 7.7383 - accuracy: 0.4953
 9664/25000 [==========>...................] - ETA: 39s - loss: 7.7396 - accuracy: 0.4952
 9696/25000 [==========>...................] - ETA: 39s - loss: 7.7362 - accuracy: 0.4955
 9728/25000 [==========>...................] - ETA: 38s - loss: 7.7407 - accuracy: 0.4952
 9760/25000 [==========>...................] - ETA: 38s - loss: 7.7373 - accuracy: 0.4954
 9792/25000 [==========>...................] - ETA: 38s - loss: 7.7433 - accuracy: 0.4950
 9824/25000 [==========>...................] - ETA: 38s - loss: 7.7384 - accuracy: 0.4953
 9856/25000 [==========>...................] - ETA: 38s - loss: 7.7366 - accuracy: 0.4954
 9888/25000 [==========>...................] - ETA: 38s - loss: 7.7348 - accuracy: 0.4956
 9920/25000 [==========>...................] - ETA: 38s - loss: 7.7300 - accuracy: 0.4959
 9952/25000 [==========>...................] - ETA: 38s - loss: 7.7282 - accuracy: 0.4960
 9984/25000 [==========>...................] - ETA: 38s - loss: 7.7311 - accuracy: 0.4958
10016/25000 [===========>..................] - ETA: 38s - loss: 7.7294 - accuracy: 0.4959
10048/25000 [===========>..................] - ETA: 38s - loss: 7.7246 - accuracy: 0.4962
10080/25000 [===========>..................] - ETA: 38s - loss: 7.7259 - accuracy: 0.4961
10112/25000 [===========>..................] - ETA: 37s - loss: 7.7242 - accuracy: 0.4962
10144/25000 [===========>..................] - ETA: 37s - loss: 7.7271 - accuracy: 0.4961
10176/25000 [===========>..................] - ETA: 37s - loss: 7.7254 - accuracy: 0.4962
10208/25000 [===========>..................] - ETA: 37s - loss: 7.7192 - accuracy: 0.4966
10240/25000 [===========>..................] - ETA: 37s - loss: 7.7250 - accuracy: 0.4962
10272/25000 [===========>..................] - ETA: 37s - loss: 7.7219 - accuracy: 0.4964
10304/25000 [===========>..................] - ETA: 37s - loss: 7.7187 - accuracy: 0.4966
10336/25000 [===========>..................] - ETA: 37s - loss: 7.7230 - accuracy: 0.4963
10368/25000 [===========>..................] - ETA: 37s - loss: 7.7169 - accuracy: 0.4967
10400/25000 [===========>..................] - ETA: 37s - loss: 7.7167 - accuracy: 0.4967
10432/25000 [===========>..................] - ETA: 37s - loss: 7.7181 - accuracy: 0.4966
10464/25000 [===========>..................] - ETA: 37s - loss: 7.7120 - accuracy: 0.4970
10496/25000 [===========>..................] - ETA: 36s - loss: 7.7207 - accuracy: 0.4965
10528/25000 [===========>..................] - ETA: 36s - loss: 7.7220 - accuracy: 0.4964
10560/25000 [===========>..................] - ETA: 36s - loss: 7.7262 - accuracy: 0.4961
10592/25000 [===========>..................] - ETA: 36s - loss: 7.7260 - accuracy: 0.4961
10624/25000 [===========>..................] - ETA: 36s - loss: 7.7229 - accuracy: 0.4963
10656/25000 [===========>..................] - ETA: 36s - loss: 7.7155 - accuracy: 0.4968
10688/25000 [===========>..................] - ETA: 36s - loss: 7.7240 - accuracy: 0.4963
10720/25000 [===========>..................] - ETA: 36s - loss: 7.7238 - accuracy: 0.4963
10752/25000 [===========>..................] - ETA: 36s - loss: 7.7251 - accuracy: 0.4962
10784/25000 [===========>..................] - ETA: 36s - loss: 7.7221 - accuracy: 0.4964
10816/25000 [===========>..................] - ETA: 36s - loss: 7.7276 - accuracy: 0.4960
10848/25000 [============>.................] - ETA: 36s - loss: 7.7246 - accuracy: 0.4962
10880/25000 [============>.................] - ETA: 35s - loss: 7.7272 - accuracy: 0.4960
10912/25000 [============>.................] - ETA: 35s - loss: 7.7327 - accuracy: 0.4957
10944/25000 [============>.................] - ETA: 35s - loss: 7.7339 - accuracy: 0.4956
10976/25000 [============>.................] - ETA: 35s - loss: 7.7309 - accuracy: 0.4958
11008/25000 [============>.................] - ETA: 35s - loss: 7.7307 - accuracy: 0.4958
11040/25000 [============>.................] - ETA: 35s - loss: 7.7361 - accuracy: 0.4955
11072/25000 [============>.................] - ETA: 35s - loss: 7.7289 - accuracy: 0.4959
11104/25000 [============>.................] - ETA: 35s - loss: 7.7288 - accuracy: 0.4959
11136/25000 [============>.................] - ETA: 35s - loss: 7.7272 - accuracy: 0.4960
11168/25000 [============>.................] - ETA: 35s - loss: 7.7270 - accuracy: 0.4961
11200/25000 [============>.................] - ETA: 35s - loss: 7.7310 - accuracy: 0.4958
11232/25000 [============>.................] - ETA: 35s - loss: 7.7321 - accuracy: 0.4957
11264/25000 [============>.................] - ETA: 34s - loss: 7.7292 - accuracy: 0.4959
11296/25000 [============>.................] - ETA: 34s - loss: 7.7318 - accuracy: 0.4958
11328/25000 [============>.................] - ETA: 34s - loss: 7.7397 - accuracy: 0.4952
11360/25000 [============>.................] - ETA: 34s - loss: 7.7395 - accuracy: 0.4952
11392/25000 [============>.................] - ETA: 34s - loss: 7.7406 - accuracy: 0.4952
11424/25000 [============>.................] - ETA: 34s - loss: 7.7391 - accuracy: 0.4953
11456/25000 [============>.................] - ETA: 34s - loss: 7.7362 - accuracy: 0.4955
11488/25000 [============>.................] - ETA: 34s - loss: 7.7414 - accuracy: 0.4951
11520/25000 [============>.................] - ETA: 34s - loss: 7.7398 - accuracy: 0.4952
11552/25000 [============>.................] - ETA: 34s - loss: 7.7370 - accuracy: 0.4954
11584/25000 [============>.................] - ETA: 34s - loss: 7.7354 - accuracy: 0.4955
11616/25000 [============>.................] - ETA: 34s - loss: 7.7392 - accuracy: 0.4953
11648/25000 [============>.................] - ETA: 33s - loss: 7.7390 - accuracy: 0.4953
11680/25000 [=============>................] - ETA: 33s - loss: 7.7362 - accuracy: 0.4955
11712/25000 [=============>................] - ETA: 33s - loss: 7.7373 - accuracy: 0.4954
11744/25000 [=============>................] - ETA: 33s - loss: 7.7293 - accuracy: 0.4959
11776/25000 [=============>................] - ETA: 33s - loss: 7.7343 - accuracy: 0.4956
11808/25000 [=============>................] - ETA: 33s - loss: 7.7380 - accuracy: 0.4953
11840/25000 [=============>................] - ETA: 33s - loss: 7.7417 - accuracy: 0.4951
11872/25000 [=============>................] - ETA: 33s - loss: 7.7467 - accuracy: 0.4948
11904/25000 [=============>................] - ETA: 33s - loss: 7.7478 - accuracy: 0.4947
11936/25000 [=============>................] - ETA: 33s - loss: 7.7527 - accuracy: 0.4944
11968/25000 [=============>................] - ETA: 33s - loss: 7.7589 - accuracy: 0.4940
12000/25000 [=============>................] - ETA: 33s - loss: 7.7548 - accuracy: 0.4942
12032/25000 [=============>................] - ETA: 33s - loss: 7.7571 - accuracy: 0.4941
12064/25000 [=============>................] - ETA: 33s - loss: 7.7619 - accuracy: 0.4938
12096/25000 [=============>................] - ETA: 32s - loss: 7.7655 - accuracy: 0.4936
12128/25000 [=============>................] - ETA: 32s - loss: 7.7614 - accuracy: 0.4938
12160/25000 [=============>................] - ETA: 32s - loss: 7.7574 - accuracy: 0.4941
12192/25000 [=============>................] - ETA: 32s - loss: 7.7559 - accuracy: 0.4942
12224/25000 [=============>................] - ETA: 32s - loss: 7.7532 - accuracy: 0.4944
12256/25000 [=============>................] - ETA: 32s - loss: 7.7479 - accuracy: 0.4947
12288/25000 [=============>................] - ETA: 32s - loss: 7.7477 - accuracy: 0.4947
12320/25000 [=============>................] - ETA: 32s - loss: 7.7513 - accuracy: 0.4945
12352/25000 [=============>................] - ETA: 32s - loss: 7.7448 - accuracy: 0.4949
12384/25000 [=============>................] - ETA: 32s - loss: 7.7384 - accuracy: 0.4953
12416/25000 [=============>................] - ETA: 32s - loss: 7.7395 - accuracy: 0.4952
12448/25000 [=============>................] - ETA: 32s - loss: 7.7393 - accuracy: 0.4953
12480/25000 [=============>................] - ETA: 31s - loss: 7.7416 - accuracy: 0.4951
12512/25000 [==============>...............] - ETA: 31s - loss: 7.7438 - accuracy: 0.4950
12544/25000 [==============>...............] - ETA: 31s - loss: 7.7424 - accuracy: 0.4951
12576/25000 [==============>...............] - ETA: 31s - loss: 7.7386 - accuracy: 0.4953
12608/25000 [==============>...............] - ETA: 31s - loss: 7.7372 - accuracy: 0.4954
12640/25000 [==============>...............] - ETA: 31s - loss: 7.7285 - accuracy: 0.4960
12672/25000 [==============>...............] - ETA: 31s - loss: 7.7307 - accuracy: 0.4958
12704/25000 [==============>...............] - ETA: 31s - loss: 7.7294 - accuracy: 0.4959
12736/25000 [==============>...............] - ETA: 31s - loss: 7.7352 - accuracy: 0.4955
12768/25000 [==============>...............] - ETA: 31s - loss: 7.7267 - accuracy: 0.4961
12800/25000 [==============>...............] - ETA: 31s - loss: 7.7229 - accuracy: 0.4963
12832/25000 [==============>...............] - ETA: 31s - loss: 7.7228 - accuracy: 0.4963
12864/25000 [==============>...............] - ETA: 30s - loss: 7.7286 - accuracy: 0.4960
12896/25000 [==============>...............] - ETA: 30s - loss: 7.7356 - accuracy: 0.4955
12928/25000 [==============>...............] - ETA: 30s - loss: 7.7295 - accuracy: 0.4959
12960/25000 [==============>...............] - ETA: 30s - loss: 7.7281 - accuracy: 0.4960
12992/25000 [==============>...............] - ETA: 30s - loss: 7.7268 - accuracy: 0.4961
13024/25000 [==============>...............] - ETA: 30s - loss: 7.7325 - accuracy: 0.4957
13056/25000 [==============>...............] - ETA: 30s - loss: 7.7300 - accuracy: 0.4959
13088/25000 [==============>...............] - ETA: 30s - loss: 7.7299 - accuracy: 0.4959
13120/25000 [==============>...............] - ETA: 30s - loss: 7.7309 - accuracy: 0.4958
13152/25000 [==============>...............] - ETA: 30s - loss: 7.7307 - accuracy: 0.4958
13184/25000 [==============>...............] - ETA: 30s - loss: 7.7294 - accuracy: 0.4959
13216/25000 [==============>...............] - ETA: 30s - loss: 7.7304 - accuracy: 0.4958
13248/25000 [==============>...............] - ETA: 29s - loss: 7.7280 - accuracy: 0.4960
13280/25000 [==============>...............] - ETA: 29s - loss: 7.7359 - accuracy: 0.4955
13312/25000 [==============>...............] - ETA: 29s - loss: 7.7346 - accuracy: 0.4956
13344/25000 [===============>..............] - ETA: 29s - loss: 7.7379 - accuracy: 0.4954
13376/25000 [===============>..............] - ETA: 29s - loss: 7.7388 - accuracy: 0.4953
13408/25000 [===============>..............] - ETA: 29s - loss: 7.7421 - accuracy: 0.4951
13440/25000 [===============>..............] - ETA: 29s - loss: 7.7351 - accuracy: 0.4955
13472/25000 [===============>..............] - ETA: 29s - loss: 7.7338 - accuracy: 0.4956
13504/25000 [===============>..............] - ETA: 29s - loss: 7.7359 - accuracy: 0.4955
13536/25000 [===============>..............] - ETA: 29s - loss: 7.7335 - accuracy: 0.4956
13568/25000 [===============>..............] - ETA: 29s - loss: 7.7356 - accuracy: 0.4955
13600/25000 [===============>..............] - ETA: 28s - loss: 7.7365 - accuracy: 0.4954
13632/25000 [===============>..............] - ETA: 28s - loss: 7.7364 - accuracy: 0.4955
13664/25000 [===============>..............] - ETA: 28s - loss: 7.7373 - accuracy: 0.4954
13696/25000 [===============>..............] - ETA: 28s - loss: 7.7316 - accuracy: 0.4958
13728/25000 [===============>..............] - ETA: 28s - loss: 7.7303 - accuracy: 0.4958
13760/25000 [===============>..............] - ETA: 28s - loss: 7.7335 - accuracy: 0.4956
13792/25000 [===============>..............] - ETA: 28s - loss: 7.7389 - accuracy: 0.4953
13824/25000 [===============>..............] - ETA: 28s - loss: 7.7398 - accuracy: 0.4952
13856/25000 [===============>..............] - ETA: 28s - loss: 7.7385 - accuracy: 0.4953
13888/25000 [===============>..............] - ETA: 28s - loss: 7.7406 - accuracy: 0.4952
13920/25000 [===============>..............] - ETA: 28s - loss: 7.7382 - accuracy: 0.4953
13952/25000 [===============>..............] - ETA: 28s - loss: 7.7392 - accuracy: 0.4953
13984/25000 [===============>..............] - ETA: 27s - loss: 7.7401 - accuracy: 0.4952
14016/25000 [===============>..............] - ETA: 27s - loss: 7.7388 - accuracy: 0.4953
14048/25000 [===============>..............] - ETA: 27s - loss: 7.7365 - accuracy: 0.4954
14080/25000 [===============>..............] - ETA: 27s - loss: 7.7363 - accuracy: 0.4955
14112/25000 [===============>..............] - ETA: 27s - loss: 7.7383 - accuracy: 0.4953
14144/25000 [===============>..............] - ETA: 27s - loss: 7.7382 - accuracy: 0.4953
14176/25000 [================>.............] - ETA: 27s - loss: 7.7413 - accuracy: 0.4951
14208/25000 [================>.............] - ETA: 27s - loss: 7.7389 - accuracy: 0.4953
14240/25000 [================>.............] - ETA: 27s - loss: 7.7388 - accuracy: 0.4953
14272/25000 [================>.............] - ETA: 27s - loss: 7.7365 - accuracy: 0.4954
14304/25000 [================>.............] - ETA: 27s - loss: 7.7395 - accuracy: 0.4952
14336/25000 [================>.............] - ETA: 27s - loss: 7.7372 - accuracy: 0.4954
14368/25000 [================>.............] - ETA: 26s - loss: 7.7360 - accuracy: 0.4955
14400/25000 [================>.............] - ETA: 26s - loss: 7.7348 - accuracy: 0.4956
14432/25000 [================>.............] - ETA: 26s - loss: 7.7357 - accuracy: 0.4955
14464/25000 [================>.............] - ETA: 26s - loss: 7.7376 - accuracy: 0.4954
14496/25000 [================>.............] - ETA: 26s - loss: 7.7417 - accuracy: 0.4951
14528/25000 [================>.............] - ETA: 26s - loss: 7.7426 - accuracy: 0.4950
14560/25000 [================>.............] - ETA: 26s - loss: 7.7393 - accuracy: 0.4953
14592/25000 [================>.............] - ETA: 26s - loss: 7.7433 - accuracy: 0.4950
14624/25000 [================>.............] - ETA: 26s - loss: 7.7390 - accuracy: 0.4953
14656/25000 [================>.............] - ETA: 26s - loss: 7.7378 - accuracy: 0.4954
14688/25000 [================>.............] - ETA: 26s - loss: 7.7324 - accuracy: 0.4957
14720/25000 [================>.............] - ETA: 26s - loss: 7.7302 - accuracy: 0.4959
14752/25000 [================>.............] - ETA: 25s - loss: 7.7279 - accuracy: 0.4960
14784/25000 [================>.............] - ETA: 25s - loss: 7.7237 - accuracy: 0.4963
14816/25000 [================>.............] - ETA: 25s - loss: 7.7225 - accuracy: 0.4964
14848/25000 [================>.............] - ETA: 25s - loss: 7.7234 - accuracy: 0.4963
14880/25000 [================>.............] - ETA: 25s - loss: 7.7274 - accuracy: 0.4960
14912/25000 [================>.............] - ETA: 25s - loss: 7.7242 - accuracy: 0.4962
14944/25000 [================>.............] - ETA: 25s - loss: 7.7251 - accuracy: 0.4962
14976/25000 [================>.............] - ETA: 25s - loss: 7.7260 - accuracy: 0.4961
15008/25000 [=================>............] - ETA: 25s - loss: 7.7249 - accuracy: 0.4962
15040/25000 [=================>............] - ETA: 25s - loss: 7.7278 - accuracy: 0.4960
15072/25000 [=================>............] - ETA: 25s - loss: 7.7287 - accuracy: 0.4960
15104/25000 [=================>............] - ETA: 25s - loss: 7.7245 - accuracy: 0.4962
15136/25000 [=================>............] - ETA: 24s - loss: 7.7213 - accuracy: 0.4964
15168/25000 [=================>............] - ETA: 24s - loss: 7.7212 - accuracy: 0.4964
15200/25000 [=================>............] - ETA: 24s - loss: 7.7221 - accuracy: 0.4964
15232/25000 [=================>............] - ETA: 24s - loss: 7.7240 - accuracy: 0.4963
15264/25000 [=================>............] - ETA: 24s - loss: 7.7299 - accuracy: 0.4959
15296/25000 [=================>............] - ETA: 24s - loss: 7.7248 - accuracy: 0.4962
15328/25000 [=================>............] - ETA: 24s - loss: 7.7246 - accuracy: 0.4962
15360/25000 [=================>............] - ETA: 24s - loss: 7.7255 - accuracy: 0.4962
15392/25000 [=================>............] - ETA: 24s - loss: 7.7284 - accuracy: 0.4960
15424/25000 [=================>............] - ETA: 24s - loss: 7.7283 - accuracy: 0.4960
15456/25000 [=================>............] - ETA: 24s - loss: 7.7252 - accuracy: 0.4962
15488/25000 [=================>............] - ETA: 24s - loss: 7.7270 - accuracy: 0.4961
15520/25000 [=================>............] - ETA: 23s - loss: 7.7289 - accuracy: 0.4959
15552/25000 [=================>............] - ETA: 23s - loss: 7.7287 - accuracy: 0.4959
15584/25000 [=================>............] - ETA: 23s - loss: 7.7247 - accuracy: 0.4962
15616/25000 [=================>............] - ETA: 23s - loss: 7.7295 - accuracy: 0.4959
15648/25000 [=================>............] - ETA: 23s - loss: 7.7293 - accuracy: 0.4959
15680/25000 [=================>............] - ETA: 23s - loss: 7.7312 - accuracy: 0.4958
15712/25000 [=================>............] - ETA: 23s - loss: 7.7330 - accuracy: 0.4957
15744/25000 [=================>............] - ETA: 23s - loss: 7.7299 - accuracy: 0.4959
15776/25000 [=================>............] - ETA: 23s - loss: 7.7317 - accuracy: 0.4958
15808/25000 [=================>............] - ETA: 23s - loss: 7.7306 - accuracy: 0.4958
15840/25000 [==================>...........] - ETA: 23s - loss: 7.7305 - accuracy: 0.4958
15872/25000 [==================>...........] - ETA: 23s - loss: 7.7333 - accuracy: 0.4957
15904/25000 [==================>...........] - ETA: 22s - loss: 7.7341 - accuracy: 0.4956
15936/25000 [==================>...........] - ETA: 22s - loss: 7.7378 - accuracy: 0.4954
15968/25000 [==================>...........] - ETA: 22s - loss: 7.7386 - accuracy: 0.4953
16000/25000 [==================>...........] - ETA: 22s - loss: 7.7423 - accuracy: 0.4951
16032/25000 [==================>...........] - ETA: 22s - loss: 7.7431 - accuracy: 0.4950
16064/25000 [==================>...........] - ETA: 22s - loss: 7.7439 - accuracy: 0.4950
16096/25000 [==================>...........] - ETA: 22s - loss: 7.7428 - accuracy: 0.4950
16128/25000 [==================>...........] - ETA: 22s - loss: 7.7398 - accuracy: 0.4952
16160/25000 [==================>...........] - ETA: 22s - loss: 7.7359 - accuracy: 0.4955
16192/25000 [==================>...........] - ETA: 22s - loss: 7.7357 - accuracy: 0.4955
16224/25000 [==================>...........] - ETA: 22s - loss: 7.7337 - accuracy: 0.4956
16256/25000 [==================>...........] - ETA: 22s - loss: 7.7355 - accuracy: 0.4955
16288/25000 [==================>...........] - ETA: 21s - loss: 7.7353 - accuracy: 0.4955
16320/25000 [==================>...........] - ETA: 21s - loss: 7.7361 - accuracy: 0.4955
16352/25000 [==================>...........] - ETA: 21s - loss: 7.7313 - accuracy: 0.4958
16384/25000 [==================>...........] - ETA: 21s - loss: 7.7256 - accuracy: 0.4962
16416/25000 [==================>...........] - ETA: 21s - loss: 7.7264 - accuracy: 0.4961
16448/25000 [==================>...........] - ETA: 21s - loss: 7.7244 - accuracy: 0.4962
16480/25000 [==================>...........] - ETA: 21s - loss: 7.7224 - accuracy: 0.4964
16512/25000 [==================>...........] - ETA: 21s - loss: 7.7205 - accuracy: 0.4965
16544/25000 [==================>...........] - ETA: 21s - loss: 7.7167 - accuracy: 0.4967
16576/25000 [==================>...........] - ETA: 21s - loss: 7.7212 - accuracy: 0.4964
16608/25000 [==================>...........] - ETA: 21s - loss: 7.7248 - accuracy: 0.4962
16640/25000 [==================>...........] - ETA: 21s - loss: 7.7247 - accuracy: 0.4962
16672/25000 [===================>..........] - ETA: 21s - loss: 7.7282 - accuracy: 0.4960
16704/25000 [===================>..........] - ETA: 20s - loss: 7.7290 - accuracy: 0.4959
16736/25000 [===================>..........] - ETA: 20s - loss: 7.7271 - accuracy: 0.4961
16768/25000 [===================>..........] - ETA: 20s - loss: 7.7224 - accuracy: 0.4964
16800/25000 [===================>..........] - ETA: 20s - loss: 7.7232 - accuracy: 0.4963
16832/25000 [===================>..........] - ETA: 20s - loss: 7.7231 - accuracy: 0.4963
16864/25000 [===================>..........] - ETA: 20s - loss: 7.7203 - accuracy: 0.4965
16896/25000 [===================>..........] - ETA: 20s - loss: 7.7211 - accuracy: 0.4964
16928/25000 [===================>..........] - ETA: 20s - loss: 7.7201 - accuracy: 0.4965
16960/25000 [===================>..........] - ETA: 20s - loss: 7.7227 - accuracy: 0.4963
16992/25000 [===================>..........] - ETA: 20s - loss: 7.7199 - accuracy: 0.4965
17024/25000 [===================>..........] - ETA: 20s - loss: 7.7171 - accuracy: 0.4967
17056/25000 [===================>..........] - ETA: 20s - loss: 7.7143 - accuracy: 0.4969
17088/25000 [===================>..........] - ETA: 19s - loss: 7.7142 - accuracy: 0.4969
17120/25000 [===================>..........] - ETA: 19s - loss: 7.7141 - accuracy: 0.4969
17152/25000 [===================>..........] - ETA: 19s - loss: 7.7113 - accuracy: 0.4971
17184/25000 [===================>..........] - ETA: 19s - loss: 7.7112 - accuracy: 0.4971
17216/25000 [===================>..........] - ETA: 19s - loss: 7.7085 - accuracy: 0.4973
17248/25000 [===================>..........] - ETA: 19s - loss: 7.7084 - accuracy: 0.4973
17280/25000 [===================>..........] - ETA: 19s - loss: 7.7065 - accuracy: 0.4974
17312/25000 [===================>..........] - ETA: 19s - loss: 7.7100 - accuracy: 0.4972
17344/25000 [===================>..........] - ETA: 19s - loss: 7.7126 - accuracy: 0.4970
17376/25000 [===================>..........] - ETA: 19s - loss: 7.7187 - accuracy: 0.4966
17408/25000 [===================>..........] - ETA: 19s - loss: 7.7186 - accuracy: 0.4966
17440/25000 [===================>..........] - ETA: 19s - loss: 7.7211 - accuracy: 0.4964
17472/25000 [===================>..........] - ETA: 18s - loss: 7.7237 - accuracy: 0.4963
17504/25000 [====================>.........] - ETA: 18s - loss: 7.7236 - accuracy: 0.4963
17536/25000 [====================>.........] - ETA: 18s - loss: 7.7235 - accuracy: 0.4963
17568/25000 [====================>.........] - ETA: 18s - loss: 7.7242 - accuracy: 0.4962
17600/25000 [====================>.........] - ETA: 18s - loss: 7.7224 - accuracy: 0.4964
17632/25000 [====================>.........] - ETA: 18s - loss: 7.7205 - accuracy: 0.4965
17664/25000 [====================>.........] - ETA: 18s - loss: 7.7196 - accuracy: 0.4965
17696/25000 [====================>.........] - ETA: 18s - loss: 7.7195 - accuracy: 0.4966
17728/25000 [====================>.........] - ETA: 18s - loss: 7.7194 - accuracy: 0.4966
17760/25000 [====================>.........] - ETA: 18s - loss: 7.7219 - accuracy: 0.4964
17792/25000 [====================>.........] - ETA: 18s - loss: 7.7244 - accuracy: 0.4962
17824/25000 [====================>.........] - ETA: 18s - loss: 7.7208 - accuracy: 0.4965
17856/25000 [====================>.........] - ETA: 17s - loss: 7.7199 - accuracy: 0.4965
17888/25000 [====================>.........] - ETA: 17s - loss: 7.7163 - accuracy: 0.4968
17920/25000 [====================>.........] - ETA: 17s - loss: 7.7120 - accuracy: 0.4970
17952/25000 [====================>.........] - ETA: 17s - loss: 7.7110 - accuracy: 0.4971
17984/25000 [====================>.........] - ETA: 17s - loss: 7.7084 - accuracy: 0.4973
18016/25000 [====================>.........] - ETA: 17s - loss: 7.7109 - accuracy: 0.4971
18048/25000 [====================>.........] - ETA: 17s - loss: 7.7125 - accuracy: 0.4970
18080/25000 [====================>.........] - ETA: 17s - loss: 7.7090 - accuracy: 0.4972
18112/25000 [====================>.........] - ETA: 17s - loss: 7.7073 - accuracy: 0.4973
18144/25000 [====================>.........] - ETA: 17s - loss: 7.7114 - accuracy: 0.4971
18176/25000 [====================>.........] - ETA: 17s - loss: 7.7096 - accuracy: 0.4972
18208/25000 [====================>.........] - ETA: 17s - loss: 7.7087 - accuracy: 0.4973
18240/25000 [====================>.........] - ETA: 16s - loss: 7.7112 - accuracy: 0.4971
18272/25000 [====================>.........] - ETA: 16s - loss: 7.7094 - accuracy: 0.4972
18304/25000 [====================>.........] - ETA: 16s - loss: 7.7093 - accuracy: 0.4972
18336/25000 [=====================>........] - ETA: 16s - loss: 7.7109 - accuracy: 0.4971
18368/25000 [=====================>........] - ETA: 16s - loss: 7.7100 - accuracy: 0.4972
18400/25000 [=====================>........] - ETA: 16s - loss: 7.7083 - accuracy: 0.4973
18432/25000 [=====================>........] - ETA: 16s - loss: 7.7099 - accuracy: 0.4972
18464/25000 [=====================>........] - ETA: 16s - loss: 7.7090 - accuracy: 0.4972
18496/25000 [=====================>........] - ETA: 16s - loss: 7.7081 - accuracy: 0.4973
18528/25000 [=====================>........] - ETA: 16s - loss: 7.7055 - accuracy: 0.4975
18560/25000 [=====================>........] - ETA: 16s - loss: 7.7063 - accuracy: 0.4974
18592/25000 [=====================>........] - ETA: 16s - loss: 7.7120 - accuracy: 0.4970
18624/25000 [=====================>........] - ETA: 16s - loss: 7.7103 - accuracy: 0.4972
18656/25000 [=====================>........] - ETA: 15s - loss: 7.7110 - accuracy: 0.4971
18688/25000 [=====================>........] - ETA: 15s - loss: 7.7117 - accuracy: 0.4971
18720/25000 [=====================>........] - ETA: 15s - loss: 7.7158 - accuracy: 0.4968
18752/25000 [=====================>........] - ETA: 15s - loss: 7.7157 - accuracy: 0.4968
18784/25000 [=====================>........] - ETA: 15s - loss: 7.7180 - accuracy: 0.4966
18816/25000 [=====================>........] - ETA: 15s - loss: 7.7163 - accuracy: 0.4968
18848/25000 [=====================>........] - ETA: 15s - loss: 7.7130 - accuracy: 0.4970
18880/25000 [=====================>........] - ETA: 15s - loss: 7.7129 - accuracy: 0.4970
18912/25000 [=====================>........] - ETA: 15s - loss: 7.7096 - accuracy: 0.4972
18944/25000 [=====================>........] - ETA: 15s - loss: 7.7119 - accuracy: 0.4970
18976/25000 [=====================>........] - ETA: 15s - loss: 7.7143 - accuracy: 0.4969
19008/25000 [=====================>........] - ETA: 15s - loss: 7.7134 - accuracy: 0.4969
19040/25000 [=====================>........] - ETA: 14s - loss: 7.7101 - accuracy: 0.4972
19072/25000 [=====================>........] - ETA: 14s - loss: 7.7084 - accuracy: 0.4973
19104/25000 [=====================>........] - ETA: 14s - loss: 7.7124 - accuracy: 0.4970
19136/25000 [=====================>........] - ETA: 14s - loss: 7.7123 - accuracy: 0.4970
19168/25000 [======================>.......] - ETA: 14s - loss: 7.7138 - accuracy: 0.4969
19200/25000 [======================>.......] - ETA: 14s - loss: 7.7161 - accuracy: 0.4968
19232/25000 [======================>.......] - ETA: 14s - loss: 7.7137 - accuracy: 0.4969
19264/25000 [======================>.......] - ETA: 14s - loss: 7.7120 - accuracy: 0.4970
19296/25000 [======================>.......] - ETA: 14s - loss: 7.7103 - accuracy: 0.4971
19328/25000 [======================>.......] - ETA: 14s - loss: 7.7087 - accuracy: 0.4973
19360/25000 [======================>.......] - ETA: 14s - loss: 7.7070 - accuracy: 0.4974
19392/25000 [======================>.......] - ETA: 14s - loss: 7.7062 - accuracy: 0.4974
19424/25000 [======================>.......] - ETA: 13s - loss: 7.7085 - accuracy: 0.4973
19456/25000 [======================>.......] - ETA: 13s - loss: 7.7076 - accuracy: 0.4973
19488/25000 [======================>.......] - ETA: 13s - loss: 7.7083 - accuracy: 0.4973
19520/25000 [======================>.......] - ETA: 13s - loss: 7.7098 - accuracy: 0.4972
19552/25000 [======================>.......] - ETA: 13s - loss: 7.7058 - accuracy: 0.4974
19584/25000 [======================>.......] - ETA: 13s - loss: 7.7026 - accuracy: 0.4977
19616/25000 [======================>.......] - ETA: 13s - loss: 7.6955 - accuracy: 0.4981
19648/25000 [======================>.......] - ETA: 13s - loss: 7.6986 - accuracy: 0.4979
19680/25000 [======================>.......] - ETA: 13s - loss: 7.6993 - accuracy: 0.4979
19712/25000 [======================>.......] - ETA: 13s - loss: 7.6970 - accuracy: 0.4980
19744/25000 [======================>.......] - ETA: 13s - loss: 7.7008 - accuracy: 0.4978
19776/25000 [======================>.......] - ETA: 13s - loss: 7.7046 - accuracy: 0.4975
19808/25000 [======================>.......] - ETA: 13s - loss: 7.7053 - accuracy: 0.4975
19840/25000 [======================>.......] - ETA: 12s - loss: 7.7060 - accuracy: 0.4974
19872/25000 [======================>.......] - ETA: 12s - loss: 7.7067 - accuracy: 0.4974
19904/25000 [======================>.......] - ETA: 12s - loss: 7.7044 - accuracy: 0.4975
19936/25000 [======================>.......] - ETA: 12s - loss: 7.7097 - accuracy: 0.4972
19968/25000 [======================>.......] - ETA: 12s - loss: 7.7058 - accuracy: 0.4974
20000/25000 [=======================>......] - ETA: 12s - loss: 7.7065 - accuracy: 0.4974
20032/25000 [=======================>......] - ETA: 12s - loss: 7.7064 - accuracy: 0.4974
20064/25000 [=======================>......] - ETA: 12s - loss: 7.7109 - accuracy: 0.4971
20096/25000 [=======================>......] - ETA: 12s - loss: 7.7086 - accuracy: 0.4973
20128/25000 [=======================>......] - ETA: 12s - loss: 7.7055 - accuracy: 0.4975
20160/25000 [=======================>......] - ETA: 12s - loss: 7.7024 - accuracy: 0.4977
20192/25000 [=======================>......] - ETA: 12s - loss: 7.7038 - accuracy: 0.4976
20224/25000 [=======================>......] - ETA: 11s - loss: 7.7000 - accuracy: 0.4978
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6999 - accuracy: 0.4978
20288/25000 [=======================>......] - ETA: 11s - loss: 7.7014 - accuracy: 0.4977
20320/25000 [=======================>......] - ETA: 11s - loss: 7.7036 - accuracy: 0.4976
20352/25000 [=======================>......] - ETA: 11s - loss: 7.7005 - accuracy: 0.4978
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6960 - accuracy: 0.4981
20416/25000 [=======================>......] - ETA: 11s - loss: 7.6937 - accuracy: 0.4982
20448/25000 [=======================>......] - ETA: 11s - loss: 7.6914 - accuracy: 0.4984
20480/25000 [=======================>......] - ETA: 11s - loss: 7.6921 - accuracy: 0.4983
20512/25000 [=======================>......] - ETA: 11s - loss: 7.6898 - accuracy: 0.4985
20544/25000 [=======================>......] - ETA: 11s - loss: 7.6905 - accuracy: 0.4984
20576/25000 [=======================>......] - ETA: 11s - loss: 7.6890 - accuracy: 0.4985
20608/25000 [=======================>......] - ETA: 11s - loss: 7.6889 - accuracy: 0.4985
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6882 - accuracy: 0.4986
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6911 - accuracy: 0.4984
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6918 - accuracy: 0.4984
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6940 - accuracy: 0.4982
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6998 - accuracy: 0.4978
20800/25000 [=======================>......] - ETA: 10s - loss: 7.7020 - accuracy: 0.4977
20832/25000 [=======================>......] - ETA: 10s - loss: 7.7042 - accuracy: 0.4976
20864/25000 [========================>.....] - ETA: 10s - loss: 7.7048 - accuracy: 0.4975
20896/25000 [========================>.....] - ETA: 10s - loss: 7.7018 - accuracy: 0.4977
20928/25000 [========================>.....] - ETA: 10s - loss: 7.7033 - accuracy: 0.4976
20960/25000 [========================>.....] - ETA: 10s - loss: 7.7039 - accuracy: 0.4976
20992/25000 [========================>.....] - ETA: 10s - loss: 7.7039 - accuracy: 0.4976
21024/25000 [========================>.....] - ETA: 9s - loss: 7.7067 - accuracy: 0.4974 
21056/25000 [========================>.....] - ETA: 9s - loss: 7.7045 - accuracy: 0.4975
21088/25000 [========================>.....] - ETA: 9s - loss: 7.7030 - accuracy: 0.4976
21120/25000 [========================>.....] - ETA: 9s - loss: 7.7029 - accuracy: 0.4976
21152/25000 [========================>.....] - ETA: 9s - loss: 7.7043 - accuracy: 0.4975
21184/25000 [========================>.....] - ETA: 9s - loss: 7.7057 - accuracy: 0.4975
21216/25000 [========================>.....] - ETA: 9s - loss: 7.7093 - accuracy: 0.4972
21248/25000 [========================>.....] - ETA: 9s - loss: 7.7078 - accuracy: 0.4973
21280/25000 [========================>.....] - ETA: 9s - loss: 7.7091 - accuracy: 0.4972
21312/25000 [========================>.....] - ETA: 9s - loss: 7.7091 - accuracy: 0.4972
21344/25000 [========================>.....] - ETA: 9s - loss: 7.7083 - accuracy: 0.4973
21376/25000 [========================>.....] - ETA: 9s - loss: 7.7097 - accuracy: 0.4972
21408/25000 [========================>.....] - ETA: 8s - loss: 7.7096 - accuracy: 0.4972
21440/25000 [========================>.....] - ETA: 8s - loss: 7.7067 - accuracy: 0.4974
21472/25000 [========================>.....] - ETA: 8s - loss: 7.7030 - accuracy: 0.4976
21504/25000 [========================>.....] - ETA: 8s - loss: 7.7030 - accuracy: 0.4976
21536/25000 [========================>.....] - ETA: 8s - loss: 7.7036 - accuracy: 0.4976
21568/25000 [========================>.....] - ETA: 8s - loss: 7.7022 - accuracy: 0.4977
21600/25000 [========================>.....] - ETA: 8s - loss: 7.7014 - accuracy: 0.4977
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6992 - accuracy: 0.4979
21664/25000 [========================>.....] - ETA: 8s - loss: 7.6999 - accuracy: 0.4978
21696/25000 [=========================>....] - ETA: 8s - loss: 7.6984 - accuracy: 0.4979
21728/25000 [=========================>....] - ETA: 8s - loss: 7.6984 - accuracy: 0.4979
21760/25000 [=========================>....] - ETA: 8s - loss: 7.6990 - accuracy: 0.4979
21792/25000 [=========================>....] - ETA: 8s - loss: 7.7004 - accuracy: 0.4978
21824/25000 [=========================>....] - ETA: 7s - loss: 7.7003 - accuracy: 0.4978
21856/25000 [=========================>....] - ETA: 7s - loss: 7.7003 - accuracy: 0.4978
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6988 - accuracy: 0.4979
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6932 - accuracy: 0.4983
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6939 - accuracy: 0.4982
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6903 - accuracy: 0.4985
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6882 - accuracy: 0.4986
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6896 - accuracy: 0.4985
22080/25000 [=========================>....] - ETA: 7s - loss: 7.6895 - accuracy: 0.4985
22112/25000 [=========================>....] - ETA: 7s - loss: 7.6888 - accuracy: 0.4986
22144/25000 [=========================>....] - ETA: 7s - loss: 7.6888 - accuracy: 0.4986
22176/25000 [=========================>....] - ETA: 7s - loss: 7.6881 - accuracy: 0.4986
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6860 - accuracy: 0.4987
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6825 - accuracy: 0.4990
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6769 - accuracy: 0.4993
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6769 - accuracy: 0.4993
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6762 - accuracy: 0.4994
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6762 - accuracy: 0.4994
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6748 - accuracy: 0.4995
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6769 - accuracy: 0.4993
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6762 - accuracy: 0.4994
22496/25000 [=========================>....] - ETA: 6s - loss: 7.6768 - accuracy: 0.4993
22528/25000 [==========================>...] - ETA: 6s - loss: 7.6734 - accuracy: 0.4996
22560/25000 [==========================>...] - ETA: 6s - loss: 7.6734 - accuracy: 0.4996
22592/25000 [==========================>...] - ETA: 6s - loss: 7.6741 - accuracy: 0.4995
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6707 - accuracy: 0.4997
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6700 - accuracy: 0.4998
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6714 - accuracy: 0.4997
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6727 - accuracy: 0.4996
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6720 - accuracy: 0.4996
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6686 - accuracy: 0.4999
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6659 - accuracy: 0.5000
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6659 - accuracy: 0.5000
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6653 - accuracy: 0.5001
22912/25000 [==========================>...] - ETA: 5s - loss: 7.6680 - accuracy: 0.4999
22944/25000 [==========================>...] - ETA: 5s - loss: 7.6660 - accuracy: 0.5000
22976/25000 [==========================>...] - ETA: 5s - loss: 7.6700 - accuracy: 0.4998
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6720 - accuracy: 0.4997
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6726 - accuracy: 0.4996
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6726 - accuracy: 0.4996
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6726 - accuracy: 0.4996
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6732 - accuracy: 0.4996
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6759 - accuracy: 0.4994
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6739 - accuracy: 0.4995
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6732 - accuracy: 0.4996
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6732 - accuracy: 0.4996
23328/25000 [==========================>...] - ETA: 4s - loss: 7.6719 - accuracy: 0.4997
23360/25000 [===========================>..] - ETA: 4s - loss: 7.6725 - accuracy: 0.4996
23392/25000 [===========================>..] - ETA: 4s - loss: 7.6732 - accuracy: 0.4996
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6764 - accuracy: 0.4994
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6751 - accuracy: 0.4994
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6731 - accuracy: 0.4996
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6718 - accuracy: 0.4997
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6660 - accuracy: 0.5000
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6660 - accuracy: 0.5000
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6679 - accuracy: 0.4999
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6705 - accuracy: 0.4997
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6724 - accuracy: 0.4996
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6731 - accuracy: 0.4996
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6737 - accuracy: 0.4995
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6718 - accuracy: 0.4997
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6718 - accuracy: 0.4997
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6737 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6724 - accuracy: 0.4996
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6756 - accuracy: 0.4994
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6723 - accuracy: 0.4996
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24192/25000 [============================>.] - ETA: 2s - loss: 7.6742 - accuracy: 0.4995
24224/25000 [============================>.] - ETA: 1s - loss: 7.6748 - accuracy: 0.4995
24256/25000 [============================>.] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
24288/25000 [============================>.] - ETA: 1s - loss: 7.6685 - accuracy: 0.4999
24320/25000 [============================>.] - ETA: 1s - loss: 7.6691 - accuracy: 0.4998
24352/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24384/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24416/25000 [============================>.] - ETA: 1s - loss: 7.6635 - accuracy: 0.5002
24448/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24480/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5004
24512/25000 [============================>.] - ETA: 1s - loss: 7.6591 - accuracy: 0.5005
24544/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5004
24576/25000 [============================>.] - ETA: 1s - loss: 7.6591 - accuracy: 0.5005
24608/25000 [============================>.] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24640/25000 [============================>.] - ETA: 0s - loss: 7.6616 - accuracy: 0.5003
24672/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24704/25000 [============================>.] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
24736/25000 [============================>.] - ETA: 0s - loss: 7.6592 - accuracy: 0.5005
24768/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24800/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24832/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24864/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24896/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24928/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24960/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24992/25000 [============================>.] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 73s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

