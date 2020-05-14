
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
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3862
 40%|████      | 2/5 [00:48<01:12, 24.26s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 12% CPU time recently (threshold: 10%)
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
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.293751472551545, 'embedding_size_factor': 1.3195856147882852, 'layers.choice': 3, 'learning_rate': 0.0015311234474665858, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 1.5045899200079102e-10} and reward: 0.3822
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd2\xcc\xd2\xf9\xf0\xaf\xc2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x1d\x05\xce<\x9a\x94X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?Y\x15\xffH\x832\xfbX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xe4\xad\xce\x1e\xe0\xdcgu.' and reward: 0.3822
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd2\xcc\xd2\xf9\xf0\xaf\xc2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x1d\x05\xce<\x9a\x94X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?Y\x15\xffH\x832\xfbX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xe4\xad\xce\x1e\xe0\xdcgu.' and reward: 0.3822
 60%|██████    | 3/5 [01:38<01:04, 32.05s/it] 60%|██████    | 3/5 [01:38<01:05, 32.92s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.32189156946503833, 'embedding_size_factor': 0.9750600938386068, 'layers.choice': 3, 'learning_rate': 0.000223478763321836, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 7.492986031365116e-07} and reward: 0.3766
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\x99\xdf\x18\xedx\x0bX\x15\x00\x00\x00embedding_size_factorq\x03G?\xef3\xb19\xd5|\xfdX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?-J\xb3\xf5\xaa\xae\x13X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xa9$m\r\xbb\xfe\x95u.' and reward: 0.3766
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\x99\xdf\x18\xedx\x0bX\x15\x00\x00\x00embedding_size_factorq\x03G?\xef3\xb19\xd5|\xfdX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?-J\xb3\xf5\xaa\xae\x13X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xa9$m\r\xbb\xfe\x95u.' and reward: 0.3766
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 149.39694261550903
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -31.73s of remaining time.
Ensemble size: 43
Ensemble weights: 
[0.60465116 0.37209302 0.02325581]
	0.39	 = Validation accuracy score
	1.02s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 152.78s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f91500d2a90> 

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
 [ 0.04128773  0.10885572 -0.06116501  0.10558589  0.10639104 -0.04148959]
 [ 0.21846889 -0.03049701  0.16783857 -0.1547343   0.04192238 -0.16792554]
 [-0.01228589  0.10038985  0.30360919 -0.05510798  0.00758401 -0.05567868]
 [ 0.37760833 -0.03066965  0.03076871 -0.00809154  0.21312994  0.0586692 ]
 [ 0.13193589  0.0077732  -0.39577881 -0.34464696 -0.38735867 -0.01595221]
 [-0.06904428  0.47573996 -0.14353618  0.08076371  0.25152183  0.03443186]
 [ 0.17983848  0.07144848 -0.16532651  0.01161264  0.05039234  0.11979762]
 [ 0.20382978  0.18597697 -0.10549703 -0.24296497  0.28688368  0.29569757]
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
{'loss': 0.4330526404082775, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-14 00:23:37.620781: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.4995996877551079, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-14 00:23:38.674719: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
  548864/17464789 [..............................] - ETA: 1s
 8986624/17464789 [==============>...............] - ETA: 0s
16695296/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 00:23:49.508476: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-14 00:23:49.512438: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-14 00:23:49.513163: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b4058059a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 00:23:49.513180: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:37 - loss: 7.6666 - accuracy: 0.5000
   64/25000 [..............................] - ETA: 2:48 - loss: 8.1458 - accuracy: 0.4688
   96/25000 [..............................] - ETA: 2:12 - loss: 7.3472 - accuracy: 0.5208
  128/25000 [..............................] - ETA: 1:53 - loss: 7.3072 - accuracy: 0.5234
  160/25000 [..............................] - ETA: 1:41 - loss: 7.2833 - accuracy: 0.5250
  192/25000 [..............................] - ETA: 1:34 - loss: 7.5069 - accuracy: 0.5104
  224/25000 [..............................] - ETA: 1:28 - loss: 7.8035 - accuracy: 0.4911
  256/25000 [..............................] - ETA: 1:24 - loss: 7.4270 - accuracy: 0.5156
  288/25000 [..............................] - ETA: 1:21 - loss: 7.3472 - accuracy: 0.5208
  320/25000 [..............................] - ETA: 1:19 - loss: 7.0916 - accuracy: 0.5375
  352/25000 [..............................] - ETA: 1:16 - loss: 6.9697 - accuracy: 0.5455
  384/25000 [..............................] - ETA: 1:15 - loss: 7.3072 - accuracy: 0.5234
  416/25000 [..............................] - ETA: 1:13 - loss: 7.2612 - accuracy: 0.5264
  448/25000 [..............................] - ETA: 1:12 - loss: 7.2217 - accuracy: 0.5290
  480/25000 [..............................] - ETA: 1:11 - loss: 6.9958 - accuracy: 0.5437
  512/25000 [..............................] - ETA: 1:10 - loss: 7.1276 - accuracy: 0.5352
  544/25000 [..............................] - ETA: 1:09 - loss: 7.2438 - accuracy: 0.5276
  576/25000 [..............................] - ETA: 1:09 - loss: 7.2939 - accuracy: 0.5243
  608/25000 [..............................] - ETA: 1:08 - loss: 7.3388 - accuracy: 0.5214
  640/25000 [..............................] - ETA: 1:07 - loss: 7.3552 - accuracy: 0.5203
  672/25000 [..............................] - ETA: 1:06 - loss: 7.3472 - accuracy: 0.5208
  704/25000 [..............................] - ETA: 1:06 - loss: 7.3835 - accuracy: 0.5185
  736/25000 [..............................] - ETA: 1:05 - loss: 7.4166 - accuracy: 0.5163
  768/25000 [..............................] - ETA: 1:05 - loss: 7.3871 - accuracy: 0.5182
  800/25000 [..............................] - ETA: 1:04 - loss: 7.3216 - accuracy: 0.5225
  832/25000 [..............................] - ETA: 1:04 - loss: 7.4455 - accuracy: 0.5144
  864/25000 [>.............................] - ETA: 1:03 - loss: 7.4359 - accuracy: 0.5150
  896/25000 [>.............................] - ETA: 1:03 - loss: 7.4613 - accuracy: 0.5134
  928/25000 [>.............................] - ETA: 1:03 - loss: 7.4683 - accuracy: 0.5129
  960/25000 [>.............................] - ETA: 1:02 - loss: 7.4111 - accuracy: 0.5167
  992/25000 [>.............................] - ETA: 1:02 - loss: 7.3575 - accuracy: 0.5202
 1024/25000 [>.............................] - ETA: 1:01 - loss: 7.2923 - accuracy: 0.5244
 1056/25000 [>.............................] - ETA: 1:01 - loss: 7.3036 - accuracy: 0.5237
 1088/25000 [>.............................] - ETA: 1:01 - loss: 7.2720 - accuracy: 0.5257
 1120/25000 [>.............................] - ETA: 1:01 - loss: 7.2833 - accuracy: 0.5250
 1152/25000 [>.............................] - ETA: 1:01 - loss: 7.3339 - accuracy: 0.5217
 1184/25000 [>.............................] - ETA: 1:01 - loss: 7.3817 - accuracy: 0.5186
 1216/25000 [>.............................] - ETA: 1:00 - loss: 7.3640 - accuracy: 0.5197
 1248/25000 [>.............................] - ETA: 1:00 - loss: 7.2980 - accuracy: 0.5240
 1280/25000 [>.............................] - ETA: 1:00 - loss: 7.2833 - accuracy: 0.5250
 1312/25000 [>.............................] - ETA: 1:00 - loss: 7.2926 - accuracy: 0.5244
 1344/25000 [>.............................] - ETA: 1:00 - loss: 7.3130 - accuracy: 0.5231
 1376/25000 [>.............................] - ETA: 59s - loss: 7.2989 - accuracy: 0.5240 
 1408/25000 [>.............................] - ETA: 59s - loss: 7.2746 - accuracy: 0.5256
 1440/25000 [>.............................] - ETA: 59s - loss: 7.3152 - accuracy: 0.5229
 1472/25000 [>.............................] - ETA: 59s - loss: 7.3229 - accuracy: 0.5224
 1504/25000 [>.............................] - ETA: 58s - loss: 7.3302 - accuracy: 0.5219
 1536/25000 [>.............................] - ETA: 58s - loss: 7.3272 - accuracy: 0.5221
 1568/25000 [>.............................] - ETA: 58s - loss: 7.3146 - accuracy: 0.5230
 1600/25000 [>.............................] - ETA: 58s - loss: 7.3312 - accuracy: 0.5219
 1632/25000 [>.............................] - ETA: 58s - loss: 7.3190 - accuracy: 0.5227
 1664/25000 [>.............................] - ETA: 58s - loss: 7.3349 - accuracy: 0.5216
 1696/25000 [=>............................] - ETA: 57s - loss: 7.3864 - accuracy: 0.5183
 1728/25000 [=>............................] - ETA: 57s - loss: 7.3561 - accuracy: 0.5203
 1760/25000 [=>............................] - ETA: 57s - loss: 7.3965 - accuracy: 0.5176
 1792/25000 [=>............................] - ETA: 57s - loss: 7.4099 - accuracy: 0.5167
 1824/25000 [=>............................] - ETA: 57s - loss: 7.4565 - accuracy: 0.5137
 1856/25000 [=>............................] - ETA: 57s - loss: 7.4518 - accuracy: 0.5140
 1888/25000 [=>............................] - ETA: 57s - loss: 7.4473 - accuracy: 0.5143
 1920/25000 [=>............................] - ETA: 57s - loss: 7.4670 - accuracy: 0.5130
 1952/25000 [=>............................] - ETA: 57s - loss: 7.4781 - accuracy: 0.5123
 1984/25000 [=>............................] - ETA: 57s - loss: 7.5043 - accuracy: 0.5106
 2016/25000 [=>............................] - ETA: 56s - loss: 7.5221 - accuracy: 0.5094
 2048/25000 [=>............................] - ETA: 56s - loss: 7.4944 - accuracy: 0.5112
 2080/25000 [=>............................] - ETA: 56s - loss: 7.4897 - accuracy: 0.5115
 2112/25000 [=>............................] - ETA: 56s - loss: 7.4996 - accuracy: 0.5109
 2144/25000 [=>............................] - ETA: 56s - loss: 7.4878 - accuracy: 0.5117
 2176/25000 [=>............................] - ETA: 56s - loss: 7.5116 - accuracy: 0.5101
 2208/25000 [=>............................] - ETA: 56s - loss: 7.5416 - accuracy: 0.5082
 2240/25000 [=>............................] - ETA: 56s - loss: 7.5708 - accuracy: 0.5063
 2272/25000 [=>............................] - ETA: 55s - loss: 7.5316 - accuracy: 0.5088
 2304/25000 [=>............................] - ETA: 55s - loss: 7.5136 - accuracy: 0.5100
 2336/25000 [=>............................] - ETA: 55s - loss: 7.5091 - accuracy: 0.5103
 2368/25000 [=>............................] - ETA: 55s - loss: 7.5436 - accuracy: 0.5080
 2400/25000 [=>............................] - ETA: 55s - loss: 7.5261 - accuracy: 0.5092
 2432/25000 [=>............................] - ETA: 55s - loss: 7.5342 - accuracy: 0.5086
 2464/25000 [=>............................] - ETA: 55s - loss: 7.4924 - accuracy: 0.5114
 2496/25000 [=>............................] - ETA: 55s - loss: 7.5069 - accuracy: 0.5104
 2528/25000 [==>...........................] - ETA: 54s - loss: 7.5029 - accuracy: 0.5107
 2560/25000 [==>...........................] - ETA: 54s - loss: 7.5169 - accuracy: 0.5098
 2592/25000 [==>...........................] - ETA: 54s - loss: 7.5246 - accuracy: 0.5093
 2624/25000 [==>...........................] - ETA: 54s - loss: 7.5264 - accuracy: 0.5091
 2656/25000 [==>...........................] - ETA: 54s - loss: 7.4992 - accuracy: 0.5109
 2688/25000 [==>...........................] - ETA: 54s - loss: 7.5183 - accuracy: 0.5097
 2720/25000 [==>...........................] - ETA: 54s - loss: 7.5144 - accuracy: 0.5099
 2752/25000 [==>...........................] - ETA: 54s - loss: 7.5329 - accuracy: 0.5087
 2784/25000 [==>...........................] - ETA: 54s - loss: 7.5069 - accuracy: 0.5104
 2816/25000 [==>...........................] - ETA: 53s - loss: 7.5142 - accuracy: 0.5099
 2848/25000 [==>...........................] - ETA: 53s - loss: 7.4997 - accuracy: 0.5109
 2880/25000 [==>...........................] - ETA: 53s - loss: 7.5175 - accuracy: 0.5097
 2912/25000 [==>...........................] - ETA: 53s - loss: 7.5192 - accuracy: 0.5096
 2944/25000 [==>...........................] - ETA: 53s - loss: 7.5104 - accuracy: 0.5102
 2976/25000 [==>...........................] - ETA: 53s - loss: 7.4966 - accuracy: 0.5111
 3008/25000 [==>...........................] - ETA: 53s - loss: 7.4882 - accuracy: 0.5116
 3040/25000 [==>...........................] - ETA: 53s - loss: 7.4951 - accuracy: 0.5112
 3072/25000 [==>...........................] - ETA: 53s - loss: 7.4869 - accuracy: 0.5117
 3104/25000 [==>...........................] - ETA: 53s - loss: 7.4937 - accuracy: 0.5113
 3136/25000 [==>...........................] - ETA: 53s - loss: 7.4710 - accuracy: 0.5128
 3168/25000 [==>...........................] - ETA: 53s - loss: 7.4779 - accuracy: 0.5123
 3200/25000 [==>...........................] - ETA: 53s - loss: 7.4845 - accuracy: 0.5119
 3232/25000 [==>...........................] - ETA: 52s - loss: 7.4863 - accuracy: 0.5118
 3264/25000 [==>...........................] - ETA: 52s - loss: 7.5069 - accuracy: 0.5104
 3296/25000 [==>...........................] - ETA: 52s - loss: 7.4991 - accuracy: 0.5109
 3328/25000 [==>...........................] - ETA: 52s - loss: 7.4915 - accuracy: 0.5114
 3360/25000 [===>..........................] - ETA: 52s - loss: 7.4932 - accuracy: 0.5113
 3392/25000 [===>..........................] - ETA: 52s - loss: 7.4994 - accuracy: 0.5109
 3424/25000 [===>..........................] - ETA: 52s - loss: 7.5009 - accuracy: 0.5108
 3456/25000 [===>..........................] - ETA: 52s - loss: 7.5025 - accuracy: 0.5107
 3488/25000 [===>..........................] - ETA: 51s - loss: 7.5172 - accuracy: 0.5097
 3520/25000 [===>..........................] - ETA: 51s - loss: 7.5403 - accuracy: 0.5082
 3552/25000 [===>..........................] - ETA: 51s - loss: 7.5457 - accuracy: 0.5079
 3584/25000 [===>..........................] - ETA: 51s - loss: 7.5383 - accuracy: 0.5084
 3616/25000 [===>..........................] - ETA: 51s - loss: 7.5309 - accuracy: 0.5088
 3648/25000 [===>..........................] - ETA: 51s - loss: 7.5405 - accuracy: 0.5082
 3680/25000 [===>..........................] - ETA: 51s - loss: 7.5333 - accuracy: 0.5087
 3712/25000 [===>..........................] - ETA: 51s - loss: 7.5468 - accuracy: 0.5078
 3744/25000 [===>..........................] - ETA: 50s - loss: 7.5683 - accuracy: 0.5064
 3776/25000 [===>..........................] - ETA: 50s - loss: 7.5692 - accuracy: 0.5064
 3808/25000 [===>..........................] - ETA: 50s - loss: 7.5660 - accuracy: 0.5066
 3840/25000 [===>..........................] - ETA: 50s - loss: 7.5628 - accuracy: 0.5068
 3872/25000 [===>..........................] - ETA: 50s - loss: 7.5637 - accuracy: 0.5067
 3904/25000 [===>..........................] - ETA: 50s - loss: 7.5763 - accuracy: 0.5059
 3936/25000 [===>..........................] - ETA: 50s - loss: 7.5575 - accuracy: 0.5071
 3968/25000 [===>..........................] - ETA: 50s - loss: 7.5584 - accuracy: 0.5071
 4000/25000 [===>..........................] - ETA: 50s - loss: 7.5555 - accuracy: 0.5073
 4032/25000 [===>..........................] - ETA: 49s - loss: 7.5639 - accuracy: 0.5067
 4064/25000 [===>..........................] - ETA: 49s - loss: 7.5572 - accuracy: 0.5071
 4096/25000 [===>..........................] - ETA: 49s - loss: 7.5655 - accuracy: 0.5066
 4128/25000 [===>..........................] - ETA: 49s - loss: 7.5700 - accuracy: 0.5063
 4160/25000 [===>..........................] - ETA: 49s - loss: 7.5782 - accuracy: 0.5058
 4192/25000 [====>.........................] - ETA: 49s - loss: 7.5605 - accuracy: 0.5069
 4224/25000 [====>.........................] - ETA: 49s - loss: 7.5650 - accuracy: 0.5066
 4256/25000 [====>.........................] - ETA: 49s - loss: 7.5585 - accuracy: 0.5070
 4288/25000 [====>.........................] - ETA: 49s - loss: 7.5629 - accuracy: 0.5068
 4320/25000 [====>.........................] - ETA: 49s - loss: 7.5637 - accuracy: 0.5067
 4352/25000 [====>.........................] - ETA: 49s - loss: 7.5609 - accuracy: 0.5069
 4384/25000 [====>.........................] - ETA: 49s - loss: 7.5617 - accuracy: 0.5068
 4416/25000 [====>.........................] - ETA: 49s - loss: 7.5625 - accuracy: 0.5068
 4448/25000 [====>.........................] - ETA: 48s - loss: 7.5632 - accuracy: 0.5067
 4480/25000 [====>.........................] - ETA: 48s - loss: 7.5811 - accuracy: 0.5056
 4512/25000 [====>.........................] - ETA: 48s - loss: 7.6021 - accuracy: 0.5042
 4544/25000 [====>.........................] - ETA: 48s - loss: 7.6126 - accuracy: 0.5035
 4576/25000 [====>.........................] - ETA: 48s - loss: 7.6164 - accuracy: 0.5033
 4608/25000 [====>.........................] - ETA: 48s - loss: 7.6167 - accuracy: 0.5033
 4640/25000 [====>.........................] - ETA: 48s - loss: 7.6270 - accuracy: 0.5026
 4672/25000 [====>.........................] - ETA: 48s - loss: 7.6305 - accuracy: 0.5024
 4704/25000 [====>.........................] - ETA: 48s - loss: 7.6308 - accuracy: 0.5023
 4736/25000 [====>.........................] - ETA: 48s - loss: 7.6375 - accuracy: 0.5019
 4768/25000 [====>.........................] - ETA: 48s - loss: 7.6377 - accuracy: 0.5019
 4800/25000 [====>.........................] - ETA: 48s - loss: 7.6443 - accuracy: 0.5015
 4832/25000 [====>.........................] - ETA: 47s - loss: 7.6476 - accuracy: 0.5012
 4864/25000 [====>.........................] - ETA: 47s - loss: 7.6572 - accuracy: 0.5006
 4896/25000 [====>.........................] - ETA: 47s - loss: 7.6572 - accuracy: 0.5006
 4928/25000 [====>.........................] - ETA: 47s - loss: 7.6635 - accuracy: 0.5002
 4960/25000 [====>.........................] - ETA: 47s - loss: 7.6512 - accuracy: 0.5010
 4992/25000 [====>.........................] - ETA: 47s - loss: 7.6390 - accuracy: 0.5018
 5024/25000 [=====>........................] - ETA: 47s - loss: 7.6300 - accuracy: 0.5024
 5056/25000 [=====>........................] - ETA: 47s - loss: 7.6242 - accuracy: 0.5028
 5088/25000 [=====>........................] - ETA: 47s - loss: 7.6154 - accuracy: 0.5033
 5120/25000 [=====>........................] - ETA: 47s - loss: 7.6127 - accuracy: 0.5035
 5152/25000 [=====>........................] - ETA: 47s - loss: 7.6190 - accuracy: 0.5031
 5184/25000 [=====>........................] - ETA: 47s - loss: 7.6163 - accuracy: 0.5033
 5216/25000 [=====>........................] - ETA: 47s - loss: 7.6225 - accuracy: 0.5029
 5248/25000 [=====>........................] - ETA: 47s - loss: 7.6199 - accuracy: 0.5030
 5280/25000 [=====>........................] - ETA: 46s - loss: 7.6143 - accuracy: 0.5034
 5312/25000 [=====>........................] - ETA: 46s - loss: 7.6147 - accuracy: 0.5034
 5344/25000 [=====>........................] - ETA: 46s - loss: 7.6035 - accuracy: 0.5041
 5376/25000 [=====>........................] - ETA: 46s - loss: 7.5982 - accuracy: 0.5045
 5408/25000 [=====>........................] - ETA: 46s - loss: 7.6014 - accuracy: 0.5043
 5440/25000 [=====>........................] - ETA: 46s - loss: 7.5764 - accuracy: 0.5059
 5472/25000 [=====>........................] - ETA: 46s - loss: 7.5657 - accuracy: 0.5066
 5504/25000 [=====>........................] - ETA: 46s - loss: 7.5524 - accuracy: 0.5074
 5536/25000 [=====>........................] - ETA: 46s - loss: 7.5614 - accuracy: 0.5069
 5568/25000 [=====>........................] - ETA: 46s - loss: 7.5620 - accuracy: 0.5068
 5600/25000 [=====>........................] - ETA: 46s - loss: 7.5598 - accuracy: 0.5070
 5632/25000 [=====>........................] - ETA: 46s - loss: 7.5523 - accuracy: 0.5075
 5664/25000 [=====>........................] - ETA: 46s - loss: 7.5529 - accuracy: 0.5074
 5696/25000 [=====>........................] - ETA: 45s - loss: 7.5563 - accuracy: 0.5072
 5728/25000 [=====>........................] - ETA: 45s - loss: 7.5542 - accuracy: 0.5073
 5760/25000 [=====>........................] - ETA: 45s - loss: 7.5468 - accuracy: 0.5078
 5792/25000 [=====>........................] - ETA: 45s - loss: 7.5422 - accuracy: 0.5081
 5824/25000 [=====>........................] - ETA: 45s - loss: 7.5323 - accuracy: 0.5088
 5856/25000 [======>.......................] - ETA: 45s - loss: 7.5409 - accuracy: 0.5082
 5888/25000 [======>.......................] - ETA: 45s - loss: 7.5390 - accuracy: 0.5083
 5920/25000 [======>.......................] - ETA: 45s - loss: 7.5345 - accuracy: 0.5086
 5952/25000 [======>.......................] - ETA: 45s - loss: 7.5301 - accuracy: 0.5089
 5984/25000 [======>.......................] - ETA: 45s - loss: 7.5308 - accuracy: 0.5089
 6016/25000 [======>.......................] - ETA: 45s - loss: 7.5417 - accuracy: 0.5081
 6048/25000 [======>.......................] - ETA: 45s - loss: 7.5551 - accuracy: 0.5073
 6080/25000 [======>.......................] - ETA: 45s - loss: 7.5481 - accuracy: 0.5077
 6112/25000 [======>.......................] - ETA: 45s - loss: 7.5512 - accuracy: 0.5075
 6144/25000 [======>.......................] - ETA: 45s - loss: 7.5518 - accuracy: 0.5075
 6176/25000 [======>.......................] - ETA: 44s - loss: 7.5400 - accuracy: 0.5083
 6208/25000 [======>.......................] - ETA: 44s - loss: 7.5382 - accuracy: 0.5084
 6240/25000 [======>.......................] - ETA: 44s - loss: 7.5487 - accuracy: 0.5077
 6272/25000 [======>.......................] - ETA: 44s - loss: 7.5591 - accuracy: 0.5070
 6304/25000 [======>.......................] - ETA: 44s - loss: 7.5620 - accuracy: 0.5068
 6336/25000 [======>.......................] - ETA: 44s - loss: 7.5601 - accuracy: 0.5069
 6368/25000 [======>.......................] - ETA: 44s - loss: 7.5607 - accuracy: 0.5069
 6400/25000 [======>.......................] - ETA: 44s - loss: 7.5684 - accuracy: 0.5064
 6432/25000 [======>.......................] - ETA: 44s - loss: 7.5593 - accuracy: 0.5070
 6464/25000 [======>.......................] - ETA: 44s - loss: 7.5670 - accuracy: 0.5065
 6496/25000 [======>.......................] - ETA: 44s - loss: 7.5698 - accuracy: 0.5063
 6528/25000 [======>.......................] - ETA: 44s - loss: 7.5680 - accuracy: 0.5064
 6560/25000 [======>.......................] - ETA: 44s - loss: 7.5708 - accuracy: 0.5063
 6592/25000 [======>.......................] - ETA: 43s - loss: 7.5689 - accuracy: 0.5064
 6624/25000 [======>.......................] - ETA: 43s - loss: 7.5833 - accuracy: 0.5054
 6656/25000 [======>.......................] - ETA: 43s - loss: 7.5860 - accuracy: 0.5053
 6688/25000 [=======>......................] - ETA: 43s - loss: 7.5818 - accuracy: 0.5055
 6720/25000 [=======>......................] - ETA: 43s - loss: 7.5845 - accuracy: 0.5054
 6752/25000 [=======>......................] - ETA: 43s - loss: 7.5985 - accuracy: 0.5044
 6784/25000 [=======>......................] - ETA: 43s - loss: 7.6011 - accuracy: 0.5043
 6816/25000 [=======>......................] - ETA: 43s - loss: 7.6036 - accuracy: 0.5041
 6848/25000 [=======>......................] - ETA: 43s - loss: 7.6106 - accuracy: 0.5037
 6880/25000 [=======>......................] - ETA: 43s - loss: 7.6109 - accuracy: 0.5036
 6912/25000 [=======>......................] - ETA: 43s - loss: 7.6067 - accuracy: 0.5039
 6944/25000 [=======>......................] - ETA: 43s - loss: 7.6070 - accuracy: 0.5039
 6976/25000 [=======>......................] - ETA: 43s - loss: 7.6095 - accuracy: 0.5037
 7008/25000 [=======>......................] - ETA: 43s - loss: 7.6119 - accuracy: 0.5036
 7040/25000 [=======>......................] - ETA: 43s - loss: 7.6100 - accuracy: 0.5037
 7072/25000 [=======>......................] - ETA: 43s - loss: 7.6016 - accuracy: 0.5042
 7104/25000 [=======>......................] - ETA: 42s - loss: 7.6040 - accuracy: 0.5041
 7136/25000 [=======>......................] - ETA: 42s - loss: 7.6151 - accuracy: 0.5034
 7168/25000 [=======>......................] - ETA: 42s - loss: 7.6067 - accuracy: 0.5039
 7200/25000 [=======>......................] - ETA: 42s - loss: 7.6134 - accuracy: 0.5035
 7232/25000 [=======>......................] - ETA: 42s - loss: 7.6179 - accuracy: 0.5032
 7264/25000 [=======>......................] - ETA: 42s - loss: 7.6138 - accuracy: 0.5034
 7296/25000 [=======>......................] - ETA: 42s - loss: 7.6183 - accuracy: 0.5032
 7328/25000 [=======>......................] - ETA: 42s - loss: 7.6164 - accuracy: 0.5033
 7360/25000 [=======>......................] - ETA: 42s - loss: 7.6125 - accuracy: 0.5035
 7392/25000 [=======>......................] - ETA: 42s - loss: 7.6127 - accuracy: 0.5035
 7424/25000 [=======>......................] - ETA: 42s - loss: 7.6088 - accuracy: 0.5038
 7456/25000 [=======>......................] - ETA: 42s - loss: 7.6070 - accuracy: 0.5039
 7488/25000 [=======>......................] - ETA: 42s - loss: 7.6072 - accuracy: 0.5039
 7520/25000 [========>.....................] - ETA: 42s - loss: 7.6054 - accuracy: 0.5040
 7552/25000 [========>.....................] - ETA: 41s - loss: 7.6118 - accuracy: 0.5036
 7584/25000 [========>.....................] - ETA: 41s - loss: 7.6100 - accuracy: 0.5037
 7616/25000 [========>.....................] - ETA: 41s - loss: 7.6062 - accuracy: 0.5039
 7648/25000 [========>.....................] - ETA: 41s - loss: 7.6125 - accuracy: 0.5035
 7680/25000 [========>.....................] - ETA: 41s - loss: 7.6127 - accuracy: 0.5035
 7712/25000 [========>.....................] - ETA: 41s - loss: 7.6109 - accuracy: 0.5036
 7744/25000 [========>.....................] - ETA: 41s - loss: 7.6092 - accuracy: 0.5037
 7776/25000 [========>.....................] - ETA: 41s - loss: 7.5956 - accuracy: 0.5046
 7808/25000 [========>.....................] - ETA: 41s - loss: 7.5959 - accuracy: 0.5046
 7840/25000 [========>.....................] - ETA: 41s - loss: 7.6021 - accuracy: 0.5042
 7872/25000 [========>.....................] - ETA: 41s - loss: 7.6023 - accuracy: 0.5042
 7904/25000 [========>.....................] - ETA: 41s - loss: 7.6026 - accuracy: 0.5042
 7936/25000 [========>.....................] - ETA: 41s - loss: 7.5990 - accuracy: 0.5044
 7968/25000 [========>.....................] - ETA: 40s - loss: 7.5973 - accuracy: 0.5045
 8000/25000 [========>.....................] - ETA: 40s - loss: 7.6053 - accuracy: 0.5040
 8032/25000 [========>.....................] - ETA: 40s - loss: 7.6113 - accuracy: 0.5036
 8064/25000 [========>.....................] - ETA: 40s - loss: 7.6096 - accuracy: 0.5037
 8096/25000 [========>.....................] - ETA: 40s - loss: 7.6155 - accuracy: 0.5033
 8128/25000 [========>.....................] - ETA: 40s - loss: 7.6176 - accuracy: 0.5032
 8160/25000 [========>.....................] - ETA: 40s - loss: 7.6196 - accuracy: 0.5031
 8192/25000 [========>.....................] - ETA: 40s - loss: 7.6236 - accuracy: 0.5028
 8224/25000 [========>.....................] - ETA: 40s - loss: 7.6237 - accuracy: 0.5028
 8256/25000 [========>.....................] - ETA: 40s - loss: 7.6258 - accuracy: 0.5027
 8288/25000 [========>.....................] - ETA: 40s - loss: 7.6259 - accuracy: 0.5027
 8320/25000 [========>.....................] - ETA: 40s - loss: 7.6205 - accuracy: 0.5030
 8352/25000 [=========>....................] - ETA: 40s - loss: 7.6134 - accuracy: 0.5035
 8384/25000 [=========>....................] - ETA: 39s - loss: 7.6099 - accuracy: 0.5037
 8416/25000 [=========>....................] - ETA: 39s - loss: 7.6192 - accuracy: 0.5031
 8448/25000 [=========>....................] - ETA: 39s - loss: 7.6176 - accuracy: 0.5032
 8480/25000 [=========>....................] - ETA: 39s - loss: 7.6142 - accuracy: 0.5034
 8512/25000 [=========>....................] - ETA: 39s - loss: 7.6234 - accuracy: 0.5028
 8544/25000 [=========>....................] - ETA: 39s - loss: 7.6289 - accuracy: 0.5025
 8576/25000 [=========>....................] - ETA: 39s - loss: 7.6326 - accuracy: 0.5022
 8608/25000 [=========>....................] - ETA: 39s - loss: 7.6328 - accuracy: 0.5022
 8640/25000 [=========>....................] - ETA: 39s - loss: 7.6329 - accuracy: 0.5022
 8672/25000 [=========>....................] - ETA: 39s - loss: 7.6330 - accuracy: 0.5022
 8704/25000 [=========>....................] - ETA: 39s - loss: 7.6349 - accuracy: 0.5021
 8736/25000 [=========>....................] - ETA: 39s - loss: 7.6333 - accuracy: 0.5022
 8768/25000 [=========>....................] - ETA: 39s - loss: 7.6351 - accuracy: 0.5021
 8800/25000 [=========>....................] - ETA: 38s - loss: 7.6318 - accuracy: 0.5023
 8832/25000 [=========>....................] - ETA: 38s - loss: 7.6197 - accuracy: 0.5031
 8864/25000 [=========>....................] - ETA: 38s - loss: 7.6216 - accuracy: 0.5029
 8896/25000 [=========>....................] - ETA: 38s - loss: 7.6184 - accuracy: 0.5031
 8928/25000 [=========>....................] - ETA: 38s - loss: 7.6168 - accuracy: 0.5032
 8960/25000 [=========>....................] - ETA: 38s - loss: 7.6136 - accuracy: 0.5035
 8992/25000 [=========>....................] - ETA: 38s - loss: 7.6172 - accuracy: 0.5032
 9024/25000 [=========>....................] - ETA: 38s - loss: 7.6309 - accuracy: 0.5023
 9056/25000 [=========>....................] - ETA: 38s - loss: 7.6378 - accuracy: 0.5019
 9088/25000 [=========>....................] - ETA: 38s - loss: 7.6346 - accuracy: 0.5021
 9120/25000 [=========>....................] - ETA: 38s - loss: 7.6347 - accuracy: 0.5021
 9152/25000 [=========>....................] - ETA: 38s - loss: 7.6314 - accuracy: 0.5023
 9184/25000 [==========>...................] - ETA: 38s - loss: 7.6316 - accuracy: 0.5023
 9216/25000 [==========>...................] - ETA: 37s - loss: 7.6267 - accuracy: 0.5026
 9248/25000 [==========>...................] - ETA: 37s - loss: 7.6318 - accuracy: 0.5023
 9280/25000 [==========>...................] - ETA: 37s - loss: 7.6286 - accuracy: 0.5025
 9312/25000 [==========>...................] - ETA: 37s - loss: 7.6304 - accuracy: 0.5024
 9344/25000 [==========>...................] - ETA: 37s - loss: 7.6322 - accuracy: 0.5022
 9376/25000 [==========>...................] - ETA: 37s - loss: 7.6323 - accuracy: 0.5022
 9408/25000 [==========>...................] - ETA: 37s - loss: 7.6340 - accuracy: 0.5021
 9440/25000 [==========>...................] - ETA: 37s - loss: 7.6390 - accuracy: 0.5018
 9472/25000 [==========>...................] - ETA: 37s - loss: 7.6407 - accuracy: 0.5017
 9504/25000 [==========>...................] - ETA: 37s - loss: 7.6456 - accuracy: 0.5014
 9536/25000 [==========>...................] - ETA: 37s - loss: 7.6409 - accuracy: 0.5017
 9568/25000 [==========>...................] - ETA: 37s - loss: 7.6410 - accuracy: 0.5017
 9600/25000 [==========>...................] - ETA: 37s - loss: 7.6299 - accuracy: 0.5024
 9632/25000 [==========>...................] - ETA: 36s - loss: 7.6364 - accuracy: 0.5020
 9664/25000 [==========>...................] - ETA: 36s - loss: 7.6396 - accuracy: 0.5018
 9696/25000 [==========>...................] - ETA: 36s - loss: 7.6445 - accuracy: 0.5014
 9728/25000 [==========>...................] - ETA: 36s - loss: 7.6493 - accuracy: 0.5011
 9760/25000 [==========>...................] - ETA: 36s - loss: 7.6525 - accuracy: 0.5009
 9792/25000 [==========>...................] - ETA: 36s - loss: 7.6619 - accuracy: 0.5003
 9824/25000 [==========>...................] - ETA: 36s - loss: 7.6651 - accuracy: 0.5001
 9856/25000 [==========>...................] - ETA: 36s - loss: 7.6697 - accuracy: 0.4998
 9888/25000 [==========>...................] - ETA: 36s - loss: 7.6620 - accuracy: 0.5003
 9920/25000 [==========>...................] - ETA: 36s - loss: 7.6635 - accuracy: 0.5002
 9952/25000 [==========>...................] - ETA: 36s - loss: 7.6620 - accuracy: 0.5003
 9984/25000 [==========>...................] - ETA: 36s - loss: 7.6651 - accuracy: 0.5001
10016/25000 [===========>..................] - ETA: 35s - loss: 7.6636 - accuracy: 0.5002
10048/25000 [===========>..................] - ETA: 35s - loss: 7.6559 - accuracy: 0.5007
10080/25000 [===========>..................] - ETA: 35s - loss: 7.6514 - accuracy: 0.5010
10112/25000 [===========>..................] - ETA: 35s - loss: 7.6484 - accuracy: 0.5012
10144/25000 [===========>..................] - ETA: 35s - loss: 7.6560 - accuracy: 0.5007
10176/25000 [===========>..................] - ETA: 35s - loss: 7.6576 - accuracy: 0.5006
10208/25000 [===========>..................] - ETA: 35s - loss: 7.6561 - accuracy: 0.5007
10240/25000 [===========>..................] - ETA: 35s - loss: 7.6576 - accuracy: 0.5006
10272/25000 [===========>..................] - ETA: 35s - loss: 7.6606 - accuracy: 0.5004
10304/25000 [===========>..................] - ETA: 35s - loss: 7.6592 - accuracy: 0.5005
10336/25000 [===========>..................] - ETA: 35s - loss: 7.6651 - accuracy: 0.5001
10368/25000 [===========>..................] - ETA: 35s - loss: 7.6622 - accuracy: 0.5003
10400/25000 [===========>..................] - ETA: 34s - loss: 7.6622 - accuracy: 0.5003
10432/25000 [===========>..................] - ETA: 34s - loss: 7.6607 - accuracy: 0.5004
10464/25000 [===========>..................] - ETA: 34s - loss: 7.6549 - accuracy: 0.5008
10496/25000 [===========>..................] - ETA: 34s - loss: 7.6535 - accuracy: 0.5009
10528/25000 [===========>..................] - ETA: 34s - loss: 7.6521 - accuracy: 0.5009
10560/25000 [===========>..................] - ETA: 34s - loss: 7.6448 - accuracy: 0.5014
10592/25000 [===========>..................] - ETA: 34s - loss: 7.6435 - accuracy: 0.5015
10624/25000 [===========>..................] - ETA: 34s - loss: 7.6507 - accuracy: 0.5010
10656/25000 [===========>..................] - ETA: 34s - loss: 7.6580 - accuracy: 0.5006
10688/25000 [===========>..................] - ETA: 34s - loss: 7.6638 - accuracy: 0.5002
10720/25000 [===========>..................] - ETA: 34s - loss: 7.6666 - accuracy: 0.5000
10752/25000 [===========>..................] - ETA: 34s - loss: 7.6666 - accuracy: 0.5000
10784/25000 [===========>..................] - ETA: 34s - loss: 7.6752 - accuracy: 0.4994
10816/25000 [===========>..................] - ETA: 33s - loss: 7.6808 - accuracy: 0.4991
10848/25000 [============>.................] - ETA: 33s - loss: 7.6793 - accuracy: 0.4992
10880/25000 [============>.................] - ETA: 33s - loss: 7.6793 - accuracy: 0.4992
10912/25000 [============>.................] - ETA: 33s - loss: 7.6765 - accuracy: 0.4994
10944/25000 [============>.................] - ETA: 33s - loss: 7.6834 - accuracy: 0.4989
10976/25000 [============>.................] - ETA: 33s - loss: 7.6806 - accuracy: 0.4991
11008/25000 [============>.................] - ETA: 33s - loss: 7.6819 - accuracy: 0.4990
11040/25000 [============>.................] - ETA: 33s - loss: 7.6763 - accuracy: 0.4994
11072/25000 [============>.................] - ETA: 33s - loss: 7.6749 - accuracy: 0.4995
11104/25000 [============>.................] - ETA: 33s - loss: 7.6818 - accuracy: 0.4990
11136/25000 [============>.................] - ETA: 33s - loss: 7.6818 - accuracy: 0.4990
11168/25000 [============>.................] - ETA: 33s - loss: 7.6872 - accuracy: 0.4987
11200/25000 [============>.................] - ETA: 33s - loss: 7.6872 - accuracy: 0.4987
11232/25000 [============>.................] - ETA: 32s - loss: 7.6885 - accuracy: 0.4986
11264/25000 [============>.................] - ETA: 32s - loss: 7.6843 - accuracy: 0.4988
11296/25000 [============>.................] - ETA: 32s - loss: 7.6843 - accuracy: 0.4988
11328/25000 [============>.................] - ETA: 32s - loss: 7.6788 - accuracy: 0.4992
11360/25000 [============>.................] - ETA: 32s - loss: 7.6828 - accuracy: 0.4989
11392/25000 [============>.................] - ETA: 32s - loss: 7.6814 - accuracy: 0.4990
11424/25000 [============>.................] - ETA: 32s - loss: 7.6827 - accuracy: 0.4989
11456/25000 [============>.................] - ETA: 32s - loss: 7.6880 - accuracy: 0.4986
11488/25000 [============>.................] - ETA: 32s - loss: 7.6840 - accuracy: 0.4989
11520/25000 [============>.................] - ETA: 32s - loss: 7.6866 - accuracy: 0.4987
11552/25000 [============>.................] - ETA: 32s - loss: 7.6812 - accuracy: 0.4990
11584/25000 [============>.................] - ETA: 32s - loss: 7.6812 - accuracy: 0.4991
11616/25000 [============>.................] - ETA: 32s - loss: 7.6759 - accuracy: 0.4994
11648/25000 [============>.................] - ETA: 32s - loss: 7.6785 - accuracy: 0.4992
11680/25000 [=============>................] - ETA: 31s - loss: 7.6758 - accuracy: 0.4994
11712/25000 [=============>................] - ETA: 31s - loss: 7.6758 - accuracy: 0.4994
11744/25000 [=============>................] - ETA: 31s - loss: 7.6771 - accuracy: 0.4993
11776/25000 [=============>................] - ETA: 31s - loss: 7.6757 - accuracy: 0.4994
11808/25000 [=============>................] - ETA: 31s - loss: 7.6731 - accuracy: 0.4996
11840/25000 [=============>................] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
11872/25000 [=============>................] - ETA: 31s - loss: 7.6718 - accuracy: 0.4997
11904/25000 [=============>................] - ETA: 31s - loss: 7.6705 - accuracy: 0.4997
11936/25000 [=============>................] - ETA: 31s - loss: 7.6705 - accuracy: 0.4997
11968/25000 [=============>................] - ETA: 31s - loss: 7.6756 - accuracy: 0.4994
12000/25000 [=============>................] - ETA: 31s - loss: 7.6768 - accuracy: 0.4993
12032/25000 [=============>................] - ETA: 31s - loss: 7.6730 - accuracy: 0.4996
12064/25000 [=============>................] - ETA: 31s - loss: 7.6730 - accuracy: 0.4996
12096/25000 [=============>................] - ETA: 30s - loss: 7.6717 - accuracy: 0.4997
12128/25000 [=============>................] - ETA: 30s - loss: 7.6717 - accuracy: 0.4997
12160/25000 [=============>................] - ETA: 30s - loss: 7.6754 - accuracy: 0.4994
12192/25000 [=============>................] - ETA: 30s - loss: 7.6767 - accuracy: 0.4993
12224/25000 [=============>................] - ETA: 30s - loss: 7.6767 - accuracy: 0.4993
12256/25000 [=============>................] - ETA: 30s - loss: 7.6766 - accuracy: 0.4993
12288/25000 [=============>................] - ETA: 30s - loss: 7.6803 - accuracy: 0.4991
12320/25000 [=============>................] - ETA: 30s - loss: 7.6791 - accuracy: 0.4992
12352/25000 [=============>................] - ETA: 30s - loss: 7.6790 - accuracy: 0.4992
12384/25000 [=============>................] - ETA: 30s - loss: 7.6802 - accuracy: 0.4991
12416/25000 [=============>................] - ETA: 30s - loss: 7.6839 - accuracy: 0.4989
12448/25000 [=============>................] - ETA: 30s - loss: 7.6839 - accuracy: 0.4989
12480/25000 [=============>................] - ETA: 30s - loss: 7.6789 - accuracy: 0.4992
12512/25000 [==============>...............] - ETA: 30s - loss: 7.6776 - accuracy: 0.4993
12544/25000 [==============>...............] - ETA: 29s - loss: 7.6764 - accuracy: 0.4994
12576/25000 [==============>...............] - ETA: 29s - loss: 7.6800 - accuracy: 0.4991
12608/25000 [==============>...............] - ETA: 29s - loss: 7.6763 - accuracy: 0.4994
12640/25000 [==============>...............] - ETA: 29s - loss: 7.6775 - accuracy: 0.4993
12672/25000 [==============>...............] - ETA: 29s - loss: 7.6799 - accuracy: 0.4991
12704/25000 [==============>...............] - ETA: 29s - loss: 7.6883 - accuracy: 0.4986
12736/25000 [==============>...............] - ETA: 29s - loss: 7.6907 - accuracy: 0.4984
12768/25000 [==============>...............] - ETA: 29s - loss: 7.6894 - accuracy: 0.4985
12800/25000 [==============>...............] - ETA: 29s - loss: 7.6798 - accuracy: 0.4991
12832/25000 [==============>...............] - ETA: 29s - loss: 7.6798 - accuracy: 0.4991
12864/25000 [==============>...............] - ETA: 29s - loss: 7.6785 - accuracy: 0.4992
12896/25000 [==============>...............] - ETA: 29s - loss: 7.6714 - accuracy: 0.4997
12928/25000 [==============>...............] - ETA: 29s - loss: 7.6666 - accuracy: 0.5000
12960/25000 [==============>...............] - ETA: 28s - loss: 7.6678 - accuracy: 0.4999
12992/25000 [==============>...............] - ETA: 28s - loss: 7.6678 - accuracy: 0.4999
13024/25000 [==============>...............] - ETA: 28s - loss: 7.6702 - accuracy: 0.4998
13056/25000 [==============>...............] - ETA: 28s - loss: 7.6666 - accuracy: 0.5000
13088/25000 [==============>...............] - ETA: 28s - loss: 7.6666 - accuracy: 0.5000
13120/25000 [==============>...............] - ETA: 28s - loss: 7.6655 - accuracy: 0.5001
13152/25000 [==============>...............] - ETA: 28s - loss: 7.6608 - accuracy: 0.5004
13184/25000 [==============>...............] - ETA: 28s - loss: 7.6596 - accuracy: 0.5005
13216/25000 [==============>...............] - ETA: 28s - loss: 7.6597 - accuracy: 0.5005
13248/25000 [==============>...............] - ETA: 28s - loss: 7.6539 - accuracy: 0.5008
13280/25000 [==============>...............] - ETA: 28s - loss: 7.6562 - accuracy: 0.5007
13312/25000 [==============>...............] - ETA: 28s - loss: 7.6539 - accuracy: 0.5008
13344/25000 [===============>..............] - ETA: 27s - loss: 7.6597 - accuracy: 0.5004
13376/25000 [===============>..............] - ETA: 27s - loss: 7.6597 - accuracy: 0.5004
13408/25000 [===============>..............] - ETA: 27s - loss: 7.6586 - accuracy: 0.5005
13440/25000 [===============>..............] - ETA: 27s - loss: 7.6598 - accuracy: 0.5004
13472/25000 [===============>..............] - ETA: 27s - loss: 7.6598 - accuracy: 0.5004
13504/25000 [===============>..............] - ETA: 27s - loss: 7.6609 - accuracy: 0.5004
13536/25000 [===============>..............] - ETA: 27s - loss: 7.6610 - accuracy: 0.5004
13568/25000 [===============>..............] - ETA: 27s - loss: 7.6576 - accuracy: 0.5006
13600/25000 [===============>..............] - ETA: 27s - loss: 7.6576 - accuracy: 0.5006
13632/25000 [===============>..............] - ETA: 27s - loss: 7.6565 - accuracy: 0.5007
13664/25000 [===============>..............] - ETA: 27s - loss: 7.6509 - accuracy: 0.5010
13696/25000 [===============>..............] - ETA: 27s - loss: 7.6476 - accuracy: 0.5012
13728/25000 [===============>..............] - ETA: 27s - loss: 7.6476 - accuracy: 0.5012
13760/25000 [===============>..............] - ETA: 26s - loss: 7.6477 - accuracy: 0.5012
13792/25000 [===============>..............] - ETA: 26s - loss: 7.6499 - accuracy: 0.5011
13824/25000 [===============>..............] - ETA: 26s - loss: 7.6511 - accuracy: 0.5010
13856/25000 [===============>..............] - ETA: 26s - loss: 7.6544 - accuracy: 0.5008
13888/25000 [===============>..............] - ETA: 26s - loss: 7.6545 - accuracy: 0.5008
13920/25000 [===============>..............] - ETA: 26s - loss: 7.6534 - accuracy: 0.5009
13952/25000 [===============>..............] - ETA: 26s - loss: 7.6512 - accuracy: 0.5010
13984/25000 [===============>..............] - ETA: 26s - loss: 7.6502 - accuracy: 0.5011
14016/25000 [===============>..............] - ETA: 26s - loss: 7.6524 - accuracy: 0.5009
14048/25000 [===============>..............] - ETA: 26s - loss: 7.6546 - accuracy: 0.5008
14080/25000 [===============>..............] - ETA: 26s - loss: 7.6590 - accuracy: 0.5005
14112/25000 [===============>..............] - ETA: 26s - loss: 7.6601 - accuracy: 0.5004
14144/25000 [===============>..............] - ETA: 26s - loss: 7.6612 - accuracy: 0.5004
14176/25000 [================>.............] - ETA: 25s - loss: 7.6590 - accuracy: 0.5005
14208/25000 [================>.............] - ETA: 25s - loss: 7.6591 - accuracy: 0.5005
14240/25000 [================>.............] - ETA: 25s - loss: 7.6580 - accuracy: 0.5006
14272/25000 [================>.............] - ETA: 25s - loss: 7.6612 - accuracy: 0.5004
14304/25000 [================>.............] - ETA: 25s - loss: 7.6613 - accuracy: 0.5003
14336/25000 [================>.............] - ETA: 25s - loss: 7.6591 - accuracy: 0.5005
14368/25000 [================>.............] - ETA: 25s - loss: 7.6602 - accuracy: 0.5004
14400/25000 [================>.............] - ETA: 25s - loss: 7.6592 - accuracy: 0.5005
14432/25000 [================>.............] - ETA: 25s - loss: 7.6528 - accuracy: 0.5009
14464/25000 [================>.............] - ETA: 25s - loss: 7.6581 - accuracy: 0.5006
14496/25000 [================>.............] - ETA: 25s - loss: 7.6634 - accuracy: 0.5002
14528/25000 [================>.............] - ETA: 25s - loss: 7.6677 - accuracy: 0.4999
14560/25000 [================>.............] - ETA: 25s - loss: 7.6677 - accuracy: 0.4999
14592/25000 [================>.............] - ETA: 24s - loss: 7.6635 - accuracy: 0.5002
14624/25000 [================>.............] - ETA: 24s - loss: 7.6624 - accuracy: 0.5003
14656/25000 [================>.............] - ETA: 24s - loss: 7.6635 - accuracy: 0.5002
14688/25000 [================>.............] - ETA: 24s - loss: 7.6562 - accuracy: 0.5007
14720/25000 [================>.............] - ETA: 24s - loss: 7.6500 - accuracy: 0.5011
14752/25000 [================>.............] - ETA: 24s - loss: 7.6541 - accuracy: 0.5008
14784/25000 [================>.............] - ETA: 24s - loss: 7.6500 - accuracy: 0.5011
14816/25000 [================>.............] - ETA: 24s - loss: 7.6511 - accuracy: 0.5010
14848/25000 [================>.............] - ETA: 24s - loss: 7.6522 - accuracy: 0.5009
14880/25000 [================>.............] - ETA: 24s - loss: 7.6553 - accuracy: 0.5007
14912/25000 [================>.............] - ETA: 24s - loss: 7.6543 - accuracy: 0.5008
14944/25000 [================>.............] - ETA: 24s - loss: 7.6615 - accuracy: 0.5003
14976/25000 [================>.............] - ETA: 24s - loss: 7.6615 - accuracy: 0.5003
15008/25000 [=================>............] - ETA: 24s - loss: 7.6636 - accuracy: 0.5002
15040/25000 [=================>............] - ETA: 23s - loss: 7.6646 - accuracy: 0.5001
15072/25000 [=================>............] - ETA: 23s - loss: 7.6646 - accuracy: 0.5001
15104/25000 [=================>............] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
15136/25000 [=================>............] - ETA: 23s - loss: 7.6646 - accuracy: 0.5001
15168/25000 [=================>............] - ETA: 23s - loss: 7.6656 - accuracy: 0.5001
15200/25000 [=================>............] - ETA: 23s - loss: 7.6636 - accuracy: 0.5002
15232/25000 [=================>............] - ETA: 23s - loss: 7.6646 - accuracy: 0.5001
15264/25000 [=================>............] - ETA: 23s - loss: 7.6686 - accuracy: 0.4999
15296/25000 [=================>............] - ETA: 23s - loss: 7.6646 - accuracy: 0.5001
15328/25000 [=================>............] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
15360/25000 [=================>............] - ETA: 23s - loss: 7.6636 - accuracy: 0.5002
15392/25000 [=================>............] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
15424/25000 [=================>............] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
15456/25000 [=================>............] - ETA: 22s - loss: 7.6696 - accuracy: 0.4998
15488/25000 [=================>............] - ETA: 22s - loss: 7.6696 - accuracy: 0.4998
15520/25000 [=================>............] - ETA: 22s - loss: 7.6735 - accuracy: 0.4995
15552/25000 [=================>............] - ETA: 22s - loss: 7.6706 - accuracy: 0.4997
15584/25000 [=================>............] - ETA: 22s - loss: 7.6715 - accuracy: 0.4997
15616/25000 [=================>............] - ETA: 22s - loss: 7.6676 - accuracy: 0.4999
15648/25000 [=================>............] - ETA: 22s - loss: 7.6656 - accuracy: 0.5001
15680/25000 [=================>............] - ETA: 22s - loss: 7.6696 - accuracy: 0.4998
15712/25000 [=================>............] - ETA: 22s - loss: 7.6705 - accuracy: 0.4997
15744/25000 [=================>............] - ETA: 22s - loss: 7.6656 - accuracy: 0.5001
15776/25000 [=================>............] - ETA: 22s - loss: 7.6647 - accuracy: 0.5001
15808/25000 [=================>............] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
15840/25000 [==================>...........] - ETA: 22s - loss: 7.6657 - accuracy: 0.5001
15872/25000 [==================>...........] - ETA: 21s - loss: 7.6657 - accuracy: 0.5001
15904/25000 [==================>...........] - ETA: 21s - loss: 7.6666 - accuracy: 0.5000
15936/25000 [==================>...........] - ETA: 21s - loss: 7.6685 - accuracy: 0.4999
15968/25000 [==================>...........] - ETA: 21s - loss: 7.6657 - accuracy: 0.5001
16000/25000 [==================>...........] - ETA: 21s - loss: 7.6666 - accuracy: 0.5000
16032/25000 [==================>...........] - ETA: 21s - loss: 7.6685 - accuracy: 0.4999
16064/25000 [==================>...........] - ETA: 21s - loss: 7.6647 - accuracy: 0.5001
16096/25000 [==================>...........] - ETA: 21s - loss: 7.6676 - accuracy: 0.4999
16128/25000 [==================>...........] - ETA: 21s - loss: 7.6676 - accuracy: 0.4999
16160/25000 [==================>...........] - ETA: 21s - loss: 7.6695 - accuracy: 0.4998
16192/25000 [==================>...........] - ETA: 21s - loss: 7.6695 - accuracy: 0.4998
16224/25000 [==================>...........] - ETA: 21s - loss: 7.6723 - accuracy: 0.4996
16256/25000 [==================>...........] - ETA: 21s - loss: 7.6751 - accuracy: 0.4994
16288/25000 [==================>...........] - ETA: 20s - loss: 7.6742 - accuracy: 0.4995
16320/25000 [==================>...........] - ETA: 20s - loss: 7.6770 - accuracy: 0.4993
16352/25000 [==================>...........] - ETA: 20s - loss: 7.6816 - accuracy: 0.4990
16384/25000 [==================>...........] - ETA: 20s - loss: 7.6807 - accuracy: 0.4991
16416/25000 [==================>...........] - ETA: 20s - loss: 7.6844 - accuracy: 0.4988
16448/25000 [==================>...........] - ETA: 20s - loss: 7.6778 - accuracy: 0.4993
16480/25000 [==================>...........] - ETA: 20s - loss: 7.6759 - accuracy: 0.4994
16512/25000 [==================>...........] - ETA: 20s - loss: 7.6768 - accuracy: 0.4993
16544/25000 [==================>...........] - ETA: 20s - loss: 7.6731 - accuracy: 0.4996
16576/25000 [==================>...........] - ETA: 20s - loss: 7.6731 - accuracy: 0.4996
16608/25000 [==================>...........] - ETA: 20s - loss: 7.6759 - accuracy: 0.4994
16640/25000 [==================>...........] - ETA: 20s - loss: 7.6749 - accuracy: 0.4995
16672/25000 [===================>..........] - ETA: 20s - loss: 7.6703 - accuracy: 0.4998
16704/25000 [===================>..........] - ETA: 19s - loss: 7.6712 - accuracy: 0.4997
16736/25000 [===================>..........] - ETA: 19s - loss: 7.6685 - accuracy: 0.4999
16768/25000 [===================>..........] - ETA: 19s - loss: 7.6648 - accuracy: 0.5001
16800/25000 [===================>..........] - ETA: 19s - loss: 7.6630 - accuracy: 0.5002
16832/25000 [===================>..........] - ETA: 19s - loss: 7.6612 - accuracy: 0.5004
16864/25000 [===================>..........] - ETA: 19s - loss: 7.6612 - accuracy: 0.5004
16896/25000 [===================>..........] - ETA: 19s - loss: 7.6639 - accuracy: 0.5002
16928/25000 [===================>..........] - ETA: 19s - loss: 7.6657 - accuracy: 0.5001
16960/25000 [===================>..........] - ETA: 19s - loss: 7.6675 - accuracy: 0.4999
16992/25000 [===================>..........] - ETA: 19s - loss: 7.6675 - accuracy: 0.4999
17024/25000 [===================>..........] - ETA: 19s - loss: 7.6702 - accuracy: 0.4998
17056/25000 [===================>..........] - ETA: 19s - loss: 7.6729 - accuracy: 0.4996
17088/25000 [===================>..........] - ETA: 19s - loss: 7.6702 - accuracy: 0.4998
17120/25000 [===================>..........] - ETA: 18s - loss: 7.6720 - accuracy: 0.4996
17152/25000 [===================>..........] - ETA: 18s - loss: 7.6684 - accuracy: 0.4999
17184/25000 [===================>..........] - ETA: 18s - loss: 7.6631 - accuracy: 0.5002
17216/25000 [===================>..........] - ETA: 18s - loss: 7.6666 - accuracy: 0.5000
17248/25000 [===================>..........] - ETA: 18s - loss: 7.6684 - accuracy: 0.4999
17280/25000 [===================>..........] - ETA: 18s - loss: 7.6657 - accuracy: 0.5001
17312/25000 [===================>..........] - ETA: 18s - loss: 7.6640 - accuracy: 0.5002
17344/25000 [===================>..........] - ETA: 18s - loss: 7.6604 - accuracy: 0.5004
17376/25000 [===================>..........] - ETA: 18s - loss: 7.6569 - accuracy: 0.5006
17408/25000 [===================>..........] - ETA: 18s - loss: 7.6552 - accuracy: 0.5007
17440/25000 [===================>..........] - ETA: 18s - loss: 7.6543 - accuracy: 0.5008
17472/25000 [===================>..........] - ETA: 18s - loss: 7.6543 - accuracy: 0.5008
17504/25000 [====================>.........] - ETA: 18s - loss: 7.6570 - accuracy: 0.5006
17536/25000 [====================>.........] - ETA: 17s - loss: 7.6518 - accuracy: 0.5010
17568/25000 [====================>.........] - ETA: 17s - loss: 7.6474 - accuracy: 0.5013
17600/25000 [====================>.........] - ETA: 17s - loss: 7.6440 - accuracy: 0.5015
17632/25000 [====================>.........] - ETA: 17s - loss: 7.6440 - accuracy: 0.5015
17664/25000 [====================>.........] - ETA: 17s - loss: 7.6493 - accuracy: 0.5011
17696/25000 [====================>.........] - ETA: 17s - loss: 7.6510 - accuracy: 0.5010
17728/25000 [====================>.........] - ETA: 17s - loss: 7.6502 - accuracy: 0.5011
17760/25000 [====================>.........] - ETA: 17s - loss: 7.6476 - accuracy: 0.5012
17792/25000 [====================>.........] - ETA: 17s - loss: 7.6502 - accuracy: 0.5011
17824/25000 [====================>.........] - ETA: 17s - loss: 7.6537 - accuracy: 0.5008
17856/25000 [====================>.........] - ETA: 17s - loss: 7.6529 - accuracy: 0.5009
17888/25000 [====================>.........] - ETA: 17s - loss: 7.6529 - accuracy: 0.5009
17920/25000 [====================>.........] - ETA: 17s - loss: 7.6529 - accuracy: 0.5009
17952/25000 [====================>.........] - ETA: 16s - loss: 7.6521 - accuracy: 0.5009
17984/25000 [====================>.........] - ETA: 16s - loss: 7.6530 - accuracy: 0.5009
18016/25000 [====================>.........] - ETA: 16s - loss: 7.6504 - accuracy: 0.5011
18048/25000 [====================>.........] - ETA: 16s - loss: 7.6505 - accuracy: 0.5011
18080/25000 [====================>.........] - ETA: 16s - loss: 7.6488 - accuracy: 0.5012
18112/25000 [====================>.........] - ETA: 16s - loss: 7.6429 - accuracy: 0.5015
18144/25000 [====================>.........] - ETA: 16s - loss: 7.6446 - accuracy: 0.5014
18176/25000 [====================>.........] - ETA: 16s - loss: 7.6413 - accuracy: 0.5017
18208/25000 [====================>.........] - ETA: 16s - loss: 7.6447 - accuracy: 0.5014
18240/25000 [====================>.........] - ETA: 16s - loss: 7.6473 - accuracy: 0.5013
18272/25000 [====================>.........] - ETA: 16s - loss: 7.6482 - accuracy: 0.5012
18304/25000 [====================>.........] - ETA: 16s - loss: 7.6474 - accuracy: 0.5013
18336/25000 [=====================>........] - ETA: 16s - loss: 7.6457 - accuracy: 0.5014
18368/25000 [=====================>........] - ETA: 15s - loss: 7.6466 - accuracy: 0.5013
18400/25000 [=====================>........] - ETA: 15s - loss: 7.6483 - accuracy: 0.5012
18432/25000 [=====================>........] - ETA: 15s - loss: 7.6491 - accuracy: 0.5011
18464/25000 [=====================>........] - ETA: 15s - loss: 7.6475 - accuracy: 0.5012
18496/25000 [=====================>........] - ETA: 15s - loss: 7.6476 - accuracy: 0.5012
18528/25000 [=====================>........] - ETA: 15s - loss: 7.6451 - accuracy: 0.5014
18560/25000 [=====================>........] - ETA: 15s - loss: 7.6493 - accuracy: 0.5011
18592/25000 [=====================>........] - ETA: 15s - loss: 7.6501 - accuracy: 0.5011
18624/25000 [=====================>........] - ETA: 15s - loss: 7.6493 - accuracy: 0.5011
18656/25000 [=====================>........] - ETA: 15s - loss: 7.6477 - accuracy: 0.5012
18688/25000 [=====================>........] - ETA: 15s - loss: 7.6453 - accuracy: 0.5014
18720/25000 [=====================>........] - ETA: 15s - loss: 7.6461 - accuracy: 0.5013
18752/25000 [=====================>........] - ETA: 15s - loss: 7.6462 - accuracy: 0.5013
18784/25000 [=====================>........] - ETA: 14s - loss: 7.6503 - accuracy: 0.5011
18816/25000 [=====================>........] - ETA: 14s - loss: 7.6487 - accuracy: 0.5012
18848/25000 [=====================>........] - ETA: 14s - loss: 7.6479 - accuracy: 0.5012
18880/25000 [=====================>........] - ETA: 14s - loss: 7.6496 - accuracy: 0.5011
18912/25000 [=====================>........] - ETA: 14s - loss: 7.6496 - accuracy: 0.5011
18944/25000 [=====================>........] - ETA: 14s - loss: 7.6448 - accuracy: 0.5014
18976/25000 [=====================>........] - ETA: 14s - loss: 7.6472 - accuracy: 0.5013
19008/25000 [=====================>........] - ETA: 14s - loss: 7.6440 - accuracy: 0.5015
19040/25000 [=====================>........] - ETA: 14s - loss: 7.6408 - accuracy: 0.5017
19072/25000 [=====================>........] - ETA: 14s - loss: 7.6385 - accuracy: 0.5018
19104/25000 [=====================>........] - ETA: 14s - loss: 7.6393 - accuracy: 0.5018
19136/25000 [=====================>........] - ETA: 14s - loss: 7.6378 - accuracy: 0.5019
19168/25000 [======================>.......] - ETA: 14s - loss: 7.6362 - accuracy: 0.5020
19200/25000 [======================>.......] - ETA: 13s - loss: 7.6363 - accuracy: 0.5020
19232/25000 [======================>.......] - ETA: 13s - loss: 7.6363 - accuracy: 0.5020
19264/25000 [======================>.......] - ETA: 13s - loss: 7.6372 - accuracy: 0.5019
19296/25000 [======================>.......] - ETA: 13s - loss: 7.6380 - accuracy: 0.5019
19328/25000 [======================>.......] - ETA: 13s - loss: 7.6365 - accuracy: 0.5020
19360/25000 [======================>.......] - ETA: 13s - loss: 7.6341 - accuracy: 0.5021
19392/25000 [======================>.......] - ETA: 13s - loss: 7.6382 - accuracy: 0.5019
19424/25000 [======================>.......] - ETA: 13s - loss: 7.6382 - accuracy: 0.5019
19456/25000 [======================>.......] - ETA: 13s - loss: 7.6382 - accuracy: 0.5019
19488/25000 [======================>.......] - ETA: 13s - loss: 7.6407 - accuracy: 0.5017
19520/25000 [======================>.......] - ETA: 13s - loss: 7.6399 - accuracy: 0.5017
19552/25000 [======================>.......] - ETA: 13s - loss: 7.6360 - accuracy: 0.5020
19584/25000 [======================>.......] - ETA: 12s - loss: 7.6377 - accuracy: 0.5019
19616/25000 [======================>.......] - ETA: 12s - loss: 7.6369 - accuracy: 0.5019
19648/25000 [======================>.......] - ETA: 12s - loss: 7.6401 - accuracy: 0.5017
19680/25000 [======================>.......] - ETA: 12s - loss: 7.6409 - accuracy: 0.5017
19712/25000 [======================>.......] - ETA: 12s - loss: 7.6378 - accuracy: 0.5019
19744/25000 [======================>.......] - ETA: 12s - loss: 7.6379 - accuracy: 0.5019
19776/25000 [======================>.......] - ETA: 12s - loss: 7.6356 - accuracy: 0.5020
19808/25000 [======================>.......] - ETA: 12s - loss: 7.6349 - accuracy: 0.5021
19840/25000 [======================>.......] - ETA: 12s - loss: 7.6380 - accuracy: 0.5019
19872/25000 [======================>.......] - ETA: 12s - loss: 7.6381 - accuracy: 0.5019
19904/25000 [======================>.......] - ETA: 12s - loss: 7.6389 - accuracy: 0.5018
19936/25000 [======================>.......] - ETA: 12s - loss: 7.6397 - accuracy: 0.5018
19968/25000 [======================>.......] - ETA: 12s - loss: 7.6397 - accuracy: 0.5018
20000/25000 [=======================>......] - ETA: 11s - loss: 7.6390 - accuracy: 0.5018
20032/25000 [=======================>......] - ETA: 11s - loss: 7.6391 - accuracy: 0.5018
20064/25000 [=======================>......] - ETA: 11s - loss: 7.6353 - accuracy: 0.5020
20096/25000 [=======================>......] - ETA: 11s - loss: 7.6369 - accuracy: 0.5019
20128/25000 [=======================>......] - ETA: 11s - loss: 7.6339 - accuracy: 0.5021
20160/25000 [=======================>......] - ETA: 11s - loss: 7.6332 - accuracy: 0.5022
20192/25000 [=======================>......] - ETA: 11s - loss: 7.6324 - accuracy: 0.5022
20224/25000 [=======================>......] - ETA: 11s - loss: 7.6317 - accuracy: 0.5023
20256/25000 [=======================>......] - ETA: 11s - loss: 7.6318 - accuracy: 0.5023
20288/25000 [=======================>......] - ETA: 11s - loss: 7.6334 - accuracy: 0.5022
20320/25000 [=======================>......] - ETA: 11s - loss: 7.6349 - accuracy: 0.5021
20352/25000 [=======================>......] - ETA: 11s - loss: 7.6335 - accuracy: 0.5022
20384/25000 [=======================>......] - ETA: 11s - loss: 7.6350 - accuracy: 0.5021
20416/25000 [=======================>......] - ETA: 10s - loss: 7.6343 - accuracy: 0.5021
20448/25000 [=======================>......] - ETA: 10s - loss: 7.6321 - accuracy: 0.5022
20480/25000 [=======================>......] - ETA: 10s - loss: 7.6277 - accuracy: 0.5025
20512/25000 [=======================>......] - ETA: 10s - loss: 7.6255 - accuracy: 0.5027
20544/25000 [=======================>......] - ETA: 10s - loss: 7.6248 - accuracy: 0.5027
20576/25000 [=======================>......] - ETA: 10s - loss: 7.6227 - accuracy: 0.5029
20608/25000 [=======================>......] - ETA: 10s - loss: 7.6205 - accuracy: 0.5030
20640/25000 [=======================>......] - ETA: 10s - loss: 7.6213 - accuracy: 0.5030
20672/25000 [=======================>......] - ETA: 10s - loss: 7.6184 - accuracy: 0.5031
20704/25000 [=======================>......] - ETA: 10s - loss: 7.6170 - accuracy: 0.5032
20736/25000 [=======================>......] - ETA: 10s - loss: 7.6208 - accuracy: 0.5030
20768/25000 [=======================>......] - ETA: 10s - loss: 7.6186 - accuracy: 0.5031
20800/25000 [=======================>......] - ETA: 10s - loss: 7.6217 - accuracy: 0.5029
20832/25000 [=======================>......] - ETA: 9s - loss: 7.6232 - accuracy: 0.5028 
20864/25000 [========================>.....] - ETA: 9s - loss: 7.6255 - accuracy: 0.5027
20896/25000 [========================>.....] - ETA: 9s - loss: 7.6270 - accuracy: 0.5026
20928/25000 [========================>.....] - ETA: 9s - loss: 7.6285 - accuracy: 0.5025
20960/25000 [========================>.....] - ETA: 9s - loss: 7.6322 - accuracy: 0.5022
20992/25000 [========================>.....] - ETA: 9s - loss: 7.6352 - accuracy: 0.5020
21024/25000 [========================>.....] - ETA: 9s - loss: 7.6374 - accuracy: 0.5019
21056/25000 [========================>.....] - ETA: 9s - loss: 7.6397 - accuracy: 0.5018
21088/25000 [========================>.....] - ETA: 9s - loss: 7.6404 - accuracy: 0.5017
21120/25000 [========================>.....] - ETA: 9s - loss: 7.6419 - accuracy: 0.5016
21152/25000 [========================>.....] - ETA: 9s - loss: 7.6420 - accuracy: 0.5016
21184/25000 [========================>.....] - ETA: 9s - loss: 7.6442 - accuracy: 0.5015
21216/25000 [========================>.....] - ETA: 9s - loss: 7.6449 - accuracy: 0.5014
21248/25000 [========================>.....] - ETA: 8s - loss: 7.6464 - accuracy: 0.5013
21280/25000 [========================>.....] - ETA: 8s - loss: 7.6428 - accuracy: 0.5016
21312/25000 [========================>.....] - ETA: 8s - loss: 7.6422 - accuracy: 0.5016
21344/25000 [========================>.....] - ETA: 8s - loss: 7.6436 - accuracy: 0.5015
21376/25000 [========================>.....] - ETA: 8s - loss: 7.6480 - accuracy: 0.5012
21408/25000 [========================>.....] - ETA: 8s - loss: 7.6523 - accuracy: 0.5009
21440/25000 [========================>.....] - ETA: 8s - loss: 7.6537 - accuracy: 0.5008
21472/25000 [========================>.....] - ETA: 8s - loss: 7.6531 - accuracy: 0.5009
21504/25000 [========================>.....] - ETA: 8s - loss: 7.6531 - accuracy: 0.5009
21536/25000 [========================>.....] - ETA: 8s - loss: 7.6545 - accuracy: 0.5008
21568/25000 [========================>.....] - ETA: 8s - loss: 7.6510 - accuracy: 0.5010
21600/25000 [========================>.....] - ETA: 8s - loss: 7.6503 - accuracy: 0.5011
21632/25000 [========================>.....] - ETA: 8s - loss: 7.6510 - accuracy: 0.5010
21664/25000 [========================>.....] - ETA: 7s - loss: 7.6518 - accuracy: 0.5010
21696/25000 [=========================>....] - ETA: 7s - loss: 7.6497 - accuracy: 0.5011
21728/25000 [=========================>....] - ETA: 7s - loss: 7.6504 - accuracy: 0.5011
21760/25000 [=========================>....] - ETA: 7s - loss: 7.6525 - accuracy: 0.5009
21792/25000 [=========================>....] - ETA: 7s - loss: 7.6511 - accuracy: 0.5010
21824/25000 [=========================>....] - ETA: 7s - loss: 7.6519 - accuracy: 0.5010
21856/25000 [=========================>....] - ETA: 7s - loss: 7.6533 - accuracy: 0.5009
21888/25000 [=========================>....] - ETA: 7s - loss: 7.6505 - accuracy: 0.5011
21920/25000 [=========================>....] - ETA: 7s - loss: 7.6519 - accuracy: 0.5010
21952/25000 [=========================>....] - ETA: 7s - loss: 7.6526 - accuracy: 0.5009
21984/25000 [=========================>....] - ETA: 7s - loss: 7.6541 - accuracy: 0.5008
22016/25000 [=========================>....] - ETA: 7s - loss: 7.6548 - accuracy: 0.5008
22048/25000 [=========================>....] - ETA: 7s - loss: 7.6569 - accuracy: 0.5006
22080/25000 [=========================>....] - ETA: 6s - loss: 7.6576 - accuracy: 0.5006
22112/25000 [=========================>....] - ETA: 6s - loss: 7.6583 - accuracy: 0.5005
22144/25000 [=========================>....] - ETA: 6s - loss: 7.6576 - accuracy: 0.5006
22176/25000 [=========================>....] - ETA: 6s - loss: 7.6569 - accuracy: 0.5006
22208/25000 [=========================>....] - ETA: 6s - loss: 7.6570 - accuracy: 0.5006
22240/25000 [=========================>....] - ETA: 6s - loss: 7.6556 - accuracy: 0.5007
22272/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22304/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22336/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22368/25000 [=========================>....] - ETA: 6s - loss: 7.6591 - accuracy: 0.5005
22400/25000 [=========================>....] - ETA: 6s - loss: 7.6570 - accuracy: 0.5006
22432/25000 [=========================>....] - ETA: 6s - loss: 7.6577 - accuracy: 0.5006
22464/25000 [=========================>....] - ETA: 6s - loss: 7.6584 - accuracy: 0.5005
22496/25000 [=========================>....] - ETA: 5s - loss: 7.6564 - accuracy: 0.5007
22528/25000 [==========================>...] - ETA: 5s - loss: 7.6571 - accuracy: 0.5006
22560/25000 [==========================>...] - ETA: 5s - loss: 7.6557 - accuracy: 0.5007
22592/25000 [==========================>...] - ETA: 5s - loss: 7.6564 - accuracy: 0.5007
22624/25000 [==========================>...] - ETA: 5s - loss: 7.6565 - accuracy: 0.5007
22656/25000 [==========================>...] - ETA: 5s - loss: 7.6538 - accuracy: 0.5008
22688/25000 [==========================>...] - ETA: 5s - loss: 7.6531 - accuracy: 0.5009
22720/25000 [==========================>...] - ETA: 5s - loss: 7.6518 - accuracy: 0.5010
22752/25000 [==========================>...] - ETA: 5s - loss: 7.6525 - accuracy: 0.5009
22784/25000 [==========================>...] - ETA: 5s - loss: 7.6532 - accuracy: 0.5009
22816/25000 [==========================>...] - ETA: 5s - loss: 7.6525 - accuracy: 0.5009
22848/25000 [==========================>...] - ETA: 5s - loss: 7.6525 - accuracy: 0.5009
22880/25000 [==========================>...] - ETA: 5s - loss: 7.6512 - accuracy: 0.5010
22912/25000 [==========================>...] - ETA: 4s - loss: 7.6526 - accuracy: 0.5009
22944/25000 [==========================>...] - ETA: 4s - loss: 7.6506 - accuracy: 0.5010
22976/25000 [==========================>...] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
23008/25000 [==========================>...] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
23040/25000 [==========================>...] - ETA: 4s - loss: 7.6520 - accuracy: 0.5010
23072/25000 [==========================>...] - ETA: 4s - loss: 7.6520 - accuracy: 0.5010
23104/25000 [==========================>...] - ETA: 4s - loss: 7.6533 - accuracy: 0.5009
23136/25000 [==========================>...] - ETA: 4s - loss: 7.6501 - accuracy: 0.5011
23168/25000 [==========================>...] - ETA: 4s - loss: 7.6521 - accuracy: 0.5009
23200/25000 [==========================>...] - ETA: 4s - loss: 7.6508 - accuracy: 0.5010
23232/25000 [==========================>...] - ETA: 4s - loss: 7.6501 - accuracy: 0.5011
23264/25000 [==========================>...] - ETA: 4s - loss: 7.6515 - accuracy: 0.5010
23296/25000 [==========================>...] - ETA: 4s - loss: 7.6541 - accuracy: 0.5008
23328/25000 [==========================>...] - ETA: 3s - loss: 7.6568 - accuracy: 0.5006
23360/25000 [===========================>..] - ETA: 3s - loss: 7.6561 - accuracy: 0.5007
23392/25000 [===========================>..] - ETA: 3s - loss: 7.6548 - accuracy: 0.5008
23424/25000 [===========================>..] - ETA: 3s - loss: 7.6548 - accuracy: 0.5008
23456/25000 [===========================>..] - ETA: 3s - loss: 7.6542 - accuracy: 0.5008
23488/25000 [===========================>..] - ETA: 3s - loss: 7.6536 - accuracy: 0.5009
23520/25000 [===========================>..] - ETA: 3s - loss: 7.6555 - accuracy: 0.5007
23552/25000 [===========================>..] - ETA: 3s - loss: 7.6542 - accuracy: 0.5008
23584/25000 [===========================>..] - ETA: 3s - loss: 7.6536 - accuracy: 0.5008
23616/25000 [===========================>..] - ETA: 3s - loss: 7.6491 - accuracy: 0.5011
23648/25000 [===========================>..] - ETA: 3s - loss: 7.6485 - accuracy: 0.5012
23680/25000 [===========================>..] - ETA: 3s - loss: 7.6517 - accuracy: 0.5010
23712/25000 [===========================>..] - ETA: 3s - loss: 7.6517 - accuracy: 0.5010
23744/25000 [===========================>..] - ETA: 2s - loss: 7.6511 - accuracy: 0.5010
23776/25000 [===========================>..] - ETA: 2s - loss: 7.6511 - accuracy: 0.5010
23808/25000 [===========================>..] - ETA: 2s - loss: 7.6525 - accuracy: 0.5009
23840/25000 [===========================>..] - ETA: 2s - loss: 7.6531 - accuracy: 0.5009
23872/25000 [===========================>..] - ETA: 2s - loss: 7.6531 - accuracy: 0.5009
23904/25000 [===========================>..] - ETA: 2s - loss: 7.6512 - accuracy: 0.5010
23936/25000 [===========================>..] - ETA: 2s - loss: 7.6570 - accuracy: 0.5006
23968/25000 [===========================>..] - ETA: 2s - loss: 7.6545 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 2s - loss: 7.6551 - accuracy: 0.5008
24032/25000 [===========================>..] - ETA: 2s - loss: 7.6532 - accuracy: 0.5009
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6558 - accuracy: 0.5007
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6552 - accuracy: 0.5007
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6571 - accuracy: 0.5006
24160/25000 [===========================>..] - ETA: 1s - loss: 7.6565 - accuracy: 0.5007
24192/25000 [============================>.] - ETA: 1s - loss: 7.6546 - accuracy: 0.5008
24224/25000 [============================>.] - ETA: 1s - loss: 7.6514 - accuracy: 0.5010
24256/25000 [============================>.] - ETA: 1s - loss: 7.6527 - accuracy: 0.5009
24288/25000 [============================>.] - ETA: 1s - loss: 7.6527 - accuracy: 0.5009
24320/25000 [============================>.] - ETA: 1s - loss: 7.6546 - accuracy: 0.5008
24352/25000 [============================>.] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
24384/25000 [============================>.] - ETA: 1s - loss: 7.6566 - accuracy: 0.5007
24416/25000 [============================>.] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
24448/25000 [============================>.] - ETA: 1s - loss: 7.6534 - accuracy: 0.5009
24480/25000 [============================>.] - ETA: 1s - loss: 7.6528 - accuracy: 0.5009
24512/25000 [============================>.] - ETA: 1s - loss: 7.6547 - accuracy: 0.5008
24544/25000 [============================>.] - ETA: 1s - loss: 7.6547 - accuracy: 0.5008
24576/25000 [============================>.] - ETA: 1s - loss: 7.6566 - accuracy: 0.5007
24608/25000 [============================>.] - ETA: 0s - loss: 7.6591 - accuracy: 0.5005
24640/25000 [============================>.] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24672/25000 [============================>.] - ETA: 0s - loss: 7.6561 - accuracy: 0.5007
24704/25000 [============================>.] - ETA: 0s - loss: 7.6548 - accuracy: 0.5008
24736/25000 [============================>.] - ETA: 0s - loss: 7.6555 - accuracy: 0.5007
24768/25000 [============================>.] - ETA: 0s - loss: 7.6567 - accuracy: 0.5006
24800/25000 [============================>.] - ETA: 0s - loss: 7.6598 - accuracy: 0.5004
24832/25000 [============================>.] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24864/25000 [============================>.] - ETA: 0s - loss: 7.6611 - accuracy: 0.5004
24896/25000 [============================>.] - ETA: 0s - loss: 7.6586 - accuracy: 0.5005
24928/25000 [============================>.] - ETA: 0s - loss: 7.6611 - accuracy: 0.5004
24960/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 70s 3ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

