
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
	Data preprocessing and feature engineering runtime = 0.27s ...
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
 40%|████      | 2/5 [00:54<01:21, 27.25s/it] 40%|████      | 2/5 [00:54<01:21, 27.25s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.28565975845852587, 'embedding_size_factor': 0.7344907629333692, 'layers.choice': 2, 'learning_rate': 0.002655570548735963, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 7.152027246333441e-07} and reward: 0.3848
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd2H?\xde\x175LX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe7\x80\xf2\xc5\xc0i\xc2X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?e\xc1"\x95\x16\xde\xb1X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xa7\xff\x8bl\x84\xe6!u.' and reward: 0.3848
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd2H?\xde\x175LX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe7\x80\xf2\xc5\xc0i\xc2X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?e\xc1"\x95\x16\xde\xb1X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xa7\xff\x8bl\x84\xe6!u.' and reward: 0.3848
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 110.76558065414429
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.73s of the 7.44s of remaining time.
Ensemble size: 48
Ensemble weights: 
[0.64583333 0.35416667]
	0.3906	 = Validation accuracy score
	0.94s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 113.54s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7ff1b7eaea90> 

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
 [-0.17453863  0.0109226  -0.01486197  0.04650356 -0.04480763  0.08365019]
 [-0.27975523  0.05463845 -0.1397583   0.1005246  -0.1411151   0.32211429]
 [ 0.02441356 -0.02307018  0.02116416 -0.05872749  0.0535289  -0.01566787]
 [-0.10637899 -0.32891509  0.14961642 -0.0269859  -0.25580031  0.07950597]
 [-0.08877295 -0.00676166  0.0097566   0.13489087 -0.2523919   0.24705541]
 [-0.57313979  0.20916685 -0.08509226  0.15214534 -0.18211532 -0.2065258 ]
 [ 0.14378579 -0.27142051  0.07913446  0.07145037  0.06777569 -0.19318962]
 [-0.45088583  0.00585479  0.39027905  0.0974078   0.06884653 -0.24234581]
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
{'loss': 0.4229573905467987, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-14 10:16:50.427991: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.3987714909017086, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-14 10:16:51.623948: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
   24576/17464789 [..............................] - ETA: 43s
   57344/17464789 [..............................] - ETA: 37s
  106496/17464789 [..............................] - ETA: 30s
  196608/17464789 [..............................] - ETA: 21s
  401408/17464789 [..............................] - ETA: 13s
  802816/17464789 [>.............................] - ETA: 7s 
 1605632/17464789 [=>............................] - ETA: 4s
 3170304/17464789 [====>.........................] - ETA: 2s
 6152192/17464789 [=========>....................] - ETA: 1s
 9166848/17464789 [==============>...............] - ETA: 0s
12230656/17464789 [====================>.........] - ETA: 0s
15228928/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 10:17:04.173286: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 10:17:04.177152: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-14 10:17:04.177312: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ebc612e1f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 10:17:04.177329: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:42 - loss: 8.1458 - accuracy: 0.4688
   64/25000 [..............................] - ETA: 2:59 - loss: 8.8645 - accuracy: 0.4219
   96/25000 [..............................] - ETA: 2:24 - loss: 8.3055 - accuracy: 0.4583
  128/25000 [..............................] - ETA: 2:06 - loss: 7.5468 - accuracy: 0.5078
  160/25000 [..............................] - ETA: 1:55 - loss: 7.3791 - accuracy: 0.5188
  192/25000 [..............................] - ETA: 1:49 - loss: 7.6666 - accuracy: 0.5000
  224/25000 [..............................] - ETA: 1:44 - loss: 7.6666 - accuracy: 0.5000
  256/25000 [..............................] - ETA: 1:39 - loss: 7.7265 - accuracy: 0.4961
  288/25000 [..............................] - ETA: 1:36 - loss: 7.6134 - accuracy: 0.5035
  320/25000 [..............................] - ETA: 1:34 - loss: 7.5708 - accuracy: 0.5063
  352/25000 [..............................] - ETA: 1:32 - loss: 7.6666 - accuracy: 0.5000
  384/25000 [..............................] - ETA: 1:30 - loss: 7.7465 - accuracy: 0.4948
  416/25000 [..............................] - ETA: 1:30 - loss: 7.8878 - accuracy: 0.4856
  448/25000 [..............................] - ETA: 1:28 - loss: 7.7693 - accuracy: 0.4933
  480/25000 [..............................] - ETA: 1:27 - loss: 7.6986 - accuracy: 0.4979
  512/25000 [..............................] - ETA: 1:27 - loss: 7.7565 - accuracy: 0.4941
  544/25000 [..............................] - ETA: 1:26 - loss: 7.7512 - accuracy: 0.4945
  576/25000 [..............................] - ETA: 1:25 - loss: 7.7731 - accuracy: 0.4931
  608/25000 [..............................] - ETA: 1:24 - loss: 7.7423 - accuracy: 0.4951
  640/25000 [..............................] - ETA: 1:24 - loss: 7.8104 - accuracy: 0.4906
  672/25000 [..............................] - ETA: 1:23 - loss: 7.8948 - accuracy: 0.4851
  704/25000 [..............................] - ETA: 1:22 - loss: 7.8626 - accuracy: 0.4872
  736/25000 [..............................] - ETA: 1:22 - loss: 7.8750 - accuracy: 0.4864
  768/25000 [..............................] - ETA: 1:21 - loss: 7.8663 - accuracy: 0.4870
  800/25000 [..............................] - ETA: 1:21 - loss: 7.8966 - accuracy: 0.4850
  832/25000 [..............................] - ETA: 1:20 - loss: 7.8509 - accuracy: 0.4880
  864/25000 [>.............................] - ETA: 1:20 - loss: 7.8973 - accuracy: 0.4850
  896/25000 [>.............................] - ETA: 1:20 - loss: 7.8549 - accuracy: 0.4877
  928/25000 [>.............................] - ETA: 1:19 - loss: 7.8649 - accuracy: 0.4871
  960/25000 [>.............................] - ETA: 1:19 - loss: 7.9541 - accuracy: 0.4812
  992/25000 [>.............................] - ETA: 1:18 - loss: 7.9139 - accuracy: 0.4839
 1024/25000 [>.............................] - ETA: 1:18 - loss: 7.9661 - accuracy: 0.4805
 1056/25000 [>.............................] - ETA: 1:18 - loss: 7.8699 - accuracy: 0.4867
 1088/25000 [>.............................] - ETA: 1:18 - loss: 7.8780 - accuracy: 0.4862
 1120/25000 [>.............................] - ETA: 1:17 - loss: 7.8583 - accuracy: 0.4875
 1152/25000 [>.............................] - ETA: 1:17 - loss: 7.8796 - accuracy: 0.4861
 1184/25000 [>.............................] - ETA: 1:17 - loss: 7.9127 - accuracy: 0.4840
 1216/25000 [>.............................] - ETA: 1:17 - loss: 7.9692 - accuracy: 0.4803
 1248/25000 [>.............................] - ETA: 1:16 - loss: 8.0106 - accuracy: 0.4776
 1280/25000 [>.............................] - ETA: 1:16 - loss: 7.9901 - accuracy: 0.4789
 1312/25000 [>.............................] - ETA: 1:16 - loss: 7.9705 - accuracy: 0.4802
 1344/25000 [>.............................] - ETA: 1:16 - loss: 7.9861 - accuracy: 0.4792
 1376/25000 [>.............................] - ETA: 1:15 - loss: 7.9786 - accuracy: 0.4797
 1408/25000 [>.............................] - ETA: 1:15 - loss: 8.0042 - accuracy: 0.4780
 1440/25000 [>.............................] - ETA: 1:15 - loss: 7.9967 - accuracy: 0.4785
 1472/25000 [>.............................] - ETA: 1:15 - loss: 7.9687 - accuracy: 0.4803
 1504/25000 [>.............................] - ETA: 1:15 - loss: 8.0132 - accuracy: 0.4774
 1536/25000 [>.............................] - ETA: 1:15 - loss: 8.0160 - accuracy: 0.4772
 1568/25000 [>.............................] - ETA: 1:14 - loss: 8.0284 - accuracy: 0.4764
 1600/25000 [>.............................] - ETA: 1:14 - loss: 8.0308 - accuracy: 0.4762
 1632/25000 [>.............................] - ETA: 1:14 - loss: 7.9767 - accuracy: 0.4798
 1664/25000 [>.............................] - ETA: 1:14 - loss: 7.9799 - accuracy: 0.4796
 1696/25000 [=>............................] - ETA: 1:14 - loss: 7.9559 - accuracy: 0.4811
 1728/25000 [=>............................] - ETA: 1:14 - loss: 7.9328 - accuracy: 0.4826
 1760/25000 [=>............................] - ETA: 1:14 - loss: 7.9715 - accuracy: 0.4801
 1792/25000 [=>............................] - ETA: 1:13 - loss: 8.0003 - accuracy: 0.4782
 1824/25000 [=>............................] - ETA: 1:13 - loss: 8.0113 - accuracy: 0.4775
 1856/25000 [=>............................] - ETA: 1:13 - loss: 7.9971 - accuracy: 0.4784
 1888/25000 [=>............................] - ETA: 1:13 - loss: 7.9752 - accuracy: 0.4799
 1920/25000 [=>............................] - ETA: 1:13 - loss: 7.9701 - accuracy: 0.4802
 1952/25000 [=>............................] - ETA: 1:13 - loss: 7.9415 - accuracy: 0.4821
 1984/25000 [=>............................] - ETA: 1:13 - loss: 7.9371 - accuracy: 0.4824
 2016/25000 [=>............................] - ETA: 1:13 - loss: 7.9328 - accuracy: 0.4826
 2048/25000 [=>............................] - ETA: 1:12 - loss: 7.9361 - accuracy: 0.4824
 2080/25000 [=>............................] - ETA: 1:12 - loss: 7.9394 - accuracy: 0.4822
 2112/25000 [=>............................] - ETA: 1:12 - loss: 7.9135 - accuracy: 0.4839
 2144/25000 [=>............................] - ETA: 1:12 - loss: 7.9026 - accuracy: 0.4846
 2176/25000 [=>............................] - ETA: 1:12 - loss: 7.8851 - accuracy: 0.4858
 2208/25000 [=>............................] - ETA: 1:12 - loss: 7.9097 - accuracy: 0.4841
 2240/25000 [=>............................] - ETA: 1:11 - loss: 7.8857 - accuracy: 0.4857
 2272/25000 [=>............................] - ETA: 1:11 - loss: 7.8826 - accuracy: 0.4859
 2304/25000 [=>............................] - ETA: 1:11 - loss: 7.8862 - accuracy: 0.4857
 2336/25000 [=>............................] - ETA: 1:11 - loss: 7.8767 - accuracy: 0.4863
 2368/25000 [=>............................] - ETA: 1:11 - loss: 7.8997 - accuracy: 0.4848
 2400/25000 [=>............................] - ETA: 1:11 - loss: 7.9030 - accuracy: 0.4846
 2432/25000 [=>............................] - ETA: 1:11 - loss: 7.8873 - accuracy: 0.4856
 2464/25000 [=>............................] - ETA: 1:11 - loss: 7.8969 - accuracy: 0.4850
 2496/25000 [=>............................] - ETA: 1:10 - loss: 7.8878 - accuracy: 0.4856
 2528/25000 [==>...........................] - ETA: 1:10 - loss: 7.8728 - accuracy: 0.4866
 2560/25000 [==>...........................] - ETA: 1:10 - loss: 7.8643 - accuracy: 0.4871
 2592/25000 [==>...........................] - ETA: 1:10 - loss: 7.8618 - accuracy: 0.4873
 2624/25000 [==>...........................] - ETA: 1:10 - loss: 7.8419 - accuracy: 0.4886
 2656/25000 [==>...........................] - ETA: 1:10 - loss: 7.8514 - accuracy: 0.4880
 2688/25000 [==>...........................] - ETA: 1:10 - loss: 7.8606 - accuracy: 0.4874
 2720/25000 [==>...........................] - ETA: 1:10 - loss: 7.8752 - accuracy: 0.4864
 2752/25000 [==>...........................] - ETA: 1:09 - loss: 7.8616 - accuracy: 0.4873
 2784/25000 [==>...........................] - ETA: 1:09 - loss: 7.8704 - accuracy: 0.4867
 2816/25000 [==>...........................] - ETA: 1:09 - loss: 7.8735 - accuracy: 0.4865
 2848/25000 [==>...........................] - ETA: 1:09 - loss: 7.9035 - accuracy: 0.4846
 2880/25000 [==>...........................] - ETA: 1:09 - loss: 7.8849 - accuracy: 0.4858
 2912/25000 [==>...........................] - ETA: 1:09 - loss: 7.8825 - accuracy: 0.4859
 2944/25000 [==>...........................] - ETA: 1:09 - loss: 7.8906 - accuracy: 0.4854
 2976/25000 [==>...........................] - ETA: 1:09 - loss: 7.8830 - accuracy: 0.4859
 3008/25000 [==>...........................] - ETA: 1:08 - loss: 7.8756 - accuracy: 0.4864
 3040/25000 [==>...........................] - ETA: 1:08 - loss: 7.8885 - accuracy: 0.4855
 3072/25000 [==>...........................] - ETA: 1:08 - loss: 7.8812 - accuracy: 0.4860
 3104/25000 [==>...........................] - ETA: 1:08 - loss: 7.9087 - accuracy: 0.4842
 3136/25000 [==>...........................] - ETA: 1:08 - loss: 7.9160 - accuracy: 0.4837
 3168/25000 [==>...........................] - ETA: 1:08 - loss: 7.9328 - accuracy: 0.4826
 3200/25000 [==>...........................] - ETA: 1:08 - loss: 7.9350 - accuracy: 0.4825
 3232/25000 [==>...........................] - ETA: 1:08 - loss: 7.9181 - accuracy: 0.4836
 3264/25000 [==>...........................] - ETA: 1:08 - loss: 7.9109 - accuracy: 0.4841
 3296/25000 [==>...........................] - ETA: 1:07 - loss: 7.9085 - accuracy: 0.4842
 3328/25000 [==>...........................] - ETA: 1:07 - loss: 7.9016 - accuracy: 0.4847
 3360/25000 [===>..........................] - ETA: 1:07 - loss: 7.9039 - accuracy: 0.4845
 3392/25000 [===>..........................] - ETA: 1:07 - loss: 7.9152 - accuracy: 0.4838
 3424/25000 [===>..........................] - ETA: 1:07 - loss: 7.9040 - accuracy: 0.4845
 3456/25000 [===>..........................] - ETA: 1:07 - loss: 7.8973 - accuracy: 0.4850
 3488/25000 [===>..........................] - ETA: 1:07 - loss: 7.9172 - accuracy: 0.4837
 3520/25000 [===>..........................] - ETA: 1:07 - loss: 7.9062 - accuracy: 0.4844
 3552/25000 [===>..........................] - ETA: 1:07 - loss: 7.9127 - accuracy: 0.4840
 3584/25000 [===>..........................] - ETA: 1:07 - loss: 7.9019 - accuracy: 0.4847
 3616/25000 [===>..........................] - ETA: 1:06 - loss: 7.8871 - accuracy: 0.4856
 3648/25000 [===>..........................] - ETA: 1:06 - loss: 7.8768 - accuracy: 0.4863
 3680/25000 [===>..........................] - ETA: 1:06 - loss: 7.8791 - accuracy: 0.4861
 3712/25000 [===>..........................] - ETA: 1:06 - loss: 7.8732 - accuracy: 0.4865
 3744/25000 [===>..........................] - ETA: 1:06 - loss: 7.8755 - accuracy: 0.4864
 3776/25000 [===>..........................] - ETA: 1:06 - loss: 7.8697 - accuracy: 0.4868
 3808/25000 [===>..........................] - ETA: 1:06 - loss: 7.8720 - accuracy: 0.4866
 3840/25000 [===>..........................] - ETA: 1:06 - loss: 7.8663 - accuracy: 0.4870
 3872/25000 [===>..........................] - ETA: 1:06 - loss: 7.8448 - accuracy: 0.4884
 3904/25000 [===>..........................] - ETA: 1:06 - loss: 7.8512 - accuracy: 0.4880
 3936/25000 [===>..........................] - ETA: 1:06 - loss: 7.8419 - accuracy: 0.4886
 3968/25000 [===>..........................] - ETA: 1:06 - loss: 7.8521 - accuracy: 0.4879
 4000/25000 [===>..........................] - ETA: 1:06 - loss: 7.8430 - accuracy: 0.4885
 4032/25000 [===>..........................] - ETA: 1:05 - loss: 7.8377 - accuracy: 0.4888
 4064/25000 [===>..........................] - ETA: 1:05 - loss: 7.8326 - accuracy: 0.4892
 4096/25000 [===>..........................] - ETA: 1:05 - loss: 7.8613 - accuracy: 0.4873
 4128/25000 [===>..........................] - ETA: 1:05 - loss: 7.8672 - accuracy: 0.4869
 4160/25000 [===>..........................] - ETA: 1:05 - loss: 7.8620 - accuracy: 0.4873
 4192/25000 [====>.........................] - ETA: 1:05 - loss: 7.8605 - accuracy: 0.4874
 4224/25000 [====>.........................] - ETA: 1:05 - loss: 7.8554 - accuracy: 0.4877
 4256/25000 [====>.........................] - ETA: 1:05 - loss: 7.8468 - accuracy: 0.4883
 4288/25000 [====>.........................] - ETA: 1:05 - loss: 7.8383 - accuracy: 0.4888
 4320/25000 [====>.........................] - ETA: 1:05 - loss: 7.8405 - accuracy: 0.4887
 4352/25000 [====>.........................] - ETA: 1:04 - loss: 7.8393 - accuracy: 0.4887
 4384/25000 [====>.........................] - ETA: 1:04 - loss: 7.8415 - accuracy: 0.4886
 4416/25000 [====>.........................] - ETA: 1:04 - loss: 7.8333 - accuracy: 0.4891
 4448/25000 [====>.........................] - ETA: 1:04 - loss: 7.8217 - accuracy: 0.4899
 4480/25000 [====>.........................] - ETA: 1:04 - loss: 7.8138 - accuracy: 0.4904
 4512/25000 [====>.........................] - ETA: 1:04 - loss: 7.8026 - accuracy: 0.4911
 4544/25000 [====>.........................] - ETA: 1:04 - loss: 7.8151 - accuracy: 0.4903
 4576/25000 [====>.........................] - ETA: 1:04 - loss: 7.8174 - accuracy: 0.4902
 4608/25000 [====>.........................] - ETA: 1:04 - loss: 7.8197 - accuracy: 0.4900
 4640/25000 [====>.........................] - ETA: 1:04 - loss: 7.8186 - accuracy: 0.4901
 4672/25000 [====>.........................] - ETA: 1:04 - loss: 7.8242 - accuracy: 0.4897
 4704/25000 [====>.........................] - ETA: 1:03 - loss: 7.8296 - accuracy: 0.4894
 4736/25000 [====>.........................] - ETA: 1:03 - loss: 7.8350 - accuracy: 0.4890
 4768/25000 [====>.........................] - ETA: 1:03 - loss: 7.8306 - accuracy: 0.4893
 4800/25000 [====>.........................] - ETA: 1:03 - loss: 7.8136 - accuracy: 0.4904
 4832/25000 [====>.........................] - ETA: 1:03 - loss: 7.8253 - accuracy: 0.4897
 4864/25000 [====>.........................] - ETA: 1:03 - loss: 7.8022 - accuracy: 0.4912
 4896/25000 [====>.........................] - ETA: 1:03 - loss: 7.8044 - accuracy: 0.4910
 4928/25000 [====>.........................] - ETA: 1:03 - loss: 7.7911 - accuracy: 0.4919
 4960/25000 [====>.........................] - ETA: 1:03 - loss: 7.7872 - accuracy: 0.4921
 4992/25000 [====>.........................] - ETA: 1:03 - loss: 7.7772 - accuracy: 0.4928
 5024/25000 [=====>........................] - ETA: 1:02 - loss: 7.7826 - accuracy: 0.4924
 5056/25000 [=====>........................] - ETA: 1:02 - loss: 7.7819 - accuracy: 0.4925
 5088/25000 [=====>........................] - ETA: 1:02 - loss: 7.7811 - accuracy: 0.4925
 5120/25000 [=====>........................] - ETA: 1:02 - loss: 7.7804 - accuracy: 0.4926
 5152/25000 [=====>........................] - ETA: 1:02 - loss: 7.7767 - accuracy: 0.4928
 5184/25000 [=====>........................] - ETA: 1:02 - loss: 7.7761 - accuracy: 0.4929
 5216/25000 [=====>........................] - ETA: 1:02 - loss: 7.7754 - accuracy: 0.4929
 5248/25000 [=====>........................] - ETA: 1:02 - loss: 7.7806 - accuracy: 0.4926
 5280/25000 [=====>........................] - ETA: 1:02 - loss: 7.7886 - accuracy: 0.4920
 5312/25000 [=====>........................] - ETA: 1:01 - loss: 7.7763 - accuracy: 0.4928
 5344/25000 [=====>........................] - ETA: 1:01 - loss: 7.7699 - accuracy: 0.4933
 5376/25000 [=====>........................] - ETA: 1:01 - loss: 7.7664 - accuracy: 0.4935
 5408/25000 [=====>........................] - ETA: 1:01 - loss: 7.7800 - accuracy: 0.4926
 5440/25000 [=====>........................] - ETA: 1:01 - loss: 7.7765 - accuracy: 0.4928
 5472/25000 [=====>........................] - ETA: 1:01 - loss: 7.7759 - accuracy: 0.4929
 5504/25000 [=====>........................] - ETA: 1:01 - loss: 7.7808 - accuracy: 0.4926
 5536/25000 [=====>........................] - ETA: 1:01 - loss: 7.7691 - accuracy: 0.4933
 5568/25000 [=====>........................] - ETA: 1:01 - loss: 7.7713 - accuracy: 0.4932
 5600/25000 [=====>........................] - ETA: 1:01 - loss: 7.7679 - accuracy: 0.4934
 5632/25000 [=====>........................] - ETA: 1:00 - loss: 7.7592 - accuracy: 0.4940
 5664/25000 [=====>........................] - ETA: 1:00 - loss: 7.7532 - accuracy: 0.4944
 5696/25000 [=====>........................] - ETA: 1:00 - loss: 7.7635 - accuracy: 0.4937
 5728/25000 [=====>........................] - ETA: 1:00 - loss: 7.7683 - accuracy: 0.4934
 5760/25000 [=====>........................] - ETA: 1:00 - loss: 7.7731 - accuracy: 0.4931
 5792/25000 [=====>........................] - ETA: 1:00 - loss: 7.7699 - accuracy: 0.4933
 5824/25000 [=====>........................] - ETA: 1:00 - loss: 7.7693 - accuracy: 0.4933
 5856/25000 [======>.......................] - ETA: 1:00 - loss: 7.7740 - accuracy: 0.4930
 5888/25000 [======>.......................] - ETA: 1:00 - loss: 7.7760 - accuracy: 0.4929
 5920/25000 [======>.......................] - ETA: 59s - loss: 7.7832 - accuracy: 0.4924 
 5952/25000 [======>.......................] - ETA: 59s - loss: 7.7800 - accuracy: 0.4926
 5984/25000 [======>.......................] - ETA: 59s - loss: 7.7768 - accuracy: 0.4928
 6016/25000 [======>.......................] - ETA: 59s - loss: 7.7660 - accuracy: 0.4935
 6048/25000 [======>.......................] - ETA: 59s - loss: 7.7630 - accuracy: 0.4937
 6080/25000 [======>.......................] - ETA: 59s - loss: 7.7574 - accuracy: 0.4941
 6112/25000 [======>.......................] - ETA: 59s - loss: 7.7645 - accuracy: 0.4936
 6144/25000 [======>.......................] - ETA: 59s - loss: 7.7615 - accuracy: 0.4938
 6176/25000 [======>.......................] - ETA: 59s - loss: 7.7659 - accuracy: 0.4935
 6208/25000 [======>.......................] - ETA: 59s - loss: 7.7506 - accuracy: 0.4945
 6240/25000 [======>.......................] - ETA: 58s - loss: 7.7502 - accuracy: 0.4946
 6272/25000 [======>.......................] - ETA: 58s - loss: 7.7449 - accuracy: 0.4949
 6304/25000 [======>.......................] - ETA: 58s - loss: 7.7493 - accuracy: 0.4946
 6336/25000 [======>.......................] - ETA: 58s - loss: 7.7465 - accuracy: 0.4948
 6368/25000 [======>.......................] - ETA: 58s - loss: 7.7437 - accuracy: 0.4950
 6400/25000 [======>.......................] - ETA: 58s - loss: 7.7505 - accuracy: 0.4945
 6432/25000 [======>.......................] - ETA: 58s - loss: 7.7477 - accuracy: 0.4947
 6464/25000 [======>.......................] - ETA: 58s - loss: 7.7425 - accuracy: 0.4950
 6496/25000 [======>.......................] - ETA: 58s - loss: 7.7469 - accuracy: 0.4948
 6528/25000 [======>.......................] - ETA: 58s - loss: 7.7512 - accuracy: 0.4945
 6560/25000 [======>.......................] - ETA: 57s - loss: 7.7484 - accuracy: 0.4947
 6592/25000 [======>.......................] - ETA: 57s - loss: 7.7411 - accuracy: 0.4951
 6624/25000 [======>.......................] - ETA: 57s - loss: 7.7314 - accuracy: 0.4958
 6656/25000 [======>.......................] - ETA: 57s - loss: 7.7357 - accuracy: 0.4955
 6688/25000 [=======>......................] - ETA: 57s - loss: 7.7331 - accuracy: 0.4957
 6720/25000 [=======>......................] - ETA: 57s - loss: 7.7351 - accuracy: 0.4955
 6752/25000 [=======>......................] - ETA: 57s - loss: 7.7234 - accuracy: 0.4963
 6784/25000 [=======>......................] - ETA: 57s - loss: 7.7231 - accuracy: 0.4963
 6816/25000 [=======>......................] - ETA: 57s - loss: 7.7161 - accuracy: 0.4968
 6848/25000 [=======>......................] - ETA: 57s - loss: 7.7047 - accuracy: 0.4975
 6880/25000 [=======>......................] - ETA: 56s - loss: 7.7134 - accuracy: 0.4969
 6912/25000 [=======>......................] - ETA: 56s - loss: 7.7176 - accuracy: 0.4967
 6944/25000 [=======>......................] - ETA: 56s - loss: 7.7152 - accuracy: 0.4968
 6976/25000 [=======>......................] - ETA: 56s - loss: 7.7084 - accuracy: 0.4973
 7008/25000 [=======>......................] - ETA: 56s - loss: 7.7060 - accuracy: 0.4974
 7040/25000 [=======>......................] - ETA: 56s - loss: 7.6993 - accuracy: 0.4979
 7072/25000 [=======>......................] - ETA: 56s - loss: 7.7013 - accuracy: 0.4977
 7104/25000 [=======>......................] - ETA: 56s - loss: 7.6925 - accuracy: 0.4983
 7136/25000 [=======>......................] - ETA: 56s - loss: 7.6967 - accuracy: 0.4980
 7168/25000 [=======>......................] - ETA: 55s - loss: 7.6987 - accuracy: 0.4979
 7200/25000 [=======>......................] - ETA: 55s - loss: 7.6964 - accuracy: 0.4981
 7232/25000 [=======>......................] - ETA: 55s - loss: 7.7048 - accuracy: 0.4975
 7264/25000 [=======>......................] - ETA: 55s - loss: 7.7046 - accuracy: 0.4975
 7296/25000 [=======>......................] - ETA: 55s - loss: 7.7065 - accuracy: 0.4974
 7328/25000 [=======>......................] - ETA: 55s - loss: 7.7127 - accuracy: 0.4970
 7360/25000 [=======>......................] - ETA: 55s - loss: 7.7062 - accuracy: 0.4974
 7392/25000 [=======>......................] - ETA: 55s - loss: 7.7060 - accuracy: 0.4974
 7424/25000 [=======>......................] - ETA: 55s - loss: 7.6976 - accuracy: 0.4980
 7456/25000 [=======>......................] - ETA: 55s - loss: 7.6995 - accuracy: 0.4979
 7488/25000 [=======>......................] - ETA: 54s - loss: 7.7117 - accuracy: 0.4971
 7520/25000 [========>.....................] - ETA: 54s - loss: 7.7217 - accuracy: 0.4964
 7552/25000 [========>.....................] - ETA: 54s - loss: 7.7296 - accuracy: 0.4959
 7584/25000 [========>.....................] - ETA: 54s - loss: 7.7313 - accuracy: 0.4958
 7616/25000 [========>.....................] - ETA: 54s - loss: 7.7290 - accuracy: 0.4959
 7648/25000 [========>.....................] - ETA: 54s - loss: 7.7368 - accuracy: 0.4954
 7680/25000 [========>.....................] - ETA: 54s - loss: 7.7365 - accuracy: 0.4954
 7712/25000 [========>.....................] - ETA: 54s - loss: 7.7382 - accuracy: 0.4953
 7744/25000 [========>.....................] - ETA: 54s - loss: 7.7498 - accuracy: 0.4946
 7776/25000 [========>.....................] - ETA: 54s - loss: 7.7514 - accuracy: 0.4945
 7808/25000 [========>.....................] - ETA: 53s - loss: 7.7452 - accuracy: 0.4949
 7840/25000 [========>.....................] - ETA: 53s - loss: 7.7449 - accuracy: 0.4949
 7872/25000 [========>.....................] - ETA: 53s - loss: 7.7465 - accuracy: 0.4948
 7904/25000 [========>.....................] - ETA: 53s - loss: 7.7423 - accuracy: 0.4951
 7936/25000 [========>.....................] - ETA: 53s - loss: 7.7381 - accuracy: 0.4953
 7968/25000 [========>.....................] - ETA: 53s - loss: 7.7340 - accuracy: 0.4956
 8000/25000 [========>.....................] - ETA: 53s - loss: 7.7452 - accuracy: 0.4949
 8032/25000 [========>.....................] - ETA: 53s - loss: 7.7544 - accuracy: 0.4943
 8064/25000 [========>.....................] - ETA: 53s - loss: 7.7598 - accuracy: 0.4939
 8096/25000 [========>.....................] - ETA: 53s - loss: 7.7556 - accuracy: 0.4942
 8128/25000 [========>.....................] - ETA: 52s - loss: 7.7572 - accuracy: 0.4941
 8160/25000 [========>.....................] - ETA: 52s - loss: 7.7625 - accuracy: 0.4938
 8192/25000 [========>.....................] - ETA: 52s - loss: 7.7565 - accuracy: 0.4941
 8224/25000 [========>.....................] - ETA: 52s - loss: 7.7729 - accuracy: 0.4931
 8256/25000 [========>.....................] - ETA: 52s - loss: 7.7706 - accuracy: 0.4932
 8288/25000 [========>.....................] - ETA: 52s - loss: 7.7739 - accuracy: 0.4930
 8320/25000 [========>.....................] - ETA: 52s - loss: 7.7698 - accuracy: 0.4933
 8352/25000 [=========>....................] - ETA: 52s - loss: 7.7566 - accuracy: 0.4941
 8384/25000 [=========>....................] - ETA: 52s - loss: 7.7507 - accuracy: 0.4945
 8416/25000 [=========>....................] - ETA: 52s - loss: 7.7486 - accuracy: 0.4947
 8448/25000 [=========>....................] - ETA: 51s - loss: 7.7429 - accuracy: 0.4950
 8480/25000 [=========>....................] - ETA: 51s - loss: 7.7408 - accuracy: 0.4952
 8512/25000 [=========>....................] - ETA: 51s - loss: 7.7369 - accuracy: 0.4954
 8544/25000 [=========>....................] - ETA: 51s - loss: 7.7366 - accuracy: 0.4954
 8576/25000 [=========>....................] - ETA: 51s - loss: 7.7346 - accuracy: 0.4956
 8608/25000 [=========>....................] - ETA: 51s - loss: 7.7325 - accuracy: 0.4957
 8640/25000 [=========>....................] - ETA: 51s - loss: 7.7287 - accuracy: 0.4959
 8672/25000 [=========>....................] - ETA: 51s - loss: 7.7250 - accuracy: 0.4962
 8704/25000 [=========>....................] - ETA: 51s - loss: 7.7159 - accuracy: 0.4968
 8736/25000 [=========>....................] - ETA: 51s - loss: 7.7140 - accuracy: 0.4969
 8768/25000 [=========>....................] - ETA: 50s - loss: 7.7121 - accuracy: 0.4970
 8800/25000 [=========>....................] - ETA: 50s - loss: 7.7084 - accuracy: 0.4973
 8832/25000 [=========>....................] - ETA: 50s - loss: 7.7083 - accuracy: 0.4973
 8864/25000 [=========>....................] - ETA: 50s - loss: 7.7116 - accuracy: 0.4971
 8896/25000 [=========>....................] - ETA: 50s - loss: 7.7149 - accuracy: 0.4969
 8928/25000 [=========>....................] - ETA: 50s - loss: 7.7181 - accuracy: 0.4966
 8960/25000 [=========>....................] - ETA: 50s - loss: 7.7094 - accuracy: 0.4972
 8992/25000 [=========>....................] - ETA: 50s - loss: 7.7041 - accuracy: 0.4976
 9024/25000 [=========>....................] - ETA: 50s - loss: 7.7023 - accuracy: 0.4977
 9056/25000 [=========>....................] - ETA: 50s - loss: 7.7089 - accuracy: 0.4972
 9088/25000 [=========>....................] - ETA: 49s - loss: 7.7122 - accuracy: 0.4970
 9120/25000 [=========>....................] - ETA: 49s - loss: 7.7137 - accuracy: 0.4969
 9152/25000 [=========>....................] - ETA: 49s - loss: 7.7135 - accuracy: 0.4969
 9184/25000 [==========>...................] - ETA: 49s - loss: 7.7150 - accuracy: 0.4968
 9216/25000 [==========>...................] - ETA: 49s - loss: 7.7115 - accuracy: 0.4971
 9248/25000 [==========>...................] - ETA: 49s - loss: 7.7180 - accuracy: 0.4966
 9280/25000 [==========>...................] - ETA: 49s - loss: 7.7129 - accuracy: 0.4970
 9312/25000 [==========>...................] - ETA: 49s - loss: 7.7078 - accuracy: 0.4973
 9344/25000 [==========>...................] - ETA: 49s - loss: 7.7142 - accuracy: 0.4969
 9376/25000 [==========>...................] - ETA: 48s - loss: 7.7108 - accuracy: 0.4971
 9408/25000 [==========>...................] - ETA: 48s - loss: 7.7041 - accuracy: 0.4976
 9440/25000 [==========>...................] - ETA: 48s - loss: 7.7072 - accuracy: 0.4974
 9472/25000 [==========>...................] - ETA: 48s - loss: 7.7152 - accuracy: 0.4968
 9504/25000 [==========>...................] - ETA: 48s - loss: 7.7263 - accuracy: 0.4961
 9536/25000 [==========>...................] - ETA: 48s - loss: 7.7181 - accuracy: 0.4966
 9568/25000 [==========>...................] - ETA: 48s - loss: 7.7211 - accuracy: 0.4964
 9600/25000 [==========>...................] - ETA: 48s - loss: 7.7145 - accuracy: 0.4969
 9632/25000 [==========>...................] - ETA: 48s - loss: 7.7096 - accuracy: 0.4972
 9664/25000 [==========>...................] - ETA: 48s - loss: 7.7063 - accuracy: 0.4974
 9696/25000 [==========>...................] - ETA: 47s - loss: 7.7077 - accuracy: 0.4973
 9728/25000 [==========>...................] - ETA: 47s - loss: 7.7044 - accuracy: 0.4975
 9760/25000 [==========>...................] - ETA: 47s - loss: 7.7075 - accuracy: 0.4973
 9792/25000 [==========>...................] - ETA: 47s - loss: 7.7089 - accuracy: 0.4972
 9824/25000 [==========>...................] - ETA: 47s - loss: 7.7088 - accuracy: 0.4973
 9856/25000 [==========>...................] - ETA: 47s - loss: 7.7195 - accuracy: 0.4966
 9888/25000 [==========>...................] - ETA: 47s - loss: 7.7116 - accuracy: 0.4971
 9920/25000 [==========>...................] - ETA: 47s - loss: 7.7084 - accuracy: 0.4973
 9952/25000 [==========>...................] - ETA: 47s - loss: 7.7128 - accuracy: 0.4970
 9984/25000 [==========>...................] - ETA: 47s - loss: 7.7204 - accuracy: 0.4965
10016/25000 [===========>..................] - ETA: 46s - loss: 7.7233 - accuracy: 0.4963
10048/25000 [===========>..................] - ETA: 46s - loss: 7.7109 - accuracy: 0.4971
10080/25000 [===========>..................] - ETA: 46s - loss: 7.7107 - accuracy: 0.4971
10112/25000 [===========>..................] - ETA: 46s - loss: 7.7121 - accuracy: 0.4970
10144/25000 [===========>..................] - ETA: 46s - loss: 7.7135 - accuracy: 0.4969
10176/25000 [===========>..................] - ETA: 46s - loss: 7.7148 - accuracy: 0.4969
10208/25000 [===========>..................] - ETA: 46s - loss: 7.7192 - accuracy: 0.4966
10240/25000 [===========>..................] - ETA: 46s - loss: 7.7190 - accuracy: 0.4966
10272/25000 [===========>..................] - ETA: 46s - loss: 7.7233 - accuracy: 0.4963
10304/25000 [===========>..................] - ETA: 46s - loss: 7.7261 - accuracy: 0.4961
10336/25000 [===========>..................] - ETA: 45s - loss: 7.7245 - accuracy: 0.4962
10368/25000 [===========>..................] - ETA: 45s - loss: 7.7213 - accuracy: 0.4964
10400/25000 [===========>..................] - ETA: 45s - loss: 7.7138 - accuracy: 0.4969
10432/25000 [===========>..................] - ETA: 45s - loss: 7.7181 - accuracy: 0.4966
10464/25000 [===========>..................] - ETA: 45s - loss: 7.7208 - accuracy: 0.4965
10496/25000 [===========>..................] - ETA: 45s - loss: 7.7265 - accuracy: 0.4961
10528/25000 [===========>..................] - ETA: 45s - loss: 7.7220 - accuracy: 0.4964
10560/25000 [===========>..................] - ETA: 45s - loss: 7.7218 - accuracy: 0.4964
10592/25000 [===========>..................] - ETA: 45s - loss: 7.7173 - accuracy: 0.4967
10624/25000 [===========>..................] - ETA: 45s - loss: 7.7157 - accuracy: 0.4968
10656/25000 [===========>..................] - ETA: 44s - loss: 7.7141 - accuracy: 0.4969
10688/25000 [===========>..................] - ETA: 44s - loss: 7.7154 - accuracy: 0.4968
10720/25000 [===========>..................] - ETA: 44s - loss: 7.7081 - accuracy: 0.4973
10752/25000 [===========>..................] - ETA: 44s - loss: 7.7151 - accuracy: 0.4968
10784/25000 [===========>..................] - ETA: 44s - loss: 7.7206 - accuracy: 0.4965
10816/25000 [===========>..................] - ETA: 44s - loss: 7.7233 - accuracy: 0.4963
10848/25000 [============>.................] - ETA: 44s - loss: 7.7203 - accuracy: 0.4965
10880/25000 [============>.................] - ETA: 44s - loss: 7.7202 - accuracy: 0.4965
10912/25000 [============>.................] - ETA: 44s - loss: 7.7242 - accuracy: 0.4962
10944/25000 [============>.................] - ETA: 44s - loss: 7.7269 - accuracy: 0.4961
10976/25000 [============>.................] - ETA: 43s - loss: 7.7169 - accuracy: 0.4967
11008/25000 [============>.................] - ETA: 43s - loss: 7.7140 - accuracy: 0.4969
11040/25000 [============>.................] - ETA: 43s - loss: 7.7166 - accuracy: 0.4967
11072/25000 [============>.................] - ETA: 43s - loss: 7.7179 - accuracy: 0.4967
11104/25000 [============>.................] - ETA: 43s - loss: 7.7177 - accuracy: 0.4967
11136/25000 [============>.................] - ETA: 43s - loss: 7.7189 - accuracy: 0.4966
11168/25000 [============>.................] - ETA: 43s - loss: 7.7160 - accuracy: 0.4968
11200/25000 [============>.................] - ETA: 43s - loss: 7.7091 - accuracy: 0.4972
11232/25000 [============>.................] - ETA: 43s - loss: 7.7048 - accuracy: 0.4975
11264/25000 [============>.................] - ETA: 43s - loss: 7.7075 - accuracy: 0.4973
11296/25000 [============>.................] - ETA: 42s - loss: 7.7114 - accuracy: 0.4971
11328/25000 [============>.................] - ETA: 42s - loss: 7.7126 - accuracy: 0.4970
11360/25000 [============>.................] - ETA: 42s - loss: 7.7112 - accuracy: 0.4971
11392/25000 [============>.................] - ETA: 42s - loss: 7.7164 - accuracy: 0.4968
11424/25000 [============>.................] - ETA: 42s - loss: 7.7123 - accuracy: 0.4970
11456/25000 [============>.................] - ETA: 42s - loss: 7.7014 - accuracy: 0.4977
11488/25000 [============>.................] - ETA: 42s - loss: 7.7000 - accuracy: 0.4978
11520/25000 [============>.................] - ETA: 42s - loss: 7.6986 - accuracy: 0.4979
11552/25000 [============>.................] - ETA: 42s - loss: 7.6998 - accuracy: 0.4978
11584/25000 [============>.................] - ETA: 42s - loss: 7.7010 - accuracy: 0.4978
11616/25000 [============>.................] - ETA: 41s - loss: 7.7009 - accuracy: 0.4978
11648/25000 [============>.................] - ETA: 41s - loss: 7.7061 - accuracy: 0.4974
11680/25000 [=============>................] - ETA: 41s - loss: 7.7021 - accuracy: 0.4977
11712/25000 [=============>................] - ETA: 41s - loss: 7.7007 - accuracy: 0.4978
11744/25000 [=============>................] - ETA: 41s - loss: 7.7019 - accuracy: 0.4977
11776/25000 [=============>................] - ETA: 41s - loss: 7.6992 - accuracy: 0.4979
11808/25000 [=============>................] - ETA: 41s - loss: 7.7030 - accuracy: 0.4976
11840/25000 [=============>................] - ETA: 41s - loss: 7.7055 - accuracy: 0.4975
11872/25000 [=============>................] - ETA: 41s - loss: 7.7105 - accuracy: 0.4971
11904/25000 [=============>................] - ETA: 41s - loss: 7.7104 - accuracy: 0.4971
11936/25000 [=============>................] - ETA: 40s - loss: 7.7090 - accuracy: 0.4972
11968/25000 [=============>................] - ETA: 40s - loss: 7.7051 - accuracy: 0.4975
12000/25000 [=============>................] - ETA: 40s - loss: 7.7024 - accuracy: 0.4977
12032/25000 [=============>................] - ETA: 40s - loss: 7.7010 - accuracy: 0.4978
12064/25000 [=============>................] - ETA: 40s - loss: 7.6946 - accuracy: 0.4982
12096/25000 [=============>................] - ETA: 40s - loss: 7.6945 - accuracy: 0.4982
12128/25000 [=============>................] - ETA: 40s - loss: 7.6932 - accuracy: 0.4983
12160/25000 [=============>................] - ETA: 40s - loss: 7.6931 - accuracy: 0.4983
12192/25000 [=============>................] - ETA: 40s - loss: 7.6943 - accuracy: 0.4982
12224/25000 [=============>................] - ETA: 40s - loss: 7.6942 - accuracy: 0.4982
12256/25000 [=============>................] - ETA: 39s - loss: 7.6941 - accuracy: 0.4982
12288/25000 [=============>................] - ETA: 39s - loss: 7.6916 - accuracy: 0.4984
12320/25000 [=============>................] - ETA: 39s - loss: 7.6928 - accuracy: 0.4983
12352/25000 [=============>................] - ETA: 39s - loss: 7.6877 - accuracy: 0.4986
12384/25000 [=============>................] - ETA: 39s - loss: 7.6939 - accuracy: 0.4982
12416/25000 [=============>................] - ETA: 39s - loss: 7.6987 - accuracy: 0.4979
12448/25000 [=============>................] - ETA: 39s - loss: 7.7023 - accuracy: 0.4977
12480/25000 [=============>................] - ETA: 39s - loss: 7.6986 - accuracy: 0.4979
12512/25000 [==============>...............] - ETA: 39s - loss: 7.6973 - accuracy: 0.4980
12544/25000 [==============>...............] - ETA: 39s - loss: 7.6984 - accuracy: 0.4979
12576/25000 [==============>...............] - ETA: 38s - loss: 7.7008 - accuracy: 0.4978
12608/25000 [==============>...............] - ETA: 38s - loss: 7.7068 - accuracy: 0.4974
12640/25000 [==============>...............] - ETA: 38s - loss: 7.7054 - accuracy: 0.4975
12672/25000 [==============>...............] - ETA: 38s - loss: 7.7065 - accuracy: 0.4974
12704/25000 [==============>...............] - ETA: 38s - loss: 7.7064 - accuracy: 0.4974
12736/25000 [==============>...............] - ETA: 38s - loss: 7.6979 - accuracy: 0.4980
12768/25000 [==============>...............] - ETA: 38s - loss: 7.6990 - accuracy: 0.4979
12800/25000 [==============>...............] - ETA: 38s - loss: 7.7014 - accuracy: 0.4977
12832/25000 [==============>...............] - ETA: 38s - loss: 7.7049 - accuracy: 0.4975
12864/25000 [==============>...............] - ETA: 38s - loss: 7.6940 - accuracy: 0.4982
12896/25000 [==============>...............] - ETA: 37s - loss: 7.6892 - accuracy: 0.4985
12928/25000 [==============>...............] - ETA: 37s - loss: 7.6892 - accuracy: 0.4985
12960/25000 [==============>...............] - ETA: 37s - loss: 7.6926 - accuracy: 0.4983
12992/25000 [==============>...............] - ETA: 37s - loss: 7.6973 - accuracy: 0.4980
13024/25000 [==============>...............] - ETA: 37s - loss: 7.6984 - accuracy: 0.4979
13056/25000 [==============>...............] - ETA: 37s - loss: 7.6995 - accuracy: 0.4979
13088/25000 [==============>...............] - ETA: 37s - loss: 7.6947 - accuracy: 0.4982
13120/25000 [==============>...............] - ETA: 37s - loss: 7.6947 - accuracy: 0.4982
13152/25000 [==============>...............] - ETA: 37s - loss: 7.6969 - accuracy: 0.4980
13184/25000 [==============>...............] - ETA: 37s - loss: 7.6945 - accuracy: 0.4982
13216/25000 [==============>...............] - ETA: 36s - loss: 7.6910 - accuracy: 0.4984
13248/25000 [==============>...............] - ETA: 36s - loss: 7.6932 - accuracy: 0.4983
13280/25000 [==============>...............] - ETA: 36s - loss: 7.6909 - accuracy: 0.4984
13312/25000 [==============>...............] - ETA: 36s - loss: 7.6874 - accuracy: 0.4986
13344/25000 [===============>..............] - ETA: 36s - loss: 7.6896 - accuracy: 0.4985
13376/25000 [===============>..............] - ETA: 36s - loss: 7.6907 - accuracy: 0.4984
13408/25000 [===============>..............] - ETA: 36s - loss: 7.6883 - accuracy: 0.4986
13440/25000 [===============>..............] - ETA: 36s - loss: 7.6872 - accuracy: 0.4987
13472/25000 [===============>..............] - ETA: 36s - loss: 7.6871 - accuracy: 0.4987
13504/25000 [===============>..............] - ETA: 36s - loss: 7.6848 - accuracy: 0.4988
13536/25000 [===============>..............] - ETA: 35s - loss: 7.6768 - accuracy: 0.4993
13568/25000 [===============>..............] - ETA: 35s - loss: 7.6779 - accuracy: 0.4993
13600/25000 [===============>..............] - ETA: 35s - loss: 7.6711 - accuracy: 0.4997
13632/25000 [===============>..............] - ETA: 35s - loss: 7.6689 - accuracy: 0.4999
13664/25000 [===============>..............] - ETA: 35s - loss: 7.6655 - accuracy: 0.5001
13696/25000 [===============>..............] - ETA: 35s - loss: 7.6677 - accuracy: 0.4999
13728/25000 [===============>..............] - ETA: 35s - loss: 7.6689 - accuracy: 0.4999
13760/25000 [===============>..............] - ETA: 35s - loss: 7.6666 - accuracy: 0.5000
13792/25000 [===============>..............] - ETA: 35s - loss: 7.6655 - accuracy: 0.5001
13824/25000 [===============>..............] - ETA: 34s - loss: 7.6677 - accuracy: 0.4999
13856/25000 [===============>..............] - ETA: 34s - loss: 7.6699 - accuracy: 0.4998
13888/25000 [===============>..............] - ETA: 34s - loss: 7.6688 - accuracy: 0.4999
13920/25000 [===============>..............] - ETA: 34s - loss: 7.6699 - accuracy: 0.4998
13952/25000 [===============>..............] - ETA: 34s - loss: 7.6688 - accuracy: 0.4999
13984/25000 [===============>..............] - ETA: 34s - loss: 7.6622 - accuracy: 0.5003
14016/25000 [===============>..............] - ETA: 34s - loss: 7.6633 - accuracy: 0.5002
14048/25000 [===============>..............] - ETA: 34s - loss: 7.6568 - accuracy: 0.5006
14080/25000 [===============>..............] - ETA: 34s - loss: 7.6546 - accuracy: 0.5008
14112/25000 [===============>..............] - ETA: 34s - loss: 7.6525 - accuracy: 0.5009
14144/25000 [===============>..............] - ETA: 33s - loss: 7.6536 - accuracy: 0.5008
14176/25000 [================>.............] - ETA: 33s - loss: 7.6536 - accuracy: 0.5008
14208/25000 [================>.............] - ETA: 33s - loss: 7.6558 - accuracy: 0.5007
14240/25000 [================>.............] - ETA: 33s - loss: 7.6526 - accuracy: 0.5009
14272/25000 [================>.............] - ETA: 33s - loss: 7.6548 - accuracy: 0.5008
14304/25000 [================>.............] - ETA: 33s - loss: 7.6495 - accuracy: 0.5011
14336/25000 [================>.............] - ETA: 33s - loss: 7.6549 - accuracy: 0.5008
14368/25000 [================>.............] - ETA: 33s - loss: 7.6581 - accuracy: 0.5006
14400/25000 [================>.............] - ETA: 33s - loss: 7.6538 - accuracy: 0.5008
14432/25000 [================>.............] - ETA: 33s - loss: 7.6560 - accuracy: 0.5007
14464/25000 [================>.............] - ETA: 32s - loss: 7.6592 - accuracy: 0.5005
14496/25000 [================>.............] - ETA: 32s - loss: 7.6603 - accuracy: 0.5004
14528/25000 [================>.............] - ETA: 32s - loss: 7.6582 - accuracy: 0.5006
14560/25000 [================>.............] - ETA: 32s - loss: 7.6582 - accuracy: 0.5005
14592/25000 [================>.............] - ETA: 32s - loss: 7.6572 - accuracy: 0.5006
14624/25000 [================>.............] - ETA: 32s - loss: 7.6551 - accuracy: 0.5008
14656/25000 [================>.............] - ETA: 32s - loss: 7.6614 - accuracy: 0.5003
14688/25000 [================>.............] - ETA: 32s - loss: 7.6656 - accuracy: 0.5001
14720/25000 [================>.............] - ETA: 32s - loss: 7.6677 - accuracy: 0.4999
14752/25000 [================>.............] - ETA: 32s - loss: 7.6645 - accuracy: 0.5001
14784/25000 [================>.............] - ETA: 31s - loss: 7.6656 - accuracy: 0.5001
14816/25000 [================>.............] - ETA: 31s - loss: 7.6656 - accuracy: 0.5001
14848/25000 [================>.............] - ETA: 31s - loss: 7.6646 - accuracy: 0.5001
14880/25000 [================>.............] - ETA: 31s - loss: 7.6687 - accuracy: 0.4999
14912/25000 [================>.............] - ETA: 31s - loss: 7.6707 - accuracy: 0.4997
14944/25000 [================>.............] - ETA: 31s - loss: 7.6707 - accuracy: 0.4997
14976/25000 [================>.............] - ETA: 31s - loss: 7.6666 - accuracy: 0.5000
15008/25000 [=================>............] - ETA: 31s - loss: 7.6676 - accuracy: 0.4999
15040/25000 [=================>............] - ETA: 31s - loss: 7.6717 - accuracy: 0.4997
15072/25000 [=================>............] - ETA: 31s - loss: 7.6697 - accuracy: 0.4998
15104/25000 [=================>............] - ETA: 30s - loss: 7.6747 - accuracy: 0.4995
15136/25000 [=================>............] - ETA: 30s - loss: 7.6707 - accuracy: 0.4997
15168/25000 [=================>............] - ETA: 30s - loss: 7.6717 - accuracy: 0.4997
15200/25000 [=================>............] - ETA: 30s - loss: 7.6737 - accuracy: 0.4995
15232/25000 [=================>............] - ETA: 30s - loss: 7.6767 - accuracy: 0.4993
15264/25000 [=================>............] - ETA: 30s - loss: 7.6837 - accuracy: 0.4989
15296/25000 [=================>............] - ETA: 30s - loss: 7.6857 - accuracy: 0.4988
15328/25000 [=================>............] - ETA: 30s - loss: 7.6866 - accuracy: 0.4987
15360/25000 [=================>............] - ETA: 30s - loss: 7.6826 - accuracy: 0.4990
15392/25000 [=================>............] - ETA: 30s - loss: 7.6865 - accuracy: 0.4987
15424/25000 [=================>............] - ETA: 29s - loss: 7.6905 - accuracy: 0.4984
15456/25000 [=================>............] - ETA: 29s - loss: 7.6914 - accuracy: 0.4984
15488/25000 [=================>............] - ETA: 29s - loss: 7.6933 - accuracy: 0.4983
15520/25000 [=================>............] - ETA: 29s - loss: 7.6933 - accuracy: 0.4983
15552/25000 [=================>............] - ETA: 29s - loss: 7.6992 - accuracy: 0.4979
15584/25000 [=================>............] - ETA: 29s - loss: 7.7050 - accuracy: 0.4975
15616/25000 [=================>............] - ETA: 29s - loss: 7.7098 - accuracy: 0.4972
15648/25000 [=================>............] - ETA: 29s - loss: 7.7097 - accuracy: 0.4972
15680/25000 [=================>............] - ETA: 29s - loss: 7.7096 - accuracy: 0.4972
15712/25000 [=================>............] - ETA: 29s - loss: 7.7086 - accuracy: 0.4973
15744/25000 [=================>............] - ETA: 28s - loss: 7.7007 - accuracy: 0.4978
15776/25000 [=================>............] - ETA: 28s - loss: 7.7036 - accuracy: 0.4976
15808/25000 [=================>............] - ETA: 28s - loss: 7.7025 - accuracy: 0.4977
15840/25000 [==================>...........] - ETA: 28s - loss: 7.7063 - accuracy: 0.4974
15872/25000 [==================>...........] - ETA: 28s - loss: 7.7062 - accuracy: 0.4974
15904/25000 [==================>...........] - ETA: 28s - loss: 7.7061 - accuracy: 0.4974
15936/25000 [==================>...........] - ETA: 28s - loss: 7.7051 - accuracy: 0.4975
15968/25000 [==================>...........] - ETA: 28s - loss: 7.7069 - accuracy: 0.4974
16000/25000 [==================>...........] - ETA: 28s - loss: 7.7069 - accuracy: 0.4974
16032/25000 [==================>...........] - ETA: 28s - loss: 7.7097 - accuracy: 0.4972
16064/25000 [==================>...........] - ETA: 27s - loss: 7.7077 - accuracy: 0.4973
16096/25000 [==================>...........] - ETA: 27s - loss: 7.7076 - accuracy: 0.4973
16128/25000 [==================>...........] - ETA: 27s - loss: 7.7104 - accuracy: 0.4971
16160/25000 [==================>...........] - ETA: 27s - loss: 7.7084 - accuracy: 0.4973
16192/25000 [==================>...........] - ETA: 27s - loss: 7.7102 - accuracy: 0.4972
16224/25000 [==================>...........] - ETA: 27s - loss: 7.7129 - accuracy: 0.4970
16256/25000 [==================>...........] - ETA: 27s - loss: 7.7166 - accuracy: 0.4967
16288/25000 [==================>...........] - ETA: 27s - loss: 7.7118 - accuracy: 0.4971
16320/25000 [==================>...........] - ETA: 27s - loss: 7.7127 - accuracy: 0.4970
16352/25000 [==================>...........] - ETA: 27s - loss: 7.7163 - accuracy: 0.4968
16384/25000 [==================>...........] - ETA: 26s - loss: 7.7115 - accuracy: 0.4971
16416/25000 [==================>...........] - ETA: 26s - loss: 7.7115 - accuracy: 0.4971
16448/25000 [==================>...........] - ETA: 26s - loss: 7.7076 - accuracy: 0.4973
16480/25000 [==================>...........] - ETA: 26s - loss: 7.7066 - accuracy: 0.4974
16512/25000 [==================>...........] - ETA: 26s - loss: 7.7019 - accuracy: 0.4977
16544/25000 [==================>...........] - ETA: 26s - loss: 7.7028 - accuracy: 0.4976
16576/25000 [==================>...........] - ETA: 26s - loss: 7.6999 - accuracy: 0.4978
16608/25000 [==================>...........] - ETA: 26s - loss: 7.7008 - accuracy: 0.4978
16640/25000 [==================>...........] - ETA: 26s - loss: 7.7007 - accuracy: 0.4978
16672/25000 [===================>..........] - ETA: 26s - loss: 7.7006 - accuracy: 0.4978
16704/25000 [===================>..........] - ETA: 26s - loss: 7.6960 - accuracy: 0.4981
16736/25000 [===================>..........] - ETA: 25s - loss: 7.6959 - accuracy: 0.4981
16768/25000 [===================>..........] - ETA: 25s - loss: 7.6959 - accuracy: 0.4981
16800/25000 [===================>..........] - ETA: 25s - loss: 7.6949 - accuracy: 0.4982
16832/25000 [===================>..........] - ETA: 25s - loss: 7.6949 - accuracy: 0.4982
16864/25000 [===================>..........] - ETA: 25s - loss: 7.6939 - accuracy: 0.4982
16896/25000 [===================>..........] - ETA: 25s - loss: 7.6948 - accuracy: 0.4982
16928/25000 [===================>..........] - ETA: 25s - loss: 7.6920 - accuracy: 0.4983
16960/25000 [===================>..........] - ETA: 25s - loss: 7.6901 - accuracy: 0.4985
16992/25000 [===================>..........] - ETA: 25s - loss: 7.6892 - accuracy: 0.4985
17024/25000 [===================>..........] - ETA: 25s - loss: 7.6873 - accuracy: 0.4986
17056/25000 [===================>..........] - ETA: 24s - loss: 7.6864 - accuracy: 0.4987
17088/25000 [===================>..........] - ETA: 24s - loss: 7.6828 - accuracy: 0.4989
17120/25000 [===================>..........] - ETA: 24s - loss: 7.6809 - accuracy: 0.4991
17152/25000 [===================>..........] - ETA: 24s - loss: 7.6836 - accuracy: 0.4989
17184/25000 [===================>..........] - ETA: 24s - loss: 7.6845 - accuracy: 0.4988
17216/25000 [===================>..........] - ETA: 24s - loss: 7.6800 - accuracy: 0.4991
17248/25000 [===================>..........] - ETA: 24s - loss: 7.6808 - accuracy: 0.4991
17280/25000 [===================>..........] - ETA: 24s - loss: 7.6782 - accuracy: 0.4992
17312/25000 [===================>..........] - ETA: 24s - loss: 7.6817 - accuracy: 0.4990
17344/25000 [===================>..........] - ETA: 24s - loss: 7.6808 - accuracy: 0.4991
17376/25000 [===================>..........] - ETA: 23s - loss: 7.6790 - accuracy: 0.4992
17408/25000 [===================>..........] - ETA: 23s - loss: 7.6807 - accuracy: 0.4991
17440/25000 [===================>..........] - ETA: 23s - loss: 7.6807 - accuracy: 0.4991
17472/25000 [===================>..........] - ETA: 23s - loss: 7.6833 - accuracy: 0.4989
17504/25000 [====================>.........] - ETA: 23s - loss: 7.6824 - accuracy: 0.4990
17536/25000 [====================>.........] - ETA: 23s - loss: 7.6771 - accuracy: 0.4993
17568/25000 [====================>.........] - ETA: 23s - loss: 7.6771 - accuracy: 0.4993
17600/25000 [====================>.........] - ETA: 23s - loss: 7.6753 - accuracy: 0.4994
17632/25000 [====================>.........] - ETA: 23s - loss: 7.6736 - accuracy: 0.4995
17664/25000 [====================>.........] - ETA: 23s - loss: 7.6718 - accuracy: 0.4997
17696/25000 [====================>.........] - ETA: 22s - loss: 7.6684 - accuracy: 0.4999
17728/25000 [====================>.........] - ETA: 22s - loss: 7.6709 - accuracy: 0.4997
17760/25000 [====================>.........] - ETA: 22s - loss: 7.6718 - accuracy: 0.4997
17792/25000 [====================>.........] - ETA: 22s - loss: 7.6727 - accuracy: 0.4996
17824/25000 [====================>.........] - ETA: 22s - loss: 7.6718 - accuracy: 0.4997
17856/25000 [====================>.........] - ETA: 22s - loss: 7.6735 - accuracy: 0.4996
17888/25000 [====================>.........] - ETA: 22s - loss: 7.6700 - accuracy: 0.4998
17920/25000 [====================>.........] - ETA: 22s - loss: 7.6675 - accuracy: 0.4999
17952/25000 [====================>.........] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
17984/25000 [====================>.........] - ETA: 21s - loss: 7.6658 - accuracy: 0.5001
18016/25000 [====================>.........] - ETA: 21s - loss: 7.6692 - accuracy: 0.4998
18048/25000 [====================>.........] - ETA: 21s - loss: 7.6683 - accuracy: 0.4999
18080/25000 [====================>.........] - ETA: 21s - loss: 7.6692 - accuracy: 0.4998
18112/25000 [====================>.........] - ETA: 21s - loss: 7.6675 - accuracy: 0.4999
18144/25000 [====================>.........] - ETA: 21s - loss: 7.6717 - accuracy: 0.4997
18176/25000 [====================>.........] - ETA: 21s - loss: 7.6725 - accuracy: 0.4996
18208/25000 [====================>.........] - ETA: 21s - loss: 7.6725 - accuracy: 0.4996
18240/25000 [====================>.........] - ETA: 21s - loss: 7.6717 - accuracy: 0.4997
18272/25000 [====================>.........] - ETA: 21s - loss: 7.6691 - accuracy: 0.4998
18304/25000 [====================>.........] - ETA: 20s - loss: 7.6716 - accuracy: 0.4997
18336/25000 [=====================>........] - ETA: 20s - loss: 7.6725 - accuracy: 0.4996
18368/25000 [=====================>........] - ETA: 20s - loss: 7.6683 - accuracy: 0.4999
18400/25000 [=====================>........] - ETA: 20s - loss: 7.6666 - accuracy: 0.5000
18432/25000 [=====================>........] - ETA: 20s - loss: 7.6633 - accuracy: 0.5002
18464/25000 [=====================>........] - ETA: 20s - loss: 7.6625 - accuracy: 0.5003
18496/25000 [=====================>........] - ETA: 20s - loss: 7.6625 - accuracy: 0.5003
18528/25000 [=====================>........] - ETA: 20s - loss: 7.6600 - accuracy: 0.5004
18560/25000 [=====================>........] - ETA: 20s - loss: 7.6625 - accuracy: 0.5003
18592/25000 [=====================>........] - ETA: 20s - loss: 7.6650 - accuracy: 0.5001
18624/25000 [=====================>........] - ETA: 19s - loss: 7.6658 - accuracy: 0.5001
18656/25000 [=====================>........] - ETA: 19s - loss: 7.6707 - accuracy: 0.4997
18688/25000 [=====================>........] - ETA: 19s - loss: 7.6724 - accuracy: 0.4996
18720/25000 [=====================>........] - ETA: 19s - loss: 7.6691 - accuracy: 0.4998
18752/25000 [=====================>........] - ETA: 19s - loss: 7.6707 - accuracy: 0.4997
18784/25000 [=====================>........] - ETA: 19s - loss: 7.6731 - accuracy: 0.4996
18816/25000 [=====================>........] - ETA: 19s - loss: 7.6731 - accuracy: 0.4996
18848/25000 [=====================>........] - ETA: 19s - loss: 7.6699 - accuracy: 0.4998
18880/25000 [=====================>........] - ETA: 19s - loss: 7.6707 - accuracy: 0.4997
18912/25000 [=====================>........] - ETA: 19s - loss: 7.6739 - accuracy: 0.4995
18944/25000 [=====================>........] - ETA: 18s - loss: 7.6731 - accuracy: 0.4996
18976/25000 [=====================>........] - ETA: 18s - loss: 7.6699 - accuracy: 0.4998
19008/25000 [=====================>........] - ETA: 18s - loss: 7.6715 - accuracy: 0.4997
19040/25000 [=====================>........] - ETA: 18s - loss: 7.6715 - accuracy: 0.4997
19072/25000 [=====================>........] - ETA: 18s - loss: 7.6714 - accuracy: 0.4997
19104/25000 [=====================>........] - ETA: 18s - loss: 7.6771 - accuracy: 0.4993
19136/25000 [=====================>........] - ETA: 18s - loss: 7.6754 - accuracy: 0.4994
19168/25000 [======================>.......] - ETA: 18s - loss: 7.6802 - accuracy: 0.4991
19200/25000 [======================>.......] - ETA: 18s - loss: 7.6786 - accuracy: 0.4992
19232/25000 [======================>.......] - ETA: 18s - loss: 7.6818 - accuracy: 0.4990
19264/25000 [======================>.......] - ETA: 17s - loss: 7.6809 - accuracy: 0.4991
19296/25000 [======================>.......] - ETA: 17s - loss: 7.6817 - accuracy: 0.4990
19328/25000 [======================>.......] - ETA: 17s - loss: 7.6817 - accuracy: 0.4990
19360/25000 [======================>.......] - ETA: 17s - loss: 7.6856 - accuracy: 0.4988
19392/25000 [======================>.......] - ETA: 17s - loss: 7.6824 - accuracy: 0.4990
19424/25000 [======================>.......] - ETA: 17s - loss: 7.6785 - accuracy: 0.4992
19456/25000 [======================>.......] - ETA: 17s - loss: 7.6800 - accuracy: 0.4991
19488/25000 [======================>.......] - ETA: 17s - loss: 7.6784 - accuracy: 0.4992
19520/25000 [======================>.......] - ETA: 17s - loss: 7.6808 - accuracy: 0.4991
19552/25000 [======================>.......] - ETA: 17s - loss: 7.6776 - accuracy: 0.4993
19584/25000 [======================>.......] - ETA: 16s - loss: 7.6768 - accuracy: 0.4993
19616/25000 [======================>.......] - ETA: 16s - loss: 7.6776 - accuracy: 0.4993
19648/25000 [======================>.......] - ETA: 16s - loss: 7.6807 - accuracy: 0.4991
19680/25000 [======================>.......] - ETA: 16s - loss: 7.6791 - accuracy: 0.4992
19712/25000 [======================>.......] - ETA: 16s - loss: 7.6775 - accuracy: 0.4993
19744/25000 [======================>.......] - ETA: 16s - loss: 7.6783 - accuracy: 0.4992
19776/25000 [======================>.......] - ETA: 16s - loss: 7.6775 - accuracy: 0.4993
19808/25000 [======================>.......] - ETA: 16s - loss: 7.6821 - accuracy: 0.4990
19840/25000 [======================>.......] - ETA: 16s - loss: 7.6836 - accuracy: 0.4989
19872/25000 [======================>.......] - ETA: 16s - loss: 7.6828 - accuracy: 0.4989
19904/25000 [======================>.......] - ETA: 15s - loss: 7.6851 - accuracy: 0.4988
19936/25000 [======================>.......] - ETA: 15s - loss: 7.6828 - accuracy: 0.4989
19968/25000 [======================>.......] - ETA: 15s - loss: 7.6843 - accuracy: 0.4988
20000/25000 [=======================>......] - ETA: 15s - loss: 7.6827 - accuracy: 0.4990
20032/25000 [=======================>......] - ETA: 15s - loss: 7.6858 - accuracy: 0.4988
20064/25000 [=======================>......] - ETA: 15s - loss: 7.6834 - accuracy: 0.4989
20096/25000 [=======================>......] - ETA: 15s - loss: 7.6834 - accuracy: 0.4989
20128/25000 [=======================>......] - ETA: 15s - loss: 7.6788 - accuracy: 0.4992
20160/25000 [=======================>......] - ETA: 15s - loss: 7.6750 - accuracy: 0.4995
20192/25000 [=======================>......] - ETA: 15s - loss: 7.6750 - accuracy: 0.4995
20224/25000 [=======================>......] - ETA: 14s - loss: 7.6734 - accuracy: 0.4996
20256/25000 [=======================>......] - ETA: 14s - loss: 7.6727 - accuracy: 0.4996
20288/25000 [=======================>......] - ETA: 14s - loss: 7.6727 - accuracy: 0.4996
20320/25000 [=======================>......] - ETA: 14s - loss: 7.6711 - accuracy: 0.4997
20352/25000 [=======================>......] - ETA: 14s - loss: 7.6726 - accuracy: 0.4996
20384/25000 [=======================>......] - ETA: 14s - loss: 7.6749 - accuracy: 0.4995
20416/25000 [=======================>......] - ETA: 14s - loss: 7.6726 - accuracy: 0.4996
20448/25000 [=======================>......] - ETA: 14s - loss: 7.6704 - accuracy: 0.4998
20480/25000 [=======================>......] - ETA: 14s - loss: 7.6689 - accuracy: 0.4999
20512/25000 [=======================>......] - ETA: 14s - loss: 7.6704 - accuracy: 0.4998
20544/25000 [=======================>......] - ETA: 13s - loss: 7.6741 - accuracy: 0.4995
20576/25000 [=======================>......] - ETA: 13s - loss: 7.6748 - accuracy: 0.4995
20608/25000 [=======================>......] - ETA: 13s - loss: 7.6748 - accuracy: 0.4995
20640/25000 [=======================>......] - ETA: 13s - loss: 7.6711 - accuracy: 0.4997
20672/25000 [=======================>......] - ETA: 13s - loss: 7.6674 - accuracy: 0.5000
20704/25000 [=======================>......] - ETA: 13s - loss: 7.6696 - accuracy: 0.4998
20736/25000 [=======================>......] - ETA: 13s - loss: 7.6703 - accuracy: 0.4998
20768/25000 [=======================>......] - ETA: 13s - loss: 7.6718 - accuracy: 0.4997
20800/25000 [=======================>......] - ETA: 13s - loss: 7.6688 - accuracy: 0.4999
20832/25000 [=======================>......] - ETA: 13s - loss: 7.6703 - accuracy: 0.4998
20864/25000 [========================>.....] - ETA: 12s - loss: 7.6659 - accuracy: 0.5000
20896/25000 [========================>.....] - ETA: 12s - loss: 7.6696 - accuracy: 0.4998
20928/25000 [========================>.....] - ETA: 12s - loss: 7.6717 - accuracy: 0.4997
20960/25000 [========================>.....] - ETA: 12s - loss: 7.6739 - accuracy: 0.4995
20992/25000 [========================>.....] - ETA: 12s - loss: 7.6717 - accuracy: 0.4997
21024/25000 [========================>.....] - ETA: 12s - loss: 7.6732 - accuracy: 0.4996
21056/25000 [========================>.....] - ETA: 12s - loss: 7.6724 - accuracy: 0.4996
21088/25000 [========================>.....] - ETA: 12s - loss: 7.6753 - accuracy: 0.4994
21120/25000 [========================>.....] - ETA: 12s - loss: 7.6739 - accuracy: 0.4995
21152/25000 [========================>.....] - ETA: 12s - loss: 7.6775 - accuracy: 0.4993
21184/25000 [========================>.....] - ETA: 11s - loss: 7.6789 - accuracy: 0.4992
21216/25000 [========================>.....] - ETA: 11s - loss: 7.6811 - accuracy: 0.4991
21248/25000 [========================>.....] - ETA: 11s - loss: 7.6767 - accuracy: 0.4993
21280/25000 [========================>.....] - ETA: 11s - loss: 7.6738 - accuracy: 0.4995
21312/25000 [========================>.....] - ETA: 11s - loss: 7.6738 - accuracy: 0.4995
21344/25000 [========================>.....] - ETA: 11s - loss: 7.6752 - accuracy: 0.4994
21376/25000 [========================>.....] - ETA: 11s - loss: 7.6745 - accuracy: 0.4995
21408/25000 [========================>.....] - ETA: 11s - loss: 7.6738 - accuracy: 0.4995
21440/25000 [========================>.....] - ETA: 11s - loss: 7.6759 - accuracy: 0.4994
21472/25000 [========================>.....] - ETA: 11s - loss: 7.6730 - accuracy: 0.4996
21504/25000 [========================>.....] - ETA: 10s - loss: 7.6709 - accuracy: 0.4997
21536/25000 [========================>.....] - ETA: 10s - loss: 7.6730 - accuracy: 0.4996
21568/25000 [========================>.....] - ETA: 10s - loss: 7.6723 - accuracy: 0.4996
21600/25000 [========================>.....] - ETA: 10s - loss: 7.6723 - accuracy: 0.4996
21632/25000 [========================>.....] - ETA: 10s - loss: 7.6730 - accuracy: 0.4996
21664/25000 [========================>.....] - ETA: 10s - loss: 7.6723 - accuracy: 0.4996
21696/25000 [=========================>....] - ETA: 10s - loss: 7.6723 - accuracy: 0.4996
21728/25000 [=========================>....] - ETA: 10s - loss: 7.6737 - accuracy: 0.4995
21760/25000 [=========================>....] - ETA: 10s - loss: 7.6723 - accuracy: 0.4996
21792/25000 [=========================>....] - ETA: 10s - loss: 7.6722 - accuracy: 0.4996
21824/25000 [=========================>....] - ETA: 9s - loss: 7.6708 - accuracy: 0.4997 
21856/25000 [=========================>....] - ETA: 9s - loss: 7.6666 - accuracy: 0.5000
21888/25000 [=========================>....] - ETA: 9s - loss: 7.6694 - accuracy: 0.4998
21920/25000 [=========================>....] - ETA: 9s - loss: 7.6701 - accuracy: 0.4998
21952/25000 [=========================>....] - ETA: 9s - loss: 7.6673 - accuracy: 0.5000
21984/25000 [=========================>....] - ETA: 9s - loss: 7.6666 - accuracy: 0.5000
22016/25000 [=========================>....] - ETA: 9s - loss: 7.6680 - accuracy: 0.4999
22048/25000 [=========================>....] - ETA: 9s - loss: 7.6694 - accuracy: 0.4998
22080/25000 [=========================>....] - ETA: 9s - loss: 7.6687 - accuracy: 0.4999
22112/25000 [=========================>....] - ETA: 9s - loss: 7.6708 - accuracy: 0.4997
22144/25000 [=========================>....] - ETA: 8s - loss: 7.6708 - accuracy: 0.4997
22176/25000 [=========================>....] - ETA: 8s - loss: 7.6715 - accuracy: 0.4997
22208/25000 [=========================>....] - ETA: 8s - loss: 7.6708 - accuracy: 0.4997
22240/25000 [=========================>....] - ETA: 8s - loss: 7.6721 - accuracy: 0.4996
22272/25000 [=========================>....] - ETA: 8s - loss: 7.6728 - accuracy: 0.4996
22304/25000 [=========================>....] - ETA: 8s - loss: 7.6762 - accuracy: 0.4994
22336/25000 [=========================>....] - ETA: 8s - loss: 7.6735 - accuracy: 0.4996
22368/25000 [=========================>....] - ETA: 8s - loss: 7.6748 - accuracy: 0.4995
22400/25000 [=========================>....] - ETA: 8s - loss: 7.6748 - accuracy: 0.4995
22432/25000 [=========================>....] - ETA: 8s - loss: 7.6755 - accuracy: 0.4994
22464/25000 [=========================>....] - ETA: 7s - loss: 7.6714 - accuracy: 0.4997
22496/25000 [=========================>....] - ETA: 7s - loss: 7.6700 - accuracy: 0.4998
22528/25000 [==========================>...] - ETA: 7s - loss: 7.6714 - accuracy: 0.4997
22560/25000 [==========================>...] - ETA: 7s - loss: 7.6714 - accuracy: 0.4997
22592/25000 [==========================>...] - ETA: 7s - loss: 7.6720 - accuracy: 0.4996
22624/25000 [==========================>...] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
22656/25000 [==========================>...] - ETA: 7s - loss: 7.6646 - accuracy: 0.5001
22688/25000 [==========================>...] - ETA: 7s - loss: 7.6646 - accuracy: 0.5001
22720/25000 [==========================>...] - ETA: 7s - loss: 7.6653 - accuracy: 0.5001
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6639 - accuracy: 0.5002
22784/25000 [==========================>...] - ETA: 6s - loss: 7.6619 - accuracy: 0.5003
22816/25000 [==========================>...] - ETA: 6s - loss: 7.6633 - accuracy: 0.5002
22848/25000 [==========================>...] - ETA: 6s - loss: 7.6626 - accuracy: 0.5003
22880/25000 [==========================>...] - ETA: 6s - loss: 7.6586 - accuracy: 0.5005
22912/25000 [==========================>...] - ETA: 6s - loss: 7.6593 - accuracy: 0.5005
22944/25000 [==========================>...] - ETA: 6s - loss: 7.6579 - accuracy: 0.5006
22976/25000 [==========================>...] - ETA: 6s - loss: 7.6593 - accuracy: 0.5005
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6606 - accuracy: 0.5004
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6620 - accuracy: 0.5003
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6640 - accuracy: 0.5002
23104/25000 [==========================>...] - ETA: 5s - loss: 7.6640 - accuracy: 0.5002
23136/25000 [==========================>...] - ETA: 5s - loss: 7.6633 - accuracy: 0.5002
23168/25000 [==========================>...] - ETA: 5s - loss: 7.6626 - accuracy: 0.5003
23200/25000 [==========================>...] - ETA: 5s - loss: 7.6607 - accuracy: 0.5004
23232/25000 [==========================>...] - ETA: 5s - loss: 7.6607 - accuracy: 0.5004
23264/25000 [==========================>...] - ETA: 5s - loss: 7.6594 - accuracy: 0.5005
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6594 - accuracy: 0.5005
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6607 - accuracy: 0.5004
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6607 - accuracy: 0.5004
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6581 - accuracy: 0.5006
23424/25000 [===========================>..] - ETA: 4s - loss: 7.6588 - accuracy: 0.5005
23456/25000 [===========================>..] - ETA: 4s - loss: 7.6607 - accuracy: 0.5004
23488/25000 [===========================>..] - ETA: 4s - loss: 7.6627 - accuracy: 0.5003
23520/25000 [===========================>..] - ETA: 4s - loss: 7.6627 - accuracy: 0.5003
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6621 - accuracy: 0.5003
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6608 - accuracy: 0.5004
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6608 - accuracy: 0.5004
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6614 - accuracy: 0.5003
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6601 - accuracy: 0.5004
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6621 - accuracy: 0.5003
23744/25000 [===========================>..] - ETA: 3s - loss: 7.6602 - accuracy: 0.5004
23776/25000 [===========================>..] - ETA: 3s - loss: 7.6615 - accuracy: 0.5003
23808/25000 [===========================>..] - ETA: 3s - loss: 7.6615 - accuracy: 0.5003
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6595 - accuracy: 0.5005
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6621 - accuracy: 0.5003
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6621 - accuracy: 0.5003
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6621 - accuracy: 0.5003
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6589 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6602 - accuracy: 0.5004
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6609 - accuracy: 0.5004
24064/25000 [===========================>..] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24096/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6673 - accuracy: 0.5000
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24192/25000 [============================>.] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
24224/25000 [============================>.] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24256/25000 [============================>.] - ETA: 2s - loss: 7.6628 - accuracy: 0.5002
24288/25000 [============================>.] - ETA: 2s - loss: 7.6660 - accuracy: 0.5000
24320/25000 [============================>.] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
24352/25000 [============================>.] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
24384/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24416/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24448/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24480/25000 [============================>.] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
24512/25000 [============================>.] - ETA: 1s - loss: 7.6647 - accuracy: 0.5001
24544/25000 [============================>.] - ETA: 1s - loss: 7.6672 - accuracy: 0.5000
24576/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24608/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24640/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24672/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24704/25000 [============================>.] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
24736/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24768/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24800/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24832/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24864/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24896/25000 [============================>.] - ETA: 0s - loss: 7.6623 - accuracy: 0.5003
24928/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24960/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 94s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

