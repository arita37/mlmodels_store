
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
	Data preprocessing and feature engineering runtime = 0.23s ...
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
 40%|████      | 2/5 [00:49<01:14, 24.96s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.05887358897129537, 'embedding_size_factor': 0.6192660380919052, 'layers.choice': 3, 'learning_rate': 0.00021009741554367195, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 0.00012238342570110772} and reward: 0.3834
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xae$\xad\xd6u\xb02X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe3\xd1\x07\x02\xa4\x1agX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?+\x89\xb3\x0e\xb6?\x87X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G? \n\x81\x9f5\xfa\xacu.' and reward: 0.3834
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xae$\xad\xd6u\xb02X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe3\xd1\x07\x02\xa4\x1agX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?+\x89\xb3\x0e\xb6?\x87X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G? \n\x81\x9f5\xfa\xacu.' and reward: 0.3834
 60%|██████    | 3/5 [01:44<01:07, 33.90s/it] 60%|██████    | 3/5 [01:44<01:09, 34.90s/it]
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
Saving dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.25268008044805823, 'embedding_size_factor': 0.7058158059916886, 'layers.choice': 1, 'learning_rate': 0.00012051123528418403, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 1.3201920343390306e-11} and reward: 0.3808
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd0+\xe9\x12x\x00\xf8X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6\x96\x0b\x07w~nX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x1f\x97_A\xe3\xf3\xcdX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xad\x08\x05;\xb6HXu.' and reward: 0.3808
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd0+\xe9\x12x\x00\xf8X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6\x96\x0b\x07w~nX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x1f\x97_A\xe3\xf3\xcdX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xad\x08\x05;\xb6HXu.' and reward: 0.3808
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 185.4057538509369
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_2_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.77s of the -67.82s of remaining time.
Ensemble size: 77
Ensemble weights: 
[0.50649351 0.16883117 0.32467532]
	0.3932	 = Validation accuracy score
	0.98s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 188.83s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f5d8cbbfa20> 

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
 [-0.04966803  0.12734601  0.0713919   0.09146102 -0.02486509 -0.03130038]
 [-0.17437214  0.11320665 -0.03698201 -0.03522266  0.04460657  0.11960361]
 [ 0.32858816  0.18231621  0.0923181  -0.25516546  0.08681861  0.42660075]
 [ 0.01532536  0.19387545  0.21226245 -0.22567159  0.06231602  0.26168162]
 [ 0.25615865 -0.17229925  0.44683635 -0.12695831 -0.08932688 -0.19733104]
 [ 0.11262126 -0.07373199  0.04900286 -0.01673576  0.16331075  0.20091949]
 [-0.06842054  0.13006875 -0.11804266  0.17842568  0.05852774  0.06005628]
 [ 0.05070845 -0.23191056 -0.08528733 -0.24100931  0.20484166  0.26471564]
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
{'loss': 0.5008648037910461, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-14 15:16:47.166151: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.4822178892791271, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-14 15:16:48.297090: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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

    8192/17464789 [..............................] - ETA: 2s
 1556480/17464789 [=>............................] - ETA: 0s
 7069696/17464789 [===========>..................] - ETA: 0s
15286272/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 15:16:59.310095: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 15:16:59.314038: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-14 15:16:59.314208: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e9fd1821b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 15:16:59.314223: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 4:11 - loss: 6.2291 - accuracy: 0.5938
   64/25000 [..............................] - ETA: 2:46 - loss: 7.1875 - accuracy: 0.5312
   96/25000 [..............................] - ETA: 2:17 - loss: 6.8680 - accuracy: 0.5521
  128/25000 [..............................] - ETA: 2:03 - loss: 6.9479 - accuracy: 0.5469
  160/25000 [..............................] - ETA: 1:56 - loss: 6.8041 - accuracy: 0.5562
  192/25000 [..............................] - ETA: 1:50 - loss: 6.4687 - accuracy: 0.5781
  224/25000 [..............................] - ETA: 1:46 - loss: 6.2291 - accuracy: 0.5938
  256/25000 [..............................] - ETA: 1:43 - loss: 6.2890 - accuracy: 0.5898
  288/25000 [..............................] - ETA: 1:40 - loss: 6.4953 - accuracy: 0.5764
  320/25000 [..............................] - ETA: 1:38 - loss: 6.6604 - accuracy: 0.5656
  352/25000 [..............................] - ETA: 1:37 - loss: 6.7954 - accuracy: 0.5568
  384/25000 [..............................] - ETA: 1:36 - loss: 6.7083 - accuracy: 0.5625
  416/25000 [..............................] - ETA: 1:35 - loss: 6.7452 - accuracy: 0.5601
  448/25000 [..............................] - ETA: 1:34 - loss: 6.8110 - accuracy: 0.5558
  480/25000 [..............................] - ETA: 1:33 - loss: 6.9958 - accuracy: 0.5437
  512/25000 [..............................] - ETA: 1:32 - loss: 7.0078 - accuracy: 0.5430
  544/25000 [..............................] - ETA: 1:31 - loss: 7.1029 - accuracy: 0.5368
  576/25000 [..............................] - ETA: 1:31 - loss: 7.0810 - accuracy: 0.5382
  608/25000 [..............................] - ETA: 1:31 - loss: 7.0866 - accuracy: 0.5378
  640/25000 [..............................] - ETA: 1:30 - loss: 7.1635 - accuracy: 0.5328
  672/25000 [..............................] - ETA: 1:30 - loss: 7.1646 - accuracy: 0.5327
  704/25000 [..............................] - ETA: 1:29 - loss: 7.2964 - accuracy: 0.5241
  736/25000 [..............................] - ETA: 1:28 - loss: 7.3333 - accuracy: 0.5217
  768/25000 [..............................] - ETA: 1:28 - loss: 7.3272 - accuracy: 0.5221
  800/25000 [..............................] - ETA: 1:28 - loss: 7.3791 - accuracy: 0.5188
  832/25000 [..............................] - ETA: 1:27 - loss: 7.2980 - accuracy: 0.5240
  864/25000 [>.............................] - ETA: 1:28 - loss: 7.3294 - accuracy: 0.5220
  896/25000 [>.............................] - ETA: 1:27 - loss: 7.3072 - accuracy: 0.5234
  928/25000 [>.............................] - ETA: 1:27 - loss: 7.3196 - accuracy: 0.5226
  960/25000 [>.............................] - ETA: 1:27 - loss: 7.3152 - accuracy: 0.5229
  992/25000 [>.............................] - ETA: 1:26 - loss: 7.3420 - accuracy: 0.5212
 1024/25000 [>.............................] - ETA: 1:26 - loss: 7.4270 - accuracy: 0.5156
 1056/25000 [>.............................] - ETA: 1:26 - loss: 7.3472 - accuracy: 0.5208
 1088/25000 [>.............................] - ETA: 1:26 - loss: 7.3566 - accuracy: 0.5202
 1120/25000 [>.............................] - ETA: 1:26 - loss: 7.3928 - accuracy: 0.5179
 1152/25000 [>.............................] - ETA: 1:25 - loss: 7.3738 - accuracy: 0.5191
 1184/25000 [>.............................] - ETA: 1:25 - loss: 7.3688 - accuracy: 0.5194
 1216/25000 [>.............................] - ETA: 1:25 - loss: 7.3766 - accuracy: 0.5189
 1248/25000 [>.............................] - ETA: 1:25 - loss: 7.3840 - accuracy: 0.5184
 1280/25000 [>.............................] - ETA: 1:24 - loss: 7.3911 - accuracy: 0.5180
 1312/25000 [>.............................] - ETA: 1:24 - loss: 7.4446 - accuracy: 0.5145
 1344/25000 [>.............................] - ETA: 1:24 - loss: 7.4499 - accuracy: 0.5141
 1376/25000 [>.............................] - ETA: 1:24 - loss: 7.4326 - accuracy: 0.5153
 1408/25000 [>.............................] - ETA: 1:23 - loss: 7.4815 - accuracy: 0.5121
 1440/25000 [>.............................] - ETA: 1:23 - loss: 7.5175 - accuracy: 0.5097
 1472/25000 [>.............................] - ETA: 1:23 - loss: 7.4791 - accuracy: 0.5122
 1504/25000 [>.............................] - ETA: 1:23 - loss: 7.4831 - accuracy: 0.5120
 1536/25000 [>.............................] - ETA: 1:22 - loss: 7.4869 - accuracy: 0.5117
 1568/25000 [>.............................] - ETA: 1:22 - loss: 7.4417 - accuracy: 0.5147
 1600/25000 [>.............................] - ETA: 1:22 - loss: 7.4270 - accuracy: 0.5156
 1632/25000 [>.............................] - ETA: 1:22 - loss: 7.4317 - accuracy: 0.5153
 1664/25000 [>.............................] - ETA: 1:22 - loss: 7.4547 - accuracy: 0.5138
 1696/25000 [=>............................] - ETA: 1:21 - loss: 7.4677 - accuracy: 0.5130
 1728/25000 [=>............................] - ETA: 1:21 - loss: 7.5069 - accuracy: 0.5104
 1760/25000 [=>............................] - ETA: 1:21 - loss: 7.5011 - accuracy: 0.5108
 1792/25000 [=>............................] - ETA: 1:21 - loss: 7.4955 - accuracy: 0.5112
 1824/25000 [=>............................] - ETA: 1:20 - loss: 7.5321 - accuracy: 0.5088
 1856/25000 [=>............................] - ETA: 1:20 - loss: 7.5427 - accuracy: 0.5081
 1888/25000 [=>............................] - ETA: 1:20 - loss: 7.5610 - accuracy: 0.5069
 1920/25000 [=>............................] - ETA: 1:20 - loss: 7.5468 - accuracy: 0.5078
 1952/25000 [=>............................] - ETA: 1:20 - loss: 7.5645 - accuracy: 0.5067
 1984/25000 [=>............................] - ETA: 1:20 - loss: 7.5430 - accuracy: 0.5081
 2016/25000 [=>............................] - ETA: 1:19 - loss: 7.5221 - accuracy: 0.5094
 2048/25000 [=>............................] - ETA: 1:19 - loss: 7.5319 - accuracy: 0.5088
 2080/25000 [=>............................] - ETA: 1:19 - loss: 7.5044 - accuracy: 0.5106
 2112/25000 [=>............................] - ETA: 1:19 - loss: 7.4996 - accuracy: 0.5109
 2144/25000 [=>............................] - ETA: 1:19 - loss: 7.5236 - accuracy: 0.5093
 2176/25000 [=>............................] - ETA: 1:19 - loss: 7.5257 - accuracy: 0.5092
 2208/25000 [=>............................] - ETA: 1:18 - loss: 7.5347 - accuracy: 0.5086
 2240/25000 [=>............................] - ETA: 1:18 - loss: 7.5160 - accuracy: 0.5098
 2272/25000 [=>............................] - ETA: 1:18 - loss: 7.4844 - accuracy: 0.5119
 2304/25000 [=>............................] - ETA: 1:18 - loss: 7.4803 - accuracy: 0.5122
 2336/25000 [=>............................] - ETA: 1:18 - loss: 7.5025 - accuracy: 0.5107
 2368/25000 [=>............................] - ETA: 1:18 - loss: 7.5242 - accuracy: 0.5093
 2400/25000 [=>............................] - ETA: 1:17 - loss: 7.5197 - accuracy: 0.5096
 2432/25000 [=>............................] - ETA: 1:17 - loss: 7.5657 - accuracy: 0.5066
 2464/25000 [=>............................] - ETA: 1:17 - loss: 7.5608 - accuracy: 0.5069
 2496/25000 [=>............................] - ETA: 1:17 - loss: 7.5622 - accuracy: 0.5068
 2528/25000 [==>...........................] - ETA: 1:17 - loss: 7.5332 - accuracy: 0.5087
 2560/25000 [==>...........................] - ETA: 1:17 - loss: 7.4989 - accuracy: 0.5109
 2592/25000 [==>...........................] - ETA: 1:17 - loss: 7.5187 - accuracy: 0.5096
 2624/25000 [==>...........................] - ETA: 1:17 - loss: 7.5498 - accuracy: 0.5076
 2656/25000 [==>...........................] - ETA: 1:17 - loss: 7.5916 - accuracy: 0.5049
 2688/25000 [==>...........................] - ETA: 1:17 - loss: 7.6096 - accuracy: 0.5037
 2720/25000 [==>...........................] - ETA: 1:16 - loss: 7.5990 - accuracy: 0.5044
 2752/25000 [==>...........................] - ETA: 1:16 - loss: 7.6053 - accuracy: 0.5040
 2784/25000 [==>...........................] - ETA: 1:16 - loss: 7.5950 - accuracy: 0.5047
 2816/25000 [==>...........................] - ETA: 1:16 - loss: 7.6176 - accuracy: 0.5032
 2848/25000 [==>...........................] - ETA: 1:16 - loss: 7.6289 - accuracy: 0.5025
 2880/25000 [==>...........................] - ETA: 1:16 - loss: 7.6294 - accuracy: 0.5024
 2912/25000 [==>...........................] - ETA: 1:15 - loss: 7.6298 - accuracy: 0.5024
 2944/25000 [==>...........................] - ETA: 1:15 - loss: 7.6406 - accuracy: 0.5017
 2976/25000 [==>...........................] - ETA: 1:15 - loss: 7.6460 - accuracy: 0.5013
 3008/25000 [==>...........................] - ETA: 1:15 - loss: 7.6513 - accuracy: 0.5010
 3040/25000 [==>...........................] - ETA: 1:15 - loss: 7.6616 - accuracy: 0.5003
 3072/25000 [==>...........................] - ETA: 1:15 - loss: 7.6766 - accuracy: 0.4993
 3104/25000 [==>...........................] - ETA: 1:15 - loss: 7.6963 - accuracy: 0.4981
 3136/25000 [==>...........................] - ETA: 1:15 - loss: 7.6960 - accuracy: 0.4981
 3168/25000 [==>...........................] - ETA: 1:15 - loss: 7.7005 - accuracy: 0.4978
 3200/25000 [==>...........................] - ETA: 1:15 - loss: 7.7241 - accuracy: 0.4963
 3232/25000 [==>...........................] - ETA: 1:15 - loss: 7.7283 - accuracy: 0.4960
 3264/25000 [==>...........................] - ETA: 1:14 - loss: 7.7230 - accuracy: 0.4963
 3296/25000 [==>...........................] - ETA: 1:14 - loss: 7.7364 - accuracy: 0.4954
 3328/25000 [==>...........................] - ETA: 1:14 - loss: 7.7127 - accuracy: 0.4970
 3360/25000 [===>..........................] - ETA: 1:14 - loss: 7.7123 - accuracy: 0.4970
 3392/25000 [===>..........................] - ETA: 1:14 - loss: 7.7254 - accuracy: 0.4962
 3424/25000 [===>..........................] - ETA: 1:14 - loss: 7.7114 - accuracy: 0.4971
 3456/25000 [===>..........................] - ETA: 1:14 - loss: 7.7332 - accuracy: 0.4957
 3488/25000 [===>..........................] - ETA: 1:14 - loss: 7.7282 - accuracy: 0.4960
 3520/25000 [===>..........................] - ETA: 1:14 - loss: 7.7276 - accuracy: 0.4960
 3552/25000 [===>..........................] - ETA: 1:14 - loss: 7.7357 - accuracy: 0.4955
 3584/25000 [===>..........................] - ETA: 1:13 - loss: 7.7265 - accuracy: 0.4961
 3616/25000 [===>..........................] - ETA: 1:13 - loss: 7.7090 - accuracy: 0.4972
 3648/25000 [===>..........................] - ETA: 1:13 - loss: 7.7087 - accuracy: 0.4973
 3680/25000 [===>..........................] - ETA: 1:13 - loss: 7.7083 - accuracy: 0.4973
 3712/25000 [===>..........................] - ETA: 1:13 - loss: 7.7038 - accuracy: 0.4976
 3744/25000 [===>..........................] - ETA: 1:13 - loss: 7.6912 - accuracy: 0.4984
 3776/25000 [===>..........................] - ETA: 1:13 - loss: 7.6747 - accuracy: 0.4995
 3808/25000 [===>..........................] - ETA: 1:13 - loss: 7.6706 - accuracy: 0.4997
 3840/25000 [===>..........................] - ETA: 1:13 - loss: 7.6427 - accuracy: 0.5016
 3872/25000 [===>..........................] - ETA: 1:13 - loss: 7.6270 - accuracy: 0.5026
 3904/25000 [===>..........................] - ETA: 1:12 - loss: 7.6234 - accuracy: 0.5028
 3936/25000 [===>..........................] - ETA: 1:12 - loss: 7.6160 - accuracy: 0.5033
 3968/25000 [===>..........................] - ETA: 1:12 - loss: 7.6241 - accuracy: 0.5028
 4000/25000 [===>..........................] - ETA: 1:12 - loss: 7.6360 - accuracy: 0.5020
 4032/25000 [===>..........................] - ETA: 1:12 - loss: 7.6438 - accuracy: 0.5015
 4064/25000 [===>..........................] - ETA: 1:12 - loss: 7.6553 - accuracy: 0.5007
 4096/25000 [===>..........................] - ETA: 1:12 - loss: 7.6554 - accuracy: 0.5007
 4128/25000 [===>..........................] - ETA: 1:12 - loss: 7.6518 - accuracy: 0.5010
 4160/25000 [===>..........................] - ETA: 1:11 - loss: 7.6408 - accuracy: 0.5017
 4192/25000 [====>.........................] - ETA: 1:11 - loss: 7.6410 - accuracy: 0.5017
 4224/25000 [====>.........................] - ETA: 1:11 - loss: 7.6231 - accuracy: 0.5028
 4256/25000 [====>.........................] - ETA: 1:11 - loss: 7.6162 - accuracy: 0.5033
 4288/25000 [====>.........................] - ETA: 1:11 - loss: 7.6166 - accuracy: 0.5033
 4320/25000 [====>.........................] - ETA: 1:11 - loss: 7.6276 - accuracy: 0.5025
 4352/25000 [====>.........................] - ETA: 1:11 - loss: 7.6279 - accuracy: 0.5025
 4384/25000 [====>.........................] - ETA: 1:11 - loss: 7.6246 - accuracy: 0.5027
 4416/25000 [====>.........................] - ETA: 1:11 - loss: 7.6284 - accuracy: 0.5025
 4448/25000 [====>.........................] - ETA: 1:11 - loss: 7.6080 - accuracy: 0.5038
 4480/25000 [====>.........................] - ETA: 1:10 - loss: 7.6016 - accuracy: 0.5042
 4512/25000 [====>.........................] - ETA: 1:10 - loss: 7.6122 - accuracy: 0.5035
 4544/25000 [====>.........................] - ETA: 1:10 - loss: 7.6093 - accuracy: 0.5037
 4576/25000 [====>.........................] - ETA: 1:10 - loss: 7.6164 - accuracy: 0.5033
 4608/25000 [====>.........................] - ETA: 1:10 - loss: 7.6200 - accuracy: 0.5030
 4640/25000 [====>.........................] - ETA: 1:10 - loss: 7.6137 - accuracy: 0.5034
 4672/25000 [====>.........................] - ETA: 1:10 - loss: 7.6075 - accuracy: 0.5039
 4704/25000 [====>.........................] - ETA: 1:09 - loss: 7.5949 - accuracy: 0.5047
 4736/25000 [====>.........................] - ETA: 1:09 - loss: 7.6116 - accuracy: 0.5036
 4768/25000 [====>.........................] - ETA: 1:09 - loss: 7.6119 - accuracy: 0.5036
 4800/25000 [====>.........................] - ETA: 1:09 - loss: 7.6123 - accuracy: 0.5035
 4832/25000 [====>.........................] - ETA: 1:09 - loss: 7.6158 - accuracy: 0.5033
 4864/25000 [====>.........................] - ETA: 1:09 - loss: 7.6067 - accuracy: 0.5039
 4896/25000 [====>.........................] - ETA: 1:09 - loss: 7.6228 - accuracy: 0.5029
 4928/25000 [====>.........................] - ETA: 1:09 - loss: 7.6355 - accuracy: 0.5020
 4960/25000 [====>.........................] - ETA: 1:09 - loss: 7.6264 - accuracy: 0.5026
 4992/25000 [====>.........................] - ETA: 1:09 - loss: 7.6359 - accuracy: 0.5020
 5024/25000 [=====>........................] - ETA: 1:08 - loss: 7.6239 - accuracy: 0.5028
 5056/25000 [=====>........................] - ETA: 1:08 - loss: 7.6242 - accuracy: 0.5028
 5088/25000 [=====>........................] - ETA: 1:08 - loss: 7.6335 - accuracy: 0.5022
 5120/25000 [=====>........................] - ETA: 1:08 - loss: 7.6367 - accuracy: 0.5020
 5152/25000 [=====>........................] - ETA: 1:08 - loss: 7.6220 - accuracy: 0.5029
 5184/25000 [=====>........................] - ETA: 1:08 - loss: 7.6193 - accuracy: 0.5031
 5216/25000 [=====>........................] - ETA: 1:08 - loss: 7.6225 - accuracy: 0.5029
 5248/25000 [=====>........................] - ETA: 1:08 - loss: 7.6257 - accuracy: 0.5027
 5280/25000 [=====>........................] - ETA: 1:08 - loss: 7.6289 - accuracy: 0.5025
 5312/25000 [=====>........................] - ETA: 1:07 - loss: 7.6233 - accuracy: 0.5028
 5344/25000 [=====>........................] - ETA: 1:07 - loss: 7.6178 - accuracy: 0.5032
 5376/25000 [=====>........................] - ETA: 1:07 - loss: 7.6096 - accuracy: 0.5037
 5408/25000 [=====>........................] - ETA: 1:07 - loss: 7.6071 - accuracy: 0.5039
 5440/25000 [=====>........................] - ETA: 1:07 - loss: 7.6046 - accuracy: 0.5040
 5472/25000 [=====>........................] - ETA: 1:07 - loss: 7.6050 - accuracy: 0.5040
 5504/25000 [=====>........................] - ETA: 1:07 - loss: 7.6025 - accuracy: 0.5042
 5536/25000 [=====>........................] - ETA: 1:07 - loss: 7.6085 - accuracy: 0.5038
 5568/25000 [=====>........................] - ETA: 1:06 - loss: 7.6143 - accuracy: 0.5034
 5600/25000 [=====>........................] - ETA: 1:06 - loss: 7.6146 - accuracy: 0.5034
 5632/25000 [=====>........................] - ETA: 1:06 - loss: 7.6312 - accuracy: 0.5023
 5664/25000 [=====>........................] - ETA: 1:06 - loss: 7.6125 - accuracy: 0.5035
 5696/25000 [=====>........................] - ETA: 1:06 - loss: 7.6101 - accuracy: 0.5037
 5728/25000 [=====>........................] - ETA: 1:06 - loss: 7.6158 - accuracy: 0.5033
 5760/25000 [=====>........................] - ETA: 1:06 - loss: 7.6160 - accuracy: 0.5033
 5792/25000 [=====>........................] - ETA: 1:06 - loss: 7.6190 - accuracy: 0.5031
 5824/25000 [=====>........................] - ETA: 1:05 - loss: 7.6113 - accuracy: 0.5036
 5856/25000 [======>.......................] - ETA: 1:05 - loss: 7.6169 - accuracy: 0.5032
 5888/25000 [======>.......................] - ETA: 1:05 - loss: 7.6145 - accuracy: 0.5034
 5920/25000 [======>.......................] - ETA: 1:05 - loss: 7.6045 - accuracy: 0.5041
 5952/25000 [======>.......................] - ETA: 1:05 - loss: 7.5971 - accuracy: 0.5045
 5984/25000 [======>.......................] - ETA: 1:05 - loss: 7.5974 - accuracy: 0.5045
 6016/25000 [======>.......................] - ETA: 1:05 - loss: 7.6080 - accuracy: 0.5038
 6048/25000 [======>.......................] - ETA: 1:05 - loss: 7.6108 - accuracy: 0.5036
 6080/25000 [======>.......................] - ETA: 1:05 - loss: 7.6010 - accuracy: 0.5043
 6112/25000 [======>.......................] - ETA: 1:04 - loss: 7.6089 - accuracy: 0.5038
 6144/25000 [======>.......................] - ETA: 1:04 - loss: 7.6067 - accuracy: 0.5039
 6176/25000 [======>.......................] - ETA: 1:04 - loss: 7.6070 - accuracy: 0.5039
 6208/25000 [======>.......................] - ETA: 1:04 - loss: 7.6123 - accuracy: 0.5035
 6240/25000 [======>.......................] - ETA: 1:04 - loss: 7.6027 - accuracy: 0.5042
 6272/25000 [======>.......................] - ETA: 1:04 - loss: 7.5982 - accuracy: 0.5045
 6304/25000 [======>.......................] - ETA: 1:04 - loss: 7.5985 - accuracy: 0.5044
 6336/25000 [======>.......................] - ETA: 1:04 - loss: 7.6061 - accuracy: 0.5039
 6368/25000 [======>.......................] - ETA: 1:03 - loss: 7.6064 - accuracy: 0.5039
 6400/25000 [======>.......................] - ETA: 1:03 - loss: 7.6067 - accuracy: 0.5039
 6432/25000 [======>.......................] - ETA: 1:03 - loss: 7.6166 - accuracy: 0.5033
 6464/25000 [======>.......................] - ETA: 1:03 - loss: 7.6287 - accuracy: 0.5025
 6496/25000 [======>.......................] - ETA: 1:03 - loss: 7.6194 - accuracy: 0.5031
 6528/25000 [======>.......................] - ETA: 1:03 - loss: 7.6243 - accuracy: 0.5028
 6560/25000 [======>.......................] - ETA: 1:03 - loss: 7.6316 - accuracy: 0.5023
 6592/25000 [======>.......................] - ETA: 1:03 - loss: 7.6224 - accuracy: 0.5029
 6624/25000 [======>.......................] - ETA: 1:03 - loss: 7.6273 - accuracy: 0.5026
 6656/25000 [======>.......................] - ETA: 1:02 - loss: 7.6275 - accuracy: 0.5026
 6688/25000 [=======>......................] - ETA: 1:02 - loss: 7.6185 - accuracy: 0.5031
 6720/25000 [=======>......................] - ETA: 1:02 - loss: 7.6255 - accuracy: 0.5027
 6752/25000 [=======>......................] - ETA: 1:02 - loss: 7.6348 - accuracy: 0.5021
 6784/25000 [=======>......................] - ETA: 1:02 - loss: 7.6214 - accuracy: 0.5029
 6816/25000 [=======>......................] - ETA: 1:02 - loss: 7.6171 - accuracy: 0.5032
 6848/25000 [=======>......................] - ETA: 1:02 - loss: 7.6218 - accuracy: 0.5029
 6880/25000 [=======>......................] - ETA: 1:02 - loss: 7.6243 - accuracy: 0.5028
 6912/25000 [=======>......................] - ETA: 1:01 - loss: 7.6200 - accuracy: 0.5030
 6944/25000 [=======>......................] - ETA: 1:01 - loss: 7.6114 - accuracy: 0.5036
 6976/25000 [=======>......................] - ETA: 1:01 - loss: 7.6073 - accuracy: 0.5039
 7008/25000 [=======>......................] - ETA: 1:01 - loss: 7.5922 - accuracy: 0.5049
 7040/25000 [=======>......................] - ETA: 1:01 - loss: 7.5926 - accuracy: 0.5048
 7072/25000 [=======>......................] - ETA: 1:01 - loss: 7.5842 - accuracy: 0.5054
 7104/25000 [=======>......................] - ETA: 1:01 - loss: 7.5932 - accuracy: 0.5048
 7136/25000 [=======>......................] - ETA: 1:01 - loss: 7.5893 - accuracy: 0.5050
 7168/25000 [=======>......................] - ETA: 1:01 - loss: 7.5896 - accuracy: 0.5050
 7200/25000 [=======>......................] - ETA: 1:00 - loss: 7.5921 - accuracy: 0.5049
 7232/25000 [=======>......................] - ETA: 1:00 - loss: 7.5945 - accuracy: 0.5047
 7264/25000 [=======>......................] - ETA: 1:00 - loss: 7.5991 - accuracy: 0.5044
 7296/25000 [=======>......................] - ETA: 1:00 - loss: 7.6015 - accuracy: 0.5042
 7328/25000 [=======>......................] - ETA: 1:00 - loss: 7.5955 - accuracy: 0.5046
 7360/25000 [=======>......................] - ETA: 1:00 - loss: 7.5937 - accuracy: 0.5048
 7392/25000 [=======>......................] - ETA: 1:00 - loss: 7.5940 - accuracy: 0.5047
 7424/25000 [=======>......................] - ETA: 1:00 - loss: 7.5881 - accuracy: 0.5051
 7456/25000 [=======>......................] - ETA: 1:00 - loss: 7.5926 - accuracy: 0.5048
 7488/25000 [=======>......................] - ETA: 59s - loss: 7.6031 - accuracy: 0.5041 
 7520/25000 [========>.....................] - ETA: 59s - loss: 7.5993 - accuracy: 0.5044
 7552/25000 [========>.....................] - ETA: 59s - loss: 7.6016 - accuracy: 0.5042
 7584/25000 [========>.....................] - ETA: 59s - loss: 7.6060 - accuracy: 0.5040
 7616/25000 [========>.....................] - ETA: 59s - loss: 7.6002 - accuracy: 0.5043
 7648/25000 [========>.....................] - ETA: 59s - loss: 7.6005 - accuracy: 0.5043
 7680/25000 [========>.....................] - ETA: 59s - loss: 7.6047 - accuracy: 0.5040
 7712/25000 [========>.....................] - ETA: 59s - loss: 7.5931 - accuracy: 0.5048
 7744/25000 [========>.....................] - ETA: 59s - loss: 7.5953 - accuracy: 0.5046
 7776/25000 [========>.....................] - ETA: 58s - loss: 7.5996 - accuracy: 0.5044
 7808/25000 [========>.....................] - ETA: 58s - loss: 7.5959 - accuracy: 0.5046
 7840/25000 [========>.....................] - ETA: 58s - loss: 7.5923 - accuracy: 0.5048
 7872/25000 [========>.....................] - ETA: 58s - loss: 7.5965 - accuracy: 0.5046
 7904/25000 [========>.....................] - ETA: 58s - loss: 7.5968 - accuracy: 0.5046
 7936/25000 [========>.....................] - ETA: 58s - loss: 7.6067 - accuracy: 0.5039
 7968/25000 [========>.....................] - ETA: 58s - loss: 7.6089 - accuracy: 0.5038
 8000/25000 [========>.....................] - ETA: 58s - loss: 7.6072 - accuracy: 0.5039
 8032/25000 [========>.....................] - ETA: 58s - loss: 7.6055 - accuracy: 0.5040
 8064/25000 [========>.....................] - ETA: 58s - loss: 7.6153 - accuracy: 0.5033
 8096/25000 [========>.....................] - ETA: 57s - loss: 7.6136 - accuracy: 0.5035
 8128/25000 [========>.....................] - ETA: 57s - loss: 7.6213 - accuracy: 0.5030
 8160/25000 [========>.....................] - ETA: 57s - loss: 7.6234 - accuracy: 0.5028
 8192/25000 [========>.....................] - ETA: 57s - loss: 7.6311 - accuracy: 0.5023
 8224/25000 [========>.....................] - ETA: 57s - loss: 7.6275 - accuracy: 0.5026
 8256/25000 [========>.....................] - ETA: 57s - loss: 7.6369 - accuracy: 0.5019
 8288/25000 [========>.....................] - ETA: 57s - loss: 7.6296 - accuracy: 0.5024
 8320/25000 [========>.....................] - ETA: 57s - loss: 7.6353 - accuracy: 0.5020
 8352/25000 [=========>....................] - ETA: 57s - loss: 7.6391 - accuracy: 0.5018
 8384/25000 [=========>....................] - ETA: 56s - loss: 7.6392 - accuracy: 0.5018
 8416/25000 [=========>....................] - ETA: 56s - loss: 7.6411 - accuracy: 0.5017
 8448/25000 [=========>....................] - ETA: 56s - loss: 7.6376 - accuracy: 0.5019
 8480/25000 [=========>....................] - ETA: 56s - loss: 7.6341 - accuracy: 0.5021
 8512/25000 [=========>....................] - ETA: 56s - loss: 7.6396 - accuracy: 0.5018
 8544/25000 [=========>....................] - ETA: 56s - loss: 7.6415 - accuracy: 0.5016
 8576/25000 [=========>....................] - ETA: 56s - loss: 7.6398 - accuracy: 0.5017
 8608/25000 [=========>....................] - ETA: 56s - loss: 7.6417 - accuracy: 0.5016
 8640/25000 [=========>....................] - ETA: 55s - loss: 7.6382 - accuracy: 0.5019
 8672/25000 [=========>....................] - ETA: 55s - loss: 7.6330 - accuracy: 0.5022
 8704/25000 [=========>....................] - ETA: 55s - loss: 7.6296 - accuracy: 0.5024
 8736/25000 [=========>....................] - ETA: 55s - loss: 7.6175 - accuracy: 0.5032
 8768/25000 [=========>....................] - ETA: 55s - loss: 7.6281 - accuracy: 0.5025
 8800/25000 [=========>....................] - ETA: 55s - loss: 7.6248 - accuracy: 0.5027
 8832/25000 [=========>....................] - ETA: 55s - loss: 7.6284 - accuracy: 0.5025
 8864/25000 [=========>....................] - ETA: 55s - loss: 7.6320 - accuracy: 0.5023
 8896/25000 [=========>....................] - ETA: 55s - loss: 7.6373 - accuracy: 0.5019
 8928/25000 [=========>....................] - ETA: 54s - loss: 7.6391 - accuracy: 0.5018
 8960/25000 [=========>....................] - ETA: 54s - loss: 7.6444 - accuracy: 0.5015
 8992/25000 [=========>....................] - ETA: 54s - loss: 7.6427 - accuracy: 0.5016
 9024/25000 [=========>....................] - ETA: 54s - loss: 7.6445 - accuracy: 0.5014
 9056/25000 [=========>....................] - ETA: 54s - loss: 7.6463 - accuracy: 0.5013
 9088/25000 [=========>....................] - ETA: 54s - loss: 7.6565 - accuracy: 0.5007
 9120/25000 [=========>....................] - ETA: 54s - loss: 7.6549 - accuracy: 0.5008
 9152/25000 [=========>....................] - ETA: 54s - loss: 7.6549 - accuracy: 0.5008
 9184/25000 [==========>...................] - ETA: 53s - loss: 7.6516 - accuracy: 0.5010
 9216/25000 [==========>...................] - ETA: 53s - loss: 7.6566 - accuracy: 0.5007
 9248/25000 [==========>...................] - ETA: 53s - loss: 7.6633 - accuracy: 0.5002
 9280/25000 [==========>...................] - ETA: 53s - loss: 7.6650 - accuracy: 0.5001
 9312/25000 [==========>...................] - ETA: 53s - loss: 7.6567 - accuracy: 0.5006
 9344/25000 [==========>...................] - ETA: 53s - loss: 7.6584 - accuracy: 0.5005
 9376/25000 [==========>...................] - ETA: 53s - loss: 7.6568 - accuracy: 0.5006
 9408/25000 [==========>...................] - ETA: 53s - loss: 7.6585 - accuracy: 0.5005
 9440/25000 [==========>...................] - ETA: 53s - loss: 7.6536 - accuracy: 0.5008
 9472/25000 [==========>...................] - ETA: 52s - loss: 7.6537 - accuracy: 0.5008
 9504/25000 [==========>...................] - ETA: 52s - loss: 7.6473 - accuracy: 0.5013
 9536/25000 [==========>...................] - ETA: 52s - loss: 7.6425 - accuracy: 0.5016
 9568/25000 [==========>...................] - ETA: 52s - loss: 7.6474 - accuracy: 0.5013
 9600/25000 [==========>...................] - ETA: 52s - loss: 7.6491 - accuracy: 0.5011
 9632/25000 [==========>...................] - ETA: 52s - loss: 7.6539 - accuracy: 0.5008
 9664/25000 [==========>...................] - ETA: 52s - loss: 7.6539 - accuracy: 0.5008
 9696/25000 [==========>...................] - ETA: 52s - loss: 7.6524 - accuracy: 0.5009
 9728/25000 [==========>...................] - ETA: 52s - loss: 7.6556 - accuracy: 0.5007
 9760/25000 [==========>...................] - ETA: 52s - loss: 7.6556 - accuracy: 0.5007
 9792/25000 [==========>...................] - ETA: 51s - loss: 7.6604 - accuracy: 0.5004
 9824/25000 [==========>...................] - ETA: 51s - loss: 7.6619 - accuracy: 0.5003
 9856/25000 [==========>...................] - ETA: 51s - loss: 7.6651 - accuracy: 0.5001
 9888/25000 [==========>...................] - ETA: 51s - loss: 7.6713 - accuracy: 0.4997
 9920/25000 [==========>...................] - ETA: 51s - loss: 7.6759 - accuracy: 0.4994
 9952/25000 [==========>...................] - ETA: 51s - loss: 7.6774 - accuracy: 0.4993
 9984/25000 [==========>...................] - ETA: 51s - loss: 7.6758 - accuracy: 0.4994
10016/25000 [===========>..................] - ETA: 51s - loss: 7.6789 - accuracy: 0.4992
10048/25000 [===========>..................] - ETA: 51s - loss: 7.6773 - accuracy: 0.4993
10080/25000 [===========>..................] - ETA: 50s - loss: 7.6818 - accuracy: 0.4990
10112/25000 [===========>..................] - ETA: 50s - loss: 7.6833 - accuracy: 0.4989
10144/25000 [===========>..................] - ETA: 50s - loss: 7.6848 - accuracy: 0.4988
10176/25000 [===========>..................] - ETA: 50s - loss: 7.6787 - accuracy: 0.4992
10208/25000 [===========>..................] - ETA: 50s - loss: 7.6696 - accuracy: 0.4998
10240/25000 [===========>..................] - ETA: 50s - loss: 7.6666 - accuracy: 0.5000
10272/25000 [===========>..................] - ETA: 50s - loss: 7.6666 - accuracy: 0.5000
10304/25000 [===========>..................] - ETA: 50s - loss: 7.6755 - accuracy: 0.4994
10336/25000 [===========>..................] - ETA: 50s - loss: 7.6785 - accuracy: 0.4992
10368/25000 [===========>..................] - ETA: 49s - loss: 7.6814 - accuracy: 0.4990
10400/25000 [===========>..................] - ETA: 49s - loss: 7.6843 - accuracy: 0.4988
10432/25000 [===========>..................] - ETA: 49s - loss: 7.6828 - accuracy: 0.4989
10464/25000 [===========>..................] - ETA: 49s - loss: 7.6827 - accuracy: 0.4989
10496/25000 [===========>..................] - ETA: 49s - loss: 7.6783 - accuracy: 0.4992
10528/25000 [===========>..................] - ETA: 49s - loss: 7.6724 - accuracy: 0.4996
10560/25000 [===========>..................] - ETA: 49s - loss: 7.6681 - accuracy: 0.4999
10592/25000 [===========>..................] - ETA: 49s - loss: 7.6681 - accuracy: 0.4999
10624/25000 [===========>..................] - ETA: 49s - loss: 7.6681 - accuracy: 0.4999
10656/25000 [===========>..................] - ETA: 48s - loss: 7.6767 - accuracy: 0.4993
10688/25000 [===========>..................] - ETA: 48s - loss: 7.6752 - accuracy: 0.4994
10720/25000 [===========>..................] - ETA: 48s - loss: 7.6695 - accuracy: 0.4998
10752/25000 [===========>..................] - ETA: 48s - loss: 7.6680 - accuracy: 0.4999
10784/25000 [===========>..................] - ETA: 48s - loss: 7.6695 - accuracy: 0.4998
10816/25000 [===========>..................] - ETA: 48s - loss: 7.6765 - accuracy: 0.4994
10848/25000 [============>.................] - ETA: 48s - loss: 7.6779 - accuracy: 0.4993
10880/25000 [============>.................] - ETA: 48s - loss: 7.6793 - accuracy: 0.4992
10912/25000 [============>.................] - ETA: 48s - loss: 7.6793 - accuracy: 0.4992
10944/25000 [============>.................] - ETA: 48s - loss: 7.6862 - accuracy: 0.4987
10976/25000 [============>.................] - ETA: 47s - loss: 7.6848 - accuracy: 0.4988
11008/25000 [============>.................] - ETA: 47s - loss: 7.6819 - accuracy: 0.4990
11040/25000 [============>.................] - ETA: 47s - loss: 7.6777 - accuracy: 0.4993
11072/25000 [============>.................] - ETA: 47s - loss: 7.6763 - accuracy: 0.4994
11104/25000 [============>.................] - ETA: 47s - loss: 7.6735 - accuracy: 0.4995
11136/25000 [============>.................] - ETA: 47s - loss: 7.6694 - accuracy: 0.4998
11168/25000 [============>.................] - ETA: 47s - loss: 7.6721 - accuracy: 0.4996
11200/25000 [============>.................] - ETA: 47s - loss: 7.6762 - accuracy: 0.4994
11232/25000 [============>.................] - ETA: 47s - loss: 7.6789 - accuracy: 0.4992
11264/25000 [============>.................] - ETA: 46s - loss: 7.6761 - accuracy: 0.4994
11296/25000 [============>.................] - ETA: 46s - loss: 7.6775 - accuracy: 0.4993
11328/25000 [============>.................] - ETA: 46s - loss: 7.6747 - accuracy: 0.4995
11360/25000 [============>.................] - ETA: 46s - loss: 7.6747 - accuracy: 0.4995
11392/25000 [============>.................] - ETA: 46s - loss: 7.6680 - accuracy: 0.4999
11424/25000 [============>.................] - ETA: 46s - loss: 7.6639 - accuracy: 0.5002
11456/25000 [============>.................] - ETA: 46s - loss: 7.6639 - accuracy: 0.5002
11488/25000 [============>.................] - ETA: 46s - loss: 7.6599 - accuracy: 0.5004
11520/25000 [============>.................] - ETA: 46s - loss: 7.6626 - accuracy: 0.5003
11552/25000 [============>.................] - ETA: 46s - loss: 7.6560 - accuracy: 0.5007
11584/25000 [============>.................] - ETA: 45s - loss: 7.6547 - accuracy: 0.5008
11616/25000 [============>.................] - ETA: 45s - loss: 7.6561 - accuracy: 0.5007
11648/25000 [============>.................] - ETA: 45s - loss: 7.6561 - accuracy: 0.5007
11680/25000 [=============>................] - ETA: 45s - loss: 7.6548 - accuracy: 0.5008
11712/25000 [=============>................] - ETA: 45s - loss: 7.6548 - accuracy: 0.5008
11744/25000 [=============>................] - ETA: 45s - loss: 7.6523 - accuracy: 0.5009
11776/25000 [=============>................] - ETA: 45s - loss: 7.6523 - accuracy: 0.5009
11808/25000 [=============>................] - ETA: 45s - loss: 7.6510 - accuracy: 0.5010
11840/25000 [=============>................] - ETA: 44s - loss: 7.6472 - accuracy: 0.5013
11872/25000 [=============>................] - ETA: 44s - loss: 7.6485 - accuracy: 0.5012
11904/25000 [=============>................] - ETA: 44s - loss: 7.6537 - accuracy: 0.5008
11936/25000 [=============>................] - ETA: 44s - loss: 7.6563 - accuracy: 0.5007
11968/25000 [=============>................] - ETA: 44s - loss: 7.6564 - accuracy: 0.5007
12000/25000 [=============>................] - ETA: 44s - loss: 7.6564 - accuracy: 0.5007
12032/25000 [=============>................] - ETA: 44s - loss: 7.6590 - accuracy: 0.5005
12064/25000 [=============>................] - ETA: 44s - loss: 7.6641 - accuracy: 0.5002
12096/25000 [=============>................] - ETA: 44s - loss: 7.6717 - accuracy: 0.4997
12128/25000 [=============>................] - ETA: 44s - loss: 7.6704 - accuracy: 0.4998
12160/25000 [=============>................] - ETA: 43s - loss: 7.6679 - accuracy: 0.4999
12192/25000 [=============>................] - ETA: 43s - loss: 7.6767 - accuracy: 0.4993
12224/25000 [=============>................] - ETA: 43s - loss: 7.6779 - accuracy: 0.4993
12256/25000 [=============>................] - ETA: 43s - loss: 7.6766 - accuracy: 0.4993
12288/25000 [=============>................] - ETA: 43s - loss: 7.6766 - accuracy: 0.4993
12320/25000 [=============>................] - ETA: 43s - loss: 7.6716 - accuracy: 0.4997
12352/25000 [=============>................] - ETA: 43s - loss: 7.6691 - accuracy: 0.4998
12384/25000 [=============>................] - ETA: 43s - loss: 7.6654 - accuracy: 0.5001
12416/25000 [=============>................] - ETA: 43s - loss: 7.6654 - accuracy: 0.5001
12448/25000 [=============>................] - ETA: 42s - loss: 7.6715 - accuracy: 0.4997
12480/25000 [=============>................] - ETA: 42s - loss: 7.6654 - accuracy: 0.5001
12512/25000 [==============>...............] - ETA: 42s - loss: 7.6691 - accuracy: 0.4998
12544/25000 [==============>...............] - ETA: 42s - loss: 7.6654 - accuracy: 0.5001
12576/25000 [==============>...............] - ETA: 42s - loss: 7.6691 - accuracy: 0.4998
12608/25000 [==============>...............] - ETA: 42s - loss: 7.6703 - accuracy: 0.4998
12640/25000 [==============>...............] - ETA: 42s - loss: 7.6751 - accuracy: 0.4994
12672/25000 [==============>...............] - ETA: 42s - loss: 7.6739 - accuracy: 0.4995
12704/25000 [==============>...............] - ETA: 42s - loss: 7.6787 - accuracy: 0.4992
12736/25000 [==============>...............] - ETA: 41s - loss: 7.6702 - accuracy: 0.4998
12768/25000 [==============>...............] - ETA: 41s - loss: 7.6714 - accuracy: 0.4997
12800/25000 [==============>...............] - ETA: 41s - loss: 7.6714 - accuracy: 0.4997
12832/25000 [==============>...............] - ETA: 41s - loss: 7.6726 - accuracy: 0.4996
12864/25000 [==============>...............] - ETA: 41s - loss: 7.6726 - accuracy: 0.4996
12896/25000 [==============>...............] - ETA: 41s - loss: 7.6714 - accuracy: 0.4997
12928/25000 [==============>...............] - ETA: 41s - loss: 7.6737 - accuracy: 0.4995
12960/25000 [==============>...............] - ETA: 41s - loss: 7.6714 - accuracy: 0.4997
12992/25000 [==============>...............] - ETA: 41s - loss: 7.6737 - accuracy: 0.4995
13024/25000 [==============>...............] - ETA: 40s - loss: 7.6725 - accuracy: 0.4996
13056/25000 [==============>...............] - ETA: 40s - loss: 7.6748 - accuracy: 0.4995
13088/25000 [==============>...............] - ETA: 40s - loss: 7.6701 - accuracy: 0.4998
13120/25000 [==============>...............] - ETA: 40s - loss: 7.6678 - accuracy: 0.4999
13152/25000 [==============>...............] - ETA: 40s - loss: 7.6655 - accuracy: 0.5001
13184/25000 [==============>...............] - ETA: 40s - loss: 7.6655 - accuracy: 0.5001
13216/25000 [==============>...............] - ETA: 40s - loss: 7.6689 - accuracy: 0.4998
13248/25000 [==============>...............] - ETA: 40s - loss: 7.6724 - accuracy: 0.4996
13280/25000 [==============>...............] - ETA: 40s - loss: 7.6655 - accuracy: 0.5001
13312/25000 [==============>...............] - ETA: 39s - loss: 7.6632 - accuracy: 0.5002
13344/25000 [===============>..............] - ETA: 39s - loss: 7.6678 - accuracy: 0.4999
13376/25000 [===============>..............] - ETA: 39s - loss: 7.6678 - accuracy: 0.4999
13408/25000 [===============>..............] - ETA: 39s - loss: 7.6632 - accuracy: 0.5002
13440/25000 [===============>..............] - ETA: 39s - loss: 7.6598 - accuracy: 0.5004
13472/25000 [===============>..............] - ETA: 39s - loss: 7.6655 - accuracy: 0.5001
13504/25000 [===============>..............] - ETA: 39s - loss: 7.6689 - accuracy: 0.4999
13536/25000 [===============>..............] - ETA: 39s - loss: 7.6655 - accuracy: 0.5001
13568/25000 [===============>..............] - ETA: 38s - loss: 7.6655 - accuracy: 0.5001
13600/25000 [===============>..............] - ETA: 38s - loss: 7.6700 - accuracy: 0.4998
13632/25000 [===============>..............] - ETA: 38s - loss: 7.6700 - accuracy: 0.4998
13664/25000 [===============>..............] - ETA: 38s - loss: 7.6734 - accuracy: 0.4996
13696/25000 [===============>..............] - ETA: 38s - loss: 7.6745 - accuracy: 0.4995
13728/25000 [===============>..............] - ETA: 38s - loss: 7.6789 - accuracy: 0.4992
13760/25000 [===============>..............] - ETA: 38s - loss: 7.6811 - accuracy: 0.4991
13792/25000 [===============>..............] - ETA: 38s - loss: 7.6766 - accuracy: 0.4993
13824/25000 [===============>..............] - ETA: 38s - loss: 7.6799 - accuracy: 0.4991
13856/25000 [===============>..............] - ETA: 38s - loss: 7.6755 - accuracy: 0.4994
13888/25000 [===============>..............] - ETA: 37s - loss: 7.6766 - accuracy: 0.4994
13920/25000 [===============>..............] - ETA: 37s - loss: 7.6776 - accuracy: 0.4993
13952/25000 [===============>..............] - ETA: 37s - loss: 7.6765 - accuracy: 0.4994
13984/25000 [===============>..............] - ETA: 37s - loss: 7.6732 - accuracy: 0.4996
14016/25000 [===============>..............] - ETA: 37s - loss: 7.6699 - accuracy: 0.4998
14048/25000 [===============>..............] - ETA: 37s - loss: 7.6633 - accuracy: 0.5002
14080/25000 [===============>..............] - ETA: 37s - loss: 7.6655 - accuracy: 0.5001
14112/25000 [===============>..............] - ETA: 37s - loss: 7.6677 - accuracy: 0.4999
14144/25000 [===============>..............] - ETA: 36s - loss: 7.6666 - accuracy: 0.5000
14176/25000 [================>.............] - ETA: 36s - loss: 7.6645 - accuracy: 0.5001
14208/25000 [================>.............] - ETA: 36s - loss: 7.6591 - accuracy: 0.5005
14240/25000 [================>.............] - ETA: 36s - loss: 7.6655 - accuracy: 0.5001
14272/25000 [================>.............] - ETA: 36s - loss: 7.6677 - accuracy: 0.4999
14304/25000 [================>.............] - ETA: 36s - loss: 7.6688 - accuracy: 0.4999
14336/25000 [================>.............] - ETA: 36s - loss: 7.6698 - accuracy: 0.4998
14368/25000 [================>.............] - ETA: 36s - loss: 7.6698 - accuracy: 0.4998
14400/25000 [================>.............] - ETA: 36s - loss: 7.6709 - accuracy: 0.4997
14432/25000 [================>.............] - ETA: 36s - loss: 7.6677 - accuracy: 0.4999
14464/25000 [================>.............] - ETA: 35s - loss: 7.6719 - accuracy: 0.4997
14496/25000 [================>.............] - ETA: 35s - loss: 7.6719 - accuracy: 0.4997
14528/25000 [================>.............] - ETA: 35s - loss: 7.6740 - accuracy: 0.4995
14560/25000 [================>.............] - ETA: 35s - loss: 7.6761 - accuracy: 0.4994
14592/25000 [================>.............] - ETA: 35s - loss: 7.6834 - accuracy: 0.4989
14624/25000 [================>.............] - ETA: 35s - loss: 7.6886 - accuracy: 0.4986
14656/25000 [================>.............] - ETA: 35s - loss: 7.6875 - accuracy: 0.4986
14688/25000 [================>.............] - ETA: 35s - loss: 7.6896 - accuracy: 0.4985
14720/25000 [================>.............] - ETA: 35s - loss: 7.6875 - accuracy: 0.4986
14752/25000 [================>.............] - ETA: 34s - loss: 7.6864 - accuracy: 0.4987
14784/25000 [================>.............] - ETA: 34s - loss: 7.6884 - accuracy: 0.4986
14816/25000 [================>.............] - ETA: 34s - loss: 7.6863 - accuracy: 0.4987
14848/25000 [================>.............] - ETA: 34s - loss: 7.6831 - accuracy: 0.4989
14880/25000 [================>.............] - ETA: 34s - loss: 7.6831 - accuracy: 0.4989
14912/25000 [================>.............] - ETA: 34s - loss: 7.6820 - accuracy: 0.4990
14944/25000 [================>.............] - ETA: 34s - loss: 7.6820 - accuracy: 0.4990
14976/25000 [================>.............] - ETA: 34s - loss: 7.6861 - accuracy: 0.4987
15008/25000 [=================>............] - ETA: 34s - loss: 7.6809 - accuracy: 0.4991
15040/25000 [=================>............] - ETA: 33s - loss: 7.6850 - accuracy: 0.4988
15072/25000 [=================>............] - ETA: 33s - loss: 7.6839 - accuracy: 0.4989
15104/25000 [=================>............] - ETA: 33s - loss: 7.6859 - accuracy: 0.4987
15136/25000 [=================>............] - ETA: 33s - loss: 7.6838 - accuracy: 0.4989
15168/25000 [=================>............] - ETA: 33s - loss: 7.6818 - accuracy: 0.4990
15200/25000 [=================>............] - ETA: 33s - loss: 7.6868 - accuracy: 0.4987
15232/25000 [=================>............] - ETA: 33s - loss: 7.6878 - accuracy: 0.4986
15264/25000 [=================>............] - ETA: 33s - loss: 7.6817 - accuracy: 0.4990
15296/25000 [=================>............] - ETA: 33s - loss: 7.6766 - accuracy: 0.4993
15328/25000 [=================>............] - ETA: 32s - loss: 7.6776 - accuracy: 0.4993
15360/25000 [=================>............] - ETA: 32s - loss: 7.6766 - accuracy: 0.4993
15392/25000 [=================>............] - ETA: 32s - loss: 7.6746 - accuracy: 0.4995
15424/25000 [=================>............] - ETA: 32s - loss: 7.6716 - accuracy: 0.4997
15456/25000 [=================>............] - ETA: 32s - loss: 7.6746 - accuracy: 0.4995
15488/25000 [=================>............] - ETA: 32s - loss: 7.6775 - accuracy: 0.4993
15520/25000 [=================>............] - ETA: 32s - loss: 7.6775 - accuracy: 0.4993
15552/25000 [=================>............] - ETA: 32s - loss: 7.6775 - accuracy: 0.4993
15584/25000 [=================>............] - ETA: 32s - loss: 7.6804 - accuracy: 0.4991
15616/25000 [=================>............] - ETA: 32s - loss: 7.6823 - accuracy: 0.4990
15648/25000 [=================>............] - ETA: 31s - loss: 7.6843 - accuracy: 0.4988
15680/25000 [=================>............] - ETA: 31s - loss: 7.6832 - accuracy: 0.4989
15712/25000 [=================>............] - ETA: 31s - loss: 7.6842 - accuracy: 0.4989
15744/25000 [=================>............] - ETA: 31s - loss: 7.6861 - accuracy: 0.4987
15776/25000 [=================>............] - ETA: 31s - loss: 7.6880 - accuracy: 0.4986
15808/25000 [=================>............] - ETA: 31s - loss: 7.6870 - accuracy: 0.4987
15840/25000 [==================>...........] - ETA: 31s - loss: 7.6869 - accuracy: 0.4987
15872/25000 [==================>...........] - ETA: 31s - loss: 7.6859 - accuracy: 0.4987
15904/25000 [==================>...........] - ETA: 30s - loss: 7.6840 - accuracy: 0.4989
15936/25000 [==================>...........] - ETA: 30s - loss: 7.6801 - accuracy: 0.4991
15968/25000 [==================>...........] - ETA: 30s - loss: 7.6791 - accuracy: 0.4992
16000/25000 [==================>...........] - ETA: 30s - loss: 7.6772 - accuracy: 0.4993
16032/25000 [==================>...........] - ETA: 30s - loss: 7.6800 - accuracy: 0.4991
16064/25000 [==================>...........] - ETA: 30s - loss: 7.6809 - accuracy: 0.4991
16096/25000 [==================>...........] - ETA: 30s - loss: 7.6771 - accuracy: 0.4993
16128/25000 [==================>...........] - ETA: 30s - loss: 7.6771 - accuracy: 0.4993
16160/25000 [==================>...........] - ETA: 30s - loss: 7.6761 - accuracy: 0.4994
16192/25000 [==================>...........] - ETA: 30s - loss: 7.6799 - accuracy: 0.4991
16224/25000 [==================>...........] - ETA: 29s - loss: 7.6770 - accuracy: 0.4993
16256/25000 [==================>...........] - ETA: 29s - loss: 7.6779 - accuracy: 0.4993
16288/25000 [==================>...........] - ETA: 29s - loss: 7.6742 - accuracy: 0.4995
16320/25000 [==================>...........] - ETA: 29s - loss: 7.6770 - accuracy: 0.4993
16352/25000 [==================>...........] - ETA: 29s - loss: 7.6741 - accuracy: 0.4995
16384/25000 [==================>...........] - ETA: 29s - loss: 7.6750 - accuracy: 0.4995
16416/25000 [==================>...........] - ETA: 29s - loss: 7.6769 - accuracy: 0.4993
16448/25000 [==================>...........] - ETA: 29s - loss: 7.6787 - accuracy: 0.4992
16480/25000 [==================>...........] - ETA: 29s - loss: 7.6769 - accuracy: 0.4993
16512/25000 [==================>...........] - ETA: 28s - loss: 7.6796 - accuracy: 0.4992
16544/25000 [==================>...........] - ETA: 28s - loss: 7.6796 - accuracy: 0.4992
16576/25000 [==================>...........] - ETA: 28s - loss: 7.6759 - accuracy: 0.4994
16608/25000 [==================>...........] - ETA: 28s - loss: 7.6740 - accuracy: 0.4995
16640/25000 [==================>...........] - ETA: 28s - loss: 7.6685 - accuracy: 0.4999
16672/25000 [===================>..........] - ETA: 28s - loss: 7.6685 - accuracy: 0.4999
16704/25000 [===================>..........] - ETA: 28s - loss: 7.6666 - accuracy: 0.5000
16736/25000 [===================>..........] - ETA: 28s - loss: 7.6611 - accuracy: 0.5004
16768/25000 [===================>..........] - ETA: 28s - loss: 7.6630 - accuracy: 0.5002
16800/25000 [===================>..........] - ETA: 27s - loss: 7.6630 - accuracy: 0.5002
16832/25000 [===================>..........] - ETA: 27s - loss: 7.6621 - accuracy: 0.5003
16864/25000 [===================>..........] - ETA: 27s - loss: 7.6639 - accuracy: 0.5002
16896/25000 [===================>..........] - ETA: 27s - loss: 7.6648 - accuracy: 0.5001
16928/25000 [===================>..........] - ETA: 27s - loss: 7.6657 - accuracy: 0.5001
16960/25000 [===================>..........] - ETA: 27s - loss: 7.6675 - accuracy: 0.4999
16992/25000 [===================>..........] - ETA: 27s - loss: 7.6648 - accuracy: 0.5001
17024/25000 [===================>..........] - ETA: 27s - loss: 7.6612 - accuracy: 0.5004
17056/25000 [===================>..........] - ETA: 27s - loss: 7.6612 - accuracy: 0.5004
17088/25000 [===================>..........] - ETA: 26s - loss: 7.6657 - accuracy: 0.5001
17120/25000 [===================>..........] - ETA: 26s - loss: 7.6675 - accuracy: 0.4999
17152/25000 [===================>..........] - ETA: 26s - loss: 7.6684 - accuracy: 0.4999
17184/25000 [===================>..........] - ETA: 26s - loss: 7.6702 - accuracy: 0.4998
17216/25000 [===================>..........] - ETA: 26s - loss: 7.6755 - accuracy: 0.4994
17248/25000 [===================>..........] - ETA: 26s - loss: 7.6764 - accuracy: 0.4994
17280/25000 [===================>..........] - ETA: 26s - loss: 7.6764 - accuracy: 0.4994
17312/25000 [===================>..........] - ETA: 26s - loss: 7.6764 - accuracy: 0.4994
17344/25000 [===================>..........] - ETA: 26s - loss: 7.6755 - accuracy: 0.4994
17376/25000 [===================>..........] - ETA: 25s - loss: 7.6772 - accuracy: 0.4993
17408/25000 [===================>..........] - ETA: 25s - loss: 7.6807 - accuracy: 0.4991
17440/25000 [===================>..........] - ETA: 25s - loss: 7.6824 - accuracy: 0.4990
17472/25000 [===================>..........] - ETA: 25s - loss: 7.6798 - accuracy: 0.4991
17504/25000 [====================>.........] - ETA: 25s - loss: 7.6789 - accuracy: 0.4992
17536/25000 [====================>.........] - ETA: 25s - loss: 7.6815 - accuracy: 0.4990
17568/25000 [====================>.........] - ETA: 25s - loss: 7.6788 - accuracy: 0.4992
17600/25000 [====================>.........] - ETA: 25s - loss: 7.6779 - accuracy: 0.4993
17632/25000 [====================>.........] - ETA: 25s - loss: 7.6771 - accuracy: 0.4993
17664/25000 [====================>.........] - ETA: 24s - loss: 7.6736 - accuracy: 0.4995
17696/25000 [====================>.........] - ETA: 24s - loss: 7.6727 - accuracy: 0.4996
17728/25000 [====================>.........] - ETA: 24s - loss: 7.6744 - accuracy: 0.4995
17760/25000 [====================>.........] - ETA: 24s - loss: 7.6761 - accuracy: 0.4994
17792/25000 [====================>.........] - ETA: 24s - loss: 7.6761 - accuracy: 0.4994
17824/25000 [====================>.........] - ETA: 24s - loss: 7.6752 - accuracy: 0.4994
17856/25000 [====================>.........] - ETA: 24s - loss: 7.6761 - accuracy: 0.4994
17888/25000 [====================>.........] - ETA: 24s - loss: 7.6760 - accuracy: 0.4994
17920/25000 [====================>.........] - ETA: 24s - loss: 7.6769 - accuracy: 0.4993
17952/25000 [====================>.........] - ETA: 23s - loss: 7.6743 - accuracy: 0.4995
17984/25000 [====================>.........] - ETA: 23s - loss: 7.6700 - accuracy: 0.4998
18016/25000 [====================>.........] - ETA: 23s - loss: 7.6692 - accuracy: 0.4998
18048/25000 [====================>.........] - ETA: 23s - loss: 7.6709 - accuracy: 0.4997
18080/25000 [====================>.........] - ETA: 23s - loss: 7.6709 - accuracy: 0.4997
18112/25000 [====================>.........] - ETA: 23s - loss: 7.6709 - accuracy: 0.4997
18144/25000 [====================>.........] - ETA: 23s - loss: 7.6734 - accuracy: 0.4996
18176/25000 [====================>.........] - ETA: 23s - loss: 7.6725 - accuracy: 0.4996
18208/25000 [====================>.........] - ETA: 23s - loss: 7.6700 - accuracy: 0.4998
18240/25000 [====================>.........] - ETA: 23s - loss: 7.6683 - accuracy: 0.4999
18272/25000 [====================>.........] - ETA: 22s - loss: 7.6700 - accuracy: 0.4998
18304/25000 [====================>.........] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
18336/25000 [=====================>........] - ETA: 22s - loss: 7.6691 - accuracy: 0.4998
18368/25000 [=====================>........] - ETA: 22s - loss: 7.6691 - accuracy: 0.4998
18400/25000 [=====================>........] - ETA: 22s - loss: 7.6700 - accuracy: 0.4998
18432/25000 [=====================>........] - ETA: 22s - loss: 7.6708 - accuracy: 0.4997
18464/25000 [=====================>........] - ETA: 22s - loss: 7.6666 - accuracy: 0.5000
18496/25000 [=====================>........] - ETA: 22s - loss: 7.6658 - accuracy: 0.5001
18528/25000 [=====================>........] - ETA: 22s - loss: 7.6708 - accuracy: 0.4997
18560/25000 [=====================>........] - ETA: 21s - loss: 7.6699 - accuracy: 0.4998
18592/25000 [=====================>........] - ETA: 21s - loss: 7.6699 - accuracy: 0.4998
18624/25000 [=====================>........] - ETA: 21s - loss: 7.6716 - accuracy: 0.4997
18656/25000 [=====================>........] - ETA: 21s - loss: 7.6716 - accuracy: 0.4997
18688/25000 [=====================>........] - ETA: 21s - loss: 7.6691 - accuracy: 0.4998
18720/25000 [=====================>........] - ETA: 21s - loss: 7.6724 - accuracy: 0.4996
18752/25000 [=====================>........] - ETA: 21s - loss: 7.6699 - accuracy: 0.4998
18784/25000 [=====================>........] - ETA: 21s - loss: 7.6658 - accuracy: 0.5001
18816/25000 [=====================>........] - ETA: 21s - loss: 7.6650 - accuracy: 0.5001
18848/25000 [=====================>........] - ETA: 20s - loss: 7.6650 - accuracy: 0.5001
18880/25000 [=====================>........] - ETA: 20s - loss: 7.6642 - accuracy: 0.5002
18912/25000 [=====================>........] - ETA: 20s - loss: 7.6642 - accuracy: 0.5002
18944/25000 [=====================>........] - ETA: 20s - loss: 7.6650 - accuracy: 0.5001
18976/25000 [=====================>........] - ETA: 20s - loss: 7.6626 - accuracy: 0.5003
19008/25000 [=====================>........] - ETA: 20s - loss: 7.6610 - accuracy: 0.5004
19040/25000 [=====================>........] - ETA: 20s - loss: 7.6642 - accuracy: 0.5002
19072/25000 [=====================>........] - ETA: 20s - loss: 7.6626 - accuracy: 0.5003
19104/25000 [=====================>........] - ETA: 20s - loss: 7.6634 - accuracy: 0.5002
19136/25000 [=====================>........] - ETA: 19s - loss: 7.6666 - accuracy: 0.5000
19168/25000 [======================>.......] - ETA: 19s - loss: 7.6658 - accuracy: 0.5001
19200/25000 [======================>.......] - ETA: 19s - loss: 7.6610 - accuracy: 0.5004
19232/25000 [======================>.......] - ETA: 19s - loss: 7.6610 - accuracy: 0.5004
19264/25000 [======================>.......] - ETA: 19s - loss: 7.6595 - accuracy: 0.5005
19296/25000 [======================>.......] - ETA: 19s - loss: 7.6603 - accuracy: 0.5004
19328/25000 [======================>.......] - ETA: 19s - loss: 7.6595 - accuracy: 0.5005
19360/25000 [======================>.......] - ETA: 19s - loss: 7.6587 - accuracy: 0.5005
19392/25000 [======================>.......] - ETA: 19s - loss: 7.6595 - accuracy: 0.5005
19424/25000 [======================>.......] - ETA: 18s - loss: 7.6619 - accuracy: 0.5003
19456/25000 [======================>.......] - ETA: 18s - loss: 7.6627 - accuracy: 0.5003
19488/25000 [======================>.......] - ETA: 18s - loss: 7.6603 - accuracy: 0.5004
19520/25000 [======================>.......] - ETA: 18s - loss: 7.6627 - accuracy: 0.5003
19552/25000 [======================>.......] - ETA: 18s - loss: 7.6619 - accuracy: 0.5003
19584/25000 [======================>.......] - ETA: 18s - loss: 7.6635 - accuracy: 0.5002
19616/25000 [======================>.......] - ETA: 18s - loss: 7.6635 - accuracy: 0.5002
19648/25000 [======================>.......] - ETA: 18s - loss: 7.6627 - accuracy: 0.5003
19680/25000 [======================>.......] - ETA: 18s - loss: 7.6643 - accuracy: 0.5002
19712/25000 [======================>.......] - ETA: 18s - loss: 7.6635 - accuracy: 0.5002
19744/25000 [======================>.......] - ETA: 17s - loss: 7.6635 - accuracy: 0.5002
19776/25000 [======================>.......] - ETA: 17s - loss: 7.6604 - accuracy: 0.5004
19808/25000 [======================>.......] - ETA: 17s - loss: 7.6627 - accuracy: 0.5003
19840/25000 [======================>.......] - ETA: 17s - loss: 7.6635 - accuracy: 0.5002
19872/25000 [======================>.......] - ETA: 17s - loss: 7.6651 - accuracy: 0.5001
19904/25000 [======================>.......] - ETA: 17s - loss: 7.6651 - accuracy: 0.5001
19936/25000 [======================>.......] - ETA: 17s - loss: 7.6712 - accuracy: 0.4997
19968/25000 [======================>.......] - ETA: 17s - loss: 7.6712 - accuracy: 0.4997
20000/25000 [=======================>......] - ETA: 17s - loss: 7.6705 - accuracy: 0.4997
20032/25000 [=======================>......] - ETA: 16s - loss: 7.6697 - accuracy: 0.4998
20064/25000 [=======================>......] - ETA: 16s - loss: 7.6712 - accuracy: 0.4997
20096/25000 [=======================>......] - ETA: 16s - loss: 7.6712 - accuracy: 0.4997
20128/25000 [=======================>......] - ETA: 16s - loss: 7.6704 - accuracy: 0.4998
20160/25000 [=======================>......] - ETA: 16s - loss: 7.6727 - accuracy: 0.4996
20192/25000 [=======================>......] - ETA: 16s - loss: 7.6757 - accuracy: 0.4994
20224/25000 [=======================>......] - ETA: 16s - loss: 7.6750 - accuracy: 0.4995
20256/25000 [=======================>......] - ETA: 16s - loss: 7.6749 - accuracy: 0.4995
20288/25000 [=======================>......] - ETA: 16s - loss: 7.6772 - accuracy: 0.4993
20320/25000 [=======================>......] - ETA: 15s - loss: 7.6772 - accuracy: 0.4993
20352/25000 [=======================>......] - ETA: 15s - loss: 7.6764 - accuracy: 0.4994
20384/25000 [=======================>......] - ETA: 15s - loss: 7.6726 - accuracy: 0.4996
20416/25000 [=======================>......] - ETA: 15s - loss: 7.6719 - accuracy: 0.4997
20448/25000 [=======================>......] - ETA: 15s - loss: 7.6719 - accuracy: 0.4997
20480/25000 [=======================>......] - ETA: 15s - loss: 7.6756 - accuracy: 0.4994
20512/25000 [=======================>......] - ETA: 15s - loss: 7.6748 - accuracy: 0.4995
20544/25000 [=======================>......] - ETA: 15s - loss: 7.6711 - accuracy: 0.4997
20576/25000 [=======================>......] - ETA: 15s - loss: 7.6718 - accuracy: 0.4997
20608/25000 [=======================>......] - ETA: 14s - loss: 7.6703 - accuracy: 0.4998
20640/25000 [=======================>......] - ETA: 14s - loss: 7.6726 - accuracy: 0.4996
20672/25000 [=======================>......] - ETA: 14s - loss: 7.6711 - accuracy: 0.4997
20704/25000 [=======================>......] - ETA: 14s - loss: 7.6681 - accuracy: 0.4999
20736/25000 [=======================>......] - ETA: 14s - loss: 7.6637 - accuracy: 0.5002
20768/25000 [=======================>......] - ETA: 14s - loss: 7.6659 - accuracy: 0.5000
20800/25000 [=======================>......] - ETA: 14s - loss: 7.6688 - accuracy: 0.4999
20832/25000 [=======================>......] - ETA: 14s - loss: 7.6688 - accuracy: 0.4999
20864/25000 [========================>.....] - ETA: 14s - loss: 7.6644 - accuracy: 0.5001
20896/25000 [========================>.....] - ETA: 13s - loss: 7.6622 - accuracy: 0.5003
20928/25000 [========================>.....] - ETA: 13s - loss: 7.6637 - accuracy: 0.5002
20960/25000 [========================>.....] - ETA: 13s - loss: 7.6622 - accuracy: 0.5003
20992/25000 [========================>.....] - ETA: 13s - loss: 7.6637 - accuracy: 0.5002
21024/25000 [========================>.....] - ETA: 13s - loss: 7.6630 - accuracy: 0.5002
21056/25000 [========================>.....] - ETA: 13s - loss: 7.6644 - accuracy: 0.5001
21088/25000 [========================>.....] - ETA: 13s - loss: 7.6608 - accuracy: 0.5004
21120/25000 [========================>.....] - ETA: 13s - loss: 7.6615 - accuracy: 0.5003
21152/25000 [========================>.....] - ETA: 13s - loss: 7.6623 - accuracy: 0.5003
21184/25000 [========================>.....] - ETA: 12s - loss: 7.6623 - accuracy: 0.5003
21216/25000 [========================>.....] - ETA: 12s - loss: 7.6637 - accuracy: 0.5002
21248/25000 [========================>.....] - ETA: 12s - loss: 7.6652 - accuracy: 0.5001
21280/25000 [========================>.....] - ETA: 12s - loss: 7.6681 - accuracy: 0.4999
21312/25000 [========================>.....] - ETA: 12s - loss: 7.6673 - accuracy: 0.5000
21344/25000 [========================>.....] - ETA: 12s - loss: 7.6688 - accuracy: 0.4999
21376/25000 [========================>.....] - ETA: 12s - loss: 7.6688 - accuracy: 0.4999
21408/25000 [========================>.....] - ETA: 12s - loss: 7.6666 - accuracy: 0.5000
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6680 - accuracy: 0.4999
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6673 - accuracy: 0.5000
21504/25000 [========================>.....] - ETA: 11s - loss: 7.6645 - accuracy: 0.5001
21536/25000 [========================>.....] - ETA: 11s - loss: 7.6652 - accuracy: 0.5001
21568/25000 [========================>.....] - ETA: 11s - loss: 7.6680 - accuracy: 0.4999
21600/25000 [========================>.....] - ETA: 11s - loss: 7.6659 - accuracy: 0.5000
21632/25000 [========================>.....] - ETA: 11s - loss: 7.6624 - accuracy: 0.5003
21664/25000 [========================>.....] - ETA: 11s - loss: 7.6638 - accuracy: 0.5002
21696/25000 [=========================>....] - ETA: 11s - loss: 7.6631 - accuracy: 0.5002
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6603 - accuracy: 0.5004
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6596 - accuracy: 0.5005
21792/25000 [=========================>....] - ETA: 10s - loss: 7.6568 - accuracy: 0.5006
21824/25000 [=========================>....] - ETA: 10s - loss: 7.6582 - accuracy: 0.5005
21856/25000 [=========================>....] - ETA: 10s - loss: 7.6568 - accuracy: 0.5006
21888/25000 [=========================>....] - ETA: 10s - loss: 7.6568 - accuracy: 0.5006
21920/25000 [=========================>....] - ETA: 10s - loss: 7.6568 - accuracy: 0.5006
21952/25000 [=========================>....] - ETA: 10s - loss: 7.6582 - accuracy: 0.5005
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6610 - accuracy: 0.5004
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6569 - accuracy: 0.5006
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6569 - accuracy: 0.5006
22080/25000 [=========================>....] - ETA: 9s - loss: 7.6555 - accuracy: 0.5007 
22112/25000 [=========================>....] - ETA: 9s - loss: 7.6555 - accuracy: 0.5007
22144/25000 [=========================>....] - ETA: 9s - loss: 7.6569 - accuracy: 0.5006
22176/25000 [=========================>....] - ETA: 9s - loss: 7.6562 - accuracy: 0.5007
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6535 - accuracy: 0.5009
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6508 - accuracy: 0.5010
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6501 - accuracy: 0.5011
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6522 - accuracy: 0.5009
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6515 - accuracy: 0.5010
22368/25000 [=========================>....] - ETA: 8s - loss: 7.6515 - accuracy: 0.5010
22400/25000 [=========================>....] - ETA: 8s - loss: 7.6481 - accuracy: 0.5012
22432/25000 [=========================>....] - ETA: 8s - loss: 7.6454 - accuracy: 0.5014
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6448 - accuracy: 0.5014
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6421 - accuracy: 0.5016
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6435 - accuracy: 0.5015
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6442 - accuracy: 0.5015
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6442 - accuracy: 0.5015
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6449 - accuracy: 0.5014
22656/25000 [==========================>...] - ETA: 7s - loss: 7.6463 - accuracy: 0.5013
22688/25000 [==========================>...] - ETA: 7s - loss: 7.6470 - accuracy: 0.5013
22720/25000 [==========================>...] - ETA: 7s - loss: 7.6450 - accuracy: 0.5014
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6471 - accuracy: 0.5013
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6484 - accuracy: 0.5012
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6505 - accuracy: 0.5011
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6498 - accuracy: 0.5011
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6525 - accuracy: 0.5009
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6573 - accuracy: 0.5006
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6586 - accuracy: 0.5005
22976/25000 [==========================>...] - ETA: 6s - loss: 7.6593 - accuracy: 0.5005
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6586 - accuracy: 0.5005
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6586 - accuracy: 0.5005
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6593 - accuracy: 0.5005
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6613 - accuracy: 0.5003
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6633 - accuracy: 0.5002
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6626 - accuracy: 0.5003
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6640 - accuracy: 0.5002
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6633 - accuracy: 0.5002
23264/25000 [==========================>...] - ETA: 5s - loss: 7.6646 - accuracy: 0.5001
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6646 - accuracy: 0.5001
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6640 - accuracy: 0.5002
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6633 - accuracy: 0.5002
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6588 - accuracy: 0.5005
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6594 - accuracy: 0.5005
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6588 - accuracy: 0.5005
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6601 - accuracy: 0.5004
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6608 - accuracy: 0.5004
23552/25000 [===========================>..] - ETA: 4s - loss: 7.6614 - accuracy: 0.5003
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6575 - accuracy: 0.5006
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6582 - accuracy: 0.5006
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6575 - accuracy: 0.5006
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6543 - accuracy: 0.5008
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6524 - accuracy: 0.5009
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6518 - accuracy: 0.5010
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6518 - accuracy: 0.5010
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6544 - accuracy: 0.5008
23840/25000 [===========================>..] - ETA: 3s - loss: 7.6544 - accuracy: 0.5008
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6563 - accuracy: 0.5007
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6557 - accuracy: 0.5007
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6551 - accuracy: 0.5008
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6557 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6577 - accuracy: 0.5006
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6577 - accuracy: 0.5006
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6564 - accuracy: 0.5007
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6571 - accuracy: 0.5006
24128/25000 [===========================>..] - ETA: 2s - loss: 7.6545 - accuracy: 0.5008
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6571 - accuracy: 0.5006
24192/25000 [============================>.] - ETA: 2s - loss: 7.6590 - accuracy: 0.5005
24224/25000 [============================>.] - ETA: 2s - loss: 7.6571 - accuracy: 0.5006
24256/25000 [============================>.] - ETA: 2s - loss: 7.6578 - accuracy: 0.5006
24288/25000 [============================>.] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
24320/25000 [============================>.] - ETA: 2s - loss: 7.6591 - accuracy: 0.5005
24352/25000 [============================>.] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
24384/25000 [============================>.] - ETA: 2s - loss: 7.6566 - accuracy: 0.5007
24416/25000 [============================>.] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
24448/25000 [============================>.] - ETA: 1s - loss: 7.6578 - accuracy: 0.5006
24480/25000 [============================>.] - ETA: 1s - loss: 7.6597 - accuracy: 0.5004
24512/25000 [============================>.] - ETA: 1s - loss: 7.6616 - accuracy: 0.5003
24544/25000 [============================>.] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
24576/25000 [============================>.] - ETA: 1s - loss: 7.6579 - accuracy: 0.5006
24608/25000 [============================>.] - ETA: 1s - loss: 7.6573 - accuracy: 0.5006
24640/25000 [============================>.] - ETA: 1s - loss: 7.6598 - accuracy: 0.5004
24672/25000 [============================>.] - ETA: 1s - loss: 7.6585 - accuracy: 0.5005
24704/25000 [============================>.] - ETA: 1s - loss: 7.6586 - accuracy: 0.5005
24736/25000 [============================>.] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24768/25000 [============================>.] - ETA: 0s - loss: 7.6592 - accuracy: 0.5005
24800/25000 [============================>.] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
24832/25000 [============================>.] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
24864/25000 [============================>.] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
24896/25000 [============================>.] - ETA: 0s - loss: 7.6629 - accuracy: 0.5002
24928/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24960/25000 [============================>.] - ETA: 0s - loss: 7.6635 - accuracy: 0.5002
24992/25000 [============================>.] - ETA: 0s - loss: 7.6672 - accuracy: 0.5000
25000/25000 [==============================] - 103s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

