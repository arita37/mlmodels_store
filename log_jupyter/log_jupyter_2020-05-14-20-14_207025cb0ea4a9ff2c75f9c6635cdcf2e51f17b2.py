
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
 40%|████      | 2/5 [00:55<01:22, 27.61s/it] 40%|████      | 2/5 [00:55<01:22, 27.61s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.4096237576526039, 'embedding_size_factor': 1.3455072053268782, 'layers.choice': 1, 'learning_rate': 0.0042508481043035075, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 1.8245142212969275e-06} and reward: 0.3746
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xda7F\x90\xb2\x15\x85X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x872\x906\x94\xaeX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?qiVYiu\xdbX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xbe\x9c:\x99\xf2\xa3\xe5u.' and reward: 0.3746
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xda7F\x90\xb2\x15\x85X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf5\x872\x906\x94\xaeX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?qiVYiu\xdbX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xbe\x9c:\x99\xf2\xa3\xe5u.' and reward: 0.3746
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 176.1499674320221
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/NeuralNetClassifier/trial_0_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_1_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.73s of the -58.37s of remaining time.
Ensemble size: 29
Ensemble weights: 
[0.75862069 0.24137931]
	0.3878	 = Validation accuracy score
	0.96s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 179.38s ...
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

  <mlmodels.model_tf.1_lstm.Model object at 0x7f6c64e74a90> 

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
 [-0.03540968  0.09678224 -0.15583111  0.021963    0.05037852  0.04582785]
 [ 0.08009101  0.24925107 -0.06184479 -0.09286365  0.11177454 -0.02815155]
 [ 0.08602928  0.02003477 -0.01821057 -0.02572824  0.0141399  -0.03973571]
 [ 0.09308508  0.06297942 -0.03505968  0.37021506  0.21439216 -0.23318273]
 [ 0.34000504 -0.0377175  -0.28935942  0.04137717  0.35854566 -0.04701618]
 [ 0.25840974  0.57732254 -0.11186349  0.32387981  0.02051971 -0.04402928]
 [ 0.06092831  0.00481592  0.32553187 -0.24114802  0.73796922 -0.3148922 ]
 [-0.26567155  0.07924023 -0.27031276  0.07612951  0.50008398 -0.4858233 ]
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
{'loss': 0.34888639859855175, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-14 20:17:51.209493: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
{'loss': 0.38263790495693684, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-14 20:17:52.431697: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
 2179072/17464789 [==>...........................] - ETA: 0s
 8380416/17464789 [=============>................] - ETA: 0s
16318464/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-14 20:18:04.726906: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-14 20:18:04.731315: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-14 20:18:04.731483: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555ee82597f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-14 20:18:04.731500: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 25000 samples, validate on 25000 samples
Epoch 1/1

   32/25000 [..............................] - ETA: 5:09 - loss: 10.5416 - accuracy: 0.3125
   64/25000 [..............................] - ETA: 3:18 - loss: 8.6249 - accuracy: 0.4375 
   96/25000 [..............................] - ETA: 2:43 - loss: 8.1458 - accuracy: 0.4688
  128/25000 [..............................] - ETA: 2:24 - loss: 8.0260 - accuracy: 0.4766
  160/25000 [..............................] - ETA: 2:12 - loss: 8.1458 - accuracy: 0.4688
  192/25000 [..............................] - ETA: 2:04 - loss: 8.1458 - accuracy: 0.4688
  224/25000 [..............................] - ETA: 1:59 - loss: 7.8035 - accuracy: 0.4911
  256/25000 [..............................] - ETA: 1:55 - loss: 7.6067 - accuracy: 0.5039
  288/25000 [..............................] - ETA: 1:53 - loss: 7.6134 - accuracy: 0.5035
  320/25000 [..............................] - ETA: 1:50 - loss: 7.8583 - accuracy: 0.4875
  352/25000 [..............................] - ETA: 1:47 - loss: 7.6666 - accuracy: 0.5000
  384/25000 [..............................] - ETA: 1:46 - loss: 7.7465 - accuracy: 0.4948
  416/25000 [..............................] - ETA: 1:44 - loss: 7.7035 - accuracy: 0.4976
  448/25000 [..............................] - ETA: 1:43 - loss: 7.8035 - accuracy: 0.4911
  480/25000 [..............................] - ETA: 1:42 - loss: 7.8902 - accuracy: 0.4854
  512/25000 [..............................] - ETA: 1:41 - loss: 7.9361 - accuracy: 0.4824
  544/25000 [..............................] - ETA: 1:40 - loss: 7.8921 - accuracy: 0.4853
  576/25000 [..............................] - ETA: 1:39 - loss: 8.0127 - accuracy: 0.4774
  608/25000 [..............................] - ETA: 1:38 - loss: 8.0953 - accuracy: 0.4720
  640/25000 [..............................] - ETA: 1:38 - loss: 8.1218 - accuracy: 0.4703
  672/25000 [..............................] - ETA: 1:37 - loss: 8.0317 - accuracy: 0.4762
  704/25000 [..............................] - ETA: 1:36 - loss: 7.9280 - accuracy: 0.4830
  736/25000 [..............................] - ETA: 1:35 - loss: 7.8958 - accuracy: 0.4851
  768/25000 [..............................] - ETA: 1:35 - loss: 7.9861 - accuracy: 0.4792
  800/25000 [..............................] - ETA: 1:34 - loss: 7.9541 - accuracy: 0.4812
  832/25000 [..............................] - ETA: 1:34 - loss: 7.9062 - accuracy: 0.4844
  864/25000 [>.............................] - ETA: 1:34 - loss: 7.8973 - accuracy: 0.4850
  896/25000 [>.............................] - ETA: 1:33 - loss: 7.9233 - accuracy: 0.4833
  928/25000 [>.............................] - ETA: 1:33 - loss: 7.9475 - accuracy: 0.4817
  960/25000 [>.............................] - ETA: 1:32 - loss: 7.9222 - accuracy: 0.4833
  992/25000 [>.............................] - ETA: 1:32 - loss: 7.9448 - accuracy: 0.4819
 1024/25000 [>.............................] - ETA: 1:32 - loss: 7.9811 - accuracy: 0.4795
 1056/25000 [>.............................] - ETA: 1:31 - loss: 7.9425 - accuracy: 0.4820
 1088/25000 [>.............................] - ETA: 1:31 - loss: 7.9485 - accuracy: 0.4816
 1120/25000 [>.............................] - ETA: 1:30 - loss: 7.9130 - accuracy: 0.4839
 1152/25000 [>.............................] - ETA: 1:30 - loss: 7.8397 - accuracy: 0.4887
 1184/25000 [>.............................] - ETA: 1:29 - loss: 7.7832 - accuracy: 0.4924
 1216/25000 [>.............................] - ETA: 1:29 - loss: 7.8179 - accuracy: 0.4901
 1248/25000 [>.............................] - ETA: 1:29 - loss: 7.7526 - accuracy: 0.4944
 1280/25000 [>.............................] - ETA: 1:29 - loss: 7.7505 - accuracy: 0.4945
 1312/25000 [>.............................] - ETA: 1:29 - loss: 7.7251 - accuracy: 0.4962
 1344/25000 [>.............................] - ETA: 1:28 - loss: 7.7351 - accuracy: 0.4955
 1376/25000 [>.............................] - ETA: 1:28 - loss: 7.7335 - accuracy: 0.4956
 1408/25000 [>.............................] - ETA: 1:28 - loss: 7.7211 - accuracy: 0.4964
 1440/25000 [>.............................] - ETA: 1:28 - loss: 7.6560 - accuracy: 0.5007
 1472/25000 [>.............................] - ETA: 1:27 - loss: 7.6041 - accuracy: 0.5041
 1504/25000 [>.............................] - ETA: 1:27 - loss: 7.5647 - accuracy: 0.5066
 1536/25000 [>.............................] - ETA: 1:27 - loss: 7.5468 - accuracy: 0.5078
 1568/25000 [>.............................] - ETA: 1:27 - loss: 7.5591 - accuracy: 0.5070
 1600/25000 [>.............................] - ETA: 1:27 - loss: 7.5516 - accuracy: 0.5075
 1632/25000 [>.............................] - ETA: 1:27 - loss: 7.5633 - accuracy: 0.5067
 1664/25000 [>.............................] - ETA: 1:26 - loss: 7.5560 - accuracy: 0.5072
 1696/25000 [=>............................] - ETA: 1:26 - loss: 7.5039 - accuracy: 0.5106
 1728/25000 [=>............................] - ETA: 1:26 - loss: 7.4714 - accuracy: 0.5127
 1760/25000 [=>............................] - ETA: 1:26 - loss: 7.4750 - accuracy: 0.5125
 1792/25000 [=>............................] - ETA: 1:25 - loss: 7.4955 - accuracy: 0.5112
 1824/25000 [=>............................] - ETA: 1:25 - loss: 7.5237 - accuracy: 0.5093
 1856/25000 [=>............................] - ETA: 1:25 - loss: 7.5262 - accuracy: 0.5092
 1888/25000 [=>............................] - ETA: 1:25 - loss: 7.5123 - accuracy: 0.5101
 1920/25000 [=>............................] - ETA: 1:24 - loss: 7.5069 - accuracy: 0.5104
 1952/25000 [=>............................] - ETA: 1:24 - loss: 7.5095 - accuracy: 0.5102
 1984/25000 [=>............................] - ETA: 1:24 - loss: 7.4966 - accuracy: 0.5111
 2016/25000 [=>............................] - ETA: 1:24 - loss: 7.4841 - accuracy: 0.5119
 2048/25000 [=>............................] - ETA: 1:24 - loss: 7.5019 - accuracy: 0.5107
 2080/25000 [=>............................] - ETA: 1:23 - loss: 7.5266 - accuracy: 0.5091
 2112/25000 [=>............................] - ETA: 1:23 - loss: 7.5432 - accuracy: 0.5080
 2144/25000 [=>............................] - ETA: 1:23 - loss: 7.5450 - accuracy: 0.5079
 2176/25000 [=>............................] - ETA: 1:23 - loss: 7.5468 - accuracy: 0.5078
 2208/25000 [=>............................] - ETA: 1:23 - loss: 7.5833 - accuracy: 0.5054
 2240/25000 [=>............................] - ETA: 1:23 - loss: 7.5639 - accuracy: 0.5067
 2272/25000 [=>............................] - ETA: 1:23 - loss: 7.5451 - accuracy: 0.5079
 2304/25000 [=>............................] - ETA: 1:23 - loss: 7.5535 - accuracy: 0.5074
 2336/25000 [=>............................] - ETA: 1:22 - loss: 7.5616 - accuracy: 0.5068
 2368/25000 [=>............................] - ETA: 1:22 - loss: 7.5436 - accuracy: 0.5080
 2400/25000 [=>............................] - ETA: 1:22 - loss: 7.5452 - accuracy: 0.5079
 2432/25000 [=>............................] - ETA: 1:22 - loss: 7.5279 - accuracy: 0.5090
 2464/25000 [=>............................] - ETA: 1:22 - loss: 7.5173 - accuracy: 0.5097
 2496/25000 [=>............................] - ETA: 1:22 - loss: 7.5192 - accuracy: 0.5096
 2528/25000 [==>...........................] - ETA: 1:22 - loss: 7.5332 - accuracy: 0.5087
 2560/25000 [==>...........................] - ETA: 1:21 - loss: 7.5408 - accuracy: 0.5082
 2592/25000 [==>...........................] - ETA: 1:21 - loss: 7.5720 - accuracy: 0.5062
 2624/25000 [==>...........................] - ETA: 1:21 - loss: 7.5731 - accuracy: 0.5061
 2656/25000 [==>...........................] - ETA: 1:21 - loss: 7.5685 - accuracy: 0.5064
 2688/25000 [==>...........................] - ETA: 1:21 - loss: 7.5582 - accuracy: 0.5071
 2720/25000 [==>...........................] - ETA: 1:21 - loss: 7.5539 - accuracy: 0.5074
 2752/25000 [==>...........................] - ETA: 1:21 - loss: 7.5830 - accuracy: 0.5055
 2784/25000 [==>...........................] - ETA: 1:21 - loss: 7.5675 - accuracy: 0.5065
 2816/25000 [==>...........................] - ETA: 1:20 - loss: 7.5686 - accuracy: 0.5064
 2848/25000 [==>...........................] - ETA: 1:20 - loss: 7.5589 - accuracy: 0.5070
 2880/25000 [==>...........................] - ETA: 1:20 - loss: 7.5548 - accuracy: 0.5073
 2912/25000 [==>...........................] - ETA: 1:20 - loss: 7.5666 - accuracy: 0.5065
 2944/25000 [==>...........................] - ETA: 1:20 - loss: 7.5781 - accuracy: 0.5058
 2976/25000 [==>...........................] - ETA: 1:20 - loss: 7.5996 - accuracy: 0.5044
 3008/25000 [==>...........................] - ETA: 1:20 - loss: 7.6105 - accuracy: 0.5037
 3040/25000 [==>...........................] - ETA: 1:19 - loss: 7.6162 - accuracy: 0.5033
 3072/25000 [==>...........................] - ETA: 1:19 - loss: 7.5967 - accuracy: 0.5046
 3104/25000 [==>...........................] - ETA: 1:19 - loss: 7.6222 - accuracy: 0.5029
 3136/25000 [==>...........................] - ETA: 1:19 - loss: 7.6226 - accuracy: 0.5029
 3168/25000 [==>...........................] - ETA: 1:19 - loss: 7.6376 - accuracy: 0.5019
 3200/25000 [==>...........................] - ETA: 1:19 - loss: 7.6379 - accuracy: 0.5019
 3232/25000 [==>...........................] - ETA: 1:19 - loss: 7.6524 - accuracy: 0.5009
 3264/25000 [==>...........................] - ETA: 1:19 - loss: 7.6713 - accuracy: 0.4997
 3296/25000 [==>...........................] - ETA: 1:18 - loss: 7.6480 - accuracy: 0.5012
 3328/25000 [==>...........................] - ETA: 1:18 - loss: 7.6482 - accuracy: 0.5012
 3360/25000 [===>..........................] - ETA: 1:18 - loss: 7.6529 - accuracy: 0.5009
 3392/25000 [===>..........................] - ETA: 1:18 - loss: 7.6485 - accuracy: 0.5012
 3424/25000 [===>..........................] - ETA: 1:18 - loss: 7.6398 - accuracy: 0.5018
 3456/25000 [===>..........................] - ETA: 1:18 - loss: 7.6400 - accuracy: 0.5017
 3488/25000 [===>..........................] - ETA: 1:18 - loss: 7.6358 - accuracy: 0.5020
 3520/25000 [===>..........................] - ETA: 1:18 - loss: 7.6492 - accuracy: 0.5011
 3552/25000 [===>..........................] - ETA: 1:17 - loss: 7.6278 - accuracy: 0.5025
 3584/25000 [===>..........................] - ETA: 1:17 - loss: 7.6153 - accuracy: 0.5033
 3616/25000 [===>..........................] - ETA: 1:17 - loss: 7.6200 - accuracy: 0.5030
 3648/25000 [===>..........................] - ETA: 1:17 - loss: 7.6078 - accuracy: 0.5038
 3680/25000 [===>..........................] - ETA: 1:17 - loss: 7.6333 - accuracy: 0.5022
 3712/25000 [===>..........................] - ETA: 1:17 - loss: 7.6418 - accuracy: 0.5016
 3744/25000 [===>..........................] - ETA: 1:17 - loss: 7.6420 - accuracy: 0.5016
 3776/25000 [===>..........................] - ETA: 1:16 - loss: 7.6463 - accuracy: 0.5013
 3808/25000 [===>..........................] - ETA: 1:16 - loss: 7.6626 - accuracy: 0.5003
 3840/25000 [===>..........................] - ETA: 1:16 - loss: 7.6467 - accuracy: 0.5013
 3872/25000 [===>..........................] - ETA: 1:16 - loss: 7.6666 - accuracy: 0.5000
 3904/25000 [===>..........................] - ETA: 1:16 - loss: 7.6823 - accuracy: 0.4990
 3936/25000 [===>..........................] - ETA: 1:16 - loss: 7.6822 - accuracy: 0.4990
 3968/25000 [===>..........................] - ETA: 1:16 - loss: 7.6782 - accuracy: 0.4992
 4000/25000 [===>..........................] - ETA: 1:16 - loss: 7.6666 - accuracy: 0.5000
 4032/25000 [===>..........................] - ETA: 1:15 - loss: 7.6666 - accuracy: 0.5000
 4064/25000 [===>..........................] - ETA: 1:15 - loss: 7.6478 - accuracy: 0.5012
 4096/25000 [===>..........................] - ETA: 1:15 - loss: 7.6516 - accuracy: 0.5010
 4128/25000 [===>..........................] - ETA: 1:15 - loss: 7.6406 - accuracy: 0.5017
 4160/25000 [===>..........................] - ETA: 1:15 - loss: 7.6408 - accuracy: 0.5017
 4192/25000 [====>.........................] - ETA: 1:15 - loss: 7.6374 - accuracy: 0.5019
 4224/25000 [====>.........................] - ETA: 1:15 - loss: 7.6521 - accuracy: 0.5009
 4256/25000 [====>.........................] - ETA: 1:15 - loss: 7.6450 - accuracy: 0.5014
 4288/25000 [====>.........................] - ETA: 1:14 - loss: 7.6344 - accuracy: 0.5021
 4320/25000 [====>.........................] - ETA: 1:14 - loss: 7.6347 - accuracy: 0.5021
 4352/25000 [====>.........................] - ETA: 1:14 - loss: 7.6208 - accuracy: 0.5030
 4384/25000 [====>.........................] - ETA: 1:14 - loss: 7.6351 - accuracy: 0.5021
 4416/25000 [====>.........................] - ETA: 1:14 - loss: 7.6354 - accuracy: 0.5020
 4448/25000 [====>.........................] - ETA: 1:14 - loss: 7.6390 - accuracy: 0.5018
 4480/25000 [====>.........................] - ETA: 1:14 - loss: 7.6392 - accuracy: 0.5018
 4512/25000 [====>.........................] - ETA: 1:14 - loss: 7.6326 - accuracy: 0.5022
 4544/25000 [====>.........................] - ETA: 1:13 - loss: 7.6228 - accuracy: 0.5029
 4576/25000 [====>.........................] - ETA: 1:13 - loss: 7.6264 - accuracy: 0.5026
 4608/25000 [====>.........................] - ETA: 1:13 - loss: 7.6167 - accuracy: 0.5033
 4640/25000 [====>.........................] - ETA: 1:13 - loss: 7.6038 - accuracy: 0.5041
 4672/25000 [====>.........................] - ETA: 1:13 - loss: 7.6108 - accuracy: 0.5036
 4704/25000 [====>.........................] - ETA: 1:13 - loss: 7.6079 - accuracy: 0.5038
 4736/25000 [====>.........................] - ETA: 1:13 - loss: 7.6148 - accuracy: 0.5034
 4768/25000 [====>.........................] - ETA: 1:13 - loss: 7.6087 - accuracy: 0.5038
 4800/25000 [====>.........................] - ETA: 1:12 - loss: 7.6091 - accuracy: 0.5038
 4832/25000 [====>.........................] - ETA: 1:12 - loss: 7.6095 - accuracy: 0.5037
 4864/25000 [====>.........................] - ETA: 1:12 - loss: 7.6225 - accuracy: 0.5029
 4896/25000 [====>.........................] - ETA: 1:12 - loss: 7.6259 - accuracy: 0.5027
 4928/25000 [====>.........................] - ETA: 1:12 - loss: 7.6324 - accuracy: 0.5022
 4960/25000 [====>.........................] - ETA: 1:12 - loss: 7.6419 - accuracy: 0.5016
 4992/25000 [====>.........................] - ETA: 1:12 - loss: 7.6328 - accuracy: 0.5022
 5024/25000 [=====>........................] - ETA: 1:11 - loss: 7.6178 - accuracy: 0.5032
 5056/25000 [=====>........................] - ETA: 1:11 - loss: 7.6211 - accuracy: 0.5030
 5088/25000 [=====>........................] - ETA: 1:11 - loss: 7.6214 - accuracy: 0.5029
 5120/25000 [=====>........................] - ETA: 1:11 - loss: 7.6247 - accuracy: 0.5027
 5152/25000 [=====>........................] - ETA: 1:11 - loss: 7.6220 - accuracy: 0.5029
 5184/25000 [=====>........................] - ETA: 1:11 - loss: 7.6282 - accuracy: 0.5025
 5216/25000 [=====>........................] - ETA: 1:11 - loss: 7.6225 - accuracy: 0.5029
 5248/25000 [=====>........................] - ETA: 1:11 - loss: 7.6374 - accuracy: 0.5019
 5280/25000 [=====>........................] - ETA: 1:10 - loss: 7.6405 - accuracy: 0.5017
 5312/25000 [=====>........................] - ETA: 1:10 - loss: 7.6493 - accuracy: 0.5011
 5344/25000 [=====>........................] - ETA: 1:10 - loss: 7.6551 - accuracy: 0.5007
 5376/25000 [=====>........................] - ETA: 1:10 - loss: 7.6581 - accuracy: 0.5006
 5408/25000 [=====>........................] - ETA: 1:10 - loss: 7.6695 - accuracy: 0.4998
 5440/25000 [=====>........................] - ETA: 1:10 - loss: 7.6751 - accuracy: 0.4994
 5472/25000 [=====>........................] - ETA: 1:10 - loss: 7.6778 - accuracy: 0.4993
 5504/25000 [=====>........................] - ETA: 1:10 - loss: 7.6750 - accuracy: 0.4995
 5536/25000 [=====>........................] - ETA: 1:09 - loss: 7.6611 - accuracy: 0.5004
 5568/25000 [=====>........................] - ETA: 1:09 - loss: 7.6529 - accuracy: 0.5009
 5600/25000 [=====>........................] - ETA: 1:09 - loss: 7.6584 - accuracy: 0.5005
 5632/25000 [=====>........................] - ETA: 1:09 - loss: 7.6612 - accuracy: 0.5004
 5664/25000 [=====>........................] - ETA: 1:09 - loss: 7.6531 - accuracy: 0.5009
 5696/25000 [=====>........................] - ETA: 1:09 - loss: 7.6532 - accuracy: 0.5009
 5728/25000 [=====>........................] - ETA: 1:09 - loss: 7.6613 - accuracy: 0.5003
 5760/25000 [=====>........................] - ETA: 1:09 - loss: 7.6613 - accuracy: 0.5003
 5792/25000 [=====>........................] - ETA: 1:08 - loss: 7.6613 - accuracy: 0.5003
 5824/25000 [=====>........................] - ETA: 1:08 - loss: 7.6640 - accuracy: 0.5002
 5856/25000 [======>.......................] - ETA: 1:08 - loss: 7.6561 - accuracy: 0.5007
 5888/25000 [======>.......................] - ETA: 1:08 - loss: 7.6536 - accuracy: 0.5008
 5920/25000 [======>.......................] - ETA: 1:08 - loss: 7.6485 - accuracy: 0.5012
 5952/25000 [======>.......................] - ETA: 1:08 - loss: 7.6460 - accuracy: 0.5013
 5984/25000 [======>.......................] - ETA: 1:08 - loss: 7.6461 - accuracy: 0.5013
 6016/25000 [======>.......................] - ETA: 1:07 - loss: 7.6360 - accuracy: 0.5020
 6048/25000 [======>.......................] - ETA: 1:07 - loss: 7.6261 - accuracy: 0.5026
 6080/25000 [======>.......................] - ETA: 1:07 - loss: 7.6111 - accuracy: 0.5036
 6112/25000 [======>.......................] - ETA: 1:07 - loss: 7.6014 - accuracy: 0.5043
 6144/25000 [======>.......................] - ETA: 1:07 - loss: 7.6167 - accuracy: 0.5033
 6176/25000 [======>.......................] - ETA: 1:07 - loss: 7.6170 - accuracy: 0.5032
 6208/25000 [======>.......................] - ETA: 1:07 - loss: 7.6320 - accuracy: 0.5023
 6240/25000 [======>.......................] - ETA: 1:07 - loss: 7.6396 - accuracy: 0.5018
 6272/25000 [======>.......................] - ETA: 1:06 - loss: 7.6446 - accuracy: 0.5014
 6304/25000 [======>.......................] - ETA: 1:06 - loss: 7.6326 - accuracy: 0.5022
 6336/25000 [======>.......................] - ETA: 1:06 - loss: 7.6424 - accuracy: 0.5016
 6368/25000 [======>.......................] - ETA: 1:06 - loss: 7.6425 - accuracy: 0.5016
 6400/25000 [======>.......................] - ETA: 1:06 - loss: 7.6307 - accuracy: 0.5023
 6432/25000 [======>.......................] - ETA: 1:06 - loss: 7.6285 - accuracy: 0.5025
 6464/25000 [======>.......................] - ETA: 1:06 - loss: 7.6334 - accuracy: 0.5022
 6496/25000 [======>.......................] - ETA: 1:06 - loss: 7.6359 - accuracy: 0.5020
 6528/25000 [======>.......................] - ETA: 1:06 - loss: 7.6408 - accuracy: 0.5017
 6560/25000 [======>.......................] - ETA: 1:05 - loss: 7.6409 - accuracy: 0.5017
 6592/25000 [======>.......................] - ETA: 1:05 - loss: 7.6364 - accuracy: 0.5020
 6624/25000 [======>.......................] - ETA: 1:05 - loss: 7.6481 - accuracy: 0.5012
 6656/25000 [======>.......................] - ETA: 1:05 - loss: 7.6390 - accuracy: 0.5018
 6688/25000 [=======>......................] - ETA: 1:05 - loss: 7.6368 - accuracy: 0.5019
 6720/25000 [=======>......................] - ETA: 1:05 - loss: 7.6392 - accuracy: 0.5018
 6752/25000 [=======>......................] - ETA: 1:05 - loss: 7.6371 - accuracy: 0.5019
 6784/25000 [=======>......................] - ETA: 1:05 - loss: 7.6372 - accuracy: 0.5019
 6816/25000 [=======>......................] - ETA: 1:04 - loss: 7.6464 - accuracy: 0.5013
 6848/25000 [=======>......................] - ETA: 1:04 - loss: 7.6532 - accuracy: 0.5009
 6880/25000 [=======>......................] - ETA: 1:04 - loss: 7.6488 - accuracy: 0.5012
 6912/25000 [=======>......................] - ETA: 1:04 - loss: 7.6511 - accuracy: 0.5010
 6944/25000 [=======>......................] - ETA: 1:04 - loss: 7.6467 - accuracy: 0.5013
 6976/25000 [=======>......................] - ETA: 1:04 - loss: 7.6424 - accuracy: 0.5016
 7008/25000 [=======>......................] - ETA: 1:04 - loss: 7.6360 - accuracy: 0.5020
 7040/25000 [=======>......................] - ETA: 1:04 - loss: 7.6339 - accuracy: 0.5021
 7072/25000 [=======>......................] - ETA: 1:03 - loss: 7.6428 - accuracy: 0.5016
 7104/25000 [=======>......................] - ETA: 1:03 - loss: 7.6407 - accuracy: 0.5017
 7136/25000 [=======>......................] - ETA: 1:03 - loss: 7.6430 - accuracy: 0.5015
 7168/25000 [=======>......................] - ETA: 1:03 - loss: 7.6452 - accuracy: 0.5014
 7200/25000 [=======>......................] - ETA: 1:03 - loss: 7.6325 - accuracy: 0.5022
 7232/25000 [=======>......................] - ETA: 1:03 - loss: 7.6306 - accuracy: 0.5024
 7264/25000 [=======>......................] - ETA: 1:03 - loss: 7.6307 - accuracy: 0.5023
 7296/25000 [=======>......................] - ETA: 1:03 - loss: 7.6246 - accuracy: 0.5027
 7328/25000 [=======>......................] - ETA: 1:02 - loss: 7.6373 - accuracy: 0.5019
 7360/25000 [=======>......................] - ETA: 1:02 - loss: 7.6375 - accuracy: 0.5019
 7392/25000 [=======>......................] - ETA: 1:02 - loss: 7.6417 - accuracy: 0.5016
 7424/25000 [=======>......................] - ETA: 1:02 - loss: 7.6356 - accuracy: 0.5020
 7456/25000 [=======>......................] - ETA: 1:02 - loss: 7.6317 - accuracy: 0.5023
 7488/25000 [=======>......................] - ETA: 1:02 - loss: 7.6195 - accuracy: 0.5031
 7520/25000 [========>.....................] - ETA: 1:02 - loss: 7.6136 - accuracy: 0.5035
 7552/25000 [========>.....................] - ETA: 1:02 - loss: 7.6159 - accuracy: 0.5033
 7584/25000 [========>.....................] - ETA: 1:02 - loss: 7.6201 - accuracy: 0.5030
 7616/25000 [========>.....................] - ETA: 1:01 - loss: 7.6203 - accuracy: 0.5030
 7648/25000 [========>.....................] - ETA: 1:01 - loss: 7.6205 - accuracy: 0.5030
 7680/25000 [========>.....................] - ETA: 1:01 - loss: 7.6227 - accuracy: 0.5029
 7712/25000 [========>.....................] - ETA: 1:01 - loss: 7.6269 - accuracy: 0.5026
 7744/25000 [========>.....................] - ETA: 1:01 - loss: 7.6330 - accuracy: 0.5022
 7776/25000 [========>.....................] - ETA: 1:01 - loss: 7.6410 - accuracy: 0.5017
 7808/25000 [========>.....................] - ETA: 1:01 - loss: 7.6431 - accuracy: 0.5015
 7840/25000 [========>.....................] - ETA: 1:01 - loss: 7.6510 - accuracy: 0.5010
 7872/25000 [========>.....................] - ETA: 1:00 - loss: 7.6510 - accuracy: 0.5010
 7904/25000 [========>.....................] - ETA: 1:00 - loss: 7.6530 - accuracy: 0.5009
 7936/25000 [========>.....................] - ETA: 1:00 - loss: 7.6434 - accuracy: 0.5015
 7968/25000 [========>.....................] - ETA: 1:00 - loss: 7.6358 - accuracy: 0.5020
 8000/25000 [========>.....................] - ETA: 1:00 - loss: 7.6321 - accuracy: 0.5023
 8032/25000 [========>.....................] - ETA: 1:00 - loss: 7.6342 - accuracy: 0.5021
 8064/25000 [========>.....................] - ETA: 1:00 - loss: 7.6305 - accuracy: 0.5024
 8096/25000 [========>.....................] - ETA: 1:00 - loss: 7.6287 - accuracy: 0.5025
 8128/25000 [========>.....................] - ETA: 59s - loss: 7.6251 - accuracy: 0.5027 
 8160/25000 [========>.....................] - ETA: 59s - loss: 7.6328 - accuracy: 0.5022
 8192/25000 [========>.....................] - ETA: 59s - loss: 7.6385 - accuracy: 0.5018
 8224/25000 [========>.....................] - ETA: 59s - loss: 7.6405 - accuracy: 0.5017
 8256/25000 [========>.....................] - ETA: 59s - loss: 7.6332 - accuracy: 0.5022
 8288/25000 [========>.....................] - ETA: 59s - loss: 7.6333 - accuracy: 0.5022
 8320/25000 [========>.....................] - ETA: 59s - loss: 7.6371 - accuracy: 0.5019
 8352/25000 [=========>....................] - ETA: 59s - loss: 7.6391 - accuracy: 0.5018
 8384/25000 [=========>....................] - ETA: 59s - loss: 7.6355 - accuracy: 0.5020
 8416/25000 [=========>....................] - ETA: 58s - loss: 7.6356 - accuracy: 0.5020
 8448/25000 [=========>....................] - ETA: 58s - loss: 7.6448 - accuracy: 0.5014
 8480/25000 [=========>....................] - ETA: 58s - loss: 7.6449 - accuracy: 0.5014
 8512/25000 [=========>....................] - ETA: 58s - loss: 7.6432 - accuracy: 0.5015
 8544/25000 [=========>....................] - ETA: 58s - loss: 7.6487 - accuracy: 0.5012
 8576/25000 [=========>....................] - ETA: 58s - loss: 7.6452 - accuracy: 0.5014
 8608/25000 [=========>....................] - ETA: 58s - loss: 7.6452 - accuracy: 0.5014
 8640/25000 [=========>....................] - ETA: 58s - loss: 7.6542 - accuracy: 0.5008
 8672/25000 [=========>....................] - ETA: 58s - loss: 7.6489 - accuracy: 0.5012
 8704/25000 [=========>....................] - ETA: 57s - loss: 7.6543 - accuracy: 0.5008
 8736/25000 [=========>....................] - ETA: 57s - loss: 7.6614 - accuracy: 0.5003
 8768/25000 [=========>....................] - ETA: 57s - loss: 7.6684 - accuracy: 0.4999
 8800/25000 [=========>....................] - ETA: 57s - loss: 7.6736 - accuracy: 0.4995
 8832/25000 [=========>....................] - ETA: 57s - loss: 7.6718 - accuracy: 0.4997
 8864/25000 [=========>....................] - ETA: 57s - loss: 7.6701 - accuracy: 0.4998
 8896/25000 [=========>....................] - ETA: 57s - loss: 7.6649 - accuracy: 0.5001
 8928/25000 [=========>....................] - ETA: 57s - loss: 7.6632 - accuracy: 0.5002
 8960/25000 [=========>....................] - ETA: 56s - loss: 7.6632 - accuracy: 0.5002
 8992/25000 [=========>....................] - ETA: 56s - loss: 7.6734 - accuracy: 0.4996
 9024/25000 [=========>....................] - ETA: 56s - loss: 7.6717 - accuracy: 0.4997
 9056/25000 [=========>....................] - ETA: 56s - loss: 7.6700 - accuracy: 0.4998
 9088/25000 [=========>....................] - ETA: 56s - loss: 7.6649 - accuracy: 0.5001
 9120/25000 [=========>....................] - ETA: 56s - loss: 7.6633 - accuracy: 0.5002
 9152/25000 [=========>....................] - ETA: 56s - loss: 7.6633 - accuracy: 0.5002
 9184/25000 [==========>...................] - ETA: 56s - loss: 7.6633 - accuracy: 0.5002
 9216/25000 [==========>...................] - ETA: 55s - loss: 7.6616 - accuracy: 0.5003
 9248/25000 [==========>...................] - ETA: 55s - loss: 7.6583 - accuracy: 0.5005
 9280/25000 [==========>...................] - ETA: 55s - loss: 7.6584 - accuracy: 0.5005
 9312/25000 [==========>...................] - ETA: 55s - loss: 7.6600 - accuracy: 0.5004
 9344/25000 [==========>...................] - ETA: 55s - loss: 7.6633 - accuracy: 0.5002
 9376/25000 [==========>...................] - ETA: 55s - loss: 7.6650 - accuracy: 0.5001
 9408/25000 [==========>...................] - ETA: 55s - loss: 7.6552 - accuracy: 0.5007
 9440/25000 [==========>...................] - ETA: 55s - loss: 7.6520 - accuracy: 0.5010
 9472/25000 [==========>...................] - ETA: 54s - loss: 7.6521 - accuracy: 0.5010
 9504/25000 [==========>...................] - ETA: 54s - loss: 7.6569 - accuracy: 0.5006
 9536/25000 [==========>...................] - ETA: 54s - loss: 7.6618 - accuracy: 0.5003
 9568/25000 [==========>...................] - ETA: 54s - loss: 7.6570 - accuracy: 0.5006
 9600/25000 [==========>...................] - ETA: 54s - loss: 7.6570 - accuracy: 0.5006
 9632/25000 [==========>...................] - ETA: 54s - loss: 7.6555 - accuracy: 0.5007
 9664/25000 [==========>...................] - ETA: 54s - loss: 7.6650 - accuracy: 0.5001
 9696/25000 [==========>...................] - ETA: 54s - loss: 7.6635 - accuracy: 0.5002
 9728/25000 [==========>...................] - ETA: 54s - loss: 7.6682 - accuracy: 0.4999
 9760/25000 [==========>...................] - ETA: 53s - loss: 7.6635 - accuracy: 0.5002
 9792/25000 [==========>...................] - ETA: 53s - loss: 7.6588 - accuracy: 0.5005
 9824/25000 [==========>...................] - ETA: 53s - loss: 7.6588 - accuracy: 0.5005
 9856/25000 [==========>...................] - ETA: 53s - loss: 7.6604 - accuracy: 0.5004
 9888/25000 [==========>...................] - ETA: 53s - loss: 7.6635 - accuracy: 0.5002
 9920/25000 [==========>...................] - ETA: 53s - loss: 7.6620 - accuracy: 0.5003
 9952/25000 [==========>...................] - ETA: 53s - loss: 7.6589 - accuracy: 0.5005
 9984/25000 [==========>...................] - ETA: 53s - loss: 7.6605 - accuracy: 0.5004
10016/25000 [===========>..................] - ETA: 52s - loss: 7.6620 - accuracy: 0.5003
10048/25000 [===========>..................] - ETA: 52s - loss: 7.6620 - accuracy: 0.5003
10080/25000 [===========>..................] - ETA: 52s - loss: 7.6666 - accuracy: 0.5000
10112/25000 [===========>..................] - ETA: 52s - loss: 7.6712 - accuracy: 0.4997
10144/25000 [===========>..................] - ETA: 52s - loss: 7.6712 - accuracy: 0.4997
10176/25000 [===========>..................] - ETA: 52s - loss: 7.6787 - accuracy: 0.4992
10208/25000 [===========>..................] - ETA: 52s - loss: 7.6801 - accuracy: 0.4991
10240/25000 [===========>..................] - ETA: 52s - loss: 7.6816 - accuracy: 0.4990
10272/25000 [===========>..................] - ETA: 52s - loss: 7.6815 - accuracy: 0.4990
10304/25000 [===========>..................] - ETA: 51s - loss: 7.6726 - accuracy: 0.4996
10336/25000 [===========>..................] - ETA: 51s - loss: 7.6711 - accuracy: 0.4997
10368/25000 [===========>..................] - ETA: 51s - loss: 7.6681 - accuracy: 0.4999
10400/25000 [===========>..................] - ETA: 51s - loss: 7.6637 - accuracy: 0.5002
10432/25000 [===========>..................] - ETA: 51s - loss: 7.6681 - accuracy: 0.4999
10464/25000 [===========>..................] - ETA: 51s - loss: 7.6608 - accuracy: 0.5004
10496/25000 [===========>..................] - ETA: 51s - loss: 7.6593 - accuracy: 0.5005
10528/25000 [===========>..................] - ETA: 51s - loss: 7.6652 - accuracy: 0.5001
10560/25000 [===========>..................] - ETA: 50s - loss: 7.6608 - accuracy: 0.5004
10592/25000 [===========>..................] - ETA: 50s - loss: 7.6608 - accuracy: 0.5004
10624/25000 [===========>..................] - ETA: 50s - loss: 7.6637 - accuracy: 0.5002
10656/25000 [===========>..................] - ETA: 50s - loss: 7.6594 - accuracy: 0.5005
10688/25000 [===========>..................] - ETA: 50s - loss: 7.6580 - accuracy: 0.5006
10720/25000 [===========>..................] - ETA: 50s - loss: 7.6623 - accuracy: 0.5003
10752/25000 [===========>..................] - ETA: 50s - loss: 7.6609 - accuracy: 0.5004
10784/25000 [===========>..................] - ETA: 50s - loss: 7.6624 - accuracy: 0.5003
10816/25000 [===========>..................] - ETA: 50s - loss: 7.6652 - accuracy: 0.5001
10848/25000 [============>.................] - ETA: 49s - loss: 7.6624 - accuracy: 0.5003
10880/25000 [============>.................] - ETA: 49s - loss: 7.6568 - accuracy: 0.5006
10912/25000 [============>.................] - ETA: 49s - loss: 7.6596 - accuracy: 0.5005
10944/25000 [============>.................] - ETA: 49s - loss: 7.6666 - accuracy: 0.5000
10976/25000 [============>.................] - ETA: 49s - loss: 7.6568 - accuracy: 0.5006
11008/25000 [============>.................] - ETA: 49s - loss: 7.6569 - accuracy: 0.5006
11040/25000 [============>.................] - ETA: 49s - loss: 7.6541 - accuracy: 0.5008
11072/25000 [============>.................] - ETA: 49s - loss: 7.6500 - accuracy: 0.5011
11104/25000 [============>.................] - ETA: 49s - loss: 7.6514 - accuracy: 0.5010
11136/25000 [============>.................] - ETA: 48s - loss: 7.6487 - accuracy: 0.5012
11168/25000 [============>.................] - ETA: 48s - loss: 7.6392 - accuracy: 0.5018
11200/25000 [============>.................] - ETA: 48s - loss: 7.6351 - accuracy: 0.5021
11232/25000 [============>.................] - ETA: 48s - loss: 7.6366 - accuracy: 0.5020
11264/25000 [============>.................] - ETA: 48s - loss: 7.6353 - accuracy: 0.5020
11296/25000 [============>.................] - ETA: 48s - loss: 7.6340 - accuracy: 0.5021
11328/25000 [============>.................] - ETA: 48s - loss: 7.6368 - accuracy: 0.5019
11360/25000 [============>.................] - ETA: 48s - loss: 7.6369 - accuracy: 0.5019
11392/25000 [============>.................] - ETA: 47s - loss: 7.6370 - accuracy: 0.5019
11424/25000 [============>.................] - ETA: 47s - loss: 7.6384 - accuracy: 0.5018
11456/25000 [============>.................] - ETA: 47s - loss: 7.6412 - accuracy: 0.5017
11488/25000 [============>.................] - ETA: 47s - loss: 7.6413 - accuracy: 0.5017
11520/25000 [============>.................] - ETA: 47s - loss: 7.6440 - accuracy: 0.5015
11552/25000 [============>.................] - ETA: 47s - loss: 7.6401 - accuracy: 0.5017
11584/25000 [============>.................] - ETA: 47s - loss: 7.6362 - accuracy: 0.5020
11616/25000 [============>.................] - ETA: 47s - loss: 7.6376 - accuracy: 0.5019
11648/25000 [============>.................] - ETA: 47s - loss: 7.6363 - accuracy: 0.5020
11680/25000 [=============>................] - ETA: 46s - loss: 7.6325 - accuracy: 0.5022
11712/25000 [=============>................] - ETA: 46s - loss: 7.6300 - accuracy: 0.5024
11744/25000 [=============>................] - ETA: 46s - loss: 7.6275 - accuracy: 0.5026
11776/25000 [=============>................] - ETA: 46s - loss: 7.6263 - accuracy: 0.5026
11808/25000 [=============>................] - ETA: 46s - loss: 7.6264 - accuracy: 0.5026
11840/25000 [=============>................] - ETA: 46s - loss: 7.6226 - accuracy: 0.5029
11872/25000 [=============>................] - ETA: 46s - loss: 7.6266 - accuracy: 0.5026
11904/25000 [=============>................] - ETA: 46s - loss: 7.6254 - accuracy: 0.5027
11936/25000 [=============>................] - ETA: 46s - loss: 7.6178 - accuracy: 0.5032
11968/25000 [=============>................] - ETA: 45s - loss: 7.6243 - accuracy: 0.5028
12000/25000 [=============>................] - ETA: 45s - loss: 7.6257 - accuracy: 0.5027
12032/25000 [=============>................] - ETA: 45s - loss: 7.6284 - accuracy: 0.5025
12064/25000 [=============>................] - ETA: 45s - loss: 7.6259 - accuracy: 0.5027
12096/25000 [=============>................] - ETA: 45s - loss: 7.6273 - accuracy: 0.5026
12128/25000 [=============>................] - ETA: 45s - loss: 7.6262 - accuracy: 0.5026
12160/25000 [=============>................] - ETA: 45s - loss: 7.6212 - accuracy: 0.5030
12192/25000 [=============>................] - ETA: 45s - loss: 7.6138 - accuracy: 0.5034
12224/25000 [=============>................] - ETA: 45s - loss: 7.6077 - accuracy: 0.5038
12256/25000 [=============>................] - ETA: 44s - loss: 7.6103 - accuracy: 0.5037
12288/25000 [=============>................] - ETA: 44s - loss: 7.6117 - accuracy: 0.5036
12320/25000 [=============>................] - ETA: 44s - loss: 7.6131 - accuracy: 0.5035
12352/25000 [=============>................] - ETA: 44s - loss: 7.6132 - accuracy: 0.5035
12384/25000 [=============>................] - ETA: 44s - loss: 7.6159 - accuracy: 0.5033
12416/25000 [=============>................] - ETA: 44s - loss: 7.6123 - accuracy: 0.5035
12448/25000 [=============>................] - ETA: 44s - loss: 7.6161 - accuracy: 0.5033
12480/25000 [=============>................] - ETA: 44s - loss: 7.6126 - accuracy: 0.5035
12512/25000 [==============>...............] - ETA: 43s - loss: 7.6102 - accuracy: 0.5037
12544/25000 [==============>...............] - ETA: 43s - loss: 7.6116 - accuracy: 0.5036
12576/25000 [==============>...............] - ETA: 43s - loss: 7.6118 - accuracy: 0.5036
12608/25000 [==============>...............] - ETA: 43s - loss: 7.6180 - accuracy: 0.5032
12640/25000 [==============>...............] - ETA: 43s - loss: 7.6205 - accuracy: 0.5030
12672/25000 [==============>...............] - ETA: 43s - loss: 7.6243 - accuracy: 0.5028
12704/25000 [==============>...............] - ETA: 43s - loss: 7.6268 - accuracy: 0.5026
12736/25000 [==============>...............] - ETA: 43s - loss: 7.6197 - accuracy: 0.5031
12768/25000 [==============>...............] - ETA: 43s - loss: 7.6210 - accuracy: 0.5030
12800/25000 [==============>...............] - ETA: 42s - loss: 7.6211 - accuracy: 0.5030
12832/25000 [==============>...............] - ETA: 42s - loss: 7.6224 - accuracy: 0.5029
12864/25000 [==============>...............] - ETA: 42s - loss: 7.6225 - accuracy: 0.5029
12896/25000 [==============>...............] - ETA: 42s - loss: 7.6214 - accuracy: 0.5029
12928/25000 [==============>...............] - ETA: 42s - loss: 7.6168 - accuracy: 0.5032
12960/25000 [==============>...............] - ETA: 42s - loss: 7.6146 - accuracy: 0.5034
12992/25000 [==============>...............] - ETA: 42s - loss: 7.6147 - accuracy: 0.5034
13024/25000 [==============>...............] - ETA: 42s - loss: 7.6078 - accuracy: 0.5038
13056/25000 [==============>...............] - ETA: 42s - loss: 7.6079 - accuracy: 0.5038
13088/25000 [==============>...............] - ETA: 41s - loss: 7.6069 - accuracy: 0.5039
13120/25000 [==============>...............] - ETA: 41s - loss: 7.6105 - accuracy: 0.5037
13152/25000 [==============>...............] - ETA: 41s - loss: 7.6083 - accuracy: 0.5038
13184/25000 [==============>...............] - ETA: 41s - loss: 7.6050 - accuracy: 0.5040
13216/25000 [==============>...............] - ETA: 41s - loss: 7.6051 - accuracy: 0.5040
13248/25000 [==============>...............] - ETA: 41s - loss: 7.6041 - accuracy: 0.5041
13280/25000 [==============>...............] - ETA: 41s - loss: 7.6066 - accuracy: 0.5039
13312/25000 [==============>...............] - ETA: 41s - loss: 7.6079 - accuracy: 0.5038
13344/25000 [===============>..............] - ETA: 41s - loss: 7.6092 - accuracy: 0.5037
13376/25000 [===============>..............] - ETA: 40s - loss: 7.6139 - accuracy: 0.5034
13408/25000 [===============>..............] - ETA: 40s - loss: 7.6117 - accuracy: 0.5036
13440/25000 [===============>..............] - ETA: 40s - loss: 7.6130 - accuracy: 0.5035
13472/25000 [===============>..............] - ETA: 40s - loss: 7.6086 - accuracy: 0.5038
13504/25000 [===============>..............] - ETA: 40s - loss: 7.6064 - accuracy: 0.5039
13536/25000 [===============>..............] - ETA: 40s - loss: 7.6100 - accuracy: 0.5037
13568/25000 [===============>..............] - ETA: 40s - loss: 7.6011 - accuracy: 0.5043
13600/25000 [===============>..............] - ETA: 40s - loss: 7.5990 - accuracy: 0.5044
13632/25000 [===============>..............] - ETA: 39s - loss: 7.5980 - accuracy: 0.5045
13664/25000 [===============>..............] - ETA: 39s - loss: 7.5970 - accuracy: 0.5045
13696/25000 [===============>..............] - ETA: 39s - loss: 7.5938 - accuracy: 0.5047
13728/25000 [===============>..............] - ETA: 39s - loss: 7.5940 - accuracy: 0.5047
13760/25000 [===============>..............] - ETA: 39s - loss: 7.5953 - accuracy: 0.5047
13792/25000 [===============>..............] - ETA: 39s - loss: 7.5921 - accuracy: 0.5049
13824/25000 [===============>..............] - ETA: 39s - loss: 7.5934 - accuracy: 0.5048
13856/25000 [===============>..............] - ETA: 39s - loss: 7.5914 - accuracy: 0.5049
13888/25000 [===============>..............] - ETA: 39s - loss: 7.5960 - accuracy: 0.5046
13920/25000 [===============>..............] - ETA: 38s - loss: 7.5994 - accuracy: 0.5044
13952/25000 [===============>..............] - ETA: 38s - loss: 7.5974 - accuracy: 0.5045
13984/25000 [===============>..............] - ETA: 38s - loss: 7.5964 - accuracy: 0.5046
14016/25000 [===============>..............] - ETA: 38s - loss: 7.6010 - accuracy: 0.5043
14048/25000 [===============>..............] - ETA: 38s - loss: 7.6066 - accuracy: 0.5039
14080/25000 [===============>..............] - ETA: 38s - loss: 7.6067 - accuracy: 0.5039
14112/25000 [===============>..............] - ETA: 38s - loss: 7.6047 - accuracy: 0.5040
14144/25000 [===============>..............] - ETA: 38s - loss: 7.6059 - accuracy: 0.5040
14176/25000 [================>.............] - ETA: 38s - loss: 7.6006 - accuracy: 0.5043
14208/25000 [================>.............] - ETA: 37s - loss: 7.6040 - accuracy: 0.5041
14240/25000 [================>.............] - ETA: 37s - loss: 7.6020 - accuracy: 0.5042
14272/25000 [================>.............] - ETA: 37s - loss: 7.6075 - accuracy: 0.5039
14304/25000 [================>.............] - ETA: 37s - loss: 7.6141 - accuracy: 0.5034
14336/25000 [================>.............] - ETA: 37s - loss: 7.6131 - accuracy: 0.5035
14368/25000 [================>.............] - ETA: 37s - loss: 7.6154 - accuracy: 0.5033
14400/25000 [================>.............] - ETA: 37s - loss: 7.6112 - accuracy: 0.5036
14432/25000 [================>.............] - ETA: 37s - loss: 7.6135 - accuracy: 0.5035
14464/25000 [================>.............] - ETA: 37s - loss: 7.6126 - accuracy: 0.5035
14496/25000 [================>.............] - ETA: 36s - loss: 7.6169 - accuracy: 0.5032
14528/25000 [================>.............] - ETA: 36s - loss: 7.6170 - accuracy: 0.5032
14560/25000 [================>.............] - ETA: 36s - loss: 7.6203 - accuracy: 0.5030
14592/25000 [================>.............] - ETA: 36s - loss: 7.6225 - accuracy: 0.5029
14624/25000 [================>.............] - ETA: 36s - loss: 7.6236 - accuracy: 0.5028
14656/25000 [================>.............] - ETA: 36s - loss: 7.6227 - accuracy: 0.5029
14688/25000 [================>.............] - ETA: 36s - loss: 7.6238 - accuracy: 0.5028
14720/25000 [================>.............] - ETA: 36s - loss: 7.6239 - accuracy: 0.5028
14752/25000 [================>.............] - ETA: 36s - loss: 7.6198 - accuracy: 0.5031
14784/25000 [================>.............] - ETA: 35s - loss: 7.6137 - accuracy: 0.5034
14816/25000 [================>.............] - ETA: 35s - loss: 7.6149 - accuracy: 0.5034
14848/25000 [================>.............] - ETA: 35s - loss: 7.6140 - accuracy: 0.5034
14880/25000 [================>.............] - ETA: 35s - loss: 7.6120 - accuracy: 0.5036
14912/25000 [================>.............] - ETA: 35s - loss: 7.6132 - accuracy: 0.5035
14944/25000 [================>.............] - ETA: 35s - loss: 7.6133 - accuracy: 0.5035
14976/25000 [================>.............] - ETA: 35s - loss: 7.6175 - accuracy: 0.5032
15008/25000 [=================>............] - ETA: 35s - loss: 7.6186 - accuracy: 0.5031
15040/25000 [=================>............] - ETA: 34s - loss: 7.6207 - accuracy: 0.5030
15072/25000 [=================>............] - ETA: 34s - loss: 7.6249 - accuracy: 0.5027
15104/25000 [=================>............] - ETA: 34s - loss: 7.6230 - accuracy: 0.5028
15136/25000 [=================>............] - ETA: 34s - loss: 7.6241 - accuracy: 0.5028
15168/25000 [=================>............] - ETA: 34s - loss: 7.6242 - accuracy: 0.5028
15200/25000 [=================>............] - ETA: 34s - loss: 7.6243 - accuracy: 0.5028
15232/25000 [=================>............] - ETA: 34s - loss: 7.6264 - accuracy: 0.5026
15264/25000 [=================>............] - ETA: 34s - loss: 7.6254 - accuracy: 0.5027
15296/25000 [=================>............] - ETA: 34s - loss: 7.6255 - accuracy: 0.5027
15328/25000 [=================>............] - ETA: 33s - loss: 7.6276 - accuracy: 0.5025
15360/25000 [=================>............] - ETA: 33s - loss: 7.6277 - accuracy: 0.5025
15392/25000 [=================>............] - ETA: 33s - loss: 7.6288 - accuracy: 0.5025
15424/25000 [=================>............] - ETA: 33s - loss: 7.6278 - accuracy: 0.5025
15456/25000 [=================>............] - ETA: 33s - loss: 7.6329 - accuracy: 0.5022
15488/25000 [=================>............] - ETA: 33s - loss: 7.6359 - accuracy: 0.5020
15520/25000 [=================>............] - ETA: 33s - loss: 7.6340 - accuracy: 0.5021
15552/25000 [=================>............] - ETA: 33s - loss: 7.6321 - accuracy: 0.5023
15584/25000 [=================>............] - ETA: 33s - loss: 7.6332 - accuracy: 0.5022
15616/25000 [=================>............] - ETA: 32s - loss: 7.6342 - accuracy: 0.5021
15648/25000 [=================>............] - ETA: 32s - loss: 7.6343 - accuracy: 0.5021
15680/25000 [=================>............] - ETA: 32s - loss: 7.6334 - accuracy: 0.5022
15712/25000 [=================>............] - ETA: 32s - loss: 7.6344 - accuracy: 0.5021
15744/25000 [=================>............] - ETA: 32s - loss: 7.6355 - accuracy: 0.5020
15776/25000 [=================>............] - ETA: 32s - loss: 7.6355 - accuracy: 0.5020
15808/25000 [=================>............] - ETA: 32s - loss: 7.6395 - accuracy: 0.5018
15840/25000 [==================>...........] - ETA: 32s - loss: 7.6453 - accuracy: 0.5014
15872/25000 [==================>...........] - ETA: 32s - loss: 7.6483 - accuracy: 0.5012
15904/25000 [==================>...........] - ETA: 31s - loss: 7.6473 - accuracy: 0.5013
15936/25000 [==================>...........] - ETA: 31s - loss: 7.6455 - accuracy: 0.5014
15968/25000 [==================>...........] - ETA: 31s - loss: 7.6513 - accuracy: 0.5010
16000/25000 [==================>...........] - ETA: 31s - loss: 7.6513 - accuracy: 0.5010
16032/25000 [==================>...........] - ETA: 31s - loss: 7.6504 - accuracy: 0.5011
16064/25000 [==================>...........] - ETA: 31s - loss: 7.6513 - accuracy: 0.5010
16096/25000 [==================>...........] - ETA: 31s - loss: 7.6504 - accuracy: 0.5011
16128/25000 [==================>...........] - ETA: 31s - loss: 7.6533 - accuracy: 0.5009
16160/25000 [==================>...........] - ETA: 31s - loss: 7.6562 - accuracy: 0.5007
16192/25000 [==================>...........] - ETA: 30s - loss: 7.6562 - accuracy: 0.5007
16224/25000 [==================>...........] - ETA: 30s - loss: 7.6609 - accuracy: 0.5004
16256/25000 [==================>...........] - ETA: 30s - loss: 7.6610 - accuracy: 0.5004
16288/25000 [==================>...........] - ETA: 30s - loss: 7.6525 - accuracy: 0.5009
16320/25000 [==================>...........] - ETA: 30s - loss: 7.6497 - accuracy: 0.5011
16352/25000 [==================>...........] - ETA: 30s - loss: 7.6488 - accuracy: 0.5012
16384/25000 [==================>...........] - ETA: 30s - loss: 7.6488 - accuracy: 0.5012
16416/25000 [==================>...........] - ETA: 30s - loss: 7.6507 - accuracy: 0.5010
16448/25000 [==================>...........] - ETA: 29s - loss: 7.6470 - accuracy: 0.5013
16480/25000 [==================>...........] - ETA: 29s - loss: 7.6452 - accuracy: 0.5014
16512/25000 [==================>...........] - ETA: 29s - loss: 7.6415 - accuracy: 0.5016
16544/25000 [==================>...........] - ETA: 29s - loss: 7.6462 - accuracy: 0.5013
16576/25000 [==================>...........] - ETA: 29s - loss: 7.6490 - accuracy: 0.5011
16608/25000 [==================>...........] - ETA: 29s - loss: 7.6482 - accuracy: 0.5012
16640/25000 [==================>...........] - ETA: 29s - loss: 7.6500 - accuracy: 0.5011
16672/25000 [===================>..........] - ETA: 29s - loss: 7.6482 - accuracy: 0.5012
16704/25000 [===================>..........] - ETA: 29s - loss: 7.6501 - accuracy: 0.5011
16736/25000 [===================>..........] - ETA: 28s - loss: 7.6501 - accuracy: 0.5011
16768/25000 [===================>..........] - ETA: 28s - loss: 7.6511 - accuracy: 0.5010
16800/25000 [===================>..........] - ETA: 28s - loss: 7.6529 - accuracy: 0.5009
16832/25000 [===================>..........] - ETA: 28s - loss: 7.6520 - accuracy: 0.5010
16864/25000 [===================>..........] - ETA: 28s - loss: 7.6539 - accuracy: 0.5008
16896/25000 [===================>..........] - ETA: 28s - loss: 7.6575 - accuracy: 0.5006
16928/25000 [===================>..........] - ETA: 28s - loss: 7.6594 - accuracy: 0.5005
16960/25000 [===================>..........] - ETA: 28s - loss: 7.6585 - accuracy: 0.5005
16992/25000 [===================>..........] - ETA: 28s - loss: 7.6585 - accuracy: 0.5005
17024/25000 [===================>..........] - ETA: 27s - loss: 7.6603 - accuracy: 0.5004
17056/25000 [===================>..........] - ETA: 27s - loss: 7.6675 - accuracy: 0.4999
17088/25000 [===================>..........] - ETA: 27s - loss: 7.6657 - accuracy: 0.5001
17120/25000 [===================>..........] - ETA: 27s - loss: 7.6639 - accuracy: 0.5002
17152/25000 [===================>..........] - ETA: 27s - loss: 7.6630 - accuracy: 0.5002
17184/25000 [===================>..........] - ETA: 27s - loss: 7.6648 - accuracy: 0.5001
17216/25000 [===================>..........] - ETA: 27s - loss: 7.6666 - accuracy: 0.5000
17248/25000 [===================>..........] - ETA: 27s - loss: 7.6657 - accuracy: 0.5001
17280/25000 [===================>..........] - ETA: 27s - loss: 7.6657 - accuracy: 0.5001
17312/25000 [===================>..........] - ETA: 26s - loss: 7.6693 - accuracy: 0.4998
17344/25000 [===================>..........] - ETA: 26s - loss: 7.6684 - accuracy: 0.4999
17376/25000 [===================>..........] - ETA: 26s - loss: 7.6684 - accuracy: 0.4999
17408/25000 [===================>..........] - ETA: 26s - loss: 7.6701 - accuracy: 0.4998
17440/25000 [===================>..........] - ETA: 26s - loss: 7.6701 - accuracy: 0.4998
17472/25000 [===================>..........] - ETA: 26s - loss: 7.6710 - accuracy: 0.4997
17504/25000 [====================>.........] - ETA: 26s - loss: 7.6701 - accuracy: 0.4998
17536/25000 [====================>.........] - ETA: 26s - loss: 7.6684 - accuracy: 0.4999
17568/25000 [====================>.........] - ETA: 26s - loss: 7.6640 - accuracy: 0.5002
17600/25000 [====================>.........] - ETA: 25s - loss: 7.6588 - accuracy: 0.5005
17632/25000 [====================>.........] - ETA: 25s - loss: 7.6562 - accuracy: 0.5007
17664/25000 [====================>.........] - ETA: 25s - loss: 7.6562 - accuracy: 0.5007
17696/25000 [====================>.........] - ETA: 25s - loss: 7.6614 - accuracy: 0.5003
17728/25000 [====================>.........] - ETA: 25s - loss: 7.6623 - accuracy: 0.5003
17760/25000 [====================>.........] - ETA: 25s - loss: 7.6632 - accuracy: 0.5002
17792/25000 [====================>.........] - ETA: 25s - loss: 7.6666 - accuracy: 0.5000
17824/25000 [====================>.........] - ETA: 25s - loss: 7.6640 - accuracy: 0.5002
17856/25000 [====================>.........] - ETA: 25s - loss: 7.6623 - accuracy: 0.5003
17888/25000 [====================>.........] - ETA: 24s - loss: 7.6640 - accuracy: 0.5002
17920/25000 [====================>.........] - ETA: 24s - loss: 7.6683 - accuracy: 0.4999
17952/25000 [====================>.........] - ETA: 24s - loss: 7.6632 - accuracy: 0.5002
17984/25000 [====================>.........] - ETA: 24s - loss: 7.6632 - accuracy: 0.5002
18016/25000 [====================>.........] - ETA: 24s - loss: 7.6641 - accuracy: 0.5002
18048/25000 [====================>.........] - ETA: 24s - loss: 7.6641 - accuracy: 0.5002
18080/25000 [====================>.........] - ETA: 24s - loss: 7.6649 - accuracy: 0.5001
18112/25000 [====================>.........] - ETA: 24s - loss: 7.6632 - accuracy: 0.5002
18144/25000 [====================>.........] - ETA: 24s - loss: 7.6649 - accuracy: 0.5001
18176/25000 [====================>.........] - ETA: 23s - loss: 7.6616 - accuracy: 0.5003
18208/25000 [====================>.........] - ETA: 23s - loss: 7.6633 - accuracy: 0.5002
18240/25000 [====================>.........] - ETA: 23s - loss: 7.6658 - accuracy: 0.5001
18272/25000 [====================>.........] - ETA: 23s - loss: 7.6649 - accuracy: 0.5001
18304/25000 [====================>.........] - ETA: 23s - loss: 7.6616 - accuracy: 0.5003
18336/25000 [=====================>........] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
18368/25000 [=====================>........] - ETA: 23s - loss: 7.6666 - accuracy: 0.5000
18400/25000 [=====================>........] - ETA: 23s - loss: 7.6650 - accuracy: 0.5001
18432/25000 [=====================>........] - ETA: 23s - loss: 7.6625 - accuracy: 0.5003
18464/25000 [=====================>........] - ETA: 22s - loss: 7.6608 - accuracy: 0.5004
18496/25000 [=====================>........] - ETA: 22s - loss: 7.6600 - accuracy: 0.5004
18528/25000 [=====================>........] - ETA: 22s - loss: 7.6617 - accuracy: 0.5003
18560/25000 [=====================>........] - ETA: 22s - loss: 7.6575 - accuracy: 0.5006
18592/25000 [=====================>........] - ETA: 22s - loss: 7.6608 - accuracy: 0.5004
18624/25000 [=====================>........] - ETA: 22s - loss: 7.6600 - accuracy: 0.5004
18656/25000 [=====================>........] - ETA: 22s - loss: 7.6609 - accuracy: 0.5004
18688/25000 [=====================>........] - ETA: 22s - loss: 7.6642 - accuracy: 0.5002
18720/25000 [=====================>........] - ETA: 22s - loss: 7.6592 - accuracy: 0.5005
18752/25000 [=====================>........] - ETA: 21s - loss: 7.6576 - accuracy: 0.5006
18784/25000 [=====================>........] - ETA: 21s - loss: 7.6593 - accuracy: 0.5005
18816/25000 [=====================>........] - ETA: 21s - loss: 7.6585 - accuracy: 0.5005
18848/25000 [=====================>........] - ETA: 21s - loss: 7.6552 - accuracy: 0.5007
18880/25000 [=====================>........] - ETA: 21s - loss: 7.6544 - accuracy: 0.5008
18912/25000 [=====================>........] - ETA: 21s - loss: 7.6504 - accuracy: 0.5011
18944/25000 [=====================>........] - ETA: 21s - loss: 7.6480 - accuracy: 0.5012
18976/25000 [=====================>........] - ETA: 21s - loss: 7.6488 - accuracy: 0.5012
19008/25000 [=====================>........] - ETA: 20s - loss: 7.6481 - accuracy: 0.5012
19040/25000 [=====================>........] - ETA: 20s - loss: 7.6497 - accuracy: 0.5011
19072/25000 [=====================>........] - ETA: 20s - loss: 7.6530 - accuracy: 0.5009
19104/25000 [=====================>........] - ETA: 20s - loss: 7.6538 - accuracy: 0.5008
19136/25000 [=====================>........] - ETA: 20s - loss: 7.6570 - accuracy: 0.5006
19168/25000 [======================>.......] - ETA: 20s - loss: 7.6586 - accuracy: 0.5005
19200/25000 [======================>.......] - ETA: 20s - loss: 7.6578 - accuracy: 0.5006
19232/25000 [======================>.......] - ETA: 20s - loss: 7.6602 - accuracy: 0.5004
19264/25000 [======================>.......] - ETA: 20s - loss: 7.6555 - accuracy: 0.5007
19296/25000 [======================>.......] - ETA: 19s - loss: 7.6531 - accuracy: 0.5009
19328/25000 [======================>.......] - ETA: 19s - loss: 7.6523 - accuracy: 0.5009
19360/25000 [======================>.......] - ETA: 19s - loss: 7.6524 - accuracy: 0.5009
19392/25000 [======================>.......] - ETA: 19s - loss: 7.6548 - accuracy: 0.5008
19424/25000 [======================>.......] - ETA: 19s - loss: 7.6540 - accuracy: 0.5008
19456/25000 [======================>.......] - ETA: 19s - loss: 7.6540 - accuracy: 0.5008
19488/25000 [======================>.......] - ETA: 19s - loss: 7.6517 - accuracy: 0.5010
19520/25000 [======================>.......] - ETA: 19s - loss: 7.6517 - accuracy: 0.5010
19552/25000 [======================>.......] - ETA: 19s - loss: 7.6572 - accuracy: 0.5006
19584/25000 [======================>.......] - ETA: 18s - loss: 7.6564 - accuracy: 0.5007
19616/25000 [======================>.......] - ETA: 18s - loss: 7.6541 - accuracy: 0.5008
19648/25000 [======================>.......] - ETA: 18s - loss: 7.6526 - accuracy: 0.5009
19680/25000 [======================>.......] - ETA: 18s - loss: 7.6518 - accuracy: 0.5010
19712/25000 [======================>.......] - ETA: 18s - loss: 7.6487 - accuracy: 0.5012
19744/25000 [======================>.......] - ETA: 18s - loss: 7.6488 - accuracy: 0.5012
19776/25000 [======================>.......] - ETA: 18s - loss: 7.6534 - accuracy: 0.5009
19808/25000 [======================>.......] - ETA: 18s - loss: 7.6550 - accuracy: 0.5008
19840/25000 [======================>.......] - ETA: 18s - loss: 7.6566 - accuracy: 0.5007
19872/25000 [======================>.......] - ETA: 17s - loss: 7.6597 - accuracy: 0.5005
19904/25000 [======================>.......] - ETA: 17s - loss: 7.6605 - accuracy: 0.5004
19936/25000 [======================>.......] - ETA: 17s - loss: 7.6582 - accuracy: 0.5006
19968/25000 [======================>.......] - ETA: 17s - loss: 7.6551 - accuracy: 0.5008
20000/25000 [=======================>......] - ETA: 17s - loss: 7.6513 - accuracy: 0.5010
20032/25000 [=======================>......] - ETA: 17s - loss: 7.6528 - accuracy: 0.5009
20064/25000 [=======================>......] - ETA: 17s - loss: 7.6529 - accuracy: 0.5009
20096/25000 [=======================>......] - ETA: 17s - loss: 7.6544 - accuracy: 0.5008
20128/25000 [=======================>......] - ETA: 17s - loss: 7.6582 - accuracy: 0.5005
20160/25000 [=======================>......] - ETA: 16s - loss: 7.6575 - accuracy: 0.5006
20192/25000 [=======================>......] - ETA: 16s - loss: 7.6560 - accuracy: 0.5007
20224/25000 [=======================>......] - ETA: 16s - loss: 7.6545 - accuracy: 0.5008
20256/25000 [=======================>......] - ETA: 16s - loss: 7.6522 - accuracy: 0.5009
20288/25000 [=======================>......] - ETA: 16s - loss: 7.6515 - accuracy: 0.5010
20320/25000 [=======================>......] - ETA: 16s - loss: 7.6523 - accuracy: 0.5009
20352/25000 [=======================>......] - ETA: 16s - loss: 7.6508 - accuracy: 0.5010
20384/25000 [=======================>......] - ETA: 16s - loss: 7.6523 - accuracy: 0.5009
20416/25000 [=======================>......] - ETA: 16s - loss: 7.6516 - accuracy: 0.5010
20448/25000 [=======================>......] - ETA: 15s - loss: 7.6546 - accuracy: 0.5008
20480/25000 [=======================>......] - ETA: 15s - loss: 7.6546 - accuracy: 0.5008
20512/25000 [=======================>......] - ETA: 15s - loss: 7.6562 - accuracy: 0.5007
20544/25000 [=======================>......] - ETA: 15s - loss: 7.6592 - accuracy: 0.5005
20576/25000 [=======================>......] - ETA: 15s - loss: 7.6592 - accuracy: 0.5005
20608/25000 [=======================>......] - ETA: 15s - loss: 7.6577 - accuracy: 0.5006
20640/25000 [=======================>......] - ETA: 15s - loss: 7.6599 - accuracy: 0.5004
20672/25000 [=======================>......] - ETA: 15s - loss: 7.6577 - accuracy: 0.5006
20704/25000 [=======================>......] - ETA: 15s - loss: 7.6570 - accuracy: 0.5006
20736/25000 [=======================>......] - ETA: 14s - loss: 7.6548 - accuracy: 0.5008
20768/25000 [=======================>......] - ETA: 14s - loss: 7.6519 - accuracy: 0.5010
20800/25000 [=======================>......] - ETA: 14s - loss: 7.6534 - accuracy: 0.5009
20832/25000 [=======================>......] - ETA: 14s - loss: 7.6526 - accuracy: 0.5009
20864/25000 [========================>.....] - ETA: 14s - loss: 7.6534 - accuracy: 0.5009
20896/25000 [========================>.....] - ETA: 14s - loss: 7.6556 - accuracy: 0.5007
20928/25000 [========================>.....] - ETA: 14s - loss: 7.6571 - accuracy: 0.5006
20960/25000 [========================>.....] - ETA: 14s - loss: 7.6549 - accuracy: 0.5008
20992/25000 [========================>.....] - ETA: 14s - loss: 7.6527 - accuracy: 0.5009
21024/25000 [========================>.....] - ETA: 13s - loss: 7.6542 - accuracy: 0.5008
21056/25000 [========================>.....] - ETA: 13s - loss: 7.6557 - accuracy: 0.5007
21088/25000 [========================>.....] - ETA: 13s - loss: 7.6557 - accuracy: 0.5007
21120/25000 [========================>.....] - ETA: 13s - loss: 7.6608 - accuracy: 0.5004
21152/25000 [========================>.....] - ETA: 13s - loss: 7.6594 - accuracy: 0.5005
21184/25000 [========================>.....] - ETA: 13s - loss: 7.6587 - accuracy: 0.5005
21216/25000 [========================>.....] - ETA: 13s - loss: 7.6572 - accuracy: 0.5006
21248/25000 [========================>.....] - ETA: 13s - loss: 7.6551 - accuracy: 0.5008
21280/25000 [========================>.....] - ETA: 13s - loss: 7.6551 - accuracy: 0.5008
21312/25000 [========================>.....] - ETA: 12s - loss: 7.6544 - accuracy: 0.5008
21344/25000 [========================>.....] - ETA: 12s - loss: 7.6551 - accuracy: 0.5007
21376/25000 [========================>.....] - ETA: 12s - loss: 7.6551 - accuracy: 0.5007
21408/25000 [========================>.....] - ETA: 12s - loss: 7.6559 - accuracy: 0.5007
21440/25000 [========================>.....] - ETA: 12s - loss: 7.6566 - accuracy: 0.5007
21472/25000 [========================>.....] - ETA: 12s - loss: 7.6552 - accuracy: 0.5007
21504/25000 [========================>.....] - ETA: 12s - loss: 7.6545 - accuracy: 0.5008
21536/25000 [========================>.....] - ETA: 12s - loss: 7.6524 - accuracy: 0.5009
21568/25000 [========================>.....] - ETA: 12s - loss: 7.6531 - accuracy: 0.5009
21600/25000 [========================>.....] - ETA: 11s - loss: 7.6524 - accuracy: 0.5009
21632/25000 [========================>.....] - ETA: 11s - loss: 7.6532 - accuracy: 0.5009
21664/25000 [========================>.....] - ETA: 11s - loss: 7.6560 - accuracy: 0.5007
21696/25000 [=========================>....] - ETA: 11s - loss: 7.6567 - accuracy: 0.5006
21728/25000 [=========================>....] - ETA: 11s - loss: 7.6518 - accuracy: 0.5010
21760/25000 [=========================>....] - ETA: 11s - loss: 7.6525 - accuracy: 0.5009
21792/25000 [=========================>....] - ETA: 11s - loss: 7.6518 - accuracy: 0.5010
21824/25000 [=========================>....] - ETA: 11s - loss: 7.6547 - accuracy: 0.5008
21856/25000 [=========================>....] - ETA: 11s - loss: 7.6568 - accuracy: 0.5006
21888/25000 [=========================>....] - ETA: 10s - loss: 7.6596 - accuracy: 0.5005
21920/25000 [=========================>....] - ETA: 10s - loss: 7.6631 - accuracy: 0.5002
21952/25000 [=========================>....] - ETA: 10s - loss: 7.6624 - accuracy: 0.5003
21984/25000 [=========================>....] - ETA: 10s - loss: 7.6610 - accuracy: 0.5004
22016/25000 [=========================>....] - ETA: 10s - loss: 7.6610 - accuracy: 0.5004
22048/25000 [=========================>....] - ETA: 10s - loss: 7.6631 - accuracy: 0.5002
22080/25000 [=========================>....] - ETA: 10s - loss: 7.6645 - accuracy: 0.5001
22112/25000 [=========================>....] - ETA: 10s - loss: 7.6652 - accuracy: 0.5001
22144/25000 [=========================>....] - ETA: 10s - loss: 7.6680 - accuracy: 0.4999
22176/25000 [=========================>....] - ETA: 9s - loss: 7.6701 - accuracy: 0.4998 
22208/25000 [=========================>....] - ETA: 9s - loss: 7.6694 - accuracy: 0.4998
22240/25000 [=========================>....] - ETA: 9s - loss: 7.6708 - accuracy: 0.4997
22272/25000 [=========================>....] - ETA: 9s - loss: 7.6701 - accuracy: 0.4998
22304/25000 [=========================>....] - ETA: 9s - loss: 7.6721 - accuracy: 0.4996
22336/25000 [=========================>....] - ETA: 9s - loss: 7.6755 - accuracy: 0.4994
22368/25000 [=========================>....] - ETA: 9s - loss: 7.6755 - accuracy: 0.4994
22400/25000 [=========================>....] - ETA: 9s - loss: 7.6748 - accuracy: 0.4995
22432/25000 [=========================>....] - ETA: 8s - loss: 7.6735 - accuracy: 0.4996
22464/25000 [=========================>....] - ETA: 8s - loss: 7.6762 - accuracy: 0.4994
22496/25000 [=========================>....] - ETA: 8s - loss: 7.6775 - accuracy: 0.4993
22528/25000 [==========================>...] - ETA: 8s - loss: 7.6809 - accuracy: 0.4991
22560/25000 [==========================>...] - ETA: 8s - loss: 7.6836 - accuracy: 0.4989
22592/25000 [==========================>...] - ETA: 8s - loss: 7.6829 - accuracy: 0.4989
22624/25000 [==========================>...] - ETA: 8s - loss: 7.6829 - accuracy: 0.4989
22656/25000 [==========================>...] - ETA: 8s - loss: 7.6815 - accuracy: 0.4990
22688/25000 [==========================>...] - ETA: 8s - loss: 7.6815 - accuracy: 0.4990
22720/25000 [==========================>...] - ETA: 7s - loss: 7.6821 - accuracy: 0.4990
22752/25000 [==========================>...] - ETA: 7s - loss: 7.6835 - accuracy: 0.4989
22784/25000 [==========================>...] - ETA: 7s - loss: 7.6848 - accuracy: 0.4988
22816/25000 [==========================>...] - ETA: 7s - loss: 7.6854 - accuracy: 0.4988
22848/25000 [==========================>...] - ETA: 7s - loss: 7.6894 - accuracy: 0.4985
22880/25000 [==========================>...] - ETA: 7s - loss: 7.6874 - accuracy: 0.4986
22912/25000 [==========================>...] - ETA: 7s - loss: 7.6880 - accuracy: 0.4986
22944/25000 [==========================>...] - ETA: 7s - loss: 7.6893 - accuracy: 0.4985
22976/25000 [==========================>...] - ETA: 7s - loss: 7.6873 - accuracy: 0.4987
23008/25000 [==========================>...] - ETA: 6s - loss: 7.6879 - accuracy: 0.4986
23040/25000 [==========================>...] - ETA: 6s - loss: 7.6866 - accuracy: 0.4987
23072/25000 [==========================>...] - ETA: 6s - loss: 7.6852 - accuracy: 0.4988
23104/25000 [==========================>...] - ETA: 6s - loss: 7.6825 - accuracy: 0.4990
23136/25000 [==========================>...] - ETA: 6s - loss: 7.6805 - accuracy: 0.4991
23168/25000 [==========================>...] - ETA: 6s - loss: 7.6812 - accuracy: 0.4991
23200/25000 [==========================>...] - ETA: 6s - loss: 7.6838 - accuracy: 0.4989
23232/25000 [==========================>...] - ETA: 6s - loss: 7.6864 - accuracy: 0.4987
23264/25000 [==========================>...] - ETA: 6s - loss: 7.6857 - accuracy: 0.4988
23296/25000 [==========================>...] - ETA: 5s - loss: 7.6870 - accuracy: 0.4987
23328/25000 [==========================>...] - ETA: 5s - loss: 7.6863 - accuracy: 0.4987
23360/25000 [===========================>..] - ETA: 5s - loss: 7.6876 - accuracy: 0.4986
23392/25000 [===========================>..] - ETA: 5s - loss: 7.6909 - accuracy: 0.4984
23424/25000 [===========================>..] - ETA: 5s - loss: 7.6908 - accuracy: 0.4984
23456/25000 [===========================>..] - ETA: 5s - loss: 7.6888 - accuracy: 0.4986
23488/25000 [===========================>..] - ETA: 5s - loss: 7.6862 - accuracy: 0.4987
23520/25000 [===========================>..] - ETA: 5s - loss: 7.6836 - accuracy: 0.4989
23552/25000 [===========================>..] - ETA: 5s - loss: 7.6829 - accuracy: 0.4989
23584/25000 [===========================>..] - ETA: 4s - loss: 7.6764 - accuracy: 0.4994
23616/25000 [===========================>..] - ETA: 4s - loss: 7.6757 - accuracy: 0.4994
23648/25000 [===========================>..] - ETA: 4s - loss: 7.6725 - accuracy: 0.4996
23680/25000 [===========================>..] - ETA: 4s - loss: 7.6744 - accuracy: 0.4995
23712/25000 [===========================>..] - ETA: 4s - loss: 7.6744 - accuracy: 0.4995
23744/25000 [===========================>..] - ETA: 4s - loss: 7.6763 - accuracy: 0.4994
23776/25000 [===========================>..] - ETA: 4s - loss: 7.6795 - accuracy: 0.4992
23808/25000 [===========================>..] - ETA: 4s - loss: 7.6789 - accuracy: 0.4992
23840/25000 [===========================>..] - ETA: 4s - loss: 7.6776 - accuracy: 0.4993
23872/25000 [===========================>..] - ETA: 3s - loss: 7.6763 - accuracy: 0.4994
23904/25000 [===========================>..] - ETA: 3s - loss: 7.6769 - accuracy: 0.4993
23936/25000 [===========================>..] - ETA: 3s - loss: 7.6749 - accuracy: 0.4995
23968/25000 [===========================>..] - ETA: 3s - loss: 7.6737 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 3s - loss: 7.6743 - accuracy: 0.4995
24032/25000 [===========================>..] - ETA: 3s - loss: 7.6743 - accuracy: 0.4995
24064/25000 [===========================>..] - ETA: 3s - loss: 7.6762 - accuracy: 0.4994
24096/25000 [===========================>..] - ETA: 3s - loss: 7.6762 - accuracy: 0.4994
24128/25000 [===========================>..] - ETA: 3s - loss: 7.6736 - accuracy: 0.4995
24160/25000 [===========================>..] - ETA: 2s - loss: 7.6730 - accuracy: 0.4996
24192/25000 [============================>.] - ETA: 2s - loss: 7.6723 - accuracy: 0.4996
24224/25000 [============================>.] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24256/25000 [============================>.] - ETA: 2s - loss: 7.6755 - accuracy: 0.4994
24288/25000 [============================>.] - ETA: 2s - loss: 7.6729 - accuracy: 0.4996
24320/25000 [============================>.] - ETA: 2s - loss: 7.6736 - accuracy: 0.4995
24352/25000 [============================>.] - ETA: 2s - loss: 7.6735 - accuracy: 0.4995
24384/25000 [============================>.] - ETA: 2s - loss: 7.6754 - accuracy: 0.4994
24416/25000 [============================>.] - ETA: 2s - loss: 7.6735 - accuracy: 0.4995
24448/25000 [============================>.] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
24480/25000 [============================>.] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
24512/25000 [============================>.] - ETA: 1s - loss: 7.6716 - accuracy: 0.4997
24544/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24576/25000 [============================>.] - ETA: 1s - loss: 7.6679 - accuracy: 0.4999
24608/25000 [============================>.] - ETA: 1s - loss: 7.6641 - accuracy: 0.5002
24640/25000 [============================>.] - ETA: 1s - loss: 7.6654 - accuracy: 0.5001
24672/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24704/25000 [============================>.] - ETA: 1s - loss: 7.6660 - accuracy: 0.5000
24736/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24768/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24800/25000 [============================>.] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
24832/25000 [============================>.] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
24864/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24896/25000 [============================>.] - ETA: 0s - loss: 7.6648 - accuracy: 0.5001
24928/25000 [============================>.] - ETA: 0s - loss: 7.6642 - accuracy: 0.5002
24960/25000 [============================>.] - ETA: 0s - loss: 7.6654 - accuracy: 0.5001
24992/25000 [============================>.] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 106s 4ms/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

